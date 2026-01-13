#!/usr/bin/env python3
"""
02_extract_features.py - Feature Extraction for Extended uTP Variant Classifier

This script extracts two types of features from mature domains:
1. ProtT5 embeddings (1024-dimensional protein language model representations)
2. Biochemical properties (length, MW, pI, GRAVY, instability, secondary structure)

Features are saved in formats ready for classifier training.

Output:
- features/embeddings.h5: ProtT5 embeddings (HDF5 format)
- features/biochemical_features.csv: Calculated properties
- figures/property_distributions.svg: Property distributions by class

Usage:
    uv run python experiments/utp_variant_classifier_extended/02_extract_features.py
"""

import gc
import re
import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy import stats
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

# ProtT5 model
PROTT5_MODEL = "Rostlab/prot_t5_xl_uniref50"
EMBEDDING_DIM = 1024
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 2000

# Random seed
RANDOM_SEED = 42

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
DATA_DIR = OUTPUT_DIR / "data"
FEATURES_DIR = OUTPUT_DIR / "features"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Input files
MATURE_DOMAINS_FILE = DATA_DIR / "mature_domains.fasta"
PROCESSED_PROTEINS_FILE = DATA_DIR / "processed_proteins.csv"

# Output files
EMBEDDINGS_FILE = FEATURES_DIR / "embeddings.h5"
BIOCHEM_FEATURES_FILE = FEATURES_DIR / "biochemical_features.csv"


# =============================================================================
# Biochemical Properties
# =============================================================================


def compute_protein_properties(seq: str) -> dict | None:
    """
    Calculate comprehensive biophysical properties of a protein sequence.

    Returns dict with:
    - Basic: length, molecular_weight
    - Charge: isoelectric_point, net_charge_at_ph7
    - Hydrophobicity: gravy, fraction_hydrophobic
    - Stability: instability_index
    - Structure: fraction_helix, fraction_sheet, fraction_coil
    - Composition: fraction_charged, fraction_polar, fraction_aromatic
    """
    # Clean sequence
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    seq_clean = "".join(c for c in seq.upper() if c in valid_aa)

    if len(seq_clean) < 10:
        return None

    try:
        pa = ProteinAnalysis(seq_clean)

        # Basic properties
        length = len(seq_clean)
        mw = pa.molecular_weight()

        # Charge properties
        pI = pa.isoelectric_point()
        # Estimate net charge at pH 7
        negative = seq_clean.count("D") + seq_clean.count("E")
        positive = (
            seq_clean.count("K") + seq_clean.count("R") + 0.5 * seq_clean.count("H")
        )
        net_charge_ph7 = positive - negative

        # Hydrophobicity
        gravy = pa.gravy()
        hydrophobic = set("AILMFVW")
        fraction_hydrophobic = sum(1 for aa in seq_clean if aa in hydrophobic) / length

        # Stability
        instability = pa.instability_index()

        # Secondary structure tendency (Chou-Fasman)
        helix, sheet, coil = pa.secondary_structure_fraction()

        # Amino acid composition groups
        charged = set("DEKRH")
        polar = set("STNQY")
        aromatic = set("FWY")

        fraction_charged = sum(1 for aa in seq_clean if aa in charged) / length
        fraction_polar = sum(1 for aa in seq_clean if aa in polar) / length
        fraction_aromatic = sum(1 for aa in seq_clean if aa in aromatic) / length

        # Disorder-promoting residues (simplified)
        disorder_promoting = set("PQSEKR")
        fraction_disorder = (
            sum(1 for aa in seq_clean if aa in disorder_promoting) / length
        )

        return {
            "length": length,
            "molecular_weight": mw,
            "isoelectric_point": pI,
            "net_charge_ph7": net_charge_ph7,
            "gravy": gravy,
            "fraction_hydrophobic": fraction_hydrophobic,
            "instability_index": instability,
            "fraction_helix": helix,
            "fraction_sheet": sheet,
            "fraction_coil": coil,
            "fraction_charged": fraction_charged,
            "fraction_polar": fraction_polar,
            "fraction_aromatic": fraction_aromatic,
            "fraction_disorder_promoting": fraction_disorder,
        }

    except Exception as e:
        print(f"Warning: Failed to compute properties: {e}")
        return None


def compute_all_biochemical_features(
    sequences: dict[str, str],
    protein_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute biochemical properties for all sequences."""
    rows = []

    for name, seq in tqdm(sequences.items(), desc="Computing biochemical properties"):
        props = compute_protein_properties(seq)
        if props is None:
            continue

        # Add protein name
        props["name"] = name

        # Add terminal class info
        protein_row = protein_df[protein_df["name"] == name]
        if len(protein_row) > 0:
            props["terminal_class"] = protein_row.iloc[0]["terminal_class"]
        else:
            props["terminal_class"] = "unknown"

        rows.append(props)

    df = pd.DataFrame(rows)

    # Reorder columns
    cols = ["name", "terminal_class"] + [
        c for c in df.columns if c not in ["name", "terminal_class"]
    ]
    df = df[cols]

    return df


# =============================================================================
# ProtT5 Embeddings
# =============================================================================


def load_prott5_model(device: str = "cpu"):
    """Load ProtT5 tokenizer and model."""
    from transformers import T5EncoderModel, T5Tokenizer

    print(f"Loading ProtT5 model ({PROTT5_MODEL})...")
    print("  This may take a while on first run (downloading ~3GB model)")

    tokenizer = T5Tokenizer.from_pretrained(PROTT5_MODEL, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(PROTT5_MODEL)
    model = model.to(device)
    model.eval()

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"  Model loaded on {device}")
    return tokenizer, model


def embed_sequence(
    seq: str,
    tokenizer,
    model,
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate ProtT5 embedding for a single sequence.

    Returns:
        Mean-pooled embedding of shape (1024,)
    """
    # Truncate if necessary
    if len(seq) > MAX_SEQ_LENGTH:
        seq = seq[:MAX_SEQ_LENGTH]

    # Clean sequence (replace non-standard AAs)
    seq = re.sub(r"[UZOJB*]", "X", seq)

    # Add spaces between amino acids (required by ProtT5)
    seq_spaced = " ".join(seq)

    # Tokenize
    ids = tokenizer.batch_encode_plus(
        [seq_spaced],
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    )

    input_ids = ids["input_ids"].to(device)
    attention_mask = ids["attention_mask"].to(device)

    # Get embeddings
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    embedding = output.last_hidden_state.cpu().numpy()

    # Get sequence length (excluding padding and special tokens)
    seq_len = (attention_mask[0] == 1).sum().item() - 1  # -1 for </s> token

    # Extract embeddings for actual sequence positions
    seq_embedding = embedding[0, :seq_len, :]  # Shape: (seq_len, 1024)

    # Mean pooling
    mean_embedding = np.mean(seq_embedding, axis=0)

    return mean_embedding.astype(np.float32)


def compute_all_embeddings(
    sequences: dict[str, str],
    output_file: Path,
    device: str = "cpu",
    force_recompute: bool = False,
) -> dict[str, np.ndarray]:
    """
    Compute ProtT5 embeddings for all sequences.

    Embeddings are cached in HDF5 format.
    """
    # Check cache
    if output_file.exists() and not force_recompute:
        print(f"Loading cached embeddings from {output_file}")
        embeddings = {}
        with h5py.File(output_file, "r") as f:
            cached_names = set(f.keys())
            missing = set(sequences.keys()) - cached_names

            if not missing:
                for name in sequences.keys():
                    if name in f:
                        embeddings[name] = f[name][()]
                print(f"  Loaded {len(embeddings)} cached embeddings")
                return embeddings
            else:
                print(f"  Cache missing {len(missing)} sequences, recomputing all")

    # Load model
    tokenizer, model = load_prott5_model(device)

    # Compute embeddings
    embeddings = {}
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_file, "w") as f:
        for name, seq in tqdm(sequences.items(), desc="Computing ProtT5 embeddings"):
            try:
                emb = embed_sequence(seq, tokenizer, model, device)
                embeddings[name] = emb

                # Save to HDF5 immediately (in case of interruption)
                f.create_dataset(name, data=emb)

            except Exception as e:
                print(f"\nWarning: Failed to embed {name}: {e}")
                continue

    print(f"Saved {len(embeddings)} embeddings to {output_file}")

    # Clean up
    del tokenizer, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return embeddings


# =============================================================================
# Visualization
# =============================================================================


def plot_property_distributions(
    biochem_df: pd.DataFrame,
    output_file: Path,
):
    """Plot distributions of biochemical properties by terminal class."""
    # Select properties to plot
    properties = [
        "length",
        "molecular_weight",
        "isoelectric_point",
        "gravy",
        "instability_index",
        "fraction_helix",
        "fraction_sheet",
        "fraction_coil",
    ]
    properties = [p for p in properties if p in biochem_df.columns]

    # Create figure
    n_props = len(properties)
    n_cols = 4
    n_rows = (n_props + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    # Get unique classes
    classes = biochem_df["terminal_class"].unique()

    for i, prop in enumerate(properties):
        ax = axes[i]

        # Violin plot
        sns.violinplot(
            data=biochem_df,
            x="terminal_class",
            y=prop,
            ax=ax,
            palette="Set2",
            inner="box",
            order=sorted(classes),
        )

        ax.set_xlabel("Terminal Class")
        ax.set_ylabel(prop.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=45)

        # Add Kruskal-Wallis test p-value
        groups = [
            biochem_df[biochem_df["terminal_class"] == c][prop].dropna().values
            for c in classes
        ]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) > 1:
            try:
                stat, pval = stats.kruskal(*groups)
                significance = (
                    "***"
                    if pval < 0.001
                    else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                )
                ax.set_title(f"{prop}\n(KW p={pval:.2e} {significance})", fontsize=10)
            except Exception:
                ax.set_title(prop.replace("_", " ").title())
        else:
            ax.set_title(prop.replace("_", " ").title())

    # Hide empty subplots
    for i in range(len(properties), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved property distributions plot to {output_file}")


def plot_embedding_umap(
    embeddings: dict[str, np.ndarray],
    protein_df: pd.DataFrame,
    output_file: Path,
):
    """Generate UMAP visualization of embeddings colored by terminal class."""
    try:
        import umap
    except ImportError:
        print("  UMAP not available, skipping embedding visualization")
        return

    # Prepare data
    names = list(embeddings.keys())
    X = np.array([embeddings[n] for n in names])

    # Get terminal classes
    name_to_class = dict(zip(protein_df["name"], protein_df["terminal_class"]))
    classes = [name_to_class.get(n, "unknown") for n in names]

    # Fit UMAP
    print("  Computing UMAP projection...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=RANDOM_SEED)
    embedding_2d = reducer.fit_transform(X)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_classes = sorted(set(classes))
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_classes)))
    color_map = dict(zip(unique_classes, colors))

    for cls in unique_classes:
        mask = np.array([c == cls for c in classes])
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[color_map[cls]],
            label=cls.replace("terminal_", "T"),
            alpha=0.7,
            s=30,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP of ProtT5 Embeddings by Terminal Class")
    ax.legend(title="Terminal Class", loc="best")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved UMAP visualization to {output_file}")


def plot_feature_correlations(
    biochem_df: pd.DataFrame,
    output_file: Path,
):
    """Plot correlation matrix of biochemical features."""
    # Select numeric columns (excluding name and class)
    numeric_cols = biochem_df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return

    # Compute correlation matrix
    corr_matrix = biochem_df[numeric_cols].corr()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        square=True,
        linewidths=0.5,
    )

    ax.set_title("Biochemical Feature Correlations")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved feature correlations plot to {output_file}")


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("02_extract_features.py - Feature Extraction for Extended Classifier")
    print("=" * 70)

    # Create output directories
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Check input files
    if not MATURE_DOMAINS_FILE.exists():
        raise FileNotFoundError(
            f"Mature domains file not found: {MATURE_DOMAINS_FILE}\n"
            f"Please run 01_prepare_data.py first."
        )
    if not PROCESSED_PROTEINS_FILE.exists():
        raise FileNotFoundError(
            f"Processed proteins file not found: {PROCESSED_PROTEINS_FILE}\n"
            f"Please run 01_prepare_data.py first."
        )

    # =========================================================================
    # Step 1: Load data
    # =========================================================================
    print("\n[1/4] Loading data...")

    # Load sequences
    sequences = {
        record.id: str(record.seq)
        for record in SeqIO.parse(MATURE_DOMAINS_FILE, "fasta")
    }
    print(f"  Loaded {len(sequences)} mature domain sequences")

    # Load protein metadata
    protein_df = pd.read_csv(PROCESSED_PROTEINS_FILE)
    print(f"  Loaded metadata for {len(protein_df)} proteins")

    # Class distribution
    class_counts = protein_df["terminal_class"].value_counts()
    print(f"\n  Class distribution:")
    for cls, count in class_counts.items():
        print(f"    {cls}: {count}")

    # =========================================================================
    # Step 2: Compute biochemical properties
    # =========================================================================
    print("\n[2/4] Computing biochemical properties...")

    biochem_df = compute_all_biochemical_features(sequences, protein_df)
    biochem_df.to_csv(BIOCHEM_FEATURES_FILE, index=False)
    print(f"  Saved biochemical features for {len(biochem_df)} proteins")

    # Summary statistics
    print("\n  Biochemical feature summary:")
    numeric_cols = biochem_df.select_dtypes(include=[np.number]).columns
    for col in list(numeric_cols)[:5]:
        print(
            f"    {col}: mean={biochem_df[col].mean():.2f}, std={biochem_df[col].std():.2f}"
        )

    # =========================================================================
    # Step 3: Compute ProtT5 embeddings
    # =========================================================================
    print("\n[3/4] Computing ProtT5 embeddings...")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    embeddings = compute_all_embeddings(
        sequences,
        EMBEDDINGS_FILE,
        device=device,
        force_recompute=False,
    )
    print(f"  Computed embeddings for {len(embeddings)} proteins")

    # =========================================================================
    # Step 4: Generate visualizations
    # =========================================================================
    print("\n[4/4] Generating visualizations...")

    plot_property_distributions(biochem_df, FIGURES_DIR / "property_distributions.svg")
    plot_feature_correlations(biochem_df, FIGURES_DIR / "feature_correlations.svg")
    plot_embedding_umap(embeddings, protein_df, FIGURES_DIR / "umap_embeddings.svg")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Feature extraction complete!")
    print("=" * 70)

    print(f"\n📁 Outputs:")
    print(f"  - {EMBEDDINGS_FILE}")
    print(f"  - {BIOCHEM_FEATURES_FILE}")
    print(f"  - {FIGURES_DIR / 'property_distributions.svg'}")
    print(f"  - {FIGURES_DIR / 'feature_correlations.svg'}")
    print(f"  - {FIGURES_DIR / 'umap_embeddings.svg'}")

    print(f"\n📊 Feature dimensions:")
    print(f"  - ProtT5 embeddings: {EMBEDDING_DIM} dimensions")
    print(f"  - Biochemical properties: {len(numeric_cols)} features")

    print(f"\nNext step: Run 03_train_classifier.py to train classifiers")


if __name__ == "__main__":
    main()
