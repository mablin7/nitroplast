#!/usr/bin/env python3
"""
02_extract_features.py - Feature Extraction for uTP Variant Classifier

This script extracts two types of features from mature domains:
1. ProtT5 embeddings (1024-dimensional protein language model representations)
2. Biochemical properties (length, MW, pI, GRAVY, instability, secondary structure)

Features are saved in formats ready for classifier training.

Output:
- features/embeddings.h5: ProtT5 embeddings (HDF5 format)
- features/biochemical_features.csv: Calculated properties
- figures/property_distributions_by_variant.svg: Property distributions

Usage:
    uv run python experiments/utp_variant_classifier/02_extract_features.py
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
EMBEDDING_DIM = 1024  # ProtT5 embedding dimension
BATCH_SIZE = 1  # Process one at a time for memory efficiency
MAX_SEQ_LENGTH = 2000  # Truncate longer sequences

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
VARIANT_MAPPING_FILE = DATA_DIR / "variant_mapping.csv"

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
        # Rough approximation: count D, E (negative) and K, R, H (positive at pH 7)
        negative = seq_clean.count('D') + seq_clean.count('E')
        positive = seq_clean.count('K') + seq_clean.count('R') + 0.5 * seq_clean.count('H')
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
        fraction_disorder = sum(1 for aa in seq_clean if aa in disorder_promoting) / length
        
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
    variant_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Compute biochemical properties for all sequences."""
    rows = []
    
    for name, seq in tqdm(sequences.items(), desc="Computing biochemical properties"):
        props = compute_protein_properties(seq)
        if props is None:
            continue
        
        # Add protein name
        props["name"] = name
        
        # Add variant info
        variant_row = variant_mapping[variant_mapping["name"] == name]
        if len(variant_row) > 0:
            props["motif_variant"] = variant_row.iloc[0]["motif_variant"]
        else:
            props["motif_variant"] = "unknown"
        
        rows.append(props)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns
    cols = ["name", "motif_variant"] + [c for c in df.columns if c not in ["name", "motif_variant"]]
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
        return_tensors="pt"
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
    
    Embeddings are cached in HDF5 format. If output_file exists and
    contains all sequences, cached embeddings are returned.
    
    Returns:
        Dict mapping sequence name to embedding array (1024,)
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


def plot_property_distributions_by_variant(
    biochem_df: pd.DataFrame,
    output_file: Path,
    top_n_variants: int = 6,
):
    """Plot distributions of biochemical properties by variant."""
    # Select top variants
    variant_counts = biochem_df["motif_variant"].value_counts()
    top_variants = variant_counts.head(top_n_variants).index.tolist()
    df_plot = biochem_df[biochem_df["motif_variant"].isin(top_variants)].copy()
    
    # Select properties to plot
    properties = [
        "length", "molecular_weight", "isoelectric_point", "gravy",
        "instability_index", "fraction_helix", "fraction_coil", "fraction_disorder_promoting"
    ]
    properties = [p for p in properties if p in df_plot.columns]
    
    # Create figure
    n_props = len(properties)
    n_cols = 4
    n_rows = (n_props + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    for i, prop in enumerate(properties):
        ax = axes[i]
        
        # Violin plot
        sns.violinplot(
            data=df_plot,
            x="motif_variant",
            y=prop,
            ax=ax,
            palette="tab10",
            inner="box",
        )
        
        ax.set_xlabel("uTP Variant")
        ax.set_ylabel(prop.replace("_", " ").title())
        ax.tick_params(axis='x', rotation=45)
        
        # Add Kruskal-Wallis test p-value
        groups = [df_plot[df_plot["motif_variant"] == v][prop].dropna() for v in top_variants]
        groups = [g.values for g in groups if len(g) > 0]
        if len(groups) > 1:
            try:
                stat, pval = stats.kruskal(*groups)
                significance = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                ax.set_title(f"{prop}\n(KW p={pval:.2e} {significance})", fontsize=10)
            except Exception:
                ax.set_title(prop.replace("_", " ").title())
        else:
            ax.set_title(prop.replace("_", " ").title())
    
    # Hide empty subplots
    for i in range(len(properties), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved property distributions plot to {output_file}")


def plot_embedding_stats(
    embeddings: dict[str, np.ndarray],
    variant_mapping: pd.DataFrame,
    output_file: Path,
):
    """Plot statistics of embedding distributions."""
    # Create dataframe
    data = []
    for name, emb in embeddings.items():
        variant_row = variant_mapping[variant_mapping["name"] == name]
        variant = variant_row.iloc[0]["motif_variant"] if len(variant_row) > 0 else "unknown"
        
        data.append({
            "name": name,
            "variant": variant,
            "mean": np.mean(emb),
            "std": np.std(emb),
            "min": np.min(emb),
            "max": np.max(emb),
            "norm": np.linalg.norm(emb),
        })
    
    df = pd.DataFrame(data)
    
    # Filter to top variants
    variant_counts = df["variant"].value_counts()
    top_variants = variant_counts.head(6).index.tolist()
    df_plot = df[df["variant"].isin(top_variants)]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Mean
    sns.boxplot(data=df_plot, x="variant", y="mean", ax=axes[0])
    axes[0].set_xlabel("uTP Variant")
    axes[0].set_ylabel("Mean Embedding Value")
    axes[0].set_title("Embedding Mean by Variant")
    axes[0].tick_params(axis='x', rotation=45)
    
    # Std
    sns.boxplot(data=df_plot, x="variant", y="std", ax=axes[1])
    axes[1].set_xlabel("uTP Variant")
    axes[1].set_ylabel("Std Embedding Value")
    axes[1].set_title("Embedding Std by Variant")
    axes[1].tick_params(axis='x', rotation=45)
    
    # Norm
    sns.boxplot(data=df_plot, x="variant", y="norm", ax=axes[2])
    axes[2].set_xlabel("uTP Variant")
    axes[2].set_ylabel("Embedding L2 Norm")
    axes[2].set_title("Embedding Norm by Variant")
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved embedding stats plot to {output_file}")


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("02_extract_features.py - Feature Extraction for uTP Variant Classifier")
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
    if not VARIANT_MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Variant mapping file not found: {VARIANT_MAPPING_FILE}\n"
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
    
    # Load variant mapping
    variant_mapping = pd.read_csv(VARIANT_MAPPING_FILE)
    print(f"  Loaded variant mapping for {len(variant_mapping)} proteins")
    
    # =========================================================================
    # Step 2: Compute biochemical properties
    # =========================================================================
    print("\n[2/4] Computing biochemical properties...")
    
    biochem_df = compute_all_biochemical_features(sequences, variant_mapping)
    biochem_df.to_csv(BIOCHEM_FEATURES_FILE, index=False)
    print(f"  Saved biochemical features for {len(biochem_df)} proteins to {BIOCHEM_FEATURES_FILE}")
    
    # Summary statistics
    print("\n  Biochemical feature summary:")
    numeric_cols = biochem_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:5]:  # Show first 5
        print(f"    {col}: mean={biochem_df[col].mean():.2f}, std={biochem_df[col].std():.2f}")
    
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
    
    plot_property_distributions_by_variant(
        biochem_df,
        FIGURES_DIR / "property_distributions_by_variant.svg"
    )
    
    plot_embedding_stats(
        embeddings,
        variant_mapping,
        FIGURES_DIR / "embedding_stats_by_variant.svg"
    )
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Feature extraction complete!")
    print("=" * 70)
    
    print(f"\nOutputs:")
    print(f"  - {EMBEDDINGS_FILE} ({EMBEDDINGS_FILE.stat().st_size / 1e6:.1f} MB)")
    print(f"  - {BIOCHEM_FEATURES_FILE}")
    print(f"  - {FIGURES_DIR / 'property_distributions_by_variant.svg'}")
    print(f"  - {FIGURES_DIR / 'embedding_stats_by_variant.svg'}")
    
    print(f"\nFeature dimensions:")
    print(f"  - ProtT5 embeddings: {EMBEDDING_DIM} dimensions")
    print(f"  - Biochemical properties: {len(numeric_cols)} features")
    
    print(f"\nNext step: Run 03_train_classifier.py to train classifiers")


if __name__ == "__main__":
    main()
