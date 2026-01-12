#!/usr/bin/env python3
"""
05_full_proteome_analysis.py - Full Proteome Validation and Interpretability

Runs inference and interpretability analysis on the ENTIRE B. bigelowii proteome:
- All ~933 uTP-containing proteins (Import_candidates.fasta)
- All ~144,000 non-uTP proteins from the proteome

This is a large-scale analysis requiring:
- Batched embedding computation (memory-efficient)
- Subsampling for visualization (UMAP/t-SNE can't handle 144k points)
- Incremental statistics computation

Usage:
    uv run python experiments/utp_presence_classifier/05_full_proteome_analysis.py

Expected runtime: Several hours for embedding computation (can be resumed if interrupted)
"""

import gc
import re
import warnings
from pathlib import Path

import h5py
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from matplotlib.patches import Ellipse
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

BATCH_SIZE = 32  # For embedding computation
VISUALIZATION_SUBSAMPLE = 2000  # Max points per class for UMAP/t-SNE
RANDOM_SEED = 42
TSNE_PERPLEXITY = 30
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

PROTT5_MODEL = "Rostlab/prot_t5_xl_uniref50"

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
DATA_DIR = SCRIPT_DIR.parent.parent / "data"

# Input files
UTP_PROTEINS_FILE = DATA_DIR / "Import_candidates.fasta"
FULL_PROTEOME_FILE = DATA_DIR / "ADK1075_proteomics_DB_2.fasta"

# Trained classifier
CLASSIFIER_FILE = OUTPUT_DIR / "exp2_best_classifier.joblib"

# Output files (full proteome analysis)
FULL_EMBEDDINGS_FILE = OUTPUT_DIR / "full_proteome_embeddings.h5"
FULL_RESULTS_DIR = OUTPUT_DIR / "full_proteome_analysis"

# =============================================================================
# Styling
# =============================================================================

sns.set_theme("paper", style="whitegrid")
matplotlib.rcParams.update(
    {
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "font.size": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "figure.dpi": 150,
    }
)

COLORS = {"uTP": "#E69F00", "Non-uTP": "#56B4E9"}


# =============================================================================
# Protein Properties
# =============================================================================


def get_protein_properties(seq_str: str) -> dict | None:
    """Calculate biophysical properties of a protein sequence."""
    seq_clean = "".join(c for c in seq_str if c in "ACDEFGHIKLMNPQRSTVWY")
    if len(seq_clean) < 10:
        return None

    pa = ProteinAnalysis(seq_clean)

    try:
        return {
            "length": len(seq_clean),
            "molecular_weight": pa.molecular_weight(),
            "isoelectric_point": pa.isoelectric_point(),
            "gravy": pa.gravy(),
            "instability_index": pa.instability_index(),
            "fraction_helix": pa.secondary_structure_fraction()[0],
            "fraction_sheet": pa.secondary_structure_fraction()[1],
            "fraction_coil": pa.secondary_structure_fraction()[2],
        }
    except Exception:
        return None


def compute_properties_batched(
    sequences: dict[str, str], desc: str = "Computing properties"
) -> pd.DataFrame:
    """Compute properties for all sequences with progress bar."""
    rows = []
    for name, seq in tqdm(sequences.items(), desc=desc):
        props = get_protein_properties(seq)
        if props:
            props["name"] = name
            rows.append(props)
    return pd.DataFrame(rows)


# =============================================================================
# Embedding Computation (Batched, Resumable)
# =============================================================================


def load_prott5_model(device: str = "cpu"):
    """Load ProtT5 tokenizer and model."""
    from transformers import T5EncoderModel, T5Tokenizer

    print("  Loading ProtT5 model...")
    tokenizer = T5Tokenizer.from_pretrained(PROTT5_MODEL, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(PROTT5_MODEL)
    model = model.to(device)
    model.eval()

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return tokenizer, model


def embed_single_sequence(seq: str, tokenizer, model, device: str) -> np.ndarray:
    """Embed a single sequence."""
    seq = re.sub(r"[UZOJB*]", "X", seq)
    seq_spaced = " ".join(seq)

    ids = tokenizer.batch_encode_plus(
        [seq_spaced], add_special_tokens=True, padding=True
    )
    input_ids = torch.tensor(ids["input_ids"]).to(device)
    attention_mask = torch.tensor(ids["attention_mask"]).to(device)

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    embedding = output.last_hidden_state.cpu().numpy()
    seq_len = (attention_mask[0] == 1).sum().item()
    seq_embedding = embedding[0][: seq_len - 1]

    return np.mean(seq_embedding, axis=0).astype(np.float32)


def compute_embeddings_batched(
    sequences: dict[str, str],
    embeddings_file: Path,
    device: str = "cpu",
    batch_size: int = BATCH_SIZE,
) -> dict[str, np.ndarray]:
    """
    Compute embeddings with batching and resumability.
    
    Saves incrementally to HDF5 file, allowing resume if interrupted.
    """
    # Check which sequences already have embeddings
    existing_keys = set()
    if embeddings_file.exists():
        with h5py.File(embeddings_file, "r") as f:
            existing_keys = set(f.keys())
        print(f"  Found {len(existing_keys)} existing embeddings")

    # Find sequences that need embedding
    to_compute = {k: v for k, v in sequences.items() if k not in existing_keys}
    
    if not to_compute:
        print("  All embeddings already computed, loading from cache...")
        embeddings = {}
        with h5py.File(embeddings_file, "r") as f:
            for key in tqdm(f.keys(), desc="Loading embeddings"):
                if key in sequences:
                    embeddings[key] = f[key][()]
        return embeddings

    print(f"  Need to compute {len(to_compute)} new embeddings")

    # Load model
    tokenizer, model = load_prott5_model(device)

    # Open file in append mode
    seq_names = list(to_compute.keys())
    seq_list = [to_compute[n] for n in seq_names]

    # Process in batches
    with h5py.File(embeddings_file, "a") as f:
        for i in tqdm(range(0, len(seq_names), batch_size), desc="Computing embeddings"):
            batch_names = seq_names[i : i + batch_size]
            batch_seqs = seq_list[i : i + batch_size]

            for name, seq in zip(batch_names, batch_seqs):
                try:
                    emb = embed_single_sequence(seq, tokenizer, model, device)
                    if name not in f:
                        f.create_dataset(name, data=emb)
                except Exception as e:
                    print(f"    Warning: Failed to embed {name}: {e}")

            # Flush periodically
            if (i // batch_size) % 100 == 0:
                f.flush()

    # Clean up
    del tokenizer, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Load all embeddings
    print("  Loading all embeddings...")
    embeddings = {}
    with h5py.File(embeddings_file, "r") as f:
        for key in tqdm(f.keys(), desc="Loading embeddings"):
            if key in sequences:
                embeddings[key] = f[key][()]

    return embeddings


# =============================================================================
# Statistical Analysis
# =============================================================================


def point_biserial_correlation(continuous: np.ndarray, binary: np.ndarray) -> tuple:
    """Point-biserial correlation coefficient."""
    return stats.pointbiserialr(binary, continuous)


def mann_whitney_u_test(group1: np.ndarray, group2: np.ndarray) -> tuple:
    """Mann-Whitney U test."""
    return stats.mannwhitneyu(group1, group2, alternative="two-sided")


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0


def analyze_property(utp_values: np.ndarray, non_utp_values: np.ndarray, prop_name: str) -> dict:
    """Comprehensive statistical analysis of a single property."""
    all_values = np.concatenate([utp_values, non_utp_values])
    labels = np.array([1] * len(utp_values) + [0] * len(non_utp_values))
    
    r, r_pval = point_biserial_correlation(all_values, labels)
    u_stat, u_pval = mann_whitney_u_test(utp_values, non_utp_values)
    d = cohens_d(utp_values, non_utp_values)

    return {
        "property": prop_name,
        "utp_mean": np.mean(utp_values),
        "utp_std": np.std(utp_values),
        "utp_median": np.median(utp_values),
        "utp_n": len(utp_values),
        "non_utp_mean": np.mean(non_utp_values),
        "non_utp_std": np.std(non_utp_values),
        "non_utp_median": np.median(non_utp_values),
        "non_utp_n": len(non_utp_values),
        "point_biserial_r": r,
        "point_biserial_pval": r_pval,
        "mann_whitney_u": u_stat,
        "mann_whitney_pval": u_pval,
        "cohens_d": d,
    }


# =============================================================================
# Visualization
# =============================================================================


def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """Create a covariance confidence ellipse."""
    if len(x) < 2:
        return None

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        **kwargs,
    )

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = (
        matplotlib.transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(np.mean(x), np.mean(y))
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def run_tsne(embeddings: np.ndarray, perplexity: int, random_state: int) -> np.ndarray:
    """Run t-SNE."""
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000,
        learning_rate="auto",
        init="pca",
    )
    return tsne.fit_transform(embeddings)


def run_umap(embeddings: np.ndarray, n_neighbors: int, min_dist: float, random_state: int) -> np.ndarray:
    """Run UMAP."""
    try:
        import umap
    except ImportError:
        print("  UMAP not installed, skipping.")
        return None

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric="cosine",
    )
    return reducer.fit_transform(embeddings)


def plot_embedding_2d(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_file: Path,
    method_params: str,
):
    """Plot 2D embedding with confidence ellipses."""
    from sklearn.metrics import silhouette_score

    fig, ax = plt.subplots(figsize=(10, 8))

    utp_mask = labels == 1
    non_utp_mask = labels == 0

    # Scatter plot
    ax.scatter(
        coords[non_utp_mask, 0],
        coords[non_utp_mask, 1],
        c=COLORS["Non-uTP"],
        label=f"Non-uTP (n={sum(non_utp_mask)})",
        alpha=0.4,
        s=15,
        edgecolors="none",
    )
    ax.scatter(
        coords[utp_mask, 0],
        coords[utp_mask, 1],
        c=COLORS["uTP"],
        label=f"uTP (n={sum(utp_mask)})",
        alpha=0.6,
        s=20,
        edgecolors="none",
    )

    # Add confidence ellipses
    confidence_ellipse(
        coords[non_utp_mask, 0],
        coords[non_utp_mask, 1],
        ax,
        n_std=2.0,
        facecolor="none",
        edgecolor=COLORS["Non-uTP"],
        linewidth=2,
        linestyle="--",
    )
    confidence_ellipse(
        coords[utp_mask, 0],
        coords[utp_mask, 1],
        ax,
        n_std=2.0,
        facecolor="none",
        edgecolor=COLORS["uTP"],
        linewidth=2,
        linestyle="--",
    )

    sil_score = silhouette_score(coords, labels)

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(f"{title}\n{method_params}\nSilhouette Score: {sil_score:.3f}")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    return sil_score


def plot_property_violin(
    utp_props: pd.DataFrame,
    non_utp_props: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_file: Path,
):
    """Create violin plots for all properties."""
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

    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()

    for i, prop in enumerate(properties):
        ax = axes[i]

        # Subsample non-uTP for visualization (too many points)
        n_utp = len(utp_props)
        non_utp_sample = non_utp_props.sample(n=min(5000, len(non_utp_props)), random_state=RANDOM_SEED)

        data = pd.DataFrame(
            {
                "value": pd.concat([utp_props[prop], non_utp_sample[prop]]),
                "group": ["uTP"] * len(utp_props) + ["Non-uTP"] * len(non_utp_sample),
            }
        )

        # Violin plot
        sns.violinplot(
            data=data,
            x="group",
            y="value",
            hue="group",
            ax=ax,
            palette=COLORS,
            inner="box",
            cut=0,
            legend=False,
        )

        # Get statistics
        row = stats_df[stats_df["property"] == prop].iloc[0]
        d = row["cohens_d"]
        pval = row["mann_whitney_pval"]

        # Significance markers
        sig = ""
        if pval < 0.001:
            sig = "***"
        elif pval < 0.01:
            sig = "**"
        elif pval < 0.05:
            sig = "*"

        ax.set_title(f"{prop}\nd={d:.2f} {sig}")
        ax.set_xlabel("")
        ax.set_ylabel(prop.replace("_", " ").title())

    plt.suptitle(
        f"Protein Properties: uTP (n={len(utp_props)}) vs Non-uTP (n={len(non_utp_props)})\n"
        "(*** p<0.001, ** p<0.01, * p<0.05)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


def plot_classification_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path,
):
    """Plot confusion matrix and ROC curve."""
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-uTP", "uTP"],
        yticklabels=["Non-uTP", "uTP"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Full Proteome Classification")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.svg", dpi=150, bbox_inches="tight")
    plt.close()

    # ROC curve
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve - Full Proteome")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve.svg", dpi=150, bbox_inches="tight")
        plt.close()

        return roc_auc
    return None


def plot_prediction_distribution(y_prob: np.ndarray, y_true: np.ndarray, output_file: Path):
    """Plot distribution of prediction probabilities by true class."""
    fig, ax = plt.subplots(figsize=(10, 6))

    utp_probs = y_prob[y_true == 1]
    non_utp_probs = y_prob[y_true == 0]

    ax.hist(non_utp_probs, bins=50, alpha=0.6, color=COLORS["Non-uTP"], 
            label=f"Non-uTP (n={len(non_utp_probs)})", density=True)
    ax.hist(utp_probs, bins=50, alpha=0.6, color=COLORS["uTP"], 
            label=f"uTP (n={len(utp_probs)})", density=True)

    ax.axvline(0.5, color="red", linestyle="--", label="Decision boundary")
    ax.set_xlabel("Predicted Probability (uTP)")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Probability Distribution by True Class")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("Full Proteome Analysis: uTP Presence Classifier")
    print("=" * 70)

    # Create output directory
    FULL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Check inputs
    if not UTP_PROTEINS_FILE.exists():
        raise FileNotFoundError(f"uTP proteins not found: {UTP_PROTEINS_FILE}")
    if not FULL_PROTEOME_FILE.exists():
        raise FileNotFoundError(f"Proteome not found: {FULL_PROTEOME_FILE}")
    if not CLASSIFIER_FILE.exists():
        raise FileNotFoundError(f"Classifier not found: {CLASSIFIER_FILE}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # =========================================================================
    # Load Sequences
    # =========================================================================
    print("\n[1/7] Loading sequences...")

    # Load uTP proteins
    utp_sequences = {
        record.id: str(record.seq)
        for record in SeqIO.parse(UTP_PROTEINS_FILE, "fasta")
    }
    utp_ids = set(utp_sequences.keys())
    print(f"  uTP proteins: {len(utp_sequences)}")

    # Load full proteome (excluding uTP proteins)
    non_utp_sequences = {}
    for record in tqdm(SeqIO.parse(FULL_PROTEOME_FILE, "fasta"), desc="Loading proteome"):
        if record.id not in utp_ids:
            non_utp_sequences[record.id] = str(record.seq)
    print(f"  Non-uTP proteins: {len(non_utp_sequences)}")

    # =========================================================================
    # Compute Properties
    # =========================================================================
    print("\n[2/7] Computing protein properties...")

    utp_props = compute_properties_batched(utp_sequences, "uTP properties")
    utp_props["label"] = "uTP"
    print(f"  uTP properties: {len(utp_props)}")

    non_utp_props = compute_properties_batched(non_utp_sequences, "Non-uTP properties")
    non_utp_props["label"] = "Non-uTP"
    print(f"  Non-uTP properties: {len(non_utp_props)}")

    # Save properties
    all_props = pd.concat([utp_props, non_utp_props], ignore_index=True)
    all_props.to_csv(FULL_RESULTS_DIR / "all_protein_properties.csv", index=False)

    # =========================================================================
    # Statistical Analysis
    # =========================================================================
    print("\n[3/7] Statistical analysis of protein properties...")

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

    stats_results = []
    for prop in properties:
        result = analyze_property(
            utp_props[prop].dropna().values,
            non_utp_props[prop].dropna().values,
            prop,
        )
        stats_results.append(result)
        print(
            f"  {prop:20s}: r={result['point_biserial_r']:+.3f}, "
            f"d={result['cohens_d']:+.3f}, p={result['mann_whitney_pval']:.2e}"
        )

    stats_df = pd.DataFrame(stats_results)
    stats_df.to_csv(FULL_RESULTS_DIR / "property_statistics.csv", index=False)

    # Plot properties
    plot_property_violin(utp_props, non_utp_props, stats_df, FULL_RESULTS_DIR / "property_violin_plots.svg")
    print("  Saved property_violin_plots.svg")

    # =========================================================================
    # Compute Embeddings
    # =========================================================================
    print("\n[4/7] Computing ProtT5 embeddings (this may take hours)...")

    # Combine all sequences
    all_sequences = dict(utp_sequences)
    all_sequences.update(non_utp_sequences)
    print(f"  Total sequences: {len(all_sequences)}")

    embeddings = compute_embeddings_batched(all_sequences, FULL_EMBEDDINGS_FILE, device)
    print(f"  Embeddings computed: {len(embeddings)}")

    # =========================================================================
    # Classification (Inference)
    # =========================================================================
    print("\n[5/7] Running classification inference...")

    # Load classifier
    clf_data = joblib.load(CLASSIFIER_FILE)
    classifier = clf_data["classifier"]
    scaler = clf_data["scaler"]

    # Prepare data
    names = []
    X = []
    y_true = []

    for name in utp_sequences:
        if name in embeddings:
            names.append(name)
            X.append(embeddings[name])
            y_true.append(1)

    for name in non_utp_sequences:
        if name in embeddings:
            names.append(name)
            X.append(embeddings[name])
            y_true.append(0)

    X = np.array(X)
    y_true = np.array(y_true)

    print(f"  Samples for classification: {len(X)} ({sum(y_true)} uTP, {len(y_true) - sum(y_true)} non-uTP)")

    # Predict
    X_scaled = scaler.transform(X)
    y_pred = classifier.predict(X_scaled)
    y_prob = classifier.predict_proba(X_scaled)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    roc_auc = plot_classification_results(y_true, y_pred, y_prob, FULL_RESULTS_DIR)

    print(f"\n  Classification Results:")
    print(f"    Accuracy:  {accuracy:.1%}")
    print(f"    Precision: {precision:.1%}")
    print(f"    Recall:    {recall:.1%}")
    print(f"    F1 Score:  {f1:.2f}")
    print(f"    ROC AUC:   {roc_auc:.3f}")

    # Save predictions
    predictions_df = pd.DataFrame({
        "name": names,
        "true_label": y_true,
        "predicted_label": y_pred,
        "probability": y_prob,
    })
    predictions_df.to_csv(FULL_RESULTS_DIR / "predictions.csv", index=False)

    # Plot prediction distribution
    plot_prediction_distribution(y_prob, y_true, FULL_RESULTS_DIR / "prediction_distribution.svg")
    print("  Saved prediction_distribution.svg")

    # =========================================================================
    # Embedding Visualization (Subsampled)
    # =========================================================================
    print("\n[6/7] Embedding visualization (subsampled for performance)...")

    # Subsample for visualization
    rng = np.random.RandomState(RANDOM_SEED)

    utp_indices = np.where(y_true == 1)[0]
    non_utp_indices = np.where(y_true == 0)[0]

    # Sample equal numbers from each class
    n_sample = min(VISUALIZATION_SUBSAMPLE, len(utp_indices), len(non_utp_indices))
    
    sampled_utp = rng.choice(utp_indices, size=n_sample, replace=False)
    sampled_non_utp = rng.choice(non_utp_indices, size=n_sample, replace=False)
    sampled_indices = np.concatenate([sampled_utp, sampled_non_utp])

    X_sampled = X_scaled[sampled_indices]
    y_sampled = y_true[sampled_indices]

    print(f"  Subsampled: {len(X_sampled)} ({n_sample} per class)")

    # t-SNE
    print("  Running t-SNE...")
    tsne_coords = run_tsne(X_sampled, TSNE_PERPLEXITY, RANDOM_SEED)
    tsne_sil = plot_embedding_2d(
        tsne_coords,
        y_sampled,
        "t-SNE: Full Proteome (subsampled)",
        FULL_RESULTS_DIR / "tsne.svg",
        f"perplexity={TSNE_PERPLEXITY}, n={len(X_sampled)}",
    )
    print(f"    Silhouette: {tsne_sil:.3f}")

    # UMAP
    print("  Running UMAP...")
    umap_coords = run_umap(X_sampled, UMAP_N_NEIGHBORS, UMAP_MIN_DIST, RANDOM_SEED)
    if umap_coords is not None:
        umap_sil = plot_embedding_2d(
            umap_coords,
            y_sampled,
            "UMAP: Full Proteome (subsampled)",
            FULL_RESULTS_DIR / "umap.svg",
            f"n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}, n={len(X_sampled)}",
        )
        print(f"    Silhouette: {umap_sil:.3f}")

    # =========================================================================
    # Summary Report
    # =========================================================================
    print("\n[7/7] Generating summary report...")

    summary_file = FULL_RESULTS_DIR / "summary.txt"
    with open(summary_file, "w") as f:
        f.write("Full Proteome Analysis Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write("Dataset:\n")
        f.write(f"  uTP proteins: {len(utp_sequences)}\n")
        f.write(f"  Non-uTP proteins: {len(non_utp_sequences)}\n")
        f.write(f"  Total: {len(all_sequences)}\n\n")

        f.write("Classification Results:\n")
        f.write(f"  Accuracy:  {accuracy:.1%}\n")
        f.write(f"  Precision: {precision:.1%}\n")
        f.write(f"  Recall:    {recall:.1%}\n")
        f.write(f"  F1 Score:  {f1:.2f}\n")
        f.write(f"  ROC AUC:   {roc_auc:.3f}\n\n")

        f.write("Embedding Visualization:\n")
        f.write(f"  t-SNE silhouette: {tsne_sil:.3f}\n")
        if umap_coords is not None:
            f.write(f"  UMAP silhouette: {umap_sil:.3f}\n")
        f.write(f"  Subsample size: {n_sample} per class\n\n")

        f.write("Significant Property Differences:\n")
        for _, row in stats_df.iterrows():
            direction = "higher in uTP" if row["cohens_d"] > 0 else "lower in uTP"
            effect = (
                "large" if abs(row["cohens_d"]) > 0.8 else (
                    "medium" if abs(row["cohens_d"]) > 0.5 else "small"
                )
            )
            f.write(f"  {row['property']}: {direction}, {effect} effect (d={row['cohens_d']:.2f})\n")

    print(f"\n  Saved summary to {summary_file}")

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)

    print(f"\nClassification on Full Proteome:")
    print(f"  {len(utp_sequences)} uTP vs {len(non_utp_sequences)} non-uTP proteins")
    print(f"  Accuracy: {accuracy:.1%}, ROC AUC: {roc_auc:.3f}")

    print(f"\nOutput files saved to: {FULL_RESULTS_DIR}")


if __name__ == "__main__":
    main()
