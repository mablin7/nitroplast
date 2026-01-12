#!/usr/bin/env python3
"""
04_interpretability.py - Interpretability Analysis for uTP Presence Classifier

This script provides rigorous interpretability analysis:
1. Embedding visualization (UMAP and t-SNE) with cluster quality metrics
2. Protein property correlations with statistical testing and effect sizes

Statistical Framework:
- Point-biserial correlation for continuous vs binary relationships
- Mann-Whitney U test (non-parametric, robust to non-normality)
- Cohen's d and Cliff's delta for effect sizes
- Bonferroni correction for multiple testing
- Silhouette score for cluster quality

Usage:
    uv run python experiments/utp_presence_classifier/04_interpretability.py
"""

import warnings
from pathlib import Path

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from matplotlib.patches import Ellipse
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

RANDOM_SEEDS = [42, 123, 456]  # Multiple seeds for stability assessment
TSNE_PERPLEXITY = 30  # Standard default, good for n > 100
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

ALPHA = 0.05  # Significance level
N_PROPERTIES = 8  # For Bonferroni correction

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"

MATURE_DOMAINS_FILE = OUTPUT_DIR / "mature_domains.fasta"
NUCLEAR_CYTOPLASMIC_CONTROLS_FILE = (
    OUTPUT_DIR / "filtered_controls_nuclear_cytoplasmic.fasta"
)
EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.h5"

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

# Colorblind-friendly palette
COLORS = {"uTP": "#E69F00", "Control": "#56B4E9"}


# =============================================================================
# Protein Properties
# =============================================================================


def get_protein_properties(seq_str: str) -> dict | None:
    """Calculate biophysical properties of a protein sequence."""
    # Remove ambiguous characters
    seq_clean = "".join(c for c in seq_str if c in "ACDEFGHIKLMNPQRSTVWY")
    if len(seq_clean) < 10:
        return None

    pa = ProteinAnalysis(seq_clean)

    try:
        return {
            "length": len(seq_clean),
            "molecular_weight": pa.molecular_weight(),
            "isoelectric_point": pa.isoelectric_point(),
            "gravy": pa.gravy(),  # hydrophobicity
            "instability_index": pa.instability_index(),
            "fraction_helix": pa.secondary_structure_fraction()[0],
            "fraction_sheet": pa.secondary_structure_fraction()[1],
            "fraction_coil": pa.secondary_structure_fraction()[2],
        }
    except Exception:
        return None


def compute_all_properties(sequences: dict[str, str]) -> pd.DataFrame:
    """Compute properties for all sequences."""
    rows = []
    for name, seq in tqdm(sequences.items(), desc="Computing protein properties"):
        props = get_protein_properties(seq)
        if props:
            props["name"] = name
            rows.append(props)
    return pd.DataFrame(rows)


# =============================================================================
# Statistical Tests
# =============================================================================


def point_biserial_correlation(continuous: np.ndarray, binary: np.ndarray) -> tuple:
    """
    Calculate point-biserial correlation coefficient.
    Equivalent to Pearson r when one variable is binary.
    """
    return stats.pointbiserialr(binary, continuous)


def mann_whitney_u_test(group1: np.ndarray, group2: np.ndarray) -> tuple:
    """
    Mann-Whitney U test (Wilcoxon rank-sum test).
    Non-parametric test for comparing two independent samples.
    """
    statistic, pvalue = stats.mannwhitneyu(group1, group2, alternative="two-sided")
    return statistic, pvalue


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.
    Interpretation: |d| < 0.2 small, 0.2-0.8 medium, > 0.8 large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0


def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cliff's delta effect size (non-parametric).
    Interpretation: |δ| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large
    """
    n1, n2 = len(group1), len(group2)
    # Count dominance
    greater = sum(1 for x in group1 for y in group2 if x > y)
    less = sum(1 for x in group1 for y in group2 if x < y)
    return (greater - less) / (n1 * n2)


def analyze_property(
    utp_values: np.ndarray, control_values: np.ndarray, property_name: str
) -> dict:
    """Comprehensive statistical analysis of a single property."""
    # Point-biserial correlation
    all_values = np.concatenate([utp_values, control_values])
    labels = np.array([1] * len(utp_values) + [0] * len(control_values))
    r, r_pval = point_biserial_correlation(all_values, labels)

    # Mann-Whitney U test
    u_stat, u_pval = mann_whitney_u_test(utp_values, control_values)

    # Effect sizes
    d = cohens_d(utp_values, control_values)
    delta = cliffs_delta(utp_values, control_values)

    # Descriptive statistics
    return {
        "property": property_name,
        "utp_mean": np.mean(utp_values),
        "utp_std": np.std(utp_values),
        "utp_median": np.median(utp_values),
        "control_mean": np.mean(control_values),
        "control_std": np.std(control_values),
        "control_median": np.median(control_values),
        "point_biserial_r": r,
        "point_biserial_pval": r_pval,
        "mann_whitney_u": u_stat,
        "mann_whitney_pval": u_pval,
        "cohens_d": d,
        "cliffs_delta": delta,
    }


# =============================================================================
# Embedding Visualization
# =============================================================================


def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """
    Create a covariance confidence ellipse.
    
    Parameters:
        n_std: Number of standard deviations (2 = ~95% confidence)
    """
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
    """Run t-SNE with specified parameters."""
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000,
        learning_rate="auto",
        init="pca",
    )
    return tsne.fit_transform(embeddings)


def run_umap(
    embeddings: np.ndarray, n_neighbors: int, min_dist: float, random_state: int
) -> np.ndarray:
    """Run UMAP with specified parameters."""
    try:
        import umap
    except ImportError:
        print("  UMAP not installed, skipping. Install with: pip install umap-learn")
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
    fig, ax = plt.subplots(figsize=(10, 8))

    # Separate by label
    utp_mask = labels == 1
    control_mask = labels == 0

    # Scatter plot
    ax.scatter(
        coords[control_mask, 0],
        coords[control_mask, 1],
        c=COLORS["Control"],
        label=f"Control (n={sum(control_mask)})",
        alpha=0.6,
        s=20,
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

    # Add confidence ellipses (95%)
    confidence_ellipse(
        coords[control_mask, 0],
        coords[control_mask, 1],
        ax,
        n_std=2.0,
        facecolor="none",
        edgecolor=COLORS["Control"],
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

    # Calculate silhouette score
    sil_score = silhouette_score(coords, labels)

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(f"{title}\n{method_params}\nSilhouette Score: {sil_score:.3f}")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    return sil_score


# =============================================================================
# Property Visualization
# =============================================================================


def plot_property_comparison(
    utp_props: pd.DataFrame,
    control_props: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_file: Path,
):
    """Create violin plots with statistical annotations for all properties."""
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

        # Prepare data for violin plot
        data = pd.DataFrame(
            {
                "value": pd.concat([utp_props[prop], control_props[prop]]),
                "group": ["uTP"] * len(utp_props) + ["Control"] * len(control_props),
            }
        )

        # Violin plot
        sns.violinplot(
            data=data,
            x="group",
            y="value",
            ax=ax,
            palette=COLORS,
            inner="box",
            cut=0,
        )

        # Add individual points (subsampled for clarity)
        for group, color in COLORS.items():
            group_data = data[data["group"] == group]["value"]
            if len(group_data) > 100:
                # Subsample for visibility
                idx = np.random.choice(len(group_data), 100, replace=False)
                group_data = group_data.iloc[idx]
            ax.scatter(
                np.random.normal(0 if group == "uTP" else 1, 0.04, len(group_data)),
                group_data,
                c=color,
                alpha=0.3,
                s=10,
                edgecolors="none",
            )

        # Get statistics
        row = stats_df[stats_df["property"] == prop].iloc[0]
        pval = row["mann_whitney_pval"]
        d = row["cohens_d"]

        # Bonferroni-corrected significance
        bonf_alpha = ALPHA / N_PROPERTIES
        sig_marker = ""
        if pval < bonf_alpha / 10:
            sig_marker = "***"
        elif pval < bonf_alpha:
            sig_marker = "**"
        elif pval < ALPHA:
            sig_marker = "*"

        # Annotation
        ax.set_title(f"{prop}\nd={d:.2f} {sig_marker}")
        ax.set_xlabel("")
        ax.set_ylabel(prop.replace("_", " ").title())

    plt.suptitle(
        "Protein Properties: uTP vs Control\n"
        f"(* p<{ALPHA:.2f}, ** p<{ALPHA/N_PROPERTIES:.4f} Bonferroni, "
        f"*** p<{ALPHA/N_PROPERTIES/10:.5f})",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(stats_df: pd.DataFrame, output_file: Path):
    """Create heatmap of effect sizes with significance markers."""
    # Prepare data
    props = stats_df["property"].tolist()
    correlations = stats_df["point_biserial_r"].values
    pvals = stats_df["mann_whitney_pval"].values
    cohens = stats_df["cohens_d"].values

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Correlation bar plot
    colors = [COLORS["uTP"] if r > 0 else COLORS["Control"] for r in correlations]
    bars = ax1.barh(props, correlations, color=colors, edgecolor="black", linewidth=0.5)

    # Add significance markers
    bonf_alpha = ALPHA / N_PROPERTIES
    for i, (r, p) in enumerate(zip(correlations, pvals)):
        marker = ""
        if p < bonf_alpha / 10:
            marker = "***"
        elif p < bonf_alpha:
            marker = "**"
        elif p < ALPHA:
            marker = "*"

        offset = 0.02 if r >= 0 else -0.02
        ha = "left" if r >= 0 else "right"
        ax1.text(r + offset, i, marker, ha=ha, va="center", fontsize=12, fontweight="bold")

    ax1.axvline(0, color="black", linewidth=0.5)
    ax1.set_xlabel("Point-Biserial Correlation (r)")
    ax1.set_title("Correlation with uTP Label")
    ax1.set_xlim(-0.5, 0.5)

    # Cohen's d bar plot
    colors = [COLORS["uTP"] if d > 0 else COLORS["Control"] for d in cohens]
    ax2.barh(props, cohens, color=colors, edgecolor="black", linewidth=0.5)

    # Add effect size interpretation lines
    for threshold, label in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        ax2.axvline(threshold, color="gray", linestyle="--", alpha=0.5)
        ax2.axvline(-threshold, color="gray", linestyle="--", alpha=0.5)

    ax2.axvline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Cohen's d (effect size)")
    ax2.set_title("Effect Size (uTP - Control)")
    ax2.set_yticklabels([])

    plt.suptitle(
        "Statistical Summary: Property Differences\n"
        "(Positive = higher in uTP, Negative = higher in Control)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("Interpretability Analysis: uTP Presence Classifier")
    print("=" * 70)

    # Check inputs
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(f"Embeddings not found: {EMBEDDINGS_FILE}")

    # =========================================================================
    # Load Data
    # =========================================================================
    print("\n[1/5] Loading data...")

    # Load sequences
    utp_sequences = {
        record.id: str(record.seq)
        for record in SeqIO.parse(MATURE_DOMAINS_FILE, "fasta")
    }
    control_sequences = {
        record.id: str(record.seq)
        for record in SeqIO.parse(NUCLEAR_CYTOPLASMIC_CONTROLS_FILE, "fasta")
    }
    print(f"  uTP sequences: {len(utp_sequences)}")
    print(f"  Control sequences: {len(control_sequences)}")

    # Load embeddings
    embeddings = {}
    with h5py.File(EMBEDDINGS_FILE, "r") as f:
        for key in f.keys():
            embeddings[key] = f[key][()]

    # Prepare embedding arrays
    utp_names = [n for n in utp_sequences if n in embeddings]
    control_names = [n for n in control_sequences if n in embeddings]

    utp_emb = np.array([embeddings[n] for n in utp_names])
    control_emb = np.array([embeddings[n] for n in control_names])

    all_emb = np.vstack([utp_emb, control_emb])
    all_labels = np.array([1] * len(utp_emb) + [0] * len(control_emb))

    print(f"  Embeddings loaded: {len(all_emb)} total")

    # =========================================================================
    # Compute Protein Properties
    # =========================================================================
    print("\n[2/5] Computing protein properties...")

    utp_props = compute_all_properties(
        {n: utp_sequences[n] for n in utp_names}
    )
    utp_props["label"] = "uTP"

    control_props = compute_all_properties(
        {n: control_sequences[n] for n in control_names}
    )
    control_props["label"] = "Control"

    print(f"  uTP properties computed: {len(utp_props)}")
    print(f"  Control properties computed: {len(control_props)}")

    # =========================================================================
    # Statistical Analysis of Properties
    # =========================================================================
    print("\n[3/5] Statistical analysis of protein properties...")

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
            utp_props[prop].values,
            control_props[prop].values,
            prop,
        )
        stats_results.append(result)

        # Print summary
        bonf_alpha = ALPHA / N_PROPERTIES
        sig = "**" if result["mann_whitney_pval"] < bonf_alpha else ""
        sig = "***" if result["mann_whitney_pval"] < bonf_alpha / 10 else sig
        print(
            f"  {prop:20s}: r={result['point_biserial_r']:+.3f}, "
            f"d={result['cohens_d']:+.3f}, p={result['mann_whitney_pval']:.2e} {sig}"
        )

    stats_df = pd.DataFrame(stats_results)
    stats_df.to_csv(OUTPUT_DIR / "property_statistics.csv", index=False)
    print(f"\n  Saved statistics to property_statistics.csv")

    # =========================================================================
    # Embedding Visualization
    # =========================================================================
    print("\n[4/5] Embedding visualization...")

    # Standardize embeddings for dimensionality reduction
    scaler = StandardScaler()
    all_emb_scaled = scaler.fit_transform(all_emb)

    silhouette_scores = {"tsne": [], "umap": []}

    # Run t-SNE with multiple seeds
    print("\n  Running t-SNE...")
    for seed in RANDOM_SEEDS:
        coords = run_tsne(all_emb_scaled, TSNE_PERPLEXITY, seed)
        sil = plot_embedding_2d(
            coords,
            all_labels,
            "t-SNE Embedding",
            OUTPUT_DIR / f"tsne_seed{seed}.svg",
            f"perplexity={TSNE_PERPLEXITY}, seed={seed}",
        )
        silhouette_scores["tsne"].append(sil)
        print(f"    Seed {seed}: silhouette={sil:.3f}")

    # Run UMAP with multiple seeds
    print("\n  Running UMAP...")
    for seed in RANDOM_SEEDS:
        coords = run_umap(all_emb_scaled, UMAP_N_NEIGHBORS, UMAP_MIN_DIST, seed)
        if coords is not None:
            sil = plot_embedding_2d(
                coords,
                all_labels,
                "UMAP Embedding",
                OUTPUT_DIR / f"umap_seed{seed}.svg",
                f"n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}, seed={seed}",
            )
            silhouette_scores["umap"].append(sil)
            print(f"    Seed {seed}: silhouette={sil:.3f}")

    # Report stability
    print("\n  Visualization Stability:")
    for method, scores in silhouette_scores.items():
        if scores:
            print(
                f"    {method.upper()}: silhouette = {np.mean(scores):.3f} ± {np.std(scores):.3f}"
            )

    # =========================================================================
    # Property Visualizations
    # =========================================================================
    print("\n[5/5] Generating property visualizations...")

    plot_property_comparison(
        utp_props,
        control_props,
        stats_df,
        OUTPUT_DIR / "property_violin_plots.svg",
    )
    print("  Saved property_violin_plots.svg")

    plot_correlation_heatmap(stats_df, OUTPUT_DIR / "property_correlations.svg")
    print("  Saved property_correlations.svg")

    # =========================================================================
    # Summary Report
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary Report")
    print("=" * 70)

    # Significant properties (Bonferroni-corrected)
    bonf_alpha = ALPHA / N_PROPERTIES
    significant = stats_df[stats_df["mann_whitney_pval"] < bonf_alpha]

    print(f"\nSignificant properties (Bonferroni α={bonf_alpha:.4f}):")
    if len(significant) > 0:
        for _, row in significant.iterrows():
            direction = "higher in uTP" if row["cohens_d"] > 0 else "higher in Control"
            effect = "large" if abs(row["cohens_d"]) > 0.8 else (
                "medium" if abs(row["cohens_d"]) > 0.5 else "small"
            )
            print(
                f"  - {row['property']}: {direction}, {effect} effect (d={row['cohens_d']:.2f})"
            )
    else:
        print("  None (after multiple testing correction)")

    # Embedding separation
    print(f"\nEmbedding cluster separation:")
    for method, scores in silhouette_scores.items():
        if scores:
            mean_sil = np.mean(scores)
            interpretation = (
                "strong" if mean_sil > 0.5 else (
                    "moderate" if mean_sil > 0.25 else "weak"
                )
            )
            print(f"  {method.upper()}: {mean_sil:.3f} ({interpretation} separation)")

    # Save summary
    summary_file = OUTPUT_DIR / "interpretability_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Interpretability Analysis Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write("Statistical Parameters:\n")
        f.write(f"  - Significance level (α): {ALPHA}\n")
        f.write(f"  - Bonferroni-corrected α: {bonf_alpha:.4f}\n")
        f.write(f"  - Number of properties tested: {N_PROPERTIES}\n\n")

        f.write("Embedding Visualization:\n")
        f.write(f"  - t-SNE perplexity: {TSNE_PERPLEXITY}\n")
        f.write(f"  - UMAP n_neighbors: {UMAP_N_NEIGHBORS}\n")
        f.write(f"  - UMAP min_dist: {UMAP_MIN_DIST}\n")
        f.write(f"  - Random seeds: {RANDOM_SEEDS}\n\n")

        f.write("Silhouette Scores:\n")
        for method, scores in silhouette_scores.items():
            if scores:
                f.write(f"  {method.upper()}: {np.mean(scores):.3f} ± {np.std(scores):.3f}\n")

        f.write("\nSignificant Properties (Bonferroni-corrected):\n")
        for _, row in significant.iterrows():
            f.write(
                f"  {row['property']}: r={row['point_biserial_r']:.3f}, "
                f"d={row['cohens_d']:.3f}, p={row['mann_whitney_pval']:.2e}\n"
            )

    print(f"\n  Saved summary to {summary_file}")


if __name__ == "__main__":
    main()
