#!/usr/bin/env python3
"""
02_generate_figures.py - Visualization for uTP-Mature Correlation Analysis

Generates publication-quality figures:
1. Correlation heatmap (mature vs uTP features)
2. Same-feature correlation plot (do properties match?)
3. Scatter plots for strongest correlations
4. CCA visualization
5. Volcano plot

Usage:
    uv run python experiments/utp_mature_correlation/02_generate_figures.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"

# Input files
PAIRWISE_RESULTS = OUTPUT_DIR / "pairwise_correlations.csv"
SAME_FEATURE = OUTPUT_DIR / "same_feature_correlations.csv"
FOCUSED_RESULTS = OUTPUT_DIR / "focused_correlations.csv"
CCA_RESULTS = OUTPUT_DIR / "cca_results.csv"
FEATURES_FILE = OUTPUT_DIR / "mature_utp_features.csv"

# Style
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["figure.dpi"] = 150

COLORS = {
    "significant": "#e74c3c",
    "non_significant": "#95a5a6",
    "positive": "#e74c3c",
    "negative": "#3498db",
    "same_feature": "#9b59b6",
}


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_correlation_heatmap(results_df: pd.DataFrame, output_path: Path):
    """Generate correlation heatmap between mature and uTP features."""

    # Filter to 'all' subset
    data = results_df[results_df["subset"] == "all"].copy()

    if len(data) == 0:
        print("  No data for heatmap")
        return

    # Create pivot table
    pivot = data.pivot(index="mature_base", columns="utp_base", values="spearman_rho")

    # Filter to features with at least one notable correlation
    threshold = 0.15
    rows_to_keep = pivot.abs().max(axis=1) > threshold
    cols_to_keep = pivot.abs().max(axis=0) > threshold

    pivot_filtered = pivot.loc[rows_to_keep, cols_to_keep]

    if pivot_filtered.empty:
        print("  No correlations above threshold for heatmap")
        # Show all features instead
        pivot_filtered = pivot

    # Sort by mean correlation
    row_order = pivot_filtered.abs().mean(axis=1).sort_values(ascending=True).index
    col_order = pivot_filtered.abs().mean(axis=0).sort_values(ascending=False).index
    pivot_sorted = pivot_filtered.loc[row_order, col_order]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))

    # Heatmap
    sns.heatmap(
        pivot_sorted,
        cmap="RdBu_r",
        center=0,
        vmin=-0.6,
        vmax=0.6,
        annot=False,
        ax=ax,
        cbar_kws={"label": "Spearman ρ"},
    )

    # Mark diagonal (same features)
    for i, row_feat in enumerate(pivot_sorted.index):
        for j, col_feat in enumerate(pivot_sorted.columns):
            if row_feat == col_feat:
                rect = plt.Rectangle(
                    (j, i), 1, 1, fill=False, edgecolor="gold", linewidth=2
                )
                ax.add_patch(rect)

    ax.set_xlabel("uTP Feature")
    ax.set_ylabel("Mature Domain Feature")
    ax.set_title("Mature vs uTP Feature Correlations\n(gold boxes = same feature type)")

    # Clean labels
    ax.set_xticklabels(
        [f.replace("_", " ") for f in pivot_sorted.columns],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_yticklabels(
        [f.replace("_", " ") for f in pivot_sorted.index], rotation=0, fontsize=8
    )

    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".svg"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap to {output_path}")


def plot_same_feature_correlations(same_df: pd.DataFrame, output_path: Path):
    """Plot correlations between same features in mature vs uTP."""

    # Filter to 'all' subset
    data = same_df[same_df["subset"] == "all"].copy()

    if len(data) == 0:
        print("  No same-feature data")
        return

    # Sort by absolute correlation
    data["abs_rho"] = data["spearman_rho"].abs()
    data = data.sort_values("abs_rho", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(data) * 0.3)))

    y_pos = np.arange(len(data))
    colors = [
        COLORS["significant"] if row["significant_fdr"] else COLORS["non_significant"]
        for _, row in data.iterrows()
    ]

    ax.barh(y_pos, data["spearman_rho"], color=colors, alpha=0.7)

    # Add zero line
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    # Add significance markers
    for i, (_, row) in enumerate(data.iterrows()):
        if row["significant_fdr"]:
            x_pos = row["spearman_rho"]
            marker_pos = x_pos + 0.02 if x_pos > 0 else x_pos - 0.02
            ha = "left" if x_pos > 0 else "right"
            ax.text(
                marker_pos, i, "*", fontsize=14, va="center", ha=ha, fontweight="bold"
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace("_", " ") for f in data["mature_base"]], fontsize=9)
    ax.set_xlabel("Spearman ρ (mature vs uTP)")
    ax.set_title(
        "Same-Feature Correlations\n(Does mature domain property match uTP property?)\n* = FDR < 0.05"
    )

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=COLORS["significant"], label="Significant"),
        Patch(facecolor=COLORS["non_significant"], label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".svg"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved same-feature plot to {output_path}")


def plot_focused_with_ci(focused_df: pd.DataFrame, output_path: Path):
    """Plot focused correlations with bootstrap confidence intervals."""

    data = focused_df[focused_df["subset"] == "all"].copy()

    if len(data) == 0:
        print("  No focused data")
        return

    # Sort by absolute correlation
    data["abs_rho"] = data["spearman_rho"].abs()
    data = data.sort_values("abs_rho", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(data) * 0.3)))

    y_pos = np.arange(len(data))

    for i, (_, row) in enumerate(data.iterrows()):
        color = (
            COLORS["significant"]
            if row["significant_fdr"]
            else COLORS["non_significant"]
        )

        # Error bars
        if not np.isnan(row["ci_lower"]):
            ax.errorbar(
                row["spearman_rho"],
                i,
                xerr=[
                    [row["spearman_rho"] - row["ci_lower"]],
                    [row["ci_upper"] - row["spearman_rho"]],
                ],
                fmt="o",
                color=color,
                capsize=3,
                capthick=1,
                markersize=8,
            )
        else:
            ax.plot(row["spearman_rho"], i, "o", color=color, markersize=8)

    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{row['category']}: {row['feature']}" for _, row in data.iterrows()],
        fontsize=9,
    )
    ax.set_xlabel("Spearman ρ (with 95% bootstrap CI)")
    ax.set_title(
        "Focused Feature Correlations (Mature vs uTP)\nSame feature type comparison"
    )

    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".svg"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved focused CI plot to {output_path}")


def plot_volcano(results_df: pd.DataFrame, output_path: Path):
    """Generate volcano plot."""

    data = results_df[results_df["subset"] == "all"].copy()

    if len(data) == 0:
        print("  No data for volcano")
        return

    data["neg_log_p"] = -np.log10(data["p_adjusted"].replace(0, 1e-300))

    fig, ax = plt.subplots(figsize=(10, 8))

    # Non-significant
    nonsig = data[~data["significant_fdr"].fillna(False)]
    ax.scatter(
        nonsig["spearman_rho"],
        nonsig["neg_log_p"],
        c=COLORS["non_significant"],
        alpha=0.3,
        s=20,
        label="Not significant",
    )

    # Significant
    sig = data[data["significant_fdr"].fillna(False)]

    # Same feature vs different
    sig_same = sig[sig["same_feature_type"]]
    sig_diff = sig[~sig["same_feature_type"]]

    ax.scatter(
        sig_diff["spearman_rho"],
        sig_diff["neg_log_p"],
        c=COLORS["significant"],
        alpha=0.7,
        s=40,
        label="Significant (different features)",
    )
    ax.scatter(
        sig_same["spearman_rho"],
        sig_same["neg_log_p"],
        c=COLORS["same_feature"],
        alpha=0.9,
        s=60,
        marker="D",
        label="Significant (same feature)",
    )

    # FDR threshold line
    fdr_line = -np.log10(0.05)
    ax.axhline(y=fdr_line, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Label top correlations
    if len(sig) > 0:
        top_sig = sig.nlargest(8, "neg_log_p")
        for _, row in top_sig.iterrows():
            label = f"{row['mature_base'][:10]}\nvs {row['utp_base'][:10]}"
            ax.annotate(
                label,
                (row["spearman_rho"], row["neg_log_p"]),
                fontsize=7,
                ha="center",
                va="bottom",
                xytext=(0, 5),
                textcoords="offset points",
            )

    ax.set_xlabel("Spearman ρ")
    ax.set_ylabel("-log₁₀(FDR-adjusted p-value)")
    ax.set_title(
        f"Mature vs uTP Feature Correlations\n"
        f"n={len(data)} tests, {len(sig)} significant"
    )
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".svg"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved volcano plot to {output_path}")


def plot_scatter_top_correlations(
    features_df: pd.DataFrame,
    results_df: pd.DataFrame,
    output_path: Path,
    n_plots: int = 6,
):
    """Scatter plots for top correlations."""

    if features_df is None or len(features_df) == 0:
        print("  No features data for scatter")
        return

    sig_data = results_df[
        (results_df["subset"] == "all") & results_df["significant_fdr"].fillna(False)
    ].nsmallest(n_plots, "p_adjusted")

    if len(sig_data) == 0:
        print("  No significant correlations for scatter")
        return

    n_cols = min(3, len(sig_data))
    n_rows = int(np.ceil(len(sig_data) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows * n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, (_, row) in enumerate(sig_data.iterrows()):
        ax = axes[idx]

        mature_col = row["mature_feature"]
        utp_col = row["utp_feature"]

        x = features_df[mature_col].values
        y = features_df[utp_col].values

        # Remove NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]

        # Color by same feature
        color = COLORS["same_feature"] if row["same_feature_type"] else "#3498db"
        ax.scatter(x_clean, y_clean, c=color, alpha=0.5, s=20)

        # Trend line
        z = np.polyfit(x_clean, y_clean, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
        ax.plot(x_line, p(x_line), "r-", linewidth=2, alpha=0.7)

        ax.set_xlabel(f"Mature: {row['mature_base']}")
        ax.set_ylabel(f"uTP: {row['utp_base']}")

        same_label = " [SAME]" if row["same_feature_type"] else ""
        ax.set_title(
            f"ρ = {row['spearman_rho']:.3f}{same_label}\np = {row['p_adjusted']:.2e}"
        )

    # Hide unused
    for idx in range(len(sig_data), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".svg"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved scatter plots to {output_path}")


def plot_cca_results(cca_df: pd.DataFrame, output_path: Path):
    """Plot CCA canonical correlations."""

    if len(cca_df) == 0:
        print("  No CCA data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    subsets = cca_df["subset"].unique()
    x = np.arange(max(cca_df["component"]))
    width = 0.25

    colors = {"all": "#2ecc71", "experimental": "#3498db", "hmm_only": "#e67e22"}

    for i, subset in enumerate(subsets):
        subset_data = cca_df[cca_df["subset"] == subset].sort_values("component")
        if len(subset_data) > 0:
            ax.bar(
                x[: len(subset_data)] + i * width,
                subset_data["canonical_correlation"].values,
                width,
                label=subset,
                color=colors.get(subset, "#333"),
            )

    ax.set_xlabel("Canonical Component")
    ax.set_ylabel("Canonical Correlation")
    ax.set_title(
        "Canonical Correlation Analysis\n(Multivariate relationship between mature and uTP features)"
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"CC{i+1}" for i in range(len(x))])
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".svg"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved CCA plot to {output_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("Generating Figures for uTP-Mature Correlation Analysis")
    print("=" * 70)

    # =========================================================================
    # Load data
    # =========================================================================
    print("\n[1/7] Loading data...")

    if not PAIRWISE_RESULTS.exists():
        print(f"  Error: {PAIRWISE_RESULTS} not found")
        print("  Run 01_correlation_analysis.py first")
        return

    results_df = pd.read_csv(PAIRWISE_RESULTS)
    print(f"  Loaded {len(results_df)} pairwise results")

    same_df = pd.read_csv(SAME_FEATURE) if SAME_FEATURE.exists() else None
    focused_df = pd.read_csv(FOCUSED_RESULTS) if FOCUSED_RESULTS.exists() else None
    cca_df = pd.read_csv(CCA_RESULTS) if CCA_RESULTS.exists() else None
    features_df = pd.read_csv(FEATURES_FILE) if FEATURES_FILE.exists() else None

    # =========================================================================
    # Generate figures
    # =========================================================================

    print("\n[2/7] Generating correlation heatmap...")
    plot_correlation_heatmap(results_df, OUTPUT_DIR / "figure_heatmap")

    print("\n[3/7] Generating same-feature correlation plot...")
    if same_df is not None:
        plot_same_feature_correlations(same_df, OUTPUT_DIR / "figure_same_feature")

    print("\n[4/7] Generating focused CI plot...")
    if focused_df is not None:
        plot_focused_with_ci(focused_df, OUTPUT_DIR / "figure_focused_ci")

    print("\n[5/7] Generating volcano plot...")
    plot_volcano(results_df, OUTPUT_DIR / "figure_volcano")

    print("\n[6/7] Generating scatter plots...")
    plot_scatter_top_correlations(
        features_df, results_df, OUTPUT_DIR / "figure_scatter"
    )

    print("\n[7/7] Generating CCA plot...")
    if cca_df is not None and len(cca_df) > 0:
        plot_cca_results(cca_df, OUTPUT_DIR / "figure_cca")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nFigures generated:")
    print(f"  - figure_heatmap.png/svg: Correlation heatmap")
    print(f"  - figure_same_feature.png/svg: Same-feature correlations")
    print(f"  - figure_focused_ci.png/svg: Focused correlations with bootstrap CI")
    print(f"  - figure_volcano.png/svg: Volcano plot")
    print(f"  - figure_scatter.png/svg: Scatter plots for top correlations")
    print(f"  - figure_cca.png/svg: Canonical correlation analysis")

    print(f"\n📁 Figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
