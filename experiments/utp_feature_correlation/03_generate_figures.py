#!/usr/bin/env python3
"""
03_generate_figures.py - Visualization for Correlation Analysis

Generates publication-quality figures for the uTP feature correlation analysis:

1. Correlation heatmap (features vs outcomes)
2. Volcano plot (effect size vs significance)
3. Top correlations with bootstrap CIs
4. Scatter plots for strongest correlations
5. Comparison across subsets

Usage:
    uv run python experiments/utp_feature_correlation/03_generate_figures.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"

# Input files
CORRELATION_RESULTS = OUTPUT_DIR / "correlation_results.csv"
SIGNIFICANT_RESULTS = OUTPUT_DIR / "significant_correlations.csv"
BOOTSTRAP_CI = OUTPUT_DIR / "bootstrap_confidence_intervals.csv"
PERMUTATION_RESULTS = OUTPUT_DIR / "permutation_validation.csv"
MERGED_DATA = OUTPUT_DIR / "utp_features_with_outcomes.csv"

# Style configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['figure.dpi'] = 150

# Color palette
COLORS = {
    'significant': '#e74c3c',
    'non_significant': '#95a5a6',
    'positive': '#e74c3c',
    'negative': '#3498db',
    'experimental': '#3498db',
    'hmm_only': '#e67e22',
    'all': '#2ecc71',
}


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_correlation_heatmap(results_df: pd.DataFrame, output_path: Path):
    """Generate correlation heatmap for features vs outcomes."""
    
    # Filter to 'all' subset and pivot
    all_data = results_df[results_df['subset'] == 'all'].copy()
    
    if len(all_data) == 0:
        print("  No data for heatmap")
        return
    
    # Get unique features and outcomes
    features = all_data['feature'].unique()
    outcomes = all_data['outcome'].unique()
    
    # Create pivot table for correlations
    pivot_rho = all_data.pivot(index='feature', columns='outcome', values='spearman_rho')
    pivot_sig = all_data.pivot(index='feature', columns='outcome', values='significant_fdr')
    
    # Filter to features with at least one significant correlation
    sig_features = []
    for feat in pivot_rho.index:
        if pivot_sig.loc[feat].any():
            sig_features.append(feat)
    
    if len(sig_features) == 0:
        # If no significant, show top features by max correlation
        max_corr = pivot_rho.abs().max(axis=1)
        sig_features = max_corr.nlargest(30).index.tolist()
    
    # Subset and sort
    pivot_rho_subset = pivot_rho.loc[sig_features]
    pivot_sig_subset = pivot_sig.loc[sig_features]
    
    # Sort by mean absolute correlation
    sort_order = pivot_rho_subset.abs().mean(axis=1).sort_values(ascending=True).index
    pivot_rho_subset = pivot_rho_subset.loc[sort_order]
    pivot_sig_subset = pivot_sig_subset.loc[sort_order]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(sig_features) * 0.3)))
    
    # Heatmap
    sns.heatmap(
        pivot_rho_subset,
        cmap='RdBu_r',
        center=0,
        vmin=-0.5,
        vmax=0.5,
        annot=False,
        ax=ax,
        cbar_kws={'label': 'Spearman ρ'}
    )
    
    # Add asterisks for significant correlations
    for i, feat in enumerate(pivot_rho_subset.index):
        for j, outcome in enumerate(pivot_rho_subset.columns):
            if pivot_sig_subset.loc[feat, outcome]:
                ax.text(j + 0.5, i + 0.5, '*', ha='center', va='center',
                       fontsize=12, fontweight='bold', color='black')
    
    ax.set_xlabel('Outcome')
    ax.set_ylabel('uTP Feature')
    ax.set_title(f'Correlation Heatmap (* = FDR < 0.05)\nn={len(sig_features)} features with significant correlations')
    
    # Clean up feature names for display
    yticklabels = [f.replace('utp_', '').replace('_', ' ') for f in pivot_rho_subset.index]
    ax.set_yticklabels(yticklabels, rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.svg'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap to {output_path}")


def plot_volcano(results_df: pd.DataFrame, output_path: Path):
    """Generate volcano plot (effect size vs significance)."""
    
    # Filter to 'all' subset
    data = results_df[results_df['subset'] == 'all'].copy()
    
    if len(data) == 0:
        print("  No data for volcano plot")
        return
    
    # Compute -log10(p_adjusted)
    data['neg_log_p'] = -np.log10(data['p_adjusted'].replace(0, 1e-300))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Separate significant and non-significant
    sig = data[data['significant_fdr'].fillna(False)]
    nonsig = data[~data['significant_fdr'].fillna(False)]
    
    # Plot non-significant
    ax.scatter(nonsig['spearman_rho'], nonsig['neg_log_p'],
               c=COLORS['non_significant'], alpha=0.3, s=20, label='Not significant')
    
    # Plot significant (color by direction)
    sig_pos = sig[sig['spearman_rho'] > 0]
    sig_neg = sig[sig['spearman_rho'] < 0]
    
    ax.scatter(sig_pos['spearman_rho'], sig_pos['neg_log_p'],
               c=COLORS['positive'], alpha=0.7, s=40, label='Positive (FDR<0.05)')
    ax.scatter(sig_neg['spearman_rho'], sig_neg['neg_log_p'],
               c=COLORS['negative'], alpha=0.7, s=40, label='Negative (FDR<0.05)')
    
    # Add significance threshold line
    fdr_threshold = -np.log10(0.05)
    ax.axhline(y=fdr_threshold, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(ax.get_xlim()[1], fdr_threshold, ' FDR=0.05', va='center', fontsize=9, color='gray')
    
    # Add effect size thresholds
    for rho_thresh in [-0.3, 0.3]:
        ax.axvline(x=rho_thresh, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Label top significant correlations
    if len(sig) > 0:
        top_sig = sig.nlargest(10, 'neg_log_p')
        for _, row in top_sig.iterrows():
            feat_short = row['feature'].replace('utp_', '').replace('_', ' ')[:15]
            out_short = row['outcome'].replace('log_', '').replace('_', ' ')[:10]
            label = f"{feat_short}\nvs {out_short}"
            ax.annotate(label, (row['spearman_rho'], row['neg_log_p']),
                       fontsize=7, ha='center', va='bottom',
                       xytext=(0, 5), textcoords='offset points')
    
    ax.set_xlabel('Spearman ρ')
    ax.set_ylabel('-log₁₀(FDR-adjusted p-value)')
    ax.set_title(f'Volcano Plot: uTP Features vs Functional Outcomes\n'
                f'n={len(data)} tests, {len(sig)} significant (FDR<0.05)')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.svg'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved volcano plot to {output_path}")


def plot_top_correlations_with_ci(results_df: pd.DataFrame, bootstrap_df: pd.DataFrame,
                                  output_path: Path):
    """Plot top correlations with bootstrap confidence intervals."""
    
    if bootstrap_df is None or len(bootstrap_df) == 0:
        print("  No bootstrap data for CI plot")
        return
    
    # Merge with significance info
    sig_data = results_df[
        (results_df['subset'] == 'all') &
        results_df['significant_fdr'].fillna(False)
    ]
    
    # Merge bootstrap with significance
    merged = bootstrap_df.merge(
        sig_data[['feature', 'outcome', 'p_adjusted']],
        on=['feature', 'outcome'],
        how='left'
    )
    
    # Sort by absolute rho
    merged['abs_rho'] = merged['rho'].abs()
    merged = merged.sort_values('abs_rho', ascending=True).tail(20)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(merged))
    
    # Plot error bars
    for i, (_, row) in enumerate(merged.iterrows()):
        color = COLORS['positive'] if row['rho'] > 0 else COLORS['negative']
        
        ax.errorbar(row['rho'], i,
                   xerr=[[row['rho'] - row['ci_lower']], [row['ci_upper'] - row['rho']]],
                   fmt='o', color=color, capsize=3, capthick=1, markersize=8)
    
    # Add zero line
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    
    # Y-axis labels
    labels = [f"{row['feature'].replace('utp_', '')[:20]}\nvs {row['outcome'][:15]}"
              for _, row in merged.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    
    ax.set_xlabel('Spearman ρ (with 95% bootstrap CI)')
    ax.set_title(f'Top {len(merged)} Significant Correlations with Bootstrap Confidence Intervals')
    
    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.svg'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved CI plot to {output_path}")


def plot_scatter_top_correlations(merged_df: pd.DataFrame, results_df: pd.DataFrame,
                                  output_path: Path, n_plots: int = 6):
    """Generate scatter plots for top correlations."""
    
    if merged_df is None or len(merged_df) == 0:
        print("  No data for scatter plots")
        return
    
    # Get top correlations
    sig_data = results_df[
        (results_df['subset'] == 'all') &
        results_df['significant_fdr'].fillna(False)
    ].nsmallest(n_plots, 'p_adjusted')
    
    if len(sig_data) == 0:
        print("  No significant correlations for scatter plots")
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
        
        feature = row['feature']
        outcome = row['outcome']
        
        x = merged_df[feature].values
        y = merged_df[outcome].values
        
        # Remove NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Color by source if available
        if 'source' in merged_df.columns:
            source = merged_df.loc[mask, 'source'].values
            colors = [COLORS.get(s, '#333333') for s in source]
            ax.scatter(x_clean, y_clean, c=colors, alpha=0.5, s=20)
        else:
            ax.scatter(x_clean, y_clean, c='#3498db', alpha=0.5, s=20)
        
        # Add trend line
        z = np.polyfit(x_clean, y_clean, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.7)
        
        # Labels
        feat_short = feature.replace('utp_', '').replace('_', ' ')
        ax.set_xlabel(feat_short)
        ax.set_ylabel(outcome)
        ax.set_title(f"ρ = {row['spearman_rho']:.3f}, p = {row['p_adjusted']:.2e}\nn = {row['n']}")
    
    # Hide unused axes
    for idx in range(len(sig_data), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.svg'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved scatter plots to {output_path}")


def plot_subset_comparison(results_df: pd.DataFrame, output_path: Path):
    """Compare correlations across subsets (experimental vs HMM-only)."""
    
    subsets = results_df['subset'].unique()
    
    if len(subsets) < 2:
        print("  Only one subset, skipping comparison plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Number of significant correlations
    ax = axes[0]
    
    sig_counts = []
    total_counts = []
    for subset in subsets:
        subset_data = results_df[results_df['subset'] == subset]
        n_sig = subset_data['significant_fdr'].fillna(False).sum()
        n_total = subset_data['spearman_p'].notna().sum()
        sig_counts.append(n_sig)
        total_counts.append(n_total)
    
    x = np.arange(len(subsets))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, total_counts, width, label='Total tests', color='#95a5a6')
    bars2 = ax.bar(x + width/2, sig_counts, width, label='Significant (FDR<0.05)', color='#e74c3c')
    
    ax.set_xticks(x)
    ax.set_xticklabels(subsets)
    ax.set_ylabel('Count')
    ax.set_title('Significant Correlations by Subset')
    ax.legend()
    
    # Add percentage labels
    for bar, sig, total in zip(bars2, sig_counts, total_counts):
        pct = 100 * sig / total if total > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Panel 2: Distribution of correlation strengths
    ax = axes[1]
    
    for subset in subsets:
        subset_data = results_df[
            (results_df['subset'] == subset) &
            results_df['spearman_rho'].notna()
        ]
        color = COLORS.get(subset, '#333333')
        sns.kdeplot(subset_data['spearman_rho'], ax=ax, label=subset, color=color, fill=True, alpha=0.3)
    
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Spearman ρ')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Correlation Coefficients')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.svg'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved subset comparison to {output_path}")


def plot_effect_size_summary(results_df: pd.DataFrame, output_path: Path):
    """Plot effect size distribution for significant correlations."""
    
    sig_data = results_df[results_df['significant_fdr'].fillna(False)].copy()
    
    if len(sig_data) == 0:
        print("  No significant correlations for effect size plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Histogram of effect sizes
    ax = axes[0]
    
    ax.hist(sig_data['spearman_rho'], bins=30, edgecolor='white', 
            color='#3498db', alpha=0.7)
    
    # Add effect size thresholds
    for thresh, label in [(-0.5, 'Large-'), (-0.3, 'Medium-'), (-0.1, 'Small-'),
                          (0.1, 'Small+'), (0.3, 'Medium+'), (0.5, 'Large+')]:
        ax.axvline(x=thresh, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Spearman ρ')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of Significant Correlations\n(n={len(sig_data)})')
    
    # Panel 2: Effect size categories
    ax = axes[1]
    
    effect_counts = sig_data['effect_size'].value_counts()
    effect_order = ['negligible', 'small', 'medium', 'large']
    effect_counts = effect_counts.reindex(effect_order, fill_value=0)
    
    colors_effect = ['#95a5a6', '#f1c40f', '#e67e22', '#e74c3c']
    ax.bar(effect_counts.index, effect_counts.values, color=colors_effect)
    
    ax.set_xlabel('Effect Size Category')
    ax.set_ylabel('Count')
    ax.set_title("Effect Size Distribution\n(Cohen's guidelines)")
    
    # Add percentage labels
    total = len(sig_data)
    for i, (cat, count) in enumerate(effect_counts.items()):
        pct = 100 * count / total if total > 0 else 0
        ax.text(i, count + 0.5, f'{pct:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.svg'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved effect size summary to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Generating Figures for Correlation Analysis")
    print("=" * 70)
    
    # =========================================================================
    # Load data
    # =========================================================================
    print("\n[1/7] Loading data...")
    
    if not CORRELATION_RESULTS.exists():
        print(f"  Error: {CORRELATION_RESULTS} not found")
        print("  Run 02_correlation_analysis.py first")
        return
    
    results_df = pd.read_csv(CORRELATION_RESULTS)
    print(f"  Loaded {len(results_df)} correlation results")
    
    # Load optional files
    bootstrap_df = None
    if BOOTSTRAP_CI.exists():
        bootstrap_df = pd.read_csv(BOOTSTRAP_CI)
        print(f"  Loaded {len(bootstrap_df)} bootstrap results")
    
    merged_df = None
    if MERGED_DATA.exists():
        merged_df = pd.read_csv(MERGED_DATA)
        print(f"  Loaded {len(merged_df)} merged data rows")
    
    # =========================================================================
    # Generate figures
    # =========================================================================
    
    print("\n[2/7] Generating correlation heatmap...")
    plot_correlation_heatmap(results_df, OUTPUT_DIR / "figure_heatmap")
    
    print("\n[3/7] Generating volcano plot...")
    plot_volcano(results_df, OUTPUT_DIR / "figure_volcano")
    
    print("\n[4/7] Generating confidence interval plot...")
    plot_top_correlations_with_ci(results_df, bootstrap_df, OUTPUT_DIR / "figure_ci")
    
    print("\n[5/7] Generating scatter plots...")
    plot_scatter_top_correlations(merged_df, results_df, OUTPUT_DIR / "figure_scatter")
    
    print("\n[6/7] Generating subset comparison...")
    plot_subset_comparison(results_df, OUTPUT_DIR / "figure_subset_comparison")
    
    print("\n[7/7] Generating effect size summary...")
    plot_effect_size_summary(results_df, OUTPUT_DIR / "figure_effect_size")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nFigures generated:")
    print(f"  - figure_heatmap.png/svg: Correlation heatmap")
    print(f"  - figure_volcano.png/svg: Volcano plot")
    print(f"  - figure_ci.png/svg: Top correlations with bootstrap CIs")
    print(f"  - figure_scatter.png/svg: Scatter plots for top correlations")
    print(f"  - figure_subset_comparison.png/svg: Experimental vs HMM comparison")
    print(f"  - figure_effect_size.png/svg: Effect size distribution")
    
    print(f"\n📁 Figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
