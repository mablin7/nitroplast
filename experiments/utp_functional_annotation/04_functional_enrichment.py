#!/usr/bin/env python3
"""
04_functional_enrichment.py - Test for functional enrichment in uTP proteins

This script:
1. Compares COG category distributions between uTP and control proteins
2. Performs Fisher's exact test for each category
3. Applies FDR correction for multiple testing
4. Creates enrichment visualization

Usage:
    uv run python experiments/utp_functional_annotation/04_functional_enrichment.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy import stats
from statsmodels.stats.multitest import multipletests

# =============================================================================
# Configuration
# =============================================================================

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']
rcParams['font.size'] = 9
rcParams['axes.linewidth'] = 1.0
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['legend.frameon'] = False

COLORS = {
    'primary': '#2E4057',
    'secondary': '#048A81',
    'accent': '#E85D04',
    'light': '#90BE6D',
    'highlight': '#F9C74F',
    'enriched': '#E85D04',
    'depleted': '#048A81',
}

COG_DESCRIPTIONS = {
    'A': 'RNA processing',
    'B': 'Chromatin structure',
    'C': 'Energy production',
    'D': 'Cell cycle',
    'E': 'Amino acid metabolism',
    'F': 'Nucleotide metabolism',
    'G': 'Carbohydrate metabolism',
    'H': 'Coenzyme metabolism',
    'I': 'Lipid metabolism',
    'J': 'Translation',
    'K': 'Transcription',
    'L': 'Replication/repair',
    'M': 'Cell wall/membrane',
    'N': 'Cell motility',
    'O': 'PTM/protein turnover',
    'P': 'Inorganic ion transport',
    'Q': 'Secondary metabolites',
    'R': 'General function',
    'S': 'Function unknown',
    'T': 'Signal transduction',
    'U': 'Trafficking/secretion',
    'V': 'Defense mechanisms',
    'W': 'Extracellular',
    'X': 'Mobilome',
    'Y': 'Nuclear structure',
    'Z': 'Cytoskeleton',
}

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"

MERGED_DATA = OUTPUT_DIR / "merged_data.csv"
ENRICHMENT_RESULTS = OUTPUT_DIR / "functional_enrichment.csv"
ENRICHMENT_FIGURE = OUTPUT_DIR / "functional_enrichment"


def fishers_exact_test(a, b, c, d):
    """
    Perform Fisher's exact test.
    
    Contingency table:
              Group1    Group2
    Category     a         b
    Other        c         d
    
    Returns: odds_ratio, p_value
    """
    odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]])
    return odds_ratio, p_value


def compute_confidence_interval(a, b, c, d, alpha=0.05):
    """
    Compute confidence interval for log odds ratio using Woolf's method.
    """
    # Add 0.5 to all cells to avoid division by zero (Haldane-Anscombe correction)
    a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    
    log_or = np.log((a * d) / (b * c))
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    
    z = stats.norm.ppf(1 - alpha/2)
    ci_lower = np.exp(log_or - z * se)
    ci_upper = np.exp(log_or + z * se)
    
    return ci_lower, ci_upper


def main():
    """Main enrichment analysis."""
    
    print("=" * 70)
    print("Functional Enrichment Analysis")
    print("=" * 70)
    
    # Load merged data
    if not MERGED_DATA.exists():
        print(f"ERROR: Merged data not found: {MERGED_DATA}")
        print("Please run 03_parse_annotations.py first")
        return
    
    merged = pd.read_csv(MERGED_DATA, low_memory=False)
    print(f"Loaded {len(merged)} sequences")
    
    # Get group column (might be 'group' or 'group_x' depending on merge)
    group_col = 'group_x' if 'group_x' in merged.columns else 'group'
    
    # Separate groups
    utp = merged[merged[group_col] == 'uTP']
    ctrl = merged[merged[group_col] == 'Control']
    
    n_utp = len(utp)
    n_ctrl = len(ctrl)
    
    print(f"\nuTP proteins: {n_utp}")
    print(f"Control proteins: {n_ctrl}")
    
    # Count COG categories
    utp_cogs = utp['primary_cog'].value_counts()
    ctrl_cogs = ctrl['primary_cog'].value_counts()
    
    all_cogs = set(utp_cogs.index) | set(ctrl_cogs.index)
    all_cogs = [c for c in all_cogs if pd.notna(c)]
    
    print(f"\nFound {len(all_cogs)} COG categories")
    
    # Perform enrichment tests
    print("\n--- Fisher's Exact Tests ---")
    results = []
    
    for cog in all_cogs:
        # Counts
        a = utp_cogs.get(cog, 0)  # uTP with this COG
        b = ctrl_cogs.get(cog, 0)  # Control with this COG
        c = n_utp - a  # uTP without this COG
        d = n_ctrl - b  # Control without this COG
        
        # Fisher's exact test
        odds_ratio, p_value = fishers_exact_test(a, b, c, d)
        
        # Confidence interval
        ci_lower, ci_upper = compute_confidence_interval(a, b, c, d)
        
        # Percentages
        pct_utp = 100 * a / n_utp if n_utp > 0 else 0
        pct_ctrl = 100 * b / n_ctrl if n_ctrl > 0 else 0
        
        results.append({
            'COG': cog,
            'Description': COG_DESCRIPTIONS.get(cog, 'Unknown'),
            'n_uTP': a,
            'n_Control': b,
            'pct_uTP': pct_utp,
            'pct_Control': pct_ctrl,
            'odds_ratio': odds_ratio,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
        })
    
    results_df = pd.DataFrame(results)
    
    # Multiple testing correction (FDR)
    _, pvals_corrected, _, _ = multipletests(
        results_df['p_value'], 
        method='fdr_bh', 
        alpha=0.05
    )
    results_df['p_adjusted'] = pvals_corrected
    results_df['significant'] = results_df['p_adjusted'] < 0.05
    
    # Log2 odds ratio for visualization
    results_df['log2_OR'] = np.log2(results_df['odds_ratio'].replace(0, np.nan))
    
    # Sort by odds ratio
    results_df = results_df.sort_values('odds_ratio', ascending=False)
    
    # Save results
    results_df.to_csv(ENRICHMENT_RESULTS, index=False)
    print(f"\nSaved enrichment results to: {ENRICHMENT_RESULTS}")
    
    # Print significant results
    print("\n--- Significantly Enriched/Depleted Categories (FDR < 0.05) ---")
    sig_results = results_df[results_df['significant']]
    
    if len(sig_results) > 0:
        for _, row in sig_results.iterrows():
            direction = "ENRICHED" if row['odds_ratio'] > 1 else "DEPLETED"
            print(f"{row['COG']} ({row['Description']}): OR={row['odds_ratio']:.2f}, "
                  f"p_adj={row['p_adjusted']:.4f}, {direction}")
            print(f"    uTP: {row['n_uTP']} ({row['pct_uTP']:.1f}%), "
                  f"Control: {row['n_Control']} ({row['pct_Control']:.1f}%)")
    else:
        print("No categories reached significance after FDR correction.")
    
    # Print all results
    print("\n--- All Categories (sorted by OR) ---")
    for _, row in results_df.iterrows():
        sig_marker = "*" if row['significant'] else " "
        print(f"{sig_marker} {row['COG']} ({row['Description'][:20]}): "
              f"OR={row['odds_ratio']:.2f} [{row['ci_lower']:.2f}-{row['ci_upper']:.2f}], "
              f"p={row['p_value']:.4f}")
    
    # Create visualization
    create_enrichment_figure(results_df)
    
    return results_df


def create_enrichment_figure(results_df):
    """Create forest plot of enrichment results."""
    
    # Filter to categories with at least some counts
    plot_df = results_df[
        (results_df['n_uTP'] + results_df['n_Control']) >= 5
    ].copy()
    
    # Sort by log2 OR
    plot_df = plot_df.sort_values('log2_OR')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [2, 1]})
    
    # Panel A: Forest plot
    ax = axes[0]
    y_positions = range(len(plot_df))
    
    # Plot odds ratios
    colors = [COLORS['enriched'] if or_val > 1 else COLORS['depleted'] 
              for or_val in plot_df['odds_ratio']]
    
    ax.scatter(plot_df['log2_OR'], y_positions, c=colors, s=50, zorder=3)
    
    # Add confidence intervals
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ci_lower = np.log2(row['ci_lower']) if row['ci_lower'] > 0 else -5
        ci_upper = np.log2(row['ci_upper']) if row['ci_upper'] < np.inf else 5
        
        # Clip to reasonable range
        ci_lower = max(ci_lower, -5)
        ci_upper = min(ci_upper, 5)
        
        ax.plot([ci_lower, ci_upper], [i, i], 
                c=colors[i], alpha=0.5, linewidth=2)
    
    # Add significance markers
    for i, (_, row) in enumerate(plot_df.iterrows()):
        if row['significant']:
            ax.scatter(plot_df['log2_OR'].iloc[i], i, 
                      marker='*', s=100, c='gold', zorder=4, edgecolors='black')
    
    # Reference line at OR=1
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    # Labels
    labels = [f"{row['COG']}: {row['Description']}" for _, row in plot_df.iterrows()]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel('log₂(Odds Ratio)')
    ax.set_title('A. Functional Enrichment in uTP Proteins\n(* = FDR < 0.05)', 
                 fontweight='bold', loc='left')
    ax.set_xlim(-4, 4)
    
    # Panel B: Percentage comparison
    ax = axes[1]
    
    width = 0.35
    y = np.arange(len(plot_df))
    
    ax.barh(y - width/2, plot_df['pct_uTP'], width, 
            label='uTP', color=COLORS['accent'], alpha=0.8)
    ax.barh(y + width/2, plot_df['pct_Control'], width, 
            label='Control', color=COLORS['primary'], alpha=0.8)
    
    ax.set_yticks([])
    ax.set_xlabel('Percentage (%)')
    ax.set_title('B. Category Frequencies', fontweight='bold', loc='left')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    # Save
    fig.savefig(f"{ENRICHMENT_FIGURE}.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(f"{ENRICHMENT_FIGURE}.svg", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\nSaved enrichment figure to: {ENRICHMENT_FIGURE}.png")


if __name__ == "__main__":
    main()
