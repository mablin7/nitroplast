#!/usr/bin/env python3
"""
06_ancova_analysis.py - ANCOVA and variance partitioning

This script performs rigorous statistical control for function using:
1. ANCOVA: property ~ uTP_status + COG_category
2. Variance partitioning: how much variance is explained by function vs uTP?
3. Propensity-score matching by function

Usage:
    uv run python experiments/utp_functional_annotation/06_ancova_analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

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
    'function': '#048A81',
    'utp': '#E85D04',
    'shared': '#F9C74F',
    'residual': '#cccccc',
}

KEY_PROPERTIES = [
    'fraction_coil',
    'isoelectric_point', 
    'instability_index',
    'gravy',
]

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"

MERGED_DATA = OUTPUT_DIR / "merged_data.csv"
ANCOVA_RESULTS = OUTPUT_DIR / "ancova_results.csv"
VARIANCE_PARTITIONING = OUTPUT_DIR / "variance_partitioning.csv"
MATCHED_PAIRS_RESULTS = OUTPUT_DIR / "matched_pairs_results.csv"
ANCOVA_FIGURE = OUTPUT_DIR / "ancova_analysis"


def run_ancova(data, property_name, group_col):
    """
    Run ANCOVA to test effect of uTP status while controlling for COG category.
    
    Model: property ~ is_uTP + COG_category
    """
    # Prepare data
    analysis_data = data[['is_uTP', 'primary_cog', property_name]].dropna().copy()
    
    if len(analysis_data) < 20:
        return None
    
    # Need at least 2 levels of each factor
    if analysis_data['primary_cog'].nunique() < 2:
        return None
    
    try:
        # Type II ANOVA (tests each factor after controlling for others)
        model = ols(f'{property_name} ~ C(is_uTP) + C(primary_cog)', data=analysis_data).fit()
        anova_table = anova_lm(model, typ=2)
        
        # Extract results
        utp_ss = anova_table.loc['C(is_uTP)', 'sum_sq']
        cog_ss = anova_table.loc['C(primary_cog)', 'sum_sq']
        resid_ss = anova_table.loc['Residual', 'sum_sq']
        total_ss = utp_ss + cog_ss + resid_ss
        
        return {
            'property': property_name,
            'utp_F': anova_table.loc['C(is_uTP)', 'F'],
            'utp_p': anova_table.loc['C(is_uTP)', 'PR(>F)'],
            'cog_F': anova_table.loc['C(primary_cog)', 'F'],
            'cog_p': anova_table.loc['C(primary_cog)', 'PR(>F)'],
            'utp_ss': utp_ss,
            'cog_ss': cog_ss,
            'resid_ss': resid_ss,
            'total_ss': total_ss,
            'utp_pct_var': 100 * utp_ss / total_ss,
            'cog_pct_var': 100 * cog_ss / total_ss,
            'resid_pct_var': 100 * resid_ss / total_ss,
            'r_squared': model.rsquared,
            'n_samples': len(analysis_data),
        }
    except Exception as e:
        print(f"ANCOVA failed for {property_name}: {e}")
        return None


def variance_partitioning(data, property_name, group_col):
    """
    Partition variance to determine unique and shared contributions.
    
    Models:
    1. property ~ is_uTP                    → R²_uTP_only
    2. property ~ COG_category              → R²_COG_only  
    3. property ~ is_uTP + COG_category     → R²_full
    
    Unique variance:
    - uTP unique = R²_full - R²_COG_only
    - COG unique = R²_full - R²_uTP_only
    - Shared = R²_full - uTP_unique - COG_unique
    """
    analysis_data = data[['is_uTP', 'primary_cog', property_name]].dropna().copy()
    
    if len(analysis_data) < 20:
        return None
    
    try:
        # Model 1: uTP only
        model_utp = ols(f'{property_name} ~ C(is_uTP)', data=analysis_data).fit()
        r2_utp_only = model_utp.rsquared
        
        # Model 2: COG only
        model_cog = ols(f'{property_name} ~ C(primary_cog)', data=analysis_data).fit()
        r2_cog_only = model_cog.rsquared
        
        # Model 3: Full model
        model_full = ols(f'{property_name} ~ C(is_uTP) + C(primary_cog)', data=analysis_data).fit()
        r2_full = model_full.rsquared
        
        # Calculate unique and shared variance
        utp_unique = r2_full - r2_cog_only
        cog_unique = r2_full - r2_utp_only
        shared = r2_utp_only + r2_cog_only - r2_full
        
        # Ensure non-negative (can happen with collinearity)
        utp_unique = max(0, utp_unique)
        cog_unique = max(0, cog_unique)
        shared = max(0, shared)
        
        return {
            'property': property_name,
            'r2_utp_only': r2_utp_only,
            'r2_cog_only': r2_cog_only,
            'r2_full': r2_full,
            'utp_unique': utp_unique,
            'cog_unique': cog_unique,
            'shared': shared,
            'unexplained': 1 - r2_full,
            'utp_unique_pct': 100 * utp_unique,
            'cog_unique_pct': 100 * cog_unique,
            'shared_pct': 100 * shared,
            'unexplained_pct': 100 * (1 - r2_full),
        }
    except Exception as e:
        print(f"Variance partitioning failed for {property_name}: {e}")
        return None


def matched_pairs_analysis(data, property_name, group_col):
    """
    Match uTP proteins to controls by COG category, then compare properties.
    
    This is the most stringent test: comparing proteins with identical functions.
    """
    analysis_data = data[['is_uTP', 'primary_cog', property_name, 'original_id']].dropna().copy()
    
    # Find categories with both uTP and control proteins
    category_counts = analysis_data.groupby(['primary_cog', 'is_uTP']).size().unstack(fill_value=0)
    
    matched_pairs = []
    
    for cog in category_counts.index:
        n_utp = category_counts.loc[cog, 1] if 1 in category_counts.columns else 0
        n_ctrl = category_counts.loc[cog, 0] if 0 in category_counts.columns else 0
        
        if n_utp > 0 and n_ctrl > 0:
            utp_vals = analysis_data[(analysis_data['primary_cog'] == cog) & 
                                     (analysis_data['is_uTP'] == 1)][property_name].values
            ctrl_vals = analysis_data[(analysis_data['primary_cog'] == cog) & 
                                      (analysis_data['is_uTP'] == 0)][property_name].values
            
            # Pair by random sampling (with replacement if necessary)
            n_pairs = min(len(utp_vals), len(ctrl_vals))
            
            for i in range(n_pairs):
                matched_pairs.append({
                    'cog': cog,
                    'utp_value': utp_vals[i % len(utp_vals)],
                    'ctrl_value': ctrl_vals[i % len(ctrl_vals)],
                })
    
    if len(matched_pairs) == 0:
        return None
    
    pairs_df = pd.DataFrame(matched_pairs)
    pairs_df['difference'] = pairs_df['utp_value'] - pairs_df['ctrl_value']
    
    # Paired t-test (or Wilcoxon signed-rank)
    stat, pval = stats.wilcoxon(pairs_df['difference'])
    
    # Effect size (mean difference / std)
    mean_diff = pairs_df['difference'].mean()
    std_diff = pairs_df['difference'].std()
    effect_size = mean_diff / std_diff if std_diff > 0 else 0
    
    return {
        'property': property_name,
        'n_pairs': len(pairs_df),
        'n_categories': pairs_df['cog'].nunique(),
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'effect_size': effect_size,
        'wilcoxon_stat': stat,
        'wilcoxon_p': pval,
        'pct_positive_diff': 100 * (pairs_df['difference'] > 0).mean(),
    }


def main():
    """Main ANCOVA analysis."""
    
    print("=" * 70)
    print("ANCOVA and Variance Partitioning Analysis")
    print("=" * 70)
    
    # Load data
    if not MERGED_DATA.exists():
        print(f"ERROR: Merged data not found: {MERGED_DATA}")
        return
    
    merged = pd.read_csv(MERGED_DATA, low_memory=False)
    print(f"Loaded {len(merged)} sequences")
    
    # Get group column and create binary indicator
    group_col = 'group_x' if 'group_x' in merged.columns else 'group'
    merged['is_uTP'] = (merged[group_col] == 'uTP').astype(int)
    
    # Filter to sequences with COG annotation
    merged_with_cog = merged[merged['primary_cog'].notna()].copy()
    print(f"Sequences with COG annotation: {len(merged_with_cog)}")
    
    # =========================================================================
    # ANCOVA Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANCOVA: property ~ uTP_status + COG_category")
    print("=" * 70)
    
    ancova_results = []
    for prop in KEY_PROPERTIES:
        result = run_ancova(merged_with_cog, prop, group_col)
        if result:
            ancova_results.append(result)
            print(f"\n{prop}:")
            print(f"  uTP effect: F={result['utp_F']:.2f}, p={result['utp_p']:.4f}, "
                  f"var explained={result['utp_pct_var']:.1f}%")
            print(f"  COG effect: F={result['cog_F']:.2f}, p={result['cog_p']:.4f}, "
                  f"var explained={result['cog_pct_var']:.1f}%")
    
    if ancova_results:
        ancova_df = pd.DataFrame(ancova_results)
        ancova_df.to_csv(ANCOVA_RESULTS, index=False)
        print(f"\nSaved ANCOVA results to: {ANCOVA_RESULTS}")
    
    # =========================================================================
    # Variance Partitioning
    # =========================================================================
    print("\n" + "=" * 70)
    print("Variance Partitioning")
    print("=" * 70)
    
    var_results = []
    for prop in KEY_PROPERTIES:
        result = variance_partitioning(merged_with_cog, prop, group_col)
        if result:
            var_results.append(result)
            print(f"\n{prop}:")
            print(f"  uTP unique: {result['utp_unique_pct']:.1f}%")
            print(f"  COG unique: {result['cog_unique_pct']:.1f}%")
            print(f"  Shared: {result['shared_pct']:.1f}%")
            print(f"  Unexplained: {result['unexplained_pct']:.1f}%")
    
    if var_results:
        var_df = pd.DataFrame(var_results)
        var_df.to_csv(VARIANCE_PARTITIONING, index=False)
        print(f"\nSaved variance partitioning to: {VARIANCE_PARTITIONING}")
    
    # =========================================================================
    # Matched Pairs Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("Matched Pairs Analysis (Matched by COG Category)")
    print("=" * 70)
    
    matched_results = []
    for prop in KEY_PROPERTIES:
        result = matched_pairs_analysis(merged_with_cog, prop, group_col)
        if result:
            matched_results.append(result)
            sig_marker = "*" if result['wilcoxon_p'] < 0.05 else ""
            print(f"\n{prop}:{sig_marker}")
            print(f"  {result['n_pairs']} pairs across {result['n_categories']} categories")
            print(f"  Mean difference: {result['mean_difference']:+.4f}")
            print(f"  Effect size: {result['effect_size']:+.3f}")
            print(f"  Wilcoxon p-value: {result['wilcoxon_p']:.4f}")
    
    if matched_results:
        matched_df = pd.DataFrame(matched_results)
        matched_df.to_csv(MATCHED_PAIRS_RESULTS, index=False)
        print(f"\nSaved matched pairs results to: {MATCHED_PAIRS_RESULTS}")
    
    # =========================================================================
    # Interpretation
    # =========================================================================
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if var_results:
        avg_utp_unique = np.mean([r['utp_unique_pct'] for r in var_results])
        avg_cog_unique = np.mean([r['cog_unique_pct'] for r in var_results])
        avg_shared = np.mean([r['shared_pct'] for r in var_results])
        
        print(f"\nAverage variance explained across properties:")
        print(f"  uTP unique: {avg_utp_unique:.1f}%")
        print(f"  Function unique: {avg_cog_unique:.1f}%")
        print(f"  Shared: {avg_shared:.1f}%")
        
        if avg_utp_unique > avg_shared:
            print("\n→ uTP explains MORE unique variance than shared with function")
            print("  This supports HYPOTHESIS B: properties are selected for import")
        else:
            print("\n→ Shared variance is larger than uTP unique variance")
            print("  Function and uTP status are confounded")
    
    if matched_results:
        n_significant = sum(1 for r in matched_results if r['wilcoxon_p'] < 0.05)
        print(f"\nMatched pairs (controlling for function): {n_significant}/{len(matched_results)} significant")
        
        if n_significant >= len(matched_results) / 2:
            print("→ Properties differ even when comparing same-function proteins")
            print("  Supports HYPOTHESIS B")
        else:
            print("→ Most properties not significantly different after matching")
            print("  Supports HYPOTHESIS A")
    
    # Create visualization
    if var_results:
        create_variance_figure(var_results)
    
    return ancova_results, var_results, matched_results


def create_variance_figure(var_results):
    """Create variance partitioning visualization."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Stacked bar chart
    ax = axes[0]
    
    properties = [r['property'] for r in var_results]
    utp_unique = [r['utp_unique_pct'] for r in var_results]
    cog_unique = [r['cog_unique_pct'] for r in var_results]
    shared = [r['shared_pct'] for r in var_results]
    unexplained = [r['unexplained_pct'] for r in var_results]
    
    x = np.arange(len(properties))
    width = 0.6
    
    ax.bar(x, utp_unique, width, label='uTP unique', color=COLORS['utp'])
    ax.bar(x, cog_unique, width, bottom=utp_unique, label='Function unique', color=COLORS['function'])
    ax.bar(x, shared, width, bottom=np.array(utp_unique) + np.array(cog_unique), 
           label='Shared', color=COLORS['shared'])
    ax.bar(x, unexplained, width, 
           bottom=np.array(utp_unique) + np.array(cog_unique) + np.array(shared),
           label='Unexplained', color=COLORS['residual'])
    
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('_', '\n') for p in properties], fontsize=8)
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('A. Variance Partitioning', fontweight='bold', loc='left')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 100)
    
    # Panel B: Summary pie chart (average across properties)
    ax = axes[1]
    
    avg_utp = np.mean(utp_unique)
    avg_cog = np.mean(cog_unique)
    avg_shared = np.mean(shared)
    avg_unexp = np.mean(unexplained)
    
    sizes = [avg_utp, avg_cog, avg_shared, avg_unexp]
    labels = [f'uTP unique\n({avg_utp:.1f}%)', 
              f'Function unique\n({avg_cog:.1f}%)',
              f'Shared\n({avg_shared:.1f}%)',
              f'Unexplained\n({avg_unexp:.1f}%)']
    colors = [COLORS['utp'], COLORS['function'], COLORS['shared'], COLORS['residual']]
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='',
                                       startangle=90, explode=(0.05, 0.05, 0, 0))
    
    ax.set_title('B. Average Variance Composition', fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    fig.savefig(f"{ANCOVA_FIGURE}.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(f"{ANCOVA_FIGURE}.svg", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\nSaved ANCOVA figure to: {ANCOVA_FIGURE}.png")


if __name__ == "__main__":
    main()
