#!/usr/bin/env python3
"""
05_within_category_analysis.py - Within-category biophysical property analysis

This is the CRITICAL test for the main hypothesis:

If uTP biophysical properties are explained by functional enrichment:
    → Properties should be SIMILAR within the same functional category
    → Effect sizes should approach zero when comparing uTP vs Control
      within each COG category

If uTP properties are intrinsic (selected for import):
    → Properties should DIFFER even within the same functional category
    → Effect sizes should remain significant

Usage:
    uv run python experiments/utp_functional_annotation/05_within_category_analysis.py
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
}

# Key properties to analyze (from utp_presence_classifier results)
KEY_PROPERTIES = [
    'fraction_coil',
    'isoelectric_point', 
    'instability_index',
    'gravy',
    'fraction_helix',
    'molecular_weight',
]

# Minimum samples per category per group for reliable statistics
MIN_SAMPLES = 10

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
WITHIN_CATEGORY_RESULTS = OUTPUT_DIR / "within_category_results.csv"
META_ANALYSIS_RESULTS = OUTPUT_DIR / "meta_analysis_results.csv"
WITHIN_CATEGORY_FIGURE = OUTPUT_DIR / "within_category_analysis"


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    if pooled_std == 0:
        return 0
    
    return (group1.mean() - group2.mean()) / pooled_std


def compute_overall_effect(merged, group_col, property_name):
    """Compute overall effect size (uTP vs Control, ignoring function)."""
    utp_vals = merged[merged[group_col] == 'uTP'][property_name].dropna()
    ctrl_vals = merged[merged[group_col] == 'Control'][property_name].dropna()
    
    if len(utp_vals) < 2 or len(ctrl_vals) < 2:
        return None, None, None
    
    d = cohens_d(utp_vals, ctrl_vals)
    stat, pval = stats.mannwhitneyu(utp_vals, ctrl_vals, alternative='two-sided')
    
    return d, pval, (utp_vals.mean(), ctrl_vals.mean())


def main():
    """Main within-category analysis."""
    
    print("=" * 70)
    print("Within-Category Biophysical Property Analysis")
    print("=" * 70)
    print("\nTESTING: Do uTP biophysical properties persist within functional categories?")
    
    # Load data
    if not MERGED_DATA.exists():
        print(f"ERROR: Merged data not found: {MERGED_DATA}")
        return
    
    merged = pd.read_csv(MERGED_DATA, low_memory=False)
    print(f"Loaded {len(merged)} sequences")
    
    # Get group column
    group_col = 'group_x' if 'group_x' in merged.columns else 'group'
    
    # Filter to sequences with COG annotation
    merged_with_cog = merged[merged['primary_cog'].notna()].copy()
    print(f"Sequences with COG annotation: {len(merged_with_cog)}")
    
    # =========================================================================
    # Step 1: Overall effects (ignoring function)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Overall Effects (Ignoring Function)")
    print("=" * 70)
    
    overall_results = []
    for prop in KEY_PROPERTIES:
        d, pval, means = compute_overall_effect(merged_with_cog, group_col, prop)
        if d is not None:
            overall_results.append({
                'property': prop,
                'cohens_d': d,
                'p_value': pval,
                'utp_mean': means[0],
                'ctrl_mean': means[1],
            })
            print(f"{prop}: d={d:+.3f}, p={pval:.4f}")
    
    overall_df = pd.DataFrame(overall_results)
    
    # =========================================================================
    # Step 2: Within-category effects
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Within-Category Effects")
    print("=" * 70)
    
    # Find categories with enough samples in both groups
    category_counts = merged_with_cog.groupby(['primary_cog', group_col]).size().unstack(fill_value=0)
    
    valid_categories = []
    for cog in category_counts.index:
        n_utp = category_counts.loc[cog, 'uTP'] if 'uTP' in category_counts.columns else 0
        n_ctrl = category_counts.loc[cog, 'Control'] if 'Control' in category_counts.columns else 0
        
        if n_utp >= MIN_SAMPLES and n_ctrl >= MIN_SAMPLES:
            valid_categories.append(cog)
            print(f"  {cog} ({COG_DESCRIPTIONS.get(cog, '?')}): uTP={n_utp}, Control={n_ctrl}")
    
    print(f"\nFound {len(valid_categories)} categories with n≥{MIN_SAMPLES} in both groups")
    
    # Analyze each category
    within_results = []
    
    for cog in valid_categories:
        cog_data = merged_with_cog[merged_with_cog['primary_cog'] == cog]
        utp_data = cog_data[cog_data[group_col] == 'uTP']
        ctrl_data = cog_data[cog_data[group_col] == 'Control']
        
        for prop in KEY_PROPERTIES:
            utp_vals = utp_data[prop].dropna()
            ctrl_vals = ctrl_data[prop].dropna()
            
            if len(utp_vals) < MIN_SAMPLES or len(ctrl_vals) < MIN_SAMPLES:
                continue
            
            d = cohens_d(utp_vals, ctrl_vals)
            stat, pval = stats.mannwhitneyu(utp_vals, ctrl_vals, alternative='two-sided')
            
            within_results.append({
                'COG': cog,
                'COG_description': COG_DESCRIPTIONS.get(cog, 'Unknown'),
                'property': prop,
                'n_uTP': len(utp_vals),
                'n_Control': len(ctrl_vals),
                'utp_mean': utp_vals.mean(),
                'ctrl_mean': ctrl_vals.mean(),
                'cohens_d': d,
                'p_value': pval,
            })
    
    within_df = pd.DataFrame(within_results)
    
    # FDR correction within each property
    if len(within_df) > 0:
        within_df['p_adjusted'] = np.nan
        for prop in KEY_PROPERTIES:
            mask = within_df['property'] == prop
            if mask.sum() > 0:
                _, pvals_adj, _, _ = multipletests(
                    within_df.loc[mask, 'p_value'], 
                    method='fdr_bh'
                )
                within_df.loc[mask, 'p_adjusted'] = pvals_adj
        
        within_df['significant'] = within_df['p_adjusted'] < 0.05
    
    # Save results
    within_df.to_csv(WITHIN_CATEGORY_RESULTS, index=False)
    print(f"\nSaved within-category results to: {WITHIN_CATEGORY_RESULTS}")
    
    # =========================================================================
    # Step 3: Meta-analysis across categories
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Meta-Analysis Across Categories")
    print("=" * 70)
    
    meta_results = []
    
    for prop in KEY_PROPERTIES:
        prop_data = within_df[within_df['property'] == prop]
        
        if len(prop_data) < 2:
            continue
        
        # Overall effect (from Step 1)
        overall_d = overall_df[overall_df['property'] == prop]['cohens_d'].values[0]
        
        # Average within-category effect
        avg_within_d = prop_data['cohens_d'].mean()
        std_within_d = prop_data['cohens_d'].std()
        
        # Weighted average (by sample size)
        weights = prop_data['n_uTP'] + prop_data['n_Control']
        weighted_avg_d = np.average(prop_data['cohens_d'], weights=weights)
        
        # How many categories show same direction as overall?
        same_direction = (np.sign(prop_data['cohens_d']) == np.sign(overall_d)).sum()
        
        # How many are significant?
        n_significant = prop_data['significant'].sum()
        
        # Percentage of effect explained by function
        # Report actual value (can be negative if within-category effect is larger than overall)
        if abs(overall_d) > 0:
            # If signs differ, report as 100% (function explains direction)
            if np.sign(weighted_avg_d) != np.sign(overall_d) and weighted_avg_d != 0:
                pct_explained = 100.0
                pct_explained_raw = 100.0
            else:
                pct_explained_raw = (1 - abs(weighted_avg_d) / abs(overall_d)) * 100
                pct_explained = max(0, pct_explained_raw)  # Floored version for summary
        else:
            pct_explained = np.nan
            pct_explained_raw = np.nan
        
        # I² heterogeneity statistic
        # I² = (Q - df) / Q * 100, where Q is Cochran's Q
        k = len(prop_data)
        if k >= 2:
            # Standard error of each effect size (approximate: SE ≈ sqrt(1/n_uTP + 1/n_ctrl))
            prop_data = prop_data.copy()
            prop_data['se'] = np.sqrt(1/prop_data['n_uTP'] + 1/prop_data['n_Control'])
            prop_data['weight'] = 1 / (prop_data['se'] ** 2)
            
            # Fixed-effect weighted mean
            weighted_mean = np.average(prop_data['cohens_d'], weights=prop_data['weight'])
            
            # Cochran's Q
            Q = np.sum(prop_data['weight'] * (prop_data['cohens_d'] - weighted_mean) ** 2)
            df = k - 1
            
            # I² 
            if Q > df:
                I_squared = ((Q - df) / Q) * 100
            else:
                I_squared = 0
            
            # p-value for heterogeneity (chi-squared test)
            hetero_p = 1 - stats.chi2.cdf(Q, df)
        else:
            I_squared = np.nan
            hetero_p = np.nan
        
        meta_results.append({
            'property': prop,
            'overall_d': overall_d,
            'avg_within_d': avg_within_d,
            'std_within_d': std_within_d,
            'weighted_avg_within_d': weighted_avg_d,
            'n_categories': len(prop_data),
            'n_same_direction': same_direction,
            'n_significant': n_significant,
            'pct_explained_by_function': pct_explained,
            'pct_explained_raw': pct_explained_raw,
            'I_squared': I_squared,
            'heterogeneity_p': hetero_p,
        })
        
        print(f"\n{prop}:")
        print(f"  Overall effect: d = {overall_d:+.3f}")
        print(f"  Within-category effect: d = {weighted_avg_d:+.3f} (weighted avg)")
        print(f"  Effect consistency: {same_direction}/{len(prop_data)} categories same direction")
        print(f"  Significant within-category: {n_significant}/{len(prop_data)}")
        print(f"  % effect explained by function: {pct_explained_raw:.1f}% (raw), {pct_explained:.1f}% (floored)")
        print(f"  Heterogeneity: I²={I_squared:.1f}%, p={hetero_p:.4f}")
    
    meta_df = pd.DataFrame(meta_results)
    meta_df.to_csv(META_ANALYSIS_RESULTS, index=False)
    print(f"\nSaved meta-analysis results to: {META_ANALYSIS_RESULTS}")
    
    # =========================================================================
    # Step 4: Sensitivity Analysis - Excluding "Function Unknown" (S)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Sensitivity Analysis (Excluding 'S' = Function Unknown)")
    print("=" * 70)
    
    # Repeat meta-analysis excluding category S
    within_df_no_S = within_df[within_df['COG'] != 'S']
    
    sensitivity_results = []
    for prop in KEY_PROPERTIES:
        prop_data = within_df_no_S[within_df_no_S['property'] == prop]
        
        if len(prop_data) < 2:
            continue
        
        overall_d = overall_df[overall_df['property'] == prop]['cohens_d'].values[0]
        
        weights = prop_data['n_uTP'] + prop_data['n_Control']
        weighted_avg_d = np.average(prop_data['cohens_d'], weights=weights)
        same_direction = (np.sign(prop_data['cohens_d']) == np.sign(overall_d)).sum()
        n_significant = prop_data['significant'].sum()
        
        if abs(overall_d) > 0:
            pct_explained_raw = (1 - abs(weighted_avg_d) / abs(overall_d)) * 100
            pct_explained = max(0, pct_explained_raw)
        else:
            pct_explained = np.nan
            pct_explained_raw = np.nan
        
        sensitivity_results.append({
            'property': prop,
            'overall_d': overall_d,
            'weighted_avg_within_d_excl_S': weighted_avg_d,
            'n_categories_excl_S': len(prop_data),
            'n_same_direction_excl_S': same_direction,
            'n_significant_excl_S': n_significant,
            'pct_explained_excl_S': pct_explained,
            'pct_explained_raw_excl_S': pct_explained_raw,
        })
        
        print(f"\n{prop}:")
        print(f"  Within-category (excl S): d = {weighted_avg_d:+.3f}")
        print(f"  Categories: {same_direction}/{len(prop_data)} same direction")
        print(f"  % explained by function: {pct_explained_raw:.1f}%")
    
    sensitivity_df = pd.DataFrame(sensitivity_results)
    
    # Merge with main meta results
    meta_df = meta_df.merge(sensitivity_df[['property', 'weighted_avg_within_d_excl_S', 
                                            'n_categories_excl_S', 'pct_explained_excl_S',
                                            'pct_explained_raw_excl_S']], 
                            on='property', how='left')
    meta_df.to_csv(META_ANALYSIS_RESULTS, index=False)
    
    # =========================================================================
    # Step 5: Interpretation
    # =========================================================================
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    # Average effect reduction across properties
    avg_pct_explained = meta_df['pct_explained_by_function'].mean()
    avg_pct_explained_excl_S = meta_df['pct_explained_excl_S'].mean()
    
    print(f"\nAverage % of effect explained by functional enrichment: {avg_pct_explained:.1f}%")
    print(f"Average % explained (excluding 'Function Unknown'): {avg_pct_explained_excl_S:.1f}%")
    
    # Report heterogeneity
    avg_I_squared = meta_df['I_squared'].mean()
    print(f"\nAverage heterogeneity (I²): {avg_I_squared:.1f}%")
    if avg_I_squared < 25:
        print("  → Low heterogeneity: effects are consistent across categories")
    elif avg_I_squared < 75:
        print("  → Moderate heterogeneity: some variation across categories")
    else:
        print("  → High heterogeneity: substantial variation across categories")
    
    if avg_pct_explained > 75:
        print("\n→ HYPOTHESIS A SUPPORTED: Most of the effect is explained by functional enrichment")
        print("  The biophysical properties appear to be a byproduct of which proteins carry uTPs,")
        print("  not a selection pressure for the import process itself.")
    elif avg_pct_explained < 25:
        print("\n→ HYPOTHESIS B SUPPORTED: The effect persists despite controlling for function")
        print("  The biophysical properties appear to be intrinsic to uTP proteins,")
        print("  suggesting selection for import-compatible properties.")
    else:
        print("\n→ MIXED RESULTS: Partial confounding by function")
        print("  Both functional enrichment and uTP-specific selection may contribute.")
    
    # Create visualization
    create_within_category_figure(within_df, meta_df, overall_df)
    
    return within_df, meta_df


def create_within_category_figure(within_df, meta_df, overall_df):
    """Create visualization of within-category analysis."""
    
    if len(within_df) == 0:
        print("No within-category results to plot")
        return
    
    # Focus on key properties
    key_props = ['fraction_coil', 'isoelectric_point', 'instability_index']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    for idx, prop in enumerate(key_props):
        ax = axes[idx]
        
        prop_data = within_df[within_df['property'] == prop].sort_values('cohens_d')
        
        if len(prop_data) == 0:
            continue
        
        # Get overall effect
        overall_d = overall_df[overall_df['property'] == prop]['cohens_d'].values
        overall_d = overall_d[0] if len(overall_d) > 0 else 0
        
        # Forest plot
        y_pos = range(len(prop_data))
        
        # Color by significance
        colors = [COLORS['accent'] if sig else COLORS['primary'] 
                  for sig in prop_data['significant']]
        
        ax.scatter(prop_data['cohens_d'], y_pos, c=colors, s=60, zorder=3)
        
        # Reference lines
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax.axvline(x=overall_d, color=COLORS['accent'], linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'Overall d={overall_d:.2f}')
        
        # Weighted average within-category
        meta_row = meta_df[meta_df['property'] == prop]
        if len(meta_row) > 0:
            weighted_d = meta_row['weighted_avg_within_d'].values[0]
            ax.axvline(x=weighted_d, color=COLORS['secondary'], linestyle=':', 
                      linewidth=2, alpha=0.7, label=f'Within-cat d={weighted_d:.2f}')
        
        # Labels
        labels = [f"{row['COG']}: {row['COG_description'][:15]}" 
                  for _, row in prop_data.iterrows()]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Cohen's d (uTP - Control)")
        ax.set_title(f"{prop.replace('_', ' ').title()}", fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim(-1.5, 1.5)
    
    plt.suptitle("Within-Category Effect Sizes\n(Orange = significant after FDR correction)",
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    fig.savefig(f"{WITHIN_CATEGORY_FIGURE}.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(f"{WITHIN_CATEGORY_FIGURE}.svg", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\nSaved within-category figure to: {WITHIN_CATEGORY_FIGURE}.png")


if __name__ == "__main__":
    main()
