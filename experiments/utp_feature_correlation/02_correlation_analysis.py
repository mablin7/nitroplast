#!/usr/bin/env python3
"""
02_correlation_analysis.py - Rigorous Correlation Analysis

Tests whether continuous uTP features predict continuous functional outcomes
using state-of-the-art statistical methods:

1. Spearman rank correlations (robust to non-normality)
2. FDR correction for multiple testing (Benjamini-Hochberg)
3. Bootstrap confidence intervals for effect sizes
4. Permutation tests for robustness validation
5. Separate analyses for experimental and HMM-predicted sets

Statistical rigor:
- Pre-specified analysis plan
- Multiple testing correction
- Effect size reporting (not just p-values)
- Bootstrap CIs for robustness
- Permutation-based validation

Usage:
    uv run python experiments/utp_feature_correlation/02_correlation_analysis.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.multitest import fdrcorrection

warnings.filterwarnings("ignore")

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"

# Input files (from previous scripts)
MERGED_DATA = OUTPUT_DIR / "utp_features_with_outcomes.csv"


# =============================================================================
# Configuration
# =============================================================================

# Minimum sample size for valid correlation
MIN_SAMPLE_SIZE = 20

# FDR threshold
FDR_THRESHOLD = 0.05

# Bootstrap parameters
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 0.95

# Permutation test parameters
N_PERMUTATIONS = 1000

# Feature categories to analyze
UTP_FEATURE_PREFIXES = [
    'utp_length', 'utp_molecular_weight', 'utp_gravy', 'utp_isoelectric_point',
    'utp_instability_index', 'utp_aromaticity',
    'utp_aa_', 'utp_group_',
    'utp_net_charge', 'utp_charge_', 'utp_positive_fraction', 'utp_negative_fraction',
    'utp_hydro_', 'utp_disorder_',
    'utp_helix_', 'utp_sheet_',
    'utp_entropy', 'utp_complexity_',
]

OUTCOME_COLUMNS = [
    'logFC_day', 'logFC_night', 'avg_logFC',
    'log_expr_ucyna_day', 'log_expr_ucyna_night', 'log_total_ucyna_expr',
    'log_ucyna_day_night_ratio', 'logFC_consistency',
    'cv_ucyna_day', 'cv_ucyna_night',
]


# =============================================================================
# Statistical Functions
# =============================================================================

def compute_spearman(x: np.ndarray, y: np.ndarray) -> tuple:
    """Compute Spearman correlation with p-value."""
    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < MIN_SAMPLE_SIZE:
        return np.nan, np.nan, 0
    
    rho, pval = spearmanr(x_clean, y_clean)
    return rho, pval, len(x_clean)


def compute_pearson(x: np.ndarray, y: np.ndarray) -> tuple:
    """Compute Pearson correlation with p-value."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < MIN_SAMPLE_SIZE:
        return np.nan, np.nan, 0
    
    r, pval = pearsonr(x_clean, y_clean)
    return r, pval, len(x_clean)


def bootstrap_correlation(x: np.ndarray, y: np.ndarray, n_bootstrap: int = N_BOOTSTRAP,
                         ci: float = BOOTSTRAP_CI) -> dict:
    """Compute bootstrap confidence interval for Spearman correlation."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < MIN_SAMPLE_SIZE:
        return {'ci_lower': np.nan, 'ci_upper': np.nan, 'se': np.nan}
    
    n = len(x_clean)
    boot_rhos = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        x_boot = x_clean[idx]
        y_boot = y_clean[idx]
        
        # Compute correlation
        rho, _ = spearmanr(x_boot, y_boot)
        if not np.isnan(rho):
            boot_rhos.append(rho)
    
    if len(boot_rhos) < n_bootstrap * 0.5:
        return {'ci_lower': np.nan, 'ci_upper': np.nan, 'se': np.nan}
    
    boot_rhos = np.array(boot_rhos)
    alpha = (1 - ci) / 2
    
    return {
        'ci_lower': np.percentile(boot_rhos, alpha * 100),
        'ci_upper': np.percentile(boot_rhos, (1 - alpha) * 100),
        'se': np.std(boot_rhos),
    }


def permutation_test(x: np.ndarray, y: np.ndarray, observed_rho: float,
                    n_permutations: int = N_PERMUTATIONS) -> float:
    """Permutation test for correlation significance."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < MIN_SAMPLE_SIZE or np.isnan(observed_rho):
        return np.nan
    
    # Count how many permuted correlations are as extreme as observed
    n_extreme = 0
    
    for _ in range(n_permutations):
        # Permute y
        y_perm = np.random.permutation(y_clean)
        rho_perm, _ = spearmanr(x_clean, y_perm)
        
        # Two-tailed test
        if abs(rho_perm) >= abs(observed_rho):
            n_extreme += 1
    
    # Add 1 to numerator and denominator for conservative estimate
    return (n_extreme + 1) / (n_permutations + 1)


def compute_effect_size_interpretation(rho: float) -> str:
    """Interpret correlation effect size (Cohen's guidelines)."""
    abs_rho = abs(rho) if not np.isnan(rho) else 0
    
    if abs_rho < 0.1:
        return "negligible"
    elif abs_rho < 0.3:
        return "small"
    elif abs_rho < 0.5:
        return "medium"
    else:
        return "large"


# =============================================================================
# Analysis Functions
# =============================================================================

def get_utp_feature_columns(df: pd.DataFrame) -> list:
    """Get columns that are uTP features."""
    cols = []
    for col in df.columns:
        for prefix in UTP_FEATURE_PREFIXES:
            if col.startswith(prefix):
                # Check if numeric
                if df[col].dtype in [np.float64, np.int64, float, int]:
                    cols.append(col)
                break
    return list(set(cols))


def get_outcome_columns(df: pd.DataFrame) -> list:
    """Get columns that are outcomes."""
    return [c for c in OUTCOME_COLUMNS if c in df.columns]


def run_correlation_analysis(df: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    """
    Run full correlation analysis for a dataset subset.
    
    Args:
        df: DataFrame with features and outcomes
        subset_name: Name of the subset (for reporting)
    
    Returns:
        DataFrame with correlation results
    """
    print(f"\n  Analyzing {subset_name} (n={len(df)})...")
    
    feature_cols = get_utp_feature_columns(df)
    outcome_cols = get_outcome_columns(df)
    
    print(f"    Features: {len(feature_cols)}")
    print(f"    Outcomes: {len(outcome_cols)}")
    
    results = []
    
    total_tests = len(feature_cols) * len(outcome_cols)
    test_count = 0
    
    for feature in feature_cols:
        for outcome in outcome_cols:
            test_count += 1
            
            x = df[feature].values
            y = df[outcome].values
            
            # Spearman correlation
            rho, pval, n = compute_spearman(x, y)
            
            # Pearson for comparison
            r_pearson, _, _ = compute_pearson(x, y)
            
            results.append({
                'subset': subset_name,
                'feature': feature,
                'outcome': outcome,
                'n': n,
                'spearman_rho': rho,
                'spearman_p': pval,
                'pearson_r': r_pearson,
                'effect_size': compute_effect_size_interpretation(rho),
            })
    
    return pd.DataFrame(results)


def run_bootstrap_analysis(df: pd.DataFrame, top_results: pd.DataFrame,
                          n_top: int = 50) -> pd.DataFrame:
    """Run bootstrap CI analysis for top correlations."""
    print(f"\n  Running bootstrap analysis for top {n_top} correlations...")
    
    boot_results = []
    
    for _, row in top_results.head(n_top).iterrows():
        feature = row['feature']
        outcome = row['outcome']
        
        x = df[feature].values
        y = df[outcome].values
        
        boot_ci = bootstrap_correlation(x, y)
        
        boot_results.append({
            'feature': feature,
            'outcome': outcome,
            'rho': row['spearman_rho'],
            'ci_lower': boot_ci['ci_lower'],
            'ci_upper': boot_ci['ci_upper'],
            'se': boot_ci['se'],
        })
    
    return pd.DataFrame(boot_results)


def run_permutation_validation(df: pd.DataFrame, top_results: pd.DataFrame,
                              n_top: int = 20) -> pd.DataFrame:
    """Run permutation tests for top correlations."""
    print(f"\n  Running permutation tests for top {n_top} correlations...")
    
    perm_results = []
    
    for _, row in top_results.head(n_top).iterrows():
        feature = row['feature']
        outcome = row['outcome']
        rho = row['spearman_rho']
        
        x = df[feature].values
        y = df[outcome].values
        
        perm_p = permutation_test(x, y, rho)
        
        perm_results.append({
            'feature': feature,
            'outcome': outcome,
            'spearman_rho': rho,
            'parametric_p': row['spearman_p'],
            'permutation_p': perm_p,
            'perm_validates': perm_p < 0.05 if not np.isnan(perm_p) else False,
        })
    
    return pd.DataFrame(perm_results)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Rigorous Correlation Analysis")
    print("uTP Features vs Functional Outcomes")
    print("=" * 70)
    
    # =========================================================================
    # Load data
    # =========================================================================
    print("\n[1/6] Loading data...")
    
    if not MERGED_DATA.exists():
        print(f"  Error: {MERGED_DATA} not found")
        print("  Run 00_extract_utp_features.py and 01_extract_functional_outcomes.py first")
        return
    
    df = pd.read_csv(MERGED_DATA)
    print(f"  Loaded {len(df)} proteins with features and outcomes")
    
    # Identify subsets
    if 'source' in df.columns:
        experimental_df = df[df['source'] == 'experimental'].copy()
        hmm_only_df = df[df['source'] == 'hmm_only'].copy()
        print(f"  Experimental: {len(experimental_df)}")
        print(f"  HMM-only: {len(hmm_only_df)}")
    else:
        experimental_df = df.copy()
        hmm_only_df = pd.DataFrame()
        print(f"  No source column found, analyzing all data")
    
    # =========================================================================
    # Run correlations for all data
    # =========================================================================
    print("\n[2/6] Computing correlations...")
    
    all_results = []
    
    # All data
    results_all = run_correlation_analysis(df, "all")
    all_results.append(results_all)
    
    # Experimental subset
    if len(experimental_df) >= MIN_SAMPLE_SIZE:
        results_exp = run_correlation_analysis(experimental_df, "experimental")
        all_results.append(results_exp)
    
    # HMM-only subset
    if len(hmm_only_df) >= MIN_SAMPLE_SIZE:
        results_hmm = run_correlation_analysis(hmm_only_df, "hmm_only")
        all_results.append(results_hmm)
    
    # Combine results
    results_df = pd.concat(all_results, ignore_index=True)
    
    # =========================================================================
    # FDR correction
    # =========================================================================
    print("\n[3/6] Applying FDR correction...")
    
    # Apply FDR correction within each subset
    for subset in results_df['subset'].unique():
        mask = results_df['subset'] == subset
        pvals = results_df.loc[mask, 'spearman_p'].values
        
        # Handle NaN p-values
        valid_mask = ~np.isnan(pvals)
        fdr_corrected = np.full_like(pvals, np.nan)
        significant = np.full_like(pvals, False, dtype=bool)
        
        if valid_mask.sum() > 0:
            _, fdr_corrected[valid_mask] = fdrcorrection(pvals[valid_mask])
            significant[valid_mask] = fdr_corrected[valid_mask] < FDR_THRESHOLD
        
        results_df.loc[mask, 'p_adjusted'] = fdr_corrected
        results_df.loc[mask, 'significant_fdr'] = significant
    
    # =========================================================================
    # Bootstrap analysis for top correlations
    # =========================================================================
    print("\n[4/6] Bootstrap confidence intervals...")
    
    # Get top correlations from "all" subset
    top_all = results_df[
        (results_df['subset'] == 'all') & 
        results_df['significant_fdr'].fillna(False)
    ].sort_values('spearman_p')
    
    if len(top_all) > 0:
        boot_df = run_bootstrap_analysis(df, top_all, n_top=min(50, len(top_all)))
        boot_df.to_csv(OUTPUT_DIR / "bootstrap_confidence_intervals.csv", index=False)
        print(f"    Saved bootstrap CIs for {len(boot_df)} correlations")
    else:
        boot_df = pd.DataFrame()
        print("    No significant correlations for bootstrap analysis")
    
    # =========================================================================
    # Permutation validation
    # =========================================================================
    print("\n[5/6] Permutation validation...")
    
    if len(top_all) > 0:
        perm_df = run_permutation_validation(df, top_all, n_top=min(20, len(top_all)))
        perm_df.to_csv(OUTPUT_DIR / "permutation_validation.csv", index=False)
        print(f"    Saved permutation tests for {len(perm_df)} correlations")
        
        # Check validation rate
        validated = perm_df['perm_validates'].sum()
        print(f"    Permutation validation rate: {validated}/{len(perm_df)} ({100*validated/len(perm_df):.1f}%)")
    else:
        perm_df = pd.DataFrame()
        print("    No significant correlations for permutation validation")
    
    # =========================================================================
    # Save results
    # =========================================================================
    print("\n[6/6] Saving results...")
    
    # Full results
    results_df.to_csv(OUTPUT_DIR / "correlation_results.csv", index=False)
    print(f"  Saved correlation_results.csv ({len(results_df)} rows)")
    
    # Significant results only
    sig_results = results_df[results_df['significant_fdr'].fillna(False)]
    sig_results.to_csv(OUTPUT_DIR / "significant_correlations.csv", index=False)
    print(f"  Saved significant_correlations.csv ({len(sig_results)} rows)")
    
    # Summary by subset
    summary_rows = []
    for subset in results_df['subset'].unique():
        subset_data = results_df[results_df['subset'] == subset]
        n_tests = len(subset_data)
        n_valid = subset_data['spearman_p'].notna().sum()
        n_sig = subset_data['significant_fdr'].fillna(False).sum()
        
        # Effect sizes
        sig_data = subset_data[subset_data['significant_fdr'].fillna(False)]
        if len(sig_data) > 0:
            mean_rho = sig_data['spearman_rho'].abs().mean()
            max_rho = sig_data['spearman_rho'].abs().max()
        else:
            mean_rho = np.nan
            max_rho = np.nan
        
        summary_rows.append({
            'subset': subset,
            'n_tests': n_tests,
            'n_valid_tests': n_valid,
            'n_significant': n_sig,
            'pct_significant': 100 * n_sig / n_valid if n_valid > 0 else 0,
            'mean_abs_rho_significant': mean_rho,
            'max_abs_rho': max_rho,
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "correlation_summary.csv", index=False)
    
    # =========================================================================
    # Print summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nAnalysis parameters:")
    print(f"  FDR threshold: {FDR_THRESHOLD}")
    print(f"  Minimum sample size: {MIN_SAMPLE_SIZE}")
    print(f"  Bootstrap iterations: {N_BOOTSTRAP}")
    print(f"  Permutation iterations: {N_PERMUTATIONS}")
    
    print(f"\nResults by subset:")
    for _, row in summary_df.iterrows():
        print(f"\n  {row['subset']}:")
        print(f"    Total tests: {row['n_tests']}")
        print(f"    Valid tests: {row['n_valid_tests']}")
        print(f"    Significant (FDR<{FDR_THRESHOLD}): {row['n_significant']} ({row['pct_significant']:.1f}%)")
        if not np.isnan(row['mean_abs_rho_significant']):
            print(f"    Mean |rho| (significant): {row['mean_abs_rho_significant']:.3f}")
            print(f"    Max |rho|: {row['max_abs_rho']:.3f}")
    
    # Top correlations
    if len(sig_results) > 0:
        print(f"\nTop 10 significant correlations (all data):")
        top_10 = results_df[
            (results_df['subset'] == 'all') &
            results_df['significant_fdr'].fillna(False)
        ].nsmallest(10, 'p_adjusted')
        
        for _, row in top_10.iterrows():
            print(f"  {row['feature']} vs {row['outcome']}: "
                  f"rho={row['spearman_rho']:.3f}, p_adj={row['p_adjusted']:.2e}, n={row['n']}")
    
    print(f"\n📁 Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
