#!/usr/bin/env python3
"""
01_correlation_analysis.py - uTP vs Mature Domain Correlation Analysis

Tests whether continuous features of the uTP region correlate with features
of the mature domain, providing insight into potential co-evolution or
bipartite signal organization.

Methods:
1. Spearman rank correlations (robust to non-normality)
2. FDR correction for multiple testing (Benjamini-Hochberg)
3. Bootstrap confidence intervals
4. Canonical correlation analysis (CCA) for multivariate relationships
5. Separate analyses for experimental and HMM-predicted sets

Usage:
    uv run python experiments/utp_mature_correlation/01_correlation_analysis.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import fdrcorrection

warnings.filterwarnings("ignore")

# Random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"

# Input
FEATURES_FILE = OUTPUT_DIR / "mature_utp_features.csv"

# =============================================================================
# Configuration
# =============================================================================

# Minimum sample size
MIN_SAMPLE_SIZE = 30

# FDR threshold
FDR_THRESHOLD = 0.05

# Bootstrap parameters
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 0.95

# CCA parameters
N_CCA_COMPONENTS = 5

# Feature categories for focused analysis
FEATURE_CATEGORIES = {
    "physicochemical": [
        "length",
        "molecular_weight",
        "gravy",
        "isoelectric_point",
        "instability_index",
        "aromaticity",
    ],
    "charge": [
        "net_charge",
        "charge_density",
        "positive_fraction",
        "negative_fraction",
    ],
    "hydrophobicity": ["hydro_mean", "hydro_std"],
    "disorder": ["disorder_mean", "disorder_fraction"],
    "structure": ["helix_propensity", "sheet_propensity"],
    "complexity": ["entropy"],
}


# =============================================================================
# Statistical Functions
# =============================================================================


def compute_spearman(x: np.ndarray, y: np.ndarray) -> tuple:
    """Compute Spearman correlation with p-value."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < MIN_SAMPLE_SIZE:
        return np.nan, np.nan, 0

    rho, pval = spearmanr(x_clean, y_clean)
    return rho, pval, len(x_clean)


def bootstrap_correlation(
    x: np.ndarray, y: np.ndarray, n_bootstrap: int = N_BOOTSTRAP
) -> dict:
    """Compute bootstrap CI for Spearman correlation."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < MIN_SAMPLE_SIZE:
        return {"ci_lower": np.nan, "ci_upper": np.nan, "se": np.nan}

    n = len(x_clean)
    boot_rhos = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        rho, _ = spearmanr(x_clean[idx], y_clean[idx])
        if not np.isnan(rho):
            boot_rhos.append(rho)

    if len(boot_rhos) < n_bootstrap * 0.5:
        return {"ci_lower": np.nan, "ci_upper": np.nan, "se": np.nan}

    boot_rhos = np.array(boot_rhos)
    alpha = (1 - BOOTSTRAP_CI) / 2

    return {
        "ci_lower": np.percentile(boot_rhos, alpha * 100),
        "ci_upper": np.percentile(boot_rhos, (1 - alpha) * 100),
        "se": np.std(boot_rhos),
    }


def effect_size_interpretation(rho: float) -> str:
    """Cohen's guidelines for correlation effect size."""
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
# Correlation Analysis
# =============================================================================


def get_feature_columns(df: pd.DataFrame, prefix: str) -> list:
    """Get numeric feature columns with given prefix."""
    cols = []
    for col in df.columns:
        if col.startswith(prefix) and df[col].dtype in [
            np.float64,
            np.int64,
            float,
            int,
        ]:
            cols.append(col)
    return cols


def run_pairwise_correlations(df: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    """Run all pairwise correlations between mature and uTP features."""

    print(f"\n  Analyzing {subset_name} (n={len(df)})...")

    mature_cols = get_feature_columns(df, "mature_")
    utp_cols = get_feature_columns(df, "utp_")

    print(f"    Mature features: {len(mature_cols)}")
    print(f"    uTP features: {len(utp_cols)}")

    results = []

    for mature_feat in mature_cols:
        for utp_feat in utp_cols:
            x = df[mature_feat].values
            y = df[utp_feat].values

            rho, pval, n = compute_spearman(x, y)

            # Get feature base names (without prefix)
            mature_base = mature_feat.replace("mature_", "")
            utp_base = utp_feat.replace("utp_", "")

            # Check if same feature type (self-correlation)
            is_same_feature = mature_base == utp_base

            results.append(
                {
                    "subset": subset_name,
                    "mature_feature": mature_feat,
                    "utp_feature": utp_feat,
                    "mature_base": mature_base,
                    "utp_base": utp_base,
                    "same_feature_type": is_same_feature,
                    "n": n,
                    "spearman_rho": rho,
                    "spearman_p": pval,
                    "effect_size": effect_size_interpretation(rho),
                }
            )

    return pd.DataFrame(results)


def run_focused_correlations(df: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    """Run correlations for key feature pairs."""

    results = []

    for category, features in FEATURE_CATEGORIES.items():
        for feat in features:
            mature_col = f"mature_{feat}"
            utp_col = f"utp_{feat}"

            if mature_col not in df.columns or utp_col not in df.columns:
                continue

            x = df[mature_col].values
            y = df[utp_col].values

            rho, pval, n = compute_spearman(x, y)
            boot = bootstrap_correlation(x, y)

            results.append(
                {
                    "subset": subset_name,
                    "category": category,
                    "feature": feat,
                    "n": n,
                    "spearman_rho": rho,
                    "spearman_p": pval,
                    "ci_lower": boot["ci_lower"],
                    "ci_upper": boot["ci_upper"],
                    "se": boot["se"],
                    "effect_size": effect_size_interpretation(rho),
                }
            )

    return pd.DataFrame(results)


def run_cca_analysis(df: pd.DataFrame, subset_name: str) -> dict:
    """Run Canonical Correlation Analysis."""

    print(f"\n  CCA for {subset_name}...")

    mature_cols = get_feature_columns(df, "mature_")
    utp_cols = get_feature_columns(df, "utp_")

    # Get complete cases
    all_cols = mature_cols + utp_cols
    df_clean = df[all_cols].dropna()

    if len(df_clean) < MIN_SAMPLE_SIZE:
        print(f"    Insufficient samples ({len(df_clean)})")
        return {"subset": subset_name, "n": len(df_clean), "canonical_correlations": []}

    X = df_clean[mature_cols].values
    Y = df_clean[utp_cols].values

    # Standardize
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)

    # Fit CCA
    n_components = min(
        N_CCA_COMPONENTS, len(mature_cols), len(utp_cols), len(df_clean) // 5
    )

    if n_components < 1:
        return {"subset": subset_name, "n": len(df_clean), "canonical_correlations": []}

    cca = CCA(n_components=n_components)
    X_c, Y_c = cca.fit_transform(X_scaled, Y_scaled)

    # Compute canonical correlations
    canonical_corrs = []
    for i in range(n_components):
        r, _ = stats.pearsonr(X_c[:, i], Y_c[:, i])
        canonical_corrs.append(r)

    print(f"    Canonical correlations: {[f'{r:.3f}' for r in canonical_corrs]}")

    return {
        "subset": subset_name,
        "n": len(df_clean),
        "n_components": n_components,
        "canonical_correlations": canonical_corrs,
        "cca_model": cca,
        "mature_cols": mature_cols,
        "utp_cols": utp_cols,
    }


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("uTP-Mature Domain Correlation Analysis")
    print("=" * 70)

    # =========================================================================
    # Load data
    # =========================================================================
    print("\n[1/5] Loading data...")

    if not FEATURES_FILE.exists():
        print(f"  Error: {FEATURES_FILE} not found")
        print("  Run 00_extract_features.py first")
        return

    df = pd.read_csv(FEATURES_FILE)
    print(f"  Loaded {len(df)} proteins")

    # Split by source
    if "source" in df.columns:
        experimental_df = df[df["source"] == "experimental"].copy()
        hmm_only_df = df[df["source"] == "hmm_only"].copy()
        print(f"  Experimental: {len(experimental_df)}")
        print(f"  HMM-only: {len(hmm_only_df)}")
    else:
        experimental_df = df.copy()
        hmm_only_df = pd.DataFrame()

    # =========================================================================
    # Pairwise correlations
    # =========================================================================
    print("\n[2/5] Computing pairwise correlations...")

    all_results = []

    # All data
    results_all = run_pairwise_correlations(df, "all")
    all_results.append(results_all)

    # Experimental
    if len(experimental_df) >= MIN_SAMPLE_SIZE:
        results_exp = run_pairwise_correlations(experimental_df, "experimental")
        all_results.append(results_exp)

    # HMM-only
    if len(hmm_only_df) >= MIN_SAMPLE_SIZE:
        results_hmm = run_pairwise_correlations(hmm_only_df, "hmm_only")
        all_results.append(results_hmm)

    results_df = pd.concat(all_results, ignore_index=True)

    # =========================================================================
    # FDR correction
    # =========================================================================
    print("\n[3/5] Applying FDR correction...")

    for subset in results_df["subset"].unique():
        mask = results_df["subset"] == subset
        pvals = results_df.loc[mask, "spearman_p"].values

        valid_mask = ~np.isnan(pvals)
        fdr_corrected = np.full_like(pvals, np.nan)
        significant = np.full_like(pvals, False, dtype=bool)

        if valid_mask.sum() > 0:
            _, fdr_corrected[valid_mask] = fdrcorrection(pvals[valid_mask])
            significant[valid_mask] = fdr_corrected[valid_mask] < FDR_THRESHOLD

        results_df.loc[mask, "p_adjusted"] = fdr_corrected
        results_df.loc[mask, "significant_fdr"] = significant

    # =========================================================================
    # Focused analysis with bootstrap CIs
    # =========================================================================
    print("\n[4/5] Running focused analysis with bootstrap CIs...")

    focused_results = []

    focused_all = run_focused_correlations(df, "all")
    focused_results.append(focused_all)

    if len(experimental_df) >= MIN_SAMPLE_SIZE:
        focused_exp = run_focused_correlations(experimental_df, "experimental")
        focused_results.append(focused_exp)

    if len(hmm_only_df) >= MIN_SAMPLE_SIZE:
        focused_hmm = run_focused_correlations(hmm_only_df, "hmm_only")
        focused_results.append(focused_hmm)

    focused_df = pd.concat(focused_results, ignore_index=True)

    # FDR for focused
    for subset in focused_df["subset"].unique():
        mask = focused_df["subset"] == subset
        pvals = focused_df.loc[mask, "spearman_p"].values

        valid_mask = ~np.isnan(pvals)
        fdr_corrected = np.full_like(pvals, np.nan)

        if valid_mask.sum() > 0:
            _, fdr_corrected[valid_mask] = fdrcorrection(pvals[valid_mask])

        focused_df.loc[mask, "p_adjusted"] = fdr_corrected
        focused_df.loc[mask, "significant_fdr"] = fdr_corrected < FDR_THRESHOLD

    # =========================================================================
    # CCA analysis
    # =========================================================================
    print("\n[5/5] Running Canonical Correlation Analysis...")

    cca_results = []

    cca_all = run_cca_analysis(df, "all")
    cca_results.append(cca_all)

    if len(experimental_df) >= MIN_SAMPLE_SIZE:
        cca_exp = run_cca_analysis(experimental_df, "experimental")
        cca_results.append(cca_exp)

    if len(hmm_only_df) >= MIN_SAMPLE_SIZE:
        cca_hmm = run_cca_analysis(hmm_only_df, "hmm_only")
        cca_results.append(cca_hmm)

    # Save CCA summary
    cca_summary = []
    for result in cca_results:
        for i, cc in enumerate(result.get("canonical_correlations", [])):
            cca_summary.append(
                {
                    "subset": result["subset"],
                    "n": result["n"],
                    "component": i + 1,
                    "canonical_correlation": cc,
                }
            )
    cca_df = pd.DataFrame(cca_summary)

    # =========================================================================
    # Save outputs
    # =========================================================================
    print("\nSaving outputs...")

    # Full pairwise results
    results_df.to_csv(OUTPUT_DIR / "pairwise_correlations.csv", index=False)
    print(f"  Saved pairwise_correlations.csv ({len(results_df)} rows)")

    # Significant only
    sig_results = results_df[results_df["significant_fdr"].fillna(False)]
    sig_results.to_csv(OUTPUT_DIR / "significant_correlations.csv", index=False)
    print(f"  Saved significant_correlations.csv ({len(sig_results)} rows)")

    # Same-feature correlations (mature vs uTP for same property)
    same_feat = results_df[results_df["same_feature_type"]].copy()
    same_feat.to_csv(OUTPUT_DIR / "same_feature_correlations.csv", index=False)
    print(f"  Saved same_feature_correlations.csv ({len(same_feat)} rows)")

    # Focused analysis with bootstrap
    focused_df.to_csv(OUTPUT_DIR / "focused_correlations.csv", index=False)
    print(f"  Saved focused_correlations.csv ({len(focused_df)} rows)")

    # CCA results
    cca_df.to_csv(OUTPUT_DIR / "cca_results.csv", index=False)
    print(f"  Saved cca_results.csv ({len(cca_df)} rows)")

    # Summary statistics
    summary_rows = []
    for subset in results_df["subset"].unique():
        subset_data = results_df[results_df["subset"] == subset]
        n_tests = len(subset_data)
        n_valid = subset_data["spearman_p"].notna().sum()
        n_sig = subset_data["significant_fdr"].fillna(False).sum()

        # Same-feature correlations
        same = subset_data[subset_data["same_feature_type"]]
        n_same_sig = same["significant_fdr"].fillna(False).sum()

        sig_data = subset_data[subset_data["significant_fdr"].fillna(False)]
        mean_rho = (
            sig_data["spearman_rho"].abs().mean() if len(sig_data) > 0 else np.nan
        )
        max_rho = sig_data["spearman_rho"].abs().max() if len(sig_data) > 0 else np.nan

        summary_rows.append(
            {
                "subset": subset,
                "n_tests": n_tests,
                "n_valid": n_valid,
                "n_significant": n_sig,
                "pct_significant": 100 * n_sig / n_valid if n_valid > 0 else 0,
                "n_same_feature_tests": len(same),
                "n_same_feature_significant": n_same_sig,
                "mean_abs_rho_significant": mean_rho,
                "max_abs_rho": max_rho,
            }
        )

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

    print(f"\nResults by subset:")
    for _, row in summary_df.iterrows():
        print(f"\n  {row['subset']}:")
        print(f"    Total tests: {row['n_tests']}")
        print(
            f"    Significant (FDR<{FDR_THRESHOLD}): {row['n_significant']} ({row['pct_significant']:.1f}%)"
        )
        print(f"    Same-feature tests: {row['n_same_feature_tests']}")
        print(f"    Same-feature significant: {row['n_same_feature_significant']}")
        if not np.isnan(row["mean_abs_rho_significant"]):
            print(f"    Mean |ρ| (significant): {row['mean_abs_rho_significant']:.3f}")
            print(f"    Max |ρ|: {row['max_abs_rho']:.3f}")

    # Top correlations
    print(f"\nTop 10 significant correlations (all data):")
    top_sig = results_df[
        (results_df["subset"] == "all") & results_df["significant_fdr"].fillna(False)
    ].nsmallest(10, "p_adjusted")

    for _, row in top_sig.iterrows():
        same_marker = " [SAME]" if row["same_feature_type"] else ""
        print(
            f"  {row['mature_base']} vs {row['utp_base']}: "
            f"ρ={row['spearman_rho']:.3f}, p_adj={row['p_adjusted']:.2e}{same_marker}"
        )

    # CCA summary
    print(f"\nCanonical Correlation Analysis:")
    for subset in ["all", "experimental", "hmm_only"]:
        cca_subset = cca_df[cca_df["subset"] == subset]
        if len(cca_subset) > 0:
            ccs = cca_subset["canonical_correlation"].tolist()
            print(f"  {subset}: {[f'{cc:.3f}' for cc in ccs[:3]]}")

    print(f"\n📁 Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
