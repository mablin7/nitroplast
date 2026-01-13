#!/usr/bin/env python3
"""
01_extract_functional_outcomes.py - Extract Functional Outcomes from Proteomics Data

Extracts continuous functional outcomes from Coale et al. proteomics data for
downstream correlation analysis with uTP features.

Outcomes extracted:
1. Expression levels (log-transformed quantities)
2. Day/night differential expression (log2 fold change)
3. UCYN-A enrichment scores
4. Temporal variability metrics

Usage:
    uv run python experiments/utp_feature_correlation/01_extract_functional_outcomes.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Input files
PROTEIN_QUANT = DATA_DIR / "ADK1075_ProteinQuantifications.csv"
DATA_S1 = DATA_DIR / "Data_S1_Bbigelowii_proteins.csv"
ANNOTATIONS = DATA_DIR / "Bbigelowii_transcriptome_annotations.csv"

# From feature extraction
UTP_FEATURES = OUTPUT_DIR / "utp_features.csv"


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_protein_quantifications(quant_file: Path) -> pd.DataFrame:
    """
    Load and parse the protein quantifications file.
    
    The file has a complex header structure:
    - Row 1: Time (Day/Night)
    - Row 2: Sample type (Isolated UCYNA / Whole culture)
    - Row 3: Column names
    """
    # Read raw to understand structure
    raw = pd.read_csv(quant_file, header=None)
    
    # Extract column metadata from first 3 rows
    time_row = raw.iloc[0].tolist()
    sample_type_row = raw.iloc[1].tolist()
    col_names_row = raw.iloc[2].tolist()
    
    # Build proper column names
    # First column is protein accession
    columns = ['protein_accession']
    
    # Sample columns (indices 1-12)
    sample_cols = []
    for i in range(1, 13):
        time = time_row[i]
        sample_type = sample_type_row[i]
        sample_cols.append(f"{time}_{sample_type}_{i}")
    columns.extend(sample_cols)
    
    # Summary columns
    summary_cols = [
        'avg_ucyna_day', 'avg_whole_day', 'avg_ucyna_night', 'avg_whole_night',
        'logFC_day', 'logFC_night', 'pvalue_day', 'pvalue_night',
        'significant_day', 'significant_night', 'sequence_source'
    ]
    columns.extend(summary_cols)
    
    # Read data (skip header rows)
    df = pd.read_csv(quant_file, skiprows=3, header=None)
    
    # Handle column count mismatch
    if len(df.columns) > len(columns):
        df = df.iloc[:, :len(columns)]
    elif len(df.columns) < len(columns):
        columns = columns[:len(df.columns)]
    
    df.columns = columns
    
    return df


def load_data_s1(s1_file: Path) -> pd.DataFrame:
    """Load Data S1 (B. bigelowii proteins with uTP annotations)."""
    # Skip the first row which is a header description
    df = pd.read_csv(s1_file, skiprows=1)
    
    # Rename columns for consistency
    df.columns = [c.strip().replace(' ', '_').lower() for c in df.columns]
    
    return df


def load_annotations(annot_file: Path) -> pd.DataFrame:
    """Load transcriptome annotations (eggNOG output)."""
    df = pd.read_csv(annot_file, skiprows=3)  # Skip eggNOG header comments
    
    # Clean up column names
    df.columns = [c.strip().replace('#', '').replace(' ', '_').lower() for c in df.columns]
    
    return df


# =============================================================================
# Outcome Extraction Functions
# =============================================================================

def extract_expression_outcomes(quant_df: pd.DataFrame) -> pd.DataFrame:
    """Extract expression-related outcomes."""
    
    outcomes = pd.DataFrame()
    outcomes['name'] = quant_df['protein_accession']
    
    # Average quantities (already in the data)
    outcomes['expr_ucyna_day'] = pd.to_numeric(quant_df['avg_ucyna_day'], errors='coerce')
    outcomes['expr_ucyna_night'] = pd.to_numeric(quant_df['avg_ucyna_night'], errors='coerce')
    outcomes['expr_whole_day'] = pd.to_numeric(quant_df['avg_whole_day'], errors='coerce')
    outcomes['expr_whole_night'] = pd.to_numeric(quant_df['avg_whole_night'], errors='coerce')
    
    # Log-transform expression (add pseudocount to handle zeros)
    pseudocount = 1.0
    outcomes['log_expr_ucyna_day'] = np.log2(outcomes['expr_ucyna_day'] + pseudocount)
    outcomes['log_expr_ucyna_night'] = np.log2(outcomes['expr_ucyna_night'] + pseudocount)
    outcomes['log_expr_whole_day'] = np.log2(outcomes['expr_whole_day'] + pseudocount)
    outcomes['log_expr_whole_night'] = np.log2(outcomes['expr_whole_night'] + pseudocount)
    
    # Log fold changes (enrichment in UCYN-A vs whole culture)
    outcomes['logFC_day'] = pd.to_numeric(quant_df['logFC_day'], errors='coerce')
    outcomes['logFC_night'] = pd.to_numeric(quant_df['logFC_night'], errors='coerce')
    
    # P-values (convert to -log10 for correlation)
    pval_day = pd.to_numeric(quant_df['pvalue_day'], errors='coerce')
    pval_night = pd.to_numeric(quant_df['pvalue_night'], errors='coerce')
    outcomes['neglog_pval_day'] = -np.log10(pval_day.replace(0, 1e-300))
    outcomes['neglog_pval_night'] = -np.log10(pval_night.replace(0, 1e-300))
    
    # Significance flags (binary)
    outcomes['sig_day_ucyna'] = (quant_df['significant_day'] == 'UCYN-A').astype(int)
    outcomes['sig_day_whole'] = (quant_df['significant_day'] == 'Whole culture').astype(int)
    outcomes['sig_night_ucyna'] = (quant_df['significant_night'] == 'UCYN-A').astype(int)
    outcomes['sig_night_whole'] = (quant_df['significant_night'] == 'Whole culture').astype(int)
    
    # Derived metrics
    # Day/night ratio for UCYN-A samples
    outcomes['ucyna_day_night_ratio'] = outcomes['expr_ucyna_day'] / (outcomes['expr_ucyna_night'] + pseudocount)
    outcomes['log_ucyna_day_night_ratio'] = np.log2(outcomes['ucyna_day_night_ratio'] + pseudocount)
    
    # Average enrichment across day and night
    outcomes['avg_logFC'] = (outcomes['logFC_day'] + outcomes['logFC_night']) / 2
    
    # Enrichment consistency (how similar is enrichment between day and night)
    outcomes['logFC_consistency'] = 1 - np.abs(outcomes['logFC_day'] - outcomes['logFC_night']) / (
        np.abs(outcomes['logFC_day']) + np.abs(outcomes['logFC_night']) + 0.001
    )
    
    # Total UCYN-A expression
    outcomes['total_ucyna_expr'] = outcomes['expr_ucyna_day'] + outcomes['expr_ucyna_night']
    outcomes['log_total_ucyna_expr'] = np.log2(outcomes['total_ucyna_expr'] + pseudocount)
    
    # Sequence source
    outcomes['sequence_source'] = quant_df['sequence_source']
    
    return outcomes


def compute_variability_metrics(quant_df: pd.DataFrame) -> pd.DataFrame:
    """Compute variability metrics from replicate data."""
    
    # Extract individual replicate columns
    day_ucyna_cols = [c for c in quant_df.columns if 'Day_Isolated' in c]
    night_ucyna_cols = [c for c in quant_df.columns if 'Night_Isolated' in c]
    day_whole_cols = [c for c in quant_df.columns if 'Day_Whole' in c]
    night_whole_cols = [c for c in quant_df.columns if 'Night_Whole' in c]
    
    var_df = pd.DataFrame()
    var_df['name'] = quant_df['protein_accession']
    
    # Convert columns to numeric
    for col_list in [day_ucyna_cols, night_ucyna_cols, day_whole_cols, night_whole_cols]:
        for col in col_list:
            quant_df[col] = pd.to_numeric(quant_df[col], errors='coerce')
    
    # Coefficient of variation for UCYN-A samples
    if day_ucyna_cols:
        day_vals = quant_df[day_ucyna_cols].values
        var_df['cv_ucyna_day'] = np.nanstd(day_vals, axis=1) / (np.nanmean(day_vals, axis=1) + 1)
        var_df['n_detected_day'] = np.sum(~np.isnan(day_vals), axis=1)
    
    if night_ucyna_cols:
        night_vals = quant_df[night_ucyna_cols].values
        var_df['cv_ucyna_night'] = np.nanstd(night_vals, axis=1) / (np.nanmean(night_vals, axis=1) + 1)
        var_df['n_detected_night'] = np.sum(~np.isnan(night_vals), axis=1)
    
    return var_df


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Functional Outcome Extraction")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Step 1: Load proteomics data
    # =========================================================================
    print("\n[1/4] Loading proteomics data...")
    
    quant_df = load_protein_quantifications(PROTEIN_QUANT)
    print(f"  Protein quantifications: {len(quant_df)} proteins")
    
    # =========================================================================
    # Step 2: Extract expression outcomes
    # =========================================================================
    print("\n[2/4] Extracting expression outcomes...")
    
    expr_outcomes = extract_expression_outcomes(quant_df)
    print(f"  Expression outcomes: {len(expr_outcomes.columns)} features")
    
    # =========================================================================
    # Step 3: Compute variability metrics
    # =========================================================================
    print("\n[3/4] Computing variability metrics...")
    
    var_metrics = compute_variability_metrics(quant_df)
    
    # Merge
    outcomes_df = expr_outcomes.merge(var_metrics, on='name', how='left')
    print(f"  Total outcome features: {len(outcomes_df.columns)}")
    
    # =========================================================================
    # Step 4: Filter to uTP proteins and merge
    # =========================================================================
    print("\n[4/4] Merging with uTP features...")
    
    if UTP_FEATURES.exists():
        utp_df = pd.read_csv(UTP_FEATURES)
        utp_names = set(utp_df['name'])
        print(f"  uTP proteins: {len(utp_names)}")
        
        # Filter outcomes to uTP proteins only
        outcomes_utp = outcomes_df[outcomes_df['name'].isin(utp_names)].copy()
        print(f"  uTP proteins with outcomes: {len(outcomes_utp)}")
        
        # Create merged dataset
        merged_df = utp_df.merge(outcomes_utp, on='name', how='inner')
        print(f"  Merged dataset: {len(merged_df)} proteins")
        
        # Save merged dataset
        merged_df.to_csv(OUTPUT_DIR / "utp_features_with_outcomes.csv", index=False)
    else:
        print("  Warning: uTP features not found, saving all outcomes")
        outcomes_utp = outcomes_df
        merged_df = None
    
    # Save outcomes
    outcomes_df.to_csv(OUTPUT_DIR / "functional_outcomes.csv", index=False)
    
    # =========================================================================
    # Summary statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nOutcome features extracted:")
    outcome_cols = [c for c in outcomes_df.columns if c not in ['name', 'sequence_source']]
    print(f"  Total features: {len(outcome_cols)}")
    
    # Count non-missing values for key outcomes
    print(f"\nData completeness (non-missing counts):")
    key_outcomes = ['logFC_day', 'logFC_night', 'log_expr_ucyna_day', 
                   'log_expr_ucyna_night', 'ucyna_day_night_ratio']
    for col in key_outcomes:
        if col in outcomes_df.columns:
            n_valid = outcomes_df[col].notna().sum()
            print(f"  {col}: {n_valid} / {len(outcomes_df)} ({100*n_valid/len(outcomes_df):.1f}%)")
    
    # Source breakdown
    print(f"\nSequence sources:")
    source_counts = outcomes_df['sequence_source'].value_counts()
    for source, count in source_counts.items():
        print(f"  {source}: {count}")
    
    if merged_df is not None:
        print(f"\nMerged dataset (uTP proteins with proteomics):")
        print(f"  Total proteins: {len(merged_df)}")
        print(f"  Total features: {len(merged_df.columns)}")
        
        # Source breakdown for merged
        source_col = 'source' if 'source' in merged_df.columns else None
        if source_col:
            print(f"  Experimental: {(merged_df[source_col] == 'experimental').sum()}")
            print(f"  HMM-only: {(merged_df[source_col] == 'hmm_only').sum()}")
    
    print(f"\n📁 Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
