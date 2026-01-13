# uTP Feature-Outcome Correlation Analysis

**Testing whether continuous uTP features predict continuous functional outcomes.**

## Methods

### Feature Extraction

Extracted 63 continuous biophysical features from uTP sequences (C-terminal 120 AA):

- **Physicochemical**: length, molecular weight, GRAVY, isoelectric point, instability index, aromaticity
- **Amino acid composition**: 20 individual amino acids + 11 grouped categories
- **Charge**: net charge, regional charges (N/C-terminal), charge density, asymmetry
- **Hydrophobicity**: mean, std, min, max, range, gradient, regional values
- **Disorder propensity**: mean, std, regional, fraction disordered
- **Secondary structure**: helix propensity, sheet propensity, ratio
- **Sequence complexity**: Shannon entropy, Wootton-Federhen complexity

### Outcome Variables

Extracted 10 continuous outcomes from Coale et al. proteomics data:

- `logFC_day`: Log2 fold change UCYN-A vs whole culture (day)
- `logFC_night`: Log2 fold change UCYN-A vs whole culture (night)
- `avg_logFC`: Average of day/night enrichment
- `log_expr_ucyna_day/night`: Log2 expression in UCYN-A samples
- `log_total_ucyna_expr`: Total UCYN-A expression
- `log_ucyna_day_night_ratio`: Day/night expression ratio
- `logFC_consistency`: Consistency of enrichment between day/night
- `cv_ucyna_day/night`: Coefficient of variation across replicates

### Statistical Analysis

- **Correlation method**: Spearman rank correlation (robust to non-normality)
- **Multiple testing correction**: Benjamini-Hochberg FDR
- **Significance threshold**: FDR < 0.05
- **Minimum sample size**: n ≥ 20
- **Subsets analyzed**: All proteins, Experimental only, HMM-only

## Results

### Sample Sizes

| Subset | n proteins | n tests | n valid tests |
|--------|-----------|---------|---------------|
| All | 362 | 630 | 630 |
| Experimental | 188 | 630 | 620 |
| HMM-only | 174 | 630 | 630 |

### Significant Correlations (FDR < 0.05)

| Subset | n significant | % significant | Mean \|ρ\| | Max \|ρ\| |
|--------|--------------|---------------|-----------|----------|
| All | 0 | 0.0% | - | - |
| Experimental | 0 | 0.0% | - | - |
| HMM-only | 9 | 1.4% | 0.296 | 0.331 |

### Significant Correlations in HMM-only Subset

| Feature | Outcome | ρ | p_adjusted | Effect Size |
|---------|---------|---|------------|-------------|
| utp_group_charged_neg | avg_logFC | 0.331 | 0.0025 | medium |
| utp_negative_fraction | avg_logFC | 0.331 | 0.0025 | medium |
| utp_aa_K | logFC_night | 0.310 | 0.0052 | medium |
| utp_group_charged_neg | logFC_night | 0.305 | 0.0052 | medium |
| utp_negative_fraction | logFC_night | 0.305 | 0.0052 | medium |
| utp_aa_D | avg_logFC | 0.282 | 0.017 | small |
| utp_aa_D | logFC_night | 0.269 | 0.030 | small |
| utp_group_cysteine | avg_logFC | 0.264 | 0.031 | small |
| utp_aa_C | avg_logFC | 0.264 | 0.031 | small |

### Key Observations

1. **No significant correlations in the full dataset (n=362)** after FDR correction
2. **No significant correlations in the experimental subset (n=188)** after FDR correction
3. **9 significant correlations in HMM-only subset (n=174)**, all involving:
   - Charged amino acid features (negative fraction, D, K)
   - UCYN-A enrichment outcomes (logFC_day, logFC_night, avg_logFC)
4. **Effect sizes are small to medium** (ρ = 0.26–0.33)
5. **Positive correlations**: Higher negative charge fraction in uTP → Higher UCYN-A enrichment

## Generated Outputs

```
output/
├── full_protein_features.csv      # Features for full proteins (933 rows)
├── utp_features.csv               # Features for uTP regions (933 rows)
├── functional_outcomes.csv        # Proteomics outcomes (1804 rows)
├── utp_features_with_outcomes.csv # Merged dataset (362 rows)
├── correlation_results.csv        # All correlation tests (1890 rows)
├── significant_correlations.csv   # FDR-significant only (9 rows)
├── correlation_summary.csv        # Summary statistics
├── figure_heatmap.png/svg         # Correlation heatmap
├── figure_volcano.png/svg         # Volcano plot
├── figure_subset_comparison.png/svg # Experimental vs HMM comparison
└── figure_effect_size.png/svg     # Effect size distribution
```

## Usage

```bash
cd /path/to/nitroplast

# Step 1: Extract uTP features
uv run python experiments/utp_feature_correlation/00_extract_utp_features.py

# Step 2: Extract functional outcomes
uv run python experiments/utp_feature_correlation/01_extract_functional_outcomes.py

# Step 3: Run correlation analysis
uv run python experiments/utp_feature_correlation/02_correlation_analysis.py

# Step 4: Generate figures
uv run python experiments/utp_feature_correlation/03_generate_figures.py
```

## Technical Notes

- uTP region defined as C-terminal 120 amino acids
- Proteins with sequence length < 120 AA use full sequence
- Missing proteomics data handled by pairwise deletion
- Spearman correlation used for robustness to non-normality
- Bootstrap CIs and permutation tests available for significant results

---

_Last updated: 2026-01-13_
