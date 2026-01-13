# uTP-Mature Domain Correlation Analysis

**Testing whether continuous features of the mature domain correlate with features of the uTP region.**

## Methods

### Domain Separation

For each protein in Import_candidates.fasta (n=933):

1. Run HMM search to detect uTP boundary
2. If HMM detected (n=712): use HMM envelope start as boundary
3. If HMM not detected (n=221): use C-terminal 120 AA as uTP
4. Filter proteins with mature domain < 50 AA or uTP < 50 AA

### Feature Extraction

Extracted 47 continuous features from each region (mature and uTP):

- **Physicochemical** (6): length, molecular weight, GRAVY, pI, instability index, aromaticity
- **Amino acid composition** (20): individual AA frequencies
- **Grouped composition** (10): hydrophobic, polar, charged_pos, charged_neg, aromatic, small, aliphatic, proline, glycine, cysteine
- **Charge** (4): net charge, charge density, positive fraction, negative fraction
- **Hydrophobicity** (2): mean, standard deviation
- **Disorder** (2): mean propensity, fraction disordered
- **Secondary structure** (2): helix propensity, sheet propensity
- **Complexity** (1): Shannon entropy

### Statistical Analysis

- **Pairwise correlations**: 47 × 48 = 2,256 tests per subset
- **Method**: Spearman rank correlation
- **Multiple testing correction**: Benjamini-Hochberg FDR
- **Bootstrap CIs**: 1,000 iterations for key features
- **CCA**: Canonical Correlation Analysis for multivariate relationships
- **Subsets**: All (n=870), Experimental (n=205), HMM-only (n=665)

## Results

### Sample Sizes

| Subset       | n proteins | Mature length (mean±SD) | uTP length (mean±SD) |
| ------------ | ---------- | ----------------------- | -------------------- |
| All          | 870        | 416 ± 224               | 188 ± 59             |
| Experimental | 205        | 432 ± 212               | 195 ± 66             |
| HMM-only     | 665        | 411 ± 228               | 186 ± 56             |

### Pairwise Correlations Summary

| Subset       | n tests | n significant (FDR<0.05) | % significant | Mean \|ρ\| (significant) |
| ------------ | ------- | ------------------------ | ------------- | ------------------------ |
| All          | 2,256   | 419                      | 18.6%         | 0.131                    |
| Experimental | 2,256   | 37                       | 1.6%          | 0.310                    |
| HMM-only     | 2,256   | 371                      | 16.4%         | 0.146                    |

### Same-Feature Correlations

Testing whether the same property matches between mature domain and uTP:

| Subset       | n same-feature tests | n significant | Top correlation          |
| ------------ | -------------------- | ------------- | ------------------------ |
| All          | 47                   | 26 (55%)      | aa_R: ρ=0.264            |
| Experimental | 47                   | 5 (11%)       | charge_density: ρ=-0.295 |
| HMM-only     | 47                   | 24 (51%)      | entropy: ρ=0.210         |

### Top Same-Feature Correlations (All Data)

| Feature           | ρ (mature vs uTP) | p_adjusted | Effect |
| ----------------- | ----------------- | ---------- | ------ |
| aa_R (Arginine)   | 0.264             | 1.2e-12    | small  |
| entropy           | 0.213             | 2.5e-08    | small  |
| group_polar       | 0.190             | 1.1e-06    | small  |
| aa_N (Asparagine) | 0.188             | 1.5e-06    | small  |
| aa_M (Methionine) | 0.189             | 1.3e-06    | small  |
| aa_W (Tryptophan) | 0.167             | 3.1e-05    | small  |
| aa_K (Lysine)     | 0.159             | 8.3e-05    | small  |
| aa_A (Alanine)    | 0.157             | 1.1e-04    | small  |

### Non-Matching Features (Anti-correlations)

| Feature           | ρ      | p_adjusted | Effect     |
| ----------------- | ------ | ---------- | ---------- |
| charge_density    | -0.101 | 0.022      | small      |
| isoelectric_point | -0.094 | 0.035      | negligible |
| net_charge        | -0.091 | 0.043      | negligible |

### Canonical Correlation Analysis

Multivariate relationship between all mature and uTP features:

| Subset       | CC1     | CC2   | CC3   | CC4   | CC5   |
| ------------ | ------- | ----- | ----- | ----- | ----- |
| All          | 1.000\* | 0.651 | 0.463 | 0.418 | 0.356 |
| Experimental | 1.000\* | 0.673 | 0.651 | 0.631 | 0.563 |
| HMM-only     | 1.000\* | 0.682 | 0.493 | 0.464 | 0.397 |

\*Note: CC1=1.0 due to perfect collinearity of length-related features

### Key Observations

1. **55% of same-feature comparisons show significant correlation** in the full dataset
2. **Strongest same-feature correlations**: Arginine (ρ=0.264), entropy (ρ=0.213), polar amino acids (ρ=0.190)
3. **Charge features show negative correlations**: Higher mature domain charge density associates with lower uTP charge density
4. **Experimental subset shows fewer significant correlations** (1.6% vs 16-19%)
5. **CCA confirms multivariate relationship**: CC2-CC5 range from 0.36-0.68
6. **Effect sizes are predominantly small** (ρ < 0.3)

## Generated Outputs

```
output/
├── mature_utp_features.csv         # Features for both regions (870 rows)
├── pairwise_correlations.csv       # All pairwise tests (6768 rows)
├── significant_correlations.csv    # FDR-significant only (827 rows)
├── same_feature_correlations.csv   # Same feature comparisons (141 rows)
├── focused_correlations.csv        # Key features with bootstrap CI (51 rows)
├── cca_results.csv                 # Canonical correlations (15 rows)
├── correlation_summary.csv         # Summary statistics
├── figure_heatmap.png/svg          # Correlation heatmap
├── figure_same_feature.png/svg     # Same-feature correlation bar plot
├── figure_focused_ci.png/svg       # Key features with bootstrap CIs
├── figure_volcano.png/svg          # Volcano plot
├── figure_scatter.png/svg          # Scatter plots for top correlations
└── figure_cca.png/svg              # Canonical correlation bar plot
```

## Usage

```bash
cd /path/to/nitroplast

# Step 1: Extract features
uv run python experiments/utp_mature_correlation/00_extract_features.py

# Step 2: Run correlation analysis
uv run python experiments/utp_mature_correlation/01_correlation_analysis.py

# Step 3: Generate figures
uv run python experiments/utp_mature_correlation/02_generate_figures.py
```

## Technical Notes

- HMM boundary detection: hmmsearch with E-value < 0.01
- Default uTP length: 120 AA (when HMM boundary not detected)
- Minimum region lengths: mature ≥ 50 AA, uTP ≥ 50 AA
- 664/870 proteins (76%) have HMM-detected boundaries
- Bootstrap uses 1,000 resamples with 95% confidence level

---

_Last updated: 2026-01-13_
