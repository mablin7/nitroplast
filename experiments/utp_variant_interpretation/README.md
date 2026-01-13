# uTP Variant Interpretation

## Motivation

Motif-based clustering shows 5-6 natural groupings of uTP sequences based on motif composition. This experiment tests two competing interpretations of why mature domain features might predict uTP variant:

**Interpretation A (Gene Family Structure):**
Proteins acquired uTP through gene duplication. When a uTP-containing gene duplicates, the daughter gene inherits the same motif combination. The classifier learns to recognize gene families, not functional targeting.

**Interpretation B (Functional Constraint):**
The mature domain has features (biophysical properties, structural motifs) that determine which import channel it uses, and different channels recognize different uTP variants. The classifier learns these biophysical features.

## Variant Group Definitions

Groups were defined based on terminal motif, with subdivision of the dominant terminal_6 group:

| Group     | Description                         | N   | %     |
| --------- | ----------------------------------- | --- | ----- |
| variant_A | Terminal 6 + has motif 5            | 70  | 9.4%  |
| variant_B | Terminal 6 + has motif 7            | 55  | 7.4%  |
| variant_C | Terminal 6 + motif 9 or many motifs | 91  | 12.2% |
| variant_D | Terminal 6 minimal                  | 244 | 32.8% |
| variant_E | Terminal 4 or 3                     | 142 | 19.1% |
| variant_F | Other terminals (5, 7, 0, rare)     | 143 | 19.2% |

Total: 745 sequences covered.

## Results

### Interpretation A: Gene Family Structure

| Test                         | Metric            | Value  | Interpretation    |
| ---------------------------- | ----------------- | ------ | ----------------- |
| Within vs Between Similarity | Within mean       | 0.0144 | Marginally higher |
|                              | Between mean      | 0.0128 |                   |
|                              | Effect size       | -0.025 | Negligible        |
| Cluster-Variant Alignment    | ARI               | -0.001 | No alignment      |
| Permutation Test             | Z-score           | 9.94   | Significant       |
|                              | Actual difference | 0.0013 | Negligible        |

**Conclusion: ✗ NOT SUPPORTED**

While the permutation test is significant, the effect size is negligible. Sequence clusters do not align with variant groups (ARI ≈ 0). Same-variant proteins are NOT substantially more similar than expected.

### Interpretation B: Functional Constraint

| Test                     | Metric               | Value         | Interpretation      |
| ------------------------ | -------------------- | ------------- | ------------------- |
| Biophysical Differences  | Significant features | 3 / 15        | Few differences     |
|                          | Top feature          | length        | Likely confound     |
| Prediction Accuracy      | Baseline             | 0.328         | Majority class      |
|                          | Logistic Regression  | 0.330 ± 0.027 | No improvement      |
|                          | Random Forest        | 0.344 ± 0.033 | +1.6% over baseline |
| Controlling for Sequence | Biophys coefficient  | 0.064         | Small positive      |
|                          | Sequence coefficient | 0.690         | Much larger         |

**Conclusion: ✗ NOT SUPPORTED**

Classification accuracy is essentially at baseline. The few significant biophysical differences (length, molecular weight) are likely correlated confounds. Sequence similarity is a much stronger predictor of same-variant than biophysical similarity.

## Overall Conclusions

Neither interpretation is strongly supported:

1. **Gene family structure (A)**: Variant groups do NOT cluster by sequence similarity. Same-variant proteins are not paralogs.

2. **Functional constraint (B)**: Biophysical properties do NOT predict variant membership better than random guessing.

### Implications

- The motif-based variant groupings may be **arbitrary divisions** of continuous variation
- The correlation between mature domain and uTP variant (from the classifier) may reflect **subtle features** not captured by simple biophysical properties
- Alternative explanations:
  - Co-evolution between mature domain and uTP at the codon level
  - Structural features not captured by standard biophysical metrics
  - The grouping criteria themselves need refinement

## Scripts

1. `01_define_groups.py` - Define 6 motif-based groups
2. `02_test_paralog_structure.py` - Test Interpretation A
3. `03_test_biophysical_constraint.py` - Test Interpretation B

## Output Files

| File                            | Description                      |
| ------------------------------- | -------------------------------- |
| `variant_assignments.csv`       | Sequence → variant group mapping |
| `variant_groups.png`            | Group size and motif composition |
| `paralog_test_results.png`      | Interpretation A test results    |
| `biophysical_test_results.png`  | Interpretation B test results    |
| `biophysical_distributions.png` | Feature distributions by variant |
| `biophysical_features.csv`      | Computed biophysical properties  |
