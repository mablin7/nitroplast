# uTP Variant Classifier - Extended Dataset

**Rigorous multiclass classification of uTP motif variants using the full HMM-predicted protein set.**

## Research Question

Can the mature domain of a protein predict which uTP variant (terminal motif class) it will receive?

**Key Improvement**: This experiment uses the extended dataset of 607 proteins with valid terminal motifs (vs. 182 in the original experiment), providing ~3.3× more statistical power.

---

## 📊 Experimental Results (Run: 2026-01-13)

### Key Result: Classification Performance

| Metric                | Value     | 95% CI         |
| --------------------- | --------- | -------------- |
| **Balanced Accuracy** | **51.8%** | [46.5%, 57.6%] |
| Chance Level          | 33.3%     | -              |
| **vs Chance**         | **1.55×** | -              |
| Permutation p-value   | **0.001** | -              |
| Accuracy              | 74.7%     | [70.9%, 78.4%] |
| Macro F1              | 53.3%     | [47.8%, 58.7%] |

**✓ Statistically significant** (p = 0.001, permutation test with 1000 iterations)

### Best Classifier: SVM (RBF kernel)

- Parameters: C=1.0, γ=auto
- Class weights: balanced

### Nested Cross-Validation Results

| Classifier          | Balanced Accuracy | Std   |
| ------------------- | ----------------- | ----- |
| **SVM**             | **51.8%**         | ±3.4% |
| Logistic Regression | 49.1%             | ±7.3% |
| Gradient Boosting   | 40.0%             | ±3.6% |
| Random Forest       | 33.8%             | ±1.0% |

### Per-Class Performance

| Class      | Precision | Recall | F1-Score | n   | vs Chance (Binomial) |
| ---------- | --------- | ------ | -------- | --- | -------------------- |
| terminal_4 | 30.2%     | 29.2%  | 29.7%    | 65  | ns                   |
| terminal_5 | 54.0%     | 39.1%  | 45.4%    | 69  | ns                   |
| terminal_7 | 83.0%     | 87.0%  | 85.0%    | 439 | **p < 0.001** ✓      |

**Note**: Only terminal_7 significantly above chance after Bonferroni correction (α = 0.0167).

### Biochemical Property Differences

| Property         | KW Statistic | p-value | FDR q-value | Significant |
| ---------------- | ------------ | ------- | ----------- | ----------- |
| molecular_weight | 11.78        | 0.0028  | 0.036       | ✓           |
| fraction_charged | 5.44         | 0.066   | 0.295       | -           |
| gravy            | 4.74         | 0.094   | 0.304       | -           |

Only molecular weight shows significant difference between terminal classes (FDR < 0.1).

### Dataset Summary

| Metric                       | Value       |
| ---------------------------- | ----------- |
| Total proteins               | 573         |
| Terminal classes             | 3           |
| Imbalance ratio              | 6.8:1       |
| Experimental (in proteomics) | 164 (28.6%) |
| HMM-only                     | 409 (71.4%) |

### Class Distribution

| Terminal Class | Count | Percentage |
| -------------- | ----- | ---------- |
| terminal_7     | 439   | 76.6%      |
| terminal_5     | 69    | 12.0%      |
| terminal_4     | 65    | 11.3%      |

**Note**: terminal_9 excluded (n=5 < minimum threshold of 10).

---

## Dataset

### Source

- **Input**: `experiments/utp_motif_coverage/output/motif_patterns.csv`
- **Sequences**: `data/Import_candidates.fasta` (933 HMM-predicted proteins)
- **Annotations**: `data/Bbigelowii_transcriptome_annotations.csv`

### Filtering Criteria

1. Valid terminal motif (4, 5, 7, or 9): 607 proteins
2. Minimum class size ≥10 for statistical validity
3. Valid mature domain length (30-3000 aa)

### Expected Class Distribution

| Terminal Class | Count | Percentage |
| -------------- | ----- | ---------- |
| terminal_7     | ~460  | ~76%       |
| terminal_5     | ~74   | ~12%       |
| terminal_4     | ~68   | ~11%       |
| terminal_9     | ~5    | ~1%        |

**Note**: terminal_9 may be excluded due to insufficient samples.

---

## Methodology

### Statistical Framework

This experiment implements state-of-the-art academic statistical techniques:

#### 1. Nested Cross-Validation

```
┌────────────────────────────────────────────────────────────────────────────┐
│  NESTED CROSS-VALIDATION WITH STRATIFIED SAMPLING                          │
│                                                                            │
│  Full Dataset (N proteins)                                                 │
│  └── Outer Loop: 5-Fold Stratified CV (performance estimation)             │
│      └── Inner Loop: 3-Fold Stratified CV (hyperparameter tuning)          │
│                                                                            │
│  Benefits:                                                                 │
│  - Unbiased performance estimates                                          │
│  - No data leakage between tuning and evaluation                           │
│  - Proper handling of class imbalance via stratification                   │
└────────────────────────────────────────────────────────────────────────────┘
```

#### 2. Class Imbalance Handling

- **Stratified sampling**: Maintains class proportions in all splits
- **Class weights**: `class_weight='balanced'` for all classifiers
- **SMOTE**: Synthetic Minority Over-sampling Technique (optional)
- **Evaluation metric**: Balanced accuracy (macro-averaged recall)

#### 3. Statistical Significance Testing

| Test             | Purpose                    | Implementation            |
| ---------------- | -------------------------- | ------------------------- |
| Permutation test | Overall model significance | 1000 label permutations   |
| McNemar's test   | Classifier comparison      | Pairwise model comparison |
| Binomial test    | Per-class vs chance        | Bonferroni-corrected      |
| Bootstrap CI     | Confidence intervals       | 1000 bootstrap samples    |
| Repeated CV      | Variance estimation        | 10×5-fold repeated CV     |

#### 4. Multiple Testing Correction

- Bonferroni correction for per-class tests
- Benjamini-Hochberg FDR for enrichment analyses
- Holm-Šidák for classifier comparisons

### Classifiers

| Classifier          | Hyperparameters                            | Rationale              |
| ------------------- | ------------------------------------------ | ---------------------- |
| Logistic Regression | C: [0.001, 0.01, 0.1, 1, 10]               | Interpretable baseline |
| SVM (RBF)           | C: [0.1, 1, 10], γ: [scale, auto]          | Non-linear boundaries  |
| Random Forest       | n_trees: [100, 200], depth: [10, 20, None] | Feature importance     |
| XGBoost             | lr: [0.01, 0.1], depth: [3, 5, 7]          | State-of-the-art       |
| Gradient Boosting   | lr: [0.01, 0.1], depth: [3, 5]             | Robust ensemble        |

### Feature Engineering

1. **ProtT5 Embeddings** (1024 dimensions)

   - Mean-pooled protein language model representations
   - Captures evolutionary and structural information

2. **Biochemical Properties** (14 features)

   - Length, molecular weight, isoelectric point
   - GRAVY, instability index
   - Secondary structure fractions (helix, sheet, coil)
   - Amino acid composition groups

3. **Feature Selection** (optional)
   - Recursive Feature Elimination (RFE)
   - Permutation importance filtering

---

## Pipeline

```bash
# Step 1: Data preparation - load extended dataset, filter, extract sequences
uv run python experiments/utp_variant_classifier_extended/01_prepare_data.py

# Step 2: Feature extraction - ProtT5 embeddings + biochemical properties
uv run python experiments/utp_variant_classifier_extended/02_extract_features.py

# Step 3: Train classifiers with rigorous nested CV and significance testing
uv run python experiments/utp_variant_classifier_extended/03_train_classifier.py

# Step 4: Interpretability analysis and feature importance
uv run python experiments/utp_variant_classifier_extended/04_interpretability.py
```

---

## Expected Outputs

```
output/
├── data/
│   ├── processed_proteins.csv          # Protein metadata + terminal class
│   ├── mature_domains.fasta            # Extracted mature domains
│   ├── utp_regions.fasta               # Extracted uTP regions
│   └── class_distribution.csv          # Class counts and proportions
├── features/
│   ├── embeddings.h5                   # ProtT5 embeddings
│   └── biochemical_features.csv        # Calculated properties
├── models/
│   ├── best_model.joblib               # Final trained model
│   ├── cv_results.csv                  # Cross-validation results
│   └── classifier_comparison.csv       # All classifier results
├── statistics/
│   ├── permutation_test.csv            # Significance test results
│   ├── bootstrap_ci.csv                # Confidence intervals
│   ├── mcnemar_tests.csv               # Classifier comparisons
│   ├── binomial_per_class.csv          # Per-class significance
│   └── repeated_cv_variance.csv        # CV variance estimates
└── figures/
    ├── confusion_matrix.svg            # Normalized confusion matrix
    ├── roc_curves.svg                  # One-vs-rest ROC curves
    ├── precision_recall_curves.svg     # PR curves (better for imbalanced)
    ├── cv_comparison.svg               # Classifier comparison boxplot
    ├── permutation_distribution.svg    # Null distribution
    ├── feature_importance.svg          # Top predictive features
    ├── umap_visualization.svg          # UMAP embedding by class
    └── calibration_curves.svg          # Probability calibration
```

---

## Comparison with Original Experiment

| Aspect              | Original (4 classes) | Extended (3 classes)    |
| ------------------- | -------------------- | ----------------------- |
| Dataset size        | 182 proteins         | 573 proteins            |
| Classes             | 4 (terminal motif)   | 3 (terminal_9 excluded) |
| Balanced Accuracy   | 50.9%                | 51.8%                   |
| Chance Level        | 25.0%                | 33.3%                   |
| vs Chance           | 2.04×                | 1.55×                   |
| Permutation p-value | 0.002                | 0.001                   |
| 95% CI              | -                    | [46.5%, 57.6%]          |
| Validation          | Train/test split     | Nested 5-fold CV        |

### Interpretation

- **Both experiments show statistically significant classification** (p < 0.01)
- The extended dataset confirms the original finding with 3× more data
- Performance relative to chance is slightly lower (1.55× vs 2.04×) due to:
  - Fewer classes (3 vs 4) → higher chance baseline
  - Severe class imbalance (76.6% terminal_7)
- **Conclusion**: The mature domain sequence does predict terminal motif class, but the signal is modest

---

## Limitations

1. **Severe class imbalance**: terminal_7 dominates (76.6%), making minority class prediction difficult
2. **terminal_9 excluded**: Only 5 samples, insufficient for statistical analysis
3. **Per-class significance**: Only terminal_7 significantly above chance (Bonferroni-corrected)
4. **Biochemical properties**: Only molecular weight differs significantly between classes
5. **Feature importance**: Individual features show very small importance values

---

## References

- Coale et al. (2024) - Nitrogen-fixing organelle discovery
- Elnaggar et al. (2021) - ProtT5 protein language model
- Cawley & Talbot (2010) - Nested cross-validation
- Bouckaert & Frank (2004) - Evaluating classifier significance
- Demšar (2006) - Statistical comparisons of classifiers

---

_Created: 2026-01-13_
