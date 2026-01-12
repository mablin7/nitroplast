# uTP Variant Classifier

**Rigorous multiclass classification of uTP motif variants from mature domain sequences.**

## Research Question

Can the mature domain of a protein predict which uTP variant (motif combination) it will receive?

**Hypothesis**: If different uTP variants encode sub-organellar localization information, we would expect:

1. Mature domains to predict uTP variants better than chance
2. Functional enrichment within uTP variant classes
3. Distinct biochemical properties per variant class

---

## 📊 Experimental Results (Run: 2026-01-12)

### 🔑 Key Result: Grouping Strategy Comparison

We compared three variant grouping strategies to maximize statistical power:

| Strategy              | N       | Classes | Accuracy  | Chance | **vs Chance** | p-value      |
| --------------------- | ------- | ------- | --------- | ------ | ------------- | ------------ |
| **Terminal Motif**    | **182** | 4       | **50.9%** | 25.0%  | **2.04×**     | **0.002** ✓✓ |
| **Fine-Grained (≥5)** | 135     | 8       | **52.6%** | 12.5%  | **4.21×**     | **0.002** ✓✓ |
| Original (≥10)        | 109     | 4       | 36.9%     | 25.0%  | 1.48×         | 0.012 ✓      |

**All strategies are statistically significant (p < 0.05)**

#### Strategy Descriptions

1. **Terminal Motif Grouping**: Groups variants by their terminal motif (4, 5, 7, or 9)

   - Uses 182 proteins with valid motif patterns
   - Classes: terminal_4 (87), terminal_7 (54), terminal_5 (25), terminal_9 (16)

2. **Fine-Grained (MIN_COUNT≥5)**: Keeps exact motif patterns with ≥5 samples

   - 8 distinct variant classes
   - 135 proteins retained

3. **Original (MIN_COUNT≥10)**: Strict filtering
   - 109 proteins retained
   - 4 variant classes

### Dataset Summary

| Metric           | Original | Terminal Motif | Fine-Grained |
| ---------------- | -------- | -------------- | ------------ |
| Total proteins   | 206      | 182            | 135          |
| Variant classes  | 4        | 4              | 8            |
| Min class size   | 18       | 16             | 5            |
| Data utilization | 53%      | **88%**        | 65%          |

### Terminal Motif Class Distribution

| Class      | Terminal Motif    | Count | Example Variants             |
| ---------- | ----------------- | ----- | ---------------------------- |
| terminal_4 | ends with motif_4 | 87    | 2+1+3+4, 2+1+6+3+4, 2+1+10+4 |
| terminal_7 | ends with motif_7 | 54    | 2+1+3+5+7, 2+1+6+3+5+7       |
| terminal_5 | ends with motif_5 | 25    | 2+1+3+5, 2+1+10+5            |
| terminal_9 | ends with motif_9 | 16    | 2+1+3+5+9, 2+1+10+5+9        |

### Fine-Grained Class Distribution (8 classes)

| Variant     | Motif Pattern               | Count |
| ----------- | --------------------------- | ----- |
| 2+1+3+4     | motif_2 → 1 → 3 → 4         | 43    |
| 2+1+6+3+4   | motif_2 → 1 → 6 → 3 → 4     | 29    |
| 2+1+3+5     | motif_2 → 1 → 3 → 5         | 19    |
| 2+1+3+5+7   | motif_2 → 1 → 3 → 5 → 7     | 18    |
| 2+1+3+7     | motif_2 → 1 → 3 → 7         | 9     |
| 2+1+10+5+9  | motif_2 → 1 → 10 → 5 → 9    | 6     |
| 2+1+10+5+7  | motif_2 → 1 → 10 → 5 → 7    | 6     |
| 2+1+6+3+5+7 | motif_2 → 1 → 6 → 3 → 5 → 7 | 5     |

### Biochemical Property Differences

| Property       | KW p-value | Significant |
| -------------- | ---------- | ----------- |
| fraction_helix | 0.033      | ✓           |
| fraction_coil  | 0.049      | ✓           |
| gravy          | 0.054      | -           |

### Functional Annotation Correlation

Functional annotations were analyzed using a two-stage approach to control for baseline uTP enrichment.

#### Annotation Coverage

| Annotation Type | Coverage        |
| --------------- | --------------- |
| COG categories  | 130/182 (71.4%) |
| GO terms        | 34/182 (18.7%)  |
| KEGG KO         | 81/182 (44.5%)  |
| KEGG Pathway    | 46/182 (25.3%)  |

#### Stage 1: uTP vs Transcriptome Background

Comparison of 182 uTP proteins against 19,395 non-uTP transcriptome proteins.

| COG Category             | OR   | 95% CI       | log2 FE | q-value | Direction |
| ------------------------ | ---- | ------------ | ------- | ------- | --------- |
| K: Transcription         | 0.11 | [0.02, 0.55] | -2.90   | 0.026   | depleted  |
| U: Intracellular traffic | 0.11 | [0.02, 0.55] | -2.92   | 0.026   | depleted  |
| Z: Cytoskeleton          | 0.00 | [0.06, 16.1] | -5.13   | 0.026   | depleted  |

uTP proteins are significantly depleted (not enriched) for transcription, trafficking, and cytoskeleton functions compared to transcriptome background. No COG categories are significantly enriched in uTP proteins.

#### Stage 2: Terminal Motif Specificity (uTP-corrected)

Comparison of each terminal motif class against other uTP proteins (controlling for uTP baseline).

**Result**: 0 significant terminal-specific enrichments after FDR correction (q < 0.1).

The chi-square tests (below) detect overall heterogeneity, but individual pairwise enrichments do not survive multiple testing correction.

#### Overall Association Tests (Chi-square)

| Annotation Type | χ²     | df   | p-value |
| --------------- | ------ | ---- | ------- |
| COG categories  | 93.6   | 54   | 0.0007  |
| KEGG KO         | 310.0  | 210  | <0.0001 |
| KEGG Pathway    | 760.8  | 615  | 0.0001  |
| GO terms        | 3484.3 | 3513 | 0.63    |

#### COG Category Distribution by Terminal Motif (%)

| COG Category               | T4 (n=87) | T5 (n=25) | T7 (n=54) | T9 (n=16) |
| -------------------------- | --------- | --------- | --------- | --------- |
| C: Energy production       | 11.5      | 0.0       | 1.9       | 12.5      |
| E: Amino acid metabolism   | 0.0       | 8.0       | 0.0       | 0.0       |
| G: Carbohydrate metabolism | 13.8      | 4.0       | 11.1      | 0.0       |
| H: Coenzyme metabolism     | 1.1       | 4.0       | 0.0       | 12.5      |
| L: Replication/repair      | 6.9       | 4.0       | 16.7      | 6.2       |
| M: Cell wall/membrane      | 4.6       | 8.0       | 5.6       | 0.0       |
| O: PTM/chaperones          | 6.9       | 8.0       | 13.0      | 6.2       |
| S: Unknown function        | 24.1      | 32.0      | 16.7      | 25.0      |
| T: Signal transduction     | 0.0       | 4.0       | 11.1      | 0.0       |

---

## Methodology

### Experimental Design

```
┌────────────────────────────────────────────────────────────────────────────┐
│  NESTED CROSS-VALIDATION WITH HELD-OUT TEST SET                            │
│                                                                            │
│  Full Dataset (N proteins)                                                 │
│  ├── Held-out Test Set (20%) ← Final evaluation only                       │
│  └── Training/Validation Pool (80%)                                        │
│      └── 5-Fold Stratified CV ← Hyperparameter tuning + model selection    │
│                                                                            │
│  Statistical Validation:                                                   │
│  - Permutation test (1000 iterations) on held-out predictions              │
│  - Exact binomial test vs random baseline                                  │
│  - Bootstrap confidence intervals                                          │
└────────────────────────────────────────────────────────────────────────────┘
```

### Key Methodological Improvements

1. **HMM-based uTP Detection**: Precise boundary detection using profile HMM
2. **Full Dataset**: All proteins from `Import_candidates.fasta` with valid uTP regions
3. **Nested CV**: Prevents information leakage during hyperparameter tuning
4. **Multiple Classifiers**: Ensemble of Logistic Regression, SVM, Random Forest, XGBoost
5. **Rich Feature Set**: ProtT5 embeddings + biochemical properties
6. **Rigorous Statistics**: Effect sizes, confidence intervals, multiple testing correction

### Statistical Framework

| Test             | Purpose                              | Threshold            |
| ---------------- | ------------------------------------ | -------------------- | --- | ----- |
| Permutation test | Overall model significance           | p < 0.05             |
| Binomial test    | Per-class performance vs chance      | p < 0.05 / n_classes |
| Bootstrap CI     | Confidence intervals for metrics     | 95% CI               |
| Cohen's d        | Effect size for property differences |                      | d   | > 0.5 |
| FDR correction   | GO/KEGG enrichment                   | q < 0.1              |

## Pipeline

```bash
# Step 1: Data preparation - detect uTP regions and assign motif variants
uv run python experiments/utp_variant_classifier/01_prepare_data.py

# Step 2: Feature extraction - ProtT5 embeddings + biochemical properties
uv run python experiments/utp_variant_classifier/02_extract_features.py

# Step 3: Train classifiers with nested CV
uv run python experiments/utp_variant_classifier/03_train_classifier.py

# Step 4: Annotation enrichment analysis per variant
uv run python experiments/utp_variant_classifier/04_annotation_analysis.py

# Step 5: Interpretability and visualization
uv run python experiments/utp_variant_classifier/05_interpretability.py

# Step 6: Compare grouping strategies (Terminal Motif vs Fine-Grained)
uv run python experiments/utp_variant_classifier/compute_additional_embeddings.py  # Compute embeddings for all proteins
uv run python experiments/utp_variant_classifier/compare_groupings.py              # Compare strategies

# Step 7: Functional annotation correlation analysis
uv run python experiments/utp_variant_classifier/analyze_functional_correlations.py

# Step 8: Relative enrichment analysis (controlling for uTP baseline)
uv run python experiments/utp_variant_classifier/analyze_relative_enrichment.py
```

## Data Sources

- `data/Import_candidates.fasta` - Full set of uTP-tagged proteins
- `data/Bbigelowii_transcriptome_annotations.csv` - EggNOG annotations (GO, KEGG, COG)
- `experiments/utp_homolog_search/utp.hmm` - Profile HMM for uTP detection
- `experiments/utp_motif_analysis/data/meme_gb.xml` - Motif definitions from MEME

## Generated Outputs

```
output/
├── data/
│   ├── processed_proteins.csv          # Protein info + motif assignments
│   ├── mature_domains.fasta            # Extracted mature domains
│   └── utp_regions.fasta               # Extracted uTP regions
├── features/
│   ├── embeddings.h5                   # ProtT5 embeddings (108 proteins, original)
│   ├── embeddings_all.h5               # ProtT5 embeddings (182 proteins, all)
│   ├── biochemical_features.csv        # 14 calculated properties (108)
│   └── biochemical_features_all.csv    # 14 calculated properties (182)
├── models/
│   └── best_model.joblib               # Logistic Regression (C=0.01, balanced)
├── comparison/                         # Strategy comparison results
│   ├── strategy_comparison.csv         # Comparison metrics
│   ├── strategy_comparison.svg         # Comparison visualization
│   └── strategy_comparison.png
├── functional_analysis/                # Functional annotation analysis
│   ├── variant_annotation_summary.csv  # Summary by terminal motif
│   ├── cog_enrichment_results.csv      # COG Fisher's exact tests
│   ├── go_enrichment_results.csv       # GO Fisher's exact tests
│   ├── kegg_enrichment_results.csv     # KEGG Fisher's exact tests
│   ├── chi_square_results.csv          # Overall association tests
│   ├── cog_by_terminal_motif.svg       # COG distribution heatmap
│   └── kegg_by_terminal_motif.svg      # KEGG distribution heatmap
├── relative_enrichment/                # Baseline-corrected enrichment
│   ├── stage1_utp_vs_background.csv    # uTP vs transcriptome
│   ├── stage2_terminal_specificity.csv # Terminal-specific (corrected)
│   ├── stage1_utp_vs_background.svg    # Stage 1 visualization
│   └── stage2_terminal_specificity.svg # Stage 2 heatmap
├── statistics/
│   ├── classification_report.csv       # Per-class metrics
│   ├── permutation_test_results.csv    # Permutation test results
│   ├── permutation_importance.csv      # Feature importance ranking
│   ├── class_separability.csv          # Variant distinctiveness scores
│   ├── go_enrichment.csv               # GO term enrichment results
│   └── cog_enrichment.csv              # COG category enrichment results
└── figures/
    ├── confusion_matrix.svg            # Test set confusion matrix
    ├── roc_curves.svg                  # One-vs-rest ROC curves
    ├── cv_comparison.svg               # Cross-validation results by classifier
    ├── permutation_test.svg            # Null distribution visualization
    ├── tsne_visualization.svg          # t-SNE embedding visualization
    ├── umap_visualization.svg          # UMAP embedding visualization
    ├── variant_distribution.svg        # Class balance visualization
    ├── length_distributions.svg        # Mature/uTP length distributions
    ├── property_distributions_by_variant.svg   # Biochemical property violin plots
    ├── biochem_property_distributions.svg      # Property comparisons with stats
    ├── biochem_feature_importance.svg          # Feature importance (biochemical)
    ├── model_feature_importance.svg            # Feature importance (all)
    ├── class_separability.svg          # Variant separability analysis
    ├── annotation_coverage.svg         # GO/COG/KEGG coverage
    ├── cog_distribution.svg            # COG category distribution
    └── go_enrichment_heatmap.svg       # GO enrichment visualization
```

## Summary of Results

### Classification Performance

| Strategy                   | Accuracy | Chance | Ratio | p-value |
| -------------------------- | -------- | ------ | ----- | ------- |
| Terminal Motif (4 classes) | 50.9%    | 25.0%  | 2.04× | 0.002   |
| Fine-Grained (8 classes)   | 52.6%    | 12.5%  | 4.21× | 0.002   |
| Original (4 classes)       | 36.9%    | 25.0%  | 1.48× | 0.012   |

### Biochemical Properties

Kruskal-Wallis tests identified significant differences in secondary structure composition between terminal motif classes:

- fraction_helix: p = 0.033
- fraction_coil: p = 0.049

### Functional Annotations

Two-stage analysis controlling for uTP baseline:

**Stage 1 (uTP vs transcriptome)**: uTP proteins are depleted for:

- K: Transcription (OR=0.11, q=0.026)
- U: Intracellular trafficking (OR=0.11, q=0.026)
- Z: Cytoskeleton (OR=0.00, q=0.026)

**Stage 2 (Terminal-specific vs other uTP)**: 0 significant enrichments after FDR correction.

Chi-square tests detect overall heterogeneity (COG p=0.0007) but individual comparisons do not survive multiple testing correction.

### Limitations

1. Class imbalance: terminal_4 (87) vs terminal_9 (16)
2. Low GO term coverage (18.7%)
3. No terminal-specific functional enrichments survive FDR correction when controlling for uTP baseline
4. Chi-square tests detect heterogeneity but individual comparisons are underpowered

## References

- Coale et al. (2024) - Nitrogen-fixing organelle discovery
- ProtT5: Elnaggar et al. (2021) - Protein language models
- Nested CV: Cawley & Talbot (2010) - Model selection bias

---

_Last updated: 2026-01-12_
