# uTP Presence Classifier

Binary classifier to predict whether a protein should have a uTP (UCYN-A transit peptide) based on its mature domain sequence.

## Key Results

**The classifier achieves 88-93% accuracy in predicting uTP presence from mature domain sequences alone.**

### Experiment 1: Balanced Dataset (117 vs 117)

| Classifier          | Accuracy     | F1 Score |
| ------------------- | ------------ | -------- |
| SVC                 | 87.7% ± 3.7% | 0.87     |
| Logistic Regression | 87.0% ± 4.0% | 0.87     |
| Random Forest       | 85.5% ± 3.4% | 0.85     |

### Experiment 2: Full Dataset (605 vs 773)

| Classifier          | Accuracy  | F1 Score | p-value   |
| ------------------- | --------- | -------- | --------- |
| Logistic Regression | **92.8%** | 0.92     | **0.002** |
| SVC                 | 91.7%     | 0.90     | 0.002     |
| Random Forest       | 87.3%     | 0.85     | 0.002     |

### Interpretability: What Makes uTP Proteins Different?

Protein property analysis reveals **distinctive biophysical signatures** in uTP-containing proteins:

| Property              | Effect Size (Cohen's d) | Direction     | Interpretation            |
| --------------------- | ----------------------- | ------------- | ------------------------- |
| **fraction_coil**     | **+1.05** (large)       | Higher in uTP | More disordered regions   |
| **isoelectric_point** | **-0.89** (large)       | Lower in uTP  | More acidic               |
| **instability_index** | **-0.81** (large)       | Lower in uTP  | More stable               |
| gravy                 | +0.45 (medium)          | Higher in uTP | Slightly more hydrophobic |

All p-values < 0.006 (Bonferroni-corrected for 8 tests).

**Biological interpretation**: uTP proteins have a distinctive profile:

- **More disordered** – could facilitate unfolding during import
- **More acidic** – might interact with positively charged import machinery
- **More stable** – selected for function in the nitroplast environment

### Embedding Visualization

| Method | Silhouette Score | Interpretation      |
| ------ | ---------------- | ------------------- |
| t-SNE  | 0.226 ± 0.003    | Weak separation     |
| UMAP   | 0.282 ± 0.014    | Moderate separation |

The classes overlap partially in embedding space, but enough structure exists for ~90% classification accuracy.

---

## Full Proteome Validation

Validation on the **entire B. bigelowii proteome** (933 uTP vs 43,430 non-uTP proteins):

### Classification Performance

| Metric    | Value     |
| --------- | --------- |
| Accuracy  | 89.1%     |
| Precision | 13.9%     |
| Recall    | 80.3%     |
| F1 Score  | 0.24      |
| ROC AUC   | **0.920** |

**Note**: The low precision/high recall reflects the extreme class imbalance (1:46 ratio). The classifier correctly identifies 80% of uTP proteins while maintaining 89% overall accuracy. The high ROC AUC (0.920) confirms strong discriminative power.

### Property Differences (Full Proteome)

| Property          | Effect Size (d) | Direction     |
| ----------------- | --------------- | ------------- |
| isoelectric_point | -0.68 (medium)  | Lower in uTP  |
| molecular_weight  | +0.52 (medium)  | Higher in uTP |
| length            | +0.51 (medium)  | Higher in uTP |
| instability_index | -0.33 (small)   | Lower in uTP  |
| fraction_coil     | +0.27 (small)   | Higher in uTP |
| fraction_helix    | +0.22 (small)   | Higher in uTP |

### Embedding Visualization (Full Proteome)

| Method | Silhouette Score |
| ------ | ---------------- |
| t-SNE  | 0.092            |
| UMAP   | 0.270            |

---

## Workflow

The analysis is split into multiple steps to allow for manual use of web services:

```
┌─────────────────────────────────────────────────────────────────┐
│  01_prepare_data.py                                              │
│  - Extract mature domains from uTP proteins (HMM-based)          │
│  - Select length-matched control candidates from proteome        │
│  - Output: mature_domains.fasta, control_candidates.fasta        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  CELLO Web Service (manual step)                                 │
│  - Upload control_candidates.fasta                               │
│  - Download results                                              │
│  - http://cello.life.nctu.edu.tw/                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  02_filter_controls.py                                           │
│  - Parse CELLO results                                           │
│  - Filter for cytoplasmic/nuclear proteins                       │
│  - Output: filtered_controls_*.fasta                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  03_train_classifier.py                                          │
│  - Compute ProtT5 embeddings (cached)                            │
│  - Run TWO experiments (balanced + full dataset)                 │
│  - Output: classifiers, ROC curves, confusion matrices           │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Step 1: Prepare data
uv run python experiments/utp_presence_classifier/01_prepare_data.py

# Step 2: Upload control_candidates.fasta to CELLO
# http://cello.life.nctu.edu.tw/
# Save results to output/

# Step 3: Filter controls by localization
uv run python experiments/utp_presence_classifier/02_filter_controls.py

# Step 4: Train classifier (runs both experiments)
uv run python experiments/utp_presence_classifier/03_train_classifier.py
```

## Research Question

Can we predict whether a B. bigelowii protein will have a uTP targeting signal based solely on its mature domain (the functional part of the protein, excluding the uTP)?

**Answer: Yes, with ~90% accuracy.** This suggests that:

1. uTP-containing proteins share common sequence features
2. There may be co-evolution between the mature domain and the targeting signal
3. The uTP system may preferentially target certain types of proteins

## Key Innovations

### 1. HMM-based uTP Clipping

Instead of using a fixed cutoff (e.g., "remove last 150 aa"), this script uses the uTP HMM profile to precisely identify where the uTP region begins:

```
Full protein:  [-------- Mature Domain --------][-- Linker --][---- uTP ----]
                                                              ^
                                                         HMM detects this
```

### 2. Robust Control Group Selection

The control group is carefully selected to avoid confounding factors:

1. **Exclude known uTP proteins**
2. **Match length distribution** - Control proteins have similar lengths to uTP mature domains
3. **Filter by CELLO** - Excludes proteins with other targeting signals (signal peptides, mitochondrial/chloroplast transit peptides)

### 3. Two Experimental Designs

To ensure robust results, we run two experiments:

1. **Exp1 (Balanced)**: Downsample uTP proteins to match cytoplasmic controls (117 vs 117), run 10 random folds
2. **Exp2 (Full)**: Use all uTP proteins vs nuclear+cytoplasmic controls (605 vs 773), with permutation testing

## Scripts

### `01_prepare_data.py`

Extracts mature domains and prepares control candidates:

- Uses HMM profile to detect uTP regions and extract mature domains
- Selects 2x control candidates (length-matched) for CELLO filtering
- Outputs FASTA files ready for next steps

**Outputs:**

- `mature_domains.fasta` - Extracted mature domains from uTP proteins (605 sequences)
- `control_candidates.fasta` - Candidates for localization filtering (1210 sequences)

### `02_filter_controls.py`

Processes CELLO results to filter control sequences:

- Parses CELLO tab-separated output
- Creates TWO control sets:
  - `filtered_controls_cytoplasmic.fasta` - Strictest (117 sequences)
  - `filtered_controls_nuclear_cytoplasmic.fasta` - Relaxed (773 sequences)

### `03_train_classifier.py`

Trains and evaluates binary classifiers:

- Computes ProtT5 embeddings ONCE for all sequences (cached in `embeddings.h5`)
- Runs **TWO experiments** with different control sets
- Trains Logistic Regression, Random Forest, SVC
- Evaluates with cross-validation and permutation tests

**Outputs:**

- `classifier_results_combined.csv` - Combined results from both experiments
- `exp1_fold_accuracies.svg` - Accuracy distribution across downsample folds
- `exp2_roc_curve.svg`, `exp2_confusion_matrix.svg` - Visualizations for Exp2
- `exp2_best_classifier.joblib` - Saved best classifier

### `04_interpretability.py`

Interpretability analysis with rigorous statistics:

- **Embedding visualization**: t-SNE and UMAP with multiple random seeds
- **Protein properties**: Length, MW, pI, GRAVY, instability index, secondary structure
- **Statistical tests**: Point-biserial correlation, Mann-Whitney U, Cohen's d, Cliff's delta
- **Multiple testing correction**: Bonferroni (α=0.05/8)

**Outputs:**

- `tsne_seed*.svg`, `umap_seed*.svg` - Embedding visualizations
- `property_violin_plots.svg` - Property distributions by class
- `property_correlations.svg` - Effect size bar charts
- `property_statistics.csv` - Full statistical results
- `interpretability_summary.txt` - Summary report

## CELLO Instructions

1. Go to http://cello.life.nctu.edu.tw/
2. Upload `control_candidates.fasta`
3. Select "Eukaryote" as organism type
4. Submit and wait for results
5. Save results to `output/` directory

## Interpretation

### Classifier performs well (accuracy ~90%, p < 0.05) ✓

The strong performance suggests that uTP-containing proteins share common sequence features in their mature domains. Possible explanations:

1. **Functional bias**: uTP proteins may be enriched for certain functional categories (metabolic enzymes filling UCYN-A gaps)
2. **Co-evolution**: The mature domain may have evolved alongside the uTP
3. **Bipartite signal**: Information in the mature domain may contribute to targeting specificity

## Technical Details

### HMM-based uTP Detection

The HMM profile (`utp.hmm`) was built from aligned C-terminal regions of uTP proteins.

```python
HMM_EVALUE_THRESHOLD = 0.01  # Lenient threshold for known uTP proteins
MIN_HMM_SCORE = 30.0         # Minimum bit score for a valid hit
MIN_HMM_COVERAGE_START = 50  # HMM match should start within first 50 positions
```

### ProtT5 Embeddings

ProtT5 embeddings are used as features:

- Model: `Rostlab/prot_t5_xl_uniref50`
- Output: 1024-dimensional mean-pooled embedding per sequence
- Captures evolutionary and structural information

## File Structure

```
experiments/utp_presence_classifier/
├── 01_prepare_data.py       # Extract mature domains, prepare controls
├── 02_filter_controls.py    # Filter by CELLO results
├── 03_train_classifier.py   # Train and evaluate classifiers (both experiments)
├── 04_interpretability.py   # Embedding visualization + property analysis
├── 05_full_proteome_analysis.py  # Full proteome validation
├── README.md
└── output/
    ├── mature_domains.fasta                      # 605 uTP mature domains
    ├── control_candidates.fasta                  # 1210 control candidates
    ├── filtered_controls_cytoplasmic.fasta       # 117 cytoplasmic controls
    ├── filtered_controls_nuclear_cytoplasmic.fasta # 773 nuclear+cytoplasmic controls
    ├── localization_summary.csv                  # CELLO localization distribution
    ├── embeddings.h5                             # Cached ProtT5 embeddings
    ├── classifier_results_combined.csv           # Results from both experiments
    ├── exp1_fold_accuracies.svg                  # Exp1: accuracy across folds
    ├── exp1_length_distribution.svg              # Exp1: length comparison
    ├── exp2_roc_curve.svg                        # Exp2: ROC curve
    ├── exp2_confusion_matrix.svg                 # Exp2: confusion matrix
    ├── exp2_length_distribution.svg              # Exp2: length comparison
    ├── exp2_best_classifier.joblib               # Saved best classifier
    ├── tsne_seed*.svg                            # t-SNE visualizations (3 seeds)
    ├── umap_seed*.svg                            # UMAP visualizations (3 seeds)
    ├── property_violin_plots.svg                 # Property distributions
    ├── property_correlations.svg                 # Effect size summary
    ├── property_statistics.csv                   # Full statistical results
    ├── interpretability_summary.txt              # Summary report
    └── full_proteome_analysis/                   # Full proteome validation
        ├── predictions.csv                       # 44k predictions with probabilities
        ├── all_protein_properties.csv            # Properties for all proteins
        ├── property_statistics.csv               # Statistical tests
        ├── property_violin_plots.svg             # Property distributions
        ├── confusion_matrix.svg                  # Classification results
        ├── roc_curve.svg                         # ROC curve (AUC=0.920)
        ├── prediction_distribution.svg           # Probability distributions
        ├── tsne.svg, umap.svg                    # Embedding visualizations
        └── summary.txt                           # Summary report
```
