# uTP Protein Functional Annotation Analysis

## Overview

This experiment addresses a critical question about the distinctive biophysical properties of uTP proteins discovered in the `utp_presence_classifier` experiment:

**Are the biophysical signatures (more disordered, more acidic, more stable) of uTP proteins explained by functional enrichment, or are they intrinsic properties selected for import?**

## Key Research Questions

1. Are uTP proteins functionally enriched for certain categories (e.g., enzymes)?
2. Do the distinctive biophysical properties persist when comparing uTP vs control proteins **within the same functional category**?
3. How much variance in biophysical properties is explained by function vs uTP status?

See `EXPERIMENT_DESIGN.md` for the full experimental design and statistical framework.

---

## Part 1: Annotation Coverage Analysis

### Data Sources

1. **High-confidence uTP proteins** (`ucyna_enriched_good_c_align_full_sequences.fasta`): 206 proteins with good C-terminal alignment
2. **Import candidates** (`Import_candidates.fasta`): 933 proteins predicted to contain uTP by HMM search
3. **Functional annotations** (`Bbigelowii_transcriptome_annotations.csv`): eggNOG-mapper annotations

### Annotation Coverage Results

| Dataset             | Total  | Annotated | Rate  |
| ------------------- | ------ | --------- | ----- |
| High-confidence uTP | 206    | 175       | 85.0% |
| Import candidates   | 933    | 667       | 71.5% |
| All B. bigelowii    | 19,574 | 19,574    | 100%  |

### COG Category Enrichment (Preliminary)

Top enriched categories in uTP proteins:

| COG | Description                   | Fold Change |
| --- | ----------------------------- | ----------- |
| M   | Cell wall/membrane biogenesis | 2.42x       |
| L   | Replication/repair            | 1.85x       |
| G   | Carbohydrate metabolism       | 1.83x       |
| C   | Energy production             | 1.78x       |

Top depleted categories:

| COG | Description           | Fold Change |
| --- | --------------------- | ----------- |
| K   | Transcription         | 0.12x       |
| U   | Trafficking/secretion | 0.12x       |

---

## Part 2: Function vs Properties Analysis

### Hypotheses

**Hypothesis A (Confounding)**: uTP biophysical properties are explained by functional enrichment

- e.g., uTP proteins are mostly enzymes, and enzymes tend to be more disordered

**Hypothesis B (Selection)**: uTP properties are intrinsic, regardless of function

- Suggests selection for import-compatible properties

### Analysis Strategy

1. **Within-category comparison**: Compare properties within each COG category
2. **ANCOVA**: `property ~ uTP_status + COG_category`
3. **Variance partitioning**: Quantify variance explained by function vs uTP
4. **Matched-pairs analysis**: Match by function, compare properties

---

## Scripts

### Part 1: Annotation Coverage

| Script                      | Description                              |
| --------------------------- | ---------------------------------------- |
| `00_annotation_coverage.py` | Overall annotation coverage analysis     |
| `01_cog_distribution.py`    | COG category distribution and enrichment |

### Part 2: Function vs Properties (Main Analysis)

| Script                           | Description                                   |
| -------------------------------- | --------------------------------------------- |
| `02_prepare_for_annotation.py`   | Prepare mature domains for annotation         |
| `03_parse_annotations.py`        | Parse eggNOG results, merge with properties   |
| `04_functional_enrichment.py`    | Fisher's exact test for COG enrichment        |
| `05_within_category_analysis.py` | **KEY**: Compare properties within categories |
| `06_ancova_analysis.py`          | ANCOVA and variance partitioning              |

---

## Workflow

```bash
# Part 1: Initial coverage analysis (already complete)
uv run python experiments/utp_functional_annotation/00_annotation_coverage.py
uv run python experiments/utp_functional_annotation/01_cog_distribution.py

# Part 2: Main analysis
# Step 1: Prepare data (requires utp_presence_classifier output)
uv run python experiments/utp_functional_annotation/02_prepare_for_annotation.py

# Step 2: Submit sequences to eggNOG-mapper OR use existing annotations
# Option A: Submit output/sequences_for_eggnog.fasta to https://eggnog-mapper.embl.de/
# Option B: Script will fall back to existing annotations

# Step 3: Parse annotations
uv run python experiments/utp_functional_annotation/03_parse_annotations.py

# Step 4: Functional enrichment
uv run python experiments/utp_functional_annotation/04_functional_enrichment.py

# Step 5: Within-category analysis (THE KEY TEST)
uv run python experiments/utp_functional_annotation/05_within_category_analysis.py

# Step 6: ANCOVA and variance partitioning
uv run python experiments/utp_functional_annotation/06_ancova_analysis.py
```

---

## Output Files

### Coverage Analysis

- `annotation_coverage_summary.csv` / `annotation_coverage.png`
- `cog_category_distribution.csv` / `cog_distribution.png`

### Function vs Properties Analysis

- `merged_data.csv` - Combined annotations and properties
- `functional_enrichment.csv` / `functional_enrichment.png` - Fisher's exact tests
- `within_category_results.csv` / `within_category_analysis.png` - Within-category effects
- `meta_analysis_results.csv` - Effect sizes across categories
- `ancova_results.csv` / `ancova_analysis.png` - ANCOVA results
- `variance_partitioning.csv` - Variance explained by function vs uTP

---

---

## Key Results

### ✅ HYPOTHESIS B SUPPORTED: Properties are intrinsic to uTP proteins

**The biophysical properties persist despite controlling for function.**

#### Within-Category Analysis (Critical Test)

Analysis restricted to COG categories with n≥10 samples in both groups (8 categories).

| Property          | Overall d | Within-Cat d | % Explained (raw) | I² Heterogeneity | Direction Consistency |
| ----------------- | --------- | ------------ | ----------------- | ---------------- | --------------------- |
| fraction_coil     | +0.96     | +0.98        | -1.7%             | 0.0%             | 8/8 (100%)            |
| instability_index | -0.76     | -0.80        | -5.7%             | 20.5%            | 8/8 (100%)            |
| isoelectric_point | -0.66     | -0.55        | +17.2%            | 46.1%            | 8/8 (100%)            |
| gravy             | +0.35     | +0.40        | -15.5%            | 51.2%            | 7/8 (88%)             |

**Note**: Negative "% explained" values indicate within-category effects are *larger* than overall effects. This occurs when controlling for function reveals a stronger uTP signal.

**Interpretation**: Effect sizes are essentially unchanged (or even increased) when comparing uTP vs control proteins _within the same functional category_. All 8 categories show the same direction for the three key properties. This rules out functional enrichment as the explanation.

#### Sensitivity Analysis (Excluding "Function Unknown")

Excluding COG category S ("Function unknown", which is not a true functional category):

| Property          | Within-Cat d (all) | Within-Cat d (excl S) | % Explained (excl S) |
| ----------------- | ------------------ | --------------------- | -------------------- |
| fraction_coil     | +0.98              | +1.04                 | -8.6%                |
| instability_index | -0.80              | -0.81                 | -6.9%                |
| isoelectric_point | -0.55              | -0.52                 | +21.9%               |
| gravy             | +0.40              | +0.41                 | -19.8%               |

Effects persist (or strengthen) when excluding the ambiguous "Function unknown" category.

#### Variance Partitioning

| Component       | Variance Explained |
| --------------- | ------------------ |
| uTP unique      | 7.3%               |
| Function unique | 5.4%               |
| Shared          | 3.4%               |
| Unexplained     | 83.9%              |

**Interpretation**: uTP status explains **more unique variance** (7.3%) than is shared with function (3.4%). The biophysical signature is genuinely associated with uTP status, not confounded by function.

#### Matched-Pairs Analysis

Comparing uTP vs control proteins **with identical COG categories** using random sampling without replacement:

| Property          | n pairs | Paired d | Independent d | p-value  |
| ----------------- | ------- | -------- | ------------- | -------- |
| fraction_coil     | 209     | +0.60    | +0.85         | <0.0001  |
| isoelectric_point | 209     | -0.48    | -0.67         | <0.0001  |
| instability_index | 209     | -0.46    | -0.60         | <0.0001  |
| gravy             | 209     | +0.22    | +0.30         | 0.008    |

**Methodological note**: "Paired d" uses mean difference / SD of differences (more conservative). "Independent d" uses pooled SD (comparable to within-category analysis). Both are reported for transparency.

**All 4 properties remain significantly different** even when comparing same-function proteins.

---

## Biological Interpretation

The distinctive biophysical properties of uTP proteins are **NOT** explained by functional enrichment. They appear to be **intrinsic characteristics selected for import compatibility**:

1. **More disordered (fraction_coil)**: May facilitate unfolding during membrane translocation
2. **More acidic (isoelectric_point)**: Could interact with positively charged import machinery
3. **More stable (instability_index)**: May be selected for function in the nitroplast environment

These properties likely represent **evolutionary constraints** on proteins that must be:

- Translated in the host cytoplasm
- Recognized by the uTP import system
- Translocated across membranes
- Functional inside the nitroplast

---

## Statistical Notes

### Effect Size Interpretation (Cohen's d)
- Small: |d| = 0.2
- Medium: |d| = 0.5
- Large: |d| = 0.8

### Heterogeneity (I²)
- Low: I² < 25%
- Moderate: 25% ≤ I² < 75%
- High: I² ≥ 75%

The low-to-moderate heterogeneity observed (I² = 0-51%) indicates that effects are reasonably consistent across functional categories.

---

## Interpretation Guide

### If effect persists within categories (Hypothesis B): ✅ OBSERVED

- Effect sizes remain large (d > 0.5) within functional categories
- ANCOVA shows significant uTP effect after controlling for function
- **Interpretation**: uTP proteins are selected for import-compatible properties

### If effect disappears (Hypothesis A): ❌ NOT OBSERVED

- Effect sizes near zero within categories
- Function explains most of the variance
- **Interpretation**: Properties are a byproduct of functional enrichment

### Mixed results:

- Some reduction in effect size but still significant
- Both function and uTP status contribute
- **Interpretation**: Partial confounding + some selection
