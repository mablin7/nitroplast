# Experiment: Are uTP Biophysical Properties Explained by Functional Enrichment?

## Research Question

The uTP presence classifier identified distinctive biophysical signatures in uTP-containing proteins:

- **More disordered** (fraction_coil: Cohen's d = +1.05)
- **More acidic** (isoelectric_point: Cohen's d = -0.89)
- **More stable** (instability_index: Cohen's d = -0.81)

**Key question**: Are these properties:

- **Hypothesis A (Confounding)**: Explained by functional enrichment (e.g., uTP proteins are mostly enzymes, and enzymes happen to have these properties)
- **Hypothesis B (Selection)**: Intrinsic to uTP proteins regardless of function, suggesting selection for import-compatible properties

## Experimental Design

### Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: Data Preparation                                                   │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Extract mature domains using HMM (reuse from utp_presence_classifier)     │
│  • Run eggNOG-mapper on MATURE DOMAINS (not full proteins)                   │
│  • Get functional annotations for matched control proteins                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: Functional Enrichment Analysis                                     │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Compare COG category distributions (uTP vs control)                       │
│  • Fisher's exact test for each category                                     │
│  • Identify enriched/depleted functional categories                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: Within-Category Property Analysis                                  │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • For each COG category with n≥10 in both groups:                          │
│    - Compare biophysical properties (uTP vs control)                         │
│    - Mann-Whitney U test + Cohen's d                                         │
│  • Meta-analysis across categories                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: Statistical Control for Function                                   │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • ANCOVA: property ~ uTP_status + COG_category                              │
│  • Variance partitioning: how much explained by function vs uTP?             │
│  • Matched-pairs analysis: match by function, compare properties             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 5: Interpretation & Conclusions                                       │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • If effect persists after controlling for function → Hypothesis B          │
│  • If effect disappears → Hypothesis A                                       │
│  • Quantify proportion of effect explained by function                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 1: Data Preparation

#### 1.1 Extract Mature Domains

- Use HMM-based extraction from `utp_presence_classifier/01_prepare_data.py`
- Copy/link existing mature domains file if available
- Key: Use MATURE domains for annotation, not full proteins (avoids uTP region confounding annotation)

#### 1.2 Run eggNOG-mapper

Since eggNOG-mapper is a web service, we'll prepare FASTA files for submission:

- `mature_domains_for_annotation.fasta` - uTP mature domains
- `controls_for_annotation.fasta` - Length-matched controls

**eggNOG-mapper settings:**

- Database: eggNOG 5.0
- Taxonomic scope: Eukaryota
- Output: Full annotations (COG, GO, KEGG, EC)

#### 1.3 Alternative: Local HMMER-based COG Assignment

If web service is impractical, use local hmmsearch against COG HMM profiles.

### Phase 2: Functional Enrichment Analysis

For each COG category:

```
                    uTP proteins    Control proteins
Category X          a               b
Other categories    c               d

Fisher's exact test: OR = (a×d)/(b×c)
```

**Statistical framework:**

- Fisher's exact test for each category
- FDR correction (Benjamini-Hochberg)
- Report odds ratios and 95% CI

### Phase 3: Within-Category Property Analysis

**Critical test**: If uTP proteins are more disordered because they're enriched for enzymes, then:

- uTP enzymes should have SIMILAR disorder to non-uTP enzymes
- The effect should disappear when comparing within categories

**Analysis for each COG category with n≥10 in both groups:**

```python
for category in ['C', 'E', 'G', 'H', 'O', 'S', ...]:
    utp_in_category = utp_proteins[utp_proteins.cog == category]
    ctrl_in_category = controls[controls.cog == category]

    if len(utp_in_category) >= 10 and len(ctrl_in_category) >= 10:
        # Compare each property
        for prop in ['fraction_coil', 'isoelectric_point', 'instability_index']:
            stat, pval = mannwhitneyu(utp_in_category[prop], ctrl_in_category[prop])
            effect = cohens_d(utp_in_category[prop], ctrl_in_category[prop])
```

**Meta-analysis:**

- Pool effect sizes across categories (random-effects meta-analysis)
- Test for heterogeneity (I² statistic)

### Phase 4: Statistical Control for Function

#### 4.1 ANCOVA

```
property ~ uTP_status + COG_category + uTP_status:COG_category
```

- Main effect of uTP_status: Does property differ by uTP status?
- Effect after controlling for COG: Does effect persist?
- Interaction: Does the uTP effect vary by category?

#### 4.2 Variance Partitioning

```
R²_total = R²_function + R²_uTP + R²_shared + R²_residual
```

Questions answered:

- How much variance is explained by function alone?
- How much additional variance is explained by uTP status?

#### 4.3 Propensity Score Matching

Match uTP proteins to controls by:

- COG category (exact match)
- Length (within 20%)

Then compare properties on matched pairs.

### Phase 5: Interpretation

**Scenario 1: Effect persists (Hypothesis B - Selection)**

- Effect sizes remain large (d > 0.5) within categories
- ANCOVA shows significant uTP effect after controlling for function
- Interpretation: uTP proteins are selected for import-compatible properties

**Scenario 2: Effect disappears (Hypothesis A - Confounding)**

- Effect sizes near zero within categories
- Function explains most of the variance
- Interpretation: Properties are a byproduct of functional enrichment

**Scenario 3: Partial explanation (Mixed)**

- Some reduction in effect size but still significant
- Both function and uTP status contribute
- Interpretation: Partial confounding + some selection

## Statistical Rigor Checklist

- [ ] Pre-registered analysis plan (this document)
- [ ] Single annotation system (COG) for primary analysis
- [ ] Multiple testing correction (FDR for enrichment, Bonferroni for properties)
- [ ] Effect sizes reported (Cohen's d, odds ratios)
- [ ] Power analysis for minimum sample size per category (n≥10)
- [ ] Sensitivity analysis with GO terms as alternative
- [ ] Bootstrap confidence intervals for meta-analysis

## Expected Outputs

### Data Files

- `mature_domains_annotated.csv` - Functional annotations for uTP proteins
- `controls_annotated.csv` - Functional annotations for controls
- `functional_enrichment.csv` - Enrichment test results
- `within_category_comparisons.csv` - Property comparisons within each category
- `ancova_results.csv` - ANCOVA output
- `variance_partitioning.csv` - Variance explained by each factor

### Figures

- `functional_enrichment.svg` - COG category enrichment
- `within_category_effects.svg` - Forest plot of effect sizes by category
- `property_by_function.svg` - Property distributions stratified by function and uTP status
- `variance_partitioning.svg` - Pie chart or bar of variance explained

### Summary Statistics

- Effect of uTP on each property (overall and within-category)
- Proportion of effect explained by functional enrichment
- Conclusion regarding Hypothesis A vs B

## Scripts

| Script                           | Description                                 |
| -------------------------------- | ------------------------------------------- |
| `02_prepare_for_annotation.py`   | Prepare FASTA files for eggNOG-mapper       |
| `03_parse_annotations.py`        | Parse eggNOG results, merge with properties |
| `04_functional_enrichment.py`    | Test for COG category enrichment            |
| `05_within_category_analysis.py` | Compare properties within categories        |
| `06_ancova_analysis.py`          | ANCOVA and variance partitioning            |
| `07_generate_figures.py`         | Create publication-quality figures          |

## Timeline

1. **Day 1**: Prepare data, submit to eggNOG-mapper
2. **Day 2**: Parse annotations, run enrichment analysis
3. **Day 3**: Within-category analysis, ANCOVA
4. **Day 4**: Generate figures, write summary

## Decision Points

1. If annotation coverage is low (<50%), consider:

   - Using GO terms instead of COG
   - Broader functional categories
   - Acknowledging limitation

2. If few categories have n≥10 in both groups:

   - Use n≥5 threshold with caution
   - Focus on largest categories
   - Report power limitations

3. If results are inconclusive:
   - Report both possibilities
   - Suggest follow-up experiments (e.g., experimental import assays)
