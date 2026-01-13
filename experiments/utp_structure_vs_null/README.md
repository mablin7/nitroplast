# uTP Structure vs Null Analysis

**Testing whether uTP sequences have more structure than expected by chance.**

## Motivation

Previous experiments tried to find discrete uTP variants/clusters. But the fundamental question is: **Is there ANY structure in uTP sequences beyond what random sequences with the same amino acid composition would show?**

This experiment answers that question using a rigorous null model comparison.

## Approach

### Null Model

For each real uTP sequence, we generate null sequences by:

1. **Shuffling** each sequence individually (preserves exact AA composition per sequence)
2. Computing ProtT5 embeddings for both real and null sequences
3. Comparing clustering metrics between real and null distributions

### Metrics Computed

| Metric                     | What it measures                                   | Interpretation                    |
| -------------------------- | -------------------------------------------------- | --------------------------------- |
| **Silhouette Score**       | How well-separated clusters are                    | Higher = more distinct groupings  |
| **Hopkins Statistic**      | Clustering tendency (0.5=uniform, >0.75=clustered) | Higher = more clustering tendency |
| **Mean Pairwise Distance** | Average distance between sequences                 | Higher = more spread out          |
| **Distance Variance**      | Uniformity of spread                               | Higher = less uniform             |

### Statistical Test

For each metric, we perform a permutation test:

- Generate 100 null sequence sets
- Compute metric for each null set
- Compare real metric to null distribution
- Report p-value and effect size (in standard deviations)

## Interpretation Guide

### Silhouette Score Results

| Result                 | Interpretation                                                        |
| ---------------------- | --------------------------------------------------------------------- |
| Real > Null (p < 0.05) | uTP has **detectable structure** - sequences cluster more than random |
| Real < Null (p < 0.05) | uTP is **anti-clustered** - sequences are actively spread out         |
| Real ≈ Null (p > 0.05) | **No detectable structure** beyond amino acid composition             |

### Hopkins Statistic Results

| Result                 | Interpretation                                |
| ---------------------- | --------------------------------------------- |
| Real > Null (p < 0.05) | uTP has **clustering tendency**               |
| Real < Null (p < 0.05) | uTP is **regularly spaced** (anti-clustered)  |
| Real ≈ Null (p > 0.05) | **No clustering tendency** beyond composition |

## Datasets Analyzed

1. **Experimental (n=206)**: Gblocks-filtered sequences from proteomics enrichment

   - Higher confidence uTP sequences
   - Smaller sample size

2. **HMM-predicted (n=933)**: Full Import_candidates.fasta
   - Broader set including potential false positives
   - Larger sample size for statistical power

## Key Questions Answered

1. **Is there signal at all?** If real uTP shows significantly different structure than null, there IS biological signal to investigate.

2. **Are discrete variants real?** If silhouette is significantly higher than null, the variant groupings may reflect real biology.

3. **Is the signal in experimental vs HMM sets different?** Comparing the two datasets reveals whether HMM predictions add noise or capture the same signal.

## Usage

```bash
cd /path/to/nitroplast
uv run python experiments/utp_structure_vs_null/analyze_structure.py
```

**Note**: This analysis requires ProtT5 embeddings and takes ~30-60 minutes depending on hardware.

## Output Files

```
output/
├── embeddings/
│   ├── experimental_embeddings.h5    # Cached ProtT5 embeddings
│   └── hmm_predicted_embeddings.h5
├── results/
│   └── structure_analysis_summary.csv  # All metrics and p-values
└── figures/
    ├── experimental_silhouette_score.svg
    ├── experimental_hopkins_statistic.svg
    ├── hmm_predicted_silhouette_score.svg
    ├── hmm_predicted_hopkins_statistic.svg
    └── comparison_summary.svg
```

## Expected Outcomes

### If Real > Null (structure detected)

- Supports the hypothesis that uTP variants encode meaningful information
- Justifies further investigation of what distinguishes variants
- Suggests the classifier results (68% accuracy) reflect real signal

### If Real ≈ Null (no structure)

- Suggests uTP variants may be artifacts of motif detection
- The 68% classifier accuracy may be driven by simple AA composition, not sequence patterns
- Would need to reconsider the biological interpretation

### If Real < Null (anti-clustered)

- Suggests uTP sequences are actively diversified
- Could indicate selection for sequence diversity within the uTP system
- Would be an interesting finding in itself

---

_Last updated: 2026-01-13_
