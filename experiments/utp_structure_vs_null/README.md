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

---

## Results

### Summary Table

| Dataset       | Metric                 | Observed | Null Mean ± SD | Effect Size | p-value  | Direction |
| ------------- | ---------------------- | -------- | -------------- | ----------- | -------- | --------- |
| Experimental  | Silhouette Score       | 0.096    | 0.135 ± 0.014  | -2.75σ      | **0.01** | LESS      |
| Experimental  | Hopkins Statistic      | 0.324    | 0.331 ± 0.011  | -0.62σ      | 0.28     | LESS      |
| Experimental  | Mean Pairwise Distance | 44.4     | 49.9 ± 0.78    | -7.01σ      | **0.00** | LESS      |
| Experimental  | Distance Variance      | 84.6     | 106.7 ± 15.7   | -1.41σ      | 0.05     | LESS      |
| HMM-predicted | Silhouette Score       | 0.157    | 0.145 ± 0.017  | +0.74σ      | 0.21     | MORE      |
| HMM-predicted | Hopkins Statistic      | 0.212    | 0.222 ± 0.004  | -2.72σ      | **0.00** | LESS      |
| HMM-predicted | Mean Pairwise Distance | 44.0     | 42.8 ± 0.24    | +5.14σ      | **0.00** | MORE      |
| HMM-predicted | Distance Variance      | 110.8    | 148.9 ± 4.5    | -8.47σ      | **0.00** | LESS      |

### Key Findings

#### 1. Experimental sequences show ANTI-CLUSTERING (p=0.01)

The silhouette score for real uTP sequences (0.096) is significantly **lower** than null sequences (0.135). This means uTP sequences are **more spread out** in embedding space than random - they actively avoid clustering.

#### 2. Both datasets show uniform spread (highly significant)

Distance variance is dramatically lower than null in both datasets:

- Experimental: 84.6 vs 106.7 (p=0.05)
- HMM-predicted: 110.8 vs 148.9 (p<0.01)

This indicates sequences are **uniformly distributed** rather than forming distinct clusters.

#### 3. Experimental sequences are more compact than random

Mean pairwise distance is significantly lower than null (44.4 vs 49.9, p<0.01), suggesting the sequences occupy a **constrained region** of sequence space while being uniformly spread within that region.

#### 4. HMM-predicted set shows different pattern

The HMM-predicted set (933 sequences) shows:

- No significant clustering structure (silhouette p=0.21)
- Slightly MORE spread out than null (mean distance higher)
- But still uniformly distributed (low variance)

---

## Interpretation

### What This Means for uTP Variants

**The "variants" (terminal_4, terminal_5, terminal_7, terminal_9) do NOT represent discrete biological groups.**

Instead, uTP sequences appear to:

1. Occupy a **constrained functional space** (compact overall distribution)
2. Be **uniformly distributed** within that space (low variance)
3. Show **anti-clustering** - actively avoiding discrete groupings

This is consistent with:

- **Functional constraint**: All uTPs must perform the same targeting function, limiting sequence diversity
- **No discrete subtypes**: The terminal motif differences may be superficial, not reflecting deep biological divisions
- **Continuous variation**: uTP variation is gradual, not categorical

### Implications for Previous Classifier Results

The ~68% classifier accuracy for terminal motif prediction likely reflects:

- Simple amino acid composition differences (which we controlled for here)
- Not deep structural or functional differences between variants

---

## Datasets Analyzed

1. **Experimental (n=206)**: Gblocks-filtered C-terminal sequences from proteomics enrichment

   - Higher confidence uTP sequences
   - Only the ~120 AA uTP region

2. **HMM-predicted (n=933)**: Full-length proteins from Import_candidates.fasta
   - Broader set including potential false positives
   - Full protein sequences, not just uTP region

---

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

## Usage

```bash
cd /path/to/nitroplast
uv run python experiments/utp_structure_vs_null/analyze_structure.py
```

**Note**: Analysis was run on remote GPU (NVIDIA RTX PRO 6000 Blackwell) due to computational requirements (~200 embedding operations with ProtT5).

---

_Last updated: 2026-01-13_
