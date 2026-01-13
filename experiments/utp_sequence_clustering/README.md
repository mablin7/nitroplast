# uTP Sequence Clustering Analysis

**Testing whether uTP sequences form natural discrete clusters.**

## Background

Previous motif-based classification of uTP variants showed inconsistent patterns:

- MEME analysis (on 206 proteins) vs MAST analysis (on 933) gave different distributions
- Terminal motif patterns varied significantly between datasets
- 20% of HMM-hit proteins had no detectable motifs

This experiment applies multiple established clustering methods directly to uTP sequences to test whether they form discrete natural groups.

## Methods

### Sequence Extraction

- Extract uTP regions using HMM hit positions (C-terminal from hit start)
- 691 sequences extracted with length 58-1091 aa (median 206)

### Clustering Methods Applied

| Method       | Type            | Parameters                         |
| ------------ | --------------- | ---------------------------------- |
| Hierarchical | Distance-based  | k=4,5,6,7; k-mer Jaccard distance  |
| Spectral     | Graph-based     | k=4,5,6,7; similarity matrix       |
| KMeans       | Embedding-based | k=4,5,6,7; ProtT5 embeddings       |
| DBSCAN       | Density-based   | eps=0.5,1.0,2.0; ProtT5 embeddings |

## Results

### Clustering Quality (Silhouette Scores)

| Method          | Best k | Silhouette | Interpretation                          |
| --------------- | ------ | ---------- | --------------------------------------- |
| DBSCAN_eps0.5   | 2      | 0.324      | Best score but trivial (1 main + noise) |
| KMeans_k4       | 4      | 0.083      | Low - weak cluster structure            |
| Hierarchical_k4 | 4      | 0.032      | Very low - no clear clusters            |
| Spectral_k7     | 7      | 0.011      | Very low - no clear clusters            |

**Silhouette interpretation**: Values near 0 indicate overlapping clusters; values > 0.5 indicate strong cluster structure.

### Cluster Size Distributions

**Hierarchical_k5**: 684 + 3 + 2 + 1 + 1 (one dominant cluster)
**Spectral_k5**: 351 + 262 + 68 + 5 + 5 (moderately balanced)
**KMeans_k5**: 208 + 184 + 146 + 117 + 36 (balanced)
**DBSCAN_eps0.5**: 640 + 18 + 33 noise (single main cluster)

### Cross-Method Agreement (Adjusted Rand Index)

| Comparison                   | ARI       | Interpretation        |
| ---------------------------- | --------- | --------------------- |
| Hierarchical methods         | 0.66-0.93 | Agree with each other |
| Spectral methods             | 0.54-0.97 | Agree with each other |
| KMeans methods               | 0.36-0.76 | Agree with each other |
| **Hierarchical vs KMeans**   | **~0.00** | **No agreement**      |
| **Spectral vs KMeans**       | **~0.04** | **No agreement**      |
| **Hierarchical vs Spectral** | **~0.00** | **No agreement**      |

**Critical finding**: Different clustering approaches produce completely different groupings.

## Key Findings

### 1. No Natural Discrete Clusters

The uTP sequences do not form discrete, well-separated clusters:

- Low silhouette scores across all methods (0.01-0.08)
- Different methods produce completely different clusterings (ARI ≈ 0)
- Hierarchical clustering produces one massive cluster + tiny outliers
- DBSCAN assigns 93% of sequences to a single cluster

### 2. Continuous Sequence Space

The UMAP visualization shows sequences form a continuous distribution:

- No clear gaps or separations between groups
- KMeans "clusters" are arbitrary divisions of continuous space
- Suggests uTPs may represent a **spectrum** rather than discrete types

### 3. Implications for uTP Classification

- **Motif-based variants may be artificial** - the underlying sequences don't show discrete groupings
- **Terminal motif differences may be superficial** - not reflecting deep sequence divergence
- **uTPs may evolve along a continuum** - with gradual variation rather than distinct types

## Generated Outputs

```
output/
├── utp_sequences.fasta        # Extracted uTP sequences (691)
├── utp_metadata.csv           # Extraction metadata
├── utp_embeddings.h5          # ProtT5 embeddings
├── cluster_assignments.csv    # All cluster assignments
├── clustering_evaluation.csv  # Quality metrics
├── cluster_visualization.png  # UMAP plots
├── ari_heatmap.png           # Cross-method agreement
└── dendrogram.png            # Hierarchical tree
```

## Usage

```bash
# Extract uTP sequences
uv run python experiments/utp_sequence_clustering/01_extract_utp_sequences.py

# Run clustering analysis
uv run python experiments/utp_sequence_clustering/02_cluster_sequences.py
```

## Conclusions

**uTP sequences appear to form a continuous distribution rather than discrete clusters.** This finding suggests:

1. The motif-pattern based classification may not reflect true sequence diversity
2. uTP "variants" may be arbitrary divisions of a continuum
3. Alternative classification approaches (e.g., by function, by import timing) may be more meaningful

---

_Last updated: 2026-01-12_
