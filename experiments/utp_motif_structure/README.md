# uTP Motif Structure Analysis

## Motivation

Previous sequence clustering of full uTP regions (58-1091 aa) failed to find discrete clusters (best silhouette: 0.08). This experiment tests whether structure emerges using motif-informed approaches:

1. **Motif-based feature matrix**: Cluster on structured motif features (presence, score, position)
2. **Motif region clustering**: Cluster only the conserved motif-containing region
3. **Sequence similarity network**: Build and analyze a similarity network for community structure

## Data Sources

- MAST results: `../utp_motif_coverage/output/mast_results/mast.xml`
- uTP sequences: `../utp_sequence_clustering/output/utp_sequences.fasta`
- Full proteins: `../../data/Import_candidates.fasta`

## Results Summary

| Approach                 | Best Method    | Silhouette | Notes               |
| ------------------------ | -------------- | ---------- | ------------------- |
| Full sequence (baseline) | KMeans k=4     | 0.083      | Previous experiment |
| **Motif features**       | DBSCAN eps=1.5 | **0.545**  | +0.46 improvement   |
| **Motif region (k-mer)** | Hier k=4       | **0.436**  | +0.35 improvement   |
| Motif region (embedding) | KMeans k=4     | 0.096      | ~same as baseline   |
| Network communities      | Louvain        | N/A        | Sparse network      |

**Key finding**: Motif-based features produce substantially better clustering than raw sequence data.

---

## Approach 1: Motif-Based Feature Matrix

### Method

For each of 745 sequences with MAST hits, constructed a 37-dimensional feature vector:

| Feature Type    | Count | Description                              |
| --------------- | ----- | ---------------------------------------- |
| Motif presence  | 10    | Binary: has_motif_1 through has_motif_10 |
| Motif score     | 10    | -log10(p-value) for best hit per motif   |
| Motif position  | 10    | Normalized position (pos/length)         |
| Global features | 7     | n_motifs, span, terminal_4/5/7/9/other   |

### Results

| Method           | Silhouette | n_clusters | Cluster Sizes          |
| ---------------- | ---------- | ---------- | ---------------------- |
| DBSCAN eps=1.5   | 0.545      | 19         | 5-317 (main: 317, 221) |
| DBSCAN eps=2.0   | 0.533      | 22         | 5-223                  |
| DBSCAN eps=1.0   | 0.394      | 10         | 5-490                  |
| KMeans k=7       | 0.302      | 7          | 21-330                 |
| KMeans k=6       | 0.280      | 6          | 21-361                 |
| Hierarchical k=5 | 0.258      | 5          | 21-440                 |

### Cluster Characterization (Hierarchical k=6)

| Cluster | Size | Avg Motifs | Key Features                            |
| ------- | ---- | ---------- | --------------------------------------- |
| 0       | 352  | 5.5        | High motif_1, motif_2, motif_4, motif_6 |
| 1       | 142  | 3.5        | Lower motif_1/2, moderate motif_4/6     |
| 2       | 88   | 6.4        | Very high motif_7 (100%)                |
| 3       | 21   | 6.2        | High motif_8 (100%)                     |
| 4       | 70   | 5.4        | Very high motif_9 (100%)                |
| 5       | 72   | 5.1        | High motif_3 (100%)                     |

The clusters appear to segregate by **which terminal/distinctive motifs are present**, consistent with the motif variant hypothesis.

---

## Approach 2: Motif Region Only Clustering

### Method

Compared three region extraction strategies:

- **first_to_end**: From first motif hit to C-terminus
- **first_to_last**: From first to last motif (+30aa buffer)
- **fixed_window**: Fixed 150aa from first motif

Clustering performed on k-mer (k=4) frequency vectors.

### Results

| Region Mode       | Length Range       | Best Method  | Silhouette |
| ----------------- | ------------------ | ------------ | ---------- |
| first_to_end      | 23-1801 (med: 219) | Hier k=4     | 0.368      |
| **first_to_last** | 23-1801 (med: 131) | **Hier k=4** | **0.436**  |
| fixed_window      | 23-150 (med: 150)  | KMeans k=4   | 0.029      |

The **first_to_last** mode (containing only the motif region) performs best with k-mer features.

### Embedding vs K-mer Features

| Feature Type     | Best Silhouette | Notes                           |
| ---------------- | --------------- | ------------------------------- |
| K-mer (k=4)      | 0.436           | Better for short motif regions  |
| ProtT5 embedding | 0.096           | Not effective for short regions |

K-mer features outperform deep learning embeddings for this short, conserved region.

---

## Approach 3: Sequence Similarity Network

### Method

Built k-mer Jaccard similarity network on 691 uTP sequences. Analyzed network structure across similarity thresholds.

### Similarity Distribution

| Statistic         | Value |
| ----------------- | ----- |
| Min similarity    | 0.000 |
| Max similarity    | 1.000 |
| Mean similarity   | 0.074 |
| Median similarity | 0.072 |

Most sequence pairs have <10% k-mer overlap, indicating high sequence diversity.

### Threshold Analysis

| Threshold | Edges  | Components | Communities | Purity |
| --------- | ------ | ---------- | ----------- | ------ |
| 0.10      | 32,930 | 26         | 44          | 0.69   |
| 0.15      | 1,233  | 378        | 388         | 0.87   |
| 0.20      | 282    | 545        | 545         | 0.95   |
| 0.25      | 160    | 561        | 562         | 0.97   |
| 0.30      | 136    | 577        | 577         | 0.98   |

At low thresholds (0.10), the network forms 44 communities with 69% terminal motif purity. As threshold increases, the network fragments into many small components (mostly singletons).

### Interpretation

- The network is **sparse** - most uTP pairs are not highly similar
- This confirms **continuous variation** rather than discrete clusters
- The 69% purity at threshold 0.10 suggests some structure exists but is not sharp

---

## Output Files

| File                           | Description                     |
| ------------------------------ | ------------------------------- |
| `motif_features.csv`           | 745 x 37 feature matrix         |
| `motif_feature_clustering.png` | PCA/t-SNE visualization         |
| `motif_feature_heatmap.png`    | Feature values by cluster       |
| `cluster_composition.csv`      | Detailed cluster statistics     |
| `region_mode_comparison.csv`   | Comparison of region extraction |
| `motif_region_umap.png`        | UMAP of motif regions           |
| `similarity_network.png`       | Network visualization           |
| `threshold_sweep.csv/png`      | Network threshold analysis      |
