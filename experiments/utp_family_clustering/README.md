# uTP Family Clustering Analysis

**Testing whether uTP proteins are more related to each other than expected by chance.**

## Hypothesis

We test two competing models for uTP evolution:

| Model                        | Prediction                                      | Mechanism                                                                           |
| ---------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------- |
| **Model A (Founder Effect)** | uTP proteins cluster into few gene families     | uTP originated in a few ancestral proteins that expanded through gene duplication   |
| **Model B (Selection)**      | uTP proteins are scattered across many families | uTP was independently acquired by diverse proteins based on functional requirements |

## Method

1. **Cluster ALL B. bigelowii proteins** by mature domain similarity
   - For uTP proteins: remove C-terminal uTP (~120 aa) to get mature domain
   - For non-uTP proteins: use full sequence
2. **Define gene families** using k-mer Jaccard distance with hierarchical clustering
   - Distance threshold ~0.7 corresponds roughly to ~40% sequence identity
3. **Calculate clustering metrics**:

   - Fraction of uTP proteins sharing a family with another uTP protein
   - Number of distinct families containing uTP proteins
   - Effective number of families (Simpson diversity)
   - Maximum uTP proteins in any single family

4. **Permutation test** (n=10,000):
   - Shuffle uTP labels among all proteins
   - Calculate null distribution for each metric
   - Compare observed to expected under random model

## Key Question

> If uTP proteins are randomly distributed across the proteome, what fraction would share gene families by chance?
> Is the observed sharing significantly different from this expectation?

## Usage

```bash
# Step 1: Prepare data (extract B. bigelowii proteins, identify uTP, get mature domains)
uv run python experiments/utp_family_clustering/01_prepare_data.py

# Step 2: Cluster proteins into families
uv run python experiments/utp_family_clustering/02_cluster_families.py

# Step 3: Run statistical test
uv run python experiments/utp_family_clustering/03_statistical_test.py
```

## Expected Output

```
output/
├── bb_proteins.fasta           # All B. bigelowii proteins
├── utp_proteins.fasta          # uTP proteins (full sequences)
├── mature_domains.fasta        # All mature domains for clustering
├── utp_mature_domains.fasta    # uTP mature domains only
├── protein_metadata.csv        # Metadata for all proteins
├── family_assignments.csv      # Family membership for each protein
├── family_statistics.csv       # Stats at different clustering thresholds
├── distance_matrix.npz         # Cached pairwise distances
├── permutation_results.csv     # Statistical test results
├── permutation_test.png/svg    # Visualization of null distributions
└── statistical_summary.txt     # Human-readable interpretation
```

## Results

### Key Finding: uTP proteins are significantly CLUSTERED

| Metric                       | Observed  | Expected (random) | p-value  |
| ---------------------------- | --------- | ----------------- | -------- |
| Fraction sharing family      | **25.4%** | 1.9%              | < 0.0001 |
| Number of families with uTP  | **624**   | 716               | < 0.0001 |
| Effective number of families | **548.5** | 708.2             | < 0.0001 |
| Max uTP per family           | 4         | 2.4               | 0.052    |

### Distribution of uTP proteins across families

| uTP proteins per family | Number of families |
| ----------------------- | ------------------ |
| 1                       | 539                |
| 2                       | 73                 |
| 3                       | 10                 |
| 4                       | 2                  |

- **184 uTP proteins** (25.4%) share a gene family with at least one other uTP protein
- **85 families** contain 2+ uTP proteins
- Largest families have 4 uTP proteins each

## Caveats

1. **Subsampling**: Analysis used 6,000 proteins (all 723 uTP + 5,277 sampled non-uTP)
2. **K-mer distance**: Jaccard distance on 3-mers is an approximation of sequence identity
3. **Threshold sensitivity**: Results are robust across thresholds 0.6-0.85

---

_Last updated: 2026-01-13_
