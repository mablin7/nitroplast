#!/usr/bin/env python3
"""
Test Interpretation A: Gene Family Structure

Hypothesis: Proteins acquired uTP through gene duplication. Same-variant proteins
are more likely to be paralogs (share sequence similarity in mature domain).

Tests:
1. Within-group vs between-group sequence similarity of mature domains
2. Clustering of mature domains - do clusters correlate with variant groups?
3. Permutation test: is within-group similarity higher than expected by chance?
"""

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
VARIANT_ASSIGNMENTS = Path(__file__).parent / "output/variant_assignments.csv"
IMPORT_FASTA = PROJECT_ROOT / "data/Import_candidates.fasta"
UTP_METADATA = (
    PROJECT_ROOT / "experiments/utp_sequence_clustering/output/utp_metadata.csv"
)
EMBEDDINGS_FILE = (
    PROJECT_ROOT / "experiments/utp_variant_classifier/output/embeddings.h5"
)
OUTPUT_DIR = Path(__file__).parent / "output"


def load_sequences(fasta_path: Path) -> dict:
    """Load sequences from FASTA file."""
    sequences = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences


def load_embeddings(embedding_file: Path, sequence_ids: list[str]) -> tuple:
    """Load ProtT5 embeddings for specified sequences."""
    if not embedding_file.exists():
        return None, []

    embeddings = []
    found_ids = []

    with h5py.File(embedding_file, "r") as f:
        for seq_id in sequence_ids:
            if seq_id in f:
                embeddings.append(f[seq_id][:])
                found_ids.append(seq_id)

    if not embeddings:
        return None, []

    return np.array(embeddings), found_ids


def compute_kmer_similarity(seq1: str, seq2: str, k: int = 4) -> float:
    """Compute k-mer Jaccard similarity between two sequences."""

    def get_kmers(seq):
        return set(seq[i : i + k] for i in range(len(seq) - k + 1))

    kmers1 = get_kmers(seq1.upper())
    kmers2 = get_kmers(seq2.upper())

    if not kmers1 or not kmers2:
        return 0.0

    intersection = len(kmers1 & kmers2)
    union = len(kmers1 | kmers2)

    return intersection / union if union > 0 else 0.0


def compute_pairwise_similarities(
    sequences: dict, sample_size: int = 500
) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise k-mer similarities (sampling if needed)."""

    names = list(sequences.keys())
    if len(names) > sample_size:
        np.random.seed(42)
        names = list(np.random.choice(names, sample_size, replace=False))

    n = len(names)
    similarity = np.zeros((n, n))

    print(f"  Computing {n * (n - 1) // 2} pairwise similarities...")
    for i in range(n):
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{n}")
        for j in range(i + 1, n):
            sim = compute_kmer_similarity(sequences[names[i]], sequences[names[j]])
            similarity[i, j] = similarity[j, i] = sim
        similarity[i, i] = 1.0

    return similarity, names


def test_within_vs_between_similarity(
    similarity: np.ndarray, names: list[str], assignments: dict[str, str]
) -> dict:
    """
    Test 1: Compare within-group vs between-group similarity.

    If Interpretation A is correct: within-group similarity >> between-group
    """

    within_sims = []
    between_sims = []

    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            sim = similarity[i, j]
            group_i = assignments.get(names[i])
            group_j = assignments.get(names[j])

            if group_i is None or group_j is None:
                continue

            if group_i == group_j:
                within_sims.append(sim)
            else:
                between_sims.append(sim)

    # Statistics
    within_mean = np.mean(within_sims)
    between_mean = np.mean(between_sims)

    # Mann-Whitney U test (non-parametric)
    statistic, pvalue = stats.mannwhitneyu(
        within_sims, between_sims, alternative="greater"
    )

    # Effect size (rank-biserial correlation)
    n1, n2 = len(within_sims), len(between_sims)
    effect_size = 1 - (2 * statistic) / (n1 * n2)

    return {
        "within_mean": within_mean,
        "between_mean": between_mean,
        "within_median": np.median(within_sims),
        "between_median": np.median(between_sims),
        "within_n": len(within_sims),
        "between_n": len(between_sims),
        "mann_whitney_statistic": statistic,
        "pvalue": pvalue,
        "effect_size": effect_size,
        "within_sims": within_sims,
        "between_sims": between_sims,
    }


def test_clustering_alignment(
    similarity: np.ndarray, names: list[str], assignments: dict[str, str]
) -> dict:
    """
    Test 2: Do mature domain clusters align with variant groups?

    If Interpretation A is correct: high ARI between sequence clusters and variants
    """

    # Convert similarity to distance
    distance = 1 - similarity

    # Hierarchical clustering on mature domains
    condensed = squareform(distance)
    Z = linkage(condensed, method="average")

    # Try different numbers of clusters
    results = []
    variant_labels = [assignments.get(n, "unknown") for n in names]
    variant_to_int = {v: i for i, v in enumerate(set(variant_labels))}
    variant_ints = [variant_to_int[v] for v in variant_labels]

    for n_clusters in [4, 5, 6, 7, 8]:
        cluster_labels = fcluster(Z, n_clusters, criterion="maxclust")
        ari = adjusted_rand_score(variant_ints, cluster_labels)
        results.append({"n_clusters": n_clusters, "ari": ari})

    return {"cluster_ari_results": results, "linkage": Z}


def permutation_test(
    similarity: np.ndarray,
    names: list[str],
    assignments: dict[str, str],
    n_permutations: int = 1000,
) -> dict:
    """
    Test 3: Permutation test for within-group similarity.

    Null hypothesis: variant group labels are independent of sequence similarity.
    """

    # Get group labels
    labels = [assignments.get(n) for n in names]
    valid_mask = [l is not None for l in labels]
    valid_labels = [l for l in labels if l is not None]

    # Observed within-group similarity
    def compute_within_similarity(labels):
        within_sims = []
        n = len(names)
        for i in range(n):
            for j in range(i + 1, n):
                if not valid_mask[i] or not valid_mask[j]:
                    continue
                if labels[i] == labels[j]:
                    within_sims.append(similarity[i, j])
        return np.mean(within_sims) if within_sims else 0

    observed = compute_within_similarity(valid_labels)

    # Permutation distribution
    permuted_means = []
    rng = np.random.default_rng(42)

    for _ in range(n_permutations):
        shuffled = rng.permutation(valid_labels).tolist()
        # Put back in full list
        full_shuffled = []
        shuffle_idx = 0
        for i in range(len(names)):
            if valid_mask[i]:
                full_shuffled.append(shuffled[shuffle_idx])
                shuffle_idx += 1
            else:
                full_shuffled.append(None)
        permuted_means.append(compute_within_similarity(full_shuffled))

    # P-value
    pvalue = np.mean(np.array(permuted_means) >= observed)

    return {
        "observed_within_sim": observed,
        "permuted_mean": np.mean(permuted_means),
        "permuted_std": np.std(permuted_means),
        "pvalue": pvalue,
        "z_score": (observed - np.mean(permuted_means)) / np.std(permuted_means),
        "permuted_distribution": permuted_means,
    }


def visualize_results(
    similarity_results: dict,
    clustering_results: dict,
    permutation_results: dict,
    output_dir: Path,
):
    """Create visualizations of the test results."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Within vs Between similarity distributions
    ax = axes[0, 0]
    ax.hist(
        similarity_results["within_sims"],
        bins=50,
        alpha=0.7,
        label=f"Within-group (mean={similarity_results['within_mean']:.3f})",
        density=True,
    )
    ax.hist(
        similarity_results["between_sims"],
        bins=50,
        alpha=0.7,
        label=f"Between-group (mean={similarity_results['between_mean']:.3f})",
        density=True,
    )
    ax.set_xlabel("K-mer Similarity")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Within vs Between Group Similarity\n"
        f"Mann-Whitney p={similarity_results['pvalue']:.2e}, "
        f"effect size={similarity_results['effect_size']:.3f}"
    )
    ax.legend()

    # Plot 2: Clustering ARI by number of clusters
    ax = axes[0, 1]
    ari_results = clustering_results["cluster_ari_results"]
    n_clusters = [r["n_clusters"] for r in ari_results]
    aris = [r["ari"] for r in ari_results]
    ax.bar(n_clusters, aris, color="steelblue")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_title("Sequence Cluster vs Variant Group Alignment")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    for i, (nc, ari) in enumerate(zip(n_clusters, aris)):
        ax.text(nc, ari + 0.01, f"{ari:.3f}", ha="center", fontsize=9)

    # Plot 3: Permutation test
    ax = axes[1, 0]
    ax.hist(
        permutation_results["permuted_distribution"],
        bins=50,
        alpha=0.7,
        label="Permuted",
        density=True,
    )
    ax.axvline(
        permutation_results["observed_within_sim"],
        color="red",
        linewidth=2,
        label=f"Observed ({permutation_results['observed_within_sim']:.4f})",
    )
    ax.set_xlabel("Mean Within-Group Similarity")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Permutation Test\n"
        f"p={permutation_results['pvalue']:.4f}, "
        f"z={permutation_results['z_score']:.2f}"
    )
    ax.legend()

    # Plot 4: Summary
    ax = axes[1, 1]
    ax.axis("off")

    summary_text = f"""
INTERPRETATION A: Gene Family Structure Test Results
=====================================================

Test 1: Within vs Between Group Similarity
  Within-group mean:  {similarity_results['within_mean']:.4f}
  Between-group mean: {similarity_results['between_mean']:.4f}
  Difference:         {similarity_results['within_mean'] - similarity_results['between_mean']:.4f}
  Mann-Whitney p:     {similarity_results['pvalue']:.2e}
  Effect size:        {similarity_results['effect_size']:.3f}

Test 2: Cluster-Variant Alignment
  Best ARI:           {max(aris):.3f} (at k={n_clusters[aris.index(max(aris))]})
  Interpretation:     {"Strong" if max(aris) > 0.3 else "Moderate" if max(aris) > 0.1 else "Weak"} alignment

Test 3: Permutation Test
  Observed:           {permutation_results['observed_within_sim']:.4f}
  Expected (null):    {permutation_results['permuted_mean']:.4f}
  Z-score:            {permutation_results['z_score']:.2f}
  P-value:            {permutation_results['pvalue']:.4f}

CONCLUSION:
{"SUPPORTS Interpretation A" if permutation_results['pvalue'] < 0.05 and similarity_results['effect_size'] > 0.1 else "DOES NOT SUPPORT Interpretation A"}
Same-variant proteins {"ARE" if permutation_results['pvalue'] < 0.05 else "are NOT"} more similar than expected.
"""
    ax.text(
        0.1,
        0.9,
        summary_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "paralog_test_results.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Test Interpretation A: Gene Family Structure")
    print("=" * 70)

    # Load variant assignments
    print("\n[1/5] Loading data...")
    if not VARIANT_ASSIGNMENTS.exists():
        print(f"  ERROR: Run 01_define_groups.py first")
        return

    assignments_df = pd.read_csv(VARIANT_ASSIGNMENTS)
    assignments = dict(
        zip(assignments_df["sequence_id"], assignments_df["variant_group"])
    )
    print(f"  Loaded {len(assignments)} variant assignments")

    # Load sequences
    sequences = load_sequences(IMPORT_FASTA)
    # Filter to assigned sequences
    sequences = {k: v for k, v in sequences.items() if k in assignments}
    print(f"  Loaded {len(sequences)} sequences with assignments")

    # Load uTP metadata to get mature domain boundaries
    # For now, we'll use full sequences minus estimated uTP region
    # A more precise approach would use HMM boundaries
    print("\n  Note: Using full sequences for similarity (mature + uTP)")
    print("        For cleaner test, should use mature domain only")

    # Compute pairwise similarities
    print("\n[2/5] Computing pairwise similarities...")
    similarity, names = compute_pairwise_similarities(sequences, sample_size=600)

    # Test 1: Within vs Between similarity
    print("\n[3/5] Test 1: Within vs Between group similarity...")
    sim_results = test_within_vs_between_similarity(similarity, names, assignments)
    print(f"  Within-group mean:  {sim_results['within_mean']:.4f}")
    print(f"  Between-group mean: {sim_results['between_mean']:.4f}")
    print(f"  Mann-Whitney p:     {sim_results['pvalue']:.2e}")
    print(f"  Effect size:        {sim_results['effect_size']:.3f}")

    # Test 2: Clustering alignment
    print("\n[4/5] Test 2: Cluster-Variant alignment...")
    cluster_results = test_clustering_alignment(similarity, names, assignments)
    best_ari = max(r["ari"] for r in cluster_results["cluster_ari_results"])
    print(f"  Best ARI: {best_ari:.3f}")

    # Test 3: Permutation test
    print("\n[5/5] Test 3: Permutation test...")
    perm_results = permutation_test(similarity, names, assignments, n_permutations=1000)
    print(f"  Observed within-sim: {perm_results['observed_within_sim']:.4f}")
    print(f"  Expected (null):     {perm_results['permuted_mean']:.4f}")
    print(f"  Z-score:             {perm_results['z_score']:.2f}")
    print(f"  P-value:             {perm_results['pvalue']:.4f}")

    # Visualize
    visualize_results(sim_results, cluster_results, perm_results, OUTPUT_DIR)
    print(f"\n  Saved visualization to {OUTPUT_DIR / 'paralog_test_results.png'}")

    # Save detailed results
    results_summary = {
        "test": "Interpretation A - Gene Family Structure",
        "within_mean_similarity": sim_results["within_mean"],
        "between_mean_similarity": sim_results["between_mean"],
        "mann_whitney_pvalue": sim_results["pvalue"],
        "effect_size": sim_results["effect_size"],
        "best_cluster_ari": best_ari,
        "permutation_pvalue": perm_results["pvalue"],
        "permutation_zscore": perm_results["z_score"],
    }
    pd.DataFrame([results_summary]).to_csv(
        OUTPUT_DIR / "paralog_test_summary.csv", index=False
    )

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    supports_A = perm_results["pvalue"] < 0.05 and sim_results["effect_size"] > 0.05

    if supports_A:
        print("\n  ✓ SUPPORTS Interpretation A (Gene Family Structure)")
        print("    Same-variant proteins are more similar than expected by chance.")
        print(f"    This could indicate shared evolutionary origin (paralogy).")
    else:
        print("\n  ✗ DOES NOT SUPPORT Interpretation A")
        print("    Variant groups do not cluster by sequence similarity.")
        print(
            "    Variants may be determined by other factors (functional constraint)."
        )

    print(f"\n📁 Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
