#!/usr/bin/env python3
"""
02_cluster_families.py - Cluster Proteins into Gene Families

Clusters ALL B. bigelowii mature proteins to define gene families,
then assigns family membership to each protein.

Uses k-mer based similarity with hierarchical clustering to define families
at a ~40% identity threshold (standard for gene family definition).

Usage:
    uv run python experiments/utp_family_clustering/02_cluster_families.py
"""

import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"

# Input files
MATURE_DOMAINS = OUTPUT_DIR / "mature_domains.fasta"
METADATA_FILE = OUTPUT_DIR / "protein_metadata.csv"

# Output files
FAMILY_ASSIGNMENTS = OUTPUT_DIR / "family_assignments.csv"
FAMILY_STATS = OUTPUT_DIR / "family_statistics.csv"
DISTANCE_MATRIX = OUTPUT_DIR / "distance_matrix.npz"

# =============================================================================
# Parameters
# =============================================================================

# K-mer size for similarity calculation
KMER_SIZE = 3

# Distance threshold for family definition
# Note: k-mer Jaccard distance ~0.7 corresponds roughly to ~40% seq identity
FAMILY_DISTANCE_THRESHOLDS = [0.6, 0.7, 0.8, 0.85]

# Subsample size for large proteomes (for speed)
# Keep all uTP proteins, sample from non-uTP to reach this total
MAX_PROTEINS = 6000  # ~18 million pairs, tractable in ~10 min

# =============================================================================
# Functions
# =============================================================================


def load_sequences(fasta_file: Path) -> dict:
    """Load sequences from FASTA file."""
    return {r.id: str(r.seq) for r in SeqIO.parse(fasta_file, "fasta")}


def get_kmers(sequence: str, k: int = 3) -> Counter:
    """Extract k-mer counts from a sequence."""
    return Counter(sequence[i : i + k] for i in range(len(sequence) - k + 1))


def jaccard_distance(kmer1: Counter, kmer2: Counter) -> float:
    """Calculate Jaccard distance between two k-mer profiles."""
    keys = set(kmer1.keys()) | set(kmer2.keys())
    if not keys:
        return 1.0

    intersection = sum(min(kmer1.get(k, 0), kmer2.get(k, 0)) for k in keys)
    union = sum(max(kmer1.get(k, 0), kmer2.get(k, 0)) for k in keys)

    if union == 0:
        return 1.0

    return 1.0 - (intersection / union)


def compute_distance_matrix(sequences: dict, k: int = 3) -> tuple:
    """
    Compute pairwise k-mer Jaccard distance matrix.

    Returns:
        distances: Condensed distance matrix (for linkage)
        names: List of sequence names in order
    """
    names = list(sequences.keys())
    n = len(names)

    print(f"  Computing k-mer profiles for {n} sequences...")
    kmers = {name: get_kmers(seq, k) for name, seq in tqdm(sequences.items())}

    print(f"  Computing pairwise distances ({n*(n-1)//2} pairs)...")
    # Use condensed form for efficiency
    distances = np.zeros(n * (n - 1) // 2)

    idx = 0
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            distances[idx] = jaccard_distance(kmers[names[i]], kmers[names[j]])
            idx += 1

    return distances, names


def cluster_hierarchical(distances, n_samples, threshold, method="average"):
    """
    Perform hierarchical clustering with distance threshold.

    Args:
        distances: Condensed distance matrix
        n_samples: Number of samples
        threshold: Distance threshold for flat clustering
        method: Linkage method

    Returns:
        Cluster labels (0-indexed)
    """
    Z = linkage(distances, method=method)
    labels = fcluster(Z, threshold, criterion="distance")
    return labels - 1  # Convert to 0-indexed


def analyze_family_membership(
    family_labels: np.ndarray, is_utp: np.ndarray, names: list
) -> dict:
    """
    Analyze how uTP proteins are distributed across families.

    Returns dict with statistics about family membership.
    """
    # Build family -> members mapping
    families = defaultdict(list)
    for i, label in enumerate(family_labels):
        families[label].append({"name": names[i], "is_utp": is_utp[i]})

    # Calculate statistics
    n_families = len(families)
    n_utp = is_utp.sum()
    n_non_utp = len(is_utp) - n_utp

    # Families containing at least one uTP protein
    utp_families = set()
    for fam_id, members in families.items():
        if any(m["is_utp"] for m in members):
            utp_families.add(fam_id)

    # How many uTP proteins share a family with another uTP protein?
    utp_sharing_family = 0
    utp_in_singleton_family = 0

    for fam_id in utp_families:
        members = families[fam_id]
        utp_members = [m for m in members if m["is_utp"]]

        if len(utp_members) >= 2:
            utp_sharing_family += len(utp_members)
        elif len(utp_members) == 1:
            utp_in_singleton_family += 1

    # Family sizes
    family_sizes = [len(members) for members in families.values()]
    utp_per_family = {}
    for fam_id, members in families.items():
        utp_count = sum(1 for m in members if m["is_utp"])
        utp_per_family[fam_id] = utp_count

    # Families with multiple uTP proteins
    multi_utp_families = sum(1 for c in utp_per_family.values() if c >= 2)

    return {
        "n_families": n_families,
        "n_utp_proteins": n_utp,
        "n_non_utp_proteins": n_non_utp,
        "families_with_utp": len(utp_families),
        "families_with_multi_utp": multi_utp_families,
        "utp_sharing_family": utp_sharing_family,
        "utp_in_singleton_family": utp_in_singleton_family,
        "fraction_utp_sharing": utp_sharing_family / n_utp if n_utp > 0 else 0,
        "mean_family_size": np.mean(family_sizes),
        "median_family_size": np.median(family_sizes),
        "max_family_size": max(family_sizes),
        "singleton_families": sum(1 for s in family_sizes if s == 1),
        "families": families,  # Full data for downstream analysis
    }


def main():
    print("=" * 70)
    print("Cluster Proteins into Gene Families")
    print("=" * 70)

    # =========================================================================
    # Step 1: Load data
    # =========================================================================
    print("\n[1/4] Loading data...")

    sequences = load_sequences(MATURE_DOMAINS)
    metadata = pd.read_csv(METADATA_FILE)

    print(f"  Loaded {len(sequences)} mature domain sequences")
    print(f"  Metadata for {len(metadata)} proteins")

    # Create lookup for uTP status
    utp_lookup = dict(zip(metadata["protein_id"], metadata["has_utp"]))

    # Optional: subsample for speed
    if MAX_PROTEINS and len(sequences) > MAX_PROTEINS:
        print(f"  Subsampling to {MAX_PROTEINS} proteins...")
        # Prioritize uTP proteins in sample
        utp_ids = [pid for pid in sequences if utp_lookup.get(pid, False)]
        non_utp_ids = [pid for pid in sequences if not utp_lookup.get(pid, False)]

        n_non_utp = MAX_PROTEINS - len(utp_ids)
        np.random.seed(42)
        sampled_non_utp = list(np.random.choice(non_utp_ids, n_non_utp, replace=False))

        sampled_ids = set(utp_ids + sampled_non_utp)
        sequences = {k: v for k, v in sequences.items() if k in sampled_ids}
        print(f"  Subsampled to {len(sequences)} proteins")

    # =========================================================================
    # Step 2: Compute distance matrix
    # =========================================================================
    print("\n[2/4] Computing distance matrix...")

    # Check if cached
    if DISTANCE_MATRIX.exists():
        print("  Loading cached distance matrix...")
        data = np.load(DISTANCE_MATRIX, allow_pickle=True)
        distances = data["distances"]
        names = list(data["names"])

        # Verify it matches current sequences
        if set(names) != set(sequences.keys()):
            print("  Cache outdated, recomputing...")
            distances, names = compute_distance_matrix(sequences, KMER_SIZE)
            np.savez(DISTANCE_MATRIX, distances=distances, names=names)
    else:
        distances, names = compute_distance_matrix(sequences, KMER_SIZE)
        np.savez(DISTANCE_MATRIX, distances=distances, names=names)
        print(f"  Saved distance matrix to {DISTANCE_MATRIX}")

    # Get uTP status array aligned with names
    is_utp = np.array([utp_lookup.get(name, False) for name in names])

    print(f"  Distance matrix: {len(names)} proteins")
    print(f"  uTP proteins: {is_utp.sum()}")

    # =========================================================================
    # Step 3: Cluster at multiple thresholds
    # =========================================================================
    print("\n[3/4] Clustering at multiple distance thresholds...")

    results = []
    best_result = None

    for threshold in FAMILY_DISTANCE_THRESHOLDS:
        print(f"\n  Threshold: {threshold} (Jaccard distance)")

        labels = cluster_hierarchical(distances, len(names), threshold)
        stats = analyze_family_membership(labels, is_utp, names)

        # Remove full families data for summary
        stats_summary = {k: v for k, v in stats.items() if k != "families"}
        stats_summary["threshold"] = threshold
        results.append(stats_summary)

        print(f"    Families: {stats['n_families']}")
        print(f"    Families with uTP: {stats['families_with_utp']}")
        print(f"    Families with 2+ uTP: {stats['families_with_multi_utp']}")
        print(f"    Fraction uTP sharing family: {stats['fraction_utp_sharing']:.3f}")

        # Keep best threshold data
        if threshold == 0.7:  # Our target threshold
            best_result = (labels, stats, names)

    # =========================================================================
    # Step 4: Save results
    # =========================================================================
    print("\n[4/4] Saving results...")

    # Save family statistics
    results_df = pd.DataFrame(results)
    results_df.to_csv(FAMILY_STATS, index=False)
    print(f"  Saved family statistics to {FAMILY_STATS}")

    # Save family assignments for best threshold
    if best_result:
        labels, stats, names = best_result

        assignments = []
        for i, name in enumerate(names):
            assignments.append(
                {
                    "protein_id": name,
                    "family_id": labels[i],
                    "is_utp": is_utp[i],
                }
            )

        assignments_df = pd.DataFrame(assignments)

        # Add family size info
        family_sizes = assignments_df.groupby("family_id").size().to_dict()
        assignments_df["family_size"] = assignments_df["family_id"].map(family_sizes)

        # Add uTP count per family
        utp_per_family = (
            assignments_df[assignments_df["is_utp"]]
            .groupby("family_id")
            .size()
            .to_dict()
        )
        assignments_df["utp_in_family"] = assignments_df["family_id"].map(
            lambda x: utp_per_family.get(x, 0)
        )

        assignments_df.to_csv(FAMILY_ASSIGNMENTS, index=False)
        print(f"  Saved family assignments to {FAMILY_ASSIGNMENTS}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary (threshold = 0.7)")
    print("=" * 70)

    if best_result:
        _, stats, _ = best_result
        print(f"\nTotal proteins: {len(names)}")
        print(f"Total families: {stats['n_families']}")
        print(f"Singleton families: {stats['singleton_families']}")
        print(f"\nuTP proteins: {stats['n_utp_proteins']}")
        print(
            f"uTP proteins sharing a family with another uTP: {stats['utp_sharing_family']}"
        )
        print(f"Fraction sharing: {stats['fraction_utp_sharing']:.3f}")

    print(f"\n📁 Output files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
