#!/usr/bin/env python3
"""
uTP Sequence Clustering with Multiple Methods

This script applies multiple established clustering methods to uTP sequences
and compares the results.

Methods:
1. CD-HIT - identity-based clustering
2. MMseqs2 - modern, fast clustering
3. Hierarchical clustering - on pairwise alignment distances
4. K-means - on ProtT5 embeddings
5. DBSCAN - density-based on embeddings
6. Agglomerative - on all-vs-all alignment similarity

Usage:
    uv run python experiments/utp_sequence_clustering/02_cluster_sequences.py
"""

import subprocess
import tempfile
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP

warnings.filterwarnings("ignore")

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
DATA_DIR = SCRIPT_DIR.parent.parent / "data"

# Input files
UTP_FASTA = OUTPUT_DIR / "utp_sequences.fasta"
UTP_METADATA = OUTPUT_DIR / "utp_metadata.csv"

# Clustering parameters
CDHIT_THRESHOLDS = [0.9, 0.8, 0.7, 0.6]
MMSEQS_THRESHOLDS = [0.9, 0.8, 0.7, 0.6]
N_CLUSTERS_RANGE = range(3, 10)

# External tools
CDHIT_BIN = "cd-hit"
MMSEQS_BIN = "mmseqs"


def load_sequences(fasta_file):
    """Load sequences from FASTA file."""
    return {r.id: str(r.seq) for r in SeqIO.parse(fasta_file, "fasta")}


def check_tool_available(tool_name):
    """Check if an external tool is available."""
    import shutil

    return shutil.which(tool_name) is not None


def run_cdhit(fasta_file, output_prefix, threshold=0.9):
    """
    Run CD-HIT clustering.

    Args:
        fasta_file: Input FASTA file
        output_prefix: Prefix for output files
        threshold: Sequence identity threshold (0-1)

    Returns:
        Dictionary mapping sequence names to cluster IDs
    """
    if not check_tool_available(CDHIT_BIN):
        return None

    output_file = f"{output_prefix}_c{int(threshold*100)}"

    cmd = [
        CDHIT_BIN,
        "-i",
        str(fasta_file),
        "-o",
        output_file,
        "-c",
        str(threshold),
        "-n",
        "5" if threshold >= 0.7 else "4",  # word size
        "-d",
        "0",  # full sequence name
        "-T",
        "4",  # threads
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  CD-HIT warning: {result.stderr[:200]}")
        return None

    # Parse cluster file
    clusters = {}
    cluster_id = -1

    clstr_file = f"{output_file}.clstr"
    if not Path(clstr_file).exists():
        return None

    with open(clstr_file) as f:
        for line in f:
            if line.startswith(">Cluster"):
                cluster_id = int(line.split()[1])
            elif line.strip():
                # Extract sequence name
                parts = line.split(">")[1].split("...")[0]
                clusters[parts] = cluster_id

    # Clean up
    Path(output_file).unlink(missing_ok=True)
    Path(clstr_file).unlink(missing_ok=True)

    return clusters


def run_mmseqs2(fasta_file, output_dir, threshold=0.9):
    """
    Run MMseqs2 clustering.

    Args:
        fasta_file: Input FASTA file
        output_dir: Directory for output files
        threshold: Minimum sequence identity

    Returns:
        Dictionary mapping sequence names to cluster IDs
    """
    if not check_tool_available(MMSEQS_BIN):
        return None

    db_path = output_dir / "mmseqs_db"
    cluster_path = output_dir / "mmseqs_cluster"
    tsv_path = output_dir / "mmseqs_clusters.tsv"

    try:
        # Create database
        subprocess.run(
            [MMSEQS_BIN, "createdb", str(fasta_file), str(db_path)],
            capture_output=True,
            check=True,
        )

        # Cluster
        subprocess.run(
            [
                MMSEQS_BIN,
                "cluster",
                str(db_path),
                str(cluster_path),
                str(output_dir / "tmp"),
                "--min-seq-id",
                str(threshold),
                "-c",
                "0.8",  # coverage
                "--cluster-mode",
                "2",  # connected component
            ],
            capture_output=True,
            check=True,
        )

        # Create TSV output
        subprocess.run(
            [
                MMSEQS_BIN,
                "createtsv",
                str(db_path),
                str(db_path),
                str(cluster_path),
                str(tsv_path),
            ],
            capture_output=True,
            check=True,
        )

        # Parse clusters
        clusters = {}
        cluster_reps = {}

        with open(tsv_path) as f:
            for line in f:
                rep, member = line.strip().split("\t")
                if rep not in cluster_reps:
                    cluster_reps[rep] = len(cluster_reps)
                clusters[member] = cluster_reps[rep]

        return clusters

    except subprocess.CalledProcessError as e:
        print(f"  MMseqs2 error: {e}")
        return None
    finally:
        # Clean up
        for pattern in ["mmseqs_*", "tmp"]:
            for f in output_dir.glob(pattern):
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    import shutil

                    shutil.rmtree(f, ignore_errors=True)


def compute_pairwise_distances(sequences, method="identity"):
    """
    Compute pairwise distance matrix.

    Args:
        sequences: Dictionary of name -> sequence
        method: 'identity' for sequence identity, 'alignment' for alignment score

    Returns:
        Distance matrix and sequence names
    """
    names = list(sequences.keys())
    n = len(names)

    if method == "identity":
        # Simple k-mer based identity approximation
        from collections import Counter

        def kmer_vector(seq, k=3):
            kmers = Counter(seq[i : i + k] for i in range(len(seq) - k + 1))
            return kmers

        kmer_vecs = {name: kmer_vector(seq) for name, seq in sequences.items()}

        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                v1, v2 = kmer_vecs[names[i]], kmer_vecs[names[j]]

                # Jaccard distance
                intersection = sum((v1 & v2).values())
                union = sum((v1 | v2).values())
                similarity = intersection / union if union > 0 else 0
                dist_matrix[i, j] = dist_matrix[j, i] = 1 - similarity

        return dist_matrix, names

    else:
        raise ValueError(f"Unknown method: {method}")


def load_embeddings(embedding_file, sequences):
    """Load ProtT5 embeddings if available."""
    if not embedding_file.exists():
        return None, None

    embeddings = []
    names = []

    with h5py.File(embedding_file, "r") as f:
        for name in sequences:
            if name in f:
                embeddings.append(f[name][:])
                names.append(name)

    if not embeddings:
        return None, None

    return np.array(embeddings), names


def compute_embeddings_prott5(sequences, output_file):
    """Compute ProtT5 embeddings for sequences."""
    try:
        import torch
        from transformers import T5EncoderModel, T5Tokenizer
    except ImportError:
        print("  transformers not available, skipping ProtT5 embeddings")
        return None, None

    print("  Loading ProtT5 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False
    )
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device)
    model.eval()

    embeddings = []
    names = list(sequences.keys())

    print(f"  Computing embeddings for {len(names)} sequences...")
    with torch.no_grad():
        for i, (name, seq) in enumerate(sequences.items()):
            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{len(names)}")

            # Add spaces between amino acids
            seq_spaced = " ".join(list(seq))

            inputs = tokenizer(seq_spaced, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            embeddings.append(embedding)

    embeddings = np.array(embeddings)

    # Save embeddings
    with h5py.File(output_file, "w") as f:
        for i, name in enumerate(names):
            f.create_dataset(name, data=embeddings[i])

    return embeddings, names


def cluster_hierarchical(dist_matrix, names, n_clusters=5, method="average"):
    """
    Perform hierarchical clustering.

    Args:
        dist_matrix: Pairwise distance matrix
        names: Sequence names
        n_clusters: Number of clusters to extract
        method: Linkage method

    Returns:
        Dictionary mapping names to cluster IDs, linkage matrix
    """
    # Convert to condensed form
    condensed = squareform(dist_matrix)

    # Perform hierarchical clustering
    Z = linkage(condensed, method=method)

    # Extract clusters
    cluster_labels = fcluster(Z, n_clusters, criterion="maxclust")

    clusters = {name: int(label) - 1 for name, label in zip(names, cluster_labels)}

    return clusters, Z


def cluster_kmeans(embeddings, names, n_clusters=5):
    """K-means clustering on embeddings."""
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    return {name: int(label) for name, label in zip(names, labels)}


def cluster_dbscan(embeddings, names, eps=0.5, min_samples=5):
    """DBSCAN clustering on embeddings."""
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)

    # Use UMAP for dimensionality reduction before DBSCAN
    reducer = UMAP(n_components=10, random_state=42, n_neighbors=15)
    X_reduced = reducer.fit_transform(X)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_reduced)

    return {name: int(label) for name, label in zip(names, labels)}


def cluster_spectral(dist_matrix, names, n_clusters=5):
    """Spectral clustering on similarity matrix."""
    # Convert distance to similarity
    similarity = 1 - dist_matrix / dist_matrix.max()

    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=42,
        assign_labels="kmeans",
    )
    labels = spectral.fit_predict(similarity)

    return {name: int(label) for name, label in zip(names, labels)}


def evaluate_clustering(clusters, embeddings=None, dist_matrix=None, names=None):
    """Evaluate clustering quality."""
    labels = [clusters[name] for name in names] if names else list(clusters.values())

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise
    cluster_sizes = Counter(labels)

    metrics = {
        "n_clusters": n_clusters,
        "cluster_sizes": dict(cluster_sizes),
        "noise_points": cluster_sizes.get(-1, 0),
    }

    # Silhouette score (if we have embeddings or distance matrix)
    if n_clusters > 1 and n_clusters < len(labels):
        try:
            if embeddings is not None and names:
                X = StandardScaler().fit_transform(embeddings)
                ordered_labels = [clusters[n] for n in names]
                # Filter out noise points for silhouette
                valid_mask = np.array(ordered_labels) >= 0
                if sum(valid_mask) > n_clusters:
                    metrics["silhouette"] = silhouette_score(
                        X[valid_mask], np.array(ordered_labels)[valid_mask]
                    )
            elif dist_matrix is not None and names:
                ordered_labels = [clusters[n] for n in names]
                valid_mask = np.array(ordered_labels) >= 0
                if sum(valid_mask) > n_clusters:
                    metrics["silhouette"] = silhouette_score(
                        dist_matrix[np.ix_(valid_mask, valid_mask)],
                        np.array(ordered_labels)[valid_mask],
                        metric="precomputed",
                    )
        except Exception as e:
            print(f"    Silhouette score error: {e}")

    return metrics


def compare_clusterings(clustering_results, names):
    """Compare multiple clustering results using Adjusted Rand Index."""
    methods = list(clustering_results.keys())
    n = len(methods)

    ari_matrix = np.zeros((n, n))

    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i == j:
                ari_matrix[i, j] = 1.0
            elif i < j:
                c1 = clustering_results[m1]
                c2 = clustering_results[m2]

                # Get common names
                common = set(c1.keys()) & set(c2.keys())
                if len(common) > 0:
                    labels1 = [c1[n] for n in common]
                    labels2 = [c2[n] for n in common]
                    ari = adjusted_rand_score(labels1, labels2)
                    ari_matrix[i, j] = ari_matrix[j, i] = ari

    return pd.DataFrame(ari_matrix, index=methods, columns=methods)


def visualize_embeddings(embeddings, names, clusters_dict, output_dir):
    """Create UMAP visualization of clusters."""

    # UMAP projection
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)

    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_2d = reducer.fit_transform(X)

    # Create visualization for each clustering method
    n_methods = len(clusters_dict)
    fig, axes = plt.subplots(
        2, (n_methods + 1) // 2, figsize=(5 * ((n_methods + 1) // 2), 10)
    )
    axes = axes.flatten()

    for ax, (method, clusters) in zip(axes, clusters_dict.items()):
        labels = np.array([clusters.get(n, -1) for n in names])

        # Color by cluster
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        scatter = ax.scatter(
            X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=10, alpha=0.6
        )
        ax.set_title(f"{method}\n({n_clusters} clusters)")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    # Hide unused axes
    for ax in axes[len(clusters_dict) :]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "cluster_visualization.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "cluster_visualization.svg", bbox_inches="tight")
    plt.close()


def plot_ari_heatmap(ari_df, output_dir):
    """Plot ARI heatmap comparing clustering methods."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        ari_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-0.1,
        vmax=1.0,
        ax=ax,
        square=True,
    )
    ax.set_title("Adjusted Rand Index: Clustering Method Agreement")

    plt.tight_layout()
    plt.savefig(output_dir / "ari_heatmap.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "ari_heatmap.svg", bbox_inches="tight")
    plt.close()


def plot_dendrogram(Z, names, output_dir, max_display=100):
    """Plot hierarchical clustering dendrogram."""
    fig, ax = plt.subplots(figsize=(15, 8))

    # Truncate if too many sequences
    if len(names) > max_display:
        dendrogram(Z, ax=ax, truncate_mode="lastp", p=max_display)
        ax.set_title(
            f"Hierarchical Clustering Dendrogram (truncated to {max_display} leaves)"
        )
    else:
        dendrogram(Z, ax=ax, labels=names, leaf_rotation=90, leaf_font_size=6)
        ax.set_title("Hierarchical Clustering Dendrogram")

    ax.set_ylabel("Distance")

    plt.tight_layout()
    plt.savefig(output_dir / "dendrogram.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "dendrogram.svg", bbox_inches="tight")
    plt.close()


def main():
    print("=" * 70)
    print("uTP Sequence Clustering - Multiple Methods")
    print("=" * 70)

    # Load data
    print("\n[1/6] Loading data...")
    sequences = load_sequences(UTP_FASTA)
    metadata = pd.read_csv(UTP_METADATA)
    print(f"  Loaded {len(sequences)} uTP sequences")

    clustering_results = {}
    evaluation_results = {}

    # =========================================================================
    # Method 1: CD-HIT (if available)
    # =========================================================================
    print("\n[2/6] CD-HIT clustering...")

    if check_tool_available(CDHIT_BIN):
        with tempfile.TemporaryDirectory() as tmpdir:
            for threshold in CDHIT_THRESHOLDS:
                method_name = f"CD-HIT_{int(threshold*100)}%"
                print(f"  Running {method_name}...")

                clusters = run_cdhit(UTP_FASTA, f"{tmpdir}/cdhit", threshold)
                if clusters:
                    clustering_results[method_name] = clusters
                    metrics = evaluate_clustering(clusters, names=list(clusters.keys()))
                    evaluation_results[method_name] = metrics
                    print(f"    {metrics['n_clusters']} clusters")
    else:
        print("  CD-HIT not available, skipping...")

    # =========================================================================
    # Method 2: MMseqs2 (if available)
    # =========================================================================
    print("\n[3/6] MMseqs2 clustering...")

    if check_tool_available(MMSEQS_BIN):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for threshold in MMSEQS_THRESHOLDS:
                method_name = f"MMseqs2_{int(threshold*100)}%"
                print(f"  Running {method_name}...")

                clusters = run_mmseqs2(UTP_FASTA, tmpdir, threshold)
                if clusters:
                    clustering_results[method_name] = clusters
                    metrics = evaluate_clustering(clusters, names=list(clusters.keys()))
                    evaluation_results[method_name] = metrics
                    print(f"    {metrics['n_clusters']} clusters")
    else:
        print("  MMseqs2 not available, skipping...")

    # =========================================================================
    # Method 3: Hierarchical clustering on k-mer distances
    # =========================================================================
    print("\n[4/6] Hierarchical clustering on k-mer distances...")

    dist_matrix, dist_names = compute_pairwise_distances(sequences, method="identity")

    for n_clusters in [4, 5, 6, 7]:
        method_name = f"Hierarchical_k{n_clusters}"
        print(f"  Running {method_name}...")

        clusters, Z = cluster_hierarchical(dist_matrix, dist_names, n_clusters)
        clustering_results[method_name] = clusters
        metrics = evaluate_clustering(
            clusters, dist_matrix=dist_matrix, names=dist_names
        )
        evaluation_results[method_name] = metrics
        print(
            f"    {metrics['n_clusters']} clusters, silhouette={metrics.get('silhouette', 'N/A'):.3f}"
            if "silhouette" in metrics
            else f"    {metrics['n_clusters']} clusters"
        )

    # Save dendrogram for one clustering
    plot_dendrogram(Z, dist_names, OUTPUT_DIR)

    # =========================================================================
    # Method 4: Spectral clustering
    # =========================================================================
    print("\n[5/6] Spectral clustering...")

    for n_clusters in [4, 5, 6, 7]:
        method_name = f"Spectral_k{n_clusters}"
        print(f"  Running {method_name}...")

        try:
            clusters = cluster_spectral(dist_matrix, dist_names, n_clusters)
            clustering_results[method_name] = clusters
            metrics = evaluate_clustering(
                clusters, dist_matrix=dist_matrix, names=dist_names
            )
            evaluation_results[method_name] = metrics
            sil = metrics.get("silhouette", None)
            if sil is not None:
                print(f"    {metrics['n_clusters']} clusters, silhouette={sil:.3f}")
            else:
                print(f"    {metrics['n_clusters']} clusters")
        except Exception as e:
            print(f"    Error: {e}")

    # =========================================================================
    # Method 5: Embedding-based clustering
    # =========================================================================
    print("\n[6/6] Embedding-based clustering...")

    embedding_file = OUTPUT_DIR / "utp_embeddings.h5"
    embeddings, emb_names = load_embeddings(embedding_file, sequences)

    if embeddings is None:
        print("  Computing ProtT5 embeddings (this may take a while)...")
        embeddings, emb_names = compute_embeddings_prott5(sequences, embedding_file)

    if embeddings is not None:
        print(f"  Loaded embeddings for {len(emb_names)} sequences")

        # K-means
        for n_clusters in [4, 5, 6, 7]:
            method_name = f"KMeans_k{n_clusters}"
            print(f"  Running {method_name}...")

            clusters = cluster_kmeans(embeddings, emb_names, n_clusters)
            clustering_results[method_name] = clusters
            metrics = evaluate_clustering(
                clusters, embeddings=embeddings, names=emb_names
            )
            evaluation_results[method_name] = metrics
            sil = metrics.get("silhouette", None)
            if sil is not None:
                print(f"    {metrics['n_clusters']} clusters, silhouette={sil:.3f}")
            else:
                print(f"    {metrics['n_clusters']} clusters")

        # DBSCAN
        for eps in [0.5, 1.0, 2.0]:
            method_name = f"DBSCAN_eps{eps}"
            print(f"  Running {method_name}...")

            clusters = cluster_dbscan(embeddings, emb_names, eps=eps)
            clustering_results[method_name] = clusters
            metrics = evaluate_clustering(
                clusters, embeddings=embeddings, names=emb_names
            )
            evaluation_results[method_name] = metrics
            print(
                f"    {metrics['n_clusters']} clusters, {metrics['noise_points']} noise points"
            )

    # =========================================================================
    # Compare and visualize
    # =========================================================================
    print("\n" + "=" * 70)
    print("Comparing clustering methods...")
    print("=" * 70)

    # Compute ARI between methods
    common_names = list(sequences.keys())
    ari_df = compare_clusterings(clustering_results, common_names)

    # Save ARI matrix
    ari_df.to_csv(OUTPUT_DIR / "ari_matrix.csv")
    plot_ari_heatmap(ari_df, OUTPUT_DIR)
    print(f"  Saved ARI heatmap to {OUTPUT_DIR / 'ari_heatmap.png'}")

    # Visualize clusters (if embeddings available)
    if embeddings is not None:
        # Select a subset of clustering methods for visualization
        viz_methods = {
            k: v
            for k, v in clustering_results.items()
            if any(
                x in k
                for x in ["Hierarchical_k5", "Spectral_k5", "KMeans_k5", "DBSCAN"]
            )
        }
        if viz_methods:
            visualize_embeddings(embeddings, emb_names, viz_methods, OUTPUT_DIR)
            print(
                f"  Saved cluster visualization to {OUTPUT_DIR / 'cluster_visualization.png'}"
            )

    # Save evaluation results
    eval_df = pd.DataFrame(evaluation_results).T
    eval_df.to_csv(OUTPUT_DIR / "clustering_evaluation.csv")
    print(f"  Saved evaluation metrics to {OUTPUT_DIR / 'clustering_evaluation.csv'}")

    # Save cluster assignments
    cluster_df = pd.DataFrame(clustering_results)
    cluster_df.index.name = "sequence"
    cluster_df.to_csv(OUTPUT_DIR / "cluster_assignments.csv")
    print(f"  Saved cluster assignments to {OUTPUT_DIR / 'cluster_assignments.csv'}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\nClustering methods applied: {len(clustering_results)}")
    print("\nBest silhouette scores:")

    silhouette_results = {
        k: v.get("silhouette", -1) for k, v in evaluation_results.items()
    }
    for method, score in sorted(
        silhouette_results.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        if score > 0:
            print(f"  {method}: {score:.3f}")

    print(f"\n📁 Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
