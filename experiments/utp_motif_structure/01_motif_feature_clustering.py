#!/usr/bin/env python3
"""
Approach 1: Motif-Based Feature Matrix Clustering

Instead of clustering raw sequences, cluster on structured motif features:
- Motif presence/absence (binary)
- Motif scores (p-values)
- Motif positions (normalized)
- Motif order patterns
"""

import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MAST_XML = PROJECT_ROOT / "experiments/utp_motif_coverage/output/mast_results/mast.xml"
MEME_XML = PROJECT_ROOT / "experiments/utp_motif_analysis/data/meme_gb.xml"
UTP_METADATA = (
    PROJECT_ROOT / "experiments/utp_sequence_clustering/output/utp_metadata.csv"
)
OUTPUT_DIR = Path(__file__).parent / "output"

# Motif names (from previous analysis)
MOTIF_NAMES = {
    "1": "motif_1",
    "2": "motif_2",
    "3": "motif_3",
    "4": "motif_4",
    "5": "motif_5",
    "6": "motif_6",
    "7": "motif_7",
    "8": "motif_8",
    "9": "motif_9",
    "10": "motif_10",
}


def parse_meme_motifs(meme_xml_path: Path) -> dict:
    """Parse MEME XML to get motif information."""
    tree = ET.parse(meme_xml_path)
    root = tree.getroot()

    motifs = {}
    for motif in root.findall(".//motif"):
        motif_id = motif.attrib.get("id", motif.attrib.get("name"))
        idx = motif.attrib.get("idx", motif_id.replace("motif_", ""))
        motifs[idx] = {
            "id": motif_id,
            "name": motif.attrib.get("name", motif_id),
            "width": int(motif.attrib.get("width", 0)),
            "sites": int(motif.attrib.get("sites", 0)),
        }
    return motifs


def parse_mast_results(mast_xml_path: Path) -> list[dict]:
    """Parse MAST XML to extract motif hits for each sequence."""
    tree = ET.parse(mast_xml_path)
    root = tree.getroot()

    # Get motif index mapping
    motif_mapping = {}
    for motif in root.findall(".//motif"):
        idx = motif.attrib.get("idx")
        motif_id = motif.attrib.get("id", f"motif_{idx}")
        if idx:
            motif_mapping[idx] = motif_id

    results = []
    for seq_elem in root.findall(".//sequence"):
        seq_id = seq_elem.attrib.get("name")
        seq_length = int(seq_elem.attrib.get("length", 0))

        hits = []
        for hit in seq_elem.findall(".//hit"):
            motif_idx = hit.attrib.get("idx")
            motif_id = motif_mapping.get(motif_idx, f"motif_{motif_idx}")

            hits.append(
                {
                    "motif_id": motif_id,
                    "motif_idx": motif_idx,
                    "position": int(hit.attrib.get("pos", 0)),
                    "pvalue": float(hit.attrib.get("pvalue", 1.0)),
                    "strand": hit.attrib.get("strand", "+"),
                }
            )

        # Sort hits by position
        hits.sort(key=lambda x: x["position"])

        results.append(
            {
                "sequence_id": seq_id,
                "length": seq_length,
                "hits": hits,
            }
        )

    return results


def build_feature_matrix(
    mast_results: list[dict], n_motifs: int = 10
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Build a feature matrix from MAST results.

    Features per motif:
    - has_motif (binary): 1 if motif present
    - score (-log10 pvalue): best score if present, 0 otherwise
    - position (normalized): position / length, 0 if absent

    Additional features:
    - n_motifs: total number of motif hits
    - motif_span: (last_pos - first_pos) / length
    - terminal_motif: one-hot encoding of C-terminal motif
    """
    feature_names = []

    # Per-motif features
    for i in range(1, n_motifs + 1):
        feature_names.extend(
            [
                f"has_motif_{i}",
                f"score_motif_{i}",
                f"position_motif_{i}",
            ]
        )

    # Global features
    feature_names.extend(
        [
            "n_motif_hits",
            "motif_span",
            "terminal_4",
            "terminal_5",
            "terminal_7",
            "terminal_9",
            "terminal_other",
        ]
    )

    n_features = len(feature_names)
    sequences = []
    features = []

    for result in mast_results:
        if not result["hits"]:
            continue

        seq_id = result["sequence_id"]
        seq_len = result["length"]
        hits = result["hits"]

        row = np.zeros(n_features)

        # Per-motif features
        motif_best = {}  # Best hit per motif
        for hit in hits:
            motif_num = hit["motif_idx"]
            if (
                motif_num not in motif_best
                or hit["pvalue"] < motif_best[motif_num]["pvalue"]
            ):
                motif_best[motif_num] = hit

        for i in range(1, n_motifs + 1):
            idx_base = (i - 1) * 3
            motif_idx = str(i)

            if motif_idx in motif_best:
                hit = motif_best[motif_idx]
                row[idx_base] = 1  # has_motif
                row[idx_base + 1] = -np.log10(max(hit["pvalue"], 1e-300))  # score
                row[idx_base + 2] = hit["position"] / seq_len  # position

        # Global features
        global_idx = n_motifs * 3
        row[global_idx] = len(hits)  # n_motif_hits

        if len(hits) >= 2:
            positions = [h["position"] for h in hits]
            row[global_idx + 1] = (
                max(positions) - min(positions)
            ) / seq_len  # motif_span

        # Terminal motif (last hit)
        terminal = hits[-1]["motif_idx"]
        terminal_map = {"4": 0, "5": 1, "7": 2, "9": 3}
        if terminal in terminal_map:
            row[global_idx + 2 + terminal_map[terminal]] = 1
        else:
            row[global_idx + 6] = 1  # terminal_other

        sequences.append(seq_id)
        features.append(row)

    return np.array(features), sequences, feature_names


def evaluate_clustering(labels: np.ndarray, X: np.ndarray, name: str) -> dict:
    """Evaluate clustering quality."""
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    metrics = {
        "method": name,
        "n_clusters": n_clusters,
        "cluster_sizes": dict(Counter(labels)),
    }

    # Silhouette score
    valid_mask = labels >= 0
    if n_clusters > 1 and sum(valid_mask) > n_clusters:
        try:
            metrics["silhouette"] = silhouette_score(X[valid_mask], labels[valid_mask])
        except Exception as e:
            print(f"  Silhouette error: {e}")

    return metrics


def cluster_and_evaluate(X: np.ndarray, sequences: list[str]) -> dict:
    """Apply multiple clustering methods and evaluate."""
    results = {}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Hierarchical clustering
    print("  Hierarchical clustering...")
    for n_clusters in [4, 5, 6, 7]:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = model.fit_predict(X_scaled)
        name = f"Hierarchical_k{n_clusters}"
        results[name] = {
            "labels": labels,
            "metrics": evaluate_clustering(labels, X_scaled, name),
        }

    # KMeans clustering
    print("  KMeans clustering...")
    for n_clusters in [4, 5, 6, 7]:
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        name = f"KMeans_k{n_clusters}"
        results[name] = {
            "labels": labels,
            "metrics": evaluate_clustering(labels, X_scaled, name),
        }

    # DBSCAN
    print("  DBSCAN clustering...")
    for eps in [0.5, 1.0, 1.5, 2.0]:
        model = DBSCAN(eps=eps, min_samples=5)
        labels = model.fit_predict(X_scaled)
        name = f"DBSCAN_eps{eps}"
        results[name] = {
            "labels": labels,
            "metrics": evaluate_clustering(labels, X_scaled, name),
        }

    return results


def visualize_features(
    X: np.ndarray,
    sequences: list[str],
    feature_names: list[str],
    clustering_results: dict,
    output_dir: Path,
):
    """Create visualizations of the feature space."""

    # PCA projection
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # t-SNE projection
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)

    # Plot with best clustering from each method type
    best_methods = {}
    for name, result in clustering_results.items():
        method_type = name.split("_")[0]
        sil = result["metrics"].get("silhouette", -1)
        if method_type not in best_methods or sil > best_methods[method_type][1]:
            best_methods[method_type] = (name, sil)

    fig, axes = plt.subplots(2, len(best_methods), figsize=(5 * len(best_methods), 10))
    if len(best_methods) == 1:
        axes = axes.reshape(-1, 1)

    for col, (method_type, (name, sil)) in enumerate(best_methods.items()):
        labels = clustering_results[name]["labels"]

        # PCA plot
        ax = axes[0, col]
        scatter = ax.scatter(
            X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=20, alpha=0.7
        )
        ax.set_title(f"{name}\nSilhouette: {sil:.3f}" if sil > 0 else name)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

        # t-SNE plot
        ax = axes[1, col]
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap="tab10", s=20, alpha=0.7)
        ax.set_title(f"t-SNE: {name}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    plt.tight_layout()
    plt.savefig(
        output_dir / "motif_feature_clustering.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Feature importance heatmap (correlation with cluster assignments)
    fig, ax = plt.subplots(figsize=(14, 8))

    # Use hierarchical k=5 for feature analysis
    if "Hierarchical_k5" in clustering_results:
        labels = clustering_results["Hierarchical_k5"]["labels"]

        # Calculate mean feature values per cluster
        cluster_means = []
        for c in sorted(set(labels)):
            if c >= 0:
                cluster_means.append(X_scaled[labels == c].mean(axis=0))

        if cluster_means:
            cluster_df = pd.DataFrame(
                cluster_means,
                columns=feature_names,
                index=[f"Cluster {i}" for i in range(len(cluster_means))],
            )

            # Select most variable features
            feature_var = cluster_df.var()
            top_features = feature_var.nlargest(20).index

            sns.heatmap(
                cluster_df[top_features].T,
                cmap="RdBu_r",
                center=0,
                annot=True,
                fmt=".2f",
                ax=ax,
            )
            ax.set_title("Top Variable Features by Cluster (Hierarchical k=5)")

    plt.tight_layout()
    plt.savefig(output_dir / "motif_feature_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def analyze_cluster_composition(
    X: np.ndarray,
    sequences: list[str],
    feature_names: list[str],
    clustering_results: dict,
) -> pd.DataFrame:
    """Analyze what distinguishes each cluster."""

    # Use best hierarchical clustering
    best_hier = None
    best_sil = -1
    for name, result in clustering_results.items():
        if name.startswith("Hierarchical"):
            sil = result["metrics"].get("silhouette", -1)
            if sil > best_sil:
                best_sil = sil
                best_hier = name

    if best_hier is None:
        return pd.DataFrame()

    labels = clustering_results[best_hier]["labels"]

    # Build summary per cluster
    summaries = []
    for cluster_id in sorted(set(labels)):
        if cluster_id < 0:
            continue

        mask = labels == cluster_id
        cluster_X = X[mask]

        # Terminal motif distribution
        terminal_cols = [
            i for i, f in enumerate(feature_names) if f.startswith("terminal_")
        ]
        terminal_dist = cluster_X[:, terminal_cols].sum(axis=0)
        terminal_names = [feature_names[i] for i in terminal_cols]
        dominant_terminal = terminal_names[np.argmax(terminal_dist)]

        # Average motif presence
        has_cols = [i for i, f in enumerate(feature_names) if f.startswith("has_motif")]
        avg_motifs = cluster_X[:, has_cols].mean(axis=0)

        summaries.append(
            {
                "cluster": cluster_id,
                "size": mask.sum(),
                "dominant_terminal": dominant_terminal,
                "avg_n_motifs": cluster_X[
                    :, feature_names.index("n_motif_hits")
                ].mean(),
                "avg_motif_span": cluster_X[
                    :, feature_names.index("motif_span")
                ].mean(),
                **{
                    f"avg_{feature_names[i]}": avg_motifs[j]
                    for j, i in enumerate(has_cols)
                },
            }
        )

    return pd.DataFrame(summaries)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Approach 1: Motif-Based Feature Matrix Clustering")
    print("=" * 70)

    # Parse MAST results
    print("\n[1/4] Parsing MAST results...")
    if not MAST_XML.exists():
        print(f"  ERROR: MAST results not found at {MAST_XML}")
        print("  Run experiments/utp_motif_coverage/analyze_motif_coverage.py first")
        return

    mast_results = parse_mast_results(MAST_XML)
    print(f"  Found {len(mast_results)} sequences with MAST results")

    # Build feature matrix
    print("\n[2/4] Building motif feature matrix...")
    X, sequences, feature_names = build_feature_matrix(mast_results)
    print(f"  Created feature matrix: {X.shape[0]} sequences x {X.shape[1]} features")
    print(f"  Features: {feature_names[:10]}... (and {len(feature_names) - 10} more)")

    # Save feature matrix
    feature_df = pd.DataFrame(X, columns=feature_names, index=sequences)
    feature_df.to_csv(OUTPUT_DIR / "motif_features.csv")
    print(f"  Saved feature matrix to {OUTPUT_DIR / 'motif_features.csv'}")

    # Cluster
    print("\n[3/4] Clustering on motif features...")
    clustering_results = cluster_and_evaluate(X, sequences)

    # Print results
    print("\n" + "-" * 50)
    print("Clustering Results:")
    print("-" * 50)

    for name, result in sorted(
        clustering_results.items(),
        key=lambda x: x[1]["metrics"].get("silhouette", -1),
        reverse=True,
    ):
        metrics = result["metrics"]
        sil = metrics.get("silhouette", "N/A")
        sil_str = f"{sil:.3f}" if isinstance(sil, float) else sil
        sizes = list(metrics["cluster_sizes"].values())
        print(f"  {name}: silhouette={sil_str}, sizes={sizes}")

    # Visualize
    print("\n[4/4] Creating visualizations...")
    visualize_features(X, sequences, feature_names, clustering_results, OUTPUT_DIR)
    print(
        f"  Saved clustering visualization to {OUTPUT_DIR / 'motif_feature_clustering.png'}"
    )
    print(f"  Saved feature heatmap to {OUTPUT_DIR / 'motif_feature_heatmap.png'}")

    # Cluster composition analysis
    composition_df = analyze_cluster_composition(
        X, sequences, feature_names, clustering_results
    )
    if not composition_df.empty:
        composition_df.to_csv(OUTPUT_DIR / "cluster_composition.csv", index=False)
        print(
            f"  Saved cluster composition to {OUTPUT_DIR / 'cluster_composition.csv'}"
        )
        print("\nCluster Composition:")
        print(
            composition_df[
                ["cluster", "size", "dominant_terminal", "avg_n_motifs"]
            ].to_string(index=False)
        )

    # Save all clustering assignments
    assignments = {"sequence": sequences}
    for name, result in clustering_results.items():
        assignments[name] = result["labels"]
    pd.DataFrame(assignments).to_csv(
        OUTPUT_DIR / "motif_cluster_assignments.csv", index=False
    )

    # Save evaluation metrics
    metrics_list = [result["metrics"] for result in clustering_results.values()]
    pd.DataFrame(metrics_list).to_csv(
        OUTPUT_DIR / "motif_clustering_evaluation.csv", index=False
    )

    print(f"\n✅ Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
