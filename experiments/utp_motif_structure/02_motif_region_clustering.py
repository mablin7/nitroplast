#!/usr/bin/env python3
"""
Approach 2: Motif Region Only Clustering

Instead of clustering the full uTP sequence (which includes variable-length
inter-motif regions), extract just the conserved motif-containing region
from first motif hit to C-terminus.

This tests whether the clustering failure was due to noise from variable regions.
"""

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MAST_XML = PROJECT_ROOT / "experiments/utp_motif_coverage/output/mast_results/mast.xml"
IMPORT_FASTA = PROJECT_ROOT / "data/Import_candidates.fasta"
UTP_SEQUENCES = (
    PROJECT_ROOT / "experiments/utp_sequence_clustering/output/utp_sequences.fasta"
)
OUTPUT_DIR = Path(__file__).parent / "output"


def parse_mast_results(mast_xml_path: Path) -> dict:
    """Parse MAST XML to get motif positions for each sequence."""
    tree = ET.parse(mast_xml_path)
    root = tree.getroot()

    results = {}
    for seq_elem in root.findall(".//sequence"):
        seq_id = seq_elem.attrib.get("name")
        seq_length = int(seq_elem.attrib.get("length", 0))

        hits = []
        for hit in seq_elem.findall(".//hit"):
            hits.append(
                {
                    "motif_idx": hit.attrib.get("idx"),
                    "position": int(hit.attrib.get("pos", 0)),
                    "pvalue": float(hit.attrib.get("pvalue", 1.0)),
                }
            )

        if hits:
            hits.sort(key=lambda x: x["position"])
            results[seq_id] = {
                "length": seq_length,
                "hits": hits,
                "first_motif_pos": min(h["position"] for h in hits),
                "last_motif_pos": max(h["position"] for h in hits),
            }

    return results


def load_sequences(fasta_path: Path) -> dict:
    """Load sequences from FASTA file."""
    sequences = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences


def extract_motif_regions(
    sequences: dict, mast_results: dict, mode: str = "first_to_end"
) -> dict:
    """
    Extract motif-containing regions from sequences.

    Modes:
    - "first_to_end": From first motif position to C-terminus
    - "first_to_last": From first motif to end of last motif
    - "fixed_window": Fixed-length window around motif center
    """
    regions = {}

    for seq_id, seq in sequences.items():
        if seq_id not in mast_results:
            continue

        info = mast_results[seq_id]

        if mode == "first_to_end":
            # From first motif to end of sequence
            start = max(0, info["first_motif_pos"] - 1)  # 1-indexed to 0-indexed
            regions[seq_id] = seq[start:]

        elif mode == "first_to_last":
            # From first motif to end of last motif (+ some buffer)
            start = max(0, info["first_motif_pos"] - 1)
            end = min(len(seq), info["last_motif_pos"] + 30)  # +30 for motif width
            regions[seq_id] = seq[start:end]

        elif mode == "fixed_window":
            # Fixed 150aa from first motif
            start = max(0, info["first_motif_pos"] - 1)
            end = min(len(seq), start + 150)
            regions[seq_id] = seq[start:end]

    return regions


def compute_kmer_features(sequences: dict, k: int = 4) -> tuple[np.ndarray, list[str]]:
    """Compute k-mer frequency vectors."""
    from itertools import product

    # All possible k-mers
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    all_kmers = ["".join(p) for p in product(amino_acids, repeat=k)]
    kmer_to_idx = {kmer: i for i, kmer in enumerate(all_kmers)}

    names = list(sequences.keys())
    features = np.zeros((len(names), len(all_kmers)))

    for i, (name, seq) in enumerate(sequences.items()):
        seq = seq.upper()
        kmer_counts = Counter(seq[j : j + k] for j in range(len(seq) - k + 1))
        total = sum(kmer_counts.values())

        for kmer, count in kmer_counts.items():
            if kmer in kmer_to_idx:
                features[i, kmer_to_idx[kmer]] = count / total if total > 0 else 0

    return features, names


def compute_embeddings(
    sequences: dict, output_file: Path
) -> tuple[np.ndarray, list[str]]:
    """Compute ProtT5 embeddings for motif regions."""

    # Check for cached embeddings
    if output_file.exists():
        print("    Loading cached embeddings...")
        embeddings = []
        names = []
        with h5py.File(output_file, "r") as f:
            for name in sequences:
                if name in f:
                    embeddings.append(f[name][:])
                    names.append(name)
        if embeddings:
            return np.array(embeddings), names

    try:
        import torch
        from transformers import T5EncoderModel, T5Tokenizer
    except ImportError:
        print("    transformers not available, using k-mer features instead")
        return None, None

    print("    Loading ProtT5 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False
    )
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device)
    model.eval()

    embeddings = []
    names = list(sequences.keys())

    print(f"    Computing embeddings for {len(names)} sequences...")
    with torch.no_grad():
        for i, (name, seq) in enumerate(sequences.items()):
            if (i + 1) % 100 == 0:
                print(f"      {i + 1}/{len(names)}")

            # Add spaces between amino acids
            seq_spaced = " ".join(list(seq))

            inputs = tokenizer(seq_spaced, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            embeddings.append(embedding)

    embeddings = np.array(embeddings)

    # Save embeddings
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file, "w") as f:
        for i, name in enumerate(names):
            f.create_dataset(name, data=embeddings[i])

    return embeddings, names


def cluster_and_evaluate(X: np.ndarray, names: list[str], method_name: str) -> dict:
    """Apply multiple clustering methods."""
    results = {}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Hierarchical clustering
    for n_clusters in [4, 5, 6, 7]:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = model.fit_predict(X_scaled)
        name = f"{method_name}_Hier_k{n_clusters}"

        sil = -1
        n_valid = sum(labels >= 0)
        if n_clusters > 1 and n_valid > n_clusters:
            try:
                sil = silhouette_score(X_scaled, labels)
            except:
                pass

        results[name] = {
            "labels": labels,
            "silhouette": sil,
            "n_clusters": n_clusters,
            "sizes": dict(Counter(labels)),
        }

    # KMeans
    for n_clusters in [4, 5, 6, 7]:
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        name = f"{method_name}_KMeans_k{n_clusters}"

        sil = -1
        if n_clusters > 1:
            try:
                sil = silhouette_score(X_scaled, labels)
            except:
                pass

        results[name] = {
            "labels": labels,
            "silhouette": sil,
            "n_clusters": n_clusters,
            "sizes": dict(Counter(labels)),
        }

    return results, X_scaled


def visualize_comparison(results_dict: dict, output_dir: Path):
    """Compare clustering results across different region extraction modes."""

    # Collect silhouette scores
    data = []
    for region_mode, results in results_dict.items():
        for method_name, result in results["clusterings"].items():
            # Parse method details
            parts = method_name.split("_")
            cluster_method = parts[-2]  # Hier or KMeans
            k = int(parts[-1].replace("k", ""))

            data.append(
                {
                    "region_mode": region_mode,
                    "method": cluster_method,
                    "k": k,
                    "silhouette": result["silhouette"],
                }
            )

    df = pd.DataFrame(data)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Hierarchical comparison
    hier_df = df[df["method"] == "Hier"]
    ax = axes[0]
    for region in hier_df["region_mode"].unique():
        subset = hier_df[hier_df["region_mode"] == region]
        ax.plot(subset["k"], subset["silhouette"], marker="o", label=region)
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Hierarchical Clustering: Region Mode Comparison")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # KMeans comparison
    kmeans_df = df[df["method"] == "KMeans"]
    ax = axes[1]
    for region in kmeans_df["region_mode"].unique():
        subset = kmeans_df[kmeans_df["region_mode"] == region]
        ax.plot(subset["k"], subset["silhouette"], marker="o", label=region)
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("KMeans Clustering: Region Mode Comparison")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "region_mode_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    return df


def visualize_best_clustering(
    X: np.ndarray,
    names: list[str],
    best_labels: np.ndarray,
    title: str,
    output_dir: Path,
    filename: str,
):
    """Create UMAP visualization of best clustering."""

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_2d = reducer.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        X_2d[:, 0], X_2d[:, 1], c=best_labels, cmap="tab10", s=30, alpha=0.7
    )

    n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
    ax.set_title(f"{title}\n({n_clusters} clusters)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.colorbar(scatter, ax=ax, label="Cluster")

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Approach 2: Motif Region Only Clustering")
    print("=" * 70)

    # Parse MAST results for motif positions
    print("\n[1/5] Loading data...")

    if not MAST_XML.exists():
        print(f"  ERROR: MAST results not found at {MAST_XML}")
        return

    mast_results = parse_mast_results(MAST_XML)
    print(f"  Loaded MAST results for {len(mast_results)} sequences")

    # Load full sequences
    sequences = load_sequences(IMPORT_FASTA)
    print(f"  Loaded {len(sequences)} sequences from Import_candidates.fasta")

    # Filter to sequences with motif hits
    sequences = {k: v for k, v in sequences.items() if k in mast_results}
    print(f"  {len(sequences)} sequences have motif hits")

    # Extract regions with different modes
    print("\n[2/5] Extracting motif regions...")

    region_modes = {
        "first_to_end": "First motif to C-terminus",
        "first_to_last": "First to last motif (+30aa)",
        "fixed_window": "Fixed 150aa from first motif",
    }

    all_results = {}

    for mode, description in region_modes.items():
        print(f"\n  --- {mode}: {description} ---")

        regions = extract_motif_regions(sequences, mast_results, mode=mode)
        print(f"    Extracted {len(regions)} regions")

        lengths = [len(r) for r in regions.values()]
        print(
            f"    Length range: {min(lengths)}-{max(lengths)} (median: {np.median(lengths):.0f})"
        )

        # Compute k-mer features
        print(f"    Computing k-mer features...")
        X_kmer, names = compute_kmer_features(regions, k=4)

        # Cluster
        print(f"    Clustering...")
        clusterings, X_scaled = cluster_and_evaluate(X_kmer, names, mode)

        all_results[mode] = {
            "regions": regions,
            "features": X_kmer,
            "names": names,
            "clusterings": clusterings,
        }

        # Print best result
        best = max(clusterings.items(), key=lambda x: x[1]["silhouette"])
        print(f"    Best: {best[0]} (silhouette={best[1]['silhouette']:.3f})")

    # Compare modes
    print("\n[3/5] Comparing region extraction modes...")
    comparison_df = visualize_comparison(all_results, OUTPUT_DIR)
    comparison_df.to_csv(OUTPUT_DIR / "region_mode_comparison.csv", index=False)
    print(f"  Saved comparison to {OUTPUT_DIR / 'region_mode_comparison.csv'}")

    # Find overall best
    best_overall = comparison_df.loc[comparison_df["silhouette"].idxmax()]
    print(
        f"\n  Best overall: {best_overall['region_mode']} + {best_overall['method']}_k{best_overall['k']}"
    )
    print(f"  Silhouette: {best_overall['silhouette']:.3f}")

    # Compute embeddings for best mode and visualize
    print("\n[4/5] Computing embeddings for best mode...")
    best_mode = best_overall["region_mode"]
    best_regions = all_results[best_mode]["regions"]

    embedding_file = OUTPUT_DIR / f"motif_region_embeddings_{best_mode}.h5"
    embeddings, emb_names = compute_embeddings(best_regions, embedding_file)

    if embeddings is not None:
        print(f"    Got embeddings for {len(emb_names)} sequences")

        # Cluster on embeddings
        print("    Clustering on embeddings...")
        emb_clusterings, _ = cluster_and_evaluate(
            embeddings, emb_names, f"{best_mode}_emb"
        )

        best_emb = max(emb_clusterings.items(), key=lambda x: x[1]["silhouette"])
        print(
            f"    Best embedding clustering: {best_emb[0]} (silhouette={best_emb[1]['silhouette']:.3f})"
        )

        # Visualize
        visualize_best_clustering(
            embeddings,
            emb_names,
            best_emb[1]["labels"],
            f"Motif Region Clustering (ProtT5)\n{best_emb[0]}",
            OUTPUT_DIR,
            "motif_region_umap.png",
        )
        print(f"    Saved UMAP to {OUTPUT_DIR / 'motif_region_umap.png'}")

    # Summary comparison with full sequence clustering
    print("\n[5/5] Comparison with full sequence clustering...")
    print("\n" + "=" * 70)
    print("SUMMARY: Motif Region vs Full Sequence Clustering")
    print("=" * 70)
    print(f"\nPrevious full uTP sequence clustering (from utp_sequence_clustering):")
    print(f"  Best silhouette: ~0.08 (KMeans k=4)")
    print(f"\nMotif region only clustering:")
    print(f"  Best mode: {best_overall['region_mode']}")
    print(f"  Best silhouette: {best_overall['silhouette']:.3f}")

    if embeddings is not None and best_emb[1]["silhouette"] > 0:
        print(f"  With embeddings: {best_emb[1]['silhouette']:.3f}")

    improvement = best_overall["silhouette"] - 0.08
    print(f"\n  Improvement: {'+' if improvement > 0 else ''}{improvement:.3f}")

    print(f"\n📁 Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
