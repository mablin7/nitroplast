#!/usr/bin/env python3
"""
Approach 3: Sequence Similarity Network Analysis

Build a network where:
- Nodes = uTP sequences
- Edges = sequence similarity above threshold

Analyze with community detection to find natural groupings that
may not emerge from hard clustering approaches.
"""

import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from scipy.spatial.distance import pdist, squareform

try:
    import networkx as nx
    from community import community_louvain

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: networkx or python-louvain not available")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MAST_XML = PROJECT_ROOT / "experiments/utp_motif_coverage/output/mast_results/mast.xml"
UTP_FASTA = (
    PROJECT_ROOT / "experiments/utp_sequence_clustering/output/utp_sequences.fasta"
)
OUTPUT_DIR = Path(__file__).parent / "output"


def load_sequences(fasta_path: Path) -> dict:
    """Load sequences from FASTA file."""
    sequences = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences


def parse_mast_terminal_motifs(mast_xml_path: Path) -> dict:
    """Get terminal motif for each sequence."""
    tree = ET.parse(mast_xml_path)
    root = tree.getroot()

    terminal_motifs = {}
    for seq_elem in root.findall(".//sequence"):
        seq_id = seq_elem.attrib.get("name")

        hits = []
        for hit in seq_elem.findall(".//hit"):
            hits.append(
                {
                    "motif_idx": hit.attrib.get("idx"),
                    "position": int(hit.attrib.get("pos", 0)),
                }
            )

        if hits:
            hits.sort(key=lambda x: x["position"])
            terminal_motifs[seq_id] = f"terminal_{hits[-1]['motif_idx']}"

    return terminal_motifs


def compute_kmer_similarity(seq1: str, seq2: str, k: int = 3) -> float:
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


def build_similarity_matrix(
    sequences: dict, k: int = 3
) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise similarity matrix."""
    names = list(sequences.keys())
    n = len(names)

    print(f"  Computing {n * (n - 1) // 2} pairwise similarities...")

    similarity = np.zeros((n, n))

    for i in range(n):
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{n} sequences processed...")

        for j in range(i + 1, n):
            sim = compute_kmer_similarity(sequences[names[i]], sequences[names[j]], k=k)
            similarity[i, j] = similarity[j, i] = sim

        similarity[i, i] = 1.0

    return similarity, names


def build_network(
    similarity: np.ndarray, names: list[str], threshold: float = 0.3
) -> "nx.Graph":
    """Build a network from similarity matrix."""
    if not NETWORKX_AVAILABLE:
        raise ImportError("networkx required for network analysis")

    G = nx.Graph()
    G.add_nodes_from(names)

    n = len(names)
    edge_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] >= threshold:
                G.add_edge(names[i], names[j], weight=similarity[i, j])
                edge_count += 1

    return G


def analyze_network(G: "nx.Graph") -> dict:
    """Compute network statistics."""
    stats = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
    }

    # Connected components
    components = list(nx.connected_components(G))
    stats["n_components"] = len(components)
    stats["largest_component_size"] = (
        max(len(c) for c in components) if components else 0
    )

    # Degree statistics
    degrees = [d for n, d in G.degree()]
    if degrees:
        stats["avg_degree"] = np.mean(degrees)
        stats["max_degree"] = max(degrees)
        stats["isolated_nodes"] = sum(1 for d in degrees if d == 0)

    # Clustering coefficient (local transitivity)
    try:
        stats["avg_clustering"] = nx.average_clustering(G)
    except:
        stats["avg_clustering"] = 0

    return stats


def detect_communities(G: "nx.Graph", method: str = "louvain") -> dict:
    """Detect communities in the network."""

    if method == "louvain":
        # Louvain community detection
        partition = community_louvain.best_partition(G, random_state=42)
        return partition

    elif method == "greedy_modularity":
        # Greedy modularity optimization
        from networkx.algorithms.community import greedy_modularity_communities

        communities = greedy_modularity_communities(G)
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        return partition

    elif method == "label_propagation":
        # Label propagation
        from networkx.algorithms.community import label_propagation_communities

        communities = list(label_propagation_communities(G))
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        return partition

    else:
        raise ValueError(f"Unknown method: {method}")


def evaluate_communities(partition: dict, terminal_motifs: dict) -> dict:
    """Evaluate how well communities align with terminal motifs."""

    # Get community assignments for nodes with terminal motifs
    community_terminal = defaultdict(list)
    terminal_community = defaultdict(list)

    for node, comm in partition.items():
        if node in terminal_motifs:
            terminal = terminal_motifs[node]
            community_terminal[comm].append(terminal)
            terminal_community[terminal].append(comm)

    # Calculate purity (how homogeneous are communities)
    total_correct = 0
    total_nodes = 0

    for comm, terminals in community_terminal.items():
        if terminals:
            most_common = Counter(terminals).most_common(1)[0][1]
            total_correct += most_common
            total_nodes += len(terminals)

    purity = total_correct / total_nodes if total_nodes > 0 else 0

    # Calculate completeness (how well are terminal groups captured)
    completeness_scores = {}
    for terminal, comms in terminal_community.items():
        if comms:
            most_common_count = Counter(comms).most_common(1)[0][1]
            completeness_scores[terminal] = most_common_count / len(comms)

    avg_completeness = (
        np.mean(list(completeness_scores.values())) if completeness_scores else 0
    )

    return {
        "n_communities": len(set(partition.values())),
        "purity": purity,
        "avg_completeness": avg_completeness,
        "community_sizes": dict(Counter(partition.values())),
        "completeness_by_terminal": completeness_scores,
    }


def visualize_network(
    G: "nx.Graph",
    partition: dict,
    terminal_motifs: dict,
    output_dir: Path,
    max_nodes: int = 500,
):
    """Visualize the network with community coloring."""

    # Subsample if too large
    if G.number_of_nodes() > max_nodes:
        print(
            f"  Subsampling network from {G.number_of_nodes()} to {max_nodes} nodes..."
        )
        # Keep nodes with highest degree
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()
        partition = {n: partition[n] for n in G.nodes() if n in partition}
        terminal_motifs = {
            n: terminal_motifs[n] for n in G.nodes() if n in terminal_motifs
        }

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Layout
    pos = nx.spring_layout(
        G, k=1 / np.sqrt(G.number_of_nodes()), iterations=50, seed=42
    )

    # Plot 1: Color by community
    ax = axes[0]
    communities = [partition.get(n, -1) for n in G.nodes()]

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=communities, node_size=30, cmap="tab20", alpha=0.7
    )
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.1, width=0.5)
    ax.set_title(f"Network Communities (n={len(set(communities))})")
    ax.axis("off")

    # Plot 2: Color by terminal motif
    ax = axes[1]
    terminal_map = {"terminal_4": 0, "terminal_5": 1, "terminal_7": 2, "terminal_9": 3}
    colors = [terminal_map.get(terminal_motifs.get(n, "other"), 4) for n in G.nodes()]

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=colors, node_size=30, cmap="Set1", alpha=0.7
    )
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.1, width=0.5)
    ax.set_title("Terminal Motifs")
    ax.axis("off")

    # Add legend for terminal motifs
    legend_elements = [
        plt.scatter([], [], c=f"C{i}", s=50, label=f"terminal_{m}")
        for m, i in terminal_map.items()
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_dir / "similarity_network.png", dpi=150, bbox_inches="tight")
    plt.close()


def visualize_threshold_sweep(
    similarity: np.ndarray, names: list[str], terminal_motifs: dict, output_dir: Path
):
    """Analyze network properties across different similarity thresholds."""

    thresholds = np.arange(0.1, 0.6, 0.05)
    results = []

    print("  Sweeping similarity thresholds...")
    for thresh in thresholds:
        G = build_network(similarity, names, threshold=thresh)
        stats = analyze_network(G)

        # Community detection
        if G.number_of_edges() > 0:
            try:
                partition = detect_communities(G, method="louvain")
                comm_stats = evaluate_communities(partition, terminal_motifs)
            except Exception as e:
                print(f"    Threshold {thresh:.2f}: community detection failed ({e})")
                comm_stats = {"n_communities": 0, "purity": 0, "avg_completeness": 0}
        else:
            comm_stats = {"n_communities": 0, "purity": 0, "avg_completeness": 0}

        results.append(
            {
                "threshold": thresh,
                **stats,
                **{
                    f"comm_{k}": v
                    for k, v in comm_stats.items()
                    if not isinstance(v, dict)
                },
            }
        )

    df = pd.DataFrame(results)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Network density
    ax = axes[0, 0]
    ax.plot(df["threshold"], df["density"], marker="o")
    ax.set_xlabel("Similarity Threshold")
    ax.set_ylabel("Network Density")
    ax.set_title("Network Density vs Threshold")

    # Number of communities
    ax = axes[0, 1]
    ax.plot(df["threshold"], df["comm_n_communities"], marker="o", color="orange")
    ax.set_xlabel("Similarity Threshold")
    ax.set_ylabel("Number of Communities")
    ax.set_title("Communities Detected vs Threshold")

    # Community purity
    ax = axes[1, 0]
    ax.plot(df["threshold"], df["comm_purity"], marker="o", color="green")
    ax.set_xlabel("Similarity Threshold")
    ax.set_ylabel("Community Purity")
    ax.set_title("Community Purity (Terminal Motif Homogeneity)")
    ax.set_ylim(0, 1)

    # Average clustering coefficient
    ax = axes[1, 1]
    ax.plot(df["threshold"], df["avg_clustering"], marker="o", color="purple")
    ax.set_xlabel("Similarity Threshold")
    ax.set_ylabel("Avg Clustering Coefficient")
    ax.set_title("Local Transitivity vs Threshold")

    plt.tight_layout()
    plt.savefig(output_dir / "threshold_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()

    return df


def analyze_hub_sequences(
    G: "nx.Graph", terminal_motifs: dict, sequences: dict, n_hubs: int = 20
) -> pd.DataFrame:
    """Identify and analyze hub sequences (high-degree nodes)."""

    degrees = dict(G.degree())
    top_hubs = sorted(degrees, key=degrees.get, reverse=True)[:n_hubs]

    hub_data = []
    for node in top_hubs:
        neighbors = list(G.neighbors(node))
        neighbor_terminals = [terminal_motifs.get(n, "unknown") for n in neighbors]
        terminal_dist = Counter(neighbor_terminals)

        hub_data.append(
            {
                "sequence": node,
                "degree": degrees[node],
                "terminal": terminal_motifs.get(node, "unknown"),
                "seq_length": len(sequences.get(node, "")),
                "neighbor_terminal_dist": dict(terminal_dist),
                "dominant_neighbor_terminal": (
                    terminal_dist.most_common(1)[0][0] if terminal_dist else "none"
                ),
            }
        )

    return pd.DataFrame(hub_data)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Approach 3: Sequence Similarity Network Analysis")
    print("=" * 70)

    if not NETWORKX_AVAILABLE:
        print("\nERROR: networkx and python-louvain required for network analysis")
        print("Install with: pip install networkx python-louvain")
        return

    # Load data
    print("\n[1/6] Loading data...")

    if not UTP_FASTA.exists():
        print(f"  ERROR: uTP sequences not found at {UTP_FASTA}")
        print(
            "  Run experiments/utp_sequence_clustering/01_extract_utp_sequences.py first"
        )
        return

    sequences = load_sequences(UTP_FASTA)
    print(f"  Loaded {len(sequences)} uTP sequences")

    terminal_motifs = {}
    if MAST_XML.exists():
        terminal_motifs = parse_mast_terminal_motifs(MAST_XML)
        print(f"  Loaded terminal motifs for {len(terminal_motifs)} sequences")

    # Compute similarity matrix
    print("\n[2/6] Computing similarity matrix...")
    similarity, names = build_similarity_matrix(sequences, k=3)

    # Save similarity matrix (compressed)
    print(f"  Saving similarity matrix...")
    np.savez_compressed(
        OUTPUT_DIR / "similarity_matrix.npz", similarity=similarity, names=names
    )

    # Summary statistics
    triu_vals = similarity[np.triu_indices_from(similarity, k=1)]
    print(f"  Similarity distribution:")
    print(f"    Min: {triu_vals.min():.3f}")
    print(f"    Max: {triu_vals.max():.3f}")
    print(f"    Mean: {triu_vals.mean():.3f}")
    print(f"    Median: {np.median(triu_vals):.3f}")

    # Threshold sweep
    print("\n[3/6] Analyzing threshold effects...")
    sweep_df = visualize_threshold_sweep(similarity, names, terminal_motifs, OUTPUT_DIR)
    sweep_df.to_csv(OUTPUT_DIR / "threshold_sweep.csv", index=False)
    print(f"  Saved threshold analysis to {OUTPUT_DIR / 'threshold_sweep.csv'}")

    # Find optimal threshold (maximize purity while keeping reasonable connectivity)
    # Heuristic: choose threshold where we get meaningful communities
    valid_thresholds = sweep_df[
        (sweep_df["comm_n_communities"] >= 3)
        & (sweep_df["comm_n_communities"] <= 20)
        & (sweep_df["density"] > 0.01)
    ]

    if not valid_thresholds.empty:
        # Maximize purity among valid thresholds
        best_idx = valid_thresholds["comm_purity"].idxmax()
        best_threshold = valid_thresholds.loc[best_idx, "threshold"]
    else:
        # Default
        best_threshold = 0.25

    print(f"\n  Selected threshold: {best_threshold:.2f}")

    # Build network at optimal threshold
    print("\n[4/6] Building network...")
    G = build_network(similarity, names, threshold=best_threshold)
    stats = analyze_network(G)

    print(f"  Network statistics:")
    print(f"    Nodes: {stats['n_nodes']}")
    print(f"    Edges: {stats['n_edges']}")
    print(f"    Density: {stats['density']:.4f}")
    print(f"    Components: {stats['n_components']}")
    print(f"    Avg degree: {stats.get('avg_degree', 0):.1f}")
    print(f"    Avg clustering: {stats.get('avg_clustering', 0):.3f}")

    # Community detection
    print("\n[5/6] Detecting communities...")

    community_results = {}
    methods = ["louvain", "greedy_modularity", "label_propagation"]

    for method in methods:
        try:
            print(f"  {method}...")
            partition = detect_communities(G, method=method)
            comm_stats = evaluate_communities(partition, terminal_motifs)
            community_results[method] = {
                "partition": partition,
                "stats": comm_stats,
            }
            print(f"    Communities: {comm_stats['n_communities']}")
            print(f"    Purity: {comm_stats['purity']:.3f}")
            print(f"    Completeness: {comm_stats['avg_completeness']:.3f}")
        except Exception as e:
            print(f"    Error: {e}")

    # Use best method (highest purity)
    if community_results:
        best_method = max(
            community_results, key=lambda x: community_results[x]["stats"]["purity"]
        )
        best_partition = community_results[best_method]["partition"]
        best_stats = community_results[best_method]["stats"]

        print(f"\n  Best method: {best_method}")
        print(f"    Purity: {best_stats['purity']:.3f}")

        # Save community assignments
        comm_df = pd.DataFrame(
            [
                {
                    "sequence": node,
                    "community": comm,
                    "terminal_motif": terminal_motifs.get(node, "unknown"),
                }
                for node, comm in best_partition.items()
            ]
        )
        comm_df.to_csv(OUTPUT_DIR / "network_communities.csv", index=False)

        # Analyze hubs
        print("\n[6/6] Analyzing network structure...")
        hub_df = analyze_hub_sequences(G, terminal_motifs, sequences, n_hubs=20)
        hub_df.to_csv(OUTPUT_DIR / "hub_sequences.csv", index=False)
        print(f"  Saved hub analysis to {OUTPUT_DIR / 'hub_sequences.csv'}")

        # Visualize
        print("  Creating visualization...")
        visualize_network(G, best_partition, terminal_motifs, OUTPUT_DIR)
        print(
            f"  Saved network visualization to {OUTPUT_DIR / 'similarity_network.png'}"
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nNetwork at threshold {best_threshold:.2f}:")
    print(f"  {stats['n_nodes']} nodes, {stats['n_edges']} edges")
    print(f"  Density: {stats['density']:.4f}")

    if community_results:
        print(f"\nCommunity detection ({best_method}):")
        print(f"  {best_stats['n_communities']} communities detected")
        print(f"  Purity (terminal motif homogeneity): {best_stats['purity']:.3f}")
        print(
            f"  Completeness (terminal groups captured): {best_stats['avg_completeness']:.3f}"
        )

        # Community-terminal alignment
        print("\n  Community sizes:", dict(Counter(best_partition.values())))

    print(f"\n📁 Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
