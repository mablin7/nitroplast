#!/usr/bin/env python3
"""
analyze_structure.py - Test whether uTP sequences have more structure than expected by chance

This experiment asks a fundamental question: Is there ANY detectable structure in uTP
sequences beyond what would be expected from random sequences with the same amino acid
composition and length distribution?

Approach:
1. Generate null sequences by shuffling AA composition (preserving length and AA frequencies)
2. Embed both real and null sequences using ProtT5
3. Compute clustering metrics (silhouette score, Hopkins statistic, etc.)
4. Compare real vs null distributions via permutation test

Interpretation:
- If real uTP has LOWER silhouette than null → sequences are anti-clustered (actively spread out)
- If real uTP has SIMILAR silhouette to null → no detectable structure beyond composition
- If real uTP has HIGHER silhouette than null → there IS structure, even if not discrete clusters

The experiment runs on TWO separate datasets:
1. Experimental set: 206 Gblocks-filtered sequences from proteomics enrichment
2. HMM-predicted set: 933 proteins with HMM hits (broader, potentially noisier)

Usage:
    uv run python experiments/utp_structure_vs_null/analyze_structure.py
"""

import gc
import warnings
from collections import Counter
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

# Null model parameters
N_NULL_SETS = 100  # Number of null sequence sets to generate
RANDOM_SEED = 42

# Embedding parameters
PROTT5_MODEL = "Rostlab/prot_t5_xl_uniref50"
EMBEDDING_DIM = 1024
MAX_SEQ_LENGTH = 500  # uTP sequences are ~100-200 AA

# Clustering parameters
K_RANGE = range(2, 11)  # Test k=2 to k=10 for silhouette

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Input files
EXPERIMENTAL_FASTA = (
    PROJECT_ROOT
    / "experiments"
    / "utp_motif_analysis"
    / "data"
    / "good-c-term-gb.fasta"
)
HMM_PREDICTED_FASTA = DATA_DIR / "Import_candidates.fasta"

# Output files
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"


# =============================================================================
# Null Sequence Generation
# =============================================================================


def compute_aa_frequencies(sequences: list[str]) -> dict[str, float]:
    """Compute amino acid frequencies from a set of sequences."""
    aa_pool = "".join(sequences)
    total = len(aa_pool)

    if total == 0:
        return {}

    return {aa: aa_pool.count(aa) / total for aa in set(aa_pool)}


def generate_null_sequences(
    real_sequences: list[str],
    n_null: int = 1,
    seed: int = None,
) -> list[str]:
    """
    Generate null sequences with same length distribution and AA composition as real sequences.

    This preserves:
    - The exact length of each sequence (1:1 mapping)
    - The overall amino acid composition of the pool

    This destroys:
    - Any position-specific conservation
    - Any motif structure
    - Any sequence-specific patterns
    """
    if seed is not None:
        np.random.seed(seed)

    # Compute AA frequencies from all real sequences
    aa_freq = compute_aa_frequencies(real_sequences)

    if not aa_freq:
        return []

    aa_list = list(aa_freq.keys())
    aa_probs = np.array([aa_freq[aa] for aa in aa_list])

    # Normalize probabilities (handle floating point errors)
    aa_probs = aa_probs / aa_probs.sum()

    null_sequences = []
    for seq in real_sequences:
        length = len(seq)
        null_seq = "".join(np.random.choice(aa_list, size=length, p=aa_probs))
        null_sequences.append(null_seq)

    return null_sequences


def generate_shuffled_sequences(
    real_sequences: list[str],
    seed: int = None,
) -> list[str]:
    """
    Generate null sequences by shuffling each sequence individually.

    This is a more conservative null model that preserves:
    - The exact length of each sequence
    - The exact AA composition of each individual sequence

    This destroys:
    - Position-specific conservation within each sequence
    - Motif structure
    """
    if seed is not None:
        np.random.seed(seed)

    shuffled = []
    for seq in real_sequences:
        aa_list = list(seq)
        np.random.shuffle(aa_list)
        shuffled.append("".join(aa_list))

    return shuffled


# =============================================================================
# Embedding Functions
# =============================================================================


def get_prott5_embeddings(
    sequences: list[str],
    names: list[str] = None,
    batch_size: int = 1,
    device: str = None,
) -> np.ndarray:
    """
    Compute ProtT5 embeddings for a list of sequences.

    Returns mean-pooled embeddings (n_sequences, 1024).
    """
    import torch
    from transformers import T5EncoderModel, T5Tokenizer

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    print(f"  Loading ProtT5 model on {device}...")
    tokenizer = T5Tokenizer.from_pretrained(PROTT5_MODEL, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(PROTT5_MODEL)
    model = model.to(device)
    model.eval()

    embeddings = []

    # Process sequences
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="  Embedding"):
            batch_seqs = sequences[i : i + batch_size]

            # Add spaces between amino acids (ProtT5 requirement)
            batch_seqs_spaced = [
                " ".join(list(seq[:MAX_SEQ_LENGTH])) for seq in batch_seqs
            ]

            # Tokenize
            encoded = tokenizer(
                batch_seqs_spaced,
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH + 2,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # Get embeddings
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Mean pool over sequence length (excluding padding)
            for j, emb in enumerate(outputs.last_hidden_state):
                mask = attention_mask[j].bool()
                seq_emb = emb[mask].mean(dim=0).cpu().numpy()
                embeddings.append(seq_emb)

    # Clean up
    del model, tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return np.array(embeddings)


def load_or_compute_embeddings(
    sequences: list[str],
    names: list[str],
    cache_file: Path,
    force_recompute: bool = False,
) -> np.ndarray:
    """Load embeddings from cache or compute them."""
    if cache_file.exists() and not force_recompute:
        print(f"  Loading cached embeddings from {cache_file}")
        with h5py.File(cache_file, "r") as f:
            # Check if all sequences are present
            cached_names = set(f.keys())
            if set(names).issubset(cached_names):
                return np.array([f[n][()] for n in names])
            print("  Cache incomplete, recomputing...")

    # Compute embeddings
    embeddings = get_prott5_embeddings(sequences, names)

    # Save to cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cache_file, "w") as f:
        for name, emb in zip(names, embeddings):
            f.create_dataset(name, data=emb)

    return embeddings


# =============================================================================
# Clustering Metrics
# =============================================================================


def compute_silhouette_best_k(
    embeddings: np.ndarray,
    k_range: range = K_RANGE,
) -> tuple[float, int]:
    """
    Compute silhouette score for best k in range.

    Returns (best_silhouette, best_k).
    """
    if len(embeddings) < max(k_range):
        k_range = range(2, min(len(embeddings), max(k_range)))

    if len(k_range) == 0:
        return 0.0, 2

    best_score = -1
    best_k = 2

    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)

            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    return best_score, best_k


def compute_hopkins_statistic(
    embeddings: np.ndarray,
    sample_size: int = None,
    seed: int = RANDOM_SEED,
) -> float:
    """
    Compute Hopkins statistic to test for clustering tendency.

    H ≈ 0.5: data is uniformly distributed (no clusters)
    H > 0.75: data has significant clustering tendency
    H < 0.5: data is regularly spaced (anti-clustered)
    """
    np.random.seed(seed)

    n = len(embeddings)
    if sample_size is None:
        sample_size = min(n // 10, 50)
    sample_size = max(sample_size, 5)

    if n < sample_size * 2:
        return 0.5  # Not enough data

    # Sample random points from the data
    sample_idx = np.random.choice(n, sample_size, replace=False)
    sample_points = embeddings[sample_idx]

    # Generate random points in the data space
    mins = embeddings.min(axis=0)
    maxs = embeddings.max(axis=0)
    random_points = np.random.uniform(mins, maxs, (sample_size, embeddings.shape[1]))

    # Fit nearest neighbors on all data
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(embeddings)

    # Distance from sample points to nearest neighbor (excluding self)
    u_distances, _ = nn.kneighbors(sample_points)
    u = u_distances[:, 1].sum()  # Second nearest (first is self)

    # Distance from random points to nearest neighbor
    w_distances, _ = nn.kneighbors(random_points)
    w = w_distances[:, 0].sum()

    # Hopkins statistic
    H = u / (u + w) if (u + w) > 0 else 0.5

    return H


def compute_mean_pairwise_distance(embeddings: np.ndarray) -> float:
    """Compute mean pairwise Euclidean distance."""
    if len(embeddings) < 2:
        return 0.0

    distances = pdist(embeddings, metric="euclidean")
    return np.mean(distances)


def compute_distance_variance(embeddings: np.ndarray) -> float:
    """Compute variance of pairwise distances (measure of spread uniformity)."""
    if len(embeddings) < 2:
        return 0.0

    distances = pdist(embeddings, metric="euclidean")
    return np.var(distances)


def compute_all_metrics(embeddings: np.ndarray) -> dict:
    """Compute all clustering metrics for a set of embeddings."""
    silhouette, best_k = compute_silhouette_best_k(embeddings)
    hopkins = compute_hopkins_statistic(embeddings)
    mean_dist = compute_mean_pairwise_distance(embeddings)
    dist_var = compute_distance_variance(embeddings)

    return {
        "silhouette_score": silhouette,
        "best_k": best_k,
        "hopkins_statistic": hopkins,
        "mean_pairwise_distance": mean_dist,
        "distance_variance": dist_var,
        "n_samples": len(embeddings),
    }


# =============================================================================
# Statistical Tests
# =============================================================================


def permutation_test_structure(
    real_metrics: dict,
    null_metrics_list: list[dict],
    metric_name: str = "silhouette_score",
) -> dict:
    """
    Perform permutation test comparing real metric to null distribution.

    Returns dict with:
    - observed: real metric value
    - null_mean: mean of null distribution
    - null_std: std of null distribution
    - p_value_greater: P(null >= observed) - is real MORE structured?
    - p_value_less: P(null <= observed) - is real LESS structured?
    - effect_size: (observed - null_mean) / null_std
    """
    observed = real_metrics[metric_name]
    null_values = np.array([m[metric_name] for m in null_metrics_list])

    null_mean = np.mean(null_values)
    null_std = np.std(null_values)

    # Two-sided p-values
    p_greater = np.mean(null_values >= observed)
    p_less = np.mean(null_values <= observed)

    # Effect size (Cohen's d-like)
    effect_size = (observed - null_mean) / null_std if null_std > 0 else 0

    return {
        "observed": observed,
        "null_mean": null_mean,
        "null_std": null_std,
        "null_min": np.min(null_values),
        "null_max": np.max(null_values),
        "null_values": null_values,
        "p_value_greater": p_greater,
        "p_value_less": p_less,
        "effect_size": effect_size,
    }


# =============================================================================
# Visualization
# =============================================================================


def plot_null_distribution(
    test_result: dict,
    metric_name: str,
    dataset_name: str,
    output_file: Path,
):
    """Plot observed value against null distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram of null values
    null_values = test_result["null_values"]
    ax.hist(
        null_values,
        bins=30,
        density=True,
        alpha=0.7,
        color="#3498db",
        edgecolor="white",
        label=f"Null distribution (n={len(null_values)})",
    )

    # Observed value
    observed = test_result["observed"]
    ax.axvline(
        observed,
        color="#e74c3c",
        linewidth=2.5,
        linestyle="-",
        label=f"Observed: {observed:.3f}",
    )

    # Null mean
    ax.axvline(
        test_result["null_mean"],
        color="#2ecc71",
        linewidth=2,
        linestyle="--",
        label=f"Null mean: {test_result['null_mean']:.3f}",
    )

    # Add text annotation
    p_val = min(test_result["p_value_greater"], test_result["p_value_less"])
    direction = "MORE" if test_result["p_value_greater"] < 0.5 else "LESS"
    significance = (
        "***"
        if p_val < 0.001
        else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    )

    text = (
        f"Effect size: {test_result['effect_size']:.2f}σ\n"
        f"p-value ({direction} structured): {p_val:.4f} {significance}\n"
        f"Real uTP is {direction} structured than random"
    )

    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    ax.set_xlabel(metric_name.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"{dataset_name}: Real vs Null {metric_name.replace('_', ' ').title()}",
        fontsize=14,
    )
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison_summary(
    results: dict,
    output_file: Path,
):
    """Plot summary comparison of experimental vs HMM-predicted sets."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    metrics = [
        "silhouette_score",
        "hopkins_statistic",
        "mean_pairwise_distance",
        "distance_variance",
    ]
    titles = [
        "Silhouette Score",
        "Hopkins Statistic",
        "Mean Pairwise Distance",
        "Distance Variance",
    ]

    for ax, metric, title in zip(axes.flat, metrics, titles):
        # Experimental
        exp_obs = results["experimental"]["tests"][metric]["observed"]
        exp_null = results["experimental"]["tests"][metric]["null_values"]

        # HMM-predicted
        hmm_obs = results["hmm_predicted"]["tests"][metric]["observed"]
        hmm_null = results["hmm_predicted"]["tests"][metric]["null_values"]

        # Box plots of null distributions
        bp = ax.boxplot(
            [exp_null, hmm_null],
            positions=[1, 2],
            widths=0.6,
            patch_artist=True,
        )

        colors = ["#3498db", "#e67e22"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        # Observed values as points
        ax.scatter(
            [1],
            [exp_obs],
            color="#e74c3c",
            s=150,
            zorder=5,
            marker="*",
            label="Observed",
        )
        ax.scatter([2], [hmm_obs], color="#e74c3c", s=150, zorder=5, marker="*")

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Experimental\n(n=206)", "HMM-predicted\n(n=933)"])
        ax.set_ylabel(title)
        ax.set_title(title)

        # Add p-values
        exp_p = min(
            results["experimental"]["tests"][metric]["p_value_greater"],
            results["experimental"]["tests"][metric]["p_value_less"],
        )
        hmm_p = min(
            results["hmm_predicted"]["tests"][metric]["p_value_greater"],
            results["hmm_predicted"]["tests"][metric]["p_value_less"],
        )

        ax.text(
            1, ax.get_ylim()[1], f"p={exp_p:.3f}", ha="center", va="bottom", fontsize=9
        )
        ax.text(
            2, ax.get_ylim()[1], f"p={hmm_p:.3f}", ha="center", va="bottom", fontsize=9
        )

    axes[0, 0].legend(loc="upper right")

    plt.suptitle("uTP Structure Analysis: Real vs Null Sequences", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main Analysis
# =============================================================================


def analyze_dataset(
    sequences: list[str],
    names: list[str],
    dataset_name: str,
    embeddings_cache: Path,
    n_null_sets: int = N_NULL_SETS,
) -> dict:
    """
    Run full structure analysis on a dataset.

    Returns dict with:
    - real_metrics: metrics for real sequences
    - null_metrics: list of metrics for each null set
    - tests: statistical test results for each metric
    """
    print(f"\n{'='*70}")
    print(f"Analyzing: {dataset_name}")
    print(f"{'='*70}")
    print(f"  Sequences: {len(sequences)}")
    print(f"  Mean length: {np.mean([len(s) for s in sequences]):.1f} AA")

    # =========================================================================
    # Step 1: Compute embeddings for real sequences
    # =========================================================================
    print("\n[1/4] Computing embeddings for real sequences...")
    real_embeddings = load_or_compute_embeddings(
        sequences, names, embeddings_cache, force_recompute=False
    )

    # Scale embeddings
    scaler = StandardScaler()
    real_embeddings_scaled = scaler.fit_transform(real_embeddings)

    # =========================================================================
    # Step 2: Compute metrics for real sequences
    # =========================================================================
    print("\n[2/4] Computing clustering metrics for real sequences...")
    real_metrics = compute_all_metrics(real_embeddings_scaled)
    print(
        f"  Silhouette score: {real_metrics['silhouette_score']:.3f} (best k={real_metrics['best_k']})"
    )
    print(f"  Hopkins statistic: {real_metrics['hopkins_statistic']:.3f}")
    print(f"  Mean pairwise distance: {real_metrics['mean_pairwise_distance']:.3f}")

    # =========================================================================
    # Step 3: Generate null sequences and compute metrics
    # =========================================================================
    print(
        f"\n[3/4] Generating {n_null_sets} null sequence sets and computing metrics..."
    )

    null_metrics_list = []

    for i in tqdm(range(n_null_sets), desc="  Null sets"):
        # Generate null sequences (shuffled within each sequence)
        null_seqs = generate_shuffled_sequences(sequences, seed=RANDOM_SEED + i)

        # Compute embeddings for null sequences
        null_embeddings = get_prott5_embeddings(null_seqs)
        null_embeddings_scaled = scaler.transform(null_embeddings)

        # Compute metrics
        null_metrics = compute_all_metrics(null_embeddings_scaled)
        null_metrics_list.append(null_metrics)

        # Clean up
        del null_embeddings, null_embeddings_scaled
        gc.collect()

    # =========================================================================
    # Step 4: Statistical tests
    # =========================================================================
    print("\n[4/4] Running statistical tests...")

    tests = {}
    for metric in [
        "silhouette_score",
        "hopkins_statistic",
        "mean_pairwise_distance",
        "distance_variance",
    ]:
        tests[metric] = permutation_test_structure(
            real_metrics, null_metrics_list, metric
        )

        p_val = min(tests[metric]["p_value_greater"], tests[metric]["p_value_less"])
        direction = ">" if tests[metric]["p_value_greater"] < 0.5 else "<"
        print(
            f"  {metric}: observed={tests[metric]['observed']:.3f}, "
            f"null={tests[metric]['null_mean']:.3f}±{tests[metric]['null_std']:.3f}, "
            f"p={p_val:.4f} (real {direction} null)"
        )

    return {
        "dataset_name": dataset_name,
        "n_sequences": len(sequences),
        "real_metrics": real_metrics,
        "null_metrics": null_metrics_list,
        "tests": tests,
    }


def main():
    print("=" * 70)
    print("uTP Structure vs Null Analysis")
    print("Testing whether uTP sequences have more structure than random")
    print("=" * 70)

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # =========================================================================
    # Dataset 1: Experimental (Gblocks-filtered)
    # =========================================================================
    print("\n" + "=" * 70)
    print("DATASET 1: Experimental (Gblocks-filtered proteomics)")
    print("=" * 70)

    exp_records = list(SeqIO.parse(EXPERIMENTAL_FASTA, "fasta"))
    exp_sequences = [str(r.seq) for r in exp_records]
    exp_names = [r.id for r in exp_records]

    results["experimental"] = analyze_dataset(
        exp_sequences,
        exp_names,
        "Experimental (n=206)",
        EMBEDDINGS_DIR / "experimental_embeddings.h5",
        n_null_sets=N_NULL_SETS,
    )

    # =========================================================================
    # Dataset 2: HMM-predicted (full Import_candidates)
    # =========================================================================
    print("\n" + "=" * 70)
    print("DATASET 2: HMM-predicted (Import_candidates)")
    print("=" * 70)

    hmm_records = list(SeqIO.parse(HMM_PREDICTED_FASTA, "fasta"))
    hmm_sequences = [str(r.seq) for r in hmm_records]
    hmm_names = [r.id for r in hmm_records]

    results["hmm_predicted"] = analyze_dataset(
        hmm_sequences,
        hmm_names,
        "HMM-predicted (n=933)",
        EMBEDDINGS_DIR / "hmm_predicted_embeddings.h5",
        n_null_sets=N_NULL_SETS,
    )

    # =========================================================================
    # Generate visualizations
    # =========================================================================
    print("\n" + "=" * 70)
    print("Generating visualizations")
    print("=" * 70)

    # Individual metric plots
    for dataset_key, dataset_name in [
        ("experimental", "Experimental"),
        ("hmm_predicted", "HMM-predicted"),
    ]:
        for metric in ["silhouette_score", "hopkins_statistic"]:
            plot_null_distribution(
                results[dataset_key]["tests"][metric],
                metric,
                dataset_name,
                FIGURES_DIR / f"{dataset_key}_{metric}.svg",
            )
            print(f"  Saved {dataset_key}_{metric}.svg")

    # Summary comparison
    plot_comparison_summary(results, FIGURES_DIR / "comparison_summary.svg")
    print("  Saved comparison_summary.svg")

    # =========================================================================
    # Save results
    # =========================================================================
    print("\n" + "=" * 70)
    print("Saving results")
    print("=" * 70)

    # Summary table
    summary_data = []
    for dataset_key in ["experimental", "hmm_predicted"]:
        r = results[dataset_key]
        for metric in [
            "silhouette_score",
            "hopkins_statistic",
            "mean_pairwise_distance",
            "distance_variance",
        ]:
            t = r["tests"][metric]
            p_val = min(t["p_value_greater"], t["p_value_less"])
            direction = "MORE" if t["p_value_greater"] < 0.5 else "LESS"

            summary_data.append(
                {
                    "dataset": dataset_key,
                    "n_sequences": r["n_sequences"],
                    "metric": metric,
                    "observed": t["observed"],
                    "null_mean": t["null_mean"],
                    "null_std": t["null_std"],
                    "effect_size": t["effect_size"],
                    "p_value": p_val,
                    "direction": direction,
                    "significant_0.05": p_val < 0.05,
                    "significant_0.01": p_val < 0.01,
                }
            )

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(RESULTS_DIR / "structure_analysis_summary.csv", index=False)
    print(f"  Saved structure_analysis_summary.csv")

    # =========================================================================
    # Print summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for dataset_key, dataset_name in [
        ("experimental", "Experimental"),
        ("hmm_predicted", "HMM-predicted"),
    ]:
        r = results[dataset_key]
        print(f"\n{dataset_name} ({r['n_sequences']} sequences):")

        sil_test = r["tests"]["silhouette_score"]
        hop_test = r["tests"]["hopkins_statistic"]

        sil_p = min(sil_test["p_value_greater"], sil_test["p_value_less"])
        hop_p = min(hop_test["p_value_greater"], hop_test["p_value_less"])

        sil_dir = "MORE" if sil_test["p_value_greater"] < 0.5 else "LESS"
        hop_dir = "MORE" if hop_test["p_value_greater"] < 0.5 else "LESS"

        print(
            f"  Silhouette: {sil_test['observed']:.3f} vs null {sil_test['null_mean']:.3f}±{sil_test['null_std']:.3f}"
        )
        print(
            f"    → Real uTP is {sil_dir} clustered than random (p={sil_p:.4f}, effect={sil_test['effect_size']:.2f}σ)"
        )

        print(
            f"  Hopkins: {hop_test['observed']:.3f} vs null {hop_test['null_mean']:.3f}±{hop_test['null_std']:.3f}"
        )
        print(
            f"    → Real uTP has {hop_dir} clustering tendency than random (p={hop_p:.4f}, effect={hop_test['effect_size']:.2f}σ)"
        )

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print(
        """
Silhouette Score:
  - Measures how well-separated clusters are
  - Higher = more distinct clusters
  - If real > null: uTP has discrete structure
  - If real < null: uTP is more spread out / anti-clustered
  - If real ≈ null: no detectable structure beyond AA composition

Hopkins Statistic:
  - Measures clustering tendency (0.5 = uniform, >0.75 = clustered)
  - If real > null: uTP has clustering tendency
  - If real < null: uTP is regularly spaced
  - If real ≈ null: no clustering tendency beyond composition

Key Question Answered:
  Is there ANY structure in uTP beyond random sequences with same composition?
  
  If p < 0.05 for silhouette (real > null):
    → YES, there is detectable structure (even if not discrete clusters)
    → Supports the hypothesis that uTP variants encode information
  
  If p > 0.05:
    → NO detectable structure beyond amino acid composition
    → Variants may be artifacts of motif detection, not biological signal
"""
    )

    print(f"\n📁 Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
