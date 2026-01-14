#!/usr/bin/env python3
"""
03_statistical_test.py - Test if uTP Proteins Cluster More Than Expected

Performs permutation test to determine if uTP proteins share gene families
more than expected by chance.

Null hypothesis: uTP proteins are randomly distributed across gene families
Alternative: uTP proteins cluster into fewer families (founder effect)

Usage:
    uv run python experiments/utp_family_clustering/03_statistical_test.py
"""

import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

warnings.filterwarnings("ignore")

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"

# Input files
FAMILY_ASSIGNMENTS = OUTPUT_DIR / "family_assignments.csv"

# Output files
PERMUTATION_RESULTS = OUTPUT_DIR / "permutation_results.csv"
FIGURE_PERMUTATION = OUTPUT_DIR / "permutation_test.png"
FIGURE_PERMUTATION_SVG = OUTPUT_DIR / "permutation_test.svg"
SUMMARY_FILE = OUTPUT_DIR / "statistical_summary.txt"

# =============================================================================
# Plotting Style (from project rules)
# =============================================================================

COLORS = {
    "primary": "#2E4057",
    "secondary": "#048A81",
    "accent": "#E85D04",
    "light": "#90BE6D",
    "highlight": "#F9C74F",
    "background": "#F8F9FA",
    "text": "#212529",
}

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
rcParams["font.size"] = 9
rcParams["axes.linewidth"] = 1.0
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["legend.frameon"] = False

# =============================================================================
# Parameters
# =============================================================================

N_PERMUTATIONS = 10000
RANDOM_SEED = 42

# =============================================================================
# Functions
# =============================================================================


def compute_clustering_metrics(family_ids: np.ndarray, is_utp: np.ndarray) -> dict:
    """
    Compute metrics describing how uTP proteins cluster.

    Returns multiple metrics to capture different aspects of clustering:
    1. fraction_sharing: Fraction of uTP proteins that share a family with another uTP
    2. n_families_with_utp: Number of distinct families containing uTP proteins
    3. effective_n_families: Reciprocal of Simpson index (how spread out uTP are)
    4. max_utp_per_family: Maximum uTP proteins in any single family
    """
    # Count uTP proteins per family
    utp_per_family = Counter()
    for fam_id, is_u in zip(family_ids, is_utp):
        if is_u:
            utp_per_family[fam_id] += 1

    n_utp = is_utp.sum()

    # Metric 1: Fraction sharing
    utp_sharing = sum(count for count in utp_per_family.values() if count >= 2)
    fraction_sharing = utp_sharing / n_utp if n_utp > 0 else 0

    # Metric 2: Number of families with uTP
    n_families_with_utp = len(utp_per_family)

    # Metric 3: Effective number of families (reciprocal Simpson index)
    # This measures how spread out the uTP proteins are
    if n_utp > 0:
        proportions = np.array(list(utp_per_family.values())) / n_utp
        simpson = np.sum(proportions**2)
        effective_n = 1 / simpson if simpson > 0 else n_families_with_utp
    else:
        effective_n = 0

    # Metric 4: Maximum uTP per family
    max_utp = max(utp_per_family.values()) if utp_per_family else 0

    return {
        "fraction_sharing": fraction_sharing,
        "n_families_with_utp": n_families_with_utp,
        "effective_n_families": effective_n,
        "max_utp_per_family": max_utp,
    }


def permutation_test(
    family_ids: np.ndarray,
    is_utp: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Perform permutation test for uTP clustering.

    Shuffles uTP labels among proteins while keeping family structure fixed,
    then computes what fraction of uTP proteins would share families by chance.

    Returns:
        Dictionary with observed values, null distributions, and p-values
    """
    np.random.seed(seed)

    # Observed metrics
    observed = compute_clustering_metrics(family_ids, is_utp)

    # Permutation null distributions
    null_distributions = {
        "fraction_sharing": [],
        "n_families_with_utp": [],
        "effective_n_families": [],
        "max_utp_per_family": [],
    }

    n_utp = is_utp.sum()
    n_total = len(is_utp)

    print(f"  Running {n_permutations} permutations...")
    for i in range(n_permutations):
        if (i + 1) % 1000 == 0:
            print(f"    Permutation {i + 1}/{n_permutations}")

        # Shuffle uTP labels
        perm_is_utp = np.zeros(n_total, dtype=bool)
        perm_idx = np.random.choice(n_total, size=n_utp, replace=False)
        perm_is_utp[perm_idx] = True

        # Compute metrics for permuted labels
        perm_metrics = compute_clustering_metrics(family_ids, perm_is_utp)

        for key in null_distributions:
            null_distributions[key].append(perm_metrics[key])

    # Convert to arrays
    for key in null_distributions:
        null_distributions[key] = np.array(null_distributions[key])

    # Compute p-values
    # For fraction_sharing: higher = more clustered, so p = P(null >= observed)
    # For n_families_with_utp: lower = more clustered, so p = P(null <= observed)
    # For effective_n_families: lower = more clustered, so p = P(null <= observed)
    # For max_utp_per_family: higher = more clustered, so p = P(null >= observed)

    p_values = {
        "fraction_sharing": np.mean(
            null_distributions["fraction_sharing"] >= observed["fraction_sharing"]
        ),
        "n_families_with_utp": np.mean(
            null_distributions["n_families_with_utp"] <= observed["n_families_with_utp"]
        ),
        "effective_n_families": np.mean(
            null_distributions["effective_n_families"]
            <= observed["effective_n_families"]
        ),
        "max_utp_per_family": np.mean(
            null_distributions["max_utp_per_family"] >= observed["max_utp_per_family"]
        ),
    }

    return {
        "observed": observed,
        "null_distributions": null_distributions,
        "p_values": p_values,
        "n_permutations": n_permutations,
    }


def plot_permutation_results(results: dict, output_path: Path):
    """Create publication-quality figure showing permutation test results."""

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    metrics = [
        (
            "fraction_sharing",
            "Fraction of uTP Sharing Family",
            "higher = more clustered",
        ),
        (
            "n_families_with_utp",
            "Number of Families with uTP",
            "lower = more clustered",
        ),
        (
            "effective_n_families",
            "Effective Number of Families",
            "lower = more clustered",
        ),
        ("max_utp_per_family", "Max uTP per Family", "higher = more clustered"),
    ]

    for ax, (metric, title, direction) in zip(axes.flatten(), metrics):
        null_dist = results["null_distributions"][metric]
        observed = results["observed"][metric]
        p_value = results["p_values"][metric]

        # Histogram of null distribution
        ax.hist(
            null_dist,
            bins=50,
            color=COLORS["secondary"],
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
            label="Null distribution",
        )

        # Observed value
        ax.axvline(
            observed,
            color=COLORS["accent"],
            linewidth=2,
            label=f"Observed = {observed:.3f}",
        )

        # Fill significance region
        if "lower" in direction:
            ax.axvspan(ax.get_xlim()[0], observed, alpha=0.15, color=COLORS["accent"])
        else:
            ax.axvspan(observed, ax.get_xlim()[1], alpha=0.15, color=COLORS["accent"])

        # Formatting
        ax.set_xlabel(title)
        ax.set_ylabel("Frequency")
        ax.set_title(f"{title}\n(p = {p_value:.4f}, {direction})")
        ax.legend(loc="upper right", fontsize=8)

        # Statistics box
        stats_text = f"Observed: {observed:.3f}\nNull mean: {null_dist.mean():.3f}\nNull std: {null_dist.std():.3f}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close()


def interpret_results(results: dict) -> str:
    """Generate interpretation of the statistical results."""

    obs = results["observed"]
    p_vals = results["p_values"]
    null = results["null_distributions"]

    lines = []
    lines.append("=" * 70)
    lines.append("STATISTICAL INTERPRETATION")
    lines.append("=" * 70)

    # Overall conclusion
    lines.append("\n## Key Question")
    lines.append("Are uTP proteins more related to each other than expected by chance?")
    lines.append("(i.e., do they cluster into fewer gene families?)")

    lines.append("\n## Observed vs Expected")

    # Fraction sharing
    exp_sharing = null["fraction_sharing"].mean()
    lines.append(f"\n1. Fraction of uTP proteins sharing a family with another uTP:")
    lines.append(f"   Observed: {obs['fraction_sharing']:.1%}")
    lines.append(f"   Expected (random): {exp_sharing:.1%}")
    lines.append(f"   p-value: {p_vals['fraction_sharing']:.4f}")

    if obs["fraction_sharing"] > exp_sharing and p_vals["fraction_sharing"] < 0.05:
        lines.append(
            f"   → SIGNIFICANT: uTP proteins share families MORE than expected"
        )
    elif obs["fraction_sharing"] < exp_sharing and p_vals["fraction_sharing"] < 0.05:
        lines.append(
            f"   → SIGNIFICANT: uTP proteins share families LESS than expected"
        )
    else:
        lines.append(f"   → NOT significant: sharing similar to random expectation")

    # Number of families
    exp_n_fam = null["n_families_with_utp"].mean()
    lines.append(f"\n2. Number of distinct families containing uTP proteins:")
    lines.append(f"   Observed: {obs['n_families_with_utp']:.0f}")
    lines.append(f"   Expected (random): {exp_n_fam:.0f}")
    lines.append(f"   p-value: {p_vals['n_families_with_utp']:.4f}")

    if obs["n_families_with_utp"] < exp_n_fam and p_vals["n_families_with_utp"] < 0.05:
        lines.append(
            f"   → SIGNIFICANT: uTP in FEWER families than expected (clustered)"
        )
    elif (
        obs["n_families_with_utp"] > exp_n_fam and p_vals["n_families_with_utp"] > 0.95
    ):
        lines.append(
            f"   → SIGNIFICANT: uTP in MORE families than expected (scattered)"
        )
    else:
        lines.append(f"   → NOT significant: distribution similar to random")

    # Effective number of families
    exp_eff = null["effective_n_families"].mean()
    lines.append(f"\n3. Effective number of families (diversity measure):")
    lines.append(f"   Observed: {obs['effective_n_families']:.1f}")
    lines.append(f"   Expected (random): {exp_eff:.1f}")
    lines.append(f"   p-value: {p_vals['effective_n_families']:.4f}")

    # Maximum per family
    exp_max = null["max_utp_per_family"].mean()
    lines.append(f"\n4. Maximum uTP proteins in any single family:")
    lines.append(f"   Observed: {obs['max_utp_per_family']:.0f}")
    lines.append(f"   Expected (random): {exp_max:.1f}")
    lines.append(f"   p-value: {p_vals['max_utp_per_family']:.4f}")

    # Overall conclusion
    lines.append("\n" + "=" * 70)
    lines.append("CONCLUSION")
    lines.append("=" * 70)

    # Determine overall pattern
    n_sig_clustered = sum(
        [
            p_vals["fraction_sharing"] < 0.05
            and obs["fraction_sharing"] > null["fraction_sharing"].mean(),
            p_vals["n_families_with_utp"] < 0.05,
            p_vals["effective_n_families"] < 0.05,
            p_vals["max_utp_per_family"] < 0.05
            and obs["max_utp_per_family"] > null["max_utp_per_family"].mean(),
        ]
    )

    n_sig_scattered = sum(
        [
            p_vals["fraction_sharing"] < 0.05
            and obs["fraction_sharing"] < null["fraction_sharing"].mean(),
            p_vals["n_families_with_utp"] > 0.95,
            p_vals["effective_n_families"] > 0.95,
            p_vals["max_utp_per_family"] > 0.95,
        ]
    )

    if n_sig_clustered >= 2:
        lines.append("\n★ Model A SUPPORTED: uTP proteins are significantly CLUSTERED")
        lines.append("  → Consistent with FOUNDER EFFECT hypothesis")
        lines.append("  → uTP may have originated in a few ancestral proteins")
        lines.append("  → These expanded through gene duplication")
    elif n_sig_scattered >= 2:
        lines.append("\n★ Model B SUPPORTED: uTP proteins are significantly SCATTERED")
        lines.append("  → Consistent with SELECTION hypothesis")
        lines.append("  → uTP was independently acquired by many diverse proteins")
        lines.append("  → Selected for based on functional requirements")
    else:
        lines.append("\n★ NO CLEAR PATTERN: Results inconclusive")
        lines.append("  → uTP distribution similar to random expectation")
        lines.append("  → Neither founder effect nor strong selection detected")
        lines.append("  → May reflect intermediate evolutionary scenario")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("Statistical Test: uTP Protein Family Clustering")
    print("=" * 70)

    # =========================================================================
    # Step 1: Load data
    # =========================================================================
    print("\n[1/3] Loading family assignments...")

    assignments = pd.read_csv(FAMILY_ASSIGNMENTS)

    family_ids = assignments["family_id"].values
    is_utp = assignments["is_utp"].values

    n_utp = is_utp.sum()
    n_total = len(is_utp)
    n_families = len(np.unique(family_ids))

    print(f"  Total proteins: {n_total}")
    print(f"  Total families: {n_families}")
    print(f"  uTP proteins: {n_utp} ({100*n_utp/n_total:.1f}%)")

    # =========================================================================
    # Step 2: Run permutation test
    # =========================================================================
    print("\n[2/3] Running permutation test...")

    results = permutation_test(
        family_ids, is_utp, n_permutations=N_PERMUTATIONS, seed=RANDOM_SEED
    )

    # =========================================================================
    # Step 3: Generate outputs
    # =========================================================================
    print("\n[3/3] Generating outputs...")

    # Save numerical results
    results_df = pd.DataFrame(
        {
            "metric": list(results["observed"].keys()),
            "observed": list(results["observed"].values()),
            "null_mean": [
                results["null_distributions"][k].mean()
                for k in results["observed"].keys()
            ],
            "null_std": [
                results["null_distributions"][k].std()
                for k in results["observed"].keys()
            ],
            "null_2.5%": [
                np.percentile(results["null_distributions"][k], 2.5)
                for k in results["observed"].keys()
            ],
            "null_97.5%": [
                np.percentile(results["null_distributions"][k], 97.5)
                for k in results["observed"].keys()
            ],
            "p_value": list(results["p_values"].values()),
        }
    )
    results_df.to_csv(PERMUTATION_RESULTS, index=False)
    print(f"  Saved results to {PERMUTATION_RESULTS}")

    # Generate figure
    plot_permutation_results(results, FIGURE_PERMUTATION)
    print(f"  Saved figure to {FIGURE_PERMUTATION}")

    # Generate interpretation
    interpretation = interpret_results(results)
    print(interpretation)

    with open(SUMMARY_FILE, "w") as f:
        f.write(interpretation)
    print(f"\n  Saved summary to {SUMMARY_FILE}")

    print(f"\n📁 Output files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
