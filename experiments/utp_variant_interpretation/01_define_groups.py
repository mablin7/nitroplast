#!/usr/bin/env python3
"""
Define 5-6 motif-based uTP variant groups covering maximum sequences.

Strategy:
1. Use MAST results to get motif composition for each sequence
2. Define groups based on terminal motif + key distinguishing motifs
3. Maximize coverage while maintaining biological interpretability
"""

import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MAST_XML = PROJECT_ROOT / "experiments/utp_motif_coverage/output/mast_results/mast.xml"
IMPORT_FASTA = PROJECT_ROOT / "data/Import_candidates.fasta"
OUTPUT_DIR = Path(__file__).parent / "output"


def parse_mast_results(mast_xml_path: Path) -> list[dict]:
    """Parse MAST XML to extract motif hits for each sequence."""
    tree = ET.parse(mast_xml_path)
    root = tree.getroot()

    results = []
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

        # Sort hits by position
        hits.sort(key=lambda x: x["position"])

        if hits:
            results.append(
                {
                    "sequence_id": seq_id,
                    "length": seq_length,
                    "hits": hits,
                    "motif_set": set(h["motif_idx"] for h in hits),
                    "terminal_motif": hits[-1]["motif_idx"],
                    "motif_pattern": "+".join(h["motif_idx"] for h in hits),
                }
            )

    return results


def define_variant_groups(mast_results: list[dict]) -> dict[str, str]:
    """
    Define 5-6 variant groups based on motif composition.
    
    Strategy: Use terminal motif as primary discriminator, then subdivide large groups.
    
    Groups designed to:
    1. Cover maximum sequences
    2. Be biologically interpretable (based on motif presence)
    3. Have reasonable sizes (not too imbalanced)
    """
    
    # First, analyze what terminal motifs exist and their frequencies
    terminal_counts = Counter(r["terminal_motif"] for r in mast_results)
    print("Terminal motif distribution:")
    for motif, count in terminal_counts.most_common():
        print(f"  terminal_{motif}: {count} ({count/len(mast_results)*100:.1f}%)")
    
    # Analyze motif co-occurrence patterns
    print("\nMotif presence rates:")
    motif_presence = defaultdict(int)
    for r in mast_results:
        for m in r["motif_set"]:
            motif_presence[m] += 1
    for motif, count in sorted(motif_presence.items(), key=lambda x: -x[1]):
        print(f"  motif_{motif}: {count} ({count/len(mast_results)*100:.1f}%)")
    
    # NEW STRATEGY: Use terminal motif directly for grouping
    # Then subdivide the dominant terminal_6 group by internal motif patterns
    
    assignments = {}
    group_counts = Counter()
    
    for r in mast_results:
        seq_id = r["sequence_id"]
        terminal = r["terminal_motif"]
        motifs = r["motif_set"]
        n_motifs = len(r["hits"])
        
        if terminal == "6":
            # Subdivide the dominant terminal_6 group
            # Based on presence of distinguishing internal motifs
            if "5" in motifs:
                group = "terminal_6_with5"  # Has motif 5
            elif "7" in motifs:
                group = "terminal_6_with7"  # Has motif 7
            elif "9" in motifs:
                group = "terminal_6_with9"  # Has motif 9
            elif n_motifs >= 6:
                group = "terminal_6_full"   # Many motifs
            else:
                group = "terminal_6_minimal" # Few motifs
        elif terminal == "4":
            group = "terminal_4"
        elif terminal == "3":
            group = "terminal_3"
        elif terminal == "5":
            group = "terminal_5"
        elif terminal == "0":
            group = "terminal_0"
        elif terminal == "7":
            group = "terminal_7"
        else:
            # Group rare terminal motifs (1, 2, 8, 9)
            group = "terminal_rare"
        
        assignments[seq_id] = group
        group_counts[group] += 1
    
    print("\nInitial group sizes:")
    for group, count in group_counts.most_common():
        print(f"  {group}: {count}")
    
    # Consolidate to get 5-6 balanced groups
    final_assignments = {}
    
    for seq_id, group in assignments.items():
        # Merge very small groups and balance sizes
        if group == "terminal_6_with5":
            final_group = "variant_A"  # Terminal 6 + motif 5
        elif group == "terminal_6_with7":
            final_group = "variant_B"  # Terminal 6 + motif 7
        elif group in ["terminal_6_with9", "terminal_6_full"]:
            final_group = "variant_C"  # Terminal 6 + other patterns
        elif group == "terminal_6_minimal":
            final_group = "variant_D"  # Terminal 6 minimal
        elif group in ["terminal_4", "terminal_3"]:
            final_group = "variant_E"  # Terminal 4 or 3
        elif group in ["terminal_5", "terminal_7", "terminal_0", "terminal_rare"]:
            final_group = "variant_F"  # Other terminals
        else:
            final_group = "variant_other"
        
        final_assignments[seq_id] = final_group
    
    return final_assignments


def analyze_group_characteristics(
    mast_results: list[dict], assignments: dict[str, str]
) -> pd.DataFrame:
    """Analyze motif composition characteristics of each group."""
    
    group_data = defaultdict(list)
    
    for r in mast_results:
        seq_id = r["sequence_id"]
        if seq_id not in assignments:
            continue
            
        group = assignments[seq_id]
        group_data[group].append(
            {
                "sequence_id": seq_id,
                "n_motifs": len(r["hits"]),
                "motif_set": r["motif_set"],
                "terminal": r["terminal_motif"],
                "pattern": r["motif_pattern"],
                "has_1": "1" in r["motif_set"],
                "has_2": "2" in r["motif_set"],
                "has_3": "3" in r["motif_set"],
                "has_4": "4" in r["motif_set"],
                "has_5": "5" in r["motif_set"],
                "has_6": "6" in r["motif_set"],
                "has_7": "7" in r["motif_set"],
                "has_8": "8" in r["motif_set"],
                "has_9": "9" in r["motif_set"],
            }
        )
    
    # Build summary
    summaries = []
    for group, items in group_data.items():
        n = len(items)
        summaries.append(
            {
                "group": group,
                "n_sequences": n,
                "avg_n_motifs": np.mean([x["n_motifs"] for x in items]),
                "pct_has_1": sum(x["has_1"] for x in items) / n * 100,
                "pct_has_2": sum(x["has_2"] for x in items) / n * 100,
                "pct_has_3": sum(x["has_3"] for x in items) / n * 100,
                "pct_has_4": sum(x["has_4"] for x in items) / n * 100,
                "pct_has_5": sum(x["has_5"] for x in items) / n * 100,
                "pct_has_6": sum(x["has_6"] for x in items) / n * 100,
                "pct_has_7": sum(x["has_7"] for x in items) / n * 100,
                "pct_has_8": sum(x["has_8"] for x in items) / n * 100,
                "pct_has_9": sum(x["has_9"] for x in items) / n * 100,
            }
        )
    
    return pd.DataFrame(summaries).sort_values("n_sequences", ascending=False)


def visualize_groups(
    mast_results: list[dict], assignments: dict[str, str], output_dir: Path
):
    """Create visualizations of the variant groups."""
    
    # Group sizes
    group_counts = Counter(assignments.values())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart of group sizes
    ax = axes[0]
    groups = sorted(group_counts.keys())
    counts = [group_counts[g] for g in groups]
    colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
    
    bars = ax.bar(groups, counts, color=colors)
    ax.set_xlabel("Variant Group")
    ax.set_ylabel("Number of Sequences")
    ax.set_title(f"uTP Variant Group Sizes (n={sum(counts)})")
    ax.tick_params(axis="x", rotation=45)
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(count),
            ha="center",
            va="bottom",
        )
    
    # Motif composition heatmap
    ax = axes[1]
    
    # Build motif presence matrix
    motif_data = defaultdict(lambda: defaultdict(int))
    group_sizes = defaultdict(int)
    
    for r in mast_results:
        seq_id = r["sequence_id"]
        if seq_id not in assignments:
            continue
        group = assignments[seq_id]
        group_sizes[group] += 1
        for m in r["motif_set"]:
            motif_data[group][f"motif_{m}"] += 1
    
    # Convert to percentages
    motif_pct = {}
    for group in groups:
        motif_pct[group] = {
            m: motif_data[group][m] / group_sizes[group] * 100
            for m in [f"motif_{i}" for i in range(1, 11)]
        }
    
    motif_df = pd.DataFrame(motif_pct).T
    motif_df = motif_df[[f"motif_{i}" for i in range(1, 11)]]
    
    sns.heatmap(
        motif_df,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "% with motif"},
    )
    ax.set_title("Motif Presence by Variant Group (%)")
    ax.set_xlabel("Motif")
    ax.set_ylabel("Variant Group")
    
    plt.tight_layout()
    plt.savefig(output_dir / "variant_groups.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Define uTP Variant Groups")
    print("=" * 70)

    # Parse MAST results
    print("\n[1/4] Loading MAST results...")
    if not MAST_XML.exists():
        print(f"  ERROR: MAST results not found at {MAST_XML}")
        return

    mast_results = parse_mast_results(MAST_XML)
    print(f"  Loaded {len(mast_results)} sequences with motif hits")

    # Define groups
    print("\n[2/4] Defining variant groups...")
    assignments = define_variant_groups(mast_results)

    # Final group counts
    final_counts = Counter(assignments.values())
    print("\nFinal variant groups:")
    total = sum(final_counts.values())
    for group, count in final_counts.most_common():
        print(f"  {group}: {count} ({count/total*100:.1f}%)")
    print(f"  TOTAL: {total} sequences covered")

    # Analyze characteristics
    print("\n[3/4] Analyzing group characteristics...")
    summary_df = analyze_group_characteristics(mast_results, assignments)
    summary_df.to_csv(OUTPUT_DIR / "group_summary.csv", index=False)
    print(summary_df.to_string(index=False))

    # Save assignments
    assignments_df = pd.DataFrame(
        [{"sequence_id": k, "variant_group": v} for k, v in assignments.items()]
    )
    assignments_df.to_csv(OUTPUT_DIR / "variant_assignments.csv", index=False)
    print(f"\n  Saved assignments to {OUTPUT_DIR / 'variant_assignments.csv'}")

    # Visualize
    print("\n[4/4] Creating visualizations...")
    visualize_groups(mast_results, assignments, OUTPUT_DIR)
    print(f"  Saved visualization to {OUTPUT_DIR / 'variant_groups.png'}")

    print(f"\n✅ Defined {len(final_counts)} variant groups covering {total} sequences")


if __name__ == "__main__":
    main()
