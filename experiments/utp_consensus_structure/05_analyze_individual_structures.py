#!/usr/bin/env python3
"""
05_analyze_individual_structures.py

Analyze secondary structure in individual AlphaFold structures to determine
if anchor motifs form helical elements.

The consensus structure averaging destroys helical geometry, so we need to
analyze individual structures and aggregate the results.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from Bio import PDB
from collections import Counter
import warnings

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "utp-structures" / "c_term" / "aligned"
OUTPUT_DIR = Path(__file__).parent / "output"

# Motif definitions
MOTIF_2_WIDTH = 13  # First anchor
MOTIF_1_WIDTH = 21  # Second anchor
TYPICAL_OFFSET = 17  # motif_2 starts ~17 aa before motif_1

# Figure styling
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
rcParams["font.size"] = 10
rcParams["axes.linewidth"] = 1.2
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False

COLORS = {
    "primary": "#2E4057",
    "secondary": "#048A81",
    "accent": "#E85D04",
    "light": "#90BE6D",
    "highlight": "#F9C74F",
}

FIGURE_DPI = 300


# =============================================================================
# Secondary Structure Analysis
# =============================================================================


def calculate_phi_psi_for_structure(structure):
    """Calculate phi/psi angles for a structure."""
    angles = []

    for model in structure:
        for chain in model:
            residues = list(chain.get_residues())

            for i, residue in enumerate(residues):
                phi = None
                psi = None

                try:
                    # Get phi angle (C(i-1) - N - CA - C)
                    if i > 0:
                        prev_c = residues[i - 1]["C"].get_vector()
                        n = residue["N"].get_vector()
                        ca = residue["CA"].get_vector()
                        c = residue["C"].get_vector()
                        phi = PDB.calc_dihedral(prev_c, n, ca, c) * 180 / np.pi
                except:
                    pass

                try:
                    # Get psi angle (N - CA - C - N(i+1))
                    if i < len(residues) - 1:
                        n = residue["N"].get_vector()
                        ca = residue["CA"].get_vector()
                        c = residue["C"].get_vector()
                        next_n = residues[i + 1]["N"].get_vector()
                        psi = PDB.calc_dihedral(n, ca, c, next_n) * 180 / np.pi
                except:
                    pass

                angles.append(
                    {
                        "residue": i + 1,
                        "phi": phi,
                        "psi": psi,
                        "resname": residue.get_resname(),
                    }
                )

            break  # Only first chain
        break  # Only first model

    return pd.DataFrame(angles)


def assign_ss_from_angles(phi, psi):
    """Assign secondary structure based on phi/psi angles."""
    if phi is None or psi is None:
        return "C"

    # Alpha helix region (more generous bounds)
    if -90 <= phi <= -30 and -70 <= psi <= -10:
        return "H"
    # 3-10 helix
    if -90 <= phi <= -30 and -10 <= psi <= 30:
        return "G"
    # Beta sheet
    if -180 <= phi <= -90 and 90 <= psi <= 180:
        return "E"
    if -180 <= phi <= -90 and -180 <= psi <= -90:
        return "E"

    return "C"


def smooth_ss(ss_list, min_length=3):
    """Smooth secondary structure by requiring minimum helix/strand length."""
    result = list(ss_list)
    n = len(result)

    # Convert short helices to coil
    i = 0
    while i < n:
        if result[i] in ["H", "G"]:
            j = i
            while j < n and result[j] in ["H", "G"]:
                j += 1
            if j - i < min_length:
                for k in range(i, j):
                    result[k] = "C"
            i = j
        else:
            i += 1

    return result


def analyze_structure(cif_path):
    """Analyze a single structure and return SS assignment."""
    parser = PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure("struct", cif_path)

    angles_df = calculate_phi_psi_for_structure(structure)

    # Assign secondary structure
    angles_df["ss_raw"] = angles_df.apply(
        lambda row: assign_ss_from_angles(row["phi"], row["psi"]), axis=1
    )

    # Smooth
    angles_df["ss"] = smooth_ss(angles_df["ss_raw"].tolist(), min_length=3)

    return angles_df


def find_helices(ss_list):
    """Find helical regions in SS assignment."""
    helices = []
    i = 0
    while i < len(ss_list):
        if ss_list[i] in ["H", "G"]:
            start = i
            while i < len(ss_list) and ss_list[i] in ["H", "G"]:
                i += 1
            helices.append((start + 1, i))  # 1-indexed
        else:
            i += 1
    return helices


# =============================================================================
# Main Analysis
# =============================================================================


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("INDIVIDUAL STRUCTURE SECONDARY STRUCTURE ANALYSIS")
    print("=" * 60)

    # Load all aligned structures
    cif_files = sorted(DATA_DIR.glob("*.cif"))
    cif_files = [
        f for f in cif_files if f.stem != "fold1" and "consensus" not in f.stem.lower()
    ]

    print(f"\nFound {len(cif_files)} aligned structures")

    # Analyze each structure
    all_results = []
    helix_counts = Counter()

    for cif_path in cif_files:
        try:
            angles_df = analyze_structure(cif_path)
            ss_string = "".join(angles_df["ss"].tolist())
            helices = find_helices(angles_df["ss"].tolist())

            # Count helices in anchor region (first ~35 residues)
            anchor_helices = [h for h in helices if h[0] <= 35]

            all_results.append(
                {
                    "structure": cif_path.stem,
                    "length": len(angles_df),
                    "ss_string": ss_string,
                    "n_helices": len(helices),
                    "n_anchor_helices": len(anchor_helices),
                    "helices": helices,
                    "anchor_helices": anchor_helices,
                }
            )

            helix_counts[len(anchor_helices)] += 1

        except Exception as e:
            print(f"Warning: Could not analyze {cif_path.name}: {e}")

    print(f"\nAnalyzed {len(all_results)} structures successfully")

    # Summary statistics
    print("\n" + "=" * 60)
    print("HELIX COUNT IN ANCHOR REGION (first ~35 residues)")
    print("=" * 60)

    for n_helices, count in sorted(helix_counts.items()):
        pct = 100 * count / len(all_results)
        print(f"  {n_helices} helices: {count} structures ({pct:.1f}%)")

    # Analyze helix positions
    print("\n" + "=" * 60)
    print("HELIX POSITION ANALYSIS")
    print("=" * 60)

    # Collect all helix positions
    helix_starts = []
    helix_ends = []
    helix_lengths = []

    for result in all_results:
        for h_start, h_end in result["anchor_helices"]:
            helix_starts.append(h_start)
            helix_ends.append(h_end)
            helix_lengths.append(h_end - h_start)

    if helix_starts:
        print(f"\nAnchor region helices (n={len(helix_starts)}):")
        print(
            f"  Start positions: mean={np.mean(helix_starts):.1f}, median={np.median(helix_starts):.1f}"
        )
        print(
            f"  End positions: mean={np.mean(helix_ends):.1f}, median={np.median(helix_ends):.1f}"
        )
        print(
            f"  Lengths: mean={np.mean(helix_lengths):.1f}, median={np.median(helix_lengths):.1f}"
        )

    # Look for two-helix pattern
    print("\n" + "=" * 60)
    print("TWO-HELIX PATTERN ANALYSIS")
    print("=" * 60)

    two_helix_structures = [r for r in all_results if r["n_anchor_helices"] >= 2]
    print(f"\nStructures with 2+ helices in anchor region: {len(two_helix_structures)}")

    if two_helix_structures:
        # Analyze first two helices
        h1_starts = []
        h1_ends = []
        h2_starts = []
        h2_ends = []

        for result in two_helix_structures:
            helices = result["anchor_helices"]
            h1_starts.append(helices[0][0])
            h1_ends.append(helices[0][1])
            h2_starts.append(helices[1][0])
            h2_ends.append(helices[1][1])

        print(f"\nHelix 1 (first anchor helix):")
        print(f"  Start: {np.mean(h1_starts):.1f} ± {np.std(h1_starts):.1f}")
        print(f"  End: {np.mean(h1_ends):.1f} ± {np.std(h1_ends):.1f}")
        print(f"  Length: {np.mean([e-s for s,e in zip(h1_starts, h1_ends)]):.1f} aa")

        print(f"\nHelix 2 (second anchor helix):")
        print(f"  Start: {np.mean(h2_starts):.1f} ± {np.std(h2_starts):.1f}")
        print(f"  End: {np.mean(h2_ends):.1f} ± {np.std(h2_ends):.1f}")
        print(f"  Length: {np.mean([e-s for s,e in zip(h2_starts, h2_ends)]):.1f} aa")

        # Compare with motif positions
        print("\n" + "-" * 40)
        print("COMPARISON WITH MOTIF POSITIONS")
        print("-" * 40)

        # Expected motif positions (in structure coordinates)
        # Motif 2: residues 1-13
        # Motif 1: residues ~13-34 (starts ~17 aa after motif_2 start, with overlap)

        h1_in_motif2 = np.mean(h1_starts) < 15 and np.mean(h1_ends) < 25
        h2_in_motif1 = np.mean(h2_starts) > 10 and np.mean(h2_ends) < 40

        print(f"\nMotif 2 region (expected): residues 1-13")
        print(f"  Helix 1 position: {np.mean(h1_starts):.0f}-{np.mean(h1_ends):.0f}")
        print(f"  Overlap: {'YES' if h1_in_motif2 else 'PARTIAL/NO'}")

        print(f"\nMotif 1 region (expected): residues ~13-34")
        print(f"  Helix 2 position: {np.mean(h2_starts):.0f}-{np.mean(h2_ends):.0f}")
        print(f"  Overlap: {'YES' if h2_in_motif1 else 'PARTIAL/NO'}")

    # Create visualization
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    # Plot helix position distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Number of helices distribution
    ax = axes[0, 0]
    n_helices_list = [r["n_anchor_helices"] for r in all_results]
    ax.hist(
        n_helices_list,
        bins=range(0, max(n_helices_list) + 2),
        color=COLORS["secondary"],
        edgecolor="white",
        alpha=0.8,
    )
    ax.set_xlabel("Number of Helices in Anchor Region")
    ax.set_ylabel("Number of Structures")
    ax.set_title("A. Helix Count Distribution", fontweight="bold", loc="left")

    # Panel B: Helix start positions
    ax = axes[0, 1]
    if helix_starts:
        ax.hist(
            helix_starts, bins=20, color=COLORS["accent"], edgecolor="white", alpha=0.8
        )
        ax.axvline(13, color="gray", linestyle="--", label="Motif 2 end")
        ax.axvline(34, color="gray", linestyle=":", label="Motif 1 end")
    ax.set_xlabel("Helix Start Position (residue)")
    ax.set_ylabel("Count")
    ax.set_title("B. Helix Start Positions", fontweight="bold", loc="left")
    ax.legend(frameon=False)

    # Panel C: Helix lengths
    ax = axes[1, 0]
    if helix_lengths:
        ax.hist(
            helix_lengths, bins=15, color=COLORS["light"], edgecolor="white", alpha=0.8
        )
        ax.axvline(
            MOTIF_2_WIDTH,
            color=COLORS["accent"],
            linestyle="--",
            label=f"Motif 2 width ({MOTIF_2_WIDTH} aa)",
        )
        ax.axvline(
            MOTIF_1_WIDTH,
            color=COLORS["highlight"],
            linestyle="--",
            label=f"Motif 1 width ({MOTIF_1_WIDTH} aa)",
        )
    ax.set_xlabel("Helix Length (residues)")
    ax.set_ylabel("Count")
    ax.set_title("C. Helix Length Distribution", fontweight="bold", loc="left")
    ax.legend(frameon=False)

    # Panel D: SS string visualization for first few structures
    ax = axes[1, 1]
    n_show = min(20, len(all_results))
    ss_matrix = np.zeros((n_show, 50))

    for i, result in enumerate(all_results[:n_show]):
        ss = result["ss_string"][:50]
        for j, s in enumerate(ss):
            if s == "H":
                ss_matrix[i, j] = 2
            elif s == "G":
                ss_matrix[i, j] = 1.5
            elif s == "E":
                ss_matrix[i, j] = 1

    im = ax.imshow(ss_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=2)
    ax.set_xlabel("Residue Position")
    ax.set_ylabel("Structure")
    ax.set_title("D. SS Assignment (first 50 residues)", fontweight="bold", loc="left")
    ax.axvline(13, color="white", linestyle="--", linewidth=2)
    ax.axvline(34, color="white", linestyle=":", linewidth=2)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "individual_ss_analysis.svg", dpi=FIGURE_DPI, bbox_inches="tight"
    )
    plt.savefig(
        OUTPUT_DIR / "individual_ss_analysis.png", dpi=FIGURE_DPI, bbox_inches="tight"
    )
    plt.close()

    print(f"Saved: {OUTPUT_DIR / 'individual_ss_analysis.png'}")

    # Save results
    results_df = pd.DataFrame(
        [
            {
                "structure": r["structure"],
                "length": r["length"],
                "n_helices": r["n_helices"],
                "n_anchor_helices": r["n_anchor_helices"],
                "ss_string": r["ss_string"][:50],
                "helices": str(r["helices"]),
                "anchor_helices": str(r["anchor_helices"]),
            }
            for r in all_results
        ]
    )

    results_df.to_csv(OUTPUT_DIR / "individual_ss_results.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'individual_ss_results.csv'}")

    # Final conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    pct_two_plus = 100 * len(two_helix_structures) / len(all_results)

    if pct_two_plus > 50:
        print(
            f"\n✓ {pct_two_plus:.0f}% of structures have 2+ helices in the anchor region"
        )
        print(
            "  This supports the hypothesis that anchor motifs form the U-bend structure"
        )

        if two_helix_structures:
            avg_h1_len = np.mean(
                [
                    r["anchor_helices"][0][1] - r["anchor_helices"][0][0]
                    for r in two_helix_structures
                ]
            )
            avg_h2_len = np.mean(
                [
                    r["anchor_helices"][1][1] - r["anchor_helices"][1][0]
                    for r in two_helix_structures
                ]
            )

            print(
                f"\n  Helix 1 avg length: {avg_h1_len:.0f} aa (Motif 2 is {MOTIF_2_WIDTH} aa)"
            )
            print(
                f"  Helix 2 avg length: {avg_h2_len:.0f} aa (Motif 1 is {MOTIF_1_WIDTH} aa)"
            )
    else:
        print(
            f"\n⚠ Only {pct_two_plus:.0f}% of structures have 2+ helices in anchor region"
        )
        print("  The relationship between motifs and helices may be more complex")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
