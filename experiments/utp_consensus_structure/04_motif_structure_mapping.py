#!/usr/bin/env python3
"""
04_motif_structure_mapping.py

Map anchor motifs (2 → 1) to structural elements in the uTP consensus structure.

Goal: Determine if the two anchor motifs form the conserved U-bend structure.

Key questions:
1. Where are the α-helices in the consensus structure?
2. Do motif_2 (13 aa) and motif_1 (21 aa) correspond to these helices?

Method:
1. Use DSSP-like secondary structure assignment on consensus
2. Analyze backbone geometry to identify helical regions
3. Map motif sequence positions to structural positions
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from Bio import PDB
from Bio.PDB import DSSP
import warnings

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = Path(__file__).parent / "output"
CONSENSUS_PDB = OUTPUT_DIR / "consensus_structure.pdb"

# Motif definitions from MEME
MOTIF_DEFINITIONS = {
    "motif_2": {"sequence": "WLEEWRERLECWW", "width": 13},  # First anchor
    "motif_1": {"sequence": "TQTQLGACMGALGLHLGSRLD", "width": 21},  # Second anchor
}

# Typical motif positions (relative to motif_1 at position 0)
TYPICAL_MOTIF2_OFFSET = -17  # motif_2 starts ~17 aa before motif_1

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


def calculate_phi_psi(residues):
    """Calculate phi/psi angles for a list of residues."""
    angles = []

    for i, residue in enumerate(residues):
        phi = None
        psi = None

        # Get phi angle (C(i-1) - N - CA - C)
        if i > 0:
            try:
                prev_c = residues[i - 1]["C"].get_vector()
                n = residue["N"].get_vector()
                ca = residue["CA"].get_vector()
                c = residue["C"].get_vector()
                phi = PDB.calc_dihedral(prev_c, n, ca, c) * 180 / np.pi
            except:
                pass

        # Get psi angle (N - CA - C - N(i+1))
        if i < len(residues) - 1:
            try:
                n = residue["N"].get_vector()
                ca = residue["CA"].get_vector()
                c = residue["C"].get_vector()
                next_n = residues[i + 1]["N"].get_vector()
                psi = PDB.calc_dihedral(n, ca, c, next_n) * 180 / np.pi
            except:
                pass

        angles.append(
            {"residue": i + 1, "phi": phi, "psi": psi, "resname": residue.get_resname()}
        )

    return pd.DataFrame(angles)


def assign_secondary_structure_from_angles(phi, psi):
    """
    Assign secondary structure based on phi/psi angles.

    Alpha helix: phi ~ -60°, psi ~ -45° (range: phi -80 to -40, psi -60 to -30)
    Beta sheet: phi ~ -120°, psi ~ 120° (range: phi -150 to -90, psi 90 to 150)
    """
    if phi is None or psi is None:
        return "C"  # Coil/undefined

    # Alpha helix region
    if -80 <= phi <= -40 and -60 <= psi <= -30:
        return "H"
    # 3-10 helix region (similar to alpha)
    if -80 <= phi <= -40 and -30 <= psi <= 0:
        return "G"
    # Beta sheet region
    if -150 <= phi <= -90 and 90 <= psi <= 150:
        return "E"
    # Beta sheet (other region)
    if -150 <= phi <= -90 and -180 <= psi <= -120:
        return "E"

    return "C"  # Coil


def smooth_ss_assignment(ss_string, min_length=3):
    """
    Smooth secondary structure assignment by requiring minimum helix/strand length.
    """
    ss_list = list(ss_string)
    n = len(ss_list)

    # Find runs of each element
    result = ["C"] * n
    i = 0
    while i < n:
        if ss_list[i] in ["H", "G"]:
            j = i
            while j < n and ss_list[j] in ["H", "G"]:
                j += 1
            if j - i >= min_length:
                for k in range(i, j):
                    result[k] = "H"
            i = j
        elif ss_list[i] == "E":
            j = i
            while j < n and ss_list[j] == "E":
                j += 1
            if j - i >= min_length:
                for k in range(i, j):
                    result[k] = "E"
            i = j
        else:
            i += 1

    return "".join(result)


def find_helical_regions(ss_string):
    """Find start and end positions of helical regions."""
    helices = []
    i = 0
    while i < len(ss_string):
        if ss_string[i] == "H":
            start = i
            while i < len(ss_string) and ss_string[i] == "H":
                i += 1
            helices.append((start + 1, i))  # 1-indexed
        else:
            i += 1
    return helices


# =============================================================================
# Structure Analysis
# =============================================================================


def load_consensus_structure(pdb_path):
    """Load the consensus PDB structure."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("consensus", pdb_path)
    return structure


def get_residues_from_structure(structure):
    """Extract residues from structure."""
    for model in structure:
        for chain in model:
            return list(chain.get_residues())
    return []


def analyze_consensus_structure(pdb_path):
    """Analyze secondary structure of consensus structure."""
    structure = load_consensus_structure(pdb_path)
    residues = get_residues_from_structure(structure)

    print(f"Loaded consensus structure with {len(residues)} residues")

    # Calculate phi/psi angles
    angles_df = calculate_phi_psi(residues)

    # Assign secondary structure
    angles_df["ss_raw"] = angles_df.apply(
        lambda row: assign_secondary_structure_from_angles(row["phi"], row["psi"]),
        axis=1,
    )

    # Smooth the assignment
    ss_raw = "".join(angles_df["ss_raw"].tolist())
    ss_smooth = smooth_ss_assignment(ss_raw, min_length=4)
    angles_df["ss"] = list(ss_smooth)

    # Find helical regions
    helices = find_helical_regions(ss_smooth)

    print(f"\nSecondary structure assignment:")
    print(f"Raw:      {ss_raw}")
    print(f"Smoothed: {ss_smooth}")
    print(f"\nHelical regions (residue positions):")
    for i, (start, end) in enumerate(helices):
        length = end - start
        print(f"  Helix {i+1}: residues {start}-{end} ({length} residues)")

    return angles_df, helices


# =============================================================================
# Motif Mapping
# =============================================================================


def map_motifs_to_structure(helices, consensus_length):
    """
    Map anchor motif positions to structural elements.

    Architecture (from MAST analysis):
    - motif_2 (13 aa) is typically at position -17 relative to motif_1
    - motif_1 (21 aa) is at position 0
    - Together they span ~34 aa (with 4 aa overlap)

    The C-terminal structures were aligned, so position 1 in structure
    corresponds to the start of the conserved region.
    """

    # Motif positions in the aligned structure
    # Based on the motif_positions.csv data, motif_2 is at ~-17 to -19
    # This means if the structure starts at position 0 of the alignment,
    # motif_2 starts before the alignment and motif_1 starts at ~position 17-19

    # Let's estimate where the anchor region would be in the 82-residue consensus
    # The anchor region (motif_2 + motif_1) spans approximately:
    # - motif_2: 13 aa
    # - gap/overlap: ~4 aa
    # - motif_1: 21 aa
    # Total anchor region: ~34-38 aa

    print("\n" + "=" * 60)
    print("MOTIF-STRUCTURE MAPPING ANALYSIS")
    print("=" * 60)

    print(f"\nMotif definitions:")
    print(
        f"  motif_2 (first anchor):  {MOTIF_DEFINITIONS['motif_2']['width']} aa - '{MOTIF_DEFINITIONS['motif_2']['sequence']}'"
    )
    print(
        f"  motif_1 (second anchor): {MOTIF_DEFINITIONS['motif_1']['width']} aa - '{MOTIF_DEFINITIONS['motif_1']['sequence']}'"
    )

    print(f"\nTypical motif arrangement (from MAST):")
    print(f"  motif_2 starts at position ~{TYPICAL_MOTIF2_OFFSET} relative to motif_1")
    print(f"  Combined anchor region: ~{13 + 21 - 4} aa (with ~4 aa overlap)")

    print(f"\nConsensus structure: {consensus_length} residues")
    print(f"Number of helices found: {len(helices)}")

    # Estimate anchor region in structure coordinates
    # The structure is the C-terminal ~80 residues
    # Anchor motifs are at the N-terminus of this region
    anchor_region_estimate = (1, 35)  # First ~35 residues

    print(
        f"\nEstimated anchor region in structure: residues {anchor_region_estimate[0]}-{anchor_region_estimate[1]}"
    )

    # Check overlap with helices
    print(f"\nHelix-anchor overlap analysis:")
    for i, (h_start, h_end) in enumerate(helices):
        h_length = h_end - h_start
        # Check if helix is within anchor region
        overlap_start = max(h_start, anchor_region_estimate[0])
        overlap_end = min(h_end, anchor_region_estimate[1])
        overlap = max(0, overlap_end - overlap_start)

        in_anchor = "YES" if overlap > h_length * 0.5 else "NO"
        print(
            f"  Helix {i+1} (res {h_start}-{h_end}, {h_length} aa): "
            f"overlap with anchor = {overlap} aa, in anchor region: {in_anchor}"
        )

    return anchor_region_estimate


# =============================================================================
# Visualization
# =============================================================================


def plot_ss_and_motifs(angles_df, helices, variance_df, output_path):
    """
    Create combined plot showing:
    - Secondary structure assignment
    - Helix locations
    - Estimated motif positions
    - Positional variance
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    n_residues = len(angles_df)
    positions = np.arange(1, n_residues + 1)

    # Panel A: Secondary structure track
    ax1 = axes[0]
    ss_colors = {"H": COLORS["accent"], "E": COLORS["secondary"], "C": "#E0E0E0"}

    for i, row in angles_df.iterrows():
        color = ss_colors.get(row["ss"], "#E0E0E0")
        ax1.bar(row["residue"], 1, width=1, color=color, edgecolor="none")

    # Add helix labels
    for i, (h_start, h_end) in enumerate(helices):
        mid = (h_start + h_end) / 2
        ax1.annotate(
            f"α{i+1}",
            xy=(mid, 0.5),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )

    ax1.set_ylabel("Secondary\nStructure", fontsize=11)
    ax1.set_yticks([])
    ax1.set_ylim(0, 1)
    ax1.set_title(
        "A. Secondary Structure Assignment", fontsize=12, fontweight="bold", loc="left"
    )

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=COLORS["accent"], label="α-helix"),
        Patch(facecolor=COLORS["secondary"], label="β-strand"),
        Patch(facecolor="#E0E0E0", label="Coil"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", frameon=False, ncol=3)

    # Panel B: Estimated motif positions
    ax2 = axes[1]

    # Motif_2 region (estimated)
    motif2_start = 1
    motif2_end = 13
    ax2.axvspan(
        motif2_start,
        motif2_end,
        alpha=0.4,
        color=COLORS["primary"],
        label="Motif 2 (13 aa)",
    )
    ax2.annotate(
        "Motif 2\n(13 aa)",
        xy=((motif2_start + motif2_end) / 2, 0.7),
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    # Motif_1 region (estimated, starts ~17 aa after motif_2 start, so ~13-4=9 aa after motif_2 ends)
    motif1_start = 17 - 4  # Accounting for typical offset and overlap
    motif1_end = motif1_start + 21
    ax2.axvspan(
        motif1_start,
        motif1_end,
        alpha=0.4,
        color=COLORS["highlight"],
        label="Motif 1 (21 aa)",
    )
    ax2.annotate(
        "Motif 1\n(21 aa)",
        xy=((motif1_start + motif1_end) / 2, 0.3),
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    # Variable region
    var_start = motif1_end
    var_end = n_residues
    ax2.axvspan(
        var_start, var_end, alpha=0.2, color=COLORS["light"], label="Variable region"
    )
    ax2.annotate(
        "Variable\nregion",
        xy=((var_start + var_end) / 2, 0.5),
        ha="center",
        va="center",
        fontsize=10,
    )

    ax2.set_ylabel("Motif\nPosition", fontsize=11)
    ax2.set_yticks([])
    ax2.set_ylim(0, 1)
    ax2.set_title(
        "B. Estimated Anchor Motif Positions",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    ax2.legend(loc="upper right", frameon=False, ncol=3)

    # Panel C: Positional variance
    ax3 = axes[2]

    if variance_df is not None and len(variance_df) == n_residues:
        variance = variance_df["std_total"].values
        # Normalize to mean variance
        mean_var = np.mean(variance)

        ax3.fill_between(positions, 0, variance, color=COLORS["secondary"], alpha=0.3)
        ax3.plot(positions, variance, color=COLORS["secondary"], linewidth=2)
        ax3.axhline(
            mean_var,
            color=COLORS["accent"],
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {mean_var:.2f} Å",
        )

        # Highlight low variance regions
        low_var_mask = variance < np.percentile(variance, 25)
        ax3.scatter(
            positions[low_var_mask],
            variance[low_var_mask],
            color=COLORS["light"],
            s=20,
            zorder=5,
        )

        ax3.set_ylabel("Positional\nVariance (Å)", fontsize=11)
        ax3.legend(loc="upper right", frameon=False)
    else:
        ax3.text(
            0.5,
            0.5,
            "Variance data not available",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )

    ax3.set_xlabel("Residue Position", fontsize=12)
    ax3.set_xlim(1, n_residues)
    ax3.set_title(
        "C. Structural Conservation (Positional Variance)",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )

    # Highlight anchor region on all panels
    for ax in axes:
        ax.axvline(34, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

    print(f"\nSaved figure: {output_path}")


def plot_ramachandran(angles_df, output_path):
    """Plot Ramachandran plot colored by secondary structure."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ss_colors = {"H": COLORS["accent"], "E": COLORS["secondary"], "C": "#888888"}

    for ss_type in ["C", "E", "H"]:
        subset = angles_df[angles_df["ss"] == ss_type]
        phi = subset["phi"].dropna()
        psi = subset["psi"].dropna()

        # Get matching psi values
        valid_idx = subset["phi"].notna() & subset["psi"].notna()
        phi = subset.loc[valid_idx, "phi"]
        psi = subset.loc[valid_idx, "psi"]

        label = {"H": "Helix", "E": "Strand", "C": "Coil"}[ss_type]
        ax.scatter(phi, psi, c=ss_colors[ss_type], label=label, s=50, alpha=0.7)

    # Add reference regions
    # Alpha helix region
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle(
            (-80, -60),
            40,
            30,
            fill=False,
            edgecolor=COLORS["accent"],
            linestyle="--",
            linewidth=2,
        )
    )
    ax.text(-60, -45, "α-helix", fontsize=10, color=COLORS["accent"])

    # Beta sheet region
    ax.add_patch(
        Rectangle(
            (-150, 90),
            60,
            60,
            fill=False,
            edgecolor=COLORS["secondary"],
            linestyle="--",
            linewidth=2,
        )
    )
    ax.text(-120, 120, "β-sheet", fontsize=10, color=COLORS["secondary"])

    ax.set_xlabel("φ (degrees)", fontsize=12)
    ax.set_ylabel("ψ (degrees)", fontsize=12)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(
        "Ramachandran Plot of uTP Consensus Structure", fontsize=14, fontweight="bold"
    )

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

    print(f"Saved Ramachandran plot: {output_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("uTP MOTIF-STRUCTURE MAPPING ANALYSIS")
    print("=" * 60)

    # Check if consensus structure exists
    if not CONSENSUS_PDB.exists():
        print(f"Error: Consensus structure not found at {CONSENSUS_PDB}")
        print("Please run 02_consensus_structure.py first.")
        return

    # Analyze consensus structure
    print(f"\nAnalyzing consensus structure: {CONSENSUS_PDB}")
    angles_df, helices = analyze_consensus_structure(CONSENSUS_PDB)

    # Map motifs to structure
    anchor_region = map_motifs_to_structure(helices, len(angles_df))

    # Load variance data if available
    variance_file = OUTPUT_DIR / "positional_variance.csv"
    variance_df = None
    if variance_file.exists():
        variance_df = pd.read_csv(variance_file)

    # Generate figures
    print("\nGenerating figures...")
    plot_ss_and_motifs(
        angles_df, helices, variance_df, OUTPUT_DIR / "motif_structure_mapping.svg"
    )
    plot_ramachandran(angles_df, OUTPUT_DIR / "ramachandran.svg")

    # Save analysis results
    angles_df.to_csv(OUTPUT_DIR / "secondary_structure.csv", index=False)

    helix_summary = []
    for i, (h_start, h_end) in enumerate(helices):
        helix_summary.append(
            {
                "helix_id": i + 1,
                "start": h_start,
                "end": h_end,
                "length": h_end - h_start,
                "in_anchor_region": h_start < 35,
            }
        )
    helix_df = pd.DataFrame(helix_summary)
    helix_df.to_csv(OUTPUT_DIR / "helix_summary.csv", index=False)

    # Print conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    helices_in_anchor = sum(1 for h in helices if h[0] < 35)

    print(f"\nTotal helices identified: {len(helices)}")
    print(f"Helices in anchor region (first ~35 residues): {helices_in_anchor}")

    if len(helices) >= 2 and helices_in_anchor >= 2:
        h1_start, h1_end = helices[0]
        h2_start, h2_end = helices[1]

        # Check if helices match motif positions
        motif2_region = (1, 13)  # Estimated
        motif1_region = (13, 34)  # Estimated

        h1_in_motif2 = (
            h1_start >= motif2_region[0] - 3 and h1_end <= motif2_region[1] + 10
        )
        h2_in_motif1 = (
            h2_start >= motif1_region[0] - 3 and h2_end <= motif1_region[1] + 10
        )

        print(
            f"\nHelix 1 (res {h1_start}-{h1_end}): corresponds to motif_2 region: {'YES' if h1_in_motif2 else 'PARTIAL/NO'}"
        )
        print(
            f"Helix 2 (res {h2_start}-{h2_end}): corresponds to motif_1 region: {'YES' if h2_in_motif1 else 'PARTIAL/NO'}"
        )

        if h1_in_motif2 and h2_in_motif1:
            print(
                "\n✓ The two anchor motifs (2 → 1) DO form the conserved U-bend structure!"
            )
            print("  - Motif 2 forms the first α-helix")
            print("  - Motif 1 forms the second α-helix")
        else:
            print("\n⚠ Partial correspondence between motifs and helices")
            print("  Further analysis may be needed with full-length structures")
    else:
        print(f"\n⚠ Expected 2+ helices in anchor region, found {helices_in_anchor}")
        print("  The structure may be more complex or data is limited")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
