#!/usr/bin/env python3
"""
02_consensus_structure.py

Build a consensus structure from aligned uTP C-terminal structures
and analyze positional variance.

The consensus is built by:
1. Using the longest structure as reference
2. For each reference residue, finding spatially corresponding residues
   across all structures (within a proximity threshold)
3. Computing consensus position as mean of all corresponding positions
4. Recording standard deviation as positional variance

Input: Pre-aligned structures from data/utp-structures/c_term/aligned/
Output: Consensus structure PDB and variance analysis
"""

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from Bio import PDB
from Bio.PDB import PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "utp-structures" / "c_term" / "aligned"
OUTPUT_DIR = Path(__file__).parent / "output"

# Proximity threshold for residue correspondence (Angstroms)
CORRESPONDENCE_RADIUS = 3.0

# Minimum number of structures a position must be present in
MIN_STRUCTURE_COUNT = 5

# Figure styling
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
rcParams['font.size'] = 10
rcParams['axes.linewidth'] = 1.2
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

FIGURE_DPI = 300


# =============================================================================
# Functions
# =============================================================================

def load_structures(data_dir: Path) -> dict[str, PDB.Structure.Structure]:
    """Load all CIF structures from directory."""
    parser = PDB.MMCIFParser(QUIET=True)
    structures = {}
    
    for cif_file in sorted(data_dir.glob("*.cif")):
        name = cif_file.stem
        if name == "fold1":
            continue
        try:
            structure = parser.get_structure(name, cif_file)
            structures[name] = structure
        except Exception as e:
            print(f"Warning: Could not load {cif_file.name}: {e}")
    
    print(f"Loaded {len(structures)} structures")
    return structures


def get_residues(structure: PDB.Structure.Structure) -> list:
    """Get all residues from first chain of structure."""
    for model in structure:
        for chain in model:
            return list(chain.get_residues())
    return []


def get_residue_positions(structures: dict) -> dict[str, np.ndarray]:
    """Get center of mass for each residue in each structure."""
    positions = {}
    for name, structure in structures.items():
        residues = get_residues(structure)
        if residues:
            positions[name] = np.stack([r.center_of_mass() for r in residues])
    return positions


def find_longest_structure(structures: dict) -> tuple[str, PDB.Structure.Structure]:
    """Find the structure with the most residues."""
    max_len = 0
    longest_name = None
    longest_struct = None
    
    for name, structure in structures.items():
        residues = get_residues(structure)
        if len(residues) > max_len:
            max_len = len(residues)
            longest_name = name
            longest_struct = structure
    
    print(f"Reference structure: {longest_name} ({max_len} residues)")
    return longest_name, longest_struct


def compute_residue_counts(positions: dict[str, np.ndarray], 
                           ref_positions: np.ndarray,
                           radius: float) -> np.ndarray:
    """
    For each reference position, count how many structures have
    a residue within the radius.
    """
    n_ref = len(ref_positions)
    counts = np.zeros(n_ref)
    
    for name, pos in positions.items():
        for i, ref_pos in enumerate(ref_positions):
            distances = np.linalg.norm(pos - ref_pos, axis=1)
            if np.min(distances) < radius:
                counts[i] += 1
    
    return counts


def find_conserved_region(counts: np.ndarray, 
                          min_count: int) -> tuple[int, int]:
    """Find the start and end of the conserved region."""
    roi_start = None
    roi_end = None
    
    for i, count in enumerate(counts):
        if count >= min_count and roi_start is None:
            roi_start = i
        elif count < min_count and roi_start is not None:
            roi_end = i
            break
    
    if roi_start is None:
        roi_start = 0
    if roi_end is None:
        roi_end = len(counts)
    
    return roi_start, roi_end


def build_consensus_structure(structures: dict,
                              ref_name: str,
                              ref_structure: PDB.Structure.Structure,
                              roi_start: int,
                              roi_end: int,
                              radius: float) -> tuple[Structure, list[np.ndarray]]:
    """
    Build consensus structure from aligned structures.
    
    For each residue in the reference structure (within ROI):
    1. Find corresponding residues in all other structures
    2. Compute mean position
    3. Use majority residue type
    
    Returns consensus structure and list of position standard deviations.
    """
    positions = get_residue_positions(structures)
    ref_residues = get_residues(ref_structure)[roi_start:roi_end]
    ref_positions = positions[ref_name][roi_start:roi_end]
    
    # Create new structure
    consensus = Structure("consensus")
    model = Model(0)
    chain = Chain("A")
    model.add(chain)
    consensus.add(model)
    
    stderrs = []
    
    for idx, (ref_residue, ref_pos) in enumerate(zip(ref_residues, ref_positions)):
        # Find corresponding residues in other structures
        corresponding = []
        
        for name, struct in structures.items():
            if name == ref_name:
                continue
            
            struct_positions = positions.get(name)
            if struct_positions is None:
                continue
            
            distances = np.linalg.norm(struct_positions - ref_pos, axis=1)
            min_dist_idx = np.argmin(distances)
            
            if distances[min_dist_idx] < radius:
                struct_residues = get_residues(struct)
                if min_dist_idx < len(struct_residues):
                    corresponding.append(struct_residues[min_dist_idx])
        
        if len(corresponding) < MIN_STRUCTURE_COUNT:
            continue
        
        # Compute consensus position
        corr_positions = np.array([r.center_of_mass() for r in corresponding])
        consensus_pos = np.mean(corr_positions, axis=0)
        pos_std = np.std(corr_positions, axis=0)
        stderrs.append(pos_std)
        
        # Get majority residue type
        resnames = [r.get_resname() for r in corresponding]
        majority_resname = Counter(resnames).most_common(1)[0][0]
        
        # Find a template residue with the majority type
        template = None
        for r in corresponding:
            if r.get_resname() == majority_resname:
                template = r.copy()
                break
        
        if template is None:
            template = corresponding[0].copy()
        
        # Transform template to consensus position
        template.detach_parent()
        trans_vector = consensus_pos - template.center_of_mass()
        template.transform(np.eye(3), trans_vector)
        
        # Update residue ID
        template.id = (' ', idx + 1, ' ')
        chain.add(template)
    
    print(f"Consensus structure: {len(list(chain.get_residues()))} residues")
    return consensus, stderrs


def plot_positional_variance(stderrs: list[np.ndarray], output_path: Path):
    """Plot positional variance along the chain."""
    # Compute mean variance at each position (mean of x, y, z std)
    variance = np.array([np.mean(s) for s in stderrs])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    positions = np.arange(1, len(variance) + 1)
    
    ax.fill_between(positions, 0, variance, color='#2E86AB', alpha=0.3)
    ax.plot(positions, variance, color='#2E86AB', linewidth=2)
    
    # Highlight low-variance regions
    low_var_threshold = np.percentile(variance, 25)
    low_var_mask = variance <= low_var_threshold
    ax.scatter(positions[low_var_mask], variance[low_var_mask], 
               color='#28A745', s=20, zorder=5, label=f'Low variance (≤{low_var_threshold:.1f} Å)')
    
    ax.axhline(np.mean(variance), color='#E94F37', linestyle='--', linewidth=1.5,
               label=f'Mean: {np.mean(variance):.2f} Å')
    
    ax.set_xlabel('Residue Position', fontsize=12)
    ax.set_ylabel('Positional Std Dev (Å)', fontsize=12)
    ax.set_title('Structural Variance Along uTP Chain', fontsize=14, fontweight='bold')
    ax.legend(frameon=False, loc='upper right')
    
    ax.set_xlim(1, len(variance))
    ax.set_ylim(0, max(variance) * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Saved variance plot: {output_path}")
    
    return variance


def plot_3d_variance(stderrs: list[np.ndarray], output_path: Path):
    """Plot variance in each dimension (x, y, z)."""
    variance_xyz = np.array(stderrs)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    
    labels = ['X', 'Y', 'Z']
    colors = ['#E94F37', '#28A745', '#2E86AB']
    
    positions = np.arange(1, len(variance_xyz) + 1)
    
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.fill_between(positions, 0, variance_xyz[:, i], color=color, alpha=0.3)
        ax.plot(positions, variance_xyz[:, i], color=color, linewidth=1.5)
        ax.set_xlabel('Residue Position', fontsize=10)
        ax.set_ylabel(f'{label} Std Dev (Å)', fontsize=10)
        ax.set_title(f'{label} Dimension', fontsize=12)
        ax.set_xlim(1, len(variance_xyz))
    
    plt.suptitle('Positional Variance by Dimension', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Saved 3D variance plot: {output_path}")


def save_consensus_pdb(consensus: Structure, output_path: Path):
    """Save consensus structure to PDB file."""
    io = PDBIO()
    io.set_structure(consensus)
    io.save(str(output_path))
    
    # Add CRYST1 record for compatibility
    with open(output_path, 'r') as f:
        lines = f.readlines()
    
    with open(output_path, 'w') as f:
        f.write("CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n")
        for line in lines:
            f.write(line)
    
    print(f"Saved consensus structure: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("uTP Consensus Structure Analysis")
    print("=" * 60)
    
    # Load structures
    print(f"\nLoading structures from: {DATA_DIR}")
    structures = load_structures(DATA_DIR)
    
    if len(structures) < 3:
        print("Error: Need at least 3 structures for consensus analysis")
        return
    
    # Find reference structure
    print("\nFinding reference structure...")
    ref_name, ref_structure = find_longest_structure(structures)
    
    # Compute residue correspondence counts
    print("\nAnalyzing structural conservation...")
    positions = get_residue_positions(structures)
    ref_positions = positions[ref_name]
    
    counts = compute_residue_counts(positions, ref_positions, CORRESPONDENCE_RADIUS)
    
    # Find conserved region
    roi_start, roi_end = find_conserved_region(counts, MIN_STRUCTURE_COUNT)
    print(f"Conserved region: residues {roi_start + 1} - {roi_end} ({roi_end - roi_start} residues)")
    
    # Build consensus structure
    print("\nBuilding consensus structure...")
    consensus, stderrs = build_consensus_structure(
        structures, ref_name, ref_structure, roi_start, roi_end, CORRESPONDENCE_RADIUS
    )
    
    # Save consensus structure
    save_consensus_pdb(consensus, OUTPUT_DIR / "consensus_structure.pdb")
    
    # Save variance data
    variance = np.array([np.linalg.norm(s) for s in stderrs])
    variance_df = pd.DataFrame({
        'residue_position': range(1, len(variance) + 1),
        'std_x': [s[0] for s in stderrs],
        'std_y': [s[1] for s in stderrs],
        'std_z': [s[2] for s in stderrs],
        'std_total': variance
    })
    variance_df.to_csv(OUTPUT_DIR / "positional_variance.csv", index=False)
    print(f"Saved variance data: {OUTPUT_DIR / 'positional_variance.csv'}")
    
    # Generate figures
    print("\nGenerating figures...")
    mean_variance = plot_positional_variance(stderrs, OUTPUT_DIR / "positional_variance.svg")
    plot_3d_variance(stderrs, OUTPUT_DIR / "variance_by_dimension.svg")
    
    # Save summary statistics
    summary = {
        'n_structures': len(structures),
        'reference_structure': ref_name,
        'roi_start': roi_start + 1,
        'roi_end': roi_end,
        'consensus_length': len(stderrs),
        'mean_variance': np.mean(mean_variance),
        'median_variance': np.median(mean_variance),
        'min_variance': np.min(mean_variance),
        'max_variance': np.max(mean_variance),
        'correspondence_radius': CORRESPONDENCE_RADIUS,
        'min_structure_count': MIN_STRUCTURE_COUNT
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(OUTPUT_DIR / "consensus_statistics.csv", index=False)
    
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Reference structure: {ref_name}")
    print(f"Conserved region: {roi_start + 1} - {roi_end}")
    print(f"Consensus length: {len(stderrs)} residues")
    print(f"Mean positional variance: {np.mean(mean_variance):.2f} Å")
    print(f"Variance range: {np.min(mean_variance):.2f} - {np.max(mean_variance):.2f} Å")
    print("=" * 60)


if __name__ == "__main__":
    main()
