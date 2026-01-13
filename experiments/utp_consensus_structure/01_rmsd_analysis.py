#!/usr/bin/env python3
"""
01_rmsd_analysis.py

Compute pairwise RMSD between aligned uTP C-terminal structures
and perform hierarchical clustering analysis.

Input: Pre-aligned structures from data/utp-structures/c_term/aligned/
Output: RMSD matrix, dendrogram, and statistics
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from Bio import PDB
from Bio.PDB import Superimposer
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "utp-structures" / "c_term" / "aligned"
OUTPUT_DIR = Path(__file__).parent / "output"

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
        # Skip fold1.cif which appears to be a template
        if name == "fold1":
            continue
        try:
            structure = parser.get_structure(name, cif_file)
            structures[name] = structure
        except Exception as e:
            print(f"Warning: Could not load {cif_file.name}: {e}")
    
    print(f"Loaded {len(structures)} structures")
    return structures


def get_ca_atoms(structure: PDB.Structure.Structure) -> list:
    """Extract CA atoms from a structure."""
    ca_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_atoms.append(residue["CA"])
    return ca_atoms


def compute_rmsd(structure1: PDB.Structure.Structure, 
                 structure2: PDB.Structure.Structure) -> float:
    """
    Compute RMSD between two structures using CA atoms.
    Truncates to the shorter structure for comparison.
    """
    atoms1 = get_ca_atoms(structure1)
    atoms2 = get_ca_atoms(structure2)
    
    # Truncate to shorter length
    min_len = min(len(atoms1), len(atoms2))
    if min_len == 0:
        return np.nan
    
    atoms1 = atoms1[:min_len]
    atoms2 = atoms2[:min_len]
    
    # Superimpose and compute RMSD
    superimposer = Superimposer()
    superimposer.set_atoms(atoms1, atoms2)
    
    return superimposer.rms


def compute_rmsd_matrix(structures: dict) -> tuple[pd.DataFrame, list]:
    """Compute pairwise RMSD matrix for all structures."""
    names = list(structures.keys())
    n = len(names)
    
    rmsd_matrix = np.zeros((n, n))
    
    print(f"Computing {n * (n - 1) // 2} pairwise RMSDs...")
    
    for i, j in combinations(range(n), 2):
        rmsd = compute_rmsd(structures[names[i]], structures[names[j]])
        rmsd_matrix[i, j] = rmsd
        rmsd_matrix[j, i] = rmsd
    
    df = pd.DataFrame(rmsd_matrix, index=names, columns=names)
    return df, names


def plot_rmsd_histogram(rmsd_matrix: pd.DataFrame, output_path: Path):
    """Plot histogram of pairwise RMSD values."""
    # Extract upper triangle (excluding diagonal)
    values = rmsd_matrix.values[np.triu_indices_from(rmsd_matrix.values, k=1)]
    values = values[~np.isnan(values)]
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    ax.hist(values, bins=30, color='#2E86AB', edgecolor='white', linewidth=0.5, alpha=0.8)
    
    # Add statistics
    mean_rmsd = np.mean(values)
    std_rmsd = np.std(values)
    
    ax.axvline(mean_rmsd, color='#E94F37', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_rmsd:.2f} ± {std_rmsd:.2f} Å')
    
    ax.set_xlabel('RMSD (Å)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Pairwise RMSD Distribution', fontsize=14, fontweight='bold')
    ax.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Saved histogram: {output_path}")
    return mean_rmsd, std_rmsd


def plot_dendrogram(rmsd_matrix: pd.DataFrame, names: list, output_path: Path):
    """Create hierarchical clustering dendrogram from RMSD matrix."""
    # Convert to condensed distance matrix
    dist_condensed = squareform(rmsd_matrix.values)
    
    # Handle any NaN values
    dist_condensed = np.nan_to_num(dist_condensed, nan=np.nanmax(dist_condensed))
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(dist_condensed, method='average')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Shorten labels for readability
    short_labels = [n.replace('_c_term', '').replace('kc1_p2_', '') for n in names]
    
    dendrogram(
        linkage_matrix,
        labels=short_labels,
        leaf_rotation=90,
        leaf_font_size=7,
        ax=ax,
        color_threshold=0.7 * max(linkage_matrix[:, 2])
    )
    
    ax.set_xlabel('Structure', fontsize=12)
    ax.set_ylabel('RMSD (Å)', fontsize=12)
    ax.set_title('Hierarchical Clustering of uTP Structures', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Saved dendrogram: {output_path}")
    
    return linkage_matrix


def plot_rmsd_heatmap(rmsd_matrix: pd.DataFrame, output_path: Path):
    """Create heatmap of RMSD matrix."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    im = ax.imshow(rmsd_matrix.values, cmap='viridis', aspect='auto')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='RMSD (Å)', shrink=0.8)
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_title('Pairwise RMSD Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Structure', fontsize=12)
    ax.set_ylabel('Structure', fontsize=12)
    
    # Set tick labels (sparse for readability)
    n = len(rmsd_matrix)
    tick_step = max(1, n // 10)
    ax.set_xticks(range(0, n, tick_step))
    ax.set_yticks(range(0, n, tick_step))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("uTP Structure RMSD Analysis")
    print("=" * 60)
    
    # Load structures
    print(f"\nLoading structures from: {DATA_DIR}")
    structures = load_structures(DATA_DIR)
    
    if len(structures) < 2:
        print("Error: Need at least 2 structures for RMSD analysis")
        return
    
    # Compute RMSD matrix
    print("\nComputing pairwise RMSD...")
    rmsd_matrix, names = compute_rmsd_matrix(structures)
    
    # Save RMSD matrix
    rmsd_path = OUTPUT_DIR / "rmsd_matrix.csv"
    rmsd_matrix.to_csv(rmsd_path)
    print(f"Saved RMSD matrix: {rmsd_path}")
    
    # Generate figures
    print("\nGenerating figures...")
    
    # Histogram
    mean_rmsd, std_rmsd = plot_rmsd_histogram(
        rmsd_matrix, 
        OUTPUT_DIR / "rmsd_histogram.svg"
    )
    
    # Dendrogram
    plot_dendrogram(
        rmsd_matrix, 
        names,
        OUTPUT_DIR / "rmsd_dendrogram.svg"
    )
    
    # Heatmap
    plot_rmsd_heatmap(
        rmsd_matrix,
        OUTPUT_DIR / "rmsd_heatmap.svg"
    )
    
    # Summary statistics
    values = rmsd_matrix.values[np.triu_indices_from(rmsd_matrix.values, k=1)]
    values = values[~np.isnan(values)]
    
    summary = {
        'n_structures': len(structures),
        'n_pairwise_comparisons': len(values),
        'mean_rmsd': mean_rmsd,
        'std_rmsd': std_rmsd,
        'min_rmsd': np.min(values),
        'max_rmsd': np.max(values),
        'median_rmsd': np.median(values)
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(OUTPUT_DIR / "rmsd_statistics.csv", index=False)
    
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Number of structures: {summary['n_structures']}")
    print(f"Pairwise comparisons: {summary['n_pairwise_comparisons']}")
    print(f"Mean RMSD: {summary['mean_rmsd']:.2f} ± {summary['std_rmsd']:.2f} Å")
    print(f"Range: {summary['min_rmsd']:.2f} - {summary['max_rmsd']:.2f} Å")
    print(f"Median RMSD: {summary['median_rmsd']:.2f} Å")
    print("=" * 60)


if __name__ == "__main__":
    main()
