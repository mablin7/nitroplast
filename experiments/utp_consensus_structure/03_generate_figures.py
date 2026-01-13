#!/usr/bin/env python3
"""
03_generate_figures.py

Generate publication-quality figures for the uTP consensus structure analysis.

This script combines outputs from 01_rmsd_analysis.py and 02_consensus_structure.py
to create cohesive figure panels suitable for the manuscript.

Input: Output files from previous scripts
Output: Combined figure panels and summary statistics
"""

from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from Bio import PDB
from Bio.PDB import PDBIO
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "utp-structures" / "c_term" / "aligned"
OUTPUT_DIR = Path(__file__).parent / "output"

# Publication figure styling
plt.style.use('default')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']
rcParams['font.size'] = 9
rcParams['axes.linewidth'] = 1.0
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['xtick.major.width'] = 1.0
rcParams['ytick.major.width'] = 1.0
rcParams['xtick.major.size'] = 4
rcParams['ytick.major.size'] = 4
rcParams['legend.frameon'] = False
rcParams['legend.fontsize'] = 8

# Color palette - coordinated scheme
COLORS = {
    'primary': '#2E4057',      # Dark blue-gray
    'secondary': '#048A81',    # Teal
    'accent': '#E85D04',       # Orange
    'light': '#90BE6D',        # Green
    'highlight': '#F9C74F',    # Yellow
    'background': '#F8F9FA',   # Light gray
    'text': '#212529',         # Dark text
}

FIGURE_DPI = 300


# =============================================================================
# Helper Functions
# =============================================================================

def add_panel_label(ax, label, x=-0.15, y=1.05, fontsize=14):
    """Add panel label (A, B, C, etc.) to axis."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight='bold', va='top', ha='left')


def style_axis(ax):
    """Apply consistent styling to axis."""
    ax.tick_params(axis='both', which='major', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(COLORS['text'])


# =============================================================================
# Figure Components
# =============================================================================

def plot_rmsd_panel(ax, rmsd_matrix_path: Path):
    """Create RMSD distribution panel."""
    rmsd_df = pd.read_csv(rmsd_matrix_path, index_col=0)
    values = rmsd_df.values[np.triu_indices_from(rmsd_df.values, k=1)]
    values = values[~np.isnan(values)]
    
    # Histogram
    n, bins, patches = ax.hist(values, bins=25, color=COLORS['secondary'], 
                                edgecolor='white', linewidth=0.5, alpha=0.85)
    
    # Statistics
    mean_rmsd = np.mean(values)
    std_rmsd = np.std(values)
    
    ax.axvline(mean_rmsd, color=COLORS['accent'], linestyle='-', linewidth=2)
    ax.axvspan(mean_rmsd - std_rmsd, mean_rmsd + std_rmsd, 
               alpha=0.15, color=COLORS['accent'])
    
    ax.set_xlabel('RMSD (Å)', fontsize=10)
    ax.set_ylabel('Structure pairs', fontsize=10)
    
    # Add statistics text
    stats_text = f'μ = {mean_rmsd:.1f} ± {std_rmsd:.1f} Å\nn = {len(values)}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
            va='top', ha='right', bbox=dict(boxstyle='round,pad=0.3', 
            facecolor='white', edgecolor='none', alpha=0.8))
    
    style_axis(ax)
    return mean_rmsd, std_rmsd, len(values)


def plot_dendrogram_panel(ax, rmsd_matrix_path: Path):
    """Create hierarchical clustering dendrogram panel."""
    rmsd_df = pd.read_csv(rmsd_matrix_path, index_col=0)
    names = list(rmsd_df.columns)
    
    # Condensed distance matrix
    dist_condensed = squareform(rmsd_df.values)
    dist_condensed = np.nan_to_num(dist_condensed, nan=np.nanmax(dist_condensed))
    
    # Hierarchical clustering
    Z = linkage(dist_condensed, method='average')
    
    # Shortened labels
    short_labels = [n.replace('_c_term', '').replace('kc1_p2_', '')[:12] 
                    for n in names]
    
    dendrogram(Z, labels=short_labels, leaf_rotation=90, leaf_font_size=5,
               ax=ax, color_threshold=0.7 * max(Z[:, 2]),
               above_threshold_color=COLORS['primary'])
    
    ax.set_xlabel('Structure', fontsize=10)
    ax.set_ylabel('RMSD (Å)', fontsize=10)
    ax.set_xticklabels([])  # Too crowded, remove labels
    
    style_axis(ax)


def plot_variance_panel(ax, variance_path: Path):
    """Create positional variance panel."""
    variance_df = pd.read_csv(variance_path)
    
    positions = variance_df['residue_position'].values
    variance = variance_df['std_total'].values
    
    # Mean of x, y, z for plotting
    mean_var = (variance_df['std_x'] + variance_df['std_y'] + variance_df['std_z']) / 3
    
    # Fill and line
    ax.fill_between(positions, 0, mean_var, color=COLORS['secondary'], alpha=0.25)
    ax.plot(positions, mean_var, color=COLORS['secondary'], linewidth=1.5)
    
    # Mark conserved regions (low variance)
    threshold = np.percentile(mean_var, 25)
    conserved_mask = mean_var <= threshold
    ax.scatter(positions[conserved_mask], mean_var[conserved_mask],
               c=COLORS['light'], s=15, zorder=5, edgecolor='white', linewidth=0.5)
    
    # Mean line
    mean_overall = np.mean(mean_var)
    ax.axhline(mean_overall, color=COLORS['accent'], linestyle='--', linewidth=1.5)
    
    ax.set_xlabel('Residue position', fontsize=10)
    ax.set_ylabel('Positional variance (Å)', fontsize=10)
    ax.set_xlim(positions.min(), positions.max())
    ax.set_ylim(0, mean_var.max() * 1.15)
    
    # Stats text
    ax.text(0.95, 0.95, f'Mean: {mean_overall:.2f} Å', transform=ax.transAxes,
            fontsize=8, va='top', ha='right', color=COLORS['accent'])
    
    style_axis(ax)
    return mean_overall


def plot_structure_schematic(ax):
    """Create schematic of U-bend structure."""
    # Simple schematic showing two helices in U-bend
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.set_aspect('equal')
    
    # Helix 1 (left, going up)
    helix1_x = np.linspace(20, 20, 50)
    helix1_y = np.linspace(10, 45, 50)
    ax.plot(helix1_x, helix1_y, color=COLORS['secondary'], linewidth=8, 
            solid_capstyle='round', zorder=2)
    
    # Turn region (top)
    turn_theta = np.linspace(np.pi/2, -np.pi/2, 30)
    turn_x = 40 + 20 * np.cos(turn_theta)
    turn_y = 45 + 8 * np.sin(turn_theta)
    ax.plot(turn_x, turn_y, color=COLORS['light'], linewidth=6, 
            solid_capstyle='round', zorder=1)
    
    # Helix 2 (right, going down)
    helix2_x = np.linspace(60, 60, 50)
    helix2_y = np.linspace(45, 10, 50)
    ax.plot(helix2_x, helix2_y, color=COLORS['secondary'], linewidth=8,
            solid_capstyle='round', zorder=2)
    
    # Labels
    ax.text(20, 5, 'N', fontsize=10, ha='center', va='top', fontweight='bold')
    ax.text(60, 5, 'C', fontsize=10, ha='center', va='top', fontweight='bold')
    ax.text(40, 55, 'Turn', fontsize=8, ha='center', va='bottom', color=COLORS['light'])
    
    # Annotations
    ax.annotate('', xy=(15, 35), xytext=(15, 20),
                arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=1.5))
    ax.annotate('', xy=(65, 20), xytext=(65, 35),
                arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=1.5))
    
    ax.text(10, 27, 'α1', fontsize=9, ha='right')
    ax.text(70, 27, 'α2', fontsize=9, ha='left')
    
    ax.axis('off')
    ax.set_title('Conserved U-bend fold', fontsize=10, fontweight='bold', pad=10)


# =============================================================================
# Main Figure Generation
# =============================================================================

def generate_main_figure():
    """Generate main combined figure for manuscript."""
    
    rmsd_path = OUTPUT_DIR / "rmsd_matrix.csv"
    variance_path = OUTPUT_DIR / "positional_variance.csv"
    
    # Check if input files exist
    if not rmsd_path.exists():
        print(f"Error: {rmsd_path} not found. Run 01_rmsd_analysis.py first.")
        return
    if not variance_path.exists():
        print(f"Error: {variance_path} not found. Run 02_consensus_structure.py first.")
        return
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(10, 7))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1.2, 1, 1],
                  hspace=0.35, wspace=0.35)
    
    # Panel A: Structure schematic
    ax_schematic = fig.add_subplot(gs[0, 0])
    plot_structure_schematic(ax_schematic)
    add_panel_label(ax_schematic, 'A', x=-0.05, y=1.1)
    
    # Panel B: RMSD histogram  
    ax_rmsd = fig.add_subplot(gs[0, 1:])
    mean_rmsd, std_rmsd, n_pairs = plot_rmsd_panel(ax_rmsd, rmsd_path)
    add_panel_label(ax_rmsd, 'B', x=-0.08, y=1.05)
    
    # Panel C: Dendrogram (spanning full width bottom left)
    ax_dendro = fig.add_subplot(gs[1, :2])
    plot_dendrogram_panel(ax_dendro, rmsd_path)
    add_panel_label(ax_dendro, 'C', x=-0.05, y=1.05)
    
    # Panel D: Positional variance
    ax_var = fig.add_subplot(gs[1, 2])
    mean_var = plot_variance_panel(ax_var, variance_path)
    add_panel_label(ax_var, 'D', x=-0.15, y=1.05)
    
    # Save figure
    output_path = OUTPUT_DIR / "figure_structure_panel.svg"
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.png'), dpi=FIGURE_DPI, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved main figure: {output_path}")
    
    return {
        'mean_rmsd': mean_rmsd,
        'std_rmsd': std_rmsd,
        'n_pairs': n_pairs,
        'mean_variance': mean_var
    }


def generate_supplementary_figure():
    """Generate supplementary figure with additional details."""
    
    variance_path = OUTPUT_DIR / "positional_variance.csv"
    
    if not variance_path.exists():
        print(f"Error: {variance_path} not found.")
        return
    
    variance_df = pd.read_csv(variance_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    
    dimensions = ['std_x', 'std_y', 'std_z']
    labels = ['X dimension', 'Y dimension', 'Z dimension']
    colors = [COLORS['accent'], COLORS['light'], COLORS['secondary']]
    
    for ax, dim, label, color in zip(axes, dimensions, labels, colors):
        positions = variance_df['residue_position'].values
        values = variance_df[dim].values
        
        ax.fill_between(positions, 0, values, color=color, alpha=0.3)
        ax.plot(positions, values, color=color, linewidth=1.5)
        ax.axhline(np.mean(values), color=color, linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Residue position', fontsize=10)
        ax.set_ylabel('Std dev (Å)', fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlim(positions.min(), positions.max())
        
        style_axis(ax)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "figure_variance_dimensions.svg"
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.png'), dpi=FIGURE_DPI,
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved supplementary figure: {output_path}")


def write_statistics_summary(stats: dict):
    """Write summary statistics for methods/results sections."""
    
    summary_path = OUTPUT_DIR / "structure_statistics_summary.txt"
    
    # Load additional data
    rmsd_stats_path = OUTPUT_DIR / "rmsd_statistics.csv"
    consensus_stats_path = OUTPUT_DIR / "consensus_statistics.csv"
    
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("uTP Structure Analysis - Statistics Summary\n")
        f.write("For use in manuscript methods/results sections\n")
        f.write("=" * 60 + "\n\n")
        
        # RMSD statistics
        if rmsd_stats_path.exists():
            rmsd_df = pd.read_csv(rmsd_stats_path)
            f.write("## RMSD Analysis\n")
            f.write(f"- Number of structures: {rmsd_df['n_structures'].values[0]}\n")
            f.write(f"- Pairwise comparisons: {rmsd_df['n_pairwise_comparisons'].values[0]}\n")
            f.write(f"- Mean RMSD: {rmsd_df['mean_rmsd'].values[0]:.2f} ± {rmsd_df['std_rmsd'].values[0]:.2f} Å\n")
            f.write(f"- RMSD range: {rmsd_df['min_rmsd'].values[0]:.2f} - {rmsd_df['max_rmsd'].values[0]:.2f} Å\n")
            f.write(f"- Median RMSD: {rmsd_df['median_rmsd'].values[0]:.2f} Å\n\n")
        
        # Consensus statistics
        if consensus_stats_path.exists():
            cons_df = pd.read_csv(consensus_stats_path)
            f.write("## Consensus Structure\n")
            f.write(f"- Reference structure: {cons_df['reference_structure'].values[0]}\n")
            f.write(f"- Conserved region: residues {cons_df['roi_start'].values[0]} - {cons_df['roi_end'].values[0]}\n")
            f.write(f"- Consensus length: {cons_df['consensus_length'].values[0]} residues\n")
            f.write(f"- Mean positional variance: {cons_df['mean_variance'].values[0]:.2f} Å\n")
            f.write(f"- Variance range: {cons_df['min_variance'].values[0]:.2f} - {cons_df['max_variance'].values[0]:.2f} Å\n\n")
        
        # Methods text
        f.write("## Suggested Methods Text\n")
        f.write("-" * 40 + "\n")
        f.write("AlphaFold3 structures were predicted for proteins containing uTP sequences. ")
        if rmsd_stats_path.exists():
            n = rmsd_df['n_structures'].values[0]
            f.write(f"{n} structures with strong C-terminal sequence similarity were selected ")
        f.write("and aligned using PyMOL's CE alignment algorithm. ")
        f.write("Pairwise RMSD was calculated for all structure pairs using BioPython's Superimposer ")
        f.write("after aligning on Cα atoms. ")
        f.write("A consensus structure was built by computing the mean atomic position for each ")
        f.write("aligned residue across all structures (within 3.0 Å radius), with positional ")
        f.write("variance recorded as standard deviation. ")
        f.write("Hierarchical clustering (average linkage) was performed on the RMSD distance matrix.\n")
        
    print(f"Saved statistics summary: {summary_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating Publication Figures")
    print("=" * 60)
    
    # Generate main figure
    print("\n1. Generating main figure panel...")
    stats = generate_main_figure()
    
    if stats:
        # Generate supplementary figure
        print("\n2. Generating supplementary figure...")
        generate_supplementary_figure()
        
        # Write statistics summary
        print("\n3. Writing statistics summary...")
        write_statistics_summary(stats)
    
    print("\n" + "=" * 60)
    print("Figure generation complete!")
    print("=" * 60)
    print(f"\nOutput files in: {OUTPUT_DIR}")
    print("- figure_structure_panel.svg/png (main figure)")
    print("- figure_variance_dimensions.svg/png (supplementary)")
    print("- structure_statistics_summary.txt (for methods section)")


if __name__ == "__main__":
    main()
