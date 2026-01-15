#!/usr/bin/env python3
"""
Panel A: Sequence organization of the uTP region.

Shows:
1. Schematic of uTP organization (Mature Domain → Anchor 2 → Anchor 1 → Variable Linker)
2. Motif detection statistics (>90% have anchors)
3. Distribution of motif patterns

Data sources:
- experiments/utp_motif_analysis/output/motif_combinations.csv
- experiments/utp_motif_coverage/output/motif_patterns.csv
"""

import sys
from pathlib import Path

# Add parent to path for style import
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd
import numpy as np

from style import COLORS, PALETTE, apply_style, add_panel_label, stats_box, save_figure

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MOTIF_COMBINATIONS = PROJECT_ROOT / "experiments/utp_motif_analysis/output/motif_combinations.csv"
MOTIF_PATTERNS = PROJECT_ROOT / "experiments/utp_motif_coverage/output/motif_patterns.csv"
OUTPUT_DIR = Path(__file__).parent


def load_data():
    """Load motif analysis data."""
    # Load motif combinations from original MEME analysis
    combinations = pd.read_csv(MOTIF_COMBINATIONS)
    
    # Load extended motif patterns from MAST analysis
    patterns = pd.read_csv(MOTIF_PATTERNS)
    
    return combinations, patterns


def calculate_statistics(combinations, patterns):
    """Calculate key statistics for the figure."""
    stats = {}
    
    # From MAST analysis on HMM-predicted proteins
    total_hmm = len(patterns)
    with_motifs = patterns['n_motifs'].notna().sum()
    stats['motif_detection_rate'] = with_motifs / total_hmm * 100
    
    # Canonical order (starts with 2→1)
    starts_2_1 = patterns['starts_with_2_1'].sum()
    stats['canonical_order_rate'] = starts_2_1 / with_motifs * 100
    
    # Valid terminal patterns
    valid_terminal = patterns['is_valid_terminal'].sum()
    stats['valid_terminal_rate'] = valid_terminal / with_motifs * 100
    
    # Total sequences with motifs
    stats['n_with_motifs'] = int(with_motifs)
    stats['n_total_hmm'] = total_hmm
    
    return stats


def create_schematic(ax):
    """Create the uTP organization schematic."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    y_center = 10
    box_height = 6
    
    # Define regions with positions and widths
    regions = [
        {'name': 'Mature\nDomain', 'start': 2, 'width': 35, 'color': COLORS['primary'], 'alpha': 0.3},
        {'name': 'Anchor 2', 'start': 38, 'width': 8, 'color': COLORS['secondary'], 'alpha': 0.9},
        {'name': 'Anchor 1', 'start': 47, 'width': 12, 'color': COLORS['accent'], 'alpha': 0.9},
        {'name': 'Variable Linker', 'start': 60, 'width': 38, 'color': COLORS['highlight'], 'alpha': 0.5},
    ]
    
    for region in regions:
        # Draw box
        rect = FancyBboxPatch(
            (region['start'], y_center - box_height/2),
            region['width'], box_height,
            boxstyle="round,pad=0.02,rounding_size=0.5",
            facecolor=region['color'],
            edgecolor='white',
            linewidth=1.5,
            alpha=region['alpha']
        )
        ax.add_patch(rect)
        
        # Add label
        label_color = 'white' if region['alpha'] > 0.7 else COLORS['text']
        ax.text(region['start'] + region['width']/2, y_center,
                region['name'], ha='center', va='center',
                fontsize=8, fontweight='bold', color=label_color)
    
    # Add bracket for uTP region
    utp_start = 38
    utp_end = 98
    bracket_y = y_center + box_height/2 + 1.5
    
    ax.annotate('', xy=(utp_start, bracket_y), xytext=(utp_end, bracket_y),
                arrowprops=dict(arrowstyle='-', color=COLORS['text'], lw=1))
    ax.plot([utp_start, utp_start], [bracket_y, bracket_y - 0.8], color=COLORS['text'], lw=1)
    ax.plot([utp_end, utp_end], [bracket_y, bracket_y - 0.8], color=COLORS['text'], lw=1)
    ax.text((utp_start + utp_end)/2, bracket_y + 1.2, 'uTP (~120 aa)',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add N-terminus and C-terminus labels
    ax.text(0, y_center, 'N', ha='right', va='center', fontsize=9, fontweight='bold')
    ax.text(100, y_center, 'C', ha='left', va='center', fontsize=9, fontweight='bold')


def create_statistics_bars(ax, stats):
    """Create horizontal bar chart of detection statistics."""
    categories = [
        ('Motif detection\n(HMM-predicted)', stats['motif_detection_rate']),
        ('Canonical order\n(Anchor 2 > 1)', stats['canonical_order_rate']),
    ]
    
    y_pos = np.arange(len(categories))
    values = [c[1] for c in categories]
    labels = [c[0] for c in categories]
    
    bars = ax.barh(y_pos, values, height=0.6, color=COLORS['secondary'],
                   edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 105)
    ax.set_xlabel('Percentage (%)')
    
    # Add percentage labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 2, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}%', va='center', fontsize=9, fontweight='bold')
    
    # Add vertical reference lines
    for x in [50, 100]:
        ax.axvline(x=x, color=COLORS['text'], alpha=0.2, linestyle='--', linewidth=0.5)
    
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)


def create_pattern_distribution(ax, combinations):
    """Create bar chart of top motif patterns."""
    # Get top 8 patterns
    top_patterns = combinations.head(8).copy()
    
    # Simplify pattern names - use > instead of arrow for font compatibility
    top_patterns['short_pattern'] = top_patterns['motif_pattern'].apply(
        lambda x: x.replace(' → ', '>').replace('→', '>')
    )
    
    y_pos = np.arange(len(top_patterns))
    
    # Color by whether starts with 2
    colors = [COLORS['secondary'] if '2 >' in p or '2>' in p or p.startswith('2') else COLORS['primary'] 
              for p in top_patterns['short_pattern']]
    
    bars = ax.barh(y_pos, top_patterns['percentage'], height=0.7,
                   color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_patterns['short_pattern'], fontsize=8)
    ax.set_xlabel('Percentage (%)')
    ax.set_xlim(0, 28)
    ax.invert_yaxis()
    
    # Add count labels
    for i, (bar, row) in enumerate(zip(bars, top_patterns.itertuples())):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'n={row.count}', va='center', fontsize=7, color=COLORS['text'])
    
    ax.set_title('Top motif patterns', fontsize=10, fontweight='bold', loc='left')


def main():
    apply_style()
    
    # Load data
    combinations, patterns = load_data()
    stats = calculate_statistics(combinations, patterns)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(10, 7))
    
    # Layout: schematic on top, two panels below
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5], hspace=0.4, wspace=0.4)
    
    # Schematic (top, spanning both columns)
    ax_schematic = fig.add_subplot(gs[0, :])
    create_schematic(ax_schematic)
    
    # Statistics bars (bottom left)
    ax_stats = fig.add_subplot(gs[1, 0])
    create_statistics_bars(ax_stats, stats)
    
    # Pattern distribution (bottom right)
    ax_patterns = fig.add_subplot(gs[1, 1])
    create_pattern_distribution(ax_patterns, combinations)
    
    # Add statistics box
    stats_text = (f"n = {stats['n_with_motifs']} proteins with motifs\n"
                  f"(of {stats['n_total_hmm']} HMM-predicted)")
    stats_box(ax_stats, stats_text, x=0.98, y=0.05, ha='right', va='bottom')
    
    plt.tight_layout()
    
    # Save
    save_figure(fig, OUTPUT_DIR / "panel_a.svg")
    save_figure(fig, OUTPUT_DIR / "panel_a.png")
    
    print(f"\nKey statistics:")
    print(f"  Motif detection rate: {stats['motif_detection_rate']:.1f}%")
    print(f"  Canonical order rate: {stats['canonical_order_rate']:.1f}%")


if __name__ == "__main__":
    main()
