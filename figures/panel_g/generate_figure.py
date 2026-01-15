#!/usr/bin/env python3
"""
Panel G: Variance partitioning.

Shows that uTP status explains more unique variance than function.

Data source:
- experiments/utp_functional_annotation/output/variance_partitioning.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from style import COLORS, apply_style, save_figure

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
VARIANCE_PARTITION = PROJECT_ROOT / "experiments/utp_functional_annotation/output/variance_partitioning.csv"
OUTPUT_DIR = Path(__file__).parent


def load_data():
    """Load variance partitioning data."""
    if VARIANCE_PARTITION.exists():
        return pd.read_csv(VARIANCE_PARTITION)
    return None


def main():
    apply_style()
    
    variance = load_data()
    
    # Create single-panel figure
    fig, ax = plt.subplots(figsize=(5, 5))
    
    if variance is not None:
        # Use fraction_coil as representative property
        row = variance[variance['property'] == 'fraction_coil'].iloc[0]
        
        labels = ['uTP status\n(unique)', 'Function\n(unique)', 'Shared']
        sizes = [
            row['utp_unique_pct'],
            row['cog_unique_pct'],
            row['shared_pct'],
        ]
        colors = [COLORS['secondary'], COLORS['primary'], COLORS['highlight']]
    else:
        # Fallback values
        labels = ['uTP status\n(unique)', 'Function\n(unique)', 'Shared']
        sizes = [12.0, 3.0, 6.2]
        colors = [COLORS['secondary'], COLORS['primary'], COLORS['highlight']]
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors,
        autopct=lambda pct: f'{pct:.1f}%',
        startangle=90, pctdistance=0.6,
        wedgeprops=dict(edgecolor='white', linewidth=2, width=0.7),
        explode=(0.02, 0, 0)
    )
    
    # Style autopct text
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    # Add legend instead of labels on pie
    ax.legend(wedges, labels, loc='center', fontsize=9,
              frameon=False, bbox_to_anchor=(0.5, 0.5))
    
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save
    save_figure(fig, OUTPUT_DIR / "panel_g.svg")
    save_figure(fig, OUTPUT_DIR / "panel_g.png")
    
    print("\nVariance explained:")
    for label, size in zip(labels, sizes):
        print(f"  {label.replace(chr(10), ' ')}: {size:.1f}%")


if __name__ == "__main__":
    main()
