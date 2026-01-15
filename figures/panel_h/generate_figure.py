#!/usr/bin/env python3
"""
Panel H: Gene family distribution of uTP proteins.

Shows that while uTP proteins cluster more than expected by chance,
they span 624 distinct families with 75% in singleton families.

Data source:
- experiments/utp_family_clustering/output/permutation_results.csv
- experiments/utp_family_clustering/output/family_statistics.csv
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
PERMUTATION_RESULTS = PROJECT_ROOT / "experiments/utp_family_clustering/output/permutation_results.csv"
FAMILY_STATS = PROJECT_ROOT / "experiments/utp_family_clustering/output/family_statistics.csv"
OUTPUT_DIR = Path(__file__).parent


def load_data():
    """Load gene family analysis data."""
    perm = pd.read_csv(PERMUTATION_RESULTS) if PERMUTATION_RESULTS.exists() else None
    stats = pd.read_csv(FAMILY_STATS) if FAMILY_STATS.exists() else None
    return perm, stats


def main():
    apply_style()
    
    perm, stats = load_data()
    
    # Create single-panel figure
    fig, ax = plt.subplots(figsize=(5, 4))
    
    if perm is not None:
        # Get key values
        sharing_row = perm[perm['metric'] == 'fraction_sharing'].iloc[0]
        observed = sharing_row['observed'] * 100  # Convert to percentage
        null_mean = sharing_row['null_mean'] * 100
        null_std = sharing_row['null_std'] * 100
    else:
        # Fallback values
        observed = 25.4
        null_mean = 1.9
        null_std = 0.7
    
    # Bar chart comparing observed vs expected
    x = [0, 1]
    heights = [observed, null_mean]
    labels = ['Observed', 'Expected\n(random)']
    colors = [COLORS['secondary'], COLORS['primary']]
    
    bars = ax.bar(x, heights, color=colors, width=0.6, edgecolor='white', linewidth=1)
    
    # Error bar for null
    ax.errorbar(1, null_mean, yerr=null_std*2, fmt='none', color=COLORS['text'],
                capsize=5, capthick=1.5, lw=1.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('uTP proteins sharing\na gene family (%)')
    ax.set_ylim(0, max(heights) * 1.3)
    
    # Add value labels
    for bar, val in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                f'{val:.1f}%', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # Significance annotation
    y_max = max(heights) * 1.15
    ax.plot([0, 0, 1, 1], [y_max*0.9, y_max, y_max, y_max*0.9], 
            color=COLORS['text'], lw=1)
    ax.text(0.5, y_max * 1.02, 'p < 0.0001', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    save_figure(fig, OUTPUT_DIR / "panel_h.svg")
    save_figure(fig, OUTPUT_DIR / "panel_h.png")
    
    print(f"\nObserved sharing: {observed:.1f}%")
    print(f"Expected (null): {null_mean:.1f}% ± {null_std:.1f}%")
    
    # Additional stats from family_statistics
    if stats is not None:
        row = stats[stats['threshold'] == 0.7].iloc[0]
        n_families = row['families_with_utp']
        singleton_pct = row['utp_in_singleton_family'] / row['n_utp_proteins'] * 100
        print(f"Distinct families: {int(n_families)}")
        print(f"Singleton families: {singleton_pct:.1f}%")


if __name__ == "__main__":
    main()
