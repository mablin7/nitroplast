#!/usr/bin/env python3
"""
Panel F: Within-category effect sizes for intrinsic disorder.

Forest plot showing that the biophysical signature persists within
each functional category.

Data source:
- experiments/utp_functional_annotation/output/within_category_results.csv
- experiments/utp_functional_annotation/output/meta_analysis_results.csv
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
WITHIN_CATEGORY = PROJECT_ROOT / "experiments/utp_functional_annotation/output/within_category_results.csv"
META_ANALYSIS = PROJECT_ROOT / "experiments/utp_functional_annotation/output/meta_analysis_results.csv"
OUTPUT_DIR = Path(__file__).parent

# COG category labels
COG_LABELS = {
    'C': 'Energy production',
    'E': 'Amino acid metabolism',
    'G': 'Carbohydrate metabolism',
    'J': 'Translation',
    'K': 'Transcription',
    'L': 'Replication/repair',
    'O': 'PTM/protein turnover',
    'S': 'Function unknown'
}


def load_data():
    """Load within-category analysis data."""
    within = pd.read_csv(WITHIN_CATEGORY) if WITHIN_CATEGORY.exists() else None
    meta = pd.read_csv(META_ANALYSIS) if META_ANALYSIS.exists() else None
    return within, meta


def main():
    apply_style()
    
    within, meta = load_data()
    
    # Create single-panel figure
    fig, ax = plt.subplots(figsize=(6, 5))
    
    if within is not None:
        # Filter to fraction_coil (intrinsic disorder)
        data = within[within['property'] == 'fraction_coil'].copy()
        data = data.sort_values('cohens_d', ascending=True)
        
        # Get overall effect from meta-analysis
        if meta is not None:
            meta_row = meta[meta['property'] == 'fraction_coil'].iloc[0]
            overall_d = meta_row['overall_d']
        else:
            overall_d = 0.96
    else:
        # Fallback data
        data = None
        overall_d = 0.96
    
    if data is not None and len(data) > 0:
        y_positions = np.arange(len(data))
        
        # Plot individual category effects as points
        for i, (_, row) in enumerate(data.iterrows()):
            d = row['cohens_d']
            sig = row['significant']
            
            # Color and size based on significance
            color = COLORS['secondary'] if sig else COLORS['primary']
            alpha = 0.9 if sig else 0.4
            size = 100 if sig else 60
            
            ax.scatter(d, y_positions[i], color=color, s=size, alpha=alpha, 
                      zorder=3, edgecolor='white', linewidth=0.5)
        
        # Overall effect line
        ax.axvline(x=overall_d, color=COLORS['accent'], lw=2, linestyle='--', 
                   zorder=2, alpha=0.8, label=f'Overall (d={overall_d:.2f})')
        
        # Zero line
        ax.axvline(x=0, color=COLORS['text'], lw=1, alpha=0.3)
        
        # Legend entries for significance
        ax.scatter([], [], color=COLORS['secondary'], s=100, alpha=0.9, 
                   edgecolor='white', label='Significant (p<0.05)')
        ax.scatter([], [], color=COLORS['primary'], s=60, alpha=0.4,
                   edgecolor='white', label='Not significant')
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
        
        # Y-axis labels
        ax.set_yticks(y_positions)
        labels = [f"{row['COG']}: {COG_LABELS.get(row['COG'], '')[:18]}" 
                  for _, row in data.iterrows()]
        ax.set_yticklabels(labels, fontsize=8)
        
        ax.set_xlim(-0.5, 2.0)
    else:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, alpha=0.5)
    
    ax.set_xlabel("Effect size (Cohen's d)")
    
    plt.tight_layout()
    
    # Save
    save_figure(fig, OUTPUT_DIR / "panel_f.svg")
    save_figure(fig, OUTPUT_DIR / "panel_f.png")
    
    if data is not None:
        n_sig = data['significant'].sum()
        print(f"\nSignificant categories: {n_sig}/{len(data)}")
        print(f"Overall effect: d = {overall_d:.2f}")


if __name__ == "__main__":
    main()
