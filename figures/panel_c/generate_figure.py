#!/usr/bin/env python3
"""
Panel C: UMAP visualization of uTP sequence distribution.

Shows continuous distribution of uTP sequences without discrete clusters.

Data source:
- experiments/utp_sequence_clustering/output/utp_embeddings.h5
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import h5py

from style import COLORS, apply_style, save_figure

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
EMBEDDINGS_FILE = PROJECT_ROOT / "experiments/utp_sequence_clustering/output/utp_embeddings.h5"
OUTPUT_DIR = Path(__file__).parent


def load_embeddings():
    """Load ProtT5 embeddings."""
    if EMBEDDINGS_FILE.exists():
        with h5py.File(EMBEDDINGS_FILE, 'r') as f:
            # Embeddings are stored per sequence ID
            if 'embeddings' in f:
                return f['embeddings'][:]
            else:
                # Load all individual embeddings
                embeddings = []
                for key in f.keys():
                    embeddings.append(f[key][:])
                if embeddings:
                    return np.array(embeddings)
    return None


def main():
    apply_style()
    
    embeddings = load_embeddings()
    
    # Create single-panel figure
    fig, ax = plt.subplots(figsize=(6, 5))
    
    if embeddings is not None and len(embeddings) > 0:
        # Compute UMAP
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
        
        # Plot points with subtle styling
        ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                   c=COLORS['secondary'], alpha=0.4, s=12, edgecolor='none')
        
        n_sequences = len(embeddings)
    else:
        # Placeholder if no data
        ax.text(0.5, 0.5, 'Embeddings not available', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color=COLORS['text'], alpha=0.5)
        n_sequences = 0
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    # Remove tick labels for cleaner look
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    plt.tight_layout()
    
    # Save
    save_figure(fig, OUTPUT_DIR / "panel_c.svg")
    save_figure(fig, OUTPUT_DIR / "panel_c.png")
    
    print(f"\nPlotted {n_sequences} sequences")


if __name__ == "__main__":
    main()
