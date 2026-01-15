"""
Shared figure styling for the uTP manuscript.

Color Palette and matplotlib rcParams following the project style guide.
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams

# Color Palette
COLORS = {
    "primary": "#2E4057",  # Dark blue-gray
    "secondary": "#048A81",  # Teal
    "accent": "#E85D04",  # Orange
    "light": "#90BE6D",  # Green
    "highlight": "#F9C74F",  # Yellow
    "background": "#F8F9FA",  # Light gray
    "text": "#212529",  # Dark text
}

# Additional colors for multi-group plots
PALETTE = [
    COLORS["secondary"],  # Teal (primary data color)
    COLORS["accent"],  # Orange (emphasis)
    COLORS["primary"],  # Dark blue-gray
    COLORS["light"],  # Green
    COLORS["highlight"],  # Yellow
]


def apply_style():
    """Apply the standard figure style."""
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = [
        "Helvetica Neue",
        "Helvetica",
        "Arial",
        "DejaVu Sans",
    ]
    rcParams["font.size"] = 9
    rcParams["axes.linewidth"] = 1.0
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False
    rcParams["legend.frameon"] = False
    rcParams["figure.facecolor"] = "white"
    rcParams["axes.facecolor"] = "white"
    rcParams["savefig.facecolor"] = "white"
    rcParams["savefig.edgecolor"] = "white"
    rcParams["axes.labelcolor"] = COLORS["text"]
    rcParams["xtick.color"] = COLORS["text"]
    rcParams["ytick.color"] = COLORS["text"]
    rcParams["text.color"] = COLORS["text"]


def add_panel_label(ax, label, x=-0.15, y=1.05, fontsize=14):
    """Add a bold panel label (A, B, C, etc.) to an axis."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def stats_box(ax, text, x=0.95, y=0.95, ha="right", va="top"):
    """Add a statistics box to an axis."""
    bbox = dict(
        boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"
    )
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8, ha=ha, va=va, bbox=bbox)


def save_figure(fig, path, dpi=300):
    """Save figure in both SVG and PNG formats."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")
