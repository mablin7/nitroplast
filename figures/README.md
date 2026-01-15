# Figure Generation

Publication-quality figures for the uTP manuscript. Each panel is a **single focused plot**.

## Structure

```
figures/
├── style.py          # Shared color palette and matplotlib styling
├── panel_a/          # [Manual] Schematic + 3D structure (Inkscape/PyMOL)
├── panel_b/          # Positional variance (structural conservation)
├── panel_c/          # UMAP (continuous sequence distribution)
├── panel_d/          # ROC curve (classifier performance)
├── panel_e/          # Effect sizes (biophysical properties)
├── panel_f/          # Forest plot (within-category effects)
├── panel_g/          # Variance partitioning
└── panel_h/          # Gene family distribution (ancestry)
```

## Panel Descriptions

| Panel | Content | Key Message |
|-------|---------|-------------|
| A | Schematic + 3D view | uTP structure: anchors form three-helix bundle |
| B | Positional variance plot | Structural core is highly conserved (<1 Å) |
| C | UMAP scatter | Sequences vary continuously, no discrete clusters |
| D | ROC curve | Mature domains predict uTP (AUC=0.92) |
| E | Effect size bars | Disorder, acidity, stability distinguish uTP |
| F | Forest plot | Properties persist within functional categories |
| G | Variance pie chart | uTP explains more variance than function |
| H | Observed vs expected bar | Shared ancestry contributes but doesn't explain |

## Usage

Generate individual panels:

```bash
cd /path/to/nitroplast

# Panel B - Structural conservation
uv run python figures/panel_b/generate_figure.py

# Panel C - UMAP
uv run python figures/panel_c/generate_figure.py

# Panel D - ROC
uv run python figures/panel_d/generate_figure.py

# Panel E - Effect sizes
uv run python figures/panel_e/generate_figure.py

# Panel F - Within-category
uv run python figures/panel_f/generate_figure.py

# Panel G - Variance partitioning
uv run python figures/panel_g/generate_figure.py

# Panel H - Gene families
uv run python figures/panel_h/generate_figure.py
```

Or generate all (except A):

```bash
for panel in b c d e f g h; do
    uv run python figures/panel_${panel}/generate_figure.py
done
```

## Output

Each script generates:
- `panel_X.svg` - Vector format for publication
- `panel_X.png` - Raster format for preview (300 DPI)

## Data Sources

| Panel | Experiment Directory |
|-------|---------------------|
| B | `utp_consensus_structure/` |
| C | `utp_sequence_clustering/` |
| D | `utp_presence_classifier/` |
| E | `utp_presence_classifier/` |
| F | `utp_functional_annotation/` |
| G | `utp_functional_annotation/` |
| H | `utp_family_clustering/` |

## Style Guide

### Color Palette

- **Teal** (`#048A81`): Primary data color
- **Orange** (`#E85D04`): Emphasis/comparison
- **Dark blue-gray** (`#2E4057`): Secondary data

### Design Principles

- One plot per panel
- No text boxes or annotations cluttering the plot
- Clean axis labels
- Subtle reference lines where needed

---

_Last updated: 2026-01-14_
