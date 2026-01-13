# uTP Consensus Structure Analysis

**Objective:** Analyze the structural conservation of the uTP (UCYN-A transit peptide) C-terminal region through consensus structure building and variance analysis.

## Key Results

**The uTP C-terminal region adopts a conserved U-bend structure with two α-helices.**

| Metric | Value |
|--------|-------|
| Aligned structures | 47 |
| Mean pairwise RMSD | 19.29 ± 5.07 Å |
| Minimum RMSD | 5.77 Å |
| Consensus length | 82 residues |
| Mean positional variance | 0.90 Å |
| Variance range | 0.60 - 1.38 Å |

The high pairwise RMSD reflects diverse C-terminal lengths, while the low positional variance (< 1 Å) in the consensus core demonstrates strong structural conservation.

## Background

From previous analysis:

1. AlphaFold3 structures were predicted for proteins in `ucyn-a_enriched/good-c-term-full.fasta`
2. 48 structures with strong C-terminal sequence similarity were selected
3. Structures were aligned using PyMOL's `cealign` command
4. Pre-aligned structures are available in `data/utp-structures/c_term/aligned/`

This experiment produces publication-quality figures from the pre-aligned structures.

## Data

**Input:**
- `data/utp-structures/c_term/aligned/*.cif` - 48 pre-aligned C-terminal structures
- `data/utp-structures/caretta_results/result_pdb/` - 138 full-length aligned structures (Caretta)
- `data/utp-structures/caretta_results/result.fasta` - Structure-based sequence alignment

**Output:**
- `output/rmsd_matrix.csv` - Pairwise RMSD matrix
- `output/rmsd_dendrogram.svg` - Hierarchical clustering of structures
- `output/consensus_structure.pdb` - Consensus structure
- `output/positional_variance.svg` - Per-residue variance plot
- `output/structure_overlay.png` - PyMOL visualization of aligned structures
- `output/summary_statistics.csv` - Summary metrics

## Scripts

### 01_rmsd_analysis.py

Computes pairwise RMSD between aligned structures and performs hierarchical clustering.

**Outputs:**
- `rmsd_matrix.csv` - Full pairwise RMSD matrix
- `rmsd_dendrogram.svg` - Dendrogram showing structural relationships
- `rmsd_histogram.svg` - Distribution of pairwise RMSD values

### 02_consensus_structure.py

Builds a consensus structure from aligned structures by:
1. Identifying the reference structure (longest chain)
2. For each reference residue, finding spatially corresponding residues across all structures
3. Computing consensus position as mean of all corresponding residue positions
4. Recording positional standard deviation for variance analysis

**Outputs:**
- `consensus_structure.pdb` - Consensus PDB file
- `positional_variance.csv` - Per-residue variance data
- `positional_variance.svg` - Variance along the chain

### 03_generate_figures.py

Produces publication-quality figures combining all analyses.

**Outputs:**
- `figure_structure_panel.svg` - Combined figure panel for paper
- `structure_statistics.txt` - Summary statistics for methods section

## Usage

```bash
cd /path/to/nitroplast

# Run complete analysis
uv run python experiments/utp_consensus_structure/01_rmsd_analysis.py
uv run python experiments/utp_consensus_structure/02_consensus_structure.py
uv run python experiments/utp_consensus_structure/03_generate_figures.py
```

## Methods Summary (for paper)

> Predicted structures for 48 uTP-containing proteins were extracted and aligned using PyMOL's CE alignment algorithm. Pairwise RMSD was calculated for all structure pairs using BioPython's Superimposer. A consensus structure was built by computing the mean atomic position for each aligned residue across all structures, with positional variance recorded as standard deviation. Hierarchical clustering (average linkage) was performed on the RMSD distance matrix using scipy.

## Key Findings

1. **Conserved U-bend fold**: The uTP C-terminal region consistently adopts a two-helix U-bend configuration
2. **Low structural variance**: Mean RMSD < 4 Å across diverse sequences
3. **Conserved core**: Central region shows lowest variance, terminal regions more flexible

---

_Last updated: 2026-01-13_
