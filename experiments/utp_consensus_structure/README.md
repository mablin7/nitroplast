# uTP Consensus Structure Analysis

**Objective:** Analyze the structural conservation of the uTP (UCYN-A transit peptide) C-terminal region through consensus structure building and variance analysis.

## Key Results

**The anchor motifs (Motif 2 → Motif 1) form the conserved U-bend helical structure.**

| Metric                            | Value       |
| --------------------------------- | ----------- |
| Aligned structures                | 47          |
| Structures with 3+ anchor helices | 46 (98%)    |
| Consensus length                  | 82 residues |
| Mean positional variance          | 0.90 Å      |

### Motif-Structure Correspondence

The two anchor motifs directly encode the structural elements of the U-bend:

| Structural Element | Position | Length | Corresponding Motif |
| ------------------ | -------- | ------ | ------------------- |
| α-helix 1          | 2-19     | 17 aa  | Motif 2 (13 aa)     |
| Turn region        | 19-22    | 3 aa   | Linker              |
| α-helix 2          | 22-30    | 8 aa   | Motif 1 N-term      |
| α-helix 3          | 32-49    | 17 aa  | Motif 1 C-term      |

**Conclusion:** Motif 2 forms the first arm of the U-bend (α1), while Motif 1 forms the second arm as a helix-turn-helix motif (α2-α3). This architecture is conserved in 98% of analyzed structures.

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

### 05_analyze_individual_structures.py

Analyzes secondary structure in individual AlphaFold structures to map anchor motifs to helical elements.

**Outputs:**

- `individual_ss_results.csv` - SS assignment for each structure
- `individual_ss_analysis.svg` - Helix position distributions

### 06_motif_helix_summary.py

Creates summary figure showing motif-helix correspondence.

**Outputs:**

- `motif_helix_summary.svg` - Publication-quality summary figure

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

1. **Anchor motifs encode the U-bend structure**: Motif 2 (13 aa) and Motif 1 (21 aa) directly form the conserved helical architecture
2. **Three-helix architecture**: The U-bend consists of α1 (Motif 2), a short turn, and α2-α3 (Motif 1 as helix-turn-helix)
3. **High conservation**: 98% of structures show this three-helix pattern in the anchor region
4. **Structural basis for recognition**: The conserved helical fold likely serves as the recognition element for import machinery

## Interpretation

The anchor motifs are not merely sequence signatures but structural determinants. The near-universal presence of the three-helix U-bend architecture suggests that:

1. **Recognition mechanism**: Import machinery likely recognizes this specific helical fold
2. **Evolutionary constraint**: The motif sequences are conserved because they encode essential structural elements
3. **Functional requirement**: Proper folding of the U-bend may be required for successful protein import

---

_Last updated: 2026-01-14_
