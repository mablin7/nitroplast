# Haptophyte Ortholog Search with OrthoFinder

**Date:** January 12, 2026  
**Objective:** Identify orthologous protein groups between _Braarudosphaera bigelowii_ and related haptophyte species to enable downstream analysis of C-terminal extensions (uTP candidates)

---

## Background

This analysis builds on the phylogenetic relationships established in `experiments/haptophyte_phylogeny/`. By identifying orthologs between _B. bigelowii_ and related species, we can:

1. Compare protein lengths and identify C-terminal extensions unique to _B. bigelowii_
2. Determine which uTP-containing proteins have clear orthologs in non-symbiotic relatives
3. Investigate whether C-terminal extensions correlate with specific protein families
4. Identify potential uTP precursor sequences in related species

---

## Methods

### Species Included

Based on phylogenetic distance from _B. bigelowii_ (see `experiments/haptophyte_phylogeny/README.md`):

| Species | Priority | Est. Divergence | Proteins | Source |
|---------|----------|-----------------|----------|--------|
| _Braarudosphaera bigelowii_ | 0 (reference) | — | ~43,000 | Transcriptome (Coale et al. 2024) |
| _Emiliania huxleyi_ CCMP1516 | 1 | ~330 Myr | ~38,500 | NCBI GCA_000372725.1 |
| _Prymnesium parvum_ 12B1 | 2 | ~460 Myr | ~24,900 | NCBI GCA_041296205.1 |
| _Prymnesium sp._ SGEUK-05 | 2 | ~460 Myr | ~15,000 | NCBI GCA_046255225.1 |
| _Chrysochromulina tobinii_ CCMP291 | 3 | ~800 Myr | ~16,700 | NCBI GCA_001275005.1 |

### OrthoFinder Configuration

- **Version:** OrthoFinder 3.x (from GitHub)
- **Sequence search:** DIAMOND (more-sensitive mode)
- **MSA method:** MAFFT (auto mode)
- **Tree inference:** VeryFastTree (FastTree-compatible)
- **Clustering:** MCL

### Dependencies

Installed via Homebrew and source:

```bash
# Homebrew packages
brew install diamond mafft veryfasttree

# MCL (built from source)
git clone https://github.com/micans/mcl.git /tmp/mcl
cd /tmp/mcl && ./install-this-mcl.sh ~/local
```

---

## Running the Analysis

### 1. Prepare Proteomes

```bash
cd /path/to/nitroplast
uv run python experiments/ortholog_search/prepare_proteomes.py --validate
```

This creates cleaned FASTA files in `experiments/ortholog_search/proteomes/`:
- One file per species
- Cleaned sequence IDs (OrthoFinder-compatible)
- Removes UCYN-A proteins from _B. bigelowii_ transcriptome

### 2. Run OrthoFinder

```bash
# Check dependencies first
uv run python experiments/ortholog_search/run_orthofinder.py --check-only

# Run full analysis (takes several hours)
uv run python experiments/ortholog_search/run_orthofinder.py -t 8
```

### 3. View Results

```bash
uv run python experiments/ortholog_search/run_orthofinder.py --summarize
```

---

## Output Files

Results are stored in `proteomes/OrthoFinder/Results_<date>/`:

```
Results_<date>/
├── Orthogroups/
│   ├── Orthogroups.tsv              # Main orthogroup assignments
│   ├── Orthogroups.GeneCount.tsv    # Gene counts per species per orthogroup
│   ├── Orthogroups_SingleCopyOrthologues.txt  # 1:1 orthologs
│   └── Orthogroups_UnassignedGenes.tsv        # Species-specific genes
├── Orthologues/
│   └── <species>__v__<species>/     # Pairwise ortholog relationships
├── Comparative_Genomics_Statistics/
│   ├── Statistics_Overall.tsv       # Summary statistics
│   ├── Statistics_PerSpecies.tsv    # Per-species breakdown
│   └── OrthologuesStats_*.tsv       # Ortholog statistics
├── Species_Tree/
│   └── SpeciesTree_rooted.txt       # Inferred species tree
├── Gene_Trees/                      # Individual gene trees
└── WorkingDirectory/                # Intermediate files
```

---

## Downstream Analysis

After OrthoFinder completes, the following analyses can be performed:

### 1. C-terminal Extension Analysis

For each orthogroup containing _B. bigelowii_ proteins:
- Align sequences with MAFFT
- Identify C-terminal extensions unique to _B. bigelowii_
- Cross-reference with known uTP-containing proteins

### 2. uTP Candidate Identification

- Filter orthogroups where _B. bigelowii_ proteins are significantly longer
- Check if extensions match uTP HMM model
- Identify conserved motifs in extensions

### 3. Evolutionary Analysis

- Calculate dN/dS for orthologous pairs
- Identify rapidly evolving proteins
- Compare selection pressure on uTP vs non-uTP proteins

---

## Key Questions to Address

1. **What fraction of uTP-containing proteins have clear orthologs in related species?**
   - If high: uTP was added to existing proteins
   - If low: uTP-containing genes may be novel or highly diverged

2. **Are C-terminal extensions present in any related species?**
   - Could indicate uTP precursors
   - May reveal intermediate evolutionary states

3. **Which protein families are enriched in uTP-containing proteins?**
   - Functional bias in protein import
   - Pathway-specific targeting

4. **Is there evidence for recent gene duplication in uTP-containing genes?**
   - Could indicate mechanism of uTP spread
   - Paralog pairs with/without uTP

---

## Notes

- The _B. bigelowii_ proteome is from transcriptome assembly, so may include:
  - Multiple isoforms per gene
  - Partial transcripts
  - Assembly artifacts
  
- Divergence times are approximate (see phylogeny analysis caveats)

- OrthoFinder may take several hours to complete with 5 species and ~140,000 total proteins

---

## References

- Emms, D.M. and Kelly, S. (2019) OrthoFinder: phylogenetic orthology inference for comparative genomics. Genome Biology 20:238
- Coale et al. (2024) - Nitrogen-fixing organelle in a marine alga
- NCBI genome assemblies for haptophyte species
