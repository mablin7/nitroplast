# Haptophyte 18S rRNA Phylogeny Analysis

**Date:** January 11, 2026  
**Objective:** Establish phylogenetic relationships and estimate divergence times among available haptophyte genomes relative to _Braarudosphaera bigelowii_ (the nitroplast host)

---

## Background

This analysis aims to provide evolutionary context for the uTP (UCYN-A Transit Peptide) homolog search in haptophyte genomes. Understanding the phylogenetic relationships and divergence times helps:

1. Prioritize which genomes to search for uTP precursors/homologs
2. Interpret negative results (absence of homologs in highly divergent species is less informative)
3. Identify the closest available relatives of _B. bigelowii_ for more sensitive comparisons

---

## Methods

### Data Sources

**Reference sequence:**

- _Braarudosphaera bigelowii_ 18S rRNA (AB250785.1, 1741 bp)

**Query genomes:**

- 17 haptophyte genome assemblies downloaded from NCBI (see `data/haptophytes/ncbi_dataset/data/`)
- 18S rRNA sequences fetched from NCBI for each species via taxonomy ID

**Additional reference species** (added for phylogenetic context):

- _Gephyrocapsa oceanica_ - close relative of _E. huxleyi_
- _Calcidiscus leptoporus_ - coccolithophore
- _Pavlova lutheri_ - Pavlovophyceae
- _Phaeocystis globosa_ - Prymnesiophyceae
- _Prymnesium polylepis_ - toxic bloom species

### Analysis Pipeline

1. **Sequence retrieval:** 18S rRNA sequences fetched from NCBI using Entrez API (taxonomy ID-based search)
2. **Multiple sequence alignment:** MUSCLE v5.3
3. **Phylogenetic tree construction:**
   - Neighbor-Joining (NJ) algorithm
   - UPGMA algorithm (for comparison)
   - Distance metric: sequence identity
4. **Divergence time estimation:** Molecular clock with 18S substitution rate of ~0.75% per 100 Myr

### Molecular Clock Assumptions

The divergence time estimates use a simple strict molecular clock with:

- **Substitution rate:** 0.75% per 100 million years (0.0075 substitutions/site/100 Myr)
- **Formula:** Time (Myr) = genetic_distance / (2 × substitution_rate)

**Important caveats:**

- This is a rough estimate without fossil calibration
- Rate variation across lineages is not accounted for
- Proper dating would require relaxed clock models (e.g., BEAST) and calibration points
- Estimates should be treated as order-of-magnitude approximations

---

## Results

### Sequences Analyzed

| Species                       | 18S Length (bp) | Genetic Distance to B. bigelowii |
| ----------------------------- | --------------- | -------------------------------- |
| _Braarudosphaera bigelowii_   | 1741            | — (reference)                    |
| _Gephyrocapsa oceanica_       | 1755            | 0.050                            |
| _Calcidiscus leptoporus_      | 1719            | 0.064                            |
| _Pavlova lutheri_             | 1755            | 0.067                            |
| _Prymnesium parvum_           | 1789            | 0.069                            |
| _Prymnesium polylepis_        | 1800            | 0.075                            |
| _Isochrysis galbana_          | 1800            | 0.086                            |
| _Chrysochromulina parva_      | 1840            | 0.120                            |
| _Chrysochromulina tobinii_    | 1634            | 0.121                            |
| _Diacronema lutheri_          | 1725            | 0.137                            |
| _Pavlovales sp. CCMP2436_     | 1734            | 0.138                            |
| _Phaeocystis globosa_         | 1721            | 0.191                            |
| _uncultured Chrysochromulina_ | 1087            | 0.391                            |

### Estimated Divergence Times

| Species                  | Distance | Est. Divergence (Myr) |
| ------------------------ | -------- | --------------------- |
| _Gephyrocapsa oceanica_  | 0.050    | ~330                  |
| _Calcidiscus leptoporus_ | 0.064    | ~430                  |
| _Pavlova lutheri_        | 0.067    | ~450                  |
| _Prymnesium parvum_      | 0.069    | ~460                  |
| _Prymnesium polylepis_   | 0.075    | ~500                  |
| _Isochrysis galbana_     | 0.086    | ~580                  |
| _Chrysochromulina_ spp.  | 0.12     | ~800                  |
| _Diacronema/Pavlovales_  | 0.14     | ~910                  |
| _Phaeocystis globosa_    | 0.19     | ~1280                 |

### Key Findings

1. **Closest relatives to _B. bigelowii_:**

   - _Gephyrocapsa oceanica_ (~5% divergence, ~330 Myr)
   - _Calcidiscus leptoporus_ (~6.4% divergence, ~430 Myr)
   - Both are coccolithophores, confirming _B. bigelowii_'s position in this group

2. **Moderate divergence (~500-600 Myr):**

   - _Prymnesium_ spp., _Pavlova lutheri_, _Isochrysis galbana_
   - Still reasonable targets for uTP homolog searches

3. **High divergence (>800 Myr):**

   - _Chrysochromulina_, _Diacronema_, _Phaeocystis_
   - Negative results in these genomes are less informative

4. **Phylogenetic structure:**
   - Clear separation of Pavlovophyceae (_Diacronema_, _Pavlovales sp._)
   - Coccolithophores cluster together
   - _Prymnesium_ species are closely related (distance 0.009)

---

## Interpretation for uTP Research

### Genome Prioritization for uTP Searches

**High priority (closest relatives):**

1. _Gephyrocapsa oceanica_ / _Emiliania huxleyi_ - Most closely related with genome annotation
2. _Calcidiscus leptoporus_ - Second closest coccolithophore

**Medium priority:** 3. _Prymnesium parvum_ - Good genome assembly, moderate divergence 4. _Isochrysis galbana_ - Well-studied model organism

**Lower priority (high divergence):** 5. _Chrysochromulina_ spp. - Substantial divergence (~800 Myr) 6. _Phaeocystis_ - Very divergent

### Caveats for Interpretation

1. **Divergence times are approximate:** Without fossil calibration, these estimates have substantial uncertainty (likely ±50% or more)

2. **No close _Braarudosphaera_ relatives available:** The closest genome (_Gephyrocapsa_) is still ~330 Myr divergent, which is substantial for detecting homology of a novel signal peptide

3. **18S may not reflect protein evolution rates:** 18S rRNA is highly conserved; proteins involved in organelle targeting may evolve faster or slower

4. **The ~90-100 Myr nitroplast origin** (Coale et al., 2024) means the uTP system evolved much more recently than the divergence from these relatives. Precursor sequences (if they existed) may not be recognizable.

---

## Output Files

```
output/
├── 18s_sequences_unaligned.fasta    # Raw 18S sequences
├── 18s_sequences_aligned.fasta      # MUSCLE alignment
├── phylogeny_nj.nwk                 # Neighbor-joining tree (Newick)
├── phylogeny_upgma.nwk              # UPGMA tree (Newick)
├── phylogeny_nj.png/svg             # NJ tree figure
├── phylogeny_upgma.png/svg          # UPGMA tree figure
├── divergence_times.csv             # Calculated divergence times
├── divergence_times.png/svg         # Divergence time bar plot
└── distance_matrix.csv              # Full pairwise distance matrix
```

---

## Next Steps

1. **Search for uTP homologs** in priority genomes using:

   - HMM searches with the Coale et al. (2024) uTP model
   - Individual motif searches
   - Structural homology detection

2. **Compare C-terminal extensions** in orthologous proteins between _B. bigelowii_ and close relatives

3. **Consider adding more genomes:**
   - Check for additional _Braarudosphaera_ species sequences
   - Search for unpublished haptophyte genomes

---

## Reproducibility

To re-run this analysis:

```bash
cd /path/to/nitroplast
uv run python experiments/haptophyte_phylogeny/haptophyte_18s_phylogeny.py
```

**Requirements:**

- Python 3.12+
- Biopython
- pandas, matplotlib
- MUSCLE v5.3 (binary in project root)
- Network access for NCBI queries

---

## References

- Takano et al. (2006) - _B. bigelowii_ 18S rRNA (AB250785)
- NCBI GenBank for additional 18S sequences
- MUSCLE: Edgar (2004) - doi:10.1093/nar/gkh340
- 18S molecular clock rate: various sources, ~0.5-1% per 100 Myr
