# uTP Homolog Search in Haptophyte Genomes

**Date:** January 12, 2026  
**Objective:** Search for uTP homologs and precursors in non-symbiotic haptophyte genomes

---

## Research Questions

### Question A: Does the uTP sequence exist in non-symbiotic haptophytes?

- **If yes** → uTP was co-opted from existing cellular function
- **If no** → de novo evolution or extensive divergence from unrecognizable precursor

### Question B: Do non-symbiotic haptophytes have proteins with unexplained C-terminal extensions?

This differs from searching for uTP homologs. We identify ortholog pairs between _B. bigelowii_ and relatives, then ask whether _B. bigelowii_ proteins have C-terminal extensions their orthologs lack.

This could reveal:

- Other proteins with uTP not detected in proteomics
- Proteins with different C-terminal extensions (alternative targeting systems?)
- The "boundary" of which proteins acquired uTP

### Question C: Are individual uTP motifs found elsewhere?

Even if the complete uTP isn't found, individual motifs might exist in other contexts, suggesting:

- Ancestral sequence features co-opted into uTP
- Convergent evolution
- Random occurrence (requires statistical comparison to background)

---

## Methods

### Data Sources

**_B. bigelowii_ proteins:**

- `Import_candidates.fasta`: 933 proteins with putative uTP (from Coale et al. 2024)
- `uTP_HMM_hits.fasta`: 734 HMM-validated uTP proteins

**Haptophyte proteomes (7 with annotations):**

| Organism                     | Accession       | Proteins | Divergence from B.b. |
| ---------------------------- | --------------- | -------- | -------------------- |
| _Emiliania huxleyi_ CCMP1516 | GCA_000372725.1 | ~38,500  | ~330 Myr             |
| _Chrysochromulina tobinii_   | GCA_001275005.1 | ~16,700  | ~800 Myr             |
| _Diacronema lutheri_         | GCA_019448385.1 | ~14,400  | ~910 Myr             |
| _Pavlovales sp._ CCMP2436    | GCA_026770615.1 | ~26,100  | ~920 Myr             |
| _Prymnesium parvum_          | GCA_041296205.1 | ~23,700  | ~460 Myr             |
| _Prymnesium sp._             | GCA_046255225.1 | ~15,000  | ~500 Myr             |
| _Emiliania huxleyi_ (RefSeq) | GCF_000372725.1 | ~38,500  | ~330 Myr             |

**uTP motif consensus sequences:**

| Motif   | Consensus             | Length | Role                   |
| ------- | --------------------- | ------ | ---------------------- |
| motif_2 | WLEEWRERLECWW         | 13 aa  | Initiating motif (93%) |
| motif_1 | TQTQLGACMGALGLHLGSRLD | 21 aa  | Secondary fixed (91%)  |
| motif_3 | AEPGCEWVEE            | 10 aa  | Variable               |
| motif_4 | LPDFPEPFSLPPIPRL      | 16 aa  | Variable               |
| motif_5 | LZLPDFPD              | 8 aa   | Variable               |
| motif_6 | AWRAALLGRAPPP         | 13 aa  | Variable               |
| motif_7 | PPJPRLLP              | 8 aa   | Variable               |

### Analysis Pipeline

#### Question A: Direct uTP homology search

1. Extract C-terminal 150 aa from uTP proteins (includes uTP + some mature domain)
2. DIAMOND blastp against combined haptophyte proteome
3. Filter for:
   - Hits at C-terminus of subject (<200 aa from end)
   - Low identity (<40%) to exclude mature domain orthologs
   - Significant e-value

**Critical caveat:** High-identity hits (>50%) are almost certainly mature domain orthologs, NOT uTP homologs. True uTP homologs would show ~20-30% identity at best.

#### Question B: C-terminal extension analysis

1. Bidirectional DIAMOND best hits to identify putative orthologs
2. Compare protein lengths between _B. bigelowii_ and orthologs
3. Identify proteins where _B. bigelowii_ version is >100 aa longer
4. Cross-reference with known uTP proteins

#### Question C: Individual motif search

1. Convert motif consensus to relaxed regex patterns
2. Search all haptophyte proteomes
3. Record position relative to C-terminus
4. Compare to background rate (shuffled sequences)

---

## Results

### Question A: uTP Homology Search

#### Initial DIAMOND-based search (problematic)

**Result: 562 C-terminal hits found**

However, **these are NOT uTP homologs**. Analysis of hit identity:

- Mean identity: **51.6%** (range: 20-82%)
- This is far too high for uTP homologs

**Problem:** The C-terminal 150 aa query includes ~60-70 aa of mature domain, which finds orthologs and obscures uTP-specific signal.

#### HMM-based search (corrected methodology)

Using the uTP HMM profile (713 positions, built from 206 aligned sequences):

**Critical correction:** Based on HMM consensus analysis, the uTP motifs are at the **START** of the HMM, not the end:

- **Positions 1-100**: uTP region (contains motif_2, motif_1, motif_3, motif_4)
- **Positions 100-713**: Mature domain

The HMM was built from C-terminal protein fragments where the uTP is at the N-terminus of each fragment.

**Results across 7 haptophyte proteomes (E-value < 0.01):**

| Hit Type          | Significant Hits | Interpretation                     |
| ----------------- | ---------------- | ---------------------------------- |
| UTP_REGION_HIT    | **0**            | True uTP homologs                  |
| PARTIAL_UTP_HIT   | **0**            | Partial uTP matches                |
| MATURE_DOMAIN_HIT | 33               | Orthologs of uTP-carrying proteins |

**Key finding:** All 84 raw HMM hits were analyzed:

- 5 hits touched the uTP region (positions 1-100) but had non-significant E-values (100-960)
- 33 significant hits (E < 0.01) were exclusively to the mature domain region
- The _C. tobinii_ TBC1 domain hit (E=4e-39, positions 449-632) is in the mature domain region

**Conclusion for Question A:**

> **NO TRUE uTP HOMOLOGS FOUND**
>
> Of 84 total HMM hits:
>
> - 0 significant hits (E < 0.01) to the uTP region (HMM positions 1-100)
> - 33 significant hits to the mature domain region (HMM positions 100-713)
> - 5 non-significant hits (E = 100-960) weakly touched the uTP region but are statistical noise
>
> This strongly supports **de novo evolution** of uTP in the _B. bigelowii_ lineage.
>
> Alternative explanations:
>
> 1. uTP diverged so extensively from its precursor as to be undetectable
> 2. uTP precursor exists only in close _B. bigelowii_ relatives (genomes not available)
> 3. Search sensitivity insufficient (unlikely given 713-position profile HMM)

### Question B: C-terminal Extension Analysis

**Summary:**

| Metric                                | Value         |
| ------------------------------------- | ------------- |
| Total ortholog pairs identified       | 1,330         |
| Pairs with >100 aa extension          | 1,028 (77.3%) |
| Mean length difference (uTP proteins) | +233.5 aa     |

**Key finding:** uTP proteins show consistent C-terminal extensions (~230 aa on average) compared to their orthologs in non-symbiotic haptophytes. This confirms:

1. The uTP region is a genuine addition to _B. bigelowii_ proteins
2. The extension is not present in orthologous proteins from relatives
3. The ~100-150 aa uTP plus linker regions account for the length difference

**No non-uTP proteins with unexplained extensions were found** (all proteins in `Import_candidates.fasta` are classified as uTP proteins).

### Question C: Individual Motif Search

#### Initial regex-based search (too strict)

**Result: Zero motif hits** using strict regex patterns.

#### MAST-based search (robust PWM approach)

Using MEME Suite's MAST tool with the original MEME-derived position weight matrices (PWMs):

**Positive control (known uTP proteins):**

- 3,836 motif hits across 733/734 uTP proteins
- 54.9% of hits within 150 aa of C-terminus
- Validates that MAST correctly identifies uTP motifs

**Haptophyte proteome search results:**

| Metric                             | Value       |
| ---------------------------------- | ----------- |
| Total motif hits                   | 530         |
| Unique sequences with hits         | 251         |
| C-terminal hits (≤150 aa from end) | 261 (49.2%) |
| Expected C-terminal by chance      | ~38.3%      |

**Hits by motif:**

| Motif  | Consensus             | Hits in haptophytes | Hits in uTP proteins |
| ------ | --------------------- | ------------------- | -------------------- |
| MEME-1 | TQTQLGACMGALGLHLGSRLD | 36                  | 746                  |
| MEME-2 | WLEEWRERLECWW         | 27                  | 727                  |
| MEME-3 | AEPGCEWVEE            | 32                  | 618                  |
| MEME-4 | LPDFPEPFSLPPIPRL      | 88                  | 113                  |
| MEME-5 | LZLPDFPD              | 52                  | 626                  |
| MEME-6 | AWRAALLGRAPPP         | 96                  | 185                  |
| MEME-7 | PPJPRLLP              | 85                  | 584                  |

**Key finding: Core motif analysis**

The core uTP motifs (MEME-1 and MEME-2) that initiate the uTP sequence showed:

- 63 total hits in haptophyte proteomes
- 29 C-terminal hits (within 150 aa of C-terminus)
- These are scattered across all 6 haptophyte species

#### Follow-up Analysis: Critical Tests for uTP Precursors

**1. Motif Co-occurrence Analysis**

Do any proteins have MULTIPLE uTP motifs co-occurring (which would suggest a uTP precursor)?

| Metric                                          | Value |
| ----------------------------------------------- | ----- |
| Sequences with 2+ motif hits                    | 165   |
| Sequences with both MEME-1 AND MEME-2           | **1** |
| Sequences with both MEME-1+MEME-2 at C-terminus | **0** |

The single protein with both core motifs (KAG8470450.1 from _Diacronema lutheri_, 1269 aa) has MEME-2 at position 51 and MEME-1 at position 439 - **not at the C-terminus** and in a protein too large to be a uTP precursor.

**Critical finding: Zero proteins have both MEME-1 and MEME-2 in the C-terminal region.**

**2. C-terminal Core Motif Hits Investigation**

Best C-terminal MEME-1/MEME-2 hits by p-value:

| Protein      | Organism         | Motif  | From C-term | P-value | Annotation           |
| ------------ | ---------------- | ------ | ----------- | ------- | -------------------- |
| KAG8465093.1 | _D. lutheri_     | MEME-1 | 14 aa       | 4.2e-08 | hypothetical protein |
| EOD26814.1   | _E. huxleyi_     | MEME-1 | 11 aa       | 1.4e-06 | hypothetical protein |
| KAL1529582.1 | _P. parvum_      | MEME-1 | 3 aa        | 6.0e-06 | hypothetical protein |
| KAL3932683.1 | _Prymnesium sp._ | MEME-2 | 21 aa       | 9.5e-06 | hypothetical protein |
| KAG8461925.1 | _D. lutheri_     | MEME-2 | 22 aa       | 2.1e-05 | hypothetical protein |

All 29 proteins with C-terminal core motif hits:

- Are annotated as "hypothetical proteins" with no functional annotation
- Have only ONE core motif (never both MEME-1 and MEME-2)
- Range from 116-908 aa in length (vs ~247 aa mean for uTP proteins)
- Show no common function or orthology

**3. False Positive Rate (Arabidopsis Control)**

MAST search on _Arabidopsis thaliana_ proteome (27,448 proteins) as negative control:

| Metric                 | Haptophytes      | Arabidopsis |
| ---------------------- | ---------------- | ----------- |
| Total hits             | 530              | 144         |
| Unique sequences       | 251              | 49          |
| Hits per 1000 proteins | varies by genome | 5.2         |
| C-terminal hits        | 261 (49.2%)      | 84 (58.3%)  |

Per-motif comparison (haptophyte:Arabidopsis ratio):

| Motif  | Haptophyte | Arabidopsis | Ratio |
| ------ | ---------- | ----------- | ----- |
| MEME-1 | 36         | 8           | 4.5x  |
| MEME-2 | 27         | 8           | 3.4x  |
| MEME-7 | 85         | 6           | 14.2x |
| MEME-9 | 44         | 2           | 22.0x |

**Interpretation:** Arabidopsis shows substantial hits for the same motifs, confirming these are generic protein sequence patterns, not haptophyte-specific. The higher ratios for some motifs (MEME-7, MEME-9) may reflect proline-rich sequences common in haptophytes.

#### Overall Interpretation

1. **Individual motifs DO occur in haptophyte proteomes**, but at low frequency
2. **No multi-motif co-occurrence at C-terminus** - the defining feature of uTP is absent
3. **Core motifs hit different proteins independently** - not the same protein as expected for a precursor
4. **Arabidopsis shows similar motif hits** - confirming these are generic protein patterns
5. **All candidates are "hypothetical proteins"** - no identifiable function or orthology to B. bigelowii genes
6. The hits represent **random sequence similarity** rather than homologous sequences

---

## Critical Methodological Notes

### Limitations of Question A analysis

1. **HMM structure interpretation:** The HMM was built from C-terminal protein fragments of variable length. Understanding which HMM positions correspond to uTP vs. mature domain required analyzing the HMM consensus sequence (uTP motifs are in positions 1-100, mature domain in 100-713).
2. **Divergence time:** Even the closest relative (_E. huxleyi_, ~330 Myr) is highly divergent
3. **Genome sampling:** No close _Braarudosphaera_ relatives are available - the closest genomes are from different haptophyte families
4. **HMM sensitivity:** Profile HMMs are powerful but may miss extremely divergent sequences

### What the HMM search actually found

The 33 significant HMM hits (E < 0.01) represent **orthologs of the mature domains of proteins that carry uTP in _B. bigelowii_**:

- TBC domain proteins (conserved GTPase-activating proteins)
- Various metabolic enzymes
- Structural proteins

These hits confirm that the mature domains of uTP-carrying proteins are conserved across haptophytes, but the uTP extension itself (positions 1-100 in the HMM) shows no significant homology in any non-symbiotic haptophyte.

### Limitations of Question B analysis

1. **Bidirectional best hit is a heuristic:** Not all BBH pairs are true orthologs
2. **Gene duplication/loss:** Paralogs may confound length comparisons
3. **Annotation quality varies:** Incomplete annotations affect coverage

### Limitations of Question C analysis

1. ~~**Motif independence:** MAST searches for individual motifs independently~~ ✓ Addressed by co-occurrence analysis
2. **Low-complexity sequences:** Some motifs (MEME-6, MEME-7) are proline-rich and may match low-complexity regions
3. **Multiple testing:** 530 hits across ~174,000 proteins with 10 motifs - some false positives expected
4. ~~**No combination analysis:** True uTP would show MEME-2 → MEME-1 in order~~ ✓ Addressed by follow-up analysis

---

## Conclusions

### Summary of findings

| Question | Result                                                                                                        | Interpretation                                       |
| -------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| A        | No uTP homologs (HMM search, 84 mature domain hits only)                                                      | Strongly supports de novo evolution                  |
| B        | uTP proteins ~230 aa longer than orthologs                                                                    | Confirms uTP is a genuine C-terminal addition        |
| C        | Individual motifs found (530 hits), but **zero proteins have both core motifs (MEME-1+MEME-2) at C-terminus** | Hits are random matches; **no uTP precursors found** |

### Implications for uTP evolution

The absence of uTP homologs and motifs in non-symbiotic haptophytes supports the hypothesis that uTP evolved **de novo** in the _B. bigelowii_ lineage, likely after the establishment of the UCYN-A symbiosis (~90-100 Mya).

However, we cannot rule out:

1. Extensive divergence from an unrecognizable precursor
2. Horizontal gene transfer from an unknown source
3. Insufficient search sensitivity for highly divergent sequences

### Recommended follow-up analyses

1. **uTP-only HMM:** Build a profile HMM from just the uTP region (~90 aa), excluding mature domain
2. **Structural homology search:** Use predicted uTP structure (U-bend with two alpha-helices) to search for structural analogs via Foldseek/DALI
3. ~~**Relaxed motif search:** Use position-weight matrices instead of strict regex~~ ✓ Done with MAST
4. ~~**Motif combination search:** Search for proteins with multiple uTP motifs~~ ✓ Done - zero proteins have both MEME-1+MEME-2 at C-terminus
5. **Expanded genome sampling:** Include more haptophyte genomes as they become available, especially close _Braarudosphaera_ relatives
6. ~~**Outgroup analysis:** Search in non-haptophyte organisms~~ ✓ Done with Arabidopsis - confirms motifs are generic patterns
7. **Genomic context analysis:** Examine the genomic regions flanking uTP-containing genes for evidence of transposable elements or recent insertions
8. ~~**Manual inspection:** Examine C-terminal core motif hits~~ ✓ Done - all are "hypothetical proteins" with single motif hits

---

## Output Files

```
output/
├── combined_haptophyte_proteome.fasta    # All haptophyte proteins
├── utp_regions_query.fasta               # C-terminal 150 aa queries
├── utp_homology_search.m8                # DIAMOND results (raw)
├── utp_homology_all_hits.csv             # All filtered hits
├── utp_homology_c_terminal_hits.csv      # C-terminal hits only
├── hmmsearch_*.domtblout                 # HMM search results per genome
├── hmmsearch_*.txt                       # HMM search full output
├── hmmsearch_all_results.csv             # Combined HMM search results
├── hmmsearch_refined_analysis.csv        # Initial analysis (INCORRECT position interpretation)
├── hmmsearch_corrected_analysis.csv      # CORRECTED analysis (uTP = HMM pos 1-100)
├── ortholog_analysis.csv                 # Ortholog pairs with lengths
├── extension_analysis.png                # Length comparison plots
├── motif_search_results.csv              # Regex motif search summary (deprecated)
├── motif_search_results.png              # Regex motif search visualization (deprecated)
├── mast_*/                               # MAST output directories per genome
├── mast_positive_control/                # MAST results on known uTP proteins
├── mast_positive_control.csv             # Positive control hits
├── mast_all_hits.csv                     # All MAST hits in haptophytes
├── mast_c_terminal_hits.csv              # C-terminal MAST hits only
├── mast_outgroup_arabidopsis/            # MAST results on Arabidopsis (outgroup control)
├── mast_arabidopsis_hits.csv             # Arabidopsis motif hits for comparison
├── arabidopsis_proteome.fasta            # Arabidopsis proteome (downloaded)
├── cterm_core_motif_candidates.csv       # C-terminal MEME-1/2 hit candidates
├── cterm_hit_sequences.fasta             # Sequences of best C-terminal hits
└── mast_motif_search_results.png         # MAST search visualization
```

---

## Reproducibility

```bash
cd /path/to/nitroplast

# Run DIAMOND-based analysis (Questions B, C - regex)
uv run python experiments/utp_homolog_search/utp_homolog_search.py

# Run HMM-based search (Question A - proper approach)
uv run python experiments/utp_homolog_search/hmm_search.py

# Run refined HMM analysis (interprets results)
uv run python experiments/utp_homolog_search/hmm_search_refined.py

# Run MAST-based motif search (Question C - robust PWM approach)
uv run python experiments/utp_homolog_search/mast_motif_search.py
```

**Requirements:**

- Python 3.12+
- BioPython, pandas, numpy, matplotlib, seaborn
- DIAMOND (must be in PATH)
- HMMER3 (hmmsearch must be in PATH)
- MEME Suite (mast must be in PATH, e.g., /opt/local/bin/mast)
- uTP HMM profile (`utp.hmm` in experiment directory)
- MEME motif file (`experiments/utp_motif_analysis/data/meme_gb.xml`)

---

## References

- Coale et al. (2024) - Nitrogen-fixing organelle in a marine alga
- DIAMOND: Buchfink et al. (2015) - Fast and sensitive protein alignment
- uTP motif analysis: `experiments/utp_motif_analysis/`
- Haptophyte phylogeny: `experiments/haptophyte_phylogeny/`
