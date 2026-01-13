# uTP Motif Coverage Analysis

**Extending motif detection from experimentally-validated to HMM-predicted proteins.**

## Background

Two independent approaches identify uTP proteins:

1. **Experimental (Proteomics)**: 368 proteins enriched inside UCYN-A (Coale et al.)

   - MEME motif discovery was run on 206 Gblocks-filtered sequences
   - Identified 10 conserved motifs

2. **Computational (HMM)**: 933 proteins with HMM hits in full proteome
   - Uses the uTP HMM profile derived from experimental set

This experiment uses MAST to scan for the known motifs in all 933 HMM-predicted proteins.

## Results

### MAST Scanning Summary

| Metric                           | Count | Percentage    |
| -------------------------------- | ----- | ------------- |
| Total HMM-predicted proteins     | 933   | 100%          |
| Proteins with motif hits         | 745   | 79.8%         |
| Valid terminal pattern (4/5/7/9) | 607   | 81.5% of hits |
| Starts with motif 2→1            | 444   | 59.6% of hits |

### Comparison: Experimental vs HMM-only

| Metric                 | Experimental (n=195) | HMM-only (n=550) |
| ---------------------- | -------------------- | ---------------- |
| Valid terminal pattern | 166 (85.1%)          | 441 (80.2%)      |
| Other terminal         | 29 (14.9%)           | 109 (19.8%)      |

The HMM-only set shows similar validity rates to the experimental set.

### Terminal Motif Distribution

| Terminal   | Experimental | HMM-only | Total |
| ---------- | ------------ | -------- | ----- |
| terminal_7 | 131          | 329      | 460   |
| terminal_5 | 15           | 59       | 74    |
| terminal_4 | 20           | 48       | 68    |
| terminal_9 | 0            | 5        | 5     |

Terminal_7 (motif PPJPRLLP) dominates in both sets.

### Top Motif Patterns (n=745)

| Pattern     | Count | Terminal   |
| ----------- | ----- | ---------- |
| 2+1+3+5+7   | 175   | terminal_7 |
| 2+1+3+7     | 48    | terminal_7 |
| 2+1+10+5+7  | 31    | terminal_7 |
| 2+1+3+5     | 26    | terminal_5 |
| 2+1+8+3+5+7 | 25    | terminal_7 |
| 2+1+5+7     | 24    | terminal_7 |
| 2+1+3+4     | 22    | terminal_4 |
| 2+1         | 18    | other      |
| 2+1+3+5+4   | 18    | terminal_4 |

### Proteins Without Motif Hits

188 proteins (20.2%) in Import_candidates.fasta had no motif hits detected by MAST. These may represent:

- False positive HMM hits
- Divergent uTP sequences not matching known motifs
- Truncated sequences

## Comparison with Original MEME Analysis

| Metric                | Original MEME (206) | MAST Extended (745) |
| --------------------- | ------------------- | ------------------- |
| Most common terminal  | terminal_4          | terminal_7          |
| Top pattern           | 2+1+3+4 (21.8%)     | 2+1+3+5+7 (23.5%)   |
| Patterns with motif_8 | ~3%                 | ~15%                |

The extended MAST analysis detects more motif_8 occurrences, suggesting MAST's p-value threshold may be more permissive than MEME's scanned_sites.

## Generated Outputs

```
output/
├── mast_results/           # Raw MAST output
│   ├── mast.xml           # XML results
│   ├── mast.txt           # Text results
│   └── mast.html          # HTML report
├── motif_patterns.csv     # Pattern assignments for all proteins
├── pattern_distribution.svg/png  # Pattern analysis visualization
└── source_comparison.svg/png     # Experimental vs HMM comparison
```

## Usage

```bash
cd /path/to/nitroplast
uv run python experiments/utp_motif_coverage/analyze_motif_coverage.py
```

Requires MAST from MEME Suite: `/opt/local/bin/mast`

## Key Findings

1. **80% of HMM-predicted proteins have motif patterns** detectable by MAST
2. **81.5% have valid terminal motifs** (4/5/7/9), similar to experimental set
3. **terminal_7 dominates** in the extended set (61.7%)
4. **188 proteins (20%) have no motif hits** despite HMM detection

---

_Last updated: 2026-01-12_
