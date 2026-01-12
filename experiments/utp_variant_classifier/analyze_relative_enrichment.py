#!/usr/bin/env python3
"""
Relative Enrichment Analysis for uTP Terminal Motif Variants.

This script performs a two-stage enrichment analysis:

Stage 1: uTP vs Transcriptome Background
    - Identifies which functional categories are enriched in uTP proteins
      compared to the full B. bigelowii transcriptome
    - This establishes the "baseline" enrichment for imported proteins

Stage 2: Terminal Motif Specificity (Corrected)
    - For each terminal motif, compares to OTHER uTP proteins
    - Uses hypergeometric test with uTP as the reference population
    - This identifies terminal-motif-specific enrichments BEYOND the
      general uTP enrichment pattern

Statistical Framework:
    - Fisher's exact test for 2x2 contingency tables
    - Benjamini-Hochberg FDR correction for multiple testing
    - Odds ratios with 95% confidence intervals
    - Effect size reporting (log2 fold enrichment)
"""

import warnings
warnings.filterwarnings("ignore")

from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import fisher_exact, false_discovery_control

# Paths
SCRIPT_DIR = Path(__file__).parent
MEME_FILE = SCRIPT_DIR.parent / "utp_motif_analysis" / "data" / "meme_gb.xml"
ANN_FILE = SCRIPT_DIR.parent.parent / "data" / "Bbigelowii_transcriptome_annotations.csv"
OUTPUT_DIR = SCRIPT_DIR / "output" / "relative_enrichment"

# COG category descriptions
COG_DESCRIPTIONS = {
    "A": "RNA processing and modification",
    "B": "Chromatin structure and dynamics",
    "C": "Energy production and conversion",
    "D": "Cell cycle control, cell division",
    "E": "Amino acid transport and metabolism",
    "F": "Nucleotide transport and metabolism",
    "G": "Carbohydrate transport and metabolism",
    "H": "Coenzyme transport and metabolism",
    "I": "Lipid transport and metabolism",
    "J": "Translation, ribosomal structure",
    "K": "Transcription",
    "L": "Replication, recombination and repair",
    "M": "Cell wall/membrane/envelope biogenesis",
    "N": "Cell motility",
    "O": "PTM, protein turnover, chaperones",
    "P": "Inorganic ion transport and metabolism",
    "Q": "Secondary metabolites biosynthesis",
    "R": "General function prediction only",
    "S": "Function unknown",
    "T": "Signal transduction mechanisms",
    "U": "Intracellular trafficking, secretion",
    "V": "Defense mechanisms",
    "W": "Extracellular structures",
    "X": "Mobilome: prophages, transposons",
    "Y": "Nuclear structure",
    "Z": "Cytoskeleton",
}


def load_annotations():
    """Load full transcriptome annotations."""
    ann = pd.read_csv(ANN_FILE, comment=None, skiprows=3, header=0)
    ann.columns = [c.strip().replace("#", "") for c in ann.columns]
    return ann


def load_terminal_variants():
    """Load terminal motif variant assignments for uTP proteins."""
    meme_xml = ET.parse(MEME_FILE)
    seq_names = {
        tag.attrib["id"]: tag.attrib["name"]
        for tag in meme_xml.findall(".//sequence")
    }
    
    sites = meme_xml.findall(".//scanned_sites")
    sequences_motifs = {}
    for tag in sites:
        seq_id = tag.attrib["sequence_id"]
        if seq_id in seq_names:
            motifs = sorted(
                [s.attrib for s in tag.findall("scanned_site")],
                key=lambda m: int(m["position"])
            )
            sequences_motifs[seq_names[seq_id]] = tuple(
                m["motif_id"] for m in motifs
            )
    
    terminal_variants = {}
    for name, motifs in sequences_motifs.items():
        variant = "+".join(m.replace("motif_", "") for m in motifs)
        if variant:
            parts = variant.split("+")
            terminal = parts[-1] if parts else None
            if terminal in ["4", "5", "7", "9"]:
                terminal_variants[name] = f"terminal_{terminal}"
    
    return terminal_variants


def extract_cog_category(row, ann_columns):
    """Extract single-letter COG category from annotation row."""
    # The COG category is typically in a column near the end
    # Look for single-letter categories
    for col in ann_columns[15:]:  # Check later columns
        val = row.get(col, "")
        if isinstance(val, str) and len(val) <= 3 and val.isalpha():
            # Return all single-letter COG categories
            return [c for c in val if c in COG_DESCRIPTIONS]
    return []


def fisher_test_with_ci(a, b, c, d):
    """
    Perform Fisher's exact test and compute odds ratio with 95% CI.
    
    Contingency table:
        | With term | Without term |
    ----|-----------|--------------|
    In  |     a     |      b       |
    Out |     c     |      d       |
    
    Returns: odds_ratio, ci_lower, ci_upper, p_value
    """
    table = [[a, b], [c, d]]
    odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
    
    # Compute 95% CI for log odds ratio using Woolf's method
    # Add 0.5 to avoid division by zero (Haldane-Anscombe correction)
    a_adj, b_adj, c_adj, d_adj = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    
    log_or = np.log(odds_ratio) if odds_ratio > 0 else 0
    se_log_or = np.sqrt(1/a_adj + 1/b_adj + 1/c_adj + 1/d_adj)
    
    ci_lower = np.exp(log_or - 1.96 * se_log_or)
    ci_upper = np.exp(log_or + 1.96 * se_log_or)
    
    return odds_ratio, ci_lower, ci_upper, p_value


def compute_log2_fold_enrichment(obs_frac, exp_frac):
    """Compute log2 fold enrichment with pseudocount."""
    pseudocount = 0.001
    return np.log2((obs_frac + pseudocount) / (exp_frac + pseudocount))


def stage1_utp_vs_transcriptome(utp_proteins, all_annotations, ann_columns):
    """
    Stage 1: Compare uTP proteins to full transcriptome background.
    
    For each COG category, test whether it is enriched in uTP proteins
    compared to the full B. bigelowii transcriptome.
    """
    print("\n" + "=" * 70)
    print("STAGE 1: uTP vs Transcriptome Background")
    print("=" * 70)
    
    # Extract COG categories for all proteins
    transcriptome_cogs = defaultdict(list)
    for _, row in all_annotations.iterrows():
        name = row["query_name"]
        cogs = extract_cog_category(row, ann_columns)
        for cog in cogs:
            transcriptome_cogs[cog].append(name)
    
    # Extract COG categories for uTP proteins
    utp_cogs = defaultdict(list)
    ann_dict = {row["query_name"]: row for _, row in all_annotations.iterrows()}
    
    for name in utp_proteins:
        if name in ann_dict:
            cogs = extract_cog_category(ann_dict[name], ann_columns)
            for cog in cogs:
                utp_cogs[cog].append(name)
    
    # Compute statistics
    n_utp = len(utp_proteins)
    n_transcriptome = len(all_annotations)
    n_non_utp = n_transcriptome - n_utp
    
    print(f"\nDataset sizes:")
    print(f"  uTP proteins: {n_utp}")
    print(f"  Full transcriptome: {n_transcriptome}")
    print(f"  Non-uTP proteins: {n_non_utp}")
    
    results = []
    
    for cog in sorted(COG_DESCRIPTIONS.keys()):
        # Count proteins with this COG
        utp_with = len(utp_cogs.get(cog, []))
        transcriptome_with = len(transcriptome_cogs.get(cog, []))
        non_utp_with = transcriptome_with - utp_with
        
        # Contingency table:
        # | With COG | Without COG |
        # | utp_with | n_utp - utp_with |
        # | non_utp_with | n_non_utp - non_utp_with |
        
        a = utp_with
        b = n_utp - utp_with
        c = non_utp_with
        d = n_non_utp - non_utp_with
        
        if a + c == 0:  # No proteins with this COG
            continue
        
        odds_ratio, ci_lower, ci_upper, p_value = fisher_test_with_ci(a, b, c, d)
        
        # Compute fractions
        utp_frac = utp_with / n_utp if n_utp > 0 else 0
        bg_frac = non_utp_with / n_non_utp if n_non_utp > 0 else 0
        log2_fe = compute_log2_fold_enrichment(utp_frac, bg_frac)
        
        results.append({
            "cog": cog,
            "description": COG_DESCRIPTIONS[cog],
            "utp_count": utp_with,
            "utp_total": n_utp,
            "utp_fraction": utp_frac,
            "background_count": non_utp_with,
            "background_total": n_non_utp,
            "background_fraction": bg_frac,
            "odds_ratio": odds_ratio,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "log2_fold_enrichment": log2_fe,
            "p_value": p_value,
        })
    
    df = pd.DataFrame(results)
    
    # FDR correction
    df["q_value"] = false_discovery_control(df["p_value"], method="bh")
    df = df.sort_values("p_value")
    
    # Print significant results
    sig = df[df["q_value"] < 0.1]
    print(f"\nSignificant enrichments in uTP vs background (FDR < 0.1): {len(sig)}")
    
    if len(sig) > 0:
        print("\nTop enrichments:")
        for _, row in sig.head(10).iterrows():
            direction = "↑" if row["odds_ratio"] > 1 else "↓"
            print(f"  {row['cog']}: {row['description'][:35]:<35} "
                  f"OR={row['odds_ratio']:.2f} [{row['ci_lower']:.2f}-{row['ci_upper']:.2f}] "
                  f"log2FE={row['log2_fold_enrichment']:+.2f} {direction} q={row['q_value']:.3f}")
    
    return df, utp_cogs


def stage2_terminal_vs_utp(terminal_variants, utp_cogs, n_utp):
    """
    Stage 2: Compare each terminal motif to OTHER uTP proteins.
    
    This tests for terminal-motif-specific enrichment BEYOND the
    general uTP enrichment pattern.
    
    Reference population: all uTP proteins (not transcriptome)
    """
    print("\n" + "=" * 70)
    print("STAGE 2: Terminal Motif Specificity (uTP-corrected)")
    print("=" * 70)
    
    variant_counts = Counter(terminal_variants.values())
    print(f"\nTerminal motif distribution:")
    for v, c in sorted(variant_counts.items()):
        print(f"  {v}: {c}")
    
    # Get COG assignments for each terminal class
    terminal_cogs = defaultdict(lambda: defaultdict(list))
    for name, variant in terminal_variants.items():
        for cog, proteins in utp_cogs.items():
            if name in proteins:
                terminal_cogs[variant][cog].append(name)
    
    results = []
    
    for variant in sorted(variant_counts.keys()):
        n_variant = variant_counts[variant]
        n_other = n_utp - n_variant
        
        for cog in sorted(COG_DESCRIPTIONS.keys()):
            # Count in this variant
            variant_with = len(terminal_cogs[variant].get(cog, []))
            
            # Count in other uTP proteins
            all_utp_with = len(utp_cogs.get(cog, []))
            other_with = all_utp_with - variant_with
            
            # Contingency table (variant vs other uTP):
            a = variant_with
            b = n_variant - variant_with
            c = other_with
            d = n_other - other_with
            
            if a + c == 0:  # No uTP proteins with this COG
                continue
            
            odds_ratio, ci_lower, ci_upper, p_value = fisher_test_with_ci(a, b, c, d)
            
            # Compute fractions
            variant_frac = variant_with / n_variant if n_variant > 0 else 0
            other_frac = other_with / n_other if n_other > 0 else 0
            
            # Expected fraction (from all uTP)
            utp_frac = all_utp_with / n_utp if n_utp > 0 else 0
            
            # Log2 fold enrichment vs other uTP
            log2_fe_vs_other = compute_log2_fold_enrichment(variant_frac, other_frac)
            # Log2 fold enrichment vs all uTP (expected)
            log2_fe_vs_expected = compute_log2_fold_enrichment(variant_frac, utp_frac)
            
            results.append({
                "variant": variant,
                "cog": cog,
                "description": COG_DESCRIPTIONS[cog],
                "variant_count": variant_with,
                "variant_total": n_variant,
                "variant_fraction": variant_frac,
                "other_utp_count": other_with,
                "other_utp_total": n_other,
                "other_utp_fraction": other_frac,
                "expected_fraction": utp_frac,
                "odds_ratio": odds_ratio,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "log2_fe_vs_other_utp": log2_fe_vs_other,
                "log2_fe_vs_expected": log2_fe_vs_expected,
                "p_value": p_value,
            })
    
    df = pd.DataFrame(results)
    
    # FDR correction
    df["q_value"] = false_discovery_control(df["p_value"], method="bh")
    df = df.sort_values("p_value")
    
    # Print results by variant
    print(f"\nTerminal-specific enrichments (vs other uTP, FDR < 0.1):")
    sig = df[df["q_value"] < 0.1]
    
    if len(sig) > 0:
        for variant in sorted(variant_counts.keys()):
            variant_sig = sig[sig["variant"] == variant]
            if len(variant_sig) > 0:
                print(f"\n  {variant}:")
                for _, row in variant_sig.iterrows():
                    direction = "↑" if row["odds_ratio"] > 1 else "↓"
                    print(f"    {row['cog']}: {row['description'][:30]:<30} "
                          f"OR={row['odds_ratio']:.2f} log2FE={row['log2_fe_vs_other_utp']:+.2f} {direction}")
    else:
        print("  No significant terminal-specific enrichments after FDR correction")
    
    return df


def plot_relative_enrichment(stage1_df, stage2_df, output_dir):
    """Generate visualization of relative enrichment."""
    
    # Plot 1: Stage 1 - uTP vs background
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter to COGs with any presence
    plot_df = stage1_df[stage1_df["utp_count"] > 0].copy()
    plot_df = plot_df.sort_values("log2_fold_enrichment", ascending=True)
    
    colors = ["#e74c3c" if q < 0.1 else "#95a5a6" for q in plot_df["q_value"]]
    
    y_pos = range(len(plot_df))
    ax.barh(y_pos, plot_df["log2_fold_enrichment"], color=colors)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['cog']}: {row['description'][:25]}" 
                        for _, row in plot_df.iterrows()])
    ax.set_xlabel("log2 Fold Enrichment (uTP vs Background)")
    ax.set_title("Stage 1: COG Enrichment in uTP Proteins vs Transcriptome")
    
    plt.tight_layout()
    plt.savefig(output_dir / "stage1_utp_vs_background.svg", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved Stage 1 plot to {output_dir / 'stage1_utp_vs_background.svg'}")
    
    # Plot 2: Stage 2 - Terminal-specific (heatmap of log2FE vs expected)
    pivot_df = stage2_df.pivot_table(
        index="cog", 
        columns="variant", 
        values="log2_fe_vs_expected",
        aggfunc="first"
    )
    
    # Add descriptions
    pivot_df.index = [f"{cog}: {COG_DESCRIPTIONS.get(cog, '')[:20]}" for cog in pivot_df.index]
    
    # Filter to COGs with some variation
    row_var = pivot_df.var(axis=1)
    pivot_df = pivot_df[row_var > 0.01]
    
    if len(pivot_df) > 0:
        fig, ax = plt.subplots(figsize=(10, max(8, len(pivot_df) * 0.4)))
        
        sns.heatmap(
            pivot_df,
            cmap="RdBu_r",
            center=0,
            annot=True,
            fmt=".1f",
            ax=ax,
            cbar_kws={"label": "log2 Fold Enrichment vs uTP Expected"}
        )
        
        ax.set_xlabel("Terminal Motif")
        ax.set_ylabel("COG Category")
        ax.set_title("Stage 2: Terminal-Specific Enrichment\n(log2 FE vs Expected from all uTP)")
        
        plt.tight_layout()
        plt.savefig(output_dir / "stage2_terminal_specificity.svg", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved Stage 2 plot to {output_dir / 'stage2_terminal_specificity.svg'}")


def main():
    print("=" * 70)
    print("Relative Enrichment Analysis")
    print("Controlling for uTP baseline enrichment")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1/4] Loading data...")
    all_annotations = load_annotations()
    terminal_variants = load_terminal_variants()
    ann_columns = list(all_annotations.columns)
    
    utp_proteins = set(terminal_variants.keys())
    print(f"  Transcriptome: {len(all_annotations)} proteins")
    print(f"  uTP proteins: {len(utp_proteins)}")
    
    # Stage 1: uTP vs background
    print("\n[2/4] Stage 1 analysis...")
    stage1_df, utp_cogs = stage1_utp_vs_transcriptome(
        utp_proteins, all_annotations, ann_columns
    )
    
    # Stage 2: Terminal motif specificity
    print("\n[3/4] Stage 2 analysis...")
    stage2_df = stage2_terminal_vs_utp(
        terminal_variants, utp_cogs, len(utp_proteins)
    )
    
    # Generate plots
    print("\n[4/4] Generating visualizations...")
    plot_relative_enrichment(stage1_df, stage2_df, OUTPUT_DIR)
    
    # Save results
    stage1_df.to_csv(OUTPUT_DIR / "stage1_utp_vs_background.csv", index=False)
    stage2_df.to_csv(OUTPUT_DIR / "stage2_terminal_specificity.csv", index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nStage 1 (uTP vs Background):")
    sig1 = stage1_df[stage1_df["q_value"] < 0.1]
    print(f"  Significant enrichments (FDR < 0.1): {len(sig1)}")
    if len(sig1) > 0:
        enriched = sig1[sig1["odds_ratio"] > 1]
        depleted = sig1[sig1["odds_ratio"] < 1]
        print(f"    Enriched in uTP: {len(enriched)}")
        print(f"    Depleted in uTP: {len(depleted)}")
    
    print("\nStage 2 (Terminal-Specific vs uTP baseline):")
    sig2 = stage2_df[stage2_df["q_value"] < 0.1]
    print(f"  Significant enrichments (FDR < 0.1): {len(sig2)}")
    
    if len(sig2) == 0:
        print("\n  Interpretation: No terminal-motif-specific enrichments")
        print("  were detected after correcting for uTP baseline.")
        print("  The functional differences observed in Stage 2 are")
        print("  not significant after multiple testing correction.")
    
    print(f"\n📁 Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
