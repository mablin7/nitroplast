#!/usr/bin/env python3
"""
Analyze functional annotation correlations with uTP terminal motif variants.
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
from scipy.stats import fisher_exact, chi2_contingency

# Paths
SCRIPT_DIR = Path(__file__).parent
MEME_FILE = SCRIPT_DIR.parent / "utp_motif_analysis" / "data" / "meme_gb.xml"
ANN_FILE = SCRIPT_DIR.parent.parent / "data" / "Bbigelowii_transcriptome_annotations.csv"
OUTPUT_DIR = SCRIPT_DIR / "output" / "functional_analysis"

# COG category descriptions
COG_DESCRIPTIONS = {
    "A": "RNA processing and modification",
    "B": "Chromatin structure",
    "C": "Energy production",
    "D": "Cell cycle",
    "E": "Amino acid metabolism",
    "F": "Nucleotide metabolism",
    "G": "Carbohydrate metabolism",
    "H": "Coenzyme metabolism",
    "I": "Lipid metabolism",
    "J": "Translation",
    "K": "Transcription",
    "L": "Replication/repair",
    "M": "Cell wall/membrane",
    "N": "Cell motility",
    "O": "PTM/chaperones",
    "P": "Inorganic ion transport",
    "Q": "Secondary metabolism",
    "R": "General function",
    "S": "Unknown function",
    "T": "Signal transduction",
    "U": "Intracellular trafficking",
    "V": "Defense mechanisms",
    "W": "Extracellular structures",
    "X": "Mobilome",
    "Y": "Nuclear structure",
    "Z": "Cytoskeleton",
}


def load_annotations():
    """Load EggNOG annotations."""
    ann = pd.read_csv(ANN_FILE, comment=None, skiprows=3, header=0)
    ann.columns = [c.strip().replace("#", "") for c in ann.columns]
    return ann


def load_terminal_variants():
    """Load terminal motif variant assignments."""
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


def parse_cog_categories(cog_string):
    """Extract COG single-letter categories from EggNOG COG column."""
    if pd.isna(cog_string):
        return []
    
    # COG categories are single letters, extract them
    categories = []
    for part in str(cog_string).split(","):
        # Look for single letter categories (may be in various formats)
        for char in part:
            if char in COG_DESCRIPTIONS:
                categories.append(char)
    
    # Also check the raw string for single-letter patterns
    # The format might be like "A" or "COG1234@1,A"
    return list(set(categories))


def parse_go_terms(go_string):
    """Extract GO terms from comma-separated string."""
    if pd.isna(go_string):
        return []
    return [g.strip() for g in str(go_string).split(",") if g.strip().startswith("GO:")]


def parse_kegg_ko(kegg_string):
    """Extract KEGG KO terms."""
    if pd.isna(kegg_string):
        return []
    return [k.strip() for k in str(kegg_string).split(",") if k.strip().startswith("ko:")]


def parse_kegg_pathway(pathway_string):
    """Extract KEGG pathway IDs."""
    if pd.isna(pathway_string):
        return []
    pathways = []
    for p in str(pathway_string).split(","):
        p = p.strip()
        if p.startswith("ko") or p.startswith("map"):
            pathways.append(p)
    return pathways


def enrichment_analysis(term_by_variant, variant_counts, term_name="term"):
    """
    Perform enrichment analysis using Fisher's exact test.
    
    Returns DataFrame with enrichment results.
    """
    results = []
    all_terms = set()
    for terms in term_by_variant.values():
        all_terms.update(terms)
    
    total_proteins = sum(variant_counts.values())
    variants = list(variant_counts.keys())
    
    for term in all_terms:
        for variant in variants:
            # Count proteins with term in this variant
            in_variant_with_term = sum(
                1 for name, terms in term_by_variant.items()
                if terms and terminal_variants.get(name) == variant and term in terms
            )
            in_variant_total = variant_counts[variant]
            
            # Count proteins with term in other variants
            other_with_term = sum(
                1 for name, terms in term_by_variant.items()
                if terms and terminal_variants.get(name) != variant and term in terms
            )
            other_total = total_proteins - in_variant_total
            
            # 2x2 contingency table
            # [[with_term_in_variant, without_term_in_variant],
            #  [with_term_other, without_term_other]]
            table = [
                [in_variant_with_term, in_variant_total - in_variant_with_term],
                [other_with_term, other_total - other_with_term],
            ]
            
            if in_variant_with_term == 0 and other_with_term == 0:
                continue
            
            try:
                odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
            except:
                continue
            
            results.append({
                "term": term,
                "variant": variant,
                "count_in_variant": in_variant_with_term,
                "total_in_variant": in_variant_total,
                "count_other": other_with_term,
                "total_other": other_total,
                "odds_ratio": odds_ratio,
                "p_value": p_value,
            })
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # FDR correction
    from scipy.stats import false_discovery_control
    if len(df) > 0:
        df["q_value"] = false_discovery_control(df["p_value"], method="bh")
    
    return df.sort_values("p_value")


def plot_cog_distribution(cog_by_variant, variant_counts, output_file):
    """Plot COG category distribution by variant."""
    # Count COG categories per variant
    cog_counts = defaultdict(lambda: defaultdict(int))
    for name, cogs in cog_by_variant.items():
        variant = terminal_variants.get(name)
        if variant and cogs:
            for cog in cogs:
                cog_counts[variant][cog] += 1
    
    # Create matrix
    variants = sorted(variant_counts.keys())
    cogs = sorted(COG_DESCRIPTIONS.keys())
    
    matrix = np.zeros((len(cogs), len(variants)))
    for i, cog in enumerate(cogs):
        for j, variant in enumerate(variants):
            count = cog_counts[variant].get(cog, 0)
            total = variant_counts[variant]
            matrix[i, j] = count / total * 100 if total > 0 else 0
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 12))
    
    sns.heatmap(
        matrix,
        xticklabels=[v.replace("terminal_", "T") for v in variants],
        yticklabels=[f"{c}: {COG_DESCRIPTIONS[c][:20]}" for c in cogs],
        cmap="YlOrRd",
        annot=True,
        fmt=".1f",
        ax=ax,
    )
    
    ax.set_xlabel("Terminal Motif")
    ax.set_ylabel("COG Category")
    ax.set_title("COG Category Distribution by Terminal Motif (%)")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved COG distribution to {output_file}")


def plot_kegg_distribution(kegg_by_variant, variant_counts, output_file):
    """Plot KEGG pathway distribution by variant."""
    # Count KEGG pathways per variant
    pathway_counts = defaultdict(lambda: defaultdict(int))
    for name, pathways in kegg_by_variant.items():
        variant = terminal_variants.get(name)
        if variant and pathways:
            for pathway in pathways:
                pathway_counts[variant][pathway] += 1
    
    # Get top pathways
    all_pathways = Counter()
    for v_pathways in pathway_counts.values():
        all_pathways.update(v_pathways)
    
    top_pathways = [p for p, _ in all_pathways.most_common(15)]
    
    if not top_pathways:
        print("No KEGG pathways found")
        return
    
    variants = sorted(variant_counts.keys())
    
    # Create matrix
    matrix = np.zeros((len(top_pathways), len(variants)))
    for i, pathway in enumerate(top_pathways):
        for j, variant in enumerate(variants):
            count = pathway_counts[variant].get(pathway, 0)
            total = variant_counts[variant]
            matrix[i, j] = count / total * 100 if total > 0 else 0
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        matrix,
        xticklabels=[v.replace("terminal_", "T") for v in variants],
        yticklabels=top_pathways,
        cmap="YlGnBu",
        annot=True,
        fmt=".1f",
        ax=ax,
    )
    
    ax.set_xlabel("Terminal Motif")
    ax.set_ylabel("KEGG Pathway")
    ax.set_title("Top KEGG Pathways by Terminal Motif (%)")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved KEGG distribution to {output_file}")


def chi_square_test(term_by_variant, variant_counts, term_type="COG"):
    """Perform chi-square test for overall association."""
    # Get all unique terms
    all_terms = set()
    for terms in term_by_variant.values():
        if terms:
            all_terms.update(terms)
    
    if len(all_terms) == 0:
        return None
    
    # Create contingency table: variants x terms
    variants = sorted(variant_counts.keys())
    terms = sorted(all_terms)
    
    # Count matrix
    matrix = np.zeros((len(variants), len(terms)))
    for name, protein_terms in term_by_variant.items():
        variant = terminal_variants.get(name)
        if variant and protein_terms:
            v_idx = variants.index(variant)
            for term in protein_terms:
                if term in terms:
                    t_idx = terms.index(term)
                    matrix[v_idx, t_idx] += 1
    
    # Remove columns with all zeros
    col_sums = matrix.sum(axis=0)
    non_zero_cols = col_sums > 0
    matrix = matrix[:, non_zero_cols]
    terms = [t for t, nz in zip(terms, non_zero_cols) if nz]
    
    if matrix.shape[1] < 2:
        return None
    
    try:
        chi2, p_value, dof, expected = chi2_contingency(matrix)
        return {
            "term_type": term_type,
            "chi2": chi2,
            "p_value": p_value,
            "dof": dof,
            "n_variants": len(variants),
            "n_terms": len(terms),
        }
    except:
        return None


# Global for enrichment function
terminal_variants = {}


def main():
    global terminal_variants
    
    print("=" * 70)
    print("Functional Annotation Correlation Analysis")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading data...")
    ann = load_annotations()
    terminal_variants = load_terminal_variants()
    
    variant_counts = Counter(terminal_variants.values())
    print(f"  Proteins: {len(terminal_variants)}")
    print(f"  Variants: {dict(variant_counts)}")
    
    # Create annotation lookup
    ann_dict = {row["query_name"]: row for _, row in ann.iterrows()}
    
    # Parse annotations for each protein
    print("\n[2/5] Parsing annotations...")
    cog_by_protein = {}
    go_by_protein = {}
    kegg_ko_by_protein = {}
    kegg_pathway_by_protein = {}
    preferred_names = {}
    
    for name in terminal_variants:
        if name in ann_dict:
            row = ann_dict[name]
            
            # COG - check multiple possible columns
            cog_str = None
            for col in ann.columns:
                if "COG" in str(row.get(col, "")):
                    cog_str = row[col]
                    break
            
            # Also check for single-letter category column
            # Looking at column index 20 which seems to have single-letter COG
            if len(ann.columns) > 20:
                cat_col = ann.columns[20]
                cat_val = row.get(cat_col, "")
                if isinstance(cat_val, str) and len(cat_val) <= 3:
                    cog_by_protein[name] = list(cat_val) if cat_val else []
                else:
                    cog_by_protein[name] = parse_cog_categories(cog_str)
            else:
                cog_by_protein[name] = parse_cog_categories(cog_str)
            
            go_by_protein[name] = parse_go_terms(row.get("GOs", ""))
            kegg_ko_by_protein[name] = parse_kegg_ko(row.get("KEGG_ko", ""))
            kegg_pathway_by_protein[name] = parse_kegg_pathway(row.get("KEGG_Pathway", ""))
            preferred_names[name] = row.get("Preferred_name", "")
    
    # Summary stats
    n_with_cog = sum(1 for cogs in cog_by_protein.values() if cogs)
    n_with_go = sum(1 for gos in go_by_protein.values() if gos)
    n_with_kegg = sum(1 for kos in kegg_ko_by_protein.values() if kos)
    n_with_pathway = sum(1 for ps in kegg_pathway_by_protein.values() if ps)
    
    print(f"  With COG: {n_with_cog}/{len(terminal_variants)} ({n_with_cog/len(terminal_variants)*100:.1f}%)")
    print(f"  With GO: {n_with_go}/{len(terminal_variants)} ({n_with_go/len(terminal_variants)*100:.1f}%)")
    print(f"  With KEGG KO: {n_with_kegg}/{len(terminal_variants)} ({n_with_kegg/len(terminal_variants)*100:.1f}%)")
    print(f"  With KEGG Pathway: {n_with_pathway}/{len(terminal_variants)} ({n_with_pathway/len(terminal_variants)*100:.1f}%)")
    
    # Chi-square tests for overall association
    print("\n[3/5] Testing overall associations (Chi-square)...")
    
    chi_results = []
    for term_type, term_dict in [
        ("COG", cog_by_protein),
        ("GO", go_by_protein),
        ("KEGG_KO", kegg_ko_by_protein),
        ("KEGG_Pathway", kegg_pathway_by_protein),
    ]:
        result = chi_square_test(term_dict, variant_counts, term_type)
        if result:
            chi_results.append(result)
            sig = "**" if result["p_value"] < 0.05 else ""
            print(f"  {term_type}: χ²={result['chi2']:.2f}, df={result['dof']}, p={result['p_value']:.4f} {sig}")
    
    # Enrichment analysis
    print("\n[4/5] Enrichment analysis (Fisher's exact)...")
    
    # COG enrichment
    cog_enrichment = enrichment_analysis(cog_by_protein, variant_counts, "COG")
    if len(cog_enrichment) > 0:
        sig_cog = cog_enrichment[cog_enrichment["q_value"] < 0.1]
        print(f"  COG: {len(sig_cog)} significant (q<0.1) out of {len(cog_enrichment)} tests")
        if len(sig_cog) > 0:
            print("  Top COG enrichments:")
            for _, row in sig_cog.head(5).iterrows():
                desc = COG_DESCRIPTIONS.get(row["term"], "Unknown")
                print(f"    {row['term']} ({desc[:25]}): {row['variant']} OR={row['odds_ratio']:.2f} q={row['q_value']:.3f}")
    
    # GO enrichment (only top GO terms to avoid multiple testing burden)
    go_counts = Counter()
    for gos in go_by_protein.values():
        go_counts.update(gos)
    top_gos = {go for go, _ in go_counts.most_common(50)}
    go_filtered = {k: [g for g in v if g in top_gos] for k, v in go_by_protein.items()}
    
    go_enrichment = enrichment_analysis(go_filtered, variant_counts, "GO")
    if len(go_enrichment) > 0:
        sig_go = go_enrichment[go_enrichment["q_value"] < 0.1]
        print(f"  GO (top 50): {len(sig_go)} significant (q<0.1) out of {len(go_enrichment)} tests")
    
    # KEGG enrichment
    kegg_enrichment = enrichment_analysis(kegg_pathway_by_protein, variant_counts, "KEGG")
    if len(kegg_enrichment) > 0:
        sig_kegg = kegg_enrichment[kegg_enrichment["q_value"] < 0.1]
        print(f"  KEGG Pathway: {len(sig_kegg)} significant (q<0.1) out of {len(kegg_enrichment)} tests")
    
    # Generate plots
    print("\n[5/5] Generating visualizations...")
    
    plot_cog_distribution(cog_by_protein, variant_counts, OUTPUT_DIR / "cog_by_terminal_motif.svg")
    plot_kegg_distribution(kegg_pathway_by_protein, variant_counts, OUTPUT_DIR / "kegg_by_terminal_motif.svg")
    
    # Summary table by variant
    print("\n" + "=" * 70)
    print("SUMMARY: Annotation Distribution by Terminal Motif")
    print("=" * 70)
    
    summary_data = []
    for variant in sorted(variant_counts.keys()):
        proteins = [n for n, v in terminal_variants.items() if v == variant]
        
        # COG category distribution
        cog_cats = []
        for p in proteins:
            if p in cog_by_protein and cog_by_protein[p]:
                cog_cats.extend(cog_by_protein[p])
        cog_dist = Counter(cog_cats)
        top_cog = cog_dist.most_common(3)
        
        # Preferred names
        names = [str(preferred_names.get(p, "")) for p in proteins 
                 if preferred_names.get(p) and not pd.isna(preferred_names.get(p))]
        
        summary_data.append({
            "variant": variant,
            "n_proteins": len(proteins),
            "n_with_cog": sum(1 for p in proteins if cog_by_protein.get(p)),
            "n_with_go": sum(1 for p in proteins if go_by_protein.get(p)),
            "top_cog": ", ".join([f"{c}({n})" for c, n in top_cog]),
            "example_names": ", ".join(names[:3]),
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save results
    summary_df.to_csv(OUTPUT_DIR / "variant_annotation_summary.csv", index=False)
    
    if len(cog_enrichment) > 0:
        cog_enrichment.to_csv(OUTPUT_DIR / "cog_enrichment_results.csv", index=False)
    if len(go_enrichment) > 0:
        go_enrichment.to_csv(OUTPUT_DIR / "go_enrichment_results.csv", index=False)
    if len(kegg_enrichment) > 0:
        kegg_enrichment.to_csv(OUTPUT_DIR / "kegg_enrichment_results.csv", index=False)
    
    # Chi-square results
    if chi_results:
        chi_df = pd.DataFrame(chi_results)
        chi_df.to_csv(OUTPUT_DIR / "chi_square_results.csv", index=False)
    
    print(f"\n📁 Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
