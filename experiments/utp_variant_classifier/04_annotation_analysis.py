#!/usr/bin/env python3
"""
04_annotation_analysis.py - Functional Annotation Analysis by uTP Variant

This script analyzes whether different uTP variants are enriched for specific
functional categories, using annotations from EggNOG mapper.

Analysis includes:
1. COG category enrichment (broad functional categories, ~25 terms)
2. KEGG pathway enrichment (metabolic pathways)
3. GO term enrichment (molecular function, biological process)
4. Statistical testing with FDR correction

Statistical Framework:
- Fisher's exact test for enrichment (variant vs all others)
- Benjamini-Hochberg FDR correction for multiple testing
- Effect size: odds ratio with 95% CI
- Clustering analysis of annotation profiles

Output:
- statistics/cog_enrichment.csv: COG category enrichment per variant
- statistics/kegg_enrichment.csv: KEGG pathway enrichment
- statistics/go_enrichment.csv: GO term enrichment (top terms)
- statistics/annotation_summary.csv: Summary of annotation coverage
- figures/enrichment_heatmap.svg: Heatmap of significant enrichments
- figures/cog_distribution.svg: COG category distribution by variant

Usage:
    uv run python experiments/utp_variant_classifier/04_annotation_analysis.py
"""

import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# =============================================================================
# Configuration
# =============================================================================

# Statistical thresholds
ALPHA = 0.05
FDR_THRESHOLD = 0.1
MIN_ANNOTATION_COUNT = 5  # Minimum count of annotation to test
MIN_VARIANT_SIZE = 10  # Minimum variant size for enrichment analysis

# COG category descriptions
COG_CATEGORIES = {
    "J": "Translation, ribosomal structure and biogenesis",
    "A": "RNA processing and modification",
    "K": "Transcription",
    "L": "Replication, recombination and repair",
    "B": "Chromatin structure and dynamics",
    "D": "Cell cycle control, cell division, chromosome partitioning",
    "Y": "Nuclear structure",
    "V": "Defense mechanisms",
    "T": "Signal transduction mechanisms",
    "M": "Cell wall/membrane/envelope biogenesis",
    "N": "Cell motility",
    "Z": "Cytoskeleton",
    "W": "Extracellular structures",
    "U": "Intracellular trafficking, secretion, and vesicular transport",
    "O": "Posttranslational modification, protein turnover, chaperones",
    "X": "Mobilome: prophages, transposons",
    "C": "Energy production and conversion",
    "G": "Carbohydrate transport and metabolism",
    "E": "Amino acid transport and metabolism",
    "F": "Nucleotide transport and metabolism",
    "H": "Coenzyme transport and metabolism",
    "I": "Lipid transport and metabolism",
    "P": "Inorganic ion transport and metabolism",
    "Q": "Secondary metabolites biosynthesis, transport and catabolism",
    "R": "General function prediction only",
    "S": "Function unknown",
}

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
DATA_DIR = OUTPUT_DIR / "data"
STATS_DIR = OUTPUT_DIR / "statistics"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Input files
PROCESSED_PROTEINS_FILE = DATA_DIR / "processed_proteins.csv"
VARIANT_MAPPING_FILE = DATA_DIR / "variant_mapping.csv"


# =============================================================================
# Data Loading
# =============================================================================


def load_processed_proteins() -> pd.DataFrame:
    """Load processed proteins with annotations."""
    df = pd.read_csv(PROCESSED_PROTEINS_FILE)
    return df


def parse_go_terms(go_string: str) -> list[str]:
    """Parse GO terms from comma-separated string."""
    if pd.isna(go_string) or go_string == "":
        return []
    return [g.strip() for g in str(go_string).split(",") if g.strip().startswith("GO:")]


def parse_kegg_ko(kegg_string: str) -> list[str]:
    """Parse KEGG KO terms from string."""
    if pd.isna(kegg_string) or kegg_string == "":
        return []
    # Match ko:K##### pattern
    return re.findall(r"ko:K\d+", str(kegg_string))


def parse_cog_categories(cog_string: str) -> list[str]:
    """Parse COG single-letter categories."""
    if pd.isna(cog_string) or cog_string == "":
        return []
    # COG categories are single uppercase letters
    categories = []
    for char in str(cog_string):
        if char in COG_CATEGORIES:
            categories.append(char)
    return list(set(categories))  # Unique categories


def extract_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and parse all annotation types from the dataframe.
    
    Returns dataframe with parsed annotation columns.
    """
    # Identify annotation columns
    go_col = None
    kegg_col = None
    cog_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if "go" in col_lower and go_col is None:
            go_col = col
        elif "kegg_ko" in col_lower and kegg_col is None:
            kegg_col = col
        elif col_lower in ["cog_category", "cog"] or (col_lower == "cog" or "COG" in col):
            cog_col = col
    
    # Try alternative column names from EggNOG format
    if go_col is None and "GOs" in df.columns:
        go_col = "GOs"
    if kegg_col is None and "KEGG_ko" in df.columns:
        kegg_col = "KEGG_ko"
    
    # Look for COG in the description or other columns
    if cog_col is None:
        # Try to find COG category from seed_eggNOG_ortholog or similar
        for col in df.columns:
            if "COG" in col.upper():
                cog_col = col
                break
    
    # Parse annotations
    if go_col and go_col in df.columns:
        df["go_terms"] = df[go_col].apply(parse_go_terms)
        print(f"  Parsed GO terms from column '{go_col}'")
    else:
        df["go_terms"] = [[] for _ in range(len(df))]
        print("  Warning: No GO term column found")
    
    if kegg_col and kegg_col in df.columns:
        df["kegg_ko"] = df[kegg_col].apply(parse_kegg_ko)
        print(f"  Parsed KEGG KO from column '{kegg_col}'")
    else:
        df["kegg_ko"] = [[] for _ in range(len(df))]
        print("  Warning: No KEGG KO column found")
    
    # For COG, we may need to extract from description or other text fields
    if cog_col and cog_col in df.columns:
        df["cog_categories"] = df[cog_col].apply(parse_cog_categories)
        print(f"  Parsed COG categories from column '{cog_col}'")
    else:
        # Try to extract from description
        df["cog_categories"] = [[] for _ in range(len(df))]
        # Look for patterns like "COG1234@1" in other columns
        for col in df.columns:
            if df["cog_categories"].apply(len).sum() > 0:
                break
            try:
                df["cog_categories"] = df[col].apply(lambda x: parse_cog_categories(str(x)) if pd.notna(x) else [])
                if df["cog_categories"].apply(len).sum() > 0:
                    print(f"  Extracted COG hints from column '{col}'")
            except Exception:
                continue
    
    return df


# =============================================================================
# Enrichment Analysis
# =============================================================================


def fisher_exact_enrichment(
    variant_annotations: list[str],
    other_annotations: list[str],
    annotation: str,
) -> dict:
    """
    Perform Fisher's exact test for enrichment of an annotation in a variant.
    
    Returns dict with test statistics.
    """
    # Build contingency table
    # [[variant_has, variant_not_has], [other_has, other_not_has]]
    variant_has = sum(1 for a in variant_annotations if annotation in a)
    variant_not = len(variant_annotations) - variant_has
    other_has = sum(1 for a in other_annotations if annotation in a)
    other_not = len(other_annotations) - other_has
    
    contingency = [[variant_has, variant_not], [other_has, other_not]]
    
    # Fisher's exact test
    odds_ratio, p_value = stats.fisher_exact(contingency, alternative="greater")
    
    # Confidence interval for odds ratio (approximate)
    # Using Woolf's method
    if min(variant_has, variant_not, other_has, other_not) > 0:
        log_or = np.log(odds_ratio) if odds_ratio > 0 else 0
        se = np.sqrt(1/variant_has + 1/variant_not + 1/other_has + 1/other_not)
        ci_lower = np.exp(log_or - 1.96 * se)
        ci_upper = np.exp(log_or + 1.96 * se)
    else:
        ci_lower, ci_upper = np.nan, np.nan
    
    return {
        "variant_count": variant_has,
        "variant_total": len(variant_annotations),
        "other_count": other_has,
        "other_total": len(other_annotations),
        "odds_ratio": odds_ratio,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
    }


def benjamini_hochberg_correction(p_values: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    if n == 0:
        return np.array([])
    
    # Sort p-values and get ranks
    sorted_idx = np.argsort(p_values)
    sorted_pvals = p_values[sorted_idx]
    
    # Calculate q-values
    q_values = np.zeros(n)
    cummin = 1.0
    
    for i in range(n - 1, -1, -1):
        rank = i + 1
        q = sorted_pvals[i] * n / rank
        cummin = min(cummin, q)
        q_values[sorted_idx[i]] = cummin
    
    return np.minimum(q_values, 1.0)


def run_enrichment_analysis(
    df: pd.DataFrame,
    annotation_col: str,
    annotation_name: str,
) -> pd.DataFrame:
    """
    Run enrichment analysis for a specific annotation type across all variants.
    
    Returns DataFrame with enrichment results.
    """
    variants = df["motif_variant"].unique()
    
    # Get all unique annotations
    all_annotations = set()
    for annot_list in df[annotation_col]:
        if isinstance(annot_list, list):
            all_annotations.update(annot_list)
    
    # Filter annotations by minimum count
    annotation_counts = Counter()
    for annot_list in df[annotation_col]:
        if isinstance(annot_list, list):
            annotation_counts.update(annot_list)
    
    valid_annotations = {a for a, c in annotation_counts.items() if c >= MIN_ANNOTATION_COUNT}
    
    results = []
    
    for variant in variants:
        variant_df = df[df["motif_variant"] == variant]
        other_df = df[df["motif_variant"] != variant]
        
        # Skip small variants
        if len(variant_df) < MIN_VARIANT_SIZE:
            continue
        
        # Get annotations as flat lists
        variant_annotations = []
        for annot_list in variant_df[annotation_col]:
            if isinstance(annot_list, list):
                variant_annotations.append(set(annot_list))
            else:
                variant_annotations.append(set())
        
        other_annotations = []
        for annot_list in other_df[annotation_col]:
            if isinstance(annot_list, list):
                other_annotations.append(set(annot_list))
            else:
                other_annotations.append(set())
        
        # Test each annotation
        for annotation in valid_annotations:
            result = fisher_exact_enrichment(
                [a for a in variant_annotations],
                [a for a in other_annotations],
                annotation,
            )
            
            # Count correctly by checking if annotation is in each set
            variant_has = sum(1 for a in variant_annotations if annotation in a)
            other_has = sum(1 for a in other_annotations if annotation in a)
            
            results.append({
                "variant": variant,
                "annotation": annotation,
                "annotation_type": annotation_name,
                "variant_count": variant_has,
                "variant_total": len(variant_annotations),
                "variant_fraction": variant_has / len(variant_annotations) if variant_annotations else 0,
                "other_count": other_has,
                "other_total": len(other_annotations),
                "other_fraction": other_has / len(other_annotations) if other_annotations else 0,
                "odds_ratio": result["odds_ratio"],
                "ci_lower": result["ci_lower"],
                "ci_upper": result["ci_upper"],
                "p_value": result["p_value"],
            })
    
    if not results:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    # FDR correction within each variant
    for variant in results_df["variant"].unique():
        mask = results_df["variant"] == variant
        p_values = results_df.loc[mask, "p_value"].values
        q_values = benjamini_hochberg_correction(p_values)
        results_df.loc[mask, "q_value"] = q_values
    
    # Mark significant results
    results_df["significant"] = results_df["q_value"] < FDR_THRESHOLD
    
    return results_df


def run_cog_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    """Run COG category enrichment analysis."""
    variants = df["motif_variant"].unique()
    results = []
    
    for variant in variants:
        variant_df = df[df["motif_variant"] == variant]
        other_df = df[df["motif_variant"] != variant]
        
        if len(variant_df) < MIN_VARIANT_SIZE:
            continue
        
        # Get COG categories
        variant_cogs = []
        for cog_list in variant_df["cog_categories"]:
            if isinstance(cog_list, list):
                variant_cogs.append(set(cog_list))
            else:
                variant_cogs.append(set())
        
        other_cogs = []
        for cog_list in other_df["cog_categories"]:
            if isinstance(cog_list, list):
                other_cogs.append(set(cog_list))
            else:
                other_cogs.append(set())
        
        # Test each COG category
        for cog, description in COG_CATEGORIES.items():
            variant_has = sum(1 for c in variant_cogs if cog in c)
            other_has = sum(1 for c in other_cogs if cog in c)
            
            # Build contingency table
            contingency = [
                [variant_has, len(variant_cogs) - variant_has],
                [other_has, len(other_cogs) - other_has],
            ]
            
            try:
                odds_ratio, p_value = stats.fisher_exact(contingency, alternative="greater")
            except Exception:
                odds_ratio, p_value = 1.0, 1.0
            
            results.append({
                "variant": variant,
                "cog_category": cog,
                "cog_description": description,
                "variant_count": variant_has,
                "variant_total": len(variant_cogs),
                "variant_fraction": variant_has / len(variant_cogs) if variant_cogs else 0,
                "other_count": other_has,
                "other_total": len(other_cogs),
                "other_fraction": other_has / len(other_cogs) if other_cogs else 0,
                "odds_ratio": odds_ratio,
                "p_value": p_value,
            })
    
    if not results:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    # FDR correction
    for variant in results_df["variant"].unique():
        mask = results_df["variant"] == variant
        p_values = results_df.loc[mask, "p_value"].values
        q_values = benjamini_hochberg_correction(p_values)
        results_df.loc[mask, "q_value"] = q_values
    
    results_df["significant"] = results_df["q_value"] < FDR_THRESHOLD
    
    return results_df


# =============================================================================
# Visualization
# =============================================================================


def plot_cog_distribution(df: pd.DataFrame, output_file: Path):
    """Plot COG category distribution by variant."""
    # Get top variants
    variant_counts = df["motif_variant"].value_counts()
    top_variants = variant_counts.head(6).index.tolist()
    
    # Compute COG fractions per variant
    cog_data = []
    for variant in top_variants:
        variant_df = df[df["motif_variant"] == variant]
        cog_counter = Counter()
        total = 0
        
        for cog_list in variant_df["cog_categories"]:
            if isinstance(cog_list, list) and len(cog_list) > 0:
                total += 1
                cog_counter.update(cog_list)
        
        for cog in COG_CATEGORIES:
            cog_data.append({
                "variant": f"uTP-{variant}",
                "cog": cog,
                "count": cog_counter.get(cog, 0),
                "fraction": cog_counter.get(cog, 0) / total if total > 0 else 0,
            })
    
    cog_df = pd.DataFrame(cog_data)
    
    # Create heatmap
    pivot = cog_df.pivot(index="cog", columns="variant", values="fraction")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        ax=ax,
        cbar_kws={"label": "Fraction"},
    )
    
    # Add COG descriptions
    yticklabels = [f"{cog} - {COG_CATEGORIES[cog][:30]}..." for cog in pivot.index]
    ax.set_yticklabels(yticklabels, fontsize=8)
    
    ax.set_title("COG Category Distribution by uTP Variant")
    ax.set_xlabel("uTP Variant")
    ax.set_ylabel("COG Category")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved COG distribution plot to {output_file}")


def plot_enrichment_heatmap(enrichment_df: pd.DataFrame, output_file: Path, title: str):
    """Plot heatmap of enrichment results (-log10 q-values)."""
    if enrichment_df.empty:
        print(f"  Warning: No enrichment data to plot for {title}")
        return
    
    # Filter to significant results
    sig_df = enrichment_df[enrichment_df["significant"]]
    
    if sig_df.empty:
        print(f"  No significant enrichments found for {title}")
        return
    
    # Get top annotations by significance
    top_annotations = (
        sig_df.groupby("annotation")["q_value"]
        .min()
        .sort_values()
        .head(20)
        .index.tolist()
    )
    
    if not top_annotations:
        print(f"  No significant annotations to plot for {title}")
        return
    
    # Filter to top annotations
    plot_df = enrichment_df[enrichment_df["annotation"].isin(top_annotations)]
    
    # Create pivot table of -log10(q-value)
    plot_df = plot_df.copy()
    plot_df["neg_log_q"] = -np.log10(plot_df["q_value"].clip(lower=1e-10))
    pivot = plot_df.pivot(index="annotation", columns="variant", values="neg_log_q")
    pivot = pivot.fillna(0)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(top_annotations) * 0.3)))
    
    sns.heatmap(
        pivot,
        cmap="Reds",
        ax=ax,
        cbar_kws={"label": "-log10(q-value)"},
    )
    
    ax.set_title(f"{title}\n(FDR < {FDR_THRESHOLD})")
    ax.set_xlabel("uTP Variant")
    ax.set_ylabel("Annotation")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved enrichment heatmap to {output_file}")


def plot_annotation_summary(df: pd.DataFrame, output_file: Path):
    """Plot summary of annotation coverage by variant."""
    # Compute annotation coverage per variant
    summary_data = []
    
    for variant in df["motif_variant"].unique():
        variant_df = df[df["motif_variant"] == variant]
        n = len(variant_df)
        
        # GO coverage
        go_count = sum(1 for g in variant_df["go_terms"] if isinstance(g, list) and len(g) > 0)
        
        # KEGG coverage
        kegg_count = sum(1 for k in variant_df["kegg_ko"] if isinstance(k, list) and len(k) > 0)
        
        # COG coverage
        cog_count = sum(1 for c in variant_df["cog_categories"] if isinstance(c, list) and len(c) > 0)
        
        summary_data.append({
            "variant": f"uTP-{variant}",
            "n_proteins": n,
            "go_coverage": go_count / n if n > 0 else 0,
            "kegg_coverage": kegg_count / n if n > 0 else 0,
            "cog_coverage": cog_count / n if n > 0 else 0,
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by protein count
    summary_df = summary_df.sort_values("n_proteins", ascending=False).head(10)
    
    # Melt for plotting
    plot_df = summary_df.melt(
        id_vars=["variant", "n_proteins"],
        value_vars=["go_coverage", "kegg_coverage", "cog_coverage"],
        var_name="annotation_type",
        value_name="coverage",
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.barplot(
        data=plot_df,
        x="variant",
        y="coverage",
        hue="annotation_type",
        ax=ax,
    )
    
    ax.set_ylabel("Annotation Coverage")
    ax.set_xlabel("uTP Variant")
    ax.set_title("Annotation Coverage by uTP Variant")
    ax.legend(title="Annotation Type", loc="upper right")
    ax.tick_params(axis='x', rotation=45)
    
    # Add protein counts as secondary labels
    for i, (_, row) in enumerate(summary_df.iterrows()):
        ax.text(i, -0.05, f"n={row['n_proteins']}", ha='center', fontsize=8, transform=ax.get_xaxis_transform())
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved annotation summary plot to {output_file}")


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("04_annotation_analysis.py - Functional Annotation Analysis")
    print("=" * 70)
    
    # Create output directories
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check input files
    if not PROCESSED_PROTEINS_FILE.exists():
        raise FileNotFoundError(
            f"Processed proteins not found: {PROCESSED_PROTEINS_FILE}\n"
            f"Run 01_prepare_data.py first."
        )
    
    # =========================================================================
    # Step 1: Load data
    # =========================================================================
    print("\n[1/5] Loading data...")
    df = load_processed_proteins()
    print(f"  Loaded {len(df)} proteins")
    
    # Extract and parse annotations
    print("\n[2/5] Parsing annotations...")
    df = extract_annotations(df)
    
    # Annotation coverage summary
    go_coverage = sum(1 for g in df["go_terms"] if isinstance(g, list) and len(g) > 0)
    kegg_coverage = sum(1 for k in df["kegg_ko"] if isinstance(k, list) and len(k) > 0)
    cog_coverage = sum(1 for c in df["cog_categories"] if isinstance(c, list) and len(c) > 0)
    
    print(f"\n  Annotation coverage:")
    print(f"    GO terms: {go_coverage}/{len(df)} ({go_coverage/len(df)*100:.1f}%)")
    print(f"    KEGG KO: {kegg_coverage}/{len(df)} ({kegg_coverage/len(df)*100:.1f}%)")
    print(f"    COG categories: {cog_coverage}/{len(df)} ({cog_coverage/len(df)*100:.1f}%)")
    
    # =========================================================================
    # Step 3: COG category enrichment
    # =========================================================================
    print("\n[3/5] Running COG category enrichment analysis...")
    cog_results = run_cog_enrichment(df)
    
    if not cog_results.empty:
        cog_results.to_csv(STATS_DIR / "cog_enrichment.csv", index=False)
        n_sig = cog_results["significant"].sum()
        print(f"  Found {n_sig} significant COG enrichments (FDR < {FDR_THRESHOLD})")
        
        # Show top results
        if n_sig > 0:
            print("\n  Top COG enrichments:")
            top_cog = cog_results[cog_results["significant"]].nsmallest(5, "q_value")
            for _, row in top_cog.iterrows():
                print(f"    {row['variant']} - {row['cog_category']} ({row['cog_description'][:30]}...): OR={row['odds_ratio']:.2f}, q={row['q_value']:.3f}")
    else:
        print("  No COG enrichment results (insufficient data)")
    
    # =========================================================================
    # Step 4: GO term enrichment
    # =========================================================================
    print("\n[4/5] Running GO term enrichment analysis...")
    go_results = run_enrichment_analysis(df, "go_terms", "GO")
    
    if not go_results.empty:
        go_results.to_csv(STATS_DIR / "go_enrichment.csv", index=False)
        n_sig = go_results["significant"].sum()
        print(f"  Found {n_sig} significant GO enrichments (FDR < {FDR_THRESHOLD})")
    else:
        print("  No GO enrichment results (insufficient data)")
    
    # =========================================================================
    # Step 5: Generate visualizations
    # =========================================================================
    print("\n[5/5] Generating visualizations...")
    
    plot_annotation_summary(df, FIGURES_DIR / "annotation_coverage.svg")
    
    if not cog_results.empty:
        plot_cog_distribution(df, FIGURES_DIR / "cog_distribution.svg")
    
    if not go_results.empty and go_results["significant"].sum() > 0:
        plot_enrichment_heatmap(go_results, FIGURES_DIR / "go_enrichment_heatmap.svg", "GO Term Enrichment")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Annotation analysis complete!")
    print("=" * 70)
    
    # Save annotation summary
    summary = {
        "total_proteins": len(df),
        "go_coverage": go_coverage,
        "kegg_coverage": kegg_coverage,
        "cog_coverage": cog_coverage,
        "significant_cog_enrichments": cog_results["significant"].sum() if not cog_results.empty else 0,
        "significant_go_enrichments": go_results["significant"].sum() if not go_results.empty else 0,
    }
    pd.DataFrame([summary]).to_csv(STATS_DIR / "annotation_summary.csv", index=False)
    
    print(f"\n📁 Outputs saved to:")
    print(f"  - {STATS_DIR / 'cog_enrichment.csv'}")
    print(f"  - {STATS_DIR / 'go_enrichment.csv'}")
    print(f"  - {STATS_DIR / 'annotation_summary.csv'}")
    print(f"  - {FIGURES_DIR / 'annotation_coverage.svg'}")
    print(f"  - {FIGURES_DIR / 'cog_distribution.svg'}")
    
    print(f"\nNext step: Run 05_interpretability.py for feature importance analysis")


if __name__ == "__main__":
    main()
