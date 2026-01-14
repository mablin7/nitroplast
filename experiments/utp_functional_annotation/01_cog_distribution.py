#!/usr/bin/env python3
"""
Analyze COG category distribution for uTP proteins.

This script examines the distribution of COG functional categories among
uTP-containing proteins compared to the background B. bigelowii proteome.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import Counter

# Configure matplotlib
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']
rcParams['font.size'] = 9
rcParams['axes.linewidth'] = 1.0
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['legend.frameon'] = False

# Color palette
COLORS = {
    'primary': '#2E4057',
    'secondary': '#048A81',
    'accent': '#E85D04',
    'light': '#90BE6D',
    'highlight': '#F9C74F',
    'background': '#F8F9FA',
    'text': '#212529',
}

# COG category descriptions
COG_DESCRIPTIONS = {
    'A': 'RNA processing and modification',
    'B': 'Chromatin structure and dynamics',
    'C': 'Energy production and conversion',
    'D': 'Cell cycle control, cell division',
    'E': 'Amino acid transport and metabolism',
    'F': 'Nucleotide transport and metabolism',
    'G': 'Carbohydrate transport and metabolism',
    'H': 'Coenzyme transport and metabolism',
    'I': 'Lipid transport and metabolism',
    'J': 'Translation, ribosomal structure',
    'K': 'Transcription',
    'L': 'Replication, recombination and repair',
    'M': 'Cell wall/membrane/envelope biogenesis',
    'N': 'Cell motility',
    'O': 'Post-translational modification, protein turnover',
    'P': 'Inorganic ion transport and metabolism',
    'Q': 'Secondary metabolites biosynthesis',
    'R': 'General function prediction only',
    'S': 'Function unknown',
    'T': 'Signal transduction mechanisms',
    'U': 'Intracellular trafficking, secretion',
    'V': 'Defense mechanisms',
    'W': 'Extracellular structures',
    'X': 'Mobilome: prophages, transposons',
    'Y': 'Nuclear structure',
    'Z': 'Cytoskeleton',
}

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(__file__).parent / "output"


def load_utp_proteins():
    """Load uTP protein IDs from both datasets."""
    
    high_conf_fasta = DATA_DIR / "ucyna_enriched_good_c_align_full_sequences.fasta"
    high_conf_ids = set()
    for record in SeqIO.parse(high_conf_fasta, "fasta"):
        high_conf_ids.add(record.id)
    
    import_candidates_fasta = DATA_DIR / "Import_candidates.fasta"
    import_candidate_ids = set()
    for record in SeqIO.parse(import_candidates_fasta, "fasta"):
        import_candidate_ids.add(record.id)
    
    return high_conf_ids, import_candidate_ids


def load_annotations():
    """Load eggNOG annotations for B. bigelowii transcriptome."""
    
    annot_file = DATA_DIR / "Bbigelowii_transcriptome_annotations.csv"
    df = pd.read_csv(annot_file, comment='#', header=None, low_memory=False)
    
    # Based on the file structure
    columns = ['query_name', 'seed_eggNOG_ortholog', 'seed_ortholog_evalue', 
               'seed_ortholog_score', 'best_tax_level', 'Preferred_name', 'GOs', 
               'EC', 'KEGG_ko', 'KEGG_Pathway', 'KEGG_Module', 'KEGG_Reaction',
               'KEGG_rclass', 'BRITE', 'KEGG_TC', 'CAZy', 'BiGG_Reaction',
               'tax_scope', 'eggNOG_OGs', 'best_OG', 'COG_category', 'Description']
    
    df.columns = columns[:df.shape[1]]
    
    return df


def parse_cog_categories(cog_string):
    """Parse COG category string (can contain multiple categories)."""
    if pd.isna(cog_string) or cog_string == '' or cog_string == '-':
        return []
    # COG categories are single letters, sometimes multiple
    return list(str(cog_string).strip())


def count_cog_categories(df, protein_ids=None):
    """Count COG category occurrences for a set of proteins."""
    
    if protein_ids is not None:
        subset = df[df['query_name'].isin(protein_ids)]
    else:
        subset = df
    
    # Count all COG categories
    all_cats = []
    for cog in subset['COG_category'].dropna():
        all_cats.extend(parse_cog_categories(cog))
    
    counts = Counter(all_cats)
    return counts, len(subset)


def main():
    """Main analysis function."""
    
    print("=" * 60)
    print("COG Category Distribution Analysis for uTP Proteins")
    print("=" * 60)
    
    # Load data
    high_conf_ids, import_candidate_ids = load_utp_proteins()
    annotations = load_annotations()
    
    print(f"\nLoaded {len(annotations)} annotations")
    print(f"COG category column sample:\n{annotations['COG_category'].head(20)}")
    
    # Count COG categories for each group
    high_conf_counts, high_conf_n = count_cog_categories(annotations, high_conf_ids)
    import_counts, import_n = count_cog_categories(annotations, import_candidate_ids)
    all_counts, all_n = count_cog_categories(annotations)
    
    print(f"\n--- COG Category Counts ---")
    print(f"High-confidence uTP proteins (n={high_conf_n}):")
    for cat, count in sorted(high_conf_counts.items(), key=lambda x: -x[1]):
        desc = COG_DESCRIPTIONS.get(cat, 'Unknown')
        print(f"  {cat}: {count} ({100*count/high_conf_n:.1f}%) - {desc}")
    
    print(f"\nImport candidates (n={import_n}):")
    for cat, count in sorted(import_counts.items(), key=lambda x: -x[1])[:15]:
        desc = COG_DESCRIPTIONS.get(cat, 'Unknown')
        print(f"  {cat}: {count} ({100*count/import_n:.1f}%) - {desc}")
    
    # Create comparison dataframe
    all_categories = sorted(set(high_conf_counts.keys()) | set(import_counts.keys()) | set(all_counts.keys()))
    
    comparison_data = []
    for cat in all_categories:
        comparison_data.append({
            'COG_category': cat,
            'Description': COG_DESCRIPTIONS.get(cat, 'Unknown'),
            'High_conf_count': high_conf_counts.get(cat, 0),
            'High_conf_pct': 100 * high_conf_counts.get(cat, 0) / high_conf_n if high_conf_n > 0 else 0,
            'Import_count': import_counts.get(cat, 0),
            'Import_pct': 100 * import_counts.get(cat, 0) / import_n if import_n > 0 else 0,
            'Background_count': all_counts.get(cat, 0),
            'Background_pct': 100 * all_counts.get(cat, 0) / all_n if all_n > 0 else 0,
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('High_conf_count', ascending=False)
    comparison_df.to_csv(OUTPUT_DIR / "cog_category_distribution.csv", index=False)
    print(f"\nSaved COG distribution to {OUTPUT_DIR / 'cog_category_distribution.csv'}")
    
    # Create visualization
    create_cog_figure(comparison_df, high_conf_n, import_n, all_n)
    
    # Enrichment analysis
    print("\n--- Enrichment Analysis (High-conf uTP vs Background) ---")
    enrichment_results = []
    for _, row in comparison_df.iterrows():
        cat = row['COG_category']
        high_pct = row['High_conf_pct']
        bg_pct = row['Background_pct']
        
        if bg_pct > 0:
            fold_change = high_pct / bg_pct
        else:
            fold_change = np.inf if high_pct > 0 else 1.0
        
        enrichment_results.append({
            'COG_category': cat,
            'Description': row['Description'],
            'uTP_pct': high_pct,
            'Background_pct': bg_pct,
            'Fold_change': fold_change
        })
    
    enrichment_df = pd.DataFrame(enrichment_results)
    enrichment_df = enrichment_df.sort_values('Fold_change', ascending=False)
    enrichment_df.to_csv(OUTPUT_DIR / "cog_enrichment.csv", index=False)
    
    print("\nTop enriched categories in uTP proteins:")
    for _, row in enrichment_df.head(10).iterrows():
        if row['Fold_change'] != np.inf:
            print(f"  {row['COG_category']}: {row['Fold_change']:.2f}x ({row['uTP_pct']:.1f}% vs {row['Background_pct']:.1f}%) - {row['Description']}")
    
    print("\nTop depleted categories in uTP proteins:")
    for _, row in enrichment_df.tail(10).iterrows():
        if row['Fold_change'] > 0:
            print(f"  {row['COG_category']}: {row['Fold_change']:.2f}x ({row['uTP_pct']:.1f}% vs {row['Background_pct']:.1f}%) - {row['Description']}")
    
    return comparison_df, enrichment_df


def create_cog_figure(comparison_df, high_conf_n, import_n, all_n):
    """Create a figure showing COG category distribution."""
    
    # Filter to categories with at least some representation
    plot_df = comparison_df[
        (comparison_df['High_conf_count'] > 0) | 
        (comparison_df['Import_count'] > 0)
    ].copy()
    
    # Sort by high-confidence count
    plot_df = plot_df.sort_values('High_conf_pct', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y = np.arange(len(plot_df))
    height = 0.25
    
    # Plot bars
    bars1 = ax.barh(y - height, plot_df['High_conf_pct'], height, 
                    label=f'High-confidence uTP (n={high_conf_n})', 
                    color=COLORS['accent'], edgecolor='white', linewidth=0.5)
    bars2 = ax.barh(y, plot_df['Import_pct'], height, 
                    label=f'Import candidates (n={import_n})', 
                    color=COLORS['secondary'], edgecolor='white', linewidth=0.5)
    bars3 = ax.barh(y + height, plot_df['Background_pct'], height, 
                    label=f'All proteins (n={all_n})', 
                    color=COLORS['primary'], alpha=0.5, edgecolor='white', linewidth=0.5)
    
    # Labels
    ax.set_yticks(y)
    labels = [f"{row['COG_category']}: {row['Description'][:40]}" for _, row in plot_df.iterrows()]
    ax.set_yticklabels(labels)
    ax.set_xlabel('Percentage of proteins (%)')
    ax.set_title('COG Category Distribution in uTP Proteins', fontweight='bold')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    fig.savefig(OUTPUT_DIR / "cog_distribution.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / "cog_distribution.svg", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\nSaved COG distribution figure to {OUTPUT_DIR / 'cog_distribution.png'}")


if __name__ == "__main__":
    main()
