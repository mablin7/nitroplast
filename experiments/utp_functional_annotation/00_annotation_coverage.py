#!/usr/bin/env python3
"""
Analyze functional annotation coverage for uTP proteins.

This script examines what functional annotations are available for uTP-containing
proteins from B. bigelowii, assessing coverage across different annotation systems
(GO, COG, KEGG, etc.) from the eggNOG-mapper results.

Data sources:
- ucyna_enriched_good_c_align_full_sequences.fasta: 206 proteins with good C-terminal 
  alignment (high-confidence uTP proteins from Coale et al. 2024)
- Import_candidates.fasta: ~900+ HMM-predicted uTP-containing proteins
- Bbigelowii_transcriptome_annotations.csv: eggNOG annotations for B. bigelowii transcriptome
"""

import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
import matplotlib.pyplot as plt
from matplotlib import rcParams

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

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(__file__).parent / "output"


def load_utp_proteins():
    """Load uTP protein IDs from both datasets."""
    
    # High-confidence uTP proteins (good C-terminal alignment)
    high_conf_fasta = DATA_DIR / "ucyna_enriched_good_c_align_full_sequences.fasta"
    high_conf_ids = set()
    for record in SeqIO.parse(high_conf_fasta, "fasta"):
        high_conf_ids.add(record.id)
    
    # All import candidates (HMM-predicted)
    import_candidates_fasta = DATA_DIR / "Import_candidates.fasta"
    import_candidate_ids = set()
    for record in SeqIO.parse(import_candidates_fasta, "fasta"):
        import_candidate_ids.add(record.id)
    
    print(f"High-confidence uTP proteins: {len(high_conf_ids)}")
    print(f"Import candidates (HMM-predicted): {len(import_candidate_ids)}")
    
    return high_conf_ids, import_candidate_ids


def load_annotations():
    """Load eggNOG annotations for B. bigelowii transcriptome."""
    
    annot_file = DATA_DIR / "Bbigelowii_transcriptome_annotations.csv"
    
    # Read the file - it's a CSV with eggNOG output format
    # First few lines are comments starting with #
    df = pd.read_csv(annot_file, comment='#', header=None, low_memory=False)
    
    # The header is in the first row after comments
    # Based on the file structure, columns are:
    # query_name, seed_eggNOG_ortholog, seed_ortholog_evalue, seed_ortholog_score,
    # best_tax_level, Preferred_name, GOs, EC, KEGG_ko, KEGG_Pathway, KEGG_Module,
    # KEGG_Reaction, KEGG_rclass, BRITE, KEGG_TC, CAZy, BiGG_Reaction, ..., COG_category, Description
    
    # Let's check the actual structure
    print(f"Annotation file shape: {df.shape}")
    print(f"First few rows:\n{df.head()}")
    
    # Rename columns based on eggNOG output format
    columns = ['query_name', 'seed_eggNOG_ortholog', 'seed_ortholog_evalue', 
               'seed_ortholog_score', 'best_tax_level', 'Preferred_name', 'GOs', 
               'EC', 'KEGG_ko', 'KEGG_Pathway', 'KEGG_Module', 'KEGG_Reaction',
               'KEGG_rclass', 'BRITE', 'KEGG_TC', 'CAZy', 'BiGG_Reaction']
    
    # Add remaining columns if present
    if df.shape[1] > len(columns):
        for i in range(len(columns), df.shape[1]):
            columns.append(f'col_{i}')
    
    df.columns = columns[:df.shape[1]]
    
    return df


def analyze_annotation_coverage(annotations_df, protein_ids, label=""):
    """Analyze annotation coverage for a set of protein IDs."""
    
    # Filter to proteins of interest
    subset = annotations_df[annotations_df['query_name'].isin(protein_ids)].copy()
    
    total = len(protein_ids)
    annotated = len(subset)
    
    print(f"\n=== {label} ===")
    print(f"Total proteins: {total}")
    print(f"Proteins with any annotation: {annotated} ({100*annotated/total:.1f}%)")
    
    coverage = {
        'total': total,
        'annotated': annotated,
        'annotation_rate': annotated / total if total > 0 else 0
    }
    
    # Check coverage for different annotation types
    annotation_types = {
        'Preferred_name': 'Gene name',
        'GOs': 'GO terms',
        'EC': 'EC number',
        'KEGG_ko': 'KEGG ortholog',
        'KEGG_Pathway': 'KEGG pathway',
        'KEGG_Module': 'KEGG module',
    }
    
    # Also check for COG category if present
    if 'col_17' in subset.columns:
        annotation_types['col_17'] = 'COG category'
    
    for col, name in annotation_types.items():
        if col in subset.columns:
            # Count non-empty, non-NA values
            has_annot = subset[col].notna() & (subset[col] != '') & (subset[col] != '-')
            count = has_annot.sum()
            pct = 100 * count / total if total > 0 else 0
            print(f"  {name}: {count} ({pct:.1f}%)")
            coverage[name] = count
            coverage[f'{name}_pct'] = pct
    
    return coverage, subset


def main():
    """Main analysis function."""
    
    print("=" * 60)
    print("uTP Protein Functional Annotation Coverage Analysis")
    print("=" * 60)
    
    # Load data
    print("\n--- Loading protein IDs ---")
    high_conf_ids, import_candidate_ids = load_utp_proteins()
    
    print("\n--- Loading annotations ---")
    annotations = load_annotations()
    
    # Get all protein IDs in the annotation file
    all_annotated_ids = set(annotations['query_name'].dropna())
    print(f"Total proteins in annotation file: {len(all_annotated_ids)}")
    
    # Check overlap
    high_conf_in_annot = high_conf_ids & all_annotated_ids
    import_in_annot = import_candidate_ids & all_annotated_ids
    
    print(f"\nHigh-confidence uTP proteins found in annotations: {len(high_conf_in_annot)}")
    print(f"Import candidates found in annotations: {len(import_in_annot)}")
    
    # Analyze coverage for each set
    results = {}
    
    # High-confidence uTP proteins
    coverage_high, subset_high = analyze_annotation_coverage(
        annotations, high_conf_ids, "High-confidence uTP proteins (good C-term alignment)"
    )
    results['high_confidence'] = coverage_high
    
    # All import candidates
    coverage_import, subset_import = analyze_annotation_coverage(
        annotations, import_candidate_ids, "All import candidates (HMM-predicted)"
    )
    results['import_candidates'] = coverage_import
    
    # All B. bigelowii proteins (for comparison)
    coverage_all, subset_all = analyze_annotation_coverage(
        annotations, all_annotated_ids, "All B. bigelowii proteins (background)"
    )
    results['all_proteins'] = coverage_all
    
    # Save detailed results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(OUTPUT_DIR / "annotation_coverage_summary.csv")
    print(f"\nSaved coverage summary to {OUTPUT_DIR / 'annotation_coverage_summary.csv'}")
    
    # Save the annotated subsets
    if len(subset_high) > 0:
        subset_high.to_csv(OUTPUT_DIR / "high_confidence_utp_annotations.csv", index=False)
    if len(subset_import) > 0:
        subset_import.to_csv(OUTPUT_DIR / "import_candidates_annotations.csv", index=False)
    
    # Create visualization
    create_coverage_figure(results)
    
    return results


def create_coverage_figure(results):
    """Create a figure showing annotation coverage comparison."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Overall annotation rate
    ax = axes[0]
    categories = ['High-confidence\nuTP', 'Import\ncandidates', 'All B. bigelowii\nproteins']
    rates = [
        results['high_confidence']['annotation_rate'] * 100,
        results['import_candidates']['annotation_rate'] * 100,
        results['all_proteins']['annotation_rate'] * 100
    ]
    
    bars = ax.bar(categories, rates, color=[COLORS['accent'], COLORS['secondary'], COLORS['primary']], 
                  edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Proteins with annotation (%)')
    ax.set_ylim(0, 105)
    ax.set_title('A. Overall annotation coverage', fontweight='bold', loc='left')
    
    # Add value labels
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Panel B: Coverage by annotation type
    ax = axes[1]
    
    annotation_types = ['Gene name', 'GO terms', 'EC number', 'KEGG ortholog', 'KEGG pathway']
    
    x = np.arange(len(annotation_types))
    width = 0.25
    
    high_conf_pcts = [results['high_confidence'].get(f'{t}_pct', 0) for t in annotation_types]
    import_pcts = [results['import_candidates'].get(f'{t}_pct', 0) for t in annotation_types]
    all_pcts = [results['all_proteins'].get(f'{t}_pct', 0) for t in annotation_types]
    
    ax.bar(x - width, high_conf_pcts, width, label='High-confidence uTP', 
           color=COLORS['accent'], edgecolor='white', linewidth=0.5)
    ax.bar(x, import_pcts, width, label='Import candidates', 
           color=COLORS['secondary'], edgecolor='white', linewidth=0.5)
    ax.bar(x + width, all_pcts, width, label='All proteins', 
           color=COLORS['primary'], edgecolor='white', linewidth=0.5)
    
    ax.set_ylabel('Proteins with annotation (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(annotation_types, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)
    ax.set_title('B. Coverage by annotation type', fontweight='bold', loc='left')
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(OUTPUT_DIR / "annotation_coverage.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / "annotation_coverage.svg", bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\nSaved figure to {OUTPUT_DIR / 'annotation_coverage.png'}")


if __name__ == "__main__":
    main()
