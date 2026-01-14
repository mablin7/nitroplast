#!/usr/bin/env python3
"""
03_parse_annotations.py - Parse eggNOG-mapper results

This script:
1. Parses eggNOG-mapper output file
2. Extracts COG categories, GO terms, KEGG pathways
3. Merges with biophysical properties
4. Creates analysis-ready datasets

Usage:
    uv run python experiments/utp_functional_annotation/03_parse_annotations.py
"""

from pathlib import Path

import pandas as pd

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"

# Input files
EGGNOG_RESULTS = OUTPUT_DIR / "eggnog_results.emapper.annotations"
PROPERTIES_FILE = OUTPUT_DIR / "protein_properties.csv"
SEQUENCE_METADATA = OUTPUT_DIR / "sequence_metadata.csv"

# Output files
MERGED_DATA = OUTPUT_DIR / "merged_data.csv"
ANNOTATION_SUMMARY = OUTPUT_DIR / "annotation_summary.csv"

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


def parse_eggnog_output(filepath: Path) -> pd.DataFrame:
    """Parse eggNOG-mapper annotations file."""
    
    # eggNOG-mapper output columns (v2.1+)
    columns = [
        'query', 'seed_ortholog', 'evalue', 'score', 'eggNOG_OGs',
        'max_annot_lvl', 'COG_category', 'Description', 'Preferred_name',
        'GOs', 'EC', 'KEGG_ko', 'KEGG_Pathway', 'KEGG_Module',
        'KEGG_Reaction', 'KEGG_rclass', 'BRITE', 'KEGG_TC', 'CAZy',
        'BiGG_Reaction', 'PFAMs'
    ]
    
    # Read file, skipping comment lines
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split('\t')
            if len(parts) >= len(columns):
                rows.append(dict(zip(columns, parts[:len(columns)])))
            elif len(parts) > 0:
                # Pad with empty strings if fewer columns
                row = parts + [''] * (len(columns) - len(parts))
                rows.append(dict(zip(columns, row)))
    
    df = pd.DataFrame(rows)
    
    # Clean up COG categories
    df['COG_category'] = df['COG_category'].replace('-', '')
    df['COG_category'] = df['COG_category'].replace('', pd.NA)
    
    return df


def extract_cog_categories(cog_string):
    """Extract individual COG categories from a string (can be multi-letter)."""
    if pd.isna(cog_string) or cog_string == '' or cog_string == '-':
        return []
    return list(str(cog_string).strip())


def get_primary_cog(cog_string):
    """Get the first/primary COG category."""
    cats = extract_cog_categories(cog_string)
    return cats[0] if cats else None


def main():
    """Main function to parse and merge annotations."""
    
    print("=" * 70)
    print("Parsing eggNOG-mapper Results")
    print("=" * 70)
    
    # Check if eggNOG results exist
    if not EGGNOG_RESULTS.exists():
        print(f"\nERROR: eggNOG results file not found: {EGGNOG_RESULTS}")
        print("\nPlease submit sequences to eggNOG-mapper and save results to:")
        print(f"  {EGGNOG_RESULTS}")
        print("\nAlternatively, we can use the existing B. bigelowii annotations.")
        print("Running fallback: using existing annotations from data/Bbigelowii_transcriptome_annotations.csv")
        
        # Fallback: use existing annotations
        return use_existing_annotations()
    
    # Parse eggNOG results
    print("\n--- Parsing eggNOG results ---")
    annotations = parse_eggnog_output(EGGNOG_RESULTS)
    print(f"Parsed {len(annotations)} annotation entries")
    
    # Load metadata and properties
    print("\n--- Loading metadata and properties ---")
    metadata = pd.read_csv(SEQUENCE_METADATA)
    properties = pd.read_csv(PROPERTIES_FILE)
    
    print(f"Metadata: {len(metadata)} sequences")
    print(f"Properties: {len(properties)} sequences")
    
    # Merge annotations with metadata
    # The query column should match eggnog_id
    annotations = annotations.rename(columns={'query': 'eggnog_id'})
    
    merged = metadata.merge(annotations, on='eggnog_id', how='left')
    merged = merged.merge(properties, left_on='original_id', right_on='sequence_id', how='left')
    
    # Extract primary COG category
    merged['primary_cog'] = merged['COG_category'].apply(get_primary_cog)
    merged['cog_description'] = merged['primary_cog'].map(COG_DESCRIPTIONS)
    
    # Count GO terms
    merged['n_go_terms'] = merged['GOs'].apply(
        lambda x: len(x.split(',')) if pd.notna(x) and x != '-' else 0
    )
    
    # Has KEGG pathway?
    merged['has_kegg_pathway'] = merged['KEGG_Pathway'].apply(
        lambda x: pd.notna(x) and x != '-' and x != ''
    )
    
    # Has EC number?
    merged['has_ec'] = merged['EC'].apply(
        lambda x: pd.notna(x) and x != '-' and x != ''
    )
    
    print(f"\nMerged dataset: {len(merged)} sequences")
    
    # Save merged data
    merged.to_csv(MERGED_DATA, index=False)
    print(f"Saved merged data to: {MERGED_DATA}")
    
    # Annotation summary
    print("\n--- Annotation Summary ---")
    summary_data = []
    
    for group in ['uTP', 'Control']:
        subset = merged[merged['group_x'] == group]
        n_total = len(subset)
        n_with_cog = subset['primary_cog'].notna().sum()
        n_with_go = (subset['n_go_terms'] > 0).sum()
        n_with_kegg = subset['has_kegg_pathway'].sum()
        n_with_ec = subset['has_ec'].sum()
        
        summary_data.append({
            'group': group,
            'total': n_total,
            'with_cog': n_with_cog,
            'cog_pct': 100 * n_with_cog / n_total if n_total > 0 else 0,
            'with_go': n_with_go,
            'go_pct': 100 * n_with_go / n_total if n_total > 0 else 0,
            'with_kegg': n_with_kegg,
            'kegg_pct': 100 * n_with_kegg / n_total if n_total > 0 else 0,
            'with_ec': n_with_ec,
            'ec_pct': 100 * n_with_ec / n_total if n_total > 0 else 0,
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(ANNOTATION_SUMMARY, index=False)
    print(summary_df.to_string(index=False))
    
    # COG distribution preview
    print("\n--- COG Category Distribution ---")
    print("\nuTP proteins:")
    utp_cogs = merged[merged['group_x'] == 'uTP']['primary_cog'].value_counts()
    print(utp_cogs.head(10))
    
    print("\nControl proteins:")
    ctrl_cogs = merged[merged['group_x'] == 'Control']['primary_cog'].value_counts()
    print(ctrl_cogs.head(10))
    
    return merged


def use_existing_annotations():
    """Fallback: Use existing B. bigelowii annotations."""
    
    print("\n--- Using Existing Annotations (Fallback) ---")
    
    # Load existing annotations
    annot_file = Path(__file__).parent.parent.parent / "data" / "Bbigelowii_transcriptome_annotations.csv"
    
    if not annot_file.exists():
        print(f"ERROR: Cannot find existing annotations at {annot_file}")
        return None
    
    # Load the annotation file
    annotations = pd.read_csv(annot_file, comment='#', header=None, low_memory=False)
    
    columns = ['query_name', 'seed_eggNOG_ortholog', 'seed_ortholog_evalue', 
               'seed_ortholog_score', 'best_tax_level', 'Preferred_name', 'GOs', 
               'EC', 'KEGG_ko', 'KEGG_Pathway', 'KEGG_Module', 'KEGG_Reaction',
               'KEGG_rclass', 'BRITE', 'KEGG_TC', 'CAZy', 'BiGG_Reaction',
               'tax_scope', 'eggNOG_OGs', 'best_OG', 'COG_category', 'Description']
    
    annotations.columns = columns[:annotations.shape[1]]
    
    print(f"Loaded {len(annotations)} existing annotations")
    
    # Load metadata and properties
    metadata = pd.read_csv(SEQUENCE_METADATA)
    properties = pd.read_csv(PROPERTIES_FILE)
    
    # Merge based on original_id matching query_name
    merged = metadata.merge(
        annotations, 
        left_on='original_id', 
        right_on='query_name', 
        how='left'
    )
    merged = merged.merge(
        properties, 
        left_on='original_id', 
        right_on='sequence_id', 
        how='left'
    )
    
    # Extract primary COG category
    merged['primary_cog'] = merged['COG_category'].apply(get_primary_cog)
    merged['cog_description'] = merged['primary_cog'].map(COG_DESCRIPTIONS)
    
    # Count GO terms
    merged['n_go_terms'] = merged['GOs'].apply(
        lambda x: len(str(x).split(',')) if pd.notna(x) and str(x) != '-' else 0
    )
    
    # Has KEGG pathway?
    merged['has_kegg_pathway'] = merged['KEGG_Pathway'].apply(
        lambda x: pd.notna(x) and str(x) != '-' and str(x) != ''
    )
    
    # Has EC number?
    merged['has_ec'] = merged['EC'].apply(
        lambda x: pd.notna(x) and str(x) != '-' and str(x) != ''
    )
    
    # Rename group column for consistency
    if 'group_x' not in merged.columns and 'group' in merged.columns:
        merged = merged.rename(columns={'group': 'group_x'})
    
    print(f"\nMerged dataset: {len(merged)} sequences")
    
    # Save merged data
    merged.to_csv(MERGED_DATA, index=False)
    print(f"Saved merged data to: {MERGED_DATA}")
    
    # Annotation summary
    print("\n--- Annotation Summary ---")
    for group in ['uTP', 'Control']:
        subset = merged[merged['group_x'] == group]
        n_total = len(subset)
        n_with_cog = subset['primary_cog'].notna().sum()
        print(f"{group}: {n_total} total, {n_with_cog} ({100*n_with_cog/n_total:.1f}%) with COG annotation")
    
    return merged


if __name__ == "__main__":
    main()
