#!/usr/bin/env python3

import pandas as pd
import argparse
from collections import defaultdict


def load_interpro_results(filepath):
    """Load InterProScan results from TSV file."""
    df = pd.read_csv(filepath, sep="\t")
    df.columns = [
        "accession",
        "sequence_md5",
        "sequence_length",
        "analysis",
        "signature_accession",
        "signature_description",
        "start",
        "stop",
        "score",
        "status",
        "date",
        "interpro_accession",
        "interpro_description",
        "go",
        "pathway",
    ]
    return df


def extract_pathway_groups(df):
    """Extract pathway groups from InterProScan results.

    Returns a dictionary mapping pathway IDs to sets of protein accessions.
    """
    pathway_groups = defaultdict(set)

    # Filter rows with pathway annotations
    pathway_rows = df[df["pathway"] != "-"]

    for _, row in pathway_rows.iterrows():
        # Split multiple pathway annotations
        pathways = row["pathway"].split("|")
        for pathway in pathways:
            pathway_groups[pathway].add(row["accession"])

    return pathway_groups


def load_rbh_mapping(blast_results_path):
    """Load reciprocal best hits mapping from BLAST results."""
    df = pd.read_csv(blast_results_path)

    # Create mapping dictionary from Crocosphaera to UCYN-A proteins
    mapping = {}

    # Get forward hits (Crocosphaera -> UCYN-A)
    forward_hits = df[df["direction"] == "crocosphaera -> ucyna"]
    for _, row in forward_hits.iterrows():
        # query_id is Crocosphaera, subject_id is UCYN-A
        mapping[row["query_id"]] = row["subject_id"]

    return mapping


def analyze_pathway_completeness(
    croco_pathway_groups, ucyna_pathway_groups, rbh_mapping
):
    """Analyze pathway completeness between the two organisms."""
    results = []

    for pathway, croco_proteins in croco_pathway_groups.items():
        # Get corresponding UCYN-A proteins for this pathway
        ucyna_proteins = ucyna_pathway_groups.get(pathway, set())

        # Map Crocosphaera proteins to their UCYN-A orthologs
        mapped_proteins = set()
        for protein in croco_proteins:
            if protein in rbh_mapping:
                mapped_proteins.add(rbh_mapping[protein])

        # Calculate completeness
        croco_count = len(croco_proteins)
        ucyna_count = len(mapped_proteins.union(ucyna_proteins))
        completeness = (ucyna_count / croco_count * 100) if croco_count > 0 else 0

        results.append(
            {
                "pathway": pathway,
                "crocosphaera_proteins": croco_count,
                "ucyna_proteins": ucyna_count,
                "completeness": completeness,
                "missing_count": croco_count - ucyna_count,
                "crocosphaera_protein_ids": ",".join(croco_proteins),
                "ucyna_protein_ids": ",".join(mapped_proteins.union(ucyna_proteins)),
            }
        )

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze pathway completeness between UCYN-A and Crocosphaera"
    )
    parser.add_argument(
        "--croco-interpro", required=True, help="Crocosphaera InterProScan results"
    )
    parser.add_argument(
        "--ucyna-interpro", required=True, help="UCYN-A InterProScan results"
    )
    parser.add_argument(
        "--blast-results", required=True, help="Bidirectional BLAST results"
    )
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument(
        "--min-proteins",
        type=int,
        default=2,
        help="Minimum number of proteins in pathway for analysis",
    )

    args = parser.parse_args()

    # Load data
    print("Loading InterProScan results...")
    croco_df = load_interpro_results(args.croco_interpro)
    ucyna_df = load_interpro_results(args.ucyna_interpro)

    print("Extracting pathway groups...")
    croco_pathway_groups = extract_pathway_groups(croco_df)
    ucyna_pathway_groups = extract_pathway_groups(ucyna_df)

    print("Loading reciprocal best hits...")
    rbh_mapping = load_rbh_mapping(args.blast_results)

    print("Analyzing pathway completeness...")
    results = analyze_pathway_completeness(
        croco_pathway_groups, ucyna_pathway_groups, rbh_mapping
    )

    # Filter pathways with minimum protein count
    results = results[results["crocosphaera_proteins"] >= args.min_proteins]

    # Sort by completeness and missing protein count
    results = results.sort_values(["completeness", "missing_count"])

    # Save results
    results.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

    # Print summary statistics
    print("\nSummary:")
    print(f"Total pathways analyzed: {len(results)}")
    print(
        f"Pathways with <50% completeness: {len(results[results['completeness'] < 50])}"
    )
    print(
        f"Pathways with >90% completeness: {len(results[results['completeness'] > 90])}"
    )
    print("\nMost incomplete pathways:")
    print(results[["pathway", "completeness", "missing_count"]].head())


if __name__ == "__main__":
    main()
