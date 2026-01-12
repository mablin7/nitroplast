#!/usr/bin/env python3
"""
Question C: Individual uTP Motif Search using MEME Suite's MAST

This script uses MAST (Motif Alignment & Search Tool) to search for uTP motifs
in haptophyte proteomes. MAST uses position weight matrices (PWMs) from MEME
output, providing much more sensitive and statistically rigorous motif detection
than simple regex patterns.

Key advantages over regex approach:
1. Uses full position weight matrices (PWMs) capturing position-specific amino acid preferences
2. Calculates proper statistical significance (E-values, p-values)
3. Accounts for background amino acid frequencies
4. Handles degenerate positions properly

Author: Generated for nitroplast/UCYN-A research project
Date: January 2026
"""

import os
import subprocess
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
EXPERIMENT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
HAPTOPHYTE_DIR = DATA_DIR / "haptophytes" / "ncbi_dataset" / "data"
MOTIF_ANALYSIS_DIR = PROJECT_ROOT / "experiments" / "utp_motif_analysis"

OUTPUT_DIR = EXPERIMENT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# MEME motif file from previous analysis
MEME_XML = MOTIF_ANALYSIS_DIR / "data" / "meme_gb.xml"

# MAST executable
MAST_PATH = "/opt/local/bin/mast"


@dataclass
class HaptophyteGenome:
    """Container for haptophyte genome information."""

    accession: str
    organism: str
    protein_file: Optional[Path]
    has_annotation: bool


def load_genome_metadata() -> pd.DataFrame:
    """Load genome metadata from NCBI dataset."""
    summary_file = HAPTOPHYTE_DIR / "data_summary.tsv"
    df = pd.read_csv(summary_file, sep="\t")
    return df


def get_available_proteomes() -> list[HaptophyteGenome]:
    """Identify which haptophyte genomes have protein annotations."""
    metadata = load_genome_metadata()
    genomes = []

    for _, row in metadata.iterrows():
        accession = row["Assembly Accession"]
        organism = row["Organism Scientific Name"]

        genome_dir = HAPTOPHYTE_DIR / accession
        if not genome_dir.exists():
            continue

        protein_file = genome_dir / "protein.faa"

        genomes.append(
            HaptophyteGenome(
                accession=accession,
                organism=organism,
                protein_file=protein_file if protein_file.exists() else None,
                has_annotation=protein_file.exists(),
            )
        )

    return genomes


def run_mast_search(
    meme_file: Path,
    sequence_file: Path,
    output_dir: Path,
    evalue_threshold: float = 10.0,
) -> Path:
    """
    Run MAST to search for motifs in a sequence file.

    Args:
        meme_file: Path to MEME XML file with motif definitions
        sequence_file: Path to FASTA file to search
        output_dir: Directory for output files
        evalue_threshold: E-value threshold for reporting hits

    Returns:
        Path to the MAST output directory
    """
    # MAST won't overwrite existing output, so remove it first
    import shutil

    if output_dir.exists():
        shutil.rmtree(output_dir)

    cmd = [
        MAST_PATH,
        str(meme_file),
        str(sequence_file),
        "-o",
        str(output_dir),
        "-ev",
        str(evalue_threshold),
        "-nostatus",  # Suppress progress messages
    ]

    print(f"Running MAST on {sequence_file.name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 and "already exists" not in result.stderr:
        print(f"MAST warning: {result.stderr[:200] if result.stderr else 'none'}")

    return output_dir


def parse_mast_results(mast_output_dir: Path) -> pd.DataFrame:
    """
    Parse MAST results from the hit_list.txt file.

    Returns DataFrame with columns:
        - sequence_name: Name of the sequence with hit
        - strand: + or - (always + for proteins)
        - motif_id: Which motif was found
        - start: Start position of hit
        - end: End position of hit
        - score: MAST score
        - pvalue: p-value of hit
    """
    hit_list_file = mast_output_dir / "mast.txt"

    if not hit_list_file.exists():
        print(f"No hit list found at {hit_list_file}")
        return pd.DataFrame()

    # Parse the hit list format
    # Format: sequence_name strand motif_id start end score p-value
    hits = []

    with open(hit_list_file) as f:
        in_hits_section = False
        for line in f:
            line = line.strip()

            # Skip header lines
            if line.startswith("#"):
                if "sequence_name" in line:
                    in_hits_section = True
                continue

            if not line or not in_hits_section:
                continue

            # Parse hit line
            parts = line.split()
            if len(parts) >= 7:
                hits.append(
                    {
                        "sequence_name": parts[0],
                        "strand": parts[1],
                        "motif_id": parts[2],
                        "start": int(parts[3]),
                        "end": int(parts[4]),
                        "score": float(parts[5]),
                        "pvalue": float(parts[6]),
                    }
                )

    return pd.DataFrame(hits)


def parse_mast_xml(mast_output_dir: Path) -> pd.DataFrame:
    """
    Parse MAST XML output for more detailed results.
    """
    xml_file = mast_output_dir / "mast.xml"

    if not xml_file.exists():
        print(f"No XML output found at {xml_file}")
        return pd.DataFrame()

    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Build motif index -> info mapping
    # MAST uses idx (0-based index) to reference motifs
    motif_list = root.findall(".//motifs/motif")
    motif_by_idx = {}
    for i, motif in enumerate(motif_list):
        motif_by_idx[str(i)] = {
            "id": motif.get("id"),
            "alt": motif.get("alt"),
            "length": int(motif.get("length", 0)),
        }

    hits = []

    # Find all sequences with hits
    for seq in root.findall(".//sequence"):
        seq_name = seq.get("name")
        seq_length = int(seq.get("length", 0))

        # Get sequence-level score element
        score_elem = seq.find("score")
        seq_evalue = (
            float(score_elem.get("evalue", 1.0)) if score_elem is not None else 1.0
        )

        # Get individual motif hits within this sequence
        for seg in seq.findall(".//seg"):
            for hit in seg.findall(".//hit"):
                motif_idx = hit.get("idx")  # MAST uses idx, not motif
                pos = int(hit.get("pos", 0))
                pvalue = float(hit.get("pvalue", 1.0))
                match = hit.get("match", "")

                # Get motif info from index
                motif_info = motif_by_idx.get(motif_idx, {})
                motif_id = motif_info.get("id", f"motif_{motif_idx}")
                motif_alt = motif_info.get("alt", "")
                motif_width = motif_info.get("length", 0)

                hits.append(
                    {
                        "sequence_name": seq_name,
                        "sequence_length": seq_length,
                        "sequence_evalue": seq_evalue,
                        "motif_idx": motif_idx,
                        "motif_id": motif_id,
                        "motif_alt": motif_alt,
                        "motif_width": motif_width,
                        "position": pos,
                        "position_from_cterm": seq_length - pos - motif_width,
                        "pvalue": pvalue,
                        "match": match,
                    }
                )

    return pd.DataFrame(hits)


def analyze_c_terminal_hits(
    df: pd.DataFrame, c_term_threshold: int = 150
) -> pd.DataFrame:
    """
    Filter for hits near the C-terminus (where uTP would be).

    Args:
        df: DataFrame of MAST hits
        c_term_threshold: Maximum distance from C-terminus to consider

    Returns:
        Filtered DataFrame with only C-terminal hits
    """
    if df.empty:
        return df

    return df[df["position_from_cterm"] <= c_term_threshold].copy()


def get_motif_info_from_meme(meme_file: Path) -> dict:
    """Extract motif information from MEME XML file."""
    tree = ET.parse(meme_file)
    root = tree.getroot()

    motifs = {}
    for motif in root.findall(".//motif"):
        motif_id = motif.get("id")
        motifs[motif_id] = {
            "name": motif.get("name"),
            "width": int(motif.get("width", 0)),
            "sites": int(motif.get("sites", 0)),
            "evalue": float(motif.get("e_value", 1.0)),
            "ic": float(motif.get("ic", 0)),  # Information content
        }

    return motifs


def run_analysis():
    """Main analysis pipeline."""
    print("=" * 70)
    print("Question C: Individual uTP Motif Search using MAST")
    print("=" * 70)

    # Check MAST is available
    result = subprocess.run([MAST_PATH, "--version"], capture_output=True, text=True)
    print(f"MAST version: {result.stdout.strip()}")

    # Load motif information
    print("\n--- Loading motif information from MEME output ---")
    motif_info = get_motif_info_from_meme(MEME_XML)
    print(f"Found {len(motif_info)} motifs:")
    for mid, info in motif_info.items():
        print(
            f"  {mid}: {info['name']} (width={info['width']}, sites={info['sites']}, E={info['evalue']:.2e})"
        )

    # Get available proteomes
    print("\n--- Loading haptophyte proteomes ---")
    genomes = get_available_proteomes()
    annotated_genomes = [g for g in genomes if g.has_annotation]
    print(f"Found {len(annotated_genomes)} genomes with protein annotations:")
    for g in annotated_genomes:
        print(f"  {g.accession}: {g.organism}")

    # Run MAST on each proteome
    all_results = []

    for genome in annotated_genomes:
        print(f"\n--- Searching {genome.organism} ({genome.accession}) ---")

        # Count proteins
        n_proteins = sum(1 for _ in SeqIO.parse(genome.protein_file, "fasta"))
        print(f"  Proteins: {n_proteins:,}")

        # Run MAST
        mast_output = OUTPUT_DIR / f"mast_{genome.accession}"

        # Remove existing output if present (MAST won't overwrite)
        if mast_output.exists():
            import shutil

            shutil.rmtree(mast_output)

        run_mast_search(
            meme_file=MEME_XML,
            sequence_file=genome.protein_file,
            output_dir=mast_output,
            evalue_threshold=10.0,  # Permissive threshold for initial search
        )

        # Parse results
        results = parse_mast_xml(mast_output)

        if not results.empty:
            results["accession"] = genome.accession
            results["organism"] = genome.organism
            all_results.append(results)

            print(f"  Total motif hits: {len(results)}")
            print(f"  Unique sequences with hits: {results['sequence_name'].nunique()}")

            # Summarize by motif
            motif_counts = results.groupby("motif_alt").size()
            for motif, count in motif_counts.items():
                print(f"    {motif}: {count} hits")
        else:
            print("  No hits found")

    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
    else:
        combined_df = pd.DataFrame()

    print("\n" + "=" * 70)
    print("SUMMARY OF ALL MAST RESULTS")
    print("=" * 70)

    if combined_df.empty:
        print("NO MOTIF HITS FOUND IN ANY HAPTOPHYTE PROTEOME")
    else:
        print(f"Total hits across all proteomes: {len(combined_df)}")
        print(f"Unique sequences with hits: {combined_df['sequence_name'].nunique()}")

        # Summarize by organism and motif
        print("\n--- Hits by organism ---")
        org_counts = combined_df.groupby("organism").size()
        for org, count in org_counts.items():
            print(f"  {org}: {count} hits")

        print("\n--- Hits by motif ---")
        motif_counts = combined_df.groupby("motif_alt").size()
        for motif, count in motif_counts.items():
            print(f"  {motif}: {count} hits")

        # Analyze C-terminal hits specifically
        print("\n--- C-terminal hits (within 150 aa of C-terminus) ---")
        c_term_hits = analyze_c_terminal_hits(combined_df, c_term_threshold=150)

        if c_term_hits.empty:
            print("No C-terminal hits found")
        else:
            print(f"C-terminal hits: {len(c_term_hits)}")

            # Filter for significant hits
            significant_cterm = c_term_hits[c_term_hits["pvalue"] < 0.0001]
            print(f"Significant C-terminal hits (p < 0.0001): {len(significant_cterm)}")

            if not significant_cterm.empty:
                print("\nTop C-terminal hits:")
                top_hits = significant_cterm.nsmallest(20, "pvalue")
                for _, hit in top_hits.iterrows():
                    print(f"  {hit['organism']}: {hit['sequence_name']}")
                    print(
                        f"    Motif: {hit['motif_alt']} ({hit['motif_id'][:20]}...), pos={hit['position']}, "
                        f"dist_from_cterm={hit['position_from_cterm']}, p={hit['pvalue']:.2e}"
                    )

        # Statistical analysis: compare to expected by chance
        print("\n--- Statistical Analysis ---")

        # For each motif, calculate expected hits by chance
        total_positions = 0
        for genome in annotated_genomes:
            for record in SeqIO.parse(genome.protein_file, "fasta"):
                total_positions += len(record.seq)

        print(f"Total amino acid positions searched: {total_positions:,}")

        # Key question: Are these hits enriched at C-terminus like in uTP proteins?
        print("\n--- C-terminal Enrichment Analysis ---")
        print("Comparing hit positions to uTP proteins (positive control):")

        # In positive control, what fraction of hits are C-terminal?
        positive_control_file = OUTPUT_DIR / "mast_positive_control.csv"
        if positive_control_file.exists():
            pos_ctrl = pd.read_csv(positive_control_file)
            pos_ctrl_cterm = pos_ctrl[pos_ctrl["position_from_cterm"] <= 150]
            pos_ctrl_cterm_frac = (
                len(pos_ctrl_cterm) / len(pos_ctrl) if len(pos_ctrl) > 0 else 0
            )
            print(
                f"  Positive control (uTP proteins): {len(pos_ctrl_cterm)}/{len(pos_ctrl)} = {pos_ctrl_cterm_frac:.1%} C-terminal"
            )

        # In haptophyte hits, what fraction are C-terminal?
        hapto_cterm_frac = (
            len(c_term_hits) / len(combined_df) if len(combined_df) > 0 else 0
        )
        print(
            f"  Haptophyte hits: {len(c_term_hits)}/{len(combined_df)} = {hapto_cterm_frac:.1%} C-terminal"
        )

        # Expected by chance (uniform distribution)
        # If hits were random, ~150/avg_protein_length would be C-terminal
        avg_protein_length = total_positions / sum(
            1
            for g in annotated_genomes
            if g.has_annotation
            for _ in SeqIO.parse(g.protein_file, "fasta")
        )
        expected_cterm_frac = 150 / avg_protein_length
        print(f"  Expected by chance (uniform): ~{expected_cterm_frac:.1%} C-terminal")

        print("\n--- Key Motif Analysis ---")
        # Focus on the core uTP motifs (MEME-1 and MEME-2) which are most specific
        core_motifs = ["MEME-1", "MEME-2"]  # These are the initiating motifs
        core_hits = combined_df[combined_df["motif_alt"].isin(core_motifs)]
        print(f"Core uTP motifs (MEME-1, MEME-2) hits: {len(core_hits)}")
        if len(core_hits) > 0:
            core_cterm = core_hits[core_hits["position_from_cterm"] <= 150]
            print(f"  C-terminal core motif hits: {len(core_cterm)}")
            print(f"  These represent potential uTP-like sequences")

            if len(core_cterm) > 0:
                print("\n  Proteins with C-terminal core motif hits:")
                for _, hit in core_cterm.iterrows():
                    print(
                        f"    {hit['organism']}: {hit['sequence_name']} - {hit['motif_alt']} (p={hit['pvalue']:.2e})"
                    )

        print("\n--- Per-Motif Statistics ---")
        for motif_id, info in motif_info.items():
            motif_name = info["name"]
            # Match by motif_id which contains the consensus sequence
            observed = len(combined_df[combined_df["motif_id"] == motif_name])

            print(f"\n{motif_name}:")
            print(f"  Observed hits: {observed}")
            print(f"  Original E-value in uTP sequences: {info['evalue']:.2e}")

        # Save results
        combined_df.to_csv(OUTPUT_DIR / "mast_all_hits.csv", index=False)

        if not c_term_hits.empty:
            c_term_hits.to_csv(OUTPUT_DIR / "mast_c_terminal_hits.csv", index=False)

    # Create visualization
    create_visualizations(combined_df, motif_info, annotated_genomes)

    return combined_df


def create_visualizations(df: pd.DataFrame, motif_info: dict, genomes: list):
    """Create summary visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Color palette
    colors = sns.color_palette("husl", len(motif_info))
    motif_colors = {
        info["name"]: colors[i] for i, (_, info) in enumerate(motif_info.items())
    }

    # 1. Hits by motif
    ax1 = axes[0, 0]
    if not df.empty and "motif_alt" in df.columns:
        motif_counts = df.groupby("motif_alt").size().sort_values(ascending=True)
        bars = ax1.barh(range(len(motif_counts)), motif_counts.values)
        ax1.set_yticks(range(len(motif_counts)))
        ax1.set_yticklabels(motif_counts.index)
        for i, (motif, count) in enumerate(motif_counts.items()):
            bars[i].set_color(motif_colors.get(motif, "gray"))
    ax1.set_xlabel("Number of hits")
    ax1.set_title("MAST Hits by Motif")

    # 2. Hits by organism
    ax2 = axes[0, 1]
    if not df.empty:
        org_counts = df.groupby("organism").size().sort_values(ascending=True)
        ax2.barh(range(len(org_counts)), org_counts.values, color="steelblue")
        ax2.set_yticks(range(len(org_counts)))
        ax2.set_yticklabels([o.replace("_", " ") for o in org_counts.index], fontsize=8)
    ax2.set_xlabel("Number of hits")
    ax2.set_title("MAST Hits by Organism")

    # 3. P-value distribution
    ax3 = axes[1, 0]
    if not df.empty and "pvalue" in df.columns:
        ax3.hist(np.log10(df["pvalue"]), bins=50, color="steelblue", edgecolor="white")
        ax3.axvline(np.log10(0.0001), color="red", linestyle="--", label="p=0.0001")
        ax3.axvline(np.log10(0.01), color="orange", linestyle="--", label="p=0.01")
        ax3.legend()
    ax3.set_xlabel("log10(p-value)")
    ax3.set_ylabel("Count")
    ax3.set_title("P-value Distribution of Hits")

    # 4. Position relative to C-terminus
    ax4 = axes[1, 1]
    if (
        not df.empty
        and "position_from_cterm" in df.columns
        and "motif_alt" in df.columns
    ):
        for motif_alt in df["motif_alt"].unique():
            motif_data = df[df["motif_alt"] == motif_alt]["position_from_cterm"]
            ax4.hist(
                motif_data,
                bins=30,
                alpha=0.5,
                label=motif_alt,
                color=motif_colors.get(motif_alt, "gray"),
            )
        ax4.axvline(150, color="red", linestyle="--", label="uTP region boundary")
        ax4.legend(fontsize=7, loc="upper right")
    ax4.set_xlabel("Distance from C-terminus (aa)")
    ax4.set_ylabel("Count")
    ax4.set_title("Hit Position Relative to C-terminus")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "mast_motif_search_results.png", dpi=150, bbox_inches="tight"
    )
    plt.savefig(OUTPUT_DIR / "mast_motif_search_results.svg", bbox_inches="tight")
    plt.close()

    print(f"\nVisualization saved to {OUTPUT_DIR / 'mast_motif_search_results.png'}")


def compare_with_known_utp():
    """
    Compare MAST hits with known uTP proteins to validate the search.
    Run MAST on B. bigelowii uTP proteins as positive control.
    """
    print("\n" + "=" * 70)
    print("POSITIVE CONTROL: MAST on known uTP proteins")
    print("=" * 70)

    # Load uTP sequences
    utp_fasta = DATA_DIR / "uTP_HMM_hits.fasta"
    if not utp_fasta.exists():
        utp_fasta = DATA_DIR / "Import_candidates.fasta"

    n_utp = sum(1 for _ in SeqIO.parse(utp_fasta, "fasta"))
    print(f"Testing on {n_utp} known uTP proteins")

    # Run MAST
    mast_output = OUTPUT_DIR / "mast_positive_control"
    if mast_output.exists():
        import shutil

        shutil.rmtree(mast_output)

    run_mast_search(
        meme_file=MEME_XML,
        sequence_file=utp_fasta,
        output_dir=mast_output,
        evalue_threshold=10.0,
    )

    # Parse results
    results = parse_mast_xml(mast_output)

    if not results.empty:
        print(f"\nPositive control results:")
        print(f"  Total motif hits: {len(results)}")
        print(f"  Sequences with hits: {results['sequence_name'].nunique()} / {n_utp}")

        # Summarize by motif
        print("\n  Hits by motif:")
        motif_counts = results.groupby("motif_alt").size().sort_values(ascending=False)
        for motif, count in motif_counts.items():
            print(f"    {motif}: {count} hits")

        # Check C-terminal enrichment
        c_term_hits = analyze_c_terminal_hits(results, c_term_threshold=150)
        print(
            f"\n  C-terminal hits (within 150 aa): {len(c_term_hits)} / {len(results)}"
        )

        results.to_csv(OUTPUT_DIR / "mast_positive_control.csv", index=False)
    else:
        print("  WARNING: No hits in positive control!")

    return results


if __name__ == "__main__":
    # Run positive control first
    positive_results = compare_with_known_utp()

    # Run main analysis
    results = run_analysis()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput files in: {OUTPUT_DIR}")
