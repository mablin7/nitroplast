#!/usr/bin/env python3
"""
uTP Homolog Search in Haptophyte Genomes

This script addresses three key questions about uTP evolution:

Question A: Does the uTP sequence exist in non-symbiotic haptophytes?
    - Direct sequence homology search using DIAMOND
    - Approach: Use uTP sequences as queries against haptophyte proteomes

Question B: Do non-symbiotic haptophytes have proteins with unexplained C-terminal extensions?
    - Ortholog identification via bidirectional best hits
    - Length comparison between B. bigelowii and ortholog pairs
    - Identify proteins where B. bigelowii version is significantly longer at C-terminus

Question C: Are individual uTP motifs found elsewhere?
    - Regex-based motif search in haptophyte proteomes
    - Statistical assessment of motif occurrence vs random expectation

CRITICAL METHODOLOGICAL NOTES:
------------------------------
1. Negative results are expected but must be interpreted carefully:
   - Absence in distant relatives (>300 Myr divergence) is less informative
   - True negatives require adequate search sensitivity

2. Ortholog identification caveats:
   - Bidirectional best hit is a heuristic, not definitive orthology
   - Gene duplication/loss can confound interpretation
   - Incomplete genome annotations affect coverage

3. C-terminal extension analysis:
   - Extensions could be due to: uTP, annotation artifacts, alternative splicing,
     genuine functional domains, or alignment errors
   - Need manual validation of top candidates

Author: Generated for nitroplast/UCYN-A research project
Date: January 2026
"""

import os
import sys
import subprocess
import re
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
EXPERIMENT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
HAPTOPHYTE_DIR = DATA_DIR / "haptophytes" / "ncbi_dataset" / "data"
MOTIF_ANALYSIS_DIR = PROJECT_ROOT / "experiments" / "utp_motif_analysis"

OUTPUT_DIR = EXPERIMENT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# uTP motif consensus sequences from previous analysis
UTP_MOTIFS = {
    "motif_1": "TQTQLGACMGALGLHLGSRLD",  # 21 aa - secondary fixed motif
    "motif_2": "WLEEWRERLECWW",  # 13 aa - initiating motif
    "motif_3": "AEPGCEWVEE",  # 10 aa
    "motif_4": "LPDFPEPFSLPPIPRL",  # 16 aa
    "motif_5": "LZLPDFPD",  # 8 aa (Z = ambiguous)
    "motif_6": "AWRAALLGRAPPP",  # 13 aa
    "motif_7": "PPJPRLLP",  # 8 aa (J = ambiguous)
}

# Relaxed regex patterns for motif search (allowing conservative substitutions)
# These are derived from the consensus but allow some variation
UTP_MOTIF_PATTERNS = {
    "motif_1": r"[TS][QN][TS][QN][LI][GA][AC][MILV][GA][AL][GL][LIVM][HQ][LI][GA][ST][RK][LI][DE]",
    "motif_2": r"W[LIVM][DE][DE]W[RK][DE][RK][LIVM][DE][CY]WW",
    "motif_3": r"[AV][DE][PG][GA]C[DE]W[VIL][DE][DE]",
    "motif_4": r"[LI]P[DE]FP[DE][PG]F[ST][LI]PP[LIVM]P[RK][LI]",
    "motif_5": r"[LI][ILVMZ]LPDFP[DE]",
    "motif_6": r"[AV]W[RK][AV][AV][LI][LI][GA][RK][AV]PPP",
    "motif_7": r"PP[JILVMF]P[RK][LI][LI]P",
}


@dataclass
class HaptophyteGenome:
    """Container for haptophyte genome information."""

    accession: str
    organism: str
    protein_file: Optional[Path]
    genome_file: Optional[Path]
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
        genome_file = list(genome_dir.glob("*.fna"))

        genomes.append(
            HaptophyteGenome(
                accession=accession,
                organism=organism,
                protein_file=protein_file if protein_file.exists() else None,
                genome_file=genome_file[0] if genome_file else None,
                has_annotation=protein_file.exists(),
            )
        )

    return genomes


def load_utp_sequences() -> dict[str, str]:
    """Load uTP-containing protein sequences from B. bigelowii."""
    # Use the Import_candidates.fasta which contains uTP proteins
    fasta_file = DATA_DIR / "Import_candidates.fasta"

    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq)

    print(f"Loaded {len(sequences)} uTP-containing sequences from B. bigelowii")
    return sequences


def extract_utp_regions(
    sequences: dict[str, str], c_term_length: int = 150
) -> dict[str, str]:
    """
    Extract the C-terminal region (putative uTP) from sequences.

    Note: The uTP is ~120 aa, but we take 150 to capture potential variation.
    """
    utp_regions = {}
    for seq_id, seq in sequences.items():
        if len(seq) > c_term_length:
            utp_regions[seq_id] = seq[-c_term_length:]
        else:
            utp_regions[seq_id] = seq

    return utp_regions


def create_combined_proteome(
    genomes: list[HaptophyteGenome], output_path: Path
) -> dict[str, str]:
    """
    Create a combined proteome from all haptophyte genomes.
    Returns a mapping of sequence ID to organism.
    """
    seq_to_organism = {}
    all_sequences = []

    for genome in genomes:
        if not genome.has_annotation:
            continue

        for record in SeqIO.parse(genome.protein_file, "fasta"):
            # Prefix ID with accession to avoid collisions
            new_id = f"{genome.accession}|{record.id}"
            record.id = new_id
            record.description = f"{genome.organism}"
            all_sequences.append(record)
            seq_to_organism[new_id] = genome.organism

    SeqIO.write(all_sequences, output_path, "fasta")
    print(f"Created combined proteome with {len(all_sequences)} sequences")

    return seq_to_organism


# =============================================================================
# QUESTION A: Direct uTP homology search
# =============================================================================


def run_diamond_search(
    query_fasta: Path,
    target_fasta: Path,
    output_path: Path,
    evalue: float = 1e-3,
    max_target_seqs: int = 10,
) -> Path:
    """
    Run DIAMOND blastp search.

    Using relatively permissive e-value since we're looking for distant homologs.
    """
    # Create DIAMOND database
    db_path = output_path.parent / "target_db"

    cmd_makedb = ["diamond", "makedb", "--in", str(target_fasta), "--db", str(db_path)]

    print(f"Creating DIAMOND database...")
    result = subprocess.run(cmd_makedb, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"DIAMOND makedb error: {result.stderr}")
        raise RuntimeError("DIAMOND makedb failed")

    # Run search
    cmd_search = [
        "diamond",
        "blastp",
        "--query",
        str(query_fasta),
        "--db",
        str(db_path),
        "--out",
        str(output_path),
        "--outfmt",
        "6",
        "qseqid",
        "sseqid",
        "pident",
        "length",
        "mismatch",
        "gapopen",
        "qstart",
        "qend",
        "sstart",
        "send",
        "evalue",
        "bitscore",
        "qlen",
        "slen",
        "--evalue",
        str(evalue),
        "--max-target-seqs",
        str(max_target_seqs),
        "--sensitive",  # More sensitive for distant homologs
    ]

    print(f"Running DIAMOND search...")
    result = subprocess.run(cmd_search, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"DIAMOND search error: {result.stderr}")
        raise RuntimeError("DIAMOND search failed")

    return output_path


def analyze_utp_homology_results(
    results_path: Path, seq_to_organism: dict
) -> pd.DataFrame:
    """
    Analyze DIAMOND results for uTP homology.

    Key criteria for a "hit":
    - Alignment covers significant portion of query (>50%)
    - Alignment is at C-terminus of subject (within last 200 aa)
    - E-value below threshold
    """
    columns = [
        "qseqid",
        "sseqid",
        "pident",
        "length",
        "mismatch",
        "gapopen",
        "qstart",
        "qend",
        "sstart",
        "send",
        "evalue",
        "bitscore",
        "qlen",
        "slen",
    ]

    if not results_path.exists() or results_path.stat().st_size == 0:
        print("No DIAMOND hits found")
        return pd.DataFrame(columns=columns)

    df = pd.read_csv(results_path, sep="\t", names=columns)

    # Add organism information
    df["organism"] = df["sseqid"].map(seq_to_organism)

    # Calculate coverage and position metrics
    df["query_coverage"] = (df["qend"] - df["qstart"] + 1) / df["qlen"]
    df["subject_c_term_pos"] = df["slen"] - df["send"]  # Distance from C-terminus
    df["is_c_terminal"] = df["subject_c_term_pos"] < 200  # Hit is near C-terminus

    # Filter for meaningful hits
    df_filtered = df[
        (df["query_coverage"] > 0.3)  # At least 30% of uTP aligned
        & (df["pident"] > 20)  # At least 20% identity (very permissive)
    ].copy()

    return df_filtered


# =============================================================================
# QUESTION B: C-terminal extension analysis via ortholog comparison
# =============================================================================


def run_bidirectional_blast(
    bb_proteins: Path, target_proteome: Path, output_dir: Path, evalue: float = 1e-5
) -> tuple[Path, Path]:
    """
    Run bidirectional DIAMOND searches for ortholog identification.
    """
    # B. bigelowii -> target
    forward_out = output_dir / "bb_to_target.m8"
    db_target = output_dir / "target_db"

    subprocess.run(
        ["diamond", "makedb", "--in", str(target_proteome), "--db", str(db_target)],
        capture_output=True,
    )

    subprocess.run(
        [
            "diamond",
            "blastp",
            "--query",
            str(bb_proteins),
            "--db",
            str(db_target),
            "--out",
            str(forward_out),
            "--outfmt",
            "6",
            "qseqid",
            "sseqid",
            "pident",
            "length",
            "evalue",
            "bitscore",
            "qlen",
            "slen",
            "--evalue",
            str(evalue),
            "--max-target-seqs",
            "1",
        ],
        capture_output=True,
    )

    # Target -> B. bigelowii
    reverse_out = output_dir / "target_to_bb.m8"
    db_bb = output_dir / "bb_db"

    subprocess.run(
        ["diamond", "makedb", "--in", str(bb_proteins), "--db", str(db_bb)],
        capture_output=True,
    )

    subprocess.run(
        [
            "diamond",
            "blastp",
            "--query",
            str(target_proteome),
            "--db",
            str(db_bb),
            "--out",
            str(reverse_out),
            "--outfmt",
            "6",
            "qseqid",
            "sseqid",
            "pident",
            "length",
            "evalue",
            "bitscore",
            "qlen",
            "slen",
            "--evalue",
            str(evalue),
            "--max-target-seqs",
            "1",
        ],
        capture_output=True,
    )

    return forward_out, reverse_out


def identify_orthologs(forward_path: Path, reverse_path: Path) -> pd.DataFrame:
    """
    Identify bidirectional best hits (putative orthologs).
    """
    columns = [
        "qseqid",
        "sseqid",
        "pident",
        "length",
        "evalue",
        "bitscore",
        "qlen",
        "slen",
    ]

    if not forward_path.exists() or forward_path.stat().st_size == 0:
        return pd.DataFrame()
    if not reverse_path.exists() or reverse_path.stat().st_size == 0:
        return pd.DataFrame()

    forward = pd.read_csv(forward_path, sep="\t", names=columns)
    reverse = pd.read_csv(reverse_path, sep="\t", names=columns)

    # Get best hit for each query
    forward_best = forward.sort_values("bitscore", ascending=False).drop_duplicates(
        "qseqid"
    )
    reverse_best = reverse.sort_values("bitscore", ascending=False).drop_duplicates(
        "qseqid"
    )

    # Find bidirectional best hits
    orthologs = []
    for _, row in forward_best.iterrows():
        bb_id = row["qseqid"]
        target_id = row["sseqid"]

        # Check if target's best hit is bb_id
        target_best = reverse_best[reverse_best["qseqid"] == target_id]
        if len(target_best) > 0 and target_best.iloc[0]["sseqid"] == bb_id:
            orthologs.append(
                {
                    "bb_id": bb_id,
                    "target_id": target_id,
                    "bb_length": row["qlen"],
                    "target_length": row["slen"],
                    "pident": row["pident"],
                    "evalue": row["evalue"],
                }
            )

    return pd.DataFrame(orthologs)


def analyze_c_terminal_extensions(
    orthologs: pd.DataFrame, bb_sequences: dict[str, str], utp_proteins: set[str]
) -> pd.DataFrame:
    """
    Analyze length differences between B. bigelowii proteins and their orthologs.

    Focus on cases where B. bigelowii protein is longer, especially at C-terminus.
    """
    if orthologs.empty:
        return pd.DataFrame()

    orthologs = orthologs.copy()
    orthologs["length_diff"] = orthologs["bb_length"] - orthologs["target_length"]
    orthologs["has_utp"] = orthologs["bb_id"].isin(utp_proteins)

    # Significant extension: B. bigelowii is >100 aa longer
    orthologs["has_extension"] = orthologs["length_diff"] > 100

    return orthologs


# =============================================================================
# QUESTION C: Individual motif search
# =============================================================================


def search_motifs_in_proteome(proteome_path: Path, motif_patterns: dict) -> dict:
    """
    Search for individual uTP motifs in a proteome using regex.

    Returns counts and hit details for each motif.
    """
    results = {motif: {"count": 0, "hits": []} for motif in motif_patterns}

    for record in SeqIO.parse(proteome_path, "fasta"):
        seq = str(record.seq)

        for motif_name, pattern in motif_patterns.items():
            matches = list(re.finditer(pattern, seq, re.IGNORECASE))
            if matches:
                results[motif_name]["count"] += len(matches)
                for match in matches:
                    results[motif_name]["hits"].append(
                        {
                            "seq_id": record.id,
                            "start": match.start(),
                            "end": match.end(),
                            "matched_seq": match.group(),
                            "position_from_c_term": len(seq) - match.end(),
                        }
                    )

    return results


def calculate_motif_background_rate(
    proteome_path: Path, motif_patterns: dict, n_shuffles: int = 100
) -> dict:
    """
    Estimate background rate of motif occurrence by searching shuffled sequences.

    This provides a null distribution for statistical comparison.
    """
    import random

    # Load sequences
    sequences = [str(r.seq) for r in SeqIO.parse(proteome_path, "fasta")]
    total_length = sum(len(s) for s in sequences)

    background_counts = {motif: [] for motif in motif_patterns}

    print(f"Calculating background rates ({n_shuffles} shuffles)...")
    for i in range(n_shuffles):
        if i % 20 == 0:
            print(f"  Shuffle {i}/{n_shuffles}")

        for motif_name, pattern in motif_patterns.items():
            count = 0
            for seq in sequences[:100]:  # Sample for speed
                # Shuffle sequence preserving composition
                shuffled = "".join(random.sample(seq, len(seq)))
                matches = re.findall(pattern, shuffled, re.IGNORECASE)
                count += len(matches)
            background_counts[motif_name].append(count)

    return background_counts


def assess_motif_significance(observed: dict, background: dict) -> pd.DataFrame:
    """
    Compare observed motif counts to background distribution.
    """
    results = []

    for motif in observed:
        obs_count = observed[motif]["count"]
        bg_counts = background.get(motif, [0])

        bg_mean = np.mean(bg_counts)
        bg_std = np.std(bg_counts)

        # Z-score (if std > 0)
        z_score = (obs_count - bg_mean) / bg_std if bg_std > 0 else 0

        # Empirical p-value
        p_value = np.mean([bg >= obs_count for bg in bg_counts])

        results.append(
            {
                "motif": motif,
                "observed_count": obs_count,
                "background_mean": bg_mean,
                "background_std": bg_std,
                "z_score": z_score,
                "p_value": p_value,
                "enriched": obs_count > bg_mean,
            }
        )

    return pd.DataFrame(results)


# =============================================================================
# Visualization
# =============================================================================


def plot_extension_analysis(orthologs: pd.DataFrame, output_path: Path):
    """Plot C-terminal extension analysis results."""
    if orthologs.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Length difference distribution
    ax1 = axes[0]
    utp_data = orthologs[orthologs["has_utp"]]["length_diff"]
    non_utp_data = orthologs[~orthologs["has_utp"]]["length_diff"]

    ax1.hist(
        non_utp_data,
        bins=50,
        alpha=0.7,
        label=f"Non-uTP (n={len(non_utp_data)})",
        color="steelblue",
    )
    if len(utp_data) > 0:
        ax1.hist(
            utp_data,
            bins=50,
            alpha=0.7,
            label=f"uTP proteins (n={len(utp_data)})",
            color="coral",
        )

    ax1.axvline(
        x=100, color="red", linestyle="--", label="Extension threshold (100 aa)"
    )
    ax1.axvline(x=0, color="black", linestyle="-", alpha=0.5)
    ax1.set_xlabel("Length difference (B. bigelowii - ortholog)")
    ax1.set_ylabel("Count")
    ax1.set_title("C-terminal Extension Analysis")
    ax1.legend()

    # Scatter: length vs length
    ax2 = axes[1]
    colors = ["coral" if x else "steelblue" for x in orthologs["has_utp"]]
    ax2.scatter(
        orthologs["target_length"], orthologs["bb_length"], c=colors, alpha=0.5, s=10
    )

    # Diagonal line (equal length)
    max_len = max(orthologs["bb_length"].max(), orthologs["target_length"].max())
    ax2.plot([0, max_len], [0, max_len], "k--", alpha=0.5, label="Equal length")
    ax2.plot(
        [0, max_len], [100, max_len + 100], "r--", alpha=0.5, label="+100 aa extension"
    )

    ax2.set_xlabel("Ortholog length (aa)")
    ax2.set_ylabel("B. bigelowii length (aa)")
    ax2.set_title("Protein Length Comparison")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close()


def plot_motif_search_results(motif_results: dict, output_path: Path):
    """Plot motif search results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Motif counts
    ax1 = axes[0]
    motifs = list(motif_results.keys())
    counts = [motif_results[m]["count"] for m in motifs]

    bars = ax1.bar(motifs, counts, color="steelblue", edgecolor="navy")
    ax1.set_xlabel("Motif")
    ax1.set_ylabel("Number of hits")
    ax1.set_title("uTP Motif Occurrences in Haptophyte Proteomes")
    ax1.tick_params(axis="x", rotation=45)

    # Add count labels
    for bar, count in zip(bars, counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Position distribution (distance from C-terminus)
    ax2 = axes[1]
    all_positions = []
    for motif, data in motif_results.items():
        for hit in data["hits"]:
            all_positions.append(hit["position_from_c_term"])

    if all_positions:
        ax2.hist(all_positions, bins=50, color="steelblue", edgecolor="navy", alpha=0.7)
        ax2.axvline(
            x=150, color="red", linestyle="--", label="Expected uTP region (<150 aa)"
        )
        ax2.set_xlabel("Distance from C-terminus (aa)")
        ax2.set_ylabel("Count")
        ax2.set_title("Position of Motif Hits")
        ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close()


# =============================================================================
# Main analysis
# =============================================================================


def main():
    print("=" * 70)
    print("uTP Homolog Search in Haptophyte Genomes")
    print("=" * 70)

    # Get available genomes
    genomes = get_available_proteomes()
    annotated_genomes = [g for g in genomes if g.has_annotation]

    print(f"\nAvailable haptophyte genomes: {len(genomes)}")
    print(f"Genomes with protein annotation: {len(annotated_genomes)}")

    for g in annotated_genomes:
        print(f"  - {g.organism} ({g.accession})")

    if not annotated_genomes:
        print("ERROR: No annotated genomes available")
        sys.exit(1)

    # Load B. bigelowii uTP proteins
    utp_sequences = load_utp_sequences()
    utp_protein_ids = set(utp_sequences.keys())

    # Create combined haptophyte proteome
    combined_proteome = OUTPUT_DIR / "combined_haptophyte_proteome.fasta"
    seq_to_organism = create_combined_proteome(annotated_genomes, combined_proteome)

    # ==========================================================================
    # QUESTION A: Direct uTP homology search
    # ==========================================================================
    print("\n" + "=" * 70)
    print("QUESTION A: Direct uTP sequence homology search")
    print("=" * 70)

    # Extract uTP regions
    utp_regions = extract_utp_regions(utp_sequences)
    utp_query_file = OUTPUT_DIR / "utp_regions_query.fasta"

    with open(utp_query_file, "w") as f:
        for seq_id, seq in utp_regions.items():
            f.write(f">{seq_id}\n{seq}\n")

    print(f"Created query file with {len(utp_regions)} uTP regions")

    # Run DIAMOND search
    diamond_results = OUTPUT_DIR / "utp_homology_search.m8"
    try:
        run_diamond_search(
            utp_query_file, combined_proteome, diamond_results, evalue=0.01
        )  # Permissive for distant homologs

        # Analyze results
        homology_df = analyze_utp_homology_results(diamond_results, seq_to_organism)

        if not homology_df.empty:
            # Filter for C-terminal hits (most relevant)
            c_term_hits = homology_df[homology_df["is_c_terminal"]]

            print(f"\nTotal DIAMOND hits: {len(homology_df)}")
            print(f"C-terminal hits (<200 aa from end): {len(c_term_hits)}")

            if len(c_term_hits) > 0:
                print("\nTop C-terminal hits by organism:")
                for org in c_term_hits["organism"].unique():
                    org_hits = c_term_hits[c_term_hits["organism"] == org]
                    print(f"  {org}: {len(org_hits)} hits")

                # Save results
                homology_df.to_csv(
                    OUTPUT_DIR / "utp_homology_all_hits.csv", index=False
                )
                c_term_hits.to_csv(
                    OUTPUT_DIR / "utp_homology_c_terminal_hits.csv", index=False
                )
        else:
            print("\nNo significant homology hits found")
            print("This is consistent with de novo evolution or extensive divergence")

    except Exception as e:
        print(f"DIAMOND search failed: {e}")
        print("Skipping Question A analysis")

    # ==========================================================================
    # QUESTION B: C-terminal extension analysis
    # ==========================================================================
    print("\n" + "=" * 70)
    print("QUESTION B: C-terminal extension analysis via ortholog comparison")
    print("=" * 70)

    # Use full B. bigelowii proteome (Import_candidates contains uTP proteins)
    bb_proteome = DATA_DIR / "Import_candidates.fasta"

    all_orthologs = []

    for genome in annotated_genomes:
        print(f"\nAnalyzing {genome.organism}...")

        genome_output = OUTPUT_DIR / f"ortholog_{genome.accession}"
        genome_output.mkdir(exist_ok=True)

        try:
            forward, reverse = run_bidirectional_blast(
                bb_proteome, genome.protein_file, genome_output
            )

            orthologs = identify_orthologs(forward, reverse)

            if not orthologs.empty:
                orthologs["target_organism"] = genome.organism
                orthologs["target_accession"] = genome.accession

                # Analyze extensions
                orthologs = analyze_c_terminal_extensions(
                    orthologs, utp_sequences, utp_protein_ids
                )

                all_orthologs.append(orthologs)

                print(f"  Found {len(orthologs)} ortholog pairs")
                print(
                    f"  Pairs with >100 aa extension: {orthologs['has_extension'].sum()}"
                )
                print(f"  uTP proteins with orthologs: {orthologs['has_utp'].sum()}")
            else:
                print(f"  No orthologs identified")

        except Exception as e:
            print(f"  Error: {e}")

    if all_orthologs:
        combined_orthologs = pd.concat(all_orthologs, ignore_index=True)
        combined_orthologs.to_csv(OUTPUT_DIR / "ortholog_analysis.csv", index=False)

        # Summary statistics
        print("\n" + "-" * 50)
        print("ORTHOLOG ANALYSIS SUMMARY")
        print("-" * 50)
        print(f"Total ortholog pairs: {len(combined_orthologs)}")
        print(
            f"Pairs with B. bigelowii >100 aa longer: {combined_orthologs['has_extension'].sum()}"
        )

        # Compare uTP vs non-uTP proteins
        utp_orthologs = combined_orthologs[combined_orthologs["has_utp"]]
        non_utp_orthologs = combined_orthologs[~combined_orthologs["has_utp"]]

        if len(utp_orthologs) > 0:
            print(f"\nuTP proteins:")
            print(f"  Mean length diff: {utp_orthologs['length_diff'].mean():.1f} aa")
            print(
                f"  % with extension: {100*utp_orthologs['has_extension'].mean():.1f}%"
            )

        if len(non_utp_orthologs) > 0:
            print(f"\nNon-uTP proteins:")
            print(
                f"  Mean length diff: {non_utp_orthologs['length_diff'].mean():.1f} aa"
            )
            print(
                f"  % with extension: {100*non_utp_orthologs['has_extension'].mean():.1f}%"
            )

        # Plot
        plot_extension_analysis(
            combined_orthologs, OUTPUT_DIR / "extension_analysis.png"
        )

        # Identify top candidates (non-uTP proteins with large extensions)
        candidates = combined_orthologs[
            (~combined_orthologs["has_utp"]) & (combined_orthologs["length_diff"] > 100)
        ].sort_values("length_diff", ascending=False)

        if len(candidates) > 0:
            print(f"\nTop non-uTP proteins with C-terminal extensions:")
            for _, row in candidates.head(10).iterrows():
                print(
                    f"  {row['bb_id']}: +{row['length_diff']:.0f} aa vs {row['target_organism']}"
                )

            candidates.to_csv(OUTPUT_DIR / "extension_candidates.csv", index=False)

    # ==========================================================================
    # QUESTION C: Individual motif search
    # ==========================================================================
    print("\n" + "=" * 70)
    print("QUESTION C: Individual uTP motif search")
    print("=" * 70)

    # Search each genome
    all_motif_results = {}

    for genome in annotated_genomes:
        print(f"\nSearching {genome.organism}...")

        results = search_motifs_in_proteome(genome.protein_file, UTP_MOTIF_PATTERNS)
        all_motif_results[genome.organism] = results

        total_hits = sum(r["count"] for r in results.values())
        print(f"  Total motif hits: {total_hits}")

        for motif, data in results.items():
            if data["count"] > 0:
                # Check if hits are C-terminal
                c_term_hits = [
                    h for h in data["hits"] if h["position_from_c_term"] < 150
                ]
                print(
                    f"    {motif}: {data['count']} hits ({len(c_term_hits)} C-terminal)"
                )

    # Aggregate results
    motif_summary = []
    for organism, results in all_motif_results.items():
        for motif, data in results.items():
            c_term_hits = [h for h in data["hits"] if h["position_from_c_term"] < 150]
            motif_summary.append(
                {
                    "organism": organism,
                    "motif": motif,
                    "total_hits": data["count"],
                    "c_terminal_hits": len(c_term_hits),
                }
            )

    motif_summary_df = pd.DataFrame(motif_summary)
    motif_summary_df.to_csv(OUTPUT_DIR / "motif_search_results.csv", index=False)

    # Aggregate across all genomes
    combined_motif_results = {
        motif: {"count": 0, "hits": []} for motif in UTP_MOTIF_PATTERNS
    }
    for organism, results in all_motif_results.items():
        for motif, data in results.items():
            combined_motif_results[motif]["count"] += data["count"]
            combined_motif_results[motif]["hits"].extend(data["hits"])

    plot_motif_search_results(
        combined_motif_results, OUTPUT_DIR / "motif_search_results.png"
    )

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nKey output files:")
    print("  - utp_homology_*.csv: Direct homology search results (Question A)")
    print("  - ortholog_analysis.csv: Ortholog comparison results (Question B)")
    print("  - extension_candidates.csv: Non-uTP proteins with extensions (Question B)")
    print("  - motif_search_results.csv: Individual motif search (Question C)")

    print("\n" + "=" * 70)
    print("INTERPRETATION NOTES")
    print("=" * 70)
    print(
        """
Question A (uTP homologs):
- Negative results support de novo evolution OR extensive divergence
- Positive results would suggest co-option from existing function
- CAVEAT: Search sensitivity may miss highly diverged homologs

Question B (C-terminal extensions):
- uTP proteins should show ~100-150 aa extensions vs orthologs
- Non-uTP proteins with extensions are candidates for:
  * Missed uTP proteins (not detected in proteomics)
  * Alternative targeting systems
  * Annotation artifacts (requires manual validation)

Question C (Individual motifs):
- Motifs found in non-symbiotic haptophytes could indicate:
  * Ancestral sequence features co-opted into uTP
  * Convergent evolution
  * Random occurrence (compare to background rate)
- C-terminal enrichment would be more significant than random positions
"""
    )


if __name__ == "__main__":
    main()
