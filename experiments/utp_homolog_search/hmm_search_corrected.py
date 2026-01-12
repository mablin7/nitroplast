#!/usr/bin/env python3
"""
CORRECTED HMM-based uTP Homolog Search

CRITICAL CORRECTION:
The HMM consensus shows that the uTP motifs are at the BEGINNING of the HMM:
- Position 1-100: uTP region (contains motif_2, motif_1, motif_3, motif_4, etc.)
- Position 100-713: Mature domain

This is because the HMM was built from C-terminal protein sequences where the
uTP is at the N-terminus of the extracted fragment.

TRUE uTP homologs should match HMM positions 1-100, NOT 620-713.

Author: Generated for nitroplast/UCYN-A research project
Date: January 2026
"""

import subprocess
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from Bio import SeqIO

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
EXPERIMENT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
HAPTOPHYTE_DIR = DATA_DIR / "haptophytes" / "ncbi_dataset" / "data"

HMM_PROFILE = EXPERIMENT_DIR / "utp.hmm"
OUTPUT_DIR = EXPERIMENT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# CORRECTED: uTP region is at the START of the HMM
HMM_LENGTH = 713
UTP_REGION_END = 100  # uTP motifs are in positions 1-100
MATURE_DOMAIN_START = 100  # Mature domain is positions 100-713


@dataclass
class HMMHit:
    """Container for hmmsearch hit."""

    target_name: str
    target_accession: str
    query_name: str
    full_evalue: float
    full_score: float
    domain_evalue: float
    domain_score: float
    domain_num: int
    domain_of: int
    hmm_from: int
    hmm_to: int
    ali_from: int
    ali_to: int
    env_from: int
    env_to: int
    acc: float
    description: str


def parse_hmmsearch_domtblout(domtblout_path: Path) -> list[HMMHit]:
    """Parse hmmsearch domain table output."""
    hits = []

    with open(domtblout_path) as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 23:
                continue

            try:
                hit = HMMHit(
                    target_name=parts[0],
                    target_accession=parts[1],
                    query_name=parts[3],
                    full_evalue=float(parts[6]),
                    full_score=float(parts[7]),
                    domain_evalue=float(parts[12]),
                    domain_score=float(parts[13]),
                    domain_num=int(parts[9]),
                    domain_of=int(parts[10]),
                    hmm_from=int(parts[15]),
                    hmm_to=int(parts[16]),
                    ali_from=int(parts[17]),
                    ali_to=int(parts[18]),
                    env_from=int(parts[19]),
                    env_to=int(parts[20]),
                    acc=float(parts[21]),
                    description=" ".join(parts[22:]) if len(parts) > 22 else "",
                )
                hits.append(hit)
            except (ValueError, IndexError):
                continue

    return hits


def get_protein_lengths(fasta_path: Path) -> dict[str, int]:
    """Get lengths of all proteins in a FASTA file."""
    lengths = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        lengths[record.id] = len(record.seq)
    return lengths


def analyze_hmm_hits_corrected(
    hits: list[HMMHit], protein_lengths: dict[str, int]
) -> pd.DataFrame:
    """
    Analyze HMM hits with CORRECTED position interpretation.

    A TRUE uTP homolog must:
    1. Match the N-terminal portion of the HMM (positions 1-100, the uTP region)
    2. Be at the C-terminus of the target protein (where uTP should be)
    """
    records = []

    for hit in hits:
        target_len = protein_lengths.get(hit.target_name, 0)
        dist_from_c_term = target_len - hit.ali_to if target_len > 0 else -1

        # Calculate what portion of the hit is in the uTP region (positions 1-100)
        hmm_match_length = hit.hmm_to - hit.hmm_from + 1

        # How much of the hit is in the uTP region (positions 1-100)?
        utp_overlap_start = max(1, hit.hmm_from)
        utp_overlap_end = min(UTP_REGION_END, hit.hmm_to)
        utp_overlap = max(0, utp_overlap_end - utp_overlap_start + 1)

        # Fraction of hit in uTP region
        utp_fraction = utp_overlap / hmm_match_length if hmm_match_length > 0 else 0

        # Does the match START in the uTP region?
        starts_in_utp = hit.hmm_from <= UTP_REGION_END

        # Does the match primarily cover uTP region?
        primarily_utp = utp_fraction > 0.5

        # Classification
        if starts_in_utp and primarily_utp:
            hit_type = "UTP_REGION_HIT"  # Potential uTP homolog
        elif starts_in_utp and utp_fraction > 0.2:
            hit_type = "PARTIAL_UTP_HIT"  # Extends from uTP into mature domain
        else:
            hit_type = "MATURE_DOMAIN_HIT"  # Hit to mature domain only

        records.append(
            {
                "target_name": hit.target_name,
                "target_length": target_len,
                "full_evalue": hit.full_evalue,
                "full_score": hit.full_score,
                "domain_evalue": hit.domain_evalue,
                "domain_score": hit.domain_score,
                "hmm_from": hit.hmm_from,
                "hmm_to": hit.hmm_to,
                "hmm_match_length": hmm_match_length,
                "ali_from": hit.ali_from,
                "ali_to": hit.ali_to,
                "dist_from_c_term": dist_from_c_term,
                "is_c_terminal": dist_from_c_term < 150 and dist_from_c_term >= 0,
                "utp_overlap": utp_overlap,
                "utp_fraction": utp_fraction,
                "starts_in_utp": starts_in_utp,
                "primarily_utp": primarily_utp,
                "hit_type": hit_type,
                "accuracy": hit.acc,
                "description": hit.description,
            }
        )

    return pd.DataFrame(records)


def main():
    print("=" * 70)
    print("CORRECTED HMM-based uTP Homolog Search")
    print("=" * 70)

    print(
        f"""
METHODOLOGY CORRECTION:
Based on HMM consensus analysis, the uTP region is at the START of the HMM:
- Position 1-{UTP_REGION_END}: uTP region (contains conserved motifs)
- Position {MATURE_DOMAIN_START}-{HMM_LENGTH}: Mature domain

The HMM was built from C-terminal fragments where uTP is at the N-terminus
of each fragment.

TRUE uTP homologs must match HMM positions 1-100 AND be at the C-terminus
of the target protein.
"""
    )

    # Get metadata
    summary_file = HAPTOPHYTE_DIR / "data_summary.tsv"
    metadata = pd.read_csv(summary_file, sep="\t")

    # Find genomes with protein annotations
    genomes = []
    for _, row in metadata.iterrows():
        accession = row["Assembly Accession"]
        organism = row["Organism Scientific Name"]
        genome_dir = HAPTOPHYTE_DIR / accession

        if not genome_dir.exists():
            continue

        protein_file = genome_dir / "protein.faa"
        if protein_file.exists():
            genomes.append((accession, organism, protein_file))

    print(f"Analyzing {len(genomes)} haptophyte proteomes\n")

    # Analyze pre-existing hmmsearch results
    all_results = []

    for accession, organism, protein_file in genomes:
        # Handle both naming conventions
        domtblout_path = OUTPUT_DIR / f"hmmsearch_{accession}.domtblout"
        if not domtblout_path.exists():
            base_accession = (
                accession.rsplit(".", 1)[0] if "." in accession else accession
            )
            domtblout_path = OUTPUT_DIR / f"hmmsearch_{base_accession}.domtblout"

        if not domtblout_path.exists():
            print(f"Skipping {organism} - no hmmsearch results")
            continue

        print(f"\n{'='*60}")
        print(f"Analyzing: {organism}")
        print("=" * 60)

        # Parse results
        hits = parse_hmmsearch_domtblout(domtblout_path)

        if not hits:
            print("  No hits")
            continue

        # Get protein lengths
        protein_lengths = get_protein_lengths(protein_file)

        # Corrected analysis
        df = analyze_hmm_hits_corrected(hits, protein_lengths)
        df["organism"] = organism
        df["accession"] = accession

        # Categorize hits
        print(f"\n  Hit type breakdown:")
        for hit_type in ["UTP_REGION_HIT", "PARTIAL_UTP_HIT", "MATURE_DOMAIN_HIT"]:
            count = len(df[df["hit_type"] == hit_type])
            if count > 0:
                print(f"    {hit_type}: {count}")

        # Show any potential uTP homologs (hits to positions 1-100)
        utp_candidates = df[df["hit_type"] == "UTP_REGION_HIT"]

        # Further filter: must also be at C-terminus of target
        true_candidates = utp_candidates[utp_candidates["is_c_terminal"]]

        if len(true_candidates) > 0:
            print(f"\n  *** TRUE UTP HOMOLOG CANDIDATES (uTP region + C-terminal): ***")
            for _, hit in true_candidates.iterrows():
                print(f"    {hit['target_name']}: {hit['description'][:50]}")
                print(
                    f"      HMM: {hit['hmm_from']}-{hit['hmm_to']}, "
                    f"Score: {hit['domain_score']:.1f}, E: {hit['domain_evalue']:.2e}"
                )
                print(
                    f"      uTP overlap: {hit['utp_overlap']} aa ({hit['utp_fraction']:.0%})"
                )
                print(f"      Distance from C-term: {hit['dist_from_c_term']} aa")
        elif len(utp_candidates) > 0:
            print(f"\n  uTP region hits (but NOT at C-terminus): {len(utp_candidates)}")
            for _, hit in utp_candidates.head(3).iterrows():
                print(
                    f"    {hit['target_name']}: pos {hit['ali_from']}-{hit['ali_to']} "
                    f"(dist from C-term: {hit['dist_from_c_term']})"
                )

        # Partial hits
        partial = df[df["hit_type"] == "PARTIAL_UTP_HIT"]
        if len(partial) > 0:
            c_term_partial = partial[partial["is_c_terminal"]]
            print(
                f"\n  Partial uTP hits: {len(partial)} ({len(c_term_partial)} at C-terminus)"
            )

        all_results.append(df)

    # Combine and summarize
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / "hmmsearch_corrected_analysis.csv", index=False)

        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        print(f"\nTotal HMM hits analyzed: {len(combined)}")

        print("\nBy hit type:")
        for hit_type in ["UTP_REGION_HIT", "PARTIAL_UTP_HIT", "MATURE_DOMAIN_HIT"]:
            count = len(combined[combined["hit_type"] == hit_type])
            print(f"  {hit_type}: {count}")

        # The key finding: hits that are BOTH in uTP region AND at C-terminus
        utp_hits = combined[combined["hit_type"] == "UTP_REGION_HIT"]
        true_homologs = utp_hits[utp_hits["is_c_terminal"]]

        print("\n" + "=" * 70)
        print("CONCLUSION: Question A (CORRECTED)")
        print("=" * 70)

        if len(true_homologs) == 0:
            print(
                f"""
RESULT: NO TRUE uTP HOMOLOGS FOUND

Hits analysis:
- Total hits: {len(combined)}
- Hits to uTP region (HMM pos 1-100): {len(utp_hits)}
- uTP region hits at C-terminus: {len(true_homologs)}

All HMM hits either:
1. Match the mature domain portion (positions 100-713), OR
2. Match the uTP region but are NOT at the C-terminus of the target protein
   (i.e., the similar sequence exists but in a different position)

INTERPRETATION:
This strongly supports DE NOVO EVOLUTION of the uTP sequence.

The uTP sequence pattern does not exist in non-symbiotic haptophytes
at the C-terminus where it would need to be for protein import function.
"""
            )
        else:
            print(f"\n*** POTENTIAL uTP HOMOLOGS FOUND: {len(true_homologs)} ***")
            true_homologs_sorted = true_homologs.sort_values(
                "domain_score", ascending=False
            )
            true_homologs_sorted.to_csv(
                OUTPUT_DIR / "true_utp_homolog_candidates.csv", index=False
            )

            print("\nThese candidates require manual validation:")
            for _, hit in true_homologs_sorted.head(10).iterrows():
                print(f"\n  {hit['organism']}: {hit['target_name']}")
                print(f"    {hit['description']}")
                print(
                    f"    HMM positions: {hit['hmm_from']}-{hit['hmm_to']} "
                    f"(uTP overlap: {hit['utp_fraction']:.0%})"
                )
                print(
                    f"    Protein positions: {hit['ali_from']}-{hit['ali_to']} "
                    f"(dist from C-term: {hit['dist_from_c_term']})"
                )
                print(
                    f"    Score: {hit['domain_score']:.1f}, E-value: {hit['domain_evalue']:.2e}"
                )


if __name__ == "__main__":
    main()
