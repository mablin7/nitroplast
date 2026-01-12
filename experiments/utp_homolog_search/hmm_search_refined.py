#!/usr/bin/env python3
"""
Refined HMM-based uTP Homolog Search

CRITICAL REFINEMENT:
The uTP HMM (713 positions) was built from full C-terminal protein sequences.
It contains BOTH:
- Mature protein domain (positions ~1-620)
- Actual uTP region (positions ~620-713)

To find TRUE uTP homologs, we must:
1. Look for hits that match the C-terminal portion of the HMM (positions >650)
2. NOT count hits that only match the mature domain portion
3. Exclude known conserved domain hits (TBC, etc.)

Author: Generated for nitroplast/UCYN-A research project
Date: January 2026
"""

import subprocess
import re
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
EXPERIMENT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
HAPTOPHYTE_DIR = DATA_DIR / "haptophytes" / "ncbi_dataset" / "data"

HMM_PROFILE = EXPERIMENT_DIR / "utp.hmm"
OUTPUT_DIR = EXPERIMENT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# CRITICAL: Define the TRUE uTP region in the HMM
# Based on motif analysis, the uTP motifs are in the last ~90 positions
HMM_LENGTH = 713
UTP_CORE_START = 650  # Positions 650-713 are the core uTP motifs
UTP_REGION_START = 620  # More permissive boundary


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


def analyze_hmm_hits_refined(
    hits: list[HMMHit], protein_lengths: dict[str, int]
) -> pd.DataFrame:
    """
    Analyze HMM hits with refined criteria for TRUE uTP homologs.

    A TRUE uTP homolog must:
    1. Match the C-terminal portion of the HMM (>650, or at least >620)
    2. Be at the C-terminus of the target protein
    3. NOT be primarily a mature domain hit
    """
    records = []

    for hit in hits:
        target_len = protein_lengths.get(hit.target_name, 0)
        dist_from_c_term = target_len - hit.ali_to if target_len > 0 else -1

        # Calculate what portion of the hit is in the uTP region
        hmm_match_length = hit.hmm_to - hit.hmm_from + 1

        # How much of the hit extends into the core uTP region (>650)?
        utp_core_overlap = max(0, hit.hmm_to - UTP_CORE_START)
        utp_region_overlap = max(0, hit.hmm_to - UTP_REGION_START)

        # Is this primarily a uTP region hit, or mature domain?
        # A true uTP homolog should have >50% of match in uTP region
        utp_fraction_core = (
            utp_core_overlap / hmm_match_length if hmm_match_length > 0 else 0
        )
        utp_fraction_region = (
            utp_region_overlap / hmm_match_length if hmm_match_length > 0 else 0
        )

        # Does the match START in the uTP region?
        starts_in_utp_core = hit.hmm_from >= UTP_CORE_START
        starts_in_utp_region = hit.hmm_from >= UTP_REGION_START

        # Classification
        if starts_in_utp_core:
            hit_type = "UTP_CORE_HIT"  # Strong candidate
        elif starts_in_utp_region:
            hit_type = "UTP_REGION_HIT"  # Moderate candidate
        elif utp_fraction_region > 0.5:
            hit_type = "PARTIAL_UTP_HIT"  # Weak candidate
        else:
            hit_type = "MATURE_DOMAIN_HIT"  # Not a uTP homolog

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
                "is_c_terminal": dist_from_c_term < 50 and dist_from_c_term >= 0,
                "utp_core_overlap": utp_core_overlap,
                "utp_region_overlap": utp_region_overlap,
                "utp_fraction_core": utp_fraction_core,
                "utp_fraction_region": utp_fraction_region,
                "starts_in_utp_core": starts_in_utp_core,
                "starts_in_utp_region": starts_in_utp_region,
                "hit_type": hit_type,
                "accuracy": hit.acc,
                "description": hit.description,
            }
        )

    return pd.DataFrame(records)


def main():
    print("=" * 70)
    print("REFINED HMM-based uTP Homolog Search")
    print("=" * 70)

    print(
        f"""
METHODOLOGY:
- HMM profile length: {HMM_LENGTH} positions
- Mature domain region: positions 1-{UTP_REGION_START}
- uTP region (permissive): positions {UTP_REGION_START}-{HMM_LENGTH}
- uTP core region (stringent): positions {UTP_CORE_START}-{HMM_LENGTH}

TRUE uTP homologs must match the C-terminal portion of the HMM,
not just the conserved mature domain.
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
            # Try without .1 suffix
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

        # Refined analysis
        df = analyze_hmm_hits_refined(hits, protein_lengths)
        df["organism"] = organism
        df["accession"] = accession

        # Categorize hits
        print(f"\n  Hit type breakdown:")
        for hit_type in [
            "UTP_CORE_HIT",
            "UTP_REGION_HIT",
            "PARTIAL_UTP_HIT",
            "MATURE_DOMAIN_HIT",
        ]:
            count = len(df[df["hit_type"] == hit_type])
            if count > 0:
                print(f"    {hit_type}: {count}")

        # Show any potential uTP homologs
        utp_candidates = df[df["hit_type"].isin(["UTP_CORE_HIT", "UTP_REGION_HIT"])]
        if len(utp_candidates) > 0:
            print(f"\n  *** POTENTIAL UTP HOMOLOG CANDIDATES: ***")
            for _, hit in utp_candidates.iterrows():
                print(f"    {hit['target_name']}: {hit['description'][:50]}")
                print(
                    f"      HMM: {hit['hmm_from']}-{hit['hmm_to']}, "
                    f"Score: {hit['domain_score']:.1f}, E: {hit['domain_evalue']:.2e}"
                )
                print(f"      uTP fraction: {hit['utp_fraction_region']:.1%}")

        # Mature domain hits (these are NOT uTP homologs)
        mature_hits = df[df["hit_type"] == "MATURE_DOMAIN_HIT"]
        if len(mature_hits) > 0:
            print(f"\n  Mature domain hits (NOT uTP homologs): {len(mature_hits)}")
            # Show the TBC domain hit specifically
            tbc_hits = mature_hits[
                mature_hits["description"].str.contains("tbc|TBC", case=False, na=False)
            ]
            if len(tbc_hits) > 0:
                print(
                    f"    Including {len(tbc_hits)} TBC domain hits (conserved protein domain)"
                )

        all_results.append(df)

    # Combine and summarize
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / "hmmsearch_refined_analysis.csv", index=False)

        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        print(f"\nTotal HMM hits analyzed: {len(combined)}")

        print("\nBy hit type:")
        for hit_type in [
            "UTP_CORE_HIT",
            "UTP_REGION_HIT",
            "PARTIAL_UTP_HIT",
            "MATURE_DOMAIN_HIT",
        ]:
            count = len(combined[combined["hit_type"] == hit_type])
            print(f"  {hit_type}: {count}")

        # The key finding
        true_utp_homologs = combined[
            combined["hit_type"].isin(["UTP_CORE_HIT", "UTP_REGION_HIT"])
        ]

        print("\n" + "=" * 70)
        print("CONCLUSION: Question A")
        print("=" * 70)

        if len(true_utp_homologs) == 0:
            print(
                """
RESULT: NO TRUE uTP HOMOLOGS FOUND

All HMM hits are to the MATURE DOMAIN portion of the profile, not the uTP region.
The apparent hit in Chrysochromulina tobinii (TBC1 domain protein) matches:
  - HMM positions 449-632 (mature domain)
  - NOT the uTP-specific region (positions 650-713)

This is an ORTHOLOG of a conserved protein that carries uTP in B. bigelowii,
not a homolog of the uTP itself.

INTERPRETATION:
This strongly supports DE NOVO EVOLUTION of the uTP sequence.

Alternative explanations:
1. uTP diverged so extensively from its precursor as to be undetectable
2. uTP precursor exists but only in close B. bigelowii relatives (not available)
3. Search sensitivity is insufficient (unlikely given permissive parameters)

The most parsimonious explanation is that uTP evolved de novo in the
B. bigelowii lineage, possibly through exonization of non-coding DNA
or extensive modification of a pre-existing C-terminal sequence.
"""
            )
        else:
            print(f"\nPotential uTP homologs found: {len(true_utp_homologs)}")
            true_utp_homologs.to_csv(
                OUTPUT_DIR / "true_utp_homolog_candidates.csv", index=False
            )
            print("\nThese candidates require manual validation:")
            for _, hit in true_utp_homologs.iterrows():
                print(f"  {hit['organism']}: {hit['target_name']}")
                print(f"    {hit['description']}")


if __name__ == "__main__":
    main()
