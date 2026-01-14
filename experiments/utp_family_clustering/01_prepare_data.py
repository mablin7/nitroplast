#!/usr/bin/env python3
"""
01_prepare_data.py - Prepare Protein Data for Family Clustering

Extracts B. bigelowii proteins (excluding UCYN-A), identifies uTP proteins,
and prepares mature domains (removing C-terminal uTP) for clustering.

Goal: Test whether uTP proteins cluster into gene families more than expected
by chance (founder effect vs. selection model).

Usage:
    uv run python experiments/utp_family_clustering/01_prepare_data.py
"""

import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Input files
PROTEOME_DB = DATA_DIR / "ADK1075_proteomics_DB_2.fasta"
UTP_HMM_HITS = DATA_DIR / "uTP_HMM_hits.fasta"
HMM_PROFILE = SCRIPT_DIR.parent / "utp_homolog_search" / "utp.hmm"

# Output files
BB_PROTEINS = OUTPUT_DIR / "bb_proteins.fasta"
UTP_PROTEINS = OUTPUT_DIR / "utp_proteins.fasta"
MATURE_DOMAINS = OUTPUT_DIR / "mature_domains.fasta"
UTP_MATURE_DOMAINS = OUTPUT_DIR / "utp_mature_domains.fasta"
METADATA_FILE = OUTPUT_DIR / "protein_metadata.csv"

# =============================================================================
# Constants
# =============================================================================

# Keywords indicating UCYN-A (cyanobacterium endosymbiont) proteins
UCYNA_KEYWORDS = [
    "cyanobacterium endosymbiont",
    "CPARK",
    "BDA4",  # UCYN-A protein accessions start with BDA4
]

# Default uTP length when HMM boundary not available
DEFAULT_UTP_LENGTH = 120

# Minimum lengths
MIN_MATURE_LENGTH = 50  # Require at least 50 aa mature domain

# =============================================================================
# Functions
# =============================================================================


def is_ucyna_protein(description: str) -> bool:
    """Check if a protein is from UCYN-A (cyanobacterium endosymbiont)."""
    desc_lower = description.lower()
    for keyword in UCYNA_KEYWORDS:
        if keyword.lower() in desc_lower:
            return True
    return False


def load_proteome(fasta_file: Path) -> dict:
    """Load proteome from FASTA file."""
    proteins = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        proteins[record.id] = {
            "id": record.id,
            "description": record.description,
            "sequence": str(record.seq),
        }
    return proteins


def load_utp_hit_ids(fasta_file: Path) -> set:
    """Load IDs of proteins with uTP detected."""
    ids = set()
    for record in SeqIO.parse(fasta_file, "fasta"):
        ids.add(record.id)
    return ids


def run_hmmscan(sequences: dict, hmm_profile: Path) -> dict:
    """
    Run hmmscan to find uTP boundaries in sequences.

    Returns dict mapping protein_id -> (hit_start, hit_end, score, evalue)
    """
    import shutil

    if not shutil.which("hmmscan"):
        print("  Warning: hmmscan not available, using default uTP length")
        return {}

    if not hmm_profile.exists():
        print(f"  Warning: HMM profile not found at {hmm_profile}")
        return {}

    results = {}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        fasta_path = Path(f.name)
        for prot_id, prot_data in sequences.items():
            f.write(f">{prot_id}\n{prot_data['sequence']}\n")

    try:
        # Run hmmscan
        cmd = [
            "hmmscan",
            "--domtblout",
            "/dev/stdout",
            "--noali",
            "-E",
            "0.01",
            str(hmm_profile),
            str(fasta_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse domain table output
        for line in result.stdout.split("\n"):
            if line.startswith("#") or not line.strip():
                continue

            parts = line.split()
            if len(parts) >= 23:
                target_name = parts[0]
                query_name = parts[3]
                evalue = float(parts[6])
                score = float(parts[7])
                # Domain envelope coordinates in query (1-based)
                env_from = int(parts[19])
                env_to = int(parts[20])

                # Store best hit per protein
                if query_name not in results or score > results[query_name][2]:
                    results[query_name] = (env_from, env_to, score, evalue)

    finally:
        fasta_path.unlink()

    return results


def extract_mature_domain(
    sequence: str, utp_start: int = None, default_utp_len: int = DEFAULT_UTP_LENGTH
) -> str:
    """
    Extract mature domain by removing C-terminal uTP.

    Args:
        sequence: Full protein sequence
        utp_start: 1-based start position of uTP region (if known from HMM)
        default_utp_len: Default uTP length to use if utp_start not provided

    Returns:
        Mature domain sequence (N-terminal portion before uTP)
    """
    if utp_start is not None and utp_start > 1:
        # Use HMM-detected boundary (convert to 0-based)
        return sequence[: utp_start - 1]
    else:
        # Use default: remove last ~120 aa
        if len(sequence) > default_utp_len:
            return sequence[:-default_utp_len]
        else:
            return sequence[: len(sequence) // 2]  # Fallback for short sequences


def main():
    print("=" * 70)
    print("Prepare Data for Family Clustering Analysis")
    print("=" * 70)
    print("\nGoal: Test if uTP proteins cluster into families more than expected")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Load all proteins and separate B. bigelowii from UCYN-A
    # =========================================================================
    print("\n[1/4] Loading proteome and separating B. bigelowii from UCYN-A...")

    all_proteins = load_proteome(PROTEOME_DB)
    print(f"  Total proteins in database: {len(all_proteins)}")

    bb_proteins = {}
    ucyna_proteins = {}

    for prot_id, prot_data in all_proteins.items():
        if is_ucyna_protein(prot_data["description"]):
            ucyna_proteins[prot_id] = prot_data
        else:
            bb_proteins[prot_id] = prot_data

    print(f"  B. bigelowii proteins: {len(bb_proteins)}")
    print(f"  UCYN-A proteins: {len(ucyna_proteins)}")

    # =========================================================================
    # Step 2: Identify uTP proteins
    # =========================================================================
    print("\n[2/4] Identifying uTP proteins...")

    utp_hit_ids = load_utp_hit_ids(UTP_HMM_HITS)

    # Filter to only B. bigelowii proteins (some hits might be misclassified)
    utp_proteins = {
        prot_id: prot_data
        for prot_id, prot_data in bb_proteins.items()
        if prot_id in utp_hit_ids
    }

    non_utp_proteins = {
        prot_id: prot_data
        for prot_id, prot_data in bb_proteins.items()
        if prot_id not in utp_hit_ids
    }

    print(f"  uTP proteins (HMM hits in B. bigelowii): {len(utp_proteins)}")
    print(f"  Non-uTP proteins: {len(non_utp_proteins)}")

    # =========================================================================
    # Step 3: Find uTP boundaries using HMM
    # =========================================================================
    print("\n[3/4] Finding uTP boundaries...")

    utp_boundaries = run_hmmscan(utp_proteins, HMM_PROFILE)
    print(f"  HMM boundaries detected: {len(utp_boundaries)}")

    # =========================================================================
    # Step 4: Extract mature domains and save all data
    # =========================================================================
    print("\n[4/4] Extracting mature domains and saving data...")

    metadata = []
    mature_records = []
    utp_mature_records = []

    # Process uTP proteins - extract mature domain
    for prot_id, prot_data in utp_proteins.items():
        seq = prot_data["sequence"]

        # Get uTP boundary if available
        utp_start = None
        if prot_id in utp_boundaries:
            utp_start = utp_boundaries[prot_id][0]

        # Extract mature domain
        mature_seq = extract_mature_domain(seq, utp_start)

        if len(mature_seq) >= MIN_MATURE_LENGTH:
            record = SeqRecord(
                Seq(mature_seq),
                id=prot_id,
                description=f"mature_domain|uTP|len={len(mature_seq)}",
            )
            mature_records.append(record)
            utp_mature_records.append(record)

            metadata.append(
                {
                    "protein_id": prot_id,
                    "has_utp": True,
                    "full_length": len(seq),
                    "mature_length": len(mature_seq),
                    "utp_start": utp_start,
                    "utp_length": len(seq) - len(mature_seq),
                    "description": prot_data["description"][:100],
                }
            )

    # Process non-uTP proteins - use full sequence as "mature domain"
    for prot_id, prot_data in non_utp_proteins.items():
        seq = prot_data["sequence"]

        if len(seq) >= MIN_MATURE_LENGTH:
            record = SeqRecord(
                Seq(seq), id=prot_id, description=f"full_protein|no_uTP|len={len(seq)}"
            )
            mature_records.append(record)

            metadata.append(
                {
                    "protein_id": prot_id,
                    "has_utp": False,
                    "full_length": len(seq),
                    "mature_length": len(seq),
                    "utp_start": None,
                    "utp_length": 0,
                    "description": prot_data["description"][:100],
                }
            )

    # Save outputs
    SeqIO.write(mature_records, MATURE_DOMAINS, "fasta")
    print(f"  Saved {len(mature_records)} mature domains to {MATURE_DOMAINS}")

    SeqIO.write(utp_mature_records, UTP_MATURE_DOMAINS, "fasta")
    print(
        f"  Saved {len(utp_mature_records)} uTP mature domains to {UTP_MATURE_DOMAINS}"
    )

    # Save all B. bigelowii proteins
    bb_records = [
        SeqRecord(Seq(p["sequence"]), id=pid, description=p["description"])
        for pid, p in bb_proteins.items()
    ]
    SeqIO.write(bb_records, BB_PROTEINS, "fasta")
    print(f"  Saved {len(bb_records)} B. bigelowii proteins to {BB_PROTEINS}")

    # Save uTP proteins
    utp_records = [
        SeqRecord(Seq(p["sequence"]), id=pid, description=p["description"])
        for pid, p in utp_proteins.items()
    ]
    SeqIO.write(utp_records, UTP_PROTEINS, "fasta")
    print(f"  Saved {len(utp_records)} uTP proteins to {UTP_PROTEINS}")

    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(METADATA_FILE, index=False)
    print(f"  Saved metadata to {METADATA_FILE}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    utp_count = metadata_df["has_utp"].sum()
    total_count = len(metadata_df)

    print(f"\nTotal proteins for clustering: {total_count}")
    print(f"  - uTP proteins: {utp_count} ({100*utp_count/total_count:.1f}%)")
    print(
        f"  - Non-uTP proteins: {total_count - utp_count} ({100*(total_count-utp_count)/total_count:.1f}%)"
    )

    print(f"\nMature domain lengths:")
    print(
        f"  - uTP proteins: median {metadata_df[metadata_df['has_utp']]['mature_length'].median():.0f} aa"
    )
    print(
        f"  - Non-uTP proteins: median {metadata_df[~metadata_df['has_utp']]['mature_length'].median():.0f} aa"
    )

    print(f"\n📁 Output files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
