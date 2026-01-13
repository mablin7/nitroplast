#!/usr/bin/env python3
"""
Extract uTP sequences for clustering analysis.

This script extracts the C-terminal uTP regions from all proteins using
HMM-based detection, preparing them for clustering analysis.
"""

import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data"
HMM_FILE = SCRIPT_DIR.parent / "utp_homolog_search" / "utp.hmm"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Input files
IMPORT_CANDIDATES = DATA_DIR / "Import_candidates.fasta"

# HMM settings
MIN_EVALUE = 1e-5
MIN_UTP_LENGTH = 50  # Minimum uTP length to include


def run_hmmsearch(fasta_file, hmm_file, output_file):
    """Run hmmsearch on sequences."""
    cmd = [
        "hmmsearch",
        "--domtblout",
        str(output_file),
        "-E",
        str(MIN_EVALUE),
        str(hmm_file),
        str(fasta_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"hmmsearch failed: {result.stderr}")
    return output_file


def parse_hmmsearch_domtbl(domtbl_file):
    """Parse hmmsearch domain table output."""
    hits = []
    with open(domtbl_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 23:
                continue

            hits.append(
                {
                    "target_name": parts[0],
                    "target_len": int(parts[2]),
                    "query_name": parts[3],
                    "evalue": float(parts[6]),
                    "score": float(parts[7]),
                    "env_from": int(parts[19]),
                    "env_to": int(parts[20]),
                }
            )

    return pd.DataFrame(hits)


def extract_utp_regions(sequences, hmm_hits):
    """
    Extract uTP regions from sequences based on HMM hits.

    Takes the region from HMM hit start to the C-terminus.
    """
    utp_sequences = []
    metadata = []

    for _, hit in hmm_hits.iterrows():
        seq_name = hit["target_name"]
        if seq_name not in sequences:
            continue

        seq = sequences[seq_name]
        seq_len = len(seq)

        # uTP region: from HMM hit start to C-terminus
        utp_start = hit["env_from"] - 1  # Convert to 0-based
        utp_seq = seq[utp_start:]

        if len(utp_seq) < MIN_UTP_LENGTH:
            continue

        # Check if hit is C-terminal (within 50aa of end)
        is_c_terminal = (seq_len - hit["env_to"]) < 50

        utp_sequences.append(
            SeqRecord(
                Seq(utp_seq), id=seq_name, description=f"uTP region len={len(utp_seq)}"
            )
        )

        metadata.append(
            {
                "name": seq_name,
                "full_length": seq_len,
                "utp_start": utp_start,
                "utp_length": len(utp_seq),
                "hmm_score": hit["score"],
                "hmm_evalue": hit["evalue"],
                "is_c_terminal": is_c_terminal,
            }
        )

    return utp_sequences, pd.DataFrame(metadata)


def main():
    print("=" * 70)
    print("uTP Sequence Extraction")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load sequences
    print("\n[1/3] Loading sequences...")
    sequences = {r.id: str(r.seq) for r in SeqIO.parse(IMPORT_CANDIDATES, "fasta")}
    print(f"  Loaded {len(sequences)} sequences from Import_candidates.fasta")

    # Run HMM search
    print("\n[2/3] Running HMM search...")
    with tempfile.NamedTemporaryFile(suffix=".domtbl", delete=False) as tmp:
        domtbl_file = Path(tmp.name)

    run_hmmsearch(IMPORT_CANDIDATES, HMM_FILE, domtbl_file)
    hmm_hits = parse_hmmsearch_domtbl(domtbl_file)
    domtbl_file.unlink()

    print(f"  Found {len(hmm_hits)} HMM hits")

    # Keep best hit per sequence
    hmm_hits = hmm_hits.sort_values("score", ascending=False)
    hmm_hits = hmm_hits.drop_duplicates(subset=["target_name"], keep="first")
    print(f"  Unique sequences with hits: {len(hmm_hits)}")

    # Extract uTP regions
    print("\n[3/3] Extracting uTP regions...")
    utp_sequences, metadata = extract_utp_regions(sequences, hmm_hits)

    # Save outputs
    output_fasta = OUTPUT_DIR / "utp_sequences.fasta"
    SeqIO.write(utp_sequences, output_fasta, "fasta")
    print(f"  Saved {len(utp_sequences)} uTP sequences to {output_fasta}")

    output_meta = OUTPUT_DIR / "utp_metadata.csv"
    metadata.to_csv(output_meta, index=False)
    print(f"  Saved metadata to {output_meta}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"  Total sequences: {len(sequences)}")
    print(f"  HMM hits: {len(hmm_hits)}")
    print(f"  Extracted uTPs: {len(utp_sequences)}")
    print(f"  C-terminal hits: {metadata['is_c_terminal'].sum()}")
    print(f"\n  uTP length distribution:")
    print(f"    Min: {metadata['utp_length'].min()}")
    print(f"    Max: {metadata['utp_length'].max()}")
    print(f"    Median: {metadata['utp_length'].median():.0f}")
    print(f"    Mean: {metadata['utp_length'].mean():.1f}")


if __name__ == "__main__":
    main()
