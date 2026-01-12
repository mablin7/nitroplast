#!/usr/bin/env python3
"""
02_filter_controls.py - Filter control candidates based on CELLO localization predictions

This script parses CELLO output and filters control candidates to create multiple
control sets for different experimental conditions.

Input:
    - CELLO output file (tab-separated)
    - control_candidates.fasta from 01_prepare_data.py

Output:
    - filtered_controls_cytoplasmic.fasta - Cytoplasmic proteins only (strictest)
    - filtered_controls_nuclear_cytoplasmic.fasta - Nuclear + Cytoplasmic (relaxed)
    - localization_summary.csv - Summary of localization predictions
"""

import pandas as pd
from pathlib import Path
from Bio import SeqIO
import re


def parse_cello_output(cello_file: Path) -> pd.DataFrame:
    """
    Parse CELLO output file into a DataFrame.

    CELLO format:
    - Tab-separated (with variable empty fields)
    - Header starts with #SeqNO.
    - Last two non-empty fields are #Most-likely-Location and #SeqName

    Example line:
    #1	Mitochondrial	Cytoplasmic	Nuclear		Cytoplasmic	Cytoplasmic	#:	0.107	...	1.360	Cytoplasmic	KC1-P2_N4_k55...
    """
    rows = []

    with open(cello_file, "r") as f:
        header_line = f.readline().strip()

        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse from the end - last field is SeqName, second-to-last is Location
            # But there may be trailing tabs, so filter empty fields
            fields = [f.strip() for f in line.split("\t")]
            non_empty_fields = [f for f in fields if f]

            if len(non_empty_fields) < 2:
                continue

            # Extract seq_no from first field
            seq_no = fields[0].replace("#", "")

            # Last non-empty field is SeqName (contains protein ID and metadata)
            seq_name = non_empty_fields[-1]

            # Second-to-last non-empty field is Most-likely-Location
            most_likely_location = non_empty_fields[-2]

            # Extract protein ID from seq_name
            # Format: "KC1-P2_N4_k55_Locus_2060_Transcript_1_1 control_candidate len=481"
            protein_id = seq_name.split(" ")[0] if seq_name else ""

            # Skip if protein_id looks like a score (float)
            try:
                float(protein_id)
                continue  # This was a score, not a valid protein ID
            except ValueError:
                pass  # Good, it's not a float

            # Extract length from seq_name
            len_match = re.search(r"len=(\d+)", seq_name)
            length = int(len_match.group(1)) if len_match else None

            # Extract cytoplasmic score - find #: marker and get 3rd value after it
            cytop_score = None
            try:
                hash_idx = fields.index("#:")
                cytop_score = float(fields[hash_idx + 3])
            except (ValueError, IndexError):
                pass

            rows.append(
                {
                    "seq_no": seq_no,
                    "protein_id": protein_id,
                    "location": most_likely_location,
                    "cytoplasmic_score": cytop_score,
                    "length": length,
                    "full_name": seq_name,
                }
            )

    return pd.DataFrame(rows)


def filter_by_localization(
    df: pd.DataFrame,
    allowed_locations: list[str] = ["Cytoplasmic"],
) -> pd.DataFrame:
    """
    Filter DataFrame to keep only proteins with specified localizations.

    Args:
        df: DataFrame from parse_cello_output
        allowed_locations: List of allowed localization predictions

    Returns:
        Filtered DataFrame
    """
    filtered = df[df["location"].isin(allowed_locations)].copy()
    return filtered


def extract_filtered_sequences(
    filtered_df: pd.DataFrame, control_fasta: Path, output_fasta: Path
) -> int:
    """
    Extract sequences for filtered proteins from control candidates FASTA.

    Returns:
        Number of sequences written
    """
    # Get set of protein IDs to keep
    keep_ids = set(filtered_df["protein_id"].tolist())

    # Load control candidates and filter
    sequences = []
    for record in SeqIO.parse(control_fasta, "fasta"):
        # Extract protein ID from FASTA header
        protein_id = record.id.split()[0]
        if protein_id in keep_ids:
            sequences.append(record)

    # Write filtered sequences
    SeqIO.write(sequences, output_fasta, "fasta")

    return len(sequences)


def main():
    # Configuration
    output_dir = Path(__file__).parent / "output"

    # Input files
    cello_file = output_dir / "805936217017187.result_save.txt"
    control_fasta = output_dir / "control_candidates.fasta"

    # Output files
    cytoplasmic_fasta = output_dir / "filtered_controls_cytoplasmic.fasta"
    nuclear_cytoplasmic_fasta = (
        output_dir / "filtered_controls_nuclear_cytoplasmic.fasta"
    )
    summary_csv = output_dir / "localization_summary.csv"

    # Check inputs exist
    if not cello_file.exists():
        raise FileNotFoundError(f"CELLO output not found: {cello_file}")
    if not control_fasta.exists():
        raise FileNotFoundError(f"Control candidates not found: {control_fasta}")

    print("=" * 60)
    print("Step 2: Filter Control Candidates by Localization")
    print("=" * 60)

    # Parse CELLO output
    print(f"\nParsing CELLO output: {cello_file}")
    df = parse_cello_output(cello_file)
    print(f"  Total predictions: {len(df)}")

    # Summarize localizations
    print("\nLocalization distribution:")
    loc_counts = df["location"].value_counts()
    for loc, count in loc_counts.items():
        pct = count / len(df) * 100
        print(f"  {loc}: {count} ({pct:.1f}%)")

    # Save summary
    summary_df = pd.DataFrame(
        {
            "location": loc_counts.index,
            "count": loc_counts.values,
            "percentage": (loc_counts.values / len(df) * 100).round(1),
        }
    )
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved localization summary to: {summary_csv}")

    # =========================================================================
    # Filter 1: Cytoplasmic only (strictest)
    # =========================================================================
    print("\n" + "-" * 40)
    print("Filter 1: Cytoplasmic only (strictest)")
    print("-" * 40)

    cyto_df = filter_by_localization(df, allowed_locations=["Cytoplasmic"])
    n_cyto = extract_filtered_sequences(cyto_df, control_fasta, cytoplasmic_fasta)
    print(f"  Cytoplasmic proteins: {len(cyto_df)}")
    print(f"  Sequences written: {n_cyto}")
    print(f"  Output: {cytoplasmic_fasta.name}")

    # =========================================================================
    # Filter 2: Nuclear + Cytoplasmic (relaxed)
    # =========================================================================
    print("\n" + "-" * 40)
    print("Filter 2: Nuclear + Cytoplasmic (relaxed)")
    print("-" * 40)

    nuc_cyto_df = filter_by_localization(
        df, allowed_locations=["Nuclear", "Cytoplasmic"]
    )
    n_nuc_cyto = extract_filtered_sequences(
        nuc_cyto_df, control_fasta, nuclear_cytoplasmic_fasta
    )
    print(f"  Nuclear + Cytoplasmic proteins: {len(nuc_cyto_df)}")
    print(f"  Sequences written: {n_nuc_cyto}")
    print(f"  Output: {nuclear_cytoplasmic_fasta.name}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    # Load mature domains to compare
    mature_fasta = output_dir / "mature_domains.fasta"
    if mature_fasta.exists():
        n_positive = sum(1 for _ in SeqIO.parse(mature_fasta, "fasta"))
        print(f"\n  Positive samples (uTP mature domains): {n_positive}")
        print(f"\n  Control sets:")
        print(f"    Cytoplasmic only:        {n_cyto:4d} ({n_cyto/len(df)*100:.1f}%)")
        print(
            f"    Nuclear + Cytoplasmic:   {n_nuc_cyto:4d} ({n_nuc_cyto/len(df)*100:.1f}%)"
        )

        print(f"\n  Experimental configurations:")
        print(f"    Exp 1 (downsampled): {n_cyto} uTP vs {n_cyto} cytoplasmic controls")
        print(
            f"    Exp 2 (full):        {n_positive} uTP vs {n_nuc_cyto} nuclear+cytoplasmic controls"
        )


if __name__ == "__main__":
    main()
