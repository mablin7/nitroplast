#!/usr/bin/env python3
"""
Prepare proteome FASTA files for OrthoFinder analysis.

This script:
1. Collects B. bigelowii proteins (from transcriptome/proteomics data)
2. Collects protein sequences from available haptophyte genomes
3. Prepares clean FASTA files in the format required by OrthoFinder
4. Prioritizes species based on phylogenetic distance to B. bigelowii

Based on phylogeny results from experiments/haptophyte_phylogeny/README.md
"""

import os
import re
import shutil
from pathlib import Path
from collections import defaultdict

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HAPTOPHYTE_DATA = DATA_DIR / "haptophytes" / "ncbi_dataset" / "data"
OUTPUT_DIR = Path(__file__).parent / "proteomes"

# Species metadata with phylogenetic priority (based on 18S phylogeny)
# Priority 1 = closest to B. bigelowii, higher = more distant
SPECIES_INFO = {
    # Emiliania huxleyi - closest relative with genome (~330 Myr divergence)
    "GCA_000372725.1": {
        "name": "Emiliania_huxleyi",
        "full_name": "Emiliania huxleyi CCMP1516",
        "priority": 1,
        "divergence_myr": 330,
    },
    "GCF_000372725.1": {
        "name": "Emiliania_huxleyi_RefSeq",
        "full_name": "Emiliania huxleyi CCMP1516 (RefSeq)",
        "priority": 1,
        "divergence_myr": 330,
        "skip": True,  # Duplicate of GCA, skip
    },
    # Prymnesium parvum - moderate divergence (~460 Myr)
    "GCA_041296205.1": {
        "name": "Prymnesium_parvum",
        "full_name": "Prymnesium parvum 12B1",
        "priority": 2,
        "divergence_myr": 460,
    },
    # Chrysochromulina tobinii - higher divergence (~800 Myr)
    "GCA_001275005.1": {
        "name": "Chrysochromulina_tobinii",
        "full_name": "Chrysochromulina tobinii CCMP291",
        "priority": 3,
        "divergence_myr": 800,
    },
    # Diacronema lutheri (Pavlovophyceae) - high divergence (~910 Myr)
    "GCA_019448385.1": {
        "name": "Diacronema_lutheri",
        "full_name": "Diacronema lutheri NIVA-4/92",
        "priority": 4,
        "divergence_myr": 910,
    },
    # Pavlovales sp. - high divergence (~910 Myr)
    "GCA_026770615.1": {
        "name": "Pavlovales_sp_CCMP2436",
        "full_name": "Pavlovales sp. CCMP2436",
        "priority": 4,
        "divergence_myr": 910,
    },
    # Prymnesium sp. - moderate divergence
    "GCA_046255225.1": {
        "name": "Prymnesium_sp",
        "full_name": "Prymnesium sp. SGEUK-05",
        "priority": 2,
        "divergence_myr": 460,
    },
}


def clean_sequence_id(seq_id: str, species_prefix: str) -> str:
    """
    Clean sequence ID for OrthoFinder compatibility.
    OrthoFinder requires unique IDs without special characters.
    """
    # Remove problematic characters
    clean_id = re.sub(r"[^\w\-\.]", "_", seq_id)
    return clean_id


def load_bbigelowii_proteins() -> list[SeqRecord]:
    """
    Load B. bigelowii proteins from the proteomics database.
    These are the transcriptome-derived proteins including uTP candidates.
    """
    fasta_path = DATA_DIR / "ADK1075_proteomics_DB_2.fasta"

    records = []
    seen_ids = set()

    for record in SeqIO.parse(fasta_path, "fasta"):
        # Filter to only B. bigelowii proteins (not UCYN-A)
        # UCYN-A proteins have "cyanobacterium endosymbiont" in description
        if "cyanobacterium endosymbiont" in record.description:
            continue

        # Clean the ID
        clean_id = clean_sequence_id(record.id, "Bb")

        # Handle duplicates
        if clean_id in seen_ids:
            continue
        seen_ids.add(clean_id)

        # Create new record with clean ID
        new_record = SeqRecord(record.seq, id=clean_id, description="")
        records.append(new_record)

    return records


def load_haptophyte_proteome(accession: str) -> list[SeqRecord]:
    """
    Load proteome from NCBI genome assembly.
    """
    protein_faa = HAPTOPHYTE_DATA / accession / "protein.faa"

    if not protein_faa.exists():
        print(f"  Warning: No protein.faa found for {accession}")
        return []

    records = []
    seen_ids = set()

    for record in SeqIO.parse(protein_faa, "fasta"):
        clean_id = clean_sequence_id(record.id, accession[:3])

        if clean_id in seen_ids:
            continue
        seen_ids.add(clean_id)

        new_record = SeqRecord(record.seq, id=clean_id, description="")
        records.append(new_record)

    return records


def prepare_proteomes(include_all: bool = False):
    """
    Prepare all proteome files for OrthoFinder.

    Args:
        include_all: If True, include all available genomes.
                    If False, only include priority 1-3 (closest relatives).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []

    # 1. B. bigelowii proteins
    print("Loading B. bigelowii proteins...")
    bb_proteins = load_bbigelowii_proteins()
    bb_output = OUTPUT_DIR / "Braarudosphaera_bigelowii.faa"
    SeqIO.write(bb_proteins, bb_output, "fasta")
    print(f"  Wrote {len(bb_proteins)} proteins to {bb_output.name}")
    summary.append(
        {
            "species": "Braarudosphaera_bigelowii",
            "accession": "transcriptome",
            "proteins": len(bb_proteins),
            "priority": 0,
            "divergence_myr": 0,
            "source": "ADK1075_proteomics_DB_2.fasta",
        }
    )

    # 2. Haptophyte genomes
    print("\nLoading haptophyte proteomes...")

    for accession, info in sorted(SPECIES_INFO.items(), key=lambda x: x[1]["priority"]):
        if info.get("skip", False):
            print(f"  Skipping {info['name']} (duplicate)")
            continue

        if not include_all and info["priority"] > 3:
            print(
                f"  Skipping {info['name']} (priority {info['priority']}, too distant)"
            )
            continue

        print(f"  Loading {info['name']} ({accession})...")
        proteins = load_haptophyte_proteome(accession)

        if proteins:
            output_path = OUTPUT_DIR / f"{info['name']}.faa"
            SeqIO.write(proteins, output_path, "fasta")
            print(f"    Wrote {len(proteins)} proteins")

            summary.append(
                {
                    "species": info["name"],
                    "accession": accession,
                    "proteins": len(proteins),
                    "priority": info["priority"],
                    "divergence_myr": info["divergence_myr"],
                    "source": f"NCBI {accession}",
                }
            )

    # Write summary
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT_DIR / "proteome_summary.csv", index=False)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"\nProteome files written to: {OUTPUT_DIR}")
    print(f"Total species: {len(summary)}")

    return summary_df


def validate_proteomes():
    """
    Validate proteome files for OrthoFinder compatibility.
    """
    print("\nValidating proteome files...")

    faa_files = list(OUTPUT_DIR.glob("*.faa"))

    for faa_file in faa_files:
        record_count = 0
        issues = []

        for record in SeqIO.parse(faa_file, "fasta"):
            record_count += 1

            # Check for stop codons in sequence
            if "*" in str(record.seq):
                issues.append(f"Stop codon in {record.id}")

            # Check for very short sequences
            if len(record.seq) < 30:
                issues.append(
                    f"Very short sequence: {record.id} ({len(record.seq)} aa)"
                )

        status = "✓" if not issues else "⚠"
        print(f"  {status} {faa_file.name}: {record_count} proteins")

        if issues and len(issues) <= 5:
            for issue in issues:
                print(f"      - {issue}")
        elif issues:
            print(f"      - {len(issues)} issues found")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare proteomes for OrthoFinder")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include all available genomes (not just close relatives)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate proteome files after preparation",
    )

    args = parser.parse_args()

    summary = prepare_proteomes(include_all=args.all)

    if args.validate:
        validate_proteomes()

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(
        """
1. Install OrthoFinder (if not already installed):
   conda create -n orthofinder python=3.10
   conda activate orthofinder
   conda install -c bioconda orthofinder

2. Run OrthoFinder:
   orthofinder -f experiments/ortholog_search/proteomes -t 8

3. Results will be in:
   experiments/ortholog_search/proteomes/OrthoFinder/Results_<date>/
"""
    )
