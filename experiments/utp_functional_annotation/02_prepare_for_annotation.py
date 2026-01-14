#!/usr/bin/env python3
"""
02_prepare_for_annotation.py - Prepare sequences for functional annotation

This script:
1. Copies/links mature domains from the utp_presence_classifier experiment
2. Links the control proteins (already filtered by localization)
3. Prepares combined FASTA files for eggNOG-mapper submission
4. Computes biophysical properties for all sequences

The key innovation here is annotating MATURE DOMAINS, not full proteins.
This ensures the uTP region doesn't confound functional annotation.

Usage:
    uv run python experiments/utp_functional_annotation/02_prepare_for_annotation.py
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Source files from utp_presence_classifier
CLASSIFIER_DIR = SCRIPT_DIR.parent / "utp_presence_classifier" / "output"
MATURE_DOMAINS_SOURCE = CLASSIFIER_DIR / "mature_domains.fasta"
CONTROLS_SOURCE = CLASSIFIER_DIR / "filtered_controls_nuclear_cytoplasmic.fasta"

# Output files
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MATURE_DOMAINS_FILE = OUTPUT_DIR / "mature_domains.fasta"
CONTROLS_FILE = OUTPUT_DIR / "controls.fasta"
COMBINED_FOR_EGGNOG = OUTPUT_DIR / "sequences_for_eggnog.fasta"
PROPERTIES_FILE = OUTPUT_DIR / "protein_properties.csv"
SEQUENCE_METADATA = OUTPUT_DIR / "sequence_metadata.csv"


def get_protein_properties(seq_str: str) -> dict | None:
    """Calculate biophysical properties of a protein sequence."""
    # Remove ambiguous characters
    seq_clean = "".join(c for c in seq_str if c in "ACDEFGHIKLMNPQRSTVWY")
    if len(seq_clean) < 10:
        return None

    pa = ProteinAnalysis(seq_clean)

    try:
        ss = pa.secondary_structure_fraction()
        return {
            "length": len(seq_clean),
            "molecular_weight": pa.molecular_weight(),
            "isoelectric_point": pa.isoelectric_point(),
            "gravy": pa.gravy(),
            "instability_index": pa.instability_index(),
            "fraction_helix": ss[0],
            "fraction_sheet": ss[1],
            "fraction_coil": ss[2],
            "aromaticity": pa.aromaticity(),
            "charge_at_pH7": pa.charge_at_pH(7.0),
        }
    except Exception:
        return None


def main():
    """Main function to prepare data for annotation."""
    
    print("=" * 70)
    print("Preparing Sequences for Functional Annotation")
    print("=" * 70)
    
    # Check if source files exist
    if not MATURE_DOMAINS_SOURCE.exists():
        print(f"ERROR: Mature domains file not found: {MATURE_DOMAINS_SOURCE}")
        print("Please run utp_presence_classifier/01_prepare_data.py first")
        return
    
    if not CONTROLS_SOURCE.exists():
        print(f"ERROR: Controls file not found: {CONTROLS_SOURCE}")
        print("Please run utp_presence_classifier/02_filter_controls.py first")
        return
    
    # Copy/link source files
    print("\n--- Copying source files ---")
    shutil.copy(MATURE_DOMAINS_SOURCE, MATURE_DOMAINS_FILE)
    shutil.copy(CONTROLS_SOURCE, CONTROLS_FILE)
    print(f"Copied mature domains to: {MATURE_DOMAINS_FILE}")
    print(f"Copied controls to: {CONTROLS_FILE}")
    
    # Load sequences
    print("\n--- Loading sequences ---")
    utp_sequences = {}
    for record in SeqIO.parse(MATURE_DOMAINS_FILE, "fasta"):
        utp_sequences[record.id] = str(record.seq)
    
    control_sequences = {}
    for record in SeqIO.parse(CONTROLS_FILE, "fasta"):
        control_sequences[record.id] = str(record.seq)
    
    print(f"Loaded {len(utp_sequences)} uTP mature domains")
    print(f"Loaded {len(control_sequences)} control sequences")
    
    # Create combined FASTA for eggNOG-mapper
    print("\n--- Creating combined FASTA for eggNOG-mapper ---")
    combined_records = []
    metadata_rows = []
    
    for seq_id, seq in utp_sequences.items():
        # Add prefix to distinguish groups
        new_id = f"uTP_{seq_id}"
        record = SeqRecord(
            seq=seq,
            id=new_id,
            description=f"uTP mature domain from {seq_id}"
        )
        combined_records.append(record)
        metadata_rows.append({
            "eggnog_id": new_id,
            "original_id": seq_id,
            "group": "uTP",
            "length": len(seq)
        })
    
    for seq_id, seq in control_sequences.items():
        new_id = f"CTRL_{seq_id}"
        record = SeqRecord(
            seq=seq,
            id=new_id,
            description=f"Control sequence from {seq_id}"
        )
        combined_records.append(record)
        metadata_rows.append({
            "eggnog_id": new_id,
            "original_id": seq_id,
            "group": "Control",
            "length": len(seq)
        })
    
    # Write combined FASTA
    SeqIO.write(combined_records, COMBINED_FOR_EGGNOG, "fasta")
    print(f"Written {len(combined_records)} sequences to: {COMBINED_FOR_EGGNOG}")
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(SEQUENCE_METADATA, index=False)
    print(f"Saved metadata to: {SEQUENCE_METADATA}")
    
    # Compute biophysical properties
    print("\n--- Computing biophysical properties ---")
    property_rows = []
    
    all_sequences = {**utp_sequences, **control_sequences}
    groups = {**{k: "uTP" for k in utp_sequences}, **{k: "Control" for k in control_sequences}}
    
    for seq_id, seq in tqdm(all_sequences.items(), desc="Computing properties"):
        props = get_protein_properties(seq)
        if props:
            props["sequence_id"] = seq_id
            props["group"] = groups[seq_id]
            property_rows.append(props)
    
    properties_df = pd.DataFrame(property_rows)
    properties_df.to_csv(PROPERTIES_FILE, index=False)
    print(f"Saved properties for {len(properties_df)} sequences to: {PROPERTIES_FILE}")
    
    # Summary statistics
    print("\n--- Summary Statistics ---")
    print(f"\nBy group:")
    print(properties_df.groupby("group").agg({
        "sequence_id": "count",
        "length": ["mean", "std"],
        "isoelectric_point": ["mean", "std"],
        "fraction_coil": ["mean", "std"],
        "instability_index": ["mean", "std"]
    }).round(2))
    
    # Quick property comparison (preview)
    print("\n--- Quick Property Comparison ---")
    utp_props = properties_df[properties_df["group"] == "uTP"]
    ctrl_props = properties_df[properties_df["group"] == "Control"]
    
    key_props = ["isoelectric_point", "fraction_coil", "instability_index", "gravy"]
    for prop in key_props:
        utp_mean = utp_props[prop].mean()
        ctrl_mean = ctrl_props[prop].mean()
        diff = utp_mean - ctrl_mean
        # Cohen's d
        pooled_std = np.sqrt((utp_props[prop].std()**2 + ctrl_props[prop].std()**2) / 2)
        d = diff / pooled_std if pooled_std > 0 else 0
        print(f"{prop}: uTP={utp_mean:.3f}, Control={ctrl_mean:.3f}, diff={diff:+.3f}, d={d:+.2f}")
    
    # Instructions for next steps
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Submit sequences to eggNOG-mapper web service:
   https://eggnog-mapper.embl.de/
   
   Upload: output/sequences_for_eggnog.fasta
   Settings:
   - Database: eggNOG 5.0
   - Taxonomic scope: Auto (or Eukaryota)
   - Output: All annotations
   
2. Download results and save to:
   output/eggnog_results.emapper.annotations
   
3. Run the next script:
   uv run python experiments/utp_functional_annotation/03_parse_annotations.py
""")
    
    return properties_df


if __name__ == "__main__":
    main()
