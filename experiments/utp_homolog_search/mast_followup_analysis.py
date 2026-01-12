#!/usr/bin/env python3
"""
Follow-up analysis of MAST motif search results.

Addresses three key questions:
1. Do any proteins have MULTIPLE motifs co-occurring?
2. What are the C-terminal MEME-1/2 hits and are they orthologs of B. bigelowii uTP genes?
3. What's the false positive rate in a non-haptophyte proteome (Arabidopsis)?
"""

import subprocess
import shutil
from pathlib import Path
from collections import defaultdict
import pandas as pd
import xml.etree.ElementTree as ET
from Bio import SeqIO
import urllib.request
import gzip


# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
MEME_FILE = PROJECT_ROOT / "experiments/utp_motif_analysis/data/meme_gb.xml"
MAST_PATH = "/opt/local/bin/mast"

# uTP region definition
UTP_REGION = 150  # amino acids from C-terminus


def load_mast_results() -> pd.DataFrame:
    """Load the MAST results from the previous analysis."""
    results_file = OUTPUT_DIR / "mast_all_hits.csv"
    if not results_file.exists():
        raise FileNotFoundError(f"Run mast_motif_search.py first: {results_file}")
    return pd.read_csv(results_file)


def analyze_motif_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Question 1: Find proteins with multiple motifs co-occurring.
    
    Key question: If MEME-1 and MEME-2 hit the same protein's C-terminus 
    in the correct order, that's a potential uTP precursor.
    """
    print("\n" + "=" * 80)
    print("QUESTION 1: MOTIF CO-OCCURRENCE ANALYSIS")
    print("=" * 80)
    
    # Group hits by sequence
    seq_motifs = df.groupby(["organism", "sequence_name"]).agg({
        "motif_alt": list,
        "position": list,
        "position_from_cterm": list,
        "pvalue": list,
        "sequence_length": "first",
    }).reset_index()
    
    # Count motifs per sequence
    seq_motifs["num_motifs"] = seq_motifs["motif_alt"].apply(len)
    seq_motifs["unique_motifs"] = seq_motifs["motif_alt"].apply(lambda x: len(set(x)))
    
    # Filter to sequences with multiple motifs
    multi_motif = seq_motifs[seq_motifs["num_motifs"] > 1].copy()
    
    print(f"\nTotal sequences with hits: {len(seq_motifs)}")
    print(f"Sequences with multiple motif hits: {len(multi_motif)}")
    
    # Breakdown by number of motifs
    for n in range(2, multi_motif["num_motifs"].max() + 1):
        count = len(multi_motif[multi_motif["num_motifs"] == n])
        if count > 0:
            print(f"  Sequences with {n} motifs: {count}")
    
    # Key analysis: Look for MEME-1 + MEME-2 co-occurrence (core uTP motifs)
    print("\n--- Core uTP Motif Co-occurrence (MEME-1 + MEME-2) ---")
    
    def has_core_motifs(motifs):
        return "MEME-1" in motifs and "MEME-2" in motifs
    
    core_cooccur = multi_motif[multi_motif["motif_alt"].apply(has_core_motifs)]
    print(f"Sequences with both MEME-1 AND MEME-2: {len(core_cooccur)}")
    
    if len(core_cooccur) > 0:
        print("\nProteins with MEME-1 + MEME-2 co-occurrence:")
        for _, row in core_cooccur.iterrows():
            print(f"\n  {row['organism']}: {row['sequence_name']}")
            print(f"    Sequence length: {row['sequence_length']} aa")
            # Show motif positions
            for motif, pos, cterm_pos, pval in zip(
                row["motif_alt"], row["position"], 
                row["position_from_cterm"], row["pvalue"]
            ):
                cterm_flag = " [C-TERMINAL]" if cterm_pos <= UTP_REGION else ""
                print(f"    {motif}: pos={pos}, from_cterm={cterm_pos}, p={pval:.2e}{cterm_flag}")
    
    # Check for C-terminal co-occurrence specifically
    print("\n--- C-terminal Co-occurrence Analysis ---")
    
    def has_cterm_core_motifs(row):
        """Check if both MEME-1 and MEME-2 are in C-terminal region."""
        cterm_motifs = set()
        for motif, cterm_pos in zip(row["motif_alt"], row["position_from_cterm"]):
            if cterm_pos <= UTP_REGION:
                cterm_motifs.add(motif)
        return "MEME-1" in cterm_motifs and "MEME-2" in cterm_motifs
    
    cterm_core_cooccur = multi_motif[multi_motif.apply(has_cterm_core_motifs, axis=1)]
    print(f"Sequences with MEME-1 + MEME-2 BOTH in C-terminal region: {len(cterm_core_cooccur)}")
    
    if len(cterm_core_cooccur) > 0:
        print("\n*** POTENTIAL uTP PRECURSORS ***")
        for _, row in cterm_core_cooccur.iterrows():
            print(f"\n  {row['organism']}: {row['sequence_name']}")
            print(f"    Sequence length: {row['sequence_length']} aa")
            # Check order: In uTP, MEME-2 comes before MEME-1
            meme1_pos = None
            meme2_pos = None
            for motif, pos, cterm_pos in zip(
                row["motif_alt"], row["position"], row["position_from_cterm"]
            ):
                if motif == "MEME-1" and cterm_pos <= UTP_REGION:
                    meme1_pos = pos
                if motif == "MEME-2" and cterm_pos <= UTP_REGION:
                    meme2_pos = pos
            
            if meme1_pos and meme2_pos:
                if meme2_pos < meme1_pos:
                    print(f"    ORDER: MEME-2 ({meme2_pos}) → MEME-1 ({meme1_pos}) ✓ CORRECT uTP ORDER")
                else:
                    print(f"    ORDER: MEME-1 ({meme1_pos}) → MEME-2 ({meme2_pos}) ✗ Reversed")
    
    # Also check for other common motif combinations
    print("\n--- Most Common Motif Combinations ---")
    combo_counts = defaultdict(int)
    for motifs in multi_motif["motif_alt"]:
        combo = tuple(sorted(set(motifs)))
        combo_counts[combo] += 1
    
    for combo, count in sorted(combo_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {' + '.join(combo)}: {count} proteins")
    
    return multi_motif


def investigate_cterm_core_hits(df: pd.DataFrame):
    """
    Question 2: What are the C-terminal MEME-1/2 hits?
    What proteins are they in? Are they orthologs of B. bigelowii uTP genes?
    """
    print("\n" + "=" * 80)
    print("QUESTION 2: C-TERMINAL CORE MOTIF HITS")
    print("=" * 80)
    
    # Filter to C-terminal MEME-1 and MEME-2 hits
    core_cterm = df[
        (df["motif_alt"].isin(["MEME-1", "MEME-2"])) & 
        (df["position_from_cterm"] <= UTP_REGION)
    ].copy()
    
    print(f"\nC-terminal core motif hits: {len(core_cterm)}")
    print(f"  MEME-1: {len(core_cterm[core_cterm['motif_alt'] == 'MEME-1'])}")
    print(f"  MEME-2: {len(core_cterm[core_cterm['motif_alt'] == 'MEME-2'])}")
    
    # Get unique proteins
    unique_proteins = core_cterm.groupby(["organism", "sequence_name"]).agg({
        "motif_alt": list,
        "position": list,
        "position_from_cterm": list,
        "pvalue": list,
        "sequence_length": "first",
    }).reset_index()
    
    print(f"\nUnique proteins with C-terminal core motif hits: {len(unique_proteins)}")
    
    print("\n--- Detailed Protein Information ---")
    for _, row in unique_proteins.iterrows():
        print(f"\nProtein: {row['sequence_name']}")
        print(f"  Organism: {row['organism']}")
        print(f"  Length: {row['sequence_length']} aa")
        print(f"  Motif hits:")
        for motif, pos, cterm_pos, pval in zip(
            row["motif_alt"], row["position"], 
            row["position_from_cterm"], row["pvalue"]
        ):
            print(f"    {motif}: pos={pos}, from_cterm={cterm_pos}, p={pval:.2e}")
    
    # Try to get protein annotations (if available)
    print("\n--- Protein Annotations (from FASTA headers) ---")
    
    # Load proteome files to get headers
    proteome_dir = DATA_DIR / "haptophyte_proteomes"
    protein_headers = {}
    
    if proteome_dir.exists():
        for fasta_file in proteome_dir.glob("*.faa"):
            try:
                for record in SeqIO.parse(fasta_file, "fasta"):
                    protein_headers[record.id] = record.description
            except Exception as e:
                print(f"  Could not load {fasta_file.name}: {e}")
    
    for _, row in unique_proteins.iterrows():
        seq_name = row["sequence_name"]
        if seq_name in protein_headers:
            desc = protein_headers[seq_name]
            # Clean up description
            if seq_name in desc:
                desc = desc.replace(seq_name, "").strip()
            print(f"\n{seq_name}:")
            print(f"  {desc[:200]}")
    
    # Save list of candidate proteins for manual inspection
    candidates_file = OUTPUT_DIR / "cterm_core_motif_candidates.csv"
    unique_proteins.to_csv(candidates_file, index=False)
    print(f"\nSaved candidate proteins to: {candidates_file}")
    
    return unique_proteins


def download_arabidopsis_proteome() -> Path:
    """Download Arabidopsis thaliana proteome for false positive analysis."""
    output_file = OUTPUT_DIR / "arabidopsis_proteome.fasta"
    
    if output_file.exists():
        print(f"Arabidopsis proteome already downloaded: {output_file}")
        return output_file
    
    print("Downloading Arabidopsis thaliana proteome from UniProt...")
    # UniProt reference proteome for A. thaliana
    url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000006548/UP000006548_3702.fasta.gz"
    
    gz_file = OUTPUT_DIR / "arabidopsis_proteome.fasta.gz"
    
    try:
        urllib.request.urlretrieve(url, gz_file)
        
        # Decompress
        with gzip.open(gz_file, "rt") as f_in:
            with open(output_file, "w") as f_out:
                f_out.write(f_in.read())
        
        gz_file.unlink()  # Remove compressed file
        print(f"Downloaded and extracted: {output_file}")
        
        # Count sequences
        num_seqs = sum(1 for _ in SeqIO.parse(output_file, "fasta"))
        print(f"Arabidopsis proteome: {num_seqs} proteins")
        
    except Exception as e:
        print(f"Failed to download from UniProt: {e}")
        print("Trying NCBI RefSeq...")
        
        # Alternative: NCBI RefSeq
        url = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/735/GCF_000001735.4_TAIR10.1/GCF_000001735.4_TAIR10.1_protein.faa.gz"
        try:
            urllib.request.urlretrieve(url, gz_file)
            with gzip.open(gz_file, "rt") as f_in:
                with open(output_file, "w") as f_out:
                    f_out.write(f_in.read())
            gz_file.unlink()
            print(f"Downloaded from NCBI: {output_file}")
        except Exception as e2:
            print(f"Also failed: {e2}")
            raise RuntimeError("Could not download Arabidopsis proteome")
    
    return output_file


def run_mast_on_outgroup(proteome_file: Path, name: str) -> pd.DataFrame:
    """Run MAST on an outgroup proteome for false positive rate estimation."""
    output_dir = OUTPUT_DIR / f"mast_outgroup_{name}"
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    cmd = [
        MAST_PATH,
        str(MEME_FILE),
        str(proteome_file),
        "-o", str(output_dir),
        "-ev", "10.0",
        "-mt", "0.0001",
    ]
    
    print(f"\nRunning MAST on {name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"MAST error: {result.stderr}")
        return pd.DataFrame()
    
    # Parse results
    xml_file = output_dir / "mast.xml"
    if not xml_file.exists():
        print(f"No XML output found")
        return pd.DataFrame()
    
    return parse_mast_xml_simple(xml_file)


def parse_mast_xml_simple(xml_file: Path) -> pd.DataFrame:
    """Simple MAST XML parser."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Get motif info
    motif_info = {}
    for motif in root.findall(".//motif"):
        idx = motif.get("idx")
        motif_info[idx] = {
            "id": motif.get("id"),
            "alt": motif.get("alt"),
            "width": int(motif.get("width", 0)),
        }
    
    hits = []
    for seq in root.findall(".//sequence"):
        seq_name = seq.get("name")
        seq_length = int(seq.get("length", 0))
        
        for seg in seq.findall(".//seg"):
            for hit in seg.findall(".//hit"):
                motif_idx = hit.get("idx")
                info = motif_info.get(motif_idx, {})
                pos = int(hit.get("pos", 0))
                
                hits.append({
                    "sequence_name": seq_name,
                    "sequence_length": seq_length,
                    "motif_alt": info.get("alt", f"MEME-{motif_idx}"),
                    "motif_width": info.get("width", 0),
                    "position": pos,
                    "position_from_cterm": seq_length - pos - info.get("width", 0),
                    "pvalue": float(hit.get("pvalue", 1.0)),
                })
    
    return pd.DataFrame(hits)


def analyze_false_positive_rate(haptophyte_df: pd.DataFrame):
    """
    Question 3: What's the false positive rate?
    Compare hit rates between haptophytes and Arabidopsis (outgroup).
    """
    print("\n" + "=" * 80)
    print("QUESTION 3: FALSE POSITIVE RATE ANALYSIS")
    print("=" * 80)
    
    # Download and run on Arabidopsis
    try:
        arab_proteome = download_arabidopsis_proteome()
        arab_df = run_mast_on_outgroup(arab_proteome, "arabidopsis")
    except Exception as e:
        print(f"Could not analyze Arabidopsis: {e}")
        return
    
    if arab_df.empty:
        print("No results from Arabidopsis search")
        return
    
    # Count sequences in each proteome
    arab_num_seqs = sum(1 for _ in SeqIO.parse(arab_proteome, "fasta"))
    
    # Calculate statistics
    print("\n--- Comparison: Haptophytes vs Arabidopsis ---")
    
    # Get haptophyte stats from loaded data
    hapto_num_seqs = len(haptophyte_df["sequence_name"].unique())
    hapto_total_hits = len(haptophyte_df)
    hapto_cterm_hits = len(haptophyte_df[haptophyte_df["position_from_cterm"] <= UTP_REGION])
    
    arab_total_hits = len(arab_df)
    arab_cterm_hits = len(arab_df[arab_df["position_from_cterm"] <= UTP_REGION])
    
    print(f"\nTotal hits:")
    print(f"  Haptophytes: {hapto_total_hits} hits in {hapto_num_seqs} unique sequences")
    print(f"  Arabidopsis: {arab_total_hits} hits in {len(arab_df['sequence_name'].unique())} unique sequences")
    
    # Normalize by proteome size
    print(f"\nHits per 1000 proteins:")
    hapto_rate = hapto_total_hits / hapto_num_seqs * 1000
    arab_rate = arab_total_hits / arab_num_seqs * 1000
    print(f"  Haptophytes: {hapto_rate:.1f}")
    print(f"  Arabidopsis: {arab_rate:.1f}")
    
    print(f"\nC-terminal hits (within {UTP_REGION} aa of C-terminus):")
    print(f"  Haptophytes: {hapto_cterm_hits} ({hapto_cterm_hits/hapto_total_hits*100:.1f}%)")
    print(f"  Arabidopsis: {arab_cterm_hits} ({arab_cterm_hits/arab_total_hits*100:.1f}% if hits > 0)")
    
    # Per-motif comparison
    print("\n--- Per-Motif Comparison ---")
    print(f"{'Motif':<10} {'Haptophyte':<15} {'Arabidopsis':<15} {'Ratio':<10}")
    print("-" * 50)
    
    all_motifs = sorted(set(haptophyte_df["motif_alt"].unique()) | set(arab_df["motif_alt"].unique()))
    
    for motif in all_motifs:
        hapto_count = len(haptophyte_df[haptophyte_df["motif_alt"] == motif])
        arab_count = len(arab_df[arab_df["motif_alt"] == motif])
        ratio = hapto_count / arab_count if arab_count > 0 else float("inf")
        print(f"{motif:<10} {hapto_count:<15} {arab_count:<15} {ratio:.2f}")
    
    # Core motif analysis
    print("\n--- Core uTP Motifs (MEME-1 & MEME-2) ---")
    for motif in ["MEME-1", "MEME-2"]:
        hapto_total = len(haptophyte_df[haptophyte_df["motif_alt"] == motif])
        hapto_cterm = len(haptophyte_df[
            (haptophyte_df["motif_alt"] == motif) & 
            (haptophyte_df["position_from_cterm"] <= UTP_REGION)
        ])
        arab_total = len(arab_df[arab_df["motif_alt"] == motif])
        arab_cterm = len(arab_df[
            (arab_df["motif_alt"] == motif) & 
            (arab_df["position_from_cterm"] <= UTP_REGION)
        ])
        
        print(f"\n{motif}:")
        print(f"  Haptophytes: {hapto_total} total, {hapto_cterm} C-terminal")
        print(f"  Arabidopsis: {arab_total} total, {arab_cterm} C-terminal")
    
    # Save Arabidopsis results
    arab_output = OUTPUT_DIR / "mast_arabidopsis_hits.csv"
    arab_df.to_csv(arab_output, index=False)
    print(f"\nSaved Arabidopsis results to: {arab_output}")
    
    # Interpretation
    print("\n--- INTERPRETATION ---")
    if arab_rate > hapto_rate * 0.8:
        print("⚠️  Arabidopsis shows SIMILAR hit rates to haptophytes.")
        print("   This suggests these motifs are common protein patterns,")
        print("   NOT haptophyte-specific sequences.")
    elif arab_rate > hapto_rate * 0.5:
        print("⚠️  Arabidopsis shows MODERATE hit rates (~50-80% of haptophytes).")
        print("   Some motifs may be generic, but there could be haptophyte enrichment.")
    else:
        print("✓  Arabidopsis shows LOWER hit rates than haptophytes.")
        print("   This suggests some specificity to haptophyte sequences.")
    
    return arab_df


def main():
    """Run all follow-up analyses."""
    print("=" * 80)
    print("MAST FOLLOW-UP ANALYSIS")
    print("=" * 80)
    
    # Load results
    df = load_mast_results()
    print(f"Loaded {len(df)} hits from previous analysis")
    
    # Question 1: Motif co-occurrence
    multi_motif = analyze_motif_cooccurrence(df)
    
    # Question 2: C-terminal core hits
    candidates = investigate_cterm_core_hits(df)
    
    # Question 3: False positive rate
    arab_df = analyze_false_positive_rate(df)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key findings to report:
1. Multi-motif proteins: Check if any have uTP-like patterns
2. C-terminal core hits: Best candidates for uTP precursors  
3. False positive rate: If similar to Arabidopsis, motifs are generic
""")


if __name__ == "__main__":
    main()
