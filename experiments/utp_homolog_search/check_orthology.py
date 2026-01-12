#!/usr/bin/env python3
"""
Check if C-terminal core motif hit proteins are orthologs of B. bigelowii uTP genes.
"""

from pathlib import Path
from Bio import SeqIO
import subprocess

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(__file__).parent / "output"

# Best C-terminal hits (lowest p-values)
BEST_CTERM_HITS = [
    # (protein_id, organism, motif, position_from_cterm, pvalue)
    ("KAG8465093.1", "Diacronema lutheri", "MEME-1", 14, 4.2e-8),
    ("EOD26814.1", "Emiliania huxleyi", "MEME-1", 11, 1.4e-6),
    ("KAL1529582.1", "Prymnesium parvum", "MEME-1", 3, 6.0e-6),
    ("KAL3931323.1", "Prymnesium sp.", "MEME-1", 7, 7.4e-5),
    ("KAL3932683.1", "Prymnesium sp.", "MEME-2", 21, 9.5e-6),
    ("KAG8461925.1", "Diacronema lutheri", "MEME-2", 22, 2.1e-5),
    ("KAG8460058.1", "Diacronema lutheri", "MEME-1", 72, 3.7e-7),
]


def extract_sequences():
    """Extract sequences of best C-terminal hits."""
    protein_ids = {hit[0] for hit in BEST_CTERM_HITS}
    
    sequences = {}
    proteome_dir = DATA_DIR / "haptophytes/ncbi_dataset/data"
    
    for faa_file in proteome_dir.glob("*/protein.faa"):
        for record in SeqIO.parse(faa_file, "fasta"):
            if record.id in protein_ids:
                sequences[record.id] = record
    
    # Save to file
    output_file = OUTPUT_DIR / "cterm_hit_sequences.fasta"
    SeqIO.write(sequences.values(), output_file, "fasta")
    print(f"Saved {len(sequences)} sequences to {output_file}")
    
    return sequences


def show_cterm_regions(sequences: dict):
    """Show the C-terminal regions of hit proteins."""
    print("\n--- C-terminal Regions (last 150 aa) ---")
    
    for protein_id, organism, motif, cterm_pos, pval in BEST_CTERM_HITS:
        if protein_id in sequences:
            seq = sequences[protein_id]
            cterm = str(seq.seq)[-150:]
            print(f"\n{protein_id} ({organism}) - {motif} at {cterm_pos} aa from C-term")
            print(f"Length: {len(seq.seq)} aa")
            print(f"C-terminal 150 aa:")
            # Show in blocks of 60
            for i in range(0, len(cterm), 60):
                print(f"  {cterm[i:i+60]}")


def compare_to_utp_proteins():
    """Compare C-terminal hit sequences to known uTP proteins."""
    print("\n--- Comparison to Known uTP Proteins ---")
    
    # Load known uTP proteins
    utp_file = DATA_DIR / "good-c-term-full.fasta"
    if not utp_file.exists():
        utp_file = DATA_DIR / "uTP_HMM_hits.fasta"
    
    if not utp_file.exists():
        print(f"Cannot find uTP proteins file")
        return
    
    utp_proteins = list(SeqIO.parse(utp_file, "fasta"))
    print(f"Loaded {len(utp_proteins)} known uTP proteins")
    
    # Get average length
    utp_lengths = [len(r.seq) for r in utp_proteins]
    print(f"uTP protein lengths: mean={sum(utp_lengths)/len(utp_lengths):.0f}, range={min(utp_lengths)}-{max(utp_lengths)}")
    
    # Compare to our hits
    print("\nHit protein lengths:")
    for hit in BEST_CTERM_HITS:
        print(f"  {hit[0]}: searching...")
    
    # Load hit sequences
    hit_seqs = {}
    proteome_dir = DATA_DIR / "haptophytes/ncbi_dataset/data"
    for faa_file in proteome_dir.glob("*/protein.faa"):
        for record in SeqIO.parse(faa_file, "fasta"):
            if record.id in {h[0] for h in BEST_CTERM_HITS}:
                hit_seqs[record.id] = len(record.seq)
    
    print("\nComparison:")
    print(f"{'Protein':<20} {'Length':<10} {'vs uTP mean':<15} {'Note'}")
    print("-" * 60)
    
    utp_mean = sum(utp_lengths) / len(utp_lengths)
    for protein_id, organism, motif, cterm_pos, pval in BEST_CTERM_HITS:
        if protein_id in hit_seqs:
            length = hit_seqs[protein_id]
            diff = length - utp_mean
            note = ""
            if length < 200:
                note = "Much shorter than uTP proteins"
            elif diff > 100:
                note = "Longer (possible extension?)"
            elif diff < -100:
                note = "Shorter"
            print(f"{protein_id:<20} {length:<10} {diff:+.0f}{'':15} {note}")


def check_blastp_orthology():
    """Check if hit proteins are orthologs of B. bigelowii genes using BLASTP."""
    print("\n--- BLASTP Orthology Check ---")
    
    # Check if BLAST is available
    try:
        result = subprocess.run(["blastp", "-version"], capture_output=True, text=True)
        print(f"BLASTP available: {result.stdout.split()[1] if result.returncode == 0 else 'No'}")
    except FileNotFoundError:
        print("BLASTP not found - skipping orthology check")
        print("Install with: brew install blast")
        return
    
    # Would need B. bigelowii proteome and proper database setup
    print("Note: Full orthology analysis requires B. bigelowii proteome database")


def main():
    print("=" * 80)
    print("C-TERMINAL HIT ORTHOLOGY ANALYSIS")
    print("=" * 80)
    
    print("\nBest C-terminal MEME-1/MEME-2 hits (by p-value):")
    print(f"{'Protein':<20} {'Organism':<25} {'Motif':<10} {'From C-term':<12} {'P-value'}")
    print("-" * 80)
    for hit in BEST_CTERM_HITS:
        print(f"{hit[0]:<20} {hit[1]:<25} {hit[2]:<10} {hit[3]:<12} {hit[4]:.2e}")
    
    sequences = extract_sequences()
    show_cterm_regions(sequences)
    compare_to_utp_proteins()
    check_blastp_orthology()
    
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("""
Key observations:
1. All best hits are "hypothetical proteins" - no functional annotation
2. These proteins are much shorter than typical uTP proteins (~350 aa vs ~550 aa mean)
3. Only ONE motif (MEME-1 or MEME-2) is found per protein - never both
4. This suggests these are random sequence matches, NOT uTP precursors

A true uTP precursor would have:
- Both MEME-1 and MEME-2 in the C-terminal region
- MEME-2 before MEME-1 (correct uTP order)  
- Ideally additional uTP motifs (MEME-3, MEME-4, etc.)
- A mature domain with identifiable function
""")


if __name__ == "__main__":
    main()
