#!/usr/bin/env python3
"""
Extract full sequences for UCYN-A enriched proteins.

Reads protein IDs from good-c-term-gb.fasta (cropped C-terminal sequences)
and extracts full sequences from the proteome database.
"""

from pathlib import Path


def parse_fasta_ids(fasta_path: Path) -> set[str]:
    """Extract all sequence IDs from a FASTA file."""
    ids = set()
    with open(fasta_path) as f:
        for line in f:
            if line.startswith('>'):
                # ID is everything after '>' up to first whitespace
                seq_id = line[1:].strip().split()[0]
                ids.add(seq_id)
    return ids


def extract_sequences(proteome_path: Path, target_ids: set[str]) -> dict[str, str]:
    """Extract full sequences for target IDs from proteome FASTA."""
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(proteome_path) as f:
        for line in f:
            if line.startswith('>'):
                # Save previous sequence if it was a target
                if current_id and current_id in target_ids:
                    sequences[current_id] = ''.join(current_seq)
                
                # Start new sequence
                current_id = line[1:].strip().split()[0]
                current_seq = []
            else:
                current_seq.append(line.strip())
        
        # Don't forget the last sequence
        if current_id and current_id in target_ids:
            sequences[current_id] = ''.join(current_seq)
    
    return sequences


def write_fasta(sequences: dict[str, str], output_path: Path):
    """Write sequences to FASTA file."""
    with open(output_path, 'w') as f:
        for seq_id, seq in sequences.items():
            f.write(f'>{seq_id}\n')
            # Write sequence in 70-character lines
            for i in range(0, len(seq), 70):
                f.write(seq[i:i+70] + '\n')


def main():
    project_root = Path(__file__).parent.parent
    
    # Input files
    c_term_fasta = project_root / 'experiments/utp_motif_analysis/data/good-c-term-gb.fasta'
    proteome_fasta = project_root / 'data/ADK1075_proteomics_DB_2.fasta'
    
    # Output file
    output_fasta = project_root / 'data/ucyna_enriched_full_sequences.fasta'
    
    print(f"Reading IDs from: {c_term_fasta}")
    target_ids = parse_fasta_ids(c_term_fasta)
    print(f"Found {len(target_ids)} target protein IDs")
    
    print(f"Extracting sequences from: {proteome_fasta}")
    sequences = extract_sequences(proteome_fasta, target_ids)
    print(f"Found {len(sequences)} matching sequences")
    
    # Check for missing IDs
    missing = target_ids - set(sequences.keys())
    if missing:
        print(f"\nWarning: {len(missing)} IDs not found in proteome:")
        for mid in sorted(missing)[:10]:
            print(f"  - {mid}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    
    print(f"\nWriting sequences to: {output_fasta}")
    write_fasta(sequences, output_fasta)
    print("Done!")


if __name__ == '__main__':
    main()
