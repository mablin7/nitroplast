#!/usr/bin/env python3
"""
uTP Motif Combination Extraction

This script parses MEME output to extract and analyze motif combinations
in UCYN-A transit peptide (uTP) sequences.

Outputs:
- motif_combinations.csv: All motif combinations and their counts
- top_utp_variants.csv: Top 5 uTP variants with motif compositions
- motif_positions.csv: Relative positions of motifs
- motif_combinations.svg: Visualization of top variants
- motif_tree.svg: Motif transition graph

Usage:
    uv run python experiments/utp_motif_analysis/motif_combination_extraction.py
"""

import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from Bio.Seq import Seq
from scipy.stats import gaussian_kde

# Configuration
C_TERM_START = 880  # Start of uTP region in MSA
C_TERM_END = 1010   # End of uTP region in MSA
MIN_COVERAGE = 0.6  # Minimum fraction of C-term region that must be non-gap
TOP_N_VARIANTS = 5  # Number of top variants to analyze

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Input files
MSA_FILE = DATA_DIR / "ucyn-a_enriched_cobalt_cleaned.fa"
MEME_FILE = DATA_DIR / "meme_gb.xml"

# Styling
sns.set_theme("paper")
sns.set(font_scale=1.3)


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_good_c_terminal_sequences(msa_file: Path) -> tuple[list, list]:
    """
    Extract sequences with good C-terminal coverage from MSA.
    
    Returns:
        Tuple of (c_term_sequences, full_sequences) as lists of SeqRecord objects
    """
    from Bio import AlignIO
    
    all_aligns = AlignIO.read(msa_file, "fasta")
    align_len = C_TERM_END - C_TERM_START
    
    c_term_seqs = []
    full_seqs = []
    
    for record in all_aligns:
        # Parse ID from description
        seq_id = " ".join(record.description.split(" ")[1:]) if " " in record.description else record.id
        seq = str(record.seq)
        
        # Extract C-terminal region
        c_term_seq = seq[C_TERM_START:C_TERM_END + 1]
        c_term_no_gaps = c_term_seq.replace("-", "")
        
        # Check coverage threshold
        if len(c_term_no_gaps) / align_len > MIN_COVERAGE:
            c_term_seqs.append(
                SeqIO.SeqRecord(Seq(seq[C_TERM_START:]), id=seq_id, description="")
            )
            full_seq = seq.replace("-", "")
            full_seqs.append(
                SeqIO.SeqRecord(Seq(full_seq), id=seq_id, description="")
            )
    
    print(f"Extracted {len(c_term_seqs)} sequences with good C-terminal coverage")
    return c_term_seqs, full_seqs


def parse_meme_xml(meme_file: Path) -> ET.ElementTree:
    """Parse MEME XML file."""
    return ET.parse(meme_file)


def extract_scanned_sites(meme_xml: ET.ElementTree) -> list[tuple]:
    """Extract scanned sites from MEME XML."""
    sites = meme_xml.findall(".//scanned_sites")
    scanned_sites = [
        (tag.attrib["sequence_id"], site.attrib)
        for tag in sites
        for site in tag.findall("scanned_site")
    ]
    return scanned_sites


def get_motif_sequences(meme_xml: ET.ElementTree) -> dict[str, str]:
    """Extract motif consensus sequences from MEME XML."""
    return {
        tag.attrib["id"]: tag.attrib["name"] 
        for tag in meme_xml.findall(".//motif")
    }


def get_sequence_names(meme_xml: ET.ElementTree) -> dict[str, str]:
    """Extract sequence ID to name mapping from MEME XML."""
    return {
        tag.attrib["id"]: tag.attrib["name"] 
        for tag in meme_xml.findall(".//sequence")
    }


def compute_relative_positions(
    scanned_sites: list[tuple], 
    reference_motif: str = "motif_1"
) -> list[tuple]:
    """
    Compute positions of all motifs relative to a reference motif.
    
    Args:
        scanned_sites: List of (sequence_id, motif_attributes) tuples
        reference_motif: Motif ID to use as position reference
    
    Returns:
        List of (sequence_id, motif_id, relative_position) tuples
    """
    # Find positions of reference motif for each sequence
    ref_positions = {
        seq: int(motif["position"])
        for seq, motif in scanned_sites
        if motif["motif_id"] == reference_motif
    }
    
    # Compute relative positions for all other motifs
    relative_positions = [
        (seq, motif["motif_id"], int(motif["position"]) - ref_positions[seq])
        for seq, motif in scanned_sites
        if seq in ref_positions and motif["motif_id"] != reference_motif
    ]
    
    return relative_positions


def extract_motif_combinations(scanned_sites: list[tuple]) -> dict[str, list[dict]]:
    """
    Group motifs by sequence and sort by position.
    
    Returns:
        Dictionary mapping sequence_id to sorted list of motif dictionaries
    """
    sequences = defaultdict(list)
    for seq, motif in scanned_sites:
        sequences[seq].append(motif)
    
    # Sort motifs by position within each sequence
    sequences = {
        seq: sorted(motifs, key=lambda m: int(m["position"]))
        for seq, motifs in sequences.items()
    }
    
    return dict(sequences)


def count_motif_combinations(sequences: dict[str, list[dict]]) -> Counter:
    """Count occurrences of each unique motif combination."""
    return Counter(
        tuple(motif["motif_id"] for motif in motifs)
        for motifs in sequences.values()
    )


def analyze_inter_motif_regions(
    sequences: dict[str, list[dict]],
    full_seqs: dict[str, str],
    motif_seqs: dict[str, str],
    seq_names: dict[str, str]
) -> tuple[dict, dict]:
    """
    Analyze the amino acid sequences between motifs.
    
    Returns:
        Tuple of (amino_acid_distributions, mean_lengths) for each motif pair
    """
    # Map sequence IDs to names and get motif positions
    seq_motifs_with_pos = {
        seq_names[seq_id]: [
            (m["motif_id"], int(m["position"])) for m in motifs
        ]
        for seq_id, motifs in sequences.items()
        if seq_id in seq_names
    }
    
    # Extract inter-motif sequences
    in_between_motifs = defaultdict(list)
    
    for seq_name, motifs in seq_motifs_with_pos.items():
        if seq_name not in full_seqs:
            continue
        seq = full_seqs[seq_name]
        
        prev_motif = None
        prev_end = -1
        
        for motif_id, pos in motifs:
            if prev_motif is not None:
                inter_seq = seq[prev_end:pos]
                if inter_seq:
                    in_between_motifs[(prev_motif, motif_id)].append(inter_seq)
            prev_motif = motif_id
            prev_end = pos + len(motif_seqs[motif_id])
    
    # Compute statistics
    aa_distributions = {
        k: Counter("".join(v)) 
        for k, v in in_between_motifs.items()
    }
    mean_lengths = {
        k: np.mean([len(s) for s in v]) 
        for k, v in in_between_motifs.items()
    }
    
    return aa_distributions, mean_lengths


def construct_utp(
    motifs: tuple[str, ...],
    motif_seqs: dict[str, str],
    aa_distributions: dict[tuple, Counter],
    mean_lengths: dict[tuple, float]
) -> str:
    """
    Construct a synthetic uTP sequence based on a motif combination.
    
    Args:
        motifs: Tuple of motif IDs in order
        motif_seqs: Dictionary of motif ID to consensus sequence
        aa_distributions: Amino acid distributions for inter-motif regions
        mean_lengths: Mean lengths of inter-motif regions
    
    Returns:
        Synthetic uTP sequence string
    """
    def sample_aa_dist(n: int, aa_dist: Counter) -> str:
        """Sample amino acids from distribution."""
        if not aa_dist:
            return ""
        choices = "".join(k * v for k, v in aa_dist.items())
        aa = np.random.choice(list(choices), int(n))
        return "".join(aa)
    
    seq_parts = []
    for motif1, motif2 in zip(motifs[:-1], motifs[1:]):
        seq_parts.append(motif_seqs[motif1])
        if (motif1, motif2) in mean_lengths:
            inter_len = round(mean_lengths[(motif1, motif2)])
            inter_aa = aa_distributions.get((motif1, motif2), Counter())
            seq_parts.append(sample_aa_dist(inter_len, inter_aa))
    seq_parts.append(motif_seqs[motifs[-1]])
    
    return "".join(seq_parts)


def plot_motif_combinations(
    sequences: dict[str, list[dict]],
    motif_combination_counts: Counter,
    motif_seqs: dict[str, str],
    output_file: Path,
    top_n: int = 5
):
    """
    Plot the distribution of top motif combinations.
    
    Creates a two-panel figure:
    - Left: Relative positions of motifs in each variant
    - Right: Bar chart of occurrence counts
    """
    sns.set_style("ticks", {"ytick.left": False})
    
    # Get top N combinations
    most_common = sorted(
        motif_combination_counts.keys(),
        key=lambda c: motif_combination_counts[c],
        reverse=True
    )[:top_n]
    
    cmap = matplotlib.colormaps["tab20"]
    motif_names = sorted(
        set(motif_seqs.keys()),
        key=lambda m: int(m.split("_")[1])
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    yh = 1  # Height per variant row
    
    for base_y, comb in enumerate(most_common):
        # Get all occurrences of this combination
        comb_occurrences = [
            motifs for motifs in sequences.values()
            if tuple(m["motif_id"] for m in motifs) == comb
        ]
        
        # Compute relative positions (relative to motif_1)
        relative_positions = defaultdict(list)
        for motifs in comb_occurrences:
            motif1 = next((m for m in motifs if m["motif_id"] == "motif_1"), None)
            if motif1 is None:
                continue
            for motif in motifs:
                if motif["motif_id"] != "motif_1":
                    rel_pos = int(motif["position"]) - int(motif1["position"])
                    relative_positions[motif["motif_id"]].append(rel_pos)
        
        # Plot each motif in the combination
        for midx, motif_id in enumerate(comb):
            w = len(motif_seqs[motif_id])
            motif_idx = motif_names.index(motif_id)
            
            if motif_id in relative_positions:
                rel_pos = np.array(relative_positions[motif_id])
                mean_pos = np.mean(rel_pos)
                std = np.std(rel_pos)
                
                rect = plt.Rectangle(
                    (mean_pos, base_y * yh), 
                    width=w, height=0.2,
                    color=cmap(motif_idx), alpha=0.5
                )
                ax1.add_patch(rect)
                ax1.errorbar(
                    mean_pos + w/2, base_y * yh + (midx+1)/10,
                    xerr=std + w, fmt='none',
                    ecolor=cmap(motif_idx), capsize=5
                )
                ax1.text(
                    mean_pos + w/2, base_y * yh + 0.1,
                    f"{motif_idx + 1}",
                    ha="center", va="center", fontsize=12
                )
            else:
                # Reference motif (motif_1) at position 0
                rect = plt.Rectangle(
                    (0, base_y * yh), 
                    width=w, height=0.2,
                    color=cmap(0), alpha=0.5
                )
                ax1.add_patch(rect)
                ax1.text(
                    w/2, base_y * yh + 0.1, "1",
                    ha="center", va="center", fontsize=12
                )
    
    # Configure left plot
    y_ticks = [f"uTP{idx+1}" for idx in range(len(most_common))]
    ax1.set_yticks(np.arange(0, len(most_common) * yh, yh))
    ax1.set_yticklabels(y_ticks)
    ax1.set_title("Relative positions of top motif combinations\n(relative to motif #1)")
    ax1.set_xlabel("Position (aa)")
    sns.despine(left=True, trim=True, ax=ax1)
    
    # Right plot: occurrence counts
    counts = [motif_combination_counts[comb] for comb in reversed(most_common)]
    sns.barplot(x=counts, y=list(range(len(counts))), orient="h", ax=ax2)
    
    # Add percentage labels
    total = sum(motif_combination_counts.values())
    for i, v in enumerate(counts):
        pct = v / total * 100
        ax2.text(v - 0.5, i, f"{pct:.1f}%", ha="right", fontsize=14, color="white")
    
    ax2.set_title("Occurrences among UCYN-A imported sequences")
    ax2.set_xlabel("Count")
    ax2.set_yticks([])
    sns.despine(ax=ax2)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved motif combination plot to {output_file}")


def create_motif_tree(
    sequences: dict[str, list[dict]],
    output_file: Path,
    min_count: int = 5
):
    """
    Create a directed graph showing motif transitions.
    """
    try:
        import graphviz
    except ImportError:
        print("graphviz not installed, skipping motif tree")
        return
    
    # Count transitions between motifs
    transitions = Counter(
        (m1["motif_id"], m2["motif_id"])
        for motifs in sequences.values()
        for m1, m2 in zip(motifs[:-1], motifs[1:])
    )
    
    # Filter by minimum count
    transitions = {
        (m1, m2): count
        for (m1, m2), count in transitions.items()
        if count >= min_count
    }
    
    # Create graph
    dot = graphviz.Digraph(format='svg')
    dot.attr(rankdir='LR')
    
    for (m1, m2), count in transitions.items():
        dot.edge(m1, m2, label=str(count))
    
    dot.render(output_file.with_suffix(''), cleanup=True)
    print(f"Saved motif tree to {output_file}")


def save_results(
    motif_combination_counts: Counter,
    most_common_combinations: list[tuple],
    relative_positions: list[tuple],
    motif_seqs: dict[str, str],
    output_dir: Path
):
    """Save analysis results to CSV files."""
    
    # All motif combinations
    combo_df = pd.DataFrame([
        {
            "motif_pattern": " → ".join(m.replace("motif_", "") for m in combo),
            "motif_ids": "+".join(combo),
            "count": count,
            "percentage": count / sum(motif_combination_counts.values()) * 100
        }
        for combo, count in motif_combination_counts.most_common()
    ])
    combo_df.to_csv(output_dir / "motif_combinations.csv", index=False)
    print(f"Saved {len(combo_df)} combinations to motif_combinations.csv")
    
    # Top variants
    top_df = pd.DataFrame([
        {
            "variant": f"uTP{i+1}",
            "motif_pattern": " → ".join(m.replace("motif_", "") for m in combo),
            "motif_ids": "+".join(combo),
            "count": motif_combination_counts[combo],
            "percentage": motif_combination_counts[combo] / sum(motif_combination_counts.values()) * 100
        }
        for i, combo in enumerate(most_common_combinations)
    ])
    top_df.to_csv(output_dir / "top_utp_variants.csv", index=False)
    print(f"Saved top {len(top_df)} variants to top_utp_variants.csv")
    
    # Motif positions
    pos_df = pd.DataFrame(
        relative_positions,
        columns=["sequence_id", "motif_id", "relative_position"]
    )
    pos_df.to_csv(output_dir / "motif_positions.csv", index=False)
    print(f"Saved {len(pos_df)} position records to motif_positions.csv")
    
    # Motif sequences
    motif_df = pd.DataFrame([
        {"motif_id": mid, "consensus_sequence": seq, "length": len(seq)}
        for mid, seq in motif_seqs.items()
    ])
    motif_df.to_csv(output_dir / "motif_sequences.csv", index=False)
    print(f"Saved {len(motif_df)} motif sequences to motif_sequences.csv")


def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("uTP Motif Combination Extraction")
    print("=" * 60)
    
    ensure_output_dir()
    
    # Check input files
    if not MSA_FILE.exists():
        raise FileNotFoundError(
            f"MSA file not found: {MSA_FILE}\n"
            f"Please copy ucyn-a_enriched_cobalt_cleaned.fa to {DATA_DIR}/"
        )
    if not MEME_FILE.exists():
        raise FileNotFoundError(
            f"MEME file not found: {MEME_FILE}\n"
            f"Please copy meme_gb.xml to {DATA_DIR}/"
        )
    
    # Step 1: Extract good C-terminal sequences
    print("\n[Step 1] Extracting C-terminal sequences from MSA...")
    c_term_seqs, full_seqs = extract_good_c_terminal_sequences(MSA_FILE)
    
    # Save extracted sequences
    SeqIO.write(c_term_seqs, OUTPUT_DIR / "good-c-term.fasta", "fasta")
    SeqIO.write(full_seqs, OUTPUT_DIR / "good-c-term-full.fasta", "fasta")
    print(f"Saved sequences to output/good-c-term.fasta and output/good-c-term-full.fasta")
    
    # Convert to dict for later use
    full_seqs_dict = {s.id: str(s.seq) for s in full_seqs}
    
    # Step 2: Parse MEME results
    print("\n[Step 2] Parsing MEME XML output...")
    meme_xml = parse_meme_xml(MEME_FILE)
    scanned_sites = extract_scanned_sites(meme_xml)
    motif_seqs = get_motif_sequences(meme_xml)
    seq_names = get_sequence_names(meme_xml)
    
    print(f"Found {len(motif_seqs)} motifs")
    print(f"Found {len(scanned_sites)} motif occurrences across sequences")
    
    # Step 3: Extract motif combinations
    print("\n[Step 3] Extracting motif combinations...")
    sequences = extract_motif_combinations(scanned_sites)
    motif_combination_counts = count_motif_combinations(sequences)
    
    print(f"Found {len(motif_combination_counts)} unique motif combinations")
    print(f"Top 5 combinations:")
    for combo, count in motif_combination_counts.most_common(5):
        pattern = " → ".join(m.replace("motif_", "") for m in combo)
        pct = count / len(sequences) * 100
        print(f"  {pattern}: {count} ({pct:.1f}%)")
    
    # Step 4: Compute relative positions
    print("\n[Step 4] Computing relative motif positions...")
    relative_positions = compute_relative_positions(scanned_sites)
    
    # Step 5: Analyze inter-motif regions
    print("\n[Step 5] Analyzing inter-motif regions...")
    aa_distributions, mean_lengths = analyze_inter_motif_regions(
        sequences, full_seqs_dict, motif_seqs, seq_names
    )
    
    # Get top combinations
    most_common = sorted(
        motif_combination_counts.keys(),
        key=lambda c: motif_combination_counts[c],
        reverse=True
    )[:TOP_N_VARIANTS]
    
    # Step 6: Generate example synthetic uTPs
    print("\n[Step 6] Example synthetic uTP sequences:")
    for i, combo in enumerate(most_common):
        synthetic = construct_utp(combo, motif_seqs, aa_distributions, mean_lengths)
        print(f"  uTP{i+1}: {synthetic[:60]}...")
    
    # Step 7: Save results
    print("\n[Step 7] Saving results...")
    save_results(
        motif_combination_counts,
        most_common,
        relative_positions,
        motif_seqs,
        OUTPUT_DIR
    )
    
    # Step 8: Generate plots
    print("\n[Step 8] Generating plots...")
    plot_motif_combinations(
        sequences,
        motif_combination_counts,
        motif_seqs,
        OUTPUT_DIR / "motif_combinations.svg",
        top_n=TOP_N_VARIANTS
    )
    
    create_motif_tree(
        sequences,
        OUTPUT_DIR / "motif_tree.svg",
        min_count=5
    )
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
