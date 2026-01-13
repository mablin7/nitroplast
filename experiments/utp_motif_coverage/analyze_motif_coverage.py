#!/usr/bin/env python3
"""
uTP Motif Coverage Analysis

This script runs MAST to scan for known uTP motifs in the full set of
HMM-predicted import candidates (933 proteins), then analyzes the
distribution of motif patterns.

Background:
- MEME motif discovery was performed on 368 experimentally enriched proteins
- HMM search identified 933 potential uTP proteins in the full proteome
- This script uses MAST to find motifs in all 933 proteins

Outputs:
- mast_results/: MAST output directory
- motif_patterns.csv: Motif patterns for all proteins
- pattern_distribution.svg: Visualization of pattern distribution
- coverage_comparison.svg: Comparison with experimental set
"""

import subprocess
import warnings
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO

warnings.filterwarnings("ignore")

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data"
MOTIF_DIR = SCRIPT_DIR.parent / "utp_motif_analysis" / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Input files
IMPORT_CANDIDATES = DATA_DIR / "Import_candidates.fasta"
MEME_FILE = MOTIF_DIR / "meme_gb.xml"
GBLOCKS_CTERM = MOTIF_DIR / "good-c-term-gb.fasta"

# MAST binary
MAST_BIN = "/opt/local/bin/mast"


def load_fasta_sequences(fasta_path):
    """Load sequences from a FASTA file."""
    seqs = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        seqs[record.id] = str(record.seq)
    return seqs


def load_meme_motif_names():
    """Load motif names from MEME XML."""
    tree = ET.parse(MEME_FILE)
    return {
        tag.attrib["id"]: tag.attrib["name"]
        for tag in tree.findall(".//motif")
    }


def run_mast(sequences_fasta, meme_file, output_dir):
    """
    Run MAST to scan sequences for known motifs.
    
    Args:
        sequences_fasta: Path to FASTA file with sequences to scan
        meme_file: Path to MEME XML file with motif definitions
        output_dir: Directory to write MAST output
    
    Returns:
        Path to MAST XML output file
    """
    mast_output = output_dir / "mast_results"
    
    # Remove existing output if present (MAST won't overwrite)
    if mast_output.exists():
        import shutil
        shutil.rmtree(mast_output)
    
    cmd = [
        MAST_BIN,
        str(meme_file),
        str(sequences_fasta),
        "-o", str(mast_output),
    ]
    
    print(f"Running MAST: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"MAST stderr: {result.stderr}")
        raise RuntimeError(f"MAST failed with return code {result.returncode}")
    
    return mast_output / "mast.xml"


def parse_mast_xml(mast_xml_path):
    """
    Parse MAST XML output to extract motif hits.
    
    Returns:
        Dictionary mapping sequence name to list of motif hit dictionaries
    """
    tree = ET.parse(mast_xml_path)
    root = tree.getroot()
    
    # Build motif index to ID mapping from the motifs section
    # MAST uses 0-based indices in hits, but motif IDs are like "MEME-1"
    motif_map = {}
    for idx, motif in enumerate(root.findall(".//motifs/motif")):
        motif_id = motif.attrib.get("id", "")
        alt = motif.attrib.get("alt", "")
        # Map index to motif_N format (MEME-1 -> motif_1)
        if alt.startswith("MEME-"):
            motif_num = alt.replace("MEME-", "")
            motif_map[str(idx)] = f"motif_{motif_num}"
        else:
            motif_map[str(idx)] = f"motif_{idx+1}"
    
    print(f"  Motif mapping: {motif_map}")
    
    # Get motif hits for each sequence
    hits = defaultdict(list)
    
    for seq in root.findall(".//sequence"):
        seq_name = seq.attrib.get("name")
        if not seq_name:
            continue
        
        # Find all segment hits within this sequence
        for seg in seq.findall(".//seg"):
            for hit in seg.findall("hit"):
                idx = hit.attrib.get("idx")
                pos = int(hit.attrib.get("pos", 0))
                pvalue = float(hit.attrib.get("pvalue", 1.0))
                
                # Convert index to motif ID
                motif_id = motif_map.get(idx, f"motif_{int(idx)+1}")
                
                hits[seq_name].append({
                    "motif_id": motif_id,
                    "position": pos,
                    "pvalue": pvalue,
                })
    
    # Sort hits by position for each sequence
    for seq_name in hits:
        hits[seq_name] = sorted(hits[seq_name], key=lambda h: h["position"])
    
    return dict(hits)


def parse_mast_hit_list(mast_output_dir):
    """
    Parse MAST hit_list.txt as fallback if XML parsing fails.
    
    Returns:
        Dictionary mapping sequence name to list of motif hits
    """
    hit_list_file = mast_output_dir / "mast.txt"
    
    if not hit_list_file.exists():
        return {}
    
    hits = defaultdict(list)
    current_seq = None
    
    with open(hit_list_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = line.split()
            if len(parts) >= 4:
                seq_name = parts[0]
                # Parse motif hits from the line
                # Format varies, need to inspect actual output
                
    return dict(hits)


def extract_motif_patterns(hits, seq_lengths=None):
    """
    Extract motif patterns from MAST hits.
    
    Args:
        hits: Dictionary from parse_mast_xml
        seq_lengths: Optional dictionary of sequence lengths
    
    Returns:
        Dictionary mapping sequence name to pattern string (e.g., "2+1+3+4")
    """
    patterns = {}
    
    for seq_name, motif_hits in hits.items():
        if not motif_hits:
            patterns[seq_name] = ""
            continue
        
        # Extract motif IDs in position order
        motif_ids = [h["motif_id"] for h in motif_hits]
        
        # Convert to pattern string (remove "motif_" prefix)
        pattern_parts = [m.replace("motif_", "") for m in motif_ids]
        patterns[seq_name] = "+".join(pattern_parts)
    
    return patterns


def classify_terminal_motif(pattern):
    """Classify a pattern by its terminal motif."""
    if not pattern:
        return "no_motifs"
    
    parts = pattern.split("+")
    terminal = parts[-1] if parts else None
    
    if terminal in ["4", "5", "7", "9"]:
        return f"terminal_{terminal}"
    return f"other_{terminal}"


def analyze_patterns(patterns, experimental_names=None):
    """
    Analyze the distribution of motif patterns.
    
    Args:
        patterns: Dictionary from extract_motif_patterns
        experimental_names: Set of names from experimental (MEME) analysis
    
    Returns:
        DataFrame with pattern analysis
    """
    results = []
    
    for seq_name, pattern in patterns.items():
        terminal = classify_terminal_motif(pattern)
        is_valid = terminal in ["terminal_4", "terminal_5", "terminal_7", "terminal_9"]
        
        # Check pattern structure
        parts = pattern.split("+") if pattern else []
        starts_with_21 = parts[:2] == ["2", "1"] if len(parts) >= 2 else False
        
        results.append({
            "name": seq_name,
            "pattern": pattern,
            "n_motifs": len(parts) if pattern else 0,
            "terminal_class": terminal,
            "is_valid_terminal": is_valid,
            "starts_with_2_1": starts_with_21,
            "in_experimental": seq_name in (experimental_names or set()),
        })
    
    return pd.DataFrame(results)


def plot_pattern_distribution(df, output_dir):
    """Generate visualizations of pattern distribution."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel 1: Terminal motif distribution
    ax = axes[0, 0]
    terminal_counts = df["terminal_class"].value_counts()
    
    colors = ["#2ecc71" if t.startswith("terminal_") and t.split("_")[1] in ["4", "5", "7", "9"]
              else "#e74c3c" if t != "no_motifs" else "#95a5a6"
              for t in terminal_counts.index]
    
    bars = ax.bar(range(len(terminal_counts)), terminal_counts.values, color=colors)
    ax.set_xticks(range(len(terminal_counts)))
    ax.set_xticklabels(terminal_counts.index, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(f"Terminal motif distribution (n={len(df)})\ngreen=valid, red=other, gray=no motifs")
    
    # Panel 2: Number of motifs distribution
    ax = axes[0, 1]
    motif_counts = df["n_motifs"].value_counts().sort_index()
    ax.bar(motif_counts.index, motif_counts.values, color="#3498db")
    ax.set_xlabel("Number of motifs")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of motif count per protein")
    
    # Panel 3: Pattern frequency (top 15)
    ax = axes[1, 0]
    pattern_counts = df["pattern"].value_counts().head(15)
    colors_pat = ["#2ecc71" if classify_terminal_motif(p) in ["terminal_4", "terminal_5", "terminal_7", "terminal_9"]
                  else "#e74c3c" if p else "#95a5a6"
                  for p in pattern_counts.index]
    
    y_pos = range(len(pattern_counts))
    ax.barh(y_pos, pattern_counts.values, color=colors_pat)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([p if p else "(no motifs)" for p in pattern_counts.index])
    ax.set_xlabel("Count")
    ax.set_title("Top 15 motif patterns")
    ax.invert_yaxis()
    
    # Panel 4: Valid vs invalid breakdown
    ax = axes[1, 1]
    breakdown = {
        "Valid terminal\n(4/5/7/9)": df["is_valid_terminal"].sum(),
        "Other terminal": ((~df["is_valid_terminal"]) & (df["n_motifs"] > 0)).sum(),
        "No motifs": (df["n_motifs"] == 0).sum(),
    }
    
    colors_bd = ["#2ecc71", "#e74c3c", "#95a5a6"]
    wedges, texts, autotexts = ax.pie(
        breakdown.values(), 
        labels=breakdown.keys(),
        autopct=lambda pct: f"{pct:.1f}%\n({int(pct/100*len(df))})",
        colors=colors_bd,
        startangle=90
    )
    ax.set_title(f"Pattern validity (n={len(df)})")
    
    plt.tight_layout()
    plt.savefig(output_dir / "pattern_distribution.svg", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "pattern_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved pattern distribution to {output_dir / 'pattern_distribution.png'}")


def plot_comparison(df, output_dir):
    """Compare experimental vs predicted sets."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Experimental vs HMM-only
    ax = axes[0]
    
    exp_df = df[df["in_experimental"]]
    hmm_only_df = df[~df["in_experimental"]]
    
    categories = ["Valid terminal", "Other terminal", "No motifs"]
    exp_counts = [
        exp_df["is_valid_terminal"].sum(),
        ((~exp_df["is_valid_terminal"]) & (exp_df["n_motifs"] > 0)).sum(),
        (exp_df["n_motifs"] == 0).sum(),
    ]
    hmm_counts = [
        hmm_only_df["is_valid_terminal"].sum(),
        ((~hmm_only_df["is_valid_terminal"]) & (hmm_only_df["n_motifs"] > 0)).sum(),
        (hmm_only_df["n_motifs"] == 0).sum(),
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, exp_counts, width, label=f"Experimental (n={len(exp_df)})", color="#3498db")
    bars2 = ax.bar(x + width/2, hmm_counts, width, label=f"HMM-only (n={len(hmm_only_df)})", color="#e67e22")
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Count")
    ax.set_title("Pattern validity: Experimental vs HMM-only")
    ax.legend()
    
    # Add percentage labels
    for bars, total in [(bars1, len(exp_df)), (bars2, len(hmm_only_df))]:
        for bar in bars:
            height = bar.get_height()
            pct = height / total * 100 if total > 0 else 0
            ax.annotate(f"{pct:.0f}%",
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", va="bottom", fontsize=9)
    
    # Panel 2: Terminal motif comparison
    ax = axes[1]
    
    terminal_classes = ["terminal_4", "terminal_5", "terminal_7", "terminal_9"]
    exp_term = [len(exp_df[exp_df["terminal_class"] == t]) for t in terminal_classes]
    hmm_term = [len(hmm_only_df[hmm_only_df["terminal_class"] == t]) for t in terminal_classes]
    
    x = np.arange(len(terminal_classes))
    
    bars1 = ax.bar(x - width/2, exp_term, width, label=f"Experimental", color="#3498db")
    bars2 = ax.bar(x + width/2, hmm_term, width, label=f"HMM-only", color="#e67e22")
    
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("terminal_", "T") for t in terminal_classes])
    ax.set_ylabel("Count")
    ax.set_title("Terminal motif distribution by source")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "source_comparison.svg", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "source_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved source comparison to {output_dir / 'source_comparison.png'}")


def main():
    print("=" * 70)
    print("uTP Motif Coverage Analysis")
    print("Scanning all HMM-predicted proteins for known uTP motifs")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Step 1: Load data
    # =========================================================================
    print("\n[1/5] Loading data...")
    
    import_seqs = load_fasta_sequences(IMPORT_CANDIDATES)
    print(f"  Import_candidates.fasta: {len(import_seqs)} sequences")
    
    # Load experimental set names (from Gblocks filtered MEME analysis)
    experimental_names = set(load_fasta_sequences(GBLOCKS_CTERM).keys())
    print(f"  Experimental (MEME) set: {len(experimental_names)} sequences")
    
    motif_names = load_meme_motif_names()
    print(f"  Motifs from MEME: {len(motif_names)}")
    
    # =========================================================================
    # Step 2: Run MAST
    # =========================================================================
    print("\n[2/5] Running MAST motif search...")
    
    try:
        mast_xml = run_mast(IMPORT_CANDIDATES, MEME_FILE, OUTPUT_DIR)
        print(f"  MAST output: {mast_xml}")
    except Exception as e:
        print(f"  Error running MAST: {e}")
        return
    
    # =========================================================================
    # Step 3: Parse MAST results
    # =========================================================================
    print("\n[3/5] Parsing MAST results...")
    
    hits = parse_mast_xml(mast_xml)
    print(f"  Sequences with motif hits: {len(hits)}")
    
    # Extract patterns
    patterns = extract_motif_patterns(hits)
    print(f"  Unique patterns: {len(set(patterns.values()))}")
    
    # =========================================================================
    # Step 4: Analyze patterns
    # =========================================================================
    print("\n[4/5] Analyzing patterns...")
    
    df = analyze_patterns(patterns, experimental_names)
    
    # Summary statistics
    print(f"\n  Pattern summary:")
    print(f"    Total proteins: {len(df)}")
    print(f"    With any motifs: {(df['n_motifs'] > 0).sum()}")
    print(f"    Valid terminal (4/5/7/9): {df['is_valid_terminal'].sum()}")
    print(f"    Starts with 2→1: {df['starts_with_2_1'].sum()}")
    
    # Terminal class distribution
    print(f"\n  Terminal class distribution:")
    for cls, count in df["terminal_class"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {cls}: {count} ({pct:.1f}%)")
    
    # Compare experimental vs HMM-only
    exp_df = df[df["in_experimental"]]
    hmm_only = df[~df["in_experimental"]]
    
    print(f"\n  Experimental set ({len(exp_df)} proteins):")
    print(f"    Valid terminal: {exp_df['is_valid_terminal'].sum()} ({exp_df['is_valid_terminal'].mean()*100:.1f}%)")
    
    print(f"\n  HMM-only set ({len(hmm_only)} proteins):")
    print(f"    Valid terminal: {hmm_only['is_valid_terminal'].sum()} ({hmm_only['is_valid_terminal'].mean()*100:.1f}%)")
    
    # Save results
    df.to_csv(OUTPUT_DIR / "motif_patterns.csv", index=False)
    print(f"\n  Saved patterns to {OUTPUT_DIR / 'motif_patterns.csv'}")
    
    # =========================================================================
    # Step 5: Generate visualizations
    # =========================================================================
    print("\n[5/5] Generating visualizations...")
    
    plot_pattern_distribution(df, OUTPUT_DIR)
    plot_comparison(df, OUTPUT_DIR)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    valid_total = df["is_valid_terminal"].sum()
    valid_exp = exp_df["is_valid_terminal"].sum()
    valid_hmm = hmm_only["is_valid_terminal"].sum()
    
    print(f"""
Data sources:
  Experimental (proteomics enriched): {len(experimental_names)} proteins
  HMM-predicted (Import_candidates): {len(import_seqs)} proteins
  Overlap: {len(exp_df)} proteins in both

MAST motif scanning results:
  Proteins with motif hits: {(df['n_motifs'] > 0).sum()} / {len(df)}
  Valid terminal patterns: {valid_total} ({valid_total/len(df)*100:.1f}%)

Breakdown by source:
  Experimental: {valid_exp} / {len(exp_df)} valid ({valid_exp/len(exp_df)*100:.1f}%)
  HMM-only: {valid_hmm} / {len(hmm_only)} valid ({valid_hmm/len(hmm_only)*100:.1f}%)

This analysis extends motif detection from {len(experimental_names)} to {len(import_seqs)} proteins.
""")
    
    print(f"📁 Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
