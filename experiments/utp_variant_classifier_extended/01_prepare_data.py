#!/usr/bin/env python3
"""
01_prepare_data.py - Data Preparation for Extended uTP Variant Classifier

This script prepares the extended dataset of 607+ proteins with valid terminal motifs
from the MAST motif coverage analysis.

Key improvements over original:
- Uses full HMM-predicted protein set (933 → 607 with valid terminals)
- Applies rigorous filtering criteria
- Handles class imbalance documentation
- Extracts mature domains using HMM boundaries

Input:
- experiments/utp_motif_coverage/output/motif_patterns.csv
- data/Import_candidates.fasta
- data/Bbigelowii_transcriptome_annotations.csv

Output:
- output/data/processed_proteins.csv
- output/data/mature_domains.fasta
- output/data/utp_regions.fasta
- output/data/class_distribution.csv
- output/figures/class_distribution.svg

Usage:
    uv run python experiments/utp_variant_classifier_extended/01_prepare_data.py
"""

import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# =============================================================================
# Configuration
# =============================================================================

# Filtering parameters
MIN_CLASS_SIZE = 10  # Minimum samples per class for statistical validity
MIN_MATURE_LENGTH = 30
MAX_MATURE_LENGTH = 3000
MIN_UTP_LENGTH = 50

# HMM parameters for boundary detection
HMM_EVALUE_THRESHOLD = 0.01
MIN_HMM_SCORE = 20.0
MIN_HMM_COVERAGE_START = 80  # Allow flexibility in HMM alignment start

# Valid terminal motifs
VALID_TERMINALS = {"terminal_4", "terminal_5", "terminal_7", "terminal_9"}

# Random seed
RANDOM_SEED = 42

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Input files
MOTIF_PATTERNS_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "utp_motif_coverage"
    / "output"
    / "motif_patterns.csv"
)
SEQUENCES_FILE = DATA_DIR / "Import_candidates.fasta"
ANNOTATIONS_FILE = DATA_DIR / "Bbigelowii_transcriptome_annotations.csv"
HMM_PROFILE = PROJECT_ROOT / "experiments" / "utp_homolog_search" / "utp.hmm"

# Output directories
DATA_OUTPUT_DIR = OUTPUT_DIR / "data"
FIGURES_DIR = OUTPUT_DIR / "figures"


# =============================================================================
# Data Classes
# =============================================================================


class HMMHit(NamedTuple):
    """Container for hmmsearch domain hit."""

    target_name: str
    target_length: int
    full_evalue: float
    full_score: float
    domain_evalue: float
    domain_score: float
    hmm_from: int
    hmm_to: int
    ali_from: int
    ali_to: int
    env_from: int
    env_to: int


class ProcessedProtein(NamedTuple):
    """Container for processed protein data."""

    name: str
    full_sequence: str
    mature_sequence: str
    utp_sequence: str
    utp_start: int
    utp_end: int
    terminal_class: str
    motif_pattern: str
    hmm_score: float
    in_experimental: bool


# =============================================================================
# HMM-based uTP Detection
# =============================================================================


def run_hmmsearch(sequences_file: Path, hmm_file: Path, output_file: Path) -> Path:
    """Run hmmsearch to detect uTP regions in sequences."""
    domtblout = output_file.with_suffix(".domtblout")

    cmd = [
        "hmmsearch",
        "--domtblout",
        str(domtblout),
        "-E",
        str(HMM_EVALUE_THRESHOLD),
        "--domE",
        str(HMM_EVALUE_THRESHOLD),
        "--noali",
        str(hmm_file),
        str(sequences_file),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"hmmsearch failed: {result.stderr}")

    return domtblout


def parse_hmmsearch_domtblout(domtblout_path: Path) -> list[HMMHit]:
    """Parse hmmsearch domain table output."""
    hits = []

    with open(domtblout_path) as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 23:
                continue

            try:
                hit = HMMHit(
                    target_name=parts[0],
                    target_length=int(parts[2]),
                    full_evalue=float(parts[6]),
                    full_score=float(parts[7]),
                    domain_evalue=float(parts[12]),
                    domain_score=float(parts[13]),
                    hmm_from=int(parts[15]),
                    hmm_to=int(parts[16]),
                    ali_from=int(parts[17]),
                    ali_to=int(parts[18]),
                    env_from=int(parts[19]),
                    env_to=int(parts[20]),
                )
                hits.append(hit)
            except (ValueError, IndexError):
                continue

    return hits


def detect_utp_regions(
    sequences: dict[str, str],
    hmm_file: Path,
    output_dir: Path,
) -> dict[str, tuple[int, int, float]]:
    """
    Detect uTP regions in sequences using HMM search.

    Returns:
        Dict mapping protein name to (utp_start_1idx, utp_end_1idx, hmm_score)
    """
    # Write sequences to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        for name, seq in sequences.items():
            f.write(f">{name}\n{seq}\n")
        temp_fasta = Path(f.name)

    try:
        domtblout = run_hmmsearch(temp_fasta, hmm_file, output_dir / "utp_detection")
        hits = parse_hmmsearch_domtblout(domtblout)
    finally:
        temp_fasta.unlink()

    # Group hits by target sequence
    hits_by_target: dict[str, list[HMMHit]] = {}
    for hit in hits:
        if hit.target_name not in hits_by_target:
            hits_by_target[hit.target_name] = []
        hits_by_target[hit.target_name].append(hit)

    # Extract best uTP boundary for each sequence
    utp_regions = {}

    for target_name, target_hits in hits_by_target.items():
        best_hit = None
        best_score = 0

        for hit in target_hits:
            if hit.domain_evalue > HMM_EVALUE_THRESHOLD:
                continue
            if hit.domain_score < MIN_HMM_SCORE:
                continue
            if hit.hmm_from > MIN_HMM_COVERAGE_START:
                continue

            if hit.domain_score > best_score:
                best_hit = hit
                best_score = hit.domain_score

        if best_hit is not None:
            utp_start = best_hit.ali_from  # 1-indexed
            utp_end = best_hit.target_length
            utp_regions[target_name] = (utp_start, utp_end, best_hit.domain_score)

    return utp_regions


# =============================================================================
# Data Loading and Processing
# =============================================================================


def load_motif_patterns(motif_file: Path) -> pd.DataFrame:
    """Load motif patterns from MAST analysis."""
    df = pd.read_csv(motif_file)
    print(f"  Loaded {len(df)} proteins from motif patterns file")
    return df


def load_sequences(fasta_file: Path) -> dict[str, str]:
    """Load sequences from FASTA file."""
    seqs = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        seqs[record.id] = str(record.seq)
    print(f"  Loaded {len(seqs)} sequences from FASTA")
    return seqs


def load_annotations(annotations_file: Path) -> pd.DataFrame:
    """Load and parse EggNOG annotations."""
    df = pd.read_csv(annotations_file, comment="#", header=0, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if df.columns[0].startswith("#"):
        df = df.rename(columns={df.columns[0]: "query_name"})
    print(f"  Loaded {len(df)} annotation records")
    return df


def filter_valid_proteins(
    motif_df: pd.DataFrame,
    min_class_size: int = MIN_CLASS_SIZE,
) -> pd.DataFrame:
    """
    Filter to proteins with valid terminal motifs and sufficient class sizes.
    """
    # Filter to valid terminal motifs
    valid_df = motif_df[motif_df["is_valid_terminal"]].copy()
    print(f"  Proteins with valid terminal motifs: {len(valid_df)}")

    # Count classes
    class_counts = valid_df["terminal_class"].value_counts()
    print(f"\n  Class distribution before filtering:")
    for cls, count in class_counts.items():
        print(f"    {cls}: {count}")

    # Filter to classes with sufficient samples
    valid_classes = class_counts[class_counts >= min_class_size].index.tolist()
    filtered_df = valid_df[valid_df["terminal_class"].isin(valid_classes)].copy()

    excluded_classes = set(class_counts.index) - set(valid_classes)
    if excluded_classes:
        print(f"\n  Excluded classes (n < {min_class_size}): {excluded_classes}")

    print(
        f"\n  Final dataset: {len(filtered_df)} proteins, {len(valid_classes)} classes"
    )

    return filtered_df


def process_proteins(
    filtered_df: pd.DataFrame,
    sequences: dict[str, str],
    utp_regions: dict[str, tuple[int, int, float]],
) -> list[ProcessedProtein]:
    """
    Process all proteins: extract mature domains and uTP regions.
    """
    processed = []
    skipped_no_seq = 0
    skipped_short_mature = 0
    skipped_short_utp = 0

    for _, row in filtered_df.iterrows():
        name = row["name"]

        if name not in sequences:
            skipped_no_seq += 1
            continue

        seq = sequences[name]

        # Get HMM-based uTP boundary if available
        if name in utp_regions:
            utp_start, utp_end, hmm_score = utp_regions[name]
        else:
            # Fallback: estimate uTP start based on sequence length
            seq_len = len(seq)
            utp_start = max(1, seq_len - 150)
            utp_end = seq_len
            hmm_score = 0.0

        # Convert to 0-indexed
        utp_start_0idx = utp_start - 1

        # Extract sequences with small buffer between mature and uTP
        linker_buffer = 10
        mature_end = max(0, utp_start_0idx - linker_buffer)

        mature_seq = seq[:mature_end]
        utp_seq = seq[utp_start_0idx:]

        # Clean sequences (remove non-standard AAs)
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        mature_seq = "".join(aa for aa in mature_seq if aa in valid_aa)
        utp_seq = "".join(aa for aa in utp_seq if aa in valid_aa)

        # Validate lengths
        if len(mature_seq) < MIN_MATURE_LENGTH or len(mature_seq) > MAX_MATURE_LENGTH:
            skipped_short_mature += 1
            continue
        if len(utp_seq) < MIN_UTP_LENGTH:
            skipped_short_utp += 1
            continue

        processed.append(
            ProcessedProtein(
                name=name,
                full_sequence=seq,
                mature_sequence=mature_seq,
                utp_sequence=utp_seq,
                utp_start=utp_start,
                utp_end=utp_end,
                terminal_class=row["terminal_class"],
                motif_pattern=row["pattern"],
                hmm_score=hmm_score,
                in_experimental=row["in_experimental"],
            )
        )

    print(f"\n  Processing summary:")
    print(f"    Successfully processed: {len(processed)}")
    print(f"    Skipped (no sequence): {skipped_no_seq}")
    print(f"    Skipped (short mature): {skipped_short_mature}")
    print(f"    Skipped (short uTP): {skipped_short_utp}")

    return processed


# =============================================================================
# Output Generation
# =============================================================================


def save_processed_data(
    proteins: list[ProcessedProtein],
    annotations: pd.DataFrame,
    output_dir: Path,
):
    """Save processed protein data to files."""
    # Create protein dataframe
    protein_data = []
    for p in proteins:
        protein_data.append(
            {
                "name": p.name,
                "terminal_class": p.terminal_class,
                "motif_pattern": p.motif_pattern,
                "mature_length": len(p.mature_sequence),
                "utp_length": len(p.utp_sequence),
                "full_length": len(p.full_sequence),
                "utp_start": p.utp_start,
                "hmm_score": p.hmm_score,
                "in_experimental": p.in_experimental,
            }
        )

    protein_df = pd.DataFrame(protein_data)

    # Merge with annotations
    if "query_name" in annotations.columns:
        merged = protein_df.merge(
            annotations,
            left_on="name",
            right_on="query_name",
            how="left",
        )
    else:
        merged = protein_df.merge(
            annotations,
            left_on="name",
            right_on=annotations.columns[0],
            how="left",
        )

    # Save processed proteins CSV
    merged.to_csv(output_dir / "processed_proteins.csv", index=False)
    print(f"  Saved protein metadata to {output_dir / 'processed_proteins.csv'}")

    # Save mature domains FASTA
    mature_records = [
        SeqRecord(
            Seq(p.mature_sequence),
            id=p.name,
            description=f"terminal={p.terminal_class} len={len(p.mature_sequence)}",
        )
        for p in proteins
    ]
    SeqIO.write(mature_records, output_dir / "mature_domains.fasta", "fasta")
    print(
        f"  Saved {len(mature_records)} mature domains to {output_dir / 'mature_domains.fasta'}"
    )

    # Save uTP regions FASTA
    utp_records = [
        SeqRecord(
            Seq(p.utp_sequence),
            id=p.name,
            description=f"terminal={p.terminal_class} len={len(p.utp_sequence)}",
        )
        for p in proteins
    ]
    SeqIO.write(utp_records, output_dir / "utp_regions.fasta", "fasta")
    print(
        f"  Saved {len(utp_records)} uTP regions to {output_dir / 'utp_regions.fasta'}"
    )

    # Save class distribution
    class_dist = protein_df["terminal_class"].value_counts().reset_index()
    class_dist.columns = ["terminal_class", "count"]
    class_dist["proportion"] = class_dist["count"] / class_dist["count"].sum()
    class_dist.to_csv(output_dir / "class_distribution.csv", index=False)
    print(f"  Saved class distribution to {output_dir / 'class_distribution.csv'}")

    return merged


# =============================================================================
# Visualization
# =============================================================================


def plot_class_distribution(
    proteins: list[ProcessedProtein],
    output_file: Path,
):
    """Plot the distribution of terminal classes."""
    class_counts = Counter(p.terminal_class for p in proteins)

    # Sort by count
    sorted_classes = sorted(class_counts.items(), key=lambda x: -x[1])
    classes = [c[0] for c in sorted_classes]
    counts = [c[1] for c in sorted_classes]

    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Bar chart
    ax = axes[0]
    colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))
    bars = ax.bar(range(len(classes)), counts, color=colors)

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(count),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.replace("terminal_", "T") for c in classes], fontsize=12)
    ax.set_xlabel("Terminal Motif Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Class Distribution (n={sum(counts)} proteins)", fontsize=14)

    # Add imbalance ratio annotation
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count
    ax.text(
        0.95,
        0.95,
        f"Imbalance ratio: {imbalance_ratio:.1f}:1",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Panel 2: Pie chart
    ax = axes[1]
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=[c.replace("terminal_", "T") for c in classes],
        autopct=lambda pct: f"{pct:.1f}%\n({int(pct/100*sum(counts))})",
        colors=colors,
        startangle=90,
        explode=[0.02] * len(classes),
    )
    ax.set_title("Class Proportions", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved class distribution plot to {output_file}")


def plot_length_distributions(
    proteins: list[ProcessedProtein],
    output_file: Path,
):
    """Plot length distributions by terminal class."""
    df = pd.DataFrame(
        [
            {
                "terminal_class": p.terminal_class.replace("terminal_", "T"),
                "mature_length": len(p.mature_sequence),
                "utp_length": len(p.utp_sequence),
            }
            for p in proteins
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mature length
    sns.boxplot(
        data=df, x="terminal_class", y="mature_length", ax=axes[0], palette="Set2"
    )
    axes[0].set_xlabel("Terminal Class")
    axes[0].set_ylabel("Mature Domain Length (aa)")
    axes[0].set_title("Mature Domain Length by Terminal Class")

    # uTP length
    sns.boxplot(data=df, x="terminal_class", y="utp_length", ax=axes[1], palette="Set2")
    axes[1].set_xlabel("Terminal Class")
    axes[1].set_ylabel("uTP Region Length (aa)")
    axes[1].set_title("uTP Region Length by Terminal Class")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved length distributions plot to {output_file}")


def plot_experimental_comparison(
    proteins: list[ProcessedProtein],
    output_file: Path,
):
    """Compare experimental vs HMM-only proteins."""
    df = pd.DataFrame(
        [
            {
                "terminal_class": p.terminal_class,
                "source": "Experimental" if p.in_experimental else "HMM-only",
            }
            for p in proteins
        ]
    )

    # Cross-tabulation
    crosstab = pd.crosstab(df["terminal_class"], df["source"])

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(crosstab.index))
    width = 0.35

    exp_counts = (
        crosstab["Experimental"].values
        if "Experimental" in crosstab.columns
        else np.zeros(len(x))
    )
    hmm_counts = (
        crosstab["HMM-only"].values
        if "HMM-only" in crosstab.columns
        else np.zeros(len(x))
    )

    ax.bar(x - width / 2, exp_counts, width, label="Experimental", color="#3498db")
    ax.bar(x + width / 2, hmm_counts, width, label="HMM-only", color="#e67e22")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("terminal_", "T") for c in crosstab.index])
    ax.set_xlabel("Terminal Class")
    ax.set_ylabel("Count")
    ax.set_title("Protein Source by Terminal Class")
    ax.legend()

    # Add count labels
    for i, (e, h) in enumerate(zip(exp_counts, hmm_counts)):
        ax.text(i - width / 2, e + 2, str(int(e)), ha="center", va="bottom", fontsize=9)
        ax.text(i + width / 2, h + 2, str(int(h)), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved experimental comparison plot to {output_file}")


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("01_prepare_data.py - Extended uTP Variant Classifier Data Preparation")
    print("=" * 70)

    # Create output directories
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Check input files
    if not MOTIF_PATTERNS_FILE.exists():
        raise FileNotFoundError(
            f"Motif patterns file not found: {MOTIF_PATTERNS_FILE}\n"
            f"Please run experiments/utp_motif_coverage/analyze_motif_coverage.py first."
        )
    if not SEQUENCES_FILE.exists():
        raise FileNotFoundError(f"Sequences file not found: {SEQUENCES_FILE}")
    if not HMM_PROFILE.exists():
        raise FileNotFoundError(f"HMM profile not found: {HMM_PROFILE}")

    # Check hmmsearch
    try:
        subprocess.run(["hmmsearch", "-h"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("hmmsearch not found. Please install HMMER.")

    # =========================================================================
    # Step 1: Load data
    # =========================================================================
    print("\n[1/5] Loading data...")
    motif_df = load_motif_patterns(MOTIF_PATTERNS_FILE)
    sequences = load_sequences(SEQUENCES_FILE)

    if ANNOTATIONS_FILE.exists():
        annotations = load_annotations(ANNOTATIONS_FILE)
    else:
        print(f"  Warning: Annotations file not found: {ANNOTATIONS_FILE}")
        annotations = pd.DataFrame()

    # =========================================================================
    # Step 2: Filter to valid proteins
    # =========================================================================
    print("\n[2/5] Filtering to valid proteins...")
    filtered_df = filter_valid_proteins(motif_df, MIN_CLASS_SIZE)

    # =========================================================================
    # Step 3: Detect uTP regions using HMM
    # =========================================================================
    print("\n[3/5] Detecting uTP regions using HMM profile...")

    # Get sequences for filtered proteins
    filtered_seqs = {
        name: sequences[name] for name in filtered_df["name"] if name in sequences
    }

    utp_regions = detect_utp_regions(filtered_seqs, HMM_PROFILE, DATA_OUTPUT_DIR)
    print(f"  Detected HMM-based uTP regions in {len(utp_regions)} sequences")

    if utp_regions:
        scores = [s for _, _, s in utp_regions.values()]
        print(
            f"  HMM scores: min={min(scores):.1f}, max={max(scores):.1f}, "
            f"median={np.median(scores):.1f}"
        )

    # =========================================================================
    # Step 4: Process proteins
    # =========================================================================
    print("\n[4/5] Processing proteins...")
    proteins = process_proteins(filtered_df, sequences, utp_regions)

    if not proteins:
        print("  ERROR: No proteins remaining after processing!")
        return

    # Final class distribution
    class_counts = Counter(p.terminal_class for p in proteins)
    print(f"\n  Final class distribution:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"    {cls}: {count} ({count/len(proteins)*100:.1f}%)")

    # =========================================================================
    # Step 5: Save outputs and generate figures
    # =========================================================================
    print("\n[5/5] Saving outputs...")
    save_processed_data(proteins, annotations, DATA_OUTPUT_DIR)

    print("\n[Figures] Generating visualizations...")
    plot_class_distribution(proteins, FIGURES_DIR / "class_distribution.svg")
    plot_length_distributions(proteins, FIGURES_DIR / "length_distributions.svg")
    plot_experimental_comparison(proteins, FIGURES_DIR / "experimental_comparison.svg")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Data preparation complete!")
    print("=" * 70)

    n_experimental = sum(1 for p in proteins if p.in_experimental)
    n_hmm_only = len(proteins) - n_experimental

    print(f"\n📊 Dataset Summary:")
    print(f"  Total proteins: {len(proteins)}")
    print(f"  Classes: {len(class_counts)}")
    print(f"  Experimental: {n_experimental} ({n_experimental/len(proteins)*100:.1f}%)")
    print(f"  HMM-only: {n_hmm_only} ({n_hmm_only/len(proteins)*100:.1f}%)")

    # Class imbalance warning
    max_class = max(class_counts.values())
    min_class = min(class_counts.values())
    imbalance = max_class / min_class

    if imbalance > 5:
        print(f"\n⚠️  WARNING: Severe class imbalance detected ({imbalance:.1f}:1)")
        print("    Consider using class weights or SMOTE in classification.")

    print(f"\n📁 Outputs saved to: {OUTPUT_DIR}")
    print(f"\nNext step: Run 02_extract_features.py to compute embeddings")


if __name__ == "__main__":
    main()
