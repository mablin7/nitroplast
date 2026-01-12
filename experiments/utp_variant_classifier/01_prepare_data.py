#!/usr/bin/env python3
"""
01_prepare_data.py - Data Preparation for uTP Variant Classifier

This script:
1. Loads sequences with known motif variants from previous MEME analysis
2. Uses HMM to detect precise uTP boundaries
3. Extracts mature domains and uTP regions
4. Links to transcriptome annotations

Output:
- processed_proteins.csv: Protein metadata + variant assignments
- mature_domains.fasta: Extracted mature domains
- utp_regions.fasta: Extracted uTP sequences
- variant_distribution.svg: Distribution of variants

Usage:
    uv run python experiments/utp_variant_classifier/01_prepare_data.py
"""

import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
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

# HMM search parameters
HMM_EVALUE_THRESHOLD = 0.01
MIN_HMM_SCORE = 25.0  # Slightly lower than presence classifier for broader coverage
MIN_HMM_COVERAGE_START = 60  # Allow some flexibility in HMM alignment start

# Sequence filtering
MIN_MATURE_LENGTH = 30
MAX_MATURE_LENGTH = 3000
MIN_UTP_LENGTH = 50

# Variant filtering
MIN_VARIANT_COUNT = 10  # Minimum samples per class for meaningful analysis

# Random seed
RANDOM_SEED = 42

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Input files - use the EXISTING motif analysis data
MEME_FILE = SCRIPT_DIR.parent / "utp_motif_analysis" / "data" / "meme_gb.xml"
FULL_SEQS_FILE = (
    SCRIPT_DIR.parent / "utp_motif_analysis" / "output" / "good-c-term-full.fasta"
)
ANNOTATIONS_FILE = DATA_DIR / "Bbigelowii_transcriptome_annotations.csv"
HMM_PROFILE = SCRIPT_DIR.parent / "utp_homolog_search" / "utp.hmm"

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
    motif_variant: str
    motif_ids: tuple
    hmm_score: float


# =============================================================================
# Load Existing Motif Analysis Data
# =============================================================================


def load_sequences_and_motifs(meme_file: Path, seqs_file: Path) -> tuple[dict, dict]:
    """
    Load sequences and their motif assignments from previous MEME analysis.

    Returns:
        Tuple of (sequences_dict, motifs_dict) where:
        - sequences_dict: {seq_name: sequence_string}
        - motifs_dict: {seq_name: tuple_of_motif_ids}
    """
    # Parse MEME XML
    meme_xml = ET.parse(meme_file)

    # Get sequence name mapping (sequence_id -> sequence_name)
    seq_names = {
        tag.attrib["id"]: tag.attrib["name"] for tag in meme_xml.findall(".//sequence")
    }

    # Extract motif assignments per sequence
    sites = meme_xml.findall(".//scanned_sites")
    sequences_motifs = defaultdict(list)
    for tag in sites:
        seq_id = tag.attrib["sequence_id"]
        for site in tag.findall("scanned_site"):
            sequences_motifs[seq_id].append(site.attrib)

    # Sort motifs by position and convert to tuple of IDs
    motifs_dict = {}
    for seq_id, motifs in sequences_motifs.items():
        if seq_id not in seq_names:
            continue
        seq_name = seq_names[seq_id]
        sorted_motifs = sorted(motifs, key=lambda m: int(m["position"]))
        motifs_dict[seq_name] = tuple(m["motif_id"] for m in sorted_motifs)

    # Load sequences
    seqs = {s.id: str(s.seq) for s in SeqIO.parse(seqs_file, "fasta")}

    # Filter to sequences that have both sequence and motif data
    common_names = set(seqs.keys()) & set(motifs_dict.keys())
    seqs = {k: v for k, v in seqs.items() if k in common_names}
    motifs_dict = {k: v for k, v in motifs_dict.items() if k in common_names}

    print(f"  Loaded {len(seqs)} sequences with motif assignments")

    return seqs, motifs_dict


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
    hits_by_target: dict[str, list[HMMHit]] = defaultdict(list)
    for hit in hits:
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
# Protein Processing
# =============================================================================


def process_proteins(
    sequences: dict[str, str],
    motifs_dict: dict[str, tuple],
    utp_regions: dict[str, tuple[int, int, float]],
) -> list[ProcessedProtein]:
    """
    Process all proteins: extract mature domains and uTP regions.
    """
    processed = []

    for name, seq in sequences.items():
        if name not in motifs_dict:
            continue

        motif_ids = motifs_dict[name]

        # Get HMM-based uTP boundary if available, otherwise use heuristic
        if name in utp_regions:
            utp_start, utp_end, hmm_score = utp_regions[name]
        else:
            # Fallback: estimate uTP start based on sequence length
            # uTP is typically ~100-150 aa at C-terminus
            seq_len = len(seq)
            utp_start = max(1, seq_len - 150)
            utp_end = seq_len
            hmm_score = 0.0

        # Create variant signature from ordered motif IDs
        variant_name = "+".join(m.replace("motif_", "") for m in motif_ids)

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
            continue
        if len(utp_seq) < MIN_UTP_LENGTH:
            continue

        processed.append(
            ProcessedProtein(
                name=name,
                full_sequence=seq,
                mature_sequence=mature_seq,
                utp_sequence=utp_seq,
                utp_start=utp_start,
                utp_end=utp_end,
                motif_variant=variant_name,
                motif_ids=motif_ids,
                hmm_score=hmm_score,
            )
        )

    return processed


def filter_variants(
    proteins: list[ProcessedProtein],
    min_count: int = MIN_VARIANT_COUNT,
) -> tuple[list[ProcessedProtein], dict[str, int]]:
    """
    Filter to keep only variants with sufficient samples.

    Returns:
        Tuple of (filtered_proteins, variant_counts)
    """
    # Count variants
    variant_counts = Counter(p.motif_variant for p in proteins)

    # Filter
    valid_variants = {v for v, c in variant_counts.items() if c >= min_count}
    filtered = [p for p in proteins if p.motif_variant in valid_variants]

    # Re-count after filtering
    final_counts = Counter(p.motif_variant for p in filtered)

    return filtered, final_counts


# =============================================================================
# Annotation Loading
# =============================================================================


def load_annotations(annotations_file: Path) -> pd.DataFrame:
    """Load and parse EggNOG annotations."""
    # Read the CSV with proper handling of the header
    df = pd.read_csv(
        annotations_file,
        comment="#",
        header=0,
        low_memory=False,
    )

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Rename first column if needed
    if df.columns[0].startswith("#"):
        df = df.rename(columns={df.columns[0]: "query_name"})

    return df


def merge_annotations(
    proteins: list[ProcessedProtein],
    annotations: pd.DataFrame,
) -> pd.DataFrame:
    """Merge protein data with annotations."""
    if not proteins:
        return pd.DataFrame()

    # Create protein dataframe
    protein_data = []
    for p in proteins:
        protein_data.append(
            {
                "name": p.name,
                "mature_length": len(p.mature_sequence),
                "utp_length": len(p.utp_sequence),
                "full_length": len(p.full_sequence),
                "utp_start": p.utp_start,
                "motif_variant": p.motif_variant,
                "motif_ids": "+".join(p.motif_ids),
                "hmm_score": p.hmm_score,
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
        # Try first column
        merged = protein_df.merge(
            annotations,
            left_on="name",
            right_on=annotations.columns[0],
            how="left",
        )

    return merged


# =============================================================================
# Visualization
# =============================================================================


def plot_variant_distribution(
    variant_counts: dict[str, int],
    output_file: Path,
):
    """Plot the distribution of uTP variants."""
    # Sort by count
    sorted_variants = sorted(variant_counts.items(), key=lambda x: -x[1])
    variants = [v[0] for v in sorted_variants]
    counts = [v[1] for v in sorted_variants]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar plot
    colors = plt.cm.tab20(np.linspace(0, 1, len(variants)))
    bars = ax.bar(range(len(variants)), counts, color=colors)

    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Labels
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([f"uTP-{v}" for v in variants], rotation=45, ha="right")
    ax.set_xlabel("Motif Variant")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Distribution of uTP Variants (n={sum(counts)} proteins, {len(variants)} variants)"
    )

    # Add percentages
    total = sum(counts)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        pct = count / total * 100
        if pct >= 5:  # Only show for variants with ≥5%
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{pct:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved variant distribution plot to {output_file}")


def plot_length_distributions(
    proteins: list[ProcessedProtein],
    output_file: Path,
):
    """Plot length distributions of mature domains and uTP regions by variant."""
    df = pd.DataFrame(
        [
            {
                "variant": p.motif_variant,
                "mature_length": len(p.mature_sequence),
                "utp_length": len(p.utp_sequence),
            }
            for p in proteins
        ]
    )

    # Get top 6 variants
    variant_counts = df["variant"].value_counts()
    top_variants = variant_counts.head(6).index.tolist()
    df_top = df[df["variant"].isin(top_variants)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mature length
    sns.boxplot(data=df_top, x="variant", y="mature_length", ax=axes[0])
    axes[0].set_xlabel("uTP Variant")
    axes[0].set_ylabel("Mature Domain Length (aa)")
    axes[0].set_title("Mature Domain Length by Variant")
    axes[0].tick_params(axis="x", rotation=45)

    # uTP length
    sns.boxplot(data=df_top, x="variant", y="utp_length", ax=axes[1])
    axes[1].set_xlabel("uTP Variant")
    axes[1].set_ylabel("uTP Region Length (aa)")
    axes[1].set_title("uTP Region Length by Variant")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved length distributions plot to {output_file}")


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("01_prepare_data.py - Data Preparation for uTP Variant Classifier")
    print("=" * 70)

    # Create output directories
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Check input files
    if not FULL_SEQS_FILE.exists():
        raise FileNotFoundError(
            f"Sequences file not found: {FULL_SEQS_FILE}\n"
            f"Please run experiments/utp_motif_analysis/motif_combination_extraction.py first."
        )
    if not HMM_PROFILE.exists():
        raise FileNotFoundError(f"HMM profile not found: {HMM_PROFILE}")
    if not MEME_FILE.exists():
        raise FileNotFoundError(f"MEME file not found: {MEME_FILE}")

    # Check hmmsearch
    try:
        subprocess.run(["hmmsearch", "-h"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("hmmsearch not found. Please install HMMER.")

    # =========================================================================
    # Step 1: Load sequences and motif assignments from previous analysis
    # =========================================================================
    print("\n[1/6] Loading sequences and motif assignments...")
    sequences, motifs_dict = load_sequences_and_motifs(MEME_FILE, FULL_SEQS_FILE)

    # Show variant distribution from MEME analysis
    variant_counts_raw = Counter(
        "+".join(m.replace("motif_", "") for m in motifs)
        for motifs in motifs_dict.values()
    )
    print(f"  Found {len(variant_counts_raw)} unique motif variants")

    # =========================================================================
    # Step 2: Detect uTP regions using HMM for precise boundaries
    # =========================================================================
    print("\n[2/6] Detecting uTP regions using HMM profile...")
    utp_regions = detect_utp_regions(sequences, HMM_PROFILE, DATA_OUTPUT_DIR)
    print(f"  Detected HMM-based uTP regions in {len(utp_regions)} sequences")

    if utp_regions:
        scores = [s for _, _, s in utp_regions.values()]
        print(
            f"  HMM scores: min={min(scores):.1f}, max={max(scores):.1f}, median={np.median(scores):.1f}"
        )

    # =========================================================================
    # Step 3: Process proteins
    # =========================================================================
    print("\n[3/6] Processing proteins (extracting mature domains and uTP regions)...")
    proteins = process_proteins(sequences, motifs_dict, utp_regions)
    print(
        f"  Processed {len(proteins)} proteins with valid mature domains and uTP regions"
    )

    # =========================================================================
    # Step 4: Filter variants
    # =========================================================================
    print(f"\n[4/6] Filtering variants (min {MIN_VARIANT_COUNT} samples per class)...")
    proteins, variant_counts = filter_variants(proteins, MIN_VARIANT_COUNT)

    if not proteins:
        print("  ERROR: No proteins remaining after filtering!")
        print("  Try reducing MIN_VARIANT_COUNT or check your data.")
        return

    print(f"  Retained {len(proteins)} proteins across {len(variant_counts)} variants:")
    for variant, count in sorted(variant_counts.items(), key=lambda x: -x[1]):
        print(f"    uTP-{variant}: {count} ({count/len(proteins)*100:.1f}%)")

    # =========================================================================
    # Step 5: Load and merge annotations
    # =========================================================================
    print("\n[5/6] Loading and merging annotations...")
    if ANNOTATIONS_FILE.exists():
        annotations = load_annotations(ANNOTATIONS_FILE)
        print(f"  Loaded {len(annotations)} annotation records")

        merged_df = merge_annotations(proteins, annotations)

        # Count proteins with annotations
        n_annotated = (
            merged_df["seed_eggNOG_ortholog"].notna().sum()
            if "seed_eggNOG_ortholog" in merged_df.columns
            else 0
        )
        print(
            f"  Merged annotations: {n_annotated}/{len(merged_df)} proteins have EggNOG hits"
        )
    else:
        print(f"  Warning: Annotations file not found: {ANNOTATIONS_FILE}")
        merged_df = pd.DataFrame(
            [
                {
                    "name": p.name,
                    "mature_length": len(p.mature_sequence),
                    "utp_length": len(p.utp_sequence),
                    "full_length": len(p.full_sequence),
                    "utp_start": p.utp_start,
                    "motif_variant": p.motif_variant,
                    "motif_ids": "+".join(p.motif_ids),
                    "hmm_score": p.hmm_score,
                }
                for p in proteins
            ]
        )

    # =========================================================================
    # Step 6: Save outputs
    # =========================================================================
    print("\n[6/6] Saving outputs...")

    # Save processed proteins CSV
    merged_df.to_csv(DATA_OUTPUT_DIR / "processed_proteins.csv", index=False)
    print(f"  Saved protein metadata to {DATA_OUTPUT_DIR / 'processed_proteins.csv'}")

    # Save mature domains FASTA
    mature_records = [
        SeqRecord(
            Seq(p.mature_sequence),
            id=p.name,
            description=f"variant={p.motif_variant} len={len(p.mature_sequence)}",
        )
        for p in proteins
    ]
    SeqIO.write(mature_records, DATA_OUTPUT_DIR / "mature_domains.fasta", "fasta")
    print(
        f"  Saved {len(mature_records)} mature domains to {DATA_OUTPUT_DIR / 'mature_domains.fasta'}"
    )

    # Save uTP regions FASTA
    utp_records = [
        SeqRecord(
            Seq(p.utp_sequence),
            id=p.name,
            description=f"variant={p.motif_variant} len={len(p.utp_sequence)}",
        )
        for p in proteins
    ]
    SeqIO.write(utp_records, DATA_OUTPUT_DIR / "utp_regions.fasta", "fasta")
    print(
        f"  Saved {len(utp_records)} uTP regions to {DATA_OUTPUT_DIR / 'utp_regions.fasta'}"
    )

    # Save variant mapping
    variant_mapping = pd.DataFrame(
        [
            {
                "name": p.name,
                "motif_variant": p.motif_variant,
                "motif_ids": "+".join(p.motif_ids),
            }
            for p in proteins
        ]
    )
    variant_mapping.to_csv(DATA_OUTPUT_DIR / "variant_mapping.csv", index=False)

    # =========================================================================
    # Generate figures
    # =========================================================================
    print("\n[Figures] Generating visualizations...")
    plot_variant_distribution(variant_counts, FIGURES_DIR / "variant_distribution.svg")
    plot_length_distributions(proteins, FIGURES_DIR / "length_distributions.svg")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Data preparation complete!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Input proteins: {len(sequences)}")
    print(f"  Valid uTP regions: {len(utp_regions)}")
    print(f"  Motif assignments: {len(motifs_dict)}")
    print(
        f"  Final dataset: {len(proteins)} proteins, {len(variant_counts)} variant classes"
    )
    print(f"\nNext step: Run 02_extract_features.py to compute embeddings")


if __name__ == "__main__":
    main()
