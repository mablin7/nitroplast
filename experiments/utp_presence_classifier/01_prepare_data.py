#!/usr/bin/env python3
"""
Step 1: Prepare Data for uTP Presence Classifier

This script:
1. Extracts mature domains from uTP proteins using HMM-based detection
2. Prepares control candidates from the proteome (length-matched, excluding uTP proteins)
3. Outputs a FASTA file of control candidates for DeepLoc web service

After running this script:
1. Upload `control_candidates.fasta` to https://services.healthtech.dtu.dk/services/DeepLoc-2.0/
2. Download the results
3. Run `02_filter_controls.py` to filter by localization
4. Run `03_train_classifier.py` to train the classifier

Usage:
    uv run python experiments/utp_presence_classifier/01_prepare_data.py
"""

import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# =============================================================================
# Configuration
# =============================================================================

# HMM search parameters
HMM_EVALUE_THRESHOLD = 0.01
MIN_HMM_SCORE = 30.0
MIN_HMM_COVERAGE_START = 50

# Sequence filtering
MIN_MATURE_LENGTH = 50
MAX_MATURE_LENGTH = 2000
MIN_UTP_LENGTH = 60

# Control group selection
LENGTH_TOLERANCE = 0.2
RANDOM_SEED = 42

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Input files
UTP_PROTEINS_FILE = DATA_DIR / "Import_candidates.fasta"
PROTEOME_FILE = DATA_DIR / "ADK1075_proteomics_DB_2.fasta"
HMM_PROFILE = SCRIPT_DIR.parent / "utp_homolog_search" / "utp.hmm"

# Output files
MATURE_DOMAINS_FILE = OUTPUT_DIR / "mature_domains.fasta"
CONTROL_CANDIDATES_FILE = OUTPUT_DIR / "control_candidates.fasta"
METADATA_FILE = OUTPUT_DIR / "preparation_metadata.txt"


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


class MatureDomain(NamedTuple):
    """Container for extracted mature domain."""

    name: str
    sequence: str
    original_length: int
    utp_start: int
    utp_end: int


# =============================================================================
# HMM-based uTP Detection
# =============================================================================


def run_hmmsearch(sequences_file: Path, hmm_file: Path, output_prefix: Path) -> Path:
    """Run hmmsearch to detect uTP regions in sequences."""
    domtblout = output_prefix.with_suffix(".domtblout")

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
) -> dict[str, tuple[int, int]]:
    """Detect uTP regions in sequences using HMM search."""
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

    # Extract uTP boundaries for each sequence
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
            utp_start = best_hit.ali_from
            utp_end = best_hit.target_length
            utp_regions[target_name] = (utp_start, utp_end)

    return utp_regions


def extract_mature_domains(
    sequences: dict[str, str],
    utp_regions: dict[str, tuple[int, int]],
) -> list[MatureDomain]:
    """Extract mature domains by clipping off the uTP region."""
    mature_domains = []

    for name, seq in sequences.items():
        if name not in utp_regions:
            continue

        utp_start, utp_end = utp_regions[name]
        seq_len = len(seq)

        # utp_start is 1-indexed from hmmsearch
        utp_start_0idx = utp_start - 1

        # Validate uTP position
        utp_length = seq_len - utp_start_0idx
        if utp_length < MIN_UTP_LENGTH:
            continue

        # Extract mature domain with buffer for linker
        buffer = 15
        mature_end = max(0, utp_start_0idx - buffer)
        mature_seq = seq[:mature_end]

        # Clean sequence
        clean_seq = [aa for aa in mature_seq if aa in "ACDEFGHIKLMNPQRSTVWY"]
        mature_seq = "".join(clean_seq)

        # Validate length
        if len(mature_seq) < MIN_MATURE_LENGTH or len(mature_seq) > MAX_MATURE_LENGTH:
            continue

        mature_domains.append(
            MatureDomain(
                name=name,
                sequence=mature_seq,
                original_length=seq_len,
                utp_start=utp_start,
                utp_end=utp_end,
            )
        )

    return mature_domains


# =============================================================================
# Control Candidate Selection
# =============================================================================


def load_proteome(proteome_file: Path) -> dict[str, str]:
    """Load proteome sequences, filtering out UCYN-A sequences."""
    sequences = {}

    for record in SeqIO.parse(proteome_file, "fasta"):
        desc = record.description.lower()
        if "cyanobacterium" in desc or "ucyn" in desc or "endosymbiont" in desc:
            continue

        seq = str(record.seq).replace("-", "").replace("*", "")
        if len(seq) >= MIN_MATURE_LENGTH:
            sequences[record.id] = seq

    return sequences


def select_control_candidates(
    proteome: dict[str, str],
    utp_names: set[str],
    target_lengths: list[int],
    n_candidates: int,
    length_tolerance: float = LENGTH_TOLERANCE,
    random_state: int = RANDOM_SEED,
) -> list[tuple[str, str]]:
    """
    Select control candidates matching the length distribution of uTP proteins.

    These candidates will be filtered by DeepLoc in the next step.
    We select MORE candidates than needed to allow for filtering.
    """
    np.random.seed(random_state)

    # Filter out uTP proteins
    candidates = {name: seq for name, seq in proteome.items() if name not in utp_names}
    print(f"  Candidates after excluding uTP proteins: {len(candidates)}")

    # Bin candidates by length
    candidate_list = [(name, seq, len(seq)) for name, seq in candidates.items()]

    # For each target length, find matching candidates
    selected = []
    used_names = set()

    # Select 2x candidates to allow for filtering losses
    target_n = n_candidates * 2

    for target_len in target_lengths:
        if len(selected) >= target_n:
            break

        min_match = int(target_len * (1 - length_tolerance))
        max_match = int(target_len * (1 + length_tolerance))

        matches = [
            (name, seq)
            for name, seq, length in candidate_list
            if min_match <= length <= max_match and name not in used_names
        ]

        if matches:
            idx = np.random.randint(len(matches))
            name, seq = matches[idx]
            selected.append((name, seq))
            used_names.add(name)

    # Fill with random samples if needed
    if len(selected) < target_n:
        remaining = [
            (name, seq) for name, seq, _ in candidate_list if name not in used_names
        ]
        np.random.shuffle(remaining)
        selected.extend(remaining[: target_n - len(selected)])

    return selected[:target_n]


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("Step 1: Prepare Data for uTP Presence Classifier")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check inputs
    if not UTP_PROTEINS_FILE.exists():
        raise FileNotFoundError(f"uTP proteins file not found: {UTP_PROTEINS_FILE}")
    if not PROTEOME_FILE.exists():
        raise FileNotFoundError(f"Proteome file not found: {PROTEOME_FILE}")
    if not HMM_PROFILE.exists():
        raise FileNotFoundError(f"HMM profile not found: {HMM_PROFILE}")

    # Check hmmsearch
    try:
        subprocess.run(["hmmsearch", "-h"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("hmmsearch not found. Please install HMMER.")

    # =========================================================================
    # Step 1: Load and process uTP proteins
    # =========================================================================
    print("\n[1/4] Loading uTP proteins...")
    utp_sequences = {
        record.id: str(record.seq).replace("-", "").replace("*", "")
        for record in SeqIO.parse(UTP_PROTEINS_FILE, "fasta")
    }
    print(f"  Loaded {len(utp_sequences)} uTP protein sequences")

    # =========================================================================
    # Step 2: Detect uTP regions and extract mature domains
    # =========================================================================
    print("\n[2/4] Detecting uTP regions using HMM profile...")
    utp_regions = detect_utp_regions(utp_sequences, HMM_PROFILE, OUTPUT_DIR)
    print(f"  Detected uTP regions in {len(utp_regions)} sequences")

    print("\n[3/4] Extracting mature domains...")
    mature_domains = extract_mature_domains(utp_sequences, utp_regions)
    print(f"  Extracted {len(mature_domains)} valid mature domains")

    # Save mature domains
    records = [
        SeqRecord(
            Seq(md.sequence),
            id=md.name,
            description=f"mature_domain len={len(md.sequence)} utp_start={md.utp_start}",
        )
        for md in mature_domains
    ]
    SeqIO.write(records, MATURE_DOMAINS_FILE, "fasta")
    print(f"  Saved to {MATURE_DOMAINS_FILE}")

    # Statistics
    lengths = [len(md.sequence) for md in mature_domains]
    print(
        f"  Length distribution: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.0f}"
    )

    # =========================================================================
    # Step 4: Prepare control candidates
    # =========================================================================
    print("\n[4/4] Preparing control candidates for DeepLoc...")
    proteome = load_proteome(PROTEOME_FILE)
    print(f"  Loaded {len(proteome)} sequences from proteome")

    utp_names = set(md.name for md in mature_domains)
    target_lengths = [len(md.sequence) for md in mature_domains]
    n_controls = len(mature_domains)

    control_candidates = select_control_candidates(
        proteome, utp_names, target_lengths, n_controls
    )
    print(f"  Selected {len(control_candidates)} control candidates")

    # Save control candidates
    records = [
        SeqRecord(Seq(seq), id=name, description=f"control_candidate len={len(seq)}")
        for name, seq in control_candidates
    ]
    SeqIO.write(records, CONTROL_CANDIDATES_FILE, "fasta")
    print(f"  Saved to {CONTROL_CANDIDATES_FILE}")

    # Control statistics
    control_lengths = [len(seq) for _, seq in control_candidates]
    print(
        f"  Length distribution: min={min(control_lengths)}, max={max(control_lengths)}, mean={np.mean(control_lengths):.0f}"
    )

    # =========================================================================
    # Save metadata
    # =========================================================================
    with open(METADATA_FILE, "w") as f:
        f.write("uTP Presence Classifier - Data Preparation Metadata\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"uTP proteins loaded: {len(utp_sequences)}\n")
        f.write(f"uTP regions detected: {len(utp_regions)}\n")
        f.write(f"Mature domains extracted: {len(mature_domains)}\n")
        f.write(f"Control candidates selected: {len(control_candidates)}\n")
        f.write(
            f"\nMature domain lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.0f}\n"
        )
        f.write(
            f"Control candidate lengths: min={min(control_lengths)}, max={max(control_lengths)}, mean={np.mean(control_lengths):.0f}\n"
        )

    # =========================================================================
    # Instructions
    # =========================================================================
    print("\n" + "=" * 70)
    print("Data preparation complete!")
    print("=" * 70)
    print("\nNext steps:")
    print(f"1. Upload {CONTROL_CANDIDATES_FILE.name} to DeepLoc 2.0:")
    print("   https://services.healthtech.dtu.dk/services/DeepLoc-2.0/")
    print("\n2. Download the results (CSV format)")
    print("\n3. Run: python 02_filter_controls.py <deeploc_results.csv>")
    print("\n4. Run: python 03_train_classifier.py")


if __name__ == "__main__":
    main()
