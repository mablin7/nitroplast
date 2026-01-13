#!/usr/bin/env python3
"""
00_extract_features.py - Extract Features from uTP and Mature Domains

Extracts continuous biophysical features from both:
1. uTP region (C-terminal ~120 AA)
2. Mature domain (N-terminal portion before uTP)

This enables correlation analysis between uTP and mature domain properties,
testing whether the mature domain "knows" about its uTP.

Usage:
    uv run python experiments/utp_mature_correlation/00_extract_features.py
"""

import subprocess
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

warnings.filterwarnings("ignore")

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Input files
IMPORT_CANDIDATES = DATA_DIR / "Import_candidates.fasta"
HMM_PROFILE = SCRIPT_DIR.parent / "utp_homolog_search" / "utp.hmm"
MOTIF_PATTERNS = (
    SCRIPT_DIR.parent / "utp_motif_coverage" / "output" / "motif_patterns.csv"
)
GOOD_CTERM_FULL = (
    SCRIPT_DIR.parent / "utp_motif_analysis" / "output" / "good-c-term-full.fasta"
)

# =============================================================================
# Constants
# =============================================================================

# Default uTP length if HMM boundary not detected
DEFAULT_UTP_LENGTH = 120

# Minimum lengths
MIN_MATURE_LENGTH = 50
MIN_UTP_LENGTH = 50

# HMM parameters
HMM_EVALUE_THRESHOLD = 0.01

# =============================================================================
# Amino Acid Properties
# =============================================================================

HYDROPATHY = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

CHARGE = {
    "A": 0,
    "R": 1,
    "N": 0,
    "D": -1,
    "C": 0,
    "Q": 0,
    "E": -1,
    "G": 0,
    "H": 0.1,
    "I": 0,
    "L": 0,
    "K": 1,
    "M": 0,
    "F": 0,
    "P": 0,
    "S": 0,
    "T": 0,
    "W": 0,
    "Y": 0,
    "V": 0,
}

DISORDER_PROPENSITY = {
    "A": 0.06,
    "R": -0.18,
    "N": -0.01,
    "D": 0.19,
    "C": 0.02,
    "Q": -0.04,
    "E": 0.74,
    "G": 0.17,
    "H": 0.05,
    "I": -0.42,
    "L": -0.34,
    "K": 0.59,
    "M": -0.23,
    "F": -0.39,
    "P": 0.41,
    "S": 0.34,
    "T": 0.04,
    "W": -0.48,
    "Y": -0.27,
    "V": -0.39,
}

HELIX_PROPENSITY = {
    "A": 1.42,
    "R": 0.98,
    "N": 0.67,
    "D": 1.01,
    "C": 0.70,
    "Q": 1.11,
    "E": 1.51,
    "G": 0.57,
    "H": 1.00,
    "I": 1.08,
    "L": 1.21,
    "K": 1.16,
    "M": 1.45,
    "F": 1.13,
    "P": 0.57,
    "S": 0.77,
    "T": 0.83,
    "W": 1.08,
    "Y": 0.69,
    "V": 1.06,
}

SHEET_PROPENSITY = {
    "A": 0.83,
    "R": 0.93,
    "N": 0.89,
    "D": 0.54,
    "C": 1.19,
    "Q": 1.10,
    "E": 0.37,
    "G": 0.75,
    "H": 0.87,
    "I": 1.60,
    "L": 1.30,
    "K": 0.74,
    "M": 1.05,
    "F": 1.38,
    "P": 0.55,
    "S": 0.75,
    "T": 1.19,
    "W": 1.37,
    "Y": 1.47,
    "V": 1.70,
}

AA_GROUPS = {
    "hydrophobic": set("AILMFVWY"),
    "polar": set("STNQ"),
    "charged_pos": set("RKH"),
    "charged_neg": set("DE"),
    "aromatic": set("FWY"),
    "small": set("AGSCT"),
    "aliphatic": set("ILV"),
    "proline": set("P"),
    "glycine": set("G"),
    "cysteine": set("C"),
}


# =============================================================================
# HMM Boundary Detection
# =============================================================================


class HMMHit(NamedTuple):
    """HMM domain hit."""

    target_name: str
    target_length: int
    env_from: int
    env_to: int
    score: float
    evalue: float


def run_hmmsearch(sequences: dict, hmm_profile: Path) -> dict:
    """Run hmmsearch to detect uTP boundaries."""

    if not hmm_profile.exists():
        print(f"  Warning: HMM profile not found at {hmm_profile}")
        return {}

    # Write sequences to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        for name, seq in sequences.items():
            f.write(f">{name}\n{seq}\n")
        temp_fasta = f.name

    # Run hmmsearch
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tbl", delete=False) as f:
        temp_tbl = f.name

    try:
        cmd = [
            "hmmsearch",
            "--domtblout",
            temp_tbl,
            "-E",
            str(HMM_EVALUE_THRESHOLD),
            str(hmm_profile),
            temp_fasta,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  Warning: hmmsearch failed: {result.stderr}")
            return {}

        # Parse results
        hits = {}
        with open(temp_tbl) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 23:
                    continue

                name = parts[0]
                target_len = int(parts[2])
                env_from = int(parts[19])
                env_to = int(parts[20])
                score = float(parts[13])
                evalue = float(parts[12])

                if name not in hits or score > hits[name].score:
                    hits[name] = HMMHit(
                        target_name=name,
                        target_length=target_len,
                        env_from=env_from,
                        env_to=env_to,
                        score=score,
                        evalue=evalue,
                    )

        return hits

    finally:
        Path(temp_fasta).unlink(missing_ok=True)
        Path(temp_tbl).unlink(missing_ok=True)


# =============================================================================
# Feature Extraction
# =============================================================================


def clean_sequence(seq: str) -> str:
    """Remove non-standard amino acids."""
    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join(aa for aa in seq.upper() if aa in standard_aa)


def compute_physicochemical(seq: str) -> dict:
    """Compute physicochemical properties."""
    seq = clean_sequence(seq)
    if len(seq) < 5:
        return {
            "length": len(seq) if seq else np.nan,
            "molecular_weight": np.nan,
            "gravy": np.nan,
            "isoelectric_point": np.nan,
            "instability_index": np.nan,
            "aromaticity": np.nan,
        }

    try:
        analysis = ProteinAnalysis(seq)
        return {
            "length": len(seq),
            "molecular_weight": analysis.molecular_weight(),
            "gravy": analysis.gravy(),
            "isoelectric_point": analysis.isoelectric_point(),
            "instability_index": analysis.instability_index(),
            "aromaticity": analysis.aromaticity(),
        }
    except Exception:
        return {
            "length": len(seq),
            "molecular_weight": np.nan,
            "gravy": np.nan,
            "isoelectric_point": np.nan,
            "instability_index": np.nan,
            "aromaticity": np.nan,
        }


def compute_aa_composition(seq: str) -> dict:
    """Compute amino acid composition."""
    seq = clean_sequence(seq)
    if len(seq) == 0:
        return {f"aa_{aa}": np.nan for aa in "ACDEFGHIKLMNPQRSTVWY"}

    counts = Counter(seq)
    total = len(seq)
    return {f"aa_{aa}": counts.get(aa, 0) / total for aa in "ACDEFGHIKLMNPQRSTVWY"}


def compute_group_composition(seq: str) -> dict:
    """Compute amino acid group composition."""
    seq = clean_sequence(seq)
    if len(seq) == 0:
        return {f"group_{name}": np.nan for name in AA_GROUPS}

    total = len(seq)
    result = {}
    for name, aa_set in AA_GROUPS.items():
        count = sum(1 for aa in seq if aa in aa_set)
        result[f"group_{name}"] = count / total

    return result


def compute_charge_features(seq: str) -> dict:
    """Compute charge features."""
    seq = clean_sequence(seq)
    if len(seq) < 10:
        return {
            "net_charge": np.nan,
            "charge_density": np.nan,
            "positive_fraction": np.nan,
            "negative_fraction": np.nan,
        }

    charges = [CHARGE.get(aa, 0) for aa in seq]
    net_charge = sum(charges)

    return {
        "net_charge": net_charge,
        "charge_density": net_charge / len(seq),
        "positive_fraction": sum(1 for aa in seq if aa in "RKH") / len(seq),
        "negative_fraction": sum(1 for aa in seq if aa in "DE") / len(seq),
    }


def compute_hydrophobicity(seq: str) -> dict:
    """Compute hydrophobicity features."""
    seq = clean_sequence(seq)
    if len(seq) < 10:
        return {
            "hydro_mean": np.nan,
            "hydro_std": np.nan,
        }

    hydro = [HYDROPATHY.get(aa, 0) for aa in seq]

    return {
        "hydro_mean": np.mean(hydro),
        "hydro_std": np.std(hydro),
    }


def compute_disorder(seq: str) -> dict:
    """Compute disorder propensity."""
    seq = clean_sequence(seq)
    if len(seq) < 10:
        return {
            "disorder_mean": np.nan,
            "disorder_fraction": np.nan,
        }

    disorder = [DISORDER_PROPENSITY.get(aa, 0) for aa in seq]

    window = min(21, len(seq))
    if len(disorder) >= window:
        smoothed = np.convolve(disorder, np.ones(window) / window, mode="valid")
    else:
        smoothed = disorder

    disorder_fraction = sum(1 for d in smoothed if d > 0) / len(smoothed)

    return {
        "disorder_mean": np.mean(disorder),
        "disorder_fraction": disorder_fraction,
    }


def compute_secondary_structure(seq: str) -> dict:
    """Compute secondary structure propensity."""
    seq = clean_sequence(seq)
    if len(seq) < 5:
        return {
            "helix_propensity": np.nan,
            "sheet_propensity": np.nan,
        }

    helix_scores = [HELIX_PROPENSITY.get(aa, 1.0) for aa in seq]
    sheet_scores = [SHEET_PROPENSITY.get(aa, 1.0) for aa in seq]

    return {
        "helix_propensity": np.mean(helix_scores),
        "sheet_propensity": np.mean(sheet_scores),
    }


def compute_complexity(seq: str) -> dict:
    """Compute sequence complexity."""
    seq = clean_sequence(seq)
    if len(seq) < 5:
        return {"entropy": np.nan}

    counts = Counter(seq)
    total = len(seq)
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)

    return {"entropy": entropy}


def extract_all_features(seq: str, prefix: str) -> dict:
    """Extract all features with a prefix."""
    features = {}

    features.update(
        {f"{prefix}_{k}": v for k, v in compute_physicochemical(seq).items()}
    )
    features.update(
        {f"{prefix}_{k}": v for k, v in compute_aa_composition(seq).items()}
    )
    features.update(
        {f"{prefix}_{k}": v for k, v in compute_group_composition(seq).items()}
    )
    features.update(
        {f"{prefix}_{k}": v for k, v in compute_charge_features(seq).items()}
    )
    features.update(
        {f"{prefix}_{k}": v for k, v in compute_hydrophobicity(seq).items()}
    )
    features.update({f"{prefix}_{k}": v for k, v in compute_disorder(seq).items()})
    features.update(
        {f"{prefix}_{k}": v for k, v in compute_secondary_structure(seq).items()}
    )
    features.update({f"{prefix}_{k}": v for k, v in compute_complexity(seq).items()})

    return features


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("uTP-Mature Domain Feature Extraction")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Load sequences
    # =========================================================================
    print("\n[1/4] Loading sequences...")

    sequences = {}
    for record in SeqIO.parse(IMPORT_CANDIDATES, "fasta"):
        sequences[record.id] = str(record.seq)
    print(f"  Loaded {len(sequences)} sequences from Import_candidates.fasta")

    # Load experimental set
    experimental_names = set()
    if GOOD_CTERM_FULL.exists():
        for record in SeqIO.parse(GOOD_CTERM_FULL, "fasta"):
            experimental_names.add(record.id)
        print(f"  Experimental set: {len(experimental_names)} sequences")

    # Load motif patterns
    if MOTIF_PATTERNS.exists():
        motif_df = pd.read_csv(MOTIF_PATTERNS)
        print(f"  Motif patterns: {len(motif_df)} entries")
    else:
        motif_df = None

    # =========================================================================
    # Step 2: Detect uTP boundaries via HMM
    # =========================================================================
    print("\n[2/4] Detecting uTP boundaries via HMM...")

    hmm_hits = run_hmmsearch(sequences, HMM_PROFILE)
    print(f"  HMM hits: {len(hmm_hits)} sequences with detected boundaries")

    # =========================================================================
    # Step 3: Extract features for both regions
    # =========================================================================
    print("\n[3/4] Extracting features from uTP and mature domains...")

    results = []
    skipped_short_mature = 0
    skipped_short_utp = 0

    for name, full_seq in sequences.items():
        # Determine uTP boundary
        if name in hmm_hits:
            hit = hmm_hits[name]
            utp_start = hit.env_from - 1  # Convert to 0-indexed
            hmm_detected = True
        else:
            # Fall back to fixed length from C-terminus
            utp_start = max(0, len(full_seq) - DEFAULT_UTP_LENGTH)
            hmm_detected = False

        # Split into mature and uTP
        mature_seq = full_seq[:utp_start]
        utp_seq = full_seq[utp_start:]

        # Filter by length
        if len(mature_seq) < MIN_MATURE_LENGTH:
            skipped_short_mature += 1
            continue
        if len(utp_seq) < MIN_UTP_LENGTH:
            skipped_short_utp += 1
            continue

        # Extract features
        row = {"name": name}
        row["full_length"] = len(full_seq)
        row["mature_length"] = len(mature_seq)
        row["utp_length"] = len(utp_seq)
        row["utp_start"] = utp_start
        row["hmm_detected"] = hmm_detected
        row["source"] = "experimental" if name in experimental_names else "hmm_only"

        # Mature domain features
        row.update(extract_all_features(mature_seq, "mature"))

        # uTP features
        row.update(extract_all_features(utp_seq, "utp"))

        results.append(row)

    print(f"  Extracted features for {len(results)} proteins")
    print(f"  Skipped (mature too short): {skipped_short_mature}")
    print(f"  Skipped (uTP too short): {skipped_short_utp}")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # =========================================================================
    # Step 4: Add motif information
    # =========================================================================
    print("\n[4/4] Adding motif information...")

    if motif_df is not None:
        motif_info = motif_df[
            ["name", "n_motifs", "terminal_class", "is_valid_terminal"]
        ].copy()
        df = df.merge(motif_info, on="name", how="left")
        print(f"  Added motif info for {df['n_motifs'].notna().sum()} proteins")

    # =========================================================================
    # Save outputs
    # =========================================================================
    print("\nSaving outputs...")

    df.to_csv(OUTPUT_DIR / "mature_utp_features.csv", index=False)
    print(f"  Saved mature_utp_features.csv ({len(df)} rows, {len(df.columns)} cols)")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nDataset:")
    print(f"  Total proteins: {len(df)}")
    print(f"  Experimental: {(df['source'] == 'experimental').sum()}")
    print(f"  HMM-only: {(df['source'] == 'hmm_only').sum()}")
    print(f"  HMM boundary detected: {df['hmm_detected'].sum()}")

    print(f"\nSequence lengths:")
    print(
        f"  Full length: {df['full_length'].mean():.1f} ± {df['full_length'].std():.1f}"
    )
    print(
        f"  Mature domain: {df['mature_length'].mean():.1f} ± {df['mature_length'].std():.1f}"
    )
    print(f"  uTP region: {df['utp_length'].mean():.1f} ± {df['utp_length'].std():.1f}")

    # Count feature columns
    mature_cols = [
        c
        for c in df.columns
        if c.startswith("mature_") and df[c].dtype in [np.float64, np.int64]
    ]
    utp_cols = [
        c
        for c in df.columns
        if c.startswith("utp_") and df[c].dtype in [np.float64, np.int64]
    ]
    print(f"\nFeatures extracted:")
    print(f"  Mature domain features: {len(mature_cols)}")
    print(f"  uTP features: {len(utp_cols)}")

    print(f"\n📁 Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
