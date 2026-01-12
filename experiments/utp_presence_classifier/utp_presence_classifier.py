#!/usr/bin/env python3
"""
uTP Presence Classifier

This script trains a binary classifier to predict whether a protein should have
a uTP (UCYN-A transit peptide) based on its mature domain sequence.

Key innovations:
1. HMM-based uTP detection and clipping - uses hmmsearch to identify the exact
   uTP region boundaries rather than a fixed cutoff
2. Robust control group selection - matches length distribution and filters
   for cytoplasmic localization to avoid other targeting signals

Outputs:
- mature_domains.fasta: Extracted mature domains from uTP proteins
- control_sequences.fasta: Selected control sequences
- embeddings.h5: ProtT5 embeddings for all sequences
- classifier_results.csv: Performance metrics
- roc_curve.svg: ROC curve visualization
- confusion_matrix.svg: Confusion matrix
- length_distribution.svg: Length distribution comparison

Usage:
    uv run python experiments/utp_presence_classifier/utp_presence_classifier.py

Requirements:
    - hmmer (hmmsearch) installed and in PATH
    - ProtT5 model (downloaded automatically on first run)
"""

import gc
import re
import subprocess
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import h5py
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from scipy.stats import binomtest, ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    permutation_test_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

# HMM search parameters
# The HMM was built from C-terminal (uTP) regions of aligned sequences (713 positions)
# The conserved uTP motifs are in the early positions (1-150) of the HMM
# ali_from in hmmsearch output tells us where the uTP region starts in the target
HMM_EVALUE_THRESHOLD = 0.01  # Lenient threshold since these are known uTP proteins
MIN_HMM_SCORE = 30.0  # Minimum bit score for a valid hit
MIN_HMM_COVERAGE_START = 50  # HMM match should start within first 50 positions

# Sequence filtering
MIN_MATURE_LENGTH = 50  # Minimum length for mature domain
MAX_MATURE_LENGTH = 2000  # Maximum length for mature domain
MIN_UTP_LENGTH = 60  # Minimum expected uTP length

# Control group selection
LENGTH_TOLERANCE = 0.2  # Allow 20% length difference for matching
RANDOM_SEED = 42

# Classifier parameters
TEST_SIZE = 0.2
N_PERMUTATIONS = 500
N_CV_FOLDS = 5

# ProtT5 model
PROTT5_MODEL = "Rostlab/prot_t5_xl_uniref50"

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
CONTROL_SEQS_FILE = OUTPUT_DIR / "control_sequences.fasta"
EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.h5"
CLASSIFIER_FILE = OUTPUT_DIR / "best_classifier.joblib"

# =============================================================================
# Styling
# =============================================================================

sns.set_theme("paper")
matplotlib.rcParams.update(
    {
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "font.size": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
    }
)


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
    """
    Run hmmsearch to detect uTP regions in sequences.

    Args:
        sequences_file: FASTA file with protein sequences
        hmm_file: HMM profile file
        output_prefix: Prefix for output files

    Returns:
        Path to domain table output file
    """
    domtblout = output_prefix.with_suffix(".domtblout")

    cmd = [
        "hmmsearch",
        "--domtblout",
        str(domtblout),
        "-E",
        str(HMM_EVALUE_THRESHOLD),
        "--domE",
        str(HMM_EVALUE_THRESHOLD),
        "--noali",  # Don't output alignments (faster)
        str(hmm_file),
        str(sequences_file),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"hmmsearch failed: {result.stderr}")

    return domtblout


def parse_hmmsearch_domtblout(domtblout_path: Path) -> list[HMMHit]:
    """
    Parse hmmsearch domain table output.

    Domain table format (columns):
    0: target name
    1: target accession
    2: target length
    3: query name
    4: query accession
    5: query length
    6: full sequence E-value
    7: full sequence score
    8: full sequence bias
    9-10: domain number info
    11: domain c-Evalue
    12: domain i-Evalue
    13: domain score
    14: domain bias
    15-16: hmm coord (from, to)
    17-18: ali coord (from, to)
    19-20: env coord (from, to)
    21: accuracy
    22+: description
    """
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


def get_hmm_length(hmm_file: Path) -> int:
    """Extract HMM profile length from file."""
    with open(hmm_file) as f:
        for line in f:
            if line.startswith("LENG"):
                return int(line.split()[1])
    raise ValueError("Could not find LENG in HMM file")


def detect_utp_regions(
    sequences: dict[str, str],
    hmm_file: Path,
    evalue_threshold: float = HMM_EVALUE_THRESHOLD,
    min_score: float = MIN_HMM_SCORE,
    min_hmm_start: int = MIN_HMM_COVERAGE_START,
) -> dict[str, tuple[int, int]]:
    """
    Detect uTP regions in sequences using HMM search.

    The HMM was built from C-terminal (uTP) regions of aligned sequences.
    The conserved uTP motifs are in the early positions (1-150) of the HMM.

    Strategy:
    1. Run hmmsearch against all sequences
    2. For each sequence, find hits where the HMM match starts near the
       beginning of the HMM (positions 1-50), indicating the conserved uTP
    3. The ali_from position tells us where the uTP starts in the target
    4. The mature domain is everything before ali_from

    Args:
        sequences: Dictionary of sequence name -> sequence
        hmm_file: Path to HMM profile
        evalue_threshold: Maximum E-value for significant hits
        min_score: Minimum bit score for a valid hit
        min_hmm_start: HMM match should start within this position

    Returns:
        Dictionary of sequence name -> (utp_start, utp_end) positions (1-indexed)
    """
    # Write sequences to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        for name, seq in sequences.items():
            f.write(f">{name}\n{seq}\n")
        temp_fasta = Path(f.name)

    try:
        # Run hmmsearch
        domtblout = run_hmmsearch(temp_fasta, hmm_file, OUTPUT_DIR / "utp_detection")
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
        # Find the best hit that represents the uTP region
        # A good uTP hit should:
        # 1. Have the HMM match starting near the beginning (hmm_from <= min_hmm_start)
        # 2. Have a good score and E-value
        # 3. Be located near the C-terminus of the target protein

        best_hit = None
        best_score = 0

        for hit in target_hits:
            # Skip hits with poor E-value or score
            if hit.domain_evalue > evalue_threshold:
                continue
            if hit.domain_score < min_score:
                continue

            # The HMM match should start near the beginning of the HMM
            # (positions 1-50) to capture the conserved uTP motifs
            if hit.hmm_from > min_hmm_start:
                continue

            # Prefer hits with higher scores
            if hit.domain_score > best_score:
                best_hit = hit
                best_score = hit.domain_score

        if best_hit is not None:
            # The ali_from position tells us where the uTP region starts
            # in the target sequence (1-indexed)
            utp_start = best_hit.ali_from
            utp_end = best_hit.target_length

            utp_regions[target_name] = (utp_start, utp_end)

    return utp_regions


def extract_mature_domains(
    sequences: dict[str, str],
    utp_regions: dict[str, tuple[int, int]],
    min_length: int = MIN_MATURE_LENGTH,
    max_length: int = MAX_MATURE_LENGTH,
    min_utp_length: int = MIN_UTP_LENGTH,
) -> list[MatureDomain]:
    """
    Extract mature domains by clipping off the uTP region.

    The uTP region was detected by HMM search. The mature domain is
    everything before the uTP start position, minus a small buffer
    to account for linker regions.

    Args:
        sequences: Dictionary of sequence name -> sequence
        utp_regions: Dictionary of sequence name -> (utp_start, utp_end) (1-indexed)
        min_length: Minimum mature domain length
        max_length: Maximum mature domain length
        min_utp_length: Minimum expected uTP length

    Returns:
        List of MatureDomain objects
    """
    mature_domains = []

    for name, seq in sequences.items():
        if name not in utp_regions:
            continue

        utp_start, utp_end = utp_regions[name]
        seq_len = len(seq)

        # utp_start is 1-indexed from hmmsearch, convert to 0-indexed
        utp_start_0idx = utp_start - 1

        # Validate uTP position (should be near C-terminus)
        utp_length = seq_len - utp_start_0idx
        if utp_length < min_utp_length:
            continue

        # The mature domain is everything before the uTP
        # Subtract a small buffer (15 aa) to account for linker regions
        # that might not be part of the HMM but are still part of the targeting signal
        buffer = 15
        mature_end = max(0, utp_start_0idx - buffer)

        mature_seq = seq[:mature_end]

        # Clean sequence (remove any non-standard characters)
        clean_seq = []
        for aa in mature_seq:
            if aa in "ACDEFGHIKLMNPQRSTVWY":
                clean_seq.append(aa)
        mature_seq = "".join(clean_seq)

        # Validate length
        if len(mature_seq) < min_length or len(mature_seq) > max_length:
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
# Control Group Selection
# =============================================================================


def load_proteome(proteome_file: Path) -> dict[str, str]:
    """Load proteome sequences, filtering out UCYN-A sequences."""
    sequences = {}

    for record in SeqIO.parse(proteome_file, "fasta"):
        # Skip UCYN-A/cyanobacterium sequences
        desc = record.description.lower()
        if "cyanobacterium" in desc or "ucyn" in desc or "endosymbiont" in desc:
            continue

        seq = str(record.seq).replace("-", "").replace("*", "")
        if len(seq) >= MIN_MATURE_LENGTH:
            sequences[record.id] = seq

    return sequences


def check_signalp_available() -> bool:
    """Check if SignalP 6 is available."""
    try:
        result = subprocess.run(
            ["signalp6", "--version"], capture_output=True, text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_targetp_available() -> bool:
    """Check if TargetP 2 is available."""
    try:
        result = subprocess.run(
            ["targetp", "--version"], capture_output=True, text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def run_signalp(sequences: dict[str, str], output_dir: Path) -> dict[str, str]:
    """
    Run SignalP 6 to predict signal peptides.

    Returns dict mapping sequence name to prediction ('SP', 'LIPO', 'TAT', 'OTHER').
    """
    # Write sequences to temp file
    input_file = output_dir / "signalp_input.fasta"
    with open(input_file, "w") as f:
        for name, seq in sequences.items():
            # SignalP only needs first 70 aa
            f.write(f">{name}\n{seq[:70]}\n")

    output_file = output_dir / "signalp_output"

    cmd = [
        "signalp6",
        "--fastafile",
        str(input_file),
        "--output_dir",
        str(output_file),
        "--format",
        "txt",
        "--organism",
        "eukarya",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"SignalP warning: {result.stderr}")
        return {}

    # Parse results
    predictions = {}
    result_file = output_file / "prediction_results.txt"
    if result_file.exists():
        with open(result_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    predictions[parts[0]] = parts[1]

    return predictions


def run_targetp(sequences: dict[str, str], output_dir: Path) -> dict[str, str]:
    """
    Run TargetP 2.0 to predict targeting peptides.

    Returns dict mapping sequence name to prediction ('mTP', 'SP', 'cTP', 'OTHER').
    mTP = mitochondrial, SP = signal peptide, cTP = chloroplast transit peptide
    """
    # Write sequences to temp file
    input_file = output_dir / "targetp_input.fasta"
    with open(input_file, "w") as f:
        for name, seq in sequences.items():
            f.write(f">{name}\n{seq[:200]}\n")  # TargetP uses first 200 aa

    output_file = output_dir / "targetp_output.txt"

    cmd = [
        "targetp",
        "-fasta",
        str(input_file),
        "-org",
        "pl",  # plant/algae
        "-format",
        "short",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"TargetP warning: {result.stderr}")
        return {}

    # Parse results from stdout
    predictions = {}
    for line in result.stdout.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            predictions[parts[0]] = parts[1]

    return predictions


def predict_localization_batch(
    sequences: dict[str, str], output_dir: Path
) -> dict[str, str]:
    """
    Predict subcellular localization for a batch of sequences.

    Uses SignalP/TargetP if available, otherwise returns 'unknown' for all.

    Returns dict mapping sequence name to localization prediction.
    Possible values: 'cytoplasmic', 'secretory', 'mitochondrial', 'chloroplast', 'unknown'
    """
    predictions = {name: "unknown" for name in sequences}

    # Try TargetP first (more comprehensive for plant/algae)
    if check_targetp_available():
        print("  Using TargetP 2.0 for localization prediction...")
        targetp_results = run_targetp(sequences, output_dir)
        for name, pred in targetp_results.items():
            if pred == "mTP":
                predictions[name] = "mitochondrial"
            elif pred == "SP":
                predictions[name] = "secretory"
            elif pred == "cTP":
                predictions[name] = "chloroplast"
            else:
                predictions[name] = "cytoplasmic"
        return predictions

    # Try SignalP (signal peptides only)
    if check_signalp_available():
        print("  Using SignalP 6 for signal peptide prediction...")
        signalp_results = run_signalp(sequences, output_dir)
        for name, pred in signalp_results.items():
            if pred in ("SP", "LIPO", "TAT"):
                predictions[name] = "secretory"
            else:
                predictions[name] = "cytoplasmic"
        return predictions

    # No tool available - warn user
    print(
        "  Warning: Neither SignalP nor TargetP found. Skipping localization filtering."
    )
    print("  Install SignalP 6 or TargetP 2.0 for better control group selection.")
    print("  Proceeding without localization-based filtering...")

    return predictions


def select_control_sequences(
    proteome: dict[str, str],
    utp_names: set[str],
    target_lengths: list[int],
    n_controls: int,
    output_dir: Path,
    length_tolerance: float = LENGTH_TOLERANCE,
    filter_by_localization: bool = True,
    random_state: int = RANDOM_SEED,
) -> list[tuple[str, str]]:
    """
    Select control sequences matching the length distribution of uTP proteins.

    Strategy:
    1. Exclude known uTP proteins
    2. Optionally filter for cytoplasmic localization using SignalP/TargetP
    3. Match length distribution using stratified sampling

    Args:
        proteome: Dictionary of all proteome sequences
        utp_names: Set of known uTP protein names to exclude
        target_lengths: List of target lengths to match
        n_controls: Number of control sequences to select
        output_dir: Directory for temporary files
        length_tolerance: Fractional tolerance for length matching
        filter_by_localization: Whether to filter by predicted localization
        random_state: Random seed for reproducibility

    Returns:
        List of (name, sequence) tuples
    """
    np.random.seed(random_state)

    # Filter out uTP proteins
    candidates = {name: seq for name, seq in proteome.items() if name not in utp_names}
    print(f"  Initial candidates (excluding uTP proteins): {len(candidates)}")

    # Optionally filter for cytoplasmic localization
    if filter_by_localization:
        # Run localization prediction on candidates
        predictions = predict_localization_batch(candidates, output_dir)

        # Keep only cytoplasmic or unknown (if tool not available)
        cytoplasmic = {
            name: seq
            for name, seq in candidates.items()
            if predictions.get(name, "unknown") in ("cytoplasmic", "unknown")
        }

        # Count predictions
        pred_counts = Counter(predictions.values())
        print(f"  Localization predictions: {dict(pred_counts)}")

        # Fall back to all candidates if not enough cytoplasmic
        if len(cytoplasmic) >= n_controls:
            candidates = cytoplasmic
            print(f"  After localization filtering: {len(candidates)} candidates")
        else:
            print(
                f"  Warning: Only {len(cytoplasmic)} cytoplasmic candidates, "
                f"using all {len(candidates)} candidates"
            )

    # Create length bins for stratified sampling
    target_lengths = np.array(target_lengths)

    # Bin candidates by length
    candidate_list = [(name, seq, len(seq)) for name, seq in candidates.items()]

    # For each target length, find matching candidates
    selected = []
    used_names = set()

    for target_len in target_lengths:
        if len(selected) >= n_controls:
            break

        # Find candidates within tolerance
        min_match = int(target_len * (1 - length_tolerance))
        max_match = int(target_len * (1 + length_tolerance))

        matches = [
            (name, seq)
            for name, seq, length in candidate_list
            if min_match <= length <= max_match and name not in used_names
        ]

        if matches:
            # Randomly select one
            idx = np.random.randint(len(matches))
            name, seq = matches[idx]
            selected.append((name, seq))
            used_names.add(name)

    # If we don't have enough, fill with random samples
    if len(selected) < n_controls:
        remaining = [
            (name, seq) for name, seq, _ in candidate_list if name not in used_names
        ]
        np.random.shuffle(remaining)
        selected.extend(remaining[: n_controls - len(selected)])

    return selected[:n_controls]


# =============================================================================
# Embedding Generation
# =============================================================================


def load_prott5_model(device: str = "cpu"):
    """Load ProtT5 tokenizer and model."""
    from transformers import T5EncoderModel, T5Tokenizer

    print(f"Loading ProtT5 model (this may take a while on first run)...")
    tokenizer = T5Tokenizer.from_pretrained(PROTT5_MODEL, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(PROTT5_MODEL)
    model = model.to(device)
    model.eval()

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return tokenizer, model


def embed_sequences(
    sequences: list[str], tokenizer, model, device: str = "cpu"
) -> list[np.ndarray]:
    """
    Generate ProtT5 embeddings for a list of sequences.

    Returns mean-pooled embeddings (1024-dim vector per sequence).
    """
    embeddings = []

    for seq in tqdm(sequences, desc="Computing embeddings"):
        # Clean sequence
        seq = re.sub(r"[UZOJB*]", "X", seq)
        seq_spaced = " ".join(seq)

        # Tokenize
        ids = tokenizer.batch_encode_plus(
            [seq_spaced], add_special_tokens=True, padding=True
        )
        input_ids = torch.tensor(ids["input_ids"]).to(device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(device)

        # Get embeddings
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)

        embedding = output.last_hidden_state.cpu().numpy()

        # Remove padding and special tokens, then mean pool
        seq_len = (attention_mask[0] == 1).sum().item()
        seq_embedding = embedding[0][: seq_len - 1]  # -1 for </s> token

        # Mean pool to get fixed-size representation
        mean_embedding = np.mean(seq_embedding, axis=0).astype(np.float32)
        embeddings.append(mean_embedding)

    return embeddings


def load_or_compute_embeddings(
    sequences: dict[str, str], embeddings_file: Path, device: str = "cpu"
) -> dict[str, np.ndarray]:
    """Load embeddings from file or compute them if not available."""

    if embeddings_file.exists():
        print(f"Loading existing embeddings from {embeddings_file}")
        embeddings = {}
        with h5py.File(embeddings_file, "r") as f:
            for key in f.keys():
                embeddings[key] = f[key][()]

        # Check if all sequences have embeddings
        missing = set(sequences.keys()) - set(embeddings.keys())
        if not missing:
            return embeddings
        print(f"Warning: {len(missing)} sequences missing embeddings, recomputing all")

    # Compute embeddings
    print("Computing ProtT5 embeddings...")
    tokenizer, model = load_prott5_model(device)

    seq_names = list(sequences.keys())
    seq_list = [sequences[n] for n in seq_names]

    embedding_list = embed_sequences(seq_list, tokenizer, model, device)

    # Save to HDF5
    embeddings = {}
    with h5py.File(embeddings_file, "w") as f:
        for name, emb in zip(seq_names, embedding_list):
            f.create_dataset(name, data=emb)
            embeddings[name] = emb

    print(f"Saved embeddings to {embeddings_file}")

    # Clean up
    del tokenizer, model
    gc.collect()

    return embeddings


# =============================================================================
# Classification
# =============================================================================


def prepare_dataset(
    utp_embeddings: dict[str, np.ndarray], control_embeddings: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Prepare dataset for binary classification.

    Returns:
        Tuple of (X, Y, names) where:
        - X: Feature matrix
        - Y: Labels (1 = uTP, 0 = control)
        - names: Sequence names
    """
    X = []
    Y = []
    names = []

    for name, emb in utp_embeddings.items():
        X.append(emb)
        Y.append(1)
        names.append(name)

    for name, emb in control_embeddings.items():
        X.append(emb)
        Y.append(0)
        names.append(name)

    return np.array(X), np.array(Y), names


def train_and_evaluate_classifiers(
    X: np.ndarray, Y: np.ndarray, n_permutations: int = N_PERMUTATIONS
) -> dict:
    """
    Train multiple classifiers and evaluate with cross-validation and permutation tests.
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=Y
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED
        ),
        "SVC": SVC(probability=True, random_state=RANDOM_SEED),
    }

    results = {}
    best_accuracy = 0
    best_classifier = None
    best_name = None

    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")

        # Fit classifier
        clf.fit(X_train, Y_train)

        # Evaluate on test set
        Y_pred = clf.predict(X_test)
        Y_prob = (
            clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
        )

        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred)
        recall = recall_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred)

        # Cross-validation
        cv_scores = cross_val_score(clf, X_scaled, Y, cv=N_CV_FOLDS, scoring="accuracy")

        # Binomial test (vs random guessing)
        n = len(Y_test)
        binom_pval = binomtest(
            round(accuracy * n), n, 0.5, alternative="greater"
        ).pvalue

        # Permutation test
        print(f"  Running permutation test ({n_permutations} permutations)...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score, perm_scores, perm_pval = permutation_test_score(
                clf,
                X_scaled,
                Y,
                scoring="accuracy",
                cv=N_CV_FOLDS,
                n_permutations=n_permutations,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            )

        results[name] = {
            "classifier": clf,
            "scaler": scaler,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "permutation_pvalue": perm_pval,
            "binomial_pvalue": binom_pval,
            "Y_test": Y_test,
            "Y_pred": Y_pred,
            "Y_prob": Y_prob,
        }

        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall: {recall:.2%}")
        print(f"  F1 Score: {f1:.2f}")
        print(f"  CV Score: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
        print(f"  Permutation p-value: {perm_pval:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_classifier = clf
            best_name = name

    results["_best"] = {
        "name": best_name,
        "classifier": best_classifier,
        "scaler": scaler,
    }

    return results


# =============================================================================
# Visualization
# =============================================================================


def plot_length_distribution(
    utp_lengths: list[int], control_lengths: list[int], output_file: Path
):
    """Plot length distribution comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(
        min(min(utp_lengths), min(control_lengths)),
        max(max(utp_lengths), max(control_lengths)),
        50,
    )

    ax.hist(
        utp_lengths,
        bins=bins,
        alpha=0.6,
        label=f"uTP proteins (n={len(utp_lengths)})",
        color="#2ecc71",
    )
    ax.hist(
        control_lengths,
        bins=bins,
        alpha=0.6,
        label=f"Control proteins (n={len(control_lengths)})",
        color="#3498db",
    )

    # KS test
    ks_stat, ks_pval = ks_2samp(utp_lengths, control_lengths)

    ax.set_xlabel("Mature domain length (aa)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Length Distribution Comparison\nKS test: D={ks_stat:.3f}, p={ks_pval:.3f}"
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved length distribution to {output_file}")


def plot_roc_curve(results: dict, output_file: Path):
    """Plot ROC curve for the best classifier."""
    best_name = results["_best"]["name"]
    Y_test = results[best_name]["Y_test"]
    Y_prob = results[best_name]["Y_prob"]

    if Y_prob is None:
        print("Classifier doesn't support probability predictions, skipping ROC curve")
        return

    fpr, tpr, _ = roc_curve(Y_test, Y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"{best_name} (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - uTP Presence Prediction")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved ROC curve to {output_file}")


def plot_confusion_matrix(results: dict, output_file: Path):
    """Plot confusion matrix for the best classifier."""
    best_name = results["_best"]["name"]
    Y_test = results[best_name]["Y_test"]
    Y_pred = results[best_name]["Y_pred"]

    cm = confusion_matrix(Y_test, Y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Control", "uTP"],
        yticklabels=["Control", "uTP"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {best_name}")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved confusion matrix to {output_file}")


def save_results(results: dict, output_dir: Path):
    """Save classification results to CSV."""
    rows = []
    for name, res in results.items():
        if name.startswith("_"):
            continue
        rows.append(
            {
                "Classifier": name,
                "Accuracy": f"{res['accuracy']:.1%}",
                "Precision": f"{res['precision']:.1%}",
                "Recall": f"{res['recall']:.1%}",
                "F1 Score": f"{res['f1_score']:.2f}",
                "CV Score": f"{res['cv_mean']:.1%} ± {res['cv_std']:.1%}",
                "Permutation p-value": f"{res['permutation_pvalue']:.4f}",
                "Binomial p-value": f"{res['binomial_pvalue']:.4e}",
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("Accuracy", ascending=False)
    df.to_csv(output_dir / "classifier_results.csv", index=False)

    print(f"\nClassifier Results:")
    print(df.to_string(index=False))


# =============================================================================
# Main Pipeline
# =============================================================================


def main():
    """Main analysis pipeline."""
    print("=" * 70)
    print("uTP Presence Classifier")
    print("Binary classification: Does a protein have uTP based on mature domain?")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check input files
    if not UTP_PROTEINS_FILE.exists():
        raise FileNotFoundError(f"uTP proteins file not found: {UTP_PROTEINS_FILE}")
    if not PROTEOME_FILE.exists():
        raise FileNotFoundError(f"Proteome file not found: {PROTEOME_FILE}")
    if not HMM_PROFILE.exists():
        raise FileNotFoundError(f"HMM profile not found: {HMM_PROFILE}")

    # Check hmmsearch is available
    try:
        subprocess.run(["hmmsearch", "-h"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("hmmsearch not found. Please install HMMER.")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # =========================================================================
    # Step 1: Load uTP proteins
    # =========================================================================
    print("\n[Step 1] Loading uTP proteins...")
    utp_sequences = {
        record.id: str(record.seq).replace("-", "").replace("*", "")
        for record in SeqIO.parse(UTP_PROTEINS_FILE, "fasta")
    }
    print(f"Loaded {len(utp_sequences)} uTP protein sequences")

    # =========================================================================
    # Step 2: Detect uTP regions using HMM
    # =========================================================================
    print("\n[Step 2] Detecting uTP regions using HMM profile...")
    utp_regions = detect_utp_regions(utp_sequences, HMM_PROFILE)
    print(f"Detected uTP regions in {len(utp_regions)} sequences")

    # =========================================================================
    # Step 3: Extract mature domains
    # =========================================================================
    print("\n[Step 3] Extracting mature domains...")
    mature_domains = extract_mature_domains(utp_sequences, utp_regions)
    print(f"Extracted {len(mature_domains)} valid mature domains")

    # Save mature domains
    records = [
        SeqRecord(
            Seq(md.sequence),
            id=md.name,
            description=f"mature_domain utp_start={md.utp_start}",
        )
        for md in mature_domains
    ]
    SeqIO.write(records, MATURE_DOMAINS_FILE, "fasta")
    print(f"Saved mature domains to {MATURE_DOMAINS_FILE}")

    # =========================================================================
    # Step 4: Load proteome and select control sequences
    # =========================================================================
    print("\n[Step 4] Loading proteome and selecting control sequences...")
    proteome = load_proteome(PROTEOME_FILE)
    print(f"Loaded {len(proteome)} non-UCYN-A sequences from proteome")

    # Get target lengths from mature domains
    utp_names = set(md.name for md in mature_domains)
    target_lengths = [len(md.sequence) for md in mature_domains]

    # Select control sequences
    n_controls = len(mature_domains)  # Equal size groups
    control_seqs = select_control_sequences(
        proteome, utp_names, target_lengths, n_controls, OUTPUT_DIR
    )
    print(f"Selected {len(control_seqs)} control sequences")

    # Save control sequences
    records = [
        SeqRecord(Seq(seq), id=name, description="control")
        for name, seq in control_seqs
    ]
    SeqIO.write(records, CONTROL_SEQS_FILE, "fasta")
    print(f"Saved control sequences to {CONTROL_SEQS_FILE}")

    # =========================================================================
    # Step 5: Plot length distributions
    # =========================================================================
    print("\n[Step 5] Comparing length distributions...")
    utp_lengths = [len(md.sequence) for md in mature_domains]
    control_lengths = [len(seq) for _, seq in control_seqs]
    plot_length_distribution(
        utp_lengths, control_lengths, OUTPUT_DIR / "length_distribution.svg"
    )

    # =========================================================================
    # Step 6: Compute embeddings
    # =========================================================================
    print("\n[Step 6] Computing ProtT5 embeddings...")

    # Combine all sequences
    all_sequences = {md.name: md.sequence for md in mature_domains}
    all_sequences.update({name: seq for name, seq in control_seqs})

    embeddings = load_or_compute_embeddings(all_sequences, EMBEDDINGS_FILE, device)

    # Split embeddings
    utp_embeddings = {
        md.name: embeddings[md.name] for md in mature_domains if md.name in embeddings
    }
    control_embeddings = {
        name: embeddings[name] for name, _ in control_seqs if name in embeddings
    }

    print(f"uTP embeddings: {len(utp_embeddings)}")
    print(f"Control embeddings: {len(control_embeddings)}")

    # =========================================================================
    # Step 7: Train classifiers
    # =========================================================================
    print("\n[Step 7] Training classifiers...")
    X, Y, names = prepare_dataset(utp_embeddings, control_embeddings)
    print(f"Dataset: {len(X)} samples ({sum(Y)} uTP, {len(Y) - sum(Y)} control)")

    results = train_and_evaluate_classifiers(X, Y, n_permutations=N_PERMUTATIONS)

    # =========================================================================
    # Step 8: Save best classifier
    # =========================================================================
    print("\n[Step 8] Saving best classifier...")
    best_clf = results["_best"]["classifier"]
    best_scaler = results["_best"]["scaler"]
    joblib.dump({"classifier": best_clf, "scaler": best_scaler}, CLASSIFIER_FILE)
    print(f"Saved to {CLASSIFIER_FILE}")

    # =========================================================================
    # Step 9: Generate plots
    # =========================================================================
    print("\n[Step 9] Generating plots...")
    plot_roc_curve(results, OUTPUT_DIR / "roc_curve.svg")
    plot_confusion_matrix(results, OUTPUT_DIR / "confusion_matrix.svg")

    # =========================================================================
    # Step 10: Save results
    # =========================================================================
    print("\n[Step 10] Saving results...")
    save_results(results, OUTPUT_DIR)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)

    best_name = results["_best"]["name"]
    best_acc = results[best_name]["accuracy"]
    best_pval = results[best_name]["permutation_pvalue"]

    print(f"\nKey Finding:")
    print(f"  Best classifier: {best_name}")
    print(f"  Accuracy: {best_acc:.1%}")
    print(f"  Permutation p-value: {best_pval:.4f}")

    if best_pval < 0.05:
        print(f"\nThe mature domain sequence can predict uTP presence significantly")
        print(f"better than chance, suggesting that uTP-containing proteins share")
        print(f"common sequence features in their mature domains.")
    else:
        print(f"\nThe classifier did not perform significantly better than chance.")
        print(f"This suggests uTP-containing proteins may not have distinctive")
        print(f"sequence features in their mature domains.")


if __name__ == "__main__":
    main()
