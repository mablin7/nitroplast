#!/usr/bin/env python3
"""
00_extract_utp_features.py - Comprehensive uTP Feature Extraction

Extracts continuous biophysical and sequence features from uTP sequences using
state-of-the-art methods for downstream correlation analysis.

Features extracted:
1. Length metrics
2. Amino acid composition (full and grouped)
3. Physicochemical properties (GRAVY, pI, MW, instability)
4. Charge distribution (net, regional, dipole moment)
5. Hydrophobicity profile statistics
6. Disorder propensity
7. Secondary structure propensity
8. Motif-based features

Usage:
    uv run python experiments/utp_feature_correlation/00_extract_utp_features.py
"""

import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils import molecular_weight
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
IMPORT_CANDIDATES = DATA_DIR / "Import_candidates.fasta"  # HMM-predicted proteins
PROTEOMICS_DB = DATA_DIR / "ADK1075_proteomics_DB_2.fasta"  # Full proteome

# From previous analyses
MOTIF_PATTERNS = SCRIPT_DIR.parent / "utp_motif_coverage" / "output" / "motif_patterns.csv"
GOOD_CTERM_FULL = SCRIPT_DIR.parent / "utp_motif_analysis" / "output" / "good-c-term-full.fasta"

# =============================================================================
# Amino Acid Properties (from literature)
# =============================================================================

# Kyte-Doolittle hydropathy scale
HYDROPATHY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# Charge at pH 7
CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'Q': 0, 'E': -1, 'G': 0, 'H': 0.1, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
}

# Disorder propensity (TOP-IDP scale, Campen et al. 2008)
DISORDER_PROPENSITY = {
    'A': 0.06, 'R': -0.18, 'N': -0.01, 'D': 0.19, 'C': 0.02,
    'Q': -0.04, 'E': 0.74, 'G': 0.17, 'H': 0.05, 'I': -0.42,
    'L': -0.34, 'K': 0.59, 'M': -0.23, 'F': -0.39, 'P': 0.41,
    'S': 0.34, 'T': 0.04, 'W': -0.48, 'Y': -0.27, 'V': -0.39
}

# Chou-Fasman helix propensity
HELIX_PROPENSITY = {
    'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70,
    'Q': 1.11, 'E': 1.51, 'G': 0.57, 'H': 1.00, 'I': 1.08,
    'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57,
    'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06
}

# Chou-Fasman sheet propensity
SHEET_PROPENSITY = {
    'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19,
    'Q': 1.10, 'E': 0.37, 'G': 0.75, 'H': 0.87, 'I': 1.60,
    'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55,
    'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70
}

# Amino acid groupings
AA_GROUPS = {
    'hydrophobic': set('AILMFVWY'),
    'polar': set('STNQ'),
    'charged_pos': set('RKH'),
    'charged_neg': set('DE'),
    'aromatic': set('FWY'),
    'small': set('AGSCT'),
    'tiny': set('AGS'),
    'aliphatic': set('ILV'),
    'proline': set('P'),
    'glycine': set('G'),
    'cysteine': set('C'),
}


# =============================================================================
# Feature Extraction Functions
# =============================================================================

def clean_sequence(seq: str) -> str:
    """Remove non-standard amino acids and clean sequence."""
    standard_aa = set('ACDEFGHIKLMNPQRSTVWY')
    return ''.join(aa for aa in seq.upper() if aa in standard_aa)


def compute_aa_composition(seq: str) -> dict:
    """Compute full amino acid composition."""
    seq = clean_sequence(seq)
    if len(seq) == 0:
        return {f'aa_{aa}': np.nan for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    
    counts = Counter(seq)
    total = len(seq)
    return {f'aa_{aa}': counts.get(aa, 0) / total for aa in 'ACDEFGHIKLMNPQRSTVWY'}


def compute_group_composition(seq: str) -> dict:
    """Compute amino acid group composition."""
    seq = clean_sequence(seq)
    if len(seq) == 0:
        return {f'group_{name}': np.nan for name in AA_GROUPS}
    
    total = len(seq)
    result = {}
    for name, aa_set in AA_GROUPS.items():
        count = sum(1 for aa in seq if aa in aa_set)
        result[f'group_{name}'] = count / total
    
    return result


def compute_physicochemical(seq: str) -> dict:
    """Compute physicochemical properties using Biopython."""
    seq = clean_sequence(seq)
    if len(seq) < 5:
        return {
            'length': len(seq) if seq else np.nan,
            'molecular_weight': np.nan,
            'gravy': np.nan,
            'isoelectric_point': np.nan,
            'instability_index': np.nan,
            'aromaticity': np.nan,
        }
    
    try:
        analysis = ProteinAnalysis(seq)
        return {
            'length': len(seq),
            'molecular_weight': analysis.molecular_weight(),
            'gravy': analysis.gravy(),
            'isoelectric_point': analysis.isoelectric_point(),
            'instability_index': analysis.instability_index(),
            'aromaticity': analysis.aromaticity(),
        }
    except Exception:
        return {
            'length': len(seq),
            'molecular_weight': np.nan,
            'gravy': np.nan,
            'isoelectric_point': np.nan,
            'instability_index': np.nan,
            'aromaticity': np.nan,
        }


def compute_charge_features(seq: str) -> dict:
    """Compute charge distribution features."""
    seq = clean_sequence(seq)
    if len(seq) < 10:
        return {
            'net_charge': np.nan,
            'charge_n20': np.nan,
            'charge_c20': np.nan,
            'charge_middle': np.nan,
            'charge_density': np.nan,
            'charge_asymmetry': np.nan,
            'positive_fraction': np.nan,
            'negative_fraction': np.nan,
        }
    
    # Compute charge at each position
    charges = [CHARGE.get(aa, 0) for aa in seq]
    
    # Net charge
    net_charge = sum(charges)
    
    # Regional charges (first 20, last 20, middle)
    n20 = sum(charges[:min(20, len(seq))])
    c20 = sum(charges[-min(20, len(seq)):])
    
    if len(seq) > 40:
        middle = sum(charges[20:-20])
        middle_len = len(seq) - 40
    else:
        middle = 0
        middle_len = 1
    
    # Charge density
    charge_density = net_charge / len(seq)
    
    # Charge asymmetry (difference between N and C terminus)
    charge_asymmetry = n20 - c20
    
    # Fraction of charged residues
    positive_fraction = sum(1 for aa in seq if aa in 'RKH') / len(seq)
    negative_fraction = sum(1 for aa in seq if aa in 'DE') / len(seq)
    
    return {
        'net_charge': net_charge,
        'charge_n20': n20,
        'charge_c20': c20,
        'charge_middle': middle / max(middle_len, 1),
        'charge_density': charge_density,
        'charge_asymmetry': charge_asymmetry,
        'positive_fraction': positive_fraction,
        'negative_fraction': negative_fraction,
    }


def compute_hydrophobicity_profile(seq: str) -> dict:
    """Compute hydrophobicity profile statistics."""
    seq = clean_sequence(seq)
    if len(seq) < 10:
        return {
            'hydro_mean': np.nan,
            'hydro_std': np.nan,
            'hydro_min': np.nan,
            'hydro_max': np.nan,
            'hydro_range': np.nan,
            'hydro_n20': np.nan,
            'hydro_c20': np.nan,
            'hydro_gradient': np.nan,
        }
    
    # Hydropathy values
    hydro = [HYDROPATHY.get(aa, 0) for aa in seq]
    
    # Basic statistics
    hydro_mean = np.mean(hydro)
    hydro_std = np.std(hydro)
    hydro_min = np.min(hydro)
    hydro_max = np.max(hydro)
    
    # Regional hydrophobicity
    n20_hydro = np.mean(hydro[:min(20, len(seq))])
    c20_hydro = np.mean(hydro[-min(20, len(seq)):])
    
    # Hydrophobicity gradient (trend from N to C)
    # Linear regression slope
    x = np.arange(len(hydro))
    slope = np.polyfit(x, hydro, 1)[0] if len(hydro) > 1 else 0
    
    return {
        'hydro_mean': hydro_mean,
        'hydro_std': hydro_std,
        'hydro_min': hydro_min,
        'hydro_max': hydro_max,
        'hydro_range': hydro_max - hydro_min,
        'hydro_n20': n20_hydro,
        'hydro_c20': c20_hydro,
        'hydro_gradient': slope,
    }


def compute_disorder_score(seq: str) -> dict:
    """Compute disorder propensity score (simplified IUPred-like)."""
    seq = clean_sequence(seq)
    if len(seq) < 10:
        return {
            'disorder_mean': np.nan,
            'disorder_std': np.nan,
            'disorder_n20': np.nan,
            'disorder_c20': np.nan,
            'disorder_fraction': np.nan,
        }
    
    # Disorder propensity scores
    disorder = [DISORDER_PROPENSITY.get(aa, 0) for aa in seq]
    
    # Use sliding window average (window=21, like IUPred)
    window = min(21, len(seq))
    if len(disorder) >= window:
        smoothed = np.convolve(disorder, np.ones(window)/window, mode='valid')
    else:
        smoothed = disorder
    
    # Fraction predicted disordered (threshold > 0)
    disorder_fraction = sum(1 for d in smoothed if d > 0) / len(smoothed)
    
    return {
        'disorder_mean': np.mean(disorder),
        'disorder_std': np.std(disorder),
        'disorder_n20': np.mean(disorder[:min(20, len(seq))]),
        'disorder_c20': np.mean(disorder[-min(20, len(seq)):]),
        'disorder_fraction': disorder_fraction,
    }


def compute_secondary_structure_propensity(seq: str) -> dict:
    """Compute secondary structure propensity scores."""
    seq = clean_sequence(seq)
    if len(seq) < 5:
        return {
            'helix_propensity': np.nan,
            'sheet_propensity': np.nan,
            'helix_sheet_ratio': np.nan,
        }
    
    helix_scores = [HELIX_PROPENSITY.get(aa, 1.0) for aa in seq]
    sheet_scores = [SHEET_PROPENSITY.get(aa, 1.0) for aa in seq]
    
    helix_mean = np.mean(helix_scores)
    sheet_mean = np.mean(sheet_scores)
    
    return {
        'helix_propensity': helix_mean,
        'sheet_propensity': sheet_mean,
        'helix_sheet_ratio': helix_mean / sheet_mean if sheet_mean > 0 else np.nan,
    }


def compute_sequence_complexity(seq: str) -> dict:
    """Compute sequence complexity metrics."""
    seq = clean_sequence(seq)
    if len(seq) < 5:
        return {
            'entropy': np.nan,
            'complexity_wootton': np.nan,
        }
    
    # Shannon entropy
    counts = Counter(seq)
    total = len(seq)
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    
    # Wootton-Federhen complexity (SEG-like)
    # Simplified version: entropy normalized by max entropy
    max_entropy = np.log2(20)  # Maximum entropy for 20 amino acids
    complexity = entropy / max_entropy
    
    return {
        'entropy': entropy,
        'complexity_wootton': complexity,
    }


def extract_all_features(seq: str, name: str) -> dict:
    """Extract all features for a single sequence."""
    features = {'name': name}
    
    features.update(compute_physicochemical(seq))
    features.update(compute_aa_composition(seq))
    features.update(compute_group_composition(seq))
    features.update(compute_charge_features(seq))
    features.update(compute_hydrophobicity_profile(seq))
    features.update(compute_disorder_score(seq))
    features.update(compute_secondary_structure_propensity(seq))
    features.update(compute_sequence_complexity(seq))
    
    return features


# =============================================================================
# uTP Extraction
# =============================================================================

def extract_utp_region(full_seq: str, utp_length: int = 120) -> str:
    """Extract the C-terminal uTP region from a full protein sequence."""
    return full_seq[-utp_length:] if len(full_seq) >= utp_length else full_seq


def load_sequences_from_fasta(fasta_path: Path) -> dict:
    """Load sequences from FASTA file."""
    seqs = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        seqs[record.id] = str(record.seq)
    return seqs


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("uTP Feature Extraction")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Step 1: Load sequences
    # =========================================================================
    print("\n[1/4] Loading sequences...")
    
    # Load import candidates (HMM-predicted)
    import_seqs = load_sequences_from_fasta(IMPORT_CANDIDATES)
    print(f"  Import candidates: {len(import_seqs)} sequences")
    
    # Load experimental set (good C-term from MEME analysis)
    if GOOD_CTERM_FULL.exists():
        experimental_seqs = load_sequences_from_fasta(GOOD_CTERM_FULL)
        experimental_names = set(experimental_seqs.keys())
        print(f"  Experimental set: {len(experimental_seqs)} sequences")
    else:
        experimental_names = set()
        print("  Warning: Experimental set not found")
    
    # Load motif patterns
    if MOTIF_PATTERNS.exists():
        motif_df = pd.read_csv(MOTIF_PATTERNS)
        print(f"  Motif patterns: {len(motif_df)} entries")
    else:
        motif_df = None
        print("  Warning: Motif patterns not found")
    
    # =========================================================================
    # Step 2: Extract features for all proteins
    # =========================================================================
    print("\n[2/4] Extracting features from full proteins...")
    
    full_features = []
    for name, seq in import_seqs.items():
        features = extract_all_features(seq, name)
        features['source'] = 'experimental' if name in experimental_names else 'hmm_only'
        full_features.append(features)
    
    full_df = pd.DataFrame(full_features)
    print(f"  Extracted {len(full_df.columns)} features for {len(full_df)} proteins")
    
    # =========================================================================
    # Step 3: Extract features for uTP regions only
    # =========================================================================
    print("\n[3/4] Extracting features from uTP regions (C-terminal 120 AA)...")
    
    utp_features = []
    for name, seq in import_seqs.items():
        utp_seq = extract_utp_region(seq, utp_length=120)
        features = extract_all_features(utp_seq, name)
        features['source'] = 'experimental' if name in experimental_names else 'hmm_only'
        features['utp_actual_length'] = len(utp_seq)
        utp_features.append(features)
    
    utp_df = pd.DataFrame(utp_features)
    utp_df = utp_df.add_prefix('utp_').rename(columns={'utp_name': 'name', 'utp_source': 'source'})
    print(f"  Extracted {len(utp_df.columns)} features for {len(utp_df)} uTP regions")
    
    # =========================================================================
    # Step 4: Add motif-based features
    # =========================================================================
    print("\n[4/4] Adding motif-based features...")
    
    if motif_df is not None:
        # Merge motif patterns
        motif_features = motif_df[['name', 'n_motifs', 'is_valid_terminal', 'starts_with_2_1', 'terminal_class']].copy()
        
        # One-hot encode terminal class
        terminal_dummies = pd.get_dummies(motif_features['terminal_class'], prefix='terminal')
        motif_features = pd.concat([motif_features, terminal_dummies], axis=1)
        
        # Merge with full features
        full_df = full_df.merge(motif_features, on='name', how='left')
        utp_df = utp_df.merge(motif_features, on='name', how='left')
        
        print(f"  Added motif features")
    
    # =========================================================================
    # Save outputs
    # =========================================================================
    print("\nSaving outputs...")
    
    full_df.to_csv(OUTPUT_DIR / "full_protein_features.csv", index=False)
    utp_df.to_csv(OUTPUT_DIR / "utp_features.csv", index=False)
    
    print(f"  Saved full_protein_features.csv ({len(full_df)} rows, {len(full_df.columns)} cols)")
    print(f"  Saved utp_features.csv ({len(utp_df)} rows, {len(utp_df.columns)} cols)")
    
    # =========================================================================
    # Summary statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nFull protein features:")
    print(f"  Total proteins: {len(full_df)}")
    print(f"  Experimental: {(full_df['source'] == 'experimental').sum()}")
    print(f"  HMM-only: {(full_df['source'] == 'hmm_only').sum()}")
    print(f"  Features: {len([c for c in full_df.columns if c not in ['name', 'source']])}")
    
    print(f"\nuTP features:")
    print(f"  Mean uTP length: {utp_df['utp_utp_actual_length'].mean():.1f}")
    print(f"  Min uTP length: {utp_df['utp_utp_actual_length'].min()}")
    print(f"  Max uTP length: {utp_df['utp_utp_actual_length'].max()}")
    
    # List feature categories
    numeric_cols = [c for c in full_df.columns if full_df[c].dtype in [np.float64, np.int64]]
    print(f"\nNumeric features available: {len(numeric_cols)}")
    
    print(f"\n📁 Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
