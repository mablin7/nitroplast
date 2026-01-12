#!/usr/bin/env python3
"""
Compare different variant grouping strategies for uTP classification.

Strategy 1: Terminal motif grouping (group by last motif: 4, 5, 7, 9)
Strategy 2: Lower MIN_COUNT to 5 (keep more fine-grained variants)

This script runs both and compares classification performance.
"""

import warnings
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import balanced_accuracy_score, classification_report
from scipy import stats

warnings.filterwarnings("ignore")

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MEME_FILE = SCRIPT_DIR.parent / "utp_motif_analysis" / "data" / "meme_gb.xml"
SEQS_FILE = SCRIPT_DIR.parent / "utp_motif_analysis" / "output" / "good-c-term-full.fasta"
EMBEDDINGS_FILE = SCRIPT_DIR / "output" / "features" / "embeddings_all.h5"
BIOCHEM_FILE = SCRIPT_DIR / "output" / "features" / "biochemical_features_all.csv"
OUTPUT_DIR = SCRIPT_DIR / "output" / "comparison"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_motif_data():
    """Load sequences and motif assignments."""
    meme_xml = ET.parse(MEME_FILE)
    seq_names = {
        tag.attrib["id"]: tag.attrib["name"]
        for tag in meme_xml.findall(".//sequence")
    }
    
    sites = meme_xml.findall(".//scanned_sites")
    sequences_motifs = {}
    for tag in sites:
        seq_id = tag.attrib["sequence_id"]
        if seq_id in seq_names:
            motifs = sorted(
                [s.attrib for s in tag.findall("scanned_site")],
                key=lambda m: int(m["position"])
            )
            sequences_motifs[seq_names[seq_id]] = tuple(
                m["motif_id"] for m in motifs
            )
    
    return sequences_motifs


def get_variant_name(motif_ids):
    """Convert motif IDs tuple to variant name."""
    return "+".join(m.replace("motif_", "") for m in motif_ids)


def get_terminal_group(variant_name):
    """Group variant by terminal motif."""
    if not variant_name:
        return "none"
    parts = variant_name.split("+")
    terminal = parts[-1] if parts else "none"
    
    # Map to main terminal groups
    if terminal in ["4"]:
        return "terminal_4"
    elif terminal in ["5"]:
        return "terminal_5"
    elif terminal in ["7"]:
        return "terminal_7"
    elif terminal in ["9"]:
        return "terminal_9"
    else:
        return f"terminal_{terminal}"


def load_features(protein_names):
    """Load pre-computed features for given proteins."""
    # Load embeddings - keys are protein names directly
    with h5py.File(EMBEDDINGS_FILE, "r") as f:
        emb_dict = {name: f[name][:] for name in f.keys()}
    
    # Load biochemical features (exclude non-numeric columns)
    biochem_df = pd.read_csv(BIOCHEM_FILE)
    numeric_cols = biochem_df.select_dtypes(include=[np.number]).columns.tolist()
    biochem_dict = {
        row["name"]: row[numeric_cols].values.astype(float)
        for _, row in biochem_df.iterrows()
    }
    
    # Combine for requested proteins
    X_list = []
    valid_names = []
    for name in protein_names:
        if name in emb_dict and name in biochem_dict:
            features = np.concatenate([emb_dict[name], biochem_dict[name]])
            X_list.append(features)
            valid_names.append(name)
    
    return np.array(X_list), valid_names


def get_available_proteins():
    """Get list of proteins that have pre-computed embeddings."""
    with h5py.File(EMBEDDINGS_FILE, "r") as f:
        return set(f.keys())


def run_classification(X, y, strategy_name):
    """Run cross-validated classification and return metrics."""
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    clf = LogisticRegression(
        C=0.01,
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )
    
    # Get CV predictions
    y_pred = cross_val_predict(clf, X_scaled, y_encoded, cv=cv)
    
    # Calculate metrics
    bal_acc = balanced_accuracy_score(y_encoded, y_pred)
    
    # Per-fold accuracy for std
    fold_accs = []
    for train_idx, test_idx in cv.split(X_scaled, y_encoded):
        clf_fold = LogisticRegression(
            C=0.01, class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
        )
        clf_fold.fit(X_scaled[train_idx], y_encoded[train_idx])
        fold_acc = balanced_accuracy_score(
            y_encoded[test_idx], clf_fold.predict(X_scaled[test_idx])
        )
        fold_accs.append(fold_acc)
    
    # Permutation test
    n_permutations = 500
    perm_scores = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y_encoded)
        y_pred_perm = cross_val_predict(clf, X_scaled, y_perm, cv=cv)
        perm_scores.append(balanced_accuracy_score(y_perm, y_pred_perm))
    
    perm_scores = np.array(perm_scores)
    p_value = (np.sum(perm_scores >= bal_acc) + 1) / (n_permutations + 1)
    
    # Class distribution
    class_counts = Counter(y)
    
    return {
        "strategy": strategy_name,
        "n_samples": len(y),
        "n_classes": len(classes),
        "classes": list(classes),
        "class_counts": dict(class_counts),
        "balanced_accuracy": bal_acc,
        "std": np.std(fold_accs),
        "p_value": p_value,
        "chance_level": 1.0 / len(classes),
        "y_true": y_encoded,
        "y_pred": y_pred,
        "label_encoder": le,
    }


def main():
    print("=" * 70)
    print("Comparing Variant Grouping Strategies")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load motif data
    print("\n[1/4] Loading motif data...")
    sequences_motifs = load_motif_data()
    print(f"  Loaded {len(sequences_motifs)} sequences with motif assignments")
    
    # Check which proteins have embeddings
    available_proteins = get_available_proteins()
    print(f"  Pre-computed embeddings available for {len(available_proteins)} proteins")
    
    # Filter to proteins with embeddings
    sequences_motifs = {
        k: v for k, v in sequences_motifs.items() if k in available_proteins
    }
    print(f"  Using {len(sequences_motifs)} proteins (intersection)")
    
    # Create variant assignments for each strategy
    print("\n[2/4] Creating variant assignments...")
    
    # Strategy 1: Terminal motif grouping
    terminal_variants = {}
    for name, motifs in sequences_motifs.items():
        variant = get_variant_name(motifs)
        if variant:  # Skip empty variants
            terminal_variants[name] = get_terminal_group(variant)
    
    # Filter to main terminal groups (4, 5, 7, 9)
    main_terminals = {"terminal_4", "terminal_5", "terminal_7", "terminal_9"}
    terminal_variants = {
        k: v for k, v in terminal_variants.items() if v in main_terminals
    }
    
    terminal_counts = Counter(terminal_variants.values())
    print(f"\n  Strategy 1 - Terminal Motif Grouping:")
    print(f"    {len(terminal_variants)} proteins across {len(terminal_counts)} classes")
    for v, c in sorted(terminal_counts.items(), key=lambda x: -x[1]):
        print(f"      {v}: {c}")
    
    # Strategy 2: Fine-grained with MIN_COUNT=5
    fine_variants = {}
    for name, motifs in sequences_motifs.items():
        variant = get_variant_name(motifs)
        if variant:
            fine_variants[name] = variant
    
    fine_counts = Counter(fine_variants.values())
    valid_fine_variants = {v for v, c in fine_counts.items() if c >= 5}
    fine_variants = {k: v for k, v in fine_variants.items() if v in valid_fine_variants}
    
    fine_counts_filtered = Counter(fine_variants.values())
    print(f"\n  Strategy 2 - Fine-Grained (MIN_COUNT=5):")
    print(f"    {len(fine_variants)} proteins across {len(fine_counts_filtered)} classes")
    for v, c in sorted(fine_counts_filtered.items(), key=lambda x: -x[1]):
        print(f"      {v}: {c}")
    
    # Load features
    print("\n[3/4] Loading features and running classification...")
    
    results = []
    
    # Run Strategy 1
    print("\n  Running Strategy 1 (Terminal Motif)...")
    X1, names1 = load_features(list(terminal_variants.keys()))
    y1 = [terminal_variants[n] for n in names1]
    result1 = run_classification(X1, y1, "Terminal Motif")
    results.append(result1)
    print(f"    Balanced Accuracy: {result1['balanced_accuracy']:.3f} ± {result1['std']:.3f}")
    print(f"    Permutation p-value: {result1['p_value']:.4f}")
    print(f"    Chance level: {result1['chance_level']:.3f}")
    
    # Run Strategy 2
    print("\n  Running Strategy 2 (Fine-Grained MIN_COUNT=5)...")
    X2, names2 = load_features(list(fine_variants.keys()))
    y2 = [fine_variants[n] for n in names2]
    result2 = run_classification(X2, y2, "Fine-Grained (MIN≥5)")
    results.append(result2)
    print(f"    Balanced Accuracy: {result2['balanced_accuracy']:.3f} ± {result2['std']:.3f}")
    print(f"    Permutation p-value: {result2['p_value']:.4f}")
    print(f"    Chance level: {result2['chance_level']:.3f}")
    
    # Also run original strategy for comparison
    print("\n  Running Original Strategy (MIN_COUNT=10)...")
    orig_variants = {}
    for name, motifs in sequences_motifs.items():
        variant = get_variant_name(motifs)
        if variant:
            orig_variants[name] = variant
    
    orig_counts = Counter(orig_variants.values())
    valid_orig = {v for v, c in orig_counts.items() if c >= 10}
    orig_variants = {k: v for k, v in orig_variants.items() if v in valid_orig}
    
    X0, names0 = load_features(list(orig_variants.keys()))
    y0 = [orig_variants[n] for n in names0]
    result0 = run_classification(X0, y0, "Original (MIN≥10)")
    results.append(result0)
    print(f"    Balanced Accuracy: {result0['balanced_accuracy']:.3f} ± {result0['std']:.3f}")
    print(f"    Permutation p-value: {result0['p_value']:.4f}")
    print(f"    Chance level: {result0['chance_level']:.3f}")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Strategy':<25} {'N':<6} {'Classes':<8} {'Acc':<8} {'Chance':<8} {'vs Chance':<10} {'p-value':<8}")
    print("-" * 80)
    for r in results:
        vs_chance = r['balanced_accuracy'] / r['chance_level']
        sig = "**" if r['p_value'] < 0.05 else "*" if r['p_value'] < 0.1 else ""
        print(f"{r['strategy']:<25} {r['n_samples']:<6} {r['n_classes']:<8} "
              f"{r['balanced_accuracy']:.3f}    {r['chance_level']:.3f}    "
              f"{vs_chance:.2f}×       {r['p_value']:.4f} {sig}")
    
    # Generate comparison plot
    print("\n[4/4] Generating comparison plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Accuracy comparison
    ax = axes[0]
    strategies = [r['strategy'] for r in results]
    accs = [r['balanced_accuracy'] for r in results]
    stds = [r['std'] for r in results]
    chances = [r['chance_level'] for r in results]
    
    x = np.arange(len(strategies))
    bars = ax.bar(x, accs, yerr=stds, capsize=5, color=['#2ecc71', '#3498db', '#9b59b6'])
    ax.scatter(x, chances, marker='_', s=200, color='red', zorder=5, label='Chance level')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15, ha='right')
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Classification Performance by Strategy')
    ax.legend()
    ax.set_ylim(0, 0.6)
    
    # Plot 2: Sample size vs performance
    ax = axes[1]
    ns = [r['n_samples'] for r in results]
    vs_chances = [r['balanced_accuracy'] / r['chance_level'] for r in results]
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    for i, (n, v, s, c) in enumerate(zip(ns, vs_chances, strategies, colors)):
        ax.scatter(n, v, s=200, c=c, label=s, zorder=5)
    ax.axhline(y=1.0, color='red', linestyle='--', label='Chance (1.0×)')
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Performance vs Chance (×)')
    ax.set_title('Sample Size vs Performance')
    ax.legend()
    
    # Plot 3: Class distribution comparison
    ax = axes[2]
    for i, r in enumerate(results):
        counts = list(r['class_counts'].values())
        ax.bar(
            [i + j*0.2 for j in range(len(counts))],
            counts,
            width=0.18,
            label=r['strategy'] if i == 0 else None,
            alpha=0.7
        )
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Samples per Class')
    ax.set_title('Class Balance by Strategy')
    ax.set_xticks([0.3, 1.3, 2.3])
    ax.set_xticklabels([r['strategy'].split()[0] for r in results])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "strategy_comparison.svg", dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "strategy_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison plot to {OUTPUT_DIR / 'strategy_comparison.svg'}")
    
    # Save results
    results_df = pd.DataFrame([
        {
            'strategy': r['strategy'],
            'n_samples': r['n_samples'],
            'n_classes': r['n_classes'],
            'balanced_accuracy': r['balanced_accuracy'],
            'std': r['std'],
            'chance_level': r['chance_level'],
            'vs_chance': r['balanced_accuracy'] / r['chance_level'],
            'p_value': r['p_value'],
        }
        for r in results
    ])
    results_df.to_csv(OUTPUT_DIR / "strategy_comparison.csv", index=False)
    print(f"  Saved results to {OUTPUT_DIR / 'strategy_comparison.csv'}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    best = max(results, key=lambda r: r['balanced_accuracy'] / r['chance_level'])
    print(f"\n  Best strategy: {best['strategy']}")
    print(f"    - {best['balanced_accuracy']/best['chance_level']:.2f}× better than chance")
    print(f"    - p = {best['p_value']:.4f}")
    
    if best['p_value'] < 0.05:
        print("\n  ✓ SIGNIFICANT: This strategy shows statistically significant classification")
    elif best['p_value'] < 0.1:
        print("\n  ~ BORDERLINE: This strategy shows borderline significance")
    else:
        print("\n  ✗ NOT SIGNIFICANT: No strategy achieves significance (may need more data)")


if __name__ == "__main__":
    main()
