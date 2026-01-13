#!/usr/bin/env python3
"""
Test Interpretation B: Functional Constraint

Hypothesis: The mature domain has biophysical features that determine which import
channel it uses. Different channels recognize different uTP variants.

Tests:
1. Do biophysical properties differ between variant groups?
2. Can biophysical features predict variant group (controlling for sequence similarity)?
3. Do biophysically similar proteins have the same variant (independent of homology)?
"""

from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
VARIANT_ASSIGNMENTS = Path(__file__).parent / "output/variant_assignments.csv"
IMPORT_FASTA = PROJECT_ROOT / "data/Import_candidates.fasta"
ANNOTATIONS = PROJECT_ROOT / "data/Bbigelowii_transcriptome_annotations.csv"
OUTPUT_DIR = Path(__file__).parent / "output"


def load_sequences(fasta_path: Path) -> dict:
    """Load sequences from FASTA file."""
    sequences = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences


def compute_biophysical_features(seq: str) -> dict:
    """Compute biophysical properties of a protein sequence."""
    # Clean sequence
    seq = seq.upper().replace("X", "A").replace("U", "C").replace("*", "")
    
    if len(seq) < 10:
        return None
    
    try:
        analysis = ProteinAnalysis(seq)
        
        features = {
            "length": len(seq),
            "molecular_weight": analysis.molecular_weight(),
            "isoelectric_point": analysis.isoelectric_point(),
            "gravy": analysis.gravy(),  # Hydrophobicity
            "instability_index": analysis.instability_index(),
            "aromaticity": analysis.aromaticity(),
        }
        
        # Secondary structure fractions
        helix, turn, sheet = analysis.secondary_structure_fraction()
        features["helix_fraction"] = helix
        features["turn_fraction"] = turn
        features["sheet_fraction"] = sheet
        
        # Amino acid composition (grouped by property)
        aa_percent = analysis.get_amino_acids_percent()
        
        # Charged residues
        features["positive_charge"] = sum(
            aa_percent.get(aa, 0) for aa in ["R", "K", "H"]
        )
        features["negative_charge"] = sum(
            aa_percent.get(aa, 0) for aa in ["D", "E"]
        )
        features["net_charge"] = features["positive_charge"] - features["negative_charge"]
        
        # Hydrophobic residues
        features["hydrophobic"] = sum(
            aa_percent.get(aa, 0) for aa in ["A", "V", "L", "I", "M", "F", "W"]
        )
        
        # Polar residues
        features["polar"] = sum(
            aa_percent.get(aa, 0) for aa in ["S", "T", "N", "Q", "Y", "C"]
        )
        
        # Special residues
        features["proline"] = aa_percent.get("P", 0)
        features["glycine"] = aa_percent.get("G", 0)
        features["cysteine"] = aa_percent.get("C", 0)
        
        return features
        
    except Exception as e:
        return None


def compute_kmer_similarity(seq1: str, seq2: str, k: int = 4) -> float:
    """Compute k-mer Jaccard similarity."""
    def get_kmers(seq):
        return set(seq[i : i + k] for i in range(len(seq) - k + 1))
    
    kmers1 = get_kmers(seq1.upper())
    kmers2 = get_kmers(seq2.upper())
    
    if not kmers1 or not kmers2:
        return 0.0
    
    return len(kmers1 & kmers2) / len(kmers1 | kmers2)


def test_biophysical_differences(
    features_df: pd.DataFrame, assignments: dict[str, str]
) -> dict:
    """
    Test 1: Do biophysical properties differ between variant groups?
    
    Use Kruskal-Wallis test (non-parametric ANOVA) for each property.
    """
    
    # Add variant group to features
    features_df["variant"] = features_df["sequence_id"].map(assignments)
    features_df = features_df.dropna(subset=["variant"])
    
    # Get biophysical columns
    biophys_cols = [
        c for c in features_df.columns 
        if c not in ["sequence_id", "variant"]
    ]
    
    results = []
    for col in biophys_cols:
        groups = [
            features_df[features_df["variant"] == v][col].dropna().values
            for v in features_df["variant"].unique()
        ]
        groups = [g for g in groups if len(g) > 5]  # Min group size
        
        if len(groups) < 2:
            continue
            
        # Kruskal-Wallis test
        stat, pvalue = stats.kruskal(*groups)
        
        # Effect size (eta-squared approximation)
        n = sum(len(g) for g in groups)
        k = len(groups)
        eta_sq = (stat - k + 1) / (n - k) if n > k else 0
        
        results.append({
            "feature": col,
            "kruskal_statistic": stat,
            "pvalue": pvalue,
            "eta_squared": eta_sq,
        })
    
    results_df = pd.DataFrame(results)
    results_df["pvalue_adjusted"] = results_df["pvalue"] * len(results_df)  # Bonferroni
    results_df = results_df.sort_values("pvalue")
    
    return {
        "results": results_df,
        "n_significant": (results_df["pvalue_adjusted"] < 0.05).sum(),
        "features_df": features_df,
    }


def test_biophysical_prediction(
    features_df: pd.DataFrame, assignments: dict[str, str]
) -> dict:
    """
    Test 2: Can biophysical features predict variant group?
    
    Train classifier on biophysical features only (no sequence information).
    """
    
    # Prepare data
    features_df = features_df.copy()
    features_df["variant"] = features_df["sequence_id"].map(assignments)
    features_df = features_df.dropna(subset=["variant"])
    
    # Get biophysical columns
    biophys_cols = [
        c for c in features_df.columns 
        if c not in ["sequence_id", "variant"]
    ]
    
    X = features_df[biophys_cols].values
    y = features_df["variant"].values
    
    # Handle NaN
    X = np.nan_to_num(X, nan=0)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_scores = cross_val_score(lr, X_scaled, y_encoded, cv=cv, scoring="accuracy")
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_scores = cross_val_score(rf, X_scaled, y_encoded, cv=cv, scoring="accuracy")
    
    # Baseline (random guessing)
    class_counts = Counter(y_encoded)
    baseline = max(class_counts.values()) / len(y_encoded)
    
    # Feature importance from RF
    rf.fit(X_scaled, y_encoded)
    importances = pd.DataFrame({
        "feature": biophys_cols,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    
    return {
        "lr_accuracy": lr_scores.mean(),
        "lr_std": lr_scores.std(),
        "rf_accuracy": rf_scores.mean(),
        "rf_std": rf_scores.std(),
        "baseline": baseline,
        "feature_importances": importances,
        "n_samples": len(y),
        "n_classes": len(le.classes_),
        "classes": le.classes_,
    }


def test_biophysical_similarity_vs_variant(
    features_df: pd.DataFrame, 
    sequences: dict,
    assignments: dict[str, str],
    n_pairs: int = 5000
) -> dict:
    """
    Test 3: Are biophysically similar proteins more likely to have same variant,
    controlling for sequence similarity?
    
    Compare pairs with:
    - High biophysical similarity but low sequence similarity
    - Do these still have same variant more often than chance?
    """
    
    # Prepare data
    features_df = features_df.copy()
    features_df["variant"] = features_df["sequence_id"].map(assignments)
    features_df = features_df.dropna(subset=["variant"])
    
    # Get biophysical columns
    biophys_cols = [
        c for c in features_df.columns 
        if c not in ["sequence_id", "variant"]
    ]
    
    # Scale features for similarity computation
    X = features_df[biophys_cols].values
    X = np.nan_to_num(X, nan=0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    seq_ids = features_df["sequence_id"].values
    variants = features_df["variant"].values
    
    # Sample random pairs
    np.random.seed(42)
    n = len(seq_ids)
    pairs_i = np.random.randint(0, n, n_pairs)
    pairs_j = np.random.randint(0, n, n_pairs)
    
    # Filter out self-pairs
    valid = pairs_i != pairs_j
    pairs_i = pairs_i[valid]
    pairs_j = pairs_j[valid]
    
    # Compute similarities
    results = []
    for i, j in zip(pairs_i[:n_pairs], pairs_j[:n_pairs]):
        seq_i, seq_j = seq_ids[i], seq_ids[j]
        
        # Biophysical similarity (cosine)
        biophys_sim = np.dot(X_scaled[i], X_scaled[j]) / (
            np.linalg.norm(X_scaled[i]) * np.linalg.norm(X_scaled[j]) + 1e-8
        )
        
        # Sequence similarity
        if seq_i in sequences and seq_j in sequences:
            seq_sim = compute_kmer_similarity(sequences[seq_i], sequences[seq_j])
        else:
            seq_sim = 0
        
        # Same variant?
        same_variant = variants[i] == variants[j]
        
        results.append({
            "biophys_sim": biophys_sim,
            "seq_sim": seq_sim,
            "same_variant": same_variant,
        })
    
    results_df = pd.DataFrame(results)
    
    # Stratify by sequence similarity
    # "Non-homologous" pairs: low sequence similarity
    low_seq_sim = results_df[results_df["seq_sim"] < 0.1]
    
    # Among non-homologs, does biophysical similarity predict same variant?
    high_biophys = low_seq_sim[low_seq_sim["biophys_sim"] > 0.5]
    low_biophys = low_seq_sim[low_seq_sim["biophys_sim"] < 0.0]
    
    high_biophys_same = high_biophys["same_variant"].mean() if len(high_biophys) > 10 else np.nan
    low_biophys_same = low_biophys["same_variant"].mean() if len(low_biophys) > 10 else np.nan
    
    # Logistic regression: does biophys_sim predict same_variant controlling for seq_sim?
    from sklearn.linear_model import LogisticRegression
    
    X_reg = results_df[["biophys_sim", "seq_sim"]].values
    y_reg = results_df["same_variant"].astype(int).values
    
    lr = LogisticRegression()
    lr.fit(X_reg, y_reg)
    
    biophys_coef = lr.coef_[0][0]
    seq_coef = lr.coef_[0][1]
    
    return {
        "n_pairs": len(results_df),
        "n_low_seq_sim": len(low_seq_sim),
        "high_biophys_same_variant_rate": high_biophys_same,
        "low_biophys_same_variant_rate": low_biophys_same,
        "n_high_biophys_pairs": len(high_biophys),
        "n_low_biophys_pairs": len(low_biophys),
        "biophys_coef": biophys_coef,
        "seq_coef": seq_coef,
        "results_df": results_df,
    }


def visualize_results(
    diff_results: dict,
    pred_results: dict,
    sim_results: dict,
    output_dir: Path,
):
    """Create visualizations of test results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Significant biophysical differences
    ax = axes[0, 0]
    results_df = diff_results["results"].head(10)
    colors = ["green" if p < 0.05 else "gray" for p in results_df["pvalue_adjusted"]]
    ax.barh(results_df["feature"], -np.log10(results_df["pvalue"]), color=colors)
    ax.axvline(-np.log10(0.05 / len(diff_results["results"])), color="red", 
               linestyle="--", label="Bonferroni threshold")
    ax.set_xlabel("-log10(p-value)")
    ax.set_title(f"Biophysical Differences by Variant\n"
                 f"({diff_results['n_significant']} significant after correction)")
    ax.legend()
    
    # Plot 2: Classification accuracy
    ax = axes[0, 1]
    methods = ["Baseline\n(majority)", "Logistic\nRegression", "Random\nForest"]
    accs = [pred_results["baseline"], pred_results["lr_accuracy"], pred_results["rf_accuracy"]]
    stds = [0, pred_results["lr_std"], pred_results["rf_std"]]
    colors = ["gray", "steelblue", "darkgreen"]
    
    bars = ax.bar(methods, accs, yerr=stds, color=colors, capsize=5)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Biophysical Features → Variant Prediction\n"
                 f"(n={pred_results['n_samples']}, {pred_results['n_classes']} classes)")
    ax.set_ylim(0, 1)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, acc + 0.05, f"{acc:.2f}", 
                ha="center", fontsize=10)
    
    # Plot 3: Feature importance
    ax = axes[1, 0]
    importances = pred_results["feature_importances"].head(10)
    ax.barh(importances["feature"], importances["importance"], color="darkgreen")
    ax.set_xlabel("Feature Importance (Random Forest)")
    ax.set_title("Top Predictive Biophysical Features")
    
    # Plot 4: Summary
    ax = axes[1, 1]
    ax.axis("off")
    
    pred_above_baseline = pred_results["rf_accuracy"] > pred_results["baseline"] + 0.05
    biophys_predictive = sim_results["biophys_coef"] > 0
    
    summary_text = f"""
INTERPRETATION B: Functional Constraint Test Results
=====================================================

Test 1: Biophysical Differences Between Variants
  Significant features: {diff_results['n_significant']} / {len(diff_results['results'])}
  Top feature:          {diff_results['results'].iloc[0]['feature']}
  
Test 2: Biophysical → Variant Prediction
  Baseline accuracy:    {pred_results['baseline']:.3f}
  Logistic Regression:  {pred_results['lr_accuracy']:.3f} ± {pred_results['lr_std']:.3f}
  Random Forest:        {pred_results['rf_accuracy']:.3f} ± {pred_results['rf_std']:.3f}
  Above baseline:       {'+' if pred_above_baseline else ''}{(pred_results['rf_accuracy'] - pred_results['baseline'])*100:.1f}%

Test 3: Biophysical Similarity vs Same Variant
  (Controlling for sequence similarity)
  Non-homologous pairs: {sim_results['n_low_seq_sim']}
  Biophys coefficient:  {sim_results['biophys_coef']:.3f}
  Seq coefficient:      {sim_results['seq_coef']:.3f}
  
CONCLUSION:
{"SUPPORTS Interpretation B" if pred_above_baseline and diff_results['n_significant'] > 0 else "DOES NOT SUPPORT Interpretation B"}
Biophysical features {"CAN" if pred_above_baseline else "CANNOT"} predict variant group.
"""
    ax.text(
        0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", fontfamily="monospace"
    )
    
    plt.tight_layout()
    plt.savefig(output_dir / "biophysical_test_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Additional plot: Biophysical distributions by variant
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    features_df = diff_results["features_df"]
    top_features = diff_results["results"].head(6)["feature"].tolist()
    
    for ax, feature in zip(axes, top_features):
        for variant in features_df["variant"].unique():
            data = features_df[features_df["variant"] == variant][feature].dropna()
            ax.hist(data, bins=30, alpha=0.5, label=variant, density=True)
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    
    plt.suptitle("Top Discriminative Biophysical Features by Variant", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "biophysical_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Test Interpretation B: Functional Constraint")
    print("=" * 70)
    
    # Load variant assignments
    print("\n[1/6] Loading data...")
    if not VARIANT_ASSIGNMENTS.exists():
        print(f"  ERROR: Run 01_define_groups.py first")
        return
    
    assignments_df = pd.read_csv(VARIANT_ASSIGNMENTS)
    assignments = dict(
        zip(assignments_df["sequence_id"], assignments_df["variant_group"])
    )
    print(f"  Loaded {len(assignments)} variant assignments")
    
    # Load sequences
    sequences = load_sequences(IMPORT_FASTA)
    sequences = {k: v for k, v in sequences.items() if k in assignments}
    print(f"  Loaded {len(sequences)} sequences")
    
    # Compute biophysical features
    print("\n[2/6] Computing biophysical features...")
    features = []
    for seq_id, seq in sequences.items():
        feat = compute_biophysical_features(seq)
        if feat:
            feat["sequence_id"] = seq_id
            features.append(feat)
    
    features_df = pd.DataFrame(features)
    print(f"  Computed features for {len(features_df)} sequences")
    features_df.to_csv(OUTPUT_DIR / "biophysical_features.csv", index=False)
    
    # Test 1: Biophysical differences
    print("\n[3/6] Test 1: Biophysical differences between variants...")
    diff_results = test_biophysical_differences(features_df, assignments)
    print(f"  Significant features: {diff_results['n_significant']}")
    print("  Top 5 features:")
    print(diff_results["results"].head()[["feature", "pvalue", "eta_squared"]].to_string(index=False))
    
    # Test 2: Biophysical prediction
    print("\n[4/6] Test 2: Biophysical feature prediction...")
    pred_results = test_biophysical_prediction(features_df, assignments)
    print(f"  Baseline accuracy:   {pred_results['baseline']:.3f}")
    print(f"  Logistic Regression: {pred_results['lr_accuracy']:.3f} ± {pred_results['lr_std']:.3f}")
    print(f"  Random Forest:       {pred_results['rf_accuracy']:.3f} ± {pred_results['rf_std']:.3f}")
    
    # Test 3: Biophysical similarity controlling for sequence
    print("\n[5/6] Test 3: Biophysical similarity vs variant (controlling for sequence)...")
    sim_results = test_biophysical_similarity_vs_variant(features_df, sequences, assignments)
    print(f"  Non-homologous pairs analyzed: {sim_results['n_low_seq_sim']}")
    print(f"  Biophysical coefficient: {sim_results['biophys_coef']:.3f}")
    print(f"  Sequence coefficient:    {sim_results['seq_coef']:.3f}")
    
    # Visualize
    print("\n[6/6] Creating visualizations...")
    visualize_results(diff_results, pred_results, sim_results, OUTPUT_DIR)
    print(f"  Saved visualizations to {OUTPUT_DIR}")
    
    # Save summary
    results_summary = {
        "test": "Interpretation B - Functional Constraint",
        "n_significant_features": diff_results["n_significant"],
        "baseline_accuracy": pred_results["baseline"],
        "lr_accuracy": pred_results["lr_accuracy"],
        "rf_accuracy": pred_results["rf_accuracy"],
        "accuracy_above_baseline": pred_results["rf_accuracy"] - pred_results["baseline"],
        "biophys_coefficient": sim_results["biophys_coef"],
        "seq_coefficient": sim_results["seq_coef"],
    }
    pd.DataFrame([results_summary]).to_csv(
        OUTPUT_DIR / "biophysical_test_summary.csv", index=False
    )
    
    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    pred_above_baseline = pred_results["rf_accuracy"] > pred_results["baseline"] + 0.05
    has_diff = diff_results["n_significant"] > 0
    
    if pred_above_baseline and has_diff:
        print("\n  ✓ SUPPORTS Interpretation B (Functional Constraint)")
        print("    Biophysical properties differ between variant groups.")
        print("    These properties can predict variant membership.")
    else:
        print("\n  ✗ DOES NOT SUPPORT Interpretation B")
        print("    Biophysical properties do not strongly predict variant.")
    
    print(f"\n📁 Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
