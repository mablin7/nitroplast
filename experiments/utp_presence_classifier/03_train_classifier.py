#!/usr/bin/env python3
"""
Step 3: Train uTP Presence Classifier

This script trains binary classifiers to predict whether a protein should have
a uTP based on its mature domain sequence using ProtT5 embeddings.

Runs TWO experiments:
1. Downsampled: 117 uTP vs 117 cytoplasmic controls (multiple folds)
2. Full dataset: All uTP vs nuclear+cytoplasmic controls

Embeddings are computed ONCE for all unique sequences and reused.

Prerequisites:
1. Run 01_prepare_data.py to extract mature domains and control candidates
2. Upload control_candidates.fasta to CELLO web service
3. Run 02_filter_controls.py to filter by localization

Usage:
    uv run python experiments/utp_presence_classifier/03_train_classifier.py
"""

import gc
import re
import warnings
from pathlib import Path

import h5py
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from Bio import SeqIO
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

TEST_SIZE = 0.2
N_PERMUTATIONS = 500
N_CV_FOLDS = 5
N_DOWNSAMPLE_FOLDS = 10  # Number of random downsample iterations
RANDOM_SEED = 42

PROTT5_MODEL = "Rostlab/prot_t5_xl_uniref50"

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"

# Input files (from previous steps)
MATURE_DOMAINS_FILE = OUTPUT_DIR / "mature_domains.fasta"
CYTOPLASMIC_CONTROLS_FILE = OUTPUT_DIR / "filtered_controls_cytoplasmic.fasta"
NUCLEAR_CYTOPLASMIC_CONTROLS_FILE = (
    OUTPUT_DIR / "filtered_controls_nuclear_cytoplasmic.fasta"
)

# Output files
EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.h5"

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
# Embedding Generation
# =============================================================================


def load_prott5_model(device: str = "cpu"):
    """Load ProtT5 tokenizer and model."""
    from transformers import T5EncoderModel, T5Tokenizer

    print("  Loading ProtT5 model (this may take a while on first run)...")
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
    """Generate ProtT5 embeddings for a list of sequences."""
    embeddings = []

    for seq in tqdm(sequences, desc="Computing embeddings"):
        seq = re.sub(r"[UZOJB*]", "X", seq)
        seq_spaced = " ".join(seq)

        ids = tokenizer.batch_encode_plus(
            [seq_spaced], add_special_tokens=True, padding=True
        )
        input_ids = torch.tensor(ids["input_ids"]).to(device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)

        embedding = output.last_hidden_state.cpu().numpy()
        seq_len = (attention_mask[0] == 1).sum().item()
        seq_embedding = embedding[0][: seq_len - 1]

        mean_embedding = np.mean(seq_embedding, axis=0).astype(np.float32)
        embeddings.append(mean_embedding)

    return embeddings


def load_or_compute_embeddings(
    sequences: dict[str, str], embeddings_file: Path, device: str = "cpu"
) -> dict[str, np.ndarray]:
    """Load embeddings from file or compute them if not available."""
    existing_embeddings = {}
    missing_seqs = {}

    if embeddings_file.exists():
        print(f"  Loading existing embeddings from {embeddings_file}")
        with h5py.File(embeddings_file, "r") as f:
            for key in f.keys():
                existing_embeddings[key] = f[key][()]

        # Find missing sequences
        for name, seq in sequences.items():
            if name not in existing_embeddings:
                missing_seqs[name] = seq

        if not missing_seqs:
            print(f"  All {len(existing_embeddings)} embeddings found in cache")
            return existing_embeddings
        print(
            f"  Found {len(existing_embeddings)} cached, {len(missing_seqs)} need computing"
        )
    else:
        missing_seqs = sequences

    if missing_seqs:
        print(f"  Computing ProtT5 embeddings for {len(missing_seqs)} sequences...")
        tokenizer, model = load_prott5_model(device)

        seq_names = list(missing_seqs.keys())
        seq_list = [missing_seqs[n] for n in seq_names]

        embedding_list = embed_sequences(seq_list, tokenizer, model, device)

        # Merge with existing and save all
        all_embeddings = dict(existing_embeddings)
        for name, emb in zip(seq_names, embedding_list):
            all_embeddings[name] = emb

        with h5py.File(embeddings_file, "w") as f:
            for name, emb in all_embeddings.items():
                f.create_dataset(name, data=emb)

        print(f"  Saved {len(all_embeddings)} embeddings to {embeddings_file}")

        del tokenizer, model
        gc.collect()

        return all_embeddings

    return existing_embeddings


# =============================================================================
# Classification
# =============================================================================


def prepare_dataset(
    utp_embeddings: dict[str, np.ndarray], control_embeddings: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare dataset for binary classification."""
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


def train_single_classifier(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    clf_name: str = "Logistic Regression",
) -> dict:
    """Train a single classifier and return metrics."""
    if clf_name == "Logistic Regression":
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    elif clf_name == "SVC":
        clf = SVC(probability=True, random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Unknown classifier: {clf_name}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf.fit(X_train_scaled, Y_train)

    Y_pred = clf.predict(X_test_scaled)
    Y_prob = (
        clf.predict_proba(X_test_scaled)[:, 1]
        if hasattr(clf, "predict_proba")
        else None
    )

    return {
        "accuracy": accuracy_score(Y_test, Y_pred),
        "precision": precision_score(Y_test, Y_pred, zero_division=0),
        "recall": recall_score(Y_test, Y_pred, zero_division=0),
        "f1_score": f1_score(Y_test, Y_pred, zero_division=0),
        "Y_test": Y_test,
        "Y_pred": Y_pred,
        "Y_prob": Y_prob,
        "classifier": clf,
        "scaler": scaler,
    }


def run_downsampled_experiment(
    utp_embeddings: dict[str, np.ndarray],
    control_embeddings: dict[str, np.ndarray],
    n_folds: int = N_DOWNSAMPLE_FOLDS,
) -> dict:
    """
    Run experiment with downsampled positive class.

    Multiple random downsample folds to get robust estimates.
    """
    n_controls = len(control_embeddings)
    utp_names = list(utp_embeddings.keys())
    control_names = list(control_embeddings.keys())

    print(f"\n  Running {n_folds} downsample folds (n={n_controls} per class)...")

    all_results = {
        "Logistic Regression": [],
        "Random Forest": [],
        "SVC": [],
    }

    rng = np.random.RandomState(RANDOM_SEED)

    for fold in range(n_folds):
        # Randomly sample uTP proteins to match control count
        sampled_utp_names = rng.choice(utp_names, size=n_controls, replace=False)

        # Prepare data for this fold
        X = []
        Y = []
        for name in sampled_utp_names:
            X.append(utp_embeddings[name])
            Y.append(1)
        for name in control_names:
            X.append(control_embeddings[name])
            Y.append(0)

        X = np.array(X)
        Y = np.array(Y)

        # Split
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED + fold, stratify=Y
        )

        # Train each classifier
        for clf_name in all_results.keys():
            result = train_single_classifier(X_train, Y_train, X_test, Y_test, clf_name)
            all_results[clf_name].append(result)

    # Aggregate results
    aggregated = {}
    for clf_name, fold_results in all_results.items():
        accuracies = [r["accuracy"] for r in fold_results]
        precisions = [r["precision"] for r in fold_results]
        recalls = [r["recall"] for r in fold_results]
        f1s = [r["f1_score"] for r in fold_results]

        aggregated[clf_name] = {
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "precision_mean": np.mean(precisions),
            "precision_std": np.std(precisions),
            "recall_mean": np.mean(recalls),
            "recall_std": np.std(recalls),
            "f1_mean": np.mean(f1s),
            "f1_std": np.std(f1s),
            "all_accuracies": accuracies,
            "fold_results": fold_results,
        }

        print(f"    {clf_name}: {np.mean(accuracies):.1%} ± {np.std(accuracies):.1%}")

    return aggregated


def run_full_experiment(
    utp_embeddings: dict[str, np.ndarray],
    control_embeddings: dict[str, np.ndarray],
    n_permutations: int = N_PERMUTATIONS,
) -> dict:
    """
    Run experiment with full dataset (potentially imbalanced).

    Includes permutation test for statistical significance.
    """
    print(f"\n  Dataset: {len(utp_embeddings)} uTP, {len(control_embeddings)} controls")

    X, Y, names = prepare_dataset(utp_embeddings, control_embeddings)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

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
    best_name = None

    for name, clf in classifiers.items():
        print(f"\n  Training {name}...")

        clf.fit(X_train, Y_train)

        Y_pred = clf.predict(X_test)
        Y_prob = (
            clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
        )

        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred, zero_division=0)
        recall = recall_score(Y_test, Y_pred, zero_division=0)
        f1 = f1_score(Y_test, Y_pred, zero_division=0)

        cv_scores = cross_val_score(clf, X_scaled, Y, cv=N_CV_FOLDS, scoring="accuracy")

        n = len(Y_test)
        # For imbalanced data, compare to class prior not 0.5
        majority_prior = max(np.mean(Y), 1 - np.mean(Y))
        binom_pval = binomtest(
            round(accuracy * n), n, majority_prior, alternative="greater"
        ).pvalue

        print(f"    Running permutation test ({n_permutations} permutations)...")
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

        print(f"    Accuracy: {accuracy:.1%}")
        print(f"    Precision: {precision:.1%}")
        print(f"    Recall: {recall:.1%}")
        print(f"    F1 Score: {f1:.2f}")
        print(f"    CV Score: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
        print(f"    Permutation p-value: {perm_pval:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_name = name

    results["_best"] = {
        "name": best_name,
        "classifier": classifiers[best_name],
        "scaler": scaler,
    }

    return results


# =============================================================================
# Visualization
# =============================================================================


def plot_length_distribution(
    utp_lengths: list[int],
    control_lengths: list[int],
    output_file: Path,
    title_suffix: str = "",
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

    ks_stat, ks_pval = ks_2samp(utp_lengths, control_lengths)

    ax.set_xlabel("Mature domain length (aa)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Length Distribution Comparison{title_suffix}\nKS test: D={ks_stat:.3f}, p={ks_pval:.3f}"
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved length distribution to {output_file}")


def plot_roc_curve(results: dict, output_file: Path, title_suffix: str = ""):
    """Plot ROC curve for the best classifier."""
    best_name = results["_best"]["name"]
    Y_test = results[best_name]["Y_test"]
    Y_prob = results[best_name]["Y_prob"]

    if Y_prob is None:
        print(
            "  Classifier doesn't support probability predictions, skipping ROC curve"
        )
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
    ax.set_title(f"ROC Curve - uTP Presence Prediction{title_suffix}")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved ROC curve to {output_file}")


def plot_confusion_matrix(results: dict, output_file: Path, title_suffix: str = ""):
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
    ax.set_title(f"Confusion Matrix - {best_name}{title_suffix}")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved confusion matrix to {output_file}")


def plot_downsample_fold_results(results: dict, output_file: Path):
    """Plot accuracy distribution across downsample folds."""
    fig, ax = plt.subplots(figsize=(10, 6))

    data = []
    for clf_name, r in results.items():
        for acc in r["all_accuracies"]:
            data.append({"Classifier": clf_name, "Accuracy": acc})

    df = pd.DataFrame(data)

    sns.boxplot(data=df, x="Classifier", y="Accuracy", ax=ax, palette="Set2")
    sns.stripplot(
        data=df, x="Classifier", y="Accuracy", ax=ax, color="black", alpha=0.5, size=4
    )

    ax.axhline(0.5, color="red", linestyle="--", label="Random baseline")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Classifier Accuracy Across {N_DOWNSAMPLE_FOLDS} Downsample Folds")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved fold results to {output_file}")


def save_results_summary(exp1_results: dict, exp2_results: dict, output_dir: Path):
    """Save combined results summary."""
    rows = []

    # Experiment 1: Downsampled
    for clf_name, r in exp1_results.items():
        rows.append(
            {
                "Experiment": "Exp1: Downsampled (balanced)",
                "Classifier": clf_name,
                "Accuracy": f"{r['accuracy_mean']:.1%} ± {r['accuracy_std']:.1%}",
                "Precision": f"{r['precision_mean']:.1%} ± {r['precision_std']:.1%}",
                "Recall": f"{r['recall_mean']:.1%} ± {r['recall_std']:.1%}",
                "F1 Score": f"{r['f1_mean']:.2f} ± {r['f1_std']:.2f}",
                "CV/Perm p-value": f"n={N_DOWNSAMPLE_FOLDS} folds",
            }
        )

    # Experiment 2: Full dataset
    for clf_name, r in exp2_results.items():
        if clf_name.startswith("_"):
            continue
        rows.append(
            {
                "Experiment": "Exp2: Nuclear+Cytoplasmic",
                "Classifier": clf_name,
                "Accuracy": f"{r['accuracy']:.1%}",
                "Precision": f"{r['precision']:.1%}",
                "Recall": f"{r['recall']:.1%}",
                "F1 Score": f"{r['f1_score']:.2f}",
                "CV/Perm p-value": f"{r['permutation_pvalue']:.4f}",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "classifier_results_combined.csv", index=False)

    print(f"\n  Combined Results:")
    print(df.to_string(index=False))

    return df


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("Step 3: Train uTP Presence Classifier")
    print("=" * 70)

    # Check input files
    if not MATURE_DOMAINS_FILE.exists():
        raise FileNotFoundError(
            f"Mature domains file not found: {MATURE_DOMAINS_FILE}\n"
            "Run 01_prepare_data.py first."
        )

    if not CYTOPLASMIC_CONTROLS_FILE.exists():
        raise FileNotFoundError(
            f"Cytoplasmic controls file not found: {CYTOPLASMIC_CONTROLS_FILE}\n"
            "Run 02_filter_controls.py first."
        )

    if not NUCLEAR_CYTOPLASMIC_CONTROLS_FILE.exists():
        raise FileNotFoundError(
            f"Nuclear+Cytoplasmic controls file not found: {NUCLEAR_CYTOPLASMIC_CONTROLS_FILE}\n"
            "Run 02_filter_controls.py first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # =========================================================================
    # Load ALL sequences
    # =========================================================================
    print("\n[1/6] Loading sequences...")

    mature_domains = {
        record.id: str(record.seq)
        for record in SeqIO.parse(MATURE_DOMAINS_FILE, "fasta")
    }
    print(f"  Loaded {len(mature_domains)} mature domains")

    cytoplasmic_controls = {
        record.id: str(record.seq)
        for record in SeqIO.parse(CYTOPLASMIC_CONTROLS_FILE, "fasta")
    }
    print(f"  Loaded {len(cytoplasmic_controls)} cytoplasmic controls")

    nuclear_cytoplasmic_controls = {
        record.id: str(record.seq)
        for record in SeqIO.parse(NUCLEAR_CYTOPLASMIC_CONTROLS_FILE, "fasta")
    }
    print(f"  Loaded {len(nuclear_cytoplasmic_controls)} nuclear+cytoplasmic controls")

    # =========================================================================
    # Compute ALL embeddings ONCE
    # =========================================================================
    print("\n[2/6] Computing ProtT5 embeddings (shared across experiments)...")

    # Combine all sequences
    all_sequences = dict(mature_domains)
    all_sequences.update(cytoplasmic_controls)
    all_sequences.update(nuclear_cytoplasmic_controls)
    print(f"  Total unique sequences: {len(all_sequences)}")

    embeddings = load_or_compute_embeddings(all_sequences, EMBEDDINGS_FILE, device)

    # Split embeddings by source
    utp_embeddings = {
        name: embeddings[name] for name in mature_domains if name in embeddings
    }
    cytoplasmic_embeddings = {
        name: embeddings[name] for name in cytoplasmic_controls if name in embeddings
    }
    nuclear_cytoplasmic_embeddings = {
        name: embeddings[name]
        for name in nuclear_cytoplasmic_controls
        if name in embeddings
    }

    print(f"  uTP embeddings: {len(utp_embeddings)}")
    print(f"  Cytoplasmic embeddings: {len(cytoplasmic_embeddings)}")
    print(f"  Nuclear+Cytoplasmic embeddings: {len(nuclear_cytoplasmic_embeddings)}")

    # =========================================================================
    # Experiment 1: Downsampled (balanced)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[3/6] EXPERIMENT 1: Downsampled Positives (Balanced)")
    print("=" * 70)
    print(
        f"  {len(cytoplasmic_embeddings)} uTP vs {len(cytoplasmic_embeddings)} cytoplasmic controls"
    )
    print(f"  Running {N_DOWNSAMPLE_FOLDS} random downsample folds")

    # Length distribution for Exp 1
    utp_lengths = [len(mature_domains[n]) for n in utp_embeddings]
    cyto_lengths = [len(cytoplasmic_controls[n]) for n in cytoplasmic_embeddings]
    plot_length_distribution(
        utp_lengths,
        cyto_lengths,
        OUTPUT_DIR / "exp1_length_distribution.svg",
        " (Exp1: Cytoplasmic)",
    )

    exp1_results = run_downsampled_experiment(
        utp_embeddings, cytoplasmic_embeddings, n_folds=N_DOWNSAMPLE_FOLDS
    )

    plot_downsample_fold_results(exp1_results, OUTPUT_DIR / "exp1_fold_accuracies.svg")

    # =========================================================================
    # Experiment 2: Full dataset with Nuclear+Cytoplasmic controls
    # =========================================================================
    print("\n" + "=" * 70)
    print("[4/6] EXPERIMENT 2: Full Dataset (Nuclear+Cytoplasmic Controls)")
    print("=" * 70)
    print(
        f"  {len(utp_embeddings)} uTP vs {len(nuclear_cytoplasmic_embeddings)} nuclear+cytoplasmic controls"
    )

    # Length distribution for Exp 2
    nuc_cyto_lengths = [
        len(nuclear_cytoplasmic_controls[n]) for n in nuclear_cytoplasmic_embeddings
    ]
    plot_length_distribution(
        utp_lengths,
        nuc_cyto_lengths,
        OUTPUT_DIR / "exp2_length_distribution.svg",
        " (Exp2: Nuclear+Cytoplasmic)",
    )

    exp2_results = run_full_experiment(
        utp_embeddings, nuclear_cytoplasmic_embeddings, n_permutations=N_PERMUTATIONS
    )

    plot_roc_curve(exp2_results, OUTPUT_DIR / "exp2_roc_curve.svg", " (Exp2)")
    plot_confusion_matrix(
        exp2_results, OUTPUT_DIR / "exp2_confusion_matrix.svg", " (Exp2)"
    )

    # Save best classifier from Exp 2
    best_clf = exp2_results["_best"]["classifier"]
    best_scaler = exp2_results["_best"]["scaler"]
    joblib.dump(
        {"classifier": best_clf, "scaler": best_scaler},
        OUTPUT_DIR / "exp2_best_classifier.joblib",
    )
    print(f"  Saved Exp2 classifier to exp2_best_classifier.joblib")

    # =========================================================================
    # Combined Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("[5/6] Saving combined results")
    print("=" * 70)

    results_df = save_results_summary(exp1_results, exp2_results, OUTPUT_DIR)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("[6/6] Summary")
    print("=" * 70)

    # Exp 1 summary
    best_exp1_clf = max(
        exp1_results.keys(), key=lambda k: exp1_results[k]["accuracy_mean"]
    )
    best_exp1_acc = exp1_results[best_exp1_clf]["accuracy_mean"]
    best_exp1_std = exp1_results[best_exp1_clf]["accuracy_std"]

    print(f"\n  Experiment 1 (Downsampled, n={len(cytoplasmic_embeddings)} per class):")
    print(f"    Best: {best_exp1_clf}")
    print(f"    Accuracy: {best_exp1_acc:.1%} ± {best_exp1_std:.1%}")

    # Exp 2 summary
    best_exp2_name = exp2_results["_best"]["name"]
    best_exp2_acc = exp2_results[best_exp2_name]["accuracy"]
    best_exp2_pval = exp2_results[best_exp2_name]["permutation_pvalue"]

    print(
        f"\n  Experiment 2 (Full dataset, {len(utp_embeddings)} vs {len(nuclear_cytoplasmic_embeddings)}):"
    )
    print(f"    Best: {best_exp2_name}")
    print(f"    Accuracy: {best_exp2_acc:.1%}")
    print(f"    Permutation p-value: {best_exp2_pval:.4f}")

    # Interpretation
    print("\n" + "-" * 40)
    print("Interpretation:")
    print("-" * 40)

    if best_exp1_acc > 0.55:
        print(f"  Exp1: Classifier performs above chance ({best_exp1_acc:.1%} > 50%)")
        print("        Suggests uTP proteins share sequence features in mature domains")
    else:
        print(f"  Exp1: Classifier near chance level ({best_exp1_acc:.1%} ≈ 50%)")
        print(
            "        Suggests uTP proteins may not have distinctive sequence features"
        )

    if best_exp2_pval < 0.05:
        print(f"\n  Exp2: Statistically significant (p={best_exp2_pval:.4f} < 0.05)")
        print("        uTP-containing proteins can be distinguished from controls")
    else:
        print(
            f"\n  Exp2: Not statistically significant (p={best_exp2_pval:.4f} >= 0.05)"
        )
        print("        Cannot reliably distinguish uTP proteins from controls")


if __name__ == "__main__":
    main()
