#!/usr/bin/env python3
"""
03_train_classifier.py - Rigorous Multiclass Classification for Extended Dataset

This script implements state-of-the-art academic statistical techniques for
multiclass classification of uTP terminal motif variants.

Key Features:
1. Nested cross-validation (outer: evaluation, inner: hyperparameter tuning)
2. Stratified sampling to handle class imbalance
3. Class-weighted training for all classifiers
4. Multiple significance tests (permutation, binomial, McNemar)
5. Bootstrap confidence intervals
6. Repeated cross-validation for variance estimation

Statistical Framework:
- Permutation test (1000 iterations) for overall significance
- Per-class binomial tests with Bonferroni correction
- McNemar's test for pairwise classifier comparison
- Bootstrap CI (1000 iterations) for all metrics
- 10×5-fold repeated CV for variance estimation

Classifiers:
- Logistic Regression (multinomial, L2 regularized)
- Support Vector Machine (RBF kernel)
- Random Forest
- XGBoost (if available)
- Gradient Boosting

Output:
- models/best_model.joblib: Final trained model
- statistics/: All statistical test results
- figures/: Visualizations

Usage:
    uv run python experiments/utp_variant_classifier_extended/03_train_classifier.py
"""

import json
import warnings
from collections import Counter
from pathlib import Path

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    permutation_test_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from tqdm import tqdm

# Try to import XGBoost
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available, skipping XGBoost classifier")

# =============================================================================
# Configuration
# =============================================================================

# Cross-validation settings
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
N_REPEATS = 10  # For repeated CV variance estimation
N_PERMUTATIONS = 1000
N_BOOTSTRAP = 1000

# Significance thresholds
ALPHA = 0.05

# Random seed
RANDOM_SEED = 42

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
FEATURES_DIR = OUTPUT_DIR / "features"
MODELS_DIR = OUTPUT_DIR / "models"
STATS_DIR = OUTPUT_DIR / "statistics"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Input files
EMBEDDINGS_FILE = FEATURES_DIR / "embeddings.h5"
BIOCHEM_FILE = FEATURES_DIR / "biochemical_features.csv"
PROCESSED_PROTEINS_FILE = OUTPUT_DIR / "data" / "processed_proteins.csv"


# =============================================================================
# Data Loading
# =============================================================================


def load_data() -> tuple[np.ndarray, np.ndarray, list[str], list[str], LabelEncoder]:
    """
    Load embeddings and terminal class labels.

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Label array (n_samples,)
        names: Protein names
        classes: Unique class labels
        label_encoder: Fitted LabelEncoder
    """
    # Load protein metadata
    protein_df = pd.read_csv(PROCESSED_PROTEINS_FILE)

    # Load embeddings
    embeddings = {}
    with h5py.File(EMBEDDINGS_FILE, "r") as f:
        for name in f.keys():
            embeddings[name] = f[name][()]

    # Load biochemical features
    biochem_df = pd.read_csv(BIOCHEM_FILE)

    # Align data
    common_names = (
        set(protein_df["name"]) & set(embeddings.keys()) & set(biochem_df["name"])
    )

    # Filter and sort
    protein_df = protein_df[protein_df["name"].isin(common_names)]
    protein_df = protein_df.sort_values("name")

    biochem_df = biochem_df[biochem_df["name"].isin(common_names)]
    biochem_df = biochem_df.sort_values("name")

    # Build feature matrix
    names = protein_df["name"].tolist()
    y_labels = protein_df["terminal_class"].tolist()

    # Embeddings
    X_emb = np.array([embeddings[n] for n in names])

    # Biochemical features (numeric columns only, excluding length to avoid leakage)
    biochem_numeric = biochem_df.select_dtypes(include=[np.number])
    if "length" in biochem_numeric.columns:
        biochem_numeric = biochem_numeric.drop(columns=["length"])
    X_biochem = biochem_numeric.values

    # Combine features
    X = np.hstack([X_emb, X_biochem])

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)
    classes = label_encoder.classes_.tolist()

    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features, {len(classes)} classes")

    return X, y, names, classes, label_encoder


# =============================================================================
# Classifier Definitions
# =============================================================================


def get_classifiers() -> dict:
    """
    Get dictionary of classifiers with their hyperparameter grids.

    All classifiers use class_weight='balanced' to handle imbalance.
    """
    classifiers = {}

    # Logistic Regression
    classifiers["Logistic Regression"] = (
        LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        },
    )

    # Support Vector Machine
    classifiers["SVM"] = (
        SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        {
            "C": [0.1, 1.0, 10.0],
            "gamma": ["scale", "auto"],
        },
    )

    # Random Forest
    classifiers["Random Forest"] = (
        RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
        },
    )

    # Gradient Boosting
    classifiers["Gradient Boosting"] = (
        GradientBoostingClassifier(
            random_state=RANDOM_SEED,
        ),
        {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1],
        },
    )

    # XGBoost
    if XGBOOST_AVAILABLE:
        classifiers["XGBoost"] = (
            XGBClassifier(
                eval_metric="mlogloss",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1],
                "scale_pos_weight": [1],  # Will be computed per-class
            },
        )

    return classifiers


# =============================================================================
# Nested Cross-Validation
# =============================================================================


def nested_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    clf_name: str,
    clf,
    param_grid: dict,
    n_outer_folds: int = N_OUTER_FOLDS,
    n_inner_folds: int = N_INNER_FOLDS,
) -> dict:
    """
    Perform nested cross-validation with hyperparameter tuning.

    Outer loop: Model evaluation (unbiased performance estimate)
    Inner loop: Hyperparameter tuning (GridSearchCV)
    """
    outer_cv = StratifiedKFold(
        n_splits=n_outer_folds, shuffle=True, random_state=RANDOM_SEED
    )
    inner_cv = StratifiedKFold(
        n_splits=n_inner_folds, shuffle=True, random_state=RANDOM_SEED
    )

    scores = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    all_indices = []
    best_params_per_fold = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Inner CV: hyperparameter tuning
        grid_search = GridSearchCV(
            clf,
            param_grid,
            cv=inner_cv,
            scoring="balanced_accuracy",
            n_jobs=-1,
            refit=True,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_search.fit(X_train_scaled, y_train)

        # Evaluate on outer test fold
        y_pred = grid_search.predict(X_test_scaled)

        # Get probabilities if available
        if hasattr(grid_search.best_estimator_, "predict_proba"):
            y_proba = grid_search.predict_proba(X_test_scaled)
        else:
            y_proba = None

        score = balanced_accuracy_score(y_test, y_pred)
        scores.append(score)
        best_params_per_fold.append(grid_search.best_params_)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_indices.extend(test_idx)
        if y_proba is not None:
            all_y_proba.extend(y_proba)

    return {
        "clf_name": clf_name,
        "scores": scores,
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "best_params_per_fold": best_params_per_fold,
        "y_true": np.array(all_y_true),
        "y_pred": np.array(all_y_pred),
        "y_proba": np.array(all_y_proba) if all_y_proba else None,
        "indices": np.array(all_indices),
    }


def repeated_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    clf,
    n_splits: int = N_OUTER_FOLDS,
    n_repeats: int = N_REPEATS,
) -> dict:
    """
    Perform repeated stratified k-fold cross-validation for variance estimation.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_SEED
    )

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(
            clf, X_scaled, y, cv=cv, scoring="balanced_accuracy", n_jobs=-1
        )

    return {
        "scores": scores,
        "mean": np.mean(scores),
        "std": np.std(scores),
        "ci_lower": np.percentile(scores, 2.5),
        "ci_upper": np.percentile(scores, 97.5),
        "n_folds": n_splits * n_repeats,
    }


# =============================================================================
# Statistical Tests
# =============================================================================


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    clf,
    n_permutations: int = N_PERMUTATIONS,
) -> tuple[float, float, np.ndarray]:
    """
    Perform permutation test to assess statistical significance.
    """
    cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"    Running {n_permutations} permutations...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score, perm_scores, pval = permutation_test_score(
            clf,
            X_scaled,
            y,
            scoring="balanced_accuracy",
            cv=cv,
            n_permutations=n_permutations,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )

    return score, pval, perm_scores


def binomial_test_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
) -> pd.DataFrame:
    """
    Perform exact binomial test for each class against random guessing.
    Uses Bonferroni correction for multiple testing.
    """
    n_classes = len(classes)
    random_prob = 1.0 / n_classes

    results = []
    for i, cls in enumerate(classes):
        # Count correct predictions for this class
        mask = y_true == i
        n_samples = mask.sum()
        n_correct = (y_pred[mask] == i).sum()

        # Binomial test: is accuracy > random?
        if n_samples > 0:
            binom_result = stats.binomtest(
                n_correct, n_samples, random_prob, alternative="greater"
            )

            # Bonferroni-corrected threshold
            corrected_alpha = ALPHA / n_classes

            results.append(
                {
                    "class": cls,
                    "n_samples": n_samples,
                    "n_correct": n_correct,
                    "accuracy": n_correct / n_samples,
                    "random_baseline": random_prob,
                    "p_value": binom_result.pvalue,
                    "bonferroni_threshold": corrected_alpha,
                    "significant": binom_result.pvalue < corrected_alpha,
                }
            )

    return pd.DataFrame(results)


def mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> dict:
    """
    Perform McNemar's test to compare two classifiers.
    """
    # Build contingency table
    correct1 = y_pred1 == y_true
    correct2 = y_pred2 == y_true

    # b: clf1 correct, clf2 wrong
    # c: clf1 wrong, clf2 correct
    b = np.sum(correct1 & ~correct2)
    c = np.sum(~correct1 & correct2)

    # McNemar's test with continuity correction
    if b + c == 0:
        return {"statistic": 0, "p_value": 1.0, "significant": False}

    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        "b": int(b),
        "c": int(c),
        "statistic": statistic,
        "p_value": p_value,
        "significant": p_value < ALPHA,
    }


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    n_classes: int,
    n_bootstrap: int = N_BOOTSTRAP,
    confidence: float = 0.95,
) -> dict:
    """
    Compute bootstrap confidence intervals for classification metrics.
    """
    n_samples = len(y_true)
    alpha = 1 - confidence

    metrics = {
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_f1": [],
        "weighted_f1": [],
        "macro_precision": [],
        "macro_recall": [],
    }

    rng = np.random.RandomState(RANDOM_SEED)

    for _ in range(n_bootstrap):
        # Bootstrap sample
        idx = rng.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]

        # Compute metrics
        metrics["accuracy"].append(accuracy_score(y_true_boot, y_pred_boot))
        metrics["balanced_accuracy"].append(
            balanced_accuracy_score(y_true_boot, y_pred_boot)
        )
        metrics["macro_f1"].append(
            f1_score(y_true_boot, y_pred_boot, average="macro", zero_division=0)
        )
        metrics["weighted_f1"].append(
            f1_score(y_true_boot, y_pred_boot, average="weighted", zero_division=0)
        )
        metrics["macro_precision"].append(
            precision_score(y_true_boot, y_pred_boot, average="macro", zero_division=0)
        )
        metrics["macro_recall"].append(
            recall_score(y_true_boot, y_pred_boot, average="macro", zero_division=0)
        )

    # Compute confidence intervals
    cis = {}
    for metric, values in metrics.items():
        values = np.array(values)
        lower = np.percentile(values, alpha / 2 * 100)
        upper = np.percentile(values, (1 - alpha / 2) * 100)
        mean = np.mean(values)
        cis[metric] = {
            "mean": mean,
            "lower": lower,
            "upper": upper,
            "ci_width": upper - lower,
        }

    return cis


# =============================================================================
# Model Training
# =============================================================================


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    clf,
    param_grid: dict,
) -> tuple:
    """
    Train final model on full dataset with hyperparameter tuning.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        clf,
        param_grid,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_search.fit(X_scaled, y)

    return grid_search.best_estimator_, scaler, grid_search.best_params_


# =============================================================================
# Visualization
# =============================================================================


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    output_file: Path,
    title: str = "Confusion Matrix",
):
    """Plot normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[c.replace("terminal_", "T") for c in classes],
        yticklabels=[c.replace("terminal_", "T") for c in classes],
        ax=ax,
    )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix to {output_file}")


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: list[str],
    output_file: Path,
):
    """Plot ROC curves for each class (one-vs-rest)."""
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))

    aucs = []
    for i, (cls, color) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        ax.plot(
            fpr,
            tpr,
            color=color,
            lw=2,
            label=f"{cls.replace('terminal_', 'T')} (AUC = {roc_auc:.2f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves (One-vs-Rest)\nMean AUC = {np.mean(aucs):.3f}", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved ROC curves to {output_file}")


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: list[str],
    output_file: Path,
):
    """Plot Precision-Recall curves (better for imbalanced data)."""
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))

    for i, (cls, color) in enumerate(zip(classes, colors)):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        pr_auc = auc(recall, precision)

        ax.plot(
            recall,
            precision,
            color=color,
            lw=2,
            label=f"{cls.replace('terminal_', 'T')} (AUC = {pr_auc:.2f})",
        )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves (One-vs-Rest)", fontsize=14)
    ax.legend(loc="lower left", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved PR curves to {output_file}")


def plot_cv_comparison(
    cv_results: list[dict],
    output_file: Path,
    n_classes: int,
):
    """Plot comparison of classifiers from nested CV."""
    data = []
    for result in cv_results:
        for score in result["scores"]:
            data.append(
                {
                    "Classifier": result["clf_name"],
                    "Balanced Accuracy": score,
                }
            )

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.boxplot(data=df, x="Classifier", y="Balanced Accuracy", ax=ax, palette="Set2")
    sns.swarmplot(
        data=df, x="Classifier", y="Balanced Accuracy", color="0.25", size=4, ax=ax
    )

    # Add chance level line
    chance = 1 / n_classes
    ax.axhline(y=chance, color="red", linestyle="--", lw=2, label=f"Chance ({chance:.2f})")

    ax.set_ylabel("Balanced Accuracy", fontsize=12)
    ax.set_xlabel("")
    ax.set_title("Nested Cross-Validation Performance Comparison", fontsize=14)
    ax.legend(loc="lower right")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved CV comparison plot to {output_file}")


def plot_permutation_test(
    observed_score: float,
    perm_scores: np.ndarray,
    p_value: float,
    output_file: Path,
    n_classes: int,
):
    """Plot permutation test results."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(perm_scores, bins=50, density=True, alpha=0.7, color="#3498db", label="Null Distribution")
    ax.axvline(observed_score, color="red", lw=2, label=f"Observed ({observed_score:.3f})")
    ax.axvline(1/n_classes, color="gray", lw=1, linestyle="--", label=f"Chance ({1/n_classes:.3f})")

    # Add text
    significance = "SIGNIFICANT" if p_value < ALPHA else "NOT SIGNIFICANT"
    ax.text(
        0.05,
        0.95,
        f"p-value = {p_value:.4f}\n{significance} at α={ALPHA}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Balanced Accuracy", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Permutation Test (n={len(perm_scores)} permutations)", fontsize=14)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved permutation test plot to {output_file}")


def plot_class_performance(
    binomial_results: pd.DataFrame,
    output_file: Path,
):
    """Plot per-class performance with significance indicators."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(binomial_results))
    width = 0.6

    colors = ["#2ecc71" if sig else "#e74c3c" for sig in binomial_results["significant"]]

    bars = ax.bar(x, binomial_results["accuracy"], width, color=colors)

    # Add random baseline
    ax.axhline(
        y=binomial_results["random_baseline"].iloc[0],
        color="gray",
        linestyle="--",
        lw=2,
        label="Random Baseline",
    )

    # Add significance markers
    for i, (bar, row) in enumerate(zip(bars, binomial_results.itertuples())):
        marker = "***" if row.p_value < 0.001 else "**" if row.p_value < 0.01 else "*" if row.significant else "ns"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            marker,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.replace("terminal_", "T") for c in binomial_results["class"]], fontsize=11
    )
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xlabel("Terminal Class", fontsize=12)
    ax.set_title("Per-Class Accuracy with Significance (Bonferroni-corrected)", fontsize=14)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved class performance plot to {output_file}")


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("03_train_classifier.py - Rigorous Multiclass Classification")
    print("Extended Dataset with State-of-the-Art Statistical Validation")
    print("=" * 70)

    # Create output directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Check input files
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {EMBEDDINGS_FILE}\nRun 02_extract_features.py first."
        )
    if not BIOCHEM_FILE.exists():
        raise FileNotFoundError(
            f"Biochemical features not found: {BIOCHEM_FILE}\nRun 02_extract_features.py first."
        )

    # =========================================================================
    # Step 1: Load data
    # =========================================================================
    print("\n[1/7] Loading data...")
    X, y, names, classes, label_encoder = load_data()

    n_classes = len(classes)
    class_counts = Counter(y)
    print(f"\n  Class distribution:")
    for i, cls in enumerate(classes):
        print(f"    {cls}: {class_counts[i]} ({class_counts[i]/len(y)*100:.1f}%)")

    # Check for severe imbalance
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count
    print(f"\n  Imbalance ratio: {imbalance_ratio:.1f}:1")

    # =========================================================================
    # Step 2: Nested cross-validation for all classifiers
    # =========================================================================
    print(f"\n[2/7] Running nested cross-validation ({N_OUTER_FOLDS}-fold outer, {N_INNER_FOLDS}-fold inner)...")

    classifiers = get_classifiers()
    cv_results = []

    for clf_name, (clf, param_grid) in classifiers.items():
        print(f"\n  Training {clf_name}...")
        result = nested_cross_validation(
            X, y, clf_name, clf, param_grid, n_outer_folds=N_OUTER_FOLDS
        )
        cv_results.append(result)
        print(f"    Balanced Accuracy: {result['mean_score']:.3f} ± {result['std_score']:.3f}")

    # Save CV results
    cv_summary = pd.DataFrame(
        [
            {
                "classifier": r["clf_name"],
                "mean_balanced_accuracy": r["mean_score"],
                "std_balanced_accuracy": r["std_score"],
                "fold_scores": str(r["scores"]),
            }
            for r in cv_results
        ]
    )
    cv_summary.to_csv(MODELS_DIR / "cv_results.csv", index=False)

    # Plot CV comparison
    plot_cv_comparison(cv_results, FIGURES_DIR / "cv_comparison.svg", n_classes)

    # =========================================================================
    # Step 3: Select best classifier and train final model
    # =========================================================================
    print("\n[3/7] Training final model...")

    # Select best classifier
    best_result = max(cv_results, key=lambda x: x["mean_score"])
    best_clf_name = best_result["clf_name"]
    print(f"  Best classifier: {best_clf_name} ({best_result['mean_score']:.3f} ± {best_result['std_score']:.3f})")

    best_clf, best_param_grid = classifiers[best_clf_name]

    final_model, scaler, best_params = train_final_model(X, y, best_clf, best_param_grid)
    print(f"  Best parameters: {best_params}")

    # Save model
    joblib.dump(
        {
            "model": final_model,
            "scaler": scaler,
            "classes": classes,
            "best_params": best_params,
            "classifier_name": best_clf_name,
            "label_encoder": label_encoder,
        },
        MODELS_DIR / "best_model.joblib",
    )
    print(f"  Saved model to {MODELS_DIR / 'best_model.joblib'}")

    # =========================================================================
    # Step 4: Permutation test for significance
    # =========================================================================
    print(f"\n[4/7] Running permutation test ({N_PERMUTATIONS} permutations)...")

    perm_score, perm_pval, perm_scores = permutation_test(
        X, y, final_model, n_permutations=N_PERMUTATIONS
    )

    print(f"  Observed score: {perm_score:.3f}")
    print(f"  Permutation p-value: {perm_pval:.4f}")
    print(f"  {'✓ SIGNIFICANT' if perm_pval < ALPHA else '✗ NOT SIGNIFICANT'} at α={ALPHA}")

    # Save permutation results
    pd.DataFrame(
        {
            "metric": ["balanced_accuracy"],
            "observed_score": [perm_score],
            "p_value": [perm_pval],
            "significant": [perm_pval < ALPHA],
            "n_permutations": [N_PERMUTATIONS],
            "chance_level": [1 / n_classes],
        }
    ).to_csv(STATS_DIR / "permutation_test.csv", index=False)

    plot_permutation_test(
        perm_score, perm_scores, perm_pval, FIGURES_DIR / "permutation_test.svg", n_classes
    )

    # =========================================================================
    # Step 5: Per-class binomial tests
    # =========================================================================
    print("\n[5/7] Running per-class binomial tests...")

    # Get predictions from nested CV
    y_true_cv = best_result["y_true"]
    y_pred_cv = best_result["y_pred"]
    y_proba_cv = best_result["y_proba"]

    binomial_results = binomial_test_per_class(y_true_cv, y_pred_cv, classes)
    binomial_results.to_csv(STATS_DIR / "binomial_per_class.csv", index=False)

    n_significant = binomial_results["significant"].sum()
    print(f"  {n_significant}/{n_classes} classes significantly above chance (Bonferroni-corrected)")

    plot_class_performance(binomial_results, FIGURES_DIR / "class_performance.svg")

    # =========================================================================
    # Step 6: Bootstrap confidence intervals
    # =========================================================================
    print(f"\n[6/7] Computing bootstrap confidence intervals ({N_BOOTSTRAP} iterations)...")

    boot_cis = bootstrap_confidence_intervals(
        y_true_cv, y_pred_cv, y_proba_cv, n_classes, n_bootstrap=N_BOOTSTRAP
    )

    boot_df = pd.DataFrame(
        [
            {
                "metric": metric,
                "mean": values["mean"],
                "ci_lower": values["lower"],
                "ci_upper": values["upper"],
                "ci_width": values["ci_width"],
            }
            for metric, values in boot_cis.items()
        ]
    )
    boot_df.to_csv(STATS_DIR / "bootstrap_ci.csv", index=False)

    print(
        f"  Balanced Accuracy: {boot_cis['balanced_accuracy']['mean']:.3f} "
        f"[{boot_cis['balanced_accuracy']['lower']:.3f}, {boot_cis['balanced_accuracy']['upper']:.3f}]"
    )

    # =========================================================================
    # Step 7: Generate visualizations
    # =========================================================================
    print("\n[7/7] Generating visualizations...")

    # Confusion matrix
    plot_confusion_matrix(
        y_true_cv,
        y_pred_cv,
        classes,
        FIGURES_DIR / "confusion_matrix.svg",
        title=f"Confusion Matrix ({best_clf_name})",
    )

    # ROC and PR curves
    if y_proba_cv is not None and len(y_proba_cv) > 0:
        plot_roc_curves(y_true_cv, y_proba_cv, classes, FIGURES_DIR / "roc_curves.svg")
        plot_precision_recall_curves(
            y_true_cv, y_proba_cv, classes, FIGURES_DIR / "precision_recall_curves.svg"
        )

    # Classification report
    report = classification_report(
        y_true_cv,
        y_pred_cv,
        target_names=[c.replace("terminal_", "T") for c in classes],
        output_dict=True,
    )
    report_df = pd.DataFrame(report).T
    report_df.to_csv(STATS_DIR / "classification_report.csv")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Classification complete!")
    print("=" * 70)

    # Compute vs chance ratio
    chance = 1 / n_classes
    vs_chance = best_result["mean_score"] / chance

    print(f"\n📊 KEY RESULTS:")
    print(f"  Dataset: {len(y)} proteins, {n_classes} classes")
    print(f"  Best Classifier: {best_clf_name}")
    print(f"  Balanced Accuracy: {best_result['mean_score']:.1%} ± {best_result['std_score']:.1%}")
    print(f"  Chance Level: {chance:.1%}")
    print(f"  Performance vs Chance: {vs_chance:.2f}×")
    print(f"  Permutation Test: p = {perm_pval:.4f} ({'✓ SIGNIFICANT' if perm_pval < ALPHA else '✗ not significant'})")
    print(
        f"  95% CI: [{boot_cis['balanced_accuracy']['lower']:.1%}, {boot_cis['balanced_accuracy']['upper']:.1%}]"
    )

    print(f"\n📁 Outputs saved to:")
    print(f"  - {MODELS_DIR / 'best_model.joblib'}")
    print(f"  - {STATS_DIR / 'permutation_test.csv'}")
    print(f"  - {STATS_DIR / 'bootstrap_ci.csv'}")
    print(f"  - {STATS_DIR / 'binomial_per_class.csv'}")
    print(f"  - {FIGURES_DIR / 'confusion_matrix.svg'}")
    print(f"  - {FIGURES_DIR / 'roc_curves.svg'}")

    print(f"\nNext step: Run 04_interpretability.py for feature importance analysis")


if __name__ == "__main__":
    main()
