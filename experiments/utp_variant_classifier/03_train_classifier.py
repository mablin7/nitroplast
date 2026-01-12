#!/usr/bin/env python3
"""
03_train_classifier.py - Rigorous Multiclass Classification for uTP Variants

This script implements a statistically rigorous classification pipeline:

Experimental Design:
1. Stratified train/test split (80/20) - test set held out until final evaluation
2. 5-fold stratified cross-validation on training set for model selection
3. Hyperparameter tuning via grid search (inner CV)
4. Final evaluation on held-out test set

Statistical Validation:
- Permutation test (1000 iterations) for overall significance
- Exact binomial test per class vs random baseline
- Bootstrap confidence intervals (1000 iterations) for metrics
- Multiple testing correction (Bonferroni/FDR)

Classifiers:
- Logistic Regression (multinomial, L2 regularized)
- Support Vector Machine (RBF kernel, one-vs-one)
- Random Forest (ensemble of decision trees)
- XGBoost (gradient boosted trees)

Output:
- models/best_model.joblib: Final trained model
- models/cv_results.csv: Cross-validation results
- statistics/classification_report.csv: Per-class metrics
- statistics/permutation_test_results.csv: Significance testing
- statistics/bootstrap_confidence_intervals.csv: CIs for metrics
- figures/confusion_matrix.svg, roc_curves.svg

Usage:
    uv run python experiments/utp_variant_classifier/03_train_classifier.py
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    permutation_test_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
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

# Data splitting
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Cross-validation
N_FOLDS = 5
N_PERMUTATIONS = 1000
N_BOOTSTRAP = 1000

# Significance thresholds
ALPHA = 0.05

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
VARIANT_MAPPING_FILE = OUTPUT_DIR / "data" / "variant_mapping.csv"


# =============================================================================
# Data Loading
# =============================================================================


def load_data() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Load embeddings and variant labels.
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Label array (n_samples,)
        names: Protein names
        classes: Unique class labels
    """
    # Load variant mapping
    variant_df = pd.read_csv(VARIANT_MAPPING_FILE)
    
    # Load embeddings
    embeddings = {}
    with h5py.File(EMBEDDINGS_FILE, "r") as f:
        for name in f.keys():
            embeddings[name] = f[name][()]
    
    # Load biochemical features
    biochem_df = pd.read_csv(BIOCHEM_FILE)
    
    # Align data
    common_names = (
        set(variant_df["name"]) &
        set(embeddings.keys()) &
        set(biochem_df["name"])
    )
    
    # Filter and sort
    variant_df = variant_df[variant_df["name"].isin(common_names)]
    variant_df = variant_df.sort_values("name")
    
    biochem_df = biochem_df[biochem_df["name"].isin(common_names)]
    biochem_df = biochem_df.sort_values("name")
    
    # Build feature matrix
    names = variant_df["name"].tolist()
    y_labels = variant_df["motif_variant"].tolist()
    
    # Embeddings
    X_emb = np.array([embeddings[n] for n in names])
    
    # Biochemical features (numeric columns only)
    biochem_numeric = biochem_df.select_dtypes(include=[np.number])
    # Remove 'length' if using with embeddings (to avoid data leakage from length)
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
    
    return X, y, names, classes


# =============================================================================
# Classifier Definitions
# =============================================================================


def get_classifiers() -> dict:
    """
    Get dictionary of classifiers with their hyperparameter grids.
    
    Returns:
        Dict mapping classifier name to (estimator, param_grid)
    """
    classifiers = {}
    
    # Logistic Regression
    classifiers["Logistic Regression"] = (
        LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_SEED,
        ),
        {
            "C": [0.01, 0.1, 1.0, 10.0],
            "class_weight": [None, "balanced"],
        }
    )
    
    # Support Vector Machine
    classifiers["SVM"] = (
        SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_SEED,
        ),
        {
            "C": [0.1, 1.0, 10.0],
            "gamma": ["scale", "auto"],
            "class_weight": [None, "balanced"],
        }
    )
    
    # Random Forest
    classifiers["Random Forest"] = (
        RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "class_weight": [None, "balanced"],
        }
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
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.3],
            }
        )
    
    return classifiers


# =============================================================================
# Model Training and Evaluation
# =============================================================================


def nested_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    clf_name: str,
    clf,
    param_grid: dict,
    n_outer_folds: int = N_FOLDS,
    n_inner_folds: int = 3,
) -> dict:
    """
    Perform nested cross-validation with hyperparameter tuning.
    
    Outer loop: Model evaluation (unbiased performance estimate)
    Inner loop: Hyperparameter tuning (GridSearchCV)
    
    Returns:
        Dict with scores, best params per fold, and predictions
    """
    outer_cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=RANDOM_SEED)
    inner_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=RANDOM_SEED)
    
    scores = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
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
    }


def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    clf,
    param_grid: dict,
) -> tuple:
    """
    Train final model on full training set with hyperparameter tuning.
    
    Returns:
        (trained_model, scaler, best_params)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
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
        grid_search.fit(X_train_scaled, y_train)
    
    return grid_search.best_estimator_, scaler, grid_search.best_params_


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
    
    Returns:
        (observed_score, p_value, permutation_scores)
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
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
    
    Returns:
        DataFrame with test results per class
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
            
            results.append({
                "class": cls,
                "n_samples": n_samples,
                "n_correct": n_correct,
                "accuracy": n_correct / n_samples,
                "random_baseline": random_prob,
                "p_value": binom_result.pvalue,
                "significant": binom_result.pvalue < ALPHA / n_classes,  # Bonferroni
            })
    
    return pd.DataFrame(results)


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    n_bootstrap: int = N_BOOTSTRAP,
    confidence: float = 0.95,
) -> dict:
    """
    Compute bootstrap confidence intervals for classification metrics.
    
    Returns:
        Dict with metric CIs
    """
    n_samples = len(y_true)
    alpha = 1 - confidence
    
    metrics = {
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_f1": [],
        "weighted_f1": [],
    }
    
    rng = np.random.RandomState(RANDOM_SEED)
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        idx = rng.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]
        
        # Compute metrics
        metrics["accuracy"].append(accuracy_score(y_true_boot, y_pred_boot))
        metrics["balanced_accuracy"].append(balanced_accuracy_score(y_true_boot, y_pred_boot))
        metrics["macro_f1"].append(f1_score(y_true_boot, y_pred_boot, average="macro", zero_division=0))
        metrics["weighted_f1"].append(f1_score(y_true_boot, y_pred_boot, average="weighted", zero_division=0))
    
    # Compute confidence intervals
    cis = {}
    for metric, values in metrics.items():
        values = np.array(values)
        lower = np.percentile(values, alpha/2 * 100)
        upper = np.percentile(values, (1 - alpha/2) * 100)
        mean = np.mean(values)
        cis[metric] = {
            "mean": mean,
            "lower": lower,
            "upper": upper,
            "ci_width": upper - lower,
        }
    
    return cis


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
        xticklabels=[f"uTP-{c}" for c in classes],
        yticklabels=[f"uTP-{c}" for c in classes],
        ax=ax,
    )
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {output_file}")


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
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for i, (cls, color) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(
            fpr, tpr,
            color=color,
            lw=2,
            label=f"uTP-{cls} (AUC = {roc_auc:.2f})"
        )
    
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curves to {output_file}")


def plot_cv_comparison(
    cv_results: list[dict],
    output_file: Path,
):
    """Plot comparison of classifiers from nested CV."""
    data = []
    for result in cv_results:
        for score in result["scores"]:
            data.append({
                "Classifier": result["clf_name"],
                "Balanced Accuracy": score,
            })
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(data=df, x="Classifier", y="Balanced Accuracy", ax=ax)
    sns.swarmplot(data=df, x="Classifier", y="Balanced Accuracy", color="0.25", size=4, ax=ax)
    
    # Add chance level line
    n_classes = cv_results[0]["y_true"].max() + 1 if cv_results else 4
    ax.axhline(y=1/n_classes, color="red", linestyle="--", label=f"Chance ({1/n_classes:.2f})")
    
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Nested Cross-Validation Performance Comparison")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved CV comparison plot to {output_file}")


def plot_permutation_test(
    observed_score: float,
    perm_scores: np.ndarray,
    p_value: float,
    output_file: Path,
):
    """Plot permutation test results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(perm_scores, bins=50, density=True, alpha=0.7, label="Permutation Distribution")
    ax.axvline(observed_score, color="red", lw=2, label=f"Observed ({observed_score:.3f})")
    
    # Add text
    ax.text(
        0.05, 0.95,
        f"p-value = {p_value:.4f}\n" +
        f"{'Significant' if p_value < ALPHA else 'Not significant'} at α={ALPHA}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    
    ax.set_xlabel("Balanced Accuracy")
    ax.set_ylabel("Density")
    ax.set_title(f"Permutation Test (n={len(perm_scores)} permutations)")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved permutation test plot to {output_file}")


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("03_train_classifier.py - Rigorous Multiclass Classification")
    print("=" * 70)
    
    # Create output directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check input files
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(f"Embeddings not found: {EMBEDDINGS_FILE}\nRun 02_extract_features.py first.")
    if not BIOCHEM_FILE.exists():
        raise FileNotFoundError(f"Biochemical features not found: {BIOCHEM_FILE}\nRun 02_extract_features.py first.")
    
    # =========================================================================
    # Step 1: Load data
    # =========================================================================
    print("\n[1/6] Loading data...")
    X, y, names, classes = load_data()
    
    n_classes = len(classes)
    class_counts = Counter(y)
    print(f"\n  Class distribution:")
    for i, cls in enumerate(classes):
        print(f"    uTP-{cls}: {class_counts[i]} ({class_counts[i]/len(y)*100:.1f}%)")
    
    # =========================================================================
    # Step 2: Train/test split
    # =========================================================================
    print(f"\n[2/6] Splitting data (test_size={TEST_SIZE})...")
    
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, names,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples (held out)")
    
    # =========================================================================
    # Step 3: Nested cross-validation
    # =========================================================================
    print(f"\n[3/6] Running nested cross-validation ({N_FOLDS}-fold outer, 3-fold inner)...")
    
    classifiers = get_classifiers()
    cv_results = []
    
    for clf_name, (clf, param_grid) in classifiers.items():
        print(f"\n  Training {clf_name}...")
        result = nested_cross_validation(
            X_train, y_train, clf_name, clf, param_grid,
            n_outer_folds=N_FOLDS,
        )
        cv_results.append(result)
        print(f"    Balanced Accuracy: {result['mean_score']:.3f} ± {result['std_score']:.3f}")
    
    # Save CV results
    cv_summary = pd.DataFrame([
        {
            "classifier": r["clf_name"],
            "mean_balanced_accuracy": r["mean_score"],
            "std_balanced_accuracy": r["std_score"],
            "fold_scores": r["scores"],
        }
        for r in cv_results
    ])
    cv_summary.to_csv(MODELS_DIR / "cv_results.csv", index=False)
    
    # Plot CV comparison
    plot_cv_comparison(cv_results, FIGURES_DIR / "cv_comparison.svg")
    
    # =========================================================================
    # Step 4: Select and train final model
    # =========================================================================
    print("\n[4/6] Training final model on full training set...")
    
    # Select best classifier
    best_result = max(cv_results, key=lambda x: x["mean_score"])
    best_clf_name = best_result["clf_name"]
    print(f"  Best classifier: {best_clf_name} ({best_result['mean_score']:.3f})")
    
    best_clf, best_param_grid = classifiers[best_clf_name]
    
    final_model, scaler, best_params = train_final_model(
        X_train, y_train, best_clf, best_param_grid
    )
    print(f"  Best parameters: {best_params}")
    
    # Save model
    joblib.dump({
        "model": final_model,
        "scaler": scaler,
        "classes": classes,
        "best_params": best_params,
        "classifier_name": best_clf_name,
    }, MODELS_DIR / "best_model.joblib")
    print(f"  Saved model to {MODELS_DIR / 'best_model.joblib'}")
    
    # =========================================================================
    # Step 5: Evaluate on held-out test set
    # =========================================================================
    print("\n[5/6] Evaluating on held-out test set...")
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = final_model.predict(X_test_scaled)
    
    if hasattr(final_model, "predict_proba"):
        y_proba = final_model.predict_proba(X_test_scaled)
    else:
        y_proba = None
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=[f"uTP-{c}" for c in classes], output_dict=True)
    report_df = pd.DataFrame(report).T
    report_df.to_csv(STATS_DIR / "classification_report.csv")
    
    # Print summary
    test_accuracy = accuracy_score(y_test, y_pred)
    test_balanced_acc = balanced_accuracy_score(y_test, y_pred)
    test_f1_macro = f1_score(y_test, y_pred, average="macro")
    
    print(f"\n  Test Set Results:")
    print(f"    Accuracy: {test_accuracy:.3f}")
    print(f"    Balanced Accuracy: {test_balanced_acc:.3f}")
    print(f"    Macro F1: {test_f1_macro:.3f}")
    print(f"    Chance level: {1/n_classes:.3f}")
    
    # =========================================================================
    # Step 6: Statistical validation
    # =========================================================================
    print(f"\n[6/6] Statistical validation...")
    
    # 6a. Permutation test
    print(f"\n  Running permutation test ({N_PERMUTATIONS} permutations)...")
    perm_score, perm_pval, perm_scores = permutation_test(
        X_train, y_train, final_model, n_permutations=N_PERMUTATIONS
    )
    
    print(f"    Observed score: {perm_score:.3f}")
    print(f"    Permutation p-value: {perm_pval:.4f}")
    print(f"    {'SIGNIFICANT' if perm_pval < ALPHA else 'NOT SIGNIFICANT'} at α={ALPHA}")
    
    # Save permutation results
    pd.DataFrame({
        "metric": ["balanced_accuracy"],
        "observed_score": [perm_score],
        "p_value": [perm_pval],
        "significant": [perm_pval < ALPHA],
        "n_permutations": [N_PERMUTATIONS],
    }).to_csv(STATS_DIR / "permutation_test_results.csv", index=False)
    
    plot_permutation_test(perm_score, perm_scores, perm_pval, FIGURES_DIR / "permutation_test.svg")
    
    # 6b. Per-class binomial tests
    print("\n  Running per-class binomial tests...")
    binomial_results = binomial_test_per_class(y_test, y_pred, classes)
    binomial_results.to_csv(STATS_DIR / "binomial_test_results.csv", index=False)
    
    n_significant = binomial_results["significant"].sum()
    print(f"    {n_significant}/{n_classes} classes significantly above chance (Bonferroni corrected)")
    
    # 6c. Bootstrap confidence intervals
    print(f"\n  Computing bootstrap confidence intervals ({N_BOOTSTRAP} iterations)...")
    boot_cis = bootstrap_confidence_intervals(y_test, y_pred, y_proba, n_bootstrap=N_BOOTSTRAP)
    
    boot_df = pd.DataFrame([
        {
            "metric": metric,
            "mean": values["mean"],
            "ci_lower": values["lower"],
            "ci_upper": values["upper"],
            "ci_width": values["ci_width"],
        }
        for metric, values in boot_cis.items()
    ])
    boot_df.to_csv(STATS_DIR / "bootstrap_confidence_intervals.csv", index=False)
    
    print(f"    Balanced Accuracy: {boot_cis['balanced_accuracy']['mean']:.3f} " +
          f"[{boot_cis['balanced_accuracy']['lower']:.3f}, {boot_cis['balanced_accuracy']['upper']:.3f}]")
    
    # =========================================================================
    # Generate visualizations
    # =========================================================================
    print("\n[Figures] Generating visualizations...")
    
    plot_confusion_matrix(y_test, y_pred, classes, FIGURES_DIR / "confusion_matrix.svg")
    
    if y_proba is not None:
        plot_roc_curves(y_test, y_proba, classes, FIGURES_DIR / "roc_curves.svg")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Classification complete!")
    print("=" * 70)
    
    print(f"\n📊 KEY RESULTS:")
    print(f"  Best Classifier: {best_clf_name}")
    print(f"  Test Balanced Accuracy: {test_balanced_acc:.1%} (chance: {1/n_classes:.1%})")
    print(f"  Permutation Test: p = {perm_pval:.4f} ({'✓ SIGNIFICANT' if perm_pval < ALPHA else '✗ not significant'})")
    print(f"  95% CI: [{boot_cis['balanced_accuracy']['lower']:.1%}, {boot_cis['balanced_accuracy']['upper']:.1%}]")
    
    print(f"\n📁 Outputs saved to:")
    print(f"  - {MODELS_DIR / 'best_model.joblib'}")
    print(f"  - {STATS_DIR / 'classification_report.csv'}")
    print(f"  - {STATS_DIR / 'permutation_test_results.csv'}")
    print(f"  - {FIGURES_DIR / 'confusion_matrix.svg'}")
    print(f"  - {FIGURES_DIR / 'roc_curves.svg'}")
    
    print(f"\nNext step: Run 04_annotation_analysis.py for functional enrichment")


if __name__ == "__main__":
    main()
