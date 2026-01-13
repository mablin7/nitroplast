#!/usr/bin/env python3
"""
04_interpretability.py - Model Interpretability and Feature Importance Analysis

This script analyzes what features drive the classification of uTP terminal variants:
1. Permutation feature importance
2. SHAP values for feature attribution
3. Biochemical property analysis per class
4. Embedding dimension analysis
5. Misclassification analysis

Output:
- statistics/feature_importance.csv
- statistics/shap_values.csv
- statistics/misclassification_analysis.csv
- figures/feature_importance.svg
- figures/shap_summary.svg
- figures/biochem_importance.svg

Usage:
    uv run python experiments/utp_variant_classifier_extended/04_interpretability.py
"""

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
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Try to import SHAP
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available, skipping SHAP analysis")

# =============================================================================
# Configuration
# =============================================================================

N_PERMUTATIONS = 100  # For permutation importance
RANDOM_SEED = 42
TOP_N_FEATURES = 30  # Number of top features to display

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
MODEL_FILE = MODELS_DIR / "best_model.joblib"


# =============================================================================
# Data Loading
# =============================================================================


def load_data_and_model():
    """Load features, labels, and trained model."""
    # Load model
    model_data = joblib.load(MODEL_FILE)
    model = model_data["model"]
    scaler = model_data["scaler"]
    classes = model_data["classes"]
    label_encoder = model_data["label_encoder"]

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

    protein_df = protein_df[protein_df["name"].isin(common_names)].sort_values("name")
    biochem_df = biochem_df[biochem_df["name"].isin(common_names)].sort_values("name")

    names = protein_df["name"].tolist()
    y_labels = protein_df["terminal_class"].tolist()

    # Build feature matrix
    X_emb = np.array([embeddings[n] for n in names])

    # Biochemical features
    biochem_numeric = biochem_df.select_dtypes(include=[np.number])
    biochem_feature_names = [c for c in biochem_numeric.columns if c != "length"]
    if "length" in biochem_numeric.columns:
        biochem_numeric = biochem_numeric.drop(columns=["length"])
    X_biochem = biochem_numeric.values

    # Combine features
    X = np.hstack([X_emb, X_biochem])

    # Feature names
    embedding_dim = X_emb.shape[1]
    feature_names = [f"emb_{i}" for i in range(embedding_dim)] + biochem_feature_names

    # Encode labels
    y = label_encoder.transform(y_labels)

    return X, y, names, classes, feature_names, model, scaler, biochem_df, protein_df


# =============================================================================
# Permutation Feature Importance
# =============================================================================


def compute_permutation_importance(
    X: np.ndarray,
    y: np.ndarray,
    model,
    scaler,
    feature_names: list[str],
    n_repeats: int = N_PERMUTATIONS,
) -> pd.DataFrame:
    """
    Compute permutation feature importance.

    This measures how much model performance decreases when each feature is randomly shuffled.
    """
    print(f"  Computing permutation importance ({n_repeats} repeats)...")

    X_scaled = scaler.transform(X)

    result = permutation_importance(
        model,
        X_scaled,
        y,
        n_repeats=n_repeats,
        random_state=RANDOM_SEED,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    importance_df = importance_df.sort_values("importance_mean", ascending=False)
    importance_df["rank"] = range(1, len(importance_df) + 1)

    return importance_df


# =============================================================================
# SHAP Analysis
# =============================================================================


def compute_shap_values(
    X: np.ndarray,
    model,
    scaler,
    feature_names: list[str],
    max_samples: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute SHAP values for feature attribution.
    """
    if not SHAP_AVAILABLE:
        return None, None

    print(f"  Computing SHAP values (max {max_samples} samples)...")

    X_scaled = scaler.transform(X)

    # Subsample if too large
    if len(X_scaled) > max_samples:
        idx = np.random.RandomState(RANDOM_SEED).choice(
            len(X_scaled), max_samples, replace=False
        )
        X_sample = X_scaled[idx]
    else:
        X_sample = X_scaled

    # Use appropriate explainer based on model type
    model_name = type(model).__name__

    if model_name in [
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "XGBClassifier",
    ]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        # Use KernelExplainer for other models (slower)
        background = shap.sample(X_scaled, min(100, len(X_scaled)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_sample, nsamples=100)

    return shap_values, X_sample


# =============================================================================
# Biochemical Property Analysis
# =============================================================================


def analyze_biochemical_properties(
    biochem_df: pd.DataFrame,
    classes: list[str],
) -> pd.DataFrame:
    """
    Analyze biochemical property differences between classes.
    """
    print("  Analyzing biochemical properties...")

    # Get numeric columns (excluding name and class)
    numeric_cols = [
        c
        for c in biochem_df.select_dtypes(include=[np.number]).columns
        if c not in ["length"]
    ]

    results = []

    for prop in numeric_cols:
        # Get values per class
        groups = []
        for cls in classes:
            values = (
                biochem_df[biochem_df["terminal_class"] == cls][prop].dropna().values
            )
            groups.append(values)

        # Kruskal-Wallis test
        if all(len(g) > 0 for g in groups):
            try:
                stat, pval = stats.kruskal(*groups)

                # Effect size (eta-squared)
                n_total = sum(len(g) for g in groups)
                eta_squared = (stat - len(groups) + 1) / (n_total - len(groups))

                results.append(
                    {
                        "property": prop,
                        "kruskal_wallis_stat": stat,
                        "p_value": pval,
                        "eta_squared": max(0, eta_squared),
                        "significant": pval < 0.05,
                    }
                )
            except Exception:
                continue

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("p_value")

    # Add FDR correction
    if len(results_df) > 0:
        from scipy.stats import false_discovery_control

        results_df["q_value"] = false_discovery_control(results_df["p_value"].values)
        results_df["significant_fdr"] = results_df["q_value"] < 0.1

    return results_df


def compute_class_property_means(
    biochem_df: pd.DataFrame,
    classes: list[str],
) -> pd.DataFrame:
    """Compute mean property values per class."""
    numeric_cols = [
        c
        for c in biochem_df.select_dtypes(include=[np.number]).columns
        if c not in ["length"]
    ]

    means = []
    for cls in classes:
        class_df = biochem_df[biochem_df["terminal_class"] == cls]
        row = {"terminal_class": cls}
        for col in numeric_cols:
            row[col] = class_df[col].mean()
        means.append(row)

    return pd.DataFrame(means)


# =============================================================================
# Misclassification Analysis
# =============================================================================


def analyze_misclassifications(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    classes: list[str],
    model,
    scaler,
    protein_df: pd.DataFrame,
    biochem_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze misclassified samples to understand error patterns.
    """
    print("  Analyzing misclassifications...")

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    # Get probabilities if available
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_scaled)
    else:
        y_proba = None

    # Find misclassified samples
    misclassified_idx = np.where(y != y_pred)[0]

    results = []
    for idx in misclassified_idx:
        name = names[idx]
        true_class = classes[y[idx]]
        pred_class = classes[y_pred[idx]]

        row = {
            "name": name,
            "true_class": true_class,
            "predicted_class": pred_class,
        }

        # Add confidence if available
        if y_proba is not None:
            row["confidence"] = y_proba[idx, y_pred[idx]]
            row["true_class_prob"] = y_proba[idx, y[idx]]

        # Add protein info
        protein_row = protein_df[protein_df["name"] == name]
        if len(protein_row) > 0:
            row["motif_pattern"] = protein_row.iloc[0].get("motif_pattern", "")
            row["in_experimental"] = protein_row.iloc[0].get("in_experimental", False)

        # Add biochemical properties
        biochem_row = biochem_df[biochem_df["name"] == name]
        if len(biochem_row) > 0:
            for col in ["isoelectric_point", "gravy", "fraction_helix"]:
                if col in biochem_row.columns:
                    row[col] = biochem_row.iloc[0][col]

        results.append(row)

    return pd.DataFrame(results)


# =============================================================================
# Visualization
# =============================================================================


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_file: Path,
    top_n: int = TOP_N_FEATURES,
):
    """Plot top feature importances."""
    # Separate embedding and biochemical features
    top_df = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by feature type
    colors = [
        "#3498db" if f.startswith("emb_") else "#e74c3c" for f in top_df["feature"]
    ]

    y_pos = np.arange(len(top_df))
    ax.barh(
        y_pos, top_df["importance_mean"], xerr=top_df["importance_std"], color=colors
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_df["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Permutation Importance (decrease in balanced accuracy)")
    ax.set_title(f"Top {top_n} Feature Importances")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#3498db", label="Embedding dimension"),
        Patch(facecolor="#e74c3c", label="Biochemical property"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved feature importance plot to {output_file}")


def plot_biochem_importance(
    biochem_results: pd.DataFrame,
    output_file: Path,
):
    """Plot biochemical property importance."""
    if len(biochem_results) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by p-value
    df = biochem_results.sort_values("p_value")

    colors = ["#2ecc71" if sig else "#95a5a6" for sig in df["significant_fdr"]]

    y_pos = np.arange(len(df))
    ax.barh(y_pos, -np.log10(df["p_value"]), color=colors)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["property"])
    ax.invert_yaxis()
    ax.set_xlabel("-log10(p-value)")
    ax.set_title("Biochemical Property Differences Between Classes (Kruskal-Wallis)")

    # Add significance threshold line
    ax.axvline(x=-np.log10(0.05), color="red", linestyle="--", label="p=0.05")

    ax.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved biochemical importance plot to {output_file}")


def plot_class_property_heatmap(
    class_means: pd.DataFrame,
    output_file: Path,
):
    """Plot heatmap of mean properties per class."""
    # Prepare data
    df = class_means.set_index("terminal_class")

    # Standardize for visualization
    df_std = (df - df.mean()) / df.std()

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(
        df_std.T,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        cbar_kws={"label": "Z-score"},
    )

    ax.set_xlabel("Terminal Class")
    ax.set_ylabel("Property")
    ax.set_title("Standardized Mean Properties by Terminal Class")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved class property heatmap to {output_file}")


def plot_confusion_patterns(
    misclass_df: pd.DataFrame,
    classes: list[str],
    output_file: Path,
):
    """Plot confusion patterns from misclassifications."""
    if len(misclass_df) == 0:
        return

    # Count confusion patterns
    confusion_counts = (
        misclass_df.groupby(["true_class", "predicted_class"])
        .size()
        .reset_index(name="count")
    )

    # Create confusion matrix
    confusion_matrix = np.zeros((len(classes), len(classes)))
    for _, row in confusion_counts.iterrows():
        i = classes.index(row["true_class"])
        j = classes.index(row["predicted_class"])
        confusion_matrix[i, j] = row["count"]

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=".0f",
        cmap="Reds",
        xticklabels=[c.replace("terminal_", "T") for c in classes],
        yticklabels=[c.replace("terminal_", "T") for c in classes],
        ax=ax,
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Misclassification Patterns")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion patterns plot to {output_file}")


def plot_shap_summary(
    shap_values,
    X_sample: np.ndarray,
    feature_names: list[str],
    classes: list[str],
    output_file: Path,
):
    """Plot SHAP summary."""
    if shap_values is None:
        return

    # For multiclass, shap_values is a list of arrays (one per class)
    # We'll plot for the first class or aggregate

    fig, ax = plt.subplots(figsize=(12, 8))

    # Get top features by mean absolute SHAP value
    if isinstance(shap_values, list):
        # Multiclass: average across classes
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    top_idx = np.argsort(mean_abs_shap)[-TOP_N_FEATURES:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    top_importance = mean_abs_shap[top_idx]

    colors = ["#3498db" if f.startswith("emb_") else "#e74c3c" for f in top_features]

    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_importance, color=colors)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {TOP_N_FEATURES} Features by SHAP Importance")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP summary plot to {output_file}")


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("04_interpretability.py - Model Interpretability Analysis")
    print("=" * 70)

    # Check input files
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}\nRun 03_train_classifier.py first."
        )

    # =========================================================================
    # Step 1: Load data and model
    # =========================================================================
    print("\n[1/6] Loading data and model...")
    X, y, names, classes, feature_names, model, scaler, biochem_df, protein_df = (
        load_data_and_model()
    )

    print(f"  Loaded {len(y)} samples, {len(feature_names)} features")
    print(f"  Model type: {type(model).__name__}")

    # =========================================================================
    # Step 2: Permutation feature importance
    # =========================================================================
    print("\n[2/6] Computing permutation feature importance...")
    importance_df = compute_permutation_importance(
        X, y, model, scaler, feature_names, n_repeats=N_PERMUTATIONS
    )
    importance_df.to_csv(STATS_DIR / "feature_importance.csv", index=False)

    # Summary
    top_10 = importance_df.head(10)
    print("\n  Top 10 features:")
    for _, row in top_10.iterrows():
        print(
            f"    {row['feature']}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}"
        )

    # Count feature types in top 30
    top_30 = importance_df.head(30)
    n_emb = sum(1 for f in top_30["feature"] if f.startswith("emb_"))
    n_biochem = len(top_30) - n_emb
    print(f"\n  Top 30 features: {n_emb} embedding dims, {n_biochem} biochemical")

    plot_feature_importance(importance_df, FIGURES_DIR / "feature_importance.svg")

    # =========================================================================
    # Step 3: SHAP analysis
    # =========================================================================
    print("\n[3/6] Computing SHAP values...")
    if SHAP_AVAILABLE:
        shap_values, X_sample = compute_shap_values(
            X, model, scaler, feature_names, max_samples=200
        )
        if shap_values is not None:
            plot_shap_summary(
                shap_values,
                X_sample,
                feature_names,
                classes,
                FIGURES_DIR / "shap_summary.svg",
            )
    else:
        print("  Skipping SHAP analysis (not available)")

    # =========================================================================
    # Step 4: Biochemical property analysis
    # =========================================================================
    print("\n[4/6] Analyzing biochemical properties...")
    biochem_results = analyze_biochemical_properties(biochem_df, classes)
    biochem_results.to_csv(STATS_DIR / "biochem_property_tests.csv", index=False)

    n_sig = (
        biochem_results["significant_fdr"].sum()
        if "significant_fdr" in biochem_results.columns
        else 0
    )
    print(f"  {n_sig} properties significantly different between classes (FDR < 0.1)")

    plot_biochem_importance(biochem_results, FIGURES_DIR / "biochem_importance.svg")

    # Class property means
    class_means = compute_class_property_means(biochem_df, classes)
    class_means.to_csv(STATS_DIR / "class_property_means.csv", index=False)
    plot_class_property_heatmap(class_means, FIGURES_DIR / "class_property_heatmap.svg")

    # =========================================================================
    # Step 5: Misclassification analysis
    # =========================================================================
    print("\n[5/6] Analyzing misclassifications...")
    misclass_df = analyze_misclassifications(
        X, y, names, classes, model, scaler, protein_df, biochem_df
    )
    misclass_df.to_csv(STATS_DIR / "misclassification_analysis.csv", index=False)

    n_misclass = len(misclass_df)
    n_total = len(y)
    print(
        f"  {n_misclass}/{n_total} samples misclassified ({n_misclass/n_total*100:.1f}%)"
    )

    if len(misclass_df) > 0:
        # Most common confusion patterns
        confusion_counts = (
            misclass_df.groupby(["true_class", "predicted_class"])
            .size()
            .reset_index(name="count")
        )
        confusion_counts = confusion_counts.sort_values("count", ascending=False)
        print("\n  Top confusion patterns:")
        for _, row in confusion_counts.head(5).iterrows():
            print(f"    {row['true_class']} → {row['predicted_class']}: {row['count']}")

        plot_confusion_patterns(
            misclass_df, classes, FIGURES_DIR / "confusion_patterns.svg"
        )

    # =========================================================================
    # Step 6: Summary statistics
    # =========================================================================
    print("\n[6/6] Generating summary...")

    # Embedding vs biochemical importance
    emb_importance = importance_df[importance_df["feature"].str.startswith("emb_")][
        "importance_mean"
    ].sum()
    biochem_importance = importance_df[
        ~importance_df["feature"].str.startswith("emb_")
    ]["importance_mean"].sum()

    summary = {
        "total_features": len(feature_names),
        "embedding_dimensions": sum(1 for f in feature_names if f.startswith("emb_")),
        "biochemical_features": sum(
            1 for f in feature_names if not f.startswith("emb_")
        ),
        "total_importance_embedding": emb_importance,
        "total_importance_biochemical": biochem_importance,
        "n_significant_biochem_properties": n_sig,
        "n_misclassified": n_misclass,
        "misclassification_rate": n_misclass / n_total,
    }

    pd.DataFrame([summary]).to_csv(
        STATS_DIR / "interpretability_summary.csv", index=False
    )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Interpretability analysis complete!")
    print("=" * 70)

    print(f"\n📊 KEY FINDINGS:")
    print(f"  Feature importance:")
    print(f"    - Embedding dimensions: {emb_importance:.4f} total importance")
    print(f"    - Biochemical properties: {biochem_importance:.4f} total importance")
    print(f"  Biochemical properties: {n_sig} significantly different between classes")
    print(f"  Misclassification rate: {n_misclass/n_total*100:.1f}%")

    print(f"\n📁 Outputs saved to:")
    print(f"  - {STATS_DIR / 'feature_importance.csv'}")
    print(f"  - {STATS_DIR / 'biochem_property_tests.csv'}")
    print(f"  - {STATS_DIR / 'misclassification_analysis.csv'}")
    print(f"  - {FIGURES_DIR / 'feature_importance.svg'}")
    print(f"  - {FIGURES_DIR / 'biochem_importance.svg'}")
    print(f"  - {FIGURES_DIR / 'class_property_heatmap.svg'}")


if __name__ == "__main__":
    main()
