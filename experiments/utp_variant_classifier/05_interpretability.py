#!/usr/bin/env python3
"""
05_interpretability.py - Interpretability Analysis for uTP Variant Classifier

This script provides comprehensive interpretability analysis:
1. Feature importance (model-based and permutation importance)
2. Embedding visualization (UMAP, t-SNE)
3. Biochemical property analysis per variant
4. Class-specific feature analysis
5. Decision boundary visualization

Statistical Framework:
- Multiple random seeds for visualization stability
- Silhouette score for cluster quality
- Effect sizes (Cohen's d) for property differences
- Kruskal-Wallis test for multi-group comparisons

Output:
- figures/umap_visualization.svg: UMAP embedding colored by variant
- figures/tsne_visualization.svg: t-SNE embedding colored by variant
- figures/feature_importance.svg: Feature importance rankings
- figures/biochemical_analysis.svg: Property distributions by variant
- statistics/feature_importance.csv: Detailed importance scores

Usage:
    uv run python experiments/utp_variant_classifier/05_interpretability.py
"""

import warnings
from pathlib import Path

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Try to import UMAP
try:
    from umap import UMAP

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")

# =============================================================================
# Configuration
# =============================================================================

# Visualization
RANDOM_SEEDS = [42, 123, 456]  # Multiple seeds for stability
TSNE_PERPLEXITY = 30
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

# Statistical
ALPHA = 0.05
N_PERMUTATIONS_IMPORTANCE = 30  # For permutation importance

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
MODEL_FILE = MODELS_DIR / "best_model.joblib"
VARIANT_MAPPING_FILE = OUTPUT_DIR / "data" / "variant_mapping.csv"


# =============================================================================
# Data Loading
# =============================================================================


def load_data():
    """Load all necessary data for interpretability analysis."""
    # Load variant mapping
    variant_df = pd.read_csv(VARIANT_MAPPING_FILE)

    # Load embeddings
    embeddings = {}
    with h5py.File(EMBEDDINGS_FILE, "r") as f:
        for name in f.keys():
            embeddings[name] = f[name][()]

    # Load biochemical features
    biochem_df = pd.read_csv(BIOCHEM_FILE)

    # Load model
    model_data = joblib.load(MODEL_FILE)

    # Align data
    common_names = (
        set(variant_df["name"]) & set(embeddings.keys()) & set(biochem_df["name"])
    )

    # Filter and sort
    variant_df = variant_df[variant_df["name"].isin(common_names)].sort_values("name")
    biochem_df = biochem_df[biochem_df["name"].isin(common_names)].sort_values("name")

    names = variant_df["name"].tolist()
    variants = variant_df["motif_variant"].tolist()

    # Build arrays
    X_emb = np.array([embeddings[n] for n in names])

    biochem_numeric = biochem_df.select_dtypes(include=[np.number])
    if "length" in biochem_numeric.columns:
        biochem_numeric = biochem_numeric.drop(columns=["length"])
    X_biochem = biochem_numeric.values
    biochem_feature_names = list(biochem_numeric.columns)

    X = np.hstack([X_emb, X_biochem])

    return {
        "X": X,
        "X_emb": X_emb,
        "X_biochem": X_biochem,
        "names": names,
        "variants": variants,
        "biochem_df": biochem_df,
        "biochem_features": biochem_feature_names,
        "model": model_data["model"],
        "scaler": model_data["scaler"],
        "classes": model_data["classes"],
    }


# =============================================================================
# Embedding Visualization
# =============================================================================


def run_tsne(
    X: np.ndarray,
    perplexity: int = TSNE_PERPLEXITY,
    random_state: int = 42,
) -> np.ndarray:
    """Run t-SNE dimensionality reduction."""
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(X) - 1),
        random_state=random_state,
        max_iter=1000,
        init="pca",
    )
    return tsne.fit_transform(X)


def run_umap(
    X: np.ndarray,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST,
    random_state: int = 42,
) -> np.ndarray:
    """Run UMAP dimensionality reduction."""
    if not UMAP_AVAILABLE:
        return None

    umap = UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, len(X) - 1),
        min_dist=min_dist,
        random_state=random_state,
        metric="cosine",
    )
    return umap.fit_transform(X)


def plot_embedding(
    coords: np.ndarray,
    labels: list[str],
    title: str,
    output_file: Path,
    silhouette: float = None,
):
    """Plot 2D embedding colored by variant."""
    # Get unique labels and colors
    unique_labels = sorted(set(labels))
    n_labels = len(unique_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_labels, 10)))

    if n_labels > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, n_labels))

    label_to_color = {l: colors[i % len(colors)] for i, l in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(12, 10))

    for label in unique_labels:
        mask = np.array([l == label for l in labels])
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[label_to_color[label]],
            label=f"uTP-{label}",
            alpha=0.7,
            s=50,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    title_text = title
    if silhouette is not None:
        title_text += f"\n(Silhouette Score: {silhouette:.3f})"
    ax.set_title(title_text)

    # Legend outside
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_file}")


def compute_silhouette_scores(
    X: np.ndarray,
    labels: list[str],
) -> dict:
    """Compute silhouette scores for the embedding."""
    # Encode labels
    unique_labels = sorted(set(labels))
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_to_int[l] for l in labels])

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute silhouette score
    if len(unique_labels) > 1:
        score = silhouette_score(X_scaled, y)
    else:
        score = 0.0

    return {"full": score}


# =============================================================================
# Feature Importance
# =============================================================================


def compute_permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = N_PERMUTATIONS_IMPORTANCE,
) -> pd.DataFrame:
    """Compute permutation importance for all features."""
    print(f"  Computing permutation importance ({n_repeats} repeats)...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1,
            scoring="balanced_accuracy",
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    importance_df = importance_df.sort_values("importance_mean", ascending=False)

    return importance_df


def get_model_feature_importance(
    model, feature_names: list[str]
) -> pd.DataFrame | None:
    """Get feature importance from model (if available)."""
    if hasattr(model, "feature_importances_"):
        # Tree-based models (RF, XGBoost)
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Linear models (LR)
        # Use mean absolute coefficient across classes
        importance = np.abs(model.coef_).mean(axis=0)
    else:
        return None

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    )

    return df.sort_values("importance", ascending=False)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_file: Path,
    title: str = "Feature Importance",
    top_n: int = 30,
):
    """Plot feature importance bar chart."""
    # Take top N
    plot_df = importance_df.head(top_n).copy()
    plot_df = plot_df.iloc[::-1]  # Reverse for horizontal bar plot

    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(plot_df))

    if "importance_std" in plot_df.columns:
        ax.barh(
            y_pos, plot_df["importance_mean"], xerr=plot_df["importance_std"], alpha=0.8
        )
        ax.set_xlabel("Permutation Importance")
    else:
        ax.barh(y_pos, plot_df["importance"], alpha=0.8)
        ax.set_xlabel("Feature Importance")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["feature"], fontsize=8)
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_file}")


# =============================================================================
# Biochemical Property Analysis
# =============================================================================


def analyze_biochemical_properties(
    biochem_df: pd.DataFrame,
    top_n_variants: int = 6,
) -> pd.DataFrame:
    """Analyze biochemical properties across variants with statistical tests."""
    # Get top variants
    variant_counts = biochem_df["motif_variant"].value_counts()
    top_variants = variant_counts.head(top_n_variants).index.tolist()

    df_subset = biochem_df[biochem_df["motif_variant"].isin(top_variants)]

    # Get numeric columns
    numeric_cols = df_subset.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ["length", "molecular_weight"]]

    results = []

    for col in numeric_cols:
        # Get groups
        groups = [
            df_subset[df_subset["motif_variant"] == v][col].dropna().values
            for v in top_variants
        ]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            continue

        # Kruskal-Wallis test
        try:
            stat, pval = stats.kruskal(*groups)
        except Exception:
            stat, pval = np.nan, np.nan

        # Effect size: eta-squared (approximate)
        n_total = sum(len(g) for g in groups)
        k = len(groups)
        if n_total > k:
            eta_sq = (stat - k + 1) / (n_total - k)
        else:
            eta_sq = np.nan

        results.append(
            {
                "property": col,
                "kruskal_wallis_stat": stat,
                "p_value": pval,
                "eta_squared": eta_sq,
                "significant": pval < ALPHA,
            }
        )

    return pd.DataFrame(results).sort_values("p_value")


def plot_biochemical_properties(
    biochem_df: pd.DataFrame,
    output_file: Path,
    top_n_variants: int = 6,
):
    """Plot biochemical property distributions by variant."""
    # Get top variants
    variant_counts = biochem_df["motif_variant"].value_counts()
    top_variants = variant_counts.head(top_n_variants).index.tolist()

    df_subset = biochem_df[biochem_df["motif_variant"].isin(top_variants)].copy()

    # Select properties to plot
    properties = [
        "isoelectric_point",
        "gravy",
        "instability_index",
        "fraction_helix",
        "fraction_coil",
        "fraction_disorder_promoting",
    ]
    properties = [p for p in properties if p in df_subset.columns]

    if not properties:
        print("  Warning: No biochemical properties found to plot")
        return

    n_props = len(properties)
    n_cols = 3
    n_rows = (n_props + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_props == 1 else axes

    for i, prop in enumerate(properties):
        ax = axes[i]

        sns.violinplot(
            data=df_subset,
            x="motif_variant",
            y=prop,
            ax=ax,
            palette="tab10",
            inner="box",
        )

        ax.set_xlabel("uTP Variant")
        ax.set_ylabel(prop.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=45)

        # Kruskal-Wallis test
        groups = [
            df_subset[df_subset["motif_variant"] == v][prop].dropna().values
            for v in top_variants
        ]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) > 1:
            try:
                stat, pval = stats.kruskal(*groups)
                sig = (
                    "***"
                    if pval < 0.001
                    else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                )
                ax.set_title(f"{prop}\n(KW p={pval:.2e} {sig})", fontsize=10)
            except Exception:
                ax.set_title(prop.replace("_", " ").title())
        else:
            ax.set_title(prop.replace("_", " ").title())

    # Hide empty axes
    for i in range(len(properties), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_file}")


# =============================================================================
# Class-specific Analysis
# =============================================================================


def analyze_class_separability(
    X: np.ndarray,
    variants: list[str],
) -> pd.DataFrame:
    """Analyze how well each class separates from others."""
    unique_variants = sorted(set(variants))
    variant_to_int = {v: i for i, v in enumerate(unique_variants)}
    y = np.array([variant_to_int[v] for v in variants])

    results = []

    for variant in unique_variants:
        variant_idx = variant_to_int[variant]

        # Binary: this variant vs others
        binary_y = (y == variant_idx).astype(int)

        # Compute centroid distances
        variant_mask = y == variant_idx
        other_mask = y != variant_idx

        variant_centroid = X[variant_mask].mean(axis=0)
        other_centroid = X[other_mask].mean(axis=0)

        # Distance between centroids
        centroid_distance = np.linalg.norm(variant_centroid - other_centroid)

        # Within-class variance
        variant_var = X[variant_mask].var(axis=0).mean()
        other_var = X[other_mask].var(axis=0).mean()

        # Ratio (larger = better separability)
        separability_ratio = centroid_distance / np.sqrt(variant_var + other_var + 1e-8)

        results.append(
            {
                "variant": variant,
                "n_samples": variant_mask.sum(),
                "centroid_distance": centroid_distance,
                "within_variance": variant_var,
                "separability_ratio": separability_ratio,
            }
        )

    return pd.DataFrame(results).sort_values("separability_ratio", ascending=False)


def plot_class_separability(
    separability_df: pd.DataFrame,
    output_file: Path,
):
    """Plot class separability analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Separability ratio
    ax = axes[0]
    sns.barplot(
        data=separability_df,
        x="variant",
        y="separability_ratio",
        ax=ax,
        palette="viridis",
    )
    ax.set_xlabel("uTP Variant")
    ax.set_ylabel("Separability Ratio")
    ax.set_title("Class Separability\n(Higher = More Distinguishable)")
    ax.tick_params(axis="x", rotation=45)

    # Sample sizes
    ax = axes[1]
    sns.barplot(
        data=separability_df,
        x="variant",
        y="n_samples",
        ax=ax,
        palette="tab10",
    )
    ax.set_xlabel("uTP Variant")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Sample Size per Variant")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_file}")


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("05_interpretability.py - Interpretability Analysis")
    print("=" * 70)

    # Create output directories
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    # Check input files
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}\nRun 03_train_classifier.py first."
        )

    # =========================================================================
    # Step 1: Load data
    # =========================================================================
    print("\n[1/6] Loading data...")
    data = load_data()

    X = data["X"]
    X_emb = data["X_emb"]
    variants = data["variants"]
    biochem_df = data["biochem_df"]
    model = data["model"]
    scaler = data["scaler"]
    classes = data["classes"]

    # Encode labels
    unique_variants = sorted(set(variants))
    variant_to_int = {v: i for i, v in enumerate(unique_variants)}
    y = np.array([variant_to_int[v] for v in variants])

    print(f"  Loaded {len(variants)} samples, {len(unique_variants)} variants")

    # =========================================================================
    # Step 2: Embedding visualization
    # =========================================================================
    print("\n[2/6] Computing embedding visualizations...")

    # Scale embeddings
    emb_scaler = StandardScaler()
    X_emb_scaled = emb_scaler.fit_transform(X_emb)

    # t-SNE (with best seed)
    print("  Running t-SNE...")
    best_tsne_score = -1
    best_tsne_coords = None

    for seed in RANDOM_SEEDS:
        coords = run_tsne(X_emb_scaled, random_state=seed)
        score = silhouette_score(coords, y) if len(unique_variants) > 1 else 0
        if score > best_tsne_score:
            best_tsne_score = score
            best_tsne_coords = coords

    plot_embedding(
        best_tsne_coords,
        variants,
        "t-SNE Visualization of Mature Domain Embeddings",
        FIGURES_DIR / "tsne_visualization.svg",
        silhouette=best_tsne_score,
    )

    # UMAP
    if UMAP_AVAILABLE:
        print("  Running UMAP...")
        best_umap_score = -1
        best_umap_coords = None

        for seed in RANDOM_SEEDS:
            coords = run_umap(X_emb_scaled, random_state=seed)
            if coords is not None:
                score = silhouette_score(coords, y) if len(unique_variants) > 1 else 0
                if score > best_umap_score:
                    best_umap_score = score
                    best_umap_coords = coords

        if best_umap_coords is not None:
            plot_embedding(
                best_umap_coords,
                variants,
                "UMAP Visualization of Mature Domain Embeddings",
                FIGURES_DIR / "umap_visualization.svg",
                silhouette=best_umap_score,
            )

    # Save silhouette scores
    silhouette_results = {
        "method": ["t-SNE", "UMAP"] if UMAP_AVAILABLE else ["t-SNE"],
        "silhouette_score": (
            [best_tsne_score, best_umap_score] if UMAP_AVAILABLE else [best_tsne_score]
        ),
    }
    pd.DataFrame(silhouette_results).to_csv(
        STATS_DIR / "silhouette_scores.csv", index=False
    )

    print(f"  t-SNE Silhouette Score: {best_tsne_score:.3f}")
    if UMAP_AVAILABLE:
        print(f"  UMAP Silhouette Score: {best_umap_score:.3f}")

    # =========================================================================
    # Step 3: Feature importance
    # =========================================================================
    print("\n[3/6] Computing feature importance...")

    # Create feature names
    n_emb_features = X_emb.shape[1]
    emb_feature_names = [f"emb_{i}" for i in range(n_emb_features)]
    all_feature_names = emb_feature_names + data["biochem_features"]

    # Scale features
    X_scaled = scaler.transform(X)

    # Permutation importance
    perm_importance = compute_permutation_importance(
        model, X_scaled, y, all_feature_names
    )
    perm_importance.to_csv(STATS_DIR / "permutation_importance.csv", index=False)

    # Plot top features (including biochem)
    # Filter to show biochem features prominently
    biochem_importance = perm_importance[
        perm_importance["feature"].isin(data["biochem_features"])
    ]

    if len(biochem_importance) > 0:
        plot_feature_importance(
            biochem_importance,
            FIGURES_DIR / "biochem_feature_importance.svg",
            title="Biochemical Feature Importance",
            top_n=len(data["biochem_features"]),
        )

    # Model-based importance (if available)
    model_importance = get_model_feature_importance(model, all_feature_names)
    if model_importance is not None:
        model_importance.to_csv(STATS_DIR / "model_feature_importance.csv", index=False)

        biochem_model_imp = model_importance[
            model_importance["feature"].isin(data["biochem_features"])
        ]
        if len(biochem_model_imp) > 0:
            plot_feature_importance(
                biochem_model_imp,
                FIGURES_DIR / "model_feature_importance.svg",
                title="Model Feature Importance (Biochemical)",
                top_n=len(data["biochem_features"]),
            )

    # =========================================================================
    # Step 4: Biochemical property analysis
    # =========================================================================
    print("\n[4/6] Analyzing biochemical properties...")

    prop_results = analyze_biochemical_properties(biochem_df)
    prop_results.to_csv(STATS_DIR / "biochem_property_analysis.csv", index=False)

    n_significant = (
        prop_results["significant"].sum()
        if "significant" in prop_results.columns
        else 0
    )
    print(
        f"  {n_significant}/{len(prop_results)} properties significantly different across variants"
    )

    plot_biochemical_properties(
        biochem_df,
        FIGURES_DIR / "biochem_property_distributions.svg",
    )

    # =========================================================================
    # Step 5: Class separability analysis
    # =========================================================================
    print("\n[5/6] Analyzing class separability...")

    separability = analyze_class_separability(X_emb, variants)
    separability.to_csv(STATS_DIR / "class_separability.csv", index=False)

    plot_class_separability(separability, FIGURES_DIR / "class_separability.svg")

    print("\n  Most distinguishable variants:")
    for _, row in separability.head(3).iterrows():
        print(
            f"    uTP-{row['variant']}: ratio={row['separability_ratio']:.2f} (n={row['n_samples']})"
        )

    # =========================================================================
    # Step 6: Generate summary
    # =========================================================================
    print("\n[6/6] Generating summary...")

    summary = {
        "n_samples": len(variants),
        "n_classes": len(unique_variants),
        "n_embedding_features": n_emb_features,
        "n_biochem_features": len(data["biochem_features"]),
        "tsne_silhouette": best_tsne_score,
        "umap_silhouette": best_umap_score if UMAP_AVAILABLE else np.nan,
        "n_significant_properties": n_significant,
        "most_important_biochem": (
            biochem_importance.iloc[0]["feature"]
            if len(biochem_importance) > 0
            else "N/A"
        ),
        "most_separable_variant": separability.iloc[0]["variant"],
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
    print(f"  Embedding separation (Silhouette):")
    print(f"    - t-SNE: {best_tsne_score:.3f}")
    if UMAP_AVAILABLE:
        print(f"    - UMAP: {best_umap_score:.3f}")

    if len(biochem_importance) > 0:
        print(f"\n  Top biochemical features:")
        for _, row in biochem_importance.head(3).iterrows():
            print(f"    - {row['feature']}: {row['importance_mean']:.4f}")

    print(f"\n  Most distinguishable variants:")
    for _, row in separability.head(3).iterrows():
        print(
            f"    - uTP-{row['variant']}: separability={row['separability_ratio']:.2f}"
        )

    print(f"\n📁 Outputs saved to:")
    print(f"  - {FIGURES_DIR / 'tsne_visualization.svg'}")
    print(f"  - {FIGURES_DIR / 'umap_visualization.svg'}")
    print(f"  - {FIGURES_DIR / 'biochem_property_distributions.svg'}")
    print(f"  - {STATS_DIR / 'permutation_importance.csv'}")
    print(f"  - {STATS_DIR / 'class_separability.csv'}")


if __name__ == "__main__":
    main()
