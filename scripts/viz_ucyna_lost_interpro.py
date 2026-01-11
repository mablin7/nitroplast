import argparse
import os
import re
from typing import Dict, Iterable, Set, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use("seaborn-v0_8-paper")
sns.set_palette("husl")


INTERPRO_COLUMNS = [
    "accession",
    "sequence_md5",
    "sequence_length",
    "analysis",
    "signature_accession",
    "signature_description",
    "start",
    "stop",
    "score",
    "status",
    "date",
    "interpro_accession",
    "interpro_description",
    "go",
    "pathway",
]


def load_interpro_table(input_path: str) -> pd.DataFrame:
    """Load InterProScan TSV (no header) and assign standard columns.

    The table is expected to have one row per hit per sequence across analyses.
    """
    df = pd.read_csv(input_path, sep="\t", header=None, dtype=str, on_bad_lines="skip")
    # Guard against unexpected column counts by truncating/exending to 15
    if df.shape[1] != len(INTERPRO_COLUMNS):
        df = df.iloc[:, : len(INTERPRO_COLUMNS)]
        while df.shape[1] < len(INTERPRO_COLUMNS):
            df[df.shape[1]] = None
    df.columns = INTERPRO_COLUMNS
    return df


def aggregate_interpro_terms(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Aggregate InterPro terms to per-protein counts.

    - Deduplicate by (accession, interpro_accession) so each protein contributes at most once per InterPro entry.
    - Exclude rows without a valid InterPro accession/description.
    Returns the aggregated table and the total number of unique proteins.
    """
    # Count unique proteins in the lost set
    total_proteins = df["accession"].nunique()

    # Keep valid InterPro mappings
    valid = df[
        (df["interpro_accession"].notna()) & (df["interpro_accession"] != "-")
    ].copy()
    if valid.empty:
        return (
            pd.DataFrame(
                columns=[
                    "interpro_accession",
                    "interpro_description",
                    "protein_count",
                    "percent",
                ]
            ),
            total_proteins,
        )

    # Deduplicate to one (protein, InterPro) assignment
    valid = valid.drop_duplicates(
        subset=["accession", "interpro_accession"]
    )  # per protein per InterPro

    # Choose a representative description per InterPro accession
    desc_map = (
        valid[
            valid["interpro_description"].notna()
            & (valid["interpro_description"] != "-")
        ][["interpro_accession", "interpro_description"]]
        .drop_duplicates()
        .set_index("interpro_accession")["interpro_description"]
    )

    counts = (
        valid.groupby("interpro_accession").size().reset_index(name="protein_count")
    )

    counts["interpro_description"] = (
        counts["interpro_accession"].map(desc_map).fillna("-")
    )
    counts["percent"] = (
        (counts["protein_count"] / total_proteins * 100.0).round(2)
        if total_proteins
        else 0.0
    )

    # Put description first for readability
    counts = counts[
        ["interpro_accession", "interpro_description", "protein_count", "percent"]
    ]
    return counts.sort_values("protein_count", ascending=False), total_proteins


def shorten(text: str, max_len: int = 70) -> str:
    if not isinstance(text, str):
        return ""
    return (
        text
        if len(text) <= max_len
        else text[: max_len - 1] + "\N{HORIZONTAL ELLIPSIS}"
    )


def plot_top_interpro(
    agg_df: pd.DataFrame,
    outdir: str,
    total_proteins: int,
    top_n: int = 20,
    min_count: int = 1,
    output_prefix: str = "lost_ucyna_interpro_top",
):
    os.makedirs(outdir, exist_ok=True)

    df = agg_df[agg_df["protein_count"] >= min_count].head(top_n).copy()
    if df.empty:
        print("No InterPro terms to plot with given filters.")
        return

    # Prepare labels: fallback to accession when description is missing or '-'
    labels = df["interpro_description"].copy()
    labels = labels.where(
        labels.notna() & (labels != "-"),
        df["interpro_accession"],
    )
    df["label"] = labels.apply(lambda s: shorten(s, 80))

    # Dynamic height
    height = max(3.5, 0.45 * len(df))
    fig, ax = plt.subplots(figsize=(10, height))

    sns.barplot(
        data=df,
        y="label",
        x="protein_count",
        ax=ax,
        orient="h",
    )

    ax.set_xlabel("Number of Cyanothece proteins (lost in UCYN-A)")
    ax.set_ylabel("InterPro functional term")

    # Annotate counts and percent on bars
    for i, row in df.reset_index(drop=True).iterrows():
        ax.text(
            row["protein_count"] + max(df["protein_count"]) * 0.01,
            i,
            f"{row['protein_count']} ({row['percent']:.1f}%)",
            va="center",
            ha="left",
            fontsize=9,
        )

    plt.tight_layout()

    svg_path = os.path.join(outdir, f"{output_prefix}.svg")
    png_path = os.path.join(outdir, f"{output_prefix}.png")
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=300)
    print(f"Saved: {svg_path}\nSaved: {png_path}")


def _categorize_text(text: str) -> Set[str]:
    """Map a free-text functional description to one or more broad categories.

    This is a heuristic grouping intended to reveal patterns at a glance.
    A single description can contribute to multiple categories.
    """
    categories: Set[str] = set()
    if not isinstance(text, str) or not text:
        return categories
    t = text.lower()

    # Nucleotide-binding / NTPases
    if re.search(
        r"aaa\+|p-?loop|ntp-?binding|nucleoside triphosphate|gtp(?:ase|-binding)|atp(?:ase|-binding)|rossmann|nad\(?p?\)?-?binding|fad-?binding|fmn-?binding",
        t,
    ):
        categories.add("Nucleotide-binding & NTPases")

    # Transporters / membrane transport
    if re.search(
        r"\b(transporter|permease|channel|porin|antiporter|symporter)\b|abc transporter",
        t,
    ):
        categories.add("Transport")

    # DNA/RNA binding & transcriptional regulation
    if re.search(
        r"dna-?binding|rna-?binding|helix-?turn-?helix|winged helix|transcription|repressor|operator|polymerase|ob-?fold",
        t,
    ):
        categories.add("DNA/RNA binding & regulation")

    # Translation and ribosome
    if re.search(
        r"ribosomal|ribosome|elongation factor|initiation factor|aminoacyl-?trna|t[rR]NA synthetase",
        t,
    ):
        categories.add("Translation & ribosome")

    # Protein folding and proteolysis
    if re.search(r"chaperone|\bhsp\b|groel|clp|protease|peptidase|proteinase", t):
        categories.add("Protein folding & proteolysis")

    # Oxidoreductases / redox
    if re.search(
        r"oxidoreductase|dehydrogenase|reductase|oxidase|peroxidase|thioredoxin|disulfide|ferredoxin|flavodoxin",
        t,
    ):
        categories.add("Redox & oxidoreductases")

    # Transferases (incl. kinases, methyl/acetyl/glycosyl transferases)
    if re.search(
        r"transferase|kinase|methyltransferase|acetyltransferase|glycosyltransferase", t
    ):
        categories.add("Transferases & kinases")

    # Hydrolases (incl. nucleases, phosphatases, esterases; peptidases captured above too)
    if re.search(r"hydrolase|nuclease|phosphatase|esterase|lipase", t):
        categories.add("Hydrolases")

    # Ligases / synthases / ligation enzymes
    if re.search(r"ligase|synthetase|synthase", t):
        categories.add("Ligases & synth(het)ases")

    # Lyases / isomerases
    if re.search(r"lyase|isomerase|mutase|epimerase|aldolase", t):
        categories.add("Lyases & isomerases")

    # Cell envelope / membrane association
    if re.search(
        r"membrane|transmembrane|cell wall|peptidoglycan|lipoprotein|s-?layer|outer membrane",
        t,
    ):
        categories.add("Membrane & cell envelope")

    # Photosynthesis / thylakoid
    if re.search(r"photosystem|thylakoid|\bpsa\b|\bpsb\b|chlorophyll|phycobilisome", t):
        categories.add("Photosynthesis & thylakoid")

    # Cofactors / radical SAM / PLP etc.
    if re.search(
        r"radical sam|s-adenosyl|\bplp\b|pyridoxal phosphate|flavin|heme|biotin|lipo(?:yl|ate)",
        t,
    ):
        categories.add("Cofactors & radical SAM")

    # Repeats / structural scaffolds
    if re.search(
        r"tetratricopeptide|\btpr\b|ankyrin|wd40|tim barrel|leucine-?rich repeat|coiled-?coil",
        t,
    ):
        categories.add("Structural repeats & folds")

    return categories


def aggregate_by_category(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Aggregate to broad functional categories at the protein level.

    Each protein contributes at most once to a given category.
    Returns (category_df, total_unique_proteins).
    """
    total_proteins = df["accession"].nunique()
    valid = df[
        (df["interpro_accession"].notna()) & (df["interpro_accession"] != "-")
    ].copy()
    if valid.empty:
        return (
            pd.DataFrame(columns=["category", "protein_count", "percent"]),
            total_proteins,
        )

    # Build per-protein category sets
    protein_to_categories: Dict[str, Set[str]] = {}
    for _, row in valid.iterrows():
        acc = row.get("accession")
        # Prefer InterPro description, fall back to signature description
        text = row.get("interpro_description")
        if not isinstance(text, str) or text == "-":
            text = row.get("signature_description", "")
        cats = _categorize_text(text)
        if not cats:
            continue
        current = protein_to_categories.get(acc)
        if current is None:
            protein_to_categories[acc] = set(cats)
        else:
            current.update(cats)

    # Count proteins per category
    cat_counts: Dict[str, int] = {}
    for acc, cats in protein_to_categories.items():
        for c in cats:
            cat_counts[c] = cat_counts.get(c, 0) + 1

    if not cat_counts:
        return (
            pd.DataFrame(columns=["category", "protein_count", "percent"]),
            total_proteins,
        )

    cat_df = (
        pd.DataFrame(
            {
                "category": list(cat_counts.keys()),
                "protein_count": list(cat_counts.values()),
            }
        )
        .sort_values("protein_count", ascending=False)
        .reset_index(drop=True)
    )
    cat_df["percent"] = (cat_df["protein_count"] / total_proteins * 100.0).round(2)
    return cat_df, total_proteins


def plot_category_bar(
    cat_df: pd.DataFrame,
    outdir: str,
    total_proteins: int,
    top_n: int = 15,
    output_prefix: str = "lost_ucyna_interpro_categories",
):
    os.makedirs(outdir, exist_ok=True)
    df = cat_df.head(top_n).copy()
    height = max(3.5, 0.55 * len(df))
    fig, ax = plt.subplots(figsize=(10.5, height))

    palette = sns.color_palette("Set2", n_colors=max(3, len(df)))
    sns.barplot(
        data=df,
        y="category",
        x="protein_count",
        ax=ax,
        orient="h",
        palette=palette,
    )

    ax.set_xlabel("Proteins annotated to category (Cyanothece, absent in UCYN-A)")
    ax.set_ylabel("Functional category")

    for i, row in df.reset_index(drop=True).iterrows():
        ax.text(
            row["protein_count"] + max(df["protein_count"]) * 0.01,
            i,
            f"{row['protein_count']} ({row['percent']:.1f}%)",
            va="center",
            ha="left",
            fontsize=10,
        )

    plt.tight_layout()
    svg_path = os.path.join(outdir, f"{output_prefix}.svg")
    png_path = os.path.join(outdir, f"{output_prefix}.png")
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=300)
    print(f"Saved: {svg_path}\nSaved: {png_path}")


def plot_category_treemap(
    cat_df: pd.DataFrame,
    outdir: str,
    output_prefix: str = "lost_ucyna_interpro_categories_treemap",
):
    try:
        import squarify  # type: ignore
    except Exception:
        print("squarify not available; skipping treemap.")
        return

    os.makedirs(outdir, exist_ok=True)
    df = cat_df.copy()
    fig, ax = plt.subplots(figsize=(10, 7))
    squarify.plot(
        sizes=df["protein_count"].tolist(),
        label=[f"{c}\n{n}" for c, n in zip(df["category"], df["protein_count"])],
        alpha=0.85,
        color=sns.color_palette("Set3", n_colors=max(3, len(df))),
        ax=ax,
        text_kwargs={"fontsize": 10},
    )
    ax.axis("off")

    svg_path = os.path.join(outdir, f"{output_prefix}.svg")
    png_path = os.path.join(outdir, f"{output_prefix}.png")
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=300)
    print(f"Saved: {svg_path}\nSaved: {png_path}")


def extract_go_ids(go_field: str) -> Set[str]:
    """Extract GO IDs from the InterPro 'go' column.

    The field often looks like: 'GO:0008150(InterPro)|GO:0003674(InterPro)'.
    """
    if not isinstance(go_field, str) or not go_field or go_field == "-":
        return set()
    return set(re.findall(r"GO:\d{7}", go_field))


def load_go_mapping(obo_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Minimal OBO parser to map GO IDs to names and aspects.

    Returns two dicts: id->name and id->aspect (BP/MF/CC).
    """
    if not obo_path or not os.path.exists(obo_path):
        return {}, {}

    id_to_name: Dict[str, str] = {}
    id_to_aspect: Dict[str, str] = {}

    current_id: str | None = None
    current_name: str | None = None
    current_namespace: str | None = None

    def flush_term():
        nonlocal current_id, current_name, current_namespace
        if current_id:
            if current_name:
                id_to_name[current_id] = current_name
            if current_namespace:
                ns = current_namespace
                aspect = {
                    "biological_process": "BP",
                    "molecular_function": "MF",
                    "cellular_component": "CC",
                }.get(ns, "-")
                id_to_aspect[current_id] = aspect
        current_id = None
        current_name = None
        current_namespace = None

    with open(obo_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                flush_term()
                continue
            if not line or line.startswith("["):
                # new stanza or empty
                continue
            if line.startswith("id: "):
                # Start of a term block (may appear after [Term] or on its own)
                flush_term()
                current_id = line.split("id: ", 1)[1].strip()
            elif line.startswith("name: "):
                current_name = line.split("name: ", 1)[1].strip()
            elif line.startswith("namespace: "):
                current_namespace = line.split("namespace: ", 1)[1].strip()
            elif line.startswith("alt_id: "):
                alt = line.split("alt_id: ", 1)[1].strip()
                # Map alt_id to current term's name and namespace as well
                if current_name:
                    id_to_name[alt] = current_name
                if current_namespace:
                    ns = current_namespace
                    aspect = {
                        "biological_process": "BP",
                        "molecular_function": "MF",
                        "cellular_component": "CC",
                    }.get(ns, "-")
                    id_to_aspect[alt] = aspect

    # Flush the last term if file doesn't end with blank line
    flush_term()
    return id_to_name, id_to_aspect


def aggregate_go_terms(
    df: pd.DataFrame, name_map: Dict[str, str], aspect_map: Dict[str, str]
) -> Tuple[pd.DataFrame, int]:
    """Aggregate GO terms at the protein level.

    - Each (protein, GO) counted once
    - Adds optional name and aspect columns (if provided)
    """
    total_proteins = df["accession"].nunique()

    records = []
    for _, row in df.iterrows():
        accession = row.get("accession")
        for go_id in extract_go_ids(row.get("go", "")):
            records.append((accession, go_id))

    if not records:
        return (
            pd.DataFrame(
                columns=["go_id", "go_name", "aspect", "protein_count", "percent"]
            ),
            total_proteins,
        )

    go_df = pd.DataFrame(records, columns=["accession", "go_id"]).drop_duplicates()
    counts = go_df.groupby("go_id").size().reset_index(name="protein_count")

    counts["go_name"] = counts["go_id"].map(name_map).fillna("") if name_map else ""
    counts["aspect"] = counts["go_id"].map(aspect_map).fillna("") if aspect_map else ""
    counts["percent"] = (counts["protein_count"] / total_proteins * 100.0).round(2)

    counts = counts[["go_id", "go_name", "aspect", "protein_count", "percent"]]
    counts = counts.sort_values(["protein_count", "go_id"], ascending=[False, True])
    return counts, total_proteins


def plot_top_go_terms(
    terms_df: pd.DataFrame,
    outdir: str,
    total_proteins: int,
    top_n: int = 20,
    min_count: int = 1,
    use_go_names: bool = False,
    output_prefix: str = "lost_ucyna_go_top",
):
    os.makedirs(outdir, exist_ok=True)
    df = terms_df[terms_df["protein_count"] >= min_count].head(top_n).copy()
    if df.empty:
        print("No GO terms to plot with given filters.")
        return

    # Label as name (GO:XXXXXXX) if available and requested, else GO ID
    def term_label(row: pd.Series) -> str:
        if use_go_names and isinstance(row.get("go_name"), str) and row["go_name"]:
            return shorten(f"{row['go_name']} ({row['go_id']})", 80)
        return row["go_id"]

    df["label"] = df.apply(term_label, axis=1)
    height = max(3.5, 0.45 * len(df))
    fig, ax = plt.subplots(figsize=(11, height))

    sns.barplot(data=df, y="label", x="protein_count", ax=ax, orient="h")

    ax.set_xlabel("Proteins annotated to GO term (Cyanothece, absent in UCYN-A)")
    ax.set_ylabel("GO term")

    for i, row in df.reset_index(drop=True).iterrows():
        ax.text(
            row["protein_count"] + max(df["protein_count"]) * 0.01,
            i,
            f"{row['protein_count']} ({row['percent']:.1f}%)",
            va="center",
            ha="left",
            fontsize=9,
        )

    plt.tight_layout()
    svg_path = os.path.join(outdir, f"{output_prefix}.svg")
    png_path = os.path.join(outdir, f"{output_prefix}.png")
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=300)
    print(f"Saved: {svg_path}\nSaved: {png_path}")


def aggregate_go_aspects(
    df: pd.DataFrame, aspect_map: Dict[str, str]
) -> Tuple[pd.DataFrame, int]:
    """Aggregate by GO aspect (BP/MF/CC) at the protein level.

    Each protein contributes at most once per aspect.
    Requires an aspect_map (from OBO).
    """
    total_proteins = df["accession"].nunique()
    if not aspect_map:
        return (
            pd.DataFrame(columns=["aspect", "protein_count", "percent"]),
            total_proteins,
        )

    protein_to_aspects: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        acc = row.get("accession")
        aspects: Set[str] = set()
        for go_id in extract_go_ids(row.get("go", "")):
            asp = aspect_map.get(go_id)
            if asp:
                aspects.add(asp)
        if aspects:
            existing = protein_to_aspects.get(acc)
            if existing is None:
                protein_to_aspects[acc] = set(aspects)
            else:
                existing.update(aspects)

    aspect_counts: Dict[str, int] = {}
    for acc, aspects in protein_to_aspects.items():
        for a in aspects:
            aspect_counts[a] = aspect_counts.get(a, 0) + 1

    if not aspect_counts:
        return (
            pd.DataFrame(columns=["aspect", "protein_count", "percent"]),
            total_proteins,
        )

    order = ["BP", "MF", "CC"]
    df_counts = (
        pd.DataFrame(
            {
                "aspect": list(aspect_counts.keys()),
                "protein_count": list(aspect_counts.values()),
            }
        )
        .sort_values("protein_count", ascending=False)
        .reset_index(drop=True)
    )
    df_counts["percent"] = (df_counts["protein_count"] / total_proteins * 100.0).round(
        2
    )
    # Reorder to BP, MF, CC where present
    cat_type = pd.Categorical(df_counts["aspect"], categories=order, ordered=True)
    df_counts["aspect"] = cat_type
    df_counts = df_counts.sort_values(["aspect"]).reset_index(drop=True)
    return df_counts, total_proteins


def plot_go_aspects_bar(
    cat_df: pd.DataFrame,
    outdir: str,
    total_proteins: int,
    output_prefix: str = "lost_ucyna_go_aspects",
):
    os.makedirs(outdir, exist_ok=True)
    df = cat_df.copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.barplot(data=df, x="aspect", y="protein_count", ax=ax)
    ax.set_xlabel("GO aspect")
    ax.set_ylabel("Proteins annotated (Cyanothece, absent in UCYN-A)")
    for i, row in df.reset_index(drop=True).iterrows():
        ax.text(
            i,
            row["protein_count"] + max(df["protein_count"]) * 0.01,
            f"{row['protein_count']} ({row['percent']:.1f}%)",
            va="bottom",
            ha="center",
            fontsize=9,
        )
    plt.tight_layout()
    svg_path = os.path.join(outdir, f"{output_prefix}.svg")
    png_path = os.path.join(outdir, f"{output_prefix}.png")
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=300)
    print(f"Saved: {svg_path}\nSaved: {png_path}")


def plot_go_terms_treemap(
    terms_df: pd.DataFrame,
    outdir: str,
    use_go_names: bool = False,
    output_prefix: str = "lost_ucyna_go_terms_treemap",
):
    try:
        import squarify  # type: ignore
    except Exception:
        print("squarify not available; skipping GO treemap.")
        return
    os.makedirs(outdir, exist_ok=True)
    df = terms_df.copy()
    labels = []
    for _, row in df.iterrows():
        if use_go_names and isinstance(row.get("go_name"), str) and row["go_name"]:
            labels.append(shorten(f"{row['go_name']}\n{row['protein_count']}", 60))
        else:
            labels.append(f"{row['go_id']}\n{row['protein_count']}")
    fig, ax = plt.subplots(figsize=(10, 7))
    squarify.plot(
        sizes=df["protein_count"].tolist(),
        label=labels,
        alpha=0.85,
        color=sns.color_palette("Set3", n_colors=max(3, len(df))),
        ax=ax,
        text_kwargs={"fontsize": 10},
    )
    ax.axis("off")
    svg_path = os.path.join(outdir, f"{output_prefix}.svg")
    png_path = os.path.join(outdir, f"{output_prefix}.png")
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=300)
    print(f"Saved: {svg_path}\nSaved: {png_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize InterPro functional annotations for proteins present in Cyanothece but absent in UCYN-A."
    )
    parser.add_argument(
        "--input",
        default="results/ucyna_missing_interpro.tsv",
        help="Path to InterProScan TSV of Cyanothece proteins missing in UCYN-A.",
    )
    parser.add_argument(
        "--outdir",
        default="results/figures",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--top", type=int, default=20, help="Number of top terms to show."
    )
    parser.add_argument(
        "--min-count", type=int, default=1, help="Minimum proteins per term to include."
    )
    parser.add_argument(
        "--output-prefix",
        default="lost_ucyna_interpro_top",
        help="Output filename prefix (without extension).",
    )
    parser.add_argument(
        "--mode",
        choices=["terms", "categories"],
        default="categories",
        help="Visualization mode: raw terms or grouped categories (depends on --source).",
    )
    parser.add_argument(
        "--source",
        choices=["interpro", "go"],
        default="interpro",
        help="Which annotation source to visualize.",
    )
    parser.add_argument(
        "--go-obo",
        default=None,
        help="Path to go-basic.obo to map GO IDs to names and aspects (BP/MF/CC).",
    )
    parser.add_argument(
        "--use-go-names",
        action="store_true",
        help="Label GO terms with names (requires --go-obo).",
    )
    parser.add_argument(
        "--treemap",
        action="store_true",
        help="Also render a treemap for category mode (requires squarify).",
    )

    args = parser.parse_args()

    df = load_interpro_table(args.input)
    if args.source == "interpro":
        if args.mode == "terms":
            agg_df, total = aggregate_interpro_terms(df)
            if agg_df.empty:
                print("No valid InterPro annotations found to visualize.")
                return
            plot_top_interpro(
                agg_df=agg_df,
                outdir=args.outdir,
                total_proteins=total,
                top_n=args.top,
                min_count=args.min_count,
                output_prefix=args.output_prefix,
            )
        else:
            cat_df, total = aggregate_by_category(df)
            if cat_df.empty:
                print("No valid InterPro annotations found to visualize.")
                return
            plot_category_bar(
                cat_df=cat_df,
                outdir=args.outdir,
                total_proteins=total,
                top_n=min(args.top, len(cat_df)),
                output_prefix="lost_ucyna_interpro_categories",
            )
            if args.treemap:
                plot_category_treemap(cat_df=cat_df, outdir=args.outdir)
    else:
        # GO visualizations
        go_name_map, go_aspect_map = (
            load_go_mapping(args.go_obo) if args.go_obo else ({}, {})
        )
        if args.mode == "terms":
            go_terms_df, total = aggregate_go_terms(df, go_name_map, go_aspect_map)
            if go_terms_df.empty:
                print("No GO terms found to visualize.")
                return
            plot_top_go_terms(
                terms_df=go_terms_df,
                outdir=args.outdir,
                total_proteins=total,
                top_n=args.top,
                min_count=args.min_count,
                use_go_names=args.use_go_names and bool(go_name_map),
                output_prefix="lost_ucyna_go_top",
            )
            if args.treemap:
                plot_go_terms_treemap(
                    terms_df=go_terms_df.head(args.top),
                    outdir=args.outdir,
                    use_go_names=args.use_go_names and bool(go_name_map),
                )
        else:
            go_cat_df, total = aggregate_go_aspects(df, go_aspect_map)
            if go_cat_df.empty:
                print(
                    "No GO terms (with aspects) found to visualize. Provide --go-obo."
                )
                return
            plot_go_aspects_bar(
                cat_df=go_cat_df,
                outdir=args.outdir,
                total_proteins=total,
                output_prefix="lost_ucyna_go_aspects",
            )


if __name__ == "__main__":
    main()
