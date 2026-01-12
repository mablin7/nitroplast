#!/usr/bin/env python3
"""
HMM-based uTP Homolog Search

This script uses hmmsearch with the uTP HMM profile to search for
uTP homologs in haptophyte proteomes.

This is the proper approach for Question A:
- Uses profile HMM built from aligned uTP sequences
- More sensitive than BLAST for detecting distant homologs
- Provides statistical framework for hit assessment

Author: Generated for nitroplast/UCYN-A research project
Date: January 2026
"""

import subprocess
import re
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
EXPERIMENT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
HAPTOPHYTE_DIR = DATA_DIR / "haptophytes" / "ncbi_dataset" / "data"

HMM_PROFILE = EXPERIMENT_DIR / "utp.hmm"
OUTPUT_DIR = EXPERIMENT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class HMMHit:
    """Container for hmmsearch hit."""

    target_name: str
    target_accession: str
    query_name: str
    full_evalue: float
    full_score: float
    full_bias: float
    domain_evalue: float
    domain_score: float
    domain_bias: float
    domain_num: int
    domain_of: int
    hmm_from: int
    hmm_to: int
    ali_from: int
    ali_to: int
    env_from: int
    env_to: int
    acc: float
    description: str


def parse_hmmsearch_domtblout(domtblout_path: Path) -> list[HMMHit]:
    """Parse hmmsearch domain table output."""
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
                    target_accession=parts[1],
                    query_name=parts[3],
                    full_evalue=float(parts[6]),
                    full_score=float(parts[7]),
                    full_bias=float(parts[8]),
                    domain_evalue=float(parts[12]),
                    domain_score=float(parts[13]),
                    domain_bias=float(parts[14]),
                    domain_num=int(parts[9]),
                    domain_of=int(parts[10]),
                    hmm_from=int(parts[15]),
                    hmm_to=int(parts[16]),
                    ali_from=int(parts[17]),
                    ali_to=int(parts[18]),
                    env_from=int(parts[19]),
                    env_to=int(parts[20]),
                    acc=float(parts[21]),
                    description=" ".join(parts[22:]) if len(parts) > 22 else "",
                )
                hits.append(hit)
            except (ValueError, IndexError) as e:
                continue

    return hits


def run_hmmsearch(
    hmm_path: Path,
    target_fasta: Path,
    output_prefix: Path,
    evalue: float = 10.0,  # Very permissive for initial search
) -> tuple[Path, Path]:
    """
    Run hmmsearch against a target proteome.

    Returns paths to standard output and domain table output.
    """
    stdout_path = output_prefix.with_suffix(".txt")
    domtblout_path = output_prefix.with_suffix(".domtblout")

    cmd = [
        "hmmsearch",
        "--domtblout",
        str(domtblout_path),
        "-E",
        str(evalue),
        "--domE",
        str(evalue),
        "-o",
        str(stdout_path),
        str(hmm_path),
        str(target_fasta),
    ]

    print(f"Running: {' '.join(cmd[:4])}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"hmmsearch error: {result.stderr}")
        raise RuntimeError("hmmsearch failed")

    return stdout_path, domtblout_path


def get_protein_lengths(fasta_path: Path) -> dict[str, int]:
    """Get lengths of all proteins in a FASTA file."""
    lengths = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        lengths[record.id] = len(record.seq)
    return lengths


def analyze_hmm_hits(
    hits: list[HMMHit], protein_lengths: dict[str, int], hmm_length: int = 713
) -> pd.DataFrame:
    """
    Analyze HMM hits for uTP-like characteristics.

    Key criteria:
    1. Hit should be at C-terminus of target protein
    2. HMM match should cover the uTP region (positions ~620-713 in HMM)
    3. Good domain score
    """
    records = []

    for hit in hits:
        target_len = protein_lengths.get(hit.target_name, 0)

        # Calculate position relative to C-terminus
        dist_from_c_term = target_len - hit.ali_to if target_len > 0 else -1

        # Check if HMM match covers uTP region (last ~100 positions of HMM)
        # The uTP motifs are in positions ~620-713 of the HMM
        utp_region_start = 620
        covers_utp_region = hit.hmm_to >= utp_region_start

        # HMM coverage
        hmm_coverage = (hit.hmm_to - hit.hmm_from + 1) / hmm_length

        records.append(
            {
                "target_name": hit.target_name,
                "target_length": target_len,
                "full_evalue": hit.full_evalue,
                "full_score": hit.full_score,
                "domain_evalue": hit.domain_evalue,
                "domain_score": hit.domain_score,
                "hmm_from": hit.hmm_from,
                "hmm_to": hit.hmm_to,
                "ali_from": hit.ali_from,
                "ali_to": hit.ali_to,
                "dist_from_c_term": dist_from_c_term,
                "is_c_terminal": dist_from_c_term < 50 and dist_from_c_term >= 0,
                "covers_utp_region": covers_utp_region,
                "hmm_coverage": hmm_coverage,
                "accuracy": hit.acc,
                "description": hit.description,
            }
        )

    return pd.DataFrame(records)


def plot_hmm_results(df: pd.DataFrame, output_path: Path, organism: str):
    """Plot HMM search results."""
    if df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Score distribution
    ax1 = axes[0, 0]
    ax1.hist(
        df["domain_score"], bins=30, color="steelblue", edgecolor="navy", alpha=0.7
    )
    ax1.set_xlabel("Domain Score (bits)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"HMM Score Distribution - {organism}")
    ax1.axvline(x=50, color="red", linestyle="--", label="Score threshold (50)")
    ax1.legend()

    # 2. E-value distribution (log scale)
    ax2 = axes[0, 1]
    evalues = df["domain_evalue"]
    evalues_log = [-1 * (e if e > 0 else 1e-300) for e in evalues]
    ax2.hist(
        [e for e in evalues if e > 0],
        bins=30,
        color="coral",
        edgecolor="darkred",
        alpha=0.7,
    )
    ax2.set_xlabel("Domain E-value")
    ax2.set_ylabel("Count")
    ax2.set_title("E-value Distribution")
    ax2.set_xscale("log")

    # 3. HMM position coverage
    ax3 = axes[1, 0]
    for _, row in df.iterrows():
        color = "coral" if row["covers_utp_region"] else "steelblue"
        ax3.plot(
            [row["hmm_from"], row["hmm_to"]],
            [row["domain_score"]] * 2,
            color=color,
            alpha=0.5,
            linewidth=2,
        )
    ax3.axvline(x=620, color="red", linestyle="--", label="uTP region start (~620)")
    ax3.set_xlabel("HMM Position")
    ax3.set_ylabel("Domain Score")
    ax3.set_title("HMM Coverage vs Score")
    ax3.legend()

    # 4. C-terminal position
    ax4 = axes[1, 1]
    c_term_data = df[df["dist_from_c_term"] >= 0]["dist_from_c_term"]
    if len(c_term_data) > 0:
        ax4.hist(c_term_data, bins=30, color="steelblue", edgecolor="navy", alpha=0.7)
        ax4.axvline(
            x=50, color="red", linestyle="--", label="C-terminal threshold (50 aa)"
        )
        ax4.set_xlabel("Distance from C-terminus (aa)")
        ax4.set_ylabel("Count")
        ax4.set_title("Hit Position Relative to C-terminus")
        ax4.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 70)
    print("HMM-based uTP Homolog Search")
    print("=" * 70)

    # Verify HMM profile exists
    if not HMM_PROFILE.exists():
        print(f"ERROR: HMM profile not found at {HMM_PROFILE}")
        return

    print(f"\nUsing HMM profile: {HMM_PROFILE}")

    # Get metadata
    summary_file = HAPTOPHYTE_DIR / "data_summary.tsv"
    metadata = pd.read_csv(summary_file, sep="\t")

    # Find genomes with protein annotations
    genomes = []
    for _, row in metadata.iterrows():
        accession = row["Assembly Accession"]
        organism = row["Organism Scientific Name"]
        genome_dir = HAPTOPHYTE_DIR / accession

        if not genome_dir.exists():
            continue

        protein_file = genome_dir / "protein.faa"
        if protein_file.exists():
            genomes.append((accession, organism, protein_file))

    print(f"\nFound {len(genomes)} genomes with protein annotations")

    # Run hmmsearch on each genome
    all_results = []

    for accession, organism, protein_file in genomes:
        print(f"\n{'='*50}")
        print(f"Searching: {organism} ({accession})")
        print("=" * 50)

        output_prefix = OUTPUT_DIR / f"hmmsearch_{accession}"

        try:
            # Run hmmsearch
            stdout_path, domtblout_path = run_hmmsearch(
                HMM_PROFILE, protein_file, output_prefix, evalue=10.0
            )

            # Parse results
            hits = parse_hmmsearch_domtblout(domtblout_path)
            print(f"  Raw hits: {len(hits)}")

            if not hits:
                print("  No hits found")
                continue

            # Get protein lengths
            protein_lengths = get_protein_lengths(protein_file)

            # Analyze hits
            df = analyze_hmm_hits(hits, protein_lengths)
            df["organism"] = organism
            df["accession"] = accession

            # Filter for significant hits
            significant = df[df["domain_evalue"] < 0.01]
            c_terminal = df[df["is_c_terminal"]]
            utp_region = df[df["covers_utp_region"]]

            print(f"  Significant hits (E<0.01): {len(significant)}")
            print(f"  C-terminal hits (<50 aa from end): {len(c_terminal)}")
            print(f"  Hits covering uTP region: {len(utp_region)}")

            # The key question: are there C-terminal hits covering the uTP region?
            key_hits = df[(df["is_c_terminal"]) & (df["covers_utp_region"])]
            print(f"  *** C-terminal + uTP region hits: {len(key_hits)} ***")

            if len(key_hits) > 0:
                print("\n  TOP KEY HITS:")
                for _, hit in key_hits.head(5).iterrows():
                    print(
                        f"    {hit['target_name']}: score={hit['domain_score']:.1f}, "
                        f"E={hit['domain_evalue']:.2e}, HMM={hit['hmm_from']}-{hit['hmm_to']}"
                    )

            all_results.append(df)

            # Plot results
            if len(df) > 0:
                plot_hmm_results(
                    df, OUTPUT_DIR / f"hmm_results_{accession}.png", organism
                )

        except Exception as e:
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()

    # Combine all results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / "hmmsearch_all_results.csv", index=False)

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        print(f"\nTotal hits across all genomes: {len(combined)}")

        # Key metrics
        significant = combined[combined["domain_evalue"] < 0.01]
        c_terminal = combined[combined["is_c_terminal"]]
        utp_region = combined[combined["covers_utp_region"]]
        key_hits = combined[
            (combined["is_c_terminal"]) & (combined["covers_utp_region"])
        ]

        print(f"Significant hits (E<0.01): {len(significant)}")
        print(f"C-terminal hits: {len(c_terminal)}")
        print(f"Hits covering uTP region (HMM pos >620): {len(utp_region)}")
        print(f"\n*** KEY: C-terminal + uTP region hits: {len(key_hits)} ***")

        if len(key_hits) > 0:
            print("\nThese are potential uTP homologs!")
            key_hits_sorted = key_hits.sort_values("domain_score", ascending=False)
            key_hits_sorted.to_csv(
                OUTPUT_DIR / "potential_utp_homologs.csv", index=False
            )

            print("\nTop candidates:")
            for _, hit in key_hits_sorted.head(10).iterrows():
                print(
                    f"  {hit['organism']}: {hit['target_name']}"
                    f"\n    Score={hit['domain_score']:.1f}, E={hit['domain_evalue']:.2e}"
                    f"\n    HMM pos: {hit['hmm_from']}-{hit['hmm_to']}, "
                    f"Protein pos: {hit['ali_from']}-{hit['ali_to']} "
                    f"(dist from C-term: {hit['dist_from_c_term']})"
                )
        else:
            print("\nNo C-terminal hits covering the uTP region found.")
            print("This strongly supports de novo evolution of uTP.")

        # By organism summary
        print("\n" + "-" * 50)
        print("HITS BY ORGANISM (E<0.01)")
        print("-" * 50)
        for org in combined["organism"].unique():
            org_hits = significant[significant["organism"] == org]
            org_key = key_hits[key_hits["organism"] == org]
            print(f"  {org}: {len(org_hits)} significant, {len(org_key)} key hits")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
