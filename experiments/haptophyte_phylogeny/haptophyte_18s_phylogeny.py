#!/usr/bin/env python3
"""
Haptophyte 18S rRNA Phylogeny Analysis

This script builds a phylogenetic tree of available haptophyte genomes using 18S rRNA sequences
to assess evolutionary relationships and estimate divergence times relative to B. bigelowii.

Steps:
1. Extract organism info from downloaded genome metadata
2. Fetch 18S rRNA sequences from NCBI for each species
3. Align sequences using MUSCLE
4. Build neighbor-joining tree
5. Estimate divergence times using molecular clock

Author: Generated for nitroplast/UCYN-A research project
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from io import StringIO
from collections import defaultdict

from Bio import Entrez, SeqIO, Phylo
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio.Align import MultipleSeqAlignment
import matplotlib.pyplot as plt
import pandas as pd

# Set email for NCBI Entrez (required)
Entrez.email = "haptophyte_phylogeny@research.local"

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
EXPERIMENT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "haptophytes"
GENOME_DATA_DIR = DATA_DIR / "ncbi_dataset" / "data"
MUSCLE_BIN = PROJECT_ROOT / "muscle-osx-arm64.v5.3"

# Output paths
OUTPUT_DIR = EXPERIMENT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_genome_metadata():
    """Load genome metadata from NCBI dataset files."""
    summary_file = GENOME_DATA_DIR / "data_summary.tsv"

    df = pd.read_csv(summary_file, sep="\t")
    print(f"Loaded metadata for {len(df)} genome assemblies")

    # Deduplicate by taxonomy ID, keeping one per species
    # Group by taxonomy id and keep most complete genome
    df_unique = df.sort_values("Gene Count", ascending=False).drop_duplicates(
        subset=["Taxonomy id"], keep="first"
    )

    print(f"Unique species: {len(df_unique)}")
    return df_unique


def load_bigelowii_18s():
    """Load the B. bigelowii 18S rRNA sequence from local file."""
    gb_file = DATA_DIR / "b_bigellow_18s_rrna.gb"

    record = SeqIO.read(gb_file, "genbank")
    print(f"Loaded B. bigelowii 18S: {len(record.seq)} bp")

    return SeqRecord(
        record.seq, id="Braarudosphaera_bigelowii", description="18S rRNA (reference)"
    )


def fetch_18s_from_ncbi(taxid, organism_name, max_results=5):
    """
    Fetch 18S rRNA sequence for a given taxonomy ID from NCBI.

    Returns the longest sequence found, or None if not available.
    """
    # Clean up organism name for search
    search_name = organism_name.split()[0]  # Use genus only for broader search
    if "uncultured" in organism_name.lower():
        search_name = organism_name.replace("uncultured ", "")

    # Search strategies (try multiple)
    search_queries = [
        f"txid{taxid}[Organism] AND 18S[Title] AND rRNA[Title]",
        f'txid{taxid}[Organism] AND "18S ribosomal RNA"[Title]',
        f"txid{taxid}[Organism] AND SSU[Title] AND rRNA",
        f'"{organism_name}"[Organism] AND 18S AND rRNA',
    ]

    for query in search_queries:
        try:
            handle = Entrez.esearch(db="nucleotide", term=query, retmax=max_results)
            results = Entrez.read(handle)
            handle.close()

            if results["IdList"]:
                # Fetch sequences
                ids = results["IdList"]
                handle = Entrez.efetch(
                    db="nucleotide", id=ids, rettype="fasta", retmode="text"
                )
                records = list(SeqIO.parse(handle, "fasta"))
                handle.close()

                if records:
                    # Return longest sequence (most likely complete)
                    longest = max(records, key=lambda r: len(r.seq))
                    # Filter for reasonable 18S length (1500-2000bp typically)
                    if 1000 < len(longest.seq) < 3000:
                        print(f"  Found 18S for {organism_name}: {len(longest.seq)} bp")
                        return longest

        except Exception as e:
            continue

    print(f"  No 18S found for {organism_name} (taxid: {taxid})")
    return None


def fetch_all_18s_sequences(metadata_df):
    """Fetch 18S sequences for all species in the metadata."""
    sequences = []

    # Load B. bigelowii reference
    bb_seq = load_bigelowii_18s()
    sequences.append(bb_seq)

    print("\nFetching 18S rRNA sequences from NCBI...")

    for _, row in metadata_df.iterrows():
        taxid = row["Taxonomy id"]
        organism = row["Organism Scientific Name"]

        # Skip duplicate E. huxleyi entries
        if (
            any(s.id.startswith("Emiliania") for s in sequences)
            and "Emiliania" in organism
        ):
            continue

        seq = fetch_18s_from_ncbi(taxid, organism)
        if seq:
            # Clean up ID
            clean_name = organism.replace(" ", "_").replace(".", "")
            seq.id = clean_name[:40]  # Truncate long names
            seq.description = ""
            sequences.append(seq)

    return sequences


def search_related_haptophytes():
    """
    Search for additional well-characterized haptophyte 18S sequences
    to provide better phylogenetic context.
    """
    additional_species = [
        ("Pavlova lutheri", 2830),  # Pavlovophyceae
        ("Phaeocystis globosa", 33651),  # Prymnesiophyceae
        ("Prymnesium polylepis", 97484),  # Toxic bloom species
        ("Gephyrocapsa oceanica", 38815),  # Close to E. huxleyi
        ("Calcidiscus leptoporus", 66481),  # Coccolithophore
        ("Isochrysis galbana", 37099),  # Common model
    ]

    sequences = []
    print("\nSearching for additional reference species...")

    for name, taxid in additional_species:
        seq = fetch_18s_from_ncbi(taxid, name)
        if seq:
            clean_name = name.replace(" ", "_")
            seq.id = clean_name
            seq.description = ""
            sequences.append(seq)

    return sequences


def write_fasta(sequences, output_path):
    """Write sequences to FASTA file."""
    SeqIO.write(sequences, output_path, "fasta")
    print(f"Wrote {len(sequences)} sequences to {output_path}")


def run_muscle_alignment(input_fasta, output_fasta):
    """Run MUSCLE alignment on sequences."""
    print(f"\nRunning MUSCLE alignment...")

    cmd = [str(MUSCLE_BIN), "-align", str(input_fasta), "-output", str(output_fasta)]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"MUSCLE error: {result.stderr}")
        raise RuntimeError("MUSCLE alignment failed")

    print(f"Alignment complete: {output_fasta}")
    return output_fasta


def build_phylogenetic_tree(alignment_file):
    """Build neighbor-joining tree from alignment."""
    print("\nBuilding phylogenetic tree...")

    # Read alignment
    alignment = list(SeqIO.parse(alignment_file, "fasta"))

    # Convert to MultipleSeqAlignment
    msa = MultipleSeqAlignment(
        [SeqRecord(Seq(str(rec.seq)), id=rec.id, description="") for rec in alignment]
    )

    # Calculate distance matrix
    calculator = DistanceCalculator("identity")
    dm = calculator.get_distance(msa)

    # Build tree using neighbor-joining
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(dm)

    # Also build UPGMA for comparison
    upgma_tree = constructor.upgma(dm)

    return tree, upgma_tree, dm


def calculate_divergence_times(distance_matrix, tree):
    """
    Estimate divergence times using molecular clock assumption.

    Uses commonly cited 18S rRNA substitution rates for eukaryotes:
    ~0.5-1% per 100 Myr (we use 0.75% = 0.0075 substitutions/site/100Myr)

    Note: This is a rough estimate - proper dating requires fossil calibration.
    """
    # 18S rRNA substitution rate: ~0.75% per 100 Myr (conservative estimate)
    # This is based on various eukaryote calibrations
    substitution_rate = 0.0075 / 100  # per site per Myr

    # Get B. bigelowii index in distance matrix
    bb_idx = None
    for i, name in enumerate(distance_matrix.names):
        if "Braarudosphaera" in name or "bigelowii" in name.lower():
            bb_idx = i
            break

    if bb_idx is None:
        print("Warning: Could not find B. bigelowii in distance matrix")
        return {}

    # Calculate divergence times from B. bigelowii
    divergence_times = {}
    for i, name in enumerate(distance_matrix.names):
        if i != bb_idx:
            distance = distance_matrix[bb_idx, i]
            # Time = distance / (2 * rate)  [factor of 2 for two lineages diverging]
            time_myr = distance / (2 * substitution_rate)
            divergence_times[name] = {"distance": distance, "divergence_myr": time_myr}

    return divergence_times


def plot_tree(tree, output_path, title="Haptophyte 18S rRNA Phylogeny"):
    """Generate publication-quality tree figure."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw tree
    Phylo.draw(tree, axes=ax, do_show=False)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Substitutions per site")

    # Highlight B. bigelowii
    for text in ax.texts:
        if (
            "Braarudosphaera" in text.get_text()
            or "bigelowii" in text.get_text().lower()
        ):
            text.set_fontweight("bold")
            text.set_color("darkred")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    print(f"Saved tree figure: {output_path}")
    plt.close()


def plot_divergence_times(divergence_times, output_path):
    """Plot divergence time estimates."""
    if not divergence_times:
        return

    # Sort by divergence time
    sorted_times = sorted(
        divergence_times.items(), key=lambda x: x[1]["divergence_myr"]
    )

    names = [name.replace("_", " ") for name, _ in sorted_times]
    times = [data["divergence_myr"] for _, data in sorted_times]

    fig, ax = plt.subplots(figsize=(10, 8))

    bars = ax.barh(names, times, color="steelblue", edgecolor="navy")

    ax.set_xlabel("Estimated Divergence Time from B. bigelowii (Myr)", fontsize=12)
    ax.set_title(
        "Haptophyte Divergence Time Estimates\n(18S rRNA molecular clock)", fontsize=14
    )

    # Add value labels
    for bar, time in zip(bars, times):
        ax.text(
            bar.get_width() + max(times) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{time:.0f} Myr",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(0, max(times) * 1.15)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    print(f"Saved divergence time plot: {output_path}")
    plt.close()


def save_results(divergence_times, distance_matrix, output_dir):
    """Save results to CSV files."""
    # Save divergence times
    if divergence_times:
        df = pd.DataFrame(
            [
                {
                    "Species": name,
                    "Genetic_Distance": data["distance"],
                    "Divergence_Myr": data["divergence_myr"],
                }
                for name, data in divergence_times.items()
            ]
        )
        df = df.sort_values("Divergence_Myr")
        df.to_csv(output_dir / "divergence_times.csv", index=False)
        print(f"\nSaved divergence times to {output_dir / 'divergence_times.csv'}")

    # Save distance matrix
    dm_df = pd.DataFrame(
        [
            [distance_matrix[i, j] for j in range(len(distance_matrix))]
            for i in range(len(distance_matrix))
        ],
        index=distance_matrix.names,
        columns=distance_matrix.names,
    )
    dm_df.to_csv(output_dir / "distance_matrix.csv")
    print(f"Saved distance matrix to {output_dir / 'distance_matrix.csv'}")


def main():
    print("=" * 60)
    print("Haptophyte 18S rRNA Phylogeny Analysis")
    print("=" * 60)

    # Load genome metadata
    metadata_df = load_genome_metadata()
    print(f"\nSpecies in dataset:")
    for _, row in metadata_df.iterrows():
        print(f"  - {row['Organism Scientific Name']}")

    # Fetch 18S sequences
    sequences = fetch_all_18s_sequences(metadata_df)

    # Add reference species for better context
    additional = search_related_haptophytes()
    # Only add if not already present
    existing_ids = {s.id for s in sequences}
    for seq in additional:
        if seq.id not in existing_ids:
            sequences.append(seq)

    if len(sequences) < 4:
        print("\nWarning: Too few sequences for meaningful phylogeny")
        print("Attempting broader search for haptophyte 18S sequences...")

        # Broader search
        handle = Entrez.esearch(
            db="nucleotide",
            term='Haptophyta[Organism] AND "18S ribosomal RNA"[Title] AND 1500:2000[Sequence Length]',
            retmax=30,
        )
        results = Entrez.read(handle)
        handle.close()

        if results["IdList"]:
            handle = Entrez.efetch(
                db="nucleotide", id=results["IdList"], rettype="fasta", retmode="text"
            )
            records = list(SeqIO.parse(handle, "fasta"))
            handle.close()

            seen_species = {s.id.split("_")[0] for s in sequences}
            for rec in records:
                # Parse organism from description
                parts = rec.description.split()
                if len(parts) >= 2:
                    genus = parts[0].replace("_", "")
                    if genus not in seen_species:
                        rec.id = f"{parts[0]}_{parts[1]}"[:40]
                        rec.description = ""
                        sequences.append(rec)
                        seen_species.add(genus)
                        print(f"  Added: {rec.id}")

    print(f"\nTotal sequences for analysis: {len(sequences)}")

    if len(sequences) < 3:
        print("ERROR: Need at least 3 sequences for phylogeny. Exiting.")
        sys.exit(1)

    # Write sequences to FASTA
    unaligned_fasta = OUTPUT_DIR / "18s_sequences_unaligned.fasta"
    write_fasta(sequences, unaligned_fasta)

    # Run alignment
    aligned_fasta = OUTPUT_DIR / "18s_sequences_aligned.fasta"
    run_muscle_alignment(unaligned_fasta, aligned_fasta)

    # Build tree
    nj_tree, upgma_tree, distance_matrix = build_phylogenetic_tree(aligned_fasta)

    # Save trees in Newick format
    Phylo.write(nj_tree, OUTPUT_DIR / "phylogeny_nj.nwk", "newick")
    Phylo.write(upgma_tree, OUTPUT_DIR / "phylogeny_upgma.nwk", "newick")
    print(f"Saved trees to {OUTPUT_DIR}")

    # Calculate divergence times
    divergence_times = calculate_divergence_times(distance_matrix, nj_tree)

    # Generate figures
    plot_tree(
        nj_tree,
        OUTPUT_DIR / "phylogeny_nj.png",
        "Haptophyte 18S rRNA Phylogeny (Neighbor-Joining)",
    )
    plot_tree(
        upgma_tree,
        OUTPUT_DIR / "phylogeny_upgma.png",
        "Haptophyte 18S rRNA Phylogeny (UPGMA)",
    )

    if divergence_times:
        plot_divergence_times(divergence_times, OUTPUT_DIR / "divergence_times.png")

    # Save results
    save_results(divergence_times, distance_matrix, OUTPUT_DIR)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nSequences analyzed: {len(sequences)}")

    if divergence_times:
        print("\nEstimated divergence times from B. bigelowii:")
        print("-" * 45)
        for name, data in sorted(
            divergence_times.items(), key=lambda x: x[1]["divergence_myr"]
        ):
            print(f"  {name.replace('_', ' '):<35} {data['divergence_myr']:>6.0f} Myr")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
