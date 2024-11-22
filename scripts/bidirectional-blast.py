#!/usr/bin/env python3

import os
from pathlib import Path
from Bio import SeqIO
from diamond4py import Diamond, OutFormat, Sensitivity
from multiprocessing import cpu_count
import pandas as pd
import tempfile


class BidirectionalBlast:
    def __init__(self, threads):
        """Initialize the BidirectionalBlast class.

        Args:
            threads (int): Number of threads to use for DIAMOND
        """
        self.threads = threads

    def _extract_proteins(self, gbk_file, output_fasta):
        """Extract protein sequences from GenBank file.

        Args:
            gbk_file (str): Path to input GenBank file
            output_fasta (str): Path to output FASTA file

        Returns:
            int: Number of sequences extracted
        """
        count = 0
        with open(output_fasta, "w") as fasta_out:
            for record in SeqIO.parse(gbk_file, "genbank"):
                for feature in record.features:
                    if feature.type == "CDS":
                        if "translation" in feature.qualifiers:
                            protein_id = feature.qualifiers.get(
                                "protein_id", [f"{record.id}_{count}"]
                            )[0]
                            translation = feature.qualifiers["translation"][0]
                            fasta_out.write(f">{protein_id}\n{translation}\n")
                            count += 1
        return count

    def run_diamond_blast(self, temp_dir, query_fasta, db_fasta, output_file):
        """Run DIAMOND BLAST for one direction.

        Args:
            temp_dir (str): Path to temporary directory
            query_fasta (str): Path to query FASTA file
            db_fasta (str): Path to database FASTA file
            output_file (str): Path to output file

        Returns:
            str: Path to output file
        """
        # Create DIAMOND database
        db_path = os.path.join(temp_dir, f"{Path(db_fasta).stem}.db")
        diamond = Diamond(database=f"{db_path}.dmnd", n_threads=self.threads)
        diamond.makedb(db_fasta)

        # Configure output format to include relevant metrics
        OutFormat.BLAST_TABULAR.with_extra_option(
            "qseqid",
            "sseqid",  # Query and subject sequence IDs
            "pident",  # Percentage identity
            "length",  # Alignment length
            "mismatch",  # Number of mismatches
            "gapopen",  # Number of gap openings
            "qstart",
            "qend",  # Query start and end positions
            "sstart",
            "send",  # Subject start and end positions
            "evalue",  # E-value
            "bitscore",  # Bit score
            "qlen",
            "slen",  # Query and subject lengths
        )

        # Run BLASTP
        diamond.blastp(
            query=query_fasta,
            out=output_file,
            outfmt=OutFormat.BLAST_TABULAR,
            iterate="sensitive",
            sensitivity=Sensitivity.ULTRA_SENSITIVE,
            max_target_seqs=1,  # Only keep top hit
        )

        return output_file

    def process_genomes(self, genome1_path, genome2_path, output_csv):
        """Process two genomes and perform bidirectional BLAST.

        Args:
            genome1_path (str): Path to first genome GenBank file
            genome2_path (str): Path to second genome GenBank file
            output_csv (str): Path to output CSV file
        """
        genome1_path = Path(genome1_path).resolve()
        genome1_name = genome1_path.stem
        genome2_path = Path(genome2_path).resolve()
        genome2_name = genome2_path.stem

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary FASTA files
            fasta1 = os.path.join(temp_dir, f"{genome1_name}.fasta")
            fasta2 = os.path.join(temp_dir, f"{genome2_name}.fasta")

            # Extract protein sequences
            print("Extracting protein sequences...")
            count1 = self._extract_proteins(genome1_path, fasta1)
            count2 = self._extract_proteins(genome2_path, fasta2)
            print(
                f"Extracted {count1} proteins from genome 1 and {count2} proteins from genome 2"
            )

            # Run DIAMOND in both directions
            print("Running DIAMOND BLAST genome1 -> genome2...")
            out1 = os.path.join(temp_dir, "blast1_out.tsv")
            self.run_diamond_blast(temp_dir, fasta1, fasta2, out1)

            print("Running DIAMOND BLAST genome2 -> genome1...")
            out2 = os.path.join(temp_dir, "blast2_out.tsv")
            self.run_diamond_blast(temp_dir, fasta2, fasta1, out2)

            # Read and process results
            print("Processing results...")
            columns = [
                "query_id",
                "subject_id",
                "percent_identity",
                "alignment_length",
                "mismatches",
                "gap_openings",
                "query_start",
                "query_end",
                "subject_start",
                "subject_end",
                "evalue",
                "bit_score",
                "query_length",
                "subject_length",
            ]

            df1 = pd.read_csv(out1, sep="\t", names=columns)
            df2 = pd.read_csv(out2, sep="\t", names=columns)

            # Add direction information
            df1["direction"] = f"{genome1_name} -> {genome2_name}"
            df2["direction"] = f"{genome2_name} -> {genome1_name}"

            # Combine results
            combined_df = pd.concat([df1, df2], ignore_index=True)

            # Calculate additional metrics
            combined_df["coverage"] = combined_df["alignment_length"] / combined_df[
                ["query_length", "subject_length"]
            ].max(axis=1)
            combined_df["normalized_bitscore"] = (
                combined_df["bit_score"] / combined_df["alignment_length"]
            )

            # Save results
            combined_df.to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}")


def main():
    """Main function to run the script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Perform bidirectional BLAST between two genomes using DIAMOND"
    )
    parser.add_argument("genome1", help="Path to first genome GenBank file")
    parser.add_argument("genome2", help="Path to second genome GenBank file")
    parser.add_argument(
        "--output",
        type=str,
        default="bidirectional-blast.csv",
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=cpu_count(),
        help="Number of threads to use (default: CPUs)",
    )

    args = parser.parse_args()

    blast = BidirectionalBlast(threads=args.threads)
    blast.process_genomes(args.genome1, args.genome2, args.output)


if __name__ == "__main__":
    main()
