#!/usr/bin/env python3
import subprocess
import pandas as pd
from Bio import SeqIO
import tempfile
import os
import logging


class ProteinExtensionFinder:
    def __init__(
        self,
        query_fasta,
        target_fasta,
        min_identity=70,
        min_extension=50,
        evalue_threshold=1e-10,
        min_subject_coverage=0.9,
    ):
        self.query_fasta = query_fasta
        self.target_fasta = target_fasta
        self.min_identity = min_identity
        self.min_extension = min_extension
        self.evalue_threshold = evalue_threshold
        self.min_subject_coverage = min_subject_coverage

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Create temporary directory for DIAMOND files
        self.temp_dir = tempfile.mkdtemp()
        self.target_db = os.path.join(self.temp_dir, "target_db")

    def create_diamond_db(self):
        """Create DIAMOND database from target sequences"""
        self.logger.info("Creating DIAMOND database...")
        cmd = ["diamond", "makedb", "--in", self.target_fasta, "-d", self.target_db]
        subprocess.run(cmd, check=True)

    def run_diamond_search(
        self,
        query_file,
        db_file,
        output_file,
        evalue_threshold=None,
        very_sensitive=True,
    ):
        """Run DIAMOND search with specified parameters"""
        cmd = [
            "diamond",
            "blastp",
            "-d",
            db_file,
            "-q",
            query_file,
            "-o",
            output_file,
            "--outfmt",
            "6",
            "qseqid",
            "sseqid",
            "pident",
            "length",
            "mismatch",
            "gapopen",
            "qstart",
            "qend",
            "qlen",
            "sstart",
            "send",
            "slen",
            "evalue",
            "bitscore",
            "cigar",
            "--very-sensitive" if very_sensitive else "--sensitive",
            "-e",
            str(evalue_threshold or self.evalue_threshold),
        ]
        subprocess.run(cmd, check=True)

    def filter_hits(self, diamond_results):
        """Filter DIAMOND results based on criteria"""
        df = pd.read_csv(
            diamond_results,
            sep="\t",
            header=None,
            names=[
                "qseqid",
                "sseqid",
                "pident",
                "length",
                "mismatch",
                "gapopen",
                "qstart",
                "qend",
                "qlen",
                "sstart",
                "send",
                "slen",
                "evalue",
                "bitscore",
                "cigar",
            ],
        )

        base_matches = (df["send"] - df["sstart"] + 1) - df[
            "mismatch"
        ]  # how many identical matches there are in the base
        df["base_ident"] = (
            base_matches / df["slen"]
        ) * 100  # what percentage of the base is identical

        # Apply filters
        filtered = df[
            (df["base_ident"] >= 0)  # Min identity
            & (df["qstart"] <= df["qlen"] * 0.1)  # Start in first 10%
            & (
                (df["send"] - df["sstart"] + 1) / df["slen"]
                >= self.min_subject_coverage
            )  # Subject coverage - what percentage of the subject is covered by the query
            & (df["qlen"] - df["qend"] >= self.min_extension)  # C-terminal extension
            & (
                df["qlen"] - df["slen"] >= self.min_extension
            )  # Query longer than subject
        ]

        # Keep best hit per query based on bitscore
        best_hits = filtered.sort_values("bitscore", ascending=False).drop_duplicates(
            subset=["qseqid"], keep="first"
        )

        if len(best_hits) == 0:
            raise ValueError("No hits found")

        return best_hits

    def extract_extensions(self, best_hits, output_fasta):
        """Extract C-terminal extensions from filtered hits"""
        # Read all query sequences
        query_records = {
            record.id: record for record in SeqIO.parse(self.query_fasta, "fasta")
        }

        with open(output_fasta, "w") as out:
            for _, row in best_hits.iterrows():
                if row["qseqid"] in query_records:
                    record = query_records[row["qseqid"]]
                    # Extract C-terminal extension
                    extension = record.seq[row["qend"] :]
                    # Write extension to file
                    out.write(f">{record.id}_extension\n{extension}\n")

    def validate_extensions(self, extensions_fasta):
        """Validate extensions by searching against target database"""
        # Run DIAMOND search for extensions
        extension_results = os.path.join(self.temp_dir, "extensions_vs_target.tsv")
        self.run_diamond_search(
            extensions_fasta,
            self.target_db,
            extension_results,
            evalue_threshold=1e-10,
            very_sensitive=False,
        )

        # Read results
        if os.path.getsize(extension_results) > 0:
            df = pd.read_csv(extension_results, sep="\t", header=None)
            # Get queries with significant extension hits
            excluded = set(df[0].str.replace("_extension", ""))
            logging.info(f"Excluded {len(excluded)} extensions")
            return excluded
        return set()

    def find_extensions(self, output_dir):
        """Main workflow to find C-terminal extensions"""
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Create DIAMOND database
            self.create_diamond_db()

            # Run initial DIAMOND search
            initial_results = os.path.join(self.temp_dir, "initial_search.tsv")
            self.run_diamond_search(self.query_fasta, self.target_db, initial_results)

            # Filter hits
            best_hits = self.filter_hits(initial_results)

            # Extract extensions
            extensions_fasta = os.path.join(output_dir, "extensions.fasta")
            self.extract_extensions(best_hits, extensions_fasta)

            # Validate extensions
            excluded = self.validate_extensions(extensions_fasta)

            # Filter out queries with homologous extensions
            final_hits = best_hits[~best_hits["qseqid"].isin(excluded)]

            # Save results
            final_hits.to_csv(
                os.path.join(output_dir, "final_candidates.tsv"), sep="\t", index=False
            )

            # Extract final candidate sequences
            query_records = SeqIO.parse(self.query_fasta, "fasta")
            final_records = (
                record
                for record in query_records
                if record.id in set(final_hits["qseqid"])
            )

            SeqIO.write(
                final_records,
                os.path.join(output_dir, "final_candidates.fasta"),
                "fasta",
            )

            self.logger.info(
                f"Found {len(final_hits)} candidates with "
                f"C-terminal extensions ≥{self.min_extension} aa"
            )

        finally:
            # Cleanup temporary files
            import shutil

            shutil.rmtree(self.temp_dir)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Find proteins with C-terminal extensions"
    )
    parser.add_argument("query_fasta", help="Query protein sequences")
    parser.add_argument("target_fasta", help="Target protein sequences")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument(
        "--min-identity",
        type=float,
        default=70,
        help="Minimum percent identity (default: 70)",
    )
    parser.add_argument(
        "--min-extension",
        type=int,
        default=50,
        help="Minimum extension length (default: 50)",
    )
    parser.add_argument(
        "--evalue", type=float, default=1e-10, help="E-value threshold (default: 1e-10)"
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.9,
        help="Minimum subject coverage (default: 0.9)",
    )

    args = parser.parse_args()

    finder = ProteinExtensionFinder(
        args.query_fasta,
        args.target_fasta,
        min_identity=args.min_identity,
        min_extension=args.min_extension,
        evalue_threshold=args.evalue,
        min_subject_coverage=args.coverage,
    )

    finder.find_extensions(args.output_dir)


if __name__ == "__main__":
    main()
