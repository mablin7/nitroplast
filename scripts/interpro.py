#!/usr/bin/env python3

import requests
import time
import json
from Bio import SeqIO
from io import StringIO
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import List, Dict, Optional
import os
from tqdm import tqdm


class InterProScanner:
    """Class to handle InterProScan protein annotation submissions and results retrieval."""

    def __init__(
        self,
        email: str,
        batch_size: int = 25,
        max_retries: int = 3,
        poll_interval: int = 60,
        timeout: int = 3600,
    ):
        """
        Initialize the InterProScanner.

        Args:
            email: Email address for InterProScan API
            batch_size: Number of sequences to submit in each batch
            max_retries: Maximum number of retry attempts for failed requests
            poll_interval: Time in seconds between status checks
            timeout: Maximum time in seconds to wait for results
        """
        self.email = email
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.poll_interval = poll_interval
        self.timeout = timeout

        # API endpoints
        self.base_url = "https://www.ebi.ac.uk/interpro/api/iprscan5"
        self.submit_url = f"{self.base_url}/run"

        # Setup logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def _prepare_fasta_batch(self, sequences: List[Dict[str, str]]) -> str:
        """Convert sequence dictionaries to FASTA format."""
        fasta_content = ""
        for seq in sequences:
            fasta_content += f">{seq['id']}\n{seq['sequence']}\n"
        return fasta_content

    def _submit_batch(self, fasta_content: str) -> Optional[str]:
        """Submit a batch of sequences to InterProScan."""
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }

        data = {
            "email": self.email,
            "sequence": fasta_content,
            "title": "Batch analysis",
            "goterms": "true",
            "pathways": "true",
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.submit_url, headers=headers, data=data)
                response.raise_for_status()
                return response.json()["jobId"]
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    self.logger.error(
                        f"Failed to submit batch after {self.max_retries} attempts"
                    )
                    return None
                time.sleep(10 * (attempt + 1))  # Exponential backoff

        return None

    def _check_status(self, job_id: str) -> tuple[bool, Optional[dict]]:
        """Check the status of a submitted job."""
        status_url = f"{self.base_url}/status/{job_id}"

        try:
            response = requests.get(status_url)
            response.raise_for_status()
            status = response.json()

            if status["status"] == "FAILURE":
                self.logger.error(
                    f"Job {job_id} failed: {status.get('message', 'Unknown error')}"
                )
                return True, None
            elif status["status"] == "FINISHED":
                results_url = f"{self.base_url}/result/{job_id}/json"
                results_response = requests.get(results_url)
                results_response.raise_for_status()
                return True, results_response.json()

            return False, None

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error checking status for job {job_id}: {str(e)}")
            return False, None

    def _process_results(self, results: dict) -> pd.DataFrame:
        """Process InterProScan results into a pandas DataFrame."""
        processed_results = []

        for result in results["results"]:
            protein_id = result["metadata"]["identifier"]

            # Process each match
            for match in result.get("matches", []):
                signature = match["signature"]

                entry_info = {}
                if "entry" in match:
                    entry_info = {
                        "interpro_id": match["entry"].get("accession"),
                        "entry_name": match["entry"].get("name"),
                        "entry_type": match["entry"].get("type"),
                    }

                # Get GO terms if available
                go_terms = []
                if "goXRefs" in match.get("entry", {}):
                    go_terms = [go["id"] for go in match["entry"]["goXRefs"]]

                # Get pathway information if available
                pathways = []
                if "pathwayXRefs" in match.get("entry", {}):
                    pathways = [p["id"] for p in match["entry"]["pathwayXRefs"]]

                processed_results.append(
                    {
                        "protein_id": protein_id,
                        "signature_id": signature.get("accession"),
                        "signature_name": signature.get("name"),
                        "signature_db": signature.get("signatureLibrary"),
                        "start": match.get("locations", [{}])[0].get("start"),
                        "end": match.get("locations", [{}])[0].get("end"),
                        "evalue": match.get("locations", [{}])[0].get("evalue"),
                        "score": match.get("locations", [{}])[0].get("score"),
                        **entry_info,
                        "go_terms": ",".join(go_terms) if go_terms else None,
                        "pathways": ",".join(pathways) if pathways else None,
                    }
                )

        return pd.DataFrame(processed_results)

    def submit_and_wait(
        self, sequences: List[Dict[str, str]], output_file: str = "interpro_results.csv"
    ) -> pd.DataFrame:
        """
        Submit sequences and wait for results.

        Args:
            sequences: List of dictionaries with 'id' and 'sequence' keys
            output_file: Path to save results CSV

        Returns:
            pandas DataFrame with annotation results
        """
        all_results = []

        # Split sequences into batches
        for i in tqdm(
            range(0, len(sequences), self.batch_size), desc="Processing batches"
        ):
            batch = sequences[i : i + self.batch_size]
            fasta_content = self._prepare_fasta_batch(batch)

            # Submit batch
            job_id = self._submit_batch(fasta_content)
            if not job_id:
                continue

            # Wait for results
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                is_done, results = self._check_status(job_id)

                if is_done:
                    if results:
                        batch_df = self._process_results(results)
                        all_results.append(batch_df)
                    break

                time.sleep(self.poll_interval)
            else:
                self.logger.error(f"Timeout waiting for results of job {job_id}")

        # Combine all results
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            final_df.to_csv(output_file, index=False)
            return final_df
        else:
            return pd.DataFrame()


def read_sequences_from_fasta(fasta_file: str) -> List[Dict[str, str]]:
    """Read sequences from a FASTA file."""
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append({"id": record.id, "sequence": str(record.seq)})
    return sequences


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Submit protein sequences to InterProScan"
    )
    parser.add_argument("fasta_file", help="Input FASTA file with protein sequences")
    parser.add_argument("email", help="Email address for InterProScan submission")
    parser.add_argument(
        "--output",
        default="interpro_results.csv",
        help="Output CSV file (default: interpro_results.csv)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of sequences per batch (default: 25)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between status checks (default: 60)",
    )

    args = parser.parse_args()

    # Read sequences
    sequences = read_sequences_from_fasta(args.fasta_file)

    # Initialize scanner
    scanner = InterProScanner(
        email=args.email, batch_size=args.batch_size, poll_interval=args.poll_interval
    )

    # Submit sequences and get results
    results_df = scanner.submit_and_wait(sequences, args.output)

    print(f"\nResults saved to {args.output}")
    print(f"Total annotations found: {len(results_df)}")
    print("\nSummary of annotation sources:")
    print(results_df["signature_db"].value_counts())
