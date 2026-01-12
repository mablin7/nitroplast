#!/usr/bin/env python3
"""
Run OrthoFinder analysis on prepared haptophyte proteomes.

This script:
1. Sets up the environment with correct tool paths
2. Runs OrthoFinder on the prepared proteome files
3. Saves results to the output directory

Dependencies (must be installed):
- diamond (via homebrew)
- mafft (via homebrew)  
- VeryFastTree (via homebrew, used as FastTree replacement)
- mcl (built from source, in ~/local/bin)
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
ORTHOFINDER_DIR = PROJECT_ROOT / "tools" / "OrthoFinder"
PROTEOMES_DIR = Path(__file__).parent / "proteomes"
OUTPUT_DIR = Path(__file__).parent / "output"

# Tool paths
TOOL_PATHS = {
    "diamond": "/opt/homebrew/bin/diamond",
    "mafft": "/opt/homebrew/bin/mafft",
    "FastTree": "/opt/homebrew/bin/VeryFastTree",  # VeryFastTree is compatible
    "mcl": os.path.expanduser("~/local/bin/mcl"),
}


def check_dependencies():
    """Check that all required tools are available."""
    print("Checking dependencies...")
    missing = []
    
    for tool, path in TOOL_PATHS.items():
        if os.path.exists(path):
            print(f"  ✓ {tool}: {path}")
        else:
            print(f"  ✗ {tool}: NOT FOUND at {path}")
            missing.append(tool)
    
    if missing:
        print(f"\nError: Missing tools: {', '.join(missing)}")
        print("\nInstallation instructions:")
        print("  brew install diamond mafft veryfasttree")
        print("  # For MCL, build from source:")
        print("  git clone https://github.com/micans/mcl.git /tmp/mcl")
        print("  cd /tmp/mcl && ./install-this-mcl.sh ~/local")
        return False
    
    return True


def check_proteomes():
    """Check that proteome files exist."""
    print("\nChecking proteome files...")
    
    faa_files = list(PROTEOMES_DIR.glob("*.faa"))
    
    if not faa_files:
        print(f"  ✗ No .faa files found in {PROTEOMES_DIR}")
        print("  Run: uv run python experiments/ortholog_search/prepare_proteomes.py")
        return False
    
    print(f"  Found {len(faa_files)} proteome files:")
    for f in sorted(faa_files):
        # Count sequences
        with open(f) as fh:
            n_seqs = sum(1 for line in fh if line.startswith('>'))
        print(f"    - {f.name}: {n_seqs} proteins")
    
    return True


def setup_orthofinder_config():
    """
    Create a custom config for OrthoFinder with correct tool paths.
    """
    config = {
        "mafft": {
            "program_type": "msa",
            "cmd_line": f"{TOOL_PATHS['mafft']} --anysymbol --auto INPUT > OUTPUT"
        },
        "fasttree": {
            "program_type": "tree", 
            "cmd_line": f"{TOOL_PATHS['FastTree']} INPUT > OUTPUT"
        },
        "diamond": {
            "program_type": "search",
            "db_cmd": f"{TOOL_PATHS['diamond']} makedb --in INPUT -d OUTPUT",
            "search_cmd": f"{TOOL_PATHS['diamond']} blastp -d DATABASE -q INPUT -o OUTPUT --more-sensitive -p 1 --quiet -e 0.001 --compress 1"
        }
    }
    
    import json
    config_path = PROTEOMES_DIR / "orthofinder_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


def run_orthofinder(threads: int = 8, extra_args: list = None):
    """
    Run OrthoFinder on the proteome files.
    """
    if extra_args is None:
        extra_args = []
    
    # Add MCL to PATH
    env = os.environ.copy()
    mcl_dir = os.path.dirname(TOOL_PATHS['mcl'])
    env['PATH'] = f"{mcl_dir}:{env.get('PATH', '')}"
    
    # Also add homebrew bin
    env['PATH'] = f"/opt/homebrew/bin:{env['PATH']}"
    
    # Create symlink for FastTree if needed (OrthoFinder expects 'FastTree' not 'VeryFastTree')
    fasttree_link = PROTEOMES_DIR / "FastTree"
    if not fasttree_link.exists():
        os.symlink(TOOL_PATHS['FastTree'], fasttree_link)
    env['PATH'] = f"{PROTEOMES_DIR}:{env['PATH']}"
    
    # Build command
    cmd = [
        sys.executable,
        "-m", "scripts_of",
        "-f", str(PROTEOMES_DIR),
        "-t", str(threads),
        "-a", str(threads),
        "-S", "diamond",
        "-M", "msa",
        "-A", "mafft",
        "-T", "fasttree",
    ] + extra_args
    
    print(f"\nRunning OrthoFinder...")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Working directory: {ORTHOFINDER_DIR}")
    print(f"  Proteomes: {PROTEOMES_DIR}")
    print(f"  Threads: {threads}")
    print()
    
    # Run from OrthoFinder directory
    result = subprocess.run(
        cmd,
        cwd=ORTHOFINDER_DIR,
        env=env,
    )
    
    return result.returncode


def find_results():
    """Find the OrthoFinder results directory."""
    results_base = PROTEOMES_DIR / "OrthoFinder"
    
    if not results_base.exists():
        return None
    
    # Find most recent results
    result_dirs = sorted(results_base.glob("Results_*"), key=lambda x: x.stat().st_mtime)
    
    if result_dirs:
        return result_dirs[-1]
    
    return None


def summarize_results(results_dir: Path):
    """Print a summary of OrthoFinder results."""
    print(f"\n{'='*60}")
    print("ORTHOFINDER RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Results directory: {results_dir}")
    
    # Check key output files
    key_files = [
        "Orthogroups/Orthogroups.tsv",
        "Orthogroups/Orthogroups_SingleCopyOrthologues.txt",
        "Orthogroups/Orthogroups.GeneCount.tsv",
        "Comparative_Genomics_Statistics/Statistics_Overall.tsv",
        "Species_Tree/SpeciesTree_rooted.txt",
    ]
    
    print("\nKey output files:")
    for f in key_files:
        path = results_dir / f
        if path.exists():
            size = path.stat().st_size
            print(f"  ✓ {f} ({size:,} bytes)")
        else:
            print(f"  ✗ {f} (not found)")
    
    # Parse statistics if available
    stats_file = results_dir / "Comparative_Genomics_Statistics" / "Statistics_Overall.tsv"
    if stats_file.exists():
        print("\nStatistics:")
        with open(stats_file) as f:
            for line in f:
                if line.strip():
                    print(f"  {line.strip()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run OrthoFinder analysis")
    parser.add_argument("-t", "--threads", type=int, default=8,
                       help="Number of threads to use (default: 8)")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check dependencies, don't run analysis")
    parser.add_argument("--summarize", action="store_true",
                       help="Summarize existing results")
    parser.add_argument("extra_args", nargs="*",
                       help="Additional arguments to pass to OrthoFinder")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check proteomes
    if not check_proteomes():
        sys.exit(1)
    
    if args.check_only:
        print("\nAll checks passed!")
        sys.exit(0)
    
    if args.summarize:
        results_dir = find_results()
        if results_dir:
            summarize_results(results_dir)
        else:
            print("No results found. Run OrthoFinder first.")
        sys.exit(0)
    
    # Run OrthoFinder
    print(f"\n{'='*60}")
    print(f"Starting OrthoFinder analysis at {datetime.now()}")
    print(f"{'='*60}")
    
    returncode = run_orthofinder(threads=args.threads, extra_args=args.extra_args)
    
    if returncode == 0:
        print("\n✓ OrthoFinder completed successfully!")
        results_dir = find_results()
        if results_dir:
            summarize_results(results_dir)
    else:
        print(f"\n✗ OrthoFinder failed with exit code {returncode}")
        sys.exit(returncode)
