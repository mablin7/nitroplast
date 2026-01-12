#!/usr/bin/env python3
"""Compute ProtT5 embeddings for additional proteins needed for terminal motif strategy."""

import warnings
warnings.filterwarnings('ignore')

from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
import h5py
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
import torch
from transformers import T5Tokenizer, T5EncoderModel

# Paths
SCRIPT_DIR = Path(__file__).parent
MEME_FILE = SCRIPT_DIR.parent / "utp_motif_analysis" / "data" / "meme_gb.xml"
SEQS_FILE = SCRIPT_DIR.parent / "utp_motif_analysis" / "output" / "good-c-term-full.fasta"
EXISTING_EMB_FILE = SCRIPT_DIR / "output" / "features" / "embeddings.h5"
OUTPUT_FILE = SCRIPT_DIR / "output" / "features" / "embeddings_all.h5"


def load_prott5_model():
    """Load ProtT5 model and tokenizer."""
    print("Loading ProtT5 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    
    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_uniref50", do_lower_case=False
    )
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, device


def embed_sequence(seq: str, model, tokenizer, device) -> np.ndarray:
    """Compute ProtT5 embedding for a single sequence."""
    # Prepare sequence (add spaces between amino acids)
    seq_spaced = " ".join(list(seq))
    
    # Tokenize
    inputs = tokenizer(
        seq_spaced,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get embedding
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling over sequence length
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    
    return embedding


def main():
    print("=" * 70)
    print("Computing Additional ProtT5 Embeddings")
    print("=" * 70)
    
    # Load motif data
    print("\n[1/5] Loading motif data...")
    meme_xml = ET.parse(MEME_FILE)
    seq_names = {
        tag.attrib["id"]: tag.attrib["name"]
        for tag in meme_xml.findall(".//sequence")
    }
    
    sites = meme_xml.findall(".//scanned_sites")
    sequences_motifs = {}
    for tag in sites:
        seq_id = tag.attrib["sequence_id"]
        if seq_id in seq_names:
            motifs = sorted(
                [s.attrib for s in tag.findall("scanned_site")],
                key=lambda m: int(m["position"])
            )
            sequences_motifs[seq_names[seq_id]] = tuple(
                m["motif_id"] for m in motifs
            )
    
    # Load sequences
    seqs = {s.id: str(s.seq) for s in SeqIO.parse(SEQS_FILE, "fasta")}
    print(f"  Loaded {len(seqs)} sequences")
    
    # Get terminal motif grouping
    def get_variant_name(motif_ids):
        return "+".join(m.replace("motif_", "") for m in motif_ids)
    
    def get_terminal_group(variant_name):
        if not variant_name:
            return "none"
        parts = variant_name.split("+")
        terminal = parts[-1] if parts else "none"
        if terminal in ["4", "5", "7", "9"]:
            return f"terminal_{terminal}"
        return f"terminal_{terminal}"
    
    # Filter to main terminal groups
    main_terminals = {"terminal_4", "terminal_5", "terminal_7", "terminal_9"}
    terminal_variants = {}
    for name, motifs in sequences_motifs.items():
        variant = get_variant_name(motifs)
        if variant:
            terminal = get_terminal_group(variant)
            if terminal in main_terminals:
                terminal_variants[name] = terminal
    
    print(f"  Proteins for terminal motif strategy: {len(terminal_variants)}")
    
    # Load existing embeddings
    print("\n[2/5] Loading existing embeddings...")
    existing_embeddings = {}
    if EXISTING_EMB_FILE.exists():
        with h5py.File(EXISTING_EMB_FILE, "r") as f:
            for name in f.keys():
                existing_embeddings[name] = f[name][:]
    print(f"  Loaded {len(existing_embeddings)} existing embeddings")
    
    # Identify proteins needing embeddings
    need_embeddings = set(terminal_variants.keys()) - set(existing_embeddings.keys())
    print(f"  Need to compute: {len(need_embeddings)} new embeddings")
    
    if not need_embeddings:
        print("  All embeddings already exist!")
        return
    
    # Load model
    print("\n[3/5] Loading ProtT5 model...")
    model, tokenizer, device = load_prott5_model()
    
    # Compute new embeddings
    print(f"\n[4/5] Computing {len(need_embeddings)} new embeddings...")
    new_embeddings = {}
    for name in tqdm(need_embeddings, desc="Computing embeddings"):
        if name in seqs:
            seq = seqs[name]
            # Clean sequence
            valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
            seq_clean = "".join(aa for aa in seq if aa in valid_aa)
            if len(seq_clean) > 0:
                emb = embed_sequence(seq_clean, model, tokenizer, device)
                new_embeddings[name] = emb
    
    print(f"  Computed {len(new_embeddings)} new embeddings")
    
    # Save all embeddings to new file
    print("\n[5/5] Saving all embeddings...")
    all_embeddings = {**existing_embeddings, **new_embeddings}
    
    with h5py.File(OUTPUT_FILE, "w") as f:
        for name, emb in all_embeddings.items():
            f.create_dataset(name, data=emb)
    
    print(f"  Saved {len(all_embeddings)} total embeddings to {OUTPUT_FILE}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Existing embeddings: {len(existing_embeddings)}")
    print(f"  New embeddings: {len(new_embeddings)}")
    print(f"  Total embeddings: {len(all_embeddings)}")
    
    # Show coverage for terminal motif strategy
    covered = sum(1 for n in terminal_variants if n in all_embeddings)
    print(f"\n  Terminal motif strategy coverage: {covered}/{len(terminal_variants)}")


if __name__ == "__main__":
    main()
