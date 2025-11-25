import os
import re
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from cyvcf2 import VCF
from pyfaidx import Fasta
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from dataclasses import dataclass

# Map RefSeq NC_* accessions to Ensembl-like chrom labels when possible
NC_TO_ENSEMBL_SPECIAL = {
    23: "X",
    24: "Y",
    12920: "MT",  # NC_012920.1 → mitochondrion
}


# ---------- ClinVar parsing (strict labels) ----------


def parse_clinvar_strict(vcf_path: str,
                         pos_label: str,
                         neg_label: str):
    """Parse ClinVar VCF and return (benign_vars, pathogenic_vars).

    Strict mode:
      - Keep only *Pathogenic* and *Benign*.
      - Drop Likely_* (both likely_pathogenic/likely_benign), Uncertain,
        Conflicting, and any mixed benign+pathogenic calls.
    """
    vcf = VCF(vcf_path)
    benign, pathogenic = [], []
    uncertain = []  # count but do not use

    def normalize_sig(val: str) -> List[str]:
        # Split on common separators and normalize.
        toks = re.split(r"[\|;,/]+", val)
        out = []
        for t in toks:
            t = t.strip().lower().replace(" ", "_").replace("-", "_")
            if not t:
                continue
            if "pathogenic" in t and "likely" not in t:
                out.append("pathogenic")
            elif "likely_pathogenic" in t:
                out.append("likely_pathogenic")
            elif "benign" in t and "likely" not in t:
                out.append("benign")
            elif "likely_benign" in t:
                out.append("likely_benign")
            elif "uncertain_significance" in t or t == "uncertain":
                out.append("uncertain_significance")
            elif "conflicting" in t:
                out.append("conflicting_interpretations_of_pathogenicity")
            else:
                out.append(t)
        return out

    for var in vcf:
        sig = var.INFO.get("CLNSIG")
        if not sig:
            continue

        toks = normalize_sig(sig)

        has_path = "pathogenic" in toks
        has_ben = "benign" in toks
        has_lpath = "likely_pathogenic" in toks
        has_lben = "likely_benign" in toks
        has_unc = "uncertain_significance" in toks
        has_conf = "conflicting_interpretations_of_pathogenicity" in toks

        # Drop uncertain/conflicting/likely*
        if has_unc or has_conf or has_lpath or has_lben:
            uncertain.append(var)
            continue

        # Drop mixed benign+pathogenic
        if has_path and has_ben:
            continue

        if has_path and not has_ben:
            pathogenic.append(var)
        elif has_ben and not has_path:
            benign.append(var)
        # anything else is ignored

    print(f"[ClinVar STRICT] benign={len(benign)}, pathogenic={len(pathogenic)}, "
          f"uncertain/other={len(uncertain)} (dropped)")
    return benign, pathogenic


# ---------- Reference genome + contig mapping ----------


def build_ref_genome(fasta_path: str):
    """Open reference genome FASTA with pyfaidx."""
    return Fasta(fasta_path)


def contig_candidates(chrom: str) -> List[str]:
    """Return plausible contig candidates across naming conventions.

    Handles inputs like: 'chr1', '1', 'NC_000001.11', 'chrX', 'X', 'MT', 'chrM'.
    """
    cands: List[str] = []

    # As-is
    cands.append(chrom)

    # Strip/add 'chr'
    if chrom.startswith("chr"):
        cands.append(chrom[3:])
    else:
        cands.append("chr" + chrom)

    # Map NC_0000XX.yy → 1..22/X/Y/MT
    if chrom.startswith("NC_"):
        m = re.match(r"NC_0*(\d+)\.\d+", chrom)
        if m:
            num = int(m.group(1))
            if 1 <= num <= 22:
                cands += [str(num), f"chr{num}"]
            elif num in NC_TO_ENSEMBL_SPECIAL:
                val = NC_TO_ENSEMBL_SPECIAL[num]
                cands += [val, f"chr{val}"]

    # Support chrM/chrMT → MT
    if chrom in ("chrM", "chrMT"):
        cands += ["MT"]

    # Deduplicate preserving order
    out = []
    for x in cands:
        if x not in out:
            out.append(x)
    return out


def extract_centered_ref_alt(ref_genome,
                             chrom: str,
                             pos_1based: int,
                             ref: str,
                             alt: str,
                             seq_len: int,
                             left_flank: int,
                             right_flank: int) -> Optional[Tuple[str, str]]:
    """Extract (ref_window, alt_window) of fixed length around pos_1based.

    The SNV sits at index `left_flank` in the window.
    Returns None if contig not found or window goes out of bounds.
    """
    fasta_contig = None
    for name in contig_candidates(chrom):
        if name in ref_genome:
            fasta_contig = name
            break
    if fasta_contig is None:
        return None

    # Skip unresolved scaffolds early
    if chrom.startswith(("NT_", "NW_")) and fasta_contig not in ref_genome:
        return None

    snv0 = pos_1based - 1
    start = snv0 - left_flank
    end = snv0 + right_flank + 1
    if start < 0 or end > len(ref_genome[fasta_contig]):
        return None

    ref_window = ref_genome[fasta_contig][start:end].seq.upper()
    if len(ref_window) != seq_len:
        return None

    center_idx = left_flank
    alt_window = ref_window[:center_idx] + alt.upper() + ref_window[center_idx + 1:]

    return ref_window, alt_window


def variants_to_dataframe(variants,
                          label: str,
                          ref_genome,
                          seq_len: int,
                          left_flank: int,
                          right_flank: int,
                          max_n: Optional[int] = None) -> pd.DataFrame:
    """Convert a list of variants into a DataFrame of ref/alt windows."""
    rows = []
    it = variants if max_n is None else variants[:max_n]

    for v in tqdm(it, desc=f"Building rows: {label}"):
        chrom, pos, ref, alts = v.CHROM, v.POS, v.REF, v.ALT
        if alts is None or len(alts) != 1:
            continue
        alt = alts[0]
        pair = extract_centered_ref_alt(ref_genome, chrom, pos, ref, alt,
                                        seq_len, left_flank, right_flank)
        if pair is None:
            continue
        ref_seq, alt_seq = pair
        if len(ref_seq) != seq_len or len(alt_seq) != seq_len:
            continue
        rows.append({
            "chrom": chrom,
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "label": label,
            "ref_seq": ref_seq,
            "alt_seq": alt_seq,
        })
    return pd.DataFrame(rows)


def build_or_load_paired_dataframe(cfg) -> pd.DataFrame:
    """Load precomputed ref/alt windows or build them from VCF+FASTA."""
    paired_csv = cfg["paths"]["paired_csv"]
    seq_len = cfg["data"]["seq_len"]

    left_flank = seq_len // 2
    right_flank = seq_len - left_flank - 1

    if os.path.exists(paired_csv):
        print(f"Loading existing paired CSV from {paired_csv}")
        df = pd.read_csv(paired_csv)
    else:
        print("Building paired CSV from VCF + FASTA...")
        ref_genome = build_ref_genome(cfg["paths"]["fasta"])
        benign_vars, path_vars = parse_clinvar_strict(
            cfg["paths"]["clinvar_vcf"],
            cfg["data"]["pos_label"],
            cfg["data"]["neg_label"],
        )
        df_benign = variants_to_dataframe(
            benign_vars,
            cfg["data"]["neg_label"],
            ref_genome, seq_len, left_flank, right_flank,
        )
        df_path = variants_to_dataframe(
            path_vars,
            cfg["data"]["pos_label"],
            ref_genome, seq_len, left_flank, right_flank,
        )
        df = pd.concat([df_benign, df_path], ignore_index=True)
        os.makedirs(os.path.dirname(paired_csv), exist_ok=True)
        df.to_csv(paired_csv, index=False)
        print(f"Saved paired CSV to {paired_csv}")
    
    df["chrom"] = df["chrom"].astype(str)

    # Map labels to ids
    use_labels = [cfg["data"]["neg_label"], cfg["data"]["pos_label"]]
    label2id = {lbl: i for i, lbl in enumerate(use_labels)}
    df["label_id"] = df["label"].map(label2id).astype(int)
    return df


def subsample_df(df: pd.DataFrame, subset_per_class: Optional[int], seed: int) -> pd.DataFrame:
    """Optionally subsample to balanced subset per class."""
    if subset_per_class is None:
        return df
    print(f"Subsampling to {subset_per_class} per class")
    df_sub = (
        df.groupby("label_id", group_keys=False)
          .apply(lambda x: x.sample(
              n=min(subset_per_class, len(x)), random_state=seed))
          .reset_index(drop=True)
    )
    print(df_sub["label"].value_counts())
    return df_sub


def chrom_split(df: pd.DataFrame, val_chroms, test_chroms):
    """Return (train_df, val_df, test_df) using chromosome-based splitting."""
    all_chroms = sorted(df["chrom"].unique().tolist())
    chr_val = set(val_chroms)
    chr_test = set(test_chroms)
    chr_train = set(all_chroms) - chr_val - chr_test

    def _sel(chroms):
        return df[df["chrom"].isin(chroms)].reset_index(drop=True)

    return _sel(chr_train), _sel(chr_val), _sel(chr_test)


def random_split(df: pd.DataFrame, seed: int):
    """Return (train_df, val_df, test_df) via stratified random split."""
    train_df, tmp = train_test_split(
        df, test_size=0.2, stratify=df["label_id"], random_state=seed
    )
    val_df, test_df = train_test_split(
        tmp, test_size=0.5, stratify=tmp["label_id"], random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_splits(cfg, df_all: pd.DataFrame):
    """Prepare train/val/test splits and compute class weights."""
    seed = cfg["training"]["seed"]
    if cfg["data"]["chrom_split"]:
        train_df, val_df, test_df = chrom_split(
            df_all,
            cfg["data"]["val_chroms"],
            cfg["data"]["test_chroms"],
        )
    else:
        train_df, val_df, test_df = random_split(df_all, seed)

    counts = train_df["label_id"].value_counts().sort_index().to_numpy()
    class_weights = (counts.sum() / (len(counts) * counts)).astype(np.float32)
    return train_df, val_df, test_df, class_weights


# ---------- Dataset + collator ----------


class ClinVarPairedDataset(Dataset):
    """Dataset that returns paired ref/alt tokenized sequences and label."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_tokens: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        ref_seq = row["ref_seq"]
        alt_seq = row["alt_seq"]
        y = int(row["label_id"])

        ref_enc = self.tokenizer(
            ref_seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        alt_enc = self.tokenizer(
            alt_seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        return {
            "ref_input_ids": ref_enc["input_ids"].squeeze(0),
            "ref_attention_mask": ref_enc["attention_mask"].squeeze(0),
            "alt_input_ids": alt_enc["input_ids"].squeeze(0),
            "alt_attention_mask": alt_enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(y, dtype=torch.long),
        }


@dataclass
class DataCollatorSiamese:
    """Simple collator that stacks all tensor fields."""

    def __call__(self, features):
        batch = {}
        for key in [
            "ref_input_ids",
            "ref_attention_mask",
            "alt_input_ids",
            "alt_attention_mask",
            "labels",
        ]:
            batch[key] = torch.stack([f[key] for f in features])
        return batch
