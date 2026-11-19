"""
Download Swiss-Prot and cluster it with mmseqs2 at the configured identity.

This is the gate on `data.train_max_residues`: the ~200M-residue figure in
earlier notes was for RAW Swiss-Prot, not for what survives clustering and the
train/eval split. So this script's real deliverable is not the clustering
itself but the RESIDUE COUNT REPORT at the end, which reports the corpus size
under BOTH candidate policies for open question 1 (keep all cluster members vs.
keep representatives only).

Clustering is agnostic to that open question -- representatives-vs-members is a
downstream filter over the same clustering -- so this can run before it is
settled, and does not need re-running when it is.

All mmseqs parameters come from configs/mvp.yaml. Nothing experiment-defining
is hardcoded here.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import yaml

from src.experiment_config import resolve

SWISSPROT_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/complete/uniprot_sprot.fasta.gz"
)


def download(dest: Path) -> Path:
    """Fetch Swiss-Prot if not already present. Idempotent."""
    gz = dest / "uniprot_sprot.fasta.gz"
    fasta = dest / "uniprot_sprot.fasta"
    if fasta.exists():
        print(f"[skip] {fasta} already present ({fasta.stat().st_size / 1e6:.0f} MB)")
        return fasta

    dest.mkdir(parents=True, exist_ok=True)
    if not gz.exists():
        print(f"downloading {SWISSPROT_URL}")
        urllib.request.urlretrieve(SWISSPROT_URL, gz)
    print(f"decompressing -> {fasta}")
    with gzip.open(gz, "rb") as fin, open(fasta, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return fasta


def fasta_lengths(path: Path) -> dict[str, int]:
    """accession -> sequence length. Accession is the UniProt acc from a
    `sp|ACC|NAME` header, else the first whitespace-delimited token."""
    lengths: dict[str, int] = {}
    acc, n = None, 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if acc is not None:
                    lengths[acc] = n
                header = line[1:].split()[0]
                parts = header.split("|")
                acc = parts[1] if len(parts) >= 3 else header
                n = 0
            else:
                n += len(line.strip())
    if acc is not None:
        lengths[acc] = n
    return lengths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--workdir", type=Path, required=True,
                   help="Scratch dir. The FASTA and mmseqs tmp are regenerable "
                        "intermediates and belong here, not in project space.")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--max-len", type=int, default=None,
                   help="Residues are counted as min(len, max_len) to match "
                        "what the embedding cache will actually store. "
                        "Read from model.max_len in --config; an explicit "
                        "value must agree with it (amendment CR).")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    # Amendment CR: the cap is the config's, and a CLI value may only agree.
    args.max_len = resolve(cfg, "model.max_len", args.max_len, "--max-len",
                           args.config)
    d = cfg["data"]
    ident = d["cluster_identity"]
    cov = d["cluster_coverage"]
    cov_mode = d["cluster_cov_mode"]
    sens = d["cluster_sensitivity"]

    args.workdir.mkdir(parents=True, exist_ok=True)
    fasta = download(args.workdir)

    prefix = args.workdir / f"sprot_clu{int(ident * 100)}"
    tsv = Path(f"{prefix}_cluster.tsv")

    if tsv.exists():
        print(f"[skip] clustering already done: {tsv}")
    else:
        cmd = [
            "mmseqs", "easy-cluster", str(fasta), str(prefix),
            str(args.workdir / "mmseqs_tmp"),
            "--min-seq-id", str(ident),
            "-c", str(cov),
            "--cov-mode", str(cov_mode),
            "-s", str(sens),
            "--threads", str(args.threads),
        ]
        print("running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    # ---- the actual deliverable: how big is the corpus, really? ----
    print("\ncounting residues...", flush=True)
    lengths = fasta_lengths(fasta)

    members: dict[str, list[str]] = defaultdict(list)
    with open(tsv) as fh:
        for line in fh:
            rep, mem = line.rstrip("\n").split("\t")
            members[rep].append(mem)

    def capped(acc: str) -> int:
        return min(lengths.get(acc, 0), args.max_len)

    n_seqs = len(lengths)
    n_clusters = len(members)
    res_all = sum(capped(a) for a in lengths)
    res_reps = sum(capped(r) for r in members)

    eval_frac = d["eval_cluster_fraction"]
    train_frac = 1.0 - eval_frac

    print(f"""
========================= CORPUS REPORT =========================
sequences                 {n_seqs:>15,}
clusters (id={ident})        {n_clusters:>15,}
mean cluster size         {n_seqs / max(1, n_clusters):>15.2f}

residues (capped at {args.max_len}):
  all members             {res_all:>15,}   ~{res_all * 2560 / 1e9:>8.0f} GB cached
  representatives only    {res_reps:>15,}   ~{res_reps * 2560 / 1e9:>8.0f} GB cached

projected TRAIN split ({train_frac:.0%} of clusters):
  all members             {int(res_all * train_frac):>15,}
  representatives only    {int(res_reps * train_frac):>15,}

projected EVAL split ({eval_frac:.0%} of clusters, NEVER subsampled):
  all members             {int(res_all * eval_frac):>15,}   ~{res_all * eval_frac * 2560 / 1e9:>8.0f} GB
  representatives only    {int(res_reps * eval_frac):>15,}   ~{res_reps * eval_frac * 2560 / 1e9:>8.0f} GB
=================================================================

data.train_max_residues is null (NO cap) -- CLAUDE.md open question 2,
RESOLVED 2026-07-23; the realised split is 180.4M train / 17.7M eval residues
(swissprot/split/split.json). The projection above is the diagnostic it was;
it sets nothing now. data.cluster_members: '{d["cluster_members"]}'.
Note these projections assume clusters split proportionally by residue; the
actual split script should report the realised numbers.
""")


if __name__ == "__main__":
    main()
