"""
Write a homology-disjoint train/eval split by holding out whole clusters.

This is the control the whole headline comparison rests on (CLAUDE.md open
question 1, DECIDED): clustering is used ONLY to split, and it discards
nothing. We hold out `eval_cluster_fraction` of the CLUSTERS -- every member of
a held-out cluster goes to eval, none to train -- so no train protein has a
near-homolog in eval. A random per-sequence split would put ~7 near-twins of
each train protein in eval (mean cluster size 7.54) and, because
annotation-aligned features track conserved functional sites, would inflate the
aligned group more than the orphan group and manufacture the result.

Determinism (CLAUDE.md reproducibility hygiene): the assignment is a pure
function of (sorted cluster reps, eval_cluster_fraction, split_seed). The
fraction and the seed are pinned in config; the cluster reps come from the
mmseqs2 cluster tsv passed as --clusters, which the split manifest records by
path and sha256 (clusters_tsv, clusters_tsv_sha256_16) alongside the two config
values and the realised counts. So, given that tsv, the split -- and the
activation cache built from it -- reproduces bit-identically rather than
approximately.

Inputs (from the clustering step, src/data/cluster_swissprot.py):
  - the source FASTA (headers `sp|ACC|NAME ...`; accession = the middle field)
  - the mmseqs `_cluster.tsv` (two bare-accession columns: rep <TAB> member,
    one line per sequence, the representative included as its own member)

Outputs (regenerable intermediates -> scratch): train.fasta, eval.fasta, and
split.json (the manifest).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from src.experiment_config import resolve

from src.sae.train import git_dirty, git_sha


def header_to_acc(header: str) -> str:
    """First token of a FASTA header line (without '>') -> UniProt accession.
    `sp|ACC|NAME` -> ACC; anything else -> the first whitespace token verbatim.
    Must match cluster_swissprot.fasta_lengths so accessions line up with the
    cluster tsv."""
    tok = header.split()[0]
    parts = tok.split("|")
    return parts[1] if len(parts) >= 3 else tok


def parse_clusters(tsv_path: Path) -> tuple[dict[str, str], list[str]]:
    """Read an mmseqs _cluster.tsv. Returns (member_to_rep, sorted unique reps).
    Reps are sorted so the downstream assignment is deterministic regardless of
    tsv line order."""
    member_to_rep: dict[str, str] = {}
    reps: set[str] = set()
    with open(tsv_path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            rep, mem = line.split("\t")
            member_to_rep[mem] = rep
            reps.add(rep)
    return member_to_rep, sorted(reps)


def assign_clusters(
    reps: list[str], eval_frac: float, seed: int
) -> tuple[set[str], set[str]]:
    """Deterministically partition cluster reps into (train, eval). `reps` must
    already be in a stable order (parse_clusters sorts them). Eval gets
    floor(eval_frac * n) clusters. A pure function of (reps, eval_frac, seed)."""
    if not 0.0 <= eval_frac < 1.0:
        raise ValueError(f"eval_frac must be in [0, 1); got {eval_frac}")
    n = len(reps)
    n_eval = int(n * eval_frac)          # floor: eval is the smaller side
    perm = np.random.default_rng(seed).permutation(n)
    eval_idx = set(perm[:n_eval].tolist())
    train, evl = set(), set()
    for i, rep in enumerate(reps):
        (evl if i in eval_idx else train).add(rep)
    return train, evl


def iter_fasta_records(path: Path):
    """Yield (accession, record_text) for each FASTA record, record_text being
    the full '>...' header line plus its sequence lines (newline-terminated),
    ready to write verbatim."""
    header = None
    buf: list[str] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if header is not None:
                    yield header_to_acc(header), "".join(buf)
                header = line[1:].rstrip("\n")
                buf = [line if line.endswith("\n") else line + "\n"]
            else:
                buf.append(line if line.endswith("\n") else line + "\n")
    if header is not None:
        yield header_to_acc(header), "".join(buf)


def seq_residues(record_text: str, max_len: int) -> int:
    """Residue count of a record, capped at max_len to match what the embedding
    cache actually stores (proteins longer than max_len are truncated there)."""
    n = 0
    for line in record_text.splitlines():
        if not line.startswith(">"):
            n += len(line.strip())
    return min(n, max_len)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--fasta", type=Path, required=True,
                   help="Source Swiss-Prot FASTA (uniprot_sprot.fasta).")
    p.add_argument("--clusters", type=Path, required=True,
                   help="mmseqs _cluster.tsv from cluster_swissprot.py.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Scratch dir for train.fasta / eval.fasta / split.json.")
    p.add_argument("--max-len", type=int, default=None,
                   help="Residue cap for the report; matches the cache. Read "
                        "from model.max_len in --config; an explicit value "
                        "must agree with it (amendment CR).")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    # Amendment CR: the cap is the config's, and a CLI value may only agree.
    args.max_len = resolve(cfg, "model.max_len", args.max_len, "--max-len",
                           args.config)
    d = cfg["data"]
    eval_frac = d["eval_cluster_fraction"]
    seed = d["split_seed"]
    members_policy = d["cluster_members"]
    if members_policy not in ("all", "representatives"):
        raise SystemExit(
            f"data.cluster_members={members_policy!r}; expected 'all' or "
            "'representatives'."
        )

    member_to_rep, reps = parse_clusters(args.clusters)
    train_reps, eval_reps = assign_clusters(reps, eval_frac, seed)
    print(f"{len(reps):,} clusters -> {len(train_reps):,} train / "
          f"{len(eval_reps):,} eval (seed {seed}, eval_frac {eval_frac})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.fasta"
    eval_path = args.out_dir / "eval.fasta"

    counts = {"train": 0, "eval": 0}          # sequences written
    residues = {"train": 0, "eval": 0}        # capped residues
    n_unclustered = 0
    with open(train_path, "w") as tf, open(eval_path, "w") as ef:
        for acc, record in iter_fasta_records(args.fasta):
            rep = member_to_rep.get(acc)
            if rep is None:
                # Every Swiss-Prot sequence should appear in the clustering; a
                # miss means the FASTA and tsv are out of sync. Count and skip
                # rather than silently misassign.
                n_unclustered += 1
                continue
            # 'representatives': keep only the rep sequence of each cluster.
            if members_policy == "representatives" and acc != rep:
                continue
            split = "eval" if rep in eval_reps else "train"
            (ef if split == "eval" else tf).write(record)
            counts[split] += 1
            residues[split] += seq_residues(record, args.max_len)

    if n_unclustered:
        print(f"WARNING: {n_unclustered:,} FASTA records had no cluster entry "
              "-- FASTA and cluster tsv may be out of sync.")

    clusters_sha = hashlib.sha256(args.clusters.read_bytes()).hexdigest()[:16]
    manifest = {
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "config_hash": hashlib.sha256(args.config.read_bytes()).hexdigest()[:16],
        "clusters_tsv": str(args.clusters),
        "clusters_tsv_sha256_16": clusters_sha,
        "source_fasta": str(args.fasta),
        "split_seed": seed,
        "eval_cluster_fraction": eval_frac,
        "cluster_members": members_policy,
        "max_len_for_report": args.max_len,
        "n_clusters": len(reps),
        "n_train_clusters": len(train_reps),
        "n_eval_clusters": len(eval_reps),
        "n_train_sequences": counts["train"],
        "n_eval_sequences": counts["eval"],
        "n_unclustered_records": n_unclustered,
        "train_residues_capped": residues["train"],
        "eval_residues_capped": residues["eval"],
        # The eval cluster reps make the split fully auditable and let anyone
        # verify determinism without re-running the RNG. Sorted for a stable file.
        "eval_cluster_reps": sorted(eval_reps),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.out_dir / "split.json").write_text(json.dumps(manifest, indent=2))

    gb = 2560 / 1e9
    print(f"""
========================= SPLIT REPORT ==========================
policy                    cluster_members={members_policy}
train  sequences          {counts['train']:>15,}
       residues (<= {args.max_len})   {residues['train']:>15,}   ~{residues['train']*gb:>7.0f} GB
eval   sequences          {counts['eval']:>15,}
       residues (<= {args.max_len})   {residues['eval']:>15,}   ~{residues['eval']*gb:>7.0f} GB
=================================================================
train.fasta -> {train_path}
eval.fasta  -> {eval_path}
split.json  -> {args.out_dir / 'split.json'}

Set data.train_max_residues from the train residue count if a cap is wanted
(currently null per open question 2). eval is NEVER subsampled.
""")


if __name__ == "__main__":
    main()
