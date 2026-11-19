"""
Split a FASTA into N residue-balanced shards for a SLURM array cache run.

Why shard, and why by RESIDUES not sequences: the train cache is ~180M
residues / ~460 GB. A single 6-hour monolith cannot backfill into the ~441-deep
GPU queue, but N short array tasks with modest requests slot into scheduling
gaps, so a sharded run starts producing output sooner at identical total compute
(CLAUDE.md). One .h5 per shard, and the training loader already accepts a list
of cache files, so there is NO merge step. Balancing on residues (not sequence
count) matters because runtime is residues x throughput -- a shard with a few
very long proteins would straggle if we balanced on counts.

Balancing is a deterministic greedy longest-processing-time pass (largest
sequences first, each to the currently-lightest shard), so re-running produces
byte-identical shards -- no seed. The residue count per sequence is
min(len, max_len), matching exactly what cache.py stores.

REPRODUCIBILITY: the shard count AND the assignment change which residues share
a 4096-row block, hence the training loader's shuffle order, hence the trained
weights. This does not vary across the 40-dictionary ensemble (all share one
sharding; `n_seeds` = 40, `n_anchors` = 20 -- CLAUDE.md pinned notation), so it
does not threaten "vary only the seed" -- but it IS an input the cache must pin
to reproduce bit-identically. runtime.cache_shards holds N; the manifest records
the realised assignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.experiment_config import resolve

from src.data.split_train_eval import (
    header_to_acc,
    iter_fasta_records,
    seq_residues,
)
from src.sae.train import git_dirty, git_sha

# Fixed-width shard ids (shard_000 ...) so the SLURM array's printf and the
# training loader's shard_*.h5 glob sort numerically == lexically, independent
# of how many shards there are. Caps the shard count at 1000.
SHARD_ID_WIDTH = 3


def balance_shards(
    records: list[tuple[int, str]], n_shards: int
) -> tuple[list[list[int]], list[int]]:
    """Greedy LPT partition. `records` is [(residues, key)] where key is a
    stable unique tie-break. Returns (shard_assignment, shard_residues) where
    shard_assignment[i] is the list of record indices (into `records`) for shard
    i. Deterministic: largest first, ties by key, each to the lightest shard
    (ties by shard index)."""
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    if n_shards > len(records):
        raise ValueError(
            f"n_shards={n_shards} exceeds {len(records)} sequences -- would "
            "leave empty shards (an empty .h5 the loader cannot use)."
        )
    order = sorted(range(len(records)), key=lambda i: (-records[i][0], records[i][1]))
    shard_res = [0] * n_shards
    assignment: list[list[int]] = [[] for _ in range(n_shards)]
    for idx in order:
        res = records[idx][0]
        # lightest shard, index as a stable tie-break
        i = min(range(n_shards), key=lambda j: (shard_res[j], j))
        assignment[i].append(idx)
        shard_res[i] += res
    return assignment, shard_res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--fasta", type=Path, required=True,
                   help="Input FASTA (typically train.fasta from the split).")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Scratch dir for shard_000.fasta ... and shards.json.")
    p.add_argument("--n-shards", type=int, default=None,
                   help="Overrides runtime.cache_shards.")
    p.add_argument("--prefix", default="shard",
                   help="Output basename prefix (default 'shard').")
    p.add_argument("--max-len", type=int, default=None,
                   help="Residue cap for balancing; read from model.max_len in "
                        "--config, the same key cache.py reads, so the balance "
                        "reflects stored rows. An explicit value must agree "
                        "with the config (amendment CR).")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    # Amendment CR: the cap is the config's, and a CLI value may only agree.
    args.max_len = resolve(cfg, "model.max_len", args.max_len, "--max-len",
                           args.config)
    n_shards = args.n_shards or cfg["runtime"]["cache_shards"]
    if n_shards >= 10 ** SHARD_ID_WIDTH:
        raise SystemExit(f"n_shards={n_shards} exceeds the {SHARD_ID_WIDTH}-digit "
                         "shard-id width; widen SHARD_ID_WIDTH.")

    # (residues, acc, record_text). acc is the stable tie-break; record_text is
    # written verbatim so headers survive into the shard.
    records: list[tuple[int, str, str]] = []
    for acc, text in iter_fasta_records(args.fasta):
        records.append((seq_residues(text, args.max_len), acc, text))
    if not records:
        raise SystemExit(f"{args.fasta} has no records")

    assignment, shard_res = balance_shards(
        [(r[0], r[1]) for r in records], n_shards)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths, shard_counts = [], []
    for i, idxs in enumerate(assignment):
        path = args.out_dir / f"{args.prefix}_{i:0{SHARD_ID_WIDTH}d}.fasta"
        with open(path, "w") as fh:
            for idx in idxs:            # keep input order within a shard
                fh.write(records[idx][2])
        shard_paths.append(str(path))
        shard_counts.append(len(idxs))

    total_res = sum(shard_res)
    mean_res = total_res / n_shards
    imbalance = (max(shard_res) - min(shard_res)) / mean_res if mean_res else 0.0

    manifest = {
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "config_hash": hashlib.sha256(args.config.read_bytes()).hexdigest()[:16],
        "source_fasta": str(args.fasta),
        "n_shards": n_shards,
        "max_len": args.max_len,
        "n_sequences": len(records),
        "total_residues_capped": total_res,
        "shard_residues": shard_res,
        "shard_sequences": shard_counts,
        "shard_paths": shard_paths,
        "residue_imbalance": imbalance,   # (max-min)/mean across shards
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.out_dir / "shards.json").write_text(json.dumps(manifest, indent=2))

    print(f"{len(records):,} sequences / {total_res:,} residues -> {n_shards} shards")
    print(f"per-shard residues: min {min(shard_res):,} max {max(shard_res):,} "
          f"mean {mean_res:,.0f}  imbalance {imbalance:.3%}")
    print(f"shards + shards.json -> {args.out_dir}")
    print(f"\nsubmit the cache array with:\n"
          f"  sbatch --array=0-{n_shards - 1} slurm/cache_train_array.sbatch")


if __name__ == "__main__":
    main()
