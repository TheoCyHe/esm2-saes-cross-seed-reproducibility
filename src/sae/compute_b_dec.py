"""
Compute the shared decoder bias b_dec ONCE from the activation cache.

CLAUDE.md locked decision (resolved 2026-07-24): b_dec is TRAINABLE from a
SHARED INIT, and this script computes that init. It is computed once and is
bit-identical across all 40 dictionaries (b_dec_init_sha256_16
f3185a6c22f91039 in every meta.json), so the starting point is never an
uncontrolled source of cross-seed variance; train.py loads it as the initial
value and trains from there (sae.b_dec_trainable: true). The script is run a
single time (per cache), before the ensemble, and its result is committed to
the config as sae.b_dec_path. (Docstring corrected 2026-09-16: it said
"bit-identical across all 20 seeds ... train.py loads the output frozen". No
behaviour change.)

The estimate must NOT depend on the ensemble seed. Sampling uses --sample-seed
(default 0), recorded in the output, so re-running reproduces the same vector
bit-for-bit given the same cache.

b_dec is the reconstruction when every latent is off (x_hat = W_dec z + b_dec,
z = 0), i.e. the center of the activation cloud. Two conventions:
  - geometric_median : Gao et al.'s init; robust to the heavy-tailed activation
                       norms of an ESM residual stream (minimizes sum of L2).
  - mean             : the L2-optimal constant baseline for an MSE loss
                       (minimizes sum of squared L2).
Which one is used is a config choice (sae.b_dec_source) and is recorded here.

Reads are chunk-aligned: whole blocks are drawn at random positions across the
cache (and across shards) rather than scattering a million single-row reads over
the whole 456 GB file. The sample is representative of the corpus center while
touching only ~sample_rows worth of bytes. Rows within a block are same-protein
correlated, so the block set is drawn from random positions to spread proteins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from src.sae.train import _jsonable, config_hash, git_dirty, git_sha

BLOCK_ROWS = 4096  # keep aligned to the cache's HDF5 chunking


def gather_blocks(paths: list[Path]) -> tuple[list[tuple[int, int, int]], int, int]:
    """List every chunk-aligned block across all shards as (file_idx, start,
    end), plus (total_rows, d_model)."""
    blocks: list[tuple[int, int, int]] = []
    total = 0
    d_model = None
    for fi, p in enumerate(paths):
        with h5py.File(p, "r") as hf:
            n, d = hf["activations"].shape
        if d_model is None:
            d_model = d
        elif d_model != d:
            raise ValueError(f"{p} has d_model={d}, expected {d_model}")
        total += n
        for s in range(0, n, BLOCK_ROWS):
            blocks.append((fi, s, min(s + BLOCK_ROWS, n)))
    return blocks, total, d_model


def sample_rows(
    paths: list[Path], n_sample: int, seed: int
) -> tuple[np.ndarray, int]:
    """Deterministically sample ~n_sample rows by reading whole blocks from
    random positions. Returns (rows float32 (n, d), total_rows_in_cache)."""
    blocks, total, _ = gather_blocks(paths)
    rng = np.random.default_rng(seed)

    if total <= n_sample:
        chosen = list(range(len(blocks)))
    else:
        chosen = []
        acc = 0
        for bi in rng.permutation(len(blocks)):
            fi, s, e = blocks[bi]
            chosen.append(int(bi))
            acc += e - s
            if acc >= n_sample:
                break

    handles: dict[int, h5py.File] = {}
    parts = []
    try:
        for bi in sorted(chosen):          # sorted -> more sequential on disk
            fi, s, e = blocks[bi]
            if fi not in handles:
                handles[fi] = h5py.File(paths[fi], "r")
            parts.append(handles[fi]["activations"][s:e].astype(np.float32))
    finally:
        for h in handles.values():
            h.close()

    X = np.concatenate(parts, axis=0)
    if X.shape[0] > n_sample:
        sub = np.sort(rng.choice(X.shape[0], size=n_sample, replace=False))
        X = X[sub]
    return X, total


def geometric_median(
    X: np.ndarray, eps: float = 1e-6, max_iter: int = 300
) -> tuple[np.ndarray, int]:
    """Weiszfeld's algorithm in float64. Returns (median, iterations run)."""
    Xd = X.astype(np.float64)
    y = Xd.mean(axis=0)
    it = 0
    for it in range(1, max_iter + 1):
        dist = np.sqrt(((Xd - y) ** 2).sum(axis=1))
        # Points coincident with the current estimate would divide by zero;
        # flooring the distance is the standard Weiszfeld guard.
        w = 1.0 / np.maximum(dist, eps)
        y_new = (w[:, None] * Xd).sum(axis=0) / w.sum()
        shift = float(np.sqrt(((y_new - y) ** 2).sum()))
        y = y_new
        if shift < eps:
            break
    return y, it


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--activations", type=Path, required=True, nargs="+",
                    help="Cache file(s). The b_dec is estimated over all shards "
                         "jointly -- it is one center for the whole train cache.")
    p.add_argument("--out", type=Path, required=True,
                    help="Output .pt. Put its path in configs sae.b_dec_path.")
    p.add_argument("--source", choices=["geometric_median", "mean"], default=None,
                    help="Overrides sae.b_dec_source.")
    p.add_argument("--sample-rows", type=int, default=None,
                    help="Overrides sae.b_dec_sample_rows. NOT the ensemble seed's "
                         "concern -- this is a one-time shared estimate.")
    p.add_argument("--sample-seed", type=int, default=0,
                    help="Fixed seed for the row sample. Recorded so the estimate "
                         "reproduces bit-for-bit. Must NOT be an ensemble seed.")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    sae_cfg = cfg["sae"]
    d_model_cfg = cfg["model"]["d_model"]

    source = args.source or sae_cfg["b_dec_source"]
    n_sample = args.sample_rows or sae_cfg["b_dec_sample_rows"]

    print(f"sampling up to {n_sample:,} rows (seed {args.sample_seed}) from "
          f"{len(args.activations)} file(s)...")
    X, total = sample_rows(args.activations, n_sample, args.sample_seed)
    if X.shape[1] != d_model_cfg:
        raise SystemExit(
            f"cache d_model={X.shape[1]} but config expects {d_model_cfg} -- "
            "wrong cache or wrong model?"
        )
    print(f"sampled {X.shape[0]:,} of {total:,} rows")

    iters = None
    if source == "geometric_median":
        b_dec, iters = geometric_median(X)
        print(f"geometric median: Weiszfeld converged in {iters} iters")
    elif source == "mean":
        b_dec = X.astype(np.float64).mean(axis=0)
        print("mean computed")
    else:
        raise SystemExit(f"unknown b_dec source {source!r}")

    b_dec = b_dec.astype(np.float32)
    b_dec_sha = hashlib.sha256(b_dec.tobytes()).hexdigest()[:16]

    with h5py.File(args.activations[0], "r") as hf:
        cache_attrs = {k: _jsonable(v) for k, v in hf.attrs.items()}

    blob = {
        "b_dec": torch.from_numpy(b_dec),
        "source": source,
        "d_model": int(b_dec.shape[0]),
        "n_rows_sampled": int(X.shape[0]),
        "n_rows_total": int(total),
        "sample_seed": args.sample_seed,
        "weiszfeld_iters": iters,
        "activations": [str(p) for p in args.activations],
        "cache_attrs": cache_attrs,
        "b_dec_sha256_16": b_dec_sha,
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "config_hash": config_hash(args.config),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)

    meta = {k: v for k, v in blob.items() if k != "b_dec"}
    meta["b_dec_norm"] = float(np.linalg.norm(b_dec))
    print(json.dumps(meta, indent=2))
    print(f"\nwrote b_dec ({source}, sha {b_dec_sha}) -> {args.out}")
    print(f"set sae.b_dec_path: {args.out}  in {args.config}")


if __name__ == "__main__":
    main()
