"""
Is the SAE activation gather DEVICE-INDEPENDENT?

WHY THIS EXISTS. `feature_ap.py` and `null_pilot.py`'s masked path call the
SAME function -- `feature_ap.sparse_activations` -- which runs the SAE encoder
over `eval.h5` and returns the feature-major non-zeros. Neither reads persisted
firing indices; `row_index_cache` is used only on the UNMASKED (pooled) path.

`feature_ap_experimental_av.h5`, the OBSERVED statistic, was produced on the
`gpu` partition. If its null is produced on CPU, the two rest on activations
gathered on different device classes -- and a null is a reference distribution
FOR a statistic computed on a particular set of activations.

**The failure mode is specific, not vague.** `TopKSAE.forward` ReLUs the
pre-activations and then takes the top k. A CPU and a GPU float32 matmul differ
in the last bits, so two pre-activations within float32 rounding of each other
can swap places and change WHICH k features fire on that row. That changes the
non-zero set, the ranks, and therefore every AP downstream. BC7 already
measured this class of effect at 4 cells in 204,800 for the MNN threshold.

**So it is measured here rather than assumed in either direction.** This script
gathers ONE seed on whichever device it is given and writes a fingerprint:
sha256 over the raw bit patterns of `feat`, `pos` and `seg_start`, plus the
per-feature firing counts and the total non-zero count. Run it once per device
class and compare the fingerprints.

An IDENTICAL fingerprint closes the question completely -- the gather is
device-independent on this data, and a CPU null is consistent with the
GPU-produced AP artifact. A DIFFERING one gives the magnitude (how many rows
changed their firing set) and turns the decision into an arithmetic one.

It also reports the GATHER'S DURATION, which is the other unmeasured quantity:
268 s on an rtx_5000_ada, unmeasured on CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.analysis.feature_ap import load_scored_row_set, sparse_activations
from src.sae.train import TopKSAE, git_dirty, git_sha


def sha_of(t: torch.Tensor) -> str:
    """sha256 over the RAW BIT PATTERN, so nothing is masked by a tolerance."""
    a = t.cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--eval-cache", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--scored-row-set", default="label_store")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    d_model = cfg["model"]["d_model"]
    dict_size, k = cfg["sae"]["dict_size"], cfg["sae"]["k"]
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    gpu_name = (torch.cuda.get_device_name(0)
                if device == "cuda" and torch.cuda.is_available() else None)
    print(f"device={device} gpu={gpu_name}")

    subset = load_scored_row_set(args.labels, args.scored_row_set)
    sd = torch.load(args.ckpt / "model.pt", map_location="cpu")
    model = TopKSAE(d_model, dict_size, k, b_dec=torch.zeros(d_model)).to(device)
    model.load_state_dict(sd)
    model.eval()

    t0 = time.time()
    feat, pos, seg = sparse_activations(model, args.eval_cache, subset, device)
    elapsed = time.time() - t0

    fire_counts = torch.bincount(feat, minlength=dict_size)
    doc = {
        "artifact": "gather_fingerprint",
        "device_requested": args.device,
        "device_used": device,
        "gpu_name": gpu_name,
        "checkpoint": str(args.ckpt),
        "eval_cache": str(args.eval_cache),
        "scored_row_set": args.scored_row_set,
        "n_rows_scored": int(len(subset)) if subset is not None else None,
        "GATHER_SECONDS": round(elapsed, 1),
        "nnz": int(feat.numel()),
        "sha256_feat": sha_of(feat),
        "sha256_pos": sha_of(pos),
        "sha256_seg_start": sha_of(seg),
        "sha256_fire_counts": sha_of(fire_counts),
        "fire_counts_sum": int(fire_counts.sum()),
        "fire_counts_min": int(fire_counts.min()),
        "fire_counts_max": int(fire_counts.max()),
        "git_sha": git_sha(), "git_dirty": bool(git_dirty()),
        "note": "Compare two of these across device classes. IDENTICAL "
                "sha256_feat AND sha256_pos means the gather is "
                "device-independent on this data and a CPU null is consistent "
                "with a GPU-produced AP artifact.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    print(json.dumps(doc, indent=2))
    print(f"\nwrote {args.out}")
    print(f"GATHER on {device}: {elapsed:.1f}s for {feat.numel():,} non-zeros")


if __name__ == "__main__":
    main()
