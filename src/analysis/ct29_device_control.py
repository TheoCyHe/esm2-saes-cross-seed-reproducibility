"""§CT29-F item 2: the device control's verdict.

One seed of the record's configuration (k = 32, width 10,240, label 40) is
trained on another partition. It passes if its raw FVU and its mean rule-1
match rate at θ = 0.70 against the 20 anchors both lie in the anchors' own
per-seed ranges, which were stated at declaration:

  raw FVU   [0.36214210341635106, 0.36656082216321473]   (fvu_eval_cf.json)
  the rate  [0.49134457236842105, 0.5000668174342104]    (each anchor's mean
            k/19 over its 19 anchor comparators, rf_counts_theta070_n19.h5)

The rate is the like-for-like object: for the control seed, the mean over the
20 anchors of (decoder-cosine mutual pairs at >= θ) / dict_size; for an anchor,
the mean over its 19 anchor comparators of the same quantity.

The instrument is checked before it is read (gotcha 13). Anchor 0's counts
against anchors 1-19, computed here by `rf_marginal.pair_statistics` with
anchor 0 on the `a` side as `rf_marginal` ran it, must equal row 0 of the stored
`rf_counts_theta070_n19.h5` cell for cell. Any differing cell is a STOP, and
the control's rate is not read.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch

FVU_RANGE = (0.36214210341635106, 0.36656082216321473)
RATE_RANGE = (0.49134457236842105, 0.5000668174342104)
THETA = 0.70


def mnn_counts(W_a: torch.Tensor, others: list[torch.Tensor], theta: float) -> np.ndarray:
    """Per-feature count, over `others`, of decoder-cosine MNN partners with
    similarity >= theta, with `W_a` on the `a` side of `pair_statistics`, the
    side `rf_marginal` puts the lower-numbered seed on."""
    from src.analysis.rf_marginal import pair_statistics
    k = np.zeros(W_a.shape[1], dtype=np.int64)
    for W_b in others:
        sim = pair_statistics(W_a, W_b)["a_mnn"].float().cpu().numpy()
        k += (sim >= theta)
    return k


def pair_rates(W_a: torch.Tensor, others: list[torch.Tensor], theta: float) -> list[float]:
    """For each comparator, the fraction of W_a's features with an MNN partner
    at >= theta there."""
    from src.analysis.rf_marginal import pair_statistics
    out = []
    for W_b in others:
        sim = pair_statistics(W_a, W_b)["a_mnn"].float().cpu().numpy()
        out.append(float((sim >= theta).mean()))
    return out


def in_range(x: float, lo_hi: tuple[float, float]) -> bool:
    return bool(lo_hi[0] <= x <= lo_hi[1])


def verdict(fvu_raw: float | None, rate: float | None) -> str:
    """PASS only if both quantities exist and both lie in their closed ranges;
    a missing quantity is NOT_EXECUTED, never a pass (gotcha 15)."""
    if fvu_raw is None or rate is None:
        return "NOT_EXECUTED"
    return "PASS" if in_range(fvu_raw, FVU_RANGE) and in_range(rate, RATE_RANGE) else "FAIL"


def main(argv: list[str] | None = None) -> None:
    from src.analysis.artifact_identity import device_class
    from src.analysis.rf_marginal import load_decoder
    from src.sae.train import git_dirty, git_sha

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", type=Path, required=True, help="the control seed's checkpoint dir")
    p.add_argument("--anchor-root", type=Path, required=True)
    p.add_argument("--anchors", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--stored-counts", type=Path, required=True,
                   help="rf_counts_theta070_n19.h5, for the instrument check")
    p.add_argument("--fvu-json", type=Path, default=None,
                   help="the control's fvu_eval output (per_seed[<seed>].raw.fvu)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    anchors = {s: load_decoder(a.anchor_root / f"seed_{s}", dev)[0] for s in a.anchors}
    W_c, meta_c = load_decoder(a.ckpt, dev)
    run_dev = str(device_class(W_c))
    print(f"[ct29-dc] device {run_dev}", flush=True)

    # The instrument check: anchor 0 against anchors 1..19, against the stored row.
    with h5py.File(a.stored_counts, "r") as hf:
        labels = [int(x) for x in json.loads(hf.attrs["seed_labels"])]
        row0 = hf["match_count"][labels.index(a.anchors[0])].astype(np.int64)
        n_cmp = int(hf.attrs["n_compare_seeds"])
    mine = mnn_counts(anchors[a.anchors[0]], [anchors[s] for s in a.anchors[1:]], THETA)
    n_diff = int((mine != row0).sum())
    check = {"anchor": a.anchors[0], "n_compare": n_cmp,
             "n_differing_cells": n_diff,
             "verdict": "EXACT" if n_diff == 0 else "STOP"}
    print(f"[ct29-dc] instrument check {check}", flush=True)

    out = {"identity": {"measurement": "ct29_device_control",
                        "governing_amendment": "CT29-F item 2",
                        "control_ckpt": str(a.ckpt),
                        "control_seed": meta_c.get("seed"),
                        "control_device_class": meta_c.get("device_class"),
                        "control_slurm_partition": meta_c.get("slurm_partition"),
                        "control_git_sha": meta_c.get("git_sha"),
                        "anchor_root": str(a.anchor_root), "anchors": a.anchors,
                        "matching_device_class": run_dev, "theta": THETA,
                        "fvu_range": FVU_RANGE, "rate_range": RATE_RANGE,
                        "git_sha": git_sha(), "git_dirty": git_dirty(),
                        "written_utc": datetime.now(timezone.utc).isoformat()},
           "instrument_check": check}
    if check["verdict"] != "EXACT":
        out["verdict"] = "STOP"
        a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        raise SystemExit("STOP: the instrument does not reproduce the stored counts")

    rates = pair_rates(W_c, [anchors[s] for s in a.anchors], THETA)
    rate = float(np.mean(rates))
    fvu_raw = None
    if a.fvu_json is not None:
        f = json.loads(a.fvu_json.read_text(encoding="utf-8"))
        fvu_raw = float(f["per_seed"][str(meta_c.get("seed"))]["raw"]["fvu"])
    v = verdict(fvu_raw, rate)
    out.update({"per_anchor_rate": dict(zip(map(str, a.anchors), rates)),
                "mean_rate": rate, "fvu_raw": fvu_raw,
                "rate_in_range": in_range(rate, RATE_RANGE),
                "fvu_in_range": (None if fvu_raw is None else in_range(fvu_raw, FVU_RANGE)),
                "verdict": v})
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[ct29-dc] rate {rate!r} fvu_raw {fvu_raw!r} verdict {v}", flush=True)


if __name__ == "__main__":
    main()
