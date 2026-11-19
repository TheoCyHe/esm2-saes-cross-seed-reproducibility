"""§CT23.3: the REGION:Disordered argmax share among ALIGNED feature-seed cells,
under the regression's own rule and under raw AP (§CT23.3-F).

Three argmax rules over the 703 pool concepts, one population each way:

  * raw AP                 -- §AW4.2's rule (the 72.4% of all cells);
  * G = (AP - p)/(1 - p)   -- §CD3's rule (2.4568% of aligned, 1.8828% of all);
  * z = G / mu_s           -- the regression's (`gtilde_compliant_b_ratio.h5`
                              `argmax_concept`, 1.05% of all cells).

"Aligned" is §CD3.2's: max over the pool of G >= the configured
`alignment_min_normalised_gain` (0.05). Every figure already on the record is a
GATE here: the run refuses unless it reproduces all five before it reports the
two new ones. The inputs, their seed alignment and §CT22's four gates come from
`concept_subspace.load` unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

from src.analysis import concept_subspace as cs

DISORDERED = "REGION:Disordered"
N_CELLS = 204_800
# The record's figures, written down (gotcha 32b), each a gate.
RECORD = {"n_aligned": 48_152,                 # §CD3.2
          "gain_dis_aligned": 1_183,           # 2.4568%, §CD3.2
          "gain_dis_all": 3_856,               # 1.8828%, job 30439301
          "z_dis_all": 2_148,                  # 1.05%, job 30433761 / NR00657
          "ap_dis_all": 148_298}               # 72.4%, §AW4.2 / job 21087505


def shares(ap_pool: np.ndarray, prevalence: np.ndarray, z_argmax_pool: np.ndarray,
           dis: int, floor: float) -> dict:
    """Counts per rule and population. `ap_pool` is (seed, feature, pool);
    `z_argmax_pool` is (seed, feature) in POOL indices (-1 where the stored
    argmax is not a pool concept). Ties: lowest index, as every producer."""
    p = np.asarray(prevalence, dtype=np.float64)
    n = {"n_cells": int(ap_pool.shape[0] * ap_pool.shape[1]), "n_aligned": 0,
         "gain_dis_all": 0, "gain_dis_aligned": 0, "z_dis_all": 0,
         "z_dis_aligned": 0, "ap_dis_all": 0, "ap_dis_aligned": 0}
    for s in range(ap_pool.shape[0]):
        a = ap_pool[s].astype(np.float64)
        g = (a - p) / (1.0 - p)
        jg = g.argmax(axis=1)
        al = g[np.arange(g.shape[0]), jg] >= floor
        ja = ap_pool[s].argmax(axis=1)
        jz = z_argmax_pool[s]
        n["n_aligned"] += int(al.sum())
        n["gain_dis_all"] += int((jg == dis).sum())
        n["gain_dis_aligned"] += int(((jg == dis) & al).sum())
        n["z_dis_all"] += int((jz == dis).sum())
        n["z_dis_aligned"] += int(((jz == dis) & al).sum())
        n["ap_dis_all"] += int((ja == dis).sum())
        n["ap_dis_aligned"] += int(((ja == dis) & al).sum())
    return n


def check_record(n: dict, record: dict = RECORD) -> list[str]:
    return [f"{k}: {n.get(k)!r} != recorded {v!r}" for k, v in record.items()
            if n.get(k) != v]


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # concept_subspace's parser carries every input path load() needs; its
    # positional mode is required there and unused by load(), so "pilot" is
    # passed and is NOT a pilot run.
    args = cs.build_parser().parse_args(["pilot"] + argv)
    if args.out.exists():
        raise SystemExit(f"FATAL: {args.out} exists")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    floor = _find_key(cfg, "alignment_min_normalised_gain")
    if floor is None:
        raise SystemExit("alignment_min_normalised_gain ABSENT from the config")
    W = cs.load(args)
    import h5py
    with h5py.File(args.ap, "r") as hf:
        in_pool = hf["pooled/in_argmax_pool"][:].astype(bool)
        prev = hf["pooled/prevalence"][:][in_pool]
    dis = W["pool"].index(DISORDERED)
    gt_to_pool = {int(g): i for i, g in enumerate(W["pool_to_gt"]) if g >= 0}
    zp = np.vectorize(lambda g: gt_to_pool.get(int(g), -1))(W["argmax"])
    n = shares(W["ap"], prev, zp, dis, floor)
    bad = check_record(n)
    if n["n_cells"] != N_CELLS:
        bad.append(f"n_cells {n['n_cells']} != {N_CELLS}")
    if bad:
        raise SystemExit("GATE FAILED: " + "; ".join(bad))
    out = {
        "counts": n,
        "shares": {
            "raw_ap_all": n["ap_dis_all"] / n["n_cells"],
            "raw_ap_aligned": n["ap_dis_aligned"] / n["n_aligned"],
            "gain_all": n["gain_dis_all"] / n["n_cells"],
            "gain_aligned": n["gain_dis_aligned"] / n["n_aligned"],
            "regression_rule_all": n["z_dis_all"] / n["n_cells"],
            "regression_rule_aligned": n["z_dis_aligned"] / n["n_aligned"]},
        "aligned_floor": floor,
        "gates": {"record_figures_reproduced": sorted(RECORD), **W["gates"]},
        "note": "the regression's rule is gtilde_compliant_b_ratio.h5's argmax_concept "
                "(z = G/mu_s); that its mu is the hard-bin mu rests on "
                "docs/CODEBASE.md, not on an attribute of the artifact (§CT23.1)",
        "identity": {"git_sha": _git("rev-parse", "HEAD"),
                     "git_dirty": bool(_git("status", "--porcelain")),
                     "inputs": W["inputs"], "section": "§CT23.3",
                     "producer": "src.analysis.ct23_disorder_shares"}}
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps(out["counts"]), flush=True)
    print(json.dumps(out["shares"]), flush=True)


def _find_key(d, key):
    if isinstance(d, dict):
        for k, v in d.items():
            if k == key:
                return float(v)
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def _git(*a) -> str:
    import subprocess
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    main()
