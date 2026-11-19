"""Leave-one-dictionary-out jackknife of the design-3 alignment effect
(PREREGISTRATION amendment CT, §CT3 and §CT11.1).

WHAT IT ESTIMATES. The headline's interval treats the 20 per-anchor fits as
independent; they are not -- at `n_compare = 39` two anchors share 38 of their
39 comparison dictionaries. The resampling unit here is the TRAINED DICTIONARY:
each of the 40 is removed in turn, every remaining anchor's match count is
recomputed against the 38 dictionaries left, the per-anchor design-3 fits are
re-run with the SAME estimator (`design3_join.fit_arm`, beta-binomial, lambda
inherited), and the fixed-effect combination is recomputed. The jackknife SE is
`sqrt((n-1)/n * sum((b_(-d) - mean)^2))`, `n = 40`.

WHY THE MATCHING IS RE-RUN (§CT-pre item 6). The stored artifact holds counts
summed over comparators, so the indicator "anchor feature f matched in
dictionary d" is not on disk. MNN is a PAIRWISE relation -- whether f and its
partner are mutual nearest neighbours depends on the two dictionaries only --
so removing d changes each count by exactly that indicator and nothing else.
The `match` step re-runs `rf_marginal.pair_statistics` on the same pairs, in the
same order, on the record's device class, and persists `mnn_sim` for the anchor
rows. It then GATES: the counts recomputed from what it persisted must equal the
stored counts (§CT11.1: EXACT / NEAR / DIVERGENT).

TWO STEPS, because they want different hardware: `match` on `rtx_5000_ada`
(the record's class, from job 20810740's sacct record -- the stored counts carry
no device attr), `fit` on CPU, as the join of record was.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

# The two tolerances §CT11.1 declares, named once.
NEAR_COS_TOL = 1e-6          # |cos - theta| below which a flipped cell is "near"
CONTROL_REL_TOL = 1e-6       # full-ensemble refit vs the join of record
GATE_BRANCHES = ("EXACT", "NEAR", "DIVERGENT")


# ---------------------------------------------------------------------------
# Pure functions -- the toy test exercises every one of these
# ---------------------------------------------------------------------------

def mnn_sim_anchor_rows(decoders: list, n_anchors: int) -> np.ndarray:
    """`rf_marginal`'s `mnn_sim`, restricted to the anchor rows.

    Same pairs, same order, same function (`pair_statistics`) as
    `rf_marginal.main`: every unordered pair with at least one anchor. Returns
    `(n_anchors, n_seeds, dict_size)` float32, `-inf` on the diagonal and
    wherever the anchor feature has no mutual partner.
    """
    from src.analysis.rf_marginal import pair_statistics
    n_seeds = len(decoders)
    dict_size = decoders[0].shape[1]
    sim = np.full((n_anchors, n_seeds, dict_size), -np.inf, dtype=np.float32)
    for a in range(n_seeds):
        for b in range(a + 1, n_seeds):
            if a >= n_anchors and b >= n_anchors:
                continue
            st = pair_statistics(decoders[a], decoders[b])
            if a < n_anchors:
                sim[a, b] = st["a_mnn"].float().cpu().numpy()
            if b < n_anchors:
                sim[b, a] = st["b_mnn"].float().cpu().numpy()
    return sim


def counts_from_sim(sim: np.ndarray, theta: float) -> np.ndarray:
    """Per-anchor match counts at one theta -- `counts_at_theta`'s arithmetic
    on the anchor rows (an anchor never compares against itself)."""
    n_anchors, n_seeds, _ = sim.shape
    off = np.ones((n_anchors, n_seeds), dtype=bool)
    off[np.arange(n_anchors), np.arange(n_anchors)] = False
    return ((sim >= theta) & off[:, :, None]).sum(axis=1).astype(np.int16)


def loo_counts(sim: np.ndarray, theta: float, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Counts with dictionary `d` (a COLUMN index) removed from the ensemble.

    Returns `(counts, anchor_rows_kept)`. If `d` is itself an anchor its row
    leaves the fit set; every remaining anchor loses exactly one comparator,
    so each count is out of `n_seeds - 2`.
    """
    n_anchors, n_seeds, _ = sim.shape
    full = counts_from_sim(sim, theta).astype(np.int32)
    # Column d of anchor row d is the diagonal, already -inf, so an anchor
    # never subtracts a match against itself.
    hit_d = (sim[:, d, :] >= theta)
    y = (full - hit_d.astype(np.int32)).astype(np.int16)
    keep = np.ones(n_anchors, dtype=bool)
    if d < n_anchors:
        keep[d] = False
    return y[keep], keep


def same_device_class(record: str, run: str) -> bool:
    """Do two device-class strings name the same class? The record's class
    comes from sacct's GRES name (`rtx_5000_ada`) and the run's from
    `artifact_identity.device_class` (`cuda:NVIDIA RTX 5000 Ada
    Generation:sm_89`) -- two vocabularies, so compare on alphanumerics: the
    record's token must appear inside the run's. A bare string inequality
    would call the same class different (found on job 30204427, where the gate
    was EXACT and the clause was not reached)."""
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())
    r, u = norm(record), norm(run)
    return bool(r) and (r == u or r in u or u in r)


def gate(sim: np.ndarray, stored: dict, record_device: str,
         run_device: str) -> dict:
    """§CT11.1's three-branch gate, total over the comparison.

    `stored` maps theta -> the stored (n_anchors, dict_size) counts. A
    differing cell (anchor a, feature f, theta) is NEAR when at least one of
    its comparators' persisted similarities lies within NEAR_COS_TOL of theta
    -- the stored side has no per-comparator data, so this is the most the
    comparison can see.
    """
    per_theta, n_diff, n_diff_near = {}, 0, 0
    for t in sorted(stored):
        mine = counts_from_sim(sim, t)
        ref = np.asarray(stored[t])
        if mine.shape != ref.shape:
            return {"verdict": "DIVERGENT", "reason":
                    f"shape {mine.shape} vs stored {ref.shape} at theta {t}"}
        diff = np.argwhere(mine != ref)
        near = 0
        for a, f in diff:
            s = sim[a, :, f]
            s = s[np.isfinite(s)]
            if s.size and np.any(np.abs(s - t) < NEAR_COS_TOL):
                near += 1
        per_theta[f"{t:.2f}"] = {"n_differing_cells": int(len(diff)),
                                 "n_differing_near_theta": int(near)}
        n_diff += len(diff)
        n_diff_near += near
    devices_differ = not same_device_class(record_device, run_device)
    if n_diff == 0:
        verdict = "EXACT"
    elif n_diff_near == n_diff and devices_differ:
        verdict = "NEAR"
    else:
        verdict = "DIVERGENT"
    return {"verdict": verdict, "n_differing_cells": int(n_diff),
            "n_differing_near_theta": int(n_diff_near),
            "record_device_class": record_device,
            "run_device_class": run_device,
            "device_classes_differ": bool(devices_differ),
            "near_cos_tol": NEAR_COS_TOL, "per_theta": per_theta}


def jackknife(loo: np.ndarray, full: float) -> dict:
    """Jackknife SE and bias of a statistic from its n leave-one-out values."""
    loo = np.asarray(loo, dtype=float)
    n = loo.size
    mean = float(loo.mean())
    se = float(np.sqrt((n - 1) / n * np.sum((loo - mean) ** 2)))
    return {"n": int(n), "se": se, "mean_of_loo": mean,
            "bias_estimate": float((n - 1) * (mean - full)),
            "bias_note": "reported, not applied (§CT3)"}


def control_verdict(mine: list[float], ref: list[float]) -> dict:
    """§CT11.1's control: the full-ensemble refit against the join of record."""
    mine, ref = np.asarray(mine, float), np.asarray(ref, float)
    if np.array_equal(mine, ref):
        return {"verdict": "BIT_IDENTICAL", "max_rel_diff": 0.0}
    rel = float(np.max(np.abs(mine - ref) / np.maximum(np.abs(ref), 1e-300)))
    return {"verdict": ("WITHIN_1E-6" if rel <= CONTROL_REL_TOL else "STOP"),
            "max_rel_diff": rel, "max_abs_diff_pp": float(np.max(np.abs(mine - ref)))}


def widest_interval(intervals: dict) -> str:
    """Name of the widest of several [lo, hi] intervals."""
    return max(intervals, key=lambda k: intervals[k][1] - intervals[k][0])


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _stored_counts(path: Path) -> tuple[dict, dict]:
    from src.analysis.design3_join import load_counts_all_theta
    allt = load_counts_all_theta(path)
    return allt["counts"], allt["meta"]


# ---------------------------------------------------------------------------
# match: GPU, the record's device class
# ---------------------------------------------------------------------------

def cmd_match(args) -> None:
    import torch
    from src.analysis.artifact_identity import device_class
    from src.analysis.rf_marginal import (check_unit_norm, load_decoder,
                                          verify_ensemble_identity)
    from src.sae.train import git_dirty, git_sha

    def _num(p: Path) -> int:
        return int(str(p.name).rsplit("_", 1)[-1])

    anchors = sorted(args.ckpts, key=_num)
    compare = sorted(args.compare_ckpts, key=_num)
    ckpts = anchors + compare
    if len(ckpts) != args.expect_total_ckpts:
        raise SystemExit(f"FATAL: {len(ckpts)} checkpoints, expected "
                         f"{args.expect_total_ckpts}")
    decoders, meta_by_seed, labels = [], {}, []
    for p in ckpts:
        W, meta = load_decoder(p, args.device)
        s = int(meta["seed"])
        if s in meta_by_seed:
            raise SystemExit(f"FATAL: seed {s} supplied twice")
        if check_unit_norm(W)["verdict"] != "PASS":
            raise SystemExit(f"FATAL: seed {s} decoder not unit-norm")
        meta_by_seed[s] = meta
        labels.append(s)
        decoders.append(W)
    ident = verify_ensemble_identity(meta_by_seed, getattr(args, "accept_git_shas", None))
    if ident["verdict"] != "PASS":
        raise SystemExit("FATAL: ensemble identity " + json.dumps(ident["problems"]))
    n_anchors = len(anchors)
    run_dev = device_class(decoders[0])
    print(f"[jk-match] {len(ckpts)} dictionaries ({n_anchors} anchors) on "
          f"{run_dev}", flush=True)

    sim = mnn_sim_anchor_rows(decoders, n_anchors)
    stored, smeta = _stored_counts(args.rf_counts_all_theta)
    if [int(x) for x in smeta["seed_labels"]] != labels[:n_anchors]:
        raise SystemExit("FATAL: anchor labels differ from the stored counts' "
                         f"{smeta['seed_labels']} vs {labels[:n_anchors]}")
    if smeta["n_compare_seeds"] != len(ckpts) - 1:
        raise SystemExit("FATAL: stored n_compare_seeds "
                         f"{smeta['n_compare_seeds']} != {len(ckpts) - 1}")
    g = gate(sim, stored, args.record_device_class, run_dev)
    print(f"[jk-match] GATE {g['verdict']}: {g['n_differing_cells']} differing "
          f"cells, {g['n_differing_near_theta']} near theta", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as hf:
        hf.create_dataset("mnn_sim_anchor_rows", data=sim, compression="gzip")
        hf.attrs.update({
            "measurement": "mnn_sim_anchor_rows_for_jackknife",
            "governing_amendment": "CT3 / CT11.1",
            "similarity": "decoder_cosine", "rule": "mnn",
            "anchor_seed_labels": json.dumps(labels[:n_anchors]),
            "column_seed_labels": json.dumps(labels),
            "n_anchors": n_anchors, "n_seeds": len(ckpts),
            "n_compare_seeds": len(ckpts) - 1,
            "dict_size": int(sim.shape[2]),
            "gather_device_class": str(run_dev),
            "record_device_class": args.record_device_class,
            "record_device_class_source": args.record_device_source,
            "gate": json.dumps(g),
            "gate_verdict": g["verdict"],
            "rf_counts_all_theta": str(args.rf_counts_all_theta),
            "rf_counts_all_theta_sha256_16": _sha16(args.rf_counts_all_theta),
            "git_sha": git_sha(), "git_dirty": git_dirty(),
            "torch": str(torch.__version__), "host": str(platform.node()),
        })
    print(f"[jk-match] wrote {args.out}", flush=True)
    if g["verdict"] == "DIVERGENT":
        raise SystemExit("GATE DIVERGENT -- stop and report (§CT11.1)")


# ---------------------------------------------------------------------------
# fit: CPU
# ---------------------------------------------------------------------------

_W: dict = {}   # worker globals, filled before the pool forks


def load_join_inputs(args) -> dict:
    """The design-3 primary's inputs, loaded and aligned by the SAME helpers
    `design3_join.main` uses, with the same identity gate."""
    from src.analysis.c1_argmax import check_shared_identity
    from src.analysis.design3_join import (SHARED_IDENTITY_KEYS,
                                           assert_floor_matches_config,
                                           firing_rate_edges, floor_report,
                                           log_M, qualifying_thetas)
    from src.analysis.seed_axis import align_to_reference
    from src.analysis.seed_axis import seed_order as resolve_gtilde_seed_order

    assert_floor_matches_config(args.config)
    sources = {"gtilde": args.gtilde, "null_reductions": args.null_reductions,
               "d_fs": args.d_fs}
    if args.degenerate_flag is not None:
        sources["degenerate_flag"] = args.degenerate_flag
    a = check_shared_identity({k: v for k, v in sources.items() if k != "d_fs"},
                              SHARED_IDENTITY_KEYS)
    b = check_shared_identity(sources, ("truncation_policy", "scored_row_set",
                                        "n_rows"))
    if a["problems"] or b["problems"]:
        raise SystemExit("PRE-FLIGHT FAILED\n  * " + "\n  * ".join(
            a["problems"] + b["problems"]))

    with h5py.File(args.sim, "r") as hf:
        sim = hf["mnn_sim_anchor_rows"][:]
        at = dict(hf.attrs)
    if at["gate_verdict"] not in ("EXACT", "NEAR"):
        raise SystemExit(f"FATAL: the match gate is {at['gate_verdict']}")
    anchor_labels = [str(s) for s in json.loads(at["anchor_seed_labels"])]
    column_labels = [int(s) for s in json.loads(at["column_seed_labels"])]

    theta_info = qualifying_thetas(args.m1_json)
    if theta_info["n_compare_seeds_in_m1"] != int(at["n_compare_seeds"]):
        raise SystemExit("FATAL: M1's n_compare differs from the match file's")
    thetas = theta_info["qualifying"]

    stored, smeta = _stored_counts(args.rf_counts_all_theta)
    if [str(s) for s in smeta["seed_labels"]] != anchor_labels:
        raise SystemExit("FATAL: anchor order differs from the stored counts")
    if at["gate_verdict"] == "EXACT":
        for t in thetas:
            if not np.array_equal(counts_from_sim(sim, t), stored[t]):
                raise SystemExit(f"FATAL: EXACT gate but counts differ at {t}")

    with h5py.File(args.gtilde, "r") as hf:
        M_hard = hf["M"][:]
        n_rows = int(hf.attrs["n_rows"])
    edges = firing_rate_edges(args.gtilde)
    with h5py.File(args.null_reductions, "r") as hf:
        fire_rate = hf["fire_rate"][:]
        null_seeds = [s.decode() if isinstance(s, bytes) else str(s)
                      for s in hf["seeds"][:]]
    with h5py.File(args.d_fs, "r") as hf:
        d_fs = hf["d_fs_cluster_entropy"][:]
        d_retained = hf["retained_at_floor"][:]
        d_seeds = [int(s) for s in hf["seeds"][:]]
        if str(hf.attrs["adopted_definition"]) != "A_cluster_entropy":
            raise SystemExit("FATAL: d_fs definition is not A (AQ2)")

    null_labels = [s.split("_")[-1] for s in null_seeds]
    gt_order = resolve_gtilde_seed_order(args.gtilde, args.null_reductions)
    fire_rate = align_to_reference(null_labels, fire_rate, anchor_labels)
    M_hard = align_to_reference(gt_order["labels"], M_hard, anchor_labels)
    d_fs = align_to_reference([str(s) for s in d_seeds], d_fs, anchor_labels)
    d_retained = align_to_reference([str(s) for s in d_seeds], d_retained,
                                    anchor_labels)
    logM = log_M(M_hard, str(args.gtilde))
    log_phi = np.log10(np.maximum(fire_rate, 1e-12))
    fl = floor_report(fire_rate, n_rows, edges, logM, d_fs, d_retained)
    return {"sim": sim, "thetas": thetas, "logM": logM, "log_phi": log_phi,
            "d_fs": d_fs, "keep": fl["keep"], "anchor_labels": anchor_labels,
            "column_labels": column_labels, "gate": json.loads(at["gate"]),
            "match_attrs": {k: (v if isinstance(v, (str, int, float, bool))
                                else str(v)) for k, v in at.items()},
            "gtilde_seed_order": gt_order["labels"]}


def _fit_one(y: np.ndarray, rows: np.ndarray, n_trials: int) -> dict:
    from src.analysis.design3_join import fit_arm
    W = _W
    arm = fit_arm(y.astype(np.float64), W["logM"][rows], W["log_phi"][rows],
                  W["d_fs"][rows], W["keep"][rows], n_trials, dispersion=True)
    meta = arm["meta_marginal_effect_betabinomial"]
    failed = [i for i, r in enumerate(arm["per_seed"])
              if (not r["converged"]) or
              (not r["betabinomial"]["optimiser_success"])]
    return {"fe_pp": 100.0 * meta["fixed_effect"],
            "fe_se_pp": 100.0 * meta["fixed_effect_se"],
            "fe_ci95_pp": [100.0 * v for v in meta["fixed_effect_ci95"]],
            "across_seed_mean_pp": 100.0 * meta["across_seed_mean"],
            "across_seed_ci95_pp": [100.0 * v for v in meta["across_seed_ci95"]],
            "n_anchors": int(len(arm["per_seed"])),
            "n_anchor_fits_failed": len(failed)}


def _loo_task(task: tuple) -> dict:
    d, t = task
    W = _W
    y, keep_rows = loo_counts(W["sim"], t, d)
    rows = np.flatnonzero(keep_rows)
    n_trials = W["sim"].shape[1] - 2
    r = _fit_one(y, rows, n_trials)
    r.update({"left_out_column": int(d),
              "left_out_seed": int(W["column_labels"][d]), "theta": t})
    return r


def cmd_fit(args) -> None:
    from src.sae.train import git_dirty, git_sha
    inp = load_join_inputs(args)
    _W.update(inp)
    sim, thetas = inp["sim"], inp["thetas"]
    n_anchors, n_seeds, _ = sim.shape
    all_rows = np.arange(n_anchors)

    qualifying = list(thetas)
    if getattr(args, "thetas", None):
        # Amendment CT §CT24/§CT25: a new arm's slope is declared AT a theta
        # (0.70), whether or not X2's rule qualifies it on that arm; the
        # qualifying set is recorded beside it, never silently substituted.
        thetas = [round(float(t), 2) for t in args.thetas]
    ref = (json.loads(args.reference_join.read_text())
           if args.reference_join is not None else None)
    ref_rows = ({round(float(r["theta"]), 2): r for r in ref["theta_curve"]["rows"]}
                if ref is not None else None)

    # ---- the control: the full ensemble, before any leave-one-out fit ----
    full = {}
    for t in thetas:
        full[t] = _fit_one(counts_from_sim(sim, t), all_rows, n_seeds - 1)
        print(f"[jk-fit] full theta {t:.2f}: FE {full[t]['fe_pp']:+.6f} pp",
              flush=True)
    if ref_rows is None:
        # A new arm has no join of record to reproduce; its code path is the
        # anchors' (§CT3's control passed on it), and that is recorded.
        ctrl = {"verdict": "NO_REFERENCE",
                "note": "new arm (amendment CT §CT24/§CT25): no join of record"}
    else:
        ctrl = control_verdict(
            [full[t]["fe_pp"] for t in thetas],
            [ref_rows[t]["marginal_effect_percentage_points_betabinomial"]
             ["fixed_effect"] for t in thetas])
    print(f"[jk-fit] CONTROL {ctrl}", flush=True)

    out = {"identity": {
        "measurement": "jackknife_over_dictionaries_design3_primary",
        "governing_amendment": "CT3 / CT11.1",
        "arm": "primary_hard_bin", "likelihood": "betabinomial_fixed_effect",
        "units": "percentage points of r_f per SD of log M",
        "n_dictionaries": int(n_seeds), "n_anchors": int(n_anchors),
        "n_compare_full": int(n_seeds - 1), "n_compare_loo": int(n_seeds - 2),
        "column_seed_labels": inp["column_labels"],
        "anchor_seed_labels": inp["anchor_labels"],
        "inputs": {k: {"path": str(getattr(args, k)),
                       "sha256_16": _sha16(getattr(args, k))}
                   for k in ("sim", "rf_counts_all_theta", "m1_json", "gtilde",
                             "null_reductions", "d_fs", "reference_join")
                   if getattr(args, k) is not None},
        "thetas_fitted": thetas, "thetas_qualifying_x2": qualifying,
        "match_gate": inp["gate"], "match_attrs": inp["match_attrs"],
        "fit_device_class": "cpu", "host": platform.node(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "git_sha": git_sha(), "git_dirty": git_dirty()},
        "control_full_refit_vs_reference": ctrl,
        "full_fit": {f"{t:.2f}": full[t] for t in thetas}}
    if ctrl["verdict"] == "STOP":
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        raise SystemExit("CONTROL FAILED -- stop and report (§CT11.1)")

    tasks = [(d, t) for d in range(n_seeds) for t in thetas]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(_loo_task, tasks, chunksize=1))

    per_theta = {}
    for t in thetas:
        rs = [r for r in res if r["theta"] == t]
        rs.sort(key=lambda r: r["left_out_column"])
        failed = [r for r in rs if r["n_anchor_fits_failed"] > 0]
        jk = jackknife([r["fe_pp"] for r in rs], full[t]["fe_pp"])
        f = full[t]
        jk_ci = [f["fe_pp"] - 1.959964 * jk["se"], f["fe_pp"] + 1.959964 * jk["se"]]
        ivs = {"fixed_effect": f["fe_ci95_pp"],
               "across_seed": f["across_seed_ci95_pp"], "jackknife": jk_ci}
        per_theta[f"{t:.2f}"] = {
            "full_fe_pp": f["fe_pp"], "full_fe_se_pp": f["fe_se_pp"],
            "jackknife": jk, "W_jackknife_se_over_fe_se": jk["se"] / f["fe_se_pp"],
            "intervals_95_pp": ivs, "widest": widest_interval(ivs),
            "n_loo_fits_failed": len(failed),
            "full_fit_anchor_failures": f["n_anchor_fits_failed"],
            "loo": rs}
        print(f"[jk-fit] theta {t:.2f}: jk SE {jk['se']:.5f} pp, W "
              f"{per_theta[f'{t:.2f}']['W_jackknife_se_over_fe_se']:.3f}, "
              f"failed {len(failed)}", flush=True)
    out["per_theta"] = per_theta
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[jk-fit] wrote {args.out}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("match")
    m.add_argument("--ckpts", nargs="+", type=Path, required=True)
    m.add_argument("--compare-ckpts", nargs="*", type=Path, default=[],
                   help="comparison-only seeds; none for an arm whose every "
                        "seed is an anchor (amendment CT §CT24/§CT25)")
    m.add_argument("--accept-git-shas", nargs="+", default=None,
                   help="rf_marginal's CS3c flag: the exact git_sha set a pool "
                        "may carry (the 40,960 arm)")
    m.add_argument("--expect-total-ckpts", type=int, required=True)
    m.add_argument("--rf-counts-all-theta", type=Path, required=True)
    m.add_argument("--record-device-class", required=True,
                   help="the class the stored counts were produced on")
    m.add_argument("--record-device-source", required=True,
                   help="where that class was read from (the counts carry no attr)")
    m.add_argument("--device", default="cuda")
    m.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("fit")
    f.add_argument("--sim", type=Path, required=True)
    f.add_argument("--rf-counts-all-theta", type=Path, required=True)
    f.add_argument("--m1-json", type=Path, required=True)
    f.add_argument("--gtilde", type=Path, required=True)
    f.add_argument("--null-reductions", type=Path, required=True)
    f.add_argument("--d-fs", type=Path, required=True)
    f.add_argument("--degenerate-flag", type=Path, default=None)
    f.add_argument("--reference-join", type=Path, default=None,
                   help="the join of record the full refit must reproduce; "
                        "absent only for a new arm (control NO_REFERENCE)")
    f.add_argument("--thetas", type=float, nargs="+", default=None,
                   help="fit at these thetas instead of X2's qualifying set")
    f.add_argument("--config", type=Path, default=None)
    f.add_argument("--workers", type=int, default=8)
    f.add_argument("--out", type=Path, required=True)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "match":
        cmd_match(args)
    else:
        if args.config is None:
            from src.experiment_config import DEFAULT_CONFIG_PATH
            args.config = DEFAULT_CONFIG_PATH
        cmd_fit(args)


if __name__ == "__main__":
    main()
