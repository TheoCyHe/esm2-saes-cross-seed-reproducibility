"""§CT28: threshold, threshold-free and shared-scale rescoring of the existing arms.

Every fit is design 3 (`primary_hard_bin`, beta-binomial, the arm's own `log M`)
through `jackknife_dictionaries.load_join_inputs` and `design3_join.fit_arm`,
the code of §CT24-R2 and §CT25-R. What this module adds:

  * the per-seed values those fits compute and `_fit_one` discards, so a slope
    can be put on another scale (§CT28-F item 2): per unit of `log M`
    (`AME_s / sd_s`) and per SD of the record's primary fit set
    (`AME_s / sd_s * SD_REF`), each pooled by `meta_analyse`, the combination of
    record. Per seed that is an exact rescaling of the fitted effect; the pooled
    value is a fixed effect of rescaled per-seed values, not a rescaled pooled
    value;
  * the rate-matched threshold θ*_arm (§CT4's rule, `theta_star_subset`) on an
    arm's stored `mnn_sim_anchor_rows`, over a named comparison set, with the
    leave-one-dictionary-out jackknife on both scales (§CT28-F item 3);
  * `s_f` per arm (§CT16 generalised, §CT28-F item 4);
  * the rules at k = 64 and rules 2-3 at the widths (§CT28-F item 5);
  * the scoring of CT28-P1-P5 (§CT28-F item 7), derived, never asserted.

Subcommands: `arm`, `sf` (GPU), `sffit`, `rules`, `widthrules`, `score`.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

SD_REF = 2.067756996018893          # mean of the 20 anchors' per-anchor SD of log M
K64_TARGET_N19 = 0.495814           # NoR §18.1 (CT28-pre item 3)
WIDTH_TARGET_N9 = 0.4957942708333333  # rf_marginal_m1_w10240_n9.json (CT28-pre item 4)
K64_RULE_TARGET = 0.21698858963815787  # rf_marginal_m1_k64_ct.json at 0.70
K32_N19_FE_CI = (0.1162, 0.3950)    # NoR §20.20 (a) rule 1, as CT28-P1 quotes it
P2_BAND = (-0.5, 1.0)
Z95 = 1.959964
SCALES = ("own", "per_unit", "shared")

_G: dict = {}    # worker globals, filled before the pool forks


# ---------------------------------------------------------------------------
# Pure functions -- tests/test_ct28_rescore.py exercises each
# ---------------------------------------------------------------------------

def scale_factors(sd: np.ndarray) -> dict:
    """Per-seed multipliers taking an effect per SD of the seed's own `log M`
    to each reported scale."""
    sd = np.asarray(sd, dtype=float)
    return {"own": np.ones_like(sd), "per_unit": 1.0 / sd, "shared": SD_REF / sd}


def pooled_on_scales(ame: np.ndarray, se: np.ndarray, sd: np.ndarray) -> dict:
    """`meta_analyse` of the per-seed effects on every scale, in pp."""
    from src.analysis.design3_join import meta_analyse
    out = {}
    for k, f in scale_factors(sd).items():
        m = meta_analyse(np.asarray(ame) * f, np.asarray(se) * f)
        out[k] = {"fe_pp": 100.0 * m["fixed_effect"],
                  "fe_se_pp": 100.0 * m["fixed_effect_se"],
                  "fe_ci95_pp": [100.0 * v for v in m["fixed_effect_ci95"]],
                  "across_seed_mean_pp": 100.0 * m["across_seed_mean"],
                  "across_seed_ci95_pp": [100.0 * v for v in m["across_seed_ci95"]]}
    return out


def rel_diff(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.array_equal(a, b):
        return 0.0
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def scale_verdict(delta: float, control_delta: float) -> str:
    """§CT28-F item 2's verdict, total over delta >= 0."""
    if delta == 0.0:
        return "BIT_IDENTICAL"
    if delta <= 1e-6:
        return "WITHIN_1E-6"
    if delta <= control_delta:
        return "WITHIN_CONTROL"
    return "DIFFERS"


def intervals_overlap(a, b) -> bool:
    return bool(a[0] <= b[1] and b[0] <= a[1])


def score_p1(fe_ci, measurable: bool) -> str:
    if not measurable:
        return "UNMEASURABLE"
    return "HIT" if intervals_overlap(fe_ci, K32_N19_FE_CI) else "MISS"


def score_p2(estimates: dict) -> str:
    """{width: fe_pp or None (unmeasurable)}: HIT if all four measurable and in
    the band; MISS if any measurable one is outside; else UNMEASURABLE."""
    lo, hi = P2_BAND
    meas = {w: v for w, v in estimates.items() if v is not None}
    if any(not (lo <= v <= hi) for v in meas.values()):
        return "MISS"
    return "HIT" if len(meas) == len(estimates) == 4 else "UNMEASURABLE"


def score_p3(intervals: dict) -> str:
    """{arm: fe_ci or None}: HIT if all five measurable and wholly above 0;
    MISS if any measurable interval contains 0 or lies below; else UNMEASURABLE."""
    meas = {k: v for k, v in intervals.items() if v is not None}
    if any(v[0] <= 0.0 for v in meas.values()):
        return "MISS"
    return "HIT" if len(meas) == len(intervals) == 5 else "UNMEASURABLE"


def score_p5(rule_fe_shared: dict, diff_shared: float | None) -> dict:
    """Spread (max - min) of rules 1-4 at their rate-matched thresholds, shared
    scale, against rule 1's k64 - k32 difference at 0.70 on the shared scale."""
    if diff_shared is None or any(v is None for v in rule_fe_shared.values()) \
            or len(rule_fe_shared) != 4:
        return {"verdict": "UNMEASURABLE", "spread": None, "difference": diff_shared}
    v = list(rule_fe_shared.values())
    spread = float(max(v) - min(v))
    return {"verdict": "HIT" if spread >= diff_shared else "MISS",
            "spread": spread, "difference": float(diff_shared)}


# ---------------------------------------------------------------------------
# Fitting on a comparison set
# ---------------------------------------------------------------------------

def _fit_scales(y: np.ndarray, rows: np.ndarray, n_trials: int, z_divisor=None) -> dict:
    """`fit_arm` (dispersion on) on the worker globals' rows, every per-seed
    value kept, pooled on every scale."""
    from src.analysis.design3_join import fit_arm
    W = _G
    arm = fit_arm(y.astype(np.float64), W["logM"][rows], W["log_phi"][rows],
                  W["d_fs"][rows], W["keep"][rows], n_trials, dispersion=True,
                  z_divisor=z_divisor)
    ps = arm["per_seed"]
    ame = np.array([r["betabinomial"]["marginal_effect"]["ame"] for r in ps])
    se = np.array([r["betabinomial"]["marginal_effect"]["se"] for r in ps])
    sd = np.array([r["sd_logM"] for r in ps])
    failed = [i for i, r in enumerate(ps)
              if (not r["converged"]) or (not r["betabinomial"]["optimiser_success"])]
    out = pooled_on_scales(ame, se, sd)
    # the own-scale pool is the one `_fit_one` reports; it is recomputed here from
    # the same per-seed values and must equal fit_arm's own meta exactly.
    own_meta = arm["meta_marginal_effect_betabinomial"]
    if out["own"]["fe_pp"] != 100.0 * own_meta["fixed_effect"]:
        raise SystemExit("FATAL: the own-scale pool differs from fit_arm's own")
    return {**out, "n_anchors": int(len(ps)), "n_anchor_fits_failed": len(failed),
            "per_seed": [{"ame": float(a), "se": float(s), "sd_logM": float(d),
                          "mean_logM": float(r["mean_logM"])}
                         for a, s, d, r in zip(ame, se, sd, ps)]}


def _counts(sim, theta, row_idx, row_labels, col_labels, use) -> tuple[np.ndarray, int]:
    from src.analysis.ct_robustness import subset_counts
    return subset_counts(sim[row_idx], theta, row_labels, col_labels, use)


def _loo_task(task) -> dict:
    """Leave dictionary `d` out of the comparison set: its row leaves the fit
    set and every other row loses it as a comparator."""
    d, theta = task
    W = _G
    rows_lab = [r for r in W["row_labels"] if r != d]
    use = [u for u in W["use"] if u != d]
    ridx = np.array([W["row_index"][r] for r in rows_lab])
    y, n = _counts(W["sim"], theta, ridx, rows_lab, W["col_labels"], use)
    r = _fit_scales(y, ridx, n)
    return {"left_out": d, "theta": theta, "n_anchor_fits_failed": r["n_anchor_fits_failed"],
            **{k: r[k]["fe_pp"] for k in SCALES}}


def _map(fn, tasks, workers):
    if workers <= 1:
        return [fn(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, tasks, chunksize=1))


def fit_with_jackknife(theta: float, workers: int) -> dict:
    """The full fit at `theta` on the comparison set, and the jackknife over its
    dictionaries on every scale."""
    from src.analysis import jackknife_dictionaries as jk
    W = _G
    ridx = np.array([W["row_index"][r] for r in W["row_labels"]])
    y, n = _counts(W["sim"], theta, ridx, W["row_labels"], W["col_labels"], W["use"])
    full = _fit_scales(y, ridx, n)
    loo = _map(_loo_task, [(d, theta) for d in W["use"]], workers)
    n_failed = sum(r["n_anchor_fits_failed"] > 0 for r in loo)
    for k in SCALES:
        j = jk.jackknife([r[k] for r in loo], full[k]["fe_pp"])
        full[k]["jackknife"] = j
        full[k]["jackknife_ci95_pp"] = [full[k]["fe_pp"] - Z95 * j["se"],
                                        full[k]["fe_pp"] + Z95 * j["se"]]
        iv = {"fixed_effect": full[k]["fe_ci95_pp"],
              "across_seed": full[k]["across_seed_ci95_pp"],
              "jackknife": full[k]["jackknife_ci95_pp"]}
        full[k]["widest"] = max(iv, key=lambda m: iv[m][1] - iv[m][0])
    full.update({"theta": theta, "n_trials": n, "mean_r_f": float(y.mean() / n),
                 "n_loo_fits_failed": n_failed,
                 "loo": [{kk: r[kk] for kk in ("left_out", "own", "shared",
                                               "n_anchor_fits_failed")} for r in loo]})
    return full


def _load_arm(args) -> dict:
    """§CT3's loader, then the comparison set: `--use` labels (default: every
    column), rows = the anchors in it."""
    from src.analysis import jackknife_dictionaries as jk
    inp = jk.load_join_inputs(args)
    anchors = [str(a) for a in inp["anchor_labels"]]
    cols = [str(c) for c in inp["column_labels"]]
    use = [str(u) for u in (args.use or cols)]
    rows = [a for a in anchors if a in use]
    if len(rows) != len(use):
        raise SystemExit(f"FATAL: every comparison dictionary must be an anchor "
                         f"here; rows {rows} vs use {use}")
    _G.update(inp)
    _G.update({"row_labels": rows, "use": use, "col_labels": cols,
               "row_index": {a: i for i, a in enumerate(anchors)}})
    return inp


def _identity(args, extra: dict) -> dict:
    from src.analysis.jackknife_dictionaries import _sha16
    from src.sae.train import git_dirty, git_sha
    ins = {}
    for k in ("sim", "rf_counts_all_theta", "m1_json", "gtilde", "null_reductions",
              "d_fs", "stored_slope", "act_counts", "oneway_h5", "rules_sim_h5",
              "rules_h5", "rf_counts_070"):
        p = getattr(args, k, None)
        if p is not None:
            ins[k] = {"path": str(p), "sha256_16": _sha16(Path(p))}
    return {"governing_amendment": "CT28", "inputs": ins, "sd_ref": SD_REF,
            "fit_device_class": "cpu", "host": platform.node(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "git_sha": git_sha(), "git_dirty": git_dirty(), **extra}


def _control(mine: float, ref: float | None) -> dict:
    from src.analysis.jackknife_dictionaries import control_verdict
    if ref is None:
        return {"verdict": "NO_REFERENCE"}
    c = control_verdict([mine], [ref])
    c["reference_fe_pp"] = ref
    return c


# ---------------------------------------------------------------------------
# arm
# ---------------------------------------------------------------------------

def cmd_arm(args) -> None:
    from src.analysis.ct_robustness import theta_star_subset
    _load_arm(args)
    W = _G
    out = {"identity": _identity(args, {"measurement": "ct28_arm", "arm": args.arm_name,
                                        "comparison_set": W["use"],
                                        "n_compare": len(W["use"]) - 1,
                                        "theta_star_target_rate": args.target})}
    ref = None
    if args.stored_slope is not None:
        ref = json.loads(Path(args.stored_slope).read_text(encoding="utf-8"))[
            "full_fit"]["0.70"]["fe_pp"]
    elif args.reference_fe is not None:
        ref = float(args.reference_fe)
    f70 = fit_with_jackknife(0.70, args.workers)
    out["control_0.70_vs_record"] = _control(f70["own"]["fe_pp"], ref)
    print(f"[ct28-arm] {args.arm_name} 0.70 own FE {f70['own']['fe_pp']!r} "
          f"control {out['control_0.70_vs_record']}", flush=True)
    if out["control_0.70_vs_record"]["verdict"] not in ("BIT_IDENTICAL", "NO_REFERENCE"):
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        raise SystemExit("CONTROL FAILED -- stop that arm (§CT28-F item 1)")
    out["at_0.70"] = f70
    sds = np.array([r["sd_logM"] for r in f70["per_seed"]])
    out["sd_logM_table"] = {"mean": float(sds.mean()), "min": float(sds.min()),
                            "max": float(sds.max()), "per_seed": sds.tolist(),
                            "row_labels": W["row_labels"], "sd_ref": SD_REF}

    ridx = np.array([W["row_index"][r] for r in W["row_labels"]])
    ts = theta_star_subset(W["sim"][ridx], W["row_labels"], W["col_labels"], W["use"],
                           args.target)
    out["theta_star"] = ts
    if ts.get("theta_star") is None:
        out["at_theta_star"] = {"status": "UNMEASURABLE", "cause": ts.get("status")}
    else:
        t = float(ts["theta_star"])
        out["at_theta_star"] = fit_with_jackknife(t, args.workers)
        if args.assert_theta_star_is_070:
            y_t, _ = _counts(W["sim"], t, ridx, W["row_labels"], W["col_labels"], W["use"])
            y_7, _ = _counts(W["sim"], 0.70, ridx, W["row_labels"], W["col_labels"], W["use"])
            n_diff = int((y_t != y_7).sum())
            same = out["at_theta_star"]["own"]["fe_pp"] == f70["own"]["fe_pp"]
            out["deduction_theta_star_equals_0.70"] = {
                "n_cells_differing": n_diff, "fe_equal": bool(same),
                "verdict": "HOLDS" if (n_diff == 0 and same) else "FAILS"}
        print(f"[ct28-arm] θ* {t!r}: own FE {out['at_theta_star']['own']['fe_pp']:+.4f} "
              f"{out['at_theta_star']['own']['fe_ci95_pp']}", flush=True)

    if args.verify_scale:
        out["scale_verification"] = verify_scale(f70)
        print(f"[ct28-arm] scale verification {out['scale_verification']['verdict']}",
              flush=True)
    if args.act_counts is not None:
        out["rule4_activation"] = rule4_shared(args, ridx)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[ct28-arm] wrote {args.out}", flush=True)


def verify_scale(f70: dict) -> dict:
    """§CT28-F item 2: refit at 0.70 with the covariate divided by SD_REF in
    place of each seed's SD, against the analytic rescaling; the x2 control
    measures the fit's own sensitivity to a pure rescaling."""
    W = _G
    ridx = np.array([W["row_index"][r] for r in W["row_labels"]])
    y, n = _counts(W["sim"], 0.70, ridx, W["row_labels"], W["col_labels"], W["use"])
    ame = np.array([r["ame"] for r in f70["per_seed"]])
    sd = np.array([r["sd_logM"] for r in f70["per_seed"]])
    rs = _fit_scales(y, ridx, n, z_divisor=SD_REF)
    ame_rs = np.array([r["ame"] for r in rs["per_seed"]])
    want = ame * SD_REF / sd
    delta = max(rel_diff(ame_rs, want), rel_diff([rs["own"]["fe_pp"]],
                                                  [f70["shared"]["fe_pp"]]))
    # the control: z x 2 (divisor sd/2, a factor exact in floating point), so a
    # unit of the new covariate is half an SD and each effect halves exactly.
    ctl = _fit_scales(y, ridx, n, z_divisor=sd / 2.0)
    ame_c = np.array([r["ame"] for r in ctl["per_seed"]])
    se0 = np.array([r["se"] for r in f70["per_seed"]])
    delta_c = max(rel_diff(ame_c, ame / 2.0), rel_diff(
        [ctl["own"]["fe_pp"]],
        [pooled_on_scales(ame / 2.0, se0 / 2.0, np.ones_like(sd))["own"]["fe_pp"]]))
    return {"delta": delta, "control_x2_delta": delta_c,
            "verdict": scale_verdict(delta, delta_c),
            "refit_shared_fe_pp": rs["own"]["fe_pp"],
            "analytic_shared_fe_pp": f70["shared"]["fe_pp"]}


def rule4_shared(args, ridx) -> dict:
    """§CT28-F item 5: §CT24-R3's activation fit at θ*, repeated with the
    per-seed values kept; must reproduce the record's own-scale figure."""
    W = _G
    with h5py.File(args.act_counts, "r") as hf:
        y = hf["match_count_theta_star"][:]
        labels = [str(s) for s in json.loads(hf.attrs["seed_labels"])]
        n = int(hf.attrs["n_compare_seeds"])
        ts = json.loads(hf.attrs["theta_star"])
    if labels != W["row_labels"]:
        raise SystemExit(f"FATAL: activation anchor order {labels} != {W['row_labels']}")
    r = _fit_scales(y, ridx, n)
    ctrl = _control(r["own"]["fe_pp"], args.act_reference_fe)
    r.update({"theta_star": ts, "control_vs_ct24_r3": ctrl})
    if ctrl["verdict"] not in ("BIT_IDENTICAL", "NO_REFERENCE"):
        raise SystemExit(f"CONTROL FAILED for rule 4: {ctrl}")
    return r


# ---------------------------------------------------------------------------
# s_f
# ---------------------------------------------------------------------------

def cmd_sf(args) -> None:
    """§CT16's step on an arm's checkpoints (every seed an anchor), gated on
    the arm's stored mnn_sim cell for cell."""
    import torch
    from src.analysis.artifact_identity import device_class
    from src.analysis.ct_robustness import oneway_and_mnn_anchor_rows
    from src.analysis.rf_marginal import check_unit_norm, load_decoder
    from src.sae.train import git_dirty, git_sha
    num = lambda p: int(str(Path(p).name).rsplit("_", 1)[-1])
    ckpts = sorted(args.ckpts, key=num)
    decs, labels = [], []
    for p in ckpts:
        W, meta = load_decoder(Path(p), args.device)
        if check_unit_norm(W)["verdict"] != "PASS":
            raise SystemExit(f"FATAL: {p} decoder not unit-norm")
        decs.append(W)
        labels.append(int(meta["seed"]))
    one, mnn = oneway_and_mnn_anchor_rows(decs, len(decs))
    with h5py.File(args.sim, "r") as hf:
        ref = hf["mnn_sim_anchor_rows"][:]
        ref_cols = [int(s) for s in json.loads(hf.attrs["column_seed_labels"])]
        ref_rows = [int(s) for s in json.loads(hf.attrs["anchor_seed_labels"])]
    if ref_cols != labels or ref_rows != labels:
        raise SystemExit(f"FATAL: labels {labels} != the stored file's {ref_rows}/{ref_cols}")
    both_fin = np.isfinite(ref) & np.isfinite(mnn)
    n_diff = int(((np.isfinite(ref) != np.isfinite(mnn))
                  | (both_fin & (ref != mnn))).sum())
    gate = "EXACT" if n_diff == 0 else "DIFFERS"
    run_dev = str(device_class(decs[0]))
    with h5py.File(args.out, "w") as ho:
        ho.create_dataset("oneway_max_cos_anchor_rows", data=one, compression="gzip")
        ho.attrs.update({"measurement": "ct28_oneway_max_decoder_cosine",
                         "governing_amendment": "CT28 (CT16 generalised)",
                         "arm": args.arm_name,
                         "anchor_seed_labels": json.dumps(labels),
                         "column_seed_labels": json.dumps(labels),
                         "gate_vs_stored_mnn_sim": gate, "n_cells_differing": n_diff,
                         "stored_sim": str(args.sim),
                         "gather_device_class": run_dev,
                         "git_sha": git_sha(), "git_dirty": git_dirty(),
                         "torch": str(torch.__version__), "host": platform.node()})
    print(f"[ct28-sf] {args.arm_name}: gate {gate} ({n_diff} cells differ) on {run_dev}",
          flush=True)
    if gate != "EXACT":
        raise SystemExit("GATE DIFFERS -- stop that arm (§CT28-F item 4)")


def sf_on_comparison_set(one: np.ndarray, labels: list, use: list) -> np.ndarray:
    """`s_f` over a comparison set (§CT9-D item D4): for each anchor in `use`,
    the mean of its one-way maximum decoder cosine over the other seeds in
    `use`, rows and columns taken by label, never by position (gotcha 29).
    With `use` equal to `labels` it is §CT28-F item 4's reduction unchanged."""
    labels = [str(s) for s in labels]
    use = [str(u) for u in use]
    if len(set(use)) != len(use) or not set(use) <= set(labels):
        raise SystemExit(f"FATAL: comparison set {use} is not a set of the file's labels {labels}")
    pos = [labels.index(u) for u in use]
    sub = one[np.ix_(pos, pos)]
    n = len(use)
    off = np.ones((n, n), dtype=bool)
    off[np.arange(n), np.arange(n)] = False
    return np.stack([sub[a][off[a]].mean(axis=0) for a in range(n)])


def cmd_sffit(args) -> None:
    from src.analysis.ct_robustness import gaussian_penalised_fit
    from src.analysis.design3_fit import build_design, fit_design3, meta_analyse
    inp = _load_arm(args)
    W = _G
    with h5py.File(args.oneway_h5, "r") as hf:
        one = hf["oneway_max_cos_anchor_rows"][:]
        if hf.attrs["gate_vs_stored_mnn_sim"] != "EXACT":
            raise SystemExit("FATAL: the s_f file's gate is not EXACT")
        a_lab = [str(s) for s in json.loads(hf.attrs["anchor_seed_labels"])]
    if a_lab != [str(a) for a in inp["anchor_labels"]]:
        raise SystemExit("FATAL: anchor order differs")
    if W["use"] != W["row_labels"]:
        # s_f rows follow `use`; the join arrays follow the anchors' order
        raise SystemExit(f"FATAL: --use {W['use']} is not in anchor order {W['row_labels']}")
    sf = sf_on_comparison_set(one, a_lab, W["use"])
    n_a, n_s = len(W["use"]), len(W["use"])
    if not np.all(np.isfinite(sf)):
        raise SystemExit("FATAL: non-finite s_f")
    ridx = np.array([W["row_index"][r] for r in W["row_labels"]])
    keep = W["keep"][ridx]
    logM, log_phi, d_fs = W["logM"][ridx], W["log_phi"][ridx], W["d_fs"][ridx]
    q = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    dist = {"n": int(keep.sum()), "mean": float(sf[keep].mean()),
            "sd": float(sf[keep].std(ddof=1)),
            "quantiles": dict(zip([str(x) for x in q], np.percentile(sf[keep], q).tolist()))}
    pam = [float(sf[a][keep[a]].mean()) for a in range(n_a)]
    y070, n_trials = _counts(inp["sim"], 0.70, ridx, W["row_labels"], W["col_labels"], W["use"])
    if n_trials != n_s - 1:
        raise SystemExit(f"FATAL: {n_trials} comparators per anchor, expected {n_s - 1}")
    rows = []
    for a in range(n_a):
        m = keep[a]
        lm = logM[a][m]
        sd = float(lm.std())
        z = (lm - lm.mean()) / sd
        des = build_design(z, log_phi[a][m], d_fs[a][m])
        lam = fit_design3(y070[a][m].astype(np.float64), n_s - 1, des["X"], des["S"])["lambda"]
        g = gaussian_penalised_fit(sf[a][m].astype(np.float64), des["X"], des["S"], lam)
        i = des["idx_logM"]
        rows.append({"anchor": W["use"][a], "lambda_inherited": lam, "edf": g["edf"],
                     "b_z": float(g["beta"][i]), "se_b_z": float(np.sqrt(g["cov"][i, i])),
                     "sd_logM": sd})
    b = np.array([r["b_z"] for r in rows])
    s = np.array([r["se_b_z"] for r in rows])
    sds = np.array([r["sd_logM"] for r in rows])
    slope = {}
    for k, f in scale_factors(sds).items():
        m = meta_analyse(b * f, s * f)
        slope[k] = {"fixed_effect": m["fixed_effect"], "fixed_effect_ci95": m["fixed_effect_ci95"],
                    "across_seed_mean": m["across_seed_mean"],
                    "across_seed_ci95": m["across_seed_ci95"]}
    out = {"identity": _identity(args, {"measurement": "ct28_threshold_independent_summary",
                                        "arm": args.arm_name, "n_compare": n_s - 1,
                                        "comparison_set": W["use"],
                                        "units_slope": "cosine per SD of log M (own / shared) "
                                                       "or per unit of log M",
                                        "likelihood": "gaussian_identity_lambda_inherited"}),
           "distribution": dist, "per_anchor_mean": pam,
           "all_cells": {"n": int(sf.size), "mean": float(sf.mean()),
                         "per_anchor_mean": [float(r.mean()) for r in sf]},
           "slope": slope, "per_anchor": rows}
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[ct28-sffit] {args.arm_name}: own FE {slope['own']['fixed_effect']!r} "
          f"{slope['own']['fixed_effect_ci95']}", flush=True)


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------

def cmd_rules(args) -> None:
    """Rules 1-3 at k = 64 from `rule_interior_mass --out-sim-h5`, at 0.70 and
    at each rule's rate-matched threshold, with the jackknife on both scales."""
    from src.analysis.ct_robustness import RULE_DATASETS, theta_star_subset
    from src.analysis.seed_axis import align_to_reference
    inp = _load_arm(args)
    W = _G
    anchors = W["row_labels"]
    if anchors != W["use"] or anchors != [str(c) for c in inp["column_labels"]]:
        raise SystemExit("FATAL: rules are fitted on the full arm, every seed an anchor")
    out = {"identity": _identity(args, {"measurement": f"ct28_rules_{args.arm_name}", "arm": args.arm_name,
                                        "theta_star_target_rate": args.target})}
    sims, stored = {}, {}
    with h5py.File(args.rules_sim_h5, "r") as hf:
        labs = [str(s) for s in json.loads(hf.attrs["anchor_seed_labels"])]
        for k, ds in RULE_DATASETS.items():
            s = align_to_reference(labs, hf[f"sim_{ds}"][:], anchors)
            sims[k] = np.stack([align_to_reference(labs, s[i], anchors)
                                for i in range(s.shape[0])])
    with h5py.File(args.rules_h5, "r") as hf:
        rlabs = [str(x) for x in json.loads(hf.attrs["seed_labels"])]
        stored = {k: align_to_reference(rlabs, hf[v][:], anchors)
                  for k, v in RULE_DATASETS.items()}
    from src.analysis.ct_robustness import subset_counts
    gate = {k: int((subset_counts(sims[k], 0.70, anchors, anchors, anchors)[0]
                    != stored[k]).sum()) for k in RULE_DATASETS}
    out["gate_sim_vs_own_counts_cells_differing"] = gate
    if any(gate.values()):
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        raise SystemExit(f"GATE FAILED: {gate}")
    with h5py.File(args.rf_counts_070, "r") as hf:
        rf = align_to_reference([str(x) for x in json.loads(hf.attrs["seed_labels"])],
                                hf["match_count"][:], anchors)
    out["rule1_cpu_vs_gpu_counts_cells_differing"] = int((stored["rule1"] != rf).sum())
    res = {}
    for k in RULE_DATASETS:
        _G["sim"] = sims[k]
        _G["col_labels"] = anchors
        ts = theta_star_subset(sims[k], anchors, anchors, anchors, args.target)
        r = {"at_0.70": fit_with_jackknife(0.70, args.workers), "theta_star": ts}
        r["at_theta_star"] = (fit_with_jackknife(float(ts["theta_star"]), args.workers)
                              if ts.get("theta_star") is not None
                              else {"status": "UNMEASURABLE", "cause": ts.get("status")})
        res[k] = r
        print(f"[ct28-rules] {k}: θ* {ts.get('theta_star')}", flush=True)
    out["rules"] = res
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")


def cmd_widthrules(args) -> None:
    """Rules 2 and 3 at 0.70 from the CS tier's stored counts (no per-comparator
    indicators, so no jackknife; CT28-pre item 9)."""
    from src.analysis.ct_robustness import RULE_DATASETS
    from src.analysis.seed_axis import align_to_reference
    _load_arm(args)
    W = _G
    anchors = W["row_labels"]
    ridx = np.array([W["row_index"][r] for r in anchors])
    with h5py.File(args.rules_h5, "r") as hf:
        rlabs = [str(x) for x in json.loads(hf.attrs["seed_labels"])]
        n = int(hf.attrs["n_compare_seeds"])
        if float(hf.attrs["theta"]) != 0.7:
            raise SystemExit("FATAL: the CS counts are not at θ = 0.70")
        counts = {k: align_to_reference(rlabs, hf[v][:], anchors)
                  for k, v in RULE_DATASETS.items()}
    with h5py.File(args.rf_counts_070, "r") as hf:
        rf = align_to_reference([str(x) for x in json.loads(hf.attrs["seed_labels"])],
                                hf["match_count"][:], anchors)
    out = {"identity": _identity(args, {"measurement": "ct28_width_rules_0.70",
                                        "arm": args.arm_name, "n_compare": n}),
           "rule1_cpu_vs_gpu_counts_cells_differing": int((counts["rule1"] != rf).sum()),
           "rules": {}}
    for k in ("rule2", "rule3"):
        r = _fit_scales(counts[k], ridx, n)
        r["mean_r_f"] = float(counts[k].mean() / n)
        out["rules"][k] = r
        print(f"[ct28-wrules] {args.arm_name} {k}: own FE {r['own']['fe_pp']:+.4f}", flush=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

WIDTHS = ("w2560", "w10240", "w20480", "w40960")


def _measurable(arm: dict, where: str) -> bool:
    """An arm's figure at `where` counts only if its control passed and the
    fit at that threshold exists with no failed anchor fit (§CT28-F items 1, 3)."""
    if arm["control_0.70_vs_record"]["verdict"] not in ("BIT_IDENTICAL", "NO_REFERENCE"):
        return False
    f = arm.get(where) or {}
    return "own" in f and f.get("n_anchor_fits_failed", 1) == 0


def score_all(arms: dict, sffits: dict, rules: dict) -> dict:
    """CT28-P1-P5 from the stored outputs, derived (gotcha 20).
    `arms`: {k64, k32n19, w2560..w40960: ct28_arm JSON}; `sffits`: {k64, w*: JSON};
    `rules`: the k64 rules JSON."""
    k64, k32 = arms["k64"], arms["k32n19"]
    out = {}
    m1 = _measurable(k64, "at_theta_star")
    out["CT28-P1"] = {"verdict": score_p1(k64["at_theta_star"]["own"]["fe_ci95_pp"]
                                          if m1 else None, m1),
                      "k64_fe_ci95_pp": k64.get("at_theta_star", {}).get("own", {}).get("fe_ci95_pp"),
                      "k64_widest": k64.get("at_theta_star", {}).get("own", {}).get("widest"),
                      "k32_fe_ci95_pp": list(K32_N19_FE_CI)}
    est = {}
    for w in WIDTHS:
        a = arms[w]
        est[w] = a["at_theta_star"]["own"]["fe_pp"] if _measurable(a, "at_theta_star") else None
    ded = arms["w10240"].get("deduction_theta_star_equals_0.70", {}).get("verdict")
    out["CT28-P2"] = {"verdict": score_p2(est), "estimates_own_pp": est,
                      "w10240_deduction": ded}
    sf_iv = {}
    for a in ("k64",) + WIDTHS:
        s = sffits.get(a)
        sf_iv[a] = (s["slope"]["own"]["fixed_effect_ci95"] if s is not None else None)
    out["CT28-P3"] = {"verdict": score_p3(sf_iv), "intervals": sf_iv}
    sv = k64.get("scale_verification", {}).get("verdict")
    withheld = sv == "DIFFERS" or sv is None
    p4 = {"scale_verification": sv, "filed": "NOT_SEALED (measured, not tallied)"}
    if not withheld and _measurable(k64, "at_0.70") and _measurable(k32, "at_0.70"):
        a, b = k64["at_0.70"]["shared"]["fe_ci95_pp"], k32["at_0.70"]["shared"]["fe_ci95_pp"]
        p4.update({"k64_shared_fe_ci95_pp": a, "k32_shared_fe_ci95_pp": b,
                   "share_no_point": not intervals_overlap(a, b)})
    else:
        p4["status"] = "WITHHELD"
    out["CT28-P4"] = p4
    r = {}
    for k in ("rule1", "rule2", "rule3"):
        f = rules["rules"][k].get("at_theta_star", {})
        r[k] = (f["shared"]["fe_pp"] if "shared" in f and f.get("n_anchor_fits_failed", 1) == 0
                else None)
    r4 = k64.get("rule4_activation")
    r["rule4"] = (r4["shared"]["fe_pp"] if r4 is not None and r4.get("n_anchor_fits_failed") == 0
                  and r4["control_vs_ct24_r3"]["verdict"] == "BIT_IDENTICAL" else None)
    diff = None
    if not withheld and _measurable(k64, "at_0.70") and _measurable(k32, "at_0.70"):
        diff = k64["at_0.70"]["shared"]["fe_pp"] - k32["at_0.70"]["shared"]["fe_pp"]
    out["CT28-P5"] = {**score_p5(r, diff), "rules_shared_fe_pp": r}
    return out


def cmd_score(args) -> None:
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
    arms = {a: load(args.ct_dir / f"ct28_arm_{a}_ct.json") for a in ("k64", "k32n19") + WIDTHS}
    sff = {}
    for a in ("k64",) + WIDTHS:
        p = args.ct_dir / f"ct28_sffit_{a}_ct.json"
        sff[a] = load(p) if p.exists() else None
    rules = load(args.ct_dir / "ct28_rules_k64_ct.json")
    out = {"identity": {"measurement": "ct28_score", "governing_amendment": "CT28-F item 7",
                        "ct_dir": str(args.ct_dir)},
           **score_all(arms, sff, rules)}
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v.get("verdict", v.get("status")) for k, v in out.items()
                      if k.startswith("CT28")}), flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _join_args(p) -> None:
    p.add_argument("--arm-name", required=True)
    p.add_argument("--sim", type=Path, required=True)
    p.add_argument("--rf-counts-all-theta", type=Path, required=True)
    p.add_argument("--m1-json", type=Path, required=True)
    p.add_argument("--gtilde", type=Path, required=True)
    p.add_argument("--null-reductions", type=Path, required=True)
    p.add_argument("--d-fs", type=Path, required=True)
    p.add_argument("--degenerate-flag", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--use", nargs="+", default=None,
                   help="the comparison set's seed labels (default: every column)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("arm"); _join_args(a)
    a.add_argument("--target", type=float, required=True)
    a.add_argument("--stored-slope", type=Path, default=None)
    a.add_argument("--reference-fe", type=float, default=None)
    a.add_argument("--assert-theta-star-is-070", action="store_true")
    a.add_argument("--verify-scale", action="store_true")
    a.add_argument("--act-counts", type=Path, default=None)
    a.add_argument("--act-reference-fe", type=float, default=None)
    s = sub.add_parser("sf")
    s.add_argument("--arm-name", required=True)
    s.add_argument("--ckpts", nargs="+", type=Path, required=True)
    s.add_argument("--sim", type=Path, required=True)
    s.add_argument("--device", default="cuda")
    s.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("sffit"); _join_args(f)
    f.add_argument("--oneway-h5", type=Path, required=True)
    r = sub.add_parser("rules"); _join_args(r)
    r.add_argument("--target", type=float, required=True)
    r.add_argument("--rules-sim-h5", type=Path, required=True)
    r.add_argument("--rules-h5", type=Path, required=True)
    r.add_argument("--rf-counts-070", type=Path, required=True)
    w = sub.add_parser("widthrules"); _join_args(w)
    w.add_argument("--rules-h5", type=Path, required=True)
    w.add_argument("--rf-counts-070", type=Path, required=True)
    c = sub.add_parser("score")
    c.add_argument("--ct-dir", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    {"arm": cmd_arm, "sf": cmd_sf, "sffit": cmd_sffit, "rules": cmd_rules,
     "widthrules": cmd_widthrules, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
