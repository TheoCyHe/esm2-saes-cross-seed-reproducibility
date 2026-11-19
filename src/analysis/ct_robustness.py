"""Amendment CT's CPU batch: §CT15, §CT16, §CT17, §CT20 (the brief's CT12) and
the data of §CT21 (the brief's CT13). PREREGISTRATION §CT-pre2 and the sections
themselves declare every choice made here; the docstrings name the clause.

Every slope is design 3's primary arm (`primary_hard_bin`, beta-binomial fixed
effect, percentage points of `r_f` per SD of `log M`) through §CT3's loader and
fitter (`jackknife_dictionaries.load_join_inputs` / `_fit_one`), so the inputs,
the seed alignment and the identity gate are the jackknife's. Only the match
COUNTS change between arms, and each arm's counts are built here from a stored
per-comparator similarity array or a stored count array.

THE COMPARATOR SET IS NAMED BY SEED LABEL, NEVER BY POSITION (gotcha 29). The
similarity arrays are (anchor, column, feature); an anchor's own column is
found by matching its label against the column labels, so a subset whose
columns do not start at the anchors (§CT15's seeds 20-39) never excludes the
wrong column.

Subcommands:
  ct15      r_f against the 20 non-anchor dictionaries (n_compare = 20), refit.
  ct20      rules 1-4 at theta = 0.70, n_compare = 19; rule 4 at its
            rate-matched threshold; jackknife over the 20 anchors for rules 1
            and 4 (the rules whose per-comparator outcomes are stored).
  ct17      per-firing-bin slopes, rule 1 at 0.70 and rule 4 at theta*.
  ct16-sf   GPU: s_f from the decoder matrices, gated against §CT3's mnn_sim.
  ct16-fit  CPU: s_f's distribution, stability and Gaussian design-3 slope.
  ct21      assemble the figure's data from §CT3's and §CT4's outputs.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import h5py
import numpy as np

Z95 = 1.959964


# ---------------------------------------------------------------------------
# Pure functions -- tests/test_ct_robustness.py exercises each
# ---------------------------------------------------------------------------

def own_column_mask(anchor_labels: list, col_labels: list,
                    use_labels: list) -> np.ndarray:
    """(n_anchors, len(use_labels)) True where the column IS the anchor's own
    dictionary. By label: an anchor never compares against itself."""
    a = [str(x) for x in anchor_labels]
    u = [str(x) for x in use_labels]
    missing = [x for x in u if x not in [str(c) for c in col_labels]]
    if missing:
        raise ValueError(f"columns {missing} are not in the similarity array")
    return np.array([[ui == ai for ui in u] for ai in a], dtype=bool)


def subset_counts(sim: np.ndarray, theta: float, anchor_labels: list,
                  col_labels: list, use_labels: list) -> tuple[np.ndarray, int]:
    """Match counts at `theta` over the columns named `use_labels`, an anchor's
    own column excluded. Returns `(counts, n_trials)`; every anchor must be
    left with the same number of comparators, or the binomial denominator is
    not one number and the function refuses."""
    own = own_column_mask(anchor_labels, col_labels, use_labels)
    cl = [str(c) for c in col_labels]
    cols = [cl.index(str(u)) for u in use_labels]
    n_trials = (~own).sum(axis=1)
    if not np.all(n_trials == n_trials[0]):
        raise ValueError(f"anchors have different comparator counts {n_trials}")
    hit = (sim[:, cols, :] >= theta) & (~own)[:, :, None]
    return hit.sum(axis=1).astype(np.int16), int(n_trials[0])


def subset_rate(sim, theta, anchor_labels, col_labels, use_labels) -> float:
    y, n = subset_counts(sim, theta, anchor_labels, col_labels, use_labels)
    return float(y.mean() / n)


def theta_star_subset(sim, anchor_labels, col_labels, use_labels,
                      target: float) -> dict:
    """The LARGEST observed similarity whose mean rate over the named columns
    is >= target -- §CT4's theta* rule, on a column subset."""
    own = own_column_mask(anchor_labels, col_labels, use_labels)
    cl = [str(c) for c in col_labels]
    cols = [cl.index(str(u)) for u in use_labels]
    vals = sim[:, cols, :][~own]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"theta_star": None, "status": "NO_FINITE_SIMILARITY"}
    cand = np.unique(vals)
    rate = lambda t: subset_rate(sim, float(t), anchor_labels, col_labels, use_labels)
    if rate(cand[0]) < target:
        return {"theta_star": None, "status": "TARGET_UNREACHABLE",
                "max_rate": rate(cand[0])}
    i, j = 0, cand.size - 1
    while i < j:
        m = (i + j + 1) // 2
        if rate(cand[m]) >= target:
            i = m
        else:
            j = m - 1
    t = float(cand[i])
    nxt = float(cand[i + 1]) if i + 1 < cand.size else None
    return {"theta_star": t, "rate_at_theta_star": rate(t),
            "rate_just_above": rate(nxt) if nxt is not None else None,
            "target": target, "status": "OK"}


def intervals_overlap(a, b) -> bool:
    """Closed intervals [lo, hi] share a point (§CT-pre2 item 15)."""
    return bool(a[0] <= b[1] and b[0] <= a[1])


def widest(intervals: dict) -> tuple[str, list]:
    """The widest of several named intervals, and its value."""
    k = max(intervals, key=lambda n: intervals[n][1] - intervals[n][0])
    return k, intervals[k]


def strictly_monotone(values: list) -> str:
    """§CT17's reading: INCREASING / DECREASING (strict, in bin order) / NEITHER."""
    v = np.asarray(values, dtype=float)
    if v.size < 2 or not np.all(np.isfinite(v)):
        return "NEITHER"
    d = np.diff(v)
    if np.all(d > 0):
        return "INCREASING"
    if np.all(d < 0):
        return "DECREASING"
    return "NEITHER"


def bin_of(fire_rate: np.ndarray, edges: list) -> np.ndarray:
    """Half-open [lo, hi) bin index per feature on its OWN seed's firing rate
    (never across seeds by index, gotcha 18)."""
    e = np.asarray(edges, dtype=float)
    return np.clip(np.digitize(fire_rate, e[1:-1], right=False), 0, len(e) - 2)


def gaussian_penalised_fit(y: np.ndarray, X: np.ndarray, S: np.ndarray,
                           lam: float) -> dict:
    """Penalised least squares `(X'X + lam S) b = X'y`, the Bayesian covariance
    `sigma^2 (X'X + lam S)^-1` (mgcv's Vp) with `sigma^2 = RSS / (n - edf)`.
    §CT16's identity-link analogue of the design-3 fit, lambda inherited."""
    XtX = X.T @ X
    A = XtX + lam * S + 1e-10 * np.eye(X.shape[1])
    Ainv = np.linalg.inv(A)
    b = Ainv @ (X.T @ y)
    r = y - X @ b
    edf = float(np.trace(Ainv @ XtX))
    sigma2 = float(r @ r) / max(y.size - edf, 1.0)
    return {"beta": b, "cov": sigma2 * Ainv, "edf": edf, "sigma2": sigma2}


def oneway_and_mnn_anchor_rows(decoders: list, n_anchors: int):
    """`jackknife_dictionaries.mnn_sim_anchor_rows`, same pairs and order, also
    persisting each anchor feature's ONE-WAY maximum cosine to every column's
    dictionary (`pair_statistics`' `*_oneway`). Returns `(oneway, mnn)`, each
    (n_anchors, n_seeds, dict_size) float32, NaN / -inf on the diagonal."""
    from src.analysis.rf_marginal import pair_statistics
    n_seeds = len(decoders)
    d = decoders[0].shape[1]
    one = np.full((n_anchors, n_seeds, d), np.nan, dtype=np.float32)
    mnn = np.full((n_anchors, n_seeds, d), -np.inf, dtype=np.float32)
    for a in range(n_seeds):
        for b in range(a + 1, n_seeds):
            if a >= n_anchors and b >= n_anchors:
                continue
            st = pair_statistics(decoders[a], decoders[b])
            if a < n_anchors:
                one[a, b] = st["a_oneway"].float().cpu().numpy()
                mnn[a, b] = st["a_mnn"].float().cpu().numpy()
            if b < n_anchors:
                one[b, a] = st["b_oneway"].float().cpu().numpy()
                mnn[b, a] = st["b_mnn"].float().cpu().numpy()
    return one, mnn


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load(args) -> dict:
    """§CT3's loader (identity gate, seed alignment, fit set), with the decoder
    match file's column labels, and §CT4's activation sim aligned to it."""
    from src.analysis import jackknife_dictionaries as jk
    inp = jk.load_join_inputs(args)
    jk._W.update(inp)
    if getattr(args, "act_counts", None) is not None:
        with h5py.File(args.act_counts, "r") as hf:
            act = hf["mnn_sim_anchor_rows"][:]
            a_lab = [str(s) for s in json.loads(hf.attrs["seed_labels"])]
            ts = json.loads(hf.attrs["theta_star"])
        a_cols = [str(s) for s in json.loads(
            args.act_json.read_text(encoding="utf-8"))["identity"]["column_seed_labels"]]
        if a_lab != inp["anchor_labels"]:
            raise SystemExit(f"FATAL: activation anchors {a_lab} != {inp['anchor_labels']}")
        if a_cols != [str(c) for c in inp["column_labels"]]:
            raise SystemExit("FATAL: activation column order differs from the decoder "
                             "match file's")
        if act.shape != inp["sim"].shape:
            raise SystemExit("FATAL: activation sim shape differs")
        inp["act_sim"] = act
        inp["theta_star_39"] = ts
    return inp


def _ref_fe_at(reference_join: Path, theta: float) -> float:
    ref = json.loads(reference_join.read_text(encoding="utf-8"))
    for r in ref["theta_curve"]["rows"]:
        if round(float(r["theta"]), 2) == round(theta, 2):
            return float(r["marginal_effect_percentage_points_betabinomial"]["fixed_effect"])
    raise SystemExit(f"FATAL: theta {theta} not in the reference join")


def _identity(args, extra: dict) -> dict:
    from src.analysis.jackknife_dictionaries import _sha16
    from src.sae.train import git_dirty, git_sha
    ins = {}
    for k in ("sim", "rf_counts_all_theta", "m1_json", "gtilde", "null_reductions",
              "d_fs", "reference_join", "act_counts", "act_json", "rules_h5",
              "oneway_h5", "jackknife_json", "curve_json", "act_half_counts",
              "act_half_json", "gtilde_half_a", "gtilde_half_b", "rules_sim_h5"):
        p = getattr(args, k, None)
        if p is not None:
            ins[k] = {"path": str(p), "sha256_16": _sha16(p)}
    return {"governing_amendment": "CT", "arm": "primary_hard_bin",
            "likelihood": "betabinomial", "inputs": ins,
            "units": "percentage points of r_f per SD of log M",
            "host": platform.node(), "git_sha": git_sha(),
            "git_dirty": git_dirty(), **extra}


def _fit(y, n_trials) -> dict:
    from src.analysis import jackknife_dictionaries as jk
    return jk._fit_one(y, np.arange(y.shape[0]), n_trials)


def _intervals(f: dict, jk_ci=None) -> dict:
    iv = {"fixed_effect": f["fe_ci95_pp"], "across_seed": f["across_seed_ci95_pp"]}
    if jk_ci is not None:
        iv["jackknife"] = jk_ci
    k, w = widest(iv)
    return {"intervals_95_pp": iv, "widest": k, "widest_interval": w}


def _control_full_refit(inp, args) -> dict:
    """The full ensemble at theta = 0.70, n_compare = 39, must reproduce the
    join of record (§CT11.1's control, reused) before any arm is fitted."""
    from src.analysis import jackknife_dictionaries as jk
    f = _fit(jk.counts_from_sim(inp["sim"], 0.70), inp["sim"].shape[1] - 1)
    ctrl = jk.control_verdict([f["fe_pp"]], [_ref_fe_at(args.reference_join, 0.70)])
    if ctrl["verdict"] == "STOP":
        raise SystemExit(f"CONTROL FAILED: {ctrl}")
    return {"control_full_refit_0.70_n39": ctrl, "primary_fit_0.70_n39": f}


# ---------------------------------------------------------------------------
# §CT15
# ---------------------------------------------------------------------------

def cmd_ct15(args) -> None:
    inp = _load(args)
    out = {"identity": _identity(args, {"measurement": "ct15_disjoint_comparators"})}
    out.update(_control_full_refit(inp, args))
    anchors = inp["anchor_labels"]
    cols = [str(c) for c in inp["column_labels"]]
    use = [c for c in cols if c not in anchors]
    if len(use) != 20:
        raise SystemExit(f"FATAL: {len(use)} non-anchor columns, expected 20")
    y, n = subset_counts(inp["sim"], 0.70, anchors, cols, use)
    f = _fit(y, n)
    prim = json.loads(args.jackknife_json.read_text(encoding="utf-8"))["per_theta"]["0.70"]
    p_iv = prim["intervals_95_pp"]
    pk, pw = widest(p_iv)
    mine = _intervals(f)
    out.update({
        "comparator_seed_labels": use, "n_compare": n,
        "mean_r_f": float(y.mean() / n),
        "fit_0.70_n20": f, **mine,
        "primary_n39_intervals_95_pp": p_iv, "primary_widest": pk,
        "primary_widest_interval": pw,
        "CT15_P1_widest_intervals_overlap": intervals_overlap(mine["widest_interval"], pw),
        "overlap_same_combination": {
            k: intervals_overlap(mine["intervals_95_pp"][k], p_iv[k])
            for k in ("fixed_effect", "across_seed")}})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_compare", "mean_r_f",
                                          "CT15_P1_widest_intervals_overlap")}),
          f"FE {f['fe_pp']:+.4f} {f['fe_ci95_pp']}", flush=True)


# ---------------------------------------------------------------------------
# §CT20 (the brief's CT12)
# ---------------------------------------------------------------------------

RULE_DATASETS = {"rule1": "rule1_mnn_decoder", "rule2": "rule2_hungarian_decoder",
                 "rule3": "rule3_hungarian_enc_and_dec"}
RULE_NAMES = {"rule1": "rule 1 -- MNN, decoder cosine",
              "rule2": "rule 2 -- Hungarian one-to-one, decoder",
              "rule3": "rule 3 -- Hungarian, encoder AND decoder agree",
              "rule4": "rule 4 -- activation correlation (CT4), MNN"}
RULE1_RATE_N19 = 0.495814      # NoR 18.1, mean k/19 for rule 1 at 0.70


def _anchor_jackknife(sim, theta, anchors, cols) -> dict:
    """Leave-one-ANCHOR-dictionary-out over the 20-anchor comparison set: the
    removed anchor leaves as anchor and as comparator (n_compare 19 -> 18)."""
    from src.analysis import jackknife_dictionaries as jk
    loo, failed = [], 0
    W = jk._W
    for d in anchors:
        keep = np.array([a != d for a in anchors])
        use = [a for a in anchors if a != d]
        y, n = subset_counts(sim[keep], theta, [a for a in anchors if a != d],
                             cols, use)
        r = jk._fit_one(y, np.flatnonzero(keep), n)
        failed += int(r["n_anchor_fits_failed"] > 0)
        loo.append(r["fe_pp"])
    return {"loo_fe_pp": loo, "n_loo_fits_failed": failed}


def cmd_ct20(args) -> None:
    from src.analysis import jackknife_dictionaries as jk
    from src.analysis.seed_axis import align_to_reference
    inp = _load(args)
    anchors = inp["anchor_labels"]
    cols = [str(c) for c in inp["column_labels"]]
    out = {"identity": _identity(args, {"measurement": "ct20_rules_n19",
                                        "n_compare": 19, "rule_names": RULE_NAMES})}
    out.update(_control_full_refit(inp, args))

    # rules 1-3 at 0.70 from the CPU count file, aligned BY LABEL
    with h5py.File(args.rules_h5, "r") as hf:
        labs = [str(s) for s in json.loads(hf.attrs["seed_labels"])]
        if int(hf.attrs["n_compare_seeds"]) != 19 or abs(float(hf.attrs["theta"]) - 0.7) > 1e-12:
            raise SystemExit("FATAL: the rule file is not n_compare = 19 at 0.70")
        rules = {k: align_to_reference(labs, hf[v][:], anchors)
                 for k, v in RULE_DATASETS.items()}
        rules_device = str(hf.attrs.get("device_class"))
    y1_gpu, n = subset_counts(inp["sim"], 0.70, anchors, cols, anchors)
    y4, _ = subset_counts(inp["act_sim"], 0.70, anchors, cols, anchors)
    cpu_vs_gpu = int((rules["rule1"] != y1_gpu).sum())
    out["rule1_cpu_vs_gpu_differing_cells"] = cpu_vs_gpu
    out["rules_h5_device_class"] = rules_device

    fits = {}
    for k in ("rule1", "rule2", "rule3"):
        fits[k] = {"source": "rule_interior_mass_co.h5 (CPU)",
                   "theta": 0.70, "mean_r_f": float(rules[k].mean() / 19),
                   "fit": _fit(rules[k], 19)}
    fits["rule1_gpu"] = {"source": "CT3 mnn_sim, anchor columns (GPU)", "theta": 0.70,
                         "mean_r_f": float(y1_gpu.mean() / n), "fit": _fit(y1_gpu, n)}
    fits["rule4"] = {"source": "CT4 mnn_sim, anchor columns (GPU)", "theta": 0.70,
                     "mean_r_f": float(y4.mean() / n), "fit": _fit(y4, n)}

    # rule 4 at its rate-matched threshold
    ts4 = theta_star_subset(inp["act_sim"], anchors, cols, anchors, RULE1_RATE_N19)
    if ts4.get("theta_star") is not None:
        y4s, _ = subset_counts(inp["act_sim"], ts4["theta_star"], anchors, cols, anchors)
        fits["rule4_rate_matched"] = {"source": "CT4 mnn_sim, anchor columns (GPU)",
                                      "theta": ts4["theta_star"], "theta_star": ts4,
                                      "mean_r_f": float(y4s.mean() / n),
                                      "fit": _fit(y4s, n)}

    # jackknife for the rules with per-comparator outcomes
    jkres = {}
    for k, sim, th in (("rule1_gpu", inp["sim"], 0.70), ("rule4", inp["act_sim"], 0.70)) + (
            (("rule4_rate_matched", inp["act_sim"], ts4["theta_star"]),)
            if ts4.get("theta_star") is not None else ()):
        j = _anchor_jackknife(sim, th, anchors, cols)
        full = fits[k]["fit"]["fe_pp"]
        jr = jk.jackknife(j["loo_fe_pp"], full)
        jkres[k] = {**jr, "n_loo_fits_failed": j["n_loo_fits_failed"],
                    "ci95_pp": [full - Z95 * jr["se"], full + Z95 * jr["se"]],
                    "loo_fe_pp": j["loo_fe_pp"]}
    for k, v in fits.items():
        v.update(_intervals(v["fit"], jkres.get(k, {}).get("ci95_pp")))
        if k in jkres:
            v["jackknife"] = jkres[k]

    w = {k: fits[k]["widest_interval"] for k in fits}
    out.update({
        "fits": fits,
        "CT20_P1_rule3_overlaps_rule1_widest": intervals_overlap(w["rule3"], w["rule1"]),
        "CT20_P2_rule4_overlaps_rule1_widest": intervals_overlap(w["rule4"], w["rule1"]),
        "CT20_P2_rule4_overlaps_rule3_widest": intervals_overlap(w["rule4"], w["rule3"]),
        "scoring_note": ("CT20-P1 and CT20-P2 at theta = 0.70 on the rule-file fits "
                         "for rules 1-3 and rule 4 at numeric 0.70, each side's widest "
                         "95% interval (CT-pre2 item 15). Rules 2 and 3 at their "
                         "rate-matched thresholds need the matching re-run (CT20, "
                         "the preamble's item 1) and are not in this file.")})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, v in fits.items():
        print(f"[ct20] {k:20s} theta {v['theta']:.5f} rate {v['mean_r_f']:.6f} "
              f"FE {v['fit']['fe_pp']:+.4f} widest {v['widest']} {v['widest_interval']}",
              flush=True)
    print(json.dumps({k: out[k] for k in out if k.startswith("CT20")}), flush=True)


THETA_GRID = [round(0.30 + 0.05 * i, 2) for i in range(13)]


def _anchor_square(path: Path, ds: str, anchors: list) -> np.ndarray:
    """An (anchor, comparator, feature) array whose rows AND columns are the
    anchors in the file's label order, realigned to `anchors` by label."""
    from src.analysis.seed_axis import align_to_reference
    with h5py.File(path, "r") as hf:
        labs = [str(s) for s in json.loads(hf.attrs["anchor_seed_labels"])]
        s = hf[ds][:]
    s = align_to_reference(labs, s, anchors)
    return np.stack([align_to_reference(labs, s[i], anchors) for i in range(s.shape[0])])


def matched_set_overlap(sim_a, theta_a, sim_b, theta_b, anchors) -> dict:
    """The (anchor, comparator, feature) cells matched under each rule, the
    number in both, and their Jaccard. An anchor's own column never counts."""
    own = own_column_mask(anchors, anchors, anchors)[:, :, None]   # (anchor, comparator)
    ma = (sim_a >= theta_a) & ~own
    mb = (sim_b >= theta_b) & ~own
    both, either = int((ma & mb).sum()), int((ma | mb).sum())
    return {"n_matched_a": int(ma.sum()), "n_matched_b": int(mb.sum()),
            "n_both": both, "n_either": either,
            "jaccard": (both / either) if either else None}


def cmd_ct27(args) -> None:
    """§CT27(b): the raw-encoder rule 3 (the author's decision 1, 2026-10-23) at its
    rate-matched theta_e. Target: normalised rule 3's mean r_f at 0.70 from
    §CT20's grid file, re-derived from its similarity file and required equal.
    Scored on fixed-effect intervals (§CT23-pre item 9)."""
    from src.analysis import jackknife_dictionaries as jk
    inp = _load(args)
    anchors = inp["anchor_labels"]
    out = {"identity": _identity(args, {"measurement": "ct27_raw_encoder_rule3",
                                        "n_compare": 19})}
    out.update(_control_full_refit(inp, args))
    var = _anchor_square(args.variant_sim_h5, "sim_ct27_raw_encoder_dec070", anchors)
    r3 = _anchor_square(args.rules_sim_h5, "sim_rule3_hungarian_enc_and_dec", anchors)
    grid = json.loads(args.grid_json.read_text(encoding="utf-8"))
    g70 = grid["rules"]["rule3"]["grid"]["0.70"]
    target = float(g70["mean_r_f"])
    y3, n = subset_counts(r3, 0.70, anchors, anchors, anchors)
    if float(y3.mean() / n) != target:
        raise SystemExit(f"GATE FAILED: rule 3 at 0.70 re-derived {y3.mean() / n!r} "
                         f"!= the grid file's {target!r}")
    ref_fe = [float(x) for x in g70["fit"]["fe_ci95_pp"]]
    out["target_rule3_mean_r_f_0.70"] = target
    out["reference_rule3_fe_ci95_pp"] = ref_fe
    ts = theta_star_subset(var, anchors, anchors, anchors, target)
    out["theta_e"] = ts
    if ts.get("theta_star") is None:
        out["CT27_P1"] = "UNMEASURABLE"
        out["CT27_P1_cause"] = ts.get("status")
    else:
        th = ts["theta_star"]
        y, n = subset_counts(var, th, anchors, anchors, anchors)
        f = _fit(y, n)
        j = _anchor_jackknife(var, th, anchors, anchors)
        jr = jk.jackknife(j["loo_fe_pp"], f["fe_pp"])
        jk_ci = ([f["fe_pp"] - Z95 * jr["se"], f["fe_pp"] + Z95 * jr["se"]]
                 if j["n_loo_fits_failed"] <= 2 else None)
        res = {"theta": th, "mean_r_f": float(y.mean() / n), "fit": f,
               "jackknife": {**jr, "n_loo_fits_failed": j["n_loo_fits_failed"],
                             "ci95_pp": jk_ci}}
        res.update(_intervals(f, jk_ci))
        out["variant"] = res
        out["overlap_with_normalised_rule3_0.70"] = matched_set_overlap(
            var, th, r3, 0.70, anchors)
        if f.get("n_anchor_fits_failed", 0) > 0:
            out["CT27_P1"] = "UNMEASURABLE"
            out["CT27_P1_cause"] = "anchor fit failed"
        else:
            hit = intervals_overlap(f["fe_ci95_pp"], ref_fe)
            out["CT27_P1"] = "HIT" if hit else "MISS"
        print(f"[ct27] theta_e {th!r} rate {res['mean_r_f']:.6f} FE {f['fe_pp']:+.4f} "
              f"{f['fe_ci95_pp']} vs rule 3 {ref_fe}: {out['CT27_P1']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")


def cmd_ct20b(args) -> None:
    """§CT20's second stage (the preamble's item 1): rules 1-3 from their
    re-run with per-comparator similarities persisted (`rule_interior_mass
    --out-sim-h5`), gated EXACT against the CO counts of record at 0.70; then
    every rule on the 13-θ grid and at its rate-matched threshold, with the
    anchor jackknife for every rule (all four now have per-comparator
    outcomes). Rule 4 is §CT4's mnn_sim on the anchor columns."""
    from src.analysis import jackknife_dictionaries as jk
    from src.analysis.seed_axis import align_to_reference
    inp = _load(args)
    anchors = inp["anchor_labels"]
    cols = [str(c) for c in inp["column_labels"]]
    out = {"identity": _identity(args, {"measurement": "ct20_rules_grid_rate_matched",
                                        "n_compare": 19, "rule_names": RULE_NAMES})}
    out.update(_control_full_refit(inp, args))
    sims = {}
    with h5py.File(args.rules_sim_h5, "r") as hf:
        labs = [str(s) for s in json.loads(hf.attrs["anchor_seed_labels"])]
        for k, ds in RULE_DATASETS.items():
            s = hf[f"sim_{ds}"][:]
            # rows AND columns are anchors in the file's label order
            s = align_to_reference(labs, s, anchors)
            s = np.stack([align_to_reference(labs, s[i], anchors) for i in range(s.shape[0])])
            sims[k] = s
    with h5py.File(args.rules_h5, "r") as hf:
        rlabs = [str(x) for x in json.loads(hf.attrs["seed_labels"])]
        stored = {k: align_to_reference(rlabs, hf[v][:], anchors) for k, v in RULE_DATASETS.items()}
    gate = {}
    for k in RULE_DATASETS:
        y, n = subset_counts(sims[k], 0.70, anchors, anchors, anchors)
        gate[k] = int((y != stored[k]).sum())
    out["gate_vs_rule_interior_mass_co_cells_differing"] = gate
    if any(gate.values()):
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        raise SystemExit(f"GATE FAILED: {gate}")
    a_idx = [cols.index(a) for a in anchors]
    sims["rule4"] = inp["act_sim"][:, a_idx, :]
    out["rate_target_rule1_n19"] = RULE1_RATE_N19
    res = {}
    for k in ("rule1", "rule2", "rule3", "rule4"):
        s = sims[k]
        grid = {}
        for t in THETA_GRID:
            y, n = subset_counts(s, t, anchors, anchors, anchors)
            grid[f"{t:.2f}"] = {"mean_r_f": float(y.mean() / n), "fit": _fit(y, n)}
        ts = theta_star_subset(s, anchors, anchors, anchors, RULE1_RATE_N19)
        rm = None
        if ts.get("theta_star") is not None:
            y, n = subset_counts(s, ts["theta_star"], anchors, anchors, anchors)
            rm = {"theta_star": ts, "mean_r_f": float(y.mean() / n), "fit": _fit(y, n)}
        jks = {}
        for tag, th in (("0.70", 0.70),) + ((("rate_matched", ts["theta_star"]),)
                                           if rm is not None else ()):
            j = _anchor_jackknife(s, th, anchors, anchors)
            full = grid["0.70"]["fit"]["fe_pp"] if tag == "0.70" else rm["fit"]["fe_pp"]
            jr = jk.jackknife(j["loo_fe_pp"], full)
            jks[tag] = {**jr, "n_loo_fits_failed": j["n_loo_fits_failed"],
                        "ci95_pp": ([full - Z95 * jr["se"], full + Z95 * jr["se"]]
                                    if j["n_loo_fits_failed"] <= 2 else None)}
        g70 = grid["0.70"]
        g70.update(_intervals(g70["fit"], jks["0.70"]["ci95_pp"]))
        if rm is not None:
            rm.update(_intervals(rm["fit"], jks["rate_matched"]["ci95_pp"]))
        res[k] = {"grid": grid, "rate_matched": rm, "jackknife": jks}
        print(f"[ct20b] {k}: 0.70 FE {g70['fit']['fe_pp']:+.4f} widest {g70['widest']} "
              f"{g70['widest_interval']}; rate-matched θ "
              f"{ts.get('theta_star')}", flush=True)
    out["rules"] = res
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# §CT17
# ---------------------------------------------------------------------------

def _bin_fit(y, n_trials, bin_masks) -> dict:
    """Per-bin design-3 beta-binomial, z(log M) in the FULL fit set's SD.

    `fit_arm` standardises log M over the rows it is given. Within a bin the
    model `logit p = b0 + b z_bin + s(phi) + g d_fs` is an exact
    reparametrisation of the same model in `z_full` (z_full = z_bin *
    sd_bin/sd_full + const; the penalty touches only the spline), so the slope
    per full-set SD is the bin's per-bin-SD effect times sd_full/sd_bin. The
    per-anchor effects and SEs are rescaled BEFORE the combination."""
    from src.analysis import jackknife_dictionaries as jk
    from src.analysis.design3_fit import meta_analyse
    from src.analysis.design3_join import fit_arm
    W = jk._W
    keep, logM = W["keep"], W["logM"]
    res = []
    for b, m in enumerate(bin_masks):
        kb = keep & m
        if kb.sum(axis=1).min() < 1:
            res.append({"bin": b, "status": "EMPTY_IN_SOME_ANCHOR",
                        "n_features_per_anchor": kb.sum(axis=1).tolist()})
            continue
        try:
            arm = fit_arm(y.astype(np.float64), logM, W["log_phi"], W["d_fs"],
                          kb, n_trials, dispersion=True)
        except Exception as e:                     # a failed fit is counted
            res.append({"bin": b, "status": "FIT_RAISED", "error": repr(e)})
            continue
        ames, ses, failed = [], [], 0
        for s, r in enumerate(arm["per_seed"]):
            sd_full = float(logM[s][keep[s]].std())
            k = sd_full / max(r["sd_logM"], 1e-300)
            me = r["betabinomial"]["marginal_effect"]
            ames.append(me["ame"] * k)
            ses.append(me["se"] * k)
            failed += int((not r["converged"]) or
                          (not r["betabinomial"]["optimiser_success"]))
        meta = meta_analyse(np.array(ames), np.array(ses))
        res.append({"bin": b, "status": "OK",
                    "n_features_per_anchor": kb.sum(axis=1).tolist(),
                    "fe_pp": 100 * meta["fixed_effect"],
                    "fe_ci95_pp": [100 * x for x in meta["fixed_effect_ci95"]],
                    "across_seed_ci95_pp": [100 * x for x in meta["across_seed_ci95"]],
                    "n_anchor_fits_failed": failed})
    return res


def cmd_ct17(args) -> None:
    from src.analysis import jackknife_dictionaries as jk
    from src.analysis.design3_join import firing_rate_edges
    inp = _load(args)
    edges = firing_rate_edges(args.gtilde)
    with h5py.File(args.null_reductions, "r") as hf:
        fr = hf["fire_rate"][:]
        seeds = [s.decode() if isinstance(s, bytes) else str(s) for s in hf["seeds"][:]]
    from src.analysis.seed_axis import align_to_reference
    fr = align_to_reference([s.split("_")[-1] for s in seeds], fr, inp["anchor_labels"])
    fb = np.stack([bin_of(fr[s], edges) for s in range(fr.shape[0])])
    masks = [fb == b for b in range(len(edges) - 1)]
    out = {"identity": _identity(args, {"measurement": "ct17_slopes_by_firing_bin",
                                        "firing_rate_edges": edges,
                                        "edges_source": "gtilde firing_rate_edges"})}
    out.update(_control_full_refit(inp, args))
    y1 = jk.counts_from_sim(inp["sim"], 0.70)
    ts = inp["theta_star_39"]["theta_star"]
    y4 = jk.counts_from_sim(inp["act_sim"], ts)
    n = inp["sim"].shape[1] - 1
    out["rule1_theta_0.70"] = _bin_fit(y1, n, masks)
    out["rule4_theta_star"] = {"theta_star": ts, "bins": _bin_fit(y4, n, masks)}
    for key, rows in (("rule1_theta_0.70", out["rule1_theta_0.70"]),
                      ("rule4_theta_star", out["rule4_theta_star"]["bins"])):
        out.setdefault("monotone_in_firing_rate", {})[key] = strictly_monotone(
            [r.get("fe_pp", np.nan) for r in rows])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["monotone_in_firing_rate"]), flush=True)


# ---------------------------------------------------------------------------
# §CT16
# ---------------------------------------------------------------------------

def cmd_ct16_sf(args) -> None:
    import torch
    from src.analysis.artifact_identity import device_class
    from src.analysis.rf_marginal import check_unit_norm, load_decoder
    from src.sae.train import git_dirty, git_sha
    num = lambda p: int(str(p.name).rsplit("_", 1)[-1])
    anchors = sorted(args.ckpts, key=num)
    ckpts = anchors + sorted(args.compare_ckpts, key=num)
    if len(ckpts) != 40:
        raise SystemExit(f"FATAL: {len(ckpts)} checkpoints, expected 40")
    decs, labels = [], []
    for p in ckpts:
        W, meta = load_decoder(p, args.device)
        if check_unit_norm(W)["verdict"] != "PASS":
            raise SystemExit(f"FATAL: {p} decoder not unit-norm")
        decs.append(W)
        labels.append(int(meta["seed"]))
    one, mnn = oneway_and_mnn_anchor_rows(decs, len(anchors))
    with h5py.File(args.sim, "r") as hf:
        ref = hf["mnn_sim_anchor_rows"][:]
        ref_cols = [int(s) for s in json.loads(hf.attrs["column_seed_labels"])]
    if ref_cols != labels:
        raise SystemExit(f"FATAL: column labels {labels} != CT3's {ref_cols}")
    both_fin = np.isfinite(ref) & np.isfinite(mnn)
    n_diff = int(((np.isfinite(ref) != np.isfinite(mnn))
                  | (both_fin & (ref != mnn))).sum())
    gate = "EXACT" if n_diff == 0 else "DIFFERS"
    run_dev = str(device_class(decs[0]))
    with h5py.File(args.out, "w") as ho:
        ho.create_dataset("oneway_max_cos_anchor_rows", data=one, compression="gzip")
        ho.attrs.update({"measurement": "ct16_oneway_max_decoder_cosine",
                         "governing_amendment": "CT16",
                         "anchor_seed_labels": json.dumps(labels[:len(anchors)]),
                         "column_seed_labels": json.dumps(labels),
                         "gate_vs_ct3_mnn_sim": gate, "n_cells_differing": n_diff,
                         "gather_device_class": run_dev,
                         "git_sha": git_sha(), "git_dirty": git_dirty(),
                         "torch": str(torch.__version__), "host": platform.node()})
    print(f"[ct16-sf] gate vs CT3 mnn_sim: {gate} ({n_diff} cells differ) on {run_dev}",
          flush=True)
    if gate != "EXACT":
        raise SystemExit("GATE DIFFERS -- stop and report")


def cmd_ct16_fit(args) -> None:
    from src.analysis import jackknife_dictionaries as jk
    from src.analysis.design3_fit import build_design, fit_design3, meta_analyse
    inp = _load(args)
    W = jk._W
    with h5py.File(args.oneway_h5, "r") as hf:
        one = hf["oneway_max_cos_anchor_rows"][:]
        if hf.attrs["gate_vs_ct3_mnn_sim"] != "EXACT":
            raise SystemExit("FATAL: the s_f file's gate is not EXACT")
        a_lab = [str(s) for s in json.loads(hf.attrs["anchor_seed_labels"])]
    if a_lab != inp["anchor_labels"]:
        raise SystemExit("FATAL: anchor order differs")
    n_a, n_s, d = one.shape
    off = np.ones((n_a, n_s), dtype=bool)
    off[np.arange(n_a), np.arange(n_a)] = False
    sf = np.stack([one[a][off[a]].mean(axis=0) for a in range(n_a)])   # (20, d)
    if not np.all(np.isfinite(sf)):
        raise SystemExit("FATAL: non-finite s_f")
    keep = W["keep"]
    q = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    dist = {"pooled_over_cells": {
        "n": int(keep.sum()), "mean": float(sf[keep].mean()), "sd": float(sf[keep].std(ddof=1)),
        "quantiles": dict(zip([str(x) for x in q], np.percentile(sf[keep], q).tolist()))}}
    per_anchor_mean = [float(sf[a][keep[a]].mean()) for a in range(n_a)]
    per_anchor_sd = [float(sf[a][keep[a]].std(ddof=1)) for a in range(n_a)]
    stab = {"per_anchor_mean": per_anchor_mean, "per_anchor_sd": per_anchor_sd,
            "sd_across_anchors_of_mean": float(np.std(per_anchor_mean, ddof=1)),
            "mean_of_means": float(np.mean(per_anchor_mean)),
            "range_of_means": [min(per_anchor_mean), max(per_anchor_mean)]}
    y070 = jk.counts_from_sim(inp["sim"], 0.70)
    rows = []
    for a in range(n_a):
        m = keep[a]
        lm = W["logM"][a][m]
        z = (lm - lm.mean()) / lm.std()
        des = build_design(z, W["log_phi"][a][m], W["d_fs"][a][m])
        lam = fit_design3(y070[a][m].astype(np.float64), n_s - 1, des["X"], des["S"])["lambda"]
        g = gaussian_penalised_fit(sf[a][m].astype(np.float64), des["X"], des["S"], lam)
        i = des["idx_logM"]
        rows.append({"anchor": a, "lambda_inherited": lam, "edf": g["edf"],
                     "b_z": float(g["beta"][i]), "se_b_z": float(np.sqrt(g["cov"][i, i]))})
    meta = meta_analyse(np.array([r["b_z"] for r in rows]),
                        np.array([r["se_b_z"] for r in rows]))
    out = {"identity": _identity(args, {"measurement": "ct16_threshold_independent_summary",
                                        "units_slope": "cosine per SD of log M",
                                        "likelihood": "gaussian_identity_lambda_inherited"}),
           "s_f_definition": "mean over the 39 comparator dictionaries of the maximum "
                             "signed decoder cosine to any feature in that dictionary",
           "distribution": dist, "across_seed_stability": stab,
           "slope": {"fixed_effect": meta["fixed_effect"],
                     "fixed_effect_ci95": meta["fixed_effect_ci95"],
                     "across_seed_mean": meta["across_seed_mean"],
                     "across_seed_ci95": meta["across_seed_ci95"]},
           "per_anchor": rows}
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["slope"]), json.dumps(dist), flush=True)


# ---------------------------------------------------------------------------
# §CT10 -- the three slopes on disjoint rows
# ---------------------------------------------------------------------------

def _jackknife_arm(sim, theta, logM, workers) -> dict:
    """§CT3's leave-one-dictionary-out over the 40 dictionaries for one arm:
    the arm's similarity array and `log M` swapped into the worker globals,
    everything else the primary's."""
    from concurrent.futures import ProcessPoolExecutor
    from src.analysis import jackknife_dictionaries as jk
    jk._W.update({"sim": sim, "logM": logM})
    n_seeds = sim.shape[1]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(jk._loo_task, [(d, theta) for d in range(n_seeds)], chunksize=1))
    return {"loo_fe_pp": [r["fe_pp"] for r in res],
            "n_loo_fits_failed": sum(r["n_anchor_fits_failed"] > 0 for r in res)}


def cmd_ct10_fit(args) -> None:
    from src.analysis import jackknife_dictionaries as jk
    from src.analysis.stratum_slope import aligned_logM
    inp = _load(args)
    anchors = inp["anchor_labels"]
    out = {"identity": _identity(args, {"measurement": "ct10_disjoint_rows",
                                        "governing_amendment": "CT10"})}
    out.update(_control_full_refit(inp, args))
    with h5py.File(args.act_half_counts, "r") as hf:
        act = hf["mnn_sim_anchor_rows"][:]
        a_lab = [str(s) for s in json.loads(hf.attrs["seed_labels"])]
        ts = json.loads(hf.attrs["theta_star"])
    a_cols = [str(s) for s in json.loads(
        args.act_half_json.read_text(encoding="utf-8"))["identity"]["column_seed_labels"]]
    if a_lab != anchors or a_cols != [str(c) for c in inp["column_labels"]] \
            or act.shape != inp["sim"].shape:
        raise SystemExit("FATAL: the half-A activation array's axes differ from the decoder file's")
    if ts.get("theta_star") is None:
        raise SystemExit(f"UNMEASURABLE: theta*_A not found: {ts}")
    tA = float(ts["theta_star"])
    logM = {}
    for half, path in (("B", args.gtilde_half_b), ("A", args.gtilde_half_a)):
        lm, rep = aligned_logM(path, anchors)        # refuses non-positive M (cause 2)
        logM[half] = lm
        out[f"half_{half}_gtilde"] = rep
    full_logM = inp["logM"]
    arms = {"dec_B": (inp["sim"], 0.70, "B"), "act_A_B": (act, tA, "B"),
            "act_A_A": (act, tA, "A")}
    n_trials = inp["sim"].shape[1] - 1
    out["theta_star_A"] = ts
    out["arms"] = {}
    for name, (sim, th, half) in arms.items():
        jk._W.update({"sim": sim, "logM": logM[half]})
        f = _fit(jk.counts_from_sim(sim, th), n_trials)
        j = _jackknife_arm(sim, th, logM[half], args.workers)
        jr = jk.jackknife(j["loo_fe_pp"], f["fe_pp"])
        jci = [f["fe_pp"] - Z95 * jr["se"], f["fe_pp"] + Z95 * jr["se"]] \
            if j["n_loo_fits_failed"] <= 2 else None
        out["arms"][name] = {"theta": th, "alignment_half": half, "fit": f,
                             "jackknife": {**jr, **j}, **_intervals(f, jci)}
        print(f"[ct10] {name}: FE {f['fe_pp']:+.4f} {f['fe_ci95_pp']} widest "
              f"{out['arms'][name]['widest']} {out['arms'][name]['widest_interval']}",
              flush=True)
    jk._W.update({"sim": inp["sim"], "logM": full_logM})
    w = {k: v["widest_interval"] for k, v in out["arms"].items()}
    failed = {k: v["fit"]["n_anchor_fits_failed"] for k, v in out["arms"].items()}
    out["CT10_P1_actAB_does_not_overlap_decB"] = not intervals_overlap(w["act_A_B"], w["dec_B"])
    out["CT10_P2_actAB_fe_ge_1pp"] = bool(out["arms"]["act_A_B"]["fit"]["fe_pp"] >= 1.0)
    out["anchor_fit_failures"] = failed
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if k.startswith("CT10")}), flush=True)


# ---------------------------------------------------------------------------
# §CT21 -- data only
# ---------------------------------------------------------------------------

def cmd_ct21(args) -> None:
    jk_ = json.loads(args.jackknife_json.read_text(encoding="utf-8"))["per_theta"]
    cv = json.loads(args.curve_json.read_text(encoding="utf-8"))["rows"]
    rows = []
    for r in cv:
        t = f"{float(r['theta']):.2f}"
        dj = jk_[t]
        if abs(dj["full_fe_pp"] - r["decoder"]["fe_pp"]) > 1e-9:
            raise SystemExit(f"FATAL: the two files disagree on the decoder FE at {t}")
        rows.append({"theta": float(t),
                     "decoder_fe_pp": dj["full_fe_pp"],
                     "decoder_intervals_95_pp": dj["intervals_95_pp"],
                     "activation_fe_pp": r["activation"]["fe_pp"],
                     "activation_intervals_95_pp": {
                         "fixed_effect": r["activation"]["fe_ci95_pp"],
                         "across_seed": r["activation"]["across_seed_ci95_pp"]}})
    out = {"identity": _identity(args, {"measurement": "ct21_figure_data"}),
           "reference_line_pp_per_sd": 1.0, "rows": rows}
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[ct21] {len(rows)} theta rows", flush=True)


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def join_args(q, act=False):
        for k in ("--sim", "--rf-counts-all-theta", "--m1-json", "--gtilde",
                  "--null-reductions", "--d-fs", "--reference-join"):
            q.add_argument(k, type=Path, required=True)
        q.add_argument("--degenerate-flag", type=Path, default=None)
        q.add_argument("--config", type=Path, default=None)
        q.add_argument("--out", type=Path, required=True)
        if act:
            q.add_argument("--act-counts", type=Path, required=True)
            q.add_argument("--act-json", type=Path, required=True)

    a = sub.add_parser("ct15"); join_args(a)
    a.add_argument("--jackknife-json", type=Path, required=True)
    b = sub.add_parser("ct20"); join_args(b, act=True)
    b.add_argument("--rules-h5", type=Path, required=True)
    b2 = sub.add_parser("ct20b"); join_args(b2, act=True)
    b2.add_argument("--rules-h5", type=Path, required=True)
    b2.add_argument("--rules-sim-h5", type=Path, required=True)
    c = sub.add_parser("ct17"); join_args(c, act=True)
    s = sub.add_parser("ct16-sf")
    s.add_argument("--ckpts", nargs="+", type=Path, required=True)
    s.add_argument("--compare-ckpts", nargs="+", type=Path, required=True)
    s.add_argument("--sim", type=Path, required=True)
    s.add_argument("--device", default="cuda")
    s.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("ct16-fit"); join_args(f)
    f.add_argument("--oneway-h5", type=Path, required=True)
    t = sub.add_parser("ct10-fit"); join_args(t)
    for k in ("--act-half-counts", "--act-half-json", "--gtilde-half-a", "--gtilde-half-b"):
        t.add_argument(k, type=Path, required=True)
    t.add_argument("--workers", type=int, default=8)
    k27 = sub.add_parser("ct27"); join_args(k27)
    k27.add_argument("--variant-sim-h5", type=Path, required=True)
    k27.add_argument("--rules-sim-h5", type=Path, required=True)
    k27.add_argument("--grid-json", type=Path, required=True)
    g = sub.add_parser("ct21")
    g.add_argument("--jackknife-json", type=Path, required=True)
    g.add_argument("--curve-json", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    return p


def resolve_config(args) -> None:
    """`load_join_inputs` asserts the A4a floor against the config, so a
    missing `--config` takes the repository default -- `stratum_slope.main`'s
    rule (found when §CT15/§CT17/§CT20/§CT16-fit refused on a None path,
    jobs 30342488-90 and 30342493)."""
    if hasattr(args, "config") and args.config is None:
        from src.experiment_config import DEFAULT_CONFIG_PATH
        args.config = DEFAULT_CONFIG_PATH


def main() -> None:
    args = build_parser().parse_args()
    resolve_config(args)
    {"ct10-fit": cmd_ct10_fit, "ct15": cmd_ct15, "ct20": cmd_ct20, "ct20b": cmd_ct20b, "ct17": cmd_ct17, "ct16-sf": cmd_ct16_sf,
     "ct16-fit": cmd_ct16_fit, "ct21": cmd_ct21, "ct27": cmd_ct27}[args.cmd](args)


if __name__ == "__main__":
    main()
