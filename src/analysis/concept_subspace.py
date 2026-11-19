"""PREREGISTRATION amendment CT, §CT22 -- concept-subspace reproducibility.

THE QUESTION (reviewer ask 14). Feature-level matching asks whether ONE
feature's decoder direction is present in another dictionary. This asks whether
the small subspace a concept occupies in one seed -- the span of its top-m
features by alignment -- is the same subspace in another seed, judged against a
firing-rate-matched null.

THE OBJECTS (§CT22.1, §CT22-F):
  * a CELL is (concept c, unordered anchor pair s < s' in numeric labels), for
    c detected in both seeds; c ranges over the 650 pool concepts detected in
    >= 2 anchor seeds (the author, 2026-10-23); cells are indexed 0..N-1 in that
    enumeration, and the cell's null draws use default_rng(20261024 + index);
  * F(c, s) = the top m features of seed s by A(c, f) (the carrier analysis's AP,
    stable sort, so the first element IS `carriers_from_ap`'s carrier), or under
    the alternative, the top m by log M among s's features whose argmax is c;
  * Omega = ||U_s^T U_s'||_F^2 / m on QR bases of the unit-normalised decoder
    directions (float64); the largest principal angle and the Grassmann geodesic
    are computed from the same singular values;
  * the NULL: 100 draws, each replacing every feature of F by a feature of the
    same seed in the same firing stratum (`carrier_identity.firing_bin`,
    `carrier_support.cell_index` -- IMPORTED, the carrier analysis's rule), F
    itself excluded, distinct within a draw;
  * the feature-level match fraction: the share of F with `mnn_sim >= 0.70` in
    the other seed (§CT3's stored array), mean of the two directions.

Seed axes are joined by LABEL, never by position (gotcha 29): the AP artifact
is lexicographic, the decoders are ordered into it by their own meta seed, and
`mnn_sim`, log M and the argmax are aligned by their stored labels.

Modes: `pilot` (the 98 tuning concepts, every configuration of §CT22.3, the
lock computed by §CT22-F item 7's rule) and `confirm` (the 552 confirmatory
concepts under the configuration the pilot locked, refusing unless the
configuration passed equals the pilot artifact's lock).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

SEED_BASE = 20261024
N_NULL = 100
THETA_FEATURE = 0.70
RANK_TOL = 1e-6
TIE_TOL_SD = 0.1
M_CANDIDATES = (3, 5, 8)
MEASURES = ("omega", "theta_max", "geodesic")      # primary first
ALIGNMENTS = ("ap", "logm_argmax")                  # primary first
# Larger is more similar for Omega; smaller for the two distances (§CT22-F 6).
LARGER_IS_SIMILAR = {"omega": True, "theta_max": False, "geodesic": False}

_W: dict = {}          # read-only worker state, shared by fork


# ---------------------------------------------------------------------------
# Pure numerics -- every one is exercised by tests/test_concept_subspace.py
# ---------------------------------------------------------------------------

def unit_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def rank_deficient(X: np.ndarray) -> np.ndarray:
    """§CT22.6: smallest singular value of the (unit-row) matrix below
    RANK_TOL x the largest. Works on (m, d) or a stack (..., m, d)."""
    s = np.linalg.svd(unit_rows(X), compute_uv=False)
    return s[..., -1] < RANK_TOL * s[..., 0]


def basis(X: np.ndarray) -> np.ndarray:
    """Orthonormal basis (..., d, m) of the span of the unit rows of X."""
    q, _ = np.linalg.qr(np.swapaxes(unit_rows(X), -1, -2))
    return q


def principal_cosines(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Singular values of U^T V, clipped to [0, 1], descending."""
    s = np.linalg.svd(np.swapaxes(U, -1, -2) @ V, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def measures_from_cosines(sig: np.ndarray) -> dict:
    m = sig.shape[-1]
    ang = np.arccos(sig)
    return {"omega": (sig ** 2).sum(axis=-1) / m,
            "theta_max": ang[..., -1],
            "geodesic": np.sqrt((ang ** 2).sum(axis=-1))}


def subspace_measures(X: np.ndarray, Y: np.ndarray) -> dict:
    """All three measures for one pair of feature sets (or two stacks)."""
    return measures_from_cosines(principal_cosines(basis(X), basis(Y)))


def top_m(values: np.ndarray, m: int, eligible: np.ndarray | None = None
          ) -> np.ndarray | None:
    """Indices of the m largest `values`, ties toward the lower index (a stable
    sort on -values, the order `np.argmax` uses). With `eligible`, only those
    indices compete; None if fewer than m are eligible."""
    v = np.asarray(values, dtype=np.float64)
    idx = np.arange(v.size) if eligible is None else np.flatnonzero(eligible)
    if idx.size < m:
        return None
    order = np.argsort(-v[idx], kind="stable")
    return idx[order[:m]]


def tie_at_boundary(values: np.ndarray, F: np.ndarray,
                    eligible: np.ndarray | None = None) -> bool:
    """True when the m-th selected value equals the best unselected one."""
    v = np.asarray(values, dtype=np.float64)
    mask = np.ones(v.size, bool) if eligible is None else eligible.copy()
    mask[F] = False
    return bool(mask.any() and v[mask].max() == v[F].min())


def strata_pools(strat: np.ndarray, F: np.ndarray) -> list | None:
    """For each stratum F occupies (ascending), (positions in F, eligible pool):
    the seed's features in that stratum with ALL of F excluded (the carrier is
    excluded from its own baseline, §AY7). None if any stratum holds fewer
    eligible features than F has in it (infeasible, §CT22.4)."""
    out = []
    inF = np.zeros(strat.size, bool)
    inF[F] = True
    for b in np.unique(strat[F]):
        pos = np.flatnonzero(strat[F] == b)
        pool = np.flatnonzero((strat == b) & ~inF)
        if pool.size < pos.size:
            return None
        out.append((pos, pool))
    return out


def draw_side(rng: np.random.Generator, pools: list, m: int) -> np.ndarray:
    """One matched draw: F's features replaced stratum by stratum, distinct."""
    row = np.empty(m, dtype=np.int64)
    for pos, pool in pools:
        row[pos] = pool[rng.choice(pool.size, size=pos.size, replace=False)]
    return row


def matched_draws(seed: int, pools_s: list, pools_t: list, m: int,
                  n_draws: int = N_NULL) -> tuple[np.ndarray, np.ndarray]:
    """§CT22-F item 5's order: default_rng(seed); per draw, the s side then
    the s' side."""
    rng = np.random.default_rng(seed)
    ds = np.empty((n_draws, m), dtype=np.int64)
    dt = np.empty((n_draws, m), dtype=np.int64)
    for d in range(n_draws):
        ds[d] = draw_side(rng, pools_s, m)
        dt[d] = draw_side(rng, pools_t, m)
    return ds, dt


def null_summary(obs: float, null: np.ndarray, larger_is_similar: bool) -> dict:
    """Separation in null-SD units, exceedance and the 95th-percentile flag,
    oriented so that larger separation / exceedance means MORE similar than
    the null (§CT22-F item 6)."""
    null = np.asarray(null, dtype=np.float64)
    mu, sd = float(null.mean()), float(null.std(ddof=0))
    if larger_is_similar:
        exc = float((null < obs).mean())
        beyond = bool(obs > np.percentile(null, 95))
        sep = (obs - mu) / sd if sd > 0 else float("nan")
    else:
        exc = float((null > obs).mean())
        beyond = bool(obs < np.percentile(null, 5))
        sep = (mu - obs) / sd if sd > 0 else float("nan")
    return {"null_mean": mu, "null_sd": sd, "separation": sep,
            "exceedance": exc, "beyond_95": beyond}


def lock_configuration(scores: dict) -> dict:
    """§CT22.3's locking rule, made total by §CT22-F item 7.

    `scores` maps (m, measure, alignment) -> median separation (NaN allowed:
    a configuration with no measurable cell cannot be locked)."""
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not finite:
        raise SystemExit("FATAL: no configuration has a finite score")
    best = max(finite.values())
    tied = [k for k, v in finite.items() if v >= best - TIE_TOL_SD]
    steps = []
    for name, key in (("measure", lambda k: k[1] == MEASURES[0]),
                      ("alignment", lambda k: k[2] == ALIGNMENTS[0])):
        pref = [k for k in tied if key(k)]
        if pref and len(tied) > 1:
            steps.append(f"{name}: {len(tied)} -> {len(pref)}")
            tied = pref
    if len(tied) > 1:
        steps.append(f"smallest m: {len(tied)} -> 1")
    locked = min(tied, key=lambda k: k[0])
    return {"locked": {"m": locked[0], "measure": locked[1],
                       "alignment": locked[2]},
            "best_score": best, "best_configuration": list(max(
                finite, key=lambda k: (finite[k], -k[0]))),
            "tied_set": [list(k) for k in finite if finite[k] >= best - TIE_TOL_SD],
            "resolution_steps": steps}


# ---------------------------------------------------------------------------
# One cell, every measure -- runs in a worker
# ---------------------------------------------------------------------------

def feature_sets(c: int, s: int, m: int, alignment: str):
    W = _W
    if alignment == "ap":
        F = top_m(W["ap"][s, :, c], m)
        tie = tie_at_boundary(W["ap"][s, :, c], F)
        return F, tie
    if W["pool_to_gt"][c] < 0:
        return None, False
    elig = W["argmax"][s] == W["pool_to_gt"][c]
    F = top_m(W["logM"][s], m, elig)
    if F is None:
        return None, False
    return F, tie_at_boundary(W["logM"][s], F, elig)


def match_fraction(F: np.ndarray, s: int, t: int) -> float:
    W = _W
    return float((W["sim"][W["row_of"][s], W["col_of"][t], F]
                  >= THETA_FEATURE).mean())


def eval_cell(task: tuple) -> dict:
    """task = (cell_index, c, s, t, m, alignment, mode) with s, t AP indices.
    mode 'real' compares F(c,s) with F(c,t); 'self' compares F(c,s) with
    itself (a control); 'shuffled' compares F(c,s) with a matched draw from t
    (a control)."""
    cell, c, s, t, m, alignment, mode = task
    W = _W
    if (t == s) != (mode == "self"):
        raise ValueError(f"mode {mode!r} with s={s}, t={t}")
    out = {"cell": cell, "c": c, "s": s, "t": t, "feasible": False,
           "rank_ok": False, "alignment_ok": False, "tie": False}
    Fs, tie_s = feature_sets(c, s, m, alignment)
    Ft, tie_t = (Fs, tie_s) if mode == "self" else feature_sets(c, t, m, alignment)
    if Fs is None or Ft is None:
        return out
    out["alignment_ok"] = True
    out["tie"] = bool(tie_s or tie_t)
    if mode == "real":
        out["match_fraction"] = 0.5 * (match_fraction(Fs, s, t)
                                       + match_fraction(Ft, t, s))
    ps = strata_pools(W["strat"][s], Fs)
    pt = strata_pools(W["strat"][t], Ft)
    if ps is None or pt is None:
        return out
    out["feasible"] = True
    Ds, Dt = W["dec"][s], W["dec"][t]
    if mode == "shuffled":
        rng = np.random.default_rng(SEED_BASE + cell + 10 ** 9)
        Ft = draw_side(rng, pt, m)
    if rank_deficient(Ds[Fs]) or rank_deficient(Dt[Ft]):
        return out
    out["rank_ok"] = True
    obs = subspace_measures(Ds[Fs], Dt[Ft])
    ds, dt = matched_draws(SEED_BASE + cell, ps, pt, m)
    Xs, Xt = Ds[ds], Dt[dt]                                   # (100, m, d)
    bad = rank_deficient(Xs) | rank_deficient(Xt)
    out["n_null_rank_deficient"] = int(bad.sum())
    null = measures_from_cosines(principal_cosines(basis(Xs[~bad]),
                                                   basis(Xt[~bad])))
    for k in MEASURES:
        out[k] = float(obs[k])
        out[k + "_null"] = null_summary(float(obs[k]), null[k],
                                        LARGER_IS_SIMILAR[k])
    if mode == "self":
        # Both null sides come from ONE seed here, so a draw can pick the same
        # set twice (only in a tiny stratum) and tie the observed Omega = 1.
        same = np.array([set(a) == set(b) for a, b in zip(ds[~bad], dt[~bad])])
        out["draws_check"] = draws_ok(ds, Fs, W["strat"][s])
        out["self_expected_exceedance"] = float(1.0 - same.mean())
    return out


def draws_ok(draws: np.ndarray, F: np.ndarray, strat: np.ndarray) -> bool:
    """Each draw: right strata, F excluded, distinct within the draw."""
    same = (strat[draws] == strat[F][None, :]).all()
    excl = not np.isin(draws, F).any()
    distinct = all(np.unique(r).size == r.size for r in draws)
    return bool(same and excl and distinct)


# ---------------------------------------------------------------------------
# Loading -- every join by label
# ---------------------------------------------------------------------------

def _dec(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def sha256_16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load(args) -> dict:
    import pandas as pd
    import torch  # noqa: F401  (load_decoder returns a tensor)
    from src.analysis import jackknife_dictionaries as jk
    from src.analysis.carrier_identity import (carriers_from_ap, firing_bin,
                                               order_decoders_to,
                                               seed_label_to_int)
    from src.analysis.carrier_support import carrier_rates_by_cell, cell_index
    from src.analysis.concept_repro import (REFERENCE_PERCENTILE,
                                            detection_matrix, null_thresholds,
                                            observed_best_ap, to_gain)
    from src.analysis.corrector_scan import BIN_EDGES
    from src.analysis.release_derived import (indicator_from_sim,
                                              partner_surrogate)
    from src.analysis.rf_marginal import load_decoder
    from src.analysis.seed_axis import align_to_reference

    gates = {}
    inp = jk.load_join_inputs(args)             # mnn_sim gate, log M, labels
    anchors = [str(a) for a in inp["anchor_labels"]]
    cols = [int(c) for c in inp["column_labels"]]

    with h5py.File(args.ap, "r") as hf:
        ap_seeds = [_dec(s) for s in hf["seeds"][:]]
        ap_concepts = [_dec(c) for c in hf["pooled/concepts"][:]]
        in_pool = hf["pooled/in_argmax_pool"][:].astype(bool)
        ap_pool = hf["pooled/ap"][:][:, :, in_pool]
        prevalence = hf["pooled/prevalence"][:][in_pool]
        fire_counts = hf["fire_counts"][:]
        ap_n_rows = int(hf.attrs["n_rows"])
    with h5py.File(args.null_reductions, "r") as hf:
        null_seeds = [_dec(s) for s in hf["seeds"][:]]
        null_cmax = hf["null_cmax_b"][:]
        null_concepts = [_dec(c) for c in hf["concepts"][:]]
    if ap_seeds != null_seeds:
        raise SystemExit("FATAL: AP and null seed orders differ")
    pool = [c for c, k in zip(ap_concepts, in_pool) if k]
    if pool != null_concepts:
        raise SystemExit("FATAL: AP pool and null concepts differ")
    labels = [seed_label_to_int(s) for s in ap_seeds]          # AP order
    detected = detection_matrix(
        to_gain(observed_best_ap(ap_pool), prevalence),
        null_thresholds(null_cmax, REFERENCE_PERCENTILE, "pooled"))
    carrier, ties = carriers_from_ap(ap_pool)

    # ---- gate: detection and carriers equal the release bundle's ----------
    b = pd.read_csv(args.bundle_concepts)
    want_det = np.zeros_like(detected)
    want_car = np.zeros_like(carrier)
    ci = {c: i for i, c in enumerate(pool)}
    si = {lab: i for i, lab in enumerate(labels)}
    for r in b.itertuples(index=False):
        want_det[si[int(r.anchor_seed)], ci[r.concept]] = bool(r.detected)
        want_car[si[int(r.anchor_seed)], ci[r.concept]] = int(r.carrier_feature)
    if len(b) != detected.size or not np.array_equal(want_det, detected) \
            or not np.array_equal(want_car, carrier):
        raise SystemExit("GATE FAILED: detection or carriers differ from the bundle")
    gates["1_detection_and_carriers_equal_bundle"] = "PASS"

    # ---- gate: the concept population equals the committed tuning file ----
    tun = json.loads(Path(args.tuning_set).read_text(encoding="utf-8"))
    k = detected.sum(axis=0)
    population = [c for c, n in zip(pool, k) if n >= 2]
    if population != tun["population"]:
        raise SystemExit("GATE FAILED: the 650 differ from docs/ct22_tuning_set.json")
    if tun["seed"] != SEED_BASE or len(population) != 650:
        raise SystemExit("GATE FAILED: tuning file seed or population size")
    gates["2_population_equals_tuning_file"] = "PASS (650)"

    # ---- gate: the stored indicator and the carriers give the carrier
    #      analysis's single arm EXACTLY (release_derived's gate 5) ---------
    fbin = firing_bin(fire_counts.astype(np.float64) / float(ap_n_rows))
    strat = cell_index(fbin, None, 3)
    ind = indicator_from_sim(inp["sim"], anchors, cols, THETA_FEATURE)
    partner = partner_surrogate(ind, labels, anchors, cols)
    single = carrier_rates_by_cell(carrier, detected, partner, strat,
                                   len(BIN_EDGES) - 1)
    ref = json.loads(Path(args.carrier_reference).read_text(encoding="utf-8"))
    ref_single = ref["section_3b_matched_ambients"]["arms"]["single_firing_only"]
    for key in ("rate_A_carrier_has_any_partner", "matched_ambient",
                "excess_A_minus_matched_pp", "n_concept_seedpair_cells_evaluated"):
        if single[key] != ref_single[key]:
            raise SystemExit(f"GATE FAILED: {key} {single[key]!r} != {ref_single[key]!r}")
    gates["3_carrier_single_arm_exact"] = {
        "excess_A_minus_matched_pp": single["excess_A_minus_matched_pp"],
        "n_cells": single["n_concept_seedpair_cells_evaluated"]}

    # ---- decoders, ordered into the AP labels by their own meta seed ------
    meta_seed, loaded = {}, {}
    for p in args.ckpts:
        Wd, meta = load_decoder(p, "cpu")
        meta_seed[p] = int(meta["seed"])
        loaded[p] = Wd.numpy().T.astype(np.float64)            # (dict, d)
    ordered = order_decoders_to(ap_seeds, meta_seed)
    if [meta_seed[p] for p in ordered] != labels:
        raise SystemExit("FATAL: decoder order differs from the AP labels")
    dec = np.stack([unit_rows(loaded[p]) for p in ordered])
    del loaded

    # ---- log M and the argmax, aligned by label ---------------------------
    with h5py.File(args.gtilde, "r") as hf:
        argmax_raw = hf["argmax_concept"][:]
        gt_concepts = [_dec(c) for c in hf["concepts"][:]]
    lab_str = [str(x) for x in labels]
    logM = align_to_reference(anchors, inp["logM"], lab_str)
    # release_derived's alignment (gtilde order -> anchors), then -> AP order
    argmax = align_to_reference(
        anchors, align_to_reference(inp["gtilde_seed_order"], argmax_raw, anchors),
        lab_str)
    gi = {c: i for i, c in enumerate(gt_concepts)}
    pool_to_gt = np.array([gi.get(c, -1) for c in pool], dtype=np.int64)

    row_of = np.array([anchors.index(str(x)) for x in labels])
    col_of = np.array([cols.index(int(x)) for x in labels])

    # ---- cells: concepts in pool order, pairs s < s' in NUMERIC labels ----
    pop_idx = [ci[c] for c in population]
    by_label = sorted(range(len(labels)), key=lambda i: labels[i])
    cells = []
    for c in pop_idx:
        for a in range(len(by_label)):
            for bb in range(a + 1, len(by_label)):
                s, t = by_label[a], by_label[bb]
                if detected[s, c] and detected[t, c]:
                    cells.append((len(cells), c, s, t))
    if len(cells) != 109044:
        raise SystemExit(f"GATE FAILED: {len(cells)} cells, not 109,044 (§CT19-R)")
    gates["4_cells_109044"] = "PASS"

    identity_inputs = {}
    for name in ("ap", "null_reductions", "gtilde", "sim", "bundle_concepts",
                 "carrier_reference", "tuning_set"):
        p = Path(getattr(args, name))
        identity_inputs[name] = {"path": str(p), "sha256_16": sha256_16(p)}
    return {"ap": ap_pool, "dec": dec, "strat": strat, "logM": logM,
            "argmax": argmax, "pool_to_gt": pool_to_gt, "sim": inp["sim"],
            "row_of": row_of, "col_of": col_of, "labels": labels,
            "pool": pool, "population": population, "tuning": tun,
            "cells": cells, "carrier": carrier, "ties": ties, "gates": gates,
            "inputs": identity_inputs}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def measurable(r: dict, measure: str) -> bool:
    return bool(r["feasible"] and r["rank_ok"]
                and r[measure + "_null"]["null_sd"] > 0)


def per_concept(rows: list, measure: str, pool: list) -> list:
    by = {}
    for r in rows:
        by.setdefault(r["c"], []).append(r)
    out = []
    for c, rs in by.items():
        ok = [r for r in rs if measurable(r, measure)]
        mf = [r["match_fraction"] for r in rs if r["alignment_ok"]]
        rec = {"concept": pool[c], "n_cells": len(rs), "n_measurable": len(ok),
               "median_match_fraction": float(np.median(mf)) if mf else None}
        if ok:
            rec.update({
                "median_" + measure: float(np.median([r[measure] for r in ok])),
                "median_exceedance": float(np.median(
                    [r[measure + "_null"]["exceedance"] for r in ok])),
                "fraction_beyond_95": float(np.mean(
                    [r[measure + "_null"]["beyond_95"] for r in ok])),
                "median_separation": float(np.median(
                    [r[measure + "_null"]["separation"] for r in ok]))})
        out.append(rec)
    return out


def cell_accounting(rows: list, measure: str) -> dict:
    n = len(rows)
    infeas = sum(1 for r in rows if r["alignment_ok"] and not r["feasible"])
    no_align = sum(1 for r in rows if not r["alignment_ok"])
    rank = sum(1 for r in rows if r["feasible"] and not r["rank_ok"])
    sd0 = sum(1 for r in rows if r["feasible"] and r["rank_ok"]
              and r[measure + "_null"]["null_sd"] == 0)
    return {"n_cells": n,
            "n_alignment_infeasible": no_align,
            "n_draw_infeasible": infeas, "share_draw_infeasible": infeas / n,
            "n_rank_deficient": rank, "share_rank_deficient": rank / n,
            "n_null_sd_zero": sd0, "share_null_sd_zero": sd0 / n,
            "n_measurable": sum(1 for r in rows if measurable(r, measure)),
            "n_ties_at_boundary": sum(1 for r in rows if r["tie"]),
            "n_null_draws_rank_deficient": int(sum(
                r.get("n_null_rank_deficient", 0) for r in rows))}


def quantiles(x) -> dict:
    x = np.asarray([v for v in x if v is not None], dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    q = np.percentile(x, [0, 10, 25, 50, 75, 90, 100])
    return {"n": int(x.size), "mean": float(x.mean()),
            **{f"q{p}": float(v) for p, v in zip((0, 10, 25, 50, 75, 90, 100), q)}}


def crosstab(concepts: list) -> dict:
    t = {"subspace_yes_features_yes": 0, "subspace_yes_features_no": 0,
         "subspace_no_features_yes": 0, "subspace_no_features_no": 0}
    for r in concepts:
        if "median_exceedance" not in r or r["median_match_fraction"] is None:
            continue
        sub = "yes" if r["median_exceedance"] >= 0.95 else "no"
        fea = "yes" if r["median_match_fraction"] >= 0.5 else "no"
        t[f"subspace_{sub}_features_{fea}"] += 1
    return t


def run_cells(tasks: list, workers: int) -> list:
    if workers <= 1:
        return [eval_cell(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(eval_cell, tasks, chunksize=64))


def controls(cells: list, m: int, alignment: str, workers: int,
             n_self: int = 200) -> dict:
    """Self-cell (Omega = 1, exceedance 1, draws in the right strata) and the
    shuffled cell (exceedance near uniform), on the first cells of the set."""
    sub = cells[:n_self]
    selfr = run_cells([(i, c, s, s, m, alignment, "self") for i, c, s, _ in sub],
                      workers)
    ok = [r for r in selfr if r["feasible"] and r["rank_ok"]]
    self_pass = bool(ok and all(
        abs(r["omega"] - 1) < 1e-12 and r["draws_check"]
        and r["omega_null"]["exceedance"] == r["self_expected_exceedance"]
        for r in ok))
    shuf = run_cells([(i, c, s, t, m, alignment, "shuffled") for i, c, s, t in sub],
                     workers)
    exc = [r["omega_null"]["exceedance"] for r in shuf
           if r["feasible"] and r["rank_ok"]]
    return {"self_cell": {"n": len(ok), "verdict": "PASS" if self_pass else "FAIL",
                          "n_below_one": sum(1 for r in ok
                                             if r["self_expected_exceedance"] < 1),
                          "max_abs_omega_minus_1": float(max(
                              abs(r["omega"] - 1) for r in ok)) if ok else None},
            "shuffled_cell": {"n": len(exc),
                              "median_exceedance": float(np.median(exc)) if exc else None,
                              "share_exceedance_ge_0_95": float(np.mean(
                                  np.asarray(exc) >= 0.95)) if exc else None}}


def identity(args, W: dict, mode: str) -> dict:
    from src.sae.train import git_dirty, git_sha
    return {"specification": "PREREGISTRATION amendment CT, §CT22",
            "measurement": f"concept_subspace_{mode}", "git_sha": git_sha(),
            "git_dirty": git_dirty(), "seed_base": SEED_BASE, "n_null": N_NULL,
            "theta_feature": THETA_FEATURE, "rank_tol": RANK_TOL,
            "device_class": "cpu", "python": platform.python_version(),
            "numpy": np.__version__, "n_anchors": len(W["labels"]),
            "ap_seed_labels": W["labels"], "inputs": W["inputs"],
            "smoke_max_cells": args.max_cells}


def write(out: dict, rows_by_config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    h5 = path.with_suffix(".h5")
    with h5py.File(h5, "w") as hf:
        hf.attrs["identity"] = json.dumps(out["identity"], ensure_ascii=False)
        for name, rows in rows_by_config.items():
            g = hf.create_group(name)
            for k in ("cell", "c", "s", "t"):
                g[k] = np.array([r[k] for r in rows], dtype=np.int64)
            for k in ("feasible", "rank_ok", "alignment_ok", "tie"):
                g[k] = np.array([r[k] for r in rows], dtype=bool)
            g["match_fraction"] = np.array(
                [r.get("match_fraction", np.nan) for r in rows])
            for meas in MEASURES:
                g[meas] = np.array([r.get(meas, np.nan) for r in rows])
                for f in ("null_mean", "null_sd", "separation", "exceedance"):
                    g[f"{meas}_{f}"] = np.array(
                        [r[meas + "_null"][f] if meas + "_null" in r else np.nan
                         for r in rows])


def cmd(args) -> None:
    W = load(args)
    _W.update(W)
    tun = set(W["tuning"]["tuning_concepts"])
    in_tuning = lambda c: W["pool"][c] in tun          # noqa: E731
    if args.cmd == "pilot":
        cells = [x for x in W["cells"] if in_tuning(x[1])]
        configs = [(m, a) for m in M_CANDIDATES for a in ALIGNMENTS]
    else:
        pilot = json.loads(Path(args.pilot).read_text(encoding="utf-8"))
        lock = pilot["lock"]["locked"]
        mine = {"m": args.m, "measure": args.measure, "alignment": args.alignment}
        if lock != mine:
            raise SystemExit(f"REFUSING: {mine} is not the pilot's lock {lock}")
        cells = [x for x in W["cells"] if not in_tuning(x[1])]
        configs = [(args.m, args.alignment)]
    if args.max_cells is not None:                 # smoke runs only
        cells = cells[:args.max_cells]
    n_concepts = len({x[1] for x in cells})
    print(f"[ct22] {args.cmd}: {len(cells)} cells over {n_concepts} concepts",
          flush=True)

    m0, a0 = configs[0] if args.cmd == "confirm" else (M_CANDIDATES[0], ALIGNMENTS[0])
    ctl = controls(cells, m0, a0, args.workers)
    print(f"[ct22] controls {json.dumps(ctl)}", flush=True)
    if ctl["self_cell"]["verdict"] != "PASS":
        raise SystemExit("CONTROL FAILED: self-cell")

    rows_by_config, table = {}, []
    for m, a in configs:
        rows = run_cells([(i, c, s, t, m, a, "real") for i, c, s, t in cells],
                         args.workers)
        rows_by_config[f"m{m}_{a}"] = rows
        for meas in MEASURES:
            acc = cell_accounting(rows, meas)
            seps = [r[meas + "_null"]["separation"] for r in rows
                    if measurable(r, meas)]
            table.append({"m": m, "measure": meas, "alignment": a,
                          "median_separation": float(np.median(seps)) if seps
                          else float("nan"),
                          "median_observed": float(np.median(
                              [r[meas] for r in rows if measurable(r, meas)]))
                          if seps else None,
                          "median_null_mean": float(np.median(
                              [r[meas + "_null"]["null_mean"] for r in rows
                               if measurable(r, meas)])) if seps else None,
                          "median_exceedance": float(np.median(
                              [r[meas + "_null"]["exceedance"] for r in rows
                               if measurable(r, meas)])) if seps else None,
                          "accounting": acc})
            print(f"[ct22] m={m} {meas} {a}: median sep "
                  f"{table[-1]['median_separation']:.4f}", flush=True)

    out = {"identity": identity(args, W, args.cmd), "gates": W["gates"],
           "argmax_ties": W["ties"], "controls": ctl,
           "n_cells": len(cells), "n_concepts": n_concepts, "table": table}
    if args.cmd == "pilot":
        out["lock"] = lock_configuration(
            {(r["m"], r["measure"], r["alignment"]): r["median_separation"]
             for r in table})
        out["note"] = "tuning set only; nothing here is scored (§CT22.3)"
    else:
        meas = args.measure
        rows = rows_by_config[f"m{args.m}_{args.alignment}"]
        conc = per_concept(rows, meas, W["pool"])
        acc = cell_accounting(rows, meas)
        out.update(confirm_summary(conc, acc, meas))
    write(out, rows_by_config, args.out)
    print(f"[ct22] wrote {args.out}", flush=True)


def confirm_summary(conc: list, acc: dict, meas: str) -> dict:
    has = [r for r in conc if "median_exceedance" in r]
    p1_hits = sum(1 for r in has if r["median_exceedance"] >= 0.95)
    p2_den = [r for r in has if r["median_match_fraction"] is not None
              and r["median_match_fraction"] < 0.5]
    p2_hits = sum(1 for r in p2_den if r["median_exceedance"] >= 0.95)
    unmeas = []
    if acc["share_null_sd_zero"] > 0.05:
        unmeas.append("null degenerate in more than 5% of cells")
    if acc["share_rank_deficient"] > 0.05:
        unmeas.append("rank-deficient feature sets in more than 5% of cells")
    if acc["share_draw_infeasible"] > 0.05:
        unmeas.append("matched-null draw infeasible in more than 5% of cells")
    p2_unmeas = list(unmeas) + (["fewer than 30 concepts in P2's denominator"]
                                if len(p2_den) < 30 else [])
    verdict = lambda hits, den, um: ("UNMEASURABLE" if um else  # noqa: E731
                                     "HIT" if 2 * hits > den else "MISS")
    return {
        "accounting": acc, "per_concept": conc,
        "n_concepts_with_a_measurable_cell": len(has),
        "n_concepts_without": len(conc) - len(has),
        "distributions": {
            "median_" + meas: quantiles(r["median_" + meas] for r in has),
            "median_exceedance": quantiles(r["median_exceedance"] for r in has),
            "fraction_beyond_95": quantiles(r["fraction_beyond_95"] for r in has),
            "median_match_fraction": quantiles(r["median_match_fraction"] for r in conc)},
        "share_subspace_reproduces": p1_hits / len(has) if has else None,
        "crosstab": crosstab(has),
        "CT22_P1": {"hits": p1_hits, "denominator": len(has),
                    "unmeasurable_causes": unmeas,
                    "outcome": verdict(p1_hits, len(has), unmeas)},
        "CT22_P2": {"hits": p2_hits, "denominator": len(p2_den),
                    "unmeasurable_causes": p2_unmeas,
                    "outcome": verdict(p2_hits, len(p2_den), p2_unmeas)},
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("pilot", "confirm"))
    for k in ("sim", "rf_counts_all_theta", "m1_json", "gtilde",
              "null_reductions", "d_fs", "ap", "bundle_concepts",
              "carrier_reference", "tuning_set", "out"):
        p.add_argument("--" + k.replace("_", "-"), dest=k, type=Path, required=True)
    p.add_argument("--degenerate-flag", type=Path, default=None)
    from src.analysis.design3_join import DEFAULT_CONFIG_PATH
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--ckpts", nargs="+", type=Path, required=True)
    p.add_argument("--expect-ckpts", type=int, required=True)
    p.add_argument("--pilot", type=Path, default=None)
    p.add_argument("--m", type=int, default=None)
    p.add_argument("--measure", choices=MEASURES, default=None)
    p.add_argument("--alignment", choices=ALIGNMENTS, default=None)
    p.add_argument("--max-cells", type=int, default=None,
                   help="SMOKE ONLY: the first N cells; recorded in identity")
    p.add_argument("--workers", type=int, default=int(os.environ.get(
        "SLURM_CPUS_PER_TASK", "1")))
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if len(args.ckpts) != args.expect_ckpts:
        raise SystemExit(f"PRE-FLIGHT FAIL: {len(args.ckpts)} checkpoints, "
                         f"expected {args.expect_ckpts}")
    if args.out.exists():
        raise SystemExit(f"FATAL: {args.out} exists; every artifact goes to a NEW path")
    if args.cmd == "confirm" and None in (args.pilot, args.m, args.measure,
                                          args.alignment):
        raise SystemExit("confirm needs --pilot, --m, --measure, --alignment")
    cmd(args)


if __name__ == "__main__":
    main()
