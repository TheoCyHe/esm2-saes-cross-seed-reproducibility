"""PREREGISTRATION amendment CT, §CT22.5 -- descriptive diagnostics, never scored.

Runs AFTER the confirmatory result is committed, on the 552 confirmatory
concepts, under the locked configuration unless a diagnostic varies it. It
imports `concept_subspace`'s primitives and loader and leaves that module's
confirm path untouched. Nothing here changes a scored quantity (§CT22.5).

  m_curve     Omega and its matched-null exceedance as a function of m
              (1, 2, 3, 5, 8, 13, 21), A(c, f), on every confirmatory cell.
  no_disorder F(c, s) built from features whose argmax concept is NOT
              REGION:Disordered (the null drawn from the same non-Disordered
              features of each stratum), the argmax being g-tilde's
              (`gtilde_compliant_b_ratio.h5` `argmax_concept`, the object log M
              is defined on). Under it ~1% of feature-seed cells are Disordered.
  no_disorder_ap  the same with the argmax of RAW AP over the 703 pool concepts
              -- the object NoR §8.5's 72.4% is a share of (§AW4.2, verbatim:
              "`feature_ap_main_compliant.h5`'s pooled `ap`/`in_argmax_pool`
              (argmax restricted to the 703-pool, per amendment I4)"). Gated:
              the share must reproduce 0.724 at three decimals before anything
              is computed. The share under the argmax of the GAIN G = (AP - p)
              / (1 - p) -- §CD3's rule for the aligned half of the pair -- is
              recorded beside it (commit 77 gated on G and refused at 0.0188,
              job 30439301).
  within_seed the overlap between DIFFERENT concepts' subspaces in the SAME
              seed (a specificity check): Omega(F(c, s), F(c', s)), c' drawn
              from the other confirmatory concepts detected in s.
  cross_concept the same across seeds: Omega(F(c, s), F(c', s')), c' != c
              detected in s' -- a concept-shuffled comparison a reviewer would
              ask for, beside the firing-rate-matched null.
  rotation    per feature f of F(c, s): the share of its direction lying in the
              other seed's concept subspace, ||P_{F(c, s')} f||^2, against its
              best single partner in F(c, s'), max_g cos^2(f, g). The first is
              never below the second (g lies in the subspace); their difference
              is what the subspace captures BEYOND one-to-one matching -- the
              "basis rotation" reading -- and it is reported overall and in the
              cells where no feature of F has an MNN partner at 0.70.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src.analysis import concept_subspace as cs

M_CURVE = (1, 2, 3, 5, 8, 13, 21)
DISORDERED = "REGION:Disordered"
DISORDERED_SHARE_NOR = 0.724          # NoR §8.5, §CD3.2: all feature-seed cells
SEED_DIAG = cs.SEED_BASE + 2 * 10 ** 9      # disjoint from the scored draws

_D: dict = {}


def _f_ap(c: int, s: int, m: int, eligible: np.ndarray | None):
    return cs.top_m(cs._W["ap"][s, :, c], m, eligible)


def _omega_with_null(cell: int, Fs, Ft, s: int, t: int, m: int,
                     elig_s=None, elig_t=None) -> dict | None:
    """Omega and the matched-null exceedance for two feature sets. With
    eligibility masks, the null pool is restricted to eligible features."""
    W = cs._W
    strat_s, strat_t = W["strat"][s].copy(), W["strat"][t].copy()
    if elig_s is not None:
        strat_s[~elig_s] = -1                    # never in F's strata
    if elig_t is not None:
        strat_t[~elig_t] = -1
    ps, pt = cs.strata_pools(strat_s, Fs), cs.strata_pools(strat_t, Ft)
    if ps is None or pt is None:
        return {"feasible": False}
    Ds, Dt = W["dec"][s], W["dec"][t]
    if cs.rank_deficient(Ds[Fs]) or cs.rank_deficient(Dt[Ft]):
        return {"feasible": True, "rank_ok": False}
    obs = float(cs.subspace_measures(Ds[Fs], Dt[Ft])["omega"])
    ds, dt = cs.matched_draws(cs.SEED_BASE + cell, ps, pt, m)
    null = cs.measures_from_cosines(cs.principal_cosines(
        cs.basis(Ds[ds]), cs.basis(Dt[dt])))["omega"]
    ns = cs.null_summary(obs, null, True)
    return {"feasible": True, "rank_ok": True, "omega": obs, **ns}


def gain_argmax_mask(ap: np.ndarray, prevalence: np.ndarray, concept: int
                     ) -> tuple[np.ndarray, float]:
    """(seed, feature) True where the argmax over pool concepts of
    G = (AP - p) / (1 - p) is NOT `concept`; and the share where it IS.
    Seed by seed, so the float64 gain never exists for all seeds at once."""
    p = np.asarray(prevalence, dtype=np.float64)
    keep = np.empty(ap.shape[:2], dtype=bool)
    for s in range(ap.shape[0]):
        g = (ap[s].astype(np.float64) - p) / (1.0 - p)
        keep[s] = g.argmax(axis=1) != concept
    return keep, float((~keep).mean())


def ap_argmax_mask(ap: np.ndarray, concept: int) -> tuple[np.ndarray, float]:
    """The same with the argmax of RAW AP over pool concepts (§AW4.2)."""
    keep = ap.argmax(axis=2) != concept
    return keep, float((~keep).mean())


def task_m_curve(t):
    cell, c, s, u, m = t
    Fs, Fu = _f_ap(c, s, m, None), _f_ap(c, u, m, None)
    r = _omega_with_null(cell, Fs, Fu, s, u, m)
    r.update({"cell": cell, "c": c, "m": m,
              "match_fraction": 0.5 * (cs.match_fraction(Fs, s, u)
                                       + cs.match_fraction(Fu, u, s))})
    return r


def task_no_disorder(t):
    cell, c, s, u, m = t
    keep = _D["not_disordered"]
    Fs, Fu = _f_ap(c, s, m, keep[s]), _f_ap(c, u, m, keep[u])
    if Fs is None or Fu is None:
        return {"cell": cell, "c": c, "feasible": False, "alignment_ok": False}
    r = _omega_with_null(cell, Fs, Fu, s, u, m, keep[s], keep[u])
    r.update({"cell": cell, "c": c, "alignment_ok": True,
              "match_fraction": 0.5 * (cs.match_fraction(Fs, s, u)
                                       + cs.match_fraction(Fu, u, s))})
    return r


def task_other_concept(t):
    """(cell, c, s, u, c_other, m, kind): kind 'within' compares F(c, s) with
    F(c_other, s); 'cross' compares F(c, s) with F(c_other, u)."""
    cell, c, s, u, co, m, kind = t
    Fs = _f_ap(c, s, m, None)
    target = s if kind == "within" else u
    Fo = _f_ap(co, target, m, None)
    W = cs._W
    # Feature INDICES are comparable only within one dictionary (gotcha 18):
    # across seeds an index intersection is chance (~0.00088 for two 3-sets).
    shared = int(np.intersect1d(Fs, Fo).size) if kind == "within" else None
    X, Y = W["dec"][s][Fs], W["dec"][target][Fo]
    om = float(cs.subspace_measures(X, Y)["omega"]) if not (
        cs.rank_deficient(X) or cs.rank_deficient(Y)) else float("nan")
    return {"cell": cell, "c": c, "c_other": co, "kind": kind,
            "omega": om, "n_shared_features": shared}


def task_rotation(t):
    cell, c, s, u, m = t
    W = cs._W
    Fs, Fu = _f_ap(c, s, m, None), _f_ap(c, u, m, None)
    out = {"cell": cell, "c": c,
           "match_fraction": 0.5 * (cs.match_fraction(Fs, s, u)
                                    + cs.match_fraction(Fu, u, s))}
    caps, bests = [], []
    for a, b, Fa, Fb in ((s, u, Fs, Fu), (u, s, Fu, Fs)):
        X, Y = W["dec"][a][Fa], W["dec"][b][Fb]                 # unit rows
        V = cs.basis(Y)                                         # (d, m)
        caps.append(((X @ V) ** 2).sum(axis=1))                 # ||P_V f||^2
        bests.append(((X @ Y.T) ** 2).max(axis=1))              # best partner
    cap, best = np.concatenate(caps), np.concatenate(bests)
    out.update({"captured": float(cap.mean()), "best_single": float(best.mean()),
                "beyond_best": float((cap - best).mean())})
    return out


def _run(fn, tasks, workers):
    if workers <= 1:
        return [fn(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, tasks, chunksize=64))


def _summ(rows, key="omega"):
    v = [r[key] for r in rows if r.get("rank_ok", True) and r.get("feasible", True)
         and key in r and np.isfinite(r[key])]
    return cs.quantiles(v)


def per_concept_median(rows, key):
    by = {}
    for r in rows:
        if r.get("feasible", True) and r.get("rank_ok", True) and key in r:
            by.setdefault(r["c"], []).append(r[key])
    return {c: float(np.median(v)) for c, v in by.items()}


def main(argv=None) -> None:
    p = cs.build_parser()
    p.add_argument("--diagnostics", nargs="+",
                   default=["rotation", "within_seed", "cross_concept",
                            "no_disorder", "m_curve"])
    args = p.parse_args(argv)
    if args.cmd != "confirm" or None in (args.pilot, args.m, args.measure,
                                         args.alignment):
        raise SystemExit("run as: confirm --pilot ... --m --measure --alignment")
    if args.out.exists():
        raise SystemExit(f"FATAL: {args.out} exists")
    W = cs.load(args)
    cs._W.update(W)
    pilot = json.loads(Path(args.pilot).read_text(encoding="utf-8"))
    lock = pilot["lock"]["locked"]
    if lock != {"m": args.m, "measure": args.measure, "alignment": args.alignment}:
        raise SystemExit(f"REFUSING: not the pilot's lock {lock}")
    tun = set(W["tuning"]["tuning_concepts"])
    cells = [x for x in W["cells"] if W["pool"][x[1]] not in tun]
    if args.max_cells is not None:
        cells = cells[:args.max_cells]
    m0 = args.m
    out = {"identity": {**cs.identity(args, W, "diagnostics"),
                        "specification": "PREREGISTRATION amendment CT, §CT22.5 "
                                         "(descriptive, never scored)",
                        "locked": lock, "seed_diag": SEED_DIAG},
           "gates": W["gates"], "n_cells": len(cells),
           "n_concepts": len({x[1] for x in cells})}

    if "m_curve" in args.diagnostics:
        curve = []
        for m in M_CURVE:
            rows = _run(task_m_curve, [(i, c, s, u, m) for i, c, s, u in cells],
                        args.workers)
            pc = per_concept_median(rows, "exceedance")
            curve.append({"m": m, "omega": _summ(rows),
                          "exceedance": _summ(rows, "exceedance"),
                          "share_concepts_median_exceedance_ge_0_95":
                              float(np.mean([v >= 0.95 for v in pc.values()])),
                          "n_concepts": len(pc),
                          "n_infeasible": sum(1 for r in rows if not r["feasible"])})
            print(f"[diag] m_curve m={m}: {json.dumps(curve[-1]['omega'])}", flush=True)
        out["m_curve"] = curve

    for nd_key in ("no_disorder", "no_disorder_ap"):
        if nd_key not in args.diagnostics:
            continue
        if nd_key == "no_disorder":
            gi = W["pool_to_gt"][W["pool"].index(DISORDERED)]
            _D["not_disordered"] = W["argmax"] != gi
            gate = None
        else:
            import h5py
            with h5py.File(args.ap, "r") as hf:
                in_pool = hf["pooled/in_argmax_pool"][:].astype(bool)
                prev = hf["pooled/prevalence"][:][in_pool]
            dis = W["pool"].index(DISORDERED)
            _, g_share = gain_argmax_mask(W["ap"], prev, dis)
            keep, share = ap_argmax_mask(W["ap"], dis)
            print(f"[diag] Disordered argmax share: raw AP {share!r}, gain {g_share!r}",
                  flush=True)
            if round(share, 3) != DISORDERED_SHARE_NOR:
                raise SystemExit(f"GATE FAILED: Disordered raw-AP argmax share {share!r} "
                                 f"does not reproduce NoR §8.5's {DISORDERED_SHARE_NOR}")
            _D["not_disordered"] = keep
            gate = {"disordered_raw_ap_argmax_share": share,
                    "reproduces_nor_8_5_at_3_decimals": True,
                    "disordered_gain_argmax_share_for_comparison": g_share}
        rows = _run(task_no_disorder, [(i, c, s, u, m0) for i, c, s, u in cells],
                    args.workers)
        pc_exc = per_concept_median(rows, "exceedance")
        pc_mf = {}
        for r in rows:
            if r.get("alignment_ok"):
                pc_mf.setdefault(r["c"], []).append(r["match_fraction"])
        pc_mf = {c: float(np.median(v)) for c, v in pc_mf.items()}
        both = [c for c in pc_exc if c in pc_mf]
        out[nd_key] = {
            "gate": gate,
            "share_feature_seed_cells_argmax_disordered":
                float((~_D["not_disordered"]).mean()),
            "omega": _summ(rows), "exceedance": _summ(rows, "exceedance"),
            "n_cells_no_F": sum(1 for r in rows if not r.get("alignment_ok")),
            "n_cells_draw_infeasible": sum(1 for r in rows if r.get("alignment_ok")
                                           and not r["feasible"]),
            "n_concepts": len(pc_exc),
            "share_subspace_reproduces": float(np.mean(
                [pc_exc[c] >= 0.95 for c in pc_exc])) if pc_exc else None,
            "crosstab": cs.crosstab([{"median_exceedance": pc_exc[c],
                                      "median_match_fraction": pc_mf[c]}
                                     for c in both]),
            "concept_REGION_Disordered_in_confirmatory_set":
                DISORDERED in [W["pool"][x[1]] for x in cells]}
        print(f"[diag] {nd_key} {json.dumps(out[nd_key]['crosstab'])}", flush=True)

    if "rotation" in args.diagnostics:
        rows = _run(task_rotation, [(i, c, s, u, m0) for i, c, s, u in cells],
                    args.workers)
        z = [r for r in rows if r["match_fraction"] == 0]
        out["rotation"] = {
            k: {"all": cs.quantiles([r[k] for r in rows]),
                "match_fraction_0": cs.quantiles([r[k] for r in z])}
            for k in ("captured", "best_single", "beyond_best")}
        out["rotation"]["n_cells_match_fraction_0"] = len(z)
        print(f"[diag] rotation {json.dumps(out['rotation']['beyond_best'])}",
              flush=True)

    for kind in ("within_seed", "cross_concept"):
        if kind not in args.diagnostics:
            continue
        # detection per (concept, seed), recovered from the cells: every
        # population concept is detected in >= 2 seeds, so each of its
        # detected seeds appears in at least one cell
        detected = {(c, x) for _, c, s, u in W["cells"] for x in (s, u)}
        det = {}
        conf = sorted({x[1] for x in cells})
        rng = np.random.default_rng(SEED_DIAG + (0 if kind == "within_seed" else 1))
        tasks = []
        for i, c, s, u in cells:
            target = s if kind == "within_seed" else u
            if target not in det:
                det[target] = [cc for cc in conf
                               if (cc, target) in detected]
            others = [cc for cc in det[target] if cc != c]
            co = others[int(rng.integers(len(others)))]
            tasks.append((i, c, s, u, co, m0,
                          "within" if kind == "within_seed" else "cross"))
        rows = _run(task_other_concept, tasks, args.workers)
        out[kind] = {"omega": _summ(rows),
                     "share_with_a_shared_feature": float(np.mean(
                         [r["n_shared_features"] > 0 for r in rows]))
                     if kind == "within_seed" else None,
                     "n_shared_features": cs.quantiles(
                         [r["n_shared_features"] for r in rows])
                     if kind == "within_seed" else None}
        print(f"[diag] {kind} {json.dumps(out[kind]['omega'])}", flush=True)

    cs.write(out, {}, args.out)
    print(f"[diag] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
