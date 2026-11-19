"""COMPONENT 4 (PREREGISTRATION amendment BT): the classification composition at
the remaining four thinness-grid points -- minimum observed experimental
positives 5, 10, 30 and 100.

---------------------------------------------------------------------------
A GRID POINT IS A BUILD, NOT A RE-RUN. WHAT THREADING `T` ACTUALLY REQUIRES.
---------------------------------------------------------------------------
Neither `classification_bh` nor `classification_2x2` takes a minimum-positive
count; their CLIs have no such argument (§BS-D0 established that by reading
them). What the restriction touches, and what it does not, follows from BI2:

  * THE POOLED / ORPHAN AXIS DOES NOT MOVE (BI2b). AV4 defines ORPHAN on the
    703-concept pooled stratum, and restricting that by EXPERIMENTAL positive
    counts is undefined for 485 of its concepts. So `clears_pooled_*` is READ
    from component 3's artifact, not recomputed -- and this module asserts the
    seed axis it is read on rather than assuming it (gotcha 29).

  * THE P-VALUES DO NOT MOVE EITHER. `pooled_exceedance` is per (concept, seed,
    arm): the null pool is that concept's own draws, so `p[f, c]` is what it is
    whatever else is in the family. The restriction changes WHICH TESTS ARE IN
    THE FAMILY, and therefore the BH threshold -- nothing else.

  * SO THE ONE THING THAT MOVES IS `bh_reject`'s family, and with it `m` and
    the realised cut. `k_min` moves with it, and so does the at-floor count,
    which is summed over retained concepts only.

The p-values are RECOMPUTED from the raw experimental null rather than read
from component 1's stored `p_*`, which is float32. Re-running BH on a float32
round-trip of a float64 computation can move a rejection at the margin, and
then the `T = 1` reproduction below would fail for a reason that has nothing to
do with the restriction. Recomputing runs the SAME code path component 1 ran,
which is what makes the reproduction a regression baseline rather than a
coincidence.

---------------------------------------------------------------------------
THE REGRESSION BASELINE IS `T = 1`, ON REAL DATA (criterion 1).
---------------------------------------------------------------------------
At `T = 1` the retained set is exactly the 218 in-pool concepts -- `>= 1` is
satisfied by every concept the AP artifact scores in the pool -- so the
restricted path must reproduce component 3 EXACTLY. Three things are asserted
and each is fatal:

    1. the recomputed `reject` is bit-identical to `classification_bh_*`'s;
    2. the restricted BH at `T = 1` is bit-identical to the unrestricted one;
    3. the cell counts equal `classification_2x2_bq.h5`'s.

A component whose unrestricted point does not reproduce the committed
unrestricted point is not measuring the restriction.

---------------------------------------------------------------------------
THE TWO CAUSES: WHY THE SET DIFFERENCE IS NOT THE DECOMPOSITION (BI3(b)).
---------------------------------------------------------------------------
BI3(b) names two causes and requires them reported separately and never netted:
`lost` -- features leaving ALIGNED because a qualifying concept left the pool;
`gained` -- features entering because the loosened threshold admitted them.

**The set difference between two grid points does not establish either.** A
feature can leave ALIGNED because its concept was removed OR because the
realised BH cut moved against it, and `A_k \\ A_{k+1}` contains both. Assigning
a cause to a set difference would be reading a mechanism off a number that
cannot distinguish them -- gotcha 30's extension, which says to add the
discriminating quantity to the MEASUREMENT rather than to the reading
afterwards.

So a COUNTERFACTUAL is computed. `A_mid` holds the pool at `T_{k+1}` while
keeping the rejections computed at `T_k`:

    A_k    = ANY_{c in Pool(T_k)}    [ reject_{T_k}[f,c]     AND G[f,c] >= 0.05 ]
    A_mid  = ANY_{c in Pool(T_k+1)}  [ reject_{T_k}[f,c]     AND G[f,c] >= 0.05 ]
    A_k+1  = ANY_{c in Pool(T_k+1)}  [ reject_{T_k+1}[f,c]   AND G[f,c] >= 0.05 ]

`A_mid` differs from `A_k` ONLY by pool removal and from `A_k+1` ONLY by the
rejection set. Because Pool(T_k+1) is a SUBSET of Pool(T_k) and the rejections
are held fixed, `A_mid` is a subset of `A_k` -- so `gained_pool` is ZERO BY
CONSTRUCTION, and this module ASSERTS it rather than assuming it. That gives

    net = gained_threshold - lost_threshold - lost_pool

and the identity is checked, not asserted (criterion 9). The committed two-way
quantities are reported alongside: `lost = |A_k \\ A_k+1|`,
`gained = |A_k+1 \\ A_k|`, `net = gained - lost`, also checked. `net` is
reported IN ADDITION, never INSTEAD (BI3(b)).

---------------------------------------------------------------------------
CENSORING (BP4) MOVES WITH THE FLOOR.
---------------------------------------------------------------------------
`k_min` falls as the pool shrinks, so a seed floor-limited at one grid point
need not be at the next. The floor-limited set is recomputed and reported PER
GRID POINT and never carried forward, and the flag rides on every per-seed row
rather than a caption (BP4 item 1).
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np

from src.analysis.bg3_propagation import (  # noqa: E402
    anti_conservative_bound,
    relative_positive_loss,
    surviving_draw_mask,
    worst_cells,
)
from src.analysis.bh_resolution_floor import k_min  # noqa: E402
from src.analysis.classification_2x2 import (  # noqa: E402
    CELL_ALIGNED_BOTH,
    CELL_ALIGNED_EXP,
    CELL_EXCLUDED,
    CELL_NAMES,
    CELL_NOT_EVALUABLE,
    CELL_ORPHAN,
    GAIN_FLOOR,
    at_floor_counts,
    clears_bar,
    classify_cells,
    gain_from_ap,
    verify_outcome_space,
)
from src.analysis.classification_bh import (  # noqa: E402
    Q_FDR,
    bh_pass_one_slab,
    bh_reject,
)
from src.sae.train import git_dirty, git_sha  # noqa: E402

# BI1's grid, RATIFIED and not a tunable. Threshold 1 is the unrestricted
# pre-registered primary and is present here so the component can prove it
# reproduces the committed classification, NOT so it can be rebuilt.
GRID = (1, 5, 10, 30, 100)

__all__ = [
    "GRID",
    "retained_mask",
    "restricted_reject",
    "decompose_aligned",
    "clearance",
    "GridRegressionFailure",
    "DecompositionViolation",
]


class GridRegressionFailure(Exception):
    """Raised when `T = 1` does not reproduce the committed unrestricted point.

    FATAL by criterion 1. An exception rather than a warning because every
    number this module produces at `T > 1` is only interpretable as a
    restriction of a baseline it has been shown to reproduce.
    """


class DecompositionViolation(Exception):
    """Raised when the decomposition does not sum to the net change.

    FATAL by criterion 9. A decomposition that is asserted rather than checked
    is not a decomposition -- it is two numbers next to each other.
    """


def retained_mask(n_positives: np.ndarray, threshold: int) -> np.ndarray:
    """BI1's restriction over the in-pool concept axis: `n_positives >= T`.

    THE BOUNDARY IS `>=` AND IT DOES REAL WORK. `BINDING:hypoxanthine` has
    exactly five observed positives, so it SURVIVES `T = 5` and leaves at
    `T = 10`; a strict `>` would have dropped it a point earlier and removed
    the only grid point at which the two-floor structure can still fire
    (§BS-D3).

    `n_positives` is the OBSERVED experimental positive count per in-pool
    concept, read from the AP artifact's own axis.
    """
    n = np.asarray(n_positives)
    if n.ndim != 1:
        raise ValueError(f"n_positives is {n.shape}, expected 1-D over the "
                         "in-pool concept axis")
    if int(threshold) < 1:
        raise ValueError(f"threshold {threshold} < 1: the grid's unrestricted "
                         "point is 1 and a threshold below it is not a "
                         "restriction of anything")
    return n >= int(threshold)


def restricted_reject(p: np.ndarray, not_evaluable: np.ndarray,
                      keep: np.ndarray, q: float = Q_FDR) -> dict:
    """BH over the family restricted to `keep`, on the FULL concept grid.

    `p` is (n_features, n_concepts) with NaN at non-evaluable concepts;
    `not_evaluable` and `keep` are (n_concepts,) bool. Returns the same keys
    `bh_pass_one_slab` returns for the family plus `reject`, a full-width
    (n_features, n_concepts) bool that is False everywhere outside `keep`.

    THE FAMILY IS THE INTERSECTION, and both halves matter. A concept outside
    `keep` is not in the pool at this grid point; a concept that is
    NOT_EVALUABLE has no reference distribution at any grid point (BH3). Either
    way its tests cannot enter `m`, and a test outside the family is not a
    non-rejection -- it is not a test.
    """
    p = np.asarray(p, dtype=np.float64)
    ne = np.asarray(not_evaluable, dtype=bool)
    kp = np.asarray(keep, dtype=bool)
    if p.ndim != 2 or ne.shape != (p.shape[1],) or kp.shape != (p.shape[1],):
        raise ValueError(
            f"restricted_reject: p is {p.shape}, not_evaluable {ne.shape}, "
            f"keep {kp.shape} -- both masks must be 1-D over p's concept axis")
    in_family = kp & ~ne
    fam = np.zeros(p.shape, dtype=bool)
    fam[:, in_family] = True
    if not fam.any():
        return {"reject": np.zeros(p.shape, dtype=bool), "n_tests": 0,
                "n_rejected": 0, "threshold": None, "q": float(q),
                "n_concepts_in_family": 0,
                "n_concepts_retained": int(kp.sum()),
                "n_concepts_retained_not_evaluable": int((kp & ne).sum())}
    res = bh_reject(p[fam], q=q)
    reject = np.zeros(p.shape, dtype=bool)
    reject[fam] = res["reject"]
    return {"reject": reject, "n_tests": res["n_tests"],
            "n_rejected": res["n_rejected"], "threshold": res["threshold"],
            "q": float(q), "n_concepts_in_family": int(in_family.sum()),
            "n_concepts_retained": int(kp.sum()),
            "n_concepts_retained_not_evaluable": int((kp & ne).sum())}


def decompose_aligned(a_prev: np.ndarray, a_mid: np.ndarray,
                      a_next: np.ndarray) -> dict:
    """BI3(b)'s two causes, SEPARATED BY A COUNTERFACTUAL and then checked.

    All three arguments are (n_features,) bool ALIGNED masks:
    `a_prev` at `T_k`; `a_mid` at pool `T_{k+1}` with `T_k`'s rejections;
    `a_next` at `T_{k+1}`.

    Raises `DecompositionViolation` if `a_mid` is not a subset of `a_prev` --
    that subset relation is what makes `gained_pool` zero BY CONSTRUCTION, so
    a violation means the counterfactual was built wrong and the causal split
    is not the split it claims to be. Also raises if either summation fails.
    """
    p_, m_, n_ = (np.asarray(x, dtype=bool) for x in (a_prev, a_mid, a_next))
    if not (p_.shape == m_.shape == n_.shape) or p_.ndim != 1:
        raise ValueError("decompose_aligned: the three masks must be 1-D and "
                         "the same shape")

    gained_pool = int(np.count_nonzero(m_ & ~p_))
    if gained_pool:
        raise DecompositionViolation(
            f"{gained_pool} features are ALIGNED under the RESTRICTED pool "
            "with the PREVIOUS point's rejections but were not ALIGNED at the "
            "previous point. Removing concepts cannot add one, so the "
            "counterfactual does not hold the rejections fixed and the causal "
            "split is not what it claims to be")

    lost_pool = int(np.count_nonzero(p_ & ~m_))
    lost_threshold = int(np.count_nonzero(m_ & ~n_))
    gained_threshold = int(np.count_nonzero(n_ & ~m_))

    lost = int(np.count_nonzero(p_ & ~n_))
    gained = int(np.count_nonzero(n_ & ~p_))
    net = int(n_.sum()) - int(p_.sum())

    if gained - lost != net:
        raise DecompositionViolation(
            f"two-way: gained {gained} - lost {lost} = {gained - lost}, "
            f"net {net}")
    if gained_threshold - lost_threshold - lost_pool != net:
        raise DecompositionViolation(
            f"three-way: gained_threshold {gained_threshold} - "
            f"lost_threshold {lost_threshold} - lost_pool {lost_pool} = "
            f"{gained_threshold - lost_threshold - lost_pool}, net {net}")

    return {
        # BI3(b)'s committed pair, plus net IN ADDITION and never instead.
        "lost": lost, "gained": gained, "net": net,
        "n_aligned_prev": int(p_.sum()), "n_aligned_next": int(n_.sum()),
        # The causal split the set difference alone cannot supply.
        "lost_pool": lost_pool, "lost_threshold": lost_threshold,
        "gained_threshold": gained_threshold, "gained_pool": gained_pool,
        "n_aligned_counterfactual": int(m_.sum()),
        "two_way_sums": True, "three_way_sums": True,
    }


def clearance(at_floor_total: int, floor: int, n_concepts: int) -> dict:
    """BT-C3's quantity: at-floor supply per unit of floor.

    `at_floor_total / k_min`, and -- ONLY WHEN `k_min == n_concepts` -- the
    same number as the mean at-floor count per concept. The equivalence is
    CHECKED rather than assumed: it holds because `q*S = 1` exactly puts
    `k_min = C` at every grid point, and if that ever stopped holding the two
    readings would silently diverge while both kept a plausible value.
    """
    if floor <= 0:
        raise ValueError(f"floor is {floor}; clearance is undefined against a "
                         "non-positive floor")
    ratio = at_floor_total / floor
    per_concept = at_floor_total / n_concepts if n_concepts else float("nan")
    equivalent = bool(floor == n_concepts)
    return {"at_floor_total": int(at_floor_total), "k_min": int(floor),
            "n_concepts": int(n_concepts), "clearance": float(ratio),
            "mean_at_floor_per_concept": float(per_concept),
            "clearance_equals_mean_per_concept": equivalent,
            "note": ("the two readings coincide only while k_min == C; "
                     "checked, not assumed")}


# --------------------------------------------------------------------------
# CLI -- the driver.
# --------------------------------------------------------------------------

def _labels(ds) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in ds[()]]


def main() -> None:
    ap_ = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--raw-null-experimental", type=Path, required=True)
    ap_.add_argument("--null-experimental-reductions", type=Path, required=True)
    ap_.add_argument("--ap-experimental", type=Path, required=True)
    ap_.add_argument("--bh-experimental", type=Path, required=True)
    ap_.add_argument("--classification-2x2", type=Path, required=True,
                     help="classification_2x2_bq.h5 -- READ for its unchanged "
                          "pooled axis and for the T=1 regression baseline.")
    ap_.add_argument("--q", type=float, default=Q_FDR)
    ap_.add_argument("--gain-floor", type=float, default=GAIN_FLOOR)
    ap_.add_argument("--out", type=Path, required=True)
    ap_.add_argument("--report", type=Path, required=True)
    ap_.add_argument("--seeds", type=int, default=None,
                     help="cost pilot only: process the first N seeds. A run "
                          "with this set writes is_cost_pilot=True and its "
                          "counts are NOT the composition.")
    args = ap_.parse_args()

    # ---- the observed experimental AP, and the grid's own variable ---------
    with h5py.File(args.ap_experimental, "r") as hf:
        g = hf["experimental"]
        inpool = g["in_argmax_pool"][()]
        prev_e = g["prevalence"][()][inpool]
        npos_e = g["n_positives"][()][inpool]
        conc_e = [c for c, k in zip(_labels(g["concepts"]), inpool) if k]
        seeds_e = _labels(hf["seeds"])
        ap_ds = g["ap"]
        n_f = ap_ds.shape[1]
        ap_e = np.empty((len(seeds_e), n_f, int(inpool.sum())), dtype=np.float32)
        for s in range(len(seeds_e)):
            ap_e[s] = ap_ds[s][:, inpool]

    # ---- component 1/2's output: the committed rejections and axes --------
    with h5py.File(args.bh_experimental, "r") as hf:
        bh_seeds = _labels(hf["seeds"])
        bh_concepts = _labels(hf["concepts"])
        notev = {a: hf[f"not_evaluable_{a}"][()] for a in ("a", "b")}
        reject_committed = {a: hf[f"reject_{a}"] for a in ("a", "b")}
        surv_draws = {a: hf[f"surviving_draws_{a}"][()] for a in ("a", "b")}
        bh_identity = {k: (hf.attrs[k].decode()
                           if isinstance(hf.attrs[k], bytes) else hf.attrs[k])
                       for k in hf.attrs}
        reject_committed = {a: reject_committed[a][()] for a in ("a", "b")}

    # ---- component 3's output: the pooled axis, which does NOT move -------
    with h5py.File(args.classification_2x2, "r") as hf:
        c3_seeds = _labels(hf["seeds"])
        clears_pooled = {a: hf[f"clears_pooled_{a}"][()] for a in ("a", "b")}
        notev_pooled = {a: hf[f"not_evaluable_pooled_{a}"][()]
                        for a in ("a", "b")}
        c3_cells = {a: hf[f"cell_{a}"][()] for a in ("a", "b")}
        c3_identity = {k: (hf.attrs[k].decode()
                           if isinstance(hf.attrs[k], bytes) else hf.attrs[k])
                       for k in hf.attrs}

    # ---- the null's own reductions: BG3's inputs --------------------------
    with h5py.File(args.null_experimental_reductions, "r") as hf:
        red_seeds = _labels(hf["seeds"])
        red_concepts = _labels(hf["concepts"])
        surv_pos = {a: hf[f"surviving_positives_{a}"][()] for a in ("a", "b")}
        n_pos_obs = hf["n_positives_observed"][()]
        surv_red = {a: hf[f"surviving_draws_{a}"][()] for a in ("a", "b")}

    # ---- SEED AXES ALIGNED ON LABELS, NEVER BY POSITION (gotcha 29) -------
    for name, lab in (("ap_experimental", seeds_e),
                      ("classification_2x2", c3_seeds),
                      ("null_reductions", red_seeds)):
        if lab != bh_seeds:
            raise SystemExit(
                f"SEED AXIS MISMATCH between the BH artifact and {name}.\n"
                f"  bh : {bh_seeds}\n  {name}: {lab}\n"
                "Joining these by position would mis-pair seeds and yield a "
                "complete, internally consistent, wrong result (gotcha 29).")
    # ---- CONCEPT AXES POSITIONAL, not set-wise (gotcha 27) ----------------
    for name, lab in (("ap_experimental", conc_e),
                      ("null_reductions", red_concepts)):
        if lab != bh_concepts:
            same = sorted(lab) == sorted(bh_concepts)
            first = next((i for i, (x, y) in enumerate(zip(lab, bh_concepts))
                          if x != y), None)
            raise SystemExit(
                f"EXPERIMENTAL CONCEPT AXIS MISMATCH against {name} (same "
                f"concepts, different order: {same}; first disagreement at "
                f"{first}). Only a POSITIONAL check discriminates a mapping "
                "defect (gotcha 27).")
    for a in ("a", "b"):
        if not np.array_equal(surv_draws[a], surv_red[a]):
            raise SystemExit(
                f"surviving_draws_{a} disagrees between the BH artifact and "
                "the reductions it was built from -- the two describe "
                "different runs.")

    n_c = len(bh_concepts)
    n_seeds = len(bh_seeds) if args.seeds is None else min(args.seeds,
                                                           len(bh_seeds))
    qf = Fraction(args.q).limit_denominator(10 ** 6)

    # ---- THE FLOORS, FROM PARAMETERS ALONE, BEFORE ANY DATA (gotcha 42) ---
    keeps = {T: retained_mask(npos_e, T) for T in GRID}
    pool_n = {T: int(keeps[T].sum()) for T in GRID}
    floors = {T: k_min(pool_n[T], n_f, 20, qf) for T in GRID}
    print("grid, floors computed from design parameters with no data read:")
    for T in GRID:
        print(f"  T >= {T:<4} C = {pool_n[T]:<4} k_min = {floors[T]:<4} "
              f"k_min == C: {floors[T] == pool_n[T]}")

    # ---- the pass ---------------------------------------------------------
    rows: dict = {}
    aligned: dict = {}            # (arm, T) -> (n_seeds, n_f) bool
    rej_by_T: dict = {}           # (arm, s, T) -> (n_f, n_c) bool, one seed
    at_floor: dict = {}           # (arm, T) -> list per seed
    regression: dict = {"reject_bit_identical": {}, "cells_equal": {}}

    with h5py.File(args.raw_null_experimental, "r") as hf:
        raw_concepts = _labels(hf["concepts"])
        raw_seeds = _labels(hf["seeds"])
        if raw_concepts != bh_concepts or raw_seeds != bh_seeds:
            raise SystemExit(
                "RAW NULL axis disagrees with the BH artifact -- refusing.")
        raw_identity = {k: (hf.attrs[k].decode()
                            if isinstance(hf.attrs[k], bytes) else hf.attrs[k])
                        for k in hf.attrs}

        for arm in ("a", "b"):
            for T in GRID:
                aligned[(arm, T)] = np.zeros((n_seeds, n_f), dtype=bool)
                at_floor[(arm, T)] = []
                rows[(arm, T)] = []
            for s in range(n_seeds):
                slab = hf[f"gain_{arm}"][s]        # (shift, concept, feature)
                g_obs = gain_from_ap(ap_e[s], prev_e)
                full = bh_pass_one_slab(slab, g_obs, ~notev[arm], q=args.q)
                del slab

                # ---- CRITERION 1, on real data: the recomputed rejections
                # must be BIT-IDENTICAL to the committed ones. ------------
                same = np.array_equal(full["reject"], reject_committed[arm][s])
                regression["reject_bit_identical"][f"{arm}:{bh_seeds[s]}"] = \
                    bool(same)
                if not same:
                    d = int(np.count_nonzero(
                        full["reject"] ^ reject_committed[arm][s]))
                    raise GridRegressionFailure(
                        f"arm {arm.upper()} seed {bh_seeds[s]}: the recomputed "
                        f"BH rejections differ from component 1's committed "
                        f"ones in {d} of {full['reject'].size} tests. Every "
                        "restricted point below would be a restriction of a "
                        "baseline this module does not reproduce.")

                for T in GRID:
                    kp = keeps[T]
                    r = restricted_reject(full["p"], notev[arm], kp, q=args.q)
                    rej_by_T[T] = r["reject"]
                    ce = clears_bar(r["reject"][:, kp], g_obs[:, kp],
                                    args.gain_floor)
                    e_ok = bool((kp & ~notev[arm]).any())
                    p_ok = bool((~notev_pooled[arm][s]).any())
                    cells = classify_cells(ce, clears_pooled[arm][s],
                                           e_ok, p_ok)
                    counts = verify_outcome_space(
                        cells, ce, clears_pooled[arm][s], e_ok, p_ok)
                    af = at_floor_counts(full["p"][:, kp],
                                         full["n_finite"][kp],
                                         notev[arm][kp])
                    at_floor[(arm, T)].append(af)
                    aligned[(arm, T)][s] = (cells == CELL_ALIGNED_BOTH) | \
                                           (cells == CELL_ALIGNED_EXP)
                    n_al = counts[CELL_ALIGNED_BOTH] + counts[CELL_ALIGNED_EXP]
                    rows[(arm, T)].append({
                        "seed": bh_seeds[s],
                        "threshold": T,
                        "n_concepts_retained": pool_n[T],
                        "k_min": int(floors[T]),
                        "m_tests": r["n_tests"],
                        "bh_threshold": r["threshold"],
                        "n_rejected_experimental": r["n_rejected"],
                        "at_floor_experimental": af,
                        # BP4 item 1: the flag rides on the ROW, every row,
                        # and it is recomputed against THIS point's floor.
                        "floor_limited_experimental": bool(af < floors[T]),
                        "cells": {CELL_NAMES[c]: counts[c]
                                  for c in sorted(counts)},
                        "n_ALIGNED": n_al,
                        "n_EXCLUDED": counts[CELL_EXCLUDED],
                        "n_ORPHAN": counts[CELL_ORPHAN],
                        "n_NOT_EVALUABLE": counts[CELL_NOT_EVALUABLE],
                    })
                    if T == 1:
                        ok = np.array_equal(cells, c3_cells[arm][s])
                        regression["cells_equal"][f"{arm}:{bh_seeds[s]}"] = \
                            bool(ok)
                        if not ok:
                            d = int(np.count_nonzero(cells != c3_cells[arm][s]))
                            raise GridRegressionFailure(
                                f"arm {arm.upper()} seed {bh_seeds[s]}: at "
                                f"T = 1 the cells differ from component 3's in "
                                f"{d} of {cells.size} pairs. T = 1 IS the "
                                "committed unrestricted classification.")
                    flag = " [FLOOR-LIMITED exp]" if af < floors[T] else ""
                    print(f"  arm {arm.upper()} {bh_seeds[s]:<8} T>={T:<4} "
                          f"C={pool_n[T]:<4} ALIGNED={n_al:>5} "
                          f"EXCL={counts[CELL_EXCLUDED]:>6} "
                          f"ORPH={counts[CELL_ORPHAN]:>6} "
                          f"NE={counts[CELL_NOT_EVALUABLE]:>3} "
                          f"rej={r['n_rejected']:>5} "
                          f"floor={af:>4}/{floors[T]}{flag}")

                # ---- the counterfactual, built while this seed's rejections
                # are still in hand ----------------------------------------
                for k in range(len(GRID) - 1):
                    Tk, Tn = GRID[k], GRID[k + 1]
                    kp_n = keeps[Tn]
                    mid = clears_bar(rej_by_T[Tk][:, kp_n], g_obs[:, kp_n],
                                     args.gain_floor)
                    aligned.setdefault(("mid", arm, Tk, Tn),
                                       np.zeros((n_seeds, n_f), dtype=bool))
                    aligned[("mid", arm, Tk, Tn)][s] = mid
                rej_by_T.clear()

    # ---- BI3(b)'s decomposition, per arm per step, per seed and pooled ----
    steps: dict = {}
    for arm in ("a", "b"):
        steps[arm.upper()] = []
        for k in range(len(GRID) - 1):
            Tk, Tn = GRID[k], GRID[k + 1]
            per_seed = []
            for s in range(n_seeds):
                d = decompose_aligned(aligned[(arm, Tk)][s],
                                      aligned[("mid", arm, Tk, Tn)][s],
                                      aligned[(arm, Tn)][s])
                d["seed"] = bh_seeds[s]
                per_seed.append(d)
            agg = {k2: int(sum(x[k2] for x in per_seed))
                   for k2 in ("lost", "gained", "lost_pool", "lost_threshold",
                              "gained_threshold", "gained_pool",
                              "n_aligned_prev", "n_aligned_next")}
            agg["net"] = agg["n_aligned_next"] - agg["n_aligned_prev"]
            if agg["gained"] - agg["lost"] != agg["net"]:
                raise DecompositionViolation(
                    f"aggregate two-way does not sum, arm {arm}, "
                    f"{Tk} -> {Tn}")
            if (agg["gained_threshold"] - agg["lost_threshold"]
                    - agg["lost_pool"] != agg["net"]):
                raise DecompositionViolation(
                    f"aggregate three-way does not sum, arm {arm}, "
                    f"{Tk} -> {Tn}")
            steps[arm.upper()].append({
                "from": Tk, "to": Tn, "aggregate": agg, "per_seed": per_seed})

    # ---- BI3(a)'s bound at each grid point, per arm -----------------------
    bounds: dict = {}
    for arm in ("a", "b"):
        loss_full = relative_positive_loss(surv_pos[arm], n_pos_obs)
        keep_full = surviving_draw_mask(surv_pos[arm])
        bounds[arm.upper()] = {}
        for T in GRID:
            kp = keeps[T]
            b = anti_conservative_bound(loss_full[kp], keep_full[kp])
            if b.get("status") == "COMPUTED":
                b["worst_cells"] = worst_cells(
                    loss_full[kp], keep_full[kp],
                    [c for c, k2 in zip(bh_concepts, kp) if k2], n=2)
            bounds[arm.upper()][str(T)] = b

    # ---- clearance, and the floor-limited set PER POINT (BP4, BT-C5) ------
    clear: dict = {}
    floor_limited: dict = {}
    for arm in ("a", "b"):
        clear[arm.upper()] = {}
        floor_limited[arm.upper()] = {}
        for T in GRID:
            tot = int(sum(at_floor[(arm, T)]))
            clear[arm.upper()][str(T)] = clearance(tot, int(floors[T]),
                                                   pool_n[T])
            clear[arm.upper()][str(T)]["n_seeds"] = n_seeds
            clear[arm.upper()][str(T)]["mean_at_floor_per_concept_per_seed"] = (
                tot / (pool_n[T] * n_seeds) if pool_n[T] and n_seeds else None)
            fl = [bh_seeds[s] for s in range(n_seeds)
                  if at_floor[(arm, T)][s] < floors[T]]
            floor_limited[arm.upper()][str(T)] = {
                "floor_limited_seeds": fl,
                "n_floor_limited": len(fl),
                "n_seeds": n_seeds,
                "k_min": int(floors[T]),
                "at_floor_min": int(min(at_floor[(arm, T)])),
                "at_floor_max": int(max(at_floor[(arm, T)])),
                "note": "recomputed at THIS point's floor; never carried "
                        "forward from another grid point (BT-C5)",
            }

    # ---- write ------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as out:
        out.create_dataset("thresholds", data=np.array(GRID, dtype=np.int64))
        out.create_dataset("n_concepts_retained", data=np.array(
            [pool_n[T] for T in GRID], dtype=np.int64))
        out.create_dataset("k_min", data=np.array(
            [int(floors[T]) for T in GRID], dtype=np.int64))
        out.create_dataset("seeds", data=np.array(
            bh_seeds[:n_seeds], dtype=h5py.string_dtype()))
        out.create_dataset("concepts_experimental", data=np.array(
            bh_concepts, dtype=h5py.string_dtype()))
        out.create_dataset("n_positives_in_pool", data=npos_e)
        for i, T in enumerate(GRID):
            out.create_dataset(f"retained_T{T}", data=keeps[T])
            for arm in ("a", "b"):
                out.create_dataset(f"aligned_{arm}_T{T}",
                                   data=aligned[(arm, T)])
                out.create_dataset(f"at_floor_{arm}_T{T}", data=np.array(
                    at_floor[(arm, T)], dtype=np.int64))
        out.create_dataset("cell_codes", data=np.array(
            [f"{c}={CELL_NAMES[c]}" for c in sorted(CELL_NAMES)],
            dtype=h5py.string_dtype()))
        # ---- IDENTITY: nothing back-filled (criterion 10) -----------------
        for src, pre in ((raw_identity, "null_experimental"),
                         (bh_identity, "bh_experimental"),
                         (c3_identity, "classification_2x2")):
            for k in ("library_sha", "n_concepts", "argmax_pool_n",
                      "argmax_pool_used_n", "scored_row_set", "negatives",
                      "stratum", "truncation_policy", "retention_floor_mode",
                      "retention_min_residues", "retention_min_instances",
                      "min_protein_count", "n_shifts", "null_seed",
                      "null_type", "n_rows", "ap_impl", "q_fdr",
                      "gather_device_class", "ap_arith_device_class"):
                if k in src:
                    out.attrs[f"{pre}_{k}"] = src[k]
        out.attrs["q_fdr"] = float(args.q)
        out.attrs["gain_floor"] = float(args.gain_floor)
        out.attrs["gain_floor_applied"] = True
        out.attrs["governing_amendment"] = "BT (BI1, BI2, BI3; BM-E1; BP4)"
        out.attrs["classified_unit"] = "(seed, feature) pair; BQ4a"
        out.attrs["population_per_arm"] = int(n_seeds * n_f)
        out.attrs["grid"] = json.dumps(list(GRID))
        out.attrs["restriction"] = ("n_positives >= T over the in-pool "
                                    "experimental concept axis; ALIGNED side "
                                    "only (BI2b)")
        out.attrs["pooled_axis_source"] = str(args.classification_2x2)
        out.attrs["is_cost_pilot"] = bool(args.seeds is not None)
        out.attrs["git_sha"] = git_sha()
        out.attrs["git_dirty"] = bool(git_dirty())
        for nm, v in (("raw_null_experimental", args.raw_null_experimental),
                      ("null_experimental_reductions",
                       args.null_experimental_reductions),
                      ("ap_experimental", args.ap_experimental),
                      ("bh_experimental", args.bh_experimental),
                      ("classification_2x2", args.classification_2x2)):
            out.attrs[f"input_{nm}"] = str(v)

    report = {
        "artifact": "bt_grid_composition",
        "governing_amendment": "BT",
        "git_sha": git_sha(), "git_dirty": bool(git_dirty()),
        "q": float(args.q), "gain_floor": float(args.gain_floor),
        "is_cost_pilot": bool(args.seeds is not None),
        "grid": list(GRID),
        "n_seeds": n_seeds, "n_features": n_f, "n_concepts_full": n_c,
        "n_concepts_retained": {str(T): pool_n[T] for T in GRID},
        "k_min": {str(T): int(floors[T]) for T in GRID},
        "k_min_equals_pool": {str(T): bool(floors[T] == pool_n[T])
                              for T in GRID},
        "regression_baseline_T1": regression,
        "per_arm": {arm.upper(): {str(T): rows[(arm, T)] for T in GRID}
                    for arm in ("a", "b")},
        "decomposition": steps,
        "anti_conservative_bound": bounds,
        "clearance": clear,
        "floor_limited_per_grid_point": floor_limited,
        "inputs": {
            "raw_null_experimental": str(args.raw_null_experimental),
            "null_experimental_reductions":
                str(args.null_experimental_reductions),
            "ap_experimental": str(args.ap_experimental),
            "bh_experimental": str(args.bh_experimental),
            "classification_2x2": str(args.classification_2x2),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}\nwrote {args.report}")


if __name__ == "__main__":
    main()
