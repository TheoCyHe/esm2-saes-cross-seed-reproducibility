"""
M3 -- does excluding the 22 DEGENERATE CONCEPTS change the per-stratum `mu_s`?

WHY THIS IS ASKED. The regression predictor is `M_f = max_c (g(f,c) / mu_s)`
(amendments U1 + V1), so `mu_s` -- the null MEAN pooled over the stratum -- is
now a denominator on every value rather than one input among several.
Amendment R10 left the 22 degenerate concepts FLAGGED rather than GATED, which
means they are still inside the strata that define `mu_s`. A concept whose
null is a point mass at zero contributes zeros to the mean it helps define.

WHAT THIS IS NOT. It is **not** the zero-division question. U1 decision 4
settled that: over 240 estimable strata the minimum `mu_s` is 3.358e-07, with
zero strata at or below 0 and zero below 1e-8. Pooling protects the
denominator. The question here is whether the 22 BIAS the denominator.

DEGENERACY CRITERION, restated from `diagnose_standardization.py` rather than
redefined: a concept's null is degenerate iff the **95th percentile of its
pooled null gain values over all (shift, feature) draws is exactly zero**.
That criterion was fixed before its counts were read and is not re-tuned here.

WHICH SET IS EXCLUDED, declared before the run. The criterion is evaluated PER
SEED. The primary excluded set is the concepts degenerate in the **first seed
analysed**, matching how R10's count of 22 was derived; the per-seed counts,
their **intersection** and their **union** are all reported, so the set's
stability across seeds is visible rather than assumed.

=============================================================================
PREDICTIONS -- COMMITTED TO GIT BEFORE THE JOB IS SUBMITTED
=============================================================================

  D1. In EVERY stratum containing at least one degenerate concept,
      `mu_s(excluding) >= mu_s(all)`.
      This is close to arithmetic -- removing members that are >=95% zeros
      from a positive mean should raise it -- but it is NOT guaranteed,
      because the criterion constrains only the 95th percentile and the
      remaining 5% of a degenerate concept's draws are unbounded. A single
      violation would mean some degenerate concept carries a tail heavy
      enough to sit ABOVE its stratum mean, which would be worth knowing.

  D2. The number of strata containing at least one degenerate concept is
      strictly between 1 and the total number of estimable strata -- i.e.
      the 22 are neither confined to one cell nor spread across all of them.

  D3. The MEDIAN ratio `mu_s(excluding) / mu_s(all)` across AFFECTED strata
      is below 1.5, and the MAXIMUM is below 10. That is: the effect is
      modest rather than order-of-magnitude.

  D4. **NO DIRECTIONAL PREDICTION IS MADE about whether exclusion should be
      ADOPTED.** The brief's instruction was explicit -- do not reason about
      the direction first, compute both and compare -- and gotcha 22 records
      directional mechanism claims at 0-for-6, now 0-for-8 with amendment
      Z4's transfer-claim category. The decision is the author's (R10 keeps the 22
      flagged, not gated) and this module measures rather than recommends.

  D5. Degeneracy is a property of the CONCEPT, not of the seed: the
      intersection of the per-seed degenerate sets is at least 80% of their
      union. (A comparison between two quantities both computed here.)
=============================================================================

WHAT IS DELIBERATELY NOT DONE HERE. This does not recompute `M_ratio` with the
22 excluded from the max pool. R10 keeps them IN the pool; the open question
is only whether they should leave `mu_s` ESTIMATION while remaining in the
pool. Those are separable, and conflating them would answer a question nobody
asked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.analysis.artifact_identity import argmax_pool_mask
from src.analysis.null_pooling import (
    FIRING_RATE_EDGES,
    MIN_FEATURES_PER_STRATUM,
    bin_index,
    collapse_firing_bins,
)
from src.analysis.standardize_gain import (
    AxisSpec,
    DEFAULT_CONCEPT_AXES,
    concept_cell_index,
    load_axis_values,
    moments_from_sums,
    pooled_moments,
)


def degenerate_mask(slab: np.ndarray) -> np.ndarray:
    """The stated criterion: per-concept 95th percentile over all draws == 0.

    `slab` is (shifts, concepts, features) for one seed. Flattening over
    (shift, feature) is what `diagnose_standardization.concept_stats` does,
    and it is reproduced rather than reinvented so the two cannot drift.
    """
    n_r, n_c, n_f = slab.shape
    flat = slab.transpose(1, 0, 2).reshape(n_c, n_r * n_f)
    return np.percentile(flat, 95, axis=1) == 0


def mu_with_and_without(
    slab: np.ndarray,
    cell_pool: np.ndarray,
    n_cells: int,
    fb: np.ndarray,
    n_fbins: int,
    drop: np.ndarray,
) -> dict:
    """`mu_s` over all pool concepts, and over the pool minus `drop`.

    Both go through the SAME `pooled_moments` / `moments_from_sums` path the
    predictor uses, so this measures the quantity the regression actually
    consumes rather than a re-derivation of it (gotcha 21).
    """
    s1, s2, cnt = pooled_moments(slab, cell_pool, n_cells, fb, n_fbins)
    mu_all, _ = moments_from_sums(s1, s2, cnt)

    keep = ~drop
    s1k, s2k, cntk = pooled_moments(
        slab[:, keep, :], cell_pool[keep], n_cells, fb, n_fbins)
    mu_excl, _ = moments_from_sums(s1k, s2k, cntk)

    # A stratum is AFFECTED iff it contains at least one dropped concept.
    affected = np.zeros((n_cells, n_fbins), dtype=bool)
    for cc in np.unique(cell_pool[drop]):
        if cc >= 0:
            affected[cc, :] = True
    # ... but only where the stratum is estimable at all.
    estimable = np.isfinite(mu_all) & (cnt > 0)
    affected &= estimable
    return {
        "mu_all": mu_all, "mu_excl": mu_excl,
        "affected": affected, "estimable": estimable,
        "cnt": cnt, "cnt_excl": cntk,
    }


def summarise(res: dict) -> dict:
    """Derived summary. Every verdict computed, none asserted (gotcha 20)."""
    mu_all, mu_excl = res["mu_all"], res["mu_excl"]
    aff, est = res["affected"], res["estimable"]

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = mu_excl / mu_all
    ok = est & np.isfinite(ratio)
    ra = ratio[aff & ok]
    ru = ratio[(~aff) & ok]

    # D1, computed: is mu_excl >= mu_all everywhere it is affected?
    viol = int(np.sum((mu_excl < mu_all) & aff & ok))
    return {
        "n_strata_estimable": int(est.sum()),
        "n_strata_affected": int(aff.sum()),
        "n_strata_unaffected_estimable": int(((~aff) & est).sum()),
        "mu_all_min": float(np.nanmin(mu_all[est])) if est.any() else None,
        "mu_all_median": float(np.nanmedian(mu_all[est])) if est.any() else None,
        "ratio_affected": {
            "n": int(ra.size),
            "min": float(ra.min()) if ra.size else None,
            "median": float(np.median(ra)) if ra.size else None,
            "max": float(ra.max()) if ra.size else None,
            "p90": float(np.percentile(ra, 90)) if ra.size else None,
        },
        "ratio_unaffected": {
            "n": int(ru.size),
            "min": float(ru.min()) if ru.size else None,
            "median": float(np.median(ru)) if ru.size else None,
            "max": float(ru.max()) if ru.size else None,
        },
        "D1_n_violations": viol,
        "D1_outcome": "HIT" if viol == 0 else "MISS",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ap", type=Path, required=True)
    ap.add_argument("--null-raw", type=Path, required=True)
    ap.add_argument("--null-reductions", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--labels-json", type=Path, default=None)
    ap.add_argument("--arm", default="b", choices=["a", "b"])
    ap.add_argument("--stratum", default="pooled")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 7])
    ap.add_argument("--concept-axes", default=",".join(DEFAULT_CONCEPT_AXES))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    spec = AxisSpec(tuple(a.strip() for a in args.concept_axes.split(",") if a.strip()))
    labels_json = args.labels_json or args.labels.with_suffix(".json")

    with h5py.File(args.ap, "r") as hf:
        g = hf[args.stratum]
        concepts = [c.decode() if isinstance(c, bytes) else c
                    for c in g["concepts"][:]]
        n_rows = int(hf.attrs["n_rows"])

    pool_mask = np.array(argmax_pool_mask(concepts))
    pool_names = [c for c, k in zip(concepts, pool_mask) if k]

    with h5py.File(args.null_raw, "r") as hf:
        null_concepts = [c.decode() if isinstance(c, bytes) else c
                         for c in hf["concepts"][:]]
        # I4's defect: same size is not the same object.
        if list(null_concepts) != pool_names:
            raise SystemExit(
                "FATAL: the null's concept axis is not the AP matrix's argmax "
                f"pool (null={len(null_concepts)}, pool={len(pool_names)}).")
        raw = hf[f"gain_{args.arm}"]

        with h5py.File(args.null_reductions, "r") as rf:
            fire_rate = rf["fire_rate"][:]

        fr_edges = collapse_firing_bins(fire_rate, FIRING_RATE_EDGES,
                                        MIN_FEATURES_PER_STRATUM)
        n_fbins = len(fr_edges) - 1
        vals = load_axis_values(args.labels, labels_json, concepts, n_rows)
        cell, n_cells = concept_cell_index(vals, spec)
        if (cell < 0).any():
            raise SystemExit(
                f"FATAL: {int((cell < 0).sum())} concepts outside the axis edges.")
        cell_pool = cell[np.flatnonzero(pool_mask)]
        print(f"axis {spec.concept_axes} -> {n_cells} cells x {n_fbins} "
              f"firing bins; pool {len(pool_names)}", flush=True)

        per_seed: dict[str, dict] = {}
        deg_sets: dict[int, set[str]] = {}
        primary_drop = None
        for s in args.seeds:
            slab = raw[s].astype(np.float64)
            deg = degenerate_mask(slab)
            deg_sets[s] = {pool_names[i] for i in np.flatnonzero(deg)}
            print(f"  seed {s}: {int(deg.sum())} degenerate concepts", flush=True)
            if primary_drop is None:
                primary_drop = deg          # first seed analysed = primary set
            fb = np.clip(bin_index(fire_rate[s], fr_edges), 0, n_fbins - 1)
            res = mu_with_and_without(slab, cell_pool, n_cells, fb, n_fbins,
                                      primary_drop)
            summ = summarise(res)
            summ["n_degenerate_this_seed"] = int(deg.sum())
            per_seed[str(s)] = summ
            print("   ", json.dumps({k: summ[k] for k in
                  ("n_strata_estimable", "n_strata_affected",
                   "ratio_affected", "D1_outcome")}), flush=True)

    inter = set.intersection(*deg_sets.values()) if deg_sets else set()
    union = set.union(*deg_sets.values()) if deg_sets else set()
    stability = (len(inter) / len(union)) if union else float("nan")

    n_est = per_seed[str(args.seeds[0])]["n_strata_estimable"]
    n_aff = per_seed[str(args.seeds[0])]["n_strata_affected"]
    ra = per_seed[str(args.seeds[0])]["ratio_affected"]
    scorecard = {
        "D1_mu_excl_ge_mu_all_in_affected_strata": {
            "violations_by_seed": {k: v["D1_n_violations"] for k, v in per_seed.items()},
            "outcome": "HIT" if all(v["D1_n_violations"] == 0
                                    for v in per_seed.values()) else "MISS",
        },
        "D2_affected_strata_strictly_between_1_and_all": {
            "n_affected": n_aff, "n_estimable": n_est,
            "outcome": "HIT" if 1 < n_aff < n_est else "MISS",
        },
        "D3_median_ratio_lt_1.5_and_max_lt_10": {
            "median": ra["median"], "max": ra["max"],
            "outcome": "HIT" if (ra["median"] is not None
                                 and ra["median"] < 1.5
                                 and ra["max"] is not None
                                 and ra["max"] < 10) else "MISS",
        },
        "D5_degeneracy_is_a_concept_property": {
            "n_intersection": len(inter), "n_union": len(union),
            "stability": stability,
            "outcome": "HIT" if stability >= 0.80 else "MISS",
        },
    }
    print(json.dumps({"scorecard": scorecard}, indent=2), flush=True)

    out = {
        "identity": {
            "measurement": "mu_s_degenerate_M3",
            "arm": args.arm, "stratum": args.stratum,
            "concept_axes": list(spec.concept_axes),
            "n_cells": int(n_cells), "n_fbins": int(n_fbins),
            "pool_n": len(pool_names), "seeds": list(args.seeds),
            "degeneracy_criterion": "per-concept p95 over all (shift, feature) draws == 0",
            "primary_excluded_set": "concepts degenerate in the first seed analysed",
        },
        "degenerate_sets": {str(k): sorted(v) for k, v in deg_sets.items()},
        "degenerate_intersection": sorted(inter),
        "degenerate_union": sorted(union),
        "per_seed": per_seed,
        "prediction_scorecard": scorecard,
        "decision": (
            "MEASUREMENT ONLY. R10 keeps the 22 flagged rather than gated; "
            "whether they leave mu_s estimation while remaining in the max "
            "pool is the author's call. No recommendation is made here (D4)."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[m3] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
