"""AMENDMENT CO ITEM A -- the rank correlation between `z(log M)` and
`log10 phi`, per anchor, at the reference threshold.

WHAT THIS IS FOR, STATED FIRST BECAUSE IT GOVERNS WHERE THE NUMBER MAY GO.
The record asserts that the predictor and firing rate are correlated -- it is
why `log10 phi` enters the design-3 regression as a PENALISED SPLINE rather
than linearly (`docs/SPECIFICATION.md` section 1, gotcha 5) -- and it has never
attached a coefficient to that assertion. This computes one, FOR REVIEWER
DEFENCE. Under amendment CO it does NOT enter the manuscript without a separate
authorisation, and no outcome changes the qualitative sentence in either
direction. Both of those are declared at CO before the number exists.

WHAT IT IS NOT. `collinear_share` in `design3_join_ar.json` is not this
quantity: it is `joint - (dD_M + dD_phi)` in DEVIANCE units, it straddles zero
(negative in 11 anchors, positive in 9), and it answers a different question
(`docs/NUMBERS_OF_RECORD.md` section 17.1, qualifier 5).

THETA IS INERT HERE, AND THAT IS RECORDED RATHER THAN LEFT TO BE ASSUMED.
Neither `M` nor `phi` depends on the matching threshold; theta enters design 3
only through the OUTCOME `r_f`, and the analysis POPULATION (A4a's firing-rate
floor plus finite `d_fs`) is threshold-free. So "at theta = 0.70" names the fit
this correlation is reported alongside, not a filter applied to the rows.

ONE STATISTIC, FIXED IN ADVANCE. Spearman only. Reporting Pearson as well and
then quoting whichever is more convenient is selection; if a second statistic is
ever wanted it needs its own declaration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr

from src.analysis.seed_axis import (
    align_to_reference,
    normalise_seed_labels,
    stored_seed_labels,
)

# A4a's floor, in ROWS, as design3_join applies it. Kept as a named constant
# rather than a literal so the two cannot drift apart unnoticed.
ANALYSIS_MIN_FIRING_ROWS = 200

# CONTROL TARGETS, quoted from the record so the loader is checked against a
# committed value rather than against itself (gotcha 13).
#   SPECIFICATION section 4 / AL9a / AL12: `M` in [1.5468e-05, 1.0460e+06],
#   with ZERO non-finite or non-positive cells across all 20 anchors.
M_MIN_RECORDED = 1.5468e-05
M_MAX_RECORDED = 1.0460e06
#   NUMBERS_OF_RECORD section 17.1 qualifier 3: `total_excluded = 0`.
EXPECTED_TOTAL_EXCLUDED = 0


def z_within_seed(x: np.ndarray) -> np.ndarray:
    """Within-seed z-score, the form the design-3 coefficient is reported on.

    The SD is the population SD (`ddof=0`), which is what a z-score means here;
    the choice is immaterial to a RANK correlation and is stated so the
    invariance control below is testing wiring rather than convention.
    """
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    if not np.all(sd > 0):
        raise SystemExit(
            "FATAL: an anchor's log M has zero spread, so z-scoring is "
            "undefined for it. A constant predictor is a CHANGED INPUT, not a "
            "case to absorb -- and a rank correlation against a constant ranks "
            "nothing (gotcha 25b).")
    return (x - mu) / sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtilde", type=Path, required=True,
                    help="hard-bin gtilde -- the predictor of record, carrying M")
    ap.add_argument("--null-reductions", type=Path, required=True,
                    help="null_pilot_compliant.h5 -- carries fire_rate")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-permutations", type=int, default=20,
                    help="permuted-null replicates per anchor (control C-A4)")
    args = ap.parse_args()

    # ---- load, and ALIGN BY LABEL, never by position (gotcha 29) -----------
    with h5py.File(args.gtilde, "r") as hf:
        M = hf["M"][:]
        n_rows = int(hf.attrs["n_rows"])
    with h5py.File(args.null_reductions, "r") as hf:
        fire_rate = hf["fire_rate"][:]

    gt_labels = stored_seed_labels(args.gtilde)
    nr_labels = stored_seed_labels(args.null_reductions)
    if gt_labels is None or nr_labels is None:
        raise SystemExit(
            f"FATAL: seed labels ABSENT -- gtilde {gt_labels is not None}, "
            f"null_reductions {nr_labels is not None}. An absent label set is "
            "absent, never agreement (gotcha 15), and a positional join here "
            "mis-pairs 18 of 20 rows (gotcha 29).")

    anchors = [str(s) for s in range(M.shape[0])]
    M = align_to_reference(gt_labels, M, anchors)
    fire_rate = align_to_reference(nr_labels, fire_rate, anchors)

    # THE STORED gtilde LABELS WERE THEMSELVES RECOVERED FROM DATA (its
    # `seed_label_provenance` attr). Re-run that recovery and require it to
    # AGREE with what the file says -- a stored label set that disagrees with
    # the data it was derived from is exactly the failure this checks for.
    from src.analysis.seed_axis import resolve_unlabelled_seed_order
    recovered = resolve_unlabelled_seed_order(args.gtilde, args.null_reductions)
    if normalise_seed_labels(recovered["labels"]) != normalise_seed_labels(gt_labels):
        raise SystemExit(
            f"FATAL: gtilde's STORED seed labels {gt_labels} disagree with the "
            f"order recovered from the data {recovered['labels']}.")

    # ---- controls on the INPUTS, before anything is computed --------------
    controls = {}
    bad = ~np.isfinite(M) | (M <= 0)
    controls["C_A1_M_range"] = {
        "n_nonfinite_or_nonpositive": int(bad.sum()),
        "min": float(M.min()), "max": float(M.max()),
        "recorded_min": M_MIN_RECORDED, "recorded_max": M_MAX_RECORDED,
        "min_matches_record": bool(
            abs(M.min() - M_MIN_RECORDED) <= 0.5e-4 * M_MIN_RECORDED),
        "max_matches_record": bool(
            abs(M.max() - M_MAX_RECORDED) <= 0.5e-4 * M_MAX_RECORDED),
    }
    if bad.any():
        raise SystemExit(f"FATAL: {int(bad.sum())} cells of M are non-finite "
                         "or non-positive; AL9a/AL12 measured that set EMPTY.")
    if not (controls["C_A1_M_range"]["min_matches_record"]
            and controls["C_A1_M_range"]["max_matches_record"]):
        raise SystemExit(
            f"FATAL: M ranges [{M.min():.6g}, {M.max():.6g}] against the "
            f"recorded [{M_MIN_RECORDED:.6g}, {M_MAX_RECORDED:.6g}]. The "
            "loader is reading something other than the predictor of record.")

    thresh = ANALYSIS_MIN_FIRING_ROWS / float(n_rows)
    keep = fire_rate >= thresh
    controls["C_A2_a4a_floor"] = {
        "firing_rate_threshold": float(thresh),
        "total_excluded": int((~keep).sum()),
        "expected_total_excluded": EXPECTED_TOTAL_EXCLUDED,
        "matches_record": bool(int((~keep).sum()) == EXPECTED_TOTAL_EXCLUDED),
        "per_anchor_excluded": [int(x) for x in (~keep).sum(axis=1)],
    }

    # ---- the statistic ----------------------------------------------------
    log_M = np.log(M)
    zlog_M = z_within_seed(log_M)
    log_phi = np.log10(np.maximum(fire_rate, 1e-12))

    rng = np.random.default_rng(0)
    rows, invariance, permuted = [], [], []
    for i, s in enumerate(anchors):
        k = keep[i]
        x, y = zlog_M[i][k], log_phi[i][k]
        rho = float(spearmanr(x, y).statistic)
        rows.append({
            "anchor": s, "n_cells": int(k.sum()), "spearman_rho": rho,
            "n_distinct_log10_phi": int(np.unique(y).size),
            "sd_log_M": float(log_M[i][k].std(ddof=1)),
        })
        # C-A3: rank correlation is invariant under the within-seed z-score,
        # because it is a strictly increasing affine map. EXACT, not
        # approximate. A broken z (a negative scale, a NaN, an axis mix-up)
        # breaks it.
        invariance.append(float(spearmanr(log_M[i][k], y).statistic))
        # C-A4: the instrument must be able to return "no correlation". A
        # within-anchor permutation of the predictor destroys the association
        # and nothing else; if this does not collapse toward zero, the
        # measured value is not evidence of anything (gotcha 28).
        permuted.append([
            float(spearmanr(rng.permutation(x), y).statistic)
            for _ in range(args.n_permutations)])

    controls["C_A3_rank_invariant_under_z"] = {
        "max_abs_difference": float(np.max(np.abs(
            np.array(invariance) - np.array([r["spearman_rho"] for r in rows])))),
        "exact_on_all_anchors": bool(np.all(
            np.array(invariance) == np.array([r["spearman_rho"] for r in rows]))),
    }
    perm = np.array(permuted)
    controls["C_A4_permuted_null"] = {
        "n_permutations_per_anchor": int(args.n_permutations),
        "max_abs_rho": float(np.abs(perm).max()),
        "median_abs_rho": float(np.median(np.abs(perm))),
    }

    vals = np.array([r["spearman_rho"] for r in rows])
    out = {
        "measurement": "spearman_z_logM_vs_log10_phi_per_anchor_CO",
        "governed_by": "PREREGISTRATION amendment CO item A",
        "statistic": "Spearman rank correlation, ONE statistic, declared "
                     "before the run; Pearson is deliberately not computed",
        "population": "the design-3 analysis set -- A4a's firing-rate floor "
                      "applied per anchor; theta is INERT for both variables",
        "reference_theta_of_the_fit_this_accompanies": 0.70,
        "n_anchors": len(anchors),
        "device_class": "CPU",
        "per_anchor": rows,
        "across_anchors": {
            "min": float(vals.min()),
            "median": float(np.median(vals)),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)),
            "n_positive": int((vals > 0).sum()),
            "n_negative": int((vals < 0).sum()),
            "n_zero": int((vals == 0).sum()),
            "sign_is_uniform": bool((vals > 0).all() or (vals < 0).all()),
        },
        "sd_log_M_across_anchors": {
            "min": float(min(r["sd_log_M"] for r in rows)),
            "median": float(np.median([r["sd_log_M"] for r in rows])),
            "max": float(max(r["sd_log_M"] for r in rows)),
        },
        "controls": controls,
        "sources": {"gtilde": str(args.gtilde),
                    "null_reductions": str(args.null_reductions)},
        "seed_alignment": {
            "anchor_order": anchors,
            "gtilde_stored_labels": normalise_seed_labels(gt_labels),
            "null_reductions_stored_labels": normalise_seed_labels(nr_labels),
            "gtilde_order_recovered_from_data": recovered["labels"],
            "recovery_min_margin_observed": recovered["min_margin_observed"],
        },
    }

    from src.sae.train import git_dirty, git_sha
    out["git_sha"], out["git_dirty"] = git_sha(), git_dirty()

    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_anchor"},
                     indent=2))
    print(f"\n[CO-A] Spearman(z(log M), log10 phi) across {len(anchors)} "
          f"anchors: min {vals.min():.4f}, median {np.median(vals):.4f}, "
          f"max {vals.max():.4f}")


if __name__ == "__main__":
    main()
