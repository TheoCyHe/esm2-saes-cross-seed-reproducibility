"""BB section 2 -- DOES THE PRIMARY MODEL'S FUNCTIONAL FORM ACTUALLY LIMIT IT?

=============================================================================
THE CLAIM THIS EXISTS TO TEST, AND WHY IT NEEDED TESTING RATHER THAN ASSERTING
=============================================================================
It has been claimed that the primary model "cannot represent" the concentration
of the alignment relationship inside the top decile -- the +8.64 pp span §BA12
measured. **That claim is not established, and there is a specific reason to
doubt it: a logit link already produces curvature on the probability scale.**
Near a mean `r_f` of 0.4955 the logit is close to linear, so it produces
little -- but "little" is a QUANTITY and it had never been measured.

So: take the fitted primary model, predict mean reproducibility across the SAME
percentile bins inside the top alignment decile that §BA12 measured observed
values for, and compare the spans.

**BOTH READINGS ARE STATED BEFORE THE NUMBERS** (`span_comparison_verdict`),
because a reading fitted to a number is not a reading (gotcha 20).

=============================================================================
NO NEW MODEL IS FITTED. THE EXISTING FIT IS RE-EXECUTED, AND THAT IS PROVED
=============================================================================
The regression artifacts store `b0`, `b1`, `gamma`, `lambda` and `edf` -- and
**not** the penalised spline's coefficients, its basis or its knots (BB's
sealing probe, job 21223585: the only `beta` key anywhere under `primary` is
`marginal_effect/beta`, the single coefficient of the term the marginal effect
is for). The fitted curve therefore cannot be evaluated by reading the JSON;
the fit has to be re-executed.

**Re-executing a fit is exactly how a "new model" gets introduced by accident**,
so the claim that this is the SAME fit is not asserted, it is an ACCEPTANCE
CRITERION: every per-seed `b1_z`, `se_b1_z`, `b1_raw`, `gamma_d_fs`,
`se_gamma`, `intercept`, `lambda`, `edf`, `deviance`, `loglik`, `n_features`,
`sd_logM`, `mean_logM` and `mean_r_f` must reproduce `design3_join_au.json`
**EXACTLY** -- as exact floats, not to a tolerance, because this run and the
reference are the same arithmetic on the same inputs. **If they differ at all,
the job exits non-zero and nothing is reported**, because a "fitted value" from
a fit that is not the fit of record describes nothing.

No interaction term, no alignment spline, no re-specification (§BB brief 6).

=============================================================================
TWO REFERENCES, EACH ARM FAITHFUL TO ITS OWN, AND THE PREMISE BETWEEN THEM
ASSERTED POSITIONALLY
=============================================================================
The fitted arm must reproduce the JOIN, which reads `fire_rate` from
`null_pilot_compliant.h5`. The observed arm must reproduce the PANEL, which
derives it from `feature_ap_main_compliant.h5:fire_counts / n_rows`. Each arm
is therefore built from its own reference's inputs.

That leaves one premise the whole comparison rests on -- **that the two arms
describe the SAME features** -- and it is asserted rather than assumed:
`keep`, `log M` and the outcome counts must agree ELEMENTWISE across the two
paths. **A count or a total could not see a permutation** (gotcha 27), so the
check is positional and it is FATAL.

=============================================================================
ONE BINNING FUNCTION, TWO VALUE ARRAYS
=============================================================================
`alignment_fine_panel.panel` is called TWICE -- once with `r_f`, once with the
fitted probabilities -- rather than reimplemented. `bin_by_quantile` bins on
`log M`, which is identical in both calls, so the two tables are provably
bin-for-bin the same construction. A second implementation is how two tables
that are supposed to share bins stop sharing them.

**The reused table's column is named `mean_r_f` by `bin_table`. In the fitted
arm it holds a mean FITTED PROBABILITY, and the output says so** rather than
letting one key name carry two meanings (gotcha 16's convention).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.special import expit

from src.analysis.alignment_fine_panel import (
    ANALYSIS_MIN_FIRING_ROWS,
    Z95,
    firing_bin,
    panel,
    per_seed_span,
    seed_permutation,
)
from src.analysis.corrector_scan import BIN_EDGES
from src.analysis.design3_fit import build_design, fit_design3
from src.analysis.design3_join import (
    THETA_LITERATURE,
    align_to_reference,
    load_counts_all_theta,
    resolve_gtilde_seed_order,
)

# The per-seed fields whose exact reproduction is what makes this "the existing
# fit". `converged` and `n_iter` are included: a fit that reached the same
# coefficients by a different number of steps is a different computation.
FIT_IDENTITY_FIELDS = (
    "b1_z", "se_b1_z", "b1_raw", "gamma_d_fs", "se_gamma", "intercept",
    "lambda", "edf", "deviance", "loglik", "n_features", "sd_logM",
    "mean_logM", "sd_d_fs", "mean_r_f", "n_iter", "converged",
    "lambda_at_grid_edge",
)


# ---------------------------------------------------------------------------
# The refit, and the proof that it is the incumbent
# ---------------------------------------------------------------------------

def refit_primary(y: np.ndarray, logM: np.ndarray, log_phi: np.ndarray,
                  d_fs: np.ndarray, keep: np.ndarray,
                  n_trials: int) -> tuple[np.ndarray, list[dict]]:
    """`design3_join.fit_arm`'s PRIMARY per-seed loop, keeping the fitted values.

    Deliberately a transcription of that loop rather than a call into it:
    `fit_arm` returns summaries and discards `eta`, and widening its return
    arity while its existing callers keep consuming the old one -- 8 call
    sites in `src/` (all in `design3_join.py`) and 6 in
    `tests/test_design3_join.py` as of 2026-10-04; 6 and 1 when this was
    written -- is a defect this project has already shipped (gotcha 13 extension 4, instance 3).
    The transcription's correctness is not argued -- it is the acceptance
    criterion, which compares every coefficient it produces against the stored
    fit.

    Returns `(fitted, rows)` where `fitted` is (n_seeds, n_features) with NaN
    OUTSIDE `keep` -- never 0.0, which would read as "fitted, and zero".
    """
    n_seeds, n_features = logM.shape
    fitted = np.full((n_seeds, n_features), np.nan, dtype=np.float64)
    rows: list[dict] = []
    for s in range(n_seeds):
        m = keep[s]
        ys = y[s][m].astype(np.float64)
        lm = logM[s][m]
        sd = float(lm.std())
        z = (lm - lm.mean()) / max(sd, 1e-300)
        des = build_design(z, log_phi[s][m], d_fs[s][m])
        f = fit_design3(ys, n_trials, des["X"], des["S"])
        i = des["idx_logM"]
        j = des["names"].index("d_fs")
        b1 = float(f["beta"][i])
        # The same clip `average_marginal_effect` applies, so the fitted
        # probabilities here are the ones the stored marginal effect was
        # computed from rather than a second convention.
        fitted[s][m] = expit(np.clip(f["eta"], -30.0, 30.0))
        rows.append({
            "seed": s,
            "n_features": int(m.sum()),
            "b1_z": b1,
            "se_b1_z": float(np.sqrt(f["cov"][i, i])),
            "b1_raw": b1 / max(sd, 1e-300),
            "sd_logM": sd,
            "mean_logM": float(lm.mean()),
            "sd_d_fs": float(d_fs[s][m].std()),
            "gamma_d_fs": float(f["beta"][j]),
            "se_gamma": float(np.sqrt(f["cov"][j, j])),
            "intercept": float(f["beta"][0]),
            "lambda": f["lambda"],
            "edf": f["edf"],
            "lambda_at_grid_edge": f["lambda_at_grid_edge"],
            "deviance": f["deviance"],
            "loglik": f["loglik"],
            "converged": f["converged"],
            "n_iter": f["n_iter"],
            "mean_r_f": float(ys.mean() / n_trials),
            # DERIVED here, not stored anywhere: the spline's own contribution
            # to the linear predictor, which is the thing that could not be
            # read off the artifact and is the reason a refit was needed.
            "spline_contribution_sd": float(np.std(
                des["X"][:, des["spline_slice"]] @ f["beta"][des["spline_slice"]]
            )),
        })
    return fitted, rows


def reproduces_join(rows: list[dict], reference: Path) -> dict:
    """ACCEPTANCE CRITERION A -- every coefficient against the stored fit.

    EXACT float equality. Both are the same arithmetic on the same inputs, so
    any difference at all is a changed computation, not rounding. An ABSENT key
    is reported ABSENT and counted as a problem, never as agreement (gotcha 15).
    """
    doc = json.loads(reference.read_text())
    ref_rows = doc.get("primary", {}).get("per_seed")
    if not isinstance(ref_rows, list):
        return {"reference": str(reference), "verdict": "DIFFERS",
                "problems": ["the reference carries no primary/per_seed list"],
                "comparisons": []}
    if len(ref_rows) != len(rows):
        return {"reference": str(reference), "verdict": "DIFFERS",
                "problems": [f"seed count {len(rows)} vs reference "
                             f"{len(ref_rows)}"],
                "comparisons": []}
    comparisons, problems = [], []
    for mine, ref in zip(rows, ref_rows):
        for k in FIT_IDENTITY_FIELDS:
            if k not in ref:
                problems.append(f"seed {mine['seed']}: {k} ABSENT from the "
                                f"reference (absent is not matching)")
                comparisons.append({"seed": mine["seed"], "field": k,
                                    "reference": "<ABSENT>",
                                    "this_run": mine.get(k), "same": False})
                continue
            a, b = mine.get(k), ref[k]
            same = (a == b)
            if not same:
                problems.append(f"seed {mine['seed']}: {k} {b!r} vs {a!r}")
                comparisons.append({"seed": mine["seed"], "field": k,
                                    "reference": b, "this_run": a,
                                    "same": False})
    return {
        "reference": str(reference),
        "reference_identity": doc.get("identity"),
        "n_fields_compared": len(rows) * len(FIT_IDENTITY_FIELDS),
        "n_disagreements": len(problems),
        "disagreements": comparisons,
        "problems": problems,
        "verdict": ("REPRODUCES_THE_STORED_FIT_EXACTLY" if not problems
                    else "DIFFERS"),
        "scope": "every per-seed coefficient, standard error, lambda, edf, "
                 "deviance, loglik and convergence field. What is NOT compared "
                 "is the spline block, because the reference does not store it "
                 "-- that absence is why this module exists.",
    }


# ---------------------------------------------------------------------------
# The premise: the two arms describe the same features
# ---------------------------------------------------------------------------

def cross_path_agreement(keep_join: np.ndarray, keep_panel: np.ndarray,
                         logM_join: np.ndarray, logM_panel: np.ndarray,
                         y_join: np.ndarray, y_panel: np.ndarray) -> dict:
    """ELEMENTWISE agreement between the join's inputs and the panel's.

    The fitted arm is built the join's way and the observed arm the panel's
    way, so that each reproduces its own reference. The comparison between them
    is only meaningful if both describe the SAME features -- and a shape check
    or a total cannot see a permutation (gotcha 27), so this is positional.
    """
    dk = int((keep_join != keep_panel).sum())
    dm = int((logM_join != logM_panel).sum())
    dy = int((y_join != y_panel).sum())
    problems = []
    if dk:
        problems.append(f"{dk} cells disagree on `keep`")
    if dm:
        problems.append(f"{dm} cells disagree on `log M`")
    if dy:
        problems.append(f"{dy} cells disagree on the outcome counts")
    return {
        "n_cells": int(keep_join.size),
        "n_cells_disagreeing_keep": dk,
        "n_cells_disagreeing_logM": dm,
        "n_cells_disagreeing_outcome": dy,
        "max_abs_logM_difference": float(np.abs(logM_join - logM_panel).max()),
        "problems": problems,
        "verdict": "SAME_FEATURES" if not problems else "DIFFERENT_FEATURES",
        "why_this_is_fatal": (
            "the fitted arm reproduces design3_join_au.json and the observed "
            "arm reproduces alignment_fine_panel_ba_v2.json; if the two paths "
            "do not select the same cells then the spans being compared "
            "describe different feature sets and the ratio means nothing"),
    }


# ---------------------------------------------------------------------------
# The comparison and its pre-declared reading
# ---------------------------------------------------------------------------

def paired_span(per_seed: list[dict]) -> dict:
    """The per-seed paired span and its across-seed interval, in pp.

    `alignment_fine_panel.per_seed_span`'s construction, reused: the same 20
    seeds contribute to both ends, so a paired span has the between-seed
    component of its noise cancelled.
    """
    spans = [s for s in (per_seed_span(x["bins"]) for x in per_seed)
             if s is not None]
    if not spans:
        return {"status": "UNDEFINED_NO_SEED_HAS_TWO_USABLE_BINS",
                "n_seeds": 0}
    v = np.asarray(spans, dtype=np.float64)
    sd = float(v.std(ddof=1)) if v.size > 1 else None
    se = (sd / np.sqrt(v.size)) if sd is not None else None
    mean = float(v.mean())
    return {
        "status": "COMPUTED",
        "n_seeds": int(v.size),
        "mean_span_pp": mean * 100.0,
        "se_span_pp": (None if se is None else se * 100.0),
        "ci95_span_pp": ([None, None] if se is None else
                         [(mean - Z95 * se) * 100.0,
                          (mean + Z95 * se) * 100.0]),
        "per_seed_span_pp": [float(x * 100.0) for x in v],
    }


def span_comparison_verdict(fitted: dict, observed: dict) -> dict:
    """BB section 2's reading, DECLARED HERE and derived from the data.

    TOTAL over the ratio's whole domain, and total over the CAUSES of each
    branch (gotcha 30): the branch that fires when the OBSERVED span is itself
    indistinguishable from zero is named separately, because a ratio against a
    denominator whose interval straddles zero ranks noise (gotcha 25b) and must
    not be reported as either "the model keeps up" or "the model falls short".

    THE TWO READINGS, stated before the numbers:
      * fitted span MATERIALLY BELOW observed -> the functional-form claim is
        established with a number and the writeup may make it;
      * fitted span CLOSE TO observed -> the logit curvature absorbs more than
        expected and **the claim must be weakened or dropped.** This is the
        outcome the measurement exists to catch.

    The DIFFERENCE in pp is reported alongside the ratio and is defined
    everywhere the ratio is not.
    """
    if fitted.get("status") != "COMPUTED" or observed.get("status") != "COMPUTED":
        return {"verdict": "UNDEFINED_A_SPAN_IS_NOT_COMPUTED",
                "fitted_status": fitted.get("status"),
                "observed_status": observed.get("status"),
                "reading": "the question was never asked rather than answered "
                           "in the negative"}
    f = fitted["mean_span_pp"]
    o = observed["mean_span_pp"]
    lo, hi = observed.get("ci95_span_pp", [None, None])
    obs_flat = (lo is None or hi is None or (lo <= 0.0 <= hi))
    out = {
        "fitted_span_pp": f,
        "observed_span_pp": o,
        "difference_fitted_minus_observed_pp": f - o,
        "observed_ci95_span_pp": [lo, hi],
        "fitted_ci95_span_pp": fitted.get("ci95_span_pp"),
        "bands": {
            "FITTED_SPAN_MATERIALLY_BELOW_OBSERVED": "ratio < 0.5",
            "FITTED_SPAN_BELOW_OBSERVED_BUT_NOT_BY_HALF": "0.5 <= ratio < 0.8",
            "FITTED_SPAN_CLOSE_TO_OBSERVED": "0.8 <= ratio <= 1.25",
            "FITTED_SPAN_EXCEEDS_OBSERVED": "ratio > 1.25",
        },
        "bands_declared": "PREREGISTRATION BB section 2, before the run",
    }
    if obs_flat:
        out.update({
            "ratio_fitted_over_observed": None,
            "verdict": "UNINFORMATIVE_OBSERVED_SPAN_STRADDLES_ZERO",
            "reading": (
                "the observed span's own 95% interval includes zero, so a "
                "ratio against it ranks noise (gotcha 25b). Reported as "
                "uninformative, never as agreement or as a shortfall. The "
                "DIFFERENCE in pp is still reported and is defined here."),
        })
        return out
    ratio = f / o
    if ratio < 0.5:
        v = "FITTED_SPAN_MATERIALLY_BELOW_OBSERVED"
        reading = ("the fitted model reproduces less than half the observed "
                   "rise across the top decile's percentile bins. THE "
                   "FUNCTIONAL-FORM CLAIM IS ESTABLISHED WITH A NUMBER and "
                   "the writeup may make it. NO MECHANISM SENTENCE IS "
                   "OFFERED (BB brief 6).")
    elif ratio < 0.8:
        v = "FITTED_SPAN_BELOW_OBSERVED_BUT_NOT_BY_HALF"
        reading = ("the fitted model reproduces most but not all of the "
                   "observed rise. The functional-form claim is WEAKENED: it "
                   "may be stated as a partial shortfall with this ratio "
                   "attached, and NOT as 'the model cannot represent it'.")
    elif ratio <= 1.25:
        v = "FITTED_SPAN_CLOSE_TO_OBSERVED"
        reading = ("the logit curvature absorbs the concentration, so THE "
                   "FUNCTIONAL-FORM CLAIM MUST BE DROPPED. The model does "
                   "represent the top-decile rise; whatever else limits it, "
                   "this is not it. This is the outcome the measurement "
                   "existed to catch.")
    else:
        v = "FITTED_SPAN_EXCEEDS_OBSERVED"
        reading = ("the fitted model predicts a LARGER rise than was "
                   "observed, which refutes the functional-form claim in the "
                   "opposite direction from the one anticipated. The claim "
                   "must be dropped. NO MECHANISM SENTENCE IS OFFERED.")
    out.update({"ratio_fitted_over_observed": float(ratio),
                "verdict": v, "reading": reading})
    return out


def score_bb_p1(comparison: dict) -> dict:
    """BB-P1, computed from the value beside it (gotcha 20).

    Filed **NOT SEALED** (§BB1): the fitted values are produced by a path that
    has already been executed -- `design3_join` computes `eta` on every run and
    reduces it before storage -- and AZ3.2 refuses non-persistence of a pure
    reduction as a seal. The RATIO is the object; the verdict is reported
    either way and is NOT tallied on the ledger.
    """
    r = comparison.get("ratio_fitted_over_observed")
    if r is None:
        outcome = "UNDEFINED -- " + str(comparison.get("verdict"))
    else:
        outcome = "UNDER_HALF" if r < 0.5 else "HALF_OR_ABOVE"
    return {
        "statement": "the fitted primary model's predicted span across the "
                     "top-decile percentile bins is LESS THAN HALF the "
                     "observed +8.64 pp",
        "standing": "FILED, NOT SEALED -- NOT TALLIED (BB1, on AZ3.2's rule "
                    "and BA-2's precedent)",
        "ratio_fitted_over_observed": r,
        "fitted_span_pp": comparison.get("fitted_span_pp"),
        "observed_span_pp": comparison.get("observed_span_pp"),
        "outcome": outcome,
        "note": "the RATIO is the object, not the verdict -- reported either "
                "way (BB brief section 2)",
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _attrs(hf) -> dict:
    out = {}
    for k, v in hf.attrs.items():
        if isinstance(v, bytes):
            v = v.decode()
        elif isinstance(v, np.generic):
            v = v.item()
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        out[k] = v
    return out


def arm(logM: np.ndarray, r_f: np.ndarray, fitted: np.ndarray,
        keep: np.ndarray, fraction: float, n_bins: int, label: str) -> dict:
    """One arm: the observed and fitted tables over the SAME bins, plus the
    span comparison.

    `panel` is called twice with the same `logM` and the same mask, so the
    quantile cuts are identical by construction rather than by assertion.
    """
    obs = panel(logM, r_f, keep, fraction, n_bins, f"{label}_observed")
    fit = panel(logM, fitted, keep, fraction, n_bins, f"{label}_fitted")
    obs_span = paired_span(obs["per_seed"])
    fit_span = paired_span(fit["per_seed"])
    rows = []
    for a, b in zip(obs["combined"]["bins"], fit["combined"]["bins"]):
        if a.get("status") != "COMPUTED" or b.get("status") != "COMPUTED":
            rows.append({"bin": a["bin"], "status": "EMPTY"})
            continue
        rows.append({
            "bin": a["bin"],
            "n_seeds": a["n_seeds"],
            "observed_mean_r_f": a["mean_of_per_seed_means"],
            "observed_ci95": [a["mean_of_per_seed_means"]
                              - Z95 * a["se_across_seeds"],
                              a["mean_of_per_seed_means"]
                              + Z95 * a["se_across_seeds"]]
            if a.get("se_across_seeds") is not None else [None, None],
            "fitted_mean_probability": b["mean_of_per_seed_means"],
            "residual_observed_minus_fitted_pp":
                (a["mean_of_per_seed_means"]
                 - b["mean_of_per_seed_means"]) * 100.0,
        })
    return {
        "arm": label,
        "top_fraction_of_logM": float(fraction),
        "n_bins": int(n_bins),
        "value_semantics": {
            "observed": "mean r_f (match_count / n_compare) over the features "
                        "in the bin",
            "fitted": "mean FITTED PROBABILITY from the primary model. The "
                      "column is named `mean_r_f` in the per-seed tables "
                      "because `bin_table` is REUSED unchanged so that both "
                      "arms are provably the same binning; it is NOT an "
                      "observed r_f.",
        },
        "table": rows,
        "observed_span": obs_span,
        "fitted_span": fit_span,
        "comparison": span_comparison_verdict(fit_span, obs_span),
        "observed_panel": obs,
        "fitted_panel": fit,
    }


def run(gtilde_path: Path, all_theta_path: Path, rf_counts_path: Path,
        null_reductions_path: Path, d_fs_path: Path, ap_path: Path,
        out_path: Path, join_reference: Path, panel_reference: Path,
        extra_join_references: list[Path] | None = None,
        fraction: float = 0.10, n_bins: int = 10) -> dict:
    # ---- the JOIN's path, for the fitted arm ------------------------------
    allt = load_counts_all_theta(all_theta_path)
    anchor_seeds = allt["meta"]["seed_labels"]
    anchor_labels = [str(s) for s in anchor_seeds]
    n_trials = int(allt["meta"]["n_compare_seeds"])
    y_join = allt["counts"][round(THETA_LITERATURE, 2)].astype(np.float64)

    with h5py.File(gtilde_path, "r") as hf:
        M_raw = hf["M"][:]
        gt_identity = _attrs(hf)
        gt_seeds_stored = ([int(x.decode() if isinstance(x, bytes) else x)
                            for x in hf["seeds"][:]]
                           if "seeds" in hf else None)
    n_rows = int(gt_identity["n_rows"])

    with h5py.File(null_reductions_path, "r") as hf:
        fire_rate_join = hf["fire_rate"][:]
        null_seeds = [s.decode() if isinstance(s, bytes) else str(s)
                      for s in hf["seeds"][:]]
    null_labels = [s.split("_")[-1] for s in null_seeds]

    with h5py.File(d_fs_path, "r") as hf:
        d_fs_raw = hf["d_fs_cluster_entropy"][:]
        d_seeds = [int(s) for s in hf["seeds"][:]]
        d_adopted = str(hf.attrs["adopted_definition"])
    if d_adopted != "A_cluster_entropy":
        raise SystemExit(f"FATAL: d_fs.h5 says adopted_definition="
                         f"{d_adopted!r}; AQ2 ratified A_cluster_entropy.")

    gt_order = resolve_gtilde_seed_order(gtilde_path, null_reductions_path)
    M_join = align_to_reference(gt_order["labels"], M_raw, anchor_labels)
    fire_rate_join = align_to_reference(null_labels, fire_rate_join,
                                        anchor_labels)
    d_fs = align_to_reference([str(s) for s in d_seeds], d_fs_raw,
                              anchor_labels)

    bad = ~np.isfinite(M_join) | (M_join <= 0)
    if bad.any():
        raise SystemExit(
            f"FATAL: {int(bad.sum())} of {M_join.size} cells have non-finite "
            "or non-positive M. SPECIFICATION section 4 records that set as "
            "EMPTY BY MEASUREMENT; flooring it would make the check unable to "
            "fire.")
    logM_join = np.log(M_join)
    log_phi = np.log10(np.maximum(fire_rate_join, 1e-12))
    thresh = ANALYSIS_MIN_FIRING_ROWS / float(n_rows)
    keep_join = fire_rate_join >= thresh

    # ---- the PANEL's path, for the observed arm ---------------------------
    with h5py.File(rf_counts_path, "r") as hf:
        match_count = hf["match_count"][:]
        rf_identity = _attrs(hf)
    raw = rf_identity.get("seed_labels")
    if raw is None:
        raise SystemExit(
            f"FATAL: {rf_counts_path} carries no `seed_labels` attr and no "
            "`seeds` dataset, so its seed order cannot be read from the file. "
            "Refusing to assume it -- absent is not matching (gotcha 15).")
    rf_seeds = [int(x) for x in (json.loads(raw) if isinstance(raw, str)
                                 else np.asarray(raw).ravel().tolist())]
    n_compare_panel = int(rf_identity["n_compare_seeds"])
    if [str(x) for x in rf_seeds] != anchor_labels:
        raise SystemExit(
            f"FATAL: rf_counts names seeds {rf_seeds} and the all-theta "
            f"artifact names {anchor_seeds}. The panel and the join would be "
            "in different seed orders and 18 of 20 rows would mis-pair "
            "(gotcha 29).")
    if n_compare_panel != n_trials:
        raise SystemExit(
            f"FATAL: n_compare {n_compare_panel} (rf_counts) vs {n_trials} "
            "(all-theta). Mixing them is a silent change of N.")

    with h5py.File(ap_path, "r") as hf:
        fire_counts = hf["fire_counts"][:]
        ap_seeds = [int(str(x.decode() if isinstance(x, bytes) else x
                            ).split("_")[-1]) for x in hf["seeds"][:]]
        ap_n_rows = int(_attrs(hf)["n_rows"])
    if ap_n_rows != n_rows:
        raise SystemExit(
            f"FATAL: the predictor was built on n_rows={n_rows} and the "
            f"firing counts come from an artifact with n_rows={ap_n_rows}.")
    perm_ap = seed_permutation(ap_seeds, rf_seeds, str(ap_path))
    fire_rate_panel = (fire_counts.astype(np.float64)
                       / float(ap_n_rows))[perm_ap]
    perm_gt_panel = seed_permutation(gt_seeds_stored, rf_seeds,
                                     str(gtilde_path))
    logM_panel = np.log(M_raw[perm_gt_panel])
    keep_panel = fire_rate_panel >= thresh
    r_f = match_count.astype(np.float64) / float(n_compare_panel)

    # ---- THE PREMISE, asserted positionally and FATAL ---------------------
    agree = cross_path_agreement(keep_join, keep_panel, logM_join, logM_panel,
                                 y_join, match_count.astype(np.float64))
    print(f"[fvo] cross-path agreement: {agree['verdict']}", flush=True)
    if agree["verdict"] != "SAME_FEATURES":
        print(json.dumps(agree, indent=2), flush=True)
        raise SystemExit(1)

    # ---- the refit, and ACCEPTANCE A --------------------------------------
    print(f"[fvo] refitting the primary specification on {len(anchor_seeds)} "
          f"seeds at n_compare = {n_trials}", flush=True)
    fitted, rows = refit_primary(y_join, logM_join, log_phi, d_fs, keep_join,
                                 n_trials)
    acceptance_a = reproduces_join(rows, join_reference)
    print(f"[fvo] acceptance A ({join_reference.name}): "
          f"{acceptance_a['verdict']}", flush=True)
    if acceptance_a["verdict"] != "REPRODUCES_THE_STORED_FIT_EXACTLY":
        print(json.dumps(acceptance_a, indent=2)[:8000], flush=True)
        raise SystemExit(1)
    extra_acc = []
    for p in (extra_join_references or []):
        a = reproduces_join(rows, p)
        extra_acc.append(a)
        print(f"[fvo] acceptance A (also {p.name}): {a['verdict']}", flush=True)
        if a["verdict"] != "REPRODUCES_THE_STORED_FIT_EXACTLY":
            print(json.dumps(a, indent=2)[:8000], flush=True)
            raise SystemExit(1)

    # ---- the arms ---------------------------------------------------------
    fbin = firing_bin(fire_rate_panel)
    arms = {
        "top_decile": arm(logM_panel, r_f, fitted, keep_panel, fraction,
                          n_bins, "top_decile"),
        "full_range_for_context": arm(logM_panel, r_f, fitted, keep_panel,
                                      1.0, n_bins, "full_range"),
    }

    # ACCEPTANCE B: the OBSERVED side must reproduce the panel of record.
    from src.analysis.alignment_fine_panel import reproduces_reference
    obs_arms = {"top_decile": arms["top_decile"]["observed_panel"],
                "full_range_for_context":
                    arms["full_range_for_context"]["observed_panel"]}

    by_stratum = []
    obs_by_stratum = []
    n_fbins = len(BIN_EDGES) - 1
    for b in range(n_fbins):
        sel = keep_panel & (fbin == b)
        if sel.sum() < 10 * n_bins:
            by_stratum.append({"stratum": b, "n_features": int(sel.sum()),
                               "status": "TOO_FEW_FEATURES"})
            obs_by_stratum.append({"stratum": b, "n_features": int(sel.sum()),
                                   "status": "TOO_FEW_FEATURES"})
            continue
        a = arm(logM_panel, r_f, fitted, sel, fraction, n_bins,
                f"stratum_{b}")
        by_stratum.append({
            "stratum": b,
            "firing_range": [BIN_EDGES[b], BIN_EDGES[b + 1]],
            "n_features": int(sel.sum()),
            "status": "COMPUTED",
            **a,
        })
        obs_by_stratum.append({
            "stratum": b,
            "firing_range": [BIN_EDGES[b], BIN_EDGES[b + 1]],
            "n_features": int(sel.sum()),
            "status": "COMPUTED",
            **a["observed_panel"],
        })
    acceptance_b = reproduces_reference(obs_arms, obs_by_stratum,
                                        panel_reference)
    print(f"[fvo] acceptance B ({panel_reference.name}): "
          f"{acceptance_b['verdict']}", flush=True)
    if acceptance_b["verdict"] != "REPRODUCES_REFERENCE_EXACTLY":
        print(json.dumps(acceptance_b, indent=2)[:8000], flush=True)
        raise SystemExit(1)

    from src.sae.train import git_dirty, git_sha

    out = {
        "specification": "PREREGISTRATION amendment BB section 2 (brief "
                         "section 2) -- the FITTED primary model against the "
                         "OBSERVED top-decile percentile bins of BA12",
        "exploratory": "AZ7/BA5 -- cannot become primary under any outcome, "
                       "and no mechanism sentence is written for it",
        "no_new_model": (
            "the EXISTING specification is re-executed because the spline is "
            "not stored (BB1). No interaction term, no alignment spline, no "
            "re-specification. Acceptance A is what makes that a measurement "
            "rather than an assertion."),
        "identity": {"git_sha": git_sha(), "git_dirty": git_dirty(),
                     "measurement": "fitted_vs_observed_BB2"},
        "acceptance_criteria": {
            "A_refit_reproduces_the_stored_fit": acceptance_a,
            "A_extra_references": extra_acc,
            "B_observed_reproduces_the_panel": acceptance_b,
            "premise_same_features": agree,
        },
        "n_compare_seeds": n_trials,
        "theta": THETA_LITERATURE,
        "seed_axis": {
            "anchor_order": anchor_labels,
            "gtilde_order_resolved_from_data": gt_order,
            "gtilde_stored_seed_labels": gt_seeds_stored,
            "ap_seed_labels": ap_seeds,
            "null_reductions_order": null_labels,
            "d_fs_order": [str(s) for s in d_seeds],
        },
        "analysis_floor": {
            "analysis_min_firing_rows": ANALYSIS_MIN_FIRING_ROWS,
            "firing_rate_threshold": thresh,
            "n_excluded_total": int((~keep_join).sum()),
        },
        "per_seed_fit": rows,
        "arms": arms,
        "by_firing_stratum": by_stratum,
        "prediction_BB_P1": score_bb_p1(arms["top_decile"]["comparison"]),
        "source": {
            "gtilde": str(gtilde_path), "all_theta": str(all_theta_path),
            "rf_counts": str(rf_counts_path),
            "null_reductions": str(null_reductions_path),
            "d_fs": str(d_fs_path), "ap": str(ap_path),
            "join_reference": str(join_reference),
            "panel_reference": str(panel_reference),
            "gtilde_identity": gt_identity, "rf_identity": rf_identity,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    return out


def _fmt(x, width: int = 0, dp: int = 2) -> str:
    """A TOTAL formatter. A span verdict may legitimately be UNDEFINED, and a
    formatter that cannot render a real outcome turned a completed measurement
    into a FAILED job once already (BA17 item 3)."""
    return (f"{x:+.{dp}f}".rjust(width) if isinstance(x, (int, float))
            and not isinstance(x, bool) else "n/a".rjust(width))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gtilde", type=Path, required=True)
    p.add_argument("--rf-counts-all-theta", type=Path, required=True)
    p.add_argument("--rf-counts", type=Path, required=True)
    p.add_argument("--null-reductions", type=Path, required=True)
    p.add_argument("--d-fs", type=Path, required=True)
    p.add_argument("--ap", type=Path, required=True)
    p.add_argument("--join-reference", type=Path, required=True,
                   help="design3_join_au.json -- ACCEPTANCE A. Every per-seed "
                        "coefficient must reproduce it EXACTLY or the job "
                        "exits non-zero.")
    p.add_argument("--also-join-reference", type=Path, nargs="*", default=[],
                   help="further join artifacts whose PRIMARY is claimed "
                        "identical (av, aw_nofeat). Checking them makes that "
                        "claim a measurement.")
    p.add_argument("--panel-reference", type=Path, required=True,
                   help="alignment_fine_panel_ba_v2.json -- ACCEPTANCE B.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fraction", type=float, default=0.10)
    p.add_argument("--n-bins", type=int, default=10)
    args = p.parse_args()

    out = run(args.gtilde, args.rf_counts_all_theta, args.rf_counts,
              args.null_reductions, args.d_fs, args.ap, args.out,
              args.join_reference, args.panel_reference,
              list(args.also_join_reference), args.fraction, args.n_bins)

    for name in ("top_decile", "full_range_for_context"):
        a = out["arms"][name]
        print(f"\n[BB2] {name}: observed vs fitted, by percentile bin")
        print("      bin   observed   fitted   obs-fit(pp)")
        for r in a["table"]:
            if r.get("status") == "EMPTY":
                print(f"      {r['bin']:>3}   --         --       --")
                continue
            print(f"      {r['bin']:>3}   {r['observed_mean_r_f']:.4f}     "
                  f"{r['fitted_mean_probability']:.4f}   "
                  f"{_fmt(r['residual_observed_minus_fitted_pp'], 7)}")
        c = a["comparison"]
        print(f"      observed span {_fmt(c.get('observed_span_pp'))} pp   "
              f"fitted span {_fmt(c.get('fitted_span_pp'))} pp   "
              f"ratio {c.get('ratio_fitted_over_observed')}")
        print(f"      VERDICT {c['verdict']}")

    print("\n[BB2] per firing stratum, inside the top fraction:")
    for r in out["by_firing_stratum"]:
        if r.get("status") != "COMPUTED":
            print(f"      stratum {r['stratum']}: {r.get('status')} "
                  f"(n={r.get('n_features')})")
            continue
        c = r["comparison"]
        print(f"      stratum {r['stratum']} n={r['n_features']:>7,}  "
              f"obs {_fmt(c.get('observed_span_pp'), 7)} pp  "
              f"fit {_fmt(c.get('fitted_span_pp'), 7)} pp  "
              f"ratio {c.get('ratio_fitted_over_observed')}  "
              f"{c['verdict']}")

    bp = out["prediction_BB_P1"]
    print(f"\n  BB-P1 ({bp['standing']}): {bp['outcome']}; "
          f"ratio {bp['ratio_fitted_over_observed']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
