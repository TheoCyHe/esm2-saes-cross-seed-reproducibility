"""BC section 2 -- BOUNDING PERPLEXITY'S CONTRIBUTION TO THE CARRIER EXCESS.

=============================================================================
WHY THIS RUNS DESPITE BB's STOPPING RULE -- A REPAIR, NOT AN EXCEPTION
=============================================================================
BB4.3's stopping rule stands. The override is authorised because **BB's brief
was internally asymmetric, and that is a drafting fact rather than a result**:
it named the candidate as "the feature-side pair -- activation perplexity AND
concentration", its enrichment half measured BOTH (BB9.1), and **its matching
half specified concentration only.** So perplexity's enrichment was measured as
instructed while its matched ambient was never requested. That is an omission
in a work order, not a candidate found by looking at results -- which is the
distinction the stopping rule exists to police.

**The override is BOUNDED: perplexity only, and no further attribution attempts
after it, whatever it returns** (PREREGISTRATION BC4.1).

=============================================================================
THE STATE OF PLAY
=============================================================================
The carrier excess over a firing-rate-matched ambient is +7.2219 pp and
UNATTRIBUTED.

  * homology support  -- carriers at the 50.52nd percentile, so the
    stratification had nothing to remove: ruled out WEAKLY (BA10);
  * concentration     -- carriers at the 81.02nd percentile and the stored
    coefficient NEGATIVE in 17 of 20 seeds, so the two halves DISAGREE in
    sign; the excess survived matching anyway: ruled out INFORMATIVELY (BB9);
  * **perplexity**    -- carriers at the 20.96th percentile (DEPLETED) and the
    stored coefficient negative in **20 of 20** seeds. **The only covariate for
    which both halves point the same way.**

=============================================================================
THE ARITHMETIC, AND WHY EACH CHOICE IS NOT A DEGREE OF FREEDOM
=============================================================================
  contribution = (average marginal effect per unit of perplexity)
                 x (carriers' perplexity displacement from their
                    firing-rate-matched peers, in the SAME units)

1. **THE AME IS `average_marginal_effect`'s OWN ARITHMETIC, NOT AN ANALOGUE.**
   That function returns `mean_f(beta * p_f(1-p_f))`; for a SCALAR beta that is
   `beta * mean_f(p_f(1-p_f))`, and `mean_p_1_minus_p` is stored per seed. The
   identity is turned into ACCEPTANCE C -- the same reconstruction applied to
   the stored `z_logM` term must reproduce that term's stored `ame`.

2. **THE DISPLACEMENT IS MEASURED, NOT CONVERTED FROM A PERCENTILE.** The brief
   forbids turning the 20.96th percentile into standard deviations under a
   normality assumption, and it is right to: nothing about that distribution is
   normal.

3. **THE PRODUCT IS FORMED ON THE RAW SCALE, so the SD choice cannot move it.**
   Per-SD coefficient x displacement-in-SD-units multiplies and divides by the
   same constant: `(ame * sd) * (delta / sd) == ame * delta` identically. The
   per-SD pair is reported so the two halves are separately readable, and the
   identity residual is REPORTED rather than assumed.

4. **THE FIXED-EFFECT INTERVAL IS UNCOMPUTABLE AND IS REPORTED AS SUCH.**
   `fit_arm` stores no standard error for `extra_coefs`, and refitting is not
   authorised (BC9). Only `meta_analyse`'s ACROSS-SEED arm is available, and
   inventing a weight would be worse than saying so.

=============================================================================
THE RESULT IS A BOUND, NOT A POINT ESTIMATE, AND THE MODULE SAYS SO
=============================================================================
Two caveats travel with every number below and are carried in the artifact:

  * the coefficient comes from the **feature-side SENSITIVITY arm's fit**, which
    is the correct fit -- it is the only one containing perplexity -- but it is
    a DIFFERENT fit from the primary, it is DEMOTED, and its own `b1_z` is
    REVERSED in sign relative to the primary's (AS11, AT, AU14);
  * the conversion crosses from a logit coefficient in a penalised-spline model
    to a match-rate DIFFERENCE, via a first-order linear approximation.

=============================================================================
THIS JOB CROSSES THE SEED-ORDER SPLIT (gotcha 29)
=============================================================================
`design3_join_au.json`'s feature-side rows are in NUMERIC anchor order; the
carrier arm's covariates are in the AP artifact's LEXICOGRAPHIC order. The join
is BY LABEL, and the assertion is POSITIONAL: the recovered permutation must be
a bijection over exactly {0..19} **and must NOT be the identity** -- 18 of 20
positions must differ, because positions 0 and 1 are the only two where the two
orders coincide. A check that passed under the identity permutation would not
be a check.

EXPLORATORY (AZ7 / BA5 / BC3): cannot become primary under any outcome, and no
mechanism sentence is written for any of it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from src.analysis.carrier_identity import (
    THETA_REFERENCE,
    carriers_from_ap,
    firing_bin,
    mnn_partner_matrix,
    order_decoders_to,
    seed_label_to_int,
    verify_counts_bit_identical,
)
from src.analysis.carrier_support import (
    align_to_ap_order,
    carrier_feature_indicator,
    carrier_rates_by_cell,
    cell_index,
    excess_comparison,
    reproduces_ay,
    support_bin_within,
    support_distribution,
)
from src.analysis.carrier_concentration import (
    build_feature_covariates,
    cell_usability,
    reproduces_ba9,
)
from src.analysis.concept_repro import (
    REFERENCE_PERCENTILE,
    assert_labels_align,
    detection_matrix,
    null_thresholds,
    observed_best_ap,
    to_gain,
)
from src.analysis.corrector_scan import BIN_EDGES
from src.analysis.rf_marginal import load_decoder

COVARIATE = "perplexity"
N_PERPLEXITY_BINS = 3

# BC4.4's cuts, in PERCENTAGE POINTS. The brief's parenthetical "under ~20% of
# the excess" is a gloss -- 1.5 / 7.2219 = 20.8% -- and the pp cut is the
# operative one. The percentage is reported everywhere regardless.
CANNOT_ACCOUNT_BELOW_PP = 1.5
LEADING_CANDIDATE_ABOVE_PP = 4.0

# AZ-P1's cut, reused rather than invented (BC5): "the excess shrinks
# materially" means below HALF the singly-matched excess.
SHRINKS_MATERIALLY_BELOW_FRACTION = 0.5

# BB9.3's PUBLISHED summary of the same stored block, to five decimals. Read
# back as ACCEPTANCE F: if the read lands on a different block or a different
# artifact, these move.
BB93_MEAN = -0.19436
BB93_SD = 0.04437
BB93_N_POSITIVE = 0

# ACCEPTANCE C's tolerance. Exact equality is NOT achievable here and the
# reason is arithmetic rather than a defect: the stored `ame` is
# `np.mean(beta * w)` (pairwise summation over 10,240 terms) while the
# reconstruction is `beta * np.mean(w)`. Those agree mathematically and need
# not agree bit-for-bit. The criterion is stated at the tightest tolerance the
# arithmetic admits, and the MEASURED residual is reported beside the verdict
# so a reader never has to take the tolerance on trust.
AME_RECONSTRUCTION_RTOL = 1e-12


# ---------------------------------------------------------------------------
# Reading the stored fit
# ---------------------------------------------------------------------------

def read_feature_side_arm(au: dict, covariate: str = COVARIATE) -> dict:
    """The per-seed coefficient, `mean(p(1-p))`, and the anchor LABELS.

    Reads, never recomputes. `label` comes from the arm's own stored
    provenance block rather than from the row's position, because the row's
    position is exactly what the seed-order split makes unsafe (gotcha 29).
    A row whose position and provenance label disagree is REPORTED, not
    silently preferred either way.
    """
    arm = au["sensitivity"]["feature_side"]
    prov = {r["anchor_position"]: str(r["anchor_label"])
            for r in arm["seed_axis_provenance_AT_A1_item_3"]["per_anchor"]}
    rows, problems = [], []
    for r in arm["per_seed"]:
        pos = int(r["seed"])
        if covariate not in r.get("extra_coefs", {}):
            problems.append(f"row {pos} carries no `{covariate}` coefficient")
            continue
        if pos not in prov:
            problems.append(f"row {pos} has no provenance record")
            continue
        rows.append({
            "position": pos,
            "label": prov[pos],
            "beta": float(r["extra_coefs"][covariate]),
            "mean_p_1_minus_p": float(r["marginal_effect"]["mean_p_1_minus_p"]),
            "stored_ame_z_logM": float(r["marginal_effect"]["ame"]),
            "beta_z_logM": float(r["marginal_effect"]["beta"]),
            "n_features": int(r["n_features"]),
        })
    return {
        "covariate": covariate,
        "n_rows": len(rows),
        "rows": rows,
        "problems": problems,
        "arm_note": arm["arm"],
        "entry_note": arm["entry"],
        "provenance_verdict":
            arm["seed_axis_provenance_AT_A1_item_3"]["verdict"],
    }


def acceptance_ame_reconstruction(rows: list[dict],
                                  rtol: float = AME_RECONSTRUCTION_RTOL
                                  ) -> dict:
    """ACCEPTANCE C -- `beta * mean(p(1-p))` reproduces the stored `ame`.

    The reconstruction below is applied to `perplexity`; this applies the SAME
    reconstruction to the term whose `ame` IS stored, so the arithmetic is
    checked against the artifact rather than against my reading of a docstring.
    Reports the measured residual whatever the verdict, and both the exact and
    the tolerance outcome, so the tolerance is visible rather than implicit.
    """
    if not rows:
        return {"verdict": "NOT_COMPUTED_NO_ROWS", "n_rows": 0}
    resid, n_exact = [], 0
    for r in rows:
        recon = r["beta_z_logM"] * r["mean_p_1_minus_p"]
        stored = r["stored_ame_z_logM"]
        if recon == stored:
            n_exact += 1
        denom = abs(stored) if stored != 0.0 else 1.0
        resid.append(abs(recon - stored) / denom)
    worst = float(max(resid))
    return {
        "n_rows": len(rows),
        "n_bit_identical": int(n_exact),
        "max_relative_residual": worst,
        "rtol": float(rtol),
        "identity_checked": "beta * mean(p(1-p)) == mean(beta * p(1-p))",
        # DERIVED (gotcha 20).
        "verdict": ("REPRODUCES_THE_STORED_AME" if worst <= rtol
                    else "DIFFERS_FROM_THE_STORED_AME"),
    }


def acceptance_bb93_summary(rows: list[dict], mean_ref: float = BB93_MEAN,
                            sd_ref: float = BB93_SD,
                            n_pos_ref: int = BB93_N_POSITIVE) -> dict:
    """ACCEPTANCE F -- the read re-derives BB9.3's published summary.

    The references are injectable so a fixture can exhibit a MISMATCH: the real
    block matches by construction, and a check that can only ever be handed
    matching inputs cannot be shown to fail (gotcha 16, and BB12 item 2's
    `tolerance_coverage` lesson applied before rather than after).
    """
    if not rows:
        return {"verdict": "NOT_COMPUTED_NO_ROWS", "n_rows": 0}
    b = np.array([r["beta"] for r in rows], dtype=np.float64)
    mean = float(b.mean())
    sd = float(b.std(ddof=1)) if b.size > 1 else float("nan")
    n_pos = int((b > 0).sum())
    ok = (round(mean, 5) == round(mean_ref, 5)
          and round(sd, 5) == round(sd_ref, 5)
          and n_pos == n_pos_ref)
    return {
        "n_rows": len(rows),
        "mean": mean, "sd": sd, "n_positive": n_pos,
        "reference_mean": float(mean_ref), "reference_sd": float(sd_ref),
        "reference_n_positive": int(n_pos_ref),
        "compared_to_decimals": 5,
        "verdict": "REPRODUCES_BB93" if ok else "DIFFERS_FROM_BB93",
    }


# ---------------------------------------------------------------------------
# ACCEPTANCE D -- the sign convention, VERIFIED rather than read
# ---------------------------------------------------------------------------

def sign_convention_probe(n: int = 800, n_trials: int = 39,
                          slope: float = 1.2, seed: int = 20260804) -> dict:
    """Does a POSITIVE extra coefficient mean "higher covariate, higher r_f"?

    BC4.4's fourth cell rests on the claim that carriers' DEPLETION and a
    NEGATIVE coefficient point the same way. That claim is a planning-thread
    reading of a report and the brief requires the convention be verified in
    code before the product is formed.

    **ONE DATASET IS GENERATED AND FITTED TWICE: once with the covariate as
    drawn, once with its SIGN FLIPPED. Both outcomes are required.** A probe
    that returned POSITIVE for any input would establish nothing (gotcha 25:
    ask which outcomes the instrument is CAPABLE of producing, and confirm the
    answer is more than one). Flipping the covariate WITHOUT regenerating `y`
    is what makes the control a control -- flipping both would restore the same
    relationship and the second arm would come back positive too, establishing
    nothing while looking like a check.
    """
    from src.analysis.design3_fit import build_design, fit_design3
    from scipy.special import expit

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    log_phi = rng.uniform(-4.0, -1.0, n)
    d_fs = rng.uniform(0.0, 3.0, n)
    x = rng.standard_normal(n)
    y = rng.binomial(n_trials, expit(slope * x)).astype(np.float64)

    def recover(xx: np.ndarray) -> float:
        des = build_design(z, log_phi, d_fs, extra={"probe": xx})
        f = fit_design3(y, n_trials, des["X"], des["S"])
        return float(f["beta"][des["names"].index("probe")])

    up = recover(x)
    down = recover(-x)
    return {
        "true_slope": float(slope), "n": int(n), "n_trials": int(n_trials),
        **probe_verdict(up, down),
    }


def probe_verdict(up: float, down: float) -> dict:
    """The REQUIREMENT, separated from the fit that supplies its two numbers.

    **Handed a real dataset the probe's two arms always behave**, so a test
    that could only call `sign_convention_probe` end to end can never exhibit a
    control arm coming back the WRONG way -- and a mutation dropping the
    control from the conjunction survives it. That is gotcha 16's fixture
    lesson: extracting the summary into a function that can be handed a series
    with a KNOWN answer is what makes its test able to fail.
    """
    ok = up > 0.0 and down < 0.0
    return {
        "recovered_positive_arm": float(up),
        "recovered_flipped_control_arm": float(down),
        "claim": ("`build_design` appends each extra covariate as a RAW column "
                  "of the linear predictor of logit(p), so a POSITIVE "
                  "coefficient means higher covariate -> higher r_f"),
        "control_note": ("the flipped arm must come back NEGATIVE; without it "
                         "the probe could only ever say POSITIVE"),
        # DERIVED (gotcha 20).
        "verdict": ("SIGN_CONVENTION_CONFIRMED" if ok else
                    "SIGN_CONVENTION_NOT_CONFIRMED"),
    }


# ---------------------------------------------------------------------------
# The seed-order join
# ---------------------------------------------------------------------------

def join_rows_to_ap_order(rows: list[dict], ap_seeds: list[str]) -> dict:
    """Map each AP (LEXICOGRAPHIC) row to the stored fit's row for that LABEL.

    Returns `perm` with `perm[i]` = the index into `rows` whose label is the
    seed named at `ap_seeds[i]`.

    **THE ASSERTION IS POSITIONAL AND IT REQUIRES THE PERMUTATION TO BE
    NON-TRIVIAL.** A bijection check alone passes under the identity, and the
    identity is exactly the mis-pairing gotcha 29 describes: the two orders
    coincide at positions 0 and 1 and nowhere else, so a correct join must move
    18 of 20 rows. `n_positions_moved` is reported whatever the verdict.
    """
    by_label = {}
    problems = []
    for i, r in enumerate(rows):
        if r["label"] in by_label:
            problems.append(f"duplicate label {r['label']} in the stored fit")
        by_label[r["label"]] = i
    ap_labels = [str(seed_label_to_int(s)) if str(s).startswith("seed_")
                 else str(s) for s in ap_seeds]
    perm = []
    for lab in ap_labels:
        if lab not in by_label:
            problems.append(f"AP seed label {lab} has no row in the stored fit")
            perm.append(-1)
        else:
            perm.append(by_label[lab])
    if len(set(p for p in perm if p >= 0)) != len([p for p in perm if p >= 0]):
        problems.append("the recovered permutation is not injective")
    moved = sum(1 for i, p in enumerate(perm) if p != i)
    if not problems and moved == 0:
        problems.append(
            "the recovered permutation is the IDENTITY. The AP artifact is "
            "LEXICOGRAPHIC and the stored fit is NUMERIC, so they coincide at "
            "positions 0 and 1 only; an identity permutation means one side "
            "was not read in the order it is stored in (gotcha 29)")
    return {
        "ap_seed_labels": ap_labels,
        "stored_fit_labels": [r["label"] for r in rows],
        "permutation_ap_row_to_stored_row": perm,
        "n_positions_moved": int(moved),
        "n_positions_total": len(perm),
        "problems": problems,
        # DERIVED (gotcha 20).
        "verdict": "JOINED_BY_LABEL" if not problems else "AMBIGUOUS",
    }


# ---------------------------------------------------------------------------
# The displacement
# ---------------------------------------------------------------------------

def covariate_displacement(carrier: np.ndarray, detected: np.ndarray,
                           value: np.ndarray, fbin: np.ndarray, n_fbins: int,
                           value_name: str = COVARIATE,
                           pair_weight: np.ndarray | None = None) -> dict:
    """Carriers' covariate displacement from their FIRING-MATCHED peers.

    Per carrier cell: `value(carrier) - mean(value of the NON-CARRIER peers in
    that carrier's own (seed, firing bin))`, in the covariate's OWN units.

    **THE PEER SET IS `carrier_feature_indicator`'s COMPLEMENT**, reused rather
    than reimplemented, so this and BB9.1's percentile arm describe the same
    two populations. Without it a carrier of some other concept would sit in
    the pool it is being compared against.

    **THE PRIMARY REDUCTION IS THE MEAN** (BC4.3): an average marginal effect
    is a first-order linear approximation and its natural argument is a mean
    shift. The MEDIAN and, where `pair_weight` is supplied, the PAIR-WEIGHTED
    mean are reported beside it and neither takes any branch.

    `sd_reference` is the carrier-weighted mean of the per-cell peer SD. It
    exists ONLY so the per-SD presentation can use ONE constant per seed, which
    is what makes `(ame * sd) * (delta / sd) == ame * delta` an identity rather
    than an approximation. It cannot move the product.
    """
    n_seeds = value.shape[0]
    is_carrier = carrier_feature_indicator(carrier, detected, value.shape[1])

    per_seed: list[dict] = []
    strata: dict[int, list[float]] = {b: [] for b in range(n_fbins)}
    unusable = 0
    for s in range(n_seeds):
        det = np.flatnonzero(detected[s])
        idx = carrier[s][det]
        d_all, dsd_all, w_all, sd_all = [], [], [], []
        for b in range(n_fbins):
            in_bin = fbin[s] == b
            pool = value[s][in_bin & ~is_carrier[s]]
            sel_mask = (fbin[s][idx] == b) if idx.size else np.zeros(0, bool)
            sel = idx[sel_mask]
            if pool.size == 0:
                unusable += int(sel.size)
                continue
            if sel.size == 0:
                continue
            peer_mean = float(pool.mean())
            peer_sd = float(pool.std(ddof=1)) if pool.size > 1 else float("nan")
            d = value[s][sel] - peer_mean
            d_all.append(d)
            sd_all.append(np.full(d.size, peer_sd))
            dsd_all.append(d / peer_sd if np.isfinite(peer_sd) and peer_sd > 0
                           else np.full(d.size, np.nan))
            strata[b].extend(d.tolist())
            if pair_weight is not None:
                w_all.append(pair_weight[s][det[sel_mask]].astype(np.float64))
        if not d_all:
            per_seed.append({"seed_row": s, "status": "NO_USABLE_CELLS"})
            continue
        d = np.concatenate(d_all)
        sd = np.concatenate(sd_all)
        dsd = np.concatenate(dsd_all)
        row = {
            "seed_row": s,
            "status": "COMPUTED",
            "n_carrier_cells": int(d.size),
            "mean_displacement": float(d.mean()),
            "median_displacement": float(np.median(d)),
            "sd_reference": float(np.nanmean(sd)),
            "mean_displacement_in_peer_sd_per_cell": float(np.nanmean(dsd)),
        }
        if pair_weight is not None and w_all:
            w = np.concatenate(w_all)
            row["pair_weighted_mean_displacement"] = (
                float((d * w).sum() / w.sum()) if w.sum() > 0 else float("nan"))
            row["total_pair_weight"] = float(w.sum())
        per_seed.append(row)

    ok = [r for r in per_seed if r.get("status") == "COMPUTED"]
    if not ok:
        return {"status": "NO_USABLE_CARRIER_CELLS", "covariate": value_name,
                "n_carrier_cells_without_a_peer_pool": int(unusable)}
    return {
        "status": "COMPUTED",
        "covariate": value_name,
        "statistic": (f"carrier {value_name} minus the MEAN {value_name} of "
                      "the NON-CARRIER features in the carrier's own (seed, "
                      "firing bin), in the covariate's own units"),
        "population": "one row per DETECTED (seed, concept) cell -- BB9.1's",
        "primary_reduction": "the MEAN over carrier cells, per seed (BC4.3)",
        "n_seeds_usable": len(ok),
        "n_carrier_cells": int(sum(r["n_carrier_cells"] for r in ok)),
        "n_carrier_cells_without_a_peer_pool": int(unusable),
        "per_seed": per_seed,
        "across_seed_mean_of_mean": float(
            np.mean([r["mean_displacement"] for r in ok])),
        "across_seed_mean_of_median": float(
            np.mean([r["median_displacement"] for r in ok])),
        "across_seed_mean_sd_reference": float(
            np.mean([r["sd_reference"] for r in ok])),
        "by_firing_stratum": [
            {"stratum": b,
             "firing_range": [BIN_EDGES[b], BIN_EDGES[b + 1]],
             "n_carrier_cells": len(strata[b]),
             "mean_displacement": (float(np.mean(strata[b])) if strata[b]
                                   else None),
             "median_displacement": (float(np.median(strata[b])) if strata[b]
                                     else None)}
            for b in range(n_fbins)],
    }


def pair_weights_from_detection(detected: np.ndarray) -> np.ndarray:
    """`w[s, c]` = the number of seeds `t > s` that also DETECT concept `c`.

    This is the weight `carrier_rates_by_cell` gives cell `(s, c)`: its loop is
    `for s, for t in s+1..` over concepts BOTH detect, and it evaluates seed
    `s`'s carrier. Supplied so the displacement can also be reduced the way the
    excess itself is reduced -- reported BESIDE the primary equal-cell mean and
    never in place of it, because it is an approximation of that weighting (it
    does not model the `n_in > 1` usability guard).
    """
    det = np.asarray(detected, dtype=bool)
    n_seeds = det.shape[0]
    out = np.zeros(det.shape, dtype=np.float64)
    for s in range(n_seeds):
        if s + 1 < n_seeds:
            out[s] = det[s + 1:].sum(axis=0)
    return out * det


# ---------------------------------------------------------------------------
# The bound
# ---------------------------------------------------------------------------

def combine_across_seeds(x: np.ndarray) -> dict:
    """`meta_analyse`'s ACROSS-SEED arm, field for field.

    The inverse-variance FIXED-EFFECT arm is deliberately absent: it needs a
    per-seed standard error and `fit_arm` stores none for `extra_coefs`.
    Reported as `NOT_AVAILABLE` with the reason, never approximated -- an
    interval that cannot be computed from what is stored is stated as
    uncomputable (BC4.3 item 4).
    """
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"status": "NO_USABLE_SEEDS", "n_seeds_used": 0}
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    se = sd / np.sqrt(v.size) if v.size > 1 else float("nan")
    return {
        "status": "OK",
        "n_seeds_used": int(v.size),
        "across_seed_mean": mean,
        "across_seed_sd": sd,
        "across_seed_se": se,
        "across_seed_ci95": ([mean - 1.959964 * se, mean + 1.959964 * se]
                             if np.isfinite(se) else [float("nan")] * 2),
        "fixed_effect": "NOT_AVAILABLE",
        "fixed_effect_reason": (
            "fit_arm stores no standard error for extra_coefs and refitting "
            "is not authorised (BC9), so no inverse-variance weight exists"),
    }


def outcome_cell(contribution_pp: float, excess_pp: float,
                 cannot_below: float = CANNOT_ACCOUNT_BELOW_PP,
                 leading_above: float = LEADING_CANDIDATE_ABOVE_PP) -> dict:
    """BC4.4's four-cell grid, TOTAL over the contribution's whole domain.

    The fourth cell exists because a rule needs every cell mapped (gotcha 24):
    a non-positive contribution means the enrichment half and the coefficient
    half do NOT in fact agree in direction, and that is an INCONSISTENCY to
    report rather than a small number to close on. A non-finite input gets its
    own branch rather than falling through a comparison chain, because
    `nan < 1.5` is False and would silently land in the wrong cell.

    `runs_followon` is DERIVED here so the conditional at BC5 is executed by a
    function rather than adjudicated by whoever reads the number.
    """
    if not np.isfinite(contribution_pp) or not np.isfinite(excess_pp):
        return {"contribution_pp": (float(contribution_pp)
                                    if np.isfinite(contribution_pp) else None),
                "excess_pp": (float(excess_pp) if np.isfinite(excess_pp)
                              else None),
                "share_of_excess": None,
                "verdict": "UNDEFINED_NON_FINITE_INPUT",
                "reading": "no cell applies; report and stop",
                "runs_followon": False}
    share = (contribution_pp / excess_pp) if excess_pp != 0 else None
    if contribution_pp <= 0.0:
        verdict = "NEGATIVE_OR_ZERO_THE_TWO_HALVES_DO_NOT_AGREE"
        reading = ("the enrichment half and the coefficient half do NOT point "
                   "the same way -- one was misread. Report as an "
                   "inconsistency and stop")
        run = False
    elif contribution_pp < cannot_below:
        verdict = "PERPLEXITY_CANNOT_ACCOUNT_FOR_THE_EXCESS"
        reading = ("close it: no stratification, and the override is not "
                   "exercised")
        run = False
    elif contribution_pp <= leading_above:
        verdict = "PARTIAL_CONTRIBUTOR"
        reading = "run the perplexity-matched ambient (BC5)"
        run = True
    else:
        verdict = "LEADING_CANDIDATE"
        reading = "run the perplexity-matched ambient (BC5)"
        run = True
    return {
        "contribution_pp": float(contribution_pp),
        "excess_pp": float(excess_pp),
        "share_of_excess": (float(share) if share is not None else None),
        "cut_cannot_account_below_pp": float(cannot_below),
        "cut_leading_candidate_above_pp": float(leading_above),
        "verdict": verdict,
        "reading": reading,
        "runs_followon": bool(run),
    }


def contribution_bound(rows: list[dict], perm: list[int], disp: dict,
                       excess_pp: float) -> dict:
    """The product, per seed, then combined across seeds.

    `rows` are the stored fit's rows and `perm[i]` names the row belonging to
    AP row `i`, so the coefficient and the displacement for a seed are joined
    BY LABEL. The per-SD presentation is emitted with its identity residual,
    because `(ame * sd) * (delta / sd) == ame * delta` is the reason the SD
    choice is not a degree of freedom and a claim like that should be measured
    rather than asserted (gotcha 20).
    """
    per_seed, contrib = [], []
    for i, r in enumerate(disp["per_seed"]):
        if r.get("status") != "COMPUTED" or i >= len(perm) or perm[i] < 0:
            per_seed.append({"ap_row": i, "status": "NOT_COMPUTED"})
            continue
        f = rows[perm[i]]
        ame = f["beta"] * f["mean_p_1_minus_p"]
        d = r["mean_displacement"]
        sd = r["sd_reference"]
        c = ame * d
        per_sd = ame * sd
        d_in_sd = d / sd if sd not in (0.0,) and np.isfinite(sd) else float("nan")
        per_seed.append({
            "ap_row": i,
            "seed_label": f["label"],
            "stored_fit_row": perm[i],
            "status": "COMPUTED",
            "beta": f["beta"],
            "mean_p_1_minus_p": f["mean_p_1_minus_p"],
            "ame_per_unit": ame,
            "ame_percentage_points_per_unit": 100.0 * ame,
            "sd_reference": sd,
            "ame_percentage_points_per_sd": 100.0 * per_sd,
            "mean_displacement": d,
            "median_displacement": r["median_displacement"],
            "displacement_in_sd_units": d_in_sd,
            "contribution_pp": 100.0 * c,
            "contribution_pp_via_sd_form": 100.0 * per_sd * d_in_sd,
            "identity_residual_pp": abs(100.0 * c - 100.0 * per_sd * d_in_sd),
            "contribution_pp_median_reduction": 100.0 * ame
            * r["median_displacement"],
            **({"contribution_pp_pair_weighted":
                100.0 * ame * r["pair_weighted_mean_displacement"]}
               if "pair_weighted_mean_displacement" in r else {}),
        })
        contrib.append(100.0 * c)

    ok = [r for r in per_seed if r.get("status") == "COMPUTED"]
    primary = combine_across_seeds(np.array(contrib, dtype=np.float64))
    median_arm = combine_across_seeds(np.array(
        [r["contribution_pp_median_reduction"] for r in ok], dtype=np.float64))
    pw = [r["contribution_pp_pair_weighted"] for r in ok
          if "contribution_pp_pair_weighted" in r]
    pair_arm = (combine_across_seeds(np.array(pw, dtype=np.float64)) if pw
                else {"status": "NOT_SUPPLIED"})

    mean_pp = primary.get("across_seed_mean", float("nan"))
    cell = outcome_cell(mean_pp, excess_pp)
    med_pp = median_arm.get("across_seed_mean", float("nan"))
    med_cell = outcome_cell(med_pp, excess_pp)
    straddles = (np.isfinite(mean_pp) and np.isfinite(med_pp)
                 and cell["verdict"] != med_cell["verdict"])
    return {
        "per_seed": per_seed,
        "n_seeds_used": len(ok),
        "primary_mean_reduction": primary,
        "secondary_median_reduction": median_arm,
        "secondary_pair_weighted_reduction": pair_arm,
        "max_identity_residual_pp": (float(max(r["identity_residual_pp"]
                                               for r in ok)) if ok else None),
        "identity_note": ("(ame * sd) * (delta / sd) == ame * delta, so the SD "
                          "choice cannot move the product. The residual above "
                          "is the measured departure from that identity in "
                          "float arithmetic"),
        "outcome": cell,
        "outcome_under_the_median_reduction": med_cell,
        # DERIVED (gotcha 20). BC4.4 declared the straddle rule BEFORE the run:
        # the primary reduction takes the branch and the straddle is reported.
        "reductions_straddle_a_cut": bool(straddles),
        "straddle_note": (
            "the mean-based and median-based reductions fall in DIFFERENT "
            "cells; BC4.4's pre-declared rule gives the branch to the primary "
            "(mean) reduction and requires the straddle be reported"
            if straddles else
            "both reductions fall in the same cell, so the branch does not "
            "depend on which reduction is primary"),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _decode(arr) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else x for x in arr]


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


def format_summary(res: dict) -> list[str]:
    """A TOTAL formatter. BB5d: a completed measurement was once reported as a
    FAILED job because a summary print died on a legitimately UNDEFINED span,
    after the artifact was written and acceptance had passed. Every branch here
    renders, including the ones with no number in them."""
    out = ["", "[BC2] perplexity's contribution to the +7.22 pp carrier excess"]
    b = res.get("bound", {})
    p = b.get("primary_mean_reduction", {})
    if p.get("status") != "OK":
        out.append(f"  bound: {p.get('status', 'ABSENT')}")
    else:
        ci = p["across_seed_ci95"]
        out.append(f"  contribution {p['across_seed_mean']:+.4f} pp "
                   f"[{ci[0]:+.4f}, {ci[1]:+.4f}] (across-seed)")
    o = b.get("outcome", {})
    share = o.get("share_of_excess")
    out.append(f"  share of the excess: "
               + ("UNDEFINED" if share is None else f"{share:.2%}"))
    out.append(f"  VERDICT {o.get('verdict', 'ABSENT')} -> "
               f"follow-on {'RUNS' if o.get('runs_followon') else 'DOES NOT RUN'}")
    if b.get("reductions_straddle_a_cut"):
        out.append("  NOTE the mean and median reductions STRADDLE a cut; the "
                   "mean takes the branch (BC4.4)")
    fo = res.get("section_2e_perplexity_matched_ambient", {})
    out.append(f"  follow-on: {fo.get('status', 'ABSENT')}")
    if fo.get("status") == "COMPUTED":
        cmp_ = fo.get("comparison", {})
        out.append(f"    excess {fo['arm']['excess_A_minus_matched_pp']:+.4f} pp"
                   f"  ratio to single "
                   + (f"{cmp_['ratio_double_over_single']:.4f}"
                      if cmp_.get("ratio_double_over_single") is not None
                      else "UNDEFINED"))
        out.append(f"    reading: {fo.get('reading', 'ABSENT')}")
    return out


def run(ap_path: Path, null_path: Path, gtilde_path: Path, d_fs_path: Path,
        au_path: Path, ckpts: list[Path], ckpts_root: Path,
        row_index_cache: Path, eval_cache: Path, labels: Path,
        labels_json: Path, counts_reference: Path, ay_reference: Path,
        ba_reference: Path, conc_reference: Path, out_path: Path,
        device: str = "cpu", theta: float = THETA_REFERENCE,
        stratum: str = "pooled", n_perp: int = N_PERPLEXITY_BINS,
        scored_row_set: str = "label_store", k: int = 32) -> dict:
    with h5py.File(ap_path, "r") as hf:
        g = hf[stratum]
        ap_concepts = _decode(g["concepts"][:])
        ap_seeds = _decode(hf["seeds"][:])
        in_pool = g["in_argmax_pool"][:].astype(bool)
        ap = g["ap"][:]
        prevalence = g["prevalence"][:]
        fire_counts = hf["fire_counts"][:]
        ap_identity = _attrs(hf)
    n_rows = int(ap_identity["n_rows"])

    with h5py.File(null_path, "r") as hf:
        null_concepts = _decode(hf["concepts"][:])
        null_seeds = _decode(hf["seeds"][:])
        null_cmax = hf["null_cmax_b"][:]

    assert_labels_align("seed labels (AP vs null)", ap_seeds, null_seeds)
    pool_concepts = [c for c, kk in zip(ap_concepts, in_pool) if kk]
    assert_labels_align("concept labels (AP pool vs null)",
                        pool_concepts, null_concepts)

    with h5py.File(gtilde_path, "r") as hf:
        gt_identity = _attrs(hf)
    with h5py.File(d_fs_path, "r") as hf:
        d_raw = hf["d_fs_cluster_entropy"][:]
        d_seeds = [int(x) for x in hf["seeds"][:]]
        d_identity = _attrs(hf)
    d_fs = d_raw[align_to_ap_order([str(x) for x in d_seeds], ap_seeds,
                                   str(d_fs_path))]

    # ---- ACCEPTANCE C, D, F: the stored fit, read and checked ---------------
    au = json.loads(au_path.read_text())
    fit = read_feature_side_arm(au, COVARIATE)
    if fit["problems"]:
        raise SystemExit("FATAL: the stored feature-side arm could not be "
                         "read.\n" + json.dumps(fit["problems"], indent=2))
    acc_c = acceptance_ame_reconstruction(fit["rows"])
    acc_f = acceptance_bb93_summary(fit["rows"])
    acc_d = sign_convention_probe()
    for nm, a, good in (("C", acc_c, "REPRODUCES_THE_STORED_AME"),
                        ("F", acc_f, "REPRODUCES_BB93"),
                        ("D", acc_d, "SIGN_CONVENTION_CONFIRMED")):
        print(f"[pbound] acceptance {nm}: {a['verdict']}", flush=True)
        if a["verdict"] != good:
            print(json.dumps(a, indent=2), flush=True)
            raise SystemExit(1)

    join = join_rows_to_ap_order(fit["rows"], ap_seeds)
    print(f"[pbound] seed join: {join['verdict']} "
          f"({join['n_positions_moved']} of {join['n_positions_total']} rows "
          f"moved)", flush=True)
    if join["problems"]:
        raise SystemExit("FATAL: the coefficient join crosses the seed-order "
                         "split and could not be verified POSITIONALLY.\n"
                         + json.dumps(join, indent=2))

    # ---- decoders and the MNN ---------------------------------------------
    meta_seed: dict[Path, int] = {}
    loaded: dict[Path, torch.Tensor] = {}
    for p in ckpts:
        W, meta = load_decoder(p, device)
        meta_seed[p] = int(meta["seed"])
        loaded[p] = W
    ordered_paths = order_decoders_to(ap_seeds, meta_seed)
    decoders = [loaded[p] for p in ordered_paths]
    assert_labels_align("decoder seed order vs AP seed labels",
                        [f"seed_{meta_seed[p]}" for p in ordered_paths],
                        ap_seeds)
    lex_to_numeric = np.argsort([meta_seed[p] for p in ordered_paths])
    n_anchors = len(decoders)
    partner, mnn_sim = mnn_partner_matrix(decoders, theta)

    acc_a = verify_counts_bit_identical(mnn_sim, n_anchors, theta,
                                        counts_reference, lex_to_numeric)
    print(f"[pbound] acceptance A: {acc_a['verdict']}", flush=True)
    if acc_a["verdict"] != "BIT_IDENTICAL":
        print(json.dumps(acc_a, indent=2), flush=True)
        raise SystemExit(1)

    # ---- carriers, detection, bins ----------------------------------------
    ap_pool = ap[:, :, in_pool]
    best_gain = to_gain(observed_best_ap(ap_pool), prevalence[in_pool])
    detected = detection_matrix(
        best_gain, null_thresholds(null_cmax, REFERENCE_PERCENTILE, "pooled"))
    carrier, ties = carriers_from_ap(ap_pool)
    fbin = firing_bin(fire_counts.astype(np.float64) / float(n_rows))
    n_fbins = len(BIN_EDGES) - 1
    n_features = int(ap_pool.shape[1])

    print(f"[pbound] assembling covariates for {n_anchors} anchors", flush=True)
    perp, _conc, cov_prov = build_feature_covariates(
        ap_seeds, ckpts_root, row_index_cache, eval_cache, labels, labels_json,
        pool_concepts, n_rows, n_features, scored_row_set, k)
    if cov_prov["problems"]:
        raise SystemExit("FATAL: the covariate arm's per-anchor provenance "
                         "does not check out.\n" + json.dumps(cov_prov,
                                                              indent=2))
    print(f"[pbound] covariate seed axis: {cov_prov['verdict']}", flush=True)

    # ---- ACCEPTANCE B: the enrichment halves reproduce exactly -------------
    dist_d_fs = support_distribution(carrier, detected, d_fs, fbin, n_fbins,
                                     value_name="d_fs")
    acc_b_dfs = reproduces_ba9(dist_d_fs, ba_reference)
    dist_perp = support_distribution(carrier, detected, perp, fbin, n_fbins,
                                     value_name=COVARIATE)
    acc_b_perp = _reproduces_stored_perplexity(dist_perp, conc_reference)
    for nm, a, good in (("B/d_fs", acc_b_dfs, "REPRODUCES_BA9_EXACTLY"),
                        ("B/perplexity", acc_b_perp,
                         "REPRODUCES_BB91_PERPLEXITY_EXACTLY")):
        print(f"[pbound] acceptance {nm}: {a['verdict']}", flush=True)
        if a["verdict"] != good:
            print(json.dumps(a, indent=2)[:8000], flush=True)
            raise SystemExit(1)

    # ---- ACCEPTANCE E: the single arm pins the denominator -----------------
    single = carrier_rates_by_cell(carrier, detected, partner,
                                   cell_index(fbin, None, n_perp), n_fbins)
    acc_e = reproduces_ay(single, ay_reference)
    print(f"[pbound] acceptance E: {acc_e['verdict']}", flush=True)
    if acc_e["verdict"] != "REPRODUCES_AY7_EXACTLY":
        print(json.dumps(acc_e, indent=2), flush=True)
        raise SystemExit(1)
    excess_pp = float(single["excess_A_minus_matched_pp"])

    # ---- the bound ---------------------------------------------------------
    disp = covariate_displacement(carrier, detected, perp, fbin, n_fbins,
                                  COVARIATE,
                                  pair_weight=pair_weights_from_detection(
                                      detected))
    bound = contribution_bound(fit["rows"],
                               join["permutation_ap_row_to_stored_row"],
                               disp, excess_pp)

    # ---- BC5: the conditional follow-on, branch taken BY THE FUNCTION ------
    if bound["outcome"]["runs_followon"]:
        pbin = support_bin_within(fbin, perp, n_perp)
        cell = cell_index(fbin, pbin, n_perp)
        n_cells = n_fbins * n_perp
        print(f"[pbound] BC5 authorised by {bound['outcome']['verdict']}: "
              f"{n_cells} cells", flush=True)
        arm = carrier_rates_by_cell(carrier, detected, partner, cell, n_cells)
        cmp_ = excess_comparison(single, arm)
        shrinks = (np.isfinite(arm["excess_A_minus_matched_pp"])
                   and arm["excess_A_minus_matched_pp"]
                   < SHRINKS_MATERIALLY_BELOW_FRACTION * excess_pp)
        followon = {
            "status": "COMPUTED",
            "authorised_by": bound["outcome"]["verdict"],
            "stratification": ("the nine firing-rate strata x three "
                               "perplexity strata, the latter quantile bins "
                               "taken WITHIN each (seed, firing bin)"),
            "n_cells": int(n_cells),
            "arm": arm,
            "comparison": cmp_,
            "cell_usability": cell_usability(cell, n_cells),
            # DERIVED from the measured excess (gotcha 20), against AZ-P1's
            # own cut reused rather than invented.
            "reading": ("SHRINKS_MATERIALLY -- perplexity is a substantial "
                        "non-alignment explanation for the carrier excess"
                        if shrinks else
                        "SURVIVES -- three enrichment-and-matching checks have "
                        "failed to attribute the excess; it is reported as "
                        "UNATTRIBUTED and the writeup says so"),
            "shrinks_materially_cut_pp": (
                SHRINKS_MATERIALLY_BELOW_FRACTION * excess_pp),
        }
    else:
        followon = {
            "status": "NOT_RUN_THE_PRE_DECLARED_BRANCH_FORBIDS_IT",
            "forbidden_by": bound["outcome"]["verdict"],
            "note": ("BC5 authorises the follow-on only for "
                     "PARTIAL_CONTRIBUTOR or LEADING_CANDIDATE. The branch is "
                     "taken by `outcome_cell`, not by a reader of the number"),
        }

    from src.sae.train import git_dirty, git_sha

    out = {
        "specification": ("PREREGISTRATION amendment BC section 2 (brief "
                          "section 2) -- bounding perplexity's contribution to "
                          "the +7.22 pp carrier excess"),
        "exploratory": ("AZ7/BA5/BC3 -- cannot become primary under any "
                        "outcome; no mechanism sentence is written for any of "
                        "it"),
        "override_scope": (
            "BC4.1 -- BB's stopping rule stands; this repairs an ASYMMETRY in "
            "BB's brief (its enrichment half measured both feature-side "
            "covariates and its matching half specified only one). Perplexity "
            "ONLY, and no further attribution attempts after it, whatever it "
            "returns"),
        "caveats_that_travel_with_every_number": [
            "the coefficient comes from the feature-side SENSITIVITY arm's "
            "fit, not the primary. It is the correct fit -- the only one "
            "containing perplexity -- but the arm is DEMOTED and its own b1_z "
            "is REVERSED relative to the primary's (AS11, AT, AU14)",
            "the conversion crosses from a logit coefficient in a penalised- "
            "spline model to a match-rate DIFFERENCE, via a first-order linear "
            "approximation. THE RESULT IS A BOUND, NOT A POINT ESTIMATE",
        ],
        "identity": {"git_sha": git_sha(), "git_dirty": git_dirty(),
                     "measurement": "perplexity_bound_BC2"},
        "theta": float(theta),
        "reference_percentile": REFERENCE_PERCENTILE,
        "acceptance_criteria": {
            "A_counts_vs_rf_counts": acc_a,
            "B_enrichment_reproduces": {"d_fs_vs_BA9": acc_b_dfs,
                                        "perplexity_vs_BB91": acc_b_perp},
            "C_ame_reconstruction": acc_c,
            "D_sign_convention": acc_d,
            "E_single_arm_vs_carrier_identity_AY": acc_e,
            "F_stored_coefficients_vs_BB93": acc_f,
        },
        "seed_axis": {
            "ap_seed_labels_lexicographic": ap_seeds,
            "decoder_meta_seeds_in_that_order":
                [meta_seed[p] for p in ordered_paths],
            "permutation_lexicographic_to_numeric": lex_to_numeric.tolist(),
            "covariate_arm_provenance": cov_prov,
            "coefficient_join": join,
        },
        "argmax_ties": ties,
        "stored_fit": {k2: v for k2, v in fit.items() if k2 != "rows"},
        "excess_reference": {
            "excess_A_minus_matched_pp": excess_pp,
            "source": "the single (firing-only) arm, re-derived here and "
                      "pinned by acceptance E",
        },
        "section_2_displacement": disp,
        "bound": bound,
        "section_2e_perplexity_matched_ambient": followon,
        "source": {"ap": str(ap_path), "null": str(null_path),
                   "gtilde": str(gtilde_path), "d_fs": str(d_fs_path),
                   "design3_join": str(au_path), "labels": str(labels),
                   "row_index_cache": str(row_index_cache),
                   "eval_cache": str(eval_cache),
                   "counts_reference": str(counts_reference),
                   "ay_reference": str(ay_reference),
                   "ba_reference": str(ba_reference),
                   "concentration_reference": str(conc_reference),
                   "ap_identity": ap_identity, "gtilde_identity": gt_identity,
                   "d_fs_identity": d_identity},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    return out


def _reproduces_stored_perplexity(dist: dict, reference: Path) -> dict:
    """ACCEPTANCE B, perplexity half -- against BB9.1's STORED block.

    Deliberately NOT `reproduces_ba9` with a different path: that function
    knows where BA9's block lives, and this one lives somewhere else. Reusing
    it by pointing it at another file would be an existence check wearing an
    identity check's clothes (gotcha 15).
    """
    ref = json.loads(Path(reference).read_text())
    try:
        block = ref["section_3a_enrichment"]["perplexity"]["distribution"]
    except (KeyError, TypeError):
        return {"verdict": "REFERENCE_BLOCK_ABSENT",
                "reference": str(reference)}
    diffs, compared = compare_exact(block, dist)
    return {
        "reference": str(reference),
        "block": "section_3a_enrichment/perplexity/distribution",
        "n_leaves_compared": compared,
        "differences": diffs,
        # DERIVED (gotcha 20).
        "verdict": ("REPRODUCES_BB91_PERPLEXITY_EXACTLY" if not diffs
                    else "DIFFERS_FROM_BB91_PERPLEXITY"),
    }


def compare_exact(ref, now, prefix: str = "") -> tuple[dict, int]:
    """EXACT equality over every leaf of two nested documents.

    An ABSENT key is a DIFFERENCE, never agreement (gotcha 15), and the count
    of leaves actually compared is returned so a comparison that silently
    compared nothing is visible: `differences == {}` over ZERO leaves is not a
    pass, and only the count can tell the two apart (gotcha 28 -- absence of
    output is not evidence).
    """
    diffs: dict = {}
    n = 0
    if isinstance(ref, dict) and isinstance(now, dict):
        for k in sorted(set(ref) | set(now)):
            d, c = compare_exact(ref.get(k, "__ABSENT__"),
                                 now.get(k, "__ABSENT__"), f"{prefix}/{k}")
            diffs.update(d)
            n += c
        return diffs, n
    if isinstance(ref, list) and isinstance(now, list):
        if len(ref) != len(now):
            return ({prefix or "/": {"reference_len": len(ref),
                                     "now_len": len(now)}}, 1)
        for i, (a, b) in enumerate(zip(ref, now)):
            d, c = compare_exact(a, b, f"{prefix}[{i}]")
            diffs.update(d)
            n += c
        return diffs, n
    if ref != now:
        diffs[prefix or "/"] = {"reference": ref, "now": now}
    return diffs, 1


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ap", type=Path, required=True)
    p.add_argument("--null", type=Path, required=True)
    p.add_argument("--gtilde", type=Path, required=True)
    p.add_argument("--d-fs", type=Path, required=True)
    p.add_argument("--design3-join", type=Path, required=True,
                   help="design3_join_au.json -- the STORED coefficients")
    p.add_argument("--ckpts", nargs="+", type=Path, required=True)
    p.add_argument("--ckpts-root", type=Path, required=True)
    p.add_argument("--row-index-cache", type=Path, required=True)
    p.add_argument("--eval-cache", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--labels-json", type=Path, default=None)
    p.add_argument("--counts-reference", type=Path, required=True)
    p.add_argument("--ay-reference", type=Path, required=True)
    p.add_argument("--ba-reference", type=Path, required=True)
    p.add_argument("--concentration-reference", type=Path, required=True,
                   help="carrier_concentration_bb.json -- acceptance B")
    p.add_argument("--expect-ckpts", type=int, default=None)
    p.add_argument("--n-perplexity-bins", type=int, default=N_PERPLEXITY_BINS)
    p.add_argument("--scored-row-set", default="label_store")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--theta", type=float, default=THETA_REFERENCE)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.expect_ckpts is not None and len(args.ckpts) != args.expect_ckpts:
        print(f"PRE-FLIGHT FAIL: --expect-ckpts {args.expect_ckpts} but "
              f"{len(args.ckpts)} supplied. A count mismatch silently changes "
              f"which ensemble every number below describes.", flush=True)
        raise SystemExit(1)

    res = run(args.ap, args.null, args.gtilde, args.d_fs, args.design3_join,
              args.ckpts, args.ckpts_root, args.row_index_cache,
              args.eval_cache, args.labels,
              args.labels_json or args.labels.with_suffix(".json"),
              args.counts_reference, args.ay_reference, args.ba_reference,
              args.concentration_reference, args.out, args.device, args.theta,
              n_perp=args.n_perplexity_bins,
              scored_row_set=args.scored_row_set, k=args.k)

    for line in format_summary(res):
        print(line)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
