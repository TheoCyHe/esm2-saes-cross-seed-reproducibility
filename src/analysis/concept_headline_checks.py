"""AZ sections 13-14 -- TWO VALIDITY CHECKS ON THE CONCEPT-LEVEL HEADLINE.

=============================================================================
AZ13 -- DOES THE PERMUTATION NULL PRESERVE EACH CONCEPT'S PREVALENCE?
=============================================================================
**This sits directly under the positive headline and has been listed as
unmeasured since AW1.4** (`docs/METHODS_PROVENANCE.md` section 6, carried
forward at AX14 and AY15).

The concept-level detection threshold is a percentile of Null B's gain
distribution. **Average precision is prevalence-dependent**, so if a concept's
positive-residue count drifts under permutation the threshold describes a
concept of a different size than the observed one -- and the direction of that
bias is not obvious.

**IT IS NOT PRESERVED BY DESIGN, AND THAT IS READ FROM THE WRITER RATHER THAN
INFERRED FROM THE MEASUREMENT.** `null_pilot.cross_protein_shift_rows` moves
each positive row onto a destination protein of similar length and RESCALES its
offset by `floor(offset * q_len / p_len)`. When the destination is SHORTER the
map is many-to-one, so distinct source rows collide. `null_pilot.py` then calls
`np.unique` and uses the REALISED count for that draw's gain normalisation:

    shifted_b = np.unique(shifted_b)
    n_pos_b = shifted_b.size
    ...
    p_c_b = n_pos_b / n_rows

So each draw is internally consistent -- and its prevalence is **at most** the
observed one, never above it. **What has never been measured is HOW FAR BELOW.**
That is this section: a pure re-execution of the stored schedule against the
stored label store, **with no average precision recomputed and no GPU**.

**THE STORED SCHEDULE IS USED, NOT A REGENERATED ONE.** `dest_schedule` is a
dataset in `null_pilot_compliant.h5`; regenerating it from `null_seed` would
measure a schedule the null never used. If the dataset is absent this module
REFUSES rather than reconstructing it (AZ9 item 4).

=============================================================================
AZ14 -- THE BOUNDED HEADLINE (AZ4)
=============================================================================
The shared-SCHEDULE channel cannot be removed from draws already stored, and no
new null pass is authorised. AZ4 declares the bound instead: report the headline
**with and without** the concepts whose WORST seed clears its threshold by less
than 2x. **Arithmetic, not a prediction** -- roughly 75 of 529 concepts move and
a mechanical shift of a few points is expected and is not a finding.

**The three detection groups separate perfectly on the margin, and that
separation is DEFINITIONAL, not evidence** (AY8.3): `D = 1` means exactly that
the worst seed's margin exceeds 1. Only the distribution WITHIN the `D = 1`
group carries information, and only that is reported here.

REUSE, NOT REIMPLEMENTATION. Detection comes from `concept_repro`; the margin
from `concept_bridge`; the shift from `null_pilot`. A second implementation of
any of them could disagree with the artifact of record with no way to tell which
produced 75.25%.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from src.analysis.concept_bridge import detection_margin
from src.analysis.concept_repro import (
    REFERENCE_PERCENTILE,
    assert_labels_align,
    detection_matrix,
    detection_rate,
    null_thresholds,
    observed_best_ap,
    to_gain,
)
from src.analysis.feature_ap import load_labels, load_scored_row_set
from src.analysis.null_pilot import (
    build_population_index,
    build_row_to_protein,
    cross_protein_shift_rows,
    protein_starts_lengths,
)

MARGIN_BOUND = 2.0


# ------------------------------------------------- AZ13: realised prevalence

def realised_positive_counts(rows: np.ndarray, row2protein: np.ndarray,
                             starts: np.ndarray, lengths: np.ndarray,
                             dest_schedule: np.ndarray) -> np.ndarray:
    """Positive-row count of ONE concept under each stored permutation.

    Exactly `null_pilot`'s two lines -- `cross_protein_shift_rows` then
    `np.unique` -- and nothing else, so this measures what the null did rather
    than what a re-derivation would do.
    """
    out = np.empty(dest_schedule.shape[0], dtype=np.int64)
    for r in range(dest_schedule.shape[0]):
        shifted = cross_protein_shift_rows(rows, row2protein, starts, lengths,
                                           dest_schedule[r])
        out[r] = np.unique(shifted).size
    return out


def drift_summary(observed: int, realised: np.ndarray) -> dict:
    """Relative drift of one concept, over its draws.

    `relative_drift = realised / observed - 1`, so 0 is exact preservation and
    a negative value is loss. Reported as a distribution over draws and not
    only as a mean: a schedule that loses nothing on 19 draws and half on one
    is not the same object as one that loses 2.5% on every draw.
    """
    obs = float(observed)
    rel = realised.astype(np.float64) / obs - 1.0
    return {
        "n_positive_observed": int(observed),
        "n_draws": int(realised.size),
        "realised_min": int(realised.min()),
        "realised_median": float(np.median(realised)),
        "realised_max": int(realised.max()),
        "relative_drift_mean": float(rel.mean()),
        "relative_drift_median": float(np.median(rel)),
        "relative_drift_min": float(rel.min()),
        "relative_drift_max": float(rel.max()),
        "n_draws_exactly_preserved": int((realised == observed).sum()),
        "n_draws_above_observed": int((realised > observed).sum()),
    }


def null_max_vs_prevalence(realised: np.ndarray, n_rows: int,
                           cmax: np.ndarray) -> dict:
    """BA10 (brief 3a) -- does the null maximum scale with the DRAW'S prevalence?

    =======================================================================
    A UNIT CORRECTION THAT CHANGES WHAT THIS MEASURES (BA7.2)
    =======================================================================
    The question is usually posed about the null's maximum AVERAGE PRECISION,
    on the textbook ground that average precision for a random ranker
    approximates prevalence. **The stored quantity is not an average
    precision.** `null_pilot.py` computes

        gain_b = (ap_b - p_c_b) / (1 - p_c_b)

    with `p_c_b` the draw's REALISED prevalence, takes the max over features,
    and stores THAT. So the stored maximum is already normalised at the very
    quantity whose effect is in question, and the textbook expectation
    describes the number BEFORE that division.

    **Both are reported, because the transform is exact and invertible.** For a
    fixed `p` the gain is a strictly increasing function of AP, so the
    maximising feature is the same one and

        ap_max = gain_max * (1 - p) + p

    recovers the raw maximum exactly. **The GAIN is what sets the threshold;
    the AP is what the textbook argument is about.** Reporting one without the
    other is the transported constant this project has shipped three times.

    A concept whose realised prevalence is CONSTANT across draws has no
    variation to correlate against, and a ranking predicate applied to a
    constant ranks the tie-break (gotcha 25b). Those are counted as
    `UNINFORMATIVE`, never as a zero correlation -- absent variation is not
    absent association.

    `realised` is `(n_draws,)` for one concept; `cmax` is `(n_seeds, n_draws)`.
    The stored schedule has NO seed axis, so the prevalence sequence is shared
    across seeds and only `cmax` varies.
    """
    from src.analysis.concept_repro import spearman

    p = np.asarray(realised, dtype=np.float64) / float(n_rows)
    if np.unique(p).size < 2:
        return {"status": "UNINFORMATIVE_PREVALENCE_IS_CONSTANT",
                "n_distinct_prevalences": int(np.unique(p).size),
                "why": "a ranking statistic against a constant ranks the "
                       "tie-break; reported as uninformative rather than as a "
                       "correlation of zero (gotcha 25b)"}
    gain = np.asarray(cmax, dtype=np.float64)
    ap = gain * (1.0 - p)[None, :] + p[None, :]

    # A SEED WHOSE MAXIMUM IS CONSTANT ACROSS DRAWS has nothing to rank either,
    # and `concept_repro.spearman` REFUSES on a constant input rather than
    # returning 0.0 -- which is the correct behaviour and would abort the job
    # if it were not handled here. Those seeds are COUNTED, and the summary is
    # taken over the informative ones (gotcha 25b, on the other argument).
    rho_gain, rho_ap = [], []
    n_const_gain = n_const_ap = 0
    for s in range(gain.shape[0]):
        if np.unique(gain[s]).size < 2:
            n_const_gain += 1
        else:
            rho_gain.append(spearman(p, gain[s]))
        if np.unique(ap[s]).size < 2:
            n_const_ap += 1
        else:
            rho_ap.append(spearman(p, ap[s]))
    if not rho_gain and not rho_ap:
        return {"status": "UNINFORMATIVE_EVERY_SEED_MAXIMUM_IS_CONSTANT",
                "n_seeds": int(gain.shape[0]),
                "why": "prevalence varies but no seed's stored maximum does, "
                       "so there is nothing to rank against it"}
    g = np.asarray(rho_gain, dtype=np.float64)
    a = np.asarray(rho_ap, dtype=np.float64)
    return {
        "status": "COMPUTED",
        "n_draws": int(p.size),
        "n_seeds": int(gain.shape[0]),
        "n_seeds_with_a_constant_gain_maximum": n_const_gain,
        "n_seeds_with_a_constant_ap_maximum": n_const_ap,
        "median_spearman_prevalence_vs_null_max_GAIN": (
            float(np.median(g)) if g.size else None),
        "median_spearman_prevalence_vs_null_max_AP": (
            float(np.median(a)) if a.size else None),
        "mean_spearman_gain": float(g.mean()) if g.size else None,
        "mean_spearman_ap": float(a.mean()) if a.size else None,
    }


def null_scaling_population(per_concept: dict[str, dict]) -> dict:
    """The distribution ACROSS concepts of BA10's per-concept correlations.

    The SIGN is the answer and it is derived from the numbers, never asserted
    (gotcha 20). The two units are kept apart throughout: they answer different
    questions and can legitimately disagree in sign, since one is the other
    divided by a function of the same `p`.
    """
    g = np.array([v["median_spearman_prevalence_vs_null_max_GAIN"]
                  for v in per_concept.values()
                  if v.get("status") == "COMPUTED"
                  and v.get("median_spearman_prevalence_vs_null_max_GAIN")
                  is not None])
    a = np.array([v["median_spearman_prevalence_vs_null_max_AP"]
                  for v in per_concept.values()
                  if v.get("status") == "COMPUTED"
                  and v.get("median_spearman_prevalence_vs_null_max_AP")
                  is not None])
    n_unin = sum(1 for v in per_concept.values()
                 if v.get("status", "").startswith("UNINFORMATIVE"))
    if g.size == 0:
        return {"status": "NO_CONCEPT_HAD_VARYING_PREVALENCE",
                "n_uninformative_constant_prevalence": int(n_unin)}

    def side(x: np.ndarray) -> dict:
        return {
            "n_concepts": int(x.size),
            "median": float(np.median(x)), "mean": float(x.mean()),
            "p05": float(np.percentile(x, 5)),
            "p25": float(np.percentile(x, 25)),
            "p75": float(np.percentile(x, 75)),
            "p95": float(np.percentile(x, 95)),
            "share_positive": float((x > 0).mean()),
            "share_negative": float((x < 0).mean()),
        }

    gs, as_ = side(g), side(a)

    def verdict(s: dict) -> str:
        if s["share_positive"] >= 0.60:
            return "SCALES_POSITIVELY_WITH_PREVALENCE"
        if s["share_negative"] >= 0.60:
            return "SCALES_NEGATIVELY_WITH_PREVALENCE"
        return "NO_CONSISTENT_DIRECTION"

    return {
        "status": "COMPUTED",
        "n_uninformative_constant_prevalence": int(n_unin),
        "gain_the_quantity_that_sets_the_threshold": gs,
        "ap_the_quantity_the_textbook_argument_is_about": as_,
        "verdict_gain": verdict(gs),
        "verdict_ap": verdict(as_),
        "bands": "at least 60% of concepts on one side is a direction; "
                 "otherwise NO_CONSISTENT_DIRECTION",
        "claimed": "the SIGN is a measurement. NO MECHANISM SENTENCE is "
                   "written for it (BA5), and the direction in which the "
                   "drift moves a detection threshold is stated only insofar "
                   "as it is arithmetic from this sign.",
    }


def margin_distribution(d: np.ndarray, worst_margin: np.ndarray,
                        cuts: tuple[float, ...] = (1.05, 1.10, 1.20, 1.30,
                                                   1.50, 2.00)) -> dict:
    """BA10 (brief 3b) -- the detection-margin distribution NEAR 1x.

    **WHY 1.2x IS THE CUT THAT BOUNDS EVERYTHING, as arithmetic.** If the
    detection threshold scales with prevalence, the worst measured per-concept
    drift of -15.85% understates the threshold by at most
    `1 / (1 - 0.1585) = 1.1884`. **So every detection whose margin exceeds
    1.19x is safe against even the worst drift measured**, and 1.2x is that
    bound rounded away from zero.

    Restricted to the `D = 1` concepts, and that restriction is not a choice:
    the three detection groups separate PERFECTLY on this statistic and the
    separation is DEFINITIONAL (AY8.3) -- `D < 1` means some seed had a margin
    at or below 1. Only the distribution WITHIN `D = 1` carries information.
    """
    det = d == 1.0
    finite = np.isfinite(worst_margin)
    use = det & finite
    m = worst_margin[use]
    if m.size == 0:
        return {"status": "NO_ALWAYS_DETECTED_CONCEPT_WITH_A_FINITE_MARGIN"}
    return {
        "status": "COMPUTED",
        "population": "the always-detected (D = 1) concepts with a finite "
                      "margin; the separation of the three detection groups "
                      "on this statistic is definitional, not a finding",
        "n_concepts": int(m.size),
        "n_always_detected": int(det.sum()),
        "n_excluded_non_finite_margin": int((det & ~finite).sum()),
        "why_1_2x": "the worst measured per-concept drift is -15.85%, which "
                    "understates a prevalence-scaled threshold by at most "
                    "1/(1-0.1585) = 1.1884; 1.2x is that bound rounded away "
                    "from zero",
        "shares_below": {f"{c:.2f}": float((m < c).mean()) for c in cuts},
        "counts_below": {f"{c:.2f}": int((m < c).sum()) for c in cuts},
        "percentiles": {f"p{q:02d}": float(np.percentile(m, q))
                        for q in (0, 1, 5, 10, 25, 50, 75, 90, 100)},
        "median": float(np.median(m)),
        "min": float(m.min()),
    }


def drift_population(per_concept: dict[str, dict]) -> dict:
    """The distribution ACROSS concepts of the per-concept drift.

    The worst case is reported by NAME as well as by value: a 30% loss
    concentrated on one rare concept and a 30% loss spread over the library are
    different limitations, and a percentile alone cannot tell them apart.
    """
    med = np.array([v["relative_drift_median"] for v in per_concept.values()])
    worst = np.array([v["relative_drift_min"] for v in per_concept.values()])
    names = list(per_concept)
    order = np.argsort(worst)
    return {
        "n_concepts": int(med.size),
        "median_of_per_concept_median_drift": float(np.median(med)),
        "mean_of_per_concept_median_drift": float(med.mean()),
        "p01": float(np.percentile(med, 1)),
        "p25": float(np.percentile(med, 25)),
        "p75": float(np.percentile(med, 75)),
        "p99": float(np.percentile(med, 99)),
        "most_negative_median_drift": float(med.min()),
        "most_positive_median_drift": float(med.max()),
        "n_concepts_with_any_draw_above_observed": int(sum(
            1 for v in per_concept.values() if v["n_draws_above_observed"])),
        "n_concepts_exactly_preserved_in_every_draw": int(sum(
            1 for v in per_concept.values()
            if v["n_draws_exactly_preserved"] == v["n_draws"])),
        "share_of_concepts_with_median_drift_within_1_percent":
            float((np.abs(med) < 0.01).mean()),
        "share_of_concepts_with_median_drift_within_5_percent":
            float((np.abs(med) < 0.05).mean()),
        "worst_ten_by_single_draw": [
            {"concept": names[i],
             "n_positive_observed": per_concept[names[i]]["n_positive_observed"],
             "worst_draw_relative_drift": float(worst[i])}
            for i in order[:10]],
    }


def preservation_verdict(pop: dict) -> dict:
    """DERIVED from the measurement, never asserted (gotcha 20).

    Total over the outcome space: exact preservation everywhere, a drift that
    is small on the stated scale, or one that is not. The bands are stated
    before the numbers are read and the verdict is a function of them.
    """
    if pop["n_concepts_exactly_preserved_in_every_draw"] == pop["n_concepts"]:
        v = "EXACTLY_PRESERVED_IN_EVERY_DRAW_OF_EVERY_CONCEPT"
    elif pop["share_of_concepts_with_median_drift_within_1_percent"] >= 0.95:
        v = "NOT_PRESERVED_BUT_UNDER_1_PERCENT_FOR_AT_LEAST_95_PERCENT"
    elif pop["share_of_concepts_with_median_drift_within_5_percent"] >= 0.95:
        v = "NOT_PRESERVED_UNDER_5_PERCENT_FOR_AT_LEAST_95_PERCENT"
    else:
        v = "NOT_PRESERVED_AND_MATERIAL_FOR_MORE_THAN_5_PERCENT_OF_CONCEPTS"
    return {
        "verdict": v,
        "by_design": "NO -- cross_protein_shift_rows rescales offsets and the "
                     "map is many-to-one onto a shorter destination; "
                     "null_pilot dedups and uses the REALISED count. So this "
                     "is a MEASUREMENT of an approximation, not a "
                     "verification of an identity.",
        "bound_direction": "realised <= observed by construction; any draw "
                           "above observed would indicate a defect and is "
                           "counted rather than assumed impossible",
        "action": "REPORTED ONLY. AZ10 does not authorise correcting it.",
    }


# ---------------------------------------------------- AZ14: bounded headline

def bounded_headline(d: np.ndarray, worst_margin: np.ndarray,
                     bound: float = MARGIN_BOUND) -> dict:
    """The headline bounded against a threshold that might be slightly wrong.

    **THE OBVIOUS CONSTRUCTION IS A TAUTOLOGY AND IS REPORTED AS ONE.**
    "Exclude every concept whose worst-seed margin is below 2" removes EVERY
    concept with `D < 1` by construction -- `D < 1` means some seed failed to
    clear its threshold, i.e. had a margin at or below 1, which is below 2 --
    so the surviving set is entirely `D = 1` and the share is exactly 100%
    whatever the data say. That arm is computed, and its tautology is DERIVED
    (`is_tautological` is a fact about these numbers, not a sentence) rather
    than asserted, because it is AY8.3's definitional-separation caveat firing
    inside the very section that carried it forward.

    **THE TWO ARMS THAT CARRY INFORMATION** both key on the FRAGILE DETECTIONS
    -- concepts at `D = 1` whose weakest seed cleared by less than `bound`:

      * `excluding_fragile_detections` drops them from numerator AND
        denominator: "among concepts whose detections are all secure, what
        share is detected everywhere?"
      * `counting_fragile_as_undetected` is the ADVERSARIAL reading: every
        fragile detection is assumed to be an artifact of a slightly wrong
        threshold and is scored as a failure, with the denominator unchanged.
        This is the lower bound on the headline.

    A concept whose margin is not finite (a non-positive threshold) cannot be
    classified and is counted separately rather than swept into either group.
    """
    finite = np.isfinite(worst_margin)
    fragile = finite & (worst_margin < bound)
    det = d == 1.0
    fragile_det = fragile & det
    n = int(d.size)
    all_share = float(det.mean())

    keep_all = ~fragile
    n_keep_all = int(keep_all.sum())
    degenerate_share = (float(det[keep_all].mean()) if n_keep_all
                        else float("nan"))
    # A FACT ABOUT THESE NUMBERS: is every under-detected concept fragile? If
    # so the arm above can only ever return 1.0.
    under = ~det
    is_taut = bool(under.sum() and bool((fragile[under]).all()))

    keep = ~fragile_det
    n_keep = int(keep.sum())
    excl_share = (float(det[keep].mean()) if n_keep else float("nan"))
    adversarial_share = float((det & ~fragile_det).mean())

    return {
        "bound": float(bound),
        "definition": "a concept is FRAGILE when its worst seed's margin "
                      "(observed gain / detection threshold) is below the "
                      "bound; the set is defined on the margin alone",
        "n_concepts": n,
        "n_with_non_finite_margin": int((~finite).sum()),
        "n_fragile_any_concept": int(fragile.sum()),
        "n_fragile_detections": int(fragile_det.sum()),
        "share_of_detections_that_are_fragile":
            (float(fragile_det.sum() / det.sum()) if det.sum() else None),
        "with_all_concepts": {
            "n_concepts": n, "n_D_eq_1": int(det.sum()),
            "share_D_eq_1": all_share,
        },
        "excluding_fragile_detections": {
            "n_concepts": n_keep,
            "n_D_eq_1": int(det[keep].sum()) if n_keep else 0,
            "share_D_eq_1": excl_share,
            "shift_pp": (excl_share - all_share) * 100.0,
        },
        "counting_fragile_as_undetected": {
            "n_concepts": n,
            "n_D_eq_1": int((det & ~fragile_det).sum()),
            "share_D_eq_1": adversarial_share,
            "shift_pp": (adversarial_share - all_share) * 100.0,
            "reading": "the LOWER BOUND: every fragile detection assumed to be "
                       "a threshold artifact",
        },
        "excluding_every_fragile_concept_DEGENERATE": {
            "n_concepts": n_keep_all,
            "n_D_eq_1": int(det[keep_all].sum()) if n_keep_all else 0,
            "share_D_eq_1": degenerate_share,
            "is_tautological": is_taut,
            "why": "every concept with D < 1 has a seed at or below its "
                   "threshold, hence a margin below any bound at or above 1, "
                   "so this arm returns 1.0 whatever the data say. Reported "
                   "because it is the construction the phrase 'with and "
                   "without the within-2x concepts' most directly names, and "
                   "a reader who computes it should find it here already "
                   "marked.",
        },
        "reading": "ARITHMETIC, NOT A PREDICTION (AZ14). The informative "
                   "numbers are the two SHIFTS; the degenerate arm is not one.",
    }


# ------------------------------------------------------------------ driver

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


def run(ap_path: Path, null_path: Path, labels_path: Path,
        eval_cache: Path, out_path: Path, scored_row_set: str = "label_store",
        stratum: str = "pooled", max_concepts: int | None = None) -> dict:
    with h5py.File(ap_path, "r") as hf:
        g = hf[stratum]
        ap_concepts = _decode(g["concepts"][:])
        ap_seeds = _decode(hf["seeds"][:])
        in_pool = g["in_argmax_pool"][:].astype(bool)
        ap = g["ap"][:]
        prevalence = g["prevalence"][:]
        n_positives = g["n_positives"][:]
        ap_identity = _attrs(hf)

    with h5py.File(null_path, "r") as hf:
        null_concepts = _decode(hf["concepts"][:])
        null_seeds = _decode(hf["seeds"][:])
        null_cmax = hf["null_cmax_b"][:]
        has_dest = "dest_schedule" in hf
        dest_schedule = hf["dest_schedule"][:] if has_dest else None
        null_identity = _attrs(hf)

    assert_labels_align("seed labels (AP vs null)", ap_seeds, null_seeds)
    pool_concepts = [c for c, k in zip(ap_concepts, in_pool) if k]
    assert_labels_align("concept labels (AP pool vs null)",
                        pool_concepts, null_concepts)

    # ---- AZ14 first: it needs nothing but the two artifacts ---------------
    ap_pool = ap[:, :, in_pool]
    best_gain = to_gain(observed_best_ap(ap_pool), prevalence[in_pool])
    thr = null_thresholds(null_cmax, REFERENCE_PERCENTILE, "pooled")
    d = detection_rate(detection_matrix(best_gain, thr))
    margin = detection_margin(best_gain, thr)
    worst = np.nanmin(margin, axis=0)
    bounded = bounded_headline(d, worst)
    # BA10 (brief 3b): the margin distribution NEAR 1x, at finer resolution
    # than the single 2x figure -- the number that bounds the drift's
    # consequence.
    bounded["section_BA10_margin_distribution"] = {
        "specification": "PREREGISTRATION amendment BA section 7.3 "
                         "(brief section 3b)",
        "not_a_prediction": "BA2 -- NOT SEALED, NOT TALLIED",
        **margin_distribution(d, worst),
    }

    # ---- AZ13: the realised prevalence under the STORED schedule ----------
    if not has_dest:
        prevalence_section = {
            "status": "NOT_COMPUTABLE_NO_DEST_SCHEDULE",
            "note": f"{null_path} carries no `dest_schedule` dataset. The "
                    "schedule is NOT reconstructed from `null_seed` -- a "
                    "regenerated schedule is not the one the null used "
                    "(AZ9 item 4). Reported and stopped.",
        }
    else:
        t0 = time.time()
        with h5py.File(eval_cache, "r") as hf:
            raw = hf["index_json"][()]
            index = json.loads(raw.decode() if isinstance(raw, bytes)
                               else str(raw))
        subset = load_scored_row_set(labels_path, scored_row_set)
        with h5py.File(labels_path, "r") as hf:
            n_rows = int(hf.attrs["n_rows"])
        if subset is None:
            row2protein = build_row_to_protein(index, n_rows)
            starts, lengths = protein_starts_lengths(index)
            remap = None
        else:
            pop_index, remap = build_population_index(index, subset)
            row2protein = build_row_to_protein(pop_index, n_rows)
            starts, lengths = protein_starts_lengths(pop_index)

        if dest_schedule.shape[1] != len(starts):
            raise SystemExit(
                f"FATAL: the stored schedule has {dest_schedule.shape[1]:,} "
                f"protein columns and the population index built here has "
                f"{len(starts):,}. Those are different populations, so the "
                "schedule cannot be applied -- refusing rather than "
                "broadcasting.")

        pooled_rows, _ = load_labels(labels_path, "pooled")
        concepts = [c for c in pool_concepts if c in pooled_rows]
        missing = [c for c in pool_concepts if c not in pooled_rows]
        if max_concepts is not None:
            concepts = concepts[:max_concepts]

        per_concept: dict[str, dict] = {}
        per_concept_scaling: dict[str, dict] = {}
        pool_pos = {c: i for i, c in enumerate(pool_concepts)}
        mismatched_counts = []
        for i, c in enumerate(concepts):
            a = pooled_rows[c].astype(np.int64)
            if remap is not None:
                a = a[a < len(remap)]
                m = remap[a]
                rows = m[m >= 0]
            else:
                rows = a
            observed = int(rows.size)
            if observed == 0:
                per_concept[c] = {"status": "NO_POSITIVE_ROWS_IN_POPULATION"}
                continue
            # The population-restricted count must agree with the AP
            # artifact's own `n_positives`, or this is a different label set.
            want = int(n_positives[in_pool][pool_concepts.index(c)])
            if observed != want:
                mismatched_counts.append(
                    {"concept": c, "here": observed, "ap_artifact": want})
            realised = realised_positive_counts(rows, row2protein, starts,
                                                lengths, dest_schedule)
            per_concept[c] = drift_summary(observed, realised)
            # BA10 (3a): the same per-draw counts the summary above reduces
            # away, against the stored null maxima for THIS concept.
            per_concept_scaling[c] = null_max_vs_prevalence(
                realised, n_rows, null_cmax[:, :, pool_pos[c]])
            if (i + 1) % 100 == 0:
                print(f"[prev] {i + 1}/{len(concepts)} concepts, "
                      f"{time.time() - t0:.0f}s", flush=True)

        usable = {k: v for k, v in per_concept.items()
                  if "relative_drift_median" in v}
        prevalence_section = {
            "status": "COMPUTED",
            "n_concepts_measured": len(usable),
            "n_concepts_without_positive_rows": len(per_concept) - len(usable),
            "n_concepts_in_pool_absent_from_the_label_store": len(missing),
            "n_concepts_whose_count_disagrees_with_the_AP_artifact":
                len(mismatched_counts),
            "count_disagreements": mismatched_counts[:20],
            "n_shifts": int(dest_schedule.shape[0]),
            "n_proteins": int(dest_schedule.shape[1]),
            "seconds": round(time.time() - t0, 1),
            "population": drift_population(usable),
            "per_concept": {k: v for k, v in per_concept.items()},
        }
        prevalence_section["verdict"] = preservation_verdict(
            prevalence_section["population"])
        # BA10 (brief 3a): the CONSEQUENCE of the drift measured above.
        prevalence_section["section_BA10_null_max_vs_prevalence"] = {
            "specification": "PREREGISTRATION amendment BA section 7.2 "
                             "(brief section 3a)",
            "not_a_prediction": "BA2 -- NOT SEALED, NOT TALLIED",
            "population": null_scaling_population(per_concept_scaling),
            "per_concept": per_concept_scaling,
        }

    from src.sae.train import git_dirty, git_sha

    out = {
        "specification": "PREREGISTRATION amendment AZ sections 13 and 14 "
                         "(brief sections 4a and 4b)",
        "identity": {"git_sha": git_sha(), "git_dirty": git_dirty(),
                     "measurement": "concept_headline_checks_AZ"},
        "reference_percentile": REFERENCE_PERCENTILE,
        "section_4a_null_prevalence": prevalence_section,
        "section_4b_bounded_headline": bounded,
        "source": {"ap": str(ap_path), "null": str(null_path),
                   "labels": str(labels_path), "eval_cache": str(eval_cache),
                   "scored_row_set": scored_row_set,
                   "ap_identity": ap_identity, "null_identity": null_identity},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ap", type=Path, required=True)
    p.add_argument("--null", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--eval-cache", type=Path, required=True)
    p.add_argument("--scored-row-set", default="label_store")
    p.add_argument("--max-concepts", type=int, default=None,
                   help="smoke-test knob only; the reported run uses the whole "
                        "pool and the count is recorded in the artifact")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    out = run(args.ap, args.null, args.labels, args.eval_cache, args.out,
              args.scored_row_set, max_concepts=args.max_concepts)

    a = out["section_4a_null_prevalence"]
    print(f"\n[4a] {a['status']}")
    if a["status"] == "COMPUTED":
        pop = a["population"]
        print(f"     {a['n_concepts_measured']} concepts x {a['n_shifts']} "
              f"stored shifts in {a['seconds']}s")
        print(f"     median per-concept median drift "
              f"{pop['median_of_per_concept_median_drift']:+.4%}  "
              f"[p01 {pop['p01']:+.4%}, p99 {pop['p99']:+.4%}]")
        print(f"     worst per-concept median drift "
              f"{pop['most_negative_median_drift']:+.4%}; "
              f"{pop['share_of_concepts_with_median_drift_within_1_percent']:.2%}"
              f" within 1%")
        print(f"     verdict: {a['verdict']['verdict']}")

    b = out["section_4b_bounded_headline"]
    print(f"\n[4b] headline with all concepts "
          f"{b['with_all_concepts']['share_D_eq_1']:.2%} "
          f"({b['with_all_concepts']['n_D_eq_1']}/{b['n_concepts']})")
    print(f"     {b['n_fragile_detections']} of "
          f"{b['with_all_concepts']['n_D_eq_1']} detections are within "
          f"{b['bound']:.0f}x of threshold "
          f"({b['share_of_detections_that_are_fragile']:.2%})")
    for arm in ("excluding_fragile_detections",
                "counting_fragile_as_undetected"):
        a = b[arm]
        print(f"     {arm:<34} {a['share_D_eq_1']:.2%} "
              f"({a['n_D_eq_1']}/{a['n_concepts']})  "
              f"shift {a['shift_pp']:+.2f} pp")
    dg = b["excluding_every_fragile_concept_DEGENERATE"]
    print(f"     degenerate arm {dg['share_D_eq_1']:.2%} "
          f"(is_tautological={dg['is_tautological']}) -- not a finding")

    md = b["section_BA10_margin_distribution"]
    if md.get("status") == "COMPUTED":
        print(f"\n[BA10 3b] margin distribution over {md['n_concepts']} "
              f"always-detected concepts (median {md['median']:.2f}x):")
        for k in sorted(md["shares_below"]):
            print(f"      below {k}x: {md['shares_below'][k]:7.2%} "
                  f"({md['counts_below'][k]:>3} concepts)")

    sec = out["section_4a_null_prevalence"].get(
        "section_BA10_null_max_vs_prevalence")
    if sec and sec["population"].get("status") == "COMPUTED":
        pop = sec["population"]
        print(f"\n[BA10 3a] does the null maximum scale with the draw's "
              f"prevalence? ({pop['n_uninformative_constant_prevalence']} "
              f"concepts uninformative -- constant prevalence)")
        for unit, key, vk in (
                ("GAIN (sets the threshold)",
                 "gain_the_quantity_that_sets_the_threshold", "verdict_gain"),
                ("AP  (the textbook argument)",
                 "ap_the_quantity_the_textbook_argument_is_about",
                 "verdict_ap")):
            s = pop[key]
            print(f"      {unit}: median rho {s['median']:+.4f}  "
                  f"[p05 {s['p05']:+.4f}, p95 {s['p95']:+.4f}]  "
                  f"{s['share_positive']:.1%} positive  -> {pop[vk]}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
