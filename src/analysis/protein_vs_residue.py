"""A DESCRIPTIVE protein-versus-residue decomposition on the CONCEPT unit.

WHAT THIS IS, AND WHAT IT IS NOT (PREREGISTRATION amendment BW).

It mirrors the committed concept-level detection construction -- same statistic
(the best feature's gain), same percentile (the 95th), same pooling (400 draws =
20 shifts x 20 seeds), same estimability argument -- with ONE SUBSTITUTION: the
WITHIN-protein circular-shift null (`null_cmax_a`) in place of the ACROSS-protein
length-stratified permutation (`null_cmax_b`). Mirroring an existing construction
rather than inventing one is the whole justification for its form.

**IT IS DESCRIPTIVE, NOT CONFIRMATORY.** Percentile thresholds carry NO
multiplicity control: there is no family, no correction, and no p-value here. It
must not be reported as a confirmatory claim.

WHY PERCENTILES AND NOT THE MULTIPLE-TESTING ROUTE -- the load-bearing choice.
The BH pass has a resolution floor at the pool size (gotcha 42,
`bh_resolution_floor.py`): below `k_min` tests at the finest attainable p-value
it rejects NOTHING, whatever the data say. At the unrestricted pool the
within-protein arm would sit under that floor, so a decomposition built on the
multiple-testing route would be **empty by construction** -- a measurement that
cannot produce a non-trivial answer. The percentile route has no family and no
minimum rejectable rank. The cost is stated rather than hidden: no multiplicity
control.

WHY THIS MODULE EXISTS INSTEAD OF A FLAG ON `concept_repro`.
`concept_repro.py` REFUSES to read `null_cmax_a` ("answers a different question
(localisation, not total grounding). Refusing to substitute it."). That refusal
is a correct guard on the HEADLINE and is deliberately left intact. This module
makes the substitution knowingly, under its own name, for a measure that is
explicitly not the headline -- so a reader cannot reach the substituted null by
passing a flag to the headline's own entry point.

THE READING, AND ITS BOUNDARY (BW C5). Under amendment G4's stated mechanism a
protein-selecting feature looks aligned under EVERY within-protein draw (the
concept stays on the protein it covers), so its within-protein null is high and
it does NOT clear that threshold; the same feature regresses toward chance under
the across-protein null and DOES clear that one. So:

    detected ACROSS + not detected WITHIN  -> protein-level candidate
    detected ACROSS + detected WITHIN      -> residue-level candidate

**The reading attaches to concepts this operation selects and to NOTHING ELSE.**
No claim is made or implied here about WHY the two nulls differ in level, why the
aligned count is small, or why the experimental supply is concentrated. Those are
mechanism claims and none is authorised. The pre-registered pejorative wording
for this population is NOT adopted; where the original clause is referenced it is
quoted as declared, and this measure's own reading is stated neutrally.

TWO LIMITATIONS INHERITED FROM THE CONSTRUCTION BEING MIRRORED, both established
rather than assumed (BW B1/B2):

  * **The permutation schedule is SHARED ACROSS SEEDS.** `draw_schedules` states
    it in terms ("ONE set, shared across every seed") and `shift_schedule` is
    stored `(n_shifts, n_proteins)` with no seed axis -- two independent handles,
    the same pair the across-protein case was established on. So twenty
    detections are NOT twenty independent confirmations, and the channel is not
    removable from stored draws.
  * **The within-protein shift preserves each concept's positive count exactly**
    on the unmasked stratum -- `shift_rows` is `p_start + (rows - p_start + s) %
    p_len`, a bijection on each protein's own row range. This is the one respect
    in which its thresholds are LESS biased than the across-protein arm's, whose
    permutation drifts prevalence one-directionally. A MASKED stratum's collision
    rule CAN drop positives; this module refuses a masked null artifact rather
    than carrying the caveat silently.

UNITS. Everything is in normalised gain `(AP - p)/(1 - p)`, the unit the null is
stored in, exactly as the mirrored construction uses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.analysis.artifact_identity import argmax_pool_mask
from src.analysis.concept_repro import (
    REFERENCE_PERCENTILE,
    detection_matrix,
    detection_rate,
    estimable_percentiles,
    null_thresholds,
    observed_best_ap,
    to_gain,
)

# The identity keys that must agree between the AP artifact and the null before
# anything is computed. Same list the mirrored construction asserts on; an ABSENT
# key is reported as absent and never treated as matching (gotcha 15).
SHARED_IDENTITY_KEYS = ("library_sha", "truncation_policy", "scored_row_set",
                        "n_rows", "argmax_pool_n", "n_concepts")

# The 2x2's four cells, named once so the artifact, the JSON and the prose cannot
# drift apart. Ordering is (across, within).
CELL_NAMES = {
    (True, True):  "residue_level_candidate",
    (True, False): "protein_level_candidate",
    (False, True): "within_only",
    (False, False): "not_detected_on_this_axis",
}


def _attrs(hf) -> dict:
    out = {}
    for k, v in hf.attrs.items():
        out[k] = v.item() if isinstance(v, np.generic) else (
            v.tolist() if isinstance(v, np.ndarray) else v)
    return out


def _labels(ds) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in ds[:]]


def threshold_relation(thr_within: np.ndarray,
                       thr_across: np.ndarray) -> dict:
    """WHICH CONCEPTS CAN REACH THE `within_only` CELL AT ALL -- computed BEFORE
    any detection, because it maps a cell of the outcome space in advance.

    `within_only` requires `gain > thr_within` and `gain <= thr_across`, which is
    satisfiable only where `thr_within < thr_across`. Where `thr_within >=
    thr_across` the cell is arithmetically unreachable for that concept AT EVERY
    observed gain -- so its emptiness there would be a deduction, not a finding.

    Reported as a COUNT of eligible concepts, never as a verdict: an eligible
    cell that comes back empty is a measurement, an ineligible one is arithmetic.
    """
    if thr_within.shape != thr_across.shape:
        raise ValueError(f"threshold shapes differ: {thr_within.shape} vs "
                         f"{thr_across.shape}")
    eligible = thr_within < thr_across
    return {
        "n_concepts": int(thr_within.size),
        "n_within_below_across": int(eligible.sum()),
        "n_within_at_or_above_across": int((~eligible).sum()),
        "share_within_at_or_above_across": float((~eligible).mean()),
        "within_only_cell_reachable_for_n_concepts": int(eligible.sum()),
        "eligible_mask": eligible,
    }


def ignorance_partition(best_gain: np.ndarray,
                        thr_within: np.ndarray,
                        thr_across: np.ndarray,
                        null_within: np.ndarray,
                        null_across: np.ndarray,
                        degenerate_mask: np.ndarray) -> dict:
    """EVERY IGNORANCE CASE IS ITS OWN CATEGORY WITH ITS OWN COUNT.

    None is folded into a denominator and none is counted as a negative finding
    (BW C3). Counts are reported even when zero -- a category absent from the
    report is indistinguishable from a category that was never checked.
    """
    nonfinite_gain = ~np.isfinite(best_gain)
    nonfinite_thr = (~np.isfinite(thr_within)) | (~np.isfinite(thr_across))
    # A threshold of exactly zero is not "an easy threshold": under STRICT
    # exceedance every positive gain clears it, so the concept is detected by
    # construction rather than by evidence. Counted separately, both arms.
    zero_thr_within = thr_within == 0.0
    zero_thr_across = thr_across == 0.0
    # The unmasked writer fills `null_cmax` with 0.0, so an all-zero draw row is
    # indistinguishable from a genuine zero maximum. Counted, never imputed.
    allzero_within = (null_within == 0.0).all(axis=(0, 1))
    allzero_across = (null_across == 0.0).all(axis=(0, 1))
    return {
        "n_concepts_nonfinite_threshold": int(nonfinite_thr.sum()),
        "n_cells_nonfinite_observed_gain": int(nonfinite_gain.sum()),
        "n_concepts_zero_threshold_within": int(zero_thr_within.sum()),
        "n_concepts_zero_threshold_across": int(zero_thr_across.sum()),
        "n_concepts_all_null_draws_zero_within": int(allzero_within.sum()),
        "n_concepts_all_null_draws_zero_across": int(allzero_across.sum()),
        "n_concepts_flagged_degenerate": int(degenerate_mask.sum()),
        "_masks": {
            "nonfinite_threshold": nonfinite_thr,
            "degenerate": degenerate_mask,
        },
    }


def two_by_two_cells(det_across: np.ndarray, det_within: np.ndarray) -> dict:
    """The 2x2 over the primitive the data actually has: the (seed, concept)
    detection cell. No seed-collapsing rule is needed at this level, so this is
    the enumeration that requires no choice (gotcha 39).

    All four cells are reported with an explicit count INCLUDING ZERO.
    """
    if det_across.shape != det_within.shape:
        raise ValueError(f"detection shapes differ: {det_across.shape} vs "
                         f"{det_within.shape}")
    if det_across.dtype != np.bool_ or det_within.dtype != np.bool_:
        raise ValueError("detection matrices must be boolean")
    counts = {}
    for (a, w), name in CELL_NAMES.items():
        counts[name] = int(((det_across == a) & (det_within == w)).sum())
    total = int(det_across.size)
    if sum(counts.values()) != total:
        raise SystemExit(
            f"FATAL: the four cells sum to {sum(counts.values())}, not {total}. "
            "A 2x2 over two booleans is exhaustive by construction, so this is "
            "an implementation defect, not a data outcome.")
    return {"cells": counts, "n_cells_total": total,
            "denominator": "(seed, concept) detection cells"}


def concept_level_two_by_two(d_across: np.ndarray, d_within: np.ndarray) -> dict:
    """The 2x2 on the CONCEPT unit, under the record's OWN seed reduction.

    `D(c) == 1` -- detected in ALL 20 anchor seeds -- is the reduction the
    committed concept-level headline already reports on ("75.25% of concepts
    detected in all 20 seeds"). It is used here rather than a fresh cutoff
    precisely so that NO cutoff is chosen anywhere in this construction: an
    invented seed-collapsing rule would be the tuning the measure is built to
    avoid.

    The full joint distribution of `(D_across, D_within)` is reported beside it,
    so nothing is hidden behind the reduction.
    """
    a, w = d_across == 1.0, d_within == 1.0
    counts = {name: int(((a == ka) & (w == kw)).sum())
              for (ka, kw), name in CELL_NAMES.items()}
    total = int(d_across.size)
    if sum(counts.values()) != total:
        raise SystemExit(f"FATAL: concept cells sum to {sum(counts.values())}, "
                         f"not {total}.")
    joint = {}
    for va in np.unique(d_across):
        for vw in np.unique(d_within):
            n = int(((d_across == va) & (d_within == vw)).sum())
            if n:
                joint[f"{va:.2f}|{vw:.2f}"] = n
    return {"reduction": "D(c) == 1 (detected in all 20 anchor seeds) -- the "
                         "reduction the committed concept-level headline reports",
            "cells": counts, "n_concepts": total,
            "joint_distribution_D_across_given_D_within": joint}


def run(ap_path: Path, null_path: Path, degenerate_path: Path | None) -> dict:
    with h5py.File(ap_path, "r") as hf:
        grp = hf["pooled"]
        ap = grp["ap"][:]
        concepts = _labels(grp["concepts"])
        prevalence = grp["prevalence"][:]
        n_positives = grp["n_positives"][:]
        in_pool_stored = grp["in_argmax_pool"][:].astype(bool)
        ap_seeds = _labels(hf["seeds"])
        ap_identity = _attrs(hf)

    in_pool = np.array(argmax_pool_mask(concepts), dtype=bool)
    if not np.array_equal(in_pool, in_pool_stored):
        raise SystemExit("FATAL: stored `in_argmax_pool` disagrees with "
                         "`argmax_pool_mask` recomputed from the concept names")

    with h5py.File(null_path, "r") as hf:
        for need in ("null_cmax_a", "null_cmax_b", "shift_schedule"):
            if need not in hf:
                raise SystemExit(
                    f"FATAL: the null artifact has no `{need}`. "
                    "`shift_schedule` is written only on a Null A run, so its "
                    "absence means `null_cmax_a` was not produced by the "
                    "within-protein construction and must not be substituted.")
        null_within = hf["null_cmax_a"][:]
        null_across = hf["null_cmax_b"][:]
        null_concepts = _labels(hf["concepts"])
        null_seeds = _labels(hf["seeds"])
        shift_sched_shape = hf["shift_schedule"].shape
        null_identity = _attrs(hf)

    # ---- positional guards: labels, elementwise, never by position (gotcha 29)
    pool_concepts = [c for c, k in zip(concepts, in_pool) if k]
    if pool_concepts != null_concepts:
        raise SystemExit("FATAL: concept labels disagree between AP pool and null")
    if ap_seeds != null_seeds:
        raise SystemExit("FATAL: seed labels disagree between AP and null")

    # ---- identity ---------------------------------------------------------
    identity_report = {}
    for k in SHARED_IDENTITY_KEYS:
        a = ap_identity.get(k, "<ABSENT>")
        b = null_identity.get(k, "<ABSENT>")
        agree = bool(a == b) and a != "<ABSENT>"
        identity_report[k] = {"ap": a, "null": b, "agree": agree}
    bad = [k for k, v in identity_report.items() if not v["agree"]]
    if bad:
        raise SystemExit(f"FATAL: identity keys disagree or are absent: {bad}")

    # ---- refuse a masked null: its collision rule can drop positives -------
    if null_identity.get("scored_row_set") != "label_store":
        raise SystemExit(
            "FATAL: this measure is specified on the unmasked (pooled) stratum, "
            "where the within-protein shift preserves each concept's positive "
            "count exactly. A masked stratum's collision rule CAN drop "
            "positives, which would make the two arms' thresholds not "
            "like-for-like. Refusing rather than carrying the caveat silently.")

    # ---- the measure ------------------------------------------------------
    best_gain = to_gain(observed_best_ap(ap[:, :, in_pool]), prevalence[in_pool])
    n_seeds, n_shifts, n_concepts = null_across.shape
    pooled_draws = n_seeds * n_shifts
    est = estimable_percentiles(pooled_draws)[REFERENCE_PERCENTILE]
    if not est["estimable"]:
        raise SystemExit(
            f"FATAL: the {REFERENCE_PERCENTILE}th percentile is not estimable "
            f"from {pooled_draws} draws.")

    thr_within = null_thresholds(null_within, REFERENCE_PERCENTILE, "pooled")
    thr_across = null_thresholds(null_across, REFERENCE_PERCENTILE, "pooled")

    degenerate_mask = np.zeros(n_concepts, dtype=bool)
    degenerate_names: list[str] = []
    if degenerate_path is not None:
        with h5py.File(degenerate_path, "r") as hf:
            if "degenerate_concepts" in hf:
                degenerate_names = _labels(hf["degenerate_concepts"])
        idx = {c: i for i, c in enumerate(pool_concepts)}
        for nm in degenerate_names:
            if nm in idx:
                degenerate_mask[idx[nm]] = True

    relation = threshold_relation(thr_within, thr_across)
    ignorance = ignorance_partition(best_gain, thr_within, thr_across,
                                    null_within, null_across, degenerate_mask)

    det_across = detection_matrix(best_gain, thr_across)
    det_within = detection_matrix(best_gain, thr_within)
    d_across = detection_rate(det_across)
    d_within = detection_rate(det_within)

    cell_2x2 = two_by_two_cells(det_across, det_within)
    concept_2x2 = concept_level_two_by_two(d_across, d_within)

    # The `within_only` cell restricted to concepts that could reach it. An
    # empty cell among ELIGIBLE concepts is a measurement; among ineligible ones
    # it is arithmetic, and the two must not be added together.
    elig = relation["eligible_mask"]
    within_only_eligible = int((det_within & ~det_across)[:, elig].sum())
    within_only_ineligible = int((det_within & ~det_across)[:, ~elig].sum())
    if within_only_ineligible:
        raise SystemExit(
            f"FATAL: {within_only_ineligible} (seed, concept) cells landed in "
            "`within_only` for concepts where `thr_within >= thr_across`, which "
            "is arithmetically impossible. This is an implementation defect.")

    out = {
        "specification": "PREREGISTRATION amendment BW -- DESCRIPTIVE "
                         "protein-versus-residue decomposition on the concept "
                         "unit. NOT confirmatory; NO multiplicity control.",
        "construction": {
            "mirrors": "the committed concept-level detection construction "
                       "(amendment AW §1, run as concept_repro)",
            "substitution": "null_cmax_a (WITHIN-protein circular shift) in "
                            "place of null_cmax_b (ACROSS-protein permutation)",
            "statistic": "best feature's gain per (seed, concept)",
            "reference_percentile": REFERENCE_PERCENTILE,
            "pooling": f"{pooled_draws} draws ({n_seeds} seeds x {n_shifts} "
                       f"shifts), pooled per concept",
            "estimability": est,
            "units": "normalised gain (AP - p)/(1 - p)",
            "multiplicity_control": "NONE -- percentile thresholds carry no "
                                    "family and no correction",
            "cutoff_chosen_anywhere": False,
        },
        "limitations": {
            "schedule_shared_across_seeds": {
                "established_from": "the writer's stated intent "
                                    "(draw_schedules: 'ONE set, shared across "
                                    "every seed') AND the array's rank",
                "shift_schedule_shape": list(shift_sched_shape),
                "has_seed_axis": False,
                "consequence": "the 20 detections are NOT 20 independent "
                               "confirmations; not removable from stored draws",
            },
            "positive_count_preservation": {
                "within_protein": "EXACT on this unmasked stratum -- the "
                                  "circular shift is a bijection on each "
                                  "protein's own row range",
                "across_protein": "NOT preserved; one-directional prevalence "
                                  "drift, median -1.83%, worst -15.85% "
                                  "(not corrected)",
            },
        },
        "identity": identity_report,
        "threshold_relation": {k: v for k, v in relation.items()
                               if k != "eligible_mask"},
        "ignorance_categories": {k: v for k, v in ignorance.items()
                                 if k != "_masks"},
        "cell_level_2x2": cell_2x2,
        "concept_level_2x2": concept_2x2,
        "within_only_detail": {
            "n_cells_among_eligible_concepts": within_only_eligible,
            "n_cells_among_ineligible_concepts": within_only_ineligible,
            "note": "an empty cell among eligible concepts is a measurement; "
                    "among ineligible concepts it is arithmetic",
        },
        "detection_rate_summary": {
            "across_protein_mean_D": float(d_across.mean()),
            "within_protein_mean_D": float(d_within.mean()),
            "across_protein_n_D_eq_1": int((d_across == 1).sum()),
            "within_protein_n_D_eq_1": int((d_within == 1).sum()),
            "across_protein_n_D_eq_0": int((d_across == 0).sum()),
            "within_protein_n_D_eq_0": int((d_within == 0).sum()),
            "n_concepts": int(n_concepts),
        },
        "source": {"ap": str(ap_path), "null": str(null_path),
                   "degenerate_flag": str(degenerate_path) if degenerate_path
                   else None},
        "_arrays": {
            "concepts": pool_concepts, "seeds": ap_seeds,
            "thr_within": thr_within, "thr_across": thr_across,
            "d_within": d_within, "d_across": d_across,
            "det_within": det_within, "det_across": det_across,
            "best_gain": best_gain, "n_positives": n_positives[in_pool],
            "degenerate_mask": degenerate_mask,
            "eligible_mask": elig,
            "ap_identity": ap_identity, "null_identity": null_identity,
        },
    }
    return out


def write_artifact(res: dict, out_h5: Path) -> None:
    """The per-concept classification, stored. NOTHING IS BACK-FILLED: keys this
    run does not itself establish are absent rather than inherited."""
    arr = res["_arrays"]
    per_concept_cell = np.empty(len(arr["concepts"]), dtype="S32")
    a1 = arr["d_across"] == 1.0
    w1 = arr["d_within"] == 1.0
    for i in range(len(arr["concepts"])):
        per_concept_cell[i] = CELL_NAMES[(bool(a1[i]), bool(w1[i]))].encode()

    with h5py.File(out_h5, "w") as hf:
        st = h5py.special_dtype(vlen=str)
        hf.create_dataset("concepts", data=np.array(arr["concepts"], dtype=object),
                          dtype=st)
        hf.create_dataset("seeds", data=np.array(arr["seeds"], dtype=object),
                          dtype=st)
        hf.create_dataset("threshold_within_protein", data=arr["thr_within"])
        hf.create_dataset("threshold_across_protein", data=arr["thr_across"])
        hf.create_dataset("D_within_protein", data=arr["d_within"])
        hf.create_dataset("D_across_protein", data=arr["d_across"])
        hf.create_dataset("detected_within_protein", data=arr["det_within"])
        hf.create_dataset("detected_across_protein", data=arr["det_across"])
        hf.create_dataset("best_gain", data=arr["best_gain"])
        hf.create_dataset("n_positives", data=arr["n_positives"])
        hf.create_dataset("flagged_degenerate", data=arr["degenerate_mask"])
        hf.create_dataset("within_only_eligible", data=arr["eligible_mask"])
        hf.create_dataset("concept_cell", data=per_concept_cell)
        hf.attrs["measurement"] = "protein_vs_residue_decomposition_BW"
        hf.attrs["specification"] = res["specification"]
        hf.attrs["reference_percentile"] = REFERENCE_PERCENTILE
        hf.attrs["multiplicity_control"] = "NONE"
        hf.attrs["confirmatory"] = False
        hf.attrs["concept_cell_reduction"] = "D(c) == 1 (all 20 anchor seeds)"
        hf.attrs["schedule_shared_across_seeds"] = True
        for k in SHARED_IDENTITY_KEYS:
            hf.attrs[k] = arr["null_identity"][k]
        hf.attrs["source_ap"] = res["source"]["ap"]
        hf.attrs["source_null"] = res["source"]["null"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ap", type=Path, required=True)
    p.add_argument("--null", type=Path, required=True)
    p.add_argument("--degenerate-flag", type=Path, default=None)
    p.add_argument("--out-h5", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    a = p.parse_args()

    res = run(a.ap, a.null, a.degenerate_flag)
    write_artifact(res, a.out_h5)
    payload = {k: v for k, v in res.items() if k != "_arrays"}
    a.out_json.write_text(json.dumps(payload, indent=1, default=str))

    print(json.dumps(payload["threshold_relation"], indent=1))
    print(json.dumps(payload["ignorance_categories"], indent=1))
    print(json.dumps(payload["cell_level_2x2"], indent=1))
    print(json.dumps(payload["concept_level_2x2"]["cells"], indent=1))
    print(json.dumps(payload["detection_rate_summary"], indent=1))
    print(f"\nwrote {a.out_h5}\nwrote {a.out_json}")


if __name__ == "__main__":
    main()
