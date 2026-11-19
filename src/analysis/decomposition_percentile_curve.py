"""THE PERCENTILE CURVE for the BW/BX protein-versus-residue decomposition.

WHY THIS EXISTS (PREREGISTRATION amendment BY, corrections entry 159). The
decomposition mirrors the committed concept-level detection construction in every
mechanic, and §AW1.5 attaches a REPORTING clause to that construction:

    "Report the detection rate as a curve over percentiles -- 50th, 75th, 90th,
    95th, 99th, 99.9th -- with the 95th percentile as the pre-declared
    reference. A conclusion counts only if it holds across the reported
    percentile range."

BW and BX are computed at the 95th ONLY. So the decomposition's conclusion had
not been held to the bar the clause it mirrors sets. This module supplies the
curve. **The mechanics are BW's, unchanged; only the percentile moves.**

NO NEW NULL RUN, AND NOTHING RECOMPUTED THAT WAS STORED. `best_gain` is read
from `protein_vs_residue_bw.h5`, where BW stored it in full; the null draws are
read from the same null artifact BW named in its own `source`. A percentile of a
stored draw array is a REDUCTION, which is why this needs no job of the kind
amendment BY does not authorise.

THE RANGE IS NOT RE-DECLARED HERE. `CURVE_PERCENTILES` is imported from
`concept_repro`, which is where AW1.5's list already lives as code. A second
copy of a committed list is a drift channel; there is one list.

ESTIMABILITY IS A PROPERTY OF THE SAMPLE, NOT A CONVENTION -- and it BINDS here.
At 400 pooled draws the 99.9th percentile expects 0.4 draws at or above it, so
it is NOT estimable and is EXCLUDED, marked excluded rather than silently
dropped. §AX2 settled exactly this for the across-protein arm ("the 99.9th is
dropped from the curve in both arms and is not interpolated or extrapolated");
this module reaches the same verdict by RUNNING the rule rather than by quoting
it.

BOTH ARMS MOVE TOGETHER. At each percentile both thresholds are recomputed, so
the two arms are never compared at different percentiles -- which would make a
difference between them partly a difference of cutoff.

DESCRIPTIVE, NOT CONFIRMATORY, INHERITED UNCHANGED. Percentile thresholds carry
no family, no correction and no p-value. NO PRIMARY SUMMARY IS DECLARED: §AW1.7
says "a single scalar summary must not be the headline", and the grid is
reported with every denominator beside its figure. NO MECHANISM READING IS
ATTACHED to any shape reported here; if the shape invites one, that reading is
not authorised by this module.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.analysis.concept_repro import (
    CURVE_PERCENTILES,
    REFERENCE_PERCENTILE,
    assert_labels_align,
    detection_matrix,
    detection_rate,
    estimable_percentiles,
    null_thresholds,
)
from src.sae.train import git_dirty, git_sha

# The pooling mode. §AX2 settled AW8's open item 2: the POOLED null is the
# reference, because it is the only one of the two under which the 99th
# percentile exists at all, and pooling is defensible for concepts (a concept has
# a real cross-seed identity) in a way it is not for feature indices (gotcha 18).
POOLING_MODE = "pooled"

# The qualitative conclusion this curve tests, written down HERE rather than
# derived from the numbers it will be checked against (gotcha 32b: a test's
# expectation must be written, not computed from the thing under test).
CONCLUSION = ("a substantial share of concepts detected under the across-protein "
              "null are NOT detected under the within-protein null")


def _labels(ds) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in ds[:]]


def _attrs(hf) -> dict:
    out = {}
    for k, v in hf.attrs.items():
        out[k] = v.tolist() if hasattr(v, "tolist") else v
    return out


def curve_grid(best_gain: np.ndarray,
               null_within: np.ndarray,
               null_across: np.ndarray,
               percentiles=CURVE_PERCENTILES) -> dict:
    """The full percentile grid for both arms.

    `best_gain` is (seed, concept); both nulls are (seed, shift, concept).

    Every percentile in `percentiles` appears in the output. A percentile the
    draw count cannot support is present with `estimable: False` and NO figures
    -- excluded, and visibly so, never absent and never interpolated.
    """
    if best_gain.ndim != 2:
        raise ValueError(f"best_gain must be (seed, concept), got {best_gain.shape}")
    for nm, arr in (("within", null_within), ("across", null_across)):
        if arr.ndim != 3:
            raise ValueError(f"null_{nm} must be (seed, shift, concept), got {arr.shape}")
        if arr.shape[0] != best_gain.shape[0] or arr.shape[2] != best_gain.shape[1]:
            raise ValueError(
                f"null_{nm} {arr.shape} does not share best_gain's seed and "
                f"concept axes {best_gain.shape}")

    n_seeds, n_shifts, n_concepts = null_across.shape
    pooled_draws = n_seeds * n_shifts
    est = estimable_percentiles(pooled_draws, percentiles)

    rows = []
    for q in percentiles:
        e = est[float(q)]
        row = {
            "percentile": float(q),
            "is_reference": bool(float(q) == float(REFERENCE_PERCENTILE)),
            "pooled_draws": int(pooled_draws),
            "expected_draws_at_or_above": e["expected_draws_at_or_above"],
            "estimable": e["estimable"],
        }
        if not e["estimable"]:
            row["excluded"] = True
            row["exclusion_reason"] = (
                f"NOT estimable from {pooled_draws} pooled draws: "
                f"{e['expected_draws_at_or_above']:.2f} draws expected at or "
                "above this quantile, so the estimate would be the sample "
                "maximum wearing a different label. Not interpolated, not "
                "extrapolated (AX2, applied to both arms).")
            rows.append(row)
            continue

        row["excluded"] = False
        thr_w = null_thresholds(null_within, float(q), POOLING_MODE)
        thr_a = null_thresholds(null_across, float(q), POOLING_MODE)
        det_w = detection_matrix(best_gain, thr_w)
        det_a = detection_matrix(best_gain, thr_a)
        d_w = detection_rate(det_w)
        d_a = detection_rate(det_a)

        # --- the CELL basis: (seed, concept) cells detected across-protein.
        n_cells_across = int(det_a.sum())
        n_cells_across_not_within = int((det_a & ~det_w).sum())
        # --- the CONCEPT basis at D_across == 1, the all-anchor across-detected
        #     set. Reported WITH its denominator, never as a bare share.
        a1 = d_a == 1.0
        n_concepts_a1 = int(a1.sum())
        n_a1_not_within_all = int((a1 & (d_w < 1.0)).sum())
        n_a1_not_within_any = int((a1 & (d_w == 0.0)).sum())

        row["arms"] = {
            "within_protein": {
                "mean_D": float(d_w.mean()),
                "n_D_eq_1": int((d_w == 1.0).sum()),
                "n_D_eq_0": int((d_w == 0.0).sum()),
                "interior_fraction": float(((d_w > 0.1) & (d_w < 0.9)).mean()),
                "median_threshold": float(np.median(thr_w)),
                "denominator_concepts": int(n_concepts),
            },
            "across_protein": {
                "mean_D": float(d_a.mean()),
                "n_D_eq_1": int((d_a == 1.0).sum()),
                "n_D_eq_0": int((d_a == 0.0).sum()),
                "interior_fraction": float(((d_a > 0.1) & (d_a < 0.9)).mean()),
                "median_threshold": float(np.median(thr_a)),
                "denominator_concepts": int(n_concepts),
            },
        }
        # Each share carries its own denominator and its own criterion IN THE
        # SAME OBJECT, so neither can be quoted without them (BX's finding).
        row["cell_basis"] = {
            "criterion": "(seed, concept) cells detected under the across-protein "
                         "null; share of those NOT detected under within-protein",
            "denominator_cells_detected_across": n_cells_across,
            "n_cells_across_not_within": n_cells_across_not_within,
            "share": (float(n_cells_across_not_within / n_cells_across)
                      if n_cells_across else None),
            "share_is_none_because": (None if n_cells_across
                                      else "denominator is zero"),
        }
        row["concept_basis_D_across_eq_1"] = {
            "criterion": "concepts with D_across == 1 (all 20 anchors); share "
                         "NOT detected in all 20 anchors under within-protein",
            "denominator_concepts_D_across_eq_1": n_concepts_a1,
            "n_not_within_in_all_anchors": n_a1_not_within_all,
            "share_not_within_all": (float(n_a1_not_within_all / n_concepts_a1)
                                     if n_concepts_a1 else None),
            "n_not_within_in_any_anchor": n_a1_not_within_any,
            "share_not_within_any": (float(n_a1_not_within_any / n_concepts_a1)
                                     if n_concepts_a1 else None),
            "share_is_none_because": (None if n_concepts_a1
                                      else "denominator is zero"),
        }
        rows.append(row)
    return {"rows": rows, "pooled_draws": int(pooled_draws),
            "n_concepts": int(n_concepts), "n_seeds": int(n_seeds),
            "n_shifts": int(n_shifts)}


def conclusion_across_range(grid: dict,
                            substantial_share: float = 0.10) -> dict:
    """Does the qualitative conclusion hold at EVERY estimable percentile?

    AW1.5: "a conclusion counts only if it holds across the reported percentile
    range." So the verdict is a conjunction over the estimable rows, and the
    per-percentile values are returned beside it so a reader re-derives rather
    than accepts it (gotcha 20).

    `substantial_share` is the reading of "substantial" in CONCLUSION and is
    stated as an argument rather than buried, because it is a READING of a word
    and not a measured quantity. THE PER-PERCENTILE SHARES ARE REPORTED
    REGARDLESS, so the verdict can be re-derived at any other reading.

    THE VERDICT IS REPORTED ON BOTH BASES, NEITHER PRIMARY. BX established that
    the cell basis (a share of across-detected cells) and the concept basis (a
    share of concepts at `D_across == 1`) are NOT comparable -- routed through
    `ratio_rule_diff` they return `NEITHER_RATIO_NOR_DIFFERENCE_IS_DEFINED`,
    `SCALES_DIFFER`. So a single verdict computed on one of them would pick a
    basis silently, which is the failure BX exists to have fixed. Both verdicts
    are returned and `basis_verdicts_agree` records whether they coincide.

    THE OUTCOME SPACE IS TOTAL over the rows (gotcha 24) and its branches are
    enumerated over their CAUSES (gotcha 30): HOLDS_ACROSS_RANGE,
    FAILS_AT_SOME_PERCENTILE, and NO_ESTIMABLE_PERCENTILE -- the last being a
    distinct cause with a distinct reading (the curve could not be computed at
    all), never reported as a failure of the conclusion.
    """
    est = [r for r in grid["rows"] if r.get("estimable")]
    if not est:
        no_est = {
            "verdict": "NO_ESTIMABLE_PERCENTILE",
            "failing_percentiles": [],
            "shares_by_percentile": {},
            "reading": "the curve has no estimable percentile, so the "
                       "conclusion is neither supported nor contradicted here. "
                       "This is NOT a failure of the conclusion.",
        }
        return {
            "conclusion": CONCLUSION,
            "substantial_share_reading": substantial_share,
            "substantial_is_a_READING_not_a_measurement": True,
            "estimable_percentiles": [],
            "by_basis": {"cell": dict(no_est, basis="cells detected under the "
                                                    "across-protein null"),
                         "concept": dict(no_est,
                                         basis="concepts with D_across == 1")},
            "basis_verdicts_agree": True,
            "no_basis_is_primary": True,
            "why_no_basis_is_primary":
                "not reached: no percentile is estimable, so neither basis has "
                "a figure to compare.",
            "no_primary_declared": True,
            "per_percentile": [],
        }
    per = []
    for r in est:
        cell = r["cell_basis"]["share"]
        conc = r["concept_basis_D_across_eq_1"]["share_not_within_all"]
        per.append({
            "percentile": r["percentile"],
            "is_reference": r["is_reference"],
            "cell_share_across_not_within": cell,
            "cell_denominator": r["cell_basis"]["denominator_cells_detected_across"],
            "concept_share_not_within_all": conc,
            "concept_denominator":
                r["concept_basis_D_across_eq_1"]["denominator_concepts_D_across_eq_1"],
            "cell_share_is_substantial": (cell is not None
                                          and cell >= substantial_share),
            "concept_share_is_substantial": (conc is not None
                                             and conc >= substantial_share),
        })

    def _verdict(flag_key: str, share_key: str, basis: str) -> dict:
        holds = all(p[flag_key] for p in per)
        failing = [p["percentile"] for p in per if not p[flag_key]]
        return {
            "basis": basis,
            "verdict": ("HOLDS_ACROSS_RANGE" if holds
                        else "FAILS_AT_SOME_PERCENTILE"),
            "failing_percentiles": failing,
            "shares_by_percentile": {str(p["percentile"]): p[share_key]
                                     for p in per},
            "reading": ("the conclusion holds at every estimable percentile in "
                        "AW1.5's range on this basis, so it meets the clause's "
                        "bar here"
                        if holds else
                        "the conclusion does NOT hold at every estimable "
                        "percentile on this basis; AW1.5 says a conclusion "
                        "counts only if it holds across the range, so the "
                        "reference percentile must not be reported alone"),
        }

    cell_v = _verdict("cell_share_is_substantial",
                      "cell_share_across_not_within",
                      "cells detected under the across-protein null")
    conc_v = _verdict("concept_share_is_substantial",
                      "concept_share_not_within_all",
                      "concepts with D_across == 1")
    return {
        "conclusion": CONCLUSION,
        "substantial_share_reading": substantial_share,
        "substantial_is_a_READING_not_a_measurement": True,
        "estimable_percentiles": [p["percentile"] for p in per],
        "by_basis": {"cell": cell_v, "concept": conc_v},
        "basis_verdicts_agree": bool(cell_v["verdict"] == conc_v["verdict"]),
        "no_basis_is_primary": True,
        "why_no_basis_is_primary":
            "BX: the two bases return NEITHER_RATIO_NOR_DIFFERENCE_IS_DEFINED "
            "against each other (SCALES_DIFFER -- a share of concepts against a "
            "share of cells). A single verdict would pick one silently.",
        "no_primary_declared": True,
        "per_percentile": per,
    }


def run(decomp_h5: Path, null_path: Path) -> dict:
    with h5py.File(decomp_h5, "r") as hf:
        for need in ("best_gain", "concepts", "seeds"):
            if need not in hf:
                raise SystemExit(
                    f"FATAL: `{need}` absent from {decomp_h5}. This module is a "
                    "reduction of the stored decomposition and cannot "
                    "reconstruct it.")
        best_gain = hf["best_gain"][:]
        d_concepts = _labels(hf["concepts"])
        d_seeds = _labels(hf["seeds"])
        d_attrs = _attrs(hf)
        stored_thr_within = hf["threshold_within_protein"][:]
        stored_thr_across = hf["threshold_across_protein"][:]
        stored_d_within = hf["D_within_protein"][:]
        stored_d_across = hf["D_across_protein"][:]

    with h5py.File(null_path, "r") as hf:
        for need in ("null_cmax_a", "null_cmax_b", "shift_schedule"):
            if need not in hf:
                raise SystemExit(
                    f"FATAL: the null artifact has no `{need}`. `shift_schedule` "
                    "is written only on a Null A run, so its absence means "
                    "`null_cmax_a` was not produced by the within-protein "
                    "construction and must not be substituted.")
        null_within = hf["null_cmax_a"][:]
        null_across = hf["null_cmax_b"][:]
        n_concepts_null = _labels(hf["concepts"])
        n_seeds_null = _labels(hf["seeds"])
        null_attrs = _attrs(hf)

    # gotcha 29: align on LABELS, elementwise, never by position.
    assert_labels_align("concept labels (decomposition vs null)",
                        d_concepts, n_concepts_null)
    assert_labels_align("seed labels (decomposition vs null)",
                        d_seeds, n_seeds_null)

    grid = curve_grid(best_gain, null_within, null_across)
    verdict = conclusion_across_range(grid)

    # ---- THE REFERENCE ROW MUST REPRODUCE THE STORED 95th-PERCENTILE ARTIFACT.
    # A curve that disagrees with BW at the one percentile BW computed is a
    # defect in the curve, not a new result -- so it is asserted, not noted.
    ref = [r for r in grid["rows"]
           if r["percentile"] == float(REFERENCE_PERCENTILE)][0]
    thr_w = null_thresholds(null_within, REFERENCE_PERCENTILE, POOLING_MODE)
    thr_a = null_thresholds(null_across, REFERENCE_PERCENTILE, POOLING_MODE)
    d_w = detection_rate(detection_matrix(best_gain, thr_w))
    d_a = detection_rate(detection_matrix(best_gain, thr_a))
    recon = {
        "threshold_within_max_abs_delta": float(np.abs(thr_w - stored_thr_within).max()),
        "threshold_across_max_abs_delta": float(np.abs(thr_a - stored_thr_across).max()),
        "D_within_max_abs_delta": float(np.abs(d_w - stored_d_within).max()),
        "D_across_max_abs_delta": float(np.abs(d_a - stored_d_across).max()),
    }
    worst = max(recon.values())
    recon["max_abs_delta_over_all_four"] = worst
    recon["bit_identical"] = bool(worst == 0.0)
    if worst != 0.0:
        raise SystemExit(
            "FATAL: the curve's 95th-percentile row does not reproduce the "
            f"stored BW artifact (worst |delta| = {worst:.3e}). The curve is a "
            "reduction of the same stored arrays by the same helpers, so a "
            "disagreement is an implementation defect in this module, not a "
            f"finding. Detail: {recon}")

    return {
        "specification": "PREREGISTRATION amendment BY -- the percentile curve "
                         "AW1.5 requires of the construction the BW "
                         "decomposition mirrors. NOT confirmatory; NO "
                         "multiplicity control; NO primary summary declared.",
        "standing": {
            "descriptive_not_confirmatory": True,
            "multiplicity_control": "NONE -- percentile thresholds carry no "
                                    "family and no correction",
            "primary_summary_declared": False,
            "primary_summary_forbidden_by": "AW1.7 -- 'a single scalar summary "
                                            "must not be the headline'",
            "mechanism_reading_attached": False,
            "new_null_run": False,
            "computation": "a REDUCTION of stored draws; no threshold, cutoff "
                           "or null was introduced",
        },
        "range": {
            "source_clause": "AW1.5",
            "percentiles_specified": [float(q) for q in CURVE_PERCENTILES],
            "reference_percentile": float(REFERENCE_PERCENTILE),
            "clause_text": "Report the detection rate as a curve over "
                           "percentiles -- 50th, 75th, 90th, 95th, 99th, "
                           "99.9th -- with the 95th percentile as the "
                           "pre-declared reference. A conclusion counts only "
                           "if it holds across the reported percentile range.",
            "pooling_mode": POOLING_MODE,
            "pooling_settled_by": "AX2 (AW8 open item 2): the pooled null is "
                                  "the reference",
            "estimable": [r["percentile"] for r in grid["rows"] if r["estimable"]],
            "excluded_not_estimable": [r["percentile"] for r in grid["rows"]
                                       if not r["estimable"]],
        },
        "reference_row_reproduces_stored_artifact": recon,
        "grid": grid,
        "conclusion_across_range": verdict,
        "limitations_inherited": {
            "schedule_shared_across_seeds": bool(
                d_attrs.get("schedule_shared_across_seeds", True)),
            "consequence": "the 20 detections are NOT 20 independent "
                           "confirmations; not removable from stored draws",
            "within_protein_preserves_positive_count": "EXACT (a circular "
                                                       "shift is a bijection)",
            "across_protein_prevalence_drift": "NOT preserved; median -1.83%, "
                                               "worst -15.85% (not corrected)",
        },
        "source": {"decomposition": str(decomp_h5), "null": str(null_path)},
        "identity": {
            "decomposition_attrs": {k: v for k, v in d_attrs.items()
                                    if k in ("library_sha", "scored_row_set",
                                             "truncation_policy", "n_concepts",
                                             "measurement",
                                             "reference_percentile")},
            "null_attrs": {k: v for k, v in null_attrs.items()
                           if k in ("library_sha", "scored_row_set",
                                    "truncation_policy", "null_type",
                                    "n_shifts")},
            "git_sha": git_sha(),
            "git_dirty": git_dirty(),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decomposition", required=True, type=Path,
                   help="protein_vs_residue_bw.h5 -- stores best_gain")
    p.add_argument("--null", required=True, type=Path,
                   help="the null artifact BW named in its own `source`")
    p.add_argument("--out-json", required=True, type=Path)
    a = p.parse_args()

    res = run(a.decomposition, a.null)
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(res, indent=1, sort_keys=True))

    print("=" * 74)
    print("THE PERCENTILE CURVE (AW1.5) FOR THE WITHIN-PROTEIN ARM -- amendment BY")
    print("=" * 74)
    r = res["range"]
    print(f"range from {r['source_clause']}: "
          f"{', '.join(str(q) for q in r['percentiles_specified'])}")
    print(f"pooling: {r['pooling_mode']} ({res['grid']['pooled_draws']} draws)")
    print(f"estimable: {r['estimable']}")
    print(f"EXCLUDED (not estimable): {r['excluded_not_estimable']}")
    print(f"reference row reproduces stored BW artifact: "
          f"{res['reference_row_reproduces_stored_artifact']['bit_identical']} "
          f"(worst |delta| = "
          f"{res['reference_row_reproduces_stored_artifact']['max_abs_delta_over_all_four']:.3e})")
    print()
    print("  pct | mean D within | mean D across | cell share across&~within (denom) "
          "| concept share ~within-all (denom)")
    for row in res["grid"]["rows"]:
        if not row["estimable"]:
            print(f"  {row['percentile']:>5} | NOT ESTIMABLE -- excluded "
                  f"({row['expected_draws_at_or_above']:.2f} draws expected above)")
            continue
        cb, kb = row["cell_basis"], row["concept_basis_D_across_eq_1"]
        star = " *" if row["is_reference"] else "  "
        print(f"  {row['percentile']:>5}{star}| "
              f"{row['arms']['within_protein']['mean_D']:>13.4f} | "
              f"{row['arms']['across_protein']['mean_D']:>13.4f} | "
              f"{cb['share']:.4f} ({cb['n_cells_across_not_within']}/"
              f"{cb['denominator_cells_detected_across']}) | "
              f"{kb['share_not_within_all']:.4f} "
              f"({kb['n_not_within_in_all_anchors']}/"
              f"{kb['denominator_concepts_D_across_eq_1']})")
    print("  (* = the pre-declared reference percentile)")
    print()
    v = res["conclusion_across_range"]
    print(f"CONCLUSION TESTED: {v['conclusion']}")
    print(f"reading of 'substantial': share >= {v['substantial_share_reading']} "
          "(a READING of a word, not a measured quantity)")
    for key in ("cell", "concept"):
        b = v["by_basis"][key]
        print(f"  [{key:>7} basis: {b['basis']}]")
        print(f"      VERDICT: {b['verdict']}")
        if b["failing_percentiles"]:
            print(f"      failing percentiles: {b['failing_percentiles']}")
        print(f"      {b['reading']}")
    print(f"  the two bases AGREE: {v['basis_verdicts_agree']}")
    print(f"  NEITHER BASIS IS PRIMARY -- {v['why_no_basis_is_primary']}")
    print()
    print("NO PRIMARY SUMMARY IS DECLARED (AW1.7). NO MECHANISM READING IS "
          "ATTACHED.")
    print(f"written: {a.out_json}")


if __name__ == "__main__":
    main()
