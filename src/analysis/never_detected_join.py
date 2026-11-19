"""A DESCRIPTIVE JOIN over the concepts no seed detected (amendment BY, Part D).

WHAT THIS IS FOR. The record states that never-detected status **conflates true
model blindness with annotation sparsity**. External biological input on that may
not arrive. This assembles what is already computable from artifacts on disk, so
the paper can say something defensible either way.

>>> THIS JOIN DOES NOT SEPARATE MODEL BLINDNESS FROM ANNOTATION SPARSITY. <<<
Separating them needs a new instrument, which amendment BY does not authorise.
What it CAN do is show whether the never-detected set is overwhelmingly sparse --
in which case the honest sentence follows -- or whether it is not, in which case
the question stays open and the paper says so. That statement is carried in the
artifact's own attributes, not only here, because an output that travels without
its boundary acquires one it was never given.

STANDING, carried in the artifact's attributes exactly as the decomposition's is:
COMPUTED BEFORE DECLARED, DESCRIPTIVE, NOT CONFIRMATORY, NO MULTIPLICITY
CONTROL. Not reportable as a finding until a declaration governs it.

WHAT IS JOINED, and every one is READ, never recomputed:
  * `D(c)` at the 95th-percentile reference -- from the committed concept-level
    headline artifact. "Never detected" is `D == 0`: no anchor seed's best
    feature exceeded that concept's own null threshold.
  * annotation density -- `n_annotated_residues` and `prevalence`, from the same
    per-concept block.
  * experimental support -- whether the concept carries positives on the
    experimental evidence axis, and how many.
  * experimental-stratum membership -- whether it is one of the 218 concepts in
    the experimental argmax pool. NOT the same as carrying experimental
    positives: the pool applies a retention floor, so a concept can carry
    experimental residues and still be outside it. The two columns are reported
    separately for exactly that reason.
  * at-floor supply -- the count of features whose p-value sits at the
    resolution floor `1/(n_finite+1)`, per (seed, arm), from the floor
    concentration artifact. This is the one column that speaks to whether a
    concept COULD have been detected at all by the multiple-testing route.

NO MECHANISM CLAIM. No statement about WHY any concept is never detected. No
threshold, no scoring, and no correlation quoted as a verdict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.analysis.concept_repro import REFERENCE_PERCENTILE, assert_labels_align
from src.sae.train import git_dirty, git_sha

NEVER_DETECTED_CRITERION = "D(c) == 0 at the 95th-percentile pooled reference"


def _labels(ds) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in ds[:]]


def summarise_column(values: np.ndarray, name: str) -> dict:
    """A column's distribution WITH its denominator. Never a bare statistic."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"column": name, "denominator": 0,
                "all_values_non_finite": True,
                "note": "no finite values -- no summary is defined, and this is "
                        "reported rather than rendered as a zero"}
    return {
        "column": name,
        "denominator": int(v.size),
        "min": float(v.min()), "p25": float(np.percentile(v, 25)),
        "median": float(np.median(v)), "p75": float(np.percentile(v, 75)),
        "max": float(v.max()), "mean": float(v.mean()),
    }


def build(repro_json: Path, floor_h5: Path, ap_exp_h5: Path) -> dict:
    # ---------------- the never-detected set, from the committed headline -----
    repro = json.loads(Path(repro_json).read_text())
    if float(repro.get("reference_percentile", -1)) != float(REFERENCE_PERCENTILE):
        raise SystemExit(
            f"FATAL: the headline artifact's reference percentile is "
            f"{repro.get('reference_percentile')}, not {REFERENCE_PERCENTILE}. "
            "The never-detected set is defined AT the reference, so a different "
            "percentile would make this a different population under the same "
            "name.")
    per = repro["per_concept"]
    concepts = [c["concept"] for c in per]
    d_vals = np.array([float(c["D"]) for c in per])
    never = d_vals == 0.0
    detected = ~never

    # ---------------- at-floor supply ----------------------------------------
    with h5py.File(floor_h5, "r") as hf:
        g = hf["pooled"]
        floor_concepts = _labels(g["concepts"])
        at_floor_a = g["at_floor_a"][:]         # (seed, concept)
        at_floor_b = g["at_floor_b"][:]
        floor_npos = g["n_positives"][:]
        floor_attrs = {k: (v.tolist() if hasattr(v, "tolist") else v)
                       for k, v in hf.attrs.items()}
    # gotcha 29/27: align on LABELS elementwise, never by position.
    assert_labels_align("concept labels (headline vs floor artifact)",
                        concepts, floor_concepts)

    # ---------------- the experimental evidence axis --------------------------
    with h5py.File(ap_exp_h5, "r") as hf:
        ge = hf["experimental"]
        exp_concepts = _labels(ge["concepts"])
        exp_npos = ge["n_positives"][:]
        exp_in_pool = ge["in_argmax_pool"][:]
    exp_npos_by_name = dict(zip(exp_concepts, exp_npos.tolist()))
    exp_pool_by_name = dict(zip(exp_concepts, exp_in_pool.tolist()))

    # ---------------- the joined table ---------------------------------------
    rows = []
    for i, c in enumerate(per):
        name = c["concept"]
        n_exp = exp_npos_by_name.get(name)
        rows.append({
            "concept": name,
            "D": float(c["D"]),
            "never_detected": bool(never[i]),
            "n_annotated_residues": int(c["n_annotated_residues"]),
            "prevalence": float(c["prevalence"]),
            "mean_A": float(c["mean_A"]),
            "null_threshold_p95": float(c["null_threshold_p95"]),
            # Experimental support and stratum membership are DIFFERENT columns.
            "on_experimental_axis": name in exp_npos_by_name,
            "n_experimental_positives": (int(n_exp) if n_exp is not None
                                         else None),
            "carries_experimental_support": bool(n_exp is not None and n_exp > 0),
            "in_experimental_argmax_pool": bool(exp_pool_by_name.get(name, False)),
            # at-floor supply, both null arms, summed over the 20 anchor seeds
            "at_floor_a_total_over_seeds": int(at_floor_a[:, i].sum()),
            "at_floor_b_total_over_seeds": int(at_floor_b[:, i].sum()),
            "at_floor_a_mean_per_seed": float(at_floor_a[:, i].mean()),
            "at_floor_b_mean_per_seed": float(at_floor_b[:, i].mean()),
            "n_positives_floor_artifact": int(floor_npos[i]),
        })

    nd_rows = [r for r in rows if r["never_detected"]]
    det_rows = [r for r in rows if not r["never_detected"]]

    # ---- the experimental subset, ESTABLISHED FROM THE ARTIFACTS -------------
    nd_exp = [r for r in nd_rows if r["carries_experimental_support"]]
    nd_pool = [r for r in nd_rows if r["in_experimental_argmax_pool"]]

    def dist(rs, col):
        return summarise_column(np.array([r[col] for r in rs
                                          if r[col] is not None]), col)

    COLS = ("n_annotated_residues", "prevalence", "n_experimental_positives",
            "at_floor_a_total_over_seeds", "at_floor_b_total_over_seeds")

    return {
        "specification": "PREREGISTRATION amendment BY Part D -- a DESCRIPTIVE "
                         "join over the never-detected concepts.",
        "what_this_does_NOT_do": {
            "separates_model_blindness_from_annotation_sparsity": False,
            "statement": "THIS JOIN DOES NOT SEPARATE TRUE MODEL BLINDNESS FROM "
                         "ANNOTATION SPARSITY. Every column here is consistent "
                         "with either. Separating them requires a new "
                         "instrument, which is not authorised by amendment BY.",
            "what_it_can_show": "whether the never-detected set is "
                                "overwhelmingly sparse (in which case the "
                                "honest sentence follows) or is not (in which "
                                "case the question stays open and the paper "
                                "says so)",
        },
        "standing": {
            "computed_before_declared": True,
            "descriptive_not_confirmatory": True,
            "multiplicity_control": "NONE",
            "mechanism_claim": "NONE -- no statement about why any concept is "
                               "never detected",
            "reportable_as_a_finding": False,
            "reportable_when": "a declaration governs it; the standing "
                               "convention for a quantity determined from data "
                               "in hand applies",
        },
        "definitions": {
            "never_detected": NEVER_DETECTED_CRITERION,
            "carries_experimental_support": "n_experimental_positives > 0 on "
                                            "the experimental evidence axis",
            "in_experimental_argmax_pool": "one of the 218 concepts retained in "
                                           "the experimental argmax pool -- NOT "
                                           "the same as carrying experimental "
                                           "positives, because the pool applies "
                                           "a retention floor",
            "at_floor": floor_attrs.get("at_floor_definition", "<ABSENT>"),
        },
        "counts": {
            "n_concepts_total": len(rows),
            "n_never_detected": len(nd_rows),
            "n_detected": len(det_rows),
            "n_never_detected_carrying_experimental_support": len(nd_exp),
            "n_never_detected_in_experimental_argmax_pool": len(nd_pool),
            "n_never_detected_on_experimental_axis_with_zero_positives":
                sum(1 for r in nd_rows if r["on_experimental_axis"]
                    and not r["carries_experimental_support"]),
            "n_never_detected_absent_from_experimental_axis":
                sum(1 for r in nd_rows if not r["on_experimental_axis"]),
        },
        "distributions": {
            "never_detected": {c: dist(nd_rows, c) for c in COLS},
            "detected": {c: dist(det_rows, c) for c in COLS},
        },
        "joined_table_never_detected": sorted(
            nd_rows, key=lambda r: r["n_annotated_residues"]),
        "source": {
            "concept_repro": str(repro_json),
            "floor_concentration": str(floor_h5),
            "ap_experimental": str(ap_exp_h5),
        },
        "identity": {
            "reference_percentile": float(REFERENCE_PERCENTILE),
            "headline_identity": repro.get("identity", {}),
            "floor_artifact_git_sha": floor_attrs.get("git_sha", "<ABSENT>"),
            "git_sha": git_sha(),
            "git_dirty": git_dirty(),
        },
        "_arrays": {"rows": rows},
    }


def write_artifact(res: dict, out_h5: Path) -> None:
    """The joined table, stored. NOTHING BACK-FILLED: a key this run does not
    itself establish is absent rather than inherited."""
    rows = res["_arrays"]["rows"]
    st = h5py.special_dtype(vlen=str)
    with h5py.File(out_h5, "w") as hf:
        hf.create_dataset("concepts",
                          data=np.array([r["concept"] for r in rows], dtype=object),
                          dtype=st)
        for col, dt in (("D", "f8"), ("never_detected", "?"),
                        ("n_annotated_residues", "i8"), ("prevalence", "f8"),
                        ("mean_A", "f8"), ("null_threshold_p95", "f8"),
                        ("on_experimental_axis", "?"),
                        ("carries_experimental_support", "?"),
                        ("in_experimental_argmax_pool", "?"),
                        ("at_floor_a_total_over_seeds", "i8"),
                        ("at_floor_b_total_over_seeds", "i8"),
                        ("n_positives_floor_artifact", "i8")):
            hf.create_dataset(col, data=np.array([r[col] for r in rows], dtype=dt))
        # -1 encodes "not on the experimental axis"; the sentinel is DOCUMENTED
        # in an attr rather than left for a reader to infer from a magic number.
        hf.create_dataset("n_experimental_positives", data=np.array(
            [(-1 if r["n_experimental_positives"] is None
              else r["n_experimental_positives"]) for r in rows], dtype="i8"))
        hf.attrs["n_experimental_positives_sentinel"] = (
            "-1 means the concept is NOT on the experimental axis at all, which "
            "is a different statement from carrying zero experimental positives")
        hf.attrs["measurement"] = "never_detected_join_BY"
        hf.attrs["specification"] = res["specification"]
        hf.attrs["never_detected_criterion"] = NEVER_DETECTED_CRITERION
        hf.attrs["reference_percentile"] = float(REFERENCE_PERCENTILE)
        hf.attrs["separates_blindness_from_sparsity"] = False
        hf.attrs["does_NOT_separate"] = res["what_this_does_NOT_do"]["statement"]
        hf.attrs["standing"] = "COMPUTED_BEFORE_DECLARED"
        hf.attrs["descriptive_not_confirmatory"] = True
        hf.attrs["multiplicity_control"] = "NONE"
        hf.attrs["mechanism_claim"] = "NONE"
        hf.attrs["reportable_as_a_finding"] = False
        hf.attrs["git_sha"] = res["identity"]["git_sha"]
        hf.attrs["git_dirty"] = res["identity"]["git_dirty"]
        for k, v in res["source"].items():
            hf.attrs[f"source_{k}"] = v


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concept-repro", required=True, type=Path)
    p.add_argument("--floor", required=True, type=Path)
    p.add_argument("--ap-experimental", required=True, type=Path)
    p.add_argument("--out-json", required=True, type=Path)
    p.add_argument("--out-h5", required=True, type=Path)
    a = p.parse_args()

    res = build(a.concept_repro, a.floor, a.ap_experimental)
    out = {k: v for k, v in res.items() if k != "_arrays"}
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(out, indent=1, sort_keys=True))
    write_artifact(res, a.out_h5)

    c = res["counts"]
    print("=" * 78)
    print("THE NEVER-DETECTED CONCEPTS -- A DESCRIPTIVE JOIN (amendment BY, Part D)")
    print("=" * 78)
    print(f"criterion: {NEVER_DETECTED_CRITERION}")
    print(f"  concepts total                              : {c['n_concepts_total']}")
    print(f"  NEVER detected (D == 0)                     : {c['n_never_detected']}")
    print(f"  ... carrying experimental support           : "
          f"{c['n_never_detected_carrying_experimental_support']}")
    print(f"  ... in the experimental argmax pool         : "
          f"{c['n_never_detected_in_experimental_argmax_pool']}")
    print(f"  ... on the experimental axis, zero positives: "
          f"{c['n_never_detected_on_experimental_axis_with_zero_positives']}")
    print(f"  ... absent from the experimental axis       : "
          f"{c['n_never_detected_absent_from_experimental_axis']}")
    print()
    print("DISTRIBUTIONS, never-detected against detected, each with its denominator:")
    for col in ("n_annotated_residues", "prevalence",
                "at_floor_a_total_over_seeds", "at_floor_b_total_over_seeds"):
        nd = res["distributions"]["never_detected"][col]
        dt = res["distributions"]["detected"][col]
        print(f"  {col}")
        for lab, s in (("never", nd), ("detected", dt)):
            if s.get("all_values_non_finite"):
                print(f"    {lab:9s} n={s['denominator']:4d}  <no finite values>")
            else:
                print(f"    {lab:9s} n={s['denominator']:4d}  "
                      f"min={s['min']:.6g}  median={s['median']:.6g}  "
                      f"max={s['max']:.6g}")
    print()
    print("THE JOINED TABLE (never-detected, ascending annotated residues):")
    print(f"  {'concept':44s} {'n_res':>7s} {'prev':>10s} {'n_exp':>7s} "
          f"{'pool':>5s} {'floorA':>7s} {'floorB':>7s}")
    for r in res["joined_table_never_detected"]:
        ne = "-" if r["n_experimental_positives"] is None \
            else str(r["n_experimental_positives"])
        print(f"  {r['concept'][:44]:44s} {r['n_annotated_residues']:7d} "
              f"{r['prevalence']:10.3e} {ne:>7s} "
              f"{'Y' if r['in_experimental_argmax_pool'] else 'n':>5s} "
              f"{r['at_floor_a_total_over_seeds']:7d} "
              f"{r['at_floor_b_total_over_seeds']:7d}")
    print()
    print("!! THIS JOIN DOES NOT SEPARATE MODEL BLINDNESS FROM ANNOTATION")
    print("!! SPARSITY. Every column is consistent with either. Separating them")
    print("!! needs a new instrument, not authorised here.")
    print("standing: COMPUTED BEFORE DECLARED, descriptive, no multiplicity")
    print("control, no mechanism claim, NOT reportable as a finding.")
    print(f"written: {a.out_json}")
    print(f"written: {a.out_h5}")


if __name__ == "__main__":
    main()
