"""THE REPORTING BASIS for the BW protein-versus-residue decomposition.

WHAT THIS IS (PREREGISTRATION amendment BX). It computes NO new measure. Every
number here is a reduction of `protein_vs_residue_bw.h5`, which stores both
detection matrices in full -- so no null is re-run, no threshold is recomputed,
and no cutoff is introduced. It exists because TWO FIGURES ON TWO BASES were in
circulation with their relationship unstated:

  * a CONCEPT-level share, restricted to the all-seed across-detected set;
  * a CELL-level share, over across-detected cells.

Neither is wrong. Quoting either without its denominator and its detection-rate
criterion is. This module reports the full grid of defensible summaries, each
carrying both.

THE UNDEFINED TERM THIS FIXES. "Also detected under the within-protein null"
does not say AT WHAT RATE -- in all twenty anchors, in at least one, or at some
other rate. Those are materially different sentences. Here the rate is a free
index `t` swept over every value a 20-seed rate can take, and every reported
figure names its own `t`.

WHY THE RATE IS SWEPT RATHER THAN CHOSEN. Choosing one after seeing the grid
would be selecting a summary on its value. The sweep is total over the
primitive the data has (gotcha 39): `D(c)` takes values in {0, 1/20, ..., 1} and
every one is reported, including the ones with an empty population.

THE `within_only` CELL IS REPORTED EVERYWHERE, INCLUDING WHERE IT IS ZERO, and
where zero it is labelled with WHICH KIND of zero it is -- measured, or
arithmetic. On any population restricted by an across-protein condition that the
across-side predicate then re-tests, the cell is empty BY CONSTRUCTION and says
nothing about the data. That distinction is the whole reason the cell is not
dropped as a rounding convenience: it is occupied on the unrestricted basis.

DESCRIPTIVE, NOT CONFIRMATORY. Inherited unchanged from the artifact being
summarised: percentile thresholds carry no family, no correction and no p-value.
Nothing here is a confirmatory claim and no mechanism reading is attached to any
shape reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.sae.train import git_dirty, git_sha

# The 2x2's four cells, named identically to `protein_vs_residue.py` so a reader
# meeting one summary can locate the other. Ordering is (across, within).
CELL_NAMES = {
    (True, True): "residue_level_candidate",
    (True, False): "protein_level_candidate",
    (False, True): "within_only",
    (False, False): "not_detected_on_this_axis",
}

# Why a zero in the `within_only` cell is zero. A measured zero is a finding
# about the data; a structural zero is arithmetic and must never be read as one
# (gotcha 25: a check that can only fail one way is as useless as one that can
# never fail).
WITHIN_ONLY_MEASURED = "MEASURED -- the cell is reachable on this population"
WITHIN_ONLY_STRUCTURAL = (
    "STRUCTURAL -- this population is restricted by an across-protein condition "
    "that the across-side predicate re-tests, so every unit is across-detected "
    "and the cell is empty by construction, not by measurement")


def rate_grid(n_seeds: int) -> np.ndarray:
    """Every value a detection rate over `n_seeds` anchors can take, excluding 0.

    A threshold of 0 is excluded because `D >= 0` is satisfied by every concept,
    so the predicate is constant and the 2x2 collapses -- a predicate applied
    where it cannot discriminate ranks nothing (gotcha 25b). The excluded value
    is named here rather than left as an off-by-one a reader has to notice.
    """
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")
    return np.arange(1, n_seeds + 1, dtype=float) / n_seeds


def two_by_two(across_detected: np.ndarray,
               within_detected: np.ndarray) -> dict[str, int]:
    """All four cells with an explicit count, including zero.

    Exhaustiveness is asserted rather than assumed: a 2x2 over two booleans sums
    to the population by construction, so a mismatch is an implementation defect
    and not a data outcome.
    """
    if across_detected.shape != within_detected.shape:
        raise ValueError(f"shape mismatch: {across_detected.shape} vs "
                         f"{within_detected.shape}")
    if across_detected.dtype != np.bool_ or within_detected.dtype != np.bool_:
        raise ValueError("both inputs must be boolean")
    counts = {name: int(((across_detected == a) & (within_detected == w)).sum())
              for (a, w), name in CELL_NAMES.items()}
    if sum(counts.values()) != across_detected.size:
        raise SystemExit(
            f"FATAL: the four cells sum to {sum(counts.values())}, not "
            f"{across_detected.size}. A 2x2 over two booleans is exhaustive by "
            "construction, so this is an implementation defect.")
    return counts


def concept_grid(d_across: np.ndarray, d_within: np.ndarray,
                 n_seeds: int) -> list[dict]:
    """THE CONCEPT-UNIT GRID: every population x every detection rate.

    Three populations, each named by its own restriction rather than by a label:

      * `all_concepts` -- the full pool, no across-protein restriction;
      * `across_D_eq_1` -- concepts the across-protein arm detects in ALL
        anchors. This is the population the figure in circulation uses;
      * `across_D_ge_t` -- concepts the across-protein arm detects at the SAME
        rate the within side is being tested at. The restriction moves with `t`,
        so the two arms are always held to one bar.

    Both arms use `D >= t`. On the two restricted populations the across-side
    predicate is satisfied by every member, which makes `within_only`
    structurally empty; that is recorded per row rather than left for a reader
    to infer from a zero.
    """
    if d_across.shape != d_within.shape:
        raise ValueError(f"shape mismatch: {d_across.shape} vs {d_within.shape}")
    rows = []
    for t in rate_grid(n_seeds):
        pops = {
            "all_concepts": np.ones(d_across.shape, dtype=bool),
            "across_D_eq_1": d_across == 1.0,
            "across_D_ge_t": d_across >= t,
        }
        for pop_name, pop_mask in pops.items():
            a = (d_across >= t)[pop_mask]
            w = (d_within >= t)[pop_mask]
            cells = two_by_two(a, w)
            n = int(pop_mask.sum())
            structural = bool(pop_name != "all_concepts")
            rows.append({
                "unit": "concept",
                "population": pop_name,
                "population_definition": {
                    "all_concepts": "no across-protein restriction",
                    "across_D_eq_1": "D_across == 1 (detected in all anchors)",
                    "across_D_ge_t": "D_across >= t, the same rate the within "
                                     "side is tested at",
                }[pop_name],
                "detection_rate_criterion": f"D >= {t:.4f} "
                                            f"({int(round(t * n_seeds))} of "
                                            f"{n_seeds} anchors)",
                "t": float(t),
                "n_seeds_required": int(round(t * n_seeds)),
                "denominator": n,
                "cells": cells,
                "share_within_fails_of_across_detected": (
                    float(cells["protein_level_candidate"]
                          / (cells["protein_level_candidate"]
                             + cells["residue_level_candidate"]))
                    if (cells["protein_level_candidate"]
                        + cells["residue_level_candidate"]) else None),
                "within_only_zero_kind": (
                    None if cells["within_only"] else
                    (WITHIN_ONLY_STRUCTURAL if structural
                     else WITHIN_ONLY_MEASURED)),
            })
    return rows


def matched_rate_comparison(d_across: np.ndarray, d_within: np.ndarray) -> dict:
    """THE THIRD CRITERION, AND IT IS A COMPARISON RATHER THAN A 2x2.

    "The within arm detects at the same rate the across arm did" is a per-concept
    inequality `D_within >= D_across`, not a pair of independent predicates. It
    therefore has no across-side predicate to cross with and no 2x2 to fill; the
    honest form is the three-way split, with the `D_within > D_across` count
    reported as this criterion's analogue of the `within_only` cell.

    Forcing it into a 2x2 would require inventing an across-side threshold, which
    is precisely the cutoff this whole construction avoids.
    """
    if d_across.shape != d_within.shape:
        raise ValueError(f"shape mismatch: {d_across.shape} vs {d_within.shape}")
    out = {}
    for pop_name, mask in (("all_concepts", np.ones(d_across.shape, dtype=bool)),
                           ("across_D_eq_1", d_across == 1.0)):
        a, w = d_across[mask], d_within[mask]
        out[pop_name] = {
            "denominator": int(mask.sum()),
            "n_within_ge_across": int((w >= a).sum()),
            "n_within_lt_across": int((w < a).sum()),
            "n_within_gt_across": int((w > a).sum()),
            "n_within_eq_across": int((w == a).sum()),
            "within_only_analogue": {
                "definition": "D_within > D_across -- the within arm detects at "
                              "a strictly higher rate",
                "count": int((w > a).sum()),
            },
        }
    return out


def cell_grid(det_across: np.ndarray, det_within: np.ndarray,
              d_across: np.ndarray) -> list[dict]:
    """THE CELL-UNIT GRID. A `(seed, concept)` cell carries no rate, so the axis
    that varies here is the DENOMINATOR alone.

    Four denominators, each one a basis a figure has been or could be quoted on.
    `across_detected_cells` is the basis of the cell-level share in circulation
    and it is named as such.
    """
    if det_across.shape != det_within.shape:
        raise ValueError(f"shape mismatch: {det_across.shape} vs "
                         f"{det_within.shape}")
    n_seeds, n_concepts = det_across.shape
    if d_across.shape != (n_concepts,):
        raise ValueError(f"d_across has shape {d_across.shape}, expected "
                         f"({n_concepts},)")
    conc_d1 = d_across == 1.0
    bases = {
        "all_cells": np.ones(det_across.shape, dtype=bool),
        "across_detected_cells": det_across,
        "cells_of_concepts_with_across_D_eq_1":
            np.broadcast_to(conc_d1, det_across.shape),
        "across_detected_cells_of_concepts_with_across_D_eq_1":
            det_across & np.broadcast_to(conc_d1, det_across.shape),
    }
    rows = []
    for name, mask in bases.items():
        a, w = det_across[mask], det_within[mask]
        cells = two_by_two(a, w)
        denom = cells["protein_level_candidate"] + cells["residue_level_candidate"]
        structural = bool(name != "all_cells")
        rows.append({
            "unit": "cell",
            "denominator_name": name,
            "denominator_definition": {
                "all_cells": "every (seed, concept) cell",
                "across_detected_cells": "cells the across-protein arm detects "
                                         "-- the basis of the cell-level share "
                                         "in circulation",
                "cells_of_concepts_with_across_D_eq_1":
                    "every cell of a concept with D_across == 1",
                "across_detected_cells_of_concepts_with_across_D_eq_1":
                    "across-detected cells of concepts with D_across == 1 "
                    "(identical to the row above, because D_across == 1 means "
                    "every one of that concept's cells is across-detected)",
            }[name],
            "denominator": int(mask.sum()),
            "cells": cells,
            "share_within_fails_of_across_detected":
                float(cells["protein_level_candidate"] / denom) if denom else None,
            "within_only_zero_kind": (
                None if cells["within_only"] else
                (WITHIN_ONLY_STRUCTURAL if structural else WITHIN_ONLY_MEASURED)),
        })
    return rows


def concept_to_cell_bridge(det_across: np.ndarray, det_within: np.ndarray,
                           d_across: np.ndarray, d_within: np.ndarray) -> dict:
    """THE RELATIONSHIP BETWEEN THE TWO FIGURES, stated as arithmetic.

    They differ in TWO ways at once, which is why neither is derivable from the
    other:

      1. UNIT -- a concept counts once however many of its seeds fail; a cell
         counts every failing seed separately. On one shared population this is
         an ANY-over-seeds count against a RATE-over-seeds count, and the first
         is always the larger share whenever failures concentrate within
         concepts.
      2. POPULATION -- the concept figure is restricted to `D_across == 1` while
         the cell figure runs over all across-detected cells, which includes
         cells from concepts the across arm detects only sometimes.

    Both differences are quantified here, so a reader who meets one figure can
    walk to the other.
    """
    conc_d1 = d_across == 1.0
    cells_d1 = np.broadcast_to(conc_d1, det_across.shape)
    fail = det_across & ~det_within

    n_conc_d1 = int(conc_d1.sum())
    n_conc_d1_any_fail = int((fail & cells_d1).any(axis=0)[conc_d1].sum())
    n_cells_d1 = int((det_across & cells_d1).sum())
    n_cells_d1_fail = int((fail & cells_d1).sum())
    n_cells_all = int(det_across.sum())
    n_cells_all_fail = int(fail.sum())

    fails_per_failing_concept = (
        n_cells_d1_fail / n_conc_d1_any_fail if n_conc_d1_any_fail else None)
    return {
        "step_1_unit_on_one_shared_population": {
            "population": "the 529-style set: concepts with D_across == 1",
            "n_concepts": n_conc_d1,
            "n_concepts_with_at_least_one_within_failure": n_conc_d1_any_fail,
            "concept_share": (n_conc_d1_any_fail / n_conc_d1) if n_conc_d1 else None,
            "n_across_detected_cells": n_cells_d1,
            "n_within_failing_cells": n_cells_d1_fail,
            "cell_share": (n_cells_d1_fail / n_cells_d1) if n_cells_d1 else None,
            "mean_failing_seeds_per_failing_concept": fails_per_failing_concept,
            "reading": "on ONE population the concept share exceeds the cell "
                       "share by exactly the factor by which failures "
                       "concentrate: a failing concept fails in "
                       "`mean_failing_seeds_per_failing_concept` of its 20 "
                       "anchors, not in all 20",
        },
        "step_2_population_on_one_shared_unit": {
            "unit": "across-detected (seed, concept) cells",
            "all_across_detected_cells": n_cells_all,
            "within_failing": n_cells_all_fail,
            "share": (n_cells_all_fail / n_cells_all) if n_cells_all else None,
            "of_which_from_concepts_with_across_D_eq_1": n_cells_d1_fail,
            "of_which_from_other_concepts": n_cells_all_fail - n_cells_d1_fail,
            "other_concepts_denominator": n_cells_all - n_cells_d1,
            "other_concepts_share": (
                (n_cells_all_fail - n_cells_d1_fail) / (n_cells_all - n_cells_d1)
                if (n_cells_all - n_cells_d1) else None),
        },
    }


def common_basis_diff(det_across: np.ndarray, det_within: np.ndarray,
                      d_across: np.ndarray, d_within: np.ndarray,
                      n_seeds: int) -> dict:
    """PART B -- the within-protein failure rate against the across-protein
    detection rate, ON ONE BASIS. Reported as a diff, never as a verdict.

    Every band a 20-seed rate can occupy gets a row, INCLUDING the empty ones:
    an omitted band is indistinguishable from a band that was never computed
    (gotcha 28). No threshold is applied, no correlation is quoted as a verdict,
    and nothing here is scored.

    Two views, because the two figures in circulation live on the two units:

      * CELL view -- among a band's across-detected cells, the share the within
        arm does not detect. This is the common basis, and it is the one that
        answers the question without mixing units.
      * CONCEPT view -- the band's mean `D_within` and its share at each
        endpoint, reported beside it so the cell view is not read alone.
    """
    bands = []
    fail = det_across & ~det_within
    for k in range(n_seeds + 1):
        t = k / n_seeds
        sel = np.isclose(d_across, t)
        n_conc = int(sel.sum())
        colmask = np.broadcast_to(sel, det_across.shape)
        n_ad = int((det_across & colmask).sum())
        n_fail = int((fail & colmask).sum())
        bands.append({
            "D_across": float(t),
            "seeds_detected_across": k,
            "n_concepts": n_conc,
            "n_across_detected_cells": n_ad,
            "n_within_failing_cells": n_fail,
            "within_failure_rate_of_across_detected_cells":
                (n_fail / n_ad) if n_ad else None,
            "mean_D_within": float(d_within[sel].mean()) if n_conc else None,
            "n_concepts_D_within_eq_1": int((d_within[sel] == 1.0).sum()),
            "n_concepts_D_within_eq_0": int((d_within[sel] == 0.0).sum()),
        })
    occupied = [b for b in bands
                if b["within_failure_rate_of_across_detected_cells"] is not None]
    return {
        "bands": bands,
        "n_bands_total": len(bands),
        "n_bands_empty": len(bands) - len(occupied),
        "standing": "DESCRIPTIVE AND COMPUTED BEFORE DECLARED. Reported with "
                    "that standing, excluded from anything requiring blindness, "
                    "and not reportable as a finding until a declaration "
                    "governs it.",
        "no_mechanism_reading": "None is written here. If the shape invites one "
                                "about why reliably detected concepts might be "
                                "less residue-specific, the reading is NOT "
                                "authorised and is not supplied in any form, "
                                "hedged or otherwise.",
    }


def inference_check(diff: dict) -> dict:
    """DOES THE COMMON-BASIS SHAPE SUPPORT THE INFERENCE DRAWN ACROSS TWO BASES?

    The inference under test, stated as its author stated it: that the most
    reliably detected concepts are disproportionately protein-level rather than
    residue-localised. On ONE basis that is the claim that the within-failure
    rate among across-detected cells RISES with `D_across`.

    THE OUTCOME SPACE IS ENUMERATED IN THE PRIMITIVE THE DATA HAS (gotcha 39),
    and it is not a two-way disjunction -- the answer may be none of `rises`,
    `falls` or `flat`:

      * `SUPPORTS`     -- the top band's rate is the largest of the occupied
                          bands AND exceeds the pooled rate over the others;
      * `REFUTES`      -- the top band's rate is at or below the minimum of the
                          other occupied bands;
      * `NEITHER`      -- the top band sits inside the range of the others;
      * `UNINFORMATIVE`-- fewer than two occupied bands, so the comparison ranks
                          nothing.

    A verdict here is a DESCRIPTION of the shape, not a test: no interval, no
    threshold, no p-value, and it is not scored in any ledger.
    """
    occ = [b for b in diff["bands"]
           if b["within_failure_rate_of_across_detected_cells"] is not None]
    if len(occ) < 2:
        return {"verdict": "UNINFORMATIVE",
                "reason": f"only {len(occ)} occupied band(s); the comparison "
                          "ranks nothing"}
    top = max(occ, key=lambda b: b["D_across"])
    others = [b for b in occ if b is not top]
    rates = [b["within_failure_rate_of_across_detected_cells"] for b in others]
    top_rate = top["within_failure_rate_of_across_detected_cells"]
    pooled_fail = sum(b["n_within_failing_cells"] for b in others)
    pooled_ad = sum(b["n_across_detected_cells"] for b in others)
    pooled_rate = pooled_fail / pooled_ad if pooled_ad else None

    if top_rate > max(rates) and pooled_rate is not None and top_rate > pooled_rate:
        verdict = "SUPPORTS"
    elif top_rate <= min(rates):
        verdict = "REFUTES"
    else:
        verdict = "NEITHER"
    return {
        "verdict": verdict,
        "inference_under_test": "the most reliably detected concepts are "
                                "disproportionately protein-level rather than "
                                "residue-localised",
        "restated_on_one_basis": "the within-failure rate among across-detected "
                                 "cells rises with D_across",
        "top_band_D_across": top["D_across"],
        "top_band_rate": top_rate,
        "other_bands_min_rate": min(rates),
        "other_bands_max_rate": max(rates),
        "other_bands_pooled_rate": pooled_rate,
        "n_occupied_bands": len(occ),
        "standing": "a DESCRIPTION of the shape, not a test. No interval, no "
                    "threshold, no p-value, not scored.",
    }


def load(path: Path) -> dict:
    with h5py.File(path, "r") as hf:
        for need in ("D_across_protein", "D_within_protein",
                     "detected_across_protein", "detected_within_protein",
                     "concepts", "seeds"):
            if need not in hf:
                raise SystemExit(f"FATAL: `{need}` absent from {path}. This "
                                 "module summarises the BW decomposition "
                                 "artifact and cannot substitute another.")
        out = {
            "d_across": hf["D_across_protein"][:],
            "d_within": hf["D_within_protein"][:],
            "det_across": hf["detected_across_protein"][:].astype(bool),
            "det_within": hf["detected_within_protein"][:].astype(bool),
            "concepts": [x.decode() if isinstance(x, bytes) else str(x)
                         for x in hf["concepts"][:]],
            "seeds": [x.decode() if isinstance(x, bytes) else str(x)
                      for x in hf["seeds"][:]],
            "attrs": {k: (v.item() if isinstance(v, np.generic) else
                          (v.tolist() if isinstance(v, np.ndarray) else v))
                      for k, v in hf.attrs.items()},
        }
    if out["attrs"].get("confirmatory") not in (False, np.False_, 0):
        raise SystemExit("FATAL: the source artifact does not declare "
                         "`confirmatory = False`. This module reports a "
                         "descriptive decomposition and refuses to summarise an "
                         "artifact claiming otherwise.")
    # The stored rates must equal the row means of the stored matrices. Both are
    # on disk, so this is checkable rather than assumable -- and a disagreement
    # would mean one of the two was written from a different run.
    for arm in ("across", "within"):
        recomputed = out[f"det_{arm}"].mean(axis=0)
        if not np.allclose(recomputed, out[f"d_{arm}"]):
            raise SystemExit(
                f"FATAL: stored D_{arm}_protein disagrees with the column mean "
                f"of detected_{arm}_protein. The two were not written from one "
                "run.")
    return out


def run(path: Path) -> dict:
    src = load(path)
    n_seeds = len(src["seeds"])
    d_a, d_w = src["d_across"], src["d_within"]
    det_a, det_w = src["det_across"], src["det_within"]
    diff = common_basis_diff(det_a, det_w, d_a, d_w, n_seeds)
    return {
        "specification": "PREREGISTRATION amendment BX -- the reporting basis "
                         "for the BW decomposition. NO new measure: every "
                         "figure is a reduction of the stored artifact.",
        "source": {"artifact": str(path),
                   "attrs": src["attrs"],
                   "n_concepts": len(src["concepts"]),
                   "n_seeds": n_seeds},
        "identity": {"git_sha": git_sha(), "git_dirty": git_dirty()},
        "concept_grid": concept_grid(d_a, d_w, n_seeds),
        "matched_rate_comparison": matched_rate_comparison(d_a, d_w),
        "cell_grid": cell_grid(det_a, det_w, d_a),
        "concept_to_cell_bridge": concept_to_cell_bridge(det_a, det_w, d_a, d_w),
        "common_basis_diff": diff,
        "inference_check": inference_check(diff),
    }


def _fmt_grid(rows: list[dict]) -> str:
    out = []
    for r in rows:
        c = r["cells"]
        share = r["share_within_fails_of_across_detected"]
        out.append(
            f"  {r.get('population', r.get('denominator_name')):<52} "
            f"{r.get('detection_rate_criterion', '-- (cell unit)'):<28} "
            f"n={r['denominator']:>6}  "
            f"res={c['residue_level_candidate']:>6} "
            f"prot={c['protein_level_candidate']:>6} "
            f"wonly={c['within_only']:>5} "
            f"none={c['not_detected_on_this_axis']:>6}  "
            f"prot_share={'--' if share is None else f'{share:.4f}'}")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    a = p.parse_args()

    res = run(a.artifact)
    a.out_json.write_text(json.dumps(res, indent=1, default=str))

    print("=== CONCEPT GRID (every population x every detection rate) ===")
    print(_fmt_grid(res["concept_grid"]))
    print("\n=== CELL GRID (denominator is the axis) ===")
    print(_fmt_grid(res["cell_grid"]))
    print("\n=== MATCHED-RATE COMPARISON (not a 2x2) ===")
    print(json.dumps(res["matched_rate_comparison"], indent=1))
    print("\n=== CONCEPT-TO-CELL BRIDGE ===")
    print(json.dumps(res["concept_to_cell_bridge"], indent=1))
    print("\n=== PART B: COMMON-BASIS DIFF ===")
    n_seeds = res["source"]["n_seeds"]
    for b in res["common_basis_diff"]["bands"]:
        r = b["within_failure_rate_of_across_detected_cells"]
        mdw = b["mean_D_within"]
        rate_s = "EMPTY" if r is None else f"{r:.4f}"
        mdw_s = "--" if mdw is None else f"{mdw:.4f}"
        print(f"  D_across={b['D_across']:.2f} "
              f"({b['seeds_detected_across']:>2}/{n_seeds})  "
              f"n_conc={b['n_concepts']:>4}  "
              f"ad_cells={b['n_across_detected_cells']:>6}  "
              f"fail={b['n_within_failing_cells']:>6}  "
              f"rate={rate_s:>7}  meanD_within={mdw_s}")
    print(f"  empty bands: {res['common_basis_diff']['n_bands_empty']} of "
          f"{res['common_basis_diff']['n_bands_total']}")
    print("\n=== INFERENCE CHECK ===")
    print(json.dumps(res["inference_check"], indent=1))
    print(f"\nwrote {a.out_json}")


if __name__ == "__main__":
    main()
