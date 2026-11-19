"""
AC4 -- reproduce amendment BH's degenerate-cell enumeration EXACTLY.

WHY THIS IS AN EXACT TEST AND NOT A COMPARISON WITH TOLERANCE. The
displacement schedules are STORED: drawn once from `--null-seed 0`, shared
across every seed, and reproducible from the seed alone. So "which (concept,
arm, shift) cells have zero surviving positives" has ONE right answer, and an
implementation of BG2's collision rule either produces it or differs from what
BH declares. Nothing here touches the average-precision computation at all --
this tests masking and collision handling on their own.

IT REPORTS THE DIFFERENCE, NOT A VERDICT ALONE. A printed conclusion has been
wrong in this project while the computation was right (gotcha 20), so every
comparison prints both sides.

AND IT EXITS NON-ZERO ON A MISMATCH. `validate_ap_restructure.py` printed FAIL
and exited 0, which would have let an `afterok` chain read a bit-level
discrepancy as success -- a validator whose failure is unreadable by the thing
gating on it is worse than no validator (gotcha 13 extension 1). The exit path
is part of the check.

It also emits the experimental positive-count distribution across the in-pool
concepts, which brief B §4b owes for the writeup's limitation magnitude.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from src.analysis.artifact_identity import argmax_pool_mask
from src.analysis.feature_ap import load_labels, load_scored_row_set
from src.analysis.null_pilot import (
    BH3_BOTH_ARMS,
    BH3_NONE,
    BH3_ONE_ARM,
    BH3_SOME,
    build_displacement_plan,
    build_masks,
    build_population_index,
    build_row_to_protein,
    classify_bh3,
    draw_schedules,
    exceedance_resolution,
    protein_starts_lengths,
)
from src.sae.train import git_dirty, git_sha


def compare(label: str, got, expected, out: list) -> bool:
    """One comparison, both sides always printed."""
    ok = got == expected
    out.append({"check": label, "expected": expected, "got": got, "match": bool(ok)})
    print(f"  [{'MATCH ' if ok else 'DIFFER'}] {label}\n"
          f"           expected: {expected!r}\n"
          f"           got:      {got!r}")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-cache", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--expected", type=Path, required=True,
                   help="degenerate_draws_bh.json -- BH's declared enumeration.")
    p.add_argument("--scored-row-set", default="label_store")
    p.add_argument("--stratum", default="experimental")
    p.add_argument("--negatives", default="unannotated")
    p.add_argument("--n-shifts", type=int, default=20)
    p.add_argument("--null-seed", type=int, default=0)
    p.add_argument("--n-length-bins", type=int, default=20)
    p.add_argument("--dict-size", type=int, default=10240)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    # ---- geometry, exactly as null_pilot builds it ----------------------
    with h5py.File(args.eval_cache, "r") as hf:
        raw = hf["index_json"][()]
        index = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    subset = load_scored_row_set(args.labels, args.scored_row_set)
    with h5py.File(args.labels, "r") as hf:
        n_rows = int(hf.attrs["n_rows"])
        library_concepts = sorted(hf["concepts"].keys())
        library_sha = hf.attrs.get("git_sha")
    library_sha = library_sha.decode() if isinstance(library_sha, bytes) else library_sha

    pop_index, remap = build_population_index(index, subset)
    row2protein = build_row_to_protein(pop_index, n_rows)
    starts, lengths = protein_starts_lengths(pop_index)
    print(f"{len(starts):,} proteins, {n_rows:,} rows, "
          f"library {len(library_concepts)} concepts, library_sha={library_sha}")

    def to_subset(arr):
        a = arr.astype(np.int64)
        if remap is None:
            return a
        a = a[a < len(remap)]
        m = remap[a]
        return m[m >= 0]

    stratum_labels, _ = load_labels(args.labels, args.stratum)
    pooled, _ = load_labels(args.labels, "pooled")
    concepts_all = sorted(stratum_labels)
    keep = argmax_pool_mask(concepts_all)
    concepts = [c for c, kk in zip(concepts_all, keep) if kk]
    excluded = [c for c, kk in zip(concepts_all, keep) if not kk]
    print(f"stratum={args.stratum}: {len(concepts_all)} concepts with positives; "
          f"{len(excluded)} pooled parents excluded -> {len(concepts)} in pool")
    print(f"  excluded: {sorted(excluded)}")

    concept_rows = {c: to_subset(stratum_labels[c]) for c in concepts}
    pooled_rows = {c: to_subset(pooled[c]) for c in concepts if c in pooled}
    keep_row, n_eff, n_masked = build_masks(
        concepts, concept_rows, pooled_rows, args.negatives, n_rows)

    shift_schedule, dest_schedule, _ = draw_schedules(
        args.null_seed, args.n_shifts, lengths, args.n_length_bins, True, True)
    plan = build_displacement_plan(
        concepts, concept_rows, keep_row, row2protein, starts, lengths,
        shift_schedule, dest_schedule, args.n_shifts)

    # ---- what we measured -----------------------------------------------
    n_pos = np.array([concept_rows[c].size for c in concepts])
    degen = {arm: plan[arm]["degenerate"] for arm in ("a", "b")}
    counts = {arm: plan[arm]["counts"] for arm in ("a", "b")}
    nd = {arm: degen[arm].sum(axis=1) for arm in ("a", "b")}
    bh3 = [classify_bh3(int(nd["a"][i]), int(nd["b"][i]), args.n_shifts)
           for i in range(len(concepts))]

    cells = []
    for i, c in enumerate(concepts):
        for arm in ("a", "b"):
            if not degen[arm][i].any():
                continue
            surv = int((~degen[arm][i]).sum())
            cells.append({
                "concept": c, "arm": arm.upper(),
                "n_observed_positives": int(n_pos[i]),
                "n_masked_rows": int(n_masked[c]),
                "n_degenerate_shifts": int(degen[arm][i].sum()),
                "degenerate_shift_indices": np.flatnonzero(degen[arm][i]).tolist(),
                "surviving_shifts": surv,
                "surviving_positive_counts": counts[arm][i].tolist(),
                "resolution_after_drop": {
                    "surviving_shifts": surv,
                    "denominator": surv * args.dict_size + 1,
                    "resolution": exceedance_resolution(surv, args.dict_size)},
                "bh3_case": bh3[i],
            })

    exp = json.loads(args.expected.read_text())
    results: list = []
    print("\n=== AC4: measured against BH's declared enumeration ===")
    ok = True
    ok &= compare("concepts in the argmax pool", len(concepts),
                  int(exp["n_concepts"]), results)
    ok &= compare("cells per arm", len(concepts) * args.n_shifts,
                  int(exp["n_cells_per_arm"]), results)
    ok &= compare("degenerate cells in the whole study", len(cells),
                  int(exp["n_degenerate_cells_total"]), results)
    ok &= compare("degenerate cells under Null A", int(nd["a"].sum()),
                  int(exp["null_A_degenerate_cells"]), results)

    exp_cells = exp["degenerate_cells"]
    ok &= compare("degenerate (concept, arm) pairs", len(cells),
                  len(exp_cells), results)
    for j, e in enumerate(exp_cells):
        got = cells[j] if j < len(cells) else None
        if got is None:
            ok &= compare(f"cell[{j}] present", None, e["concept"], results)
            continue
        for key in ("concept", "arm", "n_observed_positives", "n_masked_rows",
                    "n_degenerate_shifts", "degenerate_shift_indices",
                    "surviving_shifts", "surviving_positive_counts",
                    "bh3_case"):
            ok &= compare(f"cell[{j}].{key}", got[key], e[key], results)
        ok &= compare(f"cell[{j}].resolution_denominator",
                      got["resolution_after_drop"]["denominator"],
                      e["resolution_after_drop"]["denominator"], results)
        gr = got["resolution_after_drop"]["resolution"]
        er = e["resolution_after_drop"]["resolution"]
        ok &= compare(f"cell[{j}].resolution (exact float)", repr(gr), repr(er),
                      results)

    occ = Counter(bh3)
    exp_occ = exp["bh3_occupancy"]
    ok &= compare("BH3 SOME_BUT_NOT_ALL count", int(occ.get(BH3_SOME, 0)),
                  int(exp_occ["some_but_not_all_degenerate"]), results)
    ok &= compare("BH3 concepts NOT_EVALUABLE in one arm",
                  sorted(c for c, v in zip(concepts, bh3) if v == BH3_ONE_ARM),
                  list(exp_occ["all_degenerate_one_arm_NOT_EVALUABLE_arm"]), results)
    ok &= compare("BH3 concepts NOT_EVALUABLE in the stratum",
                  sorted(c for c, v in zip(concepts, bh3) if v == BH3_BOTH_ARMS),
                  list(exp_occ["all_degenerate_both_arms_NOT_EVALUABLE_stratum"]),
                  results)
    ok &= compare("BH3 NONE_DEGENERATE count", int(occ.get(BH3_NONE, 0)),
                  len(concepts) - int(exp_occ["some_but_not_all_degenerate"])
                  - len(exp_occ["all_degenerate_one_arm_NOT_EVALUABLE_arm"])
                  - len(exp_occ["all_degenerate_both_arms_NOT_EVALUABLE_stratum"]),
                  results)

    # BH1's "one surviving positive is NOT degenerate" -- the 20/22 cells.
    single = {arm: int((counts[arm] == 1).sum()) for arm in ("a", "b")}
    print(f"\n  cells with EXACTLY ONE surviving positive (NOT degenerate, "
          f"BH1): Null A {single['a']}, Null B {single['b']}")
    if "surviving_positive_histogram" in exp:
        print(f"  (BH's stored histogram block: "
              f"{json.dumps(exp['surviving_positive_histogram'])[:400]})")

    # ---- §4b: the experimental positive-count distribution --------------
    edges = [1, 2, 3, 5, 10, 20, 50, 100, 500, 1000, 10 ** 9]
    hist = {f"[{lo},{hi})": int(((n_pos >= lo) & (n_pos < hi)).sum())
            for lo, hi in zip(edges[:-1], edges[1:])}
    dist = {
        "n_concepts_in_pool": len(concepts),
        "min": int(n_pos.min()), "median": float(np.median(n_pos)),
        "max": int(n_pos.max()),
        "n_at_exactly_1": int((n_pos == 1).sum()),
        "n_below_5": int((n_pos < 5).sum()),
        "n_below_10": int((n_pos < 10).sum()),
        "n_below_100": int((n_pos < 100).sum()),
        "histogram": hist,
    }
    print("\n=== experimental positive counts across the in-pool concepts ===")
    print(json.dumps(dist, indent=2))

    # The 2-to-5 decomposition brief B §4b asks to stop assuming.
    thin = {}
    for arm in ("a", "b"):
        per_bin = {}
        for lo, hi in ((1, 2), (2, 3), (3, 5), (5, 10)):
            sel = (n_pos >= lo) & (n_pos < hi)
            per_bin[f"[{lo},{hi})"] = {
                "n_concepts": int(sel.sum()),
                "cells_with_one_survivor": int((counts[arm][sel] == 1).sum()),
                "cells_degenerate": int(degen[arm][sel].sum()),
                "cells_losing_a_positive": int(
                    (counts[arm][sel] < n_pos[sel][:, None]).sum()),
            }
        thin[arm.upper()] = per_bin
    print("\n=== thin-draw decomposition by observed positive count ===")
    print(json.dumps(thin, indent=2))

    payload = {
        "artifact": "ac4_bh_plan_validation",
        "governing_amendment": "BG (BG2 collision rule) + BH (BH1-BH4)",
        "git_sha": git_sha(), "git_dirty": bool(git_dirty()),
        "inputs": {"labels": str(args.labels), "eval_cache": str(args.eval_cache),
                   "expected": str(args.expected), "library_sha": library_sha},
        "identity": {"stratum": args.stratum, "negatives": args.negatives,
                     "scored_row_set": args.scored_row_set,
                     "n_shifts": args.n_shifts, "null_seed": args.null_seed,
                     "argmax_pool_scored_n": len(concepts),
                     "n_concepts_library": len(library_concepts),
                     "n_rows": n_rows, "n_proteins": len(starts)},
        "verdict": "REPRODUCES_BH_EXACTLY" if ok else "DIFFERS_FROM_BH",
        "n_checks": len(results),
        "n_differing": sum(1 for r in results if not r["match"]),
        "checks": results,
        "measured_degenerate_cells": cells,
        "single_surviving_positive_cells": single,
        "experimental_positive_distribution": dist,
        "thin_draw_decomposition": thin,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")
    print(f"VERDICT: {payload['verdict']} "
          f"({payload['n_differing']} of {len(results)} checks differ)")

    if not ok:
        # The exit path is part of the check (gotcha 13 extension 1).
        raise SystemExit(1)


if __name__ == "__main__":
    main()
