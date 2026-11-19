"""THE DEGENERATE-ARGMAX INDICATOR, PERSISTED.
PREREGISTRATION amendment AO section 8; the author's ratified decision AM6.3.

WHAT THIS IS. A per-feature, per-anchor-seed boolean: **is this feature's argmax
concept one of the 22 degenerate concepts?** AL9b measured the shares; nothing
wrote the indicator down, so the pre-declared sensitivity arm had no input.

WHAT THIS IS NOT. **No regression is run.** AM6 decision 3 is "no guard, add a
sensitivity arm", and the arm is a refit of the primary regression excluding the
flagged features -- but the primary regression IS the design-3 join, which is
next session's work. The brief that recommended this described a refit; that was
mis-scoped, and narrowing it to "build the flag, pre-declare the arm" is recorded
in AO8 rather than done silently.

NOT A FIREWALL CROSSING. It reads `argmax_concept` and `concepts` from the
predictor artifact and the degenerate concept names from M3. No `r_f`, no MNN.

=============================================================================
WHY THE ARM IS WORTH ITS COST (AM6 decision 3, restated once)
=============================================================================
The medians are indistinguishable -- 7.5065 against 7.5347 in `log M` for seed 0
-- so the BULK is unaffected and no guard is warranted. But the p99 gap is 1.46
log units, i.e. **~4.3x in `M`**, and the regression enters `log M` LINEARLY, so
the upper tail carries leverage out of proportion to its 0.93% mass. Refitting
without those features converts a limitation into a number for almost nothing.

=============================================================================
THE SHARES ARE A CORRECTNESS ASSERTION, NOT A PREDICTION (AO4 A1)
=============================================================================
This module recomputes AL9b's shares through `predictor_support`'s own function,
on the same artifact and the same 22 names, so agreement must be EXACT. A
mismatch is a defect in one of the two paths, not a finding -- and filing it as a
prediction would credit the ledger for arithmetic. The expected values are named
in `EXPECTED_AL9B` and compared here; a disagreement FAILS THE JOB, because a
check whose failure cannot reach its consumer is worse than no check (gotcha 13
extension 1).
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import h5py
import numpy as np

from src.analysis.predictor_support import degenerate_argmax_share

# AL9b's measured values, at the tolerance they were reported to. The shares are
# medians over 20 seeds of exact rationals over 10,240, so 4 decimal places is
# not a fudge factor -- it is the precision AL9b's table carries.
EXPECTED_AL9B = {
    "share_median": 0.0093,
    "share_min": 0.0070,
    "share_max": 0.0114,
    "top_decile_share_median": 0.0142,
    "enrichment_median_ratio": 1.53,
}
SHARE_TOL = 5e-4        # half of AL9b's last reported digit
RATIO_TOL = 5e-3

# Identity keys the flag carries so a consumer can assert it was built against
# the same predictor it is about (amendment I3).
CARRIED_IDENTITY = ("library_sha", "n_concepts", "argmax_pool_n",
                    "truncation_policy", "scored_row_set", "inner_mode",
                    "null_arm", "n_rows", "min_protein_count",
                    "retention_floor_mode", "retention_min_instances",
                    "retention_min_residues")


def build_flag(argmax_concept: np.ndarray, concepts: list[str],
               degenerate: list[str]) -> tuple[np.ndarray, dict]:
    """The indicator, plus the name-resolution report.

    A degenerate name absent from the library is REPORTED, never dropped
    silently: the 22 were identified on one axis and the predictor may sit on
    another, and quietly flagging fewer concepts than intended would understate
    the arm (gotcha 15's absent-is-not-matching, on a name rather than an attr).
    """
    name_to_idx = {c: i for i, c in enumerate(concepts)}
    missing = sorted(c for c in degenerate if c not in name_to_idx)
    present = sorted(c for c in degenerate if c in name_to_idx)
    is_deg = np.zeros(len(concepts), dtype=bool)
    for c in present:
        is_deg[name_to_idx[c]] = True
    flag = is_deg[argmax_concept]
    return flag, {
        "n_degenerate_requested": len(degenerate),
        "n_degenerate_resolved": len(present),
        "names_not_in_the_predictor_library": missing,
        "resolved_concept_indices": [int(name_to_idx[c]) for c in present],
    }


def check_against_al9b(deg: dict) -> dict:
    """AO4 A1. Compare the recomputed shares against AL9b's published values.

    A DIFF PER FIELD, not a single verdict -- so a mismatch says WHICH quantity
    moved rather than only that something did.
    """
    got = {
        "share_median": deg["share_across_seeds"]["median"],
        "share_min": deg["share_across_seeds"]["min"],
        "share_max": deg["share_across_seeds"]["max"],
        "top_decile_share_median": deg["top_decile_share_across_seeds"]["median"],
        "enrichment_median_ratio": deg["top_decile_enrichment"]["median_ratio"],
    }
    rows, problems = {}, []
    for key, want in EXPECTED_AL9B.items():
        tol = RATIO_TOL if key.endswith("ratio") else SHARE_TOL
        have = float(got[key])
        ok = abs(have - want) <= tol
        rows[key] = {"AL9b": want, "recomputed": have,
                     "abs_diff": abs(have - want), "tolerance": tol,
                     "agrees": bool(ok)}
        if not ok:
            problems.append(f"{key}: AL9b {want} vs recomputed {have:.6f} "
                            f"(diff {abs(have - want):.6f} > {tol})")
    return {"per_field": rows, "problems": problems,
            "verdict": "PASS" if not problems else "FAIL"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gtilde", type=Path, required=True,
                    help="the predictor of record: argmax_concept + concepts")
    ap.add_argument("--degenerate", type=Path, required=True,
                    help="mu_s_degenerate_m3.json; `degenerate_intersection` "
                         "names the 22 concepts")
    ap.add_argument("--out", type=Path, required=True, help="h5 for the flag")
    ap.add_argument("--out-json", type=Path, required=True,
                    help="the report and the A1 comparison")
    args = ap.parse_args()

    from src.sae.train import git_dirty, git_sha

    with h5py.File(args.gtilde, "r") as hf:
        M = hf["M"][:]
        argmax_concept = hf["argmax_concept"][:]
        concepts = [c.decode() if isinstance(c, bytes) else c
                    for c in hf["concepts"][:]]
        src_ident = {k: (v.decode() if isinstance(v, bytes) else v)
                     for k, v in hf.attrs.items()}

    degenerate = json.loads(args.degenerate.read_text())["degenerate_intersection"]
    flag, resolution = build_flag(argmax_concept, concepts, degenerate)
    if resolution["names_not_in_the_predictor_library"]:
        raise SystemExit(
            "degenerate concept names absent from the predictor's library:\n  * "
            + "\n  * ".join(resolution["names_not_in_the_predictor_library"]) +
            "\nThe 22 were identified on a different axis than the predictor "
            "sits on. Refusing to build a flag that silently covers fewer "
            "concepts than the arm is specified over.")

    deg = degenerate_argmax_share(M, argmax_concept, concepts, degenerate)
    a1 = check_against_al9b(deg)

    n_cells = int(flag.size)
    per_seed_counts = flag.sum(axis=1).astype(int).tolist()
    print(f"[flag] {resolution['n_degenerate_resolved']} degenerate concepts "
          f"resolved of {resolution['n_degenerate_requested']}", flush=True)
    print(f"[flag] flagged cells {int(flag.sum()):,} of {n_cells:,} "
          f"({flag.mean():.6f}); per seed {per_seed_counts}", flush=True)
    print(f"[flag] A1 vs AL9b: {a1['verdict']}", flush=True)
    for key, row in a1["per_field"].items():
        print(f"        {key}: AL9b {row['AL9b']} vs {row['recomputed']:.6f} "
              f"-> {'agrees' if row['agrees'] else 'DIFFERS'}", flush=True)

    report = {
        "identity": {
            "measurement": "degenerate_argmax_flag_AO8",
            "amendment": "AO section 8; the author decision AM6.3",
            "firewall_crossing": "NONE -- reads argmax_concept, concepts and M; "
                                 "no r_f, no MNN",
            "gtilde": str(args.gtilde),
            "gtilde_identity": {k: (v.item() if hasattr(v, "item") else v)
                                for k, v in src_ident.items()},
            "degenerate_source": str(args.degenerate),
            "flag_h5": str(args.out),
            "git_sha": git_sha(), "git_dirty": git_dirty(),
            "python_version": platform.python_version(),
        },
        "name_resolution": resolution,
        "flag_counts": {
            "n_feature_seed_cells": n_cells,
            "n_flagged": int(flag.sum()),
            "overall_rate": float(flag.mean()),
            "n_flagged_by_seed": per_seed_counts,
            "rate_by_seed": [float(x) for x in flag.mean(axis=1)],
        },
        "AL9b_recomputed": deg,
        "AO4_A1_correctness_assertion": a1,
        "sensitivity_arm": (
            "PRE-DECLARED in the specification record: the primary regression "
            "is refit EXCLUDING features flagged here, and both fits are "
            "reported. Rationale (AM6 decision 3): medians are "
            "indistinguishable so the bulk is unaffected, but the p99 gap is "
            "~4.3x in M and log M enters LINEARLY, so the upper tail carries "
            "leverage. NO REGRESSION IS RUN HERE."),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as hf:
        hf.create_dataset("degenerate_argmax", data=flag, compression="gzip")
        hf.create_dataset("argmax_concept", data=argmax_concept,
                          compression="gzip")
        hf.create_dataset("degenerate_concepts",
                          data=np.array(sorted(degenerate),
                                        dtype=h5py.string_dtype()))
        # The predictor's identity is CARRIED, not re-derived, so a consumer can
        # assert the flag and the predictor describe the same object (I3).
        for key in CARRIED_IDENTITY:
            if key in src_ident:
                hf.attrs[key] = src_ident[key]
        hf.attrs["measurement"] = "degenerate_argmax_flag_AO8"
        hf.attrs["amendment"] = "AO8 / AM6.3"
        hf.attrs["gtilde_source"] = str(args.gtilde)
        # THE SEED AXIS IS LABELLED (amendment AS decision 5). It is
        # inherited from the gtilde this flag is built from, row for
        # row -- and `argmax_concept` is copied here precisely so a
        # consumer can VERIFY that inheritance elementwise rather than
        # trust it. Absent labels are reported as absent, not invented.
        from src.analysis.seed_axis import stored_seed_labels
        _gt_seeds = stored_seed_labels(args.gtilde)
        if _gt_seeds is not None:
            hf.create_dataset("seeds",
                              data=[x.encode() for x in _gt_seeds],
                              dtype=h5py.string_dtype())
            hf.attrs["seed_label_source"] = str(args.gtilde)
        else:
            hf.attrs["seed_label_source"] = (
                "ABSENT -- the source gtilde carries no seed labels")
        hf.attrs["degenerate_source"] = str(args.degenerate)
        hf.attrs["n_degenerate_concepts"] = resolution["n_degenerate_resolved"]
        hf.attrs["git_sha"] = git_sha()
        hf.attrs["git_dirty"] = git_dirty()

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    print(f"[flag] wrote {args.out} and {args.out_json}", flush=True)

    # The verdict must be readable by whatever gates on this job (gotcha 13
    # extension 1): the artifacts are written FIRST so the diff survives for
    # inspection, and only then does the job fail.
    if a1["verdict"] != "PASS":
        raise SystemExit(
            "AO4 A1 FAILED -- the recomputed shares do not reproduce AL9b:\n  * "
            + "\n  * ".join(a1["problems"]) +
            "\nThis is the SAME computation on the SAME inputs, so a "
            "disagreement is a defect in one of the two paths, not a finding.")


if __name__ == "__main__":
    main()
