"""BP part A: the 40-cell structural check of prediction BP-P1, and the
post-hoc replication of BO3b.

TWO DIFFERENT THINGS ARE COMPUTED HERE AND THEY HAVE DIFFERENT STANDING.

  * **BP-P1, a PREDICTION**, filed at §BP3 and committed before this script
    ran. It says `n_rejected` is reproduced exactly by BH's step-up rule
    evaluated on the P-ATOM HISTOGRAM ALONE, and that every cell falls in one
    of six enumerated branches.

  * **BO3b's check, a REPLICATION.** §BO3b already ran and committed
    `rejected > 0 iff at-floor >= 218` at `759c0d3`. Re-running it here is an
    independent re-derivation from the artifact (BM-E1 criterion 12), NOT a
    prediction -- its outcome is ancestral to this session and is reported as
    a replication throughout.

---------------------------------------------------------------------------
HOW THE EXACT ATOM HISTOGRAM IS RECOVERED FROM A float32 ARTIFACT
---------------------------------------------------------------------------
The artifact stores `p` as float32, but every p-value is exactly
`(n_ge + 1)/(n_finite + 1)` -- a rational with a KNOWN denominator, since
`n_finite` is stored per (seed, concept). So the numerator is recovered as

    j = round( p_float32 * (n_finite + 1) )

and this is exact, not approximate: `j` is an integer below ~2e5 and float32's
relative error at that magnitude is ~0.01 absolute, so `round` lands on it.
The histogram is then built in EXACT RATIONALS, never in floats -- which
matters because the whole question is whether an atom sits above or below a
critical value it differs from in the seventh significant figure.

`n_rejected` is read from the artifact's OWN `reject_*` mask, never from the
JSON report that the producing run wrote (criterion 12).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np

from src.analysis.bh_resolution_floor import atom_rank_threshold, k_from_atoms

Q = Fraction(1, 20)   # AV6 component 1's level, ratified unchanged.


def classify(a20: int, a19: int, k_rule: int, n_rej: int,
             thr20: int, thr19: int, second_block_rank: int,
             t_plus_second: int) -> str:
    """BP3's SIX BRANCHES, in the order their conditions are declared.

    Total over the outcome space by construction: every cell reaching here has
    computable counts, so branch 5 is decided by the caller before this runs,
    and branch 6 is tested first because a disagreement voids the reading of
    any other branch.
    """
    if k_rule != n_rej:
        return "6_DISAGREE"
    t = a20 + a19
    if t_plus_second >= second_block_rank and n_rej > t:
        return "4_SECOND_BLOCK"
    if t >= thr19:
        return "1_TWO_ATOM"
    if a20 >= thr20:
        return "2_SINGLE_ATOM"
    return "3_ZERO"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bh", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    with h5py.File(args.bh, "r") as hf:
        seeds = [s.decode() if isinstance(s, bytes) else str(s)
                 for s in hf["seeds"][()]]
        concepts = [c.decode() if isinstance(c, bytes) else str(c)
                    for c in hf["concepts"][()]]
        n_seeds, n_f, n_c = hf["p_a"].shape
        print(f"artifact: {n_seeds} seeds x {n_f} features x {n_c} concepts")
        print(f"attrs: q_fdr={hf.attrs.get('q_fdr')} "
              f"gain_floor_applied={hf.attrs.get('gain_floor_applied')}")

        for arm in ("a", "b"):
            nfin_all = hf[f"n_finite_{arm}"][()]          # (seed, concept)
            for s in range(n_seeds):
                p = hf[f"p_{arm}"][s].astype(np.float64)   # (n_f, n_c)
                rej = hf[f"reject_{arm}"][s]
                nfin = nfin_all[s].astype(np.int64)        # (n_c,)

                finite = np.isfinite(p)
                m = int(finite.sum())
                n_rej = int(rej.sum())

                # ---- exact atom histogram, bucketed by denominator --------
                dens = np.unique(nfin[nfin > 0])
                atoms: Counter = Counter()
                for d in dens:
                    cols = np.flatnonzero(nfin == d)
                    sub = p[:, cols]
                    j = np.rint(sub * (d + 1)).astype(np.int64)
                    # THE RECOVERY IS CHECKED EXACTLY, NOT BY A TOLERANCE.
                    # Re-deriving the float32 p from the recovered integer must
                    # reproduce the stored bytes exactly. A residual bound
                    # cannot do this job: the first version of this check used
                    # 1e-3 and REFUSED on real data, because float32's relative
                    # error 2^-24 over a product as large as (d+1) admits a
                    # residual up to (d+1)*2^-24 = 1.22e-02 -- so 1e-3 was 12x
                    # tighter than the arithmetic allows. The observed maximum
                    # was 6.104e-03, half the theoretical bound and far below
                    # the 0.5 that would break `rint`. The bound was ILL-POSED,
                    # not the data; it is replaced rather than widened, because
                    # a tolerance loosened after it fails is indistinguishable
                    # from one corrected after being shown wrong (gotcha 35).
                    back = (j / (d + 1)).astype(np.float32)
                    if not np.array_equal(back, sub.astype(np.float32)):
                        n_bad = int((back != sub.astype(np.float32)).sum())
                        raise SystemExit(
                            f"NUMERATOR RECOVERY IS NOT EXACT at seed "
                            f"{seeds[s]} arm {arm}: {n_bad} of {sub.size} "
                            f"p-values do not round-trip through "
                            f"j/(n_finite+1). The stored p-values are not the "
                            f"rationals this denominator implies -- refusing.")
                    vals, cnts = np.unique(j, return_counts=True)
                    for v, c in zip(vals.tolist(), cnts.tolist()):
                        atoms[Fraction(int(v), int(d) + 1)] += int(c)

                assert sum(atoms.values()) == m, "histogram lost tests"

                k_rule = k_from_atoms(list(atoms.items()), m, Q)

                # ---- the two floors, and the counts at each ---------------
                a20 = int(atoms.get(Fraction(1, 20 * n_f + 1), 0))
                a19 = int(atoms.get(Fraction(1, 19 * n_f + 1), 0))
                thr20 = atom_rank_threshold(1, n_c, n_f, 20, Q,
                                            n_shifts_concept=20)
                thr19 = atom_rank_threshold(1, n_c, n_f, 20, Q,
                                            n_shifts_concept=19)
                second = atom_rank_threshold(2, n_c, n_f, 20, Q)
                s20 = int(atoms.get(Fraction(2, 20 * n_f + 1), 0))
                s19 = int(atoms.get(Fraction(2, 19 * n_f + 1), 0))

                branch = classify(a20, a19, k_rule, n_rej, thr20, thr19,
                                  second, a20 + a19 + s20 + s19)

                rows.append({
                    "arm": arm.upper(), "seed": seeds[s], "m": m,
                    "A20": a20, "A19": a19, "T": a20 + a19,
                    "second_atom_counts": s20 + s19,
                    "k_rule": k_rule, "n_rejected": n_rej,
                    "diff_k_rule_minus_n_rejected": k_rule - n_rej,
                    "branch": branch,
                    "bo3b_at_floor_ge_218": bool(a20 + a19 >= 218),
                    "bo3b_rejects": bool(n_rej > 0),
                    "bo3b_equality_T_eq_nrej": bool((a20 + a19) == n_rej),
                    "n_distinct_atoms": len(atoms),
                })
                print(f"  arm {arm.upper()} seed {seeds[s]:>2}: "
                      f"A20={a20:>4} A19={a19:>3} T={a20+a19:>4} "
                      f"k_rule={k_rule:>4} n_rej={n_rej:>4} "
                      f"diff={k_rule-n_rej:>3}  {branch}")
                del p, rej

    # ------------------------------------------------------------------
    # SCORING, against the outcome space declared at BP3 BEFORE this ran.
    # ------------------------------------------------------------------
    branches = Counter(r["branch"] for r in rows)
    disagree = [r for r in rows if r["branch"] == "6_DISAGREE"]
    a19_all_zero = all(r["A19"] == 0 for r in rows if r["arm"] == "B")

    if disagree:
        verdict = "REFUTED"
    elif a19_all_zero:
        verdict = "HELD_BUT_UNINFORMATIVE_ON_THE_TWO_FLOOR_REFINEMENT"
    else:
        verdict = "HELD"

    # BO3b's replication, scored separately and labelled as a replication.
    bo3b_biconditional = [r for r in rows
                          if r["bo3b_at_floor_ge_218"] != r["bo3b_rejects"]]
    bo3b_equality = [r for r in rows
                     if r["bo3b_rejects"] and not r["bo3b_equality_T_eq_nrej"]]

    summary = {
        "governing_amendment": "BP (part A)",
        "prediction": "BP-P1",
        "verdict": verdict,
        "n_cells": len(rows),
        "branch_occupancy": dict(branches),
        "cells_disagreeing": len(disagree),
        "A19_is_zero_in_every_arm_B_cell": a19_all_zero,
        "bo3b_replication": {
            "standing": "POST-HOC REPLICATION, not a prediction -- committed "
                        "at 759c0d3 before this session",
            "biconditional_violations": len(bo3b_biconditional),
            "equality_violations": len(bo3b_equality),
            "at_floor_range_A": [min(r["T"] for r in rows if r["arm"] == "A"),
                                 max(r["T"] for r in rows if r["arm"] == "A")],
            "at_floor_range_B": [min(r["T"] for r in rows if r["arm"] == "B"),
                                 max(r["T"] for r in rows if r["arm"] == "B")],
            "zero_rejection_seeds_B": sorted(
                r["seed"] for r in rows
                if r["arm"] == "B" and r["n_rejected"] == 0),
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 68)
    print(f"BP-P1 VERDICT: {verdict}")
    print(f"  branch occupancy: {dict(branches)}")
    print(f"  cells disagreeing (branch 6): {len(disagree)} of {len(rows)}")
    print(f"  A19 == 0 in every arm-B cell: {a19_all_zero}")
    print("BO3b REPLICATION (post-hoc, not a prediction):")
    print(f"  biconditional violations: {len(bo3b_biconditional)} of {len(rows)}")
    print(f"  equality violations: {len(bo3b_equality)}")
    print("=" * 68)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
