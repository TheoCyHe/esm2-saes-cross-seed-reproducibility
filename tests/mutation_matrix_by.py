"""Mutation matrix for `decomposition_percentile_curve` -- amendment BY. NOT
collected by pytest.

Run manually:  python tests/mutation_matrix_by.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

The runner is reused from `mutation_matrix_av` -- control-first, `__pycache__`
purged between invocations, byte-compared restore, decisions on pytest's EXIT
CODE. A second copy of a runner is a second thing that can silently stop
detecting (gotcha 13 extension 5).

WHAT THIS FILE EXISTS FOR. This module introduces no new measure: it sweeps a
committed construction over a committed percentile range. So its characteristic
defect is a curve that LOOKS like a curve -- monotone, plausibly shaped, every
denominator present -- while the percentile never actually moved, or the
estimability rule never actually excluded anything, or the verdict cannot return
anything but "holds". Each of those yields a complete, publication-shaped table
with no error anywhere.

THE MUTATIONS ARE CHOSEN AGAINST THAT. Every one leaves a well-formed grid.
Three of them target checks whose failure mode is "can only pass" (gotcha 25a),
which is the shape this record has been bitten by most recently (BX's own
surviving mutant, corrections entry 158).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run  # noqa: E402

ROOT = os.getcwd()
M = os.path.join(ROOT, "src", "analysis", "decomposition_percentile_curve.py")
T = "tests/test_decomposition_percentile_curve.py"

# THE CONTROL: `curve_grid` marks every percentile estimable regardless of the
# draw count. It is not a no-op and not a message edit -- the excluded row must
# carry NO figures, and several fixtures assert both the exclusion flag and the
# absence of an `arms` key, so no blanket verdict satisfies them.
CONTROL = (
    "CONTROL -- every percentile declared estimable regardless of draw count", M,
    b'        e = est[float(q)]\n',
    b'        e = dict(est[float(q)]); e["estimable"] = True\n',
    [f"{T}::test_999_is_excluded_at_400_draws_and_marked_excluded",
     f"{T}::test_per_seed_draw_count_would_exclude_the_99th_too",
     f"{T}::test_verdict_no_estimable_percentile_is_its_own_branch"],
)

MUTATIONS = [
    # ============================================ 1. THE PERCENTILE NEVER MOVES
    # Both thresholds computed at the REFERENCE percentile whatever row we are
    # on. The grid still has six rows, still excludes the 99.9th, and every
    # figure is a real figure -- it is just the same figure six times. A reader
    # checking that the table is well-formed sees nothing. This is the defect
    # that would make the whole Part vacuous while appearing to discharge it.
    ("thresholds always computed at the reference percentile, not at q", M,
     b'        thr_w = null_thresholds(null_within, float(q), POOLING_MODE)\n'
     b'        thr_a = null_thresholds(null_across, float(q), POOLING_MODE)',
     b'        thr_w = null_thresholds(null_within, REFERENCE_PERCENTILE, POOLING_MODE)\n'
     b'        thr_a = null_thresholds(null_across, REFERENCE_PERCENTILE, POOLING_MODE)',
     [f"{T}::test_thresholds_move_with_the_percentile"]),

    # ====================================== 2. ONLY ONE ARM MOVES WITH q
    # The within arm sweeps and the across arm stays pinned at the reference.
    # The two arms are then compared AT DIFFERENT PERCENTILES, so any difference
    # between them is partly a difference of cutoff -- the precise error the
    # brief's "match the across-protein side's percentiles" clause forbids, and
    # it is invisible in the output because both columns are populated.
    ("the across arm pinned to the reference while the within arm sweeps", M,
     b'        thr_a = null_thresholds(null_across, float(q), POOLING_MODE)',
     b'        thr_a = null_thresholds(null_across, REFERENCE_PERCENTILE, POOLING_MODE)',
     # NAMED FIXTURE CHANGED AFTER THIS MUTATION SURVIVED ITS FIRST RUN. The
     # single-arm fixture cannot see the across arm being pinned; the survivor
     # was a COVERAGE HOLE, not a weak fixture, and it was closed by ADDING a
     # both-arms test rather than by weakening the mutation.
     [f"{T}::test_BOTH_arms_thresholds_move_with_the_percentile"]),

    # ================================== 3. THE ESTIMABILITY BOUND'S UNSTABLE FORM
    # `n*(1 - q/100)` instead of `n*(100 - q)/100`. Equal in exact arithmetic,
    # not in floating point: at n=20, q=95 the first gives 1.0000000000000009.
    # AX2 recorded this as the boundary deciding whether the reference
    # percentile is supported at all, and required the unstable form be killed.
    # Mutating the SHARED helper, which is where the rule lives.
    ("estimability bound in the float-unstable form",
     os.path.join(ROOT, "src", "analysis", "concept_repro.py"),
     b'        above = n * (100.0 - float(q)) / 100.0',
     b'        above = n * (1.0 - float(q) / 100.0)',
     [f"{T}::test_the_95th_bound_is_computed_in_the_stable_form"]),

    # ============================ 4. THE VERDICT CAN ONLY RETURN "HOLDS"
    # `all()` over an empty generator is True, so filtering to the rows that
    # already pass makes the conjunction vacuous. The verdict prints
    # HOLDS_ACROSS_RANGE with an empty `failing_percentiles` list -- exactly
    # what a genuine pass looks like. gotcha 25a: a check that can only pass.
    ("the range conjunction taken over only the rows that already pass", M,
     b'        holds = all(p[flag_key] for p in per)',
     b'        holds = all(p[flag_key] for p in per if p[flag_key])',
     [f"{T}::test_verdict_fails_at_some_percentile_THE_DISCRIMINATING_FIXTURE",
      f"{T}::test_verdict_fails_when_only_one_percentile_fails"]),

    # ================== 5. THE CONJUNCTION BECOMES A DISJUNCTION AT THE REFERENCE
    # "Holds if it holds at the reference" -- which is precisely the standard
    # AW1.5 refuses ("a conclusion counts only if it holds across the reported
    # percentile range"). The output is well-formed and the reference row is
    # genuinely computed, so the table supports the verdict it prints.
    ("range verdict decided by the reference row alone", M,
     b'        holds = all(p[flag_key] for p in per)',
     b'        holds = any(p[flag_key] for p in per if p["is_reference"])',
     [f"{T}::test_verdict_fails_when_only_one_percentile_fails"]),

    # ============================== 6. AN EMPTY DENOMINATOR REPORTED AS ZERO
    # A share of 0.0 over an empty population reads as a measurement of nothing
    # rather than an absence of any measurement -- gotcha 25's shape, and the
    # distinction BX's `within_only` labelling exists to preserve.
    ("zero denominator yields a 0.0 share instead of None", M,
     b'            "share": (float(n_cells_across_not_within / n_cells_across)\n'
     b'                      if n_cells_across else None),',
     b'            "share": (float(n_cells_across_not_within / n_cells_across)\n'
     b'                      if n_cells_across else 0.0),',
     [f"{T}::test_zero_denominator_reports_none_with_a_stated_reason"]),

    # ========================= 7. NO_ESTIMABLE_PERCENTILE FOLDED INTO A FAILURE
    # gotcha 30: two causes of one branch with opposite readings. "The curve
    # could not be computed" reported as "the conclusion does not hold" turns an
    # absence of evidence into evidence of absence, in one word.
    ("an uncomputable curve reported as the conclusion failing", M,
     b'            "verdict": "NO_ESTIMABLE_PERCENTILE",',
     b'            "verdict": "FAILS_AT_SOME_PERCENTILE",',
     [f"{T}::test_verdict_no_estimable_percentile_is_its_own_branch"]),

    # ============================== 8. THE EXCLUDED ROW GAINS FIGURES ANYWAY
    # Dropping the `continue` lets a non-estimable percentile be computed and
    # reported. The number that comes out IS the sample maximum wearing a
    # different label -- it reads as a number and is not one (AX2). The row
    # still says `estimable: False`, so the flag and the figures disagree while
    # both are present.
    ("figures computed for a row the rule excluded", M,
     b'            rows.append(row)\n'
     b'            continue\n',
     b'            rows.append(row)\n',
     [f"{T}::test_999_is_excluded_at_400_draws_and_marked_excluded"]),
]


def main() -> int:
    print(__doc__.splitlines()[0])
    print("=" * 74)
    label, path, old, new, expect = CONTROL
    survived = apply_and_run(path, old, new, expect)
    if survived is None:
        print(f"CONTROL ABORT: anchor for {label} did not match exactly once. "
              "The runner cannot be shown to detect, so no verdict is printed.")
        return 2
    if survived:
        print(f"CONTROL ABORT: {label} was NOT detected (survivors: {survived}). "
              "Every SURVIVED verdict this runner could print would be "
              "meaningless, so none is printed.")
        return 2
    print(f"CONTROL detected ({len(expect)} fixtures) -- the runner detects.\n")

    n_surv = 0
    for label, path, old, new, expect in MUTATIONS:
        survived = apply_and_run(path, old, new, expect)
        if survived is None:
            print(f"ANCHOR MISS  {label}")
            n_surv += 1
            continue
        if survived:
            print(f"SURVIVED     {label}  -> {survived}")
            n_surv += len(survived)
        else:
            print(f"KILLED       {label}  ({len(expect)} fixtures)")
    print(f"\n{len(MUTATIONS) - n_surv if n_surv == 0 else '?'} "
          f"of {len(MUTATIONS)} mutations killed; survivors/misses: {n_surv}")
    return 0 if n_surv == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
