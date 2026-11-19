"""Mutation matrix for BH2's exclusion inside `bg3_propagation`. NOT collected
by pytest.

Run manually:  python tests/mutation_matrix_bg3.py
Exit 0 iff the CONTROL is detected, every mutation kills its named fixtures,
and the BENIGN edit survives.

The runner is reused from `mutation_matrix_av` -- AU9b's control-first rule,
AV22's `__pycache__` purge between invocations, the byte-compared restore, and
the decide-on-EXIT-CODE rule. Reused rather than re-implemented: a runner is a
check, and a second copy is a second thing that can silently stop detecting
(gotcha 13 extension 5).

WHAT THIS FILE EXISTS FOR. The defect it guards against shipped: the bound was
computed over every cell, including the cells BH2 had already dropped, so a
draw that is not in the null distribution set the reported worst case and Arm
B's bound came out INFINITE. The fixtures assert the repair; these mutations
establish that the fixtures would notice if it came undone.

THE FIVE SHAPES THE MUTATIONS COVER:

  * **THE MASK IS BUILT AND NEVER APPLIED.** The AW5 defect's shape, and the
    exact shape of the defect being repaired here. `usable` drops the `keep`
    term and every degenerate cell is back in the bound.
  * **THE CRITERION MOVES.** `> 0` becomes `>= 0` (nothing is degenerate) or
    `> 1` (BH3's fourth case, a single surviving positive, wrongly called
    degenerate -- 20 cells under Null A and 22 under Null B in the real study,
    which would silently coarsen those concepts' exceedance resolution).
  * **THE ENFORCED INVARIANT GOES BACK TO BEING PROSE.** The guard that raises
    on a degenerate cell reaching the bound is what turns "should never reach
    here in practice" from a docstring into a check.
  * **`NOT_EVALUABLE` BECOMES A VALUE.** A concept with no surviving draw
    entering the per-concept percentiles is BH3's prohibition exactly.
  * **A POSITIONAL CHECK BECOMES A COUNT.** `exclusion_derivations_agree`
    compared on totals would agree under any permutation of either axis
    (gotcha 27), which is the one thing it exists to detect.

WHAT IS NOT COVERED, stated rather than left to assumption: `run()`'s wiring --
the h5 reads, the refusal on a disagreeing derivation, the report assembly --
has no unit fixture and is exercised instead by running the entry point against
the real merged null, where Arm A came back bit-identical and Arm B's bound
came back finite. A mutation of `run()` would be reported SKIP here, not
KILLED, and that would be a coverage claim on evidence that does not exist.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run  # noqa: E402
from tests.mutation_matrix_av import run_one  # noqa: E402

ROOT = os.getcwd()
P = os.path.join(ROOT, "src", "analysis", "bg3_propagation.py")
T = "tests/test_bg3_propagation.py"

# THE CONTROL: `bound_factor` returns a constant before any branch runs.
#
# It cannot fail to break the suite -- its fixture asserts FOUR DISTINCT values
# across the argument's range (inf, 2.0, 1.0 and, via the callers, 4/3), and a
# constant collapses them to one. Deliberately a RETURN inserted ahead of the
# expression rather than an edit to a message or a docstring: BA17 item 2's
# first control changed a string while leaving every outcome reachable, and the
# runner correctly reported CONTROL NOT DETECTED.
CONTROL = (
    "CONTROL -- bound_factor returns a constant before any branch", P,
    b'    return float("inf") if d <= -1.0 else 1.0 / (1.0 + d)',
    b'    return 1.0\n'
    b'    return float("inf") if d <= -1.0 else 1.0 / (1.0 + d)',
    [f"{T}::test_the_bound_factor_is_TOTAL_including_at_a_total_loss"],
)

MUTATIONS = [
    # ======================================================= THE MASK ITSELF
    # THE ONE THIS FILE EXISTS FOR: the mask is passed, and then ignored. This
    # is the defect being repaired, restored exactly.
    ("anti_conservative_bound: the keep mask is built and NEVER APPLIED", P,
     b"    usable = keep & np.isfinite(loss)",
     b"    usable = np.isfinite(loss)",
     [f"{T}::test_MASKING_CHANGES_THE_BOUND__the_fixture_discriminates",
      f"{T}::test_a_concept_with_no_surviving_draw_is_NOT_EVALUABLE_not_a_zero",
      f"{T}::test_an_arm_with_no_surviving_draw_ANYWHERE_is_NOT_EVALUABLE"]),

    # BH1's criterion widened so that NOTHING is degenerate.
    ("surviving_draw_mask: `> 0` becomes `>= 0` (nothing is ever degenerate)", P,
     b"    return np.asarray(surviving) > 0",
     b"    return np.asarray(surviving) >= 0",
     [f"{T}::test_the_surviving_draw_mask_IS_BH1s_criterion_and_names_no_cell",
      f"{T}::test_MASKING_CHANGES_THE_BOUND__the_fixture_discriminates"]),

    # BH1's criterion narrowed so that BH3's FOURTH case is swallowed. AP is
    # defined for a single surviving positive -- it is the reciprocal of that
    # positive's rank -- and calling it degenerate discards a draw BH2 keeps.
    ("surviving_draw_mask: `> 0` becomes `> 1` (a single positive called degenerate)",
     P,
     b"    return np.asarray(surviving) > 0",
     b"    return np.asarray(surviving) > 1",
     [f"{T}::test_the_surviving_draw_mask_IS_BH1s_criterion_and_names_no_cell",
      f"{T}::test_the_worst_surviving_cells_are_NAMED_and_skip_the_degenerate_one"]),

    # ============================================== THE ENFORCED INVARIANT
    # Without this the invariant is a docstring again, and an infinity is
    # reported as a bound rather than refused as a contract violation.
    ("anti_conservative_bound: the degenerate-cell guard is removed", P,
     b"    if (finite <= -1.0).any():",
     b"    if False:",
     [f"{T}::test_an_UNMASKED_degenerate_cell_RAISES_rather_than_reporting_infinity"]),

    # A guard that fires on the wrong side is a guard that cannot fire: -1 is
    # exactly the value BH2 excludes, so a strict `<` lets it through.
    ("anti_conservative_bound: the guard misses a loss of EXACTLY -1", P,
     b"    if (finite <= -1.0).any():",
     b"    if (finite < -1.0).any():",
     [f"{T}::test_an_UNMASKED_degenerate_cell_RAISES_rather_than_reporting_infinity"]),

    # =========================================== NOT_EVALUABLE AS A VALUE
    # BH3: a concept with no surviving draw has no verdict. Letting it into the
    # per-concept percentiles is the prohibition, exactly.
    ("anti_conservative_bound: NOT_EVALUABLE concepts re-enter the percentiles",
     P,
     b"    not_evaluable = ~usable.any(axis=1)             # per concept, BH3",
     b"    not_evaluable = np.zeros(loss.shape[0], dtype=bool)",
     [f"{T}::test_a_concept_with_no_surviving_draw_is_NOT_EVALUABLE_not_a_zero"]),

    # The arm-level statement of ignorance turned into a value a consumer can
    # read as a bound.
    ("anti_conservative_bound: an empty arm reports COMPUTED instead of NOT_EVALUABLE",
     P,
     b'        return {"status": "NOT_EVALUABLE",',
     b'        return {"status": "COMPUTED", "bound_at_worst": 1.0,',
     [f"{T}::test_an_arm_with_no_surviving_draw_ANYWHERE_is_NOT_EVALUABLE",
      f"{T}::test_an_all_nonfinite_loss_is_a_STATUS_not_a_nan_bound"]),

    # ==================================== THE SECOND DERIVATION (AC1-3)
    # A degenerate cell has no maximum in ANY seed. `any` in place of `all`
    # calls a single NaN seed a degenerate cell and would drop draws BH2 keeps.
    ("exclusion_derivations_agree: `all` over seeds becomes `any`", P,
     b"    by_stored_nan = np.isnan(cmax).all(axis=0).T            # (concepts, shifts)",
     b"    by_stored_nan = np.isnan(cmax).any(axis=0).T            # (concepts, shifts)",
     [f"{T}::test_a_partially_nan_cell_is_NOT_a_degenerate_cell"]),

    # gotcha 27: a comparison of TOTALS agrees under any permutation of either
    # axis, which is the single thing this check exists to detect.
    ("exclusion_derivations_agree: the positional compare becomes a COUNT compare",
     P,
     b"    disagree = np.argwhere(by_criterion != by_stored_nan)",
     b"    disagree = (np.zeros((0, 2), dtype=int)\n"
     b"                if by_criterion.sum() == by_stored_nan.sum()\n"
     b"                else np.argwhere(by_criterion != by_stored_nan))",
     [f"{T}::test_a_disagreement_between_the_derivations_is_REPORTED_POSITIONALLY"]),

    # A shape mismatch means the two derivations were never compared at all.
    # Passing it through would report `agree` on an unperformed comparison.
    ("exclusion_derivations_agree: the shape guard is removed", P,
     b"    if by_criterion.shape != by_stored_nan.shape:",
     b"    if False:",
     [f"{T}::test_derivations_of_incompatible_shape_are_refused"]),

    # ============================================= BH3's EMPTY CATEGORIES
    # An empty category that EXISTS cannot silently become something else on a
    # re-run. Dropping the backfill makes BH3's unoccupied cases vanish from
    # the report rather than read as zero.
    ("bh3_outcomes: the empty BH3 cases are no longer carried as zeros", P,
     b"    for case in (BH3_NONE, BH3_SOME, BH3_ONE_ARM, BH3_BOTH_ARMS):\n"
     b"        occupancy.setdefault(case, 0)",
     b"    pass",
     [f"{T}::test_bh3_keeps_the_empty_cases_as_zeros_when_nothing_is_degenerate"]),

    ("bh3_outcomes: the thin-draw count uses `<=` instead of `==`", P,
     b'    thin = {arm: int((np.asarray(s) == 1).sum())',
     b'    thin = {arm: int((np.asarray(s) <= 1).sum())',
     [f"{T}::test_bh3_counts_the_single_surviving_positive_cells_without_ruling_on_them"]),

    # ==================================================== THE NAMED CELLS
    # `worst_cells` fills excluded cells with +inf so they sort last. Filling
    # with the loss itself puts the degenerate cell back at the top of the
    # list that names what drives the bound.
    ("worst_cells: excluded cells are no longer sorted out of the way", P,
     b"    flat = np.where(usable, loss, np.inf).ravel()",
     b"    flat = loss.ravel()",
     [f"{T}::test_the_worst_surviving_cells_are_NAMED_and_skip_the_degenerate_one"]),
]

# ---------------------------------------------------------------------------
# THE CASE THAT SHOULD STILL PASS.
#
# A change that alters the source WITHOUT altering behaviour must SURVIVE. If
# it is killed, the fixtures are reacting to edits rather than to defects and
# every KILLED verdict above is worth less than it looks.
# ---------------------------------------------------------------------------
BENIGN = [
    ("BENIGN -- a comment is reworded (behaviour identical; must SURVIVE)", P,
     b"    not_evaluable = ~usable.any(axis=1)             # per concept, BH3",
     b"    not_evaluable = ~usable.any(axis=1)             # BH3, per concept",
     [f"{T}::test_a_concept_with_no_surviving_draw_is_NOT_EVALUABLE_not_a_zero",
      f"{T}::test_MASKING_CHANGES_THE_BOUND__the_fixture_discriminates"]),
]


def main() -> int:
    fixtures = sorted({t for e in MUTATIONS for t in e[4]}
                      | set(CONTROL[4])
                      | {t for e in BENIGN for t in e[4]})
    failing = [t for t in fixtures if not run_one(t)]
    if failing:
        print("BASELINE BROKEN -- these fail before any mutation:")
        for t in failing:
            print(f"   {t}")
        return 1
    print(f"baseline: {len(fixtures)}/{len(fixtures)} fixtures pass")

    # ---- THE CONTROL, FIRST. Nothing else runs if it is not detected. ----
    label, path, old, new, expect = CONTROL
    survived = apply_and_run(path, old, new, expect)
    if survived is None:
        print(f"CONTROL ABORT: anchor for {label!r} did not match exactly once. "
              "The runner cannot demonstrate that it detects anything.")
        return 2
    if survived:
        print(f"CONTROL ABORT: {label} was NOT detected (survivors: "
              f"{survived}). Every SURVIVED verdict this runner could print "
              "would be meaningless, so none is printed.")
        return 2
    print(f"control: DETECTED ({label})\n")

    print(f"{'mutation':<74} {'result':<10} killed")
    bad = 0
    for label, path, old, new, expect in MUTATIONS:
        survived = apply_and_run(path, old, new, expect)
        if survived is None:
            print(f"{label:<74} {'SKIP':<10} anchor did not match exactly once")
            bad += 1
            continue
        status = "KILLED" if not survived else "SURVIVED"
        print(f"{label:<74} {status:<10} "
              f"{len(expect) - len(survived)}/{len(expect)}")
        for t in survived:
            print(f"{'':<74} {'':<10} SURVIVOR: {t}")
        bad += bool(survived)

    print()
    for label, path, old, new, expect in BENIGN:
        survived = apply_and_run(path, old, new, expect)
        if survived is None:
            print(f"{label:<74} {'SKIP':<10} anchor did not match exactly once")
            bad += 1
            continue
        ok = len(survived) == len(expect)
        print(f"{label:<74} {'SURVIVED' if ok else 'KILLED':<10} "
              f"{len(expect) - len(survived)}/{len(expect)}")
        if not ok:
            print(f"{'':<74} {'':<10} A BEHAVIOUR-PRESERVING EDIT WAS KILLED -- "
                  "the suite is reacting to edits, not to defects.")
            bad += 1

    print()
    print("ALL MUTATIONS KILLED THEIR NAMED FIXTURES; BENIGN EDIT SURVIVED"
          if not bad else f"{bad} MUTATION(S) SURVIVED, WERE SKIPPED, OR "
                          "KILLED A BENIGN EDIT")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
