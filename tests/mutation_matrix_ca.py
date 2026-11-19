"""Mutation matrix for the ledger's STRUCTURAL_CHECK exclusion -- amendment CA.
NOT collected by pytest.

Run manually:  python tests/mutation_matrix_ca.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

The runner is reused from `mutation_matrix_av` -- control-first, `__pycache__`
purged between invocations, byte-compared restore, decisions on pytest's EXIT
CODE. A second copy of a runner is a second thing that can silently stop
detecting (gotcha 13 extension 5).

WHAT THIS FILE EXISTS FOR. CA-a excludes ONE entry from a tally. The whole
change is an exclusion, and the characteristic defect of an exclusion mechanism
is that it excludes MORE or LESS than it says while the posted line still looks
like a plausible ledger line. `procedure-behaviour 9/13` is exactly as
publication-shaped as `10/14`, `8/12` or `9/14`, and nothing downstream of the
deriver would disagree with any of them.

THE MUTATIONS ARE CHOSEN AGAINST THAT, and three of them target the two
directions CA-c names explicitly: a mechanism that excludes NOTHING (the tally
never moves) and one that excludes EVERYTHING (the tally moves for the wrong
reason, and the number here happens to look right). Both are the "can only
pass" shape gotcha 25a names -- a check that cannot distinguish its own success
from a broken instrument.

TWO MUTATIONS TARGET THE PART THAT IS NOT ARITHMETIC AT ALL: the entry must
stay REPORTED, keep its category, and keep the record that the check HELD. An
exclusion that silently drops the row is how a prediction leaves a ledger by
attrition, which this module's own docstring already names as the channel it
exists to close -- and a HELD check reported without its verdict is
indistinguishable from a retraction.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run  # noqa: E402

ROOT = os.getcwd()
M = os.path.join(ROOT, "src", "analysis", "prediction_ledger.py")
T = "tests/test_prediction_ledger.py"

# THE CONTROL: STRUCTURAL_CHECK mapped back to HIT before the branch. This is
# the pre-CA behaviour exactly -- the posted line goes back to 10/14, a number
# this repository published, so the mutant is not a strawman and no "looks
# wrong" heuristic catches it.
#
# THE FIRST CONTROL WRITTEN HERE WAS A NO-OP AND THE RUNNER CAUGHT IT, which is
# recorded rather than quietly replaced. It added `"STRUCTURAL_CHECK": (1, 1)`
# to `SCORING` -- which changes nothing, because `derive` tests membership of
# NON_SCORING FIRST and `continue`s. The mutant was well-formed, plausible, and
# altered no behaviour; every SURVIVED verdict printed under it would have been
# meaningless. This is exactly why the control runs first (gotcha 13
# extension 5), and it is worth noting that the failure was in the MUTATION,
# not in the fixtures -- three fixtures were named and all three were right.
CONTROL = (
    "CONTROL -- STRUCTURAL_CHECK tallies as a HIT again (the pre-CA behaviour)",
    M,
    b'        if outcome in NON_SCORING:\n',
    b'        if outcome == "STRUCTURAL_CHECK":\n'
    b'            outcome = "HIT"\n'
    b'        if outcome in NON_SCORING:\n',
    [f"{T}::test_the_shipped_procedure_behaviour_bin_is_9_of_13_after_the_exclusion",
     f"{T}::test_a_structural_check_moves_neither_half_of_its_tally",
     f"{T}::test_the_shipped_BI3a_entry_is_excluded_and_records_that_the_check_HELD"],
)

MUTATIONS = [
    # ========================================== 1. THE MECHANISM EXCLUDES NOTHING
    # STRUCTURAL_CHECK weighted (0, 1) -- it stops adding a hit but still adds a
    # denominator, so the bin becomes 9/14. That is a plausible ledger line, it
    # moves in the direction a reader expects, and it is the arithmetic of a
    # MISS applied to an entry the ratification says HELD. The distinction
    # between "excluded" and "downgraded to a miss" is the entire content of
    # CA-a's "THE OUTCOME IS NOT RETRACTED".
    ("STRUCTURAL_CHECK downgraded to a MISS rather than excluded", M,
     b'        if outcome in NON_SCORING:\n',
     b'        if outcome == "STRUCTURAL_CHECK":\n'
     b'            outcome = "MISS"\n'
     b'        if outcome in NON_SCORING:\n',
     [f"{T}::test_a_structural_check_is_distinguishable_from_a_hit_AND_from_a_miss",
      f"{T}::test_the_shipped_procedure_behaviour_bin_is_9_of_13_after_the_exclusion"]),

    # ====================================== 2. THE MECHANISM EXCLUDES EVERYTHING
    # Every entry in the same category as an excluded one is also excluded.
    # THE SHIPPED LINE STILL COMES OUT 9/13 FOR THE OTHER BINS AND WRONG ONLY
    # HERE -- which is why CA-c requires a POSITIVE control and not just a
    # movement check. A mechanism that excluded everything would satisfy "the
    # tally moved" perfectly.
    ("every entry sharing an excluded entry's category is also excluded", M,
     b'        if outcome in NON_SCORING:\n',
     b'        if outcome in NON_SCORING or any(\n'
     b'                q.get("outcome") in NON_SCORING\n'
     b'                and q.get("category") == p.get("category")\n'
     b'                for q in doc["predictions"]):\n',
     [f"{T}::test_POSITIVE_CONTROL_excluding_one_entry_leaves_its_neighbours_tallied",
      f"{T}::test_BI3b_is_still_tallied_because_its_footprint_is_empirical"]),

    # ============================ 3. THE EXCLUDED ENTRY IS DROPPED, NOT REPORTED
    # `continue` without appending. The tallies are IDENTICAL -- 9/13 either
    # way -- and the only difference is that BI3(a) vanishes from the output
    # entirely. This module's docstring names that as the channel gotcha 26's
    # companion rule closes: dropping a prediction lets misses leave the ledger
    # by attrition. Here it would also erase a ratified decision's own record.
    ("an excluded entry is skipped silently instead of reported", M,
     b'            excluded.append(row)\n            continue\n',
     b'            continue\n',
     [f"{T}::test_a_structural_check_is_reported_never_dropped",
      f"{T}::test_the_shipped_BI3a_entry_is_excluded_and_records_that_the_check_HELD",
      f"{T}::test_a_not_sealed_prediction_moves_nothing_but_is_reported"]),

    # ================= 4. THE VERDICT IS DROPPED -- EXCLUSION READS AS RETRACTION
    # The row survives, the tally is right, and the ONE thing that distinguishes
    # "this check ran and held" from "this claim was withdrawn" is gone. Nothing
    # numeric changes, so no arithmetic fixture can see it. CA-a puts both
    # halves in the record precisely because they are separable like this.
    ("the check's HELD verdict is not carried onto the reported row", M,
     b'            if p.get("check_verdict"):\n'
     b'                row["check_verdict"] = p["check_verdict"]\n',
     b'',
     [f"{T}::test_a_structural_check_keeps_its_category_and_its_verdict_on_the_row",
      f"{T}::test_the_shipped_BI3a_entry_is_excluded_and_records_that_the_check_HELD"]),

    # =================== 5. THE CATEGORY IS DROPPED FROM THE EXCLUDED ROW
    # CA-b requires the category be RETAINED, because an absent field is not the
    # same as a considered one -- a later reader cannot tell whether the bin was
    # judged and set aside or never assigned. Again nothing numeric moves.
    ("the excluded row loses its category", M,
     b'            if p.get("category"):\n'
     b'                row["category"] = p["category"]\n',
     b'',
     [f"{T}::test_a_structural_check_keeps_its_category_and_its_verdict_on_the_row"]),

    # ============== 6. AN UNRECOGNISED OUTCOME BECOMES A SILENT EXCLUSION
    # The refusal replaced by a skip. Adding a value to NON_SCORING is exactly
    # the kind of edit that makes this failure mode live: a typo in a future
    # entry's outcome would then remove it from the ledger with no error at all.
    # The pre-existing guard must survive the addition.
    ("an unrecognised outcome is skipped instead of refused", M,
     b'        if outcome not in SCORING:\n            raise SystemExit(\n',
     b'        if outcome not in SCORING:\n            continue\n        if False:\n            raise SystemExit(\n',
     [f"{T}::test_an_unrecognised_outcome_is_refused_not_skipped"]),
]


def main() -> int:
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
