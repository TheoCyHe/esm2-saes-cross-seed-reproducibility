"""Mutation matrix for amendment BP's resolution-floor relation. NOT collected
by pytest.

Run manually:  python tests/mutation_matrix_bp.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

The runner is reused from `mutation_matrix_av` for the reason BO gives: a
runner is itself a check, and a second copy is a second thing that can silently
stop detecting (gotcha 13 extension 5).

WHY THIS FILE EXISTS. The relation is arithmetic, and every way of getting it
wrong returns a PLAUSIBLE INTEGER. `floor` for `ceil` is off by one and the
answer still looks like a rank. Dropping a `+1` from the null denominator moves
the floor by a fraction of a test. Using `<` for `<=` in the step-up loses
exactly the boundary case the whole characterisation is about. None of these
would look wrong in a table, which is the same property that makes the pipeline
this rule describes dangerous.

THE THREE SHAPES THE MUTATIONS COVER:

  * **THE CEILING OR THE DIRECTION OF THE INEQUALITY FLIPS.** `ceil` -> `floor`,
    `<=` -> `<`. Both move the floor by exactly one test, which is the margin
    by which seed 18 misses it.
  * **THE PERMUTATION `+1` IS DROPPED.** The atom becomes `1/(S*F)` rather than
    `1/(S*F+1)`, which is the same convention drift BO's matrix covers in the
    p-value, arriving here in the threshold instead.
  * **THE HISTOGRAM STOPS BEING A STEP-UP.** `k_from_atoms` returns the first
    crossing rather than the largest, or scans the blocks in the wrong order.
    Both yield a strict subset and therefore look conservative.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run  # noqa: E402
from tests.mutation_matrix_av import run_one  # noqa: E402

ROOT = os.getcwd()
C = os.path.join(ROOT, "src", "analysis", "bh_resolution_floor.py")
T = "tests/test_bh_resolution_floor.py"

# THE CONTROL: every rank threshold is 1, so nothing is ever floor-limited.
# It cannot fail to break the suite -- the founding-instance fixture asserts
# k_min == 218 exactly. If the runner does not detect this, no SURVIVED
# verdict below would mean anything, so none is printed.
CONTROL = (
    "CONTROL -- atom_rank_threshold returns 1 before any branch",
    C,
    b"    S_c = n_shifts if n_shifts_concept is None else n_shifts_concept\n",
    b"    S_c = n_shifts if n_shifts_concept is None else n_shifts_concept\n"
    b"    if True:\n        return 1\n",
    [f"{T}::test_founding_instance_floor_is_218",
     f"{T}::test_block_structure_is_j_times_C_and_then_stops",
     f"{T}::test_a_degenerate_draw_RAISES_the_threshold"],
)

MUTATIONS = [
    # ============ THE CEILING OR THE INEQUALITY FLIPS
    ("ceil -> floor in the rank threshold (off by one at the floor)", C,
     b"    return math.ceil(p * m / _fr(q))",
     b"    return math.floor(p * m / _fr(q))",
     [f"{T}::test_founding_instance_floor_is_218",
      f"{T}::test_block_structure_is_j_times_C_and_then_stops",
      f"{T}::test_k_min_equals_C_on_BOTH_SIDES_of_its_boundary"]),

    ("k_from_atoms uses < instead of <=, losing the boundary case", C,
     b"        if _fr(p) <= Fraction(cum, m) * qq:",
     b"        if _fr(p) < Fraction(cum, m) * qq:",
     # NOT the at-floor fixtures: at the real parameters the critical value at
     # cum = 218 is exactly 1/204,800 and the atom is 1/204,801, so the
     # inequality is STRICT and both conventions agree. Only a constructed tie
     # can exhibit this branch (first-run survivor, TEST-DEFECT, repaired).
     [f"{T}::TestKFromAtoms::test_an_EXACT_TIE_between_p_and_its_critical_value_is_REJECTED"]),

    # ============ THE PERMUTATION +1 IS DROPPED
    ("the null denominator loses its +1", C,
     b"    p = Fraction(j, S_c * n_features + 1)",
     b"    p = Fraction(j, S_c * n_features)",
     [f"{T}::test_k_min_equals_C_on_BOTH_SIDES_of_its_boundary",
      f"{T}::test_block_structure_is_j_times_C_and_then_stops"]),

    ("the degenerate concept's own draw count is ignored", C,
     b"    S_c = n_shifts if n_shifts_concept is None else n_shifts_concept",
     b"    S_c = n_shifts",
     [f"{T}::test_a_degenerate_draw_RAISES_the_threshold"]),

    # ============ THE HISTOGRAM STOPS BEING A STEP-UP
    ("k_from_atoms returns the FIRST crossing, not the largest", C,
     b"        if _fr(p) <= Fraction(cum, m) * qq:\n            best = cum\n",
     b"        if _fr(p) <= Fraction(cum, m) * qq:\n"
     b"            if best == 0:\n                best = cum\n",
     [f"{T}::TestKFromAtoms::test_TWO_ATOM_branch",
      f"{T}::TestKFromAtoms::test_SECOND_BLOCK_relation"]),

    ("k_from_atoms walks the blocks largest-p first", C,
     b"    for p, n in sorted(atoms, key=lambda t: t[0]):",
     b"    for p, n in sorted(atoms, key=lambda t: t[0], reverse=True):",
     # The carry-over fixture is NOT named here: with two atoms both walk
     # directions reach the same total, so it cannot detect order at all
     # (first-run survivor, TEST-DEFECT). A three-atom fixture replaces it.
     [f"{T}::TestKFromAtoms::test_SINGLE_ATOM_branch_the_one_number_formulation_cannot_express",
      f"{T}::TestKFromAtoms::test_block_ORDER_matters_and_this_fixture_can_detect_it"]),

    ("k_from_atoms counts a block without accumulating rank", C,
     b"        cum += n\n",
     b"        cum = n\n",
     [f"{T}::TestKFromAtoms::test_TWO_ATOM_branch",
      f"{T}::TestKFromAtoms::test_the_second_atom_can_carry_the_first_over_the_line"]),

    ("k_min_equals_C compares against the family size, not the pool", C,
     b"    return k_min(n_concepts, n_features, n_shifts, q) == n_concepts",
     b"    return k_min(n_concepts, n_features, n_shifts, q) <= n_concepts",
     [f"{T}::test_k_min_equals_C_on_BOTH_SIDES_of_its_boundary"]),
]


def main() -> int:
    fixtures = sorted({t for e in MUTATIONS for t in e[4]} | set(CONTROL[4]))
    failing = [t for t in fixtures if not run_one(t)]
    if failing:
        print("BASELINE BROKEN -- these fail before any mutation:")
        for t in failing:
            print(f"   {t}")
        return 1
    print(f"baseline: {len(fixtures)}/{len(fixtures)} fixtures pass")

    label, path, old, new, expect = CONTROL
    survived = apply_and_run(path, old, new, expect)
    if survived is None:
        print(f"CONTROL ABORT: anchor for {label!r} did not match exactly "
              "once. The runner cannot demonstrate that it detects anything.")
        return 2
    if survived:
        print(f"CONTROL ABORT: {label} was NOT detected (survivors: "
              f"{survived}). Every SURVIVED verdict this runner could print "
              "would be meaningless, so none is printed.")
        return 2
    print(f"control: DETECTED ({label})\n")

    print(f"{'mutation':<62} {'result':<10} killed")
    bad = 0
    for label, path, old, new, expect in MUTATIONS:
        survived = apply_and_run(path, old, new, expect)
        if survived is None:
            print(f"{label:<62} {'SKIP':<10} anchor did not match exactly once")
            bad += 1
            continue
        status = "KILLED" if not survived else "SURVIVED"
        print(f"{label:<62} {status:<10} "
              f"{len(expect) - len(survived)}/{len(expect)}")
        for t in survived:
            print(f"{'':<62} {'':<10} SURVIVOR: {t}")
        bad += bool(survived)

    print()
    print("ALL MUTATIONS KILLED THEIR NAMED FIXTURES" if not bad
          else f"{bad} MUTATION(S) NOT FULLY KILLED -- real coverage gap")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
