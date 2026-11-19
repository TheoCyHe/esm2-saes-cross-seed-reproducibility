"""Mutation matrix for amendment BT's component 4 (the thinness grid). NOT
collected by pytest.

Run manually:  python tests/mutation_matrix_bt_grid.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

The runner is reused from `mutation_matrix_av` -- a runner is itself a check
and a second copy is a second thing that can silently stop detecting (gotcha 13
extension 5). It carries BP's post-restore `__pycache__` purge (entry 122).

WHY THIS FILE EXISTS. Every way of getting the grid wrong returns a COMPLETE,
PLAUSIBLE COMPOSITION TABLE. A restriction applied to the wrong half of the
family still sums to the population. A decomposition that assigns both causes
to the same set difference still adds up to the net change -- and that is the
sharpest point here: THE SUMMATION CHECK CANNOT FAIL ON WELL-FORMED INPUTS.

    Given `mid` subset of `prev`, `gained - lost == net` and
    `gained_threshold - lost_threshold - lost_pool == net` are both ALGEBRAIC
    IDENTITIES, not measurements. Criterion 9 asks for the summation and the
    summation is there, but it is an IMPLEMENTATION guard, not evidence about
    the data (gotcha 25a: a check that can only pass). What can actually fail
    is the SUBSET relation, which is why `gained_pool == 0` is asserted rather
    than assumed -- and why the mutations below attack the mask ARITHMETIC,
    where a slip does break the sums, rather than the sums themselves.

THE FOUR SHAPES THE MUTATIONS COVER:

  * **THE BOUNDARY MOVES.** `>=` becomes `>`, which drops the degenerate
    concept a grid point early and silently removes the only point at which
    the two-floor structure can fire.
  * **THE FAMILY IS WRONG.** The restriction ignores `keep`, or ignores
    `not_evaluable`, or is applied after the BH sort instead of before. Each
    changes `m` and therefore every rejection, and none produces an impossible
    number.
  * **A CAUSE IS MISATTRIBUTED.** `lost_pool` is computed from the wrong pair
    of masks, or the counterfactual's subset guard is dropped so a malformed
    `mid` passes through and the split silently stops meaning what it says.
  * **CLEARANCE STOPS SAYING WHICH CASE IT IS IN.** The `k_min == C`
    equivalence is hardcoded true, so the two readings could diverge behind
    one number.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run  # noqa: E402
from tests.mutation_matrix_av import run_one  # noqa: E402

ROOT = os.getcwd()
C = os.path.join(ROOT, "src", "analysis", "grid_composition.py")
T = "tests/test_grid_composition.py"

BOUND = f"{T}::test_the_boundary_is_inclusive_and_five_positives_SURVIVES_T5"
T1 = f"{T}::test_T1_retains_every_concept_with_at_least_one_positive"
OUT = f"{T}::test_a_test_outside_the_retained_pool_is_FALSE_not_a_non_rejection"
NE = f"{T}::test_the_family_excludes_a_NOT_EVALUABLE_concept_even_when_retained"
LOOSE = (f"{T}::test_THE_LOOSENING_a_p_value_refused_in_a_big_family_passes"
         "_in_a_small_one")
UNRES = f"{T}::test_an_unrestricted_call_matches_bh_reject_on_the_whole_family"
EMPTY = f"{T}::test_an_empty_family_rejects_nothing_and_says_so"
HAND = f"{T}::test_the_decomposition_separates_the_two_causes_counted_by_hand"
SUBSET = f"{T}::test_a_counterfactual_that_ADDS_a_feature_is_REFUSED"
PURE = f"{T}::test_pure_pool_removal_attributes_everything_to_the_pool"
NOMOVE = f"{T}::test_no_movement_at_all_is_all_zeros_and_does_not_raise"
CLEAR = f"{T}::test_clearance_reports_the_equivalence_as_CONDITIONAL_not_as_a_fact"
GRIDT = f"{T}::test_the_grid_is_BI1s_and_is_written_out"

# THE CONTROL: the restriction never removes anything. It cannot fail to break
# the suite -- the boundary fixture asserts a False at exactly four positives.
# If the runner does not detect this, no SURVIVED verdict below would mean
# anything, so none is printed.
CONTROL = (
    "CONTROL -- retained_mask keeps every concept at every threshold",
    C,
    b"    return n >= int(threshold)\n",
    b"    return np.ones(n.shape, dtype=bool)\n",
    [BOUND],
)

MUTATIONS = [
    # ============ THE BOUNDARY MOVES
    ("the boundary becomes STRICT: `n > T` drops the five-positive concept a "
     "grid point early", C,
     b"    return n >= int(threshold)",
     b"    return n > int(threshold)",
     [BOUND, T1]),

    ("the boundary is off by one the other way: `n >= T - 1`", C,
     b"    return n >= int(threshold)",
     b"    return n >= int(threshold) - 1",
     [BOUND]),

    ("the grid loses a point", C,
     b"GRID = (1, 5, 10, 30, 100)",
     b"GRID = (1, 5, 10, 100)",
     [GRIDT]),

    # ============ THE FAMILY IS WRONG
    ("the family ignores `keep` -- the restriction stops restricting and every "
     "grid point is the unrestricted one", C,
     b"    in_family = kp & ~ne",
     b"    in_family = ~ne",
     [OUT, LOOSE]),

    ("the family ignores NOT_EVALUABLE -- a concept with no reference "
     "distribution enters `m` and drags NaNs into the sort", C,
     b"    in_family = kp & ~ne",
     b"    in_family = kp",
     [NE]),

    ("the family is the UNION rather than the intersection", C,
     b"    in_family = kp & ~ne",
     b"    in_family = kp | ~ne",
     [OUT, NE, LOOSE]),

    ("rejections are written at FULL WIDTH, so a concept outside the pool "
     "reads as rejected", C,
     b"    reject = np.zeros(p.shape, dtype=bool)\n"
     b"    reject[fam] = res[\"reject\"]",
     b"    reject = np.zeros(p.shape, dtype=bool)\n"
     b"    reject[:] = True",
     [OUT, NE]),

    ("the empty family raises instead of returning a shaped answer -- a state "
     "the grid can reach becomes a crash", C,
     b"    if not fam.any():\n"
     b"        return {\"reject\": np.zeros(p.shape, dtype=bool), \"n_tests\": 0,",
     b"    if not fam.any():\n"
     b"        raise ValueError(\"empty\")\n"
     b"    if False:\n"
     b"        return {\"reject\": np.zeros(p.shape, dtype=bool), \"n_tests\": 0,",
     [EMPTY]),

    # UNRES IS DELIBERATELY NOT NAMED HERE, and the matrix is what found that.
    # The unrestricted fixture retains every concept and marks none
    # NOT_EVALUABLE, so `p.size` IS the family size and this mutation is a
    # no-op against it -- a fixture that cannot exhibit the branch it would be
    # named for is not evidence about that branch (criterion 2, gotcha 16).
    # It survived on the first matrix run and is classified TEST-DEFECT in the
    # MATRIX, not a coverage gap in the module: OUT, NE and LOOSE all kill it.
    ("`n_tests` is reported as the FULL grid rather than the family", C,
     b"    return {\"reject\": reject, \"n_tests\": res[\"n_tests\"],",
     b"    return {\"reject\": reject, \"n_tests\": int(p.size),",
     [OUT, NE, LOOSE]),

    # ============ A CAUSE IS MISATTRIBUTED
    # PURE IS DELIBERATELY NOT NAMED HERE, for the same reason and found the
    # same way. In the pure-pool-removal fixture `mid` and `next` are the SAME
    # SET by construction, so `p & ~m` and `p & ~n` are equal and the mutation
    # is invisible to it. Classified TEST-DEFECT in the MATRIX; HAND, where
    # mid and next differ, kills it.
    ("`lost_pool` is computed from prev-vs-next, collapsing the two causes "
     "into one and making the counterfactual decorative", C,
     b"    lost_pool = int(np.count_nonzero(p_ & ~m_))",
     b"    lost_pool = int(np.count_nonzero(p_ & ~n_))",
     [HAND]),

    ("`lost_threshold` and `gained_threshold` are measured against `prev` "
     "instead of the counterfactual", C,
     b"    lost_threshold = int(np.count_nonzero(m_ & ~n_))\n"
     b"    gained_threshold = int(np.count_nonzero(n_ & ~m_))",
     b"    lost_threshold = int(np.count_nonzero(p_ & ~n_))\n"
     b"    gained_threshold = int(np.count_nonzero(n_ & ~p_))",
     [HAND]),

    ("the SUBSET guard is dropped, so a malformed counterfactual passes and "
     "the causal split silently stops meaning what it says", C,
     b"    if gained_pool:\n"
     b"        raise DecompositionViolation(",
     b"    if False:\n"
     b"        raise DecompositionViolation(",
     [SUBSET]),

    ("`gained_pool` is hardcoded to zero rather than counted, so the guard "
     "can never fire", C,
     b"    gained_pool = int(np.count_nonzero(m_ & ~p_))",
     b"    gained_pool = 0",
     [SUBSET]),

    ("`net` is computed from the counterfactual, so the two-way summation "
     "stops describing the grid step", C,
     b"    net = int(n_.sum()) - int(p_.sum())",
     b"    net = int(n_.sum()) - int(m_.sum())",
     [HAND, PURE]),

    ("`lost` and `gained` are swapped -- the two-way pair still sums to a net "
     "of the opposite sign", C,
     b"    lost = int(np.count_nonzero(p_ & ~n_))\n"
     b"    gained = int(np.count_nonzero(n_ & ~p_))",
     b"    lost = int(np.count_nonzero(n_ & ~p_))\n"
     b"    gained = int(np.count_nonzero(p_ & ~n_))",
     [HAND, PURE]),

    # ============ CLEARANCE STOPS SAYING WHICH CASE IT IS IN
    ("the k_min == C equivalence is hardcoded TRUE, so the two readings could "
     "diverge behind one number", C,
     b"    equivalent = bool(floor == n_concepts)",
     b"    equivalent = True",
     [CLEAR]),

    ("clearance divides by the concept count rather than by k_min, which is "
     "the same number today and would stop being so if the floor moved", C,
     b"    ratio = at_floor_total / floor",
     b"    ratio = at_floor_total / n_concepts",
     [CLEAR]),

    ("a non-positive floor stops being refused and clearance returns inf", C,
     b"    if floor <= 0:\n        raise ValueError(",
     b"    if False:\n        raise ValueError(",
     [f"{T}::test_clearance_REFUSES_a_non_positive_floor"]),
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
    print(f"control: DETECTED ({label})")
    print()

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
    if bad:
        print(f"{bad} MUTATION(S) NOT FULLY KILLED -- real coverage gap. Each "
              "must be classified code-defect or test-defect.")
    else:
        print(f"ALL {len(MUTATIONS)} MUTATIONS KILLED THEIR NAMED FIXTURES, "
              "control detected.")
        print("A clean first matrix is itself a claim about the matrix "
              "(criterion 3) -- scrutiny, not celebration.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
