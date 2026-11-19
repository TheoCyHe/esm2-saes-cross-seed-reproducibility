"""Mutation matrix for amendment BU's component 5 (the firing-rate-matched
paired arm). NOT collected by pytest.

Run manually:  python tests/mutation_matrix_bu_matched.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

The runner is reused from `mutation_matrix_av` -- a runner is itself a check and
a second copy is a second thing that can silently stop detecting (gotcha 13
extension 5). It carries BP's post-restore `__pycache__` purge (entry 122).

WHY THIS FILE EXISTS. Every way of getting this arm wrong returns a COMPLETE,
PLAUSIBLE TABLE of pair counts and p-values. A tie-break that quietly keeps the
first candidate produces a slightly different estimate and no error. A seed with
no treated features that is skipped rather than carried produces a denominator
of 17 that reads exactly like a denominator of 20. A caliper that has become
exclusive drops a handful of pairs nobody counts.

THE FIVE SHAPES THE MUTATIONS COVER:

  * **THE MATCHING RULE SLIPS.** Rule 1 stops using all tied candidates, or the
    caliper stops being the pinned width, or its boundary flips.
  * **THE ESTIMAND STOPS BEING A SET CONTRAST.** The matched set is summarised
    by its first member rather than its mean, or an unmatchable feature is
    given a contrast of zero instead of none.
  * **AN IGNORANCE CATEGORY BECOMES A NUMBER.** The all-zero-differences cell
    reports p = 1, the empty cell reports a statistic, or a NaN contrast is
    counted as a pair.
  * **THE FLOOR STOPS BEING THE FLOOR.** gotcha 42's relation is off by a
    factor of two, or the n = 0 case returns something reachable.
  * **§5b IS NOT IMPLEMENTED.** A seed with no treated feature is dropped from
    the denominator, or the seed axis is joined by position rather than by
    label, or the corrector exclusion does not bite.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run  # noqa: E402
from tests.mutation_matrix_av import run_one  # noqa: E402

ROOT = os.getcwd()
C = os.path.join(ROOT, "src", "analysis", "matched_pair_arm.py")
T = "tests/test_matched_pair_arm.py"

TIED = f"{T}::test_ALL_tied_candidates_are_returned_and_none_is_preferred"
UNIQ = f"{T}::test_a_unique_nearest_control_gives_a_singleton_set"
UNMATCH = (f"{T}::test_a_treated_feature_with_NOTHING_inside_the_caliper_gets"
           "_an_EMPTY_set")
REPL = f"{T}::test_matching_is_WITH_replacement_one_control_serving_two_treated"
NOFIRE = f"{T}::test_a_treated_feature_with_NO_firing_rate_matches_NOTHING"
INCL = f"{T}::test_the_caliper_boundary_is_INCLUSIVE_where_float64_can_express_it"
UNREACH = f"{T}::test_the_boundary_is_NOT_REACHABLE_at_the_PINNED_caliper_of_0_1"
AGREE = (f"{T}::test_match_sets_agrees_with_match_pool_diagnostics_on"
         "_multiplicities")

MEAN = f"{T}::test_the_contrast_averages_the_matched_SET_with_equal_weights"
NOCON = (f"{T}::test_an_unmatchable_treated_feature_has_NO_contrast_not_a_zero"
         "_one")

HAND3 = f"{T}::test_signed_rank_COMPUTED_against_a_hand_worked_exact_test"
MIXED = f"{T}::test_signed_rank_COMPUTED_on_a_mixed_sign_case_worked_by_hand"
NOPAIR = f"{T}::test_signed_rank_NO_PAIRS_is_its_own_branch_not_a_p_value"
ALLZERO = (f"{T}::test_signed_rank_reports_ALL_DIFFERENCES_ZERO_rather_than_p"
           "_equals_1")
NANDROP = f"{T}::test_NaN_contrasts_are_DROPPED_AND_COUNTED_never_silently_included"
REACH = f"{T}::test_every_TEST_BRANCH_is_reachable"

FLOOR5 = f"{T}::test_the_exact_test_floor_is_2_to_the_1_minus_n[5-0.0625]"
FLOOR0 = f"{T}::test_the_exact_test_floor_is_2_to_the_1_minus_n[0-1.0]"
ATTAIN = f"{T}::test_the_floor_is_ATTAINED_by_the_most_extreme_sample"

BINS = f"{T}::test_bin_index_places_rates_in_corrector_scan_bins"
BINEDGE = f"{T}::test_a_rate_exactly_ON_an_edge_lands_in_the_UPPER_bin"
BINOUT = f"{T}::test_a_rate_outside_every_bin_is_MINUS_ONE_and_not_clamped"

REUSE = (f"{T}::test_reuse_profile_counts_a_shared_control_and_the_treated_that"
         "_share_it")
NOREUSE = f"{T}::test_reuse_profile_with_no_reuse_reports_zero_sharing_not_None"

SPACE = f"{T}::test_the_outcome_space_counts_sum_to_the_treated_population"
UNKNOWN = f"{T}::test_the_outcome_space_REFUSES_an_unknown_branch"
UNASSIGNED = f"{T}::test_the_outcome_space_REFUSES_an_unassigned_cell"

EMPTYSEED = (f"{T}::test_a_seed_with_NO_treated_features_is_carried_as_its_own"
             "_category")
DENOM = f"{T}::test_the_empty_seed_is_NOT_dropped_from_the_stated_denominator"
ALLEMPTY = (f"{T}::test_a_run_where_EVERY_seed_is_empty_reports_zero_and_does"
            "_not_crash")
POSJOIN = f"{T}::test_the_driver_REFUSES_a_positional_seed_join"
CSMISS = f"{T}::test_the_driver_REFUSES_a_corrector_scan_missing_a_seed"
CSBITE = (f"{T}::test_a_flagged_corrector_is_actually_EXCLUDED_from_the_without"
          "_variant")

# THE CONTROL: rule 1 is abandoned and only the first tied candidate is kept.
# It cannot fail to break the suite -- the tie fixture asserts BOTH indices are
# returned. If the runner does not detect this, no SURVIVED verdict below would
# mean anything, so none is printed.
CONTROL = (
    "CONTROL -- match_sets keeps only the FIRST tied candidate",
    C,
    b"        out.append(c_idx[inside & (d == dm)].astype(np.int64))\n",
    b"        out.append(c_idx[inside & (d == dm)].astype(np.int64)[:1])\n",
    [TIED],
)

MUTATIONS = [
    # ============ THE MATCHING RULE SLIPS
    ("rule 1 keeps only the LAST tied candidate instead of all of them", C,
     b"        out.append(c_idx[inside & (d == dm)].astype(np.int64))",
     b"        out.append(c_idx[inside & (d == dm)].astype(np.int64)[-1:])",
     [TIED, AGREE]),

    ("the tie test becomes a TOLERANCE, inventing ties the arithmetic does not "
     "have", C,
     b"        out.append(c_idx[inside & (d == dm)].astype(np.int64))",
     b"        out.append(c_idx[inside & (d <= dm + 0.01)].astype(np.int64))",
     [UNIQ, AGREE]),

    ("the caliper boundary becomes EXCLUSIVE, contradicting the committed "
     "`<=` predicate", C,
     b"        inside = d <= caliper",
     b"        inside = d < caliper",
     [INCL]),

    ("the caliper is widened past its pinned width, so the unreachable "
     "boundary becomes reachable and more pairs enter", C,
     b"        inside = d <= caliper",
     b"        inside = d <= caliper + 0.1",
     [UNREACH, UNMATCH]),

    # THE FIRST FORM OF THIS MUTATION -- deleting the `isfinite` GUARD -- IS AN
    # EQUIVALENT MUTANT and was classified as such rather than left as a
    # survivor. With the guard removed a NaN treated value still matches
    # nothing, because `abs(control - nan) <= caliper` is False for every
    # control under IEEE comparison, so `inside.any()` is False and the empty
    # set is appended by the next branch instead. No fixture can distinguish
    # the two, and that is a property of the arithmetic rather than a hole in
    # the suite. What IS observable is the guard returning a PARTNER, so that
    # is what is mutated here.
    ("a treated feature with no firing rate is handed a partner rather than "
     "nothing", C,
     b"        if not np.isfinite(t[i]):\n"
     b"            out.append(np.zeros(0, dtype=np.int64))\n"
     b"            continue",
     b"        if not np.isfinite(t[i]):\n"
     b"            out.append(c_idx[:1].astype(np.int64))\n"
     b"            continue",
     [NOFIRE]),

    ("an unmatchable treated feature silently borrows the whole control pool", C,
     b"        if not inside.any():\n"
     b"            out.append(np.zeros(0, dtype=np.int64))\n"
     b"            continue",
     b"        if not inside.any():\n"
     b"            out.append(c_idx.astype(np.int64))\n"
     b"            continue",
     [UNMATCH, AGREE]),

    ("a non-positive caliper stops being refused", C,
     b"    if not np.isfinite(caliper) or caliper <= 0:",
     b"    if False:",
     [f"{T}::test_match_sets_REFUSES_a_non_positive_caliper"]),

    # ============ THE ESTIMAND STOPS BEING A SET CONTRAST
    ("the matched set is summarised by its FIRST member rather than its mean, "
     "so tied candidates stop being weighted equally", C,
     b"            out[i] = rt[i] - float(rc[s].mean())",
     b"            out[i] = rt[i] - float(rc[s[0]])",
     [MEAN]),

    ("the contrast is computed the wrong way round, so a positive result "
     "reads as a negative one", C,
     b"            out[i] = rt[i] - float(rc[s].mean())",
     b"            out[i] = float(rc[s].mean()) - rt[i]",
     [MEAN, NOCON]),

    ("an unmatchable treated feature is given a contrast of ZERO instead of "
     "none -- a feature that was never compared enters as an observation of "
     "no difference", C,
     b"    out = np.full(rt.size, np.nan, dtype=np.float64)",
     b"    out = np.zeros(rt.size, dtype=np.float64)",
     [NOCON]),

    ("a length mismatch between the contrasts and the matched sets stops "
     "being refused", C,
     b"    if len(sets) != rt.size:",
     b"    if False:",
     [f"{T}::test_set_contrast_REFUSES_a_length_mismatch"]),

    # ============ AN IGNORANCE CATEGORY BECOMES A NUMBER
    ("the all-zero-differences cell reports p = 1 -- a ranking predicate "
     "applied to a constant, reported as a real non-rejection", C,
     b'        base.update({"branch": "ALL_DIFFERENCES_ZERO", "statistic": None,\n'
     b'                     "p_value": None, "method": None})',
     b'        base.update({"branch": "COMPUTED", "statistic": 0.0,\n'
     b'                     "p_value": 1.0, "method": "exact"})',
     [ALLZERO, REACH]),

    ("the empty cell claims a statistic instead of naming itself NO_PAIRS", C,
     b'        base.update({"branch": "NO_PAIRS", "statistic": None, '
     b'"p_value": None,\n                     "method": None})',
     b'        base.update({"branch": "COMPUTED", "statistic": 0.0, '
     b'"p_value": 1.0,\n                     "method": "exact"})',
     [NOPAIR, REACH]),

    ("NaN contrasts are counted as pairs rather than dropped and counted", C,
     b"    finite = np.isfinite(x)\n"
     b"    v = x[finite]",
     b"    finite = np.ones(x.shape, dtype=bool)\n"
     b"    v = np.nan_to_num(x)",
     [NANDROP]),

    ("zeros are kept in the ranked sample, contradicting the declared "
     "zero_method and changing every p-value in a cell that has any", C,
     b'    nz = v[v != 0.0]',
     b'    nz = v',
     [ALLZERO]),

    ("the two-sided test becomes one-sided, halving every p-value", C,
     b'    res = wilcoxon(v, zero_method="wilcox", alternative="two-sided",',
     b'    res = wilcoxon(v, zero_method="wilcox", alternative="greater",',
     [HAND3, MIXED, ATTAIN]),

    ("the exact method is abandoned for the normal approximation at every "
     "size, so the small cells stop being exact", C,
     b'    method = "exact" if nz.size <= 25 else "approx"',
     b'    method = "approx"',
     [HAND3, MIXED, ATTAIN]),

    # ============ THE FLOOR STOPS BEING THE FLOOR
    ("gotcha 42's relation is off by a factor of two -- the floor becomes "
     "2**-n and a cell that cannot reach 0.05 looks as though it can", C,
     b"    return float(2.0 ** (1 - n))",
     b"    return float(2.0 ** (-n))",
     [FLOOR5, ATTAIN]),

    ("the n = 0 case returns a reachable p rather than 1.0, so a cell with no "
     "test at all reads as one that could have rejected", C,
     b"    if n == 0:\n        return 1.0",
     b"    if n == 0:\n        return 0.0",
     [FLOOR0]),

    # ============ THE BINS
    # THE HALF-OPEN CONVENTION WAS UNOBSERVABLE UNDER THE FIRST IMPLEMENTATION
    # and the matrix is what established it: a loop of `(r >= lo) & (r < hi)`
    # writes and an inclusive-upper version of the same loop return IDENTICAL
    # answers, because an edge value matches two bins and the later write wins.
    # That is a code defect -- a convention nothing can test is a convention
    # nobody is keeping -- and `bin_index` was rewritten as a single
    # `searchsorted` so that `side` IS the convention. This mutation is the one
    # that could not previously exist.
    ("the bin search side flips, so a rate exactly on an edge falls into the "
     "bin BELOW it", C,
     b'    out = (np.searchsorted(e, r, side="right") - 1).astype(np.int64)',
     b'    out = (np.searchsorted(e, r, side="left") - 1).astype(np.int64)',
     [BINEDGE]),

    ("a rate outside every bin is CLAMPED rather than carried as its own "
     "category", C,
     b"    out[outside] = -1",
     b"    out[outside] = 0",
     [BINOUT]),

    ("the bin edges are re-chosen rather than reused from corrector_scan", C,
     b"def bin_index(rates: np.ndarray, edges=BIN_EDGES) -> np.ndarray:",
     b"def bin_index(rates: np.ndarray, edges=(0.0, 0.5, 1.1)) -> np.ndarray:",
     [BINS, BINEDGE]),

    # ============ REUSE AND THE DEPENDENCE IT INDUCES
    ("the treated features sharing a control are counted as CONTROLS reused, "
     "which understates the dependence whenever a control serves several", C,
     b"                n_sharing += 1",
     b"                n_sharing += 0",
     [REUSE]),

    # NOREUSE IS DELIBERATELY NOT NAMED HERE, and the matrix is what found
    # that. Its fixture has two singleton sets and no reuse at all, so
    # `use[s[:1]] = 1` and `use[s] += 1` produce the same array by
    # construction and the mutation is invisible to it -- a fixture that
    # cannot exhibit the branch it would be named for is not evidence about
    # that branch (criterion 2, gotcha 16). Classified TEST-DEFECT in the
    # MATRIX, not a coverage gap in the module: REUSE kills it.
    ("reuse is not counted at all -- rule 2's mandatory report becomes a "
     "constant", C,
     b"    for s in sets:\n        use[s] += 1",
     b"    for s in sets:\n        use[s[:1]] = 1",
     [REUSE]),

    # ============ THE OUTCOME SPACE
    ("an unknown branch label is silently accepted, so a cell outside the "
     "enumeration stops being detectable", C,
     b"        if b not in counts:",
     b"        if False:",
     [UNKNOWN]),

    ("the count-versus-population check is dropped, so an unassigned cell "
     "passes", C,
     b"    if len(branches) != int(n_treated):",
     b"    if False:",
     [UNASSIGNED]),

    # UNKNOWN IS DELIBERATELY NOT NAMED HERE. With `counts = {}` EVERY branch
    # label -- valid or not -- is absent from the dict, so the unknown-branch
    # fixture still gets its refusal and still sees the message it asserts on.
    # It passes for the wrong reason, which is what a fixture that cannot
    # exhibit its branch looks like. Classified TEST-DEFECT in the MATRIX;
    # SPACE, which checks the explicit zero, kills it.
    ("an empty branch is dropped from the counts rather than carried as an "
     "explicit zero", C,
     b"    counts = {b: 0 for b in TREATED_BRANCHES}",
     b"    counts = {}",
     [SPACE]),

    # ============ §5b IS NOT IMPLEMENTED
    ("a seed with no treated feature is DROPPED rather than carried as its "
     "own category -- the denominator silently conditions on the outcome", C,
     b'                    seed_branch_totals["NO_TREATED_FEATURES"] += 1',
     b'                    seed_branch_totals["NO_TREATED_FEATURES"] += 0',
     [EMPTYSEED, ALLEMPTY]),

    ("the empty seed's row is omitted entirely, so the flag cannot ride on a "
     "row a reader lands on", C,
     b'                        "seed_branch": "NO_TREATED_FEATURES",',
     b'                        "seed_branch": "CONTRIBUTES",',
     [EMPTYSEED, DENOM]),

    ("the across-seed denominator counts only contributing seeds and stops "
     "declaring itself censored", C,
     b'                    "denominator_seeds_total": n_seeds,',
     b'                    "denominator_seeds_total": len(contributing),',
     [EMPTYSEED, DENOM]),

    ("the r_f seed axis is joined by POSITION rather than by label -- 18 of "
     "20 rows would be mis-paired on the real artifacts (gotcha 29)", C,
     b"        rf_row = [rf_seed_numbers.index(_seed_number(s)) for s in ap_seeds]",
     b"        rf_row = list(range(len(ap_seeds)))",
     [POSJOIN]),

    ("a seed missing from the corrector scan is skipped instead of refused", C,
     b"    if missing:",
     b"    if False and missing:",
     [CSMISS]),

    ("the corrector exclusion does not bite -- the WITHOUT variant is a copy "
     "of the WITH variant", C,
     b'                if variant == "without_correctors" and flagged[lab]:',
     b'                if False:',
     [CSBITE]),
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

    print(f"{'mutation':<78} {'result':<10} killed")
    bad = 0
    for label, path, old, new, expect in MUTATIONS:
        survived = apply_and_run(path, old, new, expect)
        if survived is None:
            print(f"{label:<78} {'SKIP':<10} anchor did not match exactly once")
            bad += 1
            continue
        status = "KILLED" if not survived else "SURVIVED"
        print(f"{label:<78} {status:<10} "
              f"{len(expect) - len(survived)}/{len(expect)}")
        for t in survived:
            print(f"{'':<78} {'':<10} SURVIVOR: {t}")
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
