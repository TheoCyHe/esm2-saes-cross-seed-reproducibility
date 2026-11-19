"""Mutation matrix for amendment BQ's 2x2 classification. NOT collected by
pytest.

Run manually:  python tests/mutation_matrix_bq.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

The runner is reused from `mutation_matrix_av` -- a runner is itself a check,
and a second copy is a second thing that can silently stop detecting (gotcha 13
extension 5). It also carries BP's post-restore `__pycache__` purge
(corrections-log entry 122), which this matrix inherits rather than reimplements.

WHY THIS FILE EXISTS. Every way of getting the classification wrong returns a
COMPLETE, PLAUSIBLE TABLE. Reducing the two conjuncts separately inflates
ALIGNED without producing a single impossible number. Swapping EXCLUDED for
ORPHAN preserves the total and the population check still passes. Reading
NOT_EVALUABLE as ORPHAN is criterion 6's named failure and shows up nowhere in
the arithmetic. None of these looks wrong in a table.

THE FOUR SHAPES THE MUTATIONS COVER:

  * **THE CONJUNCT ORDER BREAKS.** `ANY(reject AND gain)` becomes
    `ANY(reject) AND ANY(gain)`, or a conjunct is dropped entirely. This is the
    defect the module's docstring is mostly about, and it strictly INFLATES
    ALIGNED, which is the direction that manufactures a result.
  * **A CELL IS MISLABELLED.** The two axes are swapped, or EXCLUDED and ORPHAN
    trade places. Counts still sum to the population, so criterion 5's
    arithmetic alone does not catch it -- only the labelled fixtures do.
  * **AN IGNORANCE CATEGORY IS FOLDED IN.** NOT_EVALUABLE is emitted as ORPHAN,
    or its precedence over the four cells is lost.
  * **THE VERIFIER OR THE FLOOR COUNT STOPS DISCRIMINATING.** The outcome-space
    check derives its masks from the codes it is checking (gotcha 32b), or the
    at-floor count uses one family-wide atom instead of each concept's own.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run  # noqa: E402
from tests.mutation_matrix_av import run_one  # noqa: E402

ROOT = os.getcwd()
C = os.path.join(ROOT, "src", "analysis", "classification_2x2.py")
T = "tests/test_classification_2x2.py"

# THE CONTROL: every feature clears every bar. It cannot fail to break the
# suite -- the conjunct-order fixture asserts a single False. If the runner
# does not detect this, no SURVIVED verdict below would mean anything, so none
# is printed.
CONTROL = (
    "CONTROL -- clears_bar returns all-True before any reduction",
    C,
    b"    return np.any(reject & (gain >= floor), axis=1)\n",
    b"    return np.ones(reject.shape[0], dtype=bool)\n",
    [f"{T}::test_the_two_conjuncts_are_anded_before_the_existential",
     f"{T}::test_clears_bar_true_only_when_one_concept_satisfies_both",
     f"{T}::test_a_seed_that_rejects_nothing_has_zero_aligned"],
)

MUTATIONS = [
    # ============ THE CONJUNCT ORDER BREAKS
    ("the conjuncts are reduced SEPARATELY (ANY(r) and ANY(g)) -- inflates "
     "ALIGNED", C,
     b"    return np.any(reject & (gain >= floor), axis=1)",
     b"    return np.any(reject, axis=1) & np.any(gain >= floor, axis=1)",
     [f"{T}::test_the_two_conjuncts_are_anded_before_the_existential",
      f"{T}::test_clears_bar_true_only_when_one_concept_satisfies_both"]),

    ("the gain floor is dropped entirely -- AV6 component 2 stops existing", C,
     b"    return np.any(reject & (gain >= floor), axis=1)",
     b"    return np.any(reject, axis=1)",
     [f"{T}::test_the_two_conjuncts_are_anded_before_the_existential",
      f"{T}::test_clears_bar_true_only_when_one_concept_satisfies_both",
      f"{T}::test_gain_floor_is_inclusive_at_exactly_the_floor"]),

    ("the FDR half is dropped -- the bar becomes the gain floor alone", C,
     b"    return np.any(reject & (gain >= floor), axis=1)",
     b"    return np.any(gain >= floor, axis=1)",
     [f"{T}::test_the_two_conjuncts_are_anded_before_the_existential",
      f"{T}::test_a_seed_that_rejects_nothing_has_zero_aligned"]),

    ("the gain floor becomes strict (> not >=), losing the boundary", C,
     b"    return np.any(reject & (gain >= floor), axis=1)",
     b"    return np.any(reject & (gain > floor), axis=1)",
     [f"{T}::test_gain_floor_is_inclusive_at_exactly_the_floor"]),

    ("ANY becomes ALL -- 'clears against any concept' becomes 'against every'",
     C,
     b"    return np.any(reject & (gain >= floor), axis=1)",
     b"    return np.all(reject & (gain >= floor), axis=1)",
     [f"{T}::test_clears_bar_true_only_when_one_concept_satisfies_both"]),

    # ============ THE GAIN ITSELF
    ("the gain drops its normaliser: AP - p instead of (AP - p)/(1 - p)", C,
     b"    return (ap - p[None, :]) / (1.0 - p[None, :])",
     b"    return ap - p[None, :]",
     [f"{T}::test_gain_matches_hand_arithmetic",
      f"{T}::test_gain_uses_each_concepts_own_prevalence_not_a_shared_one"]),

    ("the gain uses ONE prevalence for every concept (C4 violated)", C,
     b"    return (ap - p[None, :]) / (1.0 - p[None, :])",
     b"    return (ap - p[0]) / (1.0 - p[0])",
     [f"{T}::test_gain_uses_each_concepts_own_prevalence_not_a_shared_one",
      f"{T}::test_gain_matches_hand_arithmetic"]),

    # ============ A CELL IS MISLABELLED
    ("EXCLUDED and ORPHAN trade places -- the total is unchanged", C,
     b"    out[~e & p] = CELL_EXCLUDED\n    out[~e & ~p] = CELL_ORPHAN\n",
     b"    out[~e & p] = CELL_ORPHAN\n    out[~e & ~p] = CELL_EXCLUDED\n",
     [f"{T}::test_each_of_the_four_cells_is_reached_and_labelled_per_av5",
      f"{T}::test_outcome_space_counts_are_hand_checkable_and_sum_to_the_population"]),

    ("the two axes are swapped -- ALIGNED becomes the pooled side", C,
     b"    out[e & ~p] = CELL_ALIGNED_EXP\n",
     b"    out[e & ~p] = CELL_EXCLUDED\n",
     [f"{T}::test_each_of_the_four_cells_is_reached_and_labelled_per_av5"]),

    ("the precedence rule is inverted: clears-exp/fails-pooled loses ALIGNED",
     C,
     b"    CELL_ALIGNED_EXP: \"ALIGNED\",",
     b"    CELL_ALIGNED_EXP: \"EXCLUDED\",",
     [f"{T}::test_aligned_takes_precedence_so_both_top_row_cells_group_as_aligned"]),

    # ============ AN IGNORANCE CATEGORY IS FOLDED IN
    ("NOT_EVALUABLE is emitted as ORPHAN -- criterion 6's named failure", C,
     b"        out[:] = CELL_NOT_EVALUABLE\n        return out\n",
     b"        out[:] = CELL_ORPHAN\n        return out\n",
     [f"{T}::test_an_unevaluable_experimental_pool_gives_not_evaluable_not_orphan",
      f"{T}::test_an_unevaluable_pooled_pool_also_gives_not_evaluable",
      f"{T}::test_not_evaluable_wins_over_a_feature_that_would_have_been_aligned"]),

    ("NOT_EVALUABLE loses its precedence: only BOTH pools empty triggers it",
     C,
     b"    if not (exp_pool_evaluable and pooled_pool_evaluable):",
     b"    if not (exp_pool_evaluable or pooled_pool_evaluable):",
     [f"{T}::test_an_unevaluable_experimental_pool_gives_not_evaluable_not_orphan",
      f"{T}::test_an_unevaluable_pooled_pool_also_gives_not_evaluable",
      f"{T}::test_not_evaluable_wins_over_a_feature_that_would_have_been_aligned"]),

    # ============ THE VERIFIER STOPS DISCRIMINATING
    # FIRST-RUN SURVIVOR, CLASSIFIED **TEST-DEFECT IN THIS MATRIX**, AND THE
    # DIAGNOSIS IS RECORDED RATHER THAN THE MUTATION QUIETLY SWAPPED.
    # The first version of this entry replaced ONE of the five spec masks
    # (`CELL_ALIGNED_BOTH: e & p`) with `codes == CELL_ALIGNED_BOTH`. It
    # SURVIVED, and it survived for a correct reason: mislabelling a pair as
    # ALIGNED_BOTH necessarily removes it from the cell it belongs to, so the
    # OTHER four spec-derived masks still disagree with `codes` and the verifier
    # still raises. A partial derivation is not the defect gotcha 32b names --
    # a vacuous verifier requires EVERY mask to come from the thing under test,
    # which is what this entry now mutates. The survivor was therefore a defect
    # in the mutation, not a hole in the suite; it is kept in the record because
    # what it demonstrated is worth knowing: the verifier is robust to partial
    # corruption of its own specification.
    ("the verifier derives ALL its masks FROM THE CODES it is checking, "
     "making it vacuous (gotcha 32b)", C,
     b"        masks = {\n"
     b"            CELL_ALIGNED_BOTH: e & p,\n"
     b"            CELL_ALIGNED_EXP: e & np.logical_not(p),\n"
     b"            CELL_EXCLUDED: np.logical_not(e) & p,\n"
     b"            CELL_ORPHAN: np.logical_not(e) & np.logical_not(p),\n"
     b"            CELL_NOT_EVALUABLE: np.zeros(n, dtype=bool),\n"
     b"        }\n",
     b"        masks = {c: np.asarray(codes) == c for c in CELL_NAMES}\n",
     [f"{T}::test_outcome_space_verifier_FAILS_on_a_double_assigned_pair",
      f"{T}::test_outcome_space_verifier_FAILS_when_a_pair_is_given_no_outcome"]),

    ("the verifier stops comparing against the classifier at all", C,
     b"            raise OutcomeSpaceViolation(\n"
     b"                f\"the classifier disagrees with the specification on \"\n"
     b"                f\"{bad} pairs for {CELL_NAMES[c]}\")",
     b"            pass",
     [f"{T}::test_outcome_space_verifier_FAILS_on_a_double_assigned_pair",
      f"{T}::test_outcome_space_verifier_FAILS_when_a_pair_is_given_no_outcome",
      f"{T}::test_outcome_space_verifier_FAILS_when_not_evaluable_is_silently_dropped"]),

    # ============ THE AT-FLOOR COUNT
    ("the at-floor count uses ONE family-wide atom, not each concept's own", C,
     b"        atom = np.asarray(1.0 / (n_finite[c] + 1.0)).astype(p.dtype)",
     b"        atom = np.asarray(1.0 / (n_finite[0] + 1.0)).astype(p.dtype)",
     [f"{T}::test_at_floor_count_uses_each_concepts_OWN_atom_not_a_family_constant"]),

    ("the at-floor atom loses its +1 (the permutation correction)", C,
     b"        atom = np.asarray(1.0 / (n_finite[c] + 1.0)).astype(p.dtype)",
     b"        atom = np.asarray(1.0 / n_finite[c]).astype(p.dtype)",
     [f"{T}::test_at_floor_count_uses_each_concepts_OWN_atom_not_a_family_constant",
      f"{T}::test_at_floor_compares_in_the_stored_dtype_so_float32_is_exact"]),

    ("the at-floor count includes NOT_EVALUABLE concepts", C,
     b"        if ne[c] or n_finite[c] <= 0:",
     b"        if n_finite[c] <= 0:",
     [f"{T}::test_at_floor_ignores_concepts_marked_not_evaluable"]),

    # ============ THE SEED-ORDER RECOVERY (the raw null has NO labels)
    ("the recovery takes the FIRST match instead of refusing ambiguity", C,
     b"    if len(matches) == 1:\n        return matches[0]\n",
     b"    if matches:\n        return matches[0]\n",
     [f"{T}::test_recovery_FAILS_when_two_labelled_seeds_match"]),

    ("the recovery falls back on POSITION when nothing matches", C,
     b"    if not matches:\n        raise SeedOrderUnrecoverable(",
     b"    if not matches:\n        return 0\n    if matches is None:\n"
     b"        raise SeedOrderUnrecoverable(",
     [f"{T}::test_recovery_FAILS_when_no_labelled_seed_matches"]),

    ("the recovery stops discriminating and matches every seed", C,
     b"    matches = [i for i in range(g.shape[0]) if np.array_equal(g[i], m)]",
     b"    matches = [i for i in range(g.shape[0])][:1]",
     [f"{T}::test_recovery_DETECTS_A_PERMUTED_SEED_AXIS",
      f"{T}::test_recovery_FAILS_when_no_labelled_seed_matches"]),
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

    print(f"{'mutation':<66} {'result':<10} killed")
    bad = 0
    for label, path, old, new, expect in MUTATIONS:
        survived = apply_and_run(path, old, new, expect)
        if survived is None:
            print(f"{label:<66} {'SKIP':<10} anchor did not match exactly once")
            bad += 1
            continue
        status = "KILLED" if not survived else "SURVIVED"
        print(f"{label:<66} {status:<10} "
              f"{len(expect) - len(survived)}/{len(expect)}")
        for t in survived:
            print(f"{'':<66} {'':<10} SURVIVOR: {t}")
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
