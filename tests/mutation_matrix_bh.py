"""Mutation matrix for the masked experimental-stratum null (amendments BG/BH).

NOT collected by pytest.  Run manually:  python tests/mutation_matrix_bh.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

The runner is reused from `mutation_matrix_av` -- AU9b's control mutation,
AV22's `__pycache__` purge between invocations, the byte-compared restore, and
the decide-on-EXIT-CODE rule. Reused rather than re-implemented: a runner is a
check, and a second copy is a second thing that can silently stop detecting
(gotcha 13 extension 5).

BRIEF B's AC3 asks specifically for mutations of THE MASK APPLICATION and THE
COLLISION RULE, a control that is not a no-op, and a case that should still
pass so the suite is not merely trigger-happy.

THE TWO THIS FILE EXISTS FOR:

  * **`masked_concept_arrays` returning its inputs unfiltered.** This is not a
    hypothetical mutation -- it is the defect actually present in the AW5
    variant pilot, which built a `keep_row_v1` mask and then computed AP
    through the sparse touch path. The mask was constructed and never applied.
    The mutation makes every masked AP equal to the unmasked one, and the
    fixture that catches it asserts a hand-computed 5/6 against the 0.75 the
    defect returns. Note that this mutation is invisible to any test that only
    checks "masking changed the answer by SOMETHING", because the denominator
    still moves.

  * **`build_displacement_plan` calling a one-positive draw degenerate.** BH1
    says explicitly that a cell with exactly one surviving positive is NOT
    degenerate -- AP is defined there. The real study has 20 such cells under
    Null A and 22 under Null B, so a `<= 1` in place of `== 0` would silently
    discard 42 draws and coarsen those concepts' exceedance resolution, while
    every count in the artifact stayed self-consistent.

WHAT WOULD NOT BE CAUGHT, stated rather than left for a reader to assume: these
mutations exercise the mask, the collision rule and the BH bookkeeping. They do
NOT exercise `main()`'s wiring of them -- the loop nesting, the per-seed raw
buffer, the h5 writes. That is covered by the end-to-end pooled regression
(AC1) and by AC4's exact reproduction of the degenerate cell, both of which run
the entry point against real artifacts.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run, run_one  # noqa: E402

ROOT = os.getcwd()
P = os.path.join(ROOT, "src", "analysis", "null_pilot.py")
T = "tests/test_null_pilot_masked.py"
TN = "tests/test_null_pilot.py"

# THE CONTROL: `classify_bh3` returns a constant before any branch runs.
#
# It cannot fail to break the suite -- the grid test is parametrised over
# FOUR DISTINCT expected verdicts across twelve rows, and a constant collapses
# them to one. Deliberately NOT a docstring or a message string: BA17 item 2's
# first control replaced a `why` string while leaving the verdict alone, so
# every outcome stayed reachable and the runner correctly reported CONTROL NOT
# DETECTED. This anchor returns before any branch is evaluated.
CONTROL = (
    "CONTROL -- classify_bh3 returns a constant before any branch", P,
    b'    arms = [n for n in (n_degenerate_a, n_degenerate_b) if n is not None]',
    b'    return BH3_NONE\n'
    b'    arms = [n for n in (n_degenerate_a, n_degenerate_b) if n is not None]',
    [f"{T}::test_bh3_grid_is_a_TOTAL_function_of_the_degenerate_counts"],
)

MUTATIONS = [
    # ==================================================== THE MASK APPLICATION
    # THE ONE THIS FILE EXISTS FOR. The AW5 defect, made a mutation: the mask
    # is built and never applied, so the ranks are never renumbered.
    ("masked_concept_arrays: the mask is built and NEVER APPLIED", P,
     b"    keep_t = torch.from_numpy(keep_row).to(feat.device)\n"
     b"    return filter_nonzeros(feat, pos, keep_t, n_features)",
     b"    return feat, pos, seg",
     # `test_the_tied_zero_tail_uses_the_EFFECTIVE_row_count` is DELIBERATELY
     # NOT listed. Measured: it SURVIVES this mutation, and correctly so --
     # its only firing positive is row 0 at rank 1, ABOVE the masked row, so
     # no rank it depends on moves and the masked and unmasked answers
     # coincide at 0.7. A fixture that cannot exhibit the disputed condition
     # cannot test it (gotcha 16), and listing it here would have made this
     # mutation read as under-covered when the coverage is elsewhere.
     [f"{T}::test_masking_a_FIRING_row_renumbers_the_ranks_below_it",
      f"{T}::test_masked_concept_arrays_actually_DROPS_the_masked_rows_nonzeros"]),

    # The mask INVERTED -- keeping exactly the rows that should be dropped.
    ("masked_concept_arrays: the keep-mask is inverted", P,
     b"    keep_t = torch.from_numpy(keep_row).to(feat.device)",
     b"    keep_t = torch.from_numpy(~keep_row).to(feat.device)",
     [f"{T}::test_masked_concept_arrays_actually_DROPS_the_masked_rows_nonzeros",
      f"{T}::test_masking_a_FIRING_row_renumbers_the_ranks_below_it"]),

    # The unmasked branch must be the IDENTITY. If it filters against an
    # all-True mask instead, the pooled arm stops being the same computation
    # and AC1's bit-identity criterion is no longer a criterion.
    ("masked_concept_arrays: the no-mask branch rebuilds instead of passing through",
     P,
     b"    if keep_row is None:\n        return feat, pos, seg",
     b"    if keep_row is None:\n"
     b"        import numpy as _np\n"
     b"        return filter_nonzeros(feat, pos, torch.from_numpy(\n"
     b"            _np.ones(int(pos.max()) + 1, dtype=bool)).to(feat.device),\n"
     b"            n_features)",
     [f"{T}::test_masked_concept_arrays_with_no_mask_returns_the_inputs_UNTOUCHED"]),

    # ==================================================== THE GAIN DENOMINATOR
    # Normalising by the FULL row count rather than the masked one. Both give
    # plausible numbers in the same range (0.7222 vs 0.75), which is why this
    # needs an assertion rather than a reading.
    ("masked_draw_gain: prevalence uses the unmasked row count", P,
     b"    p_draw = n_surviving / n_eff",
     b"    p_draw = n_surviving / (n_eff + 1)",
     [f"{T}::test_the_gain_is_normalised_by_the_EFFECTIVE_row_count_not_the_full_one"]),

    ("masked_draw_gain: the observed count replaces the surviving count", P,
     b"    ap = average_precision_segmented(f_c, s_c, labels, n_features,\n"
     b"                                     n_surviving, n_eff)",
     b"    ap = average_precision_segmented(f_c, s_c, labels, n_features,\n"
     b"                                     n_surviving + 1, n_eff)",
     [f"{T}::test_the_gain_is_normalised_by_the_EFFECTIVE_row_count_not_the_full_one"]),

    # ======================================================= THE COLLISION RULE
    ("apply_collision_rule: collisions are kept rather than dropped", P,
     b"    return displaced[keep_row[displaced]]",
     b"    return displaced",
     [f"{T}::test_collision_rule_drops_displaced_positives_that_land_on_a_masked_row",
      f"{T}::test_collision_rule_can_empty_a_draw_entirely",
      f"{T}::test_the_plan_holds_the_MASK_FIXED_and_displaces_only_the_POSITIVES",
      f"{T}::test_a_draw_with_ONE_surviving_positive_is_NOT_degenerate"]),

    ("apply_collision_rule: the rule is inverted -- only collisions survive", P,
     b"    return displaced[keep_row[displaced]]",
     b"    return displaced[~keep_row[displaced]]",
     [f"{T}::test_collision_rule_drops_displaced_positives_that_land_on_a_masked_row",
      f"{T}::test_the_plan_holds_the_MASK_FIXED_and_displaces_only_the_POSITIVES"]),

    # An unmasked concept must pass through untouched, or Null A's bijection
    # property stops holding and the pooled arm changes.
    ("apply_collision_rule: the no-mask branch drops everything", P,
     b"    if keep_row is None:\n        return displaced",
     b"    if keep_row is None:\n        return displaced[:0]",
     [f"{T}::test_collision_rule_with_no_mask_is_the_IDENTITY",
      f"{T}::test_the_plan_is_the_IDENTITY_on_survival_when_no_mask_is_active"]),

    # ========================================================= BH1: degeneracy
    # THE SECOND ONE THIS FILE EXISTS FOR. BH1 says a cell with exactly ONE
    # surviving positive is NOT degenerate. 42 real cells sit in that state.
    ("build_displacement_plan: a one-positive draw is called degenerate", P,
     b'                     "degenerate": counts == 0}',
     b'                     "degenerate": counts <= 1}',
     [f"{T}::test_a_draw_with_ONE_surviving_positive_is_NOT_degenerate"]),

    ("build_displacement_plan: nothing is ever degenerate", P,
     b'                     "degenerate": counts == 0}',
     b'                     "degenerate": counts < 0}',
     [f"{T}::test_the_plan_holds_the_MASK_FIXED_and_displaces_only_the_POSITIVES",
      f"{T}::test_a_draw_with_ONE_surviving_positive_is_NOT_degenerate"]),

    # Null B's many-to-one rescaling. Without the dedup the realised positive
    # count is too large and the gain normalisation is inconsistent with the
    # rows actually queried.
    ("build_displacement_plan: Null B does not dedup its many-to-one landings", P,
     b"                    d = np.unique(cross_protein_shift_rows(\n"
     b"                        src, row2protein, starts, lengths, sched[r]))",
     b"                    d = cross_protein_shift_rows(\n"
     b"                        src, row2protein, starts, lengths, sched[r])",
     [f"{T}::test_null_B_dedups_BEFORE_the_collision_rule_so_counts_are_realised"]),

    # The collision rule reaching only one arm is the asymmetry BG2 forbids.
    ("build_displacement_plan: the collision rule is skipped entirely", P,
     b"                d = apply_collision_rule(d, kr)",
     b"                d = d",
     [f"{T}::test_the_plan_holds_the_MASK_FIXED_and_displaces_only_the_POSITIVES",
      f"{T}::test_a_draw_with_ONE_surviving_positive_is_NOT_degenerate"]),

    # ===================================================== BH4: the resolution
    ("exceedance_resolution: the +1 is dropped from the denominator", P,
     b"    return 1.0 / (n_surviving_draws * n_features + 1)",
     b"    return 1.0 / (n_surviving_draws * n_features)",
     [f"{T}::test_exceedance_resolution_matches_the_two_values_BH4_names"]),

    # NOT_EVALUABLE handed a finite number is the exact confusion BH3 forbids:
    # a consumer could fold it into a denominator of concepts tested.
    ("exceedance_resolution: no surviving draw returns a NUMBER, not NaN", P,
     b'        return float("nan")            # NOT_EVALUABLE: no reference distribution',
     b"        return 1.0 / (n_features + 1)",
     [f"{T}::test_a_concept_with_no_surviving_draw_has_NO_resolution_not_a_number"]),

    # ======================================================== BH3: the grid
    # An ABSENT arm read as 0 is I3's "absent is not matching", on the rule
    # that decides whether a concept has a verdict at all.
    ("classify_bh3: an absent arm is treated as zero degenerate draws", P,
     b"    arms = [n for n in (n_degenerate_a, n_degenerate_b) if n is not None]",
     b"    arms = [0 if n is None else n for n in (n_degenerate_a, n_degenerate_b)]",
     [f"{T}::test_bh3_grid_is_a_TOTAL_function_of_the_degenerate_counts",
      f"{T}::test_bh3_distinguishes_an_ABSENT_arm_from_an_arm_with_zero_degenerate_draws"]),

    ("classify_bh3: one fully-degenerate arm reads as the STRATUM verdict", P,
     b"        return BH3_BOTH_ARMS if len(arms) > 1 else BH3_ONE_ARM",
     b"        return BH3_BOTH_ARMS",
     [f"{T}::test_bh3_grid_is_a_TOTAL_function_of_the_degenerate_counts",
      f"{T}::test_bh3_distinguishes_an_ABSENT_arm_from_an_arm_with_zero_degenerate_draws"]),

    ("classify_bh3: SOME_BUT_NOT_ALL is unreachable", P,
     b"    if any(n > 0 for n in arms):\n        return BH3_SOME",
     b"    if False:\n        return BH3_SOME",
     [f"{T}::test_bh3_grid_is_a_TOTAL_function_of_the_degenerate_counts"]),

    ("classify_bh3: the no-arm cell falls through to NONE_DEGENERATE", P,
     b"    if not arms:\n        return BH3_NO_ARM",
     b"    if False:\n        return BH3_NO_ARM",
     [f"{T}::test_bh3_grid_is_a_TOTAL_function_of_the_degenerate_counts"]),
]

# ------------------------------------------------------------------------- #
# THE CASE THAT SHOULD STILL PASS (AC3's third requirement).
#
# A change that alters the source WITHOUT altering behaviour must SURVIVE. If
# it were killed, the suite would be reacting to edits rather than to defects,
# and every KILLED verdict above would be worth less.
# ------------------------------------------------------------------------- #
BENIGN = [
    ("BENIGN -- a comment is reworded (behaviour identical; must SURVIVE)", P,
     b"    if keep_row is None:\n        return displaced\n",
     b"    if keep_row is None:\n        # no mask: every displaced positive survives\n"
     b"        return displaced\n",
     [f"{T}::test_collision_rule_with_no_mask_is_the_IDENTITY",
      f"{T}::test_collision_rule_drops_displaced_positives_that_land_on_a_masked_row",
      f"{T}::test_the_plan_is_the_IDENTITY_on_survival_when_no_mask_is_active"]),
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

    print(f"{'mutation':<70} {'result':<10} killed")
    bad = 0
    for label, path, old, new, expect in MUTATIONS:
        survived = apply_and_run(path, old, new, expect)
        if survived is None:
            print(f"{label:<70} {'SKIP':<10} anchor did not match exactly once")
            bad += 1
            continue
        status = "KILLED" if not survived else "SURVIVED"
        print(f"{label:<70} {status:<10} "
              f"{len(expect) - len(survived)}/{len(expect)}")
        for t in survived:
            print(f"{'':<70} {'':<10} SURVIVOR: {t}")
        bad += bool(survived)

    print()
    for label, path, old, new, expect in BENIGN:
        survived = apply_and_run(path, old, new, expect)
        if survived is None:
            print(f"{label:<70} {'SKIP':<10} anchor did not match exactly once")
            bad += 1
            continue
        ok = len(survived) == len(expect)
        print(f"{label:<70} {'SURVIVED' if ok else 'KILLED':<10} "
              f"{len(expect) - len(survived)}/{len(expect)}")
        if not ok:
            print(f"{'':<70} {'':<10} A BEHAVIOUR-PRESERVING EDIT WAS KILLED -- "
                  "the suite is reacting to edits, not to defects.")
            bad += 1

    print()
    print("ALL MUTATIONS KILLED THEIR NAMED FIXTURES; BENIGN EDIT SURVIVED"
          if not bad else f"{bad} MUTATION(S) SURVIVED, WERE SKIPPED, OR "
                          "KILLED A BENIGN EDIT")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
