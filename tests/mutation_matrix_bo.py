"""Mutation matrix for amendment BO's components 1 and 2. NOT collected by
pytest.

Run manually:  python tests/mutation_matrix_bo.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

The runner is reused from `mutation_matrix_av` -- AU9b's control-first rule,
AV22's `__pycache__` purge between invocations, the byte-compared restore, and
the decide-on-EXIT-CODE rule. Reused rather than re-implemented, because a
runner is a check and a second copy is a second thing that can silently stop
detecting (gotcha 13 extension 5).

WHY THIS FILE EXISTS. Components 1 and 2 are two functions of about thirty
lines whose failure modes all produce PLAUSIBLE NUMBERS. A p-value computed
with `>` instead of `>=`, a BH scan that stops at the first failure, a NaN
imputed as zero -- every one of them returns a well-formed array in the right
range, and none of them would be visible in a summary. Criterion 3 is what
distinguishes a suite that covers the code from one that merely runs it.

THE FOUR SHAPES THE MUTATIONS COVER:

  * **THE P-VALUE CONVENTION DRIFTS.** `>=` becomes `>`, or the permutation
    `+1`s are dropped so a p-value of exactly zero becomes reportable. Both
    are one-character changes and both silently move every threshold.
  * **BH STOPS BEING A STEP-UP.** The scan halts at the first failure instead
    of taking the largest crossing. This yields a STRICT SUBSET of the correct
    rejections, so it looks conservative rather than broken.
  * **BH2's DROP BECOMES AN IMPUTATION.** Degenerate draws counted, or kept at
    a value, instead of dropped -- which both inflates the denominator and
    changes the resolution BH4 makes per-(concept, arm).
  * **AN IGNORANCE CATEGORY LEAKS INTO THE FAMILY.** A NOT_EVALUABLE concept's
    tests entering `m`, or its exception being swallowed into a value. This is
    the folding-into-a-denominator failure criterion 6 bans, arriving through
    the code rather than through a table.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run  # noqa: E402
from tests.mutation_matrix_av import run_one  # noqa: E402

ROOT = os.getcwd()
C = os.path.join(ROOT, "src", "analysis", "classification_bh.py")
T = "tests/test_classification_bh.py"

# THE CONTROL: `bh_reject` rejects nothing, ever. It cannot fail to break the
# suite -- several fixtures assert a specific non-zero rejection count -- so
# if the runner does not detect this, no SURVIVED verdict below means anything.
CONTROL = (
    "CONTROL -- bh_reject rejects nothing before any branch",
    C,
    b'    p = np.asarray(pvals, dtype=np.float64).reshape(-1)\n'
    b"    m = int(p.size)\n",
    b'    p = np.asarray(pvals, dtype=np.float64).reshape(-1)\n'
    b"    m = int(p.size)\n"
    b"    if True:\n"
    b"        return {'n_tests': m, 'n_rejected': 0, 'threshold': None,\n"
    b"                'reject': np.zeros(m, dtype=bool), 'q': float(q)}\n",
    [f"{T}::test_bh_is_a_STEP_UP_not_a_scan_until_first_failure",
     f"{T}::test_bh_rejects_everything_when_all_p_are_tiny",
     f"{T}::test_bh_matches_the_independent_implementation"],
)

MUTATIONS = [
    # ===================== THE P-VALUE CONVENTION DRIFTS
    ("exceedance uses > instead of >= (side='right')", C,
     b'    n_ge = n - np.searchsorted(pool, np.asarray(g_obs), side="left")',
     b'    n_ge = n - np.searchsorted(pool, np.asarray(g_obs), side="right")',
     [f"{T}::test_exceedance_is_GE_not_GT",
      f"{T}::test_exceedance_hand_computed_fraction",
      f"{T}::test_exceedance_matches_the_independent_implementation"]),

    ("the permutation +1s are dropped, so p = 0 becomes reportable", C,
     b"    p = (n_ge + 1.0) / (n + 1.0)",
     b"    p = n_ge / n",
     [f"{T}::test_exceedance_ceiling_is_never_zero",
      f"{T}::test_exceedance_hand_computed_fraction",
      f"{T}::test_exceedance_matches_the_independent_implementation"]),

    ("the reported resolution ignores the +1", C,
     b"    return p, n, 1.0 / (n + 1.0)",
     b"    return p, n, 1.0 / n",
     [f"{T}::test_exceedance_hand_computed_fraction",
      f"{T}::test_degenerate_draws_are_DROPPED_and_shrink_the_denominator",
      f"{T}::test_THE_DEGENERATE_CELL_flows_through_as_a_COARSENED_DENOMINATOR"]),

    # ===================== BH STOPS BEING A STEP-UP
    ("BH takes the FIRST crossing instead of the LARGEST", C,
     b"    k = int(passing[-1])                       # LARGEST crossing index",
     b"    k = int(passing[0])",
     [f"{T}::test_bh_is_a_STEP_UP_not_a_scan_until_first_failure",
      f"{T}::test_bh_matches_the_independent_implementation"]),

    ("BH's critical values use q directly instead of (i/m)q", C,
     b"    crit = (np.arange(1, m + 1) / m) * q",
     b"    crit = np.full(m, q)",
     [f"{T}::test_bh_critical_values_are_RANK_DEPENDENT_not_a_flat_q",
      f"{T}::test_bh_matches_the_independent_implementation"]),

    ("BH's reject mask is written in SORTED order, not input order", C,
     b"    reject[order[: k + 1]] = True",
     b"    reject[: k + 1] = True",
     [f"{T}::test_bh_reject_mask_is_in_INPUT_order_not_sorted_order",
      f"{T}::test_slab_agrees_with_the_independent_implementation_end_to_end"]),

    # ===================== BH2's DROP BECOMES AN IMPUTATION
    ("degenerate draws are IMPUTED as 0.0 rather than dropped", C,
     b"    pool = pool[finite]\n    n = int(pool.size)",
     b"    pool = np.where(finite, pool, 0.0)\n    n = int(pool.size)",
     [f"{T}::test_degenerate_draws_are_DROPPED_and_shrink_the_denominator",
      f"{T}::test_THE_DEGENERATE_CELL_flows_through_as_a_COARSENED_DENOMINATOR",
      f"{T}::test_all_degenerate_raises_NotEvaluable_rather_than_returning_a_value"]),

    ("the denominator stays at the NOMINAL draw count, ignoring drops", C,
     b"    p = (n_ge + 1.0) / (n + 1.0)\n    return p, n, 1.0 / (n + 1.0)",
     b"    nom = int(np.asarray(null_pool).size)\n"
     b"    p = (n_ge + 1.0) / (nom + 1.0)\n    return p, n, 1.0 / (nom + 1.0)",
     [f"{T}::test_degenerate_draws_are_DROPPED_and_shrink_the_denominator",
      f"{T}::test_THE_DEGENERATE_CELL_flows_through_as_a_COARSENED_DENOMINATOR"]),

    # ===================== AN IGNORANCE CATEGORY LEAKS INTO THE FAMILY
    ("a fully degenerate pool returns NaN instead of raising NotEvaluable", C,
     b"    if n == 0:\n"
     b"        raise NotEvaluable(",
     b"    if n == 0:\n"
     b"        return np.full(np.asarray(g_obs).shape, np.nan), 0, np.nan\n"
     b"    if False:\n"
     b"        raise NotEvaluable(",
     [f"{T}::test_all_degenerate_raises_NotEvaluable_rather_than_returning_a_value",
      f"{T}::test_a_wholly_degenerate_concept_IS_NOT_EVALUABLE_and_leaves_the_family"]),

    ("NOT_EVALUABLE concepts are left IN the family", C,
     b"    fam = np.zeros((n_f, n_c), dtype=bool)\n"
     b"    fam[:, ~not_evaluable] = True",
     b"    fam = np.ones((n_f, n_c), dtype=bool)",
     [f"{T}::test_slab_family_excludes_NOT_EVALUABLE_concepts_from_m",
      f"{T}::test_a_wholly_degenerate_concept_IS_NOT_EVALUABLE_and_leaves_the_family"]),

    ("the transposed-observed refusal is removed", C,
     b"    if g_obs.shape != (n_f, n_c):",
     b"    if False:",
     [f"{T}::test_slab_refuses_a_TRANSPOSED_observed_array"]),

    ("the NOT_EVALUABLE count stops being carried", C,
     b'        "n_concepts_not_evaluable": int(not_evaluable.sum()),',
     b'        "n_concepts_not_evaluable": 0,',
     [f"{T}::test_slab_family_excludes_NOT_EVALUABLE_concepts_from_m",
      f"{T}::test_a_wholly_degenerate_concept_IS_NOT_EVALUABLE_and_leaves_the_family"]),
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
