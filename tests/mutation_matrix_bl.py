"""Mutation matrix for amendment BL's range-coherence rule. NOT collected by
pytest.

Run manually:  python tests/mutation_matrix_bl.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

The runner is reused from `mutation_matrix_av` -- AU9b's control-first rule,
AV22's `__pycache__` purge between invocations, the byte-compared restore, and
the decide-on-EXIT-CODE rule. Reused rather than re-implemented: a runner is a
check, and a second copy is a second thing that can silently stop detecting
(gotcha 13 extension 5).

WHAT THIS FILE EXISTS FOR. `range_coherence_diff` is a rule whose entire value
is that it FIRES on a case every existing check passes. Every one of BK's 13
qualifying thetas clears `ratio_rule_diff`; the range across them is still
incoherent. A rule like that is exactly the shape that can be quietly inert --
it returns COHERENT, nothing downstream disagrees, and the record gains a
citation to a check that never ran.

THE FIVE SHAPES THE MUTATIONS COVER:

  * **THE SIGN CHECK GOES BLIND TO ITS OWN FOUNDING INSTANCE.** `magnitude_of`
    is the whole reason the rule can see a reversal in a component that is an
    SD. Ignore it and the founding case returns COHERENT while looking like it
    worked -- gotcha 21's reduction hazard, in the check written to catch it.
  * **IGNORANCE BECOMES A PASS.** An undeclared non-negative component, or a
    grid point with no value, must never reach COHERENT. Both directions of
    that boundary are mutated, because a cell that can only fail is as useless
    as one that can never fail (gotcha 25).
  * **PRECEDENCE COLLAPSES.** Folding BOTH_COMPONENTS_CHANGE_SIGN into the
    single-component cell loses the one diagnosis a reader cannot recover from
    the ratio itself, and ranking NOT_EVALUABLE above a measured incoherence
    reintroduces the masking `OUTCOMES` ordered against.
  * **THE QUOTED/FULL-GRID ANSWER FLIPS.** DESIGN QUESTION 2 is a decision, so
    it is asserted: evaluating the full grid instead of the quoted span turns
    every quote near a reversal into a failure, and dropping `grid_context`
    hides the selection a reader is owed.
  * **A STRUCTURAL CELL BECOMES VACUOUS COHERENCE.** A span naming zero or one
    grid point must not report "no sign change was observed".

WHAT IS NOT COVERED, stated rather than left to assumption: that the SCREEN's
per-instance component values were read correctly out of the record. No unit
test can establish that -- it is a reading, reported as a diff in the amendment
with its source section named, which is the guarantee available here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run  # noqa: E402
from tests.mutation_matrix_av import run_one  # noqa: E402

ROOT = os.getcwd()
R = os.path.join(ROOT, "src", "analysis", "ratio_rule.py")
T = "tests/test_ratio_rule.py"

# THE CONTROL: `range_coherence_diff` returns COHERENT before it looks at
# anything. It cannot fail to break the suite -- every non-COHERENT fixture
# asserts a specific cell -- so if the runner does not detect this, no
# SURVIVED verdict it prints afterwards means anything.
CONTROL = (
    "CONTROL -- range_coherence_diff returns COHERENT before any branch",
    R,
    b"    fails = []\n    pts = list(points)",
    b"    fails = []\n    pts = list(points)\n"
    b"    if True:\n        return {'name': name, 'outcome': 'COHERENT',\n"
    b"                'failures': [], 'failure_codes': [],\n"
    b"                'sign_change_intervals': {}, 'per_point': [],\n"
    b"                'grid_context': {}, 'coherent_sub_ranges': [],\n"
    b"                'defensible_form': '', 'n_in_quoted_range': 0,\n"
    b"                'grid_ratio_min': None, 'grid_ratio_max': None,\n"
    b"                'ordering_change_intervals': [],\n"
    b"                'coherent_over_evaluable_points': True,\n"
    b"                'at_in_quoted_range': []}",
    [f"{T}::test_the_founding_instance_is_INCOHERENT_ONE_COMPONENT_CHANGES_SIGN",
     f"{T}::test_both_components_reversing_is_its_OWN_cell_and_outranks_one",
     f"{T}::test_the_term_ordering_flipping_is_INCOHERENT_AT13s_ruling"],
)

MUTATIONS = [
    # =============================== THE SIGN CHECK GOES BLIND TO SDs
    # `magnitude_of` ignored: the sign is read off the component VALUE, which
    # for two SDs is non-negative everywhere. The founding instance returns
    # COHERENT. THE FIXTURE MUST USE A DISPERSION -- one built from signed
    # values would not notice.
    ("_sign_basis ignores magnitude_of and reads the value", R,
     b'    m = comp.get("magnitude_of")\n'
     b"    if m is SIGNED:\n"
     b'        return comp.get("value"), True\n'
     b"    if m is not None:\n"
     b"        return m, True\n"
     b'    return comp.get("value"), False',
     b'    return comp.get("value"), True',
     [f"{T}::test_the_founding_instance_is_INCOHERENT_ONE_COMPONENT_CHANGES_SIGN",
      f"{T}::test_the_founding_instance_WITHOUT_a_declared_sign_basis_is_NOT_EVALUABLE"]),

    # =============================== IGNORANCE BECOMES A PASS
    ("an undeclared non-negative component stops being reported", R,
     b"        if finite_signs and all(s >= 0 for s in finite_signs) \\\n"
     b"                and not all(declared):",
     b"        if False:",
     [f"{T}::test_the_founding_instance_WITHOUT_a_declared_sign_basis_is_NOT_EVALUABLE"]),

    ("SIGN_BASIS_UNDECLARED is treated as coherent rather than unevaluable", R,
     b'    elif codes & {"GRID_POINT_VALUE_ABSENT_OR_NON_FINITE",\n'
     b'                  "SIGN_BASIS_UNDECLARED_FOR_A_NON_NEGATIVE_COMPONENT"}:',
     b'    elif codes & {"GRID_POINT_VALUE_ABSENT_OR_NON_FINITE"}:',
     [f"{T}::test_the_founding_instance_WITHOUT_a_declared_sign_basis_is_NOT_EVALUABLE"]),

    ("a grid point with no finite value stops being reported", R,
     b"        if any(not _finite(b) for b in bases):",
     b"        if False:",
     [f"{T}::test_a_non_finite_value_at_a_grid_point_is_NOT_EVALUABLE"]),

    # =============================== PRECEDENCE COLLAPSES
    ("BOTH_COMPONENTS_CHANGE_SIGN folds into the single-component cell", R,
     b'    elif {"NUMERATOR_CHANGES_SIGN", "DENOMINATOR_CHANGES_SIGN"} <= codes:\n'
     b'        outcome = "INCOHERENT_BOTH_COMPONENTS_CHANGE_SIGN"\n',
     b"",
     [f"{T}::test_both_components_reversing_is_its_OWN_cell_and_outranks_one"]),

    ("NOT_EVALUABLE is ranked ABOVE a measured incoherence", R,
     b'    if "FEWER_THAN_TWO_GRID_POINTS" in codes:',
     b'    if codes & {"GRID_POINT_VALUE_ABSENT_OR_NON_FINITE"}:\n'
     b'        outcome = "NOT_EVALUABLE_COHERENCE_IS_UNDETERMINED_ON_THIS_RANGE"\n'
     b'    elif "FEWER_THAN_TWO_GRID_POINTS" in codes:',
     [f"{T}::test_a_measured_incoherence_OUTRANKS_a_gap_elsewhere_in_the_span"]),

    # =============================== THE QUOTED/FULL-GRID ANSWER FLIPS
    ("coherence is evaluated on the FULL GRID, not the quoted span", R,
     b"        keep = [i for i, p in enumerate(pts) if lo <= p[\"at\"] <= hi]",
     b"        keep = list(range(len(pts)))",
     [f"{T}::test_a_reversal_OUTSIDE_the_quoted_span_is_REPORTED_and_does_not_score",
      f"{T}::test_one_grid_point_in_the_quoted_span_is_NOT_A_RANGE"]),

    ("grid_context stops naming an incoherence among excluded points", R,
     b'        "outside_incoherence": (probe["outcome"]\n'
     b'                                if probe["outcome"].startswith("INCOHERENT")\n'
     b"                                else None),",
     b'        "outside_incoherence": None,',
     [f"{T}::test_grid_context_names_an_incoherence_among_the_excluded_points"]),

    # =============================== A STRUCTURAL CELL BECOMES VACUOUS
    ("a span of fewer than two points reports vacuous coherence", R,
     b"    if len(in_range) < 2:",
     b"    if False:",
     [f"{T}::test_one_grid_point_in_the_quoted_span_is_NOT_A_RANGE",
      f"{T}::test_zero_grid_points_in_the_quoted_span_is_NOT_A_RANGE_not_COHERENT"]),

    # =============================== AT13's RULING STOPS BEING ENFORCED
    ("the term-ordering check is removed", R,
     b"    if ordering_changes:",
     b"    if False:",
     [f"{T}::test_the_term_ordering_flipping_is_INCOHERENT_AT13s_ruling"]),

    # =============================== THE FORMATTER LOSES ITS TOTAL BRANCH
    ("defensible_range_form's total branch asserts a verdict instead", R,
     b'    return (f"{name}: {outcome} has no formatter branch. This is a defect in "\n'
     b'            "`defensible_range_form`, not a verdict about the range.")',
     b'    return f"{name} is coherent."',
     [f"{T}::test_the_formatter_has_a_TOTAL_branch_and_it_names_itself_as_the_defect"]),
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
        print(f"CONTROL ABORT: anchor for {label!r} did not match exactly once. "
              "The runner cannot demonstrate that it detects anything.")
        return 2
    if survived:
        print(f"CONTROL ABORT: {label} was NOT detected (survivors: {survived}). "
              "Every SURVIVED verdict this runner could print would be "
              "meaningless, so none is printed.")
        return 2
    print(f"control: DETECTED ({label})\n")

    print(f"{'mutation':<66} {'result':<10} killed")
    bad = 0
    for label, path, old, new, expect in MUTATIONS:
        survived = apply_and_run(path, old, new, expect)
        if survived is None:
            print(f"{label:<66} {'SKIP':<10} anchor did not match exactly once")
            bad += 1
            continue
        status = "KILLED" if not survived else "SURVIVED"
        print(f"{label:<66} {status:<10} {len(expect) - len(survived)}/{len(expect)}")
        for t in survived:
            print(f"{'':<66} {'':<10} SURVIVOR: {t}")
        if survived:
            bad += 1
    print(f"\n{len(MUTATIONS) - bad}/{len(MUTATIONS)} mutations killed, "
          f"control detected")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
