"""Mutation matrix for `ratio_rule`, amendments BD and BE. NOT collected by
pytest.

Covers BD section 2's implementation of CLAUDE.md gotcha 33 AND BE section 3's
SYMMETRIC extension of it -- the numerator's interval, the widened
`UNVERIFIABLE` cell, and the `EXACT` sentinel that stops the extension
flagging the record's own compliant cases.

**This file SUPERSEDES `tests/mutation_matrix_bd.py`, which is deleted rather
than left beside it.** Two matrices over one module is a second thing that can
silently stop detecting (gotcha 13 extension 5), and the one left un-run is
the one that rots.

Run manually:  python tests/mutation_matrix_be.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

The runner is reused from `mutation_matrix_av` (AU9b's control-first rule,
AV22's `__pycache__` purge, the byte-compared restore, the decide-on-exit-code
rule). Reused rather than re-implemented: a runner is a check, and a second
copy is a second thing that can silently stop detecting.

THE THREE THIS FILE EXISTS FOR:

  * **THE ABSENT-INTERVAL BRANCH.** It is the quiet one and it is the one that
    fires most often on a prose record: a multiplier whose denominator has no
    measured interval at all. Deleting it turns "excludes zero is
    unestablished" into "excludes zero is satisfied" -- gotcha 15's
    absent-reads-as-agreement, inside the check written to enforce it -- and
    every bare multiplier in the record would come back DEFENSIBLE.
  * **THE PRECEDENCE ORDER.** Three conditions can fail at once and the reader
    must be handed the most severe. If a setting mismatch could pre-empt a
    scale mismatch, the check would tell an author to name two thresholds on a
    ratio that has no meaning at any threshold.
  * **THE SCALE HALF NOT COLLAPSING INTO THE DIFFERENCE REMEDY.** BC8a says
    the shared-scale half is "the one that hides", precisely because such a
    ratio is finite and looks meaningful. If a scale mismatch fell through to
    REPORT_THE_DIFFERENCE_INSTEAD, the check would offer a remedy that is
    itself undefined -- subtracting a raw coefficient from a standardised one.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run  # noqa: E402

ROOT = os.getcwd()
R = os.path.join(ROOT, "src", "analysis", "ratio_rule.py")
T = "tests/test_ratio_rule.py"

# THE CONTROL: `ratio_rule_diff` returns DEFENSIBLE before any condition is
# examined. This cannot fail to break the suite -- seven fixtures assert three
# other outcomes on inputs that reach them by different routes.
#
# It is deliberately NOT a change to a `remedy` string or to `defensible_form`:
# BA17 item 2's first control mutated an explanatory string and left every
# outcome reachable, so the runner correctly reported CONTROL NOT DETECTED. The
# anchor below returns before the precedence block runs at all.
CONTROL = (
    "CONTROL -- ratio_rule_diff returns DEFENSIBLE before any branch",
    R,
    b'    codes = {f["code"] for f in fails}\n',
    b'    codes = set()\n    fails = []\n',
    [f"{T}::test_a_standardised_over_a_raw_coefficient_defines_NEITHER",
     f"{T}::test_a_straddling_denominator_yields_the_DIFFERENCE",
     f"{T}::test_two_settings_force_both_to_be_named"],
)

MUTATIONS = [
    # ---- the denominator half -------------------------------------------
    # The quiet branch. Without it an unmeasured interval reads as an interval
    # excluding zero, and every bare multiplier in a prose record passes.
    ("absent denominator interval no longer flagged", R,
     b'    if dci is None:\n',
     b'    if False:\n',
     [f"{T}::test_an_absent_denominator_interval_is_UNVERIFIABLE_not_a_measured_failure",
      f"{T}::test_a_measured_straddle_outranks_an_absent_interval"]),

    # The two denominator cells must stay distinct. Folding the unverifiable
    # one back into the measured-failure one is the merge this module was
    # rewritten to avoid: it reports ignorance as a finding, and it swamps the
    # real findings in any prose record.
    ("the unverifiable cell collapses into the measured-failure cell", R,
     b'                  "DENOMINATOR_INTERVAL_STRADDLES_ZERO"}:\n'
     b'        outcome = "REPORT_THE_DIFFERENCE_INSTEAD"\n',
     b'                  "DENOMINATOR_INTERVAL_STRADDLES_ZERO",\n'
     b'                  "DENOMINATOR_INTERVAL_ABSENT"}:\n'
     b'        outcome = "REPORT_THE_DIFFERENCE_INSTEAD"\n',
     [f"{T}::test_an_absent_denominator_interval_is_UNVERIFIABLE_not_a_measured_failure",
      f"{T}::test_a_measured_straddle_outranks_an_absent_interval"]),

    # gotcha 25b, and the branch `fitted_vs_observed` already carries. The
    # agreement fixture must fail too, which is what makes this module an
    # extension of that one rather than a second opinion.
    ("straddles-zero branch unreachable", R,
     b'    elif _straddles_zero(dci):\n',
     b'    elif False:\n',
     [f"{T}::test_a_straddling_denominator_yields_the_DIFFERENCE",
      f"{T}::test_the_straddling_branch_agrees_with_span_comparison_verdict"]),

    # The interval is closed on both ends: an interval of exactly [0, h] does
    # not exclude zero. A strict comparison would pass it.
    ("straddling test becomes strict at the endpoints", R,
     b'    return lo <= 0.0 <= hi\n',
     b'    return lo < 0.0 < hi\n',
     [f"{T}::test_an_interval_touching_zero_does_not_exclude_zero"]),

    # NOTE the anchor carries the following line: the same condition appears
    # twice in the module (once as a failure, once in `_denominator_reason`),
    # and a two-occurrence anchor is reported as SKIP rather than silently
    # mutating the wrong one -- which is why the runner refuses on a count
    # other than exactly one.
    ("zero / non-finite denominator no longer flagged", R,
     b'    if not _finite(dv) or float(dv) == 0.0:\n        fails.append({\n',
     b'    if False:\n        fails.append({\n',
     [f"{T}::test_a_zero_denominator_reports_the_numerator_alone",
      f"{T}::test_the_zero_denominator_branch_agrees_with_excess_comparison"]),

    # ---- amendment BE: the SYMMETRIC half --------------------------------
    # Without this the rule constrains only the denominator, and a ratio of two
    # small numbers whose numerator carries no interval passes -- the exact
    # shape AC14 ruled on, and the gap BD found and reported.
    ("BE: the numerator's missing interval is no longer flagged", R,
     b'    nci = _interval_of(numerator)\n    if nci is None:\n',
     b'    nci = _interval_of(numerator)\n    if False:\n',
     [f"{T}::test_an_absent_NUMERATOR_interval_reclassifies_a_ratio_that_used_to_pass"]),

    # The widened cell must be reachable from BOTH sides. Routing only the
    # numerator into it restores the asymmetry in the other direction.
    ("BE: the widened cell drops the denominator arm", R,
     b'    elif codes & {"DENOMINATOR_INTERVAL_ABSENT", "NUMERATOR_INTERVAL_ABSENT"}:\n',
     b'    elif codes & {"NUMERATOR_INTERVAL_ABSENT"}:\n',
     [f"{T}::test_the_mirror_an_absent_DENOMINATOR_interval_reaches_the_same_cell",
      f"{T}::test_a_measured_straddle_outranks_an_absent_interval"]),

    # EXACT is what stops the extension flagging the record's own compliant
    # cases. Ignoring it makes every declared threshold and integer count
    # unverifiable, which is the swamping the cell was split out to avoid.
    ("BE: EXACT no longer supplies an interval", R,
     b'    if ci is EXACT:\n        v = comp.get("value")\n        return [v, v] if _finite(v) else None\n',
     b'    if ci is EXACT:\n        return None\n',
     [f"{T}::test_an_EXACT_component_is_not_an_unmeasured_one",
      f"{T}::test_two_EXACT_counts_are_DEFENSIBLE"]),

    # ...and EXACT must not become an escape hatch: an exactly-known ZERO is
    # still a denominator no ratio can be taken against.
    ("BE: EXACT is treated as excluding zero unconditionally", R,
     b'        return [v, v] if _finite(v) else None\n',
     b'        return [v - 1.0, v - 1.0] if _finite(v) else None\n',
     [f"{T}::test_an_EXACT_denominator_of_zero_still_fails"]),

    # ---- the scale half (BC8a) ------------------------------------------
    ("differing scales no longer flagged", R,
     b'    elif ns != ds:\n',
     b'    elif False:\n',
     [f"{T}::test_a_standardised_over_a_raw_coefficient_defines_NEITHER",
      f"{T}::test_the_precedence_puts_scale_above_denominator_above_setting"]),

    ("an absent scale reads as a shared scale", R,
     b'    if ns is None or ds is None:\n',
     b'    if False:\n',
     [f"{T}::test_an_unstated_scale_is_reported_absent_not_assumed_to_match"]),

    # The DIFFERENCE is the remedy everywhere except here. If it were computed
    # across a scale mismatch, the check would offer an undefined remedy.
    ("the difference is computed across a scale mismatch", R,
     b'    if scales_ok and _finite(nv) and _finite(dv):\n',
     b'    if _finite(nv) and _finite(dv):\n',
     [f"{T}::test_a_standardised_over_a_raw_coefficient_defines_NEITHER"]),

    # ---- the setting half -----------------------------------------------
    ("differing parameter settings no longer flagged", R,
     b'    elif na != da:\n',
     b'    elif False:\n',
     [f"{T}::test_two_settings_force_both_to_be_named",
      f"{T}::test_the_precedence_puts_scale_above_denominator_above_setting"]),

    ("an absent setting reads as a matching setting", R,
     b'    if na is None or da is None:\n',
     b'    if False:\n',
     [f"{T}::test_an_absent_setting_cannot_be_asserted_to_match"]),

    # Ignorance must not outrank a measured defect. Restoring the earlier
    # ordering puts UNVERIFIABLE above the settings cell, and since almost
    # nothing in a prose record carries an interval, every actionable finding
    # would be reported under the ambient gap instead.
    ("precedence: the unverifiable cell promoted above the settings cell", R,
     b'    elif codes & {"SETTINGS_DIFFER", "SETTING_ABSENT"}:\n'
     b'        outcome = "DEFENSIBLE_ONLY_WITH_THE_SETTINGS_NAMED"\n'
     b'    elif codes & {"DENOMINATOR_INTERVAL_ABSENT", "NUMERATOR_INTERVAL_ABSENT"}:\n'
     b'        outcome = "UNVERIFIABLE_A_COMPONENT_HAS_NO_MEASURED_INTERVAL"\n',
     b'    elif codes & {"DENOMINATOR_INTERVAL_ABSENT", "NUMERATOR_INTERVAL_ABSENT"}:\n'
     b'        outcome = "UNVERIFIABLE_A_COMPONENT_HAS_NO_MEASURED_INTERVAL"\n'
     b'    elif codes & {"SETTINGS_DIFFER", "SETTING_ABSENT"}:\n'
     b'        outcome = "DEFENSIBLE_ONLY_WITH_THE_SETTINGS_NAMED"\n',
     [f"{T}::test_a_measured_setting_mismatch_outranks_an_absent_interval"]),

    # ---- the precedence -------------------------------------------------
    # Reordering so the setting half wins would tell an author to name two
    # thresholds on a ratio meaningless at every threshold.
    ("precedence: the scale branch demoted below the setting branch", R,
     b'    if codes & {"SCALES_DIFFER", "SCALE_ABSENT"}:\n'
     b'        outcome = "NEITHER_RATIO_NOR_DIFFERENCE_IS_DEFINED"\n',
     b'    if codes & {"SETTINGS_DIFFER", "SETTING_ABSENT"}:\n'
     b'        outcome = "DEFENSIBLE_ONLY_WITH_THE_SETTINGS_NAMED"\n',
     [f"{T}::test_the_precedence_puts_scale_above_denominator_above_setting",
      f"{T}::test_a_standardised_over_a_raw_coefficient_defines_NEITHER"]),

    # ---- the arithmetic, which must stay SEPARATE ------------------------
    # A stated multiplier that does not reproduce is a fact a reader needs
    # alongside the outcome, not folded into it.
    ("the stated-multiplier check no longer runs", R,
     b'    if stated_multiplier is not None:\n',
     b'    if False:\n',
     [f"{T}::test_a_stated_multiplier_that_does_not_reproduce_is_flagged_separately"]),

    ("the stated-multiplier tolerance swallows a real disagreement", R,
     b'        elif abs(float(stated_multiplier) - ratio) > rel_tol * abs(ratio):\n',
     b'        elif abs(float(stated_multiplier) - ratio) > 10.0 * abs(ratio):\n',
     [f"{T}::test_a_stated_multiplier_that_does_not_reproduce_is_flagged_separately"]),

    # ---- the candidate finder -------------------------------------------
    # A finder that misses is worse than one that over-reports, so the forms
    # are enumerated and each must stay.
    ("the finder loses the -fold and 'times' forms", R,
     b'r"(?<![\\w.])(\\d{1,6}(?:,\\d{3})*(?:\\.\\d+)?)\\s*(x\\b|\xc3\x97|-fold\\b|\\s*times\\b)"',
     b'r"(?<![\\w.])(\\d{1,6}(?:,\\d{3})*(?:\\.\\d+)?)\\s*(x\\b|\xc3\x97)"',
     [f"{T}::test_the_finder_reports_line_and_context_for_each_form"]),

    # Without the preceding-character guard the finder fires inside version
    # strings and decimals -- and a check whose false positives land on its own
    # documentation gets disabled rather than fixed (gotcha 20's corollary).
    ("the finder's preceding-character guard removed", R,
     b"r\"(?<![\\w.])(\\d{1,6}",
     b"r\"(\\d{1,6}",
     [f"{T}::test_the_finder_does_not_fire_on_a_decimal_inside_a_word_or_version"]),

    ("the finder loses the thousands separator", R,
     b'(?:,\\d{3})*',
     b'',
     [f"{T}::test_the_finder_reads_a_thousands_separator"]),
]


def main() -> int:
    from tests.mutation_matrix_av import run_one
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

    print(f"{'mutation':<58} {'result':<10} killed")
    bad = 0
    for label, path, old, new, expect in MUTATIONS:
        survived = apply_and_run(path, old, new, expect)
        if survived is None:
            print(f"{label:<58} {'SKIP':<10} anchor did not match exactly once")
            bad += 1
            continue
        status = "KILLED" if not survived else "SURVIVED"
        print(f"{label:<58} {status:<10} "
              f"{len(expect) - len(survived)}/{len(expect)}")
        for t in survived:
            print(f"{'':<58} {'':<10} SURVIVOR: {t}")
        bad += bool(survived)

    print()
    print("ALL MUTATIONS KILLED THEIR NAMED FIXTURES" if not bad
          else f"{bad} MUTATION(S) NOT FULLY KILLED -- real coverage gap")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
