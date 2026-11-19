"""Tests for `ratio_rule` -- CLAUDE.md gotcha 33 as a check.

**THE CASES ARE WRITTEN DOWN, NOT DERIVED FROM THE MODULE.** Nothing here
iterates `ratio_rule.OUTCOMES` or `ratio_rule.FAILURES`; every expected string
is a literal. Gotcha 32(b): a test parametrised over the constant it is
checking deletes its own case when the constant changes, and the mutation
survives. The converse is the point -- adding an outcome must FAIL
`test_the_outcome_vocabulary_is_exactly_these_five` and be updated
deliberately, and it has now done so twice.

**AND THERE ARE FIXTURES THAT SHOULD PASS.** A check that flags everything
discriminates nothing, and "found N defects" from an instrument never shown to
clear a sound case is not a measurement. `test_a_sound_ratio_is_DEFENSIBLE`
and the AQ 3.20x case are the controls.
"""
import pytest

from src.analysis import carrier_support as cs
from src.analysis import fitted_vs_observed as fvo
from src.analysis import ratio_rule as rr


# --------------------------------------------------------------- vocabulary

def test_the_outcome_vocabulary_is_exactly_these_five():
    """Written out rather than read from the module (gotcha 32b). If a sixth
    outcome is added this test fails, which is what it is for -- and it has now
    failed deliberately TWICE: when the unverifiable cell was split out of
    REPORT_THE_DIFFERENCE_INSTEAD, and when amendment BE widened that cell from
    the denominator to either component and renamed it to match."""
    assert set(rr.OUTCOMES) == {
        "NEITHER_RATIO_NOR_DIFFERENCE_IS_DEFINED",
        "REPORT_THE_DIFFERENCE_INSTEAD",
        "UNVERIFIABLE_A_COMPONENT_HAS_NO_MEASURED_INTERVAL",
        "DEFENSIBLE_ONLY_WITH_THE_SETTINGS_NAMED",
        "DEFENSIBLE",
    }


def test_the_failure_codes_are_exactly_these_nine():
    """Same discipline on the codes. `NUMERATOR_INTERVAL_ABSENT` is BE's
    addition and it is written out here, not derived."""
    assert set(rr.FAILURES) == {
        "SCALES_DIFFER", "SCALE_ABSENT",
        "DENOMINATOR_IS_ZERO_OR_NON_FINITE",
        "DENOMINATOR_INTERVAL_STRADDLES_ZERO",
        "DENOMINATOR_INTERVAL_ABSENT",
        "NUMERATOR_INTERVAL_ABSENT",
        "SETTINGS_DIFFER", "SETTING_ABSENT",
        "STATED_MULTIPLIER_DISAGREES_WITH_ITS_COMPONENTS",
    }


# ------------------------------------------------- the case that must PASS

def test_a_sound_ratio_is_DEFENSIBLE():
    """The control for the whole instrument: same scale, same setting, an
    interval excluding zero. If this does not come back DEFENSIBLE, every
    failure the check reports elsewhere is uninterpretable."""
    out = rr.ratio_rule_diff(
        "a sound ratio",
        rr.component(8.0, "pp of r_f", at="theta=0.70", ci=[7.0, 9.0],
                     label="the numerator"),
        rr.component(2.0, "pp of r_f", at="theta=0.70", ci=[1.5, 2.5],
                     label="the denominator"),
        stated_multiplier=4.0)
    assert out["outcome"] == "DEFENSIBLE"
    assert out["failures"] == []
    assert out["ratio"] == pytest.approx(4.0)
    assert out["difference"] == pytest.approx(6.0)


def test_the_AQ_penalty_ratio_passes_as_recorded():
    """A REAL ratio from the record that the rule does not condemn. Amendment
    AQ's 3.20x: both sides are the same quantity -- a cross-seed CV penalty,
    the CV ratio minus one -- for two normaliser arms at one comparison count,
    and the denominator's bootstrap interval excludes zero. The record's own
    remedy was to carry the bound [1.41, 8.58], not to withhold the ratio, so
    the check must agree. **If this came back as a failure the instrument
    would be flagging the record's own compliant case.**"""
    out = rr.ratio_rule_diff(
        "AQ: the smooth arm's CV penalty over hard-bin's",
        rr.component(0.022265, "cross-seed CV penalty (CV ratio minus one)",
                     at="n_compare=39", ci=[0.015948, 0.029180],
                     label="the smooth arm's penalty"),
        rr.component(0.006947, "cross-seed CV penalty (CV ratio minus one)",
                     at="n_compare=39", ci=[0.003402, 0.011344],
                     label="hard-bin's penalty"),
        stated_multiplier=3.20)
    assert out["outcome"] == "DEFENSIBLE"
    assert out["failure_codes"] == []
    assert out["ratio"] == pytest.approx(3.2050, rel=1e-3)


# --------------------------------------------------- the scale half (BC8a)

def test_a_standardised_over_a_raw_coefficient_defines_NEITHER():
    """AS section 4's '19x': `b1_z` is standardised, `gamma` is raw-scale.
    **The usual remedy fails here too** -- a difference between them is no
    more meaningful than a ratio, so the branch must say so rather than fall
    back on 'report the difference'."""
    out = rr.ratio_rule_diff(
        "AS section 4: b1_z against gamma",
        rr.component(0.1064, "logit coefficient per SD", at="theta=0.70",
                     ci=[0.09, 0.12], label="b1_z"),
        rr.component(0.0056, "logit coefficient per raw unit", at="theta=0.70",
                     ci=[0.004, 0.007], label="gamma"))
    assert out["outcome"] == "NEITHER_RATIO_NOR_DIFFERENCE_IS_DEFINED"
    assert "SCALES_DIFFER" in out["failure_codes"]
    assert out["difference"] is None
    assert "own scale" in out["defensible_form"]


def test_an_unstated_scale_is_reported_absent_not_assumed_to_match():
    """Gotcha 15: an existence check is not an identity check, and an ABSENT
    field must never read as agreement. Two `None` scales are not one shared
    scale."""
    out = rr.ratio_rule_diff(
        "both scales unstated",
        rr.component(8.0, None, at="theta=0.70", ci=[7.0, 9.0]),
        rr.component(2.0, None, at="theta=0.70", ci=[1.5, 2.5]))
    assert out["outcome"] == "NEITHER_RATIO_NOR_DIFFERENCE_IS_DEFINED"
    assert "SCALE_ABSENT" in out["failure_codes"]


# ----------------------------------------------- the denominator half

def test_a_straddling_denominator_yields_the_DIFFERENCE():
    """BB7.3's stratum 1: a naive ratio of +0.32 against a denominator whose
    interval is [-20.12, +7.94]. The ratio must be withheld and the
    difference reported in its place."""
    out = rr.ratio_rule_diff(
        "BB7.3 stratum 1",
        rr.component(-1.95, "pp", at="theta=0.70", ci=[-3.0, -1.0],
                     label="the fitted span"),
        rr.component(-6.09, "pp", at="theta=0.70", ci=[-20.12, 7.94],
                     label="the observed span"))
    assert out["outcome"] == "REPORT_THE_DIFFERENCE_INSTEAD"
    assert "DENOMINATOR_INTERVAL_STRADDLES_ZERO" in out["failure_codes"]
    assert out["difference"] == pytest.approx(4.14)
    assert "ranks noise" in out["defensible_form"]


def test_an_interval_touching_zero_does_not_exclude_zero():
    """The rule says the interval must EXCLUDE zero, so an endpoint sitting on
    zero fails it. A strict comparison would pass [0.0, 0.5], and the
    denominator here is non-zero so no other branch can pre-empt the test."""
    out = rr.ratio_rule_diff(
        "an interval whose lower endpoint is exactly zero",
        rr.component(4.0, "pp", at="theta=0.70", ci=[3.5, 4.5]),
        rr.component(0.25, "pp", at="theta=0.70", ci=[0.0, 0.5]))
    assert out["outcome"] == "REPORT_THE_DIFFERENCE_INSTEAD"
    assert "DENOMINATOR_INTERVAL_STRADDLES_ZERO" in out["failure_codes"]
    # And the mirror: an interval whose UPPER endpoint is exactly zero.
    mirror = rr.ratio_rule_diff(
        "an interval whose upper endpoint is exactly zero",
        rr.component(-4.0, "pp", at="theta=0.70", ci=[-4.5, -3.5]),
        rr.component(-0.25, "pp", at="theta=0.70", ci=[-0.5, 0.0]))
    assert mirror["outcome"] == "REPORT_THE_DIFFERENCE_INSTEAD"


def test_a_zero_denominator_reports_the_numerator_alone():
    out = rr.ratio_rule_diff(
        "vanishing denominator",
        rr.component(6.787, "pp", at="theta=0.70", ci=[6.0, 7.5]),
        rr.component(0.0, "pp", at="theta=0.70", ci=[-0.1, 0.1]))
    assert out["outcome"] == "REPORT_THE_DIFFERENCE_INSTEAD"
    assert "DENOMINATOR_IS_ZERO_OR_NON_FINITE" in out["failure_codes"]
    assert out["ratio"] is None


def test_an_absent_denominator_interval_is_UNVERIFIABLE_not_a_measured_failure():
    """The quiet one, and the one whose CELL matters. 'Excludes zero' is
    unestablished without an interval, which is a third state -- not
    'satisfied', and not the measured straddle either. BC14's distinction:
    ignorance is not independence, and reporting it in the straddle's cell
    would tell an author to withdraw a ratio nobody has tested."""
    out = rr.ratio_rule_diff(
        "no interval on the denominator",
        rr.component(3.12, "pp of r_f per SD", at="theta=0.70",
                     ci=[2.9, 3.4], label="d_fs"),
        rr.component(0.3635, "pp of r_f per SD", at="theta=0.70", ci=None,
                     label="the log M bound"))
    assert out["outcome"] == \
        "UNVERIFIABLE_A_COMPONENT_HAS_NO_MEASURED_INTERVAL"
    assert "DENOMINATOR_INTERVAL_ABSENT" in out["failure_codes"]
    assert out["difference"] == pytest.approx(2.7565)
    assert "unestablished" in out["defensible_form"]


def test_an_absent_NUMERATOR_interval_reclassifies_a_ratio_that_used_to_pass():
    """**Amendment BE's extension, and the reason for it.** The rule as first
    drafted constrained only the denominator, so a ratio of two small numbers
    whose NUMERATOR carried no interval came back clean -- exactly the shape
    AC14 ruled on. This fixture is amendment U's real case: z's seed-variance
    penalty of 8.4% with no interval recorded, over the ratio arm's 0.69%
    [0.34%, 1.13%]."""
    out = rr.ratio_rule_diff(
        "PREREGISTRATION U: 'costing 12x less'",
        rr.component(8.4, "percent seed-variance penalty", at="M4",
                     ci=None, label="z's penalty"),
        rr.component(0.69, "percent seed-variance penalty", at="M4",
                     ci=[0.34, 1.13], label="the ratio arm's penalty"),
        stated_multiplier=12.0)
    assert out["outcome"] == "UNVERIFIABLE_A_COMPONENT_HAS_NO_MEASURED_INTERVAL"
    assert "NUMERATOR_INTERVAL_ABSENT" in out["failure_codes"]
    assert "DENOMINATOR_INTERVAL_ABSENT" not in out["failure_codes"]
    assert "z's penalty" in out["defensible_form"]


def test_the_mirror_an_absent_DENOMINATOR_interval_reaches_the_same_cell():
    """The mirror of the case above, so the widened cell is shown to be reached
    from both sides rather than only from the one BE added."""
    out = rr.ratio_rule_diff(
        "the mirror",
        rr.component(8.4, "percent", at="M4", ci=[7.0, 9.8],
                     label="the numerator"),
        rr.component(0.69, "percent", at="M4", ci=None,
                     label="the denominator"))
    assert out["outcome"] == "UNVERIFIABLE_A_COMPONENT_HAS_NO_MEASURED_INTERVAL"
    assert "DENOMINATOR_INTERVAL_ABSENT" in out["failure_codes"]
    assert "NUMERATOR_INTERVAL_ABSENT" not in out["failure_codes"]


def test_an_EXACT_component_is_not_an_unmeasured_one():
    """**The case the extension would get wrong without `EXACT`, and it is a
    real one:** amendment AB1's *"clears the rotation-null floor by 4.83x"* is
    a pre-registered threshold of exactly 0.70 over a null percentile carrying
    an MC standard error. Reporting it as unverifiable because nobody put a
    confidence interval on a declared constant is true, vacuous, and recreates
    the swamping the cell was split out to avoid."""
    out = rr.ratio_rule_diff(
        "AB1: theta over the rotation null's 99.9th percentile",
        rr.component(0.70, "decoder cosine similarity",
                     at="the reference threshold", ci=rr.EXACT,
                     label="theta"),
        rr.component(0.14487, "decoder cosine similarity",
                     at="the reference threshold", ci=[0.14440, 0.14535],
                     label="the rotation null's 99.9th percentile"),
        stated_multiplier=4.83)
    assert out["outcome"] == "DEFENSIBLE"
    assert out["failure_codes"] == []
    assert out["numerator_exact"] is True
    assert out["denominator_exact"] is False


def test_an_EXACT_denominator_of_zero_still_fails():
    """`EXACT` must not become an escape hatch. An exactly-known ZERO is still
    a denominator a ratio cannot be taken against, and the substitution of the
    point for the interval must reach the same answer the zero-value branch
    reaches independently."""
    out = rr.ratio_rule_diff(
        "an exact zero denominator",
        rr.component(4.0, "pp", at="theta=0.70", ci=[3.5, 4.5]),
        rr.component(0.0, "pp", at="theta=0.70", ci=rr.EXACT))
    assert out["outcome"] == "REPORT_THE_DIFFERENCE_INSTEAD"
    assert "DENOMINATOR_IS_ZERO_OR_NON_FINITE" in out["failure_codes"]
    assert "DENOMINATOR_INTERVAL_STRADDLES_ZERO" in out["failure_codes"]


def test_two_EXACT_counts_are_DEFENSIBLE():
    """Amendment AB7's mutual-versus-one-way agreement is two integer match
    counts. Under the denominator-only rule this was UNVERIFIABLE; under BE it
    is clean, which is a reclassification in the OPPOSITE direction from the
    one the extension was built for and is reported as such."""
    out = rr.ratio_rule_diff(
        "AB7: 1,929,316 mutual against 1,934,189 one-way",
        rr.component(1929316, "matched anchors", at="theta=0.70",
                     ci=rr.EXACT, label="MNN matches"),
        rr.component(1934189, "matched anchors", at="theta=0.70",
                     ci=rr.EXACT, label="one-way matches"))
    assert out["outcome"] == "DEFENSIBLE"
    assert out["ratio"] == pytest.approx(0.99748, rel=1e-4)


def test_a_measured_straddle_outranks_an_absent_interval():
    """The two must not be interchangeable, so the precedence is asserted on
    an input where both could apply: the denominator's interval is measured
    AND covers zero, which is a fact, not a gap."""
    measured = rr.ratio_rule_diff(
        "measured and straddling",
        rr.component(4.0, "pp", at="theta=0.70", ci=[3.5, 4.5]),
        rr.component(0.25, "pp", at="theta=0.70", ci=[-0.5, 0.5]))
    absent = rr.ratio_rule_diff(
        "not measured at all",
        rr.component(4.0, "pp", at="theta=0.70", ci=[3.5, 4.5]),
        rr.component(0.25, "pp", at="theta=0.70", ci=None))
    assert measured["outcome"] == "REPORT_THE_DIFFERENCE_INSTEAD"
    assert absent["outcome"] == \
        "UNVERIFIABLE_A_COMPONENT_HAS_NO_MEASURED_INTERVAL"
    assert measured["outcome"] != absent["outcome"]


# ------------------------------------------------------ the setting half

def test_two_settings_force_both_to_be_named():
    """AT13's case: +3.12 pp at theta=0.70 divided by a bound ATTAINED at
    theta=0.60. Arithmetic, but not a property of the two terms."""
    out = rr.ratio_rule_diff(
        "AT13: d_fs over the log M bound",
        rr.component(3.12, "pp of r_f per SD", at="theta=0.70",
                     ci=[2.9, 3.4], label="d_fs's effect"),
        rr.component(0.3635, "pp of r_f per SD", at="theta=0.60",
                     ci=[0.30, 0.42], label="the log M bound"),
        stated_multiplier=8.59)
    assert out["outcome"] == "DEFENSIBLE_ONLY_WITH_THE_SETTINGS_NAMED"
    assert "SETTINGS_DIFFER" in out["failure_codes"]
    assert "theta=0.70" in out["defensible_form"]
    assert "theta=0.60" in out["defensible_form"]


def test_an_absent_setting_cannot_be_asserted_to_match():
    out = rr.ratio_rule_diff(
        "one setting unrecorded",
        rr.component(8.0, "pp", at=None, ci=[7.0, 9.0]),
        rr.component(2.0, "pp", at="theta=0.70", ci=[1.5, 2.5]))
    assert out["outcome"] == "DEFENSIBLE_ONLY_WITH_THE_SETTINGS_NAMED"
    assert "SETTING_ABSENT" in out["failure_codes"]


def test_a_measured_setting_mismatch_outranks_an_absent_interval():
    """The cell a prose record actually lands in: two settings that differ AND
    a denominator nobody put an interval on. The outcome must name the
    mismatch, because that is the defect an author can fix in one sentence --
    if ignorance outranked it, the ambient state of the record would mask
    every actionable finding. Both codes still appear in `failures`."""
    out = rr.ratio_rule_diff(
        "the record's usual shape",
        rr.component(3.12, "pp of r_f per SD", at="theta=0.70",
                     ci=[3.08, 3.16], label="d_fs's effect"),
        rr.component(0.3635, "pp of r_f per SD", at="theta=0.60", ci=None,
                     label="the log M bound"),
        stated_multiplier=9.0)
    assert out["outcome"] == "DEFENSIBLE_ONLY_WITH_THE_SETTINGS_NAMED"
    assert set(out["failure_codes"]) == {
        "SETTINGS_DIFFER", "DENOMINATOR_INTERVAL_ABSENT",
        "STATED_MULTIPLIER_DISAGREES_WITH_ITS_COMPONENTS"}


def test_the_precedence_puts_scale_above_denominator_above_setting():
    """All three conditions fail at once. The outcome must be the most severe,
    because a reader who fixes only the setting still has a meaningless
    ratio."""
    out = rr.ratio_rule_diff(
        "everything fails",
        rr.component(8.0, "logits", at="theta=0.70", ci=[7.0, 9.0]),
        rr.component(2.0, "pp", at="theta=0.60", ci=[-1.0, 5.0]))
    assert out["outcome"] == "NEITHER_RATIO_NOR_DIFFERENCE_IS_DEFINED"
    assert set(out["failure_codes"]) == {
        "SCALES_DIFFER", "DENOMINATOR_INTERVAL_STRADDLES_ZERO",
        "SETTINGS_DIFFER"}


# ------------------------------------------------- the arithmetic, separately

def test_a_stated_multiplier_that_does_not_reproduce_is_flagged_separately():
    """'Nine times' where the components give 8.59 is an arithmetic fact, not
    a rule violation, and a reader needs the two separately. So the failure is
    reported and the OUTCOME does not move."""
    out = rr.ratio_rule_diff(
        "nine times",
        rr.component(3.12, "pp of r_f per SD", at="theta=0.70", ci=[2.9, 3.4]),
        rr.component(0.3635, "pp of r_f per SD", at="theta=0.70",
                     ci=[0.30, 0.42]),
        stated_multiplier=9.0)
    assert out["outcome"] == "DEFENSIBLE"
    assert out["failure_codes"] == [
        "STATED_MULTIPLIER_DISAGREES_WITH_ITS_COMPONENTS"]
    assert out["ratio"] == pytest.approx(8.5832, rel=1e-3)


def test_a_stated_multiplier_within_tolerance_is_not_flagged():
    out = rr.ratio_rule_diff(
        "8.59x",
        rr.component(3.12, "pp of r_f per SD", at="theta=0.70", ci=[2.9, 3.4]),
        rr.component(0.3635, "pp of r_f per SD", at="theta=0.70",
                     ci=[0.30, 0.42]),
        stated_multiplier=8.59)
    assert out["failure_codes"] == []


# --------------------------------------- agreement with the two incumbents

def test_the_straddling_branch_agrees_with_span_comparison_verdict():
    """This module EXTENDS `fitted_vs_observed` rather than duplicating it, so
    the two must agree on the condition they share. The fixture is the same
    denominator in both idioms; if either function changes its branch, this
    fails HERE."""
    span = fvo.span_comparison_verdict(
        {"status": "COMPUTED", "mean_span_pp": 0.5,
         "ci95_span_pp": [0.4, 0.6], "n_seeds": 20},
        {"status": "COMPUTED", "mean_span_pp": 1.0,
         "ci95_span_pp": [-4.0, 6.0], "n_seeds": 20})
    mine = rr.ratio_rule_diff(
        "same condition",
        rr.component(0.5, "pp", at="theta=0.70", ci=[0.4, 0.6]),
        rr.component(1.0, "pp", at="theta=0.70", ci=[-4.0, 6.0]))
    assert span["verdict"] == "UNINFORMATIVE_OBSERVED_SPAN_STRADDLES_ZERO"
    assert span["ratio_fitted_over_observed"] is None
    assert mine["outcome"] == "REPORT_THE_DIFFERENCE_INSTEAD"
    assert mine["ratio"] is not None  # the ratio EXISTS; it is not QUOTABLE
    assert span["difference_fitted_minus_observed_pp"] == pytest.approx(
        mine["difference"])


def test_the_zero_denominator_branch_agrees_with_excess_comparison():
    excess = cs.excess_comparison({"excess_A_minus_matched_pp": 0.0},
                                  {"excess_A_minus_matched_pp": 1.0})
    mine = rr.ratio_rule_diff(
        "same condition",
        rr.component(1.0, "pp", at="theta=0.70", ci=[0.5, 1.5]),
        rr.component(0.0, "pp", at="theta=0.70", ci=[-0.2, 0.2]))
    assert excess["ratio_double_over_single"] is None
    assert excess["outcome"] == "UNDEFINED_SINGLE_EXCESS_IS_ZERO_OR_NON_FINITE"
    assert mine["ratio"] is None
    assert mine["outcome"] == "REPORT_THE_DIFFERENCE_INSTEAD"
    # The shared CONDITION is that the denominator is zero, so both must name
    # it. Asserting only that the ratio is None would pass on the strength of
    # a different branch -- the interval here also covers zero.
    assert "DENOMINATOR_IS_ZERO_OR_NON_FINITE" in mine["failure_codes"]


# ------------------------------------------------------- the candidate finder

def test_the_finder_reports_line_and_context_for_each_form():
    text = ("nothing here\n"
            "a factor of 3.20x on the penalty\n"
            "the ratio is 8.59×\n"
            "roughly nine times, which is prose and not a token\n"
            "a 2.43-fold acceleration\n"
            "moves by 12.6 times the incumbent\n")
    hits = rr.find_multipliers(text)
    assert [h["line"] for h in hits] == [2, 3, 5, 6]
    assert [h["value"] for h in hits] == [3.20, 8.59, 2.43, 12.6]
    assert "penalty" in hits[0]["context"]


def test_the_finder_does_not_fire_on_a_decimal_inside_a_word_or_version():
    """False positives land on the documentation that explains the rule
    (gotcha 20's corollary), so the boundary matters. Each of these WOULD
    match without the preceding-character guard: `3x` after a dot, `6x` after
    a letter, and a six-digit tail of a job id."""
    text = ("tag v1.2.3x, image cu126x, job 20438265x\n"
            "torch 2.13.0+cu126 and theta 0.70 and 1e-4 and x2 backwards\n")
    assert rr.find_multipliers(text) == []


def test_the_finder_reads_a_thousands_separator():
    hits = rr.find_multipliers("understated by 1,024x in the worst case\n")
    assert [h["value"] for h in hits] == [1024.0]


# =====================================================================
# THE RANGE-COHERENCE RULE (amendment BL)
# =====================================================================
#
# SAME DISCIPLINE AS ABOVE: every expected string is a literal, nothing
# iterates `rr.RANGE_OUTCOMES` or `rr.RANGE_FAILURES`, and the controls that
# must return COHERENT are here so "the rule found N incoherent ranges" comes
# from an instrument shown to clear a sound one.

def _sd(v, at, signed=None):
    """A dispersion component on the shared linear-predictor-SD scale."""
    return rr.component(v, "SD of a linear-predictor contribution (logits)",
                        at=at, label="term SD", magnitude_of=signed)


def _lg(v, at):
    """A signed logit-scale component, declaring itself signed."""
    return rr.component(v, "logits", at=at, magnitude_of=rr.SIGNED)


def _pt(at, num, den):
    return rr.grid_point(at, num, den)


# --------------------------------------------------------------- vocabulary

def test_the_range_outcome_vocabulary_is_exactly_these_eight():
    """Written out, not read from the module (gotcha 32b). A ninth cell must
    fail this test and be added deliberately."""
    assert set(rr.RANGE_OUTCOMES) == {
        "NOT_A_RANGE_FEWER_THAN_TWO_GRID_POINTS",
        "NOT_A_RANGE_A_GRID_POINT_HAS_NO_DEFINED_COMPARISON",
        "INCOHERENT_BOTH_COMPONENTS_CHANGE_SIGN",
        "INCOHERENT_ONE_COMPONENT_CHANGES_SIGN",
        "INCOHERENT_A_COMPONENT_IS_ZERO_AT_A_GRID_POINT",
        "INCOHERENT_THE_TERM_ORDERING_CHANGES",
        "NOT_EVALUABLE_COHERENCE_IS_UNDETERMINED_ON_THIS_RANGE",
        "COHERENT",
    }


def test_the_range_failure_codes_are_exactly_these_ten():
    assert set(rr.RANGE_FAILURES) == {
        "FEWER_THAN_TWO_GRID_POINTS",
        "POINT_COMPARISON_UNDEFINED",
        "NUMERATOR_CHANGES_SIGN",
        "DENOMINATOR_CHANGES_SIGN",
        "NUMERATOR_IS_ZERO_AT_A_GRID_POINT",
        "DENOMINATOR_IS_ZERO_AT_A_GRID_POINT",
        "TERM_ORDERING_CHANGES",
        "GRID_POINT_VALUE_ABSENT_OR_NON_FINITE",
        "SIGN_BASIS_UNDECLARED_FOR_A_NON_NEGATIVE_COMPONENT",
        "STATED_RANGE_DISAGREES_WITH_ITS_GRID",
    }


def test_the_two_vocabularies_are_disjoint():
    """The per-point rule keeps its five outcomes and the range rule has its
    own eight; a name in both would be the one-key-two-meanings defect gotcha
    16 committed a convention against."""
    assert not (set(rr.OUTCOMES) & set(rr.RANGE_OUTCOMES))
    assert not (set(rr.FAILURES) & set(rr.RANGE_FAILURES))


# ------------------------------------------------------------- THE CONTROLS

def test_a_sound_range_is_COHERENT():
    """THE CONTROL. A check that flags everything discriminates nothing."""
    out = rr.range_coherence_diff("sound", [
        _pt(0.30, _sd(0.40, 0.30, signed=+0.40), _sd(0.10, 0.30, signed=+0.10)),
        _pt(0.50, _sd(0.42, 0.50, signed=+0.42), _sd(0.09, 0.50, signed=+0.09)),
        _pt(0.70, _sd(0.38, 0.70, signed=+0.38), _sd(0.08, 0.70, signed=+0.08)),
    ])
    assert out["outcome"] == "COHERENT"
    assert out["failure_codes"] == []
    assert out["coherent_over_evaluable_points"] is True
    assert out["grid_ratio_min"] == pytest.approx(4.0, rel=1e-3)
    assert out["grid_ratio_max"] == pytest.approx(4.75, rel=1e-3)
    assert "coherent across its span" in out["defensible_form"]


def test_a_signed_component_that_happens_to_be_positive_is_COHERENT():
    """The other control, and the one that keeps `magnitude_of` from becoming
    a tax on every sound range: SIGNED is the claim that the value is itself
    the signed quantity."""
    out = rr.range_coherence_diff("all-positive but signed", [
        _pt(0.30, _lg(0.40, 0.30), _lg(0.10, 0.30)),
        _pt(0.70, _lg(0.38, 0.70), _lg(0.08, 0.70)),
    ])
    assert out["outcome"] == "COHERENT"


# ----------------------------------------------- the founding instance, BK-C1

def test_the_founding_instance_is_INCOHERENT_ONE_COMPONENT_CHANGES_SIGN():
    """BK's live case in miniature: the DENOMINATOR is an SD -- non-negative
    at every point -- while the coefficient it is the magnitude of reverses
    sign inside the quoted span, and every point clears the per-point rule.

    THE SHAPE IS WRITTEN DOWN RATHER THAN LOADED: three thetas standing for
    AR11's thirteen, with b1_z positive, positive, negative."""
    out = rr.range_coherence_diff("spline SD over alignment SD", [
        _pt(0.70, _sd(0.3814, 0.70, signed=+0.3814),
            _sd(0.011232, 0.70, signed=+0.005596)),
        _pt(0.75, _sd(0.3800, 0.75, signed=+0.3800),
            _sd(0.009430, 0.75, signed=+0.001570)),
        _pt(0.90, _sd(0.3700, 0.90, signed=+0.3700),
            _sd(0.026430, 0.90, signed=-0.037682)),
    ])
    assert out["outcome"] == "INCOHERENT_ONE_COMPONENT_CHANGES_SIGN"
    assert out["failure_codes"] == ["DENOMINATOR_CHANGES_SIGN"]
    assert out["sign_change_intervals"]["denominator"] == [(0.75, 0.90)]
    assert out["sign_change_intervals"]["numerator"] == []
    # Every point clears the per-point rule on the denominator condition: the
    # range rule is not re-detecting something the per-point rule already had.
    assert all(d["outcome"] != "REPORT_THE_DIFFERENCE_INSTEAD"
               for d in out["per_point"])
    assert [(s["at_from"], s["at_to"]) for s in out["coherent_sub_ranges"]] \
        == [(0.70, 0.75), (0.90, 0.90)]


def test_the_founding_instance_WITHOUT_a_declared_sign_basis_is_NOT_EVALUABLE():
    """THE CHECK THAT THE CHECK CAN FAIL, and the reason `magnitude_of` exists
    (gotchas 13 and 21). The identical grid with the signed sources NOT
    supplied is two non-negative SDs, on which a sign test read off the values
    can never fire. It must land in the ignorance cell, NEVER in COHERENT."""
    out = rr.range_coherence_diff("spline SD over alignment SD, undeclared", [
        _pt(0.70, _sd(0.3814, 0.70), _sd(0.011232, 0.70)),
        _pt(0.75, _sd(0.3800, 0.75), _sd(0.009430, 0.75)),
        _pt(0.90, _sd(0.3700, 0.90), _sd(0.026430, 0.90)),
    ])
    assert out["outcome"] == \
        "NOT_EVALUABLE_COHERENCE_IS_UNDETERMINED_ON_THIS_RANGE"
    assert out["failure_codes"] == [
        "SIGN_BASIS_UNDECLARED_FOR_A_NON_NEGATIVE_COMPONENT"]
    assert "ignorance, not a pass" in out["defensible_form"]


# ------------------------------------------------------------------ the cells

def test_one_grid_point_in_the_quoted_span_is_NOT_A_RANGE():
    out = rr.range_coherence_diff("single", [
        _pt(0.30, _sd(0.40, 0.30, +0.40), _sd(0.10, 0.30, +0.10)),
        _pt(0.70, _sd(0.38, 0.70, +0.38), _sd(0.08, 0.70, +0.08)),
    ], quoted=(0.65, 0.75))
    assert out["outcome"] == "NOT_A_RANGE_FEWER_THAN_TWO_GRID_POINTS"
    assert out["n_in_quoted_range"] == 1


def test_zero_grid_points_in_the_quoted_span_is_NOT_A_RANGE_not_COHERENT():
    """The empty case must map somewhere, and it maps to the structural cell:
    a span naming no grid point is a locator defect in the quote, and the
    vacuous "no sign change was observed" must not read as a pass."""
    out = rr.range_coherence_diff("empty", [
        _pt(0.30, _sd(0.40, 0.30, +0.40), _sd(0.10, 0.30, +0.10)),
    ], quoted=(0.80, 0.90))
    assert out["outcome"] == "NOT_A_RANGE_FEWER_THAN_TWO_GRID_POINTS"
    assert out["n_in_quoted_range"] == 0


def test_a_point_whose_scales_differ_makes_the_range_UNDEFINED():
    out = rr.range_coherence_diff("mixed scales", [
        _pt(0.30, _lg(0.40, 0.30), _lg(0.10, 0.30)),
        _pt(0.70, _lg(0.38, 0.70),
            rr.component(0.08, "pp of r_f", at=0.70, magnitude_of=rr.SIGNED)),
    ])
    assert out["outcome"] == \
        "NOT_A_RANGE_A_GRID_POINT_HAS_NO_DEFINED_COMPARISON"
    assert "POINT_COMPARISON_UNDEFINED" in out["failure_codes"]


def test_both_components_reversing_is_its_OWN_cell_and_outranks_one():
    """THE CELL THE BRIEF REQUIRED TO BE DECIDED. The ratio's sign is
    preserved -- +0.4/+0.1 and -0.4/-0.1 are both +4 -- so the quoted
    multiplier carries no trace of the reversal at all. That is why it
    outranks the single-component cell rather than being folded into it."""
    out = rr.range_coherence_diff("both reverse", [
        _pt(0.30, _lg(+0.40, 0.30), _lg(+0.10, 0.30)),
        _pt(0.70, _lg(-0.40, 0.70), _lg(-0.10, 0.70)),
    ])
    assert out["outcome"] == "INCOHERENT_BOTH_COMPONENTS_CHANGE_SIGN"
    # TERM_ORDERING_CHANGES fires too, and it is not noise: on a shared scale
    # a joint reversal drags `num - den` through zero as well. THE EXPECTATION
    # WAS WRITTEN DOWN WITHOUT IT AND THE CODE WAS RIGHT -- which is the case
    # for writing expectations down rather than reading them off the module.
    # Precedence is what earns its keep here: the sign cell is the diagnosis a
    # reader needs, and the ordering code stays visible in `failure_codes`.
    assert sorted(out["failure_codes"]) == ["DENOMINATOR_CHANGES_SIGN",
                                            "NUMERATOR_CHANGES_SIGN",
                                            "TERM_ORDERING_CHANGES"]
    # The ratio is sign-stable at +4 throughout, which is the whole hazard.
    assert out["grid_ratio_min"] == pytest.approx(4.0)
    assert out["grid_ratio_max"] == pytest.approx(4.0)
    # And whether the two reversals share an interval is REPORTED, not scored.
    assert out["sign_change_intervals"]["numerator"] == [(0.30, 0.70)]
    assert out["sign_change_intervals"]["denominator"] == [(0.30, 0.70)]


def test_a_component_that_is_exactly_zero_at_a_grid_point_has_its_own_cell():
    """Distinct from a sign change: this component touches zero and returns."""
    out = rr.range_coherence_diff("touches zero", [
        _pt(0.30, _lg(+0.40, 0.30), _lg(+0.10, 0.30)),
        _pt(0.50, _lg(+0.40, 0.50), _lg(0.0, 0.50)),
        _pt(0.70, _lg(+0.40, 0.70), _lg(+0.10, 0.70)),
    ])
    assert out["outcome"] == "INCOHERENT_A_COMPONENT_IS_ZERO_AT_A_GRID_POINT"
    assert "DENOMINATOR_IS_ZERO_AT_A_GRID_POINT" in out["failure_codes"]
    assert out["sign_change_intervals"]["denominator"] == []


def test_the_term_ordering_flipping_is_INCOHERENT_AT13s_ruling():
    """No component reverses; the ratio crosses 1 and which term dominates
    flips. AT13 ruled on this in prose and nothing enforced it."""
    out = rr.range_coherence_diff("ordering flips", [
        _pt(0.30, _lg(+0.40, 0.30), _lg(+0.10, 0.30)),
        _pt(0.70, _lg(+0.05, 0.70), _lg(+0.20, 0.70)),
    ])
    assert out["outcome"] == "INCOHERENT_THE_TERM_ORDERING_CHANGES"
    assert out["ordering_change_intervals"] == [(0.30, 0.70)]
    assert out["sign_change_intervals"] == {"numerator": [],
                                            "denominator": []}


def test_a_non_finite_value_at_a_grid_point_is_NOT_EVALUABLE():
    out = rr.range_coherence_diff("gap", [
        _pt(0.30, _lg(+0.40, 0.30), _lg(+0.10, 0.30)),
        _pt(0.70, _lg(+0.38, 0.70),
            rr.component(None, "logits", at=0.70, magnitude_of=rr.SIGNED)),
    ])
    assert out["outcome"] == \
        "NOT_EVALUABLE_COHERENCE_IS_UNDETERMINED_ON_THIS_RANGE"
    assert "GRID_POINT_VALUE_ABSENT_OR_NON_FINITE" in out["failure_codes"]


def test_a_measured_incoherence_OUTRANKS_a_gap_elsewhere_in_the_span():
    """Ignorance ranked above a known defect would let the ambient state of a
    prose record mask the thing an author can fix today -- the same ordering
    argument OUTCOMES makes for the per-point rule."""
    out = rr.range_coherence_diff("gap and reversal", [
        _pt(0.30, _lg(+0.40, 0.30), _lg(+0.10, 0.30)),
        _pt(0.50, _lg(+0.40, 0.50),
            rr.component(None, "logits", at=0.50, magnitude_of=rr.SIGNED)),
        _pt(0.70, _lg(-0.40, 0.70), _lg(+0.10, 0.70)),
    ])
    assert out["outcome"] == "INCOHERENT_ONE_COMPONENT_CHANGES_SIGN"
    assert "GRID_POINT_VALUE_ABSENT_OR_NON_FINITE" in out["failure_codes"]
    assert out["coherent_over_evaluable_points"] is False


# ------------------------------------------- DESIGN QUESTION 2: quoted vs grid

def test_a_reversal_OUTSIDE_the_quoted_span_is_REPORTED_and_does_not_score():
    """The rule evaluates the QUOTED span. A quote that stops one grid point
    short of a reversal is still COHERENT -- and the reader is told, because a
    span chosen just short of a reversal is a selection (AB1 decision A)."""
    grid = [
        _pt(0.30, _lg(+0.40, 0.30), _lg(+0.10, 0.30)),
        _pt(0.50, _lg(+0.42, 0.50), _lg(+0.09, 0.50)),
        _pt(0.80, _lg(+0.38, 0.80), _lg(-0.08, 0.80)),
        _pt(0.90, _lg(+0.37, 0.90), _lg(-0.12, 0.90)),
    ]
    out = rr.range_coherence_diff("quoted low half", grid, quoted=(0.30, 0.50))
    assert out["outcome"] == "COHERENT"
    assert out["at_in_quoted_range"] == [0.30, 0.50]
    assert out["grid_context"]["n_points_outside"] == 2
    assert out["grid_context"]["at_outside"] == [0.80, 0.90]
    # Nothing reverses among the two OUTSIDE points themselves, so there is no
    # incoherence to name -- the fact reported is that two points were cut.
    assert out["grid_context"]["outside_incoherence"] is None


def test_grid_context_names_an_incoherence_among_the_excluded_points():
    grid = [
        _pt(0.30, _lg(+0.40, 0.30), _lg(+0.10, 0.30)),
        _pt(0.50, _lg(+0.42, 0.50), _lg(+0.09, 0.50)),
        _pt(0.80, _lg(+0.38, 0.80), _lg(+0.08, 0.80)),
        _pt(0.90, _lg(+0.37, 0.90), _lg(-0.12, 0.90)),
    ]
    out = rr.range_coherence_diff("quoted low half", grid, quoted=(0.30, 0.50))
    assert out["outcome"] == "COHERENT"
    assert out["grid_context"]["outside_incoherence"] == \
        "INCOHERENT_ONE_COMPONENT_CHANGES_SIGN"


def test_no_quoted_span_means_the_whole_grid_and_an_empty_context():
    out = rr.range_coherence_diff("whole grid", [
        _pt(0.30, _lg(+0.40, 0.30), _lg(+0.10, 0.30)),
        _pt(0.70, _lg(+0.38, 0.70), _lg(+0.08, 0.70)),
    ])
    assert out["grid_context"] == {"n_points_outside": 0,
                                   "outside_incoherence": None}


# ---------------------------------------------- the stated range, sub-ranges

def test_a_stated_range_disagreeing_with_the_grid_does_NOT_move_the_outcome():
    """Mirrors stated_multiplier: the outcome is about the rule, and an
    arithmetic slip is a separate fact a reader needs separately."""
    out = rr.range_coherence_diff("sound but misquoted", [
        _pt(0.30, _lg(+0.40, 0.30), _lg(+0.10, 0.30)),
        _pt(0.70, _lg(+0.38, 0.70), _lg(+0.08, 0.70)),
    ], stated_range=(4.0, 9.0))
    assert out["outcome"] == "COHERENT"
    assert out["failure_codes"] == ["STATED_RANGE_DISAGREES_WITH_ITS_GRID"]


def test_sub_ranges_are_split_at_every_incoherence_boundary():
    out = rr.range_coherence_diff("two reversals", [
        _pt(0.30, _lg(+0.40, 0.30), _lg(+0.10, 0.30)),
        _pt(0.50, _lg(+0.40, 0.50), _lg(-0.10, 0.50)),
        _pt(0.70, _lg(+0.40, 0.70), _lg(+0.10, 0.70)),
    ])
    assert [(s["at_from"], s["at_to"], s["n_points"])
            for s in out["coherent_sub_ranges"]] == [
        (0.30, 0.30, 1), (0.50, 0.50, 1), (0.70, 0.70, 1)]


def test_the_formatter_has_a_TOTAL_branch_and_it_names_itself_as_the_defect():
    """gotcha 32c: a completed measurement must never be reported as a failed
    job because a formatter had no branch. Called directly with a verdict that
    is not in the vocabulary, which is the only way to reach it."""
    s = rr.defensible_range_form("x", "A_CELL_THAT_DOES_NOT_EXIST", [], 1.0, 2.0)
    assert "has no formatter branch" in s
    assert "defect in" in s


def test_every_range_outcome_has_a_branch_that_is_not_the_fallback():
    """The other half, and the one that makes the total branch meaningful: no
    real cell may fall through to it. The eight names are written out."""
    for cell in ("NOT_A_RANGE_FEWER_THAN_TWO_GRID_POINTS",
                 "NOT_A_RANGE_A_GRID_POINT_HAS_NO_DEFINED_COMPARISON",
                 "INCOHERENT_BOTH_COMPONENTS_CHANGE_SIGN",
                 "INCOHERENT_ONE_COMPONENT_CHANGES_SIGN",
                 "INCOHERENT_A_COMPONENT_IS_ZERO_AT_A_GRID_POINT",
                 "INCOHERENT_THE_TERM_ORDERING_CHANGES",
                 "NOT_EVALUABLE_COHERENCE_IS_UNDETERMINED_ON_THIS_RANGE",
                 "COHERENT"):
        s = rr.defensible_range_form("x", cell, [], 1.0, 2.0)
        assert "has no formatter branch" not in s, cell

