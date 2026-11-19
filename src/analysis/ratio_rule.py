"""CLAUDE.md gotcha 33, implemented as a check rather than restated as a rule.

    *A ratio requires both a shared scale and a denominator whose interval
    excludes zero. Otherwise report the difference.*

**WHY THIS IS CODE AND NOT A REMINDER.** Five of the eighteen imprecise claims
in `docs/RETROSPECTIVE_VERIFICATION.md` are ONE defect -- a multiplier quoted
without the quantity it divides by -- and **the record had already ruled on
three of those exact multipliers** (AC14 on the 2.2x CV ratio, AT13's P6 on the
12.6x, BC8a consolidating five at once). The author could recite the rule and
broke it five times in one document. A more careful reader is not the fix.

**THIS EXTENDS THE TWO EXISTING BRANCHES RATHER THAN DUPLICATING THEM.**
`fitted_vs_observed.span_comparison_verdict` carries
`UNINFORMATIVE_OBSERVED_SPAN_STRADDLES_ZERO` and
`carrier_support.excess_comparison` carries
`UNDEFINED_SINGLE_EXCESS_IS_ZERO_OR_NON_FINITE`; both report the DIFFERENCE in
the ratio's absence. Those two are domain-specific -- one takes span dicts, the
other excess dicts -- so the general rule cannot be expressed by calling
either. What is shared is the VOCABULARY and the remedy, and
`tests/test_ratio_rule.py` asserts that this module's denominator branches
agree with both of theirs on the conditions they overlap on. **If either of
those functions changes its branch, that test fails here.**

**WHAT IT IS NOT.** Not a gate, not a standing audit, not a checklist. It takes
a ratio's two components and returns a DIFF -- the offending ratio, what is
missing, and the defensible form. `main()` reports and always exits 0, which is
deliberate and is stated at the call site: gotcha 13 extension 1 is about a
validator whose failure cannot reach the thing gating on it, and nothing gates
on this.

**THE SHARED-SCALE HALF IS THE ONE THAT HIDES** (BC8a). A ratio against a
vanishing denominator is loud -- 6.787, a correction factor that was an
artifact. A ratio between a standardised coefficient and a raw-scale one is
always finite and always looks meaningful; AS section 4's "19x" is that, and it
"would reach a methods section unchallenged". So `scale` is a required field on
both components and a mismatch is the most severe outcome this returns.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

# --------------------------------------------------------------- vocabulary

#: The five outcomes, in precedence order. TOTAL over the inputs: every
#: combination of the two components lands in exactly one of these.
#:
#: **`UNVERIFIABLE_...` IS IGNORANCE, NOT A MEASURED FAILURE**, and it is
#: separated from `REPORT_THE_DIFFERENCE_INSTEAD` for the reason BC14 separated
#: `NOT_ESTABLISHED` from a measured device class: collapsing the two would
#: report "nobody measured an interval" in the same cell as "the interval was
#: measured and covers zero", and a reader cannot act on the first the way they
#: must act on the second. It is also the cell most prose multipliers land in,
#: so merging it would swamp the findings that are findings.
#: **AND IT SITS BELOW THE SETTINGS CELL, NOT ABOVE IT.** A mismatch of
#: parameter settings is MEASURED and its remedy is one sentence; an absent
#: interval is a gap and its remedy is a run. Ordering ignorance above a known
#: defect would let the ambient state of a prose record -- where almost nothing
#: carries an interval -- mask the one thing an author can fix today.
#:
#: **WIDENED AT AMENDMENT BE: the cell now covers EITHER component.** The rule
#: as first drafted constrained only the denominator, so a ratio of two small
#: numbers whose NUMERATOR carried no interval passed -- exactly the shape AC14
#: ruled on. The cell is RENAMED rather than a second one added, because a key
#: whose name says `THE_DENOMINATOR` while it fires for the numerator is the
#: one-key-two-meanings defect gotcha 16 committed a convention against.
OUTCOMES = (
    "NEITHER_RATIO_NOR_DIFFERENCE_IS_DEFINED",
    "REPORT_THE_DIFFERENCE_INSTEAD",
    "DEFENSIBLE_ONLY_WITH_THE_SETTINGS_NAMED",
    "UNVERIFIABLE_A_COMPONENT_HAS_NO_MEASURED_INTERVAL",
    "DEFENSIBLE",
)


class _Exact:
    """Sentinel for a component that is exactly known BY CONSTRUCTION.

    **WHY THIS IS NECESSARY RATHER THAN CONVENIENT (amendment BE).** The
    symmetric rule says a ratio needs an interval on both components. A
    declared threshold, an integer count and a fixed grid value have none and
    need none -- they are not estimates. Without this sentinel `0.70 / 0.1449`
    would be reported as unverifiable because nobody put a confidence interval
    on a pre-registered constant, which is true, vacuous, and recreates the
    swamping the `UNVERIFIABLE` cell was split out to avoid.

    **AND IT IS A CLAIM, NOT A FACT ABOUT THE TYPE.** Marking a component
    `EXACT` asserts that it is not an estimate; nothing checks that. It has the
    same standing as an entry in `config_audit`'s `UNGOVERNED` list (gotcha
    25c) -- **it must be re-read whenever what the quantity measures changes**,
    and it is echoed in the diff (`numerator_exact` / `denominator_exact`) so
    the assertion is visible to a reader rather than buried in a call site.
    """

    def __repr__(self):  # pragma: no cover - debugging aid only
        return "EXACT"


#: Pass as `ci=EXACT` for a quantity that is exact by construction.
EXACT = _Exact()

#: Failure codes. Each is a fact about the two components, never a judgement.
FAILURES = (
    "SCALES_DIFFER",
    "SCALE_ABSENT",
    "DENOMINATOR_IS_ZERO_OR_NON_FINITE",
    "DENOMINATOR_INTERVAL_STRADDLES_ZERO",
    "DENOMINATOR_INTERVAL_ABSENT",
    "NUMERATOR_INTERVAL_ABSENT",
    "SETTINGS_DIFFER",
    "SETTING_ABSENT",
    "STATED_MULTIPLIER_DISAGREES_WITH_ITS_COMPONENTS",
)


def component(value, scale, at=None, ci=None, label=None,
              magnitude_of=None) -> dict:
    """One side of a ratio.

    `scale` is the UNIT AND AXIS the number lives on -- "pp of r_f per SD",
    "logit coefficient", "cross-seed CV ratio". `at` is the PARAMETER SETTING
    it was measured at -- a threshold, a comparison count, a grid point. `ci`
    is a two-element interval or None.

    **`at=None` and `ci=None` mean ABSENT, and absent is reported as absent,
    never as agreement** (gotcha 15). That is the difference between this and a
    check that silently passes whatever it was not told. **`ci=EXACT` is a
    third state** -- a claim that the quantity is not an estimate, so an
    interval is not missing but inapplicable.

    **`magnitude_of` IS READ ONLY BY THE RANGE RULE (amendment BL)** and is
    inert here: it names the SIGNED quantity whose magnitude this component
    reports, for a component that is a dispersion or an absolute value. It is
    additive -- every existing call site omits it and is unaffected. See the
    range-rule header for why a sign check read off an SD can never fire.
    """
    if ci is EXACT:
        stored = EXACT
    elif ci is None:
        stored = None
    else:
        stored = [ci[0], ci[1]]
    return {"label": label, "value": value, "scale": scale,
            "at": at, "ci": stored, "magnitude_of": magnitude_of}


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _interval_of(comp):
    """The comparable interval of a component, or None if it has none.

    An `EXACT` component's interval is the point itself, so a non-zero exact
    value does not straddle zero and an exact zero does -- the same answer the
    zero-value branch reaches independently, which is the check on this
    substitution rather than an accident of it.
    """
    ci = comp.get("ci")
    if ci is EXACT:
        v = comp.get("value")
        return [v, v] if _finite(v) else None
    return ci


def _straddles_zero(ci) -> bool:
    lo, hi = float(ci[0]), float(ci[1])
    lo, hi = min(lo, hi), max(lo, hi)
    return lo <= 0.0 <= hi


def ratio_rule_diff(name: str, numerator: dict, denominator: dict,
                    stated_multiplier=None, rel_tol: float = 0.02) -> dict:
    """The check. Returns a DIFF, never a bare verdict.

    **TOTAL OVER THE INPUTS** (gotcha 24): every combination of the two
    components reaches exactly one `outcome`, including the combinations where
    the ratio does not exist and the ones where the DIFFERENCE does not exist
    either.

    **TOTAL OVER THE CAUSES OF EACH BRANCH** (gotcha 30), which is the part
    that needed writing down:

      * `NEITHER_RATIO_NOR_DIFFERENCE_IS_DEFINED` fires when the two
        quantities are not on one scale. **The usual remedy -- report the
        difference -- has a precondition, and this is the branch where it
        fails**: subtracting a raw-scale coefficient from a standardised one is
        no more meaningful than dividing them. The only defensible form is to
        report each quantity with its own scale and make no comparison.
      * `REPORT_THE_DIFFERENCE_INSTEAD` fires when the scales agree and the
        denominator was MEASURED and cannot carry a ratio -- it is zero, it is
        non-finite, or its interval covers zero.
      * `UNVERIFIABLE_A_COMPONENT_HAS_NO_MEASURED_INTERVAL` fires when no
        interval was ever recorded for EITHER component -- widened from the
        denominator alone at amendment BE, and this bullet still named the
        pre-BE key and the pre-BE scope until amendment BL corrected it. **This
        is a statement
        of ignorance, not of failure**, and it is deliberately NOT the same
        cell as the branch above: "excludes zero" is unestablished rather than
        refuted, and the remedy is to measure an interval rather than to
        withdraw the ratio. Merging the two would also swamp every real finding
        in a prose record, where most multipliers carry no interval at all.
      * `DEFENSIBLE_ONLY_WITH_THE_SETTINGS_NAMED` fires when the scales agree
        and the denominator is sound, but the two sides were measured at
        different parameter settings, or a setting is missing on one side. This
        is AT13's case exactly: +3.12 pp at `theta = 0.70` over a bound
        attained at `theta = 0.60`. The ratio is arithmetic; what it is not is
        a property of the two terms.
      * `DEFENSIBLE` fires when none of the above holds. **It does not mean the
        ratio is interesting or that the reasoning behind it is right** -- only
        that the two conditions gotcha 33 names are met.

    `stated_multiplier`, when supplied, is checked against the recomputed
    ratio. A disagreement is reported as a failure and does NOT move the
    outcome: the outcome is about the rule, and an arithmetic slip is a
    separate fact that a reader needs separately.
    """
    fails = []

    ns, ds = numerator.get("scale"), denominator.get("scale")
    if ns is None or ds is None:
        fails.append({
            "code": "SCALE_ABSENT",
            "detail": f"numerator scale {ns!r}, denominator scale {ds!r}",
            "remedy": "state the unit and axis of both sides before comparing",
        })
    elif ns != ds:
        fails.append({
            "code": "SCALES_DIFFER",
            "detail": f"numerator is {ns!r}, denominator is {ds!r}",
            "remedy": "report each quantity on its own scale and make no "
                      "ratio; the difference is not defined either",
        })

    dv = denominator.get("value")
    if not _finite(dv) or float(dv) == 0.0:
        fails.append({
            "code": "DENOMINATOR_IS_ZERO_OR_NON_FINITE",
            "detail": f"denominator value {dv!r}",
            "remedy": "the ratio does not exist; report the numerator alone",
        })
    nci = _interval_of(numerator)
    if nci is None:
        fails.append({
            "code": "NUMERATOR_INTERVAL_ABSENT",
            "detail": f"no interval recorded for {numerator.get('label')!r}",
            "remedy": "the rule is symmetric (amendment BE): a ratio of two "
                      "small numbers is not rescued by an interval on only one "
                      "of them. Measure one, or carry the difference",
        })
    dci = _interval_of(denominator)
    if dci is None:
        fails.append({
            "code": "DENOMINATOR_INTERVAL_ABSENT",
            "detail": f"no interval recorded for {denominator.get('label')!r}",
            "remedy": "an absent interval is not an interval excluding zero; "
                      "report the difference until one is measured",
        })
    elif _straddles_zero(dci):
        fails.append({
            "code": "DENOMINATOR_INTERVAL_STRADDLES_ZERO",
            "detail": f"denominator 95% interval {list(dci)} covers zero",
            "remedy": "a ratio against this denominator ranks noise "
                      "(gotcha 25b); report the difference",
        })

    na, da = numerator.get("at"), denominator.get("at")
    if na is None or da is None:
        fails.append({
            "code": "SETTING_ABSENT",
            "detail": f"numerator at {na!r}, denominator at {da!r}",
            "remedy": "record the parameter setting of both sides; an "
                      "unrecorded setting cannot be asserted to match",
        })
    elif na != da:
        fails.append({
            "code": "SETTINGS_DIFFER",
            "detail": f"numerator at {na!r}, denominator at {da!r}",
            "remedy": f"quote the ratio only with both settings named "
                      f"({na} over {da}), or recompute both at one setting",
        })

    nv = numerator.get("value")
    ratio = None
    if _finite(nv) and _finite(dv) and float(dv) != 0.0:
        ratio = float(nv) / float(dv)
    difference = None
    scales_ok = (ns is not None and ds is not None and ns == ds)
    if scales_ok and _finite(nv) and _finite(dv):
        difference = float(nv) - float(dv)

    if stated_multiplier is not None:
        if ratio is None or not _finite(stated_multiplier):
            fails.append({
                "code": "STATED_MULTIPLIER_DISAGREES_WITH_ITS_COMPONENTS",
                "detail": f"stated {stated_multiplier!r}, recomputed {ratio!r}",
                "remedy": "the stated multiplier cannot be reproduced from the "
                          "components given",
            })
        elif abs(float(stated_multiplier) - ratio) > rel_tol * abs(ratio):
            fails.append({
                "code": "STATED_MULTIPLIER_DISAGREES_WITH_ITS_COMPONENTS",
                "detail": (f"stated {float(stated_multiplier):g}, recomputed "
                           f"{ratio:.4g} from the components"),
                "remedy": "restate the multiplier from its own components",
            })

    codes = {f["code"] for f in fails}
    if codes & {"SCALES_DIFFER", "SCALE_ABSENT"}:
        outcome = "NEITHER_RATIO_NOR_DIFFERENCE_IS_DEFINED"
    elif codes & {"DENOMINATOR_IS_ZERO_OR_NON_FINITE",
                  "DENOMINATOR_INTERVAL_STRADDLES_ZERO"}:
        outcome = "REPORT_THE_DIFFERENCE_INSTEAD"
    elif codes & {"SETTINGS_DIFFER", "SETTING_ABSENT"}:
        outcome = "DEFENSIBLE_ONLY_WITH_THE_SETTINGS_NAMED"
    elif codes & {"DENOMINATOR_INTERVAL_ABSENT", "NUMERATOR_INTERVAL_ABSENT"}:
        outcome = "UNVERIFIABLE_A_COMPONENT_HAS_NO_MEASURED_INTERVAL"
    else:
        outcome = "DEFENSIBLE"

    return {
        "name": name,
        "outcome": outcome,
        "failures": fails,
        "failure_codes": sorted(codes),
        "ratio": ratio,
        "difference": difference,
        "stated_multiplier": stated_multiplier,
        "numerator": numerator,
        "denominator": denominator,
        # The EXACT claims, echoed so a reader sees them (see `_Exact`).
        "numerator_exact": numerator.get("ci") is EXACT,
        "denominator_exact": denominator.get("ci") is EXACT,
        "defensible_form": defensible_form(name, outcome, numerator,
                                           denominator, ratio, difference),
    }


def defensible_form(name, outcome, numerator, denominator, ratio,
                    difference) -> str:
    """The sentence the record may carry, DERIVED from the components.

    Every word that varies is computed from the data it describes (gotcha 20).
    Nothing here asserts what the author expected to find.
    """
    nl = numerator.get("label") or "the numerator"
    dl = denominator.get("label") or "the denominator"
    if outcome == "NEITHER_RATIO_NOR_DIFFERENCE_IS_DEFINED":
        return (f"{nl} is {numerator.get('value')!r} "
                f"({numerator.get('scale')!r}) and {dl} is "
                f"{denominator.get('value')!r} ({denominator.get('scale')!r}). "
                "Report each on its own scale; neither a ratio nor a "
                "difference between them is defined.")
    if outcome == "REPORT_THE_DIFFERENCE_INSTEAD":
        if difference is None:
            return (f"{nl} minus {dl} is not computable from the values given; "
                    f"report {nl} alone.")
        return (f"{nl} exceeds {dl} by {difference:+.4g} "
                f"({numerator.get('scale')}). The ratio is withheld: "
                + _denominator_reason(denominator))
    if outcome == "UNVERIFIABLE_A_COMPONENT_HAS_NO_MEASURED_INTERVAL":
        missing = [lab for lab, comp in ((nl, numerator), (dl, denominator))
                   if _interval_of(comp) is None]
        return (f"{nl} is {ratio:.4g}x {dl} on the {numerator.get('scale')} "
                f"scale, but no interval has been measured for "
                f"{' and '.join(missing)}, so the ratio is unestablished "
                f"rather than refuted. The difference is {difference:+.4g}. "
                "Measure the interval(s) or carry the difference.")
    if outcome == "DEFENSIBLE_ONLY_WITH_THE_SETTINGS_NAMED":
        return (f"{nl} is {ratio:.4g}x {dl}, with {nl} measured at "
                f"{numerator.get('at')!r} and {dl} at "
                f"{denominator.get('at')!r}. The multiplier must not travel "
                "without both settings.")
    return (f"{nl} is {ratio:.4g}x {dl}, both on the {numerator.get('scale')} "
            f"scale at {numerator.get('at')!r}, with {dl}'s interval "
            f"{denominator.get('ci')} excluding zero.")


def _denominator_reason(denominator) -> str:
    dv = denominator.get("value")
    if not _finite(dv) or float(dv) == 0.0:
        return "the denominator is zero or non-finite."
    if denominator.get("ci") is None:
        return "the denominator has no measured interval, so 'excludes zero' " \
               "is unestablished rather than satisfied."
    return "the denominator's interval covers zero, so a ratio against it " \
           "ranks noise."


# ===================================================================
# THE RANGE-COHERENCE RULE (amendment BL). A SEPARATE ENTRY POINT.
# ===================================================================
#
# **THE GAP IT CLOSES.** `ratio_rule_diff` applies PER POINT. Nothing checked
# that a ratio quoted as a RANGE ACROSS A PARAMETER GRID -- "14x to 40x across
# the qualifying thresholds" -- is coherent across its own span. BK found the
# live instance: the alignment coefficient's interval excludes zero at every
# one of the 13 qualifying thetas, so both per-point conditions PASS at every
# point, and yet `b1_z` is positive to theta = 0.75 and negative from 0.80.
# **Every point is defensible and the range is not.** A range built over a sign
# reversal is not a range.
#
# **DESIGN QUESTION 1 -- WHY A SEPARATE ENTRY POINT AND NOT A SIXTH BRANCH.**
# Answered rather than assumed, because the brief was right that adding an
# outcome changes a documented contract:
#
#   * `ratio_rule_diff`'s docstring declares itself TOTAL over its inputs and
#     `test_the_outcome_vocabulary_is_exactly_these_five` pins the tuple. A
#     sixth member changes what every stored verdict from BJ's and BK's screens
#     means, retroactively, with nothing marking the change in the artifacts.
#   * The INPUT is a different kind. A range rule needs two SEQUENCES indexed
#     by grid point; "the sign changes between 0.75 and 0.80" cannot be
#     expressed with two scalars. Widening `component` to carry a whole grid
#     would make every existing call site pass a one-element grid.
#   * So this COMPOSES rather than replaces: `range_coherence_diff` routes
#     every grid point through `ratio_rule_diff` and carries the per-point
#     results in its diff. The per-point rule keeps its five outcomes; the
#     range rule has its own eight. Two vocabularies, each total over its own
#     input, is the shape gotcha 16's one-key-one-meaning convention asks for.
#
# **DESIGN QUESTION 2 -- QUOTED RANGE OR FULL GRID.** The rule evaluates the
# **QUOTED** range and ALWAYS reports the full grid beside it, and this is
# stated here rather than left to a call site. The defect being caught is a
# quoted multiplier that misrepresents itself, and a sign reversal at a grid
# point nobody quoted does not make the quote wrong. **But it is not nothing
# either**: a quoted range that stops one grid point short of a reversal is a
# SELECTION, which is what AB1 decision A's reporting standard exists to catch.
# So `grid_context` names every incoherence outside the quoted span, is never
# empty when one exists, and NEVER moves the outcome.
#
# **WHY THIS IS ADOPTED AS A RULE, AND THE REASONING DOES NOT USE THE COUNT.**
# BJ's straddles-zero numerator screen found zero instances and correctly chose
# GUIDANCE; BK then exhibited one instance of this condition. **That is not the
# argument, and it must not be read as the argument** -- a rule adopted because
# the screen happened to catch something is a rule whose adoption is a function
# of when the screen ran. The argument is a category difference. BJ ruled its
# case a REPORTING condition because the ratio stays arithmetically
# well-defined and a caveat travelling with it repairs the quote. **A range
# spanning a sign reversal has no such repair.** Its two endpoints describe
# comparisons of opposite character -- at one end the terms point the same way,
# at the other they point in opposite directions -- so the interval between
# them has no referent, and no sentence appended to it supplies one. The only
# defensible form is to NAME THE SUB-RANGES. A condition that cannot be
# repaired by annotation is a well-definedness condition, and those are rules.
#
# **AND THE CHECK HAD TO BE BUILT SO IT CAN FIRE ON ITS OWN FOUNDING INSTANCE**
# (gotcha 13). It nearly could not. BK's `14x-40x` divides two **standard
# deviations**, and an SD is non-negative by construction -- so a sign check
# read off the component VALUES can never fire on the very case that motivated
# it, while returning `COHERENT` and looking like it worked. That is gotcha
# 21's shape exactly: the reduction that forms the component (|.| -> SD) ERASES
# the sign the rule is looking for. Hence `magnitude_of`: a component may
# declare the SIGNED quantity whose magnitude it reports, and a component that
# is non-negative everywhere in the range while declaring nothing is reported
# as `SIGN_BASIS_UNDECLARED_FOR_A_NON_NEGATIVE_COMPONENT` -- a statement of
# ignorance that lands in the NOT_EVALUABLE cell, never in COHERENT.

class _Signed:
    """Sentinel: this component's own value IS the signed quantity.

    Pass as `magnitude_of=SIGNED` for a component that is genuinely signed and
    merely happens to be non-negative everywhere on the grid quoted. It is a
    CLAIM, with the same standing as `EXACT` (gotcha 25c): nothing checks it,
    and it must be re-read whenever what the quantity measures changes.
    """

    def __repr__(self):  # pragma: no cover - debugging aid only
        return "SIGNED"


SIGNED = _Signed()

#: The eight range-level outcomes, in PRECEDENCE ORDER, most severe first.
#: TOTAL over the inputs: every grid lands in exactly one.
#:
#: **WHY `BOTH_COMPONENTS_CHANGE_SIGN` OUTRANKS `ONE_COMPONENT_CHANGES_SIGN`,
#: which is the cell the brief required to be decided rather than defaulted.**
#: It is INCOHERENT, and it is the MORE severe of the two. When both components
#: reverse, the ratio's own sign is preserved, so the quoted multiplier carries
#: no trace at all of what happened -- whereas a single reversal shows up in
#: the ratio's sign and a reader can see it. **The more hidden defect ranks
#: higher**, which is the same ordering argument gotcha 27 makes for a
#: permutation that preserves a total. Two further reasons it is not coherent:
#: at a reversal both components pass through or near zero, so the ratio is a
#: quotient of two vanishing quantities -- the loud failure this module's own
#: header calls out; and on a shared scale the two terms point together at one
#: end of the range and oppositely at the other, so "N times" names two
#: different physical situations. Whether the two reversals fall in the SAME
#: grid interval is reported in `sign_change_intervals` -- it is a fact a
#: reader needs and it does not change the cell.
#:
#: **WHY `NOT_EVALUABLE` SITS BELOW EVERY MEASURED INCOHERENCE AND ABOVE
#: `COHERENT`.** Below, for the reason `OUTCOMES` gives for the same ordering:
#: ignorance ranked above a known defect lets the ambient state of a prose
#: record -- where almost nothing is tabulated per grid point -- mask the thing
#: an author can fix today. Above `COHERENT`, because coherence over the
#: evaluable subset is NOT coherence over the range; the diff still reports
#: `coherent_over_evaluable_points` so nothing measured is thrown away.
RANGE_OUTCOMES = (
    "NOT_A_RANGE_FEWER_THAN_TWO_GRID_POINTS",
    "NOT_A_RANGE_A_GRID_POINT_HAS_NO_DEFINED_COMPARISON",
    "INCOHERENT_BOTH_COMPONENTS_CHANGE_SIGN",
    "INCOHERENT_ONE_COMPONENT_CHANGES_SIGN",
    "INCOHERENT_A_COMPONENT_IS_ZERO_AT_A_GRID_POINT",
    "INCOHERENT_THE_TERM_ORDERING_CHANGES",
    "NOT_EVALUABLE_COHERENCE_IS_UNDETERMINED_ON_THIS_RANGE",
    "COHERENT",
)

#: Range-level failure codes. Each is a fact about the grid, never a judgement.
RANGE_FAILURES = (
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
)


def grid_point(at, numerator: dict, denominator: dict) -> dict:
    """One point of a parameter grid: the setting, and the two components."""
    return {"at": at, "numerator": numerator, "denominator": denominator}


def _sign(x) -> int:
    if not _finite(x):
        return 0
    v = float(x)
    return (v > 0) - (v < 0)


def _sign_basis(comp):
    """The SIGNED quantity a component's sign should be read off.

    Returns `(value, declared)`. `declared` is False when the component made no
    claim and the value is being used as its own sign basis -- which is only
    safe when the component actually takes both signs somewhere on the grid.
    """
    m = comp.get("magnitude_of")
    if m is SIGNED:
        return comp.get("value"), True
    if m is not None:
        return m, True
    return comp.get("value"), False


def range_coherence_diff(name: str, points, quoted=None,
                         stated_range=None, rel_tol: float = 0.02) -> dict:
    """Is a ratio quoted as a RANGE across a parameter grid coherent?

    `points` is a sequence of `grid_point`s, in grid order. `quoted` is an
    inclusive `(lo, hi)` pair of `at` values naming the span the multiplier was
    quoted over; `None` means the whole grid was quoted. `stated_range` is an
    optional `(lo, hi)` of the multiplier as written, checked against the grid.

    **TOTAL OVER THE INPUTS** (gotcha 24) and **TOTAL OVER THE CAUSES OF EACH
    BRANCH** (gotcha 30). The cells, in precedence order:

      * `NOT_A_RANGE_FEWER_THAN_TWO_GRID_POINTS` -- zero or one point falls in
        the quoted span. **A single point is not a range**; route it through
        `ratio_rule_diff`, which is the whole rule for it. Zero points means
        the quoted span names no grid point at all, which is a locator defect
        in the quote and is reported as such rather than as coherence.
      * `NOT_A_RANGE_A_GRID_POINT_HAS_NO_DEFINED_COMPARISON` -- some point in
        the span returns `NEITHER_RATIO_NOR_DIFFERENCE_IS_DEFINED` per point
        (scales differ or a scale is absent). Coherence of a range of
        comparisons is not a meaningful question when a member is not a
        comparison, so this precedes every coherence cell.
      * `INCOHERENT_BOTH_COMPONENTS_CHANGE_SIGN` -- see `RANGE_OUTCOMES`.
      * `INCOHERENT_ONE_COMPONENT_CHANGES_SIGN` -- BK's live instance.
      * `INCOHERENT_A_COMPONENT_IS_ZERO_AT_A_GRID_POINT` -- a component is
        exactly zero somewhere in the span. **Distinct from a sign change**: a
        quantity can touch zero and return without reversing, and the ratio is
        undefined (denominator) or identically zero (numerator) at that point
        either way. It ranks BELOW the sign cells because when a zero IS a
        reversal both codes fire and the reversal is the more informative
        diagnosis.
      * `INCOHERENT_THE_TERM_ORDERING_CHANGES` -- no component reverses, but
        `sign(numerator - denominator)` does: the ratio crosses 1 and which
        term dominates flips inside the span. This is AT13's ruling, which had
        lived in prose only.
      * `NOT_EVALUABLE_COHERENCE_IS_UNDETERMINED_ON_THIS_RANGE` -- a point in
        the span has an absent or non-finite component value, or a component is
        non-negative throughout while declaring no `magnitude_of`. **A
        STATEMENT OF IGNORANCE AND NEVER A COHERENCE PASS.**
      * `COHERENT` -- two or more points, every comparison defined, no sign
        change, no zero, ordering stable. **It does not mean the multiplier is
        worth quoting** -- each point must still clear `ratio_rule_diff`, whose
        verdicts are carried in `per_point`.

    `stated_range` disagreeing with the grid is reported as a failure and does
    NOT move the outcome, for the reason `stated_multiplier` does not: the
    outcome is about the rule and an arithmetic slip is a separate fact.
    """
    fails = []
    pts = list(points)

    # SELECTED BY INDEX, NOT BY VALUE. `[p for p in pts if p not in in_range]`
    # would drop a second grid point that happened to carry identical
    # components -- a set-membership test standing in for a positional one,
    # which is gotcha 27's shape in miniature.
    if quoted is None:
        keep = list(range(len(pts)))
    else:
        lo, hi = quoted
        keep = [i for i, p in enumerate(pts) if lo <= p["at"] <= hi]
    kept = set(keep)
    in_range = [pts[i] for i in keep]
    out_of_range = [p for i, p in enumerate(pts) if i not in kept]

    per_point = [
        ratio_rule_diff(f"{name} @ {p['at']!r}", p["numerator"],
                        p["denominator"])
        for p in in_range
    ]

    if len(in_range) < 2:
        fails.append({
            "code": "FEWER_THAN_TWO_GRID_POINTS",
            "detail": f"{len(in_range)} grid point(s) fall in quoted span "
                      f"{quoted!r} of {len(pts)} on the grid",
            "remedy": "a single point is not a range -- quote it with its own "
                      "setting and route it through `ratio_rule_diff`; zero "
                      "points means the quoted span names no grid point",
        })

    undefined_at = [p["at"] for p, d in zip(in_range, per_point)
                    if d["outcome"] == "NEITHER_RATIO_NOR_DIFFERENCE_IS_DEFINED"]
    if undefined_at:
        fails.append({
            "code": "POINT_COMPARISON_UNDEFINED",
            "detail": f"no comparison is defined at {undefined_at!r}",
            "remedy": "the scales must agree at every point before the range "
                      "can be asked whether it is coherent",
        })

    # ------------------------------------------------ per-side sign analysis
    sign_change_intervals = {}
    for side in ("numerator", "denominator"):
        comps = [p[side] for p in in_range]
        bases, declared = [], []
        for c in comps:
            b, d = _sign_basis(c)
            bases.append(b)
            declared.append(d)
        if any(not _finite(b) for b in bases):
            fails.append({
                "code": "GRID_POINT_VALUE_ABSENT_OR_NON_FINITE",
                "detail": f"{side} has no finite value at "
                          f"{[p['at'] for p, b in zip(in_range, bases) if not _finite(b)]!r}",
                "remedy": "record the component at every grid point in the "
                          "span; an unrecorded point is not a coherent one",
            })
        signs = [_sign(b) for b in bases]
        finite_signs = [s for s, b in zip(signs, bases) if _finite(b)]
        if finite_signs and all(s >= 0 for s in finite_signs) \
                and not all(declared):
            fails.append({
                "code": "SIGN_BASIS_UNDECLARED_FOR_A_NON_NEGATIVE_COMPONENT",
                "detail": f"{side} is non-negative at every point in the span "
                          f"and declares no `magnitude_of`",
                "remedy": "a magnitude or dispersion erases the sign this rule "
                          "looks for (gotcha 21): pass `magnitude_of=<the "
                          "signed quantity>`, or `magnitude_of=SIGNED` to claim "
                          "the value is itself signed",
            })
        zero_at = [p["at"] for p, b in zip(in_range, bases)
                   if _finite(b) and float(b) == 0.0]
        if zero_at:
            fails.append({
                "code": f"{side.upper()}_IS_ZERO_AT_A_GRID_POINT",
                "detail": f"{side} is exactly zero at {zero_at!r}",
                "remedy": "the ratio is undefined or identically zero there; "
                          "name the sub-ranges either side of it",
            })
        nz = [(p["at"], s) for p, s in zip(in_range, signs) if s != 0]
        changes = [(nz[i][0], nz[i + 1][0]) for i in range(len(nz) - 1)
                   if nz[i][1] != nz[i + 1][1]]
        sign_change_intervals[side] = changes
        if changes:
            fails.append({
                "code": f"{side.upper()}_CHANGES_SIGN",
                "detail": f"{side} reverses sign between {changes!r}",
                "remedy": "a magnitude ratio cannot express a sign reversal; "
                          "name the sub-ranges and quote a multiplier only "
                          "inside one of them",
            })

    # ------------------------------------------------ term-ordering analysis
    ordering_changes = []
    diffs = []
    for p in in_range:
        nv, dv = p["numerator"].get("value"), p["denominator"].get("value")
        diffs.append(float(nv) - float(dv)
                     if _finite(nv) and _finite(dv) else None)
    nzd = [(p["at"], _sign(d)) for p, d in zip(in_range, diffs)
           if d is not None and _sign(d) != 0]
    ordering_changes = [(nzd[i][0], nzd[i + 1][0]) for i in range(len(nzd) - 1)
                        if nzd[i][1] != nzd[i + 1][1]]
    if ordering_changes:
        fails.append({
            "code": "TERM_ORDERING_CHANGES",
            "detail": f"which term dominates flips between {ordering_changes!r}",
            "remedy": "the ratio crosses 1 inside the span (AT13); name the "
                      "sub-ranges rather than extending the multiplier",
        })

    # ------------------------------------------------ the grid, and the quote
    ratios = [d["ratio"] for d in per_point if d["ratio"] is not None]
    grid_min = min(ratios) if ratios else None
    grid_max = max(ratios) if ratios else None
    if stated_range is not None:
        if grid_min is None or grid_max is None:
            fails.append({
                "code": "STATED_RANGE_DISAGREES_WITH_ITS_GRID",
                "detail": f"stated {stated_range!r}, no ratio is computable "
                          "from the grid",
                "remedy": "the stated range cannot be reproduced from the "
                          "points given",
            })
        else:
            slo, shi = min(stated_range), max(stated_range)
            if abs(slo - grid_min) > rel_tol * abs(grid_min) \
                    or abs(shi - grid_max) > rel_tol * abs(grid_max):
                fails.append({
                    "code": "STATED_RANGE_DISAGREES_WITH_ITS_GRID",
                    "detail": (f"stated [{slo:g}, {shi:g}], recomputed "
                               f"[{grid_min:.4g}, {grid_max:.4g}]"),
                    "remedy": "restate the range from its own grid points",
                })

    codes = {f["code"] for f in fails}
    if "FEWER_THAN_TWO_GRID_POINTS" in codes:
        outcome = "NOT_A_RANGE_FEWER_THAN_TWO_GRID_POINTS"
    elif "POINT_COMPARISON_UNDEFINED" in codes:
        outcome = "NOT_A_RANGE_A_GRID_POINT_HAS_NO_DEFINED_COMPARISON"
    elif {"NUMERATOR_CHANGES_SIGN", "DENOMINATOR_CHANGES_SIGN"} <= codes:
        outcome = "INCOHERENT_BOTH_COMPONENTS_CHANGE_SIGN"
    elif codes & {"NUMERATOR_CHANGES_SIGN", "DENOMINATOR_CHANGES_SIGN"}:
        outcome = "INCOHERENT_ONE_COMPONENT_CHANGES_SIGN"
    elif codes & {"NUMERATOR_IS_ZERO_AT_A_GRID_POINT",
                  "DENOMINATOR_IS_ZERO_AT_A_GRID_POINT"}:
        outcome = "INCOHERENT_A_COMPONENT_IS_ZERO_AT_A_GRID_POINT"
    elif "TERM_ORDERING_CHANGES" in codes:
        outcome = "INCOHERENT_THE_TERM_ORDERING_CHANGES"
    elif codes & {"GRID_POINT_VALUE_ABSENT_OR_NON_FINITE",
                  "SIGN_BASIS_UNDECLARED_FOR_A_NON_NEGATIVE_COMPONENT"}:
        outcome = "NOT_EVALUABLE_COHERENCE_IS_UNDETERMINED_ON_THIS_RANGE"
    else:
        outcome = "COHERENT"

    subs = coherent_sub_ranges(in_range, sign_change_intervals,
                               ordering_changes)
    return {
        "name": name,
        "outcome": outcome,
        "failures": fails,
        "failure_codes": sorted(codes),
        "quoted": quoted,
        "n_grid_points": len(pts),
        "n_in_quoted_range": len(in_range),
        "at_in_quoted_range": [p["at"] for p in in_range],
        "grid_ratio_min": grid_min,
        "grid_ratio_max": grid_max,
        "stated_range": stated_range,
        "sign_change_intervals": sign_change_intervals,
        "ordering_change_intervals": ordering_changes,
        "coherent_sub_ranges": subs,
        # Coherence over the points that COULD be evaluated, reported
        # separately so a NOT_EVALUABLE verdict does not discard a measurement.
        "coherent_over_evaluable_points": not (codes & {
            "NUMERATOR_CHANGES_SIGN", "DENOMINATOR_CHANGES_SIGN",
            "NUMERATOR_IS_ZERO_AT_A_GRID_POINT",
            "DENOMINATOR_IS_ZERO_AT_A_GRID_POINT",
            "TERM_ORDERING_CHANGES"}),
        # DESIGN QUESTION 2: always reported, never moves the outcome.
        "grid_context": _grid_context(name, out_of_range),
        "per_point": per_point,
        "defensible_form": defensible_range_form(name, outcome, subs,
                                                 grid_min, grid_max),
    }


def _grid_context(name, out_of_range) -> dict:
    """Incoherence OUTSIDE the quoted span. Reported always; never scored.

    A quoted range that stops one grid point short of a sign reversal is a
    SELECTION, and AB1 decision A's reporting standard is the reason a reader
    has to be told. This does not make the quote incoherent, so it does not
    move the outcome -- see DESIGN QUESTION 2 above.
    """
    if not out_of_range:
        return {"n_points_outside": 0, "outside_incoherence": None}
    probe = range_coherence_diff(f"{name} (outside the quoted span)",
                                 out_of_range, quoted=None)
    return {
        "n_points_outside": len(out_of_range),
        "at_outside": [p["at"] for p in out_of_range],
        # Only an INCOHERENT_* cell is reported here. A single point outside
        # the span is not "incoherence outside the span" -- reporting
        # NOT_A_RANGE_FEWER_THAN_TWO_GRID_POINTS as context would be an
        # artifact of how the grid was split, not a fact about the record.
        "outside_incoherence": (probe["outcome"]
                                if probe["outcome"].startswith("INCOHERENT")
                                else None),
        "outside_sign_change_intervals": probe["sign_change_intervals"],
    }


def coherent_sub_ranges(in_range, sign_change_intervals, ordering_changes):
    """The maximal spans inside the quoted range with no incoherence in them.

    **A SEPARATE FUNCTION RATHER THAN A WIDER RETURN** from the analysis above
    (gotcha 27's closing rule: when a helper needs to expose more, add a
    function rather than grow a return arity).
    """
    boundaries = set()
    for changes in sign_change_intervals.values():
        for a, b in changes:
            boundaries.add((a, b))
    for a, b in ordering_changes:
        boundaries.add((a, b))
    zero_ats = {p["at"] for p in in_range
                for side in ("numerator", "denominator")
                if _finite(_sign_basis(p[side])[0])
                and float(_sign_basis(p[side])[0]) == 0.0}

    subs, current = [], []
    for i, p in enumerate(in_range):
        if p["at"] in zero_ats:
            if len(current) >= 1:
                subs.append(current)
            current = []
            continue
        current.append(p)
        if i + 1 < len(in_range) and (p["at"], in_range[i + 1]["at"]) in boundaries:
            subs.append(current)
            current = []
    if current:
        subs.append(current)

    out = []
    for s in subs:
        rs = []
        for p in s:
            nv, dv = p["numerator"].get("value"), p["denominator"].get("value")
            if _finite(nv) and _finite(dv) and float(dv) != 0.0:
                rs.append(float(nv) / float(dv))
        out.append({"at_from": s[0]["at"], "at_to": s[-1]["at"],
                    "n_points": len(s),
                    "ratio_min": min(rs) if rs else None,
                    "ratio_max": max(rs) if rs else None})
    return out


def defensible_range_form(name, outcome, subs, grid_min, grid_max) -> str:
    """The sentence the record may carry, DERIVED from the grid (gotcha 20).

    **TOTAL, WITH A TEST FOR THE TOTAL BRANCH** (gotcha 32c): a completed
    measurement must never be reported as a failed one because a formatter had
    no branch for its verdict.
    """
    def _spans():
        return "; ".join(
            f"{s['at_from']!r}..{s['at_to']!r}"
            + (f" ({s['ratio_min']:.4g}x-{s['ratio_max']:.4g}x)"
               if s["ratio_min"] is not None else " (no ratio computable)")
            for s in subs) or "none"

    if outcome == "NOT_A_RANGE_FEWER_THAN_TWO_GRID_POINTS":
        return (f"{name} does not span two grid points, so it is not a range. "
                "Quote the single point with its setting, or widen the span.")
    if outcome == "NOT_A_RANGE_A_GRID_POINT_HAS_NO_DEFINED_COMPARISON":
        return (f"{name} contains a grid point at which no comparison is "
                "defined. Report each quantity on its own scale.")
    if outcome in ("INCOHERENT_BOTH_COMPONENTS_CHANGE_SIGN",
                   "INCOHERENT_ONE_COMPONENT_CHANGES_SIGN",
                   "INCOHERENT_A_COMPONENT_IS_ZERO_AT_A_GRID_POINT",
                   "INCOHERENT_THE_TERM_ORDERING_CHANGES"):
        return (f"{name} is not coherent across its own span ({outcome}). "
                f"Report the difference, or quote a multiplier only inside a "
                f"named sub-range: {_spans()}.")
    if outcome == "NOT_EVALUABLE_COHERENCE_IS_UNDETERMINED_ON_THIS_RANGE":
        return (f"Whether {name} is coherent across its span is UNDETERMINED "
                "from what is recorded -- a grid point has no usable value, or "
                "a non-negative component declares no sign basis. This is "
                "ignorance, not a pass: record the missing values or the "
                "`magnitude_of` claim before quoting the range.")
    if outcome == "COHERENT":
        span = (f"{grid_min:.4g}x to {grid_max:.4g}x"
                if grid_min is not None else "no ratio computable")
        return (f"{name} is coherent across its span: {span}, with no "
                "component reversing sign and the term ordering stable. Each "
                "point must still clear the per-point rule.")
    # TOTAL BRANCH (gotcha 32c). Reached only if RANGE_OUTCOMES gains a member
    # without this formatter gaining a branch, which is a defect in THIS file.
    return (f"{name}: {outcome} has no formatter branch. This is a defect in "
            "`defensible_range_form`, not a verdict about the range.")


# ------------------------------------------------------- the candidate finder

#: Multiplier-shaped tokens. Deliberately broad: this finds CANDIDATES for a
#: human to adjudicate, and a candidate finder that misses is worse than one
#: that over-reports. It is NOT the check -- the check needs two components,
#: which live in prose and cannot be extracted.
_MULTIPLIER = re.compile(
    r"(?<![\w.])(\d{1,6}(?:,\d{3})*(?:\.\d+)?)\s*(x\b|×|-fold\b|\s*times\b)",
    re.IGNORECASE)


def find_multipliers(text: str, context: int = 90) -> list:
    """Every multiplier-shaped token in `text`, with its line and context.

    Returns CANDIDATES, not findings. `ratio_rule_diff` adjudicates a candidate
    once its two components have been supplied from the record.
    """
    out = []
    for i, line in enumerate(text.splitlines(), start=1):
        for m in _MULTIPLIER.finditer(line):
            start = max(0, m.start() - context)
            out.append({
                "line": i,
                "token": m.group(0).strip(),
                "value": float(m.group(1).replace(",", "")),
                "context": line[start:m.end() + context].strip(),
            })
    return out


def main(argv=None) -> int:
    """Report candidates in the named files. **Always exits 0 by design.**

    This is a reporting instrument, not a gate. Nothing consumes its exit
    status, so there is no signal path to protect (gotcha 13 extension 1
    applies to validators that something gates on). Saying so here is the
    point: a later reader must not mistake the zero for a pass.
    """
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--min-value", type=float, default=None,
                    help="report only tokens at or above this multiplier")
    args = ap.parse_args(argv)

    report = {}
    for p in args.paths:
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            hits = find_multipliers(fh.read())
        if args.min_value is not None:
            hits = [h for h in hits if h["value"] >= args.min_value]
        report[os.path.normpath(p)] = hits

    if args.json:
        print(json.dumps(report, indent=1))
    else:
        for p, hits in report.items():
            print(f"\n=== {p}: {len(hits)} multiplier-shaped tokens ===")
            for h in hits:
                print(f"  {h['line']:>6}  {h['token']:<10} {h['context']}")
        total = sum(len(v) for v in report.values())
        print(f"\nCANDIDATES, NOT FINDINGS: {total} tokens across "
              f"{len(report)} files. Each needs its two components supplied "
              "before `ratio_rule_diff` can adjudicate it. Exit 0 is not a "
              "pass -- this reports and does not gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
