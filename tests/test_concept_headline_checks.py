"""Tests for `concept_headline_checks` -- amendment AZ sections 13 and 14.

THE FIXTURE THAT MATTERS IS THE COLLISION ONE. Null B rescales a positive row's
offset onto a destination protein and the map is many-to-one when the
destination is SHORTER, after which `np.unique` drops the duplicates. A fixture
whose proteins are all the same length cannot exhibit that at all -- realised
would equal observed for every draw and every mutation to the dedup or to the
rescale would survive. So the fixture here pairs a long protein with a short
one and the expected count is computed BY HAND from the rescale formula.

And the bounded-headline fixture contains a non-finite margin, because "counted
separately rather than swept into either group" is a branch that only a
fixture carrying one can reach.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis import concept_headline_checks as hc


# ------------------------------------------------ AZ13: realised prevalence

def _two_proteins():
    """Protein 0: rows 0..9 (length 10). Protein 1: rows 10..14 (length 5)."""
    starts = np.array([0, 10], dtype=np.int64)
    lengths = np.array([10, 5], dtype=np.int64)
    row2protein = np.array([0] * 10 + [1] * 5, dtype=np.int32)
    return row2protein, starts, lengths


def test_realised_counts_LOSE_rows_when_the_destination_is_shorter():
    """Rows 0..9 of the 10-long protein are sent to the 5-long one. The
    rescale is floor(offset * 5 / 10), so offsets 0..9 collapse onto 0..4 --
    ten distinct rows become FIVE. Computed by hand, not read off the code."""
    row2protein, starts, lengths = _two_proteins()
    rows = np.arange(10, dtype=np.int64)
    dest = np.array([[1, 0]], dtype=np.int64)      # protein 0 -> protein 1
    got = hc.realised_positive_counts(rows, row2protein, starts, lengths, dest)
    assert got.tolist() == [5]


def test_realised_counts_are_PRESERVED_when_the_destination_is_longer():
    """The mirror case, which is what makes the previous test about the
    direction rather than about the machinery running at all: 5 rows sent onto
    a 10-long protein stay 5 distinct rows."""
    row2protein, starts, lengths = _two_proteins()
    rows = np.arange(10, 15, dtype=np.int64)
    dest = np.array([[1, 0]], dtype=np.int64)      # protein 1 -> protein 0
    got = hc.realised_positive_counts(rows, row2protein, starts, lengths, dest)
    assert got.tolist() == [5]


def test_realised_counts_are_exact_under_the_identity_schedule():
    row2protein, starts, lengths = _two_proteins()
    rows = np.array([0, 3, 7, 11], dtype=np.int64)
    dest = np.array([[0, 1]], dtype=np.int64)
    got = hc.realised_positive_counts(rows, row2protein, starts, lengths, dest)
    assert got.tolist() == [4]


def test_realised_counts_report_one_value_per_stored_shift():
    row2protein, starts, lengths = _two_proteins()
    rows = np.arange(10, dtype=np.int64)
    dest = np.array([[0, 1], [1, 0], [0, 1]], dtype=np.int64)
    got = hc.realised_positive_counts(rows, row2protein, starts, lengths, dest)
    assert got.tolist() == [10, 5, 10]


def test_drift_summary_reports_the_distribution_not_only_a_mean():
    """A schedule losing everything on one draw and nothing on nineteen is not
    the same object as one losing 2.5% every time; a mean alone cannot say so."""
    s = hc.drift_summary(100, np.array([100, 100, 50]))
    assert s["relative_drift_median"] == pytest.approx(0.0)
    assert s["relative_drift_min"] == pytest.approx(-0.5)
    assert s["relative_drift_mean"] == pytest.approx(-0.5 / 3)
    assert s["n_draws_exactly_preserved"] == 2
    assert s["n_draws_above_observed"] == 0


def test_drift_summary_counts_a_draw_ABOVE_observed_rather_than_assuming_none():
    """`realised <= observed` holds by construction, so a draw above it would
    be a defect. It is counted, not assumed impossible."""
    s = hc.drift_summary(10, np.array([11, 10]))
    assert s["n_draws_above_observed"] == 1


def _pop(median_drifts):
    per = {f"c{i}": hc.drift_summary(1000, np.array([int(1000 * (1 + d))] * 3))
           for i, d in enumerate(median_drifts)}
    return hc.drift_population(per)


def test_preservation_verdict_is_total_over_its_stated_bands():
    exact = hc.preservation_verdict(_pop([0.0, 0.0, 0.0]))
    assert exact["verdict"].startswith("EXACTLY_PRESERVED")

    tiny = hc.preservation_verdict(_pop([-0.001] * 100))
    assert tiny["verdict"] == "NOT_PRESERVED_BUT_UNDER_1_PERCENT_FOR_AT_LEAST_95_PERCENT"

    small = hc.preservation_verdict(_pop([-0.02] * 100))
    assert small["verdict"] == "NOT_PRESERVED_UNDER_5_PERCENT_FOR_AT_LEAST_95_PERCENT"

    big = hc.preservation_verdict(_pop([-0.30] * 100))
    assert big["verdict"].startswith("NOT_PRESERVED_AND_MATERIAL")


def test_preservation_verdict_states_that_this_is_a_measurement_not_a_check():
    v = hc.preservation_verdict(_pop([0.0]))
    assert v["by_design"].startswith("NO")
    assert "REPORTED ONLY" in v["action"]


def test_drift_population_names_its_worst_concepts():
    per = {"good": hc.drift_summary(100, np.array([100, 100])),
           "bad": hc.drift_summary(100, np.array([40, 100]))}
    pop = hc.drift_population(per)
    assert pop["worst_ten_by_single_draw"][0]["concept"] == "bad"
    assert pop["worst_ten_by_single_draw"][0][
        "worst_draw_relative_drift"] == pytest.approx(-0.6)
    assert pop["n_concepts_exactly_preserved_in_every_draw"] == 1


# ------------------------------------------------ AZ14: the bounded headline

def test_bounded_headline_keys_on_the_FRAGILE_DETECTIONS():
    """Three concepts at D = 1 (margins 10, 1.5, 30) and two below it. Only the
    1.5 one is a fragile DETECTION, so excluding it leaves 2 of 4 and counting
    it as undetected leaves 2 of 5."""
    d = np.array([1.0, 1.0, 1.0, 0.5, 0.0])
    worst = np.array([10.0, 1.5, 30.0, 0.4, 0.1])
    b = hc.bounded_headline(d, worst, bound=2.0)
    assert b["n_fragile_any_concept"] == 3
    assert b["n_fragile_detections"] == 1
    assert b["with_all_concepts"]["share_D_eq_1"] == pytest.approx(3 / 5)
    assert b["excluding_fragile_detections"]["share_D_eq_1"] == pytest.approx(0.5)
    assert b["excluding_fragile_detections"]["shift_pp"] == pytest.approx(-10.0)
    assert b["counting_fragile_as_undetected"][
        "share_D_eq_1"] == pytest.approx(2 / 5)


def test_the_degenerate_arm_is_MARKED_degenerate_from_the_data():
    """Excluding EVERY fragile concept removes every D < 1 concept by
    construction, so the arm returns 1.0 whatever the data say. The flag has to
    be DERIVED from these numbers, not asserted in prose (gotcha 20)."""
    d = np.array([1.0, 1.0, 0.5, 0.0])
    worst = np.array([10.0, 1.5, 0.4, 0.1])
    b = hc.bounded_headline(d, worst, bound=2.0)
    arm = b["excluding_every_fragile_concept_DEGENERATE"]
    assert arm["share_D_eq_1"] == pytest.approx(1.0)
    assert arm["is_tautological"] is True


def test_the_degeneracy_flag_is_FALSE_when_an_undetected_concept_is_secure():
    """The flag must be able to come out False, or it is not a measurement. A
    concept at D = 0.5 whose WORST seed still clears 2x is not fragile, so it
    survives the exclusion and the arm is no longer forced to 1.0."""
    d = np.array([1.0, 0.5])
    worst = np.array([10.0, 5.0])
    b = hc.bounded_headline(d, worst, bound=2.0)
    arm = b["excluding_every_fragile_concept_DEGENERATE"]
    assert arm["is_tautological"] is False
    assert arm["share_D_eq_1"] == pytest.approx(0.5)


def test_the_bound_is_STRICT_so_a_margin_of_exactly_2_is_not_fragile():
    d = np.array([1.0, 1.0])
    b = hc.bounded_headline(d, np.array([2.0, 1.999]), bound=2.0)
    assert b["n_fragile_detections"] == 1


def test_a_non_finite_margin_is_counted_separately_not_swept_into_a_group():
    d = np.array([1.0, 1.0, 0.0])
    worst = np.array([10.0, np.nan, 0.1])
    b = hc.bounded_headline(d, worst, bound=2.0)
    assert b["n_with_non_finite_margin"] == 1
    # The NaN concept is NOT a fragile detection (it could not be classified)
    # and stays in the kept set, so the kept count is 3 rather than 2.
    assert b["n_fragile_detections"] == 0
    assert b["excluding_fragile_detections"]["n_concepts"] == 3


def test_bounded_headline_labels_itself_arithmetic_rather_than_a_finding():
    b = hc.bounded_headline(np.array([1.0]), np.array([5.0]))
    assert "ARITHMETIC" in b["reading"]


# ============================================================================
# BA10 -- the drift's consequence (amendment BA sections 7.2 and 7.3)
# ============================================================================
# THE FIXTURE PRINCIPLE HERE IS THE UNIT ONE. `null_cmax_b` stores a GAIN that
# was already normalised at the draw's realised prevalence, so a fixture whose
# prevalence is constant, or whose gain and AP happen to agree, cannot show
# that the two units answer different questions. The load-bearing fixture is
# `test_gain_and_AP_can_disagree_in_SIGN`: a gain deliberately held CONSTANT
# across draws, whose reconstructed AP must still track prevalence.

def test_null_max_vs_prevalence_reports_a_CONSTANT_prevalence_as_uninformative():
    """Gotcha 25b: a ranking statistic against a constant ranks the tie-break.
    65 of 703 real concepts are in this class and must never be reported as a
    correlation of zero."""
    realised = np.full(20, 500, dtype=np.int64)
    cmax = np.random.default_rng(0).random((20, 20))
    out = hc.null_max_vs_prevalence(realised, 100_000, cmax)
    assert out["status"] == "UNINFORMATIVE_PREVALENCE_IS_CONSTANT"
    assert "median_spearman_prevalence_vs_null_max_GAIN" not in out


def test_null_max_vs_prevalence_recovers_the_AP_from_the_stored_GAIN():
    """`ap = gain*(1-p) + p` is exact, so a hand-built case must round-trip --
    and the assertion has to be on the MODULE'S OWN AP OUTPUT.

    An earlier draft recomputed the reconstruction in the test and asserted a
    property of its own arithmetic, never touching what the module returned;
    the mutation `ap = gain` survived it untouched. The two reported
    correlations must genuinely DIFFER, which they cannot if the module skips
    the reconstruction."""
    realised = np.arange(400, 420, dtype=np.int64)
    n_rows = 1_000
    p = realised / n_rows
    # A gain that FALLS with prevalence, gently enough that the AP still rises:
    # d(ap)/dk = (dgain/dk)(1-p) + (dp/dk)(1-gain), and the second term wins.
    row = 0.50 - 0.0002 * np.arange(20)
    gain = np.tile(row, (3, 1))
    out = hc.null_max_vs_prevalence(realised, n_rows, gain)
    assert out["status"] == "COMPUTED"
    assert out["n_draws"] == 20 and out["n_seeds"] == 3

    ap = row * (1 - p) + p
    assert np.all(np.diff(ap) > 0) and np.all(np.diff(row) < 0), (
        "the fixture does not separate the two units")
    assert out["median_spearman_prevalence_vs_null_max_GAIN"] < 0
    assert out["median_spearman_prevalence_vs_null_max_AP"] > 0


def test_gain_and_AP_can_disagree_in_SIGN_which_is_why_both_are_reported():
    """THE DISCRIMINATING FIXTURE, and the reason BA7.2 refuses to report one
    unit without the other.

    The gain falls GENTLY with prevalence -- slowly enough that the
    reconstructed `ap = gain*(1-p) + p` still RISES, because the `+p` term
    outruns it. So the two units have OPPOSITE signs on the same data. A
    fixture where they agree could not show that the choice of unit is a real
    choice rather than a presentational one."""
    realised = np.arange(100, 120, dtype=np.int64)
    n_rows = 1_000
    p = realised / n_rows
    row = 0.5 - 0.0002 * np.arange(20)          # falls with prevalence
    gain = np.tile(row, (5, 1))
    ap = row * (1 - p) + p
    assert np.all(np.diff(row) < 0) and np.all(np.diff(ap) > 0), (
        "the fixture does not exhibit the disagreement it is named for")

    out = hc.null_max_vs_prevalence(realised, n_rows, gain)
    assert out["status"] == "COMPUTED"
    assert out["median_spearman_prevalence_vs_null_max_GAIN"] < -0.99
    assert out["median_spearman_prevalence_vs_null_max_AP"] > 0.99


def test_null_scaling_population_derives_the_direction_from_the_numbers():
    """Gotcha 20: the verdict is a function of the data, not a sentence."""
    pos = {f"c{i}": {"status": "COMPUTED",
                     "median_spearman_prevalence_vs_null_max_GAIN": 0.8,
                     "median_spearman_prevalence_vs_null_max_AP": 0.9}
           for i in range(10)}
    assert hc.null_scaling_population(pos)["verdict_gain"] == \
        "SCALES_POSITIVELY_WITH_PREVALENCE"
    neg = {k: {**v,
               "median_spearman_prevalence_vs_null_max_GAIN": -0.8,
               "median_spearman_prevalence_vs_null_max_AP": -0.9}
           for k, v in pos.items()}
    assert hc.null_scaling_population(neg)["verdict_gain"] == \
        "SCALES_NEGATIVELY_WITH_PREVALENCE"


def test_null_scaling_population_is_total_and_every_verdict_is_reachable():
    def pop(vals):
        return hc.null_scaling_population(
            {f"c{i}": {"status": "COMPUTED",
                       "median_spearman_prevalence_vs_null_max_GAIN": v,
                       "median_spearman_prevalence_vs_null_max_AP": v}
             for i, v in enumerate(vals)})
    got = {pop([0.5] * 10)["verdict_gain"],
           pop([-0.5] * 10)["verdict_gain"],
           pop([0.5] * 5 + [-0.5] * 5)["verdict_gain"]}
    assert got == {"SCALES_POSITIVELY_WITH_PREVALENCE",
                   "SCALES_NEGATIVELY_WITH_PREVALENCE",
                   "NO_CONSISTENT_DIRECTION"}


def test_null_scaling_population_counts_the_uninformative_concepts_separately():
    mixed = {"a": {"status": "COMPUTED",
                   "median_spearman_prevalence_vs_null_max_GAIN": 0.5,
                   "median_spearman_prevalence_vs_null_max_AP": 0.5},
             "b": {"status": "UNINFORMATIVE_PREVALENCE_IS_CONSTANT"}}
    out = hc.null_scaling_population(mixed)
    assert out["n_uninformative_constant_prevalence"] == 1
    assert out["gain_the_quantity_that_sets_the_threshold"]["n_concepts"] == 1


def test_null_scaling_population_with_NO_varying_concept_asks_nothing():
    out = hc.null_scaling_population(
        {"a": {"status": "UNINFORMATIVE_PREVALENCE_IS_CONSTANT"}})
    assert out["status"] == "NO_CONCEPT_HAD_VARYING_PREVALENCE"


# ---------------------------------------------------- BA10 3b: the margins

def test_margin_distribution_is_restricted_to_the_ALWAYS_DETECTED_concepts():
    """The three detection groups separate perfectly on this statistic and the
    separation is DEFINITIONAL (AY8.3). A `D < 1` concept leaking in would be
    counted as a fragile detection that no seed ever made."""
    d = np.array([1.0, 1.0, 0.5, 0.0])
    worst = np.array([1.1, 5.0, 0.3, 0.1])
    out = hc.margin_distribution(d, worst)
    assert out["n_concepts"] == 2
    assert out["counts_below"]["1.20"] == 1


def test_margin_distribution_cuts_are_strict_so_the_boundary_is_reachable():
    d = np.ones(3)
    worst = np.array([1.19999, 1.20000, 1.20001])
    out = hc.margin_distribution(d, worst)
    assert out["counts_below"]["1.20"] == 1, "the cut must be strict `<`"


def test_margin_distribution_counts_non_finite_margins_separately():
    """A non-positive threshold gives a non-finite margin. It cannot be
    classified and is counted rather than swept into either group."""
    d = np.array([1.0, 1.0, 1.0])
    worst = np.array([1.1, np.nan, np.inf])
    out = hc.margin_distribution(d, worst)
    # `np.isfinite` excludes BOTH nan and inf, so only the 1.1 is classifiable.
    assert out["n_concepts"] == 1
    assert out["n_always_detected"] == 3
    assert out["n_excluded_non_finite_margin"] == 2


def test_margin_distribution_reports_the_full_ladder_not_just_2x():
    d = np.ones(6)
    worst = np.array([1.01, 1.15, 1.25, 1.4, 1.7, 3.0])
    out = hc.margin_distribution(d, worst)
    assert sorted(out["shares_below"]) == ["1.05", "1.10", "1.20", "1.30",
                                           "1.50", "2.00"]
    assert out["counts_below"]["1.05"] == 1
    assert out["counts_below"]["2.00"] == 5


def test_margin_distribution_with_no_usable_concept_asks_nothing():
    out = hc.margin_distribution(np.array([0.0]), np.array([0.5]))
    assert out["status"] == "NO_ALWAYS_DETECTED_CONCEPT_WITH_A_FINITE_MARGIN"


def test_a_seed_with_a_CONSTANT_maximum_is_counted_not_crashed_on():
    """`concept_repro.spearman` REFUSES on a constant input, which is correct
    and would abort the whole job if a single (seed, concept) cell had a flat
    stored maximum. Those seeds are counted and the summary is taken over the
    informative ones."""
    realised = np.arange(100, 120, dtype=np.int64)
    gain = np.vstack([np.full(20, 0.5),               # constant
                      np.linspace(0.1, 0.9, 20)])     # varying
    out = hc.null_max_vs_prevalence(realised, 1_000, gain)
    assert out["status"] == "COMPUTED"
    assert out["n_seeds_with_a_constant_gain_maximum"] == 1
    assert out["median_spearman_prevalence_vs_null_max_GAIN"] > 0.99


def test_every_seed_constant_is_reported_rather_than_summarised():
    realised = np.arange(100, 120, dtype=np.int64)
    # gain constant in every seed AND chosen so the reconstructed AP is too:
    # with gain == 1.0, ap = 1*(1-p) + p == 1 exactly, for every p.
    gain = np.ones((3, 20))
    out = hc.null_max_vs_prevalence(realised, 1_000, gain)
    assert out["status"] == "UNINFORMATIVE_EVERY_SEED_MAXIMUM_IS_CONSTANT"
