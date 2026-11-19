"""Tests for src/analysis/device_divergence.py.

EVERY EXPECTATION HERE IS WRITTEN DOWN BY HAND, never computed by calling the
thing under test (gotcha 32b: a test that derives its cases from the code cannot
detect changes to that code). The gain values below were worked out on paper from
G = (AP - p) / (1 - p) and are asserted as literals.
"""

import numpy as np
import pytest

from src.analysis.device_divergence import (
    GAIN_FLOOR, align_rows, compare, distribution, format_summary, gain,
    pooled_max,
)


# --------------------------------------------------------------------------
# The floor is a pre-registered constant, not a tunable.
# --------------------------------------------------------------------------
def test_the_gain_floor_is_the_preregistered_0_05():
    assert GAIN_FLOOR == 0.05


# --------------------------------------------------------------------------
# Seed alignment -- gotcha 29.
# --------------------------------------------------------------------------
def test_align_rows_recovers_the_permutation_not_the_identity():
    # A is numeric order, B is lexicographic. Only rows 0 and 1 coincide --
    # the exact shape of gotcha 29.
    a = ["seed_0", "seed_1", "seed_2", "seed_10"]
    b = ["seed_0", "seed_1", "seed_10", "seed_2"]
    assert align_rows(a, b).tolist() == [0, 1, 3, 2]


def test_align_rows_REFUSES_when_the_label_sets_differ():
    with pytest.raises(SystemExit):
        align_rows(["seed_0", "seed_1"], ["seed_0", "seed_9"])


def test_align_rows_on_identical_orders_is_the_identity():
    a = ["seed_0", "seed_1", "seed_2"]
    assert align_rows(a, a).tolist() == [0, 1, 2]


# --------------------------------------------------------------------------
# The gain transform, computed on paper.
# --------------------------------------------------------------------------
def test_gain_matches_hand_computed_values():
    # p = 0.5, AP = 0.75  ->  (0.75 - 0.5) / 0.5 = 0.5
    # p = 0.0, AP = 0.30  ->  0.30 / 1.0        = 0.30
    ap = np.array([[[0.75, 0.30]]])
    prev = np.array([0.5, 0.0])
    g = gain(ap, prev)
    assert g[0, 0, 0] == pytest.approx(0.5)
    assert g[0, 0, 1] == pytest.approx(0.30)


def test_gain_is_zero_when_AP_equals_prevalence():
    """A detector no better than the base rate has gain exactly 0 -- the
    property the floor is defined against."""
    ap = np.array([[[0.42]]])
    g = gain(ap, np.array([0.42]))
    assert g[0, 0, 0] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# The pool restriction.
# --------------------------------------------------------------------------
def test_pooled_max_IGNORES_out_of_pool_concepts_even_when_they_would_win():
    # Concept 1 has the largest gain but is OUT of the pool. The winner must be
    # concept 2, and the index must be into the FULL axis (2, not 1).
    g = np.array([[[0.1, 0.9, 0.4]]])
    in_pool = np.array([True, False, True])
    gf, arg = pooled_max(g, in_pool)
    assert gf[0, 0] == pytest.approx(0.4)
    assert arg[0, 0] == 2


def test_pooled_max_returns_full_axis_indices():
    """The pooled subset is not the concept axis; a winner reported as a subset
    index would silently name a different concept."""
    g = np.array([[[0.0, 0.0, 0.7, 0.0]]])
    in_pool = np.array([False, False, True, True])
    _, arg = pooled_max(g, in_pool)
    assert arg[0, 0] == 2


# --------------------------------------------------------------------------
# distribution -- a mean alone hides a spike at zero.
# --------------------------------------------------------------------------
def test_distribution_reports_the_spike_at_zero_separately():
    d = distribution(np.array([0.0, 0.0, 0.0, 4.0]))
    assert d["n"] == 4
    assert d["n_exactly_zero"] == 3
    assert d["frac_exactly_zero"] == pytest.approx(0.75)
    assert d["mean"] == pytest.approx(1.0)
    assert d["max"] == pytest.approx(4.0)


def test_distribution_is_a_STATUS_not_a_crash_on_empty_input():
    assert distribution(np.array([]))["status"] == "EMPTY"


# --------------------------------------------------------------------------
# compare(): the whole report, on a hand-built pair.
# --------------------------------------------------------------------------
def _art(ap, fire, seeds, prevalence=None, in_pool=None, concepts=None):
    ap = np.asarray(ap, dtype=np.float64)
    n_c = ap.shape[2]
    return {
        "ap": ap,
        "prevalence": np.zeros(n_c) if prevalence is None else np.asarray(prevalence, float),
        "in_pool": np.ones(n_c, bool) if in_pool is None else np.asarray(in_pool, bool),
        "concepts": [f"C{i}" for i in range(n_c)] if concepts is None else concepts,
        "fire_counts": np.asarray(fire, dtype=np.int64),
        "seeds": seeds,
        "attrs": {},
    }


def test_compare_counts_a_floor_flip_in_the_right_direction():
    # prevalence 0 everywhere, so G == AP. One feature crosses 0.05.
    # seed_0 feature_0: 0.04 -> 0.06  = a GAIN (was below, now above)
    # seed_0 feature_1: 0.90 -> 0.90  = unchanged
    a = _art([[[0.04], [0.90]]], [[10, 20]], ["seed_0"])
    b = _art([[[0.06], [0.90]]], [[10, 20]], ["seed_0"])
    rep = compare(a, b)
    f = rep["gain_floor_flips"]
    assert f["n_flipped"] == 1
    assert f["n_gained"] == 1
    assert f["n_lost"] == 0
    assert f["n_pass_reference"] == 1     # only feature_1
    assert f["n_pass_other"] == 2         # feature_0 joined it


def test_compare_counts_a_floor_flip_in_the_LOSING_direction_too():
    """A flip is SYMMETRIC. This test exists because mutation testing found the
    asymmetric implementation (`flips = ~pass_a & pass_b`) survived a suite whose
    only flip case was a gain -- a feature that DROPS below the floor is just as
    much a changed classification as one that rises above it."""
    # seed_0 feature_0: 0.90 -> 0.01  = a LOSS (was above, now below)
    a = _art([[[0.90]]], [[10]], ["seed_0"])
    b = _art([[[0.01]]], [[10]], ["seed_0"])
    rep = compare(a, b)
    f = rep["gain_floor_flips"]
    assert f["n_flipped"] == 1
    assert f["n_lost"] == 1
    assert f["n_gained"] == 0


def test_compare_counts_gains_and_losses_in_the_same_report():
    """Both directions at once, so a one-sided count cannot match the total."""
    # feature_0 rises through the floor, feature_1 falls through it.
    a = _art([[[0.01], [0.90]]], [[1, 2]], ["seed_0"])
    b = _art([[[0.90], [0.01]]], [[1, 2]], ["seed_0"])
    f = compare(a, b)["gain_floor_flips"]
    assert (f["n_flipped"], f["n_gained"], f["n_lost"]) == (2, 1, 1)


def test_compare_counts_an_argmax_change():
    # Two concepts. Reference wins on C0; other wins on C1.
    a = _art([[[0.8, 0.2]]], [[5]], ["seed_0"])
    b = _art([[[0.2, 0.8]]], [[5]], ["seed_0"])
    rep = compare(a, b)
    assert rep["argmax_changes"]["n_changed"] == 1
    assert rep["argmax_changes"]["n_features_changed_in_any_seed"] == 1


def test_compare_APPLIES_the_seed_permutation_rather_than_pairing_by_position():
    """The load-bearing one. B's rows are in the opposite order; after aligning
    on labels the two artifacts are IDENTICAL, so a correct implementation
    reports zero divergence and a positional one reports maximal divergence."""
    a = _art([[[0.10]], [[0.90]]], [[1], [2]], ["seed_0", "seed_1"])
    b = _art([[[0.90]], [[0.10]]], [[2], [1]], ["seed_1", "seed_0"])
    rep = compare(a, b)
    assert rep["ap_divergence"]["n_differing"] == 0
    assert rep["fire_counts_divergence"]["n_differing"] == 0
    assert rep["seed_order"]["other_permutation_applied"] == [1, 0]


def test_compare_REFUSES_when_prevalence_differs():
    """Prevalence cannot depend on the gather. If it moved, the pair differs in
    something else and every divergence below would be misattributed."""
    a = _art([[[0.5]]], [[1]], ["seed_0"], prevalence=[0.10])
    b = _art([[[0.5]]], [[1]], ["seed_0"], prevalence=[0.11])
    with pytest.raises(SystemExit):
        compare(a, b)


def test_compare_REFUSES_when_the_concept_axes_differ():
    a = _art([[[0.5]]], [[1]], ["seed_0"], concepts=["ACT_SITE"])
    b = _art([[[0.5]]], [[1]], ["seed_0"], concepts=["BINDING"])
    with pytest.raises(SystemExit):
        compare(a, b)


def test_compare_REFUSES_when_the_argmax_pools_differ():
    a = _art([[[0.5, 0.1]]], [[1]], ["seed_0"], in_pool=[True, True])
    b = _art([[[0.5, 0.1]]], [[1]], ["seed_0"], in_pool=[True, False])
    with pytest.raises(SystemExit):
        compare(a, b)


def test_identical_artifacts_report_zero_divergence_everywhere():
    """The control: the instrument must be capable of reporting NO difference,
    or a DIFFERS verdict means nothing (gotcha 25)."""
    a = _art([[[0.3, 0.7], [0.1, 0.2]]], [[7, 9]], ["seed_0"])
    b = _art([[[0.3, 0.7], [0.1, 0.2]]], [[7, 9]], ["seed_0"])
    rep = compare(a, b)
    assert rep["ap_divergence"]["n_differing"] == 0
    assert rep["fire_counts_divergence"]["n_differing"] == 0
    assert rep["gain_floor_flips"]["n_flipped"] == 0
    assert rep["argmax_changes"]["n_changed"] == 0


def test_fire_counts_sum_equality_is_reported_and_can_be_False():
    a = _art([[[0.5]]], [[10]], ["seed_0"])
    b = _art([[[0.5]]], [[11]], ["seed_0"])
    rep = compare(a, b)
    assert rep["fire_counts_divergence"]["sum_equal"] is False
    assert rep["fire_counts_divergence"]["n_differing"] == 1


# --------------------------------------------------------------------------
# The formatter must be total (gotcha 32c).
# --------------------------------------------------------------------------
def test_the_formatter_is_TOTAL_over_an_empty_report():
    assert "EMPTY" in format_summary({})


def test_the_formatter_renders_a_complete_report():
    a = _art([[[0.04], [0.90]]], [[10, 20]], ["seed_0"])
    b = _art([[[0.06], [0.90]]], [[10, 20]], ["seed_0"])
    s = format_summary(compare(a, b))
    assert "GAIN-FLOOR FLIPS" in s
    assert "ARGMAX CHANGES" in s
    assert "FIRE_COUNTS" in s
