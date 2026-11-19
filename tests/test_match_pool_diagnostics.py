"""Tests for BT part B4's matching-pool diagnostics.

Every expectation is WRITTEN DOWN and hand-counted (criterion 2). The tie
fixture in particular is constructed so the answer is obvious on inspection:
equal fire_counts give equal distances, so the multiplicity is just how many
controls share the winning count.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.match_pool_diagnostics import (
    log10_rate,
    nearest_within_caliper,
    pool_diagnostics,
)


# ---------------------------------------------------------------------------
# log10_rate
# ---------------------------------------------------------------------------

def test_a_zero_count_is_NaN_and_never_minus_infinity():
    """`-inf` would take part in a minimum and silently become somebody's
    nearest control. NaN is dropped and counted instead."""
    r = log10_rate(np.array([0, 10, 100]), 1000)
    assert np.isnan(r[0])
    assert r[1] == pytest.approx(-2.0)
    assert r[2] == pytest.approx(-1.0)


def test_log10_rate_REFUSES_a_non_positive_row_count():
    with pytest.raises(ValueError, match="positive"):
        log10_rate(np.array([1]), 0)


# ---------------------------------------------------------------------------
# nearest_within_caliper
# ---------------------------------------------------------------------------

def test_EQUAL_counts_tie_EXACTLY_and_all_of_them_are_at_the_minimum():
    """THE TIE BRANCH, hand-counted. One treated at log10 = -2. Three controls
    sit at exactly -2 and one sits further away but still inside the caliper.
    The minimum distance is 0 and THREE controls are at it."""
    t = np.array([-2.0])
    c = np.array([-2.0, -2.0, -2.0, -2.05])
    r = nearest_within_caliper(t, c, caliper=0.1)
    assert r["n_at_min"].tolist() == [3]
    assert r["min_distance"][0] == pytest.approx(0.0)
    # WITH replacement and ALL tied candidates used: the three tied controls
    # are used once each, the fourth is not used at all.
    assert r["use_count"].tolist() == [1, 1, 1, 0]


def test_a_unique_nearest_gives_multiplicity_one():
    r = nearest_within_caliper(np.array([-2.0]),
                               np.array([-2.01, -2.05, -2.09]), caliper=0.1)
    assert r["n_at_min"].tolist() == [1]
    assert r["use_count"].tolist() == [1, 0, 0]


def test_a_treated_feature_with_NOTHING_inside_the_caliper_is_counted_not_dropped():
    """Gotcha 5's mandatory unmatched count, and criterion 6: it is its own
    category, never folded into a denominator."""
    r = nearest_within_caliper(np.array([-2.0, -5.0]),
                               np.array([-2.02]), caliper=0.1)
    assert r["n_at_min"].tolist() == [1, 0]
    assert r["n_treated_unmatchable"] == 1
    assert r["n_treated"] == 2


def test_the_caliper_boundary_is_INCLUSIVE_where_float64_can_express_it():
    """A control at exactly the caliper is admissible: the predicate is
    `d <= caliper`. Stated as a test because the caliper is a recorded degree
    of freedom (AN3) and its boundary is part of what is pinned.

    THE FIXTURE USES EXACTLY-REPRESENTABLE VALUES, and the first version of it
    did not -- that is how the next test's fact was found."""
    r = nearest_within_caliper(np.array([-2.0]), np.array([-1.5]),
                               caliper=0.5)
    assert r["n_at_min"].tolist() == [1]


def test_the_boundary_is_NOT_RELIABLY_REACHABLE_at_the_pinned_caliper_of_0_1():
    """MEASURED, NOT ARGUED, and recorded because it qualifies the test above.

    `abs(-2.0 - (-2.1))` is 0.10000000000000009 in float64, which is strictly
    GREATER than 0.1 -- so a control the arithmetic places exactly one caliper
    away is EXCLUDED. Inclusivity at the boundary is therefore a property of
    the predicate that the pinned caliper of 0.1 cannot generally exercise.

    This changes nothing about the arm: which pairs enter is decided by a
    continuum of distances and a single unreachable boundary point moves no
    estimate. It is written down so that nobody later reads `<=` as a promise
    the floating point can keep, and so the failing first version of the test
    above is on the record as having found something rather than as a typo."""
    assert abs(-2.0 - (-2.1)) > 0.1
    r = nearest_within_caliper(np.array([-2.0]), np.array([-2.1]),
                               caliper=0.1)
    assert r["n_at_min"].tolist() == [0]
    assert r["n_treated_unmatchable"] == 1


def test_REUSE_is_counted_across_treated_features():
    """THE COST OF REPLACEMENT, which is what makes it reportable. Two treated
    features both have the same single nearest control, so that control is
    used twice -- under matching WITHOUT replacement one of them would have
    been pushed to a worse partner, and which one would depend on order."""
    r = nearest_within_caliper(np.array([-2.0, -2.01]), np.array([-2.005]),
                               caliper=0.1)
    assert r["use_count"].tolist() == [2]


def test_a_NaN_rate_is_dropped_from_both_sides_and_counted():
    r = nearest_within_caliper(np.array([-2.0, np.nan]),
                               np.array([-2.01, np.nan]), caliper=0.1)
    assert r["n_treated_no_rate"] == 1
    assert r["n_control_no_rate"] == 1
    assert r["n_at_min"].tolist() == [1, 0]


def test_nearest_REFUSES_a_non_positive_caliper():
    with pytest.raises(ValueError, match="must be positive"):
        nearest_within_caliper(np.array([-2.0]), np.array([-2.0]), caliper=0.0)


# ---------------------------------------------------------------------------
# pool_diagnostics
# ---------------------------------------------------------------------------

def test_a_seed_with_no_treated_feature_is_REPORTED_not_skipped():
    """Arm A has ALIGNED = 0 in all twenty seeds by the resolution floor. A
    diagnostic that skipped those seeds would read as a measurement of a pool
    nobody examined (gotcha 28)."""
    fc = np.array([[10, 10, 10], [10, 10, 10]])
    aligned = np.array([[False, False, False], [True, False, False]])
    orphan = np.array([[False, True, True], [False, True, True]])
    d = pool_diagnostics(fc, 1000, aligned, orphan, 0.1, ["s0", "s1"])
    assert d["seeds_with_no_treated_feature"] == ["s0"]
    assert d["per_seed"][0]["n_treated"] == 0
    assert d["n_seeds"] == 2
    assert d["across_seeds"]["denominator_treated_matched_pairs"] == 1


def test_a_feature_that_is_BOTH_aligned_and_orphan_is_REFUSED():
    """AV5's cells are disjoint by construction; a treated feature must never
    be available as its own control."""
    fc = np.array([[10, 10]])
    with pytest.raises(ValueError, match="both ALIGNED and ORPHAN"):
        pool_diagnostics(fc, 1000, np.array([[True, False]]),
                         np.array([[True, False]]), 0.1, ["s0"])


def test_pool_diagnostics_REFUSES_a_seed_label_count_mismatch():
    with pytest.raises(ValueError, match="joined on labels"):
        pool_diagnostics(np.zeros((2, 3)), 1000, np.zeros((2, 3), bool),
                         np.zeros((2, 3), bool), 0.1, ["only_one"])


def test_every_reported_rate_states_its_denominator():
    """Criterion 6, asserted structurally rather than trusted: the across-seed
    block must carry both denominators alongside the rates it reports."""
    fc = np.array([[10, 10, 12]])
    d = pool_diagnostics(fc, 1000, np.array([[True, False, False]]),
                         np.array([[False, True, True]]), 0.1, ["s0"])
    a = d["across_seeds"]
    assert "denominator_treated_matched_pairs" in a
    assert "denominator_distinct_controls_used" in a
    # feature 1 has the SAME count as the treated feature, so distance 0;
    # feature 2 is further. Unique nearest, multiplicity 1.
    assert a["tie_multiplicity_max"] == 1
    assert a["denominator_treated_matched_pairs"] == 1
