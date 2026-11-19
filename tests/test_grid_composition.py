"""Tests for component 4 -- the thinness grid's classification composition
(PREREGISTRATION amendment BT).

CRITERION 2 GOVERNS THIS FILE: every expectation below is WRITTEN DOWN, and
where a number is computed it is computed by an independent route (a literal
BH step-up worked by hand, a set difference counted by hand) rather than by
calling the function under test. A test parametrised over `GRID` or deriving
its answer from `restricted_reject` could not detect a change to either.

WHAT EACH FIXTURE IS NAMED FOR, and the branch it has to reach:

  * the `>=` boundary at exactly 5 -- the degenerate concept's founding
    instance (criterion 4). It must SURVIVE T=5 and LEAVE at T=10.
  * the family is `keep AND evaluable` -- both halves, and a test outside the
    family must be False rather than a non-rejection.
  * THE LOOSENING, which is the whole mechanism BI3(b) names: a p-value that
    BH refuses in a large family and accepts in a small one. Worked by hand.
  * the counterfactual's SUBSET relation, which is what makes `gained_pool`
    zero by construction and is the only part of the decomposition that CAN
    fail.

Mutation kills RUN in tests/mutation_matrix_bt_grid.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.classification_bh import bh_reject
from src.analysis.grid_composition import (
    GRID,
    DecompositionViolation,
    clearance,
    decompose_aligned,
    restricted_reject,
    retained_mask,
)


# ---------------------------------------------------------------------------
# the grid itself
# ---------------------------------------------------------------------------

def test_the_grid_is_BI1s_and_is_written_out():
    """WRITTEN OUT, not derived (gotcha 32b). BI1 ratified 1/5/10/30/100 and a
    grid value is on BT-E's not-authorised list, so a change here must fail a
    test rather than ship."""
    assert GRID == (1, 5, 10, 30, 100)


# ---------------------------------------------------------------------------
# retained_mask -- the boundary does real work
# ---------------------------------------------------------------------------

def test_the_boundary_is_inclusive_and_five_positives_SURVIVES_T5():
    """CRITERION 4's founding instance. `BINDING:hypoxanthine` has exactly five
    observed positives. With `>=` it survives T=5 and leaves at T=10; a strict
    `>` would drop it at T=5 and remove the only grid point where the
    two-floor structure can still fire (§BS-D3)."""
    n_pos = np.array([1, 4, 5, 6, 10, 30, 100, 10520])
    assert retained_mask(n_pos, 5).tolist() == [False, False, True, True,
                                                True, True, True, True]
    assert retained_mask(n_pos, 10).tolist() == [False, False, False, False,
                                                 True, True, True, True]


def test_T1_retains_every_concept_with_at_least_one_positive():
    """T=1 IS the unrestricted point, and that is a property of the data (every
    in-pool concept has >= 1 observed positive), not a special case in code."""
    n_pos = np.array([1, 2, 10520])
    assert retained_mask(n_pos, 1).all()


def test_retained_mask_REFUSES_a_threshold_below_one():
    with pytest.raises(ValueError, match="not a restriction"):
        retained_mask(np.array([1, 2]), 0)


def test_retained_mask_REFUSES_a_2D_positive_count():
    with pytest.raises(ValueError, match="expected 1-D"):
        retained_mask(np.zeros((2, 3)), 5)


# ---------------------------------------------------------------------------
# restricted_reject -- the family, and the loosening
# ---------------------------------------------------------------------------

def test_a_test_outside_the_retained_pool_is_FALSE_not_a_non_rejection():
    """A concept outside the pool is not a failed test; it is not a test. The
    distinction is invisible in the count and decides what ALIGNED means."""
    p = np.full((4, 3), 1e-9)
    ne = np.array([False, False, False])
    keep = np.array([True, False, True])
    r = restricted_reject(p, ne, keep, q=0.05)
    assert not r["reject"][:, 1].any()
    assert r["reject"][:, 0].all() and r["reject"][:, 2].all()
    assert r["n_tests"] == 8            # 4 features x 2 retained concepts
    assert r["n_concepts_in_family"] == 2


def test_the_family_excludes_a_NOT_EVALUABLE_concept_even_when_retained():
    """Both halves of `keep AND evaluable`. A concept with no reference
    distribution has no tests at any grid point (BH3), and retaining it must
    not put NaNs into the family."""
    p = np.full((4, 3), 1e-9)
    p[:, 1] = np.nan
    ne = np.array([False, True, False])
    keep = np.array([True, True, True])
    r = restricted_reject(p, ne, keep, q=0.05)
    assert r["n_tests"] == 8
    assert r["n_concepts_in_family"] == 2
    assert r["n_concepts_retained"] == 3
    assert r["n_concepts_retained_not_evaluable"] == 1
    assert not r["reject"][:, 1].any()


def test_an_empty_family_rejects_nothing_and_says_so():
    """Total over the argument (gotcha 24): every concept dropped is a state
    the grid can reach, and it must return a shaped answer, not raise."""
    p = np.full((4, 3), 1e-9)
    r = restricted_reject(p, np.zeros(3, bool), np.zeros(3, bool), q=0.05)
    assert r["n_tests"] == 0 and r["n_rejected"] == 0
    assert r["threshold"] is None
    assert r["reject"].shape == (4, 3) and not r["reject"].any()


def test_THE_LOOSENING_a_p_value_refused_in_a_big_family_passes_in_a_small_one():
    """THE MECHANISM BI3(b) NAMES, worked by hand rather than by calling the
    code under test.

    Family of 10 tests, q = 0.05. The p-values are 0.004 and nine copies of
    0.9. BH sorts them: rank 1 has critical value (1/10)*0.05 = 0.005, and
    0.004 <= 0.005, so 0.004 IS rejected -- but make the family bigger and it
    is not. With 20 tests rank 1's critical value is 0.0025 and 0.004 > 0.0025,
    and no larger rank crosses either, so NOTHING is rejected.

    Same p-value, same q, different family: refused at 20 tests, accepted at
    10. That is exactly the loosening, and the two calls below differ ONLY in
    `keep`."""
    p = np.full((10, 2), 0.9)
    p[0, 0] = 0.004
    ne = np.zeros(2, dtype=bool)

    wide = restricted_reject(p, ne, np.array([True, True]), q=0.05)
    assert wide["n_tests"] == 20
    assert wide["n_rejected"] == 0
    assert not wide["reject"][0, 0]

    narrow = restricted_reject(p, ne, np.array([True, False]), q=0.05)
    assert narrow["n_tests"] == 10
    assert narrow["n_rejected"] == 1
    assert narrow["reject"][0, 0]


def test_an_unrestricted_call_matches_bh_reject_on_the_whole_family():
    """The T=1 identity, on a fixture: with every concept retained and
    evaluable, the restricted path must be the unrestricted one. This is the
    unit-level form of the real regression baseline."""
    rng = np.random.default_rng(0)
    p = rng.random((32, 5))
    ne = np.zeros(5, dtype=bool)
    r = restricted_reject(p, ne, np.ones(5, dtype=bool), q=0.05)
    direct = bh_reject(p.reshape(-1), q=0.05)
    assert r["n_tests"] == direct["n_tests"]
    assert r["n_rejected"] == direct["n_rejected"]
    assert np.array_equal(r["reject"].reshape(-1), direct["reject"])


def test_restricted_reject_REFUSES_a_mask_on_the_wrong_axis():
    with pytest.raises(ValueError, match="must be 1-D over p's concept axis"):
        restricted_reject(np.zeros((4, 3)), np.zeros(4, bool),
                          np.zeros(3, bool))


# ---------------------------------------------------------------------------
# decompose_aligned -- the counterfactual
# ---------------------------------------------------------------------------

def _mask(*idx, n=10):
    m = np.zeros(n, dtype=bool)
    for i in idx:
        m[i] = True
    return m


def test_the_decomposition_separates_the_two_causes_counted_by_hand():
    """Six features, all counts written out.

    prev = {0,1,2,3}; mid = {0,1,2}  -> feature 3 left because its concept was
    removed, so lost_pool = 1.
    next = {0,1,4}                   -> feature 2 left because the cut moved
    (lost_threshold = 1) and feature 4 entered because it loosened
    (gained_threshold = 1).

    Two-way: lost = {2,3} = 2, gained = {4} = 1, net = 3 - 4 = -1.
    Three-way: 1 - 1 - 1 = -1. Both agree, and they are DIFFERENT splits of
    the same movement."""
    prev, mid, nxt = _mask(0, 1, 2, 3), _mask(0, 1, 2), _mask(0, 1, 4)
    d = decompose_aligned(prev, mid, nxt)
    assert d["lost_pool"] == 1
    assert d["lost_threshold"] == 1
    assert d["gained_threshold"] == 1
    assert d["gained_pool"] == 0
    assert d["lost"] == 2 and d["gained"] == 1 and d["net"] == -1
    assert d["n_aligned_prev"] == 4 and d["n_aligned_next"] == 3
    assert d["n_aligned_counterfactual"] == 3


def test_a_counterfactual_that_ADDS_a_feature_is_REFUSED():
    """THE ONLY PART OF THE DECOMPOSITION THAT CAN FAIL, and therefore the one
    that carries the evidence.

    Removing concepts from the pool cannot make a feature ALIGNED, so `mid`
    must be a subset of `prev`. A `mid` that is not means the counterfactual
    did not hold the rejections fixed -- and then the causal split is not the
    split it claims to be, however neatly the totals add up."""
    prev, mid, nxt = _mask(0, 1), _mask(0, 1, 2), _mask(0, 1, 2)
    with pytest.raises(DecompositionViolation, match="cannot add one"):
        decompose_aligned(prev, mid, nxt)


def test_no_movement_at_all_is_all_zeros_and_does_not_raise():
    """A grid step where nothing moves is BM-E2b cell 3 (UNINFORMATIVE), a real
    outcome that must be reportable rather than an error."""
    m = _mask(0, 1, 2)
    d = decompose_aligned(m, m, m)
    assert (d["lost"], d["gained"], d["net"]) == (0, 0, 0)
    assert (d["lost_pool"], d["lost_threshold"], d["gained_threshold"]) == \
        (0, 0, 0)


def test_pure_pool_removal_attributes_everything_to_the_pool():
    """The other extreme: the cut does not move, so all movement is removal.
    `gained_threshold` is 0 here, which is BM-E2b's MISS branch if it held at
    every step -- so the fixture that can produce it has to exist."""
    prev, mid, nxt = _mask(0, 1, 2), _mask(0, 1), _mask(0, 1)
    d = decompose_aligned(prev, mid, nxt)
    assert d["lost_pool"] == 1 and d["lost_threshold"] == 0
    assert d["gained_threshold"] == 0
    assert d["gained"] == 0 and d["lost"] == 1 and d["net"] == -1


def test_decompose_REFUSES_masks_of_different_shape():
    with pytest.raises(ValueError, match="1-D and the same shape"):
        decompose_aligned(np.zeros(3, bool), np.zeros(4, bool),
                          np.zeros(3, bool))


# ---------------------------------------------------------------------------
# clearance
# ---------------------------------------------------------------------------

def test_clearance_reports_the_equivalence_as_CONDITIONAL_not_as_a_fact():
    """`total / k_min` equals the mean per concept only while `k_min == C`.
    That holds at every grid point here because q*S = 1 exactly, but the
    function must SAY which case it is in rather than let the two readings
    diverge silently behind one number."""
    same = clearance(436, 218, 218)
    assert same["clearance"] == pytest.approx(2.0)
    assert same["mean_at_floor_per_concept"] == pytest.approx(2.0)
    assert same["clearance_equals_mean_per_concept"] is True

    diff = clearance(436, 218, 100)
    assert diff["clearance"] == pytest.approx(2.0)
    assert diff["mean_at_floor_per_concept"] == pytest.approx(4.36)
    assert diff["clearance_equals_mean_per_concept"] is False


def test_clearance_REFUSES_a_non_positive_floor():
    with pytest.raises(ValueError, match="non-positive floor"):
        clearance(10, 0, 10)
