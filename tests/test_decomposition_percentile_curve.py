"""Tests for the AW1.5 percentile curve over the BW decomposition (amendment BY).

EXPECTATIONS ARE WRITTEN DOWN, NOT COMPUTED FROM THE MODULE UNDER TEST
(gotcha 32b). Every fixture below is small enough that its detection counts are
worked out by hand in the docstring, and the assertion checks that hand figure.

THE FIXTURES MUST DISCRIMINATE (gotchas 13, 39). A fixture on which the
conclusion holds cannot detect a verdict function that always returns
HOLDS_ACROSS_RANGE, so there is a fixture for each branch of the outcome space
INCLUDING the failing one and the no-estimable-percentile one.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.concept_repro import CURVE_PERCENTILES, REFERENCE_PERCENTILE
from src.analysis.decomposition_percentile_curve import (
    CONCLUSION,
    POOLING_MODE,
    conclusion_across_range,
    curve_grid,
)


# --------------------------------------------------------------- the range

def test_range_is_imported_not_redeclared():
    """AW1.5's six percentiles, and the module must not carry its own copy.

    Written down: 50, 75, 90, 95, 99, 99.9 -- read off the clause, not the code.
    """
    assert tuple(CURVE_PERCENTILES) == (50.0, 75.0, 90.0, 95.0, 99.0, 99.9)
    assert REFERENCE_PERCENTILE == 95.0


def test_pooling_mode_is_pooled_per_AX2():
    """AX2 settled AW8 item 2: the POOLED null is the reference."""
    assert POOLING_MODE == "pooled"


# ------------------------------------------------------------- estimability

def test_999_is_excluded_at_400_draws_and_marked_excluded():
    """400 pooled draws: 400*(100-99.9)/100 = 0.4 < 1, so the 99.9th is NOT
    estimable and must be present-and-excluded, never absent and never
    interpolated.

    And the 99th IS estimable at 400: 400*0.01 = 4.0 >= 1.
    """
    rng = np.random.default_rng(0)
    best = rng.random((20, 7))
    nw = rng.random((20, 20, 7))
    na = rng.random((20, 20, 7))
    g = curve_grid(best, nw, na)
    assert g["pooled_draws"] == 400

    by_q = {r["percentile"]: r for r in g["rows"]}
    # every specified percentile is PRESENT
    assert set(by_q) == set(float(q) for q in CURVE_PERCENTILES)

    assert by_q[99.9]["estimable"] is False
    assert by_q[99.9]["excluded"] is True
    assert by_q[99.9]["expected_draws_at_or_above"] == pytest.approx(0.4)
    assert "arms" not in by_q[99.9]          # no figures for an excluded row
    assert "not estimable" in by_q[99.9]["exclusion_reason"].lower()

    assert by_q[99.0]["estimable"] is True
    assert by_q[99.0]["expected_draws_at_or_above"] == pytest.approx(4.0)
    assert by_q[99.0]["excluded"] is False


def test_per_seed_draw_count_would_exclude_the_99th_too():
    """A 20-draw sample: 20*0.01 = 0.2 < 1. Guards the estimability rule against
    being insensitive to the draw count -- if it were, the pooling decision of
    AX2 would carry no consequence."""
    rng = np.random.default_rng(1)
    best = rng.random((20, 4))
    nw = rng.random((20, 1, 4))           # 20 seeds x 1 shift = 20 pooled draws
    na = rng.random((20, 1, 4))
    g = curve_grid(best, nw, na)
    assert g["pooled_draws"] == 20
    by_q = {r["percentile"]: r for r in g["rows"]}
    assert by_q[99.0]["estimable"] is False
    assert by_q[95.0]["estimable"] is True    # 20*0.05 = 1.0, exactly at the bound


def test_the_95th_bound_is_computed_in_the_stable_form():
    """AX2's numerical note: at n=20, q=95 the unstable form gives
    1.0000000000000009 and the stable one gives exactly 1.0. Both are >= 1 so
    the verdict agrees here, but the assertion pins the value, because the
    boundary is what decides whether the reference percentile is supported."""
    rng = np.random.default_rng(2)
    g = curve_grid(rng.random((20, 3)), rng.random((20, 1, 3)),
                   rng.random((20, 1, 3)))
    row = {r["percentile"]: r for r in g["rows"]}[95.0]
    assert row["expected_draws_at_or_above"] == 1.0


# ------------------------------------------------------- both arms move together

def test_both_arms_recomputed_at_each_percentile():
    """A hand-built case. One concept, 2 seeds, 10 shifts = 20 pooled draws --
    the smallest count at which the 95th reference is estimable (20*0.05 = 1.0).

    within-protein draws are all 0.9; across-protein draws are all 0.1.
    observed best_gain = 0.5 for both seeds.

    So at EVERY estimable percentile: 0.5 > 0.1 (detected across) and
    0.5 < 0.9 (not detected within). Hand figures: D_across = 1.0,
    D_within = 0.0, cell share across&~within = 2/2 = 1.0.
    """
    best = np.full((2, 1), 0.5)
    nw = np.full((2, 10, 1), 0.9)
    na = np.full((2, 10, 1), 0.1)
    g = curve_grid(best, nw, na, percentiles=(50.0, 75.0, 90.0, 95.0))
    for r in g["rows"]:
        assert r["estimable"] is True, r
        assert r["arms"]["across_protein"]["mean_D"] == 1.0
        assert r["arms"]["within_protein"]["mean_D"] == 0.0
        assert r["cell_basis"]["denominator_cells_detected_across"] == 2
        assert r["cell_basis"]["n_cells_across_not_within"] == 2
        assert r["cell_basis"]["share"] == 1.0
        assert r["concept_basis_D_across_eq_1"]["denominator_concepts_D_across_eq_1"] == 1
        assert r["concept_basis_D_across_eq_1"]["share_not_within_all"] == 1.0


def test_thresholds_move_with_the_percentile():
    """A spread null must give a HIGHER threshold at a higher percentile.

    Draws 0..19 for one concept: the 50th is 9.5, the 90th 17.1. Written down:
    the 90th threshold strictly exceeds the 50th.
    """
    best = np.full((2, 1), 100.0)
    draws = np.arange(20, dtype=float).reshape(2, 10, 1)
    g = curve_grid(best, draws, draws, percentiles=(50.0, 90.0))
    by_q = {r["percentile"]: r for r in g["rows"]}
    t50 = by_q[50.0]["arms"]["within_protein"]["median_threshold"]
    t90 = by_q[90.0]["arms"]["within_protein"]["median_threshold"]
    assert t90 > t50


def test_BOTH_arms_thresholds_move_with_the_percentile():
    """THE ACROSS ARM MOVES TOO -- asserted separately, because a fixture that
    checks one arm cannot detect the other being pinned.

    Found by the BY mutation matrix: mutation 2 pinned `thr_a` to the reference
    percentile while the within arm swept, and the single-arm fixture above
    could not see it. Two arms compared at DIFFERENT percentiles would make any
    difference between them partly a difference of cutoff -- the exact error
    this module's "both arms move together" property exists to exclude -- and it
    is invisible in the output because both columns stay populated.

    Hand figures, draws 0..19 for one concept: the 50th is 9.5 and the 90th is
    17.1, on BOTH arms, since both nulls are the same array here.
    """
    best = np.full((2, 1), 100.0)
    draws = np.arange(20, dtype=float).reshape(2, 10, 1)
    g = curve_grid(best, draws, draws, percentiles=(50.0, 90.0))
    by_q = {r["percentile"]: r for r in g["rows"]}
    for arm in ("within_protein", "across_protein"):
        t50 = by_q[50.0]["arms"][arm]["median_threshold"]
        t90 = by_q[90.0]["arms"][arm]["median_threshold"]
        assert t50 == pytest.approx(9.5), arm
        assert t90 == pytest.approx(17.1), arm
        assert t90 > t50, arm
    # and the two arms agree with each other at each percentile, which is what
    # "computed at the same percentile" means when the draws are identical
    for q in (50.0, 90.0):
        assert (by_q[q]["arms"]["within_protein"]["median_threshold"]
                == by_q[q]["arms"]["across_protein"]["median_threshold"])


# ------------------------------------------------- the verdict's outcome space

def test_verdict_holds_across_range():
    """The fixture of test_both_arms_recomputed: share is 1.0 everywhere, so
    the conclusion holds at every estimable percentile."""
    best = np.full((2, 1), 0.5)
    g = curve_grid(best, np.full((2, 10, 1), 0.9), np.full((2, 10, 1), 0.1),
                   percentiles=(50.0, 95.0))
    v = conclusion_across_range(g)
    # Both bases hold here: share is 1.0 on each.
    assert v["by_basis"]["cell"]["verdict"] == "HOLDS_ACROSS_RANGE"
    assert v["by_basis"]["concept"]["verdict"] == "HOLDS_ACROSS_RANGE"
    assert v["by_basis"]["cell"]["failing_percentiles"] == []
    assert v["basis_verdicts_agree"] is True
    assert v["no_basis_is_primary"] is True
    assert v["no_primary_declared"] is True
    assert v["conclusion"] == CONCLUSION


def test_verdict_fails_at_some_percentile_THE_DISCRIMINATING_FIXTURE():
    """A fixture on which the conclusion FAILS -- without this, a verdict
    function hardcoded to HOLDS_ACROSS_RANGE passes every other test here.

    Both nulls are identical, so thr_within == thr_across at every percentile
    and the across&~within cell is EMPTY by construction. Hand figure: share
    = 0/2 = 0.0 at every percentile, which is below any positive reading of
    'substantial'.
    """
    best = np.full((2, 1), 0.5)
    same = np.full((2, 10, 1), 0.1)
    g = curve_grid(best, same, same, percentiles=(50.0, 95.0))
    v = conclusion_across_range(g)
    assert v["by_basis"]["cell"]["verdict"] == "FAILS_AT_SOME_PERCENTILE"
    assert v["by_basis"]["cell"]["failing_percentiles"] == [50.0, 95.0]
    assert v["by_basis"]["concept"]["verdict"] == "FAILS_AT_SOME_PERCENTILE"
    for p in v["per_percentile"]:
        assert p["cell_share_across_not_within"] == 0.0


def test_verdict_fails_when_only_one_percentile_fails():
    """AW1.5 is a CONJUNCTION: holding at the reference is not enough.

    Constructed so the within-protein threshold exceeds best_gain at the 95th
    but not at the 50th. within draws: eighteen 0.1s and two 0.9s; best = 0.5.
    50th percentile of within = 0.1 -> 0.5 > 0.1, detected within (share 0).
    95th percentile of within = 0.9 -> 0.5 < 0.9, not detected within (share 1).
    So the conclusion fails at the 50th and holds at the 95th -> overall FAILS.
    """
    best = np.full((2, 1), 0.5)
    nw = np.array([0.1] * 18 + [0.9] * 2, dtype=float).reshape(2, 10, 1)
    na = np.full((2, 10, 1), 0.01)
    g = curve_grid(best, nw, na, percentiles=(50.0, 95.0))
    v = conclusion_across_range(g)
    assert v["by_basis"]["cell"]["verdict"] == "FAILS_AT_SOME_PERCENTILE"
    assert 50.0 in v["by_basis"]["cell"]["failing_percentiles"]
    assert 95.0 not in v["by_basis"]["cell"]["failing_percentiles"]


def test_verdict_no_estimable_percentile_is_its_own_branch():
    """gotcha 30: a branch with a distinct CAUSE gets a distinct reading. An
    uncomputable curve must not be reported as the conclusion failing."""
    rng = np.random.default_rng(3)
    # 1 seed x 1 shift = 1 pooled draw: nothing above the 50th is estimable.
    g = curve_grid(rng.random((1, 2)), rng.random((1, 1, 2)),
                   rng.random((1, 1, 2)), percentiles=(99.0, 99.9))
    v = conclusion_across_range(g)
    for key in ("cell", "concept"):
        assert v["by_basis"][key]["verdict"] == "NO_ESTIMABLE_PERCENTILE"
        assert "NOT a failure" in v["by_basis"][key]["reading"]
    assert v["per_percentile"] == []
    assert v["estimable_percentiles"] == []


def test_substantial_reading_is_explicit_and_changes_the_verdict():
    """The reading of 'substantial' is an argument, not a buried constant --
    and the shares are reported regardless, so any other reading is derivable.
    Hand figure: share is 1.0, so a reading of 1.5 cannot be met."""
    best = np.full((2, 1), 0.5)
    g = curve_grid(best, np.full((2, 10, 1), 0.9), np.full((2, 10, 1), 0.1),
                   percentiles=(95.0,))
    assert (conclusion_across_range(g, 0.10)["by_basis"]["cell"]["verdict"]
            == "HOLDS_ACROSS_RANGE")
    assert (conclusion_across_range(g, 1.5)["by_basis"]["cell"]["verdict"]
            == "FAILS_AT_SOME_PERCENTILE")


# ----------------------------------------------------- zero-denominator paths

def test_zero_denominator_reports_none_with_a_stated_reason():
    """gotcha 25: a share over an empty denominator is not 0.0 and not an error.
    Nothing is detected across-protein here, so both denominators are zero."""
    best = np.zeros((2, 1))
    na = np.full((2, 10, 1), 5.0)     # threshold far above observed
    nw = np.full((2, 10, 1), 5.0)
    g = curve_grid(best, nw, na, percentiles=(95.0,))
    r = g["rows"][0]
    assert r["cell_basis"]["denominator_cells_detected_across"] == 0
    assert r["cell_basis"]["share"] is None
    assert r["cell_basis"]["share_is_none_because"] == "denominator is zero"
    assert r["concept_basis_D_across_eq_1"]["share_not_within_all"] is None


# ------------------------------------------------------------- shape guards

def test_shape_guards_refuse_mismatched_axes():
    best = np.zeros((2, 3))
    with pytest.raises(ValueError, match="seed and concept axes"):
        curve_grid(best, np.zeros((2, 4, 5)), np.zeros((2, 4, 3)))
    with pytest.raises(ValueError, match="seed and concept axes"):
        curve_grid(best, np.zeros((3, 4, 3)), np.zeros((2, 4, 3)))
    with pytest.raises(ValueError, match=r"\(seed, concept\)"):
        curve_grid(np.zeros((2, 3, 4)), np.zeros((2, 4, 3)), np.zeros((2, 4, 3)))
    with pytest.raises(ValueError, match=r"\(seed, shift, concept\)"):
        curve_grid(best, np.zeros((2, 3)), np.zeros((2, 4, 3)))
