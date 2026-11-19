"""Tests for amendment BO's components 1 and 2.

EVERY EXPECTATION IN THIS FILE IS WRITTEN DOWN, NOT COMPUTED FROM THE CODE
UNDER TEST (gotcha 32b, BM-E1 criterion 2). Where an expected p-value is a
fraction, the fraction is spelled out with its numerator and denominator so a
reader can check the arithmetic without running anything.

THE INDEPENDENT IMPLEMENTATIONS AT THE TOP ARE WRITTEN FROM THE SPECIFICATION
(BM-E1 criterion 7). They are deliberately naive -- explicit Python loops, no
numpy vectorisation, no shared helper with `classification_bh` -- because an
independent implementation that imports the thing it checks is the instrument
under test wearing a second name.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.classification_bh import (  # noqa: E402
    NotEvaluable,
    bh_pass_one_slab,
    bh_reject,
    pooled_exceedance,
)

# ==========================================================================
# INDEPENDENT IMPLEMENTATIONS -- from the SPECIFICATION, not from the module.
# ==========================================================================


def ref_exceedance(pool, obs):
    """Spec: p = (#{null >= obs} + 1) / (n_finite + 1), NaN dropped (BH2).

    Written as the definition reads, in plain Python.
    """
    finite = [v for v in pool if v == v and abs(v) != float("inf")]
    n = len(finite)
    if n == 0:
        raise ValueError("NOT_EVALUABLE")
    out = []
    for o in obs:
        n_ge = sum(1 for v in finite if v >= o)
        out.append((n_ge + 1.0) / (n + 1.0))
    return out, n


def ref_bh(pvals, q):
    """Spec: reject H_(i) for all i <= k, k = LARGEST i with p_(i) <= (i/m)q.

    Written as a step-up, exactly as Benjamini & Hochberg define it.
    """
    m = len(pvals)
    if m == 0:
        return 0, None
    srt = sorted(pvals)
    k = 0
    for i in range(1, m + 1):
        if srt[i - 1] <= (i / m) * q:
            k = i                      # keep going: LARGEST, not first
    return k, (srt[k - 1] if k else None)


# ==========================================================================
# COMPONENT 2 -- pooled exceedance
# ==========================================================================


def test_exceedance_hand_computed_fraction():
    """Pool = 0..9 (ten draws). obs = 5.0.

    Draws >= 5.0 are 5,6,7,8,9 -> FIVE. So p = (5+1)/(10+1) = 6/11.
    Written down, not computed from the module.
    """
    pool = np.arange(10, dtype=np.float64)
    p, n, res = pooled_exceedance(pool, np.array([5.0]))
    assert n == 10
    assert p[0] == pytest.approx(6.0 / 11.0)
    assert res == pytest.approx(1.0 / 11.0)


def test_exceedance_is_GE_not_GT():
    """The convention is `>=` (AW1.2 strict exceedance). This test is the
    one that distinguishes the two, and it is the reason it exists.

    Pool = [1,2,3], obs = 2.0. Under `>=`: draws 2 and 3 -> 2, p = 3/4.
    Under `>`:  draw 3 only     -> 1, p = 2/4.
    """
    p, _, _ = pooled_exceedance(np.array([1.0, 2.0, 3.0]), np.array([2.0]))
    assert p[0] == pytest.approx(3.0 / 4.0)
    assert p[0] != pytest.approx(2.0 / 4.0)


def test_exceedance_ceiling_is_never_zero():
    """An observed value above every draw still gets 1/(n+1), never 0.

    Pool = 0..9, obs = 100.0 -> p = (0+1)/11 = 1/11. A statistic can never be
    reported as impossible on finite draws.
    """
    p, _, _ = pooled_exceedance(np.arange(10, dtype=np.float64),
                                np.array([100.0]))
    assert p[0] == pytest.approx(1.0 / 11.0)
    assert p[0] > 0.0


def test_degenerate_draws_are_DROPPED_and_shrink_the_denominator():
    """BH2: a degenerate draw is NaN, dropped, never imputed.

    Pool = [1,2,3,NaN,NaN], obs = 2.0. Finite n = 3, draws >= 2 are {2,3} = 2.
    So p = (2+1)/(3+1) = 3/4 -- NOT (2+1)/(5+1) = 1/2, which is what treating
    NaN as a draw would give, and NOT 4/6 which is what imputing 0 would give.
    """
    pool = np.array([1.0, 2.0, 3.0, np.nan, np.nan])
    p, n, res = pooled_exceedance(pool, np.array([2.0]))
    assert n == 3
    assert p[0] == pytest.approx(3.0 / 4.0)
    assert res == pytest.approx(1.0 / 4.0)


def test_all_degenerate_raises_NotEvaluable_rather_than_returning_a_value():
    """BH3: every draw degenerate -> NOT_EVALUABLE, never a number.

    A returned NaN would be one `nanmean` away from a silent non-detection.
    """
    with pytest.raises(NotEvaluable):
        pooled_exceedance(np.array([np.nan, np.nan]), np.array([1.0]))


def test_exceedance_matches_the_independent_implementation():
    """CRITERION 7, at the function level.

    THE POOL IS DISCRETISED ON PURPOSE, AND THE TIES ARE ASSERTED TO EXIST.
    A pool of continuous normals has probability zero of an exact tie with an
    observed value, so `>=` and `>` agree everywhere and the comparison is
    BLIND to the tie convention -- which is exactly what the BO mutation
    matrix caught this fixture doing. A fixture that cannot exhibit the branch
    it covers is not evidence about that branch (criterion 2).
    """
    rng = np.random.default_rng(11)
    pool = np.round(rng.normal(size=500), 1)          # ties are possible
    pool[rng.choice(500, 25, replace=False)] = np.nan
    obs = np.round(rng.normal(size=40), 1)            # drawn on the SAME grid

    finite = pool[np.isfinite(pool)]
    n_ties = sum(int((finite == o).sum()) for o in obs)
    assert n_ties > 0, (
        "this fixture contains no obs/pool tie, so it cannot distinguish >= "
        "from > and its agreement would not be evidence about the convention")

    got, n, _ = pooled_exceedance(pool, obs)
    want, n_ref = ref_exceedance(pool.tolist(), obs.tolist())
    assert n == n_ref == 475
    assert np.allclose(got, want, rtol=0, atol=0)


# ==========================================================================
# COMPONENT 1 -- Benjamini-Hochberg
# ==========================================================================


def test_bh_is_a_STEP_UP_not_a_scan_until_first_failure():
    """THE FIXTURE THAT DISCRIMINATES THE CLASSIC BH BUG.

    m = 5, q = 0.05 -> crit = [0.01, 0.02, 0.03, 0.04, 0.05].
    Sorted p       = [0.005, 0.025, 0.030, 0.900, 0.900].
      i=1: 0.005 <= 0.01  PASS
      i=2: 0.025 <= 0.02  FAIL   <- a scan-until-failure stops here, k = 1
      i=3: 0.030 <= 0.03  PASS   <- the LARGEST crossing, so k = 3
      i=4: 0.900 <= 0.04  FAIL
      i=5: 0.900 <= 0.05  FAIL
    Correct BH rejects THREE, including p = 0.025 which fails its own
    threshold. A scan-until-failure rejects ONE. Both are plausible; only one
    is BH.
    """
    p = np.array([0.005, 0.025, 0.030, 0.900, 0.900])
    r = bh_reject(p, q=0.05)
    assert r["n_rejected"] == 3
    assert r["n_rejected"] != 1
    assert r["threshold"] == pytest.approx(0.030)
    assert r["reject"].tolist() == [True, True, True, False, False]


def test_bh_critical_values_are_RANK_DEPENDENT_not_a_flat_q():
    """The step-up fixture above happens to give the SAME answer under a flat
    `q` threshold, so it cannot see a critical-value defect. This one can, and
    it exists because the mutation matrix showed the other one could not.

    m = 5, q = 0.05 -> crit = [0.01, 0.02, 0.03, 0.04, 0.05].
    Sorted p       = [0.045, 0.9, 0.9, 0.9, 0.9].
      correct BH : 0.045 <= 0.01?  no. Nothing else crosses -> reject 0.
      flat q     : 0.045 <= 0.05?  yes                      -> reject 1.
    """
    p = np.array([0.045, 0.9, 0.9, 0.9, 0.9])
    r = bh_reject(p, q=0.05)
    assert r["n_rejected"] == 0
    assert r["n_rejected"] != 1
    assert r["threshold"] is None


def test_bh_rejects_nothing_when_no_p_crosses():
    """Total function over the outcome space: the empty rejection set is a
    real cell and reports `threshold = None`, not 0.0 (gotcha 25b -- a
    constant reported as False reads as 'checked and absent')."""
    r = bh_reject(np.array([0.9, 0.8, 0.7]), q=0.05)
    assert r["n_rejected"] == 0
    assert r["threshold"] is None
    assert not r["reject"].any()


def test_bh_rejects_everything_when_all_p_are_tiny():
    r = bh_reject(np.array([1e-9, 1e-9, 1e-9, 1e-9]), q=0.05)
    assert r["n_rejected"] == 4
    assert r["reject"].all()


def test_bh_empty_family_is_a_mapped_cell():
    r = bh_reject(np.array([]), q=0.05)
    assert r["n_tests"] == 0 and r["n_rejected"] == 0
    assert r["threshold"] is None


def test_bh_reject_mask_is_in_INPUT_order_not_sorted_order():
    """A mask returned in sorted order would be a permutation defect: same
    total, wrong positions, and no count check could see it (gotcha 27)."""
    p = np.array([0.9, 0.001, 0.9, 0.002])
    r = bh_reject(p, q=0.05)
    assert r["reject"].tolist() == [False, True, False, True]


def test_bh_matches_the_independent_implementation():
    """CRITERION 7, at the function level, over many random families."""
    rng = np.random.default_rng(3)
    for _ in range(60):
        m = int(rng.integers(1, 400))
        p = rng.random(m) ** 3
        got = bh_reject(p, q=0.05)
        k, thr = ref_bh(p.tolist(), 0.05)
        assert got["n_rejected"] == k
        if k:
            assert got["threshold"] == pytest.approx(thr)
        else:
            assert got["threshold"] is None


def test_the_agreement_check_CAN_DETECT_A_PLANTED_DISAGREEMENT():
    """THE CHECK ON THE CHECK (gotcha 13, criterion 7).

    An agreement between two implementations is evidence only once the
    comparison has been shown able to report a disagreement. Here the planted
    defect is the scan-until-first-failure bug, implemented explicitly, and
    the assertion is that the comparison CATCHES it.
    """
    def planted_bh(pvals, q):           # the defect, deliberately
        m = len(pvals)
        srt = sorted(pvals)
        k = 0
        for i in range(1, m + 1):
            if srt[i - 1] <= (i / m) * q:
                k = i
            else:
                break                   # <-- stops at the FIRST failure
        return k, (srt[k - 1] if k else None)

    p = [0.005, 0.025, 0.030, 0.900, 0.900]
    correct_k, _ = ref_bh(p, 0.05)
    planted_k, _ = planted_bh(p, 0.05)
    assert correct_k == 3 and planted_k == 1
    assert correct_k != planted_k, (
        "the planted disagreement did not disagree -- this comparison cannot "
        "detect the defect it claims to, so any agreement it reports is void")


# ==========================================================================
# THE SLAB DRIVER, and the founding / degenerate instances
# ==========================================================================


def _slab(n_shifts, n_c, n_f, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_shifts, n_c, n_f))


def test_slab_family_excludes_NOT_EVALUABLE_concepts_from_m():
    """BH3 + criterion 6: a NOT_EVALUABLE concept's tests never enter the
    family, are never counted as non-detections, and its count is carried."""
    n_shifts, n_c, n_f = 4, 3, 10
    slab = _slab(n_shifts, n_c, n_f)
    g = np.zeros((n_f, n_c))
    ev = np.array([True, False, True])
    r = bh_pass_one_slab(slab, g, ev)
    # m is over TWO concepts, not three: 2 * 10 = 20.
    assert r["n_tests"] == 20
    assert r["n_concepts_not_evaluable"] == 1
    assert r["n_concepts_in_family"] == 2
    assert np.isnan(r["p"][:, 1]).all()
    assert not r["reject"][:, 1].any()


def test_slab_carries_NOT_EVALUABLE_as_zero_when_empty():
    """Criterion 6 / BI4: the category is reported EVEN WHEN EMPTY. An empty
    category printed as 0 is a measurement; an absent one is a lost
    distinction."""
    slab = _slab(4, 3, 10)
    r = bh_pass_one_slab(slab, np.zeros((10, 3)), np.ones(3, dtype=bool))
    assert r["n_concepts_not_evaluable"] == 0
    assert "n_concepts_not_evaluable" in r


def test_slab_refuses_a_TRANSPOSED_observed_array():
    """A transposed g_obs would broadcast and produce a complete, plausible,
    wrong result. It must refuse instead."""
    slab = _slab(4, 3, 10)
    with pytest.raises(ValueError, match="transposed|expected"):
        bh_pass_one_slab(slab, np.zeros((3, 10)), np.ones(3, dtype=bool))


def test_THE_DEGENERATE_CELL_flows_through_as_a_COARSENED_DENOMINATOR():
    """THE FOUNDING INSTANCE (criterion 4), reproduced in miniature.

    `BINDING:hypoxanthine`, Null B: 5 observed positives, ONE of 20 shifts
    degenerate (shift 4), so 19 surviving draws. Read from
    `null_experimental_bg.h5`: surviving_draws_b = 19, bh3_case =
    SOME_BUT_NOT_ALL_DEGENERATE, not_evaluable_b = False.

    THE POINT OF THIS TEST: it is *not* NOT_EVALUABLE. BH3's SOME-BUT-NOT-ALL
    row says drop the degenerate draws and USE THE SURVIVORS. The concept
    stays in the family with a coarsened denominator.

    Here: 20 shifts x 10 features, shift 4 entirely NaN for concept 0.
    Finite draws = 19 * 10 = 190, so the denominator is 191 and the
    resolution is 1/191 -- against 1/201 for an undamaged concept.
    """
    n_shifts, n_c, n_f = 20, 2, 10
    slab = _slab(n_shifts, n_c, n_f, seed=7)
    slab[4, 0, :] = np.nan                       # concept 0, shift 4
    r = bh_pass_one_slab(slab, np.zeros((n_f, n_c)), np.ones(n_c, dtype=bool))

    assert r["not_evaluable"][0] == False, (          # noqa: E712
        "the degenerate cell must NOT be NOT_EVALUABLE -- it has survivors")
    assert r["n_finite"][0] == 190
    assert r["n_finite"][1] == 200
    assert r["resolution"][0] == pytest.approx(1.0 / 191.0)
    assert r["resolution"][1] == pytest.approx(1.0 / 201.0)
    # and it is STILL IN THE FAMILY: m counts both concepts.
    assert r["n_tests"] == n_f * n_c
    assert r["n_concepts_not_evaluable"] == 0


def test_a_wholly_degenerate_concept_IS_NOT_EVALUABLE_and_leaves_the_family():
    """The other side of BH3's table, so the pair of tests spans the rule."""
    n_shifts, n_c, n_f = 20, 2, 10
    slab = _slab(n_shifts, n_c, n_f, seed=8)
    slab[:, 0, :] = np.nan                       # EVERY shift, concept 0
    r = bh_pass_one_slab(slab, np.zeros((n_f, n_c)), np.ones(n_c, dtype=bool))
    assert r["not_evaluable"][0] == True          # noqa: E712
    assert r["n_concepts_not_evaluable"] == 1
    assert r["n_tests"] == n_f * 1
    assert np.isnan(r["p"][:, 0]).all()


def test_slab_agrees_with_the_independent_implementation_end_to_end():
    """CRITERION 7 at the SLAB level: the whole pass, against the naive spec
    implementation, on a slice small enough to hold in memory."""
    n_shifts, n_c, n_f = 6, 4, 12
    slab = _slab(n_shifts, n_c, n_f, seed=21)
    slab[2, 1, :] = np.nan
    rng = np.random.default_rng(5)
    g = rng.normal(size=(n_f, n_c))
    # THE FIXTURE IS FORCED TO PRODUCE REJECTIONS, AND TO PUT THEM AT THE END.
    # Two separate fixture defects the mutation matrix found here, in
    # sequence. First: with purely random observed gains this family rejects
    # NOTHING at m = 48, so the mask comparison below was unreachable.
    # Second: driving the FIRST five features above every null draw put the
    # true rejections at flat positions 0..19 -- exactly where a mask written
    # in sorted order also puts them, so the two agreed by coincidence. The
    # extreme features are therefore the LAST five, at flat positions 28..47.
    g[7:, :] = 1e6
    ev = np.ones(n_c, dtype=bool)
    got = bh_pass_one_slab(slab, g, ev, q=0.05)

    # independent: per concept, then one BH over the flat family
    ref_p = np.empty((n_f, n_c))
    for c in range(n_c):
        pool = slab[:, c, :].reshape(-1).tolist()
        col, _ = ref_exceedance(pool, g[:, c].tolist())
        ref_p[:, c] = col
    assert np.allclose(got["p"], ref_p, rtol=0, atol=0)
    k, thr = ref_bh(ref_p.reshape(-1).tolist(), 0.05)
    assert got["n_rejected"] == k
    assert k > 0, (
        "this fixture rejects nothing, so every assertion below it is "
        "unreachable and its agreement says nothing about the reject mask")
    if k:
        assert got["threshold"] == pytest.approx(thr)
        # THE MASK IS COMPARED POSITIONALLY, not only by its total. A count
        # check cannot detect a permutation (gotcha 27), and the mutation
        # matrix showed this fixture passing against a mask written in SORTED
        # order -- same total, wrong positions.
        want_mask = ref_p <= thr
        assert got["reject"].tolist() == want_mask.tolist()
        assert int(want_mask.sum()) == k


def test_the_slab_agreement_check_CAN_DETECT_A_PLANTED_DISAGREEMENT():
    """The slab-level check on the check. The planted defect is BH2's
    inverse -- imputing 0.0 for a degenerate draw instead of dropping it --
    and the assertion is that the comparison SEES it."""
    n_shifts, n_c, n_f = 6, 2, 8
    slab = _slab(n_shifts, n_c, n_f, seed=31)
    slab[1, 0, :] = np.nan
    g = np.zeros((n_f, n_c))

    got = bh_pass_one_slab(slab, g, np.ones(n_c, dtype=bool))

    planted = slab.copy()
    planted[np.isnan(planted)] = 0.0             # <-- the imputation BH2 bans
    ref_p = np.empty((n_f, n_c))
    for c in range(n_c):
        col, _ = ref_exceedance(planted[:, c, :].reshape(-1).tolist(),
                                g[:, c].tolist())
        ref_p[:, c] = col
    assert not np.allclose(got["p"], ref_p), (
        "imputing 0.0 for a degenerate draw produced the SAME p-values as "
        "dropping it -- this fixture cannot discriminate BH2's rule, so its "
        "agreement is not evidence")
