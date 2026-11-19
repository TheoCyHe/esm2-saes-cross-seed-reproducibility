"""Tests for M4's smooth-trend `mu_s` (amendment AC).

THE CENTRAL TEST IS RECOVERY OF A KNOWN SMOOTH FUNCTION. A penalised fit
returns plausible numbers from any input, so the only convincing check is:
generate `mu` from a function we chose, and require the fit to return it.

Mutations run and killed (see the AC commit message):
  * `exp(eta)` replaced by `eta` (identity link, so mu can go negative)
        -> kills test_fitted_mu_is_strictly_positive (+3 others)
  * the absorption verdict hardcoded to HIT
        -> kills test_absorption_verdict_can_say_MISS
  * a concept-side block indexed as if it were feature-side
        -> kills test_concept_side_only_fit_runs_and_is_constant_across_features

TWO MUTATIONS SURVIVED THE FIRST DRAFT AND THE TESTS WERE REWRITTEN. Recorded
here rather than quietly fixed, because a surviving mutation means the test
did not test what its own docstring claimed:
  * zeroing the difference penalty SURVIVED a bound of
    `2 < edf < basis_size` -- the identifiability ridge alone keeps edf just
    under the basis size. Replaced by a MONOTONICITY check across three
    lambdas, which a zeroed penalty cannot pass.
  * setting the Huber weights to all-ones SURVIVED an assertion on
    `huber_frac_downweighted` -- that diagnostic is computed from the
    residuals BEFORE the weights are applied, so disabling them leaves it
    untouched. Replaced by a check that the robust pass CHANGES THE
    COEFFICIENTS, plus a converse that it is a near-noop on clean data.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.smooth_mu import (
    CONCEPT_SIDE,
    COVARIATES,
    N_BASIS,
    _blocks,
    _order,
    adequacy,
    bspline_design,
    diff_penalty,
    fit_smooth_mu,
    predict_on_axis,
)

NF, NC = 200, 60


def _fixture(seed=0, nf=NF, nc=NC):
    rng = np.random.default_rng(seed)
    feat = {"perplexity": rng.uniform(0, 3, nf),
            "concentration": rng.uniform(0, 1, nf)}
    con = {"prevalence": rng.uniform(-6, -1, nc),
           "cluster_count": rng.uniform(0, 2, nc)}
    return feat, con, np.arange(nc)


def _truth(feat, con):
    """A smooth additive log-mu the fit should be able to recover."""
    f = -0.8 * feat["perplexity"] + 0.5 * np.sin(2 * feat["concentration"])
    c = 0.6 * con["prevalence"] + 0.3 * con["cluster_count"]
    return np.exp(c[:, None] + f[None, :] - 2.0)


# ---------------------------------------------------------------------------
# Basis and penalty
# ---------------------------------------------------------------------------

def test_bspline_basis_shape_and_partition_of_unity():
    x = np.linspace(0, 1, 500)
    B, knots = bspline_design(x)
    assert B.shape == (500, N_BASIS)
    # cubic B-splines sum to 1 on the interior
    inner = B[10:-10].sum(axis=1)
    assert np.allclose(inner, 1.0, atol=1e-8)


def test_knots_are_reusable_for_prediction():
    x = np.linspace(0, 1, 100)
    B1, knots = bspline_design(x)
    B2, _ = bspline_design(x, knots=knots)
    assert np.allclose(B1, B2)


def test_diff_penalty_annihilates_a_linear_sequence():
    """A second-difference penalty must cost nothing for a straight line --
    that is what makes 'smooth' mean 'not wiggly' rather than 'not sloped'."""
    P = diff_penalty(8, order=2)
    lin = np.arange(8.0)
    assert float(lin @ P @ lin) == pytest.approx(0.0, abs=1e-20)
    wig = np.array([0, 1, 0, 1, 0, 1, 0, 1.0])
    assert float(wig @ P @ wig) > 1.0


# ---------------------------------------------------------------------------
# Recovery of a known truth
# ---------------------------------------------------------------------------

def test_recovers_a_known_smooth_mu():
    feat, con, pool = _fixture()
    mu_true = _truth(feat, con)
    rng = np.random.default_rng(1)
    y = rng.poisson(mu_true * 1e4) / 1e4        # low-noise Poisson draws
    blocks, knots = _blocks(feat, con, pool)
    res = fit_smooth_mu(y, blocks, robust=False)
    mu_hat = res["mu_hat"]
    lr = np.log(mu_hat / mu_true)
    assert np.abs(lr - lr.mean()).max() < 0.35, np.abs(lr - lr.mean()).max()
    assert np.corrcoef(np.log(mu_hat).ravel(),
                       np.log(mu_true).ravel())[0, 1] > 0.99


def test_fitted_mu_is_strictly_positive():
    """`mu_hat` is a DENOMINATOR. A log link guarantees positivity; an
    identity link would not, and zeros in the response make that live."""
    feat, con, pool = _fixture(seed=2)
    y = _truth(feat, con)
    y[::3] = 0.0                                 # plenty of exact zeros
    blocks, _ = _blocks(feat, con, pool)
    res = fit_smooth_mu(y, blocks, robust=False)
    assert res["mu_hat"].min() > 0.0


def test_zero_response_is_admissible():
    """An all-but-a-few-zeros response must still fit rather than blow up --
    this is the rare-concept regime the 22 degenerate concepts live in."""
    feat, con, pool = _fixture(seed=3)
    y = np.zeros((NC, NF))
    y[0, :5] = 1.0
    blocks, _ = _blocks(feat, con, pool)
    res = fit_smooth_mu(y, blocks, robust=False)
    assert np.isfinite(res["mu_hat"]).all() and res["mu_hat"].min() > 0


def test_penalty_controls_effective_df():
    """Effective df must fall as the penalty rises.

    A FIRST DRAFT of this test only asserted `2 < edf < basis_size`, and a
    mutation zeroing the difference penalty SURVIVED it -- because the tiny
    identifiability ridge still moves edf a little, so an unpenalised fit
    lands just under the basis size and passes a loose bound. The property
    that actually matters is MONOTONICITY: the penalty, not the knots,
    controls smoothness. Measured directly here.
    """
    feat, con, pool = _fixture(seed=4)
    y = _truth(feat, con)
    blocks, _ = _blocks(feat, con, pool)
    total = sum(blocks[n].shape[1] for n in _order(blocks))

    edfs = []
    for lam in (1e-2, 1e2, 1e6):
        res = fit_smooth_mu(y, blocks, robust=False, lambda_grid=[lam])
        edfs.append(res["fit"]["edf"])
    assert edfs[0] > edfs[1] > edfs[2], edfs
    # and the span must be large: a zeroed penalty cannot produce one
    assert edfs[0] - edfs[2] > 5.0, edfs
    assert 2.0 < edfs[1] < total


# --- fix 2: assert the robust pass CHANGES THE ANSWER, not just that a
# diagnostic was recorded. A mutation setting the Huber weights to all-ones
# SURVIVED the first draft, because that draft checked
# `huber_frac_downweighted`, which is computed from the residuals BEFORE the
# weights are applied -- so it is unaffected by disabling them.
def test_robust_pass_changes_the_fit_when_an_outlier_is_present():
    feat, con, pool = _fixture(seed=5)
    y = _truth(feat, con)
    y[0, 0] = y.max() * 500.0
    blocks, _ = _blocks(feat, con, pool)
    plain = fit_smooth_mu(y, blocks, robust=False)
    rob = fit_smooth_mu(y, blocks, robust=True)
    # SUBSTANTIVE property, not a coefficient nudge: the robust fit must
    # REFUSE TO CHASE the outlier, so its fitted mu at that cell is far below
    # the plain fit's. A `> 1e-6` threshold on beta was too loose -- the
    # robust stage's extra IRLS iterations alone clear it, so a mutation
    # disabling the weights survived. Measured effect is ~5 orders of
    # magnitude at the outlier and a median |log ratio| of ~1.4 elsewhere.
    assert rob["mu_hat"][0, 0] < plain["mu_hat"][0, 0] / 10.0, (
        rob["mu_hat"][0, 0], plain["mu_hat"][0, 0])
    lr = np.abs(np.log(rob["mu_hat"] / plain["mu_hat"]))
    assert float(np.median(lr)) > 0.1, float(np.median(lr))
    assert rob["fit"]["huber_frac_downweighted"] > 0.0
    assert rob["fit"]["stage"] == "robust"


def test_robust_pass_is_a_near_noop_without_outliers():
    """The converse, so the robust stage is not simply always perturbing the
    fit: on clean data the reweighting should barely move it."""
    feat, con, pool = _fixture(seed=12)
    y = _truth(feat, con)
    blocks, _ = _blocks(feat, con, pool)
    m0 = fit_smooth_mu(y, blocks, robust=False)["mu_hat"]
    m1 = fit_smooth_mu(y, blocks, robust=True)["mu_hat"]
    assert float(np.median(np.abs(np.log(m1 / m0)))) < 0.1


# ---------------------------------------------------------------------------
# The covariate-subset path, which the confound check depends on
# ---------------------------------------------------------------------------

def test_concept_side_only_fit_runs_and_is_constant_across_features():
    """Dropping the feature-side covariates must leave `mu_hat` varying only
    along the concept axis -- otherwise the 'concept-side only' arm of the
    confound check is not what it claims to be."""
    feat, con, pool = _fixture(seed=6)
    y = _truth(feat, con)
    blocks, knots = _blocks(feat, con, pool, tuple(CONCEPT_SIDE))
    res = fit_smooth_mu(y, blocks, robust=False)
    beta = np.asarray(res["fit"]["beta"])
    mu = predict_on_axis(beta, feat, con, knots, CONCEPT_SIDE, n_concepts=NC)
    # SHAPE IS ASSERTED IN FULL. A first draft checked only `mu.shape[0] == NC`
    # and passed on a (NC, 1) result, because the missing feature axis
    # broadcasts correctly in arithmetic and only shows up when something
    # reduces over it. It then crashed in the confound job on
    # `median(mu, axis=0)`. Assert both axes.
    assert mu.shape == (NC, NF), mu.shape
    per_concept_spread = (mu.max(axis=1) - mu.min(axis=1)).max()
    assert per_concept_spread == pytest.approx(0.0, abs=1e-12)
    assert np.median(mu, axis=0).shape == (NF,)


def test_predict_on_axis_matches_the_fitted_grid():
    feat, con, pool = _fixture(seed=7)
    y = _truth(feat, con)
    blocks, knots = _blocks(feat, con, pool)
    res = fit_smooth_mu(y, blocks, robust=False)
    beta = np.asarray(res["fit"]["beta"])
    mu_full = predict_on_axis(beta, feat, con, knots, COVARIATES,
                             n_concepts=NC)
    assert mu_full.shape == (NC, NF)
    assert np.allclose(mu_full, res["mu_hat"], rtol=1e-8)


# ---------------------------------------------------------------------------
# Adequacy -- the verdicts must be derived and must be able to fail
# ---------------------------------------------------------------------------

def _adq(y, mu, feat, con, pool, nf=NF):
    fb = np.zeros(nf, dtype=int)
    fb[nf // 2:] = 1
    return adequacy(y, mu, feat, con, pool, fb, 2)


def test_absorption_verdict_can_say_MISS():
    """A `mu_hat` that is constant cannot absorb anything, so the direction
    verdict must report MISS. A check that can only pass is not a check."""
    feat, con, pool = _fixture(seed=8, nf=400)
    y = _truth(feat, con)
    rep = _adq(y, np.full_like(y, y.mean()), feat, con, pool, nf=400)
    a = rep["absorption"]
    assert a["S2_absorbs_in_the_right_direction"] == "MISS"
    assert a["ratio_after_over_before"] == pytest.approx(1.0, rel=1e-6)


def test_absorption_verdict_says_HIT_when_the_trend_absorbs():
    feat, con, pool = _fixture(seed=9, nf=400)
    y = _truth(feat, con)
    rep = _adq(y, y.copy(), feat, con, pool, nf=400)   # perfect fit
    assert rep["absorption"]["S2_absorbs_in_the_right_direction"] == "HIT"


def test_adequacy_reports_a_residual_decile_per_covariate():
    feat, con, pool = _fixture(seed=10, nf=400)
    y = _truth(feat, con)
    rep = _adq(y, y.copy(), feat, con, pool, nf=400)
    for nm in COVARIATES:
        assert nm in rep["residual_vs_covariate"]
        assert len(rep["residual_vs_covariate"][nm]
                   ["decile_mean_pearson_residual"]) == 10


def test_adequacy_records_the_gotcha_21_scope_note():
    """The absorption number is measured on the MEAN, while R7's 33x-225x is
    on the MAX. The artifact must carry that distinction, because conflating
    the two is exactly the error gotcha 21 names."""
    feat, con, pool = _fixture(seed=11, nf=400)
    y = _truth(feat, con)
    rep = _adq(y, y.copy(), feat, con, pool, nf=400)
    note = rep["absorption"]["gotcha_21_note"]
    assert "maximum over concepts" in note and "different reduction" in note
    assert "MEAN" in note
