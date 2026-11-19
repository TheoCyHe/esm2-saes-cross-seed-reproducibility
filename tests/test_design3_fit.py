"""THE DESIGN-3 ESTIMATOR MUST RECOVER A TRUTH IT WAS NEVER TOLD.

An IRLS routine produces plausible numbers from any input, and every defect
this project has shipped had that shape. So the central tests here simulate
`k ~ Binomial(39, p)` with `logit(p)` built from a CHOSEN `b1`, a CHOSEN
smooth function of `log10 phi` and a CHOSEN `gamma`, and require the fitter to
return them. This is the discipline `tests/test_rf_mixture.py` applies to the
mixture, transplanted to a new fitter.

EVERY TOLERANCE HERE IS PAIRED WITH A DISCRIMINATION TEST. A recovery test
that passes at tolerance 1.0 proves nothing if the estimator would return the
same answer for a different truth, so `test_recovery_discriminates_between_two_truths`
runs the identical pipeline at two `b1` values and requires the estimates to
separate by more than the tolerance. Gotcha 16's fixture lesson: a fixture
that cannot exhibit the disputed condition cannot test it.

MUTATIONS RUN, AND THE RESULTS ARE THE MEASURED ONES rather than the ones
this docstring first predicted (recorded 2026-07-31):

  * **spline block dropped from `build_design`** -> kills **9** tests,
    including both recovery tests, the smooth-shape test, both variance-share
    tests and the drop-one deviance test.
  * **`b1`'s column penalised as well as the spline** (`S[1,1] = 1e6`) ->
    kills **2**: `test_reml_selects_an_interior_lambda` and
    `test_standardisation_cannot_move_the_fit`. IT DOES **NOT** KILL THE
    RECOVERY TESTS, which is worth recording rather than tidying away: REML
    responds by driving `lambda` to the grid's lower edge, so `b1` is barely
    shrunk and recovery still passes. The mutation is caught -- but by the
    boundary check, not by the test one would have named. That is gotcha 13
    extension 3's shape (a test passing for the wrong reason), and it is why
    `lambda_at_grid_edge` is reported for every real fit rather than being a
    developer convenience.
  * **frequentist `inv(X'WX)` instead of the Bayesian covariance** -> kills
    **0**. The two agree closely here, so NO TEST IN THIS FILE COVERS THAT
    CHOICE and none claims to. It is recorded as a stated convention in
    `design3_fit`'s docstring instead.
  * **`Z = Q[:, 1:]` -> `Z = Q[:, :-1]`** (a constraint basis that does not
    annihilate the constraint) -> kills `test_centred_spline_sums_to_zero`.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from src.analysis.design3_fit import (
    LAMBDA_GRID,
    average_marginal_effect,
    binomial_deviance,
    binomial_loglik,
    build_design,
    centred_spline,
    endpoint_check,
    fit_betabinomial,
    fit_design3,
    meta_analyse,
    standardisation_invariance,
    variance_share,
)

N_TRIALS = 39


def _truth(n_feat=10240, b0=0.0, b1=0.6, gamma=0.15, seed=0,
           smooth_amp=1.2, rho_Mphi=0.0):
    """Simulate the design-3 data-generating process EXACTLY as specified.

    `rho_Mphi` correlates `log M` with `log10 phi`, which is gotcha 5's
    measured fact and the reason the spline is mandatory rather than a
    nicety -- a fitter that recovers `b1` only when they are orthogonal has
    not been tested on the case that matters.
    """
    rng = np.random.default_rng(seed)
    log_phi = rng.uniform(-5.0, -1.0, n_feat)
    z_phi = (log_phi - log_phi.mean()) / log_phi.std()
    logM = (rho_Mphi * z_phi
            + np.sqrt(max(1 - rho_Mphi ** 2, 0.0)) * rng.normal(size=n_feat))
    d_fs = rng.uniform(0.0, 6.0, n_feat)
    # A genuinely non-linear smooth, so a linear firing-rate term could not
    # absorb it and the spline has something to find.
    f = smooth_amp * np.sin(1.6 * (log_phi + 3.0))
    f = f - f.mean()
    eta = b0 + b1 * logM + f + gamma * (d_fs - d_fs.mean())
    y = rng.binomial(N_TRIALS, expit(eta))
    return {"y": y, "logM": logM, "log_phi": log_phi, "d_fs": d_fs,
            "eta": eta, "f": f, "b1": b1, "gamma": gamma}


def _fit(t):
    sd = t["logM"].std()
    z = (t["logM"] - t["logM"].mean()) / sd
    d = build_design(z, t["log_phi"], t["d_fs"])
    f = fit_design3(t["y"], N_TRIALS, d["X"], d["S"])
    return d, f, sd


# ---------------------------------------------------------------------------
# The basis
# ---------------------------------------------------------------------------

def test_centred_spline_sums_to_zero():
    x = np.linspace(-5, -1, 500)
    Bc, Sc, _, Z = centred_spline(x)
    assert Bc.shape == (500, 9)                 # k = 10 -> 9 after centring
    assert abs(Bc.sum(axis=0)).max() < 1e-8
    assert Sc.shape == (9, 9)
    assert np.allclose(Sc, Sc.T)
    assert np.linalg.eigvalsh(Sc).min() > -1e-10


def test_centred_spline_reuses_the_smooth_mu_basis():
    """Two bases for one covariate is how they drift (SPECIFICATION §1a)."""
    from src.analysis.smooth_mu import bspline_design
    x = np.linspace(-5, -1, 200)
    B, knots = bspline_design(x)
    Bc, _, knots2, Z = centred_spline(x)
    assert np.allclose(knots, knots2)
    assert np.allclose(Bc, B @ Z)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

def test_recovers_b1_at_a_known_truth():
    t = _truth(b1=0.6, seed=1)
    d, f, sd = _fit(t)
    b1_raw = f["beta"][d["idx_logM"]] / sd
    assert abs(b1_raw - 0.6) < 0.05, b1_raw


def test_recovers_b1_with_a_confounded_predictor():
    """gotcha 5: aligned features skew higher-firing, so `log M` and
    `log10 phi` are correlated in the real data. Recovery has to survive it."""
    t = _truth(b1=0.6, rho_Mphi=0.6, seed=2)
    d, f, sd = _fit(t)
    b1_raw = f["beta"][d["idx_logM"]] / sd
    assert abs(b1_raw - 0.6) < 0.06, b1_raw


def test_recovers_gamma():
    t = _truth(b1=0.4, gamma=0.25, seed=3)
    d, f, _ = _fit(t)
    g = f["beta"][d["names"].index("d_fs")]
    assert abs(g - 0.25) < 0.03, g


def test_recovers_a_null_b1_as_null():
    """The estimator must also return ZERO when the truth is zero -- an
    estimator that only recovers non-zero effects is a machine for finding
    them."""
    t = _truth(b1=0.0, seed=4)
    d, f, sd = _fit(t)
    b1_raw = f["beta"][d["idx_logM"]] / sd
    se = np.sqrt(f["cov"][d["idx_logM"], d["idx_logM"]]) / sd
    assert abs(b1_raw) < 4.0 * se, (b1_raw, se)
    assert abs(b1_raw) < 0.05


def test_recovery_discriminates_between_two_truths():
    """THE TOLERANCE IS ONLY MEANINGFUL IF THE ESTIMATOR SEPARATES THE CASES.
    Same pipeline, same seed, two `b1`; the estimates must differ by far more
    than the tolerance the recovery tests use."""
    a = _truth(b1=0.0, seed=5)
    b = _truth(b1=0.6, seed=5)
    da, fa, sa = _fit(a)
    db, fb, sb = _fit(b)
    ea = fa["beta"][da["idx_logM"]] / sa
    eb = fb["beta"][db["idx_logM"]] / sb
    assert eb - ea > 0.5, (ea, eb)


def test_recovers_the_smooth_shape():
    """The fitted spline must track the simulated smooth, not merely exist."""
    t = _truth(b1=0.3, smooth_amp=1.2, seed=6)
    d, f, _ = _fit(t)
    s_hat = d["X"][:, d["spline_slice"]] @ f["beta"][d["spline_slice"]]
    r = np.corrcoef(s_hat, t["f"])[0, 1]
    assert r > 0.97, r
    assert abs(s_hat.std() - t["f"].std()) < 0.25


def test_a_flat_truth_gives_a_nearly_flat_spline():
    """And the penalty must be able to shrink the smooth away, or `k = 10`
    would be a degree of freedom after all (AL8b)."""
    t = _truth(b1=0.3, smooth_amp=0.0, seed=7)
    d, f, _ = _fit(t)
    s_hat = d["X"][:, d["spline_slice"]] @ f["beta"][d["spline_slice"]]
    assert s_hat.std() < 0.05, s_hat.std()


# ---------------------------------------------------------------------------
# Machinery
# ---------------------------------------------------------------------------

def test_irls_converges_and_says_so():
    t = _truth(seed=8)
    _, f, _ = _fit(t)
    assert f["converged"] is True
    assert f["final_shift"] < 1e-10
    assert f["n_iter"] < 200


def test_reml_selects_an_interior_lambda():
    """A boundary selection means the GRID chose lambda, not REML."""
    t = _truth(seed=9)
    _, f, _ = _fit(t)
    assert f["lambda_at_grid_edge"] is False, f["lambda"]
    assert LAMBDA_GRID[0] < f["lambda"] < LAMBDA_GRID[-1]


def test_dropping_the_spline_gives_a_larger_deviance():
    t = _truth(b1=0.4, smooth_amp=1.2, seed=10)
    sd = t["logM"].std()
    z = (t["logM"] - t["logM"].mean()) / sd
    full = build_design(z, t["log_phi"], t["d_fs"])
    nos = build_design(z, t["log_phi"], t["d_fs"], include_spline=False)
    f_full = fit_design3(t["y"], N_TRIALS, full["X"], full["S"])
    f_nos = fit_design3(t["y"], N_TRIALS, nos["X"], nos["S"])
    assert nos["spline_slice"] is None
    assert f_nos["unpenalised"] is True
    assert f_nos["deviance"] > f_full["deviance"]


def test_dropping_logM_gives_a_larger_deviance():
    t = _truth(b1=0.6, seed=11)
    sd = t["logM"].std()
    z = (t["logM"] - t["logM"].mean()) / sd
    full = build_design(z, t["log_phi"], t["d_fs"])
    nom = build_design(z, t["log_phi"], t["d_fs"], include_M=False)
    f_full = fit_design3(t["y"], N_TRIALS, full["X"], full["S"])
    f_nom = fit_design3(t["y"], N_TRIALS, nom["X"], nom["S"])
    assert nom["idx_logM"] is None
    assert f_nom["deviance"] > f_full["deviance"]


def test_deviance_and_loglik_agree_on_their_difference():
    """`D_a - D_b == -2 (l_a - l_b)`: the saturated and combinatorial terms
    cancel. Gotcha 20's practice -- compute what the dependence implies and
    confirm the code produces it."""
    t = _truth(seed=12)
    sd = t["logM"].std()
    z = (t["logM"] - t["logM"].mean()) / sd
    a = build_design(z, t["log_phi"], t["d_fs"])
    b = build_design(z, t["log_phi"], t["d_fs"], include_M=False)
    fa = fit_design3(t["y"], N_TRIALS, a["X"], a["S"])
    fb = fit_design3(t["y"], N_TRIALS, b["X"], b["S"])
    assert abs((fb["deviance"] - fa["deviance"])
               + 2 * (fb["loglik"] - fa["loglik"])) < 1e-6


def test_binomial_deviance_is_zero_at_the_saturated_fit():
    y = np.array([0, 5, 20, 39])
    eta = np.log(np.clip(y / N_TRIALS, 1e-12, 1 - 1e-12)
                 / (1 - np.clip(y / N_TRIALS, 1e-12, 1 - 1e-12)))
    assert binomial_deviance(y, N_TRIALS, eta) < 1e-6


def test_binomial_loglik_matches_scipy():
    from scipy.stats import binom
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 50)
    y = rng.binomial(N_TRIALS, p)
    eta = np.log(p / (1 - p))
    assert abs(binomial_loglik(y, N_TRIALS, eta)
               - float(binom.logpmf(y, N_TRIALS, p).sum())) < 1e-8


# ---------------------------------------------------------------------------
# AL8a's assertion, as a measurement
# ---------------------------------------------------------------------------

def test_standardisation_cannot_move_the_fit():
    t = _truth(b1=0.6, seed=13)
    inv = standardisation_invariance(t["y"], N_TRIALS, t["logM"],
                                     t["log_phi"], t["d_fs"])
    assert inv["rel_residue_b1"] < 1e-6, inv
    assert inv["abs_residue_deviance"] < 1e-6, inv
    assert inv["abs_residue_wald"] < 1e-5, inv


def test_variance_share_is_a_ratio_of_two_logit_scale_sds():
    t = _truth(b1=0.6, smooth_amp=1.2, seed=14)
    d, f, _ = _fit(t)
    vs = variance_share(f, d)
    assert vs["sd_ratio_is_defined"] is True
    m = d["X"][:, d["idx_logM"]] * f["beta"][d["idx_logM"]]
    s = d["X"][:, d["spline_slice"]] @ f["beta"][d["spline_slice"]]
    assert abs(vs["ratio"] - m.std() / s.std()) < 1e-12


def test_variance_share_moves_the_expected_way_with_the_truth():
    """A ratio that cannot change is not a measurement."""
    ds, fs, _ = _fit(_truth(b1=0.1, smooth_amp=1.5, seed=15))
    dl, fl, _ = _fit(_truth(b1=1.2, smooth_amp=1.5, seed=15))
    assert variance_share(fl, dl)["ratio"] > variance_share(fs, ds)["ratio"]


# ---------------------------------------------------------------------------
# Beta-binomial dispersion arm and AK9a's endpoint check
# ---------------------------------------------------------------------------

def test_betabinomial_recovers_the_binomial_when_there_is_no_overdispersion():
    t = _truth(b1=0.5, n_feat=4000, seed=16)
    d, f, sd = _fit(t)
    bb = fit_betabinomial(t["y"], N_TRIALS, d["X"], f["lambda"] * d["S"],
                          f["beta"])
    assert bb["rho"] < 0.02, bb["rho"]
    assert abs(bb["beta"][d["idx_logM"]] - f["beta"][d["idx_logM"]]) < 0.05


def test_betabinomial_detects_planted_overdispersion():
    """The arm must be able to FIRE, or it is not a dispersion check."""
    rng = np.random.default_rng(17)
    n_feat = 4000
    log_phi = rng.uniform(-5, -1, n_feat)
    logM = rng.normal(size=n_feat)
    d_fs = rng.uniform(0, 6, n_feat)
    p_mean = expit(0.3 * logM)
    s = 4.0                                   # heavy overdispersion
    p = rng.beta(p_mean * s, (1 - p_mean) * s)
    y = rng.binomial(N_TRIALS, p)
    d = build_design((logM - logM.mean()) / logM.std(), log_phi, d_fs)
    f = fit_design3(y, N_TRIALS, d["X"], d["S"])
    bb = fit_betabinomial(y, N_TRIALS, d["X"], f["lambda"] * d["S"],
                          f["beta"])
    assert bb["rho"] > 0.1, bb["rho"]
    assert bb["loglik"] > f["loglik"]


def test_endpoint_check_direction_is_derived_not_asserted():
    """Both directions must be reachable, or the sentence is decoration."""
    t = _truth(b1=0.5, n_feat=3000, seed=18)
    _, f, _ = _fit(t)
    ok = endpoint_check(t["y"], N_TRIALS, f["eta"])
    assert ok["family"] == "binomial"
    # A deliberately over-dispersed observation vector against the same fit
    # must flip the direction.
    rng = np.random.default_rng(19)
    y_od = np.where(rng.random(t["y"].size) < 0.2,
                    rng.integers(0, 2, t["y"].size) * N_TRIALS, t["y"])
    od = endpoint_check(y_od, N_TRIALS, f["eta"])
    assert od["residual_P_k_eq_0"] > ok["residual_P_k_eq_0"]
    assert "TOO NARROW" in od["direction_of_misspecification"]


# ---------------------------------------------------------------------------
# Meta-analysis
# ---------------------------------------------------------------------------

def test_meta_analysis_recovers_a_common_effect():
    rng = np.random.default_rng(20)
    true_b = 0.4
    se = np.full(20, 0.05)
    est = true_b + rng.normal(scale=se)
    m = meta_analyse(est, se)
    assert abs(m["fixed_effect"] - true_b) < 0.03
    assert m["n_seeds_used"] == 20
    assert m["I2_percent"] < 40


def test_meta_analysis_reports_both_combinations_and_their_ratio():
    est = np.array([0.4] * 10 + [0.6] * 10)
    se = np.full(20, 0.05)
    m = meta_analyse(est, se)
    assert abs(m["fixed_effect"] - 0.5) < 1e-9
    assert abs(m["across_seed_mean"] - 0.5) < 1e-9
    # Real heterogeneity: the across-seed SE must EXCEED the fixed-effect one.
    assert m["across_seed_se_over_fixed_effect_se"] > 1.0
    # Q = 20 * (1/0.05^2) * 0.01 = 80 on df 19, so I2 = 61/80 = 76.25%.
    # The number is computed from the fixture rather than guessed at.
    assert abs(m["Q"] - 80.0) < 1e-9
    assert abs(m["I2_percent"] - 100 * 61 / 80) < 1e-9


def test_meta_analysis_reports_no_usable_seeds_rather_than_a_number():
    m = meta_analyse(np.array([np.nan] * 3), np.array([np.nan] * 3))
    assert m["status"] == "NO_USABLE_SEEDS"
    assert m["n_seeds_used"] == 0


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf])
def test_meta_analysis_drops_a_degenerate_se_and_counts_it(bad):
    est = np.array([0.4, 0.5, 0.6])
    se = np.array([0.05, bad, 0.05])
    m = meta_analyse(est, se)
    assert m["n_seeds_used"] == 2
    assert m["n_seeds_dropped"] == 1


# ---------------------------------------------------------------------------
# AMENDMENT AS -- the outcome-scale marginal effect (§3) and the `d_fs` share
# put on the estimand's own instrument (§4).
# ---------------------------------------------------------------------------

def test_ame_is_the_average_of_the_derivative_not_the_derivative_at_the_mean():
    """The brief forbids `b1 * p(1-p)` at the mean, which is the
    back-of-envelope version. On a fixture with spread-out `p` the two differ
    materially, which is the only way this test has content."""
    from scipy.special import expit
    t = _truth(b1=0.6, seed=30)
    d, f, _ = _fit(t)
    i = d["idx_logM"]
    ame = average_marginal_effect(
        f, d, float(np.sqrt(f["cov"][i, i])), n_trials=N_TRIALS)
    p = expit(np.clip(f["eta"], -30, 30))
    want = float(np.mean(f["beta"][i] * p * (1 - p)))
    assert abs(ame["ame"] - want) < 1e-12
    # and the fixture must make the two constructions differ, or the
    # distinction this test exists for is untestable (gotcha 16)
    assert abs(ame["ame"] - ame["naive_at_mean_p_NOT_USED"]) > 0.05 * abs(want)
    assert abs(ame["ame_percentage_points"] - 100 * want) < 1e-10


def test_ame_carries_its_honesty_clause_and_says_it_is_not_outcome_blind():
    """AS §3 requires the clause to live in the object, not only in the
    amendment -- a caveat that travels separately does not travel."""
    t = _truth(seed=31)
    d, f, _ = _fit(t)
    ame = average_marginal_effect(f, d, 0.01)
    assert ame["outcome_blind"] is False
    assert "never" in ame["honesty_note"]
    assert "first-order" in ame["se_note"]


def test_ame_is_absent_rather_than_zero_when_the_term_is_absent():
    t = _truth(seed=32)
    sd = t["logM"].std()
    z = (t["logM"] - t["logM"].mean()) / sd
    d = build_design(z, t["log_phi"], t["d_fs"], include_M=False)
    f = fit_design3(t["y"], N_TRIALS, d["X"], d["S"])
    got = average_marginal_effect(f, d, None)
    assert got["status"].startswith("ABSENT")


def test_ame_sign_follows_the_coefficient():
    for b1, want in ((0.6, 1), (-0.6, -1)):
        t = _truth(b1=b1, seed=33)
        d, f, _ = _fit(t)
        ame = average_marginal_effect(f, d, None)
        assert np.sign(ame["ame"]) == want, (b1, ame["ame"])


def test_variance_share_puts_d_fs_on_the_same_instrument_as_logM():
    """AS §4: `gamma / b1` is a unit artifact because the two live on
    different scales. The linear-predictor SD ratio removes the units, and
    the identity between the three returned ratios is checked here."""
    t = _truth(b1=0.6, gamma=0.25, seed=34)
    d, f, _ = _fit(t)
    vs = variance_share(f, d)
    m = d["X"][:, d["idx_logM"]] * f["beta"][d["idx_logM"]]
    s = d["X"][:, d["spline_slice"]] @ f["beta"][d["spline_slice"]]
    j = d["names"].index("d_fs")
    dd = d["X"][:, j] * f["beta"][j]
    assert abs(vs["sd_gamma_d_fs"] - dd.std()) < 1e-12
    assert abs(vs["ratio"] - m.std() / s.std()) < 1e-12
    assert abs(vs["ratio_d_fs_over_spline"] - dd.std() / s.std()) < 1e-12
    assert abs(vs["ratio_d_fs_over_logM"] - dd.std() / m.std()) < 1e-12
    # AL8a's primary ratio must be UNCHANGED in meaning by the extension
    assert abs(vs["ratio"] - m.std() / s.std()) < 1e-12


def test_the_d_fs_share_is_scale_free_where_the_raw_coefficient_is_not():
    """THE POINT OF §4, as a measurement. Rescaling `d_fs` by a constant
    changes `gamma` by exactly that factor and leaves the SD ratio invariant.
    A comparison that moves when you change a unit is not a comparison."""
    t = _truth(b1=0.6, gamma=0.25, seed=35)
    sd = t["logM"].std()
    z = (t["logM"] - t["logM"].mean()) / sd
    d1 = build_design(z, t["log_phi"], t["d_fs"])
    d2 = build_design(z, t["log_phi"], 7.0 * t["d_fs"])
    f1 = fit_design3(t["y"], N_TRIALS, d1["X"], d1["S"])
    f2 = fit_design3(t["y"], N_TRIALS, d2["X"], d2["S"])
    g1 = f1["beta"][d1["names"].index("d_fs")]
    g2 = f2["beta"][d2["names"].index("d_fs")]
    assert abs(g1 / g2 - 7.0) < 0.05                       # gamma moves 7x
    v1, v2 = variance_share(f1, d1), variance_share(f2, d2)
    assert abs(v1["ratio_d_fs_over_logM"]
               - v2["ratio_d_fs_over_logM"]) < 1e-6        # the ratio does not
