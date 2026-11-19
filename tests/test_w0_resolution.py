"""Tests for the `w0` resolution test (PREREGISTRATION amendment AO, §1-3).

MUTATION-TESTED. The mutations actually run, and each is recorded beside the
test it must kill -- gotcha 13 extension 3: verifying a test CAN fail is weaker
than verifying it fails FOR THE INTENDED REASON.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from scipy.stats import binom

from src.analysis import w0_resolution as W
from src.analysis.rf_mixture import P_GRID, fit_mixture, profile_w0


# ---------------------------------------------------------------------------
# The optional `w0_grid` must not perturb the default path. The baselined
# mixture artifacts are canary targets, so an optional argument that quietly
# changed the default would break bit-identity SILENTLY.
# MUTATION: make `w0_grid=None` fall back to anything but linspace(0,1,n_points)
# and this dies.
# ---------------------------------------------------------------------------
def _toy_counts(n_trials: int = 7, n_obs: int = 4000) -> np.ndarray:
    rng = np.random.default_rng(11)
    p = np.concatenate([np.zeros(300), rng.uniform(0.05, 0.95, 700)])
    k = rng.binomial(n_trials, p)
    return np.bincount(k, minlength=n_trials + 1)


def test_profile_w0_default_path_is_unchanged_by_the_new_argument():
    counts = _toy_counts()
    grid = np.linspace(0.0, 1.0, 9)
    a = profile_w0(counts, 7, n_points=9, max_iter=400)
    b = profile_w0(counts, 7, w0_grid=grid, max_iter=400)
    assert a["w0_grid"] == b["w0_grid"]
    assert a["profile_loglik"] == b["profile_loglik"]
    assert a["w0_interval"] == b["w0_interval"]


def test_profile_w0_honours_a_non_uniform_grid():
    counts = _toy_counts()
    grid = np.array([0.0, 0.001, 0.01, 0.1, 0.5, 1.0])
    res = profile_w0(counts, 7, w0_grid=grid, max_iter=400)
    assert res["w0_grid"] == grid.tolist()
    assert len(res["profile_loglik"]) == grid.size


# ---------------------------------------------------------------------------
# 1a. The value AT EXACTLY ZERO, or an explicit refusal. Reporting the nearest
# grid point as "the loglik at zero" is gotcha 15's absent-reads-as-agreement.
# MUTATION: replace the `g[0] != 0.0` guard with a pass and this dies.
# ---------------------------------------------------------------------------
def test_curve_near_zero_refuses_a_grid_that_omits_exact_zero():
    prof = {"w0_grid": [0.01, 0.02, 0.03], "profile_loglik": [-3.0, -1.0, -2.0],
            "drop_units": 2.0, "w0_interval": [0.01, 0.03]}
    with pytest.raises(SystemExit, match="does not start at exactly 0"):
        W.curve_near_zero(prof)


def test_curve_near_zero_reports_the_drop_at_zero_and_derives_the_ratio():
    prof = {"w0_grid": [0.0, 0.025, 0.05], "drop_units": 2.0,
            "profile_loglik": [-110.0, -101.0, -100.0], "w0_interval": [0.025, 0.05]}
    c = W.curve_near_zero(prof)
    assert c["loglik_at_exactly_zero"] == -110.0
    assert c["drop_at_exactly_zero"] == pytest.approx(10.0)
    assert c["drop_at_zero_in_units_of_the_cut"] == pytest.approx(5.0)
    assert c["deviance_1df_at_zero"] == pytest.approx(20.0)
    assert c["w0_at_max"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# AO2's committed convention. THE FIXTURE MUST CARRY AN ATOM EXACTLY AT THE
# OBSERVED VALUE, because that is the only configuration in which the two
# conventions differ -- a fixture without one cannot exercise the distinction
# (gotcha 16's fixture lesson).
# MUTATION: swap `<` for `<=` in `percentile_committed_strictly_below` and this
# dies on the first assertion.
# ---------------------------------------------------------------------------
def test_committed_convention_is_strictly_below_not_at_or_below():
    null = np.array([0.0] * 50 + [0.025] * 40 + [0.05] * 10)
    s = W.score_percentile(null, 0.025)
    assert s["atom_at_observed_frac"] == pytest.approx(0.40)
    assert s["percentile_committed_strictly_below"] == pytest.approx(50.0)
    assert s["percentile_rejected_convention_at_or_below"] == pytest.approx(90.0)
    assert s["exceedance_p_value"] == pytest.approx(0.50)
    # the two conventions must be DIFFERENT here, or the test is not testing
    # what it claims to
    assert (s["percentile_committed_strictly_below"]
            != s["percentile_rejected_convention_at_or_below"])


def test_conventions_straddle_flag_fires_when_the_atom_decides_the_outcome():
    # strictly-below 90 (<= 95, RESTORED); at-or-below 100 (> 95, would RETRACT)
    null = np.array([0.0] * 90 + [0.025] * 10)
    s = W.score_percentile(null, 0.025)
    assert s["percentile_committed_strictly_below"] == pytest.approx(90.0)
    assert s["percentile_rejected_convention_at_or_below"] == pytest.approx(100.0)
    assert s["conventions_straddle_the_cut"] is True
    assert s["verdict"] == "AJ2_THIRD_ARGUMENT_RESTORED"


@pytest.mark.parametrize("null,expected", [
    # every replicate returns 0 -> 0.025 is above the whole null -> RETRACT
    (np.zeros(100), "AJ2_THIRD_ARGUMENT_RETRACTED"),
    # every replicate returns 0.05 -> 0.025 is below the whole null -> RESTORE
    (np.full(100, 0.05), "AJ2_THIRD_ARGUMENT_RESTORED"),
    # exactly at the cut: 95 is `<= 95`, so RESTORE. The boundary is part of the
    # pre-declared rule's domain and is pinned, not left to chance.
    (np.array([0.0] * 95 + [0.05] * 5), "AJ2_THIRD_ARGUMENT_RESTORED"),
    (np.array([0.0] * 96 + [0.05] * 4), "AJ2_THIRD_ARGUMENT_RETRACTED"),
])
def test_both_branches_of_the_pre_declared_rule_are_reachable(null, expected):
    assert W.score_percentile(null, 0.025)["verdict"] == expected


# ---------------------------------------------------------------------------
# The `w0 = 0` truth. If the truth retains endpoint mass the whole simulation is
# meaningless, so it is asserted on the way out rather than trusted.
# MUTATION: drop the renormalisation (`w /= rest`) and the sum assertion dies.
# ---------------------------------------------------------------------------
def test_null_truth_removes_the_endpoint_mass_exactly_and_renormalises():
    w = np.array([0.2, 0.3, 0.5])
    t = W.null_truth(w, np.array([0.0, 0.5, 1.0]))
    assert t["weights"][0] == 0.0
    assert t["weights"].sum() == pytest.approx(1.0)
    assert t["removed_w0"] == pytest.approx(0.2)
    assert t["weights"][1] == pytest.approx(0.3 / 0.8)


def test_null_truth_refuses_a_grid_without_exact_zero():
    with pytest.raises(SystemExit, match="not exactly 0"):
        W.null_truth(np.array([0.5, 0.5]), np.array([0.01, 1.0]))


def test_null_truth_refuses_when_all_mass_is_at_zero():
    with pytest.raises(SystemExit, match="no w0 = 0 truth exists"):
        W.null_truth(np.array([1.0, 0.0]), np.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
# The multinomial sampler is claimed EXACTLY equivalent to the two-stage draw.
# AN8: measure it, do not justify it. Two checks -- the pmf itself, and the
# empirical agreement of the two routes at large n.
# ---------------------------------------------------------------------------
def test_atom_pmf_matches_explicit_enumeration():
    p = np.array([0.0, 0.25, 1.0])
    w = np.array([0.5, 0.3, 0.2])
    n = 5
    got = W.atom_pmf(w, p, n)
    want = np.array([sum(w[j] * binom.pmf(k, n, p[j]) for j in range(3))
                     for k in range(n + 1)])
    assert got == pytest.approx(want / want.sum())


def test_multinomial_and_two_stage_samplers_agree():
    p = P_GRID
    rng = np.random.default_rng(5)
    w = rng.dirichlet(np.ones(p.size))
    w[0] = 0.0
    w /= w.sum()
    n_trials, n_obs = 9, 200_000
    pk = W.atom_pmf(w, p, n_trials)
    a = W.draw_counts(pk, n_obs, 0, 0) / n_obs
    b = W.draw_counts_two_stage(w, p, n_trials, n_obs, 0, 0) / n_obs
    # both are draws of the SAME multinomial, so they agree to sampling error
    se = np.sqrt(pk * (1 - pk) / n_obs)
    assert np.all(np.abs(a - b) < 6 * np.sqrt(2) * se + 1e-12)
    assert np.all(np.abs(a - pk) < 6 * se + 1e-12)


def test_draw_counts_is_a_function_of_replicate_index_alone():
    pk = W.atom_pmf(np.array([0.0, 1.0]), np.array([0.0, 0.4]), 6)
    assert np.array_equal(W.draw_counts(pk, 500, 7, 3),
                          W.draw_counts(pk, 500, 7, 3))
    assert not np.array_equal(W.draw_counts(pk, 500, 7, 3),
                              W.draw_counts(pk, 500, 7, 4))
    assert not np.array_equal(W.draw_counts(pk, 500, 7, 3),
                              W.draw_counts(pk, 500, 8, 3))


# ---------------------------------------------------------------------------
# A parallel simulation whose answer depends on --workers is not reproducible.
# MUTATION: seed from a shared generator instead of (base_seed, r) and this dies.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(sys.platform.startswith("win"),
                    reason="fork start method is POSIX-only; the production run "
                           "is on Linux and the serial path is tested above")
def test_simulation_is_identical_at_one_worker_and_three():
    p = np.array([0.0, 0.2, 0.5, 0.9, 1.0])
    w = np.array([0.0, 0.3, 0.3, 0.3, 0.1])
    pk = W.atom_pmf(w, p, 5)
    kw = dict(n_trials=5, n_obs=3000, n_rep=6, base_seed=3, n_points=5,
              max_iter=200)
    one = W.simulate_null(pk, p, workers=1, **kw)
    three = W.simulate_null(pk, p, workers=3, **kw)
    assert [r["r"] for r in one] == [r["r"] for r in three]
    for a, b in zip(one, three):
        assert a == b


def test_simulate_null_serial_returns_the_expected_fields():
    p = np.array([0.0, 0.3, 0.7, 1.0])
    pk = W.atom_pmf(np.array([0.0, 0.5, 0.4, 0.1]), p, 4)
    reps = W.simulate_null(pk, p, n_trials=4, n_obs=1500, n_rep=3, base_seed=1,
                           n_points=5, max_iter=200)
    assert len(reps) == 3
    for r in reps:
        assert set(r) >= {"r", "point_w0", "profiled_lower", "profiled_upper",
                          "atom_frac_k0", "fit_converged"}
        assert 0.0 <= r["profiled_lower"] <= r["profiled_upper"] <= 1.0


# ---------------------------------------------------------------------------
# The fine grid and the interval reader.
# ---------------------------------------------------------------------------
def test_fine_grid_is_denser_than_the_production_grid_where_it_matters():
    g = W.fine_w0_grid()
    assert g[0] == 0.0
    assert np.isclose(g[1] - g[0], 0.001)
    near = g[g <= 0.15]
    assert np.max(np.diff(near)) <= 0.001 + 1e-12
    assert np.max(np.diff(near)) < W.PRODUCTION_PROFILE_SPACING
    assert np.isclose(g[-1], 1.0)
    assert np.all(np.diff(g) > 0)


def test_local_spacing_reports_the_resolution_at_a_bound():
    g = np.array([0.0, 0.001, 0.002, 0.5, 1.0])
    assert W.local_spacing(g, 0.001) == pytest.approx(0.001)
    assert W.local_spacing(g, 0.5) == pytest.approx(0.498)


def test_interval_from_profile_flags_exclusion_of_zero_both_ways():
    g = np.array([0.0, 0.01, 0.02, 0.03])
    inc = W.interval_from_profile(g, np.array([-1.5, -0.5, 0.0, -0.5]), drop=2.0)
    assert inc["w0_interval"] == [0.0, 0.03]
    assert inc["excludes_zero"] is False
    exc = W.interval_from_profile(g, np.array([-9.0, -0.5, 0.0, -0.5]), drop=2.0)
    assert exc["w0_interval"] == [0.01, 0.03]
    assert exc["excludes_zero"] is True
    assert exc["spacing_at_lower_bound"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# THE ENTRY POINT. A unit suite that never executes main() cannot catch an
# entry-point bug (gotcha 13 extension 4, corollary). This also exercises the
# two refusals main() owns: overwriting a canary target, and rescoring the
# pre-declared rule against a lower bound it was not written about.
# ---------------------------------------------------------------------------
def _fake_mixture(tmp_path, n_trials=7, lower=0.025, n_points=41, name="mix.json"):
    counts = _toy_counts(n_trials=n_trials, n_obs=4000)
    fit = fit_mixture(counts, n_trials, max_iter=300)
    grid = np.linspace(0.0, 1.0, n_points)
    doc = {
        "identity": {"n_trials": n_trials},
        "by_theta": {"0.70": {
            "fit": fit,
            "w0_profile": {
                "w0_grid": grid.tolist(),
                "profile_loglik": (-100.0 - 40.0 * (grid - 0.05) ** 2
                                   - 20.0 * (grid == 0.0)).tolist(),
                "w0_at_max": 0.05, "loglik_at_max": -100.0, "drop_units": 2.0,
                "w0_interval": [lower, 0.05], "w0_interval_width": 0.05 - lower,
                "n_points": n_points},
        }},
    }
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return p


def test_main_refuses_to_overwrite_its_input(tmp_path, monkeypatch):
    src = _fake_mixture(tmp_path)
    monkeypatch.setattr(sys, "argv",
                        ["w0", "--mixture", str(src), "--out", str(src),
                         "--skip-null"])
    with pytest.raises(SystemExit, match="canary target"):
        W.main()


def test_main_refuses_a_lower_bound_the_rule_was_not_written_about(tmp_path,
                                                                   monkeypatch):
    src = _fake_mixture(tmp_path, lower=0.05)
    monkeypatch.setattr(sys, "argv",
                        ["w0", "--mixture", str(src),
                         "--out", str(tmp_path / "o.json"), "--skip-null"])
    with pytest.raises(SystemExit, match="refusing to rescore"):
        W.main()


def test_main_refuses_a_non_production_profile_grid(tmp_path, monkeypatch):
    src = _fake_mixture(tmp_path, n_points=21)
    monkeypatch.setattr(sys, "argv",
                        ["w0", "--mixture", str(src),
                         "--out", str(tmp_path / "o.json"), "--skip-null"])
    with pytest.raises(SystemExit, match="SAME grid"):
        W.main()


def test_main_skip_null_reports_NOT_RUN_rather_than_a_branch(tmp_path,
                                                             monkeypatch):
    src = _fake_mixture(tmp_path)
    out = tmp_path / "o.json"
    monkeypatch.setattr(sys, "argv",
                        ["w0", "--mixture", str(src), "--out", str(out),
                         "--skip-null", "--profile-max-iter", "200"])
    W.main()
    doc = json.loads(out.read_text())
    assert doc["AO1c_null_from_w0_zero"]["verdict"].startswith("NOT_RUN")
    # "not run" must never read as a branch of the rule
    assert "RESTORED" not in doc["AO1c_null_from_w0_zero"]["verdict"]
    assert "RETRACTED" not in doc["AO1c_null_from_w0_zero"]["verdict"]
    assert doc["prediction_scorecard"]["P3"]["outcome"].startswith("NOT_SCORED")
    assert doc["AO1_curve_near_zero"]["drop_at_exactly_zero"] > 0


def test_main_end_to_end_with_a_small_null(tmp_path, monkeypatch):
    src = _fake_mixture(tmp_path)
    out = tmp_path / "o.json"
    monkeypatch.setattr(sys, "argv",
                        ["w0", "--mixture", str(src), "--out", str(out),
                         "--n-rep", "3", "--profile-max-iter", "150"])
    W.main()
    doc = json.loads(out.read_text())
    nb = doc["AO1c_null_from_w0_zero"]
    assert nb["truth"]["w0_of_truth"] == 0.0
    assert nb["PRIMARY_profiled_lower_bound"]["n_replicates"] == 3
    assert nb["PRIMARY_profiled_lower_bound"]["verdict"].startswith("AJ2_")
    assert nb["protocol"]["n_trials"] == 7
    for key in ("P1", "P2", "P3", "P4", "P5"):
        assert doc["prediction_scorecard"][key]["outcome"] in {"HIT", "MISS"}


# ---------------------------------------------------------------------------
# em_at_fixed_w0 is a COPY of profile_w0's inner loop, kept as a copy so the
# baselined mixture numerics are untouched. The copy is only defensible if the
# agreement is MEASURED, so it is -- at every grid point, exactly.
# MUTATION: drop the renormalisation of w_c[1:] and this dies.
# ---------------------------------------------------------------------------
def test_em_at_fixed_w0_reproduces_profile_w0_loglik_at_every_grid_point():
    counts = _toy_counts(n_trials=7, n_obs=4000)
    grid = np.linspace(0.0, 0.3, 7)
    prof = profile_w0(counts, 7, w0_grid=grid, max_iter=3000)
    for w0, want in zip(grid, prof["profile_loglik"]):
        got = W.em_at_fixed_w0(counts, 7, P_GRID, float(w0), max_iter=3000)
        assert got["loglik"] == pytest.approx(want, rel=0, abs=1e-9)
        assert got["weights"][0] == pytest.approx(float(w0))
        assert got["weights"].sum() == pytest.approx(1.0)


def test_purity_at_w0_reproduces_the_linear_identity_for_the_strict_endpoint():
    """AJ1: strict purity is exactly w0 / P(k in group), because
    P(k in group | p = 0) = 1 for any group containing k = 0. Re-derived and
    compared rather than quoted (gotcha 20's arithmetic practice)."""
    counts = _toy_counts(n_trials=7, n_obs=4000)
    got = W.purity_at_w0(counts, 7, P_GRID, 0.05, [0, 1], max_iter=3000)
    assert got["identity_agrees"] is True
    assert got["strict_p_eq_0"] == pytest.approx(0.05 / got["P_k_in_group"])
    # the loose endpoint carries independent information: strictly larger
    assert got["loose_p_le_0.10"] > got["strict_p_eq_0"]
    assert got["loose_minus_strict"] > 0


def test_purity_at_w0_refuses_a_group_without_k_equals_zero():
    """The strict-endpoint identity assumes P(k in group | p = 0) = 1, which
    fails the moment k = 0 is outside the group. Refuse rather than report a
    band whose construction silently does not hold."""
    counts = _toy_counts(n_trials=7, n_obs=4000)
    with pytest.raises(SystemExit, match="does not contain k = 0"):
        W.purity_at_w0(counts, 7, P_GRID, 0.05, [1, 2], max_iter=200)


def test_strict_purity_is_linear_in_w0_and_loose_is_not():
    """The reason the band is refitted at each w0 rather than rescaled."""
    counts = _toy_counts(n_trials=7, n_obs=4000)
    a = W.purity_at_w0(counts, 7, P_GRID, 0.02, [0, 1], max_iter=3000)
    b = W.purity_at_w0(counts, 7, P_GRID, 0.04, [0, 1], max_iter=3000)
    # strict scales with w0 up to the (nearly constant) group probability
    assert b["strict_p_eq_0"] / a["strict_p_eq_0"] == pytest.approx(
        (0.04 / b["P_k_in_group"]) / (0.02 / a["P_k_in_group"]))
    # loose does NOT double
    assert b["loose_p_le_0.10"] / a["loose_p_le_0.10"] < 1.5
