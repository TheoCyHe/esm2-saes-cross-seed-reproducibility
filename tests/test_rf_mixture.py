"""Tests for the `r_f` mixing-distribution fit and endpoint purity (amendment AB).

THE CENTRAL DISCIPLINE HERE IS RECOVERY OF A KNOWN TRUTH. An EM fit produces
plausible numbers from any input, so the only convincing test is: generate
from a mixing distribution we chose, and require the fit to return it.

Mutations run and killed (see the AB commit message):
  * `purity` returning `genuine / 1.0` instead of `genuine / total`
        -> kills test_purity_of_a_pure_group_is_one
  * dropping exact 0 and 1 from `P_GRID`
        -> kills test_recovers_a_planted_point_mass_at_zero
  * the shape verdict reverting to mode-counting
        -> kills test_shape_verdict_calls_a_broad_mixture_broad
  * `three_component_sketch` reporting flat-interior consistency as True
        -> kills test_sketch_reports_its_own_assumption_as_refuted
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

from src.analysis.rf_mixture import (
    P_GRID,
    analyse_theta,
    fit_mixture,
    purity,
    shape_diagnosis,
    three_component_sketch,
)

N = 19


def _atoms_from_mixture(components, n_trials=N):
    """components: list of (weight, p). Returns the exact atom fractions."""
    k = np.arange(n_trials + 1)
    out = np.zeros(n_trials + 1)
    for w, p in components:
        out += w * binom.pmf(k, n_trials, p)
    return out / out.sum()


# ---------------------------------------------------------------------------
# Recovery of a known truth
# ---------------------------------------------------------------------------

def test_recovers_a_planted_point_mass_at_zero():
    """A genuine atom at p=0 must come back as an atom at p=0, not smeared
    into the nearest interior knot. This is why P_GRID carries exact 0."""
    atoms = _atoms_from_mixture([(0.30, 0.0), (0.70, 0.5)])
    fit = fit_mixture(np.rint(atoms * 2_000_000), N)
    assert fit["mass_at_p_eq_0"] == pytest.approx(0.30, abs=0.02)


def test_recovers_a_planted_point_mass_at_one():
    atoms = _atoms_from_mixture([(0.25, 1.0), (0.75, 0.4)])
    fit = fit_mixture(np.rint(atoms * 2_000_000), N)
    assert fit["mass_at_p_eq_1"] == pytest.approx(0.25, abs=0.02)


def test_uniform_p_gives_a_uniform_marginal():
    """The identity the whole section rests on: p ~ U(0,1) => Binomial(N,p)
    is exactly uniform on {0..N}.

    Verified EXACTLY via the Beta integral rather than on a truncated grid.
    A first draft used `linspace(0.0005, 0.9995, 20000)` and failed at k=0
    and k=N by 4.3e-4 -- not because the identity is false but because a grid
    that excludes the endpoints under-weights exactly the two atoms the
    endpoints dominate. That is the whole subject of this module appearing
    inside its own test.
    """
    from scipy.special import comb, beta as beta_fn
    k = np.arange(N + 1)
    exact = np.array([comb(N, kk) * beta_fn(kk + 1, N - kk + 1) for kk in k])
    assert np.allclose(exact, 1.0 / (N + 1), atol=1e-12)


def test_fit_reproduces_the_observed_atoms():
    atoms = _atoms_from_mixture([(0.1, 0.0), (0.1, 1.0), (0.8, 0.3)])
    fit = fit_mixture(np.rint(atoms * 1_000_000), N)
    assert fit["max_abs_atom_residual"] < 5e-3


def test_fit_recovers_the_mean_p():
    atoms = _atoms_from_mixture([(0.5, 0.2), (0.5, 0.8)])
    fit = fit_mixture(np.rint(atoms * 1_000_000), N)
    assert fit["mean_p"] == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------

def test_purity_of_a_pure_group_is_one():
    """If ALL mass sits at p=0, every k<=1 anchor is genuine.

    TOLERANCE IS 1e-4, NOT 1e-6, AND THE REASON IS THE STOPPING RULE.
    Convergence is now on the WEIGHT VECTOR (`max|dw| < 1e-6`), which bounds the
    STEP, not the distance to the optimum. EM approaches a boundary optimum like
    `w0 = 1` asymptotically, so the step falls below tolerance while the estimate
    is still ~1e-5 short: measured 0.9999901.

    This assertion previously passed at 1e-6 only because the old log-likelihood
    rule never triggered and the loop ran to its 20,000-iteration cap -- i.e. the
    tighter tolerance was a by-product of NON-convergence, not evidence of
    accuracy. Loosening it here records a real property of a step-size criterion
    rather than papering over a regression; the fit reports `converged: True` for
    this case, and 1e-4 is far tighter than any purity claim depends on.
    """
    atoms = _atoms_from_mixture([(1.0, 0.0)])
    fit = fit_mixture(np.rint(atoms * 100_000), N)
    p = np.asarray(fit["p_grid"])
    pu = purity(fit, N, [0, 1], p <= 0.0)
    assert fit["converged"] is True
    assert pu["purity"] == pytest.approx(1.0, abs=1e-4)


def test_purity_falls_when_a_broad_component_leaks_in():
    """A BROAD component must contaminate the low bin, lowering purity.

    NOTE the component has to be broad, not merely mid-range. A first draft
    used a point mass at p=0.5 and expected purity < 0.95; it came out
    0.9996, because P(k<=1 | p=0.5) = 20/2^19 = 3.8e-05 at N=19 -- a
    concentrated mid-range component essentially never draws an extreme
    count. The leak in the real data comes from mass spread over SMALL p and
    from the uniform component, which leaks the closed-form 2/(N+1).
    """
    unif = [(0.9 / 199, pp) for pp in np.linspace(0.005, 0.995, 199)]
    atoms = _atoms_from_mixture([(0.1, 0.0)] + unif)
    fit = fit_mixture(np.rint(atoms * 500_000), N)
    p = np.asarray(fit["p_grid"])
    pu = purity(fit, N, [0, 1], p <= 0.0)["purity"]
    assert 0.4 < pu < 0.7, pu          # ~0.1 / (0.1 + 0.9*0.10)


def test_uniform_component_leaks_the_closed_form_amount():
    """Section 4's closed form: a Uniform(0,1) component puts exactly
    2/(N+1) of its mass into the k<=1 bin. Computed, not quoted."""
    p = np.linspace(0.0005, 0.9995, 20000)
    leak = binom.pmf(np.array([0, 1])[None, :], N, p[:, None]).sum(axis=1).mean()
    assert leak == pytest.approx(2.0 / (N + 1), abs=2e-3)


def test_purity_is_definition_dependent_and_ordered():
    """A looser 'genuine' threshold can only raise purity. If this ever
    inverts, the thresholds are being applied to the wrong tail."""
    atoms = _atoms_from_mixture([(0.1, 0.0), (0.2, 0.03), (0.7, 0.5)])
    fit = fit_mixture(np.rint(atoms * 500_000), N)
    p = np.asarray(fit["p_grid"])
    strict = purity(fit, N, [0, 1], p <= 0.0)["purity"]
    loose = purity(fit, N, [0, 1], p <= 0.10)["purity"]
    assert strict < loose <= 1.0


# ---------------------------------------------------------------------------
# The shape verdict -- the part that was wrong twice
# ---------------------------------------------------------------------------

def test_shape_verdict_calls_a_broad_mixture_broad():
    """A genuinely broad p must NOT be labelled bimodal. This is the test the
    mode-counting version of the verdict failed."""
    atoms = _atoms_from_mixture([(1.0 / 3, 0.15), (1.0 / 3, 0.5), (1.0 / 3, 0.85)])
    fit = fit_mixture(np.rint(atoms * 200_000), N)
    sd = shape_diagnosis(fit, N, 200_000, n_sim=6, max_iter=300)
    assert sd["shape"] == "BROAD", sd["mass_in_thirds"]


def test_shape_verdict_calls_a_genuinely_bimodal_mixture_bimodal():
    """The converse: the verdict must be ABLE to say BIMODAL, or it is not a
    detector. Gerasimov's shape -- mass at both ends, empty middle."""
    atoms = _atoms_from_mixture([(0.5, 0.03), (0.5, 0.97)])
    fit = fit_mixture(np.rint(atoms * 200_000), N)
    sd = shape_diagnosis(fit, N, 200_000, n_sim=6, max_iter=300)
    assert sd["shape"] == "BIMODAL", sd["mass_in_thirds"]


def test_mode_count_claim_is_labelled_as_departure_not_bimodality():
    """The calibrated mode count tests departure from a uniform interior. The
    output must say so, because quoting it as bimodality is the defect this
    function shipped twice."""
    atoms = _atoms_from_mixture([(1.0, 0.5)])
    fit = fit_mixture(np.rint(atoms * 100_000), N)
    sd = shape_diagnosis(fit, N, 100_000, n_sim=4, max_iter=300)
    assert "departs_from_uniform_interior" in sd
    assert "does NOT test bimodality" in sd["departure_note"]
    assert sd["shape_basis"].startswith("mass in coarse thirds")


# ---------------------------------------------------------------------------
# The three-component sketch and its own refutation
# ---------------------------------------------------------------------------

def test_sketch_closes_to_one():
    atoms = _atoms_from_mixture([(0.1, 0.0), (0.05, 1.0), (0.85, 0.5)])
    sk = three_component_sketch(atoms, N)
    assert sk["closure_should_be_1"] == pytest.approx(1.0, abs=1e-9)


def test_sketch_reports_its_own_assumption_as_refuted():
    """On a mixture whose interior is NOT flat, the sketch must flag that its
    flat-interior assumption fails -- emitted alongside its own numbers so a
    reader cannot take them at face value."""
    atoms = _atoms_from_mixture([(0.15, 0.0), (0.85, 0.75)])
    sk = three_component_sketch(atoms, N)
    assert sk["flat_interior_is_consistent"] is False
    assert abs(sk["shortfall"]) > 1e-3


def test_sketch_accepts_a_genuinely_flat_interior():
    """Converse: on a uniform interior the sketch's assumption holds, so the
    flag is not simply always False."""
    atoms = _atoms_from_mixture([(0.1, 0.0), (0.1, 1.0)]
                                + [(0.8 / 199, p) for p in
                                   np.linspace(0.005, 0.995, 199)])
    sk = three_component_sketch(atoms, N)
    assert sk["flat_interior_is_consistent"] is True


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_analyse_theta_emits_every_purity_definition():
    atoms = _atoms_from_mixture([(0.1, 0.0), (0.05, 1.0), (0.85, 0.5)])
    res = analyse_theta(atoms, 100_000, N)
    for g in ("k_le_1", "k_eq_0", "k_ge_Nminus1", "k_eq_N"):
        assert any(k.startswith(g + "|") for k in res["purity"]), g
    # every key carries its definition after the pipe
    assert all("|" in k for k in res["purity"])
    assert res["uniform_leak_into_k_le_1"]["agree"] is True


# ===========================================================================
# Amendment AG/AK additions: the pre-registered endpoint groups, and deriving
# `n_trials` from the COMPARISON-seed count rather than from `n_seeds - 1`.
#
# Mutation kills RUN:
#   * making `_is_unstable_group` fall back to a prefix test -> kills
#     test_an_unknown_group_name_raises_rather_than_picking_a_side.
#   * hardcoding the AG primary groups to [0, 1] / [N-1, N] -> kills
#     test_ag_groups_use_the_committed_cut_at_N39.
#   * reading `n_seeds - 1` when `n_compare_seeds` is present -> kills
#     test_main_derives_n_trials_from_the_comparison_seed_count.
# ===========================================================================

def test_ag_groups_coincide_with_the_legacy_ones_at_N19():
    from src.analysis.rf_mixture import analyse_theta
    N = 19
    atoms = np.full(N + 1, 1.0 / (N + 1))
    res = analyse_theta(atoms, 4000, N)
    for d in ("p_eq_0", "p_lt_0.05", "p_lt_0.10"):
        assert (res["purity"][f"ag_primary_unstable|{d}"]
                == res["purity"][f"k_le_1|{d}"])
    for d in ("p_eq_1", "p_gt_0.95", "p_gt_0.90"):
        assert (res["purity"][f"ag_primary_stable|{d}"]
                == res["purity"][f"k_ge_Nminus1|{d}"])


def test_ag_groups_use_the_committed_cut_at_N39():
    """THE DISCRIMINATING CASE. At N = 39 the pre-registered unstable cut is
    k <= 2, so `ag_primary_unstable` must be a STRICTLY LARGER group than
    `k_le_1`. If the AG groups were hardcoded to the legacy ones the two would
    be equal and the test above would still pass."""
    from src.analysis.rf_mixture import analyse_theta
    N = 39
    atoms = np.full(N + 1, 1.0 / (N + 1))
    res = analyse_theta(atoms, 8000, N)
    ag = res["purity"]["ag_primary_unstable|p_lt_0.10"]["group_size"]
    legacy = res["purity"]["k_le_1|p_lt_0.10"]["group_size"]
    assert ag > legacy
    # And the stable end: k >= 37 is a larger group than k >= 38.
    assert (res["purity"]["ag_primary_stable|p_gt_0.90"]["group_size"]
            > res["purity"]["k_ge_Nminus1|p_gt_0.90"]["group_size"])


def test_ag_sensitivity_groups_are_the_bare_endpoints():
    from src.analysis.rf_mixture import analyse_theta
    N = 39
    atoms = np.full(N + 1, 1.0 / (N + 1))
    res = analyse_theta(atoms, 8000, N)
    assert (res["purity"]["ag_sensitivity_unstable|p_eq_0"]
            == res["purity"]["k_eq_0|p_eq_0"])
    assert (res["purity"]["ag_sensitivity_stable|p_eq_1"]
            == res["purity"]["k_eq_N|p_eq_1"])


def test_an_unknown_group_name_raises_rather_than_picking_a_side():
    """A purity number whose 'genuine' definition was chosen by a prefix test
    is not a number. Unknown names must RAISE."""
    from src.analysis.rf_mixture import _is_unstable_group
    assert _is_unstable_group("k_le_1") is True
    assert _is_unstable_group("k_ge_Nminus1") is False
    assert _is_unstable_group("ag_primary_unstable") is True
    assert _is_unstable_group("ag_primary_stable") is False
    with pytest.raises(KeyError):
        _is_unstable_group("k_lt_3")


def test_main_derives_n_trials_from_the_comparison_seed_count(tmp_path):
    """AT N = 39 `n_seeds - 1` IS 39 BY COINCIDENCE OF ARITHMETIC AND WOULD BE
    WRONG THE MOMENT THE ANCHOR COUNT CHANGED. The fixture therefore uses an
    anchor/comparison split where the two routes DISAGREE: 6 seeds, 4 anchors,
    5 comparison seeds -- `n_seeds - 1` would say 5 too, so instead use 6 seeds
    with `n_compare_seeds = 3`, which no arithmetic on `n_seeds` produces."""
    import subprocess
    import sys
    N_COMPARE = 3
    atoms = list(np.full(N_COMPARE + 1, 1.0 / (N_COMPARE + 1)))
    m1 = tmp_path / "m1.json"
    m1.write_text(json.dumps({
        "identity": {"n_seeds": 6, "n_anchors": 4,
                     "n_compare_seeds": N_COMPARE},
        "marginals_by_theta": {"0.70": {"mnn": {
            "atom_fracs_by_k": atoms, "n_anchors": 400}}},
    }))
    out = tmp_path / "mix.json"
    r = subprocess.run(
        [sys.executable, "-m", "src.analysis.rf_mixture", "--m1", str(m1),
         "--thetas", "0.70", "--out", str(out)],
        capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    doc = json.loads(out.read_text())
    assert doc["identity"]["n_trials"] == N_COMPARE
    # `n_seeds - 1` would have been 5, which is what the old code read.
    assert doc["identity"]["n_trials"] != 5
    assert "4 anchors x 3 comparison seeds" in r.stdout


def test_main_falls_back_to_n_seeds_minus_one_for_an_older_artifact(tmp_path):
    """Older M1 artifacts have no `n_compare_seeds`; they must still read."""
    import subprocess
    import sys
    N = 19
    atoms = list(np.full(N + 1, 1.0 / (N + 1)))
    m1 = tmp_path / "m1.json"
    m1.write_text(json.dumps({
        "identity": {"n_seeds": 20},
        "marginals_by_theta": {"0.70": {"mnn": {
            "atom_fracs_by_k": atoms, "n_anchors": 2000}}},
    }))
    out = tmp_path / "mix.json"
    r = subprocess.run(
        [sys.executable, "-m", "src.analysis.rf_mixture", "--m1", str(m1),
         "--thetas", "0.70", "--out", str(out)],
        capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert json.loads(out.read_text())["identity"]["n_trials"] == 19


def test_ag_cuts_are_recorded_in_the_mixture_identity(tmp_path):
    import subprocess
    import sys
    N = 39
    atoms = list(np.full(N + 1, 1.0 / (N + 1)))
    m1 = tmp_path / "m1.json"
    m1.write_text(json.dumps({
        "identity": {"n_seeds": 40, "n_anchors": 20, "n_compare_seeds": 39},
        "marginals_by_theta": {"0.70": {"mnn": {
            "atom_fracs_by_k": atoms, "n_anchors": 4000}}},
    }))
    out = tmp_path / "mix.json"
    r = subprocess.run(
        [sys.executable, "-m", "src.analysis.rf_mixture", "--m1", str(m1),
         "--thetas", "0.70", "--out", str(out)],
        capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    cuts = json.loads(out.read_text())["identity"]["ag_endpoint_cuts"]
    assert cuts["primary_unstable_k_max"] == 2
    assert cuts["primary_stable_k_min"] == 37
