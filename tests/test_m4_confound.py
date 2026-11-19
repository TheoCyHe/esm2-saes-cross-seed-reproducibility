"""Tests for M4's confound check (amendment AC section 2).

This job crosses the X7 firewall on purpose, so two of the tests are about the
LABELLING of that crossing rather than about arithmetic: an unlabelled
crossing is how a deliberate exception becomes the ambient state.

Mutations run and killed (see the AC commit message):
  * `ols_logM_coef` skipping the standardisation of `log M`
        -> kills test_coefficient_is_scale_invariant
  * the relative-change denominator using `b` instead of `a`
        -> kills test_relative_change_is_measured_against_the_all_four_arm
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.m4_confound import ols_logM_coef, per_seed_logM


def test_ols_recovers_a_planted_slope():
    rng = np.random.default_rng(0)
    n = 5000
    logM = rng.normal(0, 1, n)
    log_fire = rng.normal(-2, 0.5, n)
    # r_f built with a known standardised slope of 0.20 on log M
    r_f = 0.5 + 0.20 * (logM - logM.mean()) / logM.std() \
        + 0.05 * log_fire + rng.normal(0, 0.01, n)
    out = ols_logM_coef(r_f, logM, log_fire)
    assert out["coef_logM"] == pytest.approx(0.20, abs=0.01)
    assert out["n"] == n


def test_coefficient_is_scale_invariant():
    """`log M` under two `mu` models lives on different scales, so the
    coefficient must be standardised or the comparison measures a units
    change rather than a signal change. Multiplying `mu` by a constant shifts
    `log M` by a constant, which must not move the coefficient at all."""
    rng = np.random.default_rng(1)
    n = 4000
    logM = rng.normal(0, 1, n)
    log_fire = rng.normal(-2, 0.5, n)
    r_f = 0.4 + 0.15 * logM + rng.normal(0, 0.05, n)
    a = ols_logM_coef(r_f, logM, log_fire)["coef_logM"]
    b = ols_logM_coef(r_f, logM * 3.0 - 7.0, log_fire)["coef_logM"]
    assert a == pytest.approx(b, rel=1e-8)


def test_per_seed_logM_is_the_pool_max_of_the_ratio():
    """`log M` must be the max over the ARGMAX POOL of g/mu, not over the full
    library -- putting a non-pool concept in the max is amendment I4's defect."""
    real = np.array([[1.0, 2.0], [50.0, 60.0], [3.0, 4.0]])
    mu = np.ones_like(real)
    pool = np.array([True, False, True])          # concept 1 excluded
    lm = per_seed_logM(real, mu, pool)
    assert np.allclose(np.exp(lm), [3.0, 4.0])    # 50/60 must NOT win


def test_per_seed_logM_divides_by_mu():
    real = np.array([[4.0, 9.0]])
    mu = np.array([[2.0, 3.0]])
    lm = per_seed_logM(real, mu, np.array([True]))
    assert np.allclose(np.exp(lm), [2.0, 3.0])


def test_relative_change_is_measured_against_the_all_four_arm():
    """The reported change is |b - a| / |a| with `a` the all-four arm, because
    that is the incumbent the decision is being made about."""
    a, b = 0.20, 0.10
    assert abs(b - a) / abs(a) == pytest.approx(0.5)
    # the other orientation would give 1.0, which is a different claim
    assert abs(b - a) / abs(b) == pytest.approx(1.0)


def test_ols_tolerates_non_finite_rows():
    rng = np.random.default_rng(2)
    n = 500
    logM = rng.normal(0, 1, n)
    log_fire = rng.normal(-2, 0.5, n)
    r_f = 0.5 + 0.1 * logM
    logM[:10] = -np.inf                      # features with M = 0
    out = ols_logM_coef(r_f, logM, log_fire)
    assert out["n"] == n - 10
    assert np.isfinite(out["coef_logM"])


# --------------------------------------------------------- C1 identity gate --
# `smooth_mu_b.h5` carried `measurement`, `arm`, `n_concepts` and
# `argmax_pool_n` and nothing else -- no truncation_policy, no library_sha, no
# floor parameters -- while feeding AC10's pre-registered sensitivity arm and
# the C1 measurement. The C1 path asserted none of it, so a mu_hat built against
# a different label store would have produced a plausible rho.

def _mk_c1_inputs(tmp_path, mu_attrs):
    """A minimal --c1-only input set: an rf_counts file and a mu_smooth file."""
    import h5py
    import numpy as np

    import json as _json
    rf = tmp_path / "rf.h5"
    with h5py.File(rf, "w") as hf:
        hf.create_dataset("match_count", data=np.array([[3, 5], [4, 6]]))
        hf.attrs["n_compare_seeds"] = 19
        hf.attrs["theta"] = 0.70
        # AMENDMENT AS: the anchors are NUMERIC and the normaliser side is
        # not, so both axes must be labelled or the job refuses.
        hf.attrs["seed_labels"] = _json.dumps([0, 1])

    mu = tmp_path / "mu.h5"
    with h5py.File(mu, "w") as hf:
        hf.create_dataset("mu_hat_full", data=np.full((2, 3, 2), 0.5))
        hf.create_dataset("seeds", data=np.array(["1", "0"],
                                                 dtype=h5py.string_dtype()))
        for k, v in mu_attrs.items():
            hf.attrs[k] = v

    lab = tmp_path / "lab.h5"
    with h5py.File(lab, "w") as hf:
        # a label store publishes a LIBRARY, never a pool -- so no
        # argmax_pool_n here, deliberately
        hf.attrs["truncation_policy"] = "untruncated"
        hf.attrs["n_concepts"] = 712
        hf.attrs["git_sha"] = "f67485d"

    ap = tmp_path / "ap.h5"
    with h5py.File(ap, "w") as hf:
        hf.attrs["argmax_pool_n"] = 703      # the pool's real peer
    return rf, mu, lab, ap


GOOD_MU_ATTRS = {"truncation_policy": "untruncated", "n_concepts": 712,
                 "argmax_pool_n": 703, "library_sha": "f67485d"}


def _run_c1(tmp_path, mu_attrs):
    import sys as _sys
    import types
    from src.analysis.m4_confound import c1_only
    rf, mu, lab, ap = _mk_c1_inputs(tmp_path, mu_attrs)
    args = types.SimpleNamespace(
        rf_counts=rf, mu_smooth=mu, labels=lab, ap=ap,
        out=tmp_path / "c1.json")
    c1_only(args)
    return args.out


def test_c1_accepts_a_fully_identified_normaliser(tmp_path):
    out = _run_c1(tmp_path, GOOD_MU_ATTRS)
    assert out.exists()


def test_c1_refuses_a_normaliser_with_an_ABSENT_identity_key(tmp_path):
    """I3: absent is reported absent, never treated as agreement. This is the
    ACTUAL state of `smooth_mu_b.h5` before this session -- it had no
    truncation_policy at all."""
    attrs = dict(GOOD_MU_ATTRS)
    del attrs["truncation_policy"]
    with pytest.raises(SystemExit, match=r"ABSENT"):
        _run_c1(tmp_path, attrs)


def test_c1_refuses_a_normaliser_built_on_a_DIFFERENT_store(tmp_path):
    attrs = dict(GOOD_MU_ATTRS)
    attrs["library_sha"] = "DIFFERENT"
    with pytest.raises(SystemExit, match=r"library_sha"):
        _run_c1(tmp_path, attrs)


def test_c1_refuses_a_normaliser_on_a_DIFFERENT_argmax_pool(tmp_path):
    """The I4 defect in a new location: a max over a different pool is a
    normaliser for a different statistic.

    The peer here is the AP MATRIX, not the label store -- a store publishes a
    library and not a pool, so comparing against it would have been a check that
    could never pass. `--ap` was already a required argument this path read
    nothing from.
    """
    attrs = dict(GOOD_MU_ATTRS)
    attrs["argmax_pool_n"] = 707
    with pytest.raises(SystemExit, match=r"argmax_pool_n"):
        _run_c1(tmp_path, attrs)


# ------------------------------------------------ the HARD-BIN normaliser --
# AC12 reported C1 for hard-bin as not computable "because the per-feature bin
# assignment is not on disk". These tests establish that it was computable, and
# that the two routes to it agree.

def _mk_gtilde(tmp_path, with_stored_bin: bool):
    import h5py
    import json as _json
    import numpy as np

    # 2 seeds, 3 concept cells, 2 firing bins, 4 features
    stratum_mu = np.array([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
                           [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]]])
    cell = np.array([0, 1, 2, 1])                 # 4 concepts -> cells
    edges = [0.0, 1e-3, 1.0]                      # 2 bins
    # features 0,1 in bin 0; features 2,3 in bin 1 -- and seed 1 DISAGREES on
    # feature 0, so a cross-seed mean would give a different answer
    fire = np.array([[1e-4, 5e-4, 5e-3, 5e-3],
                     [5e-3, 5e-4, 5e-3, 5e-3]])

    g = tmp_path / ("gt_stored.h5" if with_stored_bin else "gt_recon.h5")
    with h5py.File(g, "w") as hf:
        hf.create_dataset("stratum_mu", data=stratum_mu)
        hf.create_dataset("concept_cell", data=cell)
        hf.attrs["firing_rate_edges"] = _json.dumps(edges)
        if with_stored_bin:
            from src.analysis.null_pooling import bin_index
            fb = np.stack([np.clip(bin_index(fire[i], edges), 0, 1)
                           for i in range(2)]).astype(np.int8)
            hf.create_dataset("feature_firing_bin", data=fb)
    return g, fire


def test_hard_bin_mu_stored_and_reconstructed_routes_AGREE(tmp_path):
    """The claim that replaces AC12's blocker. If these disagreed, the
    reconstruction would be an approximation and reporting it as absent would
    have been right after all."""
    from src.analysis.m4_confound import hard_bin_mu_per_feature

    g_stored, fire = _mk_gtilde(tmp_path, with_stored_bin=True)
    g_recon, _ = _mk_gtilde(tmp_path, with_stored_bin=False)

    mu_s, prov_s = hard_bin_mu_per_feature(g_stored, fire)
    mu_r, prov_r = hard_bin_mu_per_feature(g_recon, fire)

    assert prov_s["feature_bin_source"] == "stored feature_firing_bin"
    assert prov_r["feature_bin_source"].startswith("reconstructed")
    assert np.array_equal(mu_s, mu_r)


def test_hard_bin_mu_indexes_by_BOTH_axes_not_just_the_concept_cell(tmp_path):
    """A mu that ignored the firing bin would be a different statistic. Feature
    0 and feature 2 share no bin, so their mu must differ even though the
    concept cells are identical for both."""
    from src.analysis.m4_confound import hard_bin_mu_per_feature

    g, fire = _mk_gtilde(tmp_path, with_stored_bin=True)
    mu, _ = hard_bin_mu_per_feature(g, fire)

    # seed 0: feature 0 is in bin 0, feature 2 in bin 1 -> medians 2.0 vs 20.0
    assert mu[0, 0] == pytest.approx(2.0)
    assert mu[0, 2] == pytest.approx(20.0)
    assert mu[0, 0] != mu[0, 2]


def test_hard_bin_mu_is_PER_SEED(tmp_path):
    """Seed 1 disagrees with seed 0 on feature 0's bin, so their mu must differ.
    A cross-seed-mean bin would erase that (gotcha 18)."""
    from src.analysis.m4_confound import hard_bin_mu_per_feature

    g, fire = _mk_gtilde(tmp_path, with_stored_bin=True)
    mu, _ = hard_bin_mu_per_feature(g, fire)
    # seed 0 feature 0 -> bin 0 (median of col 0 = 2.0)
    # seed 1 feature 0 -> bin 1 (median of seed-1 col 1 = 50.0)
    assert mu[0, 0] == pytest.approx(2.0)
    assert mu[1, 0] == pytest.approx(50.0)
