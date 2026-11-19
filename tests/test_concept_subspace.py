"""PREREGISTRATION §CT22 -- concept-subspace reproducibility.

The brief's toy (2 seeds, 10 features): identical subspaces give Omega = 1,
orthogonal give 0, a permuted basis of the same subspace gives 1. Then the
planted cell (gotcha 28): an exceedance of 1 when the other seed's set spans
the same subspace, and not 1 when it does not -- so the instrument is shown
able to produce both answers. Expectations are written down, never computed
from the code under test (gotcha 32b).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis import concept_subspace as cs


def _orth(d, k, seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    return q[:, :k].T                                   # (k, d) orthonormal rows


# ------------------------------------------------------------- the measures --

def test_identical_subspaces_give_one():
    X = _orth(12, 3, 0)
    r = cs.subspace_measures(X, X)
    assert r["omega"] == pytest.approx(1.0, abs=1e-12)
    assert r["theta_max"] == pytest.approx(0.0, abs=1e-6)
    assert r["geodesic"] == pytest.approx(0.0, abs=1e-6)


def test_orthogonal_subspaces_give_zero():
    Q = _orth(12, 6, 1)
    r = cs.subspace_measures(Q[:3], Q[3:])
    assert r["omega"] == pytest.approx(0.0, abs=1e-12)
    assert r["theta_max"] == pytest.approx(np.pi / 2, abs=1e-12)
    assert r["geodesic"] == pytest.approx(np.sqrt(3) * np.pi / 2, abs=1e-12)


def test_a_permuted_and_mixed_basis_of_the_same_subspace_gives_one():
    """The invariance the measure must have: the basis is not the object."""
    X = _orth(12, 3, 2)
    mix = np.array([[0.2, 1.0, -0.4], [1.0, 0.1, 0.3], [-0.5, 0.2, 1.0]])
    Y = (mix @ X)[[2, 0, 1]] * np.array([[3.0], [0.5], [7.0]])   # scaled rows too
    assert cs.subspace_measures(X, Y)["omega"] == pytest.approx(1.0, abs=1e-12)


def test_a_half_shared_subspace_gives_a_value_between():
    """One shared direction of two: cos^2 = (1, 0) -> Omega = 0.5 exactly."""
    Q = _orth(12, 3, 3)
    r = cs.subspace_measures(Q[[0, 1]], Q[[0, 2]])
    assert r["omega"] == pytest.approx(0.5, abs=1e-12)
    assert r["theta_max"] == pytest.approx(np.pi / 2, abs=1e-12)


def test_rank_deficiency_is_detected_at_the_declared_tolerance():
    X = _orth(12, 3, 4)
    dup = np.vstack([X[:2], X[0] + 1e-9 * X[1]])
    assert bool(cs.rank_deficient(dup))
    assert not bool(cs.rank_deficient(X))


def test_measures_are_float64():
    X = _orth(12, 3, 5).astype(np.float32)
    assert cs.basis(X).dtype == np.float64


# -------------------------------------------------------- selection of F -----

def test_top_m_breaks_ties_toward_the_lower_index_like_argmax():
    v = np.array([0.1, 0.9, 0.5, 0.9, 0.5])
    assert cs.top_m(v, 3).tolist() == [1, 3, 2]
    assert cs.top_m(v, 1)[0] == int(np.argmax(v))
    assert cs.tie_at_boundary(v, cs.top_m(v, 3))          # 0.5 at index 4 ties
    assert not cs.tie_at_boundary(v, cs.top_m(v, 2))


def test_top_m_with_too_few_eligible_is_none():
    v = np.arange(6.0)
    elig = np.array([1, 0, 0, 1, 0, 0], bool)
    assert cs.top_m(v, 3, elig) is None
    assert cs.top_m(v, 2, elig).tolist() == [3, 0]


# --------------------------------------------------------- the matched null --

def test_strata_pools_exclude_all_of_F_and_flag_infeasible():
    strat = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    pools = cs.strata_pools(strat, np.array([5, 0, 6]))
    assert [(p.tolist(), q.tolist()) for p, q in pools] == \
        [([1], [1, 2]), ([0, 2], [7, 8])]
    # stratum 1 has two features; F takes both, so nothing is left to draw
    assert cs.strata_pools(strat, np.array([3, 4])) is None


def test_draws_are_matched_distinct_exclude_F_and_reproducible():
    strat = np.repeat(np.arange(3), 6)
    F = np.array([0, 1, 7])
    pools = cs.strata_pools(strat, F)
    a, b = cs.matched_draws(42, pools, pools, 3)
    a2, b2 = cs.matched_draws(42, pools, pools, 3)
    assert np.array_equal(a, a2) and np.array_equal(b, b2)
    assert cs.draws_ok(a, F, strat) and cs.draws_ok(b, F, strat)
    assert not np.array_equal(a, b)          # the s side, then the s' side


def test_draws_ok_can_fail():
    strat = np.repeat(np.arange(3), 6)
    F = np.array([0, 1, 7])
    assert cs.draws_ok(np.array([[2, 3, 6]]), F, strat)       # the positive control
    assert not cs.draws_ok(np.array([[2, 0, 8]]), F, strat)   # contains F
    assert not cs.draws_ok(np.array([[2, 2, 8]]), F, strat)   # repeated
    assert not cs.draws_ok(np.array([[2, 3, 12]]), F, strat)  # wrong stratum


def test_null_summary_orientation_for_similarity_and_distance():
    null = np.arange(100, dtype=float) / 100.0                  # 0.00 .. 0.99
    up = cs.null_summary(0.995, null, larger_is_similar=True)
    assert up["exceedance"] == 1.0 and up["beyond_95"]
    down = cs.null_summary(-0.5, null, larger_is_similar=False)
    assert down["exceedance"] == 1.0 and down["beyond_95"]
    assert down["separation"] > 0 and up["separation"] > 0
    mid = cs.null_summary(0.5, null, larger_is_similar=True)
    assert mid["exceedance"] == 0.5 and not mid["beyond_95"]
    flat = cs.null_summary(0.3, np.full(100, 0.3), True)
    assert flat["null_sd"] == 0.0 and np.isnan(flat["separation"])


# -------------------------------------------- the planted cell (gotcha 28) --

def _toy_world(planted: bool):
    """Two seeds, 10 features each, d = 12, one stratum. Concept 0's top-3 by
    AP in seed 0 are features 0-2; in seed 1, features 0-2 span the SAME
    subspace (planted) or an independent random one (not planted)."""
    rng = np.random.default_rng(7)
    d, n = 12, 10
    dec0 = cs.unit_rows(rng.standard_normal((n, d)))
    dec1 = cs.unit_rows(rng.standard_normal((n, d)))
    if planted:
        mix = rng.standard_normal((3, 3))
        dec1[:3] = cs.unit_rows(mix @ dec0[:3])
    ap = np.zeros((2, n, 1))
    ap[:, :3, 0] = [0.9, 0.8, 0.7]
    sim = np.full((2, 2, n), -np.inf, dtype=np.float32)
    sim[0, 1, 0] = sim[1, 0, 0] = 0.95                           # one MNN pair
    return {"ap": ap, "dec": np.stack([dec0, dec1]),
            "strat": np.zeros((2, n), dtype=np.int64),
            "logM": np.zeros((2, n)), "argmax": np.zeros((2, n), dtype=np.int64),
            "pool_to_gt": np.array([0]), "sim": sim,
            "row_of": np.array([0, 1]), "col_of": np.array([0, 1])}


@pytest.fixture
def world():
    saved = dict(cs._W)
    yield
    cs._W.clear()
    cs._W.update(saved)


def test_planted_cell_has_exceedance_one(world):
    cs._W.clear(); cs._W.update(_toy_world(planted=True))
    r = cs.eval_cell((0, 0, 0, 1, 3, "ap", "real"))
    assert r["feasible"] and r["rank_ok"]
    assert r["omega"] == pytest.approx(1.0, abs=1e-12)
    assert r["omega_null"]["exceedance"] == 1.0
    assert r["omega_null"]["beyond_95"]
    assert r["theta_max_null"]["exceedance"] == 1.0
    # one of three features matched in each direction
    assert r["match_fraction"] == pytest.approx(1 / 3)


def test_unplanted_cell_does_not_reach_exceedance_one(world):
    cs._W.clear(); cs._W.update(_toy_world(planted=False))
    r = cs.eval_cell((0, 0, 0, 1, 3, "ap", "real"))
    assert r["omega"] < 0.9
    assert r["omega_null"]["exceedance"] < 0.95


def test_self_cell_control_passes_and_checks_the_draws(world):
    cs._W.clear(); cs._W.update(_toy_world(planted=False))
    r = cs.eval_cell((0, 0, 0, 0, 3, "ap", "self"))
    assert r["omega"] == pytest.approx(1.0, abs=1e-12) and r["draws_check"]
    # 7 eligible features, 3 per side: C(7,3) = 35 sets, so a few of the 100
    # draws pick one set twice and tie Omega = 1; exactly those fall short.
    exp = r["self_expected_exceedance"]
    assert exp < 1.0
    # THE TIE IS BETWEEN TWO COMPUTATIONS OF ONE NUMBER: the pair alone, and the
    # same pair inside the batch of 100 draws. `null_summary` counts a draw as
    # exceeded when `null < obs`, strictly. With OpenBLAS and with MKL 2025 the
    # two agree to the last bit, the same-set draws tie, and the exceedance is
    # `exp`. With MKL 2026.1 they differ in it (measured 2026-11-13:
    # 0.9999999999999999 alone, 0.9999999999999997 in the batch;
    # analysis/adhoc/probes/2026-11-13/fix_round/self_cell_tie_probe_windows.out),
    # no draw ties, and the exceedance is 1. Either way no draw OTHER than a
    # same-set one falls short, which is what the control is for: a value
    # below `exp` fails here.
    assert r["omega_null"]["exceedance"] in (exp, 1.0)


def test_mode_and_seeds_must_agree(world):
    cs._W.clear(); cs._W.update(_toy_world(planted=False))
    with pytest.raises(ValueError):
        cs.eval_cell((0, 0, 0, 0, 3, "ap", "real"))
    with pytest.raises(ValueError):
        cs.eval_cell((0, 0, 0, 1, 3, "ap", "self"))


def test_infeasible_cell_is_reported_not_patched(world):
    w = _toy_world(planted=True)
    w["strat"][1, :3] = 1                    # F(c,1) alone in its stratum
    cs._W.clear(); cs._W.update(w)
    r = cs.eval_cell((0, 0, 0, 1, 3, "ap", "real"))
    assert r["alignment_ok"] and not r["feasible"] and "omega" not in r


def test_the_logm_alternative_needs_m_argmax_features(world):
    w = _toy_world(planted=True)
    w["argmax"][1] = 1
    w["argmax"][1, :2] = 0                   # only two argmax-c features in seed 1
    cs._W.clear(); cs._W.update(w)
    r = cs.eval_cell((0, 0, 0, 1, 3, "logm_argmax", "real"))
    assert not r["alignment_ok"]


# ---------------------------------------------------------- the lock rule --

def test_lock_takes_the_clear_winner():
    s = {(3, "omega", "ap"): 1.0, (8, "geodesic", "logm_argmax"): 2.0}
    assert cs.lock_configuration(s)["locked"] == \
        {"m": 8, "measure": "geodesic", "alignment": "logm_argmax"}


def test_lock_resolves_ties_measure_then_alignment_then_smallest_m():
    s = {(8, "geodesic", "ap"): 2.00, (5, "omega", "logm_argmax"): 1.95,
         (8, "omega", "ap"): 1.92, (5, "omega", "ap"): 1.91,
         (3, "omega", "ap"): 1.50}
    lk = cs.lock_configuration(s)
    assert lk["locked"] == {"m": 5, "measure": "omega", "alignment": "ap"}
    assert len(lk["tied_set"]) == 4


def test_lock_alignment_preference_applies_before_m():
    s = {(8, "geodesic", "ap"): 2.00, (3, "theta_max", "logm_argmax"): 1.95}
    assert cs.lock_configuration(s)["locked"]["m"] == 8


def test_lock_ties_without_either_primary_fall_to_smallest_m():
    s = {(8, "geodesic", "logm_argmax"): 2.00, (3, "theta_max", "logm_argmax"): 1.95}
    assert cs.lock_configuration(s)["locked"]["m"] == 3


def test_lock_ignores_nan_and_refuses_when_nothing_is_finite():
    s = {(3, "omega", "ap"): float("nan"), (5, "omega", "ap"): 0.4}
    assert cs.lock_configuration(s)["locked"]["m"] == 5
    with pytest.raises(SystemExit):
        cs.lock_configuration({(3, "omega", "ap"): float("nan")})


# ------------------------------------------------------- the confirm verdicts --

def _conc(exc, mf):
    return {"median_exceedance": exc, "median_match_fraction": mf,
            "median_omega": 0.1, "fraction_beyond_95": 0.5}


_ACC_OK = {"share_null_sd_zero": 0.0, "share_rank_deficient": 0.0,
           "share_draw_infeasible": 0.0}


def test_confirm_summary_hit_miss_and_unmeasurable():
    conc = [_conc(0.99, 0.2)] * 40 + [_conc(0.5, 0.2)] * 20 + [_conc(0.99, 0.9)] * 5
    r = cs.confirm_summary(conc, _ACC_OK, "omega")
    assert r["CT22_P1"]["outcome"] == "HIT" and r["CT22_P1"]["hits"] == 45
    assert r["CT22_P2"]["denominator"] == 60 and r["CT22_P2"]["outcome"] == "HIT"
    assert r["crosstab"] == {"subspace_yes_features_yes": 5,
                             "subspace_yes_features_no": 40,
                             "subspace_no_features_yes": 0,
                             "subspace_no_features_no": 20}
    half = cs.confirm_summary([_conc(0.99, 0.9), _conc(0.1, 0.9)], _ACC_OK, "omega")
    assert half["CT22_P1"]["outcome"] == "MISS"              # exactly half is not more
    few = cs.confirm_summary([_conc(0.99, 0.2)] * 29, _ACC_OK, "omega")
    assert few["CT22_P2"]["outcome"] == "UNMEASURABLE"
    assert few["CT22_P1"]["outcome"] == "HIT"
    bad = dict(_ACC_OK, share_draw_infeasible=0.051)
    assert cs.confirm_summary(conc, bad, "omega")["CT22_P1"]["outcome"] == "UNMEASURABLE"


def test_the_launcher_command_parses():
    """slurm/concept_subspace.sbatch's flags are the parser's (a renamed flag
    fails here, not on the cluster)."""
    import re
    from pathlib import Path
    text = Path("slurm/concept_subspace.sbatch").read_text(encoding="utf-8")
    block = text[text.index("EXTRA=()"):text.rindex("if [[")]
    flags = set(re.findall(r"(--[a-z][a-z0-9-]+)", block))
    known = {a for act in cs.build_parser()._actions for a in act.option_strings}
    assert "--tuning-set" in flags and "--pilot" in flags
    assert flags <= known, flags - known
