"""Tests for the bin-occupancy measurement (PREREGISTRATION amendment AK3 B8).

THE B8 PREDICATE IS AN INSTRUMENT, so gotcha 24 applies: it must be a TOTAL
function of the two statistics, and it must be able to come out MISS. Both are
checked below with planted data whose answer is known by construction.

Mutation kills RUN:
  * `any(...)` -> `all(...)` in the B8 outcome -> kills
    test_one_statistic_is_enough_for_a_HIT.
  * ranking by raw value instead of |value - median| -> kills
    test_extremeness_is_two_sided.
  * exp(entropy) -> count of non-empty bins -> kills
    test_effective_bins_separates_a_concentrated_seed_from_a_spread_one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.analysis.bin_occupancy import (
    FOCUS_SEEDS,
    N_MOST_EXTREME,
    distinct_mu_per_seed,
    effective_bins,
    extreme_ranks,
    occupancy_per_seed,
    score_b8,
)

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# effective_bins
# ---------------------------------------------------------------------------

def test_effective_bins_of_uniform_occupancy_is_the_bin_count():
    assert effective_bins(np.array([25, 25, 25, 25])) == pytest.approx(4.0)


def test_effective_bins_of_a_single_occupied_bin_is_one():
    assert effective_bins(np.array([100, 0, 0, 0])) == pytest.approx(1.0)


def test_effective_bins_separates_a_concentrated_seed_from_a_spread_one():
    """THE DISCRIMINATING CASE, and the reason a non-empty-bin count is not
    used: both vectors occupy all four bins, so a bin COUNT gives 4 for each.
    exp(entropy) must separate them."""
    concentrated = np.array([9997, 1, 1, 1])
    spread = np.array([2500, 2500, 2500, 2500])
    assert (concentrated > 0).sum() == (spread > 0).sum() == 4
    assert effective_bins(concentrated) < 1.01
    assert effective_bins(spread) == pytest.approx(4.0)


def test_effective_bins_of_an_empty_vector_is_nan():
    assert np.isnan(effective_bins(np.array([0, 0])))


# ---------------------------------------------------------------------------
# extremeness
# ---------------------------------------------------------------------------

def test_extremeness_is_two_sided():
    """A seed far BELOW the median is as extreme as one far above. Ranking by
    raw value would pick only one tail, and B8 names neither direction."""
    vals = {0: 100.0, 1: 1.0, 2: 50.0, 3: 51.0, 4: 49.0}
    rk = extreme_ranks(vals)
    assert rk["median"] == pytest.approx(50.0)
    assert set(rk["three_most_extreme"]) >= {0, 1}


def test_extremeness_ties_break_deterministically():
    vals = {0: 60.0, 1: 40.0, 2: 50.0, 3: 50.0, 4: 50.0}
    a = extreme_ranks(vals)["seeds_by_extremeness"]
    b = extreme_ranks(dict(reversed(list(vals.items()))))["seeds_by_extremeness"]
    assert a == b


def test_the_ranking_covers_every_seed():
    vals = {s: float(s) for s in range(20)}
    rk = extreme_ranks(vals)
    assert sorted(rk["seeds_by_extremeness"]) == list(range(20))
    assert len(rk["three_most_extreme"]) == N_MOST_EXTREME


# ---------------------------------------------------------------------------
# the B8 predicate: both branches reachable
# ---------------------------------------------------------------------------

def _distinct(vals):
    return [{"seed": s, "n_distinct_mu": v, "n_finite": 100, "n_features": 100}
            for s, v in enumerate(vals)]


def _occ(effs):
    return {"n_bins": 4, "edges": [0, 1, 2, 3, 4],
            "per_seed": [{"seed": s, "effective_bins": e, "counts_by_bin": [],
                          "n_nonempty_bins": 4, "max_bin_share": 0.25}
                         for s, e in enumerate(effs)]}


def test_b8_hits_when_the_focus_seeds_are_the_outliers():
    n = 20
    d = [50] * n
    d[0], d[1] = 5, 200            # seeds 0 and 1 far from the median
    e = [3.0] * n
    res = score_b8(_distinct(d), _occ(e))
    assert res["outcome"] == "HIT"
    assert res["by_statistic"]["n_distinct_mu"]["both_focus_seeds_present"]


def test_b8_misses_when_the_focus_seeds_are_ordinary():
    """The MISS branch must be reachable, or the predicate is not a predicate.
    Here the outliers are seeds 7, 12 and 19."""
    n = 20
    d = [50] * n
    d[7], d[12], d[19] = 5, 200, 300
    e = [3.0] * n
    e[7], e[12], e[19] = 1.1, 3.9, 3.95
    res = score_b8(_distinct(d), _occ(e))
    assert res["outcome"] == "MISS"
    assert "UNEXPLAINED" in res["reading_if_MISS"]


def test_one_statistic_is_enough_for_a_HIT():
    """B8 says 'at least one of the two statistics'. Occupancy alone must do."""
    n = 20
    d = [50] * n
    d[7], d[12], d[19] = 5, 200, 300      # the other statistic points elsewhere
    e = [3.0] * n
    e[0], e[1] = 1.05, 3.99
    res = score_b8(_distinct(d), _occ(e))
    assert res["by_statistic"]["n_distinct_mu"]["both_focus_seeds_present"] is False
    assert res["by_statistic"]["effective_bins"]["both_focus_seeds_present"] is True
    assert res["outcome"] == "HIT"


def test_b8_reports_the_focus_seed_ranks_either_way():
    """For an INFORMATIVE statistic the focus seeds' ranks are reported whether
    or not they are extreme -- the number is the evidence, not the verdict.

    (Amended for AL1: the fixture now VARIES. It previously passed a constant,
    which is exactly the configuration that produced the spurious HIT, and it
    was asserting that ranks are reported for it.)"""
    n = 20
    res = score_b8(_distinct([50 + s for s in range(n)]), _occ([3.0] * n))
    st = res["by_statistic"]["n_distinct_mu"]
    assert st["status"] == "INFORMATIVE"
    for s in FOCUS_SEEDS:
        assert str(s) in st["focus_seed_ranks"]


# ---------------------------------------------------------------------------
# occupancy uses the SAME bin assignment as the normaliser
# ---------------------------------------------------------------------------

def test_occupancy_totals_every_feature_in_exactly_one_bin():
    rng = np.random.default_rng(0)
    fire = rng.random((5, 300)) * 0.01
    edges = [0.0, 1e-4, 1e-3, 1e-2, 1.0]
    occ = occupancy_per_seed(fire, edges)
    assert occ["n_bins"] == 4
    for row in occ["per_seed"]:
        assert sum(row["counts_by_bin"]) == 300


def test_occupancy_agrees_with_the_normaliser_bin_assignment():
    """Gotcha 20's practice: compute what the dependence implies and confirm the
    code produces it. Occupancy must be the histogram of the SAME `bin_index`
    call the hard-bin normaliser makes, not a second, subtly different one."""
    from src.analysis.null_pooling import bin_index
    rng = np.random.default_rng(1)
    fire = rng.random((3, 200)) * 0.05
    edges = [0.0, 1e-4, 1e-3, 1e-2, 1.0]
    occ = occupancy_per_seed(fire, edges)
    for s, row in enumerate(occ["per_seed"]):
        fb = np.clip(bin_index(fire[s], edges), 0, 4 - 1)
        expect = np.bincount(fb.astype(np.int64), minlength=4).tolist()
        assert row["counts_by_bin"] == expect


def test_distinct_mu_counts_a_step_function_correctly():
    mu = np.array([[1.0, 1.0, 2.0, 2.0, 3.0],
                   [5.0, 5.0, 5.0, 5.0, 5.0]])
    out = distinct_mu_per_seed(mu)
    assert out[0]["n_distinct_mu"] == 3
    assert out[1]["n_distinct_mu"] == 1


def test_distinct_mu_ignores_non_finite_values():
    mu = np.array([[1.0, np.nan, 2.0, np.inf]])
    out = distinct_mu_per_seed(mu)
    assert out[0]["n_distinct_mu"] == 2
    assert out[0]["n_finite"] == 2
    assert out[0]["n_features"] == 4


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------

def _write_inputs(tmp_path, n_seeds=20, n_feats=64, n_cells=6, n_bins=4):
    import h5py
    rng = np.random.default_rng(7)
    edges = [0.0, 1e-4, 1e-3, 1e-2, 1.0]
    g = tmp_path / "gtilde.h5"
    with h5py.File(g, "w") as hf:
        hf.create_dataset("stratum_mu",
                          data=rng.random((n_seeds, n_cells, n_bins)) + 1e-6)
        hf.create_dataset("concept_cell",
                          data=rng.integers(0, n_cells, size=11).astype(np.int64))
        hf.attrs["firing_rate_edges"] = json.dumps(edges)
    nr = tmp_path / "null.h5"
    with h5py.File(nr, "w") as hf:
        hf.create_dataset("fire_rate", data=rng.random((n_seeds, n_feats)) * 0.02)
    return g, nr


def test_main_writes_a_complete_artifact_and_derives_the_outcome(tmp_path):
    g, nr = _write_inputs(tmp_path)
    out = tmp_path / "occ.json"
    r = subprocess.run(
        [sys.executable, "-m", "src.analysis.bin_occupancy",
         "--gtilde", str(g), "--null-reductions", str(nr), "--out", str(out)],
        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    doc = json.loads(out.read_text())
    assert doc["identity"]["measurement"] == "bin_occupancy_AK_B8"
    assert doc["identity"]["firewall_crossing"].startswith("NONE")
    assert len(doc["occupancy"]["per_seed"]) == 20
    assert len(doc["distinct_mu"]) == 20
    assert doc["B8"]["outcome"] in {"HIT", "MISS"}
    assert "B8 OUTCOME:" in r.stdout


def test_main_does_not_open_an_ap_artifact(tmp_path):
    """The measurement needs no crossing, so it must not make one. Asserted by
    running under the X7 firewall, which HARD-FAILS on any alignment artifact."""
    g, nr = _write_inputs(tmp_path, n_seeds=4, n_feats=32)
    out = tmp_path / "occ.json"
    prog = (
        "from src.analysis.rf_marginal import install_firewall;"
        "install_firewall();"
        "import sys;"
        "sys.argv=['x','--gtilde',r'%s','--null-reductions',r'%s',"
        "'--out',r'%s'];"
        "from src.analysis.bin_occupancy import main; main()"
        % (g, nr, out)
    )
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                       text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert out.exists()


# ===========================================================================
# AMENDMENT AL1 -- A RANKING PREDICATE APPLIED TO A CONSTANT.
#
# The first run of this module reported HIT because `n_distinct_mu` was exactly
# 6 in all 20 seeds: every absolute deviation was 0.0, the deterministic
# tie-break by seed index put seeds 0, 1, 2 in the "three most extreme", and the
# two focus seeds are the two lowest indices. THE TIE-BREAK SUPPLIED THE ANSWER.
#
# These tests exist so that cannot recur. Mutation kills RUN:
#   * removing the `informative` guard from `extreme_ranks` -> kills
#     test_a_constant_statistic_is_not_ranked.
#   * scoring an uninformative statistic as satisfying the predicate -> kills
#     test_a_constant_statistic_cannot_produce_a_HIT.
#   * returning False instead of None for an unchecked statistic -> kills
#     test_an_unchecked_statistic_is_None_not_False.
# ===========================================================================

def test_a_constant_statistic_is_not_ranked():
    """THE FOUNDING CASE, with the real numbers: n_distinct_mu = 6 everywhere."""
    rk = extreme_ranks({s: 6.0 for s in range(20)})
    assert rk["spread"] == 0.0
    assert rk["n_unique_values"] == 1
    assert rk["informative"] is False
    assert rk["status"] == "UNINFORMATIVE_CONSTANT"
    # Reported as None so no consumer can read a position out of a tie-break.
    assert rk["three_most_extreme"] is None


def test_a_varying_statistic_is_still_ranked():
    rk = extreme_ranks({s: float(s) for s in range(20)})
    assert rk["informative"] is True
    assert rk["status"] == "INFORMATIVE"
    assert len(rk["three_most_extreme"]) == N_MOST_EXTREME


def test_a_two_valued_statistic_is_informative():
    """The boundary: n_unique == 2 is the least data that can still rank."""
    vals = {s: (1.0 if s < 10 else 2.0) for s in range(20)}
    assert extreme_ranks(vals)["informative"] is True


def test_a_constant_statistic_cannot_produce_a_HIT():
    """The exact configuration of the first run: statistic (i) constant,
    statistic (ii) varying and NOT pointing at the focus seeds. The outcome must
    be MISS -- the constant contributes nothing."""
    n = 20
    d = [6] * n                                  # constant, as measured
    e = [3.0] * n
    e[5], e[19], e[4] = 3.20, 3.55, 3.52          # the real extreme seeds
    res = score_b8(_distinct(d), _occ(e))
    assert res["by_statistic"]["n_distinct_mu"]["status"] == "UNINFORMATIVE_CONSTANT"
    assert res["n_informative_statistics"] == 1
    assert res["outcome"] == "MISS"


def test_an_unchecked_statistic_is_None_not_False():
    """False would read as 'checked and absent'. It was not checked."""
    res = score_b8(_distinct([6] * 20), _occ([3.0] * 20))
    st = res["by_statistic"]["n_distinct_mu"]
    assert st["both_focus_seeds_present"] is None
    assert st["focus_seed_ranks"] is None
    assert "tie-break" in st["why_excluded"]


def test_every_statistic_constant_is_NOT_SCORABLE():
    """Total function (gotcha 24): if nothing is informative the predicate has
    no domain, and that is a third outcome rather than a MISS."""
    res = score_b8(_distinct([6] * 20), _occ([3.0] * 20))
    assert res["outcome"] == "NOT_SCORABLE"
    assert res["n_informative_statistics"] == 0


def test_a_constant_does_not_block_a_genuine_HIT():
    """The guard must not make HIT unreachable -- a check that can only fail is
    as useless as one that can only pass."""
    n = 20
    e = [3.0] * n
    e[0], e[1] = 3.60, 3.55                       # focus seeds genuinely extreme
    res = score_b8(_distinct([6] * n), _occ(e))
    assert res["outcome"] == "HIT"
    assert res["n_informative_statistics"] == 1


# ---------------------------------------------------------------------------
# AL2: the analytic bound on mu_hat's resolution
# ---------------------------------------------------------------------------

def test_mu_hat_resolution_is_bounded_by_the_firing_bin_count(tmp_path):
    """mu_feat[f] = median over concepts of stratum_mu[cell[c], fb[f]], and for
    fixed f the column is a FIXED set determined by fb[f] alone. So the
    distinct-value count cannot exceed the bin count -- computed here rather
    than trusted, per gotcha 20's practice."""
    import h5py
    from src.analysis.m4_confound import hard_bin_mu_per_feature
    rng = np.random.default_rng(3)
    n_seeds, n_feats, n_cells, n_bins = 4, 500, 9, 6
    edges = [0.0, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1.000000001]
    assert len(edges) - 1 == n_bins
    g = tmp_path / "g.h5"
    with h5py.File(g, "w") as hf:
        hf.create_dataset("stratum_mu",
                          data=rng.random((n_seeds, n_cells, n_bins)) + 1e-6)
        hf.create_dataset("concept_cell",
                          data=rng.integers(0, n_cells, size=200).astype(np.int64))
        hf.attrs["firing_rate_edges"] = json.dumps(edges)
    fire = rng.random((n_seeds, n_feats)) * 0.05
    mu, _ = hard_bin_mu_per_feature(g, fire)
    for row in distinct_mu_per_seed(mu):
        assert row["n_distinct_mu"] <= n_bins, row


def test_main_asserts_the_analytic_bound(tmp_path):
    g, nr = _write_inputs(tmp_path, n_seeds=6, n_feats=200)
    out = tmp_path / "occ.json"
    r = subprocess.run(
        [sys.executable, "-m", "src.analysis.bin_occupancy",
         "--gtilde", str(g), "--null-reductions", str(nr), "--out", str(out)],
        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    res = json.loads(out.read_text())["mu_hat_resolution"]
    assert res["analytic_bound_holds"] is True
    assert res["seeds_over_bound"] == []
    assert res["n_firing_bins"] == 4
    assert "analytic bound holds" in r.stdout


def test_the_resolution_bound_DETECTOR_fires_on_a_violation():
    """A check that cannot fail is not a check. Real data cannot violate an
    analytic bound, so the detector is only testable by supplying the violating
    input directly -- which is why `check_resolution_bound` is a separate
    function. The first version was inline in `main()` and the mutation that
    disabled its comparison SURVIVED the suite."""
    from src.analysis.bin_occupancy import check_resolution_bound
    ok = check_resolution_bound(_distinct([6, 6, 6]), 6)
    assert ok["analytic_bound_holds"] is True
    assert ok["seeds_over_bound"] == []
    assert ok["mu_hat_resolution_equals_bin_count"] is True

    bad = check_resolution_bound(_distinct([6, 7, 6]), 6)
    assert bad["analytic_bound_holds"] is False
    assert bad["seeds_over_bound"] == [1]


def test_resolution_below_the_bound_is_not_a_violation():
    """Fewer distinct values than bins is legitimate -- an empty bin gives one
    fewer level. Only EXCEEDING the bound is impossible."""
    from src.analysis.bin_occupancy import check_resolution_bound
    r = check_resolution_bound(_distinct([5, 6, 4]), 6)
    assert r["analytic_bound_holds"] is True
    assert r["mu_hat_resolution_equals_bin_count"] is False


def test_main_exits_nonzero_on_a_bound_violation(tmp_path, monkeypatch):
    """The detector's failure must be readable by its consumer (gotcha 13
    extension 1), so the exit path is checked too, not just the verdict."""
    g, nr = _write_inputs(tmp_path, n_seeds=4, n_feats=64)
    out = tmp_path / "occ.json"
    prog = (
        "import sys;"
        "import src.analysis.bin_occupancy as bo;"
        "bo.check_resolution_bound = lambda d, n: "
        "{'analytic_bound_holds': False, 'seeds_over_bound': [3], "
        "'n_firing_bins': n, 'n_distinct_mu_by_seed': [], "
        "'mu_hat_resolution_equals_bin_count': False, 'reading': ''};"
        "sys.argv=['x','--gtilde',r'%s','--null-reductions',r'%s',"
        "'--out',r'%s'];"
        "bo.main()" % (g, nr, out)
    )
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                       text=True, cwd=REPO)
    assert r.returncode != 0, r.stdout[-2000:]
    assert "ANALYTIC BOUND VIOLATED" in (r.stdout + r.stderr)
    assert not out.exists()
