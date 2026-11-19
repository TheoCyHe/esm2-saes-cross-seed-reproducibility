"""Tests for the predictor-support measurements (PREREGISTRATION amendment AL).

BOTH MEASUREMENTS ARE CHECKS, so both must be shown to FAIL against the
condition they claim to detect (gotcha 13). AL5's answer is expected to be
"the set is empty", which is exactly the answer a broken detector also gives --
so the detector is verified against planted violations.

Mutation kills RUN:
  * dropping the non-finite test from `check_M_finiteness` -> kills
    test_a_planted_nonfinite_M_is_detected.
  * treating `M == 0` as acceptable -> kills test_a_planted_zero_M_is_detected.
  * reporting "EMPTY" whenever the count is falsy rather than zero -> covered by
    test_the_residue_status_can_come_out_NON_EMPTY.
  * comparing `mu <= floor` as `mu < 0` -> kills
    test_the_mu_floor_check_fires_on_a_planted_tiny_stratum.
  * computing the top-decile share over all features instead of the top decile
    -> kills test_top_decile_share_isolates_the_upper_tail.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.analysis.predictor_support import (
    check_M_finiteness,
    check_mu_floor,
    degenerate_argmax_share,
)

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# AL5 -- the log M residue
# ---------------------------------------------------------------------------

def test_a_clean_M_reports_the_residue_as_EMPTY():
    M = np.full((20, 1000), 2.5)
    r = check_M_finiteness(M)
    assert r["n_cells_with_nonfinite_or_nonpositive_M"] == 0
    assert r["residue_status"] == "EMPTY"
    assert "NO branch" in r["reading"]


def test_a_planted_nonfinite_M_is_detected():
    M = np.full((4, 100), 2.5)
    M[2, 7] = np.nan
    M[3, 9] = np.inf
    r = check_M_finiteness(M)
    assert r["residue_status"] == "NON_EMPTY"
    assert r["per_seed"][2]["n_nonfinite"] == 1
    assert r["per_seed"][3]["n_nonfinite"] == 1


def test_a_planted_zero_M_is_detected():
    """`M == 0` is finite but gives `log M = -inf`, so it must count."""
    M = np.full((4, 100), 2.5)
    M[1, 3] = 0.0
    r = check_M_finiteness(M)
    assert r["residue_status"] == "NON_EMPTY"
    assert r["per_seed"][1]["n_exact_zero"] == 1
    assert r["per_seed"][1]["n_nonpositive"] == 1


def test_a_planted_negative_M_is_detected():
    M = np.full((3, 50), 1.0)
    M[0, 0] = -1e-9
    assert check_M_finiteness(M)["residue_status"] == "NON_EMPTY"


def test_the_residue_status_can_come_out_NON_EMPTY():
    """A verdict that can only be EMPTY is not a verdict."""
    assert check_M_finiteness(np.zeros((2, 3)))["residue_status"] == "NON_EMPTY"


def test_the_three_failure_modes_are_reported_separately():
    """Lumping them would hide which one occurred, and they have different
    causes: a NaN comes from the AP path, an exact zero from an all-zero gain."""
    M = np.array([[1.0, np.nan, 0.0, -1.0]])
    r = check_M_finiteness(M)["per_seed"][0]
    assert r["n_nonfinite"] == 1
    assert r["n_exact_zero"] == 1
    assert r["n_nonpositive"] == 2      # the zero and the negative


# ---------------------------------------------------------------------------
# AL5 -- the mu_s floor
# ---------------------------------------------------------------------------

def test_a_bounded_mu_reports_bounded_away_from_zero():
    mu = np.full((20, 64, 6), 3.4e-07)
    r = check_mu_floor(mu, 1e-8)
    assert r["bounded_away_from_zero"] is True
    assert r["n_strata_at_or_below_floor"] == 0
    assert r["global_min_positive_mu"] == pytest.approx(3.4e-07)


def test_the_mu_floor_check_fires_on_a_planted_tiny_stratum():
    mu = np.full((4, 8, 6), 3.4e-07)
    mu[1, 2, 3] = 1e-12
    r = check_mu_floor(mu, 1e-8)
    assert r["bounded_away_from_zero"] is False
    assert r["n_strata_at_or_below_floor"] == 1


def test_the_floor_comparison_is_inclusive():
    """`<=` not `<`: a stratum sitting exactly on the floor is not bounded away
    from it."""
    mu = np.full((2, 4, 6), 3.4e-07)
    mu[0, 0, 0] = 1e-8
    assert check_mu_floor(mu, 1e-8)["n_strata_at_or_below_floor"] == 1


def test_a_zero_stratum_is_excluded_from_the_min_but_counted():
    """Exactly-zero strata are not "positive", so they must not become the
    reported minimum -- but they must not vanish either."""
    mu = np.full((2, 4, 6), 3.4e-07)
    mu[0, 0, 0] = 0.0
    r = check_mu_floor(mu, 1e-8)
    assert r["global_min_positive_mu"] == pytest.approx(3.4e-07)
    assert r["n_strata_at_or_below_floor"] == 1
    assert r["n_strata_finite_positive"] == mu.size - 1


# ---------------------------------------------------------------------------
# AL6 -- the degenerate argmax share
# ---------------------------------------------------------------------------

CONCEPTS = [f"C{i}" for i in range(10)]
DEG = ["C7", "C8", "C9"]


def test_no_degenerate_argmax_gives_a_zero_share():
    M = np.full((3, 100), 2.0)
    am = np.zeros((3, 100), dtype=np.int32)      # everyone picks C0
    r = degenerate_argmax_share(M, am, CONCEPTS, DEG)
    assert r["share_across_seeds"]["median"] == 0.0
    assert r["n_degenerate_concepts"] == 3


def test_all_degenerate_argmax_gives_a_share_of_one():
    M = np.full((3, 100), 2.0)
    am = np.full((3, 100), 7, dtype=np.int32)
    r = degenerate_argmax_share(M, am, CONCEPTS, DEG)
    assert r["share_across_seeds"]["median"] == 1.0


def test_top_decile_share_isolates_the_upper_tail():
    """THE DISCRIMINATING CASE. Degenerate concepts win ONLY for the features
    with the largest log M. The overall share is 10%, the top-decile share is
    100%, and the enrichment is 10x -- which is precisely the configuration AL6
    describes. A top-decile share computed over all features would read 0.10."""
    n = 1000
    M = np.exp(np.linspace(0.0, 5.0, n))[None, :].repeat(2, axis=0)
    am = np.zeros((2, n), dtype=np.int32)
    am[:, -n // 10:] = 8                          # the top 10% by log M
    r = degenerate_argmax_share(M, am, CONCEPTS, DEG)
    assert r["share_across_seeds"]["median"] == pytest.approx(0.1, abs=1e-9)
    assert r["top_decile_share_across_seeds"]["median"] == pytest.approx(
        1.0, abs=0.02)
    assert r["top_decile_enrichment"]["median_ratio"] > 9.0


def test_a_flat_degenerate_distribution_shows_no_enrichment():
    """The control: degenerate wins are spread uniformly over log M, so the
    top-decile share equals the overall share and the enrichment is ~1."""
    rng = np.random.default_rng(0)
    n = 2000
    M = np.exp(rng.normal(0, 1, size=(2, n)))
    am = np.where(rng.random((2, n)) < 0.2, 8, 0).astype(np.int32)
    r = degenerate_argmax_share(M, am, CONCEPTS, DEG)
    assert r["top_decile_enrichment"]["median_ratio"] == pytest.approx(1.0, abs=0.3)


def test_the_logM_shift_is_reported_with_its_sign():
    """Degenerate winners with systematically LARGER log M give a positive
    shift; the sign is the readable part and must not be absolute-valued."""
    n = 400
    M = np.concatenate([np.full(n // 2, 2.0), np.full(n // 2, 100.0)])[None, :]
    am = np.concatenate([np.zeros(n // 2), np.full(n // 2, 9)]).astype(np.int32)[None, :]
    r = degenerate_argmax_share(M, am, CONCEPTS, DEG)
    shift = r["median_logM_shift_degenerate_minus_other"]["median_over_seeds"]
    assert shift == pytest.approx(np.log(100.0) - np.log(2.0), abs=1e-9)
    assert shift > 0


def test_a_degenerate_name_absent_from_the_library_is_reported_not_ignored():
    """Gotcha 15: absent is reported absent. A renamed concept would otherwise
    silently shrink the degenerate set and understate the share."""
    M = np.full((2, 50), 2.0)
    am = np.zeros((2, 50), dtype=np.int32)
    r = degenerate_argmax_share(M, am, CONCEPTS, ["C7", "NOT_A_CONCEPT"])
    assert r["degenerate_names_not_in_library"] == ["NOT_A_CONCEPT"]
    assert r["n_degenerate_concepts"] == 1


def test_an_empty_group_summary_is_nan_not_a_crash():
    from src.analysis.predictor_support import _q
    e = _q(np.array([]))
    assert e["n"] == 0
    assert np.isnan(e["median"])


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------

def _write_inputs(tmp_path, n_seeds=20, n_feats=500, n_cells=8, n_bins=6):
    import h5py
    rng = np.random.default_rng(11)
    g = tmp_path / "g.h5"
    with h5py.File(g, "w") as hf:
        hf.create_dataset("M", data=np.exp(rng.normal(0, 2, (n_seeds, n_feats))))
        hf.create_dataset("argmax_concept",
                          data=rng.integers(0, 10, (n_seeds, n_feats)
                                            ).astype(np.int32))
        hf.create_dataset("concepts",
                          data=np.array(CONCEPTS, dtype=object),
                          dtype=h5py.special_dtype(vlen=str))
        hf.create_dataset("stratum_mu",
                          data=rng.random((n_seeds, n_cells, n_bins)) * 1e-5
                               + 3.4e-07)
        hf.attrs["inner_mode"] = "ratio"
        hf.attrs["n_concepts"] = 10
    d = tmp_path / "deg.json"
    d.write_text(json.dumps({"degenerate_intersection": DEG}))
    return g, d


def test_main_writes_both_measurements(tmp_path):
    g, d = _write_inputs(tmp_path)
    out = tmp_path / "sup.json"
    r = subprocess.run(
        [sys.executable, "-m", "src.analysis.predictor_support",
         "--gtilde", str(g), "--degenerate", str(d), "--out", str(out)],
        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    doc = json.loads(out.read_text())
    assert doc["identity"]["firewall_crossing"].startswith("NONE")
    assert doc["AL5_logM_residue"]["residue_status"] == "EMPTY"
    assert doc["AL5_mu_floor"]["bounded_away_from_zero"] is True
    assert doc["AL6_degenerate_argmax"]["n_degenerate_concepts"] == 3
    assert "AL5 log M residue" in r.stdout
    assert "AL6 degenerate argmax share" in r.stdout


def test_main_does_not_open_an_ap_artifact(tmp_path):
    """Neither measurement needs a firewall crossing, so neither may make one."""
    g, d = _write_inputs(tmp_path, n_seeds=4, n_feats=100)
    out = tmp_path / "sup.json"
    prog = (
        "from src.analysis.rf_marginal import install_firewall;"
        "install_firewall();"
        "import sys;"
        "sys.argv=['x','--gtilde',r'%s','--degenerate',r'%s','--out',r'%s'];"
        "from src.analysis.predictor_support import main; main()"
        % (g, d, out)
    )
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                       text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert out.exists()
