"""Tests for the persisted degenerate-argmax indicator (amendment AO section 8).

The A1 comparison against AL9b is a CORRECTNESS ASSERTION, so both of its
outcomes are exercised -- a check that can only pass is as useless as one that
can only fail (gotcha 25).
"""

from __future__ import annotations

import json
import sys

import h5py
import numpy as np
import pytest

from src.analysis import degenerate_flag as D

SRC_IDENT = {
    "library_sha": "f67485d9", "n_concepts": 6, "argmax_pool_n": 5,
    "truncation_policy": "untruncated", "scored_row_set": "label_store",
    "inner_mode": "ratio", "null_arm": "b", "n_rows": 15972325,
    "min_protein_count": 5, "retention_floor_mode": "interplm",
    "retention_min_instances": 10, "retention_min_residues": 1500,
}


def _gtilde(tmp_path, M, argmax, concepts, ident=None):
    p = tmp_path / "g.h5"
    with h5py.File(p, "w") as hf:
        hf.create_dataset("M", data=np.asarray(M, float))
        hf.create_dataset("argmax_concept", data=np.asarray(argmax, np.int32))
        hf.create_dataset("concepts",
                          data=np.array(concepts, dtype=h5py.string_dtype()))
        hf.attrs.update(**(ident if ident is not None else SRC_IDENT))
    return p


def _degen(tmp_path, names):
    p = tmp_path / "deg.json"
    p.write_text(json.dumps({"degenerate_intersection": list(names)}))
    return p


# ---------------------------------------------------------------------------
# The flag itself.
# ---------------------------------------------------------------------------
def test_flag_marks_exactly_the_features_whose_argmax_is_degenerate():
    concepts = ["a", "b", "c", "d"]
    argmax = np.array([[0, 1, 2, 3], [3, 3, 0, 1]])
    flag, res = D.build_flag(argmax, concepts, ["b", "d"])
    assert flag.tolist() == [[False, True, False, True],
                            [True, True, False, True]]
    assert res["n_degenerate_resolved"] == 2
    assert res["names_not_in_the_predictor_library"] == []
    assert res["resolved_concept_indices"] == [1, 3]


def test_a_degenerate_name_absent_from_the_library_is_reported_not_dropped():
    """The 22 were identified on one axis; a predictor on another axis could
    silently cover fewer concepts than the arm is specified over."""
    flag, res = D.build_flag(np.array([[0]]), ["a"], ["a", "ghost"])
    assert res["names_not_in_the_predictor_library"] == ["ghost"]
    assert res["n_degenerate_requested"] == 2
    assert res["n_degenerate_resolved"] == 1


def test_no_degenerate_concepts_gives_an_all_false_flag():
    flag, res = D.build_flag(np.array([[0, 1]]), ["a", "b"], [])
    assert flag.tolist() == [[False, False]]
    assert res["n_degenerate_resolved"] == 0


# ---------------------------------------------------------------------------
# A1: BOTH outcomes reachable, and the diff is per field.
# ---------------------------------------------------------------------------
def _deg_block(share=0.0093, lo=0.0070, hi=0.0114, top=0.0142, ratio=1.53):
    return {
        "share_across_seeds": {"median": share, "min": lo, "max": hi},
        "top_decile_share_across_seeds": {"median": top},
        "top_decile_enrichment": {"median_ratio": ratio},
    }


def test_A1_passes_when_the_shares_reproduce_AL9b():
    got = D.check_against_al9b(_deg_block())
    assert got["verdict"] == "PASS"
    assert got["problems"] == []
    assert all(r["agrees"] for r in got["per_field"].values())


def test_A1_fails_and_names_which_quantity_moved():
    got = D.check_against_al9b(_deg_block(top=0.05))
    assert got["verdict"] == "FAIL"
    assert len(got["problems"]) == 1
    assert "top_decile_share_median" in got["problems"][0]
    assert got["per_field"]["share_median"]["agrees"] is True
    assert got["per_field"]["top_decile_share_median"]["agrees"] is False


def test_A1_tolerance_is_half_the_last_reported_digit():
    assert D.check_against_al9b(_deg_block(share=0.0093 + 4e-4))["verdict"] == "PASS"
    assert D.check_against_al9b(_deg_block(share=0.0093 + 6e-4))["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# THE ENTRY POINT. The h5 must carry the predictor's identity (I3), the JSON
# must be written BEFORE a failing A1 raises, and a missing name must refuse.
# ---------------------------------------------------------------------------
def _real_shaped_fixture(tmp_path, n_seeds=20, n_feat=10240, target=0.0093):
    """A fixture whose degenerate-argmax share lands near AL9b's value, so the
    A1 path is exercised on realistic shapes rather than on 3 features."""
    rng = np.random.default_rng(0)
    concepts = [f"C{i}" for i in range(50)]
    degen = concepts[:3]
    argmax = rng.integers(3, 50, size=(n_seeds, n_feat)).astype(np.int32)
    n_hit = int(round(target * n_feat))
    for s in range(n_seeds):
        idx = rng.choice(n_feat, size=n_hit, replace=False)
        argmax[s, idx] = rng.integers(0, 3, size=n_hit)
    M = rng.lognormal(mean=7.5, sigma=1.5, size=(n_seeds, n_feat))
    return _gtilde(tmp_path, M, argmax, concepts), _degen(tmp_path, degen)


def test_main_writes_the_flag_h5_with_the_predictors_identity(tmp_path,
                                                              monkeypatch):
    g, deg = _real_shaped_fixture(tmp_path)
    out, outj = tmp_path / "f.h5", tmp_path / "f.json"
    monkeypatch.setattr(sys, "argv", [
        "flag", "--gtilde", str(g), "--degenerate", str(deg),
        "--out", str(out), "--out-json", str(outj)])
    # A1 is expected to FAIL on synthetic data -- the point of this test is the
    # h5's contents, and the artifacts are written before the raise.
    with pytest.raises(SystemExit):
        D.main()
    with h5py.File(out, "r") as hf:
        assert hf["degenerate_argmax"].shape == (20, 10240)
        assert hf["degenerate_argmax"].dtype == bool
        for key in ("library_sha", "argmax_pool_n", "truncation_policy",
                    "inner_mode", "n_rows"):
            assert key in hf.attrs
        assert hf.attrs["n_degenerate_concepts"] == 3
        assert hf.attrs["measurement"] == "degenerate_argmax_flag_AO8"
    doc = json.loads(outj.read_text())
    assert doc["flag_counts"]["n_feature_seed_cells"] == 20 * 10240
    assert doc["identity"]["firewall_crossing"].startswith("NONE")
    assert "PRE-DECLARED" in doc["sensitivity_arm"]
    assert "NO REGRESSION IS RUN HERE" in doc["sensitivity_arm"]


def test_main_writes_the_report_before_a_failing_A1_raises(tmp_path,
                                                           monkeypatch):
    """A validator whose failure destroys the evidence is not useful. The diff
    has to survive for inspection."""
    g, deg = _real_shaped_fixture(tmp_path, target=0.20)
    outj = tmp_path / "f.json"
    monkeypatch.setattr(sys, "argv", [
        "flag", "--gtilde", str(g), "--degenerate", str(deg),
        "--out", str(tmp_path / "f.h5"), "--out-json", str(outj)])
    with pytest.raises(SystemExit, match="AO4 A1 FAILED"):
        D.main()
    assert outj.exists()
    doc = json.loads(outj.read_text())
    assert doc["AO4_A1_correctness_assertion"]["verdict"] == "FAIL"


def test_main_refuses_a_degenerate_name_absent_from_the_library(tmp_path,
                                                                monkeypatch):
    g, _ = _real_shaped_fixture(tmp_path)
    deg = _degen(tmp_path, ["C0", "NOT_IN_LIBRARY"])
    monkeypatch.setattr(sys, "argv", [
        "flag", "--gtilde", str(g), "--degenerate", str(deg),
        "--out", str(tmp_path / "f.h5"), "--out-json", str(tmp_path / "f.json")])
    with pytest.raises(SystemExit, match="absent from the predictor's library"):
        D.main()


def test_main_exit_is_zero_only_when_A1_passes(tmp_path, monkeypatch):
    """The success path must actually be reachable, or the gate can only fail
    (gotcha 25a). A1's expected values are patched to the fixture's own numbers
    so the comparison is exercised in the PASS direction."""
    g, deg = _real_shaped_fixture(tmp_path)
    out, outj = tmp_path / "f.h5", tmp_path / "f.json"
    monkeypatch.setattr(sys, "argv", [
        "flag", "--gtilde", str(g), "--degenerate", str(deg),
        "--out", str(out), "--out-json", str(outj)])
    # first run to learn the fixture's own shares
    with pytest.raises(SystemExit):
        D.main()
    got = json.loads(outj.read_text())["AO4_A1_correctness_assertion"][
        "per_field"]
    monkeypatch.setattr(D, "EXPECTED_AL9B",
                        {k: v["recomputed"] for k, v in got.items()})
    D.main()                      # must NOT raise
    doc = json.loads(outj.read_text())
    assert doc["AO4_A1_correctness_assertion"]["verdict"] == "PASS"
