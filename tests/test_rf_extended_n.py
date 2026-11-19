"""Tests for the extended-N comparison (PREREGISTRATION amendments AG, AK).

THE POINT OF THIS FILE. The module is a VALIDATOR, and this project's standing
rule is that a validator is not evidence until IT has been checked (gotcha 13).
Every branch below is verified to FAIL against the condition it claims to
detect, and the process exit code is verified to carry the verdict, because a
validator whose failure is unreadable by the thing gating on it manufactures
confidence inside an automated chain (gotcha 13 extension 1).

Mutation kills RUN, not merely asserted:
  * replacing the float leaf comparison with `abs(a-b) < 1e-9` (a "tolerance")
    -> kills test_a_tiny_value_change_is_a_disagreement.
  * treating an undeclared added key as acceptable -> kills
    test_an_undeclared_added_key_is_a_failure.
  * scoring a MISSED prediction as a gating failure -> kills
    test_a_missed_prediction_does_not_gate.
  * scoring a FAILED correctness assertion as non-gating -> kills
    test_a_failed_assertion_gates.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.analysis.rf_extended_n import (
    DECLARED_ADDED_H5_ATTRS,
    DECLARED_ADDED_JSON_KEYS,
    _matches_declared,
    compare_counts_h5,
    compare_n19,
    json_diff,
    score_predictions,
)


# ---------------------------------------------------------------------------
# json_diff
# ---------------------------------------------------------------------------

def test_identical_documents_have_no_disagreements():
    d = {"a": 1, "b": [1.0, 2.0], "c": {"d": "x"}}
    dis, add = json_diff(d, json.loads(json.dumps(d)))
    assert dis == [] and add == []


def test_a_tiny_value_change_is_a_disagreement():
    """AD2 fixes the tolerance at BIT-IDENTICAL. A 1e-15 move is a change in
    behaviour, and a tolerance introduced here is how a real change gets waved
    through as noise."""
    dis, add = json_diff({"x": 0.1}, {"x": 0.1 + 1e-16})
    if 0.1 + 1e-16 == 0.1:            # below double resolution; use a bigger step
        dis, add = json_diff({"x": 0.1}, {"x": 0.1 + 1e-15})
    assert len(dis) == 1
    assert dis[0]["problem"] == "VALUE_CHANGED"
    assert dis[0]["path"] == "x"


def test_nan_equals_nan_so_a_legitimate_nan_is_not_a_diff():
    dis, _ = json_diff({"x": float("nan")}, {"x": float("nan")})
    assert dis == []


def test_a_missing_baseline_key_is_a_disagreement():
    dis, add = json_diff({"a": 1, "b": 2}, {"a": 1})
    assert [d["problem"] for d in dis] == ["MISSING_IN_CANDIDATE"]
    assert add == []


def test_an_added_key_is_reported_as_added_not_as_a_disagreement():
    dis, add = json_diff({"a": 1}, {"a": 1, "z": 9})
    assert dis == []
    assert add == ["z"]


def test_a_length_change_in_a_list_is_a_disagreement():
    dis, _ = json_diff({"a": [1, 2, 3]}, {"a": [1, 2]})
    assert [d["problem"] for d in dis] == ["LENGTH_CHANGED"]


def test_int_vs_float_is_a_disagreement():
    """`19` and `19.0` compare equal in python; they are not the same artifact
    value, and a type change is exactly the kind of silent refactor drift this
    comparison exists to catch."""
    dis, _ = json_diff({"n": 19}, {"n": 19.0})
    assert len(dis) == 1


# ---------------------------------------------------------------------------
# the declared-additions allowlist
# ---------------------------------------------------------------------------

def test_every_declared_pattern_matches_itself():
    for pat in DECLARED_ADDED_JSON_KEYS:
        concrete = pat.replace("*", "0.70", 1).replace("*", "mnn", 1)
        assert _matches_declared(concrete, DECLARED_ADDED_JSON_KEYS), concrete


def test_a_declared_prefix_covers_its_subtree():
    assert _matches_declared("preflight/b_dec_distinctness/verdict",
                             DECLARED_ADDED_JSON_KEYS)


def test_an_undeclared_key_does_not_match():
    for p in ("identity/something_new", "marginals_by_theta/0.70/mnn/surprise",
              "rotation_null/extra"):
        assert not _matches_declared(p, DECLARED_ADDED_JSON_KEYS), p


def test_the_wildcard_matches_only_one_level():
    """`marginals_by_theta/*/*/ag_endpoint_groups` must not license
    `marginals_by_theta/0.70/ag_endpoint_groups` at the wrong depth."""
    assert _matches_declared(
        "marginals_by_theta/0.70/mnn/ag_endpoint_groups",
        ("marginals_by_theta/*/*/ag_endpoint_groups",))
    assert not _matches_declared(
        "marginals_by_theta/0.70/ag_endpoint_groups",
        ("marginals_by_theta/*/*/ag_endpoint_groups",))


def test_an_undeclared_added_key_is_a_failure(tmp_path):
    base = tmp_path / "b.json"
    cand = tmp_path / "c.json"
    base.write_text(json.dumps({"identity": {"n_seeds": 20}}))
    cand.write_text(json.dumps({"identity": {"n_seeds": 20,
                                             "n_anchors": 20,
                                             "undeclared_thing": 1}}))
    rep = compare_n19(base, cand)
    assert rep["n_baseline_disagreements"] == 0
    assert rep["undeclared_added_keys"] == ["identity/undeclared_thing"]
    assert rep["verdict"] == "DIFFERS"


def test_only_declared_additions_still_reproduce(tmp_path):
    base = tmp_path / "b.json"
    cand = tmp_path / "c.json"
    base.write_text(json.dumps({"identity": {"n_seeds": 20}}))
    cand.write_text(json.dumps({"identity": {"n_seeds": 20, "n_anchors": 20,
                                             "n_compare_seeds": 19}}))
    rep = compare_n19(base, cand)
    assert rep["verdict"] == "REPRODUCES"
    assert rep["n_added_keys"] == 2


# ---------------------------------------------------------------------------
# the counts h5 comparison
# ---------------------------------------------------------------------------

def _write_counts(path: Path, data, extra_attrs=None):
    import h5py
    with h5py.File(path, "w") as hf:
        hf.create_dataset("match_count", data=np.asarray(data, dtype=np.int16))
        hf.attrs["theta"] = 0.7
        hf.attrs["n_compare_seeds"] = 19
        hf.attrs["seed_labels"] = json.dumps(list(range(3)))
        for k, v in (extra_attrs or {}).items():
            hf.attrs[k] = v


def test_counts_reproduce_when_only_declared_attrs_are_added(tmp_path):
    d = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    _write_counts(tmp_path / "b.h5", d)
    _write_counts(tmp_path / "c.h5", d, {"n_anchors": 3,
                                         "compare_only_seed_labels": "[]"})
    rep = compare_counts_h5(tmp_path / "b.h5", tmp_path / "c.h5")
    assert rep["match_count_bit_identical"] is True
    assert rep["verdict"] == "REPRODUCES"


def test_counts_differ_on_one_changed_cell(tmp_path):
    _write_counts(tmp_path / "b.h5", [[1, 2], [3, 4]])
    _write_counts(tmp_path / "c.h5", [[1, 2], [3, 5]])
    rep = compare_counts_h5(tmp_path / "b.h5", tmp_path / "c.h5")
    assert rep["match_count_bit_identical"] is False
    assert rep["n_differing_cells"] == 1
    assert rep["verdict"] == "DIFFERS"


def test_counts_differ_on_a_changed_baseline_attr(tmp_path):
    d = [[1, 2], [3, 4]]
    _write_counts(tmp_path / "b.h5", d)
    _write_counts(tmp_path / "c.h5", d, {"n_compare_seeds": 39})
    rep = compare_counts_h5(tmp_path / "b.h5", tmp_path / "c.h5")
    assert rep["verdict"] == "DIFFERS"
    assert any(a["attr"] == "n_compare_seeds"
               for a in rep["statistic_attr_disagreements"])


def test_counts_differ_on_an_undeclared_added_attr(tmp_path):
    d = [[1, 2], [3, 4]]
    _write_counts(tmp_path / "b.h5", d)
    _write_counts(tmp_path / "c.h5", d, {"surprise": 1})
    rep = compare_counts_h5(tmp_path / "b.h5", tmp_path / "c.h5")
    assert rep["undeclared_added_attrs"] == ["surprise"]
    assert rep["verdict"] == "DIFFERS"


def test_a_shape_change_does_not_crash_the_comparison(tmp_path):
    """At N = 39 the candidate has 20 anchor rows, not 40 -- a shape mismatch
    against a baseline must be REPORTED, not raise."""
    _write_counts(tmp_path / "b.h5", [[1, 2], [3, 4]])
    _write_counts(tmp_path / "c.h5", [[1, 2]])
    rep = compare_counts_h5(tmp_path / "b.h5", tmp_path / "c.h5")
    assert rep["match_count_bit_identical"] is False
    assert rep["n_differing_cells"] is None
    assert rep["verdict"] == "DIFFERS"


# ---------------------------------------------------------------------------
# the AK scorecard
# ---------------------------------------------------------------------------

def _m1_doc(n_seeds, n_anchors, n_compare, n_pairs, *, floor=0.1448,
            mnn=0.4958, oneway=0.4971, f0=0.1221, f1=0.0640, interior=0.8139,
            unstable=0.1710, stable=0.1123, ratio_mono=True):
    grid = [round(0.30 + 0.05 * i, 2) for i in range(13)]
    marg = {}
    for t in grid:
        marg[f"{t:.2f}"] = {
            rule: {
                "pairwise_match_rate": mnn if rule == "mnn" else oneway,
                "frac_r_eq_0": f0, "frac_r_eq_1": f1,
                "frac_interior": interior,
                "frac_k_le_1": unstable, "frac_k_ge_nminus1": stable,
                "n_anchors": n_anchors * 10240,
                "n_compare_seeds": n_compare,
                "ag_endpoint_groups": {
                    "primary_unstable_frac": unstable,
                    "primary_stable_frac": stable,
                    "sensitivity_unstable_frac": f0,
                    "sensitivity_stable_frac": f1,
                },
            } for rule in ("mnn", "oneway")
        }
    return {
        "identity": {"n_seeds": n_seeds, "n_anchors": n_anchors,
                     "n_compare_seeds": n_compare,
                     "n_pairs_computed": n_pairs},
        "rotation_null": {"floor": floor, "mc_se": 0.00024},
        "hungarian_arm": {"rows": [{"theta": 0.7, "iou_mnn_vs_oneway": 0.9997}]},
        "marginals_by_theta": marg,
        "prediction_scorecard": {
            "P1_mnn_nested_in_oneway": {"n_violations": 0, "outcome": "PASS"}},
        "mnn_oneway_ratio_curve": {
            "ratio_at_reference_theta": mnn / oneway,
            "is_non_decreasing_in_theta": ratio_mono,
            "first_decrease_at_theta": None if ratio_mono else 0.5,
            "rows": [{"theta": t, "ratio": mnn / oneway} for t in grid],
        },
    }


def _mix_doc(interval, thirds, shape="BROAD"):
    return {"by_theta": {"0.70": {
        "w0_profile": {"w0_interval": list(interval),
                       "w0_interval_width": interval[1] - interval[0],
                       "n_points": 41},
        "bimodality": {"mass_in_thirds": list(thirds), "shape": shape},
    }}}


def test_scorecard_all_hits_on_a_well_behaved_extension():
    n19 = _m1_doc(20, 20, 19, 190)
    n39 = _m1_doc(40, 20, 39, 590, mnn=0.4960, oneway=0.4973,
                  f0=0.0800, f1=0.0300, interior=0.8900,
                  unstable=0.1200, stable=0.0700)
    sc = score_predictions(n19, n39,
                           _mix_doc((0.0, 0.075), (0.3263, 0.3371, 0.3367)),
                           _mix_doc((0.0, 0.050), (0.3300, 0.3350, 0.3350)))
    outcomes = {k: v["outcome"] for k, v in sc["predictions"].items()}
    assert set(outcomes.values()) <= {"PASS", "HIT"}, outcomes
    # 4 correctness assertions (A1-A4) and 9 scored predictions
    # (B1-B7, C1-C2). Counted explicitly so a prediction silently dropped from
    # the scorecard shows up here rather than passing as "all hits".
    assert sc["tally"].get("PASS") == 4
    assert sc["tally"].get("HIT") == 9
    assert len(sc["predictions"]) == 13


def test_scorecard_flags_a_20_seed_run_labelled_as_40():
    """THE GLOB HAZARD, as a scorecard outcome. A run that read 20 seeds must
    FAIL A1, not merely look unremarkable."""
    n19 = _m1_doc(20, 20, 19, 190)
    n39 = _m1_doc(20, 20, 19, 190)
    sc = score_predictions(n19, n39, None, None)
    assert sc["predictions"]["A1_consumed_40_dictionaries"]["outcome"] == "FAIL"
    # And the endpoint atoms cannot have decreased if nothing changed.
    assert sc["predictions"]["A4_endpoint_atoms_decrease_with_N"][
        "outcome"] == "FAIL"


def test_scorecard_flags_a_moved_rotation_null():
    n19 = _m1_doc(20, 20, 19, 190, floor=0.1448)
    n39 = _m1_doc(40, 20, 39, 590, floor=0.1449)
    sc = score_predictions(n19, n39, None, None)
    a2 = sc["predictions"]["A2_rotation_null_and_hungarian_bit_identical"]
    assert a2["outcome"] == "FAIL"
    assert a2["rotation_bit_identical"] is False


def test_scorecard_flags_disjoint_w0_intervals():
    n19 = _m1_doc(20, 20, 19, 190)
    n39 = _m1_doc(40, 20, 39, 590)
    sc = score_predictions(n19, n39,
                           _mix_doc((0.0, 0.075), (0.33, 0.33, 0.34)),
                           _mix_doc((0.20, 0.30), (0.33, 0.33, 0.34)))
    assert sc["predictions"]["B2_w0_interval_intersects"]["outcome"] == "MISS"


def test_a_tied_w0_interval_width_is_a_miss():
    """Declared in AK3 B3 before the run: a tie is a MISS."""
    n19 = _m1_doc(20, 20, 19, 190)
    n39 = _m1_doc(40, 20, 39, 590)
    sc = score_predictions(n19, n39,
                           _mix_doc((0.0, 0.075), (0.33, 0.33, 0.34)),
                           _mix_doc((0.0, 0.075), (0.33, 0.33, 0.34)))
    assert sc["predictions"]["B3_w0_interval_narrows"]["outcome"] == "MISS"


def test_scorecard_reports_not_scored_without_a_mixture():
    n19 = _m1_doc(20, 20, 19, 190)
    n39 = _m1_doc(40, 20, 39, 590)
    sc = score_predictions(n19, n39, None, None)
    assert sc["predictions"]["B2_w0_interval_intersects"][
        "outcome"] == "NOT_SCORED"


def test_scorecard_flags_a_non_monotone_ratio_curve():
    n19 = _m1_doc(20, 20, 19, 190)
    n39 = _m1_doc(40, 20, 39, 590, ratio_mono=False)
    sc = score_predictions(n19, n39, None, None)
    assert sc["predictions"]["B5_mnn_oneway_ratio"]["outcome"] == "MISS"


def test_side_by_side_spans_the_whole_grid_and_both_rules():
    from src.analysis.rf_extended_n import side_by_side
    n19 = _m1_doc(20, 20, 19, 190)
    n39 = _m1_doc(40, 20, 39, 590)
    rows = side_by_side(n19, n39)
    assert len(rows) == 13 * 2
    assert {r["rule"] for r in rows} == {"mnn", "oneway"}


# ---------------------------------------------------------------------------
# the ENTRY POINT and its exit code (gotcha 13 extension 1 / extension 4)
# ---------------------------------------------------------------------------

def _run(args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "src.analysis.rf_extended_n", *map(str, args)],
        capture_output=True, text=True,
        cwd=cwd or Path(__file__).resolve().parents[1])


def test_main_exits_zero_when_the_canary_reproduces(tmp_path):
    base, cand, out = tmp_path / "b.json", tmp_path / "c.json", tmp_path / "o.json"
    base.write_text(json.dumps({"identity": {"n_seeds": 20}}))
    cand.write_text(json.dumps({"identity": {"n_seeds": 20, "n_anchors": 20}}))
    r = _run(["--baseline-m1", base, "--candidate-m1-n19", cand, "--out", out])
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(out.read_text())["verdict"] == "OK"


def test_main_exits_nonzero_when_the_canary_moves(tmp_path):
    base, cand, out = tmp_path / "b.json", tmp_path / "c.json", tmp_path / "o.json"
    base.write_text(json.dumps({"identity": {"n_seeds": 20}, "x": 1.0}))
    cand.write_text(json.dumps({"identity": {"n_seeds": 20}, "x": 1.0000001}))
    r = _run(["--baseline-m1", base, "--candidate-m1-n19", cand, "--out", out])
    assert r.returncode == 1, r.stdout + r.stderr
    doc = json.loads(out.read_text())
    assert doc["verdict"] == "FAILED"
    assert "canary_m1" in doc["failures"]


def test_a_failed_assertion_gates(tmp_path):
    """A correctness ASSERTION failing is a defect and must set the exit code."""
    n19, n39, out = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "o.json"
    n19.write_text(json.dumps(_m1_doc(20, 20, 19, 190)))
    n39.write_text(json.dumps(_m1_doc(20, 20, 19, 190)))   # never read 40
    r = _run(["--candidate-m1-n19", n19, "--m1-n39", n39, "--out", out])
    assert r.returncode == 1, r.stdout + r.stderr
    fails = json.loads(out.read_text())["failures"]
    assert any(f.startswith("assertion:A1") for f in fails)


def test_a_missed_prediction_does_not_gate(tmp_path):
    """A MISSED prediction is a legitimate outcome and must NOT gate -- the
    ledger exists because predictions keep getting made and sometimes miss."""
    n19, n39, out = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "o.json"
    n19.write_text(json.dumps(_m1_doc(20, 20, 19, 190)))
    # 40 dictionaries read correctly, but every B/C prediction misses.
    n39.write_text(json.dumps(_m1_doc(
        40, 20, 39, 590, mnn=0.9000, oneway=0.9100, f0=0.0100, f1=0.0100,
        interior=0.4000, unstable=0.4000, stable=0.4000, ratio_mono=False)))
    r = _run(["--candidate-m1-n19", n19, "--m1-n39", n39, "--out", out])
    doc = json.loads(out.read_text())
    tally = doc["ak_scorecard"]["tally"]
    assert tally.get("MISS", 0) >= 3, tally
    assert r.returncode == 0, r.stdout + r.stderr
    assert doc["verdict"] == "OK"


def test_declared_h5_attrs_are_exactly_the_two_added_ones():
    assert set(DECLARED_ADDED_H5_ATTRS) == {"n_anchors",
                                            "compare_only_seed_labels"}


# ===========================================================================
# AL7 -- a bit-identity criterion applied to a PROVENANCE key is a check that
# can only fail. Found by running the comparison: the first run reported one
# disagreement, `identity/git_sha`, with match_count bit-identical.
#
# Mutation kills RUN:
#   * exempting by LEAF NAME instead of full path -> kills
#     test_the_ensembles_git_sha_is_STILL_compared.
#   * letting a provenance difference gate the verdict -> kills
#     test_a_git_sha_difference_alone_still_REPRODUCES.
#   * exempting a third key -> kills test_the_exempt_set_is_exactly_two_keys.
# ===========================================================================

def test_the_exempt_set_is_exactly_two_keys():
    """The exemption's only real bound is that growing it requires editing this
    test, so the diff makes it visible. Asserted exactly, not as a subset."""
    from src.analysis.rf_extended_n import (
        PROVENANCE_H5_ATTRS_EXPECTED_TO_DIFFER,
        PROVENANCE_PATHS_EXPECTED_TO_DIFFER,
    )
    assert PROVENANCE_PATHS_EXPECTED_TO_DIFFER == ("identity/git_sha",
                                                   "identity/git_dirty")
    assert PROVENANCE_H5_ATTRS_EXPECTED_TO_DIFFER == ("git_sha", "git_dirty")


def test_a_git_sha_difference_alone_still_REPRODUCES(tmp_path):
    base, cand = tmp_path / "b.json", tmp_path / "c.json"
    base.write_text(json.dumps({"identity": {"git_sha": "aaa", "n_seeds": 20},
                                "rotation_null": {"floor": 0.1449}}))
    cand.write_text(json.dumps({"identity": {"git_sha": "bbb", "n_seeds": 20},
                                "rotation_null": {"floor": 0.1449}}))
    rep = compare_n19(base, cand)
    assert rep["n_statistic_disagreements"] == 0
    assert rep["n_provenance_disagreements"] == 1
    assert rep["verdict"] == "REPRODUCES"


def test_a_statistic_difference_beside_a_git_sha_difference_still_DIFFERS(tmp_path):
    """The exemption must not shelter a real change that happens to travel with
    a provenance change -- which is the only way it could do harm."""
    base, cand = tmp_path / "b.json", tmp_path / "c.json"
    base.write_text(json.dumps({"identity": {"git_sha": "aaa"},
                                "rotation_null": {"floor": 0.1449}}))
    cand.write_text(json.dumps({"identity": {"git_sha": "bbb"},
                                "rotation_null": {"floor": 0.1450}}))
    rep = compare_n19(base, cand)
    assert rep["n_statistic_disagreements"] == 1
    assert rep["statistic_disagreements"][0]["path"] == "rotation_null/floor"
    assert rep["verdict"] == "DIFFERS"


def test_the_ensembles_git_sha_is_STILL_compared(tmp_path):
    """The ensemble's shared SHA -- the design's central invariant -- must stay
    compared while the RUN's SHA is exempt.

    In the CURRENT artifact the ensemble SHA sits at
    `preflight/ensemble_identity/keys/git_sha/value`, whose LEAF is `value`, so
    a leaf-name exemption would coincidentally spare it. This test therefore
    checks the real shape AND the direct-leaf shape below, because relying on
    that coincidence is what made the first version of this test non-
    discriminating (the leaf-name mutation survived it)."""
    base, cand = tmp_path / "b.json", tmp_path / "c.json"
    payload = lambda run, ens: json.dumps({
        "identity": {"git_sha": run},
        "preflight": {"ensemble_identity": {"keys": {
            "git_sha": {"status": "IDENTICAL", "value": ens}}}}})
    base.write_text(payload("aaa", "61745baa1d"))
    cand.write_text(payload("bbb", "DIFFERENT"))
    rep = compare_n19(base, cand)
    assert rep["n_provenance_disagreements"] == 1          # the run's SHA
    assert rep["n_statistic_disagreements"] == 1           # the ensemble's SHA
    assert (rep["statistic_disagreements"][0]["path"]
            == "preflight/ensemble_identity/keys/git_sha/value")
    assert rep["verdict"] == "DIFFERS"


def test_a_git_sha_LEAF_at_another_depth_is_still_compared(tmp_path):
    """THE ACTUALLY DISCRIMINATING CASE. A path whose LEAF is `git_sha` but
    whose FULL PATH is not `identity/git_sha` must remain a statistic
    disagreement. Nothing in today's artifact has that shape -- this guards the
    next writer that nests a bare `git_sha` leaf, which is exactly when a
    leaf-name exemption would silently stop checking an invariant."""
    base, cand = tmp_path / "b.json", tmp_path / "c.json"
    base.write_text(json.dumps({"identity": {"git_sha": "aaa"},
                                "preflight": {"git_sha": "61745baa1d"}}))
    cand.write_text(json.dumps({"identity": {"git_sha": "bbb"},
                                "preflight": {"git_sha": "DIFFERENT"}}))
    rep = compare_n19(base, cand)
    assert rep["n_provenance_disagreements"] == 1
    assert rep["n_statistic_disagreements"] == 1
    assert rep["statistic_disagreements"][0]["path"] == "preflight/git_sha"
    assert rep["verdict"] == "DIFFERS"


def test_counts_h5_git_sha_difference_alone_still_REPRODUCES(tmp_path):
    d = [[1, 2], [3, 4]]
    _write_counts(tmp_path / "b.h5", d, {"git_sha": "aaa"})
    _write_counts(tmp_path / "c.h5", d, {"git_sha": "bbb"})
    rep = compare_counts_h5(tmp_path / "b.h5", tmp_path / "c.h5")
    assert rep["match_count_bit_identical"] is True
    assert rep["statistic_attr_disagreements"] == []
    assert len(rep["provenance_attr_disagreements"]) == 1
    assert rep["verdict"] == "REPRODUCES"
