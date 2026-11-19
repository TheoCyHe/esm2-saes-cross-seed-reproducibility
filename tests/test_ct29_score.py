"""§CT29's scorer: every verdict of each rule is reachable, and the per-arm
causes fire on the fixtures that carry them. Expectations are written down
here, not derived from the code (gotcha 32b)."""
import json

import pytest

from src.analysis import ct29_score as cs


def test_declared_constants():
    # §CT29-F item 10 / CT29-pre item 9; item 7's thresholds
    assert cs.P1_COMPARATOR == {"k64_w2560": 0.6015277777777777,
                                "k64_w40960": 0.3365565321180556}
    assert cs.DIVERGENCE == {"k64_w2560": 0.9237356483253751,
                             "k64_w40960": 0.568616779819711,
                             "k16_w10240": 0.7272, "k128_w10240": 0.7272}
    assert cs.MIN_SCORING == 8


def test_p1_all_branches():
    assert cs.score_p1({"k64_w2560": 0.5, "k64_w40960": 0.3}) == "HIT"
    # at the comparator is not strictly below: MISS
    assert cs.score_p1({"k64_w2560": 0.6015277777777777, "k64_w40960": 0.3}) == "MISS"
    # MISS takes precedence over the other arm's state
    assert cs.score_p1({"k64_w2560": None, "k64_w40960": 0.34}) == "MISS"
    assert cs.score_p1({"k64_w2560": None, "k64_w40960": 0.3}) == "UNMEASURABLE"
    assert cs.score_p1({"k64_w2560": None, "k64_w40960": None}) == "UNMEASURABLE"


def test_p2_all_branches():
    assert cs.score_p2([1.0, 2.0, 3.0, 4.0]) == "HIT"
    assert cs.score_p2([4.0, 3.0, 2.0, 1.0]) == "HIT"
    assert cs.score_p2([1.0, 3.0, 2.0, 4.0]) == "MISS"
    assert cs.score_p2([1.0, 1.0, 2.0, 3.0]) == "MISS"      # a tie is not strict
    assert cs.score_p2([1.0, 3.0, 2.0, None]) == "UNMEASURABLE"


def test_p3_closed_intervals_and_precedence():
    assert cs.intervals_share_point([0.0, 1.0], [1.0, 2.0])    # touching is sharing
    assert not cs.intervals_share_point([0.0, 1.0], [1.0001, 2.0])
    share, apart = ([0, 1], [0.5, 2]), ([0, 1], [2, 3])
    assert cs.score_p3([share, share]) == "HIT"
    assert cs.score_p3([share, apart]) == "MISS"
    assert cs.score_p3([(None, [0, 1]), apart]) == "MISS"
    assert cs.score_p3([(None, [0, 1]), share]) == "UNMEASURABLE"


def test_p4_all_branches():
    assert cs.score_p4([0.40, 0.36, 0.32, 0.29]) == "HIT"
    assert cs.score_p4([0.40, 0.36, 0.32, 0.33]) == "MISS"
    assert cs.score_p4([0.40, 0.36, 0.36, 0.29]) == "MISS"
    assert cs.score_p4([None, 0.36, 0.32, 0.29]) == "UNMEASURABLE"


def _fvu(path, values):
    path.write_text(json.dumps({"per_seed": {str(s): {"raw": {"fvu": v}}
                                             for s, v in enumerate(values)}}),
                    encoding="utf-8")


def _arm(path, fe, ci, verdict="BIT_IDENTICAL", failed=0, n=10, comparison=None):
    f = {"own": {"fe_pp": fe, "fe_ci95_pp": ci, "widest": "jackknife"},
         "shared": {"fe_ci95_pp": ci}, "n_anchors": n, "n_anchor_fits_failed": failed}
    path.write_text(json.dumps({
        "identity": {"comparison_set": comparison or [str(s) for s in range(n)]},
        "control_0.70_vs_record": {"verdict": verdict}, "at_theta_star": f}),
        encoding="utf-8")


def _world(tmp_path):
    ct, csd = tmp_path / "ct", tmp_path / "cs"
    ct.mkdir()
    csd.mkdir()
    for a in cs.GRID:
        _fvu(ct / f"fvu_eval_{a}_ct.json",
             {"k16_w10240": [0.40] * 10, "k128_w10240": [0.29] * 10}.get(a, [0.2] * 10))
        for f in (f"gtilde_{a}_b_ratio_ct.h5", f"feature_ap_{a}_ct.h5",
                  f"null_pilot_{a}_ct.h5"):
            (ct / f).write_bytes(b"")
        (ct / f"rf_marginal_m1_{a}_ct.json").write_text(json.dumps(
            {"identity": {"n_compare_seeds": 9},
             "marginals_by_theta": {"0.70": {"mnn": {"mean": 0.1}}}}), encoding="utf-8")
    _arm(ct / "ct28_arm_k16_w10240_ct.json", 1.0, [0.5, 1.5])
    _arm(ct / "ct28_arm_k128_w10240_ct.json", 4.0, [3.5, 4.5])
    _arm(ct / "ct28_arm_k64_w40960_ct.json", 0.0, [-1, 1.2])
    _arm(ct / "ct28_arm_k64_w2560_ct.json", 0.0, [-1, 1])
    _arm(ct / "ct28_arm_w2560_ct.json", 0.11, [-0.3, 0.54])
    _arm(ct / "ct28_arm_w10240_ct.json", 2.0, [1.5, 2.5])
    _arm(ct / "ct28_arm_k64n9_ct.json", 3.0, [2.5, 3.5], verdict="NO_REFERENCE")
    _fvu(csd / "fvu_eval_w10240.json", [0.36] * 10)
    _fvu(ct / "fvu_eval_k64_ct.json", [0.32] * 20)
    return ct, csd


def _score(ct, csd, external=None):
    return cs.score_all(ct, csd, external or {}, detection=False, devices=False)


def test_score_all_end_to_end(tmp_path):
    out = _score(*_world(tmp_path))
    assert out["CT29-P1"]["verdict"] == "HIT"            # 0.1 below both comparators
    assert out["CT29-P2"]["verdict"] == "HIT"            # 1 < 2 < 3 < 4
    assert out["CT29-P3"]["verdict"] == "MISS"           # [3.5, 4.5] vs [-0.3, 0.54]
    assert out["CT29-P4"]["verdict"] == "HIT"            # 0.40 > 0.36 > 0.32 > 0.29


def test_absent_input_is_not_run_and_refused(tmp_path):
    ct, csd = _world(tmp_path)
    (ct / "gtilde_k16_w10240_b_ratio_ct.h5").unlink()
    with pytest.raises(SystemExit, match="not run yet"):
        _score(ct, csd)
    # a declared cause removes the requirement and makes the arm UNMEASURABLE
    out = _score(ct, csd, {"k16_w10240": "null degenerate (job log)"})
    assert out["arm_status"]["k16_w10240"]["status"] == "UNMEASURABLE"
    assert out["CT29-P2"]["verdict"] == "UNMEASURABLE"


def test_theta_star_failure_scoped_to_p2_p3(tmp_path):
    ct, csd = _world(tmp_path)
    _arm(ct / "ct28_arm_k64_w2560_ct.json", 0.0, [-1, 1], verdict="DIFFERS")
    _arm(ct / "ct28_arm_k16_w10240_ct.json", 1.0, [0.5, 1.5], failed=3)   # 7 scoring
    out = _score(ct, csd)
    assert out["arm_status"]["k64_w2560"]["status"] == "MEASURED"
    assert out["CT29-P1"]["verdict"] == "HIT"            # P1 does not read θ*
    assert out["CT29-P4"]["verdict"] == "HIT"            # nor does P4
    assert out["CT29-P2"]["verdict"] == "UNMEASURABLE"   # k = 16 lost
    assert "7 anchor fits scoring" in out["theta_star_figure_missing_because"]["k16_w10240"]


def test_grid_arm_needs_bit_identical_side_may_have_no_reference(tmp_path):
    ct, csd = _world(tmp_path)
    _arm(ct / "ct28_arm_k128_w10240_ct.json", 4.0, [3.5, 4.5], verdict="NO_REFERENCE")
    out = _score(ct, csd)
    assert out["CT29-P2"]["theta_star_own_fe_pp"]["k=128"] is None
    assert out["CT29-P2"]["theta_star_own_fe_pp"]["k=64"] == 3.0     # NO_REFERENCE side


def test_side_with_one_failed_fit_is_lost(tmp_path):
    ct, csd = _world(tmp_path)
    _arm(ct / "ct28_arm_w2560_ct.json", 0.11, [-0.3, 0.54], failed=1)
    out = _score(ct, csd)
    assert out["CT29-P3"]["pair_one_k128_w10240_vs_k32_w2560"][1] is None


def test_divergence_kept_mean_and_refusal(tmp_path):
    ct, csd = _world(tmp_path)
    _fvu(ct / "fvu_eval_k16_w10240_ct.json", [0.40] * 9 + [0.90])
    _arm(ct / "ct28_arm_k16_w10240_ct.json", 1.0, [0.5, 1.5], n=9,
         comparison=[str(s) for s in range(9)])
    out = _score(ct, csd)
    st = out["arm_status"]["k16_w10240"]
    assert st["diverged_seeds"] == [9] and st["status"] == "MEASURED"
    assert out["CT29-P4"]["mean_raw_fvu_k16_k32_k64_k128"][0] == pytest.approx(0.40)
    assert out["CT29-P4"]["all_ten_means_k16_k128"][0] == pytest.approx(0.45)
    # three diverged: 7 kept, the arm UNMEASURABLE
    _fvu(ct / "fvu_eval_k16_w10240_ct.json", [0.40] * 7 + [0.9] * 3)
    assert _score(ct, csd)["arm_status"]["k16_w10240"]["status"] == "UNMEASURABLE"
    # a diverged seed left inside the fit: refused
    _fvu(ct / "fvu_eval_k16_w10240_ct.json", [0.40] * 9 + [0.90])
    _arm(ct / "ct28_arm_k16_w10240_ct.json", 1.0, [0.5, 1.5])
    with pytest.raises(SystemExit, match="diverged"):
        _score(ct, csd)


def test_p1_excluded_seed_changes_n_compare(tmp_path):
    ct, csd = _world(tmp_path)
    (ct / "rf_marginal_m1_k64_w2560_ct.json").write_text(json.dumps(
        {"identity": {"n_compare_seeds": 8},
         "marginals_by_theta": {"0.70": {"mnn": {"mean": 0.4}}}}), encoding="utf-8")
    out = _score(ct, csd)
    assert out["CT29-P1"]["verdict"] == "UNMEASURABLE"
    assert "not 9" in out["CT29-P1"]["unmeasured_because"]["k64_w2560"]


def test_exemption_and_devices_read_from_identity(tmp_path):
    import h5py
    a = "k64_w2560"
    with h5py.File(tmp_path / f"gtilde_{a}_b_ratio_ct.h5", "w") as hf:
        hf.attrs["accept_ap_arith_split"] = True
        hf.attrs["ap_arith_splits_accepted"] = json.dumps(["cpu:x86_64 vs cuda"])
    for f, g in ((f"feature_ap_{a}_ct.h5", "cuda:A"), (f"null_pilot_{a}_ct.h5", "cuda:A")):
        with h5py.File(tmp_path / f, "w") as hf:
            hf.attrs["gather_device_class"] = g
    d = cs.exemption_and_devices(tmp_path, a)
    assert d["gtilde"]["accept_ap_arith_split"] is True
    assert json.loads(d["gtilde"]["ap_arith_splits_accepted"]) == ["cpu:x86_64 vs cuda"]
    assert d["feature_ap"]["gather_device_class"] == "cuda:A"
    assert d["null_pilot"]["ap_arith_device_class"] is None       # absent, reported absent


def test_main_refuses_absent_input_before_hashing(tmp_path):
    ct, csd = _world(tmp_path)
    (ct / "ct28_arm_w2560_ct.json").unlink()
    with pytest.raises(SystemExit, match="not run yet"):
        cs.main(["--ct-dir", str(ct), "--cs-dir", str(csd), "--out", str(tmp_path / "o.json")])
    assert not (tmp_path / "o.json").exists()
