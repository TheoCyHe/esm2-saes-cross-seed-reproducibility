"""The new arms' preflight must refuse a k mismatch (the silent one), a
dict_size mismatch, a missing key, a wrong count, and a permuted concept axis
(gotcha 13, gotcha 27)."""
import json

from src.analysis import ct_arm_preflight as P

CFG = {"sae": {"k": 64, "dict_size": 10240}}


def _ck(tmp_path, seed, **meta):
    d = tmp_path / f"seed_{seed}"
    d.mkdir()
    m = {"seed": seed, "k": 64, "dict_size": 10240}
    m.update(meta)
    m = {k: v for k, v in m.items() if v is not None}
    (d / "meta.json").write_text(json.dumps(m), encoding="utf-8")
    return d


def test_agreeing_checkpoints_pass(tmp_path):
    cks = [_ck(tmp_path, s) for s in range(3)]
    assert P.check_ckpts(cks, CFG, 3) == []


def test_k_mismatch_is_refused(tmp_path):
    cks = [_ck(tmp_path, 0), _ck(tmp_path, 1, k=32)]
    bad = P.check_ckpts(cks, CFG, 2)
    assert len(bad) == 1 and "k=32" in bad[0]


def test_dict_size_mismatch_and_absent_key_and_count(tmp_path):
    cks = [_ck(tmp_path, 0, dict_size=40960), _ck(tmp_path, 1, k=None)]
    bad = P.check_ckpts(cks, CFG, 3)
    assert any("dict_size=40960" in b for b in bad)
    assert any("ABSENT" in b for b in bad)
    assert any("expected 3" in b for b in bad)


def test_duplicate_seed_is_refused(tmp_path):
    a = _ck(tmp_path, 0)
    b = tmp_path / "other"
    b.mkdir()
    (b / "meta.json").write_text(json.dumps({"seed": 0, "k": 64, "dict_size": 10240}),
                                 encoding="utf-8")
    assert any("duplicate" in x for x in P.check_ckpts([a, b], CFG, 2))


def test_axis_checks():
    ref = (["A", "B", "C", "P"], [True, True, True, False])
    assert P.check_axis((["A", "B", "C", "P"], [True, True, True, False]), ref, "ap") == []
    assert P.check_axis((["B", "A", "C", "P"], [True, True, True, False]), ref, "ap")
    assert P.check_axis((["A", "B", "C", "P"], [True, True, False, False]), ref, "ap")
    assert P.check_axis((["A", "B", "C"], None), ref, "null") == []
    assert P.check_axis((["A", "C", "B"], None), ref, "null")


def test_compare_ap_bar_axis_and_absent_seed():
    import numpy as np
    ref = {"seeds": ["seed_0", "seed_1"], "concepts": ["A", "B"],
           "ap": np.array([[[0.1, 0.2]], [[0.3, 0.4]]])}
    new = {"seeds": ["seed_0"], "concepts": ["A", "B"], "ap": np.array([[[0.1 + 5e-7, 0.2]]])}
    r = P.compare_ap(new, ref, "seed_0", 1e-6)
    assert r["verdict"] == "PASS" and r["n_bitwise_equal"] == 1
    new2 = {"seeds": ["seed_0"], "concepts": ["A", "B"], "ap": np.array([[[0.1 + 2e-6, 0.2]]])}
    assert P.compare_ap(new2, ref, "seed_0", 1e-6)["verdict"] == "FAIL"
    new3 = {"seeds": ["seed_0"], "concepts": ["B", "A"], "ap": new["ap"]}
    assert P.compare_ap(new3, ref, "seed_0", 1e-6)["reason"] == "concept axes differ"
    assert "ABSENT" in P.compare_ap(new, ref, "seed_1", 1e-6)["reason"]


def test_compare_gtilde_three_outcomes():
    import numpy as np
    ref = {"seeds": ["seed_0"], "concepts": ["A", "B"],
           "M": np.array([[1.0, 2.0]]), "argmax": np.array([[0, 1]])}
    same = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in ref.items()}
    assert P.compare_gtilde(same, ref)["verdict"] == "BITWISE"
    near = dict(same, M=np.array([[1.0 + 5e-7, 2.0]]))
    r = P.compare_gtilde(near, ref)
    assert r["verdict"] == "WITHIN_BAR" and r["n_values_differing"] == 1
    far = dict(same, M=np.array([[1.0 + 1e-5, 2.0]]))
    assert P.compare_gtilde(far, ref)["verdict"] == "STOP"
    flip = dict(near, argmax=np.array([[1, 1]]))
    assert P.compare_gtilde(flip, ref)["verdict"] == "STOP"
    assert P.compare_gtilde(dict(same, seeds=["seed_1"]), ref)["reason"] == "seeds differ"


def test_compare_gtilde_reports_an_absent_seed_axis():
    import numpy as np
    ref = {"seeds": None, "concepts": ["A"], "M": np.array([[1.0]]), "argmax": np.array([[0]])}
    new = {"seeds": ["seed_0"], "concepts": ["A"], "M": np.array([[1.0]]), "argmax": np.array([[0]])}
    r = P.compare_gtilde(new, ref)
    assert r["verdict"] == "BITWISE" and r["seed_axis"] == "ABSENT on reference"
