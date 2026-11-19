"""§CT28: the shared-scale arithmetic, the verdicts and the scoring rules.
Expectations are written down (gotcha 32b); each rule's outcome space is
exercised in full (gotchas 24, 25)."""
import numpy as np
import pytest

from src.analysis import ct28_rescore as cr


def test_declared_constants_written_down():
    assert cr.SD_REF == 2.067756996018893
    assert cr.K64_TARGET_N19 == 0.495814
    assert cr.WIDTH_TARGET_N9 == 0.4957942708333333
    assert cr.K64_RULE_TARGET == 0.21698858963815787
    assert cr.K32_N19_FE_CI == (0.1162, 0.3950)
    assert cr.P2_BAND == (-0.5, 1.0)


def test_scale_factors_per_seed():
    f = cr.scale_factors(np.array([2.0, 4.0]))
    assert f["own"].tolist() == [1.0, 1.0]
    assert f["per_unit"].tolist() == [0.5, 0.25]
    assert np.allclose(f["shared"], [cr.SD_REF / 2.0, cr.SD_REF / 4.0])


def test_pooled_value_is_a_pool_of_rescaled_values_not_a_rescaled_pool():
    """With unequal per-seed SDs the shared-scale pool is NOT the own-scale
    pool times one constant -- the reason (a) refits with per-seed values."""
    ame = np.array([0.010, 0.030])
    se = np.array([0.001, 0.004])
    sd = np.array([1.5, 3.0])
    p = cr.pooled_on_scales(ame, se, sd)
    naive = p["own"]["fe_pp"] * cr.SD_REF / sd.mean()
    assert not np.isclose(p["shared"]["fe_pp"], naive, rtol=1e-6)
    # equal SDs: then it IS one constant
    q = cr.pooled_on_scales(ame, se, np.array([2.0, 2.0]))
    assert np.isclose(q["shared"]["fe_pp"], q["own"]["fe_pp"] * cr.SD_REF / 2.0, rtol=1e-12)
    assert np.isclose(q["per_unit"]["fe_pp"], q["own"]["fe_pp"] / 2.0, rtol=1e-12)


def test_scale_verdict_is_total():
    assert cr.scale_verdict(0.0, 1e-3) == "BIT_IDENTICAL"
    assert cr.scale_verdict(1e-7, 0.0) == "WITHIN_1E-6"
    assert cr.scale_verdict(1e-6, 0.0) == "WITHIN_1E-6"
    assert cr.scale_verdict(1e-4, 1e-3) == "WITHIN_CONTROL"
    assert cr.scale_verdict(1e-2, 1e-3) == "DIFFERS"


def test_rel_diff():
    assert cr.rel_diff([1.0, 2.0], [1.0, 2.0]) == 0.0
    assert np.isclose(cr.rel_diff([1.1], [1.0]), 0.1)


def test_score_p1():
    assert cr.score_p1([0.3, 0.5], True) == "HIT"          # overlaps [0.1162, 0.3950]
    assert cr.score_p1([0.3950, 0.6], True) == "HIT"       # closed: touching counts
    assert cr.score_p1([0.40, 0.6], True) == "MISS"
    assert cr.score_p1([0.0, 0.1], True) == "MISS"
    assert cr.score_p1([0.3, 0.5], False) == "UNMEASURABLE"


def test_score_p2_total():
    inside = {2560: 0.2, 10240: 0.22, 20480: -0.5, 40960: 1.0}
    assert cr.score_p2(inside) == "HIT"
    assert cr.score_p2({**inside, 40960: 1.01}) == "MISS"
    assert cr.score_p2({**inside, 20480: None}) == "UNMEASURABLE"
    assert cr.score_p2({**inside, 20480: None, 40960: -2.0}) == "MISS"


def test_score_p3_total():
    pos = {a: [0.0001, 0.001] for a in ("k64", "w2560", "w10240", "w20480", "w40960")}
    assert cr.score_p3(pos) == "HIT"
    assert cr.score_p3({**pos, "w2560": [0.0, 0.001]}) == "MISS"     # contains 0
    assert cr.score_p3({**pos, "w2560": [-0.002, -0.001]}) == "MISS"
    assert cr.score_p3({**pos, "w2560": None}) == "UNMEASURABLE"


def test_score_p5_total():
    r = {"rule1": 1.0, "rule2": 0.5, "rule3": 2.0, "rule4": 1.5}
    assert cr.score_p5(r, 1.4)["verdict"] == "HIT"       # spread 1.5 >= 1.4
    assert cr.score_p5(r, 1.5)["verdict"] == "HIT"
    assert cr.score_p5(r, 1.6)["verdict"] == "MISS"
    assert cr.score_p5({**r, "rule3": None}, 1.0)["verdict"] == "UNMEASURABLE"
    assert cr.score_p5(r, None)["verdict"] == "UNMEASURABLE"


def _synthetic(n_seeds=3, n=400, seed=0):
    rng = np.random.default_rng(seed)
    sds = np.array([1.5, 2.0, 3.0])[:n_seeds]
    logM = np.stack([rng.normal(0.0, s, n) for s in sds])
    log_phi = rng.uniform(-4, -1, (n_seeds, n))
    d_fs = rng.uniform(0, 1, (n_seeds, n))
    keep = np.ones((n_seeds, n), dtype=bool)
    eta = 0.2 + 0.3 * (logM / sds[:, None])
    p = 1 / (1 + np.exp(-eta))
    y = rng.binomial(9, p).astype(np.int16)
    return y, logM, log_phi, d_fs, keep


@pytest.fixture(scope="module")
def fitted():
    y, logM, log_phi, d_fs, keep = _synthetic()
    cr._G.clear()
    cr._G.update({"logM": logM, "log_phi": log_phi, "d_fs": d_fs, "keep": keep})
    rows = np.arange(3)
    own = cr._fit_scales(y, rows, 9)
    same = cr._fit_scales(y, rows, 9,
                          z_divisor=np.array([r["sd_logM"] for r in own["per_seed"]]))
    shared = cr._fit_scales(y, rows, 9, z_divisor=cr.SD_REF)
    return own, same, shared


def test_the_default_divisor_is_the_seeds_own_sd(fitted):
    """Passing each seed's own SD reproduces the default path bit for bit, so
    the z_divisor branch adds nothing on the record's path."""
    own, same, _ = fitted
    assert [r["ame"] for r in same["per_seed"]] == [r["ame"] for r in own["per_seed"]]
    assert same["own"]["fe_pp"] == own["own"]["fe_pp"]


def test_a_rescaled_covariate_rescales_each_seeds_effect(fitted):
    """The model is equivariant, the procedure is not exactly (§CT28-pre item 5).
    The binomial stage rescales to rounding; the beta-binomial refit (L-BFGS-B
    from the binomial start) lands near, not on, the analytic value -- on this
    fixture within 1%, measured at 0.36% when §CT28 was declared. That gap is
    why the shared-scale figures are the analytic rescaling and the refit is
    only the verification, judged against the x2 control."""
    from src.analysis.design3_join import fit_arm
    y, logM, log_phi, d_fs, keep = _synthetic()
    a = fit_arm(y.astype(float), logM, log_phi, d_fs, keep, 9, dispersion=True)
    b = fit_arm(y.astype(float), logM, log_phi, d_fs, keep, 9, dispersion=True,
                z_divisor=cr.SD_REF)
    for ra, rb in zip(a["per_seed"], b["per_seed"]):
        f = cr.SD_REF / ra["sd_logM"]
        assert np.isclose(rb["b1_z"], ra["b1_z"] * f, rtol=1e-10)     # binomial: equivariant
    own, _, shared = fitted
    want = np.array([r["ame"] * cr.SD_REF / r["sd_logM"] for r in own["per_seed"]])
    got = np.array([r["ame"] for r in shared["per_seed"]])
    assert 0.0 < cr.rel_diff(got, want) < 1e-2
    assert not np.allclose(got, [r["ame"] for r in own["per_seed"]], rtol=1e-2)


def _arm(own_ci=(1.0, 2.0), own=1.5, shared_ci=(1.0, 2.0), shared=1.5, ctl="BIT_IDENTICAL",
         failed=0, theta_ok=True):
    fit = {"own": {"fe_pp": own, "fe_ci95_pp": list(own_ci), "widest": "jackknife"},
           "shared": {"fe_pp": shared, "fe_ci95_pp": list(shared_ci)},
           "n_anchor_fits_failed": failed}
    a = {"control_0.70_vs_record": {"verdict": ctl}, "at_0.70": fit,
         "at_theta_star": fit if theta_ok else {"status": "UNMEASURABLE"}}
    return a


def test_score_all_derives_each_verdict_from_the_outputs():
    arms = {"k64": _arm(own_ci=(0.3, 0.5), shared_ci=(1.6, 1.9), shared=1.75),
            "k32n19": _arm(shared_ci=(0.1, 0.4), shared=0.25),
            "w2560": _arm(own=0.2), "w10240": _arm(own=0.22), "w20480": _arm(own=-0.4),
            "w40960": _arm(own=0.9)}
    arms["k64"]["scale_verification"] = {"verdict": "WITHIN_CONTROL"}
    arms["k64"]["rule4_activation"] = {"shared": {"fe_pp": 2.4}, "n_anchor_fits_failed": 0,
                                       "control_vs_ct24_r3": {"verdict": "BIT_IDENTICAL"}}
    sff = {a: {"slope": {"own": {"fixed_effect_ci95": [1e-4, 1e-3]}}}
           for a in ("k64", "w2560", "w10240", "w20480", "w40960")}
    rules = {"rules": {k: {"at_theta_star": {"shared": {"fe_pp": v}, "n_anchor_fits_failed": 0}}
                       for k, v in (("rule1", 1.75), ("rule2", 1.0), ("rule3", 3.0))}}
    s = cr.score_all(arms, sff, rules)
    assert s["CT28-P1"]["verdict"] == "HIT"
    assert s["CT28-P2"]["verdict"] == "HIT"
    assert s["CT28-P3"]["verdict"] == "HIT"
    assert s["CT28-P4"]["share_no_point"] is True
    assert s["CT28-P5"]["verdict"] == "HIT"          # spread 2.0 >= 1.5
    # a stopped control makes the arm unmeasurable for every prediction reading it
    arms["w40960"] = _arm(own=0.9, ctl="STOP")
    assert cr.score_all(arms, sff, rules)["CT28-P2"]["verdict"] == "UNMEASURABLE"
    # DIFFERS withholds P4 and makes P5 unmeasurable
    arms["k64"]["scale_verification"] = {"verdict": "DIFFERS"}
    s = cr.score_all(arms, sff, rules)
    assert s["CT28-P4"]["status"] == "WITHHELD"
    assert s["CT28-P5"]["verdict"] == "UNMEASURABLE"


def test_sf_on_comparison_set_full_and_subset_by_label():
    """§CT9-D item D4: s_f over a comparison subset, rows and columns by label.
    one[a, c, f] = 10*a + c (+ f/100); expectations written out by hand."""
    labels = ["0", "1", "2", "3"]
    one = np.zeros((4, 4, 2))
    for a in range(4):
        for c in range(4):
            one[a, c] = [10 * a + c, 10 * a + c + 0.01]
    full = cr.sf_on_comparison_set(one, labels, labels)
    # anchor 0 over comparators 1, 2, 3 -> mean(1, 2, 3) = 2; anchor 1 over 0, 2, 3
    # -> mean(10, 12, 13) = 35/3; anchor 2 -> mean(20, 21, 23); anchor 3 -> 31
    assert full[:, 0].tolist() == [2.0, 35.0 / 3.0, 64.0 / 3.0, 31.0]
    sub = cr.sf_on_comparison_set(one, labels, ["2", "0"])
    # rows in `use` order: anchor 2 over {0} -> 20; anchor 0 over {2} -> 2
    assert sub[:, 0].tolist() == [20.0, 2.0]
    assert sub.shape == (2, 2)
    # labels permuted: the same seeds by label give the same values (gotcha 29)
    perm = [3, 1, 0, 2]
    one_p = one[np.ix_(perm, perm)]
    sub_p = cr.sf_on_comparison_set(one_p, [labels[i] for i in perm], ["2", "0"])
    assert sub_p.tolist() == sub.tolist()
    with pytest.raises(SystemExit):
        cr.sf_on_comparison_set(one, labels, ["0", "7"])
    with pytest.raises(SystemExit):
        cr.sf_on_comparison_set(one, labels, ["0", "0"])
