"""Amendment CT3 / CT11.1: the jackknife's matching, leave-one-out arithmetic,
gate and estimator on a 4-dictionary toy. Every expectation is computed by an
INDEPENDENT route (re-matching with the dictionary physically removed;
hand-computed jackknife), never by the function under test (gotcha 32b)."""
import numpy as np
import pytest
import torch

from src.analysis import jackknife_dictionaries as jk
from src.analysis.rf_marginal import counts_at_theta, pair_statistics

THETAS = (0.3, 0.5, 0.7, 0.9)


def _toy(n_seeds=4, d_model=8, dict_size=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(d_model, dict_size, generator=g)
    decs = []
    for s in range(n_seeds):
        W = base + 0.6 * torch.randn(d_model, dict_size, generator=g)
        W = W[:, torch.randperm(dict_size, generator=g)]
        decs.append(W / W.norm(dim=0, keepdim=True))
    return decs


def _rf_marginal_counts(decs, n_anchors, theta):
    """rf_marginal.main's own construction, reproduced for the toy."""
    n = len(decs)
    sim = np.full((n, n, decs[0].shape[1]), -np.inf, dtype=np.float32)
    for a in range(n):
        for b in range(a + 1, n):
            if a >= n_anchors and b >= n_anchors:
                continue
            st = pair_statistics(decs[a], decs[b])
            sim[a, b] = st["a_mnn"].numpy()
            sim[b, a] = st["b_mnn"].numpy()
    off = ~np.eye(n, dtype=bool)
    off[n_anchors:] = False
    return counts_at_theta(sim, off, n_anchors, theta)


def test_anchor_rows_reproduce_rf_marginal_counts():
    decs = _toy()
    sim = jk.mnn_sim_anchor_rows(decs, 2)
    assert sim.shape == (2, 4, 16)
    for t in THETAS:
        np.testing.assert_array_equal(jk.counts_from_sim(sim, t),
                                      _rf_marginal_counts(decs, 2, t))
    assert (jk.counts_from_sim(sim, 0.3) > 0).any()   # the toy is not degenerate


@pytest.mark.parametrize("d", [0, 1, 2, 3])
def test_loo_equals_rematching_without_the_dictionary(d):
    decs = _toy()
    n_anchors = 2
    sim = jk.mnn_sim_anchor_rows(decs, n_anchors)
    rest = [w for i, w in enumerate(decs) if i != d]
    rest_anchors = n_anchors - (1 if d < n_anchors else 0)
    for t in THETAS:
        y, keep = jk.loo_counts(sim, t, d)
        want = _rf_marginal_counts(rest, rest_anchors, t)
        np.testing.assert_array_equal(y, want)
        assert keep.sum() == rest_anchors
        assert y.max() <= len(decs) - 2


def test_gate_exact_near_divergent():
    decs = _toy()
    sim = jk.mnn_sim_anchor_rows(decs, 2)
    stored = {t: jk.counts_from_sim(sim, t) for t in THETAS}
    assert jk.gate(sim, stored, "rtx", "rtx")["verdict"] == "EXACT"

    # one cell off where every similarity is far from theta -> DIVERGENT
    bad = {t: v.copy() for t, v in stored.items()}
    a, f = 0, int(np.argmax(np.abs(sim[0, 1] - 0.7) > 0.05))
    bad[0.7][a, f] += 1
    assert jk.gate(sim, bad, "cpu", "rtx")["verdict"] == "DIVERGENT"

    # a flip at a similarity within 1e-6 of theta, on a different class -> NEAR
    sim2 = sim.copy()
    f2 = int(np.flatnonzero(np.isfinite(sim2[0, 1]))[0])
    sim2[0, 1, f2] = np.float32(0.7)          # float32 0.7 sits a hair below theta
    stored2 = {t: jk.counts_from_sim(sim2, t) for t in THETAS}
    stored2[0.7] = stored2[0.7].copy()
    stored2[0.7][0, f2] -= 1                  # the record had it just below
    g = jk.gate(sim2, stored2, "cpu", "rtx_5000_ada")
    assert g["verdict"] == "NEAR" and g["n_differing_cells"] == 1
    # the same flip on the SAME class is not NEAR (§CT11.1's device clause)
    assert jk.gate(sim2, stored2, "rtx", "rtx")["verdict"] == "DIVERGENT"


def test_jackknife_hand_computed():
    r = jk.jackknife([1.0, 2.0, 3.0, 4.0], full=2.0)
    assert r["se"] == pytest.approx(np.sqrt(3 / 4 * 5.0))
    assert r["mean_of_loo"] == pytest.approx(2.5)
    assert r["bias_estimate"] == pytest.approx(3 * 0.5)


def test_control_verdict_branches():
    assert jk.control_verdict([1.0, 2.0], [1.0, 2.0])["verdict"] == "BIT_IDENTICAL"
    assert jk.control_verdict([1.0, 2.0 + 1e-9], [1.0, 2.0])["verdict"] == "WITHIN_1E-6"
    assert jk.control_verdict([1.0, 2.1], [1.0, 2.0])["verdict"] == "STOP"


def test_widest_interval():
    assert jk.widest_interval({"a": [0, 1], "b": [-1, 1.5], "c": [0, 2]}) == "b"


def test_parser_has_both_steps():
    p = jk.build_parser()
    a = p.parse_args(["fit", "--sim", "s", "--rf-counts-all-theta", "r",
                      "--m1-json", "m", "--gtilde", "g", "--null-reductions",
                      "n", "--d-fs", "d", "--reference-join", "j", "--out", "o"])
    assert a.cmd == "fit" and a.workers == 8


def test_same_device_class_across_the_two_vocabularies():
    run = "cuda:NVIDIA RTX 5000 Ada Generation:sm_89"
    assert jk.same_device_class("rtx_5000_ada", run)
    assert not jk.same_device_class("cpu", run)
    assert not jk.same_device_class("rtx_pro_6000_blackwell", run)
    assert not jk.same_device_class("", run)


def test_gate_treats_the_record_label_and_the_run_string_as_one_class():
    decs = _toy()
    sim = jk.mnn_sim_anchor_rows(decs, 2)
    f2 = int(np.flatnonzero(np.isfinite(sim[0, 1]))[0])
    sim[0, 1, f2] = np.float32(0.7)
    stored = {t: jk.counts_from_sim(sim, t) for t in THETAS}
    stored[0.7] = stored[0.7].copy()
    stored[0.7][0, f2] -= 1
    g = jk.gate(sim, stored, "rtx_5000_ada",
                "cuda:NVIDIA RTX 5000 Ada Generation:sm_89")
    assert not g["device_classes_differ"] and g["verdict"] == "DIVERGENT"


def test_parser_accepts_a_new_arm_amendment_ct_ct24_ct25():
    """§CT24/§CT25: an arm whose every seed is an anchor has no comparison
    seeds and no join of record; the 40,960 arm needs CS3c's git_sha flag."""
    p = jk.build_parser()
    a = p.parse_args(["match", "--ckpts", "a", "b", "--expect-total-ckpts", "2",
                      "--rf-counts-all-theta", "x", "--record-device-class", "r",
                      "--record-device-source", "s", "--out", "o",
                      "--accept-git-shas", "g1", "g2"])
    assert a.compare_ckpts == [] and a.accept_git_shas == ["g1", "g2"]
    f = p.parse_args(["fit", "--sim", "s", "--rf-counts-all-theta", "r", "--m1-json", "m",
                      "--gtilde", "g", "--null-reductions", "n", "--d-fs", "d",
                      "--out", "o", "--thetas", "0.70"])
    assert f.reference_join is None and f.thetas == [0.70]
    # the record's invocation still parses unchanged
    f2 = p.parse_args(["fit", "--sim", "s", "--rf-counts-all-theta", "r", "--m1-json", "m",
                       "--gtilde", "g", "--null-reductions", "n", "--d-fs", "d",
                       "--reference-join", "j", "--out", "o"])
    assert f2.thetas is None and str(f2.reference_join) == "j"
