"""§CT27: the analytic point (a) asserted, and the raw-encoder variant's
pieces, each able to fail (gotcha 13). Expectations written down (gotcha 32b)."""
import json

import h5py
import numpy as np
import torch

from src.analysis import ct27_raw_encoder as C
from src.analysis.ct_robustness import matched_set_overlap
from src.analysis.rule_interior_mass import load_checkpoints
from src.analysis.rule_sensitivity import cosine_matrix


def test_a_cosine_rule_is_unchanged_by_row_scaling():
    """§CT27-F item 1: a deduction, asserted -- cos(au, bv) = cos(u, v)."""
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(16, 5)), rng.normal(size=(16, 7))
    sa, sb = rng.uniform(0.96, 1.72, 5), rng.uniform(0.96, 1.72, 7)
    assert np.allclose(cosine_matrix(A * sa, B * sb), cosine_matrix(A, B))
    # and a raw product IS changed by it -- the variant can differ
    assert not np.allclose((A * sa).T @ (B * sb), A.T @ B)


def test_variant_keeps_only_agreeing_decoder_matches_at_070():
    dec_p = np.array([0, 1, 2, 3, -1])
    dec_s = np.array([0.9, 0.69, 0.8, 0.95, -np.inf])
    enc_p = np.array([0, 1, 3, 3, 4])
    enc_s = np.array([1.4, 1.2, 1.1, 0.3, 2.0])
    p, s = C.variant_from_assignments(dec_p, dec_s, enc_p, enc_s)
    assert p.tolist() == [0, -1, -1, 3, -1]
    assert s[0] == 1.4 and s[3] == 0.3
    assert np.isneginf(s[[1, 2, 4]]).all()


def _ckpts(tmp_path, n_seeds=3, d=6, n=4, seed=0):
    rng = np.random.default_rng(seed)
    for s in range(n_seeds):
        Wd = torch.tensor(rng.normal(size=(d, n)), dtype=torch.float32)
        Wd = Wd / Wd.norm(dim=0, keepdim=True)
        We = torch.tensor(rng.normal(size=(n, d)) * rng.uniform(0.9, 1.8, (n, 1)),
                          dtype=torch.float32)
        (tmp_path / f"seed_{s}").mkdir()
        torch.save({"W_dec": Wd, "W_enc": We}, tmp_path / f"seed_{s}" / "model.pt")


def test_load_checkpoints_flag_changes_only_the_encoder(tmp_path):
    _ckpts(tmp_path)
    D1, E1, _ = load_checkpoints(tmp_path, [0, 1, 2])
    D2, E2, _ = load_checkpoints(tmp_path, [0, 1, 2], normalise_encoder=False)
    for s in (0, 1, 2):
        assert torch.equal(D1[s], D2[s])
        assert torch.allclose(E1[s].norm(dim=1), torch.ones(4))
        assert not torch.allclose(E2[s].norm(dim=1), torch.ones(4))
        assert torch.allclose(E2[s] / E2[s].norm(dim=1, keepdim=True), E1[s])


def test_run_control_is_exact_against_itself_and_fails_on_a_mutation(tmp_path):
    _ckpts(tmp_path)
    res = C.run(tmp_path, [0, 1, 2])
    ref = tmp_path / "ref.h5"
    with h5py.File(ref, "w") as hf:
        hf.create_dataset(C.RULE3_DS, data=res["r3"])
        hf.attrs["anchor_seed_labels"] = json.dumps(["0", "1", "2"])
    assert C.control_vs_stored(res["r3"], [0, 1, 2], ref)["verdict"] == "EXACT"
    bad = res["r3"].copy()
    i = np.argwhere(np.isfinite(bad))
    if len(i):
        bad[tuple(i[0])] += 1e-12
    else:
        bad[0, 1, 0] = 0.5
    assert C.control_vs_stored(bad, [0, 1, 2], ref)["verdict"] == "DIFFERS"
    # the diagonal is never filled: an anchor never compares against itself
    for k in range(3):
        assert np.isneginf(res["var"][k, k]).all() and np.isneginf(res["r3"][k, k]).all()


def test_matched_set_overlap_counts_and_excludes_own_column():
    a = np.full((2, 2, 3), -np.inf)
    b = np.full((2, 2, 3), -np.inf)
    a[0, 1] = [1.0, 0.2, 1.0]
    b[0, 1] = [0.8, 0.8, 0.1]
    a[0, 0] = [9, 9, 9]                      # own column: must not count
    r = matched_set_overlap(a, 0.5, b, 0.7, ["0", "1"])
    assert r == {"n_matched_a": 2, "n_matched_b": 2, "n_both": 1, "n_either": 3,
                 "jaccard": 1 / 3}
