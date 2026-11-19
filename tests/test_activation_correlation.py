"""Amendment CT4 / CT11.6: the activation-correlation arm's arithmetic on
hand-built toys, each checked against an INDEPENDENT route (np.corrcoef;
rf_marginal.pair_statistics; a brute-force scan for theta*)."""
import numpy as np
import pytest
import torch

from src.analysis import activation_correlation as ac
from src.analysis.rf_marginal import pair_statistics


def _codes(n=400, d=12, k=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.stack([rng.choice(d - 1, size=k, replace=False) for _ in range(n)])
    vals = rng.gamma(2.0, 1.0, size=(n, k)).astype(np.float32)
    return vals, idx.astype(np.int16)        # feature d-1 never fires


def test_dense_from_codes_round_trip():
    vals, idx = _codes()
    z = ac.dense_from_codes(vals, idx, 12).numpy()
    assert z.shape == (400, 12)
    for r in range(5):
        np.testing.assert_allclose(z[r, idx[r].astype(int)], vals[r])
        assert (z[r] != 0).sum() == 3


def test_pearson_from_moments_equals_corrcoef_and_flags_zero_variance():
    va, ia = _codes(seed=1)
    vb, ib = _codes(seed=2)
    A = ac.dense_from_codes(va, ia, 12).double().numpy()
    B = ac.dense_from_codes(vb, ib, 12).double().numpy()
    n = A.shape[0]
    G = A.T @ B
    R = ac.pearson_from_moments(G, A.sum(0), (A ** 2).sum(0),
                                B.sum(0), (B ** 2).sum(0), n)
    want = np.corrcoef(A.T, B.T)[:12, 12:]
    np.testing.assert_allclose(R[:11, :11], want[:11, :11], atol=1e-12)
    assert np.all(R[11, :] == -np.inf) and np.all(R[:, 11] == -np.inf)


def test_mnn_from_matrix_matches_rf_marginal_on_a_cosine_matrix():
    g = torch.Generator().manual_seed(3)
    Wa = torch.randn(6, 20, generator=g); Wa /= Wa.norm(dim=0, keepdim=True)
    Wb = torch.randn(6, 20, generator=g); Wb /= Wb.norm(dim=0, keepdim=True)
    st = pair_statistics(Wa, Wb)
    a, b = ac.mnn_from_matrix((Wa.T @ Wb).numpy())
    np.testing.assert_array_equal(a, st["a_mnn"].numpy())
    np.testing.assert_array_equal(b, st["b_mnn"].numpy())
    assert np.isfinite(a).any() and (~np.isfinite(a)).any()


def test_theta_star_is_the_largest_theta_meeting_the_target():
    rng = np.random.default_rng(4)
    sim = rng.uniform(-0.2, 1.0, size=(3, 5, 40)).astype(np.float32)
    sim[np.arange(3), np.arange(3)] = -np.inf
    sim[rng.random(sim.shape) < 0.2] = -np.inf
    target = 0.4
    r = ac.theta_star(sim, target)
    assert r["status"] == "OK"
    t = r["theta_star"]
    assert ac.mean_rate(sim, t) >= target
    finite = np.unique(sim[np.isfinite(sim)])
    above = finite[finite > t]
    assert above.size == 0 or ac.mean_rate(sim, float(above[0])) < target
    # brute force over every observed similarity
    ok = [v for v in finite if ac.mean_rate(sim, float(v)) >= target]
    assert t == pytest.approx(max(ok))


def test_theta_star_reports_an_unreachable_target():
    sim = np.full((2, 3, 4), -np.inf, dtype=np.float32)
    sim[0, 1, 0] = 0.5
    assert ac.theta_star(sim, 0.9)["status"] == "TARGET_UNREACHABLE"


def test_symmetry_report_sees_a_one_sided_pair():
    sim = np.full((2, 3, 4), -np.inf, dtype=np.float32)
    sim[0, 1, :2] = [0.8, 0.6]
    sim[1, 0, :2] = [0.6, 0.8]
    assert ac.symmetry_report(sim, 2)["n_pairs_with_different_mutual_counts"] == 0
    sim[1, 0, 2] = 0.5
    assert ac.symmetry_report(sim, 2)["n_pairs_with_different_mutual_counts"] == 1


def test_intervals_overlap_both_ways():
    assert ac.intervals_overlap([0.1, 0.3], [0.25, 0.5])
    assert ac.intervals_overlap([0.1, 0.3], [0.3, 0.5])
    assert not ac.intervals_overlap([0.1, 0.3], [0.31, 0.5])
    assert not ac.intervals_overlap([0.4, 0.5], [0.1, 0.3])


def test_parser_has_four_steps():
    p = ac.build_parser()
    for cmd in ("encode", "gram", "reduce", "fit", "jackknife", "pairs", "curve"):
        assert cmd in p._subparsers._group_actions[0].choices


def test_pearson_of_pairs_equals_corrcoef_on_named_pairs():
    va, ia = _codes(seed=5)
    vb, ib = _codes(seed=6)
    A = ac.dense_from_codes(va, ia, 12).double().numpy()
    B = ac.dense_from_codes(vb, ib, 12).double().numpy()
    g = np.array([3, 0, 7, 7, 1, 2, 9, 4, 5, 6, 8, 10])
    r = ac.pearson_of_pairs((A * B[:, g]).sum(0), A.sum(0), (A ** 2).sum(0),
                            B.sum(0)[g], (B ** 2).sum(0)[g], A.shape[0])
    for f in range(11):
        np.testing.assert_allclose(r[f], np.corrcoef(A[:, f], B[:, g[f]])[0, 1],
                                   atol=1e-12)
    assert r[11] == -np.inf              # feature 11 never fires in A


def test_pair_identity_summary_counts_same_partner_and_cells():
    act = np.array([0.80, 0.75, 0.90, -np.inf, 0.71], dtype=np.float32)
    r_dec = np.array([0.80, 0.60, 0.90, 0.95, 0.7100005])
    dec = np.array([0.72, 0.85, 0.65, 0.90, -np.inf], dtype=np.float32)
    s = ac.pair_identity_summary(act, r_dec, dec, 0.70)
    assert s["n_act_matched"] == 4                     # cells 0, 1, 2, 4
    assert s["n_both_matched_cell"] == 2               # 0 and 1 clear both
    assert s["n_same_partner_tol_1e-06"] == 2          # 0 and 2; 4 not mutual in dec
    assert s["n_same_partner_and_dec_clears_070_tol_1e-06"] == 1   # 0 only


# ---- §CT4-T: log M by cell class ---------------------------------------------

def test_cell_classes_partition_the_off_diagonal_cells():
    """2 anchors x 3 seeds x 2 features; an anchor's own column is in no class."""
    act = np.array([[[0.9, 0.1], [0.9, 0.9], [0.1, 0.1]],
                    [[0.9, 0.1], [0.9, 0.9], [0.8, 0.1]]])
    dec = np.array([[[0.9, 0.9], [0.9, 0.1], [0.9, 0.1]],
                    [[0.1, 0.9], [0.9, 0.9], [0.1, 0.9]]])
    c = ac.cell_classes(act, dec, 0.70, 0.70)
    # anchor 0: own column 0 excluded. seed 1: f0 both, f1 act-only; seed 2: f0 dec-only, f1 neither
    assert c["both"][0].tolist() == [[False, False], [True, False], [False, False]]
    assert c["activation_only"][0].tolist() == [[False, False], [False, True], [False, False]]
    assert c["decoder_only"][0].tolist() == [[False, False], [False, False], [True, False]]
    assert c["neither"][0].tolist() == [[False, False], [False, False], [False, True]]
    # anchor 1: own column 1 excluded. seed 0: f0 act-only, f1 dec-only; seed 2: f0 act-only, f1 dec-only
    assert c["both"][1].sum() == 0 and c["activation_only"][1].sum() == 2
    total = sum(m.sum() for m in c.values())
    assert total == 2 * 2 * 2                      # anchors x off-diagonal seeds x features


def test_class_distribution_counts_cells_on_the_fit_set_only():
    logM = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    keep = np.array([[True, True, False], [True, False, True]])
    mask = np.zeros((2, 2, 3), dtype=bool)
    mask[0, 1, :] = True                           # anchor 0: features 0,1,2 in one comparator
    mask[1, 0, 2] = mask[1, 1, 2] = True           # anchor 1: feature 2 in two comparators
    d = ac.class_distribution(logM, mask, keep, ref_mean=0.0, ref_sd=1.0)
    assert d["n_cells"] == 4                       # 1.0, 2.0 (3.0 floored out), 6.0 twice
    assert d["n_distinct_anchor_features"] == 3
    assert d["mean"] == pytest.approx((1 + 2 + 6 + 6) / 4)
    assert d["quantiles"]["0.5"] == pytest.approx(4.0)


def test_parser_has_the_classes_step():
    p = ac.build_parser()
    a = p.parse_args(["classes", "--act-counts", "a", "--act-json", "b", "--sim", "c",
                      "--rf-counts-all-theta", "d", "--m1-json", "e", "--gtilde", "f",
                      "--null-reductions", "g", "--d-fs", "h", "--out", "i",
                      "--pairs", "p0", "p1"])
    assert a.cmd == "classes" and a.theta_dec == 0.70 and len(a.pairs) == 2


# ---- §CT10: the Gram on a row subset ------------------------------------------

def _gram_fixture(tmp_path, n=120, d=12, k=3, seeds=(0, 1, 2)):
    """A toy label store (population = cache rows 5..124), a halves file, and
    one code file per seed holding top-k codes for every population row."""
    import argparse
    import h5py
    rng = np.random.default_rng(4)
    lab = tmp_path / "labels_toy.h5"
    with h5py.File(lab, "w") as hf:
        g = hf.create_group("protein_sets")
        g.create_dataset("population_start", data=np.array([5, 65]))
        g.create_dataset("population_n_rows", data=np.array([60, 60]))
        hf.attrs["n_rows"] = n
    pop = np.arange(5, 125)
    half_a = pop[rng.random(n) < 0.5]
    halves = tmp_path / "halves.h5"
    with h5py.File(halves, "w") as hf:
        hf.create_dataset("ct10_half_a_rows", data=half_a)
        hf.create_dataset("ct10_half_b_rows", data=np.setdiff1d(pop, half_a))
        hf.attrs["labels"] = str(lab)
    dense, paths = {}, []
    for s in seeds:
        Z = np.zeros((n, d), dtype=np.float32)
        for r in range(n):
            idx = rng.choice(d, size=k, replace=False)
            Z[r, idx] = rng.uniform(0.1, 2.0, size=k).astype(np.float32)
        vals = np.zeros((n, k), dtype=np.float32)
        ind = np.zeros((n, k), dtype=np.int16)
        for r in range(n):
            nz = np.flatnonzero(Z[r])
            ind[r], vals[r] = nz, Z[r, nz]
        p = tmp_path / f"codes_seed_{s}.h5"
        with h5py.File(p, "w") as hf:
            hf.create_dataset("values", data=vals)
            hf.create_dataset("indices", data=ind)
            hf.create_dataset("sum_a", data=Z.astype(np.float64).sum(0))
            hf.create_dataset("sum_a2", data=(Z.astype(np.float64) ** 2).sum(0))
            hf.attrs.update({"seed": s, "n_rows": n, "dict_size": d,
                             "gather_device_class": "cpu", "git_sha": "x"})
        dense[s] = Z
        paths.append(p)
    mk = lambda **kw: argparse.Namespace(codes=paths, anchor=0, group=20, chunk_rows=17,
                                         device="cpu", labels=lab, row_subset=None,
                                         row_subset_file=None, **kw)
    return mk, dense, pop, half_a, halves


def _run_gram(args):
    import h5py
    from src.analysis.activation_correlation import cmd_gram
    cmd_gram(args)
    with h5py.File(args.out, "r") as hf:
        return hf["mnn_sim_row"][:], dict(hf.attrs)


def test_gram_on_a_row_subset_equals_pearson_mnn_on_those_rows(tmp_path):
    from src.analysis.activation_correlation import mnn_from_matrix
    mk, dense, pop, half_a, halves = _gram_fixture(tmp_path)
    sim, at = _run_gram(_with(mk(out=tmp_path / "rowA.h5"),
                              row_subset="ct10_half_a", row_subset_file=halves))
    pos = np.searchsorted(pop, half_a)
    for b in (1, 2):
        R = np.corrcoef(dense[0][pos].T, dense[b][pos].T)[:12, 12:]
        R = np.where(np.isfinite(R), R, -np.inf)
        want, _ = mnn_from_matrix(R)
        np.testing.assert_allclose(sim[b], want, rtol=0, atol=1e-6)
    assert int(at["n_rows"]) == half_a.size and at["row_subset"] == "ct10_half_a"
    assert np.all(np.isneginf(sim[0]))                 # the anchor's own column


def test_gram_without_a_subset_is_the_ct4_path(tmp_path):
    from src.analysis.activation_correlation import mnn_from_matrix
    mk, dense, pop, half_a, halves = _gram_fixture(tmp_path)
    sim, at = _run_gram(mk(out=tmp_path / "rowall.h5"))
    for b in (1, 2):
        R = np.corrcoef(dense[0].T, dense[b].T)[:12, 12:]
        R = np.where(np.isfinite(R), R, -np.inf)
        want, _ = mnn_from_matrix(R)
        np.testing.assert_allclose(sim[b], want, rtol=0, atol=1e-6)
    assert int(at["n_rows"]) == 120 and at["row_subset"] == "all_code_rows"


def test_gram_subset_needs_its_file(tmp_path):
    mk, *_ = _gram_fixture(tmp_path)
    with pytest.raises(SystemExit, match="needs --labels and --row-subset-file"):
        _run_gram(_with(mk(out=tmp_path / "x.h5"), row_subset="ct10_half_a"))


def _with(ns, **kw):
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_encode_k_comes_from_the_checkpoint_gotcha_48():
    import pytest
    from src.analysis.activation_correlation import resolve_k
    assert resolve_k(None, {"k": 64}) == 64
    assert resolve_k(64, {"k": 64}) == 64
    with pytest.raises(SystemExit):
        resolve_k(32, {"k": 64})        # the silent hazard: a k = 32 default on k = 64
    with pytest.raises(SystemExit):
        resolve_k(None, {})             # absent is refused, never assumed


def test_reduce_takes_a_target_rate():
    from src.analysis.activation_correlation import build_parser, theta_star
    import numpy as np
    a = build_parser().parse_args(["reduce", "--rows", "r", "--out", "o", "--out-counts", "c",
                                   "--target-rate", "0.2169886"])
    assert a.target_rate == 0.2169886
    sim = np.full((2, 2, 4), -np.inf)
    sim[0, 1] = [0.9, 0.8, 0.5, 0.1]
    sim[1, 0] = [0.9, 0.7, 0.4, 0.2]
    # target 0.5: two of four cells per anchor -> theta* is the largest value with rate >= 0.5
    assert theta_star(sim, 0.5)["theta_star"] == 0.7
