"""
Tests for the narrow, chunked row-index path (amendment CT, §CT29-D2 item 1,
route b1): `narrow_row_index`, `gather_touches_chunked`, `feature_chunk_bounds`
and `average_precision_from_touches_chunked` in src/analysis/sparse_ap_index.py,
and `row_index_cache.load_row_index(narrow_features=...)`.

The bar is BITWISE on CPU: each feature's AP is computed from the same touches
in the same order, and CPU index_add_ sums in index order, so any difference at
all is a defect. (On CUDA, index_add_ is order-nondeterministic already --
gotcha 35 -- and the declared control measures that.) Expected chunk bounds are
written down, not derived from the code under test (gotcha 32b).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.analysis.sparse_ap_index import (
    average_precision_from_touches,
    average_precision_from_touches_chunked,
    build_row_index,
    feature_chunk_bounds,
    gather_touches,
    gather_touches_chunked,
    narrow_row_index,
)


def _layout(n_rows: int, n_features: int, k: int, seed: int, short_every: int = 7):
    """A random TopK-like layout (rows fire <= k features; every `short_every`th
    row fires fewer, and one row fires none), feature-major as
    sparse_activations returns it."""
    rng = np.random.default_rng(seed)
    Z = np.zeros((n_rows, n_features))
    for i in range(n_rows):
        kk = 0 if i == 3 else (k - 2 if i % short_every == 0 else k)
        cols = rng.choice(n_features, kk, replace=False)
        Z[i, cols] = rng.uniform(0.1, 5.0, size=kk)
    feats, poss = [], []
    for f in range(n_features):
        nz = np.flatnonzero(Z[:, f])
        order = nz[np.argsort(-Z[nz, f], kind="stable")]
        feats.append(np.full(len(order), f, dtype=np.int64))
        poss.append(order.astype(np.int64))
    feat = torch.from_numpy(np.concatenate(feats))
    pos = torch.from_numpy(np.concatenate(poss))
    counts = np.bincount(feat.numpy(), minlength=n_features)
    seg_start = torch.from_numpy(np.concatenate([[0], np.cumsum(counts)[:-1]])).to(torch.int64)
    return build_row_index(feat, pos, seg_start, n_rows)


# ---- narrow_row_index ------------------------------------------------------

def test_narrowing_keeps_every_value_and_picks_int16_up_to_32767_features():
    fbr, rbr, _ = _layout(300, 40, 6, seed=1)
    f16, r32 = narrow_row_index(fbr, rbr, 40)
    assert f16.dtype == torch.int16 and r32.dtype == torch.int32
    assert torch.equal(f16.to(torch.int64), fbr)
    assert torch.equal(r32.to(torch.int64), rbr)
    f, _ = narrow_row_index(torch.tensor([0, 32766]), torch.tensor([1, 2]), 32767)
    assert f.dtype == torch.int16 and f.tolist() == [0, 32766]


def test_narrowing_uses_int32_feature_ids_from_32768_features():
    f, _ = narrow_row_index(torch.tensor([0, 40959]), torch.tensor([1, 2]), 40960)
    assert f.dtype == torch.int32 and f.tolist() == [0, 40959]
    f, _ = narrow_row_index(torch.tensor([0, 32767]), torch.tensor([1, 2]), 32768)
    assert f.dtype == torch.int32


def test_chunked_ap_at_the_int16_boundary_keeps_the_last_ranges_touches():
    # review's case: 32,768 features, touches on the top ids. With the range
    # bound wrapping in int16 these would score at the base rate.
    n = 32768
    fbr = torch.tensor([n - 1, n - 1, n - 2, 5, n - 1], dtype=torch.int64)
    rbr = torch.tensor([1, 2, 1, 1, 3], dtype=torch.int64)
    fn, rn = narrow_row_index(fbr, rbr, n)
    want = average_precision_from_touches(fbr, rbr, n, 4, 100)
    got = average_precision_from_touches_chunked(fn, rn, n, 4, 100, 2)
    assert torch.equal(got, want)
    assert got[n - 1].item() != 4 / 100


@pytest.mark.parametrize("feat, rank, n_features", [
    ([0, 40], [1, 2], 40),              # a feature id at n_features
    ([-1, 3], [1, 2], 40),              # a negative feature id
    ([0, 3], [0, 2], 40),               # a rank below 1
    ([0, 3], [1, 2 ** 31], 40),         # a rank that int32 would wrap
])
def test_narrowing_refuses_a_value_it_cannot_hold(feat, rank, n_features):
    with pytest.raises(ValueError):
        narrow_row_index(torch.tensor(feat), torch.tensor(rank), n_features)


# ---- gather_touches_chunked -------------------------------------------------

@pytest.mark.parametrize("budget", [1, 3, 7, 50, 10 ** 9])
def test_chunked_gather_returns_the_unchunked_sequence(budget):
    fbr, rbr, off = _layout(300, 40, 6, seed=2)
    f16, r32 = narrow_row_index(fbr, rbr, 40)
    q = torch.tensor([3, 5, 0, 299, 17, 3 * 7, 120, 4], dtype=torch.int64)  # row 3 fires none
    want_f, want_r = gather_touches(f16, r32, off, q)
    got_f, got_r = gather_touches_chunked(f16, r32, off, q, budget)
    assert got_f.dtype == f16.dtype and got_r.dtype == r32.dtype
    assert torch.equal(got_f, want_f) and torch.equal(got_r, want_r)


def test_chunked_gather_counts_only_draws_it_actually_chunks():
    fbr, rbr, off = _layout(300, 40, 6, seed=3)
    q = torch.arange(0, 50, dtype=torch.int64)
    stats: dict = {}
    gather_touches_chunked(fbr, rbr, off, q, 10 ** 9, stats)
    assert stats.get("chunked_gathers", 0) == 0
    gather_touches_chunked(fbr, rbr, off, q, 5, stats)
    assert stats["chunked_gathers"] == 1


def test_chunked_gather_skips_a_piece_of_rows_that_fire_nothing():
    # rows 0..2 fire 2 each, rows 3..5 fire nothing, row 6 fires 2
    off = torch.tensor([0, 2, 4, 6, 6, 6, 6, 8], dtype=torch.int64)
    fbr = torch.tensor([0, 1, 1, 2, 0, 2, 1, 0], dtype=torch.int16)
    rbr = torch.tensor([1, 1, 2, 1, 2, 2, 3, 3], dtype=torch.int32)
    q = torch.arange(7, dtype=torch.int64)
    want = gather_touches(fbr, rbr, off, q)
    got = gather_touches_chunked(fbr, rbr, off, q, 2)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


# ---- feature_chunk_bounds ---------------------------------------------------

def test_feature_chunk_bounds_on_a_written_down_example():
    # counts 3,1,0,5,2 at budget 4: [0,3) holds 4; feature 3 holds 5 > 4 and is
    # a range alone; [4,5) holds 2.
    assert feature_chunk_bounds(torch.tensor([3, 1, 0, 5, 2]), 4) == [(0, 3), (3, 4), (4, 5)]
    assert feature_chunk_bounds(torch.tensor([3, 1, 0, 5, 2]), 100) == [(0, 5)]
    assert feature_chunk_bounds(torch.tensor([1, 1, 1]), 1) == [(0, 1), (1, 2), (2, 3)]


def test_feature_chunk_bounds_cover_every_feature_once_within_budget():
    rng = np.random.default_rng(4)
    counts = torch.from_numpy(rng.integers(0, 30, size=200))
    for budget in (1, 10, 29, 64, 1000):
        b = feature_chunk_bounds(counts, budget)
        assert b[0][0] == 0 and b[-1][1] == 200
        assert all(b[i][1] == b[i + 1][0] for i in range(len(b) - 1))
        for f0, f1 in b:
            s = int(counts[f0:f1].sum())
            assert s <= budget or f1 - f0 == 1


# ---- average_precision_from_touches_chunked --------------------------------

@pytest.mark.parametrize("seed", [5, 6, 7])
@pytest.mark.parametrize("budget", [1, 4, 13, 40, 10 ** 9])
def test_chunked_ap_is_bitwise_the_unchunked_ap_on_cpu(seed, budget):
    n_rows, n_features = 400, 30
    fbr, rbr, off = _layout(n_rows, n_features, 6, seed=seed)
    f16, r32 = narrow_row_index(fbr, rbr, n_features)
    rng = np.random.default_rng(seed + 100)
    pos_rows = np.unique(rng.choice(n_rows, 60, replace=False)).astype(np.int64)
    q = torch.from_numpy(pos_rows)
    tf, tr = gather_touches(fbr, rbr, off, q)
    want = average_precision_from_touches(tf, tr, n_features, len(pos_rows), n_rows)
    stats: dict = {}
    tfn, trn = gather_touches_chunked(f16, r32, off, q, budget, stats)
    got = average_precision_from_touches_chunked(tfn, trn, n_features, len(pos_rows),
                                                 n_rows, budget, stats)
    assert got.dtype == torch.float64
    assert torch.equal(got, want), (got - want).abs().max()
    if budget < tf.numel():
        assert stats["chunked_ap_evaluations"] == 1 and stats["max_feature_chunks"] > 1
    else:
        assert stats.get("chunked_ap_evaluations", 0) == 0


def test_chunked_ap_scores_untouched_features_at_the_base_rate():
    # touches only on feature 1; features 0 and 2 are silent for this concept
    tf = torch.tensor([1, 1, 1], dtype=torch.int16)
    tr = torch.tensor([1, 2, 5], dtype=torch.int32)
    got = average_precision_from_touches_chunked(tf, tr, 3, 4, 100, 1)
    want = average_precision_from_touches(tf.long(), tr.long(), 3, 4, 100)
    assert torch.equal(got, want)
    assert got[0].item() == got[2].item() == 4 / 100


def test_the_row_index_cache_narrows_on_load_and_leaves_the_entry_int64(tmp_path):
    import h5py
    from src.analysis.row_index_cache import load_row_index, save_row_index
    fbr, rbr, off = _layout(200, 25, 5, seed=8)
    path = tmp_path / "entry.h5"
    save_row_index(path, fbr, rbr, off, torch.zeros(25, dtype=torch.int64), meta={"k": 5})
    f, r, o, _ = load_row_index(path, "cpu", narrow_features=25)
    assert f.dtype == torch.int16 and r.dtype == torch.int32 and o.dtype == torch.int64
    assert torch.equal(f.long(), fbr) and torch.equal(r.long(), rbr) and torch.equal(o, off)
    f0, r0, _, _ = load_row_index(path, "cpu")
    assert f0.dtype == torch.int64 and r0.dtype == torch.int64
    with h5py.File(path, "r") as hf:
        assert hf["feat_by_row"].dtype == np.int64 and hf["rank_by_row"].dtype == np.int64


# ---- §CT29-D2 item 2: standardize_gain's top-bin merge ----------------------

def test_merge_top_firing_bin_on_arm_iii_edges_gives_the_declared_list():
    from src.analysis.standardize_gain import merge_top_firing_bin
    as_run = [0.0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 1.000000001]
    assert merge_top_firing_bin(as_run) == [0.0, 0.0001, 0.0003, 0.001, 0.003, 0.01,
                                            1.000000001]


def test_merge_top_firing_bin_refuses_a_single_bin_and_differs_from_the_lowest_merge():
    from src.analysis.standardize_gain import merge_lowest_firing_bin, merge_top_firing_bin
    with pytest.raises(ValueError):
        merge_top_firing_bin([0.0, 1.000000001])
    e = [0.0, 0.01, 0.03, 1.000000001]
    assert merge_top_firing_bin(e) == [0.0, 0.01, 1.000000001]
    assert merge_lowest_firing_bin(e) == [0.0, 0.03, 1.000000001]
