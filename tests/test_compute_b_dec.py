"""
Tests for src/sae/compute_b_dec.py -- the shared, frozen decoder bias.

b_dec is computed ONCE and must be identical across all 20 seeds (CLAUDE.md),
so the two properties that matter here are: (1) the geometric median is a real
robust center, not silently the mean, and (2) the sampling is deterministic
given its fixed seed, so re-running reproduces the same vector. Both are
written so they can actually fail.
"""

from __future__ import annotations

import h5py
import numpy as np

from src.sae.compute_b_dec import geometric_median, sample_rows


def _make_cache(path, rows, d=4, start=0):
    data = (np.arange(start, start + rows, dtype=np.float32)[:, None]
            * np.ones((1, d), dtype=np.float32)).astype(np.float16)
    with h5py.File(path, "w") as hf:
        hf.create_dataset("activations", data=data,
                          chunks=(min(rows, 4096), d))
    return data


def test_geometric_median_is_robust_to_an_outlier():
    """The whole reason to prefer GM over mean: one huge outlier drags the mean
    but barely moves the geometric median. If GM were secretly the mean this
    fails."""
    X = np.zeros((100, 3), dtype=np.float32)
    X[0] = 1000.0                      # a single far outlier
    gm, iters = geometric_median(X)
    mean = X.mean(axis=0)
    assert np.linalg.norm(gm) < np.linalg.norm(mean)
    assert np.linalg.norm(gm) < 1.0    # GM sits at the dense cluster (~0)
    assert iters >= 1


def test_geometric_median_of_symmetric_points_is_the_center():
    X = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.float32)
    gm, _ = geometric_median(X)
    np.testing.assert_allclose(gm, [0.0, 0.0], atol=1e-4)


def test_sample_rows_is_deterministic_given_seed(tmp_path):
    # start values kept well within fp16's exact range to stay distinguishable
    _make_cache(tmp_path / "a.h5", 20000, start=0)
    _make_cache(tmp_path / "b.h5", 20000, start=20000)
    paths = [tmp_path / "a.h5", tmp_path / "b.h5"]

    X1, total1 = sample_rows(paths, n_sample=5000, seed=0)
    X2, total2 = sample_rows(paths, n_sample=5000, seed=0)
    assert total1 == total2 == 40000
    np.testing.assert_array_equal(X1, X2)          # same seed -> same sample

    X3, _ = sample_rows(paths, n_sample=5000, seed=1)
    # different seed should draw a different set of blocks (overwhelmingly
    # likely with 5000 of 40000 rows)
    assert not (X1.shape == X3.shape and np.array_equal(X1, X3))


def test_sample_rows_returns_all_when_cache_small(tmp_path):
    _make_cache(tmp_path / "a.h5", 300)
    X, total = sample_rows([tmp_path / "a.h5"], n_sample=10000, seed=0)
    assert total == 300
    assert X.shape[0] == 300                        # nothing dropped
    # and every row is present exactly once (identifiable by contents)
    assert sorted(X[:, 0].tolist()) == list(range(300))


def test_sample_rows_pools_across_shards(tmp_path):
    _make_cache(tmp_path / "a.h5", 8192, start=0)          # values 0..8191
    _make_cache(tmp_path / "b.h5", 8192, start=20000)      # values 20000..28191
    X, total = sample_rows([tmp_path / "a.h5", tmp_path / "b.h5"],
                           n_sample=16384, seed=0)
    assert total == 16384
    vals = X[:, 0].tolist()
    assert any(v < 14000 for v in vals)             # rows from shard a
    assert any(v >= 14000 for v in vals)            # rows from shard b
