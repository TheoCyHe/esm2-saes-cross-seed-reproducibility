"""
Tests for the P1 restructure (src/analysis/sparse_ap_index.py).

The bar: EXACT agreement with `average_precision_segmented` (already
validated against sklearn to 1e-9 in test_feature_ap.py), not "close enough".
The reformulation's whole argument is that restricting to positive-row
touches is the same sum with zero terms omitted -- these tests are what turn
that argument into a checked fact.

Specifically exercised: rows with FEWER than k non-zeros (TopK's ReLU-before-
gate can produce this; a fixed-width (n_rows, k) reshape would silently
misalign here), a completely silent row, concepts whose positives sit
entirely at zero-activation rows for some feature, and the usual AP edge
cases (zero positives, perfect detector, tiny/degenerate concepts).
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from src.analysis.feature_ap import average_precision_segmented
from src.analysis.sparse_ap_index import (
    argmax_lowest_index,
    average_precision_from_touches,
    build_row_index,
    gather_touches,
)


def _feature_major(Z: np.ndarray):
    """Dense (n_rows, n_features) -> (feat, pos, seg_start), sorted (feature
    asc, value desc) -- exactly what sparse_activations produces."""
    n_rows, n_features = Z.shape
    feats, poss = [], []
    for f in range(n_features):
        nz = np.flatnonzero(Z[:, f])
        order = nz[np.argsort(-Z[nz, f], kind="stable")]
        feats.append(np.full(len(order), f, dtype=np.int64))
        poss.append(order.astype(np.int64))
    feat = np.concatenate(feats) if feats else np.zeros(0, dtype=np.int64)
    pos = np.concatenate(poss) if poss else np.zeros(0, dtype=np.int64)
    counts = np.bincount(feat, minlength=n_features)
    seg_start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return (torch.from_numpy(feat), torch.from_numpy(pos),
            torch.from_numpy(seg_start).to(torch.int64))


def _ap_old(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    feat, pos, seg_start = _feature_major(Z)
    labels = torch.from_numpy(y.astype(bool))[pos]
    return average_precision_segmented(
        feat, seg_start, labels, Z.shape[1], int(y.sum()), Z.shape[0]).numpy()


def _ap_new(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    n_rows, n_features = Z.shape
    feat, pos, seg_start = _feature_major(Z)
    feat_by_row, rank_by_row, row_offsets = build_row_index(feat, pos, seg_start, n_rows)
    query_rows = torch.from_numpy(np.flatnonzero(y).astype(np.int64))
    tf, tr = gather_touches(feat_by_row, rank_by_row, row_offsets, query_rows)
    return average_precision_from_touches(
        tf, tr, n_features, int(y.sum()), n_rows).numpy()


def _assert_matches(Z, y, atol=1e-9):
    old, new = _ap_old(Z, y), _ap_new(Z, y)
    assert np.allclose(old, new, atol=atol), (old, new, np.abs(old - new).max())
    return old, new


def test_matches_old_path_on_random_sparse_activations():
    rng = np.random.default_rng(0)
    n_rows, n_features, k = 500, 20, 5
    Z = np.zeros((n_rows, n_features))
    for i in range(n_rows):
        cols = rng.choice(n_features, k, replace=False)
        Z[i, cols] = rng.uniform(0.1, 5.0, size=k)
    y = (rng.random(n_rows) < 0.12).astype(int)
    _assert_matches(Z, y)


def test_matches_old_path_with_rare_positives_across_many_configs():
    rng = np.random.default_rng(7)
    for trial in range(8):
        n_rows = rng.integers(200, 1200)
        n_features = rng.integers(5, 30)
        k = min(n_features, rng.integers(2, 8))
        Z = np.zeros((n_rows, n_features))
        for i in range(n_rows):
            cols = rng.choice(n_features, k, replace=False)
            Z[i, cols] = rng.uniform(0.1, 5.0, size=k)
        n_pos = max(1, int(n_rows * rng.uniform(0.001, 0.15)))
        y = np.zeros(n_rows, dtype=int)
        y[rng.choice(n_rows, n_pos, replace=False)] = 1
        _assert_matches(Z, y)


def test_matches_old_path_when_rows_fire_fewer_than_k_features():
    """THE structural subtlety: TopK's ReLU-before-gate means a row can fire
    on FEWER than k features (some 'selected' slots are exactly zero and
    .nonzero() drops them). Deliberately construct rows with 0, 1, 3, and 5
    non-zeros in the SAME dataset -- a fixed-width (n_rows, k) assumption
    would misalign the moment row widths differ."""
    n_rows, n_features = 40, 10
    Z = np.zeros((n_rows, n_features))
    rng = np.random.default_rng(3)
    widths = [0, 1, 3, 5] * (n_rows // 4)
    for i, w in enumerate(widths):
        if w == 0:
            continue                              # completely silent row
        cols = rng.choice(n_features, w, replace=False)
        Z[i, cols] = rng.uniform(0.1, 5.0, size=w)
    y = np.zeros(n_rows, dtype=int)
    y[rng.choice(n_rows, 10, replace=False)] = 1
    _assert_matches(Z, y)


def test_matches_old_path_when_concept_positives_are_mostly_zero_for_a_feature():
    """A feature that fires on almost none of the concept's positive rows --
    the tail term must dominate and match exactly, not just approximately."""
    n_rows = 2000
    Z = np.zeros((n_rows, 3))
    rng = np.random.default_rng(11)
    Z[:400, 0] = rng.uniform(0.1, 5.0, 400)         # feature 0 fires broadly
    Z[5:8, 1] = rng.uniform(0.1, 5.0, 3)             # feature 1 fires rarely
    # feature 2 never fires (fully silent feature)
    y = np.zeros(n_rows, dtype=int)
    y[np.arange(500, 600)] = 1                       # positives mostly NOT overlapping feat 1
    _assert_matches(Z, y)


def test_matches_sklearn_directly_via_the_new_path():
    """Closes the loop independently of average_precision_segmented: the new
    path must ALSO match sklearn directly, not just match the old path."""
    rng = np.random.default_rng(5)
    n_rows, n_features, k = 800, 15, 4
    Z = np.zeros((n_rows, n_features))
    for i in range(n_rows):
        cols = rng.choice(n_features, k, replace=False)
        Z[i, cols] = rng.uniform(0.1, 5.0, size=k)
    y = (rng.random(n_rows) < 0.08).astype(int)
    new = _ap_new(Z, y)
    for f in range(n_features):
        ref = average_precision_score(y, Z[:, f])
        assert abs(new[f] - ref) < 1e-9, (f, new[f], ref)


def test_zero_positives_returns_base_rate_zero_not_nan():
    Z = np.zeros((50, 3))
    Z[:10, 0] = 1.0
    y = np.zeros(50, dtype=int)
    old, new = _assert_matches(Z, y)
    assert np.all(new == 0.0) and np.isfinite(new).all()


def test_perfect_detector_matches():
    y = np.zeros(100, dtype=int)
    y[:10] = 1
    Z = np.zeros((100, 1))
    Z[:10, 0] = np.linspace(5, 1, 10)
    _assert_matches(Z, y)
    assert abs(_ap_new(Z, y)[0] - 1.0) < 1e-12


def test_silent_feature_scores_base_rate():
    y = np.zeros(200, dtype=int)
    y[:20] = 1
    Z = np.zeros((200, 1))
    old, new = _assert_matches(Z, y)
    assert abs(new[0] - 0.1) < 1e-12


def test_tiny_degenerate_concept_matches():
    """The amendment-B failure mode, re-checked through the new path: a
    1-positive concept must still score exactly 1.0 for a feature that fires
    on it, matching the old path bit-for-bit."""
    Z = np.zeros((1000, 1))
    Z[7, 0] = 3.0
    y1 = np.zeros(1000, dtype=int); y1[7] = 1
    _assert_matches(Z, y1)
    assert abs(_ap_new(Z, y1)[0] - 1.0) < 1e-12

    y2 = np.zeros(1000, dtype=int); y2[[7, 500]] = 1
    _assert_matches(Z, y2)


def test_rank_is_global_within_the_feature_not_local_to_the_touched_set():
    """THE critical correctness question: AP's rank must be 'how many of this
    FEATURE's firing rows have activation >= mine' -- computed against the
    full block -- not 'how many of the CONCEPT's touched rows for this
    feature have equal-or-higher rank', which is a different (wrong) number
    whenever a feature has far more firing rows than the concept has
    positives touching it.

    Adversarial construction: feature 0 fires on 999 rows; the concept's one
    positive row is the row with the LOWEST activation among those 999 --
    i.e. the WORST possible rank (999th of 999). A bug that computed rank
    only among the concept's OWN touches for that feature (a single touch)
    would place it at rank 1 and give AP near 1.0. The correct global rank
    gives AP near the base rate (roughly last place), matching the old path
    exactly.
    """
    n_rows = 1200
    Z = np.zeros((n_rows, 2))
    rng = np.random.default_rng(0)
    firing_rows = rng.choice(n_rows, 999, replace=False)
    vals = np.sort(rng.uniform(0.01, 5.0, 999))[::-1]
    Z[firing_rows, 0] = vals
    Z[rng.choice(n_rows, 20, replace=False), 1] = rng.uniform(0.1, 5.0, 20)

    worst_row = firing_rows[np.argmin(Z[firing_rows, 0])]
    y = np.zeros(n_rows, dtype=int)
    y[worst_row] = 1
    zero_rows = [r for r in range(n_rows) if r != worst_row and Z[r, 0] == 0]
    y[rng.choice(zero_rows, 1)] = 1

    old, new = _assert_matches(Z, y)
    # sanity: a worst-rank-only positive must score LOW, not near 1.0 -- this
    # is what a rank-among-touches bug would have produced instead
    assert old[0] < 0.05, f"expected near-last-place AP, got {old[0]}"


def test_gather_touches_handles_a_query_row_with_zero_touches():
    """A queried positive row that fires on NOTHING must contribute zero
    touches, not error -- e.g. a genuinely silent token."""
    n_rows, n_features = 20, 4
    Z = np.zeros((n_rows, n_features))
    Z[0, 0] = 1.0                                    # row 0 fires; row 1 never does
    feat, pos, seg_start = _feature_major(Z)
    feat_by_row, rank_by_row, row_offsets = build_row_index(feat, pos, seg_start, n_rows)
    tf, tr = gather_touches(feat_by_row, rank_by_row, row_offsets,
                            torch.tensor([0, 1], dtype=torch.int64))
    assert tf.numel() == 1 and tf[0].item() == 0     # only row 0's touch


# ---------------------------------------------- explicit tie-break argmax --

def test_argmax_lowest_index_breaks_ties_correctly():
    x = torch.tensor([0.1, 0.5, 0.5, 0.3, 0.5, 0.2])
    assert argmax_lowest_index(x).item() == 1        # lowest of the three 0.5s


def test_argmax_lowest_index_tie_at_the_very_start():
    x = torch.tensor([0.9, 0.9, 0.1])
    assert argmax_lowest_index(x).item() == 0


def test_argmax_lowest_index_tie_at_the_very_end():
    x = torch.tensor([0.1, 0.2, 0.9, 0.9])
    assert argmax_lowest_index(x).item() == 2


def test_argmax_lowest_index_no_tie_matches_ordinary_argmax():
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.random(50))
    assert argmax_lowest_index(x).item() == torch.argmax(x).item()


def test_argmax_lowest_index_batched():
    X = torch.tensor([[0.1, 0.5, 0.5, 0.3], [0.9, 0.9, 0.1, 0.9], [0.1, 0.2, 0.3, 0.4]])
    got = argmax_lowest_index(X, dim=1)
    assert got.tolist() == [1, 0, 3]


def test_argmax_lowest_index_would_catch_a_last_wins_regression():
    """The detector must actually distinguish 'lowest index wins' from 'an
    arbitrary/last index wins' -- construct a case where a naive
    last-occurrence tie-break gives a DIFFERENT answer, and confirm ours
    does not match it."""
    x = torch.tensor([0.7, 0.9, 0.9, 0.9, 0.2])
    lowest = argmax_lowest_index(x).item()
    last_occurrence = 3      # what a 'last wins' tie-break would return
    assert lowest == 1
    assert lowest != last_occurrence
