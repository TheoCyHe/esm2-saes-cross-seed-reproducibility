"""Tests for M3 (`mu_s` with and without the degenerate concepts).

Gotcha 13: each check is verified to FAIL against the condition it detects.
Mutations run and killed (see the M3 commit message):
  * `p95 == 0` -> `p95 <= 0` in `degenerate_mask` (no kill -- equivalent on
    non-negative data, so it is NOT claimed as a covered mutation)
  * dropping the `keep` subsetting so mu_excl == mu_all -> kills
    test_excluding_a_zero_concept_raises_mu
  * marking every stratum affected -> kills test_only_containing_strata_are_affected
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.mu_s_degenerate import (
    degenerate_mask,
    mu_with_and_without,
    summarise,
)


def test_degenerate_mask_finds_a_point_mass_at_zero():
    rng = np.random.default_rng(0)
    slab = rng.random((5, 4, 50)) + 0.1          # all strictly positive
    slab[:, 2, :] = 0.0                          # concept 2 is a point mass
    m = degenerate_mask(slab)
    assert m.tolist() == [False, False, True, False]


def test_degenerate_mask_tolerates_a_sparse_tail():
    """>=95% zeros still counts as degenerate; that is the stated criterion,
    not an approximation of it."""
    slab = np.zeros((10, 2, 100))
    slab[0, 0, 0] = 5.0                          # 1 of 1000 draws non-zero
    slab[:, 1, :] = 1.0                          # concept 1 never zero
    m = degenerate_mask(slab)
    assert m[0] and not m[1]


def test_degenerate_mask_rejects_a_merely_small_concept():
    """A concept with small but ubiquitous values is NOT degenerate. Without
    this the criterion would silently become 'small' rather than 'zero'."""
    slab = np.full((4, 2, 40), 1e-9)
    slab[:, 1, :] = 1.0
    assert not degenerate_mask(slab).any()


def _fixture(n_shift=4, n_c=6, n_f=30, seed=0):
    rng = np.random.default_rng(seed)
    slab = rng.random((n_shift, n_c, n_f)) + 0.5
    cell_pool = np.array([0, 0, 0, 1, 1, 1])
    fb = np.zeros(n_f, dtype=int)
    return slab, cell_pool, fb


def test_excluding_a_zero_concept_raises_mu():
    slab, cell_pool, fb = _fixture()
    slab[:, 1, :] = 0.0                          # degenerate, in cell 0
    drop = np.array([False, True, False, False, False, False])
    res = mu_with_and_without(slab, cell_pool, 2, fb, 1, drop)
    assert res["mu_excl"][0, 0] > res["mu_all"][0, 0]
    # cell 1 contains no dropped concept, so it must be untouched EXACTLY.
    assert res["mu_excl"][1, 0] == pytest.approx(res["mu_all"][1, 0])


def test_only_containing_strata_are_affected():
    slab, cell_pool, fb = _fixture()
    drop = np.array([False, True, False, False, False, False])
    res = mu_with_and_without(slab, cell_pool, 2, fb, 1, drop)
    assert res["affected"][0, 0]
    assert not res["affected"][1, 0]


def test_dropping_nothing_is_the_identity():
    """The no-op case must be exactly a no-op -- an operation that can
    silently do nothing is the gotcha-14 defect, so its converse is pinned."""
    slab, cell_pool, fb = _fixture()
    drop = np.zeros(6, dtype=bool)
    res = mu_with_and_without(slab, cell_pool, 2, fb, 1, drop)
    assert np.allclose(res["mu_all"], res["mu_excl"])
    assert res["affected"].sum() == 0


def test_summarise_derives_D1_and_can_report_a_violation():
    """D1's verdict must be computed from the data, and it must be ABLE to
    say MISS -- a check that can only pass is not a check."""
    slab, cell_pool, fb = _fixture()
    slab[:, 1, :] = 0.0
    drop = np.array([False, True, False, False, False, False])
    res = mu_with_and_without(slab, cell_pool, 2, fb, 1, drop)
    assert summarise(res)["D1_outcome"] == "HIT"

    # Force the violation the prediction says should not occur: make the
    # dropped concept the LARGEST in its stratum instead of zero.
    slab2, cell_pool2, fb2 = _fixture(seed=1)
    slab2[:, 1, :] = 100.0
    res2 = mu_with_and_without(slab2, cell_pool2, 2, fb2, 1, drop)
    s2 = summarise(res2)
    assert s2["D1_outcome"] == "MISS"
    assert s2["D1_n_violations"] >= 1


def test_summarise_separates_affected_from_unaffected_ratios():
    slab, cell_pool, fb = _fixture()
    slab[:, 1, :] = 0.0
    drop = np.array([False, True, False, False, False, False])
    s = summarise(mu_with_and_without(slab, cell_pool, 2, fb, 1, drop))
    assert s["n_strata_affected"] == 1
    assert s["ratio_affected"]["median"] > 1.0
    # An unaffected stratum's ratio must be exactly 1.
    assert s["ratio_unaffected"]["median"] == pytest.approx(1.0)
