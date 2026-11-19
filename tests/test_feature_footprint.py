"""Tests for the per-protein footprint grouping.

THESE DID NOT EXIST, WHICH IS PART OF WHY AO6 WENT UNNOTICED FOR THREE SESSIONS.
`feature_footprint.py` had no test file at all; its only internal check compared
a TOTAL (`n_rows == len(prot_of_row)`), and a check on a total cannot see a
permutation. The fixtures below deliberately carry a NON-ASCENDING
`population_start` -- the condition the old assertion could not exhibit
(gotcha 16's fixture lesson).

MUTATION RUN AND KILLED: restoring `np.repeat(np.arange(n_prot), nrow)` on the
arrays AS STORED fails `test_grouping_follows_start_order_not_storage_order` and
`test_footprint_attributes_fires_to_the_right_protein`.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from src.analysis.feature_footprint import (
    footprint,
    population_order,
    protein_of_row,
)

# Three proteins. STORED order is sorted-accession (A, B, C) but their cache
# starts are NOT ascending in that order -- exactly the real store's shape.
#   stored idx 0 = "A": start 5, 2 rows   -> start-order rank 1, positions 3-4
#   stored idx 1 = "B": start 0, 3 rows   -> start-order rank 0, positions 0-2
#   stored idx 2 = "C": start 7, 4 rows   -> start-order rank 2, positions 5-8
STORED_START = [5, 0, 7]
STORED_NROW = [2, 3, 4]
N_POS = sum(STORED_NROW)                                        # 9


def _labels(tmp_path, start=STORED_START, nrow=STORED_NROW, name="lab.h5"):
    p = tmp_path / name
    with h5py.File(p, "w") as hf:
        pg = hf.create_group("protein_sets")
        pg.create_dataset("population_start", data=np.array(start, np.int64))
        pg.create_dataset("population_n_rows", data=np.array(nrow, np.int64))
        hf.attrs["n_rows"] = int(np.sum(nrow))
    return p


def test_population_start_fixture_is_not_ascending():
    """If the fixture were ascending it could not distinguish the two orders,
    and the whole file would pass vacuously."""
    assert not np.all(np.diff(STORED_START) > 0)


def test_grouping_follows_start_order_not_storage_order(tmp_path):
    prot, n = protein_of_row(_labels(tmp_path))
    assert n == 3
    # start order is B(3 rows), A(2 rows), C(4 rows) -> ids 0,0,0,1,1,2,2,2,2
    assert prot.tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2]
    # the pre-AO6 map would have been A(2), B(3), C(4) -> 0,0,1,1,1,2,2,2,2
    assert prot.tolist() != [0, 0, 1, 1, 1, 2, 2, 2, 2]


def test_total_length_is_the_same_under_both_orders(tmp_path):
    """The reason the old assertion was blind, stated as a test: the totals
    agree, so only an ORDER check can discriminate."""
    prot, _ = protein_of_row(_labels(tmp_path))
    assert prot.size == N_POS == np.repeat(np.arange(3), STORED_NROW).size


def test_grouping_is_non_decreasing_so_searchsorted_can_find_boundaries(tmp_path):
    prot, _ = protein_of_row(_labels(tmp_path))
    assert np.all(np.diff(prot) >= 0)


def test_population_order_is_the_permutation_to_stored_indices(tmp_path):
    order = population_order(_labels(tmp_path))
    assert order.tolist() == [1, 0, 2]          # B, A, C
    # composing a per-protein attribute through it must recover the right value
    stored_attr = np.array(["A", "B", "C"])
    assert stored_attr[order].tolist() == ["B", "A", "C"]


def test_overlapping_row_ranges_are_refused(tmp_path):
    lab = _labels(tmp_path, start=[0, 2], nrow=[5, 5])   # [0,5) and [2,7)
    with pytest.raises(SystemExit, match="OVERLAP"):
        protein_of_row(lab)


def test_negative_length_is_refused(tmp_path):
    lab = _labels(tmp_path, start=[0, 10], nrow=[5, -1])
    with pytest.raises(SystemExit, match="negative length"):
        protein_of_row(lab)


def test_a_gap_between_proteins_is_allowed(tmp_path):
    """The construction must not depend on the population being CONTIGUOUS in
    cache-row space -- it depends only on the sort. A store built under
    truncation_policy=with_truncated would have gaps."""
    lab = _labels(tmp_path, start=[100, 0], nrow=[2, 3])
    prot, n = protein_of_row(lab)
    assert n == 2
    assert prot.tolist() == [0, 0, 0, 1, 1]


# ---------------------------------------------------------------------------
# END TO END. The strongest form: a hand-built row index whose per-protein
# answer is known, so the grouping is proven rather than inspected.
# ---------------------------------------------------------------------------
def _row_index(tmp_path, fires_by_position, n_features):
    """`fires_by_position[i]` is the list of features firing at row position i."""
    offs = np.zeros(len(fires_by_position) + 1, np.int64)
    flat = []
    for i, fs in enumerate(fires_by_position):
        flat.extend(fs)
        offs[i + 1] = len(flat)
    p = tmp_path / "ri.h5"
    with h5py.File(p, "w") as hf:
        hf.create_dataset("feat_by_row", data=np.array(flat, np.int32))
        hf.create_dataset("row_offsets", data=offs)
        hf.attrs["n_features"] = n_features
    return p


def test_footprint_attributes_fires_to_the_right_protein(tmp_path):
    lab = _labels(tmp_path)
    prot, n_prot = protein_of_row(lab)
    # positions 0-2 = protein B, 3-4 = A, 5-8 = C  (start order)
    # feature 0 fires 3x, all inside B          -> 1 protein, entropy 0
    # feature 1 fires once at every position    -> 3 proteins
    # feature 2 fires 2x, both inside A         -> 1 protein
    fires = [[0, 1], [0, 1], [0, 1], [1, 2], [1, 2], [1], [1], [1], [1]]
    ri = _row_index(tmp_path, fires, 3)
    fp = footprint(ri, prot, n_prot, 3)

    assert fp["n_proteins"].tolist() == [1.0, 3.0, 1.0]
    assert fp["total_fires"].tolist() == [3.0, 9.0, 2.0]
    # a feature confined to ONE protein has zero entropy and perplexity 1
    assert fp["entropy"][0] == pytest.approx(0.0)
    assert fp["perplexity"][0] == pytest.approx(1.0)
    assert fp["entropy"][2] == pytest.approx(0.0)
    # the spread feature: counts 3 / 2 / 4 over B / A / C
    c = np.array([3.0, 2.0, 4.0])
    want_h = np.log(c.sum()) - (c * np.log(c)).sum() / c.sum()
    assert fp["entropy"][1] == pytest.approx(want_h)
    assert fp["concentration"][1] == pytest.approx((c ** 2).sum() / c.sum() ** 2)

    # AND THE DISCRIMINATION IS EXPLICIT: under the pre-AO6 grouping feature 0
    # would have straddled two proteins and reported a non-zero entropy.
    # Under it the blocks are A(positions 0-1), B(2-4), C(5-8), so feature 0's
    # three fires at 0,1,2 straddle A and B.
    buggy = np.repeat(np.arange(3, dtype=np.int32), STORED_NROW)
    fp_buggy = footprint(ri, buggy, n_prot, 3)
    assert fp_buggy["n_proteins"].tolist() == [2.0, 3.0, 1.0]
    assert fp_buggy["entropy"][0] > 0.5
    assert fp["entropy"][0] == pytest.approx(0.0)
    assert fp_buggy["perplexity"][0] != pytest.approx(fp["perplexity"][0])


def test_footprint_asserts_the_row_count_matches(tmp_path):
    lab = _labels(tmp_path)
    prot, n_prot = protein_of_row(lab)
    ri = _row_index(tmp_path, [[0]] * (N_POS - 1), 1)      # one row short
    with pytest.raises(AssertionError):
        footprint(ri, prot, n_prot, 1)
