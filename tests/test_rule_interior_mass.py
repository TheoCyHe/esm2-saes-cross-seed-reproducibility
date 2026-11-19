"""Tests for the CO item B split (amendment CO).

Expectations are hand-computed and written down, never derived from the module
under test (gotcha 32b).
"""
import numpy as np
import pytest

from src.analysis.rule_interior_mass import RECORD_N19_COUNTS, three_way_split


def test_three_way_split_hand_computed():
    """A 2x5 counter with n_compare = 3:

        [[0, 0, 1, 3, 2],
         [3, 1, 0, 2, 3]]

    zeros = 3 (cells 0,1 of row 0 and cell 2 of row 1); k = 3 = 3 (cell 3 of
    row 0, cells 0 and 4 of row 1); interior = 10 - 3 - 3 = 4.
    Sum of k = 0+0+1+3+2+3+1+0+2+3 = 15, so mean k = 1.5 and mean k/3 = 0.5.
    """
    c = np.array([[0, 0, 1, 3, 2], [3, 1, 0, 2, 3]])
    s = three_way_split(c, 3)
    assert (s["n_cells"], s["n_k_zero"], s["n_interior"], s["n_k_max"]) == (
        10, 3, 4, 3)
    assert s["P_k_zero"] == pytest.approx(0.3)
    assert s["interior_mass"] == pytest.approx(0.4)
    assert s["P_k_max"] == pytest.approx(0.3)
    assert s["mean_k_over_n_compare"] == pytest.approx(0.5)


def test_three_way_split_shares_sum_to_one():
    """The three shares partition the population by construction; a split that
    did not sum to 1 would mean a cell was counted in two bands."""
    rng = np.random.default_rng(3)
    c = rng.integers(0, 20, size=(20, 1000))
    s = three_way_split(c, 19)
    assert s["P_k_zero"] + s["interior_mass"] + s["P_k_max"] == pytest.approx(1.0)


def test_three_way_split_refuses_a_count_above_n_compare():
    """A count above `n_compare` means an anchor was credited against itself or
    a pair was counted twice -- a silent doubling that would look like an
    ordinary distribution. It must raise."""
    with pytest.raises(SystemExit, match="outside"):
        three_way_split(np.array([[0, 20]]), 19)


def test_three_way_split_refuses_a_negative_count():
    with pytest.raises(SystemExit, match="outside"):
        three_way_split(np.array([[-1, 3]]), 19)


def test_the_record_targets_partition_the_population():
    """The control target is quoted from docs/NUMBERS_OF_RECORD.md 14.1 as
    three integers. If they did not sum to 204,800 the quotation would be
    wrong, and the gate would be measuring against a target that cannot
    describe any distribution."""
    assert sum(RECORD_N19_COUNTS.values()) == 20 * 10240
    assert RECORD_N19_COUNTS["k_zero"] == 25007
    assert RECORD_N19_COUNTS["interior"] == 166690
    assert RECORD_N19_COUNTS["k_max"] == 13103


# ------------------------------------------------ amendment CS, CS3 --
# The pool geometry is DERIVED from the anchor list; until CS it was two
# module constants (20 / 19), which the width sweep (10 seeds, n_compare = 9)
# could not use. Expectations written down, not read from the code.

def test_pool_geometry_derives_n_compare_from_the_anchor_list():
    from src.analysis.rule_interior_mass import DEFAULT_ANCHORS, pool_geometry
    assert pool_geometry(list(range(10))) == (10, 9)
    assert pool_geometry(DEFAULT_ANCHORS) == (20, 19)
    assert pool_geometry([3, 7]) == (2, 1)


def test_pool_geometry_refuses_duplicates_and_singletons():
    from src.analysis.rule_interior_mass import pool_geometry
    with pytest.raises(SystemExit):
        pool_geometry([0, 0, 1])
    with pytest.raises(SystemExit):
        pool_geometry([4])


def test_record_controls_apply_only_to_the_records_twenty_anchors():
    from src.analysis.rule_interior_mass import record_controls_apply
    root = "$PROJECT/sae_ckpts"
    assert record_controls_apply(list(range(20)), root)
    assert record_controls_apply(list(range(20)), root + "/")
    assert not record_controls_apply(list(range(10)), root)
    assert not record_controls_apply(list(range(1, 21)), root)


def test_record_controls_do_not_apply_to_the_same_labels_under_another_root():
    """§CT28-pre item 8: labels 0-19 under sae_ckpts_k64 are k = 64
    dictionaries. The predicate before the fix tested labels only and would
    have gated C-B0 and C-B2 against the k = 32 record there."""
    from src.analysis.rule_interior_mass import DEFAULT_ANCHORS, record_controls_apply
    k64 = "$PROJECT/sae_ckpts_k64"
    old_predicate = lambda anchors: [int(a) for a in anchors] == DEFAULT_ANCHORS
    assert old_predicate(list(range(20)))            # the defect, written down
    assert not record_controls_apply(list(range(20)), k64)
    assert not record_controls_apply(list(range(20)), "/some/fixture/sae_ckpts")


def test_stored_pair_sims_threshold_back_to_the_counts():
    """§CT20: summing `sim >= theta` over the comparator axis must give the
    counts the driver accumulates, for every pair stored both ways, and the
    diagonal (an anchor against itself) must never count."""
    import numpy as np
    from src.analysis.rule_interior_mass import store_pair_sims
    rng = np.random.default_rng(2)
    n_a, n = 4, 30
    sim = {"r": np.full((n_a, n_a, n), -np.inf, dtype=np.float64)}
    counts = np.zeros((n_a, n), dtype=np.int32)
    for a in range(n_a):
        for b in range(a + 1, n_a):
            sa = rng.uniform(0, 1, n)
            sb = rng.uniform(0, 1, n)
            counts[a] += sa >= 0.7
            counts[b] += sb >= 0.7
            store_pair_sims(sim, "r", a, b, sa, sb)
    assert np.array_equal((sim["r"] >= 0.7).sum(axis=1), counts)
    assert np.all(np.isneginf(sim["r"][np.arange(n_a), np.arange(n_a)]))
