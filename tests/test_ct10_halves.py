"""Tests for src/analysis/ct10_halves.py (amendment CT, §CT10)."""
import numpy as np

from src.analysis.ct10_halves import assign_halves, balance_report


def test_no_cluster_is_split_between_halves():
    clus = ["c1", "c1", "c2", "c3", "c3", "c3", "c4"]
    rows = np.array([10, 20, 5, 1, 1, 1, 30])
    h = assign_halves(clus, rows, seed=1)
    for c in set(clus):
        assert len({int(h[i]) for i, x in enumerate(clus) if x == c}) == 1


def test_the_assignment_is_a_pure_function_of_its_inputs_and_seed():
    rng = np.random.default_rng(0)
    clus = [f"c{rng.integers(0, 300)}" for _ in range(2000)]
    rows = rng.integers(20, 900, size=2000)
    a = assign_halves(clus, rows, seed=20261027)
    b = assign_halves(list(clus), rows.copy(), seed=20261027)
    c = assign_halves(clus, rows, seed=20261028)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_greedy_balance_on_many_small_clusters_is_within_two_percent():
    rng = np.random.default_rng(1)
    clus = [f"c{rng.integers(0, 5000)}" for _ in range(20000)]
    rows = rng.integers(20, 1022, size=20000)
    rep = balance_report(assign_halves(clus, rows), rows)
    assert rep["within_tolerance"]
    assert rep["rows_a"] + rep["rows_b"] == int(rows.sum())


def test_balance_report_flags_an_unbalanced_split():
    rows = np.array([100, 1])
    rep = balance_report(np.array([0, 1]), rows)
    assert not rep["within_tolerance"]
    assert rep["abs_difference"] == 99


# ---- the row-set plumbing: feature_ap.load_scored_row_set's file branch -------

import h5py
import pytest


def _store(tmp_path, name="labels_toy.h5"):
    lab = tmp_path / name
    with h5py.File(lab, "w") as hf:
        g = hf.create_group("protein_sets")
        g.create_dataset("population_start", data=np.array([10, 40]))
        g.create_dataset("population_n_rows", data=np.array([5, 3]))
    return lab                                   # population = 10..14, 40..42


def _halves(tmp_path, lab, a, b, built=None):
    p = tmp_path / "halves.h5"
    with h5py.File(p, "w") as hf:
        hf.create_dataset("ct10_half_a_rows", data=np.array(a))
        hf.create_dataset("ct10_half_b_rows", data=np.array(b))
        hf.attrs["labels"] = str(built or lab)
    return p


def test_a_half_is_read_from_its_file(tmp_path):
    from src.analysis.feature_ap import load_scored_row_set
    lab = _store(tmp_path)
    h = _halves(tmp_path, lab, [10, 12, 41], [11, 13, 14, 40, 42])
    assert load_scored_row_set(lab, "ct10_half_a", h).tolist() == [10, 12, 41]
    assert load_scored_row_set(lab, "ct10_half_b", h).tolist() == [11, 13, 14, 40, 42]


def test_a_half_without_its_file_refuses(tmp_path):
    from src.analysis.feature_ap import load_scored_row_set
    with pytest.raises(SystemExit, match="needs --row-set-file"):
        load_scored_row_set(_store(tmp_path), "ct10_half_a")


def test_a_file_for_a_store_row_set_refuses(tmp_path):
    from src.analysis.feature_ap import load_scored_row_set
    lab = _store(tmp_path)
    h = _halves(tmp_path, lab, [10], [11])
    with pytest.raises(SystemExit, match="given for scored_row_set label_store"):
        load_scored_row_set(lab, "label_store", h)


def test_a_half_outside_the_population_refuses(tmp_path):
    from src.analysis.feature_ap import load_scored_row_set
    lab = _store(tmp_path)
    h = _halves(tmp_path, lab, [10, 20], [11])
    with pytest.raises(SystemExit, match="leaves the label_store population"):
        load_scored_row_set(lab, "ct10_half_a", h)


def test_a_half_built_against_another_store_refuses(tmp_path):
    from src.analysis.feature_ap import load_scored_row_set
    lab = _store(tmp_path)
    h = _halves(tmp_path, lab, [10], [11], built=tmp_path / "other_store.h5")
    with pytest.raises(SystemExit, match="was built against"):
        load_scored_row_set(lab, "ct10_half_a", h)


def test_the_store_row_sets_are_unchanged(tmp_path):
    from src.analysis.feature_ap import load_scored_row_set
    lab = _store(tmp_path)
    assert load_scored_row_set(lab, "label_store").tolist() == [10, 11, 12, 13, 14, 40, 41, 42]
    assert load_scored_row_set(lab, "full_eval") is None
