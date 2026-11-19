"""Amendment CT §CT5 / §CT12.1: the stratum's M is aligned to the anchors by its
stored labels (gotcha 29) and a non-positive M refuses rather than drops."""
import h5py
import numpy as np
import pytest

from src.analysis import stratum_slope as ss


def _write(path, labels, M):
    with h5py.File(path, "w") as hf:
        hf.create_dataset("seeds", data=[s.encode() for s in labels],
                          dtype=h5py.string_dtype())
        hf.create_dataset("M", data=M)
        hf.attrs["scored_row_set"] = "shared_experimental"


def test_aligned_logM_reorders_a_lexicographic_axis(tmp_path):
    lex = ["seed_0", "seed_1", "seed_10", "seed_2"]
    M = np.array([[1.0], [2.0], [11.0], [3.0]])          # row value = seed + 1
    _write(tmp_path / "g.h5", lex, M)
    logM, rep = ss.aligned_logM(tmp_path / "g.h5", ["0", "1", "2", "10"])
    np.testing.assert_allclose(np.exp(logM[:, 0]), [1.0, 2.0, 3.0, 11.0])
    assert rep["n_nonpositive_or_nonfinite_M"] == 0


def test_aligned_logM_refuses_nonpositive_M(tmp_path):
    _write(tmp_path / "g.h5", ["seed_0", "seed_1"], np.array([[1.0], [-0.5]]))
    with pytest.raises(SystemExit, match="UNMEASURABLE: 1"):
        ss.aligned_logM(tmp_path / "g.h5", ["0", "1"])


def test_aligned_logM_refuses_an_unlabelled_axis(tmp_path):
    with h5py.File(tmp_path / "g.h5", "w") as hf:
        hf.create_dataset("M", data=np.ones((2, 1)))
    with pytest.raises(SystemExit, match="no seed labels"):
        ss.aligned_logM(tmp_path / "g.h5", ["0", "1"])


def test_stratum_description_counts_the_scored_pool(tmp_path):
    """§CT5.1's descriptive counts: positives summed over the ADMITTED pool only,
    refusals counted on the pool before admission, cell occupancy on the pool."""
    p = tmp_path / "g.h5"
    with h5py.File(p, "w") as hf:
        hf.create_dataset("in_argmax_pool", data=np.array([False, True, True]))
        hf.create_dataset("concept_cell", data=np.array([0, 3, 3]))
        hf.attrs["admission_min_protein_count"] = 5
        g = hf.create_group("stratum_counts")
        g.create_dataset("in_argmax_pool", data=np.array([False, True, True, True]))
        g.create_dataset("admitted", data=np.array([True, True, False, True]))
        g.create_dataset("experimental/protein_count", data=np.array([9, 6, 2, 5]))
        g.create_dataset("experimental/n_positive_rows", data=np.array([100, 10, 7, 20]))
    d = ss.stratum_description(p)
    e = d["per_stratum_on_scored_pool"]["experimental"]
    assert e == {"n_concepts": 2, "n_positive_labels": 30,
                 "n_pool_concepts_below_gate_before_admission": 1}
    assert d["n_pool_concepts_scored"] == 2
    assert d["n_pool_concepts_before_admission"] == 3
    assert d["concept_cells_occupied"] == 1
    assert d["pool_concepts_in_cells_below_K1_minimum"] == 2
