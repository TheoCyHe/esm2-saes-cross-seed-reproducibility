"""Tests for `decomposition_reporting` -- amendment BX.

EVERY EXPECTATION IN THIS FILE IS WRITTEN DOWN, not computed from the module
under test (gotcha 32b). The fixtures are small enough to count by hand, and the
hand count is what appears in the assertion.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from src.analysis.decomposition_reporting import (
    WITHIN_ONLY_MEASURED,
    WITHIN_ONLY_STRUCTURAL,
    cell_grid,
    common_basis_diff,
    concept_grid,
    concept_to_cell_bridge,
    inference_check,
    load,
    matched_rate_comparison,
    rate_grid,
    two_by_two,
)


# --------------------------------------------------------------------------- #
# A FIXTURE THAT DISCRIMINATES. Four seeds, five concepts, chosen so that:
#   * concept 0 is across-detected in all 4 and within-detected in all 4;
#   * concept 1 is across-detected in all 4 and within-detected in 2 -- so the
#     ANY-over-seeds and RATE-over-seeds counts DIFFER, which is the whole point
#     of the bridge;
#   * concept 2 is across-detected in 2 and within-detected in 0;
#   * concept 3 is across-detected in 0 and within-detected in 3 -- an OCCUPIED
#     `within_only` cell, so a zero there would be a real failure, not a
#     vacuous pass;
#   * concept 4 is detected by neither arm anywhere.
# --------------------------------------------------------------------------- #
DET_ACROSS = np.array([
    [True,  True,  True,  False, False],
    [True,  True,  True,  False, False],
    [True,  True,  False, False, False],
    [True,  True,  False, False, False],
])
DET_WITHIN = np.array([
    [True,  True,  False, True,  False],
    [True,  True,  False, True,  False],
    [True,  False, False, True,  False],
    [True,  False, False, False, False],
])
D_ACROSS = np.array([1.00, 1.00, 0.50, 0.00, 0.00])
D_WITHIN = np.array([1.00, 0.50, 0.00, 0.75, 0.00])
N_SEEDS = 4


def test_fixture_is_self_consistent():
    """The fixture's hand-written rates must match its hand-written matrices --
    otherwise every downstream expectation is anchored to nothing."""
    assert DET_ACROSS.mean(axis=0).tolist() == D_ACROSS.tolist()
    assert DET_WITHIN.mean(axis=0).tolist() == D_WITHIN.tolist()


def _write_artifact(path, det_a, det_w, d_a, d_w, confirmatory=False):
    with h5py.File(path, "w") as hf:
        st = h5py.special_dtype(vlen=str)
        hf.create_dataset("detected_across_protein", data=det_a)
        hf.create_dataset("detected_within_protein", data=det_w)
        hf.create_dataset("D_across_protein", data=d_a)
        hf.create_dataset("D_within_protein", data=d_w)
        hf.create_dataset(
            "concepts",
            data=np.array([f"C{i}" for i in range(det_a.shape[1])], dtype=object),
            dtype=st)
        hf.create_dataset(
            "seeds",
            data=np.array([f"seed_{i}" for i in range(det_a.shape[0])],
                          dtype=object),
            dtype=st)
        hf.attrs["confirmatory"] = confirmatory


def test_load_accepts_a_consistent_artifact(tmp_path):
    """THE POSITIVE CONTROL for the two guards below. Without it a refusal
    proves only that the loader refuses, not that it discriminates."""
    p = tmp_path / "ok.h5"
    _write_artifact(p, DET_ACROSS, DET_WITHIN, D_ACROSS, D_WITHIN)
    got = load(p)
    assert got["d_across"].tolist() == D_ACROSS.tolist()
    assert got["seeds"] == [f"seed_{i}" for i in range(4)]


def test_load_refuses_when_a_stored_rate_disagrees_with_its_matrix(tmp_path):
    """The two are stored separately, so they CAN disagree -- which happens if
    they were written from different runs. The guard must compare each arm
    against ITS OWN matrix; comparing an arm against itself can only pass."""
    p = tmp_path / "bad.h5"
    corrupt = D_WITHIN.copy()
    corrupt[1] = 0.25          # stored says 0.25, the matrix says 0.50
    _write_artifact(p, DET_ACROSS, DET_WITHIN, D_ACROSS, corrupt)
    with pytest.raises(SystemExit, match="D_within_protein"):
        load(p)


def test_load_refuses_an_artifact_claiming_to_be_confirmatory(tmp_path):
    p = tmp_path / "conf.h5"
    _write_artifact(p, DET_ACROSS, DET_WITHIN, D_ACROSS, D_WITHIN,
                    confirmatory=True)
    with pytest.raises(SystemExit, match="confirmatory"):
        load(p)


def test_load_refuses_an_artifact_missing_a_required_dataset(tmp_path):
    p = tmp_path / "short.h5"
    _write_artifact(p, DET_ACROSS, DET_WITHIN, D_ACROSS, D_WITHIN)
    with h5py.File(p, "a") as hf:
        del hf["detected_within_protein"]
    with pytest.raises(SystemExit, match="detected_within_protein"):
        load(p)


def test_rate_grid_excludes_zero_and_includes_one():
    g = rate_grid(4)
    assert g.tolist() == [0.25, 0.50, 0.75, 1.00]
    with pytest.raises(ValueError):
        rate_grid(0)


def test_two_by_two_counts_by_hand():
    a = np.array([True, True, False, False])
    w = np.array([True, False, True, False])
    assert two_by_two(a, w) == {
        "residue_level_candidate": 1,
        "protein_level_candidate": 1,
        "within_only": 1,
        "not_detected_on_this_axis": 1,
    }


def test_two_by_two_rejects_non_boolean():
    with pytest.raises(ValueError):
        two_by_two(np.array([1, 0]), np.array([True, False]))


def test_two_by_two_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        two_by_two(np.array([True]), np.array([True, False]))


def test_concept_grid_at_t_equals_one_all_concepts():
    """Counted by hand at `D >= 1.00` over all five concepts:
      across-detected: concepts 0, 1        within-detected: concept 0
      -> residue 1 (c0), protein 1 (c1), within_only 0, none 3 (c2, c3, c4)
    """
    rows = [r for r in concept_grid(D_ACROSS, D_WITHIN, N_SEEDS)
            if r["population"] == "all_concepts" and r["t"] == 1.0]
    assert len(rows) == 1
    assert rows[0]["cells"] == {
        "residue_level_candidate": 1,
        "protein_level_candidate": 1,
        "within_only": 0,
        "not_detected_on_this_axis": 3,
    }
    assert rows[0]["denominator"] == 5
    assert rows[0]["share_within_fails_of_across_detected"] == pytest.approx(0.5)


def test_concept_grid_at_t_one_quarter_has_an_occupied_within_only():
    """At `D >= 0.25` over all five concepts:
      across-detected: 0, 1, 2      within-detected: 0, 1, 3
      -> residue 2 (c0, c1), protein 1 (c2), within_only 1 (c3), none 1 (c4)
    An OCCUPIED `within_only` here is what makes the zero at t = 1 meaningful.
    """
    rows = [r for r in concept_grid(D_ACROSS, D_WITHIN, N_SEEDS)
            if r["population"] == "all_concepts" and r["t"] == 0.25]
    assert rows[0]["cells"] == {
        "residue_level_candidate": 2,
        "protein_level_candidate": 1,
        "within_only": 1,
        "not_detected_on_this_axis": 1,
    }
    assert rows[0]["within_only_zero_kind"] is None


def test_unrestricted_zero_within_only_is_labelled_measured():
    rows = [r for r in concept_grid(D_ACROSS, D_WITHIN, N_SEEDS)
            if r["population"] == "all_concepts" and r["t"] == 1.0]
    assert rows[0]["cells"]["within_only"] == 0
    assert rows[0]["within_only_zero_kind"] == WITHIN_ONLY_MEASURED


def test_restricted_population_zero_within_only_is_labelled_structural():
    """On `across_D_eq_1` every member satisfies the across predicate at every
    `t`, so `within_only` is empty by construction and must say so."""
    rows = [r for r in concept_grid(D_ACROSS, D_WITHIN, N_SEEDS)
            if r["population"] == "across_D_eq_1"]
    assert rows, "the restricted population produced no rows"
    for r in rows:
        assert r["cells"]["within_only"] == 0
        assert r["within_only_zero_kind"] == WITHIN_ONLY_STRUCTURAL


def test_concept_grid_covers_every_population_and_rate():
    rows = concept_grid(D_ACROSS, D_WITHIN, N_SEEDS)
    assert len(rows) == 3 * 4
    assert {r["population"] for r in rows} == {
        "all_concepts", "across_D_eq_1", "across_D_ge_t"}
    assert sorted({r["t"] for r in rows}) == [0.25, 0.50, 0.75, 1.00]


def test_concept_grid_restricted_population_shares_by_hand():
    """`across_D_eq_1` is concepts 0 and 1. At `t = 1.00` within-detected is
    concept 0 only -> 1 residue, 1 protein, share 0.5. At `t = 0.50` both are
    within-detected -> 2 residue, 0 protein, share 0.0."""
    rows = {r["t"]: r for r in concept_grid(D_ACROSS, D_WITHIN, N_SEEDS)
            if r["population"] == "across_D_eq_1"}
    assert rows[1.00]["denominator"] == 2
    assert rows[1.00]["cells"]["residue_level_candidate"] == 1
    assert rows[1.00]["cells"]["protein_level_candidate"] == 1
    assert rows[0.50]["cells"]["residue_level_candidate"] == 2
    assert rows[0.50]["cells"]["protein_level_candidate"] == 0


def test_matched_rate_comparison_by_hand():
    """`D_within >= D_across` holds for concepts 0 (1.0>=1.0), 3 (0.75>=0.0)
    and 4 (0.0>=0.0) -> 3. Strictly greater for concept 3 alone -> 1."""
    m = matched_rate_comparison(D_ACROSS, D_WITHIN)
    assert m["all_concepts"]["denominator"] == 5
    assert m["all_concepts"]["n_within_ge_across"] == 3
    assert m["all_concepts"]["n_within_lt_across"] == 2
    assert m["all_concepts"]["n_within_gt_across"] == 1
    assert m["all_concepts"]["within_only_analogue"]["count"] == 1
    # Restricted to D_across == 1 the criterion collapses onto D_within == 1,
    # which concept 0 satisfies and concept 1 does not.
    assert m["across_D_eq_1"]["denominator"] == 2
    assert m["across_D_eq_1"]["n_within_ge_across"] == 1
    assert m["across_D_eq_1"]["n_within_gt_across"] == 0


def test_cell_grid_all_cells_by_hand():
    """20 cells, counted from the matrices:
      residue     = 6  -- c0 x 4, c1 x 2 (seeds 0-1)
      protein     = 4  -- c1 x 2 (seeds 2-3), c2 x 2 (seeds 0-1)
      within_only = 3  -- c3, seeds 0-2
      none        = 7  -- the remainder, and it is asserted as a sum below
    """
    rows = {r["denominator_name"]: r for r in
            cell_grid(DET_ACROSS, DET_WITHIN, D_ACROSS)}
    c = rows["all_cells"]["cells"]
    assert rows["all_cells"]["denominator"] == 20
    assert c["residue_level_candidate"] == 6
    assert c["protein_level_candidate"] == 4
    assert c["within_only"] == 3
    assert c["not_detected_on_this_axis"] == 7
    assert sum(c.values()) == 20


def test_cell_grid_across_detected_basis_by_hand():
    """Across-detected cells: concept 0 x4, concept 1 x4, concept 2 x2 = 10.
    Within-detected among them: c0 x4, c1 x2 = 6. So 4 fail, share 0.4."""
    rows = {r["denominator_name"]: r for r in
            cell_grid(DET_ACROSS, DET_WITHIN, D_ACROSS)}
    r = rows["across_detected_cells"]
    assert r["denominator"] == 10
    assert r["cells"]["residue_level_candidate"] == 6
    assert r["cells"]["protein_level_candidate"] == 4
    assert r["share_within_fails_of_across_detected"] == pytest.approx(0.4)
    assert r["within_only_zero_kind"] == WITHIN_ONLY_STRUCTURAL


def test_cell_grid_two_restricted_bases_coincide():
    """`D_across == 1` means every one of that concept's cells is
    across-detected, so restricting further by across-detection changes
    nothing. Asserted rather than left as a claim in the prose."""
    rows = {r["denominator_name"]: r for r in
            cell_grid(DET_ACROSS, DET_WITHIN, D_ACROSS)}
    a = rows["cells_of_concepts_with_across_D_eq_1"]
    b = rows["across_detected_cells_of_concepts_with_across_D_eq_1"]
    assert a["denominator"] == b["denominator"] == 8
    assert a["cells"] == b["cells"]


def test_cell_grid_rejects_wrong_d_across_length():
    with pytest.raises(ValueError):
        cell_grid(DET_ACROSS, DET_WITHIN, D_ACROSS[:3])


def test_bridge_separates_unit_from_population():
    """On the D_across == 1 population (concepts 0 and 1): 8 across-detected
    cells, 2 of which fail within (concept 1, seeds 2 and 3). One of the two
    concepts has at least one failure.
    So the concept share is 1/2 = 0.5 and the cell share is 2/8 = 0.25 -- the
    unit alone moves the figure by 2x on ONE population."""
    br = concept_to_cell_bridge(DET_ACROSS, DET_WITHIN, D_ACROSS, D_WITHIN)
    s1 = br["step_1_unit_on_one_shared_population"]
    assert s1["n_concepts"] == 2
    assert s1["n_concepts_with_at_least_one_within_failure"] == 1
    assert s1["concept_share"] == pytest.approx(0.5)
    assert s1["n_across_detected_cells"] == 8
    assert s1["n_within_failing_cells"] == 2
    assert s1["cell_share"] == pytest.approx(0.25)
    assert s1["mean_failing_seeds_per_failing_concept"] == pytest.approx(2.0)

    s2 = br["step_2_population_on_one_shared_unit"]
    assert s2["all_across_detected_cells"] == 10
    assert s2["within_failing"] == 4
    assert s2["of_which_from_concepts_with_across_D_eq_1"] == 2
    assert s2["of_which_from_other_concepts"] == 2
    assert s2["other_concepts_denominator"] == 2
    assert s2["other_concepts_share"] == pytest.approx(1.0)


def test_common_basis_diff_reports_every_band_including_empty():
    d = common_basis_diff(DET_ACROSS, DET_WITHIN, D_ACROSS, D_WITHIN, N_SEEDS)
    assert d["n_bands_total"] == 5          # 0, 1/4, 2/4, 3/4, 4/4
    # Occupied by across-detected cells: D_across = 0.50 and 1.00 only.
    assert d["n_bands_empty"] == 3
    by_rate = {b["D_across"]: b for b in d["bands"]}
    assert by_rate[0.75]["n_concepts"] == 0
    assert by_rate[0.75]["within_failure_rate_of_across_detected_cells"] is None
    assert by_rate[0.50]["n_across_detected_cells"] == 2
    assert by_rate[0.50]["within_failure_rate_of_across_detected_cells"] == 1.0
    assert by_rate[1.00]["n_across_detected_cells"] == 8
    assert by_rate[1.00]["within_failure_rate_of_across_detected_cells"] == 0.25


def test_inference_check_refutes_on_this_fixture():
    """The fixture's top band (D_across = 1.00) has rate 0.25 against the only
    other occupied band's 1.00, so the top band is at or below the minimum of
    the others -> REFUTES. Written down, not read off the module."""
    d = common_basis_diff(DET_ACROSS, DET_WITHIN, D_ACROSS, D_WITHIN, N_SEEDS)
    assert inference_check(d)["verdict"] == "REFUTES"


def test_inference_check_supports_when_the_shape_is_reversed():
    """A CONTROL: the verdict must be capable of coming out the other way, or
    REFUTES above proves nothing (gotcha 13). Swapping the two arms' failure
    pattern puts the higher rate in the top band."""
    det_a = np.array([[True, True], [True, True], [True, False], [True, False]])
    det_w = np.array([[True, True], [False, True], [False, False],
                      [False, False]])
    d_a = det_a.mean(axis=0)
    d_w = det_w.mean(axis=0)
    d = common_basis_diff(det_a, det_w, d_a, d_w, 4)
    # concept 0: D_across 1.00, 4 across-detected cells, 3 fail -> 0.75
    # concept 1: D_across 0.50, 2 across-detected cells, 0 fail -> 0.00
    assert inference_check(d)["verdict"] == "SUPPORTS"


def test_inference_check_uninformative_with_one_occupied_band():
    det_a = np.array([[True], [True]])
    det_w = np.array([[True], [False]])
    d = common_basis_diff(det_a, det_w, det_a.mean(axis=0), det_w.mean(axis=0), 2)
    assert inference_check(d)["verdict"] == "UNINFORMATIVE"


def test_inference_check_neither_when_top_sits_inside_the_range():
    """Three occupied bands with the top band's rate strictly between the other
    two -> NEITHER. The fourth branch has to be reachable too."""
    det_a = np.array([
        [True, True, True, True],
        [True, True, True, False],
        [True, True, False, False],
        [True, False, False, False],
    ])
    #  concept 0: D_across 1.00 -> 4 cells; make 2 fail  -> 0.50
    #  concept 1: D_across 0.75 -> 3 cells; make 3 fail  -> 1.00
    #  concept 2: D_across 0.50 -> 2 cells; make 0 fail  -> 0.00
    #  concept 3: D_across 0.25 -> 1 cell;  make 0 fail  -> 0.00  (ties min)
    det_w = np.array([
        [True,  False, True,  True],
        [True,  False, True,  False],
        [False, False, False, False],
        [False, False, False, False],
    ])
    d = common_basis_diff(det_a, det_w, det_a.mean(axis=0), det_w.mean(axis=0), 4)
    by_rate = {b["D_across"]: b["within_failure_rate_of_across_detected_cells"]
               for b in d["bands"]}
    assert by_rate[1.00] == pytest.approx(0.50)
    assert by_rate[0.75] == pytest.approx(1.00)
    assert by_rate[0.50] == pytest.approx(0.00)
    assert inference_check(d)["verdict"] == "NEITHER"
