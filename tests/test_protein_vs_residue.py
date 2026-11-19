"""Tests for the BW protein-versus-residue decomposition.

Every expectation here is WRITTEN DOWN rather than computed from the module
under test (gotcha 32b): a test that derives its cases from the thing it checks
cannot detect changes to that thing.
"""
import numpy as np
import pytest

from src.analysis.protein_vs_residue import (
    CELL_NAMES,
    concept_level_two_by_two,
    ignorance_partition,
    threshold_relation,
    two_by_two_cells,
)


def test_all_four_cells_are_named_and_distinct():
    """The 2x2's cell names are the contract between artifact, JSON and prose."""
    assert set(CELL_NAMES) == {(True, True), (True, False),
                               (False, True), (False, False)}
    assert len(set(CELL_NAMES.values())) == 4
    # Written down, not read from the module's own dict at assertion time.
    assert CELL_NAMES[(True, False)] == "protein_level_candidate"
    assert CELL_NAMES[(True, True)] == "residue_level_candidate"
    assert CELL_NAMES[(False, True)] == "within_only"
    assert CELL_NAMES[(False, False)] == "not_detected_on_this_axis"


def test_two_by_two_counts_are_exhaustive_and_include_zero_cells():
    """A cell with no members is REPORTED AS ZERO, never omitted -- an absent
    category is indistinguishable from one that was never checked."""
    across = np.array([[True, True, False, False]])
    within = np.array([[True, False, True, False]])
    got = two_by_two_cells(across, within)
    assert got["cells"] == {"residue_level_candidate": 1,
                            "protein_level_candidate": 1,
                            "within_only": 1,
                            "not_detected_on_this_axis": 1}
    assert got["n_cells_total"] == 4

    # Now a case where two cells are genuinely empty. They must still appear.
    across = np.array([[True, True]])
    within = np.array([[True, True]])
    got = two_by_two_cells(across, within)
    assert got["cells"]["residue_level_candidate"] == 2
    assert got["cells"]["protein_level_candidate"] == 0
    assert got["cells"]["within_only"] == 0
    assert got["cells"]["not_detected_on_this_axis"] == 0
    assert sum(got["cells"].values()) == 2


def test_two_by_two_rejects_non_boolean_and_mismatched_shapes():
    with pytest.raises(ValueError):
        two_by_two_cells(np.array([[1, 0]]), np.array([[True, False]]))
    with pytest.raises(ValueError):
        two_by_two_cells(np.array([[True]]), np.array([[True, False]]))


def test_threshold_relation_maps_the_within_only_cell_in_advance():
    """`within_only` needs `thr_within < thr_across`. Where it does not hold the
    cell is arithmetically unreachable, and that is a DEDUCTION about the
    outcome space, not a finding about the data."""
    thr_within = np.array([0.10, 0.02, 0.05, 0.05])
    thr_across = np.array([0.01, 0.09, 0.05, 0.06])
    got = threshold_relation(thr_within, thr_across)
    # Written down: index 1 (0.02 < 0.09) and index 3 (0.05 < 0.06) are eligible.
    assert got["n_within_below_across"] == 2
    assert got["n_within_at_or_above_across"] == 2
    assert got["within_only_cell_reachable_for_n_concepts"] == 2
    assert got["eligible_mask"].tolist() == [False, True, False, True]
    # Equality is NOT eligible: `gain > thr_within` and `gain <= thr_across`
    # cannot both hold when the two are equal.
    assert got["eligible_mask"][2] == False  # noqa: E712


def test_threshold_relation_is_arithmetically_sound_against_detection():
    """THE RULE MUST FIRE ON ITS OWN FOUNDING CLAIM (gotcha 39). If a concept is
    ineligible, no observed gain whatsoever can put it in `within_only`."""
    rng = np.random.default_rng(0)
    thr_within = np.array([0.30, 0.30, 0.30])
    thr_across = np.array([0.10, 0.20, 0.29])   # all ineligible
    rel = threshold_relation(thr_within, thr_across)
    assert rel["n_within_below_across"] == 0
    for _ in range(200):
        gain = rng.uniform(0, 1, size=(5, 3))
        det_w = gain > thr_within
        det_a = gain > thr_across
        assert not (det_w & ~det_a).any(), (
            "a within-only cell appeared where thr_within >= thr_across")


def test_ignorance_categories_are_counted_separately_not_folded():
    """Each ignorance case is its own count. None may be silently merged into a
    denominator and none is a negative finding."""
    best_gain = np.array([[0.5, np.nan, 0.2]])
    thr_within = np.array([0.1, 0.2, 0.0])
    thr_across = np.array([0.1, np.inf, 0.3])
    null_within = np.zeros((2, 2, 3))
    null_across = np.zeros((2, 2, 3))
    null_across[..., 0] = 0.4          # only concept 0 has non-zero draws
    degenerate = np.array([True, False, False])

    got = ignorance_partition(best_gain, thr_within, thr_across,
                              null_within, null_across, degenerate)
    # Written down, each independently:
    assert got["n_cells_nonfinite_observed_gain"] == 1
    assert got["n_concepts_nonfinite_threshold"] == 1
    assert got["n_concepts_zero_threshold_within"] == 1
    assert got["n_concepts_zero_threshold_across"] == 0
    assert got["n_concepts_all_null_draws_zero_within"] == 3
    assert got["n_concepts_all_null_draws_zero_across"] == 2
    assert got["n_concepts_flagged_degenerate"] == 1


def test_concept_level_reduction_is_D_equals_one_and_joint_is_reported():
    """The seed-collapsing rule is the record's OWN (`D(c) == 1`), and the full
    joint distribution is reported beside it so nothing hides behind it."""
    d_across = np.array([1.0, 1.0, 0.5, 0.0])
    d_within = np.array([1.0, 0.5, 1.0, 0.0])
    got = concept_level_two_by_two(d_across, d_within)
    assert got["cells"]["residue_level_candidate"] == 1   # index 0
    assert got["cells"]["protein_level_candidate"] == 1   # index 1
    assert got["cells"]["within_only"] == 1               # index 2
    assert got["cells"]["not_detected_on_this_axis"] == 1  # index 3
    assert got["n_concepts"] == 4
    joint = got["joint_distribution_D_across_given_D_within"]
    assert joint["1.00|1.00"] == 1
    assert joint["1.00|0.50"] == 1
    assert sum(joint.values()) == 4


def test_substituting_the_arm_changes_the_answer_the_fixture_discriminates():
    """THE FIXTURE MUST DISCRIMINATE (gotcha 13). If the two nulls produced the
    same classification on this fixture, the whole measure would be untestable
    and a bug that silently read one arm twice would pass."""
    # A concept whose within-protein null is HIGH and across-protein null LOW:
    # the shape amendment G4 attributes to a protein-selecting feature.
    gain = np.array([[0.50]])
    thr_high_within = np.array([0.90])
    thr_low_across = np.array([0.10])
    det_a = gain > thr_low_across
    det_w = gain > thr_high_within
    got = two_by_two_cells(det_a, det_w)
    assert got["cells"]["protein_level_candidate"] == 1
    # And reading the SAME arm twice would give a different, wrong answer.
    same = two_by_two_cells(det_a, det_a)
    assert same["cells"]["protein_level_candidate"] == 0
    assert same["cells"]["residue_level_candidate"] == 1
    assert got["cells"] != same["cells"], (
        "the fixture does not discriminate between the two arms")


# --------------------------------------------------------------------------- #
# Construction 2 -- the feature-level view (BW C2)
# --------------------------------------------------------------------------- #

def test_ratio_to_null_mean_uses_the_mean_and_reports_its_standard_error():
    """Written down, not computed from the module: draws 1,2,3,4 have mean 2.5,
    sample SD (ddof=1) sqrt(5/3)=1.29099, so SE = 1.29099/2 = 0.645497 and the
    relative SE = 0.645497/2.5 = 0.2581989."""
    from src.analysis.protein_vs_residue_feature_view import ratio_to_null_mean
    null = np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1, 1)
    obs = np.array([[5.0]])
    ratio, rel_se = ratio_to_null_mean(obs, null)
    assert ratio[0, 0] == pytest.approx(2.0)          # 5.0 / 2.5
    assert rel_se[0, 0] == pytest.approx(0.2581989, rel=1e-6)


def test_ratio_is_nan_where_the_null_mean_is_zero_never_imputed():
    """A pair with a zero null mean has NO defined ratio. It must come back NaN
    so the caller counts it, rather than silently becoming 0 or inf."""
    from src.analysis.protein_vs_residue_feature_view import ratio_to_null_mean
    null = np.zeros((3, 1, 2))
    null[:, 0, 1] = [1.0, 1.0, 1.0]
    obs = np.array([[7.0, 7.0]])
    ratio, rel_se = ratio_to_null_mean(obs, null)
    assert np.isnan(ratio[0, 0]) and np.isnan(rel_se[0, 0])
    assert ratio[0, 1] == pytest.approx(7.0)


def test_summarise_reports_nonfinite_rather_than_folding_them_away():
    from src.analysis.protein_vs_residue_feature_view import summarise
    got = summarise(np.array([1.0, 2.0, 3.0, np.nan, np.inf]))
    assert got["n_total"] == 5
    assert got["n_finite"] == 3
    assert got["n_nonfinite"] == 2
    assert got["quantiles"]["p50"] == pytest.approx(2.0)


def test_ratio_to_null_mean_rejects_mismatched_shapes():
    from src.analysis.protein_vs_residue_feature_view import ratio_to_null_mean
    with pytest.raises(ValueError):
        ratio_to_null_mean(np.zeros((2, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError):
        ratio_to_null_mean(np.zeros((2, 2)), np.zeros((3, 5, 5)))
