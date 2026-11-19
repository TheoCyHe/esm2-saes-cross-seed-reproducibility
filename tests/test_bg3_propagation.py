"""
Tests for AC7's BG3 propagation check.

The expected values are worked out by hand from the definitions, and the
formatter gets its own total-branch test: a completed measurement has already
been reported as a FAILED job in this project because a summary print died on
a legitimately undefined value (BA17 item 3).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.bg3_propagation import (
    DegenerateDrawNotExcluded,
    ExclusionDerivationsDisagree,
    anti_conservative_bound,
    bh3_outcomes,
    bound_factor,
    exclusion_derivations_agree,
    format_summary,
    relative_positive_loss,
    surviving_draw_mask,
    worst_cells,
)


def _all(loss):
    """An all-True keep mask. Used only where the fixture has no degenerate
    cell, so that the mask argument is explicit at every call site."""
    return np.ones(np.asarray(loss).shape, dtype=bool)


def test_relative_loss_is_the_fraction_of_positives_lost():
    """Concept 0 observed 10 positives and kept 10, 9, 5 across three draws;
    concept 1 observed 4 and kept 4, 4, 3."""
    surv = np.array([[10, 9, 5], [4, 4, 3]])
    obs = np.array([10, 4])
    loss = relative_positive_loss(surv, obs)
    assert loss[0].tolist() == pytest.approx([0.0, -0.1, -0.5])
    assert loss[1].tolist() == pytest.approx([0.0, 0.0, -0.25])


def test_relative_loss_does_NOT_depend_on_n_eff():
    """The module's whole argument for reusing the pooled functional form is
    that `n_eff` cancels: under BG1's fixed mask the relative prevalence drift
    equals the relative positive-count loss. This asserts the cancellation
    rather than restating it in prose."""
    surv = np.array([[8, 6]])
    obs = np.array([8])
    loss = relative_positive_loss(surv, obs)
    for n_eff in (100, 15_972_325, 15_970_339):
        p_obs = obs[0] / n_eff
        p_draw = surv[0] / n_eff
        assert (p_draw / p_obs - 1.0).tolist() == pytest.approx(loss[0].tolist())


def test_the_bound_is_one_over_one_minus_the_worst_loss():
    """Worst loss -0.5 -> 1/(1-0.5) = 2.0; median loss -0.1 -> 1/0.9 = 1.111..."""
    loss = np.array([[0.0, -0.1, -0.5]])
    b = anti_conservative_bound(loss, _all(loss))
    assert b["worst_relative_loss"] == pytest.approx(-0.5)
    assert b["bound_at_worst"] == pytest.approx(2.0)
    assert b["median_relative_loss"] == pytest.approx(-0.1)
    assert b["bound_at_median"] == pytest.approx(1.0 / 0.9)


def test_no_loss_anywhere_gives_a_bound_of_exactly_one():
    loss = np.zeros((3, 4))
    b = anti_conservative_bound(loss, _all(loss))
    assert b["bound_at_worst"] == pytest.approx(1.0)
    assert b["frac_cells_with_no_loss"] == pytest.approx(1.0)


# ---------------------------------------------------------------------- #
# BH2's exclusion. The mask is the fix; these are its tests.
# ---------------------------------------------------------------------- #

def test_the_surviving_draw_mask_IS_BH1s_criterion_and_names_no_cell():
    """Zero surviving positives, and nothing else. A count of 1 survives --
    BH3's fourth case says AP is defined there and no rule applies."""
    surv = np.array([[5, 0, 1], [0, 0, 3]])
    assert surviving_draw_mask(surv).tolist() == [[True, False, True],
                                                  [False, False, True]]


def test_the_bound_factor_is_TOTAL_including_at_a_total_loss():
    """A loss of exactly -1 makes the factor infinite: reported, not divided
    (gotcha 24). `anti_conservative_bound` now refuses to be handed such a
    value, and that guard is NOT a reason to let the branch go untested -- so
    this exercises `bound_factor` directly rather than through the guard."""
    assert bound_factor(-1.0) == float("inf")
    assert bound_factor(-1.5) == float("inf")
    assert bound_factor(-0.5) == pytest.approx(2.0)
    assert bound_factor(0.0) == pytest.approx(1.0)


def test_MASKING_CHANGES_THE_BOUND__the_fixture_discriminates():
    """AC1-4. Before asserting anything about the real artifact, show that the
    mask is capable of changing the answer.

    Concept 0 observed 4 positives and kept 4, 3 and then ZERO -- a degenerate
    draw, which BH2 drops. UNMASKED the -1.0 cell drives the worst case;
    MASKED the worst surviving loss is -0.25 and the bound is 4/3.
    """
    surv = np.array([[4, 3, 0]])
    obs = np.array([4])
    loss = relative_positive_loss(surv, obs)
    keep = surviving_draw_mask(surv)

    # the unmasked reading, computed here rather than taken from the code
    # under test: the degenerate cell is a loss of exactly -1.
    assert loss[0, 2] == pytest.approx(-1.0)
    assert bound_factor(float(loss.min())) == float("inf")

    masked = anti_conservative_bound(loss, keep)
    assert masked["worst_relative_loss"] == pytest.approx(-0.25)
    assert masked["bound_at_worst"] == pytest.approx(4.0 / 3.0)
    assert masked["n_cells"] == 2
    assert masked["n_cells_excluded_degenerate"] == 1


def test_an_UNMASKED_degenerate_cell_RAISES_rather_than_reporting_infinity():
    """The invariant used to live in a docstring saying such a cell 'should
    never reach here in practice'. Nothing enforced it, and it reached here."""
    loss = np.array([[0.0, -1.0]])
    with pytest.raises(DegenerateDrawNotExcluded) as e:
        anti_conservative_bound(loss, _all(loss))
    assert "-1" in str(e.value)


def test_a_concept_with_no_surviving_draw_is_NOT_EVALUABLE_not_a_zero():
    """BH3 case 2. Concept 1 loses every draw. It must leave the per-concept
    percentiles entirely -- not enter them as a zero, which would drag the
    median of the worst-per-concept losses toward no-loss and read as
    'checked and unaffected'."""
    surv = np.array([[4, 3], [0, 0], [8, 8]])
    obs = np.array([4, 5, 8])
    loss = relative_positive_loss(surv, obs)
    b = anti_conservative_bound(loss, surviving_draw_mask(surv))
    assert b["status"] == "COMPUTED"
    assert b["n_concepts_not_evaluable"] == 1
    assert b["n_concepts_in_percentiles"] == 2
    # the two evaluable concepts' worst losses are -0.25 and 0.0
    assert b["worst_per_concept_loss_percentiles"]["p00"] == pytest.approx(-0.25)
    assert b["worst_per_concept_loss_percentiles"]["p100"] == pytest.approx(0.0)


def test_an_arm_with_no_surviving_draw_ANYWHERE_is_NOT_EVALUABLE():
    """BH3 case 3. Not zero, not infinity, and not a division over an empty
    set -- and no `bound_at_worst` key at all, so a consumer cannot read a
    value out of a statement of ignorance."""
    surv = np.zeros((2, 3), dtype=int)
    obs = np.array([5, 7])
    loss = relative_positive_loss(surv, obs)
    b = anti_conservative_bound(loss, surviving_draw_mask(surv))
    assert b["status"] == "NOT_EVALUABLE"
    assert b["n_concepts_not_evaluable"] == 2
    assert "bound_at_worst" not in b


def test_an_all_nonfinite_loss_is_a_STATUS_not_a_nan_bound():
    loss = np.full((2, 2), np.nan)
    b = anti_conservative_bound(loss, _all(loss))
    assert b["status"] == "NOT_EVALUABLE"
    assert "bound_at_worst" not in b


def test_a_keep_mask_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError):
        anti_conservative_bound(np.zeros((2, 3)), np.ones((3, 2), dtype=bool))


# ---------------------------------------------------------------------- #
# The two derivations of the exclusion set (AC1-3).
# ---------------------------------------------------------------------- #

def test_the_two_derivations_of_the_exclusion_set_agree():
    """`surv == 0` and 'all-NaN down the seed axis of the stored cmax' are
    produced by different code at different times. Here they are built to
    agree, cell for cell."""
    surv = np.array([[3, 0], [2, 2]])                 # (concepts, shifts)
    cmax = np.zeros((4, 2, 2))                        # (seeds, shifts, concepts)
    cmax[:, 1, 0] = np.nan
    r = exclusion_derivations_agree(surv, cmax)
    assert r["agree"] is True
    assert r["n_degenerate_by_bh1_criterion"] == 1
    assert r["n_degenerate_by_stored_nan"] == 1


def test_a_disagreement_between_the_derivations_is_REPORTED_POSITIONALLY():
    """Same TOTAL, different CELLS -- the shape a count can never detect
    (gotcha 27). The NaN is on the transposed cell."""
    surv = np.array([[3, 0], [2, 2]])
    cmax = np.zeros((4, 2, 2))
    cmax[:, 0, 1] = np.nan                            # concept 1, shift 0
    r = exclusion_derivations_agree(surv, cmax)
    assert r["agree"] is False
    assert r["n_degenerate_by_bh1_criterion"] == r["n_degenerate_by_stored_nan"]
    assert sorted(r["first_disagreements_(concept,shift)"]) == [[0, 1], [1, 0]]


def test_a_partially_nan_cell_is_NOT_a_degenerate_cell():
    """A degenerate cell has no maximum in ANY seed. One NaN seed out of four
    is something else, and calling it degenerate would drop a draw BH2 keeps."""
    surv = np.array([[3, 2]])
    cmax = np.zeros((4, 2, 1))
    cmax[0, 1, 0] = np.nan
    r = exclusion_derivations_agree(surv, cmax)
    assert r["n_degenerate_by_stored_nan"] == 0
    assert r["agree"] is True


def test_derivations_of_incompatible_shape_are_refused():
    with pytest.raises(ExclusionDerivationsDisagree):
        exclusion_derivations_agree(np.array([[3, 0, 1]]), np.zeros((4, 2, 1)))


# ---------------------------------------------------------------------- #
# BH3's outcome space, and the cells that drive the bound.
# ---------------------------------------------------------------------- #

def test_bh3_maps_every_case_INCLUDING_the_ones_with_no_members():
    """Concept 0 has no degenerate draw, concept 1 some, concept 2 all of arm
    B, concept 3 all of both. The unoccupied cases must still appear as keys
    with a count of zero -- a category that exists cannot silently become
    something else on a re-run."""
    surv_a = np.array([[3, 3], [3, 0], [3, 3], [0, 0]])
    surv_b = np.array([[3, 3], [3, 3], [0, 0], [0, 0]])
    r = bh3_outcomes({"a": surv_a, "b": surv_b}, ["c0", "c1", "c2", "c3"], 2)
    occ = r["occupancy"]
    assert occ["NONE_DEGENERATE"] == 1
    assert occ["SOME_BUT_NOT_ALL_DEGENERATE"] == 1
    assert occ["ALL_DEGENERATE_ONE_ARM_NOT_EVALUABLE_ARM"] == 1
    assert occ["ALL_DEGENERATE_BOTH_ARMS_NOT_EVALUABLE_STRATUM"] == 1
    assert r["concepts_by_case"]["ALL_DEGENERATE_BOTH_ARMS_NOT_EVALUABLE_STRATUM"] == ["c3"]


def test_bh3_keeps_the_empty_cases_as_zeros_when_nothing_is_degenerate():
    surv = np.array([[3, 3], [4, 4]])
    r = bh3_outcomes({"a": surv, "b": surv}, ["c0", "c1"], 2)
    assert r["occupancy"]["NONE_DEGENERATE"] == 2
    assert r["occupancy"]["SOME_BUT_NOT_ALL_DEGENERATE"] == 0
    assert r["occupancy"]["ALL_DEGENERATE_ONE_ARM_NOT_EVALUABLE_ARM"] == 0
    assert r["occupancy"]["ALL_DEGENERATE_BOTH_ARMS_NOT_EVALUABLE_STRATUM"] == 0


def test_bh3_counts_the_single_surviving_positive_cells_without_ruling_on_them():
    """EXACTLY one, not 'at most one'. The zero in row 1 is there so the
    fixture can tell `== 1` from `<= 1`: without it the two agree and the test
    cannot exhibit the condition it claims to check (gotcha 16)."""
    surv_a = np.array([[1, 1, 5], [0, 1, 3]])
    r = bh3_outcomes({"a": surv_a}, ["c0", "c1"], 3)
    assert r["n_cells_with_exactly_one_surviving_positive"] == {"a": 3}


def test_the_worst_surviving_cells_are_NAMED_and_skip_the_degenerate_one():
    """Worked by hand. Concept 0 lost every positive in shift 1 (a loss of -1,
    which BH2 drops); concept 2's shift 1 kept 1 of 4, a loss of -0.75 and the
    worst SURVIVING cell; concept 1's shift 1 kept 6 of 10, a loss of -0.4.

    Concept 2's single surviving positive is deliberate: BH3's fourth case
    says AP is defined there, so that cell must be RANKED, and a criterion of
    `> 1` would drop the very cell this test names as the worst."""
    surv = np.array([[4, 0], [10, 6], [4, 1]])
    obs = np.array([4, 10, 4])
    loss = relative_positive_loss(surv, obs)
    w = worst_cells(loss, surviving_draw_mask(surv), ["c0", "c1", "c2"], n=3)
    assert [c["concept"] for c in w] == ["c2", "c1", "c0"]
    assert w[0]["shift"] == 1 and w[0]["relative_loss"] == pytest.approx(-0.75)
    assert w[0]["bound"] == pytest.approx(4.0)
    assert w[1]["relative_loss"] == pytest.approx(-0.4)
    assert w[1]["bound"] == pytest.approx(1.0 / 0.6)


def test_the_formatter_is_TOTAL_over_an_empty_report():
    """No arms, no concepts, no bound. Must render, not raise."""
    assert "AC7" in format_summary({})


def test_the_formatter_renders_an_UNDEFINED_correlation():
    """A stratum where every concept's surviving count is constant has no
    correlation to report. `None` must format as UNDEFINED, not crash on a
    format spec."""
    txt = format_summary({
        "null": "x.h5", "n_concepts": 3,
        "arms": {"A": {"loss_contains": "collisions only",
                       "scaling_check": {"median_spearman_GAIN": None,
                                         "median_spearman_AP": None,
                                         "n_concepts_computed": 0,
                                         "status_counts": {"UNINFORMATIVE": 3}},
                       "combined_bound": {"status": "NO_FINITE_LOSS_VALUES"}}}})
    assert "UNDEFINED" in txt
    assert "NO_FINITE_LOSS_VALUES" in txt


def test_the_formatter_renders_a_complete_report():
    scaling = {"median_spearman_GAIN": 0.0049,
               "median_spearman_AP": -0.0012,
               "iqr_spearman_GAIN": [-0.1, 0.1],
               "iqr_spearman_AP": [-0.2, 0.2],
               "n_concepts_computed": 200,
               "status_counts": {"COMPUTED": 200}}
    bound = {"status": "COMPUTED",
             "worst_relative_loss": -0.2,
             "median_relative_loss": 0.0,
             "frac_cells_with_no_loss": 0.93,
             "bound_at_worst": 1.25,
             "bound_at_median": 1.0}
    txt = format_summary({
        "null": "x.h5", "n_concepts": 218,
        "arms": {"B": {"loss_contains": "collisions and dedup",
                       "scaling_check": scaling,
                       "combined_bound": bound}}})
    assert "+0.0049" in txt and "1.2500x" in txt and "93.00%" in txt
