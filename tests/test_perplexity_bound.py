"""Tests for `perplexity_bound` -- PREREGISTRATION amendment BC section 2.

THE FIXTURES ARE BUILT TO DISCRIMINATE, which for this module means:

  * **every cell of BC4.4's grid is exercised, including the two the brief
    hopes not to reach.** A rule needs every cell mapped (gotcha 24), and the
    non-finite branch is separate because `nan < 1.5` is False and would
    silently land a NaN in the wrong cell.
  * **the seed-order join must REFUSE the identity permutation.** The AP
    artifact is lexicographic and the stored fit is numeric; they coincide at
    positions 0 and 1 only, so a correct join moves 18 of 20 rows. A bijection
    check alone passes under the identity, which is precisely the mis-pairing
    gotcha 29 describes -- so there is a fixture whose permutation IS the
    identity and it must be reported as a problem.
  * **the acceptance criteria's references are INJECTABLE.** The real
    coefficients match BB9.3 by construction, so a check that could only ever
    be called with them can never be shown to fail (gotcha 16, and BB12 item
    2's `tolerance_coverage` lesson applied before rather than after).
  * **expectations are WRITTEN OUT, not computed from the module** (BB5b). The
    cut values, the verdict strings and the arithmetic answers below are
    literals, so changing a constant fails a test rather than silently
    redefining what the test asserts.
  * **the peer pool must exclude carriers of OTHER concepts**, so the fixture
    contains a feature that carries a different concept and would move the
    baseline if it were counted as an ordinary peer.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.analysis import perplexity_bound as pb


# =========================================================== the outcome grid

@pytest.mark.parametrize("contribution,expected", [
    (-3.0, "NEGATIVE_OR_ZERO_THE_TWO_HALVES_DO_NOT_AGREE"),
    (0.0, "NEGATIVE_OR_ZERO_THE_TWO_HALVES_DO_NOT_AGREE"),
    (0.4, "PERPLEXITY_CANNOT_ACCOUNT_FOR_THE_EXCESS"),
    (1.4999, "PERPLEXITY_CANNOT_ACCOUNT_FOR_THE_EXCESS"),
    (1.5, "PARTIAL_CONTRIBUTOR"),
    (3.0, "PARTIAL_CONTRIBUTOR"),
    (4.0, "PARTIAL_CONTRIBUTOR"),
    (4.0001, "LEADING_CANDIDATE"),
    (9.0, "LEADING_CANDIDATE"),
])
def test_every_cell_of_the_grid_is_reachable(contribution, expected):
    """The verdict strings are LITERALS here, not read from the module: a test
    that derives its cases from the thing under test cannot detect changes to
    that thing (BB5b)."""
    assert pb.outcome_cell(contribution, 7.2219)["verdict"] == expected


def test_a_nan_contribution_gets_its_own_branch_not_the_smallest_cell():
    """`nan < 1.5` is False and `nan <= 0.0` is False, so a comparison chain
    would drop a NaN into LEADING_CANDIDATE -- the single most alarming cell,
    reached by an absence of information."""
    got = pb.outcome_cell(float("nan"), 7.2219)
    assert got["verdict"] == "UNDEFINED_NON_FINITE_INPUT"
    assert got["runs_followon"] is False
    assert got["contribution_pp"] is None


def test_a_nan_excess_is_also_undefined():
    assert pb.outcome_cell(2.0, float("nan"))["verdict"] == \
        "UNDEFINED_NON_FINITE_INPUT"


def test_exactly_zero_is_a_disagreement_not_a_small_contribution():
    """Zero means the two halves cancel, which is the fourth cell's condition
    and not the first's. The two readings are opposite: one says 'close it',
    the other says 'one of the two halves was misread'."""
    got = pb.outcome_cell(0.0, 7.2219)
    assert "DO_NOT_AGREE" in got["verdict"]
    assert got["runs_followon"] is False


@pytest.mark.parametrize("contribution,runs", [
    (-1.0, False), (0.5, False), (2.0, True), (6.0, True),
])
def test_the_followon_runs_only_in_the_middle_two_cells(contribution, runs):
    assert pb.outcome_cell(contribution, 7.2219)["runs_followon"] is runs


def test_the_share_of_the_excess_is_reported_and_is_none_when_undefined():
    assert pb.outcome_cell(1.5, 7.2219)["share_of_excess"] == \
        pytest.approx(1.5 / 7.2219)
    assert pb.outcome_cell(1.5, 0.0)["share_of_excess"] is None


# ================================================== the across-seed combination

def test_combine_across_seeds_matches_meta_analyse_across_seed_arm():
    """Hand-computed, so the test does not restate the implementation."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    got = pb.combine_across_seeds(x)
    assert got["across_seed_mean"] == pytest.approx(2.5)
    # sd with ddof=1 over {1,2,3,4} is sqrt(5/3)
    assert got["across_seed_sd"] == pytest.approx(math.sqrt(5.0 / 3.0))
    assert got["across_seed_se"] == pytest.approx(math.sqrt(5.0 / 3.0) / 2.0)
    lo, hi = got["across_seed_ci95"]
    assert lo == pytest.approx(2.5 - 1.959964 * math.sqrt(5.0 / 3.0) / 2.0)
    assert hi == pytest.approx(2.5 + 1.959964 * math.sqrt(5.0 / 3.0) / 2.0)


def test_the_fixed_effect_arm_is_reported_absent_with_its_reason():
    """An interval that cannot be computed from what is stored is STATED as
    uncomputable rather than approximated (BC4.3 item 4). An empty string or a
    silent omission would read as 'not applicable'."""
    got = pb.combine_across_seeds(np.array([1.0, 2.0]))
    assert got["fixed_effect"] == "NOT_AVAILABLE"
    assert "no inverse-variance weight" in got["fixed_effect_reason"]


def test_non_finite_seeds_are_dropped_not_propagated():
    got = pb.combine_across_seeds(np.array([1.0, np.nan, 3.0]))
    assert got["n_seeds_used"] == 2
    assert got["across_seed_mean"] == pytest.approx(2.0)


def test_no_usable_seeds_is_a_status_not_a_nan_mean():
    assert pb.combine_across_seeds(np.array([np.nan, np.nan]))["status"] == \
        "NO_USABLE_SEEDS"


# ======================================================== the seed-order join

def _rows(labels):
    return [{"label": str(x), "beta": 0.0, "mean_p_1_minus_p": 0.25,
             "stored_ame_z_logM": 0.0, "beta_z_logM": 0.0,
             "position": i, "n_features": 10}
            for i, x in enumerate(labels)]


LEXICOGRAPHIC_20 = [f"seed_{x}" for x in sorted(range(20), key=str)]


def test_the_real_lexicographic_to_numeric_join_moves_18_of_20_rows():
    """Positions 0 and 1 are the only two where the two orders coincide
    (gotcha 29), so this number is a property of the ensemble and is written
    out rather than computed."""
    got = pb.join_rows_to_ap_order(_rows(range(20)), LEXICOGRAPHIC_20)
    assert got["verdict"] == "JOINED_BY_LABEL"
    assert got["n_positions_moved"] == 18
    assert got["permutation_ap_row_to_stored_row"][0] == 0
    assert got["permutation_ap_row_to_stored_row"][1] == 1
    assert got["permutation_ap_row_to_stored_row"][2] == 10


def test_an_identity_permutation_is_REFUSED_not_accepted_as_a_bijection():
    """THE LOAD-BEARING TEST. A bijection check alone passes under the identity,
    and the identity is exactly what a positional read of one of the two
    artifacts would produce. Only a check that refuses it discriminates."""
    got = pb.join_rows_to_ap_order(_rows(range(20)),
                                   [f"seed_{i}" for i in range(20)])
    assert got["verdict"] == "AMBIGUOUS"
    assert got["n_positions_moved"] == 0
    assert any("IDENTITY" in p for p in got["problems"])


def test_a_missing_label_is_a_problem_and_marks_the_row_minus_one():
    got = pb.join_rows_to_ap_order(_rows([0, 1, 2]), ["seed_0", "seed_1",
                                                      "seed_9"])
    assert got["verdict"] == "AMBIGUOUS"
    assert got["permutation_ap_row_to_stored_row"][2] == -1
    assert any("seed label 9" in p for p in got["problems"])


def test_a_duplicate_label_in_the_stored_fit_is_a_problem():
    got = pb.join_rows_to_ap_order(_rows([0, 1, 1]),
                                   ["seed_0", "seed_1", "seed_2"])
    assert got["verdict"] == "AMBIGUOUS"
    assert any("duplicate label" in p for p in got["problems"])


def test_bare_integer_ap_labels_are_accepted_as_well_as_seed_prefixed():
    got = pb.join_rows_to_ap_order(_rows([0, 1, 2]), ["2", "0", "1"])
    assert got["verdict"] == "JOINED_BY_LABEL"
    assert got["permutation_ap_row_to_stored_row"] == [2, 0, 1]


# ============================================================== ACCEPTANCE C

def _ame_rows(n=3, perturb=None):
    rows = []
    for i in range(n):
        beta, w = 0.001 + 0.0001 * i, 0.24 + 0.001 * i
        stored = float(np.mean(beta * np.full(1000, w)))
        if perturb is not None and i == 0:
            stored *= (1.0 + perturb)
        rows.append({"label": str(i), "beta": -0.2,
                     "mean_p_1_minus_p": w, "beta_z_logM": beta,
                     "stored_ame_z_logM": stored, "position": i,
                     "n_features": 1000})
    return rows


def test_acceptance_C_passes_when_the_reconstruction_matches():
    got = pb.acceptance_ame_reconstruction(_ame_rows())
    assert got["verdict"] == "REPRODUCES_THE_STORED_AME"
    assert got["n_rows"] == 3


def test_acceptance_C_fails_on_a_relative_move_of_1e_minus_9():
    """A move far too small to notice in a printed table and far too large to
    be float noise in this reconstruction."""
    got = pb.acceptance_ame_reconstruction(_ame_rows(perturb=1e-9))
    assert got["verdict"] == "DIFFERS_FROM_THE_STORED_AME"
    assert got["max_relative_residual"] > 1e-12


def test_acceptance_C_reports_the_residual_whatever_the_verdict():
    """A verdict with no number beside it cannot be audited, and the tolerance
    would then be implicit."""
    for rows in (_ame_rows(), _ame_rows(perturb=1e-9)):
        got = pb.acceptance_ame_reconstruction(rows)
        assert isinstance(got["max_relative_residual"], float)
        assert got["rtol"] == 1e-12


def test_acceptance_C_on_no_rows_is_a_status_not_a_pass():
    assert pb.acceptance_ame_reconstruction([])["verdict"] == \
        "NOT_COMPUTED_NO_ROWS"


# ============================================================== ACCEPTANCE F

def _beta_rows(betas):
    return [{"label": str(i), "beta": float(b), "mean_p_1_minus_p": 0.24,
             "beta_z_logM": 0.0, "stored_ame_z_logM": 0.0, "position": i,
             "n_features": 10} for i, b in enumerate(betas)]


def test_acceptance_F_passes_on_its_own_reference():
    rows = _beta_rows([-0.20, -0.18, -0.22, -0.16])
    b = np.array([-0.20, -0.18, -0.22, -0.16])
    got = pb.acceptance_bb93_summary(rows, mean_ref=float(b.mean()),
                                     sd_ref=float(b.std(ddof=1)),
                                     n_pos_ref=0)
    assert got["verdict"] == "REPRODUCES_BB93"


def test_acceptance_F_fails_when_the_mean_moves_in_the_fifth_decimal():
    """THE FIXTURE MUST BE ABLE TO EXHIBIT THE MISMATCH. The real block matches
    by construction, so without injectable references this branch could never
    be reached (gotcha 16)."""
    rows = _beta_rows([-0.20, -0.18, -0.22, -0.16])
    got = pb.acceptance_bb93_summary(rows, mean_ref=-0.19001, sd_ref=0.02582,
                                     n_pos_ref=0)
    assert got["verdict"] == "DIFFERS_FROM_BB93"


def test_acceptance_F_fails_when_a_coefficient_changes_SIGN():
    """The unanimity of the sign is the whole content of BB9.3's second half,
    so a check that ignored `n_positive` would miss the one thing that matters
    about it."""
    b = [-0.20, -0.18, -0.22, +0.16]
    arr = np.array(b)
    got = pb.acceptance_bb93_summary(_beta_rows(b), mean_ref=float(arr.mean()),
                                     sd_ref=float(arr.std(ddof=1)),
                                     n_pos_ref=0)
    assert got["verdict"] == "DIFFERS_FROM_BB93"
    assert got["n_positive"] == 1


# ============================================================== ACCEPTANCE D

@pytest.mark.parametrize("up,down,expected", [
    (1.5, -1.5, "SIGN_CONVENTION_CONFIRMED"),
    (1.5, +1.5, "SIGN_CONVENTION_NOT_CONFIRMED"),
    (-1.5, -1.5, "SIGN_CONVENTION_NOT_CONFIRMED"),
    (0.0, -1.5, "SIGN_CONVENTION_NOT_CONFIRMED"),
])
def test_the_probes_verdict_requires_BOTH_arms(up, down, expected):
    """THE REQUIREMENT, TESTED SEPARATELY FROM THE FIT. Handed a real dataset
    the probe's two arms always behave, so a fixture that could only call
    `sign_convention_probe` end to end cannot exhibit a control arm coming back
    the WRONG way -- and a mutation dropping the control from the conjunction
    survives it (gotcha 16). Splitting the verdict out is what makes the
    requirement checkable at all."""
    assert pb.probe_verdict(up, down)["verdict"] == expected


def test_the_sign_convention_probe_confirms_and_its_control_is_the_opposite():
    """A probe that returned POSITIVE for any input would establish nothing
    (gotcha 25): ask which outcomes the instrument is capable of producing and
    confirm the answer is more than one. The flipped arm is that check."""
    got = pb.sign_convention_probe(n=400, n_trials=39, slope=1.5, seed=7)
    assert got["verdict"] == "SIGN_CONVENTION_CONFIRMED"
    assert got["recovered_positive_arm"] > 0.0
    assert got["recovered_flipped_control_arm"] < 0.0


# ================================================================ compare_exact

def test_compare_exact_treats_an_absent_key_as_a_difference():
    diffs, n = pb.compare_exact({"a": 1, "b": 2}, {"a": 1})
    assert "/b" in diffs
    assert diffs["/b"]["now"] == "__ABSENT__"
    assert n >= 2


def test_compare_exact_counts_leaves_so_a_vacuous_pass_is_visible():
    """`differences == {}` over ZERO leaves is not a pass, and only the count
    can tell the two apart (gotcha 28).

    **BOTH CASES ARE NEEDED.** The empty document alone counts zero leaves
    whether or not the counter works, so it cannot see a counter stuck at
    zero -- the matching document is what discriminates."""
    diffs, n = pb.compare_exact({}, {})
    assert diffs == {} and n == 0
    diffs, n = pb.compare_exact({"a": 1, "b": {"c": 2}}, {"a": 1, "b": {"c": 2}})
    assert diffs == {}
    assert n == 2


def test_compare_exact_sees_a_list_length_change():
    diffs, _ = pb.compare_exact({"x": [1, 2, 3]}, {"x": [1, 2]})
    assert "/x" in diffs


def test_compare_exact_sees_a_value_moved_in_the_last_bit():
    v = 0.4955123456789
    diffs, _ = pb.compare_exact({"x": v}, {"x": float(np.nextafter(v, 1.0))})
    assert "/x" in diffs


# =========================================================== the displacement

def _displacement_fixture():
    """ONE seed, 8 features in TWO firing bins, 3 concepts, all detected.

    bin 0 = features 0..3, bin 1 = features 4..7.
    carriers: concept 0 -> feature 0, concept 1 -> feature 4, concept 2 -> f 5.
    So bin 0's peers are {1, 2, 3} and bin 1's peers are {6, 7} -- features 4
    and 5 are EXCLUDED from bin 1's peer pool because they carry concepts.

    **THE PEER POOLS ARE SKEWED ON PURPOSE.** bin 0's peers are {1, 2, 9} with
    mean 4.0 and median 2.0, and bin 1's is {4, 6} with mean 5.0 and median
    5.0 only because two points have no third. A pool whose mean equalled its
    median could not exhibit a mean-for-median substitution at all, and the
    first version of this fixture had exactly that defect -- the mutation
    survived it (gotcha 16).
    """
    carrier = np.array([[0, 4, 5]])
    detected = np.ones((1, 3), dtype=bool)
    value = np.array([[10.0, 1.0, 2.0, 9.0, 20.0, 30.0, 4.0, 6.0]])
    fbin = np.array([[0, 0, 0, 0, 1, 1, 1, 1]])
    return carrier, detected, value, fbin


def test_the_displacement_is_the_carrier_minus_its_own_bins_peer_MEAN():
    carrier, detected, value, fbin = _displacement_fixture()
    got = pb.covariate_displacement(carrier, detected, value, fbin, 2)
    # bin 0: peers {1,2,9} MEAN 4.0 (median 2.0) -> carrier f0 at 10.0: +6.0
    # bin 1: peers {4,6}   MEAN 5.0             -> f4=20, f5=30: +15, +25
    row = got["per_seed"][0]
    assert row["n_carrier_cells"] == 3
    assert row["mean_displacement"] == pytest.approx((6.0 + 15.0 + 25.0) / 3.0)
    assert row["median_displacement"] == pytest.approx(15.0)
    # and the peer MEDIAN would give +8.0 there, a different answer -- which is
    # what makes this fixture able to see a mean-for-median substitution.
    assert (6.0 + 15.0 + 25.0) / 3.0 != pytest.approx((8.0 + 15.0 + 25.0) / 3.0)


def test_a_feature_carrying_ANOTHER_concept_is_not_an_ordinary_peer():
    """Without `carrier_feature_indicator`'s complement, feature 5 (a carrier
    of concept 2) would sit in feature 4's peer pool and pull the baseline
    toward the carriers by construction. The fixture is built so the two
    answers differ."""
    carrier, detected, value, fbin = _displacement_fixture()
    got = pb.covariate_displacement(carrier, detected, value, fbin, 2)
    naive_bin1_mean = float(np.mean([20.0, 30.0, 4.0, 6.0]))
    assert naive_bin1_mean != pytest.approx(5.0)
    d = got["per_seed"][0]["mean_displacement"]
    naive = np.mean([10.0 - np.mean([1.0, 2.0, 9.0, 10.0]),
                     20.0 - naive_bin1_mean, 30.0 - naive_bin1_mean])
    assert d != pytest.approx(naive)


def test_an_empty_peer_pool_is_counted_unusable_not_scored_as_zero():
    """A bin in which every feature carries a concept has no baseline. Scoring
    it 0.0 would read as 'measured, and exactly average'."""
    carrier = np.array([[0, 1]])
    detected = np.ones((1, 2), dtype=bool)
    value = np.array([[5.0, 7.0, 1.0, 2.0]])
    fbin = np.array([[0, 0, 1, 1]])
    got = pb.covariate_displacement(carrier, detected, value, fbin, 2)
    assert got["n_carrier_cells_without_a_peer_pool"] == 2
    assert got["status"] == "NO_USABLE_CARRIER_CELLS"


def test_the_sd_reference_makes_the_per_sd_form_an_identity():
    """`(ame * sd) * (delta / sd) == ame * delta` is the reason the SD choice
    is not a degree of freedom, and it is measured rather than asserted."""
    carrier, detected, value, fbin = _displacement_fixture()
    disp = pb.covariate_displacement(carrier, detected, value, fbin, 2)
    rows = [{"label": "0", "beta": -0.2, "mean_p_1_minus_p": 0.25,
             "beta_z_logM": 0.0, "stored_ame_z_logM": 0.0, "position": 0,
             "n_features": 8}]
    got = pb.contribution_bound(rows, [0], disp, 7.2219)
    assert got["max_identity_residual_pp"] == pytest.approx(0.0, abs=1e-12)


def test_the_per_stratum_breakdown_names_every_bin_even_the_empty_ones():
    carrier, detected, value, fbin = _displacement_fixture()
    got = pb.covariate_displacement(carrier, detected, value, fbin, 4)
    assert [r["stratum"] for r in got["by_firing_stratum"]] == [0, 1, 2, 3]
    assert got["by_firing_stratum"][3]["mean_displacement"] is None


# ============================================================== pair weights

def test_pair_weights_count_only_the_seeds_ABOVE_each_row():
    """`carrier_rates_by_cell` loops `for s, for t in s+1..`, so cell (s, c)'s
    weight is the number of LATER seeds that also detect `c`. Hand-computed."""
    det = np.array([[True, True], [True, False], [True, True]])
    w = pb.pair_weights_from_detection(det)
    # concept 0: seeds 1 and 2 detect -> row 0 weight 2, row 1 weight 1, row 2 = 0
    assert w[0, 0] == 2.0 and w[1, 0] == 1.0 and w[2, 0] == 0.0
    # concept 1: seed 1 does NOT detect -> row 0 sees only seed 2
    assert w[0, 1] == 1.0 and w[2, 1] == 0.0


def test_pair_weights_are_zero_where_the_row_itself_does_not_detect():
    det = np.array([[False, True], [True, True]])
    w = pb.pair_weights_from_detection(det)
    assert w[0, 0] == 0.0


# =========================================================== the bound itself

def _bound_inputs(mean_disp, median_disp, beta=-0.2, w=0.25):
    disp = {"status": "COMPUTED", "per_seed": [
        {"seed_row": 0, "status": "COMPUTED", "n_carrier_cells": 5,
         "mean_displacement": mean_disp, "median_displacement": median_disp,
         "sd_reference": 2.0, "mean_displacement_in_peer_sd_per_cell": 0.1},
        {"seed_row": 1, "status": "COMPUTED", "n_carrier_cells": 5,
         "mean_displacement": mean_disp, "median_displacement": median_disp,
         "sd_reference": 2.0, "mean_displacement_in_peer_sd_per_cell": 0.1},
    ]}
    rows = [{"label": str(i), "beta": beta, "mean_p_1_minus_p": w,
             "beta_z_logM": 0.0, "stored_ame_z_logM": 0.0, "position": i,
             "n_features": 10} for i in range(2)]
    return rows, disp


def test_the_contribution_is_the_product_of_the_two_halves_in_pp():
    """beta -0.2, mean(p(1-p)) 0.25 -> AME -0.05 per unit; displacement -0.28
    -> +0.014 -> +1.40 pp. Hand-computed."""
    rows, disp = _bound_inputs(-0.28, -0.28)
    got = pb.contribution_bound(rows, [0, 1], disp, 7.2219)
    assert got["primary_mean_reduction"]["across_seed_mean"] == \
        pytest.approx(1.40)
    assert got["outcome"]["verdict"] == \
        "PERPLEXITY_CANNOT_ACCOUNT_FOR_THE_EXCESS"


def test_two_negatives_make_a_POSITIVE_contribution():
    """The whole reason perplexity is the live candidate: carriers are DEPLETED
    (negative displacement) and the coefficient is NEGATIVE, so the model
    predicts carriers reproduce MORE."""
    rows, disp = _bound_inputs(-0.5, -0.5)
    got = pb.contribution_bound(rows, [0, 1], disp, 7.2219)
    assert got["primary_mean_reduction"]["across_seed_mean"] > 0.0


def test_a_positive_displacement_against_a_negative_coefficient_disagrees():
    """The fourth cell's condition, exhibited rather than argued: if carriers
    were ENRICHED in perplexity the same coefficient would predict them
    reproducing LESS, which contradicts a positive excess."""
    rows, disp = _bound_inputs(+0.5, +0.5)
    got = pb.contribution_bound(rows, [0, 1], disp, 7.2219)
    assert got["outcome"]["verdict"] == \
        "NEGATIVE_OR_ZERO_THE_TWO_HALVES_DO_NOT_AGREE"
    assert got["outcome"]["runs_followon"] is False


def test_a_straddle_is_DETECTED_and_the_mean_takes_the_branch():
    """BC4.4 declared the straddle rule before the run. Mean -0.32 gives
    +1.60 pp (PARTIAL); median -0.28 gives +1.40 pp (CANNOT_ACCOUNT)."""
    rows, disp = _bound_inputs(-0.32, -0.28)
    got = pb.contribution_bound(rows, [0, 1], disp, 7.2219)
    assert got["reductions_straddle_a_cut"] is True
    assert got["outcome"]["verdict"] == "PARTIAL_CONTRIBUTOR"
    assert got["outcome_under_the_median_reduction"]["verdict"] == \
        "PERPLEXITY_CANNOT_ACCOUNT_FOR_THE_EXCESS"


def test_no_straddle_is_reported_when_both_reductions_agree():
    rows, disp = _bound_inputs(-0.28, -0.28)
    got = pb.contribution_bound(rows, [0, 1], disp, 7.2219)
    assert got["reductions_straddle_a_cut"] is False
    assert "do not depend" in got["straddle_note"] or \
        "does not depend" in got["straddle_note"]


def test_the_join_permutation_is_HONOURED_not_ignored():
    """If the coefficient were taken from the row at the same POSITION rather
    than the row named by `perm`, this fixture would give the wrong answer --
    which is the seed-order defect in miniature."""
    disp = {"status": "COMPUTED", "per_seed": [
        {"seed_row": 0, "status": "COMPUTED", "n_carrier_cells": 5,
         "mean_displacement": -1.0, "median_displacement": -1.0,
         "sd_reference": 2.0, "mean_displacement_in_peer_sd_per_cell": 0.1},
    ]}
    rows = [{"label": "a", "beta": -0.4, "mean_p_1_minus_p": 0.25,
             "beta_z_logM": 0.0, "stored_ame_z_logM": 0.0, "position": 0,
             "n_features": 10},
            {"label": "b", "beta": -0.8, "mean_p_1_minus_p": 0.25,
             "beta_z_logM": 0.0, "stored_ame_z_logM": 0.0, "position": 1,
             "n_features": 10}]
    got = pb.contribution_bound(rows, [1], disp, 7.2219)
    # row 1 is beta -0.8 -> AME -0.2 -> contribution +0.2 -> +20 pp
    assert got["per_seed"][0]["contribution_pp"] == pytest.approx(20.0)
    assert got["per_seed"][0]["seed_label"] == "b"


def test_an_unjoinable_row_is_NOT_COMPUTED_not_silently_dropped():
    rows, disp = _bound_inputs(-0.28, -0.28)
    got = pb.contribution_bound(rows, [0, -1], disp, 7.2219)
    assert got["n_seeds_used"] == 1
    assert got["per_seed"][1]["status"] == "NOT_COMPUTED"


# ================================================================ the formatter

def test_the_formatter_is_TOTAL_over_an_empty_result():
    """BB5d: a completed measurement was once reported as a FAILED job because
    a summary print died on a legitimately UNDEFINED value, after the artifact
    was written and acceptance had passed."""
    lines = pb.format_summary({})
    assert any("ABSENT" in ln for ln in lines)


def test_the_formatter_renders_a_forbidden_followon():
    res = {"bound": {"primary_mean_reduction": pb.combine_across_seeds(
        np.array([1.0, 1.2])), "outcome": pb.outcome_cell(1.1, 7.2219)},
        "section_2e_perplexity_matched_ambient": {
            "status": "NOT_RUN_THE_PRE_DECLARED_BRANCH_FORBIDS_IT"}}
    lines = pb.format_summary(res)
    assert any("DOES NOT RUN" in ln for ln in lines)
    assert any("NOT_RUN_THE_PRE_DECLARED_BRANCH_FORBIDS_IT" in ln
               for ln in lines)


def test_the_formatter_renders_an_undefined_share():
    res = {"bound": {"primary_mean_reduction": {"status": "NO_USABLE_SEEDS"},
                     "outcome": pb.outcome_cell(float("nan"), 7.2219)}}
    lines = pb.format_summary(res)
    assert any("UNDEFINED" in ln for ln in lines)


# ============================================== reading the stored fit arm

def _au_doc(labels=(0, 1), covariate="perplexity"):
    return {"sensitivity": {"feature_side": {
        "arm": "AL8b", "entry": "LINEAR",
        "seed_axis_provenance_AT_A1_item_3": {
            "verdict": "VERIFIED_POSITIONALLY",
            "per_anchor": [{"anchor_position": i, "anchor_label": str(v)}
                           for i, v in enumerate(labels)]},
        "per_seed": [
            {"seed": i, "n_features": 10,
             "extra_coefs": {covariate: -0.2 - 0.01 * i, "concentration": -5.0},
             "marginal_effect": {"ame": 0.0002, "mean_p_1_minus_p": 0.24,
                                 "beta": 0.001}}
            for i in range(len(labels))]}}}


def test_read_feature_side_arm_takes_the_label_from_PROVENANCE():
    """Not from the row's position -- the row's position is exactly what the
    seed-order split makes unsafe."""
    got = pb.read_feature_side_arm(_au_doc(labels=(7, 3)))
    assert [r["label"] for r in got["rows"]] == ["7", "3"]
    assert got["problems"] == []


def test_read_feature_side_arm_flags_a_row_with_no_provenance_record():
    doc = _au_doc(labels=(0, 1))
    doc["sensitivity"]["feature_side"][
        "seed_axis_provenance_AT_A1_item_3"]["per_anchor"].pop()
    got = pb.read_feature_side_arm(doc)
    assert any("no provenance record" in p for p in got["problems"])


def test_read_feature_side_arm_flags_a_missing_covariate():
    got = pb.read_feature_side_arm(_au_doc(), covariate="not_a_covariate")
    assert len(got["problems"]) == 2
    assert got["rows"] == []


# ============================================ the perplexity acceptance block

def test_acceptance_B_perplexity_reproduces_and_then_fails_on_one_moved_value(
        tmp_path):
    block = {"status": "COMPUTED", "median_percentile": 20.962517353077278,
             "iqr_percentile": [7.6, 45.1], "n_carrier_cells": 11907}
    ref = tmp_path / "conc.json"
    ref.write_text(json.dumps(
        {"section_3a_enrichment": {"perplexity": {"distribution": block}}}))
    assert pb._reproduces_stored_perplexity(dict(block), ref)["verdict"] == \
        "REPRODUCES_BB91_PERPLEXITY_EXACTLY"
    moved = dict(block)
    moved["median_percentile"] = float(np.nextafter(20.962517353077278, 21.0))
    got = pb._reproduces_stored_perplexity(moved, ref)
    assert got["verdict"] == "DIFFERS_FROM_BB91_PERPLEXITY"
    assert "/median_percentile" in got["differences"]


def test_acceptance_B_reports_an_absent_reference_block_rather_than_passing(
        tmp_path):
    ref = tmp_path / "conc.json"
    ref.write_text(json.dumps({"section_3a_enrichment": {}}))
    assert pb._reproduces_stored_perplexity({"a": 1}, ref)["verdict"] == \
        "REFERENCE_BLOCK_ABSENT"
