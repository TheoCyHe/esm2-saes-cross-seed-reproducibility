"""Tests for the matching-rule sensitivity (amendment CJ).

EVERY EXPECTATION HERE IS HAND-COMPUTED AND WRITTEN DOWN, never derived from
the module under test (gotcha 32b). The worked arithmetic is in the docstrings
so a reader can check the expectation without running anything.
"""
import numpy as np
import pytest

from src.analysis.rule_sensitivity import (
    cosine_matrix,
    encoder_decoder_from_assignments,
    encoder_decoder_partner,
    encoder_decoder_similarity,
    fraction_at,
    hungarian_assignment,
    literal_two_sided_similarity,
    mnn_nested_in_oneway,
    mnn_pairs_not_chosen_by_hungarian,
    mnn_partner,
    mnn_similarity,
    oneway_similarity,
    partner_view_from_b,
)

# The worked example, used by most tests below.
#
#   Cd = [[0.90, 0.80, 0.10],
#         [0.85, 0.20, 0.10],
#         [0.10, 0.10, 0.70]]
#
# row argmax : row0 -> col0 (0.90), row1 -> col0 (0.85), row2 -> col2 (0.70)
# col argmax : col0 -> row0 (0.90), col1 -> row0 (0.80), col2 -> row2 (0.70)
#
# MNN        : row0 mutual (0.90); row1 NOT (col0 prefers row0); row2 mutual (0.70)
# Hungarian  : enumerating all 6 permutations, the maximum total is
#              0->1, 1->0, 2->2  =  0.80 + 0.85 + 0.70  =  2.35
#              (against 1.80 for the identity, 1.10, 1.05, 1.00, 0.40)
CD = np.array([[0.90, 0.80, 0.10],
               [0.85, 0.20, 0.10],
               [0.10, 0.10, 0.70]])

# Encoder matrix chosen so the encoder matching is the IDENTITY, which
# therefore disagrees with the decoder matching on rows 0 and 1 and agrees
# only on row 2.
#   Ce = [[0.90, 0.10, 0.20],
#         [0.10, 0.90, 0.20],
#         [0.20, 0.20, 0.75]]
# Hungarian on Ce: 0->0, 1->1, 2->2 = 0.90 + 0.90 + 0.75 = 2.55, the maximum.
CE = np.array([[0.90, 0.10, 0.20],
               [0.10, 0.90, 0.20],
               [0.20, 0.20, 0.75]])


def test_cosine_matrix_normalises_both_sides():
    """A's columns are (3,4) with norm 5 and (0,1) with norm 1; B is the
    identity. So the cosines are 3/5, 4/5, 0, 1 -- and an UNNORMALISED product
    would give 3 and 4, which is the defect this guards (encoder rows are not
    unit-norm)."""
    A = np.array([[3.0, 0.0], [4.0, 1.0]])
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(cosine_matrix(A, B),
                               np.array([[0.6, 0.8], [0.0, 1.0]]))


def test_oneway_is_the_row_max():
    np.testing.assert_allclose(oneway_similarity(CD), [0.90, 0.85, 0.70])


def test_mnn_drops_the_non_mutual_row():
    """Row 1's best partner is col0, but col0's best is row0, so row1 has no
    mutual partner and carries -inf."""
    got = mnn_similarity(CD)
    assert got[0] == pytest.approx(0.90)
    assert got[1] == -np.inf
    assert got[2] == pytest.approx(0.70)


def test_hungarian_picks_the_maximum_total_not_the_row_maxima():
    """The hand-enumerated optimum is 0->1, 1->0, 2->2 (total 2.35). Note this
    does NOT give row0 its best partner -- which is the whole reason Hungarian
    and MNN can disagree."""
    partner, sim = hungarian_assignment(CD)
    np.testing.assert_array_equal(partner, [1, 0, 2])
    np.testing.assert_allclose(sim, [0.80, 0.85, 0.70])


def test_mnn_pair_can_be_rejected_by_hungarian_W1_counterexample():
    """W1's constructive counterexample, at theta=0.7: rows 0 and 2 are MNN
    pairs; Hungarian keeps row2's and rejects row0's, so the count is 1."""
    assert mnn_pairs_not_chosen_by_hungarian(CD, 0.70) == 1


def test_encoder_decoder_rule_keeps_only_the_agreeing_row():
    """Decoder matching is [1,0,2]; encoder matching is [0,1,2]. They agree on
    row 2 only, so the agreement fraction is 1/3 and the only finite similarity
    is min(0.70, 0.75) = 0.70."""
    sim, agree = encoder_decoder_similarity(CD, CE)
    assert agree == pytest.approx(1.0 / 3.0)
    assert sim[0] == -np.inf and sim[1] == -np.inf
    assert sim[2] == pytest.approx(0.70)


def test_rule3_takes_the_MINIMUM_so_a_threshold_binds_on_both():
    """The criterion is 'above threshold according to BOTH', so a pair whose
    decoder cosine clears 0.7 but whose encoder cosine does not must NOT count.
    Here both matchings are the identity and agree everywhere; row0's decoder
    similarity is 0.95 but its encoder similarity is 0.60."""
    cd = np.array([[0.95, 0.0], [0.0, 0.80]])
    ce = np.array([[0.60, 0.0], [0.0, 0.90]])
    sim, agree = encoder_decoder_similarity(cd, ce)
    assert agree == pytest.approx(1.0)
    assert fraction_at(sim, 0.70) == pytest.approx(0.5)   # row1 only


def test_fraction_denominator_is_all_features_not_matched_features():
    """Amendment CJ3.2: the denominator is always the full feature count. Two
    of three rows clear 0.7, so the answer is 2/3 -- not 2/2 over the pairs
    that happen to be matched."""
    assert fraction_at(mnn_similarity(CD), 0.70) == pytest.approx(2.0 / 3.0)


def test_nesting_check_returns_empty_on_the_worked_example():
    thetas = [0.1, 0.5, 0.7, 0.9]
    assert mnn_nested_in_oneway(CD, thetas) == []


def test_the_nesting_check_CANNOT_FAIL_and_that_is_recorded_not_hidden():
    """GOTCHA 25(a): a check that can only pass is as useless as one that can
    never pass, and saying so is the point of this test.

    `mnn_similarity` returns the row max where mutual and -inf otherwise, so it
    is elementwise <= `oneway_similarity` BY CONSTRUCTION, for every input.
    The nesting check therefore cannot fire on any data whatever.

    It is retained as a REGRESSION guard -- it would fire if someone changed
    either function so the nesting broke -- and it is NOT evidence about the
    data. C3 (`mnn_pairs_not_chosen_by_hungarian`) is the control that can
    genuinely come out either way, and it is the one reported.

    Asserted here over random matrices so the claim is tested, not just stated.
    """
    rng = np.random.default_rng(0)
    for _ in range(50):
        C = rng.normal(size=(12, 12))
        assert np.all(mnn_similarity(C) <= oneway_similarity(C))
        assert mnn_nested_in_oneway(C, [-1.0, 0.0, 0.5, 1.0]) == []


def test_literal_criterion_variant_D_matches_on_the_decoder():
    """Decoder Hungarian gives partners [1,0,2] (hand-enumerated above). The
    two-sided minima are then min(Cd[0,1],Ce[0,1]) = min(0.80,0.10) = 0.10,
    min(Cd[1,0],Ce[1,0]) = min(0.85,0.10) = 0.10, and
    min(Cd[2,2],Ce[2,2]) = min(0.70,0.75) = 0.70. Only row 2 clears 0.70."""
    sim = literal_two_sided_similarity(CD, CE, match_on="decoder")
    np.testing.assert_allclose(sim, [0.10, 0.10, 0.70])
    assert fraction_at(sim, 0.70) == pytest.approx(1.0 / 3.0)


def test_literal_criterion_variant_E_matches_on_the_encoder():
    """Encoder Hungarian gives the identity [0,1,2]. The minima are then
    min(0.90,0.90) = 0.90, min(0.20,0.90) = 0.20, min(0.70,0.75) = 0.70, so
    rows 0 and 2 clear 0.70."""
    sim = literal_two_sided_similarity(CD, CE, match_on="encoder")
    np.testing.assert_allclose(sim, [0.90, 0.20, 0.70])
    assert fraction_at(sim, 0.70) == pytest.approx(2.0 / 3.0)


def test_the_two_variants_can_disagree_so_the_degree_of_freedom_is_real():
    """1/3 against 2/3 on this fixture. The fixture DISCRIMINATES: if the
    matching's cost matrix did not matter, these would be equal, and amendment
    CL1's "second unstated degree of freedom" would be an empty worry."""
    d = fraction_at(literal_two_sided_similarity(CD, CE, "decoder"), 0.70)
    e = fraction_at(literal_two_sided_similarity(CD, CE, "encoder"), 0.70)
    assert d != e


def test_match_on_has_no_default_and_refuses_an_unknown_value():
    """The criterion does not determine the cost matrix, so a silent default
    would be an unrecorded choice (CL1). An unknown value must raise, not fall
    back."""
    with pytest.raises(ValueError, match="cannot be guessed"):
        literal_two_sided_similarity(CD, CE, match_on="both")


def test_rule3_is_a_strict_subset_of_both_variants_CL1a():
    """Amendment CL1a's arithmetic, asserted over random matrices rather than
    argued: rule 3's set is contained in each variant, and each variant exceeds
    it only on features where the two matchings disagree. This is the property
    that makes the containment check a CHECK rather than a prediction.

    On the worked fixture: rule 3 keeps {2}, variant D keeps {2}, variant E
    keeps {0,2}, and the matchings disagree on {0,1}.
    """
    rng = np.random.default_rng(7)
    for _ in range(40):
        Cd = rng.uniform(-1, 1, size=(9, 9))
        Ce = rng.uniform(-1, 1, size=(9, 9))
        theta = 0.3
        m3, agree_frac = encoder_decoder_similarity(Cd, Ce)
        keep3 = m3 >= theta
        disagree = 1.0 - agree_frac
        for who in ("decoder", "encoder"):
            v = literal_two_sided_similarity(Cd, Ce, who) >= theta
            assert np.all(v[keep3]), "rule 3 must be contained in the variant"
            excess = float(v.mean()) - float(keep3.mean())
            assert -1e-12 <= excess <= disagree + 1e-12


# ---------------------------------------------------------------------------
# THE B-SIDE OF AN UNORDERED PAIR -- amendment CO
# ---------------------------------------------------------------------------

def test_mnn_partner_worked_example():
    """From the header arithmetic: row0 is mutual with col0, row1 is NOT (col0
    prefers row0), row2 is mutual with col2. So [0, -1, 2] -- hand-written, not
    derived from `mnn_similarity`."""
    np.testing.assert_array_equal(mnn_partner(CD), [0, -1, 2])


def test_mnn_partner_and_similarity_agree_on_which_rows_matched():
    """The two functions are separate (gotcha 27) and must not drift apart:
    -1 exactly where the similarity is -inf. Hand-written: rows {0,2} matched."""
    p, s = mnn_partner(CD), mnn_similarity(CD)
    np.testing.assert_array_equal(p >= 0, [True, False, True])
    np.testing.assert_array_equal(np.isfinite(s), [True, False, True])


def test_partner_view_from_b_reproduces_the_transposed_mnn_independently():
    """THE INDEPENDENT-PATH CHECK, and the reason this helper is trustworthy.

    B's MNN vector can be obtained two ways: by SCATTERING A's vector through
    the partner index, or by running `mnn_similarity` on the TRANSPOSE, which
    shares no code with the scatter. Mutual-nearest-neighbour is symmetric, so
    the two must agree elementwise -- on rectangular matrices too, where an
    axis mix-up would show up as a shape error rather than a wrong number.
    """
    rng = np.random.default_rng(11)
    for shape in ((6, 5), (5, 6), (7, 7)):
        C = rng.uniform(-1, 1, size=shape)
        scattered = partner_view_from_b(mnn_partner(C), mnn_similarity(C),
                                        shape[1])
        np.testing.assert_allclose(scattered, mnn_similarity(C.T))


def test_partner_view_from_b_hungarian_worked_example():
    """Hungarian on CD assigns 0->1, 1->0, 2->2 with similarities 0.80, 0.85,
    0.70. Read off B: col0 receives row1's 0.85, col1 receives row0's 0.80,
    col2 receives row2's 0.70. So [0.85, 0.80, 0.70] -- hand-computed."""
    partner, sim = hungarian_assignment(CD)
    np.testing.assert_allclose(
        partner_view_from_b(partner, sim, 3), [0.85, 0.80, 0.70])


def test_partner_view_from_b_refuses_a_repeated_index():
    """A repeat would overwrite on scatter and LOSE a matched feature -- a
    count silently too low in a quantity that is entirely a distribution over
    counts. It must raise, not absorb."""
    with pytest.raises(ValueError, match="repeats"):
        partner_view_from_b(np.array([0, 0, 2]),
                            np.array([0.9, 0.8, 0.7]), 3)


def test_partner_view_from_b_leaves_unmatched_b_columns_at_neg_inf():
    """Only col2 is claimed, so cols 0 and 1 must read -inf and count at NO
    threshold. A default of 0.0 would count them at any theta <= 0."""
    out = partner_view_from_b(np.array([-1, -1, 2]),
                              np.array([np.nan, np.nan, 0.7]), 3)
    assert np.isneginf(out[0]) and np.isneginf(out[1])
    assert out[2] == pytest.approx(0.7)
    assert fraction_at(out, -1e9) == pytest.approx(1.0 / 3.0)


def test_encoder_decoder_partner_worked_example():
    """CE's Hungarian is the identity [0,1,2]; CD's is [1,0,2]. They agree only
    on row 2, so the partner array is [-1, -1, 2] -- hand-computed from the
    header, matching the existing rule-3 similarity fixture."""
    np.testing.assert_array_equal(encoder_decoder_partner(CD, CE), [-1, -1, 2])


def test_encoder_decoder_from_assignments_is_the_single_definition():
    """The two-solve path and the pre-solved path must give the same rule 3.
    They share the delegate by construction; this asserts the delegation is
    actually wired, which is what a refactor breaks silently."""
    partner, sim = encoder_decoder_from_assignments(
        *hungarian_assignment(CD), *hungarian_assignment(CE))
    sim2, agree_frac = encoder_decoder_similarity(CD, CE)
    np.testing.assert_array_equal(partner, encoder_decoder_partner(CD, CE))
    np.testing.assert_allclose(sim, sim2)
    assert agree_frac == pytest.approx(1.0 / 3.0)


def test_hungarian_marks_unassigned_rows_rather_than_silently_dropping_them():
    """A non-square matrix leaves rows unassigned; they must carry -1 / -inf so
    they are visibly absent rather than counted as a match at 0."""
    C = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    partner, sim = hungarian_assignment(C)
    assert (partner == -1).sum() == 1
    assert np.isneginf(sim).sum() == 1
    assert fraction_at(sim, 0.0) == pytest.approx(2.0 / 3.0)
