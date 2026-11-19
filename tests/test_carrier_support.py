"""Tests for `carrier_support` -- PREREGISTRATION amendment AZ section 11.

THE FIXTURE PRINCIPLE (gotcha 16, restated by AY15's two survivors). A fixture
that cannot exhibit the disputed condition cannot test it. So:

  * the support-binning fixture gives the two firing strata **disjoint** `d_fs`
    ranges, because a global quantile and a within-stratum quantile agree
    whenever the ranges overlap symmetrically -- a fixture with overlapping
    ranges cannot tell them apart;
  * the seed-alignment fixture uses a LEXICOGRAPHIC order against a NUMERIC
    one with the two genuinely different;
  * the percentile fixture contains ties, because the half-tie convention is
    invisible without them.

AND THE STRONGEST TEST IN THE FILE IS THE CROSS-MODULE ONE: with the single
(firing-only) cell index, `carrier_rates_by_cell` must return exactly what
`carrier_identity.carrier_rates` returns on the same inputs. That is the unit
form of the job's second acceptance criterion, and it is what makes any
difference between the two arms attributable to the stratification.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis import carrier_identity as ci
from src.analysis import carrier_support as cs


# --------------------------------------------------------------- the strata

def test_support_bins_are_taken_WITHIN_each_stratum_not_globally():
    """The discriminating fixture: stratum 0's `d_fs` is entirely below
    stratum 1's. Under a GLOBAL quantile every stratum-0 feature lands in
    support bin 0 and every stratum-1 feature in bin 2, so the second axis
    would carry no information inside a stratum while looking like it did.
    """
    fbin = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])
    d_fs = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                      100., 101., 102., 103., 104., 105.]])
    sb = cs.support_bin_within(fbin, d_fs, 3)
    for b in (0, 1):
        got = sorted(set(sb[0][fbin[0] == b].tolist()))
        assert got == [0, 1, 2], f"stratum {b} was not split: {got}"
    # And the low-d_fs half of EACH stratum is in support bin 0.
    assert sb[0][0] == 0 and sb[0][6] == 0
    assert sb[0][5] == 2 and sb[0][11] == 2


def test_support_bin_within_refuses_a_shape_mismatch():
    with pytest.raises(ValueError):
        cs.support_bin_within(np.zeros((1, 4), dtype=int), np.zeros((1, 5)), 3)


def test_cell_index_with_no_support_axis_IS_the_firing_bin():
    """The single arm must be the same object AY7 used, not a re-derivation."""
    fbin = np.array([[0, 3, 8], [2, 2, 5]])
    assert np.array_equal(cs.cell_index(fbin, None, 3), fbin)


def test_cell_index_separates_every_firing_x_support_pair():
    fbin = np.array([[0, 0, 1, 1]])
    sbin = np.array([[0, 2, 0, 2]])
    cell = cs.cell_index(fbin, sbin, 3)
    assert cell.tolist() == [[0, 2, 3, 5]]
    assert len(set(cell.ravel().tolist())) == 4


# ------------------------------------------------------------- the rates

def _tiny_case():
    """Two seeds, three concepts, four features. Hand-checkable.

    partner[0,1] maps feature 0 -> 0 (the other seed's carrier for concept 0),
    feature 1 -> 3 (a partner, but NOT the other carrier), feature 2 -> -1
    (no partner at all). So over the three concepts B = 1, A = 2.
    """
    carrier = np.array([[0, 1, 2],
                        [0, 2, 3]], dtype=np.int64)
    detected = np.ones((2, 3), dtype=bool)
    partner = np.full((2, 2, 4), -1, dtype=np.int32)
    partner[0, 1] = np.array([0, 3, -1, 1], dtype=np.int32)
    partner[1, 0] = np.array([0, 3, -1, 1], dtype=np.int32)
    fbin = np.zeros((2, 4), dtype=np.int64)
    return carrier, detected, partner, fbin


def test_carrier_rates_by_cell_matches_the_hand_computation():
    carrier, detected, partner, fbin = _tiny_case()
    r = cs.carrier_rates_by_cell(carrier, detected, partner, fbin, 1)
    assert r["n_concept_seedpair_cells_evaluated"] == 3
    assert r["rate_B_carrier_match"] == pytest.approx(1 / 3)
    assert r["rate_A_carrier_has_any_partner"] == pytest.approx(2 / 3)
    assert r["conditional_B_given_A"] == pytest.approx(1 / 2)
    # 3 of 4 features in seed 0 have a partner in seed 1; each carrier is
    # EXCLUDED from its own baseline, so a carrier that itself has a partner
    # sees (3-1)/(4-1) and one that does not sees 3/3.
    assert r["matched_ambient"] == pytest.approx(
        (2 / 3 + 2 / 3 + 3 / 3) / 3)


def test_carrier_rates_by_cell_separates_B_from_A():
    """Feature 1's partner is feature 3 and the other seed's carrier is 2, so a
    B that collapsed into A would report 2/3 rather than 1/3."""
    carrier, detected, partner, fbin = _tiny_case()
    r = cs.carrier_rates_by_cell(carrier, detected, partner, fbin, 1)
    assert r["rate_B_carrier_match"] < r["rate_A_carrier_has_any_partner"]


def test_single_arm_reproduces_carrier_identity_EXACTLY():
    """THE CROSS-MODULE CHECK. The generalised function with the firing-only
    cell index must return the incumbent's numbers bit for bit -- otherwise a
    difference between the two arms could be the stratification or could be
    anything else that changed in the rewrite."""
    carrier, detected, partner, fbin = _tiny_case()
    fbin = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int64)
    mine = cs.carrier_rates_by_cell(carrier, detected, partner, fbin,
                                    len(cs.BIN_EDGES) - 1)
    theirs = ci.carrier_rates(carrier, detected, partner, fbin)
    for k in ("rate_B_carrier_match", "rate_A_carrier_has_any_partner",
              "conditional_B_given_A", "ambient_match_rate_same_pair_set",
              "n_concept_seedpair_cells_evaluated"):
        assert mine[k] == theirs[k], k
    assert mine["matched_ambient"] == theirs["firing_rate_matched_ambient"]


def test_a_singleton_cell_is_unusable_rather_than_a_division_by_zero():
    """One feature alone in its cell has no baseline once the carrier is
    excluded. It must be COUNTED, not silently contribute a rate."""
    carrier = np.array([[0], [0]], dtype=np.int64)
    detected = np.ones((2, 1), dtype=bool)
    partner = np.full((2, 2, 2), -1, dtype=np.int32)
    partner[0, 1] = np.array([1, 0], dtype=np.int32)
    partner[1, 0] = np.array([1, 0], dtype=np.int32)
    cell = np.array([[0, 1], [0, 1]], dtype=np.int64)     # each alone
    r = cs.carrier_rates_by_cell(carrier, detected, partner, cell, 2)
    assert r["n_carriers_without_usable_cell_baseline"] == 1
    assert r["n_carrier_cells_with_a_usable_baseline"] == 0
    assert np.isnan(r["matched_ambient"])


def test_undetected_concepts_contribute_no_cells():
    carrier, detected, partner, fbin = _tiny_case()
    detected = detected.copy()
    detected[1, 2] = False               # concept 2 not detected in seed 1
    r = cs.carrier_rates_by_cell(carrier, detected, partner, fbin, 1)
    assert r["n_concept_seedpair_cells_evaluated"] == 2


# ---------------------------------------------------------- the comparison

def test_excess_comparison_is_total_over_the_ratio_domain():
    under = cs.excess_comparison({"excess_A_minus_matched_pp": 8.0},
                                 {"excess_A_minus_matched_pp": 3.0})
    assert under["outcome"] == "UNDER_HALF"
    assert under["ratio_double_over_single"] == pytest.approx(0.375)

    over = cs.excess_comparison({"excess_A_minus_matched_pp": 8.0},
                                {"excess_A_minus_matched_pp": 6.0})
    assert over["outcome"] == "HALF_OR_ABOVE"

    # A NEGATIVE double excess is "under half" and is flagged as a collapse.
    neg = cs.excess_comparison({"excess_A_minus_matched_pp": 8.0},
                               {"excess_A_minus_matched_pp": -1.0})
    assert neg["outcome"] == "UNDER_HALF" and neg["collapses_entirely"]

    # A zero single excess makes the ratio undefined; no verdict is invented.
    zero = cs.excess_comparison({"excess_A_minus_matched_pp": 0.0},
                                {"excess_A_minus_matched_pp": 1.0})
    assert zero["ratio_double_over_single"] is None
    assert zero["outcome"].startswith("UNDEFINED")


def test_exactly_half_counts_as_half_or_above():
    """The boundary is named in the predicate ('less than half'), so it has to
    be tested rather than left to whichever comparison was typed."""
    r = cs.excess_comparison({"excess_A_minus_matched_pp": 8.0},
                             {"excess_A_minus_matched_pp": 4.0})
    assert r["outcome"] == "HALF_OR_ABOVE"


# ------------------------------------------------------- carrier positions

def test_percentile_positions_uses_the_half_tie_convention():
    """A value at the population median must sit at 50, not above or below it.
    The fixture is deliberately tied: with 4 features at 1,2,2,3 the value 2
    occupies ranks 1 and 2, so the half-tie position is 100*(1+3)/2/4 = 50."""
    value = np.array([[1.0, 2.0, 2.0, 3.0]])
    carrier = np.array([[1]], dtype=np.int64)
    mask = np.ones((1, 1), dtype=bool)
    p = cs.percentile_positions(carrier, mask, value)
    assert p["median"] == pytest.approx(50.0)


def test_percentile_positions_are_computed_WITHIN_each_seed():
    """Seed 1's values are shifted far above seed 0's. A pooled percentile
    would put every seed-0 carrier at the bottom; a within-seed one puts the
    top feature of each seed at the same place."""
    value = np.array([[0.0, 1.0, 2.0, 3.0],
                      [100., 101., 102., 103.]])
    carrier = np.array([[3], [3]], dtype=np.int64)
    mask = np.ones((2, 1), dtype=bool)
    p = cs.percentile_positions(carrier, mask, value)
    assert p["n_cells"] == 2
    assert p["median"] == pytest.approx(87.5)


def test_percentile_positions_reports_an_empty_population_rather_than_zero():
    p = cs.percentile_positions(np.zeros((2, 1), dtype=np.int64),
                                np.zeros((2, 1), dtype=bool),
                                np.zeros((2, 4)))
    assert p["status"] == "NO_CARRIER_CELLS"


def test_statistic_agreement_reports_a_known_correlation():
    best = np.array([[1.0, 2.0, 3.0, 4.0]])
    M = np.array([[4.0, 3.0, 2.0, 1.0]])
    a = cs.statistic_agreement(best, M)
    assert a["median"] == pytest.approx(-1.0)


# -------------------------------------------------------- the seed alignment

def test_align_to_ap_order_permutes_lexicographic_against_numeric():
    ap = ["seed_0", "seed_1", "seed_10", "seed_2"]
    have = ["0", "1", "2", "10"]                       # numeric order
    perm = cs.align_to_ap_order(have, ap, "fixture")
    assert perm.tolist() == [0, 1, 3, 2]
    # And the permutation is NOT the identity, which is what makes the fixture
    # able to fail (gotcha 29's corollary).
    assert perm.tolist() != list(range(4))


def test_align_to_ap_order_refuses_a_different_seed_set():
    with pytest.raises(SystemExit):
        cs.align_to_ap_order(["0", "1", "2"], ["seed_0", "seed_1", "seed_3"],
                             "fixture")


# ----------------------------------------------------- acceptance criterion 2

def _ay_reference(tmp_path: Path, **over) -> Path:
    rates = {
        "rate_B_carrier_match": 0.25,
        "rate_A_carrier_has_any_partner": 0.5,
        "conditional_B_given_A": 0.5,
        "ambient_match_rate_same_pair_set": 0.4,
        "firing_rate_matched_ambient": 0.45,
        "n_concept_seedpair_cells_evaluated": 8,
    }
    rates.update(over)
    p = tmp_path / "ay.json"
    p.write_text(json.dumps({"identity": {"git_sha": "abc"}, "rates": rates}))
    return p


def _mine():
    return {"rate_B_carrier_match": 0.25,
            "rate_A_carrier_has_any_partner": 0.5,
            "conditional_B_given_A": 0.5,
            "ambient_match_rate_same_pair_set": 0.4,
            "matched_ambient": 0.45,
            "n_concept_seedpair_cells_evaluated": 8}


def test_reproduces_ay_passes_on_an_exact_match(tmp_path):
    r = cs.reproduces_ay(_mine(), _ay_reference(tmp_path))
    assert r["verdict"] == "REPRODUCES_AY7_EXACTLY"
    assert r["problems"] == []


def test_reproduces_ay_fails_on_the_smallest_difference(tmp_path):
    """Compared as EXACT floats: both sides are the same arithmetic on the same
    inputs, so a last-bit difference is a changed computation."""
    ref = _ay_reference(tmp_path, firing_rate_matched_ambient=0.45 + 1e-15)
    r = cs.reproduces_ay(_mine(), ref)
    assert r["verdict"] == "DIFFERS"


def test_reproduces_ay_reports_an_ABSENT_key_as_absent(tmp_path):
    """gotcha 15 / I3: absent must never read as agreement."""
    p = tmp_path / "ay.json"
    p.write_text(json.dumps({"rates": {"rate_B_carrier_match": 0.25}}))
    r = cs.reproduces_ay(_mine(), p)
    assert r["verdict"] == "DIFFERS"
    absent = [c for c in r["comparisons"] if c["reference"] == "<ABSENT>"]
    assert len(absent) == 5
    assert any("ABSENT" in p_ for p_ in r["problems"])


# -------------------------------------------------------------- predictions

def test_score_predictions_are_derived_not_asserted():
    cmp_ = cs.excess_comparison({"excess_A_minus_matched_pp": 8.0},
                                {"excess_A_minus_matched_pp": 1.0})
    pos = {"status": "COMPUTED", "median": 80.0}
    out = cs.score_predictions(cmp_, pos)
    assert out["AZ_P1"]["outcome"] == "UNDER_HALF"
    assert out["AZ_P2"]["outcome"] == "ABOVE_75TH"
    assert "NOT TALLIED" in out["AZ_P2"]["standing"]

    below = cs.score_predictions(cmp_, {"status": "COMPUTED", "median": 75.0})
    assert below["AZ_P2"]["outcome"] == "AT_OR_BELOW_75TH"
    none = cs.score_predictions(cmp_, {"status": "NO_CARRIER_CELLS"})
    assert none["AZ_P2"]["outcome"] is None


def test_feature_best_ap_is_the_max_over_CONCEPTS_not_over_features():
    """The two reductions of the same array, and confusing them is what took
    the first production run down. `observed_best_ap` gives (seed, concept);
    this gives (seed, feature). The fixture is deliberately NON-SQUARE so a
    transposed implementation is a shape error rather than a silent one."""
    ap = np.arange(1 * 3 * 5, dtype=np.float64).reshape(1, 3, 5)
    got = cs.feature_best_ap(ap)
    assert got.shape == (1, 3)
    assert got.tolist() == [[4.0, 9.0, 14.0]]


def test_statistic_agreement_refuses_the_wrong_reduction():
    """Handed the (seed, concept) max against the (seed, feature) predictor it
    must refuse, not broadcast. This is the production failure as a test.

    THE MATCH ON THE MESSAGE IS THE WHOLE TEST, and that is deliberate.
    `spearman` ALSO raises `ValueError` on unequal lengths, so a bare
    `pytest.raises(ValueError)` passes whether or not the guard exists -- it
    did, and the mutation survived it (gotcha 13 extension 3: a test that
    passes for the wrong reason). What the guard adds over the generic failure
    is a message NAMING the two reductions, which is what turns a shape
    traceback into a diagnosis, so the message is what is asserted.
    """
    with pytest.raises(ValueError, match="statistic_agreement needs"):
        cs.statistic_agreement(np.zeros((2, 703)), np.zeros((2, 10240)))


# ============================================================================
# BA9 -- the carrier support distribution (amendment BA section 7.1)
# ============================================================================
# THE DISCRIMINATING FIXTURE IS `test_the_matched_arm_sees_what_the_pooled_one
# _cannot`. It is built so the two answers DISAGREE: carriers sit at exactly
# the median of their own firing bin (percentile 50, not enriched) while the
# pooled comparison reads them as enriched purely because they are
# concentrated in the high-firing bin, where `d_fs` is higher for everyone.
# A fixture whose bins share one `d_fs` range cannot exhibit that.

def test_percentile_rank_within_uses_the_half_tie_convention():
    """A value at the pool's median sits at 50, not above or below it."""
    pool = np.array([0.0, 1.0, 1.0, 1.0, 2.0])
    got = cs.percentile_rank_within(pool, np.array([1.0]))
    assert got[0] == pytest.approx(100.0 * (1 + 4) / 2.0 / 5.0)


def test_percentile_rank_within_returns_EMPTY_for_an_empty_pool():
    """Never a 50, which would read as 'measured, and exactly average'."""
    assert cs.percentile_rank_within(np.zeros(0), np.array([1.0])).size == 0


def test_carrier_feature_indicator_marks_only_DETECTED_concepts_carriers():
    carrier = np.array([[3, 5, 7]])
    detected = np.array([[True, False, True]])
    ind = cs.carrier_feature_indicator(carrier, detected, 10)
    assert ind[0, 3] and ind[0, 7]
    assert not ind[0, 5], "a carrier of an UNDETECTED concept is not a carrier"


def test_the_matched_arm_sees_what_the_pooled_one_cannot():
    """THE FIXTURE THAT DISCRIMINATES. Two firing bins with DISJOINT `d_fs`
    ranges; every carrier sits at its own bin's median; carriers are
    concentrated in the high bin. The matched statistic must read ~50 and the
    composition-confounded one must read far above it."""
    n_feat = 12
    fbin = np.array([[0] * 6 + [1] * 6])
    # bin 0: d_fs 0..5   bin 1: d_fs 100..105 -- disjoint, so composition bites
    d_fs = np.array([[0., 1., 2., 3., 4., 5., 100., 101., 102., 103., 104., 105.]])
    # carriers: features 8 and 9 -- both in the HIGH bin, both mid-pack there
    carrier = np.array([[8, 9]])
    detected = np.array([[True, True]])
    out = cs.support_distribution(carrier, detected, d_fs, fbin, 2)
    assert out["status"] == "COMPUTED"
    matched = out["median_percentile"]
    pooled = out["composition_confounded_pooled"]["median_percentile"]
    assert 25.0 <= matched <= 75.0, (
        f"the matched arm should read near the middle of its own bin, got "
        f"{matched}")
    assert pooled >= 75.0, (
        f"the pooled arm should be dragged up by composition, got {pooled}")
    assert pooled - matched > 20.0, (
        "the fixture does not discriminate: the two arms agree")
    assert out["n_carrier_cells"] == 2


def test_peers_EXCLUDE_carriers_and_that_changes_the_answer():
    """A bin whose top values are ALL carriers. Counted as peers they would
    drag the pool up and the carriers' percentile down."""
    fbin = np.zeros((1, 6), dtype=int)
    d_fs = np.array([[0., 1., 2., 3., 10., 11.]])
    carrier = np.array([[4, 5]])            # the two highest
    detected = np.array([[True, True]])
    out = cs.support_distribution(carrier, detected, d_fs, fbin, 1)
    # peers are features 0..3, so both carriers are above ALL of them -> 100
    assert out["median_percentile"] == pytest.approx(100.0)


def test_a_bin_with_no_peers_is_counted_unusable_not_dropped():
    """Every feature in the bin is a carrier, so there is no peer pool. The
    cells are COUNTED rather than silently vanishing from the denominator."""
    fbin = np.zeros((1, 2), dtype=int)
    d_fs = np.array([[1.0, 2.0]])
    carrier = np.array([[0, 1]])
    detected = np.array([[True, True]])
    out = cs.support_distribution(carrier, detected, d_fs, fbin, 1)
    assert out["status"] == "NO_USABLE_CARRIER_CELLS"
    assert out["n_carrier_cells_without_a_peer_pool"] == 2


def test_the_deduplicated_arm_counts_a_multi_concept_carrier_once():
    fbin = np.zeros((1, 6), dtype=int)
    d_fs = np.array([[0., 1., 2., 3., 4., 5.]])
    carrier = np.array([[5, 5, 5]])          # one feature, three concepts
    detected = np.array([[True, True, True]])
    out = cs.support_distribution(carrier, detected, d_fs, fbin, 1)
    assert out["n_carrier_cells"] == 3
    assert out["deduplicated_per_seed_feature"]["n_carrier_features"] == 1


def test_support_verdict_is_total_and_every_band_is_reachable():
    """Gotcha 24: a rule must be defined at every value the statistic can
    take. Gotcha 25: and it must be able to produce more than one outcome."""
    got = {cs.support_verdict({"status": "COMPUTED",
                               "median_percentile": m})["verdict"]
           for m in (0.0, 44.9, 45.0, 50.0, 54.9, 55.0, 100.0)}
    assert got == {"DEPLETED", "NOT_MATERIALLY_ENRICHED", "ENRICHED"}


def test_support_verdict_band_edges_are_where_BA7_1_declared_them():
    assert cs.support_verdict({"status": "COMPUTED",
                               "median_percentile": 55.0})["verdict"] \
        == "ENRICHED"
    assert cs.support_verdict({"status": "COMPUTED",
                               "median_percentile": 54.999})["verdict"] \
        == "NOT_MATERIALLY_ENRICHED"
    assert cs.support_verdict({"status": "COMPUTED",
                               "median_percentile": 45.0})["verdict"] \
        == "NOT_MATERIALLY_ENRICHED"
    assert cs.support_verdict({"status": "COMPUTED",
                               "median_percentile": 44.999})["verdict"] \
        == "DEPLETED"


def test_the_NOT_ENRICHED_reading_says_AZ11_establishes_NOTHING():
    """The whole point of BA6: an uninformative null must not read as
    support. The reading attached to the middle band has to SAY so."""
    r = cs.support_verdict({"status": "COMPUTED", "median_percentile": 50.0})
    assert "UNATTRIBUTED" in r["az11_reading"]
    assert "NOTHING" in r["az11_reading"]


def test_support_verdict_on_an_uncomputed_distribution_asks_nothing():
    r = cs.support_verdict({"status": "NO_USABLE_CARRIER_CELLS"})
    assert r["verdict"] == "NOT_COMPUTED"
    assert r["median_percentile"] is None


def test_percentile_positions_still_uses_the_shared_helper():
    """The AZ12 statistic was refactored onto `percentile_rank_within`; it
    must not have moved. A pool with ties is the case that would expose it."""
    carrier = np.array([[1]])
    mask = np.array([[True]])
    value = np.array([[0.0, 1.0, 1.0, 1.0, 2.0]])
    got = cs.percentile_positions(carrier, mask, value)
    assert got["median"] == pytest.approx(50.0)
