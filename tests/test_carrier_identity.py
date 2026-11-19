"""Tests for `carrier_identity` -- PREREGISTRATION amendment AY section 2b.

THE FIXTURE PRINCIPLE THIS FILE IS BUILT ON (gotcha 29's corollary). Three of
nine alignment mutations once killed nothing because the fixture's arms shared
one seed order. So every seed-axis fixture here uses a LEXICOGRAPHIC AP order
against NUMERIC checkpoint seeds, with the two genuinely different -- a fixture
in which `seed_2` happens to sit at index 2 cannot test an alignment.

And the bit-identity fixture's wrong-order array has the SAME TOTAL as the
right one, which is the only way to prove the check sees what a total cannot
(gotcha 27).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from src.analysis import carrier_identity as ci


# ------------------------------------------------------------- the seed axis

def test_seed_label_to_int_refuses_rather_than_guessing():
    assert ci.seed_label_to_int("seed_17") == 17
    for bad in ("17", "s17", "seed17", ""):
        with pytest.raises(ValueError):
            ci.seed_label_to_int(bad)


def test_order_decoders_to_permutes_LEXICOGRAPHIC_against_NUMERIC():
    """The discriminating case: AP order is 0, 1, 10, 2 -- the real artifacts'
    relationship, where positions 0 and 1 coincide and the rest do not.

    THE PATH NAMES ARE DELIBERATELY UNRELATED TO THE SEEDS. An earlier version
    named them `seed_<n>`, and sorting those paths as STRINGS reproduces the
    lexicographic answer exactly -- so the fixture could not tell correct
    ordering from path-sorting, and the mutation that keys on the path survived
    it. A fixture whose two orders coincide cannot test an alignment (gotcha
    29's corollary). Here path order and label order agree nowhere.
    """
    ap_labels = ["seed_0", "seed_1", "seed_10", "seed_2"]
    paths = {Path("/ck/d"): 0, Path("/ck/c"): 1,
             Path("/ck/b"): 10, Path("/ck/a"): 2}
    got = ci.order_decoders_to(ap_labels, paths)
    assert [paths[p] for p in got] == [0, 1, 10, 2]
    assert [p.name for p in got] == ["d", "c", "b", "a"]
    # Path-sorted order would be a, b, c, d -> seeds 2, 10, 1, 0. Asserting the
    # difference explicitly is what makes the fixture discriminating.
    assert [p.name for p in got] != sorted(p.name for p in paths)


def test_order_decoders_to_uses_META_seed_not_the_filename():
    """A path named seed_9 whose meta says 3 must be placed at 3. If the
    function read the filename this fixture would place it wrongly."""
    ap_labels = ["seed_3", "seed_4"]
    paths = {Path("/ck/seed_9"): 3, Path("/ck/seed_4"): 4}
    got = ci.order_decoders_to(ap_labels, paths)
    assert [p.name for p in got] == ["seed_9", "seed_4"]


@pytest.mark.parametrize("labels,paths,token", [
    (["seed_0", "seed_1"], {Path("a"): 0, Path("b"): 0}, "twice"),
    (["seed_0", "seed_5"], {Path("a"): 0, Path("b"): 1}, "no checkpoint"),
    (["seed_0"], {Path("a"): 0, Path("b"): 1}, "does not name"),
])
def test_order_decoders_to_is_fatal_on_every_mismatch(labels, paths, token):
    with pytest.raises(SystemExit) as e:
        ci.order_decoders_to(labels, paths)
    assert token in str(e.value)


# --------------------------------------------------------- partner matrices

def _unit(cols: list[list[float]]) -> torch.Tensor:
    W = torch.tensor(cols, dtype=torch.float32).T          # (d, n)
    return W / W.norm(dim=0, keepdim=True)


def test_mnn_partner_matrix_finds_the_mutual_pair_and_only_it():
    # Seed A: three features. Seed B: three features. A0 and B0 are identical
    # (cos 1); A1's best is B0 too, but B0's best is A0, so A1 is NOT mutual.
    Wa = _unit([[1, 0, 0], [0.9, 0.44, 0], [0, 0, 1]])
    Wb = _unit([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    partner, sim = ci.mnn_partner_matrix([Wa, Wb], theta=0.7)
    assert partner[0, 1, 0] == 0          # A0 <-> B0, mutual, cos = 1
    assert partner[0, 1, 1] == -1         # A1 not mutual
    assert partner[0, 1, 2] == 2          # A2 <-> B2, mutual, cos = 1
    # Symmetry: MNN is a symmetric relation.
    assert partner[1, 0, 0] == 0
    assert partner[1, 0, 2] == 2
    assert np.isneginf(sim[0, 1, 1])


def test_mnn_partner_matrix_respects_theta():
    """A mutual pair below theta must be -1. Without this the threshold could
    be ignored entirely and every other assertion here would still pass."""
    Wa = _unit([[1.0, 0.0]])
    Wb = _unit([[0.6, 0.8]])              # cos = 0.6
    lo, _ = ci.mnn_partner_matrix([Wa, Wb], theta=0.5)
    hi, _ = ci.mnn_partner_matrix([Wa, Wb], theta=0.7)
    assert lo[0, 1, 0] == 0
    assert hi[0, 1, 0] == -1


# -------------------------------------------------------------- the carriers

def test_carriers_from_ap_reports_ties_rather_than_hiding_them():
    ap = np.zeros((1, 3, 2), dtype=np.float32)
    ap[0, :, 0] = [0.1, 0.9, 0.2]         # a clear winner
    ap[0, :, 1] = [0.5, 0.5, 0.5]         # a three-way tie
    carrier, ties = ci.carriers_from_ap(ap)
    assert carrier[0, 0] == 1
    assert ties["n_cells_with_tied_argmax"] == 1
    assert ties["max_tie_multiplicity"] == 3
    assert ties["share_tied"] == pytest.approx(0.5)


def test_firing_bin_uses_the_standing_strata_edges():
    fr = np.array([[0.0, 5e-5, 2e-4, 5e-3, 0.9]])
    b = ci.firing_bin(fr)
    # BIN_EDGES = [0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 5e-1, 1+eps]
    assert b.tolist() == [[0, 0, 1, 4, 8]]
    assert b.max() <= len(ci.BIN_EDGES) - 2


# ----------------------------------------------------------------- the rates

def _hand_case():
    """A case small enough to compute by hand, and computed by hand below.

    2 seeds, 2 concepts, 5 features.
      detected: concept 0 in both seeds; concept 1 in seed 0 only
      carriers: seed0 -> [0, 1];  seed1 -> [0, 4]
      partners (seed0 <-> seed1): 0<->0, 1<->1, 3<->3; features 2, 4 unmatched
      firing bins: both seeds [0, 0, 0, 1, 1]  ->  bin 0 = {0,1,2}, bin 1 = {3,4}

    THE STRATUM IS THREE FEATURES WIDE AND ONE OF THEM IS UNMATCHED, ON
    PURPOSE. An earlier version used a two-feature stratum in which both
    features matched, and there `hit/n` and `(hit-1)/(n-1)` are both exactly
    1.0 -- so the fixture could not tell whether the carrier was excluded from
    its own baseline, and that mutation survived it. Here the two differ:
    2/3 = 0.667 without the exclusion against (2-1)/(3-1) = 0.5 with it.
    """
    carrier = np.array([[0, 1], [0, 4]])
    detected = np.array([[True, True], [True, False]])
    partner = np.full((2, 2, 5), -1, dtype=np.int32)
    for a, b in ((0, 1), (1, 0)):
        partner[a, b] = np.array([0, 1, -1, 3, -1], dtype=np.int32)
    fbin = np.array([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]])
    return carrier, detected, partner, fbin


def test_carrier_rates_match_the_hand_computation():
    r = ci.carrier_rates(*_hand_case())

    # Only concept 0 is detected in both seeds, over the single seed pair.
    assert r["n_concept_seedpair_cells_evaluated"] == 1
    assert r["n_unordered_seed_pairs"] == 1

    # carrier[0][c0] = 0; partner[0,1,0] = 0 = carrier[1][c0] -> B and A both 1.
    assert r["rate_B_carrier_match"] == pytest.approx(1.0)
    assert r["rate_A_carrier_has_any_partner"] == pytest.approx(1.0)
    assert r["conditional_B_given_A"] == pytest.approx(1.0)

    # Ambient over the off-diagonal: 3 of 5 in each direction -> 6/10.
    assert r["ambient_match_rate_same_pair_set"] == pytest.approx(0.6)

    # The carrier sits in bin 0 = {0, 1, 2}; features 0 and 1 match, 2 does not.
    # EXCLUDING the carrier itself: (2 - 1) / (3 - 1) = 0.5, NOT 2/3.
    assert r["firing_rate_matched_ambient"] == pytest.approx(0.5)
    assert r["firing_rate_matched_ambient"] != pytest.approx(2.0 / 3.0)
    assert r["n_carriers_without_usable_stratum_baseline"] == 0
    assert r["ratios"]["B_over_firing_matched"] == pytest.approx(2.0)


def test_carrier_rates_separate_B_from_A_when_the_partner_is_the_WRONG_feature():
    """The load-bearing distinction. The carrier HAS a partner (A = 1) but the
    partner is not the other seed's carrier (B = 0). A fixture where the two
    coincide cannot test that they are different quantities."""
    carrier, detected, partner, fbin = _hand_case()
    carrier[1, 0] = 2                    # seed 1's carrier is now feature 2
    r = ci.carrier_rates(carrier, detected, partner, fbin)
    assert r["rate_A_carrier_has_any_partner"] == pytest.approx(1.0)
    assert r["rate_B_carrier_match"] == pytest.approx(0.0)
    assert r["conditional_B_given_A"] == pytest.approx(0.0)


def test_carrier_rates_skip_concepts_not_detected_in_both():
    """Concept 1 is detected only in seed 0, so it contributes no cell and its
    per-concept rate is undefined rather than 0."""
    r = ci.carrier_rates(*_hand_case())
    pc = r["_per_concept_values"]
    assert pc[0] == pytest.approx(1.0)
    assert np.isnan(pc[1])
    assert r["per_concept_carrier_match_rate"][
        "n_concepts_with_at_least_one_pair"] == 1


def test_carrier_rates_report_an_unusable_singleton_stratum():
    """A stratum with one feature has no baseline once the carrier itself is
    excluded. It must be counted, not silently divided by zero."""
    carrier, detected, partner, fbin = _hand_case()
    fbin = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])  # every bin a singleton
    r = ci.carrier_rates(carrier, detected, partner, fbin)
    assert r["n_carriers_without_usable_stratum_baseline"] == 1
    assert np.isnan(r["firing_rate_matched_ambient"])


# ------------------------------------------------------------- the verdicts

def test_score_predictions_are_DERIVED_both_ways():
    base = {"rate_B_carrier_match": 0.9,
            "rate_A_carrier_has_any_partner": 0.95,
            "firing_rate_matched_ambient": 0.5,
            "per_concept_carrier_match_rate": {"share_above_0_9": 0.10}}
    hi = ci.score_predictions(base)
    assert hi["AY_P1"]["outcome"] == "EXCEEDS"
    assert hi["AY_P2"]["outcome"] == "UNDER_40_PERCENT"

    low = dict(base, rate_B_carrier_match=0.2,
               per_concept_carrier_match_rate={"share_above_0_9": 0.55})
    lo = ci.score_predictions(low)
    assert lo["AY_P1"]["outcome"] == "DOES_NOT_EXCEED"
    assert lo["AY_P2"]["outcome"] == "AT_OR_ABOVE_40_PERCENT"


def test_AY_P2_boundary_is_STRICTLY_under_forty_percent():
    base = {"rate_B_carrier_match": 0.1, "rate_A_carrier_has_any_partner": 0.1,
            "firing_rate_matched_ambient": 0.5,
            "per_concept_carrier_match_rate": {"share_above_0_9": 0.40}}
    assert ci.score_predictions(base)["AY_P2"]["outcome"] == \
        "AT_OR_ABOVE_40_PERCENT"


# ------------------------------------------- the acceptance criterion (§2b)

def _counts_ref(tmp_path: Path, data: np.ndarray, **attrs) -> Path:
    p = tmp_path / "ref.h5"
    with h5py.File(p, "w") as hf:
        hf.create_dataset("match_count", data=data)
        base = {"theta": 0.7, "rule": "mnn", "similarity": "decoder_cosine",
                "n_compare_seeds": data.shape[0] - 1,
                "n_anchors": data.shape[0]}
        base.update(attrs)
        for k, v in base.items():
            hf.attrs[k] = v
    return p


def _sim_for(counts_lex: np.ndarray, theta: float) -> np.ndarray:
    """A `mnn_sim` whose `counts_at_theta` reproduces `counts_lex` exactly."""
    n, dim = counts_lex.shape
    sim = np.full((n, n, dim), -np.inf, dtype=np.float32)
    for s in range(n):
        for j in range(dim):
            others = [t for t in range(n) if t != s]
            for t in others[:int(counts_lex[s, j])]:
                sim[s, t, j] = theta + 0.01
    return sim


def test_acceptance_passes_when_the_orders_agree(tmp_path):
    counts = np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]], dtype=np.int16)
    sim = _sim_for(counts, 0.7)
    ref = _counts_ref(tmp_path, counts)
    r = ci.verify_counts_bit_identical(sim, 3, 0.7, ref, np.array([0, 1, 2]))
    assert r["verdict"] == "BIT_IDENTICAL"
    assert r["n_cells_differing"] == 0


def test_acceptance_FAILS_on_a_permutation_with_the_SAME_TOTAL(tmp_path):
    """THE test. A permuted reference has identical shape, dtype and total, so
    none of those can see it. Only the elementwise comparison can, and this
    fixture is built so the total is provably unchanged."""
    counts = np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]], dtype=np.int16)
    permuted = counts[[1, 2, 0]]
    assert permuted.sum() == counts.sum()          # the trap, made explicit
    assert permuted.shape == counts.shape
    sim = _sim_for(counts, 0.7)
    ref = _counts_ref(tmp_path, permuted)
    r = ci.verify_counts_bit_identical(sim, 3, 0.7, ref, np.array([0, 1, 2]))
    assert r["verdict"] == "DIFFERS"
    assert r["bit_identical"] is False
    assert r["total_reference"] == r["total_this_run"]   # and the total agrees


def test_acceptance_applies_the_numeric_permutation_it_is_given(tmp_path):
    """The same computed array, consumed in the OTHER order, must match a
    reference stored in that order. This is what proves the permutation
    argument is used rather than ignored."""
    counts_lex = np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]], dtype=np.int16)
    perm = np.array([2, 0, 1])
    ref = _counts_ref(tmp_path, counts_lex[perm])
    sim = _sim_for(counts_lex, 0.7)
    assert ci.verify_counts_bit_identical(
        sim, 3, 0.7, ref, perm)["verdict"] == "BIT_IDENTICAL"
    # ...and the identity permutation must NOT match it.
    assert ci.verify_counts_bit_identical(
        sim, 3, 0.7, ref, np.array([0, 1, 2]))["verdict"] == "DIFFERS"


def test_acceptance_reports_an_absent_identity_key_as_absent(tmp_path):
    counts = np.array([[1, 0], [0, 1]], dtype=np.int16)
    p = tmp_path / "ref.h5"
    with h5py.File(p, "w") as hf:
        hf.create_dataset("match_count", data=counts)
        hf.attrs["rule"] = "mnn"                    # theta deliberately absent
    r = ci.verify_counts_bit_identical(_sim_for(counts, 0.7), 2, 0.7, p,
                                       np.array([0, 1]))
    assert r["verdict"] == "IDENTITY_MISMATCH"
    assert r["reference_identity"]["theta"] == "<ABSENT>"
    assert any("ABSENT" in s for s in r["identity_problems"])


def test_acceptance_refuses_a_reference_at_a_different_theta(tmp_path):
    counts = np.array([[1, 0], [0, 1]], dtype=np.int16)
    ref = _counts_ref(tmp_path, counts, theta=0.8)
    r = ci.verify_counts_bit_identical(_sim_for(counts, 0.7), 2, 0.7, ref,
                                       np.array([0, 1]))
    assert r["verdict"] == "IDENTITY_MISMATCH"
