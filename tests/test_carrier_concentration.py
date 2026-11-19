"""Tests for `carrier_concentration` -- PREREGISTRATION amendment BB section 3.

THE FIXTURES ARE BUILT TO DISCRIMINATE, which for this module means:

  * **the outcome grid must be exercised in ALL FOUR cells**, including the
    one declared impossible. A grid with an unmapped cell is the defect X2 and
    AC9 both hit, and the one time this project left a cell unmapped the data
    landed in it.
  * **`had_something_to_remove` must NOT be a synonym for `ENRICHED`.** A
    DEPLETED covariate is also one the two groups do not share. There is a
    case for each of the three bands, and the depleted one is the whole point.
  * **the acceptance criteria must fail on a moved value AND on an absent
    key**, in separate cases with separate expected verdicts.
  * **`support_distribution`'s generalisation must be a LABEL and not a
    switch**: the same inputs under two `value_name`s must produce identical
    NUMBERS under differently-named keys. A test that only checked the default
    could not see a `value_name` that changed the arithmetic.
  * **cell usability must count cells that are BELOW the bar**, so the fixture
    contains some. A fixture where every cell is well-occupied cannot exhibit
    the condition (gotcha 16).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.analysis import carrier_concentration as cc
from src.analysis import carrier_support as cs


# ------------------------------------------------- the label, not a switch

def _tiny_carrier_fixture(seed=0):
    """4 seeds x 40 features, 6 concepts, 3 firing bins by construction."""
    rng = np.random.default_rng(seed)
    n_seeds, n_feat, n_con = 4, 40, 6
    carrier = rng.integers(0, n_feat, size=(n_seeds, n_con))
    detected = np.ones((n_seeds, n_con), dtype=bool)
    value = rng.uniform(0.0, 5.0, size=(n_seeds, n_feat))
    fbin = np.tile(np.repeat(np.arange(4), 10), (n_seeds, 1))
    return carrier, detected, value, fbin


def test_value_name_renames_keys_and_changes_no_number():
    """THE LOAD-BEARING TEST FOR THE GENERALISATION. If `value_name` were a
    switch rather than a label, BB's concentration answer would not be
    comparable with BA10's homology answer -- which is the entire reason the
    two go through one function."""
    carrier, detected, value, fbin = _tiny_carrier_fixture()
    a = cs.support_distribution(carrier, detected, value, fbin, 4,
                                value_name="d_fs")
    b = cs.support_distribution(carrier, detected, value, fbin, 4,
                                value_name="concentration")
    for k in ("median_percentile", "mean_percentile", "share_above_50th",
              "iqr_percentile", "n_carrier_cells", "p05", "p95",
              "per_seed_median_percentile"):
        assert a[k] == b[k], k
    sa = [r for r in a["by_firing_stratum"] if r.get("status") == "COMPUTED"]
    sb = [r for r in b["by_firing_stratum"] if r.get("status") == "COMPUTED"]
    assert len(sa) == len(sb) and sa
    for ra, rb in zip(sa, sb):
        assert ra["median_percentile_in_peer_d_fs"] == \
            rb["median_percentile_in_peer_concentration"]
        assert ra["shift_nats_mean_over_seeds"] == \
            rb["shift_nats_mean_over_seeds"]
        # ... and the key really is renamed, not duplicated
        assert "median_percentile_in_peer_concentration" not in ra
        assert "median_percentile_in_peer_d_fs" not in rb


def test_the_default_value_name_is_d_fs_so_BA10_is_unchanged():
    """BA10's artifact was written before `value_name` existed. If the default
    moved, acceptance criterion 4 would fail for a reason that is not a
    defect -- and worse, could be 'fixed' by widening the check."""
    carrier, detected, value, fbin = _tiny_carrier_fixture()
    d = cs.support_distribution(carrier, detected, value, fbin, 4)
    s = [r for r in d["by_firing_stratum"] if r.get("status") == "COMPUTED"][0]
    assert "median_percentile_in_peer_d_fs" in s
    assert d["covariate"] == "d_fs"


# ------------------------------------------------- the enrichment reading

@pytest.mark.parametrize("median,verdict,had", [
    (72.0, "ENRICHED", True),
    (55.0, "ENRICHED", True),            # the band edge is inclusive
    (50.5, "NOT_MATERIALLY_ENRICHED", False),
    (45.0, "NOT_MATERIALLY_ENRICHED", False),
    (30.0, "DEPLETED", True),            # depleted STILL has something to remove
])
def test_every_enrichment_band_is_reachable_and_the_binary_is_not_ENRICHED(
        median, verdict, had):
    """`had_something_to_remove` is the grid's key and it is NOT a synonym for
    ENRICHED: a DEPLETED covariate is one the two groups differ on, so
    stratifying on it can also move the excess. Reading the grid off
    'enriched' alone would leave the depleted case unmapped."""
    out = cc.enrichment_verdict(
        {"status": "COMPUTED", "median_percentile": median,
         "iqr_percentile": [0, 100], "share_above_50th": 0.5}, "concentration")
    assert out["verdict"] == verdict
    assert out["had_something_to_remove"] is had


def test_an_uncomputed_distribution_is_NOT_COMPUTED_not_a_false():
    """`False` would read as 'checked and absent' (gotcha 25b)."""
    out = cc.enrichment_verdict({"status": "NO_USABLE_CARRIER_CELLS"}, "x")
    assert out["verdict"] == "NOT_COMPUTED"
    assert out["had_something_to_remove"] is None
    assert out["median_percentile"] is None


def test_the_bands_are_BA10s_and_say_so():
    out = cc.enrichment_verdict(
        {"status": "COMPUTED", "median_percentile": 50.0,
         "iqr_percentile": [0, 100], "share_above_50th": 0.5}, "concentration")
    assert "BA7.1" in out["bands_declared"]
    assert cc.ENRICHED_AT == 55.0 and cc.DEPLETED_BELOW == 45.0


# ------------------------------------------------------- the outcome grid

def _enr(had, verdict="ENRICHED"):
    return {"covariate": "concentration", "verdict": verdict,
            "had_something_to_remove": had}


def _arm(excess):
    return {"excess_A_minus_matched_pp": excess}


def test_all_four_cells_of_the_grid_are_reachable():
    """TOTAL over the 2x2, including the cell declared impossible. A rule
    needs every cell mapped (gotcha 24)."""
    assert cc.outcome_grid(_enr(True), _arm(7.22), _arm(2.0), "a")["cell"] \
        == "ENRICHED_AND_SHRINKS"
    assert cc.outcome_grid(_enr(True), _arm(7.22), _arm(7.0), "a")["cell"] \
        == "ENRICHED_AND_SURVIVES"
    assert cc.outcome_grid(_enr(False, "NOT_MATERIALLY_ENRICHED"),
                           _arm(7.22), _arm(7.0), "a")["cell"] \
        == "NOT_ENRICHED_AND_SURVIVES"
    impossible = cc.outcome_grid(_enr(False, "NOT_MATERIALLY_ENRICHED"),
                                 _arm(7.22), _arm(2.0), "a")
    assert impossible["cell"] == \
        "NOT_ENRICHED_AND_SHRINKS__DECLARED_IMPOSSIBLE"
    assert "INCONSISTENCY" in impossible["reading"]


def test_the_grid_does_not_manufacture_a_cell_from_an_undefined_input():
    assert cc.outcome_grid(_enr(None), _arm(7.22), _arm(2.0), "a")["cell"] \
        == "UNDEFINED"
    assert cc.outcome_grid(_enr(True), _arm(0.0), _arm(2.0), "a")["cell"] \
        == "UNDEFINED"
    assert cc.outcome_grid(_enr(True), _arm(float("nan")), _arm(2.0),
                           "a")["cell"] == "UNDEFINED"


def test_the_shrink_cut_is_AZ_P1s_own_and_is_recorded():
    """A cut invented for this run, after one covariate had been measured,
    would be a cut chosen knowing what it must say."""
    g = cc.outcome_grid(_enr(True), _arm(7.22), _arm(2.0), "a")
    assert g["shrink_cut"] == 0.5
    assert "AZ-P1" in g["shrink_cut_provenance"]


def test_a_NEGATIVE_matched_excess_counts_as_shrinking():
    """A collapse past zero is the strongest form of 'shrinks materially' and
    must not fall out of the predicate's domain."""
    g = cc.outcome_grid(_enr(True), _arm(7.22), _arm(-1.0), "a")
    assert g["shrinks_materially"] is True
    assert g["cell"] == "ENRICHED_AND_SHRINKS"


# -------------------------------------------------------- cell usability

def test_cell_usability_counts_cells_BELOW_the_bar():
    """The fixture deliberately contains empty and singleton cells; one where
    every cell is well-occupied could not exhibit the condition."""
    cell = np.array([[0, 0, 0, 1, 2, 2],
                     [0, 1, 1, 1, 3, 3]], dtype=np.int64)
    out = cc.cell_usability(cell, 5, min_usable=2)
    # seed 0: counts [3,1,2,0,0] -> 3 below 2 ; seed 1: [1,3,0,2,0] -> 3 below
    assert out["n_seed_cells_below_usable"] == 6
    assert out["n_features_in_unusable_cells"] == 2
    assert out["min_occupancy"] == 0
    assert out["max_occupancy"] == 3


def test_cell_usability_reports_zero_when_every_cell_is_occupied():
    """The check must be capable of more than one outcome (gotcha 25)."""
    cell = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int64)
    out = cc.cell_usability(cell, 2, min_usable=2)
    assert out["n_seed_cells_below_usable"] == 0


# --------------------------------------------------- ACCEPTANCE 3 and 4

def _ba_doc(dist, single, double):
    return {
        "identity": {"measurement": "test"},
        "section_3a_double_matching": {
            "single_arm_firing_only": single,
            "double_arm_firing_and_support": double,
        },
        "section_BA9_carrier_support_vs_firing_matched_peers": {
            "distribution": dist},
    }


def _full_arm(excess=7.22):
    return {"n_cells": 9, "n_concept_seedpair_cells_evaluated": 11907,
            "rate_B_carrier_match": 0.4, "rate_A_carrier_has_any_partner": 0.56,
            "conditional_B_given_A": 0.7,
            "ambient_match_rate_same_pair_set": 0.49,
            "matched_ambient": 0.4895,
            "excess_A_minus_matched_pp": excess, "ratio_A_over_matched": 1.14,
            "n_carriers_without_usable_cell_baseline": 0,
            "n_carrier_cells_with_a_usable_baseline": 11907}


def test_acceptance_4_passes_on_an_exact_match(tmp_path):
    carrier, detected, value, fbin = _tiny_carrier_fixture()
    dist = cs.support_distribution(carrier, detected, value, fbin, 4)
    p = tmp_path / "ba.json"
    p.write_text(json.dumps(_ba_doc(dist, _full_arm(), _full_arm())))
    assert cc.reproduces_ba9(dist, p)["verdict"] == "REPRODUCES_BA9_EXACTLY"


def test_acceptance_4_fails_on_a_median_moved_in_the_last_bit(tmp_path):
    carrier, detected, value, fbin = _tiny_carrier_fixture()
    dist = cs.support_distribution(carrier, detected, value, fbin, 4)
    ref = json.loads(json.dumps(dist))
    ref["median_percentile"] = np.nextafter(ref["median_percentile"], np.inf)
    p = tmp_path / "ba.json"
    p.write_text(json.dumps(_ba_doc(ref, _full_arm(), _full_arm())))
    out = cc.reproduces_ba9(dist, p)
    assert out["verdict"] == "DIFFERS"
    assert any("median_percentile" in s for s in out["problems"])


def test_acceptance_4_fails_on_a_moved_STRATUM_value(tmp_path):
    """A pooled median can agree while a stratum disagrees. Checking only the
    pooled number would be a check on a total (gotcha 27)."""
    carrier, detected, value, fbin = _tiny_carrier_fixture()
    dist = cs.support_distribution(carrier, detected, value, fbin, 4)
    ref = json.loads(json.dumps(dist))
    for r in ref["by_firing_stratum"]:
        if r.get("status") == "COMPUTED":
            r["shift_nats_mean_over_seeds"] += 1e-9
            break
    p = tmp_path / "ba.json"
    p.write_text(json.dumps(_ba_doc(ref, _full_arm(), _full_arm())))
    assert cc.reproduces_ba9(dist, p)["verdict"] == "DIFFERS"


def test_acceptance_4_reports_an_ABSENT_key_as_a_problem(tmp_path):
    carrier, detected, value, fbin = _tiny_carrier_fixture()
    dist = cs.support_distribution(carrier, detected, value, fbin, 4)
    ref = json.loads(json.dumps(dist))
    ref.pop("share_above_50th")
    p = tmp_path / "ba.json"
    p.write_text(json.dumps(_ba_doc(ref, _full_arm(), _full_arm())))
    out = cc.reproduces_ba9(dist, p)
    assert out["verdict"] == "DIFFERS"
    assert any("ABSENT" in s for s in out["problems"])


def test_acceptance_4_excludes_the_added_covariate_key(tmp_path):
    """`covariate` is the ADDITION. Comparing it against a reference that
    could not have carried it would report a difference for the one thing that
    is supposed to be new."""
    carrier, detected, value, fbin = _tiny_carrier_fixture()
    dist = cs.support_distribution(carrier, detected, value, fbin, 4)
    ref = json.loads(json.dumps(dist))
    ref.pop("covariate")                       # BA10's artifact has no such key
    p = tmp_path / "ba.json"
    p.write_text(json.dumps(_ba_doc(ref, _full_arm(), _full_arm())))
    assert cc.reproduces_ba9(dist, p)["verdict"] == "REPRODUCES_BA9_EXACTLY"


def test_acceptance_3_passes_and_fails_in_both_directions(tmp_path):
    p = tmp_path / "ba.json"
    p.write_text(json.dumps(_ba_doc({"status": "COMPUTED"},
                                    _full_arm(7.22), _full_arm(7.51))))
    assert cc.reproduces_ba_arm(_full_arm(7.22), p, "single")["verdict"] == \
        "REPRODUCES_BA_SINGLE_EXACTLY"
    assert cc.reproduces_ba_arm(_full_arm(7.51), p, "double")["verdict"] == \
        "REPRODUCES_BA_DOUBLE_EXACTLY"
    assert cc.reproduces_ba_arm(_full_arm(7.20), p, "single")["verdict"] == \
        "DIFFERS"


def test_acceptance_3_reports_an_ABSENT_arm_rather_than_passing(tmp_path):
    p = tmp_path / "ba.json"
    p.write_text(json.dumps({"section_3a_double_matching": {}}))
    out = cc.reproduces_ba_arm(_full_arm(), p, "single")
    assert out["verdict"] == "DIFFERS"


# ------------------------------------------------------------- BB-P2

def test_bb_p2_is_total_and_records_its_category_and_standing():
    e = cc.enrichment_verdict(
        {"status": "COMPUTED", "median_percentile": 80.0,
         "iqr_percentile": [0, 100], "share_above_50th": 0.9}, "concentration")
    out = cc.score_bb_p2(e)
    assert out["outcome"] == "MATERIALLY_ENRICHED"
    assert "NOT SEALED" in out["standing"] and "NOT TALLIED" in out["standing"]
    assert "directional mechanism" in out["category_if_it_were_tallied"]

    e2 = cc.enrichment_verdict(
        {"status": "COMPUTED", "median_percentile": 20.0,
         "iqr_percentile": [0, 100], "share_above_50th": 0.1}, "concentration")
    assert cc.score_bb_p2(e2)["outcome"] == "NOT_MATERIALLY_ENRICHED"
    assert cc.score_bb_p2(cc.enrichment_verdict({"status": "X"}, "c"))[
        "outcome"] is None


def test_the_stopping_rule_is_stated_in_the_module_not_only_in_the_brief():
    assert "no further attribution attempts" in cc.__doc__.lower()
