"""Tests for AK8's geometric half (amendment CH2).

EVERY FIXTURE HERE IS BUILT TO DISCRIMINATE, and where that is not obvious the
test says how. CG's own mutation matrix came back 6 of 8 with BOTH survivors in
tests written specifically to guard the properties they missed -- a determinism
test that compared a function to itself, and an EV_SVD test on a flat spectrum
where both cumulation directions give the same answer. The lesson is that a
test's expectation must be WRITTEN DOWN and the fixture must be able to exhibit
the failure (gotcha 32b).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.analysis.ak8_geometry import (
    MIN_SET_FOR_NULL, compare_to_null, rf_counts_seed_labels,
    summarise_across_seeds,
)
from src.analysis.design3_join import (
    AK8_LOW_M_QUANTILE, AK8_MIN_K, ak8_discovery_mask, ak8_discovery_set,
)


# --------------------------------------------------------------- the selection

def _toy(n_seeds=2, n_feat=40, n_bins=2):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 40, size=(n_seeds, n_feat))
    M = rng.normal(size=(n_seeds, n_feat))
    firing_bin = np.tile(np.arange(n_bins).repeat(n_feat // n_bins),
                         (n_seeds, 1))
    keep = np.ones((n_seeds, n_feat), dtype=bool)
    return y, M, firing_bin, keep


def test_mask_and_summary_agree_because_there_is_one_implementation():
    """The refactor's whole point: the mask and the summary cannot diverge."""
    y, M, fb, keep = _toy()
    fire_rate = np.full(y.shape, 1e-3)
    mask = ak8_discovery_mask(y, M, fb, keep)
    summary = ak8_discovery_set(y, M, fire_rate, fb, keep, 39)
    for s, row in enumerate(summary["per_seed"]):
        assert row["n_discovery"] == int(mask[s].sum())


def test_the_low_M_cut_is_WITHIN_bin_not_global():
    """CH2e's qualifier, as an executable claim.

    THE FIXTURE DISCRIMINATES: bin 1's M values are shifted far ABOVE bin 0's,
    so a GLOBAL bottom-decile cut would select only bin-0 features and bin 1
    would contribute nothing. A within-bin cut must select from BOTH bins.
    """
    n_feat, n_seeds = 40, 1
    y = np.full((n_seeds, n_feat), AK8_MIN_K)          # all pass the r_f cut
    firing_bin = np.zeros((n_seeds, n_feat), dtype=int)
    firing_bin[0, n_feat // 2:] = 1
    M = np.zeros((n_seeds, n_feat))
    M[0, :n_feat // 2] = np.linspace(0.0, 1.0, n_feat // 2)      # bin 0: low
    M[0, n_feat // 2:] = np.linspace(100.0, 101.0, n_feat // 2)  # bin 1: high
    keep = np.ones((n_seeds, n_feat), dtype=bool)

    mask = ak8_discovery_mask(y, M, firing_bin, keep)
    picked_bins = set(firing_bin[0][mask[0]].tolist())
    assert picked_bins == {0, 1}, (
        "a within-bin decile cut must select from every populated bin; "
        f"got {picked_bins}, which is what a GLOBAL cut would give")

    # And the control: a global cut on this same fixture really would pick
    # only bin 0 -- so the assertion above is not vacuous.
    global_cut = np.quantile(M[0], AK8_LOW_M_QUANTILE)
    assert set(firing_bin[0][M[0] <= global_cut].tolist()) == {0}


def test_the_r_f_cut_is_applied_at_all():
    """Dropping the `k >= AK8_MIN_K` conjunct must change the answer."""
    y, M, fb, keep = _toy()
    y = y.copy()
    y[:] = 0                                   # nobody passes the r_f cut
    assert ak8_discovery_mask(y, M, fb, keep).sum() == 0
    y[:] = 39                                  # everybody passes it
    assert ak8_discovery_mask(y, M, fb, keep).sum() > 0


def test_keep_mask_is_respected():
    y, M, fb, keep = _toy()
    y[:] = 39
    keep = keep.copy()
    keep[:, :] = False
    assert ak8_discovery_mask(y, M, fb, keep).sum() == 0


def test_a_bin_with_fewer_than_ten_members_is_skipped_not_crashed():
    y = np.full((1, 12), 39)
    M = np.linspace(0, 1, 12).reshape(1, 12)
    firing_bin = np.zeros((1, 12), dtype=int)
    firing_bin[0, 10:] = 1                    # bin 1 has 2 members -> skipped
    keep = np.ones((1, 12), dtype=bool)
    mask = ak8_discovery_mask(y, M, firing_bin, keep)
    assert set(firing_bin[0][mask[0]].tolist()) == {0}


# ------------------------------------------------------- the null comparison

def _null(mean, lo, hi, sd=0.01, m=96, n=20):
    return {"mean": mean, "sd": sd, "min": lo, "max": hi, "m": m, "n_draws": n}


@pytest.mark.parametrize("obs,expected", [
    (0.95, "ABOVE_ITS_SIZE_MATCHED_NULL"),
    (0.50, "BELOW_ITS_SIZE_MATCHED_NULL"),
    (0.80, "INDISTINGUISHABLE_FROM_ITS_SIZE_MATCHED_NULL"),
    (0.75, "INDISTINGUISHABLE_FROM_ITS_SIZE_MATCHED_NULL"),   # exactly at min
    (0.85, "INDISTINGUISHABLE_FROM_ITS_SIZE_MATCHED_NULL"),   # exactly at max
])
def test_every_branch_of_CH2ds_outcome_table_is_reachable(obs, expected):
    """Gotcha 24: the rule is a TOTAL function of the statistic it reads.

    The boundary cases are included because a rule that is total over the
    interior and undefined at its own edges is the incompleteness gotcha 24
    names.
    """
    assert compare_to_null(obs, _null(0.80, 0.75, 0.85))["verdict"] == expected


def test_the_reported_comparison_is_a_DIFFERENCE_not_a_ratio():
    """Gotcha 33: a ratio needs a shared scale and intervals on BOTH sides."""
    out = compare_to_null(0.9, _null(0.8, 0.75, 0.85))
    assert "difference_from_null_mean" in out
    assert out["difference_from_null_mean"] == pytest.approx(0.1)
    assert not any("ratio" in k for k in out)


# ------------------------------------------------- the across-seed summary

def _row(seed, verdict, diff, status="EVALUATED"):
    return {"seed": seed, "status": status,
            "ER_l1_over_d": {"verdict": verdict,
                             "difference_from_null_mean": diff}}


def test_a_consistent_direction_gets_a_signed_summary():
    rows = [_row(0, "ABOVE_ITS_SIZE_MATCHED_NULL", 0.01),
            _row(1, "ABOVE_ITS_SIZE_MATCHED_NULL", 0.02)]
    out = summarise_across_seeds(rows, "ER_l1_over_d")
    assert out["direction_consistent_across_seeds"] is True
    assert out["mean_difference_from_null_mean"] == pytest.approx(0.015)
    assert out["signed_summary_withheld_because"] is None


def test_an_INCONSISTENT_direction_gets_NO_SIGNED_SUMMARY():
    """CH2d declares this and it is the branch most likely to be quietly lost.

    THE FIXTURE DISCRIMINATES: the two differences are +0.01 and -0.01, whose
    MEAN IS EXACTLY ZERO. A implementation that reported the mean anyway would
    print a plausible '0.0' rather than an obvious error, so the test asserts
    the field is None rather than asserting it is not some number.
    """
    rows = [_row(0, "ABOVE_ITS_SIZE_MATCHED_NULL", +0.01),
            _row(1, "BELOW_ITS_SIZE_MATCHED_NULL", -0.01)]
    out = summarise_across_seeds(rows, "ER_l1_over_d")
    assert out["direction_consistent_across_seeds"] is False
    assert out["mean_difference_from_null_mean"] is None
    assert "INCONSISTENT" in out["signed_summary_withheld_because"]
    assert out["n_above_null"] == 1 and out["n_below_null"] == 1


def test_all_indistinguishable_is_reported_as_a_RESULT_not_as_nothing():
    rows = [_row(s, "INDISTINGUISHABLE_FROM_ITS_SIZE_MATCHED_NULL", 0.0001)
            for s in range(3)]
    out = summarise_across_seeds(rows, "ER_l1_over_d")
    assert out["status"] == "EVALUATED"
    assert out["n_indistinguishable"] == 3
    assert out["n_evaluated"] == 3


def test_NOT_EVALUABLE_seeds_are_excluded_from_the_denominator_and_named():
    rows = [_row(0, "ABOVE_ITS_SIZE_MATCHED_NULL", 0.01),
            {"seed": 1, "status": "NOT_EVALUABLE", "n_discovery": 3}]
    out = summarise_across_seeds(rows, "ER_l1_over_d")
    assert out["n_evaluated"] == 1, (
        "a NOT_EVALUABLE seed is a statement of ignorance and must not be "
        "counted as an observation")


def test_no_evaluable_seed_returns_NOT_EVALUABLE_not_an_empty_success():
    rows = [{"seed": 0, "status": "NOT_EVALUABLE", "n_discovery": 2}]
    out = summarise_across_seeds(rows, "ER_l1_over_d")
    assert out["status"] == "NOT_EVALUABLE"
    assert out["n_evaluated"] == 0


# ------------------------------------------------------- the seed-label guard

class _FakeH5(dict):
    def __init__(self, datasets=None, attrs=None):
        super().__init__(datasets or {})
        self.attrs = attrs or {}


def test_seed_labels_are_parsed_from_a_JSON_attr_not_iterated():
    """Gotcha 20: `list()` of a JSON string is a list of CHARACTERS.

    THE FIXTURE DISCRIMINATES: the labels are two-digit, so a character-wise
    read would give ['[', '1', '0', ...] rather than [10, 11] and the equality
    below would fail loudly instead of coincidentally passing.
    """
    hf = _FakeH5(attrs={"seed_labels": json.dumps([10, 11, 12])})
    assert rf_counts_seed_labels(hf) == [10, 11, 12]


def test_a_missing_label_set_returns_None_so_the_caller_can_REFUSE():
    assert rf_counts_seed_labels(_FakeH5()) is None


def test_min_set_for_null_is_a_named_constant_not_a_literal():
    assert MIN_SET_FOR_NULL >= 2


def test_seed_labels_are_normalised_across_the_three_spellings():
    """Job 22664474 failed here and the guard was right to refuse.

    THE FIXTURE DISCRIMINATES: the three inputs use the three spellings this
    repo actually contains -- `seed_7`, `7` and `b'seed_7'` -- so a comparison
    that skipped normalisation would see three different label sets.
    """
    from src.analysis.seed_axis import normalise_seed_labels
    assert normalise_seed_labels(["seed_7", 7, b"seed_7"]) == ["7", "7", "7"]
    assert normalise_seed_labels([0, 1, 19]) == ["0", "1", "19"]
