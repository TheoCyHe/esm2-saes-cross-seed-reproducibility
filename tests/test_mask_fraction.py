"""Tests for amendment AX §1a's masking-fraction measurement.

Every fixture is built so the quantity under test has a KNOWN answer that
DIFFERS from the answer a plausible bug would give (gotcha 16). In particular
the branch tests supply distributions where the three reasons for branch A are
separable, so a rule that collapsed them would fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.mask_fraction import (
    mask_counts_from_labels,
    mask_fraction_from_effective,
    percentile_rank,
    summarise,
    upper_tail_verdict,
)


# ------------------------------------------------- mask_fraction_from_effective

def test_fraction_is_the_masked_share_not_the_kept_share():
    # 1000 rows; a concept with 900 effective rows masked 100 -> 10%, NOT 90%.
    got = mask_fraction_from_effective(1000, np.array([900, 1000, 500]))
    assert got == pytest.approx([0.1, 0.0, 0.5])


def test_an_unmasked_concept_is_exactly_zero_not_merely_small():
    got = mask_fraction_from_effective(15_972_325, np.array([15_972_325]))
    assert got[0] == 0.0


def test_effective_above_total_is_refused_because_masking_only_removes():
    with pytest.raises(ValueError, match="exceeds n_rows"):
        mask_fraction_from_effective(1000, np.array([1001]))


def test_negative_effective_is_refused():
    with pytest.raises(ValueError, match="negative"):
        mask_fraction_from_effective(1000, np.array([-1]))


def test_zero_universe_is_refused_rather_than_dividing_by_zero():
    with pytest.raises(ValueError, match="must be positive"):
        mask_fraction_from_effective(0, np.array([0]))


def test_reproduces_aw5s_transmem_figure_from_the_writers_own_arithmetic():
    """AW5 reports TRANSMEM at 672,788 masked rows on a 15,972,325-row universe.

    The fixture encodes the AP writer's identity `n_eff = n_rows - other.size`,
    so a sign error or a kept/masked inversion gives 4.21% vs 95.79%.
    """
    n_rows = 15_972_325
    got = mask_fraction_from_effective(n_rows, np.array([n_rows - 672_788]))
    assert got[0] == pytest.approx(672_788 / n_rows)
    assert 0.04 < got[0] < 0.05


# ------------------------------------------------------ mask_counts_from_labels

def test_masked_count_is_the_non_experimental_remainder():
    pooled = {"A": np.array([1, 2, 3, 4, 5]), "B": np.array([7, 8])}
    exp = {"A": np.array([1, 2]), "B": np.array([7, 8])}
    # A: 5 pooled - 2 experimental = 3 masked. B: fully experimental -> 0.
    assert mask_counts_from_labels(pooled, exp, ["A", "B"]).tolist() == [3, 0]


def test_a_concept_absent_from_pooled_contributes_zero_not_a_keyerror():
    assert mask_counts_from_labels({}, {"A": np.array([1])}, ["A"]).tolist() == [0]


def test_the_set_difference_is_directional_pooled_minus_experimental():
    """Pooled {1,2,3}, exp {2,9}: forward gives {1,3} = 2, reversed gives {9} = 1.

    The counts must DIFFER between the two directions or the fixture cannot
    exhibit the condition it names -- an earlier version used pooled {1,2} and
    exp {2,9}, where both directions give 1 and a reversal mutation survived
    (gotcha 16).
    """
    pooled = {"A": np.array([1, 2, 3])}
    exp = {"A": np.array([2, 9])}
    assert mask_counts_from_labels(pooled, exp, ["A"]).tolist() == [2]
    # the reversed direction, computed here so the discrimination is explicit
    assert mask_counts_from_labels(exp, pooled, ["A"]).tolist() == [1]


def test_order_follows_the_requested_concept_list_not_the_dict_order():
    pooled = {"B": np.arange(10), "A": np.arange(4)}
    exp = {"B": np.arange(1), "A": np.arange(1)}
    # A -> 3, B -> 9. Requested ["A","B"] must give [3, 9]; dict order gives [9, 3].
    assert mask_counts_from_labels(pooled, exp, ["A", "B"]).tolist() == [3, 9]


# ------------------------------------------------------------- percentile_rank

def test_percentile_rank_is_the_share_strictly_below():
    sample = np.array([0.0, 0.0, 0.0, 1.0])
    assert percentile_rank(0.0, sample) == 0.0     # ties do not count as below
    assert percentile_rank(1.0, sample) == 75.0
    assert percentile_rank(2.0, sample) == 100.0


def test_percentile_rank_refuses_an_empty_sample():
    with pytest.raises(ValueError, match="empty sample"):
        percentile_rank(0.5, np.array([]))


# -------------------------------------------------------------------- summarise

def test_summarise_counts_exact_zeros_separately_from_small_values():
    f = np.array([0.0, 0.0, 1e-9, 0.02, 0.06])
    s = summarise(f)
    assert s["n_exactly_zero"] == 2          # 1e-9 is small, NOT zero
    assert s["n_above_1pct"] == 2            # 0.02 and 0.06
    assert s["n_above_5pct"] == 1            # 0.06 only
    assert s["percentiles"]["p100"] == pytest.approx(0.06)


def test_summarise_refuses_an_empty_set():
    with pytest.raises(ValueError, match="empty set"):
        summarise(np.array([]))


# --------------------------------------------------------- upper_tail_verdict
# The branch rule must be TOTAL and its three reasons for branch A separable.

def test_uniformly_tiny_takes_branch_a_by_arithmetic_even_when_pilot_ranks_low():
    """Both conditions point opposite ways; 'uniformly tiny' must win.

    Pilot at rank 0 would give PILOT_LOW_*; max fraction 0.5% means there is
    nothing to drift anywhere, which generalises more strongly.
    """
    f = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
    v = upper_tail_verdict(f, {"c": 0.0})
    assert v["branch"] == "A" and v["reason"] == "MASKING_UNIFORMLY_TINY"


def test_pilot_in_upper_half_takes_branch_a_by_sampling_when_tail_is_real():
    f = np.concatenate([np.zeros(90), np.linspace(0.02, 0.3, 10)])
    v = upper_tail_verdict(f, {"a": 95.0, "b": 99.0})
    assert v["branch"] == "A" and v["reason"] == "PILOT_IN_UPPER_HALF"


def test_pilot_low_with_substantial_tail_triggers_the_rerun_branch():
    f = np.concatenate([np.zeros(90), np.linspace(0.02, 0.3, 10)])
    v = upper_tail_verdict(f, {"a": 5.0, "b": 10.0})
    assert v["branch"] == "B" and v["reason"] == "PILOT_LOW_AND_SUBSTANTIAL_TAIL"


def test_the_fourth_region_is_named_rather_than_falling_through():
    """Pilot low, tail not substantial, but NOT uniformly tiny either.

    p99 must sit below 1% while the max sits above it -- a region the brief's
    two-branch wording is silent on. It must not silently read as branch B.
    """
    f = np.concatenate([np.zeros(999), np.array([0.4])])
    assert float(np.percentile(f, 99)) <= 0.01 < float(f.max())
    v = upper_tail_verdict(f, {"a": 0.0})
    assert v["branch"] == "A"
    assert v["reason"] == "PILOT_LOW_BUT_NO_SUBSTANTIAL_TAIL"


def test_the_verdict_is_total_over_the_four_reason_regions():
    """Every combination of the three booleans yields a named reason."""
    cases = [
        (np.array([0.001, 0.002]), {"a": 0.0}),
        (np.array([0.001, 0.002]), {"a": 99.0}),
        (np.concatenate([np.zeros(90), np.linspace(0.02, 0.3, 10)]), {"a": 0.0}),
        (np.concatenate([np.zeros(90), np.linspace(0.02, 0.3, 10)]), {"a": 99.0}),
        (np.concatenate([np.zeros(999), np.array([0.4])]), {"a": 0.0}),
        (np.concatenate([np.zeros(999), np.array([0.4])]), {"a": 99.0}),
    ]
    reasons = {upper_tail_verdict(f, r)["reason"] for f, r in cases}
    assert reasons <= {"MASKING_UNIFORMLY_TINY", "PILOT_IN_UPPER_HALF",
                       "PILOT_LOW_AND_SUBSTANTIAL_TAIL",
                       "PILOT_LOW_BUT_NO_SUBSTANTIAL_TAIL"}
    for f, r in cases:
        assert upper_tail_verdict(f, r)["branch"] in {"A", "B"}


def test_the_verdict_reports_the_numbers_it_decided_on():
    """A verdict word with no accompanying number cannot be re-derived by a
    reader -- gotcha 20's rule that a printed conclusion must be a function of
    the data it just computed."""
    f = np.concatenate([np.zeros(90), np.linspace(0.02, 0.3, 10)])
    v = upper_tail_verdict(f, {"a": 5.0, "b": 10.0})
    assert v["max_masking_fraction"] == pytest.approx(0.3)
    assert v["median_pilot_percentile_rank"] == pytest.approx(7.5)
    assert v["p99_masking_fraction"] > 0.01
