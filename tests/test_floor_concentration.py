"""Tests for BR part C's per-concept floor decomposition.

EVERY EXPECTATION IN THIS FILE IS WRITTEN DOWN, NOT COMPUTED FROM THE CODE
UNDER TEST (gotcha 32b). Where a number could have been produced by calling the
function being checked, it is instead derived by hand in a comment and typed as
a literal.

Each check is also verified to be able to FAIL (gotcha 13): the recovery guard,
the axis guards and the evaluability guard each have a test that plants the
condition and requires the refusal.
"""

import numpy as np
import pytest

from src.analysis.floor_concentration import (
    at_floor_per_concept,
    concentration,
    distribution,
    matched_by_positives,
    recover_numerators,
)


def _exact_p(numerators, n_finite):
    """Build a float32 p-array from exact numerators, as the artifact stores it."""
    num = np.asarray(numerators, dtype=np.float64)
    return (num / (np.asarray(n_finite, dtype=np.float64) + 1.0)).astype(np.float32)


# --------------------------------------------------------------------------
# recover_numerators -- the exact round-trip
# --------------------------------------------------------------------------

def test_recovers_numerators_exactly_at_realistic_denominators():
    # Two concepts with DIFFERENT denominators, which is the case a single
    # global floor gets wrong. 204800 is the full pool; 190000 a degenerate one.
    n_finite = np.array([204800, 190000], dtype=np.int64)
    num = np.array([[1, 1],
                    [2, 5],
                    [1, 190001],
                    [173, 2]], dtype=np.int64)
    p = _exact_p(num, n_finite)
    got = recover_numerators(p.astype(np.float64), n_finite)
    assert got.tolist() == num.tolist()


def test_recovery_refuses_when_p_is_not_a_rational_of_that_denominator():
    # CAN THIS CHECK FAIL? Plant a p-value that is not j/(n+1) for any integer j
    # and require the refusal.
    n_finite = np.array([204800], dtype=np.int64)
    p = _exact_p(np.array([[1], [2]]), n_finite).astype(np.float64)
    p[1, 0] = float(np.float32(2.5 / 204801.0))     # half-integer numerator
    with pytest.raises(ValueError, match="NUMERATOR RECOVERY IS NOT EXACT"):
        recover_numerators(p, n_finite)


def test_recovery_refuses_a_transposed_denominator():
    n_finite = np.array([204800, 190000, 100], dtype=np.int64)   # 3, not 2
    p = _exact_p(np.array([[1, 1], [2, 2]]), np.array([204800, 190000]))
    with pytest.raises(ValueError, match="n_finite is"):
        recover_numerators(p.astype(np.float64), n_finite)


def test_recovery_refuses_a_nan_at_an_evaluable_concept():
    n_finite = np.array([204800], dtype=np.int64)
    p = _exact_p(np.array([[1], [3]]), n_finite).astype(np.float64)
    p[0, 0] = np.nan
    with pytest.raises(ValueError, match="evaluability and its p-values"):
        recover_numerators(p, n_finite)


# --------------------------------------------------------------------------
# at_floor_per_concept
# --------------------------------------------------------------------------

def test_at_floor_counts_only_numerator_one():
    # Hand-counted. Column 0 numerators: 1, 2, 1, 1  -> THREE at floor.
    # Column 1 numerators: 1, 7, 999, 4              -> ONE at floor.
    n_finite = np.array([204800, 190000], dtype=np.int64)
    num = np.array([[1, 1],
                    [2, 7],
                    [1, 999],
                    [1, 4]], dtype=np.int64)
    p = _exact_p(num, n_finite).astype(np.float64)
    assert at_floor_per_concept(p, n_finite).tolist() == [3, 1]


def test_the_floor_is_per_concept_not_global():
    # THE POINT OF THE WHOLE FUNCTION. Concept 1's floor is 1/190001, which is
    # NOT the global minimum 1/204801. A global comparand would count concept
    # 1's floor-tied tests as zero. Hand-counted: [1, 2].
    n_finite = np.array([204800, 190000], dtype=np.int64)
    num = np.array([[1, 1],
                    [9, 1],
                    [4, 6]], dtype=np.int64)
    p = _exact_p(num, n_finite).astype(np.float64)
    assert at_floor_per_concept(p, n_finite).tolist() == [1, 2]

    # and the global-floor mistake really would differ here, so the test above
    # is not vacuous: p[0,1] != 1/204801.
    assert abs(p[0, 1] - 1.0 / 204801.0) > 1e-12


def test_a_concept_with_nothing_at_its_floor_counts_zero():
    """ADDED AFTER A MUTANT SURVIVED. `j == 1` -> `j == j.min()` passed the
    whole suite, because every fixture happened to contain a numerator of 1, so
    the array minimum WAS 1 and the two agreed. A concept whose smallest
    numerator is above 1 supplies NO at-floor test, and only a fixture
    containing one can tell the two apart (gotcha 13: a fixture that omits the
    case cannot exercise the check on it).

    Hand-counted. Column 0 numerators 4, 9, 7 -> minimum 4, NONE at floor -> 0.
    Column 1 numerators 1, 3, 2                                          -> 1.
    """
    n_finite = np.array([204800, 190000], dtype=np.int64)
    num = np.array([[4, 1],
                    [9, 3],
                    [7, 2]], dtype=np.int64)
    p = _exact_p(num, n_finite).astype(np.float64)
    assert at_floor_per_concept(p, n_finite).tolist() == [0, 1]


def test_a_seed_with_nothing_at_the_floor_anywhere_counts_zero_everywhere():
    """THE FIXTURE THAT ACTUALLY KILLS `j == j.min()`, and the first attempt did
    not. The test above still contains a numerator of 1 SOMEWHERE, so the array
    minimum is 1 and `j.min()` coincides with the literal -- the mutant stayed
    equivalent on it. Only a fixture whose GLOBAL minimum exceeds 1 separates
    "at this concept's floor" from "smallest p-value present in this slab".

    This is the case that matters in the data: it is exactly what a
    floor-limited seed contributing no rejections looks like.

    Hand-counted: minimum numerator anywhere is 2, so NOTHING is at floor -> 0.
    """
    n_finite = np.array([204800, 190000], dtype=np.int64)
    num = np.array([[4, 3],
                    [9, 5],
                    [7, 2]], dtype=np.int64)
    p = _exact_p(num, n_finite).astype(np.float64)
    assert at_floor_per_concept(p, n_finite).tolist() == [0, 0]


# --------------------------------------------------------------------------
# concentration
# --------------------------------------------------------------------------

def test_concentration_shares_are_hand_computed():
    # totals 50, 30, 15, 5, 0  -> grand total 100.
    # top1 = 50/100 = 0.50; top5 = 100/100 = 1.0. One concept supplies zero,
    # so share_zero = 1/5 = 0.20.
    c = concentration(np.array([15, 50, 0, 30, 5]), top_k=(1, 5))
    assert c["grand_total"] == 100
    assert c["n_concepts"] == 5
    assert c["n_concepts_zero"] == 1
    assert c["share_zero"] == pytest.approx(0.20)
    assert c["top1_share"] == pytest.approx(0.50)
    assert c["top5_share"] == pytest.approx(1.0)
    assert c["max_single_concept"] == 50


def test_concentration_k_larger_than_the_population_is_clamped():
    c = concentration(np.array([3, 1]), top_k=(25,))
    assert c["top25_share"] == pytest.approx(1.0)


def test_concentration_reports_an_all_zero_total_without_dividing_by_zero():
    c = concentration(np.array([0, 0, 0]), top_k=(1,))
    assert c["grand_total"] == 0
    assert c["n_concepts_zero"] == 3
    assert np.isnan(c["top1_share"])


# --------------------------------------------------------------------------
# distribution
# --------------------------------------------------------------------------

def test_distribution_summaries_are_hand_computed():
    # 2 seeds x 3 concepts. Values 1,2,3 / 4,5,6.
    # mean = 21/6 = 3.5; median = 3.5; max = 6.
    # per-seed totals 6 and 15 -> min 6, max 15, mean 10.5.
    d = distribution(np.array([[1, 2, 3], [4, 5, 6]]))
    assert d["n_cells"] == 6
    assert d["n_seeds"] == 2
    assert d["n_concepts"] == 3
    assert d["mean_per_concept"] == pytest.approx(3.5)
    assert d["median_per_concept"] == pytest.approx(3.5)
    assert d["max_per_concept"] == pytest.approx(6.0)
    assert d["total_per_seed_min"] == 6
    assert d["total_per_seed_max"] == 15
    assert d["total_per_seed_mean"] == pytest.approx(10.5)


# --------------------------------------------------------------------------
# matched_by_positives
# --------------------------------------------------------------------------

def test_binning_places_concepts_by_positive_count_and_states_denominators():
    # positives 12, 45, 250, 5000 -> bins [10,30), [30,100), [100,300),
    # [1000,3000) empty, [3000,10000).
    at = np.array([[2, 4, 6, 8],
                   [4, 6, 8, 10]])
    rows = matched_by_positives(at, np.array([12, 45, 250, 5000]))
    by_lo = {r["lo"]: r for r in rows}
    assert by_lo[10]["n_concepts"] == 1
    assert by_lo[10]["mean_at_floor_per_concept_per_seed"] == pytest.approx(3.0)
    assert by_lo[30]["n_concepts"] == 1
    assert by_lo[30]["mean_at_floor_per_concept_per_seed"] == pytest.approx(5.0)
    assert by_lo[100]["n_concepts"] == 1
    assert by_lo[300]["n_concepts"] == 0
    assert by_lo[3000]["n_concepts"] == 1
    assert by_lo[3000]["mean_at_floor_per_concept_per_seed"] == pytest.approx(9.0)


def test_a_concept_exactly_on_a_bin_edge_falls_in_the_UPPER_bin():
    """ADDED AFTER A MUTANT SURVIVED. `pos >= lo` -> `pos > lo` passed the whole
    suite, because no fixture placed a concept exactly on an edge. The bins are
    half-open [lo, hi), so a concept at exactly 30 belongs to [30, 100) and NOT
    to [10, 30) -- and under the mutant it would vanish from both.

    Concepts at exactly 10 (the library's own retention floor) and exactly 30.
    """
    at = np.array([[2, 6]])
    rows = matched_by_positives(at, np.array([10, 30]))
    by_lo = {r["lo"]: r for r in rows}
    assert by_lo[10]["n_concepts"] == 1
    assert by_lo[10]["mean_at_floor_per_concept_per_seed"] == pytest.approx(2.0)
    assert by_lo[30]["n_concepts"] == 1
    assert by_lo[30]["mean_at_floor_per_concept_per_seed"] == pytest.approx(6.0)
    # nothing was silently dropped: the two concepts are accounted for exactly
    assert sum(r["n_concepts"] for r in rows) == 2


def test_an_empty_bin_is_reported_not_dropped():
    # An absent row and a zero row are different statements (gotcha 28's shape).
    rows = matched_by_positives(np.array([[1]]), np.array([12]))
    assert len(rows) == 7                      # every declared bin present
    empties = [r for r in rows if r["n_concepts"] == 0]
    assert len(empties) == 6
    assert all(r["mean_at_floor_per_concept_per_seed"] is None for r in empties)
    assert all(r["total_at_floor"] == 0 for r in empties)


def test_binning_refuses_a_misjoined_positive_count_axis():
    with pytest.raises(ValueError, match="SAME concepts in the SAME order"):
        matched_by_positives(np.array([[1, 2, 3]]), np.array([10, 20]))
