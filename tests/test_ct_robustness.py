"""Tests for src/analysis/ct_robustness.py (amendment CT, §CT15-§CT17, §CT20,
§CT21). Expectations are written down, not computed from the code under test
(gotcha 32b)."""
import numpy as np
import pytest

from src.analysis.ct_robustness import (
    bin_of, gaussian_penalised_fit, intervals_overlap, own_column_mask,
    strictly_monotone, subset_counts, subset_rate, theta_star_subset, widest,
)


def _sim():
    """2 anchors (labels 0, 1) x 4 columns (labels 0, 1, 2, 3) x 3 features.
    Anchor 0's own column is column 0; anchor 1's is column 1. The diagonal
    holds a HIGH value on purpose: a function that fails to exclude the
    anchor's own column counts it."""
    s = np.full((2, 4, 3), -np.inf, dtype=np.float32)
    s[0, 0, :] = 0.99            # own column (must never count)
    s[0, 1, :] = [0.8, 0.6, -np.inf]
    s[0, 2, :] = [0.9, 0.75, 0.2]
    s[0, 3, :] = [0.71, -np.inf, 0.7]
    s[1, 1, :] = 0.99            # own column
    s[1, 0, :] = [0.5, 0.95, 0.72]
    s[1, 2, :] = [0.69, 0.7, 0.71]
    s[1, 3, :] = [-np.inf, 0.1, 0.9]
    return s


A, C = ["0", "1"], ["0", "1", "2", "3"]


def test_own_column_is_found_by_label_not_position():
    m = own_column_mask(A, C, ["2", "3"])
    assert not m.any()                      # neither anchor's own column is in 2, 3
    m = own_column_mask(A, C, ["1", "0"])
    assert m.tolist() == [[False, True], [True, False]]


def test_counts_over_all_other_columns_exclude_the_diagonal():
    y, n = subset_counts(_sim(), 0.70, A, C, ["0", "1"])
    # anchor 0 against column 1 only: [0.8, 0.6, -inf] >= 0.70 -> [1, 0, 0]
    # anchor 1 against column 0 only: [0.5, 0.95, 0.72] -> [0, 1, 1]
    assert n == 1
    assert y.tolist() == [[1, 0, 0], [0, 1, 1]]


def test_counts_over_non_anchor_columns_keep_every_column():
    """§CT15's case: columns 2, 3 are nobody's own, so n_trials is 2 and the
    first column of the subset must NOT be excluded as if it were a diagonal."""
    y, n = subset_counts(_sim(), 0.70, A, C, ["2", "3"])
    assert n == 2
    assert y.tolist() == [[2, 1, 1], [0, 1, 2]]


def test_counts_refuse_unequal_denominators():
    with pytest.raises(ValueError, match="different comparator counts"):
        subset_counts(_sim(), 0.70, A, C, ["0", "2"])   # anchor 0 loses one, anchor 1 none


def test_counts_refuse_an_unknown_column():
    with pytest.raises(ValueError, match="not in the similarity array"):
        subset_counts(_sim(), 0.70, A, C, ["2", "9"])


def test_theta_star_is_the_largest_value_reaching_the_target():
    # Over columns 2, 3 the finite values are 0.9 0.75 0.2 0.71 0.7 | 0.69 0.7 0.71 0.1 0.9
    # rate(t) = (# values >= t) / 6 cells. Target 0.5 -> need >= 3 hits of 6...
    # sorted descending: 0.9, 0.9, 0.75, 0.71, 0.71, 0.7, 0.7, 0.69, 0.2, 0.1
    # rate at 0.71 = 5 / (2 anchors * 3 features * 2 comparators = 12) = 0.4167
    # rate at 0.70 = 7 / 12 = 0.5833 >= 0.5, rate at 0.71 < 0.5 -> theta* = 0.70
    ts = theta_star_subset(_sim(), A, C, ["2", "3"], 0.5)
    assert ts["status"] == "OK"
    assert ts["theta_star"] == pytest.approx(0.70, abs=1e-7)
    assert ts["rate_at_theta_star"] == pytest.approx(7 / 12)
    assert ts["rate_just_above"] == pytest.approx(5 / 12)


def test_theta_star_ignores_the_anchors_own_column():
    """Columns 0 and 1 carry 0.99 on the diagonals; if they counted, theta*
    at a target of 1.0 would be reachable at 0.99. It must not be."""
    ts = theta_star_subset(_sim(), A, C, ["0", "1"], 1.0)
    assert ts["status"] == "TARGET_UNREACHABLE"


def test_subset_rate_matches_hand_count():
    assert subset_rate(_sim(), 0.70, A, C, ["2", "3"]) == pytest.approx(7 / 12)


def test_overlap_and_widest():
    assert intervals_overlap([0, 1], [1, 2])          # touching counts
    assert not intervals_overlap([0, 1], [1.01, 2])
    assert widest({"a": [0, 1], "b": [-1, 1.5], "c": [0, 2]}) == ("b", [-1, 1.5])


@pytest.mark.parametrize("v,want", [
    ([1, 2, 3], "INCREASING"), ([3, 2, 1], "DECREASING"),
    ([1, 2, 2], "NEITHER"), ([1, 3, 2], "NEITHER"), ([1, float("nan")], "NEITHER"),
    ([1], "NEITHER")])
def test_strictly_monotone(v, want):
    assert strictly_monotone(v) == want


def test_bin_of_is_half_open_and_clips_the_top():
    e = [0.0, 3e-4, 1e-3, 1.0 + 1e-9]
    got = bin_of(np.array([0.0, 2.9999e-4, 3e-4, 1e-3, 0.5, 1.0]), e)
    assert got.tolist() == [0, 0, 1, 2, 2, 2]


def test_gaussian_fit_recovers_an_unpenalised_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    X = np.column_stack([np.ones(500), x])
    y = 2.0 + 0.3 * x + rng.normal(scale=0.01, size=500)
    g = gaussian_penalised_fit(y, X, np.zeros((2, 2)), 0.0)
    assert g["beta"][1] == pytest.approx(0.3, abs=0.005)
    assert g["edf"] == pytest.approx(2.0, abs=1e-6)
    assert np.sqrt(g["cov"][1, 1]) < 0.002


def test_every_join_step_gets_a_config_path():
    """Jobs 30342488-90 and 30342493 refused on a None config path: the parsed
    namespace must carry the repository default before load_join_inputs runs."""
    from src.analysis.ct_robustness import build_parser, resolve_config
    from src.experiment_config import DEFAULT_CONFIG_PATH
    base = ["--sim", "a", "--rf-counts-all-theta", "b", "--m1-json", "c", "--gtilde", "d",
            "--null-reductions", "e", "--d-fs", "f", "--reference-join", "g", "--out", "h"]
    for argv in (["ct15", *base, "--jackknife-json", "j"],
                 ["ct20", *base, "--act-counts", "k", "--act-json", "l", "--rules-h5", "m"],
                 ["ct17", *base, "--act-counts", "k", "--act-json", "l"],
                 ["ct16-fit", *base, "--oneway-h5", "n"],
                 ["ct10-fit", *base, "--act-half-counts", "o", "--act-half-json", "p",
                  "--gtilde-half-a", "q", "--gtilde-half-b", "r"]):
        args = build_parser().parse_args(argv)
        assert args.config is None
        resolve_config(args)
        assert args.config == DEFAULT_CONFIG_PATH
