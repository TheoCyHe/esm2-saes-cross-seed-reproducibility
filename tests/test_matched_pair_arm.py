"""COMPONENT 5's fixtures (PREREGISTRATION amendment BU): the firing-rate-matched
paired arm.

CRITERION 2 IS THE POINT OF THIS FILE, AND IT HAS TWO HALVES. Every expectation
below is WRITTEN DOWN -- a hand-computed number, or a number read off a
published definition of the signed-rank test -- and never computed by calling
the code under test (gotcha 32b). And **every fixture that names a branch is
shown to REACH that branch**: each such test asserts the branch label as well as
the value, so a fixture that silently exercised a different path would fail
rather than pass vacuously. Fixture defects have surfaced in seven consecutive
sessions, every one a fixture that could not exhibit its named branch.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.analysis.match_pool_diagnostics import nearest_within_caliper
from src.analysis.matched_pair_arm import (
    SEED_BRANCHES,
    TEST_BRANCHES,
    TREATED_BRANCHES,
    OutcomeSpaceViolation,
    bin_index,
    match_sets,
    min_attainable_two_sided_p,
    reuse_profile,
    set_contrast,
    signed_rank,
    verify_treated_outcome_space,
)

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# match_sets -- rule 1 (all tied candidates) and rule 2 (with replacement)
# ---------------------------------------------------------------------------

def test_a_unique_nearest_control_gives_a_singleton_set():
    """Controls at -2.01, -2.015, -2.05 and -2.09 are all inside a 0.1 caliper
    of -2.0; only -2.01 is nearest, so the set is {0} and the branch is
    MATCHED_UNIQUE.

    THE SECOND-NEAREST SITS 0.005 AWAY ON PURPOSE. With the next candidate at
    -2.05 the fixture could not tell a strict minimum from a minimum plus a
    small tolerance, and the mutation matrix found exactly that hole."""
    sets = match_sets(np.array([-2.0]),
                      np.array([-2.01, -2.015, -2.05, -2.09]), caliper=0.1)
    assert sets[0].tolist() == [0]
    assert sets[0].size == 1        # -> MATCHED_UNIQUE


def test_ALL_tied_candidates_are_returned_and_none_is_preferred():
    """Rule 1. Two controls sit at exactly -2.02 and one further away inside the
    caliper. BOTH tied controls are returned -- indices 0 and 2 -- and the third
    is not. A first-index tie-break would return only [0] and fail here."""
    sets = match_sets(np.array([-2.0]),
                      np.array([-2.02, -2.06, -2.02]), caliper=0.1)
    assert sets[0].tolist() == [0, 2]
    assert sets[0].size == 2        # -> MATCHED_TIED


def test_a_treated_feature_with_NOTHING_inside_the_caliper_gets_an_EMPTY_set():
    """gotcha 5's mandatory count. The second treated feature at -2.15 is 0.13
    from the only control -- outside a 0.1 caliper -- so its set is empty ->
    UNMATCHABLE, and it must not borrow the first feature's match.

    THE GAP IS 0.13, NOT A CHASM, ON PURPOSE. With the unmatchable feature far
    away the fixture could not detect a caliper that had silently widened, and
    the mutation matrix found exactly that hole."""
    sets = match_sets(np.array([-2.0, -2.15]), np.array([-2.02]), caliper=0.1)
    assert sets[0].tolist() == [0]
    assert sets[1].size == 0        # -> UNMATCHABLE


def test_matching_is_WITH_replacement_one_control_serving_two_treated():
    """Rule 2. One control at -2.005 is the nearest for treated features at
    -2.0 and -2.01 alike; both get it. Without replacement the second would be
    left with nothing."""
    sets = match_sets(np.array([-2.0, -2.01]), np.array([-2.005]), caliper=0.1)
    assert sets[0].tolist() == [0]
    assert sets[1].tolist() == [0]


def test_a_treated_feature_with_NO_firing_rate_matches_NOTHING():
    """A NaN log rate cannot participate in a minimum. The ensemble has no dead
    features so this branch is expected empty on real data -- which is why it is
    exercised here."""
    sets = match_sets(np.array([-2.0, np.nan]),
                      np.array([-2.01, np.nan]), caliper=0.1)
    assert sets[0].tolist() == [0]      # the NaN CONTROL is not a candidate
    assert sets[1].size == 0            # -> NO_FIRING_RATE


def test_match_sets_REFUSES_a_non_positive_caliper():
    with pytest.raises(ValueError, match="positive"):
        match_sets(np.array([-2.0]), np.array([-2.0]), caliper=0.0)


# ---------------------------------------------------------------------------
# The caliper boundary -- committed as INCLUSIVE, unreachable at 0.1
# ---------------------------------------------------------------------------

def test_the_caliper_boundary_is_INCLUSIVE_where_float64_can_express_it():
    """A control exactly one caliper away IS admissible: the predicate is
    `d <= caliper`, as §5 pins it. At a caliper of 0.5 the arithmetic can
    express the boundary, so the branch is reachable and is shown reached."""
    sets = match_sets(np.array([-2.0]), np.array([-1.5]), caliper=0.5)
    assert abs(-2.0 - (-1.5)) == 0.5
    assert sets[0].tolist() == [0]


def test_the_boundary_is_NOT_REACHABLE_at_the_PINNED_caliper_of_0_1():
    """Corrections-log entry 142, re-asserted here for THIS module.
    `abs(-2.0 - (-2.1))` is 0.10000000000000009 in float64 -- strictly greater
    than 0.1 -- so a control the arithmetic places exactly one caliper away is
    EXCLUDED. The distinction between an inclusive and an exclusive boundary
    therefore cannot be exercised at the pinned width, and the branch is not
    written as though it could be."""
    assert abs(-2.0 - (-2.1)) > 0.1
    sets = match_sets(np.array([-2.0]), np.array([-2.1]), caliper=0.1)
    assert sets[0].size == 0


# ---------------------------------------------------------------------------
# set_contrast -- the estimand's set structure
# ---------------------------------------------------------------------------

def test_the_contrast_averages_the_matched_SET_with_equal_weights():
    """Treated r_f = 0.5; the two tied controls have r_f 0.2 and 0.4, mean 0.3.
    d = 0.5 - 0.3 = 0.2, computed by hand. An implementation taking the FIRST
    tied control would give 0.3 and fail; one taking the last, 0.1."""
    d = set_contrast(np.array([0.5]), np.array([0.2, 0.9, 0.4]),
                     [np.array([0, 2])])
    assert d[0] == pytest.approx(0.2)


def test_an_unmatchable_treated_feature_has_NO_contrast_not_a_zero_one():
    """NaN, never 0.0. A zero would enter a real observation of 'no difference'
    for a feature that was never compared to anything."""
    d = set_contrast(np.array([0.5, 0.6]), np.array([0.2]),
                     [np.array([0]), np.zeros(0, dtype=np.int64)])
    assert d[0] == pytest.approx(0.3)
    assert np.isnan(d[1])


def test_set_contrast_REFUSES_a_length_mismatch():
    with pytest.raises(ValueError, match="matched sets"):
        set_contrast(np.array([0.5, 0.6]), np.array([0.2]), [np.array([0])])


# ---------------------------------------------------------------------------
# signed_rank -- all three branches, each shown reached
# ---------------------------------------------------------------------------

def test_signed_rank_COMPUTED_against_a_hand_worked_exact_test():
    """d = [1, 2, 3], all positive. Ranks 1, 2, 3; W- = 0, W+ = 6; the
    two-sided statistic is min(W+, W-) = 0. The exact two-sided p is the
    probability of a sign assignment at least this extreme: 2 / 2**3 = 0.25.
    Both numbers are written down from the definition of the test, not read off
    this module."""
    r = signed_rank(np.array([1.0, 2.0, 3.0]))
    assert r["branch"] == "COMPUTED"
    assert r["statistic"] == pytest.approx(0.0)
    assert r["p_value"] == pytest.approx(0.25)
    assert r["method"] == "exact"
    assert r["n_pairs"] == 3
    assert r["n_positive_differences"] == 3
    assert r["median_difference"] == pytest.approx(2.0)


def test_signed_rank_COMPUTED_on_a_mixed_sign_case_worked_by_hand():
    """d = [1, -2, 3, 4]. |d| ranks are 1, 2, 3, 4; W- = 2, W+ = 8, statistic
    2. Exact two-sided p counts the sign assignments with min(W+, W-) <= 2:
    the rank subsets summing to 0, 1 or 2 are {}, {1}, {2} -- three on each
    side, so 6 of 2**4 = 16, i.e. 0.375."""
    r = signed_rank(np.array([1.0, -2.0, 3.0, 4.0]))
    assert r["branch"] == "COMPUTED"
    assert r["statistic"] == pytest.approx(2.0)
    assert r["p_value"] == pytest.approx(0.375)
    assert r["n_negative_differences"] == 1


def test_signed_rank_NO_PAIRS_is_its_own_branch_not_a_p_value():
    r = signed_rank(np.zeros(0))
    assert r["branch"] == "NO_PAIRS"
    assert r["p_value"] is None and r["statistic"] is None
    assert r["n_pairs"] == 0


def test_signed_rank_reports_ALL_DIFFERENCES_ZERO_rather_than_p_equals_1():
    """A ranking predicate applied to a constant ranks nothing (gotcha 25b).
    Three pairs all with d = 0: the branch is ALL_DIFFERENCES_ZERO, the p-value
    is None, and `n_pairs` is still 3 -- the pairs existed."""
    r = signed_rank(np.zeros(3))
    assert r["branch"] == "ALL_DIFFERENCES_ZERO"
    assert r["p_value"] is None
    assert r["n_pairs"] == 3 and r["n_zero_differences"] == 3


def test_NaN_contrasts_are_DROPPED_AND_COUNTED_never_silently_included():
    r = signed_rank(np.array([1.0, np.nan, 2.0, 3.0]))
    assert r["n_pairs"] == 3
    assert r["n_dropped_no_contrast"] == 1
    assert r["p_value"] == pytest.approx(0.25)


def test_every_TEST_BRANCH_is_reachable():
    """A branch nothing can reach is not a branch. All three are exhibited."""
    seen = {signed_rank(np.zeros(0))["branch"],
            signed_rank(np.zeros(3))["branch"],
            signed_rank(np.array([1.0, 2.0, 3.0]))["branch"]}
    assert seen == set(TEST_BRANCHES)


# ---------------------------------------------------------------------------
# The resolution floor -- gotcha 42, from parameters and no data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [
    (0, 1.0), (1, 1.0), (2, 0.5), (3, 0.25), (4, 0.125), (5, 0.0625),
    (6, 0.03125), (10, 0.001953125),
])
def test_the_exact_test_floor_is_2_to_the_1_minus_n(n, expected):
    """Written down from the definition: 2 / 2**n. At n = 5 the smallest
    attainable two-sided p is 0.0625 -- so a cell with five non-zero
    differences cannot reach 0.05 whatever the data say, and at n = 6 it can."""
    assert min_attainable_two_sided_p(n) == pytest.approx(expected)


def test_the_floor_is_ATTAINED_by_the_most_extreme_sample():
    """The floor is not merely a bound: d = [1, 2, 3] is the most extreme
    arrangement at n = 3 and its exact p is 0.25, which IS 2**(1-3). A floor
    that could never be reached would be a different (and weaker) claim."""
    assert signed_rank(np.array([1.0, 2.0, 3.0]))["p_value"] == \
        pytest.approx(min_attainable_two_sided_p(3))


# ---------------------------------------------------------------------------
# Binning -- the mandatory per-firing-rate-bin axis
# ---------------------------------------------------------------------------

def test_bin_index_places_rates_in_corrector_scan_bins():
    """Edges 0, 1e-4, 3e-4, 1e-3, ... so 5e-5 -> bin 0, 2e-4 -> bin 1,
    5e-4 -> bin 2, 0.2 -> bin 7 (the sparse [0.1, 0.5) bin)."""
    got = bin_index(np.array([5e-5, 2e-4, 5e-4, 0.2]))
    assert got.tolist() == [0, 1, 2, 7]


def test_a_rate_of_exactly_one_lands_in_the_TOP_bin_not_outside():
    assert int(bin_index(np.array([1.0]))[0]) == 8


def test_a_rate_exactly_ON_an_edge_lands_in_the_UPPER_bin():
    """The bins are half-open `[lo, hi)`. A rate of exactly 1e-4 is the lower
    edge of bin 1, not the upper edge of bin 0. Written down here because an
    inclusive upper edge would put a feature in two bins and the last write
    would win silently."""
    assert int(bin_index(np.array([1e-4]))[0]) == 1
    assert int(bin_index(np.array([3e-4]))[0]) == 2


def test_a_rate_outside_every_bin_is_MINUS_ONE_and_not_clamped():
    """A negative rate is impossible in this data, which is exactly why the
    out-of-range branch is exercised here rather than assumed unreachable."""
    assert int(bin_index(np.array([-0.5]))[0]) == -1


# ---------------------------------------------------------------------------
# Reuse -- rule 2's mandatory report and the dependence it induces
# ---------------------------------------------------------------------------

def test_reuse_profile_counts_a_shared_control_and_the_treated_that_share_it():
    """Three treated features: two matched to control 0, one to control 1.
    Distinct controls used = 2; max reuse = 2; one control is reused; and TWO
    treated features share a control -- the quantity the estimand note in §5a
    is about."""
    sets = [np.array([0]), np.array([0]), np.array([1])]
    r = reuse_profile(sets, n_control=5)
    assert r["n_distinct_controls_used"] == 2
    assert r["control_reuse_max"] == 2
    assert r["n_controls_reused"] == 1
    assert r["n_treated_sharing_a_control"] == 2
    assert r["n_treated_matched"] == 3
    assert r["frac_treated_sharing_a_control"] == pytest.approx(2 / 3)


def test_reuse_profile_with_no_reuse_reports_zero_sharing_not_None():
    sets = [np.array([0]), np.array([1])]
    r = reuse_profile(sets, n_control=3)
    assert r["n_treated_sharing_a_control"] == 0
    assert r["control_reuse_max"] == 1


# ---------------------------------------------------------------------------
# The outcome space -- criterion 5 in this component's primitive
# ---------------------------------------------------------------------------

def test_the_outcome_space_counts_sum_to_the_treated_population():
    branches = ["MATCHED_UNIQUE", "MATCHED_TIED", "UNMATCHABLE",
                "MATCHED_UNIQUE"]
    counts = verify_treated_outcome_space(branches, 4)
    assert counts["MATCHED_UNIQUE"] == 2
    assert counts["MATCHED_TIED"] == 1
    assert counts["UNMATCHABLE"] == 1
    # An empty branch is carried with an explicit zero, never dropped.
    assert counts["NO_FIRING_RATE"] == 0
    assert sum(counts.values()) == 4


def test_the_outcome_space_REFUSES_an_unknown_branch():
    with pytest.raises(OutcomeSpaceViolation, match="not one of"):
        verify_treated_outcome_space(["SOMETHING_ELSE"], 1)


def test_the_outcome_space_REFUSES_an_unassigned_cell():
    """Two treated cells, one assignment. This is the direction that can fail."""
    with pytest.raises(OutcomeSpaceViolation, match="unassigned"):
        verify_treated_outcome_space(["MATCHED_UNIQUE"], 2)


def test_the_branch_enumerations_are_the_ones_the_record_names():
    assert TREATED_BRANCHES == ("NO_FIRING_RATE", "UNMATCHABLE",
                                "MATCHED_UNIQUE", "MATCHED_TIED")
    assert SEED_BRANCHES == ("CONTRIBUTES", "NO_TREATED_FEATURES")


# ---------------------------------------------------------------------------
# Agreement with the independently written BT helper
# ---------------------------------------------------------------------------

def test_match_sets_agrees_with_match_pool_diagnostics_on_multiplicities():
    """`nearest_within_caliper` was written in a different session for a
    different purpose and returns only the multiplicities. Two implementations
    of one specification must agree; this is a CHECK of that structural
    necessity, not a prediction."""
    rng = np.random.default_rng(0)
    t = np.concatenate([rng.normal(-3.0, 0.4, size=40), [-9.0]])
    c = np.round(rng.normal(-3.0, 0.4, size=300), 3)   # ties by construction
    mine = np.array([s.size for s in match_sets(t, c, 0.1)])
    theirs = nearest_within_caliper(t, c, 0.1)["n_at_min"]
    assert np.array_equal(mine, theirs)
    # BOTH interesting branches must actually occur, or the agreement is over a
    # pool that never exercised them. The -9.0 treated value guarantees the
    # unmatchable one; the rounding guarantees ties.
    assert (mine > 1).any(), "the fixture must actually produce ties"
    assert (mine == 0).any(), "the fixture must produce an unmatchable feature"


# ---------------------------------------------------------------------------
# The seed-level ignorance category -- §5b, end to end through the driver
# ---------------------------------------------------------------------------

def _write_fixture(tmp_path: Path, aligned_per_seed) -> dict:
    """A three-seed, eight-feature fixture with a KNOWN answer.

    Seed 1 is given NO aligned feature, so the driver must take §5b's branch.
    Firing counts are chosen so every aligned feature has a control within the
    caliper, and `match_count` is chosen so the contrasts are known by hand.
    """
    import h5py
    from src.analysis.classification_2x2 import (CELL_ALIGNED_BOTH,
                                                 CELL_ORPHAN)
    n_seeds, n_feat, n_rows = 3, 8, 1_000_000
    seeds = [f"seed_{i}" for i in range(n_seeds)]
    fire = np.full((n_seeds, n_feat), 1000, dtype=np.int64)
    cell = np.full((n_seeds, n_feat), CELL_ORPHAN, dtype=np.uint8)
    for s, idx in enumerate(aligned_per_seed):
        for i in idx:
            cell[s, i] = CELL_ALIGNED_BOTH
    mc = np.zeros((n_seeds, n_feat), dtype=np.int16)
    mc[:, :] = 10
    for s, idx in enumerate(aligned_per_seed):
        for i in idx:
            mc[s, i] = 20          # aligned r_f is higher by 10/39 everywhere

    ap = tmp_path / "ap.h5"
    with h5py.File(ap, "w") as hf:
        hf.create_dataset("fire_counts", data=fire)
        hf.create_dataset("seeds", data=np.array(seeds,
                                                 dtype=h5py.string_dtype()))
        hf.attrs["n_rows"] = n_rows
    c3 = tmp_path / "c3.h5"
    with h5py.File(c3, "w") as hf:
        hf.create_dataset("cell_a", data=cell)
        hf.create_dataset("cell_b", data=cell)
        hf.create_dataset("seeds", data=np.array(seeds,
                                                 dtype=h5py.string_dtype()))
    rf = tmp_path / "rf.h5"
    with h5py.File(rf, "w") as hf:
        hf.create_dataset("match_count", data=mc)
        hf.attrs["seed_labels"] = json.dumps(list(range(n_seeds)))
        hf.attrs["n_compare_seeds"] = 39
        hf.attrs["theta"] = 0.7
    cs = tmp_path / "corrector.json"
    cs.write_text(json.dumps({"seeds": [
        {"ckpt": f"/x/{s}", "flagged_features": []} for s in seeds]}))
    return {"ap": ap, "c3": c3, "rf": rf, "cs": cs}


def _run(tmp_path, paths):
    out, rep = tmp_path / "o.h5", tmp_path / "r.json"
    r = subprocess.run(
        [sys.executable, "-m", "src.analysis.matched_pair_arm",
         "--ap-experimental", str(paths["ap"]),
         "--classification-2x2", str(paths["c3"]),
         "--rf-counts", str(paths["rf"]),
         "--corrector-scan", str(paths["cs"]),
         "--out", str(out), "--report", str(rep)],
        cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(rep.read_text()), r.stdout


def test_a_seed_with_NO_treated_features_is_carried_as_its_own_category(tmp_path):
    """§5b, end to end. Seed 1 contributes nothing. The known answer, written
    down before running: three seeds in the denominator, one of them in the
    NO_TREATED_FEATURES category, TWO contributing -- and the empty seed
    carries no p-value and no zero-valued contrast."""
    paths = _write_fixture(tmp_path, [[0, 1], [], [2]])
    report, _ = _run(tmp_path, paths)
    b = report["per_arm"]["B"]["with_correctors"]
    assert b["seed_branch_counts"]["NO_TREATED_FEATURES"] == 1
    assert b["seed_branch_counts"]["CONTRIBUTES"] == 2
    assert b["seeds_with_no_treated_feature"] == ["seed_1"]
    assert b["across_seeds"]["denominator_seeds_total"] == 3
    assert b["across_seeds"]["denominator_seeds_contributing"] == 2
    assert b["across_seeds"]["censored"] is True
    empty = [r for r in b["per_seed"] if r["seed"] == "seed_1"][0]
    # The flag is ON THE ROW a reader lands on (BP4 item 1).
    assert empty["seed_branch"] == "NO_TREATED_FEATURES"
    assert empty["p_value"] is None
    assert empty["median_difference"] is None
    assert empty["n_treated"] == 0


def test_the_empty_seed_is_NOT_dropped_from_the_stated_denominator(tmp_path):
    """The failure this guards against: reporting '2 seeds' as though the arm
    had two anchors. The denominator must still read 3, and the summary must
    say it is censored."""
    paths = _write_fixture(tmp_path, [[0, 1], [], [2]])
    report, _ = _run(tmp_path, paths)
    a = report["per_arm"]["B"]["with_correctors"]["across_seeds"]
    assert a["denominator_seeds_total"] == 3
    assert a["median_difference"]["n_seeds_total"] == 2, \
        "the across-seed spread is over CONTRIBUTING seeds and says so"
    assert a["censored"] is True


def test_a_run_where_EVERY_seed_is_empty_reports_zero_and_does_not_crash(tmp_path):
    """Arm A on the real data has ALIGNED = 0 in all twenty seeds. A component
    that skipped or crashed on that would report a pool nobody examined."""
    paths = _write_fixture(tmp_path, [[], [], []])
    report, _ = _run(tmp_path, paths)
    b = report["per_arm"]["B"]["with_correctors"]
    assert b["seed_branch_counts"]["NO_TREATED_FEATURES"] == 3
    assert b["seed_branch_counts"]["CONTRIBUTES"] == 0
    assert b["n_treated_total"] == 0
    assert b["pooled_over_seeds"]["branch"] == "NO_PAIRS"


def test_the_driver_REFUSES_a_positional_seed_join(tmp_path):
    """gotcha 29. The r_f axis is relabelled so seed 2 is absent; the driver
    must refuse rather than join by position."""
    import h5py
    paths = _write_fixture(tmp_path, [[0, 1], [], [2]])
    with h5py.File(paths["rf"], "a") as hf:
        hf.attrs["seed_labels"] = json.dumps([0, 1, 7])
    out, rep = tmp_path / "o2.h5", tmp_path / "r2.json"
    r = subprocess.run(
        [sys.executable, "-m", "src.analysis.matched_pair_arm",
         "--ap-experimental", str(paths["ap"]),
         "--classification-2x2", str(paths["c3"]),
         "--rf-counts", str(paths["rf"]),
         "--corrector-scan", str(paths["cs"]),
         "--out", str(out), "--report", str(rep)],
        cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0
    assert "absent from the r_f axis" in (r.stdout + r.stderr)


def test_the_driver_REFUSES_a_corrector_scan_missing_a_seed(tmp_path):
    """The corrector exclusion is mandatory reporting; a seed absent from the
    scan cannot be silently skipped."""
    paths = _write_fixture(tmp_path, [[0, 1], [], [2]])
    paths["cs"].write_text(json.dumps({"seeds": [
        {"ckpt": "/x/seed_0", "flagged_features": []}]}))
    out, rep = tmp_path / "o3.h5", tmp_path / "r3.json"
    r = subprocess.run(
        [sys.executable, "-m", "src.analysis.matched_pair_arm",
         "--ap-experimental", str(paths["ap"]),
         "--classification-2x2", str(paths["c3"]),
         "--rf-counts", str(paths["rf"]),
         "--corrector-scan", str(paths["cs"]),
         "--out", str(out), "--report", str(rep)],
        cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0
    assert "corrector scan has no entry" in (r.stdout + r.stderr)


def test_a_flagged_corrector_is_actually_EXCLUDED_from_the_without_variant(tmp_path):
    """§5 requires the arm WITH and WITHOUT the flagged features. On the real
    data 0/20 seeds carry a candidate, so the two variants coincide -- which is
    precisely why the exclusion must be shown to BITE on a fixture that has
    one, or the 'without' column would be untested."""
    paths = _write_fixture(tmp_path, [[0, 1], [], [2]])
    paths["cs"].write_text(json.dumps({"seeds": [
        {"ckpt": "/x/seed_0", "flagged_features": [0]},
        {"ckpt": "/x/seed_1", "flagged_features": []},
        {"ckpt": "/x/seed_2", "flagged_features": []}]}))
    report, _ = _run(tmp_path, paths)
    with_ = report["per_arm"]["B"]["with_correctors"]
    without = report["per_arm"]["B"]["without_correctors"]
    assert with_["n_treated_total"] == 3
    assert without["n_treated_total"] == 2, \
        "feature 0 of seed_0 is flagged and must leave the treated set"
    assert report["corrector_exclusion"]["n_flagged_total"] == 1
