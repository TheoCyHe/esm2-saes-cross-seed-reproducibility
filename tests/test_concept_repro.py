"""Tests for concept-level reproducibility (amendment AW §1, run under AX §2d).

§4d of the governing brief: *construct a fixture in which a chosen set of
concepts is detectable in a chosen set of seeds and the rest are not, and
require the pipeline to return exactly that pattern.* A maximum-and-compare
pipeline produces plausible numbers from any input, so recovery of a known
pattern is the convincing test -- and the fixture is shown to DISCRIMINATE
(each guard is exercised against an input that violates it) before any result
computed with it is trusted.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from src.analysis.concept_repro import (
    REFERENCE_PERCENTILE,
    assert_labels_align,
    association_with_size,
    assert_shared_feature_axis,
    decile_table,
    detection_matrix,
    detection_rate,
    distribution_summary,
    estimable_percentiles,
    null_thresholds,
    observed_best_ap,
    run,
    spearman,
    to_gain,
)

N_SEEDS, N_FEATURES, N_SHIFTS, N_ROWS = 4, 5, 20, 100_000
CONCEPTS = ["BINDING:X", "DOMAIN:A", "TRANSMEM"]
SEEDS = ["seed_0", "seed_1", "seed_2", "seed_3"]
N_POSITIVES = np.array([10, 100, 1000], dtype=np.int64)
PREVALENCE = N_POSITIVES.astype(np.float64) / N_ROWS

# THE KNOWN TRUTH. Rows are seeds, columns are concepts.
#   BINDING:X detected in seeds 0 and 1      -> D = 0.50
#   DOMAIN:A  detected in every seed          -> D = 1.00
#   TRANSMEM  detected in none                -> D = 0.00
TARGET = np.array([
    [True,  True,  False],
    [True,  True,  False],
    [False, True,  False],
    [False, True,  False],
])
TARGET_D = np.array([0.5, 1.0, 0.0])

NULL_LEVEL = 0.10          # every null draw, for every concept
DETECTED_GAIN = 0.50       # comfortably above
MISSED_GAIN = 0.01         # comfortably below

IDENTITY = {
    "library_sha": "f67485d9",
    "truncation_policy": "untruncated",
    "scored_row_set": "label_store",
    "n_rows": N_ROWS,
    "argmax_pool_n": len(CONCEPTS),
    "n_concepts": len(CONCEPTS),
}


def _fire_counts(rng=None):
    """Distinct per (seed, feature) so a permutation of EITHER axis is visible.

    A constant array would make `assert_shared_feature_axis` pass under any
    permutation -- gotcha 16: a fixture that cannot exhibit the disputed
    condition cannot test for it.
    """
    return (np.arange(N_SEEDS * N_FEATURES).reshape(N_SEEDS, N_FEATURES) * 7
            + 13).astype(np.int64)


def _write_ap(path, *, target=TARGET, seeds=SEEDS, concepts=CONCEPTS,
              fire_counts=None, identity=None):
    """AP fixture realising `target` exactly.

    The observed best-AP is set to whatever value makes the GAIN equal
    DETECTED_GAIN or MISSED_GAIN for that (seed, concept), inverting
    gain = (ap - p)/(1 - p). The max over features is made non-trivial: the
    intended value sits at a different feature index per concept, so a pipeline
    that took feature 0, or the mean, would not recover the pattern.
    """
    ap = np.zeros((N_SEEDS, N_FEATURES, len(concepts)), dtype=np.float64)
    for c in range(len(concepts)):
        p = PREVALENCE[c]
        for s in range(N_SEEDS):
            gain = DETECTED_GAIN if target[s, c] else MISSED_GAIN
            best = gain * (1.0 - p) + p
            ap[s, :, c] = best * 0.3          # every other feature is lower
            ap[s, (c * 2 + 1) % N_FEATURES, c] = best
    with h5py.File(path, "w") as hf:
        hf.create_dataset("fire_counts",
                          data=_fire_counts() if fire_counts is None else fire_counts)
        hf.create_dataset("seeds", data=np.array(seeds, dtype=h5py.string_dtype()))
        g = hf.create_group("pooled")
        g.create_dataset("ap", data=ap)
        g.create_dataset("concepts",
                         data=np.array(concepts, dtype=h5py.string_dtype()))
        g.create_dataset("prevalence", data=PREVALENCE)
        g.create_dataset("n_positives", data=N_POSITIVES)
        g.create_dataset("in_argmax_pool", data=np.ones(len(concepts), dtype=bool))
        for k, v in (identity or IDENTITY).items():
            hf.attrs[k] = v
    return ap


def _write_null(path, *, seeds=SEEDS, concepts=CONCEPTS, fire_rate=None,
                with_dest=True, with_cmax_b=True, identity=None, level=NULL_LEVEL):
    fr = (_fire_counts().astype(np.float64) / N_ROWS) if fire_rate is None else fire_rate
    with h5py.File(path, "w") as hf:
        hf.create_dataset("concepts",
                          data=np.array(concepts, dtype=h5py.string_dtype()))
        hf.create_dataset("seeds", data=np.array(seeds, dtype=h5py.string_dtype()))
        hf.create_dataset("fire_rate", data=fr)
        if with_cmax_b:
            hf.create_dataset("null_cmax_b", data=np.full(
                (N_SEEDS, N_SHIFTS, len(concepts)), level, dtype=np.float64))
        hf.create_dataset("null_cmax_a", data=np.full(
            (N_SEEDS, N_SHIFTS, len(concepts)), 0.99, dtype=np.float64))
        if with_dest:
            hf.create_dataset("dest_schedule",
                              data=np.zeros((N_SHIFTS, 3), dtype=np.int64))
        for k, v in (identity or IDENTITY).items():
            hf.attrs[k] = v


# =========================================================== §4d: recovery ===

def test_the_pipeline_recovers_the_planted_detection_pattern_exactly(tmp_path):
    _write_ap(tmp_path / "ap.h5")
    _write_null(tmp_path / "null.h5")
    out = run(tmp_path / "ap.h5", tmp_path / "null.h5", tmp_path / "out.json")

    got = np.array([r["D"] for r in out["per_concept"]])
    assert got.tolist() == TARGET_D.tolist()
    assert [r["concept"] for r in out["per_concept"]] == CONCEPTS
    h = out["headline"]["distribution_of_D"]
    assert h["n_exactly_zero"] == 1 and h["n_exactly_one"] == 1
    assert h["n_interior"] == 1              # the 0.5 concept


def test_the_fixture_discriminates_a_wrong_answer(tmp_path):
    """The planted pattern must be one a broken pipeline would MISS.

    If the null level sat above every observed gain, D would be all-zero, and
    an all-zero result would also 'recover' a fixture whose target was all-zero.
    Asserting the target is not degenerate is what makes the recovery test
    informative (gotcha 25: ask which outcomes the instrument can produce).
    """
    assert len(set(TARGET_D.tolist())) == 3, "target must span 0, interior, 1"
    _write_ap(tmp_path / "ap.h5")
    _write_null(tmp_path / "null.h5", level=0.99)   # nothing can clear this
    out = run(tmp_path / "ap.h5", tmp_path / "null.h5", tmp_path / "out.json")
    got = np.array([r["D"] for r in out["per_concept"]])
    assert got.tolist() == [0.0, 0.0, 0.0]
    assert got.tolist() != TARGET_D.tolist()
    # A constant D makes the association undefined; it must be reported as
    # UNINFORMATIVE, not as 0.0 (which reads as "measured, no association").
    assert out["prevalence_association"]["status"] == "UNINFORMATIVE_CONSTANT_D"
    assert out["prevalence_association"]["AX_P2"]["measured"] is None


def test_a_pipeline_that_used_null_a_would_get_a_different_answer(tmp_path):
    """`null_cmax_a` is written at 0.99 in the fixture, so consuming it instead
    of `null_cmax_b` yields all-zero D. The two datasets are therefore
    distinguishable BY RESULT, not merely by name -- which is what makes §4c
    hazard 3 testable at all."""
    _write_ap(tmp_path / "ap.h5")
    _write_null(tmp_path / "null.h5")
    with h5py.File(tmp_path / "null.h5", "r") as hf:
        a, b = hf["null_cmax_a"][:], hf["null_cmax_b"][:]
    assert not np.array_equal(a, b)
    thr_a = null_thresholds(a, REFERENCE_PERCENTILE, "pooled")
    thr_b = null_thresholds(b, REFERENCE_PERCENTILE, "pooled")
    assert (thr_a > thr_b).all()


def test_refuses_a_null_without_cmax_b(tmp_path):
    _write_ap(tmp_path / "ap.h5")
    _write_null(tmp_path / "null.h5", with_cmax_b=False)
    with pytest.raises(SystemExit, match="ACROSS-PROTEIN"):
        run(tmp_path / "ap.h5", tmp_path / "null.h5", tmp_path / "out.json")


def test_refuses_a_null_without_dest_schedule(tmp_path):
    """`dest_schedule` is written only on a Null B run, so its absence means the
    array called `null_cmax_b` was not produced by the across-protein
    construction (§4c hazard 3)."""
    _write_ap(tmp_path / "ap.h5")
    _write_null(tmp_path / "null.h5", with_dest=False)
    with pytest.raises(SystemExit, match="dest_schedule"):
        run(tmp_path / "ap.h5", tmp_path / "null.h5", tmp_path / "out.json")


# ================================================= §4c hazard 1: seed order ===

def test_refuses_a_permuted_seed_axis(tmp_path):
    """18 of 20 rows mis-pair under the lexicographic/numeric split; here 2 of 4
    do, and the refusal must be positional -- the label SETS are identical."""
    permuted = [SEEDS[0], SEEDS[1], SEEDS[3], SEEDS[2]]
    assert set(permuted) == set(SEEDS)
    _write_ap(tmp_path / "ap.h5")
    _write_null(tmp_path / "null.h5", seeds=permuted)
    with pytest.raises(SystemExit, match="seed labels"):
        run(tmp_path / "ap.h5", tmp_path / "null.h5", tmp_path / "out.json")


def test_refuses_a_permuted_concept_axis(tmp_path):
    permuted = [CONCEPTS[1], CONCEPTS[0], CONCEPTS[2]]
    assert set(permuted) == set(CONCEPTS)
    _write_ap(tmp_path / "ap.h5")
    _write_null(tmp_path / "null.h5", concepts=permuted)
    with pytest.raises(SystemExit, match="concept labels"):
        run(tmp_path / "ap.h5", tmp_path / "null.h5", tmp_path / "out.json")


# ============================================== §4c hazard 2: feature axis ===

def test_refuses_a_permuted_feature_axis(tmp_path):
    """Same MULTISET of firing rates, different positions -- a check on a total,
    or on a rank correlation, would pass (gotcha 27's measured 0.977 on a
    mapping where 0.0031% of rows were right)."""
    fr = _fire_counts().astype(np.float64) / N_ROWS
    permuted = fr[:, [1, 0, 2, 3, 4]]
    assert np.array_equal(np.sort(fr, axis=None), np.sort(permuted, axis=None))
    _write_ap(tmp_path / "ap.h5")
    _write_null(tmp_path / "null.h5", fire_rate=permuted)
    with pytest.raises(SystemExit, match="fire_rate"):
        run(tmp_path / "ap.h5", tmp_path / "null.h5", tmp_path / "out.json")


def test_refuses_a_different_dictionary_width(tmp_path):
    _write_ap(tmp_path / "ap.h5")
    _write_null(tmp_path / "null.h5",
                fire_rate=np.zeros((N_SEEDS, N_FEATURES + 1)))
    with pytest.raises(SystemExit, match="SHAPE"):
        run(tmp_path / "ap.h5", tmp_path / "null.h5", tmp_path / "out.json")


def test_shared_feature_axis_returns_a_diff_not_a_verdict():
    fc = _fire_counts()
    got = assert_shared_feature_axis(fc / N_ROWS, fc, N_ROWS)
    assert got["n_agreeing"] == got["n_cells"] == N_SEEDS * N_FEATURES
    assert got["max_abs_delta"] == pytest.approx(0.0)


# ================================================== identity / absent keys ===

def test_an_absent_identity_key_is_refused_never_treated_as_matching(tmp_path):
    """The key is absent from BOTH artifacts -- the only case where the
    absent-is-not-matching branch is load-bearing.

    Dropping it from one side alone is caught by the ordinary comparison
    ("f67485d9" != "<ABSENT>"), so that fixture cannot exhibit the disputed
    condition and a mutation removing the branch survives it (gotcha 16). Two
    absent keys COMPARE EQUAL, which is precisely how `null_pilot_g.h5`'s
    missing `population` attr read as agreement (gotcha 15).
    """
    thin = {k: v for k, v in IDENTITY.items() if k != "library_sha"}
    _write_ap(tmp_path / "ap.h5", identity=thin)
    _write_null(tmp_path / "null.h5", identity=thin)
    with pytest.raises(SystemExit, match="library_sha"):
        run(tmp_path / "ap.h5", tmp_path / "null.h5", tmp_path / "out.json")


def test_a_key_absent_from_only_one_side_is_also_refused(tmp_path):
    thin = {k: v for k, v in IDENTITY.items() if k != "library_sha"}
    _write_ap(tmp_path / "ap.h5")
    _write_null(tmp_path / "null.h5", identity=thin)
    with pytest.raises(SystemExit, match="library_sha"):
        run(tmp_path / "ap.h5", tmp_path / "null.h5", tmp_path / "out.json")


def test_a_disagreeing_identity_key_is_refused(tmp_path):
    other = dict(IDENTITY, library_sha="deadbeef")
    _write_ap(tmp_path / "ap.h5")
    _write_null(tmp_path / "null.h5", identity=other)
    with pytest.raises(SystemExit, match="library_sha"):
        run(tmp_path / "ap.h5", tmp_path / "null.h5", tmp_path / "out.json")


# =================================================== estimability (AX §1b) ===

def test_400_pooled_draws_support_the_95th_but_not_the_999th():
    e = estimable_percentiles(400)
    assert e[95.0]["estimable"] and e[95.0]["expected_draws_at_or_above"] == 20.0
    assert e[99.0]["estimable"] and e[99.0]["expected_draws_at_or_above"] == 4.0
    assert not e[99.9]["estimable"]
    assert e[99.9]["expected_draws_at_or_above"] == pytest.approx(0.4)


def test_20_per_seed_draws_stop_at_the_95th():
    e = estimable_percentiles(20)
    assert e[95.0]["estimable"] and e[95.0]["expected_draws_at_or_above"] == 1.0
    assert not e[99.0]["estimable"]


def test_estimability_reports_the_count_for_percentiles_it_rejects():
    """A rejected percentile still reports its draw count, so a reader can
    re-derive the verdict rather than take it (gotcha 20)."""
    e = estimable_percentiles(10)
    assert not e[99.0]["estimable"]
    assert e[99.0]["expected_draws_at_or_above"] == pytest.approx(0.1)


def test_zero_draws_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        estimable_percentiles(0)


# ============================================================ the mechanics ===

def test_best_ap_maxes_over_features_not_seeds_or_concepts():
    ap = np.zeros((2, 3, 4))
    ap[1, 2, 3] = 0.9
    got = observed_best_ap(ap)
    assert got.shape == (2, 4)
    assert got[1, 3] == 0.9 and got[0, 3] == 0.0


def test_best_ap_refuses_a_two_dimensional_array():
    with pytest.raises(ValueError, match="seed, feature, concept"):
        observed_best_ap(np.zeros((2, 3)))


def test_gain_uses_a_per_concept_prevalence_not_a_scalar():
    best = np.array([[0.5, 0.5]])
    got = to_gain(best, np.array([0.0, 0.5]))
    # concept 0: (0.5-0)/1 = 0.5; concept 1: (0.5-0.5)/0.5 = 0.0
    assert got[0].tolist() == pytest.approx([0.5, 0.0])


def test_gain_refuses_a_prevalence_of_the_wrong_length():
    with pytest.raises(ValueError, match="one value per concept"):
        to_gain(np.zeros((2, 3)), np.array([0.1, 0.2]))


def test_gain_commutes_with_the_max_so_taking_the_max_first_is_exact():
    rng = np.random.default_rng(0)
    ap = rng.random((3, 7, 2))
    p = np.array([0.01, 0.2])
    max_then_gain = to_gain(ap.max(axis=1), p)
    gain_then_max = ((ap - p) / (1.0 - p)).max(axis=1)
    assert np.allclose(max_then_gain, gain_then_max)


def test_pooled_and_per_seed_thresholds_are_different_reductions():
    null = np.zeros((2, 4, 1))
    null[0, :, 0] = [0.0, 0.0, 0.0, 0.0]
    null[1, :, 0] = [1.0, 1.0, 1.0, 1.0]
    pooled = null_thresholds(null, 50.0, "pooled")
    per_seed = null_thresholds(null, 50.0, "per_seed")
    assert pooled.shape == (1,) and per_seed.shape == (2, 1)
    assert pooled[0] == pytest.approx(0.5)          # median of both seeds
    assert per_seed[:, 0].tolist() == [0.0, 1.0]    # each seed's own


def test_unknown_threshold_mode_is_refused():
    with pytest.raises(ValueError, match="unknown mode"):
        null_thresholds(np.zeros((2, 2, 2)), 50.0, "average")


def test_detection_is_strict_exceedance_not_greater_or_equal():
    obs = np.array([[0.10, 0.11]])
    det = detection_matrix(obs, np.array([0.10, 0.10]))
    assert det.tolist() == [[False, True]]


def test_detection_rate_is_over_seeds():
    det = np.array([[True, False], [True, True], [False, False], [False, True]])
    assert detection_rate(det).tolist() == [0.5, 0.5]


def test_detection_rate_refuses_a_non_boolean_matrix():
    with pytest.raises(ValueError, match="boolean"):
        detection_rate(np.array([[1, 0]]))


def test_mismatched_threshold_shape_is_refused():
    with pytest.raises(ValueError, match="do not match"):
        detection_matrix(np.zeros((2, 3)), np.zeros(4))


# ------------------------------------------------------------- distribution --

def test_interior_mass_excludes_the_endpoints():
    d = np.array([0.0, 0.05, 0.5, 0.95, 1.0])
    s = distribution_summary(d)
    assert s["n_exactly_zero"] == 1 and s["n_exactly_one"] == 1
    assert s["n_interior"] == 1                     # only 0.5
    assert s["interior_mass"] == pytest.approx(0.2)


def test_a_bimodal_and_a_uniform_distribution_are_distinguished():
    """AW-P1 turns on interior mass, so the summary must separate these two."""
    bimodal = np.concatenate([np.zeros(50), np.ones(50)])
    uniform = np.linspace(0, 1, 100)
    assert distribution_summary(bimodal)["interior_mass"] == 0.0
    assert distribution_summary(uniform)["interior_mass"] > 0.7


def test_histogram_bins_land_on_the_20ths_a_20_seed_rate_can_take():
    d = np.array([0.0, 0.05, 1.0])
    h = distribution_summary(d)["histogram_21_bins"]
    assert len(h) == 21 and h[0] == 1 and h[1] == 1 and h[20] == 1


# ----------------------------------------------------------------- spearman --

def test_spearman_matches_a_hand_computed_value():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 1.0, 4.0, 3.0, 5.0])
    # ranks differ by (+1,-1,+1,-1,0): sum d^2 = 4 -> 1 - 6*4/(5*24) = 0.8
    assert spearman(x, y) == pytest.approx(0.8)


def test_spearman_uses_average_ranks_so_ties_do_not_follow_input_order():
    x = np.array([1.0, 1.0, 1.0, 2.0])
    y = np.array([4.0, 3.0, 2.0, 1.0])
    forward = spearman(x, y)
    idx = [2, 1, 0, 3]
    assert spearman(x[idx], y[idx]) == pytest.approx(forward)
    assert forward == pytest.approx(-0.7745966692414834)


def test_spearman_refuses_a_constant_input_rather_than_reporting_zero():
    with pytest.raises(ValueError, match="constant"):
        spearman(np.ones(5), np.arange(5.0))


def test_spearman_is_perfect_on_a_monotone_transform():
    x = np.array([1.0, 10.0, 100.0, 1000.0])
    assert spearman(x, np.log(x)) == pytest.approx(1.0)


# ------------------------------------------------------------ decile table --

def test_decile_table_partitions_every_concept_exactly_once():
    rng = np.random.default_rng(1)
    size = rng.random(200) * 1000
    d = rng.random(200)
    tab = decile_table(d, size)
    assert sum(r["n_concepts"] for r in tab) == 200
    assert [r["decile"] for r in tab] == list(range(10))


def test_decile_table_orders_bins_by_size():
    size = np.arange(100.0) + 1
    d = np.zeros(100)
    tab = decile_table(d, size)
    maxes = [r["size_max"] for r in tab]
    assert maxes == sorted(maxes)


# ------------------------------------------------------------------- labels --

def test_label_alignment_refuses_a_length_difference():
    with pytest.raises(SystemExit, match="lengths differ"):
        assert_labels_align("x", ["a"], ["a", "b"])


def test_label_alignment_refuses_a_permutation_of_the_same_set():
    with pytest.raises(SystemExit, match="disagree at 2"):
        assert_labels_align("x", ["a", "b", "c"], ["a", "c", "b"])


def test_label_alignment_accepts_an_exact_match():
    assert_labels_align("x", ["a", "b"], ["a", "b"])


# ------------------------------------------------------------ the JSON out --

def test_the_written_json_carries_the_curve_and_names_dropped_percentiles(tmp_path):
    _write_ap(tmp_path / "ap.h5")
    _write_null(tmp_path / "null.h5")
    run(tmp_path / "ap.h5", tmp_path / "null.h5", tmp_path / "out.json")
    doc = json.loads((tmp_path / "out.json").read_text())
    # 4 seeds x 20 shifts = 80 pooled draws: the 99th needs 0.8 -> dropped.
    assert doc["null"]["draws_per_concept_pooled"] == 80
    assert doc["percentile_curve"]["99.0"]["pooled"] is None
    assert doc["percentile_curve"]["99.0"]["pooled_estimable"] is False
    assert doc["percentile_curve"]["95.0"]["pooled"] is not None
    assert doc["null"]["dataset"] == "null_cmax_b"


def test_the_association_is_reported_at_every_estimable_percentile(tmp_path):
    """AW1.5: a conclusion counts only if it holds ACROSS the reported
    percentile range. A coefficient near its pre-declared bar is exactly where
    a single-percentile reading would mislead, so the association is computed
    on every arm the draw count supports and dropped on the ones it does not."""
    _write_ap(tmp_path / "ap.h5")
    _write_null(tmp_path / "null.h5")
    out = run(tmp_path / "ap.h5", tmp_path / "null.h5", tmp_path / "out.json")
    for q, e in out["percentile_curve"].items():
        for arm in ("pooled", "per_seed"):
            if e[arm] is not None:
                assert "association_with_log_size" in e[arm]
                assert "status" in e[arm]["association_with_log_size"]


def test_association_names_a_constant_rather_than_reporting_zero():
    got = association_with_size(np.ones(5), np.arange(5.0))
    assert got["spearman"] is None
    assert got["status"] == "UNINFORMATIVE_CONSTANT_D"


def test_association_computes_when_d_varies():
    got = association_with_size(np.array([0.0, 0.5, 1.0]), np.log([10.0, 100.0, 1000.0]))
    assert got["status"] == "COMPUTED"
    assert got["spearman"] == pytest.approx(1.0)


def test_sd_a_is_never_written_without_mean_a_beside_it(tmp_path):
    """AW1.3: `sd_A` alone cannot distinguish 'consistently never captured'
    from 'consistently well captured'."""
    _write_ap(tmp_path / "ap.h5")
    _write_null(tmp_path / "null.h5")
    out = run(tmp_path / "ap.h5", tmp_path / "null.h5", tmp_path / "out.json")
    for rec in out["per_concept"]:
        assert "sd_A" in rec and "mean_A" in rec
