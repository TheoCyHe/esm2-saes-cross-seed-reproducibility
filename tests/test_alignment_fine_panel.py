"""Tests for `alignment_fine_panel` -- PREREGISTRATION amendment AZ section 12.

THE FIXTURES ARE BUILT TO DISCRIMINATE, which for this module means three
things a careless fixture would hide:

  * the interval basis. A bin whose features all share ONE `r_f` has an
    empirical SE of exactly 0 and a NON-zero naive binomial SE, so the two
    constructions cannot be confused for each other on this fixture. A fixture
    with dispersed values would let either one pass.
  * the monotonicity criterion. Its whole content is the "allowing for interval
    overlap" clause, so there is a fixture whose means DECREASE and whose
    intervals overlap (must pass) and one whose means decrease with disjoint
    intervals (must fail). A fixture with only increasing means tests nothing.
  * the seed axis. The permutation fixture's two orders differ at more than one
    position; an order in which `2` sits at index 2 cannot test an alignment
    (gotcha 29's corollary).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.analysis import alignment_fine_panel as fp


# ---------------------------------------------------------------- the bins

def test_top_fraction_mask_takes_the_quantile_over_KEPT_FEATURES_ONLY():
    """The 10 lowest-`log M` features are excluded by the floor. If the
    quantile were taken over ALL features, the top 20% of the kept set would
    be a different set from the top 20% of the whole -- and this fixture makes
    the two answers differ in COUNT, not merely in membership."""
    logM = np.arange(20, dtype=np.float64)
    keep = np.zeros(20, dtype=bool)
    keep[10:] = True                        # only the top half is kept
    m = fp.top_fraction_mask(logM, keep, 0.2)
    # The kept values are 10..19, so the 0.8 quantile is 17.2 and two features
    # clear it.
    assert m.sum() == 2
    assert set(np.flatnonzero(m).tolist()) == {18, 19}
    # Over ALL 20 features the 20% cut sits at 15.2, giving 4 -- a different
    # answer, which is what makes this fixture able to fail.
    assert (logM >= np.quantile(logM, 0.8)).sum() == 4


def test_top_fraction_mask_on_an_empty_keep_set_is_empty_not_everything():
    logM = np.arange(5, dtype=np.float64)
    m = fp.top_fraction_mask(logM, np.zeros(5, dtype=bool), 0.5)
    assert m.sum() == 0


def test_bin_by_quantile_gives_equal_counts_on_distinct_values():
    v = np.arange(100, dtype=np.float64)
    idx = fp.bin_by_quantile(v, 10)
    counts = np.bincount(idx, minlength=10)
    assert counts.tolist() == [10] * 10


# ------------------------------------------------------------ the intervals

def test_interval_is_the_EMPIRICAL_dispersion_not_the_binomial():
    """Every feature in the bin has r_f = 0.5 exactly, so the between-feature
    SD is 0 and the empirical SE is 0 -- while the naive binomial SE is
    sqrt(.25/n) > 0. A Wilson-style construction cannot produce this row."""
    logM = np.zeros(8)
    r_f = np.full(8, 0.5)
    rows = fp.bin_table(logM, r_f, 1)
    assert rows[0]["se_between_features"] == pytest.approx(0.0)
    assert rows[0]["naive_binomial_se_for_comparison"] > 0.0
    assert rows[0]["ci95_lo"] == rows[0]["ci95_hi"] == pytest.approx(0.5)


def test_a_single_feature_bin_reports_None_rather_than_a_zero_interval():
    """A one-feature bin has no between-feature dispersion at all. Reporting 0
    would read as 'measured, and exact'."""
    rows = fp.bin_table(np.array([0.0]), np.array([0.3]), 1)
    assert rows[0]["n_features"] == 1
    assert rows[0]["sd_r_f_between_features"] is None
    assert rows[0]["ci95_lo"] is None and rows[0]["ci95_hi"] is None


def test_empty_bins_are_reported_as_EMPTY_not_dropped():
    """All values tied, so nine of ten quantile bins are empty."""
    rows = fp.bin_table(np.zeros(10), np.full(10, 0.5), 10)
    assert sum(1 for r in rows if r["status"] == "EMPTY") == 9
    assert len(rows) == 10


def test_the_ratio_against_the_naive_se_is_recomputed_not_imported():
    """AZ6.1: no multiplier from another reduction. The reported ratio must be
    this bin's own empirical SE over this bin's own binomial SE."""
    r_f = np.array([0.0, 0.0, 1.0, 1.0])
    rows = fp.bin_table(np.zeros(4), r_f, 1)
    sd = float(r_f.std(ddof=1))
    assert rows[0]["se_between_features"] == pytest.approx(sd / 2.0)
    assert rows[0]["ratio_empirical_over_naive"] == pytest.approx(
        (sd / 2.0) / np.sqrt(0.5 * 0.5 / 4))


# ------------------------------------------------------------ monotonicity

def _rows(means, half):
    return [{"bin": i, "n_features": 10, "status": "COMPUTED",
             "mean_r_f": m, "ci95_lo": m - half, "ci95_hi": m + half}
            for i, m in enumerate(means)]


def test_monotone_verdict_accepts_a_strictly_increasing_sequence():
    v = fp.monotone_verdict(_rows([0.1, 0.2, 0.3], 0.01))
    assert v["verdict"] == "MONOTONE_NON_DECREASING"
    assert v["n_increasing"] == 2 and v["n_violations"] == 0


def test_monotone_verdict_RESCUES_a_dip_whose_intervals_overlap():
    """This is the clause the predicate was declared with. Without it the
    verdict would be NOT_MONOTONE on a dip well inside the noise."""
    v = fp.monotone_verdict(_rows([0.30, 0.29, 0.31], 0.05))
    assert v["verdict"] == "MONOTONE_NON_DECREASING"
    assert v["n_decreasing_but_intervals_overlap"] == 1


def test_monotone_verdict_fails_on_a_dip_whose_intervals_are_DISJOINT():
    v = fp.monotone_verdict(_rows([0.30, 0.10, 0.31], 0.01))
    assert v["verdict"] == "NOT_MONOTONE"
    assert v["n_violations"] == 1


def test_monotone_verdict_has_a_third_outcome_for_too_few_bins():
    """A two-valued verdict on an ordered comparison is not total (gotcha 24),
    and 'no usable bins' must not read as 'not monotone'."""
    v = fp.monotone_verdict([{"bin": 0, "status": "EMPTY"}])
    assert v["verdict"] == "UNDEFINED_TOO_FEW_BINS"
    assert "n_violations" not in v


def test_monotone_verdict_treats_a_missing_interval_as_not_overlapping():
    """A single-feature bin has no interval, so it cannot RESCUE a dip. The
    conservative reading is a violation, not a free pass."""
    rows = _rows([0.30, 0.10], 0.01)
    rows[1]["ci95_lo"] = rows[1]["ci95_hi"] = None
    v = fp.monotone_verdict(rows)
    assert v["verdict"] == "NOT_MONOTONE"


# ------------------------------------------------------- combining by seed

def test_combine_across_seeds_summarises_per_seed_means_not_pooled_features():
    per_seed = [
        {"bins": _rows([0.2, 0.4], 0.01), "monotone": {"verdict": "A"}},
        {"bins": _rows([0.6, 0.8], 0.01), "monotone": {"verdict": "A"}},
    ]
    out = fp.combine_across_seeds(per_seed, 2)
    assert out["bins"][0]["mean_of_per_seed_means"] == pytest.approx(0.4)
    assert out["bins"][0]["n_seeds"] == 2
    assert out["per_seed_verdict_counts"] == {"A": 2}


def test_a_bin_absent_in_one_seed_is_absent_not_zero():
    per_seed = [
        {"bins": _rows([0.2, 0.4], 0.01), "monotone": {"verdict": "A"}},
        {"bins": [{"bin": 0, "status": "EMPTY"},
                  {"bin": 1, "status": "COMPUTED", "mean_r_f": 0.8,
                   "ci95_lo": 0.7, "ci95_hi": 0.9}],
         "monotone": {"verdict": "B"}},
    ]
    out = fp.combine_across_seeds(per_seed, 2)
    assert out["bins"][0]["n_seeds"] == 1
    assert out["bins"][0]["mean_of_per_seed_means"] == pytest.approx(0.2)
    assert out["per_seed_verdict_counts"] == {"A": 1, "B": 1}


def test_a_bin_empty_in_every_seed_is_named_rather_than_dropped():
    per_seed = [{"bins": [{"bin": 0, "status": "EMPTY"}],
                 "monotone": {"verdict": "UNDEFINED_TOO_FEW_BINS"}}]
    out = fp.combine_across_seeds(per_seed, 1)
    assert out["bins"][0]["status"] == "EMPTY_IN_ALL_SEEDS"


# --------------------------------------------------------------- seed axis

def test_seed_permutation_maps_by_label_and_differs_from_the_identity():
    src = [0, 1, 10, 2]          # lexicographic
    dst = [0, 1, 2, 10]          # numeric
    perm = fp.seed_permutation(src, dst, "fixture")
    assert perm.tolist() == [0, 1, 3, 2]
    assert perm.tolist() != list(range(4))
    assert [src[i] for i in perm] == dst


def test_seed_permutation_refuses_a_different_seed_set():
    with pytest.raises(SystemExit):
        fp.seed_permutation([0, 1, 2], [0, 1, 3], "fixture")


def test_firing_bin_uses_the_project_strata():
    fr = np.array([0.0, 5e-5, 2e-4, 5e-4, 2e-3, 5e-3, 2e-2, 5e-2, 0.2, 0.7])
    b = fp.firing_bin(fr)
    assert b.tolist() == [0, 0, 1, 2, 3, 4, 5, 6, 7, 8]


# ------------------------------------------------------------- the whole arm

def test_panel_bins_per_seed_and_never_pools_the_feature_axis():
    """Seed 1's `log M` is shifted far above seed 0's. Per-seed binning gives
    each seed its own top decile; a pooled binning would put every seed-0
    feature outside the top decile entirely and one of the two per-seed rows
    would be empty."""
    rng = np.random.default_rng(0)
    logM = np.vstack([rng.normal(0, 1, 400), rng.normal(50, 1, 400)])
    r_f = rng.random((2, 400))
    keep = np.ones((2, 400), dtype=bool)
    out = fp.panel(logM, r_f, keep, 0.10, 5, "fixture")
    assert [s["n_features"] for s in out["per_seed"]] == [40, 40]
    for s in out["per_seed"]:
        assert all(r["status"] == "COMPUTED" for r in s["bins"])


def test_the_combined_verdict_is_derived_from_the_combined_means():
    """AZ-P3's verdict at the combined level must be a FUNCTION of the combined
    sequence, not a sentence. The fixture's per-seed arms are each monotone
    while their MEANS dip with disjoint across-seed intervals, so a verdict
    copied from the per-seed counts would say MONOTONE and the derived one
    says NOT."""
    per_seed = [
        {"bins": _rows([0.10, 0.90], 0.001), "monotone": {"verdict": "M"}},
        {"bins": _rows([0.11, 0.89], 0.001), "monotone": {"verdict": "M"}},
    ]
    up = fp.combine_across_seeds(per_seed, 2)
    assert up["monotone_on_the_combined_sequence"][
        "verdict"] == "MONOTONE_NON_DECREASING"

    down = fp.combine_across_seeds(
        [{"bins": _rows([0.90, 0.10], 0.001), "monotone": {"verdict": "M"}},
         {"bins": _rows([0.89, 0.11], 0.001), "monotone": {"verdict": "M"}}], 2)
    assert down["monotone_on_the_combined_sequence"][
        "verdict"] == "NOT_MONOTONE"
    assert down["per_seed_verdict_counts"] == {"M": 2}


# ============================================================================
# BA11 -- the restated criterion (amendment BA section 7.4)
# ============================================================================
# THE DISCRIMINATING FIXTURE IS `test_the_replacement_catches_the_stratum_1
# _case`. AZ13.1's defect is a sequence that satisfies the adjacent-pair rule
# while ENDING BELOW WHERE IT STARTED, and a fixture whose sequence rises
# cannot exhibit it. So the fixture is built to fall, with each individual
# step inside its interval -- exactly stratum 1's shape.

def _seed(bins):
    """A per-seed record from a list of (mean, se) pairs."""
    rows = []
    for i, (m, se) in enumerate(bins):
        rows.append({"bin": i, "status": "COMPUTED", "n_features": 100,
                     "mean_r_f": m, "se_between_features": se,
                     "ci95_lo": m - fp.Z95 * se,
                     "ci95_hi": m + fp.Z95 * se})
    return {"bins": rows, "monotone": fp.monotone_verdict(rows)}


def test_per_seed_span_is_last_usable_minus_first():
    s = _seed([(0.50, 0.01), (0.55, 0.01), (0.60, 0.01)])
    assert fp.per_seed_span(s["bins"]) == pytest.approx(0.10)


def test_per_seed_span_is_None_with_fewer_than_two_usable_bins():
    assert fp.per_seed_span([{"bin": 0, "status": "EMPTY"}]) is None
    assert fp.per_seed_span([]) is None
    # THE DISCRIMINATING CASE: three rows, only ONE of them usable. A version
    # that counted rows rather than USABLE rows would pass the `< 2` check and
    # then read `mean_r_f` off an EMPTY bin. Without this the mutation
    # "EMPTY bins are not skipped" survives.
    assert fp.per_seed_span([{"bin": 0, "status": "EMPTY"},
                             {"bin": 1, "status": "COMPUTED",
                              "mean_r_f": 0.5},
                             {"bin": 2, "status": "EMPTY"}]) is None


def test_per_seed_span_ignores_EMPTY_bins_at_the_ends():
    rows = [{"bin": 0, "status": "EMPTY"},
            {"bin": 1, "status": "COMPUTED", "mean_r_f": 0.40},
            {"bin": 2, "status": "COMPUTED", "mean_r_f": 0.50},
            {"bin": 3, "status": "EMPTY"}]
    assert fp.per_seed_span(rows) == pytest.approx(0.10)


def test_the_replacement_catches_the_stratum_1_case():
    """THE FIXTURE THAT DISCRIMINATES, and it is AZ13.1's actual defect.

    A sequence that FALLS overall while every adjacent step is inside the
    noise. The incumbent scores it MONOTONE_NON_DECREASING; the replacement
    must not call it a rise."""
    wide = 0.05          # each step is inside its own interval
    falling = [(0.60, wide), (0.58, wide), (0.56, wide), (0.54, wide)]
    per_seed = [_seed(falling) for _ in range(20)]
    old = fp.monotone_verdict(_seed(falling)["bins"])
    new = fp.span_verdict(per_seed)
    assert old["verdict"] == "MONOTONE_NON_DECREASING", (
        "the fixture does not exhibit AZ13.1's defect")
    assert old["total_span_pp"] < 0
    assert new["verdict"] == "FALLS"
    r = fp.restated_verdict(old, new)
    assert r["old_criterion_reads_as_a_rise_but_the_span_does_not"] is True


def test_a_genuine_rise_is_scored_as_one_by_both_criteria():
    rising = [(0.50, 0.002), (0.53, 0.002), (0.56, 0.002), (0.59, 0.002)]
    per_seed = [_seed(rising) for _ in range(20)]
    old = fp.monotone_verdict(_seed(rising)["bins"])
    new = fp.span_verdict(per_seed)
    assert old["verdict"] == "MONOTONE_NON_DECREASING"
    assert new["verdict"] == "RISES"
    assert fp.restated_verdict(old, new)[
        "old_criterion_reads_as_a_rise_but_the_span_does_not"] is False


def test_a_span_whose_interval_straddles_zero_is_FLAT_not_a_rise():
    """The outcome AZ13.1 could not name.

    THE MEAN SPAN IS DELIBERATELY POSITIVE (+1 pp) while its interval still
    straddles zero. A fixture whose mean is EXACTLY zero cannot discriminate:
    a version scoring the direction on the point estimate alone would also say
    FLAT, and the mutation `lo > 0 -> mean > 0` survived the first draft of
    this test for exactly that reason. The interval has to be the thing doing
    the work."""
    up = _seed([(0.50, 0.01), (0.70, 0.01)])       # span +20 pp
    down = _seed([(0.60, 0.01), (0.42, 0.01)])     # span -18 pp
    new = fp.span_verdict([up, down] * 10)
    assert new["mean_span_pp"] > 0, "the fixture must have a POSITIVE mean"
    assert new["verdict"] == "INDISTINGUISHABLE_FROM_FLAT"
    assert new["ci95_span_pp"][0] < 0 < new["ci95_span_pp"][1]


def test_span_verdict_is_total_and_every_outcome_is_reachable():
    """Gotcha 25: confirm the instrument can produce more than one outcome."""
    up = _seed([(0.50, 0.001), (0.60, 0.001)])
    down = _seed([(0.60, 0.001), (0.50, 0.001)])
    # The FLAT case uses the ASYMMETRIC pair, so its mean span is POSITIVE
    # (+1 pp) while the interval straddles zero. A symmetric pair has a mean of
    # exactly zero and a point-estimate rule would also say FLAT -- the case
    # would then be reachable under the mutation too, and this test would stop
    # discriminating even though the totality assertion still held.
    flat = [_seed([(0.50, 0.01), (0.70, 0.01)]),
            _seed([(0.60, 0.01), (0.42, 0.01)])] * 10
    assert fp.span_verdict(flat)["mean_span_pp"] > 0
    got = {
        fp.span_verdict([up] * 20)["verdict"],
        fp.span_verdict([down] * 20)["verdict"],
        fp.span_verdict(flat)["verdict"],
        fp.span_verdict([])["verdict"],
        fp.span_verdict([up])["verdict"],
    }
    assert got == {"RISES", "FALLS", "INDISTINGUISHABLE_FROM_FLAT",
                   "UNDEFINED_NO_SEED_HAS_TWO_USABLE_BINS",
                   "UNDEFINED_SINGLE_SEED_NO_INTERVAL"}


def test_the_span_interval_is_PAIRED_not_a_difference_of_two_bin_intervals():
    """Every seed has the SAME span but wildly different levels. A paired
    construction gives a zero-width interval; differencing two bin means'
    across-seed intervals would give a wide one."""
    per_seed = [_seed([(0.20 + 0.03 * k, 0.01), (0.30 + 0.03 * k, 0.01)])
                for k in range(20)]
    new = fp.span_verdict(per_seed)
    assert new["mean_span_pp"] == pytest.approx(10.0)
    assert new["sd_span_across_seeds_pp"] == pytest.approx(0.0, abs=1e-9)
    assert new["verdict"] == "RISES"


def test_combine_across_seeds_reports_BOTH_verdicts():
    per_seed = [_seed([(0.50, 0.002), (0.60, 0.002)]) for _ in range(20)]
    comb = fp.combine_across_seeds(per_seed, 2)
    assert "monotone_on_the_combined_sequence" in comb
    assert comb["span_direction_BA11"]["verdict"] == "RISES"
    assert comb["restated_BA11"]["old_verdict_adjacent_pair_criterion"] \
        == "MONOTONE_NON_DECREASING"


# ---------------------------------- BA11's acceptance criterion (BA7.4)

def _panel_doc(mean_last=0.60, with_stratum=True):
    """A minimal artifact of the shape `reproduces_reference` reads."""
    per_seed = [_seed([(0.50, 0.002), (mean_last, 0.002)]) for _ in range(3)]
    arm = {"combined": fp.combine_across_seeds(per_seed, 2)}
    doc = {"identity": {"measurement": "fixture"},
           "arms": {"top_decile": arm}}
    if with_stratum:
        doc["by_firing_stratum"] = [{"stratum": 0, "status": "COMPUTED",
                                     **arm}]
    return doc, arm


def test_reproduces_reference_accepts_an_exact_match(tmp_path):
    doc, arm = _panel_doc()
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps(doc))
    got = fp.reproduces_reference({"top_decile": arm},
                                  doc["by_firing_stratum"], ref)
    assert got["verdict"] == "REPRODUCES_REFERENCE_EXACTLY"
    assert got["problems"] == []


def test_reproduces_reference_refuses_a_moved_bin_mean(tmp_path):
    """The criterion's whole job: if an OLD number moved, the old-vs-new
    comparison is uninterpretable and the run must stop."""
    doc, _ = _panel_doc(mean_last=0.60)
    _, moved = _panel_doc(mean_last=0.61)
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps(doc))
    got = fp.reproduces_reference({"top_decile": moved},
                                  doc["by_firing_stratum"], ref)
    assert got["verdict"] == "DIFFERS"
    assert got["problems"]


def test_reproduces_reference_reports_an_ABSENT_arm_as_absent(tmp_path):
    """Gotcha 15 / I3: absent is never agreement."""
    doc, arm = _panel_doc()
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"identity": {}, "arms": {},
                               "by_firing_stratum": []}))
    got = fp.reproduces_reference({"top_decile": arm}, [], ref)
    assert got["verdict"] == "DIFFERS"
    assert any("ABSENT" in p for p in got["problems"])


def test_reproduces_reference_ignores_the_NEW_verdict(tmp_path):
    """`span_direction_BA11` is the addition. Comparing it against a reference
    that predates it would report a difference for the one thing meant to be
    new, so the fingerprint must exclude it."""
    doc, arm = _panel_doc()
    stripped = json.loads(json.dumps(doc))
    for a in stripped["arms"].values():
        a["combined"].pop("span_direction_BA11")
        a["combined"].pop("restated_BA11")
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps(stripped))
    got = fp.reproduces_reference({"top_decile": arm},
                                  stripped["by_firing_stratum"], ref)
    assert got["verdict"] == "REPRODUCES_REFERENCE_EXACTLY"


def test_the_summary_formatter_renders_an_UNDEFINED_span():
    """A stratum whose top-decile slice leaves fewer than two usable bins in
    every seed has NO `mean_span_pp` at all. That is a real outcome, and a
    formatter that cannot render it turned a completed measurement into a
    FAILED job -- the artifact was already on disk and only the print died."""
    undefined = fp.span_verdict([])
    assert "mean_span_pp" not in undefined
    r = fp.restated_verdict({"verdict": "NOT_MONOTONE", "total_span_pp": None},
                            undefined)
    assert r["new_mean_span_pp"] is None
    assert fp._pp(r["new_mean_span_pp"]) == "n/a"
    assert fp._ci(r["new_ci95_span_pp"]) == "[n/a, n/a]"
    assert fp._pp(1.234) == "+1.23"
    assert fp._ci([-1.0, 2.0]) == "[-1.00, +2.00]"
