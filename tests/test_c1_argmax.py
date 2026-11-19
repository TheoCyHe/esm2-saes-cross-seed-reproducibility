"""Tests for AL2's restatement (PREREGISTRATION amendment AO section 7).

MUTATION-TESTED; the mutations are listed beside the tests they must kill and
are actually run (see the session's mutation harness).
"""

from __future__ import annotations

import json
import sys

import h5py
import numpy as np
import pytest

from src.analysis import c1_argmax as C

IDENT = {"library_sha": "abc", "n_concepts": 6, "argmax_pool_n": 5,
         "truncation_policy": "untruncated", "scored_row_set": "label_store"}
EDGES = [0.0, 0.001, 0.01, 1.000000001]          # 3 firing bins


def _write_gtilde(path, stratum_mu, cell, argmax, n_concepts=6, ident=None,
                  extra_attrs=None, seeds=None):
    with h5py.File(path, "w") as hf:
        if seeds is not None:
            hf.create_dataset("seeds", data=np.array(
                [str(x) for x in seeds], dtype=h5py.string_dtype()))
        hf.create_dataset("stratum_mu", data=np.asarray(stratum_mu, float))
        hf.create_dataset("concept_cell", data=np.asarray(cell, np.int64))
        hf.create_dataset("argmax_concept", data=np.asarray(argmax, np.int32))
        hf.create_dataset("concepts",
                          data=np.array([f"C{i}" for i in range(n_concepts)],
                                        dtype=h5py.string_dtype()))
        hf.attrs.update(**{**(ident if ident is not None else IDENT)})
        hf.attrs["firing_rate_edges"] = json.dumps(EDGES)
        if extra_attrs:
            hf.attrs.update(**extra_attrs)
    return path


def _write_mu_smooth(path, mu_hat_full, ident=None, seeds=None):
    with h5py.File(path, "w") as hf:
        if seeds is not None:
            hf.create_dataset("seeds", data=np.array(
                [str(x) for x in seeds], dtype=h5py.string_dtype()))
        hf.create_dataset("mu_hat_full", data=np.asarray(mu_hat_full,
                                                         np.float32))
        hf.attrs.update(**{**(ident if ident is not None else IDENT)})
    return path


def _write_fire(path, fire_rate, seeds=None):
    with h5py.File(path, "w") as hf:
        hf.create_dataset("fire_rate", data=np.asarray(fire_rate, float))
        if seeds is not None:
            hf.create_dataset("seeds", data=np.array(
                [f"seed_{x}" for x in seeds], dtype=h5py.string_dtype()))
    return path


def _write_rf(path, counts, n_compare, theta=0.70, sim="decoder_cosine",
              seeds=None):
    with h5py.File(path, "w") as hf:
        hf.create_dataset("match_count", data=np.asarray(counts, np.int32))
        if seeds is not None:
            hf.attrs["seed_labels"] = json.dumps([int(x) for x in seeds])
        hf.attrs["n_compare_seeds"] = n_compare
        hf.attrs["theta"] = theta
        hf.attrs["similarity"] = sim
    return path


# ---------------------------------------------------------------------------
# Identity. An ABSENT key is reported absent and REFUSED (I3, gotcha 15).
# MUTATION: skip missing keys instead of recording a problem -> the first test
# dies.
# ---------------------------------------------------------------------------
def test_absent_identity_key_is_refused_not_treated_as_agreement(tmp_path):
    a = _write_gtilde(tmp_path / "a.h5", np.ones((1, 1, 3)), [0], [[0]])
    short = {k: v for k, v in IDENT.items() if k != "argmax_pool_n"}
    b = _write_gtilde(tmp_path / "b.h5", np.ones((1, 1, 3)), [0], [[0]],
                      ident=short)
    res = C.check_shared_identity({"a": a, "b": b})
    assert any("argmax_pool_n" in p and "ABSENT" in p for p in res["problems"])
    assert res["per_key"]["argmax_pool_n"]["b"] == "<ABSENT>"


def test_disagreeing_identity_key_is_refused(tmp_path):
    a = _write_gtilde(tmp_path / "a.h5", np.ones((1, 1, 3)), [0], [[0]])
    b = _write_gtilde(tmp_path / "b.h5", np.ones((1, 1, 3)), [0], [[0]],
                      ident={**IDENT, "argmax_pool_n": 4})
    res = C.check_shared_identity({"a": a, "b": b})
    assert any("argmax_pool_n" in p and "disagree" in p for p in res["problems"])


def test_agreeing_identity_passes_clean(tmp_path):
    a = _write_gtilde(tmp_path / "a.h5", np.ones((1, 1, 3)), [0], [[0]])
    b = _write_gtilde(tmp_path / "b.h5", np.ones((1, 1, 3)), [0], [[0]])
    assert C.check_shared_identity({"a": a, "b": b})["problems"] == []


# ---------------------------------------------------------------------------
# 2a. Three monotonicity outcomes, all reachable. A two-valued verdict on an
# ordered comparison is not total (gotcha 24).
# MUTATION: collapse the NEITHER branch into MONOTONE_INCREASING -> the
# non-monotone case dies.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("r_by_level,expected", [
    ([0.1, 0.5, 0.9], "MONOTONE_INCREASING"),
    ([0.9, 0.5, 0.1], "MONOTONE_DECREASING"),
    ([0.1, 0.9, 0.5], "NEITHER"),
])
def test_all_three_monotonicity_outcomes_are_reachable(r_by_level, expected):
    # three levels of mu, ascending; r_f constant within each level
    mu = np.array([[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]])
    r = np.array([[r_by_level[0], r_by_level[0], r_by_level[1], r_by_level[1],
                   r_by_level[2], r_by_level[2]]])
    got = C.conditional_means_by_level(mu, r)
    assert got["per_seed"][0]["monotonicity"] == expected
    assert got["per_seed"][0]["n_distinct_mu"] == 3
    assert got["per_seed"][0]["E_r_f_by_level_ascending_mu"] == pytest.approx(
        r_by_level)


def test_a_single_level_is_UNDEFINED_not_monotone():
    mu = np.array([[2.0, 2.0, 2.0]])
    r = np.array([[0.1, 0.2, 0.3]])
    got = C.conditional_means_by_level(mu, r)["per_seed"][0]
    assert got["monotonicity"] == "UNDEFINED_TOO_FEW_LEVELS"
    assert got["n_distinct_mu"] == 1


def test_level_count_is_measured_not_hardcoded_at_six():
    mu = np.array([[1.0, 2.0, 3.0, 4.0]])
    r = np.array([[0.1, 0.2, 0.3, 0.4]])
    got = C.conditional_means_by_level(mu, r)
    assert got["n_distinct_mu_by_seed"] == [4]
    assert len(got["per_seed"][0]["levels"]) == 4


def test_nonfinite_mu_is_counted_and_excluded():
    mu = np.array([[1.0, np.nan, 2.0]])
    r = np.array([[0.1, 0.9, 0.2]])
    got = C.conditional_means_by_level(mu, r)["per_seed"][0]
    assert got["n_nonfinite_mu"] == 1
    assert got["n_distinct_mu"] == 2
    assert sum(lv["n_features"] for lv in got["levels"]) == 2


# ---------------------------------------------------------------------------
# Pooling is by LEVEL RANK, not by mu value: the six values differ between
# seeds, so pooling on the value would make singleton groups.
# MUTATION: group by value instead of rank -> this dies (4 groups, not 2).
# ---------------------------------------------------------------------------
def test_pooled_levels_group_by_rank_not_by_value():
    mu = np.array([[1.0, 1.0, 2.0, 2.0],          # seed 0 values 1, 2
                   [7.0, 7.0, 9.0, 9.0]])         # seed 1 values 7, 9
    r = np.array([[0.1, 0.2, 0.8, 0.9],
                  [0.3, 0.4, 0.6, 0.7]])
    got = C.pooled_conditional_means(mu, r)
    assert [g["level_rank"] for g in got["by_level_rank"]] == [0, 1]
    assert [g["n_cells"] for g in got["by_level_rank"]] == [4, 4]
    assert got["by_level_rank"][0]["mu_range"] == [1.0, 7.0]


# ---------------------------------------------------------------------------
# mu at the argmax concept, hand-checked. This is the whole content of 2b.
# ---------------------------------------------------------------------------
def test_mu_at_argmax_hard_indexes_cell_of_the_winner_and_the_feature_bin(
        tmp_path):
    # 1 seed, 2 concept cells, 3 firing bins
    stratum_mu = np.array([[[10.0, 11.0, 12.0],       # cell 0
                            [20.0, 21.0, 22.0]]])     # cell 1
    cell = [0, 1, 1, 0]                                # concept -> cell
    argmax = [[2, 0, 3]]                               # 3 features
    g = _write_gtilde(tmp_path / "g.h5", stratum_mu, cell, argmax,
                      n_concepts=4)
    # fire rates put features in bins 0, 1, 2 respectively
    fire = np.array([[0.0005, 0.005, 0.5]])
    got = C.mu_at_argmax_hard(g, fire)
    # feature 0: winner concept 2 -> cell 1, bin 0 -> 20.0
    # feature 1: winner concept 0 -> cell 0, bin 1 -> 11.0
    # feature 2: winner concept 3 -> cell 0, bin 2 -> 12.0
    assert got.tolist() == [[20.0, 11.0, 12.0]]


def test_mu_at_argmax_smooth_indexes_the_smooth_arms_own_argmax(tmp_path):
    mu_full = np.arange(1 * 3 * 4, dtype=float).reshape(1, 3, 4)  # (seed, c, f)
    ms = _write_mu_smooth(tmp_path / "m.h5", mu_full)
    gs = _write_gtilde(tmp_path / "gs.h5", np.ones((1, 1, 3)), [0, 0, 0],
                       [[2, 0, 1, 2]], n_concepts=3)
    got = C.mu_at_argmax_smooth(ms, gs)
    assert got.tolist() == [[mu_full[0, 2, 0], mu_full[0, 0, 1],
                            mu_full[0, 1, 2], mu_full[0, 2, 3]]]


def test_mu_at_argmax_smooth_refuses_a_shape_mismatch(tmp_path):
    ms = _write_mu_smooth(tmp_path / "m.h5", np.ones((1, 3, 4)))
    gs = _write_gtilde(tmp_path / "gs.h5", np.ones((1, 1, 3)), [0],
                       [[0, 0, 0]], n_concepts=1)      # 3 features, not 4
    with pytest.raises(SystemExit, match="refusing to index"):
        C.mu_at_argmax_smooth(ms, gs)


# ---------------------------------------------------------------------------
# The argmax reduction must have MORE resolution than the 6-level one, or 2b is
# not measuring a different object. Asserted on a fixture, so the claim is
# structural rather than a property of one dataset.
# ---------------------------------------------------------------------------
def test_argmax_selection_has_more_distinct_values_than_the_median_reduction(
        tmp_path):
    from src.analysis.m4_confound import hard_bin_mu_per_feature
    rng = np.random.default_rng(0)
    stratum_mu = rng.uniform(1.0, 2.0, size=(1, 4, 3))
    cell = rng.integers(0, 4, size=20)
    argmax = rng.integers(0, 20, size=(1, 30))
    g = _write_gtilde(tmp_path / "g.h5", stratum_mu, cell, argmax,
                      n_concepts=20)
    fire = rng.choice([0.0005, 0.005, 0.5], size=(1, 30))
    six, _ = hard_bin_mu_per_feature(g, fire)
    am = C.mu_at_argmax_hard(g, fire)
    assert np.unique(six[0]).size <= 3            # bounded by n_firing_bins
    assert np.unique(am[0]).size > np.unique(six[0]).size


# ---------------------------------------------------------------------------
# c1's summary: the SE scale is AJ5's and rho^2 is reported.
# ---------------------------------------------------------------------------
def test_spearman_null_se_matches_AJ5s_number_at_10240_features():
    assert C.spearman_null_se(10240) == pytest.approx(0.00988, abs=5e-6)


def test_c1_reports_se_units_and_rho_squared():
    rng = np.random.default_rng(1)
    mu = rng.normal(size=(3, 500))
    r = 0.5 * mu + rng.normal(size=(3, 500))
    got = C.c1(mu, r)
    assert len(got["rho_by_seed"]) == 3
    assert got["median_abs_rho"] > 0.2
    assert got["median_abs_rho_in_se"] == pytest.approx(
        got["median_abs_rho"] / C.spearman_null_se(500))
    assert got["rho_squared_median"] == pytest.approx(
        np.median(np.abs(got["rho_by_seed"])) ** 2)


def test_c1_against_a_constant_normaliser_is_uninformative_not_zero():
    """A correlation against a constant is UNDEFINED. It must not silently read
    as 0.0, which would look like 'checked and absent' -- gotcha 25(b)."""
    mu = np.ones((1, 100))
    r = np.linspace(0, 1, 100).reshape(1, 100)
    got = C.c1(mu, r)
    assert np.isnan(got["rho_by_seed"][0])
    assert got["n_seeds_with_undefined_rho"] == 1
    assert got["status"].startswith("UNINFORMATIVE_CONSTANT")
    assert got["worst_seed"] is None


def test_c1_one_undefined_seed_does_not_erase_the_other_seeds():
    """A single constant seed must not turn the whole median into NaN and hide
    the informative seeds -- the summaries are nan-aware and the exclusion is
    named in `status`."""
    rng = np.random.default_rng(3)
    mu = rng.normal(size=(3, 300))
    mu[1] = 5.0                                   # seed 1 constant
    r = 0.6 * rng.normal(size=(3, 300)) + mu
    got = C.c1(mu, r)
    assert got["n_seeds_with_undefined_rho"] == 1
    assert got["status"].startswith("PARTIAL")
    assert np.isfinite(got["median_abs_rho"])
    assert got["worst_seed"] in (0, 2)


# ---------------------------------------------------------------------------
# THE ENTRY POINT (gotcha 13 extension 4, corollary).
# ---------------------------------------------------------------------------
# AMENDMENT AS. THE ALIGNMENT SIDE IS DELIBERATELY PERMUTED, because that is
# the condition this job shipped: `gtilde` / `mu_hat_full` / `fire_rate` are
# LEXICOGRAPHIC and `rf_counts` is NUMERIC. A fixture in which the two agree
# cannot exhibit the defect and therefore cannot test the fix (gotcha 16).
PERM = [2, 0, 1]


def _full_fixture(tmp_path, n_feat=60, n_concepts=8, n_cells=4,
                  label_alignment_side=True):
    """`r_f` is built to be a monotone function of the SAME seed's firing
    rate, so `|rho|` is large under a correct join and collapses under a
    mis-join. Without that the fixture would pass either way."""
    n_seeds = len(PERM)
    rng = np.random.default_rng(7)
    # `stratum_mu` MONOTONE in the firing bin, so the 6-level (here 3-level)
    # hard-bin reduction is a monotone step function of firing rate. With a
    # random `stratum_mu` the step is non-monotone and |rho| is capped near
    # 0.47 whatever the join -- which would leave the discrimination margin
    # too small to be a test.
    stratum_mu = (rng.uniform(1e-3, 3e-3, size=(n_seeds, n_cells, 1))
                  * np.array([1.0, 3.0, 9.0]))
    cell = rng.integers(0, n_cells, size=n_concepts)
    # per TRUE seed
    fire = rng.choice([0.0005, 0.005, 0.5], size=(n_seeds, n_feat))
    rank = np.argsort(np.argsort(fire, axis=1), axis=1) / n_feat
    k19 = np.clip((rank * 19).astype(int), 0, 19)
    k39 = np.clip((rank * 39).astype(int), 0, 39)

    lbl = PERM if label_alignment_side else None
    g = _write_gtilde(tmp_path / "g.h5", stratum_mu[PERM], cell,
                      rng.integers(0, n_concepts, size=(n_seeds, n_feat)),
                      n_concepts=n_concepts, seeds=lbl)
    gs = _write_gtilde(tmp_path / "gs.h5", stratum_mu[PERM], cell,
                       rng.integers(0, n_concepts, size=(n_seeds, n_feat)),
                       n_concepts=n_concepts, seeds=lbl)
    ms = _write_mu_smooth(tmp_path / "m.h5",
                          rng.uniform(1e-3, 1e-2,
                                      size=(n_seeds, n_concepts, n_feat)),
                          seeds=lbl)
    fr = _write_fire(tmp_path / "fr.h5", fire[PERM], seeds=PERM)
    rf19 = _write_rf(tmp_path / "rf19.h5", k19, 19, seeds=range(n_seeds))
    rf39 = _write_rf(tmp_path / "rf39.h5", k39, 39, seeds=range(n_seeds))
    return g, gs, ms, fr, rf19, rf39


def test_main_end_to_end_reports_both_arms_and_all_four_reductions(tmp_path,
                                                                  monkeypatch):
    g, gs, ms, fr, rf19, rf39 = _full_fixture(tmp_path)
    out = tmp_path / "o.json"
    monkeypatch.setattr(sys, "argv", [
        "c1a", "--gtilde", str(g), "--gtilde-smooth", str(gs),
        "--mu-smooth", str(ms), "--null-reductions", str(fr),
        "--rf-counts", str(rf19), str(rf39), "--out", str(out)])
    C.main()
    doc = json.loads(out.read_text())
    assert set(doc["by_rf_arm"]) == {"n_compare_19", "n_compare_39"}
    for arm in doc["by_rf_arm"].values():
        c1s = arm["AO7_2b_C1"]
        assert set(c1s) == {
            "hard_bin_argmax_selected", "smooth_argmax_selected",
            "hard_bin_six_level_INCUMBENT_REDUCTION",
            "smooth_median_over_concepts_INCUMBENT_REDUCTION"}
        assert "argmax_selected_BOTH_ARMS" in arm[
            "smooth_over_hard_bin_ratio_of_median_abs_rho"]
    assert doc["identity"]["firewall_crossing"] == "DELIBERATE"
    assert doc["prediction_scorecard"]["P6"]["outcome"] in {"HIT", "MISS"}
    assert doc["prediction_scorecard"]["P7"]["outcome"] in {"HIT", "MISS"}


def test_main_refuses_rf_counts_not_built_on_decoder_cosine(tmp_path,
                                                            monkeypatch):
    g, gs, ms, fr, rf19, _ = _full_fixture(tmp_path)
    bad = _write_rf(tmp_path / "bad.h5", np.zeros((2, 40), int), 19,
                    sim="encoder_cosine")
    monkeypatch.setattr(sys, "argv", [
        "c1a", "--gtilde", str(g), "--gtilde-smooth", str(gs),
        "--mu-smooth", str(ms), "--null-reductions", str(fr),
        "--rf-counts", str(bad), "--out", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit, match="decoder_cosine"):
        C.main()


def test_main_refuses_when_the_normaliser_identities_disagree(tmp_path,
                                                              monkeypatch):
    g, gs, ms, fr, rf19, _ = _full_fixture(tmp_path)
    with h5py.File(gs, "a") as hf:
        hf.attrs["library_sha"] = "a-different-store"
    monkeypatch.setattr(sys, "argv", [
        "c1a", "--gtilde", str(g), "--gtilde-smooth", str(gs),
        "--mu-smooth", str(ms), "--null-reductions", str(fr),
        "--rf-counts", str(rf19), "--out", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit, match="PRE-FLIGHT FAILED"):
        C.main()


def test_main_reports_P6_NOT_SCORED_when_the_n19_arm_is_absent(tmp_path,
                                                               monkeypatch):
    """The incumbent 0.00709 was measured at n_compare = 19. Without that arm
    the comparison cannot be made, and it must be reported unscorable rather
    than scored against a different N."""
    g, gs, ms, fr, _, rf39 = _full_fixture(tmp_path)
    out = tmp_path / "o.json"
    monkeypatch.setattr(sys, "argv", [
        "c1a", "--gtilde", str(g), "--gtilde-smooth", str(gs),
        "--mu-smooth", str(ms), "--null-reductions", str(fr),
        "--rf-counts", str(rf39), "--out", str(out)])
    C.main()
    doc = json.loads(out.read_text())
    assert doc["prediction_scorecard"]["P6"]["outcome"].startswith("NOT_SCORED")


# ---------------------------------------------------------------------------
# AMENDMENT AS -- SEED ALIGNMENT. This job WAS the AR9 defect: `gtilde`,
# `mu_hat_full` and `fire_rate` are lexicographic, `rf_counts` is numeric, and
# `c1(mu[s], r_f[s])` correlated 18 of 20 seeds' normalisers against a
# different seed's `r_f`. Cross-seed pairings are approximately null (gotcha
# 18), so the incumbent median |rho| was diluted toward zero -- and AL8b
# demoted the feature-side arm citing exactly that number.
#
# The fixture builds `r_f` as a monotone function of the SAME seed's firing
# rate, so a correct join gives a large |rho| and a mis-join collapses it.
# Without that construction every test below would pass either way.
# ---------------------------------------------------------------------------

def test_the_alignment_fixture_actually_exhibits_a_permuted_axis(tmp_path):
    """PRECONDITION, asserted before any alignment result is believed."""
    g, gs, ms, fr, rf19, rf39 = _full_fixture(tmp_path)
    from src.analysis.seed_axis import stored_seed_labels
    assert stored_seed_labels(fr) == [str(x) for x in PERM]
    assert stored_seed_labels(rf39) == [str(x) for x in range(len(PERM))]
    assert stored_seed_labels(fr) != stored_seed_labels(rf39)


def test_c1_is_computed_on_ALIGNED_seeds(tmp_path, monkeypatch):
    """THE DISCRIMINATION TEST. `r_f` is monotone in its own seed's firing
    rate, and hard-bin `mu` is a step function of the firing bin, so a correct
    join gives a large |rho|. A positional join pairs unrelated seeds and
    cannot."""
    g, gs, ms, fr, rf19, rf39 = _full_fixture(tmp_path)
    out = tmp_path / "o.json"
    monkeypatch.setattr(sys, "argv", [
        "c1a", "--gtilde", str(g), "--gtilde-smooth", str(gs),
        "--mu-smooth", str(ms), "--null-reductions", str(fr),
        "--rf-counts", str(rf19), str(rf39), "--out", str(out)])
    C.main()
    doc = json.loads(out.read_text())
    arm = doc["by_rf_arm"]["n_compare_39"]
    sa = arm["seed_alignment"]
    assert sa["gtilde_order"] == [str(x) for x in PERM]
    assert sa["anchor_order"] == [str(x) for x in range(len(PERM))]
    assert sa["n_rows_mispaired_by_a_positional_join"] > 0
    # measured on this fixture: 0.94 aligned against 0.14 positional
    got = arm["AO7_2b_C1"]["hard_bin_six_level_INCUMBENT_REDUCTION"]
    assert got["median_abs_rho"] > 0.8, got


def test_a_positional_join_gives_a_materially_smaller_rho(tmp_path):
    """The fix is only load-bearing if the wrong answer differs. Same arrays,
    the two joins, computed directly."""
    import h5py
    from src.analysis.c1_argmax import c1
    from src.analysis.m4_confound import hard_bin_mu_per_feature
    from src.analysis.seed_axis import align_to_reference

    g, gs, ms, fr, rf19, rf39 = _full_fixture(tmp_path)
    with h5py.File(fr, "r") as hf:
        fire = hf["fire_rate"][:]                     # in PERM order
    with h5py.File(rf39, "r") as hf:
        r_f = hf["match_count"][:] / 39.0             # in NUMERIC order
    mu, _ = hard_bin_mu_per_feature(g, fire)          # built in PERM order
    aligned = align_to_reference([str(x) for x in PERM], mu,
                                 [str(x) for x in range(len(PERM))])
    good = c1(aligned, r_f)["median_abs_rho"]
    bad = c1(mu, r_f)["median_abs_rho"]               # the pre-AS join
    # measured: 0.9406 against 0.1379, a factor of 6.8
    assert good > 0.8, good
    assert bad < 0.25 * good, (good, bad)


def test_refuses_an_unlabelled_alignment_side(tmp_path, monkeypatch):
    """An unlabelled gtilde with no `gtilde_null` to resolve against cannot be
    ordered at all, and the job must refuse rather than default to position --
    that default IS the defect."""
    g, gs, ms, fr, rf19, rf39 = _full_fixture(
        tmp_path, label_alignment_side=False)
    monkeypatch.setattr(sys, "argv", [
        "c1a", "--gtilde", str(g), "--gtilde-smooth", str(gs),
        "--mu-smooth", str(ms), "--null-reductions", str(fr),
        "--rf-counts", str(rf39), "--out", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit) as e:
        C.main()
    assert "carries no seed labels" in str(e.value)


def test_refuses_when_stored_labels_contradict_the_data(tmp_path):
    """`seed_order`'s cross-check: once labels are on disk a consumer would
    normally stop looking, so a wrong label would be undetectable. It is not."""
    import h5py
    from src.analysis.seed_axis import seed_order
    n_seeds, n_shifts, n_feat = 3, 5, 40
    rng = np.random.default_rng(3)
    ng = rng.lognormal(size=(n_seeds, n_shifts, n_feat))
    nr = tmp_path / "nr.h5"
    with h5py.File(nr, "w") as hf:
        hf.create_dataset("null_g_b", data=ng)
        hf.create_dataset("seeds", data=np.array(
            [f"seed_{x}" for x in PERM], dtype=h5py.string_dtype()))
    gt = tmp_path / "gt.h5"
    with h5py.File(gt, "w") as hf:
        hf.create_dataset("gtilde_null", data=2.0 * ng)
        hf.create_dataset("seeds", data=np.array(
            ["9", "8", "7"], dtype=h5py.string_dtype()))   # a LIE
    with pytest.raises(SystemExit) as e:
        seed_order(gt, nr)
    assert "disagree with the order recovered" in str(e.value)


def test_stored_labels_that_agree_are_confirmed_not_merely_accepted(tmp_path):
    import h5py
    from src.analysis.seed_axis import seed_order
    n_seeds, n_shifts, n_feat = 3, 5, 40
    rng = np.random.default_rng(4)
    ng = rng.lognormal(size=(n_seeds, n_shifts, n_feat))
    nr = tmp_path / "nr.h5"
    with h5py.File(nr, "w") as hf:
        hf.create_dataset("null_g_b", data=ng)
        hf.create_dataset("seeds", data=np.array(
            [f"seed_{x}" for x in PERM], dtype=h5py.string_dtype()))
    gt = tmp_path / "gt.h5"
    with h5py.File(gt, "w") as hf:
        hf.create_dataset("gtilde_null", data=2.0 * ng)
        hf.create_dataset("seeds", data=np.array(
            [str(x) for x in PERM], dtype=h5py.string_dtype()))
    got = seed_order(gt, nr)
    assert got["source"] == "STORED_LABELS_CONFIRMED_BY_RESOLVER"
    assert got["cross_check"] == "AGREE"
    assert got["labels"] == [str(x) for x in PERM]
