"""Tests for amendment AV §4's two counts.

Every fixture is built so the quantity under test has a KNOWN answer that
differs from the answer a plausible bug would give. Gotcha 16: a fixture that
cannot exhibit the disputed condition cannot test for it, and seven of nineteen
mutations survived the last time this kind of logic lived inside `main()`.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from src.analysis.av_counts import (
    experimental_residue_counts,
    gain_floor_share,
    normalised_gain,
    score_p1,
    score_p2,
    summarise_coverage,
)


def _labels(path, spec, n_rows=1000):
    """spec: {concept: {tier: n_rows_in_that_tier}}; `all` is the union."""
    with h5py.File(path, "w") as hf:
        hf.attrs["n_rows"] = n_rows
        g = hf.create_group("concepts")
        for c, tiers in spec.items():
            cg = g.create_group(c)
            off, allrows = 0, []
            for t, n in tiers.items():
                rows = np.arange(off, off + n, dtype=np.int64)
                cg.create_dataset(t, data=rows)
                allrows.append(rows)
                off += n
            cg.create_dataset("all", data=np.concatenate(allrows)
                              if allrows else np.zeros(0, dtype=np.int64))
    return path


def _ap(path, concepts, ap_block, prevalence, pool=None, seeds=("seed_0",)):
    with h5py.File(path, "w") as hf:
        hf.create_dataset("seeds", data=np.array(seeds, dtype=h5py.string_dtype()))
        g = hf.create_group("pooled")
        g.create_dataset("ap", data=np.asarray(ap_block, dtype=np.float32))
        g.create_dataset("concepts", data=np.array(concepts,
                                                   dtype=h5py.string_dtype()))
        g.create_dataset("prevalence", data=np.asarray(prevalence, dtype=np.float64))
        if pool is not None:
            g.create_dataset("in_argmax_pool", data=np.asarray(pool, dtype=bool))
    return path


# --------------------------------------------------------------------------
# §4a
# --------------------------------------------------------------------------

def test_experimental_tiers_are_unioned_not_summed(tmp_path):
    """A concept carrying BOTH experimental spellings must not be
    double-counted; the fixture's two tiers are disjoint by construction, so
    a union and a sum give the same number ONLY if the code unions correctly on
    overlapping input too -- which the next test covers."""
    p = _labels(tmp_path / "l.h5", {
        "A": {"experimental": 3, "experimental_highthroughput": 2,
              "by_similarity": 10},
        "B": {"by_similarity": 40},
    })
    r = experimental_residue_counts(p)
    assert r["experimental"] == {"A": 5, "B": 0}
    assert r["pooled"] == {"A": 15, "B": 40}


def test_overlapping_experimental_rows_are_deduplicated(tmp_path):
    p = tmp_path / "l.h5"
    with h5py.File(p, "w") as hf:
        hf.attrs["n_rows"] = 100
        cg = hf.create_group("concepts").create_group("A")
        cg.create_dataset("experimental", data=np.array([1, 2, 3]))
        cg.create_dataset("experimental_highthroughput", data=np.array([3, 4]))
        cg.create_dataset("all", data=np.array([1, 2, 3, 4]))
    # 3 + 2 = 5 if summed; 4 if unioned. The fixture DISCRIMINATES.
    assert experimental_residue_counts(p)["experimental"]["A"] == 4


def test_coverage_counts_only_concepts_in_the_named_set(tmp_path):
    exp = {"A": 5, "B": 0, "C": 100, "PARENT": 7}
    pooled = {"A": 15, "B": 40, "C": 200, "PARENT": 70}
    pool = summarise_coverage(exp, pooled, ["A", "B", "C"], "argmax_pool")
    assert pool["n_concepts"] == 3
    assert pool["n_with_at_least_one_experimental_residue"] == 2
    assert pool["n_with_none"] == 1
    # PARENT is outside the named set and must not be counted.
    lib = summarise_coverage(exp, pooled, list(exp), "library")
    assert lib["n_with_at_least_one_experimental_residue"] == 3


def test_coverage_reports_f_c_the_quantity_that_breaks_all_rows():
    r = summarise_coverage({"A": 6, "B": 66}, {"A": 1000, "B": 100},
                           ["A", "B"], "s")
    assert r["experimental_fraction_f_c"]["min_over_nonzero"] == pytest.approx(0.006)
    assert r["experimental_fraction_f_c"]["max_over_nonzero"] == pytest.approx(0.66)


def test_threshold_ladder_is_monotone_and_correct():
    r = summarise_coverage({"a": 0, "b": 1, "c": 7, "d": 700},
                           {"a": 1, "b": 1, "c": 7, "d": 700},
                           ["a", "b", "c", "d"], "s")
    t = r["counts_at_thresholds"]
    assert [t[">=1"], t[">=5"], t[">=10"], t[">=100"]] == [3, 2, 1, 1]


# --------------------------------------------------------------------------
# §4b
# --------------------------------------------------------------------------

def test_normalised_gain_matches_the_pre_registered_formula():
    ap = np.array([[0.5, 0.2]])
    prev = np.array([0.1, 0.2])
    g = normalised_gain(ap, prev)
    assert g[0, 0] == pytest.approx((0.5 - 0.1) / 0.9)
    assert g[0, 1] == pytest.approx(0.0)          # AP exactly at chance -> G = 0


def test_gain_uses_the_PER_CONCEPT_prevalence_not_a_global_one():
    """C4's invariant. The two concepts have the same AP and very different
    prevalences, so a single reused `p` would give them the same `G`."""
    ap = np.array([[0.30, 0.30]])
    g = normalised_gain(ap, np.array([0.01, 0.25]))
    assert g[0, 0] != pytest.approx(g[0, 1])
    assert g[0, 0] == pytest.approx(0.29 / 0.99)
    assert g[0, 1] == pytest.approx(0.05 / 0.75)


def test_mismatched_prevalence_length_raises_rather_than_broadcasting():
    with pytest.raises(ValueError, match="concept axis mismatch"):
        normalised_gain(np.zeros((2, 3)), np.zeros(2))


def test_gain_floor_share_counts_the_right_pairs(tmp_path):
    # 1 seed, 2 features, 3 concepts. prevalence 0 everywhere so G == AP,
    # which makes the expected answer readable off the fixture.
    ap = [[[0.9, 0.01, 0.06],
           [0.00, 0.00, 0.04]]]
    p = _ap(tmp_path / "ap.h5", ["A", "B", "C"], ap, [0.0, 0.0, 0.0],
            pool=[True, True, True])
    r = gain_floor_share(p, "pooled", floor=0.05, pool_only=True)
    assert r["n_pairs_total"] == 6
    assert r["n_pairs_clearing"] == 2               # 0.9 and 0.06
    assert r["share_clearing_pooled_over_seeds"] == pytest.approx(2 / 6)
    assert r["per_seed"][0]["n_features_clearing_at_least_one_concept"] == 1


def test_pool_mask_is_honoured_and_changes_the_answer(tmp_path):
    """The pool/library distinction must be real, not decorative: the fixture's
    excluded concept is the ONLY one clearing the floor."""
    ap = [[[0.9, 0.0],
           [0.0, 0.0]]]
    p = _ap(tmp_path / "ap.h5", ["PARENT", "B"], ap, [0.0, 0.0],
            pool=[False, True])
    pool = gain_floor_share(p, "pooled", pool_only=True)
    lib = gain_floor_share(p, "pooled", pool_only=False)
    assert pool["n_pairs_clearing"] == 0
    assert lib["n_pairs_clearing"] == 1
    assert pool["n_concepts_scored"] == 1 and lib["n_concepts_scored"] == 2


def test_per_seed_shares_are_reported_separately(tmp_path):
    ap = [[[0.9, 0.9]], [[0.0, 0.0]]]        # seed 0 clears both, seed 1 none
    p = _ap(tmp_path / "ap.h5", ["A", "B"], ap, [0.0, 0.0],
            pool=[True, True], seeds=("seed_0", "seed_1"))
    r = gain_floor_share(p, "pooled", pool_only=True)
    assert [s["share_clearing"] for s in r["per_seed"]] == [1.0, 0.0]
    assert r["share_clearing_per_seed"]["median"] == pytest.approx(0.5)
    # The pooled figure hides exactly this, which is why both are reported.
    assert r["share_clearing_pooled_over_seeds"] == pytest.approx(0.5)


def test_below_chance_gains_are_retained_not_clipped(tmp_path):
    p = _ap(tmp_path / "ap.h5", ["A"], [[[0.0]]], [0.5], pool=[True])
    r = gain_floor_share(p, "pooled", pool_only=True)
    assert r["per_seed"][0]["median_G"] == pytest.approx(-1.0)


# --------------------------------------------------------------------------
# the verdicts are DERIVED, never asserted (gotcha 20)
# --------------------------------------------------------------------------

def test_p1_verdict_is_derived_on_both_sides_of_its_threshold():
    assert score_p1(351)["outcome"] == "HIT"
    assert score_p1(352)["outcome"] == "MISS"       # the boundary is "352 and above"
    assert score_p1(700)["outcome"] == "MISS"


def test_p2_verdict_is_derived_on_both_sides_of_its_threshold():
    assert score_p2(0.049)["outcome"] == "HIT"
    assert score_p2(0.05)["outcome"] == "MISS"
    assert score_p2(0.5)["outcome"] == "MISS"


# --------------------------------------------------------------------------
# main() -- the entry point gets its own test (gotcha 13 extension 4:
# "a unit suite that never executes the entry point cannot catch an
# entry-point bug"; 244 tests once passed while main() was broken)
# --------------------------------------------------------------------------

def test_main_runs_end_to_end_and_writes_both_counts(tmp_path):
    from src.analysis.av_counts import main
    library = ["ACT_SITE", "BINDING", "BINDING:Zn(2+)", "DOMAIN:Kinase"]
    labels = _labels(tmp_path / "l.h5", {
        "ACT_SITE": {"experimental": 4, "by_similarity": 6},
        "BINDING": {"by_similarity": 20},                    # no experimental
        "BINDING:Zn(2+)": {"experimental": 2, "predicted": 8},
        "DOMAIN:Kinase": {"curator_inference": 30},          # no experimental
    })
    ap = _ap(tmp_path / "ap.h5", library,
             [[[0.9, 0.9, 0.9, 0.9], [0.0, 0.0, 0.0, 0.0]]],
             [0.0, 0.0, 0.0, 0.0],
             # BINDING is a pooled PARENT -- excluded from the argmax pool
             pool=[True, False, True, True])
    out = tmp_path / "r.json"
    assert main(["--labels", str(labels), "--ap", str(ap),
                 "--stratum", "pooled", "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    pool = rep["s4a_experimental_coverage"]["pool"]
    lib = rep["s4a_experimental_coverage"]["library"]
    assert pool["n_concepts"] == 3 and lib["n_concepts"] == 4
    assert pool["n_with_at_least_one_experimental_residue"] == 2
    # the pool and library answers must DIFFER, or the fixture proves nothing
    assert pool["n_concepts"] != lib["n_concepts"]
    assert rep["s4b_gain_floor"]["pool"]["n_pairs_total"] == 6      # 1x2x3
    assert rep["s4b_gain_floor"]["library"]["n_pairs_total"] == 8   # 1x2x4
    assert rep["AV_P1"]["outcome"] == "HIT"
    assert rep["AV_P2"]["observed"] == pytest.approx(0.5)
    assert rep["AV_P2"]["outcome"] == "MISS"


def test_main_refuses_a_store_and_an_AP_naming_different_libraries(tmp_path):
    """gotcha 15 one level up: two artifacts that both open is not two
    artifacts describing the same experiment."""
    from src.analysis.av_counts import main
    labels = _labels(tmp_path / "l.h5", {"ACT_SITE": {"experimental": 4}})
    ap = _ap(tmp_path / "ap.h5", ["SOMETHING_ELSE"], [[[0.5]]], [0.0],
             pool=[True])
    with pytest.raises(SystemExit, match="disagree on the library"):
        main(["--labels", str(labels), "--ap", str(ap), "--out",
              str(tmp_path / "r.json")])
