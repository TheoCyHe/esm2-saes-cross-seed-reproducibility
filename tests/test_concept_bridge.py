"""Tests for `concept_bridge` -- PREREGISTRATION amendment AY sections 2a/2c/3.

Every fixture is built so it CAN exhibit the condition the assertion is about
(gotcha 16): the saturation fixtures cover all three statuses, the null-
construction fixtures cover a shared schedule AND a per-seed one, and the
margin fixture contains a zero threshold so the NaN branch is reachable.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.analysis import concept_bridge as cb


# ------------------------------------------------ null construction (§2a i)

def _null_file(tmp_path: Path, dest_shape, n_seeds=2, n_shifts=3,
               n_concepts=4) -> Path:
    p = tmp_path / "null.h5"
    with h5py.File(p, "w") as hf:
        hf.create_dataset("null_cmax_b",
                          data=np.zeros((n_seeds, n_shifts, n_concepts)))
        if dest_shape is not None:
            hf.create_dataset("dest_schedule", data=np.zeros(dest_shape,
                                                            dtype=np.int64))
    return p


def test_shared_schedule_is_detected_from_the_ARRAY_RANK(tmp_path):
    """A 2-D schedule has no seed axis, so one schedule serves all seeds."""
    p = _null_file(tmp_path, (3, 11))          # (n_shifts, n_proteins)
    with h5py.File(p, "r") as hf:
        r = cb.null_construction_report(hf)
    assert r["verdict"] == "PER_SEED_DRAWS_BUT_ONE_SHARED_SCHEDULE"
    assert r["dest_schedule"]["shared_across_seeds"] is True
    assert r["dest_schedule"]["has_seed_axis"] is False
    assert r["per_seed_draws_exist"] is True
    assert r["draws_per_concept_per_seed"] == 3
    assert r["draws_per_concept_pooled"] == 6


def test_per_seed_schedule_is_DISTINGUISHED_from_a_shared_one(tmp_path):
    """The discriminating fixture. Without this the report could return the
    shared verdict unconditionally and every test above would still pass."""
    p = _null_file(tmp_path, (2, 3, 11))       # (n_seeds, n_shifts, n_proteins)
    with h5py.File(p, "r") as hf:
        r = cb.null_construction_report(hf)
    assert r["verdict"] == "PER_SEED_DRAWS_AND_PER_SEED_SCHEDULE"
    assert r["dest_schedule"]["has_seed_axis"] is True


def test_absent_schedule_is_reported_absent_not_as_shared(tmp_path):
    p = _null_file(tmp_path, None)
    with h5py.File(p, "r") as hf:
        r = cb.null_construction_report(hf)
    assert r["verdict"] == "NO_DEST_SCHEDULE"
    assert r["dest_schedule"]["present"] is False


# --------------------------------------------------------- detection margin

def test_detection_margin_divides_and_refuses_a_nonpositive_threshold():
    obs = np.array([[2.0, 4.0], [1.0, 8.0]])
    thr = np.array([1.0, 0.0])                 # second concept has no threshold
    m = cb.detection_margin(obs, thr)
    assert m[0, 0] == pytest.approx(2.0)
    assert m[1, 0] == pytest.approx(1.0)
    # A zero threshold must be NaN, not +inf: +inf reads as "cleared by an
    # enormous margin" when in fact the ratio is undefined.
    assert np.isnan(m[:, 1]).all()


def test_margin_summary_takes_the_WEAKEST_seed():
    obs = np.array([[10.0], [1.5]])
    thr = np.array([1.0])
    m = cb.detection_margin(obs, thr)
    s = cb.margin_summary(m, np.array([1.0]), np.array([True]))
    assert s["status"] == "COMPUTED"
    # 1.5, not 10 and not the mean.
    assert s["min_over_seeds_of_gain_over_threshold"]["p50"] == pytest.approx(1.5)
    assert s["share_worst_seed_within_2x_of_threshold"] == pytest.approx(1.0)


def test_margin_summary_names_an_empty_group_rather_than_returning_zero():
    m = np.ones((2, 1))
    s = cb.margin_summary(m, np.array([1.0]), np.array([False]))
    assert s["status"] == "EMPTY_GROUP"
    assert s["n_concepts"] == 0


# ------------------------------------------------------- concept name parts

@pytest.mark.parametrize("concept,key,desc", [
    ("TRANSMEM", "TRANSMEM", ""),
    ("BINDING:heme a3", "BINDING", "heme a3"),
    # A descriptor containing its own colon: the split is on the FIRST one.
    ("REGION:Interaction with host: DAXX", "REGION",
     "Interaction with host: DAXX"),
])
def test_feature_key_and_descriptor_split_on_the_first_colon(concept, key, desc):
    assert cb.concept_feature_key(concept) == key
    assert cb.concept_descriptor(concept) == desc


# ------------------------------------------------------ saturation (§3)

def test_saturation_returns_the_smallest_size_above_the_largest_unsaturated():
    d = np.array([0.0, 0.5, 1.0, 1.0])
    size = np.array([10.0, 100.0, 150.0, 900.0])
    r = cb.saturation_threshold(d, size)
    assert r["status"] == "SATURATES"
    assert r["largest_unsaturated_size"] == pytest.approx(100.0)
    assert r["threshold"] == pytest.approx(150.0)
    assert r["n_concepts_at_or_above"] == 2


def test_saturation_names_the_everywhere_case():
    r = cb.saturation_threshold(np.ones(3), np.array([5.0, 7.0, 9.0]))
    assert r["status"] == "SATURATED_EVERYWHERE"
    assert r["threshold"] == pytest.approx(5.0)


def test_saturation_refuses_to_invent_a_threshold_when_it_never_saturates():
    """The largest concept is unsaturated, so nothing lies above it. Returning
    a number here would be a threshold computed from an empty set."""
    d = np.array([1.0, 1.0, 0.5])
    size = np.array([10.0, 20.0, 900.0])
    r = cb.saturation_threshold(d, size)
    assert r["status"] == "NEVER_SATURATES"
    assert r["threshold"] is None
    assert r["n_concepts_at_or_above"] == 0


# --------------------------------------------------------- binned curve (§3)

def test_binned_curve_uses_between_concept_dispersion_not_wilson():
    # One bin, four concepts, D = 0, 0, 1, 1 -> mean 0.5, sd(ddof=1) = 0.5774
    d = np.array([0.0, 0.0, 1.0, 1.0])
    size = np.array([1.0, 2.0, 3.0, 4.0])
    r = cb.binned_curve(d, size, n_bins=1)
    b = r["bins"][0]
    assert b["n_concepts"] == 4
    assert b["mean_D"] == pytest.approx(0.5)
    assert b["sd_D_between_concepts"] == pytest.approx(0.5773502691896257)
    assert b["se_between_concepts"] == pytest.approx(0.28867513459481287)
    # The naive binomial SE is sqrt(.25/4) = 0.25 -- SMALLER, which is the
    # understatement the clause is about.
    assert b["naive_binomial_se_for_comparison"] == pytest.approx(0.25)
    assert b["understatement_ratio_se_over_naive"] > 1.0


def test_binned_curve_reports_an_empty_bin_rather_than_crashing():
    d = np.zeros(4)
    size = np.array([1.0, 1.0, 1.0, 1.0])      # every value identical
    r = cb.binned_curve(d, size, n_bins=4)
    assert any(b.get("status") == "EMPTY" for b in r["bins"])


# ------------------------------------------------ prevalence conditioning

def test_conditioned_distribution_splits_at_reported_edges():
    d = np.array([0.0, 0.0, 0.5, 0.5, 1.0, 1.0])
    size = np.arange(1.0, 7.0)
    r = cb.conditioned_distribution(d, size, n_strata=3)
    assert r["n_strata"] == 3
    assert len(r["interior_stratum_edges"]) == 2
    assert [s["n_concepts"] for s in r["strata"]] == [2, 2, 2]
    # The lowest stratum is entirely D = 0, so no interior mass there.
    assert r["strata"][0]["distribution_of_D"]["interior_mass"] == 0.0
    # The middle is entirely D = 0.5, so all of it is interior.
    assert r["strata"][1]["distribution_of_D"]["interior_mass"] == 1.0
    assert r["pooled_distribution_of_D"]["n_concepts"] == 6


# -------------------------------------------------- experimental tier union

def test_experimental_counts_UNION_the_two_tiers(tmp_path):
    """A residue in both tiers must be counted once. Summing the dataset
    lengths would give 3; the union is 2."""
    p = tmp_path / "labels.h5"
    with h5py.File(p, "w") as hf:
        g = hf.create_group("concepts/X")
        g.create_dataset("experimental", data=np.array([5, 9]))
        g.create_dataset("experimental_highthroughput", data=np.array([9]))
        hf.create_group("concepts/Y")          # present, no experimental tier
    got = cb.experimental_counts_for(p, ["X", "Y", "ABSENT"])
    assert got["X"] == 2
    assert got["Y"] == 0
    assert got["ABSENT"] == 0


# ------------------------------------------------------------- entry point

_IDENT = {"n_rows": 100, "library_sha": "abc", "truncation_policy": "untruncated",
          "scored_row_set": "label_store", "argmax_pool_n": 3, "n_concepts": 3}


def _tiny_inputs(tmp_path: Path):
    """A complete, minimal AP + null + label triple whose answers are known.

    A unit suite that never executes `main()` cannot catch an entry-point bug
    (gotcha 13 extension 4's corollary), so `run` is exercised end to end.
    """
    n_seed, n_feat, n_con = 2, 3, 3
    concepts = ["A:x", "B", "C:y"]
    ap_path = tmp_path / "ap.h5"
    with h5py.File(ap_path, "w") as hf:
        hf.attrs.update(_IDENT)
        hf.create_dataset("seeds", data=np.array(["seed_0", "seed_1"],
                                                 dtype=h5py.string_dtype()))
        fc = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int64)
        hf.create_dataset("fire_counts", data=fc)
        g = hf.create_group("pooled")
        g.create_dataset("concepts", data=np.array(concepts,
                                                   dtype=h5py.string_dtype()))
        g.create_dataset("in_argmax_pool", data=np.ones(n_con, dtype=bool))
        ap = np.zeros((n_seed, n_feat, n_con), dtype=np.float32)
        ap[:, 0, 0] = 0.9          # A:x detected in both
        ap[0, 1, 1] = 0.9          # B detected in seed 0 only
        ap[:, 2, 2] = 0.01         # C:y never detected
        g.create_dataset("ap", data=ap)
        g.create_dataset("prevalence", data=np.zeros(n_con))
        g.create_dataset("n_positives", data=np.array([500, 50, 5]))

    null_path = tmp_path / "null.h5"
    with h5py.File(null_path, "w") as hf:
        hf.attrs.update(_IDENT)
        hf.create_dataset("concepts", data=np.array(concepts,
                                                    dtype=h5py.string_dtype()))
        hf.create_dataset("seeds", data=np.array(["seed_0", "seed_1"],
                                                 dtype=h5py.string_dtype()))
        hf.create_dataset("null_cmax_b",
                          data=np.full((n_seed, 20, n_con), 0.1))
        hf.create_dataset("dest_schedule", data=np.zeros((20, 7),
                                                         dtype=np.int64))
        hf.create_dataset("fire_rate", data=fc.astype(np.float64) / 100.0)

    labels_path = tmp_path / "labels.h5"
    with h5py.File(labels_path, "w") as hf:
        for c in concepts:
            hf.create_group(f"concepts/{c}")
    return ap_path, null_path, labels_path


def test_run_end_to_end_produces_the_known_answer(tmp_path):
    ap, null, labels = _tiny_inputs(tmp_path)
    out_path = tmp_path / "out.json"
    out = cb.run(ap, null, labels, out_path)

    assert out_path.exists()
    assert json.loads(out_path.read_text())["reference_percentile"] == 95.0

    c = out["section_2a_i_null_construction"]
    assert c["verdict"] == "PER_SEED_DRAWS_BUT_ONE_SHARED_SCHEDULE"

    nd = out["section_2a_ii_never_detected"]
    # Only C:y fails in both seeds; B is detected in seed 0 so D = 0.5.
    assert nd["n_never_detected"] == 1
    assert nd["concepts"][0]["concept"] == "C:y"
    assert nd["concepts"][0]["feature_key"] == "C"
    assert nd["concepts"][0]["descriptor"] == "y"
    assert nd["n_with_experimental_evidence"] == 0

    ref = out["section_2c_and_3_curve"]["95.0"]["pooled"]
    assert ref["distribution_of_D"]["n_exactly_one"] == 1     # A:x
    assert ref["distribution_of_D"]["n_exactly_zero"] == 1    # C:y


def test_run_REFUSES_a_seed_axis_mismatch(tmp_path):
    """The positional guard must be reachable from `run`, not only from its
    own unit test -- a guard that the driver never calls is not a guard."""
    ap, null, labels = _tiny_inputs(tmp_path)
    with h5py.File(null, "r+") as hf:
        del hf["seeds"]
        hf.create_dataset("seeds", data=np.array(["seed_1", "seed_0"],
                                                 dtype=h5py.string_dtype()))
    with pytest.raises(SystemExit) as e:
        cb.run(ap, null, labels, tmp_path / "o.json")
    assert "seed labels" in str(e.value)


@pytest.mark.parametrize("key", ["library_sha", "argmax_pool_n", "scored_row_set"])
def test_run_REFUSES_an_identity_mismatch(tmp_path, key):
    """CT8 item 1: `run` compares the identities it records (gotcha 15)."""
    ap, null, labels = _tiny_inputs(tmp_path)
    with h5py.File(null, "r+") as hf:
        hf.attrs[key] = "something_else" if key != "argmax_pool_n" else 2
    with pytest.raises(SystemExit) as e:
        cb.run(ap, null, labels, tmp_path / "o.json")
    assert key in str(e.value)


def test_run_REFUSES_an_identity_key_absent_on_one_side(tmp_path):
    ap, null, labels = _tiny_inputs(tmp_path)
    with h5py.File(null, "r+") as hf:
        del hf.attrs["library_sha"]
    with pytest.raises(SystemExit) as e:
        cb.run(ap, null, labels, tmp_path / "o.json")
    assert "library_sha" in str(e.value) and "ABSENT" in str(e.value)


def test_run_RECORDS_the_identity_comparison(tmp_path):
    ap, null, labels = _tiny_inputs(tmp_path)
    out = cb.run(ap, null, labels, tmp_path / "o.json")
    cmp = out["source"]["identity_comparison"]
    assert all(v["agree"] for v in cmp.values())
    assert set(cmp) == set(cb.SHARED_IDENTITY_KEYS)
