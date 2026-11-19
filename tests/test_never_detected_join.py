"""Tests for the never-detected descriptive join (amendment BY Part D).

EXPECTATIONS ARE WRITTEN DOWN (gotcha 32b). The fixture below has four concepts
whose every joined value is stated in this docstring, and the assertions check
those stated values rather than recomputing them from the module.

THE FIXTURE:
  A  D=1.0  n_res=1000  prev=0.10  exp_pos=50   in_pool=True    floorA sum=0
  B  D=0.0  n_res=200   prev=0.02  exp_pos=7    in_pool=True    floorA sum=40
  C  D=0.0  n_res=150   prev=0.01  exp_pos=0    in_pool=False   floorA sum=60
  D  D=0.0  n_res=120   prev=0.005 (absent from the experimental axis) floorA=20

So: 3 never-detected; 1 of them carries experimental support (B); 2 are in the
experimental argmax pool (A is detected, so among the never-detected only B);
1 is on the axis with zero positives (C); 1 is absent from the axis (D).
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from src.analysis.never_detected_join import (
    NEVER_DETECTED_CRITERION,
    build,
    summarise_column,
    write_artifact,
)

CONCEPTS = ["A", "B", "C", "D"]
D_VALS = [1.0, 0.0, 0.0, 0.0]
N_RES = [1000, 200, 150, 120]
PREV = [0.10, 0.02, 0.01, 0.005]
EXP_AXIS = ["A", "B", "C"]          # D is absent from the experimental axis
EXP_POS = [50, 7, 0]
EXP_POOL = [True, True, False]
FLOOR_A = [0, 2, 3, 1]              # per seed; 20 seeds -> totals 0, 40, 60, 20
FLOOR_B = [1, 1, 1, 1]


@pytest.fixture
def artifacts(tmp_path):
    repro = {
        "reference_percentile": 95.0,
        "identity": {"library_sha": "deadbeef"},
        "per_concept": [
            {"concept": c, "D": d, "mean_A": 0.5, "sd_A": 0.1,
             "mean_A_gain": 0.4, "null_threshold_p95": 0.001,
             "n_annotated_residues": n, "prevalence": p}
            for c, d, n, p in zip(CONCEPTS, D_VALS, N_RES, PREV)
        ],
    }
    rp = tmp_path / "repro.json"
    rp.write_text(json.dumps(repro))

    fp = tmp_path / "floor.h5"
    st = h5py.special_dtype(vlen=str)
    with h5py.File(fp, "w") as hf:
        g = hf.create_group("pooled")
        g.create_dataset("concepts", data=np.array(CONCEPTS, dtype=object), dtype=st)
        g.create_dataset("at_floor_a",
                         data=np.tile(np.array(FLOOR_A), (20, 1)))
        g.create_dataset("at_floor_b",
                         data=np.tile(np.array(FLOOR_B), (20, 1)))
        g.create_dataset("n_positives", data=np.array(N_RES))
        hf.attrs["at_floor_definition"] = "count over features of p == 1/(n+1)"
        hf.attrs["git_sha"] = "cafe1234"

    ap = tmp_path / "ap_exp.h5"
    with h5py.File(ap, "w") as hf:
        g = hf.create_group("experimental")
        g.create_dataset("concepts", data=np.array(EXP_AXIS, dtype=object), dtype=st)
        g.create_dataset("n_positives", data=np.array(EXP_POS))
        g.create_dataset("in_argmax_pool", data=np.array(EXP_POOL))
    return rp, fp, ap


# ------------------------------------------------------------------ counts

def test_counts_by_hand(artifacts):
    res = build(*artifacts)
    c = res["counts"]
    assert c["n_concepts_total"] == 4
    assert c["n_never_detected"] == 3                     # B, C, D
    assert c["n_detected"] == 1                           # A
    assert c["n_never_detected_carrying_experimental_support"] == 1     # B
    assert c["n_never_detected_in_experimental_argmax_pool"] == 1       # B
    assert c["n_never_detected_on_experimental_axis_with_zero_positives"] == 1  # C
    assert c["n_never_detected_absent_from_experimental_axis"] == 1     # D


def test_experimental_support_and_pool_membership_are_DIFFERENT_columns(artifacts):
    """A concept can be on the axis, carry zero positives, and be out of the
    pool. Collapsing the two columns would report C as 'no experimental data'
    when what is true is 'no experimental POSITIVES' -- different claims."""
    res = build(*artifacts)
    by = {r["concept"]: r for r in res["_arrays"]["rows"]}
    assert by["C"]["on_experimental_axis"] is True
    assert by["C"]["carries_experimental_support"] is False
    assert by["C"]["in_experimental_argmax_pool"] is False
    assert by["B"]["carries_experimental_support"] is True
    assert by["B"]["in_experimental_argmax_pool"] is True


def test_absent_from_the_axis_is_None_not_zero(artifacts):
    """gotcha 25: absent is reported as absent, never treated as a zero. D is
    not on the experimental axis at all; 0 positives would be a measurement."""
    res = build(*artifacts)
    by = {r["concept"]: r for r in res["_arrays"]["rows"]}
    assert by["D"]["n_experimental_positives"] is None
    assert by["D"]["on_experimental_axis"] is False
    assert by["C"]["n_experimental_positives"] == 0      # measured zero


# ------------------------------------------------------------- the at-floor sum

def test_at_floor_totals_over_seeds_by_hand(artifacts):
    """FLOOR_A per seed is [0,2,3,1] tiled over 20 seeds -> [0,40,60,20]."""
    res = build(*artifacts)
    by = {r["concept"]: r for r in res["_arrays"]["rows"]}
    assert [by[c]["at_floor_a_total_over_seeds"] for c in CONCEPTS] == [0, 40, 60, 20]
    assert [by[c]["at_floor_b_total_over_seeds"] for c in CONCEPTS] == [20, 20, 20, 20]
    assert by["B"]["at_floor_a_mean_per_seed"] == pytest.approx(2.0)


# -------------------------------------------------------- the boundary statement

def test_the_join_declares_it_does_NOT_separate_blindness_from_sparsity(artifacts):
    res = build(*artifacts)
    w = res["what_this_does_NOT_do"]
    assert w["separates_model_blindness_from_annotation_sparsity"] is False
    assert "DOES NOT SEPARATE" in w["statement"]
    assert res["standing"]["reportable_as_a_finding"] is False
    assert res["standing"]["mechanism_claim"].startswith("NONE")
    assert res["standing"]["multiplicity_control"] == "NONE"
    assert res["standing"]["computed_before_declared"] is True


def test_the_boundary_statement_reaches_the_ARTIFACT_not_only_the_json(artifacts,
                                                                      tmp_path):
    """An output that travels without its boundary acquires one it never had, so
    the disclaimer must be in the h5 attrs a downstream consumer reads."""
    res = build(*artifacts)
    out = tmp_path / "join.h5"
    write_artifact(res, out)
    with h5py.File(out, "r") as hf:
        assert hf.attrs["separates_blindness_from_sparsity"] == np.False_ or \
               hf.attrs["separates_blindness_from_sparsity"] is False
        assert "DOES NOT SEPARATE" in hf.attrs["does_NOT_separate"]
        assert hf.attrs["standing"] == "COMPUTED_BEFORE_DECLARED"
        assert hf.attrs["mechanism_claim"] == "NONE"
        assert hf.attrs["multiplicity_control"] == "NONE"
        assert hf.attrs["never_detected_criterion"] == NEVER_DETECTED_CRITERION
        # the sentinel is documented, not left as a magic number
        assert "-1 means" in hf.attrs["n_experimental_positives_sentinel"]
        npos = hf["n_experimental_positives"][:]
        assert npos[3] == -1        # D, absent from the axis
        assert npos[2] == 0         # C, measured zero


# ------------------------------------------------------------- guards

def test_a_different_reference_percentile_is_refused(artifacts, tmp_path):
    """The never-detected set is defined AT the reference, so a headline at a
    different percentile is a different population under the same name."""
    rp, fp, ap = artifacts
    bad = json.loads(rp.read_text())
    bad["reference_percentile"] = 99.0
    p2 = tmp_path / "bad.json"
    p2.write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="reference percentile"):
        build(p2, fp, ap)


def test_misaligned_concept_labels_are_refused(artifacts, tmp_path):
    """gotcha 29: a positional join across two artifacts mis-pairs silently, so
    the labels are compared elementwise and a mismatch is fatal."""
    rp, fp, ap = artifacts
    with h5py.File(fp, "r+") as hf:
        del hf["pooled/concepts"]
        st = h5py.special_dtype(vlen=str)
        hf["pooled"].create_dataset(
            "concepts", data=np.array(["A", "B", "D", "C"], dtype=object), dtype=st)
    with pytest.raises(SystemExit, match="disagree at"):
        build(rp, fp, ap)


# ------------------------------------------------------- the column summariser

def test_summarise_column_carries_its_denominator():
    s = summarise_column(np.array([1.0, 2.0, 3.0, 4.0]), "x")
    assert s["denominator"] == 4
    assert s["median"] == 2.5
    assert s["min"] == 1.0 and s["max"] == 4.0


def test_summarise_column_reports_an_empty_column_rather_than_zeroing_it():
    s = summarise_column(np.array([]), "x")
    assert s["denominator"] == 0
    assert s["all_values_non_finite"] is True
    assert "no summary is defined" in s["note"]


def test_summarise_column_drops_non_finite_but_keeps_the_denominator_honest():
    s = summarise_column(np.array([1.0, np.nan, 3.0]), "x")
    assert s["denominator"] == 2        # the two finite values, not 3
    assert s["median"] == 2.0
