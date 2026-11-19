"""Tests for the canary manifest (PREREGISTRATION amendments AD1/AD2, AM5).

This is a gate, so gotcha 13 applies twice over: it must detect a moved
artifact, and its failure must reach its consumer via the exit code.

Mutation kills RUN:
  * comparing sha256 with `in` instead of `==` -> kills
    test_a_changed_byte_is_detected.
  * dropping the ABSENT branch so a missing target is skipped -> kills
    test_a_baselined_target_that_vanishes_is_a_problem.
  * resetting `first_baselined_at` to HEAD on re-capture -> kills
    test_recapture_carries_first_baselined_at_forward.
  * hashing an M1 document whole -> kills
    test_the_M1_targets_are_not_whole_file_hashed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.analysis.canary import (
    TARGETS,
    TOLERANCE,
    capture,
    compare_manifests,
    describe,
    tolerance_coverage,
    verify,
)

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# the declared target set
# ---------------------------------------------------------------------------

def test_all_three_arms_are_present_as_separate_targets():
    """AK1's distinction must survive into the suite's STRUCTURE, not live only
    in prose. Merging the arms would collapse the per-arm tolerance.

    **The expected set is WRITTEN OUT, not derived from `TARGETS`.** A test
    parametrised over the module's own list cannot detect a change to that
    list -- deleting an entry would delete its own case (BB section 5b, from
    BA17 item 4). This test failed when BB added the `predictor` arm, which is
    exactly what it is for; it was then updated deliberately.
    """
    arms = {t["arm"] for t in TARGETS}
    # UPDATED DELIBERATELY at BM: `inferential_primary` is the sixth arm,
    # carrying `design3_join_ar.json`. This test failing on that addition is
    # the behaviour the docstring above describes, not a regression.
    # UPDATED DELIBERATELY at BY: `build_outputs` is the SEVENTH arm, carrying
    # the eleven artifacts the paper's numbers are read from. A new arm rather
    # than a widening, because `build_inputs`' committed tolerance reasons about
    # a fixed set of six and `inferential_primary`'s about one artifact.
    # UPDATED DELIBERATELY at CC: `determinism` is the EIGHTH arm, carrying the
    # probe's eight reduced comparison documents. A new arm rather than a
    # widening of `build_inputs`, on a ground none of the existing arms can
    # state: its producer CANNOT BE RE-RUN AT ALL after ~2026-08-23, because
    # null_determinism_probe.sbatch:77 consumes cache/eval.h5. Every other
    # file-integrity arm says "the code may no longer produce this"; this one
    # says "the code cannot".
    assert arms == {"n19", "n39", "predictor", "null", "build_inputs",
                    "build_outputs",
                    "inferential_primary", "determinism"}
    assert sum(1 for t in TARGETS if t["arm"] == "determinism") == 8
    assert sum(1 for t in TARGETS if t["arm"] == "n39") >= 3
    assert sum(1 for t in TARGETS if t["arm"] == "predictor") == 1
    # It failed a THIRD time when BK added the classification build's six
    # inputs, and was updated deliberately. The count is written out for the
    # same reason as the two above: BJ-C1 enumerated exactly six inputs, and a
    # seventh appearing -- or one quietly dropping -- is the thing this catches.
    #
    # UPDATED DELIBERATELY at BS: THE SEVENTH INPUT ARRIVED, AND THIS TEST
    # CAUGHT IT, WHICH IS WHAT IT IS FOR. BJ-C1's list of six was INCOMPLETE,
    # not merely superseded -- BQ7c established that the orphan side reads the
    # pooled RAW null (`null_gain_raw_compliant.h5`, 21.48 GiB) which no target
    # watched, corrections-log entry 127. The count goes 6 -> 7 because the
    # BUILD always had seven inputs, not because a seventh was invented here.
    assert sum(1 for t in TARGETS if t["arm"] == "build_inputs") == 7
    # BOTH channels, asserted as a COUNT because covering only one is the
    # mistake this arm exists to prevent: the raw is float32 and quantises
    # last-bit movement away, the reductions are float64 and preserve it, so
    # each is blind to what the other detects. It failed again here when the
    # BG/BH build added the `null` arm, and was updated deliberately -- twice
    # now, which is the evidence that a written-out set works.
    assert sum(1 for t in TARGETS if t["arm"] == "null") == 2
    assert set(TOLERANCE) == arms


def test_the_n39_tolerance_states_it_has_no_earlier_baseline():
    """The honest half: N = 39 has not reproduced across a refactor it never
    saw, and the manifest must say so rather than implying parity with N = 19."""
    assert "FROM THE FIRST BASELINE FORWARD" in TOLERANCE["n39"]
    assert "No earlier baseline exists" in TOLERANCE["n39"]


def test_the_primary_marginal_mixture_and_endpoint_groups_are_all_covered():
    """AM5 names three N = 39 things to protect: the marginal, the mixture fit
    including the w0 profile, and the endpoint groups. The endpoint groups live
    INSIDE the marginal document (`ag_endpoint_groups`), so covering the
    marginal covers them -- asserted here so that coupling is deliberate."""
    names = {t["name"] for t in TARGETS}
    assert {"rf_marginal_n39", "rf_mixture_n39", "rf_counts_n39"} <= names


def test_the_M1_targets_are_not_whole_file_hashed():
    """A whole-file sha256 of an M1 document is a check that can ONLY fail
    (gotcha 25a): it embeds `identity/git_sha`, which moves on every re-run."""
    for t in TARGETS:
        if "marginal" in t["name"]:
            assert t["hash_scope"] == "statistics_only", t["name"]
        else:
            assert t["hash_scope"] == "whole_file", t["name"]


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------

def test_describe_reports_a_missing_file_as_ABSENT(tmp_path):
    assert describe(tmp_path / "nope.json")["status"] == "ABSENT"


def test_describe_counts_json_leaves(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"a": 1, "b": [1, 2, 3], "c": {"d": 4, "e": 5}}))
    d = describe(p)
    assert d["n_leaves"] == 6
    assert d["top_level_keys"] == ["a", "b", "c"]


def test_describe_records_h5_shapes_and_identity_attrs(tmp_path):
    import h5py
    p = tmp_path / "x.h5"
    with h5py.File(p, "w") as hf:
        hf.create_dataset("match_count", data=np.zeros((20, 10240), dtype=np.int16))
        hf.attrs["n_compare_seeds"] = 39
        hf.attrs["n_anchors"] = 20
        hf.attrs["theta"] = 0.7
    d = describe(p)
    assert d["datasets"] == {"match_count": [20, 10240]}
    assert d["n_compare_seeds"] == 39 and d["n_anchors"] == 20


# ---------------------------------------------------------------------------
# capture / verify
# ---------------------------------------------------------------------------

def _root(tmp_path, n_leaves_extra=0):
    r = tmp_path / "analysis"
    r.mkdir(parents=True, exist_ok=True)
    import h5py
    for t in TARGETS:
        p = r / t["path"]
        if p.suffix == ".json":
            doc = {"identity": {"git_sha": "aaa"}, "v": list(range(3 + n_leaves_extra))}
            p.write_text(json.dumps(doc))
        else:
            with h5py.File(p, "w") as hf:
                hf.create_dataset("match_count",
                                  data=np.zeros((20, 8), dtype=np.int16))
                hf.attrs["n_compare_seeds"] = 39
    return r


def test_capture_then_verify_reproduces(tmp_path):
    r = _root(tmp_path)
    man = capture(r, REPO, None)
    rep = verify(r, man)
    assert rep["verdict"] == "OK", rep["problems"]
    # UPDATED DELIBERATELY at BY: `build_outputs` is the seventh arm.
    # UPDATED DELIBERATELY at CC: `determinism` is the eighth.
    assert set(rep["by_arm"]) == {"n19", "n39", "predictor", "null",
                                  "build_inputs", "inferential_primary",
                                  "build_outputs", "determinism"}


def test_a_changed_byte_is_detected(tmp_path):
    r = _root(tmp_path)
    man = capture(r, REPO, None)
    # perturb a whole-file-hashed target
    tgt = next(t for t in TARGETS if t["hash_scope"] == "whole_file"
               and t["path"].endswith(".json"))
    p = r / tgt["path"]
    doc = json.loads(p.read_text())
    doc["v"] = [9, 9, 9]
    p.write_text(json.dumps(doc))
    rep = verify(r, man)
    assert rep["verdict"] == "FAILED"
    assert any(tgt["name"] in q for q in rep["problems"])


def test_a_leaf_appearing_in_an_M1_target_is_detected(tmp_path):
    """Statistics-only targets are not hash-compared, so the leaf count is the
    check -- and it must fire when a field appears."""
    r = _root(tmp_path)
    man = capture(r, REPO, None)
    tgt = next(t for t in TARGETS if t["hash_scope"] == "statistics_only")
    p = r / tgt["path"]
    doc = json.loads(p.read_text())
    doc["surprise"] = 1
    p.write_text(json.dumps(doc))
    rep = verify(r, man)
    assert rep["verdict"] == "FAILED"
    assert any("leaf count" in q for q in rep["problems"])


def test_a_git_sha_change_alone_does_NOT_fail_an_M1_target(tmp_path):
    """The whole reason M1 targets are statistics-only: the embedded provenance
    moves on every run and must not be a failure."""
    r = _root(tmp_path)
    man = capture(r, REPO, None)
    tgt = next(t for t in TARGETS if t["hash_scope"] == "statistics_only")
    p = r / tgt["path"]
    doc = json.loads(p.read_text())
    doc["identity"]["git_sha"] = "bbb"
    p.write_text(json.dumps(doc))
    rep = verify(r, man)
    assert all(q for q in rep["problems"] if tgt["name"] not in q) or not rep["problems"]


def test_a_baselined_target_that_vanishes_is_a_problem(tmp_path):
    r = _root(tmp_path)
    man = capture(r, REPO, None)
    (r / TARGETS[0]["path"]).unlink()
    rep = verify(r, man)
    assert rep["verdict"] == "FAILED"
    assert any("now absent" in q for q in rep["problems"])


def test_an_absent_target_at_capture_is_NO_BASELINE_not_a_pass(tmp_path):
    """A manifest that silently omits an absent target is a check that passes on
    nothing. It must be recorded as having no baseline."""
    r = _root(tmp_path)
    (r / TARGETS[0]["path"]).unlink()
    man = capture(r, REPO, None)
    rep = verify(r, man)
    row = next(x for x in rep["rows"] if x["name"] == TARGETS[0]["name"])
    assert row["verdict"] == "NO_BASELINE"


def test_recapture_carries_first_baselined_at_forward(tmp_path):
    """A re-capture must not reset the date a target's expectation starts from,
    or 'N = 39 has reproduced since X' could be manufactured by re-capturing."""
    r = _root(tmp_path)
    first = capture(r, REPO, None)
    for t in first["targets"]:
        t["first_baselined_at"] = "0000000000deadbeef"
    second = capture(r, REPO, first)
    for t in second["targets"]:
        assert t["first_baselined_at"] == "0000000000deadbeef", t["name"]
        assert t["captured_at"] != "0000000000deadbeef"


# ---------------------------------------------------------------------------
# the entry point and its exit code
# ---------------------------------------------------------------------------

def _run(args):
    return subprocess.run(
        [sys.executable, "-m", "src.analysis.canary", *map(str, args)],
        capture_output=True, text=True, cwd=REPO)


def test_main_captures_then_verifies_zero(tmp_path):
    r = _root(tmp_path)
    m = tmp_path / "manifest.json"
    assert _run(["--root", r, "--manifest", m, "--capture",
                 "--repo", REPO]).returncode == 0
    out = _run(["--root", r, "--manifest", m])
    assert out.returncode == 0, out.stdout + out.stderr
    assert "verdict OK" in out.stdout


def test_main_exits_nonzero_when_a_target_moves(tmp_path):
    r = _root(tmp_path)
    m = tmp_path / "manifest.json"
    _run(["--root", r, "--manifest", m, "--capture", "--repo", REPO])
    tgt = next(t for t in TARGETS if t["hash_scope"] == "whole_file"
               and t["path"].endswith(".json"))
    (r / tgt["path"]).write_text(json.dumps({"v": [0]}))
    out = _run(["--root", r, "--manifest", m])
    assert out.returncode == 1, out.stdout
    assert "verdict FAILED" in out.stdout


# ---------------------------------------------------------------------------
# BB section 4 -- THE ALIGNMENT PREDICTOR, AND ADDITION-VS-REBASELINE
# ---------------------------------------------------------------------------

def test_the_alignment_predictor_is_a_target_on_its_own_arm():
    """BA14a found the gap: none of the nine existing targets covers the
    predictor, which is the axis BOTH headlines are compared through. Its own
    arm keeps its tolerance separate, exactly as AK1 keeps n19 and n39 apart."""
    t = [x for x in TARGETS if x["name"] == "alignment_predictor"]
    assert len(t) == 1
    assert t[0]["path"] == "gtilde_compliant_b_ratio.h5"
    assert t[0]["arm"] == "predictor"
    assert t[0]["hash_scope"] == "whole_file"


def test_the_predictor_tolerance_says_it_is_coverage_not_a_rebaseline():
    """The two are different operations with different justifications and a
    reader should not have to infer which happened."""
    assert "FROM THE FIRST BASELINE FORWARD" in TOLERANCE["predictor"]
    assert "cannot be back-dated" in TOLERANCE["predictor"]


def test_every_arm_named_by_a_target_has_a_declared_tolerance():
    """A target on an arm with no tolerance would be watched with no declared
    expectation, which READS as coverage and is not."""
    cov = tolerance_coverage()
    assert cov["verdict"] == "COVERED"
    assert cov["arms_without_tolerance"] == []


def test_tolerance_coverage_REPORTS_an_arm_that_has_no_tolerance():
    """THE CASE THE REAL TARGETS CANNOT EXHIBIT. They are fully covered, so a
    test that could only call this with them would pass whether or not the
    check exists -- and the mutation removing it survived exactly that way on
    the first matrix run (gotcha 16)."""
    cov = tolerance_coverage(
        targets=({"name": "x", "arm": "n19"}, {"name": "y", "arm": "orphan"}),
        tolerance={"n19": "..."})
    assert cov["verdict"] == "UNCOVERED_ARM"
    assert cov["arms_without_tolerance"] == ["orphan"]


def test_tolerance_coverage_reports_a_tolerance_with_no_targets_separately():
    """An unused tolerance is not a defect and must not be reported as one --
    the check must be capable of more than one outcome (gotcha 25)."""
    cov = tolerance_coverage(
        targets=({"name": "x", "arm": "n19"},),
        tolerance={"n19": "...", "spare": "..."})
    assert cov["verdict"] == "COVERED"
    assert cov["tolerance_without_targets"] == ["spare"]


def test_a_new_target_is_reported_as_an_ADDITION_and_dated_today(tmp_path):
    """`first_baselined_at` for a target added today must be TODAY's commit.
    If it were carried forward from anywhere, 'has reproduced since' would be
    back-dated to a period when nothing was watching."""
    r = _root(tmp_path)
    old = capture(r, REPO, None)
    # a manifest that predates the addition
    old["targets"] = [t for t in old["targets"]
                      if t["name"] != "alignment_predictor"]
    new = capture(r, REPO, old)
    delta = compare_manifests(old, new)
    assert delta["verdict"] == "ADDITION_ONLY"
    assert [a["name"] for a in delta["added"]] == ["alignment_predictor"]
    assert delta["rebaselined"] == []
    assert delta["problems"] == []
    added = next(t for t in new["targets"]
                 if t["name"] == "alignment_predictor")
    assert added["first_baselined_at"] == new["identity"]["captured_at_git_sha"]


def test_a_pre_existing_target_whose_bytes_moved_is_a_REBASELINE_not_an_addition(
        tmp_path):
    """THE LOAD-BEARING CASE. `capture` writes whatever is on disk today, so a
    drifted target would be silently re-baselined by the very call that adds a
    new one. BB section 4 authorises the addition and NOT that."""
    r = _root(tmp_path)
    old = capture(r, REPO, None)
    tgt = next(t for t in TARGETS if t["hash_scope"] == "whole_file"
               and t["path"].endswith(".json"))
    p = r / tgt["path"]
    doc = json.loads(p.read_text())
    doc["drifted"] = True
    p.write_text(json.dumps(doc))
    new = capture(r, REPO, old)
    delta = compare_manifests(old, new)
    assert delta["verdict"] == "NOT_AN_ADDITION_ONLY_CHANGE"
    assert [x["name"] for x in delta["rebaselined"]] == [tgt["name"]]
    assert any("REBASELINE" in q for q in delta["problems"])


def test_a_drift_that_changes_ONLY_the_BYTES_is_still_a_rebaseline(tmp_path):
    """THE DISCRIMINATING FIXTURE FOR `sha256`. The case above adds a KEY, so
    the leaf count and the size move too and a manifest that never hashed
    anything would still catch it -- which is how the `sha256`-dropped
    mutation survived on the first matrix run. Here the values change in
    place: SAME leaf count, SAME byte length, different bytes. Only the hash
    can see it."""
    r = _root(tmp_path)
    old = capture(r, REPO, None)
    tgt = next(t for t in TARGETS if t["hash_scope"] == "whole_file"
               and t["path"].endswith(".json"))
    p = r / tgt["path"]
    before = p.read_bytes()
    doc = json.loads(p.read_text())
    assert doc["v"] == [0, 1, 2]
    doc["v"] = [9, 9, 9]
    p.write_text(json.dumps(doc))
    after = p.read_bytes()
    assert len(before) == len(after) and before != after
    new = capture(r, REPO, old)
    delta = compare_manifests(old, new)
    # the leaf count is UNCHANGED, so this can only be caught by the hash
    was = next(t for t in old["targets"] if t["name"] == tgt["name"])
    now = next(t for t in new["targets"] if t["name"] == tgt["name"])
    assert was["n_leaves"] == now["n_leaves"] and was["bytes"] == now["bytes"]
    assert delta["verdict"] == "NOT_AN_ADDITION_ONLY_CHANGE"
    assert [x["name"] for x in delta["rebaselined"]] == [tgt["name"]]


def test_a_changed_captured_at_alone_is_NOT_a_rebaseline():
    """Gotcha 25a, behaviourally rather than by asserting the constant.
    `captured_at` moves on every capture by construction, so including it
    would make EVERY capture read as a rebaseline -- a check that can only
    fail, on the instrument built to detect a rebaseline. The two captures in
    the NO_CHANGE test share a HEAD, so they cannot exhibit this."""
    old = {"targets": [{"name": "t", "arm": "n19", "path": "a.json",
                        "hash_scope": "whole_file", "sha256": "aa",
                        "n_leaves": 3, "bytes": 10, "status": "PRESENT",
                        "first_baselined_at": "sha_old",
                        "captured_at": "sha_old"}]}
    new = json.loads(json.dumps(old))
    new["targets"][0]["captured_at"] = "sha_new"
    delta = compare_manifests(old, new)
    assert delta["verdict"] == "NO_CHANGE"
    assert delta["rebaselined"] == [] and delta["problems"] == []


def test_a_target_that_vanishes_from_the_manifest_is_a_problem(tmp_path):
    """A manifest that quietly drops a target is a check that stops checking."""
    r = _root(tmp_path)
    old = capture(r, REPO, None)
    new = json.loads(json.dumps(old))
    new["targets"] = new["targets"][:-1]
    delta = compare_manifests(old, new)
    assert delta["verdict"] == "NOT_AN_ADDITION_ONLY_CHANGE"
    assert delta["removed"]


def test_compare_manifests_reports_NO_CHANGE_when_nothing_moved(tmp_path):
    """The instrument must be capable of more than one outcome (gotcha 25):
    if it could only ever say ADDITION_ONLY it would prove nothing."""
    r = _root(tmp_path)
    old = capture(r, REPO, None)
    new = capture(r, REPO, old)
    delta = compare_manifests(old, new)
    assert delta["verdict"] == "NO_CHANGE"
    assert delta["added"] == [] and delta["rebaselined"] == []


def test_captured_at_is_excluded_so_the_check_is_not_one_that_can_only_fail():
    """`captured_at` moves on every capture by construction. Comparing it
    would make every capture read as a rebaseline -- gotcha 25a's shape, on
    the instrument built to detect a rebaseline."""
    from src.analysis.canary import BASELINE_FIELDS
    assert "captured_at" not in BASELINE_FIELDS
    assert "sha256" in BASELINE_FIELDS
    assert "first_baselined_at" in BASELINE_FIELDS


# ===================================================== BC7: the device class

def test_every_real_target_carries_a_declared_device_class():
    from src.analysis.canary import device_class_coverage
    got = device_class_coverage()
    assert got["verdict"] == "COVERED"
    assert got["targets_without_device_class"] == []
    assert got["undeclared_device_classes"] == []


def test_a_target_with_no_device_class_is_UNCOVERED():
    """THE FIXTURE MUST BE ABLE TO EXHIBIT THE CONDITION. The real TARGETS are
    fully annotated, so a check that could only be called with them would pass
    with its branch deleted -- exactly how BB12 item 2's `tolerance_coverage`
    mutation survived (gotcha 16)."""
    from src.analysis.canary import device_class_coverage
    got = device_class_coverage(
        targets=({"name": "t1", "arm": "n19", "device_class":
                  "MNN_COSINE_THRESHOLD"},
                 {"name": "t2", "arm": "n19"}))
    assert got["verdict"] == "TARGET_WITHOUT_DEVICE_CLASS"
    assert got["targets_without_device_class"] == ["t2"]


def test_a_device_class_outside_the_vocabulary_is_UNDECLARED():
    """A value nobody declared is not coverage. Separate branch from the
    missing-field one, with a separate verdict, because they need different
    fixes."""
    from src.analysis.canary import device_class_coverage
    got = device_class_coverage(
        targets=({"name": "t1", "arm": "n19", "device_class": "probably_fine"},))
    assert got["verdict"] == "UNDECLARED_DEVICE_CLASS"
    assert got["undeclared_device_classes"] == ["probably_fine"]


def test_NOT_ESTABLISHED_is_documented_as_ignorance_not_independence():
    """The distinction is the whole content of BC7's per-target claim: an
    absent measurement is reported as absent, never read as agreement."""
    from src.analysis.canary import DEVICE_CLASS_CONDITION
    assert "does NOT mean device-independent" in DEVICE_CLASS_CONDITION
    assert "never to widen the tolerance" in DEVICE_CLASS_CONDITION


def test_the_MNN_targets_are_the_ones_that_read_a_cosine_threshold():
    """WRITTEN OUT rather than derived from TARGETS, so a target silently
    changing class fails this test (BB5b). These seven descend from the MNN
    counts; the other three have no known threshold in their path."""
    from src.analysis.canary import device_class_coverage
    by = device_class_coverage()["targets_by_device_class"]
    assert by["MNN_COSINE_THRESHOLD"] == [
        "m4_c1_all_seeds", "rf_counts_n19", "rf_counts_n39",
        "rf_marginal_n19", "rf_marginal_n39", "rf_mixture_n19",
        "rf_mixture_n39"]
    # The two `null` targets are NOT_ESTABLISHED on the DEVICE axis, and that
    # is a precise claim rather than a shrug: the BG/BH build measured
    # RUN-to-RUN nondeterminism on one device class (`index_add_` atomics), and
    # never regenerated the null on a SECOND device class. An absent
    # measurement is reported absent, never read as independence.
    # `design3_join_ar` added at BM. It is NOT_ESTABLISHED for a reason worth
    # keeping distinct from the two `null` targets' reason above: design3_join
    # READS rf_counts and performs no cosine and no TopK gather of its own, so
    # no device-sensitivity measurement exists for its own path -- while it
    # still INHERITS MNN sensitivity through rf_counts. Inherited exposure with
    # no measurement is NOT_ESTABLISHED, not MNN_COSINE_THRESHOLD, because this
    # field records what was measured about the target's own production.
    assert by["NOT_ESTABLISHED"] == [
        "alignment_predictor", "cv_ci_hardbin", "design3_join_ar",
        "label_store_f67485d", "m3_degenerate", "null_pooled_raw_digests",
        "null_pooled_reductions"]
    # BK's fifth: the label store has no TopK gather and no MNN threshold in
    # its production path, so no device-sensitivity measurement exists for it.
    # That is NOT_ESTABLISHED used precisely, and it still does not assert
    # device-independence.


def test_the_TOPK_GATHER_targets_are_the_ones_whose_content_descends_from_a_gather():
    """WRITTEN OUT, not derived (BB5b). Five of BK's six additions descend
    from the TopK gather gotcha 36 measured; the label store does not, and
    the difference between those two claims is the reason the value exists."""
    from src.analysis.canary import device_class_coverage
    by = device_class_coverage()["targets_by_device_class"]
    # UPDATED DELIBERATELY at BS. `null_pooled_raw_compliant` is the SIXTH:
    # its numeric content descends from job 20142136's gather (`device=cuda`),
    # and job 20184677 only RESLICED it -- a reslice performs no gather, so the
    # mechanism gotcha 36 measured applies to it exactly as to the other five.
    # UPDATED DELIBERATELY at BY: all ELEVEN `build_outputs` targets are
    # TOPK_GATHER. None performs a gather itself -- each reduces an artifact that
    # descends from one -- so the exposure is INHERITED. That is still an
    # exposure, and recording it as NOT_ESTABLISHED would say the opposite.
    # UPDATED DELIBERATELY at CC: the eight `determinism` targets are
    # TOPK_GATHER on the SAME inherited-exposure ground BY records above -- they
    # summarise differences between null arrays that descend from a gather.
    # **AND THE FIELD NAMES A MECHANISM THESE DOCUMENTS DO NOT MEASURE, which is
    # why it is worth an explicit comment rather than a silent addition.** What
    # they quantify is RUN-TO-RUN variation on ONE device class, whose measured
    # cause is `index_add_` CUDA atomics (gotcha 35). Gotcha 36's BETWEEN-class
    # near-tie sensitivity -- what TOPK_GATHER_NEAR_TIE asserts -- has never been
    # measured on these artifacts. The value records an INHERITED exposure, not
    # a measurement of these files, exactly as BY's eleven do; NOT_ESTABLISHED
    # would say the exposure is absent, which is the stronger and false claim.
    assert by["TOPK_GATHER_NEAR_TIE"] == sorted([
        "ap_experimental_cpu", "ap_pooled_compliant",
        "null_experimental_raw", "null_experimental_reductions",
        "null_pooled_compliant", "null_pooled_raw_compliant",
        "br_floor_concentration", "classification_2x2_bq",
        "classification_bh_experimental_bo", "concept_repro_ax",
        "decomposition_percentile_curve_by", "decomposition_reporting_bx",
        "grid_composition_bt", "matched_pair_arm_bu",
        "never_detected_join_by", "protein_vs_residue_bw",
        "protein_vs_residue_feature_view_bw",
        "determ_AB_red", "determ_AB_raw", "determ_AC_red", "determ_AC_raw",
        "determ_A_vs_ref_red", "determ_A_vs_ref_raw",
        "determ_C_vs_ref_red", "determ_C_vs_ref_raw"])


def test_TOPK_GATHER_is_a_SEPARATE_value_and_not_a_widened_MNN():
    """Widening MNN_COSINE_THRESHOLD to mean 'some measured device-sensitive
    mechanism' was the alternative BJ-C3 named, and it collapses the per-target
    distinction BC7 drew. The two mechanisms have different remedies, so a
    DIFFERS verdict must say which one to look at."""
    from src.analysis.canary import (
        DEVICE_CLASSES, MNN, TOPK_GATHER, UNKNOWN)
    assert TOPK_GATHER != MNN
    assert set(DEVICE_CLASSES) == {MNN, TOPK_GATHER, UNKNOWN}
    assert MNN == "MNN_COSINE_THRESHOLD"      # unchanged meaning


def test_the_manifest_carries_the_condition_so_a_reader_of_the_JSON_sees_it(
        tmp_path):
    """Not only the module docstring: a future session reading `baseline.json`
    after a DIFFERS verdict is the exact reader BC7 is for."""
    r = _root(tmp_path)
    man = capture(r, REPO, None)
    assert "device_class_condition" in man["identity"]
    assert man["identity"]["baseline_device_class"] == "rtx_5000_ada"
    assert all("device_class" in t for t in man["targets"])


# ============================================ BK: the classification's inputs

def test_the_six_build_inputs_are_the_ones_BJ_C1_enumerated():
    """WRITTEN OUT, not derived from TARGETS (BB5b). BJ-C1 enumerated the
    classification build's inputs by reading every artifact's own attrs; this
    is that list, so adding a seventh or dropping one fails here rather than
    silently changing what 'the build's inputs are covered' means."""
    got = {t["name"]: t["path"] for t in TARGETS if t["arm"] == "build_inputs"}
    # UPDATED DELIBERATELY at BS, and the docstring above predicted this exact
    # event: "adding a seventh ... fails here". BJ-C1's enumeration was
    # INCOMPLETE (BQ7c, entry 127) -- the orphan side reads the pooled RAW
    # null, which the listed pooled input cannot substitute for because
    # `null_cmax_*` are MAXIMA and cannot produce per-pair p-values.
    assert got == {
        "ap_pooled_compliant": "feature_ap_main_compliant.h5",
        "null_pooled_compliant": "null_pilot_compliant.h5",
        "null_pooled_raw_compliant": "null_gain_raw_compliant.h5",
        "ap_experimental_cpu": "feature_ap_experimental_av_cpu.h5",
        "null_experimental_reductions": "null_experimental_bg.h5",
        "null_experimental_raw": "null_gain_raw_experimental_bg.h5",
        "label_store_f67485d":
            "residue_labels_subtyped_untruncated_pgate5_f67485d.h5",
    }


def test_NO_target_is_on_the_scratch_root_since_BT_relocated_the_seventh():
    """BS-C2, amended by BT-A3. From BS to BT `null_pooled_raw_compliant` was
    the one target NOT on project; BT copied it to the analysis root because
    scratch purges at 30 days and it is a load-bearing classification input.

    WRITTEN OUT, not derived (gotcha 32b): if any target is ever placed on
    scratch again, that is a decision with a purge clock attached and it must
    not pass silently. The previous form of this test asserted the seventh
    input WAS on scratch; the assertion moved with the artifact and the
    docstring says so rather than leaving a reader to diff it.
    """
    on_scratch = sorted(t["name"] for t in TARGETS
                        if t.get("root") == "scratch")
    assert on_scratch == []
    assert all("root" not in t for t in TARGETS)


def test_resolve_puts_a_scratch_target_under_the_scratch_root(tmp_path):
    """The indirection has to actually route, not merely be declared."""
    from src.analysis.canary import resolve
    roots = {"analysis": tmp_path / "a", "scratch": tmp_path / "s"}
    spec = {"name": "x", "path": "f.h5", "root": "scratch"}
    assert resolve(roots, spec) == tmp_path / "s" / "f.h5"
    assert resolve(roots, {"name": "y", "path": "g.h5"}) == tmp_path / "a" / "g.h5"


def test_resolve_falls_back_to_analysis_when_scratch_is_not_supplied(tmp_path):
    """A missing --scratch-root must make the target ABSENT, never make it
    match some other file. Reported-wrong beats silently-satisfied (gotcha 15).
    """
    from src.analysis.canary import resolve
    roots = {"analysis": tmp_path / "a"}
    spec = {"name": "x", "path": "f.h5", "root": "scratch"}
    assert resolve(roots, spec) == tmp_path / "a" / "f.h5"
    assert not (tmp_path / "a" / "f.h5").exists()


def test_resolve_REFUSES_an_unknown_root_name(tmp_path):
    """A typo'd root would otherwise resolve somewhere nobody checks."""
    from src.analysis.canary import resolve
    with pytest.raises(ValueError, match="not one of"):
        resolve({"analysis": tmp_path}, {"name": "x", "path": "f.h5",
                                         "root": "scrach"})


def test_the_build_inputs_tolerance_says_it_is_WEAKER_and_not_reproduction():
    """The load-bearing sentence. A green verdict on this arm reads exactly
    like a green verdict on every other arm, so the difference has to be in the
    tolerance text a reader of the JSON reaches."""
    t = TOLERANCE["build_inputs"]
    assert "FILE INTEGRITY ONLY" in t
    assert "WEAKER CLAIM THAN EVERY OTHER ARM" in t
    assert "NOT A REPRODUCTION GUARANTEE" in t
    assert "catches CODE DRIFT NOT AT ALL" in t


def test_the_pooled_null_floor_is_explicitly_NOT_transferred_to_this_arm():
    """The failure this arm is most likely to invite: reusing the pooled null's
    control-measured floor for the experimental one. Same code family is not
    same artifact, and four of the six ran on CPU where the CUDA atomics
    argument does not even apply -- which is a mechanism claim, not a
    measurement (gotcha 22)."""
    t = TOLERANCE["build_inputs"]
    assert "NOT TRANSFERRED TO ANY TARGET" in t
    assert "UNKNOWN and is NOT assumed in either direction" in t


def test_the_superseded_pooled_null_target_says_which_question_it_answers():
    """BJ-C2. A green verify on `null_pilot_k.h5` is readable as that artifact
    being fine to use, and the pre-flight gate refuses it for every consumer.
    The target is KEPT -- removing it loses reproduction history -- and its
    reason now draws the distinction."""
    t = next(x for x in TARGETS if x["name"] == "null_pooled_reductions")
    assert t["path"] == "null_pilot_k.h5"          # NOT re-pointed
    assert "REFUSED FOR CONSUMERS" in t["why"]
    assert "superseded_for_axis" in t["why"]
    assert "null_pilot_compliant.h5" in t["why"]
    # and the artifact it names is genuinely covered, on the other arm
    assert any(x["path"] == "null_pilot_compliant.h5" for x in TARGETS)


# ---------------------------------------------------------------- the fields

def test_every_real_target_pairs_its_build_device_fields():
    from src.analysis.canary import build_device_coverage
    got = build_device_coverage()
    assert got["verdict"] == "COVERED"
    assert got["class_without_provenance"] == []
    assert got["provenance_without_class"] == []


def test_only_the_BK_additions_state_their_own_build_device():
    """The half that keeps this an ADDITION. Every target predating BK states
    no `build_device_class`, so each still inherits the module default and NONE
    is re-described -- which is what 'retain the constant as a documented
    default' has to mean operationally."""
    from src.analysis.canary import build_device_coverage
    got = build_device_coverage()
    # UPDATED DELIBERATELY at BM. `design3_join_ar` states its own build device
    # on BK's evidence class (job 20813625's AllocTRES). **That is an ADDITION
    # stating a fact about itself, not a back-fill**: the two operations are
    # distinguished by whether an EXISTING target is re-described, and none is.
    # The invariant this test guards -- every target predating BK still states
    # nothing and still inherits the module default -- is unchanged, and is
    # asserted by name below rather than by the arm proxy it used to use.
    # UPDATED DELIBERATELY at BS, on the same footing as BM's addition:
    # `null_pooled_raw_compliant` states a fact about ITSELF (job 20142136's
    # log line 10 reads `device=cuda`), and NO existing target is re-described.
    # The invariant is unchanged -- every target predating BK still states
    # nothing -- and the inheriting count stays 12 because the addition states
    # its own rather than inheriting.
    # UPDATED DELIBERATELY at BY, on the same footing as BM's and BS's: each of
    # the eleven `build_outputs` targets states a fact about ITSELF, and NO
    # existing target is re-described. The invariant this test guards -- every
    # target predating BK still states nothing and inherits the default -- is
    # unchanged, which is why the inheriting count stays at 12.
    assert got["targets_stating_their_own"] == sorted([
        "ap_experimental_cpu", "ap_pooled_compliant", "design3_join_ar",
        "label_store_f67485d",
        "null_experimental_raw", "null_experimental_reductions",
        "null_pooled_compliant", "null_pooled_raw_compliant",
        "br_floor_concentration", "classification_2x2_bq",
        "classification_bh_experimental_bo", "concept_repro_ax",
        "decomposition_percentile_curve_by", "decomposition_reporting_bx",
        "grid_composition_bt", "matched_pair_arm_bu",
        "never_detected_join_by", "protein_vs_residue_bw",
        "protein_vs_residue_feature_view_bw"])
    # UPDATED DELIBERATELY at CC, AND THE PROXY IS REPLACED RATHER THAN
    # RE-TUNED. CC's eight `determinism` targets are the FIRST additions that
    # INHERIT the default rather than stating their own -- deliberately, because
    # their producing job only REQUESTS `--gpus=rtx_5000_ada:1` and no log was
    # read, so `MEASURED` and `INFERRED_FROM_PRODUCING_JOB_LOGS` would both
    # overstate the evidence (gotcha 15's pairing rule). That makes the bare
    # count 20 and, more importantly, makes a COUNT the wrong instrument: it can
    # no longer distinguish "a pre-BK target was re-described" from "an addition
    # inherited", which is the ONE distinction this test exists to draw.
    # So the twelve are asserted BY NAME -- the same move BM's comment describes
    # making for `targets_stating_their_own` -- and CC's eight are listed
    # separately, so a pre-BK target acquiring a class fails here loudly instead
    # of being absorbed into a number.
    _PRE_BK_INHERITING = sorted([
        "alignment_predictor", "cv_ci_hardbin", "m3_degenerate",
        "m4_c1_all_seeds", "null_pooled_raw_digests", "null_pooled_reductions",
        "rf_counts_n19", "rf_counts_n39", "rf_marginal_n19", "rf_marginal_n39",
        "rf_mixture_n19", "rf_mixture_n39"])
    _CC_INHERITING = sorted([
        "determ_AB_red", "determ_AB_raw", "determ_AC_red", "determ_AC_raw",
        "determ_A_vs_ref_red", "determ_A_vs_ref_raw",
        "determ_C_vs_ref_red", "determ_C_vs_ref_raw"])
    assert len(_PRE_BK_INHERITING) == 12
    # THE INVARIANT, unchanged since BK: every target predating BK states
    # nothing and inherits the module default.
    assert set(_PRE_BK_INHERITING) <= set(got["targets_inheriting_the_default"])
    assert got["targets_inheriting_the_default"] == sorted(
        _PRE_BK_INHERITING + _CC_INHERITING)
    assert got["module_default"] == "rtx_5000_ada"


def test_the_two_CPU_built_inputs_are_recorded_as_CPU_not_as_the_default():
    """LIMB 2 of BJ-C3's stop condition, discharged. Four of the six were
    produced on CPU and the manifest previously had no field in which they
    could say so, so adding them would have recorded `rtx_5000_ada` for
    artifacts measured to be CPU."""
    from src.analysis.canary import build_device_coverage
    by = build_device_coverage()["targets_by_build_device"]
    # `design3_join_ar` added at BM on the same evidence class: job 20813625,
    # partition `day`, cpu=8, AllocTRES carrying no gres/gpu.
    # UPDATED DELIBERATELY at BY: all eleven `build_outputs` targets were built
    # on CPU. Nine are INFERRED_FROM_PRODUCING_JOB_LOGS; the two BY produced
    # itself are MEASURED, read off the run rather than reconstructed. The
    # rtx_5000_ada list below is UNTOUCHED, which is the half this protects.
    assert by["cpu"] == sorted(["ap_experimental_cpu", "design3_join_ar",
                         "label_store_f67485d",
                         "null_experimental_raw",
                         "null_experimental_reductions",
                         "br_floor_concentration", "classification_2x2_bq",
                         "classification_bh_experimental_bo",
                         "concept_repro_ax",
                         "decomposition_percentile_curve_by",
                         "decomposition_reporting_bx", "grid_composition_bt",
                         "matched_pair_arm_bu", "never_detected_join_by",
                         "protein_vs_residue_bw",
                         "protein_vs_residue_feature_view_bw"])
    # UPDATED DELIBERATELY at BS: the pooled RAW null joins the two pooled
    # artifacts it shares a gather with. All three descend from GPU gathers;
    # the CPU list is untouched, which is the half this test exists to protect.
    assert by["rtx_5000_ada"] == ["ap_pooled_compliant",
                                  "null_pooled_compliant",
                                  "null_pooled_raw_compliant"]


def test_a_build_device_class_without_its_provenance_is_UNPAIRED():
    """THE CASE THE REAL TARGETS CANNOT EXHIBIT -- they are consistent, so a
    test that could only call this with them would pass with the branch
    deleted (gotcha 16, and the fixture lesson BB12 item 2 paid for)."""
    from src.analysis.canary import build_device_coverage
    got = build_device_coverage(
        targets=({"name": "t1", "build_device_class": "cpu",
                  "build_device_provenance": "MEASURED"},
                 {"name": "t2", "build_device_class": "cpu"}))
    assert got["verdict"] == "UNPAIRED_BUILD_DEVICE_FIELDS"
    assert got["class_without_provenance"] == ["t2"]


def test_a_provenance_flag_without_a_class_is_ALSO_UNPAIRED():
    """The other direction. A provenance flag with no class is a claim about
    nothing, and a check capable of only one of two symmetric failures is the
    shape gotcha 25 names."""
    from src.analysis.canary import build_device_coverage
    got = build_device_coverage(
        targets=({"name": "t2", "build_device_provenance": "MEASURED"},))
    assert got["verdict"] == "UNPAIRED_BUILD_DEVICE_FIELDS"
    assert got["provenance_without_class"] == ["t2"]


def test_an_undeclared_build_device_value_is_its_own_verdict():
    """Separate from UNPAIRED, because the fixes differ: one is a forgotten
    field, the other a value nobody declared."""
    from src.analysis.canary import build_device_coverage
    got = build_device_coverage(
        targets=({"name": "t", "build_device_class": "h200",
                  "build_device_provenance": "seems_right"},))
    assert got["verdict"] == "UNDECLARED_BUILD_DEVICE_VALUE"
    assert got["undeclared_build_device_classes"] == ["h200"]
    assert got["undeclared_build_device_provenance"] == ["seems_right"]


def test_a_target_stating_NEITHER_field_is_COVERED_not_a_failure():
    """The field is OPTIONAL by design -- inheriting the documented default is
    the correct state for every pre-BK target, and reporting it as a defect
    would make the check one that can only fail (gotcha 25)."""
    from src.analysis.canary import build_device_coverage
    got = build_device_coverage(targets=({"name": "old"},))
    assert got["verdict"] == "COVERED"
    assert got["targets_inheriting_the_default"] == ["old"]
    assert got["targets_stating_their_own"] == []


def test_INFERRED_is_a_DIFFERENT_value_from_MEASURED():
    """Part 3 of the ratified change. All six additions are inferred from
    producing job logs; recording that as MEASURED would make it
    indistinguishable from a device the artifact recorded about itself."""
    from src.analysis.canary import BUILD_DEVICE_PROVENANCE
    assert set(BUILD_DEVICE_PROVENANCE) == {
        "MEASURED", "INFERRED_FROM_PRODUCING_JOB_LOGS"}
    inferred = [t for t in TARGETS if t["arm"] == "build_inputs"]
    assert all(t["build_device_provenance"] ==
               "INFERRED_FROM_PRODUCING_JOB_LOGS" for t in inferred)
    # NOTHING IS BACK-FILLED: no pre-BK target acquires a provenance flag.
    #
    # **THE PROXY WAS REPLACED AT BM, AND THE INTENT IS UNCHANGED.** This read
    # `for t in TARGETS if t["arm"] != "build_inputs"`, which was EXACT while
    # `build_inputs` was the only arm carrying the flag and silently became an
    # over-approximation the moment a later arm carried one legitimately.
    # "Pre-BK" is now named by NAME, which is what the comment always meant and
    # is stable under further additions. A post-BK target stating its own build
    # device is an ADDITION, not a back-fill: back-filling is re-describing an
    # artifact that predates the key, and none of the twelve below is touched.
    PRE_BK = {
        "rf_marginal_n19", "rf_counts_n19", "cv_ci_hardbin", "m3_degenerate",
        "m4_c1_all_seeds", "rf_mixture_n19",
        "rf_marginal_n39", "rf_counts_n39", "rf_mixture_n39",
        "alignment_predictor",
        "null_pooled_raw_digests", "null_pooled_reductions",
    }
    # The names are written out, so a rename or a deletion fails here (BB5b).
    assert PRE_BK <= {t["name"] for t in TARGETS}
    assert all("build_device_provenance" not in t
               for t in TARGETS if t["name"] in PRE_BK)
    assert all("build_device_class" not in t
               for t in TARGETS if t["name"] in PRE_BK)


def test_the_manifest_carries_the_build_device_coverage_and_a_schema_version(
        tmp_path):
    """A reader of `baseline.json` after a DIFFERS verdict must be able to see
    which targets state their own build device without reading the module."""
    from src.analysis.canary import MANIFEST_SCHEMA_VERSION
    r = _root(tmp_path)
    man = capture(r, REPO, None)
    assert man["identity"]["schema_version"] == MANIFEST_SCHEMA_VERSION == 2
    assert man["identity"]["build_device_coverage"]["verdict"] == "COVERED"
    assert "BK" in man["identity"]["amendments"]


# -------------------------------------------- describe over a GROUPED h5

def test_describe_walks_h5_GROUPS_and_not_only_top_level_datasets(tmp_path):
    """THE CHANGE HAD TO HAPPEN OR THREE OF THE SIX COULD NOT BE DESCRIBED AT
    ALL. `{k: hf[k].shape for k in hf}` raises AttributeError on a Group, and
    both `feature_ap*` artifacts nest under `/pooled` and `/experimental`
    while the label store nests under `/concepts`. This fixture reproduces
    that structure; without the recursive walk it raises rather than fails."""
    import h5py
    p = tmp_path / "grouped.h5"
    with h5py.File(p, "w") as hf:
        g = hf.create_group("pooled")
        g.create_dataset("ap", data=np.zeros((2, 3, 4), dtype=np.float32))
        g.create_dataset("concepts", data=np.zeros((4,), dtype=np.int32))
        hf.create_dataset("fire_counts", data=np.zeros((2, 3), dtype=np.int32))
    d = describe(p)
    assert d["datasets"] == {"pooled/ap": [2, 3, 4],
                             "pooled/concepts": [4],
                             "fire_counts": [2, 3]}


def test_describe_is_unchanged_for_a_FLAT_h5(tmp_path):
    """The other half, and the one that makes the change an addition: every h5
    target predating BK is flat, and a recursive walk returns exactly the flat
    dict for a flat file -- so no baselined target is re-described. Measured on
    the four real ones on disk before the change; asserted here on the
    property that made that true."""
    import h5py
    p = tmp_path / "flat.h5"
    with h5py.File(p, "w") as hf:
        hf.create_dataset("match_count", data=np.zeros((20, 8), dtype=np.int16))
        hf.create_dataset("seeds", data=np.arange(20))
    d = describe(p)
    assert d["datasets"] == {"match_count": [20, 8], "seeds": [20]}
    assert "n_datasets" not in d          # no new key on pre-existing targets


# ---------------------------------------------------------------------------
# BT-A3: RELOCATION IS A THIRD CATEGORY, AND IT MUST NOT BECOME A BACK DOOR.
#
# The fixtures below are built by hand from manifest dicts rather than by
# calling `capture`, so the expectation is WRITTEN DOWN and not computed from
# the code under test (gotcha 32b).
#
# Mutation kills RUN for this block:
#   * `set(moved) == {"path"}` -> `"path" in moved`  kills
#     test_a_path_AND_content_change_is_a_REBASELINE_not_a_relocation.
#   * dropping the --allow-relocation requirement in main() kills
#     test_assert_addition_only_REFUSES_a_relocation_without_allow_relocation.
#   * classifying a pure path move as `rebaselined` kills
#     test_a_pure_path_move_is_classified_relocated_with_the_hash_unchanged.
# ---------------------------------------------------------------------------

def _manifest(path, sha="a" * 64, n_leaves=None, bytes_=100, status="PRESENT"):
    """One-target manifest, written out field by field."""
    return {"targets": [{"name": "t", "arm": "build_inputs", "path": path,
                         "hash_scope": "whole_file", "sha256": sha,
                         "n_leaves": n_leaves, "bytes": bytes_,
                         "status": status,
                         "first_baselined_at": "deadbeef",
                         "captured_at": "cafe0000"}]}


def test_a_pure_path_move_is_classified_relocated_with_the_hash_unchanged():
    """The move BT actually performed: same bytes, different filesystem."""
    d = compare_manifests(_manifest("/scratch/x.h5"), _manifest("/project/x.h5"))
    assert d["rebaselined"] == []
    assert d["problems"] == []
    assert [r["name"] for r in d["relocated"]] == ["t"]
    r = d["relocated"][0]
    assert r["was"] == "/scratch/x.h5" and r["now"] == "/project/x.h5"
    assert r["sha256_unchanged"] == "a" * 64
    assert d["verdict"] == "RELOCATION_ONLY"


def test_a_path_AND_content_change_is_a_REBASELINE_not_a_relocation():
    """THE DISCRIMINATING CASE. A rewrite that also moved must never reach the
    relocation branch -- that is the whole failure the branch could enable."""
    d = compare_manifests(_manifest("/scratch/x.h5", sha="a" * 64),
                          _manifest("/project/x.h5", sha="b" * 64))
    assert d["relocated"] == []
    assert [r["name"] for r in d["rebaselined"]] == ["t"]
    assert sorted(d["rebaselined"][0]["fields"]) == ["path", "sha256"]
    assert d["problems"] and d["verdict"] == "NOT_AN_ADDITION_ONLY_CHANGE"


def test_a_content_change_at_the_SAME_path_is_still_a_rebaseline():
    """The pre-BT behaviour is unchanged where nothing moved."""
    d = compare_manifests(_manifest("/p/x.h5", sha="a" * 64),
                          _manifest("/p/x.h5", sha="b" * 64))
    assert d["relocated"] == [] and d["problems"]
    assert d["verdict"] == "NOT_AN_ADDITION_ONLY_CHANGE"


def test_a_move_that_also_changed_BYTES_is_refused():
    """`bytes` is a content field too; sha256 is not the only guard."""
    d = compare_manifests(_manifest("/s/x.h5", bytes_=100),
                          _manifest("/p/x.h5", bytes_=101))
    assert d["relocated"] == [] and d["problems"]


def test_CONTENT_FIELDS_is_BASELINE_FIELDS_minus_path():
    """DERIVED, so a field added to the baseline cannot be silently left out
    of the relocation conjunction (gotcha 16's fixture-omits-a-field shape)."""
    from src.analysis.canary import BASELINE_FIELDS, CONTENT_FIELDS
    assert set(CONTENT_FIELDS) == set(BASELINE_FIELDS) - {"path"}
    assert "path" not in CONTENT_FIELDS


def test_no_change_at_all_is_still_NO_CHANGE():
    """The total-branch check: every combination of added/relocated/problems
    has a verdict, so a completed comparison is never reported as a failure
    (gotcha 32c)."""
    d = compare_manifests(_manifest("/p/x.h5"), _manifest("/p/x.h5"))
    assert d["verdict"] == "NO_CHANGE"
    assert d["unchanged"] == ["t"] and d["relocated"] == []


def test_addition_and_relocation_together_get_their_own_verdict():
    prev = _manifest("/s/x.h5")
    cur = _manifest("/p/x.h5")
    cur["targets"].append({**prev["targets"][0], "name": "t2",
                           "path": "/p/y.h5"})
    d = compare_manifests(prev, cur)
    assert d["verdict"] == "ADDITION_AND_RELOCATION"
    assert [a["name"] for a in d["added"]] == ["t2"]
    assert [r["name"] for r in d["relocated"]] == ["t"]


def _relocated_root(tmp_path):
    """Root A, and a BYTE-IDENTICAL copy at root B. `copytree` is what makes
    the copy identical by construction rather than by hope -- rewriting the
    fixtures at B would test h5py's determinism, not the relocation branch."""
    import shutil
    a = _root(tmp_path)
    b = tmp_path / "relocated"
    shutil.copytree(a, b)
    return a, b


def test_assert_addition_only_REFUSES_a_relocation_without_allow_relocation(
        tmp_path):
    """A move must be DECLARED at the call site. Without the flag the guard
    refuses, which is what keeps `--assert-addition-only` from quietly
    accepting a category it was not written for."""
    a, b = _relocated_root(tmp_path)
    m = tmp_path / "manifest.json"
    assert _run(["--root", a, "--manifest", m, "--capture",
                 "--repo", REPO]).returncode == 0
    out = _run(["--root", b, "--manifest", m, "--capture", "--repo", REPO,
                "--assert-addition-only"])
    assert out.returncode == 1, out.stdout
    assert "RELOCATION" in out.stdout
    assert "must be declared with --allow-relocation" in out.stdout


def test_allow_relocation_accepts_the_move_and_PRINTS_the_unchanged_hash(
        tmp_path):
    """With the flag the capture succeeds -- and the hash is printed next to
    the move, so a reader never takes 'only the path changed' on trust."""
    a, b = _relocated_root(tmp_path)
    m = tmp_path / "manifest.json"
    _run(["--root", a, "--manifest", m, "--capture", "--repo", REPO])
    out = _run(["--root", b, "--manifest", m, "--capture", "--repo", REPO,
                "--assert-addition-only", "--allow-relocation"])
    assert out.returncode == 0, out.stdout + out.stderr
    assert "RELOCATION_ONLY" in out.stdout
    assert "sha256 UNCHANGED" in out.stdout
    assert "RELOCATED" in out.stdout


def test_allow_relocation_does_NOT_rescue_a_content_change(tmp_path):
    """THE DISCRIMINATING CLI CASE. The flag permits a move; it must not
    permit a rewrite that happens to have moved."""
    a, b = _relocated_root(tmp_path)
    m = tmp_path / "manifest.json"
    _run(["--root", a, "--manifest", m, "--capture", "--repo", REPO])
    tgt = next(t for t in TARGETS if t["hash_scope"] == "whole_file"
               and t["path"].endswith(".json"))
    (b / tgt["path"]).write_text(json.dumps({"v": [123]}))
    out = _run(["--root", b, "--manifest", m, "--capture", "--repo", REPO,
                "--assert-addition-only", "--allow-relocation"])
    assert out.returncode == 1, out.stdout
    assert "REFUSING: this capture is not addition-only" in out.stdout
    assert "REBASELINED" in out.stdout
