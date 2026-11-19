"""Tests for the JSON-manifest migration (PREREGISTRATION amendment AK §6a).

A MIGRATION THAT CAN SILENTLY DO NOTHING IS NOT A VERIFIED MIGRATION (gotcha 14
extension 4). So the tests below check both directions: every intended key
present AFTER, and every superseded key absent -- and the dry run is checked to
write nothing, because a "dry run" that writes is the worst version of this
defect.

Mutation kills RUN:
  * making `find_retired` check only the top level -> kills
    test_a_nested_retired_key_is_found.
  * accepting an undeclared value instead of refusing -> kills
    test_an_undeclared_value_is_refused.
  * dropping the `attrs_before_AE.json` skip -> kills
    test_the_pre_migration_snapshot_is_skipped_by_design.
  * relaxing the acceptance criterion to ignore a changed unrelated leaf ->
    kills test_verify_flags_an_unrelated_leaf_change.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from migrations.ae_rename_attrs import KEY_RENAMES, VALUE_MAPS
from migrations.ak_rename_json_manifests import (
    SCHEMA_KEY,
    find_retired,
    flatten,
    map_value,
    rename_in_place,
    verify,
)

REPO = Path(__file__).resolve().parents[1]


def test_the_mapping_is_imported_not_restated():
    """One source of truth: a second copy of a value map whose purpose is
    removing a shared literal would drift."""
    import migrations.ae_rename_attrs as ae
    import migrations.ak_rename_json_manifests as ak
    assert ak.KEY_RENAMES is ae.KEY_RENAMES
    assert ak.VALUE_MAPS is ae.VALUE_MAPS


def test_find_retired_locates_a_top_level_key():
    hits = find_retired({"population": "untruncated", "n_rows": 3})
    assert hits == [("population", "population", "untruncated")]


def test_a_nested_retired_key_is_found():
    """THE DISCRIMINATING CASE. A top-level-only scan reports this clean while
    the retired key sits one level down -- AJ4's partial-function-as-total."""
    doc = {"identity": {"row_subset": "shared_experimental"},
           "runs": [{"population": "all"}]}
    paths = {p for p, _, _ in find_retired(doc)}
    assert paths == {"identity/row_subset", "runs[0]/population"}


def test_no_retired_key_gives_no_hits():
    assert find_retired({"truncation_policy": "untruncated",
                         "scored_row_set": "label_store"}) == []


def test_map_value_follows_the_declared_mapping():
    assert map_value("population", "all") == "with_truncated"
    assert map_value("population", "untruncated") == "untruncated"
    assert map_value("row_subset", "all") == "full_eval"
    assert map_value("row_subset", "population") == "label_store"
    assert map_value("row_subset", "shared_experimental") == "shared_experimental"


def test_an_undeclared_value_is_refused():
    with pytest.raises(SystemExit, match="not a declared value"):
        map_value("population", "some_new_policy")
    with pytest.raises(SystemExit):
        map_value("row_subset", None)


def test_the_two_value_sets_stay_disjoint_after_mapping():
    """AE1's invariant, re-asserted at this migration's own call site: no new
    value equals any name, and the two value sets share no literal."""
    names = set(KEY_RENAMES.values())
    pop = set(VALUE_MAPS["population"].values())
    row = set(VALUE_MAPS["row_subset"].values())
    assert not (pop | row) & names
    assert not pop & row


def test_rename_preserves_key_order_and_everything_else():
    doc = {"a": 1, "population": "untruncated", "z": {"b": [1, 2]}}
    changed: list[str] = []
    out = rename_in_place(copy.deepcopy(doc), changed)
    assert list(out) == ["a", "truncation_policy", "z"]
    assert changed == ["population"]
    assert out["a"] == 1 and out["z"] == {"b": [1, 2]}
    assert "population" not in out


def test_verify_passes_on_a_clean_rename():
    before = {"a": 1, "population": "untruncated"}
    changed: list[str] = []
    after = rename_in_place(copy.deepcopy(before), changed)
    after[SCHEMA_KEY] = "AK:deadbeef"
    assert verify(before, after, changed, "AK:deadbeef") == []


def test_verify_flags_an_unrelated_leaf_change():
    before = {"a": 1, "population": "untruncated"}
    changed: list[str] = []
    after = rename_in_place(copy.deepcopy(before), changed)
    after["a"] = 2
    after[SCHEMA_KEY] = "AK:deadbeef"
    problems = verify(before, after, changed, "AK:deadbeef")
    assert any("CHANGED" in p for p in problems), problems


def test_verify_flags_a_surviving_retired_key():
    before = {"population": "untruncated"}
    changed: list[str] = []
    after = rename_in_place(copy.deepcopy(before), changed)
    after["population"] = "untruncated"          # the silent no-op case
    after[SCHEMA_KEY] = "AK:deadbeef"
    problems = verify(before, after, changed, "AK:deadbeef")
    assert any("survives" in p for p in problems), problems


def test_verify_flags_a_missing_schema_version():
    before = {"population": "untruncated"}
    changed: list[str] = []
    after = rename_in_place(copy.deepcopy(before), changed)
    problems = verify(before, after, changed, "AK:deadbeef")
    assert any(SCHEMA_KEY in p for p in problems), problems


def test_flatten_reaches_list_elements():
    f = flatten({"a": [{"b": 1}, 2]})
    assert f == {"a[0]/b": 1, "a[1]": 2}


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------

def _run(args):
    return subprocess.run(
        [sys.executable, "-m", "migrations.ak_rename_json_manifests",
         *map(str, args)],
        capture_output=True, text=True, cwd=REPO)


def _fixture(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    (root / "labels.json").write_text(json.dumps(
        {"n_rows": 15972325, "population": "untruncated", "keep": [1, 2]},
        indent=2))
    (root / "ap.json").write_text(json.dumps(
        {"row_subset": "shared_experimental", "n_features": 10240}, indent=2))
    (root / "clean.json").write_text(json.dumps(
        {"truncation_policy": "untruncated"}, indent=2))
    (root / "attrs_before_AE.json").write_text(json.dumps(
        {"artifacts": {"x.h5": {"attrs": {"population": "all"}}}}, indent=2))
    (root / "notjson.txt").write_text("population")


def test_dry_run_writes_nothing(tmp_path):
    root = tmp_path / "analysis"
    _fixture(root)
    snapshot = {p.name: p.read_bytes() for p in root.iterdir()}
    r = _run(["--root", root, "--repo", REPO])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "DRY RUN" in r.stdout
    assert {p.name: p.read_bytes() for p in root.iterdir()} == snapshot


def test_apply_migrates_both_keys_and_verifies(tmp_path):
    root = tmp_path / "analysis"
    _fixture(root)
    report = tmp_path / "report.json"
    r = _run(["--root", root, "--repo", REPO, "--apply", "--report", report])
    assert r.returncode == 0, r.stdout + r.stderr

    lab = json.loads((root / "labels.json").read_text())
    assert lab["truncation_policy"] == "untruncated"
    assert "population" not in lab
    assert lab["n_rows"] == 15972325 and lab["keep"] == [1, 2]
    assert lab[SCHEMA_KEY].startswith("AK:")

    apd = json.loads((root / "ap.json").read_text())
    assert apd["scored_row_set"] == "shared_experimental"
    assert "row_subset" not in apd

    # BOTH DIRECTIONS (gotcha 14 extension 4): intended keys present AND
    # superseded keys absent, asserted on the raw bytes rather than on a
    # parsed view, because the audit downstream is a byte grep.
    for name in ("labels.json", "ap.json"):
        raw = (root / name).read_text()
        assert "population" not in raw or "truncation_policy" in raw
        assert '"population"' not in raw
        assert '"row_subset"' not in raw

    rep = json.loads(report.read_text())
    assert all(v == [] for v in rep["results"].values())
    assert len(rep["results"]) == 2


def test_a_clean_file_is_left_untouched(tmp_path):
    root = tmp_path / "analysis"
    _fixture(root)
    before = (root / "clean.json").read_bytes()
    r = _run(["--root", root, "--repo", REPO, "--apply"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert (root / "clean.json").read_bytes() == before


def test_the_pre_migration_snapshot_is_skipped_by_design(tmp_path):
    root = tmp_path / "analysis"
    _fixture(root)
    before = (root / "attrs_before_AE.json").read_bytes()
    r = _run(["--root", root, "--repo", REPO, "--apply"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SKIPPED BY DESIGN" in r.stdout
    assert (root / "attrs_before_AE.json").read_bytes() == before


def test_a_second_apply_is_a_no_op(tmp_path):
    """Idempotence, so re-running the pass cannot corrupt an already-migrated
    manifest -- and so 'nothing to do' is distinguishable from 'did nothing'."""
    root = tmp_path / "analysis"
    _fixture(root)
    assert _run(["--root", root, "--repo", REPO, "--apply"]).returncode == 0
    after_first = {p.name: p.read_bytes() for p in root.iterdir()}
    r = _run(["--root", root, "--repo", REPO, "--apply"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0 JSON manifests carry a retired key" in r.stdout
    assert {p.name: p.read_bytes() for p in root.iterdir()} == after_first


def test_an_undeclared_value_on_disk_stops_the_pass(tmp_path):
    root = tmp_path / "analysis"
    root.mkdir(parents=True)
    (root / "weird.json").write_text(json.dumps({"population": "half"}))
    r = _run(["--root", root, "--repo", REPO, "--apply"])
    assert r.returncode != 0
    assert "REFUSING" in (r.stdout + r.stderr)
    # And it must not have written a partial result.
    assert json.loads((root / "weird.json").read_text()) == {"population": "half"}


def test_the_audit_sees_zero_after_the_pass(tmp_path):
    """THE MIGRATION AND THE AUDIT VALIDATE EACH OTHER. The audit must report
    hits BEFORE and zero AFTER on the same tree -- an audit that reports zero
    on unmigrated data is not an audit."""
    root = tmp_path / "analysis"
    _fixture(root)
    (root / "notjson.txt").unlink()
    (root / "attrs_before_AE.json").unlink()

    def audit():
        return subprocess.run(
            [sys.executable, "-m", "src.analysis.audit_retired_literals",
             "--root", str(root)],
            capture_output=True, text=True, cwd=REPO)

    pre = audit()
    assert pre.returncode == 1, pre.stdout
    assert "un-allowlisted hits" in pre.stdout

    assert _run(["--root", root, "--repo", REPO, "--apply"]).returncode == 0

    post = audit()
    assert post.returncode == 0, post.stdout
    assert "ZERO live un-allowlisted hits" in post.stdout
