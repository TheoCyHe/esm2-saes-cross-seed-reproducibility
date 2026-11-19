"""The completeness audit must be SEEN to fail (gotcha 13).

An audit that reports "zero hits" is worthless unless it can report otherwise,
and this one exists precisely because amendment AE's predecessor reported a
completeness claim that was false when made.
"""
from __future__ import annotations

import pytest

from src.analysis.audit_retired_literals import main, scan_file


def test_a_retired_literal_in_ANY_file_type_is_found(tmp_path):
    """It does not know what an artifact is, which is what makes it total.
    A bare `.bin` with no schema must still be caught."""
    p = tmp_path / "something.bin"
    p.write_bytes(b"\x89HDF\r\n" + b"row_subset" + b"\x00\x01population\x00")
    hits = scan_file(p)
    names = sorted({h[0] for h in hits})
    assert names == ["population", "row_subset"]


def test_the_retained_DATASET_prefixes_are_allowlisted_not_flagged(tmp_path):
    """`population_start` / `_n_rows` / `_accessions` are deliberately retained
    (AE: datasets are data, not labels). Flagging them would make the audit
    fail on correct data, which is how an audit gets disabled rather than
    fixed."""
    p = tmp_path / "labels.h5"
    p.write_bytes(b"protein_sets/population_start\x00population_n_rows\x00"
                  b"population_accessions")
    assert scan_file(p) == []


def test_a_bare_population_NEXT_TO_an_allowlisted_one_is_still_found(tmp_path):
    """A legitimate `population_start` must not hide a bare `population`
    beside it."""
    p = tmp_path / "mixed.h5"
    p.write_bytes(b"population_start\x00population\x00")
    hits = scan_file(p)
    assert [h[0] for h in hits] == ["population"]


def test_blanking_an_allowlisted_context_cannot_JOIN_fragments_into_a_hit(tmp_path):
    """WHY THE ALLOWLIST BLANKS RATHER THAN DELETES -- and mutation testing is
    what forced this test to exist.

    The test above originally claimed that blanking stops a legitimate context
    from MASKING a bare hit. That claim was wrong: deletion cannot mask
    anything, because it cannot break up a literal that is already contiguous.
    So a mutation swapping the null-fill for an empty string SURVIVED -- the
    test asserted a property the code does not provide.

    The property it DOES provide is the opposite direction. Deletion splices the
    surrounding bytes together and can FABRICATE a literal that was never in the
    file: here `popu` + `population_start` + `lation` deletes down to
    `population`. That is a false positive on clean data, which is how an audit
    gets switched off rather than fixed. Blanking preserves length, so offsets
    stay true and nothing is spliced.
    """
    p = tmp_path / "splice.h5"
    p.write_bytes(b"popu" + b"population_start" + b"lation")
    assert scan_file(p) == []


def test_main_EXITS_NONZERO_on_a_hit(tmp_path, monkeypatch, capsys):
    """A validator whose failure cannot propagate is worse than none
    (gotcha 13 extension 1). The exit path is checked, not just the verdict."""
    import sys
    (tmp_path / "dirty.h5").write_bytes(b"row_subset")
    monkeypatch.setattr(sys, "argv",
                        ["audit", "--root", str(tmp_path)])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1
    assert "un-allowlisted hits" in capsys.readouterr().out


def test_main_EXITS_ZERO_on_a_clean_root(tmp_path, monkeypatch, capsys):
    import sys
    (tmp_path / "clean.h5").write_bytes(b"truncation_policy\x00scored_row_set")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "also_clean.h5").write_bytes(b"population_start")
    monkeypatch.setattr(sys, "argv", ["audit", "--root", str(tmp_path)])
    main()                                   # must NOT raise
    assert "ZERO live un-allowlisted hits" in capsys.readouterr().out


def test_it_recurses_so_a_SUBDIRECTORY_cannot_hide(tmp_path, monkeypatch):
    """The AE failure in one line: the missed location was a subdirectory
    nobody listed."""
    import sys
    deep = tmp_path / "row_index_cache" / "nested"
    deep.mkdir(parents=True)
    (deep / "idx.h5").write_bytes(b"row_subset")
    monkeypatch.setattr(sys, "argv", ["audit", "--root", str(tmp_path)])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1


def test_a_literal_STRADDLING_a_chunk_boundary_is_still_found(tmp_path,
                                                              monkeypatch):
    """Streaming introduced a boundary, and a boundary is somewhere a literal
    can hide. With a tiny CHUNK the retired literal is deliberately placed
    across it."""
    import src.analysis.audit_retired_literals as mod
    monkeypatch.setattr(mod, "CHUNK", 8)
    p = tmp_path / "straddle.h5"
    # 'row_subset' starts at byte 5, so with CHUNK=8 it spans the boundary
    p.write_bytes(b"AAAAA" + b"row_subset" + b"BBBBB")
    hits = mod.scan_file(p)
    assert [h[0] for h in hits] == ["row_subset"]
    assert hits[0][1] == 5                      # absolute offset, not chunk-local


def test_a_hit_is_reported_ONCE_not_once_per_overlapping_chunk(tmp_path,
                                                               monkeypatch):
    """Overlap is what makes the boundary safe; reporting from the core is what
    stops it double-counting."""
    import src.analysis.audit_retired_literals as mod
    monkeypatch.setattr(mod, "CHUNK", 8)
    p = tmp_path / "once.h5"
    p.write_bytes(b"A" * 40 + b"row_subset" + b"B" * 40)
    hits = mod.scan_file(p)
    assert len(hits) == 1, f"expected exactly one hit, got {len(hits)}"


def test_an_allowlisted_context_STRADDLING_a_boundary_is_still_allowlisted(
        tmp_path, monkeypatch):
    """The mirror case: if the blanking missed a straddling context, the audit
    would fail on correct data."""
    import src.analysis.audit_retired_literals as mod
    monkeypatch.setattr(mod, "CHUNK", 8)
    p = tmp_path / "ok.h5"
    p.write_bytes(b"AAAAA" + b"population_start" + b"BBBBB")
    assert mod.scan_file(p) == []


def test_an_hdf5_TOMBSTONE_is_not_reported_as_a_survival(tmp_path):
    """HDF5 does not reclaim a deleted attribute's bytes, so after the AE
    migration the name string is still IN the file while h5py reports it
    absent. Measured on real data: 134 byte hits, 0 live attrs. Reporting
    those would make the audit fail on correct data."""
    import sys
    import h5py
    from src.analysis.audit_retired_literals import live_h5_attrs, main
    f = tmp_path / "migrated.h5"
    with h5py.File(f, "w") as hf:
        hf.attrs["population"] = "untruncated"      # write it...
    with h5py.File(f, "a") as hf:
        del hf.attrs["population"]                  # ...then delete it
        hf.attrs["truncation_policy"] = "untruncated"
    assert b"population" in f.read_bytes(), "fixture needs a real tombstone"
    assert live_h5_attrs(f) == set(), "h5py must report the attr absent"

    import pytest as _pt
    monkey = _pt.MonkeyPatch()
    monkey.setattr(sys, "argv", ["audit", "--root", str(tmp_path)])
    main()                                          # must NOT raise
    monkey.undo()


def test_a_LIVE_retired_attr_in_hdf5_is_still_a_failure(tmp_path, monkeypatch):
    """The other direction, so the cross-check cannot become a blanket excuse
    for every h5 hit."""
    import sys
    import h5py
    from src.analysis.audit_retired_literals import live_h5_attrs, main
    f = tmp_path / "unmigrated.h5"
    with h5py.File(f, "w") as hf:
        hf.attrs["row_subset"] = "population"
    assert live_h5_attrs(f) == {"row_subset"}
    monkeypatch.setattr(sys, "argv", ["audit", "--root", str(tmp_path)])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1


def test_a_NON_hdf5_file_is_judged_on_bytes_alone(tmp_path):
    """live_h5_attrs returns None for anything h5py cannot open, and the byte
    verdict must then stand -- otherwise a JSON or a .py would be excused."""
    from src.analysis.audit_retired_literals import live_h5_attrs, scan_file
    f = tmp_path / "manifest.json"
    f.write_text('{"row_subset": "shared_experimental"}')
    assert live_h5_attrs(f) is None
    assert [h[0] for h in scan_file(f)] == ["row_subset"]


def test_exclude_skips_a_path_and_says_how_many(tmp_path, monkeypatch, capsys):
    """Exclusions are a CALL-SITE policy, and the count is printed so a
    completeness claim carries its own exceptions rather than hiding them."""
    import sys
    from src.analysis.audit_retired_literals import main
    (tmp_path / "snapshot.json").write_bytes(b"row_subset")
    monkeypatch.setattr(sys, "argv",
                        ["audit", "--root", str(tmp_path), "--exclude", "*.json"])
    main()                                    # must NOT raise
    out = capsys.readouterr().out
    assert "1 excluded by --exclude" in out
    assert "ZERO live un-allowlisted hits" in out


def test_exclude_does_NOT_silence_a_non_matching_path(tmp_path, monkeypatch):
    import sys
    from src.analysis.audit_retired_literals import main
    (tmp_path / "real.h5x").write_bytes(b"row_subset")
    monkeypatch.setattr(sys, "argv",
                        ["audit", "--root", str(tmp_path), "--exclude", "*.json"])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1


# ---------------------------------------------------------------------------
# AL3: `n_proteins_population` -- the first allowlisted context where the
# retired literal is a SUFFIX. Surfaced by RUNNING the audit after the AK §6a
# migration, not by reading it.
# ---------------------------------------------------------------------------

def test_the_suffix_form_count_field_is_allowlisted(tmp_path):
    p = tmp_path / "labels.json"
    p.write_text('{"n_proteins_population": 51943, "n_length_mismatch": 0}')
    assert scan_file(p) == []


def test_a_bare_population_NEXT_TO_the_suffix_form_is_still_found(tmp_path):
    """The allowlist must not become a hiding place. A real retired key sitting
    beside the benign count field must still be reported."""
    p = tmp_path / "labels.json"
    p.write_text('{"n_proteins_population": 51943, "population": "untruncated"}')
    hits = scan_file(p)
    assert len(hits) == 1
    assert hits[0][0] == "population"


def test_blanking_the_suffix_form_cannot_FABRICATE_a_hit(tmp_path):
    """AJ4's mutation finding, re-checked for the new context: blanking splices
    neighbours, so it can invent a literal on clean data. Here the neighbours
    are quote characters, which cannot form `population`."""
    p = tmp_path / "x.json"
    p.write_text('{"popu": 1, "n_proteins_population": 2, "lation": 3}')
    assert scan_file(p) == []


def test_the_suffix_form_is_allowlisted_under_population_only(tmp_path):
    """It must not silence `row_subset`, which has NO benign context at all."""
    p = tmp_path / "x.json"
    p.write_text('{"n_proteins_population": 1, "row_subset": "all"}')
    hits = scan_file(p)
    assert [h[0] for h in hits] == ["row_subset"]


def test_the_margin_covers_the_longest_allowlisted_context():
    """`_MARGIN` must be at least the longest context, or a boundary-straddling
    allowlisted string is truncated and reported as a hit. Computed from the
    table rather than hardcoded, so adding a longer context cannot break it
    silently."""
    from src.analysis.audit_retired_literals import _MARGIN, ALLOWED, RETIRED
    longest = max(len(x) for x in list(RETIRED.values())
                  + [c for v in ALLOWED.values() for c in v])
    assert _MARGIN >= longest
    assert _MARGIN >= len(b"n_proteins_population")


# ---------------------------------------------------------------------------
# AM4: THE ALLOWLIST IS AN ASSERTION, SO ITS CONTENTS ARE PINNED.
#
# An allowlist in a completeness audit is the same construct as
# `config_audit.UNGOVERNED`: a category meaning "not checked", whose membership
# is an unverified claim. No test can verify a justification is CORRECT. What a
# test CAN do is make adding an entry impossible without editing this file, so
# the addition and its justification appear together in one diff.
#
# Mutation kills RUN: adding a fifth context, or moving one between literals,
# kills test_the_allowlist_is_exactly_these_four_contexts.
# ---------------------------------------------------------------------------

def test_the_allowlist_is_exactly_these_four_contexts():
    """Pinned exactly, not as a subset. Each entry's justification is in
    `audit_retired_literals.ALLOWED`'s inline comments:
      population_start / population_n_rows / population_accessions
          -- AE's deliberately RETAINED dataset prefixes ("datasets are data,
             not labels").
      n_proteins_population
          -- a JSON manifest COUNT field written by build_residue_labels; the
             first SUFFIX-form context (amendment AL3).
    `row_subset` is FULLY retired and has NO benign context."""
    from src.analysis.audit_retired_literals import ALLOWED
    assert set(ALLOWED) == {"population", "row_subset"}
    assert ALLOWED["population"] == (
        b"population_start",
        b"population_n_rows",
        b"population_accessions",
        b"n_proteins_population",
    )
    assert ALLOWED["row_subset"] == ()


def test_every_allowlisted_context_actually_contains_its_literal():
    """A context that does not contain the literal it is filed under would
    blank bytes for no reason and could only ever mask something."""
    from src.analysis.audit_retired_literals import ALLOWED, RETIRED
    for name, ctxs in ALLOWED.items():
        for c in ctxs:
            assert RETIRED[name] in c, (name, c)


def test_no_allowlisted_context_equals_its_literal():
    """A context equal to the literal would allowlist the literal itself and
    switch the audit off entirely."""
    from src.analysis.audit_retired_literals import ALLOWED, RETIRED
    for name, ctxs in ALLOWED.items():
        for c in ctxs:
            assert c != RETIRED[name], (name, c)


def test_the_standing_invocation_declares_its_exclusions_and_their_reasons():
    """THE DOMAIN STATEMENT IS CHECKABLE (amendment AM4).

    The audit's completeness claim is only as good as the exclusion list at the
    call site, so the call site is in git -- and this test pins the glob set AND
    requires each glob's distinguishing token to appear in a comment in the same
    file. Adding an exclusion without a reason fails here."""
    import re
    from pathlib import Path
    sb = Path(__file__).resolve().parents[1] / "slurm" / "audit_retired.sbatch"
    text = sb.read_text()
    globs = re.findall(r'--exclude "([^"]+)"', text)
    assert globs == [
        "*/migrations/attrs_before_AE.json",
        "*/adhoc/logs/*",
        "*/adhoc/*.py",
        "*/adhoc/*.sbatch",
        "*/adhoc/README.txt",
        "*/analysis/preflight_measure_*.out",
        "*/analysis/*.out",
    ], globs
    # The reasons block: every glob's distinguishing token must be explained.
    reasons = text.split("# Reasons, one per exclusion")[-1]
    for token in ("attrs_before_AE.json", "adhoc/logs", "adhoc/*.py",
                  "README.txt", "analysis/*.out"):
        assert token in reasons, token
    # And the fixed-point property must be stated, since it is the one
    # exclusion that cannot be removed even in principle.
    assert "fixed point" in text.lower()


# ---------------------------------------------------------------------------
# AN4: converting an assertion into a measurement beats justifying it.
#
# Mutation kills RUN:
#   * returning CONFIRMED regardless of the name set -> kills
#     test_an_allowlisted_context_with_no_matching_name_is_UNCONFIRMED.
#   * dropping the h5-artifact check from audit_exclusions -> kills
#     test_a_glob_that_swallows_an_h5_is_flagged.
# ---------------------------------------------------------------------------

def test_an_allowlisted_context_backed_by_a_real_dataset_is_CONFIRMED(tmp_path):
    """The tombstone move, applied to the byte allowlist: the hit is explained by
    an actual dataset of that name, so it is a MEASUREMENT."""
    import h5py
    import numpy as np
    from src.analysis.audit_retired_literals import verify_allowlisted_contexts
    p = tmp_path / "x.h5"
    with h5py.File(p, "w") as hf:
        hf.create_dataset("population_start", data=np.zeros(3))
    r = verify_allowlisted_contexts(p)
    assert r["status"] == "CHECKED" and r["kind"] == "h5"
    assert r["contexts"]["population_start"] == "CONFIRMED"
    assert r["n_unconfirmed"] == 0


def test_an_allowlisted_context_with_no_matching_name_is_UNCONFIRMED(tmp_path):
    """THE DISCRIMINATING CASE. The bytes are there and no dataset, group or attr
    of that name is -- so the allowlist is carrying it on an assertion, and the
    audit says so instead of implying it was checked."""
    import h5py
    import numpy as np
    from src.analysis.audit_retired_literals import verify_allowlisted_contexts
    p = tmp_path / "x.h5"
    with h5py.File(p, "w") as hf:
        d = hf.create_dataset("unrelated", data=np.zeros(3))
        d.attrs["note"] = "population_start appears only in this free text"
    r = verify_allowlisted_contexts(p)
    assert r["status"] == "CHECKED"
    assert r["contexts"]["population_start"] == "UNCONFIRMED"
    assert r["n_confirmed"] == 0


def test_a_json_key_backed_context_is_CONFIRMED(tmp_path):
    from src.analysis.audit_retired_literals import verify_allowlisted_contexts
    p = tmp_path / "m.json"
    p.write_text('{"n_proteins_population": 51943, "other": 1}')
    r = verify_allowlisted_contexts(p)
    assert r["kind"] == "json"
    assert r["contexts"]["n_proteins_population"] == "CONFIRMED"


def test_a_nested_json_key_is_still_found(tmp_path):
    from src.analysis.audit_retired_literals import verify_allowlisted_contexts
    p = tmp_path / "m.json"
    p.write_text('{"a": {"b": [{"n_proteins_population": 1}]}}')
    assert (verify_allowlisted_contexts(p)["contexts"]
            ["n_proteins_population"] == "CONFIRMED")


def test_a_file_with_no_allowlisted_context_is_reported_as_such(tmp_path):
    from src.analysis.audit_retired_literals import verify_allowlisted_contexts
    p = tmp_path / "x.json"
    p.write_text('{"clean": 1}')
    assert verify_allowlisted_contexts(p)["status"] == "NO_ALLOWLISTED_CONTEXT"


def test_a_glob_that_matches_nothing_is_reported_dead(tmp_path):
    from src.analysis.audit_retired_literals import audit_exclusions
    (tmp_path / "a.txt").write_text("x")
    r = audit_exclusions([tmp_path], ["*.nope"])
    assert r["dead_globs"] == ["*.nope"]
    assert r["verdict"] == "OK"          # dead is worth knowing, not a failure


def test_a_glob_that_swallows_an_h5_is_flagged(tmp_path):
    """THE DANGEROUS DIRECTION, and the one that IS checkable: an exclusion that
    hides an artifact defeats the audit rather than scoping it."""
    from src.analysis.audit_retired_literals import audit_exclusions
    import h5py
    import numpy as np
    d = tmp_path / "analysis"
    d.mkdir()
    with h5py.File(d / "real_artifact.h5", "w") as hf:
        hf.create_dataset("x", data=np.zeros(2))
    r = audit_exclusions([tmp_path], ["*/analysis/*.h5"])
    assert r["globs_excluding_h5"] == ["*/analysis/*.h5"]
    assert r["verdict"] == "EXCLUDES_AN_ARTIFACT"


def test_the_committed_exclusions_swallow_no_h5_in_this_repo():
    """Run the real glob list against the repo. It should exclude nothing that
    looks like an artifact -- and the repo has no .h5 committed, so this also
    documents that the check is exercised rather than vacuous."""
    from pathlib import Path
    import re
    from src.analysis.audit_retired_literals import audit_exclusions
    repo = Path(__file__).resolve().parents[1]
    globs = re.findall(r'--exclude "([^"]+)"',
                       (repo / "slurm" / "audit_retired.sbatch").read_text())
    r = audit_exclusions([repo / "src", repo / "tests"], globs)
    assert r["verdict"] == "OK"
    assert r["globs_excluding_h5"] == []


def test_a_missing_import_on_the_json_path_RAISES_rather_than_degrading(monkeypatch,
                                                                       tmp_path):
    """The narrowed except, made discriminable.

    A bare `except Exception` here swallowed a NameError (`json` was not
    imported) and every JSON file came back UNVERIFIABLE_FILE_TYPE -- the
    MEASUREMENT silently degraded into the ASSERTION it was replacing, which is
    the worst available outcome for a verification path. With the except narrowed,
    the same defect surfaces instead of hiding."""
    import src.analysis.audit_retired_literals as ar
    p = tmp_path / "m.json"
    p.write_text('{"n_proteins_population": 1}')
    monkeypatch.delattr(ar, "json")
    with pytest.raises(NameError):
        ar._json_keys(p)


def test_context_verification_does_NOT_load_the_whole_file(tmp_path, monkeypatch):
    """AN10. The first version called `path.read_bytes()` and would have been
    OOM-killed on the 23 GB and 25 GB `null_raw` files under --mem=8G -- the exact
    defect AJ4 fixed, reintroduced in the same module.

    Asserted by making `read_bytes` explode: if the function still calls it, this
    test fails. A size assertion would not discriminate, since a small fixture
    loads fine either way."""
    from pathlib import Path
    import src.analysis.audit_retired_literals as ar
    p = tmp_path / "m.json"
    p.write_text('{"n_proteins_population": 1}')

    def _boom(self, *a, **k):
        raise AssertionError("read_bytes() must not be used -- stream instead")
    monkeypatch.setattr(Path, "read_bytes", _boom)
    r = ar.verify_allowlisted_contexts(p)
    assert r["status"] == "CHECKED"
    assert r["contexts"]["n_proteins_population"] == "CONFIRMED"


def test_a_context_straddling_a_chunk_boundary_is_still_found(tmp_path,
                                                             monkeypatch):
    """Streaming introduces a boundary, so the overlap margin must cover it --
    the same property the main scan already guarantees."""
    import src.analysis.audit_retired_literals as ar
    monkeypatch.setattr(ar, "CHUNK", 64)
    p = tmp_path / "x.bin"
    ctx = b"n_proteins_population"
    # place the context so it spans the first boundary
    p.write_bytes(b"a" * (64 - 5) + ctx + b"b" * 40)
    r = ar.verify_allowlisted_contexts(p)
    assert r["status"] in {"UNVERIFIABLE_FILE_TYPE", "CHECKED"}
    assert "n_proteins_population" in (
        r.get("contexts") if isinstance(r.get("contexts"), list)
        else list(r.get("contexts", {})))
