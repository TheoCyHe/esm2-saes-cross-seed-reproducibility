"""Tests for P4's truncation-impact measurement (parse-side, no GPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

from src.annotations.measure_truncation_impact import main


DAT = """\
ID   TEST1_HUMAN             Reviewed;          10 AA.
AC   P11111;
OS   Homo sapiens (Human).
OC   Eukaryota; Metazoa; Chordata; Craniata.
FT   DOMAIN          2..9
FT                   /note="Some Domain"
SQ   SEQUENCE   10 AA;  1000 MW;  ABC CRC64;
     MKMKMKMKMK
//
ID   TEST2_HUMAN             Reviewed;           8 AA.
AC   P22222;
OS   Escherichia coli.
OC   Bacteria; Pseudomonadota.
FT   DOMAIN          1..8
FT                   /note="Some Domain"
SQ   SEQUENCE   8 AA;  800 MW;  DEF CRC64;
     AAAAAAAA
//
"""


def _setup(tmp_path, n_rows_p1: int, min_res: int, min_inst: int):
    (tmp_path / "t.dat").write_text(DAT)
    index = [
        {"accession": "sp|P11111|TEST1_HUMAN", "start": 0, "n_rows": n_rows_p1,
         "seq_len": 10, "truncated": n_rows_p1 < 10, "residue_ids": None},
        {"accession": "sp|P22222|TEST2_HUMAN", "start": n_rows_p1, "n_rows": 8,
         "seq_len": 8, "truncated": False, "residue_ids": None},
    ]
    with h5py.File(tmp_path / "eval.h5", "w") as hf:
        hf.create_dataset("activations",
                          data=np.zeros((n_rows_p1 + 8, 4), dtype=np.float16))
        hf.create_dataset("index_json", data=json.dumps(index))
    (tmp_path / "cfg.yaml").write_text(
        "annotations:\n"
        f"  retention_min_residues: {min_res}\n"
        f"  retention_min_instances: {min_inst}\n"
        "  subtype_note_types: [DOMAIN]\n"
    )


def test_truncated_and_untruncated_counts_diverge_only_for_the_truncated_protein(
    tmp_path, monkeypatch, capsys,
):
    """P11111's DOMAIN (residues 2..9, 8 residues) is cached at only 5 rows --
    residues 2-5 survive (4), residues 6-9 are lost. P22222's identical-named
    DOMAIN (residues 1..8, all 8) is NOT truncated -- must lose nothing. The
    pooled 'DOMAIN' concept combines both proteins' instances, so its loss
    fraction must reflect ONLY P11111's contribution, not P22222's."""
    _setup(tmp_path, n_rows_p1=5, min_res=3, min_inst=1)
    out = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", [
        "measure", "--config", str(tmp_path / "cfg.yaml"),
        "--dat", str(tmp_path / "t.dat"), "--cache", str(tmp_path / "eval.h5"),
        "--out", str(out)])
    main()
    d = json.loads(out.read_text())
    by_concept = {c["concept"]: c for c in d["concepts"]}

    sub = by_concept["DOMAIN:Some Domain"]
    assert sub["untrunc_residues"] == 8 + 8            # both instances, full
    assert sub["trunc_residues"] == 4 + 8               # P11111 loses 4, P22222 loses 0
    assert abs(sub["residue_loss_frac"] - 4 / 16) < 1e-9
    assert sub["untrunc_instances"] == 2
    assert sub["trunc_instances"] == 2                  # both instances still have >=1 surviving row

    assert d["n_contributing_proteins"] == 1
    assert d["composition_contributing"]["n_proteins"] == 1
    assert d["composition_contributing"]["lineage"] == {"Eukaryota": 1}


def test_no_truncation_means_zero_loss_everywhere(tmp_path, monkeypatch):
    """Sanity: with NEITHER protein truncated, untruncated and truncated
    counts must be bit-identical and nothing should flip the retention floor."""
    _setup(tmp_path, n_rows_p1=10, min_res=3, min_inst=1)     # P11111 NOT truncated
    out = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", [
        "measure", "--config", str(tmp_path / "cfg.yaml"),
        "--dat", str(tmp_path / "t.dat"), "--cache", str(tmp_path / "eval.h5"),
        "--out", str(out)])
    main()
    d = json.loads(out.read_text())
    assert d["n_contributing_proteins"] == 0
    assert d["n_concepts_flipping"] == 0
    for c in d["concepts"]:
        assert c["untrunc_residues"] == c["trunc_residues"]
        assert c["residue_loss_frac"] == 0.0


def test_floor_status_can_flip_between_truncated_and_untruncated_counting(tmp_path, monkeypatch):
    """A concept whose SURVIVING residues sit right at the floor edge must be
    reported as flipping -- this is the number the D8 straddling-rule decision
    is blocked on, so the flip detector itself must be exercised, not just
    assumed to work from the loss-fraction fields."""
    # P11111's DOMAIN spans residues 2..9 (8 residues); cache it down to 1 row
    # so only residue 1 is cached -- OUTSIDE the span, so P11111's instance
    # survives with ZERO residues and does not count as a surviving instance
    # either. min_instances huge forces the floor to hinge on residues alone.
    _setup(tmp_path, n_rows_p1=1, min_res=10, min_inst=1000)
    out = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", [
        "measure", "--config", str(tmp_path / "cfg.yaml"),
        "--dat", str(tmp_path / "t.dat"), "--cache", str(tmp_path / "eval.h5"),
        "--out", str(out)])
    main()
    d = json.loads(out.read_text())
    by_concept = {c["concept"]: c for c in d["concepts"]}
    sub = by_concept["DOMAIN:Some Domain"]
    # untrunc: 8 (P11111) + 8 (P22222) = 16 > 10 -> passes.
    # trunc: 0 (P11111, entirely outside the 1 cached row) + 8 (P22222) = 8,
    # not > 10 -> fails. P11111's instance also drops out of trunc_instances
    # (0 surviving rows), so only P22222's instance counts there.
    assert sub["untrunc_residues"] == 16
    assert sub["trunc_residues"] == 8
    assert sub["untrunc_instances"] == 2
    assert sub["trunc_instances"] == 1
    assert sub["passes_floor_untruncated"] is True
    assert sub["passes_floor_truncated"] is False
    assert sub["floor_status_flips"] is True
    assert d["n_concepts_flipping"] >= 1


# ------------------------------------------------ amendment CR, Row B --

def test_refuses_a_config_without_subtype_note_types(tmp_path, monkeypatch):
    """The private four-type fallback is gone: an absent key is a refusal that
    names the key, not a silent four-type run."""
    import pytest
    _setup(tmp_path, n_rows_p1=5, min_res=3, min_inst=1)
    (tmp_path / "cfg.yaml").write_text(
        "annotations:\n  retention_min_residues: 3\n  retention_min_instances: 1\n")
    out = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", [
        "measure", "--config", str(tmp_path / "cfg.yaml"),
        "--dat", str(tmp_path / "t.dat"), "--cache", str(tmp_path / "eval.h5"),
        "--out", str(out)])
    with pytest.raises(SystemExit) as e:
        main()
    assert "annotations.subtype_note_types" in str(e.value)
    assert not out.exists()
