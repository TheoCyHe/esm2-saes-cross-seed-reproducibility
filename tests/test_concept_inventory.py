"""Tests for `concept_inventory` -- PREREGISTRATION amendment AY section 4.

THE FIXTURE THAT MATTERS is `test_nesting_check_sees_the_ACCESSION_ORDER_map`:
its wrong ordering has the SAME protein blocks, the SAME total row count and
the SAME instance count as the right one. That is the only way to show the
check discriminates a permutation, which neither a total nor a rank can
(gotcha 27, and the AO6 defect this column would otherwise reproduce).
"""
from __future__ import annotations

import csv
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.analysis import concept_inventory as inv


# --------------------------------------------------------- the block map

def test_block_map_argsorts_and_REPORTS_that_it_had_to():
    # Accession order: protein B starts at 0, A at 10, C at 30 -- but stored
    # as (A, B, C) sorted by accession, so the starts descend then ascend.
    start = np.array([10, 0, 30])
    nrow = np.array([20, 10, 5])
    b = inv.protein_block_map(start, nrow)
    assert b["stored_is_already_ascending"] is False
    assert b["n_descending_steps_in_stored_order"] == 1
    assert b["start"].tolist() == [0, 10, 30]
    assert b["n_rows"].tolist() == [10, 20, 5]


def test_block_map_reports_an_already_ascending_store_as_such():
    b = inv.protein_block_map(np.array([0, 10, 30]), np.array([10, 20, 5]))
    assert b["stored_is_already_ascending"] is True
    assert b["n_descending_steps_in_stored_order"] == 0


def test_protein_of_row_maps_each_row_into_its_own_block():
    b = inv.protein_block_map(np.array([10, 0, 30]), np.array([20, 10, 5]))
    rows = np.array([0, 9, 10, 29, 30, 34])
    assert inv.protein_of_row(rows, b).tolist() == [0, 0, 1, 1, 2, 2]


# ------------------------------------------------- the discriminating check

def _store(tmp_path: Path, start, nrow, instances: dict) -> Path:
    p = tmp_path / "labels.h5"
    with h5py.File(p, "w") as hf:
        hf.create_dataset("protein_sets/population_start",
                          data=np.asarray(start))
        hf.create_dataset("protein_sets/population_n_rows",
                          data=np.asarray(nrow))
        for c, (s, e) in instances.items():
            hf.create_dataset(f"instances/{c}/start_row", data=np.asarray(s))
            hf.create_dataset(f"instances/{c}/end_row", data=np.asarray(e))
            hf.create_group(f"concepts/{c}")
    return p


def test_nesting_check_passes_on_the_correct_row_order(tmp_path):
    # Blocks in row order: [0,10), [10,30), [30,35).
    lab = _store(tmp_path, [10, 0, 30], [20, 10, 5],
                 {"X": ([0, 12, 31], [5, 20, 34])})
    b = inv.protein_block_map(np.array([10, 0, 30]), np.array([20, 10, 5]))
    r = inv.assert_instances_nested(lab, b, ["X"])
    assert r["verdict"] == "ALL_NESTED"
    assert r["n_instances"] == 3 and r["share"] == pytest.approx(1.0)


def test_nesting_check_sees_the_ACCESSION_ORDER_map(tmp_path):
    """THE test. Feeding the stored (accession-order) arrays straight in --
    the AO6 defect -- must be caught. The block multiset, the total row count
    and the instance count are all IDENTICAL to the correct case, so only a
    positional check can tell them apart."""
    start, nrow = np.array([10, 0, 30]), np.array([20, 10, 5])
    lab = _store(tmp_path, start, nrow, {"X": ([0, 12, 31], [5, 20, 34])})

    wrong = {"order": np.arange(3), "start": start, "n_rows": nrow,
             "stored_is_already_ascending": False,
             "n_descending_steps_in_stored_order": 1}
    right = inv.protein_block_map(start, nrow)
    # The trap, made explicit: same blocks, same totals, different ORDER.
    assert sorted(wrong["start"].tolist()) == sorted(right["start"].tolist())
    assert wrong["n_rows"].sum() == right["n_rows"].sum()

    with pytest.raises(SystemExit) as e:
        inv.assert_instances_nested(lab, wrong, ["X"])
    assert "straddle" in str(e.value)


def test_nesting_check_catches_a_span_crossing_a_boundary(tmp_path):
    lab = _store(tmp_path, [0, 10], [10, 10], {"X": ([8], [12])})
    b = inv.protein_block_map(np.array([0, 10]), np.array([10, 10]))
    with pytest.raises(SystemExit):
        inv.assert_instances_nested(lab, b, ["X"])


# ---------------------------------------------------------- protein counts

def test_proteins_per_concept_counts_DISTINCT_proteins(tmp_path):
    lab = _store(tmp_path, [0, 10, 30], [10, 20, 5],
                 {"X": ([0, 3, 12], [2, 5, 15]),      # 2 distinct proteins
                  "Y": ([31], [33])})                  # 1
    b = inv.protein_block_map(np.array([0, 10, 30]), np.array([10, 20, 5]))
    got = inv.proteins_per_concept(lab, b, ["X", "Y", "ABSENT"])
    assert got["X"] == 2
    assert got["Y"] == 1
    # A concept with no instances group is None, NOT 0 -- "no record here" and
    # "annotated in zero proteins" are different facts.
    assert got["ABSENT"] is None


# ---------------------------------------------------------------- the rows

def test_build_rows_marks_pooled_parents_and_gives_a_reason():
    rows = inv.build_rows(
        ["BINDING", "BINDING:heme"], np.array([100, 40]),
        {"BINDING": 9, "BINDING:heme": 3},
        {"BINDING": 5, "BINDING:heme": 0}, {"BINDING:heme"}, {"BINDING"})
    parent, child = rows
    assert parent["in_screening_pool"] == "false"
    assert "excluded from the screening pool" in parent["excluded_reason"]
    assert parent["descriptor"] == ""
    assert parent["has_experimental_evidence"] == "true"
    assert parent["scored_experimental_stratum"] == "false"

    assert child["in_screening_pool"] == "true"
    assert child["excluded_reason"] == ""
    assert child["descriptor"] == "heme"
    assert child["has_experimental_evidence"] == "false"
    assert child["scored_experimental_stratum"] == "true"


def test_build_rows_leaves_an_absent_protein_count_BLANK_not_zero():
    rows = inv.build_rows(["X"], np.array([10]), {"X": None}, {"X": 0},
                          set(), set())
    assert rows[0]["n_proteins"] == ""


def test_assert_row_count_refuses_a_short_inventory():
    with pytest.raises(SystemExit) as e:
        inv.assert_row_count([{"a": 1}], 712)
    assert "does not cover the library" in str(e.value)


def test_write_csv_round_trips_every_column_and_a_unicode_descriptor(tmp_path):
    rows = inv.build_rows(
        ["ZN_FING:Phorbol-ester∕DAG-type"], np.array([21]),
        {"ZN_FING:Phorbol-ester∕DAG-type": 4},
        {"ZN_FING:Phorbol-ester∕DAG-type": 2}, set(), set())
    p = tmp_path / "inv.csv"
    inv.write_csv(rows, p)
    with p.open(encoding="utf-8-sig", newline="") as fh:
        got = list(csv.DictReader(fh))
    assert list(got[0].keys()) == inv.COLUMNS
    assert got[0]["descriptor"] == "Phorbol-ester∕DAG-type"
    assert got[0]["n_annotated_residues"] == "21"


def test_write_csv_uses_LF_so_a_rebuild_does_not_dirty_the_checkout(tmp_path):
    """The csv module writes CRLF on every platform and git stores LF, so the
    default rewrote all 713 lines on every rebuild and turned the checkout
    dirty -- which made concurrently-running jobs record git_dirty=True.

    Asserted on the RAW BYTES: reading the file back through csv would
    normalise the terminator and could not see this at all."""
    rows = inv.build_rows(["A", "B"], np.array([1, 2]), {"A": 1, "B": 1},
                          {"A": 0, "B": 0}, set(), set())
    p = tmp_path / "inv.csv"
    inv.write_csv(rows, p)
    raw = p.read_bytes()
    assert b"\r\n" not in raw
    assert raw.count(b"\n") == 3          # header + 2 rows

    # And writing twice is byte-identical, which is the property that keeps a
    # rebuild from showing up as a modification.
    first = p.read_bytes()
    inv.write_csv(rows, p)
    assert p.read_bytes() == first


def test_summarise_counts_the_groups():
    rows = inv.build_rows(
        ["A", "A:x", "B:y"], np.array([1, 2, 3]),
        {"A": 1, "A:x": 1, "B:y": 1}, {"A": 0, "A:x": 5, "B:y": 0},
        {"A:x"}, {"A"})
    s = inv.summarise(rows)
    assert s["n_rows"] == 3
    assert s["n_in_screening_pool"] == 2
    assert s["n_excluded_pooled_parents"] == 1
    assert s["n_with_experimental_evidence"] == 1
    assert s["n_scored_on_experimental_stratum"] == 1
    assert s["n_sub_typed"] == 2
    assert s["feature_keys"] == {"A": 2, "B": 1}


# ------------------------------------------------------------- entry point

def _ap(tmp_path: Path, concepts, npos, pool_n) -> Path:
    p = tmp_path / "ap.h5"
    with h5py.File(p, "w") as hf:
        hf.attrs["n_concepts"] = len(concepts)
        hf.attrs["argmax_pool_n"] = pool_n
        hf.attrs["library_sha"] = "deadbeef"
        hf.create_dataset("pooled/concepts",
                          data=np.array(concepts, dtype=h5py.string_dtype()))
        hf.create_dataset("pooled/n_positives", data=np.asarray(npos))
    return p


def _exp_ap(tmp_path: Path, concepts) -> Path:
    p = tmp_path / "exp.h5"
    with h5py.File(p, "w") as hf:
        hf.create_dataset("experimental/concepts",
                          data=np.array(concepts, dtype=h5py.string_dtype()))
    return p


def test_run_end_to_end(tmp_path):
    concepts = ["BINDING", "BINDING:heme", "DOMAIN:VWFA"]
    ap = _ap(tmp_path, concepts, [100, 40, 30], pool_n=2)
    lab = _store(tmp_path, [10, 0], [20, 10],
                 {"BINDING": ([0, 12], [3, 15]),
                  "BINDING:heme": ([1], [2]),
                  "DOMAIN:VWFA": ([11], [14])})
    with h5py.File(lab, "r+") as hf:
        hf["concepts/BINDING"].create_dataset("experimental",
                                              data=np.array([0, 1]))
    exp = _exp_ap(tmp_path, ["BINDING:heme"])

    r = inv.run(ap, lab, exp, tmp_path / "out.csv", tmp_path / "out.json")
    assert r["summary"]["n_rows"] == 3
    assert r["summary"]["n_in_screening_pool"] == 2
    assert r["excluded_pooled_parents"] == ["BINDING"]
    assert r["row_to_protein_map"]["nesting_check"]["verdict"] == "ALL_NESTED"
    with (tmp_path / "out.csv").open(encoding="utf-8-sig", newline="") as fh:
        got = list(csv.DictReader(fh))
    assert len(got) == 3
    assert got[1]["scored_experimental_stratum"] == "true"


def test_run_refuses_when_the_pool_arithmetic_disagrees(tmp_path):
    concepts = ["BINDING", "BINDING:heme"]
    ap = _ap(tmp_path, concepts, [1, 1], pool_n=2)      # should be 1
    lab = _store(tmp_path, [0], [10], {"BINDING": ([0], [1])})
    with pytest.raises(SystemExit) as e:
        inv.run(ap, lab, _exp_ap(tmp_path, []), tmp_path / "c.csv",
                tmp_path / "j.json")
    assert "argmax_pool_n" in str(e.value)


def test_run_refuses_when_the_concept_axis_contradicts_n_concepts(tmp_path):
    p = tmp_path / "ap.h5"
    with h5py.File(p, "w") as hf:
        hf.attrs["n_concepts"] = 99                     # the library, allegedly
        hf.attrs["argmax_pool_n"] = 1
        hf.create_dataset("pooled/concepts",
                          data=np.array(["A"], dtype=h5py.string_dtype()))
        hf.create_dataset("pooled/n_positives", data=np.array([1]))
    lab = _store(tmp_path, [0], [10], {"A": ([0], [1])})
    with pytest.raises(SystemExit) as e:
        inv.run(p, lab, _exp_ap(tmp_path, []), tmp_path / "c.csv",
                tmp_path / "j.json")
    assert "n_concepts" in str(e.value)
