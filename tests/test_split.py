"""
Tests for src/data/split_train_eval.py.

The property that matters is homology-disjointness: no cluster may be split
across train and eval, or the headline aligned-vs-orphan comparison leaks
homologs and manufactures the result (CLAUDE.md open question 1). The
integration test asserts it directly, and is written so a routing bug would
trip it rather than pass quietly.
"""

from __future__ import annotations

import json
import sys

import pytest

from src.data.split_train_eval import (
    assign_clusters,
    header_to_acc,
    iter_fasta_records,
    main,
    parse_clusters,
    seq_residues,
)


def test_header_to_acc_uniprot_and_fallback():
    assert header_to_acc("sp|Q6GZX4|001R_FRG3G Putative factor OS=...") == "Q6GZX4"
    assert header_to_acc("tr|A0A1|SOMENAME desc") == "A0A1"
    assert header_to_acc("plainid rest of header") == "plainid"


def test_seq_residues_caps_at_max_len():
    rec = ">sp|X|N\n" + "A" * 30 + "\n" + "A" * 30 + "\n"   # 60 residues
    assert seq_residues(rec, max_len=1022) == 60
    assert seq_residues(rec, max_len=50) == 50               # capped


def test_iter_fasta_records_multiline(tmp_path):
    fa = tmp_path / "s.fasta"
    fa.write_text(
        ">sp|P1|A_one desc\nMKMK\nAAAA\n"
        ">sp|P2|B_two\nGG\n"
    )
    recs = list(iter_fasta_records(fa))
    assert [acc for acc, _ in recs] == ["P1", "P2"]
    # the record text is verbatim and reusable, including all sequence lines
    assert recs[0][1] == ">sp|P1|A_one desc\nMKMK\nAAAA\n"
    assert seq_residues(recs[0][1], 1022) == 8


def test_assign_clusters_deterministic_disjoint_and_sized():
    reps = [f"c{i}" for i in range(100)]
    train, evl = assign_clusters(reps, eval_frac=0.10, seed=0)
    assert len(evl) == 10 and len(train) == 90
    assert train.isdisjoint(evl)
    assert train | evl == set(reps)
    # same seed -> identical partition; different seed -> different eval set
    assert assign_clusters(reps, 0.10, 0)[1] == evl
    assert assign_clusters(reps, 0.10, 1)[1] != evl


def test_assign_clusters_rejects_bad_fraction():
    with pytest.raises(ValueError):
        assign_clusters(["a", "b"], eval_frac=1.0, seed=0)


def _write_inputs(tmp_path, n_clusters=20, members_per=3):
    """Build a synthetic cluster tsv + matching FASTA. Cluster i has rep 'r{i}'
    and members 'r{i}', 'r{i}_m1', ... Accessions are unique across clusters."""
    tsv_lines, fasta_parts, all_accs = [], [], []
    for i in range(n_clusters):
        rep = f"r{i}"
        accs = [rep] + [f"r{i}_m{j}" for j in range(1, members_per)]
        for a in accs:
            tsv_lines.append(f"{rep}\t{a}")
            fasta_parts.append(f">sp|{a}|NAME_{a} desc\n" + "MKAA\n")
            all_accs.append(a)
    tsv = tmp_path / "clusters.tsv"
    tsv.write_text("\n".join(tsv_lines) + "\n")
    fasta = tmp_path / "src.fasta"
    fasta.write_text("".join(fasta_parts))
    return tsv, fasta, all_accs


def _run_split(tmp_path, monkeypatch, eval_frac=0.25, seed=0, policy="all"):
    tsv, fasta, all_accs = _write_inputs(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "model:\n  max_len: 1022\n"
        "data:\n"
        f"  eval_cluster_fraction: {eval_frac}\n"
        f"  split_seed: {seed}\n"
        f"  cluster_members: {policy}\n"
    )
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "split", "--config", str(cfg), "--fasta", str(fasta),
        "--clusters", str(tsv), "--out-dir", str(out),
    ])
    main()
    train_accs = [acc for acc, _ in iter_fasta_records(out / "train.fasta")]
    eval_accs = [acc for acc, _ in iter_fasta_records(out / "eval.fasta")]
    manifest = json.loads((out / "split.json").read_text())
    return train_accs, eval_accs, manifest, all_accs, tsv


def test_split_is_homology_disjoint(tmp_path, monkeypatch):
    train_accs, eval_accs, manifest, all_accs, tsv = _run_split(tmp_path, monkeypatch)
    train, evl = set(train_accs), set(eval_accs)

    # no accession appears in both, and nothing is duplicated within a split
    assert train.isdisjoint(evl)
    assert len(train_accs) == len(train)
    assert len(eval_accs) == len(evl)
    # 'all' keeps every clustered sequence
    assert train | evl == set(all_accs)

    # THE invariant: no cluster is split across train and eval.
    member_to_rep, _ = parse_clusters(tsv)
    eval_reps = set(manifest["eval_cluster_reps"])
    for acc in all_accs:
        in_eval = acc in evl
        assert in_eval == (member_to_rep[acc] in eval_reps), (
            f"{acc} routed to the wrong side of its cluster"
        )


def test_split_manifest_counts_match(tmp_path, monkeypatch):
    train_accs, eval_accs, manifest, _, _ = _run_split(tmp_path, monkeypatch)
    assert manifest["n_train_sequences"] == len(train_accs)
    assert manifest["n_eval_sequences"] == len(eval_accs)
    assert manifest["n_eval_clusters"] == len(manifest["eval_cluster_reps"])
    assert manifest["train_residues_capped"] == 4 * len(train_accs)  # "MKAA" = 4


def test_representatives_policy_keeps_only_reps(tmp_path, monkeypatch):
    train_accs, eval_accs, manifest, _, _ = _run_split(
        tmp_path, monkeypatch, policy="representatives")
    kept = set(train_accs) | set(eval_accs)
    # only rep accessions (no "_m" members) survive
    assert all("_m" not in a for a in kept)
    assert len(kept) == manifest["n_clusters"]


def test_unclustered_record_is_counted_not_misassigned(tmp_path, monkeypatch):
    tsv, fasta, _ = _write_inputs(tmp_path, n_clusters=5)
    # append a FASTA record with no cluster entry
    with open(fasta, "a") as fh:
        fh.write(">sp|ORPHAN|N desc\nMK\n")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model:\n  max_len: 1022\n"
                   "data:\n  eval_cluster_fraction: 0.2\n  split_seed: 0\n"
                   "  cluster_members: all\n")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "split", "--config", str(cfg), "--fasta", str(fasta),
        "--clusters", str(tsv), "--out-dir", str(out),
    ])
    main()
    manifest = json.loads((out / "split.json").read_text())
    assert manifest["n_unclustered_records"] == 1
    all_written = {acc for acc, _ in iter_fasta_records(out / "train.fasta")}
    all_written |= {acc for acc, _ in iter_fasta_records(out / "eval.fasta")}
    assert "ORPHAN" not in all_written        # not silently misassigned


# ------------------------------------------------ amendment CR, Row A --

def test_max_len_default_follows_the_config_not_a_literal(tmp_path, monkeypatch):
    """A config cap the retired literal did not have (500), no flag: the
    manifest reports 500 and residues are capped at 500."""
    tsv, fasta, _ = _write_inputs(tmp_path, n_clusters=4)
    with open(fasta, "a") as fh, open(tsv, "a") as th:
        fh.write(">sp|LONG1|N desc\n" + "M" * 900 + "\n")
        th.write("LONG1\tLONG1\n")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model:\n  max_len: 500\n"
                   "data:\n  eval_cluster_fraction: 0.2\n  split_seed: 0\n"
                   "  cluster_members: all\n")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "split", "--config", str(cfg), "--fasta", str(fasta),
        "--clusters", str(tsv), "--out-dir", str(out),
    ])
    main()
    manifest = json.loads((out / "split.json").read_text())
    assert manifest["max_len_for_report"] == 500
    # 4 clusters x 3 members x 4 residues ("MKAA") + the long one capped at 500
    assert (manifest["train_residues_capped"]
            + manifest["eval_residues_capped"]) == 12 * 4 + 500


def test_a_disagreeing_max_len_is_refused(tmp_path, monkeypatch):
    tsv, fasta, _ = _write_inputs(tmp_path, n_clusters=4)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model:\n  max_len: 1022\n"
                   "data:\n  eval_cluster_fraction: 0.2\n  split_seed: 0\n"
                   "  cluster_members: all\n")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "split", "--config", str(cfg), "--fasta", str(fasta),
        "--clusters", str(tsv), "--out-dir", str(out), "--max-len", "500",
    ])
    with pytest.raises(SystemExit) as e:
        main()
    assert "500" in str(e.value) and "1022" in str(e.value)
    assert not out.exists()
