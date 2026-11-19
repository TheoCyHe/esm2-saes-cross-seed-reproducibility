"""
Tests for src/embeddings/shard_fasta.py.

Two properties matter. (1) The union of shards must equal the input exactly --
no sequence lost or duplicated, or the cache silently trains on the wrong
corpus. (2) The partition must be deterministic and residue-balanced, because
shard count + assignment are pinned inputs to reproducibility (they set the
loader's block layout). Both are asserted so a regression trips them.
"""

from __future__ import annotations

import json
import sys

import pytest

from src.data.split_train_eval import iter_fasta_records, seq_residues
from src.embeddings.shard_fasta import SHARD_ID_WIDTH, balance_shards, main


def test_balance_shards_covers_every_record_once():
    # residues chosen so a naive round-robin would imbalance
    recs = [(r, f"acc{i}") for i, r in enumerate([100, 90, 80, 70, 60, 50, 10, 5])]
    assignment, shard_res = balance_shards(recs, n_shards=3)
    flat = [i for shard in assignment for i in shard]
    assert sorted(flat) == list(range(len(recs)))        # each index exactly once
    assert sum(shard_res) == sum(r for r, _ in recs)


def test_balance_shards_is_actually_balanced():
    recs = [(r, f"a{i}") for i, r in enumerate(range(1, 101))]  # 1..100
    _, shard_res = balance_shards(recs, n_shards=4)
    total = sum(shard_res)
    # greedy LPT on 1..100 into 4 should be near-perfect
    assert max(shard_res) - min(shard_res) <= max(recs)[0]     # within one item
    assert abs(max(shard_res) - total / 4) / (total / 4) < 0.02


def test_balance_shards_deterministic():
    recs = [(r, f"a{i}") for i, r in enumerate([5, 5, 5, 5, 5, 5])]  # all ties
    a1, r1 = balance_shards(recs, 3)
    a2, r2 = balance_shards(recs, 3)
    assert a1 == a2 and r1 == r2       # ties broken by (residues, key) then index


def test_balance_shards_rejects_too_many_shards():
    with pytest.raises(ValueError):
        balance_shards([(1, "a"), (2, "b")], n_shards=3)   # more shards than seqs
    with pytest.raises(ValueError):
        balance_shards([(1, "a")], n_shards=0)


def _write_fasta(path, n, base_len=40):
    """n records with distinct lengths (so residues differ) and unique accs."""
    accs = []
    with open(path, "w") as fh:
        for i in range(n):
            acc = f"P{i:04d}"
            accs.append(acc)
            fh.write(f">sp|{acc}|NAME_{i} desc\n" + "M" * (base_len + i) + "\n")
    return accs


def test_main_partitions_fasta_without_loss(tmp_path, monkeypatch):
    fasta = tmp_path / "train.fasta"
    accs = _write_fasta(fasta, 50)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model:\n  max_len: 1022\nruntime:\n  cache_shards: 6\n")
    out = tmp_path / "shards"
    monkeypatch.setattr(sys, "argv", [
        "shard", "--config", str(cfg), "--fasta", str(fasta), "--out-dir", str(out),
    ])
    main()

    manifest = json.loads((out / "shards.json").read_text())
    assert manifest["n_shards"] == 6
    assert manifest["n_sequences"] == 50

    # union of the shard FASTAs == the input, each sequence exactly once
    seen = []
    for i in range(6):
        shard = out / f"shard_{i:0{SHARD_ID_WIDTH}d}.fasta"
        seen += [acc for acc, _ in iter_fasta_records(shard)]
    assert sorted(seen) == sorted(accs)
    assert len(seen) == len(set(seen))


def test_main_residue_counts_match_stored_definition(tmp_path, monkeypatch):
    """The manifest's per-shard residues must equal the capped residues actually
    in each shard FASTA -- the number cache.py will store."""
    fasta = tmp_path / "train.fasta"
    _write_fasta(fasta, 30, base_len=1000)      # some will exceed max_len 1022
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model:\n  max_len: 1022\nruntime:\n  cache_shards: 4\n")
    out = tmp_path / "shards"
    monkeypatch.setattr(sys, "argv", [
        "shard", "--config", str(cfg), "--fasta", str(fasta),
        "--out-dir", str(out), "--max-len", "1022",
    ])
    main()
    manifest = json.loads((out / "shards.json").read_text())
    for i, claimed in enumerate(manifest["shard_residues"]):
        shard = out / f"shard_{i:0{SHARD_ID_WIDTH}d}.fasta"
        actual = sum(seq_residues(text, 1022)
                     for _, text in iter_fasta_records(shard))
        assert actual == claimed
    assert sum(manifest["shard_residues"]) == manifest["total_residues_capped"]


# ------------------------------------------------ amendment CR, Row A --

def test_max_len_default_follows_the_config_not_a_literal(tmp_path, monkeypatch):
    """Config cap 500 (which the retired literal 1022 was not), no flag: the
    manifest says 500 and the balance is on residues capped at 500."""
    fasta = tmp_path / "train.fasta"
    _write_fasta(fasta, 30, base_len=480)       # lengths 480..509 straddle 500
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model:\n  max_len: 500\nruntime:\n  cache_shards: 3\n")
    out = tmp_path / "shards"
    monkeypatch.setattr(sys, "argv", [
        "shard", "--config", str(cfg), "--fasta", str(fasta), "--out-dir", str(out),
    ])
    main()
    manifest = json.loads((out / "shards.json").read_text())
    assert manifest["max_len"] == 500
    expected = sum(min(480 + i, 500) for i in range(30))
    assert manifest["total_residues_capped"] == expected
    assert sum(manifest["shard_residues"]) == expected


def test_a_disagreeing_max_len_is_refused(tmp_path, monkeypatch):
    fasta = tmp_path / "train.fasta"
    _write_fasta(fasta, 5)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model:\n  max_len: 1022\nruntime:\n  cache_shards: 2\n")
    out = tmp_path / "shards"
    monkeypatch.setattr(sys, "argv", [
        "shard", "--config", str(cfg), "--fasta", str(fasta),
        "--out-dir", str(out), "--max-len", "500",
    ])
    with pytest.raises(SystemExit) as e:
        main()
    assert "500" in str(e.value) and "1022" in str(e.value)
    assert not out.exists()
