"""Amendment CR §CR1 Row A -- `cluster_swissprot.py --max-len` reads
`model.max_len` and refuses a disagreeing CLI value.

`main()` is run offline: `download` skips when `uniprot_sprot.fasta` already
sits in the workdir, and the mmseqs step skips when the `_cluster.tsv` exists,
so no network and no mmseqs binary are needed."""
from __future__ import annotations

import sys

import pytest

from src.data.cluster_swissprot import main


def _workdir(tmp_path, ident=0.30):
    wd = tmp_path / "wd"
    wd.mkdir()
    (wd / "uniprot_sprot.fasta").write_text(
        ">sp|A1|X desc\n" + "M" * 700 + "\n"
        ">sp|B2|Y desc\n" + "M" * 300 + "\n")
    (wd / f"sprot_clu{int(ident * 100)}_cluster.tsv").write_text("A1\tA1\nA1\tB2\n")
    return wd


def _config(tmp_path, max_len):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"model:\n  max_len: {max_len}\n"
        "data:\n  cluster_identity: 0.30\n  cluster_coverage: 0.8\n"
        "  cluster_cov_mode: 0\n  cluster_sensitivity: 7.5\n"
        "  eval_cluster_fraction: 0.1\n  cluster_members: all\n")
    return cfg


def test_default_max_len_follows_the_config(tmp_path, monkeypatch, capsys):
    """A config cap the old literal did not have (500), no flag: the report
    caps at 500, so the read is live and not the retired default."""
    wd = _workdir(tmp_path)
    cfg = _config(tmp_path, 500)
    monkeypatch.setattr(sys, "argv", [
        "cluster", "--config", str(cfg), "--workdir", str(wd)])
    main()
    out = capsys.readouterr().out
    assert "capped at 500" in out
    # 700 -> 500, 300 -> 300: all-members total is 800, not 1000.
    assert f"{800:>15,}" in out


def test_a_disagreeing_max_len_is_refused_before_any_work(tmp_path, monkeypatch):
    wd = _workdir(tmp_path)
    cfg = _config(tmp_path, 1022)
    monkeypatch.setattr(sys, "argv", [
        "cluster", "--config", str(cfg), "--workdir", str(wd), "--max-len", "500"])
    with pytest.raises(SystemExit) as e:
        main()
    assert "500" in str(e.value) and "1022" in str(e.value)
    assert "model.max_len" in str(e.value)


def test_an_agreeing_max_len_is_accepted(tmp_path, monkeypatch, capsys):
    wd = _workdir(tmp_path)
    cfg = _config(tmp_path, 1022)
    monkeypatch.setattr(sys, "argv", [
        "cluster", "--config", str(cfg), "--workdir", str(wd), "--max-len", "1022"])
    main()
    assert "capped at 1022" in capsys.readouterr().out
