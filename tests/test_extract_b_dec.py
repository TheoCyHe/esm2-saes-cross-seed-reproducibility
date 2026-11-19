"""
Tests for src/sae/extract_frozen_b_dec.py -- converge-then-freeze extraction.

The frozen ensemble b_dec is the MEAN of the pilots' converged b_dec, and the
cross-seed spread is the ablation metric. Both must be right: a wrong mean feeds
every ensemble seed a bad bias; a wrong spread misinforms the freeze-vs-trainable
call. Written so either mistake trips.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

from src.sae.extract_frozen_b_dec import cross_seed_spread, main


def test_cross_seed_spread_zero_for_identical_pilots():
    v = np.arange(8, dtype=np.float64)
    mat = np.stack([v, v, v])
    s = cross_seed_spread(mat)
    assert s["rms_dist_to_mean"] == 0.0
    assert s["relative_rms_spread"] == 0.0
    assert abs(s["mean_norm"] - np.linalg.norm(v)) < 1e-9


def test_cross_seed_spread_positive_when_pilots_differ():
    mat = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    s = cross_seed_spread(mat)
    assert s["rms_dist_to_mean"] > 0.0
    assert s["max_dist_to_mean"] >= s["rms_dist_to_mean"]


def _write_pilot(dir_, b_dec, seed):
    dir_.mkdir(parents=True, exist_ok=True)
    torch.save({"b_dec": torch.tensor(b_dec, dtype=torch.float32),
                "W_enc": torch.zeros(4, 2)}, dir_ / "model.pt")
    (dir_ / "meta.json").write_text(json.dumps({
        "seed": seed, "b_dec_trainable": True, "git_sha": "abc",
        "config_hash": "cfg", "n_steps": 100, "lr": 2e-4,
    }))


def test_extract_averages_pilots_and_records_provenance(tmp_path, monkeypatch):
    # three pilots whose b_dec average to a known vector
    _write_pilot(tmp_path / "s0", [0.0, 0.0, 0.0, 0.0], 0)
    _write_pilot(tmp_path / "s1", [3.0, 3.0, 3.0, 3.0], 1)
    _write_pilot(tmp_path / "s2", [6.0, 6.0, 6.0, 6.0], 2)   # mean = 3.0 each
    out = tmp_path / "b_dec.pt"
    monkeypatch.setattr(sys, "argv", [
        "extract", "--pilot-ckpts",
        str(tmp_path / "s0"), str(tmp_path / "s1"), str(tmp_path / "s2"),
        "--out", str(out),
    ])
    main()

    blob = torch.load(out, weights_only=False)
    np.testing.assert_allclose(blob["b_dec"].numpy(), [3.0, 3.0, 3.0, 3.0])
    assert blob["source"] == "converged_pilot_mean"
    assert blob["n_pilots"] == 3
    assert blob["pilot_seeds"] == [0, 1, 2]
    assert blob["cross_seed_spread"]["rms_dist_to_mean"] > 0.0
    # output is loadable by train.py's --b-dec path (dict with a "b_dec" tensor)
    assert "b_dec" in blob and blob["b_dec"].shape == (4,)


def test_extract_single_pilot_is_that_pilot(tmp_path, monkeypatch):
    _write_pilot(tmp_path / "s0", [1.0, 2.0, 3.0, 4.0], 0)
    out = tmp_path / "b_dec.pt"
    monkeypatch.setattr(sys, "argv", [
        "extract", "--pilot-ckpts", str(tmp_path / "s0"), "--out", str(out),
    ])
    main()
    blob = torch.load(out, weights_only=False)
    np.testing.assert_allclose(blob["b_dec"].numpy(), [1.0, 2.0, 3.0, 4.0])
    assert blob["cross_seed_spread"]["rms_dist_to_mean"] == 0.0
