"""`verify_cache.py` gets a test of its own (amendment CQ, A4; the gap
`docs/DATAFLOW.md` section 2.3 named -- no test exercised the validator, and
CLAUDE.md's claim that it was tested against a corrupted cache had no fixture
behind it in the tree).

The entry point runs as a subprocess so its EXIT CODE is what is asserted
(gotcha 13 extension 1: a validator that prints FAIL and exits 0 is worse than
none). A positive control runs first -- a cache that matches its FASTA and its
expectations must pass -- and then one defect per check that CQ added, plus one
structural defect (a shifted `start`) so the file exercises the pre-existing
checks too.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.embeddings.cache import ESM_DIM

ROOT = Path(__file__).resolve().parents[1]
SEQS = [("sp|P00001|A_TEST", "MKV"), ("sp|P00002|B_TEST", "MKVLA")]
MODEL = "facebook/esm2_t33_650M_UR50D"


def _write(tmp_path: Path, max_len=1022, attrs_over=None, shift_start=False):
    fasta = tmp_path / "toy.fasta"
    fasta.write_text("".join(f">{acc} some description\n{seq}\n"
                             for acc, seq in SEQS))
    index, cursor = [], 0
    for acc, seq in SEQS:
        n = min(len(seq), max_len)
        index.append({"accession": acc, "start": cursor, "n_rows": n,
                      "seq_len": n, "truncated": len(seq) > max_len,
                      "residue_ids": None})
        cursor += n
    if shift_start:
        index[1]["start"] += 1
    attrs = {"model": MODEL, "layer": 24, "dtype": "float16",
             "final_layer_norm_applied": False, "max_len": max_len,
             "subsampled": False, "seed": 0}
    attrs.update(attrs_over or {})
    cache = tmp_path / "eval.h5"
    rng = np.random.default_rng(0)
    with h5py.File(cache, "w") as hf:
        hf.create_dataset("activations", data=rng.normal(
            0, 1, (cursor, ESM_DIM)).astype(np.float16))
        hf.create_dataset("index_json", data=np.bytes_(json.dumps(index)))
        for k, v in attrs.items():
            hf.attrs[k] = v
    return cache, fasta


def _run(cache, fasta, *extra):
    cmd = [sys.executable, "-m", "src.embeddings.verify_cache",
           "--cache", str(cache), "--fasta", str(fasta),
           "--nan-check-rows", "0", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def test_positive_control_a_consistent_cache_passes(tmp_path):
    cache, fasta = _write(tmp_path)
    r = _run(cache, fasta, "--expect-layer", "24", "--expect-model", MODEL)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "all checks passed" in r.stdout
    assert "FAIL" not in r.stdout


def test_verify_cache_fails_on_a_max_len_mismatch(tmp_path):
    """The cache says 1022; the run derives its expectations from 1000. Before
    CQ this passed every per-sequence check against the WRONG cap, because
    the fixture's sequences are short -- which is exactly the silent case."""
    cache, fasta = _write(tmp_path)
    r = _run(cache, fasta, "--max-len", "1000")
    assert r.returncode == 1, r.stdout[-3000:]
    assert "FAIL  max_len attr equals --max-len (1022 vs 1000)" in r.stdout


def test_verify_cache_fails_on_a_layer_mismatch_when_asked(tmp_path):
    cache, fasta = _write(tmp_path, attrs_over={"layer": 23})
    ok = _run(cache, fasta)                          # presence-only: passes
    assert ok.returncode == 0, ok.stdout[-2000:]
    r = _run(cache, fasta, "--expect-layer", "24")
    assert r.returncode == 1
    assert "layer attr equals --expect-layer (23 vs 24)" in r.stdout


def test_verify_cache_fails_on_a_model_mismatch_when_asked(tmp_path):
    cache, fasta = _write(tmp_path, attrs_over={"model": "facebook/esm2_t6_8M_UR50D"})
    r = _run(cache, fasta, "--expect-model", MODEL)
    assert r.returncode == 1
    assert "model attr equals --expect-model" in r.stdout


@pytest.mark.parametrize("defect", ["shift_start", "layer_norm_applied"])
def test_verify_cache_fails_on_pre_existing_checks_too(tmp_path, defect):
    """The two mutations DATAFLOW section 2.3 proposed: a shifted `start`
    (offsets no longer tile) and `final_layer_norm_applied=True`."""
    if defect == "shift_start":
        cache, fasta = _write(tmp_path, shift_start=True)
    else:
        cache, fasta = _write(tmp_path, attrs_over={"final_layer_norm_applied": True})
    r = _run(cache, fasta)
    assert r.returncode == 1, r.stdout[-3000:]
    assert "FAIL" in r.stdout
