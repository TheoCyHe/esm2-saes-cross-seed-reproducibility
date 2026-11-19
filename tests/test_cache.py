"""
Tests for the silent-corruption risks called out in src/embeddings/cache.py
and CLAUDE.md: residue off-by-one, truncation bookkeeping, and reproducible
subsampling. These run on plain tensors -- no ESM model or GPU needed -- so
they should be run locally before every cluster job that touches this file.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.embeddings.cache import read_fasta, strip_and_subsample


def test_read_fasta_parses_multiline_sequences(tmp_path):
    fasta = tmp_path / "toy.fasta"
    fasta.write_text(
        ">P12345 some description here\n"
        "MKV\n"
        "LAS\n"
        ">Q9XYZ1\n"
        "GG\n"
    )
    records = read_fasta(fasta)
    assert records == [("P12345", "MKVLAS"), ("Q9XYZ1", "GG")]


def test_strip_and_subsample_no_subsample_returns_full_sequence():
    L = 10
    resid = torch.arange(L * 4, dtype=torch.float32).reshape(L, 4)
    arr, rows = strip_and_subsample(
        resid, L, attention_mask_sum=L + 2, acc="P1",
        max_residues=None, keep_frac=1.0, rng=np.random.default_rng(0),
    )
    assert rows is None
    assert arr.shape == (L, 4)
    assert arr.dtype == np.float16
    np.testing.assert_array_equal(arr, resid.numpy().astype(np.float16))


def test_off_by_one_slice_trips_assertion():
    """CLAUDE.md gotcha 2: an off-by-one in special-token stripping must
    never pass silently. Simulate it by handing over one extra row (as if
    <eos> leaked into the slice) and confirm the assertion catches it."""
    L = 10
    resid_with_leaked_eos = torch.zeros(L + 1, 4)
    with pytest.raises(AssertionError, match="special-token stripping is wrong"):
        strip_and_subsample(
            resid_with_leaked_eos, L, attention_mask_sum=L + 2, acc="P1",
            max_residues=None, keep_frac=1.0, rng=np.random.default_rng(0),
        )


def test_short_attention_mask_trips_assertion():
    """Guards against slicing into padding (module docstring, assertion 2)."""
    L = 10
    resid = torch.zeros(L, 4)
    with pytest.raises(AssertionError, match="attention mask shorter"):
        strip_and_subsample(
            resid, L, attention_mask_sum=L,  # missing the +2 for <cls>/<eos>
            acc="P1", max_residues=None, keep_frac=1.0,
            rng=np.random.default_rng(0),
        )


def test_subsampling_reproducible_with_same_seed():
    L = 200
    resid = torch.arange(L * 4, dtype=torch.float32).reshape(L, 4)

    arr_a, rows_a = strip_and_subsample(
        resid, L, L + 2, "P1", max_residues=1, keep_frac=0.1,
        rng=np.random.default_rng(42),
    )
    arr_b, rows_b = strip_and_subsample(
        resid, L, L + 2, "P1", max_residues=1, keep_frac=0.1,
        rng=np.random.default_rng(42),
    )
    np.testing.assert_array_equal(rows_a, rows_b)
    np.testing.assert_array_equal(arr_a, arr_b)


def test_subsampling_keeps_at_least_one_residue():
    L = 500
    resid = torch.zeros(L, 4)
    arr, rows = strip_and_subsample(
        resid, L, L + 2, "P1", max_residues=1, keep_frac=1e-6,
        rng=np.random.default_rng(0),
    )
    assert arr.shape[0] >= 1
    assert rows is not None
    assert rows.shape[0] == arr.shape[0]
    # rows must be valid, sorted indices into [0, L)
    assert (rows >= 0).all() and (rows < L).all()
    assert (np.diff(rows) > 0).all()


# ------------------------------------------------ amendment CR, Row A --
# `main()` needs transformers and a model; the resolution step runs BEFORE
# that import and is tested on its own. The expectations below are written
# down (1022 / 24 / the model name), not read from the code under test.

def _cfg(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return p


def test_parser_defaults_are_none_so_the_config_governs():
    from src.embeddings.cache import build_parser
    args = build_parser().parse_args(["--fasta", "x.fa", "--out", "y.h5"])
    assert args.model is None and args.layer is None and args.max_len is None


def test_resolution_fills_model_layer_and_max_len_from_the_config(tmp_path):
    from src.embeddings.cache import build_parser, resolve_experiment_args
    cfg = _cfg(tmp_path, "model:\n  name: toy/esm\n  layer: 7\n  max_len: 500\n")
    args = resolve_experiment_args(build_parser().parse_args(
        ["--config", str(cfg), "--fasta", "x.fa", "--out", "y.h5"]))
    assert (args.model, args.layer, args.max_len) == ("toy/esm", 7, 500)


def test_resolution_refuses_a_disagreeing_layer(tmp_path):
    from src.embeddings.cache import build_parser, resolve_experiment_args
    cfg = _cfg(tmp_path, "model:\n  name: toy/esm\n  layer: 24\n  max_len: 1022\n")
    with pytest.raises(SystemExit) as e:
        resolve_experiment_args(build_parser().parse_args(
            ["--config", str(cfg), "--fasta", "x.fa", "--out", "y.h5",
             "--layer", "23"]))
    assert "--layer" in str(e.value) and "23" in str(e.value) and "24" in str(e.value)


def test_resolution_refuses_a_disagreeing_max_len_and_model(tmp_path):
    from src.embeddings.cache import build_parser, resolve_experiment_args
    cfg = _cfg(tmp_path, "model:\n  name: toy/esm\n  layer: 24\n  max_len: 1022\n")
    with pytest.raises(SystemExit):
        resolve_experiment_args(build_parser().parse_args(
            ["--config", str(cfg), "--fasta", "x.fa", "--out", "y.h5",
             "--max-len", "1000"]))
    with pytest.raises(SystemExit):
        resolve_experiment_args(build_parser().parse_args(
            ["--config", str(cfg), "--fasta", "x.fa", "--out", "y.h5",
             "--model", "other/model"]))


def test_resolution_refuses_an_absent_key(tmp_path):
    from src.embeddings.cache import build_parser, resolve_experiment_args
    cfg = _cfg(tmp_path, "model:\n  name: toy/esm\n  layer: 24\n")   # no max_len
    with pytest.raises(SystemExit) as e:
        resolve_experiment_args(build_parser().parse_args(
            ["--config", str(cfg), "--fasta", "x.fa", "--out", "y.h5"]))
    assert "model.max_len" in str(e.value)


def test_the_repository_default_config_reproduces_the_old_literals():
    """No flags, the repository's own config: exactly what the sbatch files
    got before CR (`--layer 24` is what they pass, and it agrees)."""
    from src.embeddings.cache import build_parser, resolve_experiment_args
    args = resolve_experiment_args(build_parser().parse_args(
        ["--fasta", "x.fa", "--out", "y.h5", "--layer", "24"]))
    assert args.model == "facebook/esm2_t33_650M_UR50D"
    assert args.layer == 24
    assert args.max_len == 1022
