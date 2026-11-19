"""Tests for src/analysis/fvu.py -- PREREGISTRATION amendment CF-a.

The expectations here are WRITTEN DOWN, not computed from the module under test
(gotcha 32b). Two of them are closed-form: a perfect reconstruction must give
FVU exactly 0, and a reconstruction that is the pooled mean must give FVU
exactly 1 in both spaces. Those are the cases that make the accumulator's
arithmetic falsifiable rather than merely self-consistent.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.analysis.fvu import Accumulator, reconstruct


def test_perfect_reconstruction_gives_FVU_zero_in_both_spaces():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 7)) * np.array([1.0, 10.0, 0.1, 3, 3, 3, 3])
    a = Accumulator(7)
    a.update(x, x.copy())
    r = a.result()
    assert r["raw"]["fvu"] == pytest.approx(0.0, abs=1e-12)
    assert r["standardised"]["fvu"] == pytest.approx(0.0, abs=1e-12)


def test_predicting_the_mean_gives_FVU_exactly_one_in_both_spaces():
    """The closed form that pins the DENOMINATOR. If SST were computed about
    zero rather than about the pooled mean, this would not be 1."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 5)) * np.array([1.0, 20.0, 0.05, 2.0, 7.0]) + 3.0
    recon = np.broadcast_to(x.mean(axis=0), x.shape).copy()
    a = Accumulator(5)
    a.update(x, recon)
    r = a.result()
    assert r["raw"]["fvu"] == pytest.approx(1.0, rel=1e-12)
    assert r["standardised"]["fvu"] == pytest.approx(1.0, rel=1e-12)


def test_the_two_spaces_DISAGREE_when_per_dimension_variance_is_heterogeneous():
    """The whole reason CF-a declares two spaces. A fixture whose dimensions
    had equal variance could not exhibit the difference and would pass against
    a module that computed one space twice."""
    rng = np.random.default_rng(2)
    n = 2000
    # dim 0 has huge variance and is reconstructed well; dim 1 is tiny and is
    # reconstructed badly. Raw FVU is dominated by dim 0, standardised is not.
    x = np.stack([rng.normal(0, 50.0, n), rng.normal(0, 0.1, n)], axis=1)
    recon = np.stack([x[:, 0] * 0.99, np.zeros(n)], axis=1)
    a = Accumulator(2)
    a.update(x, recon)
    r = a.result()
    raw, std = r["raw"]["fvu"], r["standardised"]["fvu"]
    assert raw < 0.05, raw            # flattered by the high-variance dim
    assert std > 0.4, std             # the neglected dim now counts equally
    assert std > raw * 5


def test_chunked_updates_equal_a_single_update():
    """Streaming must not change the answer -- the module's whole shape."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(1000, 6)) * np.array([1, 5, 2, 9, 0.3, 4.0])
    recon = x + rng.normal(size=x.shape) * 0.4
    one = Accumulator(6)
    one.update(x, recon)
    many = Accumulator(6)
    for i in range(0, 1000, 97):
        many.update(x[i:i + 97], recon[i:i + 97])
    for space in ("raw", "standardised"):
        assert many.result()[space]["fvu"] == pytest.approx(
            one.result()[space]["fvu"], rel=1e-10)
    assert many.result()["n_rows"] == 1000


def test_a_constant_dimension_is_REPORTED_not_dropped_silently():
    rng = np.random.default_rng(4)
    x = np.concatenate([rng.normal(size=(300, 3)),
                        np.full((300, 1), 2.5)], axis=1)
    a = Accumulator(4)
    a.update(x, x + rng.normal(size=x.shape) * 0.1)
    r = a.result()
    assert r["n_zero_variance_dims"] == 1
    # SST_std counts only the live dimensions, and the identity is asserted.
    assert r["standardised"]["sst"] == pytest.approx(300.0 * 3)
    assert "ALGEBRAIC IDENTITY" in r["standardised"][
        "sst_is_exactly_n_times_live_dims"]


def test_standardised_FVU_equals_an_INDEPENDENT_numpy_standardisation():
    """The identity `SST_std == n * d_live` cannot be checked at runtime -- it
    is algebraic, so a runtime assertion could only pass (gotcha 25; a first
    draft of the module shipped exactly that and this test caught it).

    So it is checked HERE, against a standardisation computed a completely
    different way: standardise x and recon explicitly, then take SSE and SST in
    the standardised array. That path can disagree with the accumulator's, and
    it is what makes the streaming form falsifiable.
    """
    rng = np.random.default_rng(5)
    x = rng.normal(size=(300, 5)) * np.array([1.0, 12.0, 0.2, 4.0, 30.0]) + 1.7
    recon = x + rng.normal(size=x.shape) * np.array([0.3, 2.0, 0.05, 1.0, 9.0])

    a = Accumulator(5)
    a.update(x, recon)
    got = a.result()["standardised"]["fvu"]

    mu = x.mean(axis=0)
    sd = x.std(axis=0)                       # population sd, as SST uses
    xs = (x - mu) / sd
    rs = (recon - mu) / sd                   # SAME statistics on both sides
    want = ((xs - rs) ** 2).sum() / ((xs - xs.mean(axis=0)) ** 2).sum()

    assert got == pytest.approx(want, rel=1e-10)
    # and the identity the module states, verified on this independent path
    assert ((xs - xs.mean(axis=0)) ** 2).sum() == pytest.approx(300.0 * 5)


def test_reconstruct_AGREES_with_the_real_TopKSAE():
    """The module reimplements the forward path; this is what makes that
    duplication safe. Expectation written against the TRAINING class, not
    against this module."""
    from src.sae.train import TopKSAE
    torch.manual_seed(0)
    d_model, dict_size, k = 16, 64, 5
    sae = TopKSAE(d_model, dict_size, k,
                  b_dec=torch.randn(d_model), b_dec_trainable=True)
    with torch.no_grad():
        sae.b_enc.copy_(torch.randn(dict_size) * 0.1)
    x = torch.randn(37, d_model)
    with torch.no_grad():
        want, _, _ = sae(x)
        got = reconstruct(x, sae.W_enc, sae.b_enc, sae.W_dec, sae.b_dec, k)
    assert torch.allclose(want, got, atol=1e-6)


def test_shape_and_width_mismatches_raise_rather_than_broadcasting():
    a = Accumulator(4)
    with pytest.raises(ValueError):
        a.update(np.zeros((5, 4)), np.zeros((5, 3)))
    with pytest.raises(ValueError):
        a.update(np.zeros((5, 3)), np.zeros((5, 3)))


def test_no_rows_is_reported_as_a_status_not_a_nan_fvu():
    assert Accumulator(4).result() == {"status": "NO_ROWS", "n_rows": 0}


# ------------------------------------------------ amendment CS, CS3b --
# The shared-git_sha refusal stays the default; a caller may name the EXACT
# SHA set the record adjudicated (§CN7: the 40,960 arm's seed 0 vs seeds 1-9).

def test_one_shared_sha_passes_and_is_recorded():
    from src.analysis.fvu import resolve_ensemble_git_sha
    r = resolve_ensemble_git_sha({0: "a" * 40, 1: "a" * 40}, None)
    assert r["ensemble_git_sha"] == "a" * 40 and r["accepted_by_flag"] is False


def test_a_disagreement_is_refused_without_the_flag():
    from src.analysis.fvu import resolve_ensemble_git_sha
    with pytest.raises(SystemExit):
        resolve_ensemble_git_sha({0: "a" * 40, 1: "b" * 40}, None)


def test_the_flag_must_name_the_exact_set():
    from src.analysis.fvu import resolve_ensemble_git_sha
    shas = {0: "a" * 40, 1: "b" * 40, 2: "b" * 40}
    r = resolve_ensemble_git_sha(shas, ["b" * 40, "a" * 40])
    assert r["accepted_by_flag"] is True and r["ensemble_git_sha"] is None
    assert r["git_shas_by_seed"] == {"0": "a" * 40, "1": "b" * 40, "2": "b" * 40}
    with pytest.raises(SystemExit):                      # a superset is not the set
        resolve_ensemble_git_sha(shas, ["a" * 40, "b" * 40, "c" * 40])
    with pytest.raises(SystemExit):                      # a subset is not the set
        resolve_ensemble_git_sha(shas, ["a" * 40])


def test_the_parser_has_out_and_the_acceptance_flag():
    """CS3c: the CS3b commit dropped `--out`; a cluster job found it. Pinned."""
    from src.analysis.fvu import build_parser
    a = build_parser().parse_args(["--cache", "c", "--ckpts", "k", "--seeds", "0", "1",
                                   "--k", "32", "--accept-git-shas", "a" * 40, "b" * 40,
                                   "--out", "o.json"])
    assert str(a.out) == "o.json" and a.accept_git_shas == ["a" * 40, "b" * 40]
