"""
Tests for the layer-verification math in src/embeddings/verify_layer.py.

No ESM or reference weights here -- these pin the FVE accumulator and the SAE
reconstruction so a sign error or a backwards ratio can't sail through and make
a real off-by-one look fine on the cluster. The cluster run is the actual check.
"""

from __future__ import annotations

import numpy as np
import torch

from src.embeddings.verify_layer import FVEAccumulator, sae_reconstruct


def test_fve_is_one_for_perfect_reconstruction():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 16))
    acc = FVEAccumulator(16)
    acc.update(x, x.copy())              # recon == x
    assert abs(acc.fve() - 1.0) < 1e-9


def test_fve_is_zero_for_mean_prediction():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((500, 8))
    recon = np.repeat(x.mean(axis=0, keepdims=True), x.shape[0], axis=0)
    acc = FVEAccumulator(8)
    acc.update(x, recon)                 # predicting the mean -> FVE 0
    assert abs(acc.fve()) < 1e-9


def test_fve_streaming_matches_single_batch():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((300, 12))
    recon = x + 0.3 * rng.standard_normal((300, 12))
    whole = FVEAccumulator(12); whole.update(x, recon)
    streamed = FVEAccumulator(12)
    for i in range(0, 300, 37):
        streamed.update(x[i:i + 37], recon[i:i + 37])
    assert abs(whole.fve() - streamed.fve()) < 1e-9


def test_sae_reconstruct_shape_and_topk_sparsity():
    torch.manual_seed(0)
    d_model, d_hidden, k, n = 32, 128, 5, 10
    w_enc = torch.randn(d_model, d_hidden)
    w_dec = torch.randn(d_hidden, d_model)
    b_enc = torch.zeros(d_hidden)
    b_pre = torch.zeros(d_model)
    x = torch.randn(n, d_model)
    recon = sae_reconstruct(x, w_enc, w_dec, b_enc, b_pre, k)
    assert recon.shape == (n, d_model)
    assert torch.isfinite(recon).all()


def test_sae_reconstruct_recovers_a_planted_signal_better_than_shuffle():
    """A decoder aligned to the data reconstructs it well; a row-shuffled decoder
    does not. Same discriminative logic the layer sweep relies on, in miniature.

    TopK applies ReLU, so a reconstructing dictionary must be over-complete to
    represent both signs -- an identity autoencoder can't (its ReLU kills the
    negative half). Sign-split dictionary: latents = [relu(xn), relu(-xn)],
    decoded by [I; -I] gives relu(xn) - relu(-xn) = xn exactly."""
    torch.manual_seed(0)
    d, n = 16, 400
    w_enc = torch.cat([torch.eye(d), -torch.eye(d)], dim=1)   # (d, 2d)
    w_dec = torch.cat([torch.eye(d), -torch.eye(d)], dim=0)   # (2d, d)
    b_enc = torch.zeros(2 * d)
    b_pre = torch.zeros(d)
    x = torch.randn(n, d)
    k = d      # exactly the d positive entries per token are selected

    good = sae_reconstruct(x, w_enc, w_dec, b_enc, b_pre, k)
    # Permute ONLY the decoder rows: the encoder still fires the right latents,
    # but they now decode to the wrong directions (analogous to feeding the SAE
    # the wrong layer). Permuting both would just relabel latents -> identical.
    perm = torch.randperm(2 * d)
    bad = sae_reconstruct(x, w_enc, w_dec[perm], b_enc, b_pre, k)

    def fve(recon):
        a = FVEAccumulator(d); a.update(x.numpy(), recon.numpy()); return a.fve()

    assert fve(good) > 0.99                 # near-lossless LN round-trip
    assert fve(good) > fve(bad) + 0.2       # and clearly beats the shuffle


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
