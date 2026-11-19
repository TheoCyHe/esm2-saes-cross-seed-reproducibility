"""
Tests for the reconstruction-mse helper in src/sae/diagnose_b_dec.py.

The freeze-vs-trainable decision rests on comparing recon_mse under own vs
pooled b_dec, so recon_mse must agree with the training loss convention
(d_model_mean) exactly, and must be chunk-invariant -- otherwise a wrong number
gates a real experiment decision.
"""

from __future__ import annotations

import torch

from src.sae.diagnose_b_dec import recon_mse
from src.sae.train import TopKSAE, reduce_se


def test_recon_mse_matches_d_model_mean_and_is_chunk_invariant():
    torch.manual_seed(0)
    model = TopKSAE(d_model=16, dict_size=64, k=5, b_dec=torch.randn(16))
    model.eval()
    X = torch.randn(500, 16)

    x_hat, _, _ = model(X)
    direct = reduce_se((x_hat - X) ** 2, "d_model_mean").item()

    whole = recon_mse(model, X, "cpu", chunk=10_000)      # one chunk
    split = recon_mse(model, X, "cpu", chunk=37)          # many chunks
    assert abs(whole - direct) < 1e-5
    assert abs(split - direct) < 1e-5                     # chunking must not change it


def test_recon_mse_zero_for_identity_reconstruction():
    # a model whose reconstruction equals the input gives mse 0
    model = TopKSAE(d_model=8, dict_size=16, k=4, b_dec=torch.zeros(8))
    X = torch.randn(50, 8)
    with torch.no_grad():
        # force the reconstruction to equal x by monkeypatching forward
        real_forward = model.forward
        model.forward = lambda x: (x, None, None)
        try:
            assert recon_mse(model, X, "cpu") == 0.0
        finally:
            model.forward = real_forward


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
