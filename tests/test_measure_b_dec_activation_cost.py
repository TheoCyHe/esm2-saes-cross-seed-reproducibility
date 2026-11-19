"""
Tests for src/sae/measure_b_dec_activation_cost.py.

The measurement gates an experiment decision, so the streaming stats must be
right: swapping b_dec for the SAME value must give Pearson == 1 and Jaccard == 1
(identity), and a different b_dec must move them below 1.
"""

from __future__ import annotations

import numpy as np
import torch

from src.sae.measure_b_dec_activation_cost import measure
from src.sae.train import TopKSAE


def _model():
    torch.manual_seed(0)
    m = TopKSAE(d_model=16, dict_size=64, k=5, b_dec=torch.randn(16))
    m.eval()
    return m


def test_identical_b_dec_gives_perfect_correlation_and_jaccard():
    m = _model()
    b = m.b_dec.detach().clone()
    X = torch.randn(400, 16)
    pearson, jaccard, firing_rate, n = measure(m, b, b, X, "cpu", chunk=64)
    alive = firing_rate > 0
    # a feature identical to itself: perfect correlation and firing-set overlap
    assert np.allclose(np.nan_to_num(pearson[alive], nan=1.0), 1.0, atol=1e-6)
    assert np.allclose(jaccard[alive], 1.0, atol=1e-9)
    assert n == 400


def test_different_b_dec_degrades_some_features():
    m = _model()
    b_own = m.b_dec.detach().clone()
    b_swap = b_own + 2.0 * torch.randn(16)     # a real offset
    X = torch.randn(400, 16)
    _, jaccard, firing_rate, _ = measure(m, b_own, b_swap, X, "cpu", chunk=64)
    alive = firing_rate > 0
    # a nontrivial b_dec shift must change at least some features' firing sets
    assert np.nanmin(jaccard[alive]) < 1.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
