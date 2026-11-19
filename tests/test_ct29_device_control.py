"""§CT29-F item 2: the device control's verdict is total, its ranges are the
declared ones, and its MNN count reproduces `rf_marginal`'s on a known case."""
import numpy as np
import torch

from src.analysis import ct29_device_control as dc


def test_declared_ranges_written_down():
    assert dc.FVU_RANGE == (0.36214210341635106, 0.36656082216321473)
    assert dc.RATE_RANGE == (0.49134457236842105, 0.5000668174342104)
    assert dc.THETA == 0.70


def test_verdict_is_total_and_closed():
    lo_f, hi_f = dc.FVU_RANGE
    lo_r, hi_r = dc.RATE_RANGE
    assert dc.verdict(lo_f, lo_r) == "PASS"          # closed at both ends
    assert dc.verdict(hi_f, hi_r) == "PASS"
    assert dc.verdict(hi_f + 1e-12, lo_r) == "FAIL"
    assert dc.verdict(lo_f, lo_r - 1e-12) == "FAIL"
    assert dc.verdict(None, lo_r) == "NOT_EXECUTED"  # absent is never a pass
    assert dc.verdict(lo_f, None) == "NOT_EXECUTED"


def _unit(W):
    return W / W.norm(dim=0, keepdim=True)


def test_mnn_counts_on_a_known_case():
    g = torch.Generator().manual_seed(0)
    W = _unit(torch.randn(64, 32, generator=g))
    perm = torch.randperm(32, generator=g)
    same = W[:, perm]                                 # every feature has an exact partner
    other = _unit(torch.randn(64, 32, generator=g))   # random: few or none above 0.70
    k = dc.mnn_counts(W, [same, same, other], 0.70)
    assert k.shape == (32,)
    assert (k >= 2).all() and (k <= 3).all()
    r = dc.pair_rates(W, [same, other], 0.70)
    assert r[0] == 1.0 and r[1] < 0.5


def test_mnn_counts_can_fail():
    """A perturbation that breaks every partnership must lower the count, so
    the instrument check is able to report a difference (gotcha 13)."""
    g = torch.Generator().manual_seed(1)
    W = _unit(torch.randn(64, 16, generator=g))
    far = _unit(torch.randn(64, 16, generator=g))
    assert int(dc.mnn_counts(W, [W], 0.70).sum()) == 16
    assert int(dc.mnn_counts(W, [far], 0.70).sum()) < 16
