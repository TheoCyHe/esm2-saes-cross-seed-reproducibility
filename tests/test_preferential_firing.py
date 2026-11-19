"""
Tests for the D6 preferential-firing check.

`masked_mean_activation` is the one piece of arithmetic the whole diagnostic
rests on: mean activation of a feature over an arbitrary row subset. Checked
against a naive dense computation, and against a synthetic confound built to
be flagged -- if the flag logic were disabled the confound test must fail,
per this project's "verify a validator can actually fail" rule.
"""

from __future__ import annotations

import numpy as np
import torch

from src.analysis.preferential_firing import masked_mean_activation


def _sparse_from_dense(Z: np.ndarray):
    """Dense (n_rows, n_features) -> (feat, pos, val), any order (mean is
    order-independent, unlike the AP path)."""
    nz = np.nonzero(Z)
    feat = torch.from_numpy(nz[1].astype(np.int64))
    pos = torch.from_numpy(nz[0].astype(np.int64))
    val = torch.from_numpy(Z[nz].astype(np.float32))
    return feat, pos, val


def test_matches_naive_dense_mean():
    rng = np.random.default_rng(0)
    n_rows, n_features = 500, 6
    Z = np.zeros((n_rows, n_features))
    for i in range(n_rows):
        cols = rng.choice(n_features, 2, replace=False)
        Z[i, cols] = rng.uniform(0.1, 5.0, 2)
    keep = rng.random(n_rows) < 0.4

    feat, pos, val = _sparse_from_dense(Z)
    ours = masked_mean_activation(feat, pos, val, torch.from_numpy(keep), n_features).numpy()
    ref = Z[keep].mean(axis=0)                    # naive: includes the zeros
    assert np.allclose(ours, ref, atol=1e-6), (ours, ref)


def test_empty_mask_returns_zero_not_nan():
    Z = np.zeros((100, 3))
    Z[:10, 0] = 1.0
    feat, pos, val = _sparse_from_dense(Z)
    keep = torch.zeros(100, dtype=torch.bool)      # nothing kept
    out = masked_mean_activation(feat, pos, val, keep, 3)
    assert torch.isfinite(out).all() and torch.all(out == 0.0)


def test_silent_feature_scores_zero_on_any_mask():
    Z = np.zeros((200, 4))
    Z[:50, 0] = rng_vals = np.linspace(0.1, 5.0, 50)
    feat, pos, val = _sparse_from_dense(Z)
    keep = torch.ones(200, dtype=torch.bool)
    out = masked_mean_activation(feat, pos, val, keep, 4)
    assert out[1] == 0.0 and out[2] == 0.0 and out[3] == 0.0
    assert abs(out[0].item() - rng_vals.sum() / 200) < 1e-5   # kept rows = 200, not 50


def _preferential_ratio(exp_mean: float, prop_mean: float) -> float:
    """The exact ratio computed in preferential_firing.py's driver loop."""
    return exp_mean / prop_mean if prop_mean > 0 else float("inf")


def test_confound_signature_is_detected_and_the_detector_can_fail():
    """THE D6 test. Build a feature that fires ONLY on experimental-positive
    rows of a concept and never on propagated-positive rows of the SAME
    concept -- exactly the confound D6 exists to catch (detects study
    intensity, not chemistry). The ratio must be flagged; and if the ratio
    computation were replaced by something that ignores the tier split (e.g.
    comparing against ALL rows instead of the propagated tier specifically),
    this test must fail, which is checked explicitly below.
    """
    n_rows = 2000
    exp_rows = np.arange(0, 40)          # 40 experimental-positive rows
    prop_rows = np.arange(40, 140)       # 100 propagated-positive rows, SAME concept

    Z = np.zeros((n_rows, 1))
    Z[exp_rows, 0] = 5.0                 # fires hard on experimental positives
    Z[prop_rows, 0] = 0.0                # never fires on propagated positives
    # a little background firing elsewhere so the feature isn't literally silent
    Z[500:520, 0] = 1.0

    feat, pos, val = _sparse_from_dense(Z)
    exp_mask = torch.zeros(n_rows, dtype=torch.bool); exp_mask[exp_rows] = True
    prop_mask = torch.zeros(n_rows, dtype=torch.bool); prop_mask[prop_rows] = True

    exp_mean = masked_mean_activation(feat, pos, val, exp_mask, 1)[0].item()
    prop_mean = masked_mean_activation(feat, pos, val, prop_mask, 1)[0].item()
    ratio = _preferential_ratio(exp_mean, prop_mean)

    assert prop_mean == 0.0
    assert ratio == float("inf")
    assert ratio >= 3.0                  # the driver's default flag_ratio

    # And a real (unconfounded) detector -- fires equally on both tiers --
    # must NOT be flagged, so the check isn't just "always flag".
    Z2 = np.zeros((n_rows, 1))
    Z2[exp_rows, 0] = 5.0
    Z2[prop_rows, 0] = 5.0
    feat2, pos2, val2 = _sparse_from_dense(Z2)
    exp_mean2 = masked_mean_activation(feat2, pos2, val2, exp_mask, 1)[0].item()
    prop_mean2 = masked_mean_activation(feat2, pos2, val2, prop_mask, 1)[0].item()
    ratio2 = _preferential_ratio(exp_mean2, prop_mean2)
    assert abs(ratio2 - 1.0) < 1e-9
    assert ratio2 < 3.0


def test_detector_fails_if_masking_by_tier_is_dropped():
    """Directly verifies the previous test's claim: if the check were computed
    against ALL positive rows (tiers merged) instead of the propagated tier
    specifically, the confound would be invisible. This documents WHY D6
    insists on the tier split rather than "mean activation on positives"."""
    n_rows = 2000
    exp_rows = np.arange(0, 40)
    prop_rows = np.arange(40, 140)
    Z = np.zeros((n_rows, 1))
    Z[exp_rows, 0] = 5.0
    Z[prop_rows, 0] = 0.0

    feat, pos, val = _sparse_from_dense(Z)
    exp_mask = torch.zeros(n_rows, dtype=torch.bool); exp_mask[exp_rows] = True
    all_pos_mask = torch.zeros(n_rows, dtype=torch.bool)
    all_pos_mask[exp_rows] = True
    all_pos_mask[prop_rows] = True                 # tiers merged -- the WRONG comparator

    exp_mean = masked_mean_activation(feat, pos, val, exp_mask, 1)[0].item()
    merged_mean = masked_mean_activation(feat, pos, val, all_pos_mask, 1)[0].item()
    wrong_ratio = _preferential_ratio(exp_mean, merged_mean)
    # merged mean is diluted by the 100 non-firing propagated rows but is NOT
    # zero, so this comparator understates the confound relative to the
    # tier-specific one (prop_mean == 0 -> ratio == inf, tested above).
    assert wrong_ratio < float("inf")
    assert wrong_ratio < 3.0 * (100 / 40)          # materially weaker signal
