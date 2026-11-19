"""
Tests for src/analysis/corrector_scan.py.

This scan is a GATE on the headline result, so the thing that matters most is
that it CAN FAIL: a detector that never fires is worse than none. The central
test plants a real corrector (a latent wired to fire on every token with
constant magnitude, i.e. acting as a bias) and requires the scan to flag it,
paired with a false-positive test requiring a normal model to come back clean.
"""

from __future__ import annotations

import json
import numpy as np
import torch
import yaml

from src.analysis.corrector_scan import (
    BIN_EDGES,
    ablate_and_measure,
    bin_counts,
    scan_features,
    scan_one,
)
from src.sae.train import TopKSAE


def _normal_model(d_model=16, dict_size=64, k=5, seed=0):
    torch.manual_seed(seed)
    m = TopKSAE(d_model, dict_size, k, b_dec=torch.zeros(d_model))
    m.eval()
    return m


def _plant_corrector(m: TopKSAE, feature: int = 0, level: float = 100.0):
    """Wire one latent to act as a BIAS: zero encoder row (so its pre-activation
    ignores x) plus a large encoder bias (so it is always in the top-k at
    constant magnitude). That is exactly the corrector pathology."""
    with torch.no_grad():
        m.W_enc[feature] = 0.0
        m.b_enc[feature] = level
    return m


def test_planted_corrector_is_flagged():
    """THE test: the scan must fire on a real corrector."""
    m = _plant_corrector(_normal_model())
    X = torch.randn(2000, 16)
    st = scan_features(m, X, "cpu", chunk=256)

    # the planted latent fires on every token, at constant magnitude
    assert st["firing_rate"][0] == 1.0
    assert st["bias_fraction"][0] > 0.99      # pure DC offset

    flagged = (st["firing_rate"] >= 0.5) & (st["bias_fraction"] >= 0.5)
    assert flagged[0], "planted corrector was NOT flagged -- the gate is blind"
    assert flagged.sum() == 1, "flagged something other than the planted corrector"


def test_normal_model_has_no_corrector():
    """False-positive check: an ordinary sparse SAE must come back clean, or the
    gate would flag real features and be useless."""
    m = _normal_model()
    X = torch.randn(2000, 16)
    st = scan_features(m, X, "cpu", chunk=256)
    flagged = (st["firing_rate"] >= 0.5) & (st["bias_fraction"] >= 0.5)
    assert flagged.sum() == 0
    # sanity: bias_fraction is bounded above by firing rate for sparse latents
    alive = st["firing_rate"] > 0
    assert np.all(st["bias_fraction"][alive] <= st["firing_rate"][alive] + 1e-9)


def test_bias_fraction_separates_constant_from_variable_latent():
    """Two latents, both essentially always-on, differing only in whether their
    magnitude is constant. bias_fraction must rank the constant one higher --
    it has to measure constancy, not just always-on."""
    torch.manual_seed(1)
    m = _normal_model()
    with torch.no_grad():
        m.W_enc[0] = 0.0; m.b_enc[0] = 100.0           # always on, constant
        m.W_enc[1] = 0.0; m.b_enc[1] = 100.0           # always on, constant...
    X = torch.randn(2000, 16)
    st_const = scan_features(m, X, "cpu", chunk=256)
    with torch.no_grad():
        # large constant + moderate input-dependence: still fires ~always (mean
        # 100, sd ~40), but its magnitude now varies with the input.
        m.W_enc[1] = 10.0 * torch.randn(16)
    st_var = scan_features(m, X, "cpu", chunk=256)

    assert st_const["bias_fraction"][1] > 0.99
    assert st_var["firing_rate"][1] > 0.9              # still essentially always on
    assert st_var["bias_fraction"][1] < st_const["bias_fraction"][1] - 0.05


def test_genuine_corrector_ablation_exposes_the_hidden_offset():
    """The real pathology, end to end: data carries an offset v that b_dec does
    NOT cover, and a latent points along v and cancels it. The scan must show
    (i) the decoder aligned with the uncovered offset E[x] - b_dec, and
    (ii) ablating it EXPOSES the offset in the mean residual -- the corrector
    was holding up a systematic DC error, which is the thing that biases the
    headline."""
    torch.manual_seed(0)
    d = 16
    v = torch.zeros(d); v[3] = 40.0; v[7] = 30.0       # offset, ||v|| = 50
    m = _normal_model(d_model=d)
    with torch.no_grad():
        m.b_dec.zero_()                                # bias does NOT cover v
        m.W_enc[0] = 0.0                               # latent 0: always on,
        m.b_enc[0] = float(v.norm())                   #   constant magnitude 50,
        m.W_dec[:, 0] = v / v.norm()                   #   pointing along v
        # Silence the other latents (relu(-10) == 0) so the corrector's DC effect
        # is isolated. In an untrained model they fire hard on the large-mean
        # input and their junk contribution swamps the mechanism under test.
        m.W_enc[1:] = 0.0
        m.b_enc[1:] = -10.0
    X = torch.randn(3000, d) + v                       # E[x] = v

    st = scan_features(m, X, "cpu", chunk=256)
    # (i) decoder direction aligns with the offset b_dec failed to absorb
    assert st["cos_offset"][0] > 0.99
    assert st["firing_rate"][0] == 1.0 and st["bias_fraction"][0] > 0.99

    # (ii) ablating it exposes the offset: the mean residual jumps by ~||v||
    after = ablate_and_measure(m, X, np.array([0]), "cpu", chunk=256)
    assert after["resid_mean_norm"] > st["resid_mean_norm"] + 20.0


def test_ablation_changes_mean_residual_by_the_dc_contribution():
    """Direction-agnostic: removing an always-on constant latent must move the
    MEAN residual by about the fixed vector it was contributing."""
    m = _plant_corrector(_normal_model())
    X = torch.randn(2000, 16)
    before = scan_features(m, X, "cpu", chunk=256)
    after = ablate_and_measure(m, X, np.array([0]), "cpu", chunk=256)
    shift = abs(after["resid_mean_norm"] - before["resid_mean_norm"])
    assert shift > 0.5 * before["dc_contribution"][0]


def test_bins_partition_every_feature_exactly_once():
    rng = np.random.default_rng(0)
    fr = rng.uniform(0, 1, size=500)
    fr[:5] = 0.0                                        # dead features
    fr[5] = 1.0                                         # fully saturated
    rows = bin_counts(fr)
    assert sum(r["n_features"] for r in rows) == len(fr)
    assert BIN_EDGES[-1] > 1.0                          # 1.0 must land in a bin


def test_scan_one_end_to_end_flags_and_reports(tmp_path):
    """Integration: a saved checkpoint with a planted corrector is flagged, the
    detail/bin/ablation reporting is populated, and it survives JSON."""
    m = _plant_corrector(_normal_model())
    ck = tmp_path / "seed_0"
    ck.mkdir()
    torch.save(m.state_dict(), ck / "model.pt")

    cfg = {"model": {"d_model": 16}, "sae": {"dict_size": 64, "k": 5}}
    X = torch.randn(1500, 16)
    res = scan_one(ck, X, cfg, "cpu", min_firing=0.5, min_bias=0.5)

    assert res["n_flagged"] == 1
    assert res["flagged_features"] == [0]
    assert res["flagged_detail"][0]["feature"] == 0
    assert res["flagged_detail"][0]["bias_fraction"] > 0.99
    assert "ablation" in res
    # the corrector lives in the top firing-rate bin -- the sparse one
    top = res["bins"][-1]
    assert top["n_flagged"] == 1
    assert sum(b["n_features"] for b in res["bins"]) == 64
    json.dumps(res)                                     # must be serializable


def test_config_thresholds_present():
    """The gate's thresholds are experiment-defining -- they must live in config."""
    cfg = yaml.safe_load(open("configs/mvp.yaml"))
    cs = cfg["confound_control"]["corrector_scan"]
    assert 0.0 < cs["min_firing_rate"] <= 1.0
    assert 0.0 < cs["min_bias_fraction"] <= 1.0
