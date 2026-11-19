"""PREREGISTRATION §CT22.5 -- the descriptive diagnostics' arithmetic.

The rotation diagnostic must report 0 beyond the best single partner when the
other seed holds the SAME features, and a positive amount when it holds a
ROTATED basis of the same subspace (every direction captured, no single
partner complete) -- the planted case, so the instrument is shown able to
produce both answers (gotcha 28).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis import concept_subspace as cs
from src.analysis import concept_subspace_diag as dg


def _world(kind: str):
    rng = np.random.default_rng(11)
    d, n = 12, 10
    dec0 = cs.unit_rows(rng.standard_normal((n, d)))
    dec1 = cs.unit_rows(rng.standard_normal((n, d)))
    if kind == "same":
        dec1[:3] = dec0[:3]
    elif kind == "rotated":
        q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        dec1[:3] = q @ dec0[:3]               # same span; orthogonal mix keeps unit-ish
        dec1[:3] = cs.unit_rows(dec1[:3])
    ap = np.zeros((2, n, 2))
    ap[:, :3, 0] = [0.9, 0.8, 0.7]
    ap[:, 3:6, 1] = [0.9, 0.8, 0.7]
    sim = np.full((2, 2, n), -np.inf, dtype=np.float32)
    return {"ap": ap, "dec": np.stack([dec0, dec1]),
            "strat": np.zeros((2, n), dtype=np.int64), "sim": sim,
            "row_of": np.array([0, 1]), "col_of": np.array([0, 1]),
            "argmax": np.zeros((2, n), dtype=np.int64)}


@pytest.fixture
def world():
    saved = dict(cs._W)
    yield
    cs._W.clear()
    cs._W.update(saved)


def test_rotation_is_zero_for_identical_features(world):
    cs._W.clear(); cs._W.update(_world("same"))
    r = dg.task_rotation((0, 0, 0, 1, 3))
    assert r["captured"] == pytest.approx(1.0, abs=1e-12)
    assert r["best_single"] == pytest.approx(1.0, abs=1e-12)
    assert r["beyond_best"] == pytest.approx(0.0, abs=1e-12)


def test_rotation_is_positive_for_a_rotated_basis(world):
    cs._W.clear(); cs._W.update(_world("rotated"))
    r = dg.task_rotation((0, 0, 0, 1, 3))
    assert r["captured"] == pytest.approx(1.0, abs=1e-12)
    assert r["best_single"] < 0.95
    assert r["beyond_best"] > 0.05


def test_rotation_is_small_for_unrelated_sets_and_never_negative(world):
    cs._W.clear(); cs._W.update(_world("unrelated"))
    r = dg.task_rotation((0, 0, 0, 1, 3))
    assert r["captured"] < 0.6
    assert r["beyond_best"] >= -1e-12


def test_other_concept_within_and_cross(world):
    cs._W.clear(); cs._W.update(_world("same"))
    w = dg.task_other_concept((0, 0, 0, 1, 1, 3, "within"))
    assert w["n_shared_features"] == 0 and w["omega"] < 1.0
    x = dg.task_other_concept((0, 0, 0, 1, 0, 3, "cross"))   # c_other = c: control
    assert x["omega"] == pytest.approx(1.0, abs=1e-12)
    # feature indices are not identities across seeds (gotcha 18): no count
    assert x["n_shared_features"] is None and w["n_shared_features"] == 0


def test_no_disorder_excludes_from_F_and_from_the_null(world):
    w = _world("same")
    keep = np.ones((2, 10), bool)
    keep[:, 0] = False                         # feature 0 is "Disordered"
    cs._W.clear(); cs._W.update(w)
    dg._D["not_disordered"] = keep
    r = dg.task_no_disorder((0, 0, 0, 1, 3))
    assert r["alignment_ok"] and r["feasible"]
    # F = features 1, 2 and the next eligible by AP (index 3, AP 0 -> lowest index)
    Fs = cs.top_m(w["ap"][0, :, 0], 3, keep[0])
    assert 0 not in Fs.tolist()
    strat = w["strat"][0].copy(); strat[~keep[0]] = -1
    pools = cs.strata_pools(strat, Fs)
    assert all(0 not in pool.tolist() for _, pool in pools)


def test_the_diag_launcher_command_parses():
    import re
    from pathlib import Path
    text = Path("slurm/concept_subspace_diag.sbatch").read_text(encoding="utf-8")
    block = text[text.index("EXTRA=()"):text.rindex("if [[")]
    flags = set(re.findall(r"(--[a-z][a-z0-9-]+)", block))
    p = cs.build_parser()
    p.add_argument("--diagnostics", nargs="+")
    known = {a for act in p._actions for a in act.option_strings}
    assert "--pilot" in flags and "--diagnostics" in flags and flags <= known, flags - known


def test_gain_argmax_mask_uses_the_gain_not_raw_ap():
    """Raw AP's argmax and the gain's differ when prevalences differ: concept 1
    has the higher AP but, at prevalence 0.5, the lower gain."""
    ap = np.array([[[0.30, 0.60]], [[0.30, 0.20]]])          # (seed, feature, concept)
    prev = np.array([0.01, 0.50])
    keep, share = dg.gain_argmax_mask(ap, prev, concept=1)
    # seed 0: G = (0.293, 0.2) -> argmax 0; seed 1: G = (0.293, -0.6) -> argmax 0
    assert keep.tolist() == [[True], [True]] and share == 0.0
    keep0, share0 = dg.gain_argmax_mask(ap, prev, concept=0)
    assert keep0.tolist() == [[False], [False]] and share0 == 1.0


def test_ap_argmax_mask_follows_raw_ap():
    """The same array as the gain test: raw AP's argmax is concept 1 in seed 0
    (0.60 > 0.30) and concept 0 in seed 1; the gain's is concept 0 in both."""
    ap = np.array([[[0.30, 0.60]], [[0.30, 0.20]]])
    keep, share = dg.ap_argmax_mask(ap, concept=1)
    assert keep.tolist() == [[False], [True]] and share == 0.5
