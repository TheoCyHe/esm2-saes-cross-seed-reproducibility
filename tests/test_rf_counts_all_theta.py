"""Amendment AR route A -- per-anchor counts across the WHOLE theta grid.

WHY THIS EXISTS. `--emit-counts` wrote `match_count` at the reference theta
only, so `r_fs` -- design 3's outcome variable -- existed at theta = 0.70 and
nowhere else, while AB1 decision A and AK6's reporting standard require the
whole qualifying theta range. `--emit-counts-all-theta` closes that, and this
file is the evidence that it closes it WITHOUT moving the single-theta
artifact every downstream consumer already reads.

THE FIXTURE HAS TO DISCRIMINATE BEFORE ANY RESULT IS ASSERTED. A random
ensemble at d = 32 has a maximum cross-seed cosine near 0.5, so counts at
theta = 0.70 and at 0.75 would BOTH be all-zero, and a writer emitting one
under the other's key would survive every test here. That is gotcha 16's
fixture lesson (a fixture that cannot exhibit the disputed condition cannot
test it) and gotcha 25's "can only pass". `_write_graded_ensemble` therefore
plants a GRADED cosine spectrum straddling both thresholds, and
`test_the_all_theta_fixture_discriminates` asserts the two slices genuinely
differ before the comparator is trusted to notice.

THE SOURCE-LEVEL MUTATION HAS BEEN RUN, NOT REASONED ABOUT. Rewriting the
all-theta writer to emit the 0.75 counts under the 0.70 key --

    counts_at_theta(mnn_sim, off, n_anchors, t)
 -> counts_at_theta(mnn_sim, off, n_anchors,
                    0.75 if abs(t - 0.70) < 1e-9 else t)

-- and running this file is recorded in the amendment AR section of
PREREGISTRATION.md together with which tests died.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.analysis.rf_marginal import (
    THETA_GRID,
    THETA_LITERATURE,
    compare_counts_to_reference,
    counts_at_theta,
)
from tests.test_rf_marginal import _meta, _run_main, _unit


def _write_graded_ensemble(root: Path, n_seeds=4, d=64, n=48, seed=7):
    """An ensemble whose cross-seed MNN cosines SPAN the theta grid.

    Every seed is a noisy copy of one shared basis with a PER-COLUMN noise
    level, so column j is its own mutual nearest neighbour in every seed at a
    cosine of roughly `1 / (1 + sigma_j^2)`. Sweeping sigma across the columns
    puts matches on both sides of 0.70 and of 0.75, which is the only way a
    test can tell those two thetas apart.
    """
    g = torch.Generator().manual_seed(seed)
    base = _unit(torch.randn(d, n, generator=g, dtype=torch.float64))
    sigmas = torch.linspace(0.15, 1.10, n, dtype=torch.float64)
    for s in range(n_seeds):
        noise = torch.randn(d, n, generator=g, dtype=torch.float64) / np.sqrt(d)
        W = _unit(base + sigmas[None, :] * noise).to(torch.float32)
        p = root / f"seed_{s}"
        p.mkdir(parents=True)
        torch.save({"W_dec": W}, p / "model.pt")
        (p / "meta.json").write_text(json.dumps(
            _meta(s, dict_size=n, d_model=d)))
    return [root / f"seed_{s}" for s in range(n_seeds)]


def _read(path, name):
    import h5py
    with h5py.File(path, "r") as hf:
        return hf[name][:]


# ---------------------------------------------------------------------------
# The precondition
# ---------------------------------------------------------------------------

def test_the_all_theta_fixture_discriminates(tmp_path):
    """Asserted BEFORE any verdict is believed: the 0.70 and 0.75 slices are
    not the same array on this fixture."""
    ckpts = _write_graded_ensemble(tmp_path)
    allt = tmp_path / "all.h5"
    r = _run_main(ckpts, tmp_path / "rf.json",
                  extra=["--emit-counts-all-theta", str(allt)])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    c70 = _read(allt, "match_count_theta_0.70")
    c75 = _read(allt, "match_count_theta_0.75")
    assert c70.sum() > c75.sum() > 0, (int(c70.sum()), int(c75.sum()))
    assert not np.array_equal(c70, c75)


# ---------------------------------------------------------------------------
# The artifact
# ---------------------------------------------------------------------------

def test_all_theta_writes_every_grid_point_with_its_own_attrs(tmp_path):
    import h5py
    ckpts = _write_graded_ensemble(tmp_path)
    allt = tmp_path / "all.h5"
    assert _run_main(ckpts, tmp_path / "rf.json",
                     extra=["--emit-counts-all-theta",
                            str(allt)]).returncode == 0
    with h5py.File(allt, "r") as hf:
        assert len(list(hf.keys())) == len(THETA_GRID)
        assert json.loads(hf.attrs["theta_grid"]) == THETA_GRID
        assert int(hf.attrs["n_compare_seeds"]) == 3        # 4 dictionaries
        assert json.loads(hf.attrs["seed_labels"]) == [0, 1, 2, 3]
        assert str(hf.attrs["similarity"]) == "decoder_cosine"
        for t in THETA_GRID:
            d = hf[f"match_count_theta_{t:.2f}"]
            assert abs(float(d.attrs["theta"]) - t) < 1e-12
            assert int(d.attrs["n_compare_seeds"]) == 3
            assert str(d.attrs["similarity"]) == "decoder_cosine"
            assert d[:].shape == (4, 48)
            assert int(d[:].max()) <= 3


def test_all_theta_counts_are_monotone_non_increasing_in_theta(tmp_path):
    """Raising the threshold can only remove matches -- an analytic identity,
    so a violation is an implementation bug and not a result."""
    ckpts = _write_graded_ensemble(tmp_path)
    allt = tmp_path / "all.h5"
    assert _run_main(ckpts, tmp_path / "rf.json",
                     extra=["--emit-counts-all-theta",
                            str(allt)]).returncode == 0
    prev = None
    for t in THETA_GRID:
        c = _read(allt, f"match_count_theta_{t:.2f}")
        if prev is not None:
            assert np.all(c <= prev), t
        prev = c


# ---------------------------------------------------------------------------
# The acceptance criterion (amendment AR), and its mutation
# ---------------------------------------------------------------------------

def test_all_theta_reference_slice_is_bit_identical(tmp_path):
    ckpts = _write_graded_ensemble(tmp_path)
    one, allt = tmp_path / "one.h5", tmp_path / "all.h5"
    assert _run_main(ckpts, tmp_path / "rf.json", extra=[
        "--emit-counts", str(one),
        "--emit-counts-all-theta", str(allt)]).returncode == 0
    ref = _read(one, "match_count")
    new = _read(allt, f"match_count_theta_{THETA_LITERATURE:.2f}")
    assert ref.dtype == new.dtype
    assert np.array_equal(ref, new)
    chk = compare_counts_to_reference(new, one, THETA_LITERATURE, 3,
                                      [0, 1, 2, 3])
    assert chk["verdict"] == "BIT_IDENTICAL", chk
    assert chk["n_cells_differing"] == 0
    assert chk["identity_problems"] == []


def test_comparator_rejects_the_wrong_theta_slice(tmp_path):
    """THE MUTATION at function scope: hand the comparator the 0.75 counts
    under the 0.70 key and require DIFFERS."""
    ckpts = _write_graded_ensemble(tmp_path)
    one, allt = tmp_path / "one.h5", tmp_path / "all.h5"
    assert _run_main(ckpts, tmp_path / "rf.json", extra=[
        "--emit-counts", str(one),
        "--emit-counts-all-theta", str(allt)]).returncode == 0
    wrong = _read(allt, "match_count_theta_0.75")
    chk = compare_counts_to_reference(wrong, one, THETA_LITERATURE, 3,
                                      [0, 1, 2, 3])
    assert chk["verdict"] == "DIFFERS", chk
    assert chk["bit_identical"] is False
    assert chk["n_cells_differing"] > 0
    # the identity was fine; only the DATA differed, which is the distinction
    # the two failure verdicts exist to keep separate.
    assert chk["identity_problems"] == []


def test_comparator_reports_an_absent_key_as_absent(tmp_path):
    """Gotcha 15 / I3: absent is not agreement and must not read as one."""
    import h5py
    p = tmp_path / "ref.h5"
    data = np.zeros((2, 3), np.int16)
    with h5py.File(p, "w") as hf:
        hf.create_dataset("match_count", data=data)
        hf.attrs["theta"] = 0.70
        # n_compare_seeds / similarity / rule / seed_labels deliberately ABSENT
    chk = compare_counts_to_reference(data, p, 0.70, 3, [0, 1])
    assert chk["verdict"] == "IDENTITY_MISMATCH"
    assert chk["reference_identity"]["n_compare_seeds"] == "<ABSENT>"
    assert any("ABSENT" in s for s in chk["identity_problems"])


def test_comparator_refuses_a_reference_from_a_different_arm(tmp_path):
    """The ARRAYS agree exactly and the identity does not. A bit-identity
    check that ignored identity would accept the wrong `n_compare` arm."""
    import h5py
    p = tmp_path / "ref.h5"
    data = np.zeros((2, 3), np.int16)
    with h5py.File(p, "w") as hf:
        hf.create_dataset("match_count", data=data)
        hf.attrs["theta"] = 0.70
        hf.attrs["n_compare_seeds"] = 19
        hf.attrs["similarity"] = "decoder_cosine"
        hf.attrs["rule"] = "mnn"
        hf.attrs["seed_labels"] = json.dumps([0, 1])
    chk = compare_counts_to_reference(data, p, 0.70, 39, [0, 1])
    # `bit_identical` stays a pure statement about the ARRAYS -- it is a diff,
    # not a verdict -- so it is True here and the VERDICT is what refuses.
    assert chk["bit_identical"] is True
    assert chk["verdict"] == "IDENTITY_MISMATCH"
    assert any("n_compare_seeds" in s for s in chk["identity_problems"])


# ---------------------------------------------------------------------------
# The entry point, and the exit path of its verdict
# ---------------------------------------------------------------------------

def test_main_accepts_a_matching_counts_reference(tmp_path):
    ckpts = _write_graded_ensemble(tmp_path)
    one, allt = tmp_path / "one.h5", tmp_path / "all.h5"
    assert _run_main(ckpts, tmp_path / "rf.json",
                     extra=["--emit-counts", str(one)]).returncode == 0
    r = _run_main(ckpts, tmp_path / "rf2.json", extra=[
        "--emit-counts-all-theta", str(allt),
        "--counts-reference", str(one)])
    assert r.returncode == 0, r.stdout[-4000:]
    assert "acceptance criterion PASS" in r.stdout


def test_main_exits_nonzero_when_the_reference_disagrees(tmp_path):
    """Gotcha 13 extension 1: the verdict must be readable by its consumer."""
    import h5py
    ckpts = _write_graded_ensemble(tmp_path)
    one, allt = tmp_path / "one.h5", tmp_path / "all.h5"
    assert _run_main(ckpts, tmp_path / "rf.json",
                     extra=["--emit-counts", str(one)]).returncode == 0
    with h5py.File(one, "r+") as hf:
        hf["match_count"][0, 0] = int(hf["match_count"][0, 0]) + 1
    r = _run_main(ckpts, tmp_path / "rf2.json", extra=[
        "--emit-counts-all-theta", str(allt),
        "--counts-reference", str(one)])
    assert r.returncode == 1, r.stdout[-4000:]
    assert "ACCEPTANCE CRITERION FAILED" in r.stdout


def test_counts_reference_without_all_theta_refuses(tmp_path):
    """A check with nothing to check must not report success (gotcha 28)."""
    ckpts = _write_graded_ensemble(tmp_path)
    one = tmp_path / "one.h5"
    assert _run_main(ckpts, tmp_path / "rf.json",
                     extra=["--emit-counts", str(one)]).returncode == 0
    r = _run_main(ckpts, tmp_path / "rf2.json",
                  extra=["--counts-reference", str(one)])
    assert r.returncode == 1, r.stdout[-2000:]


# ---------------------------------------------------------------------------
# The helper itself
# ---------------------------------------------------------------------------

def test_counts_at_theta_ignores_non_anchor_rows():
    """A comparison-only row is never read, so it must not enter as r_f = 0."""
    n_seeds, n_anchors, dict_size = 4, 2, 5
    mnn = np.full((n_seeds, n_seeds, dict_size), -np.inf, dtype=np.float32)
    mnn[0, 1] = 0.9
    mnn[2, 3] = 0.9                      # a non-anchor row, deliberately full
    off = ~np.eye(n_seeds, dtype=bool)
    is_anchor = np.zeros((n_seeds, 1), dtype=bool)
    is_anchor[:n_anchors] = True
    c = counts_at_theta(mnn, off & is_anchor, n_anchors, 0.70)
    assert c.shape == (n_anchors, dict_size)
    assert c[0].tolist() == [1] * dict_size
    assert c[1].tolist() == [0] * dict_size
