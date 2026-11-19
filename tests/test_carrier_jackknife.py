"""Tests for src/analysis/carrier_jackknife.py (amendment CT, §CT19)."""
import numpy as np
import pytest

from src.analysis.carrier_jackknife import (excess_over, intervals, pair_mask,
                                            per_pair_cell_stats)
from src.analysis.carrier_support import carrier_rates_by_cell


def _toy(seed=3, n_seeds=5, n_feat=40, n_conc=6, n_cells=4):
    """A random but fully specified carrier problem: MNN partners symmetric by
    construction (partner[s,t,i] = j  <=>  partner[t,s,j] = i)."""
    rng = np.random.default_rng(seed)
    partner = np.full((n_seeds, n_seeds, n_feat), -1, dtype=np.int32)
    for s in range(n_seeds):
        for t in range(s + 1, n_seeds):
            perm = rng.permutation(n_feat)
            hit = rng.random(n_feat) < 0.5
            for i in range(n_feat):
                if hit[i]:
                    partner[s, t, i] = perm[i]
                    partner[t, s, perm[i]] = i
    carrier = rng.integers(0, n_feat, size=(n_seeds, n_conc))
    detected = rng.random((n_seeds, n_conc)) < 0.8
    cell = rng.integers(0, n_cells, size=(n_seeds, n_feat))
    return carrier, detected, partner, cell, n_cells


def test_per_pair_sums_reproduce_carrier_rates_by_cell_exactly():
    c, d, p, cell, nc = _toy()
    ref = carrier_rates_by_cell(c, d, p, cell, nc)
    st = per_pair_cell_stats(c, d, p, cell, nc)
    tot = excess_over(st, np.ones(len(st["pairs"]), dtype=bool))
    assert tot["rate_A"] == ref["rate_A_carrier_has_any_partner"]
    assert tot["matched_ambient"] == ref["matched_ambient"]
    assert tot["excess_pp"] == ref["excess_A_minus_matched_pp"]


def test_leave_one_out_equals_a_rerun_on_the_remaining_seeds():
    """THE CLAIM THE MODULE RESTS ON: removing seed d from the per-pair sums is
    the same as running carrier_rates_by_cell on the other seeds."""
    c, d, p, cell, nc = _toy()
    st = per_pair_cell_stats(c, d, p, cell, nc)
    for out in range(c.shape[0]):
        keep = [i for i in range(c.shape[0]) if i != out]
        ref = carrier_rates_by_cell(c[keep], d[keep], p[np.ix_(keep, keep)],
                                    cell[keep], nc)
        got = excess_over(st, pair_mask(st["pairs"], keep=set(keep)))
        assert got["excess_pp"] == pytest.approx(ref["excess_A_minus_matched_pp"], abs=1e-12)
        assert got["n_cells_evaluated"] == ref["n_concept_seedpair_cells_evaluated"]


def test_pair_mask_selects_the_right_pairs():
    pairs = [(0, 1), (0, 2), (1, 2)]
    assert pair_mask(pairs, keep={0, 2}).tolist() == [False, True, False]
    assert pair_mask(pairs, need=1).tolist() == [True, False, True]


def test_intervals_carry_three_named_intervals_and_pick_the_widest():
    c, d, p, cell, nc = _toy()
    st = per_pair_cell_stats(c, d, p, cell, nc)
    iv = intervals(st, c.shape[0])
    assert set(iv["intervals_95_pp"]) == {"jackknife", "across_seed", "cell_independence"}
    w = iv["intervals_95_pp"][iv["widest"]]
    assert all(w[1] - w[0] >= v[1] - v[0] for v in iv["intervals_95_pp"].values())
    assert len(iv["jackknife"]["loo_excess_pp"]) == c.shape[0]
    assert len(iv["across_seed"]["per_anchor_excess_pp"]) == c.shape[0]


def test_cli_requires_the_triple_arm_inputs_unless_single_only(monkeypatch, tmp_path):
    """amendment CT §CT24-F item 8: --single-only drops the covariate inputs; without
    it a missing triple-arm input is refused before anything loads."""
    import sys
    import pytest
    from src.analysis import carrier_jackknife as cj
    base = ["carrier_jackknife", "--ap", "a", "--null", "n", "--counts-reference", "c",
            "--out", str(tmp_path / "o.json"), "--ckpts", "k"]
    monkeypatch.setattr(sys, "argv", base)
    with pytest.raises(SystemExit) as e:
        cj.main()
    assert e.value.code == 2                       # argparse's error exit
    called = {}
    monkeypatch.setattr(cj, "run", lambda args: called.setdefault("args", args))
    monkeypatch.setattr(sys, "argv", base + ["--single-only", "--expect-n", "18"])
    cj.main()
    a = called["args"]
    assert a.single_only and a.expect_n == 18 and a.reference is None and a.d_fs is None
