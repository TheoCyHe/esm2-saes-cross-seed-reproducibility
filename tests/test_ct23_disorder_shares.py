"""§CT23.3's counting. A toy with the answer written down (gotcha 32b), and
the record gate must refuse a one-count difference (gotcha 13)."""
import numpy as np

from src.analysis import ct23_disorder_shares as D


def _toy():
    # 1 seed, 4 features, 3 pool concepts; Disordered is pool index 0.
    ap = np.array([[[0.90, 0.10, 0.10],     # raw AP -> 0; G -> 0 (aligned)
                    [0.60, 0.50, 0.10],     # raw AP -> 0; G -> 1 (p0 high)
                    [0.20, 0.10, 0.30],     # raw AP -> 2; G -> 2
                    [0.50, 0.01, 0.01]]],   # raw AP -> 0; G -> 0 but < floor
                  dtype=np.float32)
    prev = np.array([0.45, 0.01, 0.01])
    z = np.array([[0, 1, -1, 2]])           # regression's argmax in pool indices
    return ap, prev, z


def test_counts_on_a_toy():
    ap, prev, z = _toy()
    # G rows: [0.818, .09, .09], [0.273, .495, .09], [-.45, .09, .293], [.09, 0, 0]
    n = D.shares(ap, prev, z, dis=0, floor=0.1)
    assert n == {"n_cells": 4, "n_aligned": 3,
                 "gain_dis_all": 2, "gain_dis_aligned": 1,
                 "z_dis_all": 1, "z_dis_aligned": 1,
                 "ap_dis_all": 3, "ap_dis_aligned": 2}


def test_record_gate_refuses_a_one_count_difference():
    good = dict(D.RECORD)
    assert D.check_record(good) == []
    bad = dict(good, z_dis_all=2_147)
    assert D.check_record(bad) == ["z_dis_all: 2147 != recorded 2148"]
    del bad["n_aligned"]
    assert any("n_aligned" in m for m in D.check_record(bad))


def test_recorded_figures_are_the_record_s():
    assert D.RECORD == {"n_aligned": 48152, "gain_dis_aligned": 1183,
                        "gain_dis_all": 3856, "z_dis_all": 2148,
                        "ap_dis_all": 148298}
