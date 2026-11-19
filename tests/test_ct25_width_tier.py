"""§CT25's scoring, total over every outcome (gotcha 24), with expectations
written down (gotcha 32b)."""
import math

import pytest

from src.analysis import ct25_width_tier as T

W = T.WIDTHS


def _e(*v):
    return dict(zip(W, v))


def test_p1_hit_miss_unmeasurable():
    assert T.score_p1(_e(0.2, 0.3, -0.1, 0.9)) == "HIT"
    assert T.score_p1(_e(0.2, 1.01, -0.1, 0.9)) == "MISS"
    assert T.score_p1(_e(0.2, 0.3, -0.5, 1.0)) == "HIT"            # closed band
    assert T.score_p1(_e(0.2, 0.3, None, 0.9)) == "UNMEASURABLE"
    assert T.score_p1(_e(-0.6, 0.3, None, None)) == "MISS"         # already outside


def test_r1_not_monotone():
    assert T.score_r1(_e(0.1, 0.2, 0.3, 0.4)) == "MISS"
    assert T.score_r1(_e(0.4, 0.3, 0.2, 0.1)) == "MISS"
    assert T.score_r1(_e(0.1, 0.3, 0.2, 0.4)) == "HIT"
    assert T.score_r1(_e(0.1, 0.1, 0.2, 0.3)) == "HIT"             # a tie is not strict
    assert T.score_r1(_e(0.1, 0.3, 0.2, None)) == "HIT"            # 3 already non-monotone
    assert T.score_r1(_e(0.1, 0.2, 0.3, None)) == "UNMEASURABLE"


def test_cs1_class_branches():
    sds = [0.1, 0.1, 0.1, 0.1]                                     # s = 0.1, band 0.2
    assert T.cs1_class([1.0, 0.7, 0.4, 0.1], sds)["class"] == "MONOTONE_DOWN"
    assert T.cs1_class([0.1, 0.4, 0.7, 1.0], sds)["class"] == "MONOTONE_UP"
    assert T.cs1_class([0.1, 0.2, 0.1, 0.2], sds)["class"] == "FLAT"
    assert T.cs1_class([0.1, 0.5, 0.1, 0.2], sds)["class"] == "NON_MONOTONE"
    assert T.cs1_class([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4])["s"] == pytest.approx(
        math.sqrt((0.01 + 0.04 + 0.09 + 0.16) / 4))


def test_sd_recovered_from_the_across_seed_interval():
    # mean 0.3, SD 0.5 over n = 10: interval 0.3 +/- 1.959964 * 0.5 / sqrt(10)
    h = T.Z * 0.5 / math.sqrt(10)
    assert T.sd_from_across_seed([0.3 - h, 0.3 + h], 10) == pytest.approx(0.5)


def test_r1_two_equal_measured_widths_already_decide_it():
    assert T.score_r1(_e(0.2, None, 0.2, None)) == "HIT"
    assert T.score_r1(_e(0.2, None, 0.3, None)) == "UNMEASURABLE"
