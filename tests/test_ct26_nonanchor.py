"""§CT26's own pieces. The cell functions are §CT22's and are tested there; the
real end-to-end check is the anchor control on Bouchet. Expectations are
written down (gotcha 32b), and every check can fail (gotcha 13)."""
import json

import numpy as np

from src.analysis import ct26_nonanchor as N


def test_enumerate_cells_pool_order_numeric_pairs():
    # AP order is lexicographic: labels 20, 21, 3 -> numeric order 3, 20, 21
    labels = [20, 21, 3]
    det = np.array([[1, 1, 0, 1],
                    [1, 0, 0, 1],
                    [1, 1, 1, 0]], dtype=bool)
    pop, cells = N.enumerate_cells(det, labels)
    assert pop == [0, 1, 3]                    # concept 2 is detected once
    # concept 0: pairs (3,20), (3,21), (20,21) -> AP indices (2,0), (2,1), (0,1)
    assert cells[:3] == [(0, 0, 2, 0), (1, 0, 2, 1), (2, 0, 0, 1)]
    # concept 1: seeds 20 and 3 -> (3,20) only
    assert cells[3] == (3, 1, 2, 0)
    # concept 3: seeds 20 and 21
    assert cells[4] == (4, 3, 0, 1)
    assert len(cells) == 5


def _res(p1, p2, exc, sd0=0.0):
    return {"accounting": {"share_null_sd_zero": sd0, "share_rank_deficient": 0.0,
                           "share_draw_infeasible": 0.0},
            "per_concept": [{"median_exceedance": e} for e in exc] + [{"concept": "x"}],
            "rotation": {"beyond_best": {"all": {"q50": p1}}},
            "cross_concept": {"omega": {"q50": p2}}}


def test_score_bars_are_strict_and_total():
    s = N.score(_res(0.049, 0.0099, [1.0] * 10))
    assert [s[k]["outcome"] for k in ("CT26_P1", "CT26_P2", "CT26_P3")] == ["HIT"] * 3
    s = N.score(_res(0.05, 0.01, [1.0] * 9 + [0.9]))           # P3 share 0.9, not > 0.9
    assert [s[k]["outcome"] for k in ("CT26_P1", "CT26_P2", "CT26_P3")] == ["MISS"] * 3
    assert s["CT26_P3"]["n_concepts_with_a_measurable_cell"] == 10
    s = N.score(_res(0.01, 0.001, [1.0] * 10, sd0=0.06))
    assert s["CT26_P1"]["outcome"] == "UNMEASURABLE" and s["unmeasurable_causes"]


def test_control_compare_detects_a_one_ulp_difference():
    res = {"per_concept": [{"a": 0.1}], "accounting": {"n": 1},
           "rotation": {"x": 0.2}, "cross_concept": {"omega": {"q50": 0.3}}}
    confirm = {"per_concept": [{"a": 0.1}], "accounting": {"n": 1}}
    diag = {"rotation": {"x": 0.2}, "cross_concept": {"omega": {"q50": 0.3}}}
    assert N.control_compare(res, confirm, diag)["verdict"] == "EXACT"
    diag2 = json.loads(json.dumps(diag))
    diag2["cross_concept"]["omega"]["q50"] = np.nextafter(0.3, 1.0)
    c = N.control_compare(res, confirm, diag2)
    assert c["verdict"] == "DIFFERS" and c["cross_concept_omega"] is False


def test_cross_concept_draw_never_picks_the_same_concept_and_is_reproducible():
    all_cells = [(0, 0, 0, 1), (1, 1, 0, 1), (2, 2, 0, 1), (3, 2, 1, 2)]
    t1 = N.cross_concept_tasks(all_cells, all_cells, 3)
    t2 = N.cross_concept_tasks(all_cells, all_cells, 3)
    assert t1 == t2
    # the last cell's other seed (2) has only concept 2 detected: skipped, None
    assert t1[3][4] is None
    for (i, c, s, u, co, m, kind) in t1[:3]:
        assert co is not None and co != c and kind == "cross" and m == 3


def test_parser_has_both_modes():
    p = N.build_parser()
    a = p.parse_args(["run", "--ap", "a", "--null-reductions", "b", "--sim", "c",
                      "--out", "d", "--ckpts", "e"])
    assert a.cmd == "run" and a.exclude_concepts is None
    a = p.parse_args(["match", "--ckpts", "x", "--compare-ckpts", "y",
                      "--stored-sim", "z", "--out", "o"])
    assert a.cmd == "match"
