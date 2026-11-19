"""Tests for M1 / `r_f` marginals (PREREGISTRATION amendment X).

GOTCHA 13 IS THE POINT OF THIS FILE. A matching implementation that silently
returned ONE-WAY results under an "MNN" label would produce a complete table
of believable numbers -- every rate plausible, every histogram well formed,
and the design conclusion wrong. So the central test plants a KNOWN mutual
pair and a KNOWN one-way-only pair and requires MNN to accept the first and
REJECT the second.

Each of the mutation kills below has been RUN, not merely asserted:

  * replacing `torch.where(mutual_a, a_best_sim, neg)` with `a_best_sim`
    (i.e. MNN silently becoming one-way) -> kills
    test_mnn_rejects_the_one_way_only_pair and
    test_planted_pairs_distinguish_the_rules.
  * dropping the firewall's token check -> kills test_firewall_blocks_*.
  * making `verify_ensemble_identity` treat an ABSENT key as agreement ->
    kills test_identity_absent_key_is_not_agreement.
  * `>` -> `>=` in the endpoint-share comparison -> kills
    test_select_theta_boundary_is_inclusive.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from src.analysis.rf_marginal import (
    ENDPOINT_MAX_SHARE,
    assert_written_denominator,
    verify_compare_denominator,
    THETA_GRID,
    THETA_LITERATURE,
    FirewallBreach,
    check_unit_norm,
    endpoint_purity,
    hungarian_subsample,
    install_firewall,
    iou,
    marginals,
    pair_statistics,
    rotation_null,
    select_theta,
    verify_ensemble_identity,
)


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=0, keepdim=True)


def _planted_pair(d: int = 64, n: int = 40, seed: int = 0):
    """Build a decoder pair with a planted MUTUAL pair and a planted
    ONE-WAY-ONLY pair, so the two rules must disagree in a known place.

    index 0 : identical column in both  -> mutual at cos 1.0
    index 2 : identical column to Wb[:,1] -> mutual at cos 1.0
    index 1 : cos 0.8 with Wb[:,1], but Wb[:,1]'s best is Wa[:,2] at 1.0
              -> one-way match at 0.8, and NOT mutual
    """
    g = torch.Generator().manual_seed(seed)
    Wa = _unit(torch.randn(d, n, generator=g, dtype=torch.float64))
    Wb = _unit(torch.randn(d, n, generator=g, dtype=torch.float64))

    v = _unit(torch.randn(d, 1, generator=g, dtype=torch.float64))
    Wa[:, 0] = v[:, 0]
    Wb[:, 0] = v[:, 0]

    u = _unit(torch.randn(d, 1, generator=g, dtype=torch.float64))
    w = torch.randn(d, generator=g, dtype=torch.float64)
    w = w - (w @ u[:, 0]) * u[:, 0]
    w = w / w.norm()

    Wb[:, 1] = u[:, 0]
    Wa[:, 2] = u[:, 0]                       # steals b_1's argmax
    Wa[:, 1] = 0.8 * u[:, 0] + 0.6 * w       # unit norm; best partner is b_1
    return Wa, Wb


# ---------------------------------------------------------------------------
# The mutation test (gotcha 13)
# ---------------------------------------------------------------------------

def test_mnn_accepts_the_planted_mutual_pair():
    Wa, Wb = _planted_pair()
    st = pair_statistics(Wa, Wb)
    assert st["a_mnn"][0] == pytest.approx(1.0, abs=1e-9)
    assert st["a_mnn"][2] == pytest.approx(1.0, abs=1e-9)


def test_mnn_rejects_the_one_way_only_pair():
    """MNN must REJECT index 1; one-way must FIND it. This is the assertion
    that an MNN implementation silently doing one-way cannot pass."""
    Wa, Wb = _planted_pair()
    st = pair_statistics(Wa, Wb)
    assert st["a_mnn"][1] == float("-inf")
    assert st["a_oneway"][1] == pytest.approx(0.8, abs=1e-9)


def test_planted_pairs_distinguish_the_rules():
    Wa, Wb = _planted_pair()
    st = pair_statistics(Wa, Wb)
    theta = 0.7
    n_mnn = int((st["a_mnn"] >= theta).sum())
    n_oneway = int((st["a_oneway"] >= theta).sum())
    assert n_oneway == n_mnn + 1, (n_mnn, n_oneway)


def test_mnn_is_nested_inside_oneway_everywhere():
    """Amendment W1's nesting claim, as an executable check -- this is P1."""
    for seed in range(5):
        Wa, Wb = _planted_pair(seed=seed)
        st = pair_statistics(Wa, Wb)
        finite = torch.isfinite(st["a_mnn"])
        assert torch.all(st["a_mnn"][finite] <= st["a_oneway"][finite] + 1e-12)


def test_mnn_is_symmetric_across_the_pair():
    """A mutual pair must be mutual from both sides."""
    Wa, Wb = _planted_pair()
    fwd = pair_statistics(Wa, Wb)
    rev = pair_statistics(Wb, Wa)
    assert int(torch.isfinite(fwd["a_mnn"]).sum()) == int(
        torch.isfinite(rev["a_mnn"]).sum()
    )
    assert int(torch.isfinite(fwd["b_mnn"]).sum()) == int(
        torch.isfinite(rev["a_mnn"]).sum()
    )


def test_mnn_groups_are_not_equivalence_classes():
    """Amendment X5 states this as a known property. Demonstrate it holds, so
    the claim in the pre-registration is backed by a running example rather
    than by assertion: anchor a_2 matches b_1 mutually while a_1 -- whose best
    partner is also b_1 -- has no mutual partner at all."""
    Wa, Wb = _planted_pair()
    st = pair_statistics(Wa, Wb)
    assert torch.isfinite(st["a_mnn"][2])
    assert not torch.isfinite(st["a_mnn"][1])
    assert st["a_oneway"][1] > 0.7 and st["a_oneway"][2] > 0.7


# ---------------------------------------------------------------------------
# The firewall (amendment X7) -- verified to FIRE, not merely present
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "feature_ap_main.h5",
    "<cluster-fs>/x/analysis/null_pilot_g.h5",
    r"C:\proj\residue_labels_subtyped.h5",
    "feature_ap_all_pooled_orphan_screen.h5",
])
def test_firewall_blocks_alignment_artifacts(tmp_path, name):
    install_firewall()
    import h5py
    with pytest.raises(FirewallBreach):
        h5py.File(name, "r")


def test_firewall_allows_an_unrelated_file(tmp_path):
    install_firewall()
    import h5py
    p = tmp_path / "decoder_cosines.h5"
    with h5py.File(p, "w") as fh:
        fh.create_dataset("x", data=np.zeros(3))
    with h5py.File(p, "r") as fh:
        assert fh["x"].shape == (3,)


def test_firewall_is_idempotent():
    install_firewall()
    import h5py
    first = h5py.File
    install_firewall()
    assert h5py.File is first


# ---------------------------------------------------------------------------
# Pre-flight identity (amendment X8 / gotcha 15) -- each branch verified to FAIL
# ---------------------------------------------------------------------------

def _meta(seed: int, **over):
    m = {
        "seed": seed, "git_sha": "abc", "git_dirty": False, "config_hash": "cfg",
        "b_dec_init_sha256_16": "init", "dict_size": 128, "d_model": 16,
        "k": 4, "n_steps": 10, "lr": 1e-3, "batch_size": 8,
        # ONE shared init, DISTINCT finals -- the design's invariant. Distinct
        # by default so a test that wants a collision has to plant one.
        "b_dec_sha256_16": f"final{seed}",
    }
    m.update(over)
    return m


def test_identity_passes_on_a_clean_ensemble():
    rep = verify_ensemble_identity({s: _meta(s) for s in range(4)})
    assert rep["verdict"] == "PASS"
    assert rep["problems"] == []


def test_identity_fails_on_git_sha_disagreement():
    metas = {s: _meta(s) for s in range(4)}
    metas[3]["git_sha"] = "different"
    rep = verify_ensemble_identity(metas)
    assert rep["verdict"] == "FAIL"
    assert any("git_sha" in p for p in rep["problems"])


def test_identity_fails_on_b_dec_init_disagreement():
    metas = {s: _meta(s) for s in range(4)}
    metas[1]["b_dec_init_sha256_16"] = "other"
    rep = verify_ensemble_identity(metas)
    assert rep["verdict"] == "FAIL"
    assert any("b_dec_init" in p for p in rep["problems"])


def test_identity_absent_key_is_not_agreement():
    """Gotcha 15: an ABSENT key is reported absent, never treated as matching."""
    metas = {s: _meta(s) for s in range(4)}
    for s in metas:
        metas[s].pop("config_hash")
    rep = verify_ensemble_identity(metas)
    assert rep["keys"]["config_hash"]["status"] == "ABSENT"
    assert rep["verdict"] == "FAIL"


def test_identity_fails_on_dirty_provenance():
    metas = {s: _meta(s) for s in range(4)}
    metas[2]["git_dirty"] = True
    rep = verify_ensemble_identity(metas)
    assert rep["verdict"] == "FAIL"


def test_unit_norm_check_can_fail():
    W = torch.randn(8, 5, dtype=torch.float64)
    W = W / W.norm(dim=0, keepdim=True)
    assert check_unit_norm(W)["verdict"] == "PASS"
    W[:, 2] *= 1.5
    assert check_unit_norm(W)["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# theta selection (amendment X2) -- DERIVED verdicts
# ---------------------------------------------------------------------------

def _endpoints(**by_theta):
    return {float(k): {"frac_r_eq_0": v[0], "frac_r_eq_1": v[1]}
            for k, v in by_theta.items()}


def test_select_theta_takes_the_smallest_qualifying():
    ep = _endpoints(**{"0.3": (0.95, 0.0), "0.4": (0.5, 0.1),
                       "0.5": (0.4, 0.1), "0.6": (0.9, 0.0)})
    sel = select_theta(ep, floor=0.35)
    assert sel["verdict"] == "SELECTED"
    assert sel["selected_theta"] == 0.4


def test_select_theta_respects_the_floor():
    """A theta with lovely endpoints but below the floor must not be chosen."""
    ep = _endpoints(**{"0.3": (0.4, 0.1), "0.5": (0.4, 0.1)})
    sel = select_theta(ep, floor=0.45)
    assert sel["selected_theta"] == 0.5


def test_select_theta_reports_the_failure_branch():
    """Amendment X3: no qualifying theta is a legitimate OUTCOME, and the
    word is computed from the data rather than asserted."""
    ep = _endpoints(**{"0.3": (0.99, 0.0), "0.5": (0.995, 0.0),
                       "0.9": (1.0, 0.0)})
    sel = select_theta(ep, floor=0.2)
    assert sel["verdict"] == "NO_THETA_QUALIFIES"
    assert sel["selected_theta"] is None


def test_select_theta_boundary_is_inclusive():
    """Exactly ENDPOINT_MAX_SHARE is USABLE ('more than 80%' disqualifies)."""
    ep = _endpoints(**{"0.5": (ENDPOINT_MAX_SHARE, 0.0)})
    assert select_theta(ep, floor=0.2)["selected_theta"] == 0.5
    ep2 = _endpoints(**{"0.5": (ENDPOINT_MAX_SHARE + 1e-9, 0.0)})
    assert select_theta(ep2, floor=0.2)["selected_theta"] is None


def test_theta_grid_contains_the_literature_value():
    """Amendment X2 requires 0.70 be on the grid exactly."""
    assert THETA_LITERATURE in THETA_GRID
    assert THETA_GRID[0] == 0.30 and THETA_GRID[-1] == 0.90
    assert len(THETA_GRID) == 13


# ---------------------------------------------------------------------------
# Marginals
# ---------------------------------------------------------------------------

def test_marginals_atoms_and_endpoints():
    counts = np.array([[0, 0, 19, 10], [0, 19, 19, 5]], dtype=np.int16)
    m = marginals(counts, n_compare=19)
    assert m["n_anchors"] == 8
    assert sum(m["atom_counts_by_k"]) == 8
    assert m["frac_r_eq_0"] == pytest.approx(3 / 8)
    assert m["frac_r_eq_1"] == pytest.approx(3 / 8)
    assert m["frac_interior"] == pytest.approx(2 / 8)
    assert len(m["atom_fracs_by_k"]) == 20


def test_marginals_pairwise_rate_equals_mean_rf():
    counts = np.full((3, 100), 4, dtype=np.int16)
    m = marginals(counts, n_compare=19)
    assert m["pairwise_match_rate"] == pytest.approx(4 / 19)


def test_endpoint_purity_recovers_a_known_binomial():
    """Y2's obligation. Feed a marginal generated at a known p and check the
    implied p comes back -- so the recomputation is verified against a case
    whose answer is known, not just executed."""
    rng = np.random.default_rng(0)
    p = 0.02
    counts = rng.binomial(19, p, size=(20, 10240)).astype(np.int16)
    m = marginals(counts, n_compare=19)
    ep = endpoint_purity(m, 19)
    assert ep["implied_mean_p"] == pytest.approx(p, abs=0.004)
    # Y2's illustrative numbers at p = 0.02.
    assert ep["binomial_P_k_eq_0_at_pbar"] == pytest.approx(0.68, abs=0.05)
    assert ep["binomial_P_k_le_1_at_pbar"] == pytest.approx(0.95, abs=0.03)


# ---------------------------------------------------------------------------
# Rotation null and Hungarian arm
# ---------------------------------------------------------------------------

def test_rotation_null_is_far_below_a_planted_match():
    """The null must not admit the planted cos-1.0 pair."""
    Wa, Wb = _planted_pair(d=64, n=200, seed=3)
    null = rotation_null(Wa.float(), Wb.float(), n_draws=8, seed=1)
    assert 0.0 < null["floor"] < 1.0
    assert null["n_samples"] == 8 * 200
    assert null["null_max_cosine_max"] < 1.0
    assert np.isfinite(null["mc_se"])


def test_rotation_null_floor_falls_as_dimension_rises():
    """A computable comparison, not a mechanism story: random max-cosine
    shrinks with d, so the floor at d=256 must be below the floor at d=32."""
    lo = rotation_null(*[_unit(torch.randn(32, 128)).float() for _ in range(2)],
                       n_draws=6, seed=2)["floor"]
    hi = rotation_null(*[_unit(torch.randn(256, 128)).float() for _ in range(2)],
                       n_draws=6, seed=2)["floor"]
    assert hi < lo


def test_hungarian_arm_runs_and_iou_is_bounded():
    Wa, Wb = _planted_pair(d=32, n=64, seed=5)
    rng = np.random.default_rng(0)
    sub = hungarian_subsample(Wa.float(), Wb.float(), m=32, rng=rng)
    assert sub["m"] == 32 and len(sub["hungarian"]) == 32
    assert len(sub["oneway"]) <= 32
    for theta in (0.0, 0.3, 0.7):
        v = iou(sub["mnn"], sub["oneway"], theta)
        assert np.isnan(v) or 0.0 <= v <= 1.0


def test_iou_of_a_set_with_itself_is_one():
    s = {(0, 1): 0.9, (2, 3): 0.8}
    assert iou(s, s, 0.5) == pytest.approx(1.0)
    assert iou(s, {}, 0.5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The ENTRY POINT. Gotcha 14 extension 4's corollary: a unit suite that never
# executes main() cannot catch an entry-point bug, and this project has shipped
# three of those.
# ---------------------------------------------------------------------------

def _write_ensemble(root: Path, n_seeds=4, d=32, n=64, dirty=False, break_sha=False):
    for s in range(n_seeds):
        p = root / f"seed_{s}"
        p.mkdir(parents=True)
        g = torch.Generator().manual_seed(100 + s)
        W = _unit(torch.randn(d, n, generator=g))
        torch.save({"W_dec": W}, p / "model.pt")
        meta = _meta(s, dict_size=n, d_model=d, git_dirty=dirty)
        if break_sha and s == 2:
            meta["git_sha"] = "mismatch"
        (p / "meta.json").write_text(json.dumps(meta))
    return [root / f"seed_{s}" for s in range(n_seeds)]


def _run_main(ckpts, out, compare=(), extra=()):
    cmd = [sys.executable, "-m", "src.analysis.rf_marginal",
           "--ckpts", *[str(c) for c in ckpts], "--out", str(out),
           "--rotation-draws", "4", "--hungarian-sizes", "16", "32",
           "--hungarian-repeats", "2", "--device", "cpu"]
    if compare:
        cmd += ["--compare-ckpts", *[str(c) for c in compare]]
    cmd += [str(x) for x in extra]
    return subprocess.run(cmd, capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parents[1])


def test_main_produces_a_complete_artifact(tmp_path):
    ckpts = _write_ensemble(tmp_path)
    out = tmp_path / "rf.json"
    r = _run_main(ckpts, out)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    doc = json.loads(out.read_text())
    for key in ("identity", "preflight", "rotation_null", "theta_selection",
                "marginals_by_theta", "hungarian_arm", "prediction_scorecard"):
        assert key in doc, key
    assert doc["identity"]["similarity"] == "decoder_cosine"
    assert doc["identity"]["theta_grid"] == THETA_GRID
    assert set(doc["identity"]["ckpt_model_sha256_16"]) == {"0", "1", "2", "3"}
    assert doc["preflight"]["ensemble_identity"]["verdict"] == "PASS"
    assert doc["theta_selection"]["verdict"] in {"SELECTED", "NO_THETA_QUALIFIES"}
    assert doc["prediction_scorecard"]["P1_mnn_nested_in_oneway"]["outcome"] == "PASS"


def test_main_exits_nonzero_on_identity_mismatch(tmp_path):
    """Gotcha 13 extension 1: the verdict must be READABLE BY ITS CONSUMER.
    A gate that prints FAIL and exits 0 is worse than no gate."""
    ckpts = _write_ensemble(tmp_path, break_sha=True)
    r = _run_main(ckpts, tmp_path / "rf.json")
    assert r.returncode == 1, r.stdout[-2000:]
    assert "PRE-FLIGHT FAIL" in r.stdout
    assert not (tmp_path / "rf.json").exists()


def test_main_exits_nonzero_on_dirty_provenance(tmp_path):
    ckpts = _write_ensemble(tmp_path, dirty=True)
    r = _run_main(ckpts, tmp_path / "rf.json")
    assert r.returncode == 1, r.stdout[-2000:]


# ---------------------------------------------------------------------------
# Amendment AB additions: per-seed error bars, the C3 identity, the IoU scale
# curve, and the MNN-distinctness check.
# ---------------------------------------------------------------------------

def test_per_seed_atoms_error_bar_exceeds_the_naive_one_when_seeds_differ():
    """The whole point of the per-seed SE (amendment AB) is that it carries
    seed-level dependence the pooled binomial SE ignores. Construct seeds that
    genuinely differ and require the empirical bar to be the larger one."""
    from src.analysis.rf_marginal import per_seed_atoms
    counts = np.zeros((4, 1000), dtype=np.int16)
    counts[0, :] = 0
    counts[1, :] = 1
    counts[2, :] = 0
    counts[3, :] = 1
    ps = per_seed_atoms(counts, n_compare=3)
    sd = np.asarray(ps["atom_sd_over_seeds"])
    naive = np.asarray(ps["atom_se_naive_pooled"])
    assert sd[0] > naive[0] and sd[1] > naive[1]
    assert ps["n_seeds"] == 4


def test_per_seed_atoms_agree_with_the_pooled_histogram():
    """Mean over seeds of the per-seed atoms must equal the pooled atoms --
    an internal consistency identity, computed rather than assumed."""
    from src.analysis.rf_marginal import marginals, per_seed_atoms
    rng = np.random.default_rng(3)
    counts = rng.binomial(19, 0.4, size=(5, 500)).astype(np.int16)
    m = marginals(counts, 19)
    ps = per_seed_atoms(counts, 19)
    assert np.allclose(ps["atom_mean_over_seeds"], m["atom_fracs_by_k"], atol=1e-12)


def test_c3_identity_holds_when_mnn_equals_oneway():
    """C3: if IoU(MNN, one-way) == 1 then IoU(MNN, H) == IoU(one-way, H).
    Verified on constructed sets so the identity is checked independently of
    the decoder data that motivated it."""
    mnn = {(0, 0): 0.9, (1, 1): 0.8}
    ow = dict(mnn)
    hung = {(0, 0): 0.9, (1, 2): 0.7}
    assert iou(mnn, ow, 0.75) == pytest.approx(1.0)
    assert iou(mnn, hung, 0.75) == pytest.approx(iou(ow, hung, 0.75))


def test_c3_identity_would_be_violated_by_a_broken_mnn():
    """The identity must be ABLE to fail -- otherwise asserting it proves
    nothing. Break the premise and confirm the two IoUs diverge."""
    mnn = {(0, 0): 0.9}
    ow = {(0, 0): 0.9, (1, 1): 0.8}
    hung = {(0, 0): 0.9, (1, 1): 0.8}
    assert iou(mnn, ow, 0.75) < 1.0
    assert iou(mnn, hung, 0.75) != pytest.approx(iou(ow, hung, 0.75))


def test_main_emits_the_scale_curve_and_distinctness_check(tmp_path):
    ckpts = _write_ensemble(tmp_path)
    out = tmp_path / "rf.json"
    r = _run_main(ckpts, out)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    doc = json.loads(out.read_text())
    assert doc["identity"]["hungarian_sizes"] == [16, 32]
    sizes = {row["m"] for row in doc["hungarian_arm"]["rows"]}
    assert sizes == {16, 32}, sizes
    chk = doc["mnn_distinctness_check"]
    assert chk["verdict"] in {"MNN_IS_A_DISTINCT_RULE", "SUSPECT_MNN_MAY_BE_ONEWAY"}
    assert "per_seed" in doc["marginals_by_theta"]["0.70"]["mnn"]


# ===========================================================================
# The ANCHOR / COMPARISON SPLIT and the count guard (amendments AG, AK).
#
# EVERY MUTATION BELOW WAS RUN, not merely asserted:
#   * dropping the `[:n_anchors]` slice on `counts` -> kills
#     test_extension_rows_never_enter_the_marginal (the non-anchor rows arrive
#     as an all-zero block, so `n_anchors` no longer matches the anchor count).
#   * removing the `a >= n_anchors and b >= n_anchors` skip -> kills
#     test_extension_extension_pairs_are_not_computed on the pair count.
#   * changing `!=` to `<` in the --expect-total-ckpts comparison (i.e. a
#     "self-consistency" guard that passes at 20) -> kills
#     test_count_guard_refuses_a_short_ensemble.
#   * treating an absent `b_dec_sha256_16` as agreement -> kills
#     test_b_dec_distinctness_absent_is_not_agreement.
#   * computing AG cuts by a formula instead of the committed table -> kills
#     test_ag_cuts_refuse_an_unregistered_N.
# ===========================================================================

def test_ag_cuts_match_the_committed_table():
    from src.analysis.rf_marginal import ag_cuts
    c19, c39 = ag_cuts(19), ag_cuts(39)
    assert (c19["primary_unstable_k_max"], c19["primary_stable_k_min"]) == (1, 18)
    assert (c39["primary_unstable_k_max"], c39["primary_stable_k_min"]) == (2, 37)
    # The proportional criterion, which is WHY the cuts are 2 and 37 and not
    # 1 and 38. AG1 records the mapping landing within 0.0014 at both ends.
    assert abs(c19["primary_unstable_p_threshold"]
               - c39["primary_unstable_p_threshold"]) < 0.0014
    assert abs(c19["primary_stable_p_threshold"]
               - c39["primary_stable_p_threshold"]) < 0.0014
    assert c39["sensitivity_stable_k_min"] == 39
    assert c39["sensitivity_unstable_k_max"] == 0


def test_ag_cuts_refuse_an_unregistered_N():
    """A formula would silently invent a rounding rule AG1 declined to write."""
    from src.analysis.rf_marginal import ag_cuts
    for n in (9, 20, 29, 40):
        with pytest.raises(KeyError, match="AG1"):
            ag_cuts(n)


def test_ag_group_fracs_agree_with_the_legacy_keys_at_N19():
    """At N = 19 the AG primary cuts ARE `k <= 1` and `k >= N-1`. Compute what
    the dependence implies and confirm the code produces it -- gotcha 20."""
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 20, size=(20, 500)).astype(np.int16)
    m = marginals(counts, 19)
    g = m["ag_endpoint_groups"]
    assert g["primary_unstable_frac"] == pytest.approx(m["frac_k_le_1"])
    assert g["primary_stable_frac"] == pytest.approx(m["frac_k_ge_nminus1"])
    assert g["sensitivity_unstable_frac"] == pytest.approx(m["frac_r_eq_0"])
    assert g["sensitivity_stable_frac"] == pytest.approx(m["frac_r_eq_1"])


def test_ag_group_fracs_do_NOT_equal_the_legacy_keys_at_N39():
    """The discriminating half: at N = 39 the pre-registered cut is k <= 2, so a
    fixture whose mass sits at k = 2 must separate the two. Without this, the
    test above passes even if the AG cuts were hardcoded to the legacy ones."""
    counts = np.full((2, 100), 2, dtype=np.int16)   # every anchor at k = 2
    m = marginals(counts, 39)
    g = m["ag_endpoint_groups"]
    assert m["frac_k_le_1"] == pytest.approx(0.0)
    assert g["primary_unstable_frac"] == pytest.approx(1.0)


def test_b_dec_distinctness_passes_on_distinct_finals():
    from src.analysis.rf_marginal import verify_b_dec_distinctness
    rep = verify_b_dec_distinctness({s: _meta(s) for s in range(4)})
    assert rep["verdict"] == "PASS"
    assert rep["n_distinct_final_b_dec"] == 4


def test_b_dec_distinctness_fails_on_a_collision():
    from src.analysis.rf_marginal import verify_b_dec_distinctness
    metas = {s: _meta(s) for s in range(4)}
    metas[3]["b_dec_sha256_16"] = metas[1]["b_dec_sha256_16"]
    rep = verify_b_dec_distinctness(metas)
    assert rep["verdict"] == "FAIL"
    assert rep["n_distinct_final_b_dec"] == 3
    assert sorted(next(iter(rep["collisions"].values()))) == [1, 3]


def test_b_dec_distinctness_absent_is_not_agreement():
    """Gotcha 15: an absent key is reported ABSENT, never treated as matching."""
    from src.analysis.rf_marginal import verify_b_dec_distinctness
    metas = {s: _meta(s) for s in range(4)}
    del metas[2]["b_dec_sha256_16"]
    rep = verify_b_dec_distinctness(metas)
    assert rep["status"] == "ABSENT"
    assert rep["verdict"] == "FAIL"
    assert rep["missing_seeds"] == [2]


def test_count_guard_refuses_a_short_ensemble(tmp_path):
    """THE GLOB HAZARD. --expect-total-ckpts 6 against 4 supplied must refuse.
    A guard that merely checked self-consistency would pass here, which is
    exactly how an N = 19 result gets labelled N = 39."""
    ckpts = _write_ensemble(tmp_path, n_seeds=4)
    r = _run_main(ckpts, tmp_path / "rf.json",
                  extra=["--expect-total-ckpts", "6"])
    assert r.returncode == 1, r.stdout[-2000:]
    assert "expect-total-ckpts" in r.stdout
    assert not (tmp_path / "rf.json").exists()


def test_count_guard_accepts_the_exact_total(tmp_path):
    ckpts = _write_ensemble(tmp_path, n_seeds=4)
    r = _run_main(ckpts, tmp_path / "rf.json",
                  extra=["--expect-total-ckpts", "4"])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


def test_duplicate_seed_is_refused(tmp_path):
    """A dictionary supplied twice would be counted as an independent
    comparison seed and inflate `n_compare` with a copy of itself."""
    ckpts = _write_ensemble(tmp_path, n_seeds=3)
    r = _run_main(ckpts, tmp_path / "rf.json", compare=[ckpts[1]])
    assert r.returncode == 1, r.stdout[-2000:]
    assert "supplied twice" in r.stdout


def _split_ensemble(tmp_path, n_anchor=3, n_ext=2, d=32, n=64):
    """Anchors in one directory, extension seeds in another -- the AD5 layout."""
    root_a, root_b = tmp_path / "anchors", tmp_path / "ext"
    all_ck = _write_ensemble(root_a, n_seeds=n_anchor + n_ext, d=d, n=n)
    root_b.mkdir(parents=True)
    moved = []
    for p in all_ck[n_anchor:]:
        q = root_b / p.name
        p.rename(q)
        moved.append(q)
    return all_ck[:n_anchor], moved


def test_extension_rows_never_enter_the_marginal(tmp_path):
    anchors, ext = _split_ensemble(tmp_path, n_anchor=3, n_ext=2)
    out = tmp_path / "rf.json"
    r = _run_main(anchors, out, compare=ext)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    doc = json.loads(out.read_text())
    ident = doc["identity"]
    assert ident["n_seeds"] == 5
    assert ident["n_anchors"] == 3
    assert ident["n_compare_seeds"] == 4
    assert ident["anchor_seed_labels"] == [0, 1, 2]
    assert ident["compare_only_seed_labels"] == [3, 4]
    # THE DISCRIMINATING ASSERTION: 3 anchors x dict_size, not 5 x dict_size.
    # Dropping the `[:n_anchors]` slice makes this 5 * 64 and adds a block of
    # spurious r_f = 0 features.
    m = doc["marginals_by_theta"]["0.70"]["mnn"]
    assert m["n_anchors"] == 3 * 64
    assert m["n_compare_seeds"] == 4


def test_extension_extension_pairs_are_not_computed(tmp_path):
    anchors, ext = _split_ensemble(tmp_path, n_anchor=3, n_ext=3)
    out = tmp_path / "rf.json"
    r = _run_main(anchors, out, compare=ext)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    ident = json.loads(out.read_text())["identity"]
    # 6 seeds -> 15 unordered pairs; 3 extension-extension pairs are skipped.
    assert ident["n_pairs_computed"] == 12
    assert ident["n_pairs_skipped_extension_extension"] == 3
    # And the project-scale arithmetic the brief states, checked as a formula
    # rather than trusted: 20 anchors, 40 total -> 190 + 400 = 590.
    n_anchor, n_total = 20, 40
    n_ext = n_total - n_anchor
    assert (n_anchor * (n_anchor - 1) // 2) + n_anchor * n_ext == 590


def test_no_compare_ckpts_reproduces_the_all_anchor_path(tmp_path):
    """The N = 19 bit-identity guarantee, at test scale: passing every
    checkpoint as an anchor must equal passing none as a comparison seed."""
    ck_a = _write_ensemble(tmp_path / "a", n_seeds=4)
    ck_b = _write_ensemble(tmp_path / "b", n_seeds=4)
    out_a, out_b = tmp_path / "a.json", tmp_path / "b.json"
    assert _run_main(ck_a, out_a).returncode == 0
    assert _run_main(ck_b, out_b, compare=[]).returncode == 0
    da, db = json.loads(out_a.read_text()), json.loads(out_b.read_text())
    for key in ("rotation_null", "marginals_by_theta", "theta_selection",
                "hungarian_arm", "mnn_oneway_ratio_curve"):
        assert da[key] == db[key], key


def test_preflight_only_computes_no_marginal(tmp_path):
    ckpts = _write_ensemble(tmp_path, n_seeds=4)
    out = tmp_path / "pf.json"
    r = _run_main(ckpts, out, extra=["--preflight-only"])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    doc = json.loads(out.read_text())
    assert doc["identity"]["measurement"] == "rf_marginal_PREFLIGHT_ONLY"
    assert "marginals_by_theta" not in doc
    assert doc["preflight"]["b_dec_distinctness"]["verdict"] == "PASS"
    assert doc["preflight"]["ensemble_identity"]["verdict"] == "PASS"


def test_preflight_only_still_exits_nonzero_on_a_bad_ensemble(tmp_path):
    """A pre-flight mode whose failure cannot signal is worse than none."""
    ckpts = _write_ensemble(tmp_path, n_seeds=4, break_sha=True)
    r = _run_main(ckpts, tmp_path / "pf.json", extra=["--preflight-only"])
    assert r.returncode == 1, r.stdout[-2000:]
    assert not (tmp_path / "pf.json").exists()


def test_mnn_oneway_ratio_curve_is_derived_not_asserted(tmp_path):
    ckpts = _write_ensemble(tmp_path, n_seeds=4)
    out = tmp_path / "rf.json"
    assert _run_main(ckpts, out).returncode == 0
    curve = json.loads(out.read_text())["mnn_oneway_ratio_curve"]
    assert len(curve["rows"]) == len(THETA_GRID)
    for row in curve["rows"]:
        assert row["mnn_rate"] <= row["oneway_rate"] + 1e-12
        assert np.isnan(row["ratio"]) or row["ratio"] <= 1.0 + 1e-12
    assert isinstance(curve["is_non_decreasing_in_theta"], bool)


def test_ratio_curve_monotonicity_verdict_can_come_out_false():
    """A verdict that cannot be False is not a verdict."""
    ratios = [1.0] * len(THETA_GRID)
    ratios[3] = 0.5
    non_dec = all(ratios[i + 1] >= ratios[i] - 1e-12
                  for i in range(len(ratios) - 1))
    assert non_dec is False


# ===========================================================================
# Amendment CQ, A3 (the comparison denominator, CHECKED) and A5 (device class).
# ===========================================================================

def test_denominator_checker_passes_when_the_pool_matches():
    rep = verify_compare_denominator(39, 20, 20)
    assert rep["agree"] is True and rep["expected_n_compare"] == 39
    assert rep["pool"] == 40
    rep19 = verify_compare_denominator(19, 20, 0)
    assert rep19["agree"] is True and rep19["pool"] == 20


def test_denominator_checker_refuses_a_mismatch():
    """Both values in the message -- the one the code formed and the one the
    pool implies."""
    with pytest.raises(SystemExit) as ei:
        verify_compare_denominator(38, 20, 20)
    msg = str(ei.value)
    assert "38" in msg and "39" in msg and "20 anchors" in msg, msg


def test_written_denominator_readback_refuses_a_mismatch():
    assert_written_denominator("where", 39, 39)          # passes silently
    assert_written_denominator("where", np.int64(39), 39)
    with pytest.raises(SystemExit) as ei:
        assert_written_denominator("M1 identity", 19, 39)
    assert "19" in str(ei.value) and "39" in str(ei.value)


def test_main_records_the_compute_device_and_the_denominator_check(tmp_path):
    """The M1 identity carries `device_class` (read off the decoder tensor, in
    `artifact_identity.device_class`'s vocabulary) and the denominator check;
    the emitted counts file carries `device_class` beside `n_compare_seeds`."""
    ckpts = _write_ensemble(tmp_path)
    out = tmp_path / "rf.json"
    counts = tmp_path / "counts.h5"
    r = _run_main(ckpts, out, extra=["--emit-counts", str(counts)])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    doc = json.loads(out.read_text())
    ident = doc["identity"]
    assert ident["device_class"].startswith("cpu:"), ident["device_class"]
    assert ident["denominator_check"]["agree"] is True
    assert ident["denominator_check"]["n_compare"] == ident["n_compare_seeds"] == 3
    with h5py.File(counts, "r") as hf:
        dc = hf.attrs["device_class"]
        dc = dc.decode() if isinstance(dc, bytes) else str(dc)
        assert dc.startswith("cpu:"), dc
        assert int(hf.attrs["n_compare_seeds"]) == 3


# ------------------------------------------------ amendment CS, CS3c --

def _meta_cs3c(sha):
    return {"git_sha": sha, "config_hash": "c", "b_dec_init_sha256_16": "b",
            "dict_size": 8, "d_model": 4, "k": 2, "n_steps": 1, "lr": 0.1,
            "batch_size": 2, "git_dirty": False}


def test_identity_git_sha_disagreement_accepted_only_for_the_exact_set():
    from src.analysis.rf_marginal import verify_ensemble_identity
    metas = {0: _meta_cs3c("a" * 40), 1: _meta_cs3c("b" * 40), 2: _meta_cs3c("b" * 40)}
    assert verify_ensemble_identity(metas)["verdict"] == "FAIL"
    ok = verify_ensemble_identity(metas, ["b" * 40, "a" * 40])
    assert ok["verdict"] == "PASS"
    assert ok["keys"]["git_sha"]["status"] == "DISAGREES_ACCEPTED_BY_FLAG"
    assert ok["keys"]["git_sha"]["by_seed"] == {"0": "a" * 40, "1": "b" * 40, "2": "b" * 40}
    assert verify_ensemble_identity(metas, ["a" * 40, "b" * 40, "c" * 40])["verdict"] == "FAIL"
    assert verify_ensemble_identity(metas, ["a" * 40])["verdict"] == "FAIL"
    metas[1]["config_hash"] = "d"          # the flag excuses no OTHER key
    assert verify_ensemble_identity(metas, ["a" * 40, "b" * 40])["verdict"] == "FAIL"
