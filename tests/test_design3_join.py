"""THE DESIGN-3 JOIN'S ENTRY POINT, and every gate that is supposed to refuse.

A UNIT SUITE THAT NEVER EXECUTES `main()` CANNOT CATCH AN ENTRY-POINT BUG, and
this project has shipped three of those (gotcha 13 extension 4). So every test
below that matters runs the real CLI against synthetic artifacts carrying the
real attrs, and reads the real output document.

EVERY GATE GETS A FIXTURE THAT CAN EXHIBIT THE CONDITION IT REFUSES. That is
the half gotcha 16 and gotcha 25 are about: a gate whose fixture cannot show
the disputed condition is a gate that can only pass. In particular the
`_fixfp` gate is tested with a smooth artifact that is otherwise PERFECTLY
VALID and differs only in its filename, because the pre-AO6 files on disk are
exactly that.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.analysis.design3_join import (
    ANALYSIS_MIN_FIRING_ROWS,
    ak8_discovery_set,
    firing_rate_edges,
    floor_report,
    qualifying_thetas,
    resolve_gtilde_seed_order,
)
from src.analysis.rf_marginal import THETA_GRID, THETA_LITERATURE

ROOT = Path(__file__).resolve().parents[1]

N_SEEDS = 5
N_FEAT = 400
N_SHIFTS = 8
N_TRIALS = 39
N_ROWS = 1_000_000
EDGES = [0.0, 0.0003, 0.001, 0.003, 0.01, 0.03, 1.000000001]

# THE ALIGNMENT-SIDE ARTIFACTS ARE IN A DIFFERENT SEED ORDER FROM THE
# REPRODUCIBILITY-SIDE ONES, WHICH IS THE WHOLE POINT OF THE FIXTURE.
# On the real data that difference is lexicographic-vs-numeric and 18 of 20
# rows disagree; a 3-seed fixture cannot exhibit lexicographic order at all
# (0,1,2 sorts the same both ways), so the fixture uses an explicit
# permutation instead. Row `i` of null_pilot / gtilde / the degenerate flag
# holds TRUE SEED `PERM[i]`; rf_counts and d_fs are in numeric order.
# A fixture that could not exhibit the condition could not test it (gotcha 16).
PERM = [3, 0, 4, 1, 2]

# `d_fs` gets a THIRD order and the smooth gtilde a FOURTH. On the real data
# `d_fs` is numeric and both gtildes are lexicographic, so those alignments are
# genuine no-ops there -- which means a fixture sharing their order cannot kill
# a mutation that removes them, and the mutation run confirmed exactly that (0
# tests killed, three times). Distinct orders make each alignment testable
# without claiming the real artifacts need reordering.
D_PERM = [1, 4, 0, 3, 2]
S_PERM = [0, 2, 4, 1, 3]

IDENT = {
    "library_sha": "libsha0000000000",
    "n_concepts": 712,
    "argmax_pool_n": 703,
    "truncation_policy": "untruncated",
    "scored_row_set": "label_store",
    "n_rows": N_ROWS,
}


def _truth(rng, b1=0.8, gamma=0.35):
    """A data-generating process with a REAL b1, so the entry point's numbers
    can be checked against something rather than merely existing.

    Every array is indexed by TRUE SEED. Because each seed gets its own
    `fire`, `M`, `d` and its own `counts` drawn from its own `p`, a mis-joined
    seed axis destroys the relationship -- which is what makes
    `test_the_fit_recovers_the_planted_b1` a discrimination test for the
    alignment and not merely a smoke test.
    """
    from scipy.special import expit
    fire = 10 ** rng.uniform(-3.4, -1.2, (N_SEEDS, N_FEAT))
    # A SEED-SPECIFIC set of features below A4a's floor. Without this every
    # feature clears the floor in every seed, `retained_at_floor` is all-True,
    # and a permutation of it is INVISIBLE -- which is why the d_retained
    # mutation killed nothing on the first fixture.
    for t in range(N_SEEDS):
        fire[t, : 3 + 2 * t] = 1e-9
    logM = rng.normal(size=(N_SEEDS, N_FEAT))
    M = np.exp(logM)
    d = rng.uniform(0.5, 5.5, (N_SEEDS, N_FEAT))
    lp = np.log10(fire)
    f = 0.9 * np.sin(2.0 * (lp + 2.3))
    eta = b1 * logM + (f - f.mean()) + gamma * (d - d.mean())
    # per-TRUE-SEED null draws; gtilde_null is a monotone transform of the
    # same rows, which is the correspondence the resolver recovers.
    null_g = rng.lognormal(0.0, 1.0, (N_SEEDS, N_SHIFTS, N_FEAT))
    gt_null = 2.0 * null_g + 0.01 * rng.normal(size=null_g.shape)
    return fire, M, d, expit(eta), null_g, gt_null, gamma


def _write_inputs(tmp: Path, rng, *, b1=0.8, smooth_name="gt_smooth_fixfp.h5",
                  break_seed_axis=False, nonfinite_logM=False,
                  d_fs_definition="A_cluster_entropy",
                  corrupt_reference=False, scramble_gtilde_null=False,
                  mismatch_degenerate_argmax=False, rho=None):
    fire, M, d, p, null_g, gt_null, gamma = _truth(rng, b1=b1)
    if rho is None:
        counts = rng.binomial(N_TRIALS, p).astype(np.int16)
    else:
        # OVERDISPERSED counts: beta-binomial with intra-cluster correlation
        # `rho`, as the real match counts are (rho ~ 0.40). Only a test that
        # reads the beta-binomial arm's POOLED numbers asks for this -- see
        # `joined_overdispersed`.
        s_bb = 1.0 / rho - 1.0
        counts = rng.binomial(N_TRIALS, rng.beta(p * s_bb, (1.0 - p) * s_bb)
                              ).astype(np.int16)
    argmax = rng.integers(0, 4, (N_SEEDS, N_FEAT)).astype(np.int32)

    # counts fall as theta rises; the reference theta carries the real signal
    all_theta = tmp / "counts_all.h5"
    with h5py.File(all_theta, "w") as hf:
        for i, t in enumerate(THETA_GRID):
            scale = max(0.0, 1.0 - 0.10 * (t - THETA_LITERATURE) / 0.05)
            c = (counts if abs(t - THETA_LITERATURE) < 1e-9
                 else np.minimum(counts, np.int16(
                     np.floor(counts * min(scale, 1.0)))))
            ds = hf.create_dataset(f"match_count_theta_{t:.2f}", data=c)
            ds.attrs["theta"] = t
            ds.attrs["n_compare_seeds"] = N_TRIALS
            ds.attrs["similarity"] = "decoder_cosine"
            ds.attrs["rule"] = "mnn"
        hf.attrs["measurement"] = "rf_per_anchor_counts_all_theta"
        hf.attrs["theta_grid"] = json.dumps(THETA_GRID)
        hf.attrs["n_compare_seeds"] = N_TRIALS
        hf.attrs["similarity"] = "decoder_cosine"
        hf.attrs["rule"] = "mnn"
        hf.attrs["seed_labels"] = json.dumps(list(range(N_SEEDS)))
        hf.attrs["git_sha"] = "deadbeef"

    one = tmp / "counts_one.h5"
    with h5py.File(one, "w") as hf:
        c = counts.copy()
        if corrupt_reference:
            c[0, 0] = (c[0, 0] + 1) % (N_TRIALS + 1)
        hf.create_dataset("match_count", data=c)
        hf.attrs["theta"] = THETA_LITERATURE
        hf.attrs["n_compare_seeds"] = N_TRIALS
        hf.attrs["similarity"] = "decoder_cosine"
        hf.attrs["rule"] = "mnn"
        hf.attrs["seed_labels"] = json.dumps(list(range(N_SEEDS)))

    m1 = tmp / "m1.json"
    m1.write_text(json.dumps({
        "identity": {"n_compare_seeds": N_TRIALS},
        "theta_selection": {
            "floor": 0.26, "endpoint_max_share": 0.80,
            "selected_theta": 0.65,
            "grid": [{"theta": t, "clears_floor": True, "usable": True,
                      "qualifies": t in (0.65, 0.70, 0.75)}
                     for t in THETA_GRID],
        }}))

    # ---- the ALIGNMENT SIDE: each gtilde in its OWN order -----------------
    # `null_g_b` is stored in PERM order, so a gtilde whose rows are in order
    # Q must store `gtilde_null` rows matching the SAME true seeds -- i.e. the
    # resolver has to find, for gtilde row i, the null row holding true seed
    # Q[i]. That is exactly the correspondence it recovers on the real data.
    null_row_of_seed = {s: i for i, s in enumerate(PERM)}

    def _gt(path, Mv, order, scramble=False, argmax_tweak=False):
        gtn = gt_null[order]
        if scramble:
            gtn = rng.lognormal(0.0, 1.0, gtn.shape)
        am = argmax[order].copy()
        if argmax_tweak:
            am[0, 0] = (am[0, 0] + 1) % 4
        with h5py.File(path, "w") as hf:
            hf.create_dataset("M", data=Mv[order])
            hf.create_dataset("argmax_concept", data=am)
            hf.create_dataset("gtilde_null", data=gtn)
            hf.create_dataset("concepts",
                              data=np.array([f"C{i}" for i in range(4)],
                                            dtype=h5py.special_dtype(vlen=str)))
            for k, v in IDENT.items():
                hf.attrs[k] = v
            hf.attrs["firing_rate_edges"] = json.dumps(EDGES)

    gt = tmp / "gt.h5"
    Mv = M.copy()
    if nonfinite_logM:
        Mv[0, 0] = 0.0                      # log M -> -inf
    _gt(gt, Mv, PERM, scramble=scramble_gtilde_null)
    gts = tmp / smooth_name
    _gt(gts, M * 1.3, S_PERM)

    nr = tmp / "null_red.h5"
    with h5py.File(nr, "w") as hf:
        hf.create_dataset("fire_rate", data=fire[PERM])
        hf.create_dataset("null_g_b", data=null_g[PERM])
        hf.create_dataset("seeds",
                          data=np.array([f"seed_{s}" for s in PERM],
                                        dtype=h5py.special_dtype(vlen=str)))
        for k, v in IDENT.items():
            hf.attrs[k] = v

    # ---- the REPRODUCIBILITY SIDE: rows are in NUMERIC order -------------
    dfs = tmp / "d_fs.h5"
    with h5py.File(dfs, "w") as hf:
        hf.create_dataset("d_fs_cluster_entropy", data=d[D_PERM])
        hf.create_dataset(
            "retained_at_floor",
            data=(fire >= ANALYSIS_MIN_FIRING_ROWS / N_ROWS)[D_PERM])
        seeds = list(D_PERM)
        if break_seed_axis:
            seeds = [s + 100 for s in seeds]      # a genuinely different SET
        hf.create_dataset("seeds", data=np.array(seeds, dtype=np.int32))
        hf.attrs["adopted_definition"] = d_fs_definition
        for k in ("truncation_policy", "scored_row_set", "n_rows"):
            hf.attrs[k] = IDENT[k]

    deg = tmp / "deg.h5"
    with h5py.File(deg, "w") as hf:
        # A SEED-SPECIFIC flagged set. With the same 12 indices flagged in
        # every seed the array is row-invariant and a permutation of it is
        # INVISIBLE -- which is why the flag-alignment mutation killed nothing
        # on the first fixture.
        flag = np.zeros((N_SEEDS, N_FEAT), bool)
        for t in range(N_SEEDS):
            flag[t, 20 * t: 20 * t + 12] = True
        hf.create_dataset("degenerate_argmax", data=flag[PERM])
        am = argmax[PERM].copy()
        if mismatch_degenerate_argmax:
            am[0, 0] = (am[0, 0] + 1) % 4
        hf.create_dataset("argmax_concept", data=am)
        for k, v in IDENT.items():
            hf.attrs[k] = v

    return {"all_theta": all_theta, "one": one, "m1": m1, "gt": gt,
            "gts": gts, "nr": nr, "dfs": dfs, "deg": deg,
            "fire": fire, "M": M, "d": d, "counts": counts, "b1": b1,
            "gamma": gamma, "perm": PERM}


def _run(inp, out, extra=()):
    cmd = [sys.executable, "-m", "src.analysis.design3_join",
           "--rf-counts-all-theta", str(inp["all_theta"]),
           "--rf-counts", str(inp["one"]),
           "--m1-json", str(inp["m1"]),
           "--gtilde", str(inp["gt"]),
           "--gtilde-smooth", str(inp["gts"]),
           "--null-reductions", str(inp["nr"]),
           "--d-fs", str(inp["dfs"]),
           "--degenerate-flag", str(inp["deg"]),
           "--skip-feature-side",
           "--out", str(out), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


@pytest.fixture(scope="module")
def joined(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("join")
    inp = _write_inputs(tmp, np.random.default_rng(0))
    out = tmp / "join.json"
    r = _run(inp, out)
    assert r.returncode == 0, r.stdout[-6000:] + r.stderr[-6000:]
    return {"doc": json.loads(out.read_text()), "inp": inp, "stdout": r.stdout}


RHO_FIXTURE = 0.3


@pytest.fixture(scope="module")
def joined_overdispersed(tmp_path_factory):
    """The same fixture with BETA-BINOMIAL counts (rho = 0.3).

    WHY IT EXISTS (measured 2026-11-13, analysis/adhoc/probes/2026-11-13/
    fix_round/). `joined` draws its counts from a BINOMIAL, so the
    beta-binomial arm has no interior optimum there: the likelihood rises
    towards `s -> inf` and L-BFGS-B stops wherever its last-bit arithmetic
    lets it. With MKL every anchor stopped near `log_s` = 16-18 with a usable
    standard error; with OpenBLAS (the cluster's library, and conda-forge's
    default on Linux) anchor 0 ran to `log_s` = 623, its finite-difference
    covariance came out exactly 0, `se_b1_z` was 0.0, and `meta_analyse`
    dropped it by its `se > 0` condition -- 3 degrees of freedom instead of 4,
    I2 55.73 instead of 40.99. Neither number says anything about the code:
    a pooled beta-binomial quantity on binomial data is a property of the
    optimiser's stopping point. The real counts are overdispersed
    (rho 0.396-0.405 in every anchor of every stored join), so a fixture that
    is too gives the arm what it has on the real data: an interior optimum,
    and the same kept anchors under either library (all five; I2 59.6977,
    agreeing to six figures between the two).
    """
    tmp = tmp_path_factory.mktemp("join_bb")
    inp = _write_inputs(tmp, np.random.default_rng(0), rho=RHO_FIXTURE)
    out = tmp / "join.json"
    r = _run(inp, out)
    assert r.returncode == 0, r.stdout[-6000:] + r.stderr[-6000:]
    return {"doc": json.loads(out.read_text()), "inp": inp, "stdout": r.stdout}


# ---------------------------------------------------------------------------
# The entry point produces a complete document
# ---------------------------------------------------------------------------

def test_the_document_has_every_required_section(joined):
    doc = joined["doc"]
    for key in ("identity", "inference_framing", "analysis_set", "primary",
                "theta_curve", "dispersion_arm", "sensitivity",
                "ak8_discovery_set", "ak9b_pooled_se_ratio",
                "prediction_scorecard", "open_specification_items",
                "decision"):
        assert key in doc, key
    assert doc["identity"]["firewall_crossing"] == "DELIBERATE"
    assert doc["identity"]["n_compare_seeds"] == N_TRIALS
    assert doc["identity"]["d_fs_definition"] == "A_cluster_entropy"
    assert "X7 FIREWALL CROSSING" in joined["stdout"]


# ---------------------------------------------------------------------------
# SEED ALIGNMENT -- the join's largest hazard, and the one the real data has
# ---------------------------------------------------------------------------

def test_the_fixture_actually_exhibits_a_permuted_seed_axis(joined):
    """PRECONDITION. If the two sides happened to agree, every alignment test
    below would pass on a no-op and cover nothing (gotcha 16)."""
    sa = joined["doc"]["seed_alignment"]
    assert sa["anchor_order_rf_counts"] == [str(s) for s in range(N_SEEDS)]
    assert sa["null_reductions_order"] == [str(s) for s in PERM]
    # THREE different orders, so no arm's alignment is a no-op in the fixture
    assert sa["d_fs_order"] == [str(s) for s in D_PERM]
    assert sa["n_rows_that_would_be_mispaired_by_a_positional_join"] > 0


def test_gamma_recovers_the_planted_value_so_d_fs_must_be_aligned(joined):
    """`d_fs` enters only through `gamma`, so nothing else in the document can
    detect its misalignment -- and the mutation run confirmed that (0 tests
    killed). The planted `gamma` is what pins it."""
    g = joined["inp"]["gamma"]
    got = [r["gamma_d_fs"] for r in joined["doc"]["primary"]["per_seed"]]
    assert all(abs(x - g) < 0.12 for x in got), (g, got)
    assert joined["doc"]["primary"]["meta_gamma_d_fs"]["fixed_effect"] > 0


def test_the_d_fs_side_is_aligned_too(joined):
    """SEPARATE COVERAGE, for the same reason as the firing-rate one: `d_fs`
    is numeric on the real data, so its alignment is a real no-op there and a
    numeric fixture could not kill a mutation removing it (measured: 0 tests).
    `retained_at_floor` is derived from the FIRING RATE, so after alignment it
    must agree with the floor computed from the aligned `fire_rate` -- and
    that agreement fails if either side is left permuted."""
    x = joined["doc"]["analysis_set"]["d_fs_cross_check"]
    assert x["d_fs_retained_at_floor_agrees"] is True
    assert x["n_cells_disagreeing"] == 0


def test_the_unlabelled_gtilde_axis_is_resolved_from_the_data(joined):
    sa = joined["doc"]["seed_alignment"]
    got = sa["gtilde_order_resolved"]
    assert got["verdict"] == "RESOLVED"
    assert got["labels"] == [str(s) for s in PERM]
    assert got["min_margin_observed"] > got["min_margin_required"]
    # THE SMOOTH ARM IS RESOLVED INDEPENDENTLY, not by inheriting the hard-bin
    # answer -- which is only testable because the fixture gives it a
    # DIFFERENT order (on the real data both are lexicographic, so inheriting
    # would be invisible).
    assert sa["gtilde_smooth_order_resolved"]["labels"] == [str(s) for s in S_PERM]
    assert S_PERM != PERM


def test_the_degenerate_flag_inherits_only_after_an_exact_check(joined):
    d = joined["doc"]["seed_alignment"]["degenerate_flag_order"]
    assert d["verified_by"] == "elementwise equality of argmax_concept"
    assert d["labels"] == [str(s) for s in PERM]


def test_refuses_a_degenerate_flag_built_from_a_different_gtilde(tmp_path):
    """One differing cell in argmax_concept is enough -- the check is
    elementwise, because a total or a shape survives a permutation."""
    inp = _write_inputs(tmp_path, np.random.default_rng(21),
                        mismatch_degenerate_argmax=True)
    r = _run(inp, tmp_path / "o.json")
    assert r.returncode != 0
    assert "not elementwise equal" in (r.stdout + r.stderr)


def test_the_resolver_refuses_rather_than_guessing(tmp_path):
    """A resolver that always returns something cannot fail (gotcha 25)."""
    inp = _write_inputs(tmp_path, np.random.default_rng(22),
                        scramble_gtilde_null=True)
    r = _run(inp, tmp_path / "o.json")
    assert r.returncode != 0
    assert "cannot resolve" in (r.stdout + r.stderr)


def test_resolver_recovers_the_permutation_directly(tmp_path):
    inp = _write_inputs(tmp_path, np.random.default_rng(23))
    got = resolve_gtilde_seed_order(inp["gt"], inp["nr"], n_sub=N_FEAT)
    assert got["labels"] == [str(s) for s in PERM]
    assert [r["gtilde_row"] for r in got["rows"]] == list(range(N_SEEDS))
    # row i of gtilde maps to row i of null_pilot on the real data too
    assert [r["null_row"] for r in got["rows"]] == list(range(N_SEEDS))


def test_misaligning_the_seeds_destroys_the_estimate(tmp_path):
    """THE DISCRIMINATION TEST FOR THE ALIGNMENT ITSELF. If a positional join
    gave the same answer, none of the machinery above would be load-bearing
    and the real-data defect would not matter. It does not: each seed has its
    own M, firing rate and counts, so the wrong pairing collapses b1."""
    from src.analysis.design3_fit import build_design, fit_design3
    rng = np.random.default_rng(24)
    inp = _write_inputs(tmp_path, rng)
    with h5py.File(inp["gt"], "r") as hf:
        M_rows = hf["M"][:]                    # in PERM order
    with h5py.File(inp["nr"], "r") as hf:
        fire_rows = hf["fire_rate"][:]         # in PERM order
    with h5py.File(inp["dfs"], "r") as hf:
        d_rows = hf["d_fs_cluster_entropy"][:]  # numeric order
    counts = inp["counts"]                      # numeric order

    def _b1(aligned: bool):
        out = []
        for i in range(N_SEEDS):
            j = PERM.index(i) if aligned else i
            lm = np.log(M_rows[j])
            z = (lm - lm.mean()) / lm.std()
            lp = np.log10(fire_rows[j])
            des = build_design(z, lp, d_rows[i])
            f = fit_design3(counts[i].astype(float), N_TRIALS, des["X"],
                            des["S"])
            out.append(float(f["beta"][des["idx_logM"]]))
        return np.array(out)

    good, bad = _b1(True), _b1(False)
    assert good.mean() > 0.4, good
    assert bad.mean() < 0.5 * good.mean(), (good.mean(), bad.mean())


def test_the_firing_rate_side_is_aligned_too(joined):
    """SEPARATE COVERAGE, because dropping `fire_rate`'s alignment killed NO
    test when the mutation was run -- `b1` is estimated on `log M` and the
    spline absorbs whatever firing rates it is handed, so the coefficient
    barely moves in a fixture. The per-bin table and AK8's within-bin cut DO
    depend on it, so they are what pins it: the reported per-seed firing-bin
    counts must match TRUE seed `i`'s firing rates, not `PERM[i]`'s."""
    from src.analysis.null_pooling import bin_index
    fire = joined["inp"]["fire"]                      # indexed by TRUE seed
    n_bins = len(EDGES) - 1
    thresh = ANALYSIS_MIN_FIRING_ROWS / N_ROWS
    rows = joined["doc"]["analysis_set"]["per_seed"]

    def _retained_counts(fr):
        fb = np.clip(bin_index(fr, EDGES), 0, n_bins - 1)
        return np.bincount(fb[fr >= thresh], minlength=n_bins).tolist()

    mispaired = 0
    for i, row in enumerate(rows):
        want = _retained_counts(fire[i])
        assert row["n_retained_per_firing_bin"] == want, (i, want, row)
        if _retained_counts(fire[PERM[i]]) != want:
            mispaired += 1
    # the fixture must be able to tell the two apart, or this proves nothing
    assert mispaired >= N_SEEDS - 1, mispaired


def test_the_fit_recovers_the_planted_b1(joined):
    """The entry point is not merely wired -- it returns the right number.
    Because the fixture's seed axes are permuted, this ALSO fails if the
    alignment is dropped."""
    doc = joined["doc"]
    for row in doc["primary"]["per_seed"]:
        assert abs(row["b1_raw"] - joined["inp"]["b1"]) < 0.25, row
    fe = doc["primary"]["meta_b1_z"]["fixed_effect"]
    sd = np.mean([r["sd_logM"] for r in doc["primary"]["per_seed"]])
    assert abs(fe / sd - joined["inp"]["b1"]) < 0.2


def test_fits_are_per_seed_and_never_pooled(joined):
    doc = joined["doc"]
    assert len(doc["primary"]["per_seed"]) == N_SEEDS
    seeds = [r["seed"] for r in doc["primary"]["per_seed"]]
    assert seeds == list(range(N_SEEDS))
    assert doc["primary"]["meta_b1_z"]["n_seeds_used"] == N_SEEDS


def test_raw_coefficient_is_recoverable_and_sd_spread_is_visible(joined):
    """AL8a: `b1_raw = b1_z / SD`, and the SD's spread must not be absorbed."""
    doc = joined["doc"]
    for r in doc["primary"]["per_seed"]:
        assert abs(r["b1_raw"] - r["b1_z"] / r["sd_logM"]) < 1e-9
    spread = doc["primary"]["sd_logM_across_seeds"]
    assert len(spread["values"]) == N_SEEDS
    assert spread["max_over_min"] >= 1.0
    assert "cv" in spread


def test_standardisation_invariance_is_measured_not_asserted(joined):
    inv = joined["doc"]["primary"]["standardisation_invariance_seed0"]
    assert inv["verdict"] == "INVARIANT", inv
    assert inv["rel_residue_b1"] < 1e-6
    assert inv["abs_residue_deviance"] < 1e-6


def test_all_three_variance_shares_are_present(joined):
    """AL8a's third is NON-OPTIONAL: the joint drop makes collinearity a
    measured number rather than a stated caveat."""
    row = joined["doc"]["primary"]["per_seed"][0]
    dd = row["deviance_drops"]
    for k in ("delta_D_M", "delta_D_phi", "delta_D_joint", "collinear_share"):
        assert k in dd
    assert abs(dd["collinear_share"]
               - (dd["delta_D_joint"] - dd["delta_D_M"] - dd["delta_D_phi"])
               ) < 1e-9
    assert dd["collinear_share_sign"] in {"POSITIVE", "ZERO", "NEGATIVE"}
    assert "ratio" in row["variance_share"]


def test_theta_curve_covers_the_qualifying_range_only(joined):
    tc = joined["doc"]["theta_curve"]
    assert tc["qualifying_thetas"] == [0.65, 0.70, 0.75]
    assert [r["theta"] for r in tc["rows"]] == [0.65, 0.70, 0.75]
    assert isinstance(tc["sign_is_unanimous_across_qualifying_thetas"], bool)


def test_the_scorecard_outcomes_are_derived(joined):
    sc = joined["doc"]["prediction_scorecard"]
    assert set(sc) == {
        "D1_meta_b1_z_strictly_positive",
        "D2_variance_share_below_1",
        "D3_collinear_share_positive",
        "D4_seeds_0_and_1_are_not_both_extreme",
        # amendment AS
        "E3_dispersion_corrected_and_across_seed_agree_and_I2_falls",
        "E4_d_fs_share_exceeds_logM_share_by_more_than_5x",
        # amendment AT
        "AT_P1_P2_the_re_pointed_smooth_arm",
        "AT_P3_betabinomial_bound_above_0352pp_and_below_1pp",
        "AT_P4_FLAT_tolerance_is_tighter_than_0352pp",
        "AT_P5_gamma_sign_unanimous_across_the_grid",
        "AT_P6_d_fs_sd_ratio_exceeds_logM_at_every_theta",
        # amendment AU
        "AU_Q1_at_least_two_sensitivity_arms_cover_zero_under_betabinomial",
        "AU_Q2_FLAT_fires_in_under_5pct_of_permutations",
        "AU_Q3_observed_spread_is_unremarkable_against_its_own_null"}
    # AT_P1_P2 is a PAIR and carries its two verdicts one level down.
    for k, v in sc.items():
        if k == "AT_P1_P2_the_re_pointed_smooth_arm":
            for half in ("P1", "P2"):
                assert v[half]["outcome"] in {"HIT", "MISS"}, (k, half, v)
                assert "category" in v[half]
            continue
        assert v["outcome"] in {"HIT", "MISS"}, (k, v)
        assert "category" in v
    d1 = sc["D1_meta_b1_z_strictly_positive"]
    assert (d1["outcome"] == "HIT") == (d1["measured"] > 0)
    d4 = sc["D4_seeds_0_and_1_are_not_both_extreme"]
    ext = {d4["lowest_b1_seed"], d4["highest_b1_seed"]}
    assert (d4["outcome"] == "MISS") == (ext == {0, 1})


def test_the_floor_table_is_reported_per_firing_bin_even_when_empty(joined):
    """A4a's reporting requirement carries over unchanged (AL8c/AO16): the
    zeros are REPORTED, not skipped because the table is empty."""
    a = joined["doc"]["analysis_set"]
    assert a["analysis_min_firing_rows"] == ANALYSIS_MIN_FIRING_ROWS
    assert len(a["per_seed"]) == N_SEEDS
    for row in a["per_seed"]:
        assert len(row["n_excluded_per_firing_bin"]) == len(EDGES) - 1
        assert len(row["n_retained_per_firing_bin"]) == len(EDGES) - 1
    assert a["n_nonfinite_logM"] == 0
    assert a["d_fs_cross_check"]["d_fs_retained_at_floor_agrees"] is True


def test_ak8_arm_reports_counts_and_defers_the_geometric_half(joined):
    ak8 = joined["doc"]["ak8_discovery_set"]
    assert "hard_bin" in ak8 and "smooth" in ak8
    hb = ak8["hard_bin"]
    assert "DEFERRED" in hb["geometric_properties"]
    assert len(hb["per_seed"]) == N_SEEDS
    for row in hb["per_seed"]:
        assert row["n_discovery"] <= min(row["n_high_r_f"], row["n_low_M"])
        assert "log10_fire_rate_discovery" in row
        assert "log10_fire_rate_rest" in row


def test_ak9b_ratio_is_rederived_with_the_dependence_stated(joined):
    a = joined["doc"]["ak9b_pooled_se_ratio"]
    assert a["measured_at_n_compare"] == N_TRIALS
    assert a["inherited_value_must_not_be_used"] == 0.93
    dep = a["dependence_structure"]
    assert dep["shared_comparison_dictionaries_between_two_anchors"] == N_TRIALS - 1
    assert "MORE" in dep["statement"]
    assert a["direction"] in {"ACROSS_SEED_LARGER", "ACROSS_SEED_SMALLER",
                              "EQUAL"}


def test_dispersion_arm_reports_ak9a_endpoints_both_ways(joined):
    d = joined["doc"]["dispersion_arm"]
    assert len(d["per_seed"]) == N_SEEDS
    for row in d["per_seed"]:
        for fam in ("AK9a_binomial", "AK9a_betabinomial"):
            e = row[fam]
            assert "observed_P_k_eq_0" in e and "implied_P_k_eq_0" in e
            assert any(w in e["direction_of_misspecification"]
                       for w in ("TOO NARROW", "TOO WIDE", "MIXED"))
    assert "INHERITED" in d["lambda_policy"]


def test_both_meta_combinations_are_reported_and_neither_is_declared_primary(joined):
    doc = joined["doc"]
    m = doc["primary"]["meta_b1_z"]
    for k in ("fixed_effect", "fixed_effect_ci95", "across_seed_mean",
              "across_seed_ci95", "across_seed_se_over_fixed_effect_se"):
        assert k in m
    items = [o["item"] for o in doc["open_specification_items"]]
    assert any("headline" in i for i in items)


def test_the_smooth_arm_is_aligned_too(joined):
    """The fixture's smooth `M` is the hard-bin `M` times a CONSTANT, and
    `z(log(cM)) == z(log M)` exactly -- so with the seeds aligned the smooth
    arm's `b1_z` must equal the primary's to floating-point, and with them
    left permuted it cannot. An exact identity is a stronger check than a
    tolerance, and nothing else in the document detects this misalignment."""
    a = [r["b1_z"] for r in joined["doc"]["primary"]["per_seed"]]
    b = [r["b1_z"] for r in
         joined["doc"]["sensitivity"]["smooth_normaliser"]["per_seed"]]
    assert len(a) == len(b) == N_SEEDS
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < 1e-8, (i, x, y)


def test_sensitivity_arms_are_all_present_or_explicitly_not_run(joined):
    s = joined["doc"]["sensitivity"]
    assert set(s) == {"smooth_normaliser", "degenerate_excluded",
                      "feature_side"}
    assert "meta_b1_z" in s["smooth_normaliser"]
    assert "meta_b1_z" in s["degenerate_excluded"]
    # skipped in the fixture -- and the omission is RECORDED, not a missing key
    assert s["feature_side"]["status"].startswith("NOT_RUN")


def test_degenerate_arm_actually_drops_the_flagged_cells(joined):
    s = joined["doc"]["sensitivity"]["degenerate_excluded"]
    assert s["n_cells_flagged"] == N_SEEDS * 12
    # The flagged block is SEED-SPECIFIC and may overlap the below-floor
    # features, so the drop is 12 minus that overlap -- computed, not asserted.
    fire = joined["inp"]["fire"]
    thresh = ANALYSIS_MIN_FIRING_ROWS / N_ROWS
    for i in range(N_SEEDS):
        blk = slice(20 * i, 20 * i + 12)
        overlap = int((fire[i][blk] < thresh).sum())
        full = joined["doc"]["primary"]["per_seed"][i]["n_features"]
        assert s["per_seed"][i]["n_features"] == full - (12 - overlap), i


# ---------------------------------------------------------------------------
# The gates -- each with a fixture that CAN exhibit what it refuses
# ---------------------------------------------------------------------------

def test_refuses_a_pre_AO6_smooth_artifact_by_name(tmp_path):
    """The pre-AO6 files on disk are otherwise perfectly valid and differ only
    in their name, so the fixture differs only in its name too."""
    inp = _write_inputs(tmp_path, np.random.default_rng(1),
                        smooth_name="smooth_mu_b.h5")
    r = _run(inp, tmp_path / "o.json")
    assert r.returncode != 0
    assert "_fixfp" in (r.stdout + r.stderr)
    assert not (tmp_path / "o.json").exists()


def test_refuses_when_the_reference_slice_disagrees(tmp_path):
    inp = _write_inputs(tmp_path, np.random.default_rng(2),
                        corrupt_reference=True)
    r = _run(inp, tmp_path / "o.json")
    assert r.returncode != 0
    assert "bit-identical" in (r.stdout + r.stderr)


def test_refuses_a_non_finite_logM_rather_than_branching_on_it(tmp_path):
    """AL9a/AL12 measured this set EMPTY, so SPECIFICATION §4 records that no
    branch exists for it. A non-empty set is a changed input, not a case."""
    inp = _write_inputs(tmp_path, np.random.default_rng(3),
                        nonfinite_logM=True)
    r = _run(inp, tmp_path / "o.json")
    assert r.returncode != 0
    assert "non-finite or non-positive M" in (r.stdout + r.stderr)
    assert "unable to fire" in (r.stdout + r.stderr)


def test_refuses_definition_B_of_d_fs(tmp_path):
    inp = _write_inputs(tmp_path, np.random.default_rng(4),
                        d_fs_definition="B_log_distinct_clusters")
    r = _run(inp, tmp_path / "o.json")
    assert r.returncode != 0
    assert "adopted_definition" in (r.stdout + r.stderr)


def test_refuses_a_disagreeing_seed_axis(tmp_path):
    """CODEBASE §4.7: a genuinely missing seed must never be silently dropped
    by an index lookup that skips it."""
    inp = _write_inputs(tmp_path, np.random.default_rng(5),
                        break_seed_axis=True)
    r = _run(inp, tmp_path / "o.json")
    assert r.returncode != 0
    assert "seed sets differ" in (r.stdout + r.stderr)


def test_refuses_a_disagreeing_identity_attr(tmp_path):
    inp = _write_inputs(tmp_path, np.random.default_rng(6))
    with h5py.File(inp["gts"], "r+") as hf:
        hf.attrs["argmax_pool_n"] = 707            # amendment I4's exact case
    r = _run(inp, tmp_path / "o.json")
    assert r.returncode != 0
    assert "argmax_pool_n" in (r.stdout + r.stderr)


def test_reports_an_absent_identity_key_as_absent(tmp_path):
    inp = _write_inputs(tmp_path, np.random.default_rng(7))
    with h5py.File(inp["gts"], "r+") as hf:
        del hf.attrs["library_sha"]
    r = _run(inp, tmp_path / "o.json")
    assert r.returncode != 0
    assert "ABSENT" in (r.stdout + r.stderr)


def test_feature_side_arm_refuses_silently_missing_inputs(tmp_path):
    """`--skip-feature-side` RECORDS the omission; omitting the paths without
    it must refuse rather than quietly drop a pre-declared arm."""
    inp = _write_inputs(tmp_path, np.random.default_rng(8))
    cmd = [sys.executable, "-m", "src.analysis.design3_join",
           "--rf-counts-all-theta", str(inp["all_theta"]),
           "--rf-counts", str(inp["one"]), "--m1-json", str(inp["m1"]),
           "--gtilde", str(inp["gt"]), "--gtilde-smooth", str(inp["gts"]),
           "--null-reductions", str(inp["nr"]), "--d-fs", str(inp["dfs"]),
           "--out", str(tmp_path / "o.json")]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0
    assert "feature-side arm needs" in (r.stdout + r.stderr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_firing_rate_edges_parses_a_json_string_not_characters(tmp_path):
    """gotcha 20's third instance: `list()` of a JSON attr yields CHARACTERS
    and has produced plausible-looking wrong bins twice."""
    p = tmp_path / "gt.h5"
    with h5py.File(p, "w") as hf:
        hf.attrs["firing_rate_edges"] = json.dumps(EDGES)
    assert firing_rate_edges(p) == EDGES
    assert all(isinstance(x, float) for x in firing_rate_edges(p))


def test_qualifying_thetas_are_read_not_recomputed(tmp_path):
    p = tmp_path / "m1.json"
    p.write_text(json.dumps({
        "identity": {"n_compare_seeds": 39},
        "theta_selection": {"floor": 0.2, "endpoint_max_share": 0.8,
                            "selected_theta": 0.5,
                            "grid": [{"theta": 0.5, "qualifies": True},
                                     {"theta": 0.9, "qualifies": False}]}}))
    q = qualifying_thetas(p)
    assert q["qualifying"] == [0.5]


def test_floor_report_counts_a_real_exclusion():
    """The floor excludes nothing on the real data, so the fixture has to
    plant an exclusion or the table's own correctness is untested."""
    fire = np.full((2, 10), 1e-2)
    fire[0, :3] = 1e-9                             # below 200/1e6 = 2e-4
    logM = np.zeros((2, 10))
    d = np.ones((2, 10))
    rep = floor_report(fire, N_ROWS, EDGES, logM, d, None)
    assert rep["total_excluded"] == 3
    assert sum(rep["per_seed"][0]["n_excluded_per_firing_bin"]) == 3
    assert rep["per_seed"][1]["n_excluded_by_floor"] == 0
    assert rep["keep"].sum() == 17


def test_ak8_low_M_cut_is_within_bin_not_unconditional():
    """AK8: an unconditional low-`M` cut would select low-firing features and
    rediscover the confound the design controls. The fixture makes `M` and
    firing rate strongly dependent so the two cuts must disagree."""
    rng = np.random.default_rng(11)
    n = 2000
    fire = 10 ** rng.uniform(-3.4, -1.2, (1, n))
    # M rises steeply with firing rate: an unconditional bottom decile would
    # be almost entirely the lowest-firing bin.
    M = (fire * 1e4) ** 2 * rng.lognormal(0, 0.2, (1, n))
    from src.analysis.null_pooling import bin_index
    fb = np.clip(bin_index(fire[0], EDGES), 0, len(EDGES) - 2)[None, :]
    y = np.full((1, n), N_TRIALS, dtype=np.float64)     # every feature high r_f
    keep = np.ones((1, n), bool)
    res = ak8_discovery_set(y, M, fire, fb, keep, N_TRIALS)
    got = res["per_seed"][0]["firing_bin_counts_discovery"]
    uncond = np.bincount(fb[0][M[0] <= np.quantile(M[0], 0.10)],
                         minlength=len(EDGES) - 1)
    assert sum(got) > 0
    # the within-bin set is spread across bins; the unconditional one is not
    assert sum(1 for c in got if c > 0) > sum(1 for c in uncond if c > 0)


# ---------------------------------------------------------------------------
# AMENDMENT AS -- the three intervals (§2), the outcome scale (§3), the `d_fs`
# share on the estimand's own instrument (§4), and F1's model-free panel (§10).
# ---------------------------------------------------------------------------

def test_three_intervals_are_reported_with_the_pre_registered_one_first(joined):
    iv = joined["doc"]["intervals"]
    keys = list(iv)
    assert keys.index("1_binomial_fixed_effect_PRE_REGISTERED") < \
           keys.index("2_dispersion_corrected_fixed_effect") < \
           keys.index("3_across_seed_random_effects")
    assert "NOT a licence to select" in iv["reporting_rule"]
    for k in ("1_binomial_fixed_effect_PRE_REGISTERED",
              "2_dispersion_corrected_fixed_effect",
              "3_across_seed_random_effects"):
        assert len(iv[k]["ci95_logit"]) == 2
        assert iv[k]["ci95_logit"][0] <= iv[k]["ci95_logit"][1]


def test_I2_is_RECOMPUTED_from_the_betabinomial_ses_not_scaled(
        joined_overdispersed):
    """The decisive number of §2: I2 is a function of the WITHIN-study
    variances, so it must come from the beta-binomial SEs directly.

    Read on the OVERDISPERSED fixture, and the docstring of
    `joined_overdispersed` says why: on binomial counts this test compared two
    numbers 0.04 apart with MKL and failed with OpenBLAS, because one anchor
    was dropped from the pooling there."""
    doc = joined_overdispersed["doc"]
    iv = doc["intervals"]
    bb = iv["2_dispersion_corrected_fixed_effect"]
    assert "I2_percent_RECOMPUTED_FROM_BETABINOMIAL_SES" in bb
    assert bb["source"].startswith("the beta-binomial fits DIRECTLY")
    # EVERY anchor is pooled. Without this the comparison below is between
    # poolings of different anchors, which is what the binomial fixture gave.
    assert bb["df"] == N_SEEDS - 1
    rows = doc["primary"]["per_seed"]
    assert all(0.15 < r["betabinomial"]["rho"] < 0.45 for r in rows)
    # RECOMPUTED: the formula is written out here, on the per-anchor
    # beta-binomial estimates and SEs the document stores, and not taken from
    # the code under test (gotcha 32b).
    e = np.array([r["betabinomial"]["b1_z"] for r in rows])
    se = np.array([r["betabinomial"]["se_b1_z"] for r in rows])
    w = 1.0 / se ** 2
    fe = (w * e).sum() / w.sum()
    Q = (w * (e - fe) ** 2).sum()
    want = max(0.0, (Q - (N_SEEDS - 1)) / Q) * 100
    assert bb["I2_percent_RECOMPUTED_FROM_BETABINOMIAL_SES"] == pytest.approx(
        want, rel=1e-9)
    # NOT THE BINOMIAL'S, AND NOT A SCALING OF IT: a wider within-seed SE
    # lowers I2 at fixed spread, and at rho = 0.3 it is wider by a margin no
    # library moves (59.70 against 95.61, measured with MKL and with OpenBLAS).
    assert bb["I2_percent_RECOMPUTED_FROM_BETABINOMIAL_SES"] < \
           iv["1_binomial_fixed_effect_PRE_REGISTERED"]["I2_percent"] - 10.0


def test_interval_agreement_uses_the_width_scale_and_can_fail_both_ways():
    """The endpoint sits near zero, so a relative comparison against the
    endpoint is undefined in the limit. The scale is the interval WIDTH, and
    the predicate must be able to come out either way."""
    from src.analysis.design3_join import interval_agreement
    same = interval_agreement([0.0, 1.0], [0.01, 0.99], "a", "b")
    assert same["both_within_10pct"] is True
    diff = interval_agreement([0.0, 1.0], [0.5, 1.5], "a", "b")
    assert diff["both_within_10pct"] is False
    # a near-zero endpoint must not blow the comparison up
    tiny = interval_agreement([-5e-5, 0.01124], [4.1e-5, 0.011173], "a", "b")
    assert tiny["lo_gap_over_width"] < 0.05


def test_the_outcome_scale_is_reported_at_every_theta(joined):
    for row in joined["doc"]["theta_curve"]["rows"]:
        pp = row["marginal_effect_percentage_points"]
        assert pp["units"].startswith("percentage points")
        assert len(pp["fixed_effect_ci95"]) == 2
        assert "meta_b1_z_betabinomial" in row
        assert 0.0 <= row["rho_median"] < 1.0


def test_the_marginal_effect_carries_its_honesty_clause_into_the_artifact(joined):
    me = joined["doc"]["primary"]["per_seed"][0]["marginal_effect"]
    assert me["outcome_blind"] is False
    assert "never instead of" in me["honesty_note"]
    # and `b1_z` is still present beside it -- alongside, not instead
    assert "b1_z" in joined["doc"]["primary"]["per_seed"][0]


def test_d_fs_share_is_on_the_estimand_s_own_instrument(joined):
    d = joined["doc"]["primary"]["d_fs_variance_share"]
    assert "UNIT ARTIFACT" in d["why"]
    for k in ("ratio_d_fs_over_spline", "ratio_d_fs_over_logM"):
        assert set(d[k]) == {"median", "min", "max"}
    sd = joined["doc"]["primary"]["sd_d_fs_across_seeds"]
    assert len(sd["values"]) == N_SEEDS


def test_the_model_free_panel_is_built_on_the_corrector_strata(joined):
    from src.analysis.corrector_scan import BIN_EDGES
    panel = joined["doc"]["model_free_panel"]
    assert panel["strata_edges"] == list(BIN_EDGES)
    assert panel["strata_source"] == "corrector_scan.BIN_EDGES"
    assert panel["predictor_deciles_on"] == "log M"
    assert "AL7" in panel["translation"]
    assert len(panel["per_seed"]) == N_SEEDS


def test_the_panel_verdict_has_four_reachable_outcomes(joined):
    """A two-valued monotonicity verdict on an ordered comparison is not total
    (gotcha 24), and FLAT_WITHIN_CI exists so a flat panel is not reported as
    a failure of the instrument."""
    panel = joined["doc"]["model_free_panel"]
    allowed = {"MONOTONE_INCREASING", "MONOTONE_DECREASING", "NEITHER",
               "FLAT_WITHIN_CI", "UNDEFINED_TOO_FEW_DECILES"}
    for r in panel["per_seed"]:
        for st in r["strata"]:
            if "shape" in st:
                assert st["shape"] in allowed, st["shape"]
    assert "strongest form the negative result can take" in \
        panel["pre_declared_reading"]


def test_the_panel_detects_a_planted_monotone_relationship():
    """THE DISCRIMINATION TEST. The verdict must be able to say INCREASING
    when the relationship is increasing, or FLAT_WITHIN_CI is unfalsifiable."""
    from src.analysis.design3_join import model_free_panel
    rng = np.random.default_rng(41)
    n = 6000
    logM = rng.normal(size=(1, n))
    fire = np.full((1, n), 5e-3)
    p = 1.0 / (1.0 + np.exp(-(2.0 * logM)))
    y = rng.binomial(N_TRIALS, p).astype(float)
    panel = model_free_panel(y, logM, fire, np.ones((1, n), bool), N_TRIALS,
                             list(EDGES))
    shapes = [st["shape"] for st in panel["per_seed"][0]["strata"]
              if "shape" in st]
    assert "MONOTONE_INCREASING" in shapes, shapes


def test_the_panel_reports_a_null_relationship_as_FLAT_not_as_NEITHER():
    from src.analysis.design3_join import model_free_panel
    rng = np.random.default_rng(42)
    n = 6000
    logM = rng.normal(size=(1, n))
    fire = np.full((1, n), 5e-3)
    y = rng.binomial(N_TRIALS, 0.5, size=(1, n)).astype(float)
    panel = model_free_panel(y, logM, fire, np.ones((1, n), bool), N_TRIALS,
                             list(EDGES))
    shapes = [st["shape"] for st in panel["per_seed"][0]["strata"]
              if "shape" in st]
    assert "FLAT_WITHIN_CI" in shapes, shapes


# ---------------------------------------------------------------------------
# AMENDMENT AT -- the aligned smooth arm, the bound under the headline fit,
# FLAT's tolerance, and `d_fs` across the grid.
#
# Each new gate below gets a fixture that CAN exhibit what it refuses, and
# each new derived reading gets an input on which it must come out the OTHER
# way -- because an instrument capable of only one outcome is as useless as
# one capable of none (gotcha 25).
# ---------------------------------------------------------------------------

def test_the_smooth_gate_admits_aligned_as_well_as_fixfp(tmp_path):
    """AT A1 widens a single literal into a marker SET. The widening must
    admit the new generation..."""
    inp = _write_inputs(tmp_path, np.random.default_rng(7),
                        smooth_name="gt_smooth_aligned.h5")
    r = _run(inp, tmp_path / "o.json")
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    doc = json.loads((tmp_path / "o.json").read_text())
    gen = doc["identity"]["smooth_arm_generation"]
    assert gen["marker_present"] == ["_aligned"]
    assert set(gen["admissible_markers"]) == {"_fixfp", "_aligned"}


def test_the_widened_smooth_gate_still_refuses_an_unmarked_name(tmp_path):
    """...and it must NOT admit everything. A gate that can only pass is the
    mirror of a check that can only fail (gotcha 25c), and widening a gate is
    exactly the edit that produces one."""
    inp = _write_inputs(tmp_path, np.random.default_rng(8),
                        smooth_name="gt_smooth_nodegen.h5")
    r = _run(inp, tmp_path / "o.json")
    assert r.returncode != 0
    assert "generation markers" in (r.stdout + r.stderr)


def test_the_smooth_arm_scorecard_reports_both_halves_derived(joined):
    """P1 is a RATIO and P2 is a SIGN; both outcomes must follow from the
    numbers printed beside them, not from a sentence."""
    sc = joined["doc"]["prediction_scorecard"][
        "AT_P1_P2_the_re_pointed_smooth_arm"]
    p1, p2 = sc["P1"], sc["P2"]
    assert (p1["outcome"] == "HIT") == (p1["measured"] < p1["threshold"])
    assert (p2["outcome"] == "HIT") == (p2["sign_smooth"] == p2["sign_hardbin"])
    ratio = abs(sc["aligned_value"] - sc["superseded_value_on_fixfp"]) \
        / abs(sc["superseded_value_on_fixfp"])
    assert abs(ratio - sc["relative_move"]) < 1e-12


def test_the_outcome_scale_bound_is_a_max_over_the_grid_under_both_fits(joined):
    """AT A2. The bound must be the MAXIMUM of the per-theta series it also
    prints, under each fit -- computed from the series, not carried."""
    b = joined["doc"]["outcome_scale_bound"]
    per = b["per_theta"]
    assert len(per) == len(joined["doc"]["theta_curve"]["qualifying_thetas"])
    mb = max(abs(r["binomial_pp"]) for r in per)
    md = max(abs(r["betabinomial_pp"]) for r in per)
    key = "binomial_PRE_REGISTERED_INTERVAL_SUPERSEDED_AS_A_MAGNITUDE"
    assert abs(b[key]["max_abs_pp"] - mb) < 1e-12
    assert abs(b["betabinomial_HEADLINE_MAGNITUDE"]["max_abs_pp"] - md) < 1e-12
    at = b["betabinomial_HEADLINE_MAGNITUDE"]["at_theta"]
    hit = [r for r in per if r["theta"] == at][0]
    assert abs(abs(hit["betabinomial_pp"]) - md) < 1e-12


def test_the_dispersion_factor_is_measured_per_theta_not_carried(joined):
    """AS13 measured 2.1x at the reference theta. The brief forbids carrying
    it across the grid, so the factor must come from the per-theta values --
    and the DIFFERENCE must be reported too, because the ratio's denominator
    can vanish inside the qualifying range."""
    b = joined["doc"]["outcome_scale_bound"]
    f = b["factor_betabinomial_over_binomial"]
    per_theta = [r["ratio_abs"] for r in b["per_theta"]]
    assert f["values"] == pytest.approx(per_theta)
    assert (f["n_thetas_where_the_correction_SHRINKS_the_magnitude"]
            == sum(1 for v in per_theta if v < 1.0))
    d = b["difference_pp_is_the_honest_summary"]
    assert d["values"] == pytest.approx(
        [r["betabinomial_pp"] - r["binomial_pp"] for r in b["per_theta"]])


def test_the_AS14_comparison_is_derived_from_the_measured_bound(joined):
    b = joined["doc"]["outcome_scale_bound"]["comparison_to_AS14"]
    assert b["betabinomial_exceeds_AS14"] == (
        b["betabinomial_bound_pp"] > b["AS14_published_bound_pp"])
    assert b["betabinomial_below_1_pp"] == (b["betabinomial_bound_pp"] < 1.0)


def test_gamma_has_a_theta_curve_with_all_three_treatments(joined):
    """AT A4: `gamma` gets everything `b1` got. Anything less characterises
    the null term thoroughly and the material one thinly."""
    c = joined["doc"]["d_fs_across_the_grid"]
    per = c["per_theta"]
    assert len(per) == len(joined["doc"]["theta_curve"]["qualifying_thetas"])
    for r in per:
        for k in ("gamma_binomial_fe", "gamma_binomial_ci95",
                  "gamma_across_seed_ci95", "gamma_betabinomial_fe",
                  "gamma_betabinomial_ci95", "delta_r_f_pp_per_SD_d_fs",
                  "sd_ratio_d_fs_over_logM", "sd_ratio_d_fs_over_spline",
                  "sd_ratio_logM_over_spline", "ranked_terms"):
            assert k in r, (k, r["theta"])
        assert len(r["ranked_terms"]) == 3


def test_gamma_sign_unanimity_is_derived_from_the_gammas(joined):
    c = joined["doc"]["d_fs_across_the_grid"]
    signs = {("POSITIVE" if r["gamma_binomial_fe"] > 0 else
              "NEGATIVE" if r["gamma_binomial_fe"] < 0 else "ZERO")
             for r in c["per_theta"]}
    assert c["gamma_sign_is_unanimous_across_the_grid"] == (len(signs) == 1)
    assert sorted(signs) == c["gamma_signs_observed"]


def test_P6_is_derived_from_the_per_theta_minimum(joined):
    c = joined["doc"]["d_fs_across_the_grid"]["sd_ratio_d_fs_over_logM"]
    per = [r["sd_ratio_d_fs_over_logM"]
           for r in joined["doc"]["d_fs_across_the_grid"]["per_theta"]]
    assert abs(c["min"] - min(per)) < 1e-12
    assert c["exceeds_1_at_every_theta"] == all(v > 1.0 for v in per)
    assert c["n_thetas_below_1"] == sum(1 for v in per if v < 1.0)
    sc = joined["doc"]["prediction_scorecard"][
        "AT_P6_d_fs_sd_ratio_exceeds_logM_at_every_theta"]
    assert (sc["outcome"] == "HIT") == (sc["min_over_grid"] > 1.0)


def test_the_d_fs_outcome_scale_is_emitted_per_raw_unit_AND_per_SD(joined):
    """SD(d_fs) is not 1, so the two conversions differ -- by ~24% on the real
    data, small enough to look like rounding and large enough to matter. Both
    must be present, and the per-SD figure must be the raw one rescaled."""
    pp = joined["doc"]["primary"]["marginal_effect_d_fs_percentage_points"]
    raw = pp["per_raw_unit_of_d_fs"]["fixed_effect"]
    per_sd = pp["per_SD_of_d_fs"]["fixed_effect"]
    sds = pp["sd_d_fs_used_per_seed"]
    assert len(sds) == N_SEEDS
    assert raw != per_sd
    lo, hi = sorted((min(sds) * raw, max(sds) * raw))
    assert lo - 1e-9 <= per_sd <= hi + 1e-9
    assert pp["outcome_blind"] is False
    assert "alongside" in pp["honesty_note"].lower()


def test_the_term_ordering_is_ranked_by_the_measured_sds(joined):
    """'d_fs outweighs alignment' must not travel without 'firing rate
    outweighs d_fs'. The ordering is therefore emitted as an ordering, and it
    must follow from the SDs printed beside it."""
    t = joined["doc"]["primary"]["term_ordering"]
    med = {k: v["median"] for k, v in t["sd_of_contribution"].items()}
    assert t["ranked_terms"] == sorted(med, key=lambda k: -med[k])
    assert t["ordering_is_spline_then_d_fs_then_logM"] == (
        t["ranked_terms"] == ["s_hat(log10 phi)", "gamma * d_fs",
                              "b1 * z(log M)"])
    a = t["one_over_median_d_fs_over_spline"]
    b = t["ratios"]["spline_over_d_fs_from_per_seed_reciprocal"]["median"]
    assert a > 0 and b > 0
    assert "median(1/x) != 1/median(x)" in t["reciprocal_warning"]


def test_flat_tolerance_is_converted_into_the_units_it_constrains():
    """AT A3 item 1. A verdict rule whose tolerance is tighter than the
    effect being characterised is an incomplete instrument in the RULE.

    NOT run off the `joined` fixture: at N_FEAT = 400 every stratum returns
    TOO_FEW_FEATURES, so the fixture has no scored cells and cannot exhibit
    the condition (gotcha 16). The real panel has 100.
    """
    from src.analysis.design3_join import model_free_panel
    rng = np.random.default_rng(53)
    n = 4000
    logM = rng.normal(size=(1, n))
    fire = np.full((1, n), 5e-3)
    y = rng.binomial(N_TRIALS, 0.5, size=(1, n)).astype(float)
    panel = model_free_panel(y, logM, fire, np.ones((1, n), bool), N_TRIALS,
                             list(EDGES), comparison_pp=0.352)
    ft = panel["flat_tolerance_report"]
    assert ft["n_scored_cells"] > 0
    assert ft["tolerance_is_tighter_than_the_comparison"] == (
        ft["tolerance_pp_of_r_f"]["median"] < ft["comparison_pp"])
    assert (ft["observed_spread_over_tolerance"]["n_cells_below_1_i_e_FLAT"]
            == panel["n_flat_within_ci"])
    # the tolerance really is in pp of r_f: 2 x the median half-width
    st = [s for s in panel["per_seed"][0]["strata"] if "shape" in s][0]
    assert abs(200.0 * st["median_ci_half_width"]
               - ft["tolerance_pp_of_r_f"]["median"]) < 1e-9


def test_a_panel_with_no_scored_cells_says_so_rather_than_omitting_the_report(
        joined):
    """Absence reported as absence, never as agreement (gotcha 15/28). The
    400-feature fixture scores nothing, and the artifact must say that
    instead of dropping the key."""
    ft = joined["doc"]["model_free_panel"]["flat_tolerance_report"]
    assert ft == {"status": "NO_SCORED_CELLS"}
    p = joined["doc"]["model_free_panel"]
    assert p["shape_counts"] == {}
    # the join stratifies on corrector_scan.BIN_EDGES, NOT on the fixture's
    # own edges -- which is the fact AT A3 item 3 had to establish, so the
    # test reads the panel's declared edges rather than assuming either.
    from src.analysis.corrector_scan import BIN_EDGES
    assert p["strata_edges"] == list(BIN_EDGES)
    assert p["status_counts"]["TOO_FEW_FEATURES"] == \
        N_SEEDS * (len(BIN_EDGES) - 1)


def test_the_flat_tolerance_verdict_can_come_out_either_way():
    """The derived boolean must be a function of the comparison, not a
    constant. Same panel, two comparison values, two answers."""
    from src.analysis.design3_join import model_free_panel
    rng = np.random.default_rng(51)
    n = 4000
    logM = rng.normal(size=(1, n))
    fire = np.full((1, n), 5e-3)
    y = rng.binomial(N_TRIALS, 0.5, size=(1, n)).astype(float)
    loose = model_free_panel(y, logM, fire, np.ones((1, n), bool), N_TRIALS,
                             list(EDGES), comparison_pp=1e-6)
    tight = model_free_panel(y, logM, fire, np.ones((1, n), bool), N_TRIALS,
                             list(EDGES), comparison_pp=1e6)
    assert loose["flat_tolerance_report"][
        "tolerance_is_tighter_than_the_comparison"] is False
    assert tight["flat_tolerance_report"][
        "tolerance_is_tighter_than_the_comparison"] is True


def test_the_cell_count_is_reported_as_cells_not_as_a_factorisation(joined):
    """AS16's '100 informative cells' invited reading 100 as deciles x strata.
    The artifact must say what the cells are and account for all of them."""
    p = joined["doc"]["model_free_panel"]
    assert p["n_cells_total"] == sum(p["shape_counts"].values()) + \
        sum(p["status_counts"].values())
    assert p["n_cells_total"] == N_SEEDS * (len(p["strata_edges"]) - 1)
    assert "NOT a factorisation" in p["cell_definition"]


def test_flat_reachability_fires_under_a_null_and_not_under_a_trend():
    """AT A3 item 2 / gotcha 28. Reachability on an n = 6000 fixture is not
    reachability at the real cell sizes, so the demonstration is re-run at
    them -- and it must be capable of BOTH answers."""
    from src.analysis.design3_join import (flat_reachability_at_real_cell_sizes,
                                           model_free_panel)
    ok = flat_reachability_at_real_cell_sizes([200, 3000], N_TRIALS,
                                              list(EDGES), n_reps=5)
    assert ok["flat_is_reachable_at_every_real_cell_size"] is True
    assert all(r["flat_rate"] > 0 for r in ok["per_cell_size"])
    rng = np.random.default_rng(52)
    n = 3000
    logM = rng.normal(size=(1, n))
    p = 1.0 / (1.0 + np.exp(-(2.0 * logM)))
    y = rng.binomial(N_TRIALS, p).astype(float)
    pan = model_free_panel(y, logM, np.full((1, n), 5e-3),
                           np.ones((1, n), bool), N_TRIALS, list(EDGES))
    assert pan["n_flat_within_ci"] == 0


def test_the_primary_comparison_can_say_both_things(tmp_path):
    """AT changes a sensitivity arm's input and adds emissions; the PRIMARY
    must not move. A comparison that can only report agreement would not be
    evidence of that (gotcha 25)."""
    from src.analysis.design3_join import assert_primary_bit_identical
    primary = {"per_seed": [
        {"seed": 0, "b1_z": 1.0, "se_b1_z": 0.1, "gamma_d_fs": 2.0,
         "se_gamma": 0.2, "deviance": 3.0, "loglik": -4.0, "lambda": 5.0,
         "edf": 6.0}]}
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"primary": primary}))
    assert assert_primary_bit_identical(primary, ref)["verdict"] == \
        "BIT_IDENTICAL"
    moved = json.loads(json.dumps(primary))
    moved["per_seed"][0]["b1_z"] = 1.0 + 1e-15
    got = assert_primary_bit_identical(moved, ref)
    assert got["verdict"] == "DIFFERS"
    assert got["disagreements"][0]["field"] == "b1_z"
    gone = assert_primary_bit_identical(primary, tmp_path / "nope.json")
    assert gone["status"] == "REFERENCE_ABSENT"
    assert "verdict" not in gone


def test_the_primary_comparison_is_positional_not_a_summary(tmp_path):
    """A permuted seed axis has the same totals and the same multiset of
    values; only a positional comparison can see it (gotcha 27)."""
    from src.analysis.design3_join import assert_primary_bit_identical
    rows = [{"seed": i, "b1_z": float(i), "se_b1_z": 0.1, "gamma_d_fs": 2.0,
             "se_gamma": 0.2, "deviance": 3.0, "loglik": -4.0, "lambda": 5.0,
             "edf": 6.0} for i in range(4)]
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"primary": {"per_seed": rows}}))
    swapped = json.loads(json.dumps(rows))
    swapped[1]["b1_z"], swapped[2]["b1_z"] = \
        swapped[2]["b1_z"], swapped[1]["b1_z"]
    got = assert_primary_bit_identical({"per_seed": swapped}, ref)
    assert got["verdict"] == "DIFFERS"
    assert {d["seed"] for d in got["disagreements"]} == {1, 2}


def test_the_feature_side_axis_check_is_positional():
    """AT A1 item 3. The check must refuse a duplicated checkpoint and a
    label/position mismatch -- neither of which changes the RECORD COUNT."""
    from src.analysis.design3_join import verify_feature_side_seed_provenance

    def rec(pos, label, ck=None, ri=None, sha=None):
        ck = ck or f"seed_{label}"
        ri = ri or f"row_index_{ck}_x.h5"
        return {"anchor_position": pos, "anchor_label": str(label),
                "ckpt_dir": ck, "row_index": ri,
                "checkpoint_sha256_16": sha or f"sha{label}",
                "ckpt_dir_matches_anchor_label": ck == f"seed_{label}",
                "row_index_name_carries_the_ckpt_dir": ck in ri}

    good = [rec(i, i) for i in range(4)]
    assert verify_feature_side_seed_provenance(good)["verdict"] == \
        "VERIFIED_POSITIONALLY"
    dup = [rec(0, 0), rec(1, 1, sha="sha0"), rec(2, 2), rec(3, 3)]
    out_dup = verify_feature_side_seed_provenance(dup)
    assert out_dup["verdict"] == "AMBIGUOUS"
    assert out_dup["n_anchors"] == 4
    wrong = [rec(0, 0), rec(1, 1, ck="seed_2"), rec(2, 2), rec(3, 3)]
    out = verify_feature_side_seed_provenance(wrong)
    assert out["verdict"] == "AMBIGUOUS"
    assert out["n_anchors"] == 4
    assert any("seed_2" in p for p in out["problems"])


def test_the_degenerate_inheritance_check_reports_its_passing(joined):
    """A gate that only speaks when it fails leaves 'was it checked?'
    unanswerable from the artifact."""
    d = joined["doc"]["seed_alignment"]["degenerate_flag_order"]
    assert d["verdict"] == "ELEMENTWISE_EQUAL"
    assert d["argmax_concept_cells_compared"] == N_SEEDS * N_FEAT
    assert (d["argmax_concept_cells_equal"]
            == d["argmax_concept_cells_compared"])


def test_the_scored_against_existing_data_standing_is_carried(joined):
    """AT1 splits the six predictions into SEALED and
    SCORED_AGAINST_EXISTING_DATA, and the weaker standing must travel in the
    artifact rather than only in the amendment -- a caveat that travels
    separately does not travel."""
    sc = joined["doc"]["prediction_scorecard"]
    for k in ("AT_P3_betabinomial_bound_above_0352pp_and_below_1pp",
              "AT_P4_FLAT_tolerance_is_tighter_than_0352pp",
              "AT_P6_d_fs_sd_ratio_exceeds_logM_at_every_theta"):
        assert "SCORED_AGAINST_EXISTING_DATA" in sc[k]["standing"]
        assert "EXCLUDED from the bin tallies" in sc[k]["standing"]
    assert sc["AT_P5_gamma_sign_unanimous_across_the_grid"][
        "standing"].startswith("SEALED")
    for half in ("P1", "P2"):
        assert sc["AT_P1_P2_the_re_pointed_smooth_arm"][half][
            "standing"].startswith("SEALED")


# ---------------------------------------------------------------------------
# AT's extracted summaries, exercised on series whose answers are KNOWN.
#
# Seven of nineteen mutations survived while these lived inside `main()`, for
# one reason in every case: the only fixture available was whatever the fit
# happened to produce, and it could not exhibit the disputed condition
# (gotcha 16). Extracting them is what makes the tests able to fail.
# ---------------------------------------------------------------------------

def _curve_row(theta, binom_pp, bb_pp, gamma=0.1, gamma_bb=0.1,
               d_over_m=12.0, d_over_s=0.34, m_over_s=0.03, gamma_pp=2.5):
    """One theta's worth of the curve, with every field the summaries read."""
    def meta(v):
        return {"fixed_effect": v, "fixed_effect_ci95": [v - 1, v + 1],
                "across_seed_mean": v, "across_seed_ci95": [v - 2, v + 2],
                "I2_percent": 0.0}

    def pp(v):
        return {"fixed_effect": v, "fixed_effect_ci95": [v - 0.01, v + 0.01],
                "across_seed_mean": v, "across_seed_ci95": [v - 0.02, v + 0.02]}

    terms = {"s_hat(log10 phi)": 1.0, "gamma * d_fs": d_over_s,
             "b1 * z(log M)": m_over_s}
    return {
        "theta": theta,
        "meta_b1_z": meta(binom_pp),
        "marginal_effect_percentage_points": pp(binom_pp),
        "marginal_effect_percentage_points_betabinomial": pp(bb_pp),
        "meta_gamma_d_fs": meta(gamma),
        "meta_gamma_d_fs_betabinomial": meta(gamma_bb),
        "marginal_effect_d_fs_percentage_points": {
            "per_SD_of_d_fs": pp(gamma_pp)},
        "rho_median": 0.4,
        "term_ordering": {
            "ranked_terms": sorted(terms, key=lambda k: -terms[k]),
            "ratios": {"d_fs_over_logM": {"median": d_over_m},
                       "d_fs_over_spline": {"median": d_over_s},
                       "logM_over_spline": {"median": m_over_s}}},
    }


def test_the_bound_is_the_max_and_not_the_first_or_the_last():
    """A summary that reads index 0 is right whenever the maximum happens to
    sit there, which is exactly what a fit-produced fixture cannot rule out."""
    from src.analysis.design3_join import outcome_scale_bound
    rows = [_curve_row(0.3, 0.10, 0.05), _curve_row(0.5, 0.40, 0.20),
            _curve_row(0.7, 0.25, 0.90), _curve_row(0.9, -0.05, -0.10)]
    b = outcome_scale_bound(rows)
    key = "binomial_PRE_REGISTERED_INTERVAL_SUPERSEDED_AS_A_MAGNITUDE"
    assert b[key]["max_abs_pp"] == pytest.approx(0.40)
    assert b[key]["at_theta"] == 0.5
    assert b["betabinomial_HEADLINE_MAGNITUDE"]["max_abs_pp"] == \
        pytest.approx(0.90)
    assert b["betabinomial_HEADLINE_MAGNITUDE"]["at_theta"] == 0.7
    # the two fits must be allowed to peak at DIFFERENT thetas -- they do on
    # the real data, and a summary that assumed otherwise would hide it
    assert b[key]["at_theta"] != b["betabinomial_HEADLINE_MAGNITUDE"]["at_theta"]


def test_the_bound_takes_the_absolute_value_not_the_signed_maximum():
    """The curve goes negative at high theta; a signed max would report the
    positive hump and miss a larger negative excursion."""
    from src.analysis.design3_join import outcome_scale_bound
    rows = [_curve_row(0.3, 0.10, 0.10), _curve_row(0.9, -0.80, -0.70)]
    b = outcome_scale_bound(rows)
    key = "binomial_PRE_REGISTERED_INTERVAL_SUPERSEDED_AS_A_MAGNITUDE"
    assert b[key]["max_abs_pp"] == pytest.approx(0.80)
    assert b[key]["at_theta"] == 0.9
    assert b[key]["signed_pp_at_that_theta"] == pytest.approx(-0.80)


def test_the_AS14_comparison_can_come_out_either_way():
    """Asserting the comparison is the failure mode; both answers must be
    reachable from the same code."""
    from src.analysis.design3_join import (AS14_BINOMIAL_BOUND_PP,
                                           outcome_scale_bound)
    assert AS14_BINOMIAL_BOUND_PP == 0.352
    below = outcome_scale_bound([_curve_row(0.7, 0.10, 0.10)])
    assert below["comparison_to_AS14"]["betabinomial_exceeds_AS14"] is False
    assert below["comparison_to_AS14"]["betabinomial_below_1_pp"] is True
    above = outcome_scale_bound([_curve_row(0.7, 0.10, 0.50)])
    assert above["comparison_to_AS14"]["betabinomial_exceeds_AS14"] is True
    huge = outcome_scale_bound([_curve_row(0.7, 0.10, 2.50)])
    assert huge["comparison_to_AS14"]["betabinomial_below_1_pp"] is False


def test_the_dispersion_factor_reports_shrink_and_grow_separately():
    from src.analysis.design3_join import outcome_scale_bound
    rows = [_curve_row(0.3, 1.0, 0.5), _curve_row(0.5, 1.0, 2.0),
            _curve_row(0.7, 1.0, 3.0)]
    f = outcome_scale_bound(rows)["factor_betabinomial_over_binomial"]
    assert f["n_thetas_where_the_correction_SHRINKS_the_magnitude"] == 1
    assert f["n_thetas_where_it_grows_it"] == 2
    d = outcome_scale_bound(rows)["difference_pp_is_the_honest_summary"]
    assert d["values"] == pytest.approx([-0.5, 1.0, 2.0])


def test_gamma_sign_unanimity_can_be_false():
    """The whole content of P5 is a boolean, so it must be a function of the
    gammas and not a constant."""
    from src.analysis.design3_join import d_fs_grid_summary
    same = d_fs_grid_summary([_curve_row(0.3, 1, 1, gamma=0.1),
                              _curve_row(0.7, 1, 1, gamma=0.2)])
    assert same["gamma_sign_is_unanimous_across_the_grid"] is True
    assert same["gamma_signs_observed"] == ["POSITIVE"]
    mixed = d_fs_grid_summary([_curve_row(0.3, 1, 1, gamma=0.1),
                               _curve_row(0.7, 1, 1, gamma=-0.2)])
    assert mixed["gamma_sign_is_unanimous_across_the_grid"] is False
    assert mixed["gamma_signs_observed"] == ["NEGATIVE", "POSITIVE"]
    # and the beta-binomial column is scored SEPARATELY, not inherited
    split = d_fs_grid_summary([_curve_row(0.3, 1, 1, gamma=0.1, gamma_bb=0.1),
                               _curve_row(0.7, 1, 1, gamma=0.2, gamma_bb=-0.1)])
    assert split["gamma_sign_is_unanimous_across_the_grid"] is True
    assert split["gamma_sign_is_unanimous_betabinomial"] is False


def test_P6_needs_EVERY_theta_not_merely_one():
    """`all` vs `any` is the whole predicate. On the real grid four thetas sit
    below 1 while nine sit above, so a summary using `any` would report the
    opposite conclusion and look entirely plausible."""
    from src.analysis.design3_join import d_fs_grid_summary
    every = d_fs_grid_summary([_curve_row(0.3, 1, 1, d_over_m=2.0),
                               _curve_row(0.7, 1, 1, d_over_m=12.0)])
    assert every["sd_ratio_d_fs_over_logM"]["exceeds_1_at_every_theta"] is True
    mixed = d_fs_grid_summary([_curve_row(0.3, 1, 1, d_over_m=0.55),
                               _curve_row(0.7, 1, 1, d_over_m=12.0)])
    r = mixed["sd_ratio_d_fs_over_logM"]
    assert r["exceeds_1_at_every_theta"] is False
    assert r["min"] == pytest.approx(0.55)
    assert r["min_at_theta"] == 0.3
    assert r["n_thetas_below_1"] == 1 and r["n_thetas_above_1"] == 1


def test_the_grid_summary_reports_when_the_ordering_changes():
    """'d_fs outweighs alignment' is a claim about the ordering, and the
    ordering is not guaranteed constant across the grid."""
    from src.analysis.design3_join import d_fs_grid_summary
    const = d_fs_grid_summary([_curve_row(0.3, 1, 1), _curve_row(0.7, 1, 1)])
    assert const["ordering_is_constant_across_the_grid"] is True
    flip = d_fs_grid_summary([
        _curve_row(0.3, 1, 1, d_over_s=0.01, m_over_s=0.06),
        _curve_row(0.7, 1, 1, d_over_s=0.34, m_over_s=0.03)])
    assert flip["ordering_is_constant_across_the_grid"] is False
    assert len(flip["orderings_observed"]) == 2


def test_the_reachability_verdict_is_false_when_flat_never_fires():
    """AT A3 item 2. The claim is that FLAT fires at the real cell sizes. A
    boolean that is True regardless would make the real panel's 100-out-of-100
    unfalsifiable, so it must read False when nothing was scored."""
    from src.analysis.design3_join import flat_reachability_at_real_cell_sizes
    # below 10 * n_deciles, so every stratum returns TOO_FEW_FEATURES and
    # nothing is scored at all -- the honest answer is NOT reachable
    none = flat_reachability_at_real_cell_sizes([50], N_TRIALS, list(EDGES),
                                                n_reps=2)
    assert none["flat_is_reachable_at_every_real_cell_size"] is False
    assert none["per_cell_size"][0]["flat_rate"] is None
    # a strong monotone truth scores cells but never calls them flat
    trend = flat_reachability_at_real_cell_sizes([2000], N_TRIALS, list(EDGES),
                                                 n_reps=2, base_p=0.5)
    assert trend["per_cell_size"][0]["n_scored_cells"] > 0


def test_the_degenerate_inheritance_check_can_say_DIFFERS():
    """Inline, the surrounding raise meant the emitted verdict could only ever
    read ELEMENTWISE_EQUAL -- a reading that cannot disagree with its author."""
    from src.analysis.design3_join import check_degenerate_inheritance
    a = np.arange(20).reshape(5, 4)
    assert check_degenerate_inheritance(a, a.copy())["verdict"] == \
        "ELEMENTWISE_EQUAL"
    # A PERMUTED seed axis: same shape, same total, same multiset of values.
    perm = a[[0, 1, 3, 2, 4]]
    got = check_degenerate_inheritance(perm, a)
    assert got["verdict"] == "DIFFERS"
    assert got["argmax_concept_cells_compared"] == 20
    assert got["argmax_concept_cells_equal"] < 20
    # and a shape mismatch is DIFFERS rather than an exception
    assert check_degenerate_inheritance(a[:3], a)["verdict"] == "DIFFERS"


def test_the_smooth_arm_scorecard_can_report_a_sign_disagreement():
    """P2's only content is whether two signs match, so the fixture has to be
    able to produce a mismatch -- the join fixture plants b1 > 0 in both arms
    and never can."""
    from src.analysis.design3_join import (AS12_SMOOTH_B1Z_ON_FIXFP,
                                           _smooth_arm_scorecard)

    def arm(v):
        return {"meta_b1_z": {"fixed_effect": v,
                              "fixed_effect_ci95": [v - 1, v + 1]}}

    agree = _smooth_arm_scorecard(arm(0.004), 0.0056, None)
    assert agree["P2"]["outcome"] == "HIT"
    disagree = _smooth_arm_scorecard(arm(-0.004), 0.0056, None)
    assert disagree["P2"]["outcome"] == "MISS"
    assert disagree["P2"]["sign_smooth"] == "NEGATIVE"
    # P1 is a ratio against the SUPERSEDED value and must move with it
    tiny = _smooth_arm_scorecard(arm(AS12_SMOOTH_B1Z_ON_FIXFP * 1.5), 0.0056,
                                 None)
    assert tiny["P1"]["measured"] == pytest.approx(0.5)
    assert tiny["P1"]["outcome"] == "HIT"
    huge = _smooth_arm_scorecard(arm(AS12_SMOOTH_B1Z_ON_FIXFP * 100), 0.0056,
                                 None)
    assert huge["P1"]["measured"] == pytest.approx(99.0)
    assert huge["P1"]["outcome"] == "MISS"
    # an arm that did not run is UNSCORED, not a silent HIT
    assert _smooth_arm_scorecard({}, 0.0056, None)["P1"] == "UNSCORED"


# ---------------------------------------------------------------------------
# AMENDMENT AU -- the four arms under both likelihoods, and the permutation
# null that tests F1's THIRD candidate cause.
#
# Both are named functions from the start, not inline in `main()`: AT measured
# that seven of nineteen mutations survive when a summary can only be exercised
# on whatever the fit happens to produce (gotcha 16).
# ---------------------------------------------------------------------------

def _arm(b1, se, pp=None, bb=None, bbse=None):
    """A minimal arm carrying only what `arm_comparison` reads."""
    z = 1.959964

    def meta(v, s):
        return {"fixed_effect": v, "fixed_effect_ci95": [v - z * s, v + z * s],
                "I2_percent": 0.0}

    def ppb(v, s):
        return {"fixed_effect": v, "fixed_effect_ci95": [v - z * s, v + z * s],
                "across_seed_mean": v, "across_seed_ci95": [v - s, v + s]}

    out = {"meta_b1_z": meta(b1, se)}
    if pp is not None:
        out["marginal_effect_percentage_points"] = ppb(*pp)
    if bb is not None:
        out["meta_b1_z_betabinomial"] = meta(bb, bbse)
        out["marginal_effect_percentage_points_betabinomial"] = ppb(bb, bbse)
    return out


def test_arm_comparison_derives_includes_zero_and_can_say_both(tmp_path):
    """AU 1a. The whole content is a per-cell boolean, so it must follow from
    the interval and be reachable in both directions."""
    from src.analysis.design3_join import arm_comparison
    prim = _arm(0.0056, 0.00074, bb=0.0120, bbse=0.00288)      # excludes zero
    sens = {
        "degenerate_excluded": _arm(0.0055, 0.00074, bb=0.0119, bbse=0.00288),
        "smooth_normaliser": _arm(-0.0036, 0.00075, bb=0.0028, bbse=0.00290),
        "feature_side": _arm(-0.0058, 0.00076, bb=0.0010, bbse=0.00290),
    }
    a = arm_comparison(prim, sens)
    assert a["n_arms_fit_both_ways"] == 4
    assert a["binomial"]["n_arms_including_zero"] == 0
    assert a["betabinomial"]["n_arms_including_zero"] == 2
    assert a["betabinomial"]["n_sensitivity_arms_including_zero"] == 2
    assert a["betabinomial"]["n_sensitivity_arms"] == 3
    # the primary is in `rows` but excluded from the SENSITIVITY count
    assert a["betabinomial"]["n_arms_including_zero"] \
        >= a["betabinomial"]["n_sensitivity_arms_including_zero"]
    for r in a["rows"]:
        for tag in ("binomial", "betabinomial"):
            lo, hi = r[tag]["ci95_logit"]
            assert r[tag]["includes_zero"] == (lo <= 0.0 <= hi)
    # and the other direction: nothing covering zero
    tight = {k: _arm(-0.0036, 0.00075, bb=-0.0040, bbse=0.00030)
             for k in sens}
    b = arm_comparison(prim, tight)
    assert b["betabinomial"]["n_sensitivity_arms_including_zero"] == 0


def test_arm_comparison_reports_a_missing_dispersion_fit_as_missing():
    """An arm fit without dispersion must be reported as NOT_FIT, never
    silently counted as not covering zero (gotcha 15)."""
    from src.analysis.design3_join import arm_comparison
    prim = _arm(0.0056, 0.00074, bb=0.0120, bbse=0.00288)
    sens = {"smooth_normaliser": _arm(-0.0036, 0.00075),          # no bb
            "degenerate_excluded": {"status": "NOT_RUN -- x"},
            "feature_side": {"status": "NOT_RUN -- y"}}
    a = arm_comparison(prim, sens)
    smooth = [r for r in a["rows"] if r["arm"] == "smooth_normaliser"][0]
    assert smooth["betabinomial"]["status"].startswith("NOT_FIT")
    assert a["n_arms_fit_both_ways"] == 1
    assert a["betabinomial"]["n_sensitivity_arms"] == 0
    absent = [r for r in a["rows"] if r["arm"] == "degenerate_excluded"][0]
    assert absent["status"].startswith("NOT_RUN")


def test_arm_comparison_carries_the_three_prohibitions_and_the_AK9a_reversal():
    from src.analysis.design3_join import arm_comparison
    a = arm_comparison(_arm(0.0056, 0.00074, bb=0.0120, bbse=0.00288), {})
    assert len(a["prohibitions"]) == 3
    assert any("average" in p for p in a["prohibitions"])
    assert any("tiebreak" in p for p in a["prohibitions"])
    assert any("0.9219%" in p for p in a["prohibitions"])
    # AS/AT read AK9a's direction as STRENGTHENING absence; under the arms
    # framing it is the wrong sign, and the object must say so where it is read
    assert "WRONG SIGN" in a["AK9a_cuts_the_other_way_here"]


def test_the_permutation_null_detects_a_real_trend_and_clears_a_null():
    """AU 3. The instrument must place a planted trend high in its own null
    and a planted null in the middle -- otherwise the percentile it reports
    about the real panel means nothing."""
    from src.analysis.design3_join import permutation_null_for_the_panel
    rng = np.random.default_rng(7)
    n = 3000
    fire = np.full((1, n), 5e-3)
    keep = np.ones((1, n), bool)
    logM = rng.normal(size=(1, n))
    # a planted null: r_f independent of log M, WITH between-feature spread
    p_null = np.clip(rng.beta(2, 2, size=n), 0.01, 0.99)
    y_null = rng.binomial(N_TRIALS, p_null).astype(float)[None, :]
    a = permutation_null_for_the_panel(y_null, logM, fire, keep, N_TRIALS,
                                       list(EDGES), n_perm=60, seed=1)
    # a planted trend
    p_tr = 1.0 / (1.0 + np.exp(-(1.5 * logM[0])))
    y_tr = rng.binomial(N_TRIALS, p_tr).astype(float)[None, :]
    b = permutation_null_for_the_panel(y_tr, logM, fire, keep, N_TRIALS,
                                       list(EDGES), n_perm=60, seed=1)
    pa = a["observed_spread_vs_its_own_null"]["median_percentile"]
    pb = b["observed_spread_vs_its_own_null"]["median_percentile"]
    assert pa < 95.0, pa
    assert pb > 95.0, pb
    assert b["observed_spread_vs_its_own_null"]["ratio_observed_to_null_median"] \
        > a["observed_spread_vs_its_own_null"]["ratio_observed_to_null_median"]


def test_the_permutation_null_preserves_the_cells_own_dispersion():
    """The whole point of a permutation rather than a parametric draw: a
    beta-binomial r_f and a binomial one with the same MEAN must give
    DIFFERENT null spreads, because only the permutation sees the spread."""
    from src.analysis.design3_join import permutation_null_for_the_panel
    rng = np.random.default_rng(11)
    n = 3000
    fire = np.full((1, n), 5e-3)
    keep = np.ones((1, n), bool)
    logM = rng.normal(size=(1, n))
    y_bin = rng.binomial(N_TRIALS, 0.5, size=n).astype(float)[None, :]
    p_disp = np.clip(rng.beta(1, 1, size=n), 0.01, 0.99)
    y_disp = rng.binomial(N_TRIALS, p_disp).astype(float)[None, :]
    a = permutation_null_for_the_panel(y_bin, logM, fire, keep, N_TRIALS,
                                       list(EDGES), n_perm=60, seed=2)
    b = permutation_null_for_the_panel(y_disp, logM, fire, keep, N_TRIALS,
                                       list(EDGES), n_perm=60, seed=2)
    sa = a["empirical_se_of_a_decile_mean_pp"]["median"]
    sb = b["empirical_se_of_a_decile_mean_pp"]["median"]
    assert sb > 2.0 * sa, (sa, sb)
    # and the FLAT rate must fall accordingly -- this is exactly the failure
    # AT12's binomial-null reachability check could not see
    assert (b["flat_rate_under_the_permutation_null"]["pooled"]
            < a["flat_rate_under_the_permutation_null"]["pooled"])


def test_the_empirical_se_is_not_the_wilson_half_width():
    """AT12 compared the tolerance against a Wilson half-width, which measures
    the noise in a POOLED proportion, not in a MEAN OVER FEATURES. They differ
    by a factor the artifact must report rather than hide."""
    from src.analysis.design3_join import permutation_null_for_the_panel
    rng = np.random.default_rng(13)
    n = 3000
    p = np.clip(rng.beta(1, 1, size=n), 0.01, 0.99)
    y = rng.binomial(N_TRIALS, p).astype(float)[None, :]
    out = permutation_null_for_the_panel(
        y, rng.normal(size=(1, n)), np.full((1, n), 5e-3),
        np.ones((1, n), bool), N_TRIALS, list(EDGES), n_perm=40, seed=3)
    assert out["wilson_understates_the_noise_by"]["median"] > 1.5
    for c in out["per_cell"]:
        assert c["empirical_se_over_wilson_half_width"] == pytest.approx(
            c["median_empirical_se_of_a_decile_mean"]
            / c["median_wilson_half_width"])


def test_the_permutation_null_booleans_are_derived_both_ways():
    """AU-Q2 and AU-Q3 are booleans in the artifact; each must be a function of
    the number printed beside it and each must be reachable both ways."""
    from src.analysis.design3_join import permutation_null_for_the_panel
    rng = np.random.default_rng(17)
    n = 3000
    fire = np.full((1, n), 5e-3)
    keep = np.ones((1, n), bool)
    logM = rng.normal(size=(1, n))
    # dispersed + no trend  -> FLAT rate low, percentile middling
    p = np.clip(rng.beta(1, 1, size=n), 0.01, 0.99)
    hard = permutation_null_for_the_panel(
        rng.binomial(N_TRIALS, p).astype(float)[None, :], logM, fire, keep,
        N_TRIALS, list(EDGES), n_perm=60, seed=4)
    # tight + no trend      -> FLAT rate high
    easy = permutation_null_for_the_panel(
        rng.binomial(N_TRIALS, 0.5, size=n).astype(float)[None, :], logM,
        fire, keep, N_TRIALS, list(EDGES), n_perm=60, seed=4)
    for o in (hard, easy):
        r = o["flat_rate_under_the_permutation_null"]["pooled"]
        assert o["AU_Q2_flat_rate_below_5pct"] == (r < 0.05)
        assert o["verdict_rule_can_return_FLAT_on_data_like_this"] == (r >= 0.05)
        m = o["observed_spread_vs_its_own_null"]["median_percentile"]
        assert o["AU_Q3_median_percentile_in_25_75"] == (25.0 <= m <= 75.0)
    assert easy["AU_Q2_flat_rate_below_5pct"] is False
    assert hard["AU_Q2_flat_rate_below_5pct"] is True


def test_the_superseded_binomial_reachability_check_says_so_on_every_return():
    """AT12's check is retained rather than deleted, and it must carry the
    reason it no longer answers the question -- a superseded check with no
    label reads as current.

    NOT asserted through the `joined` fixture: that fixture scores no cells, so
    the call site never runs and the assertion could not fail. MEASURED -- the
    label-removal mutation SURVIVED against the fixture-based version (gotcha
    16). The label therefore lives on the function's own return, where it is
    reachable unconditionally."""
    from src.analysis.design3_join import flat_reachability_at_real_cell_sizes
    out = flat_reachability_at_real_cell_sizes([200], N_TRIALS, list(EDGES),
                                               n_reps=2)
    assert "SUPERSEDED_BY_AU7" in out
    note = out["SUPERSEDED_BY_AU7"].lower()
    assert "does not resemble" in note
    assert "permutation_null" in note
    assert "0.401" in note        # the measured rho the binomial null lacks


# ==========================================================================
# AMENDMENT AV section 5c -- the STRATIFIED permutation
# ==========================================================================

def _av_panel(y, logM, fire, d_fs, **kw):
    from src.analysis.design3_join import (
        stratified_permutation_null_for_the_panel)
    keep = np.ones(y.shape, bool)
    return stratified_permutation_null_for_the_panel(
        y, logM, fire, d_fs, keep, N_TRIALS, list(EDGES),
        n_perm=kw.pop("n_perm", 60), seed=kw.pop("seed", 3), **kw)


def test_stratified_permutation_REMOVES_a_spread_driven_entirely_by_d_fs():
    """THE FIXTURE THAT MAKES THE MEASUREMENT MEAN ANYTHING.

    `d_fs` drives `r_f`; `log M` is a deterministic function of `d_fs` and
    carries NO information of its own. So the decile structure in `log M` is
    entirely a `d_fs` shadow. The UNSTRATIFIED null must call it real (high
    percentile) and the STRATIFIED null must not -- and if the two agreed here,
    the stratified arm could not distinguish the two situations it exists to
    distinguish (gotcha 16).
    """
    rng = np.random.default_rng(11)
    n = 4000
    fire = np.full((1, n), 5e-3)
    d = rng.normal(size=n)
    logM = (d + 1e-6 * rng.normal(size=n))[None, :]     # a pure d_fs shadow
    p = 1.0 / (1.0 + np.exp(-1.5 * d))
    y = rng.binomial(N_TRIALS, p).astype(float)[None, :]
    out = _av_panel(y, logM, fire, d[None, :])
    uns = out["arms"]["unstratified"]
    strat = out["arms"]["d_fs_bins_10"]
    assert uns["median_percentile_of_the_observed_spread"] > 95.0
    assert (strat["median_percentile_of_the_observed_spread"]
            < uns["median_percentile_of_the_observed_spread"])
    assert strat["n_cells_above_their_own_95th"] < uns["n_cells_above_their_own_95th"]


def test_stratified_permutation_KEEPS_a_spread_d_fs_cannot_explain():
    """The mirror, and it is what stops the stratified arm from being a
    machine that always says "explained": `log M` drives `r_f` and `d_fs` is
    independent noise, so blocking on `d_fs` must NOT remove the structure."""
    rng = np.random.default_rng(12)
    n = 4000
    fire = np.full((1, n), 5e-3)
    d = rng.normal(size=n)
    logM = rng.normal(size=(1, n))
    p = 1.0 / (1.0 + np.exp(-1.5 * logM[0]))
    y = rng.binomial(N_TRIALS, p).astype(float)[None, :]
    out = _av_panel(y, logM, fire, d[None, :])
    for key in ("unstratified", "d_fs_bins_5", "d_fs_bins_10"):
        assert out["arms"][key]["median_percentile_of_the_observed_spread"] > 95.0


def test_a_single_block_reproduces_the_unstratified_arm_exactly():
    """One block IS no blocking. If `d_fs_bins_1` were requested it must equal
    the unstratified arm, which is the check that the two paths share one
    estimator rather than merely resembling each other."""
    rng = np.random.default_rng(13)
    n = 1500
    fire = np.full((1, n), 5e-3)
    logM = rng.normal(size=(1, n))
    d = np.zeros(n)                      # every feature in one d_fs bin
    y = rng.binomial(N_TRIALS, 0.5, size=n).astype(float)[None, :]
    out = _av_panel(y, logM, fire, d[None, :], n_dfs_bins_grid=(2,),
                    primary_bins=2)
    a, b = out["arms"]["unstratified"], out["arms"]["d_fs_bins_2"]
    # d_fs is constant, so quantile edges collapse and every feature lands in
    # ONE block -- the arms must then agree to the last digit.
    assert b["n_cells"] == a["n_cells"]
    assert out["per_cell"][0]["arms"]["d_fs_bins_2"]["n_distinct_blocks"] == 1
    assert (b["median_percentile_of_the_observed_spread"]
            == pytest.approx(a["median_percentile_of_the_observed_spread"]))


def test_AV_P3_verdict_needs_BOTH_halves_on_a_DISAGREEING_pair():
    """The conjunction, tested where it can actually be exercised.

    In every realistic panel the two halves agree, so a fixture built from the
    permutation cannot tell `and` from `or` -- the mutation runner said exactly
    that. `av_p3_verdict` is extracted so it can be handed a pair that
    DISAGREES, which is the only input on which the conjunction is visible
    (gotcha 16 / AU9b).
    """
    from src.analysis.design3_join import av_p3_verdict
    # spread inside the central mass, but too many cells above their own 95th
    a = av_p3_verdict(50.0, 26)
    assert a["spread_inside_the_central_mass"] is True
    assert a["cell_count_at_or_below_10"] is False
    assert a["AV_P3_both_halves"] is False and a["half_that_fails"] == "cell_count"
    # the mirror
    b = av_p3_verdict(99.0, 3)
    assert b["AV_P3_both_halves"] is False and b["half_that_fails"] == "spread"
    assert av_p3_verdict(99.0, 26)["half_that_fails"] == "both"
    c = av_p3_verdict(50.0, 3)
    assert c["AV_P3_both_halves"] is True and c["half_that_fails"] is None


def test_AV_P3_verdict_is_derived_and_needs_BOTH_halves():
    """gotcha 24/30: the verdict is total over the statistic, and both halves
    are required. Built by driving the arms to known aggregate values."""
    from src.analysis.design3_join import (
        stratified_permutation_null_for_the_panel as f)
    rng = np.random.default_rng(14)
    n = 4000
    fire = np.full((1, n), 5e-3)
    d = rng.normal(size=n)
    logM = rng.normal(size=(1, n))
    y = rng.binomial(N_TRIALS, 0.5, size=n).astype(float)[None, :]
    out = f(y, logM, fire, d[None, :], np.ones((1, n), bool), N_TRIALS,
            list(EDGES), n_perm=60, seed=5)
    p3 = out["AV_P3"]
    arm = out["arms"][out["primary_arm"]]
    # the outcome is a FUNCTION of the two halves, not an assertion
    assert p3["outcome"] == ("HIT" if (arm["spread_inside_the_central_mass"]
                                       and arm["cell_count_at_or_below_10"])
                             else "MISS")
    assert set(p3["verdict_at_every_grid_point"]) == {
        "d_fs_bins_2", "d_fs_bins_5", "d_fs_bins_10"}


def test_the_unstratified_arm_carries_AU14s_reference_for_comparison():
    rng = np.random.default_rng(15)
    n = 1500
    fire = np.full((1, n), 5e-3)
    out = _av_panel(rng.binomial(N_TRIALS, 0.5, size=n).astype(float)[None, :],
                    rng.normal(size=(1, n)), fire, rng.normal(size=(1, n)))
    ref = out["unstratified_reproduction_check"]["AU14_reference"]
    assert ref["median_percentile"] == 78.2 and ref["n_cells_above_95th"] == 26
    assert "recomputed" in out["unstratified_reproduction_check"]


def test_empirical_intervals_replace_the_wilson_construction():
    """AV12b: the panel's intervals come from BETWEEN-FEATURE dispersion.
    A decile with n = 1 has no such estimate and must report None rather than
    a number -- absent is not zero."""
    rng = np.random.default_rng(16)
    n = 1200
    fire = np.full((1, n), 5e-3)
    out = _av_panel(rng.binomial(N_TRIALS, 0.5, size=n).astype(float)[None, :],
                    rng.normal(size=(1, n)), fire, rng.normal(size=(1, n)))
    cell = out["per_cell"][0]
    assert len(cell["decile_means"]) == len(cell["decile_empirical_se"])
    ses = [s for s in cell["decile_empirical_se"] if s is not None]
    assert ses and all(s > 0 for s in ses)
    # sd(r_f)/sqrt(n) at n ~ 120 and sd ~ 0.08 is ~0.007; a Wilson half-width
    # on n*39 pooled Bernoullis would be several times smaller. The point of
    # the replacement is that these are DIFFERENT quantities.
    assert max(ses) < 0.1


def test_blocking_on_the_DECILE_makes_the_null_degenerate_at_the_observed():
    """The strongest available correctness property of the blocking, and it is
    a construction-level identity rather than a statistical tendency.

    Permuting `r_f` WITHIN a decile cannot change that decile's mean. So if the
    permutation blocks are the deciles themselves, every null draw must equal
    the observed spread EXACTLY. Any leakage across blocks -- an off-by-one in
    the index sets, a shared generator writing outside its block, a permutation
    applied to the wrong slice -- breaks the identity immediately, and none of
    those is visible in a distributional check.
    """
    from src.analysis.design3_join import _blocked_permutation_spread
    rng = np.random.default_rng(0)
    n = 600
    rf = rng.random(n)
    dec = np.repeat(np.arange(10), n // 10)
    null = _blocked_permutation_spread(rf, dec, dec.copy(), 10, 20,
                                       np.random.default_rng(3))
    means = [rf[dec == d].mean() for d in range(10)]
    observed = max(means) - min(means)
    assert np.allclose(null, observed)
    # and the control: with ONE block the null must NOT be degenerate, or the
    # test above would pass for a function that ignores its inputs entirely.
    loose = _blocked_permutation_spread(rf, dec, np.zeros(n, dtype=np.int64),
                                        10, 20, np.random.default_rng(3))
    assert loose.std() > 0


def test_blocking_preserves_each_blocks_multiset_of_r_f():
    """Permutation WITHIN a block must not move a value across blocks -- that
    is what makes the d_fs relationship survive untouched."""
    from src.analysis.design3_join import _blocked_permutation_spread
    rng = np.random.default_rng(0)
    n, seen = 600, []

    class Spy:
        def __init__(self, r):
            self.r = r

        def permutation(self, x):
            seen.append(np.sort(x))
            return self.r.permutation(x)

    rf = rng.random(n)
    dec = np.repeat(np.arange(10), n // 10)
    block = np.repeat(np.arange(6), n // 6)
    _blocked_permutation_spread(rf, dec, block, 10, 1, Spy(np.random.default_rng(1)))
    assert len(seen) == 6
    for b, got in zip(np.unique(block), seen):
        assert np.array_equal(got, np.sort(rf[block == b]))


# --------------------------------------------------------------------------
# firing_stratum_refit (amendment AW section 4 item 3)
# --------------------------------------------------------------------------

def _localised_effect_truth(rng, n_seeds=4, n_feat=1200, n_trials=39,
                            b1_global=0.05, b1_stratum=1.2):
    """A dataset where alignment predicts reproducibility ONLY inside one
    firing-rate band -- exactly AW's hypothesis under test. Half the
    features sit in the [1e-3, 3e-3) band with a STRONG b1; the rest are
    spread outside it with a near-zero b1. A global fit over everything
    dilutes the strong local effect toward the weak background one; a
    stratum-restricted fit should recover something close to `b1_stratum`.
    """
    from scipy.special import expit
    n_in_band = n_feat // 2
    fire = np.empty((n_seeds, n_feat))
    fire[:, :n_in_band] = 10 ** rng.uniform(np.log10(1e-3), np.log10(3e-3),
                                            (n_seeds, n_in_band))
    fire[:, n_in_band:] = 10 ** rng.uniform(-4.0, -1.5,
                                            (n_seeds, n_feat - n_in_band))
    logM = rng.normal(size=(n_seeds, n_feat))
    d_fs = rng.uniform(0.5, 5.0, (n_seeds, n_feat))
    log_phi = np.log10(fire)
    b1 = np.where(np.arange(n_feat) < n_in_band, b1_stratum, b1_global)
    eta = b1 * logM + 0.1 * np.sin(2.0 * (log_phi + 2.0)) + 0.05 * d_fs
    p = expit(eta)
    y = rng.binomial(n_trials, p).astype(np.float64)
    keep = np.ones((n_seeds, n_feat), dtype=bool)
    return y, logM, log_phi, d_fs, fire, keep


def test_firing_stratum_refit_recovers_a_localised_effect_the_global_fit_dilutes():
    from src.analysis.design3_join import fit_arm, firing_stratum_refit
    rng = np.random.default_rng(0)
    y, logM, log_phi, d_fs, fire, keep = _localised_effect_truth(rng)
    n_trials = 39

    primary = fit_arm(y, logM, log_phi, d_fs, keep, n_trials, dispersion=True)
    refit = firing_stratum_refit(y, logM, log_phi, d_fs, fire, keep, n_trials,
                                 1e-3, 3e-3, primary)

    assert refit["firing_range"] == [1e-3, 3e-3]
    assert min(refit["n_features_per_seed"]) >= 20
    global_fe = refit["global_b1_z_betabinomial_reference"]
    stratum_fe = refit["meta_b1_z_betabinomial"]["fixed_effect"]
    assert global_fe == primary["meta_b1_z_betabinomial"]["fixed_effect"]
    # THE POINT: the localised effect is bigger in z-scale too (both are
    # standardised within their own analysed set, but the stratum set is a
    # PURER population), and the ratio this function derives must reflect it.
    assert stratum_fe > global_fe > 0
    assert refit["abs_ratio_stratum_over_global"] == pytest.approx(
        abs(stratum_fe) / abs(global_fe))
    assert refit["exceeds_global_by_more_than_3x"] is (
        refit["abs_ratio_stratum_over_global"] > 3.0)


def test_firing_stratum_refit_reports_too_few_features_rather_than_fitting_noise():
    from src.analysis.design3_join import firing_stratum_refit
    rng = np.random.default_rng(1)
    y, logM, log_phi, d_fs, fire, keep = _localised_effect_truth(
        rng, n_seeds=3, n_feat=200)
    n_trials = 39
    # A band with almost nothing in it -- must refuse to fit, not silently
    # return a coefficient estimated on a handful of features.
    result = firing_stratum_refit(y, logM, log_phi, d_fs, fire, keep, n_trials,
                                  0.99, 1.0, primary={})
    assert result["status"] == "TOO_FEW_FEATURES"
    assert "meta_b1_z_betabinomial" not in result


# --------------------------------------------------------------------------
# firing_stratum_refit_curve / stratum_decile_shape (amendment CD item 2)
# --------------------------------------------------------------------------

def _curve_inputs(rng, thetas=(0.60, 0.70, 0.80)):
    """The localised-effect fixture, plus a per-theta count matrix and the
    matching GLOBAL theta-curve rows -- built the way `main()` builds them, so
    the test exercises the real join rather than a hand-written stand-in.

    Counts fall as theta rises (a stricter threshold matches fewer features),
    which is the direction the real artifact has.
    """
    from src.analysis.design3_join import fit_arm
    y, logM, log_phi, d_fs, fire, keep = _localised_effect_truth(rng)
    n_trials = 39
    counts, rows = {}, []
    for i, t in enumerate(thetas):
        yt = np.maximum(y - 3.0 * i, 0.0)
        counts[round(float(t), 2)] = yt
        arm = fit_arm(yt, logM, log_phi, d_fs, keep, n_trials, dispersion=True)
        rows.append({
            "theta": t,
            "meta_b1_z": arm["meta_b1_z"],
            "meta_b1_z_betabinomial": arm["meta_b1_z_betabinomial"],
            "marginal_effect_percentage_points":
                arm["marginal_effect_percentage_points"],
            "marginal_effect_percentage_points_betabinomial":
                arm["marginal_effect_percentage_points_betabinomial"],
            "mean_r_f": float(np.mean([r["mean_r_f"] for r in arm["per_seed"]])),
        })
    return counts, list(thetas), logM, log_phi, d_fs, fire, keep, n_trials, rows


def test_stratum_curve_READS_the_global_side_rather_than_refitting_it():
    """The global coefficient at each theta must be the one already computed
    in this run, byte for byte. A refit would land close but not equal, so
    equality is what discriminates -- `approx` here would pass a mutant."""
    from src.analysis.design3_join import firing_stratum_refit_curve
    rng = np.random.default_rng(3)
    counts, thetas, logM, log_phi, d_fs, fire, keep, n_trials, rows = \
        _curve_inputs(rng)

    out = firing_stratum_refit_curve(counts, thetas, logM, log_phi, d_fs,
                                     fire, keep, n_trials, 1e-3, 3e-3, rows)
    assert out["summary"]["n_thetas_fitted"] == len(thetas)
    for row, src in zip(out["rows"], rows):
        assert row["status"] == "OK"
        for lik in ("binomial", "betabinomial"):
            key = "meta_b1_z" if lik == "binomial" else "meta_b1_z_betabinomial"
            assert row[lik]["global"]["fixed_effect"] == \
                src[key]["fixed_effect"]
            assert row[lik]["global"]["fixed_effect_ci95"] == \
                src[key]["fixed_effect_ci95"]


def test_stratum_curve_joins_on_the_THETA_VALUE_not_on_position():
    """Gotcha 29. Shuffling the global rows must not change a single number:
    the join is on the theta value. Under a positional join the 0.60 stratum
    fit would be paired with the 0.80 global coefficient and every ratio would
    move, so this fixture discriminates."""
    from src.analysis.design3_join import firing_stratum_refit_curve
    rng = np.random.default_rng(4)
    counts, thetas, logM, log_phi, d_fs, fire, keep, n_trials, rows = \
        _curve_inputs(rng)

    ordered = firing_stratum_refit_curve(counts, thetas, logM, log_phi, d_fs,
                                         fire, keep, n_trials, 1e-3, 3e-3, rows)
    shuffled = firing_stratum_refit_curve(counts, thetas, logM, log_phi, d_fs,
                                          fire, keep, n_trials, 1e-3, 3e-3,
                                          list(reversed(rows)))
    a = ordered["summary"]["betabinomial"]["abs_ratio_by_theta"]
    b = shuffled["summary"]["betabinomial"]["abs_ratio_by_theta"]
    assert a == b
    # And the fixture must actually be able to tell them apart: the global
    # coefficients at the three thetas have to DIFFER, or reversing the list
    # would be a no-op and the test would pass against a positional join.
    globals_ = [r["meta_b1_z_betabinomial"]["fixed_effect"] for r in rows]
    assert len(set(globals_)) == len(globals_)


def test_stratum_curve_reports_an_ABSENT_global_row_as_absent():
    """An absent key is reported absent, never treated as agreement
    (gotcha 15) -- and never silently dropped from the grid."""
    from src.analysis.design3_join import firing_stratum_refit_curve
    rng = np.random.default_rng(5)
    counts, thetas, logM, log_phi, d_fs, fire, keep, n_trials, rows = \
        _curve_inputs(rng)

    out = firing_stratum_refit_curve(counts, thetas, logM, log_phi, d_fs, fire,
                                     keep, n_trials, 1e-3, 3e-3, rows[:-1])
    statuses = {r["theta"]: r["status"] for r in out["rows"]}
    assert statuses[thetas[-1]] == "GLOBAL_ROW_ABSENT"
    assert out["summary"]["n_thetas_fitted"] == len(thetas) - 1
    assert out["summary"]["n_thetas_requested"] == len(thetas)
    # A theta absent from the COUNTS artifact is a different fact from a theta
    # absent from the global curve, and the two must not collapse.
    out2 = firing_stratum_refit_curve({}, thetas, logM, log_phi, d_fs, fire,
                                      keep, n_trials, 1e-3, 3e-3, rows)
    assert {r["status"] for r in out2["rows"]} == {"ABSENT_FROM_ARTIFACT"}


def test_stratum_curve_disjointness_is_DERIVED_and_can_come_out_both_ways():
    """Gotcha 25: a check that can only fire one way is useless. Against the
    curve's own global side the intervals separate; against a global row whose
    interval is deliberately made to span the stratum's, they do not."""
    from src.analysis.design3_join import firing_stratum_refit_curve
    import copy
    rng = np.random.default_rng(6)
    counts, thetas, logM, log_phi, d_fs, fire, keep, n_trials, rows = \
        _curve_inputs(rng)

    out = firing_stratum_refit_curve(counts, thetas, logM, log_phi, d_fs, fire,
                                     keep, n_trials, 1e-3, 3e-3, rows)
    assert out["summary"]["betabinomial"]["disjoint_at_every_theta"] is True

    wide = copy.deepcopy(rows)
    for r in wide:
        r["meta_b1_z_betabinomial"]["fixed_effect_ci95"] = [-100.0, 100.0]
        r["meta_b1_z"]["fixed_effect_ci95"] = [-100.0, 100.0]
    out2 = firing_stratum_refit_curve(counts, thetas, logM, log_phi, d_fs, fire,
                                      keep, n_trials, 1e-3, 3e-3, wide)
    assert out2["summary"]["betabinomial"]["disjoint_at_no_theta"] is True
    assert out2["summary"]["betabinomial"]["disjoint_at_every_theta"] is False


def test_decile_shape_separates_an_INTERIOR_minimum_from_a_monotone_one():
    """The dip-then-rise question is answered by WHERE the minimum sits, so
    the instrument has to distinguish an interior argmin from an endpoint one.
    Both cases are planted and the verdicts must differ -- a fixture that only
    exhibits one of them could not detect a broken argmin."""
    from src.analysis.design3_join import stratum_decile_shape
    from scipy.special import expit
    rng = np.random.default_rng(7)
    n_seeds, n_feat, n_trials = 6, 3000, 39
    logM = rng.normal(size=(n_seeds, n_feat))
    keep = np.ones((n_seeds, n_feat), dtype=bool)

    # U-shaped in log M: the minimum is in the middle of the range.
    u = stratum_decile_shape(
        rng.binomial(n_trials, expit(-0.5 + 1.6 * logM ** 2)).astype(float),
        logM, keep, n_trials)
    # Monotone increasing: the minimum is the bottom decile.
    m = stratum_decile_shape(
        rng.binomial(n_trials, expit(-0.5 + 1.2 * logM)).astype(float),
        logM, keep, n_trials)

    assert u["status"] == "OK" and m["status"] == "OK"
    assert u["frac_seeds_argmin_interior"] == 1.0
    assert m["frac_seeds_argmin_interior"] == 0.0
    assert set(m["argmin_decile_seed_counts"]) == {"0"}
    assert set(m["argmax_decile_seed_counts"]) == {"9"}
    assert sum(m["argmin_decile_seed_counts"].values()) == n_seeds


def test_decile_shape_refuses_rather_than_fitting_a_handful_of_features():
    from src.analysis.design3_join import stratum_decile_shape
    rng = np.random.default_rng(8)
    logM = rng.normal(size=(3, 40))
    keep = np.zeros((3, 40), dtype=bool)
    keep[:, :5] = True
    out = stratum_decile_shape(rng.binomial(39, 0.5, (3, 40)).astype(float),
                               logM, keep, 39)
    assert out["status"] == "NO_USABLE_SEEDS"
    assert all(r["status"] == "TOO_FEW_FEATURES" for r in out["per_seed"])


# --------------------------------------------------------------------------
# post_hoc_interaction_fit (amendment CD section CD-e)
# --------------------------------------------------------------------------

def _interaction_truth(rng, b_inter, n_seeds=4, n_feat=1500, n_trials=39):
    """Data whose alignment slope varies linearly in log firing rate, by a
    controllable amount. `b_inter = 0` is the null case, and it must be
    reachable -- a check that can only report "excludes zero" is useless
    (gotcha 25)."""
    from scipy.special import expit
    fire = 10 ** rng.uniform(-4.0, -1.5, (n_seeds, n_feat))
    log_phi = np.log10(fire)
    logM = rng.normal(size=(n_seeds, n_feat))
    d_fs = rng.uniform(0.5, 5.0, (n_seeds, n_feat))
    zp = (log_phi - log_phi.mean()) / log_phi.std()
    eta = 0.30 * logM + b_inter * logM * zp + 0.05 * d_fs
    y = rng.binomial(n_trials, expit(eta)).astype(np.float64)
    keep = np.ones((n_seeds, n_feat), dtype=bool)
    return y, logM, log_phi, d_fs, keep, n_trials


def test_post_hoc_interaction_interval_EXCLUDES_zero_on_a_planted_interaction():
    from src.analysis.design3_join import fit_arm, post_hoc_interaction_fit
    rng = np.random.default_rng(11)
    y, logM, log_phi, d_fs, keep, n_trials = _interaction_truth(rng, 0.60)
    primary = fit_arm(y, logM, log_phi, d_fs, keep, n_trials, dispersion=True)
    out = post_hoc_interaction_fit(y, logM, log_phi, d_fs, keep, n_trials,
                                   primary)
    for lik in ("binomial", "betabinomial"):
        assert out[lik]["interval_covers_zero"] is False
        assert out[lik]["meta_interaction"]["fixed_effect"] > 0


def test_post_hoc_interaction_interval_COVERS_zero_when_none_was_planted():
    """The declared-in-advance branch. CD-e fixes that this outcome is
    reported exactly as the other one, so the instrument must reach it."""
    from src.analysis.design3_join import fit_arm, post_hoc_interaction_fit
    rng = np.random.default_rng(12)
    y, logM, log_phi, d_fs, keep, n_trials = _interaction_truth(rng, 0.0)
    primary = fit_arm(y, logM, log_phi, d_fs, keep, n_trials, dispersion=True)
    out = post_hoc_interaction_fit(y, logM, log_phi, d_fs, keep, n_trials,
                                   primary)
    for lik in ("binomial", "betabinomial"):
        assert out[lik]["interval_covers_zero"] is True


def test_post_hoc_interaction_CARRIES_the_primary_rather_than_refitting_it():
    """The primary must appear unchanged, byte for byte, and the object must
    hold no comparative fit statistic -- CD-e forbids offering one."""
    from src.analysis.design3_join import fit_arm, post_hoc_interaction_fit
    rng = np.random.default_rng(13)
    y, logM, log_phi, d_fs, keep, n_trials = _interaction_truth(rng, 0.40)
    primary = fit_arm(y, logM, log_phi, d_fs, keep, n_trials, dispersion=True)
    out = post_hoc_interaction_fit(y, logM, log_phi, d_fs, keep, n_trials,
                                   primary)

    assert out["label"] == "POST_HOC"
    assert out["binomial"]["primary_b1_z_unchanged"] == \
        primary["meta_b1_z"]["fixed_effect"]
    assert out["betabinomial"]["primary_b1_z_unchanged"] == \
        primary["meta_b1_z_betabinomial"]["fixed_effect"]
    assert out["binomial"]["primary_b1_z_ci95_unchanged"] == \
        primary["meta_b1_z"]["fixed_effect_ci95"]

    for word in ("NOT pre-registered", "no model selection", "files no "
                 "prediction", "Excluded from every tally"):
        assert word in out["standing"]
    # No comparative fit statistic is STORED. The scan is over KEYS, not over
    # the serialised blob: the prose above deliberately contains "prefer" and
    # "model selection" because it FORBIDS them, and a blob scan would fire on
    # the prohibition itself -- a check that cannot distinguish a rule from a
    # violation of it.
    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k.lower()
                yield from keys(v)
        elif isinstance(o, list):
            for v in o:
                yield from keys(v)

    found = set(keys(out))
    for banned in ("aic", "bic", "loglik_gain", "prefer", "preferred",
                   "better_model", "model_selection", "delta_deviance"):
        assert banned not in found
    # ... and the fixture must discriminate: a key scan that never sees any
    # key would pass this vacuously.
    assert "interval_covers_zero" in found


def test_fit_arm_extra_terms_are_inert_when_no_extra_is_passed():
    """The `extra_se` addition must not touch any existing arm: the primary
    and every sensitivity arm pass extra=None, so no stored number can move."""
    from src.analysis.design3_join import fit_arm
    rng = np.random.default_rng(14)
    y, logM, log_phi, d_fs, keep, n_trials = _interaction_truth(rng, 0.3)
    arm = fit_arm(y, logM, log_phi, d_fs, keep, n_trials, dispersion=True)
    for r in arm["per_seed"]:
        assert "extra_coefs" not in r
        assert "extra_se" not in r
        assert "extra_coefs" not in r["betabinomial"]
        assert "extra_se" not in r["betabinomial"]


# ---------------------------------------------------------------------------
# Amendment CQ: the AK8 smooth companion arm is ALIGNED (corrections entry 218)
# ---------------------------------------------------------------------------

def test_ak8_smooth_arm_is_aligned_to_the_anchor_order(joined):
    """AMENDMENT CQ (corrections entry 218). The fixture's smooth `M` is the
    hard-bin `M` times a CONSTANT, stored in S_PERM order. AK8's low-`M`
    selection is a within-bin QUANTILE, invariant under a positive constant,
    so with the smooth axis aligned every per-seed `n_low_M` and `n_discovery`
    in the smooth block must equal the hard-bin block's EXACTLY; with the rows
    left permuted, the smooth rows belong to other seeds and the counts
    cannot agree. Nothing else in the document sees this misalignment -- the
    sensitivity arm's own check (`test_the_smooth_arm_is_aligned_too`) reads a
    different call site, which is how the defect survived seven runs."""
    ak8 = joined["doc"]["ak8_discovery_set"]
    hb, sm = ak8["hard_bin"]["per_seed"], ak8["smooth"]["per_seed"]
    assert len(hb) == len(sm) == N_SEEDS
    for a, b in zip(hb, sm):
        assert a["n_low_M"] == b["n_low_M"], (a["seed"], a["n_low_M"], b["n_low_M"])
        assert a["n_discovery"] == b["n_discovery"], (
            a["seed"], a["n_discovery"], b["n_discovery"])
    assert "ALIGNED" in ak8["smooth"]["seed_axis"]


def test_identity_records_the_device_class(joined):
    """Amendment CQ, A5: the join is numpy on the host, so the recorded class
    is a `cpu:` string in `artifact_identity.device_class`'s vocabulary."""
    dc = joined["doc"]["identity"]["device_class"]
    assert isinstance(dc, str) and dc.startswith("cpu:"), dc


# ------------------------------------------------ amendment CR, Row C --
# The constant stays (imported above and by ak8_geometry / fitted_vs_observed)
# and is asserted equal to the config's floor before `main` opens anything.

def test_the_repository_config_floor_equals_the_constant():
    from src.analysis.design3_join import assert_floor_matches_config
    assert assert_floor_matches_config() == 200 == ANALYSIS_MIN_FIRING_ROWS


def test_a_config_floor_that_differs_is_refused_naming_both(tmp_path):
    from src.analysis.design3_join import assert_floor_matches_config
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("confound_control:\n  analysis_min_firing_rows: 201\n")
    with pytest.raises(SystemExit) as e:
        assert_floor_matches_config(cfg)
    assert "201" in str(e.value) and "200" in str(e.value)


def test_an_absent_floor_key_is_refused_not_defaulted(tmp_path):
    from src.analysis.design3_join import assert_floor_matches_config
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("confound_control:\n  caliper: 0.1\n")
    with pytest.raises(SystemExit) as e:
        assert_floor_matches_config(cfg)
    assert "analysis_min_firing_rows" in str(e.value)


def test_main_refuses_on_the_floor_before_opening_any_artifact(tmp_path, monkeypatch):
    """Every artifact path below is nonexistent; the floor check must fire
    first, so the message is about the floor and not about a missing file."""
    from src.analysis.design3_join import main
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("confound_control:\n  analysis_min_firing_rows: 199\n")
    nope = str(tmp_path / "nope.h5")
    monkeypatch.setattr(sys, "argv", [
        "join", "--rf-counts-all-theta", nope, "--rf-counts", nope,
        "--m1-json", nope, "--gtilde", nope, "--null-reductions", nope,
        "--d-fs", nope, "--out", str(tmp_path / "out.json"),
        "--config", str(cfg)])
    with pytest.raises(SystemExit) as e:
        main()
    assert "analysis_min_firing_rows" in str(e.value) and "199" in str(e.value)
