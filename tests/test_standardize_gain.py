"""K2's three-stage standardization.

Every stage is checked against an independently-computed value rather than
against itself, and the two properties that have historically gone wrong here
get dedicated fixtures:

  - the MAX must range over the argmax pool only (amendment I4 -- excluding a
    concept from the FDR family does not exclude it from a maximum, and that
    misreading cost a whole null re-run);
  - a null draw must not be standardized against moments containing itself.

The entry point is exercised too. A unit suite that never runs `main()`
cannot catch an entry-point bug, which this project has now hit four times
(gotcha 13 extension 4).
"""

from __future__ import annotations

import json
import sys

import h5py
import numpy as np
import pytest

from src.analysis.standardize_gain import (
    AxisSpec,
    concept_cell_index,
    masked_argmax,
    moments_from_sums,
    pooled_moments,
    standardize_seed,
)


# ------------------------------------------------------------ the axis spec --

def test_axis_spec_rejects_an_unknown_axis():
    with pytest.raises(ValueError, match="unknown concept axes"):
        AxisSpec(("prevalence", "run_length"))


def test_firing_rate_axis_cannot_be_dropped():
    """K1's remedy list forbids collapsing the firing axis to fix a count;
    dropping it is the limiting case. Firing rate is the confound the whole
    design controls for, so removing it from the pooling scheme would put the
    confound back inside mu_s and sigma_s."""
    with pytest.raises(ValueError, match="cannot be dropped"):
        AxisSpec(("prevalence",), firing_rate=False)


def test_axis_spec_is_a_parameter_and_changes_the_cell_count():
    vals = {"prevalence": np.array([1e-6, 1e-3]),
            "protein_count": np.array([6.0, 600.0]),
            "cluster_count": np.array([1.0, 50.0])}
    _, n2 = concept_cell_index(vals, AxisSpec(("prevalence", "protein_count")))
    _, n4 = concept_cell_index(vals, AxisSpec(("prevalence", "protein_count",
                                               "cluster_count")))
    assert n2 == 16 and n4 == 64          # O2's fourth axis quadruples cells


def test_out_of_scope_concepts_are_marked_not_clipped_into_bin_zero():
    """A protein count below the >=5 gate is not a member of the lowest
    stratum -- it is a concept the scheme cannot place. Silently binning it
    as 0 is how a sentinel becomes data."""
    vals = {"prevalence": np.array([1e-6, 1e-6]),
            "protein_count": np.array([6.0, 2.0])}     # 2 is below the gate
    cell, _ = concept_cell_index(vals, AxisSpec(("prevalence", "protein_count")))
    assert cell[0] >= 0 and cell[1] == -1


# ------------------------------------------------------ stage 1: the moments --

def test_pooled_moments_match_a_direct_computation():
    rng = np.random.default_rng(0)
    slab = rng.normal(size=(4, 6, 5))                  # shifts, concepts, feats
    cell = np.array([0, 0, 1, 1, 1, 0])
    fbin = np.array([0, 0, 1, 1, 1])
    s1, s2, cnt = pooled_moments(slab, cell, 2, fbin, 2)
    mu, sig = moments_from_sums(s1, s2, cnt)
    sub = slab[:, cell == 1][:, :, fbin == 1]
    assert cnt[1, 1] == 3 * 3
    assert np.isclose(mu[1, 1], sub.mean())
    assert np.isclose(sig[1, 1], sub.std())


def test_leave_one_out_equals_recomputing_without_that_shift():
    """LOO is implemented as a subtraction on per-shift sums. That is only
    valid if it reproduces the honest recomputation, so the identity is
    asserted rather than argued."""
    rng = np.random.default_rng(1)
    slab = rng.normal(size=(5, 4, 3))
    cell, fbin = np.zeros(4, int), np.zeros(3, int)
    s1, s2, cnt = pooled_moments(slab, cell, 1, fbin, 1)
    for drop in range(5):
        mu, sig = moments_from_sums(s1, s2, cnt, drop_shift=drop)
        kept = np.delete(slab, drop, axis=0)
        assert np.isclose(mu[0, 0], kept.mean())
        assert np.isclose(sig[0, 0], kept.std())


def test_sigma_is_nan_when_it_cannot_be_estimated_not_a_small_number():
    """Dividing by a fabricated sigma is how a stratum with no information
    produces a confident-looking extreme value."""
    slab = np.ones((3, 2, 2))                          # zero spread
    cell, fbin = np.zeros(2, int), np.zeros(2, int)
    s1, s2, cnt = pooled_moments(slab, cell, 1, fbin, 1)
    _, sig = moments_from_sums(s1, s2, cnt)
    assert np.isnan(sig[0, 0])


# --------------------------------------------------------- stage 2: the max --

def test_pooled_parents_cannot_win_the_max(tmp_path):
    """AMENDMENT I4, the defect that cost a 2h10m null re-run: a pooled
    parent left in the pool can still win a MAXIMUM even though it is
    excluded from the FDR family. Here concept 0 is a parent with by far the
    largest z and must not be the argmax."""
    z = np.array([[9.0, 9.0], [1.0, 5.0], [2.0, 3.0]])   # (concepts, feats)
    pool = np.array([False, True, True])
    m, am = masked_argmax(z, pool)
    assert list(am) == [2, 1]
    assert np.allclose(m, [2.0, 5.0])


def test_ties_are_broken_by_lowest_index():
    """Asserted, not assumed: two code paths breaking ties differently would
    attribute the same `g~` to different concepts, and D5's breadth
    accounting would silently disagree with the classification's."""
    z = np.array([[1.0], [7.0], [7.0]])
    _, am = masked_argmax(z, np.array([True, True, True]))
    assert am[0] == 1


# ------------------------------------------- stages composed, on one seed --

def _toy(n_shifts=6, n_pool=4, n_feats=3, seed=0):
    rng = np.random.default_rng(seed)
    null = rng.normal(loc=2.0, scale=0.5, size=(n_shifts, n_pool, n_feats))
    real = rng.normal(loc=2.0, scale=0.5, size=(n_pool + 1, n_feats))
    pool_mask = np.array([False] + [True] * n_pool)     # concept 0 is a parent
    cell = np.zeros(n_pool + 1, dtype=int)
    fbin = np.zeros(n_feats, dtype=int)
    return real, null, cell, fbin, pool_mask


def test_stage1_z_matches_the_definition():
    real, null, cell, fbin, pool = _toy()
    res = standardize_seed(real, null, cell, 1, fbin, 1, pool, loo_inner=True)
    mu, sd = null.mean(), null.std()
    assert np.allclose(res["z_real"], (real - mu) / sd)


def test_stage3_null_draws_are_not_standardized_against_themselves():
    """A draw compared to a distribution containing itself is pulled toward
    zero by 1/n of its own value plus its own contribution to sigma. With 20
    shifts that is not a rounding error, which is why K2 requires LOO."""
    real, null, cell, fbin, pool = _toy(n_shifts=6)
    res = standardize_seed(real, null, cell, 1, fbin, 1, pool, loo_inner=True)
    Mn, gn = res["M_null"], res["gtilde_null"]
    for r in range(Mn.shape[0]):
        others = np.delete(Mn, r, axis=0)
        expect = (Mn[r] - others.mean(0)) / others.std(0, ddof=1)
        assert np.allclose(gn[r], expect, equal_nan=True)
    # and it genuinely differs from the self-inclusive version
    naive = (Mn - Mn.mean(0)) / Mn.std(0, ddof=1)
    assert not np.allclose(gn, naive)


def test_real_gain_is_NOT_leave_one_out():
    """LOO applies to null draws only. The real `g` is not in the null, so it
    cannot contaminate its own moments -- applying LOO to it would compare it
    against a deliberately different distribution for no reason."""
    real, null, cell, fbin, pool = _toy()
    res = standardize_seed(real, null, cell, 1, fbin, 1, pool, loo_inner=True)
    assert np.allclose(res["z_real"], (real - null.mean()) / null.std())


def test_inner_loo_actually_changes_the_null_draws():
    """Not vacuous: the `--no-loo-inner` switch must do something, or the
    comparison it exists to support is meaningless."""
    real, null, cell, fbin, pool = _toy()
    a = standardize_seed(real, null, cell, 1, fbin, 1, pool, loo_inner=True)
    b = standardize_seed(real, null, cell, 1, fbin, 1, pool, loo_inner=False)
    assert not np.allclose(a["M_null"], b["M_null"])
    assert np.allclose(a["z_real"], b["z_real"])       # real path unaffected


def test_stratification_rescales_a_high_variance_stratum():
    """The mechanism amendment O5 turns on, in miniature. Two concepts with
    identical raw gain, one in a low-variance stratum and one in a
    high-variance stratum: standardization must rank the low-variance one
    higher. If it did not, no amount of stratification could calibrate
    `BINDING:heme a3` away from the top of `g~`."""
    rng = np.random.default_rng(3)
    n_shifts, n_feats = 40, 2
    null = np.empty((n_shifts, 2, n_feats))
    null[:, 0] = rng.normal(0.0, 0.05, (n_shifts, n_feats))   # tight stratum
    null[:, 1] = rng.normal(0.0, 0.90, (n_shifts, n_feats))   # loose stratum
    real = np.full((2, n_feats), 0.9)                         # identical raw g
    cell = np.array([0, 1])
    res = standardize_seed(real, null, cell, 2, np.zeros(n_feats, int), 1,
                           np.array([True, True]), loo_inner=True)
    assert (res["z_real"][0] > res["z_real"][1]).all()
    assert (res["argmax_concept"] == 0).all()


# ------------------------------------------------------------ entry point --

def _write_inputs(tmp_path, n_seeds=2, n_shifts=4, n_feats=6, fire_rate=None):
    concepts = ["ACT_SITE", "BINDING", "BINDING:ATP", "DOMAIN", "DOMAIN:SH2"]
    pool = ["ACT_SITE", "BINDING:ATP", "DOMAIN:SH2"]
    rng = np.random.default_rng(7)
    n_rows = 1_000_000

    ap = tmp_path / "ap.h5"
    with h5py.File(ap, "w") as hf:
        g = hf.create_group("pooled")
        g.create_dataset("concepts",
                         data=np.array(concepts, dtype=h5py.string_dtype()))
        g.create_dataset("ap", data=rng.uniform(
            0, 0.3, (n_seeds, n_feats, len(concepts))).astype(np.float32))
        g.create_dataset("prevalence",
                         data=np.array([1e-6, 1e-3, 1e-6, 1e-3, 1e-4]))
        # AMENDMENT CQ: the AP matrix's own seed labels, in the reference
        # order (feature_ap writes checkpoint names).
        hf.create_dataset("seeds", data=[f"seed_{i}".encode()
                                         for i in range(n_seeds)],
                          dtype=h5py.string_dtype())
        for k, v in {"n_rows": n_rows}.items():
            hf.attrs[k] = v

    raw = tmp_path / "raw.h5"
    gain_b = rng.normal(0, 0.2, (n_seeds, n_shifts, len(pool), n_feats)
                        ).astype(np.float32)
    with h5py.File(raw, "w") as hf:
        hf.create_dataset("concepts",
                          data=np.array(pool, dtype=h5py.string_dtype()))
        hf.create_dataset("gain_b", data=gain_b)

    red = tmp_path / "red.h5"
    with h5py.File(red, "w") as hf:
        # The default collapses to ONE firing bin, which makes per-seed and
        # cross-seed-mean binning indistinguishable -- a fixture that cannot
        # exercise the axis under test (gotcha 16). Callers testing that axis
        # pass an explicit `fire_rate`.
        hf.create_dataset("fire_rate", data=(
            rng.uniform(1e-4, 1e-2, (n_seeds, n_feats))
            if fire_rate is None else np.asarray(fire_rate, dtype=np.float64)))
        # AMENDMENT AS: the null's seed labels are what `standardize_gain` now
        # copies onto its own output, so an unlabelled null must make the job
        # refuse rather than write an unlabelled seed axis.
        hf.create_dataset("seeds", data=[f"seed_{i}".encode()
                                         for i in range(n_seeds)],
                          dtype=h5py.string_dtype())
        # AMENDMENT CQ: the reductions' max over concepts of the raw null --
        # the recovery target for the raw null's unlabelled seed axis (exact
        # equality, as on the real artifacts).
        hf.create_dataset("null_g_b",
                          data=gain_b.astype(np.float64).max(axis=2))

    lab = tmp_path / "lab.h5"
    with h5py.File(lab, "w") as hf:
        grp = hf.create_group("concepts")
        for c in concepts:
            grp.create_group(c).create_dataset("all", data=np.arange(3))
        for k, v in {"git_sha": "abc", "population": "untruncated",
                     "n_rows": n_rows, "n_concepts": len(concepts),
                     "retention_floor_mode": "interplm",
                     "retention_min_residues": 1500,
                     "retention_min_instances": 10,
                     "min_protein_count": 5}.items():
            hf.attrs[k] = v
    diag = {c: {"n_residues": int(p * n_rows), "n_proteins": pc,
                "n_clusters": cl}
            for c, p, pc, cl in zip(concepts, [1e-6, 1e-3, 1e-6, 1e-3, 1e-4],
                                    [6, 600, 7, 700, 60], [1, 50, 2, 60, 5])}
    lab.with_suffix(".json").write_text(
        json.dumps({"retention_floor_diagnostics": diag}))
    return ap, raw, red, lab


def _run(tmp_path, monkeypatch, out="g.h5", extra=()):
    from src.analysis.standardize_gain import main
    ap, raw, red, lab = _write_inputs(tmp_path)
    o = tmp_path / out
    monkeypatch.setattr(sys, "argv", [
        "standardize_gain", "--ap", str(ap), "--null-raw", str(raw),
        "--null-reductions", str(red), "--labels", str(lab),
        "--out", str(o), "--skip-preflight", *extra])
    main()
    return o


def test_main_runs_end_to_end_and_writes_every_stage(tmp_path, monkeypatch):
    o = _run(tmp_path, monkeypatch)
    with h5py.File(o, "r") as hf:
        assert hf["z"].shape == (2, 6, 5)          # FULL library, 712-analogue
        assert hf["gtilde"].shape == (2, 6)
        assert hf["M_null"].shape == (2, 4, 6)
        assert int(hf.attrs["argmax_pool_n"]) == 3
        assert hf.attrs["library_sha"] == "abc"    # from the TARGET store
        assert bool(hf.attrs["loo_outer"]) is True
        # the parents are kept in the matrix but never win
        pool = hf["in_argmax_pool"][:]
        assert list(pool) == [True, False, True, False, True]
        assert set(np.unique(hf["argmax_concept"][:])) <= {0, 2, 4}


def test_main_refuses_to_overwrite(tmp_path, monkeypatch):
    o = _run(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        _run(tmp_path, monkeypatch, out=o.name)


def test_main_refuses_a_null_drawn_on_a_different_pool(tmp_path, monkeypatch):
    """The I4 defect at the entry point. A length check cannot see this --
    the pools must be the same OBJECT, not merely the same size."""
    from src.analysis.standardize_gain import main
    ap, raw, red, lab = _write_inputs(tmp_path)
    with h5py.File(raw, "a") as hf:
        del hf["concepts"]
        hf.create_dataset("concepts", data=np.array(
            ["ACT_SITE", "BINDING:ATP", "WRONG"], dtype=h5py.string_dtype()))
    monkeypatch.setattr(sys, "argv", [
        "x", "--ap", str(ap), "--null-raw", str(raw), "--null-reductions",
        str(red), "--labels", str(lab), "--out", str(tmp_path / "z.h5"),
        "--skip-preflight"])
    with pytest.raises(SystemExit, match="amendment-I4 defect"):
        main()


def test_main_refuses_a_store_without_cluster_counts(tmp_path, monkeypatch):
    """`-1` means the store was built without --clusters. Absent is not zero,
    and the O2 cluster axis cannot be formed from a sentinel."""
    from src.analysis.standardize_gain import main
    ap, raw, red, lab = _write_inputs(tmp_path)
    j = lab.with_suffix(".json")
    d = json.loads(j.read_text())
    for v in d["retention_floor_diagnostics"].values():
        v.pop("n_clusters")
    j.write_text(json.dumps(d))
    monkeypatch.setattr(sys, "argv", [
        "x", "--ap", str(ap), "--null-raw", str(raw), "--null-reductions",
        str(red), "--labels", str(lab), "--out", str(tmp_path / "z2.h5"),
        "--skip-preflight"])
    with pytest.raises(SystemExit, match="no `n_clusters`"):
        main()


# --------------------------------- ratio normalization (amendment U1) ------

def test_ratio_mode_divides_by_the_null_MEAN_not_its_sd():
    """GSEA's normalization: `g / mu`, not `(g - mu) / sigma`.

    Asserted directly because the three inner modes differ by one operator
    each and a transposition would produce plausible numbers everywhere.
    """
    from src.analysis.standardize_gain import _inner
    g = np.array([[2.0, 4.0]])
    mu = np.array([[2.0, 2.0]])
    sig = np.array([[0.5, 0.5]])
    assert np.allclose(_inner(g, mu, sig, "ratio"), [[1.0, 2.0]])
    assert np.allclose(_inner(g, mu, sig, "centre"), [[0.0, 2.0]])
    assert np.allclose(_inner(g, mu, sig, "full"), [[0.0, 4.0]])
    assert np.allclose(_inner(g, mu, sig, "none"), g)


def test_ratio_mode_reorders_where_centring_cannot(tmp_path, monkeypatch):
    """M1 showed centring is inert for the maximum; the ratio must not be.

    CONSTRUCTED IN THE REAL REGIME, and the first version of this test was
    not. Centring's inertness is not algebraic -- it is a consequence of
    `mu` (~5e-06) being negligible against the winning `g` (~1.0). A toy
    with `mu` comparable to `g` makes centring reorder, which is true of the
    toy and false of the data. Here two strata differ 10x in `mu`, both are
    tiny, and one concept per stratum carries a large real gain: subtracting
    ~1e-06 cannot change which wins, dividing by it can.
    """
    n_shifts, n_feats = 8, 3
    rng = np.random.default_rng(5)
    # concept 0 is a pooled parent; concepts 1,2 in stratum 0; 3,4 in stratum 1
    cell = np.array([0, 0, 0, 1, 1])
    pool = np.array([False, True, True, True, True])
    null = np.abs(rng.normal(0, 1e-6, (n_shifts, 4, n_feats)))
    null[:, 2:] *= 10.0                       # stratum 1 has ~10x the mean
    real = np.zeros((5, n_feats))
    real[1] = 0.30                            # low-mu stratum
    real[3] = 0.35                            # high-mu stratum, larger RAW
    kw = dict(cell=cell, n_cells=2, fbin=np.zeros(n_feats, int), n_fbins=1,
              pool_mask=pool, loo_inner=True)
    r_none = standardize_seed(real, null, **kw, inner_mode="none")
    r_centre = standardize_seed(real, null, **kw, inner_mode="centre")
    r_ratio = standardize_seed(real, null, **kw, inner_mode="ratio")

    # raw and centred both pick concept 3 -- it has the larger raw gain and
    # mu is negligible against it
    assert (r_none["argmax_concept"] == 3).all()
    assert (r_centre["argmax_concept"] == r_none["argmax_concept"]).all()
    # the ratio divides by a 10x larger mu in stratum 1, so concept 1 wins
    assert (r_ratio["argmax_concept"] == 1).all()


def test_ratio_mode_refuses_a_nonpositive_stratum_mean(tmp_path, monkeypatch):
    """Dividing by a near-zero denominator is the failure mode this line of
    work exists to avoid, so a non-positive `mu_s` is a refusal rather than a
    silent floor -- and the check runs per seed, not once."""
    from src.analysis.standardize_gain import main
    ap, raw, red, lab = _write_inputs(tmp_path)
    with h5py.File(raw, "a") as hf:            # force a NEGATIVE-mean stratum
        d = hf["gain_b"][:]
        # Negative rather than zero (amendment CQ): all-zero slabs are
        # identical across seeds, so the raw null's seed order becomes
        # unrecoverable and THAT refusal fires first. Distinct negative slabs
        # keep the order recoverable and still make every stratum mean <= 0.
        d[:] = -np.abs(d) - np.float32(1e-3)
        del hf["gain_b"]
        hf.create_dataset("gain_b", data=d)
    with h5py.File(red, "a") as hf:            # keep the reductions CONSISTENT
        del hf["null_g_b"]                     # with the raw (CQ A2-iii)
        hf.create_dataset("null_g_b", data=d.astype(np.float64).max(axis=2))
    monkeypatch.setattr(sys, "argv", [
        "x", "--ap", str(ap), "--null-raw", str(raw), "--null-reductions",
        str(red), "--labels", str(lab), "--out", str(tmp_path / "r.h5"),
        "--inner-mode", "ratio", "--skip-preflight"])
    with pytest.raises(SystemExit, match="mu_s <= 0"):
        main()


# ------------------------------------------------- the persisted firing bin --
# `stratum_mu` is indexed by (concept cell, firing bin). `concept_cell` was on
# disk and the feature-side index was not, so `m4_confound` reported C1 for the
# HARD-BIN mu as not computable. Hard-bin is the predictor of record (AC9), so
# the confound question has to be closable for it.

def test_feature_firing_bin_is_persisted_with_the_right_shape_and_dtype(
        tmp_path, monkeypatch):
    o = _run(tmp_path, monkeypatch)
    with h5py.File(o, "r") as hf:
        fb = hf["feature_firing_bin"][:]
        n_seeds, n_feats = hf["gtilde"].shape
        assert fb.shape == (n_seeds, n_feats)
        assert fb.dtype == np.int8
        n_bins = len(json.loads(hf.attrs["firing_rate_edges"])) - 1
        assert fb.min() >= 0 and fb.max() <= n_bins - 1


def test_persisted_firing_bin_equals_the_reconstruction_from_fire_rate(
        tmp_path, monkeypatch):
    """And this settles a claim, not just the plumbing.

    `m4_confound`'s note says the per-feature normaliser "cannot be
    reconstructed from what is on disk". It could: the assignment is a pure
    function of `fire_rate` (in the null-reductions artifact) and the
    `firing_rate_edges` attr (already written by `write_identity`). This test
    computes it both ways and requires equality, so the reconstruction route is
    exact rather than approximate -- which is what "reported as absent rather
    than approximated" was protecting against.
    """
    from src.analysis.null_pooling import bin_index
    ap, raw, red, lab = _write_inputs(tmp_path)
    o = tmp_path / "g_recon.h5"
    monkeypatch.setattr(sys, "argv", [
        "standardize_gain", "--ap", str(ap), "--null-raw", str(raw),
        "--null-reductions", str(red), "--labels", str(lab),
        "--out", str(o), "--skip-preflight"])
    from src.analysis.standardize_gain import main
    main()

    with h5py.File(o, "r") as hf:
        stored = hf["feature_firing_bin"][:]
        # json.loads, NOT list(): write_identity json.dumps a list, so this
        # attr round-trips as a STRING and `list()` of it yields CHARACTERS
        # (gotcha 20). This assertion only holds under the correct read.
        edges = json.loads(hf.attrs["firing_rate_edges"])
    with h5py.File(red, "r") as hf:
        fire_rate = hf["fire_rate"][:]

    n_bins = len(edges) - 1
    recon = np.stack([np.clip(bin_index(fire_rate[s], edges), 0, n_bins - 1)
                      for s in range(fire_rate.shape[0])])
    assert np.array_equal(stored, recon.astype(np.int8))


def _discriminating_fire_rate():
    """A `fire_rate` on which per-seed and cross-seed-mean binning DISAGREE.

    The default fixture collapses to a single firing bin, so every feature lands
    in bin 0 either way and the distinction is untestable. Mutation testing
    caught that: swapping the per-seed bin for `fire_rate.mean(axis=0)` -- the
    gotcha-18 defect itself -- left the suite green.

    25 features per bin clears MIN_FEATURES_PER_STRATUM = 20 so two bins survive
    the collapse; features 0-2 have seeds in different bins, so their cross-seed
    mean lands in a bin NEITHER seed occupies.
    """
    fr = np.empty((2, 50))
    fr[0, :25] = 5e-4
    fr[1, :25] = 5e-4
    fr[0, 25:] = 5e-3
    fr[1, 25:] = 5e-3
    for j in (0, 1, 2):
        fr[0, j] = 2e-4
        fr[1, j] = 8e-3
    return fr


def test_persisted_firing_bin_is_PER_SEED_not_the_cross_seed_mean(
        tmp_path, monkeypatch):
    """Gotcha 18 on the artifact this session added.

    Firing rate by feature index has cross-seed correlation 0.0009, so a mean
    over the seed axis averages 20 unrelated variables. The persisted assignment
    must be the per-seed one; this test fails if it is the mean.
    """
    from src.analysis.null_pooling import bin_index
    from src.analysis.standardize_gain import main

    fr = _discriminating_fire_rate()
    ap, raw, red, lab = _write_inputs(
        tmp_path, n_seeds=fr.shape[0], n_feats=fr.shape[1], fire_rate=fr)
    o = tmp_path / "g_perseed.h5"
    monkeypatch.setattr(sys, "argv", [
        "standardize_gain", "--ap", str(ap), "--null-raw", str(raw),
        "--null-reductions", str(red), "--labels", str(lab),
        "--out", str(o), "--skip-preflight"])
    main()

    with h5py.File(o, "r") as hf:
        stored = hf["feature_firing_bin"][:]
        edges = json.loads(hf.attrs["firing_rate_edges"])

    n_bins = len(edges) - 1
    per_seed = np.stack([np.clip(bin_index(fr[s], edges), 0, n_bins - 1)
                         for s in range(fr.shape[0])])
    cross_mean = np.clip(bin_index(fr.mean(axis=0), edges), 0, n_bins - 1)

    # the fixture must actually discriminate, or this test proves nothing
    assert len(set(per_seed.ravel())) >= 2, "fixture uses only one bin"
    assert any(not np.array_equal(per_seed[s], cross_mean)
               for s in range(fr.shape[0])), "fixture cannot tell the two apart"

    assert np.array_equal(stored, per_seed.astype(np.int8))


# ---------------------------------------------------------------------------
# Amendment CQ, A2: the three (four) positionally-indexed inputs are CHECKED
# against the null reductions' seed order before any indexing.
# ---------------------------------------------------------------------------

def _argv(ap, raw, red, lab, out, *extra):
    return ["standardize_gain", "--ap", str(ap), "--null-raw", str(raw),
            "--null-reductions", str(red), "--labels", str(lab),
            "--out", str(out), "--skip-preflight", *extra]


def test_refuses_an_ap_matrix_whose_seed_order_differs(tmp_path, monkeypatch):
    """The AP matrix stored with its seeds REVERSED relative to the null
    reductions. Indexing them positionally would pair seed 1's AP with seed
    0's null and firing rate (gotcha 29); the refusal names BOTH orders."""
    from src.analysis.standardize_gain import main
    ap, raw, red, lab = _write_inputs(tmp_path)
    with h5py.File(ap, "a") as hf:
        del hf["seeds"]
        hf.create_dataset("seeds", data=[b"seed_1", b"seed_0"],
                          dtype=h5py.string_dtype())
    monkeypatch.setattr(sys, "argv", _argv(ap, raw, red, lab, tmp_path / "o.h5"))
    with pytest.raises(SystemExit, match="seed order") as ei:
        main()
    msg = str(ei.value)
    assert "['0', '1']" in msg and "['1', '0']" in msg, msg
    assert not (tmp_path / "o.h5").exists()


def test_refuses_an_unlabelled_ap_matrix(tmp_path, monkeypatch):
    """Absent labels are not agreement (gotcha 15)."""
    from src.analysis.standardize_gain import main
    ap, raw, red, lab = _write_inputs(tmp_path)
    with h5py.File(ap, "a") as hf:
        del hf["seeds"]
    monkeypatch.setattr(sys, "argv", _argv(ap, raw, red, lab, tmp_path / "o.h5"))
    with pytest.raises(SystemExit, match="no seed labels"):
        main()


def test_refuses_a_raw_null_whose_slabs_are_permuted(tmp_path, monkeypatch):
    """The raw null carries NO labels, so its order is recovered from the data:
    slab s's max over concepts must equal the reductions' `null_g_b[s]`
    exactly. Swap two slabs and leave the reductions alone -- slab 0 now IS
    row 1 -- and the job must refuse rather than pair slab 0 with seed 0."""
    from src.analysis.standardize_gain import main
    ap, raw, red, lab = _write_inputs(tmp_path)
    with h5py.File(raw, "a") as hf:
        d = hf["gain_b"][:]
        d[[0, 1]] = d[[1, 0]]
        del hf["gain_b"]
        hf.create_dataset("gain_b", data=d)
    monkeypatch.setattr(sys, "argv", _argv(ap, raw, red, lab, tmp_path / "o.h5"))
    with pytest.raises(SystemExit, match="raw null slab 0") as ei:
        main()
    assert "row 1" in str(ei.value), str(ei.value)


def test_refuses_a_raw_null_the_reductions_do_not_describe(tmp_path, monkeypatch):
    """Zero exact matches: the reductions were derived from some OTHER raw
    array. Refuse; do not take the nearest."""
    from src.analysis.standardize_gain import main
    ap, raw, red, lab = _write_inputs(tmp_path)
    with h5py.File(red, "a") as hf:
        g = hf["null_g_b"][:] + 1e-3
        del hf["null_g_b"]
        hf.create_dataset("null_g_b", data=g)
    monkeypatch.setattr(sys, "argv", _argv(ap, raw, red, lab, tmp_path / "o.h5"))
    with pytest.raises(SystemExit, match="matches no row"):
        main()


def _mu_smooth(tmp_path, n_seeds=2, n_concepts=5, n_feats=6, seeds="in_order"):
    """A positive `mu_hat_full` on the FULL concept axis, pool 3 (the fixture's
    argmax pool), with its seed axis labelled in the reference order, in a
    different order, or not at all."""
    p = tmp_path / "mu_smooth.h5"
    rng = np.random.default_rng(3)
    with h5py.File(p, "w") as hf:
        hf.create_dataset("mu_hat_full", data=rng.uniform(
            0.5, 1.5, (n_seeds, n_concepts, n_feats)).astype(np.float32))
        hf.attrs["argmax_pool_n"] = 3
        if seeds == "in_order":
            hf.create_dataset("seeds", data=[f"seed_{i}".encode()
                                             for i in range(n_seeds)],
                              dtype=h5py.string_dtype())
        elif seeds == "reversed":
            hf.create_dataset("seeds", data=[f"seed_{i}".encode()
                                             for i in reversed(range(n_seeds))],
                              dtype=h5py.string_dtype())
    return p


def _make_gain_positive(raw, red):
    """Ratio mode divides by the hard-stratum mean, which must be > 0."""
    with h5py.File(raw, "a") as hf:
        d = np.abs(hf["gain_b"][:]) + np.float32(0.05)
        del hf["gain_b"]
        hf.create_dataset("gain_b", data=d)
    with h5py.File(red, "a") as hf:
        del hf["null_g_b"]
        hf.create_dataset("null_g_b", data=d.astype(np.float64).max(axis=2))


def test_refuses_an_unlabelled_mu_smooth(tmp_path, monkeypatch):
    """The FOURTH positionally-indexed input (found by reading the module):
    `mu_hat_full[s]`. Unlabelled -> refuse, the AS convention."""
    from src.analysis.standardize_gain import main
    ap, raw, red, lab = _write_inputs(tmp_path)
    _make_gain_positive(raw, red)
    ms = _mu_smooth(tmp_path, seeds="none")
    monkeypatch.setattr(sys, "argv", _argv(
        ap, raw, red, lab, tmp_path / "o.h5",
        "--inner-mode", "ratio", "--mu-smooth", str(ms)))
    with pytest.raises(SystemExit, match="mu_smooth.h5:seeds carries no seed"):
        main()


def test_refuses_a_mu_smooth_in_a_different_seed_order(tmp_path, monkeypatch):
    from src.analysis.standardize_gain import main
    ap, raw, red, lab = _write_inputs(tmp_path)
    _make_gain_positive(raw, red)
    ms = _mu_smooth(tmp_path, seeds="reversed")
    monkeypatch.setattr(sys, "argv", _argv(
        ap, raw, red, lab, tmp_path / "o.h5",
        "--inner-mode", "ratio", "--mu-smooth", str(ms)))
    with pytest.raises(SystemExit, match="seed order of .*mu_smooth"):
        main()


def test_a_mu_smooth_in_the_reference_order_is_accepted(tmp_path, monkeypatch):
    """The positive control for the two refusals above (gotcha 28)."""
    from src.analysis.standardize_gain import main
    ap, raw, red, lab = _write_inputs(tmp_path)
    _make_gain_positive(raw, red)
    ms = _mu_smooth(tmp_path, seeds="in_order")
    o = tmp_path / "o.h5"
    monkeypatch.setattr(sys, "argv", _argv(
        ap, raw, red, lab, o, "--inner-mode", "ratio", "--mu-smooth", str(ms)))
    main()
    with h5py.File(o, "r") as hf:
        assert hf.attrs["inner_mode"] == "ratio"
        assert hf["M"].shape == (2, 6)


def test_the_consistent_fixture_still_runs_end_to_end(tmp_path, monkeypatch):
    """Positive control for A2 as a whole: the default fixture, whose AP seeds
    match the reductions and whose `null_g_b` IS the raw null's max over
    concepts, passes every new check (gotcha 28: a refusal is evidence only
    once the instrument is shown to pass when it should)."""
    o = _run(tmp_path, monkeypatch)
    with h5py.File(o, "r") as hf:
        assert [s.decode() for s in hf["seeds"][:]] == ["0", "1"]


def test_output_records_the_device_class(tmp_path, monkeypatch):
    """Amendment CQ, A5."""
    o = _run(tmp_path, monkeypatch)
    with h5py.File(o, "r") as hf:
        dc = hf.attrs["device_class"]
        dc = dc.decode() if isinstance(dc, bytes) else str(dc)
        assert dc.startswith("cpu:"), dc


# ---- amendment CT, §CT12.1: the stratum path ---------------------------------

def test_raw_slab_recovery_tolerates_a_degenerate_nan_draw():
    """A Null-B degenerate draw is NaN (BH2). The max over concepts must skip it,
    as classification_2x2's recovery does, or no seed can ever match."""
    import numpy as np
    from src.analysis.standardize_gain import assert_raw_slab_is_reference_row
    rng = np.random.default_rng(0)
    slabs = rng.normal(size=(3, 2, 4, 5)).astype(np.float32)   # seeds, shifts, concepts, feats
    slabs[1, 0, 2, :] = np.nan                                   # one degenerate concept draw
    null_g = np.stack([np.nanmax(s.astype(np.float64), axis=1) for s in slabs])
    for s in range(3):
        assert_raw_slab_is_reference_row(slabs[s], null_g, s, ["a", "b", "c"], "b")


def test_raw_slab_recovery_is_unchanged_without_nan():
    import numpy as np
    import pytest
    from src.analysis.standardize_gain import assert_raw_slab_is_reference_row
    rng = np.random.default_rng(1)
    slabs = rng.normal(size=(3, 2, 4, 5))
    null_g = slabs.max(axis=2)
    assert_raw_slab_is_reference_row(slabs[2], null_g, 2, ["a", "b", "c"], "b")
    with pytest.raises(SystemExit, match="IS the reductions' row 2"):
        assert_raw_slab_is_reference_row(slabs[2], null_g, 0, ["a", "b", "c"], "b")


def test_stratum_axis_values_counts_proteins_and_clusters():
    import numpy as np
    from src.analysis.standardize_gain import stratum_axis_values
    row_protein = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3])
    protein_cluster = ["k1", "k1", "k2", "k3"]
    rows = {"A": np.array([0, 1, 3]), "B": np.array([5, 8]), "C": np.array([], dtype=np.int64)}
    v = stratum_axis_values(rows, row_protein, protein_cluster, ["A", "B", "C"],
                            np.array([0.3, 0.2, 0.0]))
    assert v["protein_count"].tolist() == [2, 2, 0]        # A: proteins 0,1; B: 2,3
    assert v["cluster_count"].tolist() == [1, 2, 0]        # A: k1 only; B: k2, k3
    assert v["prevalence"].tolist() == [0.3, 0.2, 0.0]


# ---- amendment CT, §CT5.1: the admission gate (option 1, pools re-matched) ----

# Protein counts per stratum on the row set; "other" drops ACT_SITE (3 < 5).
_PC = {"pooled": {"ACT_SITE": 6, "BINDING": 600, "BINDING:ATP": 7, "DOMAIN": 700,
                  "DOMAIN:SH2": 60},
       "other": {"ACT_SITE": 3, "BINDING": 600, "BINDING:ATP": 9, "DOMAIN": 700,
                 "DOMAIN:SH2": 5}}
_CL = {"ACT_SITE": 1, "BINDING": 50, "BINDING:ATP": 2, "DOMAIN": 60, "DOMAIN:SH2": 5}


def _fake_counts(args, ap_attrs, concepts, strata):
    return {t: {"protein_count": np.array([_PC[t][c] for c in concepts], float),
                "cluster_count": np.array([_CL[c] for c in concepts], float),
                "n_positive_rows": np.array([10 * _PC[t][c] for c in concepts])}
            for t in strata}


def test_admission_mask_requires_every_stratum():
    from src.analysis.standardize_gain import admission_mask
    keep = admission_mask({"e": np.array([5, 4, 9, 0]),
                           "p": np.array([5, 9, 4, 7])}, 5)
    assert keep.tolist() == [True, False, False, False]
    assert admission_mask({"e": np.array([5, 4])}, 5).tolist() == [True, False]


def _restrict_inputs(ap, raw, red, keep_axis, keep_pool):
    """The same admission done to the FILES, by hand: the reference the code's
    in-memory restriction must reproduce bit for bit."""
    with h5py.File(ap, "a") as hf:
        g = hf["pooled"]
        c = [x.decode() for x in g["concepts"][:]]
        a, pv = g["ap"][:], g["prevalence"][:]
        for k in ("concepts", "ap", "prevalence"):
            del g[k]
        g.create_dataset("concepts", data=np.array(
            [x for x, k in zip(c, keep_axis) if k], dtype=h5py.string_dtype()))
        g.create_dataset("ap", data=a[:, :, keep_axis])
        g.create_dataset("prevalence", data=pv[keep_axis])
    with h5py.File(raw, "a") as hf:
        c = [x.decode() for x in hf["concepts"][:]]
        d = hf["gain_b"][:][:, :, keep_pool, :]
        del hf["concepts"], hf["gain_b"]
        hf.create_dataset("concepts", data=np.array(
            [x for x, k in zip(c, keep_pool) if k], dtype=h5py.string_dtype()))
        hf.create_dataset("gain_b", data=d)
    with h5py.File(red, "a") as hf:
        del hf["null_g_b"]
        hf.create_dataset("null_g_b", data=d.astype(np.float64).max(axis=2))


def test_admission_equals_restricting_the_inputs(tmp_path, monkeypatch):
    """The admitted run must be the run on inputs restricted by hand: same z,
    M, g~ and null M, bit for bit. A gate that dropped the concept from the AP
    but not from the null (or the reverse) would differ here or crash."""
    from src.analysis import standardize_gain as sg
    monkeypatch.setattr(sg, "_stratum_counts", _fake_counts)
    d1, d2 = tmp_path / "a", tmp_path / "b"
    d1.mkdir(); d2.mkdir()
    ap, raw, red, lab = _write_inputs(d1)
    _make_gain_positive(raw, red)
    o1 = d1 / "o.h5"
    monkeypatch.setattr(sys, "argv", _argv(
        ap, raw, red, lab, o1, "--inner-mode", "ratio", "--axis-values", "stratum",
        "--admission-strata", "pooled", "other"))
    sg.main()

    ap2, raw2, red2, lab2 = _write_inputs(d2)
    _make_gain_positive(raw2, red2)
    _restrict_inputs(ap2, raw2, red2, np.array([0, 1, 1, 1, 1], bool),
                     np.array([0, 1, 1], bool))
    o2 = d2 / "o.h5"
    monkeypatch.setattr(sys, "argv", _argv(
        ap2, raw2, red2, lab2, o2, "--inner-mode", "ratio", "--axis-values", "stratum"))
    sg.main()

    with h5py.File(o1, "r") as h1, h5py.File(o2, "r") as h2:
        assert [c.decode() for c in h1["concepts"][:]] == \
            ["BINDING", "BINDING:ATP", "DOMAIN", "DOMAIN:SH2"]
        for k in ("z", "M", "argmax_concept", "gtilde", "M_null", "stratum_mu",
                  "concept_cell"):
            np.testing.assert_array_equal(h1[k][:], h2[k][:], err_msg=k)
        assert int(h1.attrs["argmax_pool_used_n"]) == 2
        assert int(h1.attrs["argmax_pool_scored_n"]) == 3
        assert json.loads(h1.attrs["admission_strata"]) == ["pooled", "other"]
        assert int(h1.attrs["admission_min_protein_count"]) == 5
        g = h1["stratum_counts"]
        assert g["admitted"][:].tolist() == [False, True, True, True, True]
        assert g["other/protein_count"][:].tolist() == [3, 600, 9, 700, 5]


def test_admission_refuses_without_the_label_store_gate(tmp_path, monkeypatch):
    from src.analysis import standardize_gain as sg
    monkeypatch.setattr(sg, "_stratum_counts", _fake_counts)
    ap, raw, red, lab = _write_inputs(tmp_path)
    with h5py.File(lab, "a") as hf:
        del hf.attrs["min_protein_count"]
    monkeypatch.setattr(sys, "argv", _argv(
        ap, raw, red, lab, tmp_path / "o.h5", "--axis-values", "stratum",
        "--admission-strata", "pooled", "other"))
    with pytest.raises(SystemExit, match="no `min_protein_count`"):
        sg.main()


def test_admission_refuses_when_the_scored_stratum_is_not_gated(tmp_path, monkeypatch):
    from src.analysis import standardize_gain as sg
    monkeypatch.setattr(sg, "_stratum_counts", _fake_counts)
    ap, raw, red, lab = _write_inputs(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(
        ap, raw, red, lab, tmp_path / "o.h5", "--axis-values", "stratum",
        "--admission-strata", "other"))
    with pytest.raises(SystemExit, match="scored --stratum must be one"):
        sg.main()


def test_admission_with_an_empty_pool_is_unmeasurable(tmp_path, monkeypatch):
    from src.analysis import standardize_gain as sg

    def none_admitted(args, ap_attrs, concepts, strata):
        out = _fake_counts(args, ap_attrs, concepts, strata)
        out["other"]["protein_count"][:] = 1
        return out
    monkeypatch.setattr(sg, "_stratum_counts", none_admitted)
    ap, raw, red, lab = _write_inputs(tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(
        ap, raw, red, lab, tmp_path / "o.h5", "--axis-values", "stratum",
        "--admission-strata", "pooled", "other"))
    with pytest.raises(SystemExit, match="UNMEASURABLE: no pool concept"):
        sg.main()
    assert not (tmp_path / "o.h5").exists()


def test_pool_n_is_library_scoped_and_the_used_pool_is_separate(tmp_path, monkeypatch):
    """gotcha 16: `argmax_pool_n` is the AP's own (library) value; the pool the
    run maxed over is `argmax_pool_used_n`."""
    from src.analysis import standardize_gain as sg
    monkeypatch.setattr(sg, "_stratum_counts", _fake_counts)
    ap, raw, red, lab = _write_inputs(tmp_path)
    with h5py.File(ap, "a") as hf:
        hf.attrs["argmax_pool_n"] = 703
    o = tmp_path / "o.h5"
    monkeypatch.setattr(sys, "argv", _argv(
        ap, raw, red, lab, o, "--axis-values", "stratum",
        "--admission-strata", "pooled", "other"))
    sg.main()
    with h5py.File(o, "r") as hf:
        assert int(hf.attrs["argmax_pool_n"]) == 703
        assert int(hf.attrs["argmax_pool_used_n"]) == 2


def test_admission_never_promotes_a_pooled_parent(tmp_path, monkeypatch):
    """BINDING's only subtype is refused. Recomputing the pool on the restricted
    axis would make BINDING a pool member that can win the max (the I4 defect);
    the pool must be the original pool, restricted."""
    from src.analysis import standardize_gain as sg

    def drop_atp(args, ap_attrs, concepts, strata):
        out = _fake_counts(args, ap_attrs, concepts, strata)
        out["other"]["protein_count"][concepts.index("BINDING:ATP")] = 1
        return out
    monkeypatch.setattr(sg, "_stratum_counts", drop_atp)
    ap, raw, red, lab = _write_inputs(tmp_path)
    o = tmp_path / "o.h5"
    monkeypatch.setattr(sys, "argv", _argv(
        ap, raw, red, lab, o, "--axis-values", "stratum",
        "--admission-strata", "pooled", "other"))
    sg.main()
    with h5py.File(o, "r") as hf:
        c = [x.decode() for x in hf["concepts"][:]]
        pool = dict(zip(c, hf["in_argmax_pool"][:].tolist()))
    assert pool == {"BINDING": False, "DOMAIN": False, "DOMAIN:SH2": True}


# ---- §CT5.1-R: seed recovery in the raw array's stored precision ---------------

def _f64_null(seed=11):
    """A null whose gain was accumulated in float64 (the segmented CPU path):
    `null_g` keeps the float64 max, the raw array its float32 cast."""
    rng = np.random.default_rng(seed)
    G = rng.uniform(0, 1, (3, 4, 6, 50))                 # seeds, shifts, concepts, feats
    return G.astype(np.float32), G.max(axis=2)


def test_f64_null_fails_exact_float64_recovery_as_the_real_one_did():
    """The founding instance (§CT5.1-R): compared in float64 the slab matches no
    row, so the old check refused. This is what the fix exists for."""
    from src.analysis.standardize_gain import assert_raw_slab_is_reference_row
    raw, null_g = _f64_null()
    with pytest.raises(SystemExit, match="matches no row"):
        assert_raw_slab_is_reference_row(raw[0], null_g, 0, ["a", "b", "c"], "b")


def test_f64_null_recovers_every_slab_in_the_stored_precision():
    from src.analysis.standardize_gain import assert_raw_slab_is_reference_row
    raw, null_g = _f64_null()
    for s in range(3):
        assert_raw_slab_is_reference_row(raw[s], null_g, s, ["a", "b", "c"], "b",
                                         stored_dtype=raw.dtype)


def test_stored_precision_still_refuses_a_permuted_slab():
    from src.analysis.standardize_gain import assert_raw_slab_is_reference_row
    raw, null_g = _f64_null()
    with pytest.raises(SystemExit, match="IS the reductions' row 2"):
        assert_raw_slab_is_reference_row(raw[2], null_g, 0, ["a", "b", "c"], "b",
                                         stored_dtype=raw.dtype)


def test_stored_precision_still_refuses_foreign_reductions():
    """A difference far above float32 rounding is not absorbed by the cast."""
    from src.analysis.standardize_gain import assert_raw_slab_is_reference_row
    raw, null_g = _f64_null()
    with pytest.raises(SystemExit, match="matches no row"):
        assert_raw_slab_is_reference_row(raw[0], null_g + 1e-4, 0, ["a", "b", "c"],
                                         "b", stored_dtype=raw.dtype)


# ---- §CT14: the lowest firing bin merged into the next --------------------------

_SHARED_EDGES = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1.0 + 1e-9]   # probe 30308171


def test_merge_lowest_firing_bin_gives_the_declared_edges():
    from src.analysis.standardize_gain import merge_lowest_firing_bin
    assert merge_lowest_firing_bin(_SHARED_EDGES) == \
        [0.0, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1.0 + 1e-9]


def test_merge_lowest_firing_bin_refuses_a_single_bin():
    from src.analysis.standardize_gain import merge_lowest_firing_bin
    with pytest.raises(ValueError, match="fewer than two bins"):
        merge_lowest_firing_bin([0.0, 1.0])


def test_the_merge_removes_a_planted_zero_null_stratum():
    """§CT14's premise as a fixture: features below 1e-4 whose null is
    identically 0 make mu_s = 0 in firing bin 0 (CT5.1-R's cause 3). After the
    merge they share a stratum with bin-1 features whose null is not zero, and
    no stratum has mu <= 0. The same fixture without the merge keeps the zero."""
    from src.analysis.null_pooling import bin_index
    from src.analysis.standardize_gain import (merge_lowest_firing_bin,
                                               moments_from_sums, pooled_moments)
    fire = np.array([5e-5, 6e-5, 7e-5, 2e-4, 2.5e-4, 5e-4])   # 3 in bin 0, 2 in bin 1, 1 in bin 2
    slab = np.zeros((4, 2, 6))                                 # shifts, concepts, features
    slab[:, :, 3:] = np.arange(1, 4)[None, None, :] * 0.1      # bins 1-2 nonzero
    cell = np.zeros(2, dtype=int)

    def n_mu_le_0(edges):
        nfb = len(edges) - 1
        fb = np.clip(bin_index(fire, edges), 0, nfb - 1)
        s1, s2, cnt = pooled_moments(slab, cell, 1, fb, nfb)
        mu, _ = moments_from_sums(s1, s2, cnt)
        return int((np.isfinite(mu) & (mu <= 0)).sum())

    assert n_mu_le_0(_SHARED_EDGES) == 1
    assert n_mu_le_0(merge_lowest_firing_bin(_SHARED_EDGES)) == 0
