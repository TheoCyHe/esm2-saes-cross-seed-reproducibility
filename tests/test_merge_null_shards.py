"""
Tests for the masked-null shard merge (brief B).

The two checks this module adds over `merge_ap_shards` are the ones tested
here, and both are shaped by defects this project has already shipped:

  * `bh_invariants_identical` compares on the RAW BIT PATTERN, because
    `exceedance_resolution_*` is NaN wherever a concept is NOT_EVALUABLE and an
    `==` comparison would report every such entry as a disagreement -- a check
    that can only FAIL is as useless as one that can never fail (gotcha 25a).

  * `degeneracy_matches_nan_pattern` is POSITIONAL. A count of NaNs agrees
    under any permutation of the concept or shift axis, and so does a
    correlation; only a cell-by-cell check discriminates a mapping defect
    (gotcha 27). The decisive fixture below permutes the degeneracy flags while
    holding the TOTAL fixed -- the only way to prove the check sees what a
    total cannot.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from src.analysis.merge_null_shards import (
    bh_invariants_identical,
    degeneracy_matches_nan_pattern,
    main,
)


# ------------------------------------------- the seed-invariant bookkeeping --

def test_identical_invariants_across_shards_agree():
    a = np.array([[5, 5, 0], [3, 3, 3]], dtype=np.int64)
    r = bh_invariants_identical([a.copy(), a.copy(), a.copy()],
                                ["seed_0", "seed_1", "seed_2"], "surviving_positives_b")
    assert r["verdict"] == "IDENTICAL_ACROSS_SHARDS"
    assert r["disagreeing"] == []


def test_one_disagreeing_shard_is_NAMED_not_just_counted():
    a = np.array([[5, 5, 0]], dtype=np.int64)
    b = a.copy()
    b[0, 1] = 4
    r = bh_invariants_identical([a, b], ["seed_0", "seed_7"], "surviving_positives_b")
    assert r["verdict"] == "SHARDS_DISAGREE"
    assert r["disagreeing"][0]["shard"] == "seed_7"
    assert r["disagreeing"][0]["n_differing"] == 1


def test_NaN_carrying_invariants_that_AGREE_are_not_reported_as_differing():
    """THE 'can only fail' case. `exceedance_resolution_*` is NaN wherever a
    concept is NOT_EVALUABLE. Under an `==` comparison NaN != NaN, so every
    such entry would read as a disagreement on shards that are in fact
    byte-identical, and the merge would refuse every real run."""
    a = np.array([4.88e-06, np.nan, 5.14e-06], dtype=np.float64)
    r = bh_invariants_identical([a.copy(), a.copy()], ["seed_0", "seed_1"],
                                "exceedance_resolution_b")
    assert r["verdict"] == "IDENTICAL_ACROSS_SHARDS"


def test_a_shape_change_is_reported_as_SHAPE_not_as_a_value_difference():
    r = bh_invariants_identical([np.zeros((2, 3)), np.zeros((2, 4))],
                                ["seed_0", "seed_1"], "surviving_positives_a")
    assert r["verdict"] == "SHARDS_DISAGREE"
    assert r["disagreeing"][0]["reason"] == "SHAPE"


def test_string_invariants_compare_by_VALUE_across_shards():
    a = np.array([b"NONE_DEGENERATE", b"SOME_BUT_NOT_ALL_DEGENERATE"])
    b = np.array([b"NONE_DEGENERATE", b"NONE_DEGENERATE"])
    assert bh_invariants_identical([a, a.copy()], ["s0", "s1"], "bh3_case")[
        "verdict"] == "IDENTICAL_ACROSS_SHARDS"
    assert bh_invariants_identical([a, b], ["s0", "s1"], "bh3_case")[
        "verdict"] == "SHARDS_DISAGREE"


# -------------------------------------------------- BH2's NaN pattern --

def _gain(n_seed, n_shift, n_c, n_f, degenerate):
    """A raw gain array whose NaN cells are exactly `degenerate` (concepts x
    shifts), as the writer produces."""
    g = np.ones((n_seed, n_shift, n_c, n_f), dtype=np.float32)
    for ci in range(n_c):
        for r in range(n_shift):
            if degenerate[ci, r]:
                g[:, r, ci, :] = np.nan
    return g


def test_a_correct_NaN_pattern_matches():
    degen = np.zeros((3, 4), dtype=bool)
    degen[1, 2] = True
    r = degeneracy_matches_nan_pattern(_gain(2, 4, 3, 5, degen), degen)
    assert r["verdict"] == "NAN_PATTERN_MATCHES_BH2"
    assert r["n_degenerate_cells_with_values"] == 0
    assert r["n_surviving_cells_with_nan"] == 0


def test_a_degenerate_cell_carrying_VALUES_is_caught():
    """BH2 says no value is imputed. A degenerate cell holding 0.0 would be
    folded into every mean, max and exceedance count that reads the array."""
    degen = np.zeros((3, 4), dtype=bool)
    degen[1, 2] = True
    g = _gain(2, 4, 3, 5, degen)
    g[:, 2, 1, :] = 0.0                      # imputed, exactly the defect
    r = degeneracy_matches_nan_pattern(g, degen)
    assert r["verdict"] == "NAN_PATTERN_WRONG"
    assert r["n_degenerate_cells_with_values"] == 2      # one per seed


def test_a_SURVIVING_cell_carrying_NaN_is_caught():
    """The other direction: a draw that survived BG2 but produced NaN means the
    arithmetic failed, and a consumer dropping NaNs would silently lose it."""
    degen = np.zeros((3, 4), dtype=bool)
    g = _gain(2, 4, 3, 5, degen)
    g[0, 1, 2, 3] = np.nan
    r = degeneracy_matches_nan_pattern(g, degen)
    assert r["verdict"] == "NAN_PATTERN_WRONG"
    assert r["n_surviving_cells_with_nan"] == 1
    assert r["first_surviving_cells_with_nan_(seed,shift,concept)"][0] == [0, 1, 2]


def test_a_PERMUTED_degeneracy_with_the_SAME_TOTAL_is_caught():
    """THE decisive fixture, and the only one that proves this check is worth
    having.

    The gain array's NaN sits at (concept 1, shift 2); the flags say (concept
    2, shift 1). The TOTAL number of degenerate cells is identical, so a count
    agrees exactly -- and so would any correlation. Only the cell-by-cell
    comparison discriminates (gotcha 27: for a mapping defect, only a
    positional check discriminates).
    """
    truth = np.zeros((3, 4), dtype=bool)
    truth[1, 2] = True
    swapped = np.zeros((3, 4), dtype=bool)
    swapped[2, 1] = True
    assert truth.sum() == swapped.sum()          # the totals agree

    r = degeneracy_matches_nan_pattern(_gain(2, 4, 3, 5, truth), swapped)
    assert r["verdict"] == "NAN_PATTERN_WRONG"
    assert r["n_degenerate_cells_with_values"] == 2   # flagged, but has values
    assert r["n_surviving_cells_with_nan"] == 2       # NaN where none expected


def test_the_expected_degenerate_count_scales_with_the_SEED_axis():
    """The flags are per (concept, shift) and the array is per (seed, shift,
    concept), so the expected NaN-cell count is flags x seeds. Reporting the
    unscaled count would make a correct merge look like it had lost 19 of
    every 20 degenerate cells."""
    degen = np.zeros((3, 4), dtype=bool)
    degen[0, 0] = True
    degen[2, 3] = True
    r = degeneracy_matches_nan_pattern(_gain(5, 4, 3, 6, degen), degen)
    assert r["n_degenerate_cells_expected"] == 2 * 5
    assert r["n_cells_all_nan"] == 2 * 5
    assert r["verdict"] == "NAN_PATTERN_MATCHES_BH2"


# ==================================================== the ENTRY POINT itself ==
#
# Gotcha 13 extension 4's corollary: a unit suite that never executes the entry
# point cannot catch an entry-point bug. Four such defects have shipped in this
# project, one of which passed 244 tests because every test called the helpers
# directly and none ran `main()`. `merge` builds `write_identity(**base, ...)`
# from a shard's own attrs, and a key present in BOTH `base` and the explicit
# arguments is a TypeError that no helper test can reach.

_N_SHIFT, _N_C, _N_F = 3, 4, 5


def _write_shard(tmp_path, seed_label, degen, gain_scale, fire_rate=None,
                 bin_counts=None):
    red = tmp_path / f"red_{seed_label}.h5"
    raw = tmp_path / f"raw_{seed_label}.h5"
    surv = np.where(degen, 0, 7).astype(np.int64)          # (concepts, shifts)
    with h5py.File(red, "w") as hf:
        hf.create_dataset("seeds", data=np.array([seed_label],
                                                 dtype=h5py.string_dtype()))
        hf.create_dataset("concepts", data=np.array(
            [f"C{i}" for i in range(_N_C)], dtype=h5py.string_dtype()))
        hf.create_dataset("null_g_a", data=np.full((1, _N_SHIFT, _N_F), gain_scale))
        hf.create_dataset("fire_rate", data=(
            np.full((1, _N_F), 0.003125) if fire_rate is None
            else np.asarray(fire_rate, dtype=np.float64).reshape(1, _N_F)))
        hf.create_dataset("surviving_positives_b", data=surv)
        hf.create_dataset("surviving_draws_b",
                          data=(~degen).sum(axis=1).astype(np.int64))
        hf.create_dataset("exceedance_resolution_b", data=np.array(
            [1.0 / (d * _N_F + 1) if d else np.nan
             for d in (~degen).sum(axis=1)]))
        hf.create_dataset("not_evaluable_b", data=((~degen).sum(axis=1) == 0))
        hf.create_dataset("n_positives_observed",
                          data=np.full(_N_C, 7, dtype=np.int64))
        # Attrs deliberately include keys the merge also passes explicitly.
        hf.attrs.update(stratum="experimental", negatives="unannotated",
                        n_shifts=_N_SHIFT, null_seed=0, null_type="both",
                        n_rows=100, n_proteins=10, n_length_bins=2,
                        truncation_policy="untruncated", library_sha="deadbeef",
                        n_concepts=9, argmax_pool_n=8, argmax_pool_scored_n=_N_C,
                        n_concepts_scored=_N_C, argmax_pool_used_n=_N_C,
                        scored_row_set="label_store", ap_impl="seg/v1",
                        exclude_pooled_parents=True,
                        retention_floor_mode="interplm",
                        retention_min_residues=1500,
                        retention_min_instances=10, min_protein_count=5,
                        git_sha="abc", git_dirty=False, raw_gain_path=str(raw))
        if bin_counts is not None:
            hf.attrs["bin_counts"] = bin_counts
    g = np.full((1, _N_SHIFT, _N_C, _N_F), gain_scale, dtype=np.float32)
    for ci in range(_N_C):
        for r in range(_N_SHIFT):
            if degen[ci, r]:
                g[0, r, ci, :] = np.nan
    with h5py.File(raw, "w") as hf:
        hf.create_dataset("gain_b", data=g, chunks=(1, 1, _N_C, _N_F))
    return red, raw


def _reference(tmp_path, labels):
    ref = tmp_path / "reference.h5"
    with h5py.File(ref, "w") as hf:
        hf.create_dataset("seeds", data=np.array(labels, dtype=h5py.string_dtype()))
    return ref


def test_main_merges_and_orders_rows_by_the_reference_LABELS(tmp_path):
    """The shards are supplied in one order and the reference names another.
    The merged rows must follow the REFERENCE, not the argument order --
    gotcha 29, where a positional join mis-pairs 18 of 20 real seeds."""
    degen = np.zeros((_N_C, _N_SHIFT), dtype=bool)
    degen[1, 2] = True
    shards = {lab: _write_shard(tmp_path, lab, degen, i + 1.0)
              for i, lab in enumerate(["seed_0", "seed_1", "seed_10"])}
    ref = _reference(tmp_path, ["seed_10", "seed_0", "seed_1"])
    order_in = ["seed_0", "seed_1", "seed_10"]

    rc = main([
        "--shard-reductions", *[str(shards[s][0]) for s in order_in],
        "--shard-raw", *[str(shards[s][1]) for s in order_in],
        "--reference", str(ref),
        "--out", str(tmp_path / "merged.h5"),
        "--raw-out", str(tmp_path / "merged_raw.h5"),
        "--report", str(tmp_path / "report.json")])
    assert rc == 0

    with h5py.File(tmp_path / "merged.h5", "r") as hf:
        assert [s.decode() for s in hf["seeds"][:]] == ["seed_10", "seed_0", "seed_1"]
        # gain_scale was 1.0/2.0/3.0 for seed_0/seed_1/seed_10, so the merged
        # rows must carry 3.0, 1.0, 2.0 -- a POSITIONAL check on the payload.
        assert hf["null_g_a"][:, 0, 0].tolist() == [3.0, 1.0, 2.0]
        assert hf["surviving_positives_b"].shape == (_N_C, _N_SHIFT)
        assert "not_evaluable_b" in hf and "exceedance_resolution_b" in hf
    with h5py.File(tmp_path / "merged_raw.h5", "r") as hf:
        assert hf["gain_b"].shape == (3, _N_SHIFT, _N_C, _N_F)
        assert np.isnan(hf["gain_b"][0, 2, 1, :]).all()      # the degenerate cell
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["seed_order"]["verdict"] == "POSITIONS_OK"
    assert report["nan_pattern_ok"] is True


def test_main_REFUSES_shards_whose_seed_invariant_bookkeeping_disagrees(tmp_path):
    """The BH datasets are functions of the stored schedules and the concept's
    mask -- no SAE enters them -- so a disagreement means a shard was built
    against a different label store, schedule or null seed. No identity key
    would show it: none of them is positional."""
    degen = np.zeros((_N_C, _N_SHIFT), dtype=bool)
    other = degen.copy()
    other[0, 0] = True                       # a different degeneracy pattern
    a = _write_shard(tmp_path, "seed_0", degen, 1.0)
    b = _write_shard(tmp_path, "seed_1", other, 2.0)
    ref = _reference(tmp_path, ["seed_0", "seed_1"])
    with pytest.raises(SystemExit) as e:
        main(["--shard-reductions", str(a[0]), str(b[0]),
              "--shard-raw", str(a[1]), str(b[1]),
              "--reference", str(ref),
              "--out", str(tmp_path / "m.h5"),
              "--raw-out", str(tmp_path / "mr.h5"),
              "--report", str(tmp_path / "r.json")])
    assert "seed-invariant" in str(e.value)


def test_main_REFUSES_a_missing_seed_rather_than_shortening_the_axis(tmp_path):
    """A 2-shard merge against a 3-seed reference must fail loudly. Silently
    dropping one turns a 3-seed artifact into a 2-seed one whose shape does not
    show the gap."""
    degen = np.zeros((_N_C, _N_SHIFT), dtype=bool)
    a = _write_shard(tmp_path, "seed_0", degen, 1.0)
    b = _write_shard(tmp_path, "seed_1", degen, 2.0)
    ref = _reference(tmp_path, ["seed_0", "seed_1", "seed_10"])
    with pytest.raises(ValueError, match="seed sets differ"):
        main(["--shard-reductions", str(a[0]), str(b[0]),
              "--shard-raw", str(a[1]), str(b[1]),
              "--reference", str(ref),
              "--out", str(tmp_path / "m.h5"),
              "--raw-out", str(tmp_path / "mr.h5"),
              "--report", str(tmp_path / "r.json")])


def test_main_REFUSES_when_the_shard_counts_disagree(tmp_path):
    degen = np.zeros((_N_C, _N_SHIFT), dtype=bool)
    a = _write_shard(tmp_path, "seed_0", degen, 1.0)
    b = _write_shard(tmp_path, "seed_1", degen, 2.0)
    ref = _reference(tmp_path, ["seed_0", "seed_1"])
    with pytest.raises(SystemExit, match="refusing to pair them by"):
        main(["--shard-reductions", str(a[0]), str(b[0]),
              "--shard-raw", str(a[1]),
              "--reference", str(ref),
              "--out", str(tmp_path / "m.h5"),
              "--raw-out", str(tmp_path / "mr.h5"),
              "--report", str(tmp_path / "r.json")])


def test_merged_bin_counts_describe_ALL_shards_not_shard_0(tmp_path):
    """Amendment CT, CT10-17 preamble: the merge used to copy shard 0's
    `bin_counts` (one seed, so min = max) onto the merged null. CT5's merged
    nulls say 42 features in bin 0 while the 20 seeds hold 33-48. Here shard 0
    has 1 feature below 1e-4 and shard 1 has 4; the merged attr must say 1-4,
    in BOTH merged files, and must not be the stale string shard 0 carried."""
    from src.analysis.null_pilot import per_seed_bin_counts
    degen = np.zeros((_N_C, _N_SHIFT), dtype=bool)
    fr0 = [5e-5, 2e-3, 2e-3, 2e-3, 2e-3]
    fr1 = [5e-5, 5e-5, 5e-5, 5e-5, 2e-3]
    stale = json.dumps(per_seed_bin_counts(np.array([fr0]), _N_SHIFT))
    a = _write_shard(tmp_path, "seed_0", degen, 1.0, fire_rate=fr0, bin_counts=stale)
    b = _write_shard(tmp_path, "seed_1", degen, 2.0, fire_rate=fr1, bin_counts=stale)
    ref = _reference(tmp_path, ["seed_0", "seed_1"])
    assert main(["--shard-reductions", str(a[0]), str(b[0]),
                 "--shard-raw", str(a[1]), str(b[1]),
                 "--reference", str(ref),
                 "--out", str(tmp_path / "m.h5"),
                 "--raw-out", str(tmp_path / "mr.h5"),
                 "--report", str(tmp_path / "r.json")]) == 0
    for path in ("m.h5", "mr.h5"):
        with h5py.File(tmp_path / path, "r") as hf:
            got = json.loads(hf.attrs["bin_counts"])
        bin0 = got[0]
        assert (bin0["lo"], bin0["hi"]) == (0.0, 1e-4)
        assert bin0["n_features_min_over_seeds"] == 1
        assert bin0["n_features_max_over_seeds"] == 4
        assert bin0["n_null_draws_min"] == 1 * _N_SHIFT * 2
        assert json.dumps(got) != stale
