"""Tests for `d_fs` (PREREGISTRATION amendment AO section 4).

The population reconstruction and the cluster join are the two places this can
be silently wrong (gotcha 10 for the accession convention, AO6 for the protein
ORDER), so both are tested against fixtures that can actually exhibit the
failure: the label store's `population_start` is NON-ASCENDING, and the cache
index contains proteins the population must exclude.
"""

from __future__ import annotations

import hashlib
import json
import sys

import h5py
import numpy as np
import pytest
import yaml

from src.analysis import repr_density as R

# Four cache proteins, length-sorted like the real cache, the last TRUNCATED.
# Accession order (A, B, C) does NOT match start order (B, A, C) -- the AO6
# condition. Bare accessions are wrapped in the FASTA token form so
# `header_to_acc` has real work to do (gotcha 10).
CACHE_INDEX = [
    {"accession": "sp|B0001|B_TEST", "start": 0, "n_rows": 3, "seq_len": 3,
     "truncated": False, "residue_ids": None},
    {"accession": "sp|A0002|A_TEST", "start": 3, "n_rows": 2, "seq_len": 2,
     "truncated": False, "residue_ids": None},
    {"accession": "sp|C0003|C_TEST", "start": 5, "n_rows": 4, "seq_len": 4,
     "truncated": False, "residue_ids": None},
    {"accession": "sp|D0004|D_TEST", "start": 9, "n_rows": 5, "seq_len": 9999,
     "truncated": True, "residue_ids": None},
]
POP_ACCS_SORTED = ["A0002", "B0001", "C0003"]        # the store's own order
STORED_START = [3, 0, 5]
STORED_NROW = [2, 3, 4]
N_POS = 9


def _eval_cache(tmp_path, index=None):
    p = tmp_path / "eval.h5"
    with h5py.File(p, "w") as hf:
        hf.create_dataset("index_json",
                          data=json.dumps(index if index is not None
                                          else CACHE_INDEX))
        hf.create_dataset("activations", shape=(14, 2), dtype="f2")
    return p


def _labels(tmp_path, start=None, nrow=None, policy="untruncated",
            n_rows=N_POS):
    p = tmp_path / "lab.h5"
    with h5py.File(p, "w") as hf:
        pg = hf.create_group("protein_sets")
        pg.create_dataset("population_start",
                          data=np.array(start or STORED_START, np.int64))
        pg.create_dataset("population_n_rows",
                          data=np.array(nrow or STORED_NROW, np.int64))
        hf.attrs["n_rows"] = n_rows
        hf.attrs["truncation_policy"] = policy
    return p


def _clusters(tmp_path, rows=(("B0001", "B0001"), ("B0001", "A0002"),
                              ("C0003", "C0003"))):
    """mmseqs TSV: column 1 is the REPRESENTATIVE, column 2 the MEMBER.

    `load_cluster_map` keys on the MEMBER, so getting these round the wrong way
    leaves the second protein absent from the map entirely -- which the match-rate
    gate then catches. It caught it here, in this fixture, on the first run.
    """
    p = tmp_path / "clu.tsv"
    p.write_text("".join(f"{rep}\t{mem}\n" for rep, mem in rows))
    return p


# ---------------------------------------------------------------------------
# The population reconstruction. The two elementwise comparisons ARE the check.
# MUTATION: drop the `derived_start != stored_start` comparison and
# test_population_reconstruction_refuses_a_wrong_order dies.
# ---------------------------------------------------------------------------
def test_population_reconstruction_verifies_against_the_store(tmp_path):
    got = R.population_accessions(_eval_cache(tmp_path), _labels(tmp_path))
    assert got["accessions_stored_order"] == POP_ACCS_SORTED
    assert got["accessions_start_order"] == ["B0001", "A0002", "C0003"]
    v = got["verification"]
    assert v["n_cache_proteins"] == 4
    assert v["n_population_proteins"] == 3          # the truncated one excluded
    assert v["n_rows"] == N_POS
    # the real store's shape, recorded as a measurement rather than assumed
    assert v["population_start_is_ascending_in_stored_order"] is False


def test_population_reconstruction_refuses_a_wrong_order(tmp_path):
    """A store whose arrays were written in a different order must be refused,
    not silently joined against."""
    lab = _labels(tmp_path, start=[0, 3, 5], nrow=[3, 2, 4])   # start order
    with pytest.raises(SystemExit, match="does NOT match the label store"):
        R.population_accessions(_eval_cache(tmp_path), lab)


def test_population_reconstruction_refuses_a_start_array_that_alone_is_wrong(
        tmp_path):
    """DISCRIMINATION TEST, added because a mutation SURVIVED without it.

    `test_population_reconstruction_refuses_a_wrong_order` perturbs the start
    AND the row-count arrays together, so deleting the start comparison left it
    green -- it passed via the `n_rows` branch. That is gotcha 13 extension 3
    exactly: the test failed for the wrong reason.

    This fixture keeps `population_n_rows` EXACTLY right and moves one start
    value, so the start comparison is the only check that can fire. The row
    total still sums to 9, so that branch cannot rescue it either.
    """
    lab = _labels(tmp_path, start=[3, 0, 6], nrow=[2, 3, 4])
    with pytest.raises(SystemExit, match="does NOT match the label store"):
        R.population_accessions(_eval_cache(tmp_path), lab)


def test_population_reconstruction_refuses_a_row_count_that_alone_is_wrong(
        tmp_path):
    """The mirror of the above, so neither branch is load-bearing alone. Starts
    are exactly right; one row count is not, and the total still differs only
    through it."""
    lab = _labels(tmp_path, start=[3, 0, 5], nrow=[2, 3, 5], n_rows=10)
    with pytest.raises(SystemExit, match="does NOT match the label store"):
        R.population_accessions(_eval_cache(tmp_path), lab)


def test_population_reconstruction_refuses_a_wrong_truncation_policy(tmp_path):
    """with_truncated keeps 4 proteins and 14 rows, so the store's 3/9 must
    fail -- this is what makes the population filter part of the check."""
    lab = _labels(tmp_path, policy="with_truncated")
    with pytest.raises(SystemExit, match="does NOT match the label store"):
        R.population_accessions(_eval_cache(tmp_path), lab)


def test_population_reconstruction_refuses_an_unknown_policy(tmp_path):
    lab = _labels(tmp_path, policy="whatever")
    with pytest.raises(SystemExit, match="unknown truncation_policy"):
        R.population_accessions(_eval_cache(tmp_path), lab)


def test_accession_normalisation_is_applied(tmp_path):
    """Without header_to_acc the join would key on 'sp|B0001|B_TEST' and match
    nothing -- gotcha 10's first-run failure, which exited 0."""
    got = R.population_accessions(_eval_cache(tmp_path), _labels(tmp_path))
    assert all("|" not in a for a in got["accessions_stored_order"])


# ---------------------------------------------------------------------------
# The cluster join. Match rate asserted (gotcha 10) and unmatched proteins get
# singleton clusters rather than being dropped or lumped.
# ---------------------------------------------------------------------------
def test_cluster_join_maps_start_order_proteins_to_dense_indices(tmp_path):
    clus = R.cluster_assignment(["B0001", "A0002", "C0003"], _clusters(tmp_path))
    assert clus["report"]["match_fraction"] == 1.0
    assert clus["report"]["n_distinct_clusters_in_population"] == 2
    # B and A share a representative, C is its own
    cl = clus["cluster_index"]
    assert cl[0] == cl[1] and cl[2] != cl[0]


def test_cluster_join_refuses_a_near_zero_match_rate(tmp_path):
    tsv = _clusters(tmp_path, rows=(("Z9999", "Z9999"),))
    with pytest.raises(SystemExit, match="accession-convention"):
        R.cluster_assignment(["B0001", "A0002", "C0003"], tsv)


def test_unmatched_protein_gets_a_singleton_cluster_not_a_shared_one(tmp_path):
    """Lumping unmatched proteins into one bucket would fuse unrelated proteins
    into a fake cluster and understate a feature's effective support."""
    tsv = tmp_path / "c.tsv"
    rows = [("B0001", "B0001"), ("B0001", "A0002")] + [
        (f"X{i:04d}", f"X{i:04d}") for i in range(300)]
    tsv.write_text("".join(f"{r}\t{m}\n" for r, m in rows))
    clus = R.cluster_assignment(["B0001", "A0002"] + [f"X{i:04d}" for i in
                                                      range(300)] + ["C0003"],
                                tsv)
    assert clus["report"]["n_unmatched_given_singleton_clusters"] == 1
    assert len(set(clus["cluster_index"].tolist())) == clus["n_clusters"] - 0
    assert clus["cluster_index"][-1] not in clus["cluster_index"][:-1]


# ---------------------------------------------------------------------------
# Entropy: an unfired feature has an UNDEFINED support, and 0 would read as
# "measured, concentrated in one cluster" (gotcha 25b).
# ---------------------------------------------------------------------------
def test_zero_total_gives_nan_not_zero_entropy():
    h = R._entropy_from_moments(np.array([0.0, 4.0]), np.array([0.0, 0.0]))
    assert np.isnan(h[0])
    assert h[1] == pytest.approx(np.log(4.0))


# ---------------------------------------------------------------------------
# THE SWEEP. Hand-built row index with a known answer, so the grouping is proven.
# ---------------------------------------------------------------------------
def _row_index(tmp_path, fires_by_position, n_features, name="ri.h5",
               ckpt_sha=None, attrs=None):
    offs = np.zeros(len(fires_by_position) + 1, np.int64)
    flat = []
    for i, fs in enumerate(fires_by_position):
        flat.extend(fs)
        offs[i + 1] = len(flat)
    fc = np.bincount(np.array(flat, int), minlength=n_features)
    p = tmp_path / name
    with h5py.File(p, "w") as hf:
        hf.create_dataset("feat_by_row", data=np.array(flat, np.int32))
        hf.create_dataset("row_offsets", data=offs)
        hf.create_dataset("fire_counts", data=fc.astype(np.int64))
        if ckpt_sha is not None:
            hf.attrs["checkpoint_sha256_16"] = ckpt_sha
        hf.attrs.update(**(attrs or {}))
    return p


# positions 0-2 = B, 3-4 = A, 5-8 = C  (start order).
# B and A are in ONE cluster; C in another.
#   feature 0: fires 3x, all in B                -> 1 cluster, 1 protein
#   feature 1: fires at every position           -> 2 clusters, 3 proteins
#   feature 2: fires 2x in B and 2x in A         -> 1 cluster, 2 proteins
FIRES = [[0, 1, 2], [0, 1, 2], [0, 1], [1, 2], [1, 2], [1], [1], [1], [1]]


def test_sweep_computes_cluster_and_protein_footprints(tmp_path):
    lab = _labels(tmp_path)
    pop = R.population_accessions(_eval_cache(tmp_path), lab)
    clus = R.cluster_assignment(pop["accessions_start_order"],
                                _clusters(tmp_path))
    nrow_start = pop["stored_nrow"][pop["order"]]
    bc = np.concatenate([[0], np.cumsum(nrow_start)])
    bb = np.concatenate([[0], np.cumsum(pop["stored_nrow"])])
    ri = _row_index(tmp_path, FIRES, 3)
    res = R.sweep_seed(ri, 3, bc, bb, clus["cluster_index"], clus["n_clusters"])

    assert res["n_clusters_fired"].tolist() == [1, 2, 1]
    assert res["n_proteins_correct"].tolist() == [1, 3, 2]
    assert res["cluster_total_fires"].tolist() == [3.0, 9.0, 4.0]
    # feature 0 sits in one cluster -> entropy 0 -> effective support 1
    assert res["cluster_entropy_nats"][0] == pytest.approx(0.0)
    # feature 1: 5 fires in the B+A cluster, 4 in C
    c = np.array([5.0, 4.0])
    want = np.log(c.sum()) - (c * np.log(c)).sum() / c.sum()
    assert res["cluster_entropy_nats"][1] == pytest.approx(want)


def test_sweep_protein_footprint_differs_between_the_two_groupings(tmp_path):
    """AO6's magnitude, on a fixture: the pre-AO6 grouping gives DIFFERENT
    per-protein statistics, so measuring both in one pass is meaningful."""
    lab = _labels(tmp_path)
    pop = R.population_accessions(_eval_cache(tmp_path), lab)
    clus = R.cluster_assignment(pop["accessions_start_order"],
                                _clusters(tmp_path))
    nrow_start = pop["stored_nrow"][pop["order"]]
    bc = np.concatenate([[0], np.cumsum(nrow_start)])
    bb = np.concatenate([[0], np.cumsum(pop["stored_nrow"])])
    ri = _row_index(tmp_path, FIRES, 3)
    res = R.sweep_seed(ri, 3, bc, bb, clus["cluster_index"], clus["n_clusters"])
    # correct: feature 0 in 1 protein; pre-AO6 blocks are A(0-1), B(2-4), C(5-8)
    # so its three fires at 0,1,2 straddle two blocks
    assert res["n_proteins_correct"][0] == 1
    assert res["n_proteins_preAO6"][0] == 2
    assert res["protein_entropy_correct"][0] == pytest.approx(0.0)
    assert res["protein_entropy_preAO6"][0] > 0.5
    # the TOTAL fires must agree under either grouping -- which is exactly why a
    # total-based check could never have caught AO6
    assert res["protein_total_fires_correct"].tolist() == \
        res["protein_total_fires_preAO6"].tolist()


def test_sweep_refuses_a_zero_row_protein(tmp_path):
    """A protein occupying no positions would desync the flush counter. Refused
    up front, with a message that names the cause."""
    bc = np.array([0, 3, 3, 9])            # protein 1 has zero rows
    with pytest.raises(SystemExit, match="ZERO rows"):
        R.sweep_seed(_row_index(tmp_path, FIRES, 3), 3, bc, None,
                     np.zeros(3, np.int32), 1)


def test_sweep_refuses_a_differently_scoped_row_index(tmp_path):
    lab = _labels(tmp_path)
    pop = R.population_accessions(_eval_cache(tmp_path), lab)
    clus = R.cluster_assignment(pop["accessions_start_order"],
                                _clusters(tmp_path))
    bc = np.concatenate([[0], np.cumsum(pop["stored_nrow"][pop["order"]])])
    ri = _row_index(tmp_path, FIRES[:-1], 3)          # one position short
    with pytest.raises(SystemExit, match="differently-scoped index"):
        R.sweep_seed(ri, 3, bc, None, clus["cluster_index"], clus["n_clusters"])


def test_sweep_without_the_ao6_arm_omits_it(tmp_path):
    lab = _labels(tmp_path)
    pop = R.population_accessions(_eval_cache(tmp_path), lab)
    clus = R.cluster_assignment(pop["accessions_start_order"],
                                _clusters(tmp_path))
    bc = np.concatenate([[0], np.cumsum(pop["stored_nrow"][pop["order"]])])
    res = R.sweep_seed(_row_index(tmp_path, FIRES, 3), 3, bc, None,
                       clus["cluster_index"], clus["n_clusters"])
    assert "protein_entropy_preAO6" not in res
    assert res["n_clusters_fired"].tolist() == [1, 2, 1]


# ---------------------------------------------------------------------------
# THE ENTRY POINT (gotcha 13 extension 4, corollary).
# ---------------------------------------------------------------------------
def _main_fixture(tmp_path):
    ev = _eval_cache(tmp_path)
    lab = _labels(tmp_path)
    tsv = _clusters(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(
        {"confound_control": {"analysis_min_firing_rows": 2}}))
    ckpts = tmp_path / "ckpts"
    (ckpts / "seed_0").mkdir(parents=True)
    (ckpts / "seed_0" / "model.pt").write_bytes(b"seed-zero")
    sha = hashlib.sha256(b"seed-zero").hexdigest()[:16]
    ric = tmp_path / "ric"
    ric.mkdir()
    _row_index(ric, FIRES, 3, name="row_index_seed_0_deadbeef.h5",
               ckpt_sha=sha,
               attrs={"scored_row_set": "label_store", "k": 32,
                      "n_rows": N_POS})
    return cfg, lab, ev, tsv, ric, ckpts


def test_main_end_to_end_writes_the_flag_and_scores_the_predictions(tmp_path,
                                                                   monkeypatch):
    cfg, lab, ev, tsv, ric, ckpts = _main_fixture(tmp_path)
    out, outj = tmp_path / "d.h5", tmp_path / "d.json"
    monkeypatch.setattr(sys, "argv", [
        "dfs", "--config", str(cfg), "--labels", str(lab),
        "--eval-cache", str(ev), "--clusters", str(tsv),
        "--row-index-cache", str(ric), "--ckpts", str(ckpts),
        "--seeds", "0", "--out", str(out), "--out-json", str(outj)])
    R.main()
    doc = json.loads(outj.read_text())
    assert doc["cluster_join"]["match_fraction"] == 1.0
    assert doc["prediction_scorecard"]["P9"]["outcome"] == "HIT"
    assert doc["prediction_scorecard"]["P8"]["outcome"] in {"HIT", "MISS"}
    assert doc["prediction_scorecard"]["P8"]["adopted_definition"] in {
        "A_cluster_entropy", "B_log_distinct_clusters"}
    assert doc["prediction_scorecard"]["P10"]["outcome"] in {"HIT", "MISS"}
    with h5py.File(out, "r") as hf:
        assert hf["d_fs_cluster_entropy"].shape == (1, 3)
        assert hf["d_fs_log_distinct_clusters"].shape == (1, 3)
        assert hf.attrs["analysis_min_firing_rows"] == 2
        assert hf.attrs["truncation_policy"] == "untruncated"
        assert hf.attrs["adopted_definition"] == doc[
            "prediction_scorecard"]["P8"]["adopted_definition"]
        # B = log(distinct clusters); features 0 and 2 fire in one cluster each
        assert hf["d_fs_log_distinct_clusters"][0].tolist() == pytest.approx(
            [0.0, np.log(2.0), 0.0])


def test_main_fails_when_the_sweep_total_disagrees_with_fire_counts(tmp_path,
                                                                   monkeypatch):
    """The cross-check between the sweep's own totals and the row index's
    fire_counts is real: corrupt one and the job must refuse."""
    cfg, lab, ev, tsv, ric, ckpts = _main_fixture(tmp_path)
    ri = next(ric.glob("row_index_seed_0_*.h5"))
    with h5py.File(ri, "a") as hf:
        fc = hf["fire_counts"][:]
        fc[0] += 1
        hf["fire_counts"][:] = fc
    monkeypatch.setattr(sys, "argv", [
        "dfs", "--config", str(cfg), "--labels", str(lab),
        "--eval-cache", str(ev), "--clusters", str(tsv),
        "--row-index-cache", str(ric), "--ckpts", str(ckpts),
        "--seeds", "0", "--out", str(tmp_path / "d.h5"),
        "--out-json", str(tmp_path / "d.json")])
    with pytest.raises(SystemExit, match="counted exactly once"):
        R.main()


def test_main_reports_P10_NOT_SCORED_when_the_ao6_arm_is_skipped(tmp_path,
                                                                 monkeypatch):
    cfg, lab, ev, tsv, ric, ckpts = _main_fixture(tmp_path)
    outj = tmp_path / "d.json"
    monkeypatch.setattr(sys, "argv", [
        "dfs", "--config", str(cfg), "--labels", str(lab),
        "--eval-cache", str(ev), "--clusters", str(tsv),
        "--row-index-cache", str(ric), "--ckpts", str(ckpts),
        "--seeds", "0", "--no-ao6-comparison",
        "--out", str(tmp_path / "d.h5"), "--out-json", str(outj)])
    R.main()
    doc = json.loads(outj.read_text())
    assert doc["prediction_scorecard"]["P10"]["outcome"].startswith("NOT_SCORED")
    assert doc["AO6_grouping_comparison_across_seeds"] == []


# ---------------------------------------------------------------------------
# THE POSITIONAL CHECK (brief section 0). A total cannot see a permutation and
# neither can a rank; only this discriminates. Both verdicts and the
# not-applicable case are exercised, so the check cannot only-pass or only-fail.
# MUTATION: return POSITIONS_CORRECT unconditionally and the wrong-order test
# dies.
# ---------------------------------------------------------------------------
def test_positional_check_passes_on_the_correct_start_order_blocking():
    start = np.array(STORED_START, np.int64)
    nrow = np.array(STORED_NROW, np.int64)
    order = np.argsort(start, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(nrow[order])])
    got = R.verify_positional_attribution(start, nrow, order, bounds, N_POS)
    assert got["verdict"] == "POSITIONS_CORRECT"
    assert got["population_tiles_row_space"] is True
    assert got["n_blocks_beginning_at_the_proteins_own_start"] == 3
    assert got["fraction_rows_correctly_attributed"] == 1.0


def test_positional_check_catches_the_pre_AO6_blocking():
    """The whole point: the pre-AO6 bounds have the SAME total and the check
    still fails."""
    start = np.array(STORED_START, np.int64)
    nrow = np.array(STORED_NROW, np.int64)
    order = np.argsort(start, kind="stable")
    buggy = np.concatenate([[0], np.cumsum(nrow)])       # stored order
    assert buggy[-1] == np.cumsum(nrow[order])[-1]        # totals agree
    got = R.verify_positional_attribution(start, nrow, order, buggy, N_POS)
    assert got["verdict"] == "POSITIONS_WRONG"
    assert got["fraction_rows_correctly_attributed"] < 1.0


def test_positional_check_reports_NOT_APPLICABLE_on_a_gapped_population():
    """A gapped population is not a defect -- the construction is still correct,
    the identity just does not apply. Reporting it as a failure would be a check
    that can only fail (gotcha 25a)."""
    start = np.array([100, 0], np.int64)
    nrow = np.array([2, 3], np.int64)
    order = np.argsort(start, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(nrow[order])])
    got = R.verify_positional_attribution(start, nrow, order, bounds, 5)
    assert got["verdict"] == "NOT_APPLICABLE_GAPPED_POPULATION"
    assert got["population_tiles_row_space"] is False


def test_main_reports_the_positional_check_in_its_output(tmp_path, monkeypatch):
    cfg, lab, ev, tsv, ric, ckpts = _main_fixture(tmp_path)
    outj = tmp_path / "d.json"
    monkeypatch.setattr(sys, "argv", [
        "dfs", "--config", str(cfg), "--labels", str(lab),
        "--eval-cache", str(ev), "--clusters", str(tsv),
        "--row-index-cache", str(ric), "--ckpts", str(ckpts),
        "--seeds", "0", "--out", str(tmp_path / "d.h5"), "--out-json", str(outj)])
    R.main()
    doc = json.loads(outj.read_text())
    pa = doc["positional_attribution_check"]
    assert pa["verdict"] == "POSITIONS_CORRECT"
    assert pa["fraction_rows_correctly_attributed"] == 1.0
