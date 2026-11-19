"""Tests for G5/K1 three-axis null pooling.

Written under CLAUDE.md gotcha 13: a validation check is not evidence until
IT has been checked. Every guard in `null_pooling.py` has a test that plants
the condition the guard claims to detect and requires the guard to fail --
not merely a test that the guard passes on good input.

And under the extension recorded this session (K/4a-4b): each guard also gets
a test whose corresponding MUTATION kills it, so a test cannot pass for the
wrong reason via some other branch firing.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from src.analysis import null_pooling as np_mod
from src.analysis.null_pooling import (
    MIN_CONCEPTS_PER_STRATUM,
    MIN_FEATURES_PER_STRATUM,
    PREVALENCE_EDGES,
    PROTEIN_COUNT_EDGES,
    bin_index,
    collapse_firing_bins,
    load_concepts,
)


# --------------------------------------------------------------------- #
# bin_index
# --------------------------------------------------------------------- #

def test_bin_index_is_half_open_low_inclusive():
    edges = [0.0, 1e-5, 1e-4, 1e-3, 1.0]
    # An exact edge value belongs to the bin it OPENS, not the one it closes.
    assert bin_index(np.array([1e-5]), edges)[0] == 1
    assert bin_index(np.array([9.99e-6]), edges)[0] == 0
    assert bin_index(np.array([1e-4]), edges)[0] == 2


def test_bin_index_flags_below_range_as_minus_one_not_bin_zero():
    """The -1 sentinel is what makes the <5-protein guard able to fire. If
    below-range silently clipped to bin 0, a concept violating the G3 gate
    would be pooled into the [5,10) stratum instead of raising."""
    edges = [5, 10, 50, np.inf]
    assert bin_index(np.array([4.0]), edges)[0] == -1
    assert bin_index(np.array([5.0]), edges)[0] == 0


# --------------------------------------------------------------------- #
# collapse_firing_bins -- K1 axis 2
# --------------------------------------------------------------------- #

def _seeded(*per_bin_rates, n_seeds=3):
    """(seeds, features) array with identical per-seed composition, so tests
    exercise the per-seed path without seed-to-seed noise confusing them."""
    row = np.concatenate([np.full(n, v) for n, v in per_bin_rates])
    return np.tile(row, (n_seeds, 1))


def test_collapse_merges_sparse_TOP_bin_downward():
    edges = [0.0, 1.0, 2.0, 3.0, 4.0]
    # 30 in [0,1), 25 in [1,2), nothing above 2.
    fr = _seeded((30, 0.5), (25, 1.5))
    out = collapse_firing_bins(fr, edges, min_features=20)
    # Empty top bins merge INTO their lower neighbour: retained top is [1,4)
    # holding 25 -- NOT [0,1,2), which would drop the [2,4) range entirely.
    assert out == [0.0, 1.0, 4.0], out


def test_collapse_merges_sparse_BOTTOM_bin_upward(tmp_path=None):
    """Amendment L1. K1 pre-declared TOP-DOWN-only merging, written against
    the refuted cross-seed-mean counts. Under correct per-seed binning the
    sparse bin on the real ensemble is the BOTTOM one, which a top-down-only
    rule can never reach -- this test is the regression guard for that."""
    edges = [0.0, 1.0, 2.0, 3.0]
    # 2 in [0,1) (sparse), 40 in [1,2), 40 in [2,3)
    fr = _seeded((2, 0.5), (40, 1.5), (40, 2.5))
    out = collapse_firing_bins(fr, edges, min_features=20)
    assert out == [0.0, 2.0, 3.0], out


def test_collapse_PRESERVES_COVERAGE_of_the_outer_edges():
    """The property that makes merging safe, asserted directly because
    getting it wrong is invisible: a collapse that TRUNCATED the range
    instead of merging would leave features outside the retained edges
    unbinned, quietly changing which features enter the stratification at
    all. Both outermost edges must survive every collapse."""
    edges = [0.0, 1.0, 2.0, 3.0, 4.0]
    fr = _seeded((2, 0.5), (25, 1.5))
    out = collapse_firing_bins(fr, edges, min_features=20)
    assert out[0] == edges[0] and out[-1] == edges[-1]
    idx = bin_index(fr[0], out)
    assert (idx >= 0).all() and (idx < len(out) - 1).all()


def test_collapse_stops_as_soon_as_both_tails_qualify():
    edges = [0.0, 1.0, 2.0, 3.0]
    fr = _seeded((30, 0.5), (25, 1.5), (21, 2.5))
    assert collapse_firing_bins(fr, edges, min_features=20) == edges


def test_collapse_is_outward_in_NOT_middle_out():
    """A sparse MIDDLE bin must survive; only the tails may be merged. The
    direction is pre-declared, so a middle-out implementation is wrong even
    though it would produce better-populated cells."""
    edges = [0.0, 1.0, 2.0, 3.0]
    # middle bin [1,2) is sparse (3); both tails are fine
    fr = _seeded((30, 0.5), (3, 1.5), (40, 2.5))
    assert collapse_firing_bins(fr, edges, min_features=20) == edges


def test_collapse_uses_the_MINIMUM_over_seeds_not_the_mean():
    """Moments are formed per seed (F1.3 fits per seed and meta-analyzes),
    so a bin adequate on average but sparse in one seed is NOT adequate.
    Mutation guard: switching min->mean over seeds must break this."""
    edges = [0.0, 1.0, 2.0]
    fr = np.zeros((3, 60))
    fr[0, :] = 0.5                     # seed 0: 60 features in [0,1), 0 above
    fr[1, :30] = 0.5; fr[1, 30:] = 1.5  # seed 1: 30 / 30
    fr[2, :] = 1.5                     # seed 2: 0 in [0,1), 60 above
    # mean over seeds would say [0,1) holds 30 (>=20, passes); the MINIMUM
    # over seeds is 0, so the bin must be merged away.
    assert collapse_firing_bins(fr, edges, min_features=20) == [0.0, 2.0]


def test_collapse_never_reduces_below_a_single_bin():
    edges = [0.0, 1.0, 2.0, 3.0]
    fr = _seeded((1, 0.5))            # nothing meets the minimum anywhere
    out = collapse_firing_bins(fr, edges, min_features=20)
    assert len(out) == 2 and out == [0.0, 3.0]


def test_cross_seed_mean_collapses_onto_k_over_dict():
    """The measurement behind L1, reproduced in miniature so the claim is
    testable rather than only asserted in a docstring: averaging firing rates
    over seeds when the feature INDEX is not comparable across seeds shrinks
    the spread toward the global mean, emptying the tails."""
    rng = np.random.default_rng(0)
    # 20 seeds x 10240 features, each seed an independent heavy-tailed draw
    fr = rng.lognormal(mean=np.log(3.125e-3), sigma=1.0, size=(20, 10240))
    per_seed_tail = int((fr[0] < 3e-4).sum())
    mean_tail = int((fr.mean(axis=0) < 3e-4).sum())
    assert per_seed_tail > 0
    assert mean_tail == 0                     # the tail vanishes under averaging
    assert fr[0].std() > 3 * fr.mean(axis=0).std()


# --------------------------------------------------------------------- #
# The K1 minima are the PRE-DECLARED ones, not whatever the code happens to
# use. A silent edit to either number changes a pre-registered criterion.
# --------------------------------------------------------------------- #

def test_predeclared_minima_match_amendment_k1():
    assert MIN_CONCEPTS_PER_STRATUM == 5
    assert MIN_FEATURES_PER_STRATUM == 20
    assert np_mod.INADEQUATE_CONCEPT_SHARE == 0.10


def test_predeclared_edges_match_amendment_k1():
    assert PREVALENCE_EDGES[:4] == [0.0, 1e-5, 1e-4, 1e-3]
    assert list(PROTEIN_COUNT_EDGES[:4]) == [5, 10, 50, 500]
    # Axis 2's base grid must be corrector_scan.BIN_EDGES VERBATIM -- CLAUDE.md
    # mandates one firing-rate grid across the project.
    from src.analysis.corrector_scan import BIN_EDGES
    assert np_mod.FIRING_RATE_EDGES == list(BIN_EDGES)


# --------------------------------------------------------------------- #
# load_concepts guards -- each planted, each required to FAIL
# --------------------------------------------------------------------- #

def _write_store(tmp_path, concepts, n_rows, residues, proteins,
                 manifest_residues=None):
    h5 = tmp_path / "labels.h5"
    js = tmp_path / "labels.json"
    with h5py.File(h5, "w") as hf:
        for c in concepts:
            hf.create_dataset(f"concepts/{c}/all",
                              data=np.arange(residues[c], dtype=np.int64))
        hf.attrs["n_rows"] = n_rows
    mr = manifest_residues or residues
    js.write_text(json.dumps({"retention_floor_diagnostics": {
        c: {"n_residues": mr[c], "n_proteins": proteins[c]} for c in concepts}}))
    return h5, js


def test_load_concepts_excludes_pooled_parents(tmp_path):
    concepts = ["DOMAIN", "DOMAIN:SH2", "DOMAIN:Kinase", "ACT_SITE"]
    res = {c: 100 for c in concepts}
    prot = {c: 20 for c in concepts}
    h5, js = _write_store(tmp_path, concepts, 1000, res, prot)
    pool, parents, library, n_rows, prev, pc, cc = load_concepts(h5, js)
    assert parents == ["DOMAIN"]                 # has children -> parent
    assert "ACT_SITE" in pool                    # childless bare -> stays
    assert set(pool) == {"ACT_SITE", "DOMAIN:Kinase", "DOMAIN:SH2"}
    assert n_rows == 1000
    np.testing.assert_allclose(prev, [0.1, 0.1, 0.1])
    # n_clusters absent from the manifest => -1 sentinel, NEVER 0. Binning a
    # sentinel as a real count would stratify concepts on a value that means
    # "not measured" (amendment N2).
    np.testing.assert_array_equal(cc, [-1.0, -1.0, -1.0])


def test_manifest_residue_mismatch_is_DETECTED(tmp_path):
    """GOTCHA 13. Plant a manifest whose n_residues disagrees with the stored
    row sets and require the guard to fire. Prevalence is computed FROM the
    manifest, so a silent disagreement puts the whole prevalence axis -- and
    therefore every stratum assignment -- on wrong numbers."""
    concepts = ["ACT_SITE", "BINDING:ATP"]
    res = {c: 100 for c in concepts}
    prot = {c: 20 for c in concepts}
    bad = {"ACT_SITE": 100, "BINDING:ATP": 999}      # <-- planted
    h5, js = _write_store(tmp_path, concepts, 1000, res, prot,
                          manifest_residues=bad)
    with pytest.raises(SystemExit, match="disagrees with the stored row sets"):
        load_concepts(h5, js)


def test_manifest_residue_agreement_passes(tmp_path):
    """The other half of gotcha 13: the guard must also NOT fire on good
    input, or 'it detects the condition' is untested in the negative."""
    concepts = ["ACT_SITE", "BINDING:ATP"]
    res = {c: 100 for c in concepts}
    prot = {c: 20 for c in concepts}
    h5, js = _write_store(tmp_path, concepts, 1000, res, prot)
    pool, *_ = load_concepts(h5, js)
    assert len(pool) == 2


def test_concept_absent_from_manifest_is_DETECTED(tmp_path):
    concepts = ["ACT_SITE", "BINDING:ATP"]
    res = {c: 100 for c in concepts}
    prot = {c: 20 for c in concepts}
    h5, js = _write_store(tmp_path, concepts, 1000, res, prot)
    manifest = json.loads(js.read_text())
    del manifest["retention_floor_diagnostics"]["BINDING:ATP"]
    js.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="absent from the manifest"):
        load_concepts(h5, js)


def test_probe_only_covers_first_25_so_mismatch_beyond_is_missed(tmp_path):
    """Documents a real LIMIT of the cross-check rather than asserting a
    property it does not have: the probe reads the first 25 concepts only, so
    a manifest corrupted at concept 300 is NOT caught. Recorded so nobody
    later reads the guard as exhaustive -- the same 'a check is not evidence
    until checked' discipline applied to the check's own scope."""
    concepts = [f"C{i:04d}" for i in range(40)]
    res = {c: 100 for c in concepts}
    prot = {c: 20 for c in concepts}
    bad = dict(res)
    bad["C0039"] = 12345                       # beyond the 25-concept probe
    h5, js = _write_store(tmp_path, concepts, 100000, res, prot,
                          manifest_residues=bad)
    pool, *_ = load_concepts(h5, js)           # does NOT raise
    assert len(pool) == 40


# --------------------------------------------------------------------- #
# Adequacy arithmetic
# --------------------------------------------------------------------- #

def test_nominal_draws_overstate_a_one_concept_stratum():
    """The whole reason K1's minimum is on structural axes. One concept and
    one feature still reports 400 nominal draws at 20 seeds x 20 shifts --
    which is why nominal count is reported and never used as a criterion."""
    n_c, n_f, n_shifts, n_seeds = 1, 1, 20, 20
    assert n_c * n_f * n_shifts * n_seeds == 400
    assert n_c < MIN_CONCEPTS_PER_STRATUM       # yet it fails the real bar


def test_cluster_count_read_when_present(tmp_path):
    """N2 independence axis comes through when the manifest carries it."""
    concepts = ["ACT_SITE", "BINDING:ATP"]
    res = {c: 100 for c in concepts}
    prot = {c: 20 for c in concepts}
    h5, js = _write_store(tmp_path, concepts, 1000, res, prot)
    m = json.loads(js.read_text())
    m["retention_floor_diagnostics"]["ACT_SITE"]["n_clusters"] = 7
    m["retention_floor_diagnostics"]["BINDING:ATP"]["n_clusters"] = 1
    js.write_text(json.dumps(m))
    *_, cc = load_concepts(h5, js)
    np.testing.assert_array_equal(cc, [7.0, 1.0])


def test_cluster_edges_isolate_single_and_two_cluster_concepts():
    """N2's argument turns on these two bins existing separately: {1} is the
    187 single-cluster concepts, {2} is where BINDING:heme a3 lives. If they
    were merged, stratification could not give them distinct moments."""
    got = bin_index(np.array([1.0, 2.0, 3.0, 9.0, 10.0, 3063.0]),
                    np_mod.CLUSTER_COUNT_EDGES)
    assert got.tolist() == [0, 1, 2, 2, 3, 3]


def test_cluster_count_sentinel_is_below_the_first_edge():
    """-1 must bin to the -1 out-of-scope sentinel, not into the {1} bin."""
    assert bin_index(np.array([-1.0]), np_mod.CLUSTER_COUNT_EDGES)[0] == -1


# ------------------------------------------------- CLI smoke test (main) --
# WHY THIS EXISTS. Every test above calls `load_concepts` / `bin_index` /
# `collapse_firing_bins` DIRECTLY, so when `load_concepts` grew a seventh
# return value and `main()`'s unpacking was not updated, all 242 tests still
# passed and the failure only appeared on the cluster. A unit suite that never
# executes the entry point cannot catch an entry-point bug.

def test_main_runs_end_to_end(tmp_path, monkeypatch, capsys):
    """Exercise main() on a tiny synthetic store + null, both with and
    without the cluster axis, so an arity or wiring change breaks a test
    rather than a job."""
    import sys
    concepts = ["ACT_SITE", "BINDING", "BINDING:ATP", "DOMAIN", "DOMAIN:SH2",
                "MOTIF:X", "REGION:Y", "REPEAT:Z"]
    res = {c: 500 + 10 * i for i, c in enumerate(concepts)}
    prot = {c: 5 + i for i, c in enumerate(concepts)}
    h5, js = _write_store(tmp_path, concepts, 100000, res, prot)
    m = json.loads(js.read_text())
    for i, c in enumerate(concepts):
        m["retention_floor_diagnostics"][c]["n_clusters"] = 1 + i
    js.write_text(json.dumps(m))

    null = tmp_path / "null.h5"
    rng = np.random.default_rng(0)
    with h5py.File(null, "w") as f:
        f.create_dataset("fire_rate",
                         data=rng.lognormal(np.log(3.125e-3), 1.0, size=(4, 512)))
        f.attrs["n_shifts"] = 5
        f.attrs["argmax_pool_n"] = 6
        f.attrs["truncation_policy"] = "untruncated"
    out = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", [
        "null_pooling", "--labels", str(h5), "--labels-json", str(js),
        "--null", str(null), "--out", str(out)])
    np_mod.main()
    payload = json.loads(out.read_text())
    assert payload["has_cluster_axis"] is True
    assert payload["verdict"] in ("ADEQUATE", "INADEQUATE")
    assert len(payload["scheme_comparison_T3c"]) == 3


def test_main_runs_without_cluster_axis(tmp_path, monkeypatch):
    """The -1 sentinel path must SKIP the axis, not crash and not bin it."""
    import sys
    concepts = ["ACT_SITE", "BINDING:ATP", "DOMAIN:SH2"]
    res = {c: 500 for c in concepts}
    prot = {c: 8 for c in concepts}
    h5, js = _write_store(tmp_path, concepts, 100000, res, prot)
    null = tmp_path / "null.h5"
    with h5py.File(null, "w") as f:
        f.create_dataset("fire_rate", data=np.full((3, 256), 2e-3))
        f.attrs["n_shifts"] = 4
    out = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", [
        "null_pooling", "--labels", str(h5), "--labels-json", str(js),
        "--null", str(null), "--out", str(out)])
    np_mod.main()
    payload = json.loads(out.read_text())
    assert payload["has_cluster_axis"] is False
    assert payload["scheme_comparison_T3c"] == []
