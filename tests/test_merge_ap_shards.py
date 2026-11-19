"""Tests for `merge_ap_shards` (PREREGISTRATION amendment AV10d).

THE FIXTURE DISCRIMINATION REQUIREMENT, stated because it is the reason these
tests are shaped the way they are. Gotcha 16: a fixture that cannot exhibit the
disputed condition cannot test for it. Gotcha 29's corollary is sharper still --
three of nine alignment mutations killed NOTHING against a fixture whose arms
shared one seed order, so **a test for an alignment needs as many distinct
orders as there are arms.** The shard fixtures here are therefore built in
NUMERIC order while the reference is LEXICOGRAPHIC, which is exactly the split
that mis-pairs 18 of 20 rows in the real artifacts.

And the values are chosen so that a wrong ordering is DETECTABLE IN THE DATA,
not only in the labels: each shard's `ap` block is filled with its own seed
number, so a mis-ordered merge produces a numerically wrong array whose shape,
total and rank correlation are all unchanged. A check on a total can never
detect a permutation (gotcha 27).
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from src.analysis.merge_ap_shards import (
    check_stratum_comparability,
    compare_identities,
    concept_axes_identical,
    main,
    merge_shards,
    resolve_shard_order,
    verify_positional_seed_order,
)

CONCEPTS = ["ACT_SITE", "BINDING:Zn(2+)", "DOMAIN:Protein kinase"]
N_FEAT = 8
BASE_ATTRS = {
    "truncation_policy": "untruncated",
    "n_concepts": 3,
    "library_sha": "f67485d9",
    "retention_floor_mode": "interplm",
    "retention_min_residues": 1500,
    "retention_min_instances": 10,
    "min_protein_count": 5,
    "scored_row_set": "label_store",
    "n_rows": 15972325,
    "argmax_pool_n": 3,
    "ap_impl": "segmented/v1",
    "strata": '["experimental"]',
    "negatives": "unannotated",
    # amendment AW: the writer marker that distinguishes a shard whose
    # `n_concepts`/`argmax_pool_n` are library-scoped (post-fix) from one
    # whose are stratum-scoped (pre-fix, the bug). Present by default here
    # because every OTHER test in this file merges without `labels=` and
    # never exercises `check_library_scoped_identity` at all -- only the
    # tests that pass `labels=` below need to control this deliberately.
    "n_concepts_scored": 3,
    "argmax_pool_scored_n": 3,
}

# LEXICOGRAPHIC, as every real AP artifact is written (glob order).
LEX_SEEDS = ["seed_0", "seed_1", "seed_10", "seed_11", "seed_2", "seed_3"]
# NUMERIC, the order the array tasks produce shards in.
NUM_SEEDS = ["seed_0", "seed_1", "seed_2", "seed_3", "seed_10", "seed_11"]


def _write_shard(path, seed_label, concepts=CONCEPTS, attrs=None, ap_fill=None):
    n = int(seed_label.rsplit("_", 1)[-1])
    a = dict(BASE_ATTRS if attrs is None else attrs)
    with h5py.File(path, "w") as hf:
        hf.create_dataset("fire_counts", data=np.full((1, N_FEAT), n, dtype=np.int64))
        hf.create_dataset("seeds", data=np.array([seed_label],
                                                 dtype=h5py.string_dtype()))
        g = hf.create_group("experimental")
        fill = n if ap_fill is None else ap_fill
        g.create_dataset("ap", data=np.full((1, N_FEAT, len(concepts)),
                                            fill / 100.0, dtype=np.float32))
        g.create_dataset("concepts", data=np.array(concepts,
                                                   dtype=h5py.string_dtype()))
        g.create_dataset("prevalence", data=np.linspace(0.001, 0.01, len(concepts)))
        g.create_dataset("in_argmax_pool", data=np.ones(len(concepts), dtype=bool))
        for k, v in a.items():
            hf.attrs[k] = v
    return path


def _write_reference(path, seeds=LEX_SEEDS):
    with h5py.File(path, "w") as hf:
        hf.create_dataset("seeds", data=np.array(seeds, dtype=h5py.string_dtype()))
        for k, v in BASE_ATTRS.items():
            hf.attrs[k] = v
        hf.attrs["strata"] = '["pooled"]'
        hf.attrs["negatives"] = "all_rows"
        hf.attrs["ap_impl"] = "sparse_ap_index.average_precision_from_touches/v1"
        hf.attrs["git_sha"] = "9112a0d0"          # provenance: differs by design
    return path


@pytest.fixture
def shards(tmp_path):
    return [_write_shard(tmp_path / f"shard_{s}.h5", s) for s in NUM_SEEDS]


@pytest.fixture
def reference(tmp_path):
    return _write_reference(tmp_path / "reference.h5")


# --------------------------------------------------------------------------
# resolve_shard_order / positional verification
# --------------------------------------------------------------------------

def test_order_maps_numeric_shards_onto_a_lexicographic_reference():
    order = resolve_shard_order(NUM_SEEDS, LEX_SEEDS)
    assert [NUM_SEEDS[i] for i in order] == LEX_SEEDS
    # The two orders genuinely differ, or this test proves nothing.
    assert NUM_SEEDS != LEX_SEEDS
    assert order != list(range(len(NUM_SEEDS)))


def test_order_is_spelling_insensitive():
    assert resolve_shard_order(["7", "3"], ["seed_3", "seed_7"]) == [1, 0]


def test_missing_seed_raises_rather_than_silently_shortening():
    with pytest.raises(ValueError, match="seed sets differ"):
        resolve_shard_order(NUM_SEEDS[:-1], LEX_SEEDS)


def test_duplicate_seed_labels_raise():
    with pytest.raises(ValueError, match="duplicate"):
        resolve_shard_order(["seed_1", "seed_1"], ["seed_1", "seed_1"])


def test_positional_check_sees_a_permutation_a_total_cannot():
    # Same multiset, same length, same "total" -- only the POSITIONS differ.
    swapped = LEX_SEEDS[:2] + [LEX_SEEDS[3], LEX_SEEDS[2]] + LEX_SEEDS[4:]
    assert sorted(swapped) == sorted(LEX_SEEDS)
    bad = verify_positional_seed_order(swapped, LEX_SEEDS)
    assert bad["verdict"] == "POSITIONS_WRONG"
    assert bad["n_rows_correct"] == len(LEX_SEEDS) - 2
    good = verify_positional_seed_order(LEX_SEEDS, LEX_SEEDS)
    assert good["verdict"] == "POSITIONS_OK"
    assert good["n_rows_correct"] == len(LEX_SEEDS)


def test_positional_check_reports_length_mismatch_distinctly():
    r = verify_positional_seed_order(LEX_SEEDS[:3], LEX_SEEDS)
    assert r["verdict"] == "LENGTH_MISMATCH"


# --------------------------------------------------------------------------
# identity comparison -- ABSENT is never AGREEMENT
# --------------------------------------------------------------------------

def test_absent_key_is_reported_absent_not_matching():
    a = {"library_sha": "x", "n_concepts": 3}
    b = {"library_sha": "x"}
    r = compare_identities([a, b], ["a", "b"], ("library_sha", "n_concepts"))
    assert r["agree"] is False
    assert r["disagreements"] == []
    assert r["absent"][0]["key"] == "n_concepts"
    assert r["absent"][0]["missing_from"] == ["b"]


def test_disagreement_is_reported_with_both_values():
    r = compare_identities([{"n_concepts": 3}, {"n_concepts": 4}],
                           ["a", "b"], ("n_concepts",))
    assert r["agree"] is False
    assert r["disagreements"][0]["values"] == {"a": 3, "b": 4}


def test_numpy_scalars_compare_equal_to_python_ints():
    r = compare_identities([{"n_concepts": np.int64(3)}, {"n_concepts": 3}],
                           ["a", "b"], ("n_concepts",))
    assert r["agree"] is True


# --------------------------------------------------------------------------
# concept axis
# --------------------------------------------------------------------------

def test_concept_axis_difference_is_caught():
    r = concept_axes_identical([CONCEPTS, CONCEPTS, CONCEPTS[::-1]])
    assert r["verdict"] == "DIFFER"
    assert r["mismatched_shard_indices"] == [2]
    assert concept_axes_identical([CONCEPTS, CONCEPTS])["verdict"] == "IDENTICAL"


# --------------------------------------------------------------------------
# stratum comparability
# --------------------------------------------------------------------------

def test_comparability_passes_when_only_masking_differs():
    exp = dict(BASE_ATTRS, git_sha="aaa")
    ref = dict(BASE_ATTRS, strata='["pooled"]', negatives="all_rows",
               ap_impl="sparse/v1", git_sha="bbb")
    r = check_stratum_comparability(exp, ref, LEX_SEEDS, LEX_SEEDS)
    assert r["verdict"] == "COMPARABLE"
    assert r["unexpected_differences"] == []


def test_comparability_fails_when_a_shared_identity_key_disagrees():
    """The ENUMERATED half: a key in COMPARABILITY_KEYS differs."""
    exp = dict(BASE_ATTRS)
    ref = dict(BASE_ATTRS, strata='["pooled"]', negatives="all_rows",
               n_rows=17739363)                      # the retired wide screen's
    r = check_stratum_comparability(exp, ref, LEX_SEEDS, LEX_SEEDS)
    assert r["shared_keys"]["agree"] is False
    assert r["verdict"] == "NOT_COMPARABLE"


def test_comparability_fails_on_a_key_the_enumerated_list_has_never_SEEN():
    """The CLOSED half, and the fixture is built so ONLY it can fire.

    A default-deny check is about keys nobody enumerated, so the fixture uses
    an attribute outside `COMPARABILITY_KEYS`, outside the declared expected
    differences, and outside the provenance exemption. Using `n_rows` here
    instead — as the first draft did — made the test pass through the SHARED-KEY
    branch and survive its own mutation: a test passing for the wrong reason
    (gotcha 13 extension 3), found by the mutation runner rather than by
    re-reading the code.
    """
    exp = dict(BASE_ATTRS, evidence_tier_map_sha="aaaa")
    ref = dict(BASE_ATTRS, strata='["pooled"]', negatives="all_rows",
               evidence_tier_map_sha="bbbb")
    r = check_stratum_comparability(exp, ref, LEX_SEEDS, LEX_SEEDS)
    assert r["shared_keys"]["agree"] is True          # the enumerated half is fine
    assert [u["key"] for u in r["unexpected_differences"]] == [
        "evidence_tier_map_sha"]
    assert r["verdict"] == "NOT_COMPARABLE"


def test_comparability_fails_on_a_seed_axis_permutation_alone():
    """Everything agrees EXCEPT the row order -- the case no identity key can
    see, because all of them are invariant under a seed permutation."""
    exp = dict(BASE_ATTRS)
    ref = dict(BASE_ATTRS, strata='["pooled"]', negatives="all_rows")
    r = check_stratum_comparability(exp, ref, NUM_SEEDS, LEX_SEEDS)
    assert r["shared_keys"]["agree"] is True          # identity agrees...
    assert r["unexpected_differences"] == []          # ...and so does everything
    assert r["seed_axis"]["verdict"] == "POSITIONS_WRONG"   # except the ORDER
    assert r["verdict"] == "NOT_COMPARABLE"


def test_provenance_keys_do_not_count_as_unexpected_differences():
    """gotcha 25(a): a bit-identity criterion on a per-run provenance key can
    ONLY fail, so it must be exempted -- by full name, deliberately."""
    exp = dict(BASE_ATTRS, git_sha="aaa", git_dirty=False)
    ref = dict(BASE_ATTRS, strata='["pooled"]', negatives="all_rows",
               git_sha="bbb", git_dirty=False)
    assert check_stratum_comparability(exp, ref, LEX_SEEDS,
                                       LEX_SEEDS)["verdict"] == "COMPARABLE"


# --------------------------------------------------------------------------
# end-to-end merge -- and the DATA, not only the labels, must land right
# --------------------------------------------------------------------------

def test_merge_orders_rows_by_the_reference_and_the_values_follow(
        shards, reference, tmp_path):
    out = tmp_path / "merged.h5"
    rep = merge_shards(shards, reference, "experimental", out)
    assert rep["stratum_comparability_vs_reference"]["verdict"] == "COMPARABLE"
    assert rep["shape"] == [len(NUM_SEEDS), N_FEAT, len(CONCEPTS)]
    with h5py.File(out, "r") as hf:
        labels = [s.decode() for s in hf["seeds"][:]]
        ap = hf["experimental"]["ap"][:]
        fire = hf["fire_counts"][:]
    assert labels == LEX_SEEDS
    # THE POINT: each row's VALUE identifies which seed it came from, so a
    # correct-shaped, correct-total, wrongly-ordered merge fails here.
    expected = [int(s.rsplit("_", 1)[-1]) for s in LEX_SEEDS]
    assert [round(float(ap[i, 0, 0]) * 100) for i in range(len(labels))] == expected
    assert fire[:, 0].tolist() == expected


def _write_labels(path, library, declared=None):
    with h5py.File(path, "w") as hf:
        g = hf.create_group("concepts")
        for c in library:
            g.create_group(c).create_dataset("all", data=np.arange(3))
        hf.attrs["n_concepts"] = len(library) if declared is None else declared
    return path


def test_merge_verifies_post_fix_shards_against_the_library(
        shards, reference, tmp_path):
    """gotcha 16: `n_concepts` is always the LIBRARY, `argmax_pool_n` the POOL.

    `feature_ap.py` writes these library-scoped AT THE SOURCE now (amendment
    AW), so the merge no longer derives or corrects them -- it only VERIFIES
    the shard already agrees, then copies the shard's own attrs through
    unchanged. The fixture's library is strictly LARGER than the scored set
    and contains a POOLED PARENT (`BINDING`, whose child `BINDING:Zn(2+)` is
    present), so the library count, the library pool and the scored count are
    three DIFFERENT numbers -- a fixture where they coincided could not
    exhibit the condition (gotcha 16's own lesson), and they coincide in
    every pooled run, which is why the original bug never surfaced there.
    """
    library = CONCEPTS + ["BINDING", "LIPID", "TRANSMEM"]
    labels = _write_labels(tmp_path / "labels.h5", library)
    post_fix = dict(BASE_ATTRS, n_concepts=6, argmax_pool_n=5,
                    n_concepts_scored=3, argmax_pool_scored_n=3)
    post_fix_shards = [_write_shard(tmp_path / f"pf_{s}.h5", s, attrs=post_fix)
                       for s in NUM_SEEDS]
    out = tmp_path / "merged.h5"
    rep = merge_shards(post_fix_shards, reference, "experimental", out,
                       labels=labels)
    chk = rep["library_scoped_identity_check"]
    assert chk["n_concepts_library"] == 6
    assert chk["argmax_pool_n_library"] == 5       # library minus the pooled parent
    assert chk["n_concepts_scored_as_written"] == 3
    assert len({chk["n_concepts_library"], chk["argmax_pool_n_library"],
                chk["n_concepts_scored_as_written"]}) == 3   # three distinct numbers
    assert chk["n_concepts_as_written"] == 6
    with h5py.File(out, "r") as hf:
        assert int(hf.attrs["n_concepts"]) == 6
        assert int(hf.attrs["argmax_pool_n"]) == 5
        assert int(hf.attrs["n_concepts_scored"]) == 3


def test_merge_refuses_shards_predating_the_writer_fix(
        shards, reference, tmp_path):
    """A shard with no `n_concepts_scored` attr carries the PRE-AW format,
    where `n_concepts`/`argmax_pool_n` are stratum-scoped rather than
    library-scoped. Merging it under the canonical keys without noticing
    would silently misdescribe it -- exactly the defect this whole mechanism
    exists to prevent, one level up (gotcha 13 extension 4)."""
    library = CONCEPTS + ["BINDING", "LIPID", "TRANSMEM"]
    labels = _write_labels(tmp_path / "labels.h5", library)
    pre_fix = {k: v for k, v in BASE_ATTRS.items()
              if k not in ("n_concepts_scored", "argmax_pool_scored_n")}
    pre_fix_shards = [_write_shard(tmp_path / f"prefix_{s}.h5", s, attrs=pre_fix)
                      for s in NUM_SEEDS]
    with pytest.raises(SystemExit, match="predates the AW writer fix"):
        merge_shards(pre_fix_shards, reference, "experimental",
                     tmp_path / "m.h5", labels=labels)


def test_merge_refuses_when_writer_regresses_despite_post_fix_format(
        shards, reference, tmp_path):
    """A shard that carries `n_concepts_scored` (the post-fix marker) but
    whose `n_concepts` still equals the stratum-scoped count rather than the
    library -- i.e. the writer regressed to writing the wrong value while
    still emitting the new field. Must be caught, not silently merged."""
    library = CONCEPTS + ["BINDING", "LIPID", "TRANSMEM"]
    labels = _write_labels(tmp_path / "labels.h5", library)
    regressed = dict(BASE_ATTRS, n_concepts=3, argmax_pool_n=3,
                     n_concepts_scored=3, argmax_pool_scored_n=3)
    regressed_shards = [_write_shard(tmp_path / f"reg_{s}.h5", s, attrs=regressed)
                        for s in NUM_SEEDS]
    with pytest.raises(SystemExit, match="writer has regressed"):
        merge_shards(regressed_shards, reference, "experimental",
                     tmp_path / "m.h5", labels=labels)


def test_merge_refuses_shards_scoring_concepts_outside_the_library(
        shards, reference, tmp_path):
    labels = _write_labels(tmp_path / "labels.h5", ["LIPID", "TRANSMEM"])
    with pytest.raises(SystemExit, match="different store"):
        merge_shards(shards, reference, "experimental", tmp_path / "m.h5",
                     labels=labels)


def test_merge_refuses_shards_that_disagree_on_identity(tmp_path, reference):
    good = [_write_shard(tmp_path / f"s{s}.h5", s) for s in NUM_SEEDS[:-1]]
    bad = _write_shard(tmp_path / "sbad.h5", NUM_SEEDS[-1],
                       attrs=dict(BASE_ATTRS, library_sha="deadbeef"))
    with pytest.raises(SystemExit, match="do not describe one experiment"):
        merge_shards(good + [bad], reference, "experimental",
                     tmp_path / "m.h5")


def test_merge_refuses_a_multi_seed_shard(tmp_path, reference):
    p = tmp_path / "multi.h5"
    _write_shard(p, "seed_0")
    with h5py.File(p, "a") as hf:
        del hf["seeds"]
        hf.create_dataset("seeds", data=np.array(["seed_0", "seed_1"],
                                                 dtype=h5py.string_dtype()))
    with pytest.raises(ValueError, match="expects one"):
        merge_shards([p], reference, "experimental", tmp_path / "m.h5")


def test_merge_refuses_a_missing_shard(tmp_path, reference):
    part = [_write_shard(tmp_path / f"p{s}.h5", s) for s in NUM_SEEDS[:-1]]
    with pytest.raises(ValueError, match="seed sets differ"):
        merge_shards(part, reference, "experimental", tmp_path / "m.h5")


# --------------------------------------------------------------------------
# main() -- the entry point gets its own test (gotcha 13 extension 4)
# --------------------------------------------------------------------------

def test_main_returns_zero_on_a_clean_merge(shards, reference, tmp_path):
    out, rep = tmp_path / "m.h5", tmp_path / "r.json"
    assert main(["--shards", *map(str, shards), "--reference", str(reference),
                 "--stratum", "experimental", "--out", str(out),
                 "--report", str(rep)]) == 0
    assert out.exists() and rep.exists()


def test_main_returns_nonzero_when_not_comparable(shards, tmp_path):
    """A validator must fail DETECTABLY TO ITS CONSUMER: printing the verdict
    and exiting 0 would let an `afterok` chain read a mismatch as success."""
    ref = tmp_path / "badref.h5"
    _write_reference(ref)
    with h5py.File(ref, "a") as hf:
        hf.attrs["n_rows"] = 17739363          # the retired wide screen's count
    assert main(["--shards", *map(str, shards), "--reference", str(ref),
                 "--stratum", "experimental",
                 "--out", str(tmp_path / "m.h5")]) == 1


# --------------------------------------------------------------------------
# per-concept vectors -- prevalence scales every gain downstream
# --------------------------------------------------------------------------

def test_prevalence_disagreement_between_shards_is_fatal(tmp_path, reference):
    """Same shapes, same names, same counts, same identity -- only the VALUES
    of `prevalence` differ. Nothing but a positional value check can see it,
    and the merged artifact copies the vector from one shard, so a silent
    disagreement rescales every `G = (AP - p)/(1 - p)` downstream."""
    good = [_write_shard(tmp_path / f"g{s}.h5", s) for s in NUM_SEEDS[:-1]]
    bad = _write_shard(tmp_path / "gbad.h5", NUM_SEEDS[-1])
    with h5py.File(bad, "a") as hf:
        p = hf["experimental"]["prevalence"][:]
        p[1] = p[1] * 2.0                      # one concept, one value
        del hf["experimental"]["prevalence"]
        hf["experimental"].create_dataset("prevalence", data=p)
    with pytest.raises(SystemExit, match="per-concept vector"):
        merge_shards(good + [bad], reference, "experimental", tmp_path / "m.h5")


def test_identical_prevalence_across_shards_passes_and_is_reported(
        shards, reference, tmp_path):
    rep = merge_shards(shards, reference, "experimental", tmp_path / "m.h5")
    assert rep["per_concept_vectors"]["prevalence"]["verdict"] == "IDENTICAL"
    assert rep["per_concept_vectors"]["in_argmax_pool"]["verdict"] == "IDENTICAL"
    # a dataset no shard writes is ABSENT_FROM_ALL, never "identical"
    assert rep["per_concept_vectors"]["n_positives"]["verdict"] == "ABSENT_FROM_ALL"


def test_a_vector_present_on_only_SOME_shards_is_not_agreement(
        tmp_path, reference):
    """gotcha 15 / I3: absent from some must never read as matching."""
    good = [_write_shard(tmp_path / f"h{s}.h5", s) for s in NUM_SEEDS[:-1]]
    partial = _write_shard(tmp_path / "hpart.h5", NUM_SEEDS[-1])
    with h5py.File(partial, "a") as hf:
        del hf["experimental"]["prevalence"]
    with pytest.raises(SystemExit, match="per-concept vector"):
        merge_shards(good + [partial], reference, "experimental",
                     tmp_path / "m.h5")


# ---- amendment CT, §CT11.7-R: a shared-set stratum against the pooled reference

def test_population_difference_refused_unless_declared():
    exp = dict(BASE_ATTRS, scored_row_set="shared_experimental", n_rows=1471542)
    ref = dict(BASE_ATTRS, strata='["pooled"]', negatives="all_rows",
               ap_impl="sparse/v1")
    assert check_stratum_comparability(exp, ref, LEX_SEEDS,
                                       LEX_SEEDS)["verdict"] == "NOT_COMPARABLE"
    r = check_stratum_comparability(exp, ref, LEX_SEEDS, LEX_SEEDS,
                                    population_may_differ=True)
    assert r["verdict"] == "COMPARABLE" and r["population_may_differ"] is True


def test_population_flag_does_not_excuse_any_other_key():
    exp = dict(BASE_ATTRS, scored_row_set="shared_experimental", n_rows=1471542,
               library_sha="somethingelse")
    ref = dict(BASE_ATTRS, strata='["pooled"]', negatives="all_rows")
    r = check_stratum_comparability(exp, ref, LEX_SEEDS, LEX_SEEDS,
                                    population_may_differ=True)
    assert r["verdict"] == "NOT_COMPARABLE"
