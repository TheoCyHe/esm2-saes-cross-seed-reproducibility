"""The AS seed-label migration: it must LABEL correctly, and REFUSE otherwise.

WRITING A WRONG LABEL IS STRICTLY WORSE THAN WRITING NONE, because from then on
every reader trusts it -- gotcha 15's shape applied to the fix rather than to
the defect. So every path that could label on a guess is tested for refusal,
and the fixtures are built so each refusal is REACHABLE.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from migrations.as_add_seed_labels import inherit, migrate
from src.analysis.seed_axis import stored_seed_labels

PERM = [2, 0, 3, 1]


def _null(tmp, labels=PERM, n_shifts=6, n_feat=50, seed=0):
    rng = np.random.default_rng(seed)
    ng = rng.lognormal(size=(len(labels), n_shifts, n_feat))
    p = tmp / "nr.h5"
    with h5py.File(p, "w") as hf:
        hf.create_dataset("null_g_b", data=ng)
        hf.create_dataset("seeds", data=[f"seed_{x}".encode() for x in labels],
                          dtype=h5py.string_dtype())
    return p, ng


def _gtilde(tmp, ng, name="gt.h5", scale=2.0, argmax=None, labels=None):
    p = tmp / name
    with h5py.File(p, "w") as hf:
        hf.create_dataset("gtilde_null", data=scale * ng)
        if argmax is not None:
            hf.create_dataset("argmax_concept", data=argmax)
        if labels is not None:
            hf.create_dataset("seeds", data=[str(x).encode() for x in labels],
                              dtype=h5py.string_dtype())
    return p


def test_labels_are_recovered_from_the_data_and_read_back(tmp_path):
    nr, ng = _null(tmp_path)
    gt = _gtilde(tmp_path, ng)
    row = migrate(gt, nr, dry_run=False, schema_version="AS:test")
    assert row["action"] == "LABELLED"
    assert row["labels"] == [str(x) for x in PERM]
    assert row["read_back"] == row["labels"]      # re-read, not trusted
    assert stored_seed_labels(gt) == [str(x) for x in PERM]
    with h5py.File(gt, "r") as hf:
        assert hf.attrs["attr_schema_version"] == "AS:test"
        prov = json.loads(hf.attrs["seed_label_provenance"])
        assert prov["recovered_by"].endswith("resolve_unlabelled_seed_order")


def test_a_dry_run_writes_nothing(tmp_path):
    nr, ng = _null(tmp_path)
    gt = _gtilde(tmp_path, ng)
    row = migrate(gt, nr, dry_run=True, schema_version="AS:test")
    assert row["action"] == "WOULD_LABEL"
    assert stored_seed_labels(gt) is None


def test_an_already_labelled_artifact_is_left_alone(tmp_path):
    nr, ng = _null(tmp_path)
    gt = _gtilde(tmp_path, ng, labels=PERM)
    row = migrate(gt, nr, dry_run=False, schema_version="AS:test")
    assert row["action"] == "ALREADY_LABELLED"


def test_an_unresolvable_artifact_is_skipped_not_guessed(tmp_path):
    """The refusal must be REACHABLE, or the migration could label on noise."""
    nr, ng = _null(tmp_path)
    rng = np.random.default_rng(9)
    gt = tmp_path / "gt.h5"
    with h5py.File(gt, "w") as hf:
        hf.create_dataset("gtilde_null", data=rng.lognormal(size=ng.shape))
    row = migrate(gt, nr, dry_run=False, schema_version="AS:test")
    assert row["action"] == "UNRESOLVABLE_SKIPPED"
    assert stored_seed_labels(gt) is None


def test_an_artifact_without_gtilde_null_is_skipped(tmp_path):
    """MEASURED ON THE REAL DATA: degenerate_argmax_flag.h5 came back
    NO_GTILDE_NULL_SKIPPED, which is the honest answer and why `inherit`
    exists."""
    nr, _ = _null(tmp_path)
    p = tmp_path / "flag.h5"
    with h5py.File(p, "w") as hf:
        hf.create_dataset("degenerate_argmax", data=np.zeros((4, 5), bool))
    row = migrate(p, nr, dry_run=False, schema_version="AS:test")
    assert row["action"] == "NO_GTILDE_NULL_SKIPPED"
    assert stored_seed_labels(p) is None


def test_absent_target_is_reported_not_created(tmp_path):
    nr, _ = _null(tmp_path)
    row = migrate(tmp_path / "nope.h5", nr, dry_run=False,
                  schema_version="AS:test")
    assert row["action"] == "ABSENT"
    assert not (tmp_path / "nope.h5").exists()


# ---------------------------------------------------------------------------
# inherit -- verified elementwise, because a total survives a permutation
# ---------------------------------------------------------------------------

def test_inherit_labels_a_derived_artifact_after_an_exact_check(tmp_path):
    nr, ng = _null(tmp_path)
    am = np.arange(4 * 5).reshape(4, 5).astype(np.int32)
    gt = _gtilde(tmp_path, ng, argmax=am, labels=PERM)
    flag = tmp_path / "flag.h5"
    with h5py.File(flag, "w") as hf:
        hf.create_dataset("degenerate_argmax", data=np.zeros((4, 5), bool))
        hf.create_dataset("argmax_concept", data=am)
    row = inherit(flag, gt, dry_run=False, schema_version="AS:test")
    assert row["action"] == "LABELLED"
    assert row["labels"] == [str(x) for x in PERM]
    assert "elementwise" in row["verified_by"]
    assert stored_seed_labels(flag) == [str(x) for x in PERM]


def test_inherit_refuses_on_a_single_differing_cell(tmp_path):
    """ONE cell is enough. A shape or a total agrees under a permutation, so
    neither is the check (gotcha 27)."""
    nr, ng = _null(tmp_path)
    am = np.arange(4 * 5).reshape(4, 5).astype(np.int32)
    gt = _gtilde(tmp_path, ng, argmax=am, labels=PERM)
    flag = tmp_path / "flag.h5"
    bad = am.copy()
    bad[0, 0] += 1
    with h5py.File(flag, "w") as hf:
        hf.create_dataset("degenerate_argmax", data=np.zeros((4, 5), bool))
        hf.create_dataset("argmax_concept", data=bad)
    row = inherit(flag, gt, dry_run=False, schema_version="AS:test")
    assert row["action"] == "NOT_ELEMENTWISE_EQUAL_REFUSED"
    assert stored_seed_labels(flag) is None


def test_inherit_refuses_a_permuted_copy_with_the_same_total(tmp_path):
    """THE FIXTURE THAT PROVES THE CHECK IS POSITIONAL: same rows, permuted,
    so shape and total are identical and only an elementwise check can see it."""
    nr, ng = _null(tmp_path)
    am = np.arange(4 * 5).reshape(4, 5).astype(np.int32)
    gt = _gtilde(tmp_path, ng, argmax=am, labels=PERM)
    flag = tmp_path / "flag.h5"
    permuted = am[[1, 0, 3, 2]]
    assert permuted.sum() == am.sum() and permuted.shape == am.shape
    with h5py.File(flag, "w") as hf:
        hf.create_dataset("degenerate_argmax", data=np.zeros((4, 5), bool))
        hf.create_dataset("argmax_concept", data=permuted)
    row = inherit(flag, gt, dry_run=False, schema_version="AS:test")
    assert row["action"] == "NOT_ELEMENTWISE_EQUAL_REFUSED"


def test_inherit_refuses_when_the_source_is_itself_unlabelled(tmp_path):
    nr, ng = _null(tmp_path)
    am = np.arange(4 * 5).reshape(4, 5).astype(np.int32)
    gt = _gtilde(tmp_path, ng, argmax=am)          # no labels
    flag = tmp_path / "flag.h5"
    with h5py.File(flag, "w") as hf:
        hf.create_dataset("argmax_concept", data=am)
    row = inherit(flag, gt, dry_run=False, schema_version="AS:test")
    assert row["action"] == "SOURCE_UNLABELLED_SKIPPED"


def test_inherit_refuses_without_a_shared_dataset_to_verify_against(tmp_path):
    nr, ng = _null(tmp_path)
    gt = _gtilde(tmp_path, ng, labels=PERM)
    flag = tmp_path / "flag.h5"
    with h5py.File(flag, "w") as hf:
        hf.create_dataset("degenerate_argmax", data=np.zeros((4, 5), bool))
    row = inherit(flag, gt, dry_run=False, schema_version="AS:test")
    assert row["action"] == "NO_SHARED_DATASET_SKIPPED"
    assert "unverifiable" in row["reason"]
