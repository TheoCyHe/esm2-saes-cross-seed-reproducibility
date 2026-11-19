"""Tests for `never_detected` -- PREREGISTRATION amendment AZ section 15.

THE POSITIONAL FIXTURE IS THE POINT. AO6's defect attributed 0.0031% of rows
correctly while every TOTAL agreed, so the wrong-order fixture here has the
SAME protein count, the SAME total row count and the SAME block-length
multiset as the right one. Only a positional check can separate them, and a
fixture whose wrong arm differs in a total would let a total-based check pass.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis import never_detected as nd


def _index(starts, lengths, accs=None):
    accs = accs or [f"sp|P{i:05d}|X_Y" for i in range(len(starts))]
    return [{"accession": a, "start": int(s), "n_rows": int(n)}
            for a, s, n in zip(accs, starts, lengths)]


def test_verify_block_starts_accepts_a_correctly_ordered_index():
    idx = _index([0, 10, 25], [10, 15, 5])
    starts = np.array([0, 10, 25], dtype=np.int64)
    v = nd.verify_block_starts(idx, starts)
    assert v["verdict"] == "POSITIONS_OK"
    assert v["n_blocks_beginning_at_their_own_start"] == 3


def test_verify_block_starts_REFUSES_a_permutation_with_the_SAME_TOTAL():
    """The AO6 shape. The permuted arm has the same three proteins, the same
    30 rows and the same {10, 15, 5} block multiset -- so no total, no count
    and no correlation discriminates."""
    idx = _index([0, 10, 25], [10, 15, 5])
    permuted = np.array([25, 0, 10], dtype=np.int64)
    assert permuted.sum() == np.array([0, 10, 25]).sum()
    v = nd.verify_block_starts(idx, permuted)
    assert v["verdict"] == "POSITIONS_WRONG"
    assert v["n_blocks_beginning_at_their_own_start"] == 0


def test_verify_block_starts_catches_a_SINGLE_displaced_block():
    idx = _index([0, 10, 25], [10, 15, 5])
    nearly = np.array([0, 10, 24], dtype=np.int64)
    v = nd.verify_block_starts(idx, nearly)
    assert v["verdict"] == "POSITIONS_WRONG"
    assert v["n_blocks_beginning_at_their_own_start"] == 2


def test_proteins_of_concept_returns_distinct_protein_indices():
    row2protein = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
    rows = np.array([0, 1, 4], dtype=np.int64)
    assert nd.proteins_of_concept(rows, row2protein).tolist() == [0, 1]


def test_taxonomy_counts_reports_COUNTS_as_well_as_a_share():
    """A '100% Eukaryota' built on 2 proteins and one built on 400 are
    different statements; a share alone cannot say which."""
    t = nd.taxonomy_counts(["Eukaryota", "Eukaryota"], ["Homo sapiens"] * 2)
    assert t["n_proteins_with_taxonomy"] == 2
    assert t["lineage_counts"] == {"Eukaryota": 2}
    assert t["top_lineage_share"] == pytest.approx(1.0)
    assert t["n_distinct_organisms"] == 1


def test_taxonomy_counts_on_an_empty_protein_set_reports_None_not_zero():
    t = nd.taxonomy_counts([], [])
    assert t["n_proteins_with_taxonomy"] == 0
    assert t["top_lineage"] is None and t["top_lineage_share"] is None


def test_group_summary_names_an_empty_group_rather_than_inventing_zeros():
    assert nd.group_summary([])["status"] == "EMPTY_GROUP"


def test_group_summary_reports_the_three_distributions_and_the_key_mix():
    rows = [
        {"n_annotated_residues": 100, "n_proteins": 5,
         "best_seed_margin": 0.5, "feature_key": "REGION"},
        {"n_annotated_residues": 300, "n_proteins": 9,
         "best_seed_margin": 0.9, "feature_key": "REGION"},
        {"n_annotated_residues": 200, "n_proteins": 7,
         "best_seed_margin": 0.1, "feature_key": "BINDING"},
    ]
    g = nd.group_summary(rows)
    assert g["n"] == 3
    assert g["annotated_residues"]["median"] == pytest.approx(200)
    assert g["best_seed_margin"]["max"] == pytest.approx(0.9)
    assert g["feature_key_composition"] == {"REGION": 2, "BINDING": 1}


def test_the_accession_match_floor_is_a_real_threshold():
    """The constant is what stops an empty join exiting 0 (gotcha 10). It is
    asserted here so a change to it is a deliberate edit and not a drift."""
    assert nd.MIN_ACCESSION_MATCH == 0.5
