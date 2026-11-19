"""
Tests for null_pilot.py's core circular-shift logic (Amendment F3). The
correctness-critical property: a shift must wrap WITHIN its own protein's
row range and never cross into a neighbouring protein -- a naive global
modulo would violate this at every protein boundary.
"""

from __future__ import annotations

import numpy as np
import torch

from src.analysis.null_pilot import (
    build_length_bins,
    build_population_index,
    build_row_to_protein,
    cross_protein_shift_rows,
    permute_within_bins,
    protein_starts_lengths,
    shift_rows,
)
from src.analysis.sparse_ap_index import (
    average_precision_from_touches,
    build_row_index,
    gather_touches,
)


def _index(protein_lengths: list[int]) -> list[dict]:
    starts, out = 0, []
    for i, n in enumerate(protein_lengths):
        out.append({"accession": f"P{i}", "start": starts, "n_rows": n})
        starts += n
    return out


def test_row_to_protein_maps_every_row_to_the_right_protein():
    index = _index([5, 3, 10])
    row2protein = build_row_to_protein(index, n_rows=18)
    assert row2protein[0:5].tolist() == [0] * 5
    assert row2protein[5:8].tolist() == [1] * 3
    assert row2protein[8:18].tolist() == [2] * 10


def test_row_to_protein_asserts_full_coverage():
    """A gap in the cache (e.g. an off-by-one in start/n_rows bookkeeping
    upstream) must be caught, not silently mapped to protein -1."""
    import pytest
    index = _index([5, 3])          # covers only 8 of 10 rows
    with pytest.raises(AssertionError):
        build_row_to_protein(index, n_rows=10)


def test_shift_never_crosses_a_protein_boundary():
    """THE property. A naive GLOBAL circular shift (mod n_rows) would move
    rows across protein boundaries; a per-protein shift must not, regardless
    of how large the shift amount is relative to a SMALL protein."""
    index = _index([4, 6, 3])       # proteins at [0,4), [4,10), [10,13)
    row2protein = build_row_to_protein(index, n_rows=13)
    starts, lengths = protein_starts_lengths(index)

    rows = np.array([0, 1, 2, 3, 4, 5, 9, 10, 11, 12])
    # large shifts, deliberately exceeding several proteins' own lengths
    shift_per_protein = np.array([100, 7, 1000])
    shifted = shift_rows(rows, row2protein, starts, lengths, shift_per_protein)

    # every shifted row must land back in the SAME protein it started in
    orig_protein = row2protein[rows]
    new_protein = row2protein[shifted]
    assert np.array_equal(orig_protein, new_protein)
    # and within [start, start+length) for that protein
    for i, r in enumerate(shifted):
        p = orig_protein[i]
        assert starts[p] <= r < starts[p] + lengths[p]


def test_shift_zero_is_the_identity():
    index = _index([5, 4])
    row2protein = build_row_to_protein(index, n_rows=9)
    starts, lengths = protein_starts_lengths(index)
    rows = np.array([0, 2, 4, 5, 8])
    shifted = shift_rows(rows, row2protein, starts, lengths,
                         shift_per_protein=np.zeros(2, dtype=np.int64))
    assert np.array_equal(shifted, rows)


def test_shift_wraps_correctly_at_the_protein_end():
    """A concrete, hand-checkable case: protein [0,5), shift by 3.
    row 0 -> 3, row 1 -> 4, row 2 -> 0 (wraps), row 3 -> 1, row 4 -> 2."""
    index = _index([5])
    row2protein = build_row_to_protein(index, n_rows=5)
    starts, lengths = protein_starts_lengths(index)
    rows = np.array([0, 1, 2, 3, 4])
    shifted = shift_rows(rows, row2protein, starts, lengths,
                         shift_per_protein=np.array([3]))
    assert shifted.tolist() == [3, 4, 0, 1, 2]


def test_joint_shift_moves_two_different_concepts_the_same_way_within_one_protein():
    """THE joint-shift property (F3): two DIFFERENT concepts' positive rows
    within the SAME protein, under the SAME shift amount, must move by
    exactly the same relative offset -- this is what makes it a joint shift
    of the protein's whole annotation architecture, not independent
    per-concept shifts that happen to share a random source."""
    index = _index([6])
    row2protein = build_row_to_protein(index, n_rows=6)
    starts, lengths = protein_starts_lengths(index)
    shift = np.array([2])

    concept_a_rows = np.array([0, 1])       # e.g. a domain
    concept_b_rows = np.array([3])          # e.g. a binding site, same protein

    shifted_a = shift_rows(concept_a_rows, row2protein, starts, lengths, shift)
    shifted_b = shift_rows(concept_b_rows, row2protein, starts, lengths, shift)

    # both moved by the identical +2 (mod 6) relative offset
    assert shifted_a.tolist() == [2, 3]
    assert shifted_b.tolist() == [5]
    # the RELATIVE distance between the two concepts' rows is preserved --
    # this is exactly the co-occurrence structure an independent shift would
    # destroy
    assert (concept_b_rows[0] - concept_a_rows[0]) == (shifted_b[0] - shifted_a[0])


# ------------------------------------------------- G1: normalize-before-max --

def test_null_max_uses_gain_not_raw_ap():
    """THE G1 fix. Constructs a case (verified numerically before writing this
    test, seed=3, deterministic) where a BROAD concept (p=0.40) has HIGHER
    raw AP than a NARROW concept (p=0.02) -- 0.5230 vs 0.5 -- but the NARROW
    concept has the higher NORMALIZED GAIN (0.2050 vs 0.4898), because gain
    divides out how much of the broad concept's raw AP is just its own high
    prevalence. A null (or g_f) built by maxing RAW AP picks broad; maxing
    GAIN picks narrow. This is exactly null_pilot.py's bug before the fix:
    `running_max = torch.maximum(running_max, ap)` never normalized at all.
    """
    n_rows = 300
    act = np.arange(n_rows, 0, -1).astype(np.float64)   # row i has rank i+1

    narrow_pos = np.array([0, 1, 4, 19, 39, 79])         # hand-derived: AP=0.5 exactly
    rng = np.random.default_rng(3)
    remaining = np.setdiff1d(np.arange(n_rows), narrow_pos)
    top_pool = remaining[remaining < 150]
    bot_pool = remaining[remaining >= 150]
    n_top = min(int(120 * 0.7), len(top_pool))
    broad_pos = np.concatenate([
        rng.choice(top_pool, n_top, replace=False),
        rng.choice(bot_pool, 120 - n_top, replace=False),
    ])

    Z = act.reshape(-1, 1)     # ONE feature, activation = act
    feat, pos, seg_start = _feature_major_local(Z)
    feat_by_row, rank_by_row, row_offsets = build_row_index(feat, pos, seg_start, n_rows)

    def gain_for(positions: np.ndarray) -> tuple[float, float]:
        n_pos = positions.size
        p = n_pos / n_rows
        query_rows = torch.from_numpy(positions.astype(np.int64))
        tf, tr = gather_touches(feat_by_row, rank_by_row, row_offsets, query_rows)
        ap = average_precision_from_touches(tf, tr, 1, n_pos, n_rows)[0].item()
        gain = (ap - p) / (1.0 - p)
        return ap, gain

    ap_broad, g_broad = gain_for(broad_pos)
    ap_narrow, g_narrow = gain_for(narrow_pos)

    # sanity: reproduces the numerically-verified construction
    assert abs(ap_narrow - 0.5) < 1e-9
    assert ap_broad > ap_narrow                    # raw AP: broad wins
    assert g_narrow > g_broad                      # gain: narrow wins -- FLIPPED

    raw_ap_winner = "broad" if ap_broad > ap_narrow else "narrow"
    gain_winner = "broad" if g_broad > g_narrow else "narrow"
    assert raw_ap_winner == "broad"
    assert gain_winner == "narrow"
    assert raw_ap_winner != gain_winner             # the disagreement IS the point

    # the FIXED null_pilot.py behavior: running_max over GAIN must pick narrow
    running_max_gain = max(g_broad, g_narrow)
    assert running_max_gain == g_narrow
    # what the OLD (buggy) code would have produced: running_max over raw AP
    running_max_raw_ap_bug = max(ap_broad, ap_narrow)
    assert running_max_raw_ap_bug == ap_broad
    assert running_max_gain != running_max_raw_ap_bug   # fix changes the answer


# --------------------------------------------------- G2: population remap --

def test_build_population_index_remaps_rows_and_excludes_others():
    """Population = whole proteins P0 [0,4) and P2 [10,13); P1 [4,10) is
    excluded entirely (e.g. truncated). The subset-relative index must place
    P0's 4 rows at [0,4) and P2's 3 rows at [4,7), and `remap` must send
    every original P0/P2 row to its new subset position while leaving every
    P1 row (and any row past the last protein) at -1."""
    index = _index([4, 6, 3])          # P0 [0,4), P1 [4,10), P2 [10,13)
    subset = np.array([0, 1, 2, 3, 10, 11, 12])   # P0 + P2, P1 excluded

    pop_index, remap = build_population_index(index, subset)

    assert [e["accession"] for e in pop_index] == ["P0", "P2"]
    assert pop_index[0]["start"] == 0 and pop_index[0]["n_rows"] == 4
    assert pop_index[1]["start"] == 4 and pop_index[1]["n_rows"] == 3

    # original P0 rows 0..3 -> subset positions 0..3
    assert remap[[0, 1, 2, 3]].tolist() == [0, 1, 2, 3]
    # original P2 rows 10..12 -> subset positions 4..6
    assert remap[[10, 11, 12]].tolist() == [4, 5, 6]
    # excluded P1 rows are unmapped
    assert (remap[[4, 5, 6, 7, 8, 9]] == -1).all()


def test_build_population_index_asserts_on_partial_protein_inclusion():
    """G2's population filter is defined to include or exclude a protein
    WHOLESALE (build_residue_labels.py's --truncation-policy logic) -- a subset
    that contains only PART of a protein's rows must never happen. Verify the
    guard actually fires: pass a subset missing row 3 of P0's 4 rows."""
    import pytest
    index = _index([4, 6, 3])
    subset = np.array([0, 1, 2])       # P0 rows 0,1,2 but NOT row 3 -- partial
    with pytest.raises(AssertionError):
        build_population_index(index, subset)


# -------------------------------------- G4: Null B, cross-protein shift ----

def test_build_length_bins_groups_similar_lengths_together():
    """Ten proteins, lengths 10,20,...,100, split into 2 quantile bins. The
    5 shortest must land in one bin and the 5 longest in the other -- a
    concept confined to short proteins must never be permuted onto a
    population of long ones (that would be a length-regime confound, not the
    protein-identity test Null B is meant to be)."""
    lengths = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    bins = build_length_bins(lengths, n_bins=2)
    assert len(np.unique(bins)) == 2
    short = bins[:5]
    long = bins[5:]
    assert len(np.unique(short)) == 1
    assert len(np.unique(long)) == 1
    assert short[0] != long[0]


def test_build_length_bins_handles_fewer_unique_lengths_than_bins():
    """If every protein has the SAME length, quantile edges collapse to one
    point -- must not crash or produce more bins than there is information
    to support."""
    lengths = np.full(6, 50)
    bins = build_length_bins(lengths, n_bins=5)
    assert len(np.unique(bins)) == 1


def test_permute_within_bins_only_pairs_proteins_in_the_same_bin():
    """THE Null-B confound guard. A permutation that ever assigns a protein a
    destination from a DIFFERENT bin would let a short-protein concept get
    tested against long-protein geometry -- verify no replicate does that,
    across many draws."""
    rng = np.random.default_rng(0)
    bins = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2])
    for _ in range(20):
        dest = permute_within_bins(bins, rng)
        assert np.array_equal(bins[dest], bins), "a protein crossed bins"


def test_permute_within_bins_is_not_the_identity():
    """A permutation that always maps every protein to itself would make
    Null B silently degenerate into a no-op (each concept's positives land
    right back where they started) -- verify at least one bin actually
    reassigns proteins over several replicates."""
    rng = np.random.default_rng(1)
    bins = np.zeros(20, dtype=np.int64)      # one big bin, plenty of proteins
    saw_a_move = any(not np.array_equal(permute_within_bins(bins, rng),
                                        np.arange(20))
                     for _ in range(10))
    assert saw_a_move


def test_permute_within_bins_leaves_a_singleton_bin_as_identity():
    """A bin with only one protein has no same-length partner to permute
    with -- it must map to itself (identity), not crash or leak into another
    bin."""
    rng = np.random.default_rng(2)
    bins = np.array([0, 1, 1, 1])           # protein 0 alone in bin 0
    dest = permute_within_bins(bins, rng)
    assert dest[0] == 0


def test_cross_protein_shift_rescales_proportionally():
    """Concrete, hand-checkable case: source protein length 10, destination
    length 20 (same bin, different length). A row at RELATIVE offset 5 (the
    midpoint) must land at destination offset 10 -- the midpoint of the
    LONGER protein, not offset 5 (which would be off-center) or some other
    arbitrary point."""
    index = _index([10, 20])                # P0 [0,10), P1 [10,30)
    row2protein = build_row_to_protein(index, n_rows=30)
    starts, lengths = protein_starts_lengths(index)
    dest_protein = np.array([1, 1])          # both proteins' rows go to P1

    rows = np.array([5])                     # P0's row 5 (relative offset 5)
    shifted = cross_protein_shift_rows(rows, row2protein, starts, lengths, dest_protein)
    assert shifted.tolist() == [20]          # P1.start(10) + floor(5*20/10)=10 -> 20


def test_cross_protein_shift_clips_into_the_destination_range():
    """A source offset must never land at or past the destination protein's
    OWN length, even from floating-point rescaling at the boundary -- the
    last row (offset 9 of a length-10 source) rescaled onto a length-20
    destination is floor(9*20/10)=18, still valid, but the guard is exercised
    directly here with a rescale that would overshoot without clipping."""
    index = _index([10, 5])                  # P0 [0,10), P1 [10,15) -- SHORTER dest
    row2protein = build_row_to_protein(index, n_rows=15)
    starts, lengths = protein_starts_lengths(index)
    dest_protein = np.array([1, 1])

    rows = np.array([9])                     # P0's LAST row
    shifted = cross_protein_shift_rows(rows, row2protein, starts, lengths, dest_protein)
    p1_start, p1_len = starts[1], lengths[1]
    assert p1_start <= shifted[0] < p1_start + p1_len


def test_cross_protein_shift_moves_two_concepts_in_one_protein_to_the_same_destination():
    """THE joint property (mirrors test_joint_shift_moves_two_different_
    concepts_the_same_way_within_one_protein for Null A): two different
    concepts' positive rows in the SAME source protein, under the SAME
    dest_protein array, must land in the SAME destination protein -- that
    shared destination is what makes this a relocation of the protein's
    WHOLE annotation architecture, not independent per-concept reshuffles
    that would destroy co-occurrence structure."""
    index = _index([10, 10, 10])             # P0, P1 (same length, same bin), P2
    row2protein = build_row_to_protein(index, n_rows=30)
    starts, lengths = protein_starts_lengths(index)
    dest_protein = np.array([1, 0, 2])       # P0<->P1 swapped, P2 fixed

    concept_a_rows = np.array([0, 1])        # in P0
    concept_b_rows = np.array([5])           # also in P0

    shifted_a = cross_protein_shift_rows(concept_a_rows, row2protein, starts, lengths, dest_protein)
    shifted_b = cross_protein_shift_rows(concept_b_rows, row2protein, starts, lengths, dest_protein)

    dest_of_a = row2protein[shifted_a]
    dest_of_b = row2protein[shifted_b]
    assert np.all(dest_of_a == 1) and np.all(dest_of_b == 1)   # both landed on P1


def _feature_major_local(Z: np.ndarray):
    n_rows, n_features = Z.shape
    feats, poss = [], []
    for f in range(n_features):
        nz = np.flatnonzero(Z[:, f])
        order = nz[np.argsort(-Z[nz, f], kind="stable")]
        feats.append(np.full(len(order), f, dtype=np.int64))
        poss.append(order.astype(np.int64))
    feat = np.concatenate(feats) if feats else np.zeros(0, dtype=np.int64)
    pos = np.concatenate(poss) if poss else np.zeros(0, dtype=np.int64)
    counts = np.bincount(feat, minlength=n_features)
    seg_start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return (torch.from_numpy(feat), torch.from_numpy(pos),
            torch.from_numpy(seg_start).to(torch.int64))


# ---- amendment CT, §CT11.7-R: --concepts-from ------------------------------

def test_restrict_to_concepts_keeps_the_available_order():
    from src.analysis.null_pilot import restrict_to_concepts
    got = restrict_to_concepts(["A", "B", "C", "D"], ["D", "B"])
    assert got == ["B", "D"]


def test_restrict_to_concepts_refuses_a_concept_without_positives():
    import pytest
    from src.analysis.null_pilot import restrict_to_concepts
    with pytest.raises(SystemExit, match="no positives"):
        restrict_to_concepts(["A", "B"], ["A", "Z"])


# ---- amendment CT, §CT11.7-R: the population's row count on a subset path --

def test_population_n_rows_is_the_subset_size_on_a_smaller_population():
    """jobs 30207159/30207160 sized the row->protein index to the label store
    on the shared_experimental subset and the coverage assertion fired."""
    from src.analysis.null_pilot import (build_population_index,
                                         build_row_to_protein,
                                         population_n_rows)
    index = [{"accession": "a", "start": 0, "n_rows": 3, "truncated": False},
             {"accession": "b", "start": 3, "n_rows": 4, "truncated": False},
             {"accession": "c", "start": 7, "n_rows": 2, "truncated": False}]
    subset = np.array([3, 4, 5, 6], dtype=np.int64)         # protein 2 only
    n = population_n_rows(subset, 9, "shared_experimental")
    assert n == 4
    pop_index, _ = build_population_index(index, subset)
    r2p = build_row_to_protein(pop_index, n)
    assert r2p.tolist() == [0, 0, 0, 0]


def test_population_n_rows_refuses_a_label_store_disagreement():
    import pytest
    from src.analysis.null_pilot import population_n_rows
    assert population_n_rows(np.arange(9), 9, "label_store") == 9
    with pytest.raises(SystemExit, match="label_store subset"):
        population_n_rows(np.arange(8), 9, "label_store")
