"""Component 3's fixtures (PREREGISTRATION amendment BQ), under BM-E1
criterion 2: each has a KNOWN ANSWER COMPUTED INDEPENDENTLY of the component,
and each is SHOWN to reach the branch it is named for.

Every expected value here is WRITTEN DOWN, arithmetic done by hand in the
docstring or comment beside it. None is computed by calling the function under
test, which is gotcha 32b -- a test that derives its expectation from the thing
it checks cannot detect a change to that thing.
"""

import numpy as np
import pytest

from src.analysis.classification_2x2 import (
    CELL_ALIGNED_BOTH,
    CELL_ALIGNED_EXP,
    CELL_EXCLUDED,
    CELL_NAMES,
    CELL_NOT_EVALUABLE,
    CELL_ORPHAN,
    GAIN_FLOOR,
    GROUP_OF_CELL,
    OutcomeSpaceViolation,
    SeedOrderUnrecoverable,
    at_floor_counts,
    classify_cells,
    clears_bar,
    gain_from_ap,
    recover_raw_slab_seed,
    recover_seed_position,
    verify_outcome_space,
)


# ---------------------------------------------------------------------------
# gain_from_ap -- AV6's G = (AP - p)/(1 - p)
# ---------------------------------------------------------------------------

def test_gain_matches_hand_arithmetic():
    """AP 0.60 at p 0.20 -> (0.60-0.20)/0.80 = 0.50.
    AP 0.03 at p 0.01 -> (0.03-0.01)/0.99 = 0.0202020202...
    AP equal to prevalence -> exactly 0: a feature no better than the base rate.
    """
    ap = np.array([[0.60, 0.03, 0.20]])
    prev = np.array([0.20, 0.01, 0.20])
    got = gain_from_ap(ap, prev)
    assert got[0, 0] == pytest.approx(0.50, abs=1e-12)
    assert got[0, 1] == pytest.approx(0.02020202020202020, abs=1e-12)
    assert got[0, 2] == 0.0


def test_gain_uses_each_concepts_own_prevalence_not_a_shared_one():
    """C4's invariant. The SAME AP against two different prevalences must give
    two different gains: AP 0.50 at p 0.10 -> 0.4/0.9 = 0.4444...; at p 0.40 ->
    0.1/0.6 = 0.1666.... A global `p` would return one number twice.
    """
    got = gain_from_ap(np.array([[0.50, 0.50]]), np.array([0.10, 0.40]))
    assert got[0, 0] == pytest.approx(0.4444444444444444, abs=1e-12)
    assert got[0, 1] == pytest.approx(0.1666666666666667, abs=1e-12)
    assert got[0, 0] != got[0, 1]


def test_gain_refuses_a_prevalence_of_the_wrong_length():
    with pytest.raises(ValueError, match="prevalence"):
        gain_from_ap(np.zeros((4, 3)), np.zeros(4))


def test_gain_refuses_prevalence_at_one():
    with pytest.raises(ValueError, match="1-p"):
        gain_from_ap(np.zeros((2, 2)), np.array([0.5, 1.0]))


# ---------------------------------------------------------------------------
# clears_bar -- THE CONJUNCT ORDER. This is the fixture that discriminates.
# ---------------------------------------------------------------------------

def test_the_two_conjuncts_are_anded_before_the_existential():
    """THE FIXTURE THAT SEPARATES THE RIGHT PREDICATE FROM THE WRONG ONE.

    One feature, two concepts. It is FDR-significant against concept 0 and
    ONLY concept 0; it clears the gain floor against concept 1 and ONLY
    concept 1. It has therefore cleared AV6's bar against NOTHING.

        reject = [ True , False ]
        gain   = [ 0.01 , 0.99  ]     floor = 0.05

        correct   ANY( reject AND gain>=floor ) = ANY(False, False) = False
        wrong     ANY(reject) AND ANY(gain>=floor) = True AND True   = True

    The two orders disagree here by construction, so this test FAILS against
    the wrong implementation -- which is what makes it evidence.
    """
    reject = np.array([[True, False]])
    gain = np.array([[0.01, 0.99]])
    assert clears_bar(reject, gain, GAIN_FLOOR).tolist() == [False]

    # and the wrong predicate, written out here so the disagreement is visible
    # rather than asserted: it returns True on this same input.
    wrong = bool(reject.any(axis=1)[0] and (gain >= GAIN_FLOOR).any(axis=1)[0])
    assert wrong is True


def test_clears_bar_true_only_when_one_concept_satisfies_both():
    """Concept 1 is both significant and gainful -> the feature clears.
    Expected per feature, by hand:
      f0: (F,0.99),(T,0.90) -> concept1 satisfies both        -> True
      f1: (T,0.04),(F,0.99) -> neither concept satisfies both -> False
      f2: (T,0.05),(F,0.01) -> concept0: gain == floor, >= is inclusive -> True
    """
    reject = np.array([[False, True], [True, False], [True, False]])
    gain = np.array([[0.99, 0.90], [0.04, 0.99], [0.05, 0.01]])
    assert clears_bar(reject, gain, GAIN_FLOOR).tolist() == [True, False, True]


def test_gain_floor_is_inclusive_at_exactly_the_floor():
    """`G >= 0.05` is inclusive; a feature sitting exactly on 0.05 clears."""
    assert clears_bar(np.array([[True]]), np.array([[0.05]]),
                      GAIN_FLOOR).tolist() == [True]
    assert clears_bar(np.array([[True]]), np.array([[0.04999999]]),
                      GAIN_FLOOR).tolist() == [False]


def test_clears_bar_refuses_a_non_boolean_reject():
    """A p-value array passed where a rejection mask belongs would make every
    test 'significant' by truthiness."""
    with pytest.raises(ValueError, match="boolean"):
        clears_bar(np.array([[0.001, 0.9]]), np.array([[0.9, 0.9]]))


def test_clears_bar_refuses_mismatched_grids():
    with pytest.raises(ValueError, match="same"):
        clears_bar(np.zeros((3, 2), dtype=bool), np.zeros((3, 4)))


# ---------------------------------------------------------------------------
# classify_cells -- AV5's grid. Every branch shown to be REACHED.
# ---------------------------------------------------------------------------

def test_each_of_the_four_cells_is_reached_and_labelled_per_av5():
    """AV5's table, transcribed:

        clears exp / clears pooled -> ALIGNED   (cell 0)
        clears exp / fails  pooled -> ALIGNED   (cell 1)
        fails  exp / clears pooled -> EXCLUDED  (cell 2)
        fails  exp / fails  pooled -> ORPHAN    (cell 3)
    """
    e = np.array([True, True, False, False])
    p = np.array([True, False, True, False])
    got = classify_cells(e, p, True, True)
    assert got.tolist() == [CELL_ALIGNED_BOTH, CELL_ALIGNED_EXP,
                            CELL_EXCLUDED, CELL_ORPHAN]
    # every branch is OCCUPIED by this fixture -- the point of criterion 2
    assert set(got.tolist()) == {CELL_ALIGNED_BOTH, CELL_ALIGNED_EXP,
                                 CELL_EXCLUDED, CELL_ORPHAN}


def test_aligned_takes_precedence_so_both_top_row_cells_group_as_aligned():
    """AV5: aligned status is not revoked to agree with the broader screen."""
    assert GROUP_OF_CELL[CELL_ALIGNED_BOTH] == "ALIGNED"
    assert GROUP_OF_CELL[CELL_ALIGNED_EXP] == "ALIGNED"
    assert GROUP_OF_CELL[CELL_EXCLUDED] == "EXCLUDED"
    assert GROUP_OF_CELL[CELL_ORPHAN] == "ORPHAN"


def test_an_unevaluable_experimental_pool_gives_not_evaluable_not_orphan():
    """If the experimental pool has no evaluable concept the existential ranged
    over an empty set. Reading that as ORPHAN would count an absence of
    reference distribution as a measured failure (criterion 6)."""
    e = np.array([False, False, False])
    p = np.array([True, False, True])
    got = classify_cells(e, p, False, True)
    assert got.tolist() == [CELL_NOT_EVALUABLE] * 3
    assert CELL_ORPHAN not in got.tolist()


def test_an_unevaluable_pooled_pool_also_gives_not_evaluable():
    got = classify_cells(np.array([True, False]), np.array([False, False]),
                         True, False)
    assert got.tolist() == [CELL_NOT_EVALUABLE] * 2


def test_not_evaluable_wins_over_a_feature_that_would_have_been_aligned():
    """Precedence: the ignorance test runs FIRST. A feature clearing the
    experimental bar is still NOT_EVALUABLE if the pooled pool is empty,
    because its 2x2 row cannot be placed."""
    got = classify_cells(np.array([True]), np.array([False]), True, False)
    assert got.tolist() == [CELL_NOT_EVALUABLE]


def test_classify_refuses_axes_of_different_length():
    with pytest.raises(ValueError, match="SAME feature axis"):
        classify_cells(np.zeros(4, dtype=bool), np.zeros(5, dtype=bool),
                       True, True)


# ---------------------------------------------------------------------------
# verify_outcome_space -- criterion 5, and it must be able to FAIL
# ---------------------------------------------------------------------------

def test_outcome_space_counts_are_hand_checkable_and_sum_to_the_population():
    """Six pairs: E = T,T,F,F,T,F and P = T,F,T,F,F,F.
    By hand:  cell0 (E&P)   : index 0                  -> 1
              cell1 (E&~P)  : indices 1, 4             -> 2
              cell2 (~E&P)  : index 2                  -> 1
              cell3 (~E&~P) : indices 3, 5             -> 2
              cell4         :                          -> 0
    Total 6.
    """
    e = np.array([True, True, False, False, True, False])
    p = np.array([True, False, True, False, False, False])
    codes = classify_cells(e, p, True, True)
    counts = verify_outcome_space(codes, e, p, True, True)
    assert counts[CELL_ALIGNED_BOTH] == 1
    assert counts[CELL_ALIGNED_EXP] == 2
    assert counts[CELL_EXCLUDED] == 1
    assert counts[CELL_ORPHAN] == 2
    assert counts[CELL_NOT_EVALUABLE] == 0
    assert sum(counts.values()) == 6


def test_the_empty_category_is_reported_as_an_explicit_zero_not_omitted():
    """Criterion 6: NOT_EVALUABLE is carried even when empty."""
    e = np.array([True, False])
    p = np.array([False, False])
    counts = verify_outcome_space(classify_cells(e, p, True, True), e, p,
                                  True, True)
    assert CELL_NOT_EVALUABLE in counts
    assert counts[CELL_NOT_EVALUABLE] == 0
    assert set(counts) == set(CELL_NAMES)


def test_outcome_space_verifier_FAILS_on_a_double_assigned_pair():
    """THE VERIFIER MUST BE ABLE TO FAIL (gotcha 13). A code array that
    disagrees with the specification is caught, not waved through."""
    e = np.array([True, False, False])
    p = np.array([True, True, False])
    codes = classify_cells(e, p, True, True)
    codes[1] = CELL_ORPHAN            # was EXCLUDED; now contradicts the spec
    with pytest.raises(OutcomeSpaceViolation, match="disagrees"):
        verify_outcome_space(codes, e, p, True, True)


def test_outcome_space_verifier_FAILS_when_a_pair_is_given_no_outcome():
    """An out-of-vocabulary code leaves a pair unassigned by every mask."""
    e = np.array([True, False])
    p = np.array([True, False])
    codes = classify_cells(e, p, True, True)
    codes[0] = 9                       # not a cell at all
    with pytest.raises(OutcomeSpaceViolation, match="disagrees"):
        verify_outcome_space(codes, e, p, True, True)


def test_outcome_space_verifier_FAILS_when_not_evaluable_is_silently_dropped():
    """Folding an ignorance category into ORPHAN is criterion 6's named
    failure. The verifier catches it because it re-derives from the spec."""
    e = np.array([False, False])
    p = np.array([False, False])
    codes = np.array([CELL_ORPHAN, CELL_ORPHAN], dtype=np.uint8)
    with pytest.raises(OutcomeSpaceViolation, match="disagrees"):
        verify_outcome_space(codes, e, p, False, True)


# ---------------------------------------------------------------------------
# at_floor_counts -- and the family genuinely has more than one floor
# ---------------------------------------------------------------------------

def test_at_floor_count_uses_each_concepts_OWN_atom_not_a_family_constant():
    """TWO FLOORS IN ONE FAMILY, which is BP1d's shape in miniature.

    concept 0: n_finite = 9 -> atom 1/10 = 0.1
    concept 1: n_finite = 4 -> atom 1/5  = 0.2

        p = [[0.1, 0.2],
             [0.1, 0.4],
             [0.3, 0.2]]

    at floor, by hand: concept 0 rows 0 and 1 -> 2; concept 1 rows 0 and 2 -> 2.
    TOTAL 4.

    A single family-wide atom of 0.1 would score concept 1 as 0 and total 2, so
    this fixture DISCRIMINATES the multi-floor path from the single-floor one.
    """
    p = np.array([[0.1, 0.2], [0.1, 0.4], [0.3, 0.2]])
    n_finite = np.array([9, 4])
    assert at_floor_counts(p, n_finite) == 4

    single_floor_answer = int((p == 0.1).sum())   # what one atom would give
    assert single_floor_answer == 2               # ... and it is NOT 4


def test_at_floor_ignores_concepts_marked_not_evaluable():
    """A concept with no reference distribution contributes no floor-tied
    tests -- its p-values are NaN and must not be counted."""
    p = np.array([[0.1, 0.2], [0.1, 0.2]])
    n_finite = np.array([9, 4])
    assert at_floor_counts(p, n_finite,
                           not_evaluable=np.array([False, True])) == 2


def test_at_floor_compares_in_the_stored_dtype_so_float32_is_exact():
    """The artifact stores p as float32. The atom is cast DOWN to meet it, so
    the comparison is an exact equality and not a tolerance (entry 123)."""
    n = 204800
    atom32 = np.float32(1.0 / (n + 1.0))
    p = np.array([[atom32], [np.float32(2.0 / (n + 1.0))], [atom32]],
                 dtype=np.float32)
    assert at_floor_counts(p, np.array([n])) == 2


def test_at_floor_refuses_a_mismatched_n_finite():
    with pytest.raises(ValueError, match="at_floor_counts"):
        at_floor_counts(np.zeros((3, 2)), np.array([9, 4, 1]))


# ---------------------------------------------------------------------------
# the deductive structural checks BQ6 asserts, exercised on synthetic data
# ---------------------------------------------------------------------------

def test_aligned_count_can_never_exceed_the_rejection_count():
    """BQ6 check 1, on data where the inequality is TIGHT and where it is
    SLACK, so the test is not satisfied by a trivially loose bound.

    Tight: 3 features, each rejected against exactly one concept, all gainful
           -> 3 rejections, 3 ALIGNED.
    Slack: one feature rejected against both concepts -> 2 rejections,
           1 ALIGNED (distinct features, not tests).
    """
    reject_t = np.array([[True, False], [True, False], [False, True]])
    gain_t = np.full((3, 2), 0.9)
    aligned_t = int(clears_bar(reject_t, gain_t).sum())
    assert int(reject_t.sum()) == 3 and aligned_t == 3

    reject_s = np.array([[True, True], [False, False]])
    gain_s = np.full((2, 2), 0.9)
    aligned_s = int(clears_bar(reject_s, gain_s).sum())
    assert int(reject_s.sum()) == 2 and aligned_s == 1
    assert aligned_s <= int(reject_s.sum())


def test_a_seed_that_rejects_nothing_has_zero_aligned():
    """BQ6 check 2. With no rejection anywhere the first conjunct is False for
    every (feature, concept), so ALIGNED is 0 whatever the gains are."""
    reject = np.zeros((5, 4), dtype=bool)
    gain = np.full((5, 4), 0.99)           # every gain clears the floor
    e = clears_bar(reject, gain)
    assert int(e.sum()) == 0
    codes = classify_cells(e, np.zeros(5, dtype=bool), True, True)
    counts = verify_outcome_space(codes, e, np.zeros(5, dtype=bool),
                                  True, True)
    assert counts[CELL_ALIGNED_BOTH] == 0
    assert counts[CELL_ALIGNED_EXP] == 0
    assert counts[CELL_ORPHAN] == 5


# ---------------------------------------------------------------------------
# recover_seed_position -- the pooled raw null carries NO seed labels
# ---------------------------------------------------------------------------

def _null_g_fixture():
    """Three labelled seeds, each with a DISTINCT max-over-concepts array.
    Values are written down, not generated, so the expected match is obvious:
    seed 0 -> all 1s, seed 1 -> all 2s, seed 2 -> all 3s.
    """
    return np.array([np.full((2, 4), 1.0),
                     np.full((2, 4), 2.0),
                     np.full((2, 4), 3.0)])


def test_recovery_finds_the_one_seed_whose_null_g_matches():
    g = _null_g_fixture()
    assert recover_seed_position(np.full((2, 4), 3.0), g) == 2
    assert recover_seed_position(np.full((2, 4), 1.0), g) == 0


def test_recovery_FAILS_when_no_labelled_seed_matches():
    """Zero matches means the reductions do not describe this raw array. It
    must refuse, not fall back on position."""
    with pytest.raises(SeedOrderUnrecoverable, match="NO labelled seed"):
        recover_seed_position(np.full((2, 4), 99.0), _null_g_fixture())


def test_recovery_FAILS_when_two_labelled_seeds_match():
    """Ambiguity is not resolved by taking the first: a recovery that cannot
    discriminate must say so (gotcha 25 -- a check that can only pass)."""
    g = np.array([np.full((2, 4), 5.0), np.full((2, 4), 5.0)])
    with pytest.raises(SeedOrderUnrecoverable, match="2 labelled seeds match"):
        recover_seed_position(np.full((2, 4), 5.0), g)


def test_recovery_DETECTS_A_PERMUTED_SEED_AXIS():
    """THE FIXTURE THAT SHOWS THE RECOVERY IS WORTH HAVING.

    A raw array whose seed axis is the reverse of the reductions'. Recovering
    slab 0 must return position 2, NOT 0 -- which is what a positional join
    would have silently assumed. 18 of 20 rows disagree in the real case
    (gotcha 29).
    """
    g = _null_g_fixture()
    raw_slabs = [np.full((2, 4), 3.0), np.full((2, 4), 2.0),
                 np.full((2, 4), 1.0)]          # reversed
    got = [recover_seed_position(m, g) for m in raw_slabs]
    assert got == [2, 1, 0]
    assert got != [0, 1, 2]


def test_recovery_refuses_a_slab_of_the_wrong_shape():
    with pytest.raises(ValueError, match="recover_seed_position"):
        recover_seed_position(np.zeros((3, 4)), _null_g_fixture())


# ---- the CT10-17 preamble: raw-slab recovery in the stored precision -----------

def _f64_accumulated_null(seed=7):
    """A null whose gain was accumulated in float64 (the segmented CPU path):
    `null_g` keeps the float64 max, the raw array its float32 cast. Three
    seeds, four shifts, six concepts, fifty features."""
    rng = np.random.default_rng(seed)
    G = rng.uniform(0, 1, (3, 4, 6, 50))
    return G.astype(np.float32), G.max(axis=2)


def test_a_f64_accumulated_null_defeats_a_float64_recovery():
    """The founding instance (CT5's gtilde jobs 30306560/1, same shape): in
    float64 the float32 slab's max matches no row of the float64 null_g."""
    raw, null_g = _f64_accumulated_null()
    m = np.nanmax(raw[0], axis=1)
    with pytest.raises(SeedOrderUnrecoverable, match="NO labelled seed"):
        recover_seed_position(m, null_g)


def test_raw_slab_recovery_compares_in_the_stored_precision():
    raw, null_g = _f64_accumulated_null()
    assert [recover_raw_slab_seed(raw[s], null_g) for s in range(3)] == [0, 1, 2]


def test_raw_slab_recovery_still_detects_a_permutation():
    raw, null_g = _f64_accumulated_null()
    assert [recover_raw_slab_seed(raw[s], null_g) for s in (2, 0, 1)] == [2, 0, 1]


def test_raw_slab_recovery_still_refuses_foreign_reductions():
    """A difference far above float32 rounding is not absorbed by the cast."""
    raw, null_g = _f64_accumulated_null()
    with pytest.raises(SeedOrderUnrecoverable, match="NO labelled seed"):
        recover_raw_slab_seed(raw[0], null_g + 1e-4)


def test_raw_slab_recovery_ignores_a_degenerate_nan_draw():
    """NaN is treated as missing under the max, as BH2 drops degenerate draws."""
    raw, null_g = _f64_accumulated_null()
    raw = raw.copy()
    raw[1, 0, 3, :] = np.nan
    null_g = null_g.copy()
    null_g[1, 0, :] = np.nanmax(raw[1, 0].astype(np.float64), axis=0)
    assert recover_raw_slab_seed(raw[1], null_g) == 1


def test_raw_slab_recovery_is_unchanged_for_a_float32_representable_null():
    """Every pooled null of record: null_g is a max over float32 values held in
    float64, so the cast is lossless and the recovery is what it was."""
    raw, _ = _f64_accumulated_null()
    null_g = raw.max(axis=2).astype(np.float64)
    for s in range(3):
        assert recover_raw_slab_seed(raw[s], null_g) ==             recover_seed_position(np.nanmax(raw[s], axis=1), null_g) == s
