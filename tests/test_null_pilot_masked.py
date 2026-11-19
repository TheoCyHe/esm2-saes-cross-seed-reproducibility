"""
The masked experimental-stratum null -- PREREGISTRATION amendments BG and BH.

WHY THE EXPECTED VALUES ARE HAND-COMPUTED. Acceptance criterion AC2 asks for a
fixture that would DETECT the absence of masking, not one that agrees with
whatever the implementation does. Every number below is worked out from the AP
definition in `feature_ap.py`'s module docstring

    AP = (1/P) * sum_{firing positives i} (TP_i / rank_i)
         + (1 - P_fire/P) * (P/N)

and none is read off a run. The discriminating quantity is the one a wrong
implementation gets wrong in a NAMED way: masking a row that FIRES the feature
must RENUMBER the ranks below it. An implementation that removes the row from
the denominator `N` but leaves it in the RANKING returns a different number --
and that is not a hypothetical defect. It is what the AW5 variant pilot
(`v1v2v3_pilot_aw.py`) actually did: it built a `keep_row_v1` mask and then
computed AP through the sparse touch path, which ranks against the full row
set, leaving the mask unused. The first test below names the value that
implementation returns and asserts against it.

Gotcha 16's fixture lesson is the governing one throughout: a fixture that
cannot exhibit the disputed condition cannot test it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.analysis.null_pilot import (
    BH3_BOTH_ARMS,
    BH3_NO_ARM,
    BH3_NONE,
    BH3_ONE_ARM,
    BH3_SOME,
    apply_collision_rule,
    build_displacement_plan,
    build_row_to_protein,
    classify_bh3,
    exceedance_resolution,
    masked_concept_arrays,
    masked_draw_gain,
    protein_starts_lengths,
)


def _index(protein_lengths: list[int]) -> list[dict]:
    start, out = 0, []
    for i, n in enumerate(protein_lengths):
        out.append({"accession": f"P{i}", "start": start, "n_rows": n})
        start += n
    return out


# One feature, six rows. Rows 0..3 fire in descending activation order, so
# their ranks within the feature's block are 1, 2, 3, 4. Rows 4 and 5 never
# fire and sit in the tied-zero block.
_Z = np.array([[4.0], [3.0], [2.0], [1.0], [0.0], [0.0]])
_N_ROWS = 6


def _feature_major(Z: np.ndarray):
    n_features = Z.shape[1]
    feats, poss = [], []
    for f in range(n_features):
        nz = np.flatnonzero(Z[:, f])
        order = nz[np.argsort(-Z[nz, f], kind="stable")]
        feats.append(np.full(len(order), f, dtype=np.int64))
        poss.append(order.astype(np.int64))
    feat = np.concatenate(feats)
    pos = np.concatenate(poss)
    counts = np.bincount(feat, minlength=n_features)
    seg = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return (torch.from_numpy(feat), torch.from_numpy(pos),
            torch.from_numpy(seg).to(torch.int64))


def _keep(masked_rows):
    if masked_rows is None:
        return None, _N_ROWS
    keep = np.ones(_N_ROWS, dtype=bool)
    keep[np.asarray(masked_rows)] = False
    return keep, _N_ROWS - len(masked_rows)


def _ap(positives, masked_rows):
    """Through `masked_concept_arrays`, i.e. the function `main()` calls --
    so a mutation of the mask application is visible to these fixtures."""
    feat, pos, seg = _feature_major(_Z)
    keep, n_eff = _keep(masked_rows)
    f_c, p_c, s_c = masked_concept_arrays(feat, pos, seg, keep, 1)
    lab = torch.zeros(_N_ROWS, dtype=torch.bool)
    lab[torch.tensor(sorted(positives))] = True
    # `masked_draw_gain` returns the normalised gain; invert it to recover AP,
    # so the hand-computed AP values stay the thing being asserted.
    p_draw = len(positives) / n_eff
    gain = masked_draw_gain(f_c, s_c, lab[p_c], 1, len(positives), n_eff)
    return float(gain[0]) * (1.0 - p_draw) + p_draw


def _gain(positives, masked_rows):
    feat, pos, seg = _feature_major(_Z)
    keep, n_eff = _keep(masked_rows)
    f_c, p_c, s_c = masked_concept_arrays(feat, pos, seg, keep, 1)
    lab = torch.zeros(_N_ROWS, dtype=torch.bool)
    lab[torch.tensor(sorted(positives))] = True
    return float(masked_draw_gain(f_c, s_c, lab[p_c], 1, len(positives), n_eff)[0])


# ============================================ AC2: the discrimination case ==

def test_masking_a_FIRING_row_renumbers_the_ranks_below_it():
    """THE discrimination case. Positives {0, 3}; row 1 fires at rank 2 and is
    masked.

    Unmasked, by hand: row 0 is rank 1 with TP 1 -> 1/1; row 3 is rank 4 with
    TP 2 -> 2/4. Both positives fire, so P_fire == P and the tied-zero tail
    carries no recall. AP = (1 + 0.5) / 2 = 0.75.

    Masked, by hand: row 1 leaves the universe, so the surviving firing rows
    are 0, 2, 3 at ranks 1, 2, 3 -- row 3's rank drops from 4 to 3.
    AP = (1 + 2/3) / 2 = 5/6.

    THE WRONG IMPLEMENTATION'S VALUE IS NAMED. Dropping row 1 from `N` while
    leaving it in the ranking keeps row 3 at rank 4 and returns 0.75 --
    indistinguishable from no masking at all. The last assertion fails against
    that implementation, which is what makes this fixture evidence rather than
    agreement.
    """
    unmasked = _ap({0, 3}, None)
    masked = _ap({0, 3}, [1])
    assert unmasked == pytest.approx(0.75, abs=1e-12)
    assert masked == pytest.approx(5.0 / 6.0, abs=1e-12)
    assert masked != pytest.approx(unmasked, abs=1e-9)


def test_the_tied_zero_tail_uses_the_EFFECTIVE_row_count():
    """Positives {0, 5}: row 5 never fires, so it is carried entirely by the
    tail term, whose precision is exactly P/N. Masking row 1 changes N.

    Unmasked, by hand: ap_fire = 1 (row 0 at rank 1), P_fire = 1,
    tail = (1 - 1/2) * (2/6) = 1/6, AP = 1/2 + 1/6 = 2/3.
    Masked, by hand: N = 5, tail = (1 - 1/2) * (2/5) = 1/5, AP = 0.7.
    """
    assert _ap({0, 5}, None) == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert _ap({0, 5}, [1]) == pytest.approx(0.7, abs=1e-12)


def test_masking_a_row_that_neither_fires_nor_is_positive_leaves_AP_UNCHANGED():
    """THE CASE THAT SHOULD STILL PASS (AC2 requires one).

    Row 5 is outside the feature's firing set and is not a positive, and both
    positives fire -- so P_fire == P, the tail term is identically zero
    whatever N is, and no rank moves. AP is EXACTLY unchanged.

    A suite that only ever demanded "masking changes the answer" would be
    satisfied by an implementation that perturbs AP for any mask at all. This
    one is not.
    """
    assert _ap({0, 3}, [5]) == pytest.approx(_ap({0, 3}, None), abs=1e-12)
    assert _ap({0, 3}, [5]) == pytest.approx(0.75, abs=1e-12)


def test_the_gain_is_normalised_by_the_EFFECTIVE_row_count_not_the_full_one():
    """G1's gain is (AP - p) / (1 - p), and under masking `p` must be the
    draw's OWN prevalence on the MASKED universe.

    Positives {0, 3}, row 1 masked: AP = 5/6 and n_eff = 5, so p = 2/5 and
    gain = (5/6 - 2/5) / (1 - 2/5) = 13/18 = 0.7222...

    THE WRONG VALUE IS NAMED: normalising by the unmasked n_rows = 6 gives
    p = 1/3 and gain = (5/6 - 1/3) / (2/3) = 0.75. Both are plausible numbers
    in the same range, which is exactly why this needs an assertion rather
    than a reading.
    """
    assert _gain({0, 3}, [1]) == pytest.approx(13.0 / 18.0, abs=1e-12)
    assert _gain({0, 3}, [1]) != pytest.approx(0.75, abs=1e-9)


def test_masked_concept_arrays_with_no_mask_returns_the_inputs_UNTOUCHED():
    """The unmasked branch must be the identity on all three arrays, or the
    pooled arm is no longer the same computation and AC1's bit-identity
    criterion could not hold."""
    feat, pos, seg = _feature_major(_Z)
    out = masked_concept_arrays(feat, pos, seg, None, 1)
    assert out[0] is feat and out[1] is pos and out[2] is seg


def test_masked_concept_arrays_actually_DROPS_the_masked_rows_nonzeros():
    """The positional consequence of the mask, asserted directly rather than
    only through the AP value: row 1's non-zero must be gone."""
    feat, pos, seg = _feature_major(_Z)
    keep, _ = _keep([1])
    f_c, p_c, _ = masked_concept_arrays(feat, pos, seg, keep, 1)
    assert p_c.tolist() == [0, 2, 3]          # row 1 removed, order preserved
    assert f_c.numel() == 3


# ================================================= BG2: the collision rule ==

def test_collision_rule_drops_displaced_positives_that_land_on_a_masked_row():
    """BG2: the mask WINS. The row stays masked, and the positive is dropped
    from that draw -- not re-drawn, and it does not displace the mask."""
    keep = np.array([True, False, True, False, True])
    assert apply_collision_rule(np.array([0, 1, 2, 3, 4]), keep).tolist() == [0, 2, 4]


def test_collision_rule_with_no_mask_is_the_IDENTITY():
    """The unmasked path must stay reachable and must not perturb anything --
    the pooled arm's BIT-identity regression (AC1) depends on it."""
    displaced = np.array([3, 1, 4, 1, 5])
    assert apply_collision_rule(displaced, None).tolist() == displaced.tolist()


def test_collision_rule_can_empty_a_draw_entirely():
    """The precondition for BH1: every positive landing on a masked row."""
    assert apply_collision_rule(np.array([0, 1]),
                                np.array([False, False, True])).size == 0


# ================================================ BH4: exceedance resolution ==

def test_exceedance_resolution_matches_the_two_values_BH4_names():
    """1 / (surviving draws x features + 1): the uniform value at 20 surviving
    shifts, and the coarser one at the single dropped draw's 19."""
    assert exceedance_resolution(20, 10240) == pytest.approx(1 / 204801, rel=1e-15)
    assert exceedance_resolution(19, 10240) == pytest.approx(1 / 194561, rel=1e-15)
    ratio = exceedance_resolution(19, 10240) / exceedance_resolution(20, 10240)
    assert ratio == pytest.approx(1.0526, abs=1e-4)      # 5.26% coarser


def test_a_concept_with_no_surviving_draw_has_NO_resolution_not_a_number():
    """NOT_EVALUABLE is not NOT_DETECTED. A concept with no reference
    distribution must not carry a finite resolution a consumer could fold
    into a denominator."""
    assert np.isnan(exceedance_resolution(0, 10240))


# ==================================================== BH3: the outcome grid ==

@pytest.mark.parametrize("na,nb,expected", [
    (0, 0, BH3_NONE),
    (1, 0, BH3_SOME),
    (0, 1, BH3_SOME),
    (19, 19, BH3_SOME),
    (20, 0, BH3_ONE_ARM),
    (0, 20, BH3_ONE_ARM),
    (20, 19, BH3_ONE_ARM),
    (20, 20, BH3_BOTH_ARMS),
    (None, 20, BH3_ONE_ARM),
    (None, 0, BH3_NONE),
    (None, 3, BH3_SOME),
    (None, None, BH3_NO_ARM),
])
def test_bh3_grid_is_a_TOTAL_function_of_the_degenerate_counts(na, nb, expected):
    """Gotcha 24: a pre-declared rule must be defined over its statistic's FULL
    domain, not over the region the data are expected to land in. Two of this
    project's rules have already been well-formed and INAPPLICABLE.

    The `None` rows are what matter beyond bookkeeping: an arm that was not
    requested is reported ABSENT, never as 0 -- 0 asserts the arm ran and
    produced no degenerate draw, which is a different claim (I3).
    """
    assert classify_bh3(na, nb, n_shifts=20) == expected


def test_bh3_distinguishes_an_ABSENT_arm_from_an_arm_with_zero_degenerate_draws():
    """THE discriminating pair is (None, None) against (0, 0), and the first
    draft of this test did not contain it.

    `classify_bh3(None, 20)` and `classify_bh3(0, 20)` both return ONE_ARM, so
    a mutation reading an absent arm as zero passes them both -- which is what
    the mutation matrix measured. The pair below is the only one where the two
    readings diverge: no arm requested is NO_ARM, both arms requested and
    clean is NONE_DEGENERATE. Naming a test after a distinction it cannot
    exhibit is gotcha 16's fixture lesson in its most literal form.
    """
    assert classify_bh3(None, None, 20) == BH3_NO_ARM
    assert classify_bh3(0, 0, 20) == BH3_NONE
    assert classify_bh3(None, None, 20) != classify_bh3(0, 0, 20)
    # ...and one arm fully degenerate is not the STRATUM verdict.
    assert classify_bh3(None, 20, 20) == BH3_ONE_ARM
    assert classify_bh3(20, 20, 20) == BH3_BOTH_ARMS
    assert classify_bh3(20, 20, 20) != classify_bh3(None, 20, 20)


# ============================================== BG1/BG2: the displacement plan ==

def _plan_fixture():
    index = _index([5, 5])                 # proteins at [0,5) and [5,10)
    return (build_row_to_protein(index, n_rows=10),
            *protein_starts_lengths(index))


def test_the_plan_holds_the_MASK_FIXED_and_displaces_only_the_POSITIVES():
    """BG1. Concept positives {0}, mask {1, 2}. Under a shift of 1 the
    positive moves 0 -> 1, which is masked, so the draw is empty; under a
    shift of 3 it moves 0 -> 3 and survives. Worked out from the
    circular-shift definition, not from the plan builder."""
    row2protein, starts, lengths = _plan_fixture()
    keep = np.ones(10, dtype=bool)
    keep[[1, 2]] = False
    plan = build_displacement_plan(
        ["C"], {"C": np.array([0])}, {"C": keep},
        row2protein, starts, lengths,
        np.array([[1, 0], [3, 0]]), None, n_shifts=2)

    assert plan["b"] is None                       # arm not requested
    assert plan["a"]["counts"].tolist() == [[0, 1]]
    assert plan["a"]["degenerate"].tolist() == [[True, False]]
    assert plan["a"]["rows"]["C"][1].tolist() == [3]


def test_a_draw_with_ONE_surviving_positive_is_NOT_degenerate():
    """BH1 explicitly. AP is defined at one positive -- the reciprocal of that
    positive's rank -- so no rule applies and the draw is kept. The real study
    has 20 such cells under Null A and 22 under Null B; treating them as
    degenerate would silently discard them."""
    row2protein, starts, lengths = _plan_fixture()
    keep = np.ones(10, dtype=bool)
    keep[[2, 3, 4]] = False
    # positives {0,1}: shift 1 -> {1,2}, of which 2 is masked (one survives);
    #                  shift 2 -> {2,3}, both masked (degenerate).
    plan = build_displacement_plan(
        ["C"], {"C": np.array([0, 1])}, {"C": keep},
        row2protein, starts, lengths,
        np.array([[1, 0], [2, 0]]), None, n_shifts=2)
    assert plan["a"]["counts"].tolist() == [[1, 0]]
    assert plan["a"]["degenerate"].tolist() == [[False, True]]


def test_the_plan_is_the_IDENTITY_on_survival_when_no_mask_is_active():
    """Null A's circular shift is a bijection within a protein, so with no mask
    every positive survives every draw -- the property that lets the pooled arm
    go on using the concept's own positive count."""
    row2protein, starts, lengths = _plan_fixture()
    plan = build_displacement_plan(
        ["C"], {"C": np.array([0, 1, 2])}, {"C": None},
        row2protein, starts, lengths,
        np.array([[1, 0], [4, 0], [0, 0]]), None, n_shifts=3)
    assert plan["a"]["counts"].tolist() == [[3, 3, 3]]
    assert not plan["a"]["degenerate"].any()


def test_null_B_dedups_BEFORE_the_collision_rule_so_counts_are_realised():
    """Null B's rescaling onto a shorter destination protein is many-to-one, so
    two positives can land on one row. The realised positive count for the draw
    is what gain-normalisation must use -- the concept's own count would be too
    large, inflating the denominator of a draw that never had that many
    distinct rows."""
    index = _index([4, 2])                 # P0 [0,4), P1 [4,6) -- shorter
    row2protein = build_row_to_protein(index, n_rows=6)
    starts, lengths = protein_starts_lengths(index)
    # every P0 row relocates onto P1: offsets 0,1,2,3 -> floor(o*2/4) = 0,0,1,1
    dest = np.array([[1, 1]])
    plan = build_displacement_plan(
        ["C"], {"C": np.array([0, 1, 2, 3])}, {"C": None},
        row2protein, starts, lengths, None, dest, n_shifts=1)
    assert plan["a"] is None
    assert plan["b"]["rows"]["C"][0].tolist() == [4, 5]
    assert plan["b"]["counts"].tolist() == [[2]]        # 4 positives -> 2 rows
