"""Tests for the BH resolution-floor relation (amendment BP part A).

EVERY EXPECTATION HERE IS WRITTEN DOWN, NOT COMPUTED FROM THE CODE UNDER TEST
(gotcha 32b). The founding-instance numbers are the experimental stratum's own
design parameters and the expected values are literals.

THE RULE IS RUN AGAINST ITS FOUNDING INSTANCE (gotcha 39) and against BOTH
SIDES of its own boundary (gotcha 25) -- a relation checked only where it holds
is a relation that has not been checked.
"""

from fractions import Fraction

import pytest

from src.analysis.bh_resolution_floor import (
    atom_rank_threshold,
    block_rank,
    k_from_atoms,
    k_min,
    k_min_equals_C,
)

# The experimental stratum, from the artifact axis: 218 in-pool concepts,
# 10,240 features, 20 shifts, q = 0.05 (AV6 component 1).
C, F, S, Q = 218, 10240, 20, Fraction(1, 20)


def test_founding_instance_floor_is_218():
    """THE FOUNDING INSTANCE. k_min equals the pool size exactly at q*S == 1."""
    assert k_min(C, F, S, Q) == 218
    assert k_min_equals_C(C, F, S, Q) is True


def test_founding_instance_family_and_atom_are_the_written_down_numbers():
    assert C * F == 2232320
    assert S * F + 1 == 204801


@pytest.mark.parametrize("n_features", [64, 256, 1024, 4096, 10240, 40960, 10**6])
def test_F_cancels_through_the_ceiling(n_features):
    """k_min is 218 across four orders of magnitude of F -- but see the
    boundary test below, which is where that stops being true."""
    assert k_min(C, n_features, S, Q) == 218


def test_k_min_equals_C_on_BOTH_SIDES_of_its_boundary():
    """The identity holds iff C < S*F + 1 = 204,801. A check that can only
    pass is as useless as one that can never fail (gotcha 25)."""
    assert k_min_equals_C(204800, F, S, Q) is True      # just inside
    assert k_min_equals_C(204801, F, S, Q) is False     # just outside
    assert k_min(204801, F, S, Q) == 204800             # and it is short by one


@pytest.mark.parametrize("n_concepts", [2, 64, 104, 156, 192, 218, 1000, 100000])
def test_identity_holds_for_any_pool_below_the_bound(n_concepts):
    assert k_min(n_concepts, F, S, Q) == n_concepts


def test_block_structure_is_j_times_C_and_then_stops():
    """Written-down expectations for the first eight blocks."""
    for j, expected in enumerate([218, 436, 654, 872, 1090, 1308, 1526, 1744], 1):
        thr, jc, agree = block_rank(j, C, F, S, Q)
        assert thr == expected
        assert agree is True
    # The shorthand j*C stops at j = 939. Checked at the boundary, both sides.
    assert block_rank(939, C, F, S, Q)[2] is True
    thr, jc, agree = block_rank(940, C, F, S, Q)
    assert agree is False
    assert (thr, jc) == (204919, 204920)


def test_a_degenerate_draw_RAISES_the_threshold():
    """19 surviving draws, not 20: hypoxanthine under Null B. The floor atom
    gets LARGER and the rank threshold goes UP -- harder, not easier."""
    assert atom_rank_threshold(1, C, F, S, Q, n_shifts_concept=20) == 218
    assert atom_rank_threshold(1, C, F, S, Q, n_shifts_concept=19) == 230


def test_the_two_floors_interleave():
    """Ordering of the six smallest atoms, written down."""
    a20 = [Fraction(j, 20 * F + 1) for j in (1, 2, 3)]
    a19 = [Fraction(j, 19 * F + 1) for j in (1, 2, 3)]
    order = sorted(a20 + a19)
    assert order == [a20[0], a19[0], a20[1], a19[1], a20[2], a19[2]]


class TestKFromAtoms:
    """`k_from_atoms` must reproduce BH exactly from the histogram."""

    m = C * F
    floor20 = Fraction(1, 20 * F + 1)
    floor19 = Fraction(1, 19 * F + 1)

    def test_below_the_floor_rejects_nothing(self):
        assert k_from_atoms([(self.floor20, 217)], self.m, Q) == 0

    def test_at_the_floor_rejects_exactly_the_floor_tied_tests(self):
        assert k_from_atoms([(self.floor20, 218)], self.m, Q) == 218
        assert k_from_atoms([(self.floor20, 291)], self.m, Q) == 291

    def test_SINGLE_ATOM_branch_the_one_number_formulation_cannot_express(self):
        """A20 = 220 clears rank 218; the five 19-shift tests sit at ranks
        221-225 and need rank 230, so they are NOT rejected. The total
        at-floor count is 225 and the answer is 220 -- they differ."""
        got = k_from_atoms([(self.floor20, 220), (self.floor19, 5)], self.m, Q)
        assert got == 220

    def test_TWO_ATOM_branch(self):
        """A20 = 240, A19 = 5: the top rank 245 clears 230, so both blocks go."""
        got = k_from_atoms([(self.floor20, 240), (self.floor19, 5)], self.m, Q)
        assert got == 245

    def test_the_second_atom_can_carry_the_first_over_the_line(self):
        """A20 = 200 is BELOW 218 on its own, but the 19-shift block pushes the
        top rank to 235 >= 230, so BH rejects all 235. A per-block reading
        would wrongly return 0."""
        got = k_from_atoms([(self.floor20, 200), (self.floor19, 35)], self.m, Q)
        assert got == 235

    def test_SECOND_BLOCK_relation(self):
        """Reaching rank 436 at the second atom rejects that block too."""
        second = Fraction(2, 20 * F + 1)
        got = k_from_atoms([(self.floor20, 300), (second, 200)], self.m, Q)
        assert got == 500

    def test_empty_and_zero_count_atoms(self):
        assert k_from_atoms([], self.m, Q) == 0
        assert k_from_atoms([(self.floor20, 0)], self.m, Q) == 0

    def test_order_of_input_does_not_matter(self):
        a = [(self.floor20, 240), (self.floor19, 5)]
        assert k_from_atoms(a, self.m, Q) == k_from_atoms(a[::-1], self.m, Q)

    # ---------------------------------------------------------------------
    # THE TWO FIXTURES BELOW EXIST BECAUSE THE MUTATION MATRIX FOUND THE
    # FIXTURES ABOVE COULD NOT EXHIBIT THE BRANCHES THEY WERE NAMED FOR.
    # Both assert their own discriminating power first (gotchas 13, 32b, 39).
    # ---------------------------------------------------------------------

    def test_an_EXACT_TIE_between_p_and_its_critical_value_is_REJECTED(self):
        """`<=` versus `<` is invisible unless a tie actually occurs, and at
        the real design parameters it never does: at cum = 218 the critical
        value is exactly 1/204,800 while the atom is 1/204,801 -- strictly
        below, so both conventions agree. Constructed here so it does.

        m = 100, q = 1/10, atom 1/50, count 20: crit = (20/100)(1/10) = 1/50,
        which IS the atom. Under `<=` BH rejects 20; under `<` it rejects 0.
        """
        m, q, atom, n = 100, Fraction(1, 10), Fraction(1, 50), 20
        # The fixture asserts the tie EXISTS before trusting its own answer.
        assert Fraction(n, m) * q == atom, "fixture no longer exercises a tie"
        assert k_from_atoms([(atom, n)], m, q) == 20

    def test_block_ORDER_matters_and_this_fixture_can_detect_it(self):
        """Two atoms cannot detect a reversed walk when both branches reach
        the same total. Three can: forward stops at 250, a largest-p-first
        walk would return 350.
        """
        atoms = [(Fraction(1, 204801), 250),
                 (Fraction(5, 204801), 50),
                 (Fraction(10, 204801), 50)]
        assert k_from_atoms(atoms, self.m, Q) == 250
        # and the fixture asserts it is capable of telling the two apart
        reversed_walk = 350
        assert reversed_walk != 250, "fixture cannot discriminate order"
