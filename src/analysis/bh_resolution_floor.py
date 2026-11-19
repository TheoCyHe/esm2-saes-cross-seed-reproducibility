"""The BH RESOLUTION FLOOR, as a relation computable from design parameters
alone (PREREGISTRATION amendment BP, part A).

WHAT THIS IS FOR. A permutation-null Benjamini-Hochberg pass has a smallest
attainable p-value, `1/(n_finite + 1)`. If that atom is still larger than BH's
critical value at rank 1, the procedure CANNOT REJECT ANYTHING no matter what
the data say. How many tests must sit exactly at that floor before the first
rejection is possible is a function of THREE NUMBERS -- pool size, shift count,
level -- AND NO DATA AT ALL.

THIS IS A CHARACTERISATION, NOT A STANDARD. It states a relation. It does NOT
name a fraction a design must reach, and no threshold here is a bar any design
passes or fails. `k_min` is simply what the design can resolve; whether that is
adequate is a judgement about the design's purpose, made elsewhere.

---------------------------------------------------------------------------
THE RELATION
---------------------------------------------------------------------------
With `C` concepts in the pool, `F` features, `S` surviving shifts, level `q`:

    family size      m       = C * F
    finest p-value   p_min   = 1 / (S*F + 1)
    BH critical val  crit_i  = (i/m) * q
    an atom `p` is rejectable only at ranks  i >= p*m/q

so the smallest rejectable rank at the floor is

    k_min = ceil( C*F / (q * (S*F + 1)) )
          = ceil( (C/(q*S)) * (S*F/(S*F + 1)) )

`F` DOES NOT CANCEL ALGEBRAICALLY -- it cancels THROUGH THE CEILING, because
the second factor is strictly below 1 and the shortfall `(C/(q*S))/(S*F+1)` is
tiny. That distinction matters: the identity `k_min == C/(q*S)` is exact only
while the shortfall stays below 1, which is `C < S*F + 1`. See `k_min_equals_C`.

THE SUPPORT IS BLOCKED. Attainable p-values are `j/(S*F+1)` for integer `j`, so
the j-th atom becomes rejectable at `ceil(j*C*F/(q*(S*F+1)))`, which equals
`j*C` while `j*C < S*F + 1` and DIVERGES ABOVE THAT. See `block_rank`.

DEGENERATE DRAWS RAISE A CONCEPT'S FLOOR, THEY DO NOT LOWER IT. A concept with
`S_c < S` surviving draws has a LARGER floor atom `1/(S_c*F+1)` and therefore a
HIGHER rank threshold -- it is harder to reject, not easier. A family whose
concepts have different `S_c` therefore has SEVERAL floors with several rank
thresholds, which is why `k_from_atoms` takes a histogram rather than a single
"at-floor count": one number cannot express a set whose members have different
thresholds (gotcha 39 -- enumerate in the primitive the data actually has).

Exact arithmetic throughout via `fractions.Fraction`, so that no conclusion
here rests on a float rounding.
"""

from __future__ import annotations

import math
from fractions import Fraction

__all__ = [
    "k_min",
    "k_min_equals_C",
    "block_rank",
    "atom_rank_threshold",
    "k_from_atoms",
]


def _fr(x) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x).limit_denominator(10**9)


def k_min(n_concepts: int, n_features: int, n_shifts: int, q) -> int:
    """Smallest BH rank at which the FINEST attainable p-value is rejectable.

    Below this many floor-tied tests the procedure rejects nothing at all,
    whatever the data say. Exact; no floating point.
    """
    return atom_rank_threshold(1, n_concepts, n_features, n_shifts, q,
                               n_shifts_concept=n_shifts)


def atom_rank_threshold(j: int, n_concepts: int, n_features: int,
                        n_shifts: int, q, n_shifts_concept: int | None = None
                        ) -> int:
    """Smallest rank at which the atom `j/(S_c*F+1)` is rejectable.

    `n_shifts` sizes the FAMILY (m = C*F); `n_shifts_concept` sizes the atom's
    own null pool and defaults to `n_shifts`. They differ exactly when that
    concept lost draws to degeneracy -- which RAISES its threshold.
    """
    S_c = n_shifts if n_shifts_concept is None else n_shifts_concept
    m = n_concepts * n_features
    p = Fraction(j, S_c * n_features + 1)
    return math.ceil(p * m / _fr(q))


def block_rank(j: int, n_concepts: int, n_features: int, n_shifts: int,
               q) -> tuple[int, int, bool]:
    """`(threshold, j*C, they_agree)` for the j-th atom of a uniform family.

    The third element is the point: `j*C` is a convenient shorthand that stops
    being true above `j*C >= S*F + 1`, and a caller relying on the shorthand
    past that boundary gets a wrong answer silently.
    """
    thr = atom_rank_threshold(j, n_concepts, n_features, n_shifts, q)
    return thr, j * n_concepts, thr == j * n_concepts


def k_min_equals_C(n_concepts: int, n_features: int, n_shifts: int,
                   q) -> bool:
    """Does the floor equal the pool size exactly?

    True iff `q*S == 1` and `C < S*F + 1`. Stated as a computed predicate
    rather than as prose so it can be checked on BOTH SIDES of its boundary
    (gotcha 25: a check that can only pass is not a check).
    """
    return k_min(n_concepts, n_features, n_shifts, q) == n_concepts


def k_from_atoms(atoms: list[tuple[Fraction, int]], m: int, q) -> int:
    """BH's rejection count from the P-VALUE ATOM HISTOGRAM ALONE.

    `atoms` is `[(p_value, count), ...]`, any order; `m` is the family size.
    Returns the largest `i` with `p_(i) <= (i/m)*q` -- BH's step-up rule
    evaluated without ever touching the individual p-values.

    WHY A HISTOGRAM AND NOT A COUNT. Within a block of equal p-values the
    condition is monotone in `i`, so only the block's TOP rank can be the
    crossing. Taking the max over blocks is therefore exact. A single
    "at-floor count" cannot do this when the family has more than one floor.

    -----------------------------------------------------------------------
    THE MULTI-FLOOR PATH IS LIVE AND CORRECT, AND IT HAS NEVER BITTEN.
    READ THIS BEFORE CITING "held 40/40" AS EVIDENCE THAT IT WORKS.
    -----------------------------------------------------------------------
    On the experimental stratum the family has TWO interleaved floors: 218 for
    the full-denominator concepts and 230 for `BINDING:hypoxanthine`, whose 19
    surviving draws give it the coarser atom `1/194,561`. BP checked the rule in
    all 40 (seed, arm) cells and it agreed with the realised `n_rejected` in
    every one -- **but the agreement is not evidence about the two-floor path**,
    because the discriminating count `A19` (features whose observed gain exceeds
    ALL 194,560 surviving hypoxanthine draws) is **ZERO in every arm-B cell**.
    With `A19 == 0` the second block is empty, `T == A20` identically, and the
    single-floor and two-floor branches return the SAME number. They were never
    separated by any datum. The `2_SINGLE_ATOM` / `1_TWO_ATOM` occupancy split
    (5 / 12 cells) is a partition of the at-floor count with no consequence for
    any rejection count -- it is NOMINAL (gotcha 25b: a predicate applied to a
    constant ranks only the tie-break; report UNINFORMATIVE, never a pass).

    UNEXERCISED IS NOT IDLE. Five of the twenty arm-B seeds sit in the window
    `T in [218, 230)` where the two thresholds separate -- seeds 5, 2, 1, 0, 7
    at T = 221, 223, 224, 225, 225. One floor-tied hypoxanthine test in any of
    them and the single-floor answer would have been wrong there. The nearest
    approach to the upper floor is T = 225, **short of 230 by five tests**
    (measured from `bp_floor_check.json`, not carried: the BQ brief states this
    margin as fifteen, which does not reproduce -- see amendment BQ).

    SO: do not construct a forcing test for this path, and do not report it as
    validated. It is one non-zero count away from mattering, and any change to
    `n_shifts`, the pool, or the degenerate-draw population could make it bite
    (PREREGISTRATION BP5b, BP10 item 2; amendment BQ part A3).
    """
    if not atoms:
        return 0
    qq = _fr(q)
    cum = 0
    best = 0
    for p, n in sorted(atoms, key=lambda t: t[0]):
        if n <= 0:
            continue
        cum += n
        if _fr(p) <= Fraction(cum, m) * qq:
            best = cum
    return best
