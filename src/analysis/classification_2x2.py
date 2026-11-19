"""COMPONENT 3 (PREREGISTRATION amendment BQ): the 2x2 CLASSIFICATION at
threshold 1 -- ALIGNED / ORPHAN / EXCLUDED, plus `NOT_EVALUABLE`.

This is the classification result. It consumes component 1 and 2's experimental
Benjamini-Hochberg artifact, runs the SAME two components over the POOLED raw
null for the ORPHAN side, applies AV6's second conjunct (the raw gain floor) to
BOTH sides, and reduces the pair of booleans to AV5's grid.

---------------------------------------------------------------------------
THE UNIT IS A (SEED, FEATURE) PAIR. BM-E1 CRITERION 5 NAMES THE WRONG PRIMITIVE.
---------------------------------------------------------------------------
Criterion 5 says "every (CONCEPT, seed, grid-point) triple". The classified
object is a FEATURE: AV4 defines both sides with the concept axis QUANTIFIED
AWAY -- "clears the bar against ANY concept", "fails the bar against EVERY
concept" -- and AV5's worked mechanism is a feature. So the population is
`n_anchors x n_features = 20 x 10,240 = 204,800` pairs per arm, not 218 x 20.
Criterion 5 is committed text and stands as written; BQ4a is the correction and
criterion 5's REQUIREMENT is implemented here in full, in the right primitive.

---------------------------------------------------------------------------
THE BAR, AND THE ORDER OF ITS TWO CONJUNCTS -- WHICH IS LOAD-BEARING.
---------------------------------------------------------------------------
AV6's bar has two components, ANDed PER (feature, concept) BEFORE the existential
over concepts:

    E(s,f) = ANY_c [ reject_exp[s,f,c]  AND  G_exp[s,f,c]  >= 0.05 ]
    P(s,f) = ANY_c [ reject_pool[s,f,c] AND  G_pool[s,f,c] >= 0.05 ]

The other order is WRONG and looks identical: a feature that is FDR-significant
against concept c1 and clears the gain floor against a DIFFERENT concept c2 has
not cleared the bar against anything. `clears_bar` takes the two arrays and does
the AND itself precisely so a caller cannot reduce them separately.

`G = (AP - p)/(1 - p)` with `p` each stratum's OWN per-concept prevalence against
the effective row count that stratum's AP was computed against -- C4's invariant,
never a reused global `p`.

---------------------------------------------------------------------------
THE FIVE OUTCOMES (BQ4c), exhaustive and mutually exclusive.
---------------------------------------------------------------------------
    ALIGNED_BOTH    E and P          AV5 clears-exp / clears-pooled
    ALIGNED_EXP     E and not P      clears-exp / fails-pooled -- the cell that
                                     would otherwise be invisible, folded into
                                     ALIGNED by precedence and reported anyway
    EXCLUDED        not E and P      the buffer between the arms
    ORPHAN          not E and not P
    NOT_EVALUABLE   EITHER pool has ZERO evaluable concepts for that (seed, arm)

NOT_EVALUABLE is tested FIRST and is disjoint from the four cells by
construction. It exists because AV5's grid presupposes both axes are MEASURED,
and an existential over an empty pool returns False -- indistinguishable from a
measured failure. It is carried even when empty (criterion 6).

ALIGNED's precedence is NOT a tie-break: two cells share one label, exactly as
AV5 writes it, and no rule chooses between them.

---------------------------------------------------------------------------
CENSORING (BP4) IS ORTHOGONAL TO THE CELLS, AND THE TWO ARE NEVER MERGED.
---------------------------------------------------------------------------
`floor_limited` is a property of a (SEED, arm): its at-floor count is below that
stratum's `k_min`, so BH rejects nothing and that axis is False for every feature
BY CONSTRUCTION rather than by measurement. `NOT_EVALUABLE` is a property of a
CONCEPT. BP4 item 4 forbids merging, summing, or reporting them in one column.

The floors are computed from design parameters alone, before any data
(gotcha 42, via `bh_resolution_floor`):

    experimental  C=218  m=2,232,320  k_min=218
    pooled        C=703  m=7,198,720  k_min=703

both because `q*S = 0.05*20 = 1` exactly and `C < S*F+1 = 204,801`.

---------------------------------------------------------------------------
A DEVICE-PROVENANCE FACT ABOUT THE TWO AXES, REPORTED NOT REPAIRED.
---------------------------------------------------------------------------
The experimental axis descends from `feature_ap_experimental_av_cpu.h5` (a CPU
gather, array 21712883) and the pooled axis from `feature_ap_main_compliant.h5`
(a GPU-era artifact). Their `fire_counts` differ, max absolute difference 23 --
gotcha 36's measured CPU/GPU signature, not a defect here. Gotcha 36's rule binds
a statistic to ITS OWN null and each axis is internally matched; the 2x2 combines
two BOOLEANS rather than two numbers on a shared scale. Recorded in the artifact
(`axis_device_divergence_max_fire_count`) so no reader has to rediscover it.
NOTHING IS BACK-FILLED: device keys absent from a source stay absent here.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np

# AV6 component 2, ratified unchanged (AV6). Not a tunable.
GAIN_FLOOR = 0.05

# The outcome vocabulary. Integer codes so the artifact stores one small array
# rather than a string per pair; the names are the record.
CELL_ALIGNED_BOTH = 0
CELL_ALIGNED_EXP = 1
CELL_EXCLUDED = 2
CELL_ORPHAN = 3
CELL_NOT_EVALUABLE = 4

CELL_NAMES = {
    CELL_ALIGNED_BOTH: "ALIGNED_clears_both",
    CELL_ALIGNED_EXP: "ALIGNED_clears_experimental_only",
    CELL_EXCLUDED: "EXCLUDED_clears_pooled_only",
    CELL_ORPHAN: "ORPHAN_clears_neither",
    CELL_NOT_EVALUABLE: "NOT_EVALUABLE",
}

# The three-way groups AV5's precedence rule produces from the four cells.
GROUP_OF_CELL = {
    CELL_ALIGNED_BOTH: "ALIGNED",
    CELL_ALIGNED_EXP: "ALIGNED",
    CELL_EXCLUDED: "EXCLUDED",
    CELL_ORPHAN: "ORPHAN",
    CELL_NOT_EVALUABLE: "NOT_EVALUABLE",
}

from src.analysis.bh_resolution_floor import k_min  # noqa: E402
from src.analysis.classification_bh import (  # noqa: E402
    Q_FDR,
    bh_pass_one_slab,
)
from src.sae.train import git_dirty, git_sha  # noqa: E402

__all__ = [
    "GAIN_FLOOR",
    "CELL_ALIGNED_BOTH",
    "CELL_ALIGNED_EXP",
    "CELL_EXCLUDED",
    "CELL_ORPHAN",
    "CELL_NOT_EVALUABLE",
    "CELL_NAMES",
    "GROUP_OF_CELL",
    "gain_from_ap",
    "clears_bar",
    "classify_cells",
    "verify_outcome_space",
    "at_floor_counts",
    "recover_seed_position",
    "recover_raw_slab_seed",
    "OutcomeSpaceViolation",
    "SeedOrderUnrecoverable",
]


class SeedOrderUnrecoverable(Exception):
    """Raised when a raw slab cannot be matched to exactly one labelled seed.

    FATAL. The pooled raw null stores NO seed labels, so its seed axis has to be
    recovered from the data rather than read (gotcha 29: where labels are
    absent, recover the order and make the recovery able to fail). Zero matches
    means the reductions do not describe this array; more than one means the
    recovery cannot discriminate and must not be trusted to.
    """


class OutcomeSpaceViolation(Exception):
    """Raised when the five outcomes do not partition the population.

    FATAL by criterion 5. An exception rather than a printed warning because a
    partition that does not partition invalidates every count downstream of it,
    and a diagnostic that states a conclusion cannot disagree with its author
    (gotcha 20).
    """


def gain_from_ap(ap: np.ndarray, prevalence: np.ndarray) -> np.ndarray:
    """AV6's raw gain `G = (AP - p)/(1 - p)`, elementwise over concepts.

    `ap` is (n_features, n_concepts); `prevalence` is (n_concepts,), each
    concept's own `p` against the effective row count ITS AP was computed
    against (C4). Broadcasting is along the concept axis, and a shape mismatch
    raises rather than broadcasting into a plausible wrong answer.
    """
    ap = np.asarray(ap, dtype=np.float64)
    p = np.asarray(prevalence, dtype=np.float64)
    if ap.ndim != 2 or p.shape != (ap.shape[1],):
        raise ValueError(
            f"gain_from_ap: ap is {ap.shape}, prevalence is {p.shape}; "
            f"expected prevalence to be ({ap.shape[1]},) -- a transposed or "
            "recycled prevalence would broadcast into a plausible wrong gain")
    if np.any(p >= 1.0) or np.any(p < 0.0):
        raise ValueError("prevalence outside [0, 1): 1-p would be <= 0")
    return (ap - p[None, :]) / (1.0 - p[None, :])


def clears_bar(reject: np.ndarray, gain: np.ndarray,
               floor: float = GAIN_FLOOR) -> np.ndarray:
    """AV6's FULL bar, reduced over concepts: does the feature clear ANY?

    `reject` and `gain` are both (n_features, n_concepts). Returns
    (n_features,) bool.

    THE AND HAPPENS BEFORE THE ANY, AND THIS FUNCTION OWNS THAT ORDER. Reducing
    the two conjuncts separately -- `reject.any(1) & (gain >= floor).any(1)` --
    is a different and wrong predicate: it is satisfied by a feature significant
    against one concept and gainful against another. The two forms agree on most
    data, which is exactly why the wrong one survives inspection.
    """
    reject = np.asarray(reject)
    gain = np.asarray(gain)
    if reject.shape != gain.shape:
        raise ValueError(
            f"clears_bar: reject is {reject.shape}, gain is {gain.shape} -- "
            "the two conjuncts must be on the same (feature, concept) grid")
    if reject.dtype != np.bool_:
        raise ValueError("clears_bar: `reject` must be boolean, not a p-value "
                         "or an indicator in another dtype")
    return np.any(reject & (gain >= floor), axis=1)


def classify_cells(clears_exp: np.ndarray, clears_pooled: np.ndarray,
                   exp_pool_evaluable: bool,
                   pooled_pool_evaluable: bool) -> np.ndarray:
    """AV5's grid for one (seed, arm). Returns (n_features,) uint8 cell codes.

    `clears_exp` / `clears_pooled` are (n_features,) bool. The two flags say
    whether that stratum's pool contains AT LEAST ONE evaluable concept for this
    (seed, arm); if either is False the existential ranged over an empty set and
    the pair is NOT_EVALUABLE -- not ORPHAN, which would read an absence of
    reference distribution as a measured failure (criterion 6).
    """
    e = np.asarray(clears_exp, dtype=bool)
    p = np.asarray(clears_pooled, dtype=bool)
    if e.shape != p.shape or e.ndim != 1:
        raise ValueError(
            f"classify_cells: clears_exp {e.shape}, clears_pooled {p.shape} -- "
            "both must be 1-D over the SAME feature axis")

    out = np.empty(e.shape, dtype=np.uint8)
    if not (exp_pool_evaluable and pooled_pool_evaluable):
        out[:] = CELL_NOT_EVALUABLE
        return out
    out[e & p] = CELL_ALIGNED_BOTH
    out[e & ~p] = CELL_ALIGNED_EXP
    out[~e & p] = CELL_EXCLUDED
    out[~e & ~p] = CELL_ORPHAN
    return out


def verify_outcome_space(codes: np.ndarray, clears_exp: np.ndarray,
                         clears_pooled: np.ndarray, exp_pool_evaluable: bool,
                         pooled_pool_evaluable: bool) -> dict:
    """CRITERION 5, checked in the direction that can fail.

    Re-derives the five masks FROM THE SPECIFICATION -- not from `codes` -- then
    asserts (a) they are pairwise disjoint, (b) their union covers every pair,
    (c) the counts sum to the population, and (d) they agree with `codes`
    elementwise. Deriving the masks from `codes` would make (a)-(c) true by
    construction, which is gotcha 32b: a test that computes its expectation from
    the thing under test cannot detect a change to that thing.

    Raises `OutcomeSpaceViolation` on any failure. Returns the per-cell counts.
    """
    e = np.asarray(clears_exp, dtype=bool)
    p = np.asarray(clears_pooled, dtype=bool)
    n = int(e.size)

    if exp_pool_evaluable and pooled_pool_evaluable:
        masks = {
            CELL_ALIGNED_BOTH: e & p,
            CELL_ALIGNED_EXP: e & np.logical_not(p),
            CELL_EXCLUDED: np.logical_not(e) & p,
            CELL_ORPHAN: np.logical_not(e) & np.logical_not(p),
            CELL_NOT_EVALUABLE: np.zeros(n, dtype=bool),
        }
    else:
        masks = {c: np.zeros(n, dtype=bool) for c in CELL_NAMES}
        masks[CELL_NOT_EVALUABLE] = np.ones(n, dtype=bool)

    keys = sorted(masks)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            overlap = int(np.count_nonzero(masks[a] & masks[b]))
            if overlap:
                raise OutcomeSpaceViolation(
                    f"{overlap} pairs are in BOTH {CELL_NAMES[a]} and "
                    f"{CELL_NAMES[b]}: the outcomes are not mutually exclusive")

    covered = np.zeros(n, dtype=bool)
    for m in masks.values():
        covered |= m
    if not covered.all():
        raise OutcomeSpaceViolation(
            f"{int((~covered).sum())} of {n} pairs land in NO outcome: the "
            "enumeration is not exhaustive")

    counts = {c: int(np.count_nonzero(masks[c])) for c in keys}
    total = sum(counts.values())
    if total != n:
        raise OutcomeSpaceViolation(
            f"cell counts sum to {total}, population is {n}")

    codes = np.asarray(codes)
    for c in keys:
        if not np.array_equal(masks[c], codes == c):
            bad = int(np.count_nonzero(masks[c] ^ (codes == c)))
            raise OutcomeSpaceViolation(
                f"the classifier disagrees with the specification on "
                f"{bad} pairs for {CELL_NAMES[c]}")
    return counts


def recover_seed_position(slab_max_over_concepts: np.ndarray,
                          null_g: np.ndarray) -> int:
    """WHICH LABELLED SEED IS THIS RAW SLAB? Recovered from the data.

    `null_gain_raw_compliant.h5` has NO `seeds` dataset -- only `concepts`,
    `gain_a` and `gain_b` -- so its seed axis carries no labels to join on.
    Gotcha 29 is explicit about this case: recover the order FROM THE DATA and
    make the recovery ABLE TO FAIL. Assuming the raw array's seed axis matches
    its reductions' would be exactly the positional join that gotcha forbids,
    and 18 of 20 rows disagree when it is wrong.

    THE RECOVERY. `null_pilot_compliant.h5`'s `null_g_*` is the maximum over
    CONCEPTS of this very array (its own `derived_from_raw` attr names this
    file), so the max-over-concepts of raw slab `s` must equal exactly one row
    of `null_g_*`. `slab_max_over_concepts` is (n_shifts, n_features) and
    `null_g` is (n_seeds, n_shifts, n_features).

    Equality is EXACT, not approximate: `null_g` is a max over stored float32
    values held in float64, so a max recomputed from the same bytes reproduces
    them exactly. A tolerance here would be a tolerance on an identity.

    Raises `SeedOrderUnrecoverable` unless exactly one row matches.
    """
    m = np.asarray(slab_max_over_concepts)
    g = np.asarray(null_g)
    if g.ndim != 3 or m.shape != g.shape[1:]:
        raise ValueError(
            f"recover_seed_position: slab max is {m.shape}, null_g is "
            f"{g.shape}; expected the slab to match null_g's trailing axes")
    matches = [i for i in range(g.shape[0]) if np.array_equal(g[i], m)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SeedOrderUnrecoverable(
            "NO labelled seed's null_g equals this slab's max over concepts. "
            "The reductions artifact does not describe this raw array, so the "
            "seed axis cannot be recovered and must not be assumed.")
    raise SeedOrderUnrecoverable(
        f"{len(matches)} labelled seeds match this slab (positions {matches}). "
        "The recovery cannot discriminate between them and is not trusted to.")


def recover_raw_slab_seed(slab: np.ndarray, null_g_arm: np.ndarray) -> int:
    """`recover_seed_position` for one raw slab (shift, concept, feature),
    compared IN THE RAW ARRAY'S STORED PRECISION (amendment CT, the CT10-17
    preamble; the same fix as `standardize_gain` at `f25d6ec`, §CT5.1-R).

    The max over concepts is taken with NaN treated as missing -- a degenerate
    draw would break the equality, and BH2 drops those draws for the same
    reason. The raw array holds float32. On the sparse GPU path `null_g` is a
    max over float32 values, so the two agree in float64; on the segmented CPU
    path (masked negatives) `null_g` keeps the float64 max while the raw array
    keeps its float32 cast, and a float64 comparison matches almost no cell.
    Rounding to nearest is monotone, so max(f32(x)) == f32(max(x)) EXACTLY:
    comparing both sides in the stored dtype is still an identity, not a
    tolerance. For a null whose `null_g` is float32-representable (every
    pooled null of record) the cast is lossless and the verdict cannot change.
    """
    slab = np.asarray(slab)
    with np.errstate(invalid="ignore"):
        slab_max = np.nanmax(slab, axis=1)
    g = np.asarray(null_g_arm)
    if slab.dtype != np.float64:
        slab_max, g = slab_max.astype(slab.dtype), g.astype(slab.dtype)
    return recover_seed_position(slab_max, g)


def at_floor_counts(p: np.ndarray, n_finite: np.ndarray,
                    not_evaluable: np.ndarray | None = None) -> int:
    """How many tests sit EXACTLY at their own concept's finest p-value.

    `p` is (n_features, n_concepts) in whatever dtype the artifact stores;
    `n_finite` is (n_concepts,). A concept's floor atom is `1/(n_finite+1)`, and
    the comparison is made in `p`'s OWN dtype -- the float64 resolution is cast
    down to meet the stored value rather than the stored value being promoted,
    so this is an exact equality and not a tolerance (corrections-log entry 123:
    a tolerance chosen after a check failed is indistinguishable from one that
    was always right).

    DEGENERATE DRAWS RAISE A CONCEPT'S FLOOR, so each concept is compared
    against its own atom and never against a family-wide constant -- the family
    genuinely has more than one floor (BP1d).
    """
    p = np.asarray(p)
    n_finite = np.asarray(n_finite, dtype=np.int64)
    if p.ndim != 2 or n_finite.shape != (p.shape[1],):
        raise ValueError(
            f"at_floor_counts: p is {p.shape}, n_finite is {n_finite.shape}")
    ne = (np.zeros(p.shape[1], dtype=bool) if not_evaluable is None
          else np.asarray(not_evaluable, dtype=bool))
    total = 0
    for c in range(p.shape[1]):
        if ne[c] or n_finite[c] <= 0:
            continue
        atom = np.asarray(1.0 / (n_finite[c] + 1.0)).astype(p.dtype)
        total += int(np.count_nonzero(p[:, c] == atom))
    return total


# --------------------------------------------------------------------------
# CLI -- the driver.
# --------------------------------------------------------------------------

def _labels(ds) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in ds[()]]


def main() -> None:
    ap_ = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--bh-experimental", type=Path, required=True,
                     help="classification_bh_experimental_bo.h5 -- components "
                          "1 and 2's output, the ALIGNED side's FDR half.")
    ap_.add_argument("--ap-experimental", type=Path, required=True)
    ap_.add_argument("--ap-pooled", type=Path, required=True,
                     help="feature_ap_main_compliant.h5 -- BJ-C1 input #1.")
    ap_.add_argument("--raw-null-pooled", type=Path, required=True,
                     help="null_gain_raw_compliant.h5 -- the pooled raw null "
                          "the ORPHAN side's BH pass runs over.")
    ap_.add_argument("--null-pooled-reductions", type=Path, required=True,
                     help="null_pilot_compliant.h5 -- BJ-C1 input #2, read for "
                          "its concept and seed axes.")
    ap_.add_argument("--q", type=float, default=Q_FDR)
    ap_.add_argument("--gain-floor", type=float, default=GAIN_FLOOR)
    ap_.add_argument("--out", type=Path, required=True)
    ap_.add_argument("--report", type=Path, required=True)
    ap_.add_argument("--seeds", type=int, default=None,
                     help="cost pilot only: process the first N seeds.")
    args = ap_.parse_args()

    # ---- the ALIGNED side, from components 1 and 2 -----------------------
    with h5py.File(args.bh_experimental, "r") as hf:
        bh_seeds = _labels(hf["seeds"])
        bh_concepts = _labels(hf["concepts"])
        reject_exp = {a: hf[f"reject_{a}"] for a in ("a", "b")}
        p_exp = {a: hf[f"p_{a}"] for a in ("a", "b")}
        nfin_exp = {a: hf[f"n_finite_{a}"][()] for a in ("a", "b")}
        notev_exp = {a: hf[f"not_evaluable_{a}"][()] for a in ("a", "b")}
        bh_identity = {k: (hf.attrs[k].decode()
                           if isinstance(hf.attrs[k], bytes) else hf.attrs[k])
                       for k in hf.attrs}
        if bool(bh_identity.get("gain_floor_applied", False)):
            raise SystemExit(
                "the experimental BH artifact reports gain_floor_applied=True. "
                "This component ANDs the floor itself; consuming a "
                "floor-applied artifact would apply AV6 component 2 TWICE.")
        reject_exp = {a: reject_exp[a][()] for a in ("a", "b")}
        p_exp = {a: p_exp[a][()] for a in ("a", "b")}

    # ---- observed gains, experimental -----------------------------------
    with h5py.File(args.ap_experimental, "r") as hf:
        g = hf["experimental"]
        inpool_e = g["in_argmax_pool"][()]
        prev_e = g["prevalence"][()][inpool_e]
        npos_e = g["n_positives"][()][inpool_e]
        nre_e = g["n_rows_effective"][()][inpool_e]
        conc_e = [c for c, k in zip(_labels(g["concepts"]), inpool_e) if k]
        seeds_e = _labels(hf["seeds"])
        ap_e_ds = g["ap"]
        fire_e = hf["fire_counts"][()] if "fire_counts" in hf else None
        n_f = ap_e_ds.shape[1]
        # float32 storage: `gain_from_ap` promotes to float64 per seed, so
        # holding all 20 seeds at float64 would double peak memory for nothing.
        ap_e = np.empty((len(seeds_e), n_f, int(inpool_e.sum())),
                        dtype=np.float32)
        for s in range(len(seeds_e)):
            ap_e[s] = ap_e_ds[s][:, inpool_e]

    # ---- observed gains, pooled -----------------------------------------
    with h5py.File(args.ap_pooled, "r") as hf:
        g = hf["pooled"]
        inpool_p = g["in_argmax_pool"][()]
        prev_p = g["prevalence"][()][inpool_p]
        npos_p = g["n_positives"][()][inpool_p]
        nre_p = g["n_rows_effective"][()][inpool_p]
        conc_p = [c for c, k in zip(_labels(g["concepts"]), inpool_p) if k]
        seeds_p = _labels(hf["seeds"])
        ap_p_ds = g["ap"]
        fire_p = hf["fire_counts"][()]
        ap_p = np.empty((len(seeds_p), ap_p_ds.shape[1], int(inpool_p.sum())),
                        dtype=np.float32)
        for s in range(len(seeds_p)):
            ap_p[s] = ap_p_ds[s][:, inpool_p]

    # ---- C4's invariant, asserted on BOTH strata rather than assumed ----
    for nm, prev, npos, nre in (("experimental", prev_e, npos_e, nre_e),
                                ("pooled", prev_p, npos_p, nre_p)):
        dev = float(np.abs(prev - npos / nre).max())
        if dev > 1e-12:
            raise SystemExit(
                f"C4 VIOLATION on the {nm} stratum: stored prevalence is not "
                f"n_positives / n_rows_effective (max deviation {dev:.3e}). "
                "AV6's gain floor is defined against the EFFECTIVE row count.")

    # ---- SEED AXES ALIGNED ON LABELS, NEVER BY POSITION (gotcha 29) ------
    with h5py.File(args.null_pooled_reductions, "r") as hf:
        seeds_npool = _labels(hf["seeds"])
        conc_npool = _labels(hf["concepts"])
        null_g = {a: hf[f"null_g_{a}"][()] for a in ("a", "b")}

    with h5py.File(args.raw_null_pooled, "r") as hf:
        # THE POOLED RAW NULL STORES NO SEED LABELS. Its datasets are
        # `concepts`, `gain_a`, `gain_b` and nothing else, so there is no label
        # to join its seed axis on. Gotcha 29 names this case exactly:
        # recover the order FROM THE DATA and make the recovery able to fail.
        # `raw_seeds` below is therefore the reductions' labels used as a
        # HYPOTHESIS, and `recover_seed_position` tests that hypothesis slab by
        # slab against `null_g_*`. It is never assumed.
        if "seeds" in hf:
            raise SystemExit(
                "the pooled raw null now HAS a `seeds` dataset. The recovery "
                "below exists because it did not; read the labels instead of "
                "recovering them, and delete this branch.")
        raw_seeds = list(seeds_npool)
        raw_concepts = _labels(hf["concepts"])
        raw_identity = {k: (hf.attrs[k].decode()
                            if isinstance(hf.attrs[k], bytes) else hf.attrs[k])
                        for k in hf.attrs}

        axes = {"bh_experimental": bh_seeds, "ap_experimental": seeds_e,
                "ap_pooled": seeds_p, "null_pooled_reductions": seeds_npool}
        for name, lab in axes.items():
            if lab != bh_seeds:
                raise SystemExit(
                    "SEED AXIS MISMATCH between the experimental BH artifact "
                    f"and {name}.\n  bh : {bh_seeds}\n  {name}: {lab}\n"
                    "Joining these by position would mis-pair seeds and yield "
                    "a complete, internally consistent, wrong result "
                    "(gotcha 29).")

        # ---- CONCEPT AXES POSITIONAL, not set-wise (gotcha 27) -----------
        if conc_e != bh_concepts:
            raise SystemExit(
                "EXPERIMENTAL CONCEPT AXIS MISMATCH between the AP artifact's "
                "in-pool concepts and the BH artifact's.")
        for name, lab in (("null_pooled_reductions", conc_npool),
                          ("raw_null_pooled", raw_concepts)):
            if conc_p != lab:
                same = sorted(conc_p) == sorted(lab)
                first = next((i for i, (x, y) in enumerate(zip(conc_p, lab))
                              if x != y), None)
                raise SystemExit(
                    f"POOLED CONCEPT AXIS MISMATCH against {name} (same "
                    f"concepts, different order: {same}; first disagreement at "
                    f"{first}). Only a POSITIONAL check discriminates a "
                    "mapping defect (gotcha 27).")

        n_seeds = (len(raw_seeds) if args.seeds is None
                   else min(args.seeds, len(raw_seeds)))
        n_c_p = len(raw_concepts)
        n_c_e = len(bh_concepts)
        if hf["gain_a"].shape[3] != n_f:
            raise SystemExit(
                f"feature axis disagreement: raw pooled null has "
                f"{hf['gain_a'].shape[3]} features, AP has {n_f}")

        # ---- the floors, from parameters alone, BEFORE the pass ----------
        qf = Fraction(args.q).limit_denominator(10**6)
        floor_e = k_min(n_c_e, n_f, 20, qf)
        floor_p = k_min(n_c_p, n_f, 20, qf)
        print(f"resolution floors (from parameters, no data): "
              f"experimental k_min={floor_e}, pooled k_min={floor_p}")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        out = h5py.File(args.out, "w")

        summary: dict = {}
        for arm in ("a", "b"):
            cells = np.zeros((n_seeds, n_f), dtype=np.uint8)
            cl_e = np.zeros((n_seeds, n_f), dtype=bool)
            cl_p = np.zeros((n_seeds, n_f), dtype=bool)
            pj = out.create_dataset(
                f"p_pooled_{arm}", shape=(n_seeds, n_f, n_c_p),
                dtype="float32", chunks=(1, n_f, 1), compression="gzip",
                compression_opts=1)
            rj = out.create_dataset(
                f"reject_pooled_{arm}", shape=(n_seeds, n_f, n_c_p),
                dtype=bool, chunks=(1, n_f, 1), compression="gzip",
                compression_opts=1)
            nfin_p_all = np.zeros((n_seeds, n_c_p), dtype=np.int64)
            notev_p_all = np.zeros((n_seeds, n_c_p), dtype=bool)
            recovered = []
            rows = []

            for s in range(n_seeds):
                slab = hf[f"gain_{arm}"][s]          # (shift, concept, feat)

                # ---- RECOVER THIS SLAB'S SEED LABEL FROM THE DATA ---------
                # The raw array has no seed labels (gotcha 29). `null_g_*` is
                # its own max over concepts, so exactly one labelled seed must
                # reproduce it, compared in the raw array's stored precision
                # (`recover_raw_slab_seed`).
                pos = recover_raw_slab_seed(slab, null_g[arm])
                recovered.append(pos)
                if pos != s:
                    raise SystemExit(
                        f"SEED AXIS PERMUTED: raw slab at position {s} is "
                        f"labelled seed {seeds_npool[pos]} (position {pos}), "
                        "not the seed at the same position in the reductions. "
                        "A positional join here would mis-pair seeds and yield "
                        "a complete, internally consistent, wrong result "
                        "(gotcha 29). Refusing rather than reindexing "
                        "silently.")

                # The pooled stratum has NO reductions artifact recording
                # evaluability, so it is DERIVED from the raw array: every
                # concept is offered and `bh_pass_one_slab` marks the ones with
                # no surviving draw. The experimental side asserts against its
                # reductions; this side has nothing to assert against, and that
                # difference is stated rather than papered over.
                r = bh_pass_one_slab(
                    slab, gain_from_ap(ap_p[s], prev_p),
                    np.ones(n_c_p, dtype=bool), q=args.q)
                del slab
                pj[s] = r["p"].astype(np.float32)
                rj[s] = r["reject"]
                nfin_p_all[s] = r["n_finite"]
                notev_p_all[s] = r["not_evaluable"]

                gain_e = gain_from_ap(ap_e[s], prev_e)
                gain_p = gain_from_ap(ap_p[s], prev_p)
                e_ok = bool((~notev_exp[arm]).any())
                p_ok = bool((~r["not_evaluable"]).any())

                cl_e[s] = clears_bar(reject_exp[arm][s], gain_e,
                                     args.gain_floor)
                cl_p[s] = clears_bar(r["reject"], gain_p, args.gain_floor)
                cells[s] = classify_cells(cl_e[s], cl_p[s], e_ok, p_ok)
                counts = verify_outcome_space(cells[s], cl_e[s], cl_p[s],
                                              e_ok, p_ok)

                af_e = at_floor_counts(p_exp[arm][s], nfin_exp[arm][s],
                                       notev_exp[arm])
                af_p = at_floor_counts(r["p"], r["n_finite"],
                                       r["not_evaluable"])
                nrej_e = int(reject_exp[arm][s].sum())
                rows.append({
                    "seed": raw_seeds[s],
                    "n_rejected_experimental": nrej_e,
                    "n_rejected_pooled": int(r["n_rejected"]),
                    "at_floor_experimental": af_e,
                    "at_floor_pooled": af_p,
                    "floor_limited_experimental": bool(af_e < floor_e),
                    "floor_limited_pooled": bool(af_p < floor_p),
                    "n_concepts_not_evaluable_pooled":
                        int(r["not_evaluable"].sum()),
                    "cells": {CELL_NAMES[c]: counts[c] for c in sorted(counts)},
                    "n_ALIGNED": counts[CELL_ALIGNED_BOTH]
                    + counts[CELL_ALIGNED_EXP],
                    "n_EXCLUDED": counts[CELL_EXCLUDED],
                    "n_ORPHAN": counts[CELL_ORPHAN],
                    "n_NOT_EVALUABLE": counts[CELL_NOT_EVALUABLE],
                })
                flag = " [FLOOR-LIMITED exp]" if af_e < floor_e else ""
                print(f"  arm {arm.upper()} {raw_seeds[s]}: "
                      f"ALIGNED={rows[-1]['n_ALIGNED']:>5} "
                      f"EXCLUDED={counts[CELL_EXCLUDED]:>5} "
                      f"ORPHAN={counts[CELL_ORPHAN]:>6} "
                      f"NE={counts[CELL_NOT_EVALUABLE]:>3} | "
                      f"rej_exp={nrej_e:>4} rej_pool={r['n_rejected']:>6} "
                      f"floor_exp={af_e:>4} floor_pool={af_p:>6}{flag}")

            out.create_dataset(f"cell_{arm}", data=cells)
            out.create_dataset(f"clears_experimental_{arm}", data=cl_e)
            out.create_dataset(f"clears_pooled_{arm}", data=cl_p)
            out.create_dataset(f"n_finite_pooled_{arm}", data=nfin_p_all)
            out.create_dataset(f"not_evaluable_pooled_{arm}", data=notev_p_all)
            out.create_dataset(f"recovered_seed_position_{arm}",
                               data=np.array(recovered, dtype=np.int64))
            summary[arm.upper()] = rows

        out.create_dataset("seeds", data=np.array(
            raw_seeds[:n_seeds], dtype=h5py.string_dtype()))
        out.create_dataset("concepts_experimental", data=np.array(
            bh_concepts, dtype=h5py.string_dtype()))
        out.create_dataset("concepts_pooled", data=np.array(
            raw_concepts, dtype=h5py.string_dtype()))
        out.create_dataset("cell_codes", data=np.array(
            [f"{c}={CELL_NAMES[c]}" for c in sorted(CELL_NAMES)],
            dtype=h5py.string_dtype()))

        # ---- IDENTITY: nothing back-filled (criterion 10) ----------------
        for k in ("library_sha", "n_concepts", "argmax_pool_n",
                  "scored_row_set", "truncation_policy", "retention_floor_mode",
                  "retention_min_residues", "retention_min_instances",
                  "min_protein_count", "n_shifts", "null_seed", "null_type",
                  "n_rows", "gather_device_class", "ap_arith_device_class"):
            if k in raw_identity:
                out.attrs[f"pooled_{k}"] = raw_identity[k]
        for k in ("library_sha", "argmax_pool_used_n", "negatives", "stratum",
                  "ap_impl", "gather_device_class", "ap_arith_device_class"):
            if k in bh_identity:
                out.attrs[f"experimental_{k}"] = bh_identity[k]
        out.attrs["q_fdr"] = float(args.q)
        out.attrs["gain_floor"] = float(args.gain_floor)
        out.attrs["gain_floor_applied"] = True
        out.attrs["governing_amendment"] = "BQ (AV4, AV5, AV6; BM-E1; BP4)"
        out.attrs["classified_unit"] = "(seed, feature) pair; BQ4a"
        out.attrs["population_per_arm"] = int(n_seeds * n_f)
        out.attrs["k_min_experimental"] = int(floor_e)
        out.attrs["k_min_pooled"] = int(floor_p)
        out.attrs["threshold_grid_point"] = 1
        out.attrs["git_sha"] = git_sha()
        out.attrs["git_dirty"] = bool(git_dirty())
        for nm, v in (("bh_experimental", args.bh_experimental),
                      ("ap_experimental", args.ap_experimental),
                      ("ap_pooled", args.ap_pooled),
                      ("raw_null_pooled", args.raw_null_pooled),
                      ("null_pooled_reductions", args.null_pooled_reductions)):
            out.attrs[f"input_{nm}"] = str(v)
        if fire_e is not None:
            div = int(np.abs(fire_e.astype(np.int64)
                             - fire_p.astype(np.int64)).max())
            out.attrs["axis_device_divergence_max_fire_count"] = div
            out.attrs["axis_device_divergence_note"] = (
                "The experimental axis descends from a CPU gather and the "
                "pooled axis from a GPU-era artifact; their fire_counts differ "
                "by at most this many rows. Gotcha 36's measured signature, "
                "recorded not repaired. Nothing back-filled.")
        out.attrs["note"] = (
            "AV6's bar with BOTH conjuncts applied: FDR at q AND raw gain "
            "G >= gain_floor, ANDed per (feature, concept) BEFORE the "
            "existential over concepts.")
        out.close()

    report = {
        "artifact": "bq_classification_2x2",
        "governing_amendment": "BQ",
        "git_sha": git_sha(), "git_dirty": bool(git_dirty()),
        "q": float(args.q), "gain_floor": float(args.gain_floor),
        "threshold_grid_point": 1,
        "classified_unit": "(seed, feature) pair",
        "n_seeds": n_seeds, "n_features": n_f,
        "population_per_arm": int(n_seeds * n_f),
        "n_concepts_experimental": n_c_e, "n_concepts_pooled": n_c_p,
        "k_min_experimental": int(floor_e), "k_min_pooled": int(floor_p),
        "cell_codes": {str(c): CELL_NAMES[c] for c in sorted(CELL_NAMES)},
        "per_arm": summary,
        "inputs": {
            "bh_experimental": str(args.bh_experimental),
            "ap_experimental": str(args.ap_experimental),
            "ap_pooled": str(args.ap_pooled),
            "raw_null_pooled": str(args.raw_null_pooled),
            "null_pooled_reductions": str(args.null_pooled_reductions),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}\nwrote {args.report}")


if __name__ == "__main__":
    main()
