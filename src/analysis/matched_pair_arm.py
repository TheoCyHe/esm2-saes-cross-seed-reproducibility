"""COMPONENT 5 (PREREGISTRATION amendment BU): the FIRING-RATE-MATCHED PAIRED
ARM, at the unrestricted threshold (`T = 1`) only.

This is the last build component. It is a **pre-registered ROBUSTNESS ARM**
(H1), not the headline: concept-level reproducibility is the descriptive
headline and the design-3 regression is the inferential primary.

---------------------------------------------------------------------------
THE CONSTRUCTION, READ OFF THE RECORD RATHER THAN INVENTED (§5, AN3, §5a, §5b).
---------------------------------------------------------------------------
    test            firing-rate-matched paired Wilcoxon signed-rank on
                    reproducibility `r_f`, ALIGNED against ORPHAN   §5
    match_on        log10 firing rate                               §5, AN3
    caliper         0.1 log10 units, a recorded DEGREE OF FREEDOM   §5, AN3
    pairing unit    feature <-> feature, treated (ALIGNED) to
                    control (ORPHAN), WITHIN one seed               gotcha 5
    unmatchable     the count of treated features finding NO
                    partner is MANDATORY reporting                  gotcha 5
    binning         matched-pair counts and the statistic PER
                    FIRING-RATE BIN, never aggregate only, reusing
                    `corrector_scan.BIN_EDGES`                      open-Q5 gate
    correctors      flagged features excluded; the arm reported
                    WITH and WITHOUT them, even though 0/20 seeds
                    carry a candidate                               §5
    ties            ALL tied candidates, weighted equally           §5a rule 1
    replacement     WITH replacement; the reuse rate is a
                    MANDATORY report, not an optional diagnostic    §5a rule 2
    empty seed      a seed contributing zero treated features is an
                    EXPLICIT IGNORANCE CATEGORY with its own count  §5b rule 3
    inputs          NO artifact of its own: firing rates are
                    `fire_counts` in the AP artifact, `r_f` is
                    `match_count / n_compare`, and the treated and
                    control sets are component 3's OUTPUT           §BJ-C1, BS-E

---------------------------------------------------------------------------
THE ESTIMAND, AND WHY IT IS A SET AND NOT A PARTNER (§5a).
---------------------------------------------------------------------------
Rule 1 matches a treated feature to a SET of equidistant controls, so the
contrast is

    d_i  =  r_f(treated i)  -  mean over i's matched SET of r_f(control)

and each treated feature contributes ONE observation. It is never expanded into
one observation per (treated, control) pair: that would inflate the count of
independent units by the tie multiplicity and is exactly what §5a's estimand
note forbids.

**THE SET STRUCTURE IS ACCOUNTED FOR RATHER THAN ASSUMED AWAY, in the one place
where exchangeability is actually established.** Two dependencies exist and both
are MEASURED here rather than described:

  * WITHIN a seed, two treated features sharing a control have correlated `d_i`
    (the cost rule 2 accepts because it is measurable). `n_treated_sharing_a
    _control` and the reuse distribution are reported per seed.
  * The averaging in `d_i` shrinks control-side noise wherever a set is larger
    than one, so a within-seed p-value computed as if each `d_i` were an
    independent single-partner contrast is not exact.

So the within-seed signed-rank p-value is reported as the committed statistic,
and the arm's dispersion statement is made ACROSS THE 20 ANCHOR SEEDS, which
§BP4 pins as the exchangeable unit. The across-seed spread is reported as
min / median / max of the per-seed values: **a descriptive spread, explicitly
NOT an interval** -- no fit, refit or bootstrap is performed here.

---------------------------------------------------------------------------
THE RESOLUTION FLOOR, COMPUTED FROM PARAMETERS BEFORE ANY DATA (gotcha 42).
---------------------------------------------------------------------------
An exact two-sided signed-rank test on `n` non-zero differences cannot return a
p-value below `2 / 2**n = 2**(1-n)`, whatever the data say, because that is the
probability of the single most extreme sign assignment doubled. So

    n = 1 -> 1.0     n = 3 -> 0.25     n = 5 -> 0.0625     n = 6 -> 0.03125

and a per-bin cell with five or fewer non-zero differences cannot reach 0.05
even in principle. **THE RELATION IS RECORDED, NOT A VERDICT** -- this arm
declares no alpha (§1's inference framing) and no fraction of cells is named as
a bar a design must reach. It is reported per cell so that a small p-value and
an unreachable one are never confused, and it is computable with no data at all.

---------------------------------------------------------------------------
WHAT THIS MODULE DOES NOT DO.
---------------------------------------------------------------------------
It runs at the unrestricted threshold only -- not at any thinness-grid point
(BI5). It claims no mechanism for why the arms differ or why the aligned count
is small. It does not revisit any matched-pair specification in the light of the
diagnostics BT-B3 measured.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

# SCIPY BEFORE ANYTHING THAT PULLS IN TORCH (gotcha 8). `src.sae.train` imports
# torch, and torch binds the SYSTEM libstdc++ first, after which scipy fails
# with a CXXABI_1.3.15 error that LD_LIBRARY_PATH cannot fix.
from scipy.stats import wilcoxon  # noqa: E402  (ordering is deliberate)

from src.analysis.classification_2x2 import (  # noqa: E402
    CELL_ALIGNED_BOTH,
    CELL_ALIGNED_EXP,
    CELL_ORPHAN,
)
from src.analysis.corrector_scan import BIN_EDGES  # noqa: E402
from src.analysis.match_pool_diagnostics import log10_rate  # noqa: E402
from src.sae.train import git_dirty, git_sha  # noqa: E402

__all__ = [
    "TREATED_BRANCHES",
    "TEST_BRANCHES",
    "SEED_BRANCHES",
    "match_sets",
    "set_contrast",
    "signed_rank",
    "min_attainable_two_sided_p",
    "bin_index",
    "reuse_profile",
    "verify_treated_outcome_space",
    "OutcomeSpaceViolation",
]

# ---------------------------------------------------------------------------
# THE OUTCOME SPACE, WRITTEN DOWN BEFORE THE BRANCHES ARE IMPLEMENTED
# (criterion 5, gotcha 39: in the primitive the data actually has).
#
# The population is the set of TREATED (seed, feature) cells -- the classified
# unit is a (seed, feature) pair (BQ4a) and the treated set is ALIGNED under one
# arm at T = 1. Every treated cell lands in exactly ONE of these.
# ---------------------------------------------------------------------------
TREATED_BRANCHES = (
    # A treated feature whose fire_count is 0 has no log firing rate to match
    # on. The ensemble has NO dead features (0 in all 20 seeds), so this branch
    # is expected to be EMPTY -- which is why it is written rather than assumed
    # (gotcha 24: a branch's domain being empty is a measurement, not a reason
    # to omit it).
    "NO_FIRING_RATE",
    # gotcha 5's mandatory count: nothing inside the caliper.
    "UNMATCHABLE",
    # Rule 1 never fires: exactly one nearest control.
    "MATCHED_UNIQUE",
    # Rule 1 fires: two or more controls at exactly the minimum distance.
    "MATCHED_TIED",
)

# Per (seed) and per (seed, firing-rate bin) the test itself has three
# outcomes, and the last two are statements of ignorance carried with their own
# counts rather than folded into a rate (criterion 6).
TEST_BRANCHES = (
    "COMPUTED",
    # No treated feature reached this cell at all.
    "NO_PAIRS",
    # Pairs exist but every difference is exactly zero, so the signed-rank
    # statistic ranks nothing. Reported UNINFORMATIVE, never as a non-result
    # and never as p = 1 (gotcha 25b).
    "ALL_DIFFERENCES_ZERO",
)

SEED_BRANCHES = (
    "CONTRIBUTES",
    # §5b rule 3. NOT a zero: there is no contrast to be zero.
    "NO_TREATED_FEATURES",
)


class OutcomeSpaceViolation(Exception):
    """Raised when a treated cell is unassigned, double-assigned, or the branch
    counts do not sum to the treated population.

    FATAL by criterion 5. An enumeration that is asserted rather than checked is
    not an enumeration.
    """


def match_sets(treated: np.ndarray, control: np.ndarray,
               caliper: float) -> list[np.ndarray]:
    """Rule 1 + rule 2: for each treated value, ALL controls at the minimum
    distance within the caliper; controls are reusable across treated features.

    `treated` and `control` are 1-D log10 firing rates. Returns one array of
    CONTROL INDICES per treated value, empty where nothing is admissible.

    WRITTEN FROM THE SPECIFICATION, NOT FROM `nearest_within_caliper`, and the
    driver asserts the two agree on the multiplicities (gotcha 32b: a test whose
    expectation is computed from the thing under test cannot detect a change to
    it, and the same holds for a second implementation).

    THE COMPARISONS ARE EXACT AND THE TIE TEST IS `== d_min`. Firing rate is
    `fire_count / n_rows` with a constant denominator, so two controls with
    equal fire_counts give bit-identical distances; a tolerance would invent
    ties the arithmetic does not have.

    THE CALIPER PREDICATE IS `<=`, INCLUSIVE, exactly as §5 pins it -- and
    corrections-log entry 142 records that at the pinned width of 0.1 the
    boundary is NOT REACHABLE in float64 (`abs(-2.0 - -2.1)` is
    0.10000000000000009, strictly greater than 0.1). The predicate is written
    inclusive because that is what is committed; whether any distance actually
    lands on it is MEASURED by the driver rather than assumed either way.
    """
    t = np.asarray(treated, dtype=np.float64)
    c = np.asarray(control, dtype=np.float64)
    if t.ndim != 1 or c.ndim != 1:
        raise ValueError("treated and control must both be 1-D")
    if not np.isfinite(caliper) or caliper <= 0:
        raise ValueError(f"caliper is {caliper}; it must be positive and finite")
    c_ok = np.isfinite(c)
    cv = c[c_ok]
    c_idx = np.flatnonzero(c_ok)
    out: list[np.ndarray] = []
    for i in range(t.size):
        if not np.isfinite(t[i]):
            out.append(np.zeros(0, dtype=np.int64))
            continue
        d = np.abs(cv - t[i])
        inside = d <= caliper
        if not inside.any():
            out.append(np.zeros(0, dtype=np.int64))
            continue
        dm = d[inside].min()
        out.append(c_idx[inside & (d == dm)].astype(np.int64))
    return out


def set_contrast(rf_treated: np.ndarray, rf_control: np.ndarray,
                 sets: list[np.ndarray]) -> np.ndarray:
    """`d_i = r_f(treated i) - mean_{j in set(i)} r_f(control j)`.

    NaN where the set is empty -- an unmatchable treated feature has NO
    contrast, and returning 0.0 there would enter a real observation of "no
    difference" for a feature that was never compared to anything.

    THE MEAN IS UNWEIGHTED because rule 1 weights tied candidates EQUALLY.
    """
    rt = np.asarray(rf_treated, dtype=np.float64)
    rc = np.asarray(rf_control, dtype=np.float64)
    if rt.ndim != 1 or rc.ndim != 1:
        raise ValueError("rf_treated and rf_control must both be 1-D")
    if len(sets) != rt.size:
        raise ValueError(f"{len(sets)} matched sets for {rt.size} treated "
                         "features")
    out = np.full(rt.size, np.nan, dtype=np.float64)
    for i, s in enumerate(sets):
        if s.size:
            out[i] = rt[i] - float(rc[s].mean())
    return out


def min_attainable_two_sided_p(n_nonzero: int) -> float:
    """gotcha 42's floor for this procedure, from ITS PARAMETERS AND NO DATA.

    The exact two-sided signed-rank test on `n` non-zero differences puts
    probability `2**-n` on each sign assignment, so the most extreme one has
    two-sided p `2**(1-n)`. With `n = 0` no test exists and the floor is 1.0.

    A RELATION, NOT A VERDICT. No alpha is declared by this arm and no fraction
    of cells is named as a bar. It exists so that "did not reach a small
    p-value" and "could not have reached one" are never read as the same
    statement. The floor binds the EXACT method; where ties in |d| push scipy
    onto the normal approximation the attainable p is not bounded this way, and
    the method actually used is recorded beside it.
    """
    n = int(n_nonzero)
    if n < 0:
        raise ValueError(f"n_nonzero is {n}")
    if n == 0:
        return 1.0
    return float(2.0 ** (1 - n))


def signed_rank(d: np.ndarray) -> dict:
    """The committed test on one cell's contrasts. A TOTAL function.

    Every branch in `TEST_BRANCHES` is reachable and named in the return value;
    a summary formatter with no total branch has reported a completed
    measurement as a failed job in this repo before (gotcha 32c).

    NaN contrasts (unmatchable treated features) are dropped and counted here
    rather than by the caller, so that a cell's `n_pairs` can never silently
    include a feature that was never compared.
    """
    x = np.asarray(d, dtype=np.float64)
    finite = np.isfinite(x)
    v = x[finite]
    n_zero = int(np.count_nonzero(v == 0.0))
    nz = v[v != 0.0]
    base = {
        "n_pairs": int(v.size),
        "n_dropped_no_contrast": int((~finite).sum()),
        "n_zero_differences": n_zero,
        "n_nonzero_differences": int(nz.size),
        "min_attainable_two_sided_p_exact": min_attainable_two_sided_p(nz.size),
        "median_difference": float(np.median(v)) if v.size else None,
        "mean_difference": float(v.mean()) if v.size else None,
        "n_positive_differences": int(np.count_nonzero(v > 0)),
        "n_negative_differences": int(np.count_nonzero(v < 0)),
    }
    if v.size == 0:
        base.update({"branch": "NO_PAIRS", "statistic": None, "p_value": None,
                     "method": None})
        return base
    if nz.size == 0:
        # A predicate applied to a constant ranks nothing (gotcha 25b): report
        # UNINFORMATIVE rather than a p-value of 1 that reads as a real
        # non-rejection.
        base.update({"branch": "ALL_DIFFERENCES_ZERO", "statistic": None,
                     "p_value": None, "method": None})
        return base
    method = "exact" if nz.size <= 25 else "approx"
    res = wilcoxon(v, zero_method="wilcox", alternative="two-sided",
                   method=method)
    base.update({"branch": "COMPUTED", "statistic": float(res.statistic),
                 "p_value": float(res.pvalue), "method": method})
    return base


def bin_index(rates: np.ndarray, edges=BIN_EDGES) -> np.ndarray:
    """The firing-rate bin of each feature, on `corrector_scan.BIN_EDGES`.

    Returns -1 for a rate outside the edges, which is carried as its own count
    rather than clamped into the end bins. Bin `k` is `[edges[k], edges[k+1])`;
    the top edge is 1 + 1e-9 so a rate of exactly 1.0 lands inside.

    THE ASSIGNMENT IS A SINGLE `searchsorted`, NOT A LOOP OF INTERVAL TESTS, and
    that is deliberate. A loop writing `(r >= lo) & (r < hi)` bin by bin gives
    the RIGHT answer at an edge for the wrong reason: an inclusive upper bound
    would put an edge value in two bins and the later write would silently win,
    so the half-open convention would be unobservable and a mutation of it
    undetectable. `side="right"` makes the convention the thing being computed.

    THE EDGES ARE REUSED, NOT RE-CHOSEN. The top bin is sparse (~5 features per
    seed above 0.1) and an artifact there carries outsized weight on the matched
    pairs, which is the whole reason per-bin reporting is mandatory.
    """
    r = np.asarray(rates, dtype=np.float64)
    e = np.asarray(edges, dtype=np.float64)
    out = (np.searchsorted(e, r, side="right") - 1).astype(np.int64)
    outside = ~np.isfinite(r) | (r < e[0]) | (r >= e[-1])
    out[outside] = -1
    return out


def reuse_profile(sets: list[np.ndarray], n_control: int) -> dict:
    """Rule 2's MANDATORY report, plus the dependence it induces.

    `use_count` over controls; the number of distinct controls used; and
    `n_treated_sharing_a_control` -- how many treated features share at least
    one control with another treated feature. That last one is the quantity the
    estimand note in §5a is about: it is the measured extent to which the
    per-seed contrasts are not independent, reported rather than assumed small.
    """
    if int(n_control) < 0:
        raise ValueError(f"n_control is {n_control}")
    use = np.zeros(int(n_control), dtype=np.int64)
    for s in sets:
        use[s] += 1
    used = use[use > 0]
    shared = np.flatnonzero(use > 1)
    n_sharing = 0
    if shared.size:
        shared_set = set(shared.tolist())
        for s in sets:
            if any(int(j) in shared_set for j in s):
                n_sharing += 1
    n_matched = int(sum(1 for s in sets if s.size))
    return {
        "n_distinct_controls_used": int(used.size),
        "control_reuse_mean": float(used.mean()) if used.size else None,
        "control_reuse_max": int(used.max()) if used.size else None,
        "n_controls_reused": int(shared.size),
        "frac_used_controls_reused": (float((used > 1).mean())
                                      if used.size else None),
        "n_treated_sharing_a_control": int(n_sharing),
        "n_treated_matched": n_matched,
        "frac_treated_sharing_a_control": (n_sharing / n_matched
                                           if n_matched else None),
        "denominator_note": "rates over MATCHED treated features and over "
                            "controls actually used; both stated",
    }


def verify_treated_outcome_space(branches: list[str], n_treated: int) -> dict:
    """Criterion 5, in this component's primitive: every treated (seed,
    feature) cell lands in EXACTLY ONE branch, and the counts sum.

    Checked in the direction that can fail: an unknown branch label, a length
    mismatch, or counts that do not sum all raise.
    """
    counts = {b: 0 for b in TREATED_BRANCHES}
    for b in branches:
        if b not in counts:
            raise OutcomeSpaceViolation(
                f"branch {b!r} is not one of {TREATED_BRANCHES}; a treated cell "
                "landed outside the enumeration")
        counts[b] += 1
    if len(branches) != int(n_treated):
        raise OutcomeSpaceViolation(
            f"{len(branches)} branch assignments for {n_treated} treated "
            "cells -- a cell is unassigned or double-assigned")
    total = sum(counts.values())
    if total != int(n_treated):
        raise OutcomeSpaceViolation(
            f"branch counts sum to {total}, treated population is {n_treated}")
    return counts


# --------------------------------------------------------------------------
# CLI -- the driver.
# --------------------------------------------------------------------------

def _labels(ds) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in ds[()]]


def _seed_number(label: str) -> int:
    """`'seed_11'` -> `11`. The AP and 2x2 axes are LEXICOGRAPHIC and the `r_f`
    axis is NUMERIC, so the join is on the seed NUMBER carried by the label and
    never on row position (gotcha 29: only rows 0 and 1 coincide)."""
    s = str(label)
    if not s.startswith("seed_"):
        raise ValueError(f"seed label {s!r} is not of the form seed_<n>")
    return int(s[len("seed_"):])


def _cell_summary(rows: list[dict], key: str) -> dict:
    """Across-seed spread of a per-seed quantity: min / median / max over the
    seeds that HAVE the quantity, with the denominator stated.

    DESCRIPTIVE, AND EXPLICITLY NOT AN INTERVAL. No fit, no bootstrap. The
    censored denominator is stated on every one of these (§5b, §BP4 item 3).
    """
    vals = [r[key] for r in rows if r.get(key) is not None]
    return {
        "n_seeds_with_a_value": len(vals),
        "n_seeds_total": len(rows),
        "censored": len(vals) != len(rows),
        "min": float(min(vals)) if vals else None,
        "median": float(np.median(vals)) if vals else None,
        "max": float(max(vals)) if vals else None,
        "note": "descriptive across-seed spread over the 20 anchors, which "
                "BP4 pins as the exchangeable unit; NOT a confidence interval "
                "and not a fit",
    }


def main() -> None:
    ap_ = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--ap-experimental", type=Path, required=True,
                     help="feature_ap_experimental_av_cpu.h5 -- read for "
                          "fire_counts and n_rows only")
    ap_.add_argument("--classification-2x2", type=Path, required=True,
                     help="classification_2x2_bq.h5 -- component 3's output "
                          "supplies the treated (ALIGNED) and control (ORPHAN) "
                          "sets at T = 1")
    ap_.add_argument("--rf-counts", type=Path, required=True,
                     help="rf_counts_theta070_n39.h5 -- match_count, the "
                          "outcome; r_f = match_count / n_compare")
    ap_.add_argument("--corrector-scan", type=Path, required=True,
                     help="corrector_scan.json -- flagged features are "
                          "excluded and the arm is reported WITH and WITHOUT")
    ap_.add_argument("--caliper", type=float, default=0.1)
    ap_.add_argument("--out", type=Path, required=True)
    ap_.add_argument("--report", type=Path, required=True)
    ap_.add_argument("--seeds", type=int, default=None,
                     help="cost pilot only: process the first N seeds. Sets "
                          "is_cost_pilot=True; its counts are NOT the arm.")
    args = ap_.parse_args()

    # ---- inputs ----------------------------------------------------------
    with h5py.File(args.ap_experimental, "r") as hf:
        fire = hf["fire_counts"][()]
        ap_seeds = _labels(hf["seeds"])
        n_rows = int(hf.attrs["n_rows"])
        ap_identity = {k: (hf.attrs[k].decode()
                           if isinstance(hf.attrs[k], bytes) else hf.attrs[k])
                       for k in hf.attrs}

    with h5py.File(args.classification_2x2, "r") as hf:
        c3_seeds = _labels(hf["seeds"])
        cells = {a: hf[f"cell_{a}"][()] for a in ("a", "b")}
        c3_identity = {k: (hf.attrs[k].decode()
                           if isinstance(hf.attrs[k], bytes) else hf.attrs[k])
                       for k in hf.attrs}

    with h5py.File(args.rf_counts, "r") as hf:
        match_count = hf["match_count"][()]
        rf_attrs = {k: (hf.attrs[k].decode()
                        if isinstance(hf.attrs[k], bytes) else hf.attrs[k])
                    for k in hf.attrs}
    # An h5 attr round-trips a list through json.dumps, so it is PARSED BACK
    # rather than iterated -- `list()` of a JSON string is a list of characters
    # (gotcha 20).
    rf_seed_numbers = json.loads(rf_attrs["seed_labels"])
    n_compare = int(rf_attrs["n_compare_seeds"])

    # ---- SEED AXES JOINED ON LABELS, NEVER BY POSITION (gotcha 29) --------
    if ap_seeds != c3_seeds:
        raise SystemExit(
            f"SEED AXIS MISMATCH: AP {ap_seeds} vs 2x2 {c3_seeds}. Joining "
            "these by position would mis-pair seeds and yield a complete, "
            "internally consistent, wrong result.")
    try:
        rf_row = [rf_seed_numbers.index(_seed_number(s)) for s in ap_seeds]
    except ValueError as exc:
        raise SystemExit(
            f"a seed on the AP/2x2 axis is absent from the r_f axis "
            f"{rf_seed_numbers}: {exc}") from None
    if sorted(rf_row) != list(range(len(rf_seed_numbers))):
        raise SystemExit("the r_f join is not a bijection over the seed axis")
    print("seed join, by LABEL (AP/2x2 lexicographic -> r_f numeric row):")
    print("  " + ", ".join(f"{s}->{r}" for s, r in zip(ap_seeds, rf_row)))
    if [i for i, r in enumerate(rf_row) if i != r]:
        n_moved = sum(1 for i, r in enumerate(rf_row) if i != r)
        print(f"  {n_moved} of {len(rf_row)} rows MOVE under this join; a "
              "positional join would have been wrong for every one of them")

    # ---- the corrector exclusion (§5), keyed on the seed LABEL -------------
    scan = json.loads(Path(args.corrector_scan).read_text())
    flagged: dict[str, list[int]] = {}
    for entry in scan["seeds"]:
        lab = Path(entry["ckpt"]).name
        flagged[lab] = [int(f) for f in entry["flagged_features"]]
    missing = [s for s in ap_seeds if s not in flagged]
    if missing:
        raise SystemExit(f"the corrector scan has no entry for {missing}; the "
                         "exclusion is mandatory reporting and cannot be "
                         "silently skipped for a seed")
    n_flagged_total = sum(len(flagged[s]) for s in ap_seeds)

    n_seeds_all, n_features = fire.shape
    n_seeds = n_seeds_all if args.seeds is None else min(args.seeds,
                                                         n_seeds_all)

    # ---- the floor, from parameters alone and no data (gotcha 42) ---------
    print("\nresolution floor of the exact two-sided signed-rank test, from "
          "its parameters with no data read:")
    for n in (1, 3, 5, 6, 8, 10):
        print(f"  n = {n:<3} smallest attainable two-sided p = "
              f"{min_attainable_two_sided_p(n):.6g}")

    arms = ("a", "b")
    variants = ("with_correctors", "without_correctors")
    results: dict = {}
    boundary_hits = 0
    boundary_examined = 0

    for arm in arms:
        for variant in variants:
            per_seed = []
            branch_totals = {b: 0 for b in TREATED_BRANCHES}
            seed_branch_totals = {b: 0 for b in SEED_BRANCHES}
            bin_rows: dict[int, list[float]] = {}
            all_d: list[np.ndarray] = []
            for s in range(n_seeds):
                lab = ap_seeds[s]
                cell = cells[arm][s]
                treated_mask = ((cell == CELL_ALIGNED_BOTH)
                                | (cell == CELL_ALIGNED_EXP))
                control_mask = cell == CELL_ORPHAN
                if variant == "without_correctors" and flagged[lab]:
                    drop = np.zeros(n_features, dtype=bool)
                    drop[np.array(flagged[lab], dtype=np.int64)] = True
                    treated_mask = treated_mask & ~drop
                    control_mask = control_mask & ~drop
                if (treated_mask & control_mask).any():
                    raise SystemExit(
                        f"arm {arm} {lab}: a feature is both treated and "
                        "control; the cells are disjoint by construction (AV5) "
                        "and a treated feature must not be its own control")

                t_idx = np.flatnonzero(treated_mask)
                c_idx = np.flatnonzero(control_mask)
                rate = fire[s] / float(n_rows)
                lr = log10_rate(fire[s], n_rows)
                rf = match_count[rf_row[s]].astype(np.float64) / n_compare

                # ---- §5b: the seed-level branch, BEFORE any pairing --------
                if t_idx.size == 0:
                    seed_branch_totals["NO_TREATED_FEATURES"] += 1
                    per_seed.append({
                        "seed": lab,
                        # BP4 item 1: the flag rides on the ROW a reader lands
                        # on, never on a caption.
                        "seed_branch": "NO_TREATED_FEATURES",
                        "n_treated": 0,
                        "n_control": int(c_idx.size),
                        "branch": "NO_PAIRS",
                        "n_pairs": 0,
                        "statistic": None, "p_value": None,
                        "median_difference": None,
                        "note": ("no ALIGNED feature in this seed at T = 1. An "
                                 "EXPLICIT IGNORANCE CATEGORY (§5b): the "
                                 "per-seed contrast is UNDEFINED, not zero, "
                                 "and the seed is neither dropped from the "
                                 "denominator nor read as a null result. It is "
                                 "NOT a NOT_EVALUABLE concept -- different "
                                 "categories at different levels (§BP4 item "
                                 "4)."),
                    })
                    continue
                seed_branch_totals["CONTRIBUTES"] += 1

                sets = match_sets(lr[t_idx], lr[c_idx], args.caliper)

                # The caliper boundary, MEASURED rather than assumed either way
                # (corrections-log entry 142 says it is unreachable in float64
                # at the pinned width; this counts how often a distance lands
                # exactly on it).
                for i, ti in enumerate(t_idx):
                    d = np.abs(lr[c_idx] - lr[ti])
                    d = d[np.isfinite(d)]
                    boundary_examined += int(d.size)
                    boundary_hits += int(np.count_nonzero(d == args.caliper))

                branches = []
                for i, ti in enumerate(t_idx):
                    if not np.isfinite(lr[ti]):
                        branches.append("NO_FIRING_RATE")
                    elif sets[i].size == 0:
                        branches.append("UNMATCHABLE")
                    elif sets[i].size == 1:
                        branches.append("MATCHED_UNIQUE")
                    else:
                        branches.append("MATCHED_TIED")
                counts = verify_treated_outcome_space(branches, t_idx.size)
                for b, v in counts.items():
                    branch_totals[b] += v

                d = set_contrast(rf[t_idx], rf[c_idx], sets)
                res = signed_rank(d)
                reuse = reuse_profile(sets, c_idx.size)

                # ---- per firing-rate bin, MANDATORY (open-Q5 gate) ---------
                tb = bin_index(rate[t_idx])
                per_bin = []
                for k in range(len(BIN_EDGES) - 1):
                    sel = tb == k
                    r_k = signed_rank(d[sel]) if sel.any() else signed_rank(
                        np.zeros(0))
                    r_k.update({
                        "bin": k,
                        "lo": BIN_EDGES[k], "hi": BIN_EDGES[k + 1],
                        "n_treated_in_bin": int(sel.sum()),
                        "n_unmatchable_in_bin": int(sum(
                            1 for i in np.flatnonzero(sel) if sets[i].size == 0)),
                    })
                    per_bin.append(r_k)
                    bin_rows.setdefault(k, []).extend(
                        d[sel][np.isfinite(d[sel])].tolist())
                unbinned = int(np.count_nonzero(tb < 0))

                row = {
                    "seed": lab,
                    "seed_branch": "CONTRIBUTES",
                    "n_treated": int(t_idx.size),
                    "n_control": int(c_idx.size),
                    "n_treated_no_firing_rate": counts["NO_FIRING_RATE"],
                    # gotcha 5's mandatory count.
                    "n_unmatchable": counts["UNMATCHABLE"],
                    "n_matched_unique": counts["MATCHED_UNIQUE"],
                    "n_matched_tied": counts["MATCHED_TIED"],
                    "n_treated_outside_all_bins": unbinned,
                    "reuse": reuse,
                    "per_bin": per_bin,
                }
                row.update(res)
                per_seed.append(row)
                all_d.append(d[np.isfinite(d)])

            pooled = (np.concatenate(all_d) if all_d
                      else np.zeros(0, dtype=np.float64))
            contributing = [r for r in per_seed
                            if r["seed_branch"] == "CONTRIBUTES"]
            results[(arm, variant)] = {
                "per_seed": per_seed,
                "seed_branch_counts": seed_branch_totals,
                "seeds_with_no_treated_feature": [
                    r["seed"] for r in per_seed
                    if r["seed_branch"] == "NO_TREATED_FEATURES"],
                "treated_branch_counts": branch_totals,
                "n_treated_total": int(sum(branch_totals.values())),
                "across_seeds": {
                    "denominator_seeds_contributing": len(contributing),
                    "denominator_seeds_total": n_seeds,
                    "censored": len(contributing) != n_seeds,
                    "censoring_note": (
                        "seeds contributing no treated feature are carried as "
                        "their own category and are NOT dropped from this "
                        "denominator's statement (§5b, §BP4 item 3)"),
                    "median_difference": _cell_summary(contributing,
                                                       "median_difference"),
                    "p_value": _cell_summary(contributing, "p_value"),
                    "n_pairs_total": int(sum(r["n_pairs"]
                                             for r in contributing)),
                    "n_unmatchable_total": int(sum(r["n_unmatchable"]
                                                   for r in contributing)),
                },
                # Pooled across seeds IN ADDITION, never INSTEAD of the
                # per-seed values: the 20 anchors are the exchangeable unit.
                "pooled_over_seeds": dict(
                    signed_rank(pooled),
                    note="reported IN ADDITION to the per-seed values, never "
                         "instead; the anchors are the exchangeable unit and "
                         "pooling ignores the seed structure"),
                "per_bin_pooled": [
                    dict(signed_rank(np.asarray(bin_rows.get(k, []),
                                                dtype=np.float64)),
                         bin=k, lo=BIN_EDGES[k], hi=BIN_EDGES[k + 1])
                    for k in range(len(BIN_EDGES) - 1)],
            }
            a = results[(arm, variant)]
            print(f"\narm {arm.upper()} / {variant}: "
                  f"treated {a['n_treated_total']}, "
                  f"seeds contributing "
                  f"{a['seed_branch_counts']['CONTRIBUTES']}/{n_seeds}, "
                  f"no-treated {a['seed_branch_counts']['NO_TREATED_FEATURES']}"
                  f", branches {a['treated_branch_counts']}")

    # ---- the caliper boundary, as measured --------------------------------
    print(f"\ncaliper boundary: {boundary_hits} of {boundary_examined} "
          f"treated-control distances land EXACTLY on {args.caliper}")

    # ---- cross-check against BT's committed pool helper --------------------
    # An independent second implementation of the same rule, on the same data.
    # It is a CHECK, not a prediction: two implementations of one specification
    # must agree, and that is a structural necessity (gotcha 44).
    from src.analysis.match_pool_diagnostics import nearest_within_caliper
    agree = True
    for arm in arms:
        for s in range(n_seeds):
            cell = cells[arm][s]
            t_idx = np.flatnonzero((cell == CELL_ALIGNED_BOTH)
                                   | (cell == CELL_ALIGNED_EXP))
            c_idx = np.flatnonzero(cell == CELL_ORPHAN)
            if t_idx.size == 0:
                continue
            lr = log10_rate(fire[s], n_rows)
            mine = np.array([x.size for x in
                             match_sets(lr[t_idx], lr[c_idx], args.caliper)])
            theirs = nearest_within_caliper(lr[t_idx], lr[c_idx],
                                            args.caliper)["n_at_min"]
            if not np.array_equal(mine, theirs):
                agree = False
                raise SystemExit(
                    f"arm {arm} seed {ap_seeds[s]}: this module's matched-set "
                    "sizes disagree with match_pool_diagnostics' independently "
                    f"written multiplicities in "
                    f"{int(np.count_nonzero(mine != theirs))} treated features")
    print(f"independent agreement with match_pool_diagnostics: {agree}")

    # ---- write -------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as out:
        out.create_dataset("seeds", data=np.array(ap_seeds[:n_seeds],
                                                  dtype=h5py.string_dtype()))
        out.create_dataset("rf_row_for_seed", data=np.array(rf_row[:n_seeds],
                                                            dtype=np.int64))
        out.create_dataset("bin_edges", data=np.array(BIN_EDGES,
                                                      dtype=np.float64))
        for arm in arms:
            for variant in variants:
                g = out.create_group(f"{arm}/{variant}")
                rows = results[(arm, variant)]["per_seed"]
                g.create_dataset("n_treated", data=np.array(
                    [r["n_treated"] for r in rows], dtype=np.int64))
                g.create_dataset("n_pairs", data=np.array(
                    [r["n_pairs"] for r in rows], dtype=np.int64))
                g.create_dataset("n_unmatchable", data=np.array(
                    [r.get("n_unmatchable", 0) for r in rows], dtype=np.int64))
                g.create_dataset("median_difference", data=np.array(
                    [np.nan if r["median_difference"] is None
                     else r["median_difference"] for r in rows],
                    dtype=np.float64))
                g.create_dataset("p_value", data=np.array(
                    [np.nan if r["p_value"] is None else r["p_value"]
                     for r in rows], dtype=np.float64))
                g.create_dataset("seed_contributes", data=np.array(
                    [r["seed_branch"] == "CONTRIBUTES" for r in rows],
                    dtype=bool))
                g.attrs["seed_branch_counts"] = json.dumps(
                    results[(arm, variant)]["seed_branch_counts"])
                g.attrs["treated_branch_counts"] = json.dumps(
                    results[(arm, variant)]["treated_branch_counts"])
        out.attrs["treated_branches"] = json.dumps(list(TREATED_BRANCHES))
        out.attrs["test_branches"] = json.dumps(list(TEST_BRANCHES))
        out.attrs["seed_branches"] = json.dumps(list(SEED_BRANCHES))
        out.attrs["caliper"] = float(args.caliper)
        out.attrs["match_on"] = "log10 firing rate"
        out.attrs["tie_rule"] = "ALL tied candidates, weighted equally (§5a)"
        out.attrs["replacement"] = "WITH replacement (§5a)"
        out.attrs["empty_seed_rule"] = (
            "a seed contributing zero treated features is an explicit "
            "ignorance category with its own count; never dropped, never an "
            "informative zero (§5b)")
        out.attrs["threshold_grid_point"] = 1
        out.attrs["n_compare"] = n_compare
        out.attrs["n_rows"] = n_rows
        out.attrs["governing_amendment"] = "BU (§5, AN3, §5a, §5b; BM-E1; BP4)"
        out.attrs["classified_unit"] = "(seed, feature) pair; BQ4a"
        out.attrs["is_cost_pilot"] = bool(args.seeds is not None)
        out.attrs["git_sha"] = git_sha()
        out.attrs["git_dirty"] = bool(git_dirty())
        # IDENTITY: nothing back-filled (criterion 10). A key absent upstream
        # stays absent here and compares ABSENT_ON_BOTH, never AGREE.
        for src_, pre in ((ap_identity, "ap_experimental"),
                          (c3_identity, "classification_2x2")):
            for k in ("library_sha", "n_concepts", "argmax_pool_n",
                      "argmax_pool_used_n", "experimental_argmax_pool_used_n",
                      "scored_row_set", "pooled_scored_row_set", "negatives",
                      "experimental_negatives", "stratum",
                      "experimental_stratum", "truncation_policy",
                      "pooled_truncation_policy", "n_shifts", "n_rows",
                      "ap_impl", "experimental_ap_impl", "q_fdr", "gain_floor",
                      "gather_device_class", "ap_arith_device_class",
                      "experimental_library_sha", "pooled_library_sha"):
                if k in src_:
                    out.attrs[f"{pre}_{k}"] = src_[k]
        for k in ("theta", "rule", "similarity", "n_compare_seeds",
                  "n_anchors", "git_sha", "measurement"):
            if k in rf_attrs:
                out.attrs[f"rf_counts_{k}"] = rf_attrs[k]
        for nm, v in (("ap_experimental", args.ap_experimental),
                      ("classification_2x2", args.classification_2x2),
                      ("rf_counts", args.rf_counts),
                      ("corrector_scan", args.corrector_scan)):
            out.attrs[f"input_{nm}"] = str(v)

    report = {
        "artifact": "bu_matched_pair_arm",
        "governing_amendment": "BU",
        "component": "5 -- the firing-rate-matched paired arm",
        "git_sha": git_sha(), "git_dirty": bool(git_dirty()),
        "is_cost_pilot": bool(args.seeds is not None),
        "threshold_grid_point": 1,
        "caliper": float(args.caliper),
        "match_on": "log10 firing rate",
        "n_seeds": n_seeds, "n_features": n_features, "n_compare": n_compare,
        "seed_join_ap_to_rf": dict(zip(ap_seeds[:n_seeds], rf_row[:n_seeds])),
        "corrector_exclusion": {
            "n_flagged_total": n_flagged_total,
            "per_seed": {s: flagged[s] for s in ap_seeds[:n_seeds]},
            "note": "§5 mandates the arm be reported WITH and WITHOUT the "
                    "flagged features even when none are flagged",
        },
        "caliper_boundary": {
            "n_distances_exactly_at_the_caliper": boundary_hits,
            "n_distances_examined": boundary_examined,
            "predicate": "d <= caliper, inclusive as §5 pins it",
            "note": "corrections-log entry 142: at the pinned width of 0.1 the "
                    "boundary is not reachable in float64. MEASURED here "
                    "rather than assumed in either direction",
        },
        "resolution_floor": {
            "relation": "exact two-sided signed-rank on n non-zero "
                        "differences cannot return p below 2**(1-n)",
            "table": {str(n): min_attainable_two_sided_p(n)
                      for n in (1, 2, 3, 4, 5, 6, 8, 10, 20)},
            "note": "a RELATION, not a verdict; this arm declares no alpha "
                    "and names no fraction a design must reach (gotcha 42)",
        },
        "independent_agreement_with_match_pool_diagnostics": bool(agree),
        "per_arm": {},
        "inputs": {
            "ap_experimental": str(args.ap_experimental),
            "classification_2x2": str(args.classification_2x2),
            "rf_counts": str(args.rf_counts),
            "corrector_scan": str(args.corrector_scan),
        },
    }
    for arm in arms:
        report["per_arm"][arm.upper()] = {
            v: results[(arm, v)] for v in variants}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}\nwrote {args.report}")


if __name__ == "__main__":
    main()
