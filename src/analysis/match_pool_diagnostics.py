"""BT part B4: two properties of the firing-rate matching POOL.

WHAT THESE ARE FOR, AND WHAT THEY MUST NOT BE USED FOR
=============================================================================
BT-B3 fixes two degrees of freedom of the matched-pair robustness arm on
STRUCTURAL grounds: ties are resolved by using ALL tied candidates weighted
equally, and matching is WITH replacement. **Those choices are not conditional
on anything measured here.** These diagnostics quantify how much the choice
COULD have mattered; they do not select it. Changing the specification in
response to them would be conditioning a design choice on data, which BT-E does
not authorise.

They are admissible to run before the arm is built for one reason: they are
properties of the POOL -- who is available to match whom -- and not of any
estimate. Neither reads `r_f`, the outcome the arm compares. A number that
cannot move with the result cannot be selected on.

THE TWO QUANTITIES
=============================================================================
`tie_multiplicity` -- for each treated (ALIGNED) feature, how many control
(ORPHAN) features sit at EXACTLY the minimum |delta log10 firing rate| within
the caliper. One means the nearest control is unique and the tie rule never
fires; more means it does.

    Exact ties are expected rather than exotic here, and the reason is
    arithmetic: firing rate is `fire_count / n_rows` with `n_rows` a constant,
    so two features are exactly equidistant from a treated feature whenever
    their fire_counts are EQUAL -- which is common at low counts. The
    comparison is therefore made on the computed float64 distance, where equal
    counts give bit-identical distances, and never on a tolerance.

`control_reuse` -- under matching WITH replacement, how many times each control
is used. Reuse is the real cost of replacement (it induces correlation between
pairs), and BT-B3's ground for accepting it is that this cost is MEASURABLE and
reportable where an order dependence would not be. This is that measurement.

A treated feature with NO control inside the caliper is counted in its own
category and never folded into a denominator (gotcha 5's mandatory unmatched
count; criterion 6).
"""

from __future__ import annotations

import numpy as np

__all__ = ["log10_rate", "nearest_within_caliper", "pool_diagnostics"]


def log10_rate(fire_counts: np.ndarray, n_rows: int) -> np.ndarray:
    """`log10(fire_count / n_rows)`, with a zero count returned as NaN.

    NaN rather than `-inf`: a dead feature has no log firing rate, and `-inf`
    would participate in a minimum and silently become somebody's nearest
    control. The ensemble has no dead features (0 in all 20 seeds, CLAUDE.md),
    so this branch is expected to be empty -- which is exactly why it must be
    written rather than assumed.
    """
    c = np.asarray(fire_counts, dtype=np.float64)
    if int(n_rows) <= 0:
        raise ValueError(f"n_rows is {n_rows}; a rate needs a positive "
                         "denominator")
    out = np.full(c.shape, np.nan, dtype=np.float64)
    pos = c > 0
    out[pos] = np.log10(c[pos] / float(n_rows))
    return out


def nearest_within_caliper(treated: np.ndarray, control: np.ndarray,
                           caliper: float) -> dict:
    """For each treated value, the controls at the minimum distance <= caliper.

    `treated` and `control` are 1-D log10 firing rates; NaN entries are
    dropped and counted rather than compared. Returns per-treated arrays:
    `n_at_min` (0 when nothing is inside the caliper), `min_distance`, and the
    aggregate `use_count` over controls under WITH-replacement matching where
    every tied control is used.

    THE TIE COMPARISON IS EXACT. Two controls with equal fire_counts give
    bit-identical distances, so `== d_min` finds them; a tolerance here would
    invent ties that the arithmetic does not have and hide the distinction the
    diagnostic exists to measure.
    """
    t = np.asarray(treated, dtype=np.float64)
    c = np.asarray(control, dtype=np.float64)
    if t.ndim != 1 or c.ndim != 1:
        raise ValueError("treated and control must both be 1-D")
    if not np.isfinite(caliper) or caliper <= 0:
        raise ValueError(f"caliper is {caliper}; it must be positive and "
                         "finite -- a non-positive caliper admits nothing and "
                         "would report every treated feature as unmatchable")
    t_ok = np.isfinite(t)
    c_ok = np.isfinite(c)
    cv = c[c_ok]
    n_at_min = np.zeros(t.size, dtype=np.int64)
    min_d = np.full(t.size, np.nan, dtype=np.float64)
    use_count = np.zeros(c.size, dtype=np.int64)
    c_idx = np.flatnonzero(c_ok)
    for i in np.flatnonzero(t_ok):
        d = np.abs(cv - t[i])
        inside = d <= caliper
        if not inside.any():
            continue
        dm = d[inside].min()
        at = inside & (d == dm)
        n_at_min[i] = int(np.count_nonzero(at))
        min_d[i] = float(dm)
        use_count[c_idx[at]] += 1
    return {
        "n_at_min": n_at_min,
        "min_distance": min_d,
        "use_count": use_count,
        "n_treated": int(t.size),
        "n_treated_no_rate": int((~t_ok).sum()),
        "n_control": int(c.size),
        "n_control_no_rate": int((~c_ok).sum()),
        "n_treated_unmatchable": int(np.count_nonzero(t_ok & (n_at_min == 0))),
    }


def pool_diagnostics(fire_counts: np.ndarray, n_rows: int,
                     aligned: np.ndarray, orphan: np.ndarray,
                     caliper: float, seed_labels: list[str]) -> dict:
    """Both diagnostics per seed, plus an across-seed summary with denominators.

    `fire_counts` is (n_seeds, n_features); `aligned` and `orphan` are boolean
    masks of the same shape. EVERY reported rate states its denominator, and a
    seed with no treated feature is reported as such rather than dropped -- arm
    A has ALIGNED = 0 in all twenty seeds by the resolution floor, and a
    diagnostic that silently skipped those seeds would read as a measurement
    of a pool that was never examined.
    """
    fc = np.asarray(fire_counts)
    al = np.asarray(aligned, dtype=bool)
    orp = np.asarray(orphan, dtype=bool)
    if not (fc.shape == al.shape == orp.shape) or fc.ndim != 2:
        raise ValueError(f"shapes disagree: fire_counts {fc.shape}, aligned "
                         f"{al.shape}, orphan {orp.shape}; all must be "
                         "(n_seeds, n_features)")
    if len(seed_labels) != fc.shape[0]:
        raise ValueError(f"{len(seed_labels)} seed labels for {fc.shape[0]} "
                         "rows -- a seed axis is joined on labels, never by "
                         "position (gotcha 29)")
    if (al & orp).any():
        raise ValueError("a feature is both ALIGNED and ORPHAN in some seed; "
                         "the two cells are disjoint by construction (AV5) "
                         "and a treated feature must not be its own control")

    rows, ties, reuse = [], [], []
    seeds_without_treated = []
    for s in range(fc.shape[0]):
        lr = log10_rate(fc[s], n_rows)
        t_idx = np.flatnonzero(al[s])
        c_idx = np.flatnonzero(orp[s])
        if t_idx.size == 0:
            seeds_without_treated.append(seed_labels[s])
            rows.append({"seed": seed_labels[s], "n_treated": 0,
                         "n_control": int(c_idx.size),
                         "note": "no ALIGNED feature in this seed; reported "
                                 "as zero, never dropped"})
            continue
        r = nearest_within_caliper(lr[t_idx], lr[c_idx], caliper)
        matched = r["n_at_min"] > 0
        used = r["use_count"][r["use_count"] > 0]
        rows.append({
            "seed": seed_labels[s],
            "n_treated": int(t_idx.size),
            "n_control": int(c_idx.size),
            "n_treated_unmatchable": r["n_treated_unmatchable"],
            "n_treated_matched": int(matched.sum()),
            "tie_multiplicity_mean": (float(r["n_at_min"][matched].mean())
                                      if matched.any() else None),
            "tie_multiplicity_median": (float(np.median(r["n_at_min"][matched]))
                                        if matched.any() else None),
            "tie_multiplicity_max": (int(r["n_at_min"][matched].max())
                                     if matched.any() else None),
            "frac_treated_with_a_unique_nearest": (
                float((r["n_at_min"][matched] == 1).mean())
                if matched.any() else None),
            "n_distinct_controls_used": int(used.size),
            "control_reuse_mean": float(used.mean()) if used.size else None,
            "control_reuse_max": int(used.max()) if used.size else None,
            "frac_used_controls_reused": (float((used > 1).mean())
                                          if used.size else None),
        })
        ties.append(r["n_at_min"][matched])
        reuse.append(used)

    all_ties = np.concatenate(ties) if ties else np.zeros(0, dtype=np.int64)
    all_reuse = np.concatenate(reuse) if reuse else np.zeros(0, dtype=np.int64)
    return {
        "caliper": float(caliper),
        "n_rows": int(n_rows),
        "per_seed": rows,
        "seeds_with_no_treated_feature": seeds_without_treated,
        "n_seeds": int(fc.shape[0]),
        "across_seeds": {
            "denominator_treated_matched_pairs": int(all_ties.size),
            "denominator_distinct_controls_used": int(all_reuse.size),
            "tie_multiplicity_mean": (float(all_ties.mean())
                                      if all_ties.size else None),
            "tie_multiplicity_median": (float(np.median(all_ties))
                                        if all_ties.size else None),
            "tie_multiplicity_max": (int(all_ties.max())
                                     if all_ties.size else None),
            "frac_treated_with_a_unique_nearest": (
                float((all_ties == 1).mean()) if all_ties.size else None),
            "control_reuse_mean": (float(all_reuse.mean())
                                   if all_reuse.size else None),
            "control_reuse_max": (int(all_reuse.max())
                                  if all_reuse.size else None),
            "frac_used_controls_reused": (float((all_reuse > 1).mean())
                                          if all_reuse.size else None),
        },
        "note": "PROPERTIES OF THE MATCHING POOL, NOT OF ANY ESTIMATE. "
                "Neither quantity reads r_f. BT-B4: these do NOT condition "
                "BT-B3's choices, which are fixed on structural grounds.",
    }
