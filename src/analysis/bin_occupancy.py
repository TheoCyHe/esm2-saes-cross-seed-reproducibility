"""BIN OCCUPANCY for the two unexplained hard-bin C1 seeds.
PREREGISTRATION amendment AK3 prediction B8 (brief §6b).

THE QUESTION. AJ5 measured hard-bin C1 and found the median `|rho|` at 0.00709
-- 0.72 SE, indistinguishable from zero -- beside two ISOLATED extreme seeds
(0 at 0.1418 and 1 at 0.1336, then a 7.7x gap to third). AJ5 also killed the
obvious explanation: the apparent `|b_dec|` relationship dies on leave-one-out
and the sign even flips. So the tail is recorded as UNEXPLAINED.

THE CANDIDATE EXPLANATION THIS MODULE TESTS. Hard-bin `mu_hat` is a STEP
FUNCTION taking few distinct values -- one per (concept cell, firing bin) --
so `rho(mu_hat, r_f)` is a correlation against a coarsely discretised variable
whose effective scale depends on how features are spread across firing bins. If
seeds 0 and 1 are occupancy outliers, the tail has a discretisation
explanation. If they are not, it does not, and that is recorded as unexplained
rather than left as a loose end.

THIS IS NOT A FIREWALL CROSSING. It reads `fire_rate` and `stratum_mu` and
nothing else -- no AP matrix, no `r_f`, no label store. It is deliberately its
own module rather than a third arm of `m4_confound.py`, so that a measurement
which needs no crossing does not inherit one.

=============================================================================
THE PREDICATE, PRE-COMMITTED IN AMENDMENT AK3 B8 BEFORE THIS RAN
=============================================================================
Rank all 20 seeds on

  (i)  `n_distinct_mu` -- the count of DISTINCT hard-bin `mu_hat` values; and
  (ii) `effective_bins` -- exp(Shannon entropy) of the per-seed firing-bin
       occupancy distribution, i.e. the effective number of occupied bins.

"EXTREME" IS ABSOLUTE DEVIATION FROM THE ACROSS-SEED MEDIAN, in either
direction. That is fixed here, in code, before the numbers exist, because a
direction chosen afterwards is a predicate chosen with the answer in hand --
and B8 named the statistic without pinning the direction, which is the
incomplete-instrument shape of gotcha 24.

  HIT  if BOTH seed 0 and seed 1 are among the THREE most extreme seeds on at
       least one of the two statistics.
  MISS otherwise -- in which case the 7.7x tail gap has NO discretisation
       explanation and is recorded as unexplained. That is a finding, not a
       null result.

One shot either way. The median is the story (AJ5) and per-seed fits are
meta-analysed, so two outliers move the pooled estimate slightly at most.

WHY exp(ENTROPY) AND NOT A BIN COUNT. A bare count of non-empty bins is
insensitive to a seed that puts 99.9% of its features in one bin while
scattering a handful across the rest -- exactly the configuration that would
coarsen the correlation. exp(H) is the standard effective-count answer to that,
it is monotone in spread, and it equals the bin count exactly when occupancy is
uniform. The RAW per-bin counts are reported alongside, so nobody has to take
the summary on trust.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import h5py
import numpy as np

from src.analysis.m4_confound import hard_bin_mu_per_feature

# The pre-committed predicate's one free parameter, fixed in AK3 B8.
N_MOST_EXTREME = 3
FOCUS_SEEDS = (0, 1)


def effective_bins(counts: np.ndarray) -> float:
    """exp(Shannon entropy) of an occupancy vector: the effective bin count.

    Equals the number of bins exactly when occupancy is uniform, and falls to 1
    when every feature is in one bin. Empty bins contribute nothing (0 log 0
    taken as 0), which is the standard convention and is stated because the
    alternative silently rewards emptiness.
    """
    total = counts.sum()
    if total <= 0:
        return float("nan")
    p = counts.astype(np.float64) / total
    nz = p[p > 0]
    return float(np.exp(-(nz * np.log(nz)).sum()))


def occupancy_per_seed(fire_rate: np.ndarray, edges: list[float]) -> dict:
    """Per-seed firing-bin occupancy, using the SAME assignment the normaliser
    uses -- `bin_index` with the stored `firing_rate_edges`, clipped identically.

    Recomputing the assignment a second, subtly different way here would make
    any disagreement uninterpretable, so the one function is reused.
    """
    from src.analysis.null_pooling import bin_index

    n_bins = len(edges) - 1
    rows = []
    for s in range(fire_rate.shape[0]):
        fb = np.clip(bin_index(fire_rate[s], edges), 0, n_bins - 1)
        counts = np.bincount(fb.astype(np.int64), minlength=n_bins)
        rows.append({
            "seed": int(s),
            "counts_by_bin": counts.tolist(),
            "n_nonempty_bins": int((counts > 0).sum()),
            "effective_bins": effective_bins(counts),
            "max_bin_share": float(counts.max() / counts.sum()),
        })
    return {"n_bins": int(n_bins), "edges": list(edges), "per_seed": rows}


def distinct_mu_per_seed(mu_feat: np.ndarray) -> list[dict]:
    """How many DISTINCT values the step function actually takes, per seed.

    `mu_feat` is the per-feature hard-bin normaliser -- a median over concepts
    of a (cell, firing bin) lookup -- so its distinct-value count is the real
    resolution of the variable the correlation is taken against, not the
    nominal cell x bin product.

    AND THE COUNT IS ANALYTICALLY BOUNDED BY THE NUMBER OF FIRING BINS, which is
    the substantive finding of this module (amendment AL2) and is not obvious
    from `hard_bin_mu_per_feature`'s code. That function computes

        m[c, f]    = stratum_mu[seed][concept_cell[c], feature_firing_bin[f]]
        mu_feat[f] = median over c of m[c, f]

    For a fixed `f` the index `feature_firing_bin[f]` is fixed, so the column is
    a FIXED set of 712 values determined by the firing bin alone. **The median
    over concepts annihilates the concept-cell axis entirely**, so `mu_feat` is
    a function of the firing bin and takes at most `n_firing_bins` distinct
    values -- measured at exactly 6 of 6 in all 20 seeds.

    WHY THIS MATTERS FOR AI/AJ5. The hard-bin C1 statistic is
    `Spearman(mu_hat, r_f)`, so it is a correlation against a SIX-LEVEL step
    function of firing rate. It is therefore not measuring a concept-side
    confound at all, and it is not the same kind of object as the smooth arm's
    C1 -- whose `mu_hat` depends on the feature-side covariates `perplexity` and
    `concentration` and so has genuine per-feature resolution.

    This does NOT overturn AI's finding that hard-bin's confound is smaller; it
    explains it, and it changes what the number licenses. Hard-bin's low `|rho|`
    is a consequence of its normaliser having almost no per-feature variation,
    not evidence that hard-bin normalisation is confound-free in a way the
    smooth arm is not. Reported, not acted on.
    """
    out = []
    for s in range(mu_feat.shape[0]):
        v = mu_feat[s]
        finite = v[np.isfinite(v)]
        out.append({
            "seed": int(s),
            "n_distinct_mu": int(np.unique(finite).size),
            "n_finite": int(finite.size),
            "n_features": int(v.size),
        })
    return out


def predictor_denominator_resolution(gtilde_path: Path,
                                     fire_rate: np.ndarray) -> dict:
    """How many DISTINCT denominators the PREDICTOR divides by, per feature.

    AMENDMENT AM3, and it exists to settle a question AL2 left inferable.
    AL2 measured that `hard_bin_mu_per_feature`'s per-feature summary takes only
    6 values, and that summary is built ONLY for the C1 correlation. It is NOT
    what the predictor divides by, and the distinction is large enough that
    leaving it to be inferred from the phrase "hard-bin `mu_hat`" was a
    reporting defect.

    THE PREDICTOR, quoted from `standardize_gain.standardize_seed`:

        mu_real_grid = _grid(mu_full, cell, fb)   # [c,f] = mu_full[cell[c], fbin[f]]
        z_real       = g / mu_real_grid           # inner_mode "ratio"
        M, argc      = masked_argmax(z_real, pool_mask)

    `_grid` broadcasts to `(concepts, features)` and **varies with `c`**, so the
    denominator for a fixed feature `f` at firing bin `b` ranges over
    `{mu_full[cell(c), b] : c in pool}` -- one value per occupied concept cell,
    bounded by `n_cells`, not by `n_fbins`. Measured here rather than argued.

    So: **the four-axis stratification is doing its work in the predictor.** What
    is coarse is the C1 SUMMARY, and therefore the C1 statistic -- which is
    exactly and only what AL2's consequence applies to.
    """
    import json as _json

    with h5py.File(gtilde_path, "r") as hf:
        stratum_mu = hf["stratum_mu"][:]              # (seeds, cells, fbins)
        cell = hf["concept_cell"][:]                  # (concepts,)
        pool = (hf["in_argmax_pool"][:] if "in_argmax_pool" in hf
                else np.ones(cell.shape, dtype=bool))
        _fre = hf.attrs["firing_rate_edges"]
        if isinstance(_fre, bytes):
            _fre = _fre.decode()
        edges = _json.loads(_fre) if isinstance(_fre, str) else list(_fre)

    from src.analysis.null_pooling import bin_index

    n_bins = len(edges) - 1
    n_cells = stratum_mu.shape[1]
    pool_cells = cell[pool]
    # For a fixed firing bin `b`, the denominator varies over the DISTINCT
    # pool concept cells whose stratum is finite at `b`.
    per_seed = []
    for s in range(stratum_mu.shape[0]):
        fb = np.clip(bin_index(fire_rate[s], edges), 0, n_bins - 1).astype(np.int64)
        # distinct denominators available at each firing bin
        by_bin = []
        for b in range(n_bins):
            vals = stratum_mu[s][np.maximum(pool_cells, 0), b]
            vals = vals[np.isfinite(vals)]
            by_bin.append(int(np.unique(vals).size))
        # each feature sits in one bin, so its own denominator count is by_bin[b]
        counts = np.array([by_bin[b] for b in fb])
        per_seed.append({
            "seed": int(s),
            "distinct_denominators_by_firing_bin": by_bin,
            "median_per_feature": float(np.median(counts)),
            "min_per_feature": int(counts.min()),
            "max_per_feature": int(counts.max()),
        })
    med = float(np.median([r["median_per_feature"] for r in per_seed]))
    return {
        "n_concept_cells": int(n_cells),
        "n_firing_bins": int(n_bins),
        "n_pool_concepts": int(pool.sum()),
        "per_seed": per_seed,
        "median_distinct_denominators_per_feature": med,
        # DERIVED, and it is the answer to the question AL2 left open.
        "predictor_denominator_is_firing_bin_only": bool(med <= 1.0),
        "reading": (
            "the PREDICTOR divides g_fc by mu_s(cell(c), fbin(f)), which varies "
            "with the concept -- so it retains full concept-axis resolution and "
            "takes this many distinct values per feature. The 6-valued object "
            "in AL2 is the C1 SUMMARY (a median over concepts) and nothing "
            "else. AL2's consequence applies to the C1 STATISTIC only "
            "(amendment AM3)."),
    }


def check_resolution_bound(distinct: list[dict], n_bins: int) -> dict:
    """AL2's analytic bound: `mu_hat` cannot take more values than there are
    firing bins, because the median over concepts annihilates the cell axis.

    A SEPARATE FUNCTION SO THE DETECTOR CAN BE TESTED AGAINST A VIOLATION.
    Its first version was inline in `main()`, and the mutation that disabled the
    comparison SURVIVED the suite -- the tests only checked that the bound holds
    on valid data, which is a check that cannot fail (gotcha 13). Real data
    cannot violate an analytic bound, so the detector is only testable if the
    violating input can be supplied directly.
    """
    over = [d["seed"] for d in distinct if d["n_distinct_mu"] > n_bins]
    counts = [d["n_distinct_mu"] for d in distinct]
    return {
        "n_firing_bins": int(n_bins),
        "n_distinct_mu_by_seed": counts,
        # DERIVED from the comparison just made (gotcha 20).
        "analytic_bound_holds": not over,
        "seeds_over_bound": over,
        "mu_hat_resolution_equals_bin_count": all(c == n_bins for c in counts),
        "reading": (
            "the median over concepts annihilates the concept-cell axis, so "
            "hard-bin mu_hat is a function of the firing bin alone. AI/AJ5's "
            "hard-bin C1 is therefore a correlation against a step function "
            "with this many levels, not a concept-side confound measurement "
            "(amendment AL2). Reported, not acted on."),
    }


def extreme_ranks(values: dict[int, float]) -> dict:
    """Rank seeds by |value - median| descending. Ties broken by seed index, so
    the ranking is deterministic rather than dependent on dict order.

    A CONSTANT STATISTIC IS REPORTED `UNINFORMATIVE`, NOT RANKED, and this was
    added after the first run got it wrong (amendment AL1). `n_distinct_mu` came
    back as exactly 6 for all 20 seeds -- zero spread -- so every absolute
    deviation was 0.0 and the tie-break by seed index put seeds 0, 1, 2 in the
    "three most extreme". The predicate then read HIT because the two focus
    seeds are the two lowest indices. **The tie-break supplied the answer.**

    That is gotcha 20 in a new shape: not a printed sentence asserting an
    expectation, but a RANKING PREDICATE APPLIED TO A CONSTANT, where the
    deterministic tie-break -- added for reproducibility -- silently becomes the
    verdict. A ranking of identical values carries no information about which
    unit is unusual, and a predicate that consumes one must say so rather than
    return a position.
    """
    seeds = sorted(values)
    arr = np.array([values[s] for s in seeds], dtype=np.float64)
    med = float(np.nanmedian(arr))
    spread = float(np.nanmax(arr) - np.nanmin(arr))
    n_unique = int(np.unique(arr[np.isfinite(arr)]).size)
    dev = {s: abs(values[s] - med) for s in seeds}
    order = sorted(seeds, key=lambda s: (-dev[s], s))
    informative = bool(n_unique > 1)
    return {
        "median": med,
        "spread": spread,
        "n_unique_values": n_unique,
        # DERIVED from the spread just computed, never asserted (gotcha 20).
        "informative": informative,
        "status": "INFORMATIVE" if informative else "UNINFORMATIVE_CONSTANT",
        "value_by_seed": {str(s): float(values[s]) for s in seeds},
        "abs_deviation_by_seed": {str(s): float(dev[s]) for s in seeds},
        "seeds_by_extremeness": [int(s) for s in order],
        # An extremeness ranking of a constant is a ranking of the tie-break.
        # Reported as None so a consumer cannot read a position out of it.
        "three_most_extreme": ([int(s) for s in order[:N_MOST_EXTREME]]
                               if informative else None),
    }


def score_b8(distinct: list[dict], occ: dict) -> dict:
    """AK3 B8, DERIVED from the two rankings rather than asserted (gotcha 20)."""
    stats = {
        "n_distinct_mu": extreme_ranks(
            {d["seed"]: float(d["n_distinct_mu"]) for d in distinct}),
        "effective_bins": extreme_ranks(
            {r["seed"]: float(r["effective_bins"]) for r in occ["per_seed"]}),
    }
    per_stat = {}
    for name, rk in stats.items():
        if not rk["informative"]:
            per_stat[name] = {
                "status": rk["status"],
                "spread": rk["spread"],
                "n_unique_values": rk["n_unique_values"],
                "three_most_extreme": None,
                # A constant cannot be evidence either way. NOT False -- False
                # would read as "checked and absent", and this was not checked.
                "both_focus_seeds_present": None,
                "focus_seed_ranks": None,
                "why_excluded": (
                    "every seed has the same value, so an extremeness ranking "
                    "is a ranking of the tie-break, not of the data. Excluded "
                    "from the predicate rather than counted as satisfying it "
                    "(amendment AL1)."),
            }
            continue
        top = set(rk["three_most_extreme"])
        per_stat[name] = {
            "status": rk["status"],
            "spread": rk["spread"],
            "n_unique_values": rk["n_unique_values"],
            "three_most_extreme": rk["three_most_extreme"],
            "both_focus_seeds_present": bool(set(FOCUS_SEEDS) <= top),
            "focus_seed_ranks": {
                str(s): int(rk["seeds_by_extremeness"].index(s)) + 1
                for s in FOCUS_SEEDS},
        }
    informative = [v for v in per_stat.values() if v["both_focus_seeds_present"]
                   is not None]
    hit = any(v["both_focus_seeds_present"] for v in informative)
    return {
        "predicate": (
            f"HIT if both seeds {list(FOCUS_SEEDS)} are among the "
            f"{N_MOST_EXTREME} most extreme (|value - median|, either "
            f"direction) on at least one of the two statistics. A statistic "
            f"with ZERO SPREAD across seeds is UNINFORMATIVE and is excluded "
            f"from the predicate -- ranking identical values ranks the "
            f"tie-break (amendment AL1)."),
        "category": "within-system comparison",
        "n_informative_statistics": len(informative),
        "statistics": stats,
        "by_statistic": per_stat,
        "outcome": ("NOT_SCORABLE" if not informative
                    else "HIT" if hit else "MISS"),
        "reading_if_MISS": (
            "the 7.7x hard-bin C1 tail gap has NO discretisation explanation "
            "and is recorded as UNEXPLAINED (AJ5). That is a finding, not a "
            "null result, and it is stated here so the outcome cannot be "
            "reframed after the fact."),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gtilde", type=Path, required=True,
                    help="the gtilde h5 whose stratum_mu IS the hard-bin "
                         "normaliser")
    ap.add_argument("--null-reductions", type=Path, required=True,
                    help="carries fire_rate (seeds x features)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from src.sae.train import git_dirty, git_sha

    with h5py.File(args.null_reductions, "r") as hf:
        fire_rate = hf["fire_rate"][:]
    mu_feat, prov = hard_bin_mu_per_feature(args.gtilde, fire_rate)

    occ = occupancy_per_seed(fire_rate, prov["firing_rate_edges"])
    distinct = distinct_mu_per_seed(mu_feat)
    b8 = score_b8(distinct, occ)

    print(f"[occ] {fire_rate.shape[0]} seeds, {occ['n_bins']} firing bins, "
          f"edges {occ['edges']}", flush=True)
    print("[occ] seed  n_distinct_mu  eff_bins  nonempty  max_bin_share",
          flush=True)
    for d, r in zip(distinct, occ["per_seed"]):
        print(f"[occ] {d['seed']:>4}  {d['n_distinct_mu']:>13}  "
              f"{r['effective_bins']:>8.4f}  {r['n_nonempty_bins']:>8}  "
              f"{r['max_bin_share']:>13.4f}", flush=True)
    print("[occ] B8 " + json.dumps(b8["by_statistic"], indent=2), flush=True)
    print(f"[occ] B8 OUTCOME: {b8['outcome']}", flush=True)

    # AL2: the analytic bound, ASSERTED rather than described. `mu_feat` is a
    # function of the firing bin alone, so its distinct-value count cannot
    # exceed the bin count. A violation would mean the median-over-concepts
    # reduction is not what the docstring says it is.
    resolution = check_resolution_bound(distinct, occ["n_bins"])
    denom = predictor_denominator_resolution(args.gtilde, fire_rate)
    if not resolution["analytic_bound_holds"]:
        raise SystemExit(
            "ANALYTIC BOUND VIOLATED: seeds %s report more distinct mu_hat "
            "values than there are firing bins (%d). mu_feat is a median over "
            "concepts at a fixed firing bin and cannot do that -- the "
            "reduction is not what it is documented to be."
            % (resolution["seeds_over_bound"], resolution["n_firing_bins"]))
    print(f"[occ] mu_hat resolution: {resolution['n_distinct_mu_by_seed']} "
          f"distinct values vs {resolution['n_firing_bins']} firing bins; "
          f"analytic bound holds: {resolution['analytic_bound_holds']}; "
          f"equals bin count: "
          f"{resolution['mu_hat_resolution_equals_bin_count']}", flush=True)

    print(f"[occ] AM3 PREDICTOR denominator: median "
          f"{denom['median_distinct_denominators_per_feature']:.1f} distinct "
          f"values per feature over {denom['n_concept_cells']} concept cells "
          f"(vs {denom['n_firing_bins']} firing bins for the C1 summary); "
          f"firing-bin-only: "
          f"{denom['predictor_denominator_is_firing_bin_only']}", flush=True)

    out = {
        "identity": {
            "measurement": "bin_occupancy_AK_B8",
            "amendment": "AK",
            "firewall_crossing": "NONE -- reads fire_rate and stratum_mu only",
            "gtilde": str(args.gtilde),
            "null_reductions": str(args.null_reductions),
            "n_seeds": int(fire_rate.shape[0]),
            "n_features": int(fire_rate.shape[1]),
            "n_most_extreme": N_MOST_EXTREME,
            "focus_seeds": list(FOCUS_SEEDS),
            "git_sha": git_sha(),
            "git_dirty": git_dirty(),
            "python_version": platform.python_version(),
        },
        "provenance": prov,
        "occupancy": occ,
        "distinct_mu": distinct,
        "mu_hat_resolution": resolution,
        "predictor_denominator_resolution": denom,
        "B8": b8,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[occ] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
