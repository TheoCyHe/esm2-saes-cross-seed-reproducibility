"""AL2 RESTATED — C1 ON THE QUANTITY THAT ENTERS THE PREDICTOR.
PREREGISTRATION amendment AO section 7; the author's ratified decision AM6.2(b).

*** THIS JOB DELIBERATELY CROSSES THE X7 FIREWALL. ***
It joins `r_f` to the normaliser, exactly as `m4_confound.py --c1-only` does and
for the same authorised reason: C1 is a SPECIFICATION question. The crossing is
SCOPED (correlations and conditional means, no model fit, no regression),
LABELLED (`firewall_crossing: DELIBERATE` in the output), and NON-AMBIENT (its
own module, its own job).

=============================================================================
WHY THE INCUMBENT NUMBER NEEDED REPLACING RATHER THAN RE-QUOTING
=============================================================================
AI reported hard-bin's C1 as median `|rho| = 0.00709` against the smooth arm's
0.04987 and called the confound 7x smaller. AL2 then measured that hard-bin's
`mu_hat`, as that statistic uses it, takes **exactly 6 distinct values in every
seed** — because

    m[c, f]    = stratum_mu[cell(c), fbin(f)]
    mu_feat[f] = median over c of m[c, f]

annihilates the concept axis, leaving a function of the firing bin alone. So the
incumbent hard-bin C1 is a correlation against a SIX-LEVEL STEP FUNCTION of
firing rate, while the smooth arm's is a correlation against a genuinely
per-feature quantity. **The two numbers were never like-for-like**, which is why
AM6 withdrew the 7x comparison rather than caveating it.

TWO THINGS ARE COMPUTED HERE, AND THEY ANSWER DIFFERENT HALVES.

(2a) **THE SIX CONDITIONAL MEANS.** A rank correlation is the wrong summary for
a six-level step function: if `mu_hat` is non-monotonic across levels, per-level
structure cancels and `|rho| ~ 0` while the relationship is strong. So report
`E[r_f | level]` for each level with its feature count and within-level spread.
Six numbers, no monotonicity assumed, and it characterises the relationship
completely. **The monotonicity verdict is DERIVED with three outcomes**
(increasing / decreasing / neither), because a two-valued verdict on an ordered
comparison is not total (gotcha 24).

(2b) **C1 ON `mu_s` AT THE ARGMAX CONCEPT** — the denominator of the term that
actually wins `M = max_c (g_fc / mu_s)`, and therefore the only `mu` the
predictor is a function of. AM3 measured that the predictor's denominator takes
~40 distinct values per feature over the 64 concept cells, so this reduction has
the resolution the 6-level summary lacks.

**AND BOTH ARMS ARE RECOMPUTED THE SAME WAY.** Comparing hard-bin's 6-level
summary against a continuous smooth `mu_hat` was not a comparison of the same
object; comparing argmax-selected against continuous is closer but still not
equal. Both arms argmax-selected, in one code path, on one `r_f`, is the honest
version — and all four numbers (2 arms x 2 reductions) are emitted so the
restatement can be read off rather than reconstructed.

=============================================================================
BOTH `n_compare` VALUES, AND WHY THAT IS NOT PADDING
=============================================================================
The incumbent 0.00709 was measured at `n_compare = 19`. The arm of record is
`n_compare = 39`. Reporting only one of them would either break the
like-for-like comparison the restatement needs or quote a superseded arm, so
both run and each number carries its `n_compare`.

=============================================================================
THE SE SCALE IS AJ5's, SO THE COMPARISON IS LIKE-FOR-LIKE
=============================================================================
With 10,240 features the null SE of a Spearman `rho` is about `1/sqrt(n - 1)`
= 0.00988. AJ5 used exactly that and noted it is an UPPER BOUND on precision:
features within a seed share a dictionary and are not independent, so the true
SE is larger. Every `|rho|` here is reported in those units as well as raw.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr

from src.analysis.m4_confound import hard_bin_mu_per_feature

# AJ5's null scale for a Spearman over 10,240 features, as a function rather
# than a constant so a different feature count cannot silently inherit it.
def spearman_null_se(n: int) -> float:
    return 1.0 / np.sqrt(max(n - 1, 1))


# The incumbent numbers this restatement is measured against (AI / AJ5, at
# n_compare = 19). Named so the comparison is explicit rather than remembered.
INCUMBENT_MEDIAN_ABS_RHO = {"hard_bin": 0.00709, "smooth": 0.04987}
P6_MIN_RATIO = 2.0

# Identity keys every input must agree on before anything is correlated (I3).
# `argmax_pool_n` is included BY NAME because gotcha 16's fourth instance was
# exactly this key being read for display and not compared.
SHARED_IDENTITY_KEYS = ("library_sha", "n_concepts", "argmax_pool_n",
                        "truncation_policy", "scored_row_set")


def _attrs(path: Path) -> dict:
    with h5py.File(path, "r") as hf:
        return {k: (v.decode() if isinstance(v, bytes) else v)
                for k, v in hf.attrs.items()}


def check_shared_identity(sources: dict[str, Path],
                          keys: tuple[str, ...] = SHARED_IDENTITY_KEYS) -> dict:
    """Every source must agree on every key. AN ABSENT KEY IS REPORTED ABSENT AND
    REFUSED, never treated as agreement (amendment I3, gotcha 15)."""
    read = {name: _attrs(p) for name, p in sources.items()}
    problems, table = [], {}
    for key in keys:
        seen = {}
        for name, at in read.items():
            if key not in at:
                problems.append(f"{key}: ABSENT from {sources[name].name} "
                                f"(absent is not matching -- I3)")
                seen[name] = "<ABSENT>"
            else:
                seen[name] = at[key]
        table[key] = {n: (v.item() if hasattr(v, "item") else v)
                      for n, v in seen.items()}
        vals = {str(v) for v in seen.values() if v != "<ABSENT>"}
        if len(vals) > 1:
            problems.append(f"{key}: sources disagree -- {table[key]}")
    return {"per_key": table, "problems": problems}


def mu_at_argmax_hard(gtilde_path: Path, fire_rate: np.ndarray,
                      prov: dict | None = None) -> np.ndarray:
    """`stratum_mu` at the concept that WINS the argmax, per (seed, feature).

    The feature-side firing-bin index is reconstructed from `fire_rate` and the
    `firing_rate_edges` attr where it is not stored, exactly as
    `m4_confound.hard_bin_mu_per_feature` does -- the same helper is called for
    the incumbent reduction below so the two cannot diverge.
    """
    with h5py.File(gtilde_path, "r") as hf:
        stratum_mu = hf["stratum_mu"][:]                  # (seeds, cells, fbins)
        cell = hf["concept_cell"][:]                      # (concepts,)
        argmax = hf["argmax_concept"][:]                  # (seeds, features)
    # The firing-bin assignment comes from the SAME routine the incumbent uses,
    # so a divergence between the two reductions cannot be an artifact of two
    # different bin reconstructions. `prov` may be passed in when the caller has
    # already run that routine -- same source, no second pass.
    if prov is None:
        _, prov = hard_bin_mu_per_feature(gtilde_path, fire_rate)
    from src.analysis.null_pooling import bin_index
    edges = prov["firing_rate_edges"]
    n_bins = prov["n_firing_bins"]
    n_seeds, n_feats = argmax.shape
    out = np.empty((n_seeds, n_feats), dtype=np.float64)
    for s in range(n_seeds):
        fb = np.clip(bin_index(fire_rate[s], edges), 0, n_bins - 1)
        out[s] = stratum_mu[s][cell[argmax[s]], fb]
    return out


def mu_at_argmax_smooth(mu_smooth_path: Path,
                        gtilde_smooth_path: Path) -> np.ndarray:
    """Smooth `mu_hat` at the concept that wins the SMOOTH-normalised argmax.

    The argmax must come from the smooth arm's own gtilde: taking the hard-bin
    argmax and reading the smooth `mu` at it would be a third, meaningless
    object.
    """
    with h5py.File(gtilde_smooth_path, "r") as hf:
        argmax = hf["argmax_concept"][:]
    with h5py.File(mu_smooth_path, "r") as hf:
        mu = hf["mu_hat_full"]                            # (seeds, concepts, feats)
        n_seeds, n_concepts, n_feats = mu.shape
        if argmax.shape != (n_seeds, n_feats):
            raise SystemExit(
                f"smooth argmax shape {argmax.shape} does not match mu_hat_full "
                f"({n_seeds}, {n_feats}) -- refusing to index one arm's mu with "
                "another's argmax")
        out = np.empty((n_seeds, n_feats), dtype=np.float64)
        idx = np.arange(n_feats)
        for s in range(n_seeds):
            out[s] = mu[s][argmax[s], idx]
    return out


def _spread(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"n": 0, **{k: float("nan")
                           for k in ("mean", "sd", "p10", "median", "p90")}}
    return {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std()),
            "p10": float(np.percentile(x, 10)),
            "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90))}


def conditional_means_by_level(mu_feat: np.ndarray, r_f: np.ndarray) -> dict:
    """2a. `E[r_f | level]` for each distinct value of the six-level `mu_hat`.

    LEVELS ARE ORDERED BY THEIR `mu` VALUE, and their count is MEASURED rather
    than assumed to be six -- AL2 asserts a bound (`<= n_firing_bins`), and a
    diagnostic that hard-codes the answer it expects cannot disagree with its
    author (gotcha 20).
    """
    per_seed = []
    for s in range(mu_feat.shape[0]):
        mu, r = mu_feat[s], r_f[s]
        vals = np.unique(mu[np.isfinite(mu)])
        levels = []
        for v in vals:
            m = mu == v
            levels.append({"mu": float(v), "n_features": int(m.sum()),
                           "r_f": _spread(r[m])})
        means = [lv["r_f"]["mean"] for lv in levels if lv["n_features"] > 0]
        d = np.diff(means) if len(means) > 1 else np.array([])
        # THREE outcomes, derived. A two-valued monotonicity verdict on an
        # ordered comparison is not total (gotcha 24).
        shape = ("UNDEFINED_TOO_FEW_LEVELS" if d.size == 0 else
                 "MONOTONE_INCREASING" if np.all(d > 0) else
                 "MONOTONE_DECREASING" if np.all(d < 0) else
                 "NEITHER")
        per_seed.append({
            "seed": s, "n_distinct_mu": int(vals.size),
            "n_nonfinite_mu": int((~np.isfinite(mu)).sum()),
            "levels": levels,
            "E_r_f_by_level_ascending_mu": means,
            "monotonicity": shape,
            "spread_of_E_r_f_across_levels": (float(max(means) - min(means))
                                              if means else float("nan")),
        })
    counts = {r["monotonicity"] for r in per_seed}
    return {
        "per_seed": per_seed,
        "n_distinct_mu_by_seed": [r["n_distinct_mu"] for r in per_seed],
        "monotonicity_by_seed": [r["monotonicity"] for r in per_seed],
        "monotonicity_is_unanimous": bool(len(counts) == 1),
        "reading": (
            "E[r_f | level] with per-level counts and within-level spread. If "
            "the six means are non-monotone in mu, a rank correlation cancels "
            "per-level structure and reads ~0 while the relationship is strong "
            "-- which is AM6 decision 2(b)'s reason for reporting this instead."),
    }


def pooled_conditional_means(mu_feat: np.ndarray, r_f: np.ndarray) -> dict:
    """The same table pooled over feature-seed CELLS, grouped by the rank of the
    level within each seed.

    Grouping by RANK, not by `mu` value: the six values differ between seeds, so
    pooling on the value would create up to 120 singleton groups. Rank within
    seed is the level's identity. **This pools OBSERVATIONS, it does not reduce
    over the seed axis on a feature-indexed array** -- the distinction gotcha 18
    turns on. The per-seed table above stays primary.
    """
    ranks, vals, rs = [], [], []
    for s in range(mu_feat.shape[0]):
        mu = mu_feat[s]
        uniq = np.unique(mu[np.isfinite(mu)])
        rank = np.searchsorted(uniq, mu)
        ok = np.isfinite(mu)
        ranks.append(rank[ok]); vals.append(mu[ok]); rs.append(r_f[s][ok])
    rank = np.concatenate(ranks); val = np.concatenate(vals)
    r = np.concatenate(rs)
    out = []
    for k in np.unique(rank):
        m = rank == k
        out.append({"level_rank": int(k), "n_cells": int(m.sum()),
                    "mu_range": [float(val[m].min()), float(val[m].max())],
                    "r_f": _spread(r[m])})
    return {"by_level_rank": out,
            "grouping": "rank of the level within its own seed (values differ "
                        "between seeds); pools OBSERVATIONS, not a reduction "
                        "over the seed axis"}


def c1(mu: np.ndarray, r_f: np.ndarray) -> dict:
    """Spearman(`mu`, `r_f`) per seed, with AJ5's SE scale and rho^2."""
    rhos, n_distinct = [], []
    for s in range(mu.shape[0]):
        ok = np.isfinite(mu[s]) & np.isfinite(r_f[s])
        rhos.append(float(spearmanr(mu[s][ok], r_f[s][ok]).statistic))
        n_distinct.append(int(np.unique(mu[s][ok]).size))
    a = np.abs(np.asarray(rhos))
    se = spearman_null_se(mu.shape[1])
    # A Spearman against a CONSTANT is undefined, and `spearmanr` returns NaN.
    # It is counted and named UNINFORMATIVE rather than folded into the median,
    # because "couldn't compute" must not read as a value (gotcha 25b). The
    # summaries below are nan-aware so one undefined seed cannot turn the whole
    # median into NaN and thereby hide the other nineteen.
    n_undef = int(np.isnan(a).sum())
    med = float(np.nanmedian(a)) if n_undef < a.size else float("nan")
    return {
        "rho_by_seed": rhos,
        "n_distinct_mu_by_seed": n_distinct,
        "n_seeds_with_undefined_rho": n_undef,
        "status": ("OK" if n_undef == 0 else
                   "UNINFORMATIVE_CONSTANT_NORMALISER_IN_EVERY_SEED"
                   if n_undef == a.size else
                   f"PARTIAL -- {n_undef} of {a.size} seeds have a constant "
                   f"normaliser and are excluded from the summaries"),
        "median_abs_rho": med,
        "min_abs_rho": float(np.nanmin(a)) if n_undef < a.size else float("nan"),
        "max_abs_rho": float(np.nanmax(a)) if n_undef < a.size else float("nan"),
        "abs_rho_range": ([float(np.nanmin(a)), float(np.nanmax(a))]
                          if n_undef < a.size else [float("nan")] * 2),
        "null_se": se,
        "median_abs_rho_in_se": float(med / se),
        "max_abs_rho_in_se": (float(np.nanmax(a) / se) if n_undef < a.size
                              else float("nan")),
        "worst_seed": (int(np.nanargmax(a)) if n_undef < a.size else None),
        "rho_squared_median": (float(np.nanmedian(a ** 2)) if n_undef < a.size
                               else float("nan")),
        "rho_squared_max": (float(np.nanmax(a ** 2)) if n_undef < a.size
                            else float("nan")),
        "variance_fraction_note": (
            "rho^2 on RANKS, so it is the share of variance in the rank of r_f "
            "explained by the rank of the normaliser -- not an R^2 of the "
            "regression, which does not exist yet."),
    }



def score_as_predictions(incumbent_path: Path, arms: dict) -> dict:
    """AMENDMENT AS section 1 -- A1, E1 and E2, scored against the pre-AS run.

    THE LIKE-FOR-LIKE ARM is `n_compare_19` on the SIX-LEVEL hard-bin
    reduction, because that is where the incumbent 0.00709 = 0.72 SE was
    measured (AI / AJ5). The argmax-selected numbers are reported beside it.

    A1 APPLIES TO THE HARD-BIN ARM ONLY, and the reason is not a caveat but an
    input difference: rows 0 and 1 are the only positions where the
    lexicographic and numeric orders agree, so those two seeds were ALREADY
    correctly joined and must come back bit-identical -- but only where the
    inputs are unchanged. The SMOOTH arm is rebuilt in the same job (its
    covariates were mis-paired too), so its seed-0/1 entries SHOULD move and a
    bit-identity criterion there would be a check that can only fail
    (gotcha 25a).
    """
    doc = json.loads(Path(incumbent_path).read_text())
    key = "n_compare_19"
    red = "hard_bin_six_level_INCUMBENT_REDUCTION"
    if key not in doc.get("by_rf_arm", {}) or key not in arms:
        return {"status": f"NOT_SCORED -- no {key} arm in both documents"}
    old = doc["by_rf_arm"][key]["AO7_2b_C1"][red]
    new = arms[key]["AO7_2b_C1"][red]
    o = np.abs(np.asarray(old["rho_by_seed"], float))
    n = np.abs(np.asarray(new["rho_by_seed"], float))
    if o.size != n.size:
        return {"status": f"NOT_SCORED -- {o.size} vs {n.size} seeds"}

    # ---- A1: seeds 0 and 1 were already correct, so they must not move -----
    exact = [float(old["rho_by_seed"][i]) == float(new["rho_by_seed"][i])
             for i in (0, 1)]
    a1 = {
        "kind": "correctness assertion",
        "claim": ("rows 0 and 1 are the only positions where the two seed "
                  "orders agree, so those seeds were ALREADY correctly joined "
                  "and their hard-bin entries must be BIT-IDENTICAL"),
        "incumbent": [float(old["rho_by_seed"][i]) for i in (0, 1)],
        "aligned": [float(new["rho_by_seed"][i]) for i in (0, 1)],
        "abs_delta": [abs(float(old["rho_by_seed"][i])
                          - float(new["rho_by_seed"][i])) for i in (0, 1)],
        "scope": ("hard-bin arm only -- the smooth arm's covariates are "
                  "rebuilt in the same job, so a bit-identity criterion there "
                  "could only fail"),
        "outcome": "PASS" if all(exact) else "FAIL",
        "reading_if_FAIL": ("the resolver changed a join that was already "
                            "correct, so the RE-RUN is wrong rather than the "
                            "incumbent"),
    }

    # ---- E1: the 18 mis-joined seeds rise, and the median clears 0.72 SE ---
    mis = list(range(2, o.size))
    rose = [int(i) for i in mis if n[i] > o[i]]
    se = spearman_null_se(new.get("n_features", 10240))
    med_new = float(np.nanmedian(n))
    med_old = float(np.nanmedian(o))
    med_rises = med_new > med_old
    clears = (med_new / se) > INCUMBENT_MEDIAN_ABS_RHO["hard_bin"] / se
    e1 = {
        "category": "procedure-behaviour (the 1/3 bin)",
        "why_this_bin": (
            "both sides are ours, so gotcha 26's test says within-system -- "
            "but the CONTENT is a claim about what a mis-join does to a "
            "correlation statistic, which is a story about how a PROCEDURE "
            "behaves. When the two filings disagree the bin with the worse "
            "record is the honest one."),
        "n_previously_misjoined": len(mis),
        "n_that_rose": len(rose),
        "seeds_that_rose": rose,
        "median_abs_rho_incumbent": med_old,
        "median_abs_rho_aligned": med_new,
        "median_rises": bool(med_rises),
        "median_abs_rho_in_se_aligned": med_new / se,
        "incumbent_median_in_se": INCUMBENT_MEDIAN_ABS_RHO["hard_bin"] / se,
        "clears_its_own_0.72_SE": bool(clears),
        "outcome": "HIT" if (med_rises and clears) else "MISS",
        "note": ("both halves reported separately, as the prediction "
                 "requires; HIT needs both"),
    }

    # ---- E2: seeds 0 and 1 no longer both extreme --------------------------
    order = np.argsort(n)
    extreme = {int(order[0]), int(order[-1])}
    e2 = {
        "category": "distributional shape (the 13/14 bin)",
        "rank_order_of_seeds_by_abs_rho": order.tolist(),
        "lowest_seed": int(order[0]), "highest_seed": int(order[-1]),
        "abs_rho_aligned": n.tolist(),
        "abs_rho_incumbent": o.tolist(),
        "incumbent_extremes": [int(np.argsort(o)[0]), int(np.argsort(o)[-1])],
        "outcome": "MISS" if extreme == {0, 1} else "HIT",
        "what_a_HIT_DOES_NOT_LICENSE": (
            "AJ5 would close as EXPLAINED BY A DEFECT IN THE C1 COMPUTATION. "
            "It says nothing about seeds 0 and 1's training runs, and the "
            "~90-155 low-firing features with mean r_f 0.93-0.96 are a fact "
            "about r_f, not about C1. The mechanism is NOT stated: n = 2 was "
            "never a mechanism, and a coincidence that resolves is still a "
            "coincidence that resolved."),
    }
    return {"incumbent_artifact": str(incumbent_path),
            "like_for_like_arm": f"{key} / {red}",
            "A1": a1, "E1": e1, "E2": e2}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gtilde", type=Path, required=True,
                    help="hard-bin gtilde: stratum_mu, concept_cell, "
                         "argmax_concept")
    ap.add_argument("--gtilde-smooth", type=Path, required=True,
                    help="smooth-normalised gtilde, for ITS argmax")
    ap.add_argument("--mu-smooth", type=Path, required=True,
                    help="smooth_mu h5 with mu_hat_full")
    ap.add_argument("--null-reductions", type=Path, required=True,
                    help="for fire_rate")
    ap.add_argument("--rf-counts", type=Path, nargs="+", required=True,
                    help="one or more rf_counts h5; each is reported with its "
                         "own n_compare_seeds")
    ap.add_argument("--incumbent", type=Path, default=None,
                    help="amendment AS: the PRE-ALIGNMENT c1_argmax json, "
                         "against which A1 / E1 / E2 are scored. Read only; "
                         "never overwritten.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from src.sae.train import git_dirty, git_sha

    print("*** X7 FIREWALL CROSSING -- DELIBERATE, SCOPED (AM6 decision 2b, "
          "amendment AO7) ***", flush=True)

    ident = check_shared_identity({
        "gtilde": args.gtilde, "gtilde_smooth": args.gtilde_smooth,
        "mu_smooth": args.mu_smooth})
    if ident["problems"]:
        raise SystemExit(
            "PRE-FLIGHT FAILED on the normaliser identity -- refusing to "
            "compute C1.\n  * " + "\n  * ".join(ident["problems"]))
    print("[c1a] identity OK on " + ", ".join(SHARED_IDENTITY_KEYS), flush=True)

    # ---- SEED ALIGNMENT (amendment AS; the AR9 blast radius) --------------
    # THIS JOB WAS THE DEFECT. `gtilde` / `mu_hat_full` / `fire_rate` are in
    # LEXICOGRAPHIC seed order and `rf_counts` is NUMERIC; only rows 0 and 1
    # coincide, so `c1(mu[s], r_f[s])` correlated 18 of 20 seeds' normalisers
    # against a DIFFERENT seed's `r_f`. Cross-seed pairings are approximately
    # null by gotcha 18, so the incumbent median |rho| was diluted toward zero
    # -- and AL8b demoted the feature-side arm citing exactly that number.
    #
    # ORDER OF OPERATIONS MATTERS. `mu_at_argmax_*` index `stratum_mu[s]` and
    # `argmax[s]` positionally against `fire_rate[s]`, so the mu arrays must be
    # BUILT in the gtilde's own order with `fire_rate` aligned TO it, and only
    # the RESULT aligned onward to the anchor order. Aligning `fire_rate` to
    # the anchors first would simply move the defect one step earlier.
    from src.analysis.seed_axis import (align_to_reference, seed_order,
                                        stored_seed_labels)

    with h5py.File(args.null_reductions, "r") as hf:
        fire_rate_null = hf["fire_rate"][:]
    null_labels = stored_seed_labels(args.null_reductions)
    if null_labels is None:
        raise SystemExit("FATAL: the null reductions carry no seed labels; "
                         "refusing to assume an order (gotcha 15).")
    gt = seed_order(args.gtilde, args.null_reductions)
    gts = seed_order(args.gtilde_smooth, args.null_reductions)
    print(f"[c1a] seed order: null {null_labels}\n"
          f"[c1a]             gtilde {gt['labels']} ({gt['source']})\n"
          f"[c1a]             smooth {gts['labels']} ({gts['source']})",
          flush=True)

    fire_gt = align_to_reference(null_labels, fire_rate_null, gt["labels"])
    mu_hard_feat, prov = hard_bin_mu_per_feature(args.gtilde, fire_gt)
    mu_hard_argmax = mu_at_argmax_hard(args.gtilde, fire_gt, prov=prov)
    mu_smooth_argmax = mu_at_argmax_smooth(args.mu_smooth, args.gtilde_smooth)
    with h5py.File(args.mu_smooth, "r") as hf:
        mu_smooth_feat = np.stack([np.median(hf["mu_hat_full"][s], axis=0)
                                   for s in range(hf["mu_hat_full"].shape[0])])
    # `mu_hat_full` shares the smooth gtilde's order: `smooth_mu` writes
    # `dset[s]` over the same lexicographic null axis `standardize_gain` reads.
    mu_smooth_labels = stored_seed_labels(args.mu_smooth) or gts["labels"]
    print(f"[c1a] mu built: hard 6-level {mu_hard_feat.shape}, hard argmax "
          f"{mu_hard_argmax.shape}, smooth argmax {mu_smooth_argmax.shape}",
          flush=True)

    arms: dict = {}
    for path in args.rf_counts:
        with h5py.File(path, "r") as hf:
            counts = hf["match_count"][:]
            n_compare = int(hf.attrs["n_compare_seeds"])
            theta = float(hf.attrs["theta"])
            sim = str(hf.attrs.get("similarity"))
        if sim != "decoder_cosine":
            raise SystemExit(f"FATAL: {path.name} was not built on "
                             "decoder_cosine (amendment AB decision B)")
        anchor_labels = stored_seed_labels(path)
        if anchor_labels is None:
            raise SystemExit(f"FATAL: {path.name} carries no seed labels.")
        r_f = counts.astype(np.float64) / n_compare
        # every normaliser onto the ANCHOR order, so `c1(mu[s], r_f[s])` pairs
        # a seed's normaliser with ITS OWN reproducibility.
        mu_hard_feat_a = align_to_reference(gt["labels"], mu_hard_feat,
                                            anchor_labels)
        mu_hard_argmax_a = align_to_reference(gt["labels"], mu_hard_argmax,
                                              anchor_labels)
        mu_smooth_argmax_a = align_to_reference(gts["labels"],
                                                mu_smooth_argmax, anchor_labels)
        mu_smooth_feat_a = align_to_reference(mu_smooth_labels, mu_smooth_feat,
                                              anchor_labels)
        key = f"n_compare_{n_compare}"

        c1_hard_6lvl = c1(mu_hard_feat_a, r_f)
        c1_hard_argmax = c1(mu_hard_argmax_a, r_f)
        c1_smooth_median = c1(mu_smooth_feat_a, r_f)
        c1_smooth_argmax = c1(mu_smooth_argmax_a, r_f)

        arms[key] = {
            "rf_counts": str(path), "n_compare_seeds": n_compare,
            "theta": theta, "n_anchors": int(r_f.shape[0]),
            "seed_alignment": {
                "anchor_order": anchor_labels,
                "gtilde_order": gt["labels"],
                "gtilde_smooth_order": gts["labels"],
                "null_order": null_labels,
                "n_rows_mispaired_by_a_positional_join": int(sum(
                    a != b for a, b in zip(anchor_labels, gt["labels"]))),
            },
            "AO7_2a_conditional_means_per_seed": conditional_means_by_level(
                mu_hard_feat_a, r_f),
            "AO7_2a_conditional_means_pooled": pooled_conditional_means(
                mu_hard_feat_a, r_f),
            "AO7_2b_C1": {
                "hard_bin_argmax_selected": c1_hard_argmax,
                "smooth_argmax_selected": c1_smooth_argmax,
                "hard_bin_six_level_INCUMBENT_REDUCTION": c1_hard_6lvl,
                "smooth_median_over_concepts_INCUMBENT_REDUCTION":
                    c1_smooth_median,
            },
            # The ratio the withdrawn 7x claim was, recomputed on BOTH
            # reductions. DERIVED, and both are emitted so the restatement is
            # read off rather than reconstructed (gotcha 21: the reduction is
            # part of the claim).
            "smooth_over_hard_bin_ratio_of_median_abs_rho": {
                "argmax_selected_BOTH_ARMS": float(
                    c1_smooth_argmax["median_abs_rho"]
                    / max(c1_hard_argmax["median_abs_rho"], 1e-300)),
                "incumbent_reductions_the_withdrawn_claim_used": float(
                    c1_smooth_median["median_abs_rho"]
                    / max(c1_hard_6lvl["median_abs_rho"], 1e-300)),
                "note": ("the withdrawn 7x compared a SIX-LEVEL hard-bin "
                         "summary against a per-feature smooth one. The "
                         "argmax-selected row is the like-for-like "
                         "replacement (AL2, AM6 decision 2b)."),
            },
        }
        cm = arms[key]["AO7_2a_conditional_means_per_seed"]
        print(f"[c1a] {key}: distinct mu levels "
              f"{sorted(set(cm['n_distinct_mu_by_seed']))}, monotonicity "
              f"{sorted(set(cm['monotonicity_by_seed']))}", flush=True)
        print(f"[c1a] {key}: median |rho| hard-bin argmax "
              f"{c1_hard_argmax['median_abs_rho']:.5f} "
              f"({c1_hard_argmax['median_abs_rho_in_se']:.2f} SE) | "
              f"hard-bin 6-level {c1_hard_6lvl['median_abs_rho']:.5f} | "
              f"smooth argmax {c1_smooth_argmax['median_abs_rho']:.5f} | "
              f"smooth median {c1_smooth_median['median_abs_rho']:.5f}",
              flush=True)

    # ---- the scorecard, on the arm the incumbent was measured on -----------
    like_for_like = arms.get("n_compare_19")
    if like_for_like is None:
        p6 = {"outcome": "NOT_SCORED -- no n_compare = 19 arm was supplied, and "
                         "the incumbent 0.00709 was measured there"}
        p7 = p6
    else:
        got = like_for_like["AO7_2b_C1"]["hard_bin_argmax_selected"][
            "median_abs_rho"]
        base = INCUMBENT_MEDIAN_ABS_RHO["hard_bin"]
        p6 = {
            "prediction": f"hard-bin median |rho| on argmax-selected mu_s "
                          f"exceeds {base} by more than {P6_MIN_RATIO}x",
            "category": "within-system comparison (AO4 P6)",
            "incumbent_six_level": base,
            "recomputed_six_level": like_for_like["AO7_2b_C1"][
                "hard_bin_six_level_INCUMBENT_REDUCTION"]["median_abs_rho"],
            "argmax_selected": got,
            "ratio": float(got / base),
            "outcome": "HIT" if got > P6_MIN_RATIO * base else "MISS",
        }
        shapes = set(like_for_like["AO7_2a_conditional_means_per_seed"]
                     ["monotonicity_by_seed"])
        p7 = {
            "prediction": "E[r_f | level] is NOT monotone across the six levels",
            "category": "distributional shape (AO4 P7)",
            "monotonicity_by_seed": sorted(shapes),
            "outcome": "HIT" if shapes == {"NEITHER"} else "MISS",
            "reading": ("if this MISSES, the Spearman was not the wrong summary "
                        "for the reason AM6 decision 2(b) gave, which weakens "
                        "2b's rationale without changing its result"),
        }

    out = {
        "identity": {
            "measurement": "c1_argmax_AO7",
            "amendment": "AO section 7; the author decision AM6.2(b)",
            "firewall_crossing": "DELIBERATE",
            "firewall_scope": (
                "Correlations and conditional means between already-computed "
                "vectors. No model is fitted and no regression is run, so "
                "nothing here previews the primary coefficient. Nothing about "
                "theta, the endpoint cuts or the primary regression may be "
                "revised from this output."),
            "gtilde": str(args.gtilde),
            "gtilde_smooth": str(args.gtilde_smooth),
            "mu_smooth": str(args.mu_smooth),
            "null_reductions": str(args.null_reductions),
            "shared_identity": ident["per_key"],
            "hard_bin_firing_bin_provenance": prov,
            "seed_order_gtilde": gt,
            "seed_order_gtilde_smooth": gts,
            "seed_alignment_note": (
                "AMENDMENT AS. gtilde / mu_hat_full / fire_rate are in "
                "LEXICOGRAPHIC seed order and rf_counts is NUMERIC; only "
                "rows 0 and 1 coincide. Every normaliser is aligned to the "
                "ANCHOR order before correlating, so each seed's normaliser "
                "meets its OWN r_f. The pre-AS numbers correlated 18 of 20 "
                "seeds across the split."),
            "git_sha": git_sha(), "git_dirty": git_dirty(),
            "python_version": platform.python_version(),
        },
        "by_rf_arm": arms,
        "prediction_scorecard": {
            "P6": p6, "P7": p7,
            "AS_section_1": (score_as_predictions(args.incumbent, arms)
                             if args.incumbent is not None else
                             {"status": "NOT_SCORED -- no --incumbent given"}),
        },
        "decision": (
            "REPORT ONLY. Whether the 7x comparison is restated with its new "
            "value or withdrawn permanently follows from these numbers; the "
            "disposal is recorded in the amendment, not decided here."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[c1a] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
