"""THE `w0` RESOLUTION TEST. PREREGISTRATION amendment AO, sections 1-3;
The author's ratified decision AM6.1.

=============================================================================
THE QUESTION, AND WHY IT IS NOT ANSWERABLE BY LOOKING AT THE INTERVAL
=============================================================================
At `n_compare = 39` the profiled interval for `w0` -- the mixing distribution's
mass at exactly `p = 0` -- is `[0.025, 0.050]`. It excludes zero by EXACTLY ONE
GRID STEP, and two facts coincide almost exactly at this N:

  * the `profile_w0` grid is `linspace(0, 1, 41)`, so its spacing is 0.025;
  * `w0`'s identifiability floor scales like `1/n_compare = 1/39 = 0.0256`.

So "the data require a point mass at `p = 0`" and "0.025 is the smallest value
either the grid or 39 trials can separate from zero" predict the SAME reported
interval. AJ2's third argument for design 3 turns on which is true, which is why
that argument is SUSPENDED rather than retracted (AM6 decision 1).

Three measurements, in increasing strength:

  1a. THE CURVE, not the threshold crossing. Report the profiled log-likelihood
      near zero, including at exactly `w0 = 0`, against its value at the MLE.
      This is a diff, not a verdict, and it needs no new fit -- `profile_w0` has
      always evaluated its grid from 0.0 upward, so the number is already in the
      baselined artifact. Reported as a MEASUREMENT; amendment AO0 refuses to
      file a prediction about a number that was read before any prediction
      existed.

  1b. THE FINER GRID. Refit the profile on a grid whose spacing is well below
      0.025 and dense near zero. If the lower bound is a grid artifact it
      collapses toward zero; if it is the likelihood's own shape it moves only to
      wherever the 2-unit contour actually lies. PARTIALLY ENTAILED BY 1a and
      labelled so (AO3 P1): 1b cannot independently test 1a.

  1c. THE DECISIVE TEST -- gotcha 23 applied to a mixture weight. Simulate from
      a truth with `w0 = 0` EXACTLY, at the production `n_compare`, the
      production number of observations, the production NPMLE and the production
      profile grid; refit each replicate; and ask where the observed 0.025 falls
      in the resulting null. If the estimator spuriously returns ~0.025 when the
      truth is zero, the exclusion is an artifact of the procedure.

=============================================================================
WHAT IS PRE-DECLARED, AND WHY IT HAD TO BE (amendment AO2)
=============================================================================
Under a DISCRETE null the phrase "the percentile at which 0.025 falls" is
ambiguous in a way that DECIDES THE OUTCOME: the null's support is the profile
grid, so it has an atom exactly at 0.025, and `P(null < 0.025)` versus
`P(null <= 0.025)` can straddle the 95th percentile in opposite directions.

COMMITTED IN AO2, BEFORE THIS FILE EXISTED:

    percentile := 100 * P(null_lower_bound < 0.025)      [strictly below]

equivalently the exceedance p-value `P(null >= 0.025)`. The reasoning is that a
replicate returning EXACTLY 0.025 is a replicate in which the procedure did the
thing we are worried about, so it is evidence FOR unsurprisingness; counting it
as "below the observed value" would let the atom push the statistic toward
declaring a point mass, which inverts the test. The other convention is computed
and reported, labelled as not the one the rule reads.

AND THE STATISTIC IS THE SAME REDUCTION ON BOTH SIDES (gotcha 21). The observed
0.025 is a PROFILED INTERVAL LOWER BOUND, so the null is of the profiled lower
bound. The point-estimate `w0` distribution is reported as a separate row;
comparing a profiled bound against a distribution of point estimates would be
exactly the reduction mismatch gotcha 21 is about.

=============================================================================
THE SAMPLING STRUCTURE MATCHES THE PRODUCTION FIT EXACTLY
=============================================================================
The production fit is ONE POOLED NPMLE over all 204,800 feature-anchor
observations (`analyse_theta` calls `fit_mixture` on the pooled atom counts;
per-seed atoms exist in the M1 document but feed only the error bars). The
simulation therefore draws 204,800 observations and does ONE pooled refit per
replicate -- stated in the output rather than left for a reader to infer.

Anchors are independent draws because A FEATURE INDEX IS NOT A CROSS-SEED
IDENTITY (gotcha 18): feature `f` in seed 0 and feature `f` in seed 1 are
unrelated latents and do not share a `p`. That is what makes 204,800 the right
observation count rather than 10,240.

MULTINOMIAL RATHER THAN TWO-STAGE, AND IT IS EXACT, NOT AN APPROXIMATION. Each
observation draws `p ~ w'` then `k ~ Binomial(n_compare, p)`, so the `k` are iid
with pmf `pk = sum_j w'_j Binom(k; n, p_j)` and the count vector is EXACTLY
`Multinomial(n_obs, pk)`. `tests/test_w0_resolution.py` checks the two routes
agree rather than taking the algebra on trust.

=============================================================================
DETERMINISM ACROSS WORKER LAYOUT
=============================================================================
Replicate `r` seeds its own generator from `(base_seed, r)`, so the result does
not depend on how replicates are distributed over workers or on how many there
are. A parallel simulation whose answer depends on `--workers` is not
reproducible, and the test asserts the arrays are identical at 1 worker and at
3 -- this project's whole subject is reproducibility.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy.stats import binom

from src.analysis.rf_mixture import (
    EM_W_TOL,
    P_GRID,
    _ag_cuts_or_absent,
    fit_mixture,
    profile_w0,
    purity,
)

# `_ag_cuts_or_absent` is private to `rf_mixture` and imported deliberately
# rather than reimplemented: it already encodes the rule that a missing
# pre-registered cut is REPORTED ABSENT and never substituted (gotcha 15), and a
# second copy of that rule is a second place for it to drift.

# The production profile grid, restated as a named constant so the null
# simulation cannot silently drift from the grid the observed 0.025 came off.
PRODUCTION_PROFILE_POINTS = 41
PRODUCTION_PROFILE_SPACING = 1.0 / (PRODUCTION_PROFILE_POINTS - 1)   # 0.025

# The observed lower bound the rule is scored against (AL6, N = 39, theta 0.70).
# Read from the artifact at run time; this is only the expected value, asserted
# so a changed artifact cannot silently rescore the rule against a new number.
EXPECTED_OBSERVED_LOWER_BOUND = 0.025

# AO3 P2's threshold, and AO2's branch point. Named, not inline.
P2_MIN_LOWER_BOUND = 0.010
RULE_PERCENTILE_CUT = 95.0


def fine_w0_grid(dense_to: float = 0.15, dense_step: float = 0.001,
                 n_tail: int = 22) -> np.ndarray:
    """A grid dense near zero, coarse in the tail (AO section 1b).

    Dense where the question is (`[0, dense_to]` at `dense_step`, i.e. 25x finer
    than the production 0.025) and coarse above it, because the profile falls off
    by hundreds of loglik units past 0.1 and resolution there buys nothing. The
    spacing AT EACH REPORTED BOUND is emitted with the bound, so a bound sitting
    in the coarse region cannot be quoted as if it had 0.001 resolution.
    """
    dense = np.arange(0.0, dense_to + dense_step / 2, dense_step)
    tail = np.linspace(dense_to + dense_step, 1.0, n_tail)
    return np.unique(np.concatenate([dense, tail]))


def local_spacing(grid: np.ndarray, value: float) -> float:
    """Grid spacing at `value` -- the resolution with which a bound is known."""
    g = np.asarray(grid, dtype=float)
    i = int(np.argmin(np.abs(g - value)))
    lo = g[i] - g[i - 1] if i > 0 else np.inf
    hi = g[i + 1] - g[i] if i + 1 < g.size else np.inf
    return float(min(lo, hi))


def curve_near_zero(profile: dict, n_report: int = 6) -> dict:
    """1a. The profiled loglik near zero, as DROPS from the profile maximum.

    Takes an existing `profile_w0` result -- no refit. The value at exactly
    `w0 = 0` is `profile_loglik[0]` whenever the grid starts at 0, which is
    ASSERTED rather than assumed: if a future grid did not include exact zero,
    silently reporting its first point as "the loglik at zero" is precisely the
    absent-key-reads-as-agreement failure of gotcha 15.
    """
    g = np.asarray(profile["w0_grid"], dtype=float)
    ll = np.asarray(profile["profile_loglik"], dtype=float)
    if g[0] != 0.0:
        raise SystemExit(
            f"the profile grid does not start at exactly 0 (starts at {g[0]!r}), "
            "so there is no log-likelihood AT w0 = 0 to report. Refusing to "
            "substitute the nearest point.")
    best = float(np.nanmax(ll))
    drop = best - ll
    return {
        "grid_spacing": float(g[1] - g[0]) if g.size > 1 else float("nan"),
        "w0_at_max": float(g[int(np.nanargmax(ll))]),
        "loglik_at_max": best,
        "loglik_at_exactly_zero": float(ll[0]),
        "drop_at_exactly_zero": float(drop[0]),
        "drop_units_defining_interval": float(profile["drop_units"]),
        # DERIVED (gotcha 20): how many times the interval's own cut the drop at
        # zero is, and the equivalent 1-df deviance. Never a sentence.
        "drop_at_zero_in_units_of_the_cut": float(
            drop[0] / max(profile["drop_units"], 1e-300)),
        "deviance_1df_at_zero": float(2.0 * drop[0]),
        "reported_interval": [float(x) for x in profile["w0_interval"]],
        "first_points": [
            {"w0": float(g[i]), "loglik": float(ll[i]), "drop": float(drop[i])}
            for i in range(min(n_report, g.size))],
    }


def interval_from_profile(grid: np.ndarray, loglik: np.ndarray,
                          drop: float = 2.0) -> dict:
    """The `drop`-unit interval on an arbitrary grid, with the local spacing at
    each bound reported beside it."""
    g = np.asarray(grid, dtype=float)
    ll = np.asarray(loglik, dtype=float)
    best = int(np.nanargmax(ll))
    ok = np.flatnonzero(ll >= ll[best] - drop)
    lo, hi = float(g[ok.min()]), float(g[ok.max()])
    return {
        "w0_at_max": float(g[best]),
        "loglik_at_max": float(ll[best]),
        "drop_units": float(drop),
        "w0_interval": [lo, hi],
        "w0_interval_width": hi - lo,
        "spacing_at_lower_bound": local_spacing(g, lo),
        "spacing_at_upper_bound": local_spacing(g, hi),
        "n_grid_points": int(g.size),
        "excludes_zero": bool(lo > 0.0),
    }


def null_truth(weights: np.ndarray, p_grid: np.ndarray) -> dict:
    """The `w0 = 0` truth: drop the mass at `p = 0` and renormalise the rest.

    Returns the truth weights and the implied atom pmf. `w0 == 0` EXACTLY is
    asserted on the way out, because the whole simulation is meaningless if the
    truth it samples from retains any endpoint mass.
    """
    w = np.asarray(weights, dtype=float).copy()
    p = np.asarray(p_grid, dtype=float)
    if p[0] != 0.0:
        raise SystemExit("p_grid[0] is not exactly 0, so there is no w0 to zero")
    removed = float(w[0])
    w[0] = 0.0
    rest = w.sum()
    if rest <= 0:
        raise SystemExit("all mixing mass sits at p = 0; no w0 = 0 truth exists")
    w /= rest
    if w[0] != 0.0:
        raise AssertionError("truth still carries mass at p = 0")
    return {"weights": w, "p_grid": p, "removed_w0": removed,
            "renormalised_from": rest}


def atom_pmf(weights: np.ndarray, p_grid: np.ndarray, n_trials: int) -> np.ndarray:
    k = np.arange(n_trials + 1)
    lik = binom.pmf(k[None, :], n_trials, np.asarray(p_grid)[:, None])
    pk = (np.asarray(weights)[:, None] * lik).sum(axis=0)
    return pk / pk.sum()


def draw_counts(pk: np.ndarray, n_obs: int, base_seed: int, r: int) -> np.ndarray:
    """Replicate `r`'s atom counts. Seeded from `(base_seed, r)` so the draw is a
    function of the replicate index alone -- not of worker count or scheduling."""
    rng = np.random.default_rng([base_seed, r])
    return rng.multinomial(n_obs, pk)


def draw_counts_two_stage(weights: np.ndarray, p_grid: np.ndarray,
                          n_trials: int, n_obs: int,
                          base_seed: int, r: int) -> np.ndarray:
    """The EXPLICIT two-stage sampler: draw `p` per observation, then `k`.

    Not used in production -- `draw_counts` is exactly equivalent and far
    cheaper. It exists so the equivalence is CHECKED rather than argued
    (AN8: converting an assertion into a measurement beats justifying it).
    """
    rng = np.random.default_rng([base_seed, r])
    p = np.asarray(p_grid, dtype=float)
    w = np.asarray(weights, dtype=float)
    idx = rng.choice(p.size, size=n_obs, p=w / w.sum())
    k = rng.binomial(n_trials, p[idx])
    return np.bincount(k, minlength=n_trials + 1)


def em_at_fixed_w0(atom_counts: np.ndarray, n_trials: int, p_grid: np.ndarray,
                   w0: float, max_iter: int = 20_000,
                   w_tol: float = EM_W_TOL) -> dict:
    """Constrained EM: the mass at `p = 0` held at `w0`, the rest renormalised.

    This is `profile_w0`'s inner loop, exposed so the WEIGHT VECTOR at a given
    `w0` can be recovered -- `profile_w0` returns only the loglik, and the purity
    band needs the weights (its loose endpoint is NOT linear in `w0`, so it
    cannot be rescaled from a point estimate the way the strict one can, AJ1).

    IT IS A COPY, DELIBERATELY, AND THE COPY IS CHECKED. Refactoring
    `rf_mixture.profile_w0` to call this would put the baselined mixture
    artifacts' numerics at risk for no gain, so instead
    `tests/test_w0_resolution.py` asserts that this reproduces
    `profile_w0`'s `profile_loglik` at EVERY grid point exactly. A duplicated
    implementation whose agreement is measured beats a refactor whose safety is
    argued.
    """
    k = np.arange(n_trials + 1)
    lik = binom.pmf(k[None, :], n_trials, np.asarray(p_grid)[:, None])
    n = np.asarray(atom_counts, dtype=np.float64)
    ntot = n.sum()
    w = np.full(np.asarray(p_grid).size, (1.0 - w0) / max(np.asarray(p_grid).size - 1, 1))
    w[0] = w0
    ll = float("nan")
    it = -1
    for it in range(max_iter):
        joint = w[:, None] * lik
        msum = joint.sum(axis=0)
        marg = np.where(msum > 0, msum, 1e-300)
        resp = joint / marg[None, :]
        w_new = (resp * n[None, :]).sum(axis=1) / ntot
        rest = w_new[1:].sum()
        w_c = np.empty_like(w_new)
        w_c[0] = w0
        if rest > 0:
            w_c[1:] = w_new[1:] * ((1.0 - w0) / rest)
        else:
            w_c[1:] = 0.0
        ll = float((n * np.log(marg)).sum())
        done = np.abs(w_c - w).max() < w_tol
        w = w_c
        if done:
            break
    return {"w0": float(w0), "weights": w, "loglik": ll,
            "iterations": int(it + 1),
            "p_grid": np.asarray(p_grid, dtype=float)}


def purity_at_w0(atom_counts: np.ndarray, n_trials: int, p_grid: np.ndarray,
                 w0: float, group_ks: list[int], max_iter: int = 20_000) -> dict:
    """The purity numbers at a FIXED `w0`, under both genuine definitions.

    `strict` counts only the grid point `p == 0` as genuine; `loose` counts
    `p <= 0.10`. AJ1 established that the strict endpoint is exactly
    `w0 / P(k in group)` because `P(k in group | p = 0) = 1` for any group
    containing `k = 0` -- so it is LINEAR in `w0` and could be rescaled by hand.
    **The loose endpoint is not**, which is why the weights are refitted here
    rather than the band being scaled.

    The linear identity is RE-DERIVED and compared against the grid computation,
    because a formula quoted without checking is how this project has already
    produced two wrong intermediates (gotcha 20's arithmetic practice).
    """
    fit = em_at_fixed_w0(atom_counts, n_trials, p_grid, w0, max_iter=max_iter)
    w = fit["weights"]
    p = fit["p_grid"]
    if 0 not in group_ks:
        raise SystemExit(
            f"group {group_ks} does not contain k = 0, so P(k in group | p = 0) "
            "is not 1 and the strict-endpoint identity does not hold. Refusing "
            "to report a band whose construction assumes it.")
    strict = purity(fit, n_trials, group_ks, p <= 0.0)
    loose = purity(fit, n_trials, group_ks, p <= 0.10)
    p_group = float(strict["group_size"])
    identity = w0 / p_group if p_group > 0 else float("nan")
    return {
        "w0": float(w0),
        "group_ks": list(group_ks),
        "P_k_in_group": p_group,
        "strict_p_eq_0": strict["purity"],
        "loose_p_le_0.10": loose["purity"],
        "strict_via_the_linear_identity_w0_over_P": identity,
        "identity_agrees": bool(abs(identity - strict["purity"]) < 1e-9),
        "loose_minus_strict": loose["purity"] - strict["purity"],
        "loglik": fit["loglik"],
    }


_W: dict = {}


def _init_worker(pk, p_grid, n_trials, n_obs, base_seed, n_points, max_iter):
    _W.update(pk=pk, p_grid=p_grid, n_trials=n_trials, n_obs=n_obs,
              base_seed=base_seed, n_points=n_points, max_iter=max_iter)


def _one_replicate(r: int) -> dict:
    counts = draw_counts(_W["pk"], _W["n_obs"], _W["base_seed"], r)
    fit = fit_mixture(counts, _W["n_trials"], p_grid=_W["p_grid"])
    prof = profile_w0(counts, _W["n_trials"], p_grid=_W["p_grid"],
                      n_points=_W["n_points"], max_iter=_W["max_iter"])
    return {
        "r": int(r),
        "point_w0": float(fit["mass_at_p_eq_0"]),
        "profiled_lower": float(prof["w0_interval"][0]),
        "profiled_upper": float(prof["w0_interval"][1]),
        "profiled_at_max": float(prof["w0_at_max"]),
        "atom_frac_k0": float(counts[0] / counts.sum()),
        "fit_converged": bool(fit["converged"]),
    }


def simulate_null(pk: np.ndarray, p_grid: np.ndarray, n_trials: int,
                  n_obs: int, n_rep: int, base_seed: int = 0,
                  workers: int = 1, n_points: int = PRODUCTION_PROFILE_POINTS,
                  max_iter: int = 20_000) -> list[dict]:
    """`n_rep` replicates of the whole production estimation, under `w0 = 0`."""
    if workers <= 1:
        _init_worker(pk, p_grid, n_trials, n_obs, base_seed, n_points, max_iter)
        return [_one_replicate(r) for r in range(n_rep)]
    import multiprocessing as mp
    ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
    with ctx.Pool(processes=workers, initializer=_init_worker,
                  initargs=(pk, p_grid, n_trials, n_obs, base_seed,
                            n_points, max_iter)) as pool:
        # chunksize 1: the cost per replicate is uniform, and small chunks keep
        # the ordering of the RETURNED list irrelevant (each result carries `r`).
        return list(pool.imap(_one_replicate, range(n_rep), chunksize=1))


def score_percentile(null_values: np.ndarray, observed: float,
                     cut: float = RULE_PERCENTILE_CUT) -> dict:
    """AO2's pre-declared convention, with the rejected one reported beside it.

    THE COMMITTED STATISTIC IS `100 * P(null < observed)`. Both are computed so
    that AO3 P5 -- do the two conventions straddle the cut? -- is answerable, and
    so that nobody has to take on trust that the choice did not matter.
    """
    v = np.asarray(null_values, dtype=float)
    strictly_below = 100.0 * float((v < observed).mean())
    at_or_below = 100.0 * float((v <= observed).mean())
    exceedance = float((v >= observed).mean())
    verdict = ("AJ2_THIRD_ARGUMENT_RESTORED" if strictly_below <= cut
               else "AJ2_THIRD_ARGUMENT_RETRACTED")
    return {
        "observed": float(observed),
        "n_replicates": int(v.size),
        "percentile_committed_strictly_below": strictly_below,
        "exceedance_p_value": exceedance,
        "percentile_rejected_convention_at_or_below": at_or_below,
        "convention_committed": "100 * P(null < observed)  [AO2]",
        "convention_note": (
            "the at-or-below figure is reported for transparency and is NOT the "
            "one the rule reads; AO2 gives the reason (an atom exactly at the "
            "observed value is evidence FOR unsurprisingness)"),
        "conventions_straddle_the_cut": bool(
            (strictly_below <= cut) != (at_or_below <= cut)),
        "cut": float(cut),
        # DERIVED from the number just computed (gotcha 20).
        "verdict": verdict,
        "verdict_rule": (
            f"percentile <= {cut} -> the exclusion of zero is NOT evidence of a "
            f"point mass, AJ2's third argument is RESTORED with a resolution "
            f"caveat; percentile > {cut} -> the point mass is supported and "
            f"AJ2's third argument is RETRACTED. Pre-declared in AO2; every "
            f"percentile in [0, 100] maps to exactly one branch."),
        "atom_at_observed_frac": float((v == observed).mean()),
        "median": float(np.median(v)),
        "mean": float(v.mean()),
        "min": float(v.min()),
        "max": float(v.max()),
        "quantiles": {q: float(np.percentile(v, q))
                      for q in (1, 5, 25, 50, 75, 95, 99)},
        "value_counts": {f"{val:.6g}": int(c) for val, c in
                         zip(*np.unique(v, return_counts=True))},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mixture", type=Path, required=True,
                    help="the PRIMARY mixture artifact (rf_mixture_n39.json). "
                         "Read only; never rewritten -- it is a canary target.")
    ap.add_argument("--mixture-n19", type=Path, default=None,
                    help="the sensitivity-arm mixture, for AO1's N-contrast")
    ap.add_argument("--theta", default="0.70")
    ap.add_argument("--n-rep", type=int, default=200)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--profile-max-iter", type=int, default=20_000)
    ap.add_argument("--skip-null", action="store_true",
                    help="1a and 1b only. The verdict is then reported as "
                         "NOT_RUN, never as a branch of the rule.")
    ap.add_argument("--out", type=Path, required=True,
                    help="NEW path. Amendment AO5: the baselined mixture "
                         "artifacts must not be overwritten by this session.")
    args = ap.parse_args()

    from src.sae.train import git_dirty, git_sha

    if args.out.resolve() == args.mixture.resolve():
        raise SystemExit(
            "--out is the input mixture artifact. That is a canary target "
            "(AO5); refusing to overwrite it.")

    doc = json.loads(args.mixture.read_text())
    n_trials = int(doc["identity"]["n_trials"])
    node = doc["by_theta"][args.theta]
    fit, prof = node["fit"], node["w0_profile"]
    p_grid = np.asarray(fit["p_grid"], dtype=float)
    weights = np.asarray(fit["weights"], dtype=float)
    n_obs = int(round(float(fit["n_anchors"])))
    counts = np.rint(np.asarray(node["fit"]["observed_atom_fracs"],
                                dtype=float) * n_obs)

    # ---- The grid the observed bound came off, asserted not assumed --------
    if int(prof["n_points"]) != PRODUCTION_PROFILE_POINTS:
        raise SystemExit(
            f"the artifact's profile used n_points={prof['n_points']}, not the "
            f"production {PRODUCTION_PROFILE_POINTS}. The observed lower bound "
            "and the null must come off the SAME grid; refusing to compare "
            "across grids.")
    observed_lower = float(prof["w0_interval"][0])
    if abs(observed_lower - EXPECTED_OBSERVED_LOWER_BOUND) > 1e-12:
        raise SystemExit(
            f"the artifact's lower bound is {observed_lower!r}, not the "
            f"{EXPECTED_OBSERVED_LOWER_BOUND} that AM6 decision 1 and AO2 are "
            "written about. The pre-declared rule is scored against a specific "
            "number; refusing to rescore it silently.")

    print(f"[w0] {args.mixture.name} theta={args.theta} n_compare={n_trials} "
          f"n_obs={n_obs:,}", flush=True)

    # ---- 1a -------------------------------------------------------------
    curve = curve_near_zero(prof)
    print(f"[1a] loglik at exactly w0=0 is {curve['drop_at_exactly_zero']:.4f} "
          f"below the profile max ({curve['drop_at_zero_in_units_of_the_cut']:.2f}x "
          f"the {curve['drop_units_defining_interval']}-unit cut; 1-df deviance "
          f"{curve['deviance_1df_at_zero']:.2f})", flush=True)
    n19_curve = None
    if args.mixture_n19 is not None:
        d19 = json.loads(args.mixture_n19.read_text())
        n19_curve = curve_near_zero(d19["by_theta"][args.theta]["w0_profile"])
        n19_curve["n_trials"] = int(d19["identity"]["n_trials"])
        print(f"[1a] contrast at n_compare={n19_curve['n_trials']}: drop at "
              f"exactly zero is {n19_curve['drop_at_exactly_zero']:.4f}",
              flush=True)

    # ---- 1b -------------------------------------------------------------
    grid = fine_w0_grid()
    t0 = time.time()
    fine = profile_w0(counts, n_trials, p_grid=p_grid, w0_grid=grid,
                      max_iter=args.profile_max_iter)
    fine_iv = interval_from_profile(grid, np.asarray(fine["profile_loglik"]),
                                    drop=float(prof["drop_units"]))
    lo_fine = fine_iv["w0_interval"][0]
    print(f"[1b] fine grid: {grid.size} points, spacing {grid[1]-grid[0]:.4f} "
          f"near zero -> interval [{lo_fine:.4f}, "
          f"{fine_iv['w0_interval'][1]:.4f}] (max at "
          f"{fine_iv['w0_at_max']:.4f}) in {time.time()-t0:.1f}s", flush=True)

    p1 = {
        "prediction": "the refitted lower bound lands strictly inside (0, 0.025)",
        "category": "within-system comparison; ENTAILMENT CHECK of 1a, not an "
                    "independent test (AO3 P1)",
        "coarse_lower_bound": observed_lower,
        "fine_lower_bound": lo_fine,
        "outcome": "HIT" if (0.0 < lo_fine < observed_lower) else "MISS",
    }
    p2 = {
        "prediction": f"the refitted lower bound is at least {P2_MIN_LOWER_BOUND}",
        "category": "within-system comparison",
        "fine_lower_bound": lo_fine,
        "threshold": P2_MIN_LOWER_BOUND,
        "outcome": "HIT" if lo_fine >= P2_MIN_LOWER_BOUND else "MISS",
    }

    # ---- 1c -------------------------------------------------------------
    null_block: dict
    if args.skip_null:
        null_block = {
            "status": "NOT_RUN (--skip-null)",
            "verdict": "NOT_RUN -- no branch of AO2's rule has fired",
        }
        p3 = p4 = p5 = {"outcome": "NOT_SCORED -- the null was not simulated"}
    else:
        truth = null_truth(weights, p_grid)
        pk = atom_pmf(truth["weights"], p_grid, n_trials)
        t0 = time.time()
        reps = simulate_null(pk, p_grid, n_trials, n_obs, args.n_rep,
                             base_seed=args.base_seed, workers=args.workers,
                             max_iter=args.profile_max_iter)
        secs = time.time() - t0
        lower = np.array([r["profiled_lower"] for r in reps])
        point = np.array([r["point_w0"] for r in reps])
        primary = score_percentile(lower, observed_lower)
        secondary = score_percentile(point, observed_lower)
        obs_k0 = float(counts[0] / counts.sum())
        null_block = {
            "protocol": {
                "structure": "SINGLE POOLED NPMLE per replicate, matching the "
                             "production fit (AO2)",
                "n_observations_per_replicate": n_obs,
                "n_trials": n_trials,
                "p_grid_size": int(p_grid.size),
                "profile_grid_points": PRODUCTION_PROFILE_POINTS,
                "profile_grid_spacing": PRODUCTION_PROFILE_SPACING,
                "n_replicates": int(args.n_rep),
                "base_seed": int(args.base_seed),
                "sampler": "Multinomial(n_obs, pk) -- EXACTLY equivalent to the "
                           "two-stage p-then-k draw; checked in "
                           "tests/test_w0_resolution.py, not asserted",
                "seconds": secs,
            },
            "truth": {
                "w0_removed_and_renormalised": truth["removed_w0"],
                "mass_remaining_before_renormalisation": truth["renormalised_from"],
                "w0_of_truth": float(truth["weights"][0]),
                "pk_k0_under_truth": float(pk[0]),
                "observed_atom_frac_k0": obs_k0,
                "note": ("P(k=0) under the w0=0 truth against the observed "
                         "value. A diff, not a verdict: the gap is the same "
                         "question from the data side."),
            },
            "PRIMARY_profiled_lower_bound": primary,
            "SECONDARY_point_estimate_w0": secondary,
            "n_replicates_whose_fit_converged": int(
                sum(1 for r in reps if r["fit_converged"])),
            "replicates": reps,
        }
        print(f"[1c] {args.n_rep} replicates in {secs/60:.1f} min. "
              f"null profiled lower bound: median {primary['median']:.4f}, "
              f"max {primary['max']:.4f}", flush=True)
        print(f"[1c] percentile of {observed_lower} (AO2 convention) = "
              f"{primary['percentile_committed_strictly_below']:.2f}  "
              f"-> {primary['verdict']}", flush=True)
        print(f"[1c] rejected convention would read "
              f"{primary['percentile_rejected_convention_at_or_below']:.2f}; "
              f"straddle={primary['conventions_straddle_the_cut']}", flush=True)
        p3 = {
            "prediction": "the percentile exceeds 95, so the rule RETRACTS "
                          "AJ2's third argument",
            "category": "PROCEDURE-BEHAVIOUR (the 0/2 bin), filed there "
                        "deliberately -- AO3 P3",
            "percentile": primary["percentile_committed_strictly_below"],
            "outcome": "HIT" if primary["verdict"].endswith("RETRACTED") else "MISS",
        }
        p4 = {
            "prediction": "the null distribution of the profiled lower bound "
                          "has its MEDIAN at exactly 0.0",
            "category": "distributional shape",
            "median": primary["median"],
            "outcome": "HIT" if primary["median"] == 0.0 else "MISS",
        }
        p5 = {
            "prediction": "the two percentile conventions do NOT straddle 95",
            "category": "distributional shape",
            "straddle": primary["conventions_straddle_the_cut"],
            "outcome": "HIT" if not primary["conventions_straddle_the_cut"]
                       else "MISS",
        }

    # ---- the PURITY BAND at the fine-grid interval (brief section 3) -------
    # The strict endpoint is linear in w0 and the loose endpoint is NOT (AJ1), so
    # the weights are refitted at each of the three w0 values rather than the old
    # band being rescaled. The pre-registered unstable group is used, so the band
    # is on the group the analysis actually reports.
    ag = _ag_cuts_or_absent(n_trials)
    if "primary_unstable_k_max" in ag:
        group = list(range(int(ag["primary_unstable_k_max"]) + 1))
        band = {
            "group_ks": group,
            "group_source": "AG1's pre-registered primary unstable cut",
            "at_fine_lower_bound": purity_at_w0(counts, n_trials, p_grid,
                                                lo_fine, group),
            "at_fine_max": purity_at_w0(counts, n_trials, p_grid,
                                        fine_iv["w0_at_max"], group),
            "at_fine_upper_bound": purity_at_w0(counts, n_trials, p_grid,
                                                fine_iv["w0_interval"][1], group),
            "at_coarse_lower_bound_SUPERSEDED": purity_at_w0(
                counts, n_trials, p_grid, observed_lower, group),
            "at_coarse_upper_bound_SUPERSEDED": purity_at_w0(
                counts, n_trials, p_grid, float(prof["w0_interval"][1]), group),
            "note": ("the strict endpoint is exactly w0 / P(k in group) and the "
                     "identity is re-derived and compared, not quoted; the loose "
                     "endpoint is not linear in w0, which is why the weights are "
                     "refitted at each value rather than the band rescaled"),
        }
        print(f"[3] purity band on k <= {group[-1]}: strict "
              f"[{band['at_fine_lower_bound']['strict_p_eq_0']:.4f}, "
              f"{band['at_fine_upper_bound']['strict_p_eq_0']:.4f}] at the fine "
              f"interval; loose "
              f"[{band['at_fine_lower_bound']['loose_p_le_0.10']:.4f}, "
              f"{band['at_fine_upper_bound']['loose_p_le_0.10']:.4f}]", flush=True)
    else:
        band = {"status": "NO_PREREGISTERED_CUTS", "detail": ag}

    out = {
        "identity": {
            "measurement": "w0_resolution_AO",
            "amendment": "AO sections 1-3; the author decision AM6.1",
            "firewall_crossing": "NONE -- reads the r_f marginal's mixture fit "
                                 "only; no alignment, no M",
            "mixture_source": str(args.mixture),
            "mixture_n19_source": (str(args.mixture_n19)
                                   if args.mixture_n19 else None),
            "theta": args.theta,
            "n_trials": n_trials,
            "n_observations": n_obs,
            "p_grid_size": int(p_grid.size),
            "production_profile_points": PRODUCTION_PROFILE_POINTS,
            "production_profile_spacing": PRODUCTION_PROFILE_SPACING,
            "observed_lower_bound": observed_lower,
            "git_sha": git_sha(), "git_dirty": git_dirty(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
        "AO1_curve_near_zero": curve,
        "AO1_curve_near_zero_n19_contrast": n19_curve,
        "AO1b_fine_grid": {
            "grid": grid.tolist(),
            "profile_loglik": fine["profile_loglik"],
            "interval": fine_iv,
            "coarse_interval": [float(x) for x in prof["w0_interval"]],
            "coarse_spacing": PRODUCTION_PROFILE_SPACING,
        },
        "AO1c_null_from_w0_zero": null_block,
        "purity_band_at_the_fine_interval": band,
        "prediction_scorecard": {"P1": p1, "P2": p2, "P3": p3, "P4": p4,
                                 "P5": p5},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[w0] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
