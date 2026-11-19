"""
The MIXING DISTRIBUTION behind `r_f`, and honest ENDPOINT PURITY.
Governed by PREREGISTRATION amendment AB (sections 3 and 4).

THE QUESTION. M1 reported the `r_f` marginal as "near-uniform across
k = 1..19". That phrasing describes the histogram of OBSERVED counts. The
object the design actually cares about is the distribution of the underlying
per-feature reappearance probability `p` -- because `k ~ Binomial(19, p)`
adds sampling noise on top of whatever `p` does, and the two are easy to
confuse.

THE FACT THAT MAKES THIS URGENT. If `p ~ Uniform(0,1)` then `Binomial(N, p)`
has a marginal that is EXACTLY uniform on {0..N} -- every atom `1/(N+1)`.
So a flat observed histogram is the signature of a broad `p`, not of anything
pathological; but it also means the endpoint bins are heavily populated by
mid-range features that simply drew an extreme count. Quantifying that leak
is what section 4 asks for.

=============================================================================
EPISTEMIC STATUS OF THE SECTION-3 PREDICTIONS. READ THIS BEFORE SCORING THEM.
=============================================================================
The four shape predictions scored here were formed by the PLANNING THREAD from
the four atoms published in amendment Z -- P(0), P(1), P(18), P(19) -- without
access to the full 20-atom histogram. From that author's standpoint they are
genuine ex-ante predictions and are scored as such.

**They are NOT pre-registered by THIS module, and this module does not claim
they are.** The full histogram had already been read and printed during the
M1 session, before the brief arrived. Committing them to git now and calling
them pre-run would be a FABRICATED AUDIT TRAIL -- worse than the miss it
would dress up, and precisely the failure the amendment-policy git-order rule
exists to make impossible.

So: scored, attributed, and labelled **POST-HOC (by this module)** in the
output, with the honest caveat carried in the artifact itself rather than in a
commit message that nobody re-reads. Predictions 2 and 3 were already
falsified by data in hand at the time the brief was written; that is stated
rather than presented as a discovery.
=============================================================================

WHAT IS FITTED. A nonparametric mixing distribution over `p` by EM (an NPMLE
on a fixed grid). The grid INCLUDES exactly 0 and exactly 1, so a genuine
point mass at either endpoint can be represented rather than smeared into the
nearest interior knot -- which matters, because "is the low group a point mass
at p = 0 or a cluster at small p > 0" is the question section 4 says is not
resolvable at N = 19, and a grid without an exact 0 would prejudge it.

WHY NPMLE RATHER THAN THE THREE-COMPONENT SKETCH. The brief's
`w0 / w1 / Uniform` decomposition is exact arithmetic on four atoms, but it
ASSUMES the interior is flat -- and the brief's own mean check refutes that
assumption (a flat interior contributes 0.4069 against the 0.4318 required).
The sketch is reproduced here as a cross-check and is explicitly superseded by
the fit, per the brief's instruction.

=============================================================================
CONVERGENCE PREDICTIONS -- COMMITTED BEFORE THE RE-RUN (close-out §1.1).
=============================================================================
The EM below previously stopped on a LOG-LIKELIHOOD criterion and hit its
20,000-iteration cap at EVERY theta (0.30/0.50/0.65/0.70/0.75 all reported
`iterations: 20000`), with no `converged` flag to say so. The fix converges on
the WEIGHT VECTOR instead, because an NPMLE's loglik plateaus long before `w`
stabilises and the least-identified part of the fit is exactly the split
between an atom at `p = 0` and mass at small-but-nonzero `p` -- which is the
difference between the 33.6% and 91.3% purity endpoints. The quantity most at
risk is the one carrying the finding.

Committed here BEFORE the re-run; `git log src/analysis/rf_mixture.py` is the
audit trail. All are WITHIN-SYSTEM comparisons (both sides our data).

  C1. The weight-vector criterion will NOT be met under any practical cap.
      Reason, stated as arithmetic rather than intuition: the grid has 201
      support points and the data supply 20 atoms, so `w` has 201 free
      parameters against 20 binding constraints. The MARGINAL is identified;
      `w` is not. So EM wanders a plateau of near-equal loglik and
      `max|dw|` does not go to zero. PREDICTION: cap binds, and that is
      NON-IDENTIFIABILITY, not a broken optimiser.
  C2. The FITTED ATOM FRACTIONS will be stable to <= 1e-3 regardless, because
      they are the identified part. So `max_abs_atom_residual` stays ~5e-4.
  C3. Because of C1, the reported `w0` will move materially between the old
      and new stopping rules -- I do NOT predict a direction (that would be a
      directional mechanism claim, the 0/8 bin).
  C4. The profiled-loglik interval for `w0` (within 2 loglik units) will be
      WIDE -- spanning more than 0.10 in `w0` -- and will therefore make the
      purity band a measured interval rather than two definitional endpoints.
  C5. The coarse thirds (0.329/0.337/0.335) will be RELATIVELY robust,
      changing by less than 0.05 each, because thirds aggregate over ~67 grid
      points and averaging over a plateau is exactly what makes a functional
      of `w` identified when `w` is not.

If C1 holds, the deliverable is C4's interval, not a converged `w`.
=============================================================================

PURITY IS DEFINITION-DEPENDENT AND THE DEFINITION IS ALWAYS REPORTED.
"Genuinely unstable" can mean `p == 0` exactly, or `p < 0.05`, or `p < 0.1`.
These give materially different numbers (the addendum notes 46% versus 74%),
so every purity figure emitted here carries its threshold. A purity number
without its definition is not a number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom

from src.analysis.rf_marginal import ag_cuts

# Grid for the mixing distribution. Exact 0 and exact 1 are included
# DELIBERATELY (see module docstring); the interior is fine enough that the
# fit is not knot-limited at N = 19.
P_GRID = np.concatenate([[0.0], np.linspace(0.005, 0.995, 199), [1.0]])

# Purity thresholds. Every reported purity states which of these it used.
UNSTABLE_DEFS = {"p_eq_0": 0.0, "p_lt_0.05": 0.05, "p_lt_0.10": 0.10}
STABLE_DEFS = {"p_eq_1": 0.0, "p_gt_0.95": 0.05, "p_gt_0.90": 0.10}


# Production EM budget. ONE constant, used by the fit AND by its calibration
# arm -- they previously differed 10x (20,000 vs 2,000) while the calibration
# docstring claimed "the SAME iteration budget", so the observed fit and its
# reference were stopped at different places.
EM_MAX_ITER = 200_000
# Convergence is on the WEIGHT VECTOR, not the log-likelihood. An NPMLE loglik
# plateaus long before `w` settles, and the least-identified part of the fit is
# the split between an atom at p=0 and mass at small p>0 -- exactly the
# quantity the purity endpoints turn on.
EM_W_TOL = 1e-6


def fit_mixture(atom_counts: np.ndarray, n_trials: int,
                p_grid: np.ndarray = P_GRID,
                max_iter: int = EM_MAX_ITER, w_tol: float = EM_W_TOL) -> dict:
    """NPMLE of the mixing distribution over `p` by EM.

    `atom_counts[k]` is how many anchors were observed with exactly k matches.

    CONVERGENCE IS ON `w`, AND HITTING THE CAP IS A FAILURE, NOT A RESULT.
    The previous version stopped when the loglik stopped improving by a
    relative 1e-12 and reported only `iterations`; it hit its 20,000 cap at
    every theta and nothing said so. `converged` is now returned explicitly,
    with `max_abs_dw` beside it, so a consumer sees which it got.

    Note what this CANNOT fix: with 201 grid points against `n_trials + 1`
    atoms of data, `w` has far more free parameters than binding constraints.
    The MARGINAL is identified and `w` is not, so a plateau in `w` is the
    EXPECTED outcome and means non-identifiability rather than a broken
    optimiser. When that happens use `profile_w0`: the honest object is an
    interval for `w0`, not a point estimate that depends on where EM stopped.
    """
    k = np.arange(n_trials + 1)
    # lik[j, k] = P(k | p_grid[j])
    lik = binom.pmf(k[None, :], n_trials, p_grid[:, None])
    w = np.full(p_grid.size, 1.0 / p_grid.size)
    n = atom_counts.astype(np.float64)
    ntot = n.sum()

    ll = float("nan")
    max_abs_dw = float("inf")
    it = -1
    for it in range(max_iter):
        joint = w[:, None] * lik                       # (grid, k)
        marg = joint.sum(axis=0)                       # (k,)
        marg = np.where(marg > 0, marg, 1e-300)
        resp = joint / marg[None, :]
        w_new = (resp * n[None, :]).sum(axis=1) / ntot
        ll = float((n * np.log(marg)).sum())
        max_abs_dw = float(np.abs(w_new - w).max())
        w = w_new
        if max_abs_dw < w_tol:
            break

    iterations = int(it + 1)
    converged = bool(max_abs_dw < w_tol)
    fitted = (w[:, None] * lik).sum(axis=0)
    return {
        "p_grid": p_grid.tolist(),
        "weights": w.tolist(),
        "loglik": ll,
        "iterations": iterations,
        "max_iter": int(max_iter),
        "w_tol": float(w_tol),
        "max_abs_dw": max_abs_dw,
        # DERIVED from what was just computed, never a sentence asserting the
        # expected reading (gotcha 20).
        "converged": converged,
        "cap_bound": bool(iterations >= max_iter),
        "convergence_verdict": ("CONVERGED" if converged else
                                "CAP BOUND -- w did not settle; treat as "
                                "NON-IDENTIFIED and use profile_w0"),
        "fitted_atom_fracs": fitted.tolist(),
        "observed_atom_fracs": (n / ntot).tolist(),
        "max_abs_atom_residual": float(np.abs(fitted - n / ntot).max()),
        "mass_at_p_eq_0": float(w[0]),
        "mass_at_p_eq_1": float(w[-1]),
        "mean_p": float((w * p_grid).sum()),
        "n_anchors": float(ntot),
    }


def profile_w0(atom_counts: np.ndarray, n_trials: int,
               p_grid: np.ndarray = P_GRID,
               n_points: int = 41, drop: float = 2.0,
               max_iter: int = 20_000, w_tol: float = EM_W_TOL,
               w0_grid: np.ndarray | None = None) -> dict:
    """PROFILE the log-likelihood over `w0`, the mass at exactly `p = 0`.

    `w0_grid` OVERRIDES `n_points` (amendment AO, section 1b). It exists because
    the reported interval's endpoints are grid points, so "the interval excludes
    zero by one grid step" is a statement about THIS grid's spacing -- 0.025 at
    the default `n_points = 41` -- and separating that from the likelihood's own
    shape needs a finer, not necessarily uniform, grid. When it is None the
    behaviour is EXACTLY the pre-AO behaviour (`np.linspace(0, 1, n_points)`),
    which `tests/test_w0_resolution.py` pins bit-for-bit: the baselined mixture
    artifacts must stay reproducible, and an optional argument that quietly
    changed the default path would break that silently.

    THE PRE-DECLARED CONTINGENCY. If `w` will not settle because it is not
    identified, a tighter tolerance is the wrong response -- the right one is to
    report the RANGE of `w0` consistent with a near-maximal loglik. That turns
    the purity band from two definitional endpoints into a MEASURED interval,
    which is a better object than either.

    For each candidate `w0` the remaining grid is re-fitted by EM with the
    interior mass constrained to `1 - w0`; the profile loglik is the best
    achievable at that `w0`. The interval returned is every `w0` within `drop`
    loglik units of the profile maximum (2.0 by default, the conventional
    single-parameter ~95% cut of 1.92 rounded up).
    """
    k = np.arange(n_trials + 1)
    lik = binom.pmf(k[None, :], n_trials, p_grid[:, None])
    n = atom_counts.astype(np.float64)
    ntot = n.sum()

    grid_w0 = (np.linspace(0.0, 1.0, n_points) if w0_grid is None
               else np.asarray(w0_grid, dtype=float))
    profile = []
    for w0 in grid_w0:
        w = np.full(p_grid.size, (1.0 - w0) / max(p_grid.size - 1, 1))
        w[0] = w0
        ll = float("nan")
        for _ in range(max_iter):
            joint = w[:, None] * lik
            msum = joint.sum(axis=0)
            marg = np.where(msum > 0, msum, 1e-300)
            resp = joint / marg[None, :]
            w_new = (resp * n[None, :]).sum(axis=1) / ntot
            # re-impose the constraint: w0 held, the remainder renormalised
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
        profile.append(ll)

    profile = np.asarray(profile, dtype=float)
    best = int(np.nanargmax(profile))
    ok = np.flatnonzero(profile >= profile[best] - drop)
    return {
        "w0_grid": grid_w0.tolist(),
        "profile_loglik": profile.tolist(),
        "w0_at_max": float(grid_w0[best]),
        "loglik_at_max": float(profile[best]),
        "drop_units": float(drop),
        "w0_interval": [float(grid_w0[ok.min()]), float(grid_w0[ok.max()])],
        "w0_interval_width": float(grid_w0[ok.max()] - grid_w0[ok.min()]),
        "n_points": int(n_points),
    }


def three_component_sketch(atoms: np.ndarray, n_trials: int) -> dict:
    """The brief's closed-form preview, reproduced as a CROSS-CHECK.

    Assumes a flat interior at height u, a point mass w0 at p=0 and w1 at p=1.
    Also computes the mean the assumption implies, so the assumption's own
    refutation is emitted alongside it rather than left to a reader.
    """
    N = n_trials
    u = (1.0 - atoms[0] - atoms[N]) / (N - 1)
    w0, w1 = atoms[0] - u, atoms[N] - u
    closure = (N + 1) * u + w0 + w1
    ks = np.arange(N + 1)
    observed_mean = float((atoms * ks).sum() / N)
    interior_required = observed_mean - atoms[N] * 1.0
    interior_if_flat = float(u * ks[1:N].sum() / N)
    return {
        "u": float(u), "w0": float(w0), "w1": float(w1),
        "closure_should_be_1": float(closure),
        "observed_mean_r_f": observed_mean,
        "interior_contribution_required": float(interior_required),
        "interior_contribution_if_flat": interior_if_flat,
        "shortfall": float(interior_required - interior_if_flat),
        "flat_interior_is_consistent": bool(
            abs(interior_required - interior_if_flat) < 1e-3),
        "note": ("Superseded by the NPMLE fit. Kept because its own mean "
                 "check refutes the flat-interior assumption it rests on."),
    }


def purity(fit: dict, n_trials: int, group_ks: list[int],
           genuine_mask: np.ndarray) -> dict:
    """P(genuine | k in group) under the fitted mixture.

    `genuine_mask` selects the grid points counted as genuine. Everything else
    contributes 'leak'.
    """
    w = np.asarray(fit["weights"])
    p = np.asarray(fit["p_grid"])
    pk = binom.pmf(np.asarray(group_ks)[None, :], n_trials, p[:, None]).sum(axis=1)
    joint = w * pk
    total = float(joint.sum())
    gen = float(joint[genuine_mask].sum())
    return {
        "group_size": total,
        "genuine": gen,
        "leaked": total - gen,
        "purity": (gen / total) if total > 0 else float("nan"),
    }



# Which purity definitions apply to which group. An EXPLICIT map, not a prefix
# test: the original `startswith("k_le") or startswith("k_eq_0")` happened to be
# right for four names and silently mis-classifies any fifth, which is the
# incomplete-instrument shape of gotcha 24. Unknown names RAISE rather than
# defaulting to a side.
_UNSTABLE_GROUPS = frozenset({
    "k_le_1", "k_eq_0", "ag_primary_unstable", "ag_sensitivity_unstable"})
_STABLE_GROUPS = frozenset({
    "k_ge_Nminus1", "k_eq_N", "ag_primary_stable", "ag_sensitivity_stable"})


def _ag_cuts_or_absent(n_trials: int) -> dict:
    """AG1's cuts, or an explicit record that none are pre-registered at this N.
    Reported as absent, never substituted (gotcha 15)."""
    try:
        return ag_cuts(n_trials)
    except KeyError as exc:
        return {"status": "NO_PREREGISTERED_CUTS", "reason": str(exc)}


def _is_unstable_group(gname: str) -> bool:
    if gname in _UNSTABLE_GROUPS:
        return True
    if gname in _STABLE_GROUPS:
        return False
    raise KeyError(
        f"group {gname!r} is classified as neither unstable nor stable. A "
        f"purity number whose 'genuine' definition was picked by a prefix test "
        f"is not a number; add it to one of the two sets deliberately.")


def analyse_theta(atoms_frac: np.ndarray, n_anchors: int, n_trials: int) -> dict:
    counts = np.rint(np.asarray(atoms_frac) * n_anchors)
    fit = fit_mixture(counts, n_trials)
    p = np.asarray(fit["p_grid"])

    # The four legacy groups keep their LITERAL definitions -- each key names
    # exactly the k-set it sums, which stays true at any N -- and AG1's
    # pre-registered groups are added beside them. At N = 19 the two coincide
    # (`ag_primary_unstable` == `k_le_1`, `ag_primary_stable` == `k_ge_Nminus1`)
    # and that coincidence is ASSERTED below rather than assumed, because it is
    # the only N at which a key named `k_le_1` and the pre-registered unstable
    # cut are the same object.
    groups = {
        "k_le_1": [0, 1], "k_eq_0": [0],
        "k_ge_Nminus1": [n_trials - 1, n_trials], "k_eq_N": [n_trials],
    }
    # AG1 pre-registers cuts for N in {19, 39} only. Where they exist they are
    # added; where they do not, their ABSENCE is reported and the fit still
    # runs. Refusing the whole analysis because no endpoint cut was
    # pre-registered would make a general estimator unusable outside the two
    # values of N this study reports, and inventing a cut is the thing ag_cuts
    # exists to refuse.
    try:
        cuts = ag_cuts(n_trials)
    except KeyError as exc:
        cuts = {"status": "NO_PREREGISTERED_CUTS", "reason": str(exc)}
    else:
        groups["ag_primary_unstable"] = list(
            range(cuts["primary_unstable_k_max"] + 1))
        groups["ag_primary_stable"] = list(
            range(cuts["primary_stable_k_min"], n_trials + 1))
        groups["ag_sensitivity_unstable"] = [0]
        groups["ag_sensitivity_stable"] = [n_trials]
    if n_trials == 19:
        assert groups["ag_primary_unstable"] == groups["k_le_1"], (
            "at N = 19 AG1's unstable cut IS k <= 1; a disagreement means the "
            "committed cut table and the legacy group have diverged")
        assert groups["ag_primary_stable"] == groups["k_ge_Nminus1"], (
            "at N = 19 AG1's stable cut IS k >= N-1")
    out_pur: dict = {}
    for gname, gks in groups.items():
        defs = UNSTABLE_DEFS if _is_unstable_group(gname) else STABLE_DEFS
        for dname, eps in defs.items():
            mask = (p <= eps) if defs is UNSTABLE_DEFS else (p >= 1.0 - eps)
            out_pur[f"{gname}|{dname}"] = purity(fit, n_trials, gks, mask)

    # Closed-form cross-check the brief supplies: a Uniform(0,1) component
    # leaks exactly 2/(N+1) into the k<=1 bin. Verified against the grid,
    # because a formula quoted without checking is how this round already
    # produced two wrong intermediates.
    unif = np.ones(p.size) / p.size
    leak_grid = float((unif * binom.pmf(
        np.array([0, 1])[None, :], n_trials, p[:, None]).sum(axis=1)).sum())
    # THE CONTINGENCY, run unconditionally rather than only when the fit fails.
    # Running it always means the interval is on the record whichever way
    # convergence came out, so nobody has to decide after seeing the verdict
    # whether to compute it -- which is the move that turns a contingency into a
    # choice made with the answer in hand.
    prof = profile_w0(counts, n_trials)
    return {
        "fit": fit,
        "w0_profile": prof,
        "purity_band_reading": (
            "If fit.converged is False the point estimate of w0 depends on "
            "where EM stopped and must NOT be quoted; use "
            "w0_profile.w0_interval, which is the range of w0 within "
            f"{prof['drop_units']} loglik units of the profile maximum."),
        "sketch": three_component_sketch(np.asarray(atoms_frac), n_trials),
        "purity": out_pur,
        "uniform_leak_into_k_le_1": {
            "closed_form_2_over_Nplus1": 2.0 / (n_trials + 1),
            "grid_numeric": leak_grid,
            "agree": bool(abs(leak_grid - 2.0 / (n_trials + 1)) < 5e-3),
        },
    }


def score_shape_predictions(atoms: np.ndarray, se: np.ndarray,
                            n_trials: int, u: float) -> dict:
    """Section 3's four predictions, DERIVED (gotcha 20), labelled POST-HOC.

    `se` is the ACROSS-SEED standard error, not the pooled binomial one --
    the pooled figure ignores seed-level dependence and would make sub-0.2pp
    wiggles look like 3-sigma features.
    """
    interior = atoms[1:n_trials]
    ise = se[1:n_trials]
    ks = np.arange(1, n_trials)

    # P1 -- an interior local maximum away from the edges, exceeding BOTH
    # neighbours by more than the across-seed error.
    modes = []
    for i in range(1, interior.size - 1):
        d_lo = interior[i] - interior[i - 1]
        d_hi = interior[i] - interior[i + 1]
        thresh = max(ise[i], 1e-12)
        if d_lo > thresh and d_hi > thresh:
            modes.append({"k": int(ks[i]), "frac": float(interior[i]),
                          "excess_over_lower_in_se": float(d_lo / thresh),
                          "excess_over_upper_in_se": float(d_hi / thresh)})

    lo, hi = u - 0.006, u + 0.006
    outside = [{"k": int(ks[i]), "frac": float(interior[i])}
               for i in range(interior.size)
               if not (lo <= interior[i] <= hi)]

    diffs = np.diff(interior)
    n_down = int((diffs < 0).sum())
    return {
        "_epistemic_status": (
            "POST-HOC. Formed by the planning thread from the four atoms in "
            "amendment Z, without the full histogram; scored here against "
            "data this session had ALREADY read. Not pre-registered by this "
            "module and not presented as such."),
        "AB3_P1_no_interior_mode": {
            "interior_modes_found": modes,
            "outcome": "HIT" if not modes else "MISS",
        },
        "AB3_P2_interior_atoms_within_0.6pp_of_u": {
            "u": u, "band": [lo, hi],
            "n_outside": len(outside), "outside": outside[:8],
            "outcome": "HIT" if not outside else "MISS",
        },
        "AB3_P3_interior_monotone_nondecreasing": {
            "n_decreasing_steps": n_down,
            "first_decrease_at_k": int(ks[int(np.argmax(diffs < 0))] )
                                   if n_down else None,
            "outcome": "HIT" if n_down == 0 else "MISS",
        },
        "AB3_P4_broad_not_bimodal": {
            "note": "scored from the fitted mixing distribution, see 'bimodality'",
        },
    }


def shape_diagnosis(fit: dict, n_trials: int, n_anchors: float,
                    n_sim: int = 40, max_iter: int = EM_MAX_ITER,
                    rng_seed: int = 0) -> dict:
    """Is the fitted mixing distribution BIMODAL, or does the fit just look
    that way because an NPMLE is spiky?

    THIS FUNCTION'S FIRST DRAFT WAS WRONG AND THE WRONGNESS IS THE POINT
    (gotcha 20). It counted local maxima of the decile-binned weights and
    reported `BIMODAL` with modes at deciles [2, 4, 6, 8] -- an ALTERNATING
    pattern, which is the signature of an NPMLE artifact, not of bimodality.
    The MLE of a mixing distribution is discrete with at most `n_trials + 1`
    support points; binning ~20 spikes into 10 deciles produces high/low
    oscillation depending on where the spikes land. A detector that calls
    that "bimodal" would have contradicted the correct reading of the data
    while sounding confident.

    THE FIX IS GOTCHA 23'S MOVE: calibrate the statistic against its own
    sampling distribution. Simulate from a SMOOTH reference truth (point
    masses at 0 and 1 at the fitted heights, uniform in between), draw the
    same number of anchors, refit with the SAME EM at the SAME iteration
    budget (`EM_MAX_ITER` -- these differed 10x until the close-out, so
    the observed fit and its reference were stopped at different places
    while this sentence claimed otherwise), and count interior decile
    modes. If a smooth truth reproduces the
    observed mode count, the mode count carries no information about
    bimodality and must not be used as if it did.
    """
    w = np.asarray(fit["weights"])
    p = np.asarray(fit["p_grid"])

    def decile_modes(weights: np.ndarray) -> tuple[np.ndarray, list[int]]:
        edges = np.linspace(0, 1, 11)
        b = np.array([
            weights[(p >= edges[i]) & ((p <= edges[i + 1]) if i == 9
                                       else (p < edges[i + 1]))].sum()
            for i in range(10)])
        m = [i for i in range(1, 9)
             if b[i] > b[i - 1] and b[i] > b[i + 1]]
        return b, m

    binned, modes = decile_modes(w)

    # ---- the calibration: a SMOOTH truth, same n, same estimator ----
    w0, w1 = float(w[0]), float(w[-1])
    truth = np.zeros(p.size)
    truth[0], truth[-1] = w0, w1
    truth[1:-1] = (1.0 - w0 - w1) / (p.size - 2)      # uniform interior
    k = np.arange(n_trials + 1)
    lik = binom.pmf(k[None, :], n_trials, p[:, None])
    pk_truth = (truth[:, None] * lik).sum(axis=0)
    pk_truth = pk_truth / pk_truth.sum()

    rng = np.random.default_rng(rng_seed)
    sim_modes = []
    for _ in range(n_sim):
        draw = rng.multinomial(int(n_anchors), pk_truth)
        f = fit_mixture(draw, n_trials, p_grid=p, max_iter=max_iter)
        _, m = decile_modes(np.asarray(f["weights"]))
        sim_modes.append(len(m))
    sim_modes = np.asarray(sim_modes)

    n_obs_modes = len(modes)
    p_value = float((sim_modes >= n_obs_modes).mean())
    informative = bool(p_value < 0.05)
    interior_mass = float(binned[1:9].sum())
    return {
        "decile_mass": binned.tolist(),
        "interior_decile_modes": modes,
        "n_interior_modes_observed": n_obs_modes,
        "n_interior_modes_under_smooth_truth": {
            "n_sim": int(n_sim), "median": float(np.median(sim_modes)),
            "min": int(sim_modes.min()), "max": int(sim_modes.max()),
            "p_value_ge_observed": p_value,
        },
        "mode_count_is_informative": informative,
        "interior_mass": interior_mass,
        "mass_in_first_decile": float(binned[0]),
        "mass_in_last_decile": float(binned[9]),
        # ---- THE SHAPE VERDICT IS BASED ON COARSE MASS, NOT ON MODES ----
        # SECOND DEFECT IN THIS FUNCTION, and it survived the first fix
        # (gotcha 20 again). Once the mode count was calibrated it came out
        # "informative" (4 observed, smooth truth gives at most 3), and the
        # verdict then read BIMODAL. But the calibration tests departure from
        # a UNIFORM-INTERIOR reference -- it does NOT test bimodality, and
        # the observed modes sit at deciles [2, 4, 6, 8], an alternating
        # pattern no bimodal distribution has. Meanwhile the mass in coarse
        # thirds of p is 0.329 / 0.337 / 0.335: as broad as a distribution
        # can be. The label contradicted the data it was computed from.
        #
        # So the verdict uses THIRDS, which are coarse enough that NPMLE
        # spikiness cannot move them, and the mode result is reported
        # separately with a description of what it actually establishes.
        "mass_in_thirds": [float(binned[0:3].sum()), float(binned[3:7].sum()),
                           float(binned[7:10].sum())],
        "shape": (
            "BIMODAL" if (binned[0:3].sum() > 0.3 and binned[7:10].sum() > 0.3
                          and binned[3:7].sum() < 0.15)
            else "BROAD" if min(binned[0:3].sum(), binned[3:7].sum(),
                                binned[7:10].sum()) > 0.20
            else "EDGE_DOMINATED" if interior_mass < 0.5
            else "INTERMEDIATE"),
        "shape_basis": "mass in coarse thirds of p; robust to NPMLE spikiness",
        "departs_from_uniform_interior": informative,
        "departure_note": (
            "The calibrated mode count tests departure from a UNIFORM-INTERIOR "
            "reference. It does NOT test bimodality and must not be quoted as "
            "if it did. A significant result here means only that the interior "
            "is not flat -- which the mean-shortfall arithmetic establishes "
            "independently."),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m1", type=Path, required=True,
                    help="rf_marginal_m1.json")
    ap.add_argument("--rule", default="mnn")
    ap.add_argument("--thetas", nargs="+", default=["0.70", "0.65", "0.30"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    doc = json.loads(args.m1.read_text())
    # `n_trials` is the number of COMPARISON seeds. It is read EXPLICITLY rather
    # than derived.
    #
    # CORRECTED IN AN6: an earlier version of this comment said `n_compare`
    # "stopped equalling `n_seeds - 1`" at the anchor/comparison split. THAT IS
    # WRONG -- an anchor compares against every OTHER dictionary, so
    # `n_compare = n_seeds - 1 = 39` still holds and the fallback below would
    # have given the right answer. What broke is `n_compare = n_anchors - 1`,
    # true only while every seed was an anchor.
    #
    # Reading the attr is still right, for the reason that survives the
    # correction: an explicit value cannot be wrong about itself, while a derived
    # one is only as good as the derivation -- and this session shipped a wrong
    # derivation in a comment while the code happened to be safe.
    ident = doc["identity"]
    n_trials = int(ident.get("n_compare_seeds", int(ident["n_seeds"]) - 1))
    if "n_compare_seeds" in ident and "n_anchors" in ident:
        print(f"[mix] {ident['n_anchors']} anchors x {n_trials} comparison "
              f"seeds (n_seeds = {ident['n_seeds']})", flush=True)

    results: dict = {}
    for th in args.thetas:
        m = doc["marginals_by_theta"][th][args.rule]
        atoms = np.asarray(m["atom_fracs_by_k"], dtype=np.float64)
        n_anchors = int(m["n_anchors"])
        res = analyse_theta(atoms, n_anchors, n_trials)
        res["bimodality"] = shape_diagnosis(
            res["fit"], n_trials, n_anchors)
        f, pr = res["fit"], res["w0_profile"]
        print(f"    theta {th}: EM {f['iterations']:,}/{f['max_iter']:,} iters, "
              f"max|dw| {f['max_abs_dw']:.3e} -> {f['convergence_verdict']}")
        print(f"      atom residual {f['max_abs_atom_residual']:.3e}  "
              f"w0(point) {f['mass_at_p_eq_0']:.4f}  w1 {f['mass_at_p_eq_1']:.4f}")
        print(f"      w0 PROFILE: max at {pr['w0_at_max']:.3f}, "
              f"interval within {pr['drop_units']} loglik units "
              f"[{pr['w0_interval'][0]:.3f}, {pr['w0_interval'][1]:.3f}] "
              f"(width {pr['w0_interval_width']:.3f})")
        print(f"      thirds {[round(x, 4) for x in res['bimodality']['mass_in_thirds']]}")

        ps = m.get("per_seed")
        if ps is not None:
            se = np.asarray(ps["atom_se_over_seeds"], dtype=np.float64)
            naive = np.asarray(ps["atom_se_naive_pooled"], dtype=np.float64)
            res["error_bars"] = {
                "across_seed_se": se.tolist(),
                "naive_pooled_se": naive.tolist(),
                "median_ratio_across_seed_to_naive": float(
                    np.median(se[naive > 0] / naive[naive > 0])),
            }
            res["shape_predictions"] = score_shape_predictions(
                atoms, se, n_trials, res["sketch"]["u"])
        else:
            res["error_bars"] = {"status": "ABSENT -- rerun M1 for per-seed atoms"}
        results[th] = res
        b = res["bimodality"]
        print(f"[mix] theta={th}: shape={b['shape']} "
              f"interior_mass={b['interior_mass']:.4f} "
              f"mass@p=0 {res['fit']['mass_at_p_eq_0']:.4f} "
              f"mass@p=1 {res['fit']['mass_at_p_eq_1']:.4f} "
              f"resid {res['fit']['max_abs_atom_residual']:.2e}", flush=True)
        for key in ("k_le_1|p_eq_0", "k_le_1|p_lt_0.10",
                    "k_eq_0|p_eq_0", "k_ge_Nminus1|p_eq_1", "k_eq_N|p_eq_1",
                    "ag_primary_unstable|p_eq_0",
                    "ag_primary_unstable|p_lt_0.10",
                    "ag_primary_stable|p_eq_1",
                    "ag_primary_stable|p_gt_0.90"):
            if key not in res["purity"]:
                continue
            pu = res["purity"][key]
            print(f"        purity[{key}] = {pu['purity']:.4f} "
                  f"(group {pu['group_size']:.4f})", flush=True)

    out = {
        "identity": {
            "measurement": "rf_mixture_AB34",
            "source_m1": str(args.m1),
            "rule": args.rule,
            "n_trials": n_trials,
            "p_grid_size": int(P_GRID.size),
            "grid_includes_exact_0_and_1": True,
            "unstable_defs": UNSTABLE_DEFS,
            "stable_defs": STABLE_DEFS,
            "ag_endpoint_cuts": _ag_cuts_or_absent(n_trials),
        },
        "by_theta": results,
        "caveat": (
            "Whether the low group is a point mass at exactly p=0 or a cluster "
            "at small p>0 is NOT RESOLVABLE at N=19 -- the two predict nearly "
            "identical histograms. Both are reported; neither is adjudicated."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[mix] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
