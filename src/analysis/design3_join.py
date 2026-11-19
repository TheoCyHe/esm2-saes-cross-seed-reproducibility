"""THE DESIGN-3 JOIN. PREREGISTRATION amendment AR; specification record
`docs/SPECIFICATION.md` sections 0-4, ratified at AQ11 and dated before this.

*** THIS JOB DELIBERATELY CROSSES THE X7 FIREWALL. ***

X7 keeps `r_f` and alignment apart so `theta` cannot be selected on the
relationship between them. This is the crossing the whole design was built to
arrive at, and it is:

  * SCOPED   -- design 3's own inputs plus AK8's companion arm, nothing else;
  * LABELLED -- `firewall_crossing: DELIBERATE` in the output;
  * NON-AMBIENT -- its own module, its own SLURM job, its own output paths.

    logit(E[r_fs]) = b0 + b1 * z(log M_fs) + s(log10 phi_fs) + gamma * d_fs

Binomial, `n_compare` = 39 trials, logit link, PER-SEED fits meta-analysed
over the 20 anchor seeds. The estimator lives in `design3_fit.py` so it can be
tested against a known truth; this module is the I/O, the identity gates, the
arms and the report.

=============================================================================
PREDICTIONS -- COMMITTED TO GIT BEFORE THE JOB IS SUBMITTED (amendment AR)
=============================================================================
`git log` is the audit trail; the amendment is the record and this docstring
restates it so the module cannot be read without them. Every predicate is
TOTAL over the statistic's full domain (gotcha 24), and every outcome word in
the output is DERIVED from the data (gotcha 20).

  D1. The meta-analysed `b1_z` is STRICTLY GREATER THAN ZERO.
      CATEGORY: **directional mechanism (the 1/9 bin)**, filed there
      deliberately. Reasoning, stated so it can be wrong: if the features an
      SAE re-finds across seeds are the ones tracking real structure in the
      activations, alignment to curated biology should positively predict
      reappearance once firing rate is absorbed by the spline. That is a
      causal story about a pipeline, which is exactly what the 1/9 bin is
      for. It is filed rather than suppressed because suppressing a
      directional prediction on the grounds that the category has a bad
      record would make the record an artifact of the filing rule.
      TOTAL: `b1_z > 0` -> HIT; `b1_z <= 0` -> MISS. Zero counts as MISS.

  D2. The primary variance share `sd(b1*logM) / sd(s_hat(log10 phi))` is
      BELOW 1. CATEGORY: within-system comparison -- both quantities come
      from the same fit on our data and the prediction's content IS the
      comparison. Reasoning: firing rate has dominated `r_f` in every
      measurement so far, which is why the spline is mandatory rather than a
      nicety and why the matched design exists at all.
      TOTAL: `< 1` -> HIT; `>= 1` -> MISS.

  D3. The collinear share `joint - (dD_M + dD_phi)` is POSITIVE.
      CATEGORY: within-system comparison, and WEAK -- close to entailed by
      gotcha 5's measured fact that aligned features skew higher-firing.
      Filed so it is scored, not because it is informative. A negative share
      would be suppression and would be a finding about the covariates rather
      than about biology.
      TOTAL: positive -> HIT; zero or negative -> MISS (the sign is reported
      either way).

  D4. Seeds 0 and 1 are NOT the two most extreme seeds by `b1_z`.
      CATEGORY: distributional shape (the 12/13 bin). Reasoning: AO14/AQ
      established that their ~90-155 low-firing features with mean `r_f`
      0.93-0.96 do not reach the predictor on the argmax-selected normaliser
      -- the maximum `|rho|` falls 5.5x and the worst seed changes. D4 tests
      whether that survives into the coefficient, and it is the first
      quantity that could show the anomaly reaching `b1`.
      TOTAL: rank all 20 anchors by `b1_z`; MISS iff seeds 0 and 1 occupy
      BOTH extreme ranks (1st and 20th, in either order); HIT otherwise.

=============================================================================
WHAT IS REPORTED, AND WHY THE LIST IS NOT A SELECTION
=============================================================================
AK6 removed the confirmatory frame: no alpha, no directionality declaration,
no multiplicity correction. **Reporting the whole set rather than a selected
member is the declared substitute**, so the primary reported object is the
`b1` curve across the whole QUALIFYING theta range (AB1 decision A), with
theta = 0.70 as the reference for headline numbers. A tight interval around
zero IS evidence of absence and is stated as such, never as a failure to
detect.

=============================================================================
THREE THINGS THIS MODULE REPORTS AND DOES NOT DECIDE
=============================================================================
The session that produced it is authorised to run the join, not to settle a
specification question. Each of these is emitted under
`open_specification_items` so it cannot be lost:

  1. **WHICH META-ANALYTIC INTERVAL IS THE HEADLINE.** F1.3 and AL8a fix
     "per-seed fits meta-analysed" and fix the estimand; neither fixes the
     COMBINATION RULE. Both are computed -- inverse-variance fixed effect
     (which assumes independent per-seed fits) and the across-seed
     mean with `SD/sqrt(20)` (which carries the dependence empirically) --
     and AK9b's ratio between them is re-derived at N = 39 with the
     dependence structure stated. Neither is declared primary here.
  2. **HOW THE FEATURE-SIDE ARM'S TWO COVARIATES ENTER.** AL8b and
     SPECIFICATION §2 say `perplexity` and `concentration` "enter as
     covariates" without saying linearly or as smooths. They enter LINEARLY,
     matching `d_fs`, because that is the minimal addition; the choice is
     flagged rather than assumed.
  3. **THE SMOOTHING PARAMETER OF THE DISPERSION ARM.** It is INHERITED from
     the binomial REML fit rather than re-selected, so the beta-binomial arm
     is the same specification under a wider likelihood rather than a
     different specification. SPECIFICATION §0 makes the binomial primary,
     which is what licenses the inheritance.

=============================================================================
THE HAZARDS THIS MODULE IS BUILT AGAINST
=============================================================================
  * **`firing_rate_edges` round-trips as a JSON STRING.** `json.loads` it;
    `list()` of it yields CHARACTERS and has already produced plausible-looking
    wrong bins twice (gotcha 20).
  * **The smooth arm must consume the `_fixfp` artifacts.** The pre-AO6
    `smooth_mu_b{,_ident,_nodegen}.h5` and their gtildes are still on disk and
    were computed on a misaligned protein grouping. This module REFUSES a
    smooth gtilde whose path does not carry the `_fixfp` marker, rather than
    reading the nearest file (docs/CODEBASE.md §16e).
  * **Identity attrs on every input, and an ABSENT key is reported absent,
    never treated as matching** (I3, gotcha 15). `c1_argmax.check_shared_identity`
    is the existing helper and is reused rather than reimplemented.
  * **A feature index is not a cross-seed identity** (gotcha 18). Nothing here
    reduces over the seed axis on a feature-indexed array: every fit is
    per-seed and only the per-seed SUMMARIES are combined.
  * **THE TWO SIDES OF THIS JOIN ARE IN DIFFERENT SEED ORDERS, AND 18 OF THE
    20 ROWS DISAGREE.** `null_pilot_compliant.h5` and
    `feature_ap_main_compliant.h5` store LEXICOGRAPHIC labels
    (`0, 1, 10, 11, ..., 19, 2, ..., 9`) because `glob("seed_*")` sorts that
    way (CODEBASE §7.3); `rf_counts` and `d_fs` are NUMERIC; and the `gtilde`
    artifacts inherit the lexicographic order while storing **no labels at
    all**. Only rows 0 and 1 coincide. A positional join therefore mis-pairs
    18 seeds and produces a complete and entirely plausible wrong answer --
    gotcha 18's shape, on the seed axis rather than the feature axis. So every
    array is ALIGNED to the anchor order, and the two unlabelled axes are
    resolved by `resolve_gtilde_seed_order` FROM THE DATA (rank correlation of
    `gtilde_null` against `null_g_b`; measured margin 25x), with the
    degenerate flag's inheritance verified by ELEMENTWISE equality of
    `argmax_concept`. All of it is emitted under `seed_alignment`.
  * **Seed axes are asserted, not assumed** (CODEBASE §4.7).
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import h5py
import numpy as np

from src.analysis.artifact_identity import device_class
from src.analysis.c1_argmax import check_shared_identity
from src.analysis.design3_fit import (
    LAMBDA_GRID,
    SPLINE_K,
    average_marginal_effect,
    build_design,
    endpoint_check,
    fit_betabinomial,
    fit_design3,
    meta_analyse,
    standardisation_invariance,
    variance_share,
)
from src.analysis.null_pooling import bin_index
from src.analysis.ratio_rule import component as ratio_component
from src.analysis.ratio_rule import ratio_rule_diff
from src.analysis.seed_axis import (
    align_to_reference,
    normalise_seed_labels,
    seed_order as resolve_gtilde_seed_order,
)
from src.analysis.rf_marginal import THETA_LITERATURE, compare_counts_to_reference
from src.experiment_config import DEFAULT_CONFIG_PATH, load_config, lookup

# A4a's floor, inherited unchanged (AL8c, SPECIFICATION §4). MEASURED to
# exclude zero features in all 20 anchor seeds -- which is why the per-bin
# table below will report zeros, and why the table is printed anyway.
#
# The constant STAYS (ak8_geometry, fitted_vs_observed and the tests import
# it) and is ASSERTED equal to `confound_control.analysis_min_firing_rows`
# in the config before `main` opens any artifact (amendment CR, Row C;
# `repr_density.py` reads the key directly). A disagreement is a refusal
# printing both values, so the config cannot move without this file
# noticing.
ANALYSIS_MIN_FIRING_ROWS = 200


def assert_floor_matches_config(config_path=DEFAULT_CONFIG_PATH) -> int:
    """Refuse unless `confound_control.analysis_min_firing_rows` in the config
    equals ANALYSIS_MIN_FIRING_ROWS. Returns the (single) value. An absent key
    is refused by `lookup` naming the key; there is no default."""
    cfg = load_config(config_path)
    floor_cfg = int(lookup(cfg, "confound_control.analysis_min_firing_rows",
                           config_path))
    if floor_cfg != ANALYSIS_MIN_FIRING_ROWS:
        raise SystemExit(
            f"FATAL: confound_control.analysis_min_firing_rows = {floor_cfg} in "
            f"{config_path}, but design3_join.ANALYSIS_MIN_FIRING_ROWS = "
            f"{ANALYSIS_MIN_FIRING_ROWS}. The two must agree (amendment CR): "
            "the constant is imported by ak8_geometry and fitted_vs_observed, "
            "so change both for a declared reason or neither.")
    return floor_cfg

# AK8's committed cuts, from the amendment and not recomputed here.
AK8_MIN_K = 36                      # r_f >= 0.90 at n_compare = 39
AK8_LOW_M_QUANTILE = 0.10           # bottom decile WITHIN the firing bin

SHARED_IDENTITY_KEYS = ("library_sha", "n_concepts", "argmax_pool_n",
                        "truncation_policy", "scored_row_set")

# AMENDMENT AT §A1. THE SMOOTH ARM'S ADMISSIBLE GENERATIONS, AS A SET.
#
# The gate exists to refuse the PRE-AO6 smooth artifacts, which are still on
# disk and were built on a misaligned protein grouping (gotcha 27). It was a
# single literal, `_fixfp`. `_aligned` is post-AO6 AND post-AR9 -- the
# generation AS1 rebuilt under seed alignment -- and AT §A1 re-points AC10's
# arm at it as a DEFECT CORRECTION: the resolver aligns rows at join time and
# cannot repair `mu_s` values computed from mis-paired seeds INSIDE
# `smooth_mu`.
#
# It is a SET rather than a substring test on purpose. "any name that is not
# obviously old" is a gate that can only pass (gotcha 25c), so a new
# generation has to be added here deliberately, and the addition is what the
# amendment records.
SMOOTH_GENERATION_MARKERS = ("_fixfp", "_aligned")

# The bound AS14 published on the outcome scale under the BINOMIAL fit. AT §A2
# recomputes the same bound under the beta-binomial and compares against this
# number, so it is a named constant rather than a literal buried in a message.
AS14_BINOMIAL_BOUND_PP = 0.352


# ---------------------------------------------------------------------------
# Input loading, with the gates
# ---------------------------------------------------------------------------

def _json_default(o):
    """Preserve TYPE through serialisation.

    `default=float` would write a numpy bool as `1.0` and a numpy int as a
    float -- a silent type change in the artifact of record, and exactly the
    kind of plausible-looking wrong value this project keeps catching late.
    """
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON-serialisable: {type(o).__name__}")


def _attrs(path: Path) -> dict:
    with h5py.File(path, "r") as hf:
        return {k: (v.decode() if isinstance(v, bytes) else v)
                for k, v in hf.attrs.items()}


def firing_rate_edges(gtilde: Path) -> list[float]:
    """`json.loads`, never `list()` -- gotcha 20's third instance."""
    v = _attrs(gtilde)["firing_rate_edges"]
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, str):
        return [float(x) for x in json.loads(v)]
    return [float(x) for x in np.asarray(v).ravel()]


def log_M(M: np.ndarray, source: str) -> np.ndarray:
    """`log M`, WITHOUT a floor, and with the non-positive set checked first.

    A `np.maximum(M, 1e-300)` clamp would turn `M = 0` into a finite
    `log M = -690.8` and the emptiness check downstream could then only pass --
    gotcha 25(a), found by running this module's own gate against a fixture
    that plants `M = 0`. SPECIFICATION §4 records the non-finite / non-positive
    set as EMPTY BY MEASUREMENT (AL9a/AL12, `M` in [1.5468e-05, 1.0460e+06]),
    so a non-empty set here is a CHANGED INPUT, not a case to absorb.
    """
    bad = ~np.isfinite(M) | (M <= 0)
    if bad.any():
        raise SystemExit(
            f"FATAL: {int(bad.sum())} of {M.size} cells in {source} have "
            f"non-finite or non-positive M, so log M is undefined for them. "
            f"AL9a/AL12 measured this set as EMPTY across all 20 anchors and "
            f"SPECIFICATION §4 records that NO BRANCH is written for it. "
            f"Refusing to floor them into a finite value -- that would make "
            f"the emptiness check unable to fire.")
    return np.log(M)


def load_counts_all_theta(path: Path) -> dict:
    """Every theta's per-anchor counts, keyed by theta."""
    out, meta = {}, {}
    with h5py.File(path, "r") as hf:
        at = {k: (v.decode() if isinstance(v, bytes) else v)
              for k, v in hf.attrs.items()}
        for name in hf:
            d = hf[name]
            out[round(float(d.attrs["theta"]), 2)] = d[:]
        meta = {
            "n_compare_seeds": int(at["n_compare_seeds"]),
            "similarity": str(at["similarity"]),
            "rule": str(at["rule"]),
            "seed_labels": json.loads(at["seed_labels"]),
            "theta_grid": json.loads(at["theta_grid"]),
            "git_sha": str(at.get("git_sha")),
        }
    return {"counts": out, "meta": meta}


def qualifying_thetas(m1_json: Path) -> dict:
    """The QUALIFYING theta set, read from M1's own pre-declared rule.

    X2's two-part rule (clears the rotation-null floor; neither endpoint bin
    exceeds 80%) is evaluated in `rf_marginal.select_theta` and stored. Reading
    it here rather than recomputing it means this module cannot widen or narrow
    the range it reports over -- which is the point of AB1 decision A.
    """
    doc = json.loads(m1_json.read_text())
    sel = doc["theta_selection"]
    rows = [r for r in sel["grid"] if r["qualifies"]]
    return {
        "qualifying": [round(float(r["theta"]), 2) for r in rows],
        "selected_theta": sel["selected_theta"],
        "floor": sel["floor"],
        "endpoint_max_share": sel["endpoint_max_share"],
        "n_compare_seeds_in_m1": int(
            doc["identity"]["n_compare_seeds"]),
        "grid": sel["grid"],
    }


# ---------------------------------------------------------------------------
# The analysis set
# ---------------------------------------------------------------------------

def floor_report(fire_rate: np.ndarray, n_rows: int, edges: list[float],
                 logM: np.ndarray, d_fs: np.ndarray,
                 retained: np.ndarray | None) -> dict:
    """A4a's floor applied, with its mandated per-bin excluded counts.

    THE PER-BIN TABLE IS REPORTED EVEN THOUGH IT IS ALL ZEROS. A4a's
    reporting requirement carries over unchanged (AL8c) and the floor's domain
    is non-empty (AO16); the record states the floor's MEASURED effect rather
    than letting a reader assume it removed something. `n_rows` is the
    analysis population's row count, so the floor is a firing-RATE threshold
    of `200 / n_rows`.
    """
    thresh = ANALYSIS_MIN_FIRING_ROWS / float(n_rows)
    n_bins = len(edges) - 1
    keep = fire_rate >= thresh
    rows, fbins = [], []
    for s in range(fire_rate.shape[0]):
        fb = np.clip(bin_index(fire_rate[s], edges), 0, n_bins - 1)
        fbins.append(fb)
        excl = ~keep[s]
        rows.append({
            "seed": s,
            "n_features": int(fire_rate.shape[1]),
            "n_excluded_by_floor": int(excl.sum()),
            "n_excluded_per_firing_bin":
                np.bincount(fb[excl], minlength=n_bins).tolist(),
            "n_retained_per_firing_bin":
                np.bincount(fb[keep[s]], minlength=n_bins).tolist(),
            "n_zero_firing": int((fire_rate[s] == 0).sum()),
        })
    agreement = None
    if retained is not None:
        agreement = {
            "d_fs_retained_at_floor_agrees": bool(
                np.array_equal(retained, keep)),
            "n_cells_disagreeing": int((retained != keep).sum()),
        }
    return {
        "analysis_min_firing_rows": ANALYSIS_MIN_FIRING_ROWS,
        "n_rows": int(n_rows),
        "firing_rate_threshold": thresh,
        "firing_rate_edges": edges,
        "per_seed": rows,
        "total_excluded": int((~keep).sum()),
        "d_fs_cross_check": agreement,
        # AL9a / AL12: EMPTY SETS, verified -- not rules that fired zero
        # times. `n_nonfinite_logM` is zero BY CONSTRUCTION, because `log_M`
        # refuses a non-positive M before this runs; it is reported so the
        # record carries the number rather than the reader inferring it.
        "n_nonfinite_logM": int((~np.isfinite(logM)).sum()),
        "n_nonfinite_d_fs_among_retained": int(
            (~np.isfinite(d_fs) & keep).sum()),
        "logM_range": [float(np.nanmin(logM)), float(np.nanmax(logM))],
        "keep": keep, "firing_bin": np.stack(fbins),
    }


# ---------------------------------------------------------------------------
# One arm: per-seed fits + meta-analysis
# ---------------------------------------------------------------------------

def _spread(x: np.ndarray) -> dict:
    """min / median / max of a per-seed quantity, with the values kept.

    AL8a's rule for `SD(log M)` -- the spread across the 20 anchors must be
    VISIBLE rather than absorbed by a meta-analysis -- applied to every
    per-seed summary AT §A4 adds for `d_fs`.
    """
    x = np.asarray(x, dtype=float)
    return {"values": x.tolist(), "median": float(np.median(x)),
            "min": float(x.min()), "max": float(x.max()),
            "mean": float(x.mean()),
            "max_over_min": (float(x.max() / x.min())
                             if x.min() > 0 else float("inf"))}


def _pp(meta: dict) -> dict:
    """A meta-analysed marginal effect, converted to percentage points."""
    return {
        "fixed_effect": 100.0 * meta["fixed_effect"],
        "fixed_effect_ci95": [100.0 * v for v in meta["fixed_effect_ci95"]],
        "across_seed_mean": 100.0 * meta["across_seed_mean"],
        "across_seed_ci95": [100.0 * v for v in meta["across_seed_ci95"]],
    }


def fit_arm(y: np.ndarray, logM: np.ndarray, log_phi: np.ndarray,
            d_fs: np.ndarray, keep: np.ndarray, n_trials: int,
            extra: dict[str, np.ndarray] | None = None,
            deviance_drops: bool = False, dispersion: bool = False,
            label: str = "", z_divisor=None) -> dict:
    """Per-seed fits. NEVER a pooled fit over feature-seed rows (F1.3).

    `z_divisor` (amendment CT §CT28-F item 2, default None = unchanged): the
    centred `log M` is divided by this instead of the seed's own SD -- a
    scalar, or one value per seed -- so a fit on another scale can be run
    through the same path. `b1_z` and the marginal effects are then per unit
    of `log M / z_divisor`; `sd_logM` still records the seed's own SD."""
    n_seeds = y.shape[0]
    per_seed = []
    for s in range(n_seeds):
        m = keep[s]
        ys = y[s][m].astype(np.float64)
        lm = logM[s][m]
        sd = float(lm.std())
        if z_divisor is None:
            div = sd
        else:
            div = float(np.asarray(z_divisor, dtype=float).reshape(-1)[s]
                        if np.ndim(z_divisor) else z_divisor)
        z = (lm - lm.mean()) / max(div, 1e-300)
        ex = ({k: v[s][m] for k, v in extra.items()} if extra else None)
        des = build_design(z, log_phi[s][m], d_fs[s][m], extra=ex)
        f = fit_design3(ys, n_trials, des["X"], des["S"])
        i = des["idx_logM"]
        b1 = float(f["beta"][i])
        se = float(np.sqrt(f["cov"][i, i]))
        row = {
            "seed": s, "n_features": int(m.sum()),
            "b1_z": b1, "se_b1_z": se,
            "b1_raw": b1 / max(div, 1e-300),
            "sd_logM": sd, "mean_logM": float(lm.mean()),
            "sd_d_fs": float(d_fs[s][m].std()),
            "gamma_d_fs": float(f["beta"][des["names"].index("d_fs")]),
            "se_gamma": float(np.sqrt(
                f["cov"][des["names"].index("d_fs"),
                         des["names"].index("d_fs")])),
            "intercept": float(f["beta"][0]),
            "lambda": f["lambda"], "edf": f["edf"],
            "lambda_at_grid_edge": f["lambda_at_grid_edge"],
            "deviance": f["deviance"], "loglik": f["loglik"],
            "converged": f["converged"], "n_iter": f["n_iter"],
            "variance_share": variance_share(f, des),
            # AS section 3: the OUTCOME scale, alongside and never instead
            # of `b1_z`. Not outcome-blind, adopted after the result was
            # seen, and the object says so itself.
            "marginal_effect": average_marginal_effect(
                f, des, se, n_trials=n_trials),
            "marginal_effect_d_fs": average_marginal_effect(
                f, des, float(np.sqrt(
                    f["cov"][des["names"].index("d_fs"),
                             des["names"].index("d_fs")])),
                term="d_fs", n_trials=n_trials),
            "mean_r_f": float(ys.mean() / n_trials),
        }
        if z_divisor is not None:
            row["z_divisor"] = div      # only when set: every existing output unchanged
        if extra:
            row["extra_coefs"] = {k: float(f["beta"][des["names"].index(k)])
                                  for k in extra}
            # AMENDMENT CD (CD-e): an extra term is reported WITH ITS
            # INTERVAL. A point estimate alone cannot answer the one question
            # CD-e fixes in advance -- whether the term's interval covers
            # zero -- and that outcome is declared reportable before the fit.
            # ADDITIVE and inert for every existing caller: the primary and
            # all sensitivity arms pass `extra=None`, so this branch does not
            # execute for them and no stored number can move.
            row["extra_se"] = {
                k: float(np.sqrt(abs(f["cov"][des["names"].index(k),
                                              des["names"].index(k)])))
                for k in extra}
        if dispersion:
            # AS section 2: the dispersion-corrected fit, from the
            # beta-binomial DIRECTLY -- never by scaling the binomial
            # interval by a summary multiplier, which is the arithmetic the
            # brief flagged as a hypothesis rather than a result.
            bb = fit_betabinomial(ys, n_trials, des["X"],
                                  f["lambda"] * des["S"], f["beta"])
            se_bb = float(np.sqrt(abs(bb["cov"][i, i])))
            # AMENDMENT AT §A4: `gamma` gets EVERYTHING `b1` got, and the
            # dispersion-corrected fit is one of the three interval
            # treatments AT §A2 applies to `b1`. Emitting it for `b1` only
            # would characterise the negligible term thoroughly and the
            # material one thinly, which is the writeup A4 exists to prevent.
            jd = des["names"].index("d_fs")
            se_g_bb = float(np.sqrt(abs(bb["cov"][jd, jd])))
            bb_fit = {"beta": bb["beta"], "eta": des["X"] @ bb["beta"]}
            row["betabinomial"] = {
                "b1_z": float(bb["beta"][i]), "se_b1_z": se_bb,
                "gamma_d_fs": float(bb["beta"][jd]), "se_gamma": se_g_bb,
                "rho": bb["rho"], "s": bb["s"],
                "loglik": bb["loglik"], "loglik_gain": bb["loglik"] - f["loglik"],
                "se_inflation_over_binomial": se_bb / max(se, 1e-300),
                "se_inflation_over_binomial_gamma": se_g_bb / max(
                    float(np.sqrt(f["cov"][jd, jd])), 1e-300),
                "optimiser_success": bb["success"],
                "marginal_effect": average_marginal_effect(
                    bb_fit, des, se_bb, n_trials=n_trials),
                "marginal_effect_d_fs": average_marginal_effect(
                    bb_fit, des, se_g_bb, term="d_fs", n_trials=n_trials),
                "variance_share": variance_share(bb_fit, des),
                "AK9a": endpoint_check(ys, n_trials, des["X"] @ bb["beta"],
                                       s=bb["s"]),
            }
            if extra:
                # The same treatment under the dispersion-corrected fit, so
                # CD-e's "both likelihoods" is answerable for the interaction
                # term and not only for `b1` (amendment AT §A4's rule, applied
                # to the term CD-e adds).
                row["betabinomial"]["extra_coefs"] = {
                    k: float(bb["beta"][des["names"].index(k)]) for k in extra}
                row["betabinomial"]["extra_se"] = {
                    k: float(np.sqrt(abs(bb["cov"][des["names"].index(k),
                                                   des["names"].index(k)])))
                    for k in extra}
        if deviance_drops:
            drops = {}
            for nm, kw in (("drop_M", {"include_M": False}),
                           ("drop_phi", {"include_spline": False}),
                           ("drop_both", {"include_M": False,
                                          "include_spline": False})):
                dd = build_design(z, log_phi[s][m], d_fs[s][m], extra=ex, **kw)
                ff = fit_design3(ys, n_trials, dd["X"], dd["S"])
                drops[nm] = {"deviance": ff["deviance"],
                             "delta": ff["deviance"] - f["deviance"],
                             "edf": ff["edf"]}
            joint = drops["drop_both"]["delta"]
            collinear = joint - (drops["drop_M"]["delta"]
                                 + drops["drop_phi"]["delta"])
            row["deviance_drops"] = {
                **drops,
                "delta_D_M": drops["drop_M"]["delta"],
                "delta_D_phi": drops["drop_phi"]["delta"],
                "delta_D_joint": joint,
                "collinear_share": collinear,
                # DERIVED, never asserted (gotcha 20).
                "collinear_share_sign": ("POSITIVE" if collinear > 0 else
                                         "ZERO" if collinear == 0 else
                                         "NEGATIVE"),
            }
        per_seed.append(row)
        if label:
            print(f"  [{label}] seed {s:2d}: b1_z {b1:+.5f} +- {se:.5f} | "
                  f"lam {f['lambda']:.3g} edf {f['edf']:.2f} | "
                  f"var-share {row['variance_share']['ratio']:.4f} | "
                  f"mean r_f {row['mean_r_f']:.4f}", flush=True)

    est = np.array([r["b1_z"] for r in per_seed])
    se = np.array([r["se_b1_z"] for r in per_seed])
    meta = meta_analyse(est, se)
    sds = np.array([r["sd_logM"] for r in per_seed])
    shares = np.array([r["variance_share"]["ratio"] for r in per_seed])
    ames = np.array([r["marginal_effect"]["ame"] for r in per_seed])
    ame_ses = np.array([r["marginal_effect"]["se"] for r in per_seed])
    dratio = np.array([r["variance_share"]["ratio_d_fs_over_logM"]
                       for r in per_seed])
    dshare = np.array([r["variance_share"]["ratio_d_fs_over_spline"]
                       for r in per_seed])
    out = {
        "per_seed": per_seed,
        "meta_b1_z": meta,
        # AS section 3, on the outcome scale, in percentage points per SD.
        "meta_marginal_effect": meta_analyse(ames, ame_ses),
        "marginal_effect_percentage_points": {
            "fixed_effect": 100.0 * float(
                meta_analyse(ames, ame_ses)["fixed_effect"]),
            "fixed_effect_ci95": [
                100.0 * x for x in
                meta_analyse(ames, ame_ses)["fixed_effect_ci95"]],
            "across_seed_ci95": [
                100.0 * x for x in
                meta_analyse(ames, ame_ses)["across_seed_ci95"]],
            "units": "percentage points of r_f per SD of log M",
        },
        # AS section 4: `d_fs` on the estimand's OWN instrument.
        "d_fs_variance_share": {
            "ratio_d_fs_over_spline": {
                "median": float(np.median(dshare)),
                "min": float(dshare.min()), "max": float(dshare.max())},
            "ratio_d_fs_over_logM": {
                "median": float(np.median(dratio)),
                "min": float(dratio.min()), "max": float(dratio.max())},
            "why": ("gamma / b1 is a UNIT ARTIFACT -- b1 is on a within-seed z "
                    "scale and gamma on d_fs's raw scale (gotcha 21). Both are "
                    "put through AL8a's own linear-predictor SD ratio so the "
                    "comparison carries no units."),
        },
        "meta_gamma_d_fs": meta_analyse(
            np.array([r["gamma_d_fs"] for r in per_seed]),
            np.array([r["se_gamma"] for r in per_seed])),
        # AL8a: raw must be RECOVERABLE, and the SD's spread across the 20
        # anchors must be VISIBLE rather than absorbed by the meta-analysis.
        "sd_logM_across_seeds": {
            "values": sds.tolist(), "mean": float(sds.mean()),
            "sd": float(sds.std(ddof=1)), "min": float(sds.min()),
            "max": float(sds.max()),
            "max_over_min": float(sds.max() / max(sds.min(), 1e-300)),
            "cv": float(sds.std(ddof=1) / max(sds.mean(), 1e-300)),
            "note": ("standardised coefficients from seeds with different SDs "
                     "are on slightly different scales BEFORE meta-analysis; "
                     "b1_raw = b1_z / SD recovers the unstandardised value"),
        },
        "sd_d_fs_across_seeds": {
            "values": [float(r["sd_d_fs"]) for r in per_seed],
            "note": ("reported per seed and as a spread, exactly as AL8a "
                     "requires for SD(log M), so gamma_raw is recoverable and "
                     "the heterogeneity is visible rather than absorbed"),
        },
        "variance_share_across_seeds": {
            "values": shares.tolist(),
            "median": float(np.median(shares)),
            "min": float(shares.min()), "max": float(shares.max()),
        },
        "n_seeds_lambda_at_grid_edge": int(
            sum(r["lambda_at_grid_edge"] for r in per_seed)),
        "n_seeds_not_converged": int(
            sum(not r["converged"] for r in per_seed)),
    }

    # ---- AMENDMENT AT §A4: `d_fs` ON EVERY INSTRUMENT `b1` IS ON ----------
    # `gamma` per seed and meta-analysed already exist above. What A4 adds is
    # the OUTCOME-SCALE conversion, the three-way ordering as an ordering, and
    # (where the dispersion arm runs) the same three interval treatments.
    sd_d = np.array([r["sd_d_fs"] for r in per_seed])
    g_ame = np.array([r["marginal_effect_d_fs"]["ame"] for r in per_seed])
    g_ame_se = np.array([r["marginal_effect_d_fs"]["se"] for r in per_seed])
    meta_g_ame = meta_analyse(g_ame, g_ame_se)
    # PER SD, which is the only form comparable with `log M`'s: `b1` is
    # already on a within-seed z scale, so its AME is per SD by construction,
    # while `gamma` is on `d_fs`'s RAW scale. SD(d_fs) ~ 1.24, so the two
    # conversions differ by ~24% -- small enough to look like rounding and
    # large enough to matter, which is why both are emitted and named.
    meta_g_ame_sd = meta_analyse(g_ame * sd_d, g_ame_se * sd_d)
    out["meta_marginal_effect_d_fs"] = meta_g_ame
    out["marginal_effect_d_fs_percentage_points"] = {
        "per_raw_unit_of_d_fs": _pp(meta_g_ame),
        "per_SD_of_d_fs": _pp(meta_g_ame_sd),
        "sd_d_fs_used_per_seed": sd_d.tolist(),
        "units": "percentage points of r_f",
        "which_to_quote": (
            "per_SD_of_d_fs is the like-for-like companion to the log M "
            "figure, because b1 is on a within-seed z scale and its AME is "
            "therefore already per SD. per_raw_unit_of_d_fs is what gamma "
            "itself multiplies and is kept so the raw coefficient stays "
            "recoverable (AL8a)."),
        **{k: v for k, v in per_seed[0]["marginal_effect_d_fs"].items()
           if k in ("outcome_blind", "honesty_note", "se_note")},
    }
    # THE ORDERING, AS AN ORDERING. Three additive contributions to one linear
    # predictor, all in logits, all measured by AL8a's own SD ratio. Reported
    # as a ranked table so "d_fs outweighs alignment" cannot travel without
    # "firing rate outweighs d_fs" (AT §A4).
    r_ms = np.array([r["variance_share"]["ratio"] for r in per_seed])
    r_ds = np.array([r["variance_share"]["ratio_d_fs_over_spline"]
                     for r in per_seed])
    r_dm = np.array([r["variance_share"]["ratio_d_fs_over_logM"]
                     for r in per_seed])
    # BW-A2b: `per_seed` here is ONE ARM's 20 anchors, which is why this is
    # correct. A later reduction that reopens `design3_join_ar.json` and globs
    # `sd_s_hat_log10phi` finds 80 sites, not 20 -- the key is stored per seed
    # PER ARM (primary, dispersion_arm, three sensitivity arms) -- and pooling
    # them returns a plausible, close, wrong number. Count against n_anchors
    # before reducing. See docs/CODEBASE.md §17d.
    sd_sp = np.array([r["variance_share"]["sd_s_hat_log10phi"]
                      for r in per_seed])
    sd_m = np.array([r["variance_share"]["sd_b1_logM"] for r in per_seed])
    sd_g = np.array([r["variance_share"]["sd_gamma_d_fs"] for r in per_seed])
    terms = [("s_hat(log10 phi)", sd_sp), ("gamma * d_fs", sd_g),
             ("b1 * z(log M)", sd_m)]
    ranked = sorted(terms, key=lambda t: -float(np.median(t[1])))
    out["term_ordering"] = {
        "statistic": ("SD of each additive contribution to the linear "
                      "predictor, median over the anchors -- AL8a's own "
                      "instrument, extended to all three terms"),
        "ranked_terms": [t[0] for t in ranked],
        "sd_of_contribution": {t[0]: _spread(t[1]) for t in terms},
        "ratios": {
            "logM_over_spline": _spread(r_ms),
            "d_fs_over_spline": _spread(r_ds),
            "d_fs_over_logM": _spread(r_dm),
            "spline_over_d_fs_from_per_seed_reciprocal": _spread(1.0 / r_ds),
        },
        # DERIVED, never asserted (gotcha 20).
        "ordering_is_spline_then_d_fs_then_logM": bool(
            [t[0] for t in ranked]
            == ["s_hat(log10 phi)", "gamma * d_fs", "b1 * z(log M)"]),
        "n_seeds_with_d_fs_above_logM": int((r_dm > 1.0).sum()),
        "n_seeds_with_d_fs_above_spline": int((r_ds > 1.0).sum()),
        "reciprocal_warning": (
            "median(1/x) != 1/median(x). Both are reported: "
            "1/median(d_fs_over_spline) and the median of the per-seed "
            "reciprocal. Quoting one as the other is gotcha 21."),
        "one_over_median_d_fs_over_spline": float(
            1.0 / np.median(r_ds)) if np.median(r_ds) > 0 else float("inf"),
    }
    if dispersion:
        bb_g = np.array([r["betabinomial"]["gamma_d_fs"] for r in per_seed])
        bb_gs = np.array([r["betabinomial"]["se_gamma"] for r in per_seed])
        bb_e2 = np.array([r["betabinomial"]["b1_z"] for r in per_seed])
        bb_s2 = np.array([r["betabinomial"]["se_b1_z"] for r in per_seed])
        bb_a = np.array([r["betabinomial"]["marginal_effect"]["ame"]
                         for r in per_seed])
        bb_as = np.array([r["betabinomial"]["marginal_effect"]["se"]
                          for r in per_seed])
        bb_ga = np.array([r["betabinomial"]["marginal_effect_d_fs"]["ame"]
                          for r in per_seed])
        bb_gas = np.array([r["betabinomial"]["marginal_effect_d_fs"]["se"]
                           for r in per_seed])
        out["meta_b1_z_betabinomial"] = meta_analyse(bb_e2, bb_s2)
        out["meta_gamma_d_fs_betabinomial"] = meta_analyse(bb_g, bb_gs)
        out["meta_marginal_effect_betabinomial"] = meta_analyse(bb_a, bb_as)
        out["marginal_effect_percentage_points_betabinomial"] = _pp(
            meta_analyse(bb_a, bb_as))
        out["marginal_effect_d_fs_percentage_points_betabinomial"] = {
            "per_raw_unit_of_d_fs": _pp(meta_analyse(bb_ga, bb_gas)),
            "per_SD_of_d_fs": _pp(meta_analyse(bb_ga * sd_d, bb_gas * sd_d)),
        }
        out["term_ordering_betabinomial"] = {
            "ratios": {
                "logM_over_spline": _spread(np.array(
                    [r["betabinomial"]["variance_share"]["ratio"]
                     for r in per_seed])),
                "d_fs_over_spline": _spread(np.array(
                    [r["betabinomial"]["variance_share"]
                     ["ratio_d_fs_over_spline"] for r in per_seed])),
                "d_fs_over_logM": _spread(np.array(
                    [r["betabinomial"]["variance_share"]
                     ["ratio_d_fs_over_logM"] for r in per_seed])),
            },
        }
    if deviance_drops:
        cs = np.array([r["deviance_drops"]["collinear_share"]
                       for r in per_seed])
        out["collinear_share_across_seeds"] = {
            "values": cs.tolist(), "median": float(np.median(cs)),
            "min": float(cs.min()), "max": float(cs.max()),
            "n_positive": int((cs > 0).sum()), "n_negative": int((cs < 0).sum()),
            "n_zero": int((cs == 0).sum()),
        }
        out["deviance_drops_across_seeds"] = {
            k: {"median": float(np.median(
                [r["deviance_drops"][k] for r in per_seed]))}
            for k in ("delta_D_M", "delta_D_phi", "delta_D_joint")}
    return out


# ---------------------------------------------------------------------------
# AK8's companion arm
# ---------------------------------------------------------------------------

def ak8_discovery_mask(y: np.ndarray, M: np.ndarray, firing_bin: np.ndarray,
                       keep: np.ndarray) -> np.ndarray:
    """The AK8 SELECTION alone, as a (n_seeds, n_features) boolean mask.

    FACTORED OUT AT CH2 SO THERE IS ONE IMPLEMENTATION OF THE SELECTION AND NOT
    TWO. `ak8_discovery_set` (the distributional half) and
    `ak8_geometry` (the geometric half, amendment CH2) both call this; a second
    hand-written copy of the within-bin decile cut is exactly the divergence
    gotcha 16 exists for.

    A SEPARATE FUNCTION rather than a wider return from `ak8_discovery_set` --
    gotcha 27's convention: when a helper needs to expose an internal object,
    add a function, do not grow a return arity.

    "Low `M`" is the bottom decile WITHIN the feature's own firing-rate bin.
    Within-bin because firing rate is the confound the entire design controls;
    an unconditional low-`M` cut would select low-firing features and
    rediscover that confound instead of the biology.
    """
    n_seeds = y.shape[0]
    n_bins = int(firing_bin.max()) + 1
    out = np.zeros_like(keep, dtype=bool)
    for s in range(n_seeds):
        m = keep[s]
        high = (y[s] >= AK8_MIN_K) & m
        low = np.zeros_like(m)
        for b in range(n_bins):
            sel = m & (firing_bin[s] == b)
            if sel.sum() < 10:
                continue
            cut = float(np.quantile(M[s][sel], AK8_LOW_M_QUANTILE))
            low |= sel & (M[s] <= cut)
        out[s] = high & low
    return out


def ak8_discovery_set(y: np.ndarray, M: np.ndarray, fire_rate: np.ndarray,
                      firing_bin: np.ndarray, keep: np.ndarray,
                      n_trials: int) -> dict:
    """High `r_f`, low `M` -- defined in AK8 BEFORE the join.

    The selection itself lives in `ak8_discovery_mask`; this function is the
    DISTRIBUTIONAL half's reporting around it.

    The GEOMETRIC half needed M2 and was EXPLICITLY DEFERRED, not quietly
    dropped -- which is why `geometric_properties` is a stated status rather
    than an absent key. **M2 ran at CF3c and the geometric half ran at CH2**;
    the status string below records where it went.
    """
    n_seeds = y.shape[0]
    n_bins = int(firing_bin.max()) + 1
    disc_all = ak8_discovery_mask(y, M, firing_bin, keep)
    per_seed = []
    for s in range(n_seeds):
        m = keep[s]
        high = (y[s] >= AK8_MIN_K) & m
        low = np.zeros_like(m)
        for b in range(n_bins):
            sel = m & (firing_bin[s] == b)
            if sel.sum() < 10:
                continue
            cut = float(np.quantile(M[s][sel], AK8_LOW_M_QUANTILE))
            low |= sel & (M[s] <= cut)
        disc = disc_all[s]
        rest = m & ~disc
        lf = np.log10(np.maximum(fire_rate[s], 1e-12))
        per_seed.append({
            "seed": s,
            "n_high_r_f": int(high.sum()),
            "n_low_M": int(low.sum()),
            "n_discovery": int(disc.sum()),
            "frac_of_features": float(disc.sum() / max(m.sum(), 1)),
            "log10_fire_rate_discovery": _quantiles(lf[disc]),
            "log10_fire_rate_rest": _quantiles(lf[rest]),
            "firing_bin_counts_discovery":
                np.bincount(firing_bin[s][disc], minlength=n_bins).tolist(),
            "firing_bin_counts_rest":
                np.bincount(firing_bin[s][rest], minlength=n_bins).tolist(),
            "mean_r_f_discovery": float(y[s][disc].mean() / n_trials)
                                  if disc.any() else float("nan"),
        })
    counts = np.array([r["n_discovery"] for r in per_seed])
    return {
        "definition": {
            "high_r_f": f"k >= {AK8_MIN_K} of {n_trials} (r_f >= 0.90, above "
                        f"AG1's stable cut, so a SUBSET of the pre-declared "
                        f"stable group)",
            "low_M": f"M in the bottom {AK8_LOW_M_QUANTILE:.0%} WITHIN the "
                     f"feature's own firing-rate bin",
            "committed_in": "amendment AK8, before the join",
        },
        "per_seed": per_seed,
        "count_across_seeds": {
            "values": counts.tolist(), "median": float(np.median(counts)),
            "min": int(counts.min()), "max": int(counts.max()),
            "mean": float(counts.mean()),
        },
        "geometric_properties": (
            "RAN AT AMENDMENT CH2, in `src/analysis/ak8_geometry.py`, against "
            "size-matched nulls at this set's own per-seed size. It was "
            "DEFERRED (not dropped) from AK8 until M2 existed; M2 ran at CF3c. "
            "This key previously read DEFERRED and is updated rather than "
            "removed, because its whole point was to be a stated status."),
    }


def interval_agreement(a: list, b: list, label_a: str, label_b: str) -> dict:
    """Do two intervals agree? AS §2's E3, with its scale stated.

    "Agree within 10%" needs an operationalisation here and the brief could
    not supply one: the LOWER endpoint sits at ~5e-5, so a relative comparison
    against the endpoint itself is undefined in the limit and would read as a
    100%+ disagreement for a gap of 9e-5. The scale used is therefore the
    INTERVAL WIDTH, which does not vanish -- `|lo_a - lo_b| / width` and
    `|hi_a - hi_b| / width`. Declared in amendment AS before the run.
    """
    wa, wb = a[1] - a[0], b[1] - b[0]
    w = max(wa, wb)
    lo = abs(a[0] - b[0]) / w if w > 0 else float("inf")
    hi = abs(a[1] - b[1]) / w if w > 0 else float("inf")
    return {
        "interval_a": {"label": label_a, "lo": a[0], "hi": a[1], "width": wa},
        "interval_b": {"label": label_b, "lo": b[0], "hi": b[1], "width": wb},
        "scale": "max interval width",
        "lo_gap_over_width": lo, "hi_gap_over_width": hi,
        "lo_within_10pct": bool(lo < 0.10),
        "hi_within_10pct": bool(hi < 0.10),
        "both_within_10pct": bool(lo < 0.10 and hi < 0.10),
    }


def model_free_panel(y: np.ndarray, logM: np.ndarray, fire_rate: np.ndarray,
                     keep: np.ndarray, n_trials: int,
                     strata_edges: list[float], n_deciles: int = 10,
                     comparison_pp: float = AS14_BINOMIAL_BOUND_PP) -> dict:
    """F1's LEFT PANEL, pre-declared and never run: mean `r_f` by predictor
    decile, ONE LINE PER FIRING-RATE STRATUM, entirely model-free.

    F1: "If the left panel is monotone within every stratum, the finding does
    not depend on the model and the regression is buying precision, not the
    qualitative result -- this is reported explicitly either way, not only if
    it comes out favorably."

    TRANSLATION, and it is LICENSED rather than improvised. F1 wrote the
    deciles on `g~`; the predictor of record is now `M` (AC9). Per AL7's
    committed precedent a prediction about a quantity that is a deterministic
    function of a retained quantity is TRANSLATED and SCORED, not voided. So:
    deciles of `log M`, strata from `corrector_scan.BIN_EDGES`.

    THE MONOTONICITY VERDICT HAS THREE OUTCOMES, because a two-valued verdict
    on an ordered comparison is not total (gotcha 24) -- and a fourth,
    `FLAT`, because a stratum whose deciles differ by less than their own CI
    width is not "non-monotone", it is uninformative about direction, and
    calling it non-monotone would read as a failure of the instrument.
    """
    n_seeds = y.shape[0]
    n_strata = len(strata_edges) - 1
    per_seed = []
    for s in range(n_seeds):
        m = keep[s]
        fb = np.clip(bin_index(fire_rate[s], strata_edges), 0, n_strata - 1)
        strata = []
        for b in range(n_strata):
            sel = m & (fb == b)
            if sel.sum() < 10 * n_deciles:
                strata.append({"stratum": b, "n_features": int(sel.sum()),
                               "status": "TOO_FEW_FEATURES"})
                continue
            lm, kk = logM[s][sel], y[s][sel]
            cuts = np.quantile(lm, np.linspace(0, 1, n_deciles + 1)[1:-1])
            dec = np.digitize(lm, cuts)
            rows = []
            for d in range(n_deciles):
                dm = dec == d
                if not dm.any():
                    rows.append({"decile": d, "n": 0})
                    continue
                succ = float(kk[dm].sum())
                tot = float(dm.sum() * n_trials)
                p = succ / tot
                # Wilson, and it is the NOMINAL binomial interval: features
                # within a seed share a dictionary and are not independent
                # trials, so this understates the true width. Stated, not fixed.
                z = 1.959964
                den = 1 + z * z / tot
                ctr = (p + z * z / (2 * tot)) / den
                half = z * np.sqrt(p * (1 - p) / tot
                                   + z * z / (4 * tot * tot)) / den
                rows.append({"decile": d, "n": int(dm.sum()),
                             "mean_logM": float(lm[dm].mean()),
                             "mean_r_f": p,
                             "ci95": [float(ctr - half), float(ctr + half)],
                             "ci_half_width": float(half)})
            means = [r["mean_r_f"] for r in rows if r.get("n")]
            halves = [r["ci_half_width"] for r in rows if r.get("n")]
            diff = np.diff(means) if len(means) > 1 else np.array([])
            spread = (max(means) - min(means)) if means else float("nan")
            typ_ci = float(np.median(halves)) if halves else float("nan")
            shape = ("UNDEFINED_TOO_FEW_DECILES" if diff.size == 0 else
                     "FLAT_WITHIN_CI" if spread < 2 * typ_ci else
                     "MONOTONE_INCREASING" if np.all(diff > 0) else
                     "MONOTONE_DECREASING" if np.all(diff < 0) else
                     "NEITHER")
            strata.append({
                "stratum": b, "n_features": int(sel.sum()),
                "firing_range": [strata_edges[b], strata_edges[b + 1]],
                "deciles": rows,
                "spread_of_mean_r_f": spread,
                "median_ci_half_width": typ_ci,
                "spread_over_ci_half_width": (spread / typ_ci
                                              if typ_ci > 0 else float("inf")),
                "shape": shape,
            })
        per_seed.append({"seed": s, "strata": strata})

    shapes: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for r in per_seed:
        for st in r["strata"]:
            if "shape" in st:
                shapes[st["shape"]] = shapes.get(st["shape"], 0) + 1
            else:
                k = st.get("status", "UNKNOWN")
                statuses[k] = statuses.get(k, 0) + 1
    informative = sum(v for k, v in shapes.items()
                      if k in {"MONOTONE_INCREASING", "MONOTONE_DECREASING"})

    # ---- AMENDMENT AT section A3: FLAT'S TOLERANCE, CONVERTED --------------
    # AS16 reports NEITHER in every scored cell and concludes F1's antecedent
    # fails. That is a claim about the DATA only if the rule could have said
    # otherwise at this precision -- so the tolerance is stated in the units
    # of the thing it constrains and put beside the observed spread and beside
    # the regression's own outcome-scale bound. A verdict rule whose tolerance
    # is tighter than the effect being characterised is an incomplete
    # instrument in the RULE (gotcha 24), not a property of the data.
    tol, spr, nf = [], [], []
    for r in per_seed:
        for st in r["strata"]:
            if "shape" not in st:
                continue
            tol.append(2.0 * st["median_ci_half_width"])
            spr.append(st["spread_of_mean_r_f"])
            nf.append(st["n_features"])
    tol_a, spr_a, nf_a = np.array(tol), np.array(spr), np.array(nf)
    if tol_a.size:
        exceed = spr_a / np.maximum(tol_a, 1e-300)
        flat_report = {
            "rule": ("FLAT_WITHIN_CI iff spread(max - min of the decile mean "
                     "r_f) < 2 * median(decile Wilson CI half-width). The "
                     "tolerance is DATA-DRIVEN, not a constant, so it is "
                     "reported as a distribution over the scored cells."),
            "n_scored_cells": int(tol_a.size),
            "tolerance_pp_of_r_f": {
                "min": float(100 * tol_a.min()),
                "p25": float(100 * np.percentile(tol_a, 25)),
                "median": float(100 * np.median(tol_a)),
                "p75": float(100 * np.percentile(tol_a, 75)),
                "max": float(100 * tol_a.max())},
            "observed_spread_pp_of_r_f": {
                "min": float(100 * spr_a.min()),
                "median": float(100 * np.median(spr_a)),
                "max": float(100 * spr_a.max())},
            "observed_spread_over_tolerance": {
                "min": float(exceed.min()), "median": float(np.median(exceed)),
                "max": float(exceed.max()),
                "n_cells_below_1_i_e_FLAT": int((exceed < 1.0).sum())},
            "n_features_per_scored_cell": {
                "min": int(nf_a.min()), "median": int(np.median(nf_a)),
                "max": int(nf_a.max())},
            "comparison_pp": comparison_pp,
            "n_cells_whose_tolerance_is_below_the_comparison": int(
                (100 * tol_a < comparison_pp).sum()),
            "n_cells_whose_observed_spread_is_below_the_comparison": int(
                (100 * spr_a < comparison_pp).sum()),
            # DERIVED (gotcha 20): the reading is computed, not asserted.
            "tolerance_is_tighter_than_the_comparison": bool(
                float(100 * np.median(tol_a)) < comparison_pp),
            "reading": (
                "If the tolerance were TIGHTER than the regression's own "
                "outcome-scale bound, a NEITHER verdict would be a property "
                "of the rule and the binned means would be the report "
                "instead. If it is LOOSER, the verdict is a property of the "
                "data. Both numbers are above; the boolean is derived from "
                "them."),
        }
    else:
        flat_report = {"status": "NO_SCORED_CELLS"}

    return {
        "per_seed": per_seed,
        "shape_counts": shapes,
        "status_counts": statuses,
        "n_cells_total": int(sum(shapes.values()) + sum(statuses.values())),
        "cell_definition": (
            f"{len(per_seed)} anchors x {n_strata} firing-rate strata = "
            f"{len(per_seed) * n_strata} cells, of which "
            f"{sum(statuses.values())} return a status rather than a shape. "
            "The count of SCORED cells is NOT a factorisation of deciles by "
            "strata: the deciles are of the predictor WITHIN each stratum, "
            "and the strata are corrector_scan.BIN_EDGES."),
        "n_informative_strata": informative,
        "n_flat_within_ci": shapes.get("FLAT_WITHIN_CI", 0),
        "flat_tolerance_report": flat_report,
        "strata_edges": strata_edges,
        "strata_source": "corrector_scan.BIN_EDGES",
        "predictor_deciles_on": "log M",
        "translation": (
            "F1 wrote the deciles on g~; the predictor of record is M (AC9). "
            "AL7's precedent translates a prediction about a deterministic "
            "function of a retained quantity rather than voiding it."),
        "pre_declared_reading": (
            "MONOTONE-AND-FLAT within every stratum is the MODEL-FREE version "
            "of 'no material association', and it is the strongest form the "
            "negative result can take: the qualitative conclusion then holds "
            "without the spline, the basis, the penalty, or any of AS §5's "
            "four conventions. A flat panel is NOT a failure of the "
            "instrument, and the FLAT_WITHIN_CI verdict exists so it cannot "
            "be reported as one."),
    }


def outcome_scale_bound(rows_ok: list[dict]) -> dict:
    """AMENDMENT AT section A2. The outcome-scale bound under BOTH fits.

    0.352 pp is the most load-bearing number in the writeup and it was
    computed under the binomial. Quoting a binomial-derived magnitude beside
    a beta-binomial interval is incoherent, so the bound is recomputed under
    the dispersion-corrected fits at EVERY grid point -- and the 2.1x factor
    AS13 measured at the reference theta is measured everywhere rather than
    carried, because it is a per-fit consequence of `rho` and the spline.
    """
    # 0.352 pp is the most load-bearing number in the writeup and it was
    # computed under the binomial. Quoting a binomial-derived magnitude beside
    # a beta-binomial interval is incoherent, so the bound is recomputed under
    # the dispersion-corrected fits at EVERY grid point.
    b_pp = np.array([c["marginal_effect_percentage_points"]["fixed_effect"]
                     for c in rows_ok])
    d_pp = np.array([c["marginal_effect_percentage_points_betabinomial"]
                     ["fixed_effect"] for c in rows_ok])
    t_ax = [c["theta"] for c in rows_ok]
    b_ci = [c["marginal_effect_percentage_points"]["fixed_effect_ci95"]
            for c in rows_ok]
    d_ci = [c["marginal_effect_percentage_points_betabinomial"]
            ["fixed_effect_ci95"] for c in rows_ok]
    i_b, i_d = int(np.argmax(np.abs(b_pp))), int(np.argmax(np.abs(d_pp)))
    ratio = np.abs(d_pp) / np.maximum(np.abs(b_pp), 1e-300)
    out = {
        "quantity": ("|average marginal effect of z(log M) on r_f|, in "
                     "percentage points per SD, maximised over the qualifying "
                     "theta grid"),
        "binomial_PRE_REGISTERED_INTERVAL_SUPERSEDED_AS_A_MAGNITUDE": {
            "max_abs_pp": float(np.abs(b_pp).max()),
            "at_theta": t_ax[i_b],
            "signed_pp_at_that_theta": float(b_pp[i_b]),
            "ci95_pp_at_that_theta": b_ci[i_b],
            "max_abs_ci_endpoint_pp": float(max(abs(v) for c in b_ci for v in c)),
            "status": ("SUPERSEDED AS A MAGNITUDE -- rho = 0.401 is measured "
                       "and AK9a fires 40/40, so this interval is known too "
                       "narrow. It REMAINS the pre-registered interval and is "
                       "reported first."),
        },
        "betabinomial_HEADLINE_MAGNITUDE": {
            "max_abs_pp": float(np.abs(d_pp).max()),
            "at_theta": t_ax[i_d],
            "signed_pp_at_that_theta": float(d_pp[i_d]),
            "ci95_pp_at_that_theta": d_ci[i_d],
            "max_abs_ci_endpoint_pp": float(max(abs(v) for c in d_ci for v in c)),
        },
        "per_theta": [
            {"theta": t, "binomial_pp": float(b), "binomial_ci95_pp": bc,
             "betabinomial_pp": float(d), "betabinomial_ci95_pp": dc,
             "difference_pp": float(d - b),
             "ratio_abs": float(abs(d) / max(abs(b), 1e-300)),
             "rho_median": c["rho_median"]}
            for t, b, d, bc, dc, c in zip(t_ax, b_pp, d_pp, b_ci, d_ci, rows_ok)],
        # THE FACTOR IS NOT UNIFORM AND IS NOT ASSUMED TO BE. AS13 measured
        # 2.1x at the reference theta; the brief explicitly forbids carrying
        # that across the grid, so it is measured at every point.
        "factor_betabinomial_over_binomial": {
            **_spread(ratio),
            "at_reference_theta": float(
                ratio[t_ax.index(THETA_LITERATURE)]
                if THETA_LITERATURE in t_ax else float("nan")),
            "n_thetas_where_the_correction_SHRINKS_the_magnitude":
                int((ratio < 1.0).sum()),
            "n_thetas_where_it_grows_it": int((ratio > 1.0).sum()),
        },
        "difference_pp_is_the_honest_summary": {
            **_spread(d_pp - b_pp),
            "why": (
                "The binomial marginal effect CROSSES ZERO inside the "
                "qualifying range, so the ratio's denominator vanishes and the "
                "ratio is not a summary of anything -- AS15's 362x in a "
                "different costume, and the third time this project has met "
                "that shape. The DIFFERENCE is bounded and says what the "
                "correction actually does."),
        },
        "comparison_to_AS14": {
            "AS14_published_bound_pp": AS14_BINOMIAL_BOUND_PP,
            "betabinomial_bound_pp": float(np.abs(d_pp).max()),
            # DERIVED (gotcha 20).
            "betabinomial_exceeds_AS14": bool(
                np.abs(d_pp).max() > AS14_BINOMIAL_BOUND_PP),
            "betabinomial_below_1_pp": bool(np.abs(d_pp).max() < 1.0),
        },
        "AK9a_direction": (
            "Both families under-predict the endpoint atoms 40/40, so EVERY "
            "bound here is too narrow. The direction of the misspecification "
            "STRENGTHENS the evidence-of-absence reading."),
        "honesty_clause": (
            "The marginal effect is NOT outcome-blind and was adopted after "
            "the result was seen (AS section 3). Reported alongside and never "
            "instead of b1_z; no conclusion depends on it; the SE is "
            "first-order."),
    }
    return out


def d_fs_grid_summary(rows_ok: list[dict]) -> dict:
    """AMENDMENT AT section A4. `gamma` across the qualifying grid.

    E4's 12.56x was read off ONE theta. Characterising the null term
    thoroughly and the material one thinly would be the wrong writeup, so
    `gamma` gets the curve, the three interval treatments, the outcome-scale
    conversion per SD, and the ordering stated as an ordering.
    """
    t_ax = [c["theta"] for c in rows_ok]
    g_fe = np.array([c["meta_gamma_d_fs"]["fixed_effect"] for c in rows_ok])
    g_bb = np.array([c["meta_gamma_d_fs_betabinomial"]["fixed_effect"]
                     for c in rows_ok])
    dm = np.array([c["term_ordering"]["ratios"]["d_fs_over_logM"]["median"]
                   for c in rows_ok])
    ds = np.array([c["term_ordering"]["ratios"]["d_fs_over_spline"]["median"]
                   for c in rows_ok])
    ms = np.array([c["term_ordering"]["ratios"]["logM_over_spline"]["median"]
                   for c in rows_ok])
    g_sd_pp = np.array([c["marginal_effect_d_fs_percentage_points"]
                        ["per_SD_of_d_fs"]["fixed_effect"] for c in rows_ok])
    orderings = [tuple(c["term_ordering"]["ranked_terms"]) for c in rows_ok]
    out = {
        "quantity": ("gamma and its share of the linear predictor at every "
                     "qualifying theta -- everything b1 has had since AR"),
        "per_theta": [
            {"theta": t,
             "gamma_binomial_fe": float(gb),
             "gamma_binomial_ci95": c["meta_gamma_d_fs"]["fixed_effect_ci95"],
             "gamma_across_seed_ci95": c["meta_gamma_d_fs"]["across_seed_ci95"],
             "gamma_betabinomial_fe": float(gd),
             "gamma_betabinomial_ci95":
                 c["meta_gamma_d_fs_betabinomial"]["fixed_effect_ci95"],
             "gamma_I2_percent": c["meta_gamma_d_fs"]["I2_percent"],
             "gamma_I2_percent_betabinomial":
                 c["meta_gamma_d_fs_betabinomial"]["I2_percent"],
             "delta_r_f_pp_per_SD_d_fs": float(gp),
             "delta_r_f_pp_per_SD_d_fs_ci95":
                 c["marginal_effect_d_fs_percentage_points"]
                  ["per_SD_of_d_fs"]["fixed_effect_ci95"],
             "sd_ratio_d_fs_over_logM": float(a),
             "sd_ratio_d_fs_over_spline": float(b),
             "sd_ratio_logM_over_spline": float(cc),
             "ranked_terms": list(o)}
            for t, gb, gd, gp, a, b, cc, o, c in
            zip(t_ax, g_fe, g_bb, g_sd_pp, dm, ds, ms, orderings, rows_ok)],
        # DERIVED, never asserted (gotcha 20).
        "gamma_sign_is_unanimous_across_the_grid": bool(
            len({("POSITIVE" if v > 0 else "NEGATIVE" if v < 0 else "ZERO")
                 for v in g_fe}) == 1),
        "gamma_signs_observed": sorted(
            {("POSITIVE" if v > 0 else "NEGATIVE" if v < 0 else "ZERO")
             for v in g_fe}),
        "gamma_sign_is_unanimous_betabinomial": bool(
            len({("POSITIVE" if v > 0 else "NEGATIVE" if v < 0 else "ZERO")
                 for v in g_bb}) == 1),
        "b1_signs_observed": sorted(
            {("POSITIVE" if c["meta_b1_z"]["fixed_effect"] > 0 else
              "NEGATIVE" if c["meta_b1_z"]["fixed_effect"] < 0 else "ZERO")
             for c in rows_ok}),
        "sd_ratio_d_fs_over_logM": {
            **_spread(dm),
            "min_at_theta": t_ax[int(np.argmin(dm))],
            "n_thetas_above_1": int((dm > 1.0).sum()),
            "n_thetas_below_1": int((dm < 1.0).sum()),
            "exceeds_1_at_every_theta": bool(bool((dm > 1.0).all())),
        },
        "sd_ratio_d_fs_over_spline": _spread(ds),
        "sd_ratio_logM_over_spline": _spread(ms),
        "ordering_is_constant_across_the_grid": bool(len(set(orderings)) == 1),
        "orderings_observed": sorted({" > ".join(o) for o in orderings}),
        "gamma_is_a_confound_control": (
            "d_fs entered as a confound control (F1.6, gotcha 17). A large "
            "gamma is a finding ABOUT THE CONFOUND, reported as a "
            "characterised association on exactly b1's terms -- no alpha, "
            "interval plus sensitivity, NO MECHANISM STORY. If it points at "
            "J8 that is a note in future work, not a claim here."),
    }
    return out


def arm_comparison(primary: dict, sensitivity: dict) -> dict:
    """AMENDMENT AU §1a. The four arms side by side, under BOTH likelihoods,
    on BOTH scales, with `includes_zero` DERIVED per cell.

    Every arm figure on record was a binomial fit, whose per-seed SEs are ~3.89x
    too narrow at the measured `rho = 0.401`. Whether the arms genuinely
    DISAGREE, or are four point estimates each individually consistent with
    zero, is a different statement -- and it cannot be settled by scaling a
    binomial interval, because the beta-binomial RELOCATES the estimate as well
    as widening it (+0.0056 -> +0.0120 for the primary). AS2's E3 is the
    instance: its summary arithmetic held the estimate fixed and its MISS had a
    cause its own reading did not cover (gotcha 30).

    AU1 ratifies the reporting standard indexing over ARMS as well as thetas, so
    this table is the object that standard is evaluated on.
    """
    def one(name, arm):
        if "meta_b1_z" not in arm:
            return {"arm": name, "status": arm.get("status", "ABSENT")}
        row = {"arm": name}
        for key, tag in (("meta_b1_z", "binomial"),
                         ("meta_b1_z_betabinomial", "betabinomial")):
            if key not in arm:
                row[tag] = {"status": "NOT_FIT -- dispersion was not requested "
                                      "for this arm"}
                continue
            m = arm[key]
            ci = m["fixed_effect_ci95"]
            row[tag] = {
                "b1_z": m["fixed_effect"], "ci95_logit": ci,
                "I2_percent": m["I2_percent"],
                # DERIVED (gotcha 20).
                "includes_zero": bool(ci[0] <= 0.0 <= ci[1]),
                "sign": ("POSITIVE" if m["fixed_effect"] > 0 else
                         "NEGATIVE" if m["fixed_effect"] < 0 else "ZERO"),
            }
        for key, tag in (
                ("marginal_effect_percentage_points", "binomial"),
                ("marginal_effect_percentage_points_betabinomial",
                 "betabinomial")):
            if key not in arm or "status" in row.get(tag, {}):
                continue
            pp = arm[key]
            row[tag]["delta_r_f_pp"] = pp["fixed_effect"]
            row[tag]["delta_r_f_pp_ci95"] = pp["fixed_effect_ci95"]
            row[tag]["pp_includes_zero"] = bool(
                pp["fixed_effect_ci95"][0] <= 0.0 <= pp["fixed_effect_ci95"][1])
        return row

    rows = [one("primary_hard_bin", primary)]
    for k, nm in (("degenerate_excluded", "degenerate_excluded"),
                  ("smooth_normaliser", "smooth_normaliser"),
                  ("feature_side", "feature_side")):
        rows.append(one(nm, sensitivity.get(k, {})))

    fitted = [r for r in rows if "betabinomial" in r
              and "status" not in r["betabinomial"]]
    sens = [r for r in fitted if r["arm"] != "primary_hard_bin"]
    signs_b = {r["binomial"]["sign"] for r in fitted}
    signs_d = {r["betabinomial"]["sign"] for r in fitted}
    return {
        "why": ("AU1 ratifies AB1 decision A's reporting standard indexing over "
                "ARMS as well as thetas. AU section 1a requires the arms be "
                "compared under the likelihood the dispersion measurement says "
                "is correct, not under the one whose SEs are 3.89x too narrow."),
        "rows": rows,
        "n_arms_fit_both_ways": len(fitted),
        # DERIVED, every one (gotcha 20).
        "binomial": {
            "n_arms_including_zero": sum(
                r["binomial"]["includes_zero"] for r in fitted),
            "signs_observed": sorted(signs_b),
            "sign_is_unanimous_across_arms": bool(len(signs_b) == 1)},
        "betabinomial": {
            "n_arms_including_zero": sum(
                r["betabinomial"]["includes_zero"] for r in fitted),
            "n_sensitivity_arms_including_zero": sum(
                r["betabinomial"]["includes_zero"] for r in sens),
            "n_sensitivity_arms": len(sens),
            "signs_observed": sorted(signs_d),
            "sign_is_unanimous_across_arms": bool(len(signs_d) == 1)},
        "reading": (
            "If every interval covers zero then 'two of three arms reverse' "
            "describes four point estimates each individually consistent with "
            "zero -- materially WEAKER than 'the arms disagree', and the weaker "
            "true statement is the one to make. If some exclude zero that is "
            "reported exactly as plainly. The counts above decide it; this "
            "sentence does not."),
        "AK9a_cuts_the_other_way_here": (
            "Both likelihoods under-predict the endpoint atoms 40/40, so even "
            "these intervals are somewhat too narrow. For an interval-around-"
            "zero framing that STRENGTHENS evidence of absence (AS2, AT11); for "
            "the ARMS-COVER-ZERO framing AU1 adopts it is the WRONG SIGN -- "
            "too-narrow intervals make the arms look MORE discrepant than they "
            "are. Corrected at AU6."),
        "prohibitions": [
            "do not average the arms",
            "do not build a tiebreak, consensus rule, or any procedure that "
            "selects among arms",
            "do not promote the degenerate-excluded arm because it agrees with "
            "the primary -- it perturbs 0.9219% of cells, so its agreement is "
            "close to mechanical and is not independent corroboration",
        ],
    }


def permutation_null_for_the_panel(
        y: np.ndarray, logM: np.ndarray, fire_rate: np.ndarray,
        keep: np.ndarray, n_trials: int, strata_edges: list[float],
        n_deciles: int = 10, n_perm: int = 200, seed: int = 20260802) -> dict:
    """AMENDMENT AU §3. THE THIRD CANDIDATE CAUSE of F1's blanket `NEITHER`.

    Two causes were enumerated: a real non-monotone pattern, or a FLAT tolerance
    too tight to pass. AT12 eliminated the second by measuring the tolerance at
    a median 2.19 pp -- LOOSER than the effect being bounded. **The third was
    not enumerated: if the SAMPLING NOISE on a decile mean is comparable to the
    tolerance, no cell can be FLAT regardless of the truth.**

    AND AT12's REACHABILITY DEMONSTRATION CANNOT RULE IT OUT, BECAUSE IT USED
    THE WRONG NULL. It plants `r_f ~ Binomial(39, 0.5)`, which has NO
    between-feature dispersion in `p` and hence `sd(r_f) = 0.080`. The real data
    has `rho = 0.401` and a fitted mixing distribution with real spread. A
    demonstration that FLAT fires under a null the data does not resemble is not
    a demonstration that FLAT can fire on this data (gotcha 13, original form).

    SO THE NULL HERE IS A PERMUTATION, not a parametric draw: within each cell
    the feature-to-decile assignment is shuffled and the spread recomputed. That
    preserves the cell's OWN `r_f` distribution exactly -- overdispersion
    included, which is the property the parametric null lacked -- and destroys
    only the association with `log M`.
    """
    rng = np.random.default_rng(seed)
    n_seeds = y.shape[0]
    n_strata = len(strata_edges) - 1
    cells = []
    for s in range(n_seeds):
        m = keep[s]
        fb = np.clip(bin_index(fire_rate[s], strata_edges), 0, n_strata - 1)
        for b in range(n_strata):
            sel = m & (fb == b)
            if sel.sum() < 10 * n_deciles:
                continue
            lm, kk = logM[s][sel], y[s][sel]
            rf = kk / float(n_trials)
            cuts = np.quantile(lm, np.linspace(0, 1, n_deciles + 1)[1:-1])
            dec = np.digitize(lm, cuts)
            sizes, means, ses, halves = [], [], [], []
            z = 1.959964
            for d in range(n_deciles):
                dm = dec == d
                n_d = int(dm.sum())
                if not n_d:
                    continue
                sizes.append(n_d)
                means.append(float(rf[dm].mean()))
                # THE EMPIRICAL sampling SE of the decile mean -- sd of r_f
                # ACROSS FEATURES over sqrt(n). This is the quantity the
                # tolerance has to be compared against, and it is not the
                # Wilson half-width: Wilson treats each feature's 39 trials as
                # independent Bernoullis and so measures a different (much
                # smaller) thing entirely.
                ses.append(float(rf[dm].std(ddof=1) / np.sqrt(n_d))
                           if n_d > 1 else float("nan"))
                p = float(kk[dm].sum()) / (n_d * n_trials)
                tot = n_d * n_trials
                den = 1 + z * z / tot
                halves.append(float(
                    z * np.sqrt(p * (1 - p) / tot + z * z / (4 * tot * tot))
                    / den))
            if len(means) < 2:
                continue
            obs_spread = float(max(means) - min(means))
            tol = 2.0 * float(np.median(halves))
            # the permutation null on the spread
            sizes_a = np.array(sizes)
            bounds = np.cumsum(sizes_a)[:-1]
            null = np.empty(n_perm)
            n_flat = 0
            for j in range(n_perm):
                perm = rng.permutation(rf)
                gm = np.array([g.mean() for g in np.split(perm, bounds)])
                sp = float(gm.max() - gm.min())
                null[j] = sp
                n_flat += int(sp < tol)
            cells.append({
                "seed": s, "stratum": b, "n_features": int(sel.sum()),
                "n_deciles_scored": len(means),
                "observed_spread": obs_spread,
                "tolerance": tol,
                "median_empirical_se_of_a_decile_mean": float(np.median(ses)),
                "median_wilson_half_width": float(np.median(halves)),
                "empirical_se_over_wilson_half_width": float(
                    np.median(ses) / max(np.median(halves), 1e-300)),
                "null_spread_median": float(np.median(null)),
                "null_spread_p05": float(np.percentile(null, 5)),
                "null_spread_p95": float(np.percentile(null, 95)),
                "flat_rate_under_permutation_null": n_flat / n_perm,
                "observed_spread_percentile_in_null": float(
                    100.0 * (null < obs_spread).mean()),
            })
    if not cells:
        return {"status": "NO_SCORED_CELLS"}

    se = np.array([c["median_empirical_se_of_a_decile_mean"] for c in cells])
    tol = np.array([c["tolerance"] for c in cells])
    fr = np.array([c["flat_rate_under_permutation_null"] for c in cells])
    pct = np.array([c["observed_spread_percentile_in_null"] for c in cells])
    obs = np.array([c["observed_spread"] for c in cells])
    nul = np.array([c["null_spread_median"] for c in cells])
    pooled_flat = float(fr.mean())
    med_pct = float(np.median(pct))
    return {
        "null": ("PERMUTATION -- the feature-to-decile assignment is shuffled "
                 "WITHIN each cell, preserving that cell's own r_f "
                 "distribution exactly (overdispersion included) and "
                 "destroying only the association with log M"),
        "n_permutations_per_cell": n_perm,
        "n_cells": len(cells),
        "per_cell": cells,
        "empirical_se_of_a_decile_mean_pp": {
            "min": float(100 * se.min()), "median": float(100 * np.median(se)),
            "max": float(100 * se.max())},
        "tolerance_pp": {
            "min": float(100 * tol.min()),
            "median": float(100 * np.median(tol)),
            "max": float(100 * tol.max())},
        "empirical_se_over_tolerance": {
            "min": float((se / tol).min()),
            "median": float(np.median(se / tol)),
            "max": float((se / tol).max()),
            # DERIVED (gotcha 20).
            "n_cells_where_noise_meets_or_exceeds_the_tolerance": int(
                (se >= tol).sum())},
        "wilson_understates_the_noise_by": {
            "median": float(np.median(
                [c["empirical_se_over_wilson_half_width"] for c in cells])),
            "why": ("the Wilson interval treats each feature's n_compare trials "
                    "as independent Bernoullis, so it measures the noise in a "
                    "POOLED proportion and not the noise in a MEAN OVER "
                    "FEATURES -- which is the quantity the decile mean is")},
        "flat_rate_under_the_permutation_null": {
            "pooled": pooled_flat,
            "min_over_cells": float(fr.min()),
            "max_over_cells": float(fr.max()),
            "n_cells_with_rate_zero": int((fr == 0).sum())},
        "observed_spread_vs_its_own_null": {
            "median_percentile": med_pct,
            "min_percentile": float(pct.min()),
            "max_percentile": float(pct.max()),
            "n_cells_above_the_95th": int((pct > 95).sum()),
            "median_observed_spread_pp": float(100 * np.median(obs)),
            "median_null_spread_pp": float(100 * np.median(nul)),
            "ratio_observed_to_null_median": float(
                np.median(obs) / max(np.median(nul), 1e-300))},
        # DERIVED, both branches of AU7's pre-declared reading.
        "verdict_rule_can_return_FLAT_on_data_like_this": bool(
            pooled_flat >= 0.05),
        "AU_Q2_flat_rate_below_5pct": bool(pooled_flat < 0.05),
        "AU_Q3_median_percentile_in_25_75": bool(25.0 <= med_pct <= 75.0),
        "reading": (
            "If FLAT effectively cannot fire under a null built from this "
            "data's own r_f distribution, then NEITHER in every cell is a "
            "property of the VERDICT RULE and not of the data, the binned means "
            "with intervals become the report, and F1's figure specification "
            "contained a broken instrument from the day it was written -- its "
            "tolerance was never checked against the noise scale it would be "
            "applied at. If FLAT can fire, NEITHER stands as a property of the "
            "data. The booleans above are computed from the numbers above them; "
            "this sentence decides nothing."),
    }


def _blocked_permutation_spread(rf: np.ndarray, dec: np.ndarray,
                                block: np.ndarray, n_deciles: int,
                                n_perm: int, rng) -> np.ndarray:
    """The decile-mean SPREAD under `n_perm` permutations of `rf` WITHIN blocks.

    ONE estimator, two callers. `block` all-zeros gives the unstratified null
    AU13/AU14 report; `block` = the feature's homology-density bin gives
    amendment AV12c's stratified null. **Nothing else differs between them** --
    same deciles, same cells, same spread statistic, same generator sequence
    shape -- so the comparison between the two is a comparison of the BLOCKING
    and of nothing else. Computing the incumbent a second, slightly different
    way and then differencing it against the new one would confound the two
    (gotcha 21: measure on the reduction the consumer takes).

    Permuting WITHIN a block preserves that block's own `r_f` distribution
    exactly -- so the `d_fs`-to-`r_f` relationship survives untouched -- and
    destroys the association between `log M` and `r_f` NET of the blocking
    variable.
    """
    out = np.empty(n_perm)
    order = [np.flatnonzero(block == b) for b in np.unique(block)]
    dec_masks = [dec == d for d in range(n_deciles)]
    dec_masks = [m for m in dec_masks if m.any()]
    for j in range(n_perm):
        perm = rf.copy()
        for idx in order:
            if idx.size > 1:
                perm[idx] = rng.permutation(rf[idx])
        gm = np.array([perm[m].mean() for m in dec_masks])
        out[j] = float(gm.max() - gm.min())
    return out


def av_p3_verdict(median_percentile: float, n_cells_above_95th: int,
                  max_percentile: float = 95.0, max_cells: int = 10) -> dict:
    """AV-P3's verdict, as a named function rather than an expression inside a
    closure.

    **It is extracted for one reason and it is the reason AU9b gives:** inside
    `_agg` the only fixture available is whatever the permutation happens to
    produce, and in every realistic panel the two halves agree — so an `and`
    and an `or` are indistinguishable there, and the mutation testing said so.
    A function that can be handed a DISAGREEING pair is what makes the
    conjunction testable at all. Seven of nineteen mutations survived the last
    time this kind of logic had no such handle.

    Total over the statistic (gotcha 24): every (percentile, count) pair maps
    to a branch, and **HIT requires BOTH halves**, as AV12c declared.
    """
    inside = bool(median_percentile < max_percentile)
    few = bool(n_cells_above_95th <= max_cells)
    return {"spread_inside_the_central_mass": inside,
            "cell_count_at_or_below_10": few,
            "AV_P3_both_halves": bool(inside and few),
            "half_that_fails": (None if inside and few else
                                "both" if not inside and not few else
                                "spread" if not inside else "cell_count")}


def stratified_permutation_null_for_the_panel(
        y: np.ndarray, logM: np.ndarray, fire_rate: np.ndarray,
        d_fs: np.ndarray, keep: np.ndarray, n_trials: int,
        strata_edges: list[float], n_deciles: int = 10,
        n_dfs_bins_grid: tuple[int, ...] = (2, 5, 10), primary_bins: int = 5,
        n_perm: int = 200, seed: int = 20260802) -> dict:
    """AMENDMENT AV §5c. **The one measurement that closes the panel's loose end.**

    AU14 left a **1.2709x excess** of observed decile spread over the
    permutation null -- 26 of 100 cells above their own 95th percentile against
    5 expected -- and it is UNEXPLAINED. AU's permutation preserves each cell's
    own `r_f` distribution, and the cells are within firing-rate strata, so
    firing rate is controlled. **Homology density is not**, and `d_fs` is worth
    **+3.1217 pp per SD** at the reference threshold against `log M`'s bound of
    **<= 0.3635 pp** (AT13) -- roughly 9x the entire alignment effect. A large
    uncontrolled covariate is the obvious candidate for residual decile
    structure.

    So: permute within **firing-rate-stratum x homology-density** blocks.

    THE BIN COUNT IS A NUISANCE PARAMETER AND IS TREATED AS ONE. AV12c did not
    fix it, so it is reported across a grid rather than chosen: the verdict is
    read at `primary_bins` and **every grid point's verdict is emitted beside
    it**. If they disagree the disagreement is the finding, not something to
    resolve by picking one. The UNSTRATIFIED arm is recomputed here through the
    identical estimator and is expected to reproduce AU14; that reproduction is
    reported as a check rather than assumed.
    """
    n_seeds = y.shape[0]
    n_strata = len(strata_edges) - 1
    cells = []
    for s in range(n_seeds):
        m = keep[s]
        fb = np.clip(bin_index(fire_rate[s], strata_edges), 0, n_strata - 1)
        for b in range(n_strata):
            sel = m & (fb == b)
            if sel.sum() < 10 * n_deciles:
                continue
            lm, kk, dd = logM[s][sel], y[s][sel], d_fs[s][sel]
            rf = kk / float(n_trials)
            cuts = np.quantile(lm, np.linspace(0, 1, n_deciles + 1)[1:-1])
            dec = np.digitize(lm, cuts)
            present = [d for d in range(n_deciles) if (dec == d).any()]
            if len(present) < 2:
                continue
            means = np.array([rf[dec == d].mean() for d in present])
            sizes = np.array([int((dec == d).sum()) for d in present])
            sds = np.array([rf[dec == d].std(ddof=1) if (dec == d).sum() > 1
                            else np.nan for d in present])
            obs = float(means.max() - means.min())

            arms = {}
            for nb in (0,) + tuple(n_dfs_bins_grid):
                if nb == 0:
                    block = np.zeros(dd.size, dtype=np.int64)
                    key = "unstratified"
                else:
                    edges = np.quantile(dd, np.linspace(0, 1, nb + 1)[1:-1])
                    block = np.digitize(dd, edges)
                    key = f"d_fs_bins_{nb}"
                # EVERY ARM OF A CELL GETS THE SAME GENERATOR STATE, so the
                # arms differ by the BLOCKING and by nothing else -- including
                # not by which random draws they happened to consume. Sharing
                # one sequential generator across arms (the first version) made
                # a one-block arm disagree with the unstratified arm by pure
                # Monte-Carlo noise, which would have put that noise into every
                # arm-to-arm comparison this function exists to make.
                null = _blocked_permutation_spread(
                    rf, dec, block, n_deciles, n_perm,
                    np.random.default_rng([seed, s, b]))
                arms[key] = {
                    "null_spread_median": float(np.median(null)),
                    "null_spread_p95": float(np.percentile(null, 95)),
                    "observed_percentile": float(100.0 * (null < obs).mean()),
                    "observed_above_p95": bool(
                        obs > float(np.percentile(null, 95))),
                    "n_distinct_blocks": int(len(np.unique(block))),
                }
            cells.append({
                "seed": s, "stratum": b, "n_features": int(sel.sum()),
                "n_deciles_scored": len(present),
                "observed_spread": obs,
                # AV12b's replacement interval: the EMPIRICAL between-feature
                # dispersion, not a Wilson construction on a pooled proportion.
                "decile_means": [float(x) for x in means],
                "decile_n": [int(x) for x in sizes],
                "decile_empirical_se": [
                    float(sd / np.sqrt(n)) if n > 1 else None
                    for sd, n in zip(sds, sizes)],
                "arms": arms,
            })
    if not cells:
        return {"status": "NO_SCORED_CELLS"}

    def _agg(key):
        pct = np.array([c["arms"][key]["observed_percentile"] for c in cells])
        above = np.array([c["arms"][key]["observed_above_p95"] for c in cells])
        nul = np.array([c["arms"][key]["null_spread_median"] for c in cells])
        obs = np.array([c["observed_spread"] for c in cells])
        med_pct = float(np.median(pct))
        n_above = int(above.sum())
        return {
            "median_observed_spread_pp": float(100 * np.median(obs)),
            "median_null_spread_pp": float(100 * np.median(nul)),
            "ratio_observed_to_null_median": float(
                np.median(obs) / max(np.median(nul), 1e-300)),
            "median_percentile_of_the_observed_spread": med_pct,
            "n_cells_above_their_own_95th": n_above,
            "n_cells": len(cells),
            "n_expected_above_by_chance": 0.05 * len(cells),
            # AV-P3's two halves, DERIVED (gotcha 20). "Within the null's
            # central mass" is read at the cell level as the MEDIAN cell's
            # observed spread sitting below its own null's 95th percentile.
            **av_p3_verdict(med_pct, n_above),
        }

    arms = {k: _agg(k) for k in cells[0]["arms"]}
    primary_key = f"d_fs_bins_{primary_bins}"
    grid_keys = [f"d_fs_bins_{nb}" for nb in n_dfs_bins_grid]
    verdicts = {k: arms[k]["AV_P3_both_halves"] for k in grid_keys}
    return {
        "null": ("PERMUTATION of r_f WITHIN blocks, one estimator for every "
                 "arm -- the arms differ ONLY in the blocking, so their "
                 "comparison is a comparison of the blocking"),
        "blocking": ("firing-rate stratum (the cell) x homology-density bin "
                     "(quantiles of d_fs within the cell)"),
        "n_permutations_per_cell": n_perm,
        "n_cells": len(cells),
        "primary_arm": primary_key,
        "bin_count_is_a_nuisance_parameter": (
            "AV12c fixed the blocking VARIABLE and not the number of bins, so "
            "the grid is reported and the verdict at every grid point is "
            "emitted. A disagreement across the grid IS the finding."),
        "arms": arms,
        "per_cell": cells,
        "AV_P3": {
            "prediction": ("under the stratified permutation the observed "
                           "decile spread falls within the null's central mass "
                           "AND the count of cells above their own 95th "
                           "percentile falls to <= 10 (from 26)"),
            "primary_bins": primary_bins,
            "outcome": "HIT" if verdicts[primary_key] else "MISS",
            "verdict_at_every_grid_point": verdicts,
            "grid_is_unanimous": len(set(verdicts.values())) == 1,
        },
        "unstratified_reproduction_check": {
            "why": ("AU14 reported median observed 9.0157 pp, median null "
                    "7.0937 pp, ratio 1.2709, median percentile 78.2, 26 of "
                    "100 cells above their own 95th. The unstratified arm here "
                    "runs the SAME null through this module's estimator; close "
                    "agreement is what licenses reading the stratified arms as "
                    "a change in the blocking rather than in the machinery."),
            "AU14_reference": {"median_observed_pp": 9.0157,
                               "median_null_pp": 7.0937, "ratio": 1.2709,
                               "median_percentile": 78.2,
                               "n_cells_above_95th": 26, "n_cells": 100},
            "recomputed": arms["unstratified"],
        },
    }


def firing_stratum_refit(y: np.ndarray, logM: np.ndarray, log_phi: np.ndarray,
                         d_fs: np.ndarray, fire_rate: np.ndarray,
                         keep: np.ndarray, n_trials: int,
                         lo: float, hi: float, primary: dict) -> dict:
    """AMENDMENT AW section 4, item 3 -- EXPLORATORY, per AW's declared §2a
    scope: this refit CANNOT become primary under any outcome (the caller
    reports the branch, not this function).

    Refits the SAME design-3 model with `keep` additionally restricted to
    `fire_rate in [lo, hi)` -- i.e. hard-bin's `[1e-3, 3e-3)` stratum, the one
    AV18 found still carrying the panel's residual excess after blocking on
    `d_fs` (ratio 1.69 in 16 of 20 seeds, against 1.01-1.13 elsewhere). If
    alignment predicts reproducibility only within that band, the single
    global `b1` (no `logM x firing-rate` interaction in the primary model)
    would average that signal over all bands and dilute it toward the bound
    AT13 measured. This is the direct test: does the coefficient estimated
    ONLY on that stratum's features exceed the global one materially?

    `fit_arm`'s ARM-LEVEL `meta_b1_z_betabinomial` (already meta-analysed
    over seeds, per-seed dispersion-corrected) is what is compared against
    the same field on `primary` -- the same treatment the primary result
    reports, not the plain-binomial one whose SEs amendment AU already showed
    understate by `rho = 0.401`.
    """
    keep_stratum = keep & (fire_rate >= lo) & (fire_rate < hi)
    n_features_per_seed = [int(m.sum()) for m in keep_stratum]
    if min(n_features_per_seed) < 20:
        return {"status": "TOO_FEW_FEATURES",
               "n_features_per_seed": n_features_per_seed,
               "firing_range": [lo, hi]}
    arm = fit_arm(y, logM, log_phi, d_fs, keep_stratum, n_trials,
                 dispersion=True, label="AW_firing_stratum_refit")
    stratum_fe = arm["meta_b1_z_betabinomial"]["fixed_effect"]
    global_fe = primary["meta_b1_z_betabinomial"]["fixed_effect"]
    ratio = abs(stratum_fe) / abs(global_fe) if global_fe else float("nan")
    return {
        "firing_range": [lo, hi],
        "n_features_per_seed": n_features_per_seed,
        "meta_b1_z_betabinomial": arm["meta_b1_z_betabinomial"],
        "global_b1_z_betabinomial_reference": global_fe,
        "global_b1_z_betabinomial_ci95_reference":
            primary["meta_b1_z_betabinomial"]["fixed_effect_ci95"],
        # DERIVED (gotcha 20): "materially exceeds" is a number, not asserted.
        "abs_ratio_stratum_over_global": ratio,
        "exceeds_global_by_more_than_3x": (
            bool(ratio > 3.0) if ratio == ratio else None),  # NaN-safe
    }


def stratum_decile_shape(y: np.ndarray, logM: np.ndarray,
                         keep_stratum: np.ndarray, n_trials: int,
                         n_deciles: int = 10) -> dict:
    """AMENDMENT CD item 2, part 4. The decile shape WITHIN the refit stratum,
    at one theta.

    `model_free_panel` writes deciles per firing-rate stratum from
    `corrector_scan.BIN_EDGES`; the AW refit's `[lo, hi)` is NOT one of those
    edges, so the panel does not answer "what shape does the refit's own
    stratum have". This computes it on the SAME rows the refit fits, with the
    panel's conventions reused verbatim -- deciles of `log M`, the same nominal
    Wilson interval, and the same understatement caveat (features within a seed
    share a dictionary and are not independent trials).

    WHAT IS REPORTED IS THE ARGMIN AND ARGMAX DECILE POSITION PER SEED, and the
    seed counts at each position. "Dip-then-rise" is a SHAPE CLAIM, and the
    honest instrument for it is where the minimum sits: a dip-then-rise curve
    puts its argmin in the interior, a monotone one puts it at an end. So the
    interior fraction is derived (gotcha 20) rather than the shape being
    asserted -- and `n_deciles` is reported so the "interior" is readable.
    """
    n_seeds = y.shape[0]
    per_seed = []
    for s in range(n_seeds):
        sel = keep_stratum[s]
        if sel.sum() < 10 * n_deciles:
            per_seed.append({"seed": s, "n_features": int(sel.sum()),
                             "status": "TOO_FEW_FEATURES"})
            continue
        lm, kk = logM[s][sel], y[s][sel]
        cuts = np.quantile(lm, np.linspace(0, 1, n_deciles + 1)[1:-1])
        dec = np.digitize(lm, cuts)
        means, ns = [], []
        for d in range(n_deciles):
            dm = dec == d
            ns.append(int(dm.sum()))
            means.append(float(kk[dm].sum()) / float(dm.sum() * n_trials)
                         if dm.any() else float("nan"))
        ok = [i for i, v in enumerate(means) if v == v]      # NaN-safe
        if not ok:
            per_seed.append({"seed": s, "n_features": int(sel.sum()),
                             "status": "NO_POPULATED_DECILES"})
            continue
        amin = min(ok, key=lambda i: means[i])
        amax = max(ok, key=lambda i: means[i])
        per_seed.append({
            "seed": s, "n_features": int(sel.sum()),
            "decile_mean_r_f": means, "decile_n": ns,
            "argmin_decile": amin, "argmax_decile": amax,
            "spread_of_mean_r_f": float(max(means[i] for i in ok)
                                        - min(means[i] for i in ok)),
        })

    usable = [r for r in per_seed if "argmin_decile" in r]
    if not usable:
        return {"status": "NO_USABLE_SEEDS", "n_deciles": n_deciles,
                "per_seed": per_seed}
    amins = [r["argmin_decile"] for r in usable]
    amaxs = [r["argmax_decile"] for r in usable]
    interior = [a for a in amins if 0 < a < n_deciles - 1]
    return {
        "status": "OK",
        "n_deciles": n_deciles,
        "n_seeds_usable": len(usable),
        # Seed COUNTS at each position, which is what the brief asks for. A
        # modal position with a count is readable; a mean decile index is not.
        "argmin_decile_seed_counts": {str(d): amins.count(d)
                                      for d in sorted(set(amins))},
        "argmax_decile_seed_counts": {str(d): amaxs.count(d)
                                      for d in sorted(set(amaxs))},
        "n_seeds_argmin_interior": len(interior),
        "frac_seeds_argmin_interior": len(interior) / len(usable),
        "per_seed": per_seed,
    }


def firing_stratum_refit_curve(
        counts_by_theta: dict, thetas: list, logM: np.ndarray,
        log_phi: np.ndarray, d_fs: np.ndarray, fire_rate: np.ndarray,
        keep: np.ndarray, n_trials: int, lo: float, hi: float,
        global_rows: list[dict]) -> dict:
    """AMENDMENT CD item 2. AW's stratum refit at EVERY qualifying theta.

    WHY IT EXISTS. `firing_stratum_refit` above fits the stratum at the
    REFERENCE theta only, and its result lives in a single field. This
    project's standing requirement is that a sensitivity curve over theta
    accompanies anything reportable, so a one-threshold coefficient cannot
    carry the localised-structure claim on its own -- and the specific question
    is whether the stratum-over-global ratio, the near-disjointness of the two
    intervals, and the `I2 = 0%` reading are properties of the TERM or of the
    reference threshold.

    THE GLOBAL SIDE IS READ, NOT REFITTED. `global_rows` is the theta curve
    computed in this same run, on the same rows with the same `keep`, so the
    two sides of every ratio come from one job on one input. Refitting the
    global arm here would create a second global estimate that could silently
    disagree with the one the run reports -- the collision gotcha 16 is about.
    Rows are joined ON THE THETA VALUE, never by position (gotcha 29).

    EVERY RATIO GOES THROUGH `ratio_rule_diff` (gotcha 33). Both sides are
    logit coefficients per SD measured at the SAME theta in the SAME run, so
    the settings match by construction -- but "by construction" is exactly the
    claim that stops being true when someone changes a caller, so it is
    checked rather than asserted, and the diff is stored per grid point.

    EXPLORATORY, and inherits AW §2a's scope unchanged: this cannot become
    primary under any outcome, at any theta. Adding thresholds does not
    promote it.
    """
    keep_stratum = keep & (fire_rate >= lo) & (fire_rate < hi)
    by_theta = {round(float(r["theta"]), 2): r for r in global_rows
                if "meta_b1_z" in r}

    rows = []
    for t in thetas:
        key = round(float(t), 2)
        if key not in counts_by_theta:
            rows.append({"theta": t, "status": "ABSENT_FROM_ARTIFACT"})
            continue
        if key not in by_theta:
            # An absent global side is reported absent, never treated as
            # agreeing or silently skipped (gotcha 15).
            rows.append({"theta": t, "status": "GLOBAL_ROW_ABSENT"})
            continue
        n_per_seed = [int(m.sum()) for m in keep_stratum]
        if min(n_per_seed) < 20:
            rows.append({"theta": t, "status": "TOO_FEW_FEATURES",
                         "n_features_per_seed": n_per_seed})
            continue

        y_t = counts_by_theta[key].astype(np.float64)
        arm = fit_arm(y_t, logM, log_phi, d_fs, keep_stratum, n_trials,
                      dispersion=True, label=f"CD_stratum_refit_theta_{key:.2f}")
        g = by_theta[key]

        row = {"theta": t, "status": "OK", "n_features_per_seed": n_per_seed}
        for lik, sk, gk in (
            ("binomial", "meta_b1_z", "meta_b1_z"),
            ("betabinomial", "meta_b1_z_betabinomial", "meta_b1_z_betabinomial"),
        ):
            sm, gm = arm[sk], g[gk]
            entry = {
                "stratum": sm, "global": gm,
                "intervals": interval_agreement(
                    sm["fixed_effect_ci95"], gm["fixed_effect_ci95"],
                    f"stratum [{lo:.0e}, {hi:.0e}) theta {key:.2f}",
                    f"global theta {key:.2f}"),
                # DISJOINTNESS is the brief's question and it is a DERIVED
                # boolean, not a reading of the agreement scale above.
                "intervals_disjoint": bool(
                    sm["fixed_effect_ci95"][0] > gm["fixed_effect_ci95"][1]
                    or gm["fixed_effect_ci95"][0] > sm["fixed_effect_ci95"][1]),
                "I2_percent_stratum": sm.get("I2_percent"),
                "I2_percent_global": gm.get("I2_percent"),
                "ratio_rule": ratio_rule_diff(
                    f"stratum-over-global b1_z ({lik}) at theta {key:.2f}",
                    ratio_component(
                        sm["fixed_effect"], "logit coefficient per SD of log M",
                        at=f"theta={key:.2f}, fire_rate in [{lo:.0e}, {hi:.0e}), "
                           f"{lik}, n_compare={n_trials}",
                        ci=sm["fixed_effect_ci95"], label="stratum b1_z"),
                    ratio_component(
                        gm["fixed_effect"], "logit coefficient per SD of log M",
                        at=f"theta={key:.2f}, all qualifying features, "
                           f"{lik}, n_compare={n_trials}",
                        ci=gm["fixed_effect_ci95"], label="global b1_z"),
                ),
            }
            row[lik] = entry

        # THE OUTCOME SCALE, so the coefficient is commensurable with the
        # bound. Computed by `fit_arm` on the stratum's own rows -- NOT the
        # global pp scaled by the coefficient ratio, which would assume the
        # outcome-scale map is linear in the coefficient.
        row["marginal_effect_percentage_points"] = {
            "stratum": arm["marginal_effect_percentage_points"],
            "global": g["marginal_effect_percentage_points"],
        }
        row["marginal_effect_percentage_points_betabinomial"] = {
            "stratum": arm["marginal_effect_percentage_points_betabinomial"],
            "global": g["marginal_effect_percentage_points_betabinomial"],
        }
        row["mean_r_f_stratum"] = float(
            np.mean([r["mean_r_f"] for r in arm["per_seed"]]))
        row["mean_r_f_global"] = g.get("mean_r_f")
        row["decile_shape"] = stratum_decile_shape(
            y_t, logM, keep_stratum, n_trials)
        rows.append(row)

    ok = [r for r in rows if r.get("status") == "OK"]
    summary = {"n_thetas_fitted": len(ok), "n_thetas_requested": len(thetas)}
    if ok:
        for lik in ("binomial", "betabinomial"):
            ratios = [abs(r[lik]["stratum"]["fixed_effect"])
                      / abs(r[lik]["global"]["fixed_effect"])
                      for r in ok
                      if r[lik]["global"]["fixed_effect"]]
            disj = [r[lik]["intervals_disjoint"] for r in ok]
            i2s = [r[lik]["I2_percent_stratum"] for r in ok]
            i2g = [r[lik]["I2_percent_global"] for r in ok]
            summary[lik] = {
                # DERIVED, per gotcha 20: whether the reference-theta reading
                # is a property of the term is a question about the RANGE, so
                # the range is what is reported.
                "abs_ratio_min": min(ratios) if ratios else None,
                "abs_ratio_max": max(ratios) if ratios else None,
                "abs_ratio_by_theta": {f"{r['theta']:.2f}": (
                    abs(r[lik]["stratum"]["fixed_effect"])
                    / abs(r[lik]["global"]["fixed_effect"])
                    if r[lik]["global"]["fixed_effect"] else None) for r in ok},
                "n_thetas_intervals_disjoint": int(sum(disj)),
                "n_thetas_total": len(disj),
                "disjoint_at_every_theta": bool(all(disj)),
                "disjoint_at_no_theta": bool(not any(disj)),
                "I2_percent_stratum_max": max(i2s) if i2s else None,
                "I2_percent_global_max": max(i2g) if i2g else None,
                "I2_zero_both_ways_at_every_theta": bool(all(
                    (a == 0.0 and b == 0.0) for a, b in zip(i2s, i2g))),
                "exceeds_3x_at_any_theta": bool(
                    any(r > 3.0 for r in ratios)) if ratios else None,
            }
        argmins = [r["decile_shape"].get("frac_seeds_argmin_interior")
                   for r in ok if r["decile_shape"].get("status") == "OK"]
        summary["decile_argmin_interior_fraction_by_theta"] = {
            f"{r['theta']:.2f}": r["decile_shape"].get("frac_seeds_argmin_interior")
            for r in ok if r["decile_shape"].get("status") == "OK"}
        summary["decile_argmin_interior_fraction_min"] = (
            min(argmins) if argmins else None)
        summary["decile_argmin_interior_fraction_max"] = (
            max(argmins) if argmins else None)

    return {
        "scope": ("EXPLORATORY -- inherits AW section 2a unchanged. This refit "
                  "cannot become primary at ANY theta; adding thresholds does "
                  "not promote it (amendment CD item 2)."),
        "firing_range": [lo, hi],
        "global_side": ("READ from this run's own theta curve, joined on the "
                        "theta VALUE, not refitted and not joined by position"),
        "rows": rows,
        "summary": summary,
    }


def post_hoc_interaction_fit(y: np.ndarray, logM: np.ndarray,
                             log_phi: np.ndarray, d_fs: np.ndarray,
                             keep: np.ndarray, n_trials: int,
                             primary: dict) -> dict:
    """AMENDMENT CD section CD-e. The firing-rate x alignment INTERACTION.

    **`POST_HOC`. NOT pre-registered, NOT a sensitivity arm, NOT a
    specification variant.** The shape this term targets was known before the
    fit -- two instruments had already located concentrated structure -- and no
    labelling changes that. It is excluded from every tally, enters no
    denominator, and FILES NO PREDICTION: a fit whose target shape is already
    known cannot be a prediction.

    **THE PRIMARY IS UNCHANGED AND IS NOT COMPARED FOR PREFERENCE.** The
    primary's coefficients are carried alongside so a reader sees them
    together, and that is the whole of the comparison: no model selection, no
    fit statistic offered as a reason to prefer either, no averaging, and no
    verdict on which model is better. Those prohibitions are AU's and they
    bind here unchanged.

    THE SPECIFICATION, fixed in the amendment before this ran. The primary is
    `logit(r_f) ~ 1 + z(log M) + B_c(log10 phi) + d_fs`, with firing rate
    entering as a PENALISED SPLINE. This adds ONE column,
    `z(log M) x z(log10 phi)`, both standardised within the same per-seed kept
    set the primary standardises over. So the spline remains the main effect of
    firing rate and the new term asks whether the alignment slope varies
    LINEARLY in log firing rate. That is a restriction, it is deliberate --
    one term, one interval, no term selection -- and it is stated because a
    reader must not take a null interaction as evidence against any
    interaction whatever.

    WHAT IS REPORTED IS FIXED IN ADVANCE, REGARDLESS OF OUTCOME: the
    interaction coefficient with its 95% interval under both likelihoods, the
    per-seed values, whether the interval covers zero, and the primary's `b1`
    beside it. **The branch where the interval covers zero is reported exactly
    as the branch where it does not** -- that is the point of fixing this
    before the fit.
    """
    n_seeds = y.shape[0]
    inter = np.zeros_like(logM, dtype=float)
    for s in range(n_seeds):
        m = keep[s]
        lm, lp = logM[s][m], log_phi[s][m]
        z_m = (lm - lm.mean()) / max(float(lm.std()), 1e-300)
        z_p = (lp - lp.mean()) / max(float(lp.std()), 1e-300)
        inter[s][m] = z_m * z_p

    name = "z_logM_x_z_log_phi"
    arm = fit_arm(y, logM, log_phi, d_fs, keep, n_trials,
                  extra={name: inter}, dispersion=True,
                  label="CD_post_hoc_interaction")

    out = {
        "label": "POST_HOC",
        "standing": (
            "POST_HOC. NOT pre-registered, NOT a sensitivity arm, NOT a "
            "specification variant. Excluded from every tally, enters no "
            "denominator, files no prediction. The primary is unchanged and "
            "is NOT compared for preference: no model selection, no fit "
            "statistic offered as a reason to prefer either model, no "
            "averaging with any arm, no promotion on agreement."),
        "term": name,
        "specification": (
            "primary + one column z(log M) x z(log10 phi), both standardised "
            "within the same per-seed kept set as the primary. Firing rate's "
            "MAIN effect remains the penalised centred spline; the added term "
            "tests only a LINEAR-in-log-phi change of the alignment slope."),
        "per_seed": [
            {"seed": r["seed"], "n_features": r["n_features"],
             "interaction": r["extra_coefs"][name],
             "se": r["extra_se"][name],
             "interaction_betabinomial": r["betabinomial"]["extra_coefs"][name],
             "se_betabinomial": r["betabinomial"]["extra_se"][name],
             "b1_z": r["b1_z"]}
            for r in arm["per_seed"]],
    }
    for lik, get in (
        ("binomial", lambda r: (r["extra_coefs"][name], r["extra_se"][name])),
        ("betabinomial", lambda r: (r["betabinomial"]["extra_coefs"][name],
                                    r["betabinomial"]["extra_se"][name])),
    ):
        est = np.array([get(r)[0] for r in arm["per_seed"]])
        se = np.array([get(r)[1] for r in arm["per_seed"]])
        meta = meta_analyse(est, se)
        ci = meta["fixed_effect_ci95"]
        out[lik] = {
            "meta_interaction": meta,
            # DERIVED (gotcha 20), and the outcome CD-e names explicitly.
            "interval_covers_zero": bool(ci[0] <= 0.0 <= ci[1]),
            # The primary's own coefficient, CARRIED not recomputed, so the
            # two sit side by side without a second estimate of the primary
            # existing anywhere (gotcha 16).
            "primary_b1_z_unchanged": (
                primary.get("meta_b1_z" if lik == "binomial"
                            else "meta_b1_z_betabinomial", {})
                .get("fixed_effect")),
            "primary_b1_z_ci95_unchanged": (
                primary.get("meta_b1_z" if lik == "binomial"
                            else "meta_b1_z_betabinomial", {})
                .get("fixed_effect_ci95")),
        }
    out["no_verdict_on_which_model_is_better"] = (
        "By construction: this object stores no comparative fit statistic, "
        "and CD-e forbids offering one.")
    return out


def flat_reachability_at_real_cell_sizes(
        cell_sizes: list[int], n_trials: int, strata_edges: list[float],
        n_reps: int = 20, base_p: float = 0.5) -> dict:
    """AMENDMENT AT §A3 item 2. Does the panel PRODUCE `FLAT_WITHIN_CI` at the
    sizes the real cells actually have?

    `tests/test_design3_join.py` plants a null at n = 6000 and requires FLAT,
    which establishes reachability ON A FIXTURE. Gotcha 28's operative form is
    that empty output counts as information only if the instrument has been
    shown to produce output when it should -- and "at this scale" is part of
    "when it should", because FLAT's tolerance is data-driven and both the
    tolerance and the sampling spread move with `n`. So the demonstration is
    RE-RUN here at the real minimum, median and maximum scored-cell sizes, and
    the rate is reported rather than a pass/fail.

    A single stratum is used per replicate (constant firing rate), which is
    what isolates the decile logic from the stratification.
    """
    lo = strata_edges[len(strata_edges) // 2]
    hi = strata_edges[len(strata_edges) // 2 + 1]
    fire_value = float(np.sqrt(max(lo, 1e-12) * hi))
    out = []
    for n in cell_sizes:
        counts = {"FLAT_WITHIN_CI": 0, "NEITHER": 0,
                  "MONOTONE_INCREASING": 0, "MONOTONE_DECREASING": 0,
                  "UNDEFINED_TOO_FEW_DECILES": 0}
        scored = 0
        for rep in range(n_reps):
            rng = np.random.default_rng(90_000 + 1000 * rep + n)
            lm = rng.normal(size=(1, n))
            fr = np.full((1, n), fire_value)
            yy = rng.binomial(n_trials, base_p, size=(1, n)).astype(float)
            pan = model_free_panel(yy, lm, fr, np.ones((1, n), bool), n_trials,
                                   list(strata_edges))
            for st in pan["per_seed"][0]["strata"]:
                if "shape" in st:
                    counts[st["shape"]] = counts.get(st["shape"], 0) + 1
                    scored += 1
        out.append({
            "n_features": int(n), "n_reps": n_reps, "n_scored_cells": scored,
            "counts": counts,
            "flat_rate": (counts["FLAT_WITHIN_CI"] / scored) if scored else None,
        })
    rates = [r["flat_rate"] for r in out if r["flat_rate"] is not None]
    return {
        # AMENDMENT AU7: this check is SUPERSEDED and says so on every
        # return. A superseded check carrying no label reads as current,
        # and the label cannot live at the call site -- the call site only
        # runs when cells were scored, so a fixture that scores none could
        # not exercise it (gotcha 16; measured, it survived its mutation).
        "SUPERSEDED_BY_AU7": (
            "This plants r_f ~ Binomial(n_compare, base_p), a null with NO "
            "between-feature dispersion in p, so sd(r_f) = 0.080 against a "
            "measured rho of 0.401. It shows FLAT is reachable under a null "
            "THE DATA DOES NOT RESEMBLE, which is not what gotcha 28 asks "
            "for. Read `permutation_null` instead; this is retained as the "
            "superseded check rather than deleted."),
        "truth": f"a TRUE NULL -- r_f ~ Binomial({n_trials}, {base_p}) "
                 "independent of log M",
        "per_cell_size": out,
        "min_flat_rate": (min(rates) if rates else None),
        # DERIVED (gotcha 20).
        "flat_is_reachable_at_every_real_cell_size": bool(
            bool(rates) and min(rates) > 0.0),
        "reading": (
            "If FLAT fires under a true null at these sizes, the absence of "
            "FLAT in the real panel is a property of the DATA and gotcha 28 "
            "does not apply. If it does not fire, the demonstration on the "
            "n = 6000 fixture does not establish reachability here and the "
            "verdict must be withdrawn in favour of the binned means."),
        "note": (
            "The rate is NOT 1.0 even under a true null -- FLAT is a "
            "tolerance comparison, not a 95% test -- so the residual "
            "NEITHER rate under the null is reported and is the right "
            "baseline against which 100-out-of-100 should be read."),
    }


# The value AC10's arm reported in `design3_join_as.json`, computed on the
# pre-AR9-alignment `_fixfp` smooth artifact. AT §A1 re-points the arm; this is
# the SUPERSEDED number the move is measured against, and it stays on disk.
AS12_SMOOTH_B1Z_ON_FIXFP = 0.004361285152375474
# The factor by which AS10 found the hard-bin C1 median moved under alignment.
# P1's threshold is this number, and the brief's interest is in the RATIO, not
# the verdict: nothing else in the project measures how much a seed-order
# defect contaminates a stratum-AVERAGED quantity.
AS10_C1_DILUTION_FACTOR = 19.7


def _smooth_arm_scorecard(arm: dict, hardbin_b1z: float,
                          source: Path | None) -> dict:
    """AMENDMENT AT §A1's P1 and P2, DERIVED (gotcha 20).

    P1 is a claim about how a defect propagates through a REDUCTION: `mu_s` is
    a stratum-level average over many features, so seed-level mis-pairing
    partially averages out, whereas C1's `|rho|` was a direct per-seed
    correlation with nothing to absorb it. Filed procedure-behaviour (2/4).
    P2 asks only whether the re-pointed arm keeps hard-bin's sign.
    """
    if "meta_b1_z" not in arm:
        return {"status": "NOT_RUN -- the smooth arm did not run",
                "P1": "UNSCORED", "P2": "UNSCORED"}
    new = float(arm["meta_b1_z"]["fixed_effect"])
    old = AS12_SMOOTH_B1Z_ON_FIXFP
    ratio = abs(new - old) / abs(old)
    sign_new = "POSITIVE" if new > 0 else "NEGATIVE" if new < 0 else "ZERO"
    sign_hb = ("POSITIVE" if hardbin_b1z > 0 else
               "NEGATIVE" if hardbin_b1z < 0 else "ZERO")
    return {
        "source_read_by_this_run": str(source) if source else None,
        "superseded_value_on_fixfp": old,
        "aligned_value": new,
        "abs_delta": abs(new - old),
        "relative_move": ratio,
        "ci95_aligned": arm["meta_b1_z"]["fixed_effect_ci95"],
        "P1": {
            "category": "procedure-behaviour (the 2/4 bin)",
            "standing": "SEALED before amendment AT",
            "statistic": "|new - superseded| / |superseded|",
            "threshold": AS10_C1_DILUTION_FACTOR,
            "measured": ratio,
            "the_number_is_more_interesting_than_the_verdict": (
                "this ratio measures how much a seed-order defect contaminates "
                "a STRATUM-AVERAGED quantity, against AS10's 19.7x for a "
                "direct per-seed correlation. Nothing else in the project "
                "measures that contrast."),
            "outcome": "HIT" if ratio < AS10_C1_DILUTION_FACTOR else "MISS",
        },
        "P2": {
            "category": "within-system comparison",
            "standing": "SEALED before amendment AT",
            "statistic": "sign of the aligned smooth b1_z vs hard-bin's",
            "sign_smooth": sign_new,
            "sign_hardbin": sign_hb,
            "hardbin_b1z": hardbin_b1z,
            "outcome": "HIT" if sign_new == sign_hb else "MISS",
        },
    }


def check_degenerate_inheritance(am_flag: np.ndarray,
                                 am_gt: np.ndarray) -> dict:
    """The degenerate flag's unlabelled seed axis may be inherited from the
    gtilde it was built from ONLY if it really was built row-for-row from it.

    An EXACT elementwise comparison of `argmax_concept` is the check, and a
    total or a shape is not: both are invariant under a seed permutation
    (gotcha 27). Extracted from `main()` so the verdict can be exercised in
    BOTH directions -- inline, the surrounding `raise` meant the emitted
    verdict could only ever read `ELEMENTWISE_EQUAL`, which is a reading that
    cannot disagree with its author (gotcha 20).
    """
    same_shape = bool(am_flag.shape == am_gt.shape)
    equal = int(np.count_nonzero(am_flag == am_gt)) if same_shape else 0
    return {
        "inherited_from": "gtilde (hard-bin)",
        "verified_by": "elementwise equality of argmax_concept",
        "same_shape": same_shape,
        "argmax_concept_cells_compared": int(am_gt.size),
        "argmax_concept_cells_equal": equal,
        # DERIVED (gotcha 20).
        "verdict": ("ELEMENTWISE_EQUAL"
                    if same_shape and equal == am_gt.size else "DIFFERS"),
        "why_elementwise": (
            "a total or a shape is invariant under a seed permutation and "
            "could not see one; only a positional comparison can (gotcha 27)"),
    }


def verify_feature_side_seed_provenance(records: list[dict]) -> dict:
    """AMENDMENT AT §A1 item 3, as a testable function rather than inline.

    AS1 corrected three consumers of AR9's split seed axis. This arm is not
    one of them, and the brief requires that be ASSERTED. The assertion is
    POSITIONAL: anchor row `s` must have resolved the row index belonging to
    `anchor_seeds[s]`, and the checkpoint hashes must all be distinct. A count
    of records, or set-equality of the labels, is invariant under a
    permutation and could not see one (gotcha 27).
    """
    problems = []
    n_distinct = len({r["checkpoint_sha256_16"] for r in records})
    if n_distinct != len(records):
        problems.append(
            f"only {n_distinct} distinct checkpoint hashes across "
            f"{len(records)} anchors -- two anchors resolved the same "
            "checkpoint, so at least one seed's covariates are another "
            "seed's")
    for r in records:
        if not r["ckpt_dir_matches_anchor_label"]:
            problems.append(
                f"anchor position {r['anchor_position']} (label "
                f"{r['anchor_label']}) read checkpoint dir {r['ckpt_dir']}")
        if not r["row_index_name_carries_the_ckpt_dir"]:
            problems.append(
                f"anchor {r['anchor_label']}'s row index {r['row_index']} "
                f"does not belong to {r['ckpt_dir']}")
    return {
        "claim": ("this arm does NOT inherit AR9's mis-pairing: the row index "
                  "is resolved by checkpoint IDENTITY for each anchor LABEL, "
                  "never by position in a glob"),
        "per_anchor": records,
        "n_anchors": len(records),
        "n_distinct_checkpoint_hashes": n_distinct,
        "problems": problems,
        # DERIVED (gotcha 20).
        "verdict": "VERIFIED_POSITIONALLY" if not problems else "AMBIGUOUS",
    }


def assert_primary_bit_identical(primary: dict, reference: Path) -> dict:
    """AMENDMENT AT. This run changes a SENSITIVITY arm's input and ADDS
    emissions. The PRIMARY must not move, and that is checked rather than
    hoped for -- the same check AS9 used when it added arms to AR's join.

    Compared elementwise on the per-seed estimates, which is a POSITIONAL
    check; a total or a meta-analysed summary would be invariant under a seed
    permutation and could not see one (gotcha 27).
    """
    if not reference.exists():
        return {"status": "REFERENCE_ABSENT", "reference": str(reference),
                "note": "reported as absent, never treated as agreement "
                        "(gotcha 15)"}
    ref = json.loads(reference.read_text())["primary"]
    fields = ("b1_z", "se_b1_z", "gamma_d_fs", "se_gamma", "deviance",
              "loglik", "lambda", "edf")
    diffs = []
    for a, b in zip(primary["per_seed"], ref["per_seed"]):
        for k in fields:
            if a[k] != b[k]:
                diffs.append({"seed": a["seed"], "field": k,
                              "this_run": a[k], "reference": b[k],
                              "abs_delta": abs(float(a[k]) - float(b[k]))})
    n_rows_compared = min(len(primary["per_seed"]), len(ref["per_seed"]))
    return {
        "reference": str(reference),
        "n_seeds_compared": n_rows_compared,
        "n_fields_per_seed": len(fields),
        "fields": list(fields),
        "n_disagreements": len(diffs),
        "disagreements": diffs[:20],
        "same_length": bool(len(primary["per_seed"]) == len(ref["per_seed"])),
        # DERIVED (gotcha 20).
        "verdict": ("BIT_IDENTICAL" if not diffs
                    and len(primary["per_seed"]) == len(ref["per_seed"])
                    else "DIFFERS"),
        "why": ("AT changes only the smooth SENSITIVITY arm's input and adds "
                "emissions. A primary that moved would mean the change was "
                "not scoped as described."),
    }


def _quantiles(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"n": 0, **{k: float("nan")
                           for k in ("p10", "median", "mean", "p90")}}
    return {"n": int(x.size), "p10": float(np.percentile(x, 10)),
            "median": float(np.median(x)), "mean": float(x.mean()),
            "p90": float(np.percentile(x, 90))}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rf-counts-all-theta", type=Path, required=True)
    ap.add_argument("--rf-counts", type=Path, required=True,
                    help="the single-theta artifact, re-checked BIT-IDENTICAL "
                         "at the reference theta by the consumer as well as "
                         "by the producer")
    ap.add_argument("--m1-json", type=Path, required=True,
                    help="the N=39 M1 artifact, for X2's QUALIFYING theta set")
    ap.add_argument("--gtilde", type=Path, required=True,
                    help="hard-bin gtilde -- the predictor of record")
    ap.add_argument("--gtilde-smooth", type=Path, default=None,
                    help="AC10's smooth-normaliser arm. MUST carry one of "
                         f"{SMOOTH_GENERATION_MARKERS} (CODEBASE 16e, "
                         "amendment AT A1); the job refuses otherwise")
    ap.add_argument("--compare-primary-to", type=Path, default=None,
                    help="an earlier join artifact whose PRIMARY arm this run "
                         "must reproduce bit-identically. AT changes only a "
                         "sensitivity arm's input and adds emissions, so a "
                         "moved primary means the change was not scoped as "
                         "described.")
    ap.add_argument("--stratified-panel-permutation", action="store_true",
                    help="AMENDMENT AV section 5c: also permute the alignment "
                         "predictor WITHIN firing-rate-stratum x "
                         "homology-density blocks, to test whether AU14's "
                         "unexplained 1.27x decile-spread excess survives "
                         "controlling for d_fs. Off by default so AU's "
                         "artifact stays reproducible.")
    ap.add_argument("--firing-stratum-refit", type=float, nargs=2,
                    default=None, metavar=("LO", "HI"),
                    help="AMENDMENT AW section 4, item 3 -- EXPLORATORY "
                         "(§2a: cannot become primary under any outcome). "
                         "Refit the primary model with `keep` additionally "
                         "restricted to fire_rate in [LO, HI), and compare "
                         "the beta-binomial b1 against the global fit. Off "
                         "by default so AV's artifact stays reproducible.")
    ap.add_argument("--firing-stratum-refit-all-theta", action="store_true",
                    help="AMENDMENT CD item 2 -- EXPLORATORY, and it inherits "
                         "AW §2a unchanged (cannot become primary at ANY "
                         "theta). Requires --firing-stratum-refit. Refits that "
                         "stratum at EVERY qualifying theta and reports the "
                         "stratum and global coefficients side by side under "
                         "both likelihoods, with intervals, the ratio through "
                         "ratio_rule at each point, and the within-stratum "
                         "decile shape. Off by default so AW's artifact stays "
                         "reproducible.")
    ap.add_argument("--post-hoc-interaction", action="store_true",
                    help="AMENDMENT CD section CD-e -- OPENLY POST_HOC. Fit "
                         "the primary plus a z(log M) x z(log10 phi) "
                         "interaction and report the term with its interval "
                         "under both likelihoods, the primary's coefficients "
                         "unchanged beside it, and nothing else. NOT "
                         "pre-registered, NOT a sensitivity arm, files no "
                         "prediction, and no verdict on which model is "
                         "better. Off by default.")
    ap.add_argument("--null-reductions", type=Path, required=True)
    ap.add_argument("--d-fs", type=Path, required=True)
    ap.add_argument("--degenerate-flag", type=Path, default=None)
    ap.add_argument("--labels", type=Path, default=None,
                    help="feature-side arm: the label store")
    ap.add_argument("--labels-json", type=Path, default=None)
    ap.add_argument("--row-index-cache", type=Path, default=None)
    ap.add_argument("--ckpts", type=Path, default=None)
    ap.add_argument("--eval-cache", type=Path, default=None)
    ap.add_argument("--scored-row-set", default="label_store")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--skip-feature-side", action="store_true",
                    help="omit AL8b's feature-side arm (it is the only part "
                         "that streams the row index). The output RECORDS the "
                         "omission rather than silently lacking the arm.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                    help="configs/mvp.yaml, whose "
                         "confound_control.analysis_min_firing_rows must equal "
                         "ANALYSIS_MIN_FIRING_ROWS (amendment CR). Defaults to "
                         "the repository's own copy.")
    args = ap.parse_args()

    # Amendment CR, Row C: before any artifact is opened.
    assert_floor_matches_config(args.config)

    from src.sae.train import git_dirty, git_sha

    print("*** X7 FIREWALL CROSSING -- DELIBERATE, SCOPED (the design-3 join) "
          "***", flush=True)

    # ---- gates, before anything is fitted --------------------------------
    problems: list[str] = []

    if args.gtilde_smooth is not None and not any(
            mk in args.gtilde_smooth.name for mk in SMOOTH_GENERATION_MARKERS):
        problems.append(
            f"--gtilde-smooth is {args.gtilde_smooth.name}, which carries none "
            f"of the admissible generation markers {SMOOTH_GENERATION_MARKERS}. "
            "The pre-AO6 smooth artifacts were built on a misaligned protein "
            "grouping (CODEBASE 15d/16e) and the pre-AR9 ones on a mis-paired "
            "seed axis; refusing to read the nearest file.")

    sources = {"gtilde": args.gtilde, "null_reductions": args.null_reductions,
               "d_fs": args.d_fs}
    if args.gtilde_smooth is not None:
        sources["gtilde_smooth"] = args.gtilde_smooth
    if args.degenerate_flag is not None:
        sources["degenerate_flag"] = args.degenerate_flag
    # `d_fs` publishes truncation_policy / scored_row_set but not the concept
    # axis, so it is compared on the keys it can carry -- demanding equality
    # against an absent value is a gate that cannot pass.
    ident_full = check_shared_identity(
        {k: v for k, v in sources.items() if k != "d_fs"},
        SHARED_IDENTITY_KEYS)
    ident_pop = check_shared_identity(
        sources, ("truncation_policy", "scored_row_set", "n_rows"))
    problems += ident_full["problems"] + ident_pop["problems"]

    if problems:
        raise SystemExit("PRE-FLIGHT FAILED -- refusing to run the join.\n  * "
                         + "\n  * ".join(problems))
    print("[join] identity OK on "
          + ", ".join(SHARED_IDENTITY_KEYS + ("n_rows",)), flush=True)

    # ---- inputs ----------------------------------------------------------
    allt = load_counts_all_theta(args.rf_counts_all_theta)
    n_trials = allt["meta"]["n_compare_seeds"]
    anchor_seeds = allt["meta"]["seed_labels"]
    if allt["meta"]["similarity"] != "decoder_cosine":
        raise SystemExit("FATAL: r_f counts were not built on decoder_cosine "
                         "(amendment AB decision B).")

    ref_check = compare_counts_to_reference(
        allt["counts"][round(THETA_LITERATURE, 2)], args.rf_counts,
        THETA_LITERATURE, n_trials, anchor_seeds)
    print("[join] reference-theta slice vs the single-theta artifact: "
          f"{ref_check['verdict']}", flush=True)
    if ref_check["verdict"] != "BIT_IDENTICAL":
        raise SystemExit(
            "FATAL: the all-theta artifact's reference slice is not "
            "bit-identical to the single-theta artifact every earlier "
            "consumer read. Refusing to fit two different outcomes under one "
            "name.\n" + json.dumps(ref_check, indent=2))

    theta_info = qualifying_thetas(args.m1_json)
    if theta_info["n_compare_seeds_in_m1"] != n_trials:
        raise SystemExit(
            f"FATAL: M1 artifact is n_compare={theta_info['n_compare_seeds_in_m1']} "
            f"but the counts are n_compare={n_trials}.")
    thetas = theta_info["qualifying"]
    print(f"[join] n_compare = {n_trials}; qualifying thetas {thetas}; "
          f"reference {THETA_LITERATURE}", flush=True)

    with h5py.File(args.gtilde, "r") as hf:
        M_hard = hf["M"][:]
        concepts = [c.decode() if isinstance(c, bytes) else c
                    for c in hf["concepts"][:]]
        n_rows = int(hf.attrs["n_rows"])
    edges = firing_rate_edges(args.gtilde)

    with h5py.File(args.null_reductions, "r") as hf:
        fire_rate = hf["fire_rate"][:]
        null_seeds = [s.decode() if isinstance(s, bytes) else str(s)
                      for s in hf["seeds"][:]]

    with h5py.File(args.d_fs, "r") as hf:
        d_fs = hf["d_fs_cluster_entropy"][:]        # DEFINITION A (AQ2)
        d_retained = hf["retained_at_floor"][:]
        d_seeds = [int(s) for s in hf["seeds"][:]]
        d_adopted = str(hf.attrs["adopted_definition"])
    if d_adopted != "A_cluster_entropy":
        raise SystemExit(f"FATAL: d_fs.h5 says adopted_definition="
                         f"{d_adopted!r}; AQ2 ratified A_cluster_entropy.")

    # ---- SEED ALIGNMENT. This is the join's single largest hazard ---------
    # THE TWO SIDES OF THIS JOIN ARE IN DIFFERENT SEED ORDERS, and 18 of the
    # 20 rows disagree. `null_pilot_compliant.h5` and
    # `feature_ap_main_compliant.h5` store LEXICOGRAPHIC labels
    # (0, 1, 10, 11, ..., 19, 2, ..., 9) because `glob("seed_*")` sorts that
    # way (CODEBASE 7.3); `rf_counts` and `d_fs` are NUMERIC. `gtilde` inherits
    # the lexicographic order from its inputs and stores NO labels at all.
    #
    # Only rows 0 and 1 coincide between the two orders, so a positional join
    # would mis-pair 18 seeds and produce a complete, plausible set of numbers
    # -- gotcha 18 exactly, on the seed axis rather than the feature axis.
    # Everything is therefore ALIGNED to the anchor order, and the unlabelled
    # axis is resolved FROM THE DATA rather than assumed.
    anchor_labels = [str(s) for s in anchor_seeds]
    null_labels = [s.split("_")[-1] for s in null_seeds]
    gt_order = resolve_gtilde_seed_order(args.gtilde, args.null_reductions)
    print(f"[join] gtilde seed order resolved POSITIONALLY: "
          f"{gt_order['labels']} (min margin "
          f"{gt_order['min_margin_observed']:.1f}x)", flush=True)

    seed_alignment = {
        "anchor_order_rf_counts": anchor_labels,
        "null_reductions_order": null_labels,
        "d_fs_order": [str(s) for s in d_seeds],
        "gtilde_order_resolved": gt_order,
        "n_rows_that_would_be_mispaired_by_a_positional_join": int(sum(
            a != b for a, b in zip(anchor_labels, gt_order["labels"]))),
    }
    print("[join] a positional join would mis-pair "
          f"{seed_alignment['n_rows_that_would_be_mispaired_by_a_positional_join']}"
          f" of {len(anchor_labels)} rows", flush=True)

    fire_rate = align_to_reference(null_labels, fire_rate, anchor_labels)
    M_hard = align_to_reference(gt_order["labels"], M_hard, anchor_labels)
    d_fs = align_to_reference([str(s) for s in d_seeds], d_fs, anchor_labels)
    d_retained = align_to_reference([str(s) for s in d_seeds], d_retained,
                                    anchor_labels)
    # `align_to_reference` already raises if the seed SETS differ rather than
    # only their order, which is the check that matters -- a genuinely missing
    # seed must never be silently dropped by an index lookup that skips it.

    for nm, arr in (("M_hard", M_hard), ("fire_rate", fire_rate),
                    ("d_fs", d_fs)):
        if arr.shape != (len(anchor_seeds), M_hard.shape[1]):
            raise SystemExit(f"FATAL: {nm} has shape {arr.shape}, expected "
                             f"{(len(anchor_seeds), M_hard.shape[1])}")

    logM_hard = log_M(M_hard, str(args.gtilde))
    log_phi = np.log10(np.maximum(fire_rate, 1e-12))

    fl = floor_report(fire_rate, n_rows, edges, logM_hard, d_fs, d_retained)
    keep, firing_bin = fl.pop("keep"), fl.pop("firing_bin")
    print(f"[join] A4a floor at {ANALYSIS_MIN_FIRING_ROWS} rows "
          f"({fl['firing_rate_threshold']:.3e}) excludes "
          f"{fl['total_excluded']} of {keep.size} feature-seed cells", flush=True)

    # AL9a/AL12's OTHER empty set. `log M` is already refused upstream by
    # `log_M`, so there is deliberately NO second gate for it here: a branch
    # that can never be reached is the mirror of a check that can only pass
    # (gotcha 25), and it would read as coverage it does not provide.
    if fl["n_nonfinite_d_fs_among_retained"]:
        raise SystemExit(
            f"FATAL: {fl['n_nonfinite_d_fs_among_retained']} non-finite d_fs "
            "cells survive the floor. An unfired feature's entropy is NaN by "
            "design (CODEBASE 15b); SPECIFICATION §3 requires this be "
            "confirmed rather than assumed.")

    y_ref = allt["counts"][round(THETA_LITERATURE, 2)].astype(np.float64)

    # ---- PRIMARY: hard-bin, reference theta -------------------------------
    print("[join] PRIMARY arm (hard-bin, theta = "
          f"{THETA_LITERATURE})", flush=True)
    primary = fit_arm(y_ref, logM_hard, log_phi, d_fs, keep, n_trials,
                      deviance_drops=True, dispersion=True,
                      label="primary")

    # AL8a's assertion, as a measurement, on one representative seed.
    m0 = keep[0]
    inv = standardisation_invariance(y_ref[0][m0], n_trials, logM_hard[0][m0],
                                     log_phi[0][m0], d_fs[0][m0])
    inv["verdict"] = ("INVARIANT" if (inv["rel_residue_b1"] < 1e-6
                                      and inv["abs_residue_deviance"] < 1e-6
                                      and inv["abs_residue_wald"] < 1e-5)
                      else "NOT_INVARIANT -- REPORTABLE BUG (AL8a)")
    primary["standardisation_invariance_seed0"] = inv
    print(f"[join] standardisation invariance: {inv['verdict']}", flush=True)

    # ---- AMENDMENT AW section 4 item 3: firing-stratum-restricted refit ---
    # EXPLORATORY (§2a) -- gated, off by default, cannot become primary.
    firing_stratum_refit_result = None
    if args.firing_stratum_refit is not None:
        lo, hi = args.firing_stratum_refit
        print(f"[join] AW firing-stratum refit: [{lo:.3e}, {hi:.3e})", flush=True)
        firing_stratum_refit_result = firing_stratum_refit(
            y_ref, logM_hard, log_phi, d_fs, fire_rate, keep, n_trials,
            lo, hi, primary)
        if "meta_b1_z_betabinomial" in firing_stratum_refit_result:
            r = firing_stratum_refit_result
            print(f"[join]   stratum b1_z(bb) fixed effect "
                  f"{r['meta_b1_z_betabinomial']['fixed_effect']:+.5f} vs "
                  f"global {r['global_b1_z_betabinomial_reference']:+.5f} "
                  f"(ratio {r['abs_ratio_stratum_over_global']:.3f})", flush=True)

    # ---- dispersion arm + AK9a -------------------------------------------
    print("[join] dispersion arm (beta-binomial, lambda inherited)", flush=True)
    disp_rows = []
    for s in range(len(anchor_seeds)):
        m = keep[s]
        lm = logM_hard[s][m]
        z = (lm - lm.mean()) / max(lm.std(), 1e-300)
        des = build_design(z, log_phi[s][m], d_fs[s][m])
        f = fit_design3(y_ref[s][m], n_trials, des["X"], des["S"])
        bb = fit_betabinomial(y_ref[s][m], n_trials, des["X"],
                              f["lambda"] * des["S"], f["beta"])
        i = des["idx_logM"]
        disp_rows.append({
            "seed": s,
            "b1_z_binomial": float(f["beta"][i]),
            "se_binomial": float(np.sqrt(f["cov"][i, i])),
            "b1_z_betabinomial": float(bb["beta"][i]),
            "se_betabinomial": float(np.sqrt(abs(bb["cov"][i, i]))),
            "rho": bb["rho"], "s": bb["s"],
            "loglik_binomial": f["loglik"], "loglik_betabinomial": bb["loglik"],
            "loglik_gain": bb["loglik"] - f["loglik"],
            "optimiser_success": bb["success"],
            "hessian_is_pd": bb["hessian_is_pd"],
            "AK9a_binomial": endpoint_check(y_ref[s][m], n_trials, f["eta"]),
            "AK9a_betabinomial": endpoint_check(
                y_ref[s][m], n_trials, des["X"] @ bb["beta"], s=bb["s"]),
        })
        print(f"  [disp] seed {s:2d}: rho {bb['rho']:.5f} | b1 binom "
              f"{f['beta'][i]:+.5f} -> betabinom "
              f"{bb['beta'][i]:+.5f} | dll {bb['loglik'] - f['loglik']:+.1f}",
              flush=True)
    dispersion = {
        "per_seed": disp_rows,
        "meta_b1_z_betabinomial": meta_analyse(
            np.array([r["b1_z_betabinomial"] for r in disp_rows]),
            np.array([r["se_betabinomial"] for r in disp_rows])),
        "rho_across_seeds": _quantiles(
            np.array([r["rho"] for r in disp_rows])),
        "se_inflation_betabinom_over_binom": _quantiles(np.array(
            [r["se_betabinomial"] / max(r["se_binomial"], 1e-300)
             for r in disp_rows])),
        "lambda_policy": (
            "INHERITED from the binomial REML fit, not re-selected. "
            "SPECIFICATION §0 makes the binomial primary, so this arm is the "
            "same specification under a wider likelihood rather than a "
            "different specification."),
        "AK9a_note": (
            "The Beta(1,1) recovery check "
            "(tests/test_rf_mixture.py::test_uniform_p_gives_a_uniform_marginal) "
            "verifies the INTERIOR exactly via the Beta integral and is silent "
            "about the atoms. AK9a's endpoint comparison is per seed above, "
            "with the direction DERIVED from the sign of observed - implied."),
    }

    # ---- the theta curve, the primary reported object (AB1 decision A) ----
    print("[join] theta curve across the qualifying range", flush=True)
    curve = []
    for t in thetas:
        if t not in allt["counts"]:
            curve.append({"theta": t, "status": "ABSENT_FROM_ARTIFACT"})
            continue
        arm = fit_arm(allt["counts"][t].astype(np.float64), logM_hard,
                      log_phi, d_fs, keep, n_trials, dispersion=True)
        bb_e = np.array([r["betabinomial"]["b1_z"] for r in arm["per_seed"]])
        bb_s = np.array([r["betabinomial"]["se_b1_z"] for r in arm["per_seed"]])
        bb_a = np.array([r["betabinomial"]["marginal_effect"]["ame"]
                         for r in arm["per_seed"]])
        bb_as = np.array([r["betabinomial"]["marginal_effect"]["se"]
                          for r in arm["per_seed"]])
        curve.append({
            "theta": t,
            "meta_b1_z": arm["meta_b1_z"],
            # AS section 2: the dispersion-corrected arm is computed at
            # EVERY theta, not scaled from the reference by a summary
            # multiplier.
            "meta_b1_z_betabinomial": meta_analyse(bb_e, bb_s),
            "meta_marginal_effect": arm["meta_marginal_effect"],
            "meta_marginal_effect_betabinomial": meta_analyse(bb_a, bb_as),
            "marginal_effect_percentage_points":
                arm["marginal_effect_percentage_points"],
            "marginal_effect_percentage_points_betabinomial":
                arm["marginal_effect_percentage_points_betabinomial"],
            # AMENDMENT AT §A4: THE THETA CURVE FOR `gamma`. `b1` has had a
            # curve since AR; `d_fs` had one summary number at one theta, and
            # E4's 12.56x was read off it. The curve is what says whether that
            # figure is a property of the term or of the reference theta.
            "meta_gamma_d_fs": arm["meta_gamma_d_fs"],
            "meta_gamma_d_fs_betabinomial": arm["meta_gamma_d_fs_betabinomial"],
            "marginal_effect_d_fs_percentage_points":
                arm["marginal_effect_d_fs_percentage_points"],
            "marginal_effect_d_fs_percentage_points_betabinomial":
                arm["marginal_effect_d_fs_percentage_points_betabinomial"],
            "gamma_d_fs_by_seed": [r["gamma_d_fs"] for r in arm["per_seed"]],
            "sd_d_fs_by_seed": [r["sd_d_fs"] for r in arm["per_seed"]],
            "term_ordering": arm["term_ordering"],
            "rho_median": float(np.median(
                [r["betabinomial"]["rho"] for r in arm["per_seed"]])),
            "b1_z_by_seed": [r["b1_z"] for r in arm["per_seed"]],
            "variance_share_median": arm["variance_share_across_seeds"]["median"],
            "d_fs_share_median": arm["d_fs_variance_share"]
                                    ["ratio_d_fs_over_logM"]["median"],
            "mean_r_f": float(np.mean([r["mean_r_f"] for r in arm["per_seed"]])),
            "n_seeds_lambda_at_grid_edge": arm["n_seeds_lambda_at_grid_edge"],
        })
        fe = arm["meta_b1_z"]["fixed_effect"]
        ci = arm["meta_b1_z"]["fixed_effect_ci95"]
        pp = curve[-1]["marginal_effect_percentage_points"]
        print(f"  [curve] theta {t:.2f}: b1_z {fe:+.5f} "
              f"[{ci[0]:+.5f}, {ci[1]:+.5f}] | dr_f {pp['fixed_effect']:+.4f} pp "
              f"| var-share "
              f"{arm['variance_share_across_seeds']['median']:.4f}", flush=True)
    signs = {("POSITIVE" if c.get("meta_b1_z", {}).get("fixed_effect", 0) > 0
              else "NEGATIVE" if c.get("meta_b1_z", {}).get("fixed_effect", 0) < 0
              else "ZERO")
             for c in curve if "meta_b1_z" in c}
    theta_curve = {
        "qualifying_thetas": thetas,
        "selection_rule": {
            "floor": theta_info["floor"],
            "endpoint_max_share": theta_info["endpoint_max_share"],
            "source": "rf_marginal.select_theta, amendment X2 -- read, not "
                      "recomputed, so this module cannot widen the range",
        },
        "rows": curve,
        "sign_is_unanimous_across_qualifying_thetas": bool(len(signs) == 1),
        "signs_observed": sorted(signs),
        "reading": (
            "AB1 decision A and AK6: the full curve is the primary reported "
            "object and reporting the whole set is the declared substitute "
            "for multiplicity control. theta = 0.70 is the reference for "
            "headline numbers, not a selected member."),
    }

    # ---- AMENDMENT CD section CD-e: the OPENLY POST_HOC interaction -------
    # Declared in the amendment BEFORE it was run, with what is reported fixed
    # in advance regardless of outcome -- including the outcome where the
    # term's interval covers zero.
    interaction_result = None
    if args.post_hoc_interaction:
        print("[join] CD-e POST_HOC interaction: z(log M) x z(log10 phi)",
              flush=True)
        interaction_result = post_hoc_interaction_fit(
            y_ref, logM_hard, log_phi, d_fs, keep, n_trials, primary)
        for lik in ("binomial", "betabinomial"):
            e = interaction_result[lik]
            m = e["meta_interaction"]
            print(f"  [inter/{lik}] {m['fixed_effect']:+.6f} "
                  f"[{m['fixed_effect_ci95'][0]:+.6f}, "
                  f"{m['fixed_effect_ci95'][1]:+.6f}] | covers_zero="
                  f"{e['interval_covers_zero']} | primary b1_z "
                  f"{e['primary_b1_z_unchanged']:+.6f} (unchanged)", flush=True)

    # ---- AMENDMENT CD item 2: the stratum refit ACROSS the grid -----------
    # Placed AFTER the theta curve because the global side of every comparison
    # is READ from `curve`, not refitted (see the function's docstring).
    stratum_curve_result = None
    if args.firing_stratum_refit_all_theta:
        if args.firing_stratum_refit is None:
            raise SystemExit(
                "--firing-stratum-refit-all-theta needs --firing-stratum-refit "
                "LO HI: the curve extends that refit across theta and has no "
                "stratum of its own to fall back on.")
        lo, hi = args.firing_stratum_refit
        print(f"[join] CD stratum refit across the grid: "
              f"[{lo:.3e}, {hi:.3e}) at {len(thetas)} thetas", flush=True)
        stratum_curve_result = firing_stratum_refit_curve(
            allt["counts"], thetas, logM_hard, log_phi, d_fs, fire_rate,
            keep, n_trials, lo, hi, curve)
        for r in stratum_curve_result["rows"]:
            if r.get("status") != "OK":
                print(f"  [strat] theta {r['theta']:.2f}: {r['status']}",
                      flush=True)
                continue
            bb = r["betabinomial"]
            print(f"  [strat] theta {r['theta']:.2f}: stratum "
                  f"{bb['stratum']['fixed_effect']:+.6f} "
                  f"[{bb['stratum']['fixed_effect_ci95'][0]:+.6f}, "
                  f"{bb['stratum']['fixed_effect_ci95'][1]:+.6f}] | global "
                  f"{bb['global']['fixed_effect']:+.6f} "
                  f"[{bb['global']['fixed_effect_ci95'][0]:+.6f}, "
                  f"{bb['global']['fixed_effect_ci95'][1]:+.6f}] | disjoint="
                  f"{bb['intervals_disjoint']} | I2 s/g="
                  f"{bb['I2_percent_stratum']:.2f}/"
                  f"{bb['I2_percent_global']:.2f}", flush=True)

    # ---- AMENDMENT AT sections A2 and A4 ---------------------------------
    # Both summaries are computed by named functions rather than inline, so
    # each can be exercised on a series whose answer is KNOWN. Inline, the
    # only fixture available is whatever the fit happens to produce, and a
    # fixture that cannot exhibit the disputed condition cannot test it
    # (gotcha 16) -- measured: seven of nineteen mutations survived while
    # these blocks lived in `main()`.
    rows_ok = [c for c in curve if "meta_b1_z" in c]
    outcome_scale = outcome_scale_bound(rows_ok)
    print("[join] outcome-scale bound: binomial "
          f"{outcome_scale['binomial_PRE_REGISTERED_INTERVAL_SUPERSEDED_AS_A_MAGNITUDE']['max_abs_pp']:.4f}"
          " pp (theta "
          f"{outcome_scale['binomial_PRE_REGISTERED_INTERVAL_SUPERSEDED_AS_A_MAGNITUDE']['at_theta']})"
          " | beta-binomial "
          f"{outcome_scale['betabinomial_HEADLINE_MAGNITUDE']['max_abs_pp']:.4f}"
          f" pp (theta {outcome_scale['betabinomial_HEADLINE_MAGNITUDE']['at_theta']})"
          " | factor "
          f"{outcome_scale['factor_betabinomial_over_binomial']['min']:.3f}-"
          f"{outcome_scale['factor_betabinomial_over_binomial']['max']:.3f}",
          flush=True)
    d_fs_curve = d_fs_grid_summary(rows_ok)
    print("[join] d_fs across the grid: gamma "
          f"{min(r['gamma_binomial_fe'] for r in d_fs_curve['per_theta']):+.5f}.."
          f"{max(r['gamma_binomial_fe'] for r in d_fs_curve['per_theta']):+.5f}"
          " | sign unanimous "
          f"{d_fs_curve['gamma_sign_is_unanimous_across_the_grid']} | "
          f"d_fs/logM {d_fs_curve['sd_ratio_d_fs_over_logM']['min']:.3f}.."
          f"{d_fs_curve['sd_ratio_d_fs_over_logM']['max']:.3f}", flush=True)

    # ---- sensitivity arms, all pre-declared -------------------------------
    sensitivity: dict = {}

    if args.gtilde_smooth is not None:
        # The smooth arm has its own unlabelled seed axis; it is resolved the
        # same way, independently, rather than inheriting the hard-bin answer.
        gts_order = resolve_gtilde_seed_order(args.gtilde_smooth,
                                              args.null_reductions)
        seed_alignment["gtilde_smooth_order_resolved"] = gts_order
        with h5py.File(args.gtilde_smooth, "r") as hf:
            M_smooth = align_to_reference(gts_order["labels"], hf["M"][:],
                                          anchor_labels)
        print(f"[join] sensitivity: SMOOTH normaliser (AC10); seed order "
              f"{gts_order['labels']}", flush=True)
        sensitivity["smooth_normaliser"] = {
            "source": str(args.gtilde_smooth),
            "arm": "AC10, both normalisers -- on the CORRECTED footprint",
            # AMENDMENT AU section 1a: DISPERSION ON EVERY ARM. All four arm
            # figures on record were binomial, whose per-seed SEs are ~3.89x
            # too narrow at rho = 0.401. Whether an arm's interval covers
            # zero cannot be answered by scaling a binomial CI -- the
            # beta-binomial RELOCATES the estimate too.
            **fit_arm(y_ref, log_M(M_smooth, str(args.gtilde_smooth)),
                      log_phi, d_fs, keep, n_trials, dispersion=True,
                      label="smooth"),
        }
    else:
        sensitivity["smooth_normaliser"] = {
            "status": "NOT_RUN -- no --gtilde-smooth supplied"}

    if args.degenerate_flag is not None:
        # ANOTHER unlabelled seed axis. It was built FROM the hard-bin gtilde,
        # so it inherits that order -- and the inheritance is CHECKED rather
        # than assumed, exactly, because the flag artifact carries a copy of
        # `argmax_concept`: an EXACT elementwise comparison against gtilde's
        # own is a positional check, and a total or a shape is not (gotcha 27).
        with h5py.File(args.degenerate_flag, "r") as hf:
            deg_raw = hf["degenerate_argmax"][:]
            am_flag = hf["argmax_concept"][:]
        with h5py.File(args.gtilde, "r") as hf:
            am_gt = hf["argmax_concept"][:]
        # AMENDMENT AT §A1 item 3: the check already existed and would have
        # raised; what was missing is the MEASUREMENT of its passing. A gate
        # that only speaks when it fails leaves "was it checked?" unanswerable
        # from the artifact.
        deg_inherit = check_degenerate_inheritance(am_flag, am_gt)
        if deg_inherit["verdict"] != "ELEMENTWISE_EQUAL":
            raise SystemExit(
                "FATAL: degenerate_argmax_flag.h5's `argmax_concept` is not "
                "elementwise equal to the gtilde's, so it was NOT built row-"
                "for-row from this artifact and its unlabelled seed axis "
                "cannot be inherited from it. Refusing to guess an order.\n"
                + json.dumps(deg_inherit, indent=2, default=_json_default))
        seed_alignment["degenerate_flag_order"] = {
            **deg_inherit, "labels": gt_order["labels"]}
        deg = align_to_reference(gt_order["labels"], deg_raw, anchor_labels)
        keep_nodeg = keep & ~deg
        print(f"[join] sensitivity: DEGENERATE-ARGMAX EXCLUSION "
              f"({int(deg.sum())} of {deg.size} cells)", flush=True)
        sensitivity["degenerate_excluded"] = {
            "source": str(args.degenerate_flag),
            "arm": "AM6.3 / AO8",
            "n_cells_flagged": int(deg.sum()),
            "frac_cells_flagged": float(deg.mean()),
            **fit_arm(y_ref, logM_hard, log_phi, d_fs, keep_nodeg, n_trials,
                      dispersion=True, label="nodegen"),
        }
    else:
        sensitivity["degenerate_excluded"] = {
            "status": "NOT_RUN -- no --degenerate-flag supplied"}

    # ---- feature-side arm (AL8b) -----------------------------------------
    if args.skip_feature_side:
        sensitivity["feature_side"] = {
            "status": "NOT_RUN -- --skip-feature-side was passed. The arm is "
                      "PRE-DECLARED (AL8b, SPECIFICATION §2) and its absence "
                      "is recorded rather than left as a missing key."}
    else:
        need = {"--labels": args.labels, "--row-index-cache": args.row_index_cache,
                "--ckpts": args.ckpts, "--eval-cache": args.eval_cache}
        missing = [k for k, v in need.items() if v is None]
        if missing:
            raise SystemExit(
                "the feature-side arm needs " + ", ".join(missing) +
                " (or --skip-feature-side, which RECORDS the omission)")
        from src.analysis.smooth_mu import assemble_covariates, resolve_row_index
        print("[join] sensitivity: FEATURE-SIDE covariates (AL8b) -- "
              "recomputed per seed from that seed's own row index", flush=True)
        labels_json = args.labels_json or args.labels.with_suffix(".json")
        n_feat = M_hard.shape[1]
        perp = np.empty_like(logM_hard)
        conc = np.empty_like(logM_hard)
        # AMENDMENT AT §A1 item 3. AS1's re-run corrected three consumers of
        # the split seed axis; this arm is NOT one of them, and the brief
        # requires that be ASSERTED rather than assumed. The assertion is
        # POSITIONAL (gotcha 27): anchor row `s` must resolve the row index
        # belonging to `anchor_seeds[s]`, and the 20 checkpoint hashes must be
        # distinct -- a count or a set-equality could not see a permutation.
        feat_prov = []
        for s in range(len(anchor_seeds)):
            ck = args.ckpts / f"seed_{anchor_seeds[s]}"
            ri = resolve_row_index(args.row_index_cache, ck,
                                   args.eval_cache, args.scored_row_set,
                                   args.k, n_rows)
            with h5py.File(ri, "r") as h:
                ck_sha = str(h.attrs.get("checkpoint_sha256_16"))
            feat_prov.append({
                "anchor_position": s,
                "anchor_label": str(anchor_seeds[s]),
                "ckpt_dir": ck.name,
                "row_index": ri.name,
                "checkpoint_sha256_16": ck_sha,
                "ckpt_dir_matches_anchor_label": bool(
                    ck.name == f"seed_{anchor_seeds[s]}"),
                "row_index_name_carries_the_ckpt_dir": bool(ck.name in ri.name),
            })
            cov = assemble_covariates(args.labels, labels_json, ri, concepts,
                                      n_rows, n_feat)["feature"]
            perp[s] = cov["perplexity"]
            conc[s] = cov["concentration"]
            print(f"  [feat] seed {anchor_seeds[s]:2d}: {ri.name}", flush=True)
        feat_axis = verify_feature_side_seed_provenance(feat_prov)
        if feat_axis["problems"]:
            raise SystemExit(
                "FATAL: the feature-side arm's per-anchor provenance does not "
                "check out. Refusing to build covariates on an axis that "
                "cannot be verified.\n" + json.dumps(feat_axis, indent=2))
        sensitivity["feature_side"] = {
            "arm": "AL8b -- perplexity and concentration as covariates",
            "entry": ("LINEAR, matching d_fs. AL8b and SPECIFICATION §2 say "
                      "'enter as covariates' without saying linearly or as "
                      "smooths; the minimal addition is taken and the choice "
                      "is flagged under open_specification_items."),
            "footprint_provenance": (
                "assemble_covariates recomputes perplexity and concentration "
                "from each seed's own row index AT CALL TIME, so any checkout "
                "at or after 518df52 carries the AO6 fix; the recorded git_sha "
                "on this artifact makes that checkable."),
            "seed_axis_provenance_AT_A1_item_3": feat_axis,
            **fit_arm(y_ref, logM_hard, log_phi, d_fs, keep, n_trials,
                      extra={"perplexity": perp, "concentration": conc},
                      dispersion=True, label="featside"),
        }

    # ---- AK8 --------------------------------------------------------------
    print("[join] AK8 companion arm", flush=True)
    ak8 = {"hard_bin": ak8_discovery_set(y_ref, M_hard, fire_rate, firing_bin,
                                         keep, n_trials)}
    if args.gtilde_smooth is not None:
        # AMENDMENT CQ (corrections entry 218): the smooth M's stored seed
        # axis is LEXICOGRAPHIC while y_ref / fire_rate / firing_bin / keep
        # are in the NUMERIC anchor order, so it is aligned exactly as the
        # sensitivity arm aligns the same array above. Passing hf["M"][:] raw
        # here mis-paired 18 of 20 rows in every stored design3_join_*.json.
        with h5py.File(args.gtilde_smooth, "r") as hf:
            M_smooth_ak8 = align_to_reference(gts_order["labels"], hf["M"][:],
                                              anchor_labels)
        ak8["smooth"] = ak8_discovery_set(y_ref, M_smooth_ak8, fire_rate,
                                          firing_bin, keep, n_trials)
        ak8["smooth"]["seed_axis"] = (
            "ALIGNED to the anchor order through gts_order['labels'] "
            "(amendment CQ). Runs before CQ passed this array unaligned; "
            "their smooth block is corrections entry 218.")

    # ---- AK9b -------------------------------------------------------------
    est = np.array([r["b1_z"] for r in primary["per_seed"]])
    se = np.array([r["se_b1_z"] for r in primary["per_seed"]])
    meta = primary["meta_b1_z"]
    n_anchors = len(anchor_seeds)
    shared = n_trials - 1
    ak9b = {
        "quantity": "across-seed SE / inverse-variance fixed-effect SE, for b1_z",
        "measured_at_n_compare": n_trials,
        "ratio": meta["across_seed_se_over_fixed_effect_se"],
        "across_seed_se": meta["across_seed_se"],
        "fixed_effect_se": meta["fixed_effect_se"],
        "dependence_structure": {
            "n_anchors": n_anchors,
            "n_compare": n_trials,
            "shared_comparison_dictionaries_between_two_anchors": shared,
            "shared_fraction": shared / n_trials,
            "shared_fraction_at_n_compare_19": 18 / 19,
            "statement": (
                f"Two anchors share {shared} of their {n_trials} comparison "
                f"dictionaries ({shared / n_trials:.4f}), against 18 of 19 "
                f"({18/19:.4f}) before. The per-seed fits are therefore MORE "
                "positively correlated, not less. Under positive dependence "
                "the naive independent combination UNDERSTATES the variance of "
                "the combined estimate, so a ratio BELOW 1 is the "
                "counterintuitive direction -- which is why AK9b forbids "
                "inheriting the 0.93x rather than merely re-measuring it. All "
                "20 anchors additionally share ONE eval set and ONE label "
                "matrix (F6/D4), a dependence neither combination can see."),
        },
        "inherited_value_must_not_be_used": 0.93,
        "I2_percent": meta["I2_percent"],
        "Q": meta["Q"], "df": meta["df"],
        # DERIVED (gotcha 20).
        "direction": ("ACROSS_SEED_LARGER"
                      if meta["across_seed_se_over_fixed_effect_se"] > 1
                      else "ACROSS_SEED_SMALLER"
                      if meta["across_seed_se_over_fixed_effect_se"] < 1
                      else "EQUAL"),
    }

    # ---- AS section 2: THE THREE INTERVALS, SIDE BY SIDE -----------------
    # The hypothesis the brief raised is that decisions 2 and 3 are ONE
    # question: if the binomial's within-seed variance was understated ~3.9x
    # then I^2 was inflated BY CONSTRUCTION, and the "between-seed
    # heterogeneity" is within-seed overdispersion wearing its clothes. I^2 is
    # a function of the within-study variances, so recomputing it from the
    # BETA-BINOMIAL SEs is the decisive number -- not a scaled interval.
    bb_e = np.array([r["betabinomial"]["b1_z"] for r in primary["per_seed"]])
    bb_s = np.array([r["betabinomial"]["se_b1_z"] for r in primary["per_seed"]])
    meta_bb = meta_analyse(bb_e, bb_s)
    binom_fe = primary["meta_b1_z"]["fixed_effect_ci95"]
    disp_fe = meta_bb["fixed_effect_ci95"]
    across = primary["meta_b1_z"]["across_seed_ci95"]
    ame_pp = primary["marginal_effect_percentage_points"]
    intervals = {
        "reporting_rule": (
            "THE PRE-REGISTERED BINOMIAL INTERVAL IS REPORTED FIRST, with the "
            "dispersion-corrected and across-seed intervals BESIDE it. "
            "Switching which interval is the headline after seeing the result "
            "is what the pre-registration exists to prevent. rho = 0.40 was "
            "not anticipated by SPECIFICATION section 0; this amendment "
            "records the gap and reports all three, and is NOT a licence to "
            "select."),
        "1_binomial_fixed_effect_PRE_REGISTERED": {
            "estimate": primary["meta_b1_z"]["fixed_effect"],
            "se": primary["meta_b1_z"]["fixed_effect_se"],
            "ci95_logit": binom_fe,
            "I2_percent": primary["meta_b1_z"]["I2_percent"],
        },
        "2_dispersion_corrected_fixed_effect": {
            "estimate": meta_bb["fixed_effect"],
            "se": meta_bb["fixed_effect_se"],
            "ci95_logit": disp_fe,
            "I2_percent_RECOMPUTED_FROM_BETABINOMIAL_SES": meta_bb["I2_percent"],
            "Q": meta_bb["Q"], "df": meta_bb["df"],
            "source": "the beta-binomial fits DIRECTLY, not a scaled binomial",
        },
        "3_across_seed_random_effects": {
            "estimate": primary["meta_b1_z"]["across_seed_mean"],
            "se": primary["meta_b1_z"]["across_seed_se"],
            "ci95_logit": across,
        },
        "on_the_outcome_scale_percentage_points": ame_pp,
        "agreement_2_vs_3": interval_agreement(
            disp_fe, across, "dispersion-corrected FE", "across-seed"),
        "I2_binomial_vs_betabinomial": {
            "binomial": primary["meta_b1_z"]["I2_percent"],
            "betabinomial": meta_bb["I2_percent"],
            "note": ("I^2 is a function of the WITHIN-study variances, so an "
                     "understated within-seed SE inflates it by construction. "
                     "The recomputed value is the answer to whether the "
                     "heterogeneity was real."),
        },
        "not_decision_critical": (
            "On the outcome scale every candidate interval sits within "
            f"{max(abs(x) for x in ame_pp['fixed_effect_ci95'] + ame_pp['across_seed_ci95']):.3f} "
            "percentage points of zero, so which interval is the headline does "
            "not change the reading."),
        "AK9a_compounds_this_in_a_stated_direction": (
            "Both families under-predict the endpoint atoms in every seed, so "
            "EVERY interval here -- including the dispersion-corrected one -- "
            "is too narrow. The direction of the misspecification therefore "
            "STRENGTHENS the evidence-of-absence reading rather than "
            "weakening it."),
    }
    print("[join] intervals: binomial "
          f"[{binom_fe[0]:+.6f}, {binom_fe[1]:+.6f}] | dispersion-corrected "
          f"[{disp_fe[0]:+.6f}, {disp_fe[1]:+.6f}] | across-seed "
          f"[{across[0]:+.6f}, {across[1]:+.6f}]", flush=True)
    print(f"[join] I2 binomial {primary['meta_b1_z']['I2_percent']:.1f}% -> "
          f"beta-binomial {meta_bb['I2_percent']:.1f}%", flush=True)

    # ---- AS section 10 / F1's MODEL-FREE PANEL ---------------------------
    from src.analysis.corrector_scan import BIN_EDGES as CORRECTOR_BIN_EDGES
    print("[join] F1's model-free panel (pre-declared in F1, never run)",
          flush=True)
    panel = model_free_panel(y_ref, logM_hard, fire_rate, keep, n_trials,
                             list(CORRECTOR_BIN_EDGES),
                             comparison_pp=outcome_scale[
                                 "binomial_PRE_REGISTERED_INTERVAL_"
                                 "SUPERSEDED_AS_A_MAGNITUDE"]["max_abs_pp"])
    print(f"[join] panel shapes: {panel['shape_counts']} "
          f"statuses {panel['status_counts']}", flush=True)
    ft = panel["flat_tolerance_report"]
    if ft.get("n_scored_cells"):
        print(f"[join] FLAT tolerance: median "
              f"{ft['tolerance_pp_of_r_f']['median']:.4f} pp vs observed "
              f"spread median {ft['observed_spread_pp_of_r_f']['median']:.4f} "
              f"pp vs comparison {ft['comparison_pp']:.4f} pp", flush=True)
        # AMENDMENT AT §A3 item 2. Reachability is RE-DEMONSTRATED at the real
        # scored-cell sizes rather than cited from an n = 6000 fixture.
        sizes = sorted({ft["n_features_per_scored_cell"]["min"],
                        ft["n_features_per_scored_cell"]["median"],
                        ft["n_features_per_scored_cell"]["max"]})
        panel["flat_reachability_at_real_cell_sizes"] = (
            flat_reachability_at_real_cell_sizes(
                sizes, n_trials, list(CORRECTOR_BIN_EDGES)))
        print("[join] FLAT reachability under a BINOMIAL null (SUPERSEDED, "
              f"AU7) at n = {sizes}: "
              + ", ".join(
                  f"{r['n_features']}:{r['flat_rate']:.2f}"
                  for r in panel["flat_reachability_at_real_cell_sizes"]
                  ["per_cell_size"]), flush=True)
        # AMENDMENT AU §3: the null that preserves this data's own dispersion.
        print("[join] permutation null for the panel (AU section 3)",
              flush=True)
        panel["permutation_null"] = permutation_null_for_the_panel(
            y_ref, logM_hard, fire_rate, keep, n_trials,
            list(CORRECTOR_BIN_EDGES))
        if args.stratified_panel_permutation:
            # AMENDMENT AV section 5c. The 1.27x excess AU14 left unexplained,
            # re-measured with homology density BLOCKED. Gated on a flag so
            # AU's artifact is reproducible without it.
            print("[join] STRATIFIED permutation for the panel (AV section 5c)",
                  flush=True)
            panel["stratified_permutation_null"] = (
                stratified_permutation_null_for_the_panel(
                    y_ref, logM_hard, fire_rate, d_fs, keep, n_trials,
                    list(CORRECTOR_BIN_EDGES)))
            sp = panel["stratified_permutation_null"]
            if "arms" in sp:
                for k, a in sp["arms"].items():
                    print(f"[join]   {k:<16} ratio "
                          f"{a['ratio_observed_to_null_median']:.4f}  median "
                          f"pct {a['median_percentile_of_the_observed_spread']:.1f}"
                          f"  cells>p95 {a['n_cells_above_their_own_95th']}"
                          f"/{a['n_cells']}", flush=True)
                print(f"[join]   AV-P3 = {sp['AV_P3']['outcome']} "
                      f"(grid unanimous: {sp['AV_P3']['grid_is_unanimous']})",
                      flush=True)
        pn = panel["permutation_null"]
        if "per_cell" in pn:
            print("[join] FLAT rate under the PERMUTATION null: "
                  f"{pn['flat_rate_under_the_permutation_null']['pooled']:.4f}"
                  " | empirical SE of a decile mean "
                  f"{pn['empirical_se_of_a_decile_mean_pp']['median']:.4f} pp "
                  f"vs tolerance {pn['tolerance_pp']['median']:.4f} pp "
                  f"(ratio {pn['empirical_se_over_tolerance']['median']:.3f}) "
                  "| observed spread median percentile "
                  f"{pn['observed_spread_vs_its_own_null']['median_percentile']:.1f}",
                  flush=True)
    # The RIGHT panel: the fitted partial effect of log M at MEDIAN firing
    # rate, which is the model side of the same comparison.
    m0 = keep[0]
    lm0 = logM_hard[0][m0]
    z0 = (lm0 - lm0.mean()) / max(lm0.std(), 1e-300)
    des0 = build_design(z0, log_phi[0][m0], d_fs[0][m0])
    fit0 = fit_design3(y_ref[0][m0], n_trials, des0["X"], des0["S"])
    grid_z = np.linspace(np.percentile(z0, 1), np.percentile(z0, 99), 25)
    panel["right_panel_fitted_partial_effect_seed0"] = {
        "z_logM_grid": grid_z.tolist(),
        "partial_effect_logit": (fit0["beta"][des0["idx_logM"]]
                                 * grid_z).tolist(),
        "at": "median firing rate (the spline term is additive, so the partial "
              "effect of log M does not depend on where it is evaluated)",
        "b1_z_seed0": float(fit0["beta"][des0["idx_logM"]]),
    }

    # ---- AMENDMENT AU section 1a: the four arms under BOTH likelihoods --
    arms = arm_comparison(primary, sensitivity)
    print("[join] arms covering zero -- binomial "
          f"{arms['binomial']['n_arms_including_zero']}/"
          f"{arms['n_arms_fit_both_ways']} | beta-binomial "
          f"{arms['betabinomial']['n_arms_including_zero']}/"
          f"{arms['n_arms_fit_both_ways']} "
          f"(sensitivity arms "
          f"{arms['betabinomial']['n_sensitivity_arms_including_zero']}/"
          f"{arms['betabinomial']['n_sensitivity_arms']})", flush=True)

    # ---- the prediction scorecard, DERIVED (gotcha 20) --------------------
    b1_meta = meta["fixed_effect"]
    vshare = primary["variance_share_across_seeds"]["median"]
    coll = primary["collinear_share_across_seeds"]["median"]
    order = np.argsort(est)
    extreme = {int(order[0]), int(order[-1])}
    scorecard = {
        "D1_meta_b1_z_strictly_positive": {
            "category": "DIRECTIONAL MECHANISM (the 1/9 bin), filed deliberately",
            "statistic": "inverse-variance fixed-effect meta-estimate of b1_z",
            "measured": b1_meta,
            "ci95": meta["fixed_effect_ci95"],
            "across_seed_mean": meta["across_seed_mean"],
            "across_seed_ci95": meta["across_seed_ci95"],
            "signs_agree_between_combinations": meta["signs_agree"],
            "outcome": "HIT" if b1_meta > 0 else "MISS",
        },
        "D2_variance_share_below_1": {
            "category": "within-system comparison",
            "statistic": "median over seeds of sd(b1*logM)/sd(s_hat(log10 phi))",
            "measured": vshare,
            "range": [primary["variance_share_across_seeds"]["min"],
                      primary["variance_share_across_seeds"]["max"]],
            "outcome": "HIT" if vshare < 1.0 else "MISS",
        },
        "D3_collinear_share_positive": {
            "category": "within-system comparison (WEAK -- close to entailed "
                        "by gotcha 5)",
            "statistic": "median over seeds of joint - (dD_M + dD_phi)",
            "measured": coll,
            "sign": ("POSITIVE" if coll > 0 else "ZERO" if coll == 0
                     else "NEGATIVE"),
            "n_seeds_positive":
                primary["collinear_share_across_seeds"]["n_positive"],
            "n_seeds_negative":
                primary["collinear_share_across_seeds"]["n_negative"],
            "outcome": "HIT" if coll > 0 else "MISS",
        },
        "E3_dispersion_corrected_and_across_seed_agree_and_I2_falls": {
            "category": "within-system comparison (WEAK -- close to an "
                        "entailment check, derived from summary numbers "
                        "already in hand)",
            "statistic": ("both endpoints of the dispersion-corrected FE "
                          "interval within 10% of the across-seed interval's, "
                          "on the max-interval-width scale, AND I2 recomputed "
                          "from the beta-binomial SEs below 50%"),
            "lo_gap_over_width": intervals["agreement_2_vs_3"]["lo_gap_over_width"],
            "hi_gap_over_width": intervals["agreement_2_vs_3"]["hi_gap_over_width"],
            "lo_within_10pct": intervals["agreement_2_vs_3"]["lo_within_10pct"],
            "hi_within_10pct": intervals["agreement_2_vs_3"]["hi_within_10pct"],
            "I2_recomputed": meta_bb["I2_percent"],
            "I2_below_50": bool(meta_bb["I2_percent"] < 50.0),
            "reading_if_MISS": (
                "genuine between-seed heterogeneity in b1, so decisions 2 and "
                "3 stay SEPARATE questions rather than collapsing into one"),
            "outcome": ("HIT" if (intervals["agreement_2_vs_3"]["both_within_10pct"]
                                  and meta_bb["I2_percent"] < 50.0) else "MISS"),
        },
        "E4_d_fs_share_exceeds_logM_share_by_more_than_5x": {
            "category": "within-system comparison",
            "statistic": ("median over seeds of sd(gamma*d_fs) / sd(b1*logM), "
                          "both on the linear-predictor scale AL8a defines"),
            "measured": primary["d_fs_variance_share"]["ratio_d_fs_over_logM"]["median"],
            "range": [primary["d_fs_variance_share"]["ratio_d_fs_over_logM"]["min"],
                      primary["d_fs_variance_share"]["ratio_d_fs_over_logM"]["max"]],
            "d_fs_over_spline_median":
                primary["d_fs_variance_share"]["ratio_d_fs_over_spline"]["median"],
            "logM_over_spline_median": vshare,
            "why_not_gamma_over_b1": (
                "gamma / b1 is a UNIT ARTIFACT: b1 is on a within-seed z scale "
                "and gamma on d_fs's raw scale (gotcha 21). The raw ratio was "
                "19x; this is the same comparison with the units removed."),
            "reading_if_HIT": (
                "reproducibility tracks TRAINING-SET HOMOLOGY DENSITY more than "
                "biological alignment -- J8 arriving from the Stage 1 side. "
                "Reported as a characterised association on the same terms as "
                "b1: no alpha, interval plus sensitivity, NO MECHANISM STORY. "
                "d_fs was included as a confound control; a large gamma is a "
                "finding about the confound, not a hypothesis this session "
                "gets to explain."),
            "outcome": ("HIT" if primary["d_fs_variance_share"]
                        ["ratio_d_fs_over_logM"]["median"] > 5.0 else "MISS"),
        },
        "AT_P1_P2_the_re_pointed_smooth_arm": _smooth_arm_scorecard(
            sensitivity.get("smooth_normaliser", {}), b1_meta,
            args.gtilde_smooth),
        "AU_Q1_at_least_two_sensitivity_arms_cover_zero_under_betabinomial": {
            "category": "within-system comparison",
            "standing": ("SEALED -- the three sensitivity arms were fit with "
                         "dispersion=False before AU, so their beta-binomial "
                         "intervals did not exist. The PRIMARY's answer was "
                         "already on disk (AS13: [+0.006358, +0.017640], "
                         "excludes zero) and is EXCLUDED from the count."),
            "statistic": ("of the three sensitivity arms, how many have a "
                          "beta-binomial fixed-effect interval containing 0"),
            "measured": arms["betabinomial"]["n_sensitivity_arms_including_zero"],
            "of": arms["betabinomial"]["n_sensitivity_arms"],
            "primary_includes_zero_EXCLUDED_FROM_THE_COUNT": next(
                (r["betabinomial"].get("includes_zero")
                 for r in arms["rows"] if r["arm"] == "primary_hard_bin"), None),
            "binomial_comparison": arms["binomial"]["n_arms_including_zero"],
            "reading": (
                "if the arms' intervals all cover zero then 'two of three arms "
                "reverse' describes point estimates each individually "
                "consistent with zero -- materially weaker than 'the arms "
                "disagree', and the weaker true statement is the one to make"),
            "outcome": ("HIT" if arms["betabinomial"]
                        ["n_sensitivity_arms_including_zero"] >= 2 else "MISS"),
        },
        "AU_Q2_FLAT_fires_in_under_5pct_of_permutations": {
            "category": "procedure-behaviour (the 3/5 bin)",
            "standing": ("SEALED -- no permutation null for the panel existed "
                         "before AU. Labelled STRONGLY INDICATED rather than "
                         "uncertain, and filed anyway so it is scored."),
            "statistic": ("pooled fraction of within-cell permutations whose "
                          "spread falls below that cell's FLAT tolerance"),
            "measured": panel.get("permutation_null", {}).get(
                "flat_rate_under_the_permutation_null", {}).get("pooled"),
            "empirical_se_over_tolerance_median": panel.get(
                "permutation_null", {}).get(
                    "empirical_se_over_tolerance", {}).get("median"),
            "why_AT12_did_not_settle_this": (
                "AT12's reachability check planted a BINOMIAL null with no "
                "between-feature dispersion (sd(r_f) = 0.080 against a measured "
                "rho of 0.401), so it demonstrated reachability under a null "
                "the data does not resemble -- gotcha 13 in its original form, "
                "and mine from the immediately preceding amendment"),
            "outcome": ("HIT" if panel.get("permutation_null", {}).get(
                "AU_Q2_flat_rate_below_5pct") else "MISS"),
        },
        "AU_Q3_observed_spread_is_unremarkable_against_its_own_null": {
            "category": "within-system comparison",
            "standing": "SEALED",
            "statistic": ("median over cells of the observed spread's "
                          "percentile in that cell's permutation null"),
            "measured": panel.get("permutation_null", {}).get(
                "observed_spread_vs_its_own_null", {}).get("median_percentile"),
            "n_cells_above_the_95th": panel.get("permutation_null", {}).get(
                "observed_spread_vs_its_own_null", {}).get(
                    "n_cells_above_the_95th"),
            "ratio_observed_to_null_median": panel.get(
                "permutation_null", {}).get(
                    "observed_spread_vs_its_own_null", {}).get(
                        "ratio_observed_to_null_median"),
            "reading_if_HIT": (
                "the observed decile structure is unremarkable against noise, "
                "which RETRACTS AT12's and PROJECT 5.16(c)'s description of a "
                "median 9.02 pp of non-monotone structure as a property of the "
                "data"),
            "outcome": ("HIT" if panel.get("permutation_null", {}).get(
                "AU_Q3_median_percentile_in_25_75") else "MISS"),
        },
        "AT_P3_betabinomial_bound_above_0352pp_and_below_1pp": {
            "category": "within-system comparison",
            "standing": ("SCORED_AGAINST_EXISTING_DATA -- determinable from "
                         "design3_join_as.json before amendment AT was "
                         "committed (AT1). Reported and EXCLUDED from the bin "
                         "tallies; it cannot calibrate a bin it was not blind "
                         "to."),
            "statistic": ("max over the qualifying grid of |meta-analysed "
                          "average marginal effect| under the BETA-BINOMIAL "
                          "fits, in pp of r_f per SD of log M"),
            "measured_pp": outcome_scale["betabinomial_HEADLINE_MAGNITUDE"]
                                        ["max_abs_pp"],
            "at_theta": outcome_scale["betabinomial_HEADLINE_MAGNITUDE"]
                                     ["at_theta"],
            "AS14_binomial_bound_pp": AS14_BINOMIAL_BOUND_PP,
            "exceeds_0352": outcome_scale["comparison_to_AS14"]
                                         ["betabinomial_exceeds_AS14"],
            "below_1pp": outcome_scale["comparison_to_AS14"]
                                      ["betabinomial_below_1_pp"],
            "entailment_note": (
                "The brief labelled the lower half close to entailed -- a "
                "uniform 2.1x would give ~0.74 pp. The factor is NOT uniform "
                "(measured range in outcome_scale_bound), so the lower half "
                "was not entailed and the margin is thin. The UPPER half is "
                "the content: it decides whether 'no material association' "
                "survives as a phrase."),
            "outcome": ("HIT" if (outcome_scale["comparison_to_AS14"]
                                  ["betabinomial_exceeds_AS14"]
                                  and outcome_scale["comparison_to_AS14"]
                                  ["betabinomial_below_1_pp"]) else "MISS"),
        },
        "AT_P4_FLAT_tolerance_is_tighter_than_0352pp": {
            "category": "procedure-behaviour (the 2/4 bin)",
            "standing": ("SCORED_AGAINST_EXISTING_DATA -- arithmetic on the "
                         "panel already inside design3_join_as.json (AT1). "
                         "Reported and EXCLUDED from the bin tallies."),
            "statistic": ("median over scored cells of FLAT's tolerance, "
                          "2 x median decile CI half-width, in pp of r_f"),
            "measured_pp": panel["flat_tolerance_report"]
                                .get("tolerance_pp_of_r_f", {}).get("median"),
            "comparison_pp": panel["flat_tolerance_report"].get("comparison_pp"),
            "observed_spread_median_pp": panel["flat_tolerance_report"]
                                              .get("observed_spread_pp_of_r_f", {})
                                              .get("median"),
            "reachability": panel.get("flat_reachability_at_real_cell_sizes", {})
                                 .get("flat_is_reachable_at_every_real_cell_size"),
            "reading_if_HIT": (
                "the NEITHER verdict is uninformative about the data and the "
                "binned means become the report"),
            "reading_if_MISS": (
                "the tolerance is LOOSER than the effect being characterised, "
                "so NEITHER is a property of the data -- and AS16's "
                "conclusion stands on its own"),
            "outcome": ("HIT" if panel["flat_tolerance_report"].get(
                "tolerance_is_tighter_than_the_comparison") else "MISS"),
        },
        "AT_P5_gamma_sign_unanimous_across_the_grid": {
            "category": "distributional shape (the 14/15 bin)",
            "standing": ("SEALED -- gamma is not emitted per theta by any "
                         "artifact that existed when AT was committed."),
            "statistic": "sign of the meta-analysed gamma at every qualifying theta",
            "gamma_signs_observed": d_fs_curve["gamma_signs_observed"],
            "b1_signs_observed": d_fs_curve["b1_signs_observed"],
            "betabinomial_unanimous":
                d_fs_curve["gamma_sign_is_unanimous_betabinomial"],
            "why_it_is_worth_recording_either_way": (
                "a confound control that is sign-stable across the grid while "
                "the predictor of interest reverses twice is itself a result "
                "about which term the data constrain"),
            "outcome": ("HIT" if d_fs_curve[
                "gamma_sign_is_unanimous_across_the_grid"] else "MISS"),
        },
        "AT_P6_d_fs_sd_ratio_exceeds_logM_at_every_theta": {
            "category": "within-system comparison",
            "standing": ("SCORED_AGAINST_EXISTING_DATA -- the per-theta "
                         "medians of this ratio are stored in "
                         "design3_join_as.json as d_fs_share_median (AT1). "
                         "Reported and EXCLUDED from the bin tallies."),
            "statistic": ("median over anchors of sd(gamma*d_fs)/sd(b1*logM) "
                          "at each qualifying theta"),
            "min_over_grid": d_fs_curve["sd_ratio_d_fs_over_logM"]["min"],
            "min_at_theta": d_fs_curve["sd_ratio_d_fs_over_logM"]["min_at_theta"],
            "max_over_grid": d_fs_curve["sd_ratio_d_fs_over_logM"]["max"],
            "n_thetas_above_1": d_fs_curve["sd_ratio_d_fs_over_logM"]
                                          ["n_thetas_above_1"],
            "n_thetas_below_1": d_fs_curve["sd_ratio_d_fs_over_logM"]
                                          ["n_thetas_below_1"],
            "scope_note_for_E4": (
                "AS15's E4 was measured AT THE REFERENCE THETA and stands "
                "there. This predicate asks whether it is a property of the "
                "TERM or of the reference theta, which E4 did not and could "
                "not address."),
            "outcome": ("HIT" if d_fs_curve["sd_ratio_d_fs_over_logM"]
                        ["exceeds_1_at_every_theta"] else "MISS"),
        },
        "D4_seeds_0_and_1_are_not_both_extreme": {
            "category": "distributional shape (the 12/13 bin)",
            "statistic": "rank of the 20 anchors by b1_z",
            "rank_order_of_seeds": order.tolist(),
            "lowest_b1_seed": int(order[0]),
            "highest_b1_seed": int(order[-1]),
            "b1_z_by_seed": est.tolist(),
            "outcome": "MISS" if extreme == {0, 1} else "HIT",
        },
    }
    print(json.dumps({"scorecard": scorecard}, indent=2), flush=True)

    # ---- output ----------------------------------------------------------
    out = {
        "identity": {
            # The AS run inherited the literal "design3_join_AR" and so
            # mislabelled itself; corrected to name the amendments the run
            # actually implements, in order.
            "measurement": "design3_join",
            "amendment": ("AR (the join) + AS (three intervals, outcome "
                          "scale, d_fs share, F1's panel) + AT (the aligned "
                          "smooth arm, the bound under the headline fit, "
                          "FLAT's tolerance, d_fs across the grid); "
                          "docs/SPECIFICATION.md sections 0-4"),
            "firewall_crossing": "DELIBERATE",
            "firewall_scope": (
                "The design-3 join itself, plus AK8's companion arm. Scoped to "
                "design 3's own inputs; labelled; non-ambient (its own module "
                "and its own job). Nothing about theta, the endpoint cuts or "
                "the specification may be revised from this output -- the "
                "specification record was ratified and dated before it."),
            "model": ("logit(E[r_fs]) = b0 + b1*z(log M_fs) "
                      "+ s(log10 phi_fs) + gamma*d_fs"),
            "likelihood": "binomial", "link": "logit",
            "n_compare_seeds": n_trials,
            "n_anchors": n_anchors,
            "anchor_seed_labels": anchor_seeds,
            "reference_theta": THETA_LITERATURE,
            "spline_k": SPLINE_K,
            "spline_constraint": "sum-to-zero over the observations",
            "smoothing_parameter_selection": "REML over a fixed log grid",
            "lambda_grid": [float(LAMBDA_GRID[0]), float(LAMBDA_GRID[-1]),
                            int(LAMBDA_GRID.size)],
            "d_fs_definition": d_adopted,
            "analysis_min_firing_rows": ANALYSIS_MIN_FIRING_ROWS,
            "inputs": {
                "rf_counts_all_theta": str(args.rf_counts_all_theta),
                "rf_counts": str(args.rf_counts),
                "m1_json": str(args.m1_json),
                "gtilde": str(args.gtilde),
                "gtilde_smooth": (str(args.gtilde_smooth)
                                  if args.gtilde_smooth else None),
                "null_reductions": str(args.null_reductions),
                "d_fs": str(args.d_fs),
                "degenerate_flag": (str(args.degenerate_flag)
                                    if args.degenerate_flag else None),
                "labels": str(args.labels) if args.labels else None,
                "compare_primary_to": (str(args.compare_primary_to)
                                       if args.compare_primary_to else None),
            },
            "smooth_arm_generation": {
                "admissible_markers": list(SMOOTH_GENERATION_MARKERS),
                "marker_present": ([mk for mk in SMOOTH_GENERATION_MARKERS
                                    if args.gtilde_smooth
                                    and mk in args.gtilde_smooth.name]
                                   or None),
                "why_a_set": (
                    "the gate refuses the pre-AO6 (misaligned protein "
                    "grouping) and pre-AR9 (mis-paired seed axis) "
                    "generations, both still on disk. A new generation is "
                    "added deliberately; 'any name that is not obviously old' "
                    "would be a gate that can only pass (gotcha 25c)."),
            },
            "shared_identity": ident_full["per_key"],
            "population_identity": ident_pop["per_key"],
            "rf_counts_reference_check": ref_check,
            # amendment CQ, A5: numpy on the host, CPU by construction.
            "device_class": device_class("cpu"),
            "git_sha": git_sha(), "git_dirty": git_dirty(),
            "python_version": platform.python_version(),
        },
        "inference_framing": (
            "A CHARACTERISED ASSOCIATION (AK6). No alpha, no directionality "
            "declaration, no multiplicity correction, no equivalence bound. A "
            "TIGHT INTERVAL AROUND ZERO IS EVIDENCE OF ABSENCE and is to be "
            "read as such, not as a failure to detect. The combined CI "
            "describes precision on THIS eval set -- all 20 seeds share one "
            "label matrix and one eval cache -- not generalisation to a "
            "different Swiss-Prot sample (F1.3, F6, D4)."),
        "seed_alignment": {
            **seed_alignment,
            "why_this_section_exists": (
                "The two sides of this join were in DIFFERENT seed orders. "
                "null_pilot_compliant.h5 and feature_ap_main_compliant.h5 "
                "store LEXICOGRAPHIC labels (0, 1, 10, 11, ..., 19, 2, ..., 9) "
                "because glob('seed_*') sorts that way (CODEBASE 7.3); "
                "rf_counts and d_fs are NUMERIC; and the gtilde artifacts "
                "inherit the lexicographic order while storing NO labels at "
                "all. Only rows 0 and 1 coincide, so a positional join "
                "mis-pairs 18 of 20 seeds and yields a complete, plausible, "
                "wrong result. Every array here is aligned to the ANCHOR "
                "order, and the unlabelled axes are resolved FROM THE DATA."),
        },
        "analysis_set": fl,
        "primary": primary,
        "firing_stratum_refit_aw": (
            firing_stratum_refit_result if firing_stratum_refit_result is not None
            else {"status": "NOT_REQUESTED -- no --firing-stratum-refit supplied"}),
        "post_hoc_interaction_cd_e": (
            interaction_result if interaction_result is not None
            else {"status": "NOT_REQUESTED -- no --post-hoc-interaction "
                            "supplied"}),
        "firing_stratum_refit_curve_cd": (
            stratum_curve_result if stratum_curve_result is not None
            else {"status": "NOT_REQUESTED -- no "
                            "--firing-stratum-refit-all-theta supplied"}),
        "theta_curve": theta_curve,
        "outcome_scale_bound": outcome_scale,
        "arm_comparison": arms,
        "d_fs_across_the_grid": d_fs_curve,
        "primary_vs_reference": (
            assert_primary_bit_identical(primary, args.compare_primary_to)
            if args.compare_primary_to is not None else
            {"status": "NOT_REQUESTED -- no --compare-primary-to supplied"}),
        "intervals": intervals,
        "model_free_panel": panel,
        "dispersion_arm": dispersion,
        "sensitivity": sensitivity,
        "ak8_discovery_set": ak8,
        "ak9b_pooled_se_ratio": ak9b,
        "prediction_scorecard": scorecard,
        "open_specification_items": [
            {"item": "which meta-analytic interval is the headline",
             "why_open": "F1.3 and AL8a fix 'per-seed fits meta-analysed' and "
                         "fix the estimand; neither fixes the COMBINATION "
                         "RULE. Both are computed and reported.",
             "owner": "the author"},
            {"item": "how the feature-side arm's covariates enter",
             "why_open": "AL8b and SPECIFICATION §2 say 'enter as covariates' "
                         "without saying linearly or as smooths. LINEAR is "
                         "used, matching d_fs.",
             "owner": "the author"},
            {"item": "the dispersion arm's smoothing parameter",
             "why_open": "inherited from the binomial REML fit rather than "
                         "re-selected, so the arm is the same specification "
                         "under a wider likelihood.",
             "owner": "the author"},
        ],
        "decision": (
            "REPORT ONLY. This job runs the join; it settles no specification "
            "question, promotes no design, and makes no recommendation. If the "
            "estimate is at or near zero, that is a result and is reported as "
            "one -- the pre-registration licenses an unbiased effect size and "
            "a specification-robustness claim, and never licensed a positive "
            "finding."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=_json_default))
    print(f"[join] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
