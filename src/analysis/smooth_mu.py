"""
M4 -- SMOOTH-TREND `mu_s`. PREREGISTRATION amendment U3, implemented.

WHAT THIS REPLACES. `mu_s` has been the null MEAN pooled over hard strata
(concept cell x firing bin). U3 replaces it with a smooth function of four
covariates -- **perplexity** and **concentration** on the feature side,
**prevalence** and **cluster count** on the concept side -- because M3
measured per-feature mean `M_null` spanning **33x-225x within a single firing
bin**, so hard-bin exchangeability fails BY MEASUREMENT. A smooth trend never
requires within-bin exchangeability; it borrows strength locally along a
continuum.

DELIBERATELY MINIMAL, per the simplicity mandate. ONE fit: one additive
penalized-spline model for `mu` over exactly the four covariates U3 names.
No model selection framework, no basis-family comparison, no extra covariates,
no diagnostic suite beyond the minimum sufficient set.

WHY A LOG LINK RATHER THAN A LOG RESPONSE. `mu_hat` is a DENOMINATOR, so it
must be strictly positive -- `exp(X beta)` is positive by construction. And
the response (a per-(concept, feature) mean over 20 null shifts) can be
exactly zero or negative for rare concepts, which `log(y)` cannot take. So
the model is quasi-Poisson-style IRLS: `E[y] = exp(X beta)`, working weights
`mu`. Non-positive `y` is floored at 0 and THE COUNT IS REPORTED, because a
silently floored response is a silently changed model.

"ROBUST" IS AN ANALOGUE AND IS LABELLED AS ONE. U3 cites limma's
`robust=TRUE` (Phipson et al. 2016). limma is R and its robust option
moderates *variances*; here the trend being fitted is a *mean*. What is
implemented is the honest analogue -- one Huber reweighting pass on the
Pearson residuals after convergence, which downweights exactly the
outlier-response case the 22 degenerate concepts create. It is not limma and
is not described as limma.

WHAT THE ADEQUACY CHECK CAN AND CANNOT SAY (gotcha 21, which bites here).
R7's 33x-225x figure is a spread of per-feature mean **M_null**, a MAXIMUM
over concepts. This model fits the per-(feature, concept) **mean**. Those are
different reductions and a statistic measured on one does not license a claim
about the other. So the absorption check here is run on the object the model
actually fits -- the within-firing-bin spread of per-feature mean `y` versus
of per-feature mean `y / mu_hat` -- and the reduction that feeds downstream
(the max) is checked separately, by the three-armed CV comparison. Stated so
the two are not conflated.

=============================================================================
THE BRANCH, DECLARED BEFORE THE RUN (amendment AC)
=============================================================================
U3 says the smooth trend "dissolves" the exchangeability blocker. That is
CONTINGENT on the resulting predictor clearing V1's CV bar, which has never
been tested.

  * smooth CLEARS the 1.084 bar -> it becomes the predictor of record and
    U3's supersession stands.
  * smooth FAILS and hard-bin passes -> the HARD-BIN predictor remains of
    record, U3's supersession is WITHDRAWN, and the exchangeability blocker
    is recorded as UNRESOLVED rather than dissolved. A predictor is not
    adopted because its motivation was better.
  * BOTH fail -> stop and report. No third normalization is searched for.
=============================================================================
PREDICTIONS, committed to git before the run
=============================================================================
  S1. The smooth fit's chosen effective df is strictly between 4 (one
      near-linear term per covariate) and 40 (the unpenalised basis size).
      CATEGORY: within-system comparison.

  S2. Within-firing-bin spread of per-feature mean `y / mu_hat` is BELOW the
      same spread of per-feature mean `y`. Absorption in the right direction
      is the entire point of U3, and a value at or above 1.0 would mean the
      trend achieves nothing on the object it fits.
      CATEGORY: within-system comparison.

  S3. The spread is NOT fully absorbed: the residual within-bin spread stays
      **above 5x**. Reason, and it is arithmetic rather than mechanism -- M3
      measured the best SINGLE covariate (perplexity) leaving a median
      within-decile spread of 98x, and four additive smooths cannot reduce a
      98x spread to nothing unless the covariates jointly determine the
      response, which four scalars over a 703-concept maximum will not.
      CATEGORY: distributional shape.

  S4. `corr(mu_hat, r_f)` is non-zero and NEGATIVE. Feature-side covariates
      predict the null mean (perplexity Spearman -0.560), and the
      stable/unstable literature associates reproducibility with activation
      geometry, so a correlation is expected. Direction is the guess here.
      CATEGORY: **DIRECTIONAL MECHANISM** -- the category with an 0/7 record
      in this project. Filed anyway and labelled, per amendment AB17: the
      category calibrates confidence, it does not gate filing.

  S5. Non-positive per-(concept, feature) null means are present but a
      minority: **below 25%** of the 703 x 10,240 cells.
      CATEGORY: distributional shape.
=============================================================================
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.analysis.artifact_identity import (argmax_pool_mask, device_class,
                                            write_identity)
from src.sae.train import git_dirty, git_sha
from src.analysis.mu_s_degenerate import degenerate_mask
from src.analysis.feature_footprint import footprint, protein_of_row
from src.analysis.row_index_cache import checkpoint_sha256_16
from src.analysis.null_pooling import (
    FIRING_RATE_EDGES,
    MIN_FEATURES_PER_STRATUM,
    bin_index,
    collapse_firing_bins,
)
from src.analysis.standardize_gain import load_axis_values

# The four covariates U3 names, and nothing else.
COVARIATES = ("perplexity", "concentration", "prevalence", "cluster_count")
N_BASIS = 10          # per covariate; 4 x 10 = 40 columns
DEGREE = 3
LAMBDA_GRID = np.logspace(-2, 6, 17)
HUBER_C = 2.0


# ---------------------------------------------------------------------------
# Basis and penalty
# ---------------------------------------------------------------------------

def bspline_design(x: np.ndarray, n_basis: int = N_BASIS,
                   degree: int = DEGREE,
                   knots: np.ndarray | None = None):
    """Cubic B-spline design matrix on quantile-placed interior knots.

    Knot PLACEMENT is discretion, which is why the penalty (not the knots)
    controls smoothness -- the same reasoning amendment AB's spline decision
    records for the regression's firing-rate term.
    """
    from scipy.interpolate import BSpline

    if knots is None:
        n_int = n_basis - degree - 1
        qs = np.linspace(0, 100, n_int + 2)[1:-1]
        lo, hi = np.nanmin(x), np.nanmax(x)
        interior = np.percentile(x[np.isfinite(x)], qs) if n_int > 0 else np.array([])
        interior = np.unique(interior)
        knots = np.concatenate([[lo] * (degree + 1), interior, [hi] * (degree + 1)])
    n_col = len(knots) - degree - 1
    xc = np.clip(x, knots[0], knots[-1])
    B = np.empty((xc.size, n_col))
    for j in range(n_col):
        c = np.zeros(n_col)
        c[j] = 1.0
        B[:, j] = BSpline(knots, c, degree, extrapolate=False)(xc)
    return np.nan_to_num(B), knots


def diff_penalty(n_col: int, order: int = 2) -> np.ndarray:
    D = np.eye(n_col)
    for _ in range(order):
        D = np.diff(D, axis=0)
    return D.T @ D


# ---------------------------------------------------------------------------
# Covariate assembly
# ---------------------------------------------------------------------------

def resolve_row_index(cache_dir: Path, ckpt_dir: Path, eval_cache: Path,
                      scored_row_set: str, k: int, n_rows: int) -> Path:
    """Find a seed's row index by IDENTITY, not by cache key.

    `row_index_cache.cache_path` folds `git_sha()` into the key as
    conservative cache-busting, so a path computed today never matches an
    index written under an earlier SHA -- even though the index's CONTENT
    depends only on (checkpoint, eval cache, scored_row_set, k). Resolving by key
    would therefore force a needless 20-seed rebuild.

    So this matches on the identity attrs the files already carry, which is
    the rule this project applies everywhere else: assert on IDENTITY, never
    on file presence (gotcha 15). Every candidate must agree on checkpoint
    hash, scored_row_set, k and n_rows; if several do they are content-identical
    and the newest is taken, with the choice recorded.
    """
    want = checkpoint_sha256_16(ckpt_dir)
    cands = []
    for f in sorted(cache_dir.glob(f"row_index_{ckpt_dir.name}_*.h5")):
        with h5py.File(f, "r") as h:
            a = h.attrs
            got = (str(a.get("checkpoint_sha256_16")),
                   str(a.get("scored_row_set")), int(a.get("k", -1)),
                   int(a.get("n_rows", -1)))
        if got == (want, scored_row_set, k, n_rows):
            cands.append(f)
    if not cands:
        raise SystemExit(
            f"FATAL: no row index for {ckpt_dir.name} matching "
            f"(ckpt {want}, subset {scored_row_set}, k {k}, n_rows {n_rows}). "
            "Refusing to substitute a differently-scoped index -- a footprint "
            "computed on the wrong population is a wrong covariate.")
    return max(cands, key=lambda f: f.stat().st_mtime)


def assemble_covariates(labels_h5: Path, labels_json: Path, row_index: Path,
                        concepts: list[str], n_rows: int,
                        n_features: int) -> dict:
    """Per-feature and per-concept covariates, on their natural scales.

    Feature side comes from `feature_footprint.footprint` -- reused rather
    than reimplemented, so the covariate M3 measured and the covariate the
    trend uses cannot drift.
    """
    prot_of_row, n_prot = protein_of_row(labels_h5)
    fp = footprint(row_index, prot_of_row, n_prot, n_features)
    vals = load_axis_values(labels_h5, labels_json, concepts, n_rows)

    feat = {
        # log scales: both are counts/effective-counts spanning orders of
        # magnitude, and a spline on the raw scale would spend all its knots
        # in the upper tail.
        "perplexity": np.log10(np.maximum(fp["perplexity"], 1.0)),
        "concentration": fp["concentration"].astype(np.float64),
    }
    con = {
        "prevalence": np.log10(np.maximum(vals["prevalence"], 1e-12)),
        "cluster_count": np.log10(np.maximum(vals["cluster_count"], 1.0)),
    }
    return {"feature": feat, "concept": con, "footprint": fp}


# ---------------------------------------------------------------------------
# The fit -- ONE additive penalised-spline model, log link
# ---------------------------------------------------------------------------

FEATURE_SIDE = ("perplexity", "concentration")
CONCEPT_SIDE = ("prevalence", "cluster_count")


def _blocks(feat: dict, con: dict, pool_idx: np.ndarray,
            covariates: tuple[str, ...] = COVARIATES):
    """B-spline blocks for each covariate, plus the knots for prediction.

    `covariates` is a SUBSET selector, present for exactly one reason: the
    confound check in amendment AC section 2 needs the same model fitted with
    the feature-side covariates dropped, and refitting through the same code
    path is the only way the two coefficients are comparable.
    """
    out, knots, order = {}, {}, []
    for nm in covariates:
        x = feat[nm] if nm in FEATURE_SIDE else con[nm][pool_idx]
        out[nm], knots[nm] = bspline_design(x)
        order.append(nm)
    out["__order__"] = order
    return out, knots


def _order(blocks: dict) -> list[str]:
    return blocks.get("__order__", list(COVARIATES))


def _accumulate(blocks: dict, y: np.ndarray, eta: np.ndarray,
                obs_w: np.ndarray | None):
    """Accumulate X'WX and X'Wz for quasi-Poisson IRLS, chunked by concept.

    `y` and `eta` are (n_concepts, n_features). Rows of the design are
    [B_perp[f] | B_conc[f] | B_prev[c] | B_clus[c]], so for a fixed concept
    the concept-side blocks are constant across features -- which is what
    makes the accumulation cheap and exact rather than binned.
    """
    names = _order(blocks)
    nf = y.shape[1]
    sizes = [blocks[n].shape[1] for n in names]
    P = sum(sizes)
    XtWX = np.zeros((P, P))
    XtWz = np.zeros(P)
    for c in range(y.shape[0]):
        mu = np.exp(eta[c])
        w = mu.copy()
        if obs_w is not None:
            w = w * obs_w[c]
        z = eta[c] + (y[c] - mu) / np.maximum(mu, 1e-30)
        X = np.empty((nf, P))
        o = 0
        for n, k in zip(names, sizes):
            B = blocks[n]
            X[:, o:o + k] = B if n in FEATURE_SIDE else B[c]
            o += k
        Xw = X * w[:, None]
        XtWX += X.T @ Xw
        XtWz += Xw.T @ z
    return XtWX, XtWz, P


def fit_smooth_mu(y: np.ndarray, blocks: dict, n_iter: int = 100,
                  robust: bool = True,
                  lambda_grid=None, tol: float = 1e-8) -> dict:
    """One penalised additive fit with a log link; GCV over one shared lambda.

    `y` is (n_concepts, n_features) non-negative. Returns coefficients, the
    selected lambda and effective df, and the fitted (n_concepts, n_features)
    `mu_hat` -- strictly positive by construction.
    """
    names = _order(blocks)
    sizes = [blocks[nm].shape[1] for nm in names]
    P = sum(sizes)
    Pen = np.zeros((P, P))
    o = 0
    for s in sizes:
        Pen[o:o + s, o:o + s] = diff_penalty(s)
        o += s
    # A tiny ridge keeps the null space of the difference penalty identifiable.
    Pen += 1e-8 * np.eye(P)

    n_obs = y.size
    ybar = max(float(y.mean()), 1e-12)
    eta = np.full(y.shape, np.log(ybar))
    obs_w = None
    chosen: dict = {}

    for stage in ("main", "robust") if robust else ("main",):
        for _ in range(n_iter):
            XtWX, XtWz, _ = _accumulate(blocks, y, eta, obs_w)
            best = None
            for lam in (LAMBDA_GRID if lambda_grid is None else lambda_grid):
                A = XtWX + lam * Pen
                try:
                    beta = np.linalg.solve(A, XtWz)
                except np.linalg.LinAlgError:
                    continue
                # effective df = trace(H) with H = (X'WX + lam P)^-1 X'WX
                edf = float(np.trace(np.linalg.solve(A, XtWX)))
                # GCV on the working response, which is the standard
                # IRLS-GCV construction.
                resid = XtWz - XtWX @ beta
                rss = float(beta @ (XtWX @ beta) - 2 * beta @ XtWz)
                score = (rss + 1e-30) / max((1.0 - edf / n_obs) ** 2, 1e-12)
                if best is None or score < best[0]:
                    best = (score, lam, beta, edf)
            score, lam, beta, edf = best
            new_eta = _predict_eta(blocks, beta)
            shift = float(np.abs(new_eta - eta).max())
            eta = np.clip(new_eta, -60.0, 20.0)
            carry = {k: chosen[k] for k in
                     ("huber_scale", "huber_frac_downweighted")
                     if k in chosen}
            chosen = {"lambda": float(lam), "edf": edf, "gcv": score,
                      "beta": beta.tolist(), "stage": stage, **carry}
            it_used = _ + 1
            if shift < tol:
                break
        # CONVERGENCE IS REPORTED, NOT ASSUMED. A first version capped this
        # loop at 8 iterations and an extreme outlier left it badly
        # under-converged -- fitted mu at the outlier cell was 1550 after 8
        # iterations and 5.96 after 16. An under-converged `mu_hat` is a wrong
        # denominator, and it looked like a converged one. Found by mutation
        # testing: a mutation disabling the robust weights SURVIVED, because
        # the "robust" stage's extra iterations were doing what the weights
        # were being credited for.
        chosen[f"{stage}_n_iter_used"] = int(it_used)
        chosen[f"{stage}_final_shift"] = shift
        chosen[f"{stage}_converged"] = bool(shift < tol)
        if stage == "main" and robust:
            mu = np.exp(eta)
            pr = (y - mu) / np.sqrt(np.maximum(mu, 1e-30))
            s = 1.4826 * np.median(np.abs(pr - np.median(pr)))
            s = max(s, 1e-12)
            a = np.abs(pr) / s
            obs_w = np.where(a <= HUBER_C, 1.0, HUBER_C / np.maximum(a, 1e-12))
            chosen["huber_scale"] = float(s)
            chosen["huber_frac_downweighted"] = float((a > HUBER_C).mean())

    mu_hat = np.exp(eta)
    chosen["n_obs"] = int(n_obs)
    return {"fit": chosen, "mu_hat": mu_hat}


def _predict_eta(blocks: dict, beta: np.ndarray) -> np.ndarray:
    """Additive predictor. Feature-side terms vary along axis 1, concept-side
    along axis 0, so the sum broadcasts to (n_concepts, n_features)."""
    names = _order(blocks)
    feat_part = con_part = None
    o = 0
    for n in names:
        B = blocks[n]
        k = B.shape[1]
        part = B @ beta[o:o + k]
        o += k
        if n in FEATURE_SIDE:
            feat_part = part if feat_part is None else feat_part + part
        else:
            con_part = part if con_part is None else con_part + part
    if feat_part is None:
        return con_part[:, None] * np.ones((1, 1))
    if con_part is None:
        return feat_part[None, :]
    return con_part[:, None] + feat_part[None, :]


def predict_on_axis(beta: np.ndarray, feat: dict, con: dict, knots: dict,
                    covariates: tuple[str, ...] | list[str] = COVARIATES,
                    n_concepts: int | None = None) -> np.ndarray:
    """`mu_hat` on the FULL concept axis, for the real-gain matrix."""
    blocks = {"__order__": list(covariates)}
    for nm in covariates:
        x = feat[nm] if nm in FEATURE_SIDE else con[nm]
        blocks[nm], _ = bspline_design(x, knots=knots[nm])
    eta = _predict_eta(blocks, beta)
    # BROADCAST TO THE FULL GRID. With concept-side covariates only there is
    # no feature-side term, so `_predict_eta` returns (n_concepts, 1); with
    # feature-side only it returns (1, n_features). Those broadcast correctly
    # in arithmetic, which is exactly why the defect was invisible -- but any
    # per-axis reduction (a median over concepts, say) then silently operates
    # on a length-1 axis. Materialise the full shape here.
    n_feat = len(feat[next(iter(FEATURE_SIDE))])
    n_con = n_concepts if n_concepts else len(con[next(iter(CONCEPT_SIDE))])
    eta = np.broadcast_to(eta, (n_con, n_feat))
    return np.exp(np.ascontiguousarray(eta))


# ---------------------------------------------------------------------------
# Adequacy -- the minimum sufficient set, and no more
# ---------------------------------------------------------------------------

def adequacy(y: np.ndarray, mu_hat: np.ndarray, feat: dict, con: dict,
             pool_idx: np.ndarray, fb: np.ndarray, n_fbins: int) -> dict:
    """Residuals against each covariate, and the ABSORPTION check.

    The absorption check is the point of the exercise and the one most likely
    to be skipped: does the trend actually remove the within-firing-bin
    spread that motivated U3? Measured on the object the model FITS (the
    per-(feature, concept) mean), not on the max -- gotcha 21.
    """
    mu = np.maximum(mu_hat, 1e-30)
    pr = (y - mu) / np.sqrt(mu)

    rep: dict = {"residual_vs_covariate": {}}
    per_feature_resid = pr.mean(axis=0)
    per_concept_resid = pr.mean(axis=1)
    for nm, v, r in (("perplexity", feat["perplexity"], per_feature_resid),
                     ("concentration", feat["concentration"], per_feature_resid),
                     ("prevalence", con["prevalence"][pool_idx], per_concept_resid),
                     ("cluster_count", con["cluster_count"][pool_idx],
                      per_concept_resid)):
        ok = np.isfinite(v)
        edges = np.percentile(v[ok], np.arange(10, 100, 10))
        g = np.digitize(v, edges)
        means = [float(r[ok & (g == i)].mean()) if (ok & (g == i)).sum() else None
                 for i in range(10)]
        rep["residual_vs_covariate"][nm] = {
            "decile_mean_pearson_residual": means,
            "max_abs": float(np.nanmax(np.abs([m for m in means if m is not None]))),
        }

    # ---- absorption: within-firing-bin spread, before vs after ----
    raw_pf = y.mean(axis=0)                       # per-feature mean null gain
    nrm_pf = (y / mu).mean(axis=0)                # after dividing by the trend
    before, after = [], []
    for b in range(n_fbins):
        m = fb == b
        if m.sum() < MIN_FEATURES_PER_STRATUM:
            continue
        for src, dst in ((raw_pf, before), (nrm_pf, after)):
            v = src[m]
            v = v[v > 0]
            if v.size >= MIN_FEATURES_PER_STRATUM:
                dst.append(float(v.max() / v.min()))
    med_before = float(np.median(before)) if before else float("nan")
    med_after = float(np.median(after)) if after else float("nan")
    rep["absorption"] = {
        "median_within_firing_bin_spread_before": med_before,
        "median_within_firing_bin_spread_after": med_after,
        "ratio_after_over_before": (med_after / med_before
                                    if med_before > 0 else float("nan")),
        "n_bins_used": len(before),
        "reference_r7_within_firing_bin": "33x-225x on per-feature mean M_null",
        "reference_m3_best_single_covariate": 98.0967025756836,
        # DERIVED verdicts (gotcha 20).
        "S2_absorbs_in_the_right_direction":
            "HIT" if med_after < med_before else "MISS",
        "S3_residual_spread_above_5x":
            "HIT" if med_after > 5.0 else "MISS",
        "gotcha_21_note": (
            "Measured on the per-(feature, concept) MEAN, which is what this "
            "model fits. R7's 33x-225x is a spread of per-feature mean "
            "M_NULL, a maximum over concepts -- a different reduction. The "
            "max is checked downstream by the three-armed CV comparison."),
    }
    return rep


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ap", type=Path, required=True)
    ap.add_argument("--null-raw", type=Path, required=True)
    ap.add_argument("--null-reductions", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--labels-json", type=Path, default=None)
    ap.add_argument("--row-index-cache", type=Path, required=True,
                    help="directory of row_index_seed_*.h5")
    ap.add_argument("--ckpts", type=Path, required=True,
                    help="parent directory containing seed_N/")
    ap.add_argument("--eval-cache", type=Path, required=True)
    ap.add_argument("--scored-row-set", default="label_store")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--arm", default="b", choices=["a", "b"])
    ap.add_argument("--stratum", default="pooled")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="default: all seeds in the null")
    ap.add_argument("--no-robust", action="store_true")
    ap.add_argument("--exclude-degenerate", action="store_true",
                    help="AC section 3's smooth-WITHOUT arm: drop the "
                         "degenerate concepts from mu ESTIMATION while they "
                         "remain in the max pool. The criterion is imported "
                         "from mu_s_degenerate so the two cannot drift.")
    ap.add_argument("--out", type=Path, required=True,
                    help="h5 with mu_hat per seed on the FULL concept axis")
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    labels_json = args.labels_json or args.labels.with_suffix(".json")

    with h5py.File(args.ap, "r") as hf:
        g = hf[args.stratum]
        concepts = [c.decode() if isinstance(c, bytes) else c
                    for c in g["concepts"][:]]
        n_rows = int(hf.attrs["n_rows"])
        n_features = g["ap"].shape[1]

    pool_mask = np.array(argmax_pool_mask(concepts))
    pool_idx = np.flatnonzero(pool_mask)
    pool_names = [concepts[i] for i in pool_idx]

    # ---- WHICH SEED IS ROW s? (amendment AS; the AR9 blast radius) --------
    # THIS WAS THE SECOND DEFECT AR9 NAMES, and it is subtler than the C1 one
    # because both halves look local and correct. The response `raw[s]` is read
    # from the null, whose seed axis is LEXICOGRAPHIC (`glob("seed_*")` sorts
    # `0, 1, 10, 11, ..., 19, 2, ..., 9`); the feature-side covariates were then
    # taken from the checkpoint at `ckpts / f"seed_{s}"`, a NUMERIC label. Only
    # rows 0 and 1 agree, so for 18 of 20 seeds `perplexity` and
    # `concentration` came from a DIFFERENT SAE than the null response they
    # were fitted against.
    #
    # The fix is to look the checkpoint up by the LABEL of row s, not by s.
    from src.analysis.seed_axis import stored_seed_labels
    seed_labels = stored_seed_labels(args.null_reductions)
    if seed_labels is None:
        raise SystemExit(
            "FATAL: the null reductions carry no seed labels, so row s cannot "
            "be mapped to a checkpoint. Refusing to assume s -> seed_s -- that "
            "assumption is exactly the AR9 defect (gotcha 15).")

    def _feat_for_row(s: int) -> dict:
        """Covariates from the checkpoint that row `s` of the null actually is."""
        ri = resolve_row_index(args.row_index_cache,
                               args.ckpts / f"seed_{seed_labels[s]}",
                               args.eval_cache, args.scored_row_set, args.k,
                               n_rows)
        return assemble_covariates(args.labels, labels_json, ri, concepts,
                                   n_rows, n_features)

    # Concept-side covariates are seed-independent; feature-side are NOT.
    # A feature index is not a cross-seed identity (gotcha 18), so perplexity
    # and concentration are recomputed PER SEED from that seed's own row
    # index. Reusing seed 0's would be the exact error gotcha 18 names.
    cov0 = _feat_for_row(0)
    con = cov0["concept"]
    print(f"[m4] seed labels for the null's rows: {seed_labels}", flush=True)
    print(f"covariates: {n_features} features x {len(concepts)} concepts "
          f"(pool {len(pool_names)}); feature side is PER SEED", flush=True)

    with h5py.File(args.null_raw, "r") as hf:
        null_concepts = [c.decode() if isinstance(c, bytes) else c
                         for c in hf["concepts"][:]]
        if list(null_concepts) != pool_names:
            raise SystemExit(
                "FATAL: the null's concept axis is not the AP matrix's argmax "
                f"pool (null={len(null_concepts)}, pool={len(pool_names)}).")
        raw = hf[f"gain_{args.arm}"]
        n_seeds = raw.shape[0]
        seeds = args.seeds if args.seeds else list(range(n_seeds))

        with h5py.File(args.null_reductions, "r") as rf:
            fire_rate = rf["fire_rate"][:]
        fr_edges = collapse_firing_bins(fire_rate, FIRING_RATE_EDGES,
                                        MIN_FEATURES_PER_STRATUM)
        n_fbins = len(fr_edges) - 1

        report: dict = {"per_seed": {}, "seeds": seeds}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.out, "w") as out:
            out.attrs["measurement"] = "smooth_mu_M4"
            # THE SEED AXIS IS NOW LABELLED (amendment AS decision 5). Its
            # absence is what made AR9 invisible: a consumer could not read
            # the order off the file and defaulted to position.
            out.create_dataset("seeds", data=np.array(
                seed_labels, dtype=h5py.special_dtype(vlen=str)))
            out.attrs["seed_label_source"] = str(args.null_reductions)
            out.attrs["amendment"] = "U3 / AC"
            out.attrs["covariates"] = json.dumps(list(COVARIATES))
            out.attrs["n_basis_per_covariate"] = N_BASIS
            out.attrs["arm"] = args.arm
            out.attrs["robust"] = (not args.no_robust)
            out.attrs["n_concepts"] = len(concepts)
            out.attrs["argmax_pool_n"] = len(pool_names)
            # THE IDENTITY SET (I3). This artifact carried `measurement`,
            # `arm`, `n_concepts` and `argmax_pool_n` and nothing else -- no
            # truncation_policy, no library_sha, no floor parameters -- while
            # feeding AC10's committed sensitivity arm and the C1 measurement.
            # A consumer could not assert what population or label store it was
            # built against, so a mu_hat from a different store would have been
            # accepted silently. Copied from the label store the same way
            # standardize_gain does it, so there is one convention.
            with h5py.File(args.labels, "r") as _lab:
                _lab_attrs = {k: v for k, v in _lab.attrs.items()}
            for _k in ("truncation_policy", "n_rows", "retention_floor_mode",
                       "retention_min_residues", "retention_min_instances",
                       "min_protein_count"):
                if _k in _lab_attrs:
                    out.attrs[_k] = _lab_attrs[_k]
            _sha = _lab_attrs.get("git_sha")
            write_identity(
                out,
                library_sha=(_sha.decode() if isinstance(_sha, bytes) else _sha),
                labels=str(args.labels),
                scored_row_set=args.scored_row_set,
                exclude_pooled_parents=True,
                null_arm=args.arm,
                ap_source=str(args.ap),
                null_raw_source=str(args.null_raw),
                # amendment CQ, A5: numpy IRLS on the host, CPU by construction.
                device_class=device_class("cpu"),
                git_sha=git_sha(), git_dirty=bool(git_dirty()))
            dset = out.create_dataset(
                "mu_hat_full", shape=(n_seeds, len(concepts), n_features),
                dtype="float32", compression="gzip", compression_opts=1)

            for s in seeds:
                # BY LABEL, never by position (amendment AS).
                feat = (cov0["feature"] if s == 0
                        else _feat_for_row(s)["feature"])
                blocks, knots = _blocks(feat, con, pool_idx)
                slab = raw[s].astype(np.float64)          # (shifts, pool, feat)
                ybar = slab.mean(axis=0)                   # (pool, feat)
                n_neg = int((ybar < 0).sum())
                n_zero = int((ybar == 0).sum())
                y = np.maximum(ybar, 0.0)

                # AC3's smooth-WITHOUT arm. The concepts leave the FIT only;
                # they stay in the max pool, which is what R10's
                # flag-don't-gate policy requires. Predictions are still made
                # for them, from the trend the other concepts define.
                deg = degenerate_mask(slab) if args.exclude_degenerate                     else np.zeros(y.shape[0], dtype=bool)
                if deg.any():
                    keep = ~deg
                    fit_blocks = dict(blocks)
                    fit_blocks["__order__"] = blocks["__order__"]
                    for nm in CONCEPT_SIDE:
                        fit_blocks[nm] = blocks[nm][keep]
                    res = fit_smooth_mu(y[keep], fit_blocks,
                                        robust=not args.no_robust)
                else:
                    res = fit_smooth_mu(y, blocks, robust=not args.no_robust)
                beta = np.asarray(res["fit"]["beta"])
                fb = np.clip(bin_index(fire_rate[s], fr_edges), 0, n_fbins - 1)
                mu_pool = predict_on_axis(beta, feat, {
                    nm: con[nm][pool_idx] for nm in CONCEPT_SIDE},
                    knots, COVARIATES, n_concepts=y.shape[0])
                adq = adequacy(y, mu_pool, feat, con, pool_idx, fb, n_fbins)

                mu_full = predict_on_axis(beta, feat, con, knots)
                dset[s] = mu_full.astype(np.float32)

                report["per_seed"][str(s)] = {
                    "seed_label": seed_labels[s],
                    "n_degenerate_excluded_from_fit": int(deg.sum()),
                    "fit": {k: v for k, v in res["fit"].items() if k != "beta"},
                    "response": {
                        "n_cells": int(ybar.size),
                        "n_negative": n_neg,
                        "frac_negative": n_neg / ybar.size,
                        "n_exact_zero": n_zero,
                        "frac_non_positive": (n_neg + n_zero) / ybar.size,
                        "mean": float(ybar.mean()),
                        "median": float(np.median(ybar)),
                    },
                    "mu_hat": {
                        "min": float(mu_full.min()),
                        "max": float(mu_full.max()),
                        "median": float(np.median(mu_full)),
                        "all_positive": bool(mu_full.min() > 0),
                    },
                    "adequacy": adq,
                }
                a = adq["absorption"]
                print(f"  seed {s:2d}: edf {res['fit']['edf']:.1f} "
                      f"lam {res['fit']['lambda']:.3g} | "
                      f"non-positive y {100*(n_neg+n_zero)/ybar.size:.1f}% | "
                      f"spread {a['median_within_firing_bin_spread_before']:.1f}x "
                      f"-> {a['median_within_firing_bin_spread_after']:.1f}x "
                      f"({a['S2_absorbs_in_the_right_direction']}) | "
                      f"mu_hat min {mu_full.min():.3e}", flush=True)

    # ---- score the committed predictions, DERIVED (gotcha 20) ----
    edfs = [report["per_seed"][str(s)]["fit"]["edf"] for s in seeds]
    fracs = [report["per_seed"][str(s)]["response"]["frac_non_positive"]
             for s in seeds]
    befores = [report["per_seed"][str(s)]["adequacy"]["absorption"]
               ["median_within_firing_bin_spread_before"] for s in seeds]
    afters = [report["per_seed"][str(s)]["adequacy"]["absorption"]
              ["median_within_firing_bin_spread_after"] for s in seeds]
    n_basis_total = N_BASIS * len(COVARIATES)
    report["scorecard"] = {
        "S1_edf_strictly_between_4_and_basis_size": {
            "min_edf": min(edfs), "max_edf": max(edfs),
            "basis_size": n_basis_total,
            "outcome": "HIT" if all(4.0 < e < n_basis_total for e in edfs) else "MISS",
            "category": "within-system comparison",
        },
        "S2_absorbs_in_the_right_direction": {
            "median_before": float(np.median(befores)),
            "median_after": float(np.median(afters)),
            "outcome": "HIT" if all(a < b for a, b in zip(afters, befores)) else "MISS",
            "category": "within-system comparison",
        },
        "S3_residual_spread_above_5x": {
            "median_after": float(np.median(afters)),
            "outcome": "HIT" if float(np.median(afters)) > 5.0 else "MISS",
            "category": "distributional shape",
        },
        "S5_non_positive_response_below_25pct": {
            "max_frac": float(max(fracs)), "median_frac": float(np.median(fracs)),
            "outcome": "HIT" if max(fracs) < 0.25 else "MISS",
            "category": "distributional shape",
        },
        "S4_corr_mu_hat_r_f_negative": {
            "outcome": "DEFERRED -- measured by m4_confound.py, which crosses "
                       "the X7 firewall and therefore runs as its own job",
            "category": "DIRECTIONAL MECHANISM (0/7 record; filed anyway "
                        "per amendment AB17)",
        },
    }
    print(json.dumps({"scorecard": report["scorecard"]}, indent=2), flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(f"[m4] wrote {args.out} and {args.report}", flush=True)


if __name__ == "__main__":
    main()
