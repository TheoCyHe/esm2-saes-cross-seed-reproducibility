"""THE DESIGN-3 ESTIMATOR. Split from the join so it can be tested against a
KNOWN TRUTH without touching a cluster artifact.

    logit(E[r_fs]) = b0 + b1 * z(log M_fs) + s(log10 phi_fs) + gamma * d_fs

Binomial likelihood, `n_compare` = 39 trials, logit link, penalised cubic
spline on `log10 phi`. Governed by `docs/SPECIFICATION.md` sections 0 and 2,
which F1 derives and AL8 ratifies.

=============================================================================
WHY THIS FILE EXISTS SEPARATELY: AN IRLS ROUTINE PRODUCES PLAUSIBLE NUMBERS
FROM ANY INPUT
=============================================================================
Every defect this project has shipped had that shape -- a wrong statistic
that read as a right one. So the estimator is a module with no I/O, and
`tests/test_design3_fit.py` requires it to RECOVER a `b1` it was never told,
from data simulated at a chosen `b1`, a chosen smooth function of
`log10 phi` and a chosen `gamma`. Recovery is the only convincing test; a
fit that runs is not evidence that it fits.

=============================================================================
THE FOUR CHOICES THAT ARE MINE, STATED SO THEY CAN BE OVERRULED
=============================================================================
`docs/SPECIFICATION.md` fixes the model, the trials, the link, the covariate
set and `k = 10`. It does not fix these, and each is a numerical convention
rather than a specification question -- but a convention that is not written
down is a specification question in disguise, so:

  1. **The spline is identifiability-constrained to SUM TO ZERO over the
     observations**, and the intercept carries its level. `k = 10` B-spline
     columns therefore become 9 after the constraint, plus `b0`. Without a
     constraint the basis is a partition of unity and is exactly collinear
     with the intercept; some constraint is forced, and this is the standard
     one. AL8b's "`k = 10` is not a degree of freedom" is unaffected -- the
     penalty still decides the smoothness.
  2. **The smoothing parameter is chosen by REML** on the Laplace-approximate
     restricted likelihood, over a fixed log-spaced grid, per fit. SPECIFICATION
     §2 names REML; the grid is the numerical realisation of it.
  3. **Standard errors come from the Bayesian covariance**
     `(X'WX + S_lambda)^-1`, the usual choice for a penalised GLM. For `b1`,
     which is UNPENALISED, this differs from the frequentist covariance only
     through the spline block's influence.
  4. **Only the spline block is penalised.** `b0`, `b1` and `gamma` are
     parametric terms and enter unpenalised, which is what makes `b1` the
     estimand AL8a describes rather than a shrunk version of it.

=============================================================================
THE STANDARDISATION MUST NOT MOVE THE FIT, AND THAT IS CHECKABLE
=============================================================================
AL8a calls `z(log M)` "a reporting scale" -- outcome-blind, so it "cannot
change the fit, the Wald statistic, or the interval's coverage". That is an
ASSERTION about a linear reparameterisation, and this project's rule is to
convert assertions into measurements. `standardisation_invariance` refits on
raw `log M` and returns the residue on all three: `b1_raw * SD == b1_z`, the
deviance, and the Wald statistic. A non-zero residue is a bug and is
reportable, not a tolerance to be widened.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, gammaln, psi

from src.analysis.smooth_mu import bspline_design, diff_penalty

# The spline's basis size. AL8b: an UPPER BOUND under penalisation, not a
# degree of freedom. Imported by the join so there is one value.
SPLINE_K = 10
SPLINE_DEGREE = 3
SPLINE_PENALTY_ORDER = 2

# REML grid for the smoothing parameter. Wide enough that the selected value
# is interior for every seed -- which the join CHECKS and reports rather than
# assuming, because a boundary selection means the grid, not REML, chose it.
LAMBDA_GRID = np.logspace(-4, 8, 25)


# ---------------------------------------------------------------------------
# Basis
# ---------------------------------------------------------------------------

def centred_spline(x: np.ndarray, n_basis: int = SPLINE_K,
                   degree: int = SPLINE_DEGREE,
                   knots: np.ndarray | None = None):
    """Cubic B-spline basis with a SUM-TO-ZERO identifiability constraint.

    Returns `(B_c, S_c, knots, Z)` where `B_c = B @ Z` has `n_basis - 1`
    columns and `S_c = Z' S Z` is the second-difference penalty carried
    through the same reparameterisation.

    `bspline_design` is REUSED from `smooth_mu.py` rather than reimplemented:
    two bases for one covariate is how they drift, and the smooth normaliser
    arm and this regression both spline the same kind of quantity.
    """
    B, knots = bspline_design(x, n_basis=n_basis, degree=degree, knots=knots)
    S = diff_penalty(B.shape[1], order=SPLINE_PENALTY_ORDER)
    # C = column sums: the constraint is `sum_i f(x_i) = 0`.
    C = B.sum(axis=0).reshape(-1, 1)
    Q, _ = np.linalg.qr(C, mode="complete")
    Z = Q[:, 1:]                                  # (n_basis, n_basis - 1)
    return B @ Z, Z.T @ S @ Z, knots, Z


# ---------------------------------------------------------------------------
# Likelihood pieces
# ---------------------------------------------------------------------------

def binomial_loglik(y: np.ndarray, n: int, eta: np.ndarray) -> float:
    """Full binomial log-likelihood INCLUDING the combinatorial term.

    The term is constant in `beta`, so it cancels in every comparison made
    here -- it is included so the number printed is a log-likelihood rather
    than something that only behaves like one.
    """
    p = expit(np.clip(eta, -700, 700))
    lp = np.log(np.clip(p, 1e-300, 1.0))
    l1p = np.log(np.clip(1.0 - p, 1e-300, 1.0))
    const = (gammaln(n + 1) - gammaln(y + 1.0) - gammaln(n - y + 1.0))
    return float((const + y * lp + (n - y) * l1p).sum())


def binomial_deviance(y: np.ndarray, n: int, eta: np.ndarray) -> float:
    """`2 * (l_saturated - l_model)`. The combinatorial term cancels."""
    p = np.clip(expit(np.clip(eta, -700, 700)), 1e-12, 1 - 1e-12)
    ph = np.clip(y / n, 1e-12, 1 - 1e-12)
    t1 = np.where(y > 0, y * np.log(ph / p), 0.0)
    t2 = np.where(y < n, (n - y) * np.log((1 - ph) / (1 - p)), 0.0)
    return float(2.0 * (t1 + t2).sum())


# ---------------------------------------------------------------------------
# Penalised IRLS + REML
# ---------------------------------------------------------------------------

def penalised_irls(y: np.ndarray, n: int, X: np.ndarray, S: np.ndarray,
                   max_iter: int = 200, tol: float = 1e-10) -> dict:
    """Fisher-scoring IRLS for a binomial GLM with a quadratic penalty `S`.

    CONVERGENCE IS RETURNED, NOT ASSUMED. `smooth_mu`'s fit shipped
    under-converged once because its loop reported an iteration count and no
    verdict; the same mistake is not available here.
    """
    p_dim = X.shape[1]
    beta = np.zeros(p_dim)
    shift = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        eta = np.clip(X @ beta, -30.0, 30.0)
        mu_p = expit(eta)
        w = np.maximum(n * mu_p * (1.0 - mu_p), 1e-10)
        z = eta + (y - n * mu_p) / w
        A = X.T @ (X * w[:, None]) + S
        rhs = X.T @ (w * z)
        try:
            new = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            new = np.linalg.lstsq(A, rhs, rcond=None)[0]
        shift = float(np.abs(new - beta).max())
        beta = new
        if shift < tol:
            break
    eta = np.clip(X @ beta, -30.0, 30.0)
    mu_p = expit(eta)
    w = np.maximum(n * mu_p * (1.0 - mu_p), 1e-10)
    XtWX = X.T @ (X * w[:, None])
    return {"beta": beta, "eta": eta, "w": w, "XtWX": XtWX,
            "n_iter": int(it), "final_shift": shift,
            "converged": bool(shift < tol)}


def _logdet_pos(M: np.ndarray, tol_scale: float = 1e-10) -> tuple[float, int]:
    """`log` of the product of the POSITIVE eigenvalues, and their count."""
    ev = np.linalg.eigvalsh((M + M.T) / 2.0)
    if ev.size == 0:
        return 0.0, 0
    cut = max(ev.max(), 0.0) * tol_scale
    pos = ev[ev > cut]
    return float(np.log(pos).sum()) if pos.size else 0.0, int(pos.size)


def reml_score(y: np.ndarray, n: int, X: np.ndarray, S_lam: np.ndarray,
               fit: dict) -> float:
    """Laplace-approximate REML score to MINIMISE (binomial, scale known).

        V_r = -l(beta_hat) + 0.5 * beta' S_lam beta
              - 0.5 * log|S_lam|_+ + 0.5 * log|X'WX + S_lam|

    Terms constant in lambda are dropped, which is why this is comparable
    ACROSS the lambda grid and not interpretable on its own.
    """
    beta = fit["beta"]
    ll = binomial_loglik(y, n, fit["eta"])
    pen = 0.5 * float(beta @ (S_lam @ beta))
    ls, _ = _logdet_pos(S_lam)
    sign, lh = np.linalg.slogdet(fit["XtWX"] + S_lam)
    if sign <= 0:
        lh = _logdet_pos(fit["XtWX"] + S_lam)[0]
    return float(-ll + pen - 0.5 * ls + 0.5 * lh)


def fit_design3(y: np.ndarray, n: int, X: np.ndarray,
                S_unit: np.ndarray,
                lambda_grid: np.ndarray = LAMBDA_GRID) -> dict:
    """One per-seed fit: penalised IRLS at every lambda, REML picks one.

    `S_unit` is the penalty at `lambda = 1` on the FULL coefficient vector
    (zero on the unpenalised parametric columns). If it is all zero the model
    has no smooth term and a single unpenalised fit is returned -- which is
    what the drop-`phi` deviance refits need, and it must not silently run a
    lambda search over a penalty that does nothing.
    """
    unpenalised = not np.any(S_unit)
    grid = np.array([0.0]) if unpenalised else np.asarray(lambda_grid, float)

    best = None
    rows = []
    for lam in grid:
        S_lam = lam * S_unit
        # A tiny ridge keeps the penalty's null space identifiable without
        # materially shrinking anything: 1e-10 against a Fisher information
        # of order n_features.
        fit = penalised_irls(y, n, X, S_lam + 1e-10 * np.eye(X.shape[1]))
        v = (float("nan") if unpenalised
             else reml_score(y, n, X, S_lam, fit))
        rows.append({"lambda": float(lam), "reml": v,
                     "converged": fit["converged"]})
        if best is None or (not unpenalised and v < best[0]):
            best = (v, lam, fit)
    v_best, lam_best, fit = best

    A = fit["XtWX"] + lam_best * S_unit + 1e-10 * np.eye(X.shape[1])
    V = np.linalg.inv(A)                       # Bayesian covariance
    edf = float(np.trace(V @ fit["XtWX"]))
    dev = binomial_deviance(y, n, fit["eta"])
    finite = [r["reml"] for r in rows if np.isfinite(r["reml"])]
    return {
        "beta": fit["beta"], "cov": V, "eta": fit["eta"],
        "lambda": float(lam_best), "reml": v_best, "edf": edf,
        "deviance": dev, "loglik": binomial_loglik(y, n, fit["eta"]),
        "n_iter": fit["n_iter"], "converged": fit["converged"],
        "final_shift": fit["final_shift"],
        "lambda_grid_rows": rows,
        # A boundary selection means the GRID chose lambda, not REML. Derived,
        # and reported rather than corrected.
        "lambda_at_grid_edge": bool(
            (not unpenalised) and len(grid) > 1
            and (lam_best == grid[0] or lam_best == grid[-1])),
        "unpenalised": bool(unpenalised),
        "n_lambda_with_finite_reml": len(finite),
    }


# ---------------------------------------------------------------------------
# Design assembly
# ---------------------------------------------------------------------------

def build_design(logM_z: np.ndarray, log_phi: np.ndarray, d_fs: np.ndarray,
                 extra: dict[str, np.ndarray] | None = None,
                 include_M: bool = True, include_spline: bool = True,
                 knots: np.ndarray | None = None) -> dict:
    """`[1 | z(log M) | B_c(log10 phi) | d_fs | extra...]` and its penalty.

    `include_M` / `include_spline` exist for AL8a's DROP-ONE and JOINT
    deviance drops. They are the same code path as the full model, which is
    what makes the three deviances comparable at all.
    """
    n_obs = logM_z.size
    cols: list[np.ndarray] = []
    names: list[str] = []
    ncol = 0
    spline_slice = None
    Sc = None
    knots_out = knots

    def add(block: np.ndarray, labels: list[str]) -> slice:
        nonlocal ncol
        b = block if block.ndim == 2 else block[:, None]
        cols.append(b)
        names.extend(labels)
        sl = slice(ncol, ncol + b.shape[1])
        ncol += b.shape[1]
        return sl

    add(np.ones(n_obs), ["intercept"])
    if include_M:
        add(logM_z, ["z_logM"])
    if include_spline:
        Bc, Sc, knots_out, _ = centred_spline(log_phi, knots=knots)
        spline_slice = add(Bc, [f"spline_{i}" for i in range(Bc.shape[1])])
    add(d_fs, ["d_fs"])
    for nm, v in (extra or {}).items():
        add(v, [nm])

    X = np.column_stack(cols)
    S = np.zeros((X.shape[1], X.shape[1]))
    if spline_slice is not None:
        S[spline_slice, spline_slice] = Sc
    return {"X": X, "S": S, "names": names, "spline_slice": spline_slice,
            "knots": knots_out,
            "idx_logM": (names.index("z_logM") if include_M else None)}


# ---------------------------------------------------------------------------
# The estimand and its variance share
# ---------------------------------------------------------------------------

def variance_share(fit: dict, design: dict) -> dict:
    """AL8a's PRIMARY variance share: `sd(b1 * z(logM)) / sd(s_hat(log10 phi))`.

    Both are additive contributions to the SAME linear predictor and both are
    in logits, so the ratio needs no convention -- which is the reason AL8a
    prefers it to any GLM pseudo-R^2, five of which disagree.
    """
    X, beta = design["X"], fit["beta"]
    i = design["idx_logM"]
    sl = design["spline_slice"]
    m_part = X[:, i] * beta[i] if i is not None else np.zeros(X.shape[0])
    s_part = (X[:, sl] @ beta[sl] if sl is not None
              else np.zeros(X.shape[0]))
    sd_m, sd_s = float(np.std(m_part)), float(np.std(s_part))
    # AMENDMENT AS section 4: THE SAME INSTRUMENT, EXTENDED TO `d_fs`.
    # `gamma = +0.104` must not travel as "19x b1": `b1` is on a within-seed z
    # scale and `gamma` is on `d_fs`'s raw scale, so that ratio is a UNIT
    # ARTIFACT (gotcha 21) and would reach a methods section unchallenged.
    # Putting `d_fs` through AL8a's own linear-predictor SD ratio makes all
    # three additive contributions comparable, and the comparison is then the
    # same measured quantity the estimand already uses.
    j = design["names"].index("d_fs") if "d_fs" in design["names"] else None
    d_part = X[:, j] * beta[j] if j is not None else np.zeros(X.shape[0])
    sd_d = float(np.std(d_part))
    return {
        "sd_b1_logM": sd_m,
        "sd_s_hat_log10phi": sd_s,
        "sd_gamma_d_fs": sd_d,
        # UNCHANGED KEY AND MEANING: `log M` over the spline, AL8a's primary.
        "ratio": (sd_m / sd_s) if sd_s > 0 else float("inf"),
        "ratio_d_fs_over_spline": (sd_d / sd_s) if sd_s > 0 else float("inf"),
        # AS section 4's statistic: `d_fs`'s share against `log M`'s, both now
        # on the linear-predictor scale, so the comparison carries no units.
        "ratio_d_fs_over_logM": (sd_d / sd_m) if sd_m > 0 else float("inf"),
        "sd_ratio_is_defined": bool(sd_s > 0),
    }


def average_marginal_effect(fit: dict, design: dict, se_beta: float | None,
                            term: str = "z_logM",
                            n_trials: int | None = None) -> dict:
    """The AVERAGE over features of `d E[r_f] / d x`, on the OUTCOME scale.

    `b1` in logits is not readable as a magnitude, and the magnitude is the
    finding. For a logit link `d p / d x = beta * p (1 - p)`, so the average
    marginal effect is `mean_f(beta * p_f (1 - p_f))` -- **averaged over the
    features**, NOT `beta * p(1-p)` evaluated at the mean `p`, which is the
    back-of-envelope version and is not what this returns.

    THE STANDARD ERROR IS FIRST-ORDER AND SAYS SO. It scales `se(beta)` by
    `mean(p(1-p))`, i.e. the delta method holding the fitted probabilities
    fixed. It therefore ignores the contribution of the OTHER coefficients'
    uncertainty through `p`. That term is second-order here because the AME is
    dominated by `beta`, but it is an approximation and the output labels it
    as one rather than presenting an exact interval.

    HONESTY CLAUSE (AS section 3, mandatory and carried in the artifact).
    Unlike AL8a's within-seed z-standardisation, this transform is **NOT
    outcome-blind** -- it depends on the fitted probabilities and therefore on
    `r_f` -- and it is being adopted AFTER the result was seen. It is reported
    **alongside and never instead of** `b1_z`, and no conclusion may depend on
    it.
    """
    if term not in design["names"]:
        return {"status": f"ABSENT -- {term} is not in this design"}
    idx = design["names"].index(term)
    p = expit(np.clip(fit["eta"], -30.0, 30.0))
    w = p * (1.0 - p)
    beta = float(fit["beta"][idx])
    ame = float(np.mean(beta * w))
    mean_w = float(np.mean(w))
    out = {
        "term": term,
        "ame": ame,
        "ame_percentage_points": 100.0 * ame,
        "mean_p_1_minus_p": mean_w,
        "beta": beta,
        "naive_at_mean_p_NOT_USED": float(
            beta * np.mean(p) * (1.0 - np.mean(p))),
        "se_note": ("delta method holding the fitted probabilities fixed; "
                    "first-order, ignores the other coefficients' uncertainty "
                    "through p"),
        "outcome_blind": False,
        "honesty_note": (
            "NOT outcome-blind, unlike AL8a's z-standardisation, and adopted "
            "AFTER the result was seen. Reported ALONGSIDE b1_z and never "
            "instead of it; no conclusion depends on it (amendment AS §3)."),
    }
    if se_beta is not None:
        se = float(se_beta * mean_w)
        out["se"] = se
        out["se_percentage_points"] = 100.0 * se
        out["ci95_percentage_points"] = [100.0 * (ame - 1.959964 * se),
                                         100.0 * (ame + 1.959964 * se)]
    if n_trials is not None:
        out["n_trials"] = int(n_trials)
    return out


def standardisation_invariance(y: np.ndarray, n: int, logM: np.ndarray,
                               log_phi: np.ndarray, d_fs: np.ndarray) -> dict:
    """AL8a's assertion, converted into a MEASUREMENT.

    Refits on RAW `log M` and returns the residue on the three quantities
    AL8a says standardisation cannot touch. A non-zero residue is a defect in
    the fit, not a tolerance question.
    """
    sd = float(np.std(logM))
    mean = float(np.mean(logM))
    z = (logM - mean) / max(sd, 1e-300)

    dz = build_design(z, log_phi, d_fs)
    dr = build_design(logM, log_phi, d_fs)
    fz = fit_design3(y, n, dz["X"], dz["S"])
    fr = fit_design3(y, n, dr["X"], dr["S"])

    b_z = float(fz["beta"][dz["idx_logM"]])
    b_r = float(fr["beta"][dr["idx_logM"]])
    se_z = float(np.sqrt(fz["cov"][dz["idx_logM"], dz["idx_logM"]]))
    se_r = float(np.sqrt(fr["cov"][dr["idx_logM"], dr["idx_logM"]]))
    return {
        "sd_logM": sd, "mean_logM": mean,
        "b1_z": b_z, "b1_raw": b_r, "b1_raw_times_sd": b_r * sd,
        "abs_residue_b1": abs(b_z - b_r * sd),
        "rel_residue_b1": abs(b_z - b_r * sd) / max(abs(b_z), 1e-300),
        "deviance_z": fz["deviance"], "deviance_raw": fr["deviance"],
        "abs_residue_deviance": abs(fz["deviance"] - fr["deviance"]),
        "wald_z": b_z / max(se_z, 1e-300),
        "wald_raw": b_r / max(se_r, 1e-300),
        "abs_residue_wald": abs(b_z / max(se_z, 1e-300)
                                - b_r / max(se_r, 1e-300)),
    }


# ---------------------------------------------------------------------------
# Beta-binomial dispersion arm
# ---------------------------------------------------------------------------

def betabinom_loglik(y: np.ndarray, n: int, eta: np.ndarray,
                     log_s: float) -> float:
    p = expit(np.clip(eta, -30.0, 30.0))
    s = np.exp(log_s)
    a, b = p * s, (1.0 - p) * s
    const = gammaln(n + 1) - gammaln(y + 1.0) - gammaln(n - y + 1.0)
    return float((const
                  + gammaln(y + a) + gammaln(n - y + b) - gammaln(n + a + b)
                  - gammaln(a) - gammaln(b) + gammaln(a + b)).sum())


def _betabinom_grad(y: np.ndarray, n: int, X: np.ndarray, beta: np.ndarray,
                    log_s: float, S_lam: np.ndarray):
    eta = np.clip(X @ beta, -30.0, 30.0)
    p = expit(eta)
    s = np.exp(log_s)
    a, b = p * s, (1.0 - p) * s
    da = psi(y + a) - psi(a)
    db = psi(n - y + b) - psi(b)
    dc = psi(n + a + b) - psi(a + b)
    dl_deta = s * p * (1.0 - p) * (da - db)
    g_beta = X.T @ dl_deta - S_lam @ beta
    dl_dlogs = s * (p * da + (1.0 - p) * db - dc)
    return np.concatenate([g_beta, [float(dl_dlogs.sum())]])


def fit_betabinomial(y: np.ndarray, n: int, X: np.ndarray, S_lam: np.ndarray,
                     beta0: np.ndarray, log_s0: float = 3.0) -> dict:
    """Penalised beta-binomial MLE by L-BFGS, at a FIXED smoothing parameter.

    `lambda` is inherited from the binomial REML fit rather than re-selected.
    That is deliberate and is stated in the output: re-selecting it would make
    the dispersion arm a different SPECIFICATION rather than the same
    specification under a wider likelihood, and SPECIFICATION section 0 makes
    the binomial primary.

    `rho = 1 / (s + 1)` is the intra-cluster correlation; `s -> inf` is the
    binomial limit, which is why `log_s` and not `rho` is the free parameter.
    """
    from scipy.optimize import minimize

    def nll(th):
        beta, log_s = th[:-1], float(th[-1])
        eta = X @ beta
        return -(betabinom_loglik(y, n, eta, log_s)
                 - 0.5 * float(beta @ (S_lam @ beta)))

    def grad(th):
        beta, log_s = th[:-1], float(th[-1])
        return -_betabinom_grad(y, n, X, beta, log_s, S_lam)

    th0 = np.concatenate([beta0, [log_s0]])
    res = minimize(nll, th0, jac=grad, method="L-BFGS-B",
                   options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
    th = res.x
    beta, log_s = th[:-1], float(th[-1])

    # Observed information by finite differences on the analytic gradient.
    k = th.size
    H = np.zeros((k, k))
    step = 1e-5
    for j in range(k):
        e = np.zeros(k)
        e[j] = step
        H[:, j] = (grad(th + e) - grad(th - e)) / (2 * step)
    H = (H + H.T) / 2.0
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)

    s = np.exp(log_s)
    return {
        "beta": beta, "log_s": log_s, "s": float(s),
        "rho": float(1.0 / (s + 1.0)),
        "cov": cov[:-1, :-1], "cov_full": cov,
        "loglik": betabinom_loglik(y, n, X @ beta, log_s),
        "success": bool(res.success), "n_iter": int(res.nit),
        "message": str(res.message),
        "hessian_is_pd": bool(np.all(np.linalg.eigvalsh(H) > 0)),
    }


def betabinom_atom(n: int, k: int, p: np.ndarray, s: float) -> np.ndarray:
    """`P(Y = k)` per observation under the beta-binomial."""
    a, b = p * s, (1.0 - p) * s
    lg = (gammaln(n + 1) - gammaln(k + 1.0) - gammaln(n - k + 1.0)
          + gammaln(k + a) + gammaln(n - k + b) - gammaln(n + a + b)
          - gammaln(a) - gammaln(b) + gammaln(a + b))
    return np.exp(lg)


def endpoint_check(y: np.ndarray, n: int, eta: np.ndarray,
                   s: float | None = None) -> dict:
    """AK9a: implied `P(k = 0)` and `P(k = n)` against OBSERVED.

    A beta-binomial cannot fit atoms at the endpoints, and the Beta(1,1)
    recovery check is silent about exactly that. The DIRECTION in which any
    misspecification moves the intervals is DERIVED from the sign of
    `observed - implied`, never asserted (gotcha 20).
    """
    p = expit(np.clip(eta, -30.0, 30.0))
    if s is None:
        imp0 = float(np.mean((1.0 - p) ** n))
        impn = float(np.mean(p ** n))
        family = "binomial"
    else:
        imp0 = float(np.mean(betabinom_atom(n, 0, p, s)))
        impn = float(np.mean(betabinom_atom(n, n, p, s)))
        family = "beta_binomial"
    obs0 = float(np.mean(y == 0))
    obsn = float(np.mean(y == n))
    under = (obs0 > imp0) or (obsn > impn)
    over = (obs0 < imp0) and (obsn < impn)
    direction = (
        "OBSERVED_ATOMS_EXCEED_IMPLIED -- the data are more overdispersed "
        "than the fitted family, so the model's information is overstated "
        "and the reported intervals are TOO NARROW (anti-conservative)."
        if under else
        "IMPLIED_ATOMS_EXCEED_OBSERVED -- the fitted family puts more mass at "
        "the endpoints than the data do, so the model's information is "
        "understated and the reported intervals are TOO WIDE (conservative)."
        if over else
        "MIXED -- one atom is under-predicted and the other over-predicted, "
        "so no single direction follows for the intervals.")
    return {
        "family": family,
        "observed_P_k_eq_0": obs0, "implied_P_k_eq_0": imp0,
        "observed_P_k_eq_n": obsn, "implied_P_k_eq_n": impn,
        "residual_P_k_eq_0": obs0 - imp0,
        "residual_P_k_eq_n": obsn - impn,
        "ratio_P_k_eq_0": obs0 / imp0 if imp0 > 0 else float("inf"),
        "ratio_P_k_eq_n": obsn / impn if impn > 0 else float("inf"),
        "direction_of_misspecification": direction,
    }


# ---------------------------------------------------------------------------
# Meta-analysis
# ---------------------------------------------------------------------------

def meta_analyse(est: np.ndarray, se: np.ndarray) -> dict:
    """BOTH combinations, because they answer different questions and this
    module is not authorised to pick one.

      * inverse-variance FIXED EFFECT -- the conventional combination, and the
        one that assumes the per-seed fits are INDEPENDENT. They are not: at
        `n_compare = 39` two anchors share 38 of their 39 comparison
        dictionaries.
      * ACROSS-SEED -- the unweighted mean with `SD / sqrt(20)`. It carries
        seed-level dependence empirically, because each seed's estimate is one
        draw of the whole ensemble.

    AK9b requires the ratio between them to be RE-DERIVED at N = 39 with the
    dependence structure stated, not inherited from the N = 19 measurement.
    Both are returned; which is the headline interval is a REPORTING
    convention the specification record does not fix, and it is flagged as
    open rather than decided here.
    """
    est = np.asarray(est, float)
    se = np.asarray(se, float)
    ok = np.isfinite(est) & np.isfinite(se) & (se > 0)
    if ok.sum() == 0:
        return {"status": "NO_USABLE_SEEDS", "n_seeds_used": 0}
    e, s = est[ok], se[ok]
    w = 1.0 / s ** 2
    fe = float((w * e).sum() / w.sum())
    fe_se = float(np.sqrt(1.0 / w.sum()))
    mean = float(e.mean())
    sd = float(e.std(ddof=1)) if e.size > 1 else float("nan")
    as_se = sd / np.sqrt(e.size) if e.size > 1 else float("nan")
    Q = float((w * (e - fe) ** 2).sum())
    df = e.size - 1
    i2 = float(max(0.0, (Q - df) / Q) * 100) if Q > 0 and df > 0 else 0.0
    return {
        "status": "OK",
        "n_seeds_used": int(ok.sum()),
        "n_seeds_dropped": int((~ok).sum()),
        "fixed_effect": fe, "fixed_effect_se": fe_se,
        "fixed_effect_ci95": [fe - 1.959964 * fe_se, fe + 1.959964 * fe_se],
        "across_seed_mean": mean,
        "across_seed_sd": sd, "across_seed_se": as_se,
        "across_seed_ci95": ([mean - 1.959964 * as_se, mean + 1.959964 * as_se]
                             if np.isfinite(as_se) else [float("nan")] * 2),
        # AK9b's quantity, re-derived rather than inherited.
        "across_seed_se_over_fixed_effect_se": (
            float(as_se / fe_se) if np.isfinite(as_se) and fe_se > 0
            else float("nan")),
        "Q": Q, "df": df, "I2_percent": i2,
        "signs_agree": bool(np.sign(fe) == np.sign(mean)),
    }
