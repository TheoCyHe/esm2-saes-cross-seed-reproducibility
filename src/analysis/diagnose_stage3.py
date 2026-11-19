"""Measurements 2, 4 and 5 -- the stage-3 statistic decision.

PREDICTIONS ARE RECORDED HERE AND COMMITTED BEFORE THE MODULE RUNS. The git
timestamp of this file is the audit trail.

-----------------------------------------------------------------------------
M2 -- DOES CLUSTER COUNT PREDICT mu, AS OPPOSED TO sigma?
-----------------------------------------------------------------------------
Q1 measured rho(cluster count, log10 per-concept null SD) = -0.218. That was
about SIGMA. If centring survives measurement 1, the quantity the
concept-property axes have to predict is MU, and nobody has measured it.
Retiring O2's fourth axis on the sigma result alone would foreclose this.

  Prediction: rho(prevalence, log10 mu) will be STRONGLY negative -- I expect
  below -0.5, and stronger than anything measured for sigma. Reason (G0c):
  under joint shift the null AP for a common broad concept converges close to
  its own prevalence, so the analytic normalization already removes nearly all
  of it and the residual mean gain is ~0. For a RARE concept a single lucky
  placement produces a large gain relative to a tiny prevalence, so the mean
  residual is larger. Chance level therefore falls with prevalence.

  Prediction: rho(cluster count, log10 mu) will be negative and of similar or
  slightly larger magnitude than the sigma result (-0.22 to -0.45), and will
  WEAKEN substantially when controlled within n_positives deciles -- because I
  expect cluster count to act here mostly as a proxy for concept size, which
  is the opposite of what the sigma measurement found (there it STRENGTHENED
  to -0.323 under the same control).

  That last contrast is the informative one: if cluster count survives the
  n_positives control for mu as it did for sigma, the axis is measuring
  independence and should be kept. If it collapses, it was a size proxy.

-----------------------------------------------------------------------------
M4 -- IS A PER-FEATURE GUMBEL BETTER CONDITIONED THAN AN EMPIRICAL QUANTILE?
-----------------------------------------------------------------------------
  Prediction: NO, and for a structural reason that is already measured. A
  Gumbel has two parameters and therefore a FIXED skewness of +1.1395. The
  measured median per-feature skewness of M_null is +1.762. A distribution
  with skewness 1.76 is not a Gumbel at any (location, scale) -- the
  two-parameter family cannot reach it.

  So I predict held-out calibration fails in the direction that matters:
  upper-tail quantiles fitted too LOW, hence observed exceedances of the
  fitted 95th/99th percentile MORE frequent than nominal. A three-parameter
  GEV with a positive shape (Frechet-type) is what the skewness implies.

  What would falsify this: held-out exceedance rates near nominal, which
  would mean the skewness excess is estimation noise at n=20 rather than a
  real departure from Gumbel.

  Note the asymmetry that makes this worth running anyway: even a
  mis-specified parametric fit may beat a 20-draw empirical quantile, because
  the empirical version cannot resolve beyond 1/21 at all. Goodness-of-fit is
  reported alongside the statistic, not instead of it.

-----------------------------------------------------------------------------
M5 -- WHAT DOES CENSORING COST? The number that decides exceedance
-----------------------------------------------------------------------------
  Prediction: the ceiling group is heavily enriched for well-aligned features
  -- I expect its median raw max-G to exceed the NON-ceiling p90. If so,
  exceedance censors the predictor exactly where the regression needs signal,
  and pooled exceedance alone is disqualified as the primary predictor.

  This is close to tautological in direction (high M implies ceiling), so the
  informative quantity is the OVERLAP: how much raw-G range the ceiling group
  spans internally, and how many non-ceiling features exceed the ceiling
  group's own 25th percentile. A ceiling group that is internally
  homogeneous is a smaller loss than one spanning the whole useful range.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr

# A Gumbel's skewness is fixed: 12*sqrt(6)*zeta(3)/pi**3.
GUMBEL_SKEW = 1.1395470994046486


def gumbel_fit(x: np.ndarray):
    """Method-of-moments Gumbel fit. beta = sd*sqrt(6)/pi, mu = mean - gamma*beta.

    Moments rather than MLE deliberately: at n=20 the MLE's advantage is
    small and MoM cannot fail to converge, so a fit failure cannot be
    confused with a distributional failure -- which is the thing being tested.
    """
    sd = x.std(axis=0, ddof=1)
    beta = sd * np.sqrt(6.0) / np.pi
    mu = x.mean(axis=0) - 0.5772156649 * beta
    return mu, beta


def gumbel_ppf(q, mu, beta):
    return mu - beta * np.log(-np.log(q))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gtilde", type=Path, required=True)
    p.add_argument("--ap", type=Path, required=True)
    p.add_argument("--null-raw", type=Path, required=True)
    p.add_argument("--labels-json", type=Path, required=True)
    p.add_argument("--arm", default="b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rep: dict = {}

    with h5py.File(args.gtilde, "r") as h:
        concepts = [c.decode() if isinstance(c, bytes) else c
                    for c in h["concepts"][:]]
        pool = h["in_argmax_pool"][:].astype(bool)
        M = h["M"][:]
        Mn = h["M_null"][:]
        clus = h["axis_cluster_count"][:]
        prev_axis = h["axis_prevalence"][:]
        _fre = h.attrs["firing_rate_edges"]
        if isinstance(_fre, bytes):
            _fre = _fre.decode()
        fr_edges = json.loads(_fre) if isinstance(_fre, str) else list(_fre)
    pool_idx = np.flatnonzero(pool)
    diag = json.loads(args.labels_json.read_text())["retention_floor_diagnostics"]
    npos = np.array([diag[c]["n_residues"] for c in concepts], dtype=float)

    # ======================================================== M2: mu ========
    print("=" * 74)
    print("M2. DOES CLUSTER COUNT PREDICT mu (not sigma)?")
    print("=" * 74)
    with h5py.File(args.null_raw, "r") as h:
        slab = h[f"gain_{args.arm}"][args.seed].astype(np.float64)
    n_r, n_cp, n_f = slab.shape
    flat = slab.transpose(1, 0, 2).reshape(n_cp, n_r * n_f)
    mu_c = flat.mean(axis=1)
    sd_c = flat.std(axis=1)
    del slab, flat

    lmu = np.log10(np.where(mu_c > 0, mu_c, np.nan))
    lsd = np.log10(np.where(sd_c > 0, sd_c, np.nan))
    cl, pv, np_ = clus[pool_idx], prev_axis[pool_idx], npos[pool_idx]
    ok = np.isfinite(lmu) & np.isfinite(lsd)
    print(f"  n = {int(ok.sum())} pool concepts with mu > 0")
    for nm, v in [("prevalence", pv), ("cluster count", cl),
                  ("n_positives", np_)]:
        rm = spearmanr(v[ok], lmu[ok])
        rs = spearmanr(v[ok], lsd[ok])
        print(f"    Spearman({nm:14s}, log10 mu)    = {rm.statistic:+.3f} "
              f"(p={rm.pvalue:.1e})   [vs log10 sigma: {rs.statistic:+.3f}]")
    dec = np.digitize(np_, np.percentile(np_, np.arange(10, 100, 10)))
    wm, ws = [], []
    for d in range(10):
        m = ok & (dec == d)
        if m.sum() >= 15 and np.unique(cl[m]).size > 2:
            wm.append(spearmanr(cl[m], lmu[m]).statistic)
            ws.append(spearmanr(cl[m], lsd[m]).statistic)
    print(f"  WITHIN n_positives deciles, cluster count vs:")
    print(f"    log10 mu   : median {np.median(wm):+.3f} over {len(wm)} deciles")
    print(f"    log10 sigma: median {np.median(ws):+.3f}  (Q1 measured -0.323)")
    print("  -> if the mu relationship SURVIVES the size control the axis is "
          "measuring\n     independence; if it collapses it was a size proxy.")
    rep["m2"] = {"spearman_prev_mu": float(spearmanr(pv[ok], lmu[ok]).statistic),
                 "spearman_cluster_mu": float(spearmanr(cl[ok], lmu[ok]).statistic),
                 "within_decile_cluster_mu": float(np.median(wm)),
                 "within_decile_cluster_sigma": float(np.median(ws))}

    # ======================================================== M4: Gumbel ====
    print("\n" + "=" * 74)
    print("M4. PER-FEATURE GUMBEL vs A 20-DRAW EMPIRICAL QUANTILE")
    print("=" * 74)
    x = Mn[args.seed]                       # (shifts, features)
    n_shifts = x.shape[0]
    skew = (((x - x.mean(0)) / x.std(0, ddof=1)) ** 3).mean(0)
    print(f"  measured median per-feature skewness: {np.median(skew):+.3f}   "
          f"Gumbel is FIXED at {GUMBEL_SKEW:+.3f}")
    print(f"  features whose skewness exceeds Gumbel's: "
          f"{100*np.mean(skew > GUMBEL_SKEW):.1f}%")

    # LEAVE-ONE-OUT calibration: fit on 19, ask where the held-out draw falls.
    # Under a correct model the LOO draw's fitted CDF value is Uniform(0,1).
    u = np.empty_like(x)
    for r in range(n_shifts):
        kept = np.delete(x, r, axis=0)
        mu_g, beta_g = gumbel_fit(kept)
        with np.errstate(over="ignore", invalid="ignore"):
            u[r] = np.exp(-np.exp(-(x[r] - mu_g) / beta_g))
    u = u[np.isfinite(u)]
    print(f"  LOO probability-integral transform (should be Uniform(0,1)):")
    print(f"    mean {u.mean():.4f} (0.5)   median {np.median(u):.4f} (0.5)")
    for q in (0.90, 0.95, 0.99):
        obs = float(np.mean(u > q))
        print(f"    exceedance of the fitted {q:.2f} quantile: "
              f"{obs:.4f}  (nominal {1-q:.4f})  ratio {obs/(1-q):.2f}x")
    print("  -> ratio > 1 means the fitted upper tail is too LOW, i.e. the "
          "model\n     under-states how extreme the null can get -- the "
          "direction that\n     would make a Gumbel-based statistic "
          "anti-conservative.")
    rep["m4"] = {"median_skew": float(np.median(skew)),
                 "gumbel_skew": GUMBEL_SKEW,
                 "pit_mean": float(u.mean()),
                 "exceed_95_ratio": float(np.mean(u > 0.95) / 0.05),
                 "exceed_99_ratio": float(np.mean(u > 0.99) / 0.01)}

    # ======================================================== M5: censoring =
    print("\n" + "=" * 74)
    print("M5. WHAT DOES CENSORING COST? raw max-G inside the ceiling group")
    print("=" * 74)
    with h5py.File(args.ap, "r") as h:
        g = h["pooled"]
        prevalence = g["prevalence"][:]
        G = (g["ap"][args.seed][:] - prevalence[None, :]) / (1 - prevalence[None, :])
    rawmax = np.where(pool[None, :], G, -np.inf).max(axis=1)

    # Pooled over ALL features, deliberately. Measurement 3 is still deciding
    # what the right grouping variable is (firing rate demonstrably is not:
    # per-feature mean M_null spans 33x-225x inside one firing bin), so
    # pooling globally here gives the LARGEST reference set and therefore the
    # LOWEST achievable ceiling. If censoring is disqualifying even at that
    # bound, no grouping rescues it -- which is what makes this the number
    # that decides exceedance.
    from src.analysis.diagnose_stages import exceedance

    pooled = x.ravel()
    pv_all, res = exceedance(M[args.seed], pooled)
    ceiling = pv_all <= res + 1e-12
    print(f"  pooled over ALL features: {pooled.size:,} draws, "
          f"resolution {res:.2e}")
    print("    (a LOWER BOUND on the ceiling: any real grouping uses fewer "
          "draws and censors more)")
    print(f"  features at the ceiling: {ceiling.sum():,} of {ceiling.size:,} "
          f"({100*ceiling.mean():.2f}%)")
    a, b = rawmax[ceiling], rawmax[~ceiling]
    print(f"  raw max-G  ceiling group : median {np.median(a):.4f}  "
          f"p25 {np.percentile(a,25):.4f}  p75 {np.percentile(a,75):.4f}")
    print(f"  raw max-G  non-ceiling   : median {np.median(b):.4f}  "
          f"p90 {np.percentile(b,90):.4f}  max {b.max():.4f}")
    print(f"  ceiling median vs non-ceiling p90: "
          f"{np.median(a):.4f} vs {np.percentile(b,90):.4f} -> "
          f"{'ABOVE' if np.median(a) > np.percentile(b,90) else 'BELOW'}")
    overlap = float(np.mean(b > np.percentile(a, 25)))
    print(f"  non-ceiling features above the ceiling group's p25: "
          f"{100*overlap:.2f}%")
    print(f"  raw max-G RANGE inside the ceiling group: "
          f"{a.min():.4f} to {a.max():.4f} "
          f"({a.max()/max(a.min(),1e-9):.1f}x)")
    rep["m5"] = {"frac_ceiling": float(ceiling.mean()),
                 "ceiling_median_rawG": float(np.median(a)),
                 "nonceiling_p90_rawG": float(np.percentile(b, 90)),
                 "overlap_share": overlap,
                 "ceiling_rawG_min": float(a.min()),
                 "ceiling_rawG_max": float(a.max())}

    args.out.write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
