"""Does stage 1 earn its place, and can stage 3 be given a usable scale?

Three measurements (2a, 2b, 2c), and the PREDICTIONS ARE RECORDED HERE,
COMMITTED BEFORE THE MODULE IS RUN. This file's git timestamp is the audit
trail; the last two rounds both produced a "prediction" written after the
numbers were visible, and disclosing that honestly does not repair it.

-----------------------------------------------------------------------------
PRE-RECORDED PREDICTIONS
-----------------------------------------------------------------------------

2a -- DROP STAGE 1. Compute `M_f` from raw `G` directly and compare against
`M_f` from `z`.

  The author's prediction: dropping stage 1 loses almost nothing and removes a
  noise source. Grounds: Q3's rho = 0.9961, the 2d redundancy argument, and
  heme a3's 30x cross-seed spread in `z`.

  My prediction, and it differs in one place: Spearman(M_raw, M_z) will be
  HIGH but NOT ~1 -- I expect ~0.90-0.98 rather than >0.99, and argmax
  agreement near Q3's measured 81.65% rather than ~100%. Reason: Q3's 0.9961
  is for the SCALING transform alone. The full `z` also subtracts `mu_s`,
  which measurably reorders (rho 0.2582), and although that churn is
  concentrated in the sub-maximal bulk it is not confined to it -- 18.35% of
  argmaxes already move. So stage 1 is near-inert for the RANKING of `M`,
  not literally inert.

  Where I expect the author to be exactly right: cross-seed CV of `M_raw` will be
  LOWER than that of `M_z`. That is the claim that matters, and it is 2b.

  What would falsify "drop stage 1": if `M_raw` had HIGHER cross-seed CV, or
  if the two `M` vectors disagreed enough that `g~` changed materially for a
  large share of features.

2b -- CROSS-SEED CV THROUGH THE PIPELINE.

  Prediction: CV rises monotonically raw `G` -> `z`. `sigma_s` is estimated
  per seed from a finite draw set, so it is noisy; dividing by a noisy
  estimate of a near-zero quantity turns small denominator noise into large
  quotient noise. heme a3 (raw `G` = 1.0 in 17/20 seeds, `z` spanning
  792-23,897) is the visible case -- but per the lesson of this session I
  explicitly do NOT assume it is representative. It may be a 3.1% effect
  again, and the measurement is over all 703 concepts for that reason.

  A GOTCHA-18 CONSTRAINT, stated because it changes what can be measured.
  `g~` is indexed by FEATURE, and a feature index means nothing across
  independently-seeded SAEs (measured cross-seed correlation of fire_rate by
  index: 0.0009). So "cross-seed CV of `g~` per feature" is not a computable
  quantity -- it would average 20 unrelated random variables. Concepts DO
  have stable cross-seed identity, so `G` and `z` are measured per concept as
  max-over-features per seed, then compared across seeds (the legitimate
  per-seed-then-combine pattern). For `g~` only per-seed DISTRIBUTIONAL
  summaries can be compared, and that is what is reported.

2c -- STAGE-3 EXCEEDANCE BY POOLING ACROSS FEATURES.

  At 20 shifts, exceedance is hopeless: if ~68% of features beat all 20 draws
  the predictor is constant for two-thirds of the dictionary. Pooling null-max
  draws across features within a firing bin gives ~40,040 draws (2,002
  features x 20 shifts) and resolution ~2.5e-5, from data already on disk.

  Prediction: pooling reduces the ceiling fraction by a lot but does NOT
  eliminate it -- I expect single-digit percent still censored, because the
  same features that beat all 20 of their own draws are largely the ones that
  beat the pooled set too. Exchangeability within a firing bin will be
  IMPERFECT: per-feature null-max distributions will differ measurably inside
  a bin, since firing rate is not the only property that governs a feature's
  null max.

-----------------------------------------------------------------------------
1b -- THE EXTREME-VALUE CHECK, computed rather than asserted
-----------------------------------------------------------------------------
`M` is a maximum, so its null is right-skewed and its median sits BELOW its
mean; standardizing by mean and sd therefore yields a negative null median by
construction, not by defect. That is a directional argument, and this module
turns it into a NUMBER: for a Gumbel, mean = mu + gamma*beta, median = mu -
beta*ln(ln 2), sd = pi*beta/sqrt(6), so the standardized median is

    (median - mean)/sd = (-ln(ln 2) - gamma) / (pi/sqrt(6)) = -0.1643

If the measured value is near -0.164 the extreme-value account is complete.
If it is materially more negative, the null max is HEAVIER-tailed than Gumbel
and the mean/sd scale is even less appropriate than the argument implies.
Reported either way.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr

# Standardized median of a Gumbel: (-ln(ln 2) - Euler gamma) / (pi/sqrt(6)).
GUMBEL_STANDARDIZED_MEDIAN = (
    (-np.log(np.log(2.0)) - 0.5772156649) / (np.pi / np.sqrt(6.0)))


def standardize_max(M_obs: np.ndarray, M_null: np.ndarray):
    """Stage 3 as currently specified: (M - mean)/sd over the shift axis.

    `M_obs` is (features,), `M_null` is (shifts, features).
    """
    mean, sd = M_null.mean(axis=0), M_null.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (M_obs - mean) / sd


def exceedance(M_obs: np.ndarray, pooled_null: np.ndarray):
    """(count of pooled null draws >= M_obs + 1) / (N + 1).

    The +1s are the standard permutation-p correction: a statistic can never
    be reported as impossible on finite draws. Returns (p, resolution), where
    resolution = 1/(N+1) is the smallest reportable value -- the CEILING, and
    the number that decides whether pooling bought anything.
    """
    n = pooled_null.size
    srt = np.sort(pooled_null)
    n_ge = n - np.searchsorted(srt, M_obs, side="left")
    return (n_ge + 1.0) / (n + 1.0), 1.0 / (n + 1.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gtilde", type=Path, required=True)
    p.add_argument("--ap", type=Path, required=True)
    p.add_argument("--null-reductions", type=Path, required=True,
                   help="null_g_* = max over CONCEPTS per feature, i.e. M_null "
                        "computed on RAW gain -- exactly the stage-1-free "
                        "comparator 2a needs. Already derived (N11), so this "
                        "measurement does not touch the 23 GB raw array.")
    p.add_argument("--arm", default="b")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rep: dict = {}

    with h5py.File(args.gtilde, "r") as h:
        concepts = [c.decode() if isinstance(c, bytes) else c
                    for c in h["concepts"][:]]
        pool = h["in_argmax_pool"][:].astype(bool)
        M_z = h["M"][:]                       # (seeds, feats)
        Mn_z = h["M_null"][:]                 # (seeds, shifts, feats)
        gt_z = h["gtilde"][:]
        gtn_z = h["gtilde_null"][:]
        argc_z = h["argmax_concept"][:]
        _fre = h.attrs["firing_rate_edges"]
        if isinstance(_fre, bytes):
            _fre = _fre.decode()
        # write_identity json.dumps() lists, so this attr comes back as a
        # STRING. `list()` of a string yields CHARACTERS, and the first run
        # of this module silently grouped features into "bins" 24/49/50 --
        # plausible-looking integers that were not firing bins at all.
        fr_edges = json.loads(_fre) if isinstance(_fre, str) else list(_fre)
        assert len(fr_edges) >= 3 and all(
            isinstance(x, (int, float)) for x in fr_edges), fr_edges
    with h5py.File(args.null_reductions, "r") as h:
        Mn_raw = h[f"null_g_{args.arm}"][:]   # (seeds, shifts, feats) RAW max
        fire = h["fire_rate"][:]
    with h5py.File(args.ap, "r") as h:
        g = h["pooled"]
        prevalence = g["prevalence"][:]
        ap_ds = g["ap"]
        n_seeds, n_feats, n_c = ap_ds.shape
        M_raw = np.empty((n_seeds, n_feats))
        argc_raw = np.empty((n_seeds, n_feats), dtype=np.int32)
        Gmax_concept_raw = np.empty((n_seeds, n_c))
        for s in range(n_seeds):
            G = ((ap_ds[s][:] - prevalence[None, :]) / (1 - prevalence[None, :]))
            masked = np.where(pool[None, :], G, -np.inf)
            M_raw[s] = masked.max(axis=1)
            argc_raw[s] = masked.argmax(axis=1)
            Gmax_concept_raw[s] = G.max(axis=0)      # per-seed max over features

    # ================================================================= 1b ===
    print("=" * 74)
    print("1b. IS THE NEGATIVE NULL MEDIAN JUST EXTREME-VALUE SKEW?")
    print("=" * 74)
    med = float(np.nanmedian(gtn_z))
    print(f"  measured standardized null median : {med:+.4f}")
    print(f"  Gumbel prediction                 : "
          f"{GUMBEL_STANDARDIZED_MEDIAN:+.4f}")
    print(f"  ratio                             : "
          f"{med/GUMBEL_STANDARDIZED_MEDIAN:.2f}x")
    print("  -> near 1.0x means the extreme-value account is complete; "
          "materially\n     more negative means HEAVIER than Gumbel.")
    # skew of the raw null max, directly
    # SKEWNESS IS THE MEAN OF CUBED STANDARDIZED DEVIATIONS, NOT THE MEDIAN.
    # The first version of this line took the median, which is approximately
    # the cube of the standardized median -- so it restated the number above
    # instead of testing it independently, and read as confirmation.
    sk = []
    for s in range(Mn_z.shape[0]):
        x = Mn_z[s]
        m, sd = x.mean(axis=0), x.std(axis=0, ddof=1)
        per_feature_skew = (((x - m) / sd) ** 3).mean(axis=0)   # over SHIFTS
        sk.append(float(np.nanmedian(per_feature_skew)))
    print(f"  median per-feature SKEWNESS of M_null: {np.median(sk):+.3f} "
          "(0 = symmetric; Gumbel = +1.14)")
    print("  -> if this is ~0 the Gumbel account is NOT what produces the "
          "negative median.")
    print("     Synthetic control: LOO-standardizing a SYMMETRIC null gives "
          "median exactly 0.000,")
    print("     so a non-zero median REQUIRES genuine right-skew in M_null.")
    rep["extreme_value"] = {"measured_std_null_median": med,
                            "gumbel_prediction": GUMBEL_STANDARDIZED_MEDIAN,
                            "median_per_feature_skew": float(np.median(sk))}

    # ================================================================= 2a ===
    print("\n" + "=" * 74)
    print("2a. DROP STAGE 1 -- M from raw G vs M from z")
    print("=" * 74)
    rho = [spearmanr(M_raw[s], M_z[s]).statistic for s in range(n_seeds)]
    agree = [(argc_raw[s] == argc_z[s]).mean() for s in range(n_seeds)]
    print(f"  Spearman(M_raw, M_z) per seed: median {np.median(rho):.4f} "
          f"(min {min(rho):.4f} max {max(rho):.4f})")
    print(f"  argmax agreement            : median {100*np.median(agree):.2f}%")

    gt_raw = np.stack([standardize_max(M_raw[s], Mn_raw[s])
                       for s in range(n_seeds)])
    r_gt = [spearmanr(gt_raw[s], gt_z[s]).statistic for s in range(n_seeds)]
    print(f"  Spearman(g~_raw, g~_z)      : median {np.median(r_gt):.4f}")
    for nm, v in [("g~_z  ", gt_z), ("g~_raw", gt_raw)]:
        print(f"    {nm}: median {np.nanmedian(v):8.3f}  "
              f"frac > 5: {100*np.nanmean(v > 5):5.2f}%  "
              f"max {np.nanmax(v):.1f}")
    rep["drop_stage1"] = {
        "spearman_M": float(np.median(rho)),
        "argmax_agreement": float(np.median(agree)),
        "spearman_gtilde": float(np.median(r_gt)),
        "frac_gt5_z": float(np.nanmean(gt_z > 5)),
        "frac_gt5_raw": float(np.nanmean(gt_raw > 5)),
    }

    # ================================================================= 2b ===
    print("\n" + "=" * 74)
    print("2b. CROSS-SEED CV -- raw G vs z, PER CONCEPT (gotcha 18 safe)")
    print("=" * 74)
    with h5py.File(args.gtilde, "r") as h:
        zmax_concept = np.empty((n_seeds, n_c))
        for s in range(n_seeds):
            zmax_concept[s] = np.nanmax(h["z"][s][:], axis=0)

    def cv(a):                       # a is (seeds, concepts)
        m = a.mean(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(np.abs(m) > 0, a.std(axis=0, ddof=1) / np.abs(m),
                            np.nan)
    cv_G, cv_z = cv(Gmax_concept_raw), cv(zmax_concept)
    pi = np.flatnonzero(pool)
    print(f"  cross-seed CV of max-over-features, over {len(pi)} pool concepts:")
    for nm, v in [("raw G", cv_G[pi]), ("z    ", cv_z[pi])]:
        q = np.nanpercentile(v, [50, 90, 99])
        print(f"    {nm}: median {q[0]:.4f}   p90 {q[1]:.4f}   p99 {q[2]:.4f}")
    ratio = np.nanmedian(cv_z[pi] / np.where(cv_G[pi] > 0, cv_G[pi], np.nan))
    print(f"  median per-concept ratio CV(z)/CV(G) = {ratio:.3f}x")
    print(f"  concepts where CV rises G -> z: "
          f"{100*np.nanmean(cv_z[pi] > cv_G[pi]):.1f}%")
    for nm in ("BINDING:heme a3", "BINDING:Fe(II)-heme a"):
        i = concepts.index(nm)
        print(f"    {nm:24s}: CV(G) {cv_G[i]:.4f}  CV(z) {cv_z[i]:.4f}  "
              f"z range {zmax_concept[:, i].min():.0f}-"
              f"{zmax_concept[:, i].max():.0f}")
    print(f"  g~ is per-FEATURE, so cross-seed CV by index is undefined "
          f"(gotcha 18).\n    per-seed g~ medians: min "
          f"{np.nanmedian(gt_z, axis=1).min():.2f} max "
          f"{np.nanmedian(gt_z, axis=1).max():.2f}")
    rep["cross_seed_cv"] = {
        "median_cv_rawG": float(np.nanmedian(cv_G[pi])),
        "median_cv_z": float(np.nanmedian(cv_z[pi])),
        "median_ratio": float(ratio),
        "share_rising": float(np.nanmean(cv_z[pi] > cv_G[pi])),
    }

    # ================================================================= 2c ===
    print("\n" + "=" * 74)
    print("2c. STAGE-3 EXCEEDANCE BY POOLING ACROSS FEATURES (firing bins)")
    print("=" * 74)
    fb = np.clip(np.searchsorted(np.array(fr_edges), fire[0], side="right") - 1,
                 0, len(fr_edges) - 2)
    print(f"  own-draws-only resolution: 1/21 = {1/21:.4f}  "
          f"(ceiling fraction now: {100*np.mean(gt_z[0] > 5):.1f}% of features "
          "at g~ > 5)")
    ceil_tot = res_tot = 0
    per_bin = []
    for b in range(len(fr_edges) - 1):
        idx = np.flatnonzero(fb == b)
        if idx.size == 0:
            continue
        pooled = Mn_z[0][:, idx].ravel()
        pv, res = exceedance(M_z[0][idx], pooled)
        at_ceiling = float(np.mean(pv <= res + 1e-12))
        per_bin.append({"bin": b, "n_features": int(idx.size),
                        "n_draws": int(pooled.size), "resolution": res,
                        "frac_at_ceiling": at_ceiling})
        print(f"    bin {b}: {idx.size:5d} features, {pooled.size:7,d} pooled "
              f"draws, resolution {res:.2e}, at ceiling {100*at_ceiling:5.2f}%")
        ceil_tot += at_ceiling * idx.size
        res_tot += idx.size
    print(f"  OVERALL at ceiling after pooling: {100*ceil_tot/res_tot:.2f}% "
          f"of features (was ~{100*np.mean(gt_z[0] > 5):.1f}% at g~ > 5)")

    # exchangeability within a bin -- the same question G5 asked for stage 1
    print("\n  exchangeability within firing bin (per-feature null-max "
          "distributions):")
    for b in range(len(fr_edges) - 1):
        idx = np.flatnonzero(fb == b)
        if idx.size < 20:
            continue
        pf_mean = Mn_z[0][:, idx].mean(axis=0)
        print(f"    bin {b}: per-feature mean M_null spans "
              f"{pf_mean.min():.1f} to {pf_mean.max():.1f} "
              f"(ratio {pf_mean.max()/max(pf_mean.min(),1e-9):.1f}x), "
              f"CV {pf_mean.std()/pf_mean.mean():.3f}")
    rep["exceedance_pooling"] = {"per_bin": per_bin,
                                 "overall_frac_at_ceiling": ceil_tot / res_tot}

    args.out.write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
