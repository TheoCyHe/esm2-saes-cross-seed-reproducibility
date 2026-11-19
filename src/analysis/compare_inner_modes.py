"""Measurement 1 -- stage 1 split into CENTRING and SCALING, compared.

Q3 measured that scaling alone reorders essentially nothing (Spearman vs raw
`G` = 0.9961) while centring alone does nearly all of it (0.2582 against
0.2738 for the full transform). So stage 1 is a centring operation with a
cosmetic scale attached, and the retirement question is not
standardize-or-not: it is whether to subtract a per-stratum empirical mean.

The case FOR centring, which the wholesale-retirement argument skipped:
`mu` spans **21.7x** across cells at one firing bin. That is real
chance-level structure the analytic `(AP - p_c)/(1 - p_c)` cannot see, since
`p_c` knows the concept's prevalence and nothing about firing rate or how
clustered the feature's activations are.

The case AGAINST scaling: R6 measured that stage 1 as a whole adds ~8%
cross-seed variance (CV ratio 1.084x, rising for 72.4% of concepts) for a
~7% rank change, on a stage whose scale approximately cancels downstream.

This module reports all variants on the same footing so the intermediate --
centring without scaling -- is measured rather than assumed.

-----------------------------------------------------------------------------
RATIO NORMALIZATION (amendment U1). PREDICTIONS COMMITTED BEFORE THE RUN.
-----------------------------------------------------------------------------
`w = g / mu_s`, GSEA's normalization, is a THIRD operation: M1 measured
centring (inert for the max) and full `z` (18% argmax change, 8.4% CV cost).
The ratio has not been measured, and because `mu` spans 21.7x across cells it
is NOT predicted to be inert.

  The author's prediction: cross-seed CV of `M_ratio` comparable to raw `g` and
  well below `z`'s 1.084x, because `mu` is better conditioned than `sigma`.

  My prediction, and it differs on one point. I expect the CV to come out
  BETWEEN raw `g` and `z` -- closer to raw, but not equal to it. Reason:
  `mu_s` is estimated per seed from the same finite draws `sigma_s` is, so it
  carries seed noise too; a mean is simply better conditioned than an SD, not
  noise-free. Predicted median CV ratio vs none in the range 1.00-1.04,
  against `z`'s 1.084.

  Predicted reordering: SUBSTANTIAL and larger than `z`'s. Spearman(M_ratio,
  M_raw) below 0.93 and argmax agreement below 80%, because dividing by a
  quantity spanning 21.7x moves the winner more than dividing by one spanning
  38x does when the latter is order-preserving within stratum (S0a).

  Predicted ceiling: NONE. The statistic is continuous and uncensored, so the
  61.4x raw-`G` range inside the old exceedance ceiling stays fully ordered.

  WHAT KILLS THIS ROUTE: a median CV ratio not materially below `z`'s 1.084.
  Then the mechanism argument fails and ratio joins the other four.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr


def load(path: Path):
    with h5py.File(path, "r") as h:
        return {"M": h["M"][:], "gtilde": h["gtilde"][:],
                "argmax": h["argmax_concept"][:]}


def cross_seed_cv_by_concept(gt_path: Path, n_concepts: int):
    """Cross-seed CV of max-over-features per CONCEPT.

    Concepts have stable cross-seed identity; feature indices do not
    (gotcha 18), so this is the only valid cross-seed comparison available
    for a per-(feature, concept) quantity.
    """
    with h5py.File(gt_path, "r") as h:
        n_seeds = h["z"].shape[0]
        zmax = np.empty((n_seeds, n_concepts))
        for s in range(n_seeds):
            zmax[s] = np.nanmax(h["z"][s][:], axis=0)
    m = zmax.mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(m) > 0, zmax.std(axis=0, ddof=1) / np.abs(m),
                        np.nan)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", type=Path, required=True)
    p.add_argument("--centre", type=Path, required=True)
    p.add_argument("--none", type=Path, required=True)
    p.add_argument("--ratio", type=Path, default=None,
                   help="amendment U1's g/mu variant, if computed")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    with h5py.File(args.full, "r") as h:
        pool = h["in_argmax_pool"][:].astype(bool)
        concepts = [c.decode() if isinstance(c, bytes) else c
                    for c in h["concepts"][:]]
    n_c = len(concepts)
    pi = np.flatnonzero(pool)

    v = {"full": load(args.full), "centre": load(args.centre),
         "none": load(args.none)}
    if args.ratio is not None:
        v["ratio"] = load(args.ratio)
    n_seeds, n_feats = v["full"]["M"].shape

    print("=" * 74)
    print("M1. STAGE 1 SPLIT: full (g-mu)/sigma | centre (g-mu) | none (g)")
    print("=" * 74)

    print("\n  Spearman between the M vectors (median over seeds):")
    pairs = [("full", "centre"), ("full", "none"), ("centre", "none")]
    if "ratio" in v:
        pairs += [("ratio", "none"), ("ratio", "full")]
    rho = {}
    for a, b in pairs:
        r = [spearmanr(v[a]["M"][s], v[b]["M"][s]).statistic
             for s in range(n_seeds)]
        rho[f"{a}_vs_{b}"] = float(np.median(r))
        print(f"    {a:6s} vs {b:6s}: {np.median(r):.4f}")

    print("\n  argmax agreement (median over seeds):")
    agree = {}
    for a, b in pairs:
        g = [(v[a]["argmax"][s] == v[b]["argmax"][s]).mean()
             for s in range(n_seeds)]
        agree[f"{a}_vs_{b}"] = float(np.median(g))
        print(f"    {a:6s} vs {b:6s}: {100*np.median(g):.2f}%")

    print("\n  inflation share and scale:")
    infl = {}
    for k in v:
        gt = v[k]["gtilde"]
        infl[k] = float(np.nanmean(gt > 5))
        print(f"    {k:6s}: g~ median {np.nanmedian(gt):7.3f}   "
              f"frac > 5 {100*np.nanmean(gt > 5):5.2f}%   "
              f"max {np.nanmax(gt):10.1f}")

    print("\n  CROSS-SEED CV of max-over-features per concept "
          f"({len(pi)} pool concepts):")
    cvs = {}
    paths = [("full", args.full), ("centre", args.centre),
             ("none", args.none)]
    if args.ratio is not None:
        paths.append(("ratio", args.ratio))
    for k, path in paths:
        c = cross_seed_cv_by_concept(path, n_c)[pi]
        cvs[k] = c
        print(f"    {k:6s}: median {np.nanmedian(c):.4f}   "
              f"p90 {np.nanpercentile(c, 90):.4f}")
    base = cvs["none"]
    # BOOTSTRAP CI ON THE MEDIAN CV RATIO (amendment AC10). The median summarises
    # ~703 concepts and has been quoted bare -- V1's 1.0069 came with "rises for
    # 57.8% of concepts", barely above a coin flip, so the point estimate alone
    # overstates what is known. Concepts are the resampling unit because the
    # ratio is a per-concept quantity.
    boot_rng = np.random.default_rng(0)
    ci = {}
    for k in [x for x in cvs if x != "none"]:
        per = cvs[k] / np.where(base > 0, base, np.nan)
        ok = per[np.isfinite(per)]
        ratio = float(np.nanmedian(per))
        rises = float(np.nanmean(cvs[k] > base))
        if ok.size:
            draws = np.median(
                boot_rng.choice(ok, (2000, ok.size), replace=True), axis=1)
        else:
            draws = np.array([np.nan])
        lo = float(np.percentile(draws, 2.5))
        hi = float(np.percentile(draws, 97.5))
        ci[k] = {"median_cv_ratio": ratio, "ci95": [lo, hi],
                 "boot_se": float(draws.std(ddof=1)),
                 "frac_concepts_rising": rises, "n_concepts": int(ok.size)}
        print(f"    {k:6s} vs none: median CV ratio {ratio:.4f} "
              f"[95% CI {lo:.4f}, {hi:.4f}], "
              f"rises for {100*rises:.1f}% of concepts")

    print("\n  READING:")
    print("    - if centre ~ none on CV but reorders vs none, centring is "
          "free signal;")
    print("    - if centre ~ full on CV, the seed noise is in the CENTRING, "
          "not the scaling,")
    print("      and R6's case for dropping stage 1 wholesale survives.")

    rep = {"spearman_M": rho, "argmax_agreement": agree,
           "frac_gtilde_gt5": infl,
           "median_cv": {k: float(np.nanmedian(c)) for k, c in cvs.items()},
           "cv_ratio_vs_none": {
               k: float(np.nanmedian(cvs[k] / np.where(base > 0, base, np.nan)))
               for k in cvs if k != "none"},
           "cv_ratio_uncertainty": ci}
    args.out.write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
