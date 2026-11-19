"""Close S1's open caveat, disambiguate the CV ratio, and size the shape
parameter's variability. PREDICTIONS COMMITTED BEFORE THE RUN.

-----------------------------------------------------------------------------
ITEM 1 -- DOES sigma-SCALING MAKE THE WINNING CONCEPT MORE REPRODUCIBLE?
-----------------------------------------------------------------------------
S1 dropped stage 1 while recording that sigma-scaling's 18% argmax change is
NOT shown harmful -- only to cost 8.4% cross-seed variance for an unmeasured
benefit. This closes that on a comparison rather than an intuition.

**A CORRECTION TO THE MEASUREMENT AS PROPOSED.** "Cross-seed argmax
agreement" per feature is NOT a computable quantity: feature `f` in seed 0
and feature `f` in seed 7 are unrelated latents sharing an array position
(gotcha 18; measured cross-seed correlation of `fire_rate` by index 0.0009).
Asking whether feature `f` picks the same concept in two seeds compares two
different features. That is the exact error gotcha 21 was just written for,
and it would have produced a plausible number.

What IS computable, because CONCEPT identity is stable across seeds: the
per-seed distribution of WINNERS over the 703 concepts. For each seed, count
how many features take each concept as their argmax; that is a per-seed
summary, and comparing summaries across seeds is the legitimate pattern
(amendment N9). If sigma-scaling finds real structure, the winner
distribution should be MORE reproducible across seeds; if it is noise, less.

  Prediction: sigma-scaling makes the winner distribution LESS reproducible.
  Grounds: `sigma_s` is estimated per seed from finite draws, varies 38x
  across cells, and demonstrably injects cross-seed variance (CV ratio
  1.0837, rising for 72.4% of concepts). A noisy per-seed denominator that
  differs by stratum perturbs which concept wins in a seed-dependent way.

  What would falsify it, and would reopen S1's decision: HIGHER cross-seed
  correlation of the winner distribution under scaling. That would mean the
  18% argmax change is systematic rather than noise, and the seed-variance
  cost is buying something.

-----------------------------------------------------------------------------
ITEM 5a -- THE TWO CV RATIOS ARE DIFFERENT QUANTITIES
-----------------------------------------------------------------------------
S1 reported medians 0.2975 and 0.3342, whose ratio is 1.123, alongside a
reported ratio of 1.0837. Both are correct and they are not the same
statistic: one is the MEDIAN OF PER-CONCEPT RATIOS, the other the RATIO OF
MEDIANS. Reported here side by side and labelled, since 1.08 and 1.12 will
otherwise be compared as though they were one number.

-----------------------------------------------------------------------------
ITEM 3 -- CAN THE GEV SHAPE BE POOLED? Report the skewness SPREAD
-----------------------------------------------------------------------------
S4 reported only "74.0% of features exceed Gumbel's fixed +1.140". The
spread is what decides whether a shape parameter can be pooled at all, which
is the difference between fitting 3 parameters per feature on 20 draws (badly
conditioned, and the failure mode that just killed sigma) and fitting 2 per
feature with a shared shape.

  Prediction: the per-feature skewness distribution is WIDE -- I expect an
  interquartile range spanning roughly 1.0 to 2.6 and a long right tail. If
  so, a single global shape is not defensible, but a coarse grouping might
  be. Partial pooling of the shape alone requires exchangeability only of
  the tail index, which is a far weaker assumption than the full
  exchangeability M3 refuted at 98.1x within-group spread.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr


def winner_counts(argmax: np.ndarray, n_concepts: int) -> np.ndarray:
    """(seeds, concepts) counts of features taking each concept as argmax."""
    n_seeds = argmax.shape[0]
    out = np.zeros((n_seeds, n_concepts), dtype=np.float64)
    for s in range(n_seeds):
        out[s] = np.bincount(argmax[s], minlength=n_concepts)
    return out


def pairwise_agreement(counts: np.ndarray):
    """Mean pairwise correlation of the per-seed winner distributions."""
    n = counts.shape[0]
    sp, pe = [], []
    for i in range(n):
        for j in range(i + 1, n):
            sp.append(spearmanr(counts[i], counts[j]).statistic)
            pe.append(pearsonr(counts[i], counts[j]).statistic)
    return float(np.mean(sp)), float(np.mean(pe))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", type=Path, required=True)
    p.add_argument("--none", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rep: dict = {}

    with h5py.File(args.full, "r") as h:
        pool = h["in_argmax_pool"][:].astype(bool)
        n_c = pool.size
        am_full = h["argmax_concept"][:]
        Mn = h["M_null"][:]
    with h5py.File(args.none, "r") as h:
        am_none = h["argmax_concept"][:]

    # ---------------------------------------------------------- item 1 -----
    print("=" * 74)
    print("ITEM 1. DOES sigma-SCALING MAKE THE WINNER DISTRIBUTION MORE")
    print("        REPRODUCIBLE ACROSS SEEDS?")
    print("=" * 74)
    print("  (per-feature cross-seed argmax agreement is NOT computable --")
    print("   feature index is not a cross-seed identity, gotcha 18/21)")
    res = {}
    for nm, am in (("none  (raw g)", am_none), ("full  ((g-mu)/sigma)", am_full)):
        c = winner_counts(am, n_c)
        sp, pe = pairwise_agreement(c)
        nwin = [int((c[s] > 0).sum()) for s in range(c.shape[0])]
        # concentration of the winner distribution, per seed
        with np.errstate(invalid="ignore", divide="ignore"):
            pr = c / c.sum(axis=1, keepdims=True)
            ent = -np.nansum(np.where(pr > 0, pr * np.log(pr), 0.0), axis=1)
        res[nm] = (sp, pe, nwin, ent)
        print(f"\n  {nm}")
        print(f"    mean pairwise Spearman of winner counts : {sp:.4f}")
        print(f"    mean pairwise Pearson  of winner counts : {pe:.4f}")
        print(f"    distinct concepts winning >=1 feature   : "
              f"median {int(np.median(nwin))} (min {min(nwin)}, max {max(nwin)})")
        print(f"    winner-distribution entropy (nats)      : "
              f"mean {ent.mean():.3f}")
    a = res["none  (raw g)"][0]
    b = res["full  ((g-mu)/sigma)"][0]
    print(f"\n  VERDICT: scaling {'RAISES' if b > a else 'LOWERS'} cross-seed "
          f"reproducibility of the winner distribution "
          f"({a:.4f} -> {b:.4f}, {b - a:+.4f})")
    print("    higher under scaling => the 18% argmax change is systematic, "
          "and S1 reopens;")
    print("    lower  => it is noise, and S1's decision is confirmed on a "
          "measured comparison.")
    rep["item1"] = {"spearman_none": a, "spearman_full": b,
                    "pearson_none": res["none  (raw g)"][1],
                    "pearson_full": res["full  ((g-mu)/sigma)"][1]}

    # ---------------------------------------------------------- item 5a ----
    print("\n" + "=" * 74)
    print("ITEM 5a. THE TWO CV RATIOS ARE DIFFERENT STATISTICS")
    print("=" * 74)
    from src.analysis.compare_inner_modes import cross_seed_cv_by_concept
    pi = np.flatnonzero(pool)
    cv_f = cross_seed_cv_by_concept(args.full, n_c)[pi]
    cv_n = cross_seed_cv_by_concept(args.none, n_c)[pi]
    ratio_of_medians = float(np.nanmedian(cv_f) / np.nanmedian(cv_n))
    median_of_ratios = float(np.nanmedian(
        cv_f / np.where(cv_n > 0, cv_n, np.nan)))
    print(f"  median CV, full = {np.nanmedian(cv_f):.4f}   "
          f"median CV, none = {np.nanmedian(cv_n):.4f}")
    print(f"  RATIO OF MEDIANS  : {ratio_of_medians:.4f}")
    print(f"  MEDIAN OF RATIOS  : {median_of_ratios:.4f}   <- the figure S1 "
          "quotes as 1.0837")
    print("  Both correct; they are not interchangeable. The median of "
          "per-concept\n  ratios is the paired statistic and is the one to "
          "quote for a per-concept claim.")
    rep["item5a"] = {"ratio_of_medians": ratio_of_medians,
                     "median_of_ratios": median_of_ratios}

    # ---------------------------------------------------------- item 3 -----
    print("\n" + "=" * 74)
    print("ITEM 3. PER-FEATURE SKEWNESS SPREAD -- can a GEV shape be pooled?")
    print("=" * 74)
    GUMBEL = 1.1395470994046486
    sk_all = []
    for s in range(Mn.shape[0]):
        x = Mn[s]
        sk_all.append((((x - x.mean(0)) / x.std(0, ddof=1)) ** 3).mean(0))
    sk = np.concatenate(sk_all)
    q = np.nanpercentile(sk, [1, 5, 25, 50, 75, 95, 99])
    print(f"  per-feature skewness over {sk.size:,} (seed, feature) pairs:")
    print(f"    p1 {q[0]:+.3f}  p5 {q[1]:+.3f}  p25 {q[2]:+.3f}  "
          f"MEDIAN {q[3]:+.3f}  p75 {q[4]:+.3f}  p95 {q[5]:+.3f}  "
          f"p99 {q[6]:+.3f}")
    print(f"    IQR width {q[4]-q[2]:.3f}   Gumbel's fixed value {GUMBEL:+.3f}")
    print(f"    share below Gumbel: {100*np.mean(sk < GUMBEL):.1f}%")
    # A hard bound worth stating: sample skewness at n=20 is bounded.
    n = Mn.shape[1]
    bound = (n - 2) / np.sqrt(n - 1)
    print(f"    NOTE: sample skewness at n={n} is bounded by "
          f"+/-{bound:.2f}, so the upper tail is CENSORED by the estimator")
    print(f"    features at >90% of that bound: "
          f"{100*np.mean(sk > 0.9*bound):.2f}%")
    rep["item3"] = {"pct": {k: float(v) for k, v in
                            zip(["p1", "p5", "p25", "p50", "p75", "p95", "p99"], q)},
                    "iqr": float(q[4] - q[2]),
                    "skew_bound_at_n": float(bound)}
    args.out.write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
