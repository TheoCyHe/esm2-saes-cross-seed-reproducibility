"""Construction 2 -- the FEATURE-LEVEL view, reported BESIDE the concept-level
decomposition and explicitly SECONDARY to it (PREREGISTRATION amendment BW).

WHAT IT REPORTS. For each (feature, concept, seed): the ratio of the observed
gain to the MEAN of that pair's 20 null draws, once against the WITHIN-protein
null and once against the ACROSS-protein null. The two are reported as a JOINT
DISTRIBUTION, as a DIFF -- not as a verdict.

**MEANS, NOT PERCENTILES, AND THAT IS THE POINT.** 20 draws estimate a mean far
better than they estimate a tail: the estimability rule (percentile q needs
`n*(100-q)/100 >= 1`) puts the 95th exactly at the boundary from 20 draws, where
it is the maximum wearing a different label. The committed regression predictor
normalises by a null MEAN for exactly this reason. The mean's standard error is
reported alongside every summary so the noise is visible rather than implied.

**WHAT THIS IS NOT, and the constraints are enforced, not merely stated:**

  * **NO multiplicity control, NO cutoff, and NO classification of individual
    features.** This module computes no threshold and emits no per-feature
    label. It is context for the concept-level decomposition and must not be
    quoted as the measure.
  * **NO POOLING ACROSS SEEDS.** A feature index carries no cross-seed identity
    -- measured pairwise correlation of `fire_rate` by index between seeds is
    0.0009 (gotcha 18) -- which is why the matching machinery exists at all.
    Every reduction here is computed WITHIN a seed and summarised across seeds
    only at the summary level, which is the one legitimate combination. The seed
    axis is never averaged over a feature-indexed array.
  * **NO MECHANISM READING.** Nothing here accounts for why the two nulls sit at
    different levels. The difference is reported as a measured diff.

The raw null is streamed ONE SEED AT A TIME: the array is
`(20 seeds, 20 shifts, 703 concepts, 10240 features)` float32, ~11.5 GB per arm
in full, and holding both arms entire would be pointless as well as large when
every reduction is within-seed anyway.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.analysis.artifact_identity import argmax_pool_mask

# Quantiles reported for every distribution. A median alone would hide the tail
# the concept-level measure's maximum actually selects on (gotcha 21).
QUANTILES = (1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0)


def _labels(ds) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in ds[:]]


def ratio_to_null_mean(observed_gain: np.ndarray,
                       null_draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ratio of observed gain to the MEAN of that (concept, feature) pair's own
    null draws, for ONE seed, plus the relative standard error of that mean.

    `observed_gain` is (concept, feature); `null_draws` is (shift, concept,
    feature). Returns (ratio, relative_se), both (concept, feature).

    A pair whose null mean is zero has no defined ratio. It is returned as NaN
    and COUNTED by the caller -- never imputed, never silently dropped into a
    denominator.
    """
    if null_draws.ndim != 3:
        raise ValueError(f"expected (shift, concept, feature), got "
                         f"{null_draws.shape}")
    if observed_gain.shape != null_draws.shape[1:]:
        raise ValueError(f"observed {observed_gain.shape} does not match null "
                         f"{null_draws.shape[1:]}")
    n = null_draws.shape[0]
    mean = null_draws.mean(axis=0, dtype=np.float64)
    # ddof=1: the sample SD of the 20 draws, so the SE is an honest estimate of
    # this mean's own noise rather than an optimistic one.
    sd = null_draws.std(axis=0, ddof=1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(mean > 0, observed_gain / mean, np.nan)
        rel_se = np.where(mean > 0, (sd / np.sqrt(n)) / mean, np.nan)
    return ratio, rel_se


def summarise(x: np.ndarray) -> dict:
    """Quantiles over the finite entries, with the non-finite count reported
    rather than folded away."""
    finite = np.isfinite(x)
    vals = x[finite]
    out = {"n_total": int(x.size), "n_finite": int(finite.sum()),
           "n_nonfinite": int((~finite).sum())}
    if vals.size:
        qs = np.percentile(vals, QUANTILES)
        out["quantiles"] = {f"p{q:g}": float(v) for q, v in zip(QUANTILES, qs)}
        out["mean"] = float(vals.mean())
    return out


def run(ap_path: Path, null_path: Path, raw_path: Path) -> dict:
    with h5py.File(ap_path, "r") as hf:
        grp = hf["pooled"]
        concepts = _labels(grp["concepts"])
        prevalence = grp["prevalence"][:]
        in_pool = np.array(argmax_pool_mask(concepts), dtype=bool)
        ap_seeds = _labels(hf["seeds"])
        ap = grp["ap"]                      # (seed, feature, concept), streamed
        pool_idx = np.flatnonzero(in_pool)
        prev_pool = prevalence[in_pool]

        with h5py.File(null_path, "r") as nf:
            null_concepts = _labels(nf["concepts"])
            null_seeds = _labels(nf["seeds"])
        pool_concepts = [c for c, k in zip(concepts, in_pool) if k]
        if pool_concepts != null_concepts:
            raise SystemExit("FATAL: concept labels disagree (AP pool vs null)")
        if ap_seeds != null_seeds:
            raise SystemExit("FATAL: seed labels disagree (AP vs null)")

        with h5py.File(raw_path, "r") as rf:
            raw_concepts = _labels(rf["concepts"])
            if raw_concepts != pool_concepts:
                raise SystemExit(
                    "FATAL: the raw null's concept labels disagree with the AP "
                    "pool. The raw array carries NO seed labels either, so a "
                    "positional join on this axis would be exactly what gotcha "
                    "29 forbids.")
            ga, gb = rf["gain_a"], rf["gain_b"]
            n_seeds, n_shifts, n_c, n_f = ga.shape

            per_seed = []
            for s in range(n_seeds):
                obs = ap[s][:, pool_idx].astype(np.float64)      # (feature, concept)
                obs_gain = ((obs - prev_pool[None, :])
                            / (1.0 - prev_pool[None, :])).T      # (concept, feature)
                r_w, se_w = ratio_to_null_mean(obs_gain, ga[s].astype(np.float64))
                r_a, se_a = ratio_to_null_mean(obs_gain, gb[s].astype(np.float64))
                both = np.isfinite(r_w) & np.isfinite(r_a)
                # THE DIFF, computed WITHIN the seed. Reported as a difference
                # rather than a ratio-of-ratios: a ratio needs a shared scale and
                # intervals on both components (gotcha 33), and the two arms'
                # nulls sit at different levels by construction.
                diff = np.where(both, r_w - r_a, np.nan)
                per_seed.append({
                    "seed": ap_seeds[s],
                    "ratio_within_protein": summarise(r_w),
                    "ratio_across_protein": summarise(r_a),
                    "difference_within_minus_across": summarise(diff),
                    "relative_se_of_null_mean_within": summarise(se_w),
                    "relative_se_of_null_mean_across": summarise(se_a),
                    "n_pairs_ratio_undefined_within": int((~np.isfinite(r_w)).sum()),
                    "n_pairs_ratio_undefined_across": int((~np.isfinite(r_a)).sum()),
                    "n_pairs_within_ratio_exceeds_across": int(
                        (both & (r_w > r_a)).sum()),
                    "n_pairs_comparable": int(both.sum()),
                })

    return {
        "specification": "PREREGISTRATION amendment BW, Construction 2 -- the "
                         "FEATURE-LEVEL view. SECONDARY to the concept-level "
                         "decomposition and not to be quoted as the measure.",
        "constraints_enforced": {
            "multiplicity_control": "NONE",
            "cutoff_chosen": False,
            "individual_features_classified": False,
            "pooled_across_seeds": False,
            "mechanism_claimed": False,
        },
        "statistic": "observed gain / MEAN of that (concept, feature) pair's 20 "
                     "null draws, computed within each seed",
        "n_pairs_per_seed": int(n_c * n_f),
        "per_seed": per_seed,
        "across_seed_summary": {
            "median_ratio_within_protein": {
                "values_by_seed": [p["ratio_within_protein"]["quantiles"]["p50"]
                                   for p in per_seed]},
            "median_ratio_across_protein": {
                "values_by_seed": [p["ratio_across_protein"]["quantiles"]["p50"]
                                   for p in per_seed]},
            "median_difference": {
                "values_by_seed": [
                    p["difference_within_minus_across"]["quantiles"]["p50"]
                    for p in per_seed]},
            "median_relative_se_within": {
                "values_by_seed": [
                    p["relative_se_of_null_mean_within"]["quantiles"]["p50"]
                    for p in per_seed]},
            "median_relative_se_across": {
                "values_by_seed": [
                    p["relative_se_of_null_mean_across"]["quantiles"]["p50"]
                    for p in per_seed]},
            "_note": "per-seed values listed, NOT averaged into one number. "
                     "Each was computed within its own seed; the list is the "
                     "honest object.",
        },
        "source": {"ap": str(ap_path), "null": str(null_path),
                   "raw_null": str(raw_path)},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ap", type=Path, required=True)
    p.add_argument("--null", type=Path, required=True)
    p.add_argument("--raw-null", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    a = p.parse_args()
    res = run(a.ap, a.null, a.raw_null)
    a.out_json.write_text(json.dumps(res, indent=1, default=str))

    print(json.dumps(res["across_seed_summary"], indent=1))
    p0 = res["per_seed"][0]
    print("\nseed 0 detail:")
    for k in ("ratio_within_protein", "ratio_across_protein",
              "difference_within_minus_across",
              "relative_se_of_null_mean_within",
              "relative_se_of_null_mean_across"):
        print(f"  {k}: {json.dumps(p0[k])}")
    print(f"  n_pairs_comparable: {p0['n_pairs_comparable']}")
    print(f"  n_pairs_within_ratio_exceeds_across: "
          f"{p0['n_pairs_within_ratio_exceeds_across']}")
    print(f"\nwrote {a.out_json}")


if __name__ == "__main__":
    main()
