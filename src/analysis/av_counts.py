"""
Amendment AV §4 — the two counts (AV11).

Both are facts about the ANNOTATION VOCABULARY or about the ALIGNMENT SIDE.
Neither touches reproducibility, which is why neither raises the concern AV7
governs: the ALIGNED/ORPHAN definitions are already fixed and do not move when
these numbers arrive.

  §4a (AV11a). Of the 703 concepts in the screening pool, how many have at
      least one EXPERIMENTALLY-VALIDATED residue in the current population? A
      concept with none cannot contribute an aligned feature, so this bounds the
      aligned group before the classification exists. It is a property of
      Swiss-Prot, not of our results. Sealed prediction: AV-P1.

  §4b (AV11b). What share of feature-concept pairs clears the pre-registered
      gain floor `G = (AP - p) / (1 - p) >= 0.05` on the POOLED stratum? The
      bar's two halves cost very differently — the floor is a subtraction on an
      array already on disk, the FDR half needs a streaming pass over 23 GB of
      raw null and, on the experimental stratum, the ~1,980 GPU-hours AV10e
      measured. Sealed prediction: AV-P2.

TWO INVARIANTS THIS MODULE WILL NOT BEND.

  * **`p` is the concept's own prevalence against the EFFECTIVE row count.**
    Under `--negatives unannotated` the denominator differs per concept, and
    C4's invariant is that `G` uses the same denominator the AP was computed
    against — never a reused global `p`. This module reads `prevalence` from
    the artifact rather than recomputing it from `n_positives / n_rows`.
  * **The POOL is `in_argmax_pool`, not the library.** `g_f` is a max over the
    pool, the null is a max over the same pool, and the classification bar in
    AV4 is stated over the pool. The library figure is reported second, as a
    secondary, so a reader can see the difference rather than infer it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.analysis.artifact_identity import argmax_pool_mask

EXPERIMENTAL_TIERS = ("experimental", "experimental_highthroughput")


def experimental_residue_counts(labels: Path) -> dict:
    """Per concept: the number of residues carrying DIRECT experimental
    evidence, in the population the store was built against.

    The two experimental tiers are UNIONED with `np.unique`, exactly as
    `feature_ap.load_labels` does, because tiers partition `all` and a residue
    must not be counted twice if both spellings ever appear for one row. The
    tier→group map is `uniprot_sites.TIER_GROUPS`; the two names below are its
    `experimental` group.
    """
    counts, pooled_counts = {}, {}
    with h5py.File(labels, "r") as hf:
        n_rows = int(hf.attrs["n_rows"])
        for concept in hf["concepts"]:
            g = hf["concepts"][concept]
            present = [t for t in EXPERIMENTAL_TIERS if t in g]
            if present:
                arr = np.unique(np.concatenate([g[t][:] for t in present]))
                counts[concept] = int(arr.size)
            else:
                counts[concept] = 0
            pooled_counts[concept] = int(g["all"].shape[0])
    return {"experimental": counts, "pooled": pooled_counts, "n_rows": n_rows}


def summarise_coverage(exp_counts: dict[str, int], pooled_counts: dict[str, int],
                       concepts: list[str], label: str) -> dict:
    """§4a's answer over one concept set, with the distribution beside it.

    Every outcome word below is DERIVED from the numbers it describes
    (gotcha 20) — nothing here asserts an expected reading.
    """
    n = len(concepts)
    vals = np.array([exp_counts.get(c, 0) for c in concepts], dtype=np.int64)
    pooled = np.array([pooled_counts.get(c, 0) for c in concepts], dtype=np.int64)
    nz = vals[vals > 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(pooled > 0, vals / np.maximum(pooled, 1), np.nan)
    return {
        "set": label,
        "n_concepts": n,
        "n_with_at_least_one_experimental_residue": int((vals > 0).sum()),
        "n_with_none": int((vals == 0).sum()),
        "share_with_at_least_one": float((vals > 0).mean()) if n else 0.0,
        "counts_at_thresholds": {
            f">={t}": int((vals >= t).sum()) for t in (1, 5, 10, 50, 100, 1000)},
        "experimental_residues_per_concept": {
            "total": int(vals.sum()),
            "min_over_nonzero": int(nz.min()) if nz.size else 0,
            "p10": float(np.percentile(nz, 10)) if nz.size else 0.0,
            "median": float(np.median(nz)) if nz.size else 0.0,
            "p90": float(np.percentile(nz, 90)) if nz.size else 0.0,
            "max": int(vals.max()) if n else 0},
        # f_c = P_exp / P_pooled, the quantity C4 measured at 0.006-0.66 and
        # the reason `--negatives all_rows` breaks the gain floor (AV10b).
        "experimental_fraction_f_c": {
            "median_over_nonzero": float(np.nanmedian(frac[vals > 0]))
            if nz.size else 0.0,
            "min_over_nonzero": float(np.nanmin(frac[vals > 0]))
            if nz.size else 0.0,
            "max_over_nonzero": float(np.nanmax(frac[vals > 0]))
            if nz.size else 0.0},
        "top_10_by_experimental_residues": [
            {"concept": c, "experimental": int(exp_counts.get(c, 0)),
             "pooled": int(pooled_counts.get(c, 0))}
            for c in sorted(concepts, key=lambda c: -exp_counts.get(c, 0))[:10]],
    }


def normalised_gain(ap: np.ndarray, prevalence: np.ndarray) -> np.ndarray:
    """`G = (AP - p) / (1 - p)`, broadcast over the concept axis.

    Bounded [0, 1] at every prevalence *when AP >= p*; below chance it goes
    negative, which is retained rather than clipped because clipping would make
    the share-clearing-the-floor insensitive to how far below chance the bulk
    sits, and that is a fact worth reporting.
    """
    if ap.shape[-1] != prevalence.shape[0]:
        raise ValueError(
            f"concept axis mismatch: ap has {ap.shape[-1]}, prevalence has "
            f"{prevalence.shape[0]}. Using a reused or mis-ordered prevalence "
            "vector silently rescales every gain (C4's invariant).")
    return (ap - prevalence[None, :]) / (1.0 - prevalence[None, :])


def gain_floor_share(ap_path: Path, stratum: str, floor: float = 0.05,
                     pool_only: bool = True) -> dict:
    """§4b's answer: the share of (feature, concept) pairs clearing `floor`.

    Computed **per seed and pooled over seeds**. Reported per seed because a
    share that is stable across seeds and a share driven by one seed are
    different facts, and the pooled number alone cannot tell them apart.
    """
    with h5py.File(ap_path, "r") as hf:
        g = hf[stratum]
        concepts = [c.decode() if isinstance(c, bytes) else c
                    for c in g["concepts"][:]]
        prevalence = g["prevalence"][:].astype(np.float64)
        if pool_only:
            mask = (g["in_argmax_pool"][:].astype(bool)
                    if "in_argmax_pool" in g
                    else np.array(argmax_pool_mask(concepts), dtype=bool))
        else:
            mask = np.ones(len(concepts), dtype=bool)
        n_seeds, n_feat, _ = g["ap"].shape
        seeds = [s.decode() if isinstance(s, bytes) else s
                 for s in hf["seeds"][:]]
        idx = np.flatnonzero(mask)
        prev = prevalence[idx]
        per_seed, clears_any = [], np.zeros((n_seeds, n_feat), dtype=bool)
        n_clear_total = 0
        gmax_per_seed = []
        for s in range(n_seeds):
            gs = normalised_gain(g["ap"][s][:, idx].astype(np.float64), prev)
            clear = gs >= floor
            n_clear_total += int(clear.sum())
            clears_any[s] = clear.any(axis=1)
            gmax_per_seed.append(float(gs.max()))
            per_seed.append({
                "seed": seeds[s],
                "n_pairs": int(clear.size),
                "n_clearing": int(clear.sum()),
                "share_clearing": float(clear.mean()),
                "n_features_clearing_at_least_one_concept": int(clears_any[s].sum()),
                "share_of_features_clearing_at_least_one": float(clears_any[s].mean()),
                "median_G": float(np.median(gs)),
                "p99_G": float(np.percentile(gs, 99)),
                "max_G": float(gs.max()),
            })
    n_pairs = n_seeds * n_feat * len(idx)
    shares = np.array([r["share_clearing"] for r in per_seed])
    feat_shares = np.array(
        [r["share_of_features_clearing_at_least_one"] for r in per_seed])
    return {
        "artifact": str(ap_path), "stratum": stratum, "floor": floor,
        "concept_set": "argmax_pool" if pool_only else "library",
        "n_concepts_scored": int(mask.sum()),
        "n_seeds": n_seeds, "n_features": n_feat,
        "n_pairs_total": int(n_pairs),
        "n_pairs_clearing": int(n_clear_total),
        "share_clearing_pooled_over_seeds": float(n_clear_total / n_pairs),
        "share_clearing_per_seed": {
            "min": float(shares.min()), "median": float(np.median(shares)),
            "max": float(shares.max())},
        "share_of_features_clearing_at_least_one_concept": {
            "min": float(feat_shares.min()),
            "median": float(np.median(feat_shares)),
            "max": float(feat_shares.max())},
        "max_G_over_all_seeds": float(max(gmax_per_seed)),
        "per_seed": per_seed,
    }


def score_p1(n_with_experimental: int, threshold: int = 352) -> dict:
    """AV-P1's verdict, DERIVED from the count rather than stated."""
    return {"prediction": f"fewer than {threshold} of 703 pool concepts have "
                          "at least one experimentally-validated residue",
            "threshold": threshold, "observed": int(n_with_experimental),
            "outcome": "HIT" if n_with_experimental < threshold else "MISS"}


def score_p2(share: float, threshold: float = 0.05) -> dict:
    """AV-P2's verdict, DERIVED."""
    return {"prediction": f"fewer than {threshold:.0%} of feature-concept "
                          "pairs clear G >= 0.05",
            "threshold": threshold, "observed": float(share),
            "outcome": "HIT" if share < threshold else "MISS"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--ap", type=Path, required=True)
    p.add_argument("--stratum", default="pooled")
    p.add_argument("--floor", type=float, default=0.05)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    counts = experimental_residue_counts(args.labels)
    with h5py.File(args.ap, "r") as hf:
        ap_concepts = [c.decode() if isinstance(c, bytes) else c
                       for c in hf[args.stratum]["concepts"][:]]
        pool_mask = (hf[args.stratum]["in_argmax_pool"][:].astype(bool)
                     if "in_argmax_pool" in hf[args.stratum]
                     else np.array(argmax_pool_mask(ap_concepts), dtype=bool))
    pool = [c for c, m in zip(ap_concepts, pool_mask) if m]

    # The label store and the AP artifact must name the same library, or "703
    # pool concepts" means two different sets in one report.
    only_store = sorted(set(counts["experimental"]) - set(ap_concepts))
    only_ap = sorted(set(ap_concepts) - set(counts["experimental"]))
    if only_store or only_ap:
        raise SystemExit(
            f"FATAL: the label store and {args.ap.name} disagree on the "
            f"library. Only in store: {only_store[:5]}; only in AP: "
            f"{only_ap[:5]}. Refusing to report counts over two different sets.")

    s4a_pool = summarise_coverage(counts["experimental"], counts["pooled"],
                                  pool, "argmax_pool")
    s4a_lib = summarise_coverage(counts["experimental"], counts["pooled"],
                                 ap_concepts, "library")
    s4b_pool = gain_floor_share(args.ap, args.stratum, args.floor, pool_only=True)
    s4b_lib = gain_floor_share(args.ap, args.stratum, args.floor, pool_only=False)

    report = {
        "amendment": "AV §4 (AV11)",
        "labels": str(args.labels),
        "population_n_rows": counts["n_rows"],
        "s4a_experimental_coverage": {"pool": s4a_pool, "library": s4a_lib},
        "s4b_gain_floor": {"pool": s4b_pool, "library": s4b_lib},
        "AV_P1": score_p1(s4a_pool["n_with_at_least_one_experimental_residue"]),
        "AV_P2": score_p2(s4b_pool["share_clearing_pooled_over_seeds"]),
    }
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("s4a_experimental_coverage", "s4b_gain_floor")},
                     indent=2))
    print(json.dumps(s4a_pool, indent=2))
    print(json.dumps({k: v for k, v in s4b_pool.items() if k != "per_seed"},
                     indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
