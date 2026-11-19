"""Masking-fraction distribution across the experimental-stratum concepts.

WHAT THIS ANSWERS (amendment AX, §1a). Amendment AW5 compared three null-mask
variants (V1/V2/V3) on five concepts chosen to span the PREVALENCE range, and
found they differ by at most 0.028% against a 5% concern. But the mechanism by
which the variants can differ at all is MASKING: under V1's fixed mask a shifted
positive landing on a masked row is dropped, so drift should scale with the
fraction of the scored universe that is masked -- NOT with prevalence. Those are
different quantities (gotcha 21's shape: a statistic measured on one reduction
does not license a claim about a different one). A high-prevalence concept whose
annotation is almost entirely experimental has essentially no mask, and all three
variants coincide trivially.

So this module reports the distribution of the MASKING FRACTION

    m(c) = (# rows annotated for c under a NON-experimental tier) / n_rows

over the in-pool experimental concepts, and where AW5's five tested concepts sit
in it.

TWO INDEPENDENT PATHS, deliberately (this project's standing "verify by running"
rule, and gotcha 27's rule that agreement of a total is not agreement of a
mapping):

  A. From the experimental AP artifact's own stored per-concept
     `n_rows_effective`, which is EXACTLY `n_rows - other.size` as written by
     `feature_ap.py` under `--negatives unannotated`.
  B. Recomputed from the label store by the same set difference
     `setdiff1d(pooled_positives, experimental_positives)` that `feature_ap.py`
     performs, restricted to the same scored row set.

They are compared CONCEPT BY CONCEPT (positionally, on the label), not by
summary. A and B disagreeing means one of them is wrong; agreeing on 218
elementwise values is what makes either usable.

A NOTE ON SEALING (amendment AX §1a). Path A exists because the quantity is
ALREADY STORED on disk -- `n_rows_effective` is a dataset of
`feature_ap_experimental_av.h5`. Any prediction filed about this distribution is
therefore not sealed; that is reported by `main()` rather than left implicit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.analysis.artifact_identity import argmax_pool_mask

# AW5's five pilot concepts, pre-declared there by PREVALENCE quantile over the
# 218 in-pool experimental concepts (min / p10 / median / p90 / max by
# experimental positive-residue count). Reproduced verbatim; this module asks
# where they land on a DIFFERENT axis.
AW5_PILOT_CONCEPTS = (
    "BINDING:Mo-molybdopterin",
    "BINDING:tRNA",
    "REGION:Hydrophobic repeat HR-C",
    "DOMAIN:VWFA",
    "TRANSMEM",
)


# ----------------------------------------------------------------- pure part --

def mask_fraction_from_effective(n_rows: int,
                                 n_rows_effective: np.ndarray) -> np.ndarray:
    """m(c) = masked rows / n_rows, from the AP writer's own stored quantity.

    `feature_ap.py` sets `n_eff[st][c] = n_rows - other.size` when masking is
    active and `n_rows` when it is not, so the masked count is recoverable
    exactly by subtraction. Pure, so it can be handed a fixture with a known
    answer (gotcha 13 extension 5).
    """
    n_rows = int(n_rows)
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}")
    eff = np.asarray(n_rows_effective, dtype=np.int64)
    if eff.size and (eff > n_rows).any():
        raise ValueError(
            "n_rows_effective exceeds n_rows -- the two do not come from the "
            "same run (masking can only REMOVE rows)")
    if eff.size and (eff < 0).any():
        raise ValueError("negative n_rows_effective")
    return (n_rows - eff).astype(np.float64) / float(n_rows)


def mask_counts_from_labels(pooled: dict[str, np.ndarray],
                            experimental: dict[str, np.ndarray],
                            concepts: list[str]) -> np.ndarray:
    """Masked-row COUNT per concept, by the same set difference feature_ap does.

    `other = setdiff1d(all_tiers[c], experimental[c])` -- rows carrying a label
    for `c` under some other evidence tier. A concept absent from `pooled` has
    no annotation at all and contributes 0.
    """
    empty = np.zeros(0, dtype=np.int64)
    return np.array(
        [np.setdiff1d(pooled.get(c, empty), experimental.get(c, empty),
                      assume_unique=False).size for c in concepts],
        dtype=np.int64)


def percentile_rank(value: float, sample: np.ndarray) -> float:
    """Share of `sample` STRICTLY BELOW `value`, in percent.

    Reported rather than a rank so ties (many concepts share a fraction of
    exactly 0) do not silently become a rank that reads as a position.
    """
    s = np.asarray(sample, dtype=np.float64)
    if s.size == 0:
        raise ValueError("empty sample -- a percentile rank against nothing is "
                         "not a measurement")
    return 100.0 * float((s < float(value)).sum()) / float(s.size)


def summarise(fracs: np.ndarray) -> dict:
    """Distribution summary. Every reported word is a function of the data."""
    f = np.asarray(fracs, dtype=np.float64)
    if f.size == 0:
        raise ValueError("no concepts -- refusing to summarise an empty set")
    qs = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    return {
        "n_concepts": int(f.size),
        "percentiles": {f"p{q}": float(np.percentile(f, q)) for q in qs},
        "mean": float(f.mean()),
        "n_exactly_zero": int((f == 0.0).sum()),
        "share_exactly_zero": float((f == 0.0).mean()),
        "n_above_1pct": int((f > 0.01).sum()),
        "n_above_5pct": int((f > 0.05).sum()),
        "max_concept_index": int(np.argmax(f)),
    }


def upper_tail_verdict(fracs: np.ndarray, pilot_ranks: dict[str, float]) -> dict:
    """AX §1a's branch, as a TOTAL function of the two statistics it reads.

    Branch A -- the pilot concepts sit in the upper half (median pilot rank
    >= 50), OR masking is uniformly tiny (max fraction below 1%): AW5's
    conclusion generalises. Which of the two holds is reported, because they
    generalise for DIFFERENT reasons: "uniformly tiny" generalises by
    arithmetic (there is almost nothing to drop anywhere), "upper half"
    generalises by sampling (the pilot probed the hard end).

    Branch B -- the pilot sits low AND there is a substantial upper tail:
    re-run the variant comparison on the highest-masking concepts.

    The third outcome (pilot low, no substantial tail) is NOT left to a reading:
    it is named explicitly, because a rule silent on a region of its own outcome
    space actively points away from the right answer when it fires there
    (gotcha 24), and a reading defined on only some causes of a branch is the
    same defect one level up (gotcha 30).
    """
    f = np.asarray(fracs, dtype=np.float64)
    ranks = np.array(list(pilot_ranks.values()), dtype=np.float64)
    median_pilot_rank = float(np.median(ranks))
    max_frac = float(f.max())
    uniformly_tiny = bool(max_frac < 0.01)
    pilot_upper_half = bool(median_pilot_rank >= 50.0)
    # "substantial upper tail" is fixed here, before the numbers are seen by a
    # reader, as: the top percentile of the distribution masks more than 1% of
    # the universe -- i.e. there exists a region where V1 has room to drift.
    substantial_tail = bool(float(np.percentile(f, 99)) > 0.01)

    if uniformly_tiny:
        branch, reason = "A", "MASKING_UNIFORMLY_TINY"
    elif pilot_upper_half:
        branch, reason = "A", "PILOT_IN_UPPER_HALF"
    elif substantial_tail:
        branch, reason = "B", "PILOT_LOW_AND_SUBSTANTIAL_TAIL"
    else:
        branch, reason = "A", "PILOT_LOW_BUT_NO_SUBSTANTIAL_TAIL"
    return {
        "branch": branch,
        "reason": reason,
        "median_pilot_percentile_rank": median_pilot_rank,
        "max_masking_fraction": max_frac,
        "p99_masking_fraction": float(np.percentile(f, 99)),
        "uniformly_tiny_lt_1pct": uniformly_tiny,
        "pilot_median_rank_ge_50": pilot_upper_half,
        "substantial_upper_tail_p99_gt_1pct": substantial_tail,
    }


# ---------------------------------------------------------------- the driver --

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ap-experimental", type=Path, required=True,
                   help="feature_ap_experimental_av.h5 -- path A's source")
    p.add_argument("--labels", type=Path, required=True,
                   help="the label store -- path B's source, recomputed "
                        "independently and compared elementwise")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    # ---- path A -----------------------------------------------------------
    with h5py.File(args.ap_experimental, "r") as hf:
        g = hf["experimental"]
        concepts = [c.decode() if isinstance(c, bytes) else c
                    for c in g["concepts"][:]]
        n_eff = g["n_rows_effective"][:]
        in_pool_stored = g["in_argmax_pool"][:].astype(bool)
        n_positives = g["n_positives"][:]
        n_rows = int(hf.attrs["n_rows"])
        negatives = hf.attrs.get("negatives")
        negatives = negatives.decode() if isinstance(negatives, bytes) else negatives
        identity = {k: (v.decode() if isinstance(v, bytes) else
                        (v.item() if isinstance(v, np.generic) else v))
                    for k, v in hf.attrs.items()}

    if negatives != "unannotated":
        raise SystemExit(
            f"FATAL: this artifact was written with negatives={negatives!r}. "
            "Masking is only active under 'unannotated'; `n_rows_effective` "
            "would be a constant and the masking fraction identically zero. "
            "Refusing to report a structural zero as a measurement.")

    # gotcha 15: the stored pool mask is an EXISTENCE claim; recompute it and
    # require agreement rather than trusting the dataset.
    in_pool_recomputed = np.array(argmax_pool_mask(concepts), dtype=bool)
    if not np.array_equal(in_pool_stored, in_pool_recomputed):
        raise SystemExit(
            "FATAL: the artifact's stored `in_argmax_pool` disagrees with "
            f"`argmax_pool_mask` recomputed from its own concept names "
            f"({int((in_pool_stored != in_pool_recomputed).sum())} cells). "
            "Refusing -- one of the two is wrong and this module cannot say "
            "which.")

    frac_all = mask_fraction_from_effective(n_rows, n_eff)

    # The 218: in the argmax pool AND carrying >= 1 experimental residue. Every
    # concept in this artifact was scored because it has experimental positives,
    # so the pool mask is the only filter -- asserted, not assumed.
    if not (n_positives > 0).all():
        raise SystemExit("FATAL: a scored experimental concept has 0 positives")
    sel = in_pool_recomputed
    pool_concepts = [c for c, keep in zip(concepts, sel) if keep]
    frac = frac_all[sel]
    print(f"scored concepts: {len(concepts)}; in argmax pool: {len(pool_concepts)}; "
          f"excluded pooled parents: {len(concepts) - len(pool_concepts)}")

    # ---- path B: recompute from the label store, independently -------------
    from src.analysis.feature_ap import load_labels, load_scored_row_set
    scored_row_set = identity.get("scored_row_set")
    subset = load_scored_row_set(args.labels, scored_row_set)
    remap = None
    if subset is not None:
        remap = -np.ones(int(subset.max()) + 1, dtype=np.int64)
        remap[subset] = np.arange(len(subset))

    def to_subset(d):
        if remap is None:
            return d
        out = {}
        for c, arr in d.items():
            a = arr[arr < len(remap)]
            m = remap[a]
            m = m[m >= 0]
            if m.size:
                out[c] = m
        return out

    pooled_lab = to_subset(load_labels(args.labels, "pooled")[0])
    exp_lab = to_subset(load_labels(args.labels, "experimental")[0])
    n_rows_b = len(subset) if subset is not None else int(
        h5py.File(args.labels, "r").attrs["n_rows"])
    if n_rows_b != n_rows:
        raise SystemExit(
            f"FATAL: scored-universe size disagrees -- AP artifact says "
            f"{n_rows:,}, label store under scored_row_set={scored_row_set!r} "
            f"says {n_rows_b:,}. The two paths are not measuring the same "
            "universe.")
    counts_b = mask_counts_from_labels(pooled_lab, exp_lab, pool_concepts)
    frac_b = counts_b.astype(np.float64) / float(n_rows)

    # Elementwise, on the label -- a total or a correlation cannot see a
    # permutation (gotcha 27).
    n_disagree = int((counts_b != (n_rows - n_eff[sel])).sum())
    worst = float(np.abs(frac - frac_b).max()) if frac.size else 0.0
    print(f"path A vs path B: {len(pool_concepts) - n_disagree}/"
          f"{len(pool_concepts)} concepts agree exactly; "
          f"max |A-B| = {worst:.3e}")
    if n_disagree:
        bad = [(pool_concepts[i], int(n_rows - n_eff[sel][i]), int(counts_b[i]))
               for i in np.nonzero(counts_b != (n_rows - n_eff[sel]))[0][:10]]
        raise SystemExit(
            f"FATAL: {n_disagree} concepts disagree between the stored "
            f"`n_rows_effective` and the recomputed set difference. First 10 "
            f"(concept, A, B): {bad}")

    # ---- the five pilot concepts ------------------------------------------
    idx = {c: i for i, c in enumerate(pool_concepts)}
    pilot = {}
    for c in AW5_PILOT_CONCEPTS:
        if c not in idx:
            raise SystemExit(
                f"FATAL: AW5 pilot concept {c!r} is not among the "
                f"{len(pool_concepts)} in-pool experimental concepts. The "
                "pilot and this distribution are not on the same axis.")
        i = idx[c]
        pilot[c] = {
            "masked_rows": int(counts_b[i]),
            "masking_fraction": float(frac[i]),
            "percentile_rank": percentile_rank(frac[i], frac),
            "n_experimental_positives": int(n_positives[sel][i]),
        }

    summary = summarise(frac)
    pilot_ranks = {c: v["percentile_rank"] for c, v in pilot.items()}
    verdict = upper_tail_verdict(frac, pilot_ranks)

    median = summary["percentiles"]["p50"]
    ax_p1 = {
        "statement": "median masking fraction across the in-pool experimental "
                     "concepts is below 1% of the scored universe",
        "measured_median": median,
        "outcome": "BELOW_1PCT" if median < 0.01 else "AT_OR_ABOVE_1PCT",
        "sealing_status": "NOT_SEALED",
        "sealing_reason": (
            "the quantity is stored on disk as "
            f"{args.ap_experimental.name}:experimental/n_rows_effective "
            "together with the `n_rows` identity attr; the masking fraction is "
            "one subtraction and one division away and was computable before "
            "the prediction was filed"),
    }

    out = {
        "n_rows_scored_universe": n_rows,
        "scored_row_set": scored_row_set,
        "negatives": negatives,
        "n_scored_concepts": len(concepts),
        "n_in_pool_concepts": len(pool_concepts),
        "distribution": summary,
        "pilot_concepts": pilot,
        "branch": verdict,
        "AX_P1": ax_p1,
        "path_agreement": {"n_concepts": len(pool_concepts),
                           "n_disagreements": n_disagree,
                           "max_abs_difference": worst},
        "source_identity": identity,
        "top_10_masking": [
            {"concept": pool_concepts[i], "masking_fraction": float(frac[i]),
             "masked_rows": int(counts_b[i])}
            for i in np.argsort(-frac)[:10]],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    print(f"\nmasking fraction over {len(pool_concepts)} in-pool experimental "
          f"concepts (universe {n_rows:,} rows):")
    for q, v in summary["percentiles"].items():
        print(f"  {q:>5}: {v:.6%}")
    print(f"  exactly zero: {summary['n_exactly_zero']} "
          f"({summary['share_exactly_zero']:.1%}); "
          f"above 1%: {summary['n_above_1pct']}; above 5%: {summary['n_above_5pct']}")
    print("\nAW5's five pilot concepts on the MASKING axis:")
    for c, v in pilot.items():
        print(f"  {c:<36} {v['masking_fraction']:>10.6%}  "
              f"percentile rank {v['percentile_rank']:>6.2f}  "
              f"({v['masked_rows']:,} masked rows)")
    print(f"\nBRANCH {verdict['branch']} ({verdict['reason']})")
    print(f"AX-P1: median {median:.6%} -> {ax_p1['outcome']} "
          f"[{ax_p1['sealing_status']}]")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
