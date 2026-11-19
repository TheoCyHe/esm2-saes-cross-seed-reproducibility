"""AK8's GEOMETRIC HALF -- amendment CH2, declared at `68f0851` before this
module existed.

WHAT THIS IS. AK8 defined a discovery set before the design-3 join ran: features
that are highly reproducible (`k >= 36` of 39) AND carry a low alignment
predictor `M` -- "low" meaning the bottom decile WITHIN the feature's own
firing-rate bin. AK8 has two halves. The distributional half ran then. The
geometric half was deferred with one stated blocker, "needs M2", and that
blocker was discharged when M2 ran at CF3c. This is the deferred half.

THE SELECTION IS NOT REIMPLEMENTED HERE. `design3_join.ak8_discovery_mask` is
the one implementation; a second hand-written copy of the within-bin decile cut
is the divergence gotcha 16 exists for. Likewise the spectral statistics come
from `effective_rank.spectrum_statistics` and the null from
`effective_rank.null_curve` -- the same code that produced M2's numbers, so the
two arms are commensurable by construction rather than by inspection.

THE SIZE-MATCHED NULL IS MANDATORY AND CH2c SAYS WHY. This set is on the order
of 10^2 features; M2's quintiles are 2,048 each. `ER/d` depends mechanically on
submatrix width, and CF3c2 measured that dependence DOMINATING a naive contrast:
the endpoint sets read 0.780 against 0.948 and came out at -0.0056 and +0.0025
once each side was referred to a null at its own size. A naive figure from a
~10^2-feature submatrix, compared against anything computed at 2,048, is a large
wrong number. So every seed's set is referred to a null drawn at THAT SEED'S OWN
set size.

THE QUALIFIER THAT TRAVELS WITH EVERY FIGURE (CH2e). The bottom-decile cut on
`M` is taken WITHIN each firing-rate bin, specifically so the set does not
rediscover the firing-rate confound the whole design controls for. Any figure
reported from this set names that construction; the output carries it as a
string rather than leaving it to a reader.

WHAT IS DECLARED, INCLUDING THE NULL OUTCOME (CH2d). Every branch was fixed
before this file existed: outside the null's range -> reported as a difference
in both directions; INSIDE it -> reported as INDISTINGUISHABLE FROM ITS
SIZE-MATCHED NULL, plainly, as a result rather than an omission; inconsistent
across seeds -> the seed count in each direction is reported and no across-seed
summary is given a sign; a seed whose set is too small to draw a null for ->
`NOT_EVALUABLE`, named, and excluded from no denominator silently.

NO MECHANISM IS CLAIMED and NO PREDICTION WAS FILED (CH2d).
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import h5py
import numpy as np

from src.analysis.design3_join import (
    ANALYSIS_MIN_FIRING_ROWS, AK8_MIN_K, AK8_LOW_M_QUANTILE,
    ak8_discovery_mask, firing_rate_edges,
)
from src.analysis.effective_rank import null_curve, spectrum_statistics
from src.analysis.null_pooling import bin_index
from src.analysis.seed_axis import align_to_reference, normalise_seed_labels

# A null needs at least this many features in the set before a draw is
# meaningful. Below it the seed is NOT_EVALUABLE -- a statement of ignorance,
# never a zero and never folded into a denominator.
MIN_SET_FOR_NULL = 10


def rf_counts_seed_labels(hf) -> list[int] | None:
    """The rf-counts seed labels, by the SAME lookup M2 uses.

    `seed_labels` is what `rf_marginal` actually writes and it is an ATTR, not
    a dataset -- CF3d's incomplete lookup refused rather than joining
    positionally, which is the outcome the guard exists for. An attr
    round-trips through `json.dumps` for lists, so it is PARSED back rather
    than iterated: `list()` of a JSON string is a list of characters
    (gotcha 20).
    """
    for key in ("seed_labels", "anchor_seeds", "seeds"):
        if key in hf:
            return [int(x) for x in hf[key][:]]
        if key in hf.attrs:
            return [int(x) for x in json.loads(str(hf.attrs[key]))]
    return None


def compare_to_null(observed: float, null: dict) -> dict:
    """CH2d's branch table, as a total function of the two inputs.

    Gotcha 24: every value the statistic can take has a branch. Gotcha 30: the
    branches are named for what they mean, not only for where the number falls.
    """
    lo, hi = null["min"], null["max"]
    if observed > hi:
        verdict = "ABOVE_ITS_SIZE_MATCHED_NULL"
    elif observed < lo:
        verdict = "BELOW_ITS_SIZE_MATCHED_NULL"
    else:
        verdict = "INDISTINGUISHABLE_FROM_ITS_SIZE_MATCHED_NULL"
    return {
        "observed": float(observed),
        "null_mean": null["mean"], "null_sd": null["sd"],
        "null_min": lo, "null_max": hi, "null_m": null["m"],
        "null_n_draws": null["n_draws"],
        # The DIFFERENCE, not a ratio: gotcha 33 -- a ratio needs intervals on
        # both components and a shared scale, and the null has a spread while
        # the observed value is a point.
        "difference_from_null_mean": float(observed - null["mean"]),
        "verdict": verdict,
    }


def summarise_across_seeds(per_seed: list[dict], key: str) -> dict:
    """Across-seed summary that REFUSES A SIGN when the seeds disagree.

    CH2d declares this: "if the direction is inconsistent across the 20 seeds,
    the seed count in each direction is reported, and no across-seed summary is
    given a sign."
    """
    ev = [r for r in per_seed if r["status"] == "EVALUATED"]
    if not ev:
        return {"status": "NOT_EVALUABLE", "n_evaluated": 0,
                "why": "no seed had a set large enough to draw a null for"}
    diffs = np.array([r[key]["difference_from_null_mean"] for r in ev])
    above = sum(1 for r in ev if r[key]["verdict"].startswith("ABOVE"))
    below = sum(1 for r in ev if r[key]["verdict"].startswith("BELOW"))
    inside = sum(1 for r in ev
                 if r[key]["verdict"].startswith("INDISTINGUISHABLE"))
    consistent = (above == 0) or (below == 0)
    return {
        "status": "EVALUATED",
        "n_evaluated": len(ev),
        "n_above_null": above, "n_below_null": below,
        "n_indistinguishable": inside,
        "direction_consistent_across_seeds": bool(consistent),
        "mean_difference_from_null_mean":
            float(diffs.mean()) if consistent else None,
        "sd_difference_from_null_mean":
            float(diffs.std(ddof=1)) if consistent and len(diffs) > 1 else None,
        "min_difference": float(diffs.min()),
        "max_difference": float(diffs.max()),
        "signed_summary_withheld_because": None if consistent else (
            "the direction is INCONSISTENT across seeds "
            f"({above} above, {below} below), so CH2d forbids giving the "
            "across-seed summary a sign"),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rf-counts", type=Path, required=True,
                   help="match_count (n_seeds, n_features) at the reference theta")
    p.add_argument("--gtilde", type=Path, required=True,
                   help="supplies M (hard-bin), firing_rate_edges and n_rows")
    p.add_argument("--null-reductions", type=Path, required=True,
                   help="supplies fire_rate and the seed labels to align on")
    p.add_argument("--ckpts", type=Path, required=True,
                   help="directory of seed_N checkpoints, for W_dec")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--n-compare", type=int, required=True)
    p.add_argument("--null-draws", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260819)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    import torch

    with h5py.File(args.gtilde, "r") as hf:
        M = hf["M"][:]
        gt_seeds = [s.decode() if isinstance(s, bytes) else str(s)
                    for s in hf["seeds"][:]]
        n_rows = int(hf.attrs["n_rows"])
    edges = firing_rate_edges(args.gtilde)

    with h5py.File(args.null_reductions, "r") as hf:
        fire_rate = hf["fire_rate"][:]
        null_seeds = [s.decode() if isinstance(s, bytes) else str(s)
                      for s in hf["seeds"][:]]

    with h5py.File(args.rf_counts, "r") as hf:
        y_raw = hf["match_count"][:]
        rf_labels = rf_counts_seed_labels(hf)

    # ---- SEED AXIS: align on LABELS, never by position (gotcha 29) ---------
    # 18 of 20 rows disagree between a lexicographic and a numeric seed order,
    # and every identity key the gates compare is invariant under a seed
    # permutation, so preflight cannot see it by construction. Where labels are
    # missing we REFUSE rather than fall back to range(n) -- CF3d's guard,
    # which fired correctly and caught exactly this class of defect.
    if rf_labels is None:
        raise SystemExit(
            "the rf-counts artifact carries no seed labels; a positional join "
            "would mis-pair 18 of 20 rows (gotcha 29). Refusing.")
    for s in args.seeds:
        if s not in rf_labels:
            raise SystemExit(f"seed {s} absent from the rf-counts labels")
    y = np.stack([y_raw[rf_labels.index(s)] for s in args.seeds])

    # THREE WRITERS SPELL THIS AXIS THREE WAYS -- `seed_7`, `7`, `b'seed_7'` --
    # which is how the collision stayed invisible. `normalise_seed_labels` is
    # the one place that reconciles them; spelling them by hand here is what
    # made job 22664474 fail, and the guard refusing was the correct outcome.
    anchor_labels = normalise_seed_labels(args.seeds)
    fire_rate = align_to_reference(normalise_seed_labels(null_seeds),
                                   fire_rate, anchor_labels)
    M = align_to_reference(normalise_seed_labels(gt_seeds), M, anchor_labels)

    # ---- the A4a floor and the firing bins, exactly as the join builds them
    thresh = ANALYSIS_MIN_FIRING_ROWS / float(n_rows)
    keep = fire_rate >= thresh
    n_bins = len(edges) - 1
    firing_bin = np.stack([
        np.clip(bin_index(fire_rate[s], edges), 0, n_bins - 1)
        for s in range(fire_rate.shape[0])])

    disc = ak8_discovery_mask(y, M, firing_bin, keep)

    rng = np.random.default_rng(args.seed)
    per_seed: list[dict] = []
    for i, s in enumerate(args.seeds):
        idx = np.flatnonzero(disc[i])
        # Same checkpoint path and same unwrapping M2 uses, so the two arms
        # read identical weights rather than merely similar ones.
        sd = torch.load(args.ckpts / f"seed_{s}" / "model.pt", map_location="cpu")
        sd = sd.get("model", sd)
        W_full = sd["W_dec"].numpy().astype(np.float64)   # (d_model, dict_size)
        d_model = W_full.shape[0]

        if idx.size < MIN_SET_FOR_NULL:
            per_seed.append({
                "seed": int(s), "status": "NOT_EVALUABLE",
                "n_discovery": int(idx.size),
                "why": (f"set size {idx.size} is below MIN_SET_FOR_NULL="
                        f"{MIN_SET_FOR_NULL}; a size-matched null cannot be "
                        "drawn meaningfully. This is a statement of ignorance "
                        "and is excluded from no denominator silently."),
            })
            continue

        obs = spectrum_statistics(W_full[:, idx], d_model)
        null = null_curve(W_full, int(idx.size), args.null_draws, rng, d_model)
        rec = {
            "seed": int(s), "status": "EVALUATED",
            "n_discovery": int(idx.size),
            "observed": obs,
            "ER_l1_over_d": compare_to_null(obs["ER_l1_over_d"],
                                            null["ER_l1_over_d"]),
            "ER_energy_over_d": compare_to_null(obs["ER_energy_over_d"],
                                                null["ER_energy_over_d"]),
            "stable_rank": compare_to_null(obs["stable_rank"],
                                           null["stable_rank"]),
        }
        per_seed.append(rec)
        print(f"[ak8geo] seed {s}: n={idx.size} "
              f"ER_l1/d={obs['ER_l1_over_d']:.5f} "
              f"null={null['ER_l1_over_d']['mean']:.5f} "
              f"-> {rec['ER_l1_over_d']['verdict']}", flush=True)

    out = {
        "what": "AK8's geometric half -- amendment CH2",
        "declared_before_implementation_at": "68f0851 (CH2)",
        "selection": {
            "source": "design3_join.ak8_discovery_mask -- ONE implementation",
            "high_r_f": f"k >= {AK8_MIN_K} of {args.n_compare}",
            "low_M": f"M in the bottom {AK8_LOW_M_QUANTILE:.0%} WITHIN the "
                     f"feature's own firing-rate bin",
            "MANDATORY_QUALIFIER_CH2e": (
                "The bottom-decile cut on M is taken WITHIN each firing-rate "
                "bin, specifically so the set does not rediscover the "
                "firing-rate confound the whole design controls for. Any "
                "figure from this set names this construction."),
        },
        "null": {
            "kind": "random selection of the SAME NUMBER of features from the "
                    "same dictionary, drawn per seed at THAT seed's set size",
            "why_mandatory_CH2c": (
                "ER/d depends mechanically on submatrix width. CF3c2 measured "
                "that dependence dominating a naive contrast: 0.780 vs 0.948 "
                "became -0.0056 and +0.0025 once each side was referred to a "
                "null at its own size. This set is ~10^2 features against "
                "M2's 2,048-feature quintiles."),
            "n_draws": args.null_draws, "rng_seed": args.seed,
        },
        "primary_normalisation": (
            "l1, settled on a READ source at AF1 (Gerasimov Appendix F.1 via "
            "alphaxiv), not provisional -- see CG-C1"),
        "no_mechanism_claim": (
            "No mechanism is asserted for any difference found. ER and stable "
            "rank weight the spectrum differently and are not required to "
            "agree; CF3c1's precedent governs."),
        "no_prediction_filed": True,
        "per_seed": per_seed,
        "across_seeds": {
            k: summarise_across_seeds(per_seed, k)
            for k in ("ER_l1_over_d", "ER_energy_over_d", "stable_rank")
        },
        "provenance": {
            "rf_counts": str(args.rf_counts), "gtilde": str(args.gtilde),
            "null_reductions": str(args.null_reductions),
            "ckpts": str(args.ckpts), "seeds": list(args.seeds),
            "n_compare": args.n_compare,
            "python": platform.python_version(), "numpy": np.__version__,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[ak8geo] wrote {args.out}")


if __name__ == "__main__":
    main()
