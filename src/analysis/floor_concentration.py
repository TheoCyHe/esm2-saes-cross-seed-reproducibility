"""BR part C: the per-concept decomposition of the BH resolution floor.

WHAT THIS IS, AND WHAT STANDING IT HAS
--------------------------------------
A **descriptive** diagnostic, declared at PREREGISTRATION §BR-C before it was
recomputed, and labelled **COMPUTED_BEFORE_DECLARED** in the artifact's own
attrs. The figures were seen in a prior read-only session whose script no longer
exists and whose only trace is an exit report -- so this is not blind, NO
PREDICTION IS FILED, and the result is excluded from the prediction ledger and
from every tally, exactly as a `NOT_SEALED` entry is (§AX1.4, §AY6.3). Exclusion
from a tally is not permission to omit the number: every figure is reported.

THE QUANTITY
------------
For arm x, seed s, evaluable concept c:

    at_floor[s, c] = #{ f : p_x[s, f, c] == 1 / (n_finite_x[s, c] + 1) }

the number of features whose observed gain exceeds EVERY surviving null draw for
that concept. THE FLOOR IS PER-CONCEPT, NOT GLOBAL: `n_finite` varies with the
concept's surviving-draw count (BH2, BH4), so a single global `1/204801` would be
the wrong comparand for any concept with a degenerate draw.

THE NUMERATOR IS RECOVERED EXACTLY, NOT BY A TOLERANCE
------------------------------------------------------
`p` is stored float32, but every p-value is the rational
`(n_ge + 1)/(n_finite + 1)` with a KNOWN denominator, so

    j = rint( p_f32 * (n_finite + 1) )      and at-floor <=> j == 1

and the recovery is checked by re-deriving the float32 from `j` and demanding the
stored bytes back EXACTLY. This reuses `bp_floor_check.py`'s committed method
verbatim and deliberately does not re-derive it: corrections entry 123 records
that a residual bound here was ILL-POSED and was REPLACED by the exact round-trip
rather than widened, and re-inventing a tolerance would walk back a correction
the record already paid for.

WHAT THIS DOES NOT CLAIM (§BR-C4, and it is load-bearing)
---------------------------------------------------------
No mechanism claim about why the arms differ or why the experimental supply is
concentrated -- §BP9 and §BQ15 both decline that (gotcha 22). No threshold, no
pass/fail, no correlation quoted as a verdict. The positive-sparsity account is a
CANDIDATE THE DIAGNOSTIC DOES NOT ADJUDICATE.

FOUR between-stratum differences are visible here and NONE is adjudicated, so no
between-stratum comparison below is a controlled one:

  1. positive count per concept (reported, and matched on, below);
  2. row masking -- the strata score different row sets;
  3. device class -- `classification_2x2_bq.h5:axis_device_divergence_note`
     records the experimental axis descending from a CPU gather and the pooled
     axis from a GPU-era artifact (gotcha 36);
  4. the AP artifacts' own settings -- `negatives` is `unannotated`
     experimentally and `all_rows` pooled, and `ap_impl` differs between them.

Stated in advance rather than repaired in the reading afterwards (gotcha 30's
extension: a measurement that cannot distinguish two causes cannot be fixed by a
better reading later).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.sae.train import git_dirty, git_sha

# Bin edges on per-concept positive count. FIXED AND DECLARED, not chosen after
# looking at the answer: they are round log-spaced decades, and the lowest edge
# is the library's own retention floor (`retention_min_instances = 10`).
POSITIVE_BINS = [10, 30, 100, 300, 1000, 3000, 10000, np.inf]

# Ranks at which concentration is reported. Declared at §BR-C6.
TOP_K = (1, 5, 10, 25)


def recover_numerators(p: np.ndarray, n_finite: np.ndarray) -> np.ndarray:
    """Exact integer numerators of the stored float32 p-values.

    `p` is (n_features, n_concepts); `n_finite` is (n_concepts,). Returns int64
    of `p`'s shape, with -1 where `p` is not finite (non-evaluable concepts).

    RAISES if the recovery does not round-trip through float32 EXACTLY. This is
    not a tolerance and must never become one -- see the module docstring and
    corrections entry 123.
    """
    if p.ndim != 2:
        raise ValueError(f"p must be (n_features, n_concepts), got {p.shape}")
    if n_finite.shape != (p.shape[1],):
        raise ValueError(
            f"n_finite is {n_finite.shape}, expected {(p.shape[1],)} -- a "
            "transposed or mis-joined denominator would silently produce a "
            "complete, plausible, wrong at-floor count")

    j = np.full(p.shape, -1, dtype=np.int64)
    for d in np.unique(n_finite[n_finite > 0]):
        cols = np.flatnonzero(n_finite == d)
        sub = p[:, cols]
        if not np.all(np.isfinite(sub)):
            # A positive denominator with a NaN p-value means the reductions and
            # the p-array disagree about evaluability. Refused, never skipped.
            raise ValueError(
                f"non-finite p-value at a concept with n_finite={d}: the "
                "artifact's evaluability and its p-values disagree")
        jj = np.rint(sub.astype(np.float64) * (int(d) + 1)).astype(np.int64)
        back = (jj / (int(d) + 1)).astype(np.float32)
        if not np.array_equal(back, sub.astype(np.float32)):
            n_bad = int((back != sub.astype(np.float32)).sum())
            raise ValueError(
                f"NUMERATOR RECOVERY IS NOT EXACT at n_finite={d}: {n_bad} of "
                f"{sub.size} p-values do not round-trip through j/(n_finite+1). "
                "The stored p-values are not the rationals this denominator "
                "implies -- refusing.")
        j[:, cols] = jj
    return j


def at_floor_per_concept(p: np.ndarray, n_finite: np.ndarray) -> np.ndarray:
    """Per-concept count of tests sitting at that concept's own p-floor.

    `p` is (n_features, n_concepts) for ONE (seed, arm). Returns int64
    (n_concepts,). A non-evaluable concept contributes 0 and is masked by the
    caller -- it is NOT the same as a concept that is evaluable and supplies
    none, and the two are reported separately.
    """
    j = recover_numerators(p, n_finite)
    return (j == 1).sum(axis=0).astype(np.int64)


def concentration(totals: np.ndarray, top_k=TOP_K) -> dict:
    """Share of all at-floor tests supplied by the top-k concepts.

    `totals` is per-concept totals over seeds. Returns the share at each k, the
    number of concepts supplying zero, and the denominator -- never a verdict.
    """
    tot = np.asarray(totals, dtype=np.int64)
    grand = int(tot.sum())
    order = np.sort(tot)[::-1]
    out = {
        "n_concepts": int(tot.size),
        "grand_total": grand,
        "n_concepts_zero": int((tot == 0).sum()),
        "share_zero": (float((tot == 0).mean()) if tot.size else float("nan")),
        "max_single_concept": int(tot.max()) if tot.size else 0,
    }
    for k in top_k:
        kk = min(int(k), int(tot.size))
        out[f"top{k}_share"] = (float(order[:kk].sum()) / grand
                                if grand else float("nan"))
    return out


def distribution(at_floor: np.ndarray) -> dict:
    """Summaries of the (seed, concept) at-floor matrix. Descriptive only."""
    a = np.asarray(at_floor, dtype=np.float64)
    return {
        "n_cells": int(a.size),
        "n_seeds": int(a.shape[0]),
        "n_concepts": int(a.shape[1]),
        "mean_per_concept": float(a.mean()) if a.size else float("nan"),
        "median_per_concept": float(np.median(a)) if a.size else float("nan"),
        "max_per_concept": float(a.max()) if a.size else float("nan"),
        "total_per_seed_mean": float(a.sum(axis=1).mean()) if a.size else 0.0,
        "total_per_seed_min": int(a.sum(axis=1).min()) if a.size else 0,
        "total_per_seed_max": int(a.sum(axis=1).max()) if a.size else 0,
    }


def matched_by_positives(at_floor: np.ndarray, n_positives: np.ndarray,
                         bins=POSITIVE_BINS) -> list[dict]:
    """Mean at-floor per concept per seed, within fixed positive-count bins.

    Every row carries its own denominator (`n_concepts` in the bin). A bin with
    no concepts is REPORTED AS EMPTY, never dropped -- an absent row and a zero
    row are different statements.
    """
    a = np.asarray(at_floor, dtype=np.float64)
    pos = np.asarray(n_positives)
    if pos.shape != (a.shape[1],):
        raise ValueError(
            f"n_positives is {pos.shape}, expected {(a.shape[1],)} -- the "
            "positive counts and the at-floor columns must be the SAME "
            "concepts in the SAME order (gotcha 27)")
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (pos >= lo) & (pos < hi)
        n = int(sel.sum())
        rows.append({
            "lo": int(lo), "hi": (None if np.isinf(hi) else int(hi)),
            "n_concepts": n,
            "mean_at_floor_per_concept_per_seed":
                (float(a[:, sel].mean()) if n else None),
            "total_at_floor": (int(a[:, sel].sum()) if n else 0),
        })
    return rows


def _labels(hf, key) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in hf[key][()]]


def _arm_block(p_ds, nfin_all, evaluable, n_seeds, n_positives) -> dict:
    """One arm, streamed per seed. `evaluable` is (n_seeds, n_concepts) bool."""
    n_c = nfin_all.shape[1]
    at = np.zeros((n_seeds, n_c), dtype=np.int64)
    for s in range(n_seeds):
        at[s] = at_floor_per_concept(p_ds[s].astype(np.float64),
                                     nfin_all[s].astype(np.int64))
    # A non-evaluable (seed, concept) contributes nothing and is not counted as
    # an evaluable concept that supplied zero. The two are reported separately.
    at = np.where(evaluable, at, 0)
    ev_cols = evaluable.all(axis=0)
    totals = at[:, ev_cols].sum(axis=0)
    return {
        "at_floor": at,
        "evaluable_all_seeds": ev_cols,
        "distribution": distribution(at[:, ev_cols]),
        "concentration": concentration(totals),
        "n_concepts_not_evaluable_some_seed": int((~ev_cols).sum()),
        "by_positive_count": matched_by_positives(
            at[:, ev_cols], n_positives[ev_cols]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bh-experimental", type=Path, required=True)
    ap.add_argument("--ap-experimental", type=Path, required=True)
    ap.add_argument("--twobytwo-pooled", type=Path, required=True)
    ap.add_argument("--ap-pooled", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    result: dict = {"experimental": {}, "pooled": {}}
    ident: dict = {}

    # ---------------- EXPERIMENTAL ----------------------------------------
    with h5py.File(args.ap_experimental, "r") as hf:
        g = hf["experimental"]
        inpool = g["in_argmax_pool"][()]
        pos_exp = g["n_positives"][()][inpool]
        conc_exp_ap = [c for c, k in zip(_labels(g, "concepts"), inpool) if k]
        ident["ap_experimental_negatives"] = str(
            hf.attrs.get("negatives", "ABSENT"))

    with h5py.File(args.bh_experimental, "r") as hf:
        conc_exp = _labels(hf, "concepts")
        seeds_exp = _labels(hf, "seeds")
        if conc_exp != conc_exp_ap:
            raise SystemExit(
                "CONCEPT AXIS MISMATCH between the BH artifact and the AP "
                "artifact supplying positive counts. Only a POSITIONAL check "
                "discriminates a mapping defect (gotcha 27).")
        n_seeds = len(seeds_exp)
        for k in ("library_sha", "stratum", "n_shifts", "q_fdr",
                  "argmax_pool_used_n", "gather_device_class",
                  "ap_arith_device_class", "git_sha"):
            if k in hf.attrs:
                v = hf.attrs[k]
                ident[f"bh_experimental_{k}"] = (
                    v.decode() if isinstance(v, bytes) else
                    (v.item() if hasattr(v, "item") else v))
        for arm in ("a", "b"):
            nfin = hf[f"n_finite_{arm}"][()]
            # experimental not_evaluable is per CONCEPT (218,), not per seed
            ne = hf[f"not_evaluable_{arm}"][()]
            ev = np.broadcast_to(~ne, (n_seeds, ne.size)).copy()
            blk = _arm_block(hf[f"p_{arm}"], nfin, ev, n_seeds, pos_exp)
            result["experimental"][arm] = {
                k: v for k, v in blk.items() if k != "at_floor"
            }
            result["experimental"][arm]["evaluable_all_seeds"] = int(
                blk["evaluable_all_seeds"].sum())
            result["experimental"][arm]["_at_floor"] = blk["at_floor"]

    # ---------------- POOLED ----------------------------------------------
    with h5py.File(args.ap_pooled, "r") as hf:
        g = hf["pooled"]
        inpool_p = g["in_argmax_pool"][()]
        pos_pool = g["n_positives"][()][inpool_p]
        conc_pool_ap = [c for c, k in zip(_labels(g, "concepts"), inpool_p) if k]
        ident["ap_pooled_negatives"] = str(hf.attrs.get("negatives", "ABSENT"))

    with h5py.File(args.twobytwo_pooled, "r") as hf:
        conc_pool = _labels(hf, "concepts_pooled")
        seeds_pool = _labels(hf, "seeds")
        if conc_pool != conc_pool_ap:
            raise SystemExit(
                "POOLED CONCEPT AXIS MISMATCH between the 2x2 artifact and the "
                "pooled AP artifact (gotcha 27).")
        if seeds_pool != seeds_exp:
            raise SystemExit(
                "SEED AXIS MISMATCH between the experimental and pooled "
                "artifacts. Joining by position would mis-pair seeds (gotcha "
                f"29).\n  exp : {seeds_exp}\n  pool: {seeds_pool}")
        for k in ("axis_device_divergence_note", "k_min_experimental",
                  "k_min_pooled", "git_sha"):
            if k in hf.attrs:
                v = hf.attrs[k]
                ident[f"twobytwo_{k}"] = (
                    v.decode() if isinstance(v, bytes) else
                    (v.item() if hasattr(v, "item") else v))
        for arm in ("a", "b"):
            nfin = hf[f"n_finite_pooled_{arm}"][()]
            ne = hf[f"not_evaluable_pooled_{arm}"][()]   # (n_seeds, n_c)
            blk = _arm_block(hf[f"p_pooled_{arm}"], nfin, ~ne,
                             len(seeds_pool), pos_pool)
            result["pooled"][arm] = {
                k: v for k, v in blk.items() if k != "at_floor"
            }
            result["pooled"][arm]["evaluable_all_seeds"] = int(
                blk["evaluable_all_seeds"].sum())
            result["pooled"][arm]["_at_floor"] = blk["at_floor"]

    # ---------------- artifact -------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as out:
        for stratum, concepts, pos in (
                ("experimental", conc_exp, pos_exp),
                ("pooled", conc_pool, pos_pool)):
            grp = out.create_group(stratum)
            grp.create_dataset("concepts", data=np.array(
                concepts, dtype=h5py.string_dtype()))
            grp.create_dataset("n_positives", data=pos)
            for arm in ("a", "b"):
                grp.create_dataset(
                    f"at_floor_{arm}", data=result[stratum][arm].pop("_at_floor"))
        out.create_dataset("seeds", data=np.array(
            seeds_exp, dtype=h5py.string_dtype()))
        out.attrs["governing_amendment"] = "BR (§BR-C)"
        out.attrs["standing"] = "COMPUTED_BEFORE_DECLARED"
        out.attrs["standing_note"] = (
            "Descriptive. The figures were seen in a prior read-only session "
            "before this was declared, so the filing was NOT blind. Treated as "
            "NOT_SEALED is: reported in full, excluded from every tally and "
            "from the prediction ledger. NO PREDICTION IS FILED.")
        out.attrs["not_claimed"] = (
            "No mechanism claim about why the arms differ or why the "
            "experimental supply is concentrated (BP9, BQ15, gotcha 22). No "
            "threshold, no pass/fail, no correlation as a verdict. Positive "
            "sparsity is a CANDIDATE THIS DOES NOT ADJUDICATE. Row masking, "
            "device class and the AP artifacts' own negatives/ap_impl settings "
            "are further between-stratum differences, none controlled.")
        out.attrs["enters_no_denominator"] = (
            "Informs how the resolution limit is DESCRIBED only. Not part of "
            "the classification result; enters no denominator; does not settle "
            "the reporting framing, which is the author's.")
        out.attrs["at_floor_definition"] = (
            "count over features of p == 1/(n_finite+1) for that (seed, arm, "
            "concept); numerator recovered exactly via rint and verified by an "
            "exact float32 round-trip, never a tolerance (entry 123)")
        out.attrs["input_bh_experimental"] = str(args.bh_experimental)
        out.attrs["input_ap_experimental"] = str(args.ap_experimental)
        out.attrs["input_twobytwo_pooled"] = str(args.twobytwo_pooled)
        out.attrs["input_ap_pooled"] = str(args.ap_pooled)
        out.attrs["git_sha"] = git_sha()
        out.attrs["git_dirty"] = bool(git_dirty())
        for k, v in ident.items():
            out.attrs[k] = v

    report = {
        "artifact": "br_floor_concentration",
        "governing_amendment": "BR",
        "standing": "COMPUTED_BEFORE_DECLARED",
        "identity": ident,
        "seeds": seeds_exp,
        "result": result,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str))

    # ---------------- stdout: a DIFF, not a verdict -----------------------
    for stratum in ("experimental", "pooled"):
        for arm in ("a", "b"):
            r = result[stratum][arm]
            d, c = r["distribution"], r["concentration"]
            print(f"\n=== {stratum.upper()} arm {arm.upper()} ===")
            print(f"  concepts evaluable in all seeds : {r['evaluable_all_seeds']}"
                  f"  (not evaluable in >=1 seed: "
                  f"{r['n_concepts_not_evaluable_some_seed']})")
            print(f"  (seed, concept) cells           : {d['n_cells']:,}")
            print(f"  mean at-floor per concept       : {d['mean_per_concept']:.4f}")
            print(f"  median / max                    : "
                  f"{d['median_per_concept']:.1f} / {d['max_per_concept']:.0f}")
            print(f"  total per seed  min..max        : "
                  f"{d['total_per_seed_min']:,}..{d['total_per_seed_max']:,}"
                  f"  (mean {d['total_per_seed_mean']:.1f})")
            print(f"  concepts supplying ZERO         : {c['n_concepts_zero']}"
                  f" of {c['n_concepts']}  ({c['share_zero']:.1%})")
            print(f"  grand total at-floor tests      : {c['grand_total']:,}")
            for k in TOP_K:
                print(f"  top-{k:<2} concept share           : "
                      f"{c[f'top{k}_share']:.1%}")
            print("  by positive count (mean at-floor per concept per seed):")
            for row in r["by_positive_count"]:
                hi = "inf" if row["hi"] is None else f"{row['hi']:,}"
                m = ("      EMPTY" if row["mean_at_floor_per_concept_per_seed"]
                     is None else
                     f"{row['mean_at_floor_per_concept_per_seed']:>11.4f}")
                print(f"    [{row['lo']:>6,}, {hi:>6}) "
                      f"n={row['n_concepts']:>4}  {m}")
    print(f"\nwrote {args.out}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
