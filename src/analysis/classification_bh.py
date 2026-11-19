"""Classification build components 1 and 2 (PREREGISTRATION amendment BO):
the POOLED EXCEEDANCE p-value and the BENJAMINI-HOCHBERG multiple-testing pass
over the raw experimental null.

WHAT THIS IS AND IS NOT. It computes AV6 component 1 -- the FDR half of the
alignment bar -- and nothing else. AV6 component 2, the raw gain floor
`G >= 0.05`, is a SEPARATE CONJUNCT applied at classification time; it is NOT
applied here and must not be. The two are ANDed by the consumer, and folding
the floor in here would silently change what "FDR-significant" means.

---------------------------------------------------------------------------
THE DEPENDENCY RUNS COMPONENT 2 -> COMPONENT 1, WHICH IS THE REVERSE OF THE
ORDER BM-E3 LISTS THEM IN.
---------------------------------------------------------------------------
Benjamini-Hochberg consumes p-values; pooled exceedance is what produces them.
So the two "components" are one streaming pass in practice, and this module
implements them as two separately-testable functions called in dependency
order. BM-E3's numbering is a listing, not a claim about data flow -- but the
inversion is recorded here rather than quietly worked around, because a reader
following BM-E3 literally would try to build the BH pass against nothing.

---------------------------------------------------------------------------
THE STATISTIC, and every choice in it traced to the clause that fixes it.
---------------------------------------------------------------------------
For one (seed, arm, concept):

  * the NULL POOL is that concept's raw null gains over SURVIVING shifts x ALL
    10,240 features -- 204,800 values at 20 surviving shifts (T4: "204,800
    pooled draws per concept per seed"). Pooling is ACROSS FEATURES, which is
    what makes the resolution 1/204,801 rather than 1/21.

  * a DEGENERATE DRAW IS NaN AND IS DROPPED, never imputed and never counted
    (BH2). The raw artifact says so in its own `note` attr. Dropping shrinks
    the denominator, which is BH4's per-(concept, arm) resolution: the
    denominator is `n_finite + 1`, NOT a global 204,801.

  * the p-value is `(#{null >= obs} + 1) / (n_finite + 1)`, the standard
    permutation correction. The `+1`s are not optional: a statistic can never
    be reported as impossible on finite draws. This matches
    `diagnose_stages.exceedance` EXACTLY -- deliberately, since that is the
    implementation T4's adequacy numbers were measured against -- and the
    `>=` is AW1.2's strict-exceedance convention (`concept_repro.py:204`).

  * BH runs at `q = 0.05` over the feature x concept tests IN THE POOL,
    PER (SEED, ARM). m = n_pool_concepts x 10,240 = 2,232,320 at the full
    218-concept pool (BF2, BG4). The family is NOT pooled across seeds: AV5
    reports all four cell counts per seed, and AW3.2's adequacy argument is
    stated at the per-seed m.

A concept that is NOT_EVALUABLE under an arm (BH3: EVERY draw degenerate) has
no reference distribution, so its tests CANNOT ENTER THE FAMILY -- they are
excluded from m and carried as their own count. That count is written even
when it is zero (BH3, BI4, criterion 6): an empty category printed as 0 is a
measurement, an absent one is a lost distinction.

`BINDING:hypoxanthine` under Null B is NOT such a concept. It has 19 of 20
surviving draws, so it is EVALUABLE with a coarsened denominator (194,561
rather than 204,801). See `bo_degenerate_cell_check` below, which is the
founding-instance check criterion 4 requires.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

# AV6 component 1's level, ratified unchanged. Not a tunable.
Q_FDR = 0.05

# BH3 vocabulary, reused verbatim from `null_pilot.classify_bh3` so the two
# cannot drift apart. Imported rather than restated for that reason.
from src.analysis.null_pilot import (  # noqa: E402
    BH3_BOTH_ARMS,
    BH3_NONE,
    BH3_ONE_ARM,
    BH3_SOME,
)
from src.sae.train import git_dirty, git_sha  # noqa: E402

__all__ = [
    "Q_FDR",
    "pooled_exceedance",
    "bh_reject",
    "bh_pass_one_slab",
    "NotEvaluable",
]


class NotEvaluable(Exception):
    """Raised when a (concept, arm) has NO surviving draw.

    An exception rather than a sentinel value, because BH3 requires the case
    to be carried as its OWN category: a returned NaN would be one `nanmean`
    away from silently becoming a non-detection, which is the exact failure
    BH3's rule exists to prevent.
    """


def pooled_exceedance(null_pool: np.ndarray,
                      g_obs: np.ndarray) -> tuple[np.ndarray, int, float]:
    """COMPONENT 2. Pooled-exceedance p-values for one (seed, arm, concept).

    `null_pool` is that concept's raw null gains, any shape -- it is flattened.
    NaN entries are DEGENERATE DRAWS and are dropped (BH2). `g_obs` is the
    observed gain per feature, shape (n_features,).

    Returns `(p, n_finite, resolution)` where `resolution = 1/(n_finite+1)` is
    the smallest attainable p -- the CEILING, and the number BH4 makes a
    per-(concept, arm) property.

    Raises `NotEvaluable` if no draw survives.
    """
    pool = np.asarray(null_pool).reshape(-1)
    finite = np.isfinite(pool)
    pool = pool[finite]
    n = int(pool.size)
    if n == 0:
        raise NotEvaluable(
            "every draw is degenerate: no reference distribution exists for "
            "this (concept, arm). BH3 requires NOT_EVALUABLE, never a value.")
    pool = np.sort(pool)
    # count of null draws >= each observed value; `side="left"` is what makes
    # this `>=` rather than `>`.
    n_ge = n - np.searchsorted(pool, np.asarray(g_obs), side="left")
    p = (n_ge + 1.0) / (n + 1.0)
    return p, n, 1.0 / (n + 1.0)


def bh_reject(pvals: np.ndarray, q: float = Q_FDR) -> dict:
    """COMPONENT 1. Benjamini-Hochberg at level `q` over a FLAT family.

    `pvals` is the whole family, flattened by the caller so that this function
    cannot make an assumption about which axes the family spans -- the family's
    definition lives at the CALL SITE, where it is auditable, not in here.

    Returns a dict with `n_tests`, `n_rejected`, `threshold` (the realised
    p-value cut, `None` when nothing is rejected) and `reject`, a boolean array
    in the INPUT's order.

    THE RULE: reject H_(i) for all i <= k, where k is the LARGEST i with
    p_(i) <= (i/m)*q. Note "largest i", not "first failure" -- the step-up
    procedure rejects everything below the last crossing, including p-values
    that individually exceed their own (i/m)*q. Writing it as a scan-until-
    failure is the classic BH implementation bug and yields a STRICT SUBSET of
    the correct rejections, which looks conservative and is simply wrong.
    """
    p = np.asarray(pvals, dtype=np.float64).reshape(-1)
    m = int(p.size)
    if m == 0:
        return {"n_tests": 0, "n_rejected": 0, "threshold": None,
                "reject": np.zeros(0, dtype=bool), "q": float(q)}
    order = np.argsort(p, kind="stable")
    srt = p[order]
    crit = (np.arange(1, m + 1) / m) * q
    passing = np.flatnonzero(srt <= crit)
    reject = np.zeros(m, dtype=bool)
    if passing.size == 0:
        return {"n_tests": m, "n_rejected": 0, "threshold": None,
                "reject": reject, "q": float(q)}
    k = int(passing[-1])                       # LARGEST crossing index
    reject[order[: k + 1]] = True
    return {"n_tests": m, "n_rejected": k + 1, "threshold": float(srt[k]),
            "q": float(q), "reject": reject}


def bh_pass_one_slab(slab: np.ndarray, g_obs: np.ndarray,
                     concept_evaluable: np.ndarray,
                     q: float = Q_FDR) -> dict:
    """Both components for ONE (seed, arm): p-values then BH over the family.

    `slab` is (n_shifts, n_concepts, n_features) -- the raw null for one seed
    and arm. `g_obs` is (n_features, n_concepts). `concept_evaluable` is a
    per-concept bool; False concepts are EXCLUDED FROM THE FAMILY and counted.

    The family is every (feature, concept) test whose concept is evaluable.
    Non-evaluable concepts get p = NaN, which is never fed to `bh_reject`.
    """
    n_shifts, n_c, n_f = slab.shape
    if g_obs.shape != (n_f, n_c):
        raise ValueError(
            f"g_obs is {g_obs.shape}, expected {(n_f, n_c)} -- a transposed "
            "observed array would produce a complete, plausible, wrong result")
    if concept_evaluable.shape != (n_c,):
        raise ValueError(f"concept_evaluable is {concept_evaluable.shape}, "
                         f"expected {(n_c,)}")

    p = np.full((n_f, n_c), np.nan, dtype=np.float64)
    n_finite = np.zeros(n_c, dtype=np.int64)
    resolution = np.full(n_c, np.nan, dtype=np.float64)
    not_evaluable = np.zeros(n_c, dtype=bool)

    for c in range(n_c):
        if not concept_evaluable[c]:
            not_evaluable[c] = True
            continue
        try:
            pc, nfin, res = pooled_exceedance(slab[:, c, :], g_obs[:, c])
        except NotEvaluable:
            # The reductions artifact and the raw array disagree about
            # evaluability. Recorded, never papered over.
            not_evaluable[c] = True
            continue
        p[:, c] = pc
        n_finite[c] = nfin
        resolution[c] = res

    fam = np.zeros((n_f, n_c), dtype=bool)
    fam[:, ~not_evaluable] = True
    res = bh_reject(p[fam], q=q)

    reject = np.zeros((n_f, n_c), dtype=bool)
    reject[fam] = res["reject"]

    return {
        "p": p, "reject": reject, "n_finite": n_finite,
        "resolution": resolution, "not_evaluable": not_evaluable,
        "n_tests": res["n_tests"], "n_rejected": res["n_rejected"],
        "threshold": res["threshold"], "q": float(q),
        "n_concepts_not_evaluable": int(not_evaluable.sum()),
        "n_concepts_in_family": int((~not_evaluable).sum()),
    }


# --------------------------------------------------------------------------
# CLI -- the streaming driver.
# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-null", type=Path, required=True,
                   help="null_gain_raw_experimental_bg.h5 -- (seed, shift, "
                        "concept, feature) gain, both arms.")
    p.add_argument("--null-reductions", type=Path, required=True,
                   help="null_experimental_bg.h5 -- surviving_draws_*, "
                        "not_evaluable_*, concepts.")
    p.add_argument("--ap", type=Path, required=True,
                   help="feature_ap_experimental_av_cpu.h5 -- the OBSERVED "
                        "experimental AP and prevalence.")
    p.add_argument("--stratum", default="experimental")
    p.add_argument("--q", type=float, default=Q_FDR)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=None,
                   help="cost pilot only: process the first N seeds.")
    args = p.parse_args()

    # ---- observed gains, from the AP artifact's own axis -----------------
    with h5py.File(args.ap, "r") as hf:
        grp = hf[args.stratum]
        ap_all = grp["ap"]                        # (seed, feature, concept)
        inpool = grp["in_argmax_pool"][()]
        prev = grp["prevalence"][()][inpool]
        ap_concepts = [c.decode() if isinstance(c, bytes) else str(c)
                       for c in grp["concepts"][()]]
        ap_seeds = [s.decode() if isinstance(s, bytes) else str(s)
                    for s in hf["seeds"][()]]
        n_seed_ap = ap_all.shape[0]
        ap_pool_concepts = [c for c, k in zip(ap_concepts, inpool) if k]
        g_obs_all = np.empty((n_seed_ap, ap_all.shape[1], int(inpool.sum())),
                             dtype=np.float64)
        for s in range(n_seed_ap):
            a = ap_all[s][:, inpool].astype(np.float64)
            g_obs_all[s] = (a - prev[None, :]) / (1.0 - prev[None, :])

    # ---- the null's own axis and evaluability ---------------------------
    with h5py.File(args.null_reductions, "r") as hf:
        null_concepts = [c.decode() if isinstance(c, bytes) else str(c)
                         for c in hf["concepts"][()]]
        null_seeds = [s.decode() if isinstance(s, bytes) else str(s)
                      for s in hf["seeds"][()]]
        surviving = {a: hf[f"surviving_draws_{a}"][()] for a in ("a", "b")}
        not_eval = {a: hf[f"not_evaluable_{a}"][()] for a in ("a", "b")}
        bh3_case = [c.decode() if isinstance(c, bytes) else str(c)
                    for c in hf["bh3_case"][()]]

    # ---- SEED AXIS ALIGNED ON LABELS, NEVER BY POSITION (gotcha 29) ------
    if list(ap_seeds) != list(null_seeds):
        raise SystemExit(
            "SEED AXIS MISMATCH between the AP artifact and the null.\n"
            f"  ap  : {ap_seeds}\n  null: {null_seeds}\n"
            "Joining these by position would mis-pair seeds and yield a "
            "complete, internally consistent, wrong result (gotcha 29).")
    # ---- CONCEPT AXIS POSITIONAL, not set-wise (gotcha 27) ---------------
    if list(ap_pool_concepts) != list(null_concepts):
        same = sorted(ap_pool_concepts) == sorted(null_concepts)
        first = next((i for i, (x, y) in enumerate(
            zip(ap_pool_concepts, null_concepts)) if x != y), None)
        raise SystemExit(
            f"CONCEPT AXIS MISMATCH (same concepts, different order: {same}; "
            f"first disagreement at {first}). Only a POSITIONAL check "
            "discriminates a mapping defect (gotcha 27).")

    with h5py.File(args.raw_null, "r") as hf:
        raw_seeds = [s.decode() if isinstance(s, bytes) else str(s)
                     for s in hf["seeds"][()]]
        raw_concepts = [c.decode() if isinstance(c, bytes) else str(c)
                        for c in hf["concepts"][()]]
        if raw_seeds != null_seeds or raw_concepts != null_concepts:
            raise SystemExit(
                "RAW NULL axis disagrees with the reductions it is paired "
                "with -- refusing.")
        raw_identity = {k: (hf.attrs[k].decode()
                            if isinstance(hf.attrs[k], bytes) else hf.attrs[k])
                        for k in hf.attrs}
        n_seeds = len(raw_seeds) if args.seeds is None else min(
            args.seeds, len(raw_seeds))
        n_c, n_f = len(raw_concepts), hf["gain_a"].shape[3]

        args.out.parent.mkdir(parents=True, exist_ok=True)
        out = h5py.File(args.out, "w")
        per_arm_summary: dict = {}
        for arm in ("a", "b"):
            pv = out.create_dataset(
                f"p_{arm}", shape=(n_seeds, n_f, n_c), dtype="float32",
                chunks=(1, n_f, 1), compression="gzip", compression_opts=1)
            rj = out.create_dataset(
                f"reject_{arm}", shape=(n_seeds, n_f, n_c), dtype=bool,
                chunks=(1, n_f, 1), compression="gzip", compression_opts=1)
            nfin = np.zeros((n_seeds, n_c), dtype=np.int64)
            rows = []
            for s in range(n_seeds):
                slab = hf[f"gain_{arm}"][s]          # (shift, concept, feat)
                r = bh_pass_one_slab(
                    slab, g_obs_all[s], ~not_eval[arm], q=args.q)
                pv[s] = r["p"].astype(np.float32)
                rj[s] = r["reject"]
                nfin[s] = r["n_finite"]
                # BH4 cross-check: the finite count MUST equal the reductions
                # artifact's surviving_draws x dict_size. A disagreement means
                # the raw array and its own reductions describe different runs.
                want = surviving[arm].astype(np.int64) * n_f
                ev = ~not_eval[arm]
                if not np.array_equal(r["n_finite"][ev], want[ev]):
                    bad = np.flatnonzero(r["n_finite"] != want)[:5]
                    raise SystemExit(
                        f"BH4 CROSS-CHECK FAILED, arm {arm.upper()} seed "
                        f"{raw_seeds[s]}: finite draw count disagrees with "
                        f"surviving_draws x {n_f} at concepts "
                        f"{[raw_concepts[i] for i in bad]} "
                        f"(got {r['n_finite'][bad].tolist()}, "
                        f"want {want[bad].tolist()})")
                rows.append({
                    "seed": raw_seeds[s], "n_tests": r["n_tests"],
                    "n_rejected": r["n_rejected"], "threshold": r["threshold"],
                    "n_concepts_in_family": r["n_concepts_in_family"],
                    "n_concepts_not_evaluable": r["n_concepts_not_evaluable"],
                })
                print(f"  arm {arm.upper()} seed {raw_seeds[s]}: "
                      f"m={r['n_tests']:,} rejected={r['n_rejected']:,} "
                      f"thresh={r['threshold']}")
                del slab
            out.create_dataset(f"n_finite_{arm}", data=nfin)
            per_arm_summary[arm.upper()] = rows

        out.create_dataset("concepts", data=np.array(
            raw_concepts, dtype=h5py.string_dtype()))
        out.create_dataset("seeds", data=np.array(
            raw_seeds[:n_seeds], dtype=h5py.string_dtype()))
        out.create_dataset("bh3_case", data=np.array(
            bh3_case, dtype=h5py.string_dtype()))
        for a in ("a", "b"):
            out.create_dataset(f"not_evaluable_{a}", data=not_eval[a])
            out.create_dataset(f"surviving_draws_{a}", data=surviving[a])

        # ---- IDENTITY: inherited from the raw null, NOTHING BACK-FILLED --
        # Device keys are copied ONLY if the source carries them. An artifact
        # predating the key compares ABSENT_ON_BOTH, never AGREE (gotcha 36).
        for k in ("library_sha", "n_concepts", "argmax_pool_n",
                  "argmax_pool_used_n", "scored_row_set", "negatives",
                  "stratum", "truncation_policy", "retention_floor_mode",
                  "retention_min_residues", "retention_min_instances",
                  "min_protein_count", "n_shifts", "null_seed", "null_type",
                  "n_rows", "ap_impl", "gather_device_class",
                  "ap_arith_device_class"):
            if k in raw_identity:
                out.attrs[k] = raw_identity[k]
        out.attrs["q_fdr"] = float(args.q)
        out.attrs["governing_amendment"] = "BO (AV6 component 1; T4; BH2-BH4)"
        out.attrs["family_scope"] = "per (seed, arm); m = n_pool_concepts x n_features"
        out.attrs["p_convention"] = "(#{null >= obs} + 1) / (n_finite + 1)"
        out.attrs["gain_floor_applied"] = False
        out.attrs["git_sha"] = git_sha()
        out.attrs["git_dirty"] = bool(git_dirty())
        out.attrs["raw_null_path"] = str(args.raw_null)
        out.attrs["ap_path"] = str(args.ap)
        out.attrs["reductions_path"] = str(args.null_reductions)
        out.attrs["note"] = (
            "FDR half of AV6's bar ONLY. The raw gain floor G >= 0.05 is NOT "
            "applied here and must be ANDed by the consumer.")
        out.close()

    report = {
        "artifact": "bo_classification_bh",
        "governing_amendment": "BO",
        "git_sha": git_sha(), "git_dirty": bool(git_dirty()),
        "q": float(args.q),
        "n_seeds": n_seeds, "n_concepts": n_c, "n_features": n_f,
        "family_scope": "per (seed, arm)",
        "per_arm": per_arm_summary,
        "bh3_occupancy": {k: int(sum(1 for x in bh3_case if x == k))
                          for k in (BH3_NONE, BH3_SOME, BH3_ONE_ARM,
                                    BH3_BOTH_ARMS)},
        "n_concepts_not_evaluable": {a.upper(): int(not_eval[a].sum())
                                     for a in ("a", "b")},
        "inputs": {"raw_null": str(args.raw_null), "ap": str(args.ap),
                   "reductions": str(args.null_reductions)},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}\nwrote {args.report}")


if __name__ == "__main__":
    main()
