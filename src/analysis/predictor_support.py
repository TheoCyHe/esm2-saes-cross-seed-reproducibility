"""CHARACTERISING THE PREDICTOR `M`, before the design-3 join.
PREREGISTRATION amendment AL, items AL5 and AL6 (the author's item 3(2) and 3(3)).

TWO QUESTIONS, both about `M = max_c(g_fc / mu_s)` alone. Neither reads `r_f`,
so neither is an X7 firewall crossing: the firewall forbids joining `r_f` to
alignment, and nothing here touches `r_f`.

=============================================================================
AL5 -- IS THE `log M` EXCLUSION RESIDUE EMPTY? (item 3(2))
=============================================================================
The exclusion rule needs a branch for features whose `log M` is not finite. The
right first move is to check whether that set EXISTS before writing a rule for
it -- a rule with an empty domain is not a rule, and adopting one creates a
false impression that exclusions were specified. That is the same defect as
`fire_rate == 0`, whose domain the 40-seed ensemble measured as empty.

U1.4's premise, restated so it can be checked rather than inherited: the pool
max is positive for every feature in every seed, and `mu_s` is bounded away from
zero. If both hold, `M > 0` and finite everywhere and there is nothing to rule
on. **Checked across all 20 anchor seeds here, not the 2 that M3 measured.**

VERDICT SEMANTICS, stated because "empty" is the interesting answer: if the set
is empty it is RECORDED AS EMPTY -- not as a rule that fired zero times, and
not as a rule at all.

=============================================================================
AL6 -- DOES THE PREDICTOR'S UPPER TAIL RIDE ON DEGENERATE CONCEPTS? (item 3(3))
=============================================================================
Small `mu_s` AMPLIFIES. With `mu_s` as low as ~3.36e-07, any concept in that
stratum yields `g / mu_s` of order 1e6 whenever `g` is non-trivial, so it wins
the argmax by orders of magnitude. And M3 characterises the 22 degenerate
concepts as "almost nothing almost always, something large rarely" -- and those
rare occasions are exactly when such a concept would capture the max.

So: what fraction of features have their argmax among the 22, and how does
`log M` distribute for those features against the rest? If the fraction is
material, the predictor's upper tail is degenerate-concept-driven and `log M` is
partly measuring proximity to a degenerate stratum rather than biological
alignment.

**THIS IS A DIFFERENT QUESTION FROM AC4.** AC4 measured the 22 concepts'
LEVERAGE ON `mu_s` -- how much the stratum normaliser moves when they are
excluded from it (median ratio 1.03, max 1.56, i.e. small). This asks whether
they WIN THE ARGMAX, which is a question about the numerator's fate rather than
the denominator's stability, and a concept with negligible leverage on `mu_s`
can still take the max.

REPORT, DO NOT ACT. If it is material it is a the author decision (AL6).
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import h5py
import numpy as np


def check_M_finiteness(M: np.ndarray) -> dict:
    """AL5. Is `log M` finite everywhere, i.e. is the exclusion residue empty?

    Reports the three ways it could fail separately -- non-positive, exactly
    zero, non-finite -- because they would have different causes and lumping
    them would hide which one occurred.
    """
    n_seeds, n_feats = M.shape
    nonfinite = ~np.isfinite(M)
    nonpos = np.isfinite(M) & (M <= 0)
    exact_zero = np.isfinite(M) & (M == 0)
    per_seed = [
        {"seed": int(s),
         "n_nonfinite": int(nonfinite[s].sum()),
         "n_nonpositive": int(nonpos[s].sum()),
         "n_exact_zero": int(exact_zero[s].sum()),
         "min_M": float(M[s][np.isfinite(M[s])].min()) if np.isfinite(M[s]).any()
                  else float("nan")}
        for s in range(n_seeds)
    ]
    n_bad = int(nonfinite.sum() + nonpos.sum())
    return {
        "n_seeds": int(n_seeds),
        "n_features": int(n_feats),
        "n_feature_seed_cells": int(M.size),
        "per_seed": per_seed,
        "n_cells_with_nonfinite_or_nonpositive_M": n_bad,
        "global_min_M": float(M[np.isfinite(M)].min()),
        "global_max_M": float(M[np.isfinite(M)].max()),
        # DERIVED (gotcha 20). "EMPTY" is a distinct verdict from "a rule that
        # fired zero times", and the distinction is the point of AL5.
        "residue_status": "EMPTY" if n_bad == 0 else "NON_EMPTY",
        "reading": (
            "log M is finite and defined for every feature in every anchor "
            "seed, so the exclusion rule needs NO branch for non-finite log M. "
            "Recorded as an empty set rather than as a rule (AL5)."
            if n_bad == 0 else
            "the residue is NON-EMPTY: a branch is required, and the counts "
            "above say how many cells it would cover."),
    }


def check_mu_floor(stratum_mu: np.ndarray, floor: float = 1e-8) -> dict:
    """AL5's other half: is `mu_s` bounded away from zero across ALL seeds?

    M3 measured this on 2 seeds (`mu_all_min` 3.3576e-07 and 2.7271e-07). Twenty
    seeds are checked here, because a bound that holds on two seeds is not a
    bound on the ensemble -- gotcha 19's n=1 lesson at n=2.
    """
    finite = np.isfinite(stratum_mu) & (stratum_mu > 0)
    n_at_or_below = int((np.isfinite(stratum_mu) & (stratum_mu <= floor)).sum())
    return {
        "floor_tested": float(floor),
        "n_strata_total": int(stratum_mu.size),
        "n_strata_finite_positive": int(finite.sum()),
        "n_strata_at_or_below_floor": n_at_or_below,
        "global_min_positive_mu": float(stratum_mu[finite].min()),
        "min_positive_mu_by_seed": [
            float(stratum_mu[s][np.isfinite(stratum_mu[s])
                                & (stratum_mu[s] > 0)].min())
            if (np.isfinite(stratum_mu[s]) & (stratum_mu[s] > 0)).any()
            else float("nan")
            for s in range(stratum_mu.shape[0])],
        "bounded_away_from_zero": bool(n_at_or_below == 0),
    }


def _q(x: np.ndarray) -> dict:
    """Distribution summary. Empty input gives NaNs rather than raising, so an
    empty group is reported as empty instead of crashing the measurement."""
    if x.size == 0:
        return {"n": 0, **{k: float("nan") for k in
                           ("min", "p10", "p25", "median", "p75", "p90",
                            "p99", "max", "mean")}}
    return {
        "n": int(x.size),
        "min": float(np.min(x)), "p10": float(np.percentile(x, 10)),
        "p25": float(np.percentile(x, 25)), "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)), "p90": float(np.percentile(x, 90)),
        "p99": float(np.percentile(x, 99)), "max": float(np.max(x)),
        "mean": float(np.mean(x)),
    }


def degenerate_argmax_share(M: np.ndarray, argmax_concept: np.ndarray,
                            concepts: list[str],
                            degenerate: list[str]) -> dict:
    """AL6. How often do the degenerate concepts WIN the argmax, and with what
    `log M`?

    The comparison that matters is the UPPER TAIL, so the top-decile share is
    reported beside the overall share: a degenerate concept winning for a
    typical feature is uninteresting, one winning for the features with the
    largest `log M` is the failure mode described in AL6.
    """
    name_to_idx = {c: i for i, c in enumerate(concepts)}
    missing = [c for c in degenerate if c not in name_to_idx]
    deg_idx = np.array(sorted(name_to_idx[c] for c in degenerate
                              if c in name_to_idx), dtype=np.int64)
    is_deg = np.zeros(len(concepts), dtype=bool)
    is_deg[deg_idx] = True

    n_seeds = M.shape[0]
    logM = np.log(np.where(M > 0, M, np.nan))
    won = is_deg[argmax_concept]                       # (seeds, features)

    per_seed = []
    for s in range(n_seeds):
        w = won[s]
        lm = logM[s]
        ok = np.isfinite(lm)
        top_cut = np.percentile(lm[ok], 90) if ok.any() else np.nan
        in_top = ok & (lm >= top_cut)
        per_seed.append({
            "seed": int(s),
            "n_features": int(w.size),
            "n_degenerate_argmax": int(w.sum()),
            "share_degenerate_argmax": float(w.mean()),
            "share_in_top_decile_of_logM": (
                float(w[in_top].mean()) if in_top.any() else float("nan")),
            "logM_degenerate": _q(lm[w & ok]),
            "logM_other": _q(lm[~w & ok]),
        })

    shares = np.array([r["share_degenerate_argmax"] for r in per_seed])
    top_shares = np.array([r["share_in_top_decile_of_logM"] for r in per_seed])
    med_deg = np.array([r["logM_degenerate"]["median"] for r in per_seed])
    med_oth = np.array([r["logM_other"]["median"] for r in per_seed])
    return {
        "n_degenerate_concepts": int(deg_idx.size),
        "degenerate_names_not_in_library": missing,
        "per_seed": per_seed,
        "share_across_seeds": {
            "median": float(np.median(shares)), "min": float(shares.min()),
            "max": float(shares.max())},
        "top_decile_share_across_seeds": {
            "median": float(np.nanmedian(top_shares)),
            "min": float(np.nanmin(top_shares)),
            "max": float(np.nanmax(top_shares))},
        "median_logM_shift_degenerate_minus_other": {
            "median_over_seeds": float(np.nanmedian(med_deg - med_oth)),
            "by_seed": [float(v) for v in (med_deg - med_oth)]},
        # The enrichment the question turns on, DERIVED: is the degenerate share
        # higher in the top decile than overall? Ratio, not a sentence.
        "top_decile_enrichment": {
            "median_ratio": float(np.nanmedian(
                np.where(shares > 0, top_shares / np.where(shares > 0, shares, 1),
                         np.nan))),
            "note": ("top-decile share divided by overall share, per seed. "
                     ">1 means degenerate concepts win disproportionately "
                     "among the features with the LARGEST log M, which is the "
                     "configuration AL6 is about. NaN where the overall share "
                     "is zero, since the ratio is undefined there."),
        },
        "reading": (
            "REPORT ONLY. If the share -- especially the top-decile share -- is "
            "material, the predictor's upper tail is degenerate-concept-driven "
            "and log M is partly measuring proximity to a degenerate stratum "
            "rather than biological alignment. Acting on it is a the author decision "
            "(AL6); this module states the numbers."),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gtilde", type=Path, required=True)
    ap.add_argument("--degenerate", type=Path, required=True,
                    help="mu_s_degenerate_m3.json; its "
                         "`degenerate_intersection` names the 22 concepts")
    ap.add_argument("--mu-floor", type=float, default=1e-8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from src.sae.train import git_dirty, git_sha

    with h5py.File(args.gtilde, "r") as hf:
        M = hf["M"][:]
        argmax_concept = hf["argmax_concept"][:]
        concepts = [c.decode() if isinstance(c, bytes) else c
                    for c in hf["concepts"][:]]
        stratum_mu = hf["stratum_mu"][:]
        ident = {k: (v.decode() if isinstance(v, bytes) else v)
                 for k, v in hf.attrs.items()}

    deg_doc = json.loads(args.degenerate.read_text())
    degenerate = deg_doc["degenerate_intersection"]

    finiteness = check_M_finiteness(M)
    mu_floor = check_mu_floor(stratum_mu, args.mu_floor)
    deg = degenerate_argmax_share(M, argmax_concept, concepts, degenerate)

    print(f"[sup] AL5 log M residue: {finiteness['residue_status']} "
          f"({finiteness['n_cells_with_nonfinite_or_nonpositive_M']} of "
          f"{finiteness['n_cells_with_nonfinite_or_nonpositive_M'] or 1} bad "
          f"cells out of {finiteness['n_feature_seed_cells']:,})", flush=True)
    print(f"[sup] M range [{finiteness['global_min_M']:.6g}, "
          f"{finiteness['global_max_M']:.6g}]", flush=True)
    print(f"[sup] AL5 mu_s floor: min positive "
          f"{mu_floor['global_min_positive_mu']:.6g}, "
          f"{mu_floor['n_strata_at_or_below_floor']} strata <= "
          f"{mu_floor['floor_tested']:.0e}, bounded away from zero: "
          f"{mu_floor['bounded_away_from_zero']}", flush=True)
    print(f"[sup] AL6 degenerate argmax share: median "
          f"{deg['share_across_seeds']['median']:.6f} "
          f"(range {deg['share_across_seeds']['min']:.6f}-"
          f"{deg['share_across_seeds']['max']:.6f})", flush=True)
    print(f"[sup] AL6 top-decile share: median "
          f"{deg['top_decile_share_across_seeds']['median']:.6f}; "
          f"enrichment ratio "
          f"{deg['top_decile_enrichment']['median_ratio']:.4f}", flush=True)
    print(f"[sup] AL6 median log M shift (degenerate - other): "
          f"{deg['median_logM_shift_degenerate_minus_other']['median_over_seeds']:.4f}",
          flush=True)

    out = {
        "identity": {
            "measurement": "predictor_support_AL5_AL6",
            "amendment": "AL",
            "firewall_crossing": "NONE -- reads M, argmax_concept, stratum_mu; "
                                 "no r_f, no MNN",
            "gtilde": str(args.gtilde),
            "gtilde_identity": {k: (v.item() if hasattr(v, "item") else v)
                                for k, v in ident.items()},
            "degenerate_source": str(args.degenerate),
            "git_sha": git_sha(), "git_dirty": git_dirty(),
            "python_version": platform.python_version(),
        },
        "AL5_logM_residue": finiteness,
        "AL5_mu_floor": mu_floor,
        "AL6_degenerate_argmax": deg,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[sup] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
