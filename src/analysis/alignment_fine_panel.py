"""AZ section 12 -- THE FINE PANEL INSIDE THE TOP ALIGNMENT DECILE.

=============================================================================
WHY A FINER BIN, AND WHAT IT IS A TEST OF
=============================================================================
The concept-level measure looks at ONE best carrier per concept per seed -- at
most 703 distinct features of 10,240, **under 7% of the dictionary**. The
feature-level result averages a gradient over all 204,800 feature-seed cells.
**A relationship confined to the top few percent of the alignment distribution
is arithmetically compatible with a flat linear gradient across the bulk**, and
it is INVISIBLE at decile resolution, where carriers sit mixed with roughly nine
times as many non-carriers.

`design3_join.model_free_panel` already bins `mean r_f` by `log M` DECILE per
firing stratum. This module re-bins **inside the top decile alone**, at
percentile resolution. It is the direct test of the step reading on the
feature-level data itself, and it is EXPLORATORY (AZ7): it cannot become
primary under any outcome and no mechanism sentence is written for it.

=============================================================================
INTERVALS: EMPIRICAL BETWEEN-FEATURE DISPERSION, AND NO IMPORTED MULTIPLIER
=============================================================================
`sd(r_f within bin) / sqrt(n_bin)`, not a Wilson construction. Features in a bin
do not share a match probability -- that heterogeneity IS the measurement -- and
a Wilson interval describes a pooled proportion under independent trials at a
common rate. The nominal binomial half-width is reported ALONGSIDE so the size
of the difference is visible rather than argued.

**NO MULTIPLIER FROM ANOTHER REDUCTION IS APPLIED** (AZ6.1). The "roughly 2x"
figure that travelled into a planning brief was measured on feature-level decile
means and came out at **0.756-0.937x** on concept-level detection -- the
opposite direction. The ratio is recomputed here on THIS quantity and reported.

=============================================================================
THE SEED AXIS, AND WHY NOTHING IS POOLED ACROSS IT BEFORE BINNING
=============================================================================
`gtilde` is LEXICOGRAPHIC (`0, 1, 10, ...`) and `rf_counts` carries NO `seeds`
dataset at all -- its order is the NUMERIC `seed_labels` attribute. Positions 0
and 1 are the only ones that coincide (gotcha 29). The permutation is taken by
LABEL and asserted.

And binning is done **per seed**, with only the per-seed SUMMARIES combined: a
feature index is not a cross-seed identity (gotcha 18), so a bin defined on
`log M` averaged across seeds would be a bin of 20 unrelated random variables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.analysis.corrector_scan import BIN_EDGES

# A4a, inherited by AL8: features firing on fewer than this many rows are
# outside the analysis set. Measured effect on this ensemble: ZERO features
# excluded in all 20 anchors (AO16). Applied and REPORTED anyway -- the record
# states the floor's measured effect rather than letting a reader assume it
# removed something.
ANALYSIS_MIN_FIRING_ROWS = 200

Z95 = 1.959964


def firing_bin(fire_rate: np.ndarray) -> np.ndarray:
    """This project's nine standing strata (`corrector_scan.BIN_EDGES`)."""
    edges = np.asarray(BIN_EDGES, dtype=np.float64)
    return np.clip(np.digitize(fire_rate, edges[1:-1], right=False),
                   0, len(edges) - 2)


def top_fraction_mask(logM_row: np.ndarray, keep: np.ndarray,
                      fraction: float) -> np.ndarray:
    """The top `fraction` of the KEPT features by `log M`, within one seed.

    The quantile is taken over the kept features only. Taking it over all
    features and then intersecting would make the retained count depend on how
    the floor and the tail overlap, which is a different set with the same
    name.
    """
    v = logM_row[keep]
    if v.size == 0:
        return np.zeros_like(keep, dtype=bool)
    cut = np.quantile(v, 1.0 - fraction)
    return keep & (logM_row >= cut)


def bin_by_quantile(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-count bins by quantile of `values`, returned as indices.

    Ties can leave a bin empty; that is reported by the occupancy rather than
    silently rebalanced, because a rebalanced bin is no longer a percentile bin.
    """
    if values.size == 0:
        return np.zeros(0, dtype=np.int64)
    cuts = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.clip(np.digitize(values, cuts, right=False), 0, n_bins - 1)


def bin_table(logM: np.ndarray, r_f: np.ndarray, n_bins: int) -> list[dict]:
    """One row per percentile bin: mean `r_f`, and its EMPIRICAL interval.

    A single-feature bin has no between-feature dispersion, so its SE is
    reported as None and its interval as None -- never as 0, which would read
    as "measured, and exact".
    """
    idx = bin_by_quantile(logM, n_bins)
    rows = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            rows.append({"bin": b, "n_features": 0, "status": "EMPTY"})
            continue
        v = r_f[m]
        p = float(v.mean())
        sd = float(v.std(ddof=1)) if n > 1 else None
        se = (sd / np.sqrt(n)) if sd is not None else None
        naive = float(np.sqrt(p * (1 - p) / n)) if 0 < n else None
        rows.append({
            "bin": b, "n_features": n, "status": "COMPUTED",
            "mean_logM": float(logM[m].mean()),
            "logM_min": float(logM[m].min()),
            "logM_max": float(logM[m].max()),
            "mean_r_f": p,
            "sd_r_f_between_features": sd,
            "se_between_features": (float(se) if se is not None else None),
            "ci95_lo": (float(p - Z95 * se) if se is not None else None),
            "ci95_hi": (float(p + Z95 * se) if se is not None else None),
            # Reported for COMPARISON only, never the interval of record.
            "naive_binomial_se_for_comparison": naive,
            "ratio_empirical_over_naive":
                (float(se / naive) if (se is not None and naive) else None),
        })
    return rows


def monotone_verdict(rows: list[dict]) -> dict:
    """AZ-P3's criterion, applied exactly as pre-declared.

    "Monotone non-decreasing, allowing for interval overlap": every adjacent
    pair either INCREASES, or its two 95% intervals OVERLAP. Three outcomes,
    because a two-valued verdict on an ordered comparison is not total
    (gotcha 24) -- and the third is reported when there are fewer than two
    usable bins, in which case the question was never asked rather than
    answered in the negative.
    """
    use = [r for r in rows if r.get("status") == "COMPUTED"]
    if len(use) < 2:
        return {"verdict": "UNDEFINED_TOO_FEW_BINS", "n_usable_bins": len(use)}
    n_dec, n_rescued, n_viol = 0, 0, 0
    detail = []
    for a, b in zip(use, use[1:]):
        up = b["mean_r_f"] >= a["mean_r_f"]
        lo_a, hi_a = a.get("ci95_lo"), a.get("ci95_hi")
        lo_b, hi_b = b.get("ci95_lo"), b.get("ci95_hi")
        overlap = (None if None in (lo_a, hi_a, lo_b, hi_b)
                   else (lo_a <= hi_b and lo_b <= hi_a))
        ok = bool(up or overlap)
        detail.append({"from_bin": a["bin"], "to_bin": b["bin"],
                       "delta_r_f": b["mean_r_f"] - a["mean_r_f"],
                       "increases": bool(up), "intervals_overlap": overlap,
                       "satisfies_criterion": ok})
        if up:
            n_dec += 1
        elif ok:
            n_rescued += 1
        else:
            n_viol += 1
    return {
        "verdict": ("MONOTONE_NON_DECREASING" if n_viol == 0
                    else "NOT_MONOTONE"),
        "n_usable_bins": len(use),
        "n_adjacent_pairs": len(detail),
        "n_increasing": n_dec,
        "n_decreasing_but_intervals_overlap": n_rescued,
        "n_violations": n_viol,
        "total_span_pp": (use[-1]["mean_r_f"] - use[0]["mean_r_f"]) * 100.0,
        "adjacent_pairs": detail,
    }


def per_seed_span(rows: list[dict]) -> float | None:
    """Last usable bin minus first, WITHIN one seed, in `r_f` units.

    Returned per seed rather than differenced from two combined bin means: the
    same 20 seeds contribute to both ends, so a paired span has the between-seed
    component of its noise cancelled, and differencing two independent-looking
    intervals would overstate it.
    """
    use = [r for r in rows if r.get("status") == "COMPUTED"]
    if len(use) < 2:
        return None
    return float(use[-1]["mean_r_f"] - use[0]["mean_r_f"])


def span_verdict(per_seed: list[dict]) -> dict:
    """BA11 -- **§AZ13.1's REPLACEMENT CRITERION**, applied.

    =======================================================================
    WHY THE INCUMBENT CRITERION IS BEING REPLACED, AND WHAT IT GOT WRONG
    =======================================================================
    AZ-P3's rule was "every adjacent pair either increases or has overlapping
    95% intervals". It is TOTAL over its statistic (gotcha 24) and its READING
    is incomplete (gotcha 30): **a sequence that ends BELOW where it started
    satisfies it** whenever each individual step is inside the noise. Stratum 1
    scored `MONOTONE_NON_DECREASING` with a span of **-6.09 pp** and all twenty
    of its seeds individually `NOT_MONOTONE`.

    **The replacement, stated at AZ13.1 and implemented here: score the
    DIRECTION on the span and its interval, and use the adjacent-pair test ONLY
    to say whether the rise is monotone. One statistic per question.**

    So this returns a DIRECTION verdict on the paired per-seed spans, and it is
    reported BESIDE `monotone_verdict`, which is retained unchanged so the old
    and new answers can be compared row by row. Neither replaces the other:
    "does it rise" and "does it rise monotonically" are different questions and
    the incumbent conflated them.

    Total over the outcome space, including the case AZ13.1 could not name: a
    span whose interval straddles zero is `INDISTINGUISHABLE_FROM_FLAT`, which
    is neither a rise nor a fall and must not be reported as either.
    """
    spans = [s for s in (per_seed_span(x["bins"]) for x in per_seed)
             if s is not None]
    if not spans:
        return {"verdict": "UNDEFINED_NO_SEED_HAS_TWO_USABLE_BINS",
                "n_seeds": 0}
    v = np.asarray(spans, dtype=np.float64)
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if v.size > 1 else None
    se = (sd / np.sqrt(v.size)) if sd is not None else None
    lo = (mean - Z95 * se) if se is not None else None
    hi = (mean + Z95 * se) if se is not None else None
    if lo is None:
        verdict = "UNDEFINED_SINGLE_SEED_NO_INTERVAL"
    elif lo > 0:
        verdict = "RISES"
    elif hi < 0:
        verdict = "FALLS"
    else:
        verdict = "INDISTINGUISHABLE_FROM_FLAT"
    return {
        "verdict": verdict,
        "criterion": "AZ13.1's replacement: the DIRECTION is scored on the "
                     "paired per-seed span and its 95% across-seed interval; "
                     "the adjacent-pair test says only whether the rise is "
                     "monotone",
        "n_seeds": int(v.size),
        "mean_span_pp": mean * 100.0,
        "sd_span_across_seeds_pp": (None if sd is None else sd * 100.0),
        "se_span_pp": (None if se is None else se * 100.0),
        "ci95_span_pp": [None if lo is None else lo * 100.0,
                         None if hi is None else hi * 100.0],
        "n_seeds_with_positive_span": int((v > 0).sum()),
        "n_seeds_with_negative_span": int((v < 0).sum()),
        "per_seed_span_pp": [float(x * 100.0) for x in v],
    }


def restated_verdict(monotone: dict, span: dict) -> dict:
    """The OLD verdict and the NEW one side by side, with the disagreement
    DERIVED rather than asserted (gotcha 20).

    `changes_the_reading` is True exactly when the incumbent's monotonicity
    label points in a direction the span does not support -- which is the
    stratum-1 case AZ13.1 found by running the rule, and the reason this
    function exists.
    """
    old = monotone.get("verdict")
    new = span.get("verdict")
    misleading = bool(old == "MONOTONE_NON_DECREASING" and new != "RISES")
    return {
        "old_verdict_adjacent_pair_criterion": old,
        "old_total_span_pp": monotone.get("total_span_pp"),
        "new_verdict_span_direction": new,
        "new_mean_span_pp": span.get("mean_span_pp"),
        "new_ci95_span_pp": span.get("ci95_span_pp"),
        "old_criterion_reads_as_a_rise_but_the_span_does_not": misleading,
        "note": "the two answer DIFFERENT questions and both are retained. "
                "The flag fires when the incumbent's label would be read as a "
                "rise that the span does not support -- AZ13.1's defect.",
    }


def combine_across_seeds(per_seed: list[dict], n_bins: int) -> dict:
    """Per-seed bin means combined at the SUMMARY level, never by pooling
    feature-indexed values across seeds (gotcha 18).

    A bin that is missing in some seeds is summarised over the seeds that have
    it, with the count reported -- an absent seed is absent, not zero.
    """
    out = []
    for b in range(n_bins):
        vals = [r["mean_r_f"] for s in per_seed for r in s["bins"]
                if r["bin"] == b and r.get("status") == "COMPUTED"]
        if not vals:
            out.append({"bin": b, "n_seeds": 0, "status": "EMPTY_IN_ALL_SEEDS"})
            continue
        v = np.asarray(vals, dtype=np.float64)
        sd = float(v.std(ddof=1)) if v.size > 1 else None
        out.append({
            "bin": b, "n_seeds": int(v.size), "status": "COMPUTED",
            "mean_of_per_seed_means": float(v.mean()),
            "sd_across_seeds": sd,
            "se_across_seeds": (float(sd / np.sqrt(v.size))
                                if sd is not None else None),
            "min": float(v.min()), "max": float(v.max()),
        })
    verdicts: dict[str, int] = {}
    for s in per_seed:
        k = s["monotone"]["verdict"]
        verdicts[k] = verdicts.get(k, 0) + 1
    # The same criterion applied to the COMBINED sequence, with intervals from
    # the ACROSS-SEED spread. Derived here rather than read off the printed
    # numbers by hand: a verdict a human eyeballs from a row of means is
    # exactly the sentence gotcha 20 forbids.
    combined_rows = [
        {"bin": r["bin"],
         "status": "COMPUTED" if r.get("status") == "COMPUTED" else "EMPTY",
         "n_features": r.get("n_seeds", 0),
         "mean_r_f": r.get("mean_of_per_seed_means"),
         "ci95_lo": (None if r.get("se_across_seeds") is None
                     else r["mean_of_per_seed_means"]
                     - Z95 * r["se_across_seeds"]),
         "ci95_hi": (None if r.get("se_across_seeds") is None
                     else r["mean_of_per_seed_means"]
                     + Z95 * r["se_across_seeds"])}
        for r in out]
    mono = monotone_verdict(combined_rows)
    span = span_verdict(per_seed)
    return {"bins": out, "per_seed_verdict_counts": verdicts,
            "monotone_on_the_combined_sequence": mono,
            "combined_interval_basis": "across-seed dispersion of the per-seed "
                                       "bin means, sd/sqrt(n_seeds)",
            # BA11: AZ13.1's replacement, reported BESIDE the incumbent.
            "span_direction_BA11": span,
            "restated_BA11": restated_verdict(mono, span)}


def panel(logM: np.ndarray, r_f: np.ndarray, keep: np.ndarray,
          fraction: float, n_bins: int, label: str) -> dict:
    """One arm: per-seed fine binning within the top `fraction` of `log M`."""
    per_seed = []
    for s in range(logM.shape[0]):
        m = top_fraction_mask(logM[s], keep[s], fraction)
        rows = bin_table(logM[s][m], r_f[s][m], n_bins)
        per_seed.append({"seed_row": int(s), "n_features": int(m.sum()),
                         "bins": rows, "monotone": monotone_verdict(rows)})
    return {
        "arm": label,
        "top_fraction_of_logM": float(fraction),
        "n_bins": int(n_bins),
        "interval_basis": "empirical between-feature dispersion "
                          "sd(r_f)/sqrt(n_bin); NOT Wilson; no multiplier "
                          "imported from another reduction (AZ6.1)",
        "per_seed": per_seed,
        "combined": combine_across_seeds(per_seed, n_bins),
    }


def _attrs(hf) -> dict:
    out = {}
    for k, v in hf.attrs.items():
        if isinstance(v, bytes):
            v = v.decode()
        elif isinstance(v, np.generic):
            v = v.item()
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        out[k] = v
    return out


def seed_permutation(src_labels: list[int], dst_labels: list[int],
                     what: str) -> np.ndarray:
    """Rows of `src` into `dst`'s order, BY LABEL. Refuses on a set mismatch."""
    if sorted(src_labels) != sorted(dst_labels):
        raise SystemExit(
            f"FATAL: {what} names seeds {sorted(src_labels)} against "
            f"{sorted(dst_labels)}. Refusing to align two different seed sets.")
    where = {v: i for i, v in enumerate(src_labels)}
    return np.array([where[v] for v in dst_labels], dtype=np.int64)


def _combined_fingerprint(arm: dict) -> list:
    """Every combined bin mean and the OLD verdict, as exact floats.

    The quantities BA11's acceptance criterion compares. `span_direction_BA11`
    is deliberately EXCLUDED: it is the addition, and comparing it against a
    reference that predates it would report a difference for the one thing that
    is supposed to be new.
    """
    rows = [(r["bin"], r.get("status"), r.get("mean_of_per_seed_means"))
            for r in arm["combined"]["bins"]]
    mono = arm["combined"]["monotone_on_the_combined_sequence"]
    return [rows, mono.get("verdict"), mono.get("total_span_pp"),
            mono.get("n_violations"),
            arm["combined"]["per_seed_verdict_counts"]]


def reproduces_reference(arms: dict, by_stratum: list, reference: Path) -> dict:
    """ACCEPTANCE CRITERION (BA7.4): every bin mean and every OLD verdict must
    reproduce the reference artifact EXACTLY.

    Compared as exact values, not to a tolerance: this run and the reference are
    the same arithmetic on the same inputs, so any difference at all is a
    changed computation. **If the old verdicts move, the comparison between old
    and new is uninterpretable** -- something other than the criterion changed.

    An ABSENT reference arm is reported ABSENT, never as agreement (gotcha 15).
    """
    doc = json.loads(reference.read_text())
    ref_arms = doc.get("arms", {})
    rows, problems = [], []
    for name, arm in arms.items():
        if name not in ref_arms:
            rows.append({"arm": name, "same": False, "detail": "<ABSENT>"})
            problems.append(f"arm {name}: ABSENT from the reference")
            continue
        a, b = _combined_fingerprint(arm), _combined_fingerprint(ref_arms[name])
        same = a == b
        rows.append({"arm": name, "same": bool(same)})
        if not same:
            problems.append(f"arm {name}: bin means or the OLD verdict moved")

    ref_strata = {r.get("stratum"): r for r in doc.get("by_firing_stratum", [])}
    for r in by_stratum:
        s = r.get("stratum")
        if s not in ref_strata:
            rows.append({"stratum": s, "same": False, "detail": "<ABSENT>"})
            problems.append(f"stratum {s}: ABSENT from the reference")
            continue
        ref = ref_strata[s]
        if r.get("status") != ref.get("status"):
            problems.append(f"stratum {s}: status {r.get('status')} vs "
                            f"{ref.get('status')}")
            rows.append({"stratum": s, "same": False})
            continue
        if r.get("status") != "COMPUTED":
            rows.append({"stratum": s, "same": True, "detail": r.get("status")})
            continue
        same = _combined_fingerprint(r) == _combined_fingerprint(ref)
        rows.append({"stratum": s, "same": bool(same)})
        if not same:
            problems.append(f"stratum {s}: bin means or the OLD verdict moved")
    return {
        "reference": str(reference),
        "reference_identity": doc.get("identity"),
        "n_compared": len(rows),
        "comparisons": rows,
        "problems": problems,
        "verdict": ("REPRODUCES_REFERENCE_EXACTLY" if not problems
                    else "DIFFERS"),
        "scope": "combined bin means, the OLD adjacent-pair verdict, its span "
                 "and violation count, and the per-seed verdict counts. The "
                 "NEW span verdict is excluded -- it is the addition.",
    }


def run(gtilde_path: Path, rf_counts_path: Path, ap_path: Path,
        out_path: Path, fraction: float = 0.10, n_bins: int = 10,
        reference: Path | None = None) -> dict:
    with h5py.File(gtilde_path, "r") as hf:
        M = hf["M"][:]
        gt_seeds = [int(x.decode() if isinstance(x, bytes) else x)
                    for x in hf["seeds"][:]]
        fbin_stored_raw = (hf["feature_firing_bin"][:]
                           if "feature_firing_bin" in hf else None)
        gt_identity = _attrs(hf)
    n_rows = int(gt_identity["n_rows"])

    with h5py.File(rf_counts_path, "r") as hf:
        match_count = hf["match_count"][:]
        rf_identity = _attrs(hf)
    # `rf_counts` has NO `seeds` dataset; its order is the `seed_labels` attr.
    raw = rf_identity.get("seed_labels")
    if raw is None:
        raise SystemExit(
            f"FATAL: {rf_counts_path} carries no `seed_labels` attr and no "
            "`seeds` dataset, so its seed order cannot be read from the file. "
            "Refusing to assume it -- absent is not matching (gotcha 15).")
    rf_seeds = [int(x) for x in (json.loads(raw) if isinstance(raw, str)
                                 else np.asarray(raw).ravel().tolist())]
    n_compare = int(rf_identity["n_compare_seeds"])

    # gtilde's rows are permuted into rf_counts' NUMERIC order, so every array
    # below is in one stated order rather than in whichever arrived first.
    perm = seed_permutation(gt_seeds, rf_seeds, str(gtilde_path))
    M = M[perm]

    bad = ~np.isfinite(M) | (M <= 0)
    if bad.any():
        raise SystemExit(
            f"FATAL: {int(bad.sum())} of {M.size} cells have non-finite or "
            "non-positive M. SPECIFICATION section 4 records that set as EMPTY "
            "BY MEASUREMENT; flooring it would make the check unable to fire.")
    logM = np.log(M)
    r_f = match_count.astype(np.float64) / float(n_compare)

    fbin_stored = None if fbin_stored_raw is None else fbin_stored_raw[perm]

    # The firing rate comes from the artifact `log M` was BUILT on, named
    # explicitly and then checked against the predictor's own `ap_source` attr
    # -- an existence check on a path is not an identity check (gotcha 15).
    declared = gt_identity.get("ap_source")
    ap_source_agrees = (None if declared is None
                        else str(declared) == str(ap_path))
    with h5py.File(ap_path, "r") as hf:
        fire_counts = hf["fire_counts"][:]
        ap_seeds = [int(str(x.decode() if isinstance(x, bytes) else x
                            ).split("_")[-1]) for x in hf["seeds"][:]]
        ap_n_rows = int(_attrs(hf)["n_rows"])
    if ap_n_rows != n_rows:
        raise SystemExit(
            f"FATAL: the predictor was built on n_rows={n_rows} and the firing "
            f"counts come from an artifact with n_rows={ap_n_rows}. Those are "
            "different populations and the firing rates would not be the ones "
            "the strata were defined on.")
    fire_rate = (fire_counts.astype(np.float64) / float(ap_n_rows)
                 )[seed_permutation(ap_seeds, rf_seeds, str(ap_path))]

    fbin = firing_bin(fire_rate)
    # A POSITIONAL agreement check, not a total: the smooth artifact stores the
    # firing bin it used, and equality at every one of the 204,800 cells ties
    # the seed axis and the feature axis at once. Reported as None when the
    # dataset is absent -- absent is not agreement.
    agree = (None if fbin_stored is None
             else float((fbin == fbin_stored).mean()))

    thresh = ANALYSIS_MIN_FIRING_ROWS / float(n_rows)
    keep = fire_rate >= thresh
    floor_report = {
        "analysis_min_firing_rows": ANALYSIS_MIN_FIRING_ROWS,
        "firing_rate_threshold": thresh,
        "n_excluded_total": int((~keep).sum()),
        "n_excluded_per_seed": [int(x) for x in (~keep).sum(axis=1)],
        "per_firing_bin_excluded": [
            int(((~keep) & (fbin == b)).sum())
            for b in range(len(BIN_EDGES) - 1)],
        "note": "A4a's floor, inherited by AL8c. Its measured effect on this "
                "ensemble is ZERO exclusions (AO16); the per-bin table is "
                "reported anyway because A4a mandates it.",
    }

    arms = {
        "top_decile": panel(logM, r_f, keep, fraction, n_bins, "top_decile"),
        "full_range_for_context": panel(logM, r_f, keep, 1.0, n_bins,
                                        "full_range"),
    }

    # Per firing-rate stratum, inside the top fraction. Firing rate is the
    # dominant driver, so a pooled shape could be a composition effect.
    by_stratum = []
    n_fbins = len(BIN_EDGES) - 1
    for b in range(n_fbins):
        sel = keep & (fbin == b)
        if sel.sum() < 10 * n_bins:
            by_stratum.append({"stratum": b, "n_features": int(sel.sum()),
                               "status": "TOO_FEW_FEATURES"})
            continue
        by_stratum.append({
            "stratum": b,
            "firing_range": [BIN_EDGES[b], BIN_EDGES[b + 1]],
            "n_features": int(sel.sum()),
            "status": "COMPUTED",
            **panel(logM, r_f, sel, fraction, n_bins, f"stratum_{b}"),
        })

    acceptance = (None if reference is None
                  else reproduces_reference(arms, by_stratum, reference))
    if acceptance is not None and acceptance["verdict"] != \
            "REPRODUCES_REFERENCE_EXACTLY":
        print("ACCEPTANCE (reference) FAILED",
              json.dumps(acceptance, indent=2), flush=True)
        raise SystemExit(1)
    if acceptance is not None:
        print(f"[panel] acceptance: {acceptance['verdict']}", flush=True)

    from src.sae.train import git_dirty, git_sha

    out = {
        "specification": "PREREGISTRATION amendment AZ section 12 "
                         "(brief section 3c); RESTATED under BA11 "
                         "(amendment BA section 7.4)",
        "acceptance_criterion_BA11": acceptance,
        "exploratory": "AZ7 -- cannot become primary under any outcome; no "
                       "mechanism sentence is written for it",
        "identity": {"git_sha": git_sha(), "git_dirty": git_dirty(),
                     "measurement": "alignment_fine_panel_AZ"},
        "seed_axis": {
            "order_of_every_array_below": "NUMERIC, from rf_counts' "
                                          "`seed_labels` attr",
            "rf_seed_labels": rf_seeds,
            "gtilde_seed_labels": gt_seeds,
            "permutation_gtilde_to_rf": perm.tolist(),
            "firing_source_seed_labels": ap_seeds,
        },
        "n_compare_seeds": n_compare,
        "firing_strata": {"source": "corrector_scan.BIN_EDGES",
                          "edges": list(BIN_EDGES),
                          "agreement_with_stored_feature_firing_bin": agree,
                          "stored_bin_dataset_present":
                              bool(fbin_stored is not None)},
        "predictor_source_check": {
            "ap_path_supplied": str(ap_path),
            "ap_source_recorded_in_the_predictor": declared,
            "agrees": ap_source_agrees,
            "note": "reported, not enforced: the predictor may legitimately "
                    "have been built from a copy at another path. An ABSENT "
                    "attr is reported as None, never as agreement.",
        },
        "analysis_floor": floor_report,
        "arms": arms,
        "by_firing_stratum": by_stratum,
        "prediction_AZ_P3": {
            "statement": "mean reproducibility is monotone non-decreasing "
                         "across the percentile bins within the top log M "
                         "decile, allowing for interval overlap",
            "standing": "FILED, NOT SEALED -- NOT TALLIED (AZ3.3)",
            "per_seed_verdict_counts":
                arms["top_decile"]["combined"]["per_seed_verdict_counts"],
            "combined_sequence_verdict":
                arms["top_decile"]["combined"][
                    "monotone_on_the_combined_sequence"]["verdict"],
        },
        "source": {"gtilde": str(gtilde_path), "rf_counts": str(rf_counts_path),
                   "firing_counts": str(ap_path),
                   "gtilde_identity": gt_identity, "rf_identity": rf_identity},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    return out


def _pp(x, width: int = 0) -> str:
    """Format a percentage-point value that may legitimately be absent.

    `span_verdict` returns `UNDEFINED_NO_SEED_HAS_TWO_USABLE_BINS` -- with no
    `mean_span_pp` at all -- for a stratum whose top-decile slice leaves fewer
    than two usable bins in every seed. That is a real outcome, not an error,
    and a formatter that cannot render it turned a completed measurement into a
    FAILED job on the first run of this section. The artifact had already been
    written; only the summary print died.
    """
    return f"{x:+.2f}".rjust(width) if isinstance(x, (int, float)) \
        else "n/a".rjust(width)


def _ci(pair) -> str:
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return "[n/a, n/a]"
    return f"[{_pp(pair[0])}, {_pp(pair[1])}]"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gtilde", type=Path, required=True)
    p.add_argument("--rf-counts", type=Path, required=True)
    p.add_argument("--ap", type=Path, required=True,
                   help="the AP artifact the predictor was built on -- the "
                        "source of `fire_counts`, checked against the "
                        "predictor's own `ap_source` attr")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fraction", type=float, default=0.10)
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--reference", type=Path, default=None,
                   help="alignment_fine_panel_az_v2.json -- BA7.4's acceptance "
                        "criterion. Every bin mean and every OLD verdict must "
                        "reproduce it EXACTLY or the job exits non-zero.")
    args = p.parse_args()

    out = run(args.gtilde, args.rf_counts, args.ap, args.out, args.fraction,
              args.n_bins, args.reference)

    for name in ("top_decile", "full_range_for_context"):
        arm = out["arms"][name]
        rows = arm["combined"]["bins"]
        vals = [f"{r['mean_of_per_seed_means']:.4f}"
                if r.get("status") == "COMPUTED" else "--" for r in rows]
        print(f"\n[3c] {name}: mean r_f by bin  " + "  ".join(vals))
        rs = arm["combined"]["restated_BA11"]
        print(f"      OLD {rs['old_verdict_adjacent_pair_criterion']} "
              f"(span {_pp(rs['old_total_span_pp'])} pp)   "
              f"NEW {rs['new_verdict_span_direction']} "
              f"(span {_pp(rs['new_mean_span_pp'])} pp, CI "
              f"{_ci(rs['new_ci95_span_pp'])})"
              + ("   <-- OLD READS AS A RISE, SPAN DOES NOT"
                 if rs["old_criterion_reads_as_a_rise_but_the_span_does_not"]
                 else ""))

    print("\n[BA11] per firing stratum, inside the top fraction:")
    for r in out["by_firing_stratum"]:
        if r.get("status") != "COMPUTED":
            print(f"      stratum {r['stratum']}: {r.get('status')} "
                  f"(n={r.get('n_features')})")
            continue
        rs = r["combined"]["restated_BA11"]
        print(f"      stratum {r['stratum']} n={r['n_features']:>7,}  "
              f"OLD {str(rs['old_verdict_adjacent_pair_criterion']):<24} "
              f"NEW {str(rs['new_verdict_span_direction']):<38} "
              f"span {_pp(rs['new_mean_span_pp'], 7)} pp "
              f"{_ci(rs['new_ci95_span_pp'])}"
              + ("  <-- RESTATED"
                 if rs["old_criterion_reads_as_a_rise_but_the_span_does_not"]
                 else ""))

    print(f"\n  AZ-P3 (not tallied): "
          f"{out['prediction_AZ_P3']['per_seed_verdict_counts']}; combined "
          f"{out['prediction_AZ_P3']['combined_sequence_verdict']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
