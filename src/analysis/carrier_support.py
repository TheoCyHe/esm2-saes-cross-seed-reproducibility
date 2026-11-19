"""AZ sections 11 -- CARRIER SUPPORT. Is the carrier excess ALIGNMENT, or is it
HOMOLOGY SUPPORT wearing alignment's clothes? And where do carriers actually sit
in the predictor the regression uses?

=============================================================================
WHY THE SINGLY-MATCHED COMPARISON IS NOT ENOUGH (AZ11, brief section 3a)
=============================================================================
AY7 measured carriers' "has ANY MNN partner" rate -- rate **A**, not rate B's
carrier-MATCH rate -- at **0.5617** against a FIRING-RATE-matched ambient of
**0.4895**: an excess of 7.22 percentage points (PREREGISTRATION §AZ11's table
labels the row `A`, and `excess_A_minus_matched_pp` below is A minus the
matched ambient; this sentence said "matching", corrected 2026-09-16). That
baseline
controls firing rate and NOTHING ELSE. But homology support `d_fs` is worth
**+3.1217 pp of `r_f` per SD** (AT13) against `log M`'s beta-binomial bound of
**0.3635 pp** (AT11) -- **8.59x the entire bounded alignment effect** -- and a
well-annotated concept plausibly appears across many sequence clusters. So the
7.22 pp excess could be mostly homology support.

This module re-draws the baseline from the SAME (firing-rate x homology-support)
cell rather than the same firing-rate stratum alone, and reports both excesses
and their ratio side by side. **If the excess survives double matching it is
alignment; if it collapses it is homology support** -- and the latter would be
the THIRD time the interesting variance has landed in a density variable.

**THE SUPPORT BINS ARE QUANTILES WITHIN (SEED, FIRING BIN), NOT GLOBAL
QUANTILES, AND THAT IS THE WHOLE DESIGN.** `d_fs` rises with firing rate almost
by construction -- a feature that fires more touches more clusters -- so global
support bins nested inside a firing stratum would be nearly constant and the
"double" match would silently be the single one. Splitting WITHIN the stratum is
what makes the second axis carry information, and the realised cell occupancies
and `d_fs` ranges are reported so a reader can see that it does.

=============================================================================
WHERE DO CARRIERS SIT IN `log M`? (AZ11, brief section 3b)
=============================================================================
The regression's bound is per STANDARD DEVIATION of alignment. Without knowing
how far up the alignment distribution carriers sit, the concept-level result and
the feature-level bound cannot be compared at all. So: the percentile position
of every carrier in `log M` -- **the regression's own predictor**, not the
average precision that selected it.

**THE TWO STATISTICS ARE DIFFERENT REDUCTIONS OF THE SAME ARRAY AND THAT IS
REPORTED, NOT ASSUMED AWAY.** A carrier is `argmax_f AP(f, c)` for ONE concept;
`M_f` is `max_c ( gain(f, c) / mu_s(cell(c), fbin(f)) )` over the 703-concept
pool. A carrier for a rare concept can therefore have an unremarkable `log M`.
The rank correlation between the per-feature reductions of the two -- `max_c
AP(f, c)` against `M_f`, within a seed -- is reported beside the positions,
because it is what says how far apart the two instruments are.

=============================================================================
THE ACCEPTANCE CRITERION, AND WHY IT IS THE LOAD-BEARING PART
=============================================================================
Two, both of which can fail:

  1. **The per-anchor MNN counts must be BIT-IDENTICAL to `match_count` in
     `rf_counts_theta070_n19.h5`**, in NUMERIC seed order -- `carrier_identity`'s
     criterion, reused rather than re-implemented.
  2. **The SINGLY-matched arm must reproduce `carrier_identity_ay_v3.json`'s
     stored rates EXACTLY** -- B, A, B|A, the ambient and the firing-rate-matched
     ambient. This module computes the single and the double arm through ONE
     function differing only in its cell index, so if the single arm reproduces
     AY7 to the last bit then the new stratification is the ONLY thing that
     differs in the double arm. Without it, a difference between the two arms
     could be the stratification or could be any accidental change.

**THIS JOB CROSSES BOTH SEED-ORDER SPLITS** (gotcha 29). AP and `gtilde` are
LEXICOGRAPHIC (`seed_0, seed_1, seed_10, ...` and `0, 1, 10, ...`); `d_fs` and
`rf_counts` are NUMERIC. Positions 0 and 1 are the only ones that coincide, so a
positional join mis-pairs 18 of 20 seeds and every identity key the standard
gates compare is invariant under the permutation. Every axis is permuted by
LABEL and the permutation is asserted elementwise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from src.analysis.carrier_identity import (
    THETA_REFERENCE,
    carriers_from_ap,
    firing_bin,
    mnn_partner_matrix,
    order_decoders_to,
    seed_label_to_int,
    verify_counts_bit_identical,
)
from src.analysis.corrector_scan import BIN_EDGES
from src.analysis.concept_repro import (
    REFERENCE_PERCENTILE,
    assert_labels_align,
    detection_matrix,
    detection_rate,
    null_thresholds,
    observed_best_ap,
    spearman,
    to_gain,
)
from src.analysis.rf_marginal import load_decoder

N_SUPPORT_BINS = 3


# ------------------------------------------------------------ the cell index

def support_bin_within(fbin: np.ndarray, d_fs: np.ndarray,
                       n_support: int) -> np.ndarray:
    """Quantile bins of `d_fs` computed WITHIN each (seed, firing bin).

    Returns `(n_seeds, n_features)` of support-bin indices in `[0, n_support)`.

    Global quantiles would not do: `d_fs` rises with firing rate, so a global
    bin nested inside a firing stratum is close to constant and the second
    matching axis would do nothing while appearing to. Quantiles are taken over
    the features actually in the stratum, so every stratum is split.

    A stratum with fewer than `n_support` distinct `d_fs` values cannot be split
    into that many bins; `np.digitize` on its (collapsed) quantile edges then
    yields fewer occupied bins, which is correct behaviour and is visible in the
    reported occupancies rather than raising.
    """
    fbin = np.asarray(fbin)
    d_fs = np.asarray(d_fs, dtype=np.float64)
    if fbin.shape != d_fs.shape:
        raise ValueError(f"shape mismatch: fbin {fbin.shape}, d_fs {d_fs.shape}")
    out = np.zeros(fbin.shape, dtype=np.int64)
    qs = np.linspace(0, 100, n_support + 1)[1:-1]
    for s in range(fbin.shape[0]):
        for b in np.unique(fbin[s]):
            m = fbin[s] == b
            v = d_fs[s][m]
            edges = np.percentile(v, qs) if v.size else np.array([])
            out[s][m] = np.digitize(v, edges, right=False)
    return np.clip(out, 0, n_support - 1)


def cell_index(fbin: np.ndarray, sbin: np.ndarray | None,
               n_support: int) -> np.ndarray:
    """One integer per (seed, feature) naming its matching cell.

    `sbin is None` gives the SINGLE (firing-rate only) arm, which is exactly
    AY7's cell. That is deliberate: both arms go through one function, so the
    only difference between them is this index.
    """
    if sbin is None:
        return np.asarray(fbin, dtype=np.int64)
    return np.asarray(fbin, dtype=np.int64) * n_support + np.asarray(
        sbin, dtype=np.int64)


def cell_occupancy(cell: np.ndarray, d_fs: np.ndarray, n_cells: int) -> list:
    """Per-cell counts and `d_fs` ranges, for the seed-0 row.

    Reported because the double arm's whole claim is that its cells are
    non-degenerate; an empty or singleton cell contributes no baseline and is
    counted as unusable rather than silently skipped.
    """
    rows = []
    for c in range(n_cells):
        m = cell[0] == c
        n = int(m.sum())
        rows.append({
            "cell": int(c), "n_features_seed0": n,
            "d_fs_min": float(d_fs[0][m].min()) if n else None,
            "d_fs_max": float(d_fs[0][m].max()) if n else None,
        })
    return rows


# ---------------------------------------------------------------- the rates

def carrier_rates_by_cell(carrier: np.ndarray, detected: np.ndarray,
                          partner: np.ndarray, cell: np.ndarray,
                          n_cells: int) -> dict:
    """AY7's `carrier_rates`, generalised from a firing bin to ANY cell index.

    Identical in every other respect, deliberately: the ordered-pair iteration,
    the `s -> t` direction, the exclusion of the carrier from its own cell, and
    the `n_in > 1` usability guard are AY7's, because the singly-matched arm has
    to reproduce AY7's stored numbers exactly for the double arm's difference to
    mean anything.
    """
    n_seeds, n_concepts = carrier.shape
    has_partner = partner >= 0                                  # (s, t, i)

    cell_n = np.zeros((n_seeds, n_cells), dtype=np.int64)
    cell_hit = np.zeros((n_seeds, n_seeds, n_cells), dtype=np.int64)
    for s in range(n_seeds):
        cell_n[s] = np.bincount(cell[s], minlength=n_cells)
        for t in range(n_seeds):
            if s == t:
                continue
            cell_hit[s, t] = np.bincount(
                cell[s], weights=has_partner[s, t],
                minlength=n_cells).astype(np.int64)

    b_hits = a_hits = 0
    n_pairs_evaluated = 0
    matched_sum, matched_n = 0.0, 0
    unusable = 0
    for s in range(n_seeds):
        for t in range(s + 1, n_seeds):
            both = detected[s] & detected[t]
            cs, ct = carrier[s][both], carrier[t][both]
            if cs.size == 0:
                continue
            p = partner[s, t, cs]
            a = p >= 0
            b = p == ct
            a_hits += int(a.sum())
            b_hits += int(b.sum())
            n_pairs_evaluated += int(cs.size)

            cc = cell[s, cs]
            n_in = cell_n[s, cc]
            hit_in = cell_hit[s, t, cc]
            ok = n_in > 1
            unusable += int((~ok).sum())
            if ok.any():
                num = hit_in[ok] - a[ok].astype(np.int64)
                den = n_in[ok] - 1
                matched_sum += float((num / den).sum())
                matched_n += int(ok.sum())

    off = ~np.eye(n_seeds, dtype=bool)
    ambient = float(has_partner[off].mean())
    matched = matched_sum / matched_n if matched_n else float("nan")
    b_rate = b_hits / n_pairs_evaluated if n_pairs_evaluated else float("nan")
    a_rate = a_hits / n_pairs_evaluated if n_pairs_evaluated else float("nan")
    return {
        "n_cells": int(n_cells),
        "n_concept_seedpair_cells_evaluated": int(n_pairs_evaluated),
        "rate_B_carrier_match": b_rate,
        "rate_A_carrier_has_any_partner": a_rate,
        "conditional_B_given_A": (b_hits / a_hits) if a_hits else float("nan"),
        "ambient_match_rate_same_pair_set": ambient,
        "matched_ambient": matched,
        "n_carriers_without_usable_cell_baseline": int(unusable),
        "n_carrier_cells_with_a_usable_baseline": int(matched_n),
        # The LIKE-FOR-LIKE excess: A and the baseline are the same event
        # ("has ANY partner"). B against this baseline compares a strict subset
        # against a superset and is reported by AY7, not re-scored here.
        "excess_A_minus_matched_pp": (a_rate - matched) * 100.0,
        "ratio_A_over_matched": a_rate / matched if matched else float("nan"),
    }


def excess_comparison(single: dict, double: dict) -> dict:
    """AZ-P1, DERIVED from the two arms rather than asserted (gotcha 20).

    TOTAL over the ratio's whole domain, including a NEGATIVE double excess
    (which is "under half") and a zero single excess (for which the ratio is
    undefined and is reported as such, never as 0 or 1).
    """
    e1 = single["excess_A_minus_matched_pp"]
    e2 = double["excess_A_minus_matched_pp"]
    if not np.isfinite(e1) or e1 == 0.0:
        return {
            "single_excess_pp": e1, "double_excess_pp": e2,
            "ratio_double_over_single": None,
            "outcome": "UNDEFINED_SINGLE_EXCESS_IS_ZERO_OR_NON_FINITE",
            "note": "the predicate is a ratio against the single excess; with "
                    "that excess zero or non-finite the ratio does not exist "
                    "and no verdict is manufactured",
        }
    ratio = e2 / e1
    return {
        "single_excess_pp": e1,
        "double_excess_pp": e2,
        "ratio_double_over_single": float(ratio),
        "half_of_single_excess_pp": float(e1 / 2.0),
        "outcome": ("UNDER_HALF" if e2 < e1 / 2.0 else "HALF_OR_ABOVE"),
        "collapses_entirely": bool(e2 <= 0.0),
    }


# ------------------------------------------- where carriers sit in `log M`

def percentile_positions(carrier: np.ndarray, mask: np.ndarray,
                         value: np.ndarray) -> dict:
    """Percentile rank of each carrier's `value` within ITS OWN seed.

    Within-seed, because `log M` is z-scored within seed in the model of record
    (AL8a) and because a cross-seed pooled percentile would compare features
    across dictionaries, which a feature index does not support (gotcha 18).
    Percentile rank is invariant under the z-scoring, so this is the same
    position the model sees.

    Ranks are computed with the `<` convention plus half the ties, so a value
    at the population median sits at 50 rather than above or below it.
    """
    n_seeds, n_concepts = carrier.shape
    pos = []
    for s in range(n_seeds):
        v = np.asarray(value[s], dtype=np.float64)
        idx = carrier[s][mask[s]]
        if idx.size == 0:
            continue
        pos.append(percentile_rank_within(v, v[idx]))
    if not pos:
        return {"status": "NO_CARRIER_CELLS"}
    p = np.concatenate(pos)
    return {
        "status": "COMPUTED",
        "n_cells": int(p.size),
        "median": float(np.median(p)),
        "iqr": [float(np.percentile(p, 25)), float(np.percentile(p, 75))],
        "mean": float(p.mean()),
        "p05": float(np.percentile(p, 5)),
        "p95": float(np.percentile(p, 95)),
        "share_above_75th": float((p > 75.0).mean()),
        "share_above_90th": float((p > 90.0).mean()),
        "share_below_50th": float((p < 50.0).mean()),
        "histogram_10_bins": np.histogram(p, bins=10, range=(0, 100))[0].tolist(),
    }


def percentile_rank_within(pool: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Percentile rank of each `x` within `pool`, `<` plus half the ties.

    The same convention `percentile_positions` uses, factored out so the two
    cannot drift: a value at the pool's median sits at 50 rather than above or
    below it. An EMPTY pool returns an empty array -- the caller counts the cell
    as unusable rather than receiving a 50 that would read as "measured, and
    exactly average".
    """
    order = np.sort(np.asarray(pool, dtype=np.float64))
    x = np.asarray(x, dtype=np.float64)
    if order.size == 0:
        return np.zeros(0, dtype=np.float64)
    lo = np.searchsorted(order, x, side="left")
    hi = np.searchsorted(order, x, side="right")
    return 100.0 * (lo + hi) / 2.0 / order.size


def carrier_feature_indicator(carrier: np.ndarray, detected: np.ndarray,
                              n_features: int) -> np.ndarray:
    """(n_seeds, n_features) bool -- carrier of at least one DETECTED concept.

    The PEER set is its complement within a firing bin, so a feature that
    carries some OTHER concept is not counted as an ordinary peer. Without this
    the comparison would put ~700 carriers into the very population they are
    being compared against, moving the baseline toward the carriers by
    construction.
    """
    out = np.zeros((carrier.shape[0], int(n_features)), dtype=bool)
    for s in range(carrier.shape[0]):
        idx = carrier[s][detected[s]]
        if idx.size:
            out[s, idx] = True
    return out


def support_distribution(carrier: np.ndarray, detected: np.ndarray,
                         d_fs: np.ndarray, fbin: np.ndarray,
                         n_fbins: int, value_name: str = "d_fs") -> dict:
    """BA9 -- carriers' homology support against their FIRING-MATCHED peers.

    **`value_name` IS A LABEL, NOT A SWITCH** (amendment BB section 3). The
    arithmetic is identical for any per-(seed, feature) covariate, so BB's
    concentration and perplexity arms go through THIS function rather than a
    second implementation -- two implementations of one statistic is how they
    drift, and the whole point of BB section 3a is that the concentration
    result be comparable with BA10's homology result. What `value_name`
    changes is only the two key names that NAME the covariate, so that one key
    never carries two meanings (gotcha 16's committed convention).

    Called with the default it is BA10 exactly, which is why BB can assert
    bit-identical reproduction of `carrier_support_ba.json` as an acceptance
    criterion -- the check that this generalisation changed no arithmetic.

    THE MEASUREMENT §AZ11.3 SAID WOULD SEPARATE ITS TWO CAUSES, and did not run.
    AZ11 found the carrier excess intact under double matching and named two
    ways to produce that branch: the excess is alignment, or carriers were never
    support-enriched so the second axis had nothing to remove. Only this tells
    them apart.

    **THE POOLED COMPARISON IS ITSELF FIRING-MATCHED, AND THAT IS THE DESIGN.**
    `d_fs` rises with firing rate almost by construction and carriers skew
    high-firing, so a raw pooled shift would be a composition effect wearing an
    enrichment's clothes. The primary statistic is the carrier's PERCENTILE
    POSITION within its own (seed, firing-bin) NON-CARRIER `d_fs` distribution
    -- matched by construction, and the same reduction §AZ12 took on `log M`.
    **50 means not enriched.** The raw per-stratum medians and the
    composition-confounded pooled shift are both reported beside it, the latter
    labelled as what it is.

    One row per DETECTED (seed, concept) cell, matching §AZ12's population so
    the two are comparable. The per-(seed, feature) deduplicated arm is reported
    beside it, because a feature carrying five concepts otherwise counts five
    times.
    """
    n_seeds, n_features = d_fs.shape
    is_carrier = carrier_feature_indicator(carrier, detected, n_features)

    per_cell: list[np.ndarray] = []
    per_feat: list[np.ndarray] = []
    per_seed_median: list[float | None] = []
    unusable = 0
    strata: list[dict] = []

    by_stratum_vals: dict[int, list[np.ndarray]] = {b: [] for b in range(n_fbins)}
    carrier_med: dict[int, list[float]] = {b: [] for b in range(n_fbins)}
    peer_med: dict[int, list[float]] = {b: [] for b in range(n_fbins)}

    for s in range(n_seeds):
        idx_cell = carrier[s][detected[s]]
        seed_vals = []
        for b in range(n_fbins):
            in_bin = fbin[s] == b
            peers = in_bin & ~is_carrier[s]
            pool = d_fs[s][peers]
            sel = idx_cell[fbin[s][idx_cell] == b] if idx_cell.size else idx_cell
            if pool.size == 0:
                unusable += int(sel.size)
                continue
            if sel.size:
                p = percentile_rank_within(pool, d_fs[s][sel])
                seed_vals.append(p)
                by_stratum_vals[b].append(p)
                carrier_med[b].append(float(np.median(d_fs[s][sel])))
                peer_med[b].append(float(np.median(pool)))
        if seed_vals:
            v = np.concatenate(seed_vals)
            per_cell.append(v)
            per_seed_median.append(float(np.median(v)))
        else:
            per_seed_median.append(None)

        # the deduplicated arm: each carrier FEATURE once, in its own bin
        fvals = []
        for b in range(n_fbins):
            in_bin = fbin[s] == b
            peers = in_bin & ~is_carrier[s]
            pool = d_fs[s][peers]
            sel = np.flatnonzero(in_bin & is_carrier[s])
            if pool.size and sel.size:
                fvals.append(percentile_rank_within(pool, d_fs[s][sel]))
        if fvals:
            per_feat.append(np.concatenate(fvals))

    for b in range(n_fbins):
        if not by_stratum_vals[b]:
            strata.append({"stratum": b, "status": "NO_USABLE_CELLS",
                           "firing_range": [BIN_EDGES[b], BIN_EDGES[b + 1]]})
            continue
        v = np.concatenate(by_stratum_vals[b])
        cm = np.asarray(carrier_med[b], dtype=np.float64)
        pm = np.asarray(peer_med[b], dtype=np.float64)
        strata.append({
            "stratum": b,
            "firing_range": [BIN_EDGES[b], BIN_EDGES[b + 1]],
            "status": "COMPUTED",
            "n_carrier_cells": int(v.size),
            "n_seeds_contributing": int(cm.size),
            f"median_percentile_in_peer_{value_name}": float(np.median(v)),
            "iqr_percentile": [float(np.percentile(v, 25)),
                               float(np.percentile(v, 75))],
            # raw units, per seed then combined at the SUMMARY level (gotcha 18)
            f"carrier_median_{value_name}_mean_over_seeds": float(cm.mean()),
            f"peer_median_{value_name}_mean_over_seeds": float(pm.mean()),
            "shift_nats_mean_over_seeds": float((cm - pm).mean()),
            "shift_nats_sd_across_seeds": (float((cm - pm).std(ddof=1))
                                           if cm.size > 1 else None),
        })

    if not per_cell:
        return {"status": "NO_USABLE_CARRIER_CELLS",
                "n_carrier_cells_without_a_peer_pool": int(unusable)}

    p = np.concatenate(per_cell)
    pf = np.concatenate(per_feat) if per_feat else np.zeros(0)
    med_seed = np.asarray([m for m in per_seed_median if m is not None],
                          dtype=np.float64)

    # THE COMPOSITION-CONFOUNDED ARM, reported so the difference is visible
    # rather than argued: carriers against ALL non-carriers regardless of
    # firing bin. If this and the matched arm disagree, the disagreement IS the
    # composition effect the matched arm exists to remove.
    raw = []
    for s in range(n_seeds):
        pool = d_fs[s][~is_carrier[s]]
        sel = carrier[s][detected[s]]
        if pool.size and sel.size:
            raw.append(percentile_rank_within(pool, d_fs[s][sel]))
    raw_p = np.concatenate(raw) if raw else np.zeros(0)

    return {
        "status": "COMPUTED",
        "population": "one row per DETECTED (seed, concept) cell -- AZ12's "
                      "population, so the two percentile statistics are "
                      "comparable",
        "covariate": value_name,
        "statistic": f"percentile position of the carrier's {value_name} "
                     "within the NON-CARRIER features of its OWN (seed, "
                     "firing bin); 50 means not enriched",
        "n_carrier_cells": int(p.size),
        "n_carrier_cells_without_a_peer_pool": int(unusable),
        "median_percentile": float(np.median(p)),
        "iqr_percentile": [float(np.percentile(p, 25)),
                           float(np.percentile(p, 75))],
        "mean_percentile": float(p.mean()),
        "p05": float(np.percentile(p, 5)), "p95": float(np.percentile(p, 95)),
        "share_above_50th": float((p > 50.0).mean()),
        "per_seed_median_percentile": [
            (None if m is None else float(m)) for m in per_seed_median],
        "across_seed_median_of_medians": float(np.median(med_seed)),
        "across_seed_sd_of_medians": (float(med_seed.std(ddof=1))
                                      if med_seed.size > 1 else None),
        "across_seed_se_of_medians": (
            float(med_seed.std(ddof=1) / np.sqrt(med_seed.size))
            if med_seed.size > 1 else None),
        "deduplicated_per_seed_feature": {
            "n_carrier_features": int(pf.size),
            "median_percentile": (float(np.median(pf)) if pf.size else None),
            "note": "each carrier FEATURE counted once per seed; a feature "
                    "carrying several concepts counts once here and several "
                    "times in the cell arm",
        },
        "composition_confounded_pooled": {
            "median_percentile": (float(np.median(raw_p)) if raw_p.size
                                  else None),
            "note": "carriers against ALL non-carriers regardless of firing "
                    "bin. NOT the answer: d_fs rises with firing rate and "
                    "carriers skew high-firing, so any shift here is partly "
                    "composition. Reported so the size of that effect is "
                    "visible rather than asserted.",
        },
        "by_firing_stratum": strata,
    }


def support_verdict(dist: dict) -> dict:
    """BA7.1's reading, DERIVED from the measured median (gotcha 20).

    Total over the statistic's whole domain AND over the CAUSES of each branch
    (gotcha 30 / BA6): each branch states what it establishes about §AZ11 and,
    where the answer is "nothing", says so instead of letting an uninformative
    null read as support.
    """
    if dist.get("status") != "COMPUTED":
        return {
            "verdict": "NOT_COMPUTED",
            "az11_reading": "no carrier cell had a usable peer pool, so the "
                            "question was never asked rather than answered in "
                            "the negative",
            "median_percentile": None,
        }
    m = dist["median_percentile"]
    if m >= 55.0:
        v = "ENRICHED"
        reading = ("carriers WERE homology-enriched relative to their "
                   "firing-matched peers, so the second matching axis had "
                   "something to remove and removing it left the excess "
                   "intact. AZ11's null result is INFORMATIVE and the "
                   "7.22 pp excess is about alignment.")
    elif m >= 45.0:
        v = "NOT_MATERIALLY_ENRICHED"
        reading = ("carriers were NOT enriched in homology support relative "
                   "to their firing-matched peers, so stratifying on support "
                   "could not have changed anything. AZ11 says NOTHING either "
                   "way and the 7.22 pp excess remains UNATTRIBUTED. An "
                   "uninformative null is not support.")
    else:
        v = "DEPLETED"
        reading = ("carriers sit LOWER in homology support than their "
                   "firing-matched peers. The second axis should then have "
                   "RAISED the excess, and it did (+7.5147 against +7.2219). "
                   "AZ11's survival is informative, in a direction nobody "
                   "predicted. NO MECHANISM IS OFFERED (BA5).")
    return {
        "verdict": v,
        "median_percentile": float(m),
        "bands": {"ENRICHED": ">= 55", "NOT_MATERIALLY_ENRICHED": "45 to 55",
                  "DEPLETED": "< 45"},
        "bands_declared": "PREREGISTRATION BA7.1, before the run",
        "az11_reading": reading,
        "not_a_prediction": "BA2 -- an argmax and a groupby over stored "
                            "arrays; NOT SEALED, NOT TALLIED",
    }


def feature_best_ap(ap_pool: np.ndarray) -> np.ndarray:
    """`max_c AP(f, c)` -- the max over CONCEPTS, one value per feature.

    THE OTHER REDUCTION OF THE SAME ARRAY, and naming it is the point.
    `concept_repro.observed_best_ap` takes the max over FEATURES and returns
    (seed, concept); this takes the max over CONCEPTS and returns
    (seed, feature). They are the two faces of the transpose and confusing them
    is a shape error at best -- it was one here, caught by a run rather than by
    reading (gotcha 21, in its most literal form).
    """
    return ap_pool.max(axis=2)


def statistic_agreement(best_ap: np.ndarray, M: np.ndarray) -> dict:
    """Spearman, per seed, between the two per-feature reductions.

    `max_c AP(f, c)` is the reduction that SELECTS carriers; `M_f` is the
    reduction the regression uses. They are different summaries of the same
    (feature, concept) array and the correlation between them is what says how
    far apart the two instruments are -- gotcha 21, measured rather than
    argued.

    Both arguments are (seed, FEATURE); handing this the (seed, concept) max
    is the mistake this function's shape check exists to refuse.
    """
    if best_ap.shape != M.shape:
        raise ValueError(
            f"statistic_agreement needs two (seed, FEATURE) arrays, got "
            f"{best_ap.shape} and {M.shape}. The (seed, concept) max is a "
            f"different reduction (feature_best_ap vs observed_best_ap).")
    rho = [spearman(best_ap[s], M[s]) for s in range(best_ap.shape[0])]
    r = np.asarray(rho, dtype=np.float64)
    return {
        "per_seed_spearman": [float(x) for x in r],
        "median": float(np.median(r)),
        "min": float(r.min()),
        "max": float(r.max()),
        "statistic_a": "max over the argmax pool of AP(f, c) -- selects carriers",
        "statistic_b": "M_f, the ratio-normalised gain maximum -- the predictor",
    }


def score_predictions(cmp_: dict, positions: dict) -> dict:
    """AZ-P1 and AZ-P2, computed from the values beside them."""
    p2 = (None if positions.get("status") != "COMPUTED"
          else "ABOVE_75TH" if positions["median"] > 75.0
          else "AT_OR_BELOW_75TH")
    return {
        "AZ_P1": {
            "statement": "the carrier excess over the DOUBLY-matched ambient is "
                         "less than half the excess over the singly-matched one",
            "standing": "SEALED -- TALLIED",
            **cmp_,
        },
        "AZ_P2": {
            "statement": "the median carrier sits above the 75th percentile of "
                         "the feature population's log M",
            "standing": "FILED, NOT SEALED -- NOT TALLIED (AZ3.2)",
            "median_percentile": positions.get("median"),
            "outcome": p2,
        },
    }


# ------------------------------------------------------------------ driver

def _decode(arr) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else x for x in arr]


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


def align_to_ap_order(labels: list[str], ap_seeds: list[str],
                      what: str) -> np.ndarray:
    """Permutation taking `labels`' rows into the AP artifact's order.

    Refuses on a set mismatch rather than aligning what it can. `labels` may be
    numeric (`0, 1, 2, ...`) or `seed_<n>`; both are normalised through the
    integer, never through string order -- string order IS the defect.
    """
    want = [seed_label_to_int(s) if str(s).startswith("seed_") else int(s)
            for s in ap_seeds]
    have = [seed_label_to_int(s) if str(s).startswith("seed_") else int(s)
            for s in labels]
    if sorted(have) != sorted(want):
        raise SystemExit(
            f"FATAL: {what} names seeds {sorted(have)} and the AP artifact "
            f"names {sorted(want)}. Refusing to align two different seed sets.")
    where = {v: i for i, v in enumerate(have)}
    return np.array([where[v] for v in want], dtype=np.int64)


def reproduces_ay(single: dict, reference: Path) -> dict:
    """ACCEPTANCE CRITERION 2 -- the single arm against AY7's stored rates.

    Compared as EXACT floats, not to a tolerance: both are the same arithmetic
    on the same inputs, so any difference at all is a changed computation and
    not rounding. An ABSENT key is reported absent, never as agreement
    (gotcha 15 / I3).
    """
    doc = json.loads(reference.read_text())
    ref = doc.get("rates", {})
    pairs = [
        ("rate_B_carrier_match", "rate_B_carrier_match"),
        ("rate_A_carrier_has_any_partner", "rate_A_carrier_has_any_partner"),
        ("conditional_B_given_A", "conditional_B_given_A"),
        ("ambient_match_rate_same_pair_set", "ambient_match_rate_same_pair_set"),
        ("matched_ambient", "firing_rate_matched_ambient"),
        ("n_concept_seedpair_cells_evaluated",
         "n_concept_seedpair_cells_evaluated"),
    ]
    rows, problems = [], []
    for mine, theirs in pairs:
        if theirs not in ref:
            rows.append({"key": theirs, "reference": "<ABSENT>",
                         "this_run": single.get(mine), "same": False})
            problems.append(f"{theirs}: ABSENT (absent is not matching)")
            continue
        a, b = single.get(mine), ref[theirs]
        same = (a == b)
        rows.append({"key": theirs, "reference": b, "this_run": a,
                     "same": bool(same)})
        if not same:
            problems.append(f"{theirs}: reference {b!r} vs this run {a!r}")
    return {
        "reference": str(reference),
        "reference_identity": doc.get("identity"),
        "comparisons": rows,
        "problems": problems,
        "verdict": "REPRODUCES_AY7_EXACTLY" if not problems else "DIFFERS",
    }


def run(ap_path: Path, null_path: Path, gtilde_path: Path, d_fs_path: Path,
        ckpts: list[Path], counts_reference: Path, ay_reference: Path,
        out_path: Path, device: str = "cpu",
        theta: float = THETA_REFERENCE, stratum: str = "pooled",
        n_support: int = N_SUPPORT_BINS) -> dict:
    with h5py.File(ap_path, "r") as hf:
        g = hf[stratum]
        ap_concepts = _decode(g["concepts"][:])
        ap_seeds = _decode(hf["seeds"][:])
        in_pool = g["in_argmax_pool"][:].astype(bool)
        ap = g["ap"][:]
        prevalence = g["prevalence"][:]
        fire_counts = hf["fire_counts"][:]
        ap_identity = _attrs(hf)
    n_rows = int(ap_identity["n_rows"])

    with h5py.File(null_path, "r") as hf:
        null_concepts = _decode(hf["concepts"][:])
        null_seeds = _decode(hf["seeds"][:])
        null_cmax = hf["null_cmax_b"][:]

    assert_labels_align("seed labels (AP vs null)", ap_seeds, null_seeds)
    pool_concepts = [c for c, k in zip(ap_concepts, in_pool) if k]
    assert_labels_align("concept labels (AP pool vs null)",
                        pool_concepts, null_concepts)

    # ---- log M and d_fs, each permuted into the AP order BY LABEL ----------
    with h5py.File(gtilde_path, "r") as hf:
        M_raw = hf["M"][:]
        gt_seeds = _decode(hf["seeds"][:])
        gt_identity = _attrs(hf)
    perm_gt = align_to_ap_order(gt_seeds, ap_seeds, str(gtilde_path))
    M = M_raw[perm_gt]

    with h5py.File(d_fs_path, "r") as hf:
        d_raw = hf["d_fs_cluster_entropy"][:]
        d_seeds = [int(x) for x in hf["seeds"][:]]
        d_identity = _attrs(hf)
    perm_d = align_to_ap_order([str(x) for x in d_seeds], ap_seeds,
                               str(d_fs_path))
    d_fs = d_raw[perm_d]

    bad = ~np.isfinite(M) | (M <= 0)
    if bad.any():
        raise SystemExit(
            f"FATAL: {int(bad.sum())} of {M.size} cells have non-finite or "
            f"non-positive M, so log M is undefined. SPECIFICATION section 4 "
            f"records that set as EMPTY BY MEASUREMENT; a non-empty one is a "
            f"CHANGED INPUT, not a case to absorb (design3_join.log_M).")
    logM = np.log(M)

    # ---- decoders, ordered into the AP artifact's LEXICOGRAPHIC order ------
    meta_seed: dict[Path, int] = {}
    loaded: dict[Path, torch.Tensor] = {}
    for p in ckpts:
        W, meta = load_decoder(p, device)
        meta_seed[p] = int(meta["seed"])
        loaded[p] = W
    ordered_paths = order_decoders_to(ap_seeds, meta_seed)
    decoders = [loaded[p] for p in ordered_paths]
    assert_labels_align("decoder seed order vs AP seed labels",
                        [f"seed_{meta_seed[p]}" for p in ordered_paths],
                        ap_seeds)
    lex_to_numeric = np.argsort([meta_seed[p] for p in ordered_paths])

    n_anchors = len(decoders)
    partner, mnn_sim = mnn_partner_matrix(decoders, theta)

    # ---- ACCEPTANCE 1: the counts, before anything is reported -------------
    acceptance_counts = verify_counts_bit_identical(
        mnn_sim, n_anchors, theta, counts_reference, lex_to_numeric)
    if acceptance_counts["verdict"] != "BIT_IDENTICAL":
        print("ACCEPTANCE (counts) FAILED",
              json.dumps(acceptance_counts, indent=2), flush=True)
        raise SystemExit(1)
    print(f"[support] acceptance 1: {acceptance_counts['verdict']}", flush=True)

    # ---- detection, carriers, cells ---------------------------------------
    ap_pool = ap[:, :, in_pool]
    best_ap = observed_best_ap(ap_pool)
    best_gain = to_gain(best_ap, prevalence[in_pool])
    thr = null_thresholds(null_cmax, REFERENCE_PERCENTILE, "pooled")
    detected = detection_matrix(best_gain, thr)
    d = detection_rate(detected)

    carrier, ties = carriers_from_ap(ap_pool)
    fbin = firing_bin(fire_counts.astype(np.float64) / float(n_rows))
    sbin = support_bin_within(fbin, d_fs, n_support)

    n_fbins = len(BIN_EDGES) - 1
    cell_single = cell_index(fbin, None, n_support)
    cell_double = cell_index(fbin, sbin, n_support)

    single = carrier_rates_by_cell(carrier, detected, partner,
                                   cell_single, n_fbins)
    double = carrier_rates_by_cell(carrier, detected, partner,
                                   cell_double, n_fbins * n_support)

    # ---- ACCEPTANCE 2: the single arm reproduces AY7 exactly ---------------
    acceptance_ay = reproduces_ay(single, ay_reference)
    if acceptance_ay["verdict"] != "REPRODUCES_AY7_EXACTLY":
        print("ACCEPTANCE (AY7 rates) FAILED",
              json.dumps(acceptance_ay, indent=2), flush=True)
        raise SystemExit(1)
    print(f"[support] acceptance 2: {acceptance_ay['verdict']}", flush=True)

    cmp_ = excess_comparison(single, double)

    # ---- where carriers sit in log M --------------------------------------
    all_cells = np.ones_like(detected, dtype=bool)
    positions_detected = percentile_positions(carrier, detected, logM)
    positions_all = percentile_positions(carrier, all_cells, logM)
    agreement = statistic_agreement(feature_best_ap(ap_pool), M)

    # ---- BA9: the groupby AZ11.3 named and AZ did not run -----------------
    support_dist = support_distribution(carrier, detected, d_fs, fbin, n_fbins)
    support_read = support_verdict(support_dist)

    # The carrier-side rank correlation: over the DETECTED (seed, concept)
    # cells, does a carrier's own selecting statistic track its log M?
    sel, lm = [], []
    for s in range(n_anchors):
        idx = carrier[s][detected[s]]
        if idx.size:
            sel.append(best_ap[s][detected[s]])
            lm.append(logM[s][idx])
    carrier_rho = (spearman(np.concatenate(sel), np.concatenate(lm))
                   if sel else None)

    from src.sae.train import git_dirty, git_sha

    out = {
        "specification": "PREREGISTRATION amendment AZ section 11 "
                         "(brief sections 3a and 3b)",
        "exploratory": "AZ7 -- cannot become primary under any outcome; no "
                       "mechanism sentence is written for any of it",
        "identity": {"git_sha": git_sha(), "git_dirty": git_dirty(),
                     "measurement": "carrier_support_AZ"},
        "theta": float(theta),
        "reference_percentile": REFERENCE_PERCENTILE,
        "acceptance_criteria": {
            "counts_vs_rf_counts": acceptance_counts,
            "single_arm_vs_carrier_identity_AY": acceptance_ay,
        },
        "seed_axis": {
            "ap_seed_labels_lexicographic": ap_seeds,
            "decoder_meta_seeds_in_that_order":
                [meta_seed[p] for p in ordered_paths],
            "permutation_lexicographic_to_numeric": lex_to_numeric.tolist(),
            "gtilde_seed_labels": gt_seeds,
            "permutation_gtilde_to_ap": perm_gt.tolist(),
            "d_fs_seed_labels": d_seeds,
            "permutation_d_fs_to_ap": perm_d.tolist(),
        },
        "argmax_ties": ties,
        "strata": {
            "firing_source": "corrector_scan.BIN_EDGES",
            "firing_edges": list(BIN_EDGES),
            "n_firing_bins": int(n_fbins),
            "n_support_bins": int(n_support),
            "support_definition": "quantile bins of d_fs_cluster_entropy "
                                  "computed WITHIN each (seed, firing bin)",
            "double_cell_occupancy_seed0": cell_occupancy(
                cell_double, d_fs, n_fbins * n_support),
        },
        "section_3a_double_matching": {
            "single_arm_firing_only": single,
            "double_arm_firing_and_support": double,
            "comparison": cmp_,
        },
        "section_3b_carrier_position_in_logM": {
            "population": "carriers of (seed, concept) cells where the concept "
                          "is DETECTED in that seed; the all-cells arm is "
                          "reported beside it",
            "detected_cells": positions_detected,
            "all_cells": positions_all,
            "statistic_agreement_per_feature": agreement,
            "carrier_side_spearman_bestAP_vs_logM": carrier_rho,
        },
        "section_BA9_carrier_support_vs_firing_matched_peers": {
            "specification": "PREREGISTRATION amendment BA section 7.1 "
                             "(brief section 2) -- the measurement AZ11.3 "
                             "named as what would separate its two causes",
            "not_a_prediction": "BA2 -- NOT SEALED, NOT TALLIED",
            "distribution": support_dist,
            "reading": support_read,
        },
        "detection": {"n_concepts": int(d.size),
                      "share_D_eq_1": float((d == 1.0).mean()),
                      "share_D_eq_0": float((d == 0.0).mean())},
        "predictions": score_predictions(cmp_, positions_detected),
        "source": {"ap": str(ap_path), "null": str(null_path),
                   "gtilde": str(gtilde_path), "d_fs": str(d_fs_path),
                   "counts_reference": str(counts_reference),
                   "ay_reference": str(ay_reference),
                   "ap_identity": ap_identity,
                   "gtilde_identity": gt_identity,
                   "d_fs_identity": d_identity},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ap", type=Path, required=True)
    p.add_argument("--null", type=Path, required=True)
    p.add_argument("--gtilde", type=Path, required=True)
    p.add_argument("--d-fs", type=Path, required=True)
    p.add_argument("--ckpts", nargs="+", type=Path, required=True)
    p.add_argument("--counts-reference", type=Path, required=True)
    p.add_argument("--ay-reference", type=Path, required=True,
                   help="carrier_identity_ay_v3.json -- the single arm must "
                        "reproduce its rates EXACTLY or the job exits non-zero")
    p.add_argument("--expect-ckpts", type=int, default=None)
    p.add_argument("--n-support-bins", type=int, default=N_SUPPORT_BINS)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--theta", type=float, default=THETA_REFERENCE)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.expect_ckpts is not None and len(args.ckpts) != args.expect_ckpts:
        print(f"PRE-FLIGHT FAIL: --expect-ckpts {args.expect_ckpts} but "
              f"{len(args.ckpts)} supplied. A count mismatch silently changes "
              f"which ensemble every rate below describes.", flush=True)
        raise SystemExit(1)

    out = run(args.ap, args.null, args.gtilde, args.d_fs, args.ckpts,
              args.counts_reference, args.ay_reference, args.out,
              args.device, args.theta, n_support=args.n_support_bins)

    s = out["section_3a_double_matching"]
    for name, arm in (("single (firing only)", s["single_arm_firing_only"]),
                      ("double (firing x support)",
                       s["double_arm_firing_and_support"])):
        print(f"\n[3a] {name}: cells {arm['n_cells']}, "
              f"A {arm['rate_A_carrier_has_any_partner']:.4f}  "
              f"matched {arm['matched_ambient']:.4f}  "
              f"excess {arm['excess_A_minus_matched_pp']:+.3f} pp  "
              f"(unusable {arm['n_carriers_without_usable_cell_baseline']:,})")
    c = s["comparison"]
    print(f"\n  AZ-P1 {c['outcome']}: double {c['double_excess_pp']:+.3f} pp "
          f"against half of single "
          f"{c.get('half_of_single_excess_pp', float('nan')):+.3f} pp; "
          f"ratio {c['ratio_double_over_single']}")

    b = out["section_3b_carrier_position_in_logM"]["detected_cells"]
    if b.get("status") == "COMPUTED":
        print(f"\n[3b] carrier percentile in log M over {b['n_cells']:,} "
              f"detected cells: median {b['median']:.1f}, "
              f"IQR [{b['iqr'][0]:.1f}, {b['iqr'][1]:.1f}], "
              f"share above the 75th {b['share_above_75th']:.2%}")
    ag = out["section_3b_carrier_position_in_logM"][
        "statistic_agreement_per_feature"]
    print(f"     Spearman(max AP, M) per feature: median {ag['median']:.4f} "
          f"[{ag['min']:.4f}, {ag['max']:.4f}]")

    ba = out["section_BA9_carrier_support_vs_firing_matched_peers"]
    sd, rd = ba["distribution"], ba["reading"]
    if sd.get("status") == "COMPUTED":
        print(f"\n[BA9] carrier d_fs percentile within its own (seed, firing "
              f"bin) PEER pool, over {sd['n_carrier_cells']:,} cells:")
        print(f"      median {sd['median_percentile']:.2f}  IQR "
              f"[{sd['iqr_percentile'][0]:.2f}, {sd['iqr_percentile'][1]:.2f}]"
              f"  share above 50th {sd['share_above_50th']:.2%}")
        print(f"      composition-confounded pooled median "
              f"{sd['composition_confounded_pooled']['median_percentile']}")
        for r in sd["by_firing_stratum"]:
            if r.get("status") != "COMPUTED":
                continue
            print(f"      stratum {r['stratum']} "
                  f"[{r['firing_range'][0]:.0e}, {r['firing_range'][1]:.0e}) "
                  f"n={r['n_carrier_cells']:>6,}  "
                  f"median pct {r['median_percentile_in_peer_d_fs']:6.2f}  "
                  f"shift {r['shift_nats_mean_over_seeds']:+.4f} nats")
    print(f"\n      VERDICT {rd['verdict']}: {rd['az11_reading']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
