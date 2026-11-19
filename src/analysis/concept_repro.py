"""Concept-level reproducibility -- the detection rate `D(c)`.

PREREGISTRATION amendment AW §1 specifies this measure; amendment AX §2d
authorises it to run. This module does not restate AW1's values, it implements
them.

    A(c, s) = max over the 10,240 features of AP(f, c, s)
    c is DETECTED in seed s when A(c, s) exceeds the reference percentile of
        that concept's OWN across-protein permutation null (Null B)
    D(c) = (# of the 20 anchor seeds in which c is detected) / 20

WHY THE NULL IS ALSO THE SELECTION CONTROL (AW1.4, the load-bearing part).
`A(c, s)` is a maximum over 10,240 features, so it is inflated by selection --
and that inflation is SHARED across seeds, so it inflates apparent
reproducibility too. A shared upward bias looks like shared signal and points
toward this project's hypothesis. The null is taken as a maximum over the SAME
10,240 features, so it carries the same inflation and the comparison absorbs it.
**That is why §`assert_shared_feature_axis` refuses rather than warns**: if the
two maxima range over different feature sets the control silently stops working
and the result still looks fine.

UNITS. `null_cmax_b` stores NORMALISED GAIN `(AP - p)/(1 - p)`, not raw AP
(AW1.4 flags this). Gain is strictly increasing in AP at fixed `p`, so
`max_f gain = (max_f AP - p)/(1 - p)` exactly and the two maxima select the same
feature. Detection is therefore decided in GAIN units -- the unit the null is
stored in -- while `mean_A`/`sd_A` are reported in AW1.2's own AP units. Under
Null B `p` drifts draw to draw (the length-bin rescaling in
`cross_protein_shift_rows` can merge positions), and each null draw was
normalised by its OWN realised `p`; the observed side is normalised by the
observed `p`. That is the internally consistent pairing. **AW1.4's flagged
question -- whether that drift materially moves the percentile a real `A(c, s)`
receives -- is NOT answered here**, and saying so is the point:
`null_pilot_compliant.h5` stores only the max gain per (seed, shift, concept),
not each draw's realised positive count, so measuring the drift means re-running
the shift schedule against the row-to-protein map. That is a separate job. The
item stays OPEN and is named as open rather than left to read as handled.

POOLED VS PER-SEED NULL (AW8 item 2, flagged there and decided here).
`null_cmax_b` is `(seed, shift, concept)` = 20 x 20 x 703, so a concept has 20
draws per seed and 400 pooled across seeds. Pooling is defensible HERE in a way
it is not for features, because a concept has a real cross-seed identity while a
feature index does not (gotcha 18). Both are computed; the pooled null is the
reference because it is the only one of the two under which the 99th percentile
is estimable, and the per-seed arm is reported beside it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.analysis.artifact_identity import argmax_pool_mask

# AW1.5's curve. The reference is the 95th; every one of these is reported if
# and only if `estimable_percentiles` says the draw count supports it.
CURVE_PERCENTILES = (50.0, 75.0, 90.0, 95.0, 99.0, 99.9)
REFERENCE_PERCENTILE = 95.0


# ------------------------------------------------------- estimability (AX §1b)

def estimable_percentiles(n_draws: int,
                          percentiles=CURVE_PERCENTILES) -> dict[float, dict]:
    """Which percentiles an `n_draws`-sample can estimate WITHOUT extrapolating.

    THE RULE, stated as a property of the sample rather than as a convention:
    percentile `q` is estimable from `n` draws when `n * (1 - q/100) >= 1`, i.e.
    at least one draw lies at or above the quantile, so the estimate is an
    ORDER STATISTIC OF THE SAMPLE rather than a point beyond its support. Below
    that bound the sample maximum is the largest value that exists and any
    higher percentile is the maximum wearing a different label -- which reads as
    a number and is not one.

    `expected_draws_above` is reported for every percentile, estimable or not,
    so a reader can re-derive the verdict instead of taking it (gotcha 20).
    """
    n = int(n_draws)
    if n <= 0:
        raise ValueError(f"n_draws must be positive, got {n}")
    out = {}
    for q in percentiles:
        # `n * (100 - q) / 100`, NOT `n * (1 - q/100)`. The two are equal in
        # exact arithmetic and not in floating point: the second gives
        # 1.0000000000000009 at n=20, q=95 -- a boundary case that decides
        # whether the per-seed arm supports the reference percentile at all.
        # A verdict must not turn on which way the last bit rounded.
        above = n * (100.0 - float(q)) / 100.0
        out[float(q)] = {
            "expected_draws_at_or_above": float(above),
            "estimable": bool(above >= 1.0),
        }
    return out


# --------------------------------------------------------- positional guards
# gotcha 27/29: for a MAPPING defect only a POSITIONAL check discriminates.
# Neither a total nor a rank correlation can see a permutation.

def assert_labels_align(name: str, a: list[str], b: list[str]) -> None:
    """Elementwise label agreement -- never `set(a) == set(b)`, which is blind
    to order, and never `len(a) == len(b)`, which is blind to everything."""
    if len(a) != len(b):
        raise SystemExit(f"FATAL: {name} lengths differ: {len(a)} vs {len(b)}")
    bad = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if bad:
        raise SystemExit(
            f"FATAL: {name} disagree at {len(bad)} of {len(a)} positions. "
            f"First 5 (index, A, B): {bad[:5]}. A positional join across these "
            "two artifacts would mis-pair silently.")


def assert_shared_feature_axis(fire_rate_null: np.ndarray,
                               fire_counts_ap: np.ndarray,
                               n_rows: int,
                               tol: float = 1e-9) -> dict:
    """The observed and null maxima range over the SAME features of the SAME
    dictionaries -- asserted positionally, not inherited from provenance.

    `null_pilot_compliant.h5` stores `fire_rate` (seed, feature) and the AP
    artifact stores `fire_counts` (seed, feature) for the same scored universe.
    Requiring `fire_rate[s, j] == fire_counts[s, j] / n_rows` at EVERY cell ties
    both axes at once: permuting seeds breaks it, permuting features breaks it,
    and a different dictionary breaks it. AW1.4's selection control is valid
    only while this holds, which is why a mismatch is fatal rather than noted.

    Returns the agreement figures so the check reports a DIFF, not a verdict.
    """
    if fire_rate_null.shape != fire_counts_ap.shape:
        raise SystemExit(
            f"FATAL: feature/seed axes differ in SHAPE -- null {fire_rate_null.shape} "
            f"vs AP {fire_counts_ap.shape}. The two maxima cannot be over the "
            "same feature set.")
    implied = fire_counts_ap.astype(np.float64) / float(n_rows)
    delta = np.abs(fire_rate_null - implied)
    n_cells = int(delta.size)
    n_ok = int((delta <= tol).sum())
    worst = float(delta.max()) if n_cells else 0.0
    if n_ok != n_cells:
        raise SystemExit(
            f"FATAL: the null's `fire_rate` and the AP artifact's implied "
            f"firing rate agree at only {n_ok:,}/{n_cells:,} cells "
            f"(max |delta| = {worst:.3e}). Either the seed axis, the feature "
            "axis, or the scored universe differs -- AW1.4's selection control "
            "requires both maxima over the same 10,240 features.")
    return {"n_cells": n_cells, "n_agreeing": n_ok, "max_abs_delta": worst}


# ---------------------------------------------------------------- the measure

def observed_best_ap(ap: np.ndarray) -> np.ndarray:
    """A(c, s) in AP units: max over the FEATURE axis of `ap[seed, feature, concept]`."""
    if ap.ndim != 3:
        raise ValueError(f"expected (seed, feature, concept), got {ap.shape}")
    return ap.max(axis=1)          # -> (seed, concept)


def to_gain(best_ap: np.ndarray, prevalence: np.ndarray) -> np.ndarray:
    """(A - p)/(1 - p), broadcasting a per-CONCEPT prevalence over seeds.

    Strictly increasing in `A` at fixed `p`, so this commutes with the max --
    `max_f gain = (max_f AP - p)/(1 - p)`. Taking the max first is exact, not an
    approximation.
    """
    p = np.asarray(prevalence, dtype=np.float64)
    if p.ndim != 1 or p.shape[0] != best_ap.shape[-1]:
        raise ValueError(
            f"prevalence must be one value per concept: got {p.shape} against "
            f"best_ap {best_ap.shape}. A prevalence broadcast along the wrong "
            "axis would silently normalise every concept by the first one's p.")
    if (p >= 1.0).any() or (p < 0.0).any():
        raise ValueError("prevalence outside [0, 1) -- (1 - p) would not be a "
                         "positive normaliser")
    return (best_ap - p) / (1.0 - p)


def null_thresholds(null_cmax: np.ndarray, percentile: float,
                    mode: str) -> np.ndarray:
    """Per-concept detection threshold from `null_cmax[seed, shift, concept]`.

    mode 'pooled'   -> one threshold per concept, over all seed x shift draws,
                       shape (n_concepts,)
    mode 'per_seed' -> one threshold per (seed, concept), over that seed's own
                       shifts, shape (n_seeds, n_concepts)

    The mode is explicit rather than inferred from shape, because both
    reductions are legal on the same array and picking the wrong one produces a
    complete, plausible, wrong answer.
    """
    if null_cmax.ndim != 3:
        raise ValueError(f"expected (seed, shift, concept), got {null_cmax.shape}")
    if mode == "pooled":
        flat = null_cmax.reshape(-1, null_cmax.shape[2])
        return np.percentile(flat, percentile, axis=0)
    if mode == "per_seed":
        return np.percentile(null_cmax, percentile, axis=1)
    raise ValueError(f"unknown mode {mode!r}; expected 'pooled' or 'per_seed'")


def detection_matrix(observed_gain: np.ndarray,
                     thresholds: np.ndarray) -> np.ndarray:
    """(seed, concept) booleans. STRICT exceedance, per AW1.2's 'exceeds'.

    Broadcasting handles both threshold shapes: `(n_concepts,)` pooled and
    `(n_seeds, n_concepts)` per-seed.
    """
    if thresholds.ndim == 1:
        if thresholds.shape[0] != observed_gain.shape[1]:
            raise ValueError(
                f"pooled thresholds {thresholds.shape} do not match the concept "
                f"axis of {observed_gain.shape}")
    elif thresholds.shape != observed_gain.shape:
        raise ValueError(
            f"per-seed thresholds {thresholds.shape} do not match "
            f"{observed_gain.shape}")
    return observed_gain > thresholds


def detection_rate(detected: np.ndarray) -> np.ndarray:
    """D(c) -- the share of the anchor seeds detecting each concept."""
    if detected.dtype != np.bool_:
        raise ValueError("detection_matrix must be boolean")
    return detected.mean(axis=0)


def distribution_summary(d: np.ndarray, interior=(0.1, 0.9)) -> dict:
    """The headline object's shape (AW1.7). A single scalar is NOT the headline,
    so the summary reports the mass structure, not just a mean."""
    d = np.asarray(d, dtype=np.float64)
    if d.size == 0:
        raise ValueError("no concepts -- refusing to summarise an empty set")
    lo, hi = interior
    return {
        "n_concepts": int(d.size),
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "n_exactly_zero": int((d == 0.0).sum()),
        "share_exactly_zero": float((d == 0.0).mean()),
        "n_exactly_one": int((d == 1.0).sum()),
        "share_exactly_one": float((d == 1.0).mean()),
        "n_interior": int(((d > lo) & (d < hi)).sum()),
        "interior_mass": float(((d > lo) & (d < hi)).mean()),
        "interior_bounds": [float(lo), float(hi)],
        "histogram_21_bins": np.bincount(
            np.rint(d * 20).astype(int), minlength=21).tolist(),
    }


# ---------------------------------------------------------- §4b: D(c) vs size

def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation with AVERAGE ranks for ties.

    Hand-rolled rather than imported so it is unit-tested here against a known
    value; `D(c)` is heavily tied (many concepts at exactly 0 or 1), and a
    tie-handling that assigns ordinal ranks would report a correlation driven by
    input order.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"need two 1-D arrays of equal length, got {x.shape}, {y.shape}")
    if x.size < 2:
        raise ValueError("a correlation over fewer than 2 points is not defined")
    rx, ry = _average_ranks(x), _average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    if denom == 0.0:
        raise ValueError(
            "a ranked input is constant -- Spearman is undefined, and "
            "reporting 0.0 would read as 'measured, no association'")
    return float((rx * ry).sum() / denom)


def _average_ranks(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.size, dtype=np.float64)
    ranks[order] = np.arange(1, v.size + 1, dtype=np.float64)
    sv = v[order]
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def decile_table(d: np.ndarray, size: np.ndarray, n_bins: int = 10) -> list[dict]:
    """`D(c)` binned by prevalence decile. Reported beside the correlation
    because a rank correlation says nothing about the SHAPE (gotcha 21)."""
    d = np.asarray(d, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    edges = np.percentile(size, np.linspace(0, 100, n_bins + 1))
    # np.digitize with the interior edges; last bin closed on the right.
    idx = np.clip(np.digitize(size, edges[1:-1], right=True), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        out.append({
            "decile": b,
            "n_concepts": int(m.sum()),
            "size_min": float(size[m].min()) if m.any() else None,
            "size_max": float(size[m].max()) if m.any() else None,
            "mean_D": float(d[m].mean()) if m.any() else None,
            "median_D": float(np.median(d[m])) if m.any() else None,
        })
    return out


# ----------------------------------------------------------------- the driver

def run(ap_path: Path, null_path: Path, out_path: Path,
        stratum: str = "pooled") -> dict:
    """Everything except argument parsing, so the whole pipeline can be handed a
    fixture with a known answer (gotcha 13 extension 4's corollary: a suite that
    never executes the entry point cannot catch an entry-point bug)."""
    # ---- read the observed side ------------------------------------------
    with h5py.File(ap_path, "r") as hf:
        g = hf[stratum]
        ap_concepts = [c.decode() if isinstance(c, bytes) else c
                       for c in g["concepts"][:]]
        ap_seeds = [s.decode() if isinstance(s, bytes) else s
                    for s in hf["seeds"][:]]
        in_pool = g["in_argmax_pool"][:].astype(bool)
        ap = g["ap"][:]
        prevalence = g["prevalence"][:]
        n_positives = g["n_positives"][:]
        fire_counts = hf["fire_counts"][:]
        ap_identity = _attrs(hf)
    n_rows = int(ap_identity["n_rows"])

    # gotcha 15: the stored pool mask is recomputed and required to agree.
    if not np.array_equal(in_pool, np.array(argmax_pool_mask(ap_concepts), dtype=bool)):
        raise SystemExit("FATAL: stored `in_argmax_pool` disagrees with "
                         "`argmax_pool_mask` recomputed from the concept names")

    # ---- read the null side ----------------------------------------------
    with h5py.File(null_path, "r") as hf:
        null_concepts = [c.decode() if isinstance(c, bytes) else c
                         for c in hf["concepts"][:]]
        null_seeds = [s.decode() if isinstance(s, bytes) else s
                      for s in hf["seeds"][:]]
        if "null_cmax_b" not in hf:
            raise SystemExit(
                "FATAL: no `null_cmax_b` in the null artifact. AW1.4 adopts the "
                "ACROSS-PROTEIN permutation (Null B); `null_cmax_a` is the "
                "within-protein circular shift and answers a different question "
                "(localisation, not total grounding). Refusing to substitute it.")
        if "dest_schedule" not in hf:
            raise SystemExit(
                "FATAL: the null artifact has no `dest_schedule`. That dataset "
                "is written only on a Null B run (`null_pilot.py` writes it "
                "beside `null_cmax_b`), so its absence means `null_cmax_b` was "
                "not produced by the across-protein construction.")
        null_cmax = hf["null_cmax_b"][:]
        fire_rate = hf["fire_rate"][:]
        null_identity = _attrs(hf)

    # ---- §4c hazard 1 and 2: positional guards ---------------------------
    assert_labels_align("seed labels (AP vs null)", ap_seeds, null_seeds)
    pool_concepts = [c for c, k in zip(ap_concepts, in_pool) if k]
    assert_labels_align("concept labels (AP pool vs null)",
                        pool_concepts, null_concepts)
    axis = assert_shared_feature_axis(fire_rate, fire_counts, n_rows)

    # ---- identity ---------------------------------------------------------
    shared_keys = ["library_sha", "truncation_policy", "scored_row_set",
                   "n_rows", "argmax_pool_n", "n_concepts"]
    identity_report = {}
    for k in shared_keys:
        a, b = ap_identity.get(k, "<ABSENT>"), null_identity.get(k, "<ABSENT>")
        identity_report[k] = {"ap": a, "null": b, "agree": bool(a == b)}
        if a == "<ABSENT>" or b == "<ABSENT>":
            identity_report[k]["agree"] = False   # absent is never matching
    mismatched = [k for k, v in identity_report.items() if not v["agree"]]
    if mismatched:
        raise SystemExit(
            f"FATAL: identity keys disagree or are absent: {mismatched}. "
            f"Detail: { {k: identity_report[k] for k in mismatched} }")

    # ---- the measure ------------------------------------------------------
    ap_pool = ap[:, :, in_pool]
    prev_pool = prevalence[in_pool]
    npos_pool = n_positives[in_pool]
    best_ap = observed_best_ap(ap_pool)                 # (seed, concept), AP units
    best_gain = to_gain(best_ap, prev_pool)             # (seed, concept), gain units

    n_seeds, n_shifts, n_concepts = null_cmax.shape
    pooled_draws = n_seeds * n_shifts
    est_pooled = estimable_percentiles(pooled_draws)
    est_per_seed = estimable_percentiles(n_shifts)
    log_size = np.log(npos_pool.astype(np.float64))

    curve = {}
    for q in CURVE_PERCENTILES:
        entry = {"pooled": None, "per_seed": None,
                 "pooled_estimable": est_pooled[q]["estimable"],
                 "per_seed_estimable": est_per_seed[q]["estimable"],
                 "pooled_expected_draws_at_or_above":
                     est_pooled[q]["expected_draws_at_or_above"],
                 "per_seed_expected_draws_at_or_above":
                     est_per_seed[q]["expected_draws_at_or_above"]}
        if est_pooled[q]["estimable"]:
            det = detection_matrix(best_gain, null_thresholds(null_cmax, q, "pooled"))
            entry["pooled"] = _arm(det, detection_rate(det), log_size)
        if est_per_seed[q]["estimable"]:
            det = detection_matrix(best_gain, null_thresholds(null_cmax, q, "per_seed"))
            entry["per_seed"] = _arm(det, detection_rate(det), log_size)
        curve[str(q)] = entry

    if not est_pooled[REFERENCE_PERCENTILE]["estimable"]:
        raise SystemExit(
            f"FATAL: the {REFERENCE_PERCENTILE}th-percentile reference is NOT "
            f"estimable from {pooled_draws} pooled draws. AX §1b: report and "
            "stop -- this is a specification question, not a judgement call.")

    ref_thr = null_thresholds(null_cmax, REFERENCE_PERCENTILE, "pooled")
    ref_det = detection_matrix(best_gain, ref_thr)
    d_ref = detection_rate(ref_det)

    # ---- §4b: D(c) against annotated-residue count ------------------------
    # A CONSTANT `D` makes Spearman undefined. Reporting 0.0 there would read as
    # "measured, no association" and reporting False would read as "checked and
    # absent" -- gotcha 25(b): a constant is UNINFORMATIVE and its contribution
    # is None, never a value. So the association gets a third outcome rather
    # than crashing or fabricating one.
    if len(np.unique(d_ref)) < 2:
        rho, rho_status = None, "UNINFORMATIVE_CONSTANT_D"
    else:
        rho, rho_status = spearman(d_ref, log_size), "COMPUTED"

    out = {
        "specification": "PREREGISTRATION amendment AW §1; authorised by AX §2d",
        "null": {
            "dataset": "null_cmax_b",
            "construction": "across-protein, length-stratified permutation "
                            "(Null B, amendment G4) -- confirmed from the "
                            "writer and from `dest_schedule`'s presence",
            "shape": [int(n_seeds), int(n_shifts), int(n_concepts)],
            "draws_per_concept_per_seed": int(n_shifts),
            "draws_per_concept_pooled": int(pooled_draws),
            "estimable_pooled": {str(k): v for k, v in est_pooled.items()},
            "estimable_per_seed": {str(k): v for k, v in est_per_seed.items()},
            "units": "normalised gain (AP - p)/(1 - p)",
        },
        "positional_guards": {
            "seed_labels_elementwise": f"{len(ap_seeds)}/{len(ap_seeds)}",
            "concept_labels_elementwise": f"{len(pool_concepts)}/{len(pool_concepts)}",
            "shared_feature_axis": axis,
        },
        "identity": identity_report,
        "reference_percentile": REFERENCE_PERCENTILE,
        "headline": {
            "distribution_of_D": distribution_summary(d_ref),
            "mean_A_over_concepts": float(best_ap.mean()),
            "detection_rate_mean": float(d_ref.mean()),
        },
        "percentile_curve": curve,
        "prevalence_association": {
            "spearman_D_vs_log_annotated_residues": rho,
            "status": rho_status,
            "AX_P2": {
                "statement": "Spearman(D(c), log annotated-residue count) > 0.5",
                "measured": rho,
                "outcome": ("UNSCOREABLE_CONSTANT_D" if rho is None
                            else "ABOVE_0.5" if rho > 0.5 else "AT_OR_BELOW_0.5"),
            },
            "decile_table": decile_table(d_ref, npos_pool.astype(np.float64)),
        },
        "per_concept": [
            {"concept": c,
             "D": float(d_ref[i]),
             "mean_A": float(best_ap[:, i].mean()),
             "sd_A": float(best_ap[:, i].std(ddof=1)),
             "mean_A_gain": float(best_gain[:, i].mean()),
             "null_threshold_p95": float(ref_thr[i]),
             "n_annotated_residues": int(npos_pool[i]),
             "prevalence": float(prev_pool[i])}
            for i, c in enumerate(pool_concepts)],
        "source": {"ap": str(ap_path), "null": str(null_path),
                   "stratum": stratum,
                   "ap_identity": ap_identity, "null_identity": null_identity},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    return out


def association_with_size(d: np.ndarray, log_size: np.ndarray) -> dict:
    """Spearman(D, log annotated-residue count), with the constant case named.

    Computed at EVERY estimable percentile, not only at the reference, because
    AW1.5 requires a conclusion to hold across the reported percentile range
    (AB1 decision A, AU1's extension) -- and a coefficient sitting close to its
    pre-declared bar is exactly where a single-percentile reading would mislead.
    """
    if len(np.unique(d)) < 2:
        return {"spearman": None, "status": "UNINFORMATIVE_CONSTANT_D"}
    return {"spearman": spearman(d, log_size), "status": "COMPUTED"}


def _arm(det: np.ndarray, d: np.ndarray, log_size: np.ndarray) -> dict:
    return {"detection_rate_mean": float(d.mean()),
            "share_of_concept_seed_cells_detected": float(det.mean()),
            "distribution_of_D": distribution_summary(d),
            "association_with_log_size": association_with_size(d, log_size)}


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ap", type=Path)
    p.add_argument("--null", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--stratum", default="pooled")
    p.add_argument("--inspect-only", action="store_true",
                   help="AX §1b: report the null's draw count and which "
                        "percentiles it supports, then exit without computing "
                        "any concept-level quantity.")
    args = p.parse_args()

    if args.inspect_only:
        _inspect(args.null)
        return
    if args.ap is None or args.out is None:
        raise SystemExit("--ap and --out are required unless --inspect-only")

    out = run(args.ap, args.null, args.out, args.stratum)
    h = out["headline"]["distribution_of_D"]
    print(f"\nD(c) over {h['n_concepts']} concepts at the "
          f"{REFERENCE_PERCENTILE}th-percentile pooled null:")
    print(f"  mean {h['mean']:.4f}  median {h['median']:.4f}")
    print(f"  exactly 0: {h['n_exactly_zero']} ({h['share_exactly_zero']:.1%})   "
          f"exactly 1: {h['n_exactly_one']} ({h['share_exactly_one']:.1%})")
    print(f"  interior (0.1 < D < 0.9): {h['n_interior']} "
          f"({h['interior_mass']:.1%})")
    ax2 = out["prevalence_association"]["AX_P2"]
    shown = "undefined" if ax2["measured"] is None else f"{ax2['measured']:.4f}"
    print(f"\nAX-P2: Spearman(D, log annotated residues) = {shown} "
          f"-> {ax2['outcome']}")
    print("\npercentile curve (pooled arm):")
    for q, e in out["percentile_curve"].items():
        if e["pooled"] is None:
            print(f"  p{q}: NOT ESTIMABLE "
                  f"({e['pooled_expected_draws_at_or_above']:.2f} draws at or "
                  "above) -- DROPPED, not interpolated")
        else:
            print(f"  p{q}: mean D = {e['pooled']['detection_rate_mean']:.4f}, "
                  f"interior "
                  f"{e['pooled']['distribution_of_D']['interior_mass']:.1%}")


def _inspect(null_path: Path) -> None:
    with h5py.File(null_path, "r") as hf:
        shapes = {k: (hf[k].shape, str(hf[k].dtype)) for k in hf.keys()
                  if isinstance(hf[k], h5py.Dataset)}
        attrs = _attrs(hf)
        has_b = "null_cmax_b" in hf
        has_dest = "dest_schedule" in hf
        shape = hf["null_cmax_b"].shape if has_b else None
    print(f"{null_path}")
    for k, (s, d) in sorted(shapes.items()):
        print(f"  {k:<20} {str(s):<22} {d}")
    print(f"\nattrs: {json.dumps(attrs, indent=2, default=str)}")
    print(f"\nnull_cmax_b present: {has_b}; dest_schedule present: {has_dest}")
    if not has_b:
        raise SystemExit("FATAL: no null_cmax_b")
    n_seeds, n_shifts, n_concepts = shape
    pooled = n_seeds * n_shifts
    print(f"\ndraws per concept: {n_shifts} per seed, {pooled} pooled "
          f"({n_seeds} seeds x {n_shifts} shifts), over {n_concepts} concepts")
    for label, n in (("pooled", pooled), ("per-seed", n_shifts)):
        print(f"\n  {label} ({n} draws):")
        for q, e in estimable_percentiles(n).items():
            mark = "estimable" if e["estimable"] else "NOT ESTIMABLE"
            print(f"    p{q:<6} {e['expected_draws_at_or_above']:>8.2f} draws "
                  f"at or above  -> {mark}")
    ref = estimable_percentiles(pooled)[REFERENCE_PERCENTILE]["estimable"]
    print(f"\n{REFERENCE_PERCENTILE}th-percentile reference estimable from the "
          f"pooled null: {ref}")
    if not ref:
        raise SystemExit(
            f"the {REFERENCE_PERCENTILE}th-percentile reference is not "
            "estimable -- AX §1b says report and stop")


if __name__ == "__main__":
    main()
