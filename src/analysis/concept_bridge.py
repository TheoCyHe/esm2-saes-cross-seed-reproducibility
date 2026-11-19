"""The bridge between the two headlines -- PREREGISTRATION amendment AY.

Feature-level reproducibility does not track alignment (bounded at 0.29 pp per
SD, sign unidentified); concept-level reproducibility is strongly positive
(75.25% of concepts detected in all 20 seeds). This module measures the three
things that decide what the writeup may say about that pair, ALL from artifacts
already on disk and with no activation pass:

  AY §2a(i)   what the null actually is -- per (concept, seed) or pooled -- and
              how much of the 75.25% survives the per-seed arm. PLUS the
              DETECTION MARGIN, which is the quantity the independence concern
              is really about and is not stored anywhere.
  AY §2a(ii)  the 43 never-detected concepts: identity, size, feature key,
              experimental evidence.
  AY §2c      does the bimodality survive conditioning on prevalence, at EVERY
              estimable percentile.
  AY §3       the binned detection curve and the SATURATION THRESHOLD, which
              replace the rank correlation as the summary of the
              detection-versus-annotation relationship.

WHY THE RANK CORRELATION IS RETIRED (§3, and the reasoning is already
measured). `Spearman(D, log annotated residues)` reads 0.5115 at the reference
percentile but runs 0.2122 / 0.2979 / 0.4335 / 0.5115 / 0.6035 across the
curve, crossing 0.5 only between the 90th and the 95th. Under AW1.5's standing
rule a conclusion counts only if it holds across the reported range, so three
of five percentiles sit below the bar and the annotation-density reading is NOT
licensed. A rank correlation is also the wrong summary for a STEP: with 529 of
703 values tied at exactly 1.0, raising the detection bar pushes concepts below
saturation and steepens the apparent association mechanically. `binned_curve`
and `saturation_threshold` are what replace it.

INTERVALS ARE BUILT FROM THE EMPIRICAL BETWEEN-CONCEPT DISPERSION, NOT WILSON.
A Wilson interval describes a pooled proportion drawn from independent trials
at a common rate. The concepts in a bin do not share a rate -- that is the
entire content of the measurement -- so Wilson understates the relevant noise.
`sd(D within bin) / sqrt(n_bin)` carries the between-concept spread, which is
the same substitution amendment AV12b made for the alignment-decile panel.

REUSE, NOT REIMPLEMENTATION. Detection is `concept_repro`'s
`detection_matrix(to_gain(observed_best_ap(ap)), null_thresholds(...))`. If
this module re-derived any of that, the two could disagree and the artifact of
record would be ambiguous about which one produced 75.25%.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.analysis.concept_repro import (
    CURVE_PERCENTILES,
    REFERENCE_PERCENTILE,
    assert_labels_align,
    assert_shared_feature_axis,
    detection_matrix,
    detection_rate,
    distribution_summary,
    estimable_percentiles,
    null_thresholds,
    observed_best_ap,
    spearman,
    to_gain,
)

# `av_counts` owns the definition of an experimental residue (the two tiers
# unioned). Importing it is what stops this module inventing a second one.
from src.analysis.av_counts import EXPERIMENTAL_TIERS


# ----------------------------------------------- AY §2a(i): what the null IS

def null_construction_report(hf) -> dict:
    """Is the null per (concept, seed), or per concept pooled across seeds?

    ANSWERED FROM THE ARRAY'S OWN SHAPE, not from the artifact's name and not
    from a code comment. Two separate questions hide under one phrasing and
    they have different answers, which is why both are reported:

      1. ARE THERE PER-SEED DRAWS?  `null_cmax_b` is (seed, shift, concept), so
         yes -- 20 draws per concept per seed. A per-seed threshold is
         therefore computable, and `concept_repro` computes it.
      2. IS THE PERMUTATION SCHEDULE PER-SEED?  `dest_schedule` is
         (n_shifts, n_proteins) with NO SEED AXIS. So the SAME `n_shifts` label
         permutations are applied in every seed. That is deliberate
         (`null_pilot.py`: "ONE dest-protein permutation schedule shared across
         ALL seeds ... so seed-to-seed differences in the null are due to the
         SEED, not to the schedule") and it means the per-seed arm is NOT a
         fully independent re-draw. It removes the shared-THRESHOLD channel and
         leaves the shared-SCHEDULE channel standing.

    The rank of `dest_schedule` is the evidence for (2) and it is checked, not
    asserted: a seed-varying schedule would have to be 3-D.
    """
    out: dict = {}
    cmax = hf["null_cmax_b"]
    n_seeds, n_shifts, n_concepts = cmax.shape
    out["null_cmax_b_shape"] = [int(n_seeds), int(n_shifts), int(n_concepts)]
    out["draws_per_concept_per_seed"] = int(n_shifts)
    out["draws_per_concept_pooled"] = int(n_seeds * n_shifts)
    out["per_seed_draws_exist"] = True

    for name in ("dest_schedule", "shift_schedule"):
        if name not in hf:
            out[name] = {"present": False}
            continue
        shp = tuple(int(x) for x in hf[name].shape)
        # A schedule that varied by seed would carry a seed axis. Rank is the
        # discriminating fact; a total or a shape-length check is not.
        shared = len(shp) == 2 and shp[0] == n_shifts
        out[name] = {
            "present": True,
            "shape": list(shp),
            "has_seed_axis": bool(len(shp) == 3),
            "shared_across_seeds": bool(shared),
        }

    dest = out.get("dest_schedule", {})
    if not dest.get("present"):
        verdict = "NO_DEST_SCHEDULE"
    elif dest["has_seed_axis"]:
        verdict = "PER_SEED_DRAWS_AND_PER_SEED_SCHEDULE"
    elif dest["shared_across_seeds"]:
        verdict = "PER_SEED_DRAWS_BUT_ONE_SHARED_SCHEDULE"
    else:
        verdict = "SCHEDULE_SHAPE_UNRECOGNISED"
    out["verdict"] = verdict
    return out


def detection_margin(observed_gain: np.ndarray,
                     thresholds: np.ndarray) -> np.ndarray:
    """(seed, concept) ratio of observed gain to its detection threshold.

    THE QUANTITY THE INDEPENDENCE CONCERN IS ACTUALLY ABOUT. "Detected in all
    20 seeds" is fragile if a single slightly-underestimated threshold produced
    all twenty exceedances, and robust if every seed cleared by a wide margin.
    The share of concepts at `D = 1` cannot distinguish those; this can, and it
    is stored nowhere.

    A non-positive threshold would make the ratio meaningless rather than
    large, so it is reported as NaN instead of silently becoming +inf.
    """
    thr = np.asarray(thresholds, dtype=np.float64)
    if thr.ndim == 1:
        thr = np.broadcast_to(thr, observed_gain.shape)
    out = np.divide(observed_gain, thr,
                    out=np.full(observed_gain.shape, np.nan),
                    where=thr > 0)
    return out


def margin_summary(margin: np.ndarray, d: np.ndarray,
                   group_mask: np.ndarray) -> dict:
    """Distribution of the WEAKEST seed's margin within a group of concepts.

    The minimum over seeds is the right reduction: a concept is detected in all
    20 seeds only as securely as its worst seed does it.
    """
    if not group_mask.any():
        return {"n_concepts": 0, "status": "EMPTY_GROUP"}
    m = np.nanmin(margin[:, group_mask], axis=0)
    finite = m[np.isfinite(m)]
    if finite.size == 0:
        return {"n_concepts": int(group_mask.sum()),
                "status": "NO_FINITE_MARGIN"}
    return {
        "n_concepts": int(group_mask.sum()),
        "status": "COMPUTED",
        "min_over_seeds_of_gain_over_threshold": {
            "p0": float(np.min(finite)),
            "p10": float(np.percentile(finite, 10)),
            "p50": float(np.median(finite)),
            "p90": float(np.percentile(finite, 90)),
            "p100": float(np.max(finite)),
        },
        "share_worst_seed_within_2x_of_threshold":
            float((finite < 2.0).mean()),
        "share_worst_seed_within_1_1x_of_threshold":
            float((finite < 1.1).mean()),
    }


# ------------------------------------------- AY §2a(ii): the never-detected

def concept_feature_key(concept: str) -> str:
    """The Swiss-Prot feature key -- everything before the first ':'.

    Concepts are stored as `KEY` or `KEY:descriptor`, and a descriptor may
    itself contain a colon, so the split is on the FIRST one only.
    """
    return concept.split(":", 1)[0]


def concept_descriptor(concept: str) -> str:
    """The `/note` sub-type, or '' for a concept that was never sub-typed."""
    parts = concept.split(":", 1)
    return parts[1] if len(parts) == 2 else ""


def never_detected_report(concepts: list[str], d: np.ndarray,
                          n_annotated: np.ndarray,
                          mean_a: np.ndarray, sd_a: np.ndarray,
                          experimental: dict[str, int]) -> dict:
    """The concepts with `D` exactly 0, with the composition that makes them
    informative rather than merely small.

    AY §2a(ii): the saturating step implies these are sparsely annotated, so
    their SIZE is the expected part. The interesting question is whether any is
    WELL annotated or carries experimental evidence -- a well-annotated concept
    that no seed detects is a genuine negative about what the model misses.
    """
    idx = np.flatnonzero(d == 0.0)
    rows = [{
        "concept": concepts[i],
        "feature_key": concept_feature_key(concepts[i]),
        "descriptor": concept_descriptor(concepts[i]),
        "n_annotated_residues": int(n_annotated[i]),
        "n_experimental_residues": int(experimental.get(concepts[i], 0)),
        "mean_A": float(mean_a[i]),
        "sd_A": float(sd_a[i]),
    } for i in idx]
    rows.sort(key=lambda r: -r["n_annotated_residues"])

    keys: dict[str, int] = {}
    for r in rows:
        keys[r["feature_key"]] = keys.get(r["feature_key"], 0) + 1
    with_exp = [r for r in rows if r["n_experimental_residues"] > 0]
    sizes = np.array([r["n_annotated_residues"] for r in rows], dtype=float)

    return {
        "n_never_detected": len(rows),
        "share_of_pool": float(len(rows) / len(concepts)),
        "size": ({"min": float(sizes.min()), "median": float(np.median(sizes)),
                  "max": float(sizes.max())} if len(rows) else None),
        "feature_key_composition": dict(sorted(keys.items(),
                                               key=lambda kv: -kv[1])),
        "n_with_experimental_evidence": len(with_exp),
        "share_with_experimental_evidence":
            float(len(with_exp) / len(rows)) if rows else None,
        "with_experimental_evidence": with_exp,
        "concepts": rows,
    }


# ------------------------------- AY §2c: does bimodality survive prevalence?

def stratum_edges(size: np.ndarray, n_strata: int) -> np.ndarray:
    """Interior quantile boundaries, REPORTED so the strata are reproducible."""
    qs = np.linspace(0, 100, n_strata + 1)[1:-1]
    return np.percentile(np.asarray(size, dtype=float), qs)


def stratify(size: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(size, dtype=float), edges)


def conditioned_distribution(d: np.ndarray, size: np.ndarray,
                             n_strata: int = 3) -> dict:
    """`D`'s distribution WITHIN prevalence strata.

    THE DECISION THIS INFORMS (AY §2c). If bimodality survives conditioning,
    detection reflects something about the MODEL. If it does not, concept-level
    reproducibility is substantially reporting which concepts Swiss-Prot
    annotates densely enough to be detectable -- the same shape as homology
    support dominating at feature level, and the second time the interesting
    variance would sit in an annotation-density variable.

    Reports the pooled figure beside every stratum so the comparison is in the
    artifact rather than left to a reader with two files open.
    """
    d = np.asarray(d, dtype=np.float64)
    edges = stratum_edges(size, n_strata)
    g = stratify(size, edges)
    strata = []
    for b in range(n_strata):
        m = g == b
        if not m.any():
            strata.append({"stratum": b, "n_concepts": 0,
                           "status": "EMPTY"})
            continue
        s = np.asarray(size, dtype=float)[m]
        strata.append({
            "stratum": b,
            "n_concepts": int(m.sum()),
            "size_min": float(s.min()),
            "size_max": float(s.max()),
            "distribution_of_D": distribution_summary(d[m]),
        })
    return {
        "n_strata": int(n_strata),
        "interior_stratum_edges": [float(e) for e in edges],
        "strata": strata,
        "pooled_distribution_of_D": distribution_summary(d),
    }


# ------------------------------------ AY §3: the step, not the correlation

def saturation_threshold(d: np.ndarray, size: np.ndarray) -> dict:
    """The annotated-residue count above which EVERY concept is detected in
    every seed.

    DEFINED SO IT IS TOTAL (gotcha 24). Three regions of the outcome space, all
    reachable and all named:
      * no concept has `D < 1`      -> SATURATED_EVERYWHERE, threshold is the
                                       minimum observed size
      * no concept above the largest unsaturated one -> NEVER_SATURATES, and
                                       the threshold is None rather than a
                                       number invented from an empty set
      * otherwise                   -> the smallest observed size strictly
                                       greater than the largest size at which a
                                       concept fails to saturate

    The third is the interesting case and the other two are why this returns a
    status rather than a bare float.
    """
    d = np.asarray(d, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    unsat = d < 1.0
    if not unsat.any():
        return {"status": "SATURATED_EVERYWHERE",
                "threshold": float(size.min()),
                "n_concepts_at_or_above": int(size.size)}
    largest_unsat = float(size[unsat].max())
    above = size > largest_unsat
    if not above.any():
        return {"status": "NEVER_SATURATES", "threshold": None,
                "largest_unsaturated_size": largest_unsat,
                "n_concepts_at_or_above": 0}
    thr = float(size[above].min())
    return {
        "status": "SATURATES",
        "threshold": thr,
        "largest_unsaturated_size": largest_unsat,
        "n_concepts_at_or_above": int((size >= thr).sum()),
        "share_of_pool_at_or_above": float((size >= thr).mean()),
    }


def binned_curve(d: np.ndarray, size: np.ndarray, n_bins: int = 10) -> dict:
    """Detection rate against annotated-residue count, with intervals from the
    EMPIRICAL BETWEEN-CONCEPT DISPERSION.

    NOT a Wilson interval. Wilson describes a pooled proportion under
    independent trials at a common rate; the concepts in a bin do not share a
    rate, and that heterogeneity is the measurement. `sd/sqrt(n)` over the
    concepts in the bin carries it. Both are computed and the Wilson half-width
    is reported ALONGSIDE, so the size of the understatement is visible rather
    than argued -- the same move `rf_marginal.per_seed_atoms` makes for the
    naive binomial SE.
    """
    d = np.asarray(d, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    edges = np.percentile(size, np.linspace(0, 100, n_bins + 1))
    idx = np.clip(np.digitize(size, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            rows.append({"bin": b, "n_concepts": 0, "status": "EMPTY"})
            continue
        v = d[m]
        n = int(m.sum())
        sd = float(v.std(ddof=1)) if n > 1 else float("nan")
        se = sd / np.sqrt(n) if n > 1 else float("nan")
        p = float(v.mean())
        wilson = float(np.sqrt(p * (1 - p) / n)) if n > 0 else float("nan")
        rows.append({
            "bin": b,
            "n_concepts": n,
            "size_min": float(size[m].min()),
            "size_max": float(size[m].max()),
            "mean_D": p,
            "sd_D_between_concepts": sd,
            "se_between_concepts": float(se),
            "ci95_lo": float(p - 1.96 * se) if np.isfinite(se) else None,
            "ci95_hi": float(p + 1.96 * se) if np.isfinite(se) else None,
            # Reported for comparison ONLY. Never the interval of record.
            "naive_binomial_se_for_comparison": wilson,
            "understatement_ratio_se_over_naive":
                float(se / wilson) if (np.isfinite(se) and wilson > 0)
                else None,
        })
    return {"n_bins": int(n_bins),
            "bin_edges": [float(e) for e in edges],
            "interval_basis": "empirical between-concept dispersion "
                              "sd(D)/sqrt(n_bin); NOT Wilson",
            "bins": rows}


# ------------------------------------------------------------------ driver

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


def _decode(arr) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else x for x in arr]


def experimental_counts_for(labels: Path, concepts: list[str]) -> dict[str, int]:
    """Experimental residues per concept, using `av_counts`' definition.

    The two tiers are UNIONED (a residue in both must not be counted twice),
    which is why this reads the datasets rather than summing their lengths.
    """
    out: dict[str, int] = {}
    with h5py.File(labels, "r") as hf:
        grp = hf["concepts"]
        for c in concepts:
            if c not in grp:
                out[c] = 0
                continue
            rows = [grp[c][t][:] for t in EXPERIMENTAL_TIERS if t in grp[c]]
            out[c] = int(np.unique(np.concatenate(rows)).size) if rows else 0
    return out


# The identity keys both inputs must carry and agree on -- `concept_repro`'s
# list, because the two modules read the same AP and null artifacts.
SHARED_IDENTITY_KEYS = ("library_sha", "truncation_policy", "scored_row_set",
                        "n_rows", "argmax_pool_n", "n_concepts")


def compare_identities(ap_identity: dict, null_identity: dict,
                       keys: tuple[str, ...] = SHARED_IDENTITY_KEYS) -> dict:
    """Compare the identities `run` records, and REFUSE on any disagreement.

    Amendment CT, §CT8 item 1 (gotcha 15): `run` stored both identities in its
    output and compared neither. An ABSENT key on either side is a refusal,
    never a match -- the same rule `concept_repro` applies to the same pair.
    """
    report = {}
    for k in keys:
        a = ap_identity.get(k, "<ABSENT>")
        b = null_identity.get(k, "<ABSENT>")
        agree = (a == b) and a != "<ABSENT>" and b != "<ABSENT>"
        report[k] = {"ap": a, "null": b, "agree": bool(agree)}
    bad = [k for k, v in report.items() if not v["agree"]]
    if bad:
        raise SystemExit(
            f"FATAL: AP and null identity keys disagree or are absent: {bad}. "
            f"Detail: { {k: report[k] for k in bad} }")
    return report


def run(ap_path: Path, null_path: Path, labels_path: Path,
        out_path: Path, stratum: str = "pooled", n_strata: int = 3) -> dict:
    """Everything except argument parsing, so the whole pipeline can be handed
    a fixture with a known answer (gotcha 13 extension 4's corollary)."""
    with h5py.File(ap_path, "r") as hf:
        g = hf[stratum]
        ap_concepts = _decode(g["concepts"][:])
        ap_seeds = _decode(hf["seeds"][:])
        in_pool = g["in_argmax_pool"][:].astype(bool)
        ap = g["ap"][:]
        prevalence = g["prevalence"][:]
        n_positives = g["n_positives"][:]
        fire_counts = hf["fire_counts"][:]
        ap_identity = _attrs(hf)
    n_rows = int(ap_identity["n_rows"])

    with h5py.File(null_path, "r") as hf:
        null_concepts = _decode(hf["concepts"][:])
        null_seeds = _decode(hf["seeds"][:])
        null_cmax = hf["null_cmax_b"][:]
        fire_rate = hf["fire_rate"][:]
        null_identity = _attrs(hf)
        construction = null_construction_report(hf)

    # Same positional guards as `concept_repro`. They are cheap and a join that
    # silently mis-pairs 18 of 20 seeds produces a complete, plausible, wrong
    # answer (gotcha 29).
    assert_labels_align("seed labels (AP vs null)", ap_seeds, null_seeds)
    pool_concepts = [c for c, k in zip(ap_concepts, in_pool) if k]
    assert_labels_align("concept labels (AP pool vs null)",
                        pool_concepts, null_concepts)
    axis = assert_shared_feature_axis(fire_rate, fire_counts, n_rows)
    identity_report = compare_identities(ap_identity, null_identity)

    ap_pool = ap[:, :, in_pool]
    prev_pool = prevalence[in_pool]
    npos_pool = n_positives[in_pool].astype(np.float64)
    best_ap = observed_best_ap(ap_pool)
    best_gain = to_gain(best_ap, prev_pool)

    n_seeds, n_shifts, _ = null_cmax.shape
    est_pooled = estimable_percentiles(n_seeds * n_shifts)
    est_per_seed = estimable_percentiles(n_shifts)
    log_size = np.log(npos_pool)

    # ---- the whole curve, both arms, EVERY estimable percentile -----------
    curve = {}
    for q in CURVE_PERCENTILES:
        entry: dict = {}
        for mode, est in (("pooled", est_pooled), ("per_seed", est_per_seed)):
            if not est[q]["estimable"]:
                entry[mode] = None
                entry[f"{mode}_estimable"] = False
                continue
            thr = null_thresholds(null_cmax, q, mode)
            det = detection_matrix(best_gain, thr)
            d = detection_rate(det)
            entry[mode] = {
                "distribution_of_D": distribution_summary(d),
                "conditioned_on_prevalence": conditioned_distribution(
                    d, npos_pool, n_strata),
                "saturation": saturation_threshold(d, npos_pool),
                "binned_curve": binned_curve(d, npos_pool),
                "spearman_D_vs_log_size": (
                    spearman(d, log_size) if len(np.unique(d)) > 1 else None),
            }
            entry[f"{mode}_estimable"] = True
        curve[str(q)] = entry

    # ---- the reference arm, and the margin diagnostic ---------------------
    ref_thr = null_thresholds(null_cmax, REFERENCE_PERCENTILE, "pooled")
    ref_det = detection_matrix(best_gain, ref_thr)
    d_ref = detection_rate(ref_det)
    margin = detection_margin(best_gain, ref_thr)

    experimental = experimental_counts_for(labels_path, pool_concepts)

    from src.sae.train import git_dirty, git_sha

    out = {
        "specification": "PREREGISTRATION amendment AY sections 2a, 2c, 3",
        # This artifact's OWN provenance, not only its inputs'. An artifact
        # that records what it consumed but not what produced it cannot be
        # tied back to a commit.
        "identity": {"git_sha": git_sha(), "git_dirty": git_dirty(),
                     "measurement": "concept_bridge_AY"},
        "source": {"ap": str(ap_path), "null": str(null_path),
                   "labels": str(labels_path), "stratum": stratum,
                   "ap_identity": ap_identity, "null_identity": null_identity,
                   "identity_comparison": identity_report},
        "positional_guards": {
            "seed_labels_elementwise": f"{len(ap_seeds)}/{len(ap_seeds)}",
            "concept_labels_elementwise":
                f"{len(pool_concepts)}/{len(pool_concepts)}",
            "shared_feature_axis": axis,
        },
        "section_2a_i_null_construction": {
            "read_from": "the artifact's own dataset shapes, not the filename",
            **construction,
            "sensitivity_of_the_headline_to_the_pooling_choice": {
                "note": "the per-seed arm removes the shared-THRESHOLD "
                        "channel; the shared-SCHEDULE channel is a property of "
                        "the stored draws and cannot be removed without "
                        "re-running null_pilot.py with a per-seed schedule",
                "pooled": distribution_summary(d_ref),
                "per_seed": distribution_summary(detection_rate(
                    detection_matrix(best_gain, null_thresholds(
                        null_cmax, REFERENCE_PERCENTILE, "per_seed")))),
            },
            "detection_margin_at_reference": {
                "definition": "observed gain / detection threshold, minimised "
                              "over the 20 anchor seeds",
                "always_detected_D_eq_1": margin_summary(
                    margin, d_ref, d_ref == 1.0),
                "interior": margin_summary(
                    margin, d_ref, (d_ref > 0.1) & (d_ref < 0.9)),
                "never_detected_D_eq_0": margin_summary(
                    margin, d_ref, d_ref == 0.0),
            },
        },
        "section_2a_ii_never_detected": never_detected_report(
            pool_concepts, d_ref, npos_pool,
            best_ap.mean(axis=0), best_ap.std(axis=0, ddof=1), experimental),
        "section_2c_and_3_curve": curve,
        "reference_percentile": REFERENCE_PERCENTILE,
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
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stratum", default="pooled")
    p.add_argument("--n-strata", type=int, default=3)
    args = p.parse_args()

    out = run(args.ap, args.null, args.labels, args.out,
              args.stratum, args.n_strata)

    c = out["section_2a_i_null_construction"]
    print(f"\n[2a-i] null construction: {c['verdict']}")
    print(f"       null_cmax_b {c['null_cmax_b_shape']}, "
          f"dest_schedule {c['dest_schedule']['shape']}")
    s = c["sensitivity_of_the_headline_to_the_pooling_choice"]
    for arm in ("pooled", "per_seed"):
        a = s[arm]
        print(f"       {arm:<9} D=1 {a['n_exactly_one']:>4} "
              f"({a['share_exactly_one']:.2%})   D=0 {a['n_exactly_zero']:>3} "
              f"({a['share_exactly_zero']:.2%})   interior "
              f"{a['interior_mass']:.2%}")
    m = c["detection_margin_at_reference"]["always_detected_D_eq_1"]
    if m.get("status") == "COMPUTED":
        q = m["min_over_seeds_of_gain_over_threshold"]
        print(f"       margin at the WEAKEST seed, D=1 concepts: "
              f"p0 {q['p0']:.2f}x  p50 {q['p50']:.2f}x  p100 {q['p100']:.2f}x")
        print(f"       share whose worst seed is within 2x of threshold: "
              f"{m['share_worst_seed_within_2x_of_threshold']:.2%}")

    nd = out["section_2a_ii_never_detected"]
    print(f"\n[2a-ii] never detected: {nd['n_never_detected']} concepts "
          f"({nd['share_of_pool']:.2%}); "
          f"{nd['n_with_experimental_evidence']} carry experimental residues")
    print(f"        feature keys: {nd['feature_key_composition']}")

    print("\n[2c/3] per percentile (pooled arm): interior mass by prevalence "
          "tercile, and the saturation threshold")
    for q, e in out["section_2c_and_3_curve"].items():
        arm = e.get("pooled")
        if arm is None:
            print(f"  p{q}: NOT ESTIMABLE -- dropped, not interpolated")
            continue
        cond = arm["conditioned_on_prevalence"]
        ims = [f"{st['distribution_of_D']['interior_mass']:.2%}"
               if st.get("distribution_of_D") else "n/a"
               for st in cond["strata"]]
        sat = arm["saturation"]
        thr = ("none" if sat["threshold"] is None
               else f"{sat['threshold']:,.0f}")
        print(f"  p{q}: pooled {cond['pooled_distribution_of_D']['interior_mass']:.2%}"
              f"  terciles {' / '.join(ims)}"
              f"  saturation {thr} ({sat['status']})"
              f"  rho {arm['spearman_D_vs_log_size']:.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
