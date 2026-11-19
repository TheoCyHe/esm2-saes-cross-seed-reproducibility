"""PREREGISTRATION amendment CT, §CT25 -- the width null tier, scored.

Inputs per width: the arm's AP and null (detection, §AW1.2 as measured at
§AX12.2: best-feature gain above the pooled null's 95th percentile, strict,
pooled over that width's ten seeds) and its slope file
(`jackknife_dictionaries fit --thetas 0.70`, design 3 at theta = 0.70,
n_compare = 9). A width whose files are absent, or whose fit has fewer than 8
anchors scoring, is UNMEASURABLE and named.

Scoring (§CT25-F item 5, total):
  P1  MISS if any measurable width's fixed-effect estimate is outside
      [-0.5, +1.0] pp per SD; else HIT if all four are measurable; else
      UNMEASURABLE.
  R1  HIT as soon as the measurable widths, in width order, are not strictly
      monotone; else (all four measurable) MISS if strictly monotone; else
      UNMEASURABLE.
  §CS1's band classification of the slope is reported beside R1, unscored:
  s = sqrt(mean over widths of the across-anchor variance), each variance
  recovered from the stored across-seed interval (half-width / 1.959964, times
  sqrt(n_anchors), squared).
  R3 = the fraction of the 703 pool concepts detected in all ten seeds, per
  width; §CS1 has no per-unit values for it, so its class is NOT_APPLICABLE
  (§CT23-pre item 10).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

WIDTHS = (2560, 10240, 20480, 40960)
BAND = (-0.5, 1.0)
Z = 1.959964
MIN_SCORING = 8


def strictly_monotone(v: list[float]) -> bool:
    d = np.diff(np.asarray(v, dtype=np.float64))
    return bool((d > 0).all() or (d < 0).all())


def score_p1(est: dict) -> str:
    """est: width -> fixed-effect estimate, or None if unmeasurable."""
    meas = {w: e for w, e in est.items() if e is not None}
    if any(not (BAND[0] <= e <= BAND[1]) for e in meas.values()):
        return "MISS"
    return "HIT" if len(meas) == len(WIDTHS) else "UNMEASURABLE"


def score_r1(est: dict) -> str:
    seq = [est[w] for w in WIDTHS if est.get(w) is not None]
    if len(seq) >= 2 and not strictly_monotone(seq):
        return "HIT"          # already not monotone, whatever the missing width
    if len(seq) == len(WIDTHS):
        return "MISS" if strictly_monotone(seq) else "HIT"
    return "UNMEASURABLE"


def cs1_class(means: list[float], sds: list[float]) -> dict:
    """§CS1, verbatim rule: s = sqrt(mean within-width variance); Delta_i the
    successive differences; MONOTONE down/up if every Delta beyond -/+2s, FLAT if
    every |Delta| <= 2s, NON-MONOTONE otherwise."""
    s = math.sqrt(float(np.mean(np.square(sds))))
    d = np.diff(np.asarray(means, dtype=np.float64))
    if (d < -2 * s).all():
        c = "MONOTONE_DOWN"
    elif (d > 2 * s).all():
        c = "MONOTONE_UP"
    elif (np.abs(d) <= 2 * s).all():
        c = "FLAT"
    else:
        c = "NON_MONOTONE"
    return {"s": s, "deltas": d.tolist(), "class": c}


def sd_from_across_seed(ci: list[float], n: int) -> float:
    return (ci[1] - ci[0]) / (2 * Z) * math.sqrt(n)


def detection_counts(ap_path: Path, null_path: Path) -> dict:
    import h5py
    from src.analysis.concept_repro import (REFERENCE_PERCENTILE, detection_matrix,
                                            null_thresholds, observed_best_ap, to_gain)
    with h5py.File(ap_path, "r") as hf:
        seeds = [s.decode() if isinstance(s, bytes) else str(s) for s in hf["seeds"][:]]
        pool = hf["pooled/in_argmax_pool"][:].astype(bool)
        ap = hf["pooled/ap"][:][:, :, pool]
        prev = hf["pooled/prevalence"][:][pool]
    with h5py.File(null_path, "r") as hf:
        nseeds = [s.decode() if isinstance(s, bytes) else str(s) for s in hf["seeds"][:]]
        cmax = hf["null_cmax_b"][:]
    if seeds != nseeds:
        raise SystemExit(f"FATAL: AP and null seed orders differ in {ap_path}")
    det = detection_matrix(to_gain(observed_best_ap(ap), prev),
                           null_thresholds(cmax, REFERENCE_PERCENTILE, "pooled"))
    n_seeds = det.shape[0]
    return {"seed_labels": seeds,
            "detected_per_seed": {s: int(det[i].sum()) for i, s in enumerate(seeds)},
            "n_pool": int(det.shape[1]),
            "n_detected_in_all_seeds": int((det.sum(axis=0) == n_seeds).sum()),
            "R3_fraction_detected_in_all_seeds": float((det.sum(axis=0) == n_seeds).mean()),
            "histogram_seeds_detected": np.bincount(det.sum(axis=0),
                                                    minlength=n_seeds + 1).tolist()}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis-dir", type=Path, required=True,
                   help="$PROJECT/analysis/ct")
    p.add_argument("--memory-note", default=None,
                   help="if 40,960 failed on memory: the peak, recorded verbatim")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"FATAL: {a.out} exists")
    per, est = {}, {}
    for w in WIDTHS:
        tag = f"w{w}"
        ap, nl = a.analysis_dir / f"feature_ap_{tag}_ct.h5", a.analysis_dir / f"null_pilot_{tag}_ct.h5"
        sl = a.analysis_dir / f"slope_{tag}_ct.json"
        rec = {"files": {k: str(v) for k, v in (("ap", ap), ("null", nl), ("slope", sl))}}
        missing = [str(f) for f in (ap, nl, sl) if not f.exists()]
        if missing:
            rec["status"] = "UNMEASURABLE"
            rec["cause"] = f"absent: {missing}"
            est[w] = None
            per[tag] = rec
            continue
        rec["detection"] = detection_counts(ap, nl)
        sj = json.loads(sl.read_text(encoding="utf-8"))
        f = sj["full_fit"]["0.70"]
        n_ok = f["n_anchors"] - f["n_anchor_fits_failed"]
        rec["slope_0.70"] = {
            "fixed_effect_pp": f["fe_pp"], "fixed_effect_ci95_pp": f["fe_ci95_pp"],
            "across_seed_mean_pp": f["across_seed_mean_pp"],
            "across_seed_ci95_pp": f["across_seed_ci95_pp"],
            "n_anchors": f["n_anchors"], "n_anchor_fits_failed": f["n_anchor_fits_failed"],
            "thetas_qualifying_x2": sj["identity"].get("thetas_qualifying_x2"),
            "across_anchor_sd_pp": sd_from_across_seed(f["across_seed_ci95_pp"], f["n_anchors"])}
        if n_ok < MIN_SCORING:
            rec["status"] = "UNMEASURABLE"
            rec["cause"] = f"{n_ok} of {f['n_anchors']} anchors scoring (< {MIN_SCORING})"
            est[w] = None
        else:
            rec["status"] = "MEASURED"
            est[w] = f["fe_pp"]
        per[tag] = rec
    out = {"specification": "PREREGISTRATION amendment CT, §CT25", "per_width": per,
           "CT25_P1": {"band_pp_per_SD": list(BAND),
                       "estimates": {f"w{w}": est[w] for w in WIDTHS},
                       "outcome": score_p1(est)},
           "CT25_R1": {"rule": "not strictly monotone in width (point estimates)",
                       "outcome": score_r1(est)},
           "memory_note_40960": a.memory_note}
    if all(est[w] is not None for w in WIDTHS):
        out["CS1_band_classification_of_slope_unscored"] = cs1_class(
            [est[w] for w in WIDTHS],
            [per[f"w{w}"]["slope_0.70"]["across_anchor_sd_pp"] for w in WIDTHS])
    out["R3"] = {f"w{w}": (per[f"w{w}"]["detection"]["R3_fraction_detected_in_all_seeds"]
                           if "detection" in per[f"w{w}"] else None) for w in WIDTHS}
    out["R3_CS1_class"] = "NOT_APPLICABLE: R3 has no per-seed values (§CT23-pre item 10)"
    from src.sae.train import git_dirty, git_sha
    out["identity"] = {"git_sha": git_sha(), "git_dirty": git_dirty()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("CT25_P1", "CT25_R1", "R3")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
