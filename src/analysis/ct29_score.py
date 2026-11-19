"""PREREGISTRATION amendment CT, §CT29 -- the k x width grid, scored.

CT29-P1..P4 by §CT29-F item 10's rules, derived from the stored outputs
(gotcha 20), with item 7's UNMEASURABLE causes applied per arm and the
detection and R3 tier (§CT25's computation) reported beside, unscored.

Per grid arm, in --ct-dir (`$PROJECT/analysis/ct`):
  fvu_eval_<arm>_ct.json         raw FVU per seed (divergence, P4)
  rf_marginal_m1_<arm>_ct.json   mean r_f at 0.70, n_compare = 9 (P1)
  gtilde_<arm>_b_ratio_ct.h5     its exemption keys (§CT29-D item 3); its
                                 existence = standardize_gain completed (it
                                 refuses on §CT5.1's cause 3, mu_s <= 0)
  ct28_arm_<arm>_ct.json         slope at θ*_arm, own SD (P2, P3), with its
                                 control against slope_<arm>_ct.json
  feature_ap_<arm>_ct.h5, null_pilot_<arm>_ct.h5   detection and R3
The k = 32 and k = 64 sides are §CT28's: ct28_arm_{w2560,w10240,k64n9}_ct.json,
fvu_eval_k64_ct.json (seeds 0-9), and the CS tier's fvu_eval_w10240.json.

Every input must exist: an absent file is "not run", and the scorer refuses.
Item 7, per arm:
  declared    a cause established outside these files -- arm (ii)'s memory
              outcome (§CT29-D item 4), a degenerate null that stopped
              standardize_gain or the fit (§CT5.1's causes, read from the
              job, as §CT24-R2 read them) -- named with --unmeasurable;
  divergence  raw FVU above the arm's declared threshold: the seed is excluded,
              counted and reported (EXCLUDE= on the passes); the scorer refuses
              if an excluded seed is in the fit;
  < 8 seeds   fewer than 8 seeds kept.
A grid arm's θ* figure (P2, P3) also needs its control BIT_IDENTICAL and at
least 8 anchor fits scoring; its absence removes the arm from P2/P3 only.

Scoring (§CT29-F item 10, total):
  P1  MISS if a measured arm (i)/(ii) has mean r_f(0.70) at or above its
      comparator; HIT if both measured and strictly below; else UNMEASURABLE.
  P2  k = 16, 32, 64, 128 own-SD fixed effects at θ*: HIT if strictly monotone,
      MISS if not; UNMEASURABLE if any of the four is.
  P3  closed fixed-effect intervals at θ*, (iv) vs k = 32 @ 2,560 and (iii) vs
      (ii): MISS if a measured pair shares no point; HIT if both share; else
      UNMEASURABLE.
  P4  mean raw FVU at 10,240, k = 16 > 32 > 64 > 128 strictly: HIT, else MISS;
      UNMEASURABLE if (iii) or (iv) is. Each mean is over the arm's kept seeds
      among 0-9 (the session's reading: a diverged seed is excluded from its
      arm, item 7); the all-ten mean is reported beside.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

GRID = ("k64_w2560", "k64_w40960", "k16_w10240", "k128_w10240")
# §CT29-F item 7: twice the k = 32 raw median at each arm's width
DIVERGENCE = {"k64_w2560": 2 * 0.46186782416268757,
              "k64_w40960": 2 * 0.2843083899098555,
              "k16_w10240": 0.7272, "k128_w10240": 0.7272}
# §CT29-F item 10 / CT29-pre item 9: k = 32 rates at 0.70, n_compare = 9
P1_COMPARATOR = {"k64_w2560": 0.6015277777777777, "k64_w40960": 0.3365565321180556}
SEEDS = tuple(range(10))
MIN_SCORING = 8


def intervals_share_point(a, b) -> bool:
    """Closed intervals."""
    return max(a[0], b[0]) <= min(a[1], b[1])


def strictly_monotone(v) -> bool:
    d = np.diff(np.asarray(v, dtype=np.float64))
    return bool((d > 0).all() or (d < 0).all())


def score_p1(rates: dict) -> str:
    """rates: arm -> mean r_f at 0.70, or None if the arm is unmeasurable."""
    meas = {a: r for a, r in rates.items() if r is not None}
    if any(r >= P1_COMPARATOR[a] for a, r in meas.items()):
        return "MISS"
    return "HIT" if len(meas) == len(P1_COMPARATOR) else "UNMEASURABLE"


def score_p2(est: list) -> str:
    """est: the four θ* fixed effects in the order k = 16, 32, 64, 128."""
    if any(e is None for e in est):
        return "UNMEASURABLE"
    return "HIT" if strictly_monotone(est) else "MISS"


def score_p3(pairs: list) -> str:
    """pairs: [(interval or None, interval or None), ...]."""
    shares = [None if a is None or b is None else intervals_share_point(a, b)
              for a, b in pairs]
    if any(s is False for s in shares):
        return "MISS"
    return "HIT" if all(s is True for s in shares) else "UNMEASURABLE"


def score_p4(fvu: list) -> str:
    """fvu: the four mean raw FVUs in the order k = 16, 32, 64, 128; the k = 32
    and k = 64 sides are stored, so only (iii) and (iv) can be None."""
    if any(f is None for f in fvu):
        return "UNMEASURABLE"
    return "HIT" if bool((np.diff(np.asarray(fvu, dtype=np.float64)) < 0).all()) else "MISS"


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fvu_by_seed(f: dict) -> dict:
    return {int(s): float(v["raw"]["fvu"]) for s, v in f["per_seed"].items()}


def theta_star_fit(arm: dict | None, side: bool) -> tuple[dict | None, str | None]:
    """The own-SD fit at θ*, or None and why. A grid arm's control must be
    BIT_IDENTICAL against its own 0.70 slope and at least 8 anchor fits must
    score (§CT29-F item 7); §CT28's sides keep §CT28-F items 1 and 3 (control
    BIT_IDENTICAL or NO_REFERENCE -- `k64n9` has none -- and 0 failed fits).
    A missing figure removes the arm from P2/P3 only, not from P1/P4."""
    if arm is None:
        return None, "absent"
    ok = ("BIT_IDENTICAL", "NO_REFERENCE") if side else ("BIT_IDENTICAL",)
    v = arm["control_0.70_vs_record"]["verdict"]
    if v not in ok:
        return None, f"control {v}"
    f = arm.get("at_theta_star") or {}
    if "own" not in f:
        return None, f"no θ* fit ({f.get('cause', 'absent')})"
    failed = f.get("n_anchor_fits_failed")
    if failed is None:
        return None, "no failure count"
    if side and failed != 0:
        return None, f"{failed} anchor fits failed"
    if f.get("n_anchors", 0) - failed < MIN_SCORING:
        return None, f"{f.get('n_anchors', 0) - failed} anchor fits scoring (< {MIN_SCORING})"
    return f, None


def required_inputs(ct: Path, arm: str) -> list[Path]:
    return [ct / f"fvu_eval_{arm}_ct.json", ct / f"rf_marginal_m1_{arm}_ct.json",
            ct / f"gtilde_{arm}_b_ratio_ct.h5", ct / f"ct28_arm_{arm}_ct.json",
            ct / f"feature_ap_{arm}_ct.h5", ct / f"null_pilot_{arm}_ct.h5"]


def side_inputs(ct: Path, cs: Path) -> list[Path]:
    return [ct / "ct28_arm_w2560_ct.json", ct / "ct28_arm_w10240_ct.json",
            ct / "ct28_arm_k64n9_ct.json", ct / "fvu_eval_k64_ct.json",
            cs / "fvu_eval_w10240.json"]


def arm_status(ct: Path, arm: str, external: dict) -> dict:
    """Item 7's causes. An arm named with --unmeasurable is UNMEASURABLE by
    that cause and its files are not required; otherwise every input must
    exist (absence is "not run", which the scorer refuses, never UNMEASURABLE)."""
    if arm in external:
        return {"status": "UNMEASURABLE", "causes": [f"declared: {external[arm]}"],
                "kept_seeds": []}
    rec: dict = {"causes": []}
    by = fvu_by_seed(_load(ct / f"fvu_eval_{arm}_ct.json"))
    missing = [s for s in SEEDS if s not in by]
    if missing:
        raise SystemExit(f"FATAL: {arm}: FVU lacks seeds {missing}")
    div = sorted(s for s in SEEDS if by[s] > DIVERGENCE[arm])
    rec["fvu_raw_by_seed"] = {str(s): by[s] for s in SEEDS}
    rec["divergence_threshold"] = DIVERGENCE[arm]
    rec["diverged_seeds"] = div
    rec["kept_seeds"] = [s for s in SEEDS if s not in div]
    if len(rec["kept_seeds"]) < MIN_SCORING:
        rec["causes"].append(f"{len(rec['kept_seeds'])} seeds kept (< {MIN_SCORING})")
    rec["status"] = "UNMEASURABLE" if rec["causes"] else "MEASURED"
    return rec


def _attrs(path: Path, keys) -> dict:
    import h5py
    with h5py.File(path, "r") as hf:
        out = {}
        for k in keys:
            v = hf.attrs.get(k)
            out[k] = (v.decode() if isinstance(v, bytes) else
                      v.item() if hasattr(v, "item") else v)
        return out


def exemption_and_devices(ct: Path, arm: str) -> dict:
    """§CT29-D items 3 and 4: the split each arm's gtilde accepted, and the
    gather device class of its AP and null (stated in every comparison
    involving an arm whose class differs, §CT29-D item 4's fallback)."""
    return {"gtilde": _attrs(ct / f"gtilde_{arm}_b_ratio_ct.h5",
                             ("accept_ap_arith_split", "ap_arith_splits_accepted")),
            "feature_ap": _attrs(ct / f"feature_ap_{arm}_ct.h5",
                                 ("gather_device_class", "ap_arith_device_class")),
            "null_pilot": _attrs(ct / f"null_pilot_{arm}_ct.h5",
                                 ("gather_device_class", "ap_arith_device_class"))}


def score_all(ct: Path, cs: Path, external: dict, detection: bool = True,
              devices: bool = True) -> dict:
    need = [p for p in side_inputs(ct, cs) if not p.exists()]
    for a in GRID:
        if a not in external:
            need += [p for p in required_inputs(ct, a) if not p.exists()]
    if need:
        raise SystemExit("FATAL: not run yet (absent inputs; a declared cause is named with "
                         f"--unmeasurable ARM=cause): {[str(p) for p in need]}")
    st = {a: arm_status(ct, a, external) for a in GRID}
    grid_fit, why_fit = {}, {}
    for a in GRID:
        if st[a]["status"] != "MEASURED":
            grid_fit[a], why_fit[a] = None, "arm UNMEASURABLE"
            continue
        arm = _load(ct / f"ct28_arm_{a}_ct.json")
        fitted = [str(s) for s in arm["identity"]["comparison_set"]]
        bad = [s for s in st[a]["diverged_seeds"] if str(s) in fitted]
        if bad:
            raise SystemExit(f"FATAL: {a}: diverged seeds {bad} are in the fit")
        grid_fit[a], why_fit[a] = theta_star_fit(arm, side=False)
    side, why_side = {}, {}
    for a in ("w2560", "w10240", "k64n9"):
        side[a], why_side[a] = theta_star_fit(_load(ct / f"ct28_arm_{a}_ct.json"), side=True)
    out: dict = {"arm_status": st,
                 "theta_star_figure_missing_because": {
                     **{a: w for a, w in why_fit.items() if w},
                     **{a: w for a, w in why_side.items() if w}}}
    if devices:
        out["exemption_and_devices"] = {a: exemption_and_devices(ct, a)
                                        for a in GRID if a not in external}

    rates, why = {}, {}
    for a in P1_COMPARATOR:
        rates[a] = None
        if st[a]["status"] != "MEASURED":
            why[a] = "arm UNMEASURABLE"
            continue
        m1 = _load(ct / f"rf_marginal_m1_{a}_ct.json")
        if m1["identity"].get("n_compare_seeds") != 9:
            # P1 is declared at n_compare = 9; an excluded seed changes it (item 7)
            why[a] = f"n_compare_seeds {m1['identity'].get('n_compare_seeds')}, not 9"
        else:
            rates[a] = float(m1["marginals_by_theta"]["0.70"]["mnn"]["mean"])
    out["CT29-P1"] = {"verdict": score_p1(rates), "mean_r_f_0.70": rates,
                      "unmeasured_because": why, "comparators_k32_n9": P1_COMPARATOR}

    order = (("k=16", grid_fit["k16_w10240"]), ("k=32", side["w10240"]),
             ("k=64", side["k64n9"]), ("k=128", grid_fit["k128_w10240"]))
    est = [f["own"]["fe_pp"] if f is not None else None for _, f in order]
    out["CT29-P2"] = {"verdict": score_p2(est),
                      "theta_star_own_fe_pp": {k: e for (k, _), e in zip(order, est)}}

    iv = lambda f: f["own"]["fe_ci95_pp"] if f is not None else None
    pairs = [(iv(grid_fit["k128_w10240"]), iv(side["w2560"])),
             (iv(grid_fit["k16_w10240"]), iv(grid_fit["k64_w40960"]))]
    out["CT29-P3"] = {"verdict": score_p3(pairs),
                      "pair_one_k128_w10240_vs_k32_w2560": pairs[0],
                      "pair_two_k16_w10240_vs_k64_w40960": pairs[1],
                      "reported_beside_unscored": {
                          a: ({"widest": f["own"].get("widest"),
                               "shared_fe_ci95_pp": (f.get("shared") or {}).get("fe_ci95_pp")}
                              if f is not None else None)
                          for a, f in (("k128_w10240", grid_fit["k128_w10240"]),
                                       ("k32_w2560", side["w2560"]),
                                       ("k16_w10240", grid_fit["k16_w10240"]),
                                       ("k64_w40960", grid_fit["k64_w40960"]))}}

    def grid_mean(a):
        if st[a]["status"] != "MEASURED":
            return None, None
        by = {int(s): v for s, v in st[a]["fvu_raw_by_seed"].items()}
        return (float(np.mean([by[s] for s in st[a]["kept_seeds"]])),
                float(np.mean([by[s] for s in SEEDS])))
    k32 = fvu_by_seed(_load(cs / "fvu_eval_w10240.json"))
    k64 = fvu_by_seed(_load(ct / "fvu_eval_k64_ct.json"))
    m16, a16 = grid_mean("k16_w10240")
    m128, a128 = grid_mean("k128_w10240")
    seq = [m16, float(np.mean([k32[s] for s in SEEDS])),
           float(np.mean([k64[s] for s in SEEDS])), m128]
    out["CT29-P4"] = {"verdict": score_p4(seq),
                      "mean_raw_fvu_k16_k32_k64_k128": seq,
                      "all_ten_means_k16_k128": [a16, a128]}

    if detection:
        from src.analysis.ct25_width_tier import detection_counts
        out["detection_and_R3_unscored"] = {
            a: (detection_counts(ct / f"feature_ap_{a}_ct.h5", ct / f"null_pilot_{a}_ct.h5")
                if a not in external else None) for a in GRID}
    return out


def input_hashes(ct: Path, cs: Path, external: dict) -> dict:
    from src.analysis.jackknife_dictionaries import _sha16
    paths = side_inputs(ct, cs) + [p for a in GRID if a not in external
                                   for p in required_inputs(ct, a)]
    return {str(p): _sha16(p) for p in paths}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ct-dir", type=Path, required=True, help="$PROJECT/analysis/ct")
    p.add_argument("--cs-dir", type=Path, required=True, help="$PROJECT/analysis/cs")
    p.add_argument("--unmeasurable", nargs="*", default=[],
                   help="ARM=cause, a cause established outside these files")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"FATAL: {a.out} exists")
    external = {}
    for item in a.unmeasurable:
        arm, _, cause = item.partition("=")
        if arm not in GRID or not cause:
            raise SystemExit(f"FATAL: --unmeasurable takes ARM=cause, ARM in {GRID}")
        external[arm] = cause
    from src.sae.train import git_dirty, git_sha
    scored = score_all(a.ct_dir, a.cs_dir, external)   # refuses on an absent input first
    out = {"identity": {"measurement": "ct29_score", "governing_amendment": "CT29-F item 10",
                        "ct_dir": str(a.ct_dir), "cs_dir": str(a.cs_dir),
                        "unmeasurable_declared": external,
                        "inputs_sha256_16": input_hashes(a.ct_dir, a.cs_dir, external),
                        "git_sha": git_sha(), "git_dirty": git_dirty()},
           **scored}
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v["verdict"] for k, v in out.items() if k.startswith("CT29")}),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
