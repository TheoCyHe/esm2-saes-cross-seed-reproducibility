"""Pre- and post-flight for amendment CT's new arms (§CT24 `k = 64`, §CT25's
widths, §CT26's seeds 20-39): the checks `feature_ap` / `null_pilot` do not
make themselves.

`feature_ap.py` and `null_pilot.py` take `k` and `dict_size` from `--config`,
NOT from the checkpoint. A `dict_size` mismatch fails at `load_state_dict`; a
`k` mismatch does not -- a k = 64 checkpoint run under configs/mvp.yaml would
encode with k = 32 and return a complete, wrong AP matrix. So:

  `ckpts`  every checkpoint's meta.json `k` and `dict_size` equal the config's,
           the count is the expected one, and the seeds are distinct;
  `axis`   an AP or null artifact's concept axis (names AND order) and argmax
           pool equal `feature_ap_main_compliant.h5`'s (§CT23-pre item 12).

Exit 1 on any disagreement; an absent key is a failure, never a match.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def check_ckpts(ckpts: list[Path], cfg: dict, expect: int) -> list[str]:
    bad = []
    if len(ckpts) != expect:
        bad.append(f"{len(ckpts)} checkpoints, expected {expect}")
    k, d = cfg["sae"]["k"], cfg["sae"]["dict_size"]
    seeds = []
    for c in ckpts:
        mp = Path(c) / "meta.json"
        if not mp.exists():
            bad.append(f"{mp} ABSENT")
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        for key, want in (("k", k), ("dict_size", d)):
            if key not in m:
                bad.append(f"{c}: meta.json has no {key!r} (ABSENT, not a match)")
            elif m[key] != want:
                bad.append(f"{c}: meta {key}={m[key]} but config {key}={want}")
        seeds.append(m.get("seed"))
    if len(set(seeds)) != len(seeds):
        bad.append(f"duplicate seeds {seeds}")
    return bad


def _dec(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def read_axis(path: Path, kind: str) -> tuple[list, list | None]:
    import h5py
    with h5py.File(path, "r") as hf:
        if kind == "ap":
            names = [_dec(c) for c in hf["pooled/concepts"][:]]
            pool = [bool(x) for x in hf["pooled/in_argmax_pool"][:]]
            return names, pool
        return [_dec(c) for c in hf["concepts"][:]], None


def check_axis(new: tuple, ref: tuple, kind: str) -> list[str]:
    names, pool = new
    rnames, rpool = ref
    if kind == "ap":
        bad = []
        if names != rnames:
            bad.append(f"concept axis differs ({len(names)} vs {len(rnames)})")
        if pool != rpool:
            bad.append("in_argmax_pool differs")
        return bad
    # a null artifact's axis is the argmax pool
    want = [c for c, k in zip(rnames, rpool) if k]
    return [] if names == want else [f"null concepts differ from the pool "
                                     f"({len(names)} vs {len(want)})"]


def compare_ap(new: dict, ref: dict, seed_label: str, bar: float) -> dict:
    """§CT26-F item 1's control: one seed's pooled AP from a fresh pass against
    the stored artifact's, concept axis first, then the maximum absolute
    difference against `bar` (CODEBASE §13.4: a recomputation is not bitwise,
    because --ap-device auto moves the arithmetic; <= 1e-6 is the record's bar)."""
    import numpy as np
    if new["concepts"] != ref["concepts"]:
        return {"verdict": "FAIL", "reason": "concept axes differ"}
    for d, name in ((new, "new"), (ref, "ref")):
        if seed_label not in d["seeds"]:
            return {"verdict": "FAIL", "reason": f"{seed_label} ABSENT from {name}"}
    a = np.asarray(new["ap"][new["seeds"].index(seed_label)], dtype=np.float64)
    r = np.asarray(ref["ap"][ref["seeds"].index(seed_label)], dtype=np.float64)
    if a.shape != r.shape:
        return {"verdict": "FAIL", "reason": f"shapes {a.shape} vs {r.shape}"}
    diff = float(np.max(np.abs(a - r)))
    return {"verdict": "PASS" if diff <= bar else "FAIL", "max_abs_diff": diff,
            "bar": bar, "n_values": int(a.size),
            "n_bitwise_equal": int((a == r).sum())}


def compare_gtilde(new: dict, ref: dict, rel_tol: float = 1e-6) -> dict:
    """The author's condition (a), 2026-10-24: `M` (and the argmax) of a rerun against
    the artifact of record. BITWISE -> proceed; max relative difference <=
    rel_tol (the tolerance of `jackknife_dictionaries.control_verdict`, the
    rule §CT10-R's control used) -> proceed, difference recorded; otherwise
    STOP. Seed order and concept axis must be equal first."""
    import numpy as np
    seeds_note = "compared"
    if new["seeds"] is None or ref["seeds"] is None:
        # gotcha 15: absent is reported, never a match. M is then compared by
        # position; a BITWISE match under a different seed order is not credible.
        seeds_note = "ABSENT on " + ("new" if new["seeds"] is None else "reference")
    elif new["seeds"] != ref["seeds"]:
        return {"verdict": "STOP", "reason": "seeds differ"}
    if new["concepts"] != ref["concepts"]:
        return {"verdict": "STOP", "reason": "concepts differ"}
    a, r = np.asarray(new["M"], np.float64), np.asarray(ref["M"], np.float64)
    if a.shape != r.shape:
        return {"verdict": "STOP", "reason": f"M shapes {a.shape} vs {r.shape}"}
    same = (a == r) | (np.isnan(a) & np.isnan(r))
    am_same = bool(np.array_equal(new["argmax"], ref["argmax"]))
    if bool(same.all()) and am_same:
        return {"verdict": "BITWISE", "n_values": int(a.size), "argmax_equal": True,
                "seed_axis": seeds_note}
    rel = float(np.nanmax(np.abs(a - r) / np.maximum(np.abs(r), 1e-300)))
    return {"verdict": "WITHIN_BAR" if (rel <= rel_tol and am_same) else "STOP",
            "max_rel_diff": rel, "max_abs_diff": float(np.nanmax(np.abs(a - r))),
            "n_values_differing": int((~same).sum()), "n_values": int(a.size),
            "argmax_equal": am_same, "n_argmax_differing": int(
                (np.asarray(new["argmax"]) != np.asarray(ref["argmax"])).sum()),
            "rel_tol": rel_tol, "seed_axis": seeds_note}


def _read_gtilde(path: Path) -> dict:
    import h5py
    with h5py.File(path, "r") as hf:
        return {"seeds": [_dec(s) for s in hf["seeds"][:]] if "seeds" in hf else None,
                "concepts": [_dec(c) for c in hf["concepts"][:]],
                "M": hf["M"][:], "argmax": hf["argmax_concept"][:]}


def _read_ap(path: Path) -> dict:
    import h5py
    with h5py.File(path, "r") as hf:
        return {"seeds": [_dec(s) for s in hf["seeds"][:]],
                "concepts": [_dec(c) for c in hf["pooled/concepts"][:]],
                "ap": hf["pooled/ap"][:]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ckpts")
    a.add_argument("--config", type=Path, required=True)
    a.add_argument("--expect", type=int, required=True)
    a.add_argument("--ckpts", type=Path, nargs="+", required=True)
    b = sub.add_parser("axis")
    b.add_argument("--artifact", type=Path, required=True)
    b.add_argument("--kind", choices=("ap", "null"), required=True)
    b.add_argument("--reference", type=Path, required=True,
                   help="feature_ap_main_compliant.h5")
    c = sub.add_parser("apcompare")
    c.add_argument("--new", type=Path, required=True)
    c.add_argument("--reference", type=Path, required=True)
    c.add_argument("--seed-label", required=True)
    c.add_argument("--bar", type=float, default=1e-6)
    c.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("gtcompare")
    g.add_argument("--new", type=Path, required=True)
    g.add_argument("--reference", type=Path, required=True)
    g.add_argument("--rel-tol", type=float, default=1e-6)
    g.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.cmd == "gtcompare":
        res = compare_gtilde(_read_gtilde(args.new), _read_gtilde(args.reference), args.rel_tol)
        res.update({"new": str(args.new), "reference": str(args.reference)})
        args.out.write_text(json.dumps(res, indent=1) + chr(10), encoding="utf-8")
        print(json.dumps(res))
        return 0 if res["verdict"] in ("BITWISE", "WITHIN_BAR") else 1
    if args.cmd == "apcompare":
        res = compare_ap(_read_ap(args.new), _read_ap(args.reference), args.seed_label, args.bar)
        res.update({"new": str(args.new), "reference": str(args.reference),
                    "seed_label": args.seed_label})
        args.out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(res))
        return 0 if res["verdict"] == "PASS" else 1
    if args.cmd == "ckpts":
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        bad = check_ckpts(args.ckpts, cfg, args.expect)
        ok_msg = (f"{len(args.ckpts)} checkpoints agree with {args.config}: "
                  f"k={cfg['sae']['k']} dict_size={cfg['sae']['dict_size']}")
    else:
        bad = check_axis(read_axis(args.artifact, args.kind),
                         read_axis(args.reference, "ap"), args.kind)
        ok_msg = f"{args.artifact}: concept axis equals the compliant artifact's"
    if bad:
        print("PREFLIGHT FAIL: " + "; ".join(bad), file=sys.stderr)
        return 1
    print("PREFLIGHT OK: " + ok_msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
