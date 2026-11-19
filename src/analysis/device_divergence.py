"""How much does the device-dependent activation gather actually move the
average-precision statistic?

WHY A DIFF AND NOT A HASH. `gather_fingerprint.py` already established that the
gather is not device-independent: nnz is identical (TopK takes exactly k per row
either way) but `sha256_feat`, `sha256_pos`, `sha256_seg_start` and
`sha256_fire_counts` all DIFFER between CPU and `rtx_5000_ada`. A sha256 is a
maximally sensitive detector, so `DIFFER` is equally consistent with one flipped
entry of 511,114,400 and with a pervasive change. **Deriving a population claim
from a binary detector is the error this project keeps re-committing.**

So this module diffs the two AP artifacts instead, which measures divergence on
the REDUCTION THE CONSUMER ACTUALLY TAKES rather than on the gather itself
(gotcha 21: a statistic measured on one reduction licenses no claim about a
different one, and this project's predictor is an extremum almost everywhere).

WHAT IT REPORTS, in order of decision-relevance:

  1. **Gain-floor flips.** Per (seed, feature), `g_f` is the MAX over in-pool
     concepts of `G = (AP - prevalence) / (1 - prevalence)`. A flip is a pair
     that lands on opposite sides of the pre-registered `G >= 0.05` floor. This
     is the only quantity here that changes an outcome.
  2. **Argmax changes.** Which concept WINS that max, per seed and pooled. The
     discrete quantity, and the one a rank statistic would hide.
  3. **AP divergence** over the full (seed, feature, concept) array: how many
     elements differ, and the DISTRIBUTION of |difference|.
  4. **`fire_counts` divergence** over (seed, feature).

**DISTRIBUTIONS, NOT SUMMARY STATISTICS ALONE.** Two recent measurements in this
project were spikes at zero with thin heavy tails, where a mean was actively
misleading. Every divergence here is reported with percentiles over the DIFFERING
subset as well as over everything.

WHAT THIS DOES NOT LICENSE. Nothing about the seed-reproducibility headline. That
matching rule is decoder cosine, a property of stored parameters; no gather
touches it.

SEED ALIGNMENT IS ON LABELS, NEVER POSITION (gotcha 29): two orders of the seed
axis exist in this repo and only rows 0 and 1 coincide.

PREVALENCE IS AN INVARIANT AND IS ASSERTED AS ONE. `prevalence` and
`n_rows_effective` derive from the label store and the row population, not from
activations, so they MUST be identical across device classes. If they are not,
the two artifacts differ in something other than the gather and the comparison is
not the one this module claims to make -- so it refuses rather than reporting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

# The pre-registered gain floor. Written down here rather than imported from a
# consumer, so a change to a consumer cannot silently redefine what a "flip" is.
GAIN_FLOOR = 0.05

PERCENTILES = [0, 1, 5, 25, 50, 75, 90, 95, 99, 99.9, 100]


def read_ap(path: Path, stratum: str) -> dict:
    with h5py.File(path, "r") as hf:
        g = hf[stratum]
        return {
            "ap": g["ap"][:].astype(np.float64),
            "prevalence": g["prevalence"][:].astype(np.float64),
            "in_pool": g["in_argmax_pool"][:].astype(bool),
            "concepts": [c.decode() if isinstance(c, bytes) else str(c)
                         for c in g["concepts"][:]],
            "fire_counts": hf["fire_counts"][:].astype(np.int64),
            "seeds": [s.decode() if isinstance(s, bytes) else str(s)
                      for s in hf["seeds"][:]],
            "attrs": {k: hf.attrs[k] for k in hf.attrs},
        }


def align_rows(a_seeds: list[str], b_seeds: list[str]) -> np.ndarray:
    """Row permutation taking B's seed order to A's. Raises if the label SETS
    differ -- a positional fallback here is exactly gotcha 29's defect."""
    if sorted(a_seeds) != sorted(b_seeds):
        raise SystemExit(
            f"FATAL: seed label sets differ.\n  A: {sorted(a_seeds)}\n  B: {sorted(b_seeds)}")
    pos = {s: i for i, s in enumerate(b_seeds)}
    return np.array([pos[s] for s in a_seeds], dtype=np.int64)


def gain(ap: np.ndarray, prevalence: np.ndarray) -> np.ndarray:
    """G = (AP - p) / (1 - p), broadcast over (seed, feature, concept)."""
    return (ap - prevalence[None, None, :]) / (1.0 - prevalence[None, None, :])


def pooled_max(gain_arr: np.ndarray, in_pool: np.ndarray):
    """(g_f, argmax concept index) over the in-pool concepts only.

    Returns the argmax as an index into the FULL concept axis, not into the
    pooled subset, so the two artifacts' winners are directly comparable."""
    pool_idx = np.flatnonzero(in_pool)
    sub = gain_arr[:, :, pool_idx]
    amax = sub.argmax(axis=2)
    return sub.max(axis=2), pool_idx[amax]


def distribution(values: np.ndarray) -> dict:
    """Percentiles plus the counts a mean alone would hide."""
    if values.size == 0:
        return {"status": "EMPTY", "n": 0}
    finite = values[np.isfinite(values)]
    return {
        "status": "COMPUTED",
        "n": int(values.size),
        "n_nonfinite": int(values.size - finite.size),
        "n_exactly_zero": int((finite == 0).sum()),
        "frac_exactly_zero": float((finite == 0).mean()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "percentiles": {str(p): float(np.percentile(finite, p))
                        for p in PERCENTILES} if finite.size else None,
    }


def compare(a: dict, b: dict) -> dict:
    """A is the reference (GPU); B is re-ordered to A's seed labels."""
    perm = align_rows(a["seeds"], b["seeds"])
    b_ap = b["ap"][perm]
    b_fire = b["fire_counts"][perm]

    if a["concepts"] != b["concepts"]:
        raise SystemExit("FATAL: concept axes differ; this is not a device diff.")
    if not np.array_equal(a["in_pool"], b["in_pool"]):
        raise SystemExit("FATAL: argmax pools differ; this is not a device diff.")

    # THE INVARIANT. Prevalence cannot depend on the gather. If it moved, the
    # two artifacts differ in the label store or the row population and every
    # number below would be attributed to the wrong cause.
    prev_delta = np.abs(a["prevalence"] - b["prevalence"])
    if prev_delta.max() != 0.0:
        raise SystemExit(
            "FATAL: prevalence differs between the two artifacts "
            f"(max |delta| = {prev_delta.max():.3e}). Prevalence derives from the "
            "label store and the row population, never from activations, so this "
            "is not a pure device comparison. Refusing to report a divergence "
            "that would be attributed to the gather.")

    g_a = gain(a["ap"], a["prevalence"])
    g_b = gain(b_ap, b["prevalence"])
    gf_a, arg_a = pooled_max(g_a, a["in_pool"])
    gf_b, arg_b = pooled_max(g_b, b["in_pool"])

    # ---- 1. gain-floor flips -------------------------------------------
    pass_a, pass_b = gf_a >= GAIN_FLOOR, gf_b >= GAIN_FLOOR
    flips = pass_a != pass_b
    n_seeds, n_feats = gf_a.shape
    floor = {
        "gain_floor": GAIN_FLOOR,
        "n_pairs": int(gf_a.size),
        "n_pass_reference": int(pass_a.sum()),
        "n_pass_other": int(pass_b.sum()),
        "n_flipped": int(flips.sum()),
        "frac_flipped": float(flips.mean()),
        "n_gained": int((~pass_a & pass_b).sum()),
        "n_lost": int((pass_a & ~pass_b).sum()),
        "per_seed_flips": [int(x) for x in flips.sum(axis=1)],
        "gf_abs_delta": distribution(np.abs(gf_a - gf_b).ravel()),
        "gf_abs_delta_where_flipped": distribution(np.abs(gf_a - gf_b)[flips]),
    }

    # ---- 2. argmax (winning concept) changes ---------------------------
    changed = arg_a != arg_b
    argmax = {
        "n_pairs": int(arg_a.size),
        "n_changed": int(changed.sum()),
        "frac_changed": float(changed.mean()),
        "per_seed_changed": [int(x) for x in changed.sum(axis=1)],
        "n_features_changed_in_any_seed": int((changed.any(axis=0)).sum()),
        "n_features_changed_in_all_seeds": int((changed.all(axis=0)).sum()),
        "gf_abs_delta_where_changed": distribution(np.abs(gf_a - gf_b)[changed]),
        "note": "A changed winner with a near-zero gain delta is a TIE being "
                "broken differently, not a different answer. The delta "
                "distribution above is what separates the two readings.",
    }

    # ---- 3. AP divergence over the whole array -------------------------
    ap_delta = np.abs(a["ap"] - b_ap)
    ap = {
        "shape": list(a["ap"].shape),
        "n_elements": int(a["ap"].size),
        "n_differing": int((ap_delta != 0).sum()),
        "frac_differing": float((ap_delta != 0).mean()),
        "abs_delta_all": distribution(ap_delta.ravel()),
        "abs_delta_where_differing": distribution(ap_delta[ap_delta != 0]),
    }

    # ---- 4. fire_counts divergence -------------------------------------
    fc_delta = a["fire_counts"] - b_fire
    fc_abs = np.abs(fc_delta).astype(np.float64)
    fc = {
        "shape": list(a["fire_counts"].shape),
        "n_elements": int(a["fire_counts"].size),
        "n_differing": int((fc_delta != 0).sum()),
        "frac_differing": float((fc_delta != 0).mean()),
        "sum_reference": int(a["fire_counts"].sum()),
        "sum_other": int(b_fire.sum()),
        "sum_equal": bool(a["fire_counts"].sum() == b_fire.sum()),
        "abs_delta_all": distribution(fc_abs.ravel()),
        "abs_delta_where_differing": distribution(fc_abs[fc_delta != 0]),
        "per_seed_n_differing": [int(x) for x in (fc_delta != 0).sum(axis=1)],
        "note": "The SUM over features is nnz and is invariant by construction "
                "(TopK selects exactly k per row on either device), so a "
                "difference in the sum would mean something other than the "
                "gather changed.",
    }

    return {
        "artifact": "device_divergence",
        "gain_floor": GAIN_FLOOR,
        "n_seeds": int(n_seeds),
        "n_features": int(n_feats),
        "n_concepts_scored": len(a["concepts"]),
        "n_concepts_in_pool": int(a["in_pool"].sum()),
        "seed_order": {"reference": a["seeds"], "other_permutation_applied":
                       [int(x) for x in perm],
                       "aligned_on": "seed LABELS, never position (gotcha 29)"},
        "prevalence_invariant": "IDENTICAL (asserted; max |delta| == 0)",
        "gain_floor_flips": floor,
        "argmax_changes": argmax,
        "ap_divergence": ap,
        "fire_counts_divergence": fc,
    }


def format_summary(rep: dict) -> str:
    """TOTAL over its input: every branch prints something, so a completed
    measurement is never reported as a failed job (gotcha 32c)."""
    if not rep:
        return "device_divergence: EMPTY REPORT"
    out = [f"device_divergence -- {rep.get('n_seeds', '?')} seeds x "
           f"{rep.get('n_features', '?')} features x "
           f"{rep.get('n_concepts_scored', '?')} concepts "
           f"({rep.get('n_concepts_in_pool', '?')} in argmax pool)"]
    f = rep.get("gain_floor_flips", {})
    out.append(f"  1. GAIN-FLOOR FLIPS at G >= {rep.get('gain_floor')}: "
               f"{f.get('n_flipped')} of {f.get('n_pairs')} "
               f"({100 * (f.get('frac_flipped') or 0):.6f}%)  "
               f"[+{f.get('n_gained')} / -{f.get('n_lost')}]")
    a = rep.get("argmax_changes", {})
    out.append(f"  2. ARGMAX CHANGES: {a.get('n_changed')} of {a.get('n_pairs')} "
               f"({100 * (a.get('frac_changed') or 0):.6f}%); "
               f"{a.get('n_features_changed_in_any_seed')} features in >=1 seed")
    p = rep.get("ap_divergence", {})
    out.append(f"  3. AP: {p.get('n_differing')} of {p.get('n_elements')} differ "
               f"({100 * (p.get('frac_differing') or 0):.4f}%)")
    d = (p.get("abs_delta_where_differing") or {}).get("percentiles")
    if d:
        out.append(f"       |delta| where differing: median {d['50']:.3e}  "
                   f"p99 {d['99']:.3e}  max {d['100']:.3e}")
    c = rep.get("fire_counts_divergence", {})
    out.append(f"  4. FIRE_COUNTS: {c.get('n_differing')} of {c.get('n_elements')} "
               f"differ ({100 * (c.get('frac_differing') or 0):.4f}%); "
               f"nnz sum equal: {c.get('sum_equal')}")
    e = (c.get("abs_delta_where_differing") or {}).get("percentiles")
    if e:
        out.append(f"       |delta| where differing: median {e['50']:.1f}  "
                   f"p99 {e['99']:.1f}  max {e['100']:.1f}")
    return "\n".join(out)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True,
                   help="the GPU-gathered AP artifact")
    p.add_argument("--other", type=Path, required=True,
                   help="the CPU-gathered AP artifact")
    p.add_argument("--stratum", default="experimental")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    a = read_ap(args.reference, args.stratum)
    b = read_ap(args.other, args.stratum)
    rep = compare(a, b)
    rep["reference"] = str(args.reference)
    rep["other"] = str(args.other)
    rep["stratum"] = args.stratum
    rep["reference_git_sha"] = str(a["attrs"].get("git_sha", "ABSENT"))
    rep["other_git_sha"] = str(b["attrs"].get("git_sha", "ABSENT"))
    rep["device_attr_on_reference"] = str(a["attrs"].get("gather_device", "ABSENT"))
    rep["device_attr_on_other"] = str(b["attrs"].get("gather_device", "ABSENT"))
    rep["device_provenance_note"] = (
        "NEITHER artifact records its gather device -- write_identity has no such "
        "key. The device class is established EXTERNALLY: the reference was "
        "produced by slurm/feature_ap_experimental_array.sbatch, which requests "
        "--gpus=rtx_5000_ada:1; the other by "
        "slurm/feature_ap_experimental_array_cpu.sbatch, which requests no GPU "
        "and whose log prints 'WARNING: no GPU visible.'. Reported ABSENT rather "
        "than assumed to match.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2))
    print(format_summary(rep))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
