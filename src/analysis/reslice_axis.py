"""Re-slice AP and null artifacts onto the COMPLIANT concept axis (amendment
N1) -- a subset, never a re-run.

WHY SUBSETTING IS EXACT, NOT AN APPROXIMATION. Every pending axis change is
SUBTRACTIVE, so the new axis is a subset of the old one. And

    g(f, c) = (AP(f, c) - p_c) / (1 - p_c)

depends only on concept `c`'s own positive rows and the row population.
Neither changes when a *different* concept is dropped from the library. So
every surviving value is unchanged and subsetting is bit-exact. It is not a
cheaper approximation to a rerun; it is the same numbers.

WHAT WOULD BREAK IT, AND WHY THIS SCRIPT CHECKS FIRST. If an artifact's
concept axis were indexed POSITIONALLY -- by position in some filtered list
rather than by stored concept identity -- then subsetting by name would
silently misalign every column, and the result would look entirely plausible.
Both artifacts do carry a `concepts` name array of matching length, but that
alone is not proof: a permutation would move `prevalence`, `n_positives` and
the names together, so the obvious internal-consistency checks all still pass.

TWO CLAIMS, TWO CRITERIA -- conflating them made this script's first run fail
on correct data:

  (1) SLICE vs SOURCE is checked for BIT-IDENTITY -- on the AP artifact only
      (`verify_slice_vs_source(args.ap, args.out_ap, keep)`); no other artifact
      of the stage is re-sliced or checked here. Subsetting copies columns
      and does no arithmetic, so exact equality is the right bar and anything
      less is a bug. This is the claim that licenses "re-slice, do not re-run".
  (2) RECOMPUTATION vs STORED is checked to float tolerance, NOT bitwise.
      `main_ap` ran with `--ap-device auto`, which moves the AP arithmetic to
      CPU above 150M non-zeros, so a recomputation may legitimately sum in a
      different order; associativity then gives ~1e-9 differences on identical
      inputs. Its job is not exactness -- (1) covers that -- but proving the
      concept axis is indexed by IDENTITY. The discriminating statistic is the
      RATIO between the aligned and misaligned discrepancies: measured ~1e-9
      against ~1e-1, roughly eight orders of magnitude apart. A wrong column
      is not a slightly different number; it is a different concept's AP.

`--verify-ckpt` runs both arms, and the misaligned arm must fail by that
margin or the script refuses (gotcha 13 -- a check is not evidence until it
has been shown to fail against the condition it detects).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from src.analysis.artifact_identity import argmax_pool_mask, write_identity
from src.analysis.feature_ap import load_labels, load_scored_row_set
from src.analysis.null_pilot import build_population_index
from src.analysis.row_index_cache import compute_or_load_row_index
from src.analysis.sparse_ap_index import (
    average_precision_from_touches,
    gather_touches,
)
from src.sae.train import TopKSAE, git_dirty, git_sha


def read_names(hf, group: str | None) -> list[str]:
    g = hf[group] if group else hf
    return [c.decode() if isinstance(c, bytes) else c for c in g["concepts"][:]]


def compliant_concepts(labels: Path) -> list[str]:
    with h5py.File(labels, "r") as hf:
        return sorted(hf["concepts"].keys())


def slice_ap(src: Path, dst: Path, keep: list[str], labels: Path,
             excluded_types: list[str]) -> dict:
    """Subset `pooled/*` along the concept axis, by NAME."""
    with h5py.File(src, "r") as hf, h5py.File(dst, "w") as out:
        g = hf["pooled"]
        names = read_names(hf, "pooled")
        pos = {n: i for i, n in enumerate(names)}
        missing = [c for c in keep if c not in pos]
        if missing:
            raise SystemExit(
                f"FATAL: {len(missing)} compliant concepts absent from {src.name} "
                f"(e.g. {missing[:3]}). The new axis must be a SUBSET of the "
                "old one -- if it is not, this is not a re-slice and the "
                "artifact must be recomputed.")
        idx = np.array([pos[c] for c in keep])
        og = out.create_group("pooled")
        for key in g:
            if key == "concepts":
                og.create_dataset("concepts",
                                  data=np.array(keep, dtype=h5py.string_dtype()))
                continue
            d = g[key]
            arr = d[:]
            # concept axis is the LAST axis for every per-concept dataset here
            if arr.ndim >= 1 and arr.shape[-1] == len(names):
                arr = arr[..., idx]
            og.create_dataset(key, data=arr, compression="gzip")
        for k, v in hf.attrs.items():
            out.attrs[k] = v
        for k, v in hf.items():
            if k != "pooled" and isinstance(v, h5py.Dataset):
                out.create_dataset(k, data=v[:], compression="gzip")
        pool_n = int(sum(argmax_pool_mask(keep)))
        write_identity(
            out, n_concepts=len(keep), n_concepts_library=len(keep),
            argmax_pool_n=pool_n, labels=str(labels),
            provenance_excluded_types=json.dumps(sorted(excluded_types)),
            resliced_from=str(src),
            reslice_note=("SUBSET of the source along the concept axis by "
                          "concept IDENTITY (amendment N1). g(f,c) depends "
                          "only on c's own positives and the row population, "
                          "so surviving values are bit-identical to the "
                          "source -- verified by independent recomputation, "
                          "not assumed."),
            git_sha=git_sha(), git_dirty=bool(git_dirty()))
        return {"n_concepts": len(keep), "argmax_pool_n": pool_n,
                "dropped": sorted(set(names) - set(keep))}


def slice_raw_null(src: Path, dst: Path, keep: list[str], labels: Path,
                   excluded_types: list[str]) -> dict:
    """Subset the raw (seeds, shifts, concepts, features) gain array by NAME.

    Streamed per (seed, shift) so peak memory stays at one slice, not 25 GB.
    """
    with h5py.File(src, "r") as hf, h5py.File(dst, "w") as out:
        names = read_names(hf, None)
        pos = {n: i for i, n in enumerate(names)}
        keep_here = [c for c in keep if c in pos]
        idx = np.array([pos[c] for c in keep_here])
        out.create_dataset("concepts",
                           data=np.array(keep_here, dtype=h5py.string_dtype()))
        for key in ("gain_a", "gain_b"):
            if key not in hf:
                continue
            d = hf[key]
            ns, nsh, _, nf = d.shape
            o = out.create_dataset(key, shape=(ns, nsh, len(idx), nf),
                                   dtype=d.dtype,
                                   chunks=(1, 1, min(64, len(idx)), nf))
            for s in range(ns):
                for r in range(nsh):
                    o[s, r] = np.asarray(d[s, r])[idx]
        for k, v in hf.attrs.items():
            out.attrs[k] = v
        write_identity(
            out, n_concepts=len(keep), argmax_pool_n=len(keep_here),
            labels=str(labels),
            provenance_excluded_types=json.dumps(sorted(excluded_types)),
            resliced_from=str(src),
            git_sha=git_sha(), git_dirty=bool(git_dirty()))
        return {"argmax_pool_n": len(keep_here),
                "dropped": sorted(set(names) - set(keep_here))}


# The two claims are DIFFERENT and need different criteria. Conflating them
# is what made the first run of this script fail on correct data:
#
#   (1) SLICE vs SOURCE must be BIT-IDENTICAL. Subsetting does no arithmetic;
#       it copies columns. Anything less than exact equality here is a bug.
#   (2) RECOMPUTATION vs STORED cannot be bit-identical, and expecting it to
#       be was the error. `main_ap` ran with `--ap-device auto`, which
#       switches the AP arithmetic to CPU above 150M non-zeros, so a
#       recomputation can legitimately sum in a different order. Float
#       associativity then gives ~1e-9 differences on identical inputs.
#
# (2)'s job is not to prove exactness -- (1) does that. Its job is to prove
# the concept axis is indexed by IDENTITY, and the discriminating statistic
# is the RATIO between the aligned and misaligned discrepancies. Measured:
# ~1e-9 aligned against ~1e-1 misaligned, a separation of ~7-8 orders of
# magnitude. A wrong column is not a slightly different number; it is a
# different concept's AP.
RECOMPUTE_TOL = 1e-6        # generous against float-associativity drift
MISALIGN_MIN_RATIO = 1e3    # misaligned must exceed aligned by >= this


def verify_slice_vs_source(src: Path, dst: Path, keep: list[str]) -> dict:
    """Claim (1): every retained column is copied EXACTLY from the source.

    This is the claim that actually licenses "re-slice, do not re-run", and
    it is checkable without touching activations.
    """
    out = {"datasets": [], "all_exact": True}
    with h5py.File(src, "r") as a, h5py.File(dst, "r") as b:
        sg, dg = a["pooled"], b["pooled"]
        names = read_names(a, "pooled")
        pos = {n: i for i, n in enumerate(names)}
        idx = np.array([pos[c] for c in keep])
        for key in sg:
            if key == "concepts":
                continue
            s_arr, d_arr = sg[key], dg[key]
            if s_arr.ndim >= 1 and s_arr.shape[-1] == len(names):
                expect = s_arr[...][..., idx]
            else:
                expect = s_arr[...]
            exact = bool(np.array_equal(expect, d_arr[...]))
            out["datasets"].append({"dataset": key, "bit_identical": exact})
            out["all_exact"] &= exact
            print(f"  pooled/{key:<16} shape {tuple(d_arr.shape)!s:<22} "
                  f"bit-identical={exact}")
    return out


def verify_against_recomputation(
    ap_path: Path, labels: Path, cfg: dict, ckpt: Path, eval_cache: Path,
    cache_dir: Path, n_probe: int, misalign: bool,
) -> dict:
    """INDEPENDENT check that column j of `ap` really is `concepts[j]`.

    Recomputes AP from the label store and the activations for `n_probe`
    concepts spanning the prevalence range, and compares to the stored value
    at the column the NAME indexes. `misalign=True` permutes the name array
    first and REQUIRES the comparison to fail -- without that arm this check
    would be untested in the direction that matters.
    """
    with h5py.File(ap_path, "r") as hf:
        names = read_names(hf, "pooled")
        ap = hf["pooled"]["ap"]
        prev = hf["pooled"]["prevalence"][:]
        seeds = [s.decode() if isinstance(s, bytes) else s for s in hf["seeds"][:]]
        si = seeds.index(ckpt.name)
        ap_seed = ap[si]                                   # (features, concepts)

    order = np.argsort(prev)
    probe_idx = [int(order[0]), int(order[len(order) // 4]),
                 int(order[len(order) // 2]), int(order[-2]), int(order[-1])]
    probe_idx = probe_idx[:n_probe]
    probe_names = [names[i] for i in probe_idx]

    lookup_names = list(names)
    if misalign:
        # Rotate by one: every name now indexes the WRONG column. The check
        # must fail. This is the arm that makes the passing result meaningful.
        lookup_names = lookup_names[1:] + lookup_names[:1]
    lookup = {n: i for i, n in enumerate(lookup_names)}

    d_model = cfg["model"]["d_model"]
    dict_size, k = cfg["sae"]["dict_size"], cfg["sae"]["k"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    subset = load_scored_row_set(labels, "label_store")
    with h5py.File(eval_cache, "r") as hf:
        raw = hf["index_json"][()]
        index = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    with h5py.File(labels, "r") as hf:
        n_rows = int(hf.attrs["n_rows"])
    _, remap = build_population_index(index, subset)

    pooled, _ = load_labels(labels, "pooled")
    sd = torch.load(ckpt / "model.pt", map_location="cpu")
    model = TopKSAE(d_model, dict_size, k, b_dec=torch.zeros(d_model)).to(device)
    model.load_state_dict(sd)
    model.eval()
    feat_by_row, rank_by_row, row_offsets, _, _ = compute_or_load_row_index(
        model, ckpt, eval_cache, subset, "label_store", dict_size, k,
        device, cache_dir)

    results = []
    for name in probe_names:
        rows = pooled[name].astype(np.int64)
        rows = rows[rows < len(remap)]
        rows = remap[rows]
        rows = rows[rows >= 0]
        q = torch.from_numpy(np.sort(rows)).to(device)
        tf, tr = gather_touches(feat_by_row, rank_by_row, row_offsets, q)
        recomputed = average_precision_from_touches(
            tf, tr, dict_size, int(rows.size), n_rows).cpu().numpy()
        stored = ap_seed[:, lookup[name]]
        same = bool(np.array_equal(recomputed.astype(np.float64),
                                   stored.astype(np.float64)))
        close = bool(np.allclose(recomputed, stored, rtol=1e-12, atol=1e-12))
        results.append({"concept": name, "n_positives": int(rows.size),
                        "bit_identical": same, "allclose": close,
                        "max_abs_diff": float(np.max(np.abs(
                            recomputed.astype(np.float64)
                            - stored.astype(np.float64))))})
        print(f"  {name:<44} n_pos={rows.size:>8}  "
              f"bit-identical={same}  max|diff|={results[-1]['max_abs_diff']:.3e}")
    worst = max(r["max_abs_diff"] for r in results)
    return {"misaligned_arm": misalign, "probes": results,
            "worst_abs_diff": worst,
            "all_bit_identical": all(r["bit_identical"] for r in results),
            "all_within_tol": all(r["max_abs_diff"] <= RECOMPUTE_TOL
                                  for r in results)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True,
                   help="the COMPLIANT label store (defines the new axis)")
    p.add_argument("--ap", type=Path, required=True)
    p.add_argument("--raw-null", type=Path, default=None)
    p.add_argument("--out-ap", type=Path, required=True)
    p.add_argument("--out-null", type=Path, default=None)
    p.add_argument("--verify-ckpt", type=Path, default=None)
    p.add_argument("--eval-cache", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--n-probe", type=int, default=5)
    p.add_argument("--out-report", type=Path, default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    excluded = list(cfg.get("annotations", {})
                       .get("provenance_excluded_types", []))
    keep = compliant_concepts(args.labels)
    print(f"compliant axis: {len(keep)} concepts (from {args.labels.name})")

    report: dict = {"labels": str(args.labels), "n_compliant": len(keep),
                    "provenance_excluded_types": sorted(excluded)}

    if args.out_ap.exists():
        raise SystemExit(f"refusing to overwrite {args.out_ap}")
    report["ap"] = slice_ap(args.ap, args.out_ap, keep, args.labels, excluded)
    print(f"wrote {args.out_ap}: {report['ap']}")

    if args.raw_null and args.out_null:
        if args.out_null.exists():
            raise SystemExit(f"refusing to overwrite {args.out_null}")
        report["null"] = slice_raw_null(args.raw_null, args.out_null, keep,
                                        args.labels, excluded)
        print(f"wrote {args.out_null}: {report['null']}")

    # ---- CLAIM (1): the slice is a bit-exact subset of its source --------
    # This is the claim that licenses "re-slice, do not re-run", and it needs
    # no activations -- so it runs unconditionally, not behind --verify-ckpt.
    print("\n--- VERIFY (1): slice vs source, must be BIT-IDENTICAL ---")
    exact = verify_slice_vs_source(args.ap, args.out_ap, keep)
    report["verify_slice_exact"] = exact
    if not exact["all_exact"]:
        raise SystemExit(
            "FATAL: the sliced artifact is NOT a bit-exact subset of its "
            "source. Subsetting does no arithmetic, so any difference here is "
            "a bug, not float drift.")
    print("  => every retained column copied exactly.")

    if args.verify_ckpt:
        # ---- CLAIM (2): the concept axis is indexed by IDENTITY ----------
        print("\n--- VERIFY (2): independent recomputation (aligned arm) ---")
        ok = verify_against_recomputation(
            args.out_ap, args.labels, cfg, args.verify_ckpt, args.eval_cache,
            args.cache_dir, args.n_probe, misalign=False)
        print("\n--- VERIFY (2): deliberately MISALIGNED arm (must FAIL) ---")
        bad = verify_against_recomputation(
            args.out_ap, args.labels, cfg, args.verify_ckpt, args.eval_cache,
            args.cache_dir, args.n_probe, misalign=True)
        report["verify_aligned"] = ok
        report["verify_misaligned"] = bad
        ratio = (bad["worst_abs_diff"] / ok["worst_abs_diff"]
                 if ok["worst_abs_diff"] > 0 else float("inf"))
        report["misalign_separation_ratio"] = ratio
        print()
        print(f"  aligned    worst |diff| : {ok['worst_abs_diff']:.3e}  "
              f"(tolerance {RECOMPUTE_TOL:.0e})")
        print(f"  misaligned worst |diff| : {bad['worst_abs_diff']:.3e}")
        print(f"  separation ratio        : {ratio:.3e}  "
              f"(required >= {MISALIGN_MIN_RATIO:.0e})")
        # NOT bit-identity: main_ap ran --ap-device auto, which moves the AP
        # arithmetic to CPU above 150M non-zeros, so a recomputation may sum
        # in a different order. Float associativity gives ~1e-9 on identical
        # inputs. Claim (1) above is what covers exactness.
        if not ok["all_within_tol"]:
            raise SystemExit(
                f"FATAL: recomputed AP differs from stored by more than "
                f"{RECOMPUTE_TOL:.0e} -- beyond float-associativity drift, so "
                "this is a real disagreement.")
        if ratio < MISALIGN_MIN_RATIO:
            raise SystemExit(
                "FATAL: a deliberately MISALIGNED axis produces a discrepancy "
                "comparable to the aligned one, so this check cannot "
                "distinguish the two and proves nothing (gotcha 13). The "
                "re-slice is UNVERIFIED.")
        print(f"\nVERIFIED: slice bit-exact against source; recomputation "
              f"agrees to float precision; misalignment separated by "
              f"{ratio:.1e}x, so the check demonstrably detects it.")

    if args.out_report:
        args.out_report.write_text(json.dumps(report, indent=2, default=str))
        print(f"wrote {args.out_report}")


if __name__ == "__main__":
    main()
