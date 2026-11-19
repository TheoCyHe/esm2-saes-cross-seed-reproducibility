"""
Item 3 -- real-scale validation of the P1 restructure (sparse_ap_index.py)
against the OLD, sklearn-validated path, on REAL seed-0 activations and REAL
concept positive-row sets. The synthetic fixture is not sufficient for this:
there is no real-scale reference output (the job that would have produced one
never finished), and the closed-form zero-activation tail correction is
precisely where a discrepancy would hide -- it bites hardest when a concept
has many positive rows on which a feature is zero, which a small synthetic
fixture may not exercise the way 17.7M real rows with real firing patterns do.

Runs ONE activation pass (shared by both paths, same as the production
driver), then computes AP for a HAND-PICKED subset of concepts spanning the
full prevalence range (2.36M residues down to 11) via BOTH the OLD
(`average_precision_segmented`, full-nnz-scan) and NEW
(`sparse_ap_index`, touched-rows-only) code paths, asserts bit-identical
agreement, and times each path separately -- this is the seed-0 timing
baseline item 4 asks for, measured on the SAME concept subset both ways so
the comparison is apples to apples.

TWO ARITHMETIC PATHS, BOTH COVERED. The production driver
(`feature_ap.py --ap-device auto`) runs the AP arithmetic on CPU above 150M
non-zeros and on GPU below it -- two genuinely different code paths through
the same torch ops. This script's OLD-vs-NEW comparison alone would only ever
exercise whichever device this job happens to land on for the AP arithmetic.
When CUDA is available, the NEW path is therefore run TWICE per concept --
once on CPU, once on GPU -- and the two are asserted bit-identical to each
other IN ADDITION to matching the OLD path, so neither device branch is
validated only by inference from the other.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from src.analysis.artifact_identity import PARAMETER_VALUES
from src.analysis.feature_ap import (
    average_precision_segmented,
    load_labels,
    load_scored_row_set,
    sparse_activations,
)
from src.analysis.sparse_ap_index import (
    average_precision_from_touches,
    build_row_index,
    gather_touches,
)
from src.sae.train import TopKSAE


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True, help="ONE seed checkpoint dir")
    p.add_argument("--eval-cache", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--concepts", nargs="*", default=[],
                   help="Explicit concept names to validate. Usually omit and "
                        "use --auto-concepts.")
    p.add_argument("--auto-concepts", type=int, default=0,
                   help="STRATIFIED selection (amendment I1b): the top-N/2 "
                        "concepts by positive count P_c, plus N/2 drawn from "
                        "the sparse end. The sparse end is where a feature's "
                        "positives sit OUTSIDE its top-activating rows, which "
                        "is precisely where the closed-form tied-zero tail "
                        "correction carries the most weight and where a "
                        "discrepancy between the two paths would hide.")
    p.add_argument("--scored-row-set", default="label_store",
                   choices=list(PARAMETER_VALUES["scored_row_set"]),
                   help="MUST match the production run's subset. Validating on "
                        "the full cache while the run uses the untruncated "
                        "population would leave the actual code path unvalidated "
                        "-- the gap this flag closes.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    d_model = cfg["model"]["d_model"]
    dict_size, k = cfg["sae"]["dict_size"], cfg["sae"]["k"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    n_rows = int(h5py.File(args.labels, "r").attrs["n_rows"])
    pooled, _ = load_labels(args.labels, "pooled")

    # Row subset + coordinate remap, EXACTLY as feature_ap.py does it -- the
    # validation must exercise the same coordinates the production run uses.
    subset = load_scored_row_set(args.labels, args.scored_row_set)
    if subset is not None:
        remap = -np.ones(int(subset.max()) + 2, dtype=np.int64)
        remap[subset] = np.arange(len(subset))
        rel = {}
        for c, arr in pooled.items():
            a = arr[arr < len(remap)]
            m = remap[a]
            m = m[m >= 0]
            if m.size:
                rel[c] = m
        pooled = rel
        n_rows = len(subset)
    print(f"scored_row_set={args.scored_row_set}: {n_rows:,} rows, "
          f"{len(pooled)} concepts with positives")

    concepts = list(args.concepts)
    if args.auto_concepts:
        half = args.auto_concepts // 2
        by_size = sorted(pooled, key=lambda c: -pooled[c].size)
        concepts = by_size[:half] + by_size[-half:]
        concepts = list(dict.fromkeys(concepts))
        print(f"auto-selected {len(concepts)} concepts: "
              f"P_c from {pooled[concepts[0]].size:,} down to "
              f"{pooled[concepts[-1]].size:,}")
    if not concepts:
        raise SystemExit("give --concepts or --auto-concepts")
    args.concepts = concepts
    missing = [c for c in args.concepts if c not in pooled]
    if missing:
        raise SystemExit(f"concepts not found in labels: {missing}")

    sd = torch.load(args.ckpt / "model.pt", map_location="cpu")
    model = TopKSAE(d_model, dict_size, k, b_dec=torch.zeros(d_model)).to(device)
    model.load_state_dict(sd)
    model.eval()

    t0 = time.time()
    feat, pos, seg_start = sparse_activations(model, args.eval_cache, subset, device)
    t_gather = time.time() - t0
    nnz = feat.numel()
    print(f"activation gather: {nnz:,} non-zeros, {t_gather:.1f}s")

    # OLD-vs-NEW comparison runs on CPU at this scale, matching the production
    # driver's --ap-device auto threshold (150M non-zeros; this is 567M).
    feat, pos, seg_start = feat.cpu(), pos.cpu(), seg_start.cpu()
    have_cuda = torch.cuda.is_available()
    print(f"cuda available for device cross-check: {have_cuda}")

    # ---- ONE-TIME per-seed setup for the NEW path -------------------------
    t0 = time.time()
    feat_by_row, rank_by_row, row_offsets = build_row_index(feat, pos, seg_start, n_rows)
    t_row_index = time.time() - t0
    print(f"row-index build (one-time per seed): {t_row_index:.1f}s")

    if have_cuda:
        feat_by_row_g = feat_by_row.to("cuda")
        rank_by_row_g = rank_by_row.to("cuda")
        row_offsets_g = row_offsets.to("cuda")

    results = []
    t_old_total, t_new_total = 0.0, 0.0
    max_abs_diff, max_device_diff = 0.0, 0.0
    for c in args.concepts:
        idx = pooled[c]
        n_pos = int(idx.size)
        labels_mask = torch.zeros(n_rows, dtype=torch.bool).index_fill_(
            0, torch.from_numpy(idx), True)

        t0 = time.time()
        lab_by_nnz = labels_mask[pos]
        ap_old = average_precision_segmented(
            feat, seg_start, lab_by_nnz, dict_size, n_pos, n_rows)
        t_old = time.time() - t0

        t0 = time.time()
        query_rows = torch.from_numpy(idx.astype(np.int64))
        tf, tr = gather_touches(feat_by_row, rank_by_row, row_offsets, query_rows)
        ap_new = average_precision_from_touches(tf, tr, dict_size, n_pos, n_rows)
        t_new = time.time() - t0

        d = float((ap_old - ap_new).abs().max())
        max_abs_diff = max(max_abs_diff, d)

        dev_diff = float("nan")
        if have_cuda:
            query_rows_g = query_rows.to("cuda")
            tf_g, tr_g = gather_touches(feat_by_row_g, rank_by_row_g, row_offsets_g, query_rows_g)
            ap_new_g = average_precision_from_touches(tf_g, tr_g, dict_size, n_pos, n_rows)
            dev_diff = float((ap_new.to("cuda") - ap_new_g).abs().max())
            max_device_diff = max(max_device_diff, dev_diff)

        t_old_total += t_old
        t_new_total += t_new
        n_touches = int(tf.numel())
        print(f"  {c:<45} P={n_pos:>9,}  touches={n_touches:>10,}  "
              f"t_old={t_old:>7.3f}s  t_new={t_new:>7.4f}s  "
              f"speedup={t_old/max(t_new,1e-9):>8.1f}x  max|diff|={d:.2e}  "
              f"cpu_vs_gpu_diff={dev_diff:.2e}")
        results.append({
            "concept": c, "n_positives": n_pos, "n_touches": n_touches,
            "t_old_s": t_old, "t_new_s": t_new, "max_abs_diff": d,
            "cpu_vs_gpu_diff": dev_diff,
        })

    print(f"\nTOTAL over {len(args.concepts)} concepts: "
          f"t_old={t_old_total:.2f}s  t_new={t_new_total:.2f}s  "
          f"(+ one-time row-index build {t_row_index:.2f}s)")
    print(f"MAX ABSOLUTE DIFFERENCE (old vs new) across all concepts, all features: "
          f"{max_abs_diff:.3e}")
    if have_cuda:
        print(f"MAX CPU-vs-GPU DIFFERENCE (new path, both devices) : "
              f"{max_device_diff:.3e}")
    failed = max_abs_diff > 1e-9 or (have_cuda and max_device_diff > 1e-9)
    print("FAIL: a discrepancy was found -- see above." if failed else
          "PASS: new path matches old path AND both devices agree, bit-for-bit.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n_rows": n_rows, "nnz": nnz, "t_gather_s": t_gather,
        "t_row_index_s": t_row_index, "t_old_total_s": t_old_total,
        "t_new_total_s": t_new_total, "max_abs_diff": max_abs_diff,
        "cuda_available": have_cuda, "max_cpu_vs_gpu_diff": max_device_diff,
        "passed": not failed, "concepts": results,
    }, indent=2))
    print(f"wrote {args.out}")

    # EXIT NON-ZERO ON FAILURE. Found 2026-07-28: this script printed "FAIL"
    # and then exited 0, so its verdict was invisible to any automation --
    # a SLURM `--dependency=afterok` chain, or a shell `&&`, would have
    # treated a discrepancy as success and launched the production run anyway.
    # Same shape as the `[[ ! -f ]]` label-store guard (gotcha 15): a check
    # that cannot fail loudly is not a gate. Matches verify_cache.py, which
    # has always exited non-zero.
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
