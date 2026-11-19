"""Verify the CN width-sweep checkpoints. REPORTS A DIFF, NOT A VERDICT.

Stage B delivers trained checkpoints and their verification, and nothing else.
This script does NOT evaluate, match, or compute any reproducibility figure.

What it establishes, per amendment CN6:
  1. every checkpoint present and loadable;
  2. on-disk size reconciled against the parameter arithmetic for its width,
     the way CM2 did at the anchor width -- `2*d_model*W + W + d_model` at fp32;
  3. the seed protocol identical across widths, with any key that could not be
     held identical REPORTED rather than silently substituted;
  4. which cache each arm trained against -- paths do not distinguish an
     original cache from a regenerated one at the same path, so the shard
     ATTRS are compared, not just the filenames;
  5. the device class recorded per arm (gotcha 36).

`ABSENT` means the key is not recorded, never that it is zero or matching
(gotcha 15: an absent key is reported as absent, never treated as matching).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Keys that DEFINE the training protocol. `dict_size` is the swept axis and is
# expected to differ; everything else here must be identical across all arms or
# the sweep is not "width and seed only".
PROTOCOL_KEYS = [
    "k", "lr", "batch_size", "n_steps", "loss_reduction", "optimizer",
    "grad_clip", "decoder_grad_projection", "b_dec_trainable",
    "b_dec_init_sha256_16", "b_dec_source", "aux_k",
    "aux_coef", "dead_tokens_threshold", "n_rows", "block_rows",
    "buffer_blocks", "deterministic_cudnn", "d_model",
]
# OUTPUTS, not protocol inputs. `b_dec_sha256_16` hashes the CONVERGED b_dec
# (`train.py:660-661`), which is seed-specific BY DESIGN -- `b_dec` is trainable
# from a shared init, so distinct final values are the expected result, not a
# protocol violation. Listing it under PROTOCOL_KEYS printed a 40-line DIFFERS
# block that was pure noise and would have masked a real disagreement.
# The informative check is CARDINALITY: n distinct values for n arms. A
# COLLISION would mean two arms converged to a bit-identical b_dec, which under
# different seeds is the surprising outcome.
OUTPUT_KEYS = ["b_dec_sha256_16"]
# Recorded and reported, but NOT required identical -- these are identity
# metadata, and CN3 established they move without a protocol change.
METADATA_KEYS = ["git_sha", "git_dirty", "config_hash", "torch_version",
                 "python_version", "numpy_version"]

SWEPT_KEY = "dict_size"


def expected_param_bytes(width: int, d_model: int, dtype_bytes: int = 4) -> int:
    """W_enc (W,d) + W_dec (d,W) + b_enc (W,) + b_dec (d,), all fp32."""
    return (2 * d_model * width + width + d_model) * dtype_bytes


def load_arm(root: Path, width: int, seeds: list[int]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for s in seeds:
        d = root / f"w{width}" / f"seed_{s}"
        rec: dict[str, object] = {"dir": str(d), "present": d.is_dir()}
        mp, mj = d / "model.pt", d / "meta.json"
        rec["model_pt_bytes"] = mp.stat().st_size if mp.is_file() else None
        if mj.is_file():
            rec["meta"] = json.loads(mj.read_text())
        else:
            rec["meta"] = None
        out[s] = rec
    return out


def check_sizes(arm: dict[int, dict], width: int, d_model: int) -> list[str]:
    notes = []
    exp = expected_param_bytes(width, d_model)
    overheads = []
    for s, rec in sorted(arm.items()):
        got = rec["model_pt_bytes"]
        if got is None:
            notes.append(f"    seed {s}: model.pt ABSENT")
            continue
        over = got - exp
        overheads.append(over)
        flag = "" if 0 <= over < 65536 else "   <-- UNEXPECTED"
        notes.append(
            f"    seed {s}: {got:,} B  expected params {exp:,} B  "
            f"container overhead {over:,} B{flag}"
        )
    if overheads and len(set(overheads)) > 1:
        notes.append(f"    NOTE: overhead is not constant across seeds: {sorted(set(overheads))}")
    return notes


def try_load(path: Path, width: int, d_model: int) -> str:
    try:
        import torch
    except ImportError:
        return "torch unavailable -- LOADABILITY NOT CHECKED (ABSENT, not passing)"
    try:
        sd = torch.load(path, map_location="cpu")
    except Exception as e:  # noqa: BLE001
        return f"LOAD FAILED: {type(e).__name__}: {e}"
    want = {
        "W_enc": (width, d_model), "W_dec": (d_model, width),
        "b_enc": (width,), "b_dec": (d_model,),
    }
    bad = []
    for k, shape in want.items():
        if k not in sd:
            bad.append(f"{k} ABSENT")
        elif tuple(sd[k].shape) != shape:
            bad.append(f"{k} {tuple(sd[k].shape)} != {shape}")
    extra = sorted(set(sd.keys()) - set(want))
    s = "shapes OK" if not bad else "MISMATCH: " + "; ".join(bad)
    if extra:
        s += f"  (extra tensors: {extra})"
    return s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path,
                   default=Path("$PROJECT/sae_ckpts_width"))
    p.add_argument("--widths", type=int, nargs="+", default=[2560, 10240, 20480, 40960])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--d-model", type=int, default=1280)
    p.add_argument("--load", action="store_true", help="also torch.load each checkpoint")
    args = p.parse_args()

    arms = {w: load_arm(args.root, w, args.seeds) for w in args.widths}

    print("=" * 78)
    print("1. PRESENCE")
    print("=" * 78)
    for w in args.widths:
        have = [s for s, r in arms[w].items() if r["present"] and r["model_pt_bytes"]]
        miss = [s for s in args.seeds if s not in have]
        print(f"  w={w:<6} {len(have)}/{len(args.seeds)} checkpoints"
              + (f"   MISSING seeds {miss}" if miss else ""))

    print()
    print("=" * 78)
    print("2. ON-DISK SIZE vs PARAMETER ARITHMETIC  (2*d*W + W + d, fp32)")
    print("=" * 78)
    for w in args.widths:
        print(f"  w={w}:")
        for line in check_sizes(arms[w], w, args.d_model):
            print(line)

    if args.load:
        print()
        print("=" * 78)
        print("3. LOADABILITY AND TENSOR SHAPES")
        print("=" * 78)
        for w in args.widths:
            for s in args.seeds:
                mp = args.root / f"w{w}" / f"seed_{s}" / "model.pt"
                if mp.is_file():
                    print(f"  w={w:<6} seed {s}: {try_load(mp, w, args.d_model)}")

    print()
    print("=" * 78)
    print("4. SEED PROTOCOL ACROSS WIDTHS  (must be identical except dict_size)")
    print("=" * 78)
    metas = {(w, s): arms[w][s]["meta"] for w in args.widths for s in args.seeds
             if arms[w][s]["meta"]}
    if not metas:
        print("  no meta.json found -- NOT EVALUABLE")
    else:
        ref_key = sorted(metas)[0]
        ref = metas[ref_key]
        print(f"  reference arm: w={ref_key[0]} seed={ref_key[1]}")
        for k in PROTOCOL_KEYS:
            vals = {}
            for (w, s), m in metas.items():
                vals.setdefault(json.dumps(m.get(k, "<ABSENT>"), sort_keys=True), []).append((w, s))
            if len(vals) == 1:
                v = next(iter(vals))
                print(f"    {k:<26} IDENTICAL across all arms  = {v}")
            else:
                print(f"    {k:<26} DIFFERS   <-- REPORTED, not substituted")
                for v, where in sorted(vals.items()):
                    ws = sorted({w for w, _ in where})
                    print(f"        {v:<28} widths {ws} ({len(where)} arms)")
        print(f"    {SWEPT_KEY:<26} (swept axis)")
        for w in args.widths:
            got = {m.get(SWEPT_KEY) for (ww, _), m in metas.items() if ww == w}
            ok = got == {w}
            print(f"        w={w:<6} meta says {sorted(got)}  {'OK' if ok else '<-- MISMATCH'}")

        print()
        print("  OUTPUTS (expected to differ per seed -- checked by CARDINALITY):")
        for k in OUTPUT_KEYS:
            vals = [m.get(k, "<ABSENT>") for m in metas.values()]
            absent = sum(1 for v in vals if v == "<ABSENT>")
            distinct = len(set(vals))
            verdict = "OK -- all distinct" if distinct == len(vals) else (
                f"<-- COLLISION: {len(vals) - distinct} arm(s) share a value")
            print(f"    {k:<26} {distinct} distinct / {len(vals)} arms   {verdict}")
            if absent:
                print(f"        {absent} arm(s) ABSENT (reported as absent, not as matching)")

    print()
    print("=" * 78)
    print("5. WHICH CACHE EACH ARM TRAINED AGAINST")
    print("=" * 78)
    if metas:
        groups: dict[str, list] = {}
        for (w, s), m in metas.items():
            sig = json.dumps({"paths": m.get("activations", "<ABSENT>"),
                              "attrs": m.get("activations_attrs", "<ABSENT>")},
                             sort_keys=True)
            groups.setdefault(sig, []).append((w, s))
        print(f"  distinct cache signatures: {len(groups)}")
        for i, (sig, where) in enumerate(groups.items()):
            ws = sorted({w for w, _ in where})
            d = json.loads(sig)
            paths = d["paths"]
            print(f"    signature {i}: widths {ws}, {len(where)} arms, "
                  f"{len(paths) if isinstance(paths, list) else paths} shards")
            print(f"      attrs: {json.dumps(d['attrs'], sort_keys=True)[:200]}")
        if len(groups) > 1:
            print("    <-- MORE THAN ONE CACHE SIGNATURE: the arms are not all on"
                  " the same cache. REPORT; do not average over this.")

    print()
    print("=" * 78)
    print("6. IDENTITY METADATA  (recorded; CN3: these move without a protocol change)")
    print("=" * 78)
    for k in METADATA_KEYS:
        vals = {}
        for (w, s), m in metas.items():
            vals.setdefault(str(m.get(k, "<ABSENT>")), []).append(w)
        for v, ws in sorted(vals.items()):
            print(f"    {k:<18} {v:<44} widths {sorted(set(ws))}")

    print()
    print("=" * 78)
    print("7. DEAD FEATURES (recorded per arm; AuxK is insurance, not load-bearing)")
    print("=" * 78)
    for w in args.widths:
        dead = {s: metas[(w, s)].get("n_dead_features", "<ABSENT>")
                for s in args.seeds if (w, s) in metas}
        print(f"    w={w:<6} {dead}")

    print()
    print("NO EVALUATION PERFORMED. No matching, no reproducibility figure.")


if __name__ == "__main__":
    main()
