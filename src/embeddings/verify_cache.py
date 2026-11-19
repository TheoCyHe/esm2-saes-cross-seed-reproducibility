"""
Verify an activation cache against the FASTA it was built from.

The failure mode this exists to catch is the one CLAUDE.md warns about: a
cache that is the right shape and full of plausible numbers but misaligned by
a residue, which silently corrupts every downstream AP number. "It ran and
produced a big file" is not evidence of correctness, so this re-derives the
expected layout from the source sequences and checks it row by row.

Reads only h5py/numpy -- no torch, no transformers, no GPU -- so it can be
run on a login node or locally against a downloaded cache.

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from src.embeddings.cache import ESM_DIM, read_fasta


def _plain(v):
    """h5 attrs are numpy scalars / bytes; compare on plain values."""
    if isinstance(v, bytes):
        return v.decode()
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        return v.item()
    return v


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--max-len", type=int, default=1022)
    # amendment CQ, A4: assert the VALUES of the identity attrs, not only their
    # presence. Optional so existing call sites keep their behaviour; when
    # given, a disagreement is a FAIL like every other check here.
    p.add_argument("--expect-layer", type=int, default=None,
                   help="Require attrs['layer'] to equal this value.")
    p.add_argument("--expect-model", default=None,
                   help="Require attrs['model'] to equal this string.")
    p.add_argument("--nan-check-rows", type=int, default=200_000,
                   help="Rows to sample for the NaN/Inf scan. 0 checks all "
                        "(slow on a full-size cache).")
    args = p.parse_args()

    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if cond:
            print(f"  ok    {msg}")
        else:
            print(f"  FAIL  {msg}")
            failures.append(msg)

    records = read_fasta(args.fasta)
    by_acc = {acc: seq for acc, seq in records}
    print(f"{len(records)} sequences in {args.fasta}")

    with h5py.File(args.cache, "r") as hf:
        acts = hf["activations"]
        index = json.loads(hf["index_json"][()])
        attrs = dict(hf.attrs)

        print(f"\nattrs: {attrs}")
        print(f"activations: shape={acts.shape} dtype={acts.dtype}\n")

        print("-- file-level --")
        check(acts.dtype == np.float16, f"dtype is float16 (got {acts.dtype})")
        check(acts.shape[1] == ESM_DIM, f"width is {ESM_DIM} (got {acts.shape[1]})")
        check("layer" in attrs, "layer recorded in attrs")
        check("model" in attrs, "model recorded in attrs")
        # amendment CQ, A4 -- values, not presence. `max_len` is compared to
        # the --max-len this run derives its expectations from; a cache built
        # at another cap would pass every per-sequence check below against
        # the wrong expectation.
        check(_plain(attrs.get("max_len")) == args.max_len,
              f"max_len attr equals --max-len ({_plain(attrs.get('max_len'))!r} "
              f"vs {args.max_len})")
        if args.expect_layer is not None:
            check(_plain(attrs.get("layer")) == args.expect_layer,
                  f"layer attr equals --expect-layer "
                  f"({_plain(attrs.get('layer'))!r} vs {args.expect_layer})")
        if args.expect_model is not None:
            check(_plain(attrs.get("model")) == args.expect_model,
                  f"model attr equals --expect-model "
                  f"({_plain(attrs.get('model'))!r} vs {args.expect_model!r})")
        check(attrs.get("final_layer_norm_applied") in (False, np.False_),
              "final layer norm NOT applied (reference SAEs expect raw stream)")

        print("\n-- index integrity --")
        check(len(index) == len(records),
              f"index has one entry per sequence ({len(index)} vs {len(records)})")

        idx_accs = {e["accession"] for e in index}
        check(idx_accs == set(by_acc),
              "index accessions match the FASTA exactly")

        subsampled = bool(attrs.get("subsampled", False))
        total = sum(e["n_rows"] for e in index)
        check(total == acts.shape[0],
              f"index row count equals dataset rows ({total} vs {acts.shape[0]})")

        # Offsets must tile [0, n) with no gaps or overlaps -- a gap here means
        # some protein's activations are being read from another protein's rows.
        ordered = sorted(index, key=lambda e: e["start"])
        contiguous = all(
            ordered[i]["start"] + ordered[i]["n_rows"] == ordered[i + 1]["start"]
            for i in range(len(ordered) - 1)
        )
        check(ordered[0]["start"] == 0 if ordered else True, "first offset is 0")
        check(contiguous, "index offsets are contiguous (no gaps or overlaps)")

        print("\n-- per-sequence alignment --")
        len_mismatch = [
            e["accession"] for e in index
            if e["seq_len"] != min(len(by_acc[e["accession"]]), args.max_len)
        ]
        check(not len_mismatch,
              f"seq_len matches min(len(seq), max_len) for all "
              f"({len(len_mismatch)} bad)")

        trunc_bad = [
            e["accession"] for e in index
            if e["truncated"] != (len(by_acc[e["accession"]]) > args.max_len)
        ]
        check(not trunc_bad,
              f"truncated flag matches raw length > max_len "
              f"({len(trunc_bad)} bad)")

        if not subsampled:
            row_mismatch = [e["accession"] for e in index
                            if e["n_rows"] != e["seq_len"]]
            check(not row_mismatch,
                  f"unsubsampled cache stores one row per residue "
                  f"({len(row_mismatch)} bad)")
            expected_total = sum(min(len(s), args.max_len) for _, s in records)
            check(acts.shape[0] == expected_total,
                  f"total rows equals summed sequence lengths "
                  f"({acts.shape[0]} vs {expected_total})")
        else:
            print("  note  cache is subsampled; per-residue counts not expected "
                  "to equal sequence lengths")
            rid_bad = [
                e["accession"] for e in index
                if e["residue_ids"] is None
                or len(e["residue_ids"]) != e["n_rows"]
                or (e["residue_ids"] and max(e["residue_ids"]) >= e["seq_len"])
            ]
            check(not rid_bad,
                  f"residue_ids present, sized correctly, and in range "
                  f"({len(rid_bad)} bad)")

        print("\n-- numerical sanity --")
        n = acts.shape[0]
        if args.nan_check_rows and n > args.nan_check_rows:
            rng = np.random.default_rng(0)
            sample_idx = np.sort(rng.choice(n, size=args.nan_check_rows,
                                            replace=False))
            sample = acts[sample_idx].astype(np.float32)
            scope = f"{args.nan_check_rows:,} sampled rows"
        else:
            sample = acts[:].astype(np.float32)
            scope = "all rows"
        check(np.isfinite(sample).all(), f"no NaN/Inf ({scope})")
        check(not np.all(sample == 0, axis=1).any(),
              f"no all-zero rows ({scope})")
        print(f"  info  activation range [{sample.min():.3f}, {sample.max():.3f}], "
              f"mean {sample.mean():.4f}, std {sample.std():.4f}")

    print()
    if failures:
        print(f"FAILED {len(failures)} check(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
