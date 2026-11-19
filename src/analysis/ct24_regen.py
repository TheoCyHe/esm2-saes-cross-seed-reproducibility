"""§CT24-F item 2: the training cache's regeneration, gated against the record.

The original 24 training shards carry NO content hash (§CT23-pre item 7), so a
regeneration cannot be shown whole-file identical. What the record does hold:

  * `$PROJECT/archive/shards.json` -- the shard manifest of 2026-07-24;
  * every k = 32 checkpoint's `meta.json`: `n_rows` (180,416,896) and shard 0's
    attributes (`activations_attrs`);
  * `b_dec_init_sha256_16` = `f3185a6c22f91039` in all 40 checkpoints -- a hash
    of the geometric median of a fixed-seed sample of one million rows, the one
    recorded fingerprint that depends on the cache's CONTENT.

Each gate below compares against one of those and REFUSES (exit 1) on a
difference. Provenance keys are exempted by FULL KEY, never by a leaf-name
pattern, because a bit-identity criterion on a provenance key can only fail
(gotcha 25a). A key absent on one side is reported ABSENT and never counts as
agreeing (gotcha 15).

Subcommands: `shards` (after shard_fasta.py), `cache` (after the cache array:
row count and shard-0 attributes), `bdec` (after compute_b_dec.py), `hash`
(the sha256 manifest). Each writes a JSON verdict with identity attributes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

RECORDED_B_DEC_INIT = "f3185a6c22f91039"

# shards.json: written by shard_fasta.py. These keys describe the RUN, not the
# shards -- they differ by construction between 2026-07-24 and a regeneration.
SHARDS_PROVENANCE_KEYS = ("git_sha", "git_dirty", "config_hash", "source_fasta",
                          "timestamp_utc")

# Shard attributes that the regenerated cache.py writes and the 2026-07-24 one
# did not (amendment CQ, `efb6d54`; corrections entry 229). Reported, not gated.
ATTRS_ADDED_SINCE_RECORD = ("device_class",)


def _canon(v):
    """JSON-comparable form: numpy scalars and arrays to Python values."""
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return v


def compare_mapping(new: dict, old: dict, exempt: tuple = (),
                    expected_new_only: tuple = ()) -> dict:
    """Key-by-key comparison. Returns the verdict and every disagreement.

    `exempt` keys are reported and not compared. A key on one side only is
    ABSENT_ON_NEW / ABSENT_ON_OLD -- a failure, unless it is listed in
    `expected_new_only`, which is reported as EXPECTED_NEW_ONLY."""
    rows, ok = [], True
    for key in sorted(set(new) | set(old)):
        if key in exempt:
            rows.append({"key": key, "status": "EXEMPT_PROVENANCE",
                         "new": _canon(new.get(key)), "old": _canon(old.get(key))})
            continue
        if key not in old:
            if key in expected_new_only:
                rows.append({"key": key, "status": "EXPECTED_NEW_ONLY",
                             "new": _canon(new[key])})
            else:
                rows.append({"key": key, "status": "ABSENT_ON_OLD",
                             "new": _canon(new[key])})
                ok = False
            continue
        if key not in new:
            rows.append({"key": key, "status": "ABSENT_ON_NEW",
                         "old": _canon(old[key])})
            ok = False
            continue
        a, b = _canon(new[key]), _canon(old[key])
        if a == b:
            rows.append({"key": key, "status": "AGREE"})
        else:
            rows.append({"key": key, "status": "DIFFERS", "new": a, "old": b})
            ok = False
    return {"verdict": "PASS" if ok else "FAIL", "keys": rows}


def gate_shards(new_manifest: dict, old_manifest: dict) -> dict:
    return compare_mapping(new_manifest, old_manifest,
                           exempt=SHARDS_PROVENANCE_KEYS)


def gate_cache(shard_rows: list[int], shard0_attrs: dict, meta: dict) -> dict:
    """Summed rows against meta's n_rows; shard 0's attrs against meta's
    `activations_attrs` (the k = 32 checkpoints read shard 0's)."""
    if "n_rows" not in meta or "activations_attrs" not in meta:
        return {"verdict": "FAIL", "reason": "meta.json lacks n_rows or "
                "activations_attrs (ABSENT, never treated as matching)"}
    total = int(sum(shard_rows))
    rows_ok = total == int(meta["n_rows"])
    attrs = compare_mapping(shard0_attrs, meta["activations_attrs"],
                            expected_new_only=ATTRS_ADDED_SINCE_RECORD)
    return {"verdict": "PASS" if rows_ok and attrs["verdict"] == "PASS" else "FAIL",
            "n_shards": len(shard_rows), "rows_total": total,
            "rows_recorded": int(meta["n_rows"]), "rows_agree": rows_ok,
            "shard0_attrs": attrs}


def gate_bdec(b_dec_sha256_16: str | None,
              recorded: str = RECORDED_B_DEC_INIT) -> dict:
    """The author's decision 3 (2026-10-23): a mismatch stops §CT24's training."""
    if not b_dec_sha256_16:
        return {"verdict": "FAIL", "reason": "b_dec_sha256_16 ABSENT"}
    return {"verdict": "PASS" if b_dec_sha256_16 == recorded else "FAIL",
            "recomputed": b_dec_sha256_16, "recorded": recorded}


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _identity() -> dict:
    from src.sae.train import git_dirty, git_sha
    return {"git_sha": git_sha(), "git_dirty": git_dirty(),
            "written_utc": datetime.now(timezone.utc).isoformat(),
            "producer": "src.analysis.ct24_regen", "section": "§CT24-F item 2"}


def _write(out: Path, payload: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shards")
    s.add_argument("--new", type=Path, required=True)
    s.add_argument("--old", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("cache")
    c.add_argument("--shards", type=Path, nargs="+", required=True)
    c.add_argument("--meta", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("bdec")
    b.add_argument("--b-dec", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    h = sub.add_parser("hash")
    h.add_argument("--files", type=Path, nargs="+", required=True)
    h.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)

    if a.cmd == "shards":
        res = gate_shards(json.loads(a.new.read_text(encoding="utf-8")),
                          json.loads(a.old.read_text(encoding="utf-8")))
        res["new"], res["old"] = str(a.new), str(a.old)
    elif a.cmd == "cache":
        import h5py
        rows, attrs0 = [], None
        for i, path in enumerate(a.shards):
            with h5py.File(path, "r") as hf:
                rows.append(int(hf["activations"].shape[0]))
                if i == 0:
                    attrs0 = {k: _canon(v) for k, v in hf.attrs.items()}
        meta = json.loads(a.meta.read_text(encoding="utf-8"))
        res = gate_cache(rows, attrs0, meta)
        res["shard_rows"], res["meta"] = rows, str(a.meta)
    elif a.cmd == "bdec":
        import torch
        blob = torch.load(a.b_dec, map_location="cpu")
        stored = blob.get("b_dec_sha256_16") if isinstance(blob, dict) else None
        # Recomputed exactly as train.py hashes the init it loads (float32
        # bytes), so the gate does not rest on the producer's own field.
        vec = blob["b_dec"] if isinstance(blob, dict) else blob
        recomputed = hashlib.sha256(torch.as_tensor(vec, dtype=torch.float32)
                                    .numpy().tobytes()).hexdigest()[:16]
        res = gate_bdec(recomputed)
        res["stored_field"] = stored
        if stored != recomputed:
            res["verdict"] = "FAIL"
            res["reason"] = "the blob's b_dec_sha256_16 disagrees with its vector"
        res["b_dec_path"] = str(a.b_dec)
        res["blob_keys"] = sorted(blob) if isinstance(blob, dict) else None
    else:
        res = {"verdict": "RECORDED",
               "files": [{"path": str(f), "bytes": f.stat().st_size,
                          "sha256": sha256_file(f)} for f in a.files],
               "note": "no content hash of the original 24 shards is recorded "
                       "(§CT23-pre item 7); these hashes identify the "
                       "regeneration only."}
    res["identity"] = _identity()
    _write(a.out, res)
    print(json.dumps({k: v for k, v in res.items() if k in
                      ("verdict", "rows_total", "rows_recorded", "recomputed",
                       "recorded")}))
    return 0 if res["verdict"] in ("PASS", "RECORDED") else 1


if __name__ == "__main__":
    sys.exit(main())
