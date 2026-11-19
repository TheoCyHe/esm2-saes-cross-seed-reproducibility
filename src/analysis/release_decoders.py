"""Amendment CT §CT18-D4 item 2 (the author, 2026-11-01): the decoders of all 140
dictionaries of record, one release asset per set, each hashed.

    python -m src.analysis.release_decoders --out-dir DIR [--set NAME ...]
    python -m src.analysis.release_decoders --verify-dir DIR [--sums TSV] --out-json FILE

Each asset `decoders_<set>.h5` holds one dataset, `W_dec`, of shape
(n_seeds, d_model, width), float32 and uncompressed. Each slice is a checkpoint's
state-dict `W_dec` (`src/sae/train.py`: (d_model, dict_size)), and the slices
are in NUMERIC seed order. The labels are stored as an attribute, never implied
(gotcha 29), alongside each checkpoint's path, sha256 and recorded `git_sha`.

GATES, each on the written file re-read from disk: every slice equals its
checkpoint's `W_dec` bit for bit; the labels and the count match the checkpoint
directories; the shape is (d_model, width) for the set's width; the asset is no
larger than ASSET_LIMIT (GitHub's per-asset limit as recalled, 2 GiB; not
verified, §CT18-D4 check 3). Determinism: the smallest set is written twice and
must hash identically. `MANIFEST.csv` is written only when every gate passed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

PROJECT = Path("$PROJECT")
D_MODEL = 1280
ASSET_LIMIT = 2 * 1024 ** 3
# set -> (checkpoint directories holding seed_<n>/model.pt, width, n_seeds)
SETS = {
    "ensemble_n40": ((PROJECT / "sae_ckpts", PROJECT / "sae_ckpts_ext"), 10240, 40),
    "k64": ((PROJECT / "sae_ckpts_k64",), 10240, 20),
    "w2560": ((PROJECT / "sae_ckpts_width" / "w2560",), 2560, 10),
    "w10240": ((PROJECT / "sae_ckpts_width" / "w10240",), 10240, 10),
    "w20480": ((PROJECT / "sae_ckpts_width" / "w20480",), 20480, 10),
    "w40960": ((PROJECT / "sae_ckpts_width" / "w40960",), 40960, 10),
    "k64_w2560": ((PROJECT / "sae_ckpts_ct29" / "k64_w2560",), 2560, 10),
    "k64_w40960": ((PROJECT / "sae_ckpts_ct29" / "k64_w40960",), 40960, 10),
    "k16_w10240": ((PROJECT / "sae_ckpts_ct29" / "k16_w10240",), 10240, 10),
    "k128_w10240": ((PROJECT / "sae_ckpts_ct29" / "k128_w10240",), 10240, 10),
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def seed_dirs(dirs) -> list[tuple[int, Path]]:
    """(label, directory) for every seed_<n> under `dirs`, in numeric order;
    refuses a duplicate label or a directory without model.pt."""
    out = {}
    for d in dirs:
        for p in Path(d).glob("seed_*"):
            if not p.is_dir():
                continue
            lab = int(p.name[len("seed_"):])
            if lab in out:
                raise SystemExit(f"REFUSING: seed {lab} is in both {out[lab]} and {p}")
            if not (p / "model.pt").exists():
                raise SystemExit(f"REFUSING: {p} has no model.pt")
            out[lab] = p
    return sorted(out.items())


def load_w_dec(seed_dir: Path) -> np.ndarray:
    import torch
    sd = torch.load(seed_dir / "model.pt", map_location="cpu", weights_only=True)
    w = sd["W_dec"]
    if w.dtype != torch.float32:
        raise SystemExit(f"REFUSING: {seed_dir} W_dec is {w.dtype}, not float32")
    return w.numpy()


def recorded_git_sha(seed_dir: Path) -> str | None:
    m = seed_dir / "meta.json"
    if not m.exists():
        return None
    return json.loads(m.read_text(encoding="utf-8")).get("git_sha")


def write_asset(path: Path, seeds: list[tuple[int, Path]], width: int, commit: str) -> None:
    """One slice at a time, so a 2 GB asset never sits in memory whole."""
    with h5py.File(path, "w") as hf:
        ds = hf.create_dataset("W_dec", shape=(len(seeds), D_MODEL, width), dtype="<f4",
                               track_times=False)
        for i, (_, d) in enumerate(seeds):
            w = load_w_dec(d)
            if w.shape != (D_MODEL, width):
                raise SystemExit(f"REFUSING: {d} W_dec is {w.shape}, not {(D_MODEL, width)}")
            ds[i] = w
        hf.attrs["seed_labels"] = json.dumps([lab for lab, _ in seeds])
        hf.attrs["checkpoints"] = json.dumps([str(d / "model.pt") for _, d in seeds])
        hf.attrs["checkpoint_sha256"] = json.dumps([sha256_file(d / "model.pt") for _, d in seeds])
        hf.attrs["checkpoint_git_sha"] = json.dumps([recorded_git_sha(d) for _, d in seeds])
        hf.attrs["d_model"] = D_MODEL
        hf.attrs["width"] = width
        hf.attrs["producing_commit"] = commit
        hf.attrs["governing_amendment"] = "CT §CT18-D4 item 2"


def gate_asset(path: Path, seeds: list[tuple[int, Path]], width: int, n_expected: int,
               limit: int = ASSET_LIMIT) -> list[dict]:
    g = []
    with h5py.File(path, "r") as hf:
        labels = json.loads(hf.attrs["seed_labels"])
        g.append({"what": f"{path.name}: {n_expected} seeds, labels as the checkpoint directories",
                  "verdict": "PASS" if (labels == [lab for lab, _ in seeds]
                                        and len(labels) == n_expected) else "FAIL",
                  "labels": labels})
        ds = hf["W_dec"]
        ok_shape = ds.shape == (n_expected, D_MODEL, width) and ds.dtype == np.dtype("<f4")
        g.append({"what": f"{path.name}: shape (n, {D_MODEL}, {width}) float32",
                  "verdict": "PASS" if ok_shape else "FAIL", "shape": list(ds.shape)})
        bad = []
        for i, (lab, d) in enumerate(seeds):
            if i >= ds.shape[0] or not np.array_equal(ds[i], load_w_dec(d)):
                bad.append(lab)
        g.append({"what": f"{path.name}: every slice equals its checkpoint's W_dec bit for bit",
                  "verdict": "PASS" if not bad and ok_shape else "FAIL", "differing_seeds": bad})
    size = path.stat().st_size
    g.append({"what": f"{path.name}: {size} bytes <= {limit}",
              "verdict": "PASS" if size <= limit else "FAIL", "bytes": size})
    return g


def build(out_dir: Path, names: list[str], commit: str, sets: dict = SETS) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    gates, rows = [], []
    for name in names:
        dirs, width, n = sets[name]
        seeds = seed_dirs(dirs)
        path = out_dir / f"decoders_{name}.h5"
        if path.exists():
            raise SystemExit(f"REFUSING: {path} exists")
        write_asset(path, seeds, width, commit)
        g = gate_asset(path, seeds, width, n)
        gates += g
        print("\n".join(f"{x['verdict']:<6} {x['what']}" for x in g), flush=True)
        rows.append([path.name, sha256_file(path), path.stat().st_size, n,
                     json.dumps([lab for lab, _ in seeds]), ";".join(str(d) for d in dirs), commit])
    # determinism: the smallest set again, which must hash as the first write
    small = min(names, key=lambda s: sets[s][1] * sets[s][2])
    again = out_dir / f"_determinism_decoders_{small}.h5"
    dirs, width, _ = sets[small]
    write_asset(again, seed_dirs(dirs), width, commit)
    first = next(r[1] for r in rows if r[0] == f"decoders_{small}.h5")
    same = sha256_file(again) == first
    again.unlink()
    gates.append({"what": f"decoders_{small}.h5 written twice hashes identically",
                  "verdict": "PASS" if same else "FAIL"})
    all_pass = all(x["verdict"] == "PASS" for x in gates)
    if all_pass:
        with open(out_dir / "MANIFEST.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["asset", "sha256", "bytes", "n_dictionaries", "seed_labels",
                        "source_dirs", "producing_commit"])
            w.writerows(rows)
    res = {"governing_amendment": "CT §CT18-D4 item 2", "producing_commit": commit,
           "gates": gates, "all_pass": all_pass, "assets": rows}
    (out_dir / "gates_decoders.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def verify(dir_: Path, names: list[str], sums: Path | None, commit: str, sets: dict = SETS) -> dict:
    """Decoder files that already exist in `dir_`, checked against the checkpoints themselves
    (section CT18-D11 decision 6): `gate_asset`'s gates on each file (labels, shape, every
    slice equal to its checkpoint's W_dec bit for bit, size), and, with `sums`, each file's
    sha256, md5 and size against that table's row (columns asset, sha256, bytes, md5). Writes
    nothing into `dir_`. An absent file, or a file the table does not name, fails."""
    table = None
    if sums is not None:
        with open(sums, encoding="utf-8", newline="") as f:
            table = {r["asset"]: r for r in csv.DictReader(f, delimiter="\t")}
    gates = []
    for name in names:
        dirs, width, n = sets[name]
        path = dir_ / f"decoders_{name}.h5"
        if not path.exists():
            gates.append({"what": f"{path.name}: present", "verdict": "FAIL"})
            continue
        g = gate_asset(path, seed_dirs(dirs), width, n)
        if table is not None:
            r = table.get(path.name)
            got = {"sha256": sha256_file(path), "md5": md5_file(path), "bytes": path.stat().st_size}
            ok = (r is not None and got["sha256"] == r["sha256"] and got["md5"] == r["md5"]
                  and got["bytes"] == int(r["bytes"]))
            g.append({"what": f"{path.name}: sha256, md5 and size as the table's row",
                      "verdict": "PASS" if ok else "FAIL", **got})
        gates += g
        print("\n".join(f"{x['verdict']:<6} {x['what']}" for x in g), flush=True)
    if table is not None:
        extra = sorted(set(table) - {f"decoders_{n}.h5" for n in names})
        gates.append({"what": "the table names no file outside the sets verified",
                      "verdict": "PASS" if not extra else "FAIL", "extra": extra})
    all_pass = bool(names) and all(x["verdict"] == "PASS" for x in gates)
    return {"governing_amendment": "CT §CT18-D11 decision 6", "verifying_commit": commit,
            "directory": str(dir_), "sums": None if sums is None else str(sums),
            "gates": gates, "all_pass": all_pass}


def main(argv=None) -> int:
    from src.sae.train import git_dirty, git_sha
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--set", action="append", choices=sorted(SETS), default=None)
    ap.add_argument("--verify-dir", type=Path, default=None,
                    help="check the decoder files already in this directory against the checkpoints")
    ap.add_argument("--sums", type=Path, default=None,
                    help="with --verify-dir: a table (asset, sha256, bytes, md5; tab-separated) each file must match")
    ap.add_argument("--out-json", type=Path, default=None, help="with --verify-dir: where the gates are written")
    a = ap.parse_args(argv)
    if (a.out_dir is None) == (a.verify_dir is None):
        raise SystemExit("give exactly one of --out-dir and --verify-dir")
    if git_dirty():
        raise SystemExit("REFUSING: the checkout is dirty")
    if a.verify_dir is not None:
        if a.out_json is None:
            raise SystemExit("--verify-dir needs --out-json")
        res = verify(a.verify_dir, a.set or list(SETS), a.sums, git_sha())
        a.out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"ALL PASS: {res['all_pass']}", flush=True)
        return 0 if res["all_pass"] else 1
    res = build(a.out_dir, a.set or list(SETS), git_sha())
    print(f"ALL PASS: {res['all_pass']}", flush=True)
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
