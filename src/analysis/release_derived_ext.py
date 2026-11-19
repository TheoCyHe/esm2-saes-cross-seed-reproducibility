"""Amendment CT §CT18-F: the extended release bundle -- every camera-ready
number's derived inputs, written as flat files, each family gated against its
owner artifact before anything reaches `$PROJECT/release/derived/`.

Subcommands (each writes into `--out-dir`, which must be empty for a family's
files, and appends its gate verdicts to `<out-dir>/gates_<family>.json`):

  arm <name>     one of ARMS: `arm_<name>_features.csv.gz`,
                 `arm_<name>_sim_decoder.csv.gz`, `arm_<name>_fvu_per_seed.csv`
                 (and for k64 `arm_k64_sim_activation.csv.gz`)
  anchors        `sim_decoder_n39.csv.gz`, gated on Table 1's jackknife
  replay         phase 2a: the activation arm, the halves, the rules and the
                 raw-dot-product variant, §CT15-§CT17, §CT14's strata -- their
                 similarity files, `features_logM_variants_n39.csv.gz` and the
                 owner JSONs verbatim; gated by replaying each producer's own
                 command on the re-read files (below)
  phase2b        the concentration covariate and both carrier arms (§20.17),
                 the disorder shares (§20.23), the classification (§20.6), the
                 concept subspaces' per-cell rows (RECOUNTED), and every other
                 owner JSON verbatim with the record's hashes checked
  manifest       refuses unless every family's gates passed; writes MANIFEST.csv
                 (the 94ca8b9 rows kept) and meta_ext.json; with --copy, puts
                 the new files into the project bundle and re-hashes them there

THE GATES RE-READ THE WRITTEN FILES FROM DISK and recompute from them through
this repository's own fitting code (`jackknife_dictionaries._fit_one` /
`loo_counts` / `jackknife`; `ct28_rescore.fit_with_jackknife`). The criterion is
BIT_IDENTICAL against the owner JSON's block (§CT18-F item 5.3): the
recomputed block is round-tripped through JSON and compared leaf for leaf. On a
difference, the owner's own producer path is rerun on the owner's own inputs in
the same process (the control, gotcha 35); the verdict names which of the two
differs. Nothing is loosened.

THE SEED AXIS IS WRITTEN AS LABELS (gotcha 29): every row carries its anchor
label and every similarity column its comparison-dictionary label.

Fits run serially in-process: three of the producers' pools rely on globals a
forked worker inherits, which a spawned worker would not (§CT18-F item 6).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

PROJECT = Path("$PROJECT")
A = PROJECT / "analysis"

# Each arm's inputs exactly as `slurm/ct_pass.sbatch` (match|fit, fit_topmerge)
# and `slurm/ct28.sbatch` named them.
ARMS = {
    "w2560": {"kind": "width", "config": "configs/width/mvp_w2560.yaml"},
    "w10240": {"kind": "width", "config": "configs/width/mvp_w10240.yaml"},
    "w20480": {"kind": "width", "config": "configs/width/mvp_w20480.yaml"},
    "w40960": {"kind": "width", "config": "configs/width/mvp_w40960.yaml"},
    "k64": {"kind": "k64", "config": "configs/k64/mvp_k64.yaml"},
    "k64_w2560": {"kind": "grid", "config": "configs/grid/mvp_k64_w2560.yaml"},
    "k64_w40960": {"kind": "grid", "config": "configs/grid/mvp_k64_w40960.yaml"},
    "k16_w10240": {"kind": "topmerged", "config": "configs/grid/mvp_k16_w10240.yaml"},
    "k128_w10240": {"kind": "topmerged", "config": "configs/grid/mvp_k128_w10240.yaml"},
}
K64N9_USE = [str(s) for s in range(10)]


def arm_paths(name: str, root: Path = A) -> dict:
    kind = ARMS[name]["kind"]
    ct = root / "ct"
    p = {"sim": ct / f"jackknife_match_{name}_ct.h5",
         "null_reductions": ct / f"null_pilot_{name}_ct.h5",
         "d_fs": ct / f"d_fs_{name}_ct.h5",
         "sf_h5": ct / f"ct28_sf_{name}_ct.h5"}
    if kind == "width":
        p.update(rf_counts_all_theta=root / "cs" / f"rf_counts_all_theta_{name}_n9.h5",
                 m1_json=root / "cs" / f"rf_marginal_m1_{name}_n9.json",
                 fvu=root / "cs" / f"fvu_eval_{name}.json")
    else:
        p.update(rf_counts_all_theta=ct / f"rf_counts_all_theta_{name}_ct.h5",
                 m1_json=ct / f"rf_marginal_m1_{name}_ct.json",
                 fvu=ct / f"fvu_eval_{name}_ct.json")
    if kind == "topmerged":
        p.update(gtilde=ct / f"gtilde_{name}_b_ratio_topmerged_ct.h5",
                 slope_owner=ct / f"slope_{name}_topmerged_ct.json", sffit_owner=None)
    else:
        p.update(gtilde=ct / f"gtilde_{name}_b_ratio_ct.h5",
                 slope_owner=ct / f"slope_{name}_ct.json",
                 sffit_owner=ct / f"ct28_sffit_{name}_ct.json")
    if name == "k64":
        p.update(act_counts=ct / "activation_correlation_counts_k64_ct.h5",
                 k64n9_arm_owner=ct / "ct28_arm_k64n9_ct.json",
                 k64n9_sffit_owner=ct / "ct28_sffit_k64n9_ct.json")
    return p


# ---------------------------------------------------------------------------
# Flat-file I/O -- pure, exercised by tests/test_release_derived_ext.py
# ---------------------------------------------------------------------------

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def text_of(a: np.ndarray) -> np.ndarray:
    """Exact round-trip text: numpy's shortest form for the array's own dtype
    (float32 or float64), `-inf` / `inf` / `nan` as such, ints as ints."""
    a = np.asarray(a)
    if a.dtype == np.bool_:
        return np.where(a, "1", "0")
    if a.dtype.kind in "iu":
        return a.astype(str)
    if a.dtype not in (np.float32, np.float64):
        raise TypeError(f"unsupported dtype {a.dtype}")
    return a.astype(str)


def write_csv_gz(path: Path, header: list[str], rows) -> None:
    """Deterministic gzip: no file name and mtime 0 in the header."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow(r)
    with open(path, "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:
            g.write(buf.getvalue().encode("utf-8"))


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        return head, list(rd)


def write_sim(path: Path, sim: np.ndarray, anchors: list, cols: list) -> None:
    """(anchor, column, feature) similarity as one row per (anchor, feature) and
    one column per comparison dictionary, `s_<label>`; an anchor's own column
    is written empty and must hold no similarity (-inf or nan)."""
    anchors, cols = [str(a) for a in anchors], [str(c) for c in cols]
    if sim.shape[:2] != (len(anchors), len(cols)):
        raise ValueError(f"sim {sim.shape} vs {len(anchors)} anchors x {len(cols)} columns")
    own = [cols.index(a) if a in cols else None for a in anchors]
    for i, j in enumerate(own):
        if j is not None and np.isfinite(sim[i, j]).any():
            raise ValueError(f"anchor {anchors[i]}'s own column holds finite values")

    def rows():
        for i, a in enumerate(anchors):
            t = text_of(sim[i].T)             # (feature, column)
            if own[i] is not None:
                t[:, own[i]] = ""
            for f in range(t.shape[0]):
                yield [a, f] + t[f].tolist()
    write_csv_gz(path, ["anchor_seed", "feature"] + [f"s_{c}" for c in cols], rows())


def read_sim(path: Path, dtype, own_fill: str = "-inf") -> tuple[np.ndarray, list[str], list[str]]:
    """Inverse of `write_sim`; an empty cell is read as `own_fill` (-inf, the MNN
    producers' value on the diagonal; nan for the one-way array). Rows must be
    anchor-major and feature-ordered."""
    head, rows = read_csv(path)
    if head[:2] != ["anchor_seed", "feature"] or not all(h.startswith("s_") for h in head[2:]):
        raise ValueError(f"{path.name}: not a similarity file")
    cols = [h[2:] for h in head[2:]]
    anchors = list(dict.fromkeys(r[0] for r in rows))
    n_f = len(rows) // len(anchors)
    if n_f * len(anchors) != len(rows):
        raise ValueError(f"{path.name}: rows are not anchors x features")
    vals = np.array([[v if v != "" else own_fill for v in r[2:]] for r in rows], dtype=dtype)
    feat = np.array([int(r[1]) for r in rows]).reshape(len(anchors), n_f)
    if not np.array_equal(feat, np.tile(np.arange(n_f), (len(anchors), 1))):
        raise ValueError(f"{path.name}: features are not in order")
    lab = [r[0] for r in rows]
    if lab != [a for a in anchors for _ in range(n_f)]:
        raise ValueError(f"{path.name}: rows are not anchor-major")
    sim = vals.reshape(len(anchors), n_f, len(cols)).transpose(0, 2, 1).copy()
    return sim, anchors, cols


def write_table(path: Path, anchors: list, columns: dict) -> None:
    """Per-(anchor, feature) table: `columns` maps name -> (A, D) array (any
    dtype) or None (written empty for the whole column)."""
    anchors = [str(a) for a in anchors]
    names = list(columns)
    shapes = {k: v.shape for k, v in columns.items() if v is not None}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"column shapes differ: {shapes}")
    n_a, n_f = next(iter(shapes.values()))
    if n_a != len(anchors):
        raise ValueError("anchor count differs from the arrays")
    txt = {k: (text_of(v) if v is not None else None) for k, v in columns.items()}

    def rows():
        for i, a in enumerate(anchors):
            for f in range(n_f):
                yield [a, f] + [("" if txt[k] is None else txt[k][i, f]) for k in names]
    write_csv_gz(path, ["anchor_seed", "feature"] + names, rows())


def read_table(path: Path, dtypes: dict) -> tuple[dict, list[str]]:
    """Inverse of `write_table` for the named columns, each cast to its dtype;
    a column written empty for an anchor comes back as nan (floats only)."""
    head, rows = read_csv(path)
    col = {h: i for i, h in enumerate(head)}
    anchors = list(dict.fromkeys(r[0] for r in rows))
    n_f = len(rows) // len(anchors)
    if n_f * len(anchors) != len(rows):
        raise ValueError(f"{path.name}: rows are not anchors x features")
    out = {}
    for k, dt in dtypes.items():
        raw = [r[col[k]] for r in rows]
        if np.dtype(dt) == np.bool_:
            arr = np.array([v == "1" for v in raw], dtype=bool)
        else:
            arr = np.array([v if v != "" else "nan" for v in raw], dtype=dt)
        out[k] = arr.reshape(len(anchors), n_f)
    return out, anchors


def same_json(a, b) -> bool:
    """Leaf-for-leaf equality after a JSON round trip (floats compared by
    their exact repr; the order of dict keys ignored)."""
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


def first_difference(a, b, path="") -> str | None:
    a, b = json.loads(json.dumps(a)), json.loads(json.dumps(b))
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}/{k}: present on one side only"
            d = first_difference(a[k], b[k], f"{path}/{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {a!r} vs {b!r}"


def arrays_equal(got: np.ndarray, want: np.ndarray, dtype) -> bool:
    """Exact equality after casting `want` to the written dtype; nan equals nan
    for floats only."""
    want = np.asarray(want).astype(dtype)
    if got.shape != want.shape:
        return False
    if np.dtype(dtype).kind == "f":
        return bool(np.array_equal(got, want, equal_nan=True))
    return bool(np.array_equal(got, want))


def verdict(mine, owner, what: str) -> dict:
    d = first_difference(mine, owner)
    return {"what": what, "verdict": "BIT_IDENTICAL" if d is None else "DIFFERS",
            "first_difference": d}


# ---------------------------------------------------------------------------
# Recomputation from re-read arrays, through the producers' own functions
# ---------------------------------------------------------------------------

def slope_block(sim, logM, log_phi, d_fs, keep, column_labels, theta=0.70) -> dict:
    """`jackknife_dictionaries.cmd_fit`'s `full_fit` and `per_theta` blocks at
    one theta, serially, from arrays."""
    from src.analysis import jackknife_dictionaries as jk
    jk._W.clear()
    jk._W.update({"sim": sim, "logM": logM, "log_phi": log_phi, "d_fs": d_fs,
                  "keep": keep, "column_labels": [int(c) for c in column_labels]})
    n_anchors, n_seeds, _ = sim.shape
    full = jk._fit_one(jk.counts_from_sim(sim, theta), np.arange(n_anchors), n_seeds - 1)
    res = [jk._loo_task((d, theta)) for d in range(n_seeds)]
    res.sort(key=lambda r: r["left_out_column"])
    failed = [r for r in res if r["n_anchor_fits_failed"] > 0]
    j = jk.jackknife([r["fe_pp"] for r in res], full["fe_pp"])
    jk_ci = [full["fe_pp"] - 1.959964 * j["se"], full["fe_pp"] + 1.959964 * j["se"]]
    ivs = {"fixed_effect": full["fe_ci95_pp"], "across_seed": full["across_seed_ci95_pp"],
           "jackknife": jk_ci}
    per = {"full_fe_pp": full["fe_pp"], "full_fe_se_pp": full["fe_se_pp"],
           "jackknife": j, "W_jackknife_se_over_fe_se": j["se"] / full["fe_se_pp"],
           "intervals_95_pp": ivs, "widest": jk.widest_interval(ivs),
           "n_loo_fits_failed": len(failed),
           "full_fit_anchor_failures": full["n_anchor_fits_failed"], "loo": res}
    return {"full_fit": full, "per_theta": per}


def k64n9_block(sim, logM, log_phi, d_fs, keep, anchors, cols, use) -> dict:
    """`ct28_rescore.cmd_arm`'s `at_0.70` on a comparison set, serially."""
    from src.analysis import ct28_rescore as c28
    c28._G.clear()
    c28._G.update({"sim": sim, "logM": logM, "log_phi": log_phi, "d_fs": d_fs,
                   "keep": keep, "row_labels": [a for a in anchors if a in use],
                   "use": list(use), "col_labels": list(cols),
                   "row_index": {a: i for i, a in enumerate(anchors)}})
    return c28.fit_with_jackknife(0.70, workers=1)


def sf_summary(sf: np.ndarray, keep: np.ndarray) -> dict:
    """`ct28_rescore.cmd_sffit`'s three summaries of s_f, its own expressions:
    the fit set's distribution, the per-anchor fit-set means, and all cells."""
    q = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    return {"distribution": {"n": int(keep.sum()), "mean": float(sf[keep].mean()),
                             "sd": float(sf[keep].std(ddof=1)),
                             "quantiles": dict(zip([str(x) for x in q],
                                                   np.percentile(sf[keep], q).tolist()))},
            "per_anchor_mean": [float(sf[a][keep[a]].mean()) for a in range(sf.shape[0])],
            "all_cells": {"n": int(sf.size), "mean": float(sf.mean()),
                          "per_anchor_mean": [float(r.mean()) for r in sf]}}


def sf_reduce(one: np.ndarray, labels: list, use: list) -> np.ndarray:
    from src.analysis.ct28_rescore import sf_on_comparison_set
    return sf_on_comparison_set(one, labels, use)


# ---------------------------------------------------------------------------
# arm
# ---------------------------------------------------------------------------

def _load(paths: dict, config: Path) -> dict:
    """The arm's join inputs through the record's own loader and gate, plus the
    un-logged arrays the loader does not return (M, firing rate, d_fs's
    retained flag), aligned by label exactly as the loader aligns them."""
    from src.analysis import jackknife_dictionaries as jk
    from src.analysis.seed_axis import align_to_reference
    args = SimpleNamespace(sim=paths["sim"], rf_counts_all_theta=paths["rf_counts_all_theta"],
                           m1_json=paths["m1_json"], gtilde=paths["gtilde"],
                           null_reductions=paths["null_reductions"], d_fs=paths["d_fs"],
                           degenerate_flag=None, config=config)
    inp = jk.load_join_inputs(args)
    anchors = [str(a) for a in inp["anchor_labels"]]
    with h5py.File(paths["gtilde"], "r") as hf:
        M = align_to_reference(inp["gtilde_seed_order"], hf["M"][:], anchors)
        n_rows = int(hf.attrs["n_rows"])
    with h5py.File(paths["null_reductions"], "r") as hf:
        labels = [(s.decode() if isinstance(s, bytes) else str(s)).split("_")[-1]
                  for s in hf["seeds"][:]]
        fire_rate = align_to_reference(labels, hf["fire_rate"][:], anchors)
    with h5py.File(paths["d_fs"], "r") as hf:
        ds = [str(int(s)) for s in hf["seeds"][:]]
        d_ret = align_to_reference(ds, hf["retained_at_floor"][:], anchors)
    stored, smeta = jk._stored_counts(paths["rf_counts_all_theta"])
    # the exported arrays must be the loader's, bit for bit
    from src.analysis.design3_join import log_M
    if not np.array_equal(log_M(M, str(paths["gtilde"])), inp["logM"]):
        raise SystemExit("FATAL: log M from the exported M differs from the loader's")
    if not np.array_equal(np.log10(np.maximum(fire_rate, 1e-12)), inp["log_phi"]):
        raise SystemExit("FATAL: log10 phi from the exported firing rate differs")
    return {**inp, "M": M, "fire_rate": fire_rate, "d_retained": d_ret,
            "n_rows": n_rows, "stored": stored, "anchors": anchors,
            "cols": [str(c) for c in inp["column_labels"]]}


def build_arm(name: str, out_dir: Path, root: Path = A) -> dict:
    from src.experiment_config import REPO_ROOT
    paths = arm_paths(name, root)
    inp = _load(paths, REPO_ROOT / ARMS[name]["config"])
    anchors, cols = inp["anchors"], inp["cols"]
    thetas = sorted(inp["stored"])
    with h5py.File(paths["sf_h5"], "r") as hf:
        one = hf["oneway_max_cos_anchor_rows"][:]
        sf_gate = str(hf.attrs["gate_vs_stored_mnn_sim"])
        sf_lab = [str(s) for s in json.loads(hf.attrs["anchor_seed_labels"])]
    if sf_lab != anchors:
        raise SystemExit(f"FATAL: {name}: s_f anchor order {sf_lab} != {anchors}")
    if sf_gate != "EXACT":
        raise SystemExit(f"FATAL: {name}: the s_f file's gate is {sf_gate}")
    sf = sf_reduce(one, sf_lab, anchors)
    columns = {f"k_{t:.2f}": np.asarray(inp["stored"][t], dtype=np.int64) for t in thetas}
    columns.update({"M": inp["M"], "log_M": inp["logM"], "fire_rate": inp["fire_rate"],
                    "d_fs": inp["d_fs"], "d_fs_retained_at_floor": inp["d_retained"],
                    "keep_a4a": inp["keep"], "s_f": sf})
    if name == "k64":
        sf9 = np.full_like(sf, np.nan)
        part = sf_reduce(one, sf_lab, K64N9_USE)
        for i, u in enumerate(K64N9_USE):
            sf9[anchors.index(u)] = part[i]
        columns["s_f_use_0to9"] = sf9
    out_dir.mkdir(parents=True, exist_ok=True)
    f_feat = out_dir / f"arm_{name}_features.csv.gz"
    f_sim = out_dir / f"arm_{name}_sim_decoder.csv.gz"
    f_fvu = out_dir / f"arm_{name}_fvu_per_seed.csv"
    for f in (f_feat, f_sim, f_fvu):
        if f.exists():
            raise SystemExit(f"REFUSING: {f} exists")
    write_table(f_feat, anchors, columns)
    write_sim(f_sim, inp["sim"], anchors, cols)
    fvu = json.loads(paths["fvu"].read_text(encoding="utf-8"))
    with open(f_fvu, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["seed", "fvu_raw", "fvu_standardised"])
        for s in sorted(fvu["per_seed"], key=int):
            w.writerow([s, repr(float(fvu["per_seed"][s]["raw"]["fvu"])),
                        repr(float(fvu["per_seed"][s]["standardised"]["fvu"]))])
    files = [f_feat, f_sim, f_fvu]
    if name == "k64":
        with h5py.File(paths["act_counts"], "r") as hf:
            asim = hf["mnn_sim_anchor_rows"][:]
            alab = [str(s) for s in json.loads(hf.attrs["seed_labels"])]
        if alab != anchors:
            raise SystemExit(f"FATAL: k64 activation anchor order {alab} != {anchors}")
        f_act = out_dir / "arm_k64_sim_activation.csv.gz"
        write_sim(f_act, asim, anchors, cols)
        files.append(f_act)
    return gate_arm(name, out_dir, inp, sf, columns, paths, fvu)


def gate_arm(name, out_dir, inp, sf, columns, paths, fvu) -> dict:
    """Every gate re-reads the written files."""
    g = {"arm": name, "gates": []}
    G = g["gates"]
    thetas = sorted(inp["stored"])
    dt = {f"k_{t:.2f}": np.int64 for t in thetas}
    dt.update({"M": np.float64, "log_M": np.float64, "fire_rate": np.float64,
               "d_fs": np.float64, "d_fs_retained_at_floor": bool, "keep_a4a": bool,
               "s_f": np.float32})
    if name == "k64":
        dt["s_f_use_0to9"] = np.float32
    tab, anchors = read_table(out_dir / f"arm_{name}_features.csv.gz", dt)
    sim, s_anchors, cols = read_sim(out_dir / f"arm_{name}_sim_decoder.csv.gz", np.float32)
    ok = (anchors == inp["anchors"] and s_anchors == inp["anchors"] and cols == inp["cols"])
    G.append({"what": "labels re-read equal the loader's", "verdict": "PASS" if ok else "FAIL"})
    bad_rt = [k for k, v in columns.items() if not arrays_equal(tab[k], v, dt[k])]
    if not np.array_equal(sim, inp["sim"]):
        bad_rt.append("sim_decoder")
    G.append({"what": "every written array round-trips exactly",
              "verdict": "PASS" if not bad_rt else "FAIL", "differing": bad_rt})
    from src.analysis.jackknife_dictionaries import counts_from_sim
    bad = [t for t in thetas if not np.array_equal(counts_from_sim(sim, t), inp["stored"][t])]
    G.append({"what": f"counts from the re-read similarity equal the stored counts at {len(thetas)} thetas",
              "verdict": "PASS" if not bad else "FAIL", "thetas_differing": bad,
              "match_gate_of_record": inp["match_attrs"].get("gate_verdict")})
    n_c = sim.shape[1] - 1
    m1 = json.loads(paths["m1_json"].read_text(encoding="utf-8"))
    from src.analysis.rf_marginal import marginals
    rate = marginals(tab["k_0.70"], n_c)["mean"]       # M1's own reduction (mean of k/n)
    rec = m1["marginals_by_theta"]["0.70"]["mnn"]["mean"]
    G.append({"what": "mean match rate at 0.70 equals the M1 file's", "verdict":
              "BIT_IDENTICAL" if rate == rec else "DIFFERS", "recomputed": rate, "owner": rec})
    owner = json.loads(paths["slope_owner"].read_text(encoding="utf-8"))
    mine = slope_block(sim, tab["log_M"], np.log10(np.maximum(tab["fire_rate"], 1e-12)),
                       tab["d_fs"], tab["keep_a4a"], cols)
    for blk, ob in (("full_fit", owner["full_fit"]["0.70"]), ("per_theta", owner["per_theta"]["0.70"])):
        v = verdict(mine[blk], ob, f"{paths['slope_owner'].name} {blk} 0.70")
        if v["verdict"] != "BIT_IDENTICAL":
            # the control (gotcha 35): the producer's path on the loader's own arrays
            ctl = slope_block(inp["sim"], inp["logM"], inp["log_phi"], inp["d_fs"],
                              inp["keep"], inp["cols"])
            v["control_on_loader_arrays"] = verdict(ctl[blk], ob, "control")["verdict"]
            v["reading"] = ("the export differs; the control reproduces the owner"
                            if v["control_on_loader_arrays"] == "BIT_IDENTICAL"
                            else "the control differs too: environmental; held to the control")
        G.append(v)
    if paths["sffit_owner"] is not None:
        so = json.loads(paths["sffit_owner"].read_text(encoding="utf-8"))
        mine_sf = sf_summary(tab["s_f"], tab["keep_a4a"])
        keys = [k for k in mine_sf if k in so]
        G.append(verdict({k: mine_sf[k] for k in keys}, {k: so[k] for k in keys},
                         f"{paths['sffit_owner'].name} {'/'.join(keys)}"))
    else:
        G.append({"what": "mean s_f (no sffit owner: the primary gtilde was refused)",
                  "verdict": "RECORDED", "mean": float(tab["s_f"].mean())})
    fv = {s: (float(fvu["per_seed"][s]["raw"]["fvu"]), float(fvu["per_seed"][s]["standardised"]["fvu"]))
          for s in fvu["per_seed"]}
    _, rows = read_csv(out_dir / f"arm_{name}_fvu_per_seed.csv")
    back = {r[0]: (float(r[1]), float(r[2])) for r in rows}
    G.append({"what": "FVU per seed round-trips exactly", "verdict": "PASS" if back == fv else "FAIL"})
    if name == "k64":
        o9 = json.loads(paths["k64n9_arm_owner"].read_text(encoding="utf-8"))["at_0.70"]
        m9 = k64n9_block(sim, tab["log_M"], np.log10(np.maximum(tab["fire_rate"], 1e-12)),
                         tab["d_fs"], tab["keep_a4a"], anchors, cols, K64N9_USE)
        G.append(verdict(m9, o9, "ct28_arm_k64n9_ct.json at_0.70"))
        s9 = json.loads(paths["k64n9_sffit_owner"].read_text(encoding="utf-8"))
        idx = [anchors.index(u) for u in K64N9_USE]
        mine9 = sf_summary(np.stack([tab["s_f_use_0to9"][i] for i in idx]),
                           np.stack([tab["keep_a4a"][i] for i in idx]))
        keys = [k for k in mine9 if k in s9]
        G.append(verdict({k: mine9[k] for k in keys}, {k: s9[k] for k in keys},
                         f"ct28_sffit_k64n9_ct.json {'/'.join(keys)}"))
        asim, aa, ac = read_sim(out_dir / "arm_k64_sim_activation.csv.gz", np.float32)
        with h5py.File(paths["act_counts"], "r") as hf:
            ref = hf["mnn_sim_anchor_rows"][:]
        G.append({"what": "k64 activation similarity round-trips exactly",
                  "verdict": "PASS" if (np.array_equal(asim, ref) and aa == anchors and ac == cols) else "FAIL"})
        G.extend(gate_k64_replays(out_dir, inp, tab, sim, asim, anchors, cols, paths))
    g["all_pass"] = all(x["verdict"] in ("PASS", "BIT_IDENTICAL", "RECORDED") for x in G)
    return g


def gate_k64_replays(out_dir, inp, tab, sim, asim, anchors, cols, paths) -> list:
    """§20.26's activation block and §20.28 (a)'s rate-matched θ*: the
    producers' own commands (`activation_correlation fit`, `ct28_rescore arm`)
    replayed on the re-read k64 files, as phase 2a replays its families."""
    from src.analysis import activation_correlation as ac
    from src.analysis import ct28_rescore as c28
    from src.analysis.design3_join import qualifying_thetas
    ct = paths["sim"].parent
    rp = out_dir / "_replay_k64"
    rp.mkdir(parents=True, exist_ok=True)
    r_inp = {"sim": sim, "thetas": qualifying_thetas(paths["m1_json"])["qualifying"],
             "logM": tab["log_M"], "log_phi": np.log10(np.maximum(tab["fire_rate"], 1e-12)),
             "d_fs": tab["d_fs"], "keep": tab["keep_a4a"], "anchor_labels": anchors,
             "column_labels": [int(c) for c in cols], "gate": inp["gate"],
             "match_attrs": inp["match_attrs"], "gtilde_seed_order": anchors}
    o_red = json.loads((ct / "activation_correlation_k64_ct.json").read_text(encoding="utf-8"))
    red = reduce_summary(asim, len(anchors), owner_target(o_red))
    out = [verdict({k: red[k] for k in red if k in o_red}, {k: o_red[k] for k in red if k in o_red},
                   "activation_correlation_k64_ct.json (reduce)")]
    h_act = act_counts_h5(rp / "act_counts_k64.h5", asim, anchors, red["theta_star"])
    runs = [("activation_correlation_fit_k64_ct.json", ac.cmd_fit,
             dict(act_counts=h_act)),
            ("ct28_arm_k64_ct.json", c28.cmd_arm,
             dict(arm_name="k64", use=None, target=c28.K64_TARGET_N19,
                  stored_slope=paths["slope_owner"], reference_fe=None, verify_scale=True,
                  act_counts=h_act, act_reference_fe=2.403689791008184,
                  assert_theta_star_is_070=False))]
    for owner_name, fn, kw in runs:
        outp = rp / owner_name
        with patched_loader(r_inp):
            fn(SimpleNamespace(config=None, degenerate_flag=None, workers=2, out=outp, **kw))
        out.append(replay_verdict(outp, ct / owner_name, [], f"replay {owner_name}"))
    return out


# ---------------------------------------------------------------------------
# anchors: the decoder similarity at n_compare = 39, gated on Table 1's jackknife
# ---------------------------------------------------------------------------

ANCHOR_INPUTS = {"sim": "ct/jackknife_match_ct.h5",
                 "rf_counts_all_theta": "rf_counts_all_theta_n39.h5",
                 "m1_json": "rf_marginal_m1_n39.json",
                 "gtilde": "gtilde_compliant_b_ratio.h5",
                 "null_reductions": "null_pilot_compliant.h5",
                 "d_fs": "d_fs.h5"}
TABLE1_JK_THETAS = (0.55, 0.60, 0.70)


def build_anchors(out_dir: Path, root: Path = A, bundle: Path | None = None) -> dict:
    from src.experiment_config import DEFAULT_CONFIG_PATH
    paths = {k: root / v for k, v in ANCHOR_INPUTS.items()}
    inp = _load(paths, DEFAULT_CONFIG_PATH)
    f = out_dir / "sim_decoder_n39.csv.gz"
    if f.exists():
        raise SystemExit(f"REFUSING: {f} exists")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_sim(f, inp["sim"], inp["anchors"], inp["cols"])
    g = {"family": "anchors_decoder_n39", "gates": []}
    G = g["gates"]
    sim, aa, cc = read_sim(f, np.float32)
    G.append({"what": "decoder similarity round-trips exactly", "verdict":
              "PASS" if (np.array_equal(sim, inp["sim"]) and aa == inp["anchors"] and cc == inp["cols"]) else "FAIL"})
    # the features come from the 94ca8b9 bundle, re-read: the regeneration a reader runs
    b = bundle if bundle is not None else out_dir
    ft, fa = read_table(b / "features_n39.csv.gz", {"log_M": np.float64, "fire_rate": np.float64,
                                                    "d_fs": np.float64, "keep_a4a": bool})
    if fa != inp["anchors"]:
        raise SystemExit("FATAL: features_n39 anchor order differs")
    jko = json.loads((root / "ct" / "jackknife_ct.json").read_text(encoding="utf-8"))
    for t in TABLE1_JK_THETAS:
        mine = slope_block(sim, ft["log_M"], np.log10(np.maximum(ft["fire_rate"], 1e-12)),
                           ft["d_fs"], ft["keep_a4a"], cc, theta=t)
        G.append(verdict(mine["per_theta"], jko["per_theta"][f"{t:.2f}"],
                         f"jackknife_ct.json per_theta {t:.2f}"))
    g["all_pass"] = all(x["verdict"] in ("PASS", "BIT_IDENTICAL") for x in G)
    return g


# ---------------------------------------------------------------------------
# Phase 2a: the families whose producers take the join inputs. GATED BY REPLAY:
# each producer's own command runs with `load_join_inputs` patched to return the
# arrays re-read from the bundle, and with every other h5 it opens rebuilt from
# the bundle's files; its whole output JSON is compared with the owner's, the
# `identity` block (paths, hashes, host, commit) exempt by full path (gotcha
# 25 (a)), plus the named subtrees each family states.
# ---------------------------------------------------------------------------

OWNER_DIR = "ct"
REPLAY_OWNERS = {
    "activation_correlation_ct.json": "ct", "activation_correlation_fit_ct.json": "ct",
    "activation_correlation_jackknife_ct.json": "ct",
    "activation_vs_decoder_slope_curve_ct.json": "ct",
    "activation_logM_by_class_ct.json": "ct", "ct10_actA_ct.json": "ct",
    "ct10_disjoint_rows_ct.json": "ct", "ct15_disjoint_comparators_ct.json": "ct",
    "ct16_threshold_independent_ct.json": "ct", "ct17_slopes_by_firing_bin_ct.json": "ct",
    "ct20_rules_n19_ct.json": "ct", "ct20b_rules_grid_ct.json": "ct",
    "ct27_raw_encoder_ct.json": "ct", "stratum_slope_merged_ct.json": "ct",
    "jackknife_ct.json": "ct", "design3_join_aw_nofeat.json": "",
}


def drop_paths(obj, paths: list[str]):
    """A copy of a JSON object without the named paths ("/a/b", "*" matches any
    dict key at that level)."""
    obj = json.loads(json.dumps(obj))

    def rm(o, parts):
        if not parts or not isinstance(o, dict):
            return
        head, rest = parts[0], parts[1:]
        keys = list(o) if head == "*" else ([head] if head in o else [])
        for k in keys:
            if rest:
                rm(o[k], rest)
            else:
                del o[k]
    for p in paths:
        rm(obj, [x for x in p.split("/") if x])
    return obj


def replay_verdict(mine_path: Path, owner_path: Path, exempt: list[str], what: str) -> dict:
    mine = json.loads(mine_path.read_text(encoding="utf-8"))
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    ex = ["/identity"] + list(exempt)
    v = verdict(drop_paths(mine, ex), drop_paths(owner, ex), what)
    v["exempt_paths"] = ex
    return v


class patched_loader:
    """`jackknife_dictionaries.load_join_inputs` returns a shallow copy of
    `inp` for the duration (the producers mutate the dict they receive)."""

    def __init__(self, inp: dict):
        self.inp = inp

    def __enter__(self):
        from src.analysis import jackknife_dictionaries as jk
        self.orig = jk.load_join_inputs
        jk.load_join_inputs = lambda args: dict(self.inp)
        return self

    def __exit__(self, *exc):
        from src.analysis import jackknife_dictionaries as jk
        jk.load_join_inputs = self.orig
        jk._W.clear()
        return False


def anchors_inp(out_dir: Path, bundle: Path, root: Path) -> dict:
    """The primary's join inputs rebuilt from the bundle alone: the decoder
    similarity (`sim_decoder_n39`) and `features_n39`'s log M, firing rate,
    d_fs and floor; the match file's attributes and X2's qualifying thetas are
    read from the owners (metadata, not data)."""
    from src.analysis.design3_join import qualifying_thetas
    sim, anchors, cols = read_sim(out_dir / "sim_decoder_n39.csv.gz", np.float32)
    ft, fa = read_table(bundle / "features_n39.csv.gz",
                        {"log_M": np.float64, "fire_rate": np.float64, "d_fs": np.float64,
                         "keep_a4a": bool})
    if fa != anchors:
        raise SystemExit("FATAL: features_n39 and sim_decoder_n39 anchor orders differ")
    with h5py.File(root / ANCHOR_INPUTS["sim"], "r") as hf:
        at = dict(hf.attrs)
    return {"sim": sim, "thetas": qualifying_thetas(root / ANCHOR_INPUTS["m1_json"])["qualifying"],
            "logM": ft["log_M"], "log_phi": np.log10(np.maximum(ft["fire_rate"], 1e-12)),
            "d_fs": ft["d_fs"], "keep": ft["keep_a4a"], "anchor_labels": anchors,
            "column_labels": [int(c) for c in cols], "gate": json.loads(at["gate"]),
            "match_attrs": {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                            for k, v in at.items()},
            "gtilde_seed_order": anchors, "fire_rate": ft["fire_rate"]}


def _h5(path: Path, datasets: dict, attrs: dict) -> Path:
    with h5py.File(path, "w") as ho:
        for k, v in datasets.items():
            ho.create_dataset(k, data=v)
        ho.attrs.update(attrs)
    return path


def act_counts_h5(path: Path, sim: np.ndarray, anchors: list, ts: dict) -> Path:
    """`activation_correlation.cmd_reduce`'s count file, rebuilt from a re-read
    similarity and its re-derived theta*."""
    from src.analysis.activation_correlation import THETA_GRID as AG
    from src.analysis.jackknife_dictionaries import counts_from_sim
    ds = {f"match_count_theta_{t:.2f}": counts_from_sim(sim, t) for t in AG}
    if ts.get("theta_star") is not None:
        ds["match_count_theta_star"] = counts_from_sim(sim, ts["theta_star"])
    ds["mnn_sim_anchor_rows"] = sim
    return _h5(path, ds, {"n_compare_seeds": int(sim.shape[1] - 1),
                          "seed_labels": json.dumps([int(a) for a in anchors]),
                          "theta_star": json.dumps(ts)})


def owner_target(o: dict) -> float:
    """θ*'s target rate as the owner recorded it: `identity.theta_star_target_rate`
    where the producer wrote one (from §CT24 on), else the target inside the
    owner's own `theta_star` block (the earlier files)."""
    t = o.get("identity", {}).get("theta_star_target_rate")
    return float(t if t is not None else o["theta_star"]["target"])


def reduce_summary(sim: np.ndarray, n_anchors: int, target: float) -> dict:
    """`cmd_reduce`'s reported numbers, from the producer's own functions."""
    from src.analysis import activation_correlation as ac
    curve = {f"{t:.2f}": ac.mean_rate(sim, t) for t in ac.THETA_GRID}
    return {"mean_match_rate_by_theta": curve,
            "mean_match_rate_at_numeric_0.70": curve["0.70"],
            "anchor_anchor_symmetry": ac.symmetry_report(sim, n_anchors),
            "theta_star": ac.theta_star(sim, target)}


def square_to_anchors(path: Path, ds: str, anchors: list) -> np.ndarray:
    from src.analysis.ct_robustness import _anchor_square
    return _anchor_square(path, ds, anchors)


def build_replay(out_dir: Path, root: Path = A, bundle: Path | None = None) -> dict:
    """Phase 2a's exports and their replay gates. Needs `sim_decoder_n39.csv.gz`
    (the anchors step) in `out_dir` and `features_n39.csv.gz` in `bundle`."""
    from src.analysis import activation_correlation as ac
    from src.analysis import ct_robustness as cr
    from src.analysis import stratum_slope as ss
    from src.analysis.seed_axis import align_to_reference
    from src.experiment_config import DEFAULT_CONFIG_PATH
    bundle = bundle if bundle is not None else out_dir
    ct = root / "ct"
    rp = out_dir / "_replay"
    rp.mkdir(parents=True, exist_ok=True)
    g = {"family": "replay", "gates": []}
    G = g["gates"]

    # ---- the owners, verbatim -------------------------------------------
    for name, sub in REPLAY_OWNERS.items():
        src = (root / sub / name) if sub else (root / name)
        dst = out_dir / f"owner__{name}"
        if dst.exists():
            raise SystemExit(f"REFUSING: {dst} exists")
        dst.write_bytes(src.read_bytes())
        G.append({"what": f"owner__{name} is the owner's bytes", "verdict":
                  "PASS" if sha256_file(dst) == sha256_file(src) else "FAIL",
                  "sha256": sha256_file(dst)})
    own = lambda n: out_dir / f"owner__{n}"

    inp = anchors_inp(out_dir, bundle, root)
    anchors = inp["anchor_labels"]
    cols = [str(c) for c in inp["column_labels"]]

    # ---- exports ----------------------------------------------------------
    with h5py.File(ct / "activation_correlation_counts_ct.h5", "r") as hf:
        act0 = hf["mnn_sim_anchor_rows"][:]
        lab = [str(s) for s in json.loads(hf.attrs["seed_labels"])]
    with h5py.File(ct / "ct10_actA_counts_ct.h5", "r") as hf:
        half0 = hf["mnn_sim_anchor_rows"][:]
        hlab = [str(s) for s in json.loads(hf.attrs["seed_labels"])]
    if lab != anchors or hlab != anchors:
        raise SystemExit("FATAL: an activation file's anchor order differs")
    with h5py.File(ct / "ct16_oneway_max_cos_ct.h5", "r") as hf:
        one0 = hf["oneway_max_cos_anchor_rows"][:]
        olab = [str(s) for s in json.loads(hf.attrs["anchor_seed_labels"])]
        o_gate = str(hf.attrs["gate_vs_ct3_mnn_sim"])
    if olab != anchors:
        raise SystemExit("FATAL: the one-way file's anchor order differs")
    rules0 = {k: square_to_anchors(ct / "ct20_rules_sim_ct.h5", f"sim_{ds}", anchors)
              for k, ds in cr.RULE_DATASETS.items()}
    var0 = square_to_anchors(ct / "ct27_raw_encoder_sim_ct.h5",
                             "sim_ct27_raw_encoder_dec070", anchors)
    files = {"sim_activation_n39.csv.gz": (act0, anchors, cols, "-inf", np.float32),
             "sim_activation_halfA_n39.csv.gz": (half0, anchors, cols, "-inf", np.float32),
             "sim_oneway_n39.csv.gz": (one0, anchors, cols, "nan", np.float32),
             "sim_rule3_rawdot_n19.csv.gz": (var0, anchors, anchors, "-inf", np.float64)}
    for k, s in rules0.items():
        files[f"sim_{k}_n19.csv.gz"] = (s, anchors, anchors, "-inf", np.float64)
    for name, (s, a, c, _, _) in files.items():
        if (out_dir / name).exists():
            raise SystemExit(f"REFUSING: {out_dir / name} exists")
        write_sim(out_dir / name, s, a, c)
    Ms = {}
    for key, path in (("M_half_a", ct / "gtilde_ct10_half_a_ct.h5"),
                      ("M_half_b", ct / "gtilde_ct10_half_b_ct.h5"),
                      ("M_stratum_experimental_merged", ct / "gtilde_shared_experimental_merged_ct.h5"),
                      ("M_stratum_propagated_merged", ct / "gtilde_shared_propagated_merged_ct.h5")):
        with h5py.File(path, "r") as hf:
            labs = [(s.decode() if isinstance(s, bytes) else str(s)).split("_")[-1]
                    for s in hf["seeds"][:]]
            Ms[key] = align_to_reference(labs, hf["M"][:], anchors)
    f_m = out_dir / "features_logM_variants_n39.csv.gz"
    write_table(f_m, anchors, Ms)

    # ---- re-read --------------------------------------------------------
    back = {n: read_sim(out_dir / n, dt, own_fill=fill) for n, (_, _, _, fill, dt) in files.items()}
    rt = [n for n, (s, a, c, _, _) in files.items()
          if not (np.array_equal(back[n][0], s, equal_nan=True) and back[n][1] == [str(x) for x in a]
                  and back[n][2] == [str(x) for x in c])]
    Mb, _ = read_table(f_m, {k: np.float64 for k in Ms})
    rt += [k for k in Ms if not np.array_equal(Mb[k], Ms[k])]
    G.append({"what": "every phase-2a file round-trips exactly", "verdict":
              "PASS" if not rt else "FAIL", "differing": rt})
    act, half, one = (back[n][0] for n in ("sim_activation_n39.csv.gz",
                                           "sim_activation_halfA_n39.csv.gz",
                                           "sim_oneway_n39.csv.gz"))
    rules = {k: back[f"sim_{k}_n19.csv.gz"][0] for k in cr.RULE_DATASETS}
    var = back["sim_rule3_rawdot_n19.csv.gz"][0]

    # ---- rebuilt inputs ---------------------------------------------------
    o_ac = json.loads(own("activation_correlation_ct.json").read_text(encoding="utf-8"))
    red = reduce_summary(act, len(anchors), owner_target(o_ac))
    G.append(verdict(red, {k: o_ac[k] for k in red}, "activation_correlation_ct.json (reduce)"))
    o_ha = json.loads(own("ct10_actA_ct.json").read_text(encoding="utf-8"))
    red_h = reduce_summary(half, len(anchors), owner_target(o_ha))
    G.append(verdict({k: red_h[k] for k in red_h if k in o_ha}, {k: o_ha[k] for k in red_h if k in o_ha},
                     "ct10_actA_ct.json (reduce)"))
    h_act = act_counts_h5(rp / "act_counts.h5", act, anchors, red["theta_star"])
    h_half = act_counts_h5(rp / "act_half_counts.h5", half, anchors, red_h["theta_star"])
    with h5py.File(root / "rule_interior_mass_co.h5", "r") as hf:
        r_attrs = {k: hf.attrs[k] for k in ("n_compare_seeds", "theta", "device_class")
                   if k in hf.attrs}
    from src.analysis.ct_robustness import subset_counts
    h_rules = _h5(rp / "rules_counts.h5",
                  {ds: subset_counts(rules[k], 0.70, anchors, anchors, anchors)[0]
                   for k, ds in cr.RULE_DATASETS.items()},
                  {**r_attrs, "seed_labels": json.dumps([int(a) for a in anchors])})
    h_rsim = _h5(rp / "rules_sim.h5", {f"sim_{ds}": rules[k] for k, ds in cr.RULE_DATASETS.items()},
                 {"anchor_seed_labels": json.dumps([int(a) for a in anchors])})
    h_var = _h5(rp / "variant_sim.h5", {"sim_ct27_raw_encoder_dec070": var},
                {"anchor_seed_labels": json.dumps([int(a) for a in anchors])})
    h_one = _h5(rp / "oneway.h5", {"oneway_max_cos_anchor_rows": one},
                {"anchor_seed_labels": json.dumps([int(a) for a in anchors]),
                 "gate_vs_ct3_mnn_sim": o_gate})
    from src.analysis.design3_join import firing_rate_edges
    edges = firing_rate_edges(root / ANCHOR_INPUTS["gtilde"])
    h_gt = _h5(rp / "edges.h5", {}, {"firing_rate_edges": json.dumps(edges)})
    h_null = _h5(rp / "fire_rate.h5", {"fire_rate": inp["fire_rate"],
                                       "seeds": np.array([f"seed_{a}" for a in anchors], dtype="S")}, {})
    h_ga = _h5(rp / "gt_half_a.h5", {"M": Mb["M_half_a"],
                                     "seeds": np.array([f"seed_{a}" for a in anchors], dtype="S")}, {})
    h_gb = _h5(rp / "gt_half_b.h5", {"M": Mb["M_half_b"],
                                     "seeds": np.array([f"seed_{a}" for a in anchors], dtype="S")}, {})
    h_se = _h5(rp / "gt_exp.h5", {"M": Mb["M_stratum_experimental_merged"],
                                  "seeds": np.array([f"seed_{a}" for a in anchors], dtype="S")}, {})
    h_sp = _h5(rp / "gt_prop.h5", {"M": Mb["M_stratum_propagated_merged"],
                                   "seeds": np.array([f"seed_{a}" for a in anchors], dtype="S")}, {})
    pairs = sorted((Path(__file__).resolve().parents[2] / "analysis/adhoc/probes/2026-10-22/ct/pairs")
                   .glob("pairs_anchor_*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    ref_join = own("design3_join_aw_nofeat.json")

    # ---- replays ----------------------------------------------------------
    def ns(**kw):
        return SimpleNamespace(config=DEFAULT_CONFIG_PATH, degenerate_flag=None, workers=2, **kw)

    runs = [
        ("activation_correlation_fit_ct.json", ac.cmd_fit, dict(act_counts=h_act), []),
        ("activation_correlation_jackknife_ct.json", ac.cmd_jackknife,
         dict(act_counts=h_act, fit_json=rp / "activation_correlation_fit_ct.json"), []),
        ("activation_vs_decoder_slope_curve_ct.json", ac.cmd_curve, dict(act_counts=h_act), []),
        ("activation_logM_by_class_ct.json", ac.cmd_classes,
         dict(act_counts=h_act, act_json=own("activation_correlation_ct.json"), pairs=pairs,
              theta_dec=0.70), []),
        ("ct15_disjoint_comparators_ct.json", cr.cmd_ct15,
         dict(jackknife_json=own("jackknife_ct.json"), reference_join=ref_join), []),
        ("ct20_rules_n19_ct.json", cr.cmd_ct20,
         dict(act_counts=h_act, act_json=own("activation_correlation_ct.json"), rules_h5=h_rules,
              reference_join=ref_join), []),
        ("ct20b_rules_grid_ct.json", cr.cmd_ct20b,
         dict(act_counts=h_act, act_json=own("activation_correlation_ct.json"), rules_h5=h_rules,
              rules_sim_h5=h_rsim, reference_join=ref_join), []),
        ("ct27_raw_encoder_ct.json", cr.cmd_ct27,
         dict(variant_sim_h5=h_var, rules_sim_h5=h_rsim, grid_json=rp / "ct20b_rules_grid_ct.json",
              reference_join=ref_join), []),
        ("ct17_slopes_by_firing_bin_ct.json", cr.cmd_ct17,
         dict(act_counts=h_act, act_json=own("activation_correlation_ct.json"), gtilde=h_gt,
              null_reductions=h_null, reference_join=ref_join), []),
        ("ct16_threshold_independent_ct.json", cr.cmd_ct16_fit,
         dict(oneway_h5=h_one, reference_join=ref_join), []),
        ("ct10_disjoint_rows_ct.json", cr.cmd_ct10_fit,
         dict(act_half_counts=h_half, act_half_json=own("ct10_actA_ct.json"), gtilde_half_a=h_ga,
              gtilde_half_b=h_gb, reference_join=ref_join),
         ["/half_A_gtilde", "/half_B_gtilde"]),
        ("stratum_slope_merged_ct.json", ss.cmd_fit,
         dict(theta=0.70, strata=["experimental", "propagated"], gtilde_strata=[h_se, h_sp]),
         ["/strata/*/gtilde", "/strata/*/gtilde_attrs", "/strata/*/description"]),
    ]
    for owner_name, fn, kw, exempt in runs:
        outp = rp / owner_name
        with patched_loader(inp):
            fn(ns(out=outp, **kw))
        G.append(replay_verdict(outp, own(owner_name), exempt, f"replay {owner_name}"))
        print(f"[replay] {G[-1]['verdict']:<14} {owner_name}", flush=True)
    g["all_pass"] = all(x["verdict"] in ("PASS", "BIT_IDENTICAL") for x in G)
    return g


# ---------------------------------------------------------------------------
# Phase 2b: the carrier excess with the concentration covariate (§20.17), the
# disorder shares (§20.23), the classification (§20.6), the concept subspaces
# (§20.21, §20.22, §20.25; RECOUNTED), and every other owner JSON a camera-ready
# number is read from (RE-READ), with the record's hashes checked.
# ---------------------------------------------------------------------------

LABELS_H5 = "residue_labels_subtyped_untruncated_pgate5_f67485d.h5"
SCRATCH = Path("$SCRATCH")


def _bundle_carrier_inputs(bundle: Path) -> dict:
    """The carrier inputs exactly as `scripts/reproduce_table1.py` reads them,
    in the AP's lexicographic seed order (`meta.json`)."""
    from src.analysis.release_derived import partner_surrogate
    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    order = [int(s) for s in meta["carrier_pair_order_lexicographic"]]
    ft, anchors = read_table(bundle / "features_n39.csv.gz",
                             {"fire_count_ap": np.int64, "d_fs": np.float64})
    ih, irows = read_csv(bundle / "mnn_indicator_theta070.csv.gz")
    cols = [int(h[2:]) for h in ih[2:]]
    ind = np.zeros((len(anchors), len(cols), len(irows) // len(anchors)), dtype=bool)
    for r in irows:                                   # reproduce_table1.load_bundle's reading
        ind[anchors.index(r[0]), :, int(r[1])] = [v == "1" for v in r[2:]]
    head, rows = read_csv(bundle / "concepts_carriers.csv.gz")
    c = {h: i for i, h in enumerate(head)}
    carrier, detected = {}, {}
    for r in rows:
        s = int(r[c["anchor_seed"]])
        carrier.setdefault(s, []).append(int(r[c["carrier_feature"]]))
        detected.setdefault(s, []).append(r[c["detected"]] == "1")
    lex = [anchors.index(str(s)) for s in order]
    return {"order": order, "anchors": anchors, "lex": lex,
            "carrier": np.array([carrier[s] for s in order], dtype=np.int64),
            "detected": np.array([detected[s] for s in order], dtype=bool),
            "partner": partner_surrogate(ind, order, anchors, cols),
            "fire_count": ft["fire_count_ap"][lex], "d_fs": ft["d_fs"][lex],
            "n_rows": int(meta["n_rows_ap"])}


def carrier_arms(ci: dict, conc_lex: np.ndarray | None) -> dict:
    """`carrier_jackknife.run`'s arms from bundle inputs: the cells, then
    `per_pair_cell_stats` and `intervals` (pure)."""
    from src.analysis import carrier_concentration as cc
    from src.analysis import carrier_jackknife as cj
    fbin = cc.firing_bin(ci["fire_count"].astype(np.float64) / float(ci["n_rows"]))
    n_f = len(cc.BIN_EDGES) - 1
    cells = {"single_firing_only": (cc.cell_index(fbin, None, cc.N_SUPPORT_BINS), n_f)}
    if conc_lex is not None:
        sbin = cc.support_bin_within(fbin, ci["d_fs"], cc.N_SUPPORT_BINS)
        cbin = cc.support_bin_within(fbin, conc_lex, cc.N_CONC_BINS)
        cells["triple_firing_x_support_x_concentration"] = (
            (np.asarray(fbin, dtype=np.int64) * cc.N_SUPPORT_BINS
             + np.asarray(sbin, dtype=np.int64)) * cc.N_CONC_BINS
            + np.asarray(cbin, dtype=np.int64), n_f * cc.N_SUPPORT_BINS * cc.N_CONC_BINS)
    out = {}
    for name, (cell, nc) in cells.items():
        st = cj.per_pair_cell_stats(ci["carrier"], ci["detected"], ci["partner"], cell, nc)
        out[name] = cj.intervals(st, ci["carrier"].shape[0])
    return out


def build_carrier(out_dir: Path, root: Path, bundle: Path, g: list) -> None:
    from src.analysis import carrier_concentration as cc
    ci = _bundle_carrier_inputs(bundle)
    with h5py.File(root / "feature_ap_main_compliant.h5", "r") as hf:
        ap_seeds = cc._decode(hf["seeds"][:])
        concepts = cc._decode(hf["pooled/concepts"][:])
        in_pool = hf["pooled/in_argmax_pool"][:].astype(bool)
        n_feat = int(hf["pooled/ap"].shape[1])
        n_rows = int(cc._attrs(hf)["n_rows"])
    if [f"seed_{s}" for s in ci["order"]] != ap_seeds:
        raise SystemExit("FATAL: the bundle's carrier order is not the AP's")
    labels = root / LABELS_H5
    _, conc, prov = cc.build_feature_covariates(
        ap_seeds, PROJECT / "sae_ckpts", SCRATCH / "row_index_cache", SCRATCH / "cache" / "eval.h5",
        labels, labels.with_suffix(".json"), [c for c, k in zip(concepts, in_pool) if k],
        n_rows, n_feat, "label_store", 32)
    g.append({"what": "concentration: the covariate builder's seed provenance",
              "verdict": "PASS" if not prov["problems"] else "FAIL", "problems": prov["problems"]})
    num = np.empty_like(conc)
    for i, s in enumerate(ci["order"]):
        num[ci["anchors"].index(str(s))] = conc[i]
    f = out_dir / "features_carrier_n39.csv.gz"
    write_table(f, ci["anchors"], {"concentration": num})
    back, _ = read_table(f, {"concentration": np.float64})
    g.append({"what": "concentration round-trips exactly", "verdict":
              "PASS" if np.array_equal(back["concentration"], num, equal_nan=True) else "FAIL"})
    mine = carrier_arms(ci, back["concentration"][ci["lex"]])
    owner = json.loads((root / "ct" / "carrier_jackknife_ct.json").read_text(encoding="utf-8"))["arms"]
    for name, blk in mine.items():
        g.append(verdict(blk, {k: v for k, v in owner[name].items() if k != "control"},
                         f"carrier_jackknife_ct.json arms/{name} (all but its control)"))


DISORDERED = "REGION:Disordered"


def build_disorder(out_dir: Path, root: Path, bundle: Path, g: list) -> None:
    from src.analysis import ct23_disorder_shares as d23
    from src.analysis.seed_axis import align_to_reference
    with h5py.File(root / "feature_ap_main_compliant.h5", "r") as hf:
        seeds = [s.decode() if isinstance(s, bytes) else str(s) for s in hf["seeds"][:]]
        concepts = [c.decode() if isinstance(c, bytes) else str(c) for c in hf["pooled/concepts"][:]]
        in_pool = hf["pooled/in_argmax_pool"][:].astype(bool)
        ap_pool = hf["pooled/ap"][:][:, :, in_pool]
        prev = hf["pooled/prevalence"][:][in_pool]
    pool = [c for c, k in zip(concepts, in_pool) if k]
    p = prev.astype(np.float64)
    ja, jg, mg = [], [], []
    for s in range(ap_pool.shape[0]):
        gg = (ap_pool[s].astype(np.float64) - p) / (1.0 - p)
        j = gg.argmax(axis=1)
        jg.append(j)
        mg.append(gg[np.arange(gg.shape[0]), j])
        ja.append(ap_pool[s].argmax(axis=1))
    labs = [s.split("_")[-1] for s in seeds]
    ft, anchors = read_table(bundle / "features_n39.csv.gz", {})
    _, rows = read_csv(bundle / "features_n39.csv.gz")
    head, _ = read_csv(bundle / "features_n39.csv.gz")
    zc = np.array([r[head.index("argmax_concept")] for r in rows]).reshape(len(anchors), -1)
    name = np.array(pool, dtype=object)
    cols = {"argmax_raw_ap_pool": align_to_reference(labs, name[np.stack(ja)], anchors),
            "argmax_gain_pool": align_to_reference(labs, name[np.stack(jg)], anchors),
            "max_gain_pool": align_to_reference(labs, np.stack(mg), anchors)}
    f = out_dir / "features_disorder_n39.csv.gz"
    names = list(cols)
    write_csv_gz(f, ["anchor_seed", "feature"] + names,
                 ([a, i, cols["argmax_raw_ap_pool"][ai, i], cols["argmax_gain_pool"][ai, i],
                   repr(float(cols["max_gain_pool"][ai, i]))]
                  for ai, a in enumerate(anchors) for i in range(cols["max_gain_pool"].shape[1])))
    h2, r2 = read_csv(f)
    back = {k: np.array([r[h2.index(k)] for r in r2]).reshape(len(anchors), -1) for k in names}
    back["max_gain_pool"] = back["max_gain_pool"].astype(np.float64)
    ok = (np.array_equal(back["max_gain_pool"], cols["max_gain_pool"])
          and all(np.array_equal(back[k], cols[k].astype(str)) for k in names[:2]))
    g.append({"what": "disorder columns round-trip exactly", "verdict": "PASS" if ok else "FAIL"})
    import yaml
    from src.experiment_config import DEFAULT_CONFIG_PATH
    floor = d23._find_key(yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8")),
                          "alignment_min_normalised_gain")
    if floor is None:
        raise SystemExit("alignment_min_normalised_gain ABSENT from the config")
    al = back["max_gain_pool"] >= floor
    isd = lambda x: x == DISORDERED                      # noqa: E731
    n = {"n_cells": int(al.size), "n_aligned": int(al.sum()),
         "gain_dis_all": int(isd(back["argmax_gain_pool"]).sum()),
         "gain_dis_aligned": int((isd(back["argmax_gain_pool"]) & al).sum()),
         "z_dis_all": int(isd(zc).sum()), "z_dis_aligned": int((isd(zc) & al).sum()),
         "ap_dis_all": int(isd(back["argmax_raw_ap_pool"]).sum()),
         "ap_dis_aligned": int((isd(back["argmax_raw_ap_pool"]) & al).sum())}
    owner = json.loads((root / "ct" / "ct23_disorder_aligned_ct.json").read_text(encoding="utf-8"))
    g.append(verdict(n, owner["counts"], "ct23_disorder_aligned_ct.json counts (recounted)"))
    g.append({"what": "the record's five figures (ct23_disorder_shares.RECORD)",
              "verdict": "PASS" if not d23.check_record(n) else "FAIL",
              "differing": d23.check_record(n)})


def build_classification(out_dir: Path, root: Path, g: list) -> None:
    from src.analysis.classification_2x2 import CELL_NAMES as _cn
    from src.analysis.seed_axis import align_to_reference
    cols, owners = {}, {}
    anchors = None
    for tag, stem in (("g005", "classification_2x2_control_ct"),
                      ("g001", "classification_2x2_g001_ct")):
        with h5py.File(root / "ct" / f"{stem}.h5", "r") as hf:
            seeds = [s.decode() if isinstance(s, bytes) else str(s) for s in hf["seeds"][:]]
            labs = [s.split("_")[-1] for s in seeds]
            anchors = anchors or sorted(labs, key=int)
            for arm in ("a", "b"):
                cols[f"class_{arm}_{tag}"] = align_to_reference(labs, hf[f"cell_{arm}"][:], anchors)
        owners[tag] = json.loads((root / "ct" / f"{stem}.json").read_text(encoding="utf-8"))
    f = out_dir / "features_classification_n39.csv.gz"
    write_table(f, anchors, cols)
    back, _ = read_table(f, {k: np.int64 for k in cols})
    g.append({"what": "classification codes round-trip exactly", "verdict":
              "PASS" if all(np.array_equal(back[k], cols[k]) for k in cols) else "FAIL"})
    for tag, o in owners.items():
        for arm in ("a", "b"):
            want = {r["seed"].split("_")[-1]: {"cells": r["cells"], "n_ALIGNED": r["n_ALIGNED"],
                                                "n_EXCLUDED": r["n_EXCLUDED"], "n_ORPHAN": r["n_ORPHAN"],
                                                "n_NOT_EVALUABLE": r["n_NOT_EVALUABLE"]}
                    for r in o["per_arm"][arm.upper()]}
            got = {}
            for i, a in enumerate(anchors):
                v = back[f"class_{arm}_{tag}"][i]
                cnt = {c: int((v == c).sum()) for c in sorted(_cn)}
                got[a] = {"cells": {_cn[c]: cnt[c] for c in sorted(cnt)},
                          "n_ALIGNED": cnt[0] + cnt[1], "n_EXCLUDED": cnt[2],
                          "n_ORPHAN": cnt[3], "n_NOT_EVALUABLE": cnt[4]}
            g.append(verdict(got, want, f"classification {tag} arm {arm.upper()} per seed (recounted)"))


SUBSPACE = (("ct22_confirm_ct", "m3_ap"), ("ct26_nonanchor_ct", None),
            ("ct26_anchor_control_ct", None))


def _subspace_rows(path: Path, group: str | None):
    from src.analysis.concept_subspace import MEASURES
    with h5py.File(path, "r") as hf:
        names = [k for k in hf if isinstance(hf[k], h5py.Group)]
        grp = group if group is not None else (names[0] if len(names) == 1 else None)
        if grp is None:
            raise SystemExit(f"FATAL: {path.name} has groups {names}; name one")
        g = hf[grp]
        cols = ["cell", "c", "s", "t", "feasible", "rank_ok", "alignment_ok", "tie",
                "match_fraction"]
        for m in MEASURES:
            cols += [m] + [f"{m}_{x}" for x in ("null_mean", "null_sd", "separation", "exceedance")]
        return grp, {k: g[k][:] for k in cols}


def subspace_recount(t: dict, pool: list) -> dict:
    """`concept_subspace.per_concept` / `cell_accounting` / `confirm_summary`
    on the stored per-cell rows, omitting the two fields the rows do not keep
    (each cell's `beyond_95` and its null draws' rank-deficiency count)."""
    from src.analysis import concept_subspace as cs
    meas = "omega"
    rows = [{"c": int(t["c"][i]), "feasible": bool(t["feasible"][i]), "rank_ok": bool(t["rank_ok"][i]),
             "alignment_ok": bool(t["alignment_ok"][i]), "tie": bool(t["tie"][i]),
             "match_fraction": float(t["match_fraction"][i]), meas: float(t[meas][i]),
             meas + "_null": {f: float(t[f"{meas}_{f}"][i])
                              for f in ("null_mean", "null_sd", "separation", "exceedance")}}
            for i in range(len(t["c"]))]
    for r in rows:
        r[meas + "_null"]["beyond_95"] = False          # not stored; its outputs are dropped below
    conc = cs.per_concept(rows, meas, pool)
    acc = cs.cell_accounting(rows, meas)
    s = cs.confirm_summary(conc, acc, meas)
    for r in s["per_concept"]:
        r.pop("fraction_beyond_95", None)
    s["accounting"].pop("n_null_draws_rank_deficient", None)
    s["distributions"].pop("fraction_beyond_95", None)
    has = [r for r in s["per_concept"] if "median_exceedance" in r]
    s["CT26_P3_share"] = (sum(r["median_exceedance"] >= 0.95 for r in has) / len(has)) if has else None
    return s


def build_subspace(out_dir: Path, root: Path, g: list) -> None:
    with h5py.File(root / "feature_ap_main_compliant.h5", "r") as hf:
        concepts = [c.decode() if isinstance(c, bytes) else str(c) for c in hf["pooled/concepts"][:]]
        in_pool = hf["pooled/in_argmax_pool"][:].astype(bool)
    pool = [c for c, k in zip(concepts, in_pool) if k]
    for stem, group in SUBSPACE:
        grp, t = _subspace_rows(root / "ct" / f"{stem}.h5", group)
        f = out_dir / f"subspace_{stem}__{grp}.csv.gz"
        n = len(t["c"])
        names = list(t)
        txt = {k: text_of(np.asarray(v)) for k, v in t.items()}
        write_csv_gz(f, names, ([txt[k][i] for k in names] for i in range(n)))
        head, rows = read_csv(f)
        dt = {k: (bool if np.asarray(v).dtype == np.bool_ else np.asarray(v).dtype) for k, v in t.items()}
        back = {k: (np.array([r[head.index(k)] == "1" for r in rows]) if dt[k] is bool
                    else np.array([r[head.index(k)] for r in rows], dtype=dt[k])) for k in names}
        ok = all(np.array_equal(back[k], np.asarray(t[k]), equal_nan=(dt[k] is not bool and
                                                                      np.dtype(dt[k]).kind == "f"))
                 for k in names)
        g.append({"what": f"subspace {stem}/{grp} round-trips exactly", "verdict": "PASS" if ok else "FAIL"})
        mine = subspace_recount(back, pool)
        owner = json.loads((root / "ct" / f"{stem}.json").read_text(encoding="utf-8"))
        want = {}
        for k in ("accounting", "per_concept", "distributions", "crosstab", "CT22_P1", "CT22_P2",
                  "n_concepts_with_a_measurable_cell", "share_subspace_reproduces"):
            if k in owner:
                want[k] = owner[k]
        if "CT26_P3" in owner:
            want["CT26_P3_share"] = owner["CT26_P3"]["share_concepts_median_exceedance_ge_0_95"]
        want = drop_paths(want, ["/accounting/n_null_draws_rank_deficient",
                                 "/distributions/fraction_beyond_95"])
        for r in want.get("per_concept", []):
            r.pop("fraction_beyond_95", None)
        got = {k: mine[k] for k in want if k in mine}
        g.append(verdict(got, want, f"{stem}.json recounted ({', '.join(sorted(want))})"))


# Every owner JSON a camera-ready number is read from (the edit list's §(f)),
# shipped verbatim beside the replay owners: RE-READ level for the numbers
# whose inputs are not derived data, and the reference for every recount.
OWNERS_REREAD = [
    "ct/carrier_jackknife_ct.json", "carrier_concentration_bb.json",
    "ct/carrier_jackknife_k64_ct.json", "ct/ct23_disorder_aligned_ct.json",
    "ct/classification_2x2_g001_ct.json", "ct/classification_2x2_control_ct.json",
    "ct/ct22_confirm_ct.json", "ct/ct22_pilot_ct.json", "ct/ct22_diagnostics_ct.json",
    "ct/ct22_diagnostics_ap_argmax_ct.json", "ct/ct26_nonanchor_ct.json",
    "ct/ct26_anchor_control_ct.json", "ct/ct21_figure_data_ct.json",
    "ct/activation_correlation_k64_ct.json", "ct/activation_correlation_fit_k64_ct.json",
    "ct/ct28_score_ct.json", "ct/ct28_rules_k64_ct.json", "ct/ct28_arm_k64_ct.json",
    "ct/ct28_arm_k64n9_ct.json", "ct/ct28_arm_k32n19_ct.json", "ct/ct28_sffit_k64n9_ct.json",
    "ct/ct29_score_ct.json", "ct/ct25_width_tier_ct.json", "ct/slope_k64_ct.json",
    "ct/rf_marginal_m1_k64_ct.json", "ct/fvu_eval_k64_ct.json", "fvu_eval_cf.json",
    "cj_rule_sensitivity.json", "cl_literal_criterion.json", "rf_marginal_m1_n39.json",
] + [f"ct/slope_{a}_ct.json" for a in ("w2560", "w10240", "w20480", "w40960", "k64_w2560", "k64_w40960")] \
  + [f"ct/slope_{a}_topmerged_ct.json" for a in ("k16_w10240", "k128_w10240")] \
  + [f"ct/ct28_sffit_{a}_ct.json" for a in ("w2560", "w10240", "w20480", "w40960", "k64", "k64_w2560", "k64_w40960")] \
  + [f"cs/rf_marginal_m1_{w}_n9.json" for w in ("w2560", "w10240", "w20480", "w40960")] \
  + [f"cs/fvu_eval_{w}.json" for w in ("w2560", "w10240", "w20480", "w40960")] \
  + [f"ct/rf_marginal_m1_{a}_ct.json" for a in ("k64_w2560", "k64_w40960", "k16_w10240", "k128_w10240")] \
  + [f"ct/fvu_eval_{a}_ct.json" for a in ("k64_w2560", "k64_w40960", "k16_w10240", "k128_w10240")]


def record_hash_check(name: str, sha: str, texts: dict, window: int = 2) -> dict:
    """Does the record state this file's hash? Each line naming the file is read
    with the `window` lines after it (NoR often puts an owner's hash on the next
    line). A candidate is a 16- or 64-digit hex run with at least one letter a-f,
    not preceded by a digit, a letter a-f or a decimal point (so a decimal like
    0.4594441105769231 is not one, and a 40-digit git sha is not one).
    MATCH: a candidate is a prefix of ours. MISMATCH (a stop): the line naming
    the file, alone, states exactly one candidate and it is not ours. UNRESOLVED:
    candidates near the name, none ours, not a single-hash line. NOT_STATED: none."""
    import re
    pat = re.compile(r"(?<![0-9a-f.])(?=[0-9a-f]*[a-f])[0-9a-f]{16}(?:[0-9a-f]{48})?(?![0-9a-f])")
    hexes, single_bad = [], []
    for doc, txt in texts.items():
        lines = txt.splitlines()
        for k, ln in enumerate(lines):
            if name not in ln:
                continue
            own = pat.findall(ln)
            if len(own) == 1 and ln.count(name) == 1 and not sha.startswith(own[0]):
                single_bad.append(f"{doc}: {ln.strip()[:160]}")
            hexes += own
            for nxt in lines[k + 1:k + 1 + window]:
                hexes += pat.findall(nxt)
    if any(sha.startswith(h) for h in hexes):
        v = "MATCH"
    elif single_bad:
        v = "MISMATCH"
    elif hexes:
        v = "UNRESOLVED"
    else:
        v = "NOT_STATED"
    return {"what": f"record hash for {name}", "verdict": v, "sha256": sha,
            "mismatch_lines": single_bad[:3]}


def record_texts(repo: Path) -> dict:
    """The two record texts the stated-hash check searches, keyed by the name each is cited
    under. The pre-registration log is `docs/record/preregistration/` read as one text; it was
    the single file `PREREGISTRATION.md` until section CT18-D10 split it."""
    from src.analysis.prereg_log import read_log
    return {"docs/NUMBERS_OF_RECORD.md": (repo / "docs/NUMBERS_OF_RECORD.md").read_text(encoding="utf-8"),
            "PREREGISTRATION.md": read_log(repo / "docs" / "record" / "preregistration")}


def build_owners(out_dir: Path, root: Path, repo: Path, g: list) -> None:
    texts = record_texts(repo)
    shipped = sorted(p.name[len("owner__"):] for p in out_dir.glob("owner__*.json"))
    for rel in OWNERS_REREAD:
        src = root / rel
        name = Path(rel).name
        dst = out_dir / f"owner__{name}"
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
        g.append({"what": f"owner__{name} is the owner's bytes",
                  "verdict": "PASS" if sha256_file(dst) == sha256_file(src) else "FAIL"})
    for p in sorted(out_dir.glob("owner__*.json")):
        name = p.name[len("owner__"):]
        g.append(record_hash_check(name, sha256_file(p), texts))
    g.append({"what": "owner files shipped", "verdict": "RECORDED",
              "n": len(list(out_dir.glob("owner__*.json"))), "before_this_step": shipped})


def build_phase2b(out_dir: Path, root: Path = A, bundle: Path | None = None) -> dict:
    from src.experiment_config import REPO_ROOT
    bundle = bundle if bundle is not None else out_dir
    g = {"family": "phase2b", "gates": []}
    G = g["gates"]
    for step in (lambda: build_classification(out_dir, root, G),
                 lambda: build_disorder(out_dir, root, bundle, G),
                 lambda: build_subspace(out_dir, root, G),
                 lambda: build_owners(out_dir, root, REPO_ROOT, G),
                 lambda: build_carrier(out_dir, root, bundle, G)):
        step()
        print(f"[phase2b] {G[-1]['verdict']:<14} {G[-1]['what']}", flush=True)
    ok = ("PASS", "BIT_IDENTICAL", "MATCH", "NOT_STATED", "UNRESOLVED", "RECORDED")
    g["all_pass"] = all(x["verdict"] in ok for x in G)
    return g


# ---------------------------------------------------------------------------
# inputs (§CT18-F items 4 and 5.1, owed at §CT18-D2): every file a family opens
# for reading is traced -- Python's `open` audit event, plus h5py.File, which
# opens in C and raises no event -- then hashed, and every artifact input is
# checked against the hash the record states. A MISMATCH stops the family.
# ---------------------------------------------------------------------------

SCRATCH_ROOT = Path("<cluster-fs>/scratch")
_TRACE: set | None = None
_TRACE_INSTALLED = False


def _note(path, mode=None, flags=0) -> None:
    import os
    if _TRACE is None or not isinstance(path, (str, bytes, os.PathLike)):
        return
    if isinstance(mode, str):
        if any(c in mode for c in "wax+"):
            return
    elif flags & (os.O_WRONLY | os.O_RDWR):
        return
    _TRACE.add(Path(os.fsdecode(path)).resolve())


def _audit(event, args) -> None:
    if event == "open" and args:
        _note(args[0], args[1] if len(args) > 1 else None, args[2] if len(args) > 2 else 0)


class trace_inputs:
    """`with trace_inputs() as seen:` -- `seen` collects every path opened for
    reading inside the block. An audit hook cannot be removed, so it is
    installed once and gated by `_TRACE`."""

    def __enter__(self) -> set:
        global _TRACE, _TRACE_INSTALLED
        import sys
        if not _TRACE_INSTALLED:
            sys.addaudithook(_audit)
            orig = h5py.File.__init__

            def _init(self_, name, mode="r", *a, **k):
                _note(name, mode if isinstance(mode, str) else "r")
                return orig(self_, name, mode, *a, **k)
            h5py.File.__init__ = _init
            _TRACE_INSTALLED = True
        _TRACE = set()
        return _TRACE

    def __exit__(self, *exc) -> None:
        global _TRACE
        _TRACE = None


def inputs_block(seen: set, out_dir: Path, bundle: Path | None, repo: Path, texts: dict,
                 roots: tuple = (PROJECT, SCRATCH_ROOT)) -> list[dict]:
    """One row per input file: its path, kind, size and sha256; for an artifact
    input, the record-hash verdict. Files the family wrote into `out_dir`
    (re-read by its gates) are outputs, not inputs. Source code and anything
    outside `roots` and the repository (the interpreter, /proc, /etc) are
    listed by path only, so nothing opened is silently dropped."""
    out_dir = out_dir.resolve()
    bundle = bundle.resolve() if bundle is not None else None
    repo = repo.resolve()
    rows = []
    for p in sorted(seen):
        if not p.is_file() or out_dir == p.parent or out_dir in p.parents:
            continue
        if p.suffix in (".py", ".pyc"):
            continue
        if bundle is not None and bundle in p.parents:
            kind = "bundle"
        elif repo in p.parents:
            kind = "repo"
        elif any(r.resolve() in p.parents for r in roots):
            kind = "artifact"
        else:
            rows.append({"path": str(p), "kind": "other"})
            continue
        sha = sha256_file(p)
        row = {"path": str(p), "kind": kind, "bytes": p.stat().st_size, "sha256": sha}
        if kind == "artifact":
            rc = record_hash_check(p.name, sha, texts)
            row["record_hash"] = rc["verdict"]
            if rc["mismatch_lines"]:
                row["mismatch_lines"] = rc["mismatch_lines"]
        rows.append(row)
    return rows


def input_gates(rows: list[dict], primary: Path) -> list[dict]:
    """The trace saw the family's primary input (so the tracer works on this
    path), and no artifact input contradicts a hash the record states."""
    paths = {r["path"] for r in rows}
    counts = {}
    for r in rows:
        if "record_hash" in r:
            counts[r["record_hash"]] = counts.get(r["record_hash"], 0) + 1
    bad = [r["path"] for r in rows if r.get("record_hash") == "MISMATCH"]
    return [{"what": f"inputs traced ({len(rows)}); the primary input {primary.name} among them",
             "verdict": "PASS" if str(primary.resolve()) in paths else "FAIL"},
            {"what": "record hashes of the artifact inputs", "counts": counts,
             "verdict": "FAIL" if bad else "PASS", "mismatch": bad}]


def primary_input(cmd: str, name: str | None, root: Path) -> Path:
    if cmd == "arm":
        return arm_paths(name, root)["d_fs"]
    return {"anchors": root / ANCHOR_INPUTS["sim"],
            "replay": root / "ct" / "activation_correlation_counts_ct.h5",
            "phase2b": root / "feature_ap_main_compliant.h5"}[cmd]


# ---------------------------------------------------------------------------
# manifest: one MANIFEST.csv for the whole bundle, written only when every
# family's gates passed; then the copy to $PROJECT, each file re-hashed there.
# ---------------------------------------------------------------------------

FAMILIES = [f"arm_{a}" for a in ARMS] + ["anchors", "replay", "phase2b"]
SIZE_STOP_BYTES = 1 << 30            # §CT18-F item 9: above 1 GiB, stop and ask first


def owning_row(name: str) -> str:
    """The NoR rows each new file serves (§CT18-F item 3's table)."""
    arm_rows = {"w2560": "§19, §20.27 (b), §20.30 (c)", "w10240": "§19, §20.27 (b), §20.30 (c)",
                "w20480": "§19, §20.27 (b), §20.30 (c)", "w40960": "§19, §20.27 (b), §20.30 (c)",
                "k64": "§20.26-§20.28, §20.30 (b), §20.33 (seeds 0-9)",
                "k64_w2560": "§20.29 (a), §20.33, §20.37", "k64_w40960": "§20.29 (a), §20.33, §20.37",
                "k16_w10240": "§20.29 (b)-(c), §20.32 (top-merged sensitivity), §20.37",
                "k128_w10240": "§20.29 (b)-(c), §20.32 (top-merged sensitivity), §20.37"}
    fixed = {"sim_decoder_n39.csv.gz": "§20.1, §20.4 (Table 1's intervals); §20.14; §20.16 rule 1",
             "sim_activation_n39.csv.gz": "§20.8-§20.10, §20.12; §20.16, §20.18, §20.20 rule 4",
             "sim_activation_halfA_n39.csv.gz": "§20.19",
             "sim_oneway_n39.csv.gz": "§20.15",
             "features_logM_variants_n39.csv.gz": "§20.19 (the halves' M); §20.13 (the merged strata's M)",
             "features_carrier_n39.csv.gz": "§20.17 (the triple arm's concentration covariate)",
             "features_disorder_n39.csv.gz": "§20.23",
             "features_classification_n39.csv.gz": "§20.6; §4.1"}
    if name in fixed:
        return fixed[name]
    if name.startswith("sim_rule"):
        return "§20.18, §20.20, §20.24; §13.1-§13.2 rules 1-3"
    if name.startswith("subspace_"):
        return "§20.21, §20.22, §20.25 (per-cell rows; RECOUNTED)"
    if name.startswith("owner__"):
        return "the owner JSON, verbatim (RE-READ; the reference of every gate)"
    if name == "arm_k64_sim_activation.csv.gz":
        return "§20.26 (the activation block)"
    for a, r in arm_rows.items():
        if name.startswith(f"arm_{a}_") and not any(
                name.startswith(f"arm_{b}_") for b in arm_rows if b != a and b.startswith(a)):
            return r
    raise SystemExit(f"FATAL: no owning row for {name}")


def build_manifest(out_dir: Path, project_bundle: Path, copy: bool) -> dict:
    from src.sae.train import git_dirty, git_sha
    gates = {}
    for fam in FAMILIES:
        p = out_dir / f"gates_{fam}.json"
        if not p.exists():
            raise SystemExit(f"REFUSING: {p.name} is absent; every family must have run")
        g = json.loads(p.read_text(encoding="utf-8"))
        if not g.get("all_pass"):
            raise SystemExit(f"REFUSING: {p.name} has a failed gate")
        gates[fam] = {"n_gates": len(g["gates"]), "all_pass": True,
                      "verdicts": sorted({x["verdict"] for x in g["gates"]})}
    old = list(csv.DictReader(open(project_bundle / "MANIFEST.csv", encoding="utf-8", newline="")))
    for r in old:
        f = project_bundle / Path(r["path"]).name
        if sha256_file(f) != r["sha256"]:
            raise SystemExit(f"FATAL: {f.name} in the project bundle does not hash as its manifest")
    old_names = {Path(r["path"]).name for r in old}
    new = sorted(p for p in out_dir.iterdir() if p.is_file()
                 and not p.name.startswith("gates_") and p.name not in old_names
                 and p.name not in ("MANIFEST.csv", "meta_ext.json"))
    size = sum(p.stat().st_size for p in new)
    if size > SIZE_STOP_BYTES:
        raise SystemExit(f"STOP: the new files total {size} bytes, above 1 GiB (§CT18-F item 9)")
    commit, dirty = git_sha(), git_dirty()
    if dirty:
        raise SystemExit("REFUSING: the checkout is dirty")
    rows = [[r["path"], r["sha256"], r["producing_commit"], r["owning_nor_row"]] for r in old]
    rows += [[f"release/derived/{p.name}", sha256_file(p), commit, owning_row(p.name)] for p in new]
    meta = {"governing_amendment": "CT §CT18-F", "producing_commit": commit, "git_dirty": dirty,
            "n_files_new": len(new), "bytes_new": size, "n_files_total": len(rows) + 1,
            "families": gates,
            "levels": {"REGENERATED": "scripts/reproduce_table1.py (Table 1, the headline, the "
                                      "nine-arm table)",
                       "RECOMPUTED": "the build gates, BIT_IDENTICAL against each owner",
                       "RECOUNTED": "the subspaces, the classification, the disorder shares",
                       "RE-READ": "owner__*.json, verbatim"}}
    (out_dir / "meta_ext.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    rows.append(["release/derived/meta_ext.json", sha256_file(out_dir / "meta_ext.json"), commit,
                 "— (the extension's families, gates and levels)"])
    with open(out_dir / "MANIFEST.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["path", "sha256", "producing_commit", "owning_nor_row"])
        w.writerows(rows)
    out = {"family": "manifest", "gates": [], "manifest_rows": len(rows), "bytes_new": size,
           "producing_commit": commit}
    if copy:
        import shutil
        for p in new + [out_dir / "meta_ext.json"]:
            dst = project_bundle / p.name
            if dst.exists():
                raise SystemExit(f"REFUSING: {dst} exists in the project bundle")
            shutil.copyfile(p, dst)
        shutil.copyfile(out_dir / "MANIFEST.csv", project_bundle / "MANIFEST.csv")
        bad = [r[0] for r in rows if sha256_file(project_bundle / Path(r[0]).name) != r[1]]
        out["gates"].append({"what": "every manifest row re-hashed in the project bundle",
                             "verdict": "PASS" if not bad else "FAIL", "differing": bad})
    out["all_pass"] = all(x["verdict"] == "PASS" for x in out["gates"])
    return out


def update_meta(out_dir: Path, project_bundle: Path, copy: bool) -> dict:
    """§CT18-D2's two owed items on the bundle already shipped: a traced rebuild
    in `out_dir` supplies every family's inputs and their record-hash verdicts.
    Refuses unless every family passed with an input record, and every rebuilt
    data file is byte-identical to the shipped file its manifest row names --
    so only `meta_ext.json` and its manifest row change. The other rows keep
    their producing commit."""
    from src.sae.train import git_dirty, git_sha
    gates, inputs = {}, {}
    for fam in FAMILIES:
        p = out_dir / f"gates_{fam}.json"
        if not p.exists():
            raise SystemExit(f"REFUSING: {p.name} is absent; every family must have run")
        g = json.loads(p.read_text(encoding="utf-8"))
        if not g.get("all_pass"):
            raise SystemExit(f"REFUSING: {p.name} has a failed gate")
        if "inputs" not in g:
            raise SystemExit(f"REFUSING: {p.name} has no input record (run without the trace)")
        gates[fam] = {"n_gates": len(g["gates"]), "all_pass": True,
                      "verdicts": sorted({x["verdict"] for x in g["gates"]})}
        inputs[fam] = g["inputs"]
    old = list(csv.DictReader(open(project_bundle / "MANIFEST.csv", encoding="utf-8", newline="")))
    for r in old:
        if sha256_file(project_bundle / Path(r["path"]).name) != r["sha256"]:
            raise SystemExit(f"FATAL: {Path(r['path']).name} in the project bundle does not "
                             "hash as its manifest")
    old_meta = json.loads((project_bundle / "meta_ext.json").read_text(encoding="utf-8"))
    built = old_meta["producing_commit"]
    want = {Path(r["path"]).name: r["sha256"] for r in old
            if r["producing_commit"] == built and Path(r["path"]).name != "meta_ext.json"}
    got = {p.name: p for p in out_dir.iterdir() if p.is_file() and not p.name.startswith("gates_")
           and p.name not in ("MANIFEST.csv", "meta_ext.json")}
    if set(got) != set(want):
        raise SystemExit(f"REFUSING: the rebuild's files differ from the manifest's {built[:7]} rows: "
                         f"absent {sorted(set(want) - set(got))[:3]}, extra {sorted(set(got) - set(want))[:3]}")
    differ = sorted(n for n, p in got.items() if sha256_file(p) != want[n])
    if differ:
        raise SystemExit(f"REFUSING: {len(differ)} rebuilt file(s) are not byte-identical to the "
                         f"shipped ones, e.g. {differ[:3]}")
    commit, dirty = git_sha(), git_dirty()
    if dirty:
        raise SystemExit("REFUSING: the checkout is dirty")
    counts = {}
    for rows in inputs.values():
        for r in rows:
            if "record_hash" in r:
                counts[r["record_hash"]] = counts.get(r["record_hash"], 0) + 1
    meta = dict(old_meta)
    meta["families"] = gates
    meta["inputs_recorded"] = {
        "governing_amendment": "CT §CT18-D2 (the build's owed items: §CT18-F items 4 and 5.1)",
        "producing_commit": commit, "git_dirty": dirty,
        "data_files_byte_identical_to_build": built, "n_data_files_compared": len(got),
        "record_hash_verdicts_artifact_inputs": dict(sorted(counts.items()))}
    meta["inputs"] = inputs
    (out_dir / "meta_ext.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    sha = sha256_file(out_dir / "meta_ext.json")
    rows = [[r["path"], sha if Path(r["path"]).name == "meta_ext.json" else r["sha256"],
             commit if Path(r["path"]).name == "meta_ext.json" else r["producing_commit"],
             r["owning_nor_row"]] for r in old]
    with open(out_dir / "MANIFEST.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["path", "sha256", "producing_commit", "owning_nor_row"])
        w.writerows(rows)
    out = {"family": "update_meta", "gates": [], "manifest_rows": len(rows),
           "n_data_files_byte_identical": len(got), "record_hash_counts": counts,
           "producing_commit": commit}
    if copy:
        import shutil
        for n in ("meta_ext.json", "MANIFEST.csv"):
            shutil.copyfile(out_dir / n, project_bundle / n)
        bad = [r[0] for r in rows if sha256_file(project_bundle / Path(r[0]).name) != r[1]]
        out["gates"].append({"what": "every manifest row re-hashed in the project bundle",
                             "verdict": "PASS" if not bad else "FAIL", "differing": bad})
    out["all_pass"] = all(x["verdict"] == "PASS" for x in out["gates"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("arm")
    a.add_argument("name", choices=sorted(ARMS))
    n = sub.add_parser("anchors")
    n.add_argument("--bundle", type=Path, default=None,
                   help="where features_n39.csv.gz is (default: --out-dir)")
    r = sub.add_parser("replay")
    r.add_argument("--bundle", type=Path, default=None,
                   help="where features_n39.csv.gz is (default: --out-dir)")
    b = sub.add_parser("phase2b")
    b.add_argument("--bundle", type=Path, default=None,
                   help="where the 94ca8b9 bundle is (default: --out-dir)")
    mf = sub.add_parser("manifest")
    mf.add_argument("--project-bundle", type=Path, required=True)
    mf.add_argument("--copy", action="store_true",
                    help="copy the new files, meta_ext.json and the manifest into the project bundle")
    mf.add_argument("--update-meta", action="store_true",
                    help="the bundle is already shipped: record the traced rebuild's inputs in "
                         "meta_ext.json and change nothing else (refuses unless every rebuilt "
                         "file is byte-identical to the shipped one)")
    for p in (a, n, r, b, mf):
        p.add_argument("--out-dir", type=Path, required=True)
        p.add_argument("--root", type=Path, default=A)
    args = ap.parse_args(argv)
    builders = {"arm": lambda: build_arm(args.name, args.out_dir, args.root),
                "anchors": lambda: build_anchors(args.out_dir, args.root, bundle=args.bundle),
                "replay": lambda: build_replay(args.out_dir, args.root, bundle=args.bundle),
                "phase2b": lambda: build_phase2b(args.out_dir, args.root, bundle=args.bundle)}
    if args.cmd in builders:
        from src.experiment_config import REPO_ROOT
        tag = f"arm_{args.name}" if args.cmd == "arm" else args.cmd
        with trace_inputs() as seen:
            g = builders[args.cmd]()
        texts = record_texts(REPO_ROOT)
        bundle = getattr(args, "bundle", None) or (None if args.cmd == "arm" else args.out_dir)
        g["inputs"] = inputs_block(seen, args.out_dir, bundle, REPO_ROOT, texts)
        ig = input_gates(g["inputs"], primary_input(args.cmd, getattr(args, "name", None),
                                                    args.root))
        g["gates"] += ig
        g["all_pass"] = bool(g["all_pass"]) and all(x["verdict"] == "PASS" for x in ig)
    elif args.update_meta:
        g = update_meta(args.out_dir, args.project_bundle, args.copy)
        tag = "update_meta"
    else:
        g = build_manifest(args.out_dir, args.project_bundle, args.copy)
        tag = "manifest"
    (args.out_dir / f"gates_{tag}.json").write_text(json.dumps(g, indent=2), encoding="utf-8")
    for x in g["gates"]:
        print(f"{x['verdict']:<14} {x['what']}" + (f"  -- {x['first_difference']}"
                                                  if x.get("first_difference") else ""), flush=True)
    print(f"ALL PASS: {g['all_pass']}", flush=True)
    return 0 if g["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
