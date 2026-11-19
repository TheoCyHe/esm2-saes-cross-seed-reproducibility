"""The activation-correlation robustness arm (PREREGISTRATION §AB2, declared;
amendment CT, §CT4 and §CT11.6, executed).

THE DEFINITION (§CT4, fixed before this code existed). For feature f of
dictionary s, a_f is its post-TopK activation over every residue row of the
untruncated scored population (the label store's population: 51,943 proteins,
15,972,325 rows; §G2/§H2e), zeros included. The similarity between f in s and
g in s' is the Pearson correlation of a_f and a_g, from sparse sums
(sum a, sum a^2, sum a*b). Matching is MNN under that similarity, on the same 20
anchors against 39 comparators as the primary, at the 13-point theta grid and at
a rate-matched theta* whose mean match rate equals the decoder-cosine rate at
theta = 0.70 (0.495513, NoR §1.1).

THREE STEPS, one per hardware shape:
  encode  one task per dictionary, rtx_5000_ada (the gather is device-dependent,
          gotcha 36): TopK codes over the scored rows, written sparse (k values
          and int16 indices per row) with per-feature sum a and sum a^2 in fp64.
  gram    one task per anchor, GPU: for each comparator, the dense cross-moment
          sum a_f b_g accumulated in fp64 over row chunks, Pearson r formed, and
          the MNN similarity of every anchor feature recorded (rf_marginal's
          `mnn_sim` semantics: r of the mutual partner, -inf if not mutual).
  reduce  CPU: counts on the theta grid and at theta*, and the design-3 slope
          at theta* through the same fitter and loader as §CT3.

A feature with zero variance over the population has undefined r; it is given
r = -inf against everything (never matched), and the count is reported.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import h5py
import numpy as np

TARGET_RATE = 0.495513          # NoR §1.1, the decoder-cosine headline
THETA_GRID = [round(0.30 + 0.05 * i, 2) for i in range(13)]


# ---------------------------------------------------------------------------
# Pure functions (toy-tested)
# ---------------------------------------------------------------------------

def pearson_from_moments(G: np.ndarray, s1a: np.ndarray, s2a: np.ndarray,
                         s1b: np.ndarray, s2b: np.ndarray, n: int) -> np.ndarray:
    """Pearson r between every a-feature and every b-feature from sums.

    G = sum_rows a_f b_g, s1 = sum a, s2 = sum a^2, n = rows. Zero-variance
    features give -inf (never matched)."""
    ma, mb = s1a / n, s1b / n
    va = s2a / n - ma ** 2
    vb = s2b / n - mb ** 2
    cov = G / n - np.outer(ma, mb)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cov / np.sqrt(np.outer(np.maximum(va, 0), np.maximum(vb, 0)))
    bad = (va <= 0)[:, None] | (vb <= 0)[None, :]
    r[bad] = -np.inf
    return r


def mnn_from_matrix(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`rf_marginal.pair_statistics`' MNN semantics on a similarity matrix:
    per a-row, r of the best b if that b's best a is this row, else -inf; and
    the same for the b side."""
    a_best = R.argmax(axis=1)
    b_best = R.argmax(axis=0)
    a_sim = R[np.arange(R.shape[0]), a_best]
    b_sim = R[b_best, np.arange(R.shape[1])]
    a_mut = b_best[a_best] == np.arange(R.shape[0])
    b_mut = a_best[b_best] == np.arange(R.shape[1])
    return (np.where(a_mut, a_sim, -np.inf).astype(np.float32),
            np.where(b_mut, b_sim, -np.inf).astype(np.float32))


def mean_rate(sim: np.ndarray, theta: float) -> float:
    """Mean match rate over (anchor, feature) cells: counts / n_compare."""
    from src.analysis.jackknife_dictionaries import counts_from_sim
    n_compare = sim.shape[1] - 1
    return float(counts_from_sim(sim, theta).mean() / n_compare)


def theta_star(sim: np.ndarray, target: float = TARGET_RATE,
               lo: float = -1.0, hi: float = 1.0) -> dict:
    """The LARGEST theta whose mean match rate is >= target (the rate is a
    non-increasing step function of theta, so exact equality is not generally
    attainable). Found on the finite similarity values themselves: the rate
    only changes at an observed similarity, so theta* is one of them."""
    vals = sim[np.isfinite(sim)]
    if vals.size == 0:
        return {"theta_star": None, "status": "NO_FINITE_SIMILARITY"}
    cand = np.unique(vals)
    if mean_rate(sim, float(cand[0])) < target:
        return {"theta_star": None, "status": "TARGET_UNREACHABLE",
                "max_rate": mean_rate(sim, float(cand[0]))}
    i, j = 0, cand.size - 1          # invariant: rate(cand[i]) >= target
    while i < j:
        m = (i + j + 1) // 2
        if mean_rate(sim, float(cand[m])) >= target:
            i = m
        else:
            j = m - 1
    t = float(cand[i])
    nxt = float(cand[i + 1]) if i + 1 < cand.size else None
    return {"theta_star": t, "rate_at_theta_star": mean_rate(sim, t),
            "rate_just_above": (mean_rate(sim, nxt) if nxt is not None else None),
            "target": target, "status": "OK"}


def dense_from_codes(vals: np.ndarray, idx: np.ndarray, d: int,
                     device: str = "cpu"):
    """(n, k) values + (n, k) int16 indices -> (n, d) dense float32 tensor on
    `device` (the scatter runs there)."""
    import torch
    v = torch.as_tensor(vals).to(device)
    i = torch.as_tensor(idx.astype(np.int64)).to(device)
    z = torch.zeros((v.shape[0], d), dtype=v.dtype, device=device)
    z.scatter_(1, i, v)
    return z


def symmetry_report(sim: np.ndarray, n_anchors: int) -> dict:
    """Anchor-anchor pairs are matched from BOTH sides (task a and task b).
    MNN is symmetric, so each side must find the same number of mutual pairs
    with the same similarities, up to the float summation order of two
    separately accumulated Grams (gotcha 35). Reported, not gated."""
    n_pairs, n_count_mismatch, max_abs = 0, 0, 0.0
    for a in range(n_anchors):
        for b in range(a + 1, n_anchors):
            x = np.sort(sim[a, b][np.isfinite(sim[a, b])])
            y = np.sort(sim[b, a][np.isfinite(sim[b, a])])
            n_pairs += 1
            if x.size != y.size:
                n_count_mismatch += 1
            elif x.size:
                max_abs = max(max_abs, float(np.max(np.abs(x - y))))
    return {"n_anchor_pairs": n_pairs,
            "n_pairs_with_different_mutual_counts": n_count_mismatch,
            "max_abs_similarity_difference": max_abs}


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------

def resolve_k(arg_k: int | None, meta: dict) -> int:
    """Gotcha 48: `k` is the CHECKPOINT's (`meta.json`), never a default. A
    given `--k` must equal it; an absent meta `k` is refused, never assumed."""
    if "k" not in meta:
        raise SystemExit("FATAL: the checkpoint's meta.json records no k (ABSENT)")
    mk = int(meta["k"])
    if arg_k is not None and int(arg_k) != mk:
        raise SystemExit(f"FATAL: --k {arg_k} but the checkpoint records k={mk}")
    return mk


def cmd_encode(args) -> None:
    import torch
    from src.analysis.artifact_identity import device_class
    from src.analysis.feature_ap import load_scored_row_set
    from src.analysis.fvu import load_sae
    from src.sae.train import git_dirty, git_sha
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    w = load_sae(args.ckpt, dev)
    seed = int(w["meta"]["seed"])
    d = int(w["W_dec"].shape[1])
    k = resolve_k(args.k, w["meta"])
    rows = load_scored_row_set(args.labels, "label_store")
    with h5py.File(args.cache, "r") as hf:
        n_cache = hf["activations"].shape[0]
    mask = np.zeros(n_cache, dtype=bool)
    mask[rows] = True
    n = int(mask.sum())
    s1 = torch.zeros(d, dtype=torch.float64, device=dev)
    s2 = torch.zeros(d, dtype=torch.float64, device=dev)
    nnz = torch.zeros(d, dtype=torch.int64, device=dev)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".partial.h5")
    with h5py.File(args.cache, "r") as hc, h5py.File(tmp, "w") as ho:
        acts = hc["activations"]
        V = ho.create_dataset("values", (n, k), dtype="float32", chunks=(65536, k))
        I = ho.create_dataset("indices", (n, k), dtype="int16", chunks=(65536, k))
        pos = 0
        for i in range(0, n_cache, args.chunk_rows):
            j = min(i + args.chunk_rows, n_cache)
            m = mask[i:j]
            if not m.any():
                continue
            x = torch.from_numpy(np.asarray(acts[i:j], dtype=np.float32)[m]).to(dev)
            with torch.no_grad():
                pre = torch.relu((x - w["b_dec"]) @ w["W_enc"].T + w["b_enc"])
                tk = torch.topk(pre, k, dim=-1)
                vals, idx = tk.values, tk.indices
                z = torch.zeros_like(pre)
                z.scatter_(-1, idx, vals)
                s1 += z.sum(0, dtype=torch.float64)
                s2 += (z.double() ** 2).sum(0)
                nnz += (z > 0).sum(0)
            b = vals.shape[0]
            V[pos:pos + b] = vals.cpu().numpy()
            I[pos:pos + b] = idx.cpu().numpy().astype(np.int16)
            pos += b
        if pos != n:
            raise SystemExit(f"FATAL: wrote {pos} rows, expected {n}")
        ho.create_dataset("sum_a", data=s1.cpu().numpy())
        ho.create_dataset("sum_a2", data=s2.cpu().numpy())
        ho.create_dataset("n_nonzero", data=nnz.cpu().numpy())
        ho.attrs.update({
            "measurement": "topk_codes_scored_rows", "governing_amendment": "CT4",
            "seed": seed, "k": k, "dict_size": d, "n_rows": n,
            "scored_row_set": "label_store", "labels": str(args.labels),
            "cache": str(args.cache), "ckpt": str(args.ckpt),
            "ckpt_git_sha": str(w["meta"].get("git_sha")),
            "gather_device_class": str(device_class(z)),
            "git_sha": git_sha(), "git_dirty": git_dirty(),
            "host": str(platform.node())})
    tmp.replace(args.out)
    print(f"[ac-encode] seed {seed}: {n:,} rows -> {args.out}", flush=True)


# ---------------------------------------------------------------------------
# gram
# ---------------------------------------------------------------------------

def _open_codes(path: Path) -> dict:
    hf = h5py.File(path, "r")
    a = dict(hf.attrs)
    return {"h": hf, "seed": int(a["seed"]), "n": int(a["n_rows"]),
            "d": int(a["dict_size"]), "s1": hf["sum_a"][:],
            "s2": hf["sum_a2"][:], "dev": str(a["gather_device_class"]),
            "git_sha": str(a["git_sha"])}


def gram_row_mask(args, n_code_rows: int) -> np.ndarray | None:
    """§CT10: which of the codes' rows a Gram uses. The codes hold the
    `label_store` rows in ascending cache order (`cmd_encode`), so a subset of
    cache rows maps to code positions by `searchsorted`, checked exact. None
    means every row (§CT4's arm, unchanged)."""
    if getattr(args, "row_subset", None) is None:
        if getattr(args, "row_subset_file", None) is not None:
            raise SystemExit("FATAL: --row-subset-file without --row-subset")
        return None
    from src.analysis.feature_ap import load_scored_row_set
    if args.labels is None or args.row_subset_file is None:
        raise SystemExit("FATAL: --row-subset needs --labels and --row-subset-file")
    pop = load_scored_row_set(args.labels, "label_store")
    if pop.size != n_code_rows:
        raise SystemExit(f"FATAL: the codes hold {n_code_rows} rows, the label_store "
                         f"population {pop.size}")
    rows = load_scored_row_set(args.labels, args.row_subset, args.row_subset_file)
    pos = np.searchsorted(pop, rows)
    if not np.array_equal(pop[np.minimum(pos, pop.size - 1)], rows):
        raise SystemExit("FATAL: the subset's rows are not all code rows")
    m = np.zeros(n_code_rows, dtype=bool)
    m[pos] = True
    return m


def cmd_gram(args) -> None:
    import torch
    from src.analysis.artifact_identity import device_class
    from src.sae.train import git_dirty, git_sha
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = getattr(args, "device", "cuda")
    codes = sorted(args.codes, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    C = [_open_codes(p) for p in codes]
    seeds = [c["seed"] for c in C]
    a_i = seeds.index(args.anchor)
    A = C[a_i]
    n, d = A["n"], A["d"]
    for c in C:
        if c["n"] != n or c["d"] != d:
            raise SystemExit(f"FATAL: seed {c['seed']} rows/width differ")
    # §CT10: a row subset of the codes' rows. The codes' moments describe ALL
    # their rows, so on a subset every moment is recomputed over the subset in
    # the same chunk pass as the Gram -- the stored sums are never reused.
    mask = gram_row_mask(args, n)
    n_used = n if mask is None else int(mask.sum())
    sub = {}                     # seed index -> [s1, s2] over the subset
    others = [i for i in range(len(C)) if i != a_i]
    sim_row = np.full((len(C), d), -np.inf, dtype=np.float32)
    n_zero_var = {}
    for g0 in range(0, len(others), args.group):
        grp = others[g0:g0 + args.group]
        G = {b: torch.zeros((d, d), dtype=torch.float64, device=dev) for b in grp}
        want = ([a_i] if (mask is not None and a_i not in sub) else []) + \
               ([b for b in grp] if mask is not None else [])
        acc = {i: [torch.zeros(d, dtype=torch.float64, device=dev),
                   torch.zeros(d, dtype=torch.float64, device=dev)] for i in want}
        for i in range(0, n, args.chunk_rows):
            j = min(i + args.chunk_rows, n)
            sel = None
            if mask is not None:
                m = mask[i:j]
                if not m.any():
                    continue
                sel = torch.from_numpy(np.flatnonzero(m)).to(dev)
            za = dense_from_codes(A["h"]["values"][i:j],
                                  A["h"]["indices"][i:j], d, dev)
            if sel is not None:
                za = za.index_select(0, sel)
                if a_i in acc:
                    acc[a_i][0] += za.sum(0, dtype=torch.float64)
                    acc[a_i][1] += (za.double() ** 2).sum(0)
            for b in grp:
                zb = dense_from_codes(C[b]["h"]["values"][i:j],
                                      C[b]["h"]["indices"][i:j], d, dev)
                if sel is not None:
                    zb = zb.index_select(0, sel)
                    acc[b][0] += zb.sum(0, dtype=torch.float64)
                    acc[b][1] += (zb.double() ** 2).sum(0)
                G[b] += (za.T @ zb).double()
        for i_, (t1, t2) in acc.items():
            sub[i_] = [t1.cpu().numpy(), t2.cpu().numpy()]
        mom = (lambda i_: (C[i_]["s1"], C[i_]["s2"])) if mask is None else \
              (lambda i_: (sub[i_][0], sub[i_][1]))
        s1a, s2a = mom(a_i)
        for b in grp:
            s1b, s2b = mom(b)
            R = pearson_from_moments(G[b].cpu().numpy(), s1a, s2a, s1b, s2b, n_used)
            a_sim, _ = mnn_from_matrix(R)
            sim_row[b] = a_sim
            n_zero_var[str(seeds[b])] = int(np.sum(s2b / n_used - (s1b / n_used) ** 2 <= 0))
        print(f"[ac-gram] anchor {args.anchor}: comparators "
              f"{[seeds[b] for b in grp]} done", flush=True)
        del G
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Written to a partial file and renamed: a sequential gpu_devel runner and
    # the gpu array may both reach one anchor (§CT11.6-R addendum), and a
    # consumer must never see a half-written row.
    tmp = args.out.with_suffix(".partial.h5")
    with h5py.File(tmp, "w") as ho:
        ho.create_dataset("mnn_sim_row", data=sim_row)
        ho.attrs.update({
            "measurement": "activation_correlation_mnn_row", "governing_amendment": "CT4",
            "anchor_seed": args.anchor, "column_seed_labels": json.dumps(seeds),
            "n_rows": n_used, "dict_size": d, "similarity": "activation_pearson",
            "rule": "mnn", "n_zero_variance_features_by_seed": json.dumps(n_zero_var),
            "anchor_zero_variance_features": int(np.sum(s2a / n_used - (s1a / n_used) ** 2 <= 0)),
            "row_subset": (getattr(args, "row_subset", None) or "all_code_rows"),
            "row_subset_file": str(getattr(args, "row_subset_file", None) or ""),
            "n_code_rows": n,
            "encode_device_classes": json.dumps(sorted({c["dev"] for c in C})),
            "gram_device_class": str(device_class(torch.zeros(1, device=dev))),
            "git_sha": git_sha(), "git_dirty": git_dirty(),
            "host": str(platform.node())})
    tmp.replace(args.out)
    print(f"[ac-gram] wrote {args.out}", flush=True)


# ---------------------------------------------------------------------------
# reduce
# ---------------------------------------------------------------------------

def cmd_reduce(args) -> None:
    from src.analysis.jackknife_dictionaries import counts_from_sim
    from src.sae.train import git_dirty, git_sha
    rows = sorted(args.rows, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    sims, anchors, cols = [], [], None
    for p in rows:
        with h5py.File(p, "r") as hf:
            sims.append(hf["mnn_sim_row"][:])
            anchors.append(int(hf.attrs["anchor_seed"]))
            c = json.loads(hf.attrs["column_seed_labels"])
            if cols is None:
                cols = c
            elif c != cols:
                raise SystemExit("FATAL: column order differs between anchors")
    if anchors != cols[:len(anchors)]:
        raise SystemExit(f"FATAL: anchors {anchors} are not the first columns {cols}")
    sim = np.stack(sims)                               # (n_anchors, n_seeds, d)
    sym = symmetry_report(sim, len(anchors))
    curve = {f"{t:.2f}": mean_rate(sim, t) for t in THETA_GRID}
    target = float(getattr(args, "target_rate", None) or TARGET_RATE)
    ts = theta_star(sim, target)
    out = {"identity": {"measurement": "activation_correlation_arm",
                        "theta_star_target_rate": target,
                        "governing_amendment": "CT4", "n_anchors": len(anchors),
                        "n_compare": int(sim.shape[1] - 1),
                        "anchor_seed_labels": anchors, "column_seed_labels": cols,
                        "git_sha": git_sha(), "git_dirty": git_dirty()},
           "mean_match_rate_by_theta": curve,
           "mean_match_rate_at_numeric_0.70": curve["0.70"],
           "anchor_anchor_symmetry": sym,
           "theta_star": ts}
    with h5py.File(args.out_counts, "w") as ho:
        for t in THETA_GRID:
            ho.create_dataset(f"match_count_theta_{t:.2f}", data=counts_from_sim(sim, t))
        if ts.get("theta_star") is not None:
            ho.create_dataset("match_count_theta_star", data=counts_from_sim(sim, ts["theta_star"]))
        ho.create_dataset("mnn_sim_anchor_rows", data=sim, compression="gzip")
        ho.attrs.update({"similarity": "activation_pearson", "rule": "mnn",
                         "n_compare_seeds": int(sim.shape[1] - 1),
                         "seed_labels": json.dumps(anchors),
                         "theta_star": json.dumps(ts), "git_sha": git_sha()})
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("mean_match_rate_at_numeric_0.70", "theta_star")}),
          flush=True)


def intervals_overlap(a: list, b: list) -> bool:
    """Do two closed intervals [lo, hi] share a point? (CT4-P1's predicate.)"""
    return bool(a[0] <= b[1] and b[0] <= a[1])


def cmd_fit(args) -> None:
    """The design-3 slope at theta* under activation-space match outcomes:
    the SAME loader, inputs, seed alignment and fitter as §CT3
    (`jackknife_dictionaries.load_join_inputs` / `_fit_one`), with only the
    counts replaced. The decoder-cosine fit at theta = 0.70 is refit beside it
    from the jackknife match file, so both intervals come from one code path."""
    from src.analysis import jackknife_dictionaries as jk
    from src.sae.train import git_dirty, git_sha
    inp = jk.load_join_inputs(args)
    jk._W.update(inp)
    with h5py.File(args.act_counts, "r") as hf:
        ts = json.loads(hf.attrs["theta_star"])
        if ts.get("theta_star") is None:
            raise SystemExit(f"FATAL: theta* not available: {ts}")
        y_star = hf["match_count_theta_star"][:]
        y_070 = hf["match_count_theta_0.70"][:]
        labels = [str(s) for s in json.loads(hf.attrs["seed_labels"])]
        n_compare = int(hf.attrs["n_compare_seeds"])
    if labels != inp["anchor_labels"]:
        raise SystemExit(f"FATAL: anchor order {labels} != {inp['anchor_labels']}")
    rows = np.arange(len(labels))
    act_star = jk._fit_one(y_star, rows, n_compare)
    act_070 = jk._fit_one(y_070, rows, n_compare)
    dec_070 = jk._fit_one(jk.counts_from_sim(inp["sim"], 0.70), rows, n_compare)
    out = {"identity": {"measurement": "activation_correlation_slope",
                        "governing_amendment": "CT4", "arm": "primary_hard_bin",
                        "likelihood": "betabinomial", "n_compare": n_compare,
                        "units": "percentage points of r_f per SD of log M",
                        "act_counts": str(args.act_counts),
                        "git_sha": git_sha(), "git_dirty": git_dirty()},
           "theta_star": ts,
           "activation_at_theta_star": act_star,
           "activation_at_numeric_0.70": act_070,
           "decoder_cosine_at_0.70": dec_070,
           "CT4_P1_fixed_effect_intervals_overlap": intervals_overlap(
               act_star["fe_ci95_pp"], dec_070["fe_ci95_pp"]),
           "CT4_P1_across_seed_intervals_overlap": intervals_overlap(
               act_star["across_seed_ci95_pp"], dec_070["across_seed_ci95_pp"])}
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if k.startswith("CT4")}), flush=True)


def cmd_jackknife(args) -> None:
    """§CT4's jackknife, at theta* only: §CT3's leave-one-dictionary-out
    machinery (`jackknife_dictionaries.loo_counts` / `_loo_task` / `jackknife`)
    run over the ACTIVATION-space `mnn_sim` instead of the decoder one. The
    join inputs are loaded through the same gate as §CT3; the control is the
    full refit at theta*, which must reproduce `--fit-json`'s value exactly."""
    from concurrent.futures import ProcessPoolExecutor
    from src.analysis import jackknife_dictionaries as jk
    from src.sae.train import git_dirty, git_sha
    inp = jk.load_join_inputs(args)
    with h5py.File(args.act_counts, "r") as hf:
        act_sim = hf["mnn_sim_anchor_rows"][:]
        ts = json.loads(hf.attrs["theta_star"])["theta_star"]
        labels = [str(s) for s in json.loads(hf.attrs["seed_labels"])]
    if labels != inp["anchor_labels"] or act_sim.shape != inp["sim"].shape:
        raise SystemExit("FATAL: the activation sim's anchors or shape differ "
                         "from the decoder match file's")
    inp["sim"] = act_sim
    jk._W.update(inp)
    n_anchors, n_seeds, _ = act_sim.shape
    full = jk._fit_one(jk.counts_from_sim(act_sim, ts), np.arange(n_anchors),
                       n_seeds - 1)
    ref = json.loads(args.fit_json.read_text(encoding="utf-8"))
    ctrl = jk.control_verdict([full["fe_pp"]],
                              [ref["activation_at_theta_star"]["fe_pp"]])
    print(f"[ac-jk] CONTROL {ctrl}", flush=True)
    out = {"identity": {"measurement": "activation_correlation_jackknife",
                        "governing_amendment": "CT4", "theta_star": ts,
                        "n_dictionaries": int(n_seeds),
                        "n_compare_loo": int(n_seeds - 2),
                        "act_counts": str(args.act_counts),
                        "git_sha": git_sha(), "git_dirty": git_dirty()},
           "control_full_refit_vs_fit_json": ctrl, "full_fit": full}
    if ctrl["verdict"] == "STOP":
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        raise SystemExit("CONTROL FAILED")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(jk._loo_task, [(d, ts) for d in range(n_seeds)],
                          chunksize=1))
    jkr = jk.jackknife([r["fe_pp"] for r in res], full["fe_pp"])
    ci = [full["fe_pp"] - 1.959964 * jkr["se"],
          full["fe_pp"] + 1.959964 * jkr["se"]]
    out.update({"jackknife": jkr, "jackknife_ci95_pp": ci,
                "W_jackknife_se_over_fe_se": jkr["se"] / full["fe_se_pp"],
                "n_loo_fits_failed": sum(r["n_anchor_fits_failed"] > 0 for r in res),
                "loo": res})
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[ac-jk] SE_jk {jkr['se']:.5f} pp, CI {ci}, W "
          f"{out['W_jackknife_se_over_fe_se']:.3f}", flush=True)


def pearson_of_pairs(G_pairs: np.ndarray, s1a: np.ndarray, s2a: np.ndarray,
                     s1b_g: np.ndarray, s2b_g: np.ndarray, n: int) -> np.ndarray:
    """Pearson r of each a-feature f with ONE named b-feature g(f), from sums:
    `G_pairs[f] = sum_rows a_f b_g(f)`, `s1b_g`/`s2b_g` the b-sums already
    gathered at g(f). Zero variance on either side gives -inf."""
    ma, mb = s1a / n, s1b_g / n
    va, vb = s2a / n - ma ** 2, s2b_g / n - mb ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (G_pairs / n - ma * mb) / np.sqrt(np.maximum(va, 0) * np.maximum(vb, 0))
    r[(va <= 0) | (vb <= 0)] = -np.inf
    return r


def pair_identity_summary(act_sim: np.ndarray, r_dec_partner: np.ndarray,
                          dec_sim: np.ndarray, theta_star: float,
                          tols=(1e-6, 1e-5, 1e-4)) -> dict:
    """§CT12.4(a) for one (anchor, comparator) row of features.

    `act_sim` is the stored activation MNN similarity (-inf if not mutual),
    `r_dec_partner` the activation correlation of each feature with its
    DECODER MNN partner, `dec_sim` the decoder MNN similarity (-inf if not
    mutual). A cell matched under activation at theta* has the decoder's
    partner as its activation partner when r_dec_partner equals act_sim
    (the activation partner is the argmax of r, so equality identifies it,
    ties aside)."""
    am = act_sim >= theta_star
    dm = dec_sim >= 0.70
    dec_mutual = np.isfinite(dec_sim)
    out = {"n_act_matched": int(am.sum()), "n_both_matched_cell": int((am & dm).sum())}
    for t in tols:
        same = am & dec_mutual & np.isfinite(r_dec_partner) & (
            np.abs(r_dec_partner - act_sim) < t)
        out[f"n_same_partner_tol_{t:g}"] = int(same.sum())
        out[f"n_same_partner_and_dec_clears_070_tol_{t:g}"] = int((same & dm).sum())
    return out


def cmd_pairs(args) -> None:
    """§CT12.4(a), one anchor per call: the decoder MNN partner of every anchor
    feature in every comparator (decoder cosine, rf_marginal's semantics with
    the index kept), the activation correlation of each feature with that
    partner from the stored codes, and the comparison with the stored
    activation similarity at theta*."""
    import torch
    from src.analysis.rf_marginal import load_decoder
    from src.sae.train import git_dirty, git_sha
    dev = "cuda"
    codes = {int(p.stem.rsplit("_", 1)[-1]): p for p in args.codes}
    ckpts = {int(json.loads((p / "meta.json").read_text())["seed"]): p
             for p in args.ckpts}
    with h5py.File(args.act_counts, "r") as hf:
        act_sim = hf["mnn_sim_anchor_rows"][:]
        ts = json.loads(hf.attrs["theta_star"])["theta_star"]
        anchors = [int(s) for s in json.loads(hf.attrs["seed_labels"])]
    cols = sorted(codes)
    a = args.anchor
    ai = anchors.index(a)
    with h5py.File(args.dec_sim, "r") as hf:          # §CT3's match file
        dec_stored = hf["mnn_sim_anchor_rows"][ai]
        if [int(s) for s in json.loads(hf.attrs["column_seed_labels"])] != cols:
            raise SystemExit("FATAL: decoder match file's columns differ")
    n_dec_mismatch = 0
    Wa, _ = load_decoder(ckpts[a], dev)
    A = _open_codes(codes[a])
    n, d = A["n"], A["d"]
    per_comp = {}
    for b in cols:
        if b == a:
            continue
        Wb, _ = load_decoder(ckpts[b], dev)
        C = Wa.T @ Wb
        a_best = C.argmax(dim=1)
        b_best = C.argmax(dim=0)
        mutual = (b_best[a_best] == torch.arange(d, device=dev))
        dec_sim = torch.where(mutual, C.max(dim=1).values,
                              torch.full((d,), float("-inf"), device=dev))
        g = a_best
        B = _open_codes(codes[b])
        acc = torch.zeros(d, dtype=torch.float64, device=dev)
        for i in range(0, n, args.chunk_rows):
            j = min(i + args.chunk_rows, n)
            za = dense_from_codes(A["h"]["values"][i:j], A["h"]["indices"][i:j], d, dev)
            zb = dense_from_codes(B["h"]["values"][i:j], B["h"]["indices"][i:j], d, dev)
            acc += (za.double() * zb[:, g].double()).sum(0)
        gi = g.cpu().numpy()
        r = pearson_of_pairs(acc.cpu().numpy(), A["s1"], A["s2"],
                             B["s1"][gi], B["s2"][gi], n)
        ds = dec_sim.float().cpu().numpy()
        # CONTROL: the decoder MNN recomputed here must be the stored one.
        n_dec_mismatch += int((ds != dec_stored[cols.index(b)]).sum())
        per_comp[str(b)] = pair_identity_summary(
            act_sim[ai, cols.index(b)], r, ds, ts)
        B["h"].close()
    tot = {}
    for v in per_comp.values():
        for k, x in v.items():
            tot[k] = tot.get(k, 0) + x
    out = {"identity": {"measurement": "activation_vs_decoder_pair_identity",
                        "governing_amendment": "CT12.4(a)", "anchor_seed": a,
                        "theta_star": ts, "git_sha": git_sha(), "git_dirty": git_dirty()},
           "control_decoder_mnn_cells_differing_from_stored": n_dec_mismatch,
           "per_comparator": per_comp, "anchor_total": tot}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[ac-pairs] anchor {a}: {tot}", flush=True)


def cmd_curve(args) -> None:
    """§CT12.4(b): the design-3 slope under activation matching at every grid
    theta, beside the decoder-space slope at the same theta -- both through
    §CT3's loader and fitter."""
    from src.analysis import jackknife_dictionaries as jk
    from src.sae.train import git_dirty, git_sha
    inp = jk.load_join_inputs(args)
    dec_sim = inp["sim"]
    with h5py.File(args.act_counts, "r") as hf:
        act_sim = hf["mnn_sim_anchor_rows"][:]
    jk._W.update(inp)
    rows = np.arange(act_sim.shape[0])
    n_c = act_sim.shape[1] - 1
    out_rows = []
    for t in THETA_GRID:
        a = jk._fit_one(jk.counts_from_sim(act_sim, t), rows, n_c)
        dd = jk._fit_one(jk.counts_from_sim(dec_sim, t), rows, n_c)
        out_rows.append({"theta": t, "activation": a, "decoder": dd})
        print(f"[ac-curve] {t:.2f}: act {a['fe_pp']:+.4f} dec {dd['fe_pp']:+.4f}",
              flush=True)
    above = [r["activation"]["fe_pp"] > r["decoder"]["fe_pp"] for r in out_rows]
    verdict = ("ABOVE_AT_EVERY_THETA" if all(above) else
               "BELOW_AT_EVERY_THETA" if not any(above) else "CROSSES")
    out = {"identity": {"measurement": "activation_vs_decoder_slope_curve",
                        "governing_amendment": "CT12.4(b)",
                        "git_sha": git_sha(), "git_dirty": git_dirty()},
           "rows": out_rows, "verdict_fixed_effect": verdict,
           "activation_above_decoder_by_theta": dict(
               zip([f"{t:.2f}" for t in THETA_GRID], above))}
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[ac-curve] {verdict}", flush=True)


CLASS_QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)


def cell_classes(act_sim: np.ndarray, dec_sim: np.ndarray, theta_act: float,
                 theta_dec: float) -> dict[str, np.ndarray]:
    """§CT4-T's cell classes over (anchor, comparator, anchor feature): matched
    under both criteria, activation-only, decoder-only, and neither. An
    anchor's own column is in no class (it never compares against itself)."""
    n_a, n_s, _ = act_sim.shape
    if dec_sim.shape != act_sim.shape:
        raise SystemExit(f"FATAL: shapes differ {act_sim.shape} vs {dec_sim.shape}")
    off = np.ones((n_a, n_s), dtype=bool)
    off[np.arange(n_a), np.arange(n_a)] = False
    off = off[:, :, None]
    A = (act_sim >= theta_act) & off
    D = (dec_sim >= theta_dec) & off
    return {"both": A & D, "activation_only": A & ~D,
            "decoder_only": D & ~A, "neither": off & ~A & ~D}


def class_distribution(logM: np.ndarray, mask: np.ndarray, keep: np.ndarray,
                       ref_mean: float, ref_sd: float) -> dict:
    """The anchor feature's `log M` over the cells of one class, restricted to
    the fit's feature set (`keep`). A feature counts once per comparator in
    which its cell falls in the class. The mean is also given in SDs from the
    fit set's own mean (`ref_*`, over features, not cells)."""
    sel = mask & keep[:, None, :]
    v = np.broadcast_to(logM[:, None, :], mask.shape)[sel]
    out = {"n_cells": int(v.size),
           "n_distinct_anchor_features": int((sel.any(axis=1)).sum())}
    if v.size == 0:
        return {**out, "empty": True}
    q = np.quantile(v, CLASS_QUANTILES)
    return {**out, "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else None,
            "quantiles": {f"{p:g}": float(x) for p, x in zip(CLASS_QUANTILES, q)},
            "mean_in_fit_set_sd_units": float((v.mean() - ref_mean) / ref_sd)}


def cmd_classes(args) -> None:
    """§CT4-T's descriptive item: the anchor feature's `log M` across the cell
    classes at theta* (activation) / 0.70 (decoder). Inputs through §CT3's
    loader, so `log M`, the anchor order and the fit's feature set are the
    primary's. CONTROL (can fail): the class counts over ALL features must equal
    the §CT12.4(a) pair files' totals exactly, computed by another path (GPU,
    decoder MNN recomputed from checkpoints)."""
    from src.analysis import jackknife_dictionaries as jk
    from src.sae.train import git_dirty, git_sha
    inp = jk.load_join_inputs(args)
    dec_sim = inp["sim"]
    with h5py.File(args.act_counts, "r") as hf:
        act_sim = hf["mnn_sim_anchor_rows"][:]
        ts = json.loads(hf.attrs["theta_star"])["theta_star"]
        labels = [str(s) for s in json.loads(hf.attrs["seed_labels"])]
    if labels != inp["anchor_labels"]:
        raise SystemExit(f"FATAL: anchor order {labels} != {inp['anchor_labels']}")
    act_cols = [int(s) for s in json.loads(args.act_json.read_text(encoding="utf-8"))
                ["identity"]["column_seed_labels"]]
    if act_cols != list(inp["column_labels"]):
        raise SystemExit(f"FATAL: column order {act_cols} != {inp['column_labels']}")
    cls = cell_classes(act_sim, dec_sim, ts, args.theta_dec)

    pair_tot = {}
    for p in args.pairs:
        for k, v in json.loads(p.read_text(encoding="utf-8"))["anchor_total"].items():
            pair_tot[k] = pair_tot.get(k, 0) + v
    mine = {"n_act_matched": int((cls["both"] | cls["activation_only"]).sum()),
            "n_both_matched_cell": int(cls["both"].sum())}
    ctrl = {k: {"classes": mine[k], "pairs": int(pair_tot.get(k, -1))} for k in mine}
    ok = len(args.pairs) == len(labels) and all(v["classes"] == v["pairs"]
                                                for v in ctrl.values())
    print(f"[ac-classes] control {'EXACT' if ok else 'DIFFERS'}: {ctrl}", flush=True)
    if not ok:
        raise SystemExit("STOP: class counts differ from the pair files' totals")

    logM, keep = inp["logM"], inp["keep"].astype(bool)
    ref = logM[keep]
    ref_mean, ref_sd = float(ref.mean()), float(ref.std(ddof=1))
    n_off = int(sum(m.sum() for m in cls.values()))
    out = {"identity": {"measurement": "logM_by_cell_class", "governing_amendment": "CT4-T",
                        "theta_star_activation": ts, "theta_decoder": args.theta_dec,
                        "units": "log M (natural log), the primary's (design 3, hard bin)",
                        "feature_set": "the design-3 fit set (A4a floor)",
                        "git_sha": git_sha(), "git_dirty": git_dirty()},
           "control_against_pair_files": {"verdict": "EXACT", **ctrl},
           "n_cells_all_features": {k: int(m.sum()) for k, m in cls.items()},
           "n_cells_off_diagonal": n_off,
           "fit_set": {"n_feature_seed_cells": int(keep.sum()),
                       "n_excluded_by_floor": int((~keep).sum()),
                       "logM_mean": ref_mean, "logM_sd": ref_sd},
           "classes": {k: class_distribution(logM, m, keep, ref_mean, ref_sd)
                       for k, m in cls.items()},
           "partner_partition_of_activation_matched_cells_1e-6": {
               "same_decoder_mnn_partner_and_decoder_clears_0.70":
                   pair_tot["n_same_partner_and_dec_clears_070_tol_1e-06"] / pair_tot["n_act_matched"],
               "same_decoder_mnn_partner_decoder_below_0.70":
                   (pair_tot["n_same_partner_tol_1e-06"]
                    - pair_tot["n_same_partner_and_dec_clears_070_tol_1e-06"]) / pair_tot["n_act_matched"],
               "different_partner_or_no_decoder_mnn_partner":
                   1 - pair_tot["n_same_partner_tol_1e-06"] / pair_tot["n_act_matched"]}}
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, v in out["classes"].items():
        print(f"[ac-classes] {k}: n={v['n_cells']} median={v.get('quantiles', {}).get('0.5')}",
              flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode")
    e.add_argument("--ckpt", type=Path, required=True)
    e.add_argument("--cache", type=Path, required=True)
    e.add_argument("--labels", type=Path, required=True)
    e.add_argument("--k", type=int, default=None,
                   help="must equal the checkpoint's meta k (gotcha 48); read from it if absent")
    e.add_argument("--chunk-rows", type=int, default=65536)
    e.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("gram")
    g.add_argument("--codes", nargs="+", type=Path, required=True)
    g.add_argument("--anchor", type=int, required=True)
    g.add_argument("--group", type=int, default=20)
    g.add_argument("--chunk-rows", type=int, default=32768)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--device", default="cuda")
    g.add_argument("--labels", type=Path, default=None,
                   help="§CT10: the label store the codes were encoded on")
    g.add_argument("--row-subset", default=None, choices=["ct10_half_a", "ct10_half_b"])
    g.add_argument("--row-subset-file", type=Path, default=None)
    r = sub.add_parser("reduce")
    r.add_argument("--rows", nargs="+", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--out-counts", type=Path, required=True)
    r.add_argument("--target-rate", type=float, default=None,
                   help="theta*'s target rate (default the k = 32 headline 0.495513); "
                        "amendment CT §CT24: the arm's own decoder-cosine rate at 0.70")
    f = sub.add_parser("fit")
    f.add_argument("--act-counts", type=Path, required=True)
    f.add_argument("--sim", type=Path, required=True,
                   help="the §CT3 match file (decoder-cosine, gate EXACT)")
    f.add_argument("--rf-counts-all-theta", type=Path, required=True)
    f.add_argument("--m1-json", type=Path, required=True)
    f.add_argument("--gtilde", type=Path, required=True)
    f.add_argument("--null-reductions", type=Path, required=True)
    f.add_argument("--d-fs", type=Path, required=True)
    f.add_argument("--degenerate-flag", type=Path, default=None)
    f.add_argument("--config", type=Path, default=None)
    f.add_argument("--out", type=Path, required=True)
    j = sub.add_parser("jackknife")
    for name in ("--act-counts", "--sim", "--rf-counts-all-theta", "--m1-json",
                 "--gtilde", "--null-reductions", "--d-fs", "--fit-json", "--out"):
        j.add_argument(name, type=Path, required=True)
    j.add_argument("--degenerate-flag", type=Path, default=None)
    j.add_argument("--config", type=Path, default=None)
    j.add_argument("--workers", type=int, default=8)
    pr = sub.add_parser("pairs")
    pr.add_argument("--codes", nargs="+", type=Path, required=True)
    pr.add_argument("--ckpts", nargs="+", type=Path, required=True)
    pr.add_argument("--act-counts", type=Path, required=True)
    pr.add_argument("--dec-sim", type=Path, required=True)
    pr.add_argument("--anchor", type=int, required=True)
    pr.add_argument("--chunk-rows", type=int, default=32768)
    pr.add_argument("--out", type=Path, required=True)
    cv = sub.add_parser("curve")
    for name in ("--act-counts", "--sim", "--rf-counts-all-theta", "--m1-json",
                 "--gtilde", "--null-reductions", "--d-fs", "--out"):
        cv.add_argument(name, type=Path, required=True)
    cv.add_argument("--degenerate-flag", type=Path, default=None)
    cv.add_argument("--config", type=Path, default=None)
    cl = sub.add_parser("classes")
    for name in ("--act-counts", "--act-json", "--sim", "--rf-counts-all-theta",
                 "--m1-json", "--gtilde", "--null-reductions", "--d-fs", "--out"):
        cl.add_argument(name, type=Path, required=True)
    cl.add_argument("--pairs", nargs="+", type=Path, required=True)
    cl.add_argument("--theta-dec", type=float, default=0.70)
    cl.add_argument("--degenerate-flag", type=Path, default=None)
    cl.add_argument("--config", type=Path, default=None)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd in ("fit", "jackknife", "curve", "classes") and args.config is None:
        from src.experiment_config import DEFAULT_CONFIG_PATH
        args.config = DEFAULT_CONFIG_PATH
    {"encode": cmd_encode, "gram": cmd_gram, "reduce": cmd_reduce,
     "fit": cmd_fit, "jackknife": cmd_jackknife, "pairs": cmd_pairs,
     "curve": cmd_curve, "classes": cmd_classes}[args.cmd](args)


if __name__ == "__main__":
    main()
