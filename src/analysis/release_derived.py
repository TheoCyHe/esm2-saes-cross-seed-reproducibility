"""Amendment CT §CT18 (b): `release/derived/`, the derived artifacts the
camera-ready's Appendix E.3 names, as flat files with a manifest.

What is written (one row per (anchor seed, feature) unless stated):

  features_n39.csv.gz           match counts at the 13 grid thresholds
                                (`n_compare = 39`), the alignment score `M` and
                                `log M`, the firing rate, `d_fs` and the A4a
                                floor, the argmax concept, the AP artifact's
                                fire count
  mnn_indicator_theta070.csv.gz the per-comparator MNN indicator at 0.70, one
                                column per comparison dictionary (the carrier
                                excess needs which comparators matched, not
                                only how many)
  concepts_carriers.csv.gz      one row per (anchor seed, pool concept): the
                                carrier feature, its AP and gain, the pooled
                                null threshold, detected
  fvu_per_seed.csv              one row per anchor seed: FVU raw and
                                standardised
  meta.json                     the scalars and orders a reader needs
  MANIFEST.csv                  path, sha256, producing commit, owning NoR row

THE SEED AXIS IS WRITTEN AS NUMERIC LABELS, NEVER AS POSITION (gotcha 29). The
feature table is in numeric anchor order. The carrier excess iterates over
unordered pairs `s < t` in the AP artifact's LEXICOGRAPHIC order, and its "has a
partner" event is evaluated in the `s -> t` direction, so the order changes the
number; `meta.json` carries that order and the reproduction uses it.

Nothing is written until five gates pass, each of which can fail:
  1. counts from the persisted MNN similarities equal the stored counts at all
     13 thresholds (§CT3's EXACT gate, through the record's own loader);
  2. the indicator's row sums equal the stored 0.70 counts, and its anchor
     columns' sums equal `rf_counts_theta070_n19.h5`;
  3. `log M` and `log10 phi` rebuilt from the written `M` and firing rate equal
     the loader's arrays bit for bit;
  4. the headline mean `r_f` at 0.70 is NoR §1.1's 0.495513;
  5. the carrier excess recomputed from the indicator, the carriers and the
     firing bins equals `carrier_concentration_bb.json`'s single arm exactly.
Floats are written with `repr`, so they round-trip exactly; the gzip header
carries no name or time, so a rerun on the same inputs writes the same bytes.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import h5py
import numpy as np

THETA_GRID = ("0.30", "0.35", "0.40", "0.45", "0.50", "0.55", "0.60", "0.65",
              "0.70", "0.75", "0.80", "0.85", "0.90")
HEADLINE_NOR = 0.495513

OWNING_NOR = {
    "features_n39.csv.gz": "§1.1 (headline and threshold curve); §2.1 (the bound); "
                           "§2.2 (homology support); §20.1 (the grid fits)",
    "mnn_indicator_theta070.csv.gz": "§20.17 (carrier excess); §1.1 (its row sums)",
    "concepts_carriers.csv.gz": "§3 (concept-level detection); §20.17 (carriers)",
    "fvu_per_seed.csv": "§6.2; §20.2",
    "meta.json": "— (orders and scalars for the files above)",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def fmt(x) -> str:
    """Exact round-trip text for a float, plain text for an int or a bool."""
    if isinstance(x, (bool, np.bool_)):
        return "1" if x else "0"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return repr(float(x))


def write_csv_gz(path: Path, header: list[str], rows) -> None:
    """Deterministic gzip: no file name and mtime 0 in the header."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow(r)
    raw = buf.getvalue().encode("utf-8")
    with open(path, "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:
            g.write(raw)


def indicator_from_sim(sim: np.ndarray, anchor_labels: list, column_labels: list,
                       theta: float) -> np.ndarray:
    """(anchor, column, feature) bool: an MNN partner at or above `theta`; an
    anchor's own column is False (by label, never by position)."""
    own = np.array([[str(c) == str(a) for c in column_labels]
                    for a in anchor_labels], dtype=bool)
    return (sim >= theta) & (~own)[:, :, None]


def partner_surrogate(ind: np.ndarray, order: list[int], anchor_labels: list,
                      column_labels: list) -> np.ndarray:
    """(s, t, feature) int in `order`: 0 where s's feature has an MNN partner in
    t at the threshold, -1 where it has none. Carries the "has a partner" event
    only; partner IDENTITY is not released, so a `partner == carrier` rate
    cannot be computed from it and is not."""
    a_idx = [[str(x) for x in anchor_labels].index(str(s)) for s in order]
    c_idx = [[int(x) for x in column_labels].index(int(s)) for s in order]
    n = len(order)
    out = np.full((n, n, ind.shape[2]), -1, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if i != j:
                out[i, j][ind[a_idx[i], c_idx[j]]] = 0
    return out


def build(args) -> dict:
    from src.analysis import jackknife_dictionaries as jk
    from src.analysis.carrier_identity import (carriers_from_ap, firing_bin,
                                               seed_label_to_int)
    from src.analysis.carrier_support import carrier_rates_by_cell, cell_index
    from src.analysis.concept_repro import (REFERENCE_PERCENTILE,
                                            detection_matrix, null_thresholds,
                                            observed_best_ap, to_gain)
    from src.analysis.design3_join import log_M
    from src.analysis.seed_axis import align_to_reference
    from src.sae.train import git_dirty, git_sha

    out_dir: Path = args.out_dir
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"REFUSING: {out_dir} is not empty")
    commit, dirty = git_sha(), git_dirty()
    if dirty and not args.allow_dirty:
        raise SystemExit("REFUSING: the checkout is dirty; the manifest's "
                         "producing commit would not describe the code")

    # ---- the design-3 inputs, through the record's own loader and gate -----
    inp = jk.load_join_inputs(args)
    anchors = [str(a) for a in inp["anchor_labels"]]
    cols = [int(c) for c in inp["column_labels"]]
    if inp["thetas"] != [float(t) for t in THETA_GRID]:
        raise SystemExit(f"FATAL: qualifying thetas {inp['thetas']}")
    stored, smeta = jk._stored_counts(args.rf_counts_all_theta)
    n_compare = int(inp["sim"].shape[1] - 1)
    gates = {"1_counts_from_sim_equal_stored": "EXACT (load_join_inputs)"}

    with h5py.File(args.gtilde, "r") as hf:
        M_raw = hf["M"][:]
        argmax_raw = hf["argmax_concept"][:]
        gt_concepts = [c.decode() if isinstance(c, bytes) else str(c)
                       for c in hf["concepts"][:]]
        n_rows = int(hf.attrs["n_rows"])
    M = align_to_reference(inp["gtilde_seed_order"], M_raw, anchors)
    argmax_c = align_to_reference(inp["gtilde_seed_order"], argmax_raw, anchors)
    with h5py.File(args.null_reductions, "r") as hf:
        fr_raw = hf["fire_rate"][:]
        null_labels = [(s.decode() if isinstance(s, bytes) else str(s)).split("_")[-1]
                       for s in hf["seeds"][:]]
        null_cmax = hf["null_cmax_b"][:]
        null_concepts = [c.decode() if isinstance(c, bytes) else str(c)
                         for c in hf["concepts"][:]]
    fire_rate = align_to_reference(null_labels, fr_raw, anchors)
    with h5py.File(args.d_fs, "r") as hf:
        d_seeds = [str(int(s)) for s in hf["seeds"][:]]
        d_fs = align_to_reference(d_seeds, hf["d_fs_cluster_entropy"][:], anchors)
        d_ret = align_to_reference(d_seeds, hf["retained_at_floor"][:], anchors)

    # ---- gate 3 ------------------------------------------------------------
    if not np.array_equal(log_M(M, str(args.gtilde)), inp["logM"]):
        raise SystemExit("GATE 3 FAILED: log M from the written M differs")
    if not np.array_equal(np.log10(np.maximum(fire_rate, 1e-12)), inp["log_phi"]):
        raise SystemExit("GATE 3 FAILED: log10 phi from the written firing rate differs")
    if not np.array_equal(d_fs, inp["d_fs"]):
        raise SystemExit("GATE 3 FAILED: d_fs differs from the loader's")
    gates["3_logM_logphi_dfs_rebuilt_bit_identical"] = "PASS"

    # ---- gate 2: the indicator --------------------------------------------
    ind = indicator_from_sim(inp["sim"], anchors, cols, 0.70)
    if not np.array_equal(ind.sum(axis=1), stored[0.70]):
        raise SystemExit("GATE 2 FAILED: indicator row sums != stored 0.70 counts")
    a_cols = [cols.index(int(a)) for a in anchors]
    with h5py.File(args.rf_counts_n19, "r") as hf:
        n19 = hf["match_count"][:]
        n19_labels = [str(s) for s in json.loads(hf.attrs["seed_labels"])]
    n19 = align_to_reference(n19_labels, n19, anchors)
    if not np.array_equal(ind[:, a_cols, :].sum(axis=1), n19):
        raise SystemExit("GATE 2 FAILED: anchor-column sums != rf_counts_theta070_n19")
    gates["2_indicator_sums_equal_n39_and_n19_counts"] = "PASS"

    # ---- gate 4 ------------------------------------------------------------
    headline = float(stored[0.70].mean() / n_compare)
    if round(headline, 6) != HEADLINE_NOR:
        raise SystemExit(f"GATE 4 FAILED: mean r_f at 0.70 = {headline!r}")
    gates["4_headline_0.495513"] = headline

    # ---- carriers and detection, as carrier_concentration computes them ----
    with h5py.File(args.ap, "r") as hf:
        ap_seeds = [s.decode() if isinstance(s, bytes) else str(s)
                    for s in hf["seeds"][:]]
        ap_concepts = [c.decode() if isinstance(c, bytes) else str(c)
                       for c in hf["pooled/concepts"][:]]
        in_pool = hf["pooled/in_argmax_pool"][:].astype(bool)
        ap_pool = hf["pooled/ap"][:][:, :, in_pool]
        prevalence = hf["pooled/prevalence"][:][in_pool]
        fire_counts = hf["fire_counts"][:]
        ap_n_rows = int(hf.attrs["n_rows"])
    null_seed_names = [f"seed_{s}" for s in null_labels]
    if ap_seeds != null_seed_names:
        raise SystemExit("FATAL: AP and null seed orders differ")
    pool_concepts = [c for c, k in zip(ap_concepts, in_pool) if k]
    if pool_concepts != null_concepts:
        raise SystemExit("FATAL: AP pool and null concepts differ")
    order = [seed_label_to_int(s) for s in ap_seeds]           # lexicographic
    best_ap = observed_best_ap(ap_pool)
    gain = to_gain(best_ap, prevalence)
    thr = null_thresholds(null_cmax, REFERENCE_PERCENTILE, "pooled")
    detected = detection_matrix(gain, thr)
    carrier, ties = carriers_from_ap(ap_pool)

    # ---- gate 5: the carrier excess from the RELEASED quantities ----------
    fbin = firing_bin(fire_counts.astype(np.float64) / float(ap_n_rows))
    partner = partner_surrogate(ind, order, anchors, cols)
    from src.analysis.corrector_scan import BIN_EDGES
    single = carrier_rates_by_cell(carrier, detected, partner,
                                   cell_index(fbin, None, 3), len(BIN_EDGES) - 1)
    ref = json.loads(args.carrier_reference.read_text(encoding="utf-8"))
    ref_single = ref["section_3b_matched_ambients"]["arms"]["single_firing_only"]
    for k in ("rate_A_carrier_has_any_partner", "matched_ambient",
              "excess_A_minus_matched_pp", "n_concept_seedpair_cells_evaluated"):
        if single[k] != ref_single[k]:
            raise SystemExit(f"GATE 5 FAILED: {k} {single[k]!r} != {ref_single[k]!r}")
    gates["5_carrier_excess_exact"] = single["excess_A_minus_matched_pp"]

    # ---- FVU ---------------------------------------------------------------
    fvu = json.loads(args.fvu.read_text(encoding="utf-8"))
    fvu_rows = [[s, fvu["per_seed"][s]["raw"]["fvu"],
                 fvu["per_seed"][s]["standardised"]["fvu"]]
                for s in sorted(fvu["per_seed"], key=int)]

    # ---- write ------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    fc_num = align_to_reference([str(s) for s in order], fire_counts, anchors)
    keep = inp["keep"]
    head = (["anchor_seed", "feature"] + [f"k_{t}" for t in THETA_GRID]
            + ["M", "log_M", "fire_rate", "fire_count_ap", "d_fs",
               "d_fs_retained_at_floor", "keep_a4a", "argmax_concept"])

    def feat_rows():
        for i, a in enumerate(anchors):
            for f in range(M.shape[1]):
                yield ([a, f] + [int(stored[float(t)][i, f]) for t in THETA_GRID]
                       + [fmt(M[i, f]), fmt(inp["logM"][i, f]), fmt(fire_rate[i, f]),
                          int(fc_num[i, f]), fmt(d_fs[i, f]), fmt(bool(d_ret[i, f])),
                          fmt(bool(keep[i, f])), gt_concepts[int(argmax_c[i, f])]])
    write_csv_gz(out_dir / "features_n39.csv.gz", head, feat_rows())

    def ind_rows():
        for i, a in enumerate(anchors):
            for f in range(ind.shape[2]):
                yield [a, f] + ["" if str(c) == a else int(ind[i, j, f])
                                for j, c in enumerate(cols)]
    write_csv_gz(out_dir / "mnn_indicator_theta070.csv.gz",
                 ["anchor_seed", "feature"] + [f"m_{c}" for c in cols], ind_rows())

    def conc_rows():
        for si, s in enumerate(order):
            for ci, c in enumerate(pool_concepts):
                yield [s, c, int(carrier[si, ci]), fmt(best_ap[si, ci]),
                       fmt(gain[si, ci]), fmt(thr[ci]), fmt(bool(detected[si, ci]))]
    write_csv_gz(out_dir / "concepts_carriers.csv.gz",
                 ["anchor_seed", "concept", "carrier_feature", "best_ap", "best_gain",
                  "null_threshold_pooled_p95", "detected"], conc_rows())

    with open(out_dir / "fvu_per_seed.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["anchor_seed", "fvu_raw", "fvu_standardised"])
        for r in fvu_rows:
            w.writerow([r[0], fmt(r[1]), fmt(r[2])])

    sources = {}
    for k in ("sim", "rf_counts_all_theta", "rf_counts_n19", "m1_json", "gtilde",
              "null_reductions", "d_fs", "degenerate_flag", "ap",
              "carrier_reference", "fvu"):
        p = getattr(args, k)
        sources[k] = (None if p is None
                      else {"path": str(p), "sha256": sha256_file(p)})
    meta = {
        "governing_amendment": "CT §CT18 (b)",
        "producing_commit": commit, "git_dirty": dirty,
        "n_compare": n_compare, "n_anchors": len(anchors),
        "anchor_seed_order_features": [int(a) for a in anchors],
        "comparison_columns": cols,
        "theta_grid": [float(t) for t in THETA_GRID],
        "reference_theta": 0.70,
        "n_rows_scored": n_rows, "n_rows_ap": ap_n_rows,
        "firing_bin_edges_carrier": [float(x) for x in BIN_EDGES],
        "carrier_pair_order_lexicographic": order,
        "carrier_pair_rule": "unordered pairs s < t in carrier_pair_order; 'has a "
                             "partner' evaluated s -> t with s's carrier; cells with "
                             "both seeds detecting the concept",
        "a4a_floor_rows": 200,
        "concepts_in_pool": len(pool_concepts),
        "carrier_ties": ties,
        "gates": gates,
        "sources": sources,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n",
                                       encoding="utf-8")
    with open(out_dir / "MANIFEST.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["path", "sha256", "producing_commit", "owning_nor_row"])
        for name in ("features_n39.csv.gz", "mnn_indicator_theta070.csv.gz",
                     "concepts_carriers.csv.gz", "fvu_per_seed.csv", "meta.json"):
            w.writerow([f"release/derived/{name}", sha256_file(out_dir / name),
                        commit, OWNING_NOR[name]])
    return meta


def build_parser() -> argparse.ArgumentParser:
    from src.analysis.design3_join import DEFAULT_CONFIG_PATH
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("sim", "rf-counts-all-theta", "rf-counts-n19", "m1-json", "gtilde",
              "null-reductions", "d-fs", "ap", "carrier-reference", "fvu"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--degenerate-flag", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    return ap


def main() -> None:
    meta = build(build_parser().parse_args())
    print(json.dumps(meta["gates"], indent=2))


if __name__ == "__main__":
    main()
