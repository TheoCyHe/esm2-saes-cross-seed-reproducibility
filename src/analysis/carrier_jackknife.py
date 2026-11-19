"""Dictionary-level intervals for the carrier excess (PREREGISTRATION amendment
CT, §CT19 -- the brief's CT11).

THE OBJECT. Appendix G's carrier excess: the carrier "has ANY partner" rate `A`
(0.561691) minus its firing-rate-matched ambient, over the (concept, seed-pair)
cells of the 190 unordered anchor pairs at theta = 0.70, `n_compare = 19`
(§BB9.2): **+7.2219 pp singly matched** (9 firing cells) and **+6.8807 pp
triply matched** (firing x support x concentration, 81 cells). The submitted
paper carries NO interval for either (§CT2-R); §CT19 declares three.

WHY PER-PAIR SUMS ARE EXACT. Every input to `carrier_rates_by_cell` is a
property of ONE seed (carrier, detection, cell index, the cell's size and the
matched-ambient numerator for seed `s` against `t`) or of ONE PAIR (the MNN
partner matrix, which depends only on the two dictionaries). The excess is
`100 * (sum a_hits / sum n_eval - sum matched / sum matched_n)` over pairs, so
removing anchor `d` as anchor AND as comparator is exactly the same sums over
the 171 pairs without `d`, and anchor `s`'s across-seed value is the sums over
the 19 pairs with `s`. `per_pair_cell_stats` keeps the sums per pair; summed
over all 190 in `carrier_rates_by_cell`'s pair order they must reproduce it
EXACTLY (the full-run control).

THE DECLARED MATCHING RE-RUN (§CT19). For each removed anchor the MNN partner
matrix is recomputed on the remaining 19 decoders and must equal the full
matrix restricted to them, cell for cell -- a gate that can fail (gotcha 35's
run-to-run concern), not a slice that cannot.

THE INTERVALS, as §CT19 defines them before any value: jackknife over the 20
anchors (SE `sqrt((n-1)/n * sum(loo - mean)^2)`, estimate +- 1.959964 SE, bias
reported not applied); across-seed (per-anchor excess over its 19 pairs, mean
+- 1.959964 SD/sqrt(20) -- each cell enters two anchors); cell-independence
(`sqrt(A(1-A)/N)`, the matched ambient held fixed), labelled as assuming
independent cells.

`--single-only` (amendment CT, §CT24-F item 8): the single firing-matched arm
alone, for an arm with no record to reproduce (the k = 64 dictionaries). It needs
no covariates, so no row index and no `d_fs`; its control is the per-pair sums
reproducing `carrier_rates_by_cell` exactly, and, with `--control-jackknife`, the
whole single-arm result reproducing a stored run (the anchors' §CT19 artifact)
EXACTLY. The triple arm's covariates assume k = 32 (`build_feature_covariates`,
`build_row_indices` read it from configs/mvp.yaml), so the triple arm refuses any
checkpoint whose own meta.json records a different k (gotcha 48).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

Z95 = 1.959964
ARMS = ("single_firing_only", "triple_firing_x_support_x_concentration")


def per_pair_cell_stats(carrier: np.ndarray, detected: np.ndarray,
                        partner: np.ndarray, cell: np.ndarray,
                        n_cells: int) -> dict:
    """`carrier_support.carrier_rates_by_cell`'s loop, with its four running
    sums kept PER UNORDERED PAIR (s < t) instead of totalled. Same pair order,
    same arithmetic per pair, so the totals reproduce it bit for bit."""
    n_seeds = carrier.shape[0]
    has_partner = partner >= 0
    cell_n = np.zeros((n_seeds, n_cells), dtype=np.int64)
    cell_hit = np.zeros((n_seeds, n_seeds, n_cells), dtype=np.int64)
    for s in range(n_seeds):
        cell_n[s] = np.bincount(cell[s], minlength=n_cells)
        for t in range(n_seeds):
            if s == t:
                continue
            cell_hit[s, t] = np.bincount(cell[s], weights=has_partner[s, t],
                                         minlength=n_cells).astype(np.int64)
    pairs, a_hits, n_eval, m_sum, m_n = [], [], [], [], []
    for s in range(n_seeds):
        for t in range(s + 1, n_seeds):
            both = detected[s] & detected[t]
            cs, ct = carrier[s][both], carrier[t][both]
            ah = ne = mn = 0
            ms = 0.0
            if cs.size:
                p = partner[s, t, cs]
                a = p >= 0
                ah, ne = int(a.sum()), int(cs.size)
                cc = cell[s, cs]
                n_in = cell_n[s, cc]
                hit_in = cell_hit[s, t, cc]
                ok = n_in > 1
                if ok.any():
                    num = hit_in[ok] - a[ok].astype(np.int64)
                    den = n_in[ok] - 1
                    ms = float((num / den).sum())
                    mn = int(ok.sum())
            pairs.append((s, t))
            a_hits.append(ah); n_eval.append(ne); m_sum.append(ms); m_n.append(mn)
    return {"pairs": pairs, "a_hits": np.array(a_hits, dtype=np.int64),
            "n_eval": np.array(n_eval, dtype=np.int64),
            "matched_sum": np.array(m_sum, dtype=np.float64),
            "matched_n": np.array(m_n, dtype=np.int64)}


def excess_over(stats: dict, use: np.ndarray) -> dict:
    """The excess over the pairs selected by boolean `use`, summed in pair
    order (Python float accumulation, as `carrier_rates_by_cell` does)."""
    ah = int(stats["a_hits"][use].sum())
    ne = int(stats["n_eval"][use].sum())
    ms = 0.0
    for v in stats["matched_sum"][use]:
        ms += float(v)
    mn = int(stats["matched_n"][use].sum())
    a = ah / ne if ne else float("nan")
    m = ms / mn if mn else float("nan")
    return {"rate_A": a, "matched_ambient": m, "excess_pp": (a - m) * 100.0,
            "n_cells_evaluated": ne, "n_usable": mn}


def pair_mask(pairs: list, keep=None, need=None) -> np.ndarray:
    """Pairs with neither seed outside `keep` (leave-one-out) or with `need`
    among their two seeds (one anchor's across-seed value)."""
    out = []
    for s, t in pairs:
        ok = True
        if keep is not None:
            ok &= (s in keep) and (t in keep)
        if need is not None:
            ok &= need in (s, t)
        out.append(ok)
    return np.array(out, dtype=bool)


def intervals(stats: dict, n_seeds: int) -> dict:
    from src.analysis.jackknife_dictionaries import jackknife
    allp = np.ones(len(stats["pairs"]), dtype=bool)
    full = excess_over(stats, allp)
    loo = [excess_over(stats, pair_mask(stats["pairs"],
                                        keep=set(range(n_seeds)) - {d}))["excess_pp"]
           for d in range(n_seeds)]
    jk = jackknife(loo, full["excess_pp"])
    per_anchor = [excess_over(stats, pair_mask(stats["pairs"], need=s))["excess_pp"]
                  for s in range(n_seeds)]
    sd = float(np.std(per_anchor, ddof=1))
    mean = float(np.mean(per_anchor))
    a, n = full["rate_A"], full["n_cells_evaluated"]
    se_ind = 100.0 * float(np.sqrt(a * (1 - a) / n))
    iv = {"jackknife": [full["excess_pp"] - Z95 * jk["se"], full["excess_pp"] + Z95 * jk["se"]],
          "across_seed": [mean - Z95 * sd / np.sqrt(n_seeds), mean + Z95 * sd / np.sqrt(n_seeds)],
          "cell_independence": [full["excess_pp"] - Z95 * se_ind,
                                full["excess_pp"] + Z95 * se_ind]}
    wk = max(iv, key=lambda k: iv[k][1] - iv[k][0])
    return {"full": full, "jackknife": {**jk, "loo_excess_pp": loo},
            "across_seed": {"per_anchor_excess_pp": per_anchor, "mean": mean, "sd": sd,
                            "note": "each (concept, seed-pair) cell enters two anchors' "
                                    "values; not an independent-anchor interval"},
            "cell_independence": {"se_pp": se_ind,
                                  "label": "assumes independent cells; matched ambient "
                                           "held fixed"},
            "intervals_95_pp": {k: [float(x) for x in v] for k, v in iv.items()},
            "widest": wk, "widest_interval": [float(x) for x in iv[wk]],
            "widest_excludes_0": bool(iv[wk][0] > 0 or iv[wk][1] < 0)}


def build_row_indices(args, n_rows: int) -> None:
    """Rebuild each anchor's `label_store` row index into `--row-index-cache`
    through `row_index_cache.compute_or_load_row_index` -- the builder the
    pooled null uses -- so `smooth_mu.resolve_row_index` finds it by identity.
    Needed because the 20 entries §BB's covariates read had been purged from
    scratch by 2026-10-22 (only seeds 3 and 17 remain). A rebuilt index is a new
    gather: the triple arm's exact-reproduction gate below is what shows
    whether the covariates it yields are the record's."""
    import yaml
    from src.analysis.feature_ap import load_scored_row_set
    from src.analysis.row_index_cache import compute_or_load_row_index
    from src.sae.train import TopKSAE
    cfg = yaml.safe_load(Path("configs/mvp.yaml").read_text())
    d_model, dict_size, k = cfg["model"]["d_model"], cfg["sae"]["dict_size"], cfg["sae"]["k"]
    subset = load_scored_row_set(args.labels, "label_store")
    if subset.size != n_rows:
        raise SystemExit(f"FATAL: label_store has {subset.size} rows, the AP {n_rows}")
    for ck in sorted(args.ckpts, key=lambda p: int(p.name.rsplit("_", 1)[-1])):
        sd = torch.load(ck / "model.pt", map_location="cpu")
        model = TopKSAE(d_model, dict_size, k, b_dec=torch.zeros(d_model)).to(args.device)
        model.load_state_dict(sd)
        model.eval()
        rec: dict = {}
        *_, hit = compute_or_load_row_index(model, ck, args.eval_cache, subset, "label_store",
                                            dict_size, k, args.device, args.row_index_cache,
                                            record=rec)
        print(f"[carrier-jk] row index {ck.name}: {'hit' if hit else 'built'} {rec}", flush=True)
        del model
        if args.device != "cpu":
            torch.cuda.empty_cache()


def run(args) -> dict:
    from src.analysis import carrier_concentration as cc
    from src.analysis.carrier_identity import mnn_partner_matrix
    from src.analysis.jackknife_dictionaries import _sha16
    from src.sae.train import git_dirty, git_sha

    with h5py.File(args.ap, "r") as hf:
        g = hf["pooled"]
        ap_concepts = cc._decode(g["concepts"][:])
        ap_seeds = cc._decode(hf["seeds"][:])
        in_pool = g["in_argmax_pool"][:].astype(bool)
        ap = g["ap"][:]
        prevalence = g["prevalence"][:]
        fire_counts = hf["fire_counts"][:]
        n_rows = int(cc._attrs(hf)["n_rows"])
    with h5py.File(args.null, "r") as hf:
        null_concepts = cc._decode(hf["concepts"][:])
        null_seeds = cc._decode(hf["seeds"][:])
        null_cmax = hf["null_cmax_b"][:]
    cc.assert_labels_align("seed labels (AP vs null)", ap_seeds, null_seeds)
    pool_concepts = [c for c, k in zip(ap_concepts, in_pool) if k]
    cc.assert_labels_align("concept labels (AP pool vs null)", pool_concepts, null_concepts)
    if not args.single_only:
        with h5py.File(args.d_fs, "r") as hf:
            d_raw = hf["d_fs_cluster_entropy"][:]
            d_seeds = [int(x) for x in hf["seeds"][:]]
        d_fs = d_raw[cc.align_to_ap_order([str(x) for x in d_seeds], ap_seeds, str(args.d_fs))]

    meta_seed, loaded = {}, {}
    for p in args.ckpts:
        W, meta = cc.load_decoder(p, args.device)
        meta_seed[p] = int(meta["seed"])
        loaded[p] = W
        if not args.single_only and int(meta.get("k", -1)) != 32:
            # gotcha 48: the triple arm's covariates read k from configs/mvp.yaml
            raise SystemExit(f"FATAL: {p} records k={meta.get('k')!r}; the triple arm's "
                             "covariates assume k = 32 -- use --single-only")
    ordered = cc.order_decoders_to(ap_seeds, meta_seed)
    decoders = [loaded[p] for p in ordered]
    cc.assert_labels_align("decoder seed order vs AP seed labels",
                           [f"seed_{meta_seed[p]}" for p in ordered], ap_seeds)
    lex_to_numeric = np.argsort([meta_seed[p] for p in ordered])
    n = len(decoders)
    if n != args.expect_n:
        raise SystemExit(f"FATAL: {n} anchors, expected {args.expect_n}")
    partner, mnn_sim = mnn_partner_matrix(decoders, args.theta)
    acc = cc.verify_counts_bit_identical(mnn_sim, n, args.theta,
                                         args.counts_reference, lex_to_numeric)
    if acc["verdict"] != "BIT_IDENTICAL":
        raise SystemExit("ACCEPTANCE (counts) FAILED " + json.dumps(acc))

    ap_pool = ap[:, :, in_pool]
    best_gain = cc.to_gain(cc.observed_best_ap(ap_pool), prevalence[in_pool])
    detected = cc.detection_matrix(best_gain, cc.null_thresholds(
        null_cmax, cc.REFERENCE_PERCENTILE, "pooled"))
    carrier, _ = cc.carriers_from_ap(ap_pool)
    fbin = cc.firing_bin(fire_counts.astype(np.float64) / float(n_rows))
    n_f = len(cc.BIN_EDGES) - 1
    cells = {"single_firing_only": (cc.cell_index(fbin, None, cc.N_SUPPORT_BINS), n_f)}
    if not args.single_only:
        sbin = cc.support_bin_within(fbin, d_fs, cc.N_SUPPORT_BINS)
        if args.build_row_index:
            build_row_indices(args, n_rows)
        perp, conc, prov = cc.build_feature_covariates(
            ap_seeds, args.ckpts_root, args.row_index_cache, args.eval_cache, args.labels,
            args.labels.with_suffix(".json"), pool_concepts, n_rows, int(ap_pool.shape[1]),
            "label_store", 32)
        if prov["problems"]:
            raise SystemExit("FATAL: covariate provenance " + json.dumps(prov))
        cbin = cc.support_bin_within(fbin, conc, cc.N_CONC_BINS)
        cells["triple_firing_x_support_x_concentration"] = (
            (np.asarray(fbin, dtype=np.int64) * cc.N_SUPPORT_BINS
             + np.asarray(sbin, dtype=np.int64)) * cc.N_CONC_BINS
            + np.asarray(cbin, dtype=np.int64),
            n_f * cc.N_SUPPORT_BINS * cc.N_CONC_BINS)

    stored = (json.loads(args.reference.read_text())["section_3b_matched_ambients"]["arms"]
              if args.reference is not None else None)
    out = {"identity": {"measurement": "carrier_excess_dictionary_level_intervals",
                        "governing_amendment": "CT19 (the brief's CT11)",
                        "theta": args.theta, "n_anchors": n, "n_compare": n - 1,
                        "ap_seed_labels_lexicographic": ap_seeds,
                        "reference": str(args.reference),
                        "reference_sha256_16": (_sha16(args.reference)
                                                if args.reference is not None else None),
                        "single_only": bool(args.single_only),
                        "device": args.device, "git_sha": git_sha(),
                        "git_dirty": git_dirty()},
           "acceptance_counts_vs_rf_counts_n19": acc["verdict"], "arms": {}}
    for name in [a for a in ARMS if a in cells]:
        cell, nc = cells[name]
        full_ref = cc.carrier_rates_by_cell(carrier, detected, partner, cell, nc)
        st = per_pair_cell_stats(carrier, detected, partner, cell, nc)
        tot = excess_over(st, np.ones(len(st["pairs"]), dtype=bool))
        ctrl = {
            "per_pair_sums_reproduce_carrier_rates_by_cell": bool(
                tot["rate_A"] == full_ref["rate_A_carrier_has_any_partner"]
                and tot["matched_ambient"] == full_ref["matched_ambient"]),
            "full_run_reproduces_stored_excess": (bool(
                full_ref["excess_A_minus_matched_pp"] == stored[name]["excess_A_minus_matched_pp"])
                if stored is not None else "NO_REFERENCE"),
            "stored_excess_pp": stored[name]["excess_A_minus_matched_pp"] if stored else None,
            "recomputed_excess_pp": full_ref["excess_A_minus_matched_pp"]}
        if not (ctrl["per_pair_sums_reproduce_carrier_rates_by_cell"]
                and ctrl["full_run_reproduces_stored_excess"] in (True, "NO_REFERENCE")):
            out["arms"][name] = {"control": ctrl}
            args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
            raise SystemExit(f"CONTROL FAILED for {name}: {ctrl}")
        out["arms"][name] = {"control": ctrl, **intervals(st, n)}
        print(f"[carrier-jk] {name}: {out['arms'][name]['full']['excess_pp']:+.4f} pp; "
              f"widest {out['arms'][name]['widest']} {out['arms'][name]['widest_interval']}",
              flush=True)

    # the declared matching re-run per removed anchor (§CT19)
    gate, failed = [], 0
    for d in range(n):
        keep = [i for i in range(n) if i != d]
        try:
            p19, _ = mnn_partner_matrix([decoders[i] for i in keep], args.theta)
            same = bool(np.array_equal(p19, partner[np.ix_(keep, keep)]))
        except Exception as e:                      # a failed recomputation is counted
            same = False
            print(f"[carrier-jk] re-run without {ap_seeds[d]} raised {e!r}", flush=True)
        gate.append({"removed": ap_seeds[d], "partner_matrix_equal": same})
        failed += int(not same)
    out["loo_matching_rerun"] = {"per_removed_anchor": gate,
                                 "n_failed": failed, "n": n}
    if not args.single_only:
        out["CT19_P1_widest_excludes_0_for_7_22"] = (
            out["arms"]["single_firing_only"]["widest_excludes_0"] if failed <= 2 else None)
    if args.control_jackknife is not None:
        ref = json.loads(args.control_jackknife.read_text(encoding="utf-8"))["arms"]["single_firing_only"]
        mine = out["arms"]["single_firing_only"]
        keys = ("full", "intervals_95_pp", "widest", "widest_interval")
        same = {k: json.dumps(mine[k], sort_keys=True) == json.dumps(ref[k], sort_keys=True)
                for k in keys}
        out["control_vs_stored_single_arm"] = {**same, "verdict":
                                               "EXACT" if all(same.values()) else "DIFFERS"}
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[carrier-jk] control vs stored single arm: {out['control_vs_stored_single_arm']}",
              flush=True)
        if not all(same.values()):
            raise SystemExit("CONTROL FAILED: the stored single arm is not reproduced")
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[carrier-jk] re-run gate: {n - failed} of {n} equal", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("--ap", "--null", "--counts-reference", "--out"):
        p.add_argument(k, type=Path, required=True)
    # the triple arm's inputs; not needed with --single-only
    for k in ("--d-fs", "--ckpts-root", "--row-index-cache", "--eval-cache", "--labels",
              "--reference"):
        p.add_argument(k, type=Path, default=None)
    p.add_argument("--single-only", action="store_true",
                   help="amendment CT §CT24-F item 8: the single firing-matched arm only")
    p.add_argument("--control-jackknife", type=Path, default=None,
                   help="a stored carrier_jackknife JSON whose single arm must be "
                        "reproduced EXACTLY (the anchors' §CT19 run)")
    p.add_argument("--expect-n", type=int, default=20)
    p.add_argument("--ckpts", nargs="+", type=Path, required=True)
    p.add_argument("--theta", type=float, default=0.70)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--build-row-index", action="store_true",
                   help="rebuild the anchors' label_store row indices first (see "
                        "build_row_indices)")
    args = p.parse_args()
    if not args.single_only:
        missing = [k for k in ("d_fs", "ckpts_root", "row_index_cache", "eval_cache",
                               "labels", "reference") if getattr(args, k) is None]
        if missing:
            p.error(f"the triple arm needs {missing} (or pass --single-only)")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    run(args)


if __name__ == "__main__":
    main()
