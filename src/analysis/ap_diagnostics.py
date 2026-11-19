"""
Read-only diagnostics on an existing feature-AP matrix. NO recompute, NO GPU.

Three questions, all answerable from `feature_ap_*.h5` alone:

1. **THE NEGATIVE-DEFINITION TEST.** In the experimental stratum, a residue
   carrying a BY-SIMILARITY label for the same concept is currently an ordinary
   NEGATIVE (see `feature_ap.py`: `n_positives` is the stratum's positive count
   while `n_rows` is the whole subset). A genuine detector fires on it and is
   charged a false positive. If that is what is happening, a concept's
   attainable experimental AP is capped near its *experimental fraction*
   `f_c = P_exp / P_pooled`, so max experimental AP should track `f_c` across
   concepts. Under a masked construction it should not.

   Reported: `f_c` per concept, the OLS/Spearman relationship against max
   experimental AP, the ratio `maxAP_exp / maxAP_pooled` vs `f_c`, and the
   concept ranking (the sanity anchor: under the capped reading the top of the
   experimental ranking is the high-`f_c` concepts, not the interesting ones).

2. **The analysis firing floor** (`confound_control.analysis_min_firing_rows`).
   The floor was set against the 17.7M-row full eval set; this subset is 1.9M
   rows, so the same absolute count is a ~9x stricter rate floor. Reported: how
   many features it excludes, per firing-rate bin, and -- gotcha 4 -- whether
   the excluded set is disproportionately orphan under an alignment PROXY
   (normalised gain only; the FDR half of the pre-registered rule needs the
   null, which does not exist yet).

3. **`g_f` argmax composition.** `g_f` is a max over concepts whose positive
   counts span two orders of magnitude, and small-n concepts win maxima by
   variance alone. Reported: the argmax-concept distribution against concept
   size, with the per-concept positive counts alongside.

Plus a degenerate-value screen (exact 0.5000 / 1.0000 AP was the amendment-B
symptom) and the stratum composition table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.analysis.corrector_scan import BIN_EDGES


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(v):
        o = np.argsort(v, kind="stable")
        r = np.empty_like(o, dtype=np.float64)
        r[o] = np.arange(len(v), dtype=np.float64)
        # average ties
        for val in np.unique(v):
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    return pearson(rank(x), rank(y))


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """slope, intercept, R^2."""
    if x.size < 3 or np.std(x) == 0:
        return float("nan"), float("nan"), float("nan")
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, icpt), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = slope * x + icpt
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(slope), float(icpt), 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def norm_gain(ap: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """G = (AP - p) / (1 - p), broadcast over a (..., n_concepts) AP array."""
    return (ap - prev) / (1.0 - prev)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ap", type=Path, required=True)
    p.add_argument("--labels-json", type=Path, default=None,
                   help="residue_labels.json, for the composition table.")
    p.add_argument("--min-firing-rows", type=int, default=200,
                   help="confound_control.analysis_min_firing_rows")
    p.add_argument("--gain-floor", type=float, default=0.05)
    p.add_argument("--gain-floor-sensitivity", type=float, default=0.01)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    report: dict = {"ap_file": str(args.ap)}

    with h5py.File(args.ap, "r") as hf:
        strata = json.loads(hf.attrs["strata"])
        n_rows = int(hf.attrs["n_rows"])
        scored_row_set = hf.attrs["scored_row_set"]
        seeds = [s.decode() if isinstance(s, bytes) else s for s in hf["seeds"][:]]
        fire = hf["fire_counts"][:]                       # (n_seeds, n_features)
        negatives = hf.attrs.get("negatives", "all_rows")   # absent => pre-flag file
        ap = {st: hf[st]["ap"][:].astype(np.float64) for st in strata}
        prev = {st: hf[st]["prevalence"][:] for st in strata}
        # Under --negatives unannotated the row count differs per concept, so the
        # positive count cannot be recovered as prevalence * n_rows.
        npos = {st: (hf[st]["n_positives"][:] if "n_positives" in hf[st] else None)
                for st in strata}
        names = {st: [c.decode() if isinstance(c, bytes) else c
                      for c in hf[st]["concepts"][:]] for st in strata}
        attrs = {"git_sha": hf.attrs.get("git_sha"),
                 "git_dirty": bool(hf.attrs.get("git_dirty", False))}

    n_seeds, n_features = fire.shape
    print(f"{args.ap.name}: subset={scored_row_set}  n_rows={n_rows:,}  "
          f"seeds={n_seeds}  features={n_features:,}  strata={strata}")
    print(f"  produced by git_sha={attrs['git_sha']} dirty={attrs['git_dirty']}  "
          f"negatives={negatives}")
    report.update(n_rows=n_rows, n_seeds=n_seeds, n_features=n_features,
                  scored_row_set=str(scored_row_set), strata=strata, produced_by=attrs,
                  negatives=str(negatives))

    # The common-concept-set restriction should have made the orderings identical.
    ref = names[strata[0]]
    for st in strata:
        assert names[st] == ref, f"concept ordering differs in {st}"
    concepts = np.array(ref)
    n_concepts = len(concepts)
    print(f"  {n_concepts} concepts, identical ordering across strata")

    # ---------------------------------------------------- 0. degenerate values --
    print("\n" + "=" * 78)
    print("0. DEGENERATE-VALUE SCREEN (the amendment-B symptom)")
    print("=" * 78)
    deg = {}
    for st in strata:
        a = ap[st]
        exact = {v: int(np.isclose(a, v, atol=1e-12).sum())
                 for v in (1.0, 0.5, 1.0 / 3.0)}
        deg[st] = exact
        print(f"  {st:<14} max={a.max():.6f}  exact-1.0 {exact[1.0]}  "
              f"exact-0.5 {exact[0.5]}  exact-1/3 {exact[1.0/3.0]}")
    report["degenerate_counts"] = deg

    # ------------------------------------- 1. the negative-definition test -----
    print("\n" + "=" * 78)
    print("1. NEGATIVE-DEFINITION TEST -- is experimental AP capped by provenance?")
    print("=" * 78)
    if not {"experimental", "pooled"} <= set(strata):
        print("  needs both 'experimental' and 'pooled' strata; skipping")
    else:
        P = {st: (npos[st] if npos[st] is not None
                  else np.rint(prev[st] * n_rows).astype(np.int64))
             for st in strata}
        # Experimental fraction WITHIN the analysed subset. Tiers partition the
        # pooled positives, so this is exactly the share of a concept's positives
        # that the experimental stratum still calls positive.
        f_exp = P["experimental"] / np.maximum(P["pooled"], 1)
        max_exp = ap["experimental"].max(axis=(0, 1))     # over seeds x features
        max_pool = ap["pooled"].max(axis=(0, 1))
        ratio = max_exp / np.maximum(max_pool, 1e-12)
        # A detector that ranks ALL pooled positives at the top scores roughly
        # f_exp in the experimental stratum (experimental positives interleaved
        # among propagated ones). Exceeding it requires separating provenance.
        over = max_exp / np.maximum(f_exp, 1e-12)

        order = np.argsort(-max_exp)
        print(f"\n  {'concept':<26} {'P_pool':>8} {'P_exp':>7} {'f_exp':>7} "
              f"{'maxAP_ex':>9} {'maxAP_pl':>9} {'ratio':>7} {'AP/f':>6}")
        for i in order:
            print(f"  {concepts[i]:<26} {P['pooled'][i]:>8,} {P['experimental'][i]:>7,} "
                  f"{f_exp[i]:>7.3f} {max_exp[i]:>9.4f} {max_pool[i]:>9.4f} "
                  f"{ratio[i]:>7.3f} {over[i]:>6.2f}")

        s1, i1, r1 = ols(f_exp, max_exp)
        s2, i2, r2 = ols(f_exp, ratio)
        print(f"\n  maxAP_exp ~ f_exp   : slope {s1:+.4f}  intercept {i1:+.4f}  "
              f"R2 {r1:.3f}  pearson {pearson(f_exp, max_exp):+.3f}  "
              f"spearman {spearman(f_exp, max_exp):+.3f}")
        print(f"  ratio     ~ f_exp   : slope {s2:+.4f}  intercept {i2:+.4f}  "
              f"R2 {r2:.3f}  pearson {pearson(f_exp, ratio):+.3f}  "
              f"spearman {spearman(f_exp, ratio):+.3f}")
        print(f"\n  concepts whose maxAP_exp EXCEEDS the provenance cap f_exp: "
              f"{int((max_exp > f_exp).sum())}/{n_concepts}")
        print(f"  median maxAP_exp / f_exp = {np.median(over):.3f}   "
              f"(~=1 means the cap is exactly what is being hit)")
        print("\n  READ: a strong positive relationship => experimental AP is being")
        print("  set by label provenance, not by detection. A flat one => not.")
        report["negative_definition_test"] = {
            "concepts": concepts.tolist(),
            "P_pooled": P["pooled"].tolist(),
            "P_experimental": P["experimental"].tolist(),
            "P_propagated": P.get("propagated", np.zeros(n_concepts, int)).tolist(),
            "f_exp": f_exp.tolist(),
            "max_ap_experimental": max_exp.tolist(),
            "max_ap_pooled": max_pool.tolist(),
            "ratio_exp_over_pooled": ratio.tolist(),
            "max_ap_exp_over_f_exp": over.tolist(),
            "ols_maxap_vs_fexp": {"slope": s1, "intercept": i1, "r2": r1,
                                  "pearson": pearson(f_exp, max_exp),
                                  "spearman": spearman(f_exp, max_exp)},
            "ols_ratio_vs_fexp": {"slope": s2, "intercept": i2, "r2": r2,
                                  "pearson": pearson(f_exp, ratio),
                                  "spearman": spearman(f_exp, ratio)},
            "n_exceeding_cap": int((max_exp > f_exp).sum()),
            "median_ap_over_cap": float(np.median(over)),
        }

    # ------------------------------------------- 2. the analysis firing floor --
    print("\n" + "=" * 78)
    print(f"2. ANALYSIS FIRING FLOOR (analysis_min_firing_rows = "
          f"{args.min_firing_rows} on a {n_rows:,}-row subset "
          f"= {args.min_firing_rows / n_rows:.2e} firing rate)")
    print("=" * 78)
    rate = fire / n_rows
    below = fire < args.min_firing_rows
    print(f"  excluded features per seed: mean {below.sum(1).mean():.1f}  "
          f"min {below.sum(1).min()}  max {below.sum(1).max()}  "
          f"({below.sum(1).mean() / n_features:.2%} of the dictionary)")
    print(f"  never-firing features (fire_count == 0): "
          f"mean {(fire == 0).sum(1).mean():.1f} per seed")

    print(f"\n  per firing-rate bin (mean features/seed; 'excl' = below the floor):")
    print(f"  {'bin':>20} {'n_feat':>8} {'excl':>8}")
    bin_rows = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        m = (rate >= lo) & (rate < hi)
        n_b = m.sum(1).mean()
        n_e = (m & below).sum(1).mean()
        bin_rows.append({"lo": lo, "hi": min(hi, 1.0), "n_features": float(n_b),
                         "n_excluded": float(n_e)})
        print(f"  [{lo:.0e},{min(hi,1.0):.0e})".rjust(20) +
              f" {n_b:>8.1f} {n_e:>8.1f}")

    # gotcha 4: is the excluded set disproportionately orphan? Only a PROXY is
    # available -- the FDR half of the classification rule needs the null.
    print(f"\n  gotcha-4 check (PROXY: normalised gain only, no FDR -- the null "
          f"does not exist yet)")
    g4 = {}
    for st in strata:
        G = norm_gain(ap[st], prev[st][None, None, :]).max(axis=2)   # (seeds, feats)
        for floor in (args.gain_floor, args.gain_floor_sensitivity):
            passes = G >= floor
            inc = ~below
            r_inc = passes[inc].mean() if inc.any() else float("nan")
            r_exc = passes[below].mean() if below.any() else float("nan")
            print(f"    {st:<13} G>={floor:<5}  'aligned' above floor "
                  f"{r_inc:6.2%}   below floor {r_exc:6.2%}   "
                  f"(n_below={int(below.sum())})")
            g4[f"{st}_G{floor}"] = {"above_floor_aligned_frac": float(r_inc),
                                    "below_floor_aligned_frac": float(r_exc)}
    print("    => a LOWER rate below the floor means the excluded set is")
    print("       disproportionately ORPHAN, and the floor removes control mass.")
    report["firing_floor"] = {
        "min_firing_rows": args.min_firing_rows,
        "implied_rate": args.min_firing_rows / n_rows,
        "excluded_per_seed_mean": float(below.sum(1).mean()),
        "excluded_per_seed_min": int(below.sum(1).min()),
        "excluded_per_seed_max": int(below.sum(1).max()),
        "never_firing_per_seed_mean": float((fire == 0).sum(1).mean()),
        "bins": bin_rows,
        "gotcha4_proxy": g4,
    }

    # ------------------------------------------ 3. g_f argmax composition ------
    print("\n" + "=" * 78)
    print("3. g_f ARGMAX COMPOSITION (does a small concept win maxima by variance?)")
    print("=" * 78)
    argmax_report = {}
    for st in strata:
        P_st = (npos[st] if npos[st] is not None
                else np.rint(prev[st] * n_rows).astype(np.int64))
        # G0 fix: argmax must run on NORMALIZED gain, not raw AP -- raw AP is
        # inflated by prevalence (a broad concept has high raw AP even at
        # chance), so a raw-AP argmax can attribute a feature to a broad
        # concept even when a narrower concept has the higher GAIN. This is
        # a real, separate issue from null_pilot.py's bug (that one never
        # normalized at all; here normalization exists but argmax bypassed
        # it) -- found while checking whether the two scripts shared a bug.
        G = norm_gain(ap[st], prev[st][None, None, :])
        am = G.argmax(axis=2)                             # (seeds, features)
        counts = np.bincount(am.ravel(), minlength=n_concepts)
        share = counts / counts.sum()
        # restrict to features that clear the effect floor -- those are the ones
        # that would actually be called aligned
        keep = G.max(axis=2) >= args.gain_floor
        counts_k = np.bincount(am[keep], minlength=n_concepts) if keep.any() \
            else np.zeros(n_concepts, dtype=np.int64)
        share_k = counts_k / max(1, counts_k.sum())

        print(f"\n  {st}  ({int(keep.sum())} of {am.size} feature-seeds clear "
              f"G>={args.gain_floor})")
        print(f"  {'concept':<26} {'P':>8} {'argmax%':>9} {'argmax% (G-pass)':>18}")
        for i in np.argsort(-counts):
            if counts[i] == 0 and counts_k[i] == 0:
                continue
            print(f"  {concepts[i]:<26} {P_st[i]:>8,} {share[i]:>8.2%} "
                  f"{share_k[i]:>17.2%}")
        rho = spearman(P_st.astype(float), share)
        print(f"  spearman(concept positives, argmax share) = {rho:+.3f}"
              f"   (NEGATIVE => small concepts are winning the max)")
        argmax_report[st] = {
            "concepts": concepts.tolist(), "positives": P_st.tolist(),
            "argmax_share": share.tolist(),
            "argmax_share_gain_pass": share_k.tolist(),
            "n_gain_pass": int(keep.sum()), "n_feature_seeds": int(am.size),
            "spearman_positives_vs_share": rho,
        }
    report["gf_argmax"] = argmax_report

    # -------------------------------------------------- 4. composition table --
    if args.labels_json and args.labels_json.exists():
        print("\n" + "=" * 78)
        print("4. STRATUM COMPOSITION (A1 shared-protein set vs full eval)")
        print("=" * 78)
        man = json.loads(args.labels_json.read_text())
        a, e = man["composition_all_eval"], man["composition_shared_experimental"]
        print(f"  {'':<22}{'ALL eval':>14}{'shared (exp)':>16}{'ratio':>9}")
        for key, fmt in (("n_proteins", ",d"), ("n_rows", ",d"),
                         ("mean_length", ".1f"), ("median_length", ".1f")):
            print(f"  {key:<22}{a[key]:>14{fmt}}{e[key]:>16{fmt}}"
                  f"{e[key]/max(1e-9, a[key]):>9.2f}x")
        lins = sorted(set(a["lineage"]) | set(e["lineage"]),
                      key=lambda L: -a["lineage"].get(L, 0))
        for L in lins:
            fa = a["lineage"].get(L, 0) / max(1, a["n_proteins"])
            fe = e["lineage"].get(L, 0) / max(1, e["n_proteins"])
            print(f"  {L:<22}{fa:>13.1%}{fe:>15.1%}"
                  f"{fe/max(1e-9, fa):>9.2f}x")
        report["composition"] = {"all_eval": a, "shared_experimental": e}

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
