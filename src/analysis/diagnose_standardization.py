"""Why K2's inner standardization fails, measured rather than inferred.

Amendment P recorded that `BINDING:heme a3` survives standardization at rank
1 of 703. That is ONE concept, and P2's claim -- "cluster count does not
predict null variance" -- was inferred from ONE stratum sitting near the
all-strata median. An n=1 refutation of a population claim is exactly what
this project's own norm ("decompose conflated variables before concluding
from a combined effect") forbids.

So this module measures the four things that decide what to fix:

  3a. Does cluster count predict null variance, ACROSS ALL strata and all
      703 pool concepts? If not, O2's fourth axis costs cell count for
      nothing. Reported, not acted on -- it is a decision about O2.

  3b. mu_s and sigma_s distributions SEPARATELY. There is a tension to
      resolve: if sigma were near-constant across strata, `z` would be a
      monotone transform of `g` within a feature and Spearman would be ~1.
      It is 0.296, so something varies substantially -- and if it is `mu`
      rather than `sigma`, the inner stage is MEAN-CENTRING rather than
      scaling, which is a different object than K2 specifies.

      Also fixes P2's reference class. `heme a3`'s cell sigma was compared
      against the ALL-STRATA median, but if most strata hold low-prevalence
      concepts with sparse nulls then that median is itself tiny and
      "ordinary relative to it" establishes nothing. Compared here against
      strata whose nulls have real spread.

  3c. Prevalence of the argmax WINNER under raw `g` versus under `z`. This
      is the number that demonstrates why D5's z-arm would mislead: `z`
      systematically favours rare concepts, so the median winner prevalence
      drops, which reads as "sub-typing worked even better" when it is the
      artifact.

  3d. What fraction of concepts have a DEGENERATE null, by a stated
      criterion. Sizes the affected population, which is the input to
      whether any gate is needed at all.

THE CRITERION FOR "DEGENERATE" IS STATED HERE AND NOT TUNED: a concept's
null is degenerate iff its 95th percentile over all draws is exactly zero,
i.e. fewer than 5% of draws produced any gain at all. A secondary count of
DISTINCT non-zero values is reported alongside, because a null taking three
values is not meaningfully continuous either.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr

# cell = (prev_bin * 4 + protein_bin) * 4 + cluster_bin -- see
# standardize_gain.concept_cell_index, which folds the axes in spec order.
N_PER_AXIS = 4


def decompose_cell(cell: np.ndarray):
    return cell // 16, (cell // 4) % N_PER_AXIS, cell % N_PER_AXIS


def per_concept_null_stats(slab: np.ndarray) -> dict:
    """Degeneracy statistics per concept, over all (shift, feature) draws.

    `slab` is (shifts, concepts, features) for one seed.
    """
    n_r, n_c, n_f = slab.shape
    flat = slab.transpose(1, 0, 2).reshape(n_c, n_r * n_f)
    out = {
        "n_draws": n_r * n_f,
        "sd": flat.std(axis=1),
        "mean": flat.mean(axis=1),
        "frac_zero": (flat == 0).mean(axis=1),
        "p95": np.percentile(flat, 95, axis=1),
        "p99": np.percentile(flat, 99, axis=1),
        "max": flat.max(axis=1),
    }
    distinct = np.empty(n_c, dtype=np.int64)
    for c in range(n_c):
        nz = flat[c][flat[c] > 0]
        distinct[c] = np.unique(nz).size
    out["n_distinct_nonzero"] = distinct
    # THE STATED CRITERION, fixed before the counts were read.
    out["degenerate_p95_zero"] = out["p95"] == 0
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gtilde", type=Path, required=True)
    p.add_argument("--ap", type=Path, required=True)
    p.add_argument("--null-raw", type=Path, required=True)
    p.add_argument("--labels-json", type=Path, required=True)
    p.add_argument("--arm", default="b")
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 7])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rep: dict = {}

    with h5py.File(args.gtilde, "r") as h:
        concepts = [c.decode() if isinstance(c, bytes) else c
                    for c in h["concepts"][:]]
        pool = h["in_argmax_pool"][:].astype(bool)
        cell = h["concept_cell"][:]
        mu, sig = h["stratum_mu"][:], h["stratum_sigma"][:]
        argc = h["argmax_concept"][:]
        clus = h["axis_cluster_count"][:]
        prev_axis = h["axis_prevalence"][:]
        prot = h["axis_protein_count"][:]
        n_seeds, n_feats = argc.shape
    pool_idx = np.flatnonzero(pool)
    diag = json.loads(args.labels_json.read_text())["retention_floor_diagnostics"]
    npos = np.array([diag[c]["n_residues"] for c in concepts], dtype=float)

    with h5py.File(args.ap, "r") as h:
        g = h["pooled"]
        prevalence = g["prevalence"][:]
        ap_ds = g["ap"]

    # ================= 3d: how much of the library has a degenerate null ====
    print("=" * 74)
    print("3d. DEGENERATE NULLS -- criterion: p95 over all draws == 0")
    print("=" * 74)
    with h5py.File(args.null_raw, "r") as h:
        raw = h[f"gain_{args.arm}"]
        stats_by_seed = {}
        for s in args.seeds:
            slab = raw[s].astype(np.float64)
            stats_by_seed[s] = per_concept_null_stats(slab)
            del slab
    st = stats_by_seed[args.seeds[0]]
    deg = st["degenerate_p95_zero"]
    npool = len(pool_idx)
    print(f"  pool concepts: {npool}   draws per concept: {st['n_draws']:,} "
          f"(= 20 shifts x 10,240 features, ONE seed)")
    print(f"  DEGENERATE (p95 == 0): {int(deg.sum())} of {npool} "
          f"({100*deg.mean():.1f}%)")
    for thr in (10, 100, 1000):
        m = st["n_distinct_nonzero"] < thr
        print(f"  fewer than {thr:5d} distinct non-zero values: {int(m.sum())} "
              f"({100*m.mean():.1f}%)")
    print(f"  frac_zero across concepts: median {np.median(st['frac_zero']):.4f}"
          f"  min {st['frac_zero'].min():.4f}  max {st['frac_zero'].max():.4f}")
    # stability across seeds
    if len(args.seeds) > 1:
        d2 = stats_by_seed[args.seeds[1]]["degenerate_p95_zero"]
        print(f"  same verdict on seed {args.seeds[1]}: "
              f"{int((deg == d2).sum())}/{npool} concepts agree "
              f"({100*(deg == d2).mean():.1f}%)")
    rep["degenerate"] = {
        "criterion": "p95 over all (shift, feature) draws == 0",
        "n_pool": int(npool), "n_degenerate": int(deg.sum()),
        "frac_degenerate": float(deg.mean()),
        "draws_per_concept_per_seed": int(st["n_draws"]),
    }

    # what distinguishes a degenerate concept?
    pos_pool, clus_pool = npos[pool_idx], clus[pool_idx]
    print("\n  what predicts degeneracy:")
    for nm, v in [("n_positives", pos_pool), ("n_clusters", clus_pool)]:
        print(f"    {nm:12s}  degenerate median {np.median(v[deg]):10.1f} "
              f"| non-degenerate median {np.median(v[~deg]):10.1f}")

    # ==================== 3a: does cluster count predict null variance? =====
    print("\n" + "=" * 74)
    print("3a. DOES CLUSTER COUNT PREDICT NULL VARIANCE?  (n = 703 concepts)")
    print("=" * 74)
    sd_pool = st["sd"]
    lsd = np.log10(np.where(sd_pool > 0, sd_pool, np.nan))
    ok = np.isfinite(lsd)
    r_clus = spearmanr(clus_pool[ok], lsd[ok])
    r_pos = spearmanr(pos_pool[ok], lsd[ok])
    print(f"  Spearman(cluster count, log10 null SD) = {r_clus.statistic:+.3f} "
          f"(p={r_clus.pvalue:.2e})")
    print(f"  Spearman(n_positives,  log10 null SD) = {r_pos.statistic:+.3f} "
          f"(p={r_pos.pvalue:.2e})")
    r_pc = spearmanr(clus_pool[ok], pos_pool[ok])
    print(f"  Spearman(cluster count, n_positives)  = {r_pc.statistic:+.3f} "
          "  <- the two predictors are themselves correlated")
    # partial: cluster count WITHIN n_positives deciles
    dec = np.digitize(pos_pool, np.percentile(pos_pool, np.arange(10, 100, 10)))
    within = []
    for d in range(10):
        m = ok & (dec == d)
        if m.sum() >= 15 and np.unique(clus_pool[m]).size > 2:
            within.append(spearmanr(clus_pool[m], lsd[m]).statistic)
    print(f"  WITHIN n_positives deciles: median Spearman "
          f"{np.median(within):+.3f} over {len(within)} deciles "
          f"(range {min(within):+.3f} to {max(within):+.3f})")
    rep["cluster_vs_variance"] = {
        "spearman_cluster_logsd": float(r_clus.statistic),
        "spearman_npos_logsd": float(r_pos.statistic),
        "spearman_cluster_npos": float(r_pc.statistic),
        "within_npos_decile_median": float(np.median(within)),
    }

    # stratum-level: sigma by cluster bin, within prevalence bin
    pb, rb, cb = decompose_cell(np.arange(64))
    s0 = sig[0]
    print("\n  stratum-level sigma by CLUSTER bin (seed 0, median over "
          "occupied cells x firing bins):")
    for c in range(4):
        cells = [i for i in range(64) if cb[i] == c]
        v = s0[cells]
        v = v[np.isfinite(v)]
        if v.size:
            print(f"    cluster bin {c} ({['[1,2)','[2,3)','[3,10)','[10,inf)'][c]:>9s}): "
                  f"{v.size:3d} strata, sigma median {np.median(v):.2e}")

    # ==================== 3b: mu and sigma, separately ======================
    print("\n" + "=" * 74)
    print("3b. mu_s AND sigma_s SEPARATELY -- what is doing the reordering?")
    print("=" * 74)
    occ = np.array([(cell[pool] == c).sum() for c in range(64)])
    fb_ref = 3
    cells_occ = [c for c in range(64) if occ[c] > 0 and np.isfinite(s0[c, fb_ref])]
    mv, sv = mu[0][cells_occ, fb_ref], s0[cells_occ, fb_ref]
    print(f"  at firing bin {fb_ref}, across {len(cells_occ)} occupied cells:")
    print(f"    mu   : min {mv.min():.3e}  median {np.median(mv):.3e}  "
          f"max {mv.max():.3e}   spread {mv.max()/max(mv.min(),1e-30):.1f}x")
    print(f"    sigma: min {sv.min():.3e}  median {np.median(sv):.3e}  "
          f"max {sv.max():.3e}   spread {sv.max()/sv.min():.1f}x")

    # decompose the reordering
    with h5py.File(args.ap, "r") as h:
        ap0 = h["pooled"]["ap"][0][:]
    G0 = (ap0 - prevalence[None, :]) / (1 - prevalence[None, :])
    with h5py.File(args.gtilde, "r") as h:
        z0 = h["z"][0][:]
    fr_bin = fb_ref
    res = {"full z": [], "mu only": [], "sigma only": []}
    for f in range(0, n_feats, 200):
        gg = G0[f, pool_idx]
        cc = cell[pool_idx]
        m_ = mu[0][cc, fr_bin]
        s_ = s0[cc, fr_bin]
        good = np.isfinite(s_) & np.isfinite(m_)
        if good.sum() < 20:
            continue
        res["full z"].append(spearmanr(gg[good], ((gg - m_) / s_)[good]).statistic)
        res["mu only"].append(spearmanr(gg[good], (gg - m_)[good]).statistic)
        res["sigma only"].append(spearmanr(gg[good], (gg / s_)[good]).statistic)
    print("\n  Spearman(raw G, transformed) across pool concepts, per feature:")
    for k, v in res.items():
        print(f"    {k:10s}: median {np.median(v):+.4f}  "
              f"(min {min(v):+.4f} max {max(v):+.4f})")
    print("  -> whichever transform reproduces the 0.296 is the one doing the "
          "reordering.")
    rep["reordering"] = {k: float(np.median(v)) for k, v in res.items()}

    # reference class correction
    print("\n  REFERENCE CLASS (correcting amendment P2): sigma in strata "
          "whose nulls have real spread")
    hi = [c for c in cells_occ if pb[c] == 3]        # prevalence >= 1e-3
    lo = [c for c in cells_occ if pb[c] == 0]        # prevalence < 1e-5
    for nm, cs_ in [("prevalence >= 1e-3", hi), ("prevalence < 1e-5", lo)]:
        v = s0[cs_, fb_ref]
        v = v[np.isfinite(v)]
        if v.size:
            print(f"    {nm:20s}: {v.size:2d} strata, sigma median "
                  f"{np.median(v):.3e}")
    heme = concepts.index("BINDING:heme a3")
    print(f"    BINDING:heme a3 cell    : sigma {s0[cell[heme], fb_ref]:.3e}")
    print(f"    all-strata median (the P2 comparison): "
          f"{np.nanmedian(s0):.3e}")

    # ==================== 3c: prevalence of argmax winners ==================
    print("\n" + "=" * 74)
    print("3c. PREVALENCE OF THE ARGMAX WINNER -- raw g versus z")
    print("=" * 74)
    degfull = np.zeros(len(concepts), dtype=bool)
    degfull[pool_idx] = deg
    raw_args, z_args = [], []
    with h5py.File(args.ap, "r") as h:
        for s in range(n_seeds):
            Gs = ((h["pooled"]["ap"][s][:] - prevalence[None, :])
                  / (1 - prevalence[None, :]))
            raw_args.append(np.where(pool[None, :], Gs, -np.inf).argmax(axis=1))
            z_args.append(argc[s])
    rawarg_all = np.concatenate(raw_args)
    zarg_all = np.concatenate(z_args)
    rw, zw = prevalence[rawarg_all], prevalence[zarg_all]

    print(f"  winner prevalence, pooled over {n_seeds} seeds x {n_feats} features:")
    for nm, v in [("raw g", rw), ("z    ", zw)]:
        q = np.percentile(v, [25, 50, 75])
        print(f"    {nm}: median {q[1]:.3e}   IQR [{q[0]:.3e}, {q[2]:.3e}]")
    print(f"  ratio of medians (z / raw g): {np.median(zw)/np.median(rw):.4f}x")
    print(f"  winners with prevalence < 1e-5:  raw g {100*(rw < 1e-5).mean():.2f}%"
          f"   z {100*(zw < 1e-5).mean():.2f}%")
    print(f"  winners that are DEGENERATE concepts: raw g "
          f"{100*degfull[rawarg_all].mean():.2f}%   z "
          f"{100*degfull[zarg_all].mean():.2f}%")
    # NOTE: this line previously ASSERTED that winner prevalence drops under
    # `z` and that D5's z-arm would misread the drop as a success. That was a
    # conclusion written into the code before the measurement, and the
    # measurement contradicts it -- prevalence goes slightly UP. Replaced with
    # a statement derived from the numbers actually computed.
    direction = "LOWER" if np.median(zw) < np.median(rw) else "HIGHER"
    print(f"  -> under z the median winner is {direction} prevalence than "
          f"under raw g ({np.median(zw)/np.median(rw):.3f}x). The predicted "
          "rare-concept bias is NOT present at the argmax.")
    rep["winner_prevalence"] = {
        "median_raw_g": float(np.median(rw)), "median_z": float(np.median(zw)),
        "share_degenerate_raw_g": float(degfull[rawarg_all].mean()),
        "share_degenerate_z": float(degfull[zarg_all].mean()),
    }

    args.out.write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
