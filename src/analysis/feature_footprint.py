"""Measurement 3 -- what actually groups features for stage 3?

PREDICTIONS COMMITTED BEFORE THE RUN; the git timestamp is the audit trail.

THE PROBLEM. Stage 3 standardizes `M_f`, a per-FEATURE maximum, so pooling
null draws to give it resolution requires features whose null-max
distributions are exchangeable. Firing rate demonstrably does not deliver
that: within a single firing bin, per-feature mean `M_null` spans **33x to
225x** (amendment R7).

THE HYPOTHESIS BEING TESTED. `M_f` is a max over 703 concepts, so its null
depends on how many concepts the feature could plausibly score against at
all -- which is driven by ACTIVATION FOOTPRINT, not firing rate. Two features
firing on the same number of residues differ completely if one concentrates
them inside a handful of annotated proteins and the other scatters them
across thousands: the first gets few effective comparisons and a low
`M_null`, the second many and a high one.

CANDIDATE GROUPING VARIABLES, all from the row index already on disk:
  - `n_proteins`  -- distinct proteins the feature fires in;
  - `entropy`     -- Shannon entropy of the feature's firing counts across
                    proteins (scatter vs concentration), and its exponential
                    (perplexity), the "effective number of proteins";
  - `gini`        -- concentration of the same distribution;
  - `fire_rate`   -- the incumbent, for comparison.

-----------------------------------------------------------------------------
PREDICTIONS
-----------------------------------------------------------------------------
  1. `n_proteins` and `perplexity` will each predict per-feature mean
     `M_null` BETTER than firing rate does. I expect |Spearman| > 0.5 for the
     footprint variables against < 0.35 for firing rate.

  2. The relationship will be NEGATIVE for concentration and POSITIVE for
     spread: a feature scattered over many proteins gets more chances for a
     shuffled concept to land on its rows, so a higher null max.

  3. **No single variable will fix exchangeability.** Residual within-group
     spread will stay above 10x on the same 33x-225x diagnostic. Reason: the
     null max is a max over 703 heterogeneous concepts, and no scalar
     summary of a feature determines the whole shape of that maximum. If
     that holds, pooled exceedance needs either a multivariate grouping or a
     per-feature parametric model (measurement 4) rather than a better
     single axis.

  What would falsify prediction 3, and I would count it as a clean win for
  pooling: residual spread under ~5x within groups defined by the best
  variable, which would make stage-3 pooling straightforwardly defensible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr


def population_order(labels_h5: Path) -> np.ndarray:
    """Rank in START order -> index in the STORED (accession) order.

    The store writes its population arrays in `sorted(accession)` order
    (`build_residue_labels.py`: `pop_accs = sorted(protein_meta)`), while the
    row axis is ordered by ascending cache row. Any caller that needs a
    per-protein ATTRIBUTE (an accession, a cluster, a taxon) alongside
    `protein_of_row`'s ids must compose through this permutation, because those
    ids are ranks in start order and not positions in the stored arrays.

    Returned separately rather than as a third element of `protein_of_row` so
    the existing two-value signature is untouched -- a helper that grows a
    return value while a caller keeps unpacking the old arity is a defect this
    project has already shipped once (gotcha 13 extension 4, instance 3).
    """
    with h5py.File(labels_h5, "r") as hf:
        start = hf["protein_sets"]["population_start"][:]
    return np.argsort(start, kind="stable")


def protein_of_row(labels_h5: Path) -> tuple[np.ndarray, int]:
    """Row POSITION -> protein id, over the store's own population.

    CORRECTED (PREREGISTRATION amendment AO6). This function previously did

        np.repeat(np.arange(n_prot), population_n_rows)

    on the arrays AS STORED. The store's population arrays are in
    `sorted(bare accession)` order, but the row POSITIONS a row-index carries
    are in ascending CACHE ROW order, because `feature_ap.load_scored_row_set`
    returns `np.sort(rows)`. The two orders disagree -- `population_start` has
    25,508 descending steps of 51,942 -- so the old map blocked the row axis
    into runs whose LENGTHS were the protein lengths in a scrambled order.
    Measured: **489 of 15,972,325 row positions landed in the right protein
    (0.0031%)**, and 0 of 51,943 proteins had the right interval.

    The old assertion (`n_rows == len(prot_of_row)`) could not see it: both
    orders have the same TOTAL, and a check on a total cannot detect a
    permutation. Hence the checks below are on ORDER and on NON-OVERLAP.

    THE CORRECT CONSTRUCTION, and it does not depend on the population being
    contiguous in cache-row space: row positions index `np.sort(rows)`, so
    taking proteins in ascending-`start` order, protein at rank `i` occupies
    positions `[sum of n_rows over ranks < i, + n_rows[i])`. That is exactly a
    repeat in START order. Ids are therefore RANKS IN START ORDER; compose
    through `population_order` for anything per-protein.
    """
    with h5py.File(labels_h5, "r") as hf:
        ps = hf["protein_sets"]
        start = ps["population_start"][:].astype(np.int64)
        nrow = ps["population_n_rows"][:].astype(np.int64)
    order = np.argsort(start, kind="stable")
    s, n = start[order], nrow[order]
    # Overlapping protein row ranges would make the row axis ambiguous, and no
    # grouping of it would be meaningful. Refuse rather than block anyway.
    if s.size > 1 and np.any(s[:-1] + n[:-1] > s[1:]):
        bad = int(np.flatnonzero(s[:-1] + n[:-1] > s[1:])[0])
        raise SystemExit(
            f"population protein row ranges OVERLAP at start-order rank {bad} "
            f"([{s[bad]}, {s[bad] + n[bad]}) then [{s[bad + 1]}, ...)). The row "
            "axis cannot be partitioned by protein; refusing to produce a "
            "grouping.")
    if np.any(n < 0):
        raise SystemExit("population_n_rows contains a negative length")
    prot = np.repeat(np.arange(len(s), dtype=np.int32), n)
    # The grouping must be non-decreasing for `footprint`'s `searchsorted` to
    # find run boundaries at all. Asserted rather than assumed, because the
    # pre-AO6 map violated it silently for every real store.
    assert np.all(np.diff(prot) >= 0), "protein ids are not non-decreasing"
    return prot, len(s)


def footprint(row_index_h5: Path, prot_of_row: np.ndarray, n_prot: int,
              n_features: int) -> dict:
    """Per-feature protein-level footprint, streamed protein by protein.

    STREAMED RATHER THAN TRANSPOSED. The row index is row-major
    (`row_offsets` + `feat_by_row`, ~511M entries), and materialising the
    feature-major transpose would cost ~4 GB and a full argsort. Entropy and
    Gini both need only per-feature accumulators, so one pass over proteins
    with `bincount` suffices: state is O(n_features), not O(nnz).
    """
    with h5py.File(row_index_h5, "r") as hf:
        offs = hf["row_offsets"][:]
        feats = hf["feat_by_row"]
        n_rows = len(offs) - 1
        assert n_rows == len(prot_of_row), (n_rows, len(prot_of_row))

        n_proteins = np.zeros(n_features, dtype=np.int64)
        total = np.zeros(n_features, dtype=np.float64)
        sum_clogc = np.zeros(n_features, dtype=np.float64)
        sum_c2 = np.zeros(n_features, dtype=np.float64)

        # protein p occupies a contiguous run of rows
        bounds = np.searchsorted(prot_of_row, np.arange(n_prot + 1))
        CHUNK = 512
        for lo in range(0, n_prot, CHUNK):
            hi = min(lo + CHUNK, n_prot)
            for p in range(lo, hi):
                r0, r1 = bounds[p], bounds[p + 1]
                if r1 <= r0:
                    continue
                f = feats[offs[r0]:offs[r1]]
                if f.size == 0:
                    continue
                c = np.bincount(f, minlength=n_features).astype(np.float64)
                nz = c > 0
                n_proteins[nz] += 1
                total += c
                sum_clogc[nz] += c[nz] * np.log(c[nz])
                sum_c2 += c * c
    with np.errstate(invalid="ignore", divide="ignore"):
        entropy = np.log(total) - sum_clogc / total          # Shannon, nats
        perplexity = np.exp(entropy)
        # Simpson-style concentration; 1/sum(p^2) is the effective count.
        conc = sum_c2 / (total ** 2)
    return {"n_proteins": n_proteins.astype(float), "total_fires": total,
            "entropy": entropy, "perplexity": perplexity,
            "concentration": conc}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gtilde", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--null-reductions", type=Path, required=True)
    p.add_argument("--row-index", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    with h5py.File(args.gtilde, "r") as h:
        Mn = h["M_null"][args.seed][:]            # (shifts, features)
        _fre = h.attrs["firing_rate_edges"]
        if isinstance(_fre, bytes):
            _fre = _fre.decode()
        fr_edges = json.loads(_fre) if isinstance(_fre, str) else list(_fre)
    with h5py.File(args.null_reductions, "r") as h:
        fire = h["fire_rate"][args.seed][:]
    n_features = Mn.shape[1]
    target = Mn.mean(axis=0)                       # per-feature mean M_null

    prot_of_row, n_prot = protein_of_row(args.labels)
    print(f"population: {n_prot:,} proteins / {len(prot_of_row):,} rows")
    fp = footprint(args.row_index, prot_of_row, n_prot, n_features)
    fp["fire_rate"] = fire

    print("\n" + "=" * 74)
    print("M3. WHAT PREDICTS PER-FEATURE mean M_null?")
    print("=" * 74)
    order = []
    for nm in ("fire_rate", "n_proteins", "perplexity", "entropy",
               "concentration", "total_fires"):
        v = fp[nm]
        ok = np.isfinite(v) & np.isfinite(target)
        r = spearmanr(v[ok], target[ok]).statistic
        order.append((abs(r), nm, r))
        tag = "  <- incumbent" if nm == "fire_rate" else ""
        print(f"  Spearman({nm:14s}, mean M_null) = {r:+.3f}{tag}")
    order.sort(reverse=True)
    best = order[0][1]
    print(f"\n  BEST single predictor: {best} (|rho| = {order[0][0]:.3f})")

    # Residual spread within groups defined by the best variable -- the SAME
    # 33x-225x diagnostic R7 applied to firing bins, so it is like-for-like.
    print("\n  residual within-group spread, deciles of the best predictor:")
    v = fp[best]
    ok = np.isfinite(v)
    edges = np.percentile(v[ok], np.arange(10, 100, 10))
    grp = np.digitize(v, edges)
    spreads = []
    for gth in range(10):
        m = ok & (grp == gth)
        if m.sum() < 20:
            continue
        t = target[m]
        t = t[t > 0]
        if t.size < 20:
            continue
        ratio = t.max() / t.min()
        spreads.append(ratio)
        print(f"    decile {gth}: {int(m.sum()):5d} features, mean M_null "
              f"{t.min():.2f} to {t.max():.2f} ({ratio:.1f}x), "
              f"CV {t.std()/t.mean():.3f}")
    print(f"\n  median within-decile spread: {np.median(spreads):.1f}x")
    print("  (R7 measured 33x-225x within FIRING bins -- like-for-like)")
    if np.median(spreads) > 10:
        print("  -> prediction 3 HOLDS: no single variable fixes "
              "exchangeability.")
    else:
        print("  -> prediction 3 FALSIFIED: grouping on this variable makes "
              "pooling defensible.")

    rep = {"spearman": {nm: float(r) for _, nm, r in order},
           "best": best,
           "median_within_decile_spread": float(np.median(spreads))}
    args.out.write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
