"""G5 three-axis null pooling: prevalence x firing rate x surviving protein
count (PREREGISTRATION amendments G5 and K1).

WHAT THIS ANSWERS, AND WHY IT CAN RUN ON A SUPERSEDED NULL.

G5's deliverable is whether the stratification gives ADEQUATE PER-STRATUM
COUNTS. That is a property of the LIBRARY'S STRUCTURE -- how concepts
distribute over prevalence and protein count, and how features distribute
over firing rate -- not a property of the null's VALUES. So it is answerable
against `null_pilot_g.h5` even though amendment I4 superseded that file
(it maxed over all 716 concepts including the 9 pooled parents, making it a
null for a different statistic than `g_f` now is).

Only two things are read from the null artifact: `fire_rate`, which is a
property of the trained SAEs and is identical under any concept pool, and the
seed/shift counts. No null VALUE is read. Re-running this against the
corrected null takes minutes.

**Every output is marked as computed on the superseded input, in the file's
ATTRIBUTES and not only in its filename** -- an unlabelled artifact derived
from a superseded input is precisely the stale-identity problem amendment I3
and `preflight.py` exist to prevent (CLAUDE.md gotcha 15).

THE ADEQUACY CRITERION IS PRE-DECLARED IN K1, NOT CHOSEN HERE. It is stated
on the STRUCTURAL axes (>=5 concepts, >=20 features per occupied stratum),
not on the nominal draw count, because `n_shifts * n_seeds = 400` makes a
stratum holding one concept and one feature report 400 nominal draws while
`sigma_s` in fact rests on a single concept's idiosyncrasies replicated 400
times. Nominal count is the misleading number; it is reported, never used as
a pass criterion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.analysis.artifact_identity import argmax_pool_mask
from src.analysis.corrector_scan import BIN_EDGES
from src.sae.train import git_dirty, git_sha

# ---- K1 pre-declared bin edges -------------------------------------------
# Fixed in PREREGISTRATION amendment K1, committed BEFORE these counts were
# computed. Do not tune them against the output -- that is the exact failure
# K1 exists to prevent.
PREVALENCE_EDGES = [0.0, 1e-5, 1e-4, 1e-3, 1.0 + 1e-9]
PROTEIN_COUNT_EDGES = [5, 10, 50, 500, np.inf]
FIRING_RATE_EDGES = list(BIN_EDGES)   # verbatim, per K1 axis 2

# ---- N2 independence axis: mmseqs CLUSTER count ------------------------
# Edges are the thresholds already REPORTED in the N8 cost curve, so they
# were fixed before this stratification was computed rather than chosen to
# make the cells look well-populated. Four bins, matching the protein-count
# axis's granularity.
#
# The first two bins are the ones N2's argument turns on: {1} isolates the
# 187 single-cluster concepts (26.3%), and {2} is where `BINDING:heme a3`
# lives. If stratification is going to calibrate those away from the top of
# `g~`, it does it by giving them their own high-variance strata.
CLUSTER_COUNT_EDGES = [1, 2, 3, 10, np.inf]

# ---- K1 pre-declared minima ----------------------------------------------
MIN_CONCEPTS_PER_STRATUM = 5     # primary criterion
MIN_FEATURES_PER_STRATUM = 20    # secondary criterion
INADEQUATE_CONCEPT_SHARE = 0.10  # >=10% of pool in failing strata => INADEQUATE


def bin_index(values: np.ndarray, edges) -> np.ndarray:
    """Half-open [lo, hi) binning. Values below the first edge get -1, which
    callers must treat as out-of-scope rather than as bin 0."""
    idx = np.searchsorted(np.asarray(edges, dtype=np.float64),
                          np.asarray(values, dtype=np.float64),
                          side="right") - 1
    idx = np.where(np.asarray(values, dtype=np.float64) < edges[0], -1, idx)
    return np.clip(idx, -1, len(edges) - 2)


def min_features_per_bin(fire_rate: np.ndarray, lo: float, hi: float) -> int:
    """Feature count in [lo, hi), taken as the MINIMUM OVER SEEDS.

    Per-seed, never cross-seed-averaged. See `collapse_firing_bins` for why
    averaging is invalid here; and the minimum rather than the mean because
    the standardization forms moments per seed (F1.3 fits per seed and
    meta-analyzes), so a stratum must be adequate in EVERY seed, not on
    average across them.
    """
    return int(min(((fire_rate[s] >= lo) & (fire_rate[s] < hi)).sum()
                   for s in range(fire_rate.shape[0])))


def collapse_firing_bins(fire_rate: np.ndarray, edges: list,
                         min_features: int) -> list:
    """K1 axis 2: merge adjacent firing-rate bins FROM THE TOP DOWN until
    every retained bin holds at least `min_features` features in every seed.

    Top-down only, never from the middle -- the direction is pre-declared so
    the merge is a rule rather than a choice made after seeing the counts.

    `fire_rate` is (seeds, features) and is binned PER SEED. Cross-seed
    averaging is INVALID here, and not merely imprecise:

      A feature INDEX means nothing across independently-seeded SAEs. Measured
      on the 20-seed ensemble, the pairwise correlation of `fire_rate` by
      feature index between seeds is **0.0009** (range -0.0067 to 0.0130) --
      i.e. zero. That is expected, and is the entire reason MNN matching has
      to exist at all. So `fire_rate.mean(axis=0)` averages 20 UNRELATED
      random variables: its SD collapses by ~sqrt(20) (measured 3.95x) onto
      the global mean, which TopK pins at exactly `k/dict = 32/10240 =
      0.003125`. The resulting distribution has a MINIMUM of 1.12e-3, so no
      feature can land in any bin below 1e-3 -- the low-firing bins come out
      structurally empty rather than empirically empty.

      That erases exactly the region CLAUDE.md says the care goes into. Per
      seed the tail is well populated: seed 0's minimum firing rate is
      6.4e-5, its median 1.75e-3, with ~2,200 features in [3e-4, 1e-3).

    `null_pilot.py`'s stored `bin_counts` use the mean-based convention and
    report 0 features in each of the three lowest bins for that reason. Those
    counts are an artifact of the averaging, not a property of the ensemble.

    COLLAPSE IS OUTWARD-IN (both tails), NOT TOP-DOWN ONLY -- amendment L1.
    K1 pre-declared top-down merging, but that rule was written against the
    refuted mean-based counts, in which the top bins looked empty and the
    bottom ones did too. Under correct per-seed binning the top bins are
    well populated (88-98 features in [3e-2, 1)) and the sparse bin is the
    BOTTOM one ([0, 1e-4), 2-10 features) -- which a top-down-only rule can
    never reach. Merging proceeds from whichever tail is under-populated,
    inward, and NEVER from the middle, so bulk resolution is preserved.
    """
    edges = list(edges)
    while len(edges) > 2:
        lo_n = min_features_per_bin(fire_rate, edges[0], edges[1])
        hi_n = min_features_per_bin(fire_rate, edges[-2], edges[-1])
        if lo_n < min_features:
            edges = [edges[0]] + edges[2:]      # merge bottom bin upward
        elif hi_n < min_features:
            edges = edges[:-2] + [edges[-1]]    # merge top bin downward
        else:
            break
    return edges


def load_concepts(labels_h5: Path, labels_json: Path):
    """Return (pool_concepts, n_rows, per-concept prevalence and protein
    count), restricted to the I4 ARGMAX POOL.

    The concept list comes from the H5 store -- what was actually built --
    not from the manifest's candidate list, which includes concepts the floor
    and gate dropped.
    """
    with h5py.File(labels_h5, "r") as hf:
        library = sorted(hf["concepts"].keys())
        n_rows = int(hf.attrs["n_rows"])
        # Verify-by-running: the manifest's residue counts are what the
        # prevalence axis rests on, so cross-check them against the stored
        # row sets rather than trusting the sidecar.
        probe = {c: int(hf[f"concepts/{c}/all"].shape[0]) for c in library[:25]}

    manifest = json.loads(labels_json.read_text())
    diag = manifest["retention_floor_diagnostics"]

    mismatches = [(c, probe[c], diag[c]["n_residues"])
                  for c in probe
                  if c in diag and probe[c] != diag[c]["n_residues"]]
    if mismatches:
        raise SystemExit(
            "FATAL: manifest n_residues disagrees with the stored row sets on "
            f"{len(mismatches)} of {len(probe)} probed concepts, e.g. "
            f"{mismatches[:3]}. The prevalence axis would be wrong.")

    keep = argmax_pool_mask(library)
    pool = [c for c, k in zip(library, keep) if k]
    parents = [c for c, k in zip(library, keep) if not k]

    missing = [c for c in pool if c not in diag]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} pool concepts absent from the "
                         f"manifest diagnostics, e.g. {missing[:3]}")

    prevalence = np.array([diag[c]["n_residues"] / n_rows for c in pool])
    protein_count = np.array([diag[c]["n_proteins"] for c in pool], dtype=float)
    # N2 independence axis. -1 means the store was built WITHOUT --clusters;
    # absent is not 0, and a -1 must never be binned as though it were a real
    # count -- callers check and refuse rather than silently stratifying on a
    # sentinel.
    cluster_count = np.array([diag[c].get("n_clusters", -1) for c in pool],
                             dtype=float)
    return (pool, parents, library, n_rows, prevalence, protein_count,
            cluster_count)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--labels-json", type=Path, default=None,
                   help="defaults to --labels with a .json suffix")
    p.add_argument("--null", type=Path, required=True,
                   help="null artifact; ONLY `fire_rate` and the seed/shift "
                        "counts are read, never a null value.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    labels_json = args.labels_json or args.labels.with_suffix(".json")
    (pool, parents, library, n_rows, prevalence, protein_count,
     cluster_count) = load_concepts(args.labels, labels_json)
    has_clusters = bool((cluster_count >= 0).all())
    if not has_clusters:
        n_missing = int((cluster_count < 0).sum())
        print(f"NOTE: {n_missing} concepts carry n_clusters=-1 (store built "
              "without --clusters). The N2 independence axis is UNAVAILABLE "
              "and is SKIPPED -- never silently binned on a sentinel.")

    with h5py.File(args.null, "r") as hf:
        fire_rate = hf["fire_rate"][:]              # (seeds, features)
        n_shifts = int(hf.attrs["n_shifts"])
        null_argmax_pool_n = hf.attrs.get("argmax_pool_n")
        null_excl = hf.attrs.get("exclude_pooled_parents")
        null_superseded = hf.attrs.get("superseded_by")
        null_truncation_policy = hf.attrs.get("truncation_policy")
        null_library_sha = hf.attrs.get("library_sha")
    n_seeds, n_features = fire_rate.shape
    mean_fire = fire_rate.mean(axis=0)

    def _dec(v):
        return v.decode() if isinstance(v, bytes) else v
    null_superseded = _dec(null_superseded)
    null_truncation_policy = _dec(null_truncation_policy)

    print(f"label store : {len(library)} concepts, {len(parents)} pooled "
          f"parents excluded -> ARGMAX POOL {len(pool)}")
    print(f"              n_rows={n_rows:,}  truncation_policy={_dec(null_truncation_policy)}")
    print(f"null artifact: {n_seeds} seeds x {n_shifts} shifts, "
          f"argmax_pool_n={null_argmax_pool_n}, "
          f"exclude_pooled_parents={null_excl}")
    if null_superseded:
        print(f"  *** SOURCE NULL IS SUPERSEDED: {null_superseded}")
        print("  *** Only `fire_rate` is read from it (a property of the "
              "SAEs, invariant to the concept pool). No null VALUE is used.")
    print()

    # ---- axis 2 collapse, per K1's pre-declared top-down rule -------------
    # Binned PER SEED (min over seeds), never on the cross-seed mean -- see
    # collapse_firing_bins' docstring for the measurement that rules the mean
    # out. `mean_fire` is retained ONLY to report the refuted convention
    # alongside, so the difference is visible rather than silently corrected.
    fr_edges = collapse_firing_bins(fire_rate, FIRING_RATE_EDGES,
                                    MIN_FEATURES_PER_STRATUM)
    if len(fr_edges) != len(FIRING_RATE_EDGES):
        print(f"K1 axis-2 collapse: {len(FIRING_RATE_EDGES)-1} base bins -> "
              f"{len(fr_edges)-1} retained (outward-in merge, amendment L1; "
              f"per-seed counts, min over seeds)")
        print(f"  merged edges: {fr_edges}")
    else:
        print("K1 axis-2 collapse: not triggered; all base bins meet the "
              f"{MIN_FEATURES_PER_STRATUM}-feature minimum")
    print()

    prev_bin = bin_index(prevalence, PREVALENCE_EDGES)
    prot_bin = bin_index(protein_count, PROTEIN_COUNT_EDGES)
    # Per-seed feature counts per retained firing bin. The pass criterion uses
    # the MINIMUM over seeds; the mean-based count is carried only to show
    # what the refuted convention would have reported.
    fr_counts = [min_features_per_bin(fire_rate, fr_edges[m], fr_edges[m + 1])
                 for m in range(len(fr_edges) - 1)]
    fr_counts_mean_conv = [int(((mean_fire >= fr_edges[m]) &
                                (mean_fire < fr_edges[m + 1])).sum())
                           for m in range(len(fr_edges) - 1)]

    if (prot_bin < 0).any():
        n_bad = int((prot_bin < 0).sum())
        raise SystemExit(
            f"FATAL: {n_bad} pool concepts have surviving protein count < 5, "
            "below the G3 gate the store claims to enforce. Either the store "
            "is not the gate-5 build or the gate did not apply.")

    n_prev = len(PREVALENCE_EDGES) - 1
    n_prot = len(PROTEIN_COUNT_EDGES) - 1
    n_fr = len(fr_edges) - 1

    strata, failing_concepts = [], set()
    for i in range(n_prev):
        for j in range(n_prot):
            c_mask = (prev_bin == i) & (prot_bin == j)
            n_c = int(c_mask.sum())
            for m in range(n_fr):
                n_f = fr_counts[m]
                if n_c == 0 or n_f == 0:
                    continue                       # unoccupied, not failing
                ok_c = n_c >= MIN_CONCEPTS_PER_STRATUM
                ok_f = n_f >= MIN_FEATURES_PER_STRATUM
                if not (ok_c and ok_f):
                    failing_concepts.update(np.flatnonzero(c_mask).tolist())
                strata.append({
                    "prevalence_bin": [PREVALENCE_EDGES[i], PREVALENCE_EDGES[i + 1]],
                    "protein_count_bin": [PROTEIN_COUNT_EDGES[j],
                                          float(PROTEIN_COUNT_EDGES[j + 1])],
                    "firing_rate_bin": [fr_edges[m], fr_edges[m + 1]],
                    "n_concepts": n_c,
                    "n_features": n_f,                    # min over seeds
                    "n_features_mean_convention": fr_counts_mean_conv[m],
                    "nominal_draws": n_c * n_f * n_shifts * n_seeds,
                    "passes_concept_min": ok_c, "passes_feature_min": ok_f,
                    "passes": ok_c and ok_f,
                })

    occupied = len(strata)
    n_fail = sum(1 for s in strata if not s["passes"])

    # ---- DECOMPOSE the verdict (amendment L2) ----------------------------
    # A joint pass/fail over all three axes conflates two independent
    # failures and is uninterpretable: if ONE firing bin is under-populated,
    # every concept appears in at least one failing stratum and the joint
    # metric reads "100% of concepts failing" -- which says nothing about the
    # concept axes and everything about one bin on the feature axis. The two
    # are separable because concept-axis occupancy does not depend on firing
    # rate and vice versa, so they are reported separately and the joint
    # verdict is their conjunction.
    concept_cell_fail = set()
    concept_cells = []
    for i in range(n_prev):
        for j in range(n_prot):
            n_c = int(((prev_bin == i) & (prot_bin == j)).sum())
            if n_c == 0:
                continue
            ok = n_c >= MIN_CONCEPTS_PER_STRATUM
            if not ok:
                concept_cell_fail.update(
                    np.flatnonzero((prev_bin == i) & (prot_bin == j)).tolist())
            concept_cells.append({
                "prevalence_bin": [PREVALENCE_EDGES[i], PREVALENCE_EDGES[i + 1]],
                "protein_count_bin": [PROTEIN_COUNT_EDGES[j],
                                      float(PROTEIN_COUNT_EDGES[j + 1])],
                "n_concepts": n_c, "passes": ok})
    concept_share = len(concept_cell_fail) / len(pool)
    concept_verdict = ("INADEQUATE" if concept_share >= INADEQUATE_CONCEPT_SHARE
                       else "ADEQUATE")

    firing_fail = [m for m in range(n_fr) if fr_counts[m] < MIN_FEATURES_PER_STRATUM]
    firing_verdict = "INADEQUATE" if firing_fail else "ADEQUATE"

    share = len(failing_concepts) / len(pool)
    verdict = ("ADEQUATE" if concept_verdict == "ADEQUATE"
               and firing_verdict == "ADEQUATE" else "INADEQUATE")

    # Per-seed stability of the firing-rate axis: features move between bins
    # across seeds, and the standardization forms moments per seed, so a
    # scheme that is adequate only at the cross-seed mean is not adequate.
    per_seed_min = []
    for m in range(n_fr):
        counts = [int(((fire_rate[s] >= fr_edges[m]) &
                       (fire_rate[s] < fr_edges[m + 1])).sum())
                  for s in range(n_seeds)]
        per_seed_min.append({"bin": [fr_edges[m], fr_edges[m + 1]],
                             "min_over_seeds": min(counts),
                             "max_over_seeds": max(counts),
                             "mean_convention_REFUTED": fr_counts_mean_conv[m]})

    print(f"{'prevalence':>22} {'proteins':>14} {'firing rate':>18} "
          f"{'#con':>5} {'#feat':>6} {'nominal':>11}  verdict")
    for s in strata:
        pv = f"[{s['prevalence_bin'][0]:.0e},{s['prevalence_bin'][1]:.0e})"
        pc = f"[{int(s['protein_count_bin'][0])},{s['protein_count_bin'][1]:.0f})"
        fb = f"[{s['firing_rate_bin'][0]:.0e},{s['firing_rate_bin'][1]:.0e})"
        flag = "ok" if s["passes"] else (
            "FAIL-concepts" if not s["passes_concept_min"] else "FAIL-features")
        print(f"{pv:>22} {pc:>14} {fb:>18} {s['n_concepts']:>5} "
              f"{s['n_features']:>6} {s['nominal_draws']:>11,}  {flag}")

    print()
    print(f"occupied strata           : {occupied}")
    print(f"failing strata            : {n_fail}")
    print(f"concepts in failing strata: {len(failing_concepts)} of {len(pool)} "
          f"({share:.1%})  [joint, conflated -- see decomposition]")
    print()
    print("DECOMPOSED VERDICT (amendment L2) -- the two axes fail "
          "independently and a joint number hides which:")
    print(f"  concept axes (prevalence x protein count): {concept_verdict}  "
          f"-- {len(concept_cell_fail)} of {len(pool)} concepts "
          f"({concept_share:.1%}) in under-populated cells, "
          f"threshold {INADEQUATE_CONCEPT_SHARE:.0%}")
    print(f"  firing-rate axis                         : {firing_verdict}  "
          f"-- {len(firing_fail)} of {n_fr} retained bins below the "
          f"{MIN_FEATURES_PER_STRATUM}-feature minimum")
    print(f"  JOINT                                    : {verdict}")
    print()
    print("firing-rate bins, PER SEED (the criterion) vs the cross-seed-mean "
          "convention (REFUTED -- feature index is not comparable across "
          "seeds; pairwise corr 0.0009):")
    for r in per_seed_min:
        print(f"  [{r['bin'][0]:.0e},{r['bin'][1]:.0e}): per-seed "
              f"{r['min_over_seeds']:>5}-{r['max_over_seeds']:<5} "
              f"| mean-convention would report {r['mean_convention_REFUTED']:>5}")

    # ---- T3c / N2: does cluster count REPLACE protein count, or sit
    # alongside it? Reported, not assumed. The two are correlated and a fourth
    # axis costs per-stratum sample size directly, so all three schemes are
    # evaluated against K1's SAME structural minima and the cost is shown.
    scheme_report = []
    if has_clusters:
        clus_bin = bin_index(cluster_count, CLUSTER_COUNT_EDGES)
        schemes = {
            "prevalence x protein (K1, current)": [prev_bin, prot_bin],
            "prevalence x cluster (replace)":     [prev_bin, clus_bin],
            "prevalence x protein x cluster (add)": [prev_bin, prot_bin, clus_bin],
        }
        print()
        print("SCHEME COMPARISON (T3c) -- concept-axis occupancy only; the "
              "firing axis is identical across schemes.")
        print(f"  {'scheme':<40} {'cells':>6} {'occupied':>9} "
              f"{'failing':>8} {'concepts in failing':>20}")
        for name, axes in schemes.items():
            keys = list(zip(*[a.tolist() for a in axes]))
            cells: dict = {}
            for i, k in enumerate(keys):
                cells.setdefault(k, []).append(i)
            occupied_cells = len(cells)
            fail_idx = set()
            n_fail_cells = 0
            for k, members in cells.items():
                if len(members) < MIN_CONCEPTS_PER_STRATUM:
                    n_fail_cells += 1
                    fail_idx.update(members)
            total_cells = int(np.prod([len(a_edges) - 1 for a_edges in
                                       ([PREVALENCE_EDGES, PROTEIN_COUNT_EDGES]
                                        if len(axes) == 2 and axes[1] is prot_bin
                                        else [PREVALENCE_EDGES, CLUSTER_COUNT_EDGES]
                                        if len(axes) == 2
                                        else [PREVALENCE_EDGES, PROTEIN_COUNT_EDGES,
                                              CLUSTER_COUNT_EDGES])]))
            share_f = len(fail_idx) / len(pool)
            scheme_report.append({
                "scheme": name, "n_cells": total_cells,
                "occupied": occupied_cells, "failing_cells": n_fail_cells,
                "concepts_in_failing": len(fail_idx),
                "failing_share": share_f,
                "verdict": ("INADEQUATE" if share_f >= INADEQUATE_CONCEPT_SHARE
                            else "ADEQUATE"),
            })
            print(f"  {name:<40} {total_cells:>6} {occupied_cells:>9} "
                  f"{n_fail_cells:>8} {len(fail_idx):>12} "
                  f"({share_f:>5.1%}) {scheme_report[-1]['verdict']}")
        print("  (concept-axis criterion only: >= "
              f"{MIN_CONCEPTS_PER_STRATUM} concepts per occupied cell)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scheme_comparison_T3c": scheme_report,
        "cluster_count_edges": [float(x) for x in CLUSTER_COUNT_EDGES],
        "has_cluster_axis": has_clusters,
        "verdict": verdict,
        "verdict_concept_axes": concept_verdict,
        "verdict_firing_axis": firing_verdict,
        "concept_axis_cells": concept_cells,
        "concepts_in_failing_concept_cells": len(concept_cell_fail),
        "concept_axis_failing_share": concept_share,
        "firing_bins_below_minimum": firing_fail,
        "occupied_strata": occupied,
        "failing_strata": n_fail,
        "n_concepts_pool": len(pool),
        "concepts_in_failing_strata": len(failing_concepts),
        "failing_concept_share": share,
        "strata": strata,
        "per_seed_firing_bin_stability": per_seed_min,
        "edges": {
            "prevalence": PREVALENCE_EDGES,
            "protein_count": [float(x) for x in PROTEIN_COUNT_EDGES],
            "firing_rate_base": FIRING_RATE_EDGES,
            "firing_rate_retained": fr_edges,
        },
        "minima": {
            "concepts_per_stratum": MIN_CONCEPTS_PER_STRATUM,
            "features_per_stratum": MIN_FEATURES_PER_STRATUM,
            "inadequate_concept_share": INADEQUATE_CONCEPT_SHARE,
        },
        # ---- IDENTITY / PROVENANCE (I3, gotcha 15) ----------------------
        # Marked in the payload itself, never only in the filename.
        "source_null": str(args.null),
        "source_null_superseded_by": null_superseded,
        "source_null_argmax_pool_n": (int(null_argmax_pool_n)
                                      if null_argmax_pool_n is not None else None),
        "source_null_exclude_pooled_parents": (bool(null_excl)
                                               if null_excl is not None else None),
        "computed_on_superseded_null": bool(null_superseded),
        "reads_from_null": ["fire_rate", "n_shifts", "seed count"],
        "reads_no_null_values": True,
        "labels": str(args.labels),
        "library_sha": _dec(null_library_sha),
        "truncation_policy": null_truncation_policy,
        "n_rows": n_rows,
        "n_seeds": n_seeds,
        "n_shifts": n_shifts,
        "n_features": n_features,
        "prereg": "amendments G5 + K1",
        "git_sha": git_sha(),
        "git_dirty": bool(git_dirty()),
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")
    if null_superseded:
        print("NOTE: `computed_on_superseded_null` is TRUE in the output. "
              "Re-run against the corrected null before P2 fixes any "
              "threshold value.")


if __name__ == "__main__":
    main()
