"""`d_fs` -- THE REPRESENTATION-DENSITY COVARIATE.
PREREGISTRATION F1.6, ratified in AL8b, built here per the author's decision AM6.5
and amendment AO section 4.

WHY IT IS NOT DECORATION. `d_fs` is the operationalisation of gotcha 17 -- **a
protein count is not an independence count.** A feature supported by many
near-duplicate sequences plausibly gets both more reliable activations (higher
`r_f`) and better AP against concepts enriched in the same homology cluster, so it
is the strongest candidate confound after firing rate. `BINDING:heme a3` is the
worked example: 14 positive residues in 14 cytochrome c oxidase orthologs
spanning **2** mmseqs clusters at 30% identity, credited by the G3 gate with 14
independent placements.

NOT A FIREWALL CROSSING. It reads firing patterns and the cluster assignment.
No `r_f`, no MNN, no AP.

=============================================================================
TWO CANDIDATE DEFINITIONS, AND THE CHOICE IS MADE BY MEASUREMENT
=============================================================================
F1.6 pre-registers the covariate but not WHICH density measure, so that is a
genuine specification choice. Both are computed and their Spearman decides:

  A. **EFFECTIVE cluster count.** `d_fs = H`, the Shannon entropy (nats) of the
     feature's firing mass distributed over mmseqs clusters -- which IS
     `log(exp(H))`, the log of the effective number of clusters. This answers
     gotcha 17's question directly: a feature firing 1,000 times inside one
     cluster has effective support ~1; spread over 1,000 clusters, ~1,000. Same
     functional form as the L1 effective rank settled in AF1.
  B. **RAW distinct-cluster count.** `d_fs = log(number of clusters with any
     firing)`. Cruder, simpler.

**Pre-declared rule (AO4 P8): if Spearman(A, B) > 0.95, take B for simplicity and
record the choice as immaterial BY MEASUREMENT; if not, A is the recommendation
and the choice is the author's.** Log scale either way, since cluster counts span
orders of magnitude and the covariate enters the model linearly.

=============================================================================
THE JOIN IS THE HAZARD, AND IT HAS TWO HALVES
=============================================================================
1. **THE ACCESSION CONVENTION (gotcha 10).** The cache index stores the whole
   FASTA token (`sp|P19918|DCMS_AFICR`); the cluster TSV uses the bare accession
   (`P19918`). The first real annotation build matched **0 of 53,672** proteins
   for exactly this reason and wrote an empty file while exiting 0. Everything
   here normalises through `split_train_eval.header_to_acc`, and the match rate
   is ASSERTED and REPORTED.

2. **THE PROTEIN ORDER (amendment AO6), which is the half that had never been
   checked.** The label store's population arrays are in `sorted(accession)`
   order while the row axis is in ascending cache-row order, and the two
   disagree. So the protein a row belongs to CANNOT be read off the stored order
   -- `feature_footprint.protein_of_row` now blocks in start order and
   `population_order` supplies the permutation back to stored (accession)
   positions. **Getting this wrong is silent**: both orders have the same total,
   which is why the pre-AO6 assertion could not see it.

   And the reconstruction of the accession list is VERIFIED rather than trusted:
   the derived `start` / `n_rows` arrays must equal the store's own, elementwise
   and exactly. If the population filter or the sort key were wrong the arrays
   would disagree, so the check has real content -- it is not a tautology.

=============================================================================
AO6's MAGNITUDE IS MEASURED IN THE SAME PASS
=============================================================================
The per-protein footprint is computed BOTH ways -- correct (start order) and
pre-AO6 (stored order) -- in one sweep, so how far `perplexity` and
`concentration` actually move is a number rather than an argument (AN8). The
common refinement of the two protein partitions is walked once; each segment's
`bincount` serves both accumulators.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import h5py
import numpy as np
import yaml
from scipy.stats import spearmanr

from src.analysis.feature_footprint import population_order, protein_of_row
from src.analysis.smooth_mu import resolve_row_index
from src.annotations.build_residue_labels import load_cache_index, load_cluster_map
from src.data.split_train_eval import header_to_acc

# Gotcha 10's floor, as a number the job fails below. The split partitions ALL of
# Swiss-Prot with `n_unclustered_records = 0`, so anything materially under 100%
# is a convention or release mismatch, not missing data.
MIN_CLUSTER_MATCH_FRACTION = 0.99
# AO4 P8's adoption threshold.
P8_SPEARMAN_THRESHOLD = 0.95
# AO4 P10's materiality threshold for the AO6 magnitude.
P10_MIN_MEDIAN_REL_CHANGE = 0.10
# Positions read per buffered block. 200k positions is ~6.4M index entries at
# k = 32, so the flat read is tens of MB and the number of reads is ~100/seed.
BLOCK_POSITIONS = 200_000


def population_accessions(eval_cache: Path, labels_h5: Path) -> dict:
    """Population accessions in START order, with the reconstruction VERIFIED.

    The store does not persist `population_accessions` -- only `population_start`
    and `population_n_rows`, in `sorted(bare accession)` order. So the accession
    list is rebuilt the way `build_residue_labels.py` builds it and then CHECKED
    against those two arrays elementwise. A mismatch means the population filter
    or the sort key is wrong, and the job refuses rather than joining clusters
    onto the wrong proteins.
    """
    with h5py.File(labels_h5, "r") as hf:
        ps = hf["protein_sets"]
        stored_start = ps["population_start"][:].astype(np.int64)
        stored_nrow = ps["population_n_rows"][:].astype(np.int64)
        policy = hf.attrs.get("truncation_policy")
        policy = policy.decode() if isinstance(policy, bytes) else policy
        store_n_rows = int(hf.attrs["n_rows"])

    index, _ = load_cache_index(eval_cache)
    by_acc = {header_to_acc(e["accession"]): e for e in index}
    n_all = len(by_acc)
    if policy == "untruncated":
        by_acc = {a: e for a, e in by_acc.items() if not e["truncated"]}
    elif policy != "with_truncated":
        raise SystemExit(f"unknown truncation_policy {policy!r} in {labels_h5}")

    pop_accs = sorted(by_acc)                       # the store's own order
    derived_start = np.array([by_acc[a]["start"] for a in pop_accs], np.int64)
    derived_nrow = np.array([by_acc[a]["n_rows"] for a in pop_accs], np.int64)

    problems = []
    if derived_start.size != stored_start.size:
        problems.append(f"protein count: derived {derived_start.size} vs stored "
                        f"{stored_start.size}")
    else:
        nbad_s = int((derived_start != stored_start).sum())
        nbad_n = int((derived_nrow != stored_nrow).sum())
        if nbad_s or nbad_n:
            problems.append(f"{nbad_s} start values and {nbad_n} row counts "
                            f"disagree with the store")
    if int(derived_nrow.sum()) != store_n_rows:
        problems.append(f"row total: derived {int(derived_nrow.sum())} vs the "
                        f"store's n_rows {store_n_rows}")
    if problems:
        raise SystemExit(
            "the population reconstruction does NOT match the label store:\n  * "
            + "\n  * ".join(problems) +
            "\nThe accession order is what maps a row to a protein and a protein "
            "to a cluster. Refusing to join on an unverified order (AO6).")

    order = population_order(labels_h5)
    return {
        "accessions_stored_order": pop_accs,
        "accessions_start_order": [pop_accs[i] for i in order],
        "stored_start": stored_start, "stored_nrow": stored_nrow,
        "order": order,
        "verification": {
            "truncation_policy": policy,
            "n_cache_proteins": n_all,
            "n_population_proteins": len(pop_accs),
            "n_rows": int(derived_nrow.sum()),
            "start_arrays_match_exactly": True,
            "n_rows_arrays_match_exactly": True,
            "population_start_is_ascending_in_stored_order": bool(
                np.all(np.diff(stored_start) > 0)),
            "note": ("the two match assertions are the CONTENT of this check: a "
                     "wrong population filter or sort key would break them. "
                     "`population_start_is_ascending_in_stored_order` is False "
                     "on the real store -- that is AO6's defect, recorded as a "
                     "measurement here."),
        },
    }


def cluster_assignment(accs_start_order: list[str], clusters_tsv: Path) -> dict:
    """Protein (in START order) -> dense cluster index, with the match rate.

    Gotcha 10: the rate is measured and the job fails below
    `MIN_CLUSTER_MATCH_FRACTION` rather than joining silently onto a hole.
    """
    cluster_of = load_cluster_map(clusters_tsv)
    reps = [cluster_of.get(a) for a in accs_start_order]
    matched = sum(1 for r in reps if r is not None)
    frac = matched / max(1, len(reps))
    if frac < MIN_CLUSTER_MATCH_FRACTION:
        raise SystemExit(
            f"only {frac:.4%} of the {len(reps):,} population proteins were "
            f"found in {clusters_tsv}. The split partitions ALL of Swiss-Prot "
            "with n_unclustered_records = 0, so this is an accession-convention "
            "or release mismatch, not missing data (gotcha 10).")
    uniq = sorted({r for r in reps if r is not None})
    idx = {r: i for i, r in enumerate(uniq)}
    # An unmatched protein gets its OWN singleton cluster rather than being
    # dropped or lumped: dropping would silently shrink a feature's support and
    # lumping would fuse unrelated proteins into one fake cluster. With the match
    # rate asserted near 100% this affects a handful of proteins at most, and the
    # count is reported.
    n_unmatched = len(reps) - matched
    cl = np.empty(len(reps), dtype=np.int32)
    nxt = len(uniq)
    for i, r in enumerate(reps):
        if r is None:
            cl[i] = nxt
            nxt += 1
        else:
            cl[i] = idx[r]
    return {
        "cluster_index": cl,
        "n_clusters": int(nxt),
        "report": {
            "clusters_tsv": str(clusters_tsv),
            "n_accessions_in_tsv": len(cluster_of),
            "n_population_proteins": len(reps),
            "n_matched": matched,
            "match_fraction": frac,
            "n_unmatched_given_singleton_clusters": n_unmatched,
            "n_distinct_clusters_in_population": len(uniq),
            "n_cluster_slots": int(nxt),
            "min_match_fraction_required": MIN_CLUSTER_MATCH_FRACTION,
        },
    }


def verify_positional_attribution(stored_start: np.ndarray,
                                  stored_nrow: np.ndarray,
                                  order: np.ndarray,
                                  bounds: np.ndarray, n_rows: int) -> dict:
    """Is each protein's BLOCK the protein's own row interval? (brief section 0)

    THE ONLY KIND OF CHECK THAT DISCRIMINATES A MAPPING DEFECT, and the reason is
    worth stating where the code is rather than only in an amendment:

      * a TOTAL cannot see a permutation -- both orderings sum to `n_rows`, which
        is why `footprint`'s original assertion never fired;
      * a RANK cannot either -- Spearman between the pre-AO6 and corrected
        `perplexity` is 0.977 on a mapping where 0.0031% of rows were right,
        because a per-feature aggregate over a permuted-but-same-multiset
        assignment is dominated by the total firing count.

    So the check is POSITIONAL: protein at start-order rank `i` must be given the
    block that begins at its own `start`.

    TOTAL OVER THE OUTCOME SPACE. That identity holds only when the population
    tiles `[0, n_rows)` with no gaps -- true here because the eval cache is
    length-sorted so the truncated proteins sit at the end. When there ARE gaps
    the construction is still correct (row positions index the SORTED row set, so
    the offsets are the cumsum) but `bounds != start`, and the verdict is
    `NOT_APPLICABLE_GAPPED_POPULATION` rather than a failure. Reporting a
    gapped population as a defect would be a check that can only fail.
    """
    s, n = stored_start[order], stored_nrow[order]
    contiguous = bool(s.size and s[0] == 0
                      and (s[:-1] + n[:-1] == s[1:]).all()
                      and s[-1] + n[-1] == n_rows)
    if not contiguous:
        return {
            "verdict": "NOT_APPLICABLE_GAPPED_POPULATION",
            "population_tiles_row_space": False,
            "reason": ("row positions index the SORTED row set, so the block "
                       "offsets are the cumsum of row counts in start order and "
                       "need not equal the starts. The construction is still "
                       "correct; the positional identity simply does not apply."),
        }
    n_right = int((bounds[:-1] == s).sum())
    lo = np.maximum(bounds[:-1], s)
    hi = np.minimum(bounds[:-1] + n, s + n)
    rows_right = int(np.maximum(hi - lo, 0).sum())
    return {
        "verdict": ("POSITIONS_CORRECT" if n_right == s.size and rows_right == n_rows
                    else "POSITIONS_WRONG"),
        "population_tiles_row_space": True,
        "n_proteins": int(s.size),
        "n_blocks_beginning_at_the_proteins_own_start": n_right,
        "n_rows_correctly_attributed": rows_right,
        "n_rows": int(n_rows),
        "fraction_rows_correctly_attributed": rows_right / max(n_rows, 1),
    }


def _entropy_from_moments(total: np.ndarray, sum_clogc: np.ndarray) -> np.ndarray:
    """Shannon entropy in nats from `sum(c)` and `sum(c log c)`.

    `H = log(T) - (1/T) sum c log c`. Zero-total features give NaN rather than 0:
    a feature that never fires has an UNDEFINED support, and 0 would read as
    "measured, concentrated in one cluster" (gotcha 25b). A4a's floor removes
    them from the analysis set anyway, and the count is reported.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.log(total) - sum_clogc / total
    return np.where(total > 0, h, np.nan)


def sweep_seed(row_index_h5: Path, n_features: int,
               bounds_correct: np.ndarray, bounds_buggy: np.ndarray | None,
               cluster_of_prot: np.ndarray, n_clusters: int) -> dict:
    """One pass over a seed's row index.

    Accumulates, in a single sweep over the common refinement of the two protein
    partitions:
      * per (feature, cluster) firing counts, dense -- clusters are scattered
        over the row axis, so nothing can be finalised incrementally;
      * the per-protein footprint moments under the CORRECT (start-order)
        grouping;
      * the same under the pre-AO6 (stored-order) grouping, when asked.
    """
    cuts = (bounds_correct if bounds_buggy is None
            else np.union1d(bounds_correct, bounds_buggy))
    n_prot = bounds_correct.size - 1
    # A ZERO-ROW PROTEIN IS REFUSED UP FRONT rather than left to the flush loop.
    # Its `end` would coincide with the previous protein's, and a leading one's
    # `end` of 0 is never reached at all -- so the protein counter would desync.
    # The `ic != n_prot` check at the bottom would catch it, but only after a
    # whole pass, and "flushed 51942, expected 51943" is a worse error message
    # than this one. Real stores have no empty proteins; saying so explicitly is
    # what makes that an assumption the code states rather than relies on.
    if np.any(np.diff(bounds_correct) == 0):
        raise SystemExit(
            f"{int((np.diff(bounds_correct) == 0).sum())} population proteins "
            "have ZERO rows. The row axis cannot be blocked by protein when a "
            "protein occupies no positions; refusing rather than desyncing.")

    cl_counts = np.zeros((n_features, n_clusters), dtype=np.int64)
    tot_c = np.zeros(n_features, np.float64)
    clogc_c = np.zeros(n_features, np.float64)
    sumc2_c = np.zeros(n_features, np.float64)
    nprot_c = np.zeros(n_features, np.int64)
    tot_b = np.zeros(n_features, np.float64)
    clogc_b = np.zeros(n_features, np.float64)
    sumc2_b = np.zeros(n_features, np.float64)
    nprot_b = np.zeros(n_features, np.int64)

    def flush(acc: np.ndarray, tot, clogc, sumc2, nprot):
        nz = acc > 0
        if not nz.any():
            return
        c = acc[nz].astype(np.float64)
        tot[nz] += c
        clogc[nz] += c * np.log(c)
        sumc2[nz] += c * c
        nprot[nz] += 1

    cur_c = np.zeros(n_features, np.int64)
    cur_b = np.zeros(n_features, np.int64)
    ic = ib = 0

    with h5py.File(row_index_h5, "r") as hf:
        offs = hf["row_offsets"][:]
        feats = hf["feat_by_row"]
        if offs.size - 1 != int(bounds_correct[-1]):
            raise SystemExit(
                f"{row_index_h5.name} has {offs.size - 1:,} row positions but "
                f"the population has {int(bounds_correct[-1]):,}. Refusing a "
                "differently-scoped index (gotcha 15).")
        i = 0
        while i < cuts.size - 1:
            j = i + 1
            while (j < cuts.size - 1
                   and cuts[j] - cuts[i] < BLOCK_POSITIONS):
                j += 1
            lo, hi = int(cuts[i]), int(cuts[j])
            base = int(offs[lo])
            flat = feats[base:int(offs[hi])]
            for s in range(i, j):
                a, b = int(offs[int(cuts[s])]) - base, int(offs[int(cuts[s + 1])]) - base
                if b > a:
                    seg = np.bincount(flat[a:b], minlength=n_features)
                    cur_c += seg
                    if bounds_buggy is not None:
                        cur_b += seg
                end = int(cuts[s + 1])
                # WHILE, not IF: a zero-row protein would make two consecutive
                # bounds equal, and one flush per cut would desync the protein
                # counter silently. Real stores have no empty proteins; a loop
                # costs nothing and removes the assumption.
                while (ic + 1 < bounds_correct.size
                       and end == int(bounds_correct[ic + 1])):
                    flush(cur_c, tot_c, clogc_c, sumc2_c, nprot_c)
                    nzc = cur_c > 0
                    if nzc.any():
                        cl_counts[nzc, cluster_of_prot[ic]] += cur_c[nzc]
                    cur_c[:] = 0
                    ic += 1
                while (bounds_buggy is not None and ib + 1 < bounds_buggy.size
                       and end == int(bounds_buggy[ib + 1])):
                    flush(cur_b, tot_b, clogc_b, sumc2_b, nprot_b)
                    cur_b[:] = 0
                    ib += 1
            i = j

    if ic != n_prot:
        raise SystemExit(f"flushed {ic} correct proteins, expected {n_prot}")

    # ---- cluster-level reductions, CHUNKED over features -----------------
    # A whole-matrix `c * log(c)` would materialise two float64 copies of a
    # (features x clusters) array -- ~1.9 GB transient at the real sizes, on top
    # of the accumulator. Chunking bounds it without changing the arithmetic.
    cl_total = np.zeros(n_features, np.float64)
    cl_clogc = np.zeros(n_features, np.float64)
    n_cl_fired = np.zeros(n_features, np.int64)
    CH = 512
    for f0 in range(0, n_features, CH):
        blk = cl_counts[f0:f0 + CH]
        c = blk.astype(np.float64)
        nz = c > 0
        cl_total[f0:f0 + CH] = c.sum(axis=1)
        cl_clogc[f0:f0 + CH] = np.where(nz, c * np.log(np.where(nz, c, 1.0)),
                                        0.0).sum(axis=1)
        n_cl_fired[f0:f0 + CH] = nz.sum(axis=1)
    out = {
        "cluster_total_fires": cl_total,
        "cluster_entropy_nats": _entropy_from_moments(cl_total, cl_clogc),
        "n_clusters_fired": n_cl_fired,
        "protein_total_fires_correct": tot_c,
        "protein_entropy_correct": _entropy_from_moments(tot_c, clogc_c),
        "n_proteins_correct": nprot_c,
        "protein_concentration_correct": np.where(
            tot_c > 0, sumc2_c / np.maximum(tot_c, 1) ** 2, np.nan),
    }
    if bounds_buggy is not None:
        out.update({
            "protein_entropy_preAO6": _entropy_from_moments(tot_b, clogc_b),
            "n_proteins_preAO6": nprot_b.astype(np.float64),
            "protein_concentration_preAO6": np.where(
                tot_b > 0, sumc2_b / np.maximum(tot_b, 1) ** 2, np.nan),
            "protein_total_fires_preAO6": tot_b,
        })
    return out


def _rel_change(new: np.ndarray, old: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(new - old) / np.abs(old)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--eval-cache", type=Path, required=True)
    ap.add_argument("--clusters", type=Path, required=True)
    ap.add_argument("--row-index-cache", type=Path, required=True)
    ap.add_argument("--ckpts", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(range(20)),
                    help="the ANCHOR seeds -- the ones r_f is measured for")
    ap.add_argument("--scored-row-set", default="label_store")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--no-ao6-comparison", action="store_true",
                    help="skip the pre-AO6 grouping arm. It costs nothing extra "
                         "in I/O, so this exists only for tests.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    from src.sae.train import git_dirty, git_sha

    cfg = yaml.safe_load(args.config.read_text())
    floor = int(cfg["confound_control"]["analysis_min_firing_rows"])

    pop = population_accessions(args.eval_cache, args.labels)
    print(f"[dfs] population verified: {pop['verification']} ", flush=True)
    clus = cluster_assignment(pop["accessions_start_order"], args.clusters)
    print(f"[dfs] cluster join: {clus['report']['n_matched']:,}/"
          f"{clus['report']['n_population_proteins']:,} = "
          f"{clus['report']['match_fraction']:.4%}, "
          f"{clus['report']['n_distinct_clusters_in_population']:,} clusters",
          flush=True)

    prot_of_row, n_prot = protein_of_row(args.labels)
    n_pos = int(prot_of_row.size)
    nrow_start_order = pop["stored_nrow"][pop["order"]]
    bounds_correct = np.concatenate([[0], np.cumsum(nrow_start_order)])
    bounds_buggy = (None if args.no_ao6_comparison else
                    np.concatenate([[0], np.cumsum(pop["stored_nrow"])]))
    assert int(bounds_correct[-1]) == n_pos

    with h5py.File(args.labels, "r") as hf:
        store_n_rows = int(hf.attrs["n_rows"])

    # THE POSITIONAL CHECK, run before any firing is attributed. A total and a
    # rank both survive the AO6 defect; only this discriminates.
    positional = verify_positional_attribution(
        pop["stored_start"], pop["stored_nrow"], pop["order"], bounds_correct,
        store_n_rows)
    print(f"[dfs] positional attribution: {positional}", flush=True)
    if positional["verdict"] == "POSITIONS_WRONG":
        raise SystemExit(
            "the protein blocks do NOT begin at their own row starts: only "
            f"{positional['n_rows_correctly_attributed']:,} of "
            f"{positional['n_rows']:,} rows are correctly attributed. Refusing "
            "to join clusters onto a wrong protein mapping (AO6).")

    per_seed: dict = {}
    stack_A, stack_B, stack_nc, stack_tot, stack_keep = [], [], [], [], []
    ao6_rows = []
    for pos, sd in enumerate(args.seeds):
        ri = resolve_row_index(args.row_index_cache, args.ckpts / f"seed_{sd}",
                               args.eval_cache, args.scored_row_set, args.k,
                               store_n_rows)
        with h5py.File(ri, "r") as hf:
            fire_counts = hf["fire_counts"][:].astype(np.float64)
        n_features = int(fire_counts.size)
        res = sweep_seed(ri, n_features, bounds_correct, bounds_buggy,
                         clus["cluster_index"], clus["n_clusters"])

        # INTERNAL CHECK, not an assertion: the total fires this sweep counted
        # must equal the row index's OWN fire_counts. They are computed by
        # different code on different passes, so agreement is real evidence the
        # grouping walked every position exactly once.
        tot = res["cluster_total_fires"]
        mismatch = int((tot != fire_counts).sum())
        if mismatch:
            raise SystemExit(
                f"seed {sd}: {mismatch} features' swept total fires disagree "
                f"with the row index's own fire_counts (max abs diff "
                f"{np.abs(tot - fire_counts).max():.0f}). Every row position "
                "must be counted exactly once.")

        A = res["cluster_entropy_nats"]                     # = log(effective)
        B = np.log(np.maximum(res["n_clusters_fired"], 1).astype(np.float64))
        keep = fire_counts >= floor
        ok = keep & np.isfinite(A) & np.isfinite(B)
        rho = float(spearmanr(A[ok], B[ok]).statistic)

        row = {
            "seed": int(sd), "row_index": str(ri),
            "n_features": n_features,
            "n_retained_at_floor": int(keep.sum()),
            "n_excluded_at_floor": int((~keep).sum()),
            "floor": floor,
            "n_zero_firing_features": int((fire_counts == 0).sum()),
            "n_zero_firing_among_retained": int((fire_counts[keep] == 0).sum()),
            "spearman_A_vs_B": rho,
            "A_entropy_nats": {
                "min": float(np.nanmin(A[ok])), "median": float(np.nanmedian(A[ok])),
                "max": float(np.nanmax(A[ok]))},
            "B_log_distinct_clusters": {
                "min": float(np.nanmin(B[ok])), "median": float(np.nanmedian(B[ok])),
                "max": float(np.nanmax(B[ok]))},
            "effective_clusters_exp_A": {
                "median": float(np.exp(np.nanmedian(A[ok])))},
            "n_clusters_fired": {
                "min": int(res["n_clusters_fired"][ok].min()),
                "median": float(np.median(res["n_clusters_fired"][ok])),
                "max": int(res["n_clusters_fired"][ok].max())},
        }
        if bounds_buggy is not None:
            perp_new = np.exp(res["protein_entropy_correct"])
            perp_old = np.exp(res["protein_entropy_preAO6"])
            m = ok & np.isfinite(perp_new) & np.isfinite(perp_old) & (perp_old > 0)
            rc_p = _rel_change(perp_new, perp_old)[m]
            rc_c = _rel_change(res["protein_concentration_correct"],
                               res["protein_concentration_preAO6"])[m]
            ao6 = {
                "seed": int(sd),
                "n_features_compared": int(m.sum()),
                "perplexity_median_abs_rel_change": float(np.nanmedian(rc_p)),
                "perplexity_p90_abs_rel_change": float(np.nanpercentile(rc_p, 90)),
                "concentration_median_abs_rel_change": float(np.nanmedian(rc_c)),
                "spearman_perplexity_new_vs_old": float(
                    spearmanr(perp_new[m], perp_old[m]).statistic),
                "spearman_concentration_new_vs_old": float(
                    spearmanr(res["protein_concentration_correct"][m],
                              res["protein_concentration_preAO6"][m]).statistic),
                "median_n_proteins_correct": float(
                    np.median(res["n_proteins_correct"][m])),
                "median_n_proteins_preAO6": float(
                    np.median(res["n_proteins_preAO6"][m])),
            }
            ao6_rows.append(ao6)
            row["AO6_grouping_comparison"] = ao6
        per_seed[str(sd)] = row
        print(f"[dfs] seed {sd}: retained {row['n_retained_at_floor']:,}, "
              f"Spearman(A,B) {rho:.4f}, median effective clusters "
              f"{row['effective_clusters_exp_A']['median']:.1f}, median "
              f"distinct {row['n_clusters_fired']['median']:.0f}"
              + (f", perplexity median |rel change| "
                 f"{ao6_rows[-1]['perplexity_median_abs_rel_change']:.4f}"
                 if bounds_buggy is not None else ""), flush=True)

        stack_A.append(A); stack_B.append(B)
        stack_nc.append(res["n_clusters_fired"]); stack_tot.append(tot)
        stack_keep.append(keep)

    A = np.stack(stack_A); B = np.stack(stack_B)
    rhos = np.array([per_seed[str(s)]["spearman_A_vs_B"] for s in args.seeds])
    med_rho = float(np.median(rhos))
    adopted = "B_log_distinct_clusters" if med_rho > P8_SPEARMAN_THRESHOLD else "A_cluster_entropy"
    p8 = {
        "prediction": f"Spearman(A, B) exceeds {P8_SPEARMAN_THRESHOLD}",
        "category": "within-system comparison (AO4 P8)",
        "median_spearman": med_rho,
        "spearman_by_seed": rhos.tolist(),
        "range": [float(rhos.min()), float(rhos.max())],
        "threshold": P8_SPEARMAN_THRESHOLD,
        "outcome": "HIT" if med_rho > P8_SPEARMAN_THRESHOLD else "MISS",
        # DERIVED from the number just computed (gotcha 20).
        "adopted_definition": adopted,
        "adoption_rule": (
            f"> {P8_SPEARMAN_THRESHOLD} -> take B for simplicity and record the "
            f"choice as immaterial BY MEASUREMENT; <= -> A is the "
            f"recommendation and the definition is the author's (AO9)."),
    }
    p9 = {
        "prediction": "the cluster join match rate exceeds 99%",
        "category": "within-system comparison (AO4 P9)",
        "match_fraction": clus["report"]["match_fraction"],
        "outcome": "HIT" if clus["report"]["match_fraction"] > 0.99 else "MISS",
    }
    if ao6_rows:
        med = float(np.median([r["perplexity_median_abs_rel_change"]
                               for r in ao6_rows]))
        p10 = {
            "prediction": "the median absolute relative change in perplexity "
                          f"exceeds {P10_MIN_MEDIAN_REL_CHANGE}",
            "category": "within-system comparison (AO4 P10)",
            "median_over_seeds_of_median_abs_rel_change": med,
            "threshold": P10_MIN_MEDIAN_REL_CHANGE,
            "spearman_new_vs_old_median_over_seeds": float(np.median(
                [r["spearman_perplexity_new_vs_old"] for r in ao6_rows])),
            "outcome": "HIT" if med > P10_MIN_MEDIAN_REL_CHANGE else "MISS",
        }
    else:
        p10 = {"outcome": "NOT_SCORED -- the AO6 comparison arm was skipped"}

    print(f"[dfs] P8 median Spearman(A,B) {med_rho:.4f} -> adopt {adopted} "
          f"({p8['outcome']})", flush=True)
    print(f"[dfs] P9 match {p9['match_fraction']:.4%} ({p9['outcome']}); "
          f"P10 {p10['outcome']}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as hf:
        hf.create_dataset("d_fs_cluster_entropy", data=A, compression="gzip")
        hf.create_dataset("d_fs_log_distinct_clusters", data=B,
                          compression="gzip")
        hf.create_dataset("n_clusters_fired", data=np.stack(stack_nc),
                          compression="gzip")
        hf.create_dataset("total_fires", data=np.stack(stack_tot),
                          compression="gzip")
        hf.create_dataset("retained_at_floor", data=np.stack(stack_keep),
                          compression="gzip")
        hf.create_dataset("seeds", data=np.array(args.seeds, np.int32))
        hf.attrs["measurement"] = "repr_density_d_fs_AO4"
        hf.attrs["amendment"] = "F1.6 / AL8b / AM6.5 / AO section 4"
        hf.attrs["adopted_definition"] = adopted
        hf.attrs["definitions"] = json.dumps({
            "A_cluster_entropy": "Shannon entropy (nats) of firing mass over "
                                 "mmseqs clusters = log(effective clusters)",
            "B_log_distinct_clusters": "log(number of clusters with any firing)"})
        hf.attrs["analysis_min_firing_rows"] = floor
        hf.attrs["clusters_tsv"] = str(args.clusters)
        hf.attrs["cluster_match_fraction"] = clus["report"]["match_fraction"]
        hf.attrs["n_clusters_in_population"] = clus["report"][
            "n_distinct_clusters_in_population"]
        hf.attrs["labels"] = str(args.labels)
        hf.attrs["truncation_policy"] = pop["verification"]["truncation_policy"]
        hf.attrs["scored_row_set"] = args.scored_row_set
        hf.attrs["n_rows"] = store_n_rows
        hf.attrs["k"] = args.k
        hf.attrs["git_sha"] = git_sha()
        hf.attrs["git_dirty"] = git_dirty()

    report = {
        "identity": {
            "measurement": "repr_density_d_fs_AO4",
            "amendment": "F1.6 / AL8b / AM6.5 / AO section 4",
            "firewall_crossing": "NONE -- reads firing patterns and the cluster "
                                 "assignment; no r_f, no AP, no M",
            "labels": str(args.labels),
            "eval_cache": str(args.eval_cache),
            "clusters": str(args.clusters),
            "scored_row_set": args.scored_row_set,
            "k": args.k,
            "seeds": list(args.seeds),
            "analysis_min_firing_rows": floor,
            "d_fs_h5": str(args.out),
            "git_sha": git_sha(), "git_dirty": git_dirty(),
            "python_version": platform.python_version(),
        },
        "population_verification": pop["verification"],
        "positional_attribution_check": positional,
        "cluster_join": clus["report"],
        "per_seed": per_seed,
        "AO6_grouping_comparison_across_seeds": ao6_rows,
        "prediction_scorecard": {"P8": p8, "P9": p9, "P10": p10},
        "decision": (
            "P8's rule fires on the measured Spearman and names the adopted "
            "definition. If it lands at or below the threshold, A is the "
            "recommendation and the choice is the author's (AO9)."),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    print(f"[dfs] wrote {args.out} and {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
