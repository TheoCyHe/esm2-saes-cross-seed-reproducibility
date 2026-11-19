"""
A3 pilot (PREREGISTRATION Amendment F3/F4) -- null g_f draws via a JOINT
circular shift: one shift per protein per replicate, ALL concepts' labels
for that protein move together (not an independent shift per concept -- see
Amendment F3 for why an independent-per-concept shift is anti-conservative).

Per replicate: draw one random offset per protein, shift every concept's
positive rows within that protein's row range by that offset (circular,
wraps within the protein -- never crosses a protein boundary), compute AP
for every concept against every feature using the persisted row index
(row_index_cache.py), and take the MAX over concepts per feature. That max
IS one null g_f draw. Repeated across replicates and seeds.

Uses the SAME row-index cache the production AP path uses (E8) -- the
activation gather and row-index build are shift-invariant and paid once per
seed regardless of how many replicates run against them.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from src.analysis.artifact_identity import (
    AP_ARITH_DEVICE_KEY,
    AP_IMPL_SEGMENTED,
    AP_IMPL_SPARSE,
    GATHER_DEVICE_KEY,
    PARAMETER_VALUES,
    argmax_pool_mask,
    device_class,
    device_identity_attrs,
    write_identity,
)
from src.analysis.feature_ap import (
    average_precision_segmented,
    filter_nonzeros,
    library_scoped_concept_counts,
    load_labels,
    load_scored_row_set,
    sparse_activations,
)
from src.analysis.row_index_cache import DEVICE_ABSENT, compute_or_load_row_index
from src.analysis.sparse_ap_index import (average_precision_from_touches,
                                          average_precision_from_touches_chunked,
                                          gather_touches, gather_touches_chunked)
from src.sae.train import TopKSAE, git_dirty, git_sha


def per_seed_bin_counts(fire_rate_all: np.ndarray, n_shifts: int) -> list[dict]:
    """PER-SEED firing-rate bin counts over `corrector_scan.BIN_EDGES`: for each
    bin, the min and max over seeds of the number of features in it, and the
    pooled null draws that min guarantees. `fire_rate_all` is (seeds, features).

    One function for the pilot and the shard merge (`merge_null_shards`), so
    the attr a merged null carries describes ALL its seeds -- before amendment
    CT's CT10-17 preamble the merge carried shard 0's (CT5's merged nulls say
    42 features in bin 0, min = max; the 20 seeds hold 33-48).
    """
    from src.analysis.corrector_scan import BIN_EDGES
    fire_rate_all = np.asarray(fire_rate_all)
    n_seeds_ = fire_rate_all.shape[0]
    out = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        per_seed = [int(((fire_rate_all[s] >= lo) & (fire_rate_all[s] < hi)).sum())
                    for s in range(n_seeds_)]
        lo_n, hi_n = min(per_seed), max(per_seed)
        out.append({
            "lo": lo, "hi": min(hi, 1.0),
            "n_features_min_over_seeds": lo_n,
            "n_features_max_over_seeds": hi_n,
            "n_null_draws_min": lo_n * n_shifts * n_seeds_,
            "binning": "per_seed",
        })
    return out


def build_row_to_protein(index: list[dict], n_rows: int) -> np.ndarray:
    """row -> protein index, for O(1) per-row shift lookup. Proteins are
    assumed contiguous and non-overlapping in the cache (start:start+n_rows),
    asserted below rather than trusted."""
    row2protein = np.full(n_rows, -1, dtype=np.int32)
    for i, e in enumerate(index):
        row2protein[e["start"]:e["start"] + e["n_rows"]] = i
    assert (row2protein >= 0).all(), "cache rows not fully covered by proteins"
    return row2protein


def protein_starts_lengths(index: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    starts = np.array([e["start"] for e in index], dtype=np.int64)
    lengths = np.array([e["n_rows"] for e in index], dtype=np.int64)
    return starts, lengths


def build_population_index(
    cache_index: list[dict], subset: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    """G2: build a SUBSET-RELATIVE protein index (starts/n_rows in positions
    0..len(subset)-1) plus a `remap` array (original cache row -> subset
    position, -1 if not in the subset), matching how sparse_activations
    reindexes its output when given a row subset.

    `subset` (from load_scored_row_set) is the ASCENDING-sorted concatenation of
    every population protein's [start, start+n_rows) block. Since these
    blocks are disjoint and internally contiguous in the ORIGINAL cache,
    sorting the concatenation preserves each protein's own row order AND
    orders the blocks by original start -- so the population's proteins,
    read off `subset` in original-start order, are exactly what the subset-
    relative layout needs.
    """
    remap = -np.ones(int(subset.max()) + 2, dtype=np.int64)
    remap[subset] = np.arange(len(subset))

    # Population proteins, sorted by ORIGINAL start (matches subset's order).
    pop = sorted((e for e in cache_index if e["start"] < len(remap)
                 if remap[e["start"]] >= 0),
                key=lambda e: e["start"])
    pop_index = []
    cursor = 0
    for e in pop:
        pop_index.append({"accession": e["accession"], "start": cursor,
                          "n_rows": e["n_rows"]})
        cursor += e["n_rows"]
    assert cursor == len(subset), (
        f"population protein lengths ({cursor:,}) don't sum to the subset "
        f"size ({len(subset):,}) -- a protein is only partially in the "
        "subset, which amendment G2's population filter should never produce "
        "(whole proteins are included or excluded, never split)")
    return pop_index, remap


def shift_rows(rows: np.ndarray, row2protein: np.ndarray, starts: np.ndarray,
               lengths: np.ndarray, shift_per_protein: np.ndarray) -> np.ndarray:
    """Circular-shift each row WITHIN its own protein's range. `rows` may
    belong to many different proteins; each uses ITS protein's shift amount
    and wraps only within its own [start, start+length) -- never crosses
    into a neighbouring protein."""
    pidx = row2protein[rows]
    p_start = starts[pidx]
    p_len = lengths[pidx]
    s = shift_per_protein[pidx]
    return p_start + (rows - p_start + s) % p_len


# ------------------------------------------------------------------------ #
# Null B (amendment G4): across-protein, length-stratified permutation.
#
# Null A (joint circular shift) only tests LOCALIZATION -- it keeps a
# protein's own annotation architecture on that SAME protein and asks only
# "does this feature prefer these positions within proteins that already
# carry the concept". It cannot detect a feature that has simply learned
# "this is one of the proteins with this concept" (protein selection), since
# every null draw still fires on the same protein population as the real
# data.
#
# Null B tests TOTAL GROUNDING (protein selection + localization): each
# protein's WHOLE annotation architecture -- every concept's positive rows,
# jointly -- is moved onto a DIFFERENT protein of similar length, drawn
# uniformly at random from proteins in the same length bin. A feature that
# merely detects "protein has high overall hydrophobicity" (or any bulk
# property correlated with concept-bearing proteins) will still look aligned
# under Null A but should regress toward chance under Null B, because the
# concept no longer co-occurs with whatever protein-level property the
# feature actually tracks.
# ------------------------------------------------------------------------ #

def build_length_bins(lengths: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-bin proteins by length so permutation partners are always of
    SIMILAR length -- a concept confined to short proteins must not be
    permuted onto a population of long ones, or Null B would conflate
    "wrong protein" with "wrong length regime" (a confound of its own)."""
    n_bins = min(n_bins, len(np.unique(lengths)))
    edges = np.quantile(lengths, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)                     # duplicate quantiles collapse bins
    bins = np.searchsorted(edges, lengths, side="right") - 1
    bins = np.clip(bins, 0, len(edges) - 2)       # the max value falls on the last edge
    return bins.astype(np.int64)


def permute_within_bins(bins: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One random permutation PER BIN, independently. Returns `dest[i]` = the
    protein index whose annotation architecture protein i now RECEIVES.
    Proteins in a singleton bin (no same-length partner) map to themselves --
    identity, not a crash and not a cross-bin leak."""
    n = len(bins)
    dest = np.arange(n)
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        if len(idx) < 2:
            continue
        dest[idx] = rng.permutation(idx)
    return dest


def cross_protein_shift_rows(rows: np.ndarray, row2protein: np.ndarray,
                             starts: np.ndarray, lengths: np.ndarray,
                             dest_protein: np.ndarray) -> np.ndarray:
    """Move each row from its OWN protein onto `dest_protein[own]`'s row
    range, rescaling its relative position proportionally (source and
    destination proteins share a length BIN, not an exact length) and
    clipping into the destination's range. `dest_protein` must be the SAME
    array for every concept in one replicate -- that shared assignment is
    what makes the move a JOINT relocation of a protein's whole annotation
    architecture, mirroring shift_rows' joint circular shift within Null A."""
    pidx = row2protein[rows]
    p_start, p_len = starts[pidx], lengths[pidx]
    offset = (rows - p_start).astype(np.float64)
    qidx = dest_protein[pidx]
    q_start, q_len = starts[qidx], lengths[qidx]
    new_offset = np.floor(offset * q_len / p_len).astype(np.int64)
    new_offset = np.clip(new_offset, 0, q_len - 1)
    return q_start + new_offset


# ------------------------------------------------------------------------ #
# THE MASKED STRATUM (PREREGISTRATION amendments BG and BH).
#
# BG: under `--negatives unannotated` a concept's rows carrying that concept's
# label in some OTHER evidence tier are dropped from the evaluated universe.
# The observed statistic is therefore AP on the universe "all rows MINUS this
# concept's masked rows". THE MASKED SET IS HELD FIXED ACROSS DRAWS and only
# the POSITIVES are displaced -- computing each draw on a different universe
# would make the null and the observed statistic not like-for-like.
#
# BG2 (the collision rule): a displaced positive landing on a masked row of
# the SAME concept is DROPPED from that draw. The row stays masked. The mask
# wins; the positive does not displace it and does not get re-drawn.
#
# BH (the degenerate-draw rule): a (concept, arm, shift) cell in which EVERY
# observed positive has been displaced onto a masked row has zero surviving
# positives, and AP is undefined there. The draw is DROPPED, the count is
# recorded per concept and arm, and NO VALUE IS IMPUTED. A cell with exactly
# one surviving positive is NOT degenerate -- AP is defined (the reciprocal of
# that positive's rank) and no rule applies.
# ------------------------------------------------------------------------ #

def apply_collision_rule(displaced: np.ndarray,
                         keep_row: np.ndarray | None) -> np.ndarray:
    """BG2. Drop displaced positives that landed on a masked row.

    `keep_row` is a boolean array over the analysed row axis, True where the
    row is part of the evaluated universe. None means no masking is active,
    in which case every displaced positive survives -- the unmasked path must
    be reachable and must be the identity, or the pooled arm changes.
    """
    if keep_row is None:
        return displaced
    return displaced[keep_row[displaced]]


def exceedance_resolution(n_surviving_draws: int, n_features: int) -> float:
    """BH4. The finest exceedance probability a concept's reference
    distribution can express: 1 / (surviving draws x features + 1).

    NOT a uniform constant once BH2 drops a draw, which is exactly why it is
    stored per (concept, arm) rather than left for a consumer to assume.
    """
    if n_surviving_draws <= 0:
        return float("nan")            # NOT_EVALUABLE: no reference distribution
    return 1.0 / (n_surviving_draws * n_features + 1)


# BH3's outcome grid. Written over the FULL domain of (degenerate count per
# arm), not over the cases expected to occur -- gotcha 24: a rule must be a
# total function of the statistic it reads, and two of this project's
# pre-declared rules have already been well-formed and INAPPLICABLE because
# they were defined on a subset of the outcome space.
BH3_NONE = "NONE_DEGENERATE"
BH3_SOME = "SOME_BUT_NOT_ALL_DEGENERATE"
BH3_ONE_ARM = "ALL_DEGENERATE_ONE_ARM_NOT_EVALUABLE_ARM"
BH3_BOTH_ARMS = "ALL_DEGENERATE_BOTH_ARMS_NOT_EVALUABLE_STRATUM"
BH3_NO_ARM = "NO_ARM_REQUESTED"


def classify_bh3(n_degenerate_a: int | None, n_degenerate_b: int | None,
                 n_shifts: int) -> str:
    """BH3. Which cell of the outcome grid a concept occupies.

    `None` for an arm that was not requested -- distinct from 0, which asserts
    the arm ran and produced no degenerate draw. An absent arm is reported
    absent, never treated as matching (I3).
    """
    arms = [n for n in (n_degenerate_a, n_degenerate_b) if n is not None]
    if not arms:
        return BH3_NO_ARM
    all_degen = [n >= n_shifts for n in arms]
    if all(all_degen):
        return BH3_BOTH_ARMS if len(arms) > 1 else BH3_ONE_ARM
    if any(all_degen):
        return BH3_ONE_ARM
    if any(n > 0 for n in arms):
        return BH3_SOME
    return BH3_NONE


def build_masks(concepts: list[str], concept_rows: dict[str, np.ndarray],
                pooled_rows: dict[str, np.ndarray], negatives: str,
                n_rows: int):
    """Per-concept keep-mask, effective row count and masked-row count.

    Identical construction to feature_ap.py's, and WITHIN-CONCEPT only: a
    BINDING residue is genuine evidence that it is not an ACT_SITE, so
    cross-concept rows stay negative and only the SAME concept's other-tier
    rows leave the universe.

    Extracted so the BH validator and `main()` share ONE implementation. Two
    copies of a mask construction would make the validator's agreement with
    the run a statement about the copies rather than about the rule -- the
    cross-component-agreement failure this project keeps shipping (gotcha 16).
    """
    keep_row: dict[str, np.ndarray | None] = {}
    n_eff: dict[str, int] = {}
    n_masked: dict[str, int] = {}
    for c in concepts:
        other = np.setdiff1d(pooled_rows.get(c, np.zeros(0, dtype=np.int64)),
                             concept_rows[c], assume_unique=False)
        if negatives == "unannotated" and other.size:
            kr = np.ones(n_rows, dtype=bool)
            kr[other] = False
            keep_row[c], n_eff[c], n_masked[c] = kr, n_rows - int(other.size), int(other.size)
        else:
            keep_row[c], n_eff[c], n_masked[c] = None, n_rows, 0
    return keep_row, n_eff, n_masked


def draw_schedules(null_seed: int, n_shifts: int, lengths: np.ndarray,
                   n_length_bins: int, want_a: bool, want_b: bool):
    """The stored displacement schedules -- ONE set, shared across every seed.

    Null A and Null B draw from INDEPENDENT SeedSequence-spawned streams off
    one seed, so `--null-type both` reproduces EXACTLY the draws either null
    would get run alone; neither consumes the other's randomness.

    Extracted for the same reason as `build_masks`: BH's expectation is exact
    BECAUSE the schedules are stored rather than drawn fresh, and a validator
    that redrew them from a second copy of this code would be checking its own
    copy.
    """
    ss_a, ss_b = np.random.SeedSequence(null_seed).spawn(2)
    shift_schedule = bins = dest_schedule = None
    if want_a:
        rng_a = np.random.default_rng(ss_a)
        shift_schedule = rng_a.integers(
            0, np.maximum(lengths, 1)[None, :].repeat(n_shifts, 0))
        assert shift_schedule.shape == (n_shifts, len(lengths))
    if want_b:
        rng_b = np.random.default_rng(ss_b)
        bins = build_length_bins(lengths, n_length_bins)
        dest_schedule = np.stack(
            [permute_within_bins(bins, rng_b) for _ in range(n_shifts)])
        assert dest_schedule.shape == (n_shifts, len(lengths))
    return shift_schedule, dest_schedule, bins


def masked_concept_arrays(feat: torch.Tensor, pos: torch.Tensor,
                          seg: torch.Tensor, keep_row: np.ndarray | None,
                          n_features: int):
    """The non-zeros one concept's draws are scored against.

    Extracted from `main()` deliberately. The mask application is the single
    step this whole build turns on, and a computation left inside `main()`
    cannot be handed a fixture with a known answer (gotcha 13 extension 5) --
    so it could not be mutation-tested, and the defect it guards against is
    not hypothetical. `v1v2v3_pilot_aw.py` built a keep-mask and then computed
    AP through the sparse touch path, which ranks against the FULL row set:
    the mask was constructed, never applied, and every AP it produced was the
    unmasked one with a different denominator.

    HOISTED OUT OF THE SHIFT LOOP by the caller: BG1 holds the masked set
    fixed across draws, so this is a property of the CONCEPT alone.
    """
    if keep_row is None:
        return feat, pos, seg
    keep_t = torch.from_numpy(keep_row).to(feat.device)
    return filter_nonzeros(feat, pos, keep_t, n_features)


def masked_draw_gain(f_c: torch.Tensor, s_c: torch.Tensor,
                     labels: torch.Tensor, n_features: int,
                     n_surviving: int, n_eff: int) -> torch.Tensor:
    """G1's normalised gain for ONE draw, on the masked universe.

    `n_surviving` is the count AFTER BG2, and `n_eff` the row count after the
    mask -- both must be the draw's OWN, or the gain normalisation is
    internally inconsistent with what was actually queried.
    """
    ap = average_precision_segmented(f_c, s_c, labels, n_features,
                                     n_surviving, n_eff)
    p_draw = n_surviving / n_eff
    return (ap - p_draw) / (1.0 - p_draw)


def build_displacement_plan(
    concepts: list[str],
    concept_rows: dict[str, np.ndarray],
    keep_row: dict[str, np.ndarray | None],
    row2protein: np.ndarray, starts: np.ndarray, lengths: np.ndarray,
    shift_schedule: np.ndarray | None, dest_schedule: np.ndarray | None,
    n_shifts: int,
) -> dict:
    """Displace every concept's positives under every stored shift, apply BG2,
    and record what survives.

    COMPUTED ONCE, BEFORE THE SEED LOOP, and this is a correctness property
    rather than an optimisation: displacement is pure combinatorics on ROW
    indices and does not involve an SAE at all. Computing it inside the seed
    loop would make the surviving counts LOOK seed-dependent -- 20 chances for
    them to differ, and no structural reason they could not. Here they cannot,
    because there is one array.

    It is also what makes BH's expectation checkable without a GPU: the
    schedules are stored, not drawn fresh, so the surviving counts have one
    right answer.
    """
    plan: dict = {"a": None, "b": None}
    for arm, sched in (("a", shift_schedule), ("b", dest_schedule)):
        if sched is None:
            continue
        rows_per_shift: dict[str, list[np.ndarray]] = {}
        counts = np.zeros((len(concepts), n_shifts), dtype=np.int64)
        for ci, c in enumerate(concepts):
            src = concept_rows[c]
            kr = keep_row.get(c)
            per_shift = []
            for r in range(n_shifts):
                if arm == "a":
                    d = shift_rows(src, row2protein, starts, lengths, sched[r])
                else:
                    # Rescaling onto a shorter destination protein is
                    # many-to-one, so dedup FIRST -- the realised positive
                    # count for the draw is what gain-normalisation must use.
                    d = np.unique(cross_protein_shift_rows(
                        src, row2protein, starts, lengths, sched[r]))
                d = apply_collision_rule(d, kr)
                per_shift.append(d)
                counts[ci, r] = d.size
            rows_per_shift[c] = per_shift
        plan[arm] = {"rows": rows_per_shift, "counts": counts,
                     "degenerate": counts == 0}
    return plan


def population_n_rows(subset: np.ndarray, store_n_rows: int,
                      scored_row_set: str) -> int:
    """The analysed population's row count -- `len(subset)`, as feature_ap.py
    uses. Before amendment CT (§CT11.7-R) this module used the label store's
    `n_rows` attr on every subset path, which equals `len(subset)` only for
    `label_store` (the store's own population); on `shared_experimental` the
    row->protein index was sized to the store and its coverage assertion
    fired (jobs 30207159/30207160, no value computed). On `label_store` the two
    must agree exactly, and a disagreement refuses rather than picks one."""
    n = int(len(subset))
    if scored_row_set == "label_store" and n != int(store_n_rows):
        raise SystemExit(f"FATAL: label_store subset has {n:,} rows but the "
                         f"label store's n_rows attr is {int(store_n_rows):,}")
    return n


def restrict_to_concepts(available: list[str], wanted: list[str]) -> list[str]:
    """The concepts of `wanted`, in `available`'s (sorted) order -- refusing if
    any wanted concept is not available, because a null silently missing a
    concept the AP carries is a null over a different pool (I4's shape)."""
    have = set(available)
    missing = [c for c in wanted if c not in have]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} concepts of the requested axis "
                         f"have no positives in this stratum, e.g. {missing[:3]}")
    want = set(wanted)
    return [c for c in available if c in want]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--ckpts", type=Path, nargs="+", required=True)
    p.add_argument("--eval-cache", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--scored-row-set", default="full_eval",
                   choices=list(PARAMETER_VALUES["scored_row_set"]))
    p.add_argument("--row-set-file", type=Path, default=None,
                   help="amendment CT §CT10: the halves file for "
                        "--scored-row-set ct10_half_a / ct10_half_b")
    # SINGULAR, deliberately. feature_ap.py's `--strata` takes a LIST because
    # every requested stratum shares one activation pass and lands in its own
    # h5 group. A null is a null FOR one statistic on one stratum: the argmax
    # pool, the mask, the effective row count and the degenerate-draw
    # bookkeeping are all stratum-specific. Two parameters differing by a
    # plural would be the `population`/`row_subset` collision again.
    p.add_argument("--stratum", default="pooled",
                   choices=["pooled", "experimental", "propagated"],
                   help="Evidence stratum the null is drawn for. 'pooled' is "
                        "the pre-BG behaviour and the only one that needs no "
                        "masking.")
    p.add_argument("--negatives", default="all_rows",
                   choices=["all_rows", "unannotated"],
                   help="amendment BG. 'unannotated' drops rows carrying the "
                        "concept's label in ANOTHER evidence tier from the "
                        "evaluated universe, so the contrast is 'this tier's "
                        "positives' vs 'no annotation for this concept at "
                        "all'. THE MASKED SET IS HELD FIXED ACROSS DRAWS and "
                        "only positives are displaced (BG1); a displaced "
                        "positive landing on a masked row is dropped (BG2). "
                        "No effect on the pooled stratum, which already owns "
                        "every tier's positives.")
    p.add_argument("--null-type", default="both",
                   choices=["joint_shift", "cross_protein", "both"],
                   help="amendment G4: joint_shift is Null A (within-protein "
                        "circular shift, localization only); cross_protein is "
                        "Null B (across-protein length-stratified permutation, "
                        "the PRIMARY predictor -- protein selection + "
                        "localization). 'both' computes each per replicate off "
                        "the SAME row index (built once per seed either way).")
    p.add_argument("--n-length-bins", type=int, default=20,
                   help="Null B quantile bins on protein length -- permutation "
                        "partners are always drawn from the SAME bin.")
    p.add_argument("--n-shifts", type=int, default=20)
    p.add_argument("--null-seed", type=int, default=0,
                   help="RNG seed for the shift generator -- distinct from, and "
                        "not to be confused with, the ensemble training seeds. "
                        "Null A and Null B draw from independent SeedSequence-"
                        "spawned streams off this one seed, so 'both' is "
                        "reproducible identically to running either alone.")
    p.add_argument("--cache-dir", type=Path, required=True,
                   help="row_index_cache.py storage dir (scratch, regenerable).")
    p.add_argument("--exclude-pooled-parents", action="store_true", default=True,
                   help="amendment I4: drop bare FT types that have sub-typed "
                        "children (BINDING, DOMAIN, REGION, ...) from the "
                        "argmax pool, matching feature_ap.py. On by default; "
                        "the null must range over the same pool as the "
                        "statistic it is a null for.")
    p.add_argument("--include-pooled-parents", dest="exclude_pooled_parents",
                   action="store_false",
                   help="Restore the pre-amendment-I4 behaviour. Produces a "
                        "null for a DIFFERENT statistic than feature_ap.py's "
                        "g_f -- diagnostic use only.")
    p.add_argument("--pilot-n-concepts", type=int, default=0,
                   help="COST-PILOT ONLY (AC5). Restrict the pool to N "
                        "concepts EVENLY SPACED BY MASKED-ROW COUNT, so the "
                        "measured seconds-per-evaluation span the range that "
                        "actually drives cost rather than whatever the "
                        "alphabet puts first. The output is stamped "
                        "`is_cost_pilot=True` and `argmax_pool_scored_n` no "
                        "longer describes the study's pool, so the identity "
                        "gate will refuse it as an input -- a timing run must "
                        "not be able to become an artifact of record.")
    p.add_argument("--concepts-from", type=Path, default=None,
                   help="amendment CT, §CT11.7-R: restrict the stratum's "
                        "concepts to the concept axis of this AP file's "
                        "`<stratum>/concepts` (a multi-stratum feature_ap run "
                        "keeps the strata's COMMON set, A1). Every listed "
                        "concept must have positives in this stratum, or the "
                        "run refuses. Default: every concept with positives.")
    p.add_argument("--ap-device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="Device for the masked (segmented) AP arithmetic, "
                        "separate from the SAE forward pass, which always "
                        "prefers GPU. Same 'auto' rule and same measured "
                        "threshold as feature_ap.py: the float64 temporaries "
                        "scale with the non-zero count and OOM a single card "
                        "at full-eval scale. Ignored on the unmasked path, "
                        "which uses the sparse touch arithmetic.")
    p.add_argument("--narrow-row-index", action="store_true",
                   help="amendment CT, §CT29-D2 item 1 (route b1): hold the "
                        "row index on the device as int16/int32 (narrowing "
                        "refuses rather than wrap) and chunk the touch gather "
                        "and the AP sorts at --ap-max-touches. Each feature's "
                        "AP uses the same touches in the same order. Unmasked "
                        "path only; off by default.")
    p.add_argument("--ap-max-touches", type=int, default=None,
                   help="with --narrow-row-index: the most touches one gather "
                        "chunk or one AP sort holds (default 2^22, §CT29-D2); refused without "
                        "--narrow-row-index rather than ignored.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--raw-out", type=Path, default=None,
                   help="amendment K3: path for the RAW per-(seed, shift, "
                        "concept, feature) gain array. Omitted => raw array "
                        "not stored, which is what forced two re-runs "
                        "already. WHERE IT GOES IS A JUDGEMENT, NOT A RULE: "
                        "the 707-concept pooled raw is ~23 GB and belongs on "
                        "SCRATCH per CLAUDE.md's storage policy, but the "
                        "218-concept masked raw is ~6.7 GB and is the input "
                        "to the FDR pass at ~330-390 GPU-hours to reproduce, "
                        "so it goes to PROJECT and survives the purge "
                        "(amendment BH / brief B §4b item 1).")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    d_model = cfg["model"]["d_model"]
    dict_size, k = cfg["sae"]["dict_size"], cfg["sae"]["k"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  n_shifts={args.n_shifts}  null_seed={args.null_seed}")

    with h5py.File(args.eval_cache, "r") as hf:
        index = json.loads(hf["index_json"][()].decode()
                           if isinstance(hf["index_json"][()], bytes)
                           else str(hf["index_json"][()]))
    subset = load_scored_row_set(args.labels, args.scored_row_set,
                                 args.row_set_file)
    with h5py.File(args.labels, "r") as _hf:
        n_rows = int(_hf.attrs["n_rows"])
        lab_truncation_policy = _hf.attrs.get("truncation_policy")
        lab_git_sha = _hf.attrs.get("git_sha")
        floor_mode = _hf.attrs.get("retention_floor_mode")
        floor_res = _hf.attrs.get("retention_min_residues")
        floor_inst = _hf.attrs.get("retention_min_instances")
        floor_prot = _hf.attrs.get("min_protein_count")
    lab_truncation_policy = (lab_truncation_policy.decode()
                      if isinstance(lab_truncation_policy, bytes) else lab_truncation_policy)
    lab_git_sha = (lab_git_sha.decode()
                   if isinstance(lab_git_sha, bytes) else lab_git_sha)
    floor_mode = (floor_mode.decode()
                  if isinstance(floor_mode, bytes) else floor_mode)

    remap = None
    if subset is None:
        print("building row->protein index (full cache)...")
        row2protein = build_row_to_protein(index, n_rows)
        starts, lengths = protein_starts_lengths(index)
    else:
        # G2: restrict to the labels file's own population (e.g. untruncated).
        # sparse_activations, given `subset`, returns positions RELATIVE TO
        # THE SUBSET (0..len(subset)-1), not original cache rows -- so
        # row2protein/starts/lengths must ALSO be subset-relative, and concept
        # row sets (loaded in original cache-row coordinates) must be remapped
        # the same way feature_ap.py already does.
        print(f"building row->protein index ({args.scored_row_set} population, "
              f"{len(subset):,} of {sum(e['n_rows'] for e in index):,} rows)...")
        pop_index, remap = build_population_index(index, subset)
        n_rows = population_n_rows(subset, n_rows, args.scored_row_set)
        row2protein = build_row_to_protein(pop_index, n_rows)
        starts, lengths = protein_starts_lengths(pop_index)
    n_proteins = len(starts)
    print(f"{n_proteins:,} proteins, {n_rows:,} rows")

    # The stratum supplies the POSITIVES and hence the concept axis; the
    # pooled tier is loaded regardless because `--negatives unannotated` needs
    # it to tell which rows carry the concept under some OTHER evidence tier.
    stratum_labels, _ = load_labels(args.labels, args.stratum)
    pooled, _ = load_labels(args.labels, "pooled")
    with h5py.File(args.labels, "r") as _hf:
        library_concepts = sorted(_hf["concepts"].keys())
    concepts_all = sorted(stratum_labels)
    if args.concepts_from is not None:
        with h5py.File(args.concepts_from, "r") as _hf:
            wanted = [c.decode() if isinstance(c, bytes) else str(c)
                      for c in _hf[args.stratum]["concepts"][:]]
        concepts_all = restrict_to_concepts(concepts_all, wanted)
        print(f"--concepts-from {args.concepts_from}: restricted to "
              f"{len(concepts_all)} concepts")
    if args.stratum != "pooled":
        print(f"stratum={args.stratum}: {len(concepts_all)} of "
              f"{len(library_concepts)} library concepts have positives in it")
    # ---- ARGMAX POOL (amendment I4) -------------------------------------
    # g_f is a max over concepts, so the null for g_f must be a max over the
    # SAME pool. Pooled parents (bare FT types with sub-typed children) are
    # excluded from that pool in feature_ap.py; excluding them there and not
    # here would make this a null for a different statistic -- exactly the
    # defect amendment H6 caught, in a different variable.
    if args.exclude_pooled_parents:
        keep = argmax_pool_mask(concepts_all)
        concepts = [c for c, k in zip(concepts_all, keep) if k]
        excluded = [c for c, k in zip(concepts_all, keep) if not k]
        print(f"{len(concepts_all)} concepts in the library; "
              f"{len(excluded)} pooled parents EXCLUDED from the argmax pool "
              f"-> {len(concepts)} in the pool")
        print(f"  excluded: {sorted(excluded)}")
    else:
        concepts = concepts_all
        print(f"{len(concepts)} concepts (pooled parents NOT excluded -- this "
              f"does NOT match feature_ap.py's pool)")
    def to_subset(arr: np.ndarray) -> np.ndarray:
        """Concept row sets are stored in ORIGINAL cache-absolute coordinates
        (build_residue_labels.py never rewrites entry["start"]) -- remap to
        subset-relative, the same array sparse_activations' `pos` output
        uses."""
        a = arr.astype(np.int64)
        if remap is None:
            return a
        a = a[a < len(remap)]
        m = remap[a]
        return m[m >= 0]

    concept_rows = {c: to_subset(stratum_labels[c]) for c in concepts}
    concept_n_pos = {c: int(v.size) for c, v in concept_rows.items()}

    # ---- THE MASK (amendment BG) ----------------------------------------
    # Identical construction to feature_ap.py's: WITHIN-CONCEPT only. A
    # BINDING residue is genuine evidence that it is not an ACT_SITE, so
    # cross-concept rows stay negative and only the SAME concept's
    # other-tier rows are dropped.
    pooled_rows = {c: to_subset(pooled[c]) for c in concepts if c in pooled}
    keep_row, n_eff, n_masked = build_masks(
        concepts, concept_rows, pooled_rows, args.negatives, n_rows)
    if args.pilot_n_concepts:
        order = sorted(concepts, key=lambda c: (n_masked[c], c))
        step = max(len(order) // args.pilot_n_concepts, 1)
        picked = set(order[::step][:args.pilot_n_concepts])
        concepts = [c for c in concepts if c in picked]
        print(f"COST PILOT: pool restricted to {len(concepts)} concepts evenly "
              f"spaced by masked-row count "
              f"({min(n_masked[c] for c in concepts):,} .. "
              f"{max(n_masked[c] for c in concepts):,} masked rows). "
              f"THIS OUTPUT IS NOT AN ARTIFACT OF RECORD.")
    any_masked = any(keep_row[c] is not None for c in concepts)
    _md = [n_masked[c] for c in concepts]
    print(f"negatives={args.negatives}; masking active: {any_masked}; rows "
          f"masked per concept max {max(_md, default=0):,} "
          f"({max(_md, default=0) / n_rows:.2%}), "
          f"median {int(np.median(_md)) if _md else 0:,}")

    # ---- AP IMPLEMENTATION (amendment I1b) -------------------------------
    # The sparse touch path ranks against the FULL evaluated row set, so a
    # masked row would keep a rank it no longer earns. feature_ap.py REFUSES
    # `--ap-impl sparse` whenever masking is active; the same rule binds here,
    # and for the same reason -- the null must be computed by the same
    # arithmetic as the statistic it is a null for.
    ap_impl = AP_IMPL_SEGMENTED if any_masked else AP_IMPL_SPARSE
    # Same threshold and same reasoning as feature_ap.py's: TopK pins the
    # non-zero count at n_rows*k for a fixed row subset, so the decision is
    # made once, up front, rather than per seed.
    NNZ_GPU_SAFE = 150_000_000
    nnz_estimate = n_rows * k
    if args.ap_device == "auto":
        ap_device = "cpu" if nnz_estimate > NNZ_GPU_SAFE else device
    else:
        ap_device = args.ap_device
    print(f"ap_impl={ap_impl}  ap_device={ap_device} (nnz~{nnz_estimate:,})")
    # amendment CT, §CT29-D2 item 1: the narrow, chunked row-index path.
    if args.narrow_row_index and any_masked:
        raise SystemExit("FATAL: --narrow-row-index is for the unmasked path; "
                         "the masked path does not use the row index")
    if args.ap_max_touches is not None and not args.narrow_row_index:
        raise SystemExit("FATAL: --ap-max-touches applies only with "
                         "--narrow-row-index; refusing rather than ignoring it")
    if args.narrow_row_index and args.ap_max_touches is None:
        args.ap_max_touches = 2 ** 22
    if args.narrow_row_index and args.ap_max_touches < 1:
        raise SystemExit("FATAL: --ap-max-touches must be >= 1")
    narrow_features = dict_size if args.narrow_row_index else None
    chunk_stats: dict = {}
    gpu_peaks: list = []
    index_dtypes: dict | None = None
    if args.narrow_row_index:
        print(f"row index on the device NARROWED (§CT29-D2), gather and AP "
              f"chunked at {args.ap_max_touches:,} touches", flush=True)
    # G1 fix: g_f is a max over NORMALIZED gain, not raw AP. Raw AP is
    # inflated by prevalence (a broad concept has high raw AP even at
    # chance), so a null built by maxing raw AP is dominated by whatever
    # concept has the highest prevalence, at roughly ITS OWN prevalence --
    # exactly what the first pilot run measured (median 0.1331 against
    # pooled DOMAIN's prevalence 0.1334) and mistook for a real finding.
    concept_prevalence = {c: concept_n_pos[c] / n_rows for c in concepts}

    # Null A and Null B draw from INDEPENDENT streams spawned off one seed
    # (numpy SeedSequence), so running --null-type both reproduces EXACTLY
    # the draws either null would get run alone -- neither consumes the
    # other's randomness.
    want_a = args.null_type in ("joint_shift", "both")
    want_b = args.null_type in ("cross_protein", "both")
    # ONE schedule set shared across ALL seeds and reused identically for
    # each -- so seed-to-seed differences in the null are due to the SAEs, not
    # to different proteins happening to get different shifts.
    shift_schedule, dest_schedule, bins = draw_schedules(
        args.null_seed, args.n_shifts, lengths, args.n_length_bins,
        want_a, want_b)
    if want_b:
        print(f"Null B: {args.n_length_bins} requested length bins -> "
              f"{len(np.unique(bins))} realized (duplicate quantile edges "
              f"collapse small ones)")

    n_ck, n_c = len(args.ckpts), len(concepts)

    # ---- DISPLACEMENT PLAN + BH BOOKKEEPING (amendments BG2 / BH) --------
    # Only when masking is active. The unmasked path keeps its original
    # per-shift computation verbatim, so the pooled arm is unchanged
    # operation-for-operation and its regression test can be a BIT-identity
    # criterion rather than a tolerance.
    plan = None
    surviving_counts = {"a": None, "b": None}
    surviving_draws = {"a": None, "b": None}
    resolution = {"a": None, "b": None}
    bh3_case = None
    if any_masked:
        t0 = time.time()
        plan = build_displacement_plan(
            concepts, concept_rows, keep_row, row2protein, starts, lengths,
            shift_schedule, dest_schedule, args.n_shifts)
        for arm in ("a", "b"):
            if plan[arm] is None:
                continue
            counts = plan[arm]["counts"]
            surviving_counts[arm] = counts
            surviving_draws[arm] = (~plan[arm]["degenerate"]).sum(axis=1)
            resolution[arm] = np.array(
                [exceedance_resolution(int(d), dict_size)
                 for d in surviving_draws[arm]], dtype=np.float64)
        nd_a = (plan["a"]["degenerate"].sum(axis=1) if plan["a"] is not None
                else None)
        nd_b = (plan["b"]["degenerate"].sum(axis=1) if plan["b"] is not None
                else None)
        bh3_case = [classify_bh3(None if nd_a is None else int(nd_a[i]),
                                 None if nd_b is None else int(nd_b[i]),
                                 args.n_shifts)
                    for i in range(n_c)]
        tot_a = int(nd_a.sum()) if nd_a is not None else 0
        tot_b = int(nd_b.sum()) if nd_b is not None else 0
        print(f"displacement plan built in {time.time()-t0:.1f}s "
              f"({n_c} concepts x {args.n_shifts} shifts x "
              f"{int(want_a)+int(want_b)} arms)")
        print(f"  BH degenerate draws: Null A {tot_a}, Null B {tot_b} "
              f"(of {n_c * args.n_shifts} cells per arm)")
        for i, c in enumerate(concepts):
            if bh3_case[i] != BH3_NONE:
                da = int(nd_a[i]) if nd_a is not None else None
                db = int(nd_b[i]) if nd_b is not None else None
                print(f"    {c}: {bh3_case[i]} (degenerate A={da} B={db})")
        from collections import Counter
        print(f"  BH3 occupancy: {dict(Counter(bh3_case))}")

    null_g_a = (np.zeros((n_ck, args.n_shifts, dict_size), dtype=np.float64)
               if want_a else None)
    null_g_b = (np.zeros((n_ck, args.n_shifts, dict_size), dtype=np.float64)
               if want_b else None)
    # amendment K3 / I6a: max over FEATURES per concept -- the TRANSPOSE of
    # null_g_*, and NOT derivable from it. Required by the concept-level
    # measure's primary (null-standardization) correction.
    # BH2: a degenerate cell gets NO VALUE. NaN, never 0 -- a zero here would
    # be indistinguishable from a real draw whose gain happened to be zero,
    # and would then be folded into any mean, max or exceedance count that
    # reads the array. `NOT_EVALUABLE` is not `NOT_DETECTED`.
    _cmax_fill = np.nan if any_masked else 0.0
    null_cmax_a = (np.full((n_ck, args.n_shifts, n_c), _cmax_fill, dtype=np.float64)
                   if want_a else None)
    null_cmax_b = (np.full((n_ck, args.n_shifts, n_c), _cmax_fill, dtype=np.float64)
                   if want_b else None)
    seed_names = []
    fire_rate_all = np.zeros((n_ck, dict_size), dtype=np.float64)
    # gotcha 36's device class, MEASURED at the gather rather than read off
    # the SLURM request. `device_seen` catches an array that ran across mixed
    # hardware: one value is the run's, more than one is a fact about the
    # artifact and must not collapse to whichever seed finished last.
    device_record: dict = {}
    device_seen: dict[str, set] = {}

    def _note_devices() -> None:
        for _k, _v in device_record.items():
            device_seen.setdefault(_k, set()).add(_v)

    # ---- RAW per-(seed, shift, concept, feature) array (amendment K3) -----
    # Stored so every downstream reduction is DERIVED rather than baked in.
    # Both re-runs of this pilot were forced by the stored reduction being
    # the wrong one; at ~23 GB against a 10 TiB scratch quota, keeping the
    # raw object is far cheaper than a third re-run.
    raw_hf = raw_a = raw_b = None
    if args.raw_out is not None:
        args.raw_out.parent.mkdir(parents=True, exist_ok=True)
        raw_hf = h5py.File(args.raw_out, "w")
        # float32, NOT float16: sigma estimation squares differences and
        # float16's ~3 significant digits are inadequate for gains this
        # small (Null B per-replicate max ~0.086 +/- 0.033).
        chunks = (1, 1, min(64, n_c), dict_size)
        shape = (n_ck, args.n_shifts, n_c, dict_size)
        if want_a:
            raw_a = raw_hf.create_dataset("gain_a", shape=shape,
                                          dtype=np.float32, chunks=chunks)
        if want_b:
            raw_b = raw_hf.create_dataset("gain_b", shape=shape,
                                          dtype=np.float32, chunks=chunks)
        raw_hf.create_dataset("concepts",
                              data=np.array(concepts, dtype=h5py.string_dtype()))
        # THE SEED AXIS GETS ITS LABELS. Gotcha 29: two orders of this axis
        # exist in the repo, only rows 0 and 1 coincide, and "the artifacts
        # that most needed labels carry none" -- this raw file was one of
        # them. Without it a consumer joining to a seed-indexed array has to
        # RECOVER the order from the data, and a comparison against a
        # different-length reference cannot even align. Written as a
        # placeholder here and filled after the seed loop, since the names
        # are not all known until then.
        raw_hf.create_dataset("seeds", shape=(n_ck,),
                              dtype=h5py.string_dtype())
        gb = n_ck * args.n_shifts * n_c * dict_size * 4 / 1e9
        print(f"raw gain array -> {args.raw_out} "
              f"({gb:.1f} GB per null, {int(want_a)+int(want_b)} nulls)")

    for si, ck in enumerate(args.ckpts):
        seed_names.append(ck.name)
        sd = torch.load(ck / "model.pt", map_location="cpu")
        model = TopKSAE(d_model, dict_size, k, b_dec=torch.zeros(d_model)).to(device)
        model.load_state_dict(sd)
        model.eval()

        if any_masked:
            # ---- MASKED SEED BODY (amendments BG / BH) -------------------
            # The SEGMENTED path, on the feature-major layout, exactly as
            # feature_ap.py's masked branch consumes it. The row-index cache
            # is deliberately NOT used: it stores ranks computed against the
            # full row set, which is the very thing masking invalidates.
            t0 = time.time()
            feat, pos, seg_start = sparse_activations(
                model, args.eval_cache, subset, device, record=device_record)
            fire_rate_all[si] = (torch.bincount(feat, minlength=dict_size)
                                 .numpy() / n_rows)
            feat_d = feat.to(ap_device)
            pos_d = pos.to(ap_device)
            seg_d = seg_start.to(ap_device)
            device_record[AP_ARITH_DEVICE_KEY] = device_class(feat_d.device)
            _note_devices()
            del feat, pos, seg_start
            print(f"seed {ck.name}: activation gather {feat_d.numel():,} "
                  f"non-zeros in {time.time()-t0:.1f}s (ap_device={ap_device})",
                  flush=True)

            run_max_a = (np.zeros((args.n_shifts, dict_size), dtype=np.float64)
                         if want_a else None)
            run_max_b = (np.zeros((args.n_shifts, dict_size), dtype=np.float64)
                         if want_b else None)
            # Per-SEED raw buffers (not per-shift): the loop is now
            # concept-outer, because the filtered non-zeros depend only on the
            # concept's mask and are reused by all of its draws.
            buf_a = (np.full((args.n_shifts, n_c, dict_size), np.nan, np.float32)
                     if raw_a is not None else None)
            buf_b = (np.full((args.n_shifts, n_c, dict_size), np.nan, np.float32)
                     if raw_b is not None else None)
            lab_buf = torch.zeros(n_rows, dtype=torch.bool, device=ap_device)
            t_seed, n_eval = time.time(), 0

            for ci, c in enumerate(concepts):
                kr = keep_row[c]
                # HOISTED OUT OF THE SHIFT LOOP. BG1 holds the masked set
                # FIXED across draws, so the filtered non-zeros and their
                # segment offsets are a property of the CONCEPT alone. This
                # is loop-invariant code motion, not a reformulation of the
                # arithmetic: `average_precision_segmented` below is called
                # on exactly the arrays feature_ap.py would hand it.
                f_c, p_c_pos, s_c = masked_concept_arrays(
                    feat_d, pos_d, seg_d, kr, dict_size)
                n_eff_c = n_eff[c]

                for arm, want, run_max, buf, cmax in (
                        ("a", want_a, run_max_a, buf_a, null_cmax_a),
                        ("b", want_b, run_max_b, buf_b, null_cmax_b)):
                    if not want:
                        continue
                    for r in range(args.n_shifts):
                        surviving = plan[arm]["rows"][c][r]
                        if surviving.size == 0:
                            continue          # BH2: drop the degenerate draw
                        q = torch.from_numpy(surviving).to(ap_device)
                        lab_buf[q] = True
                        gain = masked_draw_gain(
                            f_c, s_c, lab_buf[p_c_pos], dict_size,
                            int(surviving.size), n_eff_c).cpu().numpy()
                        lab_buf[q] = False     # restore, do not reallocate
                        np.maximum(run_max[r], gain, out=run_max[r])
                        cmax[si, r, ci] = float(gain.max())
                        if buf is not None:
                            buf[r, ci] = gain.astype(np.float32)
                        n_eval += 1
                if kr is not None:
                    del f_c, p_c_pos, s_c
                # Every 5 concepts, and always the first: at 218 concepts and
                # ~40 draws each this is a progress line roughly every 25
                # minutes of a many-hour task. A stride of 25 printed nothing
                # for the first two hours, and on the 12-concept cost pilot it
                # printed nothing at all until the very end -- a progress
                # report that cannot report during the run is not one.
                if ci == 0 or (ci + 1) % 5 == 0 or ci + 1 == n_c:
                    el = time.time() - t_seed
                    print(f"  seed {ck.name}: {ci+1}/{n_c} concepts, "
                          f"{n_eval} evaluations, {el:.1f}s "
                          f"({el/max(n_eval,1):.3f}s/evaluation)", flush=True)

            if want_a:
                null_g_a[si] = run_max_a
                if raw_a is not None:
                    raw_a[si] = buf_a
            if want_b:
                null_g_b[si] = run_max_b
                if raw_b is not None:
                    raw_b[si] = buf_b
            el = time.time() - t_seed
            print(f"  seed {ck.name}: {n_eval} masked evaluations in {el:.1f}s "
                  f"= {el/max(n_eval,1):.3f}s each", flush=True)
            del feat_d, pos_d, seg_d, buf_a, buf_b, lab_buf
            torch.cuda.empty_cache() if device == "cuda" else None
            continue

        t0 = time.time()
        if args.narrow_row_index and device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        row_index_record: dict = {}
        feat_by_row, rank_by_row, row_offsets, fire_counts, from_cache = \
            compute_or_load_row_index(model, ck, args.eval_cache, subset,
                                      args.scored_row_set, dict_size, k, device,
                                      args.cache_dir, record=row_index_record,
                                      narrow_features=narrow_features)
        if args.narrow_row_index:
            index_dtypes = {"feat_by_row": str(feat_by_row.dtype),
                            "rank_by_row": str(rank_by_row.dtype),
                            "row_offsets": str(row_offsets.dtype)}
        fire_rate_all[si] = fire_counts.numpy() / n_rows
        # THE UNMASKED PATH NOW STATES ITS GATHER DEVICE, and where it cannot,
        # it still says so (amendment BJ; gotcha 36b).
        #
        # Until BJ the cache key omitted the device, so a HIT was a gather that
        # happened in some earlier job on hardware this run had no record of,
        # and this line wrote `UNKNOWN_CACHED_ROW_INDEX` rather than dress this
        # job's device up as a measurement. The key now carries the device
        # class, so a hit is same-device-class by construction AND the entry
        # records the value MEASURED when it was built -- which is read from
        # the entry rather than assumed. An entry predating the meta field
        # yields `DEVICE_ABSENT`, and that is reported as unknown, not as this
        # job's device: absence is never agreement (gotcha 15).
        _row_index_dc = row_index_record.get(GATHER_DEVICE_KEY, DEVICE_ABSENT)
        device_record[GATHER_DEVICE_KEY] = (
            "UNKNOWN_CACHED_ROW_INDEX" if _row_index_dc == DEVICE_ABSENT
            else _row_index_dc)
        device_record[AP_ARITH_DEVICE_KEY] = device_class(feat_by_row.device)
        _note_devices()
        print(f"seed {ck.name}: row-index {'loaded from cache' if from_cache else 'built'} "
              f"in {time.time()-t0:.1f}s", flush=True)

        for r in range(args.n_shifts):
            t0 = time.time()
            running_max_a = (torch.zeros(dict_size, dtype=torch.float64, device=device)
                             if want_a else None)
            running_max_b = (torch.zeros(dict_size, dtype=torch.float64, device=device)
                             if want_b else None)
            # K3: per-shift buffers, (concepts, features). One shift at a
            # time (~29 MB) rather than a whole seed (~579 MB per null).
            buf_a = (np.empty((n_c, dict_size), dtype=np.float32)
                     if raw_a is not None else None)
            buf_b = (np.empty((n_c, dict_size), dtype=np.float32)
                     if raw_b is not None else None)
            for ci, c in enumerate(concepts):
                if want_a:
                    sp = shift_schedule[r]
                    shifted = shift_rows(concept_rows[c], row2protein, starts, lengths, sp)
                    query_rows = torch.from_numpy(shifted).to(device)
                    if args.narrow_row_index:
                        tf, tr = gather_touches_chunked(
                            feat_by_row, rank_by_row, row_offsets, query_rows,
                            args.ap_max_touches, chunk_stats)
                        ap = average_precision_from_touches_chunked(
                            tf, tr, dict_size, concept_n_pos[c], n_rows,
                            args.ap_max_touches, chunk_stats)
                    else:
                        tf, tr = gather_touches(feat_by_row, rank_by_row, row_offsets, query_rows)
                        ap = average_precision_from_touches(
                            tf, tr, dict_size, concept_n_pos[c], n_rows)
                    p_c = concept_prevalence[c]
                    gain = (ap - p_c) / (1.0 - p_c)        # G1: normalize BEFORE max
                    running_max_a = torch.maximum(running_max_a, gain)
                    null_cmax_a[si, r, ci] = float(gain.max())   # K3/I6a
                    if buf_a is not None:
                        buf_a[ci] = gain.cpu().numpy().astype(np.float32)
                if want_b:
                    dp = dest_schedule[r]
                    shifted_b = cross_protein_shift_rows(
                        concept_rows[c], row2protein, starts, lengths, dp)
                    # Rescaling is many-to-one when the destination protein is
                    # SHORTER than the source (rounding collisions) -- dedup
                    # and use the REALIZED positive count for this draw, not
                    # the concept's true count, so gain-normalization stays
                    # internally consistent with what was actually queried.
                    shifted_b = np.unique(shifted_b)
                    n_pos_b = shifted_b.size
                    query_rows_b = torch.from_numpy(shifted_b).to(device)
                    if args.narrow_row_index:
                        tf_b, tr_b = gather_touches_chunked(
                            feat_by_row, rank_by_row, row_offsets, query_rows_b,
                            args.ap_max_touches, chunk_stats)
                        ap_b = average_precision_from_touches_chunked(
                            tf_b, tr_b, dict_size, n_pos_b, n_rows,
                            args.ap_max_touches, chunk_stats)
                    else:
                        tf_b, tr_b = gather_touches(feat_by_row, rank_by_row, row_offsets, query_rows_b)
                        ap_b = average_precision_from_touches(tf_b, tr_b, dict_size, n_pos_b, n_rows)
                    p_c_b = n_pos_b / n_rows
                    gain_b = (ap_b - p_c_b) / (1.0 - p_c_b)
                    running_max_b = torch.maximum(running_max_b, gain_b)
                    null_cmax_b[si, r, ci] = float(gain_b.max())   # K3/I6a
                    if buf_b is not None:
                        buf_b[ci] = gain_b.cpu().numpy().astype(np.float32)
            msg = f"  seed {ck.name} shift {r}: {time.time()-t0:.1f}s"
            if want_a:
                null_g_a[si, r] = running_max_a.cpu().numpy()
                msg += f"  max null_g_a: {null_g_a[si, r].max():.4f}"
                if raw_a is not None:
                    raw_a[si, r] = buf_a
            if want_b:
                null_g_b[si, r] = running_max_b.cpu().numpy()
                msg += f"  max null_g_b: {null_g_b[si, r].max():.4f}"
                if raw_b is not None:
                    raw_b[si, r] = buf_b
            print(msg, flush=True)

        if args.narrow_row_index and device == "cuda":
            gpu_peaks.append(int(torch.cuda.max_memory_allocated()))
            print(f"  seed {ck.name}: GPU peak allocated "
                  f"{gpu_peaks[-1] / 2**30:.2f} GiB; chunk counts so far "
                  f"{chunk_stats}", flush=True)
        del feat_by_row, rank_by_row, row_offsets
        torch.cuda.empty_cache() if device == "cuda" else None

    # Per-firing-rate-bin sample counts -- how many (feature, seed, replicate)
    # null draws fall in each bin, which is what a pooled-null resolution
    # check needs (amendment A3's run-length calibration is a SEPARATE next
    # step against this output, not computed in this pass).
    # PER-SEED firing-rate bin counts. NOT the cross-seed mean, which is
    # invalid here and was what this block used to do: a feature INDEX is not
    # comparable across independently-seeded SAEs (measured pairwise
    # correlation of fire_rate by index = 0.0009), so averaging over seeds
    # averages 20 unrelated variables. Its SD collapses by ~sqrt(20) onto the
    # global mean -- which TopK pins at exactly k/dict = 0.003125 -- giving a
    # minimum of 1.12e-3, so NO feature can land in any bin below 1e-3 and
    # the three lowest bins come out structurally empty. `null_pilot_g.h5`
    # and the first write of `null_pilot_k.h5` both carry those spurious
    # zeros. Per seed those bins hold 130-2282 features -- exactly the
    # low-firing tail CLAUDE.md says the care goes into.
    #
    # `src/analysis/null_pooling.py` is the authoritative consumer of these
    # counts (G5/K1 three-axis pooling); this block is the quick per-run
    # report and must agree with it.
    bin_counts = per_seed_bin_counts(fire_rate_all, args.n_shifts)
    for b in bin_counts:
        print(f"  bin [{b['lo']:.0e},{b['hi']:.0e}): "
              f"{b['n_features_min_over_seeds']}-{b['n_features_max_over_seeds']} "
              f"features per seed, >= {b['n_null_draws_min']:,} pooled null draws")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as hf:
        # (seeds, shifts, features) each, when present.
        if want_a:
            hf.create_dataset("null_g_a", data=null_g_a, compression="gzip")
            # K3/I6a: max over FEATURES per concept. The transpose of
            # null_g_a and NOT derivable from it.
            hf.create_dataset("null_cmax_a", data=null_cmax_a, compression="gzip")
            hf.create_dataset("shift_schedule", data=shift_schedule, compression="gzip")
        if want_b:
            hf.create_dataset("null_g_b", data=null_g_b, compression="gzip")
            hf.create_dataset("null_cmax_b", data=null_cmax_b, compression="gzip")
            hf.create_dataset("dest_schedule", data=dest_schedule, compression="gzip")
            hf.create_dataset("length_bins", data=bins, compression="gzip")
        hf.create_dataset("fire_rate", data=fire_rate_all, compression="gzip")
        hf.create_dataset("seeds", data=np.array(seed_names, dtype=h5py.string_dtype()))
        # The concept axis the null actually ranged over, in the order every
        # stored array is indexed by. Without it, null_cmax_* and the raw
        # array cannot be joined to the AP matrix -- an identity, not a
        # convenience (I3).
        hf.create_dataset("concepts",
                          data=np.array(concepts, dtype=h5py.string_dtype()))
        # ---- BH BOOKKEEPING (§4b requirements 2, 3 and 4) ----------------
        # Written as DATASETS so no consumer reconstructs any of it from
        # prose, and so the categories exist on disk before anything needs
        # them.
        if any_masked:
            hf.create_dataset("n_masked_rows", data=np.array(
                [n_masked[c] for c in concepts], dtype=np.int64))
            hf.create_dataset("n_rows_effective", data=np.array(
                [n_eff[c] for c in concepts], dtype=np.int64))
            hf.create_dataset("n_positives_observed", data=np.array(
                [concept_n_pos[c] for c in concepts], dtype=np.int64))
            hf.create_dataset("bh3_case", data=np.array(
                bh3_case, dtype=h5py.string_dtype()))
            for arm, want in (("a", want_a), ("b", want_b)):
                if not want:
                    continue
                # (2) per-(concept, shift) surviving POSITIVE count, and the
                #     per-(concept, arm) surviving DRAW count derived from it.
                hf.create_dataset(f"surviving_positives_{arm}",
                                  data=surviving_counts[arm], compression="gzip")
                hf.create_dataset(f"surviving_draws_{arm}",
                                  data=surviving_draws[arm].astype(np.int64))
                # (3) the exceedance resolution, per (concept, arm). NOT a
                #     uniform constant once BH2 drops a draw.
                hf.create_dataset(f"exceedance_resolution_{arm}",
                                  data=resolution[arm])
                # (4) NOT_EVALUABLE, carried even where it is empty. An empty
                #     category that EXISTS cannot silently become a
                #     non-detection on a re-run; one that was never built has
                #     to be retrofitted by whoever notices.
                hf.create_dataset(
                    f"not_evaluable_{arm}",
                    data=(surviving_draws[arm] == 0).astype(bool))
            if want_a and want_b:
                hf.create_dataset(
                    "not_evaluable_stratum",
                    data=((surviving_draws["a"] == 0)
                          & (surviving_draws["b"] == 0)).astype(bool))
        if args.raw_out is not None:
            hf.attrs["raw_gain_path"] = str(args.raw_out)
        write_identity(
            hf, n_shifts=args.n_shifts, null_seed=args.null_seed,
            null_type=args.null_type, n_length_bins=args.n_length_bins,
            n_proteins=n_proteins, n_rows=n_rows,
            bin_counts=json.dumps(bin_counts),
            git_sha=git_sha(), git_dirty=bool(git_dirty()),
            labels=str(args.labels), scored_row_set=args.scored_row_set,
            row_set_file=(str(args.row_set_file) if args.row_set_file else None),
            # gotcha 36 -- MEASURED at the gather, not read off the request.
            **device_identity_attrs(device_record, device_seen),
            # Identity set (I3a). `n_concepts` is the LIBRARY size in every
            # artifact -- the label store's own meaning -- and the pool this
            # null maxed over is `argmax_pool_n`. One key, one meaning.
            #
            # This used to write the POOL into `n_concepts`, which was
            # invisible while no exclusion happened (pool == library == 716)
            # and became a cross-artifact disagreement the moment I4 turned
            # exclusion on: the null said 707 where the label store and
            # feature_ap.py said 716. Caught by the pre-flight gate on the
            # first corrected run -- the same "a decision landed in one place
            # and did not propagate" shape as gotcha 16, in the identity keys
            # themselves.
            #
            # gotcha 16's committed convention, now computed by the same
            # function feature_ap.py uses rather than restated: `n_concepts`
            # and `argmax_pool_n` are always LIBRARY-scoped, `*_scored` carry
            # the stratum-scoped counts. On a pooled run all four collapse
            # pairwise, which is exactly why writing the scored count into a
            # library-scoped key stayed invisible until a stratified run.
            **library_scoped_concept_counts(library_concepts, concepts_all),
            n_concepts_library=len(library_concepts),
            stratum=args.stratum, negatives=args.negatives, ap_impl=ap_impl,
            # §CT29-D2 item 1: the narrow, chunked path, recorded when used
            # (None, so not written, on every other run).
            row_index_device_dtypes=index_dtypes,
            ap_max_touches=(int(args.ap_max_touches) if args.narrow_row_index else None),
            n_chunked_gathers=(int(chunk_stats.get("chunked_gathers", 0))
                               if args.narrow_row_index else None),
            n_chunked_ap_evaluations=(int(chunk_stats.get("chunked_ap_evaluations", 0))
                                      if args.narrow_row_index else None),
            max_feature_chunks=(int(chunk_stats.get("max_feature_chunks", 0))
                                if args.narrow_row_index else None),
            gpu_peak_allocated_bytes_per_seed=(gpu_peaks if gpu_peaks else None),
            is_cost_pilot=bool(args.pilot_n_concepts),
            pilot_n_concepts=int(args.pilot_n_concepts),
            # What the run ACTUALLY maxed over. Equals `argmax_pool_scored_n`
            # on any real run; on a cost pilot it does not, and that
            # disagreement is the signal.
            argmax_pool_used_n=len(concepts),
            exclude_pooled_parents=bool(args.exclude_pooled_parents),
            truncation_policy=lab_truncation_policy, library_sha=lab_git_sha,
            retention_floor_mode=floor_mode,
            retention_min_residues=floor_res,
            retention_min_instances=floor_inst,
            min_protein_count=floor_prot)
    print(f"\nwrote {args.out}")

    if raw_hf is not None:
        raw_hf["seeds"][:] = np.array(seed_names, dtype=object)
        # The raw array carries the SAME identity set as the reductions --
        # a 23 GB file that cannot prove which axis it was drawn on is the
        # stale-identity problem at scale (I3, gotcha 15).
        write_identity(
            raw_hf, n_shifts=args.n_shifts, null_seed=args.null_seed,
            null_type=args.null_type, n_proteins=n_proteins, n_rows=n_rows,
            git_sha=git_sha(), git_dirty=bool(git_dirty()),
            labels=str(args.labels), scored_row_set=args.scored_row_set,
            row_set_file=(str(args.row_set_file) if args.row_set_file else None),
            **device_identity_attrs(device_record, device_seen),
            **library_scoped_concept_counts(library_concepts, concepts_all),
            n_concepts_library=len(library_concepts),
            stratum=args.stratum, negatives=args.negatives, ap_impl=ap_impl,
            # §CT29-D2 item 1: the narrow, chunked path, recorded when used
            # (None, so not written, on every other run).
            row_index_device_dtypes=index_dtypes,
            ap_max_touches=(int(args.ap_max_touches) if args.narrow_row_index else None),
            n_chunked_gathers=(int(chunk_stats.get("chunked_gathers", 0))
                               if args.narrow_row_index else None),
            n_chunked_ap_evaluations=(int(chunk_stats.get("chunked_ap_evaluations", 0))
                                      if args.narrow_row_index else None),
            max_feature_chunks=(int(chunk_stats.get("max_feature_chunks", 0))
                                if args.narrow_row_index else None),
            gpu_peak_allocated_bytes_per_seed=(gpu_peaks if gpu_peaks else None),
            is_cost_pilot=bool(args.pilot_n_concepts),
            pilot_n_concepts=int(args.pilot_n_concepts),
            # What the run ACTUALLY maxed over. Equals `argmax_pool_scored_n`
            # on any real run; on a cost pilot it does not, and that
            # disagreement is the signal.
            argmax_pool_used_n=len(concepts),
            exclude_pooled_parents=bool(args.exclude_pooled_parents),
            truncation_policy=lab_truncation_policy, library_sha=lab_git_sha,
            retention_floor_mode=floor_mode,
            retention_min_residues=floor_res,
            retention_min_instances=floor_inst,
            min_protein_count=floor_prot,
            reductions_path=str(args.out),
            note="RAW per-(seed, shift, concept, feature) gain, amendment K3. "
                 + ("A degenerate draw (BH2) is stored as NaN, never 0 -- no "
                    "value is imputed, and a consumer must not fold a NaN "
                    "into a mean, a max or an exceedance count."
                    if any_masked else
                    "Regenerable intermediate on scratch -- subject to the "
                    "purge, not archived to project."))
        raw_hf.close()
        print(f"wrote {args.raw_out}")


if __name__ == "__main__":
    main()
