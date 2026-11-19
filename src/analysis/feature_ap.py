"""
Per-feature Average Precision against residue concepts.

`g_f` -- a feature's alignment score -- is its best AP across the active concept
set. This module computes the AP matrix (features x concepts) for each seed,
restricted to a row subset and an evidence stratum, per PREREGISTRATION.md.

WHY THIS IS TRACTABLE. A dense AP would be 10,240 features x ~700 concepts x
(the current axis is the 712-concept library / 703-concept argmax pool, since
amendment N; "~97" was the first library, and this paragraph's arithmetic was
written against it) x
17.7M rows. But a TopK SAE fires only k=32 of 10,240 latents per token, so each
feature is non-zero on ~0.3% of rows and every row where it is zero is tied at
the bottom of the ranking. AP therefore splits into
  (a) a segmented scan over only the feature's FIRING rows, and
  (b) a closed-form term for the single tied block of zeros.
That turns an intractable dense problem into one pass over the non-zeros.

    AP = (1/P) * sum_{firing rows i with y_i=1} (TP_i / rank_i)
         + (1 - P_fire/P) * (P/N)

with P = positives in the analysed subset, N = rows in it, TP_i = positives at or
above rank i within the feature, and the second term the tied-zero block (its
precision is exactly P/N and it carries the remaining recall). Ties among
non-zero activations are treated as distinct thresholds; with continuous float
activations exact ties are vanishingly rare, and the dominant tie -- the zero
block -- is handled exactly above. Validated against sklearn in the tests.

Sorting by (feature, -activation) is done ONCE per seed; each concept then costs
one gather plus a segmented scan.

WHAT COUNTS AS A NEGATIVE (`--negatives`). AP needs a negative set, and in a
stratified run the choice is not obvious. For concept `c` in the `experimental`
stratum the positives are `c`'s experimental residues; the question is what to do
with residues carrying a BY-SIMILARITY `c` label.

  `all_rows`     -- every non-positive row in the subset is a negative, so a
                    by-similarity active site IS a negative. A genuine detector
                    fires on it and is charged a false positive, which caps
                    attainable AP near the concept's experimental fraction
                    `f_c = P_exp / P_pooled`. `f_c` ranged 0.006-0.66 across the
                    29 concepts the FIRST library's stratified run analysed (the
                    current axis is 712 / 703), so the cap is concept-dependent and
                    NOT comparable between strata.
  `unannotated`  -- rows carrying a label for `c` in any OTHER evidence tier are
                    DROPPED from the evaluation, so the contrast is
                    "experimentally-confirmed `c`" vs "no `c` annotation at all".
                    Absence of experimental evidence is not evidence of absence
                    -- the same rule that excluded HELIX/STRAND/TURN.

Masking is WITHIN-CONCEPT only: a `BINDING` residue is genuine evidence that it
is not an `ACT_SITE`, so cross-concept rows stay negative. The `pooled` stratum
is unaffected either way (it already owns every tier's positives). The choice is
recorded in the output's `negatives` attribute; a file without that attribute
predates the flag and was built as `all_rows`.
"""

from __future__ import annotations

import argparse
import json
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
    TIER_GROUP_MAP as _TIER_GROUP_MAP,
    argmax_pool_mask,
    assert_eval_cache_identity,
    device_class,
    device_identity_attrs,
    expected_eval_cache_attrs_from_config,
    write_identity,
)
from src.analysis.sparse_ap_index import (
    average_precision_from_touches,
    build_row_index,
    gather_touches,
)
from src.sae.train import TopKSAE, git_dirty, git_sha


# --------------------------------------------------------------- core math --

def average_precision_segmented(
    feat_ids: torch.Tensor,      # (nnz,) feature id per non-zero, sorted ascending
    seg_start: torch.Tensor,     # (n_features,) first index of each feature's block
    labels: torch.Tensor,        # (nnz,) bool: is this firing row a concept positive
    n_features: int,
    n_positives: int,            # P, positives in the analysed row subset
    n_rows: int,                 # N, rows in the analysed subset
) -> torch.Tensor:
    """AP for every feature at once against ONE concept.

    Non-zeros must already be ordered by (feature asc, activation DESC), so
    within each feature's block the rows run from strongest to weakest.
    """
    if n_positives <= 0:
        return torch.zeros(n_features, dtype=torch.float64, device=feat_ids.device)

    nnz = feat_ids.numel()
    if nnz == 0:
        # No feature fires anywhere: every row is tied, so AP is the base rate.
        return torch.full((n_features,), n_positives / n_rows,
                          dtype=torch.float64, device=feat_ids.device)

    lab = labels.to(torch.float64)
    # rank of each non-zero within its own feature block (1-based)
    idx = torch.arange(nnz, device=feat_ids.device, dtype=torch.long)
    rank = (idx - seg_start[feat_ids] + 1).to(torch.float64)

    # TP at each rank, within the feature block: global cumsum minus the cumsum
    # standing at the block's start (exclusive).
    cum = torch.cumsum(lab, dim=0)
    excl = cum - lab                                  # exclusive prefix sum
    # A feature with NO firing rows has seg_start == nnz (one past the end), so
    # clamp before gathering. Its offset is then unused -- no non-zero carries
    # its id -- but the gather itself must stay in bounds.
    seg_off = excl[seg_start.clamp(max=nnz - 1)]
    tp = cum - seg_off[feat_ids]

    contrib = lab * tp / rank                         # precision at each positive
    ap_fire = torch.zeros(n_features, dtype=torch.float64, device=feat_ids.device)
    ap_fire.index_add_(0, feat_ids, contrib)
    p_fire = torch.zeros(n_features, dtype=torch.float64, device=feat_ids.device)
    p_fire.index_add_(0, feat_ids, lab)

    # the tied-zero block: remaining recall at precision exactly P/N
    tail = (1.0 - p_fire / n_positives) * (n_positives / n_rows)
    return ap_fire / n_positives + tail


def filter_nonzeros(
    feat: torch.Tensor, pos: torch.Tensor, keep_row: torch.Tensor, n_features: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop non-zeros whose row is masked out, and rebuild `seg_start`.

    A boolean filter preserves relative order, so the surviving non-zeros are
    still ordered by (feature asc, activation desc) and only the block offsets
    need recomputing -- no re-sort.
    """
    keep = keep_row[pos]
    f2, p2 = feat[keep], pos[keep]
    counts = torch.bincount(f2, minlength=n_features)
    seg2 = torch.cat([torch.zeros(1, dtype=torch.long, device=f2.device),
                      torch.cumsum(counts, 0)[:-1]])
    return f2, p2, seg2


# ------------------------------------------------------- activation gather --

@torch.no_grad()
def sparse_activations(
    model: TopKSAE, cache_path: Path, rows: np.ndarray | None,
    device: str, chunk: int = 16384, record: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-k activations over the analysed rows. THE GATHER.

    Returns (feat_ids, order_row_pos, seg_start) with non-zeros ordered by
    (feature asc, activation desc). `order_row_pos` indexes into the ANALYSED
    subset (0..n_rows-1), not the raw cache, so concept labels must be
    subset-relative too.

    `record`, when given, receives `GATHER_DEVICE_KEY` -> the class of the
    device THE GATHER ACTUALLY RAN ON, read off the encoder's own output rather
    than off the `device` argument. Gotcha 36: this gather is device-dependent
    -- near-tie pre-activations swap under a CPU-versus-GPU float32 matmul and
    TopK then selects a different feature set -- so which device it ran on is
    part of what the resulting artifact IS.

    An OUT-PARAMETER rather than a fourth return value: five of this function's
    six call sites do not want the device, and widening the tuple would edit
    all of them for the benefit of two. The returned tensors are `.cpu()` by
    construction, so a caller CANNOT recover the gather device from them --
    which is the whole reason this has to be recorded here rather than
    inferred by the caller.
    """
    feats_l, pos_l, vals_l = [], [], []
    with h5py.File(cache_path, "r") as hf:
        ds = hf["activations"]
        total = ds.shape[0] if rows is None else len(rows)
        for start in range(0, total, chunk):
            if rows is None:
                sl = slice(start, min(start + chunk, total))
                x = torch.from_numpy(ds[sl].astype(np.float32))
                positions = torch.arange(sl.start, sl.stop)
            else:
                sel = rows[start:start + chunk]        # already sorted ascending
                x = torch.from_numpy(ds[sel].astype(np.float32))
                positions = torch.arange(start, start + len(sel))
            x = x.to(device)
            _, z, _ = model(x)
            if record is not None:
                # From `z`, not from `device`: a job can request one device and
                # have the arithmetic run on another, and that discrepancy is
                # exactly what got four tasks reclaimed.
                record[GATHER_DEVICE_KEY] = device_class(z.device)
            nz = z.nonzero(as_tuple=False)             # (m, 2): row, feature
            feats_l.append(nz[:, 1].to(torch.int32).cpu())
            pos_l.append(positions[nz[:, 0].cpu()].to(torch.int64))
            vals_l.append(z[nz[:, 0], nz[:, 1]].to(torch.float32).cpu())

    feat = torch.cat(feats_l)
    pos = torch.cat(pos_l)
    val = torch.cat(vals_l)
    del feats_l, pos_l, vals_l

    # order by (feature asc, activation desc) -- stable sort on value first,
    # then a stable sort on feature preserves the within-feature ordering.
    o = torch.argsort(val, descending=True, stable=True)
    feat, pos = feat[o], pos[o]
    o2 = torch.argsort(feat.to(torch.int64), stable=True)
    feat, pos = feat[o2].to(torch.int64), pos[o2]

    n_features = model.W_enc.shape[0]
    counts = torch.bincount(feat, minlength=n_features)
    seg_start = torch.cat([torch.zeros(1, dtype=torch.long),
                           torch.cumsum(counts, 0)[:-1]])
    return feat, pos, seg_start


# ------------------------------------------------------------------ driver --

def load_labels(path: Path, stratum: str) -> tuple[dict[str, np.ndarray], int]:
    """{concept: positive cache-row indices} for one evidence stratum."""
    out = {}
    with h5py.File(path, "r") as hf:
        groups = json.loads(hf.attrs["tier_groups"]) if "tier_groups" in hf.attrs \
            else None
        n_rows = int(hf.attrs["n_rows"])
        for concept in hf["concepts"]:
            g = hf["concepts"][concept]
            if stratum == "pooled":
                arr = g["all"][:]
            else:
                tiers = [t for t in g if t != "all"
                         and TIER_GROUP_MAP.get(t) == stratum]
                if not tiers:
                    continue
                arr = np.unique(np.concatenate([g[t][:] for t in tiers]))
            if arr.size:
                out[concept] = arr
    return out, n_rows


# RE-EXPORTED, not defined here. The table moved to `artifact_identity` on
# 2026-08-11 so `preflight` can derive a stratified artifact's expected concept
# axis from the SAME grouping this module applies, without importing torch into
# a launch gate. Kept under this name because `preferential_firing` imports it
# from here; one definition, two import sites.
TIER_GROUP_MAP = _TIER_GROUP_MAP


_SCORED_ROW_SET_PREFIX = {
    "shared_experimental": "has_experimental",   # PREREGISTRATION A1
    "label_store": "population",                 # PREREGISTRATION amendment G2:
                                                 # value -> h5 DATASET prefix,
                                                 # which is unchanged.
                                                   # the build's own population
                                                   # (e.g. the <1024-residue,
                                                   # untruncated set) -- written by
                                                   # build_residue_labels.py
                                                   # whenever --truncation-policy is
                                                   # 'all'; a no-op-sized subset
                                                   # (== every row) otherwise.
}


FILE_ROW_SETS = ("ct10_half_a", "ct10_half_b")   # amendment CT §CT10


def load_row_set_file(path: Path, which: str, labels_path: Path) -> np.ndarray:
    """A row set that lives in a FILE, not the label store (amendment CT §CT10:
    `ct10_halves.py`'s cluster halves). Refuses a file built against another
    label store, and rows outside that store's own population -- a half is a
    subset of `label_store` or it is not the object §CT10 declares."""
    with h5py.File(path, "r") as hf:
        if f"{which}_rows" not in hf:
            raise SystemExit(f"FATAL: {path} has no {which}_rows")
        rows = hf[f"{which}_rows"][:].astype(np.int64)
        built = hf.attrs.get("labels", "")
        built = built.decode() if isinstance(built, bytes) else str(built)
    if Path(built).name != Path(labels_path).name:
        raise SystemExit(f"FATAL: {path} was built against {built}, not {labels_path}")
    if rows.size == 0 or np.any(np.diff(rows) <= 0):
        raise SystemExit(f"FATAL: {path}:{which}_rows is empty or not strictly ascending")
    pop = load_scored_row_set(labels_path, "label_store")
    if not np.all(np.isin(rows, pop)):
        raise SystemExit(f"FATAL: {path}:{which}_rows leaves the label_store population")
    return rows


def load_scored_row_set(labels_path: Path, which: str,
                        row_set_file: Path | None = None) -> np.ndarray | None:
    """Analysed row subset. 'shared_experimental' = rows of proteins carrying at
    least one experimental label (PREREGISTRATION A1). 'label_store' = the row
    subset the labels file was itself BUILT against (amendment G2).
    'ct10_half_a' / 'ct10_half_b' = §CT10's cluster halves, from `row_set_file`
    (required for them, refused for every other value)."""
    if which in FILE_ROW_SETS:
        if row_set_file is None:
            raise SystemExit(f"FATAL: scored_row_set {which} needs --row-set-file")
        return load_row_set_file(row_set_file, which, labels_path)
    if row_set_file is not None:
        raise SystemExit(f"FATAL: --row-set-file given for scored_row_set {which}")
    if which == "full_eval":
        return None
    prefix = _SCORED_ROW_SET_PREFIX[which]
    with h5py.File(labels_path, "r") as hf:
        pg = hf["protein_sets"]
        starts = pg[f"{prefix}_start"][:]
        counts = pg[f"{prefix}_n_rows"][:]
    rows = np.concatenate([np.arange(s, s + c) for s, c in zip(starts, counts)])
    # MUST be ascending: the protein sets are stored in ACCESSION order, but
    # h5py fancy indexing requires strictly increasing indices. Sorting also
    # keeps `remap` (cache row -> subset position) consistent with the read
    # order used in sparse_activations, since both use this same array.
    return np.sort(rows)


def library_scoped_concept_counts(library_concepts, scored_concepts) -> dict:
    """The four concept-count identity fields, computed as a pure function of
    two concept lists so this can be unit-tested without a GPU, a checkpoint,
    or an activation cache (gotcha 13 extension 5: a computation left inside
    `main()` cannot be handed a fixture with a known answer).

    amendment AW / gotcha 16's committed convention, now enforced by
    construction rather than by which list happened to get passed where:
    `n_concepts` and `argmax_pool_n` are always LIBRARY-scoped (independent of
    which stratum was requested); `n_concepts_scored` and
    `argmax_pool_scored_n` carry the STRATUM-scoped counts under names that
    say what they are. On a pooled run `scored_concepts == library_concepts`
    and all four collapse pairwise -- which is exactly why the previous bug
    (writing the scored count into the library-scoped key) was invisible
    until a stratified run, where a concept can be in the library with zero
    positives in that stratum, made the two diverge.
    """
    return {
        "n_concepts": len(library_concepts),
        "argmax_pool_n": int(sum(argmax_pool_mask(library_concepts))),
        "n_concepts_scored": len(scored_concepts),
        "argmax_pool_scored_n": int(sum(argmax_pool_mask(scored_concepts))),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--ckpts", type=Path, nargs="+", required=True)
    p.add_argument("--eval-cache", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--strata", nargs="+", default=["pooled"],
                   choices=["pooled", "experimental", "propagated"],
                   help="All requested strata share ONE activation pass per "
                        "seed -- activations do not depend on the labels.")
    p.add_argument("--scored-row-set", default="shared_experimental",
                   choices=list(PARAMETER_VALUES["scored_row_set"]),
                   help="A1's shared_experimental; 'label_store' (amendment G2) "
                        "restricts to whatever population the labels file was "
                        "built against (e.g. the untruncated/<1024 set).")
    p.add_argument("--row-set-file", type=Path, default=None,
                   help="amendment CT §CT10: the halves file for "
                        "--scored-row-set ct10_half_a / ct10_half_b")
    p.add_argument("--negatives", default="all_rows",
                   choices=["all_rows", "unannotated"],
                   help="What a negative is in a STRATIFIED run (see module "
                        "docstring). 'all_rows' keeps the pre-amendment-C "
                        "behaviour, in which a by-similarity label for the same "
                        "concept is an ordinary negative; 'unannotated' drops "
                        "those rows. No effect on the pooled stratum.")
    p.add_argument("--per-stratum-concepts", action="store_true",
                   help="Disable the common-concept-set restriction and let each "
                        "stratum keep its own surviving concepts. Not the "
                        "pre-registered primary analysis.")
    p.add_argument("--ap-device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="Device for the per-concept AP ARITHMETIC (segmented "
                        "cumsum/index_add), separate from --device/the SAE "
                        "forward pass, which always prefers GPU. 'auto' (the "
                        "default) switches to CPU above a non-zero-count "
                        "threshold where the AP math's float64 temporaries "
                        "exceed a single card's VRAM: measured, 61M non-zeros "
                        "(1.9M-row shared_experimental subset) fits a 32GB "
                        "rtx_5000_ada; 567M (17.7M-row full eval) OOMs it.")
    p.add_argument("--ap-impl", default="auto",
                   choices=["auto", "sparse", "segmented"],
                   help="AP arithmetic implementation (amendment I1b). "
                        "'sparse' is the P1 restructure -- cost is bounded by "
                        "the concept's touched (feature, rank) pairs rather "
                        "than a full non-zero scan per concept; measured 14.7x "
                        "faster (162.66s -> 11.09s, job 19958667) and validated "
                        "bit-identical. 'auto' uses sparse wherever no "
                        "per-concept row masking is active (i.e. always, unless "
                        "--negatives unannotated), and segmented where it is, "
                        "because masking changes within-feature ranks.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    d_model = cfg["model"]["d_model"]
    dict_size, k = cfg["sae"]["dict_size"], cfg["sae"]["k"]
    # amendment CQ, A4: the eval cache's identity attrs must EQUAL the config's
    # model / layer / normalisation / length cap, not merely be present. A
    # layer-23 cache would otherwise be gathered silently. Raises on mismatch.
    eval_cache_check = assert_eval_cache_identity(
        args.eval_cache, expected_eval_cache_attrs_from_config(cfg),
        label=f"config:{args.config}")
    print(f"  eval cache identity: {eval_cache_check['verdict']} "
          f"({ {k: v['cache'] for k, v in eval_cache_check['per_key'].items()} })")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU visible.")

    subset = load_scored_row_set(args.labels, args.scored_row_set,
                                 args.row_set_file)
    n_rows = len(subset) if subset is not None else \
        int(h5py.File(args.labels, "r").attrs["n_rows"])
    print(f"subset={args.scored_row_set} ({n_rows:,} rows), "
          f"strata={args.strata}, {len(args.ckpts)} seeds")

    # TopK guarantees exactly k firing features per row, so non-zero count is
    # constant across seeds for a fixed row subset -- decide the AP-arithmetic
    # device once, up front, rather than per seed.
    NNZ_GPU_SAFE = 150_000_000
    nnz_estimate = n_rows * k
    if args.ap_device == "auto":
        ap_device = "cpu" if nnz_estimate > NNZ_GPU_SAFE else device
    else:
        ap_device = args.ap_device
    if ap_device != device:
        print(f"  ap_device={ap_device} (nnz~{nnz_estimate:,} > {NNZ_GPU_SAFE:,} "
              f"-- the AP arithmetic's float64 temporaries would OOM a GPU at "
              f"this scale; the SAE forward pass still runs on {device})")

    remap = None
    if subset is not None:
        remap = -np.ones(int(subset.max()) + 2, dtype=np.int64)
        remap[subset] = np.arange(len(subset))

    def to_subset(concepts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Cache row indices -> positions within the analysed subset."""
        if remap is None:
            return concepts
        rel = {}
        for c, arr in concepts.items():
            a = arr[arr < len(remap)]
            m = remap[a]
            m = m[m >= 0]
            if m.size:
                rel[c] = m
        return rel

    # Per stratum: concept -> positive positions RELATIVE to the analysed subset.
    per_stratum: dict[str, dict[str, np.ndarray]] = {}
    for st in args.strata:
        per_stratum[st] = to_subset(load_labels(args.labels, st)[0])
        print(f"  {st}: {len(per_stratum[st])} concepts with positives in the subset")
    # Always needed for `--negatives unannotated`: the union over every tier, so
    # a stratum can tell which rows are annotated for the concept but not by its
    # own evidence tier. Loaded even when 'pooled' was not requested.
    all_tiers = to_subset(load_labels(args.labels, "pooled")[0])

    # ---- THE LIBRARY'S FLOOR IS THE FLOOR. Consumers do not impose their own.
    # (PREREGISTRATION amendment I1a; the defect it fixes is amendment H6.)
    #
    # This block used to re-apply `annotations.min_positive_residues` (100)
    # after subsetting -- amendment B's rule, written for the 97-concept
    # LEGACY-floor axis. Against the current library, built under the
    # `interplm` DISJUNCTIVE floor (residues > 1500 OR instances > 10), that
    # silently dropped 105 of 716 concepts -- 75 of them `BINDING:<ligand>`,
    # i.e. precisely the sparse functional-site targets the disjunctive floor
    # exists to retain. Meanwhile null_pilot.py applied NO floor and maxed
    # over all 716, so the null was a max over 716 and the real g_f a max over
    # 611: a systematic upward bias in the null that under-calls alignment,
    # invisibly.
    #
    # SINGLE SOURCE OF TRUTH: floor parameters are read from the LABEL STORE's
    # own attrs, by the builder and by every consumer alike. Nothing here
    # re-derives or re-imposes them. That prevents this whole class by
    # construction rather than by vigilance.
    with h5py.File(args.labels, "r") as _hf:
        floor_mode = _hf.attrs.get("retention_floor_mode")
        floor_res = _hf.attrs.get("retention_min_residues")
        floor_inst = _hf.attrs.get("retention_min_instances")
        floor_prot = _hf.attrs.get("min_protein_count")
        lab_truncation_policy = _hf.attrs.get("truncation_policy")
        lab_git_sha = _hf.attrs.get("git_sha")
        # amendment AW / gotcha 16: the LIBRARY, read from the store's own
        # concept group -- independent of which stratum was requested. On a
        # pooled run this coincides with names['pooled'], which is why the
        # bug (below) was invisible until a stratified run diverged from it.
        library_concepts = sorted(_hf["concepts"].keys())
    lab_truncation_policy = (lab_truncation_policy.decode()
                      if isinstance(lab_truncation_policy, bytes) else lab_truncation_policy)
    lab_git_sha = (lab_git_sha.decode()
                   if isinstance(lab_git_sha, bytes) else lab_git_sha)
    if floor_mode is None:
        raise SystemExit(
            f"{args.labels} records no `retention_floor_mode`. Floor "
            "parameters are read from the label store, never re-derived here "
            "(amendment I1a) -- refusing to guess which floor built it.")
    floor_mode = floor_mode.decode() if isinstance(floor_mode, bytes) else floor_mode
    print(f"  floor (from the label store, NOT re-applied here): "
          f"mode={floor_mode} residues>{floor_res} OR instances>{floor_inst}, "
          f"min_protein_count={floor_prot}")
    for st in args.strata:
        print(f"  {st}: {len(per_stratum[st])} concepts carried through "
              f"(no consumer-side floor)")

    # ---- COMMON concept set across the compared strata --------------------
    # Same principle as A1: if the strata carry different concept sets, "aligned"
    # means something different in each and a stratum difference could come from
    # the concept set rather than evidence provenance. Restrict to concepts
    # meeting the floor in EVERY requested stratum so tier is the only variable.
    if len(args.strata) > 1 and not args.per_stratum_concepts:
        common = set.intersection(*(set(per_stratum[st]) for st in args.strata))
        for st in args.strata:
            per_stratum[st] = {c: v for c, v in per_stratum[st].items()
                               if c in common}
        print(f"  COMMON concept set across strata: {len(common)} concepts")
    if any(not per_stratum[st] for st in args.strata):
        raise SystemExit("a stratum has no concepts left after the floor -- "
                         "refusing to write a degenerate AP matrix")

    def as_mask(idx: np.ndarray) -> torch.Tensor:
        return torch.zeros(n_rows, dtype=torch.bool, device=ap_device).index_fill_(
            0, torch.from_numpy(idx).to(ap_device), True)

    lab_bool = {st: {c: as_mask(idx) for c, idx in concepts.items()}
                for st, concepts in per_stratum.items()}
    names = {st: sorted(c) for st, c in per_stratum.items()}

    # ---- Negative definition (see module docstring / PREREGISTRATION C) -----
    # Under 'unannotated', rows annotated for the concept under ANOTHER evidence
    # tier are dropped rather than scored as negatives, so each stratum's
    # negatives are rows with no annotation for that concept at all. That changes
    # N per (stratum, concept), hence prevalence, hence the normalised-gain
    # threshold -- so both are recomputed here rather than reused.
    keep_row: dict[str, dict[str, torch.Tensor | None]] = {}
    n_eff: dict[str, dict[str, int]] = {}
    for st in args.strata:
        keep_row[st], n_eff[st] = {}, {}
        for c in names[st]:
            other = np.setdiff1d(all_tiers.get(c, np.zeros(0, dtype=np.int64)),
                                 per_stratum[st][c], assume_unique=False)
            if args.negatives == "unannotated" and other.size:
                keep_row[st][c] = ~as_mask(other)
                n_eff[st][c] = n_rows - int(other.size)
            else:
                keep_row[st][c] = None
                n_eff[st][c] = n_rows
        dropped = [n_rows - n_eff[st][c] for c in names[st]]
        print(f"  {st}: negatives={args.negatives}; rows masked per concept "
              f"max {max(dropped, default=0):,} ({max(dropped, default=0)/n_rows:.2%}), "
              f"median {int(np.median(dropped)) if dropped else 0:,}")

    # ---- AP implementation (amendment I1b) ------------------------------
    any_masked = any(keep_row[st][c] is not None
                     for st in args.strata for c in names[st])
    if args.ap_impl == "auto":
        use_sparse = not any_masked
    else:
        use_sparse = args.ap_impl == "sparse"
    if use_sparse and any_masked:
        raise SystemExit(
            "--ap-impl sparse with per-concept masking active: the sparse "
            "path ranks against the FULL evaluated row set, so masked rows "
            "would keep ranks they no longer earn. Use 'auto' or 'segmented'.")
    ap_impl = AP_IMPL_SPARSE if use_sparse else AP_IMPL_SEGMENTED
    print(f"  ap_impl={ap_impl} (masking active: {any_masked})")

    ap_all = {st: np.zeros((len(args.ckpts), dict_size, len(names[st])),
                           dtype=np.float32) for st in args.strata}
    fire_all = np.zeros((len(args.ckpts), dict_size), dtype=np.int64)
    # Filled by the gather itself, seed by seed, and written into the
    # artifact's identity. Every seed of one run gathers on the same device, so
    # the last write is the run's; a disagreement across seeds would mean the
    # array landed on mixed hardware and is caught below.
    device_record: dict = {}
    device_seen: dict[str, set] = {}

    for si, ck in enumerate(args.ckpts):
        sd = torch.load(ck / "model.pt", map_location="cpu")
        model = TopKSAE(d_model, dict_size, k, b_dec=torch.zeros(d_model)).to(device)
        model.load_state_dict(sd)
        model.eval()
        # ONE activation pass per seed, reused by every stratum. The forward
        # pass itself always runs on `device` (GPU); only the AP arithmetic
        # that follows moves to `ap_device`.
        feat, pos, seg_start = sparse_activations(model, args.eval_cache, subset,
                                                  device, record=device_record)
        feat_d, pos_d, seg_d = feat.to(ap_device), pos.to(ap_device), seg_start.to(ap_device)
        # The AP arithmetic's device is READ BACK off a tensor that has landed
        # there, not copied from `ap_device` -- same rule as the gather.
        device_record[AP_ARITH_DEVICE_KEY] = device_class(feat_d.device)
        for _k, _v in device_record.items():
            device_seen.setdefault(_k, set()).add(_v)
        fire_all[si] = torch.bincount(feat, minlength=dict_size).numpy()

        # ---- ROW INDEX for the restructured sparse path (amendment I1b) ----
        # Built once per seed from the SAME gather, then reused by every
        # concept in every stratum. O(nnz) once, against the old path's
        # O(nnz) PER CONCEPT.
        row_idx = None
        if use_sparse:
            row_idx = build_row_index(feat_d, pos_d, seg_d, n_rows)

        for st in args.strata:
            for ci, c in enumerate(names[st]):
                kr = keep_row[st][c]
                # The sparse path computes ranks against the FULL evaluated row
                # set. Under `--negatives unannotated` some rows are dropped
                # per concept, which changes those ranks, so a masked concept
                # must go through the segmented path on the filtered arrays.
                # Unmasked concepts -- every concept of the pooled stratum, and
                # therefore the whole main run -- take the fast path.
                if use_sparse and kr is None:
                    q = torch.from_numpy(
                        per_stratum[st][c].astype(np.int64)).to(ap_device)
                    tf, tr = gather_touches(*row_idx, q)
                    ap = average_precision_from_touches(
                        tf, tr, dict_size,
                        int(per_stratum[st][c].size), n_eff[st][c])
                else:
                    if kr is None:
                        f_c, p_c, s_c = feat_d, pos_d, seg_d
                    else:
                        f_c, p_c, s_c = filter_nonzeros(feat_d, pos_d, kr, dict_size)
                    ap = average_precision_segmented(
                        f_c, s_c, lab_bool[st][c][p_c], dict_size,
                        int(per_stratum[st][c].size), n_eff[st][c])
                ap_all[st][si, :, ci] = ap.float().cpu().numpy()
        del row_idx
        print(f"  seed {ck.name}: {feat.numel():,} non-zeros; " +
              "  ".join(f"{st} maxAP {ap_all[st][si].max():.4f}"
                        for st in args.strata), flush=True)
        del feat, pos, seg_start, feat_d, pos_d, seg_d
        torch.cuda.empty_cache() if device == "cuda" else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as hf:
        hf.create_dataset("fire_counts", data=fire_all, compression="gzip")
        hf.create_dataset("seeds", data=np.array(
            [c.name for c in args.ckpts], dtype=h5py.string_dtype()))
        for st in args.strata:
            g = hf.create_group(st)
            g.create_dataset("ap", data=ap_all[st], compression="gzip")
            g.create_dataset("concepts", data=np.array(
                names[st], dtype=h5py.string_dtype()))
            # Prevalence is against the EFFECTIVE row count, which under
            # --negatives unannotated differs per concept -- G = (AP - p)/(1 - p)
            # must use the same denominator the AP was computed against.
            g.create_dataset("prevalence", data=np.array(
                [per_stratum[st][c].size / n_eff[st][c] for c in names[st]],
                dtype=np.float64))
            g.create_dataset("n_rows_effective", data=np.array(
                [n_eff[st][c] for c in names[st]], dtype=np.int64))
            g.create_dataset("n_positives", data=np.array(
                [per_stratum[st][c].size for c in names[st]], dtype=np.int64))
            # ---- ARGMAX POOL (amendment I4) -----------------------------
            # Pooled parents (bare FT types that have sub-typed children --
            # BINDING, DOMAIN, REGION, TOPO_DOM, ...) stay in the matrix as a
            # separately-reported diagnostic arm, but MUST NOT be eligible to
            # win the argmax that defines g_f. Excluding a concept from the
            # FDR family is not the same as excluding it from a MAXIMUM: g_f
            # is a max, so a parent left in the pool can still win it and D5's
            # breadth-dominance problem would be unfixed by construction.
            # Measured on the 716-concept library: 9 parents, each 83.5-99.8%
            # covered by its own children, and 4 of them among the 10 broadest
            # concepts. Consumers MUST apply this mask before any max/argmax,
            # and the null must apply the identical mask.
            in_pool = argmax_pool_mask(names[st])
            g.create_dataset("in_argmax_pool", data=np.array(in_pool, dtype=bool))
        write_identity(
            hf,
            scored_row_set=args.scored_row_set, n_rows=n_rows,
            row_set_file=(str(args.row_set_file) if args.row_set_file else None),
            negatives=args.negatives, strata=json.dumps(args.strata),
            git_sha=git_sha(), git_dirty=bool(git_dirty()),
            ap_impl=ap_impl,
            # amendment CQ, A4: the cache's asserted identity, carried.
            eval_cache_identity=json.dumps(
                {k: v["cache"] for k, v in eval_cache_check["per_key"].items()}),
            # gotcha 36 -- MEASURED at the gather, not read off the request.
            **device_identity_attrs(device_record, device_seen),
            # identity set -- carried FROM the label store so a consumer can
            # assert agreement without re-opening it (amendment I3a)
            truncation_policy=lab_truncation_policy,
            library_sha=lab_git_sha,
            retention_floor_mode=floor_mode,
            retention_min_residues=floor_res,
            retention_min_instances=floor_inst,
            min_protein_count=floor_prot,
            # gotcha 16's committed convention, now true AT THE WRITER rather
            # than patched downstream (amendment AW / §2d) -- see
            # `library_scoped_concept_counts`'s docstring for why.
            **library_scoped_concept_counts(library_concepts, names[args.strata[0]]),
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
