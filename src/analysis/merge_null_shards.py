"""
Merge per-seed masked-null shards into one seed-indexed artifact (brief B).

WHY SHARDS EXIST. Amendment BG's null runs under `--negatives unannotated`,
which forces the SEGMENTED average-precision path (I1b), and the whole
20-seed pass is many times the 48 h wall limit. One task per seed costs
nothing extra -- the per-seed activation gather is paid per seed either way --
and buys 20x wall clock.

EVERY GUARD HERE IS REUSED FROM `merge_ap_shards`, NOT REIMPLEMENTED. The seed
ordering, the identity comparison and the positional verification are the same
problems with the same answers, and a second copy is a second thing that can
silently stop agreeing (gotcha 16). What this module adds is the BH
bookkeeping, which is SEED-INVARIANT by construction and is therefore checked
for that rather than concatenated:

  * `surviving_positives_*`, `surviving_draws_*`, `exceedance_resolution_*`,
    `not_evaluable_*` and `bh3_case` are functions of the STORED SCHEDULES and
    the concept's mask. No SAE enters them. So all 20 shards must carry
    BYTE-IDENTICAL copies, and a disagreement means a shard was built against
    a different label store or a different `--null-seed` -- which the identity
    keys would not necessarily show, because none of them is positional.

  * That check is the reason these arrays are worth storing per shard at all.
    Writing them once from shard 0 would have been cheaper and would have made
    the disagreement undetectable.

A degenerate draw is carried through as NaN, never 0 (BH2). The merge asserts
that the NaN pattern in the raw gain matches the degeneracy flags exactly --
a positional check, because a count of NaNs agrees under any permutation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.analysis.artifact_identity import IDENTITY_KEYS, write_identity
from src.analysis.merge_ap_shards import (
    compare_identities,
    concept_axes_identical,
    read_attrs,
    resolve_shard_order,
    verify_positional_seed_order,
)
from src.analysis.null_pilot import per_seed_bin_counts
from src.analysis.seed_axis import normalise_seed_labels
from src.sae.train import git_dirty, git_sha

# Identity keys every shard must agree on with every other. `git_sha` is
# excluded by construction (it is provenance, and identical here only because
# the array runs from one checkout); the run-defining keys are what matter.
SHARD_AGREEMENT_KEYS = IDENTITY_KEYS + (
    "argmax_pool_n", "argmax_pool_scored_n", "n_concepts_scored",
    "scored_row_set", "stratum", "negatives", "ap_impl", "n_shifts",
    "null_seed", "null_type", "n_rows", "n_proteins", "n_length_bins",
    "exclude_pooled_parents", "argmax_pool_used_n",
)

# Seed-invariant by construction: no SAE enters them.
BH_INVARIANT_DATASETS = (
    "surviving_positives_a", "surviving_positives_b",
    "surviving_draws_a", "surviving_draws_b",
    "exceedance_resolution_a", "exceedance_resolution_b",
    "not_evaluable_a", "not_evaluable_b", "not_evaluable_stratum",
    "n_masked_rows", "n_rows_effective", "n_positives_observed",
    "bh3_case", "shift_schedule", "dest_schedule", "length_bins",
)

SEED_AXIS_DATASETS = ("null_g_a", "null_g_b", "null_cmax_a", "null_cmax_b",
                      "fire_rate")


def bh_invariants_identical(arrays: list[np.ndarray],
                            labels: list[str], name: str) -> dict:
    """Byte-identity of a seed-invariant array across shards.

    Compared on the RAW BIT PATTERN for float arrays, so a NaN matches itself
    and +0.0 does not match -0.0. `exceedance_resolution_*` is NaN wherever a
    concept is NOT_EVALUABLE, and an `==` comparison would report every such
    entry as a disagreement -- a check that can only fail is as useless as one
    that can never fail (gotcha 25).
    """
    def view(a):
        if a.dtype.kind == "f":
            return a.view({4: np.uint32, 8: np.uint64}[a.dtype.itemsize])
        return a

    ref = arrays[0]
    wrong = []
    for lab, a in zip(labels[1:], arrays[1:]):
        if a.shape != ref.shape:
            wrong.append({"shard": lab, "reason": "SHAPE",
                          "shape": list(a.shape), "reference": list(ref.shape)})
            continue
        if a.dtype.kind in "OSU":
            same = [x if isinstance(x, str) else x.decode() for x in a.tolist()] == \
                   [x if isinstance(x, str) else x.decode() for x in ref.tolist()]
            n_diff = 0 if same else -1
        else:
            n_diff = int((view(a) != view(ref)).sum())
        if n_diff:
            wrong.append({"shard": lab, "reason": "VALUES", "n_differing": n_diff})
    return {"dataset": name,
            "verdict": "IDENTICAL_ACROSS_SHARDS" if not wrong else "SHARDS_DISAGREE",
            "n_shards": len(arrays), "disagreeing": wrong}


def degeneracy_matches_nan_pattern(gain: np.ndarray,
                                   degenerate: np.ndarray) -> dict:
    """BH2, checked POSITIONALLY on the merged raw array.

    `gain` is (seeds, shifts, concepts, features); `degenerate` is
    (concepts, shifts) and True where every observed positive was displaced
    onto a masked row. A degenerate cell must be ALL NaN and a surviving cell
    must be NO NaN, in every seed.

    A COUNT OF NaNs WOULD AGREE UNDER ANY PERMUTATION of the concept or shift
    axis -- gotcha 27's rule, that only a positional check discriminates a
    mapping defect -- so this compares cell by cell and names the first
    disagreements.
    """
    n_seed, n_shift, n_c, _ = gain.shape
    any_nan = np.isnan(gain).any(axis=3)          # (seeds, shifts, concepts)
    all_nan = np.isnan(gain).all(axis=3)
    expect = degenerate.T[None, :, :]             # (1, shifts, concepts)
    should_be_nan = np.broadcast_to(expect, (n_seed, n_shift, n_c))
    bad_missing = should_be_nan & ~all_nan        # degenerate but has values
    bad_extra = ~should_be_nan & any_nan          # surviving but has NaN
    idx_m = np.argwhere(bad_missing)[:20].tolist()
    idx_e = np.argwhere(bad_extra)[:20].tolist()
    ok = not bad_missing.any() and not bad_extra.any()
    return {"verdict": "NAN_PATTERN_MATCHES_BH2" if ok else "NAN_PATTERN_WRONG",
            "n_degenerate_cells_expected": int(degenerate.sum()) * n_seed,
            "n_cells_all_nan": int(all_nan.sum()),
            "n_degenerate_cells_with_values": int(bad_missing.sum()),
            "n_surviving_cells_with_nan": int(bad_extra.sum()),
            "first_degenerate_cells_with_values_(seed,shift,concept)": idx_m,
            "first_surviving_cells_with_nan_(seed,shift,concept)": idx_e}


def merge(shard_reds: list[Path], shard_raws: list[Path], reference: Path,
          out_red: Path, out_raw: Path) -> dict:
    labels = [read_attrs(p).get("__none__", p.stem) for p in shard_reds]
    identities, seed_labels, concept_lists = [], [], []
    for p in shard_reds:
        identities.append(read_attrs(p))
        with h5py.File(p, "r") as hf:
            seed_labels.append([s.decode() if isinstance(s, bytes) else s
                                for s in hf["seeds"][:]])
            concept_lists.append([c.decode() if isinstance(c, bytes) else c
                                  for c in hf["concepts"][:]])
    labels = [sl[0] for sl in seed_labels]
    for sl, p in zip(seed_labels, shard_reds):
        if len(sl) != 1:
            raise SystemExit(f"{p} carries {len(sl)} seeds; a shard must carry "
                             "exactly one, or the merge's row mapping is a guess.")

    report: dict = {}
    report["identity_agreement"] = compare_identities(
        identities, labels, SHARD_AGREEMENT_KEYS)
    report["concept_axis"] = concept_axes_identical(concept_lists)
    if not report["identity_agreement"]["agree"]:
        raise SystemExit("shards disagree on the run-defining identity keys:\n"
                         + json.dumps(report["identity_agreement"], indent=2))
    if report["concept_axis"]["verdict"] != "IDENTICAL":
        raise SystemExit("shards carry different concept axes:\n"
                         + json.dumps(report["concept_axis"], indent=2))

    with h5py.File(reference, "r") as hf:
        ref_labels = [s.decode() if isinstance(s, bytes) else s
                      for s in hf["seeds"][:]]
    order = resolve_shard_order(labels, ref_labels)
    merged_labels = [labels[i] for i in order]
    report["seed_order"] = verify_positional_seed_order(merged_labels, ref_labels)
    if report["seed_order"]["verdict"] != "POSITIONS_OK":
        raise SystemExit("merged seed axis does not match the reference:\n"
                         + json.dumps(report["seed_order"], indent=2))
    report["reference"] = str(reference)
    report["seed_order_note"] = (
        "Rows are ordered to the REFERENCE artifact's own seed labels, never "
        "positionally. Gotcha 29: two orders of this axis exist in the repo "
        "and only rows 0 and 1 coincide.")

    # ---- the seed-invariant BH bookkeeping -----------------------------
    inv_reports, invariants = [], {}
    with h5py.File(shard_reds[0], "r") as hf0:
        present = [d for d in BH_INVARIANT_DATASETS if d in hf0]
    for name in present:
        arrays = []
        for p in shard_reds:
            with h5py.File(p, "r") as hf:
                arrays.append(hf[name][:])
        r = bh_invariants_identical(arrays, labels, name)
        inv_reports.append(r)
        invariants[name] = arrays[0]
    report["bh_invariants"] = inv_reports
    bad = [r for r in inv_reports if r["verdict"] != "IDENTICAL_ACROSS_SHARDS"]
    if bad:
        raise SystemExit(
            "seed-invariant BH bookkeeping DIFFERS between shards -- a shard "
            "was built against a different label store, schedule or null "
            "seed:\n" + json.dumps(bad, indent=2))

    # ---- write the merged reductions -----------------------------------
    out_red.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_red, "w") as out:
        with h5py.File(shard_reds[0], "r") as hf0:
            seed_axis_present = [d for d in SEED_AXIS_DATASETS if d in hf0]
        for name in seed_axis_present:
            rows = []
            for i in order:
                with h5py.File(shard_reds[i], "r") as hf:
                    rows.append(hf[name][:])
            out.create_dataset(name, data=np.concatenate(rows, axis=0),
                               compression="gzip")
        for name, arr in invariants.items():
            out.create_dataset(name, data=arr, compression="gzip")
        out.create_dataset("concepts", data=np.array(
            concept_lists[0], dtype=h5py.string_dtype()))
        out.create_dataset("seeds", data=np.array(
            merged_labels, dtype=h5py.string_dtype()))
        base = {k: v for k, v in identities[0].items()
                if k not in ("git_sha", "git_dirty", "raw_gain_path", "note")}
        # `bin_counts` is a per-seed summary: shard 0's copy describes one
        # seed (min = max). Recomputed over the MERGED fire_rate, or dropped
        # if there is none, so the merged null never carries one shard's
        # counts under an all-seed label (amendment CT, CT10-17 preamble).
        base.pop("bin_counts", None)
        if "fire_rate" in out:
            base["bin_counts"] = json.dumps(per_seed_bin_counts(
                out["fire_rate"][:], int(base["n_shifts"])))
        write_identity(out, **base, git_sha=git_sha(), git_dirty=bool(git_dirty()),
                       merged_by="merge_null_shards/briefB",
                       n_shards=len(shard_reds),
                       raw_gain_path=str(out_raw),
                       shards=json.dumps([str(p) for p in shard_reds]))

    # ---- write the merged raw ------------------------------------------
    with h5py.File(shard_raws[0], "r") as hf0:
        raw_names = [d for d in ("gain_a", "gain_b") if d in hf0]
        shape0 = hf0[raw_names[0]].shape
        chunks0 = hf0[raw_names[0]].chunks
    n_shift, n_c, n_f = shape0[1], shape0[2], shape0[3]
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_raw, "w") as out:
        for name in raw_names:
            ds = out.create_dataset(
                name, shape=(len(order), n_shift, n_c, n_f),
                dtype=np.float32, chunks=chunks0)
            for row, i in enumerate(order):
                with h5py.File(shard_raws[i], "r") as hf:
                    ds[row] = hf[name][0]
        out.create_dataset("concepts", data=np.array(
            concept_lists[0], dtype=h5py.string_dtype()))
        out.create_dataset("seeds", data=np.array(
            merged_labels, dtype=h5py.string_dtype()))
        write_identity(out, **base, git_sha=git_sha(),
                       git_dirty=bool(git_dirty()),
                       merged_by="merge_null_shards/briefB",
                       n_shards=len(shard_raws),
                       reductions_path=str(out_red),
                       note="A degenerate draw (BH2) is NaN, never 0. No value "
                            "is imputed; do not fold a NaN into a mean, a max "
                            "or an exceedance count.")

    # ---- BH2's NaN pattern, positionally -------------------------------
    degen = {}
    for arm in ("a", "b"):
        key = f"surviving_positives_{arm}"
        if key in invariants:
            degen[arm] = invariants[key] == 0
    nan_reports = []
    with h5py.File(out_raw, "r") as hf:
        for name, arm in (("gain_a", "a"), ("gain_b", "b")):
            if name in hf and arm in degen:
                nan_reports.append({"dataset": name,
                                    **degeneracy_matches_nan_pattern(
                                        hf[name][:], degen[arm])})
    report["bh2_nan_pattern"] = nan_reports
    report["nan_pattern_ok"] = all(
        r["verdict"] == "NAN_PATTERN_MATCHES_BH2" for r in nan_reports)
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shard-reductions", type=Path, nargs="+", required=True)
    p.add_argument("--shard-raw", type=Path, nargs="+", required=True)
    p.add_argument("--reference", type=Path, required=True,
                   help="Artifact whose seed LABELS define the merged row "
                        "order -- normally feature_ap_experimental_av.h5.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--raw-out", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if len(args.shard_reductions) != len(args.shard_raw):
        raise SystemExit(
            f"{len(args.shard_reductions)} reduction shards against "
            f"{len(args.shard_raw)} raw shards -- refusing to pair them by "
            "position when the counts already disagree.")
    report = merge(args.shard_reductions, args.shard_raw, args.reference,
                   args.out, args.raw_out)
    report["git_sha"] = git_sha()
    report["git_dirty"] = bool(git_dirty())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("bh_invariants",)}, indent=2, default=str))
    print(f"\nwrote {args.out}\nwrote {args.raw_out}\nwrote {args.report}")
    if not report["nan_pattern_ok"]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
