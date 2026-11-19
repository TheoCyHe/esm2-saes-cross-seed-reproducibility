"""
Merge per-seed AP shards into one seed-indexed artifact (PREREGISTRATION
amendment AV10d).

WHY SHARDS EXIST AT ALL. The experimental-stratum pass runs under
`--negatives unannotated` (C4), which forces the SEGMENTED AP path: the sparse
path (I1b) ranks against the full evaluated row set, so masked rows would keep
ranks they no longer earn. AV10c measured the segmented path at ~12.7 s per
concept-evaluation at 15,972,325 rows against the sparse path's 0.024 s, which
puts a 20-seed monolithic run at ~52 h -- OVER the 48 h wall-time limit. The
per-seed activation gather is already paid per seed, so one task per seed costs
nothing extra and buys 20x wall clock.

WHAT THIS MODULE GUARDS, AND WHY EACH GUARD IS POSITIONAL RATHER THAN A TOTAL.

  * **Seed order.** Gotcha 29: AP artifacts are written in LEXICOGRAPHIC seed
    order (`glob("seed_*")` sorts that way) and the reproducibility artifacts in
    NUMERIC order; only rows 0 and 1 coincide, so a positional join mis-pairs 18
    of 20 seeds and produces a complete, internally consistent, entirely
    plausible, wrong result. **Every identity key the standard gates compare is
    invariant under a seed permutation** -- `library_sha`, `n_concepts`,
    `argmax_pool_n`, `truncation_policy`, `scored_row_set`, `n_rows` -- so
    `preflight` and `check_shared_identity` cannot see it, by construction and
    not by oversight. The merged rows are therefore ordered to match a NAMED
    REFERENCE artifact's own seed labels, and the ordering is asserted
    elementwise afterwards.

  * **Concept axis.** The concept set is a function of the label store and the
    row subset, never of the checkpoint, so all shards must carry byte-identical
    concept lists. A shard that does not is a shard built against a different
    store.

  * **Identity.** Absent is reported ABSENT, never treated as matching (I3).

  * **Stratum comparability (AV10d).** The experimental artifact and the pooled
    one must agree on population, library, row subset, floor parameters and the
    seed label sequence, and differ in `strata` and `negatives` and NOTHING
    else. That is asserted and REPORTED rather than assumed -- the two are
    compared cell by cell downstream, and a difference in any other key would
    make the comparison meaningless while looking fine.

Every check below lives in a NAMED FUNCTION that can be handed a constructed
input with a known answer. Seven of nineteen mutations survived the last time
this kind of logic sat inside `main()` (AU9b), every one because the only
fixture available was whatever the pipeline happened to produce.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.analysis.artifact_identity import IDENTITY_KEYS, argmax_pool_mask
from src.analysis.seed_axis import normalise_seed_labels

# Keys the shards must agree on with each other AND, for the comparability
# check, with the reference pooled artifact. `strata` and `negatives` are
# DELIBERATELY absent: they are the two the experimental artifact is allowed
# -- and expected -- to differ on.
SHARD_AGREEMENT_KEYS = IDENTITY_KEYS + (
    "scored_row_set", "n_rows", "argmax_pool_n", "ap_impl",
)

# The keys AV10d requires the experimental and pooled artifacts to SHARE.
# `ap_impl` is not among them: masking forces `segmented` on one side and the
# other ran `sparse`, which is a consequence of the negative convention rather
# than an independent difference.
COMPARABILITY_KEYS = IDENTITY_KEYS + ("scored_row_set", "n_rows")

# The keys they are ALLOWED to differ on. Anything outside this set that
# differs is a failure, so the check is closed rather than open-ended.
COMPARABILITY_EXPECTED_DIFFERENCES = ("strata", "negatives", "ap_impl")


def read_attrs(path: Path) -> dict:
    """Every root attr of an h5, decoded. Missing keys are ABSENT, not
    defaulted -- a defaulted identity is how a mismatch hides (I3)."""
    out = {}
    with h5py.File(path, "r") as hf:
        for k, v in hf.attrs.items():
            out[k] = v.decode() if isinstance(v, bytes) else v
    return out


def compare_identities(identities: list[dict], labels: list[str],
                       keys: tuple[str, ...]) -> dict:
    """Agreement report over `keys` across N artifacts.

    Returns a dict with `agree`, `disagreements` and `absent`. **A key absent
    from SOME artifacts is reported under `absent` and never counted as
    agreement** -- `null_pilot_g.h5` shipped with no `truncation_policy` at all,
    and 'absent' silently reading as 'agreement' is exactly how a mismatch
    hides.
    """
    disagreements, absent = [], []
    for k in keys:
        present = [(lab, i[k]) for lab, i in zip(labels, identities) if k in i]
        if len(present) != len(identities):
            missing = [lab for lab, i in zip(labels, identities) if k not in i]
            absent.append({"key": k, "missing_from": missing,
                           "n_missing": len(missing)})
            continue
        values = {_hashable(v) for _, v in present}
        if len(values) > 1:
            disagreements.append({"key": k,
                                  "values": {lab: _plain(v) for lab, v in present}})
    return {"agree": not disagreements and not absent,
            "n_keys_checked": len(keys),
            "disagreements": disagreements,
            "absent": absent}


def _hashable(v):
    return v.item() if isinstance(v, np.generic) else v


def _plain(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def resolve_shard_order(shard_labels: list[str],
                        reference_labels: list[str]) -> list[int]:
    """Index into the shard list for each position of `reference_labels`.

    Both sides are normalised through `seed_axis.normalise_seed_labels` first,
    because three writers in this repo spell the same axis three ways
    (`seed_7`, `7`, `b'seed_7'`) and two lists naming the same seeds cannot be
    compared until they are spelled alike.

    **Raises if the seed SETS differ, not merely their order.** A genuinely
    missing seed must never be silently dropped by an index lookup that skips
    it -- that turns a 20-seed artifact into a 19-seed one with a correct-looking
    shape.
    """
    src = normalise_seed_labels(shard_labels)
    ref = normalise_seed_labels(reference_labels)
    if len(set(src)) != len(src):
        raise ValueError(f"duplicate seed labels among the shards: {src}")
    if set(src) != set(ref):
        raise ValueError(
            f"seed sets differ, not just order: missing from shards "
            f"{sorted(set(ref) - set(src))}, missing from reference "
            f"{sorted(set(src) - set(ref))}. Refusing to merge a partial "
            "ensemble into an artifact whose shape would not show the gap.")
    return [src.index(r) for r in ref]


def verify_positional_seed_order(merged_labels: list[str],
                                 reference_labels: list[str]) -> dict:
    """Row-by-row confirmation that the merged axis IS the reference axis.

    This is deliberately separate from `resolve_shard_order` and is run on the
    RESULT rather than on the plan. Gotcha 27's rule: for a mapping defect only
    a POSITIONAL check discriminates -- a total agrees under any permutation and
    so does a rank correlation, so neither can see a reordering. This one
    reports the first disagreeing ROW.
    """
    a = normalise_seed_labels(merged_labels)
    b = normalise_seed_labels(reference_labels)
    if len(a) != len(b):
        return {"verdict": "LENGTH_MISMATCH", "n_merged": len(a),
                "n_reference": len(b), "n_rows_correct": 0}
    wrong = [{"row": i, "merged": a[i], "reference": b[i]}
             for i in range(len(a)) if a[i] != b[i]]
    return {"verdict": "POSITIONS_OK" if not wrong else "POSITIONS_WRONG",
            "n_rows": len(a), "n_rows_correct": len(a) - len(wrong),
            "wrong_rows": wrong[:20]}


# Amendment CT, §CT11.7-R: A1's shared-protein-set strata have no reference AP
# on their own population, so the only seed-order reference (the pooled
# label-store AP) legitimately differs from them in the POPULATION. These two
# keys may then differ, and only when the caller says so; the report records
# that it did.
POPULATION_KEYS = ("scored_row_set", "n_rows")


def check_stratum_comparability(experimental: dict, reference: dict,
                                exp_seeds: list[str],
                                ref_seeds: list[str],
                                population_may_differ: bool = False) -> dict:
    """AV10d. The two artifacts must differ ONLY in evidence masking.

    Two halves, and both are needed:
      * every key in `COMPARABILITY_KEYS` AGREES (absent counts as failure);
      * every key present in both that DISAGREES is in
        `COMPARABILITY_EXPECTED_DIFFERENCES`.

    The second half is what makes this a closed check rather than a list of
    things someone remembered to compare: an unexpected difference in a key
    nobody enumerated is reported, instead of passing because no clause named
    it. That is gotcha 25(c)'s coverage lesson -- proving a key is accounted
    for is not proving its accounting is correct.
    """
    keys = (tuple(k for k in COMPARABILITY_KEYS if k not in POPULATION_KEYS)
            if population_may_differ else COMPARABILITY_KEYS)
    expected = (COMPARABILITY_EXPECTED_DIFFERENCES
                + (POPULATION_KEYS if population_may_differ else ()))
    shared = compare_identities([experimental, reference],
                                ["experimental", "reference"], keys)
    unexpected = []
    for k in set(experimental) & set(reference):
        if k in expected:
            continue
        if _hashable(experimental[k]) != _hashable(reference[k]):
            unexpected.append({"key": k,
                               "experimental": _plain(experimental[k]),
                               "reference": _plain(reference[k])})
    # Provenance keys legitimately differ per run and are not experiment
    # identity. Exempted by FULL NAME, and the exemption is stated rather than
    # implied -- gotcha 25(a) was a bit-identity criterion on `git_sha` that
    # could only ever fail.
    provenance = {"git_sha", "git_dirty", "attr_schema_version",
                  "labels", "resliced_from", "reslice_note",
                  "identity_correction_library_sha", "created", "shards",
                  "merged_by", "n_shards", "seed_labels"}
    unexpected = [u for u in unexpected if u["key"] not in provenance]
    seeds = verify_positional_seed_order(exp_seeds, ref_seeds)
    return {
        "shared_keys": shared,
        "unexpected_differences": unexpected,
        "expected_differences_declared": list(expected),
        "population_may_differ": bool(population_may_differ),
        "seed_axis": seeds,
        "provenance_keys_exempted": sorted(provenance),
        # DERIVED, never asserted (gotcha 20).
        "verdict": ("COMPARABLE"
                    if shared["agree"] and not unexpected
                    and seeds["verdict"] == "POSITIONS_OK"
                    else "NOT_COMPARABLE"),
    }


def concept_axes_identical(concept_lists: list[list[str]]) -> dict:
    """All shards must carry the SAME concept list, in the same order.

    The concept set is a function of the label store and the row subset, never
    of the checkpoint, so a disagreement means a shard was built against a
    different store -- which no shape check would show, because the shapes would
    still be consistent with each other.
    """
    first = list(concept_lists[0])
    mismatched = [i for i, c in enumerate(concept_lists) if list(c) != first]
    return {"verdict": "IDENTICAL" if not mismatched else "DIFFER",
            "n_shards": len(concept_lists), "n_concepts": len(first),
            "mismatched_shard_indices": mismatched}


def library_counts(labels: Path) -> dict:
    """The LIBRARY's concept count and argmax-pool size, from the store itself."""
    with h5py.File(labels, "r") as hf:
        concepts = sorted(hf["concepts"].keys())
        declared = hf.attrs.get("n_concepts")
    return {
        "library_concepts": concepts,
        "n_concepts_library": len(concepts),
        "n_concepts_declared_by_store": (int(declared) if declared is not None
                                         else None),
        "argmax_pool_n_library": int(sum(argmax_pool_mask(concepts))),
    }


def check_library_scoped_identity(shard_identity: dict, labels: Path,
                                  scored: list[str]) -> dict:
    """VERIFY (never correct) that the shard's own `n_concepts` /
    `argmax_pool_n` already describe the LIBRARY, and that its
    `n_concepts_scored` / `argmax_pool_scored_n` describe the scored stratum
    (amendment AW / gotcha 16).

    **This used to be a relabel-on-write.** `feature_ap.py` wrote
    `n_concepts = len(names[stratum])` -- the count of concepts with positives
    *in the scored stratum* -- under the canonical LIBRARY-scoped key. On a
    pooled run those two meanings coincide, which is why the bug was invisible
    until the experimental stratum (227 of 712 concepts) diverged from it.
    That divergence is now fixed AT THE WRITER, so this function's job changed
    from correcting the attrs to CHECKING them: a shard that predates the fix,
    or a writer that regresses, must be caught here rather than silently
    merged under a canonical key it does not actually satisfy (gotcha 13
    extension 4 -- an operation that can silently do nothing is not a
    verified operation; the old relabel-and-report was exactly that shape one
    level up, since a caller who forgot `--labels` got no relabelling and no
    warning either).
    """
    lib = library_counts(labels)
    report = {
        "n_concepts_library": lib["n_concepts_library"],
        "argmax_pool_n_library": lib["argmax_pool_n_library"],
        "n_concepts_as_written": _plain(shard_identity.get("n_concepts")),
        "argmax_pool_n_as_written": _plain(shard_identity.get("argmax_pool_n")),
        "n_concepts_scored_as_written": _plain(
            shard_identity.get("n_concepts_scored")),
        "argmax_pool_scored_n_as_written": _plain(
            shard_identity.get("argmax_pool_scored_n")),
        "scored_concepts_are_a_subset_of_the_library": bool(
            set(scored) <= set(lib["library_concepts"])),
    }
    if not report["scored_concepts_are_a_subset_of_the_library"]:
        raise SystemExit(
            "FATAL: the shards score concepts absent from the label "
            "store's library -- the shards were built against a different "
            "store than the one passed as --labels.\n"
            + json.dumps(report, indent=2, default=str))
    if shard_identity.get("n_concepts_scored") is None:
        raise SystemExit(
            "FATAL: shard carries no `n_concepts_scored` attr -- it predates "
            "the AW writer fix, so its `n_concepts`/`argmax_pool_n` are "
            "STRATUM-SCOPED rather than library-scoped and merging it under "
            "the canonical keys would silently misdescribe it. Re-run "
            "feature_ap.py against the current writer.\n"
            + json.dumps(report, indent=2, default=str))
    mismatches = {
        k: (report[f"{k}_library"], report[f"{k}_as_written"])
        for k in ("n_concepts", "argmax_pool_n")
        if report[f"{k}_library"] != report[f"{k}_as_written"]
    }
    if mismatches:
        raise SystemExit(
            "FATAL: the shard's own n_concepts/argmax_pool_n disagree with "
            "the label store's library, even though the shard carries "
            "`n_concepts_scored` (post-AW-fix format) -- the writer has "
            "regressed.\n" + json.dumps(mismatches, indent=2, default=str))
    return report


def per_concept_vectors_identical(vectors: list[np.ndarray],
                                  name: str) -> dict:
    """All shards must carry BIT-IDENTICAL per-concept vectors.

    `prevalence`, `n_positives` and `n_rows_effective` are deterministic
    functions of (label store, row subset, negative convention) and contain no
    dependence on the checkpoint at all — so a disagreement means two shards
    were built against different inputs. It is checked **positionally and by
    VALUE**, not by length: the merged artifact copies these vectors from ONE
    shard, and `G = (AP - p)/(1 - p)` is scaled directly by `prevalence`, so a
    silent disagreement would rescale every gain computed downstream while
    every count, name and shape still agreed (gotcha 27 — for a mapping or
    value defect only a positional check discriminates).
    """
    first = np.asarray(vectors[0])
    bad = []
    for i, v in enumerate(vectors[1:], start=1):
        v = np.asarray(v)
        if v.shape != first.shape:
            bad.append({"shard_index": i, "reason": "shape",
                        "shape": list(v.shape)})
        elif not np.array_equal(v, first):
            d = np.flatnonzero(v != first)
            bad.append({"shard_index": i, "reason": "value",
                        "n_differing": int(d.size),
                        "first_differing_position": int(d[0]),
                        "values": [float(first[d[0]]), float(v[d[0]])]})
    return {"dataset": name, "n_shards": len(vectors),
            "verdict": "IDENTICAL" if not bad else "DIFFER",
            "disagreements": bad}


def merge_shards(shard_paths: list[Path], reference: Path, stratum: str,
                 out: Path, labels: Path | None = None,
                 population_may_differ: bool = False) -> dict:
    """Read N single-seed AP shards, order them by the reference's seed labels,
    and write one `(n_seeds, dict_size, n_concepts)` artifact."""
    PER_CONCEPT = ("prevalence", "n_positives", "n_rows_effective",
                   "in_argmax_pool")
    identities, seeds, concepts, report_paths = [], [], [], []
    vectors: dict[str, list] = {k: [] for k in PER_CONCEPT}
    for p in shard_paths:
        identities.append(read_attrs(p))
        report_paths.append(str(p))
        with h5py.File(p, "r") as hf:
            lab = normalise_seed_labels(hf["seeds"][:])
            if len(lab) != 1:
                raise ValueError(
                    f"{p} carries {len(lab)} seeds; this merge expects one "
                    "seed per shard so that a missing shard is a missing "
                    "LABEL rather than a silently short axis.")
            seeds.append(lab[0])
            concepts.append([c.decode() if isinstance(c, bytes) else c
                             for c in hf[stratum]["concepts"][:]])
            for k in PER_CONCEPT:
                if k in hf[stratum]:
                    vectors[k].append(hf[stratum][k][:])

    vector_checks = {}
    for k in PER_CONCEPT:
        if not vectors[k]:
            vector_checks[k] = {"dataset": k, "verdict": "ABSENT_FROM_ALL"}
            continue
        if len(vectors[k]) != len(shard_paths):
            # ABSENT from SOME is never AGREEMENT (I3).
            vector_checks[k] = {"dataset": k, "verdict": "ABSENT_FROM_SOME",
                                "n_present": len(vectors[k]),
                                "n_shards": len(shard_paths)}
            continue
        vector_checks[k] = per_concept_vectors_identical(vectors[k], k)
    bad_vectors = {k: v for k, v in vector_checks.items()
                   if v["verdict"] not in ("IDENTICAL", "ABSENT_FROM_ALL")}
    if bad_vectors:
        raise SystemExit(
            "FATAL: shards disagree on a per-concept vector. The merged "
            "artifact copies these from ONE shard, and `prevalence` scales "
            "every normalised gain downstream.\n"
            + json.dumps(bad_vectors, indent=2, default=str))

    shard_agreement = compare_identities(identities, seeds, SHARD_AGREEMENT_KEYS)
    if not shard_agreement["agree"]:
        raise SystemExit(
            "FATAL: the shards do not describe one experiment.\n"
            + json.dumps(shard_agreement, indent=2, default=str))

    axes = concept_axes_identical(concepts)
    if axes["verdict"] != "IDENTICAL":
        raise SystemExit("FATAL: shards carry different concept axes.\n"
                         + json.dumps(axes, indent=2))

    library_check = None
    if labels is not None:
        scored = list(concepts[0])
        library_check = check_library_scoped_identity(
            identities[0], labels, scored)

    ref_attrs = read_attrs(reference)
    with h5py.File(reference, "r") as hf:
        ref_seeds = normalise_seed_labels(hf["seeds"][:])

    order = resolve_shard_order(seeds, ref_seeds)
    ordered_paths = [shard_paths[i] for i in order]
    ordered_seeds = [seeds[i] for i in order]

    positional = verify_positional_seed_order(ordered_seeds, ref_seeds)
    if positional["verdict"] != "POSITIONS_OK":
        raise SystemExit("FATAL: merged seed axis is not the reference axis.\n"
                         + json.dumps(positional, indent=2))

    with h5py.File(ordered_paths[0], "r") as hf:
        n_feat, n_con = hf[stratum]["ap"].shape[1:]
    ap = np.zeros((len(ordered_paths), n_feat, n_con), dtype=np.float32)
    fire = np.zeros((len(ordered_paths), n_feat), dtype=np.int64)
    for i, p in enumerate(ordered_paths):
        with h5py.File(p, "r") as hf:
            ap[i] = hf[stratum]["ap"][0]
            fire[i] = hf["fire_counts"][0]

    with h5py.File(ordered_paths[0], "r") as src, h5py.File(out, "w") as dst:
        dst.create_dataset("fire_counts", data=fire, compression="gzip")
        dst.create_dataset("seeds", data=np.array(
            [f"seed_{s}" for s in ordered_seeds], dtype=h5py.string_dtype()))
        g = dst.create_group(stratum)
        g.create_dataset("ap", data=ap, compression="gzip")
        for name in ("concepts", "prevalence", "n_rows_effective",
                     "n_positives", "in_argmax_pool"):
            if name in src[stratum]:
                g.create_dataset(name, data=src[stratum][name][:])
        # Copies n_concepts/n_concepts_scored/argmax_pool_n/
        # argmax_pool_scored_n straight from the shard's own attrs -- correct
        # as-is now that feature_ap.py writes them library-scoped /
        # stratum-scoped at the source (amendment AW). No override: a
        # disagreement is caught above, by `check_library_scoped_identity`,
        # never patched here.
        for k, v in identities[order[0]].items():
            dst.attrs[k] = v
        dst.attrs["n_shards"] = len(ordered_paths)
        dst.attrs["merged_by"] = "merge_ap_shards/AV10d"
        dst.attrs["shards"] = json.dumps([str(p) for p in ordered_paths])

    merged_identity = read_attrs(out)
    comparability = check_stratum_comparability(
        merged_identity, ref_attrs, ordered_seeds, ref_seeds,
        population_may_differ=population_may_differ)
    return {
        "library_scoped_identity_check": library_check,
        "out": str(out), "stratum": stratum,
        "n_shards": len(ordered_paths), "shape": list(ap.shape),
        "seed_order": ordered_seeds,
        "reference": str(reference),
        "shard_identity_agreement": shard_agreement,
        "per_concept_vectors": vector_checks,
        "concept_axis": axes,
        "merged_seed_axis_is_positionally_the_reference": positional,
        "stratum_comparability_vs_reference": comparability,
        "ap_finite": bool(np.isfinite(ap).all()),
        "ap_in_unit_interval": bool((ap >= 0).all() and (ap <= 1).all()),
        "max_ap": float(ap.max()),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", type=Path, nargs="+", required=True)
    p.add_argument("--reference", type=Path, required=True,
                   help="the artifact whose seed order the merged axis adopts, "
                        "and against which comparability is asserted")
    p.add_argument("--stratum", default="experimental")
    p.add_argument("--labels", type=Path, default=None,
                   help="the label store, so the LIBRARY's n_concepts and "
                        "argmax_pool_n can be written under the canonical keys "
                        "(gotcha 16) instead of the shards' stratum-scoped ones")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--population-may-differ", action="store_true",
                   help="amendment CT §CT11.7-R: the reference is used for its "
                        "SEED ORDER and its identity, and may differ from the "
                        "shards in scored_row_set and n_rows (A1's shared set "
                        "has no reference on its own population). Recorded in "
                        "the report; every other key is still compared.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rep = merge_shards(args.shards, args.reference, args.stratum, args.out,
                       labels=args.labels,
                       population_may_differ=args.population_may_differ)
    print(json.dumps(rep, indent=2, default=str))
    if args.report:
        args.report.write_text(json.dumps(rep, indent=2, default=str))
    verdict = rep["stratum_comparability_vs_reference"]["verdict"]
    if verdict != "COMPARABLE":
        # A validator must fail DETECTABLY TO ITS CONSUMER (gotcha 13
        # extension 1): printing NOT_COMPARABLE and exiting 0 would let an
        # `afterok` chain read a stratum mismatch as success.
        print(f"FATAL: stratum comparability verdict is {verdict}")
        return 1
    print(f"wrote {args.out}  ({verdict})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
