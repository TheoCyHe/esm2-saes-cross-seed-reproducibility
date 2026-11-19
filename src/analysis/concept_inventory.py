"""AY §4 -- THE CONCEPT INVENTORY, for external biological review.

The author circulates this to colleagues to check that the concept vocabulary is
biologically sensible, so it must be readable by someone with no knowledge of
this repository. `docs/CONCEPT_INVENTORY.csv` is the deliverable and
`docs/CONCEPT_INVENTORY.md` is its data dictionary.

ONE ROW PER CONCEPT IN THE FULL LIBRARY -- all 712, not only the 703 screening
pool -- with `in_screening_pool` distinguishing them. The row count is ASSERTED
against the library size and the build fails if it disagrees: a silently short
inventory would be reviewed as though it were complete.

NO RESULTS COLUMNS. No detection rates, no reproducibility figures, no average
precision. The ask is a check on the VOCABULARY, and including results invites
comment on the results instead.

=============================================================================
THE ROW-TO-PROTEIN MAP IS THE DANGEROUS PART (gotcha 27)
=============================================================================
`n_proteins` needs each concept's annotated rows attributed to proteins, and
that is exactly the mapping `feature_footprint.protein_of_row` got wrong for
three sessions: `protein_sets/population_start` is stored in
`sorted(bare accession)` order, NOT in ascending cache-row order. On the real
store it has **25,508 descending steps**. Building a block map from it as
stored attributed 489 of 15,972,325 rows correctly -- 0.0031% -- while every
total still agreed, because a total cannot see a permutation.

So the starts are ARGSORTED into row order before use, and the result is
checked POSITIONALLY by an identity a permutation cannot satisfy: **every
annotated instance must lie entirely within a single protein.** An instance is
a contiguous residue span inside one protein by construction, so a wrong map
splits instances across protein boundaries wholesale. `assert_instances_nested`
refuses below a total match, and it is the only check here that can distinguish
the two orderings -- neither a total nor a rank can.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from src.analysis.artifact_identity import pooled_parents
from src.analysis.av_counts import EXPERIMENTAL_TIERS
from src.analysis.concept_bridge import concept_descriptor, concept_feature_key

COLUMNS = [
    "concept_id", "feature_key", "descriptor", "in_screening_pool",
    "excluded_reason", "n_annotated_residues", "n_proteins",
    "n_experimental_residues", "has_experimental_evidence",
    "scored_experimental_stratum",
]

POOLED_PARENT_REASON = (
    "bare feature key retained as a diagnostic arm but excluded from the "
    "screening pool: it pools every sub-type of the same key, so leaving it "
    "in a maximum over concepts would let the broad parent win over the "
    "specific child it contains"
)


# ------------------------------------------------------- the row/protein map

def protein_block_map(population_start: np.ndarray,
                      population_n_rows: np.ndarray) -> dict:
    """Protein blocks in ASCENDING ROW ORDER, with the permutation reported.

    Returns the argsorted starts and row counts. The stored arrays are in
    accession order; using them as stored is the AO6 defect.
    """
    start = np.asarray(population_start, dtype=np.int64)
    nrow = np.asarray(population_n_rows, dtype=np.int64)
    if start.shape != nrow.shape:
        raise ValueError(f"start {start.shape} vs n_rows {nrow.shape}")
    order = np.argsort(start, kind="mergesort")
    return {
        "order": order,
        "start": start[order],
        "n_rows": nrow[order],
        "stored_is_already_ascending": bool((np.diff(start) > 0).all()),
        "n_descending_steps_in_stored_order": int((np.diff(start) < 0).sum()),
    }


def protein_of_row(rows: np.ndarray, block: dict) -> np.ndarray:
    """Protein index (in ascending-row order) for each row.

    `searchsorted(..., 'right') - 1` on ascending starts. A row past the last
    block or before the first would give -1 or an out-of-range index; the
    nesting check below is what catches that rather than a silent clip.
    """
    return np.searchsorted(block["start"], np.asarray(rows, dtype=np.int64),
                           side="right") - 1


def assert_instances_nested(labels_path: Path, block: dict,
                            concepts: list[str]) -> dict:
    """Every instance must lie entirely inside ONE protein.

    THE CHECK THAT CAN TELL THE TWO ORDERINGS APART. An instance is a
    contiguous span within a single protein by construction, so under a
    permuted block map instances straddle boundaries wholesale. A total cannot
    see that and neither can a rank (gotcha 27), which is why the criterion is
    "the first and last row of every instance land in the same protein, and
    that protein's block contains them both".

    Refuses on any failure rather than reporting a percentage: this map feeds a
    column an external reviewer will read as fact.
    """
    n_inst = n_ok = 0
    offenders: list[str] = []
    with h5py.File(labels_path, "r") as hf:
        if "instances" not in hf:
            raise SystemExit("FATAL: the label store has no `instances` group; "
                             "n_proteins cannot be derived")
        grp = hf["instances"]
        for c in concepts:
            if c not in grp:
                continue
            s = grp[c]["start_row"][:]
            e = grp[c]["end_row"][:]
            if s.size == 0:
                continue
            pa = protein_of_row(s, block)
            pb = protein_of_row(np.maximum(e - 1, s), block)
            within = ((pa >= 0) & (pa == pb)
                      & (s >= block["start"][np.clip(pa, 0, None)])
                      & (np.maximum(e - 1, s)
                         < block["start"][np.clip(pa, 0, None)]
                         + block["n_rows"][np.clip(pa, 0, None)]))
            n_inst += int(s.size)
            n_ok += int(within.sum())
            if not within.all() and len(offenders) < 5:
                offenders.append(f"{c}: {int((~within).sum())} of {s.size}")
    report = {
        "n_instances": n_inst,
        "n_instances_nested_in_one_protein": n_ok,
        "share": float(n_ok / n_inst) if n_inst else None,
        "first_offenders": offenders,
        "verdict": "ALL_NESTED" if (n_inst and n_ok == n_inst)
                   else "NO_INSTANCES" if not n_inst else "STRADDLES",
    }
    if report["verdict"] == "STRADDLES":
        raise SystemExit(
            f"FATAL: {n_inst - n_ok:,} of {n_inst:,} annotated instances "
            f"straddle a protein boundary. An instance is a span inside one "
            f"protein by construction, so the row-to-protein map is wrong -- "
            f"this is the AO6 ordering defect. Offenders: {offenders}")
    return report


def proteins_per_concept(labels_path: Path, block: dict,
                         concepts: list[str]) -> dict[str, int | None]:
    """Distinct proteins carrying at least one annotated residue per concept.

    A concept with NO `instances` group returns None, not 0. "No instance
    record in this store" and "annotated in zero proteins" are different facts,
    and an external reviewer reading a 0 would take the second.
    """
    out: dict[str, int | None] = {}
    with h5py.File(labels_path, "r") as hf:
        grp = hf["instances"]
        for c in concepts:
            if c not in grp:
                out[c] = None
                continue
            s = grp[c]["start_row"][:]
            out[c] = int(np.unique(protein_of_row(s, block)).size) if s.size \
                else 0
    return out


# ------------------------------------------------------------- the rows

def build_rows(concepts: list[str], n_annotated: np.ndarray,
               n_proteins: dict[str, int | None],
               experimental: dict[str, int],
               scored_experimental: set[str],
               parents: set[str]) -> list[dict]:
    rows = []
    for i, c in enumerate(concepts):
        in_pool = c not in parents
        exp = int(experimental.get(c, 0))
        npr = n_proteins.get(c)
        rows.append({
            "concept_id": c,
            "feature_key": concept_feature_key(c),
            "descriptor": concept_descriptor(c),
            "in_screening_pool": "true" if in_pool else "false",
            "excluded_reason": "" if in_pool else POOLED_PARENT_REASON,
            "n_annotated_residues": int(n_annotated[i]),
            "n_proteins": "" if npr is None else npr,
            "n_experimental_residues": exp,
            "has_experimental_evidence": "true" if exp > 0 else "false",
            "scored_experimental_stratum":
                "true" if c in scored_experimental else "false",
        })
    return rows


def assert_row_count(rows: list[dict], expected: int) -> None:
    """A silently short inventory would be reviewed as though complete."""
    if len(rows) != expected:
        raise SystemExit(
            f"FATAL: built {len(rows)} rows against a library of {expected}. "
            "Refusing to write an inventory that does not cover the library.")


def write_csv(rows: list[dict], path: Path) -> None:
    """Byte-stable across platforms, which is not the csv module's default.

    `csv` writes `\\r\\n` regardless of platform, and git normalises to `\\n` on
    commit -- so a re-run of this build rewrote every one of the 713 lines and
    turned the checkout DIRTY, producing an artifact with `git_dirty: True`
    (AM11's defect, by a different route). `lineterminator="\\n"` makes the
    output identical to what git stores, so regenerating the file is a no-op
    when nothing has changed.

    `newline=""` is still required by csv's own contract; `utf-8-sig` so
    spreadsheets open the non-ASCII descriptors without mangling them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def summarise(rows: list[dict]) -> dict:
    keys: dict[str, int] = {}
    for r in rows:
        keys[r["feature_key"]] = keys.get(r["feature_key"], 0) + 1
    return {
        "n_rows": len(rows),
        "n_in_screening_pool": sum(r["in_screening_pool"] == "true"
                                   for r in rows),
        "n_excluded_pooled_parents": sum(r["in_screening_pool"] == "false"
                                         for r in rows),
        "n_with_experimental_evidence":
            sum(r["has_experimental_evidence"] == "true" for r in rows),
        "n_without_experimental_evidence":
            sum(r["has_experimental_evidence"] == "false" for r in rows),
        "n_scored_on_experimental_stratum":
            sum(r["scored_experimental_stratum"] == "true" for r in rows),
        "n_sub_typed": sum(bool(r["descriptor"]) for r in rows),
        "n_without_protein_count": sum(r["n_proteins"] == "" for r in rows),
        "feature_keys": dict(sorted(keys.items(), key=lambda kv: -kv[1])),
    }


# ------------------------------------------------------------------ driver

def _decode(a) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else x for x in a]


def run(ap_path: Path, labels_path: Path, experimental_ap: Path,
        out_csv: Path, out_json: Path, stratum: str = "pooled") -> dict:
    with h5py.File(ap_path, "r") as hf:
        concepts = _decode(hf[f"{stratum}/concepts"][:])
        n_annotated = hf[f"{stratum}/n_positives"][:]
        n_concepts_attr = int(hf.attrs["n_concepts"])
        argmax_pool_n = int(hf.attrs["argmax_pool_n"])
        library_sha = hf.attrs.get("library_sha")
        library_sha = (library_sha.decode() if isinstance(library_sha, bytes)
                       else library_sha)

    # The library size is taken from the artifact's own identity attr and the
    # concept axis must agree with it -- one key, one meaning (gotcha 16).
    if len(concepts) != n_concepts_attr:
        raise SystemExit(
            f"FATAL: the concept axis has {len(concepts)} entries but "
            f"`n_concepts` says {n_concepts_attr}. `n_concepts` is ALWAYS the "
            "library; refusing to guess which is right.")

    parents = set(pooled_parents(concepts))
    if len(concepts) - len(parents) != argmax_pool_n:
        raise SystemExit(
            f"FATAL: {len(concepts)} concepts minus {len(parents)} pooled "
            f"parents is {len(concepts) - len(parents)}, but `argmax_pool_n` "
            f"says {argmax_pool_n}.")

    with h5py.File(labels_path, "r") as hf:
        block = protein_block_map(hf["protein_sets/population_start"][:],
                                  hf["protein_sets/population_n_rows"][:])
    nesting = assert_instances_nested(labels_path, block, concepts)
    n_proteins = proteins_per_concept(labels_path, block, concepts)

    experimental: dict[str, int] = {}
    with h5py.File(labels_path, "r") as hf:
        grp = hf["concepts"]
        for c in concepts:
            if c not in grp:
                experimental[c] = 0
                continue
            arrs = [grp[c][t][:] for t in EXPERIMENTAL_TIERS if t in grp[c]]
            experimental[c] = (int(np.unique(np.concatenate(arrs)).size)
                               if arrs else 0)

    with h5py.File(experimental_ap, "r") as hf:
        scored = set(_decode(hf["experimental/concepts"][:]))

    rows = build_rows(concepts, n_annotated, n_proteins, experimental,
                      scored, parents)
    assert_row_count(rows, n_concepts_attr)
    write_csv(rows, out_csv)

    from src.sae.train import git_dirty, git_sha

    report = {
        "specification": "PREREGISTRATION amendment AY section 4",
        # This artifact's OWN provenance, not only its inputs'.
        "identity": {"git_sha": git_sha(), "git_dirty": git_dirty(),
                     "measurement": "concept_inventory_AY"},
        "csv": str(out_csv),
        "library_sha": library_sha,
        "n_concepts_library": n_concepts_attr,
        "argmax_pool_n": argmax_pool_n,
        "excluded_pooled_parents": sorted(parents),
        "row_to_protein_map": {
            "stored_is_already_ascending":
                block["stored_is_already_ascending"],
            "n_descending_steps_in_stored_order":
                block["n_descending_steps_in_stored_order"],
            "nesting_check": nesting,
        },
        "summary": summarise(rows),
        "source": {"ap": str(ap_path), "labels": str(labels_path),
                   "experimental_ap": str(experimental_ap)},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ap", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--experimental-ap", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args()

    r = run(args.ap, args.labels, args.experimental_ap,
            args.out_csv, args.out_json)
    s = r["summary"]
    m = r["row_to_protein_map"]
    print(f"\nrow-to-protein map: stored ascending = "
          f"{m['stored_is_already_ascending']}, "
          f"{m['n_descending_steps_in_stored_order']:,} descending steps")
    print(f"nesting check: {m['nesting_check']['verdict']} "
          f"({m['nesting_check']['n_instances_nested_in_one_protein']:,}"
          f"/{m['nesting_check']['n_instances']:,})")
    print(f"\n{s['n_rows']} rows = {s['n_in_screening_pool']} in the screening "
          f"pool + {s['n_excluded_pooled_parents']} excluded pooled parents")
    print(f"  sub-typed: {s['n_sub_typed']}")
    print(f"  with experimental evidence: {s['n_with_experimental_evidence']}; "
          f"without: {s['n_without_experimental_evidence']}")
    print(f"  scored on the experimental stratum: "
          f"{s['n_scored_on_experimental_stratum']}")
    print(f"  without a protein count: {s['n_without_protein_count']}")
    print(f"  feature keys: {s['feature_keys']}")
    print(f"\nwrote {args.out_csv}\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
