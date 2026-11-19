"""AZ section 15 -- THE NEVER-DETECTED CONCEPTS, characterised for external
biological review.

AY9 established the fact: of the **43** concepts in the 703-concept screening
pool that NO seed detects, **11 carry experimentally-validated residues**. A
sparsely-annotated concept nobody detects is unsurprising; an experimentally
grounded one is **a genuine negative about what the model misses**, and a
negative about a miss is much harder to explain away than a null coefficient.

This module builds the characterisation that lets a domain expert judge them:
per concept, its identity, its annotated- and experimental-residue counts, the
number of distinct proteins and distinct mmseqs clusters it lives in, its
best-carrier shortfall against its own null, and the taxonomic composition of
the proteins carrying it. **The other 32 are reported in summary form too**, so
a reader can see what distinguishes the 11 without the 32 being implied
uninteresting.

=============================================================================
THE SHORTFALL IS THE MARGIN, READ THE OTHER WAY, AND IT IS NOT A FINDING
=============================================================================
`margin = observed gain / detection threshold`, so a never-detected concept has
`margin < 1` in EVERY seed BY DEFINITION -- that is what never-detected means.
The number that carries information is HOW FAR below: 0.99 is a concept that
nearly cleared and 0.01 is one that is nowhere near. **The between-group
separation on this statistic is definitional and is never reported as evidence**
(AY8.3).

=============================================================================
TAXONOMY COMES FROM THE FLAT FILE, AND THE ACCESSION JOIN IS THE HAZARD
=============================================================================
Two accession conventions exist in this repo and they do not match (gotcha 10):
the cache index stores the whole FASTA token `sp|P19918|DCMS_AFICR`, while the
Swiss-Prot flat file uses the bare `P19918`. Every join here goes through
`split_train_eval.header_to_acc`, and **the realised match rate is asserted**:
`build_residue_labels.py`'s first run matched 0 of 53,672 proteins for exactly
this reason and wrote an empty file while exiting 0.

Taxonomy is **not stored anywhere** -- `build_residue_labels.py` reads
`lineage` and aggregates it into composition summaries without persisting it
per protein -- so it is re-read from the flat file for the accessions of
interest ONLY, in one streaming pass with `iter_dat_records(keep=...)`.

=============================================================================
THE ROW-TO-PROTEIN MAP IS BUILT IN ROW ORDER, NOT IN ACCESSION ORDER
=============================================================================
The AO6 defect. `protein_sets/population_start` is stored in sorted-ACCESSION
order and has 25,508 descending steps on the real store, while row positions are
ascending; building a block map from it as stored attributed 0.0031% of rows
correctly while every TOTAL agreed. This module uses `null_pilot`'s
`build_population_index`, which sorts by original start, and then **asserts
positionally**: every protein block must begin at its own start. A permutation
has the same total and the same block multiset, so only a positional check can
see it (gotcha 27).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from src.analysis.concept_bridge import (
    concept_descriptor,
    concept_feature_key,
    detection_margin,
    experimental_counts_for,
)
from src.analysis.concept_repro import (
    REFERENCE_PERCENTILE,
    assert_labels_align,
    detection_matrix,
    detection_rate,
    null_thresholds,
    observed_best_ap,
    to_gain,
)
from src.analysis.feature_ap import load_labels, load_scored_row_set
from src.analysis.null_pilot import (
    build_population_index,
    build_row_to_protein,
    protein_starts_lengths,
)
from src.annotations.uniprot_sites import iter_dat_records
from src.data.split_train_eval import header_to_acc

MIN_ACCESSION_MATCH = 0.5


def verify_block_starts(pop_index: list[dict], starts: np.ndarray) -> dict:
    """Every protein's block must begin at its own recorded start.

    A POSITIONAL check, and the only kind that can see the AO6 ordering defect:
    a permuted block assignment has the same protein count, the same total row
    count and the same block-length multiset as the right one, so no total and
    no correlation discriminates.
    """
    want = np.array([e["start"] for e in pop_index], dtype=np.int64)
    ok = int((want == starts).sum())
    return {
        "n_proteins": int(want.size),
        "n_blocks_beginning_at_their_own_start": ok,
        "verdict": "POSITIONS_OK" if ok == want.size else "POSITIONS_WRONG",
    }


def proteins_of_concept(rows: np.ndarray, row2protein: np.ndarray) -> np.ndarray:
    """Distinct population-protein indices a concept's positive rows fall in."""
    return np.unique(row2protein[rows])


def taxonomy_counts(lineages: list[str], organisms: list[str],
                    top: int = 5) -> dict:
    """Composition of the proteins carrying one concept.

    Counts, not shares alone: a "100% Eukaryota" built on two proteins and one
    built on four hundred are different statements, and a share hides which.
    """
    lc, oc = Counter(lineages), Counter(organisms)
    n = len(lineages)
    return {
        "n_proteins_with_taxonomy": n,
        "lineage_counts": dict(lc.most_common()),
        "top_lineage": (lc.most_common(1)[0][0] if lc else None),
        "top_lineage_share": (lc.most_common(1)[0][1] / n if n else None),
        "top_organisms": dict(oc.most_common(top)),
        "n_distinct_organisms": len(oc),
    }


def group_summary(rows: list[dict]) -> dict:
    """The 32 without experimental evidence, and the 11 with, in one shape."""
    if not rows:
        return {"n": 0, "status": "EMPTY_GROUP"}
    res = np.array([r["n_annotated_residues"] for r in rows], dtype=float)
    prot = np.array([r["n_proteins"] for r in rows], dtype=float)
    marg = np.array([r["best_seed_margin"] for r in rows], dtype=float)
    keys = Counter(r["feature_key"] for r in rows)
    return {
        "n": len(rows),
        "annotated_residues": {"min": float(res.min()),
                               "median": float(np.median(res)),
                               "max": float(res.max())},
        "n_proteins": {"min": float(prot.min()),
                       "median": float(np.median(prot)),
                       "max": float(prot.max())},
        "best_seed_margin": {"min": float(np.nanmin(marg)),
                             "median": float(np.nanmedian(marg)),
                             "max": float(np.nanmax(marg))},
        "feature_key_composition": dict(keys.most_common()),
    }


def _decode(arr) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else x for x in arr]


def _attrs(hf) -> dict:
    out = {}
    for k, v in hf.attrs.items():
        if isinstance(v, bytes):
            v = v.decode()
        elif isinstance(v, np.generic):
            v = v.item()
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        out[k] = v
    return out


def run(ap_path: Path, null_path: Path, labels_path: Path, eval_cache: Path,
        dat_path: Path, clusters_tsv: Path | None, out_path: Path,
        scored_row_set: str = "label_store", stratum: str = "pooled") -> dict:
    with h5py.File(ap_path, "r") as hf:
        g = hf[stratum]
        ap_concepts = _decode(g["concepts"][:])
        ap_seeds = _decode(hf["seeds"][:])
        in_pool = g["in_argmax_pool"][:].astype(bool)
        ap = g["ap"][:]
        prevalence = g["prevalence"][:]
        n_positives = g["n_positives"][:]
        ap_identity = _attrs(hf)

    with h5py.File(null_path, "r") as hf:
        null_concepts = _decode(hf["concepts"][:])
        null_seeds = _decode(hf["seeds"][:])
        null_cmax = hf["null_cmax_b"][:]
        null_identity = _attrs(hf)

    assert_labels_align("seed labels (AP vs null)", ap_seeds, null_seeds)
    pool_concepts = [c for c, k in zip(ap_concepts, in_pool) if k]
    assert_labels_align("concept labels (AP pool vs null)",
                        pool_concepts, null_concepts)

    ap_pool = ap[:, :, in_pool]
    best_ap = observed_best_ap(ap_pool)
    best_gain = to_gain(best_ap, prevalence[in_pool])
    thr = null_thresholds(null_cmax, REFERENCE_PERCENTILE, "pooled")
    d = detection_rate(detection_matrix(best_gain, thr))
    margin = detection_margin(best_gain, thr)
    npos = n_positives[in_pool].astype(np.int64)

    never = np.flatnonzero(d == 0.0)
    names = [pool_concepts[i] for i in never]

    # ---- the population index, verified POSITIONALLY ----------------------
    with h5py.File(eval_cache, "r") as hf:
        raw = hf["index_json"][()]
        index = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    subset = load_scored_row_set(labels_path, scored_row_set)
    with h5py.File(labels_path, "r") as hf:
        n_rows_labels = int(hf.attrs["n_rows"])
    if subset is None:
        pop_index = index
        remap = None
    else:
        pop_index, remap = build_population_index(index, subset)
    row2protein = build_row_to_protein(pop_index, n_rows_labels)
    starts, lengths = protein_starts_lengths(pop_index)
    positional = verify_block_starts(pop_index, starts)
    if positional["verdict"] != "POSITIONS_OK":
        raise SystemExit(
            f"FATAL: {positional['n_blocks_beginning_at_their_own_start']} of "
            f"{positional['n_proteins']} protein blocks begin at their own "
            "start. Every count below would be attributed to the wrong "
            "proteins while every total agreed (AO6).")

    pooled_rows, _ = load_labels(labels_path, "pooled")
    experimental = experimental_counts_for(labels_path, pool_concepts)

    # ---- clusters per concept, if the tsv is available --------------------
    acc_of_protein = [header_to_acc(e["accession"]) for e in pop_index]
    cluster_of: dict[str, str] = {}
    if clusters_tsv is not None and Path(clusters_tsv).exists():
        with open(clusters_tsv) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    cluster_of[header_to_acc(parts[1])] = header_to_acc(parts[0])
        hit = sum(1 for a in acc_of_protein if a in cluster_of)
        cluster_match = hit / max(1, len(acc_of_protein))
        if cluster_match < MIN_ACCESSION_MATCH:
            raise SystemExit(
                f"FATAL: only {cluster_match:.1%} of population accessions "
                f"were found in {clusters_tsv}. Two accession conventions "
                "exist in this repo and do not match (gotcha 10); a low match "
                "rate means the join is wrong, not that clusters are missing.")
    else:
        cluster_match = None

    # ---- per-concept rows, proteins, accessions ---------------------------
    per_concept, wanted_accs = [], set()
    for i in never:
        c = pool_concepts[i]
        a = pooled_rows.get(c)
        if a is None:
            prot_idx = np.empty(0, dtype=np.int64)
        else:
            a = a.astype(np.int64)
            if remap is not None:
                a = a[a < len(remap)]
                m = remap[a]
                a = m[m >= 0]
            prot_idx = proteins_of_concept(a, row2protein) if a.size else \
                np.empty(0, dtype=np.int64)
        accs = [acc_of_protein[p] for p in prot_idx]
        wanted_accs.update(accs)
        clusters = ({cluster_of[x] for x in accs if x in cluster_of}
                    if cluster_of else set())
        per_concept.append({
            "concept": c,
            "feature_key": concept_feature_key(c),
            "descriptor": concept_descriptor(c),
            "n_annotated_residues": int(npos[i]),
            "n_experimental_residues": int(experimental.get(c, 0)),
            "n_proteins": int(prot_idx.size),
            "n_mmseqs_clusters": (int(len(clusters)) if cluster_of else None),
            "mean_A": float(best_ap[:, i].mean()),
            "max_A": float(best_ap[:, i].max()),
            "detection_threshold_gain": float(thr[i]),
            "best_seed_margin": float(np.nanmax(margin[:, i])),
            "worst_seed_margin": float(np.nanmin(margin[:, i])),
            "accessions": accs,
        })

    # ---- taxonomy, one streaming pass over the flat file ------------------
    tax: dict[str, tuple[str, str]] = {}
    if dat_path is not None and Path(dat_path).exists():
        for rec in iter_dat_records(Path(dat_path), keep=set(wanted_accs)):
            tax[rec.acc] = (rec.lineage or "unknown", rec.organism or "unknown")
        matched = len(tax) / max(1, len(wanted_accs))
        if wanted_accs and matched < MIN_ACCESSION_MATCH:
            raise SystemExit(
                f"FATAL: only {matched:.1%} of the {len(wanted_accs)} "
                f"accessions were found in {dat_path}. gotcha 10's accession "
                "convention mismatch produces exactly this, and an empty "
                "taxonomy table that exits 0 is the failure mode this check "
                "exists to stop.")
    else:
        matched = None

    for row in per_concept:
        accs = row.pop("accessions")
        lin = [tax[a][0] for a in accs if a in tax]
        org = [tax[a][1] for a in accs if a in tax]
        row["taxonomy"] = (taxonomy_counts(lin, org) if tax else
                           {"status": "NOT_AVAILABLE"})
    per_concept.sort(key=lambda r: (-r["n_experimental_residues"],
                                    -r["n_annotated_residues"]))

    with_exp = [r for r in per_concept if r["n_experimental_residues"] > 0]
    without = [r for r in per_concept if r["n_experimental_residues"] == 0]

    # The pooled taxonomic composition of the two groups, so "do the 11
    # concentrate" is answerable rather than left to a reader with 11 tables.
    def pooled_tax(rows):
        lc = Counter()
        for r in rows:
            for k, v in (r["taxonomy"].get("lineage_counts") or {}).items():
                lc[k] += v
        tot = sum(lc.values())
        return {"lineage_counts": dict(lc.most_common()),
                "n_protein_slots": tot,
                "top_share": (lc.most_common(1)[0][1] / tot if tot else None)}

    from src.sae.train import git_dirty, git_sha

    out = {
        "specification": "PREREGISTRATION amendment AZ section 15 "
                         "(brief section 5a)",
        "identity": {"git_sha": git_sha(), "git_dirty": git_dirty(),
                     "measurement": "never_detected_AZ"},
        "reference_percentile": REFERENCE_PERCENTILE,
        "positional_guard": positional,
        "joins": {
            "accession_convention": "cache tokens normalised through "
                                    "split_train_eval.header_to_acc "
                                    "(gotcha 10)",
            "n_accessions_sought": len(wanted_accs),
            "flat_file_match_rate": matched,
            "cluster_tsv_match_rate": cluster_match,
        },
        "n_never_detected": int(never.size),
        "n_with_experimental_evidence": len(with_exp),
        "group_summaries": {
            "with_experimental_evidence": group_summary(with_exp),
            "without_experimental_evidence": group_summary(without),
        },
        "pooled_taxonomy": {
            "with_experimental_evidence": pooled_tax(with_exp),
            "without_experimental_evidence": pooled_tax(without),
        },
        "margin_note": "a never-detected concept has margin < 1 in EVERY seed "
                       "BY DEFINITION; the informative quantity is the "
                       "best_seed_margin, i.e. how far the closest seed got. "
                       "The between-group separation on this statistic is "
                       "definitional and is not evidence (AY8.3).",
        "concepts": per_concept,
        "source": {"ap": str(ap_path), "null": str(null_path),
                   "labels": str(labels_path), "eval_cache": str(eval_cache),
                   "dat": str(dat_path),
                   "clusters_tsv": (str(clusters_tsv) if clusters_tsv else None),
                   "ap_identity": ap_identity, "null_identity": null_identity},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ap", type=Path, required=True)
    p.add_argument("--null", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--eval-cache", type=Path, required=True)
    p.add_argument("--dat", type=Path, required=True)
    p.add_argument("--clusters-tsv", type=Path, default=None)
    p.add_argument("--scored-row-set", default="label_store")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    out = run(args.ap, args.null, args.labels, args.eval_cache, args.dat,
              args.clusters_tsv, args.out, args.scored_row_set)

    print(f"\n[5a] {out['n_never_detected']} never-detected concepts; "
          f"{out['n_with_experimental_evidence']} carry experimental residues")
    print(f"     positional guard: {out['positional_guard']['verdict']} "
          f"({out['positional_guard']['n_blocks_beginning_at_their_own_start']}"
          f"/{out['positional_guard']['n_proteins']})")
    print(f"     flat-file accession match "
          f"{out['joins']['flat_file_match_rate']}")
    print(f"\n{'concept':<48}{'res':>7}{'exp':>6}{'prot':>6}{'clus':>6}"
          f"{'best margin':>13}  top lineage")
    for r in out["concepts"]:
        if r["n_experimental_residues"] == 0:
            continue
        print(f"{r['concept'][:47]:<48}{r['n_annotated_residues']:>7}"
              f"{r['n_experimental_residues']:>6}{r['n_proteins']:>6}"
              f"{str(r['n_mmseqs_clusters']):>6}{r['best_seed_margin']:>13.4f}"
              f"  {r['taxonomy'].get('top_lineage')}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
