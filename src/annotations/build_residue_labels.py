"""
Build residue-level concept labels aligned to the eval activation cache.

Output: for each concept, the sorted array of CACHE ROW INDICES that are
positive. Row-indexed (not protein-indexed) because that is exactly what the
per-feature AP computation consumes: feature activations are a vector over cache
rows, so the label must be a vector over the same rows.

Alignment is the whole job, and it is where this pipeline silently breaks:
  - UniProt positions are 1-based inclusive; cache rows are 0-based. Residue p
    of a protein whose block starts at `start` is row `start + (p - 1)`.
  - TRUNCATED proteins (longer than max_len) have fewer cached rows than
    residues. Annotations past the cached length are DROPPED and counted -- never
    wrapped or clamped, which would label the wrong residues.
  - Subsampled caches (train) store `residue_ids`, so row j is residue
    residue_ids[j], not residue j. Handled, though eval is never subsampled.
Each of these is asserted and reported, and the sequence length in the flat file
is cross-checked against the cache's own record for every protein.

Concepts with fewer than `annotations.min_positive_residues` positives are
dropped: AP on a handful of positives is noise, and keeping them would inflate
the multiple-comparison burden for nothing.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import yaml

from src.experiment_config import lookup

from src.analysis.artifact_identity import PARAMETER_VALUES
from src.annotations.uniprot_sites import (
    PAIRED_RESIDUE_TYPES,
    TIER_GROUPS,
    features_to_concept_occurrences,
    iter_dat_records,
    occurrences_to_concept_positions,
    residues_for_feature,
)
from src.data.split_train_eval import header_to_acc
from src.sae.train import git_dirty, git_sha

# Below this fraction of cache proteins matched in the flat file, the run is
# almost certainly an ID-convention or release mismatch rather than genuinely
# missing annotations -- and emitting an empty/partial label set silently would
# corrupt every downstream AP. Fail instead.
MIN_MATCH_FRACTION = 0.5


def sanitize_concept_name(name: str) -> str:
    """HDF5 group names cannot contain '/' -- it is the path separator, so
    `create_group("DOMAIN:Alpha/beta hydrolase")` silently creates NESTED
    groups instead of one concept, corrupting the concept axis. This is not a
    rare edge case: 'Alpha/beta hydrolase' is a common, real UniProt DOMAIN
    /note value, and sub-typing (amendment E) makes many more such names
    reachable than the handful of ligand names ever were. Replaced with the
    Unicode DIVISION SLASH (U+2215), visually near-identical but not the HDF5
    path separator. Applied once, consistently, everywhere a concept name is
    used as a file-structural identifier (dict keys here, h5 group names,
    the 'concepts' string arrays feature_ap.py writes) -- never reversed,
    since nothing downstream needs the original ASCII '/' back.
    """
    return name.replace("/", "∕")


def provenance_excluded(concepts, excluded_types) -> list[str]:
    """Concepts dropped by the PROVENANCE exclusion (amendment M2).

    Matches on the FT TYPE -- the part before ':' -- so a sub-typed child is
    excluded along with its parent type. `HELIX`, `STRAND`, `TURN` (PDB-derived:
    'absent' means NOT MEASURED, while AP scores unlabelled rows as negative)
    and `COMPBIAS` (zero experimental residues) are excluded by
    PREREGISTRATION section 1.

    This existed only as a config COMMENT until 2026-07-28, so all four sat in
    the 716-concept library and in the 707-concept argmax pool that defines
    `g_f` -- three of them among the ten broadest concepts in it. The exclusion
    list is passed in from config, never hardcoded here.
    """
    ex = set(excluded_types or ())
    return sorted(c for c in concepts if c.split(":", 1)[0] in ex)


def load_cluster_map(tsv: Path) -> dict[str, str]:
    """Bare accession -> mmseqs cluster representative, from the clustering TSV.

    Both columns are whole FASTA tokens (`sp|P19918|DCMS_AFICR`), so both are
    normalised through `header_to_acc` -- the two-accession-convention sharp
    edge (gotcha 10) that once matched 0 of 53,672 proteins while exiting 0.
    """
    m: dict[str, str] = {}
    with tsv.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            rep, mem = parts
            m[header_to_acc(mem)] = header_to_acc(rep)
    if not m:
        raise SystemExit(f"FATAL: no cluster assignments parsed from {tsv}")
    return m


def load_cache_index(cache_path: Path) -> tuple[list[dict], int]:
    """Read the cache's index table and total row count."""
    with h5py.File(cache_path, "r") as hf:
        index = json.loads(hf["index_json"][()].decode()
                           if isinstance(hf["index_json"][()], bytes)
                           else str(hf["index_json"][()]))
        n_rows = hf["activations"].shape[0]
    return index, n_rows


def map_residues_to_rows(
    entry: dict, positions_1based, n_dropped: dict
) -> list[int]:
    """1-based UniProt residue positions -> cache row indices for one protein.

    Positions past the cached length (truncation) are dropped and counted.
    """
    start = entry["start"]
    n_rows = entry["n_rows"]
    rid = entry.get("residue_ids")
    rows = []
    if rid is None:
        # unsubsampled: row j == residue j (0-based)
        for p in positions_1based:
            r0 = p - 1
            if 0 <= r0 < n_rows:
                rows.append(start + r0)
            else:
                n_dropped["truncated_or_out_of_range"] += 1
    else:
        lookup = {r: start + j for j, r in enumerate(rid)}
        for p in positions_1based:
            row = lookup.get(p - 1)
            if row is None:
                n_dropped["not_subsampled"] += 1
            else:
                rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dat", type=Path, required=True,
                   help="uniprot_sprot.dat(.gz) -- download on the login node.")
    p.add_argument("--cache", type=Path, required=True,
                   help="Eval activation cache the labels must align to.")
    p.add_argument("--out", type=Path, required=True, help="Output .h5 of labels.")
    p.add_argument("--no-ligand-subconcepts", action="store_true")
    p.add_argument("--no-note-subconcepts", action="store_true",
                   help="Disable /note sub-typing (amendment E). Off by default "
                        "-- sub-typing is now the main-test concept library.")
    p.add_argument("--retention-floor", default=None,
                   choices=["legacy", "interplm"],
                   help="Override annotations.retention_floor_mode from config. "
                        "'legacy' = residues >= min_positive_residues only (the "
                        "original rule). 'interplm' = residues > "
                        "retention_min_residues OR instances > "
                        "retention_min_instances (PREREGISTRATION amendment E, "
                        "matching InterPLM's disjunctive floor exactly).")
    p.add_argument("--truncation-policy", default="with_truncated",
                   choices=list(PARAMETER_VALUES["truncation_policy"]),
                   help="PREREGISTRATION amendment G2. 'with_truncated' (default): every "
                        "cache protein, disjunctive floor applied to UNTRUNCATED "
                        "counts (the amendment-E fix -- a concept's true size can "
                        "exceed its surviving/analysable size). 'untruncated': "
                        "proteins with entry['truncated']==True are EXCLUDED "
                        "ENTIRELY (not counted toward any concept, any floor "
                        "decision, or n_rows) -- population counts ARE surviving "
                        "counts by construction, so the true-size-vs-analysable-"
                        "size gap that let a single-protein, single-surviving-"
                        "instance concept (DOMAIN:Ferric oxidoreductase) clear "
                        "the floor cannot exist. 'untruncated' means <= 1,022 "
                        "residues -- ESM-2's max_len=1022 cap, so 'untruncated' "
                        "== 'protein fits in one pass'. Close to InterPLM's "
                        "population (<1024 residues) but not the same set: a "
                        "1,023-residue protein is in theirs and not ours "
                        "(corrections entry 269).")
    p.add_argument("--clusters", type=Path, default=None,
                   help="mmseqs cluster TSV (sprot_clu30_cluster.tsv). When "
                        "given, per-concept distinct CLUSTER counts are "
                        "recorded as a stratification axis (amendment N2) and "
                        "the cluster cost curve is printed. Omitted => "
                        "n_clusters is recorded as -1 (NOT 0).")
    p.add_argument("--min-protein-count", type=int, default=1,
                   help="PREREGISTRATION amendment G3. A floor-passing concept "
                        "is additionally required to have SURVIVING instances "
                        "in at least this many distinct proteins -- an "
                        "estimability requirement, not a truncation guard: "
                        "under Null A's joint circular shift, every instance "
                        "within one protein moves together, so the effective "
                        "number of independent null placements is the protein "
                        "count, not the instance count. Default 1 = the floor "
                        "alone (no additional gate). See the measured "
                        "protein-count cost curve printed below before "
                        "choosing a stricter value.")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    ann_cfg = cfg["annotations"]
    min_pos = ann_cfg["min_positive_residues"]                    # legacy floor
    floor_mode = args.retention_floor or ann_cfg.get("retention_floor_mode", "legacy")
    min_residues_interplm = ann_cfg.get("retention_min_residues", 1500)
    min_instances_interplm = ann_cfg.get("retention_min_instances", 10)
    # Amendment CR: the sub-typed feature types come from the config and from
    # nowhere else. Until CR an absent key fell back SILENTLY to a four-type
    # code default while the config listed eight; now absence is a refusal.
    # `--no-note-subconcepts` is the one explicit way to run with none.
    if args.no_note_subconcepts:
        subtype_note_types = frozenset()
    else:
        subtype_note_types = frozenset(
            lookup(cfg, "annotations.subtype_note_types", args.config))

    index, n_rows_cache = load_cache_index(args.cache)
    # The cache index stores the FULL FASTA id token (e.g. "sp|P19918|DCMS_AFICR")
    # because cache.py keys on the first whitespace token, while the flat file
    # uses the bare accession ("P19918"). Normalising through the same
    # header_to_acc the split uses is what makes the two sides comparable -- the
    # first run of this script matched 0/53,672 proteins for exactly this reason.
    by_acc = {header_to_acc(e["accession"]): e for e in index}
    print(f"cache: {len(index):,} proteins, {n_rows_cache:,} rows")

    # ---- N2: mmseqs cluster map (stratification axis, NOT a gate) --------
    cluster_of = None
    if args.clusters is not None:
        cluster_of = load_cluster_map(args.clusters)
        matched = sum(1 for a in by_acc if a in cluster_of)
        frac = matched / max(1, len(by_acc))
        print(f"clusters: {len(cluster_of):,} accessions mapped; "
              f"{matched:,}/{len(by_acc):,} cache proteins matched ({frac:.1%}), "
              f"{len(set(cluster_of.values())):,} distinct representatives")
        # Same hard-fail discipline as MIN_MATCH_FRACTION: a near-zero match
        # rate here means an accession-convention mismatch (gotcha 10), and a
        # silently-empty cluster axis would make every concept look like a
        # 0-cluster concept rather than raising.
        if frac < MIN_MATCH_FRACTION:
            raise SystemExit(
                f"FATAL: only {frac:.1%} of cache proteins found in "
                f"{args.clusters}. Expected near 100% -- the split partitions "
                "ALL of Swiss-Prot with n_unclustered_records=0. This is an "
                "accession-convention or release mismatch, not missing data.")
    n_trunc = sum(1 for e in index if e["truncated"])
    print(f"       {n_trunc:,} truncated proteins (their tail annotations are dropped)")

    # ---- G2: population filter -----------------------------------------
    # 'untruncated': truncated proteins are EXCLUDED ENTIRELY here, before any
    # protein is processed -- not just their tail residues. Population counts
    # then ARE surviving counts, by construction; the true-size-vs-analysable-
    # size gap that let DOMAIN:Ferric oxidoreductase (11 true instances, 1
    # surviving, in 1 protein) clear the floor cannot exist under this
    # population, because there is no "true size" distinct from what is here.
    if args.truncation_policy == "untruncated":
        excluded_accs = {a for a, e in by_acc.items() if e["truncated"]}
        by_acc = {a: e for a, e in by_acc.items() if not e["truncated"]}
        n_rows = sum(e["n_rows"] for e in by_acc.values())
        print(f"  truncation_policy=untruncated: excluded {len(excluded_accs):,} truncated "
              f"proteins entirely. {len(by_acc):,} proteins, {n_rows:,} rows remain "
              f"({n_rows/n_rows_cache:.1%} of the full cache's rows).")
    else:
        n_rows = n_rows_cache

    # concept -> evidence tier -> cache rows. Tiers are disjoint (most direct
    # evidence wins per residue), so they partition each concept's positives.
    concept_rows: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list))
    # concept -> instance count (PREREGISTRATION amendment E's disjunctive
    # floor) and concept -> [(start_row, end_row), ...] (instance BOUNDARIES,
    # for D8's domain-level recall). Instances are only meaningful for SPAN
    # concepts -- PAIRED_RESIDUE_TYPES (DISULFID/CROSSLNK) are excluded, since
    # a bonded pair's (min,max) row range is not the instance, it is a false
    # span across everything in between.
    concept_instance_count: dict[str, int] = defaultdict(int)
    concept_instances: dict[str, list[tuple[int, int]]] = defaultdict(list)
    # UNTRUNCATED counters -- P4 (measure_truncation_impact.py) found 60 of
    # 3,122 concepts flip retention-floor status depending on whether the
    # floor is measured on truncated (surviving) or untruncated counts. A
    # concept's TRUE size is a property of the annotation, not of where
    # ESM-2's 1022-residue cap happened to fall -- so the floor DECISION uses
    # these, while the actual DATA written (concept_rows/concept_instances
    # above) still only ever contains rows that exist in the cache. Same
    # "compute artifact must not determine which concepts exist" principle
    # as gotcha 9.
    untrunc_residue_count: dict[str, int] = defaultdict(int)
    untrunc_instance_count: dict[str, int] = defaultdict(int)
    # G3: distinct SURVIVING proteins per concept. NOT the same as instance
    # count -- a repeat-domain protein can contribute >10 instances from ONE
    # protein, which the disjunctive floor accepts but which gives a null with
    # essentially zero estimability (see the amendment-G protein gate).
    concept_proteins: dict[str, set[int]] = defaultdict(set)
    # N2: distinct mmseqs CLUSTERS per concept. The quantity that actually
    # governs estimability under joint shift is the number of INDEPENDENT
    # placements, and homologous proteins do not supply independent ones --
    # `BINDING:heme a3`'s 14 cytochrome c oxidase orthologs sit in 2 clusters,
    # so its effective observation count is ~2 while the G3 protein gate
    # credited it with 14. A protein count is not an independence count
    # (gotcha 17). Recorded as a STRATIFICATION axis; no gate is applied here.
    concept_clusters: dict[str, set[str]] = defaultdict(set)
    instance_dropped = defaultdict(int)
    dropped = defaultdict(int)
    seen, len_mismatch = 0, 0

    # Per-protein records for the STRATUM COMPOSITION report (see the shared-set
    # rationale in the manifest / PREREGISTRATION.md amendment A1).
    protein_meta: dict[str, dict] = {}

    occ_kwargs = dict(use_ligand_subconcepts=not args.no_ligand_subconcepts,
                      subtype_note_types=subtype_note_types)

    for rec in iter_dat_records(args.dat, keep=set(by_acc)):
        entry = by_acc.get(rec.acc)
        if entry is None:
            continue
        seen += 1
        # The flat file and the cache must describe the SAME protein. seq_len in
        # the cache is min(len, max_len), so compare against the untruncated
        # length only when the protein was not truncated.
        if not entry["truncated"] and rec.seq_len and rec.seq_len != entry["seq_len"]:
            len_mismatch += 1
            continue                      # wrong isoform/version -- do not label
        has_experimental = False
        # Sanitize concept names ONCE, here, before they become dict keys --
        # everything downstream (concept_rows, concept_instance_count, the h5
        # group names, feature_ap.py's "concepts" arrays) then sees the same
        # safe string consistently. ftype extraction (concept.split(":", 1)[0])
        # is unaffected: sanitization only ever touches characters after the
        # first ':', never the FT-type prefix itself.
        occurrences = {
            sanitize_concept_name(c): occs
            for c, occs in features_to_concept_occurrences(rec.feats, **occ_kwargs).items()
        }
        for concept, res_tiers in occurrences_to_concept_positions(occurrences).items():
            by_tier: dict[str, list[int]] = defaultdict(list)
            for residue, tier in res_tiers.items():
                by_tier[tier].append(residue)
            # UNTRUNCATED: res_tiers is built from the raw FT positions
            # (residues_for_feature), before map_residues_to_rows drops
            # anything past the cache's truncated length -- so len() here is
            # exactly this protein's contribution to the concept's TRUE size,
            # already deduplicated within-protein (res_tiers is a dict keyed
            # by residue). Different proteins' residue numbers never collide
            # in this sum (each protein's dict is independent), so summing
            # across proteins needs no further dedup.
            untrunc_residue_count[concept] += len(res_tiers)
            for tier, residues in by_tier.items():
                rows = map_residues_to_rows(entry, sorted(residues), dropped)
                concept_rows[concept][tier].extend(rows)
                if rows:
                    # G3: distinct SURVIVING proteins per concept -- the
                    # estimability requirement joint-shift needs (all instances
                    # within one protein move together under Null A, so the
                    # effective number of independent null placements is the
                    # protein count, not the instance count). entry["start"]
                    # is a stable per-protein identifier (unique per protein).
                    concept_proteins[concept].add(entry["start"])
                    if cluster_of is not None:
                        cl = cluster_of.get(header_to_acc(entry["accession"]))
                        if cl is not None:
                            concept_clusters[concept].add(cl)
                if rows and TIER_GROUPS.get(tier) == "experimental":
                    has_experimental = True
        for concept, occs in occurrences.items():
            ftype = concept.split(":", 1)[0]
            if ftype in PAIRED_RESIDUE_TYPES:
                continue
            untrunc_instance_count[concept] += len(occs)   # ALL occurrences,
                                                            # regardless of survival
            for o in occs:
                rows = map_residues_to_rows(
                    entry, residues_for_feature(ftype, o.start, o.end),
                    instance_dropped)
                # Count only instances that SURVIVE into the cached rows -- an
                # occurrence entirely past a protein's truncation point
                # contributes zero eval-set residues and must not count toward
                # the disjunctive floor's instance side, or a concept could
                # pass with instances>10 while having 0 actual residues (hit
                # in practice: crashed the h5 write on an empty per-tier dict).
                if rows:
                    concept_instance_count[concept] += 1
                    concept_instances[concept].append((min(rows), max(rows)))
        protein_meta[rec.acc] = {
            "start": entry["start"], "n_rows": entry["n_rows"],
            "seq_len": entry["seq_len"], "lineage": rec.lineage or "unknown",
            "organism": rec.organism or "unknown",
            "has_experimental": has_experimental,
            "truncated": entry["truncated"],
        }

    frac = seen / max(1, len(index))
    print(f"matched {seen:,}/{len(index):,} cache proteins in the flat file "
          f"({frac:.1%})")
    if frac < MIN_MATCH_FRACTION:
        raise SystemExit(
            f"FATAL: only {frac:.1%} of cache proteins were found in {args.dat} "
            f"(threshold {MIN_MATCH_FRACTION:.0%}). Writing labels now would "
            "silently produce empty/partial concepts and corrupt every AP "
            "downstream. Likely causes: accession-format mismatch between the "
            "cache index and the flat file, or a Swiss-Prot release skew "
            "between the FASTA the cache was built from and this .dat."
        )
    if len_mismatch:
        print(f"WARNING: {len_mismatch:,} proteins had a length mismatch and were "
              "skipped (flat file vs cache disagree -- different release?)")
    for k, v in dropped.items():
        print(f"dropped {v:,} annotated residues ({k})")

    # Deduplicate per tier, then resolve any row claimed by two tiers (possible
    # across different proteins' features mapping to the same row is impossible,
    # but a row can appear twice within a tier) -- np.unique handles the latter.
    #
    # ---- Retention floor (PREREGISTRATION amendment E) ---------------------
    # 'legacy': the original rule, residues >= min_positive_residues (100).
    # 'interplm': InterPLM's own disjunctive rule, residues > 1500 OR
    # instances > 10 -- protects a rare-but-well-defined family (few residues,
    # but each occurrence is a clean, real instance) that a residue-only floor
    # would drop, which matters once concepts are sub-typed by /note and a
    # specific domain family can be genuinely rare. Both counts are ALWAYS
    # computed and reported (`n_instances`, `passes_legacy`, `passes_interplm`
    # per concept in the manifest) regardless of which mode is active, so the
    # measured difference between the two is never silently lost.
    kept: dict[str, dict[str, np.ndarray]] = {}
    small = []
    floor_diagnostics: dict[str, dict] = {}
    for concept, tiers in concept_rows.items():
        per_tier = {t: np.unique(np.asarray(r, dtype=np.int64))
                    for t, r in tiers.items() if r}
        total = int(sum(a.size for a in per_tier.values()))
        n_instances = concept_instance_count.get(concept, 0)      # truncated/surviving
        res_untrunc = untrunc_residue_count.get(concept, total)
        inst_untrunc = untrunc_instance_count.get(concept, n_instances)
        passes_legacy = total >= min_pos
        # `total > 0` guards against a concept passing while having zero
        # SURVIVING residues -- an empty per_tier dict would otherwise reach
        # the h5 write and crash np.concatenate on an empty list. The floor's
        # SIZE criteria use the UNTRUNCATED counts (P4 measured 60 of 3,122
        # concepts flip status depending on which is used -- a concept's true
        # size should not depend on where the 1022-residue cap happened to
        # fall), but `total > 0` still requires that SOME of that size
        # actually survived into the cache; there is no data to write
        # otherwise, regardless of how large the concept truly is.
        passes_interplm = total > 0 and (res_untrunc > min_residues_interplm
                                         or inst_untrunc > min_instances_interplm)
        n_proteins_concept = len(concept_proteins.get(concept, ()))
        floor_diagnostics[concept] = {
            "n_residues": total, "n_instances": n_instances,
            "n_residues_untruncated": res_untrunc,
            "n_instances_untruncated": inst_untrunc,
            "truncation_residue_loss_frac":
                (res_untrunc - total) / res_untrunc if res_untrunc else 0.0,
            "n_proteins": n_proteins_concept,
            # N2 stratification axis. -1 means "not computed" (no --clusters
            # given) and is NEVER to be read as 0 -- absent is not a value.
            "n_clusters": (len(concept_clusters.get(concept, ()))
                           if cluster_of is not None else -1),
            "passes_legacy": passes_legacy, "passes_interplm": passes_interplm,
        }
        passes = passes_interplm if floor_mode == "interplm" else passes_legacy
        if passes:
            kept[concept] = per_tier
        else:
            small.append((concept, total))

    # Report the measured floor-choice impact regardless of which is active --
    # this is what makes "decide + document, don't silently change" auditable.
    gained = [c for c, d in floor_diagnostics.items()
             if d["passes_interplm"] and not d["passes_legacy"]]
    lost = [c for c, d in floor_diagnostics.items()
           if d["passes_legacy"] and not d["passes_interplm"]]
    print(f"\nRETENTION FLOOR comparison (mode active: {floor_mode}):")
    print(f"  legacy (residues>={min_pos}): {sum(d['passes_legacy'] for d in floor_diagnostics.values())} concepts")
    print(f"  interplm (residues>{min_residues_interplm} OR instances>{min_instances_interplm}): "
          f"{sum(d['passes_interplm'] for d in floor_diagnostics.values())} concepts")
    print(f"  interplm GAINS {len(gained)} concepts legacy would drop "
          f"(rare-family sub-concepts kept by instance count):")
    for c in sorted(gained, key=lambda c: -floor_diagnostics[c]["n_instances"])[:15]:
        d = floor_diagnostics[c]
        print(f"    {c:<32} residues={d['n_residues']:>6,}  instances={d['n_instances']:>5,}")
    print(f"  interplm DROPS {len(lost)} concepts legacy would keep "
          f"(passed on raw residue count with few real instances):")
    for c in sorted(lost, key=lambda c: -floor_diagnostics[c]["n_residues"])[:15]:
        d = floor_diagnostics[c]
        print(f"    {c:<32} residues={d['n_residues']:>6,}  instances={d['n_instances']:>5,}")

    # ---- G3: protein-count cost curve ----------------------------------
    # NOT a decision -- a measurement, to inform one. Under joint shift (Null
    # A), every instance within one protein moves by the SAME offset, so
    # instances in the same protein are perfectly correlated in the null; the
    # effective number of independent null placements a concept gets is its
    # SURVIVING PROTEIN COUNT, not its instance count. Report how many
    # floor-passing concepts survive at each candidate protein-count
    # threshold so the gate value is chosen from the actual shape of this
    # curve, not a guess.
    print(f"\nPROTEIN-COUNT COST CURVE (measured on the {len(kept)} concepts "
          f"passing the '{floor_mode}' floor):")
    for thresh in (1, 2, 3, 5, 10):
        n_pass = sum(1 for c in kept if floor_diagnostics[c]["n_proteins"] >= thresh)
        print(f"  proteins >= {thresh:>2}: {n_pass:>4} of {len(kept)} concepts "
              f"({n_pass/max(1,len(kept)):.1%})")
    single_protein = sorted(
        (c for c in kept if floor_diagnostics[c]["n_proteins"] <= 1),
        key=lambda c: -floor_diagnostics[c]["n_residues"])
    print(f"  {len(single_protein)} concepts confined to <=1 protein despite "
          f"passing the floor -- first 15 by residue count:")
    for c in single_protein[:15]:
        d = floor_diagnostics[c]
        print(f"    {c:<32} residues={d['n_residues']:>6,}  instances={d['n_instances']:>5,}"
              f"  proteins={d['n_proteins']}")

    # ---- G3: apply the protein-count gate (THE decision, from the curve
    # above) ------------------------------------------------------------
    # Not a truncation guard -- an ESTIMABILITY requirement. Under joint
    # shift (Null A) every instance within one protein moves together, so a
    # concept confined to few proteins gets a null with essentially no
    # independent placements to draw from, regardless of how many residues
    # or instances it has (a repeat-array protein natively produces >10
    # instances in 1 protein and would otherwise sail through the floor).
    for c in kept:
        floor_diagnostics[c]["passes_gate"] = (
            floor_diagnostics[c]["n_proteins"] >= args.min_protein_count)
    gated_out = [c for c in kept if not floor_diagnostics[c]["passes_gate"]]
    for c in gated_out:
        del kept[c]
    print(f"\nPROTEIN-COUNT GATE (amendment G3): min_protein_count="
          f"{args.min_protein_count} -- excluded {len(gated_out)} of "
          f"{len(kept) + len(gated_out)} floor-passing concepts "
          f"({len(kept)} remain).")

    # ---- M2: PROVENANCE EXCLUSION, enforced rather than documented -------
    # PREREGISTRATION section 1 excludes HELIX/STRAND/TURN (PDB-derived, so
    # 'absent' means NOT MEASURED while AP scores unlabelled rows as negative)
    # and COMPBIAS (zero experimental residues). Until 2026-07-28 that
    # exclusion lived ONLY in a config comment and this file did not apply it,
    # so all four sat in the library and in the argmax pool that defines g_f --
    # three of them among the ten broadest concepts in it. Sourced from config,
    # never hardcoded, per the "nothing experiment-defining in src/" rule.
    prov_excluded = list(
        cfg.get("annotations", {}).get("provenance_excluded_types", []))
    prov_dropped = provenance_excluded(kept, prov_excluded)
    for c in prov_dropped:
        del kept[c]
        floor_diagnostics[c]["provenance_excluded"] = True
    print(f"\nPROVENANCE EXCLUSION (amendment M2): types {prov_excluded} -- "
          f"dropped {len(prov_dropped)} concepts ({sorted(prov_dropped)}); "
          f"{len(kept)} remain.")

    # ---- N2: CLUSTER COST CURVE -- reported, NOT applied ----------------
    # No threshold is fixed here. Amendment N2 defers the gate decision to
    # this curve precisely so it is made with the numbers in hand; gating is
    # irreversible while stratifying is not.
    if cluster_of is not None:
        clusters_now = {c: floor_diagnostics[c]["n_clusters"] for c in kept}
        vals = sorted(clusters_now.values())
        n = len(vals)
        print(f"\nCLUSTER COST CURVE (amendment N2) -- REPORTED, NOT APPLIED.")
        print(f"  clusters per concept over the {n} compliant concepts: "
              f"min {vals[0]}, median {vals[n // 2]}, "
              f"mean {sum(vals) / n:.1f}, max {vals[-1]}")
        print(f"  {'>= clusters':>12} {'concepts':>9} {'share':>7}")
        for t in (1, 2, 3, 5, 10):
            k = sum(1 for v in vals if v >= t)
            print(f"  {t:>12} {k:>9} {k / n:>6.1%}")
        # By TYPE: a threshold that wipes out BINDING while sparing DOMAIN is
        # a materially different decision from a uniform loss.
        by_type: dict[str, list[int]] = defaultdict(list)
        for c, v in clusters_now.items():
            by_type[c.split(":", 1)[0]].append(v)
        print(f"\n  by concept type (types with >= 5 concepts), "
              f"share surviving each threshold:")
        print(f"  {'type':<16} {'n':>5} {'med':>5} " +
              " ".join(f"{'>=' + str(t):>7}" for t in (1, 2, 3, 5, 10)))
        for t_name, vs in sorted(by_type.items(),
                                 key=lambda kv: -len(kv[1])):
            if len(vs) < 5:
                continue
            vs_sorted = sorted(vs)
            row = " ".join(
                f"{sum(1 for v in vs if v >= t) / len(vs):>6.0%} "
                for t in (1, 2, 3, 5, 10))
            print(f"  {t_name:<16} {len(vs):>5} "
                  f"{vs_sorted[len(vs) // 2]:>5} {row}")
    if prov_excluded and not prov_dropped:
        print("  NOTE: exclusion list is non-empty but matched nothing. That is "
              "possible legitimately (a prior floor may have removed them "
              "already) but is worth confirming rather than assuming.")

    def group_counts(per_tier: dict[str, np.ndarray]) -> dict[str, int]:
        g = defaultdict(int)
        for tier, arr in per_tier.items():
            g[TIER_GROUPS.get(tier, "other")] += int(arr.size)
        return dict(g)

    totals = {c: int(sum(a.size for a in t.values())) for c, t in kept.items()}
    floor_desc = (f">={min_pos} residues" if floor_mode == "legacy" else
                 f">{min_residues_interplm} residues OR >{min_instances_interplm} instances")
    print(f"\n{len(kept)} concepts pass the '{floor_mode}' floor ({floor_desc}) "
          f"({len(small)} dropped as too rare)")

    # THE evidence-provenance report. A concept whose positives are mostly
    # propagated is not independent ground truth (see uniprot_sites docstring).
    print(f"\n{'concept':<30} {'positives':>10}  {'%rows':>7}  "
          f"{'exper':>7}  {'propag':>7}  {'other':>7}")
    for concept in sorted(kept, key=lambda c: -totals[c])[:30]:
        g = group_counts(kept[concept])
        tot = totals[concept]
        print(f"  {concept:<28} {tot:>10,}  {tot / n_rows:6.3%}  "
              f"{g.get('experimental', 0) / tot:6.1%}  "
              f"{g.get('propagated', 0) / tot:6.1%}  "
              f"{g.get('other', 0) / tot:6.1%}")

    all_groups = defaultdict(int)
    for c, t in kept.items():
        for k, v in group_counts(t).items():
            all_groups[k] += v
    gtot = sum(all_groups.values())
    print(f"\nOVERALL evidence composition across all kept concepts "
          f"({gtot:,} positive residue-labels):")
    for k in ("experimental", "propagated", "other"):
        print(f"  {k:<14} {all_groups.get(k, 0):>12,}  "
              f"{all_groups.get(k, 0) / max(1, gtot):6.1%}")

    # ---- SHARED PROTEIN SET (PREREGISTRATION amendment A1) -----------------
    # Experimental annotation is not randomly distributed: it concentrates in
    # heavily-studied, soluble/crystallisable, model-organism proteins. So the
    # experimental and propagated strata are NOT the same proteins with
    # different label quality -- they are different protein populations, with
    # different domain content, lengths and taxonomy. Comparing them directly
    # would confound evidence tier with protein composition, and a
    # propagated-only effect could not be attributed to homology circularity
    # rather than composition. Restricting BOTH strata to proteins carrying at
    # least one experimental label makes evidence tier the only thing varying.
    exp_prots = {a: m for a, m in protein_meta.items() if m["has_experimental"]}
    print(f"\nSHARED PROTEIN SET (>=1 experimental label): {len(exp_prots):,} of "
          f"{len(protein_meta):,} proteins "
          f"({sum(m['n_rows'] for m in exp_prots.values()):,} rows)")

    def composition(meta: dict[str, dict]) -> dict:
        lin = defaultdict(int)
        for m in meta.values():
            lin[m["lineage"]] += 1
        lens = [m["seq_len"] for m in meta.values()] or [0]
        return {
            "n_proteins": len(meta),
            "n_rows": int(sum(m["n_rows"] for m in meta.values())),
            "mean_length": float(np.mean(lens)),
            "median_length": float(np.median(lens)),
            "lineage": dict(sorted(lin.items(), key=lambda kv: -kv[1])[:10]),
        }

    comp_all, comp_exp = composition(protein_meta), composition(exp_prots)
    print(f"  {'':<22}{'ALL eval':>14}{'shared (exp)':>16}")
    print(f"  {'n proteins':<22}{comp_all['n_proteins']:>14,}"
          f"{comp_exp['n_proteins']:>16,}")
    print(f"  {'mean length':<22}{comp_all['mean_length']:>14.1f}"
          f"{comp_exp['mean_length']:>16.1f}")
    for lin in list(comp_all["lineage"])[:5]:
        a = comp_all["lineage"].get(lin, 0) / max(1, comp_all["n_proteins"])
        e = comp_exp["lineage"].get(lin, 0) / max(1, comp_exp["n_proteins"])
        print(f"  {lin:<22}{a:>13.1%}{e:>16.1%}")
    print("  ^ divergence here IS the confound the shared-protein set removes")

    # ---- G2 cost accounting: do the proteins EXCLUDED by the untruncated
    # population (>1022 residues) skew taxonomically, independent of the
    # per-concept residue-survival measurement? Meaningful only under
    # truncation_policy='with_truncated' (protein_meta has 0 truncated otherwise,
    # by construction) -- printed either way, gracefully empty under
    # 'untruncated'. ----
    trunc_prots = {a: m for a, m in protein_meta.items() if m["truncated"]}
    untrunc_prots = {a: m for a, m in protein_meta.items() if not m["truncated"]}
    comp_trunc, comp_untrunc = composition(trunc_prots), composition(untrunc_prots)
    print(f"\nTRUNCATED (>1022 residues, excluded under truncation_policy=untruncated) "
          f"vs UNTRUNCATED protein composition:")
    print(f"  {'':<22}{'truncated':>14}{'untruncated':>16}")
    print(f"  {'n proteins':<22}{comp_trunc['n_proteins']:>14,}"
          f"{comp_untrunc['n_proteins']:>16,}")
    print(f"  {'mean length':<22}{comp_trunc['mean_length']:>14.1f}"
          f"{comp_untrunc['mean_length']:>16.1f}")
    for lin in list(comp_trunc["lineage"])[:5]:
        t = comp_trunc["lineage"].get(lin, 0) / max(1, comp_trunc["n_proteins"])
        u = comp_untrunc["lineage"].get(lin, 0) / max(1, comp_untrunc["n_proteins"])
        print(f"  {lin:<22}{t:>13.1%}{u:>16.1%}")
    print("  ^ this is the taxonomic skew amendment G2's population filter "
          "introduces, at the protein level (independent of any concept)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as hf:
        pg = hf.create_group("protein_sets")
        accs = sorted(exp_prots)
        pg.create_dataset("has_experimental_accessions",
                          data=np.array(accs, dtype=h5py.string_dtype()))
        pg.create_dataset("has_experimental_start",
                          data=np.array([exp_prots[a]["start"] for a in accs],
                                        dtype=np.int64))
        pg.create_dataset("has_experimental_n_rows",
                          data=np.array([exp_prots[a]["n_rows"] for a in accs],
                                        dtype=np.int64))
        # G2: the population's own row subset, for analysis-time scripts
        # (feature_ap.py --scored-row-set population, null_pilot.py) to restrict
        # the activation gather to exactly this population -- n_rows above
        # already reflects it, but the GATHER needs the actual row indices,
        # not just the count. Meaningful (non-trivial) only when
        # truncation_policy=='untruncated'; written either way for a uniform
        # consumer-side code path.
        pop_accs = sorted(protein_meta)
        pg.create_dataset("population_start",
                          data=np.array([protein_meta[a]["start"] for a in pop_accs],
                                        dtype=np.int64))
        pg.create_dataset("population_n_rows",
                          data=np.array([protein_meta[a]["n_rows"] for a in pop_accs],
                                        dtype=np.int64))
        grp = hf.create_group("concepts")
        for concept, per_tier in kept.items():
            cg = grp.create_group(concept)
            # "all" is the pooled set; the tier datasets partition it, enabling
            # the pre-registered stratified analysis without a re-parse.
            cg.create_dataset(
                "all",
                data=np.unique(np.concatenate(list(per_tier.values()))),
                compression="gzip")
            for tier, arr in per_tier.items():
                cg.create_dataset(tier, data=arr, compression="gzip")
        # Instance boundaries (amendment E / D8): per KEPT, non-paired concept,
        # the (start_row, end_row) of every occurrence -- what a domain-level
        # recall metric needs to credit a whole instance from one firing
        # residue. Absent for paired types (DISULFID/CROSSLNK) and for any
        # concept with zero recorded instances.
        ig = hf.create_group("instances")
        for concept in kept:
            insts = concept_instances.get(concept)
            if not insts:
                continue
            arr = np.array(insts, dtype=np.int64)          # (n_instances, 2)
            cg = ig.create_group(concept)
            cg.create_dataset("start_row", data=arr[:, 0], compression="gzip")
            cg.create_dataset("end_row", data=arr[:, 1], compression="gzip")
        hf.attrs.update(
            n_rows=n_rows, cache=str(args.cache), dat=str(args.dat),
            min_positive_residues=min_pos, n_concepts=len(kept),
            truncation_policy=args.truncation_policy,
            retention_floor_mode=floor_mode,
            retention_min_residues=min_residues_interplm,
            retention_min_instances=min_instances_interplm,
            min_protein_count=args.min_protein_count,
            # M2: the exclusion the store was BUILT under, recorded so a
            # consumer can assert on it instead of trusting a config comment.
            provenance_excluded_types=json.dumps(sorted(prov_excluded)),
            # N2: which clustering the n_clusters axis came from. A stratum
            # axis whose source cannot be identified is not assertable.
            clusters_tsv=str(args.clusters) if args.clusters else "",
            git_sha=git_sha(), git_dirty=bool(git_dirty()),
        )
    manifest = {
        "cache": str(args.cache), "dat": str(args.dat), "n_rows": int(n_rows),
        "n_rows_full_cache": int(n_rows_cache),
        "truncation_policy": args.truncation_policy,
        "n_proteins_cache": len(index), "n_proteins_matched": seen,
        "n_proteins_population": len(by_acc),
        "n_length_mismatch": len_mismatch, "n_truncated_proteins": n_trunc,
        "dropped_annotated_residues": dict(dropped),
        "dropped_instance_boundary_residues": dict(instance_dropped),
        "min_positive_residues": min_pos,
        "retention_floor_mode": floor_mode,
        "retention_min_residues": min_residues_interplm,
        "retention_min_instances": min_instances_interplm,
        "min_protein_count": args.min_protein_count,
        # PREREGISTRATION amendment E: the measured impact of the floor choice,
        # for EVERY concept seen (kept or not), so "decide + document, don't
        # silently change" is auditable rather than asserted.
        "retention_floor_diagnostics": floor_diagnostics,
        "tier_groups": TIER_GROUPS,
        "concepts": {c: totals[c] for c in sorted(kept)},
        "evidence_tiers": {
            c: {t: int(a.size) for t, a in sorted(kept[c].items())}
            for c in sorted(kept)
        },
        "evidence_groups": {c: group_counts(kept[c]) for c in sorted(kept)},
        "evidence_groups_overall": dict(all_groups),
        # Stratum composition -- the diagnostic that shows whether the
        # experimental and propagated strata are different protein populations.
        "composition_all_eval": comp_all,
        "composition_shared_experimental": comp_exp,
        "composition_truncated": comp_trunc,
        "composition_untruncated": comp_untrunc,
        "dropped_small_concepts": dict(small),
        "git_sha": git_sha(), "git_dirty": git_dirty(),
    }
    args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {args.out} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
