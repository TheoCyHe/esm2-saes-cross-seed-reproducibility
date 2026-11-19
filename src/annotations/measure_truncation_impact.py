"""
P4 -- what truncation costs the sub-typed concept library. Parse-side,
CPU-only, independent of the AP restructure (P1). Blocks D8's straddling-
instance rule (a domain crossing residue 1022 has an ill-defined
domains-hit/total-domains contribution) and the truncation/retention-floor
decoupling amendment E left open.

For every FT occurrence of every concept, computes BOTH the UNTRUNCATED
residue/instance count (from the raw 1-based FT positions, ignoring the
cache's truncated length) and the TRUNCATED/surviving count (the same
map_residues_to_rows logic build_residue_labels.py already uses). Reports,
per concept: fraction of instances and residues lost past residue 1022, and
whether the concept's retention-floor status (E2's disjunctive rule) would
FLIP if computed on untruncated counts instead. Also profiles the length and
taxonomic composition of the proteins that actually contribute a truncated
instance -- only proteins the cache index marks `truncated` can ever do so
(an untruncated protein's n_rows == its full seq_len, so no in-range FT
position is ever dropped).
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

from src.annotations.build_residue_labels import (
    load_cache_index,
    map_residues_to_rows,
    sanitize_concept_name,
)
from src.annotations.uniprot_sites import (
    PAIRED_RESIDUE_TYPES,
    features_to_concept_occurrences,
    iter_dat_records,
    residues_for_feature,
)
from src.data.split_train_eval import header_to_acc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dat", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    ann = cfg["annotations"]
    min_residues = ann.get("retention_min_residues", 1500)
    min_instances = ann.get("retention_min_instances", 10)
    # Amendment CR: no private four-type fallback; an absent key is a refusal.
    subtype_types = frozenset(
        lookup(cfg, "annotations.subtype_note_types", args.config))

    index, n_rows = load_cache_index(args.cache)
    by_acc = {header_to_acc(e["accession"]): e for e in index}
    n_trunc_proteins = sum(1 for e in index if e["truncated"])
    print(f"cache: {len(index):,} proteins, {n_trunc_proteins:,} truncated")

    # per concept: residues/instances under BOTH counting conventions
    untrunc_res: dict[str, int] = defaultdict(int)
    trunc_res: dict[str, int] = defaultdict(int)
    untrunc_inst: dict[str, int] = defaultdict(int)
    trunc_inst: dict[str, int] = defaultdict(int)

    # proteins that contribute >=1 truncated (partially or fully lost)
    # instance to ANY concept -- for the length/taxonomy profile
    contributing: dict[str, dict] = {}
    all_truncated_meta: dict[str, dict] = {}
    dropped = defaultdict(int)
    seen = 0

    for rec in iter_dat_records(args.dat, keep=set(by_acc)):
        entry = by_acc.get(rec.acc)
        if entry is None:
            continue
        seen += 1
        if entry["truncated"]:
            all_truncated_meta[rec.acc] = {
                "seq_len": rec.seq_len or entry["seq_len"],
                "lineage": rec.lineage or "unknown",
            }
        occurrences = {
            sanitize_concept_name(c): occs
            for c, occs in features_to_concept_occurrences(
                rec.feats, subtype_note_types=subtype_types).items()
        }
        protein_contributed = False
        for concept, occs in occurrences.items():
            ftype = concept.split(":", 1)[0]
            if ftype in PAIRED_RESIDUE_TYPES:
                continue                      # P4 scopes span concepts only,
                                              # same as the instance-boundary
                                              # capture amendment E already does
            for o in occs:
                full_residues = residues_for_feature(ftype, o.start, o.end)
                n_full = len(full_residues)
                rows = map_residues_to_rows(entry, full_residues, dropped)
                n_surv = len(rows)

                untrunc_res[concept] += n_full
                trunc_res[concept] += n_surv
                untrunc_inst[concept] += 1
                if n_surv > 0:
                    trunc_inst[concept] += 1
                if n_surv < n_full:
                    protein_contributed = True
        if protein_contributed:
            contributing[rec.acc] = {
                "seq_len": rec.seq_len or entry["seq_len"],
                "lineage": rec.lineage or "unknown",
            }

    print(f"matched {seen:,}/{len(index):,} proteins")
    print(f"{len(contributing):,} proteins contribute >=1 truncated instance "
          f"to >=1 concept (of {n_trunc_proteins:,} truncated proteins total)")

    all_concepts = sorted(set(untrunc_res) | set(trunc_res))

    def passes(res: int, inst: int) -> bool:
        return res > min_residues or inst > min_instances

    rows_out = []
    n_flip = 0
    for c in all_concepts:
        ur, tr = untrunc_res[c], trunc_res[c]
        ui, ti = untrunc_inst[c], trunc_inst[c]
        p_untrunc, p_trunc = passes(ur, ui), passes(tr, ti)
        flips = p_untrunc != p_trunc
        n_flip += flips
        rows_out.append({
            "concept": c,
            "untrunc_residues": ur, "trunc_residues": tr,
            "residue_loss_frac": (ur - tr) / ur if ur else 0.0,
            "untrunc_instances": ui, "trunc_instances": ti,
            "instance_loss_frac": (ui - ti) / ui if ui else 0.0,
            "passes_floor_untruncated": p_untrunc,
            "passes_floor_truncated": p_trunc,
            "floor_status_flips": flips,
        })
    rows_out.sort(key=lambda r: -r["untrunc_residues"])

    print(f"\n{n_flip} of {len(all_concepts)} concepts FLIP retention-floor "
          f"status between truncated and untruncated counting")
    print(f"\n{'concept':<28}{'res_loss%':>10}{'inst_loss%':>11}{'flips':>7}")
    for r in rows_out[:40]:
        if r["untrunc_residues"] < 100:
            continue
        flag = "  <-- FLIP" if r["floor_status_flips"] else ""
        print(f"{r['concept']:<28}{r['residue_loss_frac']:>9.1%}"
              f"{r['instance_loss_frac']:>10.1%}{flag}")

    # length / taxonomy profile: contributing proteins vs all truncated
    # proteins vs the full eval population (composition_all_eval numbers are
    # already on record in residue_labels_subtyped.json; recomputed here too
    # so this script is self-contained)
    def composition(meta: dict[str, dict]) -> dict:
        lens = [m["seq_len"] for m in meta.values()] or [0]
        lin = defaultdict(int)
        for m in meta.values():
            lin[m["lineage"]] += 1
        return {
            "n_proteins": len(meta),
            "mean_length": float(np.mean(lens)),
            "median_length": float(np.median(lens)),
            "lineage": dict(sorted(lin.items(), key=lambda kv: -kv[1])[:10]),
        }

    comp_contrib = composition(contributing)
    comp_all_trunc = composition(all_truncated_meta)
    print(f"\nCOMPOSITION: proteins contributing a truncated instance vs all "
          f"truncated proteins")
    print(f"  {'':<20}{'contributing':>14}{'all truncated':>16}")
    print(f"  {'n proteins':<20}{comp_contrib['n_proteins']:>14,}"
          f"{comp_all_trunc['n_proteins']:>16,}")
    print(f"  {'mean length':<20}{comp_contrib['mean_length']:>14.1f}"
          f"{comp_all_trunc['mean_length']:>16.1f}")
    for lin in list(comp_all_trunc["lineage"])[:5]:
        a = comp_contrib["lineage"].get(lin, 0) / max(1, comp_contrib["n_proteins"])
        b = comp_all_trunc["lineage"].get(lin, 0) / max(1, comp_all_trunc["n_proteins"])
        print(f"  {lin:<20}{a:>13.1%}{b:>16.1%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "min_residues": min_residues, "min_instances": min_instances,
        "n_proteins_matched": seen, "n_truncated_proteins": n_trunc_proteins,
        "n_contributing_proteins": len(contributing),
        "n_concepts_flipping": n_flip,
        "concepts": rows_out,
        "composition_contributing": comp_contrib,
        "composition_all_truncated": comp_all_trunc,
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
