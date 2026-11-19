"""Derive null reductions from the RAW per-(seed, shift, concept, feature)
gain array (amendment K3), for a given concept axis.

WHY THIS EXISTS, AND WHY IT IS NOT OPTIONAL AFTER A RE-SLICE.

The raw gain values are per (feature, concept): dropping a *different*
concept cannot change any of them, which is what makes the re-slice exact.
**The REDUCTIONS are not like that.** Both stored reductions are MAXIMA over
the concept axis:

    null_g_*    = max over CONCEPTS per feature   -> changes if the pool changes
    null_cmax_* = max over FEATURES per concept   -> unchanged per concept, but
                                                     its concept axis shrinks

So after the provenance exclusion (amendment N1) removed four concepts from
the pool, `null_pilot_k.h5`'s `null_g_*` are maxima over 707 concepts
INCLUDING `HELIX`, `STRAND`, `TURN` and `COMPBIAS` -- a null for a statistic
the compliant axis no longer computes. Consuming them against a 703-pool
`g_f` is exactly the amendment-I4 defect in a new place.

This is precisely what K3 stored the raw object for: the reduction is
re-derived, not re-run. ~2 h 10 m of GPU is not spent again.

The source file is also MARKED stale for the compliant axis, in its own
attributes -- a stale artifact that looks authoritative is the failure mode
this project keeps hitting (gotcha 15).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.analysis.artifact_identity import write_identity
from src.sae.train import git_dirty, git_sha


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, required=True,
                   help="sliced raw gain array (K3), on the target axis")
    p.add_argument("--source-reductions", type=Path, required=True,
                   help="the reductions file to inherit fire_rate/schedules "
                        "from, and to MARK as stale for this axis")
    p.add_argument("--labels", type=Path, required=True,
                   help="the label store defining the TARGET axis. Required "
                        "because `n_concepts` means the LIBRARY size (L4b) and "
                        "must come from the new axis, NOT be inherited from "
                        "the source reductions -- inheriting it wrote 716 onto "
                        "a 712-concept library and the pre-flight gate "
                        "correctly refused the result.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mark-source-stale", action="store_true",
                   help="write a `superseded_for_axis` attr onto the source, "
                        "so a consumer cannot read its maxima as current for "
                        "the new pool without tripping the pre-flight gate.")
    args = p.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    with h5py.File(args.raw, "r") as raw, h5py.File(args.out, "w") as out:
        concepts = [c.decode() if isinstance(c, bytes) else c
                    for c in raw["concepts"][:]]
        out.create_dataset("concepts",
                           data=np.array(concepts, dtype=h5py.string_dtype()))
        n_c = len(concepts)
        for arm in ("a", "b"):
            key = f"gain_{arm}"
            if key not in raw:
                continue
            d = raw[key]
            ns, nsh, nc_, nf = d.shape
            assert nc_ == n_c, (nc_, n_c)
            g = np.zeros((ns, nsh, nf), dtype=np.float64)     # max over concepts
            cm = np.zeros((ns, nsh, n_c), dtype=np.float64)   # max over features
            for s in range(ns):
                for r in range(nsh):
                    sl = np.asarray(d[s, r])                  # (concepts, feats)
                    g[s, r] = sl.max(axis=0)
                    cm[s, r] = sl.max(axis=1)
            out.create_dataset(f"null_g_{arm}", data=g, compression="gzip")
            out.create_dataset(f"null_cmax_{arm}", data=cm, compression="gzip")
            print(f"derived null_g_{arm} {g.shape} and null_cmax_{arm} {cm.shape}")

        # fire_rate and the schedules are properties of the SAEs and the
        # permutation, not of the concept axis, so they carry over unchanged.
        with h5py.File(args.source_reductions, "r") as src:
            for key in ("fire_rate", "seeds", "shift_schedule",
                        "dest_schedule", "length_bins"):
                if key in src:
                    out.create_dataset(key, data=src[key][:], compression="gzip")
            inherited = {k: v for k, v in src.attrs.items()}

        # ------------------------------------------------------------------
        # DEFAULT-DENY. A derived artifact inherits NOTHING unless the key is
        # named here.
        #
        # WHY THE ALLOWLIST, AND NOT ONE MORE EXCLUSION. This was a
        # `DO_NOT_INHERIT` denylist and it leaked twice in one session:
        #
        #   1. the mis-attributed replacement, and
        #   2. `superseded_for_axis` -- the first pass marked the source
        #      stale, the second pass inherited that marker onto its own
        #      replacement, which then declared itself superseded and the
        #      pre-flight gate refused it.
        #
        # Both were fixed by adding the offending key. That repairs the case
        # and not the class: a denylist is only correct if every attr that
        # will EVER appear on a source has been thought of in advance, and the
        # attr set grows (`identity_correction`, `bin_counts_superseded_note`,
        # `raw_gain_path` all arrived after the list was written). The failure
        # is silent by construction -- a new key on the source propagates onto
        # a file it does not describe, and nothing reads wrong.
        #
        # A THIRD LEAK, FOUND BY THIS CONVERSION AND NOT BY REASONING:
        # `library_sha` and `labels` were inherited, so
        # `null_pilot_compliant.h5` -- built against the 712-concept f67485d
        # store -- shipped naming the 716-concept 9bfa94b store. The gate
        # passed it, because its `library_sha` check was itself dead (see
        # `artifact_identity`). Two independent defects lined up to hide one
        # stale identity, which is the argument for default-deny in one line.
        #
        # The split is by WHAT A KEY DESCRIBES:
        #   - the PERMUTATION and the SAEs -> a property of the source run,
        #     genuinely carried over, because the re-derivation does not
        #     change the shifts or the firing rates;
        #   - the CONCEPT AXIS and the label store -> re-read from --labels,
        #     never inherited, because changing that axis is the entire
        #     reason this script runs.
        INHERIT = frozenset({
            "n_shifts", "n_length_bins", "null_seed", "null_type",
            "exclude_pooled_parents",
            "bin_counts", "bin_counts_superseded_note",
            "scored_row_set", "n_proteins",
        })
        # The record is built FROM THE LOOP, not from the rule. Recomputing it
        # from `INHERIT` would restate the policy rather than report the act,
        # so it would keep reading correct even if the copy above did
        # something else -- a validator that cannot disagree with the thing it
        # describes (gotcha 13). Caught by mutation testing: with the copy
        # reverted to inherit-everything, the independently-computed version
        # still listed the right keys as dropped.
        inherited_keys: list[str] = []
        dropped_keys: list[str] = []
        for k, v in inherited.items():
            if k in INHERIT:
                out.attrs[k] = v
                inherited_keys.append(k)
            else:
                dropped_keys.append(k)
        out.attrs["inherited_keys"] = json.dumps(sorted(inherited_keys))
        out.attrs["not_inherited_keys"] = json.dumps(sorted(dropped_keys))

        # Everything describing the TARGET axis comes from the label store.
        # `n_concepts` is the LIBRARY size (L4b: one key, one meaning);
        # inheriting it wrote 716 onto a 712-concept library once already.
        with h5py.File(args.labels, "r") as lab:
            n_library = len(lab["concepts"].keys())
            lab_attrs = {k: v for k, v in lab.attrs.items()}
        for k in ("truncation_policy", "n_rows", "retention_floor_mode",
                  "retention_min_residues", "retention_min_instances",
                  "min_protein_count"):
            if k in lab_attrs:
                out.attrs[k] = lab_attrs[k]
        write_identity(
            out, n_concepts=n_library,
            n_concepts_library=n_library,
            argmax_pool_n=len(concepts),
            labels=str(args.labels),
            library_sha=(lab_attrs["git_sha"].decode()
                         if isinstance(lab_attrs.get("git_sha"), bytes)
                         else lab_attrs.get("git_sha")),
            derived_from_raw=str(args.raw),
            derived_from_reductions=str(args.source_reductions),
            derivation_note=(
                "Reductions RE-DERIVED from the K3 raw array for this concept "
                "axis, not re-run. null_g_* is a max over CONCEPTS, so it "
                "changes when the pool changes -- unlike the raw per-(f,c) "
                "values, which do not."),
            git_sha=git_sha(), git_dirty=bool(git_dirty()))
    print(f"wrote {args.out}")

    if args.mark_source_stale:
        with h5py.File(args.source_reductions, "a") as src:
            src.attrs["superseded_for_axis"] = json.dumps({
                "axis_argmax_pool_n": len(concepts),
                "reason": ("null_g_* here are maxima over a LARGER pool that "
                           "still contains the provenance-excluded concepts "
                           "(amendment N1). Valid only for the pool this file "
                           "was drawn on; use the derived file for the "
                           "compliant axis."),
                "replacement": str(args.out),
            })
        print(f"marked {args.source_reductions.name} stale for the "
              f"{len(concepts)}-concept axis")


if __name__ == "__main__":
    main()
