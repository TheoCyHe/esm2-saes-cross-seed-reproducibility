"""
Scan a residue-labels store for concepts with IDENTICAL positive row sets.

Why this matters: g_f is a max over concepts, and its ARGMAX identity feeds
the concentration diagnostic (D5's pooled-DOMAIN/REGION caveat, the
per-feature granularity comparison). Two concepts with identical positive
sets give a feature an EXACT AP tie between them -- not a hypothetical: with
pooled type-level parents retained as a diagnostic arm (amendment E), any
type where every occurrence collapses to ONE surviving /note family (or to
":unspecified" alone) gives parent === that one child, positive-set-identical
by construction. Duplicates also inflate the FDR test count with perfectly
DEPENDENT tests -- BH assumes tests are independent or positively dependent
in the PRDS sense; two literally identical tests are the extreme case of that,
worth knowing the prevalence of before trusting the correction.

Hashing each concept's sorted row array is O(rows) per concept; grouping by
hash then verifying exact equality within each hash bucket avoids an O(n^2)
pairwise comparison over ~950 concepts while still being exact (the hash is
only used to CANDIDATE-group, never trusted as a substitute for the
elementwise check).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


def concept_row_hash(rows: np.ndarray) -> str:
    return hashlib.sha256(rows.tobytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    with h5py.File(args.labels, "r") as hf:
        names = list(hf["concepts"])
        rows_by_hash: dict[str, list[str]] = defaultdict(list)
        sizes: dict[str, int] = {}
        for name in names:
            rows = hf["concepts"][name]["all"][:]
            sizes[name] = int(rows.size)
            rows_by_hash[concept_row_hash(rows)].append(name)

    # Verify hash collisions are TRUE duplicates, not hash accidents -- reread
    # and compare elementwise for any bucket with >1 member.
    dup_groups = []
    with h5py.File(args.labels, "r") as hf:
        for h, members in rows_by_hash.items():
            if len(members) < 2:
                continue
            ref = hf["concepts"][members[0]]["all"][:]
            verified = [members[0]]
            for m in members[1:]:
                cand = hf["concepts"][m]["all"][:]
                if cand.size == ref.size and np.array_equal(cand, ref):
                    verified.append(m)
            if len(verified) > 1:
                dup_groups.append(sorted(verified, key=lambda c: names.index(c)))

    n_concepts = len(names)
    n_in_dup_group = sum(len(g) for g in dup_groups)
    print(f"{n_concepts} concepts scanned")
    print(f"{len(dup_groups)} duplicate GROUPS found, covering {n_in_dup_group} concepts "
          f"({n_in_dup_group / n_concepts:.1%})")
    print()
    for g in sorted(dup_groups, key=lambda g: -sizes[g[0]]):
        print(f"  size={sizes[g[0]]:>10,}  {g}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n_concepts": n_concepts,
        "n_duplicate_groups": len(dup_groups),
        "n_concepts_in_duplicate_groups": n_in_dup_group,
        "duplicate_groups": [
            {"concepts": g, "n_positives": sizes[g[0]]} for g in dup_groups
        ],
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
