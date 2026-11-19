"""ONE-TIME: give the UNLABELLED seed axes their labels. Amendment AS,
decision 5 (the author's authorisation, brief section 12).

WHY THIS IS NOT OPTIONAL TIDYING. AR9's defect was not that the hazard was
unknown -- `glob("seed_*")` sorting lexicographically is documented in
CODEBASE 7.3, and `src/analysis/seed_axis.py` was written for exactly this
failure. **The defect was that no consumer called it**, and the reason no
consumer called it is that the artifacts most in need of alignment carry no
labels to align on. `resolve_unlabelled_seed_order`'s 22.7x margin makes
deferring safe; it does not make it right, because a resolver carries the same
exposure the unused module did -- it only works if someone remembers to call
it. A LABEL IN THE FILE IS CHECKED BY EVERY READER; A CONVENTION IS CHECKED BY
WHOEVER REMEMBERS.

WHAT IT DOES. Adds a `seeds` dataset (and a short provenance attr) to
artifacts written by `standardize_gain.py` and `degenerate_flag.py`, which
write none. The precedent is `migrations/ak_rename_json_manifests.py` and the
AE rename before it: an in-place migration carrying its own
`attr_schema_version` so a migrated attr is distinguishable from an original.

THE LABELS ARE RECOVERED FROM THE DATA, NEVER ASSUMED. Every target is ordered
by `seed_axis.resolve_unlabelled_seed_order`, which refuses unless the
permutation is a bijection with every margin above 3x. A target that cannot be
resolved is REPORTED AND SKIPPED, not labelled on a guess -- writing a wrong
label would be strictly worse than writing none, because from then on every
reader would trust it (gotcha 15's shape: the stated spec of a file is not its
identity).

AND THE MIGRATION IS RE-READABLE AFTERWARDS. `seed_axis.seed_order` prefers
stored labels but STILL runs the resolver when it can and RAISES on
disagreement, so a mislabelled artifact cannot hide behind its own label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from src.analysis.seed_axis import (
    resolve_unlabelled_seed_order,
    stored_seed_labels,
)

SEED_DATASET = "seeds"


def migrate(target: Path, null_reductions: Path, *, dry_run: bool,
            schema_version: str) -> dict:
    """Label one artifact. Returns a row for the report; never raises on a
    target it cannot resolve -- it reports and skips."""
    row: dict = {"path": str(target), "exists": target.exists()}
    if not target.exists():
        row["action"] = "ABSENT"
        return row

    already = stored_seed_labels(target)
    if already is not None:
        row["action"] = "ALREADY_LABELLED"
        row["labels"] = already
        return row

    try:
        res = resolve_unlabelled_seed_order(target, null_reductions)
    except SystemExit as exc:
        row["action"] = "UNRESOLVABLE_SKIPPED"
        row["reason"] = str(exc)
        return row
    except (KeyError, OSError) as exc:
        row["action"] = "NO_GTILDE_NULL_SKIPPED"
        row["reason"] = f"{type(exc).__name__}: {exc}"
        return row

    row["labels"] = res["labels"]
    row["min_margin_observed"] = res["min_margin_observed"]
    row["method"] = res["method"]
    if dry_run:
        row["action"] = "WOULD_LABEL"
        return row

    with h5py.File(target, "r+") as hf:
        hf.create_dataset(SEED_DATASET,
                          data=[s.encode() for s in res["labels"]],
                          dtype=h5py.string_dtype())
        hf.attrs["seed_label_provenance"] = json.dumps({
            "recovered_by": "seed_axis.resolve_unlabelled_seed_order",
            "against": str(null_reductions),
            "min_margin_observed": res["min_margin_observed"],
            "amendment": "AS decision 5",
        })
        hf.attrs["attr_schema_version"] = schema_version
    # RE-READ AND VERIFY, rather than trusting the write. An operation that can
    # silently do nothing is not a verified operation (gotcha 13 extension 4).
    back = stored_seed_labels(target)
    row["action"] = "LABELLED" if back == res["labels"] else "WRITE_FAILED"
    row["read_back"] = back
    return row


def inherit(target: Path, source: Path, *, dry_run: bool,
            schema_version: str, shared: str = "argmax_concept") -> dict:
    """Label a DERIVED artifact from the one it was built from — VERIFIED.

    `degenerate_argmax_flag.h5` carries no `gtilde_null`, so the resolver
    correctly REFUSES it (measured: `NO_GTILDE_NULL_SKIPPED`). Its order is
    nonetheless recoverable, because it was built row-for-row from a gtilde and
    carries a COPY of that gtilde's `argmax_concept`. An ELEMENTWISE comparison
    of that copy is a positional check; a shape or a total is not, since both
    survive a permutation (gotcha 27). This is the same check `design3_join`
    already makes before it trusts the inheritance, reused rather than
    reimplemented.

    IT REFUSES ON ANY DIFFERENCE. One differing cell means the flag was not
    built from this artifact, and the order cannot be inherited from it.
    """
    row: dict = {"path": str(target), "source": str(source),
                 "mode": "inherit", "exists": target.exists()}
    if not target.exists():
        row["action"] = "ABSENT"
        return row
    if stored_seed_labels(target) is not None:
        row["action"] = "ALREADY_LABELLED"
        row["labels"] = stored_seed_labels(target)
        return row

    src_labels = stored_seed_labels(source)
    if src_labels is None:
        row["action"] = "SOURCE_UNLABELLED_SKIPPED"
        return row

    import numpy as np
    with h5py.File(target, "r") as hf:
        if shared not in hf:
            row["action"] = "NO_SHARED_DATASET_SKIPPED"
            row["reason"] = f"{shared} absent, so inheritance is unverifiable"
            return row
        a = hf[shared][:]
    with h5py.File(source, "r") as hf:
        b = hf[shared][:] if shared in hf else None
    if b is None or a.shape != b.shape or not np.array_equal(a, b):
        row["action"] = "NOT_ELEMENTWISE_EQUAL_REFUSED"
        row["reason"] = (f"`{shared}` differs, so this artifact was NOT built "
                         "row-for-row from that source")
        return row

    row["labels"] = src_labels
    row["verified_by"] = f"elementwise equality of `{shared}`"
    if dry_run:
        row["action"] = "WOULD_LABEL"
        return row
    with h5py.File(target, "r+") as hf:
        hf.create_dataset(SEED_DATASET,
                          data=[s.encode() for s in src_labels],
                          dtype=h5py.string_dtype())
        hf.attrs["seed_label_provenance"] = json.dumps({
            "inherited_from": str(source),
            "verified_by": f"elementwise equality of {shared}",
            "amendment": "AS decision 5",
        })
        hf.attrs["attr_schema_version"] = schema_version
    back = stored_seed_labels(target)
    row["action"] = "LABELLED" if back == src_labels else "WRITE_FAILED"
    row["read_back"] = back
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", type=Path, nargs="*", default=[])
    ap.add_argument("--inherit", type=Path, nargs=2, action="append",
                    default=[], metavar=("TARGET", "SOURCE"),
                    help="label TARGET from SOURCE, verified by elementwise "
                         "equality of argmax_concept. For artifacts with no "
                         "gtilde_null, which the resolver correctly refuses.")
    ap.add_argument("--null-reductions", type=Path, required=True)
    ap.add_argument("--schema-version", required=True,
                    help="e.g. AS:<git sha>, so a migrated axis is "
                         "distinguishable from an original one")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = [migrate(t, args.null_reductions, dry_run=args.dry_run,
                    schema_version=args.schema_version)
            for t in args.targets]
    rows += [inherit(Path(t), Path(src), dry_run=args.dry_run,
                     schema_version=args.schema_version)
             for t, src in args.inherit]
    for r in rows:
        print(f"  {r['action']:22s} {Path(r['path']).name}"
              + (f"  labels={r.get('labels')}" if "labels" in r else "")
              + (f"  margin={r['min_margin_observed']:.1f}x"
                 if "min_margin_observed" in r else ""), flush=True)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["action"]] = counts.get(r["action"], 0) + 1
    # DERIVED verdict with more than one reachable outcome (gotcha 25).
    bad = [r for r in rows if r["action"] in {
        "WRITE_FAILED", "NOT_ELEMENTWISE_EQUAL_REFUSED"}]
    verdict = ("REFUSED" if bad else
               "NOTHING_TO_DO" if not any(
                   r["action"] in {"LABELLED", "WOULD_LABEL"} for r in rows)
               else "OK")
    report = {"rows": rows, "counts": counts, "verdict": verdict,
              "dry_run": bool(args.dry_run),
              "schema_version": args.schema_version}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"[as-migrate] {counts} -> {verdict}; wrote {args.out}", flush=True)
    if verdict != "OK" and bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
