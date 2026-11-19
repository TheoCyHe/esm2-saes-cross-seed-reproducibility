"""ONE-TIME migration for PREREGISTRATION amendment AE: rename the identity
attrs `population` -> `truncation_policy` and `row_subset` -> `scored_row_set`
on every artifact already on disk, remapping the colliding value literals.

WHY A MIGRATION AND NOT A COMPATIBILITY SHIM. By amendment I3 an absent
identity key is reported ABSENT, never treated as matching, so renaming the key
in code alone would make every existing artifact fail the gate. The alternative
-- teaching the gate to read either name -- is a second source of truth for the
same fact, carried forever, which is exactly the accretion this cleanup exists
to reverse. So: one pass, then the old names never appear again.

ACCEPTANCE CRITERION (amendment AE3, condition 4), and it is the whole point:
the reproduction suite never calls preflight, so it CANNOT cover this. This
script's own before/after attr-dict comparison is the substitute, not an extra.
It verifies, for every artifact touched, that
  * exactly the intended keys changed,
  * every other attr is byte-identical,
  * no attr appeared or vanished beyond the intended rename, and
  * every value round-trips through the declared mapping.

ORDER (condition 5): the small artifacts first, the two 21-23 GB null_raw files
LAST, so a defect surfaces on a cheap file. Recovery path for those two is
recorded in AE4 -- regenerable by slurm/null_pilot.sbatch from the eval cache,
~2:10 of cluster time, ~54 days of purge runway remaining as of 2026-07-29.

PROVENANCE (condition 6): each migrated artifact gains `attr_schema_version` =
"AE:<sha>" where <sha> is the HEAD of the checkout running this. `git_sha`
still names the code that PRODUCED the file; this new attr says the labels were
rewritten later, by a known commit. It is a plain record, NOT a branch point --
the gate reads one key name, and a version-aware gate carrying two schemas
forever is more machinery than this one-time pass.

Usage:
    python -m migrations.ae_rename_attrs --before <attrs_before_AE.json>   # dry run
    python -m migrations.ae_rename_attrs --before <...> --apply
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import h5py

# key -> new key
KEY_RENAMES = {"population": "truncation_policy", "row_subset": "scored_row_set"}

# (old key, old value) -> new value. An unlisted value is REFUSED, never
# guessed: the collision this migration removes was two parameters sharing the
# literal "all", so a permissive fallback here would reintroduce it silently.
VALUE_MAPS: dict[str, dict[str, str]] = {
    "population": {"all": "with_truncated", "untruncated": "untruncated"},
    "row_subset": {"all": "full_eval",
                   "population": "label_store",
                   "shared_experimental": "shared_experimental"},
}

SCHEMA_KEY = "attr_schema_version"


def _dec(v):
    return v.decode() if isinstance(v, bytes) else v


def head_sha(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=repo).decode().strip()


def order_artifacts(paths: list[str]) -> list[str]:
    """Small first, the big null_raw files last (condition 5)."""
    return sorted(paths, key=lambda p: (Path(p).stat().st_size if Path(p).exists()
                                        else 0))


def plan_for(attrs: dict) -> dict[str, tuple[str, object]]:
    """{old_key: (new_key, new_value)} for one artifact. Empty = nothing to do."""
    plan = {}
    for old, new in KEY_RENAMES.items():
        if old not in attrs:
            continue
        val = _dec(attrs[old])
        mapping = VALUE_MAPS[old]
        if val not in mapping:
            raise SystemExit(
                f"REFUSING: {old}={val!r} is not a declared value "
                f"({sorted(mapping)}). A permissive fallback is how the shared "
                f"literal came back.")
        plan[old] = (new, mapping[val])
    return plan


def migrate(path: str, plan: dict, schema_version: str) -> None:
    with h5py.File(path, "r+") as h:
        for old, (new, new_val) in plan.items():
            if new in h.attrs:
                raise SystemExit(f"REFUSING {path}: {new} already present")
            h.attrs[new] = new_val
            del h.attrs[old]
        h.attrs[SCHEMA_KEY] = schema_version


def verify(path: str, before: dict, plan: dict, schema_version: str) -> list[str]:
    """Condition 4. Returns a list of problems; empty means the criterion held."""
    problems = []
    with h5py.File(path, "r") as h:
        after = {k: _dec(h.attrs[k]) for k in h.attrs}
    before = {k: _dec(v) for k, v in before.items()}

    for old, (new, new_val) in plan.items():
        if old in after:
            problems.append(f"old key {old!r} survives")
        if after.get(new) != new_val:
            problems.append(f"{new!r} is {after.get(new)!r}, expected {new_val!r}")
    if after.get(SCHEMA_KEY) != schema_version:
        problems.append(f"{SCHEMA_KEY} is {after.get(SCHEMA_KEY)!r}")

    # every OTHER attr byte-identical, and nothing extra appeared
    intended_new = {new for new, _ in plan.values()} | {SCHEMA_KEY}
    intended_gone = set(plan)
    for k in sorted(set(before) | set(after)):
        if k in intended_new or k in intended_gone:
            continue
        if k not in after:
            problems.append(f"attr {k!r} VANISHED")
        elif k not in before:
            problems.append(f"attr {k!r} APPEARED unexpectedly")
        elif before[k] != after[k]:
            problems.append(f"attr {k!r} CHANGED: {before[k]!r} -> {after[k]!r}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", type=Path, required=True,
                    help="attrs_before_AE.json from the condition-3 dump")
    ap.add_argument("--apply", action="store_true",
                    help="without this, plan and report only")
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    dump = json.loads(args.before.read_text())["artifacts"]
    schema_version = f"AE:{head_sha(args.repo)}"
    print(f"{SCHEMA_KEY} = {schema_version}")

    todo = {p: a for p, a in dump.items()
            if set(a["attrs"]) & set(KEY_RENAMES)}
    print(f"{len(todo)} of {len(dump)} artifacts carry a retired key\n")

    results, failed = {}, []
    for path in order_artifacts(list(todo)):
        attrs = todo[path]["attrs"]
        plan = plan_for(attrs)
        size_gb = Path(path).stat().st_size / 2**30 if Path(path).exists() else 0
        desc = ", ".join(f"{o}={_dec(attrs[o])!r} -> {n}={v!r}"
                         for o, (n, v) in plan.items())
        print(f"[{size_gb:6.2f} GB] {Path(path).name}\n            {desc}")
        if not args.apply:
            continue
        if not Path(path).exists():
            print("            SKIP: not on this filesystem"); continue
        migrate(path, plan, schema_version)
        problems = verify(path, attrs, plan, schema_version)
        results[path] = problems
        if problems:
            failed.append(path)
            print("            *** ACCEPTANCE CRITERION FAILED ***")
            for p in problems:
                print(f"              - {p}")
        else:
            print("            verified: intended keys only, all else identical")

    if args.apply:
        n_ok = sum(1 for v in results.values() if not v)
        print(f"\n{n_ok}/{len(results)} artifacts passed the acceptance criterion")
        if args.report:
            args.report.write_text(json.dumps(
                {"schema_version": schema_version,
                 "results": {k: v for k, v in results.items()}}, indent=2))
            print(f"wrote {args.report}")
        if failed:
            raise SystemExit(f"FAILED on {len(failed)}: {failed}")
    else:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")


if __name__ == "__main__":
    main()
