"""ONE-TIME migration for PREREGISTRATION amendment AK §6a: the JSON manifests
amendment AE's HDF5 pass did not touch.

WHAT AE MISSED, AND WHY IT IS WORTH A SECOND PASS. AE renamed
`population` -> `truncation_policy` and `row_subset` -> `scored_row_set` on the
HDF5 artifacts, driven by an attr dump of h5 files. Ten JSON manifests sitting
beside those h5 files carry the same retired keys and were not in that dump.

**Severity is low and is stated rather than inflated: no consumer reads either
key from JSON.** But `build_residue_labels.py` now writes the NEW key into its
manifest, so old and new manifests disagree about what the same field is
called -- and a reader comparing two label stores would find one describing
itself with a key the other does not have. That is the same defect shape as
gotcha 15: the file opens, the field is there, and it is not the field the
current schema names.

ONE SOURCE OF TRUTH FOR THE MAPPING. `KEY_RENAMES` and `VALUE_MAPS` are
IMPORTED from `ae_rename_attrs`, not restated. A second copy of a value map
whose whole purpose is removing a shared literal would be the accretion this
kind of pass exists to reverse, and the two copies would drift.

ACCEPTANCE CRITERION, the same one AE3 condition 4 states, adapted to JSON:
for every file touched, exactly the intended key paths changed, every other
value is byte-identical after a round-trip, no key appeared or vanished beyond
the intended rename, and every value round-trips through the declared mapping.
An undeclared value is REFUSED, never guessed.

RECURSION, AND WHY IT IS NOT OPTIONAL. The scan walks nested dicts and lists
rather than checking the top level, because a manifest that nests its identity
under `identity/` would otherwise be reported clean while carrying the retired
key one level down -- a partial function presented as total, which is precisely
AJ4's finding about AE's own blast-radius scan.

WHAT IS DELIBERATELY NOT MIGRATED, and this is a domain statement rather than
an omission: `migrations/attrs_before_AE.json` is the PRE-MIGRATION SNAPSHOT.
It records the retired names BY DESIGN -- it is the revert path AE3 condition 3
exists to provide -- so rewriting it would destroy the only record of what the
attrs were before. It is skipped by name, with that reason, and never by a
heuristic.

Usage:
    python -m migrations.ak_rename_json_manifests --root <dir> [--root <dir>]
    python -m migrations.ak_rename_json_manifests --root <dir> --apply
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

from migrations.ae_rename_attrs import KEY_RENAMES, VALUE_MAPS

SCHEMA_KEY = "attr_schema_version"

# Skipped BY NAME with a stated reason, never by a pattern that might catch
# something else. This is the revert path for AE, not a stale manifest.
SKIP_BASENAMES = ("attrs_before_AE.json",)


def find_retired(obj, path: str = "") -> list[tuple[str, str, object]]:
    """Every (key_path, retired_key, value) in a nested JSON document."""
    out: list[tuple[str, str, object]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            sub = f"{path}/{k}" if path else str(k)
            if k in KEY_RENAMES:
                out.append((sub, k, v))
            out += find_retired(v, sub)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += find_retired(v, f"{path}[{i}]")
    return out


def map_value(retired_key: str, value):
    """The declared mapping, or a refusal. No permissive fallback -- the shared
    literal `all` is exactly what a fallback would let back in."""
    mapping = VALUE_MAPS[retired_key]
    if not isinstance(value, str) or value not in mapping:
        raise SystemExit(
            f"REFUSING: {retired_key}={value!r} is not a declared value "
            f"({sorted(mapping)}). A permissive fallback is how the shared "
            f"literal came back.")
    return mapping[value]


def rename_in_place(obj, changed: list[str], path: str = "") -> object:
    """Rewrite retired keys throughout, preserving key ORDER.

    Order is preserved because the acceptance criterion below compares the
    serialised document, and a reordered manifest would report as a change
    everywhere while being semantically identical -- an instrument that fails
    on correct data, which is how one gets switched off (AJ4).
    """
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            sub = f"{path}/{k}" if path else str(k)
            if k in KEY_RENAMES:
                new[KEY_RENAMES[k]] = map_value(k, v)
                changed.append(sub)
            else:
                new[k] = rename_in_place(v, changed, sub)
        return new
    if isinstance(obj, list):
        return [rename_in_place(v, changed, f"{path}[{i}]")
                for i, v in enumerate(obj)]
    return obj


def flatten(obj, path: str = "") -> dict[str, object]:
    """Every leaf as `key_path -> value`, for the before/after comparison."""
    out: dict[str, object] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{path}/{k}" if path else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{path}[{i}]"))
    else:
        out[path] = obj
    return out


def verify(before: dict, after: dict, changed: list[str],
           schema_version: str) -> list[str]:
    """The acceptance criterion. Returns problems; empty means it held."""
    problems: list[str] = []
    fb, fa = flatten(before), flatten(after)

    expect_gone = set()
    expect_new = {SCHEMA_KEY}
    for cp in changed:
        expect_gone.add(cp)
        head, _, leaf = cp.rpartition("/")
        expect_new.add(f"{head}/{KEY_RENAMES[leaf]}" if head
                       else KEY_RENAMES[leaf])

    for cp in changed:
        head, _, leaf = cp.rpartition("/")
        newp = f"{head}/{KEY_RENAMES[leaf]}" if head else KEY_RENAMES[leaf]
        if cp in fa:
            problems.append(f"retired path {cp!r} survives")
        if newp not in fa:
            problems.append(f"renamed path {newp!r} is absent")
        elif fa[newp] != VALUE_MAPS[leaf].get(fb.get(cp)):
            problems.append(
                f"{newp!r} is {fa[newp]!r}, expected "
                f"{VALUE_MAPS[leaf].get(fb.get(cp))!r}")

    if fa.get(SCHEMA_KEY) != schema_version:
        problems.append(f"{SCHEMA_KEY} is {fa.get(SCHEMA_KEY)!r}, "
                        f"expected {schema_version!r}")

    for k in sorted(set(fb) | set(fa)):
        if k in expect_gone or k in expect_new:
            continue
        if k not in fa:
            problems.append(f"leaf {k!r} VANISHED")
        elif k not in fb:
            problems.append(f"leaf {k!r} APPEARED unexpectedly")
        elif fb[k] != fa[k]:
            problems.append(f"leaf {k!r} CHANGED: {fb[k]!r} -> {fa[k]!r}")
    return problems


def head_sha(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=repo).decode().strip()


def plan_file(path: Path) -> tuple[dict, dict, list[str]] | None:
    """(before, after, changed_paths), or None if there is nothing to do."""
    try:
        before = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    if not find_retired(before):
        return None
    changed: list[str] = []
    after = rename_in_place(copy.deepcopy(before), changed)
    return before, after, changed


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, action="append", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="without this, plan and report only")
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    schema_version = f"AK:{head_sha(args.repo)}"
    print(f"{SCHEMA_KEY} = {schema_version}")

    candidates: list[Path] = []
    skipped_by_design: list[Path] = []
    for root in args.root:
        for p in sorted(root.rglob("*.json")):
            if not p.is_file():
                continue
            if p.name in SKIP_BASENAMES:
                skipped_by_design.append(p)
                continue
            candidates.append(p)

    for p in skipped_by_design:
        print(f"SKIPPED BY DESIGN: {p.name} -- the pre-migration snapshot; it "
              f"records the retired names deliberately and is AE's revert path")

    results: dict[str, list[str]] = {}
    failed: list[str] = []
    n_todo = 0
    for p in candidates:
        planned = plan_file(p)
        if planned is None:
            continue
        n_todo += 1
        before, after, changed = planned
        desc = ", ".join(
            f"{cp} -> {KEY_RENAMES[cp.rpartition('/')[2]]}"
            f"={map_value(cp.rpartition('/')[2], flatten(before)[cp])!r}"
            for cp in changed)
        print(f"{p}\n      {desc}")
        if not args.apply:
            continue
        after[SCHEMA_KEY] = schema_version
        p.write_text(json.dumps(after, indent=2))
        reread = json.loads(p.read_text())
        problems = verify(before, reread, changed, schema_version)
        results[str(p)] = problems
        if problems:
            failed.append(str(p))
            print("      *** ACCEPTANCE CRITERION FAILED ***")
            for q in problems:
                print(f"        - {q}")
        else:
            print("      verified: intended paths only, all else identical")

    print(f"\n{n_todo} JSON manifests carry a retired key "
          f"({len(skipped_by_design)} skipped by design)")
    if args.apply:
        n_ok = sum(1 for v in results.values() if not v)
        print(f"{n_ok}/{len(results)} passed the acceptance criterion")
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(
                {"schema_version": schema_version, "results": results,
                 "skipped_by_design": [str(p) for p in skipped_by_design]},
                indent=2))
            print(f"wrote {args.report}")
        if failed:
            raise SystemExit(f"FAILED on {len(failed)}: {failed}")
    else:
        print("DRY RUN -- nothing written. Re-run with --apply.")


if __name__ == "__main__":
    main()
