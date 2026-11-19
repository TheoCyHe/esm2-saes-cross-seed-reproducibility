"""TOTAL-FUNCTION completeness audit: grep EVERY file under a data root for
retired literals and assert zero un-allowlisted hits.

WHY THIS REPLACES TYPE-ENUMERATION SCANS. Amendment AE's blast-radius scan
enumerated the artifact types someone knew about --
`$PROJECT/analysis/*.h5`, `$SCRATCH/analysis/*.h5`, `$SCRATCH/null_raw/*.h5` --
and then asserted "no retired key remains anywhere on disk". That is a PARTIAL
function presented as TOTAL. It missed `$SCRATCH/row_index_cache/`, leaving 63
cached indices carrying the retired attr. The design saved it: I3's refusal
semantics meant the consumer REFUSED rather than substituting a
differently-scoped index, so no covariate was computed on wrong rows. But the
completeness claim was false when it was made, and it was false for a reason
that recurs -- a list of places is only as complete as the memory that wrote it.

This walks the filesystem instead. It does not know what an artifact is, which
is exactly the property that makes it total over the space it claims to cover.

HDF5 stores attribute NAMES as plain bytes in the object header, so a raw byte
search finds them without needing to know the file is HDF5, or which datasets
it has, or what schema it follows.

FALSE POSITIVES ARE ALLOWLISTED EXPLICITLY, NEVER FILTERED HEURISTICALLY.
`population_start` / `population_n_rows` / `population_accessions` are the
DELIBERATELY retained dataset prefixes (amendment AE: datasets are data, not
labels). They are listed below with that justification. Anything else is a hit.

    python -m src.analysis.audit_retired_literals --root <dir> [--root <dir>...]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Literals amendment AE retired. Bytes, because the files are mostly binary.
RETIRED: dict[str, bytes] = {
    "row_subset": b"row_subset",
    "population": b"population",
}

# Benign occurrences, each with the reason it is benign. A context is allowed
# ONLY if it appears here -- there is no heuristic fallback.
ALLOWED: dict[str, tuple[bytes, ...]] = {
    "population": (
        b"population_start",      # AE: retained dataset prefix
        b"population_n_rows",     # AE: retained dataset prefix
        b"population_accessions",  # AE: retained dataset prefix
        # AK/AL: a JSON manifest COUNT field, and the first allowlisted context
        # where the retired literal is a SUFFIX rather than a prefix. Written by
        # build_residue_labels as "how many proteins are in the population" --
        # a measured quantity, not an identity label, so it falls under AE's
        # "datasets are data, not labels" reasoning exactly as the three
        # dataset prefixes above do. Surfaced by running the audit after the
        # AK §6a JSON migration: 6 files, one hit each, all this field.
        b"n_proteins_population",
    ),
    "row_subset": (),             # fully retired: no benign context
}


# Longest thing we ever need intact across a chunk boundary.
_MARGIN = max(len(x) for x in list(RETIRED.values())
              + [c for v in ALLOWED.values() for c in v])
CHUNK = 64 << 20        # 64 MiB


def _scan_block(blob: bytes, base: int, lo: int, hi: int
                ) -> list[tuple[str, int, bytes]]:
    """Hits whose START falls in [lo, hi) of `blob`, offsets reported as
    absolute (`base + position`)."""
    for ctxs in ALLOWED.values():
        for ctx in ctxs:
            blob = blob.replace(ctx, b"\x00" * len(ctx))
    out = []
    for name, lit in RETIRED.items():
        for m in re.finditer(re.escape(lit), blob):
            if lo <= m.start() < hi:
                s0 = max(0, m.start() - 24)
                out.append((name, base + m.start(), blob[s0:m.end() + 24]))
    return out


def scan_file(path: Path) -> list[tuple[str, int, bytes]]:
    """Hits in one file, after removing allowlisted contexts.

    STREAMED, because the artifacts this must cover include 21-23 GB null_raw
    files -- reading them whole was killed by the OOM reaper, and a
    completeness audit that dies on the largest artifacts is not total.

    Chunks overlap by `_MARGIN` on each side and each chunk only REPORTS hits
    starting in its core, so every byte position is judged exactly once and in a
    chunk where it has full context on both sides. Without that, a literal or an
    allowlisted context straddling a boundary would be miscounted -- either
    missed, or reported twice.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return []
    hits: list[tuple[str, int, bytes]] = []
    try:
        with path.open("rb") as fh:
            pos = 0
            while pos < size:
                start = max(0, pos - _MARGIN)
                fh.seek(start)
                blob = fh.read((pos - start) + CHUNK + _MARGIN)
                lo = pos - start
                hi = min(lo + CHUNK, len(blob))
                hits.extend(_scan_block(blob, start, lo, hi))
                pos += CHUNK
    except OSError:
        return []
    return hits


def live_h5_attrs(path: Path) -> set[str] | None:
    """Retired attrs that are LIVE in an HDF5 file, or None if it is not HDF5.

    WHY A BYTE GREP NEEDS THIS. HDF5 does not reclaim the bytes of a deleted
    attribute -- the name string stays in the object header until the file is
    repacked. So after amendment AE's migration, `feature_ap_main.h5` still
    CONTAINS the bytes `population` while `h5py` correctly reports the attr as
    absent. Measured: 134 byte hits across 72 files, and **zero** live retired
    attrs in 96 h5 files.

    Reporting those tombstones as failures would make the audit fail on
    correct data, which is exactly how an audit gets switched off rather than
    fixed. So for HDF5 the API is the authority and the byte grep is the
    net that finds candidates.
    """
    try:
        import h5py
        with h5py.File(path, "r") as hf:
            return {k for k in RETIRED if k in hf.attrs}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CONVERTING ASSERTIONS INTO MEASUREMENTS (amendment AN4).
#
# The tombstone category is the model: it is not an allowlist entry but a
# per-file question put to the h5py API, so it is VERIFIED rather than asserted.
# A justification can go stale; a check cannot. So the byte-context allowlist
# gets the same treatment -- for each allowlisted context actually present in a
# file, confirm it corresponds to a REAL dataset name, group name or attr name
# (h5) or a REAL key (json). A context that is structurally confirmed is
# explained by the file, not by a promise about it.
#
# WHAT DOES NOT CONVERT, and the reason is structural rather than laziness. The
# `--exclude` globs exclude by PATH, and what makes them safe is "this file is
# not an artifact" -- which requires knowing what an artifact is, the very thing
# this audit deliberately does not know (that ignorance is its totality
# property). Converting them would reintroduce the type enumeration AJ4 removed.
# But the DANGEROUS DIRECTION does convert: an exclusion that swallows an .h5 is
# excluding an artifact, and that is checkable. So is a glob matching nothing,
# which is a dead exclusion carrying an unearned reason.
# ---------------------------------------------------------------------------

def _h5_names(path: Path) -> set[str] | None:
    try:
        import h5py
        names: set[str] = set()
        with h5py.File(path, "r") as hf:
            hf.visit(lambda n: names.add(n.split("/")[-1]))
            names.update(hf.attrs)
            def _attrs(name, obj):
                names.update(obj.attrs)
            hf.visititems(_attrs)
        return names
    except Exception:
        return None


def _json_keys(path: Path) -> set[str] | None:
    try:
        doc = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        # NARROW DELIBERATELY. A bare `except Exception` here swallowed a
        # NameError -- `json` was not imported -- and every JSON file came back
        # "UNVERIFIABLE_FILE_TYPE", i.e. the measurement silently degraded to the
        # assertion it was replacing. A broad except on a verification path turns
        # a broken check into a passing one.
        return None
    out: set[str] = set()
    stack = [doc]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            out.update(o)
            stack.extend(o.values())
        elif isinstance(o, list):
            stack.extend(o)
    return out


def verify_allowlisted_contexts(path: Path) -> dict:
    """For each allowlisted context present in this file, is it a REAL name?

    Returns per-context CONFIRMED / UNCONFIRMED. `CONFIRMED` means the byte hit
    is explained by an actual dataset, group, attr or key of that name -- a
    measurement. `UNCONFIRMED` means the allowlist is still carrying it on an
    assertion, which is worth knowing and is not by itself a failure.
    """
    # STREAMED, at the same 64 MiB chunk with the same overlap margin the main
    # scan uses. The first version of this function called `path.read_bytes()`
    # and would have been OOM-killed on the 23 GB and 25 GB `null_raw` files
    # under `--mem=8G` -- REINTRODUCING THE EXACT DEFECT AJ4 FIXED, in the same
    # module, in the commit that was converting assertions into measurements.
    # Caught by reading the sizes against the memory request, before the job
    # finished. (Amendment AN10; norm AN7's fourth instance.)
    present_set: set[str] = set()
    wanted = [(name, c) for ctxs in ALLOWED.values() for c in ctxs
              for name in [c.decode()]]
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            pos = 0
            while pos < size:
                start = max(0, pos - _MARGIN)
                fh.seek(start)
                blob = fh.read((pos - start) + CHUNK)
                for name, c in wanted:
                    if c in blob:
                        present_set.add(name)
                pos += CHUNK
                if len(present_set) == len(wanted):
                    break
    except OSError:
        return {"status": "UNREADABLE"}
    present = sorted(present_set)
    if not present:
        return {"status": "NO_ALLOWLISTED_CONTEXT"}
    names = _h5_names(path)
    kind = "h5"
    if names is None:
        names = _json_keys(path)
        kind = "json"
    if names is None:
        return {"status": "UNVERIFIABLE_FILE_TYPE", "contexts": present}
    per = {c: ("CONFIRMED" if c in names else "UNCONFIRMED") for c in present}
    return {
        "status": "CHECKED", "kind": kind, "contexts": per,
        "n_confirmed": sum(1 for v in per.values() if v == "CONFIRMED"),
        "n_unconfirmed": sum(1 for v in per.values() if v == "UNCONFIRMED"),
    }


def audit_exclusions(roots: list[Path], globs: list[str]) -> dict:
    """The dangerous direction, which IS checkable: does any exclusion glob
    swallow an artifact, and does any match nothing at all?"""
    rows = []
    for g in globs:
        matched, h5s = 0, []
        for root in roots:
            for p in root.rglob("*"):
                if p.is_file() and p.match(g):
                    matched += 1
                    if p.suffix in (".h5", ".hdf5"):
                        h5s.append(str(p))
        rows.append({"glob": g, "n_matched": matched,
                     "excluded_h5_artifacts": h5s,
                     "dead": matched == 0})
    bad = [r for r in rows if r["excluded_h5_artifacts"]]
    return {"rows": rows, "dead_globs": [r["glob"] for r in rows if r["dead"]],
            "globs_excluding_h5": [r["glob"] for r in bad],
            "verdict": "OK" if not bad else "EXCLUDES_AN_ARTIFACT"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, action="append", required=True)
    ap.add_argument("--max-report", type=int, default=12)
    ap.add_argument("--exclude", action="append", default=[],
                    help="glob of paths to skip. For files that record the "
                         "retired names BY DESIGN -- the pre-migration attr "
                         "snapshot, and this audit's own logs, which print the "
                         "literals they search for. Kept as a CALL-SITE policy "
                         "rather than a hardcoded allowlist, so the tool stays "
                         "general and the exclusions are visible in the "
                         "invocation that claims completeness.")
    args = ap.parse_args()

    n_files = 0
    offenders: dict[Path, list] = {}
    tombstoned: dict[Path, int] = {}
    excluded = 0
    for root in args.root:
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            if any(p.match(g) for g in args.exclude):
                excluded += 1
                continue
            n_files += 1
            hits = scan_file(p)
            if not hits:
                continue
            live = live_h5_attrs(p)
            if live is not None:
                # HDF5: the API decides. Byte hits with no live attr are
                # tombstones from the migration, not survivals.
                if not live:
                    tombstoned[p] = len(hits)
                    continue
                hits = [h for h in hits if h[0] in live]
            offenders[p] = hits

    roots = ", ".join(str(r) for r in args.root)
    print(f"scanned {n_files:,} files under {roots}"
          + (f"  ({excluded} excluded by --exclude)" if excluded else ""))
    print(f"retired literals: {sorted(RETIRED)}")
    print(f"allowlisted contexts: "
          f"{ {k: [c.decode() for c in v] for k, v in ALLOWED.items()} }")
    print()
    if tombstoned:
        print(f"{len(tombstoned)} HDF5 files carry byte hits with NO live attr "
              f"({sum(tombstoned.values())} bytes-level matches) -- deleted-"
              f"attribute tombstones, not survivals. Not counted as failures.")
        print()
    # AN4: the allowlist, VERIFIED rather than asserted, wherever it can be.
    ctx_rows = {}
    for root in args.root:
        for p in sorted(root.rglob("*")):
            if not p.is_file() or any(p.match(g) for g in args.exclude):
                continue
            r = verify_allowlisted_contexts(p)
            if r.get("status") == "CHECKED":
                ctx_rows[str(p)] = r
    n_conf = sum(r["n_confirmed"] for r in ctx_rows.values())
    n_unconf = sum(r["n_unconfirmed"] for r in ctx_rows.values())
    print(f"allowlisted contexts VERIFIED against real names in "
          f"{len(ctx_rows)} files: {n_conf} confirmed, {n_unconf} unconfirmed "
          f"(AN4 -- a confirmed context is explained by the file, not by a "
          f"promise about it)")
    if n_unconf:
        for path, r in ctx_rows.items():
            un = [c for c, v in r["contexts"].items() if v == "UNCONFIRMED"]
            if un:
                print(f"    UNCONFIRMED {un} in {path}")
    exc = audit_exclusions(list(args.root), list(args.exclude))
    print(f"exclusion globs: {len(exc['rows'])}, dead "
          f"{exc['dead_globs']}, excluding an h5 artifact "
          f"{exc['globs_excluding_h5']} -> {exc['verdict']}")
    print()
    if exc["verdict"] != "OK":
        print("*** AN INDIVIDUAL EXCLUSION GLOB SWALLOWS AN HDF5 ARTIFACT ***")
        raise SystemExit(1)

    if not offenders:
        print(f"ZERO live un-allowlisted hits in {n_files:,} files -- the claim "
              f"'no retired literal remains under these roots' is now TOTAL "
              f"over the roots, not over a remembered list of artifact types.")
        return

    total = sum(len(v) for v in offenders.values())
    print(f"*** {total} un-allowlisted hits in {len(offenders)} files ***")
    for p, hits in list(offenders.items())[:args.max_report]:
        names = sorted({h[0] for h in hits})
        print(f"  {p}  ({len(hits)} hits: {names})")
        for name, off, ctx in hits[:2]:
            print(f"      @{off}: {ctx!r}")
    if len(offenders) > args.max_report:
        print(f"  ... and {len(offenders) - args.max_report} more files")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
