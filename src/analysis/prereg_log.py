"""The pre-registration log, read as one text (amendment CT, section CT18-D10).

The log is the original pre-registration and its dated amendments. Until 2026-11-09 it was
one file, `PREREGISTRATION.md`. It is now one file per amendment -- and, for amendment CT,
one file per section -- under `docs/record/preregistration/`, named `NNNN_<label>.md`.
File-name order is the log's order: the files concatenated in that order are byte-identical
to the single file as it stood when it was split (`PRESPLIT`), followed by whatever sections
were added afterwards. `PREREGISTRATION.md` is the index.

    python -m src.analysis.prereg_log check             # the invariants; exit 1 on a problem
    python -m src.analysis.prereg_log index --write     # regenerate the index and INDEX.tsv
    python -m src.analysis.prereg_log index --check     # exit 1 if the committed index is stale
    python -m src.analysis.prereg_log locate CT18-D9    # a section label -> file and line
    python -m src.analysis.prereg_log locate --line 18938   # a line of the old single file

Adding a section: write `NNNN_<label>.md` with the next number, then `index --write`.
Nothing already in the directory is edited; the log grows by addition.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG_DIR = REPO / "docs" / "record" / "preregistration"
INDEX_MD = REPO / "PREREGISTRATION.md"
INDEX_TSV_NAME = "INDEX.tsv"

# The single file at the commit that split it (22d6748, the commit adding section CT18-D10).
PRESPLIT = {"sha256": "82e1d2102889646c326fbf85745f2f4869865090e48794a779dd5650bfe902fe",
            "bytes": 3233196, "lines": 51667}

FILE = re.compile(r"^(\d{4})_[A-Za-z0-9._-]+\.md$")
_LABEL = r"[A-Z]{1,3}[0-9]*[a-z]?[0-9]?(?:\([a-z]\))?(?:\.[0-9]+)*(?:-[A-Za-z0-9]+)*"
HEADING = re.compile(r"^(#{1,4})\s+(.*)$")
AMENDMENT = re.compile(r"^Amendment\s+([A-Z]{1,3})(?=[\s.:—-]|$)")
SECTION = re.compile(rf"^({_LABEL})(?=[\s.:—-]|$)")
ALSO = re.compile(rf"^\s+and\s+({_LABEL})(?=[\s.:—-]|$)")
NUMBERED = re.compile(r"^([0-9]+)\.\s")
# `CT12.4-R (a).` / `CT12.4-R (b) and (c).`: the lettered part of a section that has several files
ITEM = re.compile(rf"^{_LABEL}\s+(\([a-z]\)(?:\s+and\s+\([a-z]\))?)[.:]")
# A section that opens with a dated bold marker instead of a heading: `**§CT29-R (2026-10-25, ...`
MARKER = re.compile(rf"^\*\*§({_LABEL})(?:, addendum)? \(20[0-9][0-9]-")


def labels_of(heading_text: str) -> list[str]:
    """The section labels a heading line carries: `Amendment AR` -> AR; `CT18-D9.` -> CT18-D9;
    `CT24-F-C3 and CT25-F-C1.` -> both; `8. Amendment policy` -> 8 (the original document's
    own numbering). A bare capitalised word (`THE RESULTS`) is not a label: outside an
    `Amendment` heading a label must carry a digit or a hyphen."""
    m = AMENDMENT.match(heading_text)
    if m:
        return [m.group(1)]
    m = NUMBERED.match(heading_text)
    if m:
        return [m.group(1)]
    m = SECTION.match(heading_text)
    if not m or not re.search(r"[0-9-]", m.group(1)):
        return []
    out = [m.group(1)]
    also = ALSO.match(heading_text[m.end():])
    if also:
        out.append(also.group(1))
    return out


def log_files(log_dir: Path = LOG_DIR) -> list[Path]:
    files = sorted(p for p in log_dir.iterdir() if FILE.match(p.name))
    seqs = [int(FILE.match(p.name).group(1)) for p in files]
    if seqs != list(range(len(files))):
        raise ValueError(f"sequence numbers under {log_dir} are not 0..{len(files) - 1} without a gap "
                         f"or a repeat")
    return files


def read_log_bytes(log_dir: Path = LOG_DIR) -> bytes:
    return b"".join(p.read_bytes() for p in log_files(log_dir))


def read_log(log_dir: Path = LOG_DIR) -> str:
    """The whole log as one text, in order -- what `PREREGISTRATION.md` used to hold."""
    return read_log_bytes(log_dir).decode("utf-8")


def file_table(log_dir: Path = LOG_DIR) -> list[dict]:
    """One row per file: name, bytes, lines, and its line range in the pre-split single file
    (None for a file added after the split)."""
    rows, off, line = [], 0, 0
    for p in log_files(log_dir):
        b = p.read_bytes()
        n = b.count(b"\n")
        pre = off < PRESPLIT["bytes"]
        rows.append({"file": p.name, "bytes": len(b), "lines": n, "offset": off,
                     "old_first": line + 1 if pre else None, "old_last": line + n if pre else None})
        off += len(b)
        line += n
    return rows


def check(log_dir: Path = LOG_DIR) -> list[str]:
    """Problems with the split, as sentences. Empty when the invariants hold."""
    bad = []
    try:
        rows = file_table(log_dir)
    except ValueError as e:
        return [str(e)]
    data = read_log_bytes(log_dir)
    if b"\r" in data:
        bad.append("a log file contains a carriage return")
    head = data[:PRESPLIT["bytes"]]
    if hashlib.sha256(head).hexdigest() != PRESPLIT["sha256"]:
        bad.append("the first %d bytes of the concatenation do not hash to the pre-split file's sha256: "
                   "a pre-split section was edited, removed or reordered" % PRESPLIT["bytes"])
    if PRESPLIT["bytes"] not in {r["offset"] + r["bytes"] for r in rows}:
        bad.append("no file boundary falls at the end of the pre-split text")
    for r in rows:
        if r["bytes"] == 0 or not (log_dir / r["file"]).read_bytes().endswith(b"\n"):
            bad.append(f"{r['file']} is empty or does not end with a newline")
    return bad


def heading_rows(log_dir: Path = LOG_DIR) -> list[dict]:
    """One row per heading line (levels 1-4) and per dated bold section marker (level 0, the
    form `**§CT29-R (2026-10-25, ...`, for a section the log opens without a heading), in log
    order."""
    out = []
    for r in file_table(log_dir):
        text = (log_dir / r["file"]).read_bytes().decode("utf-8")
        prev = None                                  # (line number, level) of the heading line before
        for i, ln in enumerate(text.split("\n"), 1):
            m = HEADING.match(ln)
            k = None if m else MARKER.match(ln)
            if not m and not k:
                continue
            level = len(m.group(1)) if m else 0
            # a title wrapped over several heading lines: only its first line can carry a label
            # (`## Amendment P -- ...` / `## O5 FAILS, so ...` is one title, and names P, not O5)
            wrapped = bool(m) and prev == (i - 1, level)
            if m:
                prev = (i, level)
            labels = [] if wrapped else (labels_of(m.group(2)) if m else [k.group(1)])
            item = ITEM.match(m.group(2)) if m and labels else None
            out.append({"labels": labels, "item": item.group(1) if item else "",
                        "level": level, "file": r["file"],
                        "line": i, "old_line": (r["old_first"] + i - 1) if r["old_first"] else None,
                        "heading": ln if m else ln[:160]})
    return out


def render_index(log_dir: Path = LOG_DIR, omitted: dict | None = None,
                 rel: str = "docs/record/preregistration") -> tuple[str, str]:
    """(the index's markdown, INDEX.tsv). `omitted` maps a file name to the reason it is not
    carried (the release export leaves some sections out); such a file keeps its row and its
    labels, and its headings are not listed."""
    omitted = omitted or {}
    files = file_table(log_dir)
    heads = heading_rows(log_dir)
    by_file: dict[str, list[str]] = {r["file"]: [] for r in files}
    for h in heads:
        for lab in h["labels"]:
            shown = f"{lab} {h['item']}" if h["item"] else lab
            if shown not in by_file[h["file"]]:
                by_file[h["file"]].append(shown)
    n_pre = sum(1 for r in files if r["old_first"])
    md = [
        "# Pre-registration log — index",
        "",
        "**The pre-registration log is the original pre-registration and its dated amendments, A to CT.** "
        f"It is kept in `{rel}/`: one file per amendment, and one file per section for amendment CT. "
        "This path held the whole log as a single file until 2026-11-09 (§CT18-D10). It is now the index, "
        "kept here because the record cites `PREREGISTRATION.md` §<label> throughout.",
        "",
        "- **Order.** File-name order is the log's order. Where anything in the repository disagrees with "
        "the log, the log wins, and its last section is the live record.",
    ]
    if not omitted:
        md += [
            f"- **Nothing was reworded.** The first {n_pre} files, concatenated in order, are byte-identical "
            f"to the single file as it stood when it was split: {PRESPLIT['bytes']:,} bytes, "
            f"{PRESPLIT['lines']:,} lines, sha256 `{PRESPLIT['sha256']}`. A test holds that hash.",
        ]
    else:
        md += [
            f"- **The text is the working repository's, with identifiers replaced.** There, the first {n_pre} "
            "files concatenate byte for byte to the single file as it stood when it was split "
            f"({PRESPLIT['bytes']:,} bytes, {PRESPLIT['lines']:,} lines). Here, account names, paths and "
            "other identifiers are replaced by placeholders and some sections are left out (`README_RECORD.md` §1 and §4), "
            "so that file's hash does not apply. No line was added or removed inside a file, so the line "
            "numbers below hold.",
        ]
    md += [
        "- **Finding a section.** The table below gives the file for every section label. "
        f"`{rel}/{INDEX_TSV_NAME}` has one row per heading, with its line in the file and its line in the "
        "old single file. `python -m src.analysis.prereg_log locate <label>` and `locate --line <n>` do the "
        "lookup; the old line ranges are what a citation of the form `PREREGISTRATION.md` line n resolves "
        "against.",
    ]
    if not omitted:
        md += [
            "- **Adding a section.** A new section is a new file with the next number, on `main`, and then "
            "`python -m src.analysis.prereg_log index --write`. Existing files are never edited (the "
            "amendment policy is in file 0008).",
        ]
    md += [
        "- **Reading it as one text.** `src.analysis.prereg_log.read_log()` returns the concatenation; the "
        "tools and tests that took this file as input read through it.",
    ]
    if omitted:
        md += ["- **Not in this release.** " + str(len(omitted)) + " sections are left out; each keeps a stub file of the same length and "
               "its row below, marked, and `README_RECORD.md` §4 gives the reason for each."]
    md += ["", "| file | lines in the old single file | section labels it holds |", "|---|---|---|"]
    for r in files:
        old = f"{r['old_first']}–{r['old_last']}" if r["old_first"] else "— (added after the split)"
        labs = ", ".join(by_file[r["file"]]) or "—"
        name = (f"`{r['file']}`" if r["file"] not in omitted
                else f"`{r['file']}` — not in this release")
        md.append(f"| {name} | {old} | {labs} |")
    tsv = ["label\tlevel\tfile\tline\told_line\theading"]
    for h in heads:
        if h["file"] in omitted:
            continue
        head = h["heading"].replace("\t", " ")
        for lab in h["labels"] or [""]:          # one row per label, so a label is looked up exactly
            tsv.append("\t".join([lab, str(h["level"]), h["file"], str(h["line"]),
                                  str(h["old_line"]) if h["old_line"] else "", head]))
    return "\n".join(md) + "\n", "\n".join(tsv) + "\n"


def locate(label: str | None = None, old_line: int | None = None,
           log_dir: Path = LOG_DIR) -> list[dict]:
    """Rows for a section label (every heading carrying it, in order) or for a line of the old
    single file (one row: the file and the line within it). Empty when nothing matches."""
    if old_line is not None:
        for r in file_table(log_dir):
            if r["old_first"] and r["old_first"] <= old_line <= r["old_last"]:
                return [{"file": r["file"], "line": old_line - r["old_first"] + 1, "old_line": old_line}]
        return []
    return [h for h in heading_rows(log_dir) if label in h["labels"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    ix = sub.add_parser("index")
    g = ix.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    lo = sub.add_parser("locate")
    lo.add_argument("label", nargs="?")
    lo.add_argument("--line", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "check":
        bad = check()
        for b in bad:
            print("PROBLEM:", b)
        print(f"{len(log_files())} files, {len(read_log_bytes()):,} bytes; "
              + ("OK" if not bad else f"{len(bad)} problem(s)"))
        return 1 if bad else 0
    if a.cmd == "index":
        md, tsv = render_index()
        tsv_path = LOG_DIR / INDEX_TSV_NAME
        if a.write:
            INDEX_MD.write_bytes(md.encode("utf-8"))
            tsv_path.write_bytes(tsv.encode("utf-8"))
            print(f"wrote {INDEX_MD.name} and {tsv_path.relative_to(REPO).as_posix()}")
            return 0
        stale = [p.name for p, want in ((INDEX_MD, md), (tsv_path, tsv))
                 if not p.exists() or p.read_bytes() != want.encode("utf-8")]
        print("index current" if not stale else f"STALE: {', '.join(stale)}")
        return 1 if stale else 0
    rows = locate(a.label, a.line)
    for r in rows:
        print(f"docs/record/preregistration/{r['file']}:{r['line']}"
              + (f"  (old line {r['old_line']})" if r.get("old_line") else "")
              + (f"  {r['heading'][:100]}" if "heading" in r else ""))
    if not rows:
        print("NOT FOUND")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
