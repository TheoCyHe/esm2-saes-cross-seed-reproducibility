"""Appendix E.2's ordering check, run against this repository's history
(amendment CT §CT18-D item 3: each artifact's OWN declared amendments).

    python scripts/e2_order_check.py ARTIFACT [ARTIFACT ...] [--json OUT]

For each artifact (a JSON file or an h5 file):
  * the RUN commit is the commit the artifact records -- `identity.git_sha`,
    `git_sha` or `producing_commit` (h5: the `git_sha` attribute);
  * the DECLARED sections are what the artifact names in `governing_amendment`,
    `amendment`, `specification` or `section` (in `identity`, at the top level,
    or as an h5 attribute) --
    every token that is the key of a heading in the pre-registration log
    ("## Amendment AR -- ...", "### CT19. ...", "#### CT28-F. ...");
    the log is `docs/record/preregistration/`, read as one text, and was the
    single file `PREREGISTRATION.md` until it was split (section CT18-D10);
  * each declared section's commit is the FIRST commit on the current branch
    whose diff adds that heading line (`git log -S <heading> --reverse`, over
    the old single file and the split directory, so the split moves no date), and
    the declaring commit is the latest of them;
  * the verdict is PRECEDES (the declaring commit is a proper ancestor of the
    run commit), SAME_COMMIT, DOES_NOT_PRECEDE, or UNDETERMINED -- no run
    commit, no declared section, no heading found for any token named, or a
    run commit not in this history. An UNDETERMINED row says which, and is
    never resolved by guessing.
Exit 0 when it ran; the verdicts are the output, not the exit status.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# where the log's headings have lived: the single file, then the split directory
LOG_PATHS = ("PREREGISTRATION.md", "docs/record/preregistration")
# where an artifact names the section that declared it, in the order read
DECLARED_KEYS = ("governing_amendment", "amendment", "specification", "section")
TOKEN = re.compile(r"\b[A-Z]{1,3}[0-9]*(?:\.[0-9]+)*(?:-[A-Z0-9]+)*\b")
HEAD = re.compile(r"^#{2,4}\s+(?:Amendment\s+)?([A-Z]{1,3}[0-9]*(?:\.[0-9]+)*(?:-[A-Z0-9]+)*)[\s.:—-]")


def git(*a: str, cwd: Path = REPO) -> str:
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def heading_index(prereg: str) -> dict[str, str]:
    """Section key -> the heading line, first occurrence."""
    out = {}
    for ln in prereg.splitlines():
        m = HEAD.match(ln)
        if m and m.group(1) not in out:
            out[m.group(1)] = ln
    return out


def read_artifact(path: Path) -> tuple[str | None, str | None]:
    """(run commit, declared-amendment text) as the artifact records them."""
    if path.suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as hf:
            at = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in hf.attrs.items()}
        sha = at.get("git_sha")
        dec = next((at.get(k) for k in DECLARED_KEYS if at.get(k)), None)
        if dec is None and "identity" in at:
            idn = json.loads(at["identity"])
            sha = sha or idn.get("git_sha")
            dec = next((idn.get(k) for k in DECLARED_KEYS if idn.get(k)), None)
        return (str(sha) if sha else None), (str(dec) if dec else None)
    d = json.loads(path.read_text(encoding="utf-8"))
    idn = d.get("identity", {}) if isinstance(d.get("identity"), dict) else {}
    sha = idn.get("git_sha") or d.get("git_sha") or d.get("producing_commit")
    dec = next((src.get(k) for src in (idn, d) for k in DECLARED_KEYS if src.get(k)), None)
    return (str(sha) if sha else None), (str(dec) if dec else None)


def declared_sections(text: str, index: dict) -> list[str]:
    return [t for t in dict.fromkeys(TOKEN.findall(text)) if t in index]


def first_commit_adding(line: str, cwd: Path = REPO) -> str | None:
    out = git("log", "-S", line, "--reverse", "--format=%H", "--", *LOG_PATHS, cwd=cwd)
    return out.split()[0] if out.split() else None


def is_ancestor(a: str, b: str, cwd: Path = REPO) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", a, b], cwd=cwd).returncode == 0


def check(path: Path, index: dict, cache: dict, cwd: Path = REPO) -> dict:
    run, dec = read_artifact(path)
    row = {"artifact": path.name, "run_commit": run, "declared": dec}
    if run is None:
        return {**row, "verdict": "UNDETERMINED", "why": "the artifact records no commit"}
    if dec is None:
        return {**row, "verdict": "UNDETERMINED", "why": "the artifact declares no amendment"}
    secs = declared_sections(dec, index)
    if not secs:
        return {**row, "verdict": "UNDETERMINED",
                "why": f"no heading in the pre-registration log for any token of {dec!r}"}
    try:
        run_full = git("rev-parse", "--verify", f"{run}^{{commit}}", cwd=cwd).strip()
    except subprocess.CalledProcessError:
        return {**row, "verdict": "UNDETERMINED", "why": f"run commit {run} is not in this history"}
    decl = {}
    for s in secs:
        if s not in cache:
            cache[s] = first_commit_adding(index[s], cwd)
        if cache[s] is not None:
            decl[s] = cache[s]
    if not decl:
        return {**row, "sections": secs, "verdict": "UNDETERMINED",
                "why": "no commit adds the named headings"}
    latest = None
    for s, c in decl.items():
        if latest is None or is_ancestor(decl[latest], c, cwd):
            latest = s
    c = decl[latest]
    if c == run_full:
        v = "SAME_COMMIT"
    elif is_ancestor(c, run_full, cwd):
        v = "PRECEDES"
    else:
        v = "DOES_NOT_PRECEDE"
    return {**row, "sections": secs, "declaring_section": latest, "declaring_commit": c,
            "run_commit_full": run_full, "verdict": v}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("artifacts", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(REPO))
    from src.analysis.prereg_log import read_log
    index = heading_index(read_log())
    cache: dict = {}
    rows = [check(p, index, cache) for p in a.artifacts]
    counts: dict = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        print(f"{r['verdict']:<17} {r['artifact']:<48} run {str(r['run_commit'])[:8]:<8} "
              f"declared by {r.get('declaring_section', '-')} {str(r.get('declaring_commit', ''))[:8]}"
              + (f"  ({r['why']})" if "why" in r else ""))
    print(json.dumps(counts))
    if a.json is not None:
        a.json.write_text(json.dumps({"counts": counts, "rows": rows}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
