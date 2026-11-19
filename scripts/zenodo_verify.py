"""After the Zenodo data record's files are uploaded and BEFORE it is published:
compare the draft's file list, as Zenodo reports it, with
`release/zenodo/DATA_UPLOAD_CHECKSUMS.tsv`.

    python scripts/zenodo_verify.py --from-json LISTING.json
    python scripts/zenodo_verify.py --deposit-id N [--sandbox]      (token: $ZENODO_TOKEN)

LISTING.json is the draft's file list as Zenodo returns it, either from
`GET /api/deposit/depositions/<id>/files` (a list of {filename, filesize,
checksum}, the checksum an md5 in hex) or in the newer form ({entries: [{key,
size, checksum: "md5:..."}]}); both are read. With --deposit-id the script
fetches the list itself, sending the token from $ZENODO_TOKEN; the token is
never printed or written.

Exit 0 only if the draft holds exactly the files the table names, each with
the table's size and md5. Every difference is printed: an absent file, an extra
file, a size or md5 that differs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def read_table(p: Path) -> dict:
    return {r["file"]: (int(r["bytes"]), r["md5"])
            for r in csv.DictReader(open(p, encoding="utf-8", newline=""), delimiter="\t")}


def read_listing(obj) -> dict:
    """{file name: (size, md5)} from either listing form."""
    entries = obj.get("entries", obj) if isinstance(obj, dict) else obj
    out = {}
    for e in entries:
        name = e.get("filename", e.get("key"))
        size = e.get("filesize", e.get("size"))
        ck = str(e.get("checksum", ""))
        if name is None or size is None or not ck:
            raise SystemExit(f"REFUSING: a listing entry lacks a name, size or checksum: {e}")
        if ck.startswith("md5:"):
            ck = ck[4:]
        elif ":" in ck:
            raise SystemExit(f"REFUSING: {name}: checksum {ck!r} is not md5")
        out[name] = (int(size), ck.lower())
    return out


def compare(want: dict, got: dict) -> list[str]:
    diffs = [f"ABSENT on Zenodo: {n}" for n in sorted(set(want) - set(got))]
    diffs += [f"EXTRA on Zenodo: {n}" for n in sorted(set(got) - set(want))]
    for n in sorted(set(want) & set(got)):
        if want[n][0] != got[n][0]:
            diffs.append(f"SIZE differs: {n}: Zenodo {got[n][0]}, table {want[n][0]}")
        if want[n][1] != got[n][1]:
            diffs.append(f"MD5 differs: {n}: Zenodo {got[n][1]}, table {want[n][1]}")
    return diffs


def fetch(deposit_id: int, sandbox: bool):
    import urllib.request
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        raise SystemExit("REFUSING: $ZENODO_TOKEN is not set")
    host = "sandbox.zenodo.org" if sandbox else "zenodo.org"
    req = urllib.request.Request(f"https://{host}/api/deposit/depositions/{deposit_id}/files",
                                 headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-json", type=Path)
    src.add_argument("--deposit-id", type=int)
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--table", type=Path,
                    default=REPO / "release" / "zenodo" / "DATA_UPLOAD_CHECKSUMS.tsv")
    a = ap.parse_args(argv)
    obj = (json.loads(a.from_json.read_text(encoding="utf-8")) if a.from_json
           else fetch(a.deposit_id, a.sandbox))
    want, got = read_table(a.table), read_listing(obj)
    diffs = compare(want, got)
    for n in sorted(want):
        print(f"{'OK  ' if n in got and got[n] == want[n] else 'DIFF'} {n}")
    for d in diffs:
        print(d)
    print(f"{len(want) - len({d.split(': ')[1] for d in diffs})} of {len(want)} files match; "
          + ("PUBLISH MAY PROCEED" if not diffs else "DO NOT PUBLISH"))
    return 0 if not diffs else 1


if __name__ == "__main__":
    sys.exit(main())
