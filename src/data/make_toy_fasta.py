"""
Build data/toy_200.fasta for the stage-1 smoke test.

Pulls ~200 short (50-300 residue), reviewed Swiss-Prot sequences from the
UniProt REST API. Run this on the login node (needs internet egress);
compute nodes on most clusters, including Bouchet, do not have it. The
resulting fasta is a local artifact used only to sanity-check
src/embeddings/cache.py -- it is not a stand-in for the real Swiss-Prot
train/eval split (see CLAUDE.md stage 2).
"""

from __future__ import annotations

import argparse
import urllib.parse
import urllib.request
from pathlib import Path

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"


def fetch_fasta(n: int, min_len: int, max_len: int) -> str:
    query = f"reviewed:true AND length:[{min_len} TO {max_len}]"
    params = {
        "query": query,
        "format": "fasta",
        "size": str(n),
        "sort": "accession asc",
    }
    url = f"{UNIPROT_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read().decode("utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/toy_200.fasta"))
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--min-len", type=int, default=50)
    p.add_argument("--max-len", type=int, default=300)
    args = p.parse_args()

    text = fetch_fasta(args.n, args.min_len, args.max_len)
    n_records = text.count(">")
    if n_records == 0:
        raise RuntimeError("UniProt returned no records -- check network/query.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {n_records} sequences -> {args.out}")


if __name__ == "__main__":
    main()
