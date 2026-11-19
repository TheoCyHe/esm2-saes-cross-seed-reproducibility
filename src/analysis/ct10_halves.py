"""Split the untruncated scored population into two halves by protein cluster
(PREREGISTRATION amendment CT, §CT10).

§CT10: "split into halves A and B by protein cluster (the cluster assignment
the eval split already uses), seeded assignment (seed 20261027), balanced to
within 2% of rows." The clusters are the mmseqs 30%-identity clusters of the
tsv the eval split was built from (`split_train_eval.py`; §CT-pre2 item 4). A
cluster's proteins are never split between halves.

THE ASSIGNMENT, a pure function of (clusters, rows per protein, seed): the
clusters are sorted by representative, permuted with
`np.random.default_rng(seed)`, and each in turn goes to the half with fewer
rows so far (ties to A). The result is then CHECKED, not assumed:
|rows(A) - rows(B)| <= 0.02 x total, or the split refuses (§CT-pre2's reading of
"within 2% of rows").

OUTPUT (h5): `ct10_half_a_rows`, `ct10_half_b_rows` -- ascending cache-row
indices, the form `feature_ap.load_scored_row_set` returns -- plus each half's
protein accessions, and attrs naming every input by path and sha256 prefix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

SEED = 20261027
TOLERANCE = 0.02
ROW_SETS = ("ct10_half_a", "ct10_half_b")


def assign_halves(cluster_of_protein: list[str], rows_of_protein: np.ndarray,
                  seed: int = SEED) -> np.ndarray:
    """0 (half A) or 1 (half B) per protein; whole clusters only."""
    rows = np.asarray(rows_of_protein, dtype=np.int64)
    if len(cluster_of_protein) != rows.size:
        raise ValueError("one cluster id per protein")
    clusters = sorted(set(cluster_of_protein))
    idx = {c: i for i, c in enumerate(clusters)}
    cid = np.array([idx[c] for c in cluster_of_protein], dtype=np.int64)
    rows_per_cluster = np.bincount(cid, weights=rows, minlength=len(clusters)).astype(np.int64)
    order = np.random.default_rng(seed).permutation(len(clusters))
    half_of_cluster = np.zeros(len(clusters), dtype=np.int8)
    load = [0, 0]
    for k in order:
        h = 0 if load[0] <= load[1] else 1
        half_of_cluster[k] = h
        load[h] += int(rows_per_cluster[k])
    return half_of_cluster[cid]


def balance_report(half: np.ndarray, rows_of_protein: np.ndarray,
                   tolerance: float = TOLERANCE) -> dict:
    rows = np.asarray(rows_of_protein, dtype=np.int64)
    a, b = int(rows[half == 0].sum()), int(rows[half == 1].sum())
    tot = a + b
    return {"rows_a": a, "rows_b": b, "rows_total": tot,
            "abs_difference": abs(a - b), "relative_difference": abs(a - b) / tot,
            "tolerance": tolerance, "within_tolerance": abs(a - b) <= tolerance * tot}


def _sha16(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main() -> None:
    from src.analysis.feature_ap import load_scored_row_set
    from src.annotations.build_residue_labels import header_to_acc, load_cluster_map
    from src.sae.train import git_dirty, git_sha
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--eval-cache", type=Path, required=True)
    p.add_argument("--clusters-tsv", type=Path, required=True)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    pop = load_scored_row_set(args.labels, "label_store")
    with h5py.File(args.eval_cache, "r") as hf:
        raw = hf["index_json"][()]
    raw = raw.decode() if isinstance(raw, bytes) else str(raw)
    index = json.loads(raw)
    n_cache = sum(e["n_rows"] for e in index)
    in_pop = np.zeros(n_cache, dtype=bool)
    in_pop[pop] = True
    # a protein is in the population when ALL its rows are (the label store
    # selects whole proteins); a partial protein would be a construction error
    prots, rows_of, starts = [], [], []
    for e in index:
        sl = in_pop[e["start"]:e["start"] + e["n_rows"]]
        if sl.any():
            if not sl.all():
                raise SystemExit(f"FATAL: {e['accession']} is partly in the population")
            prots.append(header_to_acc(e["accession"]))
            rows_of.append(int(e["n_rows"]))
            starts.append(int(e["start"]))
    rows_of = np.array(rows_of, dtype=np.int64)
    if int(rows_of.sum()) != pop.size:
        raise SystemExit(f"FATAL: proteins cover {rows_of.sum()} rows, population {pop.size}")
    cmap = load_cluster_map(args.clusters_tsv)
    missing = [a for a in prots if a not in cmap]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} proteins have no cluster, e.g. {missing[:3]}")
    clus = [cmap[a] for a in prots]
    half = assign_halves(clus, rows_of, args.seed)
    bal = balance_report(half, rows_of)
    print(json.dumps(bal), flush=True)
    if not bal["within_tolerance"]:
        raise SystemExit("REFUSED: the halves are not within 2% of rows (§CT10)")

    out_rows = {}
    for h, name in enumerate(ROW_SETS):
        sel = np.flatnonzero(half == h)
        r = np.concatenate([np.arange(starts[i], starts[i] + rows_of[i]) for i in sel])
        out_rows[name] = np.sort(r)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as ho:
        for h, name in enumerate(ROW_SETS):
            ho.create_dataset(f"{name}_rows", data=out_rows[name], compression="gzip")
            ho.create_dataset(f"{name}_proteins", data=np.array(
                [prots[i] for i in np.flatnonzero(half == h)], dtype=h5py.string_dtype()))
        ho.attrs.update({
            "measurement": "ct10_cluster_halves", "governing_amendment": "CT10",
            "seed": args.seed, "tolerance": TOLERANCE, **{k: v for k, v in bal.items()
                                                          if k != "within_tolerance"},
            "n_proteins": len(prots), "n_proteins_a": int((half == 0).sum()),
            "n_proteins_b": int((half == 1).sum()), "n_clusters": len(set(clus)),
            "n_clusters_a": len({c for c, h in zip(clus, half) if h == 0}),
            "n_clusters_b": len({c for c, h in zip(clus, half) if h == 1}),
            "population": "label_store", "population_n_rows": int(pop.size),
            "labels": str(args.labels), "eval_cache": str(args.eval_cache),
            "clusters_tsv": str(args.clusters_tsv),
            "clusters_tsv_sha256_16": _sha16(args.clusters_tsv),
            "git_sha": git_sha(), "git_dirty": git_dirty()})
    print(f"[ct10-halves] {len(prots)} proteins, {len(set(clus))} clusters -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
