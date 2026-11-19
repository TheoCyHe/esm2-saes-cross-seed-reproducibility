"""ONE-TIME migration: re-label the row-index cache for amendment AE's rename.

WHY THIS EXISTS, AND IT IS A GAP IN AE's OWN BLAST-RADIUS SCAN. That scan
covered `$PROJECT/analysis/*.h5`, `$SCRATCH/analysis/*.h5` and
`$SCRATCH/null_raw/*.h5` -- and NOT `$SCRATCH/row_index_cache/`. So 63 cached
row indices kept the retired attr name `row_subset` with the retired value
`population`, and `smooth_mu.resolve_row_index` -- which matches on IDENTITY
ATTRS rather than on the filename key -- found nothing:

    FATAL: no row index for seed_0 matching (ckpt ..., subset label_store, ...)
    Refusing to substitute a differently-scoped index -- a footprint computed
    on the wrong population is a wrong covariate.

TWO MISMATCHES AT ONCE, which is why it could not accidentally pass: the attr
NAME is wrong (`row_subset`, so `a.get("scored_row_set")` is None) AND the value
is wrong (`population`). THE GUARD WORKED -- it refused rather than substituting
a differently-scoped index, so no covariate was computed on the wrong rows.
Found by RUNNING the smooth_mu regeneration, not by reading code.

WHY RE-LABELLING IS CORRECT AND NOT A FUDGE. `label_store` and `population` name
the SAME row set -- the label store's own rows -- so every cached index is
content-valid and only its label is stale. The alternative is a GPU pass over
the eval cache per seed to reproduce byte-identical arrays.

No file is renamed: the lookup is by attrs, so the filename key is irrelevant
(it folds in a git SHA precisely so that it is NOT the matching criterion).

    python -m migrations.ae_rename_row_index_cache --cache-dir <dir> [--apply]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py

OLD_KEY, NEW_KEY = "row_subset", "scored_row_set"
VALUE_MAP = {"population": "label_store",
             "all": "full_eval",
             "shared_experimental": "shared_experimental",
             # already-migrated values pass through unchanged
             "label_store": "label_store",
             "full_eval": "full_eval"}


def _dec(v):
    return v.decode() if isinstance(v, bytes) else v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    files = sorted(args.cache_dir.glob("row_index_*.h5"))
    print(f"{len(files)} cached row indices in {args.cache_dir}")

    plan, skip, refuse = [], [], []
    for f in files:
        with h5py.File(f, "r") as hf:
            old = _dec(hf.attrs.get(OLD_KEY))
            new = _dec(hf.attrs.get(NEW_KEY))
        if new is not None and old is None:
            skip.append((f, new)); continue
        if old is None:
            refuse.append(f"{f.name}: neither {OLD_KEY} nor {NEW_KEY} present")
            continue
        if old not in VALUE_MAP:
            refuse.append(f"{f.name}: {OLD_KEY}={old!r} is not a declared value")
            continue
        plan.append((f, old, VALUE_MAP[old]))

    print(f"  to re-label: {len(plan)}   already new: {len(skip)}   "
          f"refused: {len(refuse)}")
    for r in refuse:
        print("   REFUSE " + r)
    if refuse:
        raise SystemExit(1)

    from collections import Counter
    print("  value mapping: " + str(dict(Counter(
        f"{o} -> {n}" for _, o, n in plan))))

    if not args.apply:
        for f, o, n in plan[:4]:
            print(f"    {f.name}: {OLD_KEY}={o!r} -> {NEW_KEY}={n!r}")
        print("DRY RUN -- nothing written. Re-run with --apply.")
        return

    bad = []
    for f, old, new in plan:
        with h5py.File(f, "r+") as hf:
            before = {k: _dec(hf.attrs[k]) for k in hf.attrs}
            hf.attrs[NEW_KEY] = new
            del hf.attrs[OLD_KEY]
        # verify both directions, and that nothing ELSE moved
        with h5py.File(f, "r") as hf:
            after = {k: _dec(hf.attrs[k]) for k in hf.attrs}
        if OLD_KEY in after:
            bad.append(f"{f.name}: {OLD_KEY} survives")
        if after.get(NEW_KEY) != new:
            bad.append(f"{f.name}: {NEW_KEY} is {after.get(NEW_KEY)!r}")
        for k in set(before) | set(after):
            if k in (OLD_KEY, NEW_KEY):
                continue
            if before.get(k) != after.get(k):
                bad.append(f"{f.name}: {k} changed unexpectedly")
    print(f"re-labelled {len(plan) - len(bad)}/{len(plan)}")
    if bad:
        for b in bad:
            print("   " + b)
        raise SystemExit(1)
    print("every index reports the new key, the old key is gone, "
          "and no other attr moved")


if __name__ == "__main__":
    main()
