"""PREREGISTRATION §CT22.3 -- draw the pilot's tuning set, once, and write it.

The population is §CT22.1's concept set as the author fixed it (2026-10-23): the
concepts of the 703-concept argmax pool detected in at least two anchor seeds
(650), in the AP artifact's pool order, read here from the release bundle's
`concepts_carriers.csv.gz` (`detected` = §AX12.2's pooled 95th-percentile
detection). The draw: `numpy.random.default_rng(20261024).choice(N,
size=floor(0.15 N + 0.5), replace=False)`, sorted.

`src/analysis/concept_subspace.py` re-derives the 650 from the AP artifact and
refuses unless they equal this file's population, element by element.

    python scripts/ct22_draw_tuning_set.py \
        --bundle release/derived/concepts_carriers.csv.gz \
        --out docs/ct22_tuning_set.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261024
FRACTION = 0.15
MIN_DETECTED_SEEDS = 2


def population(bundle: Path) -> list[str]:
    c = pd.read_csv(bundle)
    orders = [tuple(g["concept"]) for _, g in c.groupby("anchor_seed", sort=True)]
    if len(orders) != 20 or any(o != orders[0] for o in orders):
        raise SystemExit("FATAL: the bundle's concept order differs across seeds")
    k = c.groupby("concept", sort=False)["detected"].sum()
    return [x for x in orders[0] if int(k[x]) >= MIN_DETECTED_SEEDS]


def draw(n_population: int) -> np.ndarray:
    n = int(np.floor(FRACTION * n_population + 0.5))
    rng = np.random.default_rng(SEED)
    return np.sort(rng.choice(n_population, size=n, replace=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    pop = population(a.bundle)
    idx = draw(len(pop))
    out = {
        "specification": "PREREGISTRATION amendment CT, §CT22.3",
        "seed": SEED,
        "fraction": FRACTION,
        "draw": "numpy.random.default_rng(seed).choice(N, size=floor(0.15 N + 0.5), "
                "replace=False), sorted",
        "population_rule": "argmax-pool concepts detected in >= 2 of the 20 anchor "
                           "seeds (pooled 95th-percentile null, §AX12.2), AP pool order",
        "population_source": {"path": str(a.bundle).replace("\\", "/"),
                              "sha256": hashlib.sha256(a.bundle.read_bytes()).hexdigest()},
        "n_population": len(pop),
        "n_tuning": int(idx.size),
        "n_confirmatory": len(pop) - int(idx.size),
        "population": pop,
        "tuning_indices": idx.tolist(),
        "tuning_concepts": [pop[i] for i in idx],
    }
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"population {len(pop)}, tuning {idx.size}, wrote {a.out}")


if __name__ == "__main__":
    main()
