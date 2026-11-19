"""PREREGISTRATION amendment CT, §CT26 -- §CT22's locked subspace analysis on
the 20 comparison-only seeds (20-39), untouched by §CT22.

Two subcommands:

  `match` (GPU, rtx_5000_ada -- §CT3's device class): decoder-cosine MNN
      similarity for every ordered pair of the 40 dictionaries, by
      `rf_marginal.pair_statistics`, the function `jackknife_dictionaries
      match` used. GATE: the 20 anchor rows equal the stored
      `jackknife_match_ct.h5` `mnn_sim_anchor_rows` EXACTLY (bitwise), aligned
      by label -- a misordered seed axis or a different similarity fails it
      (§CT26-F item 4, replacing §CT22's gate 3).

  `run` (CPU): §CT22's lock (m = 3, Omega = mean cos^2, A(c, f)) on a seed
      set, with §CT22's cell functions UNCHANGED (`concept_subspace.eval_cell`,
      `controls`, `per_concept`, `cell_accounting`; the diagnostics'
      `task_rotation`, `task_other_concept`). Detection is §CT22-F item 1's
      definition with the pooled null over the seeds analysed; the concept set
      is the pool concepts detected in >= 2 of them; cells are enumerated as
      §CT22-F item 2 (pool order; unordered pairs, numeric labels; null seed
      20261024 + index). No tuning split (§CT26).

      CONTROL, run first and able to fail: the same code given the ANCHORS'
      inputs, with the 98 tuning concepts excluded from evaluation only (so the
      cell indices, hence the null seeds, are §CT22's), must reproduce
      `ct22_confirm_ct.json`'s per-concept list and `ct22_diagnostics_ct.json`'s
      `rotation` and `cross_concept.omega` EXACTLY (`--control-*`).
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import h5py
import numpy as np

from src.analysis import concept_subspace as cs
from src.analysis import concept_subspace_diag as dg

LOCK = {"m": 3, "measure": "omega", "alignment": "ap"}


def _dec(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def _sha16(p: Path) -> str:
    return cs.sha256_16(Path(p))


# ---------------------------------------------------------------------------
# match (GPU)
# ---------------------------------------------------------------------------

def cmd_match(args) -> None:
    import torch
    from src.analysis.artifact_identity import device_class
    from src.analysis.jackknife_dictionaries import mnn_sim_anchor_rows
    from src.analysis.rf_marginal import check_unit_norm, load_decoder
    from src.sae.train import git_dirty, git_sha

    def _num(p: Path) -> int:
        return int(str(p.name).rsplit("_", 1)[-1])

    ckpts = sorted(args.ckpts, key=_num) + sorted(args.compare_ckpts, key=_num)
    if len(ckpts) != 40:
        raise SystemExit(f"FATAL: {len(ckpts)} checkpoints, expected 40")
    decoders, labels = [], []
    for p in ckpts:
        W, meta = load_decoder(p, args.device)
        if check_unit_norm(W)["verdict"] != "PASS":
            raise SystemExit(f"FATAL: {p} decoder not unit-norm")
        decoders.append(W)
        labels.append(int(meta["seed"]))
    if len(set(labels)) != 40:
        raise SystemExit(f"FATAL: seeds not distinct: {labels}")
    run_dev = device_class(decoders[0])
    print(f"[ct26-match] 40 dictionaries on {run_dev}; labels {labels}", flush=True)
    sim = mnn_sim_anchor_rows(decoders, 40)            # every row, (40, 40, d)

    with h5py.File(args.stored_sim, "r") as hf:
        stored = hf["mnn_sim_anchor_rows"][:]
        s_rows = [int(s) for s in json.loads(hf.attrs["anchor_seed_labels"])]
        s_cols = [int(s) for s in json.loads(hf.attrs["column_seed_labels"])]
    ri = [labels.index(s) for s in s_rows]
    ci = [labels.index(s) for s in s_cols]
    mine = sim[np.ix_(ri, ci)]
    same = (mine == stored) | (np.isneginf(mine) & np.isneginf(stored))
    gate = {"cells": int(same.size), "cells_differing": int((~same).sum()),
            "verdict": "EXACT" if bool(same.all()) else "DIFFERS"}
    print(f"[ct26-match] GATE vs stored anchor rows: {gate}", flush=True)
    if gate["verdict"] != "EXACT":
        raise SystemExit(f"GATE FAILED: {gate}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as hf:
        hf.create_dataset("mnn_sim", data=sim, compression="gzip")
        hf.attrs.update({
            "measurement": "mnn_sim_all_40_rows", "governing_amendment": "CT26",
            "similarity": "decoder_cosine", "rule": "mnn",
            "row_seed_labels": json.dumps(labels),
            "column_seed_labels": json.dumps(labels),
            "gate_vs_jackknife_match_ct": json.dumps(gate),
            "stored_sim": str(args.stored_sim), "stored_sim_sha256_16": _sha16(args.stored_sim),
            "gather_device_class": str(run_dev), "git_sha": git_sha(),
            "git_dirty": json.dumps(git_dirty()), "torch": str(torch.__version__),
            "host": platform.node()})
    print(f"[ct26-match] wrote {args.out}", flush=True)


# ---------------------------------------------------------------------------
# run (CPU)
# ---------------------------------------------------------------------------

def read_sim(path: Path) -> tuple[np.ndarray, list[int], list[int]]:
    """Either the 40-row match file or §CT3's anchor-row file."""
    with h5py.File(path, "r") as hf:
        if "mnn_sim" in hf:
            return (hf["mnn_sim"][:], [int(s) for s in json.loads(hf.attrs["row_seed_labels"])],
                    [int(s) for s in json.loads(hf.attrs["column_seed_labels"])])
        return (hf["mnn_sim_anchor_rows"][:],
                [int(s) for s in json.loads(hf.attrs["anchor_seed_labels"])],
                [int(s) for s in json.loads(hf.attrs["column_seed_labels"])])


def enumerate_cells(detected: np.ndarray, labels: list[int]) -> tuple[list, list]:
    """§CT22-F item 2: concepts detected in >= 2 seeds, in pool order; within a
    concept, unordered pairs s < s' in NUMERIC label; 0-based over all cells."""
    population = [c for c in range(detected.shape[1]) if detected[:, c].sum() >= 2]
    by_label = sorted(range(len(labels)), key=lambda i: labels[i])
    cells = []
    for c in population:
        for a in range(len(by_label)):
            for b in range(a + 1, len(by_label)):
                s, t = by_label[a], by_label[b]
                if detected[s, c] and detected[t, c]:
                    cells.append((len(cells), c, s, t))
    return population, cells


def load(args) -> dict:
    from src.analysis.carrier_identity import (firing_bin, order_decoders_to,
                                               seed_label_to_int)
    from src.analysis.carrier_support import cell_index
    from src.analysis.concept_repro import (REFERENCE_PERCENTILE, detection_matrix,
                                            null_thresholds, observed_best_ap, to_gain)
    from src.analysis.rf_marginal import load_decoder

    with h5py.File(args.ap, "r") as hf:
        ap_seeds = [_dec(s) for s in hf["seeds"][:]]
        ap_concepts = [_dec(c) for c in hf["pooled/concepts"][:]]
        in_pool = hf["pooled/in_argmax_pool"][:].astype(bool)
        ap_pool = hf["pooled/ap"][:][:, :, in_pool]
        prevalence = hf["pooled/prevalence"][:][in_pool]
        fire_counts = hf["fire_counts"][:]
        n_rows = int(hf.attrs["n_rows"])
    with h5py.File(args.null_reductions, "r") as hf:
        null_seeds = [_dec(s) for s in hf["seeds"][:]]
        null_cmax = hf["null_cmax_b"][:]
        null_concepts = [_dec(c) for c in hf["concepts"][:]]
    if ap_seeds != null_seeds:
        raise SystemExit("FATAL: AP and null seed orders differ")
    pool = [c for c, k in zip(ap_concepts, in_pool) if k]
    if pool != null_concepts:
        raise SystemExit("FATAL: AP pool and null concepts differ")
    labels = [seed_label_to_int(s) for s in ap_seeds]
    detected = detection_matrix(to_gain(observed_best_ap(ap_pool), prevalence),
                                null_thresholds(null_cmax, REFERENCE_PERCENTILE, "pooled"))
    strat = cell_index(firing_bin(fire_counts.astype(np.float64) / float(n_rows)), None, 3)

    meta_seed, loaded = {}, {}
    for p in args.ckpts:
        Wd, meta = load_decoder(p, "cpu")
        meta_seed[p] = int(meta["seed"])
        loaded[p] = Wd.numpy().T.astype(np.float64)
    ordered = order_decoders_to(ap_seeds, meta_seed)
    if [meta_seed[p] for p in ordered] != labels:
        raise SystemExit("FATAL: decoder order differs from the AP labels")
    dec = np.stack([cs.unit_rows(loaded[p]) for p in ordered])
    del loaded

    sim, rows, cols = read_sim(args.sim)
    missing = [x for x in labels if x not in rows or x not in cols]
    if missing:
        raise SystemExit(f"FATAL: seeds {missing} absent from the similarity file")
    row_of = np.array([rows.index(x) for x in labels])
    col_of = np.array([cols.index(x) for x in labels])
    population, cells = enumerate_cells(detected, labels)
    return {"ap": ap_pool, "dec": dec, "strat": strat, "sim": sim,
            "row_of": row_of, "col_of": col_of, "labels": labels, "pool": pool,
            "population": population, "cells": cells, "detected": detected,
            "inputs": {k: {"path": str(getattr(args, k)), "sha256_16": _sha16(getattr(args, k))}
                       for k in ("ap", "null_reductions", "sim")}}


def cross_concept_tasks(all_cells: list, cells: list, m: int) -> list:
    """§CT22.5-D5 (a)'s draw, as `concept_subspace_diag` makes it: c' from the
    evaluated concepts detected in s', default_rng(SEED_DIAG + 1), cell order."""
    detected = {(c, x) for _, c, s, u in all_cells for x in (s, u)}
    conf = sorted({x[1] for x in cells})
    rng = np.random.default_rng(dg.SEED_DIAG + 1)
    det, tasks = {}, []
    for i, c, s, u in cells:
        if u not in det:
            det[u] = [cc for cc in conf if (cc, u) in detected]
        others = [cc for cc in det[u] if cc != c]
        if not others:
            # no other evaluated concept detected in s': nothing to draw. Skipped
            # and counted, and no draw is consumed (§CT22's diagnostic had no
            # such cell; there it would have raised).
            tasks.append((i, c, s, u, None, m, "cross"))
            continue
        tasks.append((i, c, s, u, others[int(rng.integers(len(others)))], m, "cross"))
    return tasks


def analyse(W: dict, cells: list, workers: int) -> dict:
    m, meas, al = LOCK["m"], LOCK["measure"], LOCK["alignment"]
    ctl = cs.controls(cells, m, al, workers)
    if ctl["self_cell"]["verdict"] != "PASS":
        raise SystemExit("CONTROL FAILED: self-cell")
    rows = cs.run_cells([(i, c, s, t, m, al, "real") for i, c, s, t in cells], workers)
    conc = cs.per_concept(rows, meas, W["pool"])
    acc = cs.cell_accounting(rows, meas)
    rot = dg._run(dg.task_rotation, [(i, c, s, u, m) for i, c, s, u in cells], workers)
    z = [r for r in rot if r["match_fraction"] == 0]
    rotation = {k: {"all": cs.quantiles([r[k] for r in rot]),
                    "match_fraction_0": cs.quantiles([r[k] for r in z])}
                for k in ("captured", "best_single", "beyond_best")}
    rotation["n_cells_match_fraction_0"] = len(z)
    ctasks = cross_concept_tasks(W["cells"], cells, m)
    cross = dg._run(dg.task_other_concept, [t for t in ctasks if t[4] is not None], workers)
    n_skip = sum(1 for t in ctasks if t[4] is None)
    cc = {"omega": dg._summ(cross)}
    if n_skip:
        cc["n_cells_without_another_concept"] = n_skip
    return {"rows": rows, "controls": ctl, "per_concept": conc, "accounting": acc,
            "rotation": rotation, "cross_concept": cc}


def score(res: dict) -> dict:
    """§CT26-F item 5."""
    acc = res["accounting"]
    unmeas = []
    if acc["share_null_sd_zero"] > 0.05:
        unmeas.append("null degenerate in more than 5% of cells")
    if acc["share_rank_deficient"] > 0.05:
        unmeas.append("rank-deficient feature sets in more than 5% of cells")
    if acc["share_draw_infeasible"] > 0.05:
        unmeas.append("matched-null draw infeasible in more than 5% of cells")
    has = [r for r in res["per_concept"] if "median_exceedance" in r]
    p1 = res["rotation"]["beyond_best"]["all"]["q50"]
    p2 = res["cross_concept"]["omega"]["q50"]
    p3 = (sum(r["median_exceedance"] >= 0.95 for r in has) / len(has)) if has else None
    out = lambda hit: "UNMEASURABLE" if unmeas else ("HIT" if hit else "MISS")  # noqa: E731
    return {"unmeasurable_causes": unmeas,
            "CT26_P1": {"median_rotation_excess": p1, "bar": "< 0.05", "outcome": out(p1 < 0.05)},
            "CT26_P2": {"median_different_concept_overlap": p2, "bar": "< 0.01",
                        "outcome": out(p2 < 0.01)},
            "CT26_P3": {"share_concepts_median_exceedance_ge_0_95": p3,
                        "n_concepts_with_a_measurable_cell": len(has), "bar": "> 0.9",
                        "outcome": out(p3 is not None and p3 > 0.9)}}


def control_compare(res: dict, confirm: dict, diag: dict) -> dict:
    """The anchor control: EXACT equality with §CT22's stored outputs."""
    def eq(a, b):
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    got = {"per_concept": eq(res["per_concept"], confirm["per_concept"]),
           "accounting": eq(res["accounting"], confirm["accounting"]),
           "rotation": eq(res["rotation"], diag["rotation"]),
           "cross_concept_omega": eq(res["cross_concept"]["omega"],
                                     diag["cross_concept"]["omega"])}
    return {**got, "verdict": "EXACT" if all(got.values()) else "DIFFERS"}


def cmd_run(args) -> None:
    from src.sae.train import git_dirty, git_sha
    if args.out.exists():
        raise SystemExit(f"FATAL: {args.out} exists")
    W = load(args)
    cs._W.clear()
    cs._W.update(W)
    excl = set()
    if args.exclude_concepts is not None:
        excl = set(json.loads(args.exclude_concepts.read_text(encoding="utf-8"))
                   ["tuning_concepts"])
    cells = [x for x in W["cells"] if W["pool"][x[1]] not in excl]
    print(f"[ct26] seeds {W['labels']}: population {len(W['population'])}, "
          f"{len(W['cells'])} cells, evaluated {len(cells)}", flush=True)
    res = analyse(W, cells, args.workers)
    out = {"identity": {"specification": "PREREGISTRATION amendment CT, §CT26",
                        "lock": LOCK, "git_sha": git_sha(), "git_dirty": git_dirty(),
                        "seed_base": cs.SEED_BASE, "n_null": cs.N_NULL,
                        "seed_diag_cross": dg.SEED_DIAG + 1, "device_class": "cpu",
                        "python": platform.python_version(), "numpy": np.__version__,
                        "ap_seed_labels": W["labels"], "inputs": W["inputs"],
                        "excluded_from_evaluation": sorted(excl)},
           "n_population": len(W["population"]),
           "n_never_detected": int((W["detected"].sum(axis=0) == 0).sum()),
           "n_detected_once": int((W["detected"].sum(axis=0) == 1).sum()),
           "n_cells_all": len(W["cells"]), "n_cells_evaluated": len(cells),
           "n_concepts_evaluated": len({x[1] for x in cells}),
           **{k: res[k] for k in ("controls", "accounting", "per_concept",
                                  "rotation", "cross_concept")},
           "crosstab": cs.crosstab([r for r in res["per_concept"]
                                    if "median_exceedance" in r])}
    if args.control_confirm is not None:
        ctl = control_compare(res, json.loads(args.control_confirm.read_text(encoding="utf-8")),
                              json.loads(args.control_diag.read_text(encoding="utf-8")))
        out["anchor_control"] = ctl
        print(f"[ct26] ANCHOR CONTROL {json.dumps(ctl)}", flush=True)
        cs.write(out, {"m3_ap": res["rows"]}, args.out)
        if ctl["verdict"] != "EXACT":
            raise SystemExit("CONTROL FAILED: §CT22's outputs not reproduced")
        return
    out.update(score(res))
    cs.write(out, {"m3_ap": res["rows"]}, args.out)
    print(f"[ct26] {json.dumps({k: out[k] for k in ('CT26_P1', 'CT26_P2', 'CT26_P3')})}",
          flush=True)


def build_parser() -> argparse.ArgumentParser:
    import os
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("match")
    m.add_argument("--ckpts", nargs="+", type=Path, required=True)
    m.add_argument("--compare-ckpts", nargs="+", type=Path, required=True)
    m.add_argument("--stored-sim", type=Path, required=True)
    m.add_argument("--device", default="cuda")
    m.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    for k in ("ap", "null_reductions", "sim", "out"):
        r.add_argument("--" + k.replace("_", "-"), dest=k, type=Path, required=True)
    r.add_argument("--ckpts", nargs="+", type=Path, required=True)
    r.add_argument("--exclude-concepts", type=Path, default=None,
                   help="CONTROL ONLY: docs/ct22_tuning_set.json")
    r.add_argument("--control-confirm", type=Path, default=None)
    r.add_argument("--control-diag", type=Path, default=None)
    r.add_argument("--workers", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "run" and (args.control_confirm is None) != (args.control_diag is None):
        raise SystemExit("--control-confirm and --control-diag go together")
    {"match": cmd_match, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
