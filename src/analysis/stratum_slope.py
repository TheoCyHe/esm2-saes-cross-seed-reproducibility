"""The design-3 alignment slope with `log M` recomputed on an evidence stratum
(PREREGISTRATION amendment CT, §CT5 / §CT12.1).

Only the alignment score changes: the match counts, `log10 phi`, `d_fs`, the
A4a floor and the fitter are the primary's, loaded through §CT3's
`jackknife_dictionaries.load_join_inputs` (the join's own helpers and identity
gate). The stratum's `M` comes from `standardize_gain` run on that stratum's AP
and null; its seed axis is aligned to the anchors BY LABEL (gotcha 29), never by
position. A feature whose stratum `M` is not positive cannot enter `log M`; the
count is reported and, if any exist, the run refuses rather than dropping them
silently (the declared outcome is then UNMEASURABLE with the count).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def aligned_logM(gtilde_path: Path, anchor_labels: list[str]) -> tuple[np.ndarray, dict]:
    """The stratum's `M`, aligned to the anchors' order by its stored labels,
    and its log. Refuses unlabelled axes and non-positive `M`."""
    from src.analysis.seed_axis import align_to_reference
    with h5py.File(gtilde_path, "r") as hf:
        if "seeds" not in hf:
            raise SystemExit(f"FATAL: {gtilde_path} has no seed labels (gotcha 29)")
        labels = [s.decode() if isinstance(s, bytes) else str(s) for s in hf["seeds"][:]]
        M = hf["M"][:]
        attrs = {k: (v.decode() if isinstance(v, bytes) else
                     (v.item() if hasattr(v, "item") else v))
                 for k, v in hf.attrs.items()}
    M = align_to_reference([s.split("_")[-1] for s in labels], M, anchor_labels)
    bad = ~(np.isfinite(M) & (M > 0))
    report = {"n_nonpositive_or_nonfinite_M": int(bad.sum()), "shape": list(M.shape)}
    if bad.any():
        raise SystemExit(f"UNMEASURABLE: {int(bad.sum())} feature-seed cells have "
                         f"a non-positive or non-finite stratum M; log M is undefined "
                         f"there and they are not dropped silently. {report}")
    return np.log(M), {**report, "gtilde_attrs": {k: attrs[k] for k in (
        "scored_row_set", "axis_values", "argmax_pool_n", "argmax_pool_scored_n",
        "argmax_pool_used_n", "admission_strata", "admission_min_protein_count",
        "stratum", "n_concepts", "inner_mode", "null_arm", "git_sha") if k in attrs},
        "description": stratum_description(gtilde_path)}


def stratum_description(gtilde_path: Path) -> dict:
    """§CT5's descriptive counts, read from the stratum's `standardize_gain`
    output: the scored pool, each stratum's concepts and positive labels on it,
    and how many pool concepts share each concept cell (K1's per-cell minimum is
    MIN_CONCEPTS_PER_STRATUM; reported, not gated)."""
    from src.analysis.null_pooling import MIN_CONCEPTS_PER_STRATUM
    with h5py.File(gtilde_path, "r") as hf:
        missing = [k for k in ("in_argmax_pool", "concept_cell") if k not in hf]
        if missing:
            return {"absent": missing}        # reported absent, never defaulted
        pool = hf["in_argmax_pool"][:].astype(bool)
        cell = hf["concept_cell"][:]
        out = {"n_pool_concepts_scored": int(pool.sum())}
        if "stratum_counts" in hf:
            g = hf["stratum_counts"]
            axis_pool = g["in_argmax_pool"][:].astype(bool)
            adm = (g["admitted"][:].astype(bool) if "admitted" in g
                   else np.ones_like(axis_pool))
            sel = axis_pool & adm
            out["n_pool_concepts_before_admission"] = int(axis_pool.sum())
            out["per_stratum_on_scored_pool"] = {}
            for t in g:
                if not isinstance(g[t], h5py.Group):
                    continue
                pc = g[t]["protein_count"][:]
                out["per_stratum_on_scored_pool"][t] = {
                    "n_concepts": int(sel.sum()),
                    "n_positive_labels": int(g[t]["n_positive_rows"][:][sel].sum()),
                    "n_pool_concepts_below_gate_before_admission": (
                        int((axis_pool & (pc < int(hf.attrs["admission_min_protein_count"]))).sum())
                        if "admission_min_protein_count" in hf.attrs else None)}
    cells, n = np.unique(cell[pool], return_counts=True)
    out["concept_cells_occupied"] = int(cells.size)
    out["concepts_per_occupied_cell"] = {"min": int(n.min()), "median": float(np.median(n)),
                                         "max": int(n.max())}
    out["pool_concepts_in_cells_below_K1_minimum"] = int(n[n < MIN_CONCEPTS_PER_STRATUM].sum())
    out["K1_min_concepts_per_stratum"] = MIN_CONCEPTS_PER_STRATUM
    return out


def cmd_fit(args) -> None:
    from src.analysis import jackknife_dictionaries as jk
    from src.sae.train import git_dirty, git_sha
    inp = jk.load_join_inputs(args)
    rows = np.arange(len(inp["anchor_labels"]))
    n_c = inp["sim"].shape[1] - 1
    y = jk.counts_from_sim(inp["sim"], args.theta)
    jk._W.update(inp)
    primary = jk._fit_one(y, rows, n_c)
    out = {"identity": {"measurement": "stratum_alignment_slope",
                        "governing_amendment": "CT5 / CT12.1", "theta": args.theta,
                        "arm": "primary_hard_bin", "likelihood": "betabinomial",
                        "units": "percentage points of r_f per SD of log M",
                        "git_sha": git_sha(), "git_dirty": git_dirty()},
           "full_score_primary": primary, "strata": {}}
    for name, path in zip(args.strata, args.gtilde_strata):
        logM, rep = aligned_logM(path, inp["anchor_labels"])
        jk._W.update({"logM": logM})
        r = jk._fit_one(y, rows, n_c)
        out["strata"][name] = {"fit": r, "gtilde": str(path), **rep}
        print(f"[stratum-slope] {name}: {r['fe_pp']:+.4f} pp "
              f"[{r['fe_ci95_pp'][0]:+.4f}, {r['fe_ci95_pp'][1]:+.4f}]", flush=True)
    jk._W.update({"logM": inp["logM"]})
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("--sim", "--rf-counts-all-theta", "--m1-json", "--gtilde",
                 "--null-reductions", "--d-fs", "--out"):
        ap.add_argument(name, type=Path, required=True)
    ap.add_argument("--degenerate-flag", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--theta", type=float, default=0.70)
    ap.add_argument("--strata", nargs="+", required=True)
    ap.add_argument("--gtilde-strata", nargs="+", type=Path, required=True)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if len(args.strata) != len(args.gtilde_strata):
        raise SystemExit("FATAL: one --gtilde-strata file per --strata name")
    if args.config is None:
        from src.experiment_config import DEFAULT_CONFIG_PATH
        args.config = DEFAULT_CONFIG_PATH
    cmd_fit(args)


if __name__ == "__main__":
    main()
