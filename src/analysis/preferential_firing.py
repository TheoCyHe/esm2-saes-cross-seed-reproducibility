"""
D6 -- the preferential-firing check (PREREGISTRATION amendment D6).

MOTIVATION. A feature that beats its provenance-blind cap in the experimental
stratum (amendment C3) is, by construction, selective for WHICH residues were
studied experimentally -- and amendment A1 already measured that the
shared-experimental protein set is 84.5% eukaryotic against 36.2% in the full
eval set. "Detects well-studied/model-organism protein context" is therefore a
live alternative to "detects the biology" for any experimental-stratum
alignment, not a hypothetical to wave off.

THE CHECK. For a candidate feature aligned to concept `c`, compare its mean
activation on `c`'s EXPERIMENTAL-positive rows against its mean activation on
`c`'s PROPAGATED-positive rows (same concept, disjoint evidence tiers, both
restricted to the analysed subset). A real detector does not know which tier
labelled a residue, so it should fire comparably on both. A feature whose
activation is markedly higher on experimental- than propagated-positive rows is
exhibiting the signature of a confound tracking STUDY INTENSITY rather than
chemistry.

This does NOT auto-exclude features -- it is a flagged, reported diagnostic
run before any experimental-stratum alignment is interpreted as biological
(same gate role as `corrector_scan.py` plays for the headline).

Candidate features are those clearing the normalised-gain proxy at their
arg-max concept (no FDR needed for a screening pass; the same proxy used
throughout the amendment-C/D diagnostics), read from an existing feature-AP
matrix (`feature_ap.py`'s output) so this module needs only ONE additional
activation pass, not a re-derivation of alignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from src.analysis.artifact_identity import PARAMETER_VALUES
from src.analysis.feature_ap import TIER_GROUP_MAP, load_labels, load_scored_row_set
from src.sae.train import TopKSAE, git_dirty, git_sha


# --------------------------------------------------------------------- core --

def masked_mean_activation(
    feat: torch.Tensor,          # (nnz,) feature id per non-zero
    pos: torch.Tensor,           # (nnz,) row position (subset-relative)
    val: torch.Tensor,           # (nnz,) activation value
    keep_row: torch.Tensor,      # (n_rows,) bool: rows to average over
    n_features: int,
) -> torch.Tensor:
    """Per-feature mean activation over exactly the rows in `keep_row`.

    Non-firing rows contribute 0, so this is sum(z_f over kept firing rows) /
    count(kept rows) -- the ordinary "how hard does this feature fire, on
    average, over this row set" statistic. Independent of any sort order.
    """
    m = keep_row[pos]
    f2, v2 = feat[m], val[m]
    s = torch.zeros(n_features, dtype=torch.float64, device=feat.device)
    s.index_add_(0, f2, v2.to(torch.float64))
    n = int(keep_row.sum().item())
    return s / max(n, 1)


@torch.no_grad()
def gather_with_values(
    model: TopKSAE, cache_path: Path, rows: np.ndarray | None,
    device: str, chunk: int = 16384,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Like `feature_ap.sparse_activations` but keeps activation VALUES (needed
    for a mean; the AP path only needs presence/rank and drops them)."""
    feats_l, pos_l, vals_l = [], [], []
    with h5py.File(cache_path, "r") as hf:
        ds = hf["activations"]
        total = ds.shape[0] if rows is None else len(rows)
        for start in range(0, total, chunk):
            if rows is None:
                sl = slice(start, min(start + chunk, total))
                x = torch.from_numpy(ds[sl].astype(np.float32))
                positions = torch.arange(sl.start, sl.stop)
            else:
                sel = rows[start:start + chunk]
                x = torch.from_numpy(ds[sel].astype(np.float32))
                positions = torch.arange(start, start + len(sel))
            x = x.to(device)
            _, z, _ = model(x)
            nz = z.nonzero(as_tuple=False)
            feats_l.append(nz[:, 1].to(torch.int64).cpu())
            pos_l.append(positions[nz[:, 0].cpu()].to(torch.int64))
            vals_l.append(z[nz[:, 0], nz[:, 1]].to(torch.float32).cpu())
    return torch.cat(feats_l), torch.cat(pos_l), torch.cat(vals_l)


# ------------------------------------------------------------------ driver --

def candidate_features(ap_path: Path, stratum: str, gain_floor: float
                       ) -> tuple[list[str], dict[str, np.ndarray]]:
    """{concept: feature ids whose arg-max is this concept AND clear the
    normalised-gain proxy}, per seed averaged is not meaningful here -- this
    returns the UNION across seeds per concept, since a feature id means a
    different feature in each seed's dictionary but we scan seed-by-seed
    downstream and just need "is this (seed, feature) worth checking"."""
    with h5py.File(ap_path, "r") as hf:
        concepts = [c.decode() if isinstance(c, bytes) else c
                    for c in hf[stratum]["concepts"][:]]
        ap = hf[stratum]["ap"][:].astype(np.float64)          # (seeds, feats, concepts)
        prev = hf[stratum]["prevalence"][:]
    gain = (ap - prev[None, None, :]) / (1.0 - prev[None, None, :])
    # G0 fix: argmax on the GAIN array, not raw AP -- raw AP is inflated by
    # prevalence, so it can attribute a feature to a broad concept even when
    # a narrower one has the higher gain. `passes` was already correct
    # (computed on `gain`); only the concept IDENTITY attached to a passing
    # feature was wrong.
    argmax = gain.argmax(axis=2)                               # (seeds, feats)
    passes = gain.max(axis=2) >= gain_floor                    # (seeds, feats)
    return concepts, {"argmax": argmax, "passes": passes}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--ckpts", type=Path, nargs="+", required=True)
    p.add_argument("--eval-cache", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--ap", type=Path, required=True,
                   help="feature_ap.py output (experimental stratum) to read "
                        "candidate (seed, feature, concept) triples from.")
    p.add_argument("--stratum", default="experimental")
    p.add_argument("--gain-floor", type=float, default=0.05)
    p.add_argument("--flag-ratio", type=float, default=3.0,
                   help="flag a candidate if mean(exp-positive activation) / "
                        "mean(propagated-positive activation) exceeds this.")
    p.add_argument("--scored-row-set", default="shared_experimental",
                   # label_store is deliberately unsupported here: this script has no
                   # row->protein remap, so it cannot honour it.
                   choices=[v for v in PARAMETER_VALUES["scored_row_set"]
                            if v != "label_store"])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    d_model = cfg["model"]["d_model"]
    dict_size, k = cfg["sae"]["dict_size"], cfg["sae"]["k"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU visible.")

    concepts, cand = candidate_features(args.ap, args.stratum, args.gain_floor)
    n_seeds, n_features = cand["argmax"].shape
    print(f"{cand['passes'].sum():,} candidate (seed, feature) pairs across "
          f"{len(concepts)} concepts (G >= {args.gain_floor})")

    subset = load_scored_row_set(args.labels, args.scored_row_set)
    n_rows = len(subset) if subset is not None else \
        int(h5py.File(args.labels, "r").attrs["n_rows"])
    remap = None
    if subset is not None:
        remap = -np.ones(int(subset.max()) + 2, dtype=np.int64)
        remap[subset] = np.arange(len(subset))

    def to_subset(idx: np.ndarray) -> np.ndarray:
        if remap is None:
            return idx
        a = idx[idx < len(remap)]
        m = remap[a]
        return m[m >= 0]

    exp_labels, _ = load_labels(args.labels, "experimental")
    prop_labels, _ = load_labels(args.labels, "propagated")
    exp_labels = {c: to_subset(v) for c, v in exp_labels.items() if c in concepts}
    prop_labels = {c: to_subset(v) for c, v in prop_labels.items() if c in concepts}

    def mask(idx: np.ndarray) -> torch.Tensor:
        return torch.zeros(n_rows, dtype=torch.bool, device=device).index_fill_(
            0, torch.from_numpy(idx).to(device), True)

    exp_mask = {c: mask(v) for c, v in exp_labels.items() if v.size}
    prop_mask = {c: mask(v) for c, v in prop_labels.items() if v.size}

    results = []
    for si, ck in enumerate(args.ckpts):
        sd = torch.load(ck / "model.pt", map_location="cpu")
        model = TopKSAE(d_model, dict_size, k, b_dec=torch.zeros(d_model)).to(device)
        model.load_state_dict(sd)
        model.eval()
        feat, pos, val = gather_with_values(model, args.eval_cache, subset, device)
        feat, pos, val = feat.to(device), pos.to(device), val.to(device)

        seed_flagged = []
        cand_feats = np.flatnonzero(cand["passes"][si])
        for fidx in cand_feats:
            c = concepts[cand["argmax"][si, fidx]]
            if c not in exp_mask or c not in prop_mask:
                continue                     # concept has no positives in one tier here
            fmask = (feat == fidx)
            f_sub, p_sub, v_sub = feat[fmask], pos[fmask], val[fmask]
            exp_mean = masked_mean_activation(f_sub, p_sub, v_sub, exp_mask[c], dict_size)[fidx].item()
            prop_mean = masked_mean_activation(f_sub, p_sub, v_sub, prop_mask[c], dict_size)[fidx].item()
            ratio = exp_mean / prop_mean if prop_mean > 0 else float("inf")
            flagged = ratio >= args.flag_ratio
            seed_flagged.append({
                "feature": int(fidx), "concept": c,
                "exp_mean_activation": exp_mean, "prop_mean_activation": prop_mean,
                "ratio": ratio, "flagged": flagged,
            })
        n_flag = sum(r["flagged"] for r in seed_flagged)
        print(f"  seed {ck.name}: {len(seed_flagged)} candidates checked, "
              f"{n_flag} flagged (ratio >= {args.flag_ratio})", flush=True)
        results.append({"ckpt": str(ck), "candidates": seed_flagged})
        del feat, pos, val
        torch.cuda.empty_cache() if device == "cuda" else None

    ratios = np.array([r["ratio"] for res in results for r in res["candidates"]
                       if np.isfinite(r["ratio"])])
    if ratios.size:
        print(f"\nratio distribution across {ratios.size} finite-ratio candidates: "
              f"median {np.median(ratios):.2f}  p90 {np.percentile(ratios, 90):.2f}  "
              f"max {ratios.max():.2f}")
    total_flagged = sum(1 for res in results for r in res["candidates"] if r["flagged"])
    total_cand = sum(len(res["candidates"]) for res in results)
    print(f"TOTAL: {total_flagged}/{total_cand} candidates flagged "
          f"(preferential firing on experimental-positive rows)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "stratum": args.stratum, "gain_floor": args.gain_floor,
        "flag_ratio": args.flag_ratio, "scored_row_set": args.scored_row_set,
        "n_rows": n_rows, "concepts": concepts,
        "git_sha": git_sha(), "git_dirty": bool(git_dirty()),
        "seeds": results,
    }, indent=2))
    print(f"wrote {args.out}")
    print("Report the ratio DISTRIBUTION, not just the flagged count -- this is "
          "a diagnostic to run before interpreting alignment as biological, not "
          "an automatic exclusion (amendment D6).")


if __name__ == "__main__":
    main()
