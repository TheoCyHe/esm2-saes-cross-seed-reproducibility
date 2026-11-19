"""
Offset-corrector scan -- MANDATORY gate before any headline result (CLAUDE.md
open Q5 / locked decisions).

WHAT A CORRECTOR IS. With `x_hat = W_dec.z + b_dec`, any constant offset the
bias cannot absorb (b_dec frozen at the wrong value, or simply not yet
converged) can instead be absorbed by a latent that fires on ~every token with
~constant magnitude: its contribution E[z_f]*d_f is then a fixed vector added to
every reconstruction. Such a latent is functioning as a BIAS, not as a feature.

WHY IT THREATENS THE HEADLINE. A corrector is an orphan (it matches no
annotation) that reproduces across seeds (every seed faces the same offset
incentive) and sits at the very top of the firing-rate distribution -- exactly
the bin where orphans are scarce and each matched pair carries outsized weight.
So it can move the firing-rate-matched aligned-vs-orphan comparison while being
invisible in an aggregate statistic. Hence: report per firing-rate BIN, and
verify the headline is robust to removing the candidates.

DETECTION. Per feature, over eval tokens:
  firing_rate  = P(z_f > 0)
  bias_fraction = E[z_f]^2 / E[z_f^2]   over ALL tokens
      -- 1.0 for a perfectly constant always-on latent; for a sparse latent it is
      bounded above by its firing rate (E[z]^2/E[z^2] = p * mu^2/(mu^2+s^2)), so
      this ONE scalar captures both "always on" and "constant magnitude".
  cos_offset   = cos(d_f, E[x] - b_dec)
      -- alignment with the offset the bias does not cover. Computed from x and
      b_dec only, so it is NOT circular w.r.t. feature f. (cos with the mean
      residual E[x - x_hat] is also reported, but note a converged model drives
      that to ~0 precisely BECAUSE the correctors are doing their job, so a small
      value there is not exoneration.)
A feature is flagged when firing_rate >= min_firing_rate AND
bias_fraction >= min_bias_fraction (thresholds in configs/mvp.yaml).

ROBUSTNESS. Flagged features are ablated (their z zeroed post-selection, which
is what "exclude from the analysis" means) and we report the change in mse and
in the mean-residual norm. A large mean-residual jump with a small mse jump is
the corrector signature: it was holding up a systematic DC offset, not
explaining token-specific variance.

Run over several seeds to see whether the same corrector-like population forms
reproducibly -- that is the property that would actually bias the comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from src.sae.train import TopKSAE

# Firing-rate bin edges. Deliberately fine at the TOP, where correctors live and
# where orphans are scarce; the matched-pair reporting must reuse these edges.
BIN_EDGES = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 5e-1, 1.0 + 1e-9]


def load_eval_sample(path: Path, n: int, seed: int = 0) -> torch.Tensor:
    with h5py.File(path, "r") as hf:
        total = hf["activations"].shape[0]
        n = min(n, total)
        idx = np.sort(np.random.default_rng(seed).choice(total, n, replace=False))
        return torch.from_numpy(hf["activations"][idx].astype(np.float32))


@torch.no_grad()
def scan_features(model: TopKSAE, X: torch.Tensor, device: str,
                  chunk: int = 8192) -> dict:
    """Streaming per-feature statistics over the token sample. O(dict_size)."""
    dict_size = model.W_enc.shape[0]
    d_model = model.b_dec.shape[0]
    n = 0
    n_fire = torch.zeros(dict_size, dtype=torch.float64, device=device)
    sum_z = torch.zeros(dict_size, dtype=torch.float64, device=device)
    sum_z2 = torch.zeros(dict_size, dtype=torch.float64, device=device)
    sum_x = torch.zeros(d_model, dtype=torch.float64, device=device)
    sum_resid = torch.zeros(d_model, dtype=torch.float64, device=device)
    sse = 0.0

    for i in range(0, X.shape[0], chunk):
        xb = X[i:i + chunk].to(device)
        x_hat, z, _ = model(xb)
        r = xb - x_hat
        n += xb.shape[0]
        n_fire += (z > 0).sum(0)
        zd = z.double()
        sum_z += zd.sum(0)
        sum_z2 += (zd * zd).sum(0)
        sum_x += xb.double().sum(0)
        sum_resid += r.double().sum(0)
        sse += float((r * r).sum())

    mean_z = sum_z / n
    mean_z2 = sum_z2 / n
    bias_fraction = torch.where(mean_z2 > 0, mean_z ** 2 / mean_z2.clamp_min(1e-30),
                                torch.zeros_like(mean_z))
    mean_x = sum_x / n
    resid_mean = sum_resid / n
    offset = mean_x - model.b_dec.double()          # E[x] - b_dec

    W = model.W_dec.double()                        # (d_model, dict_size)
    wn = W.norm(dim=0).clamp_min(1e-12)
    cos_offset = (W * offset.unsqueeze(1)).sum(0) / (wn * offset.norm().clamp_min(1e-12))
    cos_resid = (W * resid_mean.unsqueeze(1)).sum(0) / (wn * resid_mean.norm().clamp_min(1e-12))

    return {
        "n_tokens": n,
        "firing_rate": (n_fire / n).cpu().numpy(),
        "mean_z": mean_z.cpu().numpy(),
        "bias_fraction": bias_fraction.cpu().numpy(),
        # magnitude of the constant vector this latent adds to every reconstruction
        "dc_contribution": (mean_z * wn).cpu().numpy(),
        "cos_offset": cos_offset.cpu().numpy(),
        "cos_resid": cos_resid.cpu().numpy(),
        "mse": sse / (n * X.shape[1]),
        "resid_mean_norm": float(resid_mean.norm()),
        "offset_norm": float(offset.norm()),
    }


@torch.no_grad()
def ablate_and_measure(model: TopKSAE, X: torch.Tensor, drop: np.ndarray,
                       device: str, chunk: int = 8192) -> dict:
    """Zero the flagged latents post-selection and re-measure mse + mean residual.
    Post-hoc zeroing (not re-running TopK) is what excluding a feature from the
    analysis actually means."""
    mask = torch.ones(model.W_enc.shape[0], device=device)
    mask[torch.as_tensor(drop, dtype=torch.long, device=device)] = 0.0
    n, sse = 0, 0.0
    sum_resid = torch.zeros(model.b_dec.shape[0], dtype=torch.float64, device=device)
    for i in range(0, X.shape[0], chunk):
        xb = X[i:i + chunk].to(device)
        _, z, _ = model(xb)
        x_hat = (z * mask) @ model.W_dec.T + model.b_dec
        r = xb - x_hat
        n += xb.shape[0]
        sse += float((r * r).sum())
        sum_resid += r.double().sum(0)
    return {"mse": sse / (n * X.shape[1]),
            "resid_mean_norm": float((sum_resid / n).norm())}


def bin_counts(firing_rate: np.ndarray, mask: np.ndarray | None = None) -> list[dict]:
    """Feature counts per firing-rate bin. The matched-pair analysis MUST reuse
    these edges so a top-bin artifact cannot hide behind a coarser aggregate."""
    rows = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        in_bin = (firing_rate >= lo) & (firing_rate < hi)
        rows.append({
            "lo": lo, "hi": min(hi, 1.0),
            "n_features": int(in_bin.sum()),
            "n_flagged": int((in_bin & mask).sum()) if mask is not None else 0,
        })
    return rows


def scan_one(ckpt: Path, X: torch.Tensor, cfg: dict, device: str,
             min_firing: float, min_bias: float) -> dict:
    d_model = cfg["model"]["d_model"]
    dict_size, k = cfg["sae"]["dict_size"], cfg["sae"]["k"]
    sd = torch.load(ckpt / "model.pt", map_location="cpu")
    model = TopKSAE(d_model, dict_size, k, b_dec=torch.zeros(d_model)).to(device)
    model.load_state_dict(sd)
    model.eval()

    st = scan_features(model, X, device)
    flagged = (st["firing_rate"] >= min_firing) & (st["bias_fraction"] >= min_bias)
    idx = np.flatnonzero(flagged)

    out = {
        "ckpt": str(ckpt),
        "n_tokens": st["n_tokens"],
        "mse": st["mse"],
        "resid_mean_norm": st["resid_mean_norm"],
        "offset_norm": st["offset_norm"],
        "max_firing_rate": float(st["firing_rate"].max()),
        "max_bias_fraction": float(st["bias_fraction"].max()),
        "n_flagged": int(flagged.sum()),
        "flagged_features": idx.tolist(),
        "flagged_detail": [
            {"feature": int(i),
             "firing_rate": float(st["firing_rate"][i]),
             "bias_fraction": float(st["bias_fraction"][i]),
             "dc_contribution": float(st["dc_contribution"][i]),
             "cos_offset": float(st["cos_offset"][i]),
             "cos_resid": float(st["cos_resid"][i])}
            for i in idx[np.argsort(-st["bias_fraction"][idx])][:20]
        ],
        "bins": bin_counts(st["firing_rate"], flagged),
    }
    if len(idx):
        ab = ablate_and_measure(model, X, idx, device)
        out["ablation"] = {
            "mse_before": st["mse"], "mse_after": ab["mse"],
            "mse_increase_pct": 100 * (ab["mse"] - st["mse"]) / st["mse"],
            "resid_mean_norm_before": st["resid_mean_norm"],
            "resid_mean_norm_after": ab["resid_mean_norm"],
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--ckpts", type=Path, nargs="+", required=True,
                   help="Ensemble checkpoint dirs (one per seed).")
    p.add_argument("--eval-cache", type=Path, required=True)
    p.add_argument("--n-rows", type=int, default=200000)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    cs = cfg.get("confound_control", {}).get("corrector_scan", {})
    min_firing = cs.get("min_firing_rate", 0.5)
    min_bias = cs.get("min_bias_fraction", 0.5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU visible.")

    X = load_eval_sample(args.eval_cache, args.n_rows)
    print(f"corrector scan: {len(args.ckpts)} seeds, {X.shape[0]:,} eval tokens, "
          f"flag if firing_rate >= {min_firing} AND bias_fraction >= {min_bias}\n")

    results = [scan_one(c, X, cfg, device, min_firing, min_bias) for c in args.ckpts]

    print(f"{'seed ckpt':>22}  {'mse':>7}  {'max_fire':>8}  {'max_bias':>8}  "
          f"{'flagged':>7}  {'ablate Δmse%':>12}  {'‖r̄‖ before→after':>20}")
    for r in results:
        ab = r.get("ablation")
        ab_s = f"{ab['mse_increase_pct']:11.2f}%" if ab else " " * 12
        rn = (f"{ab['resid_mean_norm_before']:.3f} → {ab['resid_mean_norm_after']:.3f}"
              if ab else f"{r['resid_mean_norm']:.3f}")
        print(f"{Path(r['ckpt']).name:>22}  {r['mse']:7.3f}  "
              f"{r['max_firing_rate']:8.4f}  {r['max_bias_fraction']:8.4f}  "
              f"{r['n_flagged']:7d}  {ab_s}  {rn:>20}")

    n_with = sum(1 for r in results if r["n_flagged"] > 0)
    print(f"\nseeds with >=1 corrector candidate: {n_with}/{len(results)}")
    print("  (a corrector that forms in MOST seeds is a reproducing top-bin orphan"
          " -- the case that biases the headline)")

    print(f"\nfeature counts per firing-rate bin (flagged in parens) -- "
          f"the matched-pair analysis MUST reuse these bins:")
    print(f"  {'bin':>18}" + "".join(f"  {Path(r['ckpt']).name[-6:]:>10}" for r in results))
    for bi in range(len(BIN_EDGES) - 1):
        b0 = results[0]["bins"][bi]
        cells = "".join(f"  {r['bins'][bi]['n_features']:>6}"
                        f"({r['bins'][bi]['n_flagged']:>2})" for r in results)
        print(f"  [{b0['lo']:.0e},{b0['hi']:.0e})".rjust(20) + cells)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "min_firing_rate": min_firing, "min_bias_fraction": min_bias,
            "bin_edges": BIN_EDGES, "n_rows": int(X.shape[0]),
            "seeds": results,
        }, indent=2))
        print(f"\nwrote {args.out}")
        print("Downstream analyses MUST exclude each seed's flagged_features and "
              "report the headline with and without them.")


if __name__ == "__main__":
    main()
