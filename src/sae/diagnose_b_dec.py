"""
Decide freeze-vs-trainable for b_dec from pilot checkpoints (CLAUDE.md open Q5).

A nonzero cross-seed spread in the converged b_dec is consistent with two
opposite regimes, and freezing is only safe in one:
  (a) optimizer scatter around ONE shared optimum  -> freezing at the pooled
      mean is harmless nuisance removal;
  (b) genuinely seed-specific optima (b* = E[x] - W_dec.E[z], and W_dec/E[z]
      differ by seed) -> freezing bakes an uncorrectable per-seed offset into
      every ensemble run and re-creates the corrector incentive per seed.

"Spread is nonzero" does not distinguish them. This computes the tests that do:

  CHECK 2 (decisive): reconstruction mse for each seed's trained model using
    (i) its OWN b_dec vs (ii) the POOLED-mean b_dec. Under (a) the mse barely
    changes; under (b) it degrades. (Swap-at-eval is an UPPER bound on the harm
    -- a frozen-from-start model could partly adapt -- so a small delta here
    means freezing is definitely fine.)

  RESIDUAL-RELATIVE METRIC: ||b_dec_seed - pooled|| / mean(||x - x_hat||), the
    number that actually predicts corrector formation (offset vs residual, not
    vs ||b_dec||).

  CHECK 3 (structure): SVD of the per-seed deviation vectors. Isotropic ->
    (a)-like; concentrated in a low-dim subspace -> (b). (With few seeds this is
    weak; reported for completeness.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from src.sae.train import TopKSAE


def load_eval_sample(path: Path, n: int, seed: int = 0) -> torch.Tensor:
    """A representative sample of eval activations (sorted random rows)."""
    with h5py.File(path, "r") as hf:
        total = hf["activations"].shape[0]
        n = min(n, total)
        idx = np.sort(np.random.default_rng(seed).choice(total, n, replace=False))
        x = hf["activations"][idx].astype(np.float32)
    return torch.from_numpy(x)


@torch.no_grad()
def recon_mse(model: TopKSAE, X: torch.Tensor, device: str, chunk: int = 8192) -> float:
    """Mean squared reconstruction error (d_model_mean) over X."""
    sse, n_elem = 0.0, 0
    for i in range(0, X.shape[0], chunk):
        xb = X[i:i + chunk].to(device)
        x_hat, _, _ = model(xb)
        sse += float(((x_hat - xb) ** 2).sum())
        n_elem += xb.numel()
    return sse / n_elem


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--pilot-ckpts", type=Path, nargs="+", required=True)
    p.add_argument("--eval-cache", type=Path, required=True)
    p.add_argument("--n-rows", type=int, default=100000)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    d_model = cfg["model"]["d_model"]
    dict_size = cfg["sae"]["dict_size"]
    k = cfg["sae"]["k"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU visible.")

    sds = [torch.load(c / "model.pt", map_location="cpu") for c in args.pilot_ckpts]
    b_decs = torch.stack([sd["b_dec"].float() for sd in sds])   # (N, d_model)
    pooled = b_decs.mean(dim=0)

    X = load_eval_sample(args.eval_cache, args.n_rows)
    print(f"{len(args.pilot_ckpts)} seeds, {X.shape[0]:,} eval rows, d_model={d_model}\n")

    print("CHECK 2 -- reconstruction cost of freezing at the pooled b_dec:")
    print(f"  {'seed':>18}  {'mse_own':>9}  {'mse_pooled':>10}  {'Δmse':>7}  "
          f"{'Δmse %':>7}  {'‖Δb_dec‖':>9}  {'‖resid‖':>8}  {'Δb/resid':>8}")
    ratios = []
    for c, sd in zip(args.pilot_ckpts, sds):
        model = TopKSAE(d_model, dict_size, k, b_dec=torch.zeros(d_model)).to(device)
        model.load_state_dict(sd)
        model.eval()
        mse_own = recon_mse(model, X, device)
        with torch.no_grad():
            model.b_dec.copy_(pooled.to(device))
        mse_pooled = recon_mse(model, X, device)

        own_b = sd["b_dec"].float()
        d_bdec = float((own_b - pooled).norm())
        resid = (d_model * mse_own) ** 0.5          # mean ||x - x_hat||
        ratio = d_bdec / resid
        ratios.append(ratio)
        dmse = mse_pooled - mse_own
        print(f"  {c.name:>18}  {mse_own:9.4f}  {mse_pooled:10.4f}  {dmse:7.4f}  "
              f"{100*dmse/mse_own:6.2f}%  {d_bdec:9.3f}  {resid:8.2f}  {ratio:7.2%}")

    print(f"\n  residual-relative offset ‖Δb_dec‖/‖resid‖: mean {np.mean(ratios):.2%}")
    print("  small Δmse (<~0.5%) => freeze harmless (a); large => per-seed optima (b)")

    # CHECK 3 -- structure of the deviations
    dev = (b_decs - pooled).numpy()                 # (N, d_model)
    sv = np.linalg.svd(dev, compute_uv=False)
    energy = sv ** 2
    print(f"\nCHECK 3 -- deviation structure (N={len(sds)} seeds, rank <= N-1):")
    print(f"  singular values: {np.array2string(sv, precision=3)}")
    print(f"  top-1 fraction of deviation energy: {energy[0]/energy.sum():.1%}"
          f"  (isotropic ~ 1/(N-1) = {1/max(1,len(sds)-1):.0%}; higher => structured => (b))")


if __name__ == "__main__":
    main()
