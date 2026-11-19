"""
Quantify the measurement-noise cost of a TRAINABLE (per-seed) b_dec, resolved by
firing rate (CLAUDE.md open Q5).

Under trainable-shared-init the 20 seeds have different b_dec (~5% spread). The
worry: does that per-seed offset degrade the cross-seed activation-correlation
the reproducibility metric relies on -- and if so, does it bite in the low-firing
tail, the analysis's most delicate region?

Isolation: hold ONE seed's dictionary FIXED and only swap b_dec to another
seed's value (a real cross-seed difference). Any change in a feature's activation
pattern is then attributable to b_dec alone, not to a different dictionary. For
each feature, over the eval tokens, report:
  - Pearson activation correlation between own-b_dec and swapped-b_dec activations
    (the 'activation_correlation' the metric uses; co-zeros inflate it for sparse
    features, so also:)
  - firing-set Jaccard |fire_own & fire_swap| / |fire_own | fire_swap| -- how
    stable the SET of firing tokens is, which is what actually moves in the tail.
Both are binned by the feature's firing rate. Streaming per-feature accumulators,
so memory is O(dict_size), not O(tokens x dict_size).
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
    with h5py.File(path, "r") as hf:
        total = hf["activations"].shape[0]
        n = min(n, total)
        idx = np.sort(np.random.default_rng(seed).choice(total, n, replace=False))
        return torch.from_numpy(hf["activations"][idx].astype(np.float32))


@torch.no_grad()
def measure(model: TopKSAE, b_own: torch.Tensor, b_swap: torch.Tensor,
            X: torch.Tensor, device: str, chunk: int = 8192):
    """Streaming per-feature Pearson correlation and firing-set Jaccard between
    the activations under b_own and under b_swap (same dictionary)."""
    d = model.b_dec.shape[0]
    dict_size = model.W_enc.shape[0]
    n = 0
    sx = torch.zeros(dict_size, dtype=torch.float64, device=device)
    sy = torch.zeros(dict_size, dtype=torch.float64, device=device)
    sxx = torch.zeros(dict_size, dtype=torch.float64, device=device)
    syy = torch.zeros(dict_size, dtype=torch.float64, device=device)
    sxy = torch.zeros(dict_size, dtype=torch.float64, device=device)
    f_own = torch.zeros(dict_size, dtype=torch.float64, device=device)
    f_swap = torch.zeros(dict_size, dtype=torch.float64, device=device)
    f_both = torch.zeros(dict_size, dtype=torch.float64, device=device)

    for i in range(0, X.shape[0], chunk):
        xb = X[i:i + chunk].to(device)
        model.b_dec.copy_(b_own)
        z_own = model(xb)[1].double()
        model.b_dec.copy_(b_swap)
        z_swap = model(xb)[1].double()
        n += xb.shape[0]
        sx += z_own.sum(0); sy += z_swap.sum(0)
        sxx += (z_own * z_own).sum(0); syy += (z_swap * z_swap).sum(0)
        sxy += (z_own * z_swap).sum(0)
        o = z_own > 0; s = z_swap > 0
        f_own += o.sum(0); f_swap += s.sum(0); f_both += (o & s).sum(0)

    cov = sxy - sx * sy / n
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    denom = (vx * vy).clamp_min(0).sqrt()
    pearson = torch.where(denom > 0, cov / denom.clamp_min(1e-12),
                          torch.full_like(cov, float("nan")))
    union = (f_own + f_swap - f_both).clamp_min(1)
    jaccard = f_both / union
    firing_rate = (f_own / n)
    return (pearson.cpu().numpy(), jaccard.cpu().numpy(),
            firing_rate.cpu().numpy(), n)


def report(pearson, jaccard, firing_rate, label):
    alive = firing_rate > 0
    print(f"\n=== swap: {label}  ({alive.sum()} of {len(firing_rate)} features fire) ===")
    # log-spaced firing-rate bins
    edges = np.array([0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1.0])
    print(f"  {'firing-rate bin':>20}  {'n_feat':>7}  {'med Pearson':>11}  "
          f"{'med Jaccard':>11}  {'p10 Jaccard':>11}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = alive & (firing_rate >= lo) & (firing_rate < hi)
        if m.sum() == 0:
            continue
        pr = np.nanmedian(pearson[m])
        jac = np.nanmedian(jaccard[m])
        j10 = np.nanpercentile(jaccard[m], 10)
        print(f"  [{lo:.0e}, {hi:.0e})".rjust(22) +
              f"  {int(m.sum()):>7}  {pr:>11.3f}  {jac:>11.3f}  {j10:>11.3f}")
    print(f"  {'OVERALL (alive)':>20}  {int(alive.sum()):>7}  "
          f"{np.nanmedian(pearson[alive]):>11.3f}  "
          f"{np.nanmedian(jaccard[alive]):>11.3f}  "
          f"{np.nanpercentile(jaccard[alive], 10):>11.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model-ckpt", type=Path, required=True,
                   help="Checkpoint dir whose DICTIONARY is held fixed.")
    p.add_argument("--swap-ckpts", type=Path, nargs="+", required=True,
                   help="Other seeds' ckpt dirs; their b_dec is swapped in.")
    p.add_argument("--eval-cache", type=Path, required=True)
    p.add_argument("--n-rows", type=int, default=100000)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    d_model = cfg["model"]["d_model"]
    dict_size, k = cfg["sae"]["dict_size"], cfg["sae"]["k"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU visible.")

    sd = torch.load(args.model_ckpt / "model.pt", map_location="cpu")
    model = TopKSAE(d_model, dict_size, k, b_dec=torch.zeros(d_model)).to(device)
    model.load_state_dict(sd)
    model.eval()
    b_own = sd["b_dec"].float().to(device)

    X = load_eval_sample(args.eval_cache, args.n_rows)
    print(f"fixed dictionary: {args.model_ckpt.name}; {X.shape[0]:,} eval rows")

    for sc in args.swap_ckpts:
        b_swap = torch.load(sc / "model.pt", map_location="cpu")["b_dec"].float().to(device)
        dist = float((b_own - b_swap).norm())
        pr, jac, fr, n = measure(model, b_own, b_swap, X, device)
        report(pr, jac, fr, f"{args.model_ckpt.name} b_dec -> {sc.name} "
                            f"(||Δb_dec||={dist:.2f})")


if __name__ == "__main__":
    main()
