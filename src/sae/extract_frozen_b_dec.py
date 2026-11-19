"""
Turn the b_dec PILOT checkpoints into the single frozen b_dec the ensemble uses.

Converge-then-freeze (CLAUDE.md open question 5): a handful of pilot SAEs are
trained with a TRAINABLE b_dec (geometric-median init); their converged b_dec
approaches b* = E[x] - W_dec.E[z], the reconstruction-optimal offset that leaves
nothing for a "corrector" latent to absorb. This script:

  1. reads the trained b_dec from each pilot checkpoint (model.pt),
  2. averages them into ONE vector -- shared across all 20 ensemble seeds, so it
     injects zero cross-seed variance, while sitting near b*,
  3. reports the cross-seed SPREAD of the pilot b_dec (the ablation output that
     informs freeze-vs-plain-trainable: if the spread is tiny, freezing removes
     little variance),
  4. writes the frozen vector with FULL provenance -- pilot seeds, git SHAs,
     config hashes, n_steps, source checkpoints -- because the frozen b_dec is
     now an INPUT to the experiment (conditioned on a pilot trajectory), not a
     closed-form statistic.

Output format matches compute_b_dec.py (a dict with a "b_dec" tensor), so
train.py loads it identically via --b-dec.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.sae.train import git_dirty, git_sha


def load_pilot_b_dec(ckpt_dir: Path) -> tuple[np.ndarray, dict]:
    """Read the trained b_dec and meta from one pilot checkpoint directory."""
    sd = torch.load(ckpt_dir / "model.pt", map_location="cpu")
    if "b_dec" not in sd:
        raise SystemExit(f"{ckpt_dir}/model.pt has no b_dec")
    b = sd["b_dec"].cpu().float().numpy()
    meta = {}
    meta_path = ckpt_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    return b, meta


def cross_seed_spread(mat: np.ndarray) -> dict:
    """mat: (n_pilots, d_model). Report how much the pilot b_dec varies across
    seeds -- the ablation metric. Small spread => freezing removes little."""
    mean = mat.mean(axis=0)
    # per-pilot L2 distance to the mean, and the componentwise std
    dists = np.linalg.norm(mat - mean, axis=1)
    return {
        "mean_norm": float(np.linalg.norm(mean)),
        "per_pilot_dist_to_mean": [float(d) for d in dists],
        "max_dist_to_mean": float(dists.max()),
        "rms_dist_to_mean": float(np.sqrt((dists ** 2).mean())),
        # spread relative to the vector's own size -- a scale-free read
        "relative_rms_spread": float(np.sqrt((dists ** 2).mean())
                                     / (np.linalg.norm(mean) + 1e-12)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pilot-ckpts", type=Path, nargs="+", required=True,
                   help="Pilot checkpoint DIRECTORIES (each holds model.pt + "
                        "meta.json), one per pilot seed.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output .pt for the frozen ensemble b_dec.")
    args = p.parse_args()

    if len(args.pilot_ckpts) < 1:
        raise SystemExit("need at least one pilot checkpoint")

    b_list, metas, d_model = [], [], None
    for ck in args.pilot_ckpts:
        b, meta = load_pilot_b_dec(ck)
        if d_model is None:
            d_model = b.shape[0]
        elif b.shape[0] != d_model:
            raise SystemExit(f"{ck}: b_dec dim {b.shape[0]} != {d_model}")
        if not meta.get("b_dec_trainable", False):
            print(f"WARNING: {ck} was NOT trained with b_dec_trainable=true -- "
                  "its b_dec is the frozen init, not a converged pilot value.")
        b_list.append(b)
        metas.append(meta)

    mat = np.stack(b_list, axis=0)
    frozen = mat.mean(axis=0).astype(np.float32)
    spread = cross_seed_spread(mat)
    frozen_sha = hashlib.sha256(frozen.tobytes()).hexdigest()[:16]

    print(f"averaged {len(b_list)} pilot b_dec -> frozen vector "
          f"(norm {spread['mean_norm']:.4f}, sha {frozen_sha})")
    print(f"cross-seed spread: rms dist to mean {spread['rms_dist_to_mean']:.4f} "
          f"({spread['relative_rms_spread']:.2%} of the vector norm)")
    print("  -> small relative spread means freezing removes little variance; "
          "large means freezing is doing real work (open Q5 freeze-vs-trainable).")

    blob = {
        "b_dec": torch.from_numpy(frozen),
        "source": "converged_pilot_mean",
        "d_model": int(d_model),
        "n_pilots": len(b_list),
        "pilot_ckpts": [str(c) for c in args.pilot_ckpts],
        "pilot_seeds": [m.get("seed") for m in metas],
        "pilot_git_shas": [m.get("git_sha") for m in metas],
        "pilot_config_hashes": [m.get("config_hash") for m in metas],
        "pilot_n_steps": [m.get("n_steps") for m in metas],
        "pilot_lrs": [m.get("lr") for m in metas],
        "cross_seed_spread": spread,
        "b_dec_sha256_16": frozen_sha,
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)

    meta_out = {k: v for k, v in blob.items() if k != "b_dec"}
    print(json.dumps(meta_out, indent=2))
    print(f"\nwrote frozen ensemble b_dec -> {args.out}")
    print(f"set sae.b_dec_path: {args.out}  and keep sae.b_dec_trainable: false")


if __name__ == "__main__":
    main()
