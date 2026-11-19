"""
FVU on the evaluation split, in TWO spaces. PREREGISTRATION amendment CF-a.

WHAT THIS MEASURES, and why it needed a declaration before it could be written.
Amendment AB13 asked for "our FVU on the eval split" alongside the comparator's
~0.36, and amendment CE5 established that clause is not runnable as written: it
fixes the population and the intent, and fixes neither the formula, the space,
the seeds nor the rows. CF-a supplies all four and is committed BEFORE this
module existed.

THE TWO SPACES, and the reason there are two.

  * RAW -- the native quantity. It matches the training objective (train.py:
    both the main and AuxK losses are raw MSE, not the variance-normalised MSE
    of the reference paper) and matches verify_layer.py's existing
    FVEAccumulator, which is the in-repo precedent this adopts rather than
    re-derives.
  * PER-DIMENSION STANDARDISED -- the comparator's convention. The SAME
    per-dimension (mu_d, sigma_d) is applied to target and reconstruction.

FVU is NOT invariant to a per-dimension standardisation: rescaling a coordinate
reweights it in numerator and denominator differently across dimensions. So the
two numbers are different statistics of the same reconstruction, and reporting
one alone would silently pick a convention.

WHY THE SECOND SPACE IS ALMOST FREE. Standardising both sides by statistics
computed on the SAME rows gives, exactly,

    SSE_std = sum_d SSE_d / var_d
    SST_std = sum_d (n * var_d) / var_d = n * d

so the denominator is a constant and the only new state is a (d_model,) vector
of per-dimension squared error. One pass, one set of reconstructions, O(d)
memory. `SST_std = n * d` is asserted at report time rather than assumed --
see `Accumulator.result`.

WHERE THE STATISTICS COME FROM (CF-a2 (ii)): the evaluation split itself, so
SST_std is exact rather than estimated, the standardisation and the statistic
describe the same rows, and no RNG enters an evaluation path that open question
2's decision deliberately left free of one. THIS DIFFERS FROM THE COMPARATOR,
who use float64 statistics from 50K samples, and the difference is recorded
wherever the comparison appears rather than smoothed over.

WHAT THIS DOES NOT ESTABLISH (CF-a3). The comparator NORMALISES ITS INPUTS AND
THEREFORE TRAINS IN NORMALISED SPACE; this project trains in raw space and the
standardised figure here is a post-hoc measurement transform only. The second
space removes the METRIC mismatch. It does not remove the OBJECTIVE mismatch,
and no figure from this module is commensurable with the comparator's without
that sentence attached -- nor without `k = 32` against their `k = 64`.

NUMERICS. Accumulation is float64 throughout; the cache is fp16 and the SAE
runs in float32. Pairwise summation over chunks is not used -- the accumulators
are plain running sums in float64, whose error at n ~ 1.8e7 is far below the
precision any reported digit claims.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import h5py
import numpy as np
import torch

ESM_DIM = 1280


class Accumulator:
    """Streaming FVU in both spaces. O(d_model) memory, exact in one pass.

    Kept as a class with an injectable `d_model` so a test can exercise it on a
    tiny problem whose answer is known by hand -- a fixture that can only be
    built at the real width could not discriminate (gotcha 16).
    """

    def __init__(self, d_model: int):
        self.d = d_model
        self.n = 0
        self.sum_x = np.zeros(d_model, dtype=np.float64)     # per-dim
        self.sum_x2 = np.zeros(d_model, dtype=np.float64)    # per-dim
        self.sse_d = np.zeros(d_model, dtype=np.float64)     # per-dim SSE

    def update(self, x: np.ndarray, recon: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        recon = np.asarray(recon, dtype=np.float64)
        if x.shape != recon.shape:
            raise ValueError(f"shape mismatch {x.shape} vs {recon.shape}")
        if x.shape[1] != self.d:
            raise ValueError(f"width {x.shape[1]} != d_model {self.d}")
        self.n += x.shape[0]
        self.sum_x += x.sum(axis=0)
        self.sum_x2 += (x * x).sum(axis=0)
        d = x - recon
        self.sse_d += (d * d).sum(axis=0)

    def result(self) -> dict:
        if self.n == 0:
            return {"status": "NO_ROWS", "n_rows": 0}
        n = float(self.n)
        # Per-dimension variance about the pooled mean (population form: the
        # same convention SST uses, so SST_std lands on n*d exactly).
        ss_d = self.sum_x2 - (self.sum_x ** 2) / n          # per-dim SS
        var_d = ss_d / n

        sse_raw = float(self.sse_d.sum())
        sst_raw = float(ss_d.sum())

        # A zero-variance dimension is constant on these rows; it contributes
        # nothing to SST and cannot be standardised. It is REPORTED, never
        # silently dropped or floored (gotcha 15).
        live = var_d > 0.0
        n_dead_dims = int((~live).sum())
        sse_std = float((self.sse_d[live] / var_d[live]).sum())
        sst_std = n * float(live.sum())

        out = {
            "status": "OK",
            "n_rows": int(self.n),
            "d_model": int(self.d),
            "n_zero_variance_dims": n_dead_dims,
            "raw": {
                "sse": sse_raw, "sst": sst_raw,
                "fvu": sse_raw / sst_raw if sst_raw > 0 else float("nan"),
                "fve": 1.0 - sse_raw / sst_raw if sst_raw > 0 else float("nan"),
            },
            "standardised": {
                "sse": sse_std, "sst": sst_std,
                "fvu": sse_std / sst_std if sst_std > 0 else float("nan"),
                "fve": 1.0 - sse_std / sst_std if sst_std > 0 else float("nan"),
            },
        }
        # `SST_std == n * d_live` is an ALGEBRAIC IDENTITY, not a measurement:
        # var_d is DEFINED as ss_d / n, so ss_d / var_d is n whatever the data
        # are. A runtime "check" of it could therefore only ever pass, which is
        # exactly the useless instrument gotcha 25 names -- and a first draft of
        # this module shipped one before the test caught it (amendment CF-a5).
        #
        # So it is recorded as a STATED CONSEQUENCE here, and VERIFIED in
        # `tests/test_fvu.py` against an independent numpy standardisation,
        # which is a computation that can disagree.
        out["standardised"]["sst_is_exactly_n_times_live_dims"] = (
            "ALGEBRAIC IDENTITY given the statistics come from these same rows "
            "(CF-a2 ii). NOT a runtime check -- it cannot fail here. Verified "
            "in tests/test_fvu.py against an independent standardisation.")
        return out


def reconstruct(x: torch.Tensor, W_enc: torch.Tensor, b_enc: torch.Tensor,
                W_dec: torch.Tensor, b_dec: torch.Tensor, k: int
                ) -> torch.Tensor:
    """TopKSAE.forward's reconstruction path, from weights alone.

    Mirrors `src/sae/train.py`'s TopKSAE exactly:
        pre   = relu((x - b_dec) @ W_enc.T + b_enc)
        z     = topk(pre, k)                      # values kept, rest zero
        x_hat = z @ W_dec.T + b_dec
    Reimplemented here rather than importing the module so this does not depend
    on the training file's optimiser-side imports; `tests/test_fvu.py` asserts
    it agrees with the real TopKSAE on random weights, which is what makes the
    duplication safe.
    """
    pre = torch.relu((x - b_dec) @ W_enc.T + b_enc)
    topk = torch.topk(pre, k, dim=-1)
    z = torch.zeros_like(pre)
    z.scatter_(-1, topk.indices, topk.values)
    return z @ W_dec.T + b_dec


def resolve_ensemble_git_sha(shas: dict, accept: list | None) -> dict:
    """One shared git_sha across the ensemble is the design's central invariant
    (CLAUDE.md); a disagreement means the seeds were not trained from one
    commit and no cross-seed statistic is interpretable -- so it is a REFUSAL,
    unless the caller names the exact set of SHAs the record has adjudicated
    (amendment CS, CS3b: an identity check on the set, not an existence check
    -- gotcha 15). Returns what the output records."""
    found = set(shas.values())
    if len(found) == 1:
        return {"ensemble_git_sha": next(iter(found)),
                "git_shas_by_seed": {str(k): v for k, v in shas.items()},
                "accepted_by_flag": False}
    if accept is not None and set(accept) == found:
        return {"ensemble_git_sha": None,
                "git_shas_by_seed": {str(k): v for k, v in shas.items()},
                "accepted_by_flag": True}
    raise SystemExit(f"checkpoints disagree on git_sha: {shas}"
                     + ("" if accept is None else
                        f"; --accept-git-shas named {sorted(accept)}, which is "
                        f"not the set found {sorted(found)}"))


def load_sae(ckpt_dir: Path, device: str) -> dict:
    """Weights plus the identity a consumer must assert on (gotcha 15)."""
    sd = torch.load(ckpt_dir / "model.pt", map_location="cpu")
    sd = sd.get("model", sd)
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    w = {k: sd[k].to(device=device, dtype=torch.float32)
         for k in ("W_enc", "b_enc", "W_dec", "b_dec")}
    w["meta"] = meta
    return w


def build_parser() -> argparse.ArgumentParser:
    """The CLI, as a function so a test can assert its arguments exist: the
    CS3b commit (455bbb8) dropped `--out` while adding `--accept-git-shas` and
    no test noticed until a cluster job printed the usage line (CS3c)."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--ckpts", type=Path, required=True,
                   help="parent of seed_NN directories")
    p.add_argument("--seeds", type=int, nargs="+", required=True,
                   help="ANCHOR seeds, named explicitly -- never range(n) "
                        "(gotcha 29: the seed axis has two orders in this repo)")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--chunk-rows", type=int, default=131072)
    p.add_argument("--max-rows", type=int, default=None,
                   help="DEBUG ONLY, AND IT IS NOT A SUBSAMPLE. The cache is "
                        "written in LENGTH-SORTED protein order, so a row "
                        "prefix is the SHORTEST proteins: the first 500,000 "
                        "rows come from 6,885 proteins of length 4-117 (mean "
                        "72.6) against the split's mean of 330.5, and the "
                        "smoke FVU came out 0.348 against the full split's "
                        "0.364. CF-a fixes the FULL split; using this makes "
                        "the result non-compliant and it is recorded as such "
                        "in the output.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--accept-git-shas", nargs="+", default=None,
                   help="AMENDMENT CS, CS3b. The EXACT set of training git_sha "
                        "values the pool is allowed to carry (full 40-hex). "
                        "Without it, any disagreement is a refusal, as before. "
                        "With it, the run proceeds only if the pool's set of "
                        "SHAs EQUALS this set, and the output records every "
                        "seed's SHA and that the flag was used. Exists for the "
                        "width sweep's 40,960 arm, whose seed 0 (the CN4 probe) "
                        "recorded eb8657b9... and seeds 1-9 recorded "
                        "42376f67..., with the training path byte-identical "
                        "between the two commits (PREREGISTRATION §CN7).")
    return p


def main() -> None:
    args = build_parser().parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[fvu] device {device}", flush=True)
    if device == "cpu":
        print("[fvu] WARNING: no GPU visible", flush=True)

    saes = {}
    for s in args.seeds:
        d = args.ckpts / f"seed_{s}"
        if not d.is_dir():
            raise SystemExit(f"missing checkpoint dir {d}")
        saes[s] = load_sae(d, device)
    print(f"[fvu] loaded {len(saes)} SAEs", flush=True)

    # One shared git_sha across the ensemble is the design's central invariant
    # (CLAUDE.md); a disagreement means the seeds were not trained from one
    # commit and no cross-seed statistic below is interpretable.
    shas = {s: saes[s]["meta"].get("git_sha") for s in args.seeds}
    sha_record = resolve_ensemble_git_sha(shas, args.accept_git_shas)

    accs = {s: Accumulator(ESM_DIM) for s in args.seeds}

    with h5py.File(args.cache, "r") as hf:
        acts = hf["activations"]
        n_total = acts.shape[0]
        n_rows = min(n_total, args.max_rows) if args.max_rows else n_total
        print(f"[fvu] {n_rows:,} of {n_total:,} rows", flush=True)
        done = 0
        for i in range(0, n_rows, args.chunk_rows):
            j = min(i + args.chunk_rows, n_rows)
            block = acts[i:j]
            xt = torch.from_numpy(np.asarray(block, dtype=np.float32)).to(device)
            xn = xt.cpu().numpy()
            for s in args.seeds:
                w = saes[s]
                with torch.no_grad():
                    rec = reconstruct(xt, w["W_enc"], w["b_enc"],
                                      w["W_dec"], w["b_dec"], args.k)
                accs[s].update(xn, rec.cpu().numpy())
            done = j
            if (i // args.chunk_rows) % 10 == 0:
                print(f"[fvu]   {done:,}/{n_rows:,} rows", flush=True)

    per_seed = {}
    for s in args.seeds:
        r = accs[s].result()
        per_seed[str(s)] = r
        print(f"[fvu] seed {s:2d}: FVU raw {r['raw']['fvu']:.6f} | "
              f"FVU standardised {r['standardised']['fvu']:.6f}", flush=True)

    def spread(key: str) -> dict:
        v = np.array([per_seed[str(s)][key]["fvu"] for s in args.seeds],
                     dtype=float)
        return {"values_by_seed": {str(s): float(per_seed[str(s)][key]["fvu"])
                                   for s in args.seeds},
                "min": float(v.min()), "median": float(np.median(v)),
                "max": float(v.max()), "mean": float(v.mean()),
                "sd": float(v.std(ddof=1)) if v.size > 1 else float("nan")}

    out = {
        "measurement": "fvu_eval_split_CF",
        "declaration": "PREREGISTRATION amendment CF-a",
        "formula": "FVU = SSE / SST, SST the total variance about the pooled mean",
        "standardising_statistics_from": (
            "the evaluation split itself (CF-a2 ii) -- NOT a held-out 50K "
            "sample, which is the comparator's convention"),
        "spaces": {
            "raw": "native; matches the training objective (raw MSE) and "
                   "verify_layer.py's FVEAccumulator precedent",
            "standardised": "per-dimension, same statistics applied to target "
                            "and reconstruction; the comparator's convention",
        },
        "incommensurability_that_remains": (
            "CF-a3: the comparator NORMALISES ITS INPUTS AND TRAINS IN "
            "NORMALISED SPACE; this project trains in RAW space and the "
            "standardised figure here is a post-hoc measurement transform "
            "only. The second space removes the METRIC mismatch, NOT the "
            "OBJECTIVE mismatch. And k = 32 here against k = 64 there. No "
            "figure in this artifact is commensurable with the comparator's "
            "without both sentences attached."),
        "k": args.k,
        "n_seeds": len(args.seeds),
        "anchor_seeds": list(args.seeds),
        "full_split": args.max_rows is None,
        "compliance": ("CF-a COMPLIANT" if args.max_rows is None else
                       f"NON-COMPLIANT: --max-rows {args.max_rows} was used; "
                       "CF-a fixes the FULL evaluation split"),
        "identity": {
            "cache": str(args.cache),
            "ckpts": str(args.ckpts),
            "ensemble_git_sha": sha_record["ensemble_git_sha"],
            "ensemble_git_shas_by_seed": sha_record["git_shas_by_seed"],
            "git_sha_set_accepted_by_flag": sha_record["accepted_by_flag"],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device_class": (torch.cuda.get_device_name(0)
                             if torch.cuda.is_available() else "cpu"),
        },
        "per_seed": per_seed,
        "spread_raw": spread("raw"),
        "spread_standardised": spread("standardised"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[fvu] wrote {args.out}", flush=True)
    print(f"[fvu] RAW          min {out['spread_raw']['min']:.6f} "
          f"median {out['spread_raw']['median']:.6f} "
          f"max {out['spread_raw']['max']:.6f}", flush=True)
    print(f"[fvu] STANDARDISED min {out['spread_standardised']['min']:.6f} "
          f"median {out['spread_standardised']['median']:.6f} "
          f"max {out['spread_standardised']['max']:.6f}", flush=True)


if __name__ == "__main__":
    main()
