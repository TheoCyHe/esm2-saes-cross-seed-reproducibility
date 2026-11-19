"""
Verify that HF EsmModel `hidden_states[24]` is the "layer 24" a released
reference SAE was trained on -- the one stage-1 assumption the smoke test could
NOT check (it proved internal consistency, not agreement with the references).

The idea: an SAE reconstructs its OWN training layer far better than neighbours.
So feed HF `hidden_states[L]` -- extracted and special-token-stripped EXACTLY as
src/embeddings/cache.py does -- into InterProt's released layer-24 SAE, for L in
a sweep around 24, and measure fraction of variance explained (FVE). If our
indexing matches theirs, L=24 is the clear argmax. An off-by-one in cache.py's
slicing (or a HF-vs-fair-esm layer convention mismatch) would move the argmax
off 24 -- that is the way this check FAILS, which is the point.

This reimplements InterProt's SparseAutoencoder.forward (interprot/sae_model.py)
from its weights so we do not need their package: per-token LayerNorm, subtract
b_pre, encode, TopK+ReLU, decode, add b_pre, un-normalize. k defaults to 128,
matching the README's `SparseAutoencoder(ESM_DIM, SAE_DIM)` usage.

The SAE weights must be downloaded on the login node first (compute nodes have
no outbound internet); see slurm/verify_layer.sbatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.embeddings.cache import read_fasta

ESM_DIM = 1280


# `sae_reconstruct` reimplements the reconstruction path of InterProt's
# SparseAutoencoder (https://github.com/etowahadams/interprot,
# interprot/sae_model.py: LN, forward, topK_activation). InterProt's notice,
# reproduced as its licence requires (the author's decision, PREREGISTRATION.md
# §CT18-D4 item 1; text fetched twice from the repository's LICENSE, 2026-10-29
# and 2026-11-01):
#
# MIT License
#
# Copyright (c) 2026 InterProt Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
def sae_reconstruct(
    x: torch.Tensor,
    w_enc: torch.Tensor,
    w_dec: torch.Tensor,
    b_enc: torch.Tensor,
    b_pre: torch.Tensor,
    k: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """InterProt SparseAutoencoder.forward, reconstruction path only. x is raw
    (N, d_model) activations; returns the reconstruction in the same raw space.
    Matches interprot/sae_model.py exactly, including the per-token LN and the
    un-normalization by `std` (not std+eps)."""
    mu = x.mean(dim=-1, keepdim=True)
    xc = x - mu
    std = xc.std(dim=-1, keepdim=True)
    xn = xc / (std + eps)
    xn = xn - b_pre
    pre = xn @ w_enc + b_enc
    topk = torch.topk(pre, k, dim=-1)
    vals = torch.relu(topk.values)
    z = torch.zeros_like(pre)
    z.scatter_(-1, topk.indices, vals)
    recons = z @ w_dec + b_pre
    return recons * std + mu


class FVEAccumulator:
    """Streaming fraction-of-variance-explained: FVE = 1 - SSE/SST, with SST the
    total variance around the pooled mean. O(d_model) memory, exact."""

    def __init__(self, d_model: int):
        self.n = 0
        self.sum_x = np.zeros(d_model, dtype=np.float64)
        self.sum_x2 = 0.0     # sum ||x||^2
        self.sse = 0.0        # sum ||x - recon||^2

    def update(self, x: np.ndarray, recon: np.ndarray) -> None:
        x = x.astype(np.float64)
        recon = recon.astype(np.float64)
        self.n += x.shape[0]
        self.sum_x += x.sum(axis=0)
        self.sum_x2 += float((x * x).sum())
        self.sse += float(((x - recon) ** 2).sum())

    def fve(self) -> float:
        if self.n == 0:
            return float("nan")
        sst = self.sum_x2 - float((self.sum_x @ self.sum_x)) / self.n
        return 1.0 - self.sse / sst if sst > 0 else float("nan")


def main() -> None:
    from transformers import AutoTokenizer, EsmModel
    from safetensors.torch import load_file

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sae", type=Path, required=True,
                   help="InterProt SAE .safetensors (download on the login node).")
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--ref-layer", type=int, default=24,
                   help="The layer the reference SAE claims to be trained on.")
    p.add_argument("--layers", type=int, nargs="+", default=[22, 23, 24, 25, 26],
                   help="Layer sweep. Must include --ref-layer.")
    p.add_argument("--k", type=int, nargs="+", default=[128],
                   help="TopK value(s) for the reference SAE (README default 128). "
                        "Multiple values sweep k reusing one ESM pass; the verdict "
                        "uses the k that best reconstructs the reference layer.")
    p.add_argument("--n-seqs", type=int, default=50)
    p.add_argument("--max-len", type=int, default=1022)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if args.ref_layer not in args.layers:
        raise SystemExit(f"--ref-layer {args.ref_layer} must be in --layers {args.layers}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU visible.")

    sd = load_file(str(args.sae))
    w_enc = sd["w_enc"].to(device).float()   # (d_model, d_hidden)
    w_dec = sd["w_dec"].to(device).float()   # (d_hidden, d_model)
    b_enc = sd["b_enc"].to(device).float()
    b_pre = sd["b_pre"].to(device).float()
    d_hidden = w_enc.shape[1]
    ks = sorted(set(args.k))
    print(f"loaded SAE {args.sae.name}: d_model={w_enc.shape[0]} d_hidden={d_hidden} "
          f"k={ks}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = EsmModel.from_pretrained(args.model, add_pooling_layer=False).eval().half().to(device)

    records = read_fasta(args.fasta)[: args.n_seqs]
    print(f"{len(records)} sequences from {args.fasta}")

    # One accumulator per (layer, k). ESM runs once per sequence; the SAE
    # reconstruction is cheap, so sweeping k is nearly free.
    accs = {(L, k): FVEAccumulator(ESM_DIM) for L in args.layers for k in ks}
    with torch.no_grad():
        for acc, seq in records:
            seq = seq[: args.max_len]
            inputs = tokenizer(seq, return_tensors="pt").to(device)
            hs = model(**inputs, output_hidden_states=True).hidden_states
            L_seq = len(seq)
            for L in args.layers:
                # EXACTLY cache.py's extraction: row i == residue i, specials off
                resid = hs[L][0, 1 : 1 + L_seq].float()
                assert resid.shape[0] == L_seq, f"{acc}: strip misaligned at layer {L}"
                x_np = resid.cpu().numpy()
                for k in ks:
                    recon = sae_reconstruct(resid, w_enc, w_dec, b_enc, b_pre, k)
                    accs[(L, k)].update(x_np, recon.cpu().numpy())

    fve = {(L, k): accs[(L, k)].fve() for L in args.layers for k in ks}
    ref = args.ref_layer
    # The "most faithful" k is the one that best reconstructs the reference layer.
    best_k = max(ks, key=lambda k: fve[(ref, k)])

    print("\n============== LAYER x k  FVE  (rows=layer) ===============")
    print("  layer  " + "".join(f"  k={k:<7d}" for k in ks))
    for L in sorted(args.layers):
        row = "".join(f"  {fve[(L, k)]:+.4f} " for k in ks)
        mark = "  <- reference" if L == ref else ""
        print(f"  [{L:>2}]   {row}{mark}")
    print("==========================================================")

    # Verdict: at the most-faithful k, the reference layer must be the argmax.
    col = {L: fve[(L, best_k)] for L in args.layers}
    argmax_layer = max(col, key=col.get)
    others = [v for L, v in col.items() if L != ref]
    margin = col[ref] - max(others) if others else float("nan")
    verdict = argmax_layer == ref
    print(f"best k = {best_k} (highest ref-layer FVE = {col[ref]:+.4f}); "
          f"argmax layer at that k = {argmax_layer}; margin = {margin:+.4f}")
    print("VERDICT:", f"PASS -- hidden_states[{ref}] matches the reference SAE's layer"
          if verdict else
          "FAIL -- argmax is not the reference layer; indexing is suspect")

    if args.out:
        args.out.write_text(json.dumps({
            "sae": str(args.sae), "model": args.model, "ks": ks,
            "n_seqs": len(records), "ref_layer": ref, "best_k": best_k,
            "fve": {f"L{L}_k{k}": fve[(L, k)] for L in args.layers for k in ks},
            "argmax_layer_at_best_k": int(argmax_layer),
            "margin_over_next_best": margin, "pass": bool(verdict),
        }, indent=2))
    raise SystemExit(0 if verdict else 1)


if __name__ == "__main__":
    main()
