"""
Train one TopK SAE on a cached activation file (one seed of the ensemble).

Architecture (Gao et al. 2024 / InterProt convention -- see CLAUDE.md gotcha 1
for why this must stay TopK, never ReLU+L1):
    z_pre  = W_enc @ (x - b_dec) + b_enc
    z      = TopK(ReLU(z_pre), k)      # keep the k largest activations, zero rest
    x_hat  = W_dec @ z + b_dec
Decoder columns are renormalized to unit norm after every optimizer step, the
usual convention for keeping decoder-cosine a meaningful similarity metric
(needed later for feature-level MNN matching). Before the step, the component
of the decoder gradient parallel to each unit column is projected out
(sae.decoder_grad_projection) so Adam does not fight that renorm -- the Gao et
al. / Anthropic trick, matching InterProt's norm_grad().

b_dec is TRAINABLE from a SHARED geometric-median INIT (CLAUDE.md locked
decision, resolved 2026-07-24 after two reversals; sae.b_dec_trainable: true).
The init is computed once from the activation cache by
src/sae/compute_b_dec.py and is bit-identical across all 40 dictionaries
(b_dec_init_sha256_16 in every meta.json), so the starting point is
seed-invariant; each dictionary then trains its own b_dec and the final values
differ by seed. That is seed-determined variance and does not violate "vary
only the seed"; freezing was rejected because the pilot showed b* is genuinely
seed-specific (within-run drift ~0.8 vs cross-seed spread ~13). --b-dec (or
sae.b_dec_path) is required, exactly like lr, so a zeros smoke default can
never silently become the real run. (Docstring corrected 2026-09-16: it said
"FROZEN and SHARED ... bit-identical across all 20 seeds", the pre-resolution
design. No behaviour change.)

AuxK dead-latent revival (LOCKED: in): every step the top-k_aux currently-dead
latents are asked to reconstruct the main model's residual (x - x_hat); the
term is weighted by sae.aux_coef. Without it the dead fraction floats by seed,
putting optimizer luck inside the variance the experiment interprets.

The loss reduction (sae.loss_reduction) is explicit because an lr lifted from
a paper is only valid for the paper's reduction -- 'd_model_sum' is ~1280x
'd_model_mean'. The aux term uses the same reduction so aux_coef keeps its
weight relative to the main loss. Both terms are raw MSE, not the
variance-normalized MSE of the paper; that normalization is deferred with
sae.lr.

Reproducibility (CLAUDE.md "vary only the seed"): every checkpoint records
git SHA, a hash of the config file, the seed, and library versions. Python /
numpy / torch RNGs and the dataloader's shuffle order are all seeded from
--seed. cuDNN determinism is left OFF by default (can be slow); the run
metadata records whether it was on so this is never ambiguous after the fact.

configs/mvp.yaml sets sae.lr 2e-4, sae.batch_size 4096 and sae.n_steps 30000,
all from this project's own pilot sweep (CLAUDE.md open question 3, resolved).
This script still refuses to invent any of the three if the config leaves one
null; --lr/--batch-size/--n-steps override the config for smoke testing only.
(Docstring corrected 2026-09-16: it said the config "intentionally leaves"
them null. No behaviour change.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml

from src.analysis.artifact_identity import device_class


REPO_ROOT = Path(__file__).resolve().parents[2]


def git_sha() -> str:
    # Anchored to the repo, not the process CWD: a SLURM job launched from a
    # scratch directory would otherwise silently record "unknown".
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool | None:
    """True if the working tree has uncommitted changes. A bare SHA is not
    enough to reproduce a run that was launched from a dirty tree, so this is
    recorded alongside it. None means git could not be queried at all."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        return bool(out)
    except Exception:
        return None


def config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _jsonable(v):
    """h5py attrs come back as numpy scalars/arrays, which json can't encode."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, bytes):
        return v.decode()
    return v


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reduce_se(se: torch.Tensor, reduction: str) -> torch.Tensor:
    """Reduce a per-element squared-error tensor (batch, d_model) to a scalar
    under the named convention. Shared by the main and AuxK losses so aux_coef
    keeps its paper meaning regardless of which convention is chosen."""
    if reduction == "d_model_mean":
        return se.mean()                  # mean over batch AND d_model = MSE
    if reduction == "d_model_sum":
        return se.sum(dim=-1).mean()      # sum over d_model, mean over batch
    raise ValueError(
        f"unknown loss_reduction {reduction!r}; expected "
        "'d_model_mean' or 'd_model_sum'"
    )


class TopKSAE(nn.Module):
    """TopK SAE. b_dec is normally a FROZEN, shared buffer (the ensemble; see
    module docstring / CLAUDE.md) -- it must not receive gradients and must not
    appear in model.parameters(), yet must travel in the state_dict.

    For the b_dec PILOT (CLAUDE.md open question 5), pass b_dec_trainable=True:
    b_dec becomes an nn.Parameter initialised to the same value (a geometric
    median from compute_b_dec.py) and is trained, so it converges toward the
    reconstruction-optimal offset b* = E[x] - W_dec.E[z]. The converged pilot
    b_dec is then extracted and frozen for the ensemble. Either way the value is
    in the state_dict; the None default (zeros) exists only for unit tests.
    """

    def __init__(self, d_model: int, dict_size: int, k: int,
                 b_dec: torch.Tensor | None = None,
                 b_dec_trainable: bool = False):
        super().__init__()
        self.k = k
        self.b_dec_trainable = b_dec_trainable
        self.W_enc = nn.Parameter(torch.empty(dict_size, d_model))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(torch.empty(d_model, dict_size))

        if b_dec is None:
            b_dec = torch.zeros(d_model)
        else:
            b_dec = torch.as_tensor(b_dec, dtype=torch.float32).clone()
            if b_dec.shape != (d_model,):
                raise ValueError(
                    f"b_dec has shape {tuple(b_dec.shape)}, expected ({d_model},)"
                )
        # Parameter for the pilot (trained), buffer for the ensemble (frozen).
        # A buffer is excluded from the optimizer; a Parameter is trained. Both
        # are saved in the state_dict so "which b_dec was this trained with" is
        # answerable from the checkpoint alone.
        if b_dec_trainable:
            self.b_dec = nn.Parameter(b_dec)
        else:
            self.register_buffer("b_dec", b_dec)

        nn.init.kaiming_uniform_(self.W_enc)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self.W_dec /= self.W_dec.norm(dim=0, keepdim=True)

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        """ReLU pre-activations before the TopK gate. Exposed so AuxK can pick
        the top dead latents from the same forward pass."""
        return torch.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)

    @staticmethod
    def _gate_topk(pre: torch.Tensor, k: int) -> torch.Tensor:
        topk = torch.topk(pre, k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk.indices, topk.values)
        return z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre = self.encode_pre(x)
        z = self._gate_topk(pre, self.k)
        x_hat = z @ self.W_dec.T + self.b_dec
        return x_hat, z, pre

    def auxk_loss(
        self,
        residual: torch.Tensor,
        pre: torch.Tensor,
        dead_mask: torch.Tensor,
        aux_k: int,
        reduction: str,
    ) -> torch.Tensor | None:
        """AuxK term: ask the top-`aux_k` currently-dead latents to reconstruct
        the main model's residual (x - x_hat). Returns None when nothing is
        dead (no term to add). `residual` should be detached by the caller --
        the aux term revives dead latents, it must not push the live path
        around to make its own job easier.

        Non-dead latents are masked to -inf so the top-aux_k are drawn strictly
        from dead ones; aux_k is clamped to the number actually dead. The dead
        latents' reconstruction uses the decoder WITHOUT b_dec, because it
        models the residual, not x itself.
        """
        n_dead = int(dead_mask.sum())
        if n_dead == 0:
            return None
        kk = min(aux_k, n_dead)
        masked = pre.masked_fill(~dead_mask.unsqueeze(0), float("-inf"))
        z_aux = self._gate_topk(masked, kk)
        e_hat = z_aux @ self.W_dec.T
        return reduce_se((e_hat - residual) ** 2, reduction)

    @torch.no_grad()
    def project_decoder_grad(self) -> None:
        """Remove the component of the decoder gradient parallel to each
        unit-norm decoder column, so an Adam step does not push columns off the
        unit sphere for renorm_decoder() to then fight. Gao et al. / Anthropic;
        InterProt norm_grad(). No-op before the first backward."""
        if self.W_dec.grad is None:
            return
        # per-column dot(grad, column); columns are unit norm, so this is the
        # parallel-component magnitude. Subtract column * that.
        dots = (self.W_dec * self.W_dec.grad).sum(dim=0, keepdim=True)
        self.W_dec.grad.sub_(self.W_dec * dots)

    @torch.no_grad()
    def renorm_decoder(self) -> None:
        self.W_dec /= self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)


class ActivationDataset(torch.utils.data.Dataset):
    """Random-access view over a cached activation file.

    Indexing accepts either a single row or a list of rows. The batched form
    matters: the cache is chunked at H5_CHUNK_ROWS rows, so fetching a
    shuffled batch one row at a time makes HDF5 read (and discard most of) a
    ~10 MB chunk per sample. One sorted multi-row read per batch instead of
    `batch_size` scattered reads is the difference between an I/O-bound and a
    compute-bound training loop on a cache this size. The loop in main()
    therefore drives this with a BatchSampler, not per-sample indexing.

    The file handle is opened lazily so that each dataloader worker gets its
    own -- an h5py handle inherited across a fork is not safe to share.
    """

    def __init__(self, h5_path: Path):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as hf:
            self.n = hf["activations"].shape[0]
            self.d_model = hf["activations"].shape[1]
        self._hf = None

    def _ensure_open(self):
        if self._hf is None:
            self._hf = h5py.File(self.h5_path, "r")
        return self._hf

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx) -> torch.Tensor:
        hf = self._ensure_open()
        if isinstance(idx, (list, np.ndarray)):
            # h5py fancy indexing requires strictly increasing indices. Sorting
            # only reorders rows *within* a batch, which the loss is invariant
            # to, and keeps the read sequential on disk.
            order = np.sort(np.asarray(idx))
            rows = hf["activations"][order].astype(np.float32)
            return torch.from_numpy(rows)
        return torch.from_numpy(hf["activations"][idx].astype(np.float32))


class ShuffledBlockActivations(torch.utils.data.IterableDataset):
    """Streams shuffled batches from one or more activation caches.

    Why not just index random rows: the cache is chunked at H5_CHUNK_ROWS
    (4096) rows, ~10 MB per chunk. A batch of 256 uniformly random rows out of
    tens of millions lands in ~256 *distinct* chunks, so HDF5 faults in ~2.5 GB
    to hand back ~1.3 MB. Sorting the indices within the batch removes syscall
    overhead but not that read amplification -- the bytes still have to come off
    disk. At 20 seeds an I/O-bound trainer multiplies into real wall-clock.

    So: read whole chunk-aligned BLOCKS sequentially, hold `buffer_blocks` of
    them in memory, shuffle across that buffer, and emit batches from it. Read
    amplification drops to ~1x. The cost is that shuffling is approximate --
    rows from the same protein are correlated and only get mixed against the
    other blocks resident in the buffer, not the whole corpus. A buffer holding
    many blocks drawn from randomly-ordered positions across all shards makes
    that mixing good enough, and it is *identical across seeds except for the
    seed*, which is the property this project actually needs.

    Accepts a list of files so sharded caches need no merge step, and so
    cross-shard shuffling falls out naturally: blocks from all shards go into
    one shuffled pool.
    """

    def __init__(
        self,
        paths: list[Path],
        batch_size: int,
        seed: int,
        block_rows: int = 4096,
        buffer_blocks: int = 32,
        drop_last: bool = True,
    ):
        super().__init__()
        self.paths = [Path(p) for p in paths]
        self.batch_size = batch_size
        self.seed = seed
        self.block_rows = block_rows
        self.buffer_blocks = buffer_blocks
        self.drop_last = drop_last
        self._epoch = 0

        self.d_model: int | None = None
        self.blocks: list[tuple[int, int, int]] = []   # (file_idx, start, end)
        self.n_rows = 0
        for fi, p in enumerate(self.paths):
            with h5py.File(p, "r") as hf:
                n, d = hf["activations"].shape
            if self.d_model is None:
                self.d_model = d
            elif self.d_model != d:
                raise ValueError(f"{p} has d_model={d}, expected {self.d_model}")
            self.n_rows += n
            for s in range(0, n, block_rows):
                self.blocks.append((fi, s, min(s + block_rows, n)))

    def __len__(self) -> int:
        """Batches per epoch (approximate under drop_last within buffers)."""
        return self.n_rows // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nworkers = (info.id, info.num_workers) if info else (0, 1)

        # Epoch is folded into the seed so successive passes differ, while a
        # given (seed, epoch) is always reproducible.
        rng = np.random.default_rng([self.seed, self._epoch])
        order = rng.permutation(len(self.blocks))
        mine = order[wid::nworkers]          # disjoint block subset per worker

        handles: dict[int, h5py.File] = {}

        def get(fi: int):
            if fi not in handles:
                handles[fi] = h5py.File(self.paths[fi], "r")
            return handles[fi]

        try:
            buf: list[np.ndarray] = []
            held = 0
            # Rows left over when a buffer does not divide evenly into batches.
            # These are carried into the next buffer rather than dropped --
            # dropping per-drain silently loses up to batch_size-1 rows *every*
            # buffer, so an "epoch" would not actually cover the data.
            carry = np.empty((0, self.d_model), dtype=np.float16)

            for bi in mine:
                fi, s, e = self.blocks[bi]
                buf.append(get(fi)["activations"][s:e])   # contiguous read
                held += e - s
                if held >= self.buffer_blocks * self.block_rows:
                    carry = yield from self._drain(buf, carry, rng)
                    buf, held = [], 0

            carry = yield from self._drain(buf, carry, rng)
            # At most one short batch per epoch reaches here.
            if len(carry) and not self.drop_last:
                yield torch.from_numpy(carry.astype(np.float32))
        finally:
            for h in handles.values():
                h.close()

    def _drain(self, buf: list[np.ndarray], carry: np.ndarray, rng):
        """Shuffle the buffer (plus any carried rows) and emit full batches.
        Returns the new leftover to carry forward."""
        parts = ([carry] if len(carry) else []) + buf
        if not parts:
            return carry
        block = np.concatenate(parts, axis=0)
        perm = rng.permutation(block.shape[0])

        n_full = (block.shape[0] // self.batch_size) * self.batch_size
        for i in range(0, n_full, self.batch_size):
            idx = np.sort(perm[i: i + self.batch_size])
            yield torch.from_numpy(block[idx].astype(np.float32))
        return block[np.sort(perm[n_full:])]


def worker_init_fn(worker_id: int, base_seed: int) -> None:
    seed = base_seed * 1000 + worker_id
    np.random.seed(seed)
    random.seed(seed)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--activations", type=Path, required=True, nargs="+",
                    help="One or more cache files. A sharded cache needs no "
                         "merge step -- blocks from all shards go into one "
                         "shuffled pool.")
    p.add_argument("--seed", type=int, required=True,
                    help="Seeds RNGs and the dataloader shuffle. This is the one "
                         "thing that should vary across the 20-seed ensemble.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lr", type=float, default=None,
                    help="Overrides sae.lr in the config. Required if the config "
                         "leaves it null (still TODO as of this writing).")
    p.add_argument("--batch-size", type=int, default=None,
                    help="Overrides sae.batch_size. Same null-is-TODO rule as --lr.")
    p.add_argument("--n-steps", type=int, default=None,
                    help="Overrides sae.n_steps. Same null-is-TODO rule as --lr.")
    p.add_argument("--b-dec", type=Path, default=None,
                    help="Path to the shared frozen b_dec (.pt from "
                         "compute_b_dec.py). Overrides sae.b_dec_path. Required "
                         "if the config leaves it null -- no zeros default, so a "
                         "smoke run can never quietly become the real run.")
    p.add_argument("--b-dec-trainable", action="store_true",
                    help="PILOT mode: train b_dec from the loaded value instead "
                         "of freezing it (open Q5). Forces sae.b_dec_trainable "
                         "true. The ensemble run must NOT pass this.")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--block-rows", type=int, default=4096,
                    help="Rows per contiguous read. Keep aligned to the "
                         "cache's HDF5 chunking (4096) or read amplification "
                         "comes back.")
    p.add_argument("--buffer-blocks", type=int, default=32,
                    help="Blocks held in memory for shuffling. Higher = better "
                         "mixing, more RAM (32 x 4096 x 1280 fp16 ~= 335 MB).")
    p.add_argument("--deterministic-cudnn", action="store_true",
                    help="Full cuDNN determinism. Slower; off by default. Whichever "
                         "way this is set, it is recorded in the checkpoint metadata.")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    sae_cfg = cfg["sae"]
    if sae_cfg["arch"] != "topk":
        raise SystemExit(
            f"configs says sae.arch={sae_cfg['arch']!r} -- this script only "
            "implements TopK (see CLAUDE.md gotcha 1: ReLU+L1 must never enter "
            "the ensemble)."
        )

    lr = args.lr if args.lr is not None else sae_cfg["lr"]
    batch_size = args.batch_size if args.batch_size is not None else sae_cfg["batch_size"]
    n_steps = args.n_steps if args.n_steps is not None else sae_cfg["n_steps"]
    missing = [name for name, v in
               [("lr", lr), ("batch_size", batch_size), ("n_steps", n_steps)] if v is None]
    if missing:
        raise SystemExit(
            f"{', '.join(missing)} not set in {args.config} and no CLI override "
            f"given. These are TODO in configs/mvp.yaml (recipe not yet decided) "
            f"-- pass --{missing[0].replace('_', '-')} explicitly for a smoke "
            "test; don't silently default the real ensemble run."
        )

    dict_size = sae_cfg["dict_size"]
    k = sae_cfg["k"]
    d_model = cfg["model"]["d_model"]

    reduction = sae_cfg["loss_reduction"]
    if reduction not in ("d_model_mean", "d_model_sum"):
        raise SystemExit(
            f"sae.loss_reduction={reduction!r} in {args.config}; expected "
            "'d_model_mean' or 'd_model_sum'."
        )
    aux_k = sae_cfg["aux_k"]
    aux_coef = sae_cfg["aux_coef"]
    dead_tokens_threshold = sae_cfg["dead_tokens_threshold"]

    # Optimizer / stability, sourced from InterProt (see configs/mvp.yaml).
    optimizer_name = sae_cfg.get("optimizer", "adam")
    if optimizer_name not in ("adam", "adamw"):
        raise SystemExit(
            f"sae.optimizer={optimizer_name!r}; expected 'adam' or 'adamw'."
        )
    grad_clip = sae_cfg.get("grad_clip")
    decoder_grad_projection = sae_cfg.get("decoder_grad_projection", False)
    # Pilot mode (CLAUDE.md open Q5): b_dec is TRAINED (from a GM init) so it
    # converges toward b*, then the converged value is extracted and frozen for
    # the ensemble. Default false = the frozen-buffer ensemble mode; --b-dec-
    # trainable (pilot only) forces it on.
    b_dec_trainable = args.b_dec_trainable or sae_cfg.get("b_dec_trainable", False)

    # b_dec is required exactly like lr (even as a trainable pilot's init): a
    # zeros fallback here is the smoke-becomes-real footgun the working pattern
    # warns about. Load it BEFORE the seed touches anything.
    b_dec_path = args.b_dec if args.b_dec is not None else sae_cfg.get("b_dec_path")
    if b_dec_path is None:
        raise SystemExit(
            "no shared b_dec: sae.b_dec_path is null and --b-dec was not given. "
            "b_dec must be computed once and shared across all seeds -- run "
            "`python -m src.sae.compute_b_dec --config configs/mvp.yaml "
            "--activations <cache.h5> --out <b_dec.pt>` first, then pass "
            "--b-dec <b_dec.pt>."
        )
    b_dec_path = Path(b_dec_path)
    b_dec_blob = torch.load(b_dec_path, map_location="cpu")
    b_dec = b_dec_blob["b_dec"] if isinstance(b_dec_blob, dict) else b_dec_blob
    b_dec = torch.as_tensor(b_dec, dtype=torch.float32)
    if b_dec.shape != (d_model,):
        raise SystemExit(
            f"{b_dec_path} holds b_dec of shape {tuple(b_dec.shape)}, config "
            f"expects ({d_model},) -- wrong cache or wrong model?"
        )

    set_all_seeds(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic_cudnn
    torch.backends.cudnn.benchmark = not args.deterministic_cudnn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU visible.")

    dataset = ShuffledBlockActivations(
        args.activations, batch_size=batch_size, seed=args.seed,
        block_rows=args.block_rows, buffer_blocks=args.buffer_blocks,
    )
    if dataset.d_model != d_model:
        raise SystemExit(
            f"{args.activations} has d_model={dataset.d_model}, config expects "
            f"{d_model} -- wrong activation cache?"
        )
    print(f"{dataset.n_rows:,} rows across {len(args.activations)} file(s), "
          f"{len(dataset.blocks):,} blocks")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=None,
        num_workers=args.num_workers,
        worker_init_fn=(lambda wid: worker_init_fn(wid, args.seed))
                       if args.num_workers > 0 else None,
    )

    model = TopKSAE(d_model, dict_size, k, b_dec=b_dec,
                    b_dec_trainable=b_dec_trainable).to(device)
    # In ensemble mode b_dec is a frozen buffer; assert it so a refactor that
    # turns it back into a Parameter fails loudly rather than silently letting
    # the optimizer drift it per seed. In pilot mode it is intentionally trained.
    assert model.b_dec.requires_grad == b_dec_trainable

    # AdamW (InterProt) brings default weight_decay=0.01. In pilot mode b_dec is
    # trained but must be EXCLUDED from weight decay -- decay would shrink it
    # toward 0, away from the b* the pilot exists to find. So it gets its own
    # no-decay param group. In ensemble mode b_dec is a buffer (not in
    # parameters()), so this is a no-op.
    opt_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    if b_dec_trainable:
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            (no_decay if name == "b_dec" else decay).append(param)
        optimizer = opt_cls([
            {"params": decay},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=lr)
    else:
        optimizer = opt_cls(model.parameters(), lr=lr)

    # Snapshot the b_dec init so the pilot can report how far it has moved and
    # whether it has stopped moving (converged toward b*).
    b_dec_init_dev = model.b_dec.detach().clone() if b_dec_trainable else None

    # Kept on-device: a .cpu() here would force a host sync every step.
    fire_counts = torch.zeros(dict_size, dtype=torch.long, device=device)
    # Tokens each latent has gone without firing. A latent is "dead" for AuxK
    # once this passes dead_tokens_threshold; firing resets it to 0.
    tokens_since_fired = torch.zeros(dict_size, dtype=torch.long, device=device)
    step = 0
    epoch = 0
    running_loss = 0.0
    running_aux = 0.0
    model.train()
    while step < n_steps:
        # Folded into the shuffle seed so successive passes see a different
        # block order, while (seed, epoch) stays reproducible.
        dataset.set_epoch(epoch)
        epoch += 1
        for batch in loader:
            if step >= n_steps:
                break
            x = batch.to(device)
            n_tok = x.shape[0]

            x_hat, z, pre = model(x)
            main_loss = reduce_se((x_hat - x) ** 2, reduction)

            # AuxK on latents dead as of the PRIOR step's history. Residual is
            # detached: the aux term revives dead latents, it must not perturb
            # the live reconstruction to ease its own target.
            dead_mask = tokens_since_fired > dead_tokens_threshold
            aux_loss = model.auxk_loss(
                (x - x_hat).detach(), pre, dead_mask, aux_k, reduction
            )
            loss = main_loss if aux_loss is None else main_loss + aux_coef * aux_loss

            optimizer.zero_grad()
            loss.backward()
            # Project out the decoder gradient's parallel component before the
            # step, then clip, then step, then renorm -- InterProt's order.
            if decoder_grad_projection:
                model.project_decoder_grad()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            model.renorm_decoder()

            fired = (z > 0).any(dim=0)
            tokens_since_fired += n_tok
            tokens_since_fired[fired] = 0
            fire_counts += (z > 0).sum(dim=0)
            running_loss += main_loss.item()
            running_aux += 0.0 if aux_loss is None else aux_loss.item()
            step += 1

            if step % args.log_every == 0:
                n_dead_now = int((tokens_since_fired > dead_tokens_threshold).sum())
                msg = (f"step {step}/{n_steps}  mse {running_loss / args.log_every:.4f}"
                       f"  aux {running_aux / args.log_every:.4f}  dead {n_dead_now}")
                if b_dec_init_dev is not None:
                    # PILOT: |b_dec| and how far it has drifted from the GM init.
                    # A shift that plateaus means b_dec has converged toward b*.
                    bd = model.b_dec.detach()
                    msg += (f"  |b_dec| {bd.norm().item():.3f}"
                            f"  shift {(bd - b_dec_init_dev).norm().item():.3f}")
                print(msg)
                running_loss = 0.0
                running_aux = 0.0

    fire_counts = fire_counts.cpu()
    n_dead = int((fire_counts == 0).sum())
    print(f"done: {n_dead}/{dict_size} features never fired in training")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "model.pt")

    # Provenance of the cache itself (layer, model, whether the final LN was
    # applied) travels with the checkpoint -- otherwise "which activations was
    # this trained on" is only answerable by finding the original h5.
    with h5py.File(args.activations[0], "r") as hf:
        cache_attrs = {k: _jsonable(v) for k, v in hf.attrs.items()}

    # Hash the FINAL b_dec in the checkpoint. In ensemble mode (frozen) this
    # equals the init and pins that every seed shared the SAME vector -- the
    # locked invariant; a disagreement means it was not actually shared. In
    # pilot mode (trainable) this is the CONVERGED value the extractor will read.
    b_dec_final = model.b_dec.detach().cpu().float().numpy()
    b_dec_sha = hashlib.sha256(b_dec_final.tobytes()).hexdigest()[:16]
    b_dec_init_sha = hashlib.sha256(
        b_dec.detach().cpu().numpy().tobytes()).hexdigest()[:16]

    meta = {
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "config_hash": config_hash(args.config),
        "seed": args.seed,
        # Affects dict/set iteration order; must be set before interpreter
        # start, so it can only be recorded here, not enforced.
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "activations": [str(p) for p in args.activations],
        "activations_attrs": cache_attrs,
        "n_rows": dataset.n_rows,
        "block_rows": args.block_rows,
        "buffer_blocks": args.buffer_blocks,
        "d_model": d_model,
        "dict_size": dict_size,
        "k": k,
        "lr": lr,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "loss_reduction": reduction,
        "optimizer": optimizer_name,
        "grad_clip": grad_clip,
        "decoder_grad_projection": decoder_grad_projection,
        "b_dec_trainable": b_dec_trainable,
        "b_dec_init_sha256_16": b_dec_init_sha,
        "aux_k": aux_k,
        "aux_coef": aux_coef,
        "dead_tokens_threshold": dead_tokens_threshold,
        "b_dec_path": str(b_dec_path),
        "b_dec_source": sae_cfg.get("b_dec_source"),
        "b_dec_sha256_16": b_dec_sha,
        "deterministic_cudnn": args.deterministic_cudnn,
        # amendment CQ, A5: the device the optimisation ran on, read off W_dec.
        "device_class": device_class(model.W_dec),
        # §CT29-F item 1: where the job ran, so a partition choice is on the
        # checkpoint itself and not only in a log (None outside SLURM).
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "n_dead_features": n_dead,
        "fire_counts": fire_counts.tolist(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
