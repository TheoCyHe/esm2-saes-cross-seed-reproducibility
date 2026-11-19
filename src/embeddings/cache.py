"""
Cache ESM-2 residual-stream activations at a single layer.

Design notes (these are the silent-corruption risks; do not "simplify" them away):

1. SPECIAL TOKENS. HF's ESM tokenizer wraps each sequence as
       <cls> r_1 r_2 ... r_L <eos>
   so hidden_states have length L+2. We slice [1 : 1+L]. Row i of the output
   therefore corresponds to residue i (0-based) of the raw sequence. An
   off-by-one here silently wrecks every downstream AP number, so it is
   asserted per sequence rather than trusted.

2. LAYER INDEXING. `hidden_states` is a tuple of length n_layers+1 where
   hidden_states[0] is the embedding output and hidden_states[i] is the output
   of transformer block i. So hidden_states[24] == "layer 24" in the sense used
   by InterPLM/InterProt. VERIFIED (src/embeddings/verify_layer.py, 2026-07-24):
   InterProt's released layer-24 SAE reconstructs hidden_states[24] better than
   any neighbouring layer, at every k tested -- so this indexing is confirmed,
   not assumed.

3. FINAL LAYER NORM. ESM-2 applies `emb_layer_norm_after` only after the last
   block. Intermediate hidden_states are therefore pre-that-LN. Reference SAEs
   were trained on the raw residual stream at the layer, so we do NOT apply it.
   Recorded in the file attrs so the choice is auditable.

4. TRUNCATION. ESM-2's HF config caps positions at 1026 (=1024+2). Proteins
   longer than `--max-len` are truncated and the truncation is RECORDED in the
   index. If you drop that record, annotation alignment silently misindexes on
   long proteins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.analysis.artifact_identity import device_class
from src.experiment_config import DEFAULT_CONFIG_PATH, load_config, resolve

ESM_DIM = 1280  # esm2_t33_650M_UR50D
H5_CHUNK_ROWS = 4096          # ~10 MB per chunk at 1280 fp16
WRITE_CHUNK_ROWS = H5_CHUNK_ROWS  # flush threshold; keep aligned to the chunking


def strip_and_subsample(
    resid: torch.Tensor,
    L: int,
    attention_mask_sum: int,
    acc: str,
    max_residues: int | None,
    keep_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate special-token stripping for one sequence and optionally
    subsample its residues. Pulled out of the main loop so the two
    silent-corruption risks it guards against (see module docstring, points
    1 and 4) are unit-testable without a real ESM forward pass.

    Returns (fp16 array of shape (n_kept, ESM_DIM), residue ids kept -- or
    None if not subsampled).
    """
    # Assertion 1: one activation row per residue.
    assert resid.shape[0] == L, (
        f"{acc}: got {resid.shape[0]} rows for {L} residues -- "
        "special-token stripping is wrong"
    )
    # Assertion 2: we never sliced into padding.
    assert attention_mask_sum >= L + 2, (
        f"{acc}: attention mask shorter than sequence + specials"
    )

    # The model runs in fp16 and the cache is fp16, so cast on-device and
    # transfer once. Going via .float() would double the GPU->CPU copy only
    # to immediately discard the extra precision.
    arr = resid.to(torch.float16).cpu().numpy()
    rows = None

    if max_residues is not None:
        n_keep = max(1, int(round(L * keep_frac)))
        rows = np.sort(rng.choice(L, size=n_keep, replace=False))
        arr = arr[rows]

    return arr, rows


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Minimal FASTA reader -> [(accession, sequence)]. Accession = first
    whitespace-delimited token of the header."""
    records: list[tuple[str, str]] = []
    acc: str | None = None
    chunks: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if acc is not None:
                    records.append((acc, "".join(chunks)))
                acc = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if acc is not None:
        records.append((acc, "".join(chunks)))
    return records


def build_parser() -> argparse.ArgumentParser:
    """The CLI. `--model`, `--layer` and `--max-len` default to None and are
    resolved against `--config` by `resolve_experiment_args` (amendment CR):
    an explicit value must equal the config's or the run is refused."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                   help="configs/mvp.yaml; the source of model.name, "
                        "model.layer and model.max_len. Defaults to the "
                        "repository's own copy.")
    p.add_argument("--model", default=None,
                   help="model.name from --config unless restated here.")
    p.add_argument("--layer", type=int, default=None,
                   help="model.layer from --config unless restated here.")
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-len", type=int, default=None,
                   help="Residues kept per protein (model.max_len from --config, "
                        "1022): with <cls> and <eos> that is ESM-2's 1024-token "
                        "window. Longer proteins are truncated and the "
                        "truncation is recorded in the index.")
    p.add_argument(
        "--max-residues", type=int, default=None,
        help="Cap total cached residues by random per-protein subsampling. "
             "Use for TRAIN caches (an SAE does not need all of Swiss-Prot). "
             "Leave unset for EVAL caches, which need full residue coverage "
             "for annotation alignment.",
    )
    p.add_argument("--seed", type=int, default=0,
                   help="Seeds the subsampling only. Kept separate from SAE seeds.")
    return p


def resolve_experiment_args(args: argparse.Namespace) -> argparse.Namespace:
    """Fill `model`, `layer`, `max_len` from the config, or refuse a CLI value
    that disagrees with it. Runs BEFORE the model is loaded, so the refusal
    needs no GPU and no `transformers`. Mutates and returns `args`."""
    cfg = load_config(args.config)
    args.model = resolve(cfg, "model.name", args.model, "--model", args.config)
    args.layer = resolve(cfg, "model.layer", args.layer, "--layer", args.config)
    args.max_len = resolve(cfg, "model.max_len", args.max_len, "--max-len",
                           args.config)
    return args


def main() -> None:
    args = resolve_experiment_args(build_parser().parse_args())

    # Deferred: only main() needs the model/tokenizer, so the helpers above
    # stay importable/testable without transformers installed -- and the
    # config refusal above fires before this import is attempted.
    from transformers import AutoTokenizer, EsmModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU visible. Are you on a login node, or did you "
              "forget --gpus=1 on the SLURM request?")

    records = read_fasta(args.fasta)
    print(f"{len(records)} sequences from {args.fasta}")

    # Length-sorted batching minimises padding waste. Order is restored via the
    # index table, so this does not affect correctness.
    records.sort(key=lambda r: len(r[1]))

    # Subsampling fraction is derived from the ACTUAL residue total, not a
    # guessed mean length, so --max-residues is a real budget.
    total_residues = sum(min(len(s), args.max_len) for _, s in records)
    keep_frac = 1.0
    if args.max_residues is not None:
        keep_frac = min(1.0, args.max_residues / max(1, total_residues))
        print(f"{total_residues:,} residues available; keeping {keep_frac:.1%} "
              f"(budget {args.max_residues:,})")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # add_pooling_layer=False: the MLM checkpoint has no pooler, so EsmModel
    # would randomly initialize one ("pooler.dense.weight MISSING" in the load
    # report). We only read hidden_states, so it is dead weight -- but this
    # project varies only the seed, and an unused randomly-initialized layer is
    # exactly the kind of stray nondeterminism not worth carrying.
    #
    # .half() before .to(device): casting on CPU first halves the bytes copied
    # across PCIe, and keeps the failure mode legible if the GPU arch is
    # unsupported (see the sm_120 note in slurm/*.sbatch).
    model = EsmModel.from_pretrained(
        args.model, add_pooling_layer=False
    ).eval().half().to(device)

    rng = np.random.default_rng(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.out, "w") as hf:
        acts = hf.create_dataset(
            "activations", shape=(0, ESM_DIM), maxshape=(None, ESM_DIM),
            dtype="float16", chunks=(H5_CHUNK_ROWS, ESM_DIM), compression=None,
        )
        hf.attrs.update(
            model=args.model, layer=args.layer, dtype="float16",
            final_layer_norm_applied=False, max_len=args.max_len,
            subsampled=args.max_residues is not None, seed=args.seed,
            # amendment CQ, A5: the device the forward pass ran on, read off
            # the model's parameters rather than off the request (gotcha 36).
            device_class=device_class(next(model.parameters()).device),
        )

        index: list[dict] = []
        cursor = 0
        n_truncated = 0

        # Writes are buffered to whole chunks. Writing one protein at a time
        # into a (4096, 1280) chunked dataset forces a read-modify-write of a
        # ~10 MB chunk per sequence, which dominates runtime on a cache this
        # size. Buffering is invisible to the index: `cursor` stays the logical
        # row offset and is asserted against the physical write position below.
        write_buf: list[np.ndarray] = []
        buf_rows = 0
        write_pos = 0

        def flush() -> None:
            nonlocal write_buf, buf_rows, write_pos
            if not write_buf:
                return
            block = np.concatenate(write_buf, axis=0)
            acts.resize(write_pos + block.shape[0], axis=0)
            acts[write_pos: write_pos + block.shape[0]] = block
            write_pos += block.shape[0]
            write_buf = []
            buf_rows = 0

        for start in tqdm(range(0, len(records), args.batch_size), desc="embedding"):
            batch = records[start:start + args.batch_size]
            accs = [a for a, _ in batch]
            # A sequence of exactly max_len is NOT truncated -- derive the flag
            # from the raw length before slicing, so the per-record flag and the
            # n_truncated counter cannot disagree.
            was_truncated = [len(s) > args.max_len for _, s in batch]
            seqs = [s[: args.max_len] for _, s in batch]
            n_truncated += sum(was_truncated)
            lengths = [len(s) for s in seqs]

            enc = tokenizer(seqs, return_tensors="pt", padding=True,
                            add_special_tokens=True).to(device)
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states[args.layer]

            for j, (acc, L) in enumerate(zip(accs, lengths)):
                # Strip <cls> at 0 and <eos> at L+1. Padding sits beyond L+1.
                resid = hs[j, 1: 1 + L, :]
                attention_mask_sum = int(enc["attention_mask"][j].sum())

                arr, rows = strip_and_subsample(
                    resid, L, attention_mask_sum, acc,
                    args.max_residues, keep_frac, rng,
                )

                write_buf.append(arr)
                buf_rows += arr.shape[0]
                if buf_rows >= WRITE_CHUNK_ROWS:
                    flush()

                index.append({
                    "accession": acc,
                    "start": cursor,
                    "n_rows": int(arr.shape[0]),
                    "seq_len": L,
                    "truncated": bool(was_truncated[j]),
                    # Residue positions retained, so subsampled train caches can
                    # still be mapped back to annotations if ever needed.
                    "residue_ids": rows.astype(np.int32).tolist()
                    if rows is not None else None,
                })
                cursor += arr.shape[0]

        flush()
        # If these ever disagree, every `start` offset in the index is wrong
        # and the whole cache is silently misaligned.
        assert write_pos == cursor, (
            f"wrote {write_pos} rows but index accounts for {cursor}"
        )

        # np.string_ was removed in numpy 2.0; np.bytes_ is the replacement.
        hf.create_dataset(
            "index_json", data=np.bytes_(json.dumps(index)),
        )

    print(f"wrote {cursor:,} residues x {ESM_DIM} fp16 -> {args.out}")
    print(f"  ~{cursor * ESM_DIM * 2 / 1e9:.1f} GB")
    if n_truncated:
        print(f"  {n_truncated} proteins truncated to {args.max_len} residues (recorded in index)")


if __name__ == "__main__":
    main()
