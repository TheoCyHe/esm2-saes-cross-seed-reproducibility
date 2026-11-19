#!/bin/bash
#SBATCH --job-name=esm2_cache_train
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; torch cu126 has no sm_120 (Blackwell) kernels
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00         # ~13 min/shard at ~9.3k res/s; headroom for queue jitter
#SBATCH --array=0-23            # MUST match runtime.cache_shards (24). If you
                                # re-shard with a different N, override on submit:
                                #   sbatch --array=0-$((N-1)) slurm/cache_train_array.sbatch
#SBATCH --output=logs/%x_%A_%a.out

# One residue-balanced train shard -> one .h5. No merge step: train.py accepts
# the whole shard glob (the loader pools blocks across files). Run shard_fasta.py
# FIRST to produce the shard FASTAs and shards.json.

set -euo pipefail

SCRATCH=$SCRATCH
SHARDS=$SCRATCH/swissprot/split/shards
CACHE=$SCRATCH/cache/train

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
mkdir -p "$CACHE"

# 3-digit id matches shard_fasta.py's SHARD_ID_WIDTH.
SHARD=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
FASTA="$SHARDS/shard_${SHARD}.fasta"
OUT="$CACHE/shard_${SHARD}.h5"

if [[ ! -f "$FASTA" ]]; then
    echo "missing shard FASTA $FASTA -- did shard_fasta.py run with the same N?" >&2
    exit 1
fi

# No --max-residues: train_max_residues is null (no cap; CLAUDE.md open Q2).
python -m src.embeddings.cache --layer 24 --fasta "$FASTA" --out "$OUT"

# Validate the shard cache against its own FASTA (per-residue alignment).
python -m src.embeddings.verify_cache --cache "$OUT" --fasta "$FASTA"
