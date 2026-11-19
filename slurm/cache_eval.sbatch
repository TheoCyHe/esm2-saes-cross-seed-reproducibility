#!/bin/bash
#SBATCH --job-name=esm2_cache_eval
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; torch cu126 has no sm_120 (Blackwell) kernels
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00         # ~17.7M residues at ~9.3k res/s ~= 32 min
#SBATCH --output=logs/%x_%j.out

# Eval cache: a single job with FULL residue coverage. NOT sharded and NOT
# subsampled -- residue-level AP needs every residue. It is small enough
# (17.7M residues / ~45 GB) to build in one job.

set -euo pipefail

SCRATCH=$SCRATCH
SPLIT=$SCRATCH/swissprot/split
CACHE=$SCRATCH/cache

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
mkdir -p "$CACHE"

python -m src.embeddings.cache \
    --layer 24 \
    --fasta "$SPLIT/eval.fasta" \
    --out   "$CACHE/eval.h5"

python -m src.embeddings.verify_cache --cache "$CACHE/eval.h5" --fasta "$SPLIT/eval.fasta"
