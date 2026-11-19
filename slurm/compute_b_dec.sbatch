#!/bin/bash
#SBATCH --job-name=sae_compute_bdec
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G              # Weiszfeld on 1M x 1280 in float64 peaks ~25 GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# Geometric-median b_dec from the train cache -- the PILOT INITIALIZER (open Q5),
# NOT the final ensemble bias. Reads a fixed-seed sample across all shards.
# CPU only.

set -euo pipefail

SCRATCH=$SCRATCH
CACHE=$SCRATCH/cache

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

SHARDS=("$CACHE"/train/shard_*.h5)
python -m src.sae.compute_b_dec \
    --config configs/mvp.yaml \
    --activations "${SHARDS[@]}" \
    --out "$CACHE/b_dec_gm_init.pt"
