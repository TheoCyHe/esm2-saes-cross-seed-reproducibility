#!/bin/bash
#SBATCH --job-name=sae_bdec_actcost
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; torch cu126 has no sm_120 (Blackwell) kernels
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

# Measurement-noise cost of a trainable per-seed b_dec (open Q5): hold seed 0's
# dictionary fixed, swap in seeds 1 and 2's b_dec, and report per-feature
# activation Pearson + firing-set Jaccard binned by firing rate. Defaults to the
# lr=2e-4 pilots.

set -euo pipefail

SCRATCH=$SCRATCH
LR=${1:-2e-4}

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

python -m src.sae.measure_b_dec_activation_cost \
    --config configs/mvp.yaml \
    --model-ckpt "$SCRATCH/pilot/lr${LR}_seed0" \
    --swap-ckpts "$SCRATCH/pilot/lr${LR}_seed1" "$SCRATCH/pilot/lr${LR}_seed2" \
    --eval-cache "$SCRATCH/cache/eval.h5" \
    --n-rows 100000
