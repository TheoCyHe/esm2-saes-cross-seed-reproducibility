#!/bin/bash
#SBATCH --job-name=sae_diag_bdec
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; torch cu126 has no sm_120 (Blackwell) kernels
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out

# Freeze-vs-trainable diagnostics on the pilot b_dec (open Q5): reconstruction
# cost of forcing each seed onto the pooled b_dec, residual-relative offset, and
# deviation structure. Defaults to the lr=2e-4 pilot seeds.

set -euo pipefail

SCRATCH=$SCRATCH
LR=${1:-2e-4}

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

python -m src.sae.diagnose_b_dec \
    --config configs/mvp.yaml \
    --pilot-ckpts "$SCRATCH"/pilot/lr${LR}_seed* \
    --eval-cache "$SCRATCH/cache/eval.h5" \
    --n-rows 100000
