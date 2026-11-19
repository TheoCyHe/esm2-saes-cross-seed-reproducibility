#!/bin/bash
#SBATCH --job-name=sae_pilot_bdec
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; torch cu126 has no sm_120 (Blackwell) kernels
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-8             # 3 lr x 3 seeds; index = lr_i*3 + seed
#SBATCH --output=logs/%x_%A_%a.out

# Combined b_dec PILOT + lr sweep (CLAUDE.md open Q5 + open Q3), one phase:
#   - TRAINABLE b_dec (GM init) converges toward b* on each run;
#     extract_frozen_b_dec.py then averages the seeds into the frozen ensemble
#     b_dec and reports cross-seed spread (freeze-vs-trainable ablation).
#   - Sweeping lr {1e-4, 2e-4, 4e-4} lets us pick the ensemble lr by final
#     mse / dead fraction (task 17). b* is ~lr-independent, so the b_dec values
#     across lr also cross-check convergence.
# Validated: seed 0 @ 2e-4 converged by ~20k (mse ~12.6, 0 dead); 30k tightens
# the last ~2%. PROVISIONAL still: batch 4096 (GPU-mem fit, worked).

set -euo pipefail

LRS=(1e-4 2e-4 4e-4)
LR=${LRS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=$((SLURM_ARRAY_TASK_ID % 3))
PILOT_BS=4096
PILOT_STEPS=30000

SCRATCH=$SCRATCH
CACHE=$SCRATCH/cache
PILOT_OUT=$SCRATCH/pilot

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

SHARDS=("$CACHE"/train/shard_*.h5)
if [[ ! -e "${SHARDS[0]}" ]]; then
    echo "no train shards in $CACHE/train" >&2; exit 1
fi
if [[ ! -f "$CACHE/b_dec_gm_init.pt" ]]; then
    echo "missing $CACHE/b_dec_gm_init.pt -- run compute_b_dec.sbatch first" >&2; exit 1
fi

echo "pilot: lr=$LR seed=$SEED steps=$PILOT_STEPS"
python -m src.sae.train \
    --config configs/mvp.yaml \
    --activations "${SHARDS[@]}" \
    --b-dec "$CACHE/b_dec_gm_init.pt" \
    --b-dec-trainable \
    --seed "$SEED" \
    --lr "$LR" --batch-size "$PILOT_BS" --n-steps "$PILOT_STEPS" \
    --log-every 500 \
    --out "$PILOT_OUT/lr${LR}_seed${SEED}"

# After all 9 finish:
#  - lr choice: compare final mse / dead across lr${LR}_seed*/meta.json
#  - frozen b_dec: python -m src.sae.extract_frozen_b_dec \
#       --pilot-ckpts $PILOT_OUT/lr<best>_seed0 ..._seed1 ..._seed2 \
#       --out $CACHE/b_dec.pt
