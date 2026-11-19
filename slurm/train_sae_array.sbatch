#!/bin/bash
#SBATCH --job-name=sae_train
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; torch cu126 has no sm_120 (Blackwell) kernels
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-19              # 20 seeds
#SBATCH --output=logs/%x_%A_%a.out

# From `mydirectories` on Bouchet (group <allocation>).
SCRATCH=$SCRATCH
PROJECT=$PROJECT
CACHE=$SCRATCH/cache

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

# All shards, in sorted (== numeric) order -- pinned, since file order sets the
# loader's block layout. No merge step: train.py pools blocks across the list.
SHARDS=("$CACHE"/train/shard_*.h5)
if [[ ! -e "${SHARDS[0]}" ]]; then
    echo "no train shards in $CACHE/train -- run shard_fasta.py + cache_train_array.sbatch first" >&2
    exit 1
fi
# b_dec: CLAUDE.md locked decisions (open Q5). Path + b_dec_trainable come from
# configs/mvp.yaml; the init file must exist.
if [[ ! -f "$CACHE/b_dec_gm_init.pt" ]]; then
    echo "missing $CACHE/b_dec_gm_init.pt -- run compute_b_dec.sbatch first" >&2
    exit 1
fi

# lr/batch_size/n_steps/b_dec_path/b_dec_trainable are all set in
# configs/mvp.yaml (from the pilot). Nothing experiment-defining is passed here.
python -m src.sae.train \
    --config configs/mvp.yaml \
    --activations "${SHARDS[@]}" \
    --seed "$SLURM_ARRAY_TASK_ID" \
    --out "$PROJECT/sae_ckpts/seed_${SLURM_ARRAY_TASK_ID}"
