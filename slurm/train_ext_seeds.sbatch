#!/bin/bash
# ARCHIVED PRODUCER -- ran from $PROJECT/analysis/adhoc/
# (sacct WorkDir; SubmitLine `sbatch train_ext_seeds.sbatch`) on 2026-07-29, array
# 20438265_20..39, submitted 20:19:34, tasks 20:19:50 to 22:00:51, all COMPLETED,
# against SHA 61745baa1d87a46313917fb41c18ba9dfbd16ad2 -- the git_sha recorded in
# all twenty sae_ckpts_ext/seed_20..39/meta.json (git_dirty false, config_hash
# 3a5a0c99a8ed7ade in all twenty; identical to the twenty anchors), executed from
# the pinned worktree $HOME/esm2_train_pin61745ba per amendment AD4.
# Committed unchanged on 2026-09-15 so the released tree contains the code that
# produced $PROJECT/sae_ckpts_ext/seed_20..39. Not re-run since. See
# docs/DATAFLOW.md section 4 (stage 4 row). sha256 of the body below, cluster and
# tree: 98d8ed88dd75c71cb065facc6df266ad1450775906d50b400a5290be8b7b2a81.
#SBATCH --job-name=sae_train_ext
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; torch cu126 has no sm_120 (Blackwell) kernels
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=20-39             # the S=40 EXTENSION seeds (AB4/AD4)
#SBATCH --output=$PROJECT/analysis/adhoc/logs/%x_%A_%a.out

# From `mydirectories` on Bouchet (group <allocation>).
SCRATCH=$SCRATCH
PROJECT=$PROJECT
CACHE=$SCRATCH/cache

cd $HOME/esm2_train_pin61745ba   # PINNED at 61745ba (amendment AD4)
source env/activate.sh

# All shards, in sorted (== numeric) order -- pinned, since file order sets the
# loader's block layout. No merge step: train.py pools blocks across the list.
SHARDS=("$CACHE"/train/shard_*.h5)
if [[ ! -e "${SHARDS[0]}" ]]; then
    echo "no train shards in $CACHE/train -- run shard_fasta.py + cache_train_array.sbatch first" >&2
    exit 1
fi
# b_dec is TRAINABLE from a shared GM init (open Q5, RESOLVED): each seed trains
# its own b_dec from this one shared vector. Path + b_dec_trainable come from
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
    --out "$PROJECT/sae_ckpts_ext/seed_${SLURM_ARRAY_TASK_ID}"
