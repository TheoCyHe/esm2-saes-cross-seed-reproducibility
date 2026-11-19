#!/bin/bash
#SBATCH --job-name=sae_null_exp_pilot
#SBATCH --partition=gpu_devel
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
# GPU AUDIT (2026-08-10): A ONE-OFF, AND THE GPU REQUEST IS UNNECESSARY.
# Measured on job 21438269: the gather is 268.1 s of 3,221 s elapsed = 8.3%;
# the other 91.7% is masked AP arithmetic that `feature_ap.py`'s
# NNZ_GPU_SAFE threshold has already routed to CPU. The array this pilot sized
# has since run, on CPU, so re-running it is not expected. NOT converted to a
# CPU partition here because that would change WHAT IT MEASURES -- a cost pilot
# reports the cost of the device it ran on -- and this file is the record of
# the GPU-path timing. If it is ever re-run to size a CPU array, move it to
# `--partition=day` and treat the new numbers as a different measurement.

# AC5 -- measure the displacement cost BEFORE launching the array.
#
# WHY gpu_devel. CLAUDE.md's partition note (now its Cluster section) was written
# about a SIX-HOUR cache run; the guidance is about WALLTIME, not legitimacy,
# and reading it otherwise once cost five hours of queue for a job that ran in
# 5m28s. This pilot is well under gpu_devel's 6h QOS wall. The architecture is
# still requested explicitly, because gpu_devel is heterogeneous and a
# Blackwell node has no cu126 kernels (gotcha 7).
#
# WHAT IS MEASURED, and why the shape of the pilot matters. 12 concepts x 20
# shifts x 2 arms = 480 masked evaluations. The 20-shifts-x-2-arms figure is
# NOT reduced: the per-concept filtered non-zeros are built ONCE and reused by
# all 40 of that concept's draws, so the amortisation ratio is exactly
# production's. Reducing the shift count instead of the concept count would
# have measured a per-evaluation cost that does not transfer.
#
# The 12 concepts are EVENLY SPACED BY MASKED-ROW COUNT (--pilot-n-concepts),
# because the mask size is what drives the filter, not the alphabet. The
# output is stamped `is_cost_pilot=True` with `argmax_pool_used_n` disagreeing
# with `argmax_pool_scored_n`, so a timing run cannot become an artifact of
# record.
#
# NO --raw-out. A pilot has no consumer.

set -euo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT
LABELS="$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5"
SEED="${1:-0}"
CKPT="$PROJECT/sae_ckpts/seed_$SEED"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

if [[ ! -f "$CKPT/model.pt" ]]; then
    echo "FATAL: no checkpoint at $CKPT" >&2; exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m src.analysis.null_pilot \
    --config configs/mvp.yaml \
    --ckpts "$CKPT" \
    --eval-cache "$SCRATCH/cache/eval.h5" \
    --labels "$LABELS" \
    --scored-row-set label_store \
    --stratum experimental \
    --negatives unannotated \
    --null-type both \
    --n-length-bins 20 \
    --n-shifts 20 \
    --null-seed 0 \
    --pilot-n-concepts 12 \
    --cache-dir "$SCRATCH/row_index_cache" \
    --exclude-pooled-parents \
    --ap-device auto \
    --out "$SCRATCH/ac5_pilot/null_exp_cost_pilot_seed${SEED}.h5"

echo "AC5 pilot complete. Read the per-evaluation seconds off the "
echo "'masked evaluations in Ns = X.XXXs each' line above."
