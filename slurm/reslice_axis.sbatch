#!/bin/bash
#SBATCH --job-name=sae_reslice_axis
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; torch cu126 has no sm_120 kernels
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# GPU AUDIT (2026-08-10): THE DEVICE MUST STAY. This reslices
# `feature_ap_main.h5` -- produced on rtx_5000_ada -- and then VERIFIES the
# slice by recomputing AP for probe concepts and requiring BIT-IDENTITY against
# the stored values. Recomputing on CPU would fail that check for device
# reasons rather than for slicing reasons (gotcha 36), which is the worst kind
# of failure: a real criterion made to report on the wrong thing. Its gather is
# normally a `row_index_cache` HIT, so the GPU cost is small either way.

# Re-slice AP + raw null onto the COMPLIANT concept axis (amendment N1).
# A SUBSET, not a re-run: g(f,c) depends only on c's own positives and the row
# population, neither of which changes when a different concept is dropped.
#
# GPU is needed only for the VERIFICATION arm, which recomputes AP for a few
# concepts from the activations to prove the concept axis is indexed by
# IDENTITY rather than position. The slicing itself is pure I/O.
#
# Writes to NEW versioned paths and refuses to overwrite: the existing
# artifacts are the only record of the numbers already reported.

set -euo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT
LABELS="${1:?pass the compliant label store as \$1}"
TAG="${2:?pass a short tag for the output filenames as \$2}"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

# The compliant store must itself pass the spec check before anything is
# sliced onto its axis (amendment M2).
python -m src.analysis.preflight --config configs/mvp.yaml --labels "$LABELS"

python -m src.analysis.reslice_axis \
    --config configs/mvp.yaml \
    --labels "$LABELS" \
    --ap        "$PROJECT/analysis/feature_ap_main.h5" \
    --raw-null  "$PROJECT/analysis/null_gain_raw_k.h5" \
    --out-ap    "$PROJECT/analysis/feature_ap_main_${TAG}.h5" \
    --out-null  "$SCRATCH/null_raw/null_gain_raw_${TAG}.h5" \
    --verify-ckpt "$PROJECT/sae_ckpts/seed_0" \
    --eval-cache  "$SCRATCH/cache/eval.h5" \
    --cache-dir   "$SCRATCH/row_index_cache" \
    --n-probe 5 \
    --out-report "$PROJECT/analysis/reslice_${TAG}.json"
