#!/bin/bash
#SBATCH --job-name=sae_corrector_scan
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; torch cu126 has no sm_120 (Blackwell) kernels
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# The GATE in CLAUDE.md's locked decisions (open Q5): scan every
# ensemble seed for offset-corrector latents -- always-on, constant-magnitude
# latents acting as a bias, which would be reproducing top-bin ORPHANS and could
# move the firing-rate-matched comparison invisibly.
#
# Output JSON carries per-seed flagged_features; their downstream handling is
# PREREGISTRATION.md §5 (corrector exclusion).

set -euo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

CKPTS=("$PROJECT"/sae_ckpts/seed_*)
if [[ ! -e "${CKPTS[0]}/model.pt" ]]; then
    echo "no ensemble checkpoints in $PROJECT/sae_ckpts -- run train_sae_array.sbatch first" >&2
    exit 1
fi

python -m src.analysis.corrector_scan \
    --config configs/mvp.yaml \
    --ckpts "${CKPTS[@]}" \
    --eval-cache "$SCRATCH/cache/eval.h5" \
    --n-rows 200000 \
    --out "$PROJECT/analysis/corrector_scan.json"
