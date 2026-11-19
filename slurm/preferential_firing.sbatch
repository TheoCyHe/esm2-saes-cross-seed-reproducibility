#!/bin/bash
#SBATCH --job-name=sae_preferential_firing
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; torch cu126 has no sm_120 (Blackwell) kernels
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# GPU AUDIT (2026-08-10): NOT A GATHER-ONLY JOB, so it is outside the category
# this audit covers. `preferential_firing.py` keeps `feat`/`pos`/`val` and its
# per-concept boolean masks ON the device and does every candidate's reduction
# there; it never routes anything through `feature_ap.py`'s NNZ_GPU_SAFE
# switch. The GPU is used throughout.

# PREREGISTRATION amendment D6 -- REQUIRED before any experimental-stratum
# alignment is interpreted as biological. For every feature whose arg-max
# concept clears the G-proxy in the experimental stratum, compares its mean
# activation on that concept's EXPERIMENTAL-positive rows against its mean
# activation on the SAME concept's PROPAGATED-positive rows. A large ratio is
# the signature of "detects well-studied protein context" rather than
# chemistry -- flagged and reported, not auto-excluded.
#
# Depends on feature_ap.py's --negatives unannotated output (amendment C, now
# primary) for the candidate list.

set -euo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT
AP="${1:-$PROJECT/analysis/feature_ap_shared_experimental_unannotated.h5}"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

CKPTS=("$PROJECT"/sae_ckpts/seed_*)
if [[ ! -e "${CKPTS[0]}/model.pt" ]]; then
    echo "no ensemble checkpoints in $PROJECT/sae_ckpts" >&2; exit 1
fi
if [[ ! -f "$AP" ]]; then
    echo "missing $AP -- run feature_ap.sbatch (amendment C, unannotated) first" >&2
    exit 1
fi

python -m src.analysis.preferential_firing \
    --config configs/mvp.yaml \
    --ckpts "${CKPTS[@]}" \
    --eval-cache "$SCRATCH/cache/eval.h5" \
    --labels "$PROJECT/analysis/residue_labels.h5" \
    --ap "$AP" \
    --stratum experimental \
    --out "$PROJECT/analysis/preferential_firing.json"
