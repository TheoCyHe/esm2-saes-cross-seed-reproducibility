#!/bin/bash
#SBATCH --job-name=cf_fvu_eval
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; and the device class is RECORDED in
                                # the artifact, per gotcha 36
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# FVU on the evaluation split, in BOTH spaces. PREREGISTRATION amendment CF-a,
# which is committed BEFORE this script existed.
#
# USAGE: sbatch slurm/fvu_eval.sbatch [smoke]
#   smoke  -- --max-rows 500000, writes a NON-COMPLIANT artifact to a separate
#             path. CF-a fixes the FULL split, and the output records which it
#             is rather than leaving a reader to infer it.

set -euo pipefail
MODE="${1:-full}"
PROJECT=$PROJECT
SCRATCH=$SCRATCH

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
echo "[fvu.sbatch] git $(git rev-parse HEAD)  status [$(git status --porcelain | tr '\n' ';')]"

EXTRA=()
if [[ "$MODE" == "smoke" ]]; then
    OUT="$PROJECT/analysis/fvu_eval_cf_SMOKE.json"
    EXTRA=(--max-rows 500000)
else
    OUT="$PROJECT/analysis/fvu_eval_cf.json"
fi
if [[ -e "$OUT" ]]; then
    echo "$OUT already exists -- refusing to overwrite" >&2; exit 1
fi

# Anchor seeds NAMED, never range(n) (gotcha 29).
python -m src.analysis.fvu \
    --cache  "$SCRATCH/cache/eval.h5" \
    --ckpts  "$PROJECT/sae_ckpts" \
    --seeds  0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --k 32 \
    "${EXTRA[@]+"${EXTRA[@]}"}" \
    --out "$OUT"
