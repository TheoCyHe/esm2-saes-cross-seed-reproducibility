#!/bin/bash
#SBATCH --job-name=ct_class_floor
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# Amendment CT, §CT11.3 (CF00198): the 2x2 classification at A2's pre-specified
# sensitivity floor G >= 0.01, beside the 0.05 of record.
#
#   sbatch slurm/ct_classification_floor.sbatch 0.05 control   # must reproduce
#   sbatch slurm/ct_classification_floor.sbatch 0.01 g001      #   classification_2x2_bq
#
# The 0.05 run of record (job 22394396) read the pooled raw null from SCRATCH,
# since purged; the durable copy on project is used here, and the 0.05 control
# is what shows it is the same input (its counts must equal the stored ones).
set -euo pipefail
FLOOR="${1:?floor}"
TAG="${2:?tag}"
PROJECT=$PROJECT
A="$PROJECT/analysis"
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
[[ -z "$(git status --porcelain)" ]] || { echo "checkout dirty" >&2; exit 1; }
mkdir -p "$A/ct"
OUT="$A/ct/classification_2x2_${TAG}_ct.h5"
[[ -e "$OUT" ]] && { echo "$OUT exists -- refusing to overwrite" >&2; exit 1; }
python -m src.analysis.classification_2x2 \
    --bh-experimental "$A/classification_bh_experimental_bo.h5" \
    --ap-experimental "$A/feature_ap_experimental_av_cpu.h5" \
    --ap-pooled "$A/feature_ap_main_compliant.h5" \
    --raw-null-pooled "$A/null_gain_raw_compliant.h5" \
    --null-pooled-reductions "$A/null_pilot_compliant.h5" \
    --gain-floor "$FLOOR" \
    --out "$OUT" \
    --report "$A/ct/classification_2x2_${TAG}_ct.json"
[[ -z "$(git status --porcelain)" ]] || { echo "checkout dirty on exit" >&2; exit 1; }
