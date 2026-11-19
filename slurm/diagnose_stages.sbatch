#!/bin/bash
#SBATCH --job-name=sae_diag_stages
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# 2a/2b/2c. Predictions are committed in the module docstring BEFORE this runs.
# Does not touch the 23 GB raw array: null_g_* already IS the raw-gain max.
set -euo pipefail
SCRATCH=$SCRATCH
PROJECT=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
python -m src.analysis.diagnose_stages \
    --gtilde "$SCRATCH/analysis/gtilde_compliant_b.h5" \
    --ap "$PROJECT/analysis/feature_ap_main_compliant.h5" \
    --null-reductions "$PROJECT/analysis/null_pilot_compliant.h5" \
    --out "$PROJECT/analysis/stage_diagnosis.json"
