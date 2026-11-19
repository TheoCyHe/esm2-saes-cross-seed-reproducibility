#!/bin/bash
#SBATCH --job-name=sae_diag_std
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

# The four measurements amendment P's next step depends on (3a-3d).
# CPU-only; reads two seeds of the raw null (~576 MB each).
set -euo pipefail
SCRATCH=$SCRATCH
PROJECT=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

python -m src.analysis.diagnose_standardization \
    --gtilde      "$SCRATCH/analysis/gtilde_compliant_b.h5" \
    --ap          "$PROJECT/analysis/feature_ap_main_compliant.h5" \
    --null-raw    "$PROJECT/analysis/null_gain_raw_compliant.h5" \
    --labels-json "$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.json" \
    --out         "$PROJECT/analysis/standardization_diagnosis.json"
