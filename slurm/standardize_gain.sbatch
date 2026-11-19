#!/bin/bash
#SBATCH --job-name=sae_standardize
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# K2's three-stage standardization (PREREGISTRATION amendment K2) on
# amendment O2's four pooling axes: prevalence x firing rate x surviving
# protein count x cluster count.
#
# CPU-only. Reads the 23 GB K3 raw gain array one SEED at a time (~576 MB per
# seed per arm), the compliant AP matrix, and `fire_rate` from the compliant
# reductions. Not on the login node.
#
# The pre-flight gate runs INSIDE the script -- on the inputs before anything
# is computed, and on the output afterwards. `check()` raises, so `set -e`
# turns a mismatch into a non-zero exit that an `afterok` chain can read
# (gotcha 13, extension 1).

set -euo pipefail
SCRATCH=$SCRATCH
PROJECT=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

ARM="${1:-b}"                 # Null B is primary (amendment G4)
AXES="${2:-prevalence,protein_count,cluster_count}"
OUT="${3:-$SCRATCH/analysis/gtilde_compliant_${ARM}.h5}"
mkdir -p "$(dirname "$OUT")"

python -m src.analysis.standardize_gain \
    --ap          "$PROJECT/analysis/feature_ap_main_compliant.h5" \
    --null-raw    "$PROJECT/analysis/null_gain_raw_compliant.h5" \
    --null-reductions "$PROJECT/analysis/null_pilot_compliant.h5" \
    --labels      "$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5" \
    --arm         "$ARM" \
    --concept-axes "$AXES" \
    --config      configs/mvp.yaml \
    --out         "$OUT"
