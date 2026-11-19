#!/bin/bash
#SBATCH --job-name=sae_derive_null
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# Re-derive null reductions on the COMPLIANT axis from the K3 raw array.
# CPU-only; streams the raw array one (seed, shift) slice at a time.
# NOT on the login node -- this reads ~25 GB.

set -euo pipefail
SCRATCH=$SCRATCH
PROJECT=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

python -m src.analysis.derive_null_reductions \
    --raw "$PROJECT/analysis/null_gain_raw_compliant.h5" \
    --labels "$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5" \
    --source-reductions "$PROJECT/analysis/null_pilot_k.h5" \
    --out "$PROJECT/analysis/null_pilot_compliant.h5" \
    --mark-source-stale
