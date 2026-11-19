#!/bin/bash
#SBATCH --job-name=sae_predictor_support
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

# AL5 and AL6 -- characterising the predictor M before the design-3 join.
# NOT a firewall crossing: reads M, argmax_concept and stratum_mu; no r_f.

set -euo pipefail
SCRATCH=$SCRATCH
PROJECT=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

python -m src.analysis.predictor_support \
    --gtilde "$PROJECT/analysis/gtilde_compliant_b_ratio.h5" \
    --degenerate "$PROJECT/analysis/mu_s_degenerate_m3.json" \
    --mu-floor 1e-8 \
    --out "$PROJECT/analysis/predictor_support_al.json"
