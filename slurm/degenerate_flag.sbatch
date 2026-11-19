#!/bin/bash
#SBATCH --job-name=sae_degenerate_flag
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

# THE DEGENERATE-ARGMAX INDICATOR, PERSISTED -- amendment AO section 8,
# The author's ratified decision AM6.3.
#
# NOT a firewall crossing: reads argmax_concept, concepts and M. No r_f.
# NO REGRESSION IS RUN -- the regression is the design-3 join, next session.
#
# The job FAILS if the recomputed shares do not reproduce AL9b exactly (AO4 A1);
# that is a correctness assertion on the same computation over the same inputs,
# not a prediction.

set -euo pipefail
SCRATCH=$SCRATCH
PROJECT=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

python -m src.analysis.degenerate_flag \
    --gtilde "$PROJECT/analysis/gtilde_compliant_b_ratio.h5" \
    --degenerate "$PROJECT/analysis/mu_s_degenerate_m3.json" \
    --out "$PROJECT/analysis/degenerate_argmax_flag.h5" \
    --out-json "$PROJECT/analysis/degenerate_argmax_flag.json"
