#!/bin/bash
#SBATCH --job-name=sae_d5_breadth
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# D5 recomputation on the current axis, against amendment I5d's PRE-DECLARED
# success criteria (committed before main_ap ran).
#
# CPU-only: it reads feature_ap_main.h5 (~586 MB of AP) and streams the K3 raw
# null gain array (25.2 GB on scratch) one (seed, shift) slice at a time to get
# the argmax concept per feature under shuffled labels -- criterion (b)'s
# reference. No GPU needed, so this does not queue behind the gpu partition.
#
# NOT ON THE LOGIN NODE. The raw-null stream alone is 25 GB of I/O; CLAUDE.md's
# rule against login-node compute applies squarely.

set -euo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

python -m src.analysis.d5_breadth \
    --ap "$PROJECT/analysis/feature_ap_main.h5" \
    --raw-null "$PROJECT/analysis/null_gain_raw_k.h5" \
    --null-arm b \
    --out "$PROJECT/analysis/d5_breadth_main.json"
