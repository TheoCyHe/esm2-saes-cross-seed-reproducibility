#!/bin/bash
#SBATCH --job-name=sae_bg3_prop
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# AC7 -- BG3's propagation check. Runs AFTER the array and the merge, and is
# post-hoc by construction: the instrument correlating null maxima against a
# draw's realised prevalence takes the STORED MAXIMA as an argument, and for
# this stratum those do not exist until the null has run. It was post-hoc for
# the pooled null too.
#
# NO GPU and no recompute -- it reads the merged null only. The surviving
# counts it needs were stored by the run itself (brief B section 4b item 2),
# so this measures what the null DID rather than what a re-derivation would do.

set -euo pipefail

PROJECT=$PROJECT

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

python -m src.analysis.bg3_propagation \
    --null "$PROJECT/analysis/null_experimental_bg.h5" \
    --out "$PROJECT/analysis/bg3_propagation_ac7.json"
