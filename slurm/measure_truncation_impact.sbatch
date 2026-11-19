#!/bin/bash
#SBATCH --job-name=measure_truncation
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

# P4 -- what truncation costs the sub-typed concept library. Parse-side,
# CPU-only, independent of the AP restructure (P1). Dispatched in parallel
# per instruction: this blocks D8's straddling-instance rule, so it should
# not wait behind P1.

set -euo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

python -m src.annotations.measure_truncation_impact \
    --config configs/mvp.yaml \
    --dat   "$SCRATCH/swissprot/uniprot_sprot.dat.gz" \
    --cache "$SCRATCH/cache/eval.h5" \
    --out   "$PROJECT/analysis/truncation_impact.json"
