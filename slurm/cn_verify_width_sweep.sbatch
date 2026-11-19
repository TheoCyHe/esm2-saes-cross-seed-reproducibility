#!/bin/bash
#SBATCH --job-name=cn_verify
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

# Verify the CN width-sweep checkpoints. Reports a diff, not a verdict.
# CPU only -- this loads tensors to check shapes; it does not evaluate.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

python -m src.sae.verify_width_sweep --load
