#!/bin/bash
#SBATCH --job-name=sae_w0_resolution
#SBATCH --partition=day
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# THE w0 RESOLUTION TEST -- PREREGISTRATION amendment AO sections 1-3,
# The author's ratified decision AM6.1.
#
# NOT a firewall crossing: it reads the r_f marginal's mixture fit only.
#
# AO5: this job writes to NEW paths. rf_mixture_n39.json and rf_mixture_n19.json
# are canary targets hashed WHOLE, and the module refuses --out equal to its
# input. Nothing baselined is touched.
#
# THE OUTPUT PATH MOVED FROM `_ao` TO `_aq` (amendment AQ4), and deliberately
# rather than as tidying. This run adds the purity band at the fine-grid
# interval, so its output is a SUPERSET of the AO run's -- but overwriting
# `w0_resolution_ao.json` would destroy the artifact AO10-AO12's reported
# numbers were read from, and "the numbers are deterministic so it is safe" is an
# argument, not a check. Both files stay, and the AO one remains the record of
# what AO reported.
#
# One process per replicate, so BLAS must not also thread -- the arrays are
# 201x40 and thread startup would dominate.

set -euo pipefail
PROJECT=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -m src.analysis.w0_resolution \
    --mixture "$PROJECT/analysis/rf_mixture_n39.json" \
    --mixture-n19 "$PROJECT/analysis/rf_mixture_n19.json" \
    --theta 0.70 \
    --n-rep 200 \
    --base-seed 0 \
    --workers "${SLURM_CPUS_PER_TASK:-20}" \
    --out "$PROJECT/analysis/w0_resolution_aq.json"
