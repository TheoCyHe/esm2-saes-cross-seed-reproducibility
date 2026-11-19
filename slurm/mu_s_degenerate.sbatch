#!/bin/bash
#SBATCH --job-name=sae_mu_s_degen
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

# M3 -- does excluding the 22 degenerate concepts change the per-stratum
# `mu_s`? `mu_s` is now the DENOMINATOR of the regression predictor
# (amendments U1 + V1), and R10 leaves the 22 flagged rather than gated, so
# they are still inside the strata that define it.
#
# NOT the zero-division question (U1 decision 4 settled that: min mu_s
# 3.358e-07 over 240 strata). This asks whether they BIAS the denominator.
#
# Predictions D1-D5 are committed in the module docstring BEFORE this runs.
# D4 explicitly declines to predict a direction -- the brief's instruction was
# "do not reason about the direction first; compute both and compare".

set -euo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

python -m src.analysis.mu_s_degenerate \
    --ap "$PROJECT/analysis/feature_ap_main_compliant.h5" \
    --null-raw "$PROJECT/analysis/null_gain_raw_compliant.h5" \
    --null-reductions "$PROJECT/analysis/null_pilot_compliant.h5" \
    --labels "$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5" \
    --arm b --seeds 0 7 \
    --out "$PROJECT/analysis/mu_s_degenerate_m3.json"
