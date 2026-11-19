#!/bin/bash
#SBATCH --job-name=sae_m4_confound
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# AC section 2 -- THE CONFOUND CHECK.
#
# *** THIS JOB DELIBERATELY CROSSES THE X7 FIREWALL. ***
# It joins r_f to alignment because the question is a SPECIFICATION question
# and cannot be asked otherwise. The crossing is SCOPED (two quantities),
# LABELLED (the output records firewall_crossing: DELIBERATE), and
# NON-AMBIENT (its own script, its own job). Nothing about theta, the endpoint
# cuts, or the primary regression may be revised from what it shows.

set -euo pipefail
SCRATCH=$SCRATCH
PROJECT=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

python -m src.analysis.m4_confound \
    --ap "$PROJECT/analysis/feature_ap_main_compliant.h5" \
    --null-raw "$PROJECT/analysis/null_gain_raw_compliant.h5" \
    --null-reductions "$PROJECT/analysis/null_pilot_compliant.h5" \
    --labels "$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5" \
    --row-index-cache "$SCRATCH/row_index_cache" \
    --ckpts "$PROJECT/sae_ckpts" \
    --eval-cache "$SCRATCH/cache/eval.h5" \
    --rf-counts "$PROJECT/analysis/rf_counts_theta070.h5" \
    --arm b --seeds 0 7 \
    --out "$PROJECT/analysis/m4_confound.json"
