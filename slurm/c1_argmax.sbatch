#!/bin/bash
#SBATCH --job-name=sae_c1_argmax
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# AL2 RESTATED -- amendment AO section 7, the author's ratified decision AM6.2(b).
#
# *** THIS JOB DELIBERATELY CROSSES THE X7 FIREWALL. ***
# Same authorised crossing as m4_confound --c1-only: C1 is a SPECIFICATION
# question. Scoped to correlations and conditional means; no model fit, no
# regression; nothing about theta, the endpoint cuts or the primary regression
# may be revised from what it shows.
#
# AO5: writes to a NEW path. m4_c1_all_seeds.json is a canary target.
#
# BOTH r_f arms are supplied deliberately: the incumbent 0.00709 was measured at
# n_compare = 19 (the like-for-like comparison P6 is scored on) and n_compare =
# 39 is the arm of record.

set -euo pipefail
SCRATCH=$SCRATCH
PROJECT=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

# 2026-09-16: `--mu-smooth` below names `$SCRATCH/analysis/smooth_mu_b_ident.h5`.
# That file lived on scratch, was consumed by this job to produce
# `$PROJECT/analysis/c1_argmax_ao.json` (2026-07-31, job 20751549), and has
# since been purged: `find $PROJECT $SCRATCH -name 'smooth_mu_b_ident*'` returns
# nothing (docs/DATAFLOW.md §4, cluster read 2026-09-16), and no producer for
# that path exists in the tree. THIS SCRIPT CANNOT RUN AS WRITTEN; its output
# `c1_argmax_ao.json` stands as the artifact of record. The comment sits above
# the command rather than beside the argument because a comment line inside a
# backslash-continued command would split it.
python -m src.analysis.c1_argmax \
    --gtilde "$PROJECT/analysis/gtilde_compliant_b_ratio.h5" \
    --gtilde-smooth "$SCRATCH/analysis/gtilde_compliant_b_ratio_smooth.h5" \
    --mu-smooth "$SCRATCH/analysis/smooth_mu_b_ident.h5" \
    --null-reductions "$PROJECT/analysis/null_pilot_compliant.h5" \
    --rf-counts "$PROJECT/analysis/rf_counts_theta070_n19.h5" \
                "$PROJECT/analysis/rf_counts_theta070_n39.h5" \
    --out "$PROJECT/analysis/c1_argmax_ao.json"
