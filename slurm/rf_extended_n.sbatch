#!/bin/bash
#SBATCH --job-name=sae_rf_extended_n
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

# THE EXTENDED-N COMPARISON. PREREGISTRATION amendments AG and AK.
#
# TWO ACCEPTANCE CRITERIA, DELIBERATELY NOT COLLAPSED (AK1):
#   * the N = 19 re-run must reproduce analysis/repro/final2/ BIT-IDENTICALLY on
#     every key the baseline carries; the keys the anchor/comparison split ADDS
#     are allowlisted by name in the module and an undeclared addition FAILS.
#   * the N = 39 outputs are scored against AK's pre-committed predictions.
#
# The job EXITS NON-ZERO on a canary movement or on a failed correctness
# assertion, and exits ZERO on a merely MISSED prediction -- a miss is a
# legitimate outcome and the ledger exists because predictions keep being made.

set -euo pipefail
PROJECT=$PROJECT
A="$PROJECT/analysis"
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

python -m src.analysis.rf_extended_n \
    --baseline-m1      "$A/repro/final2/rf_marginal_m1.json" \
    --candidate-m1-n19 "$A/rf_marginal_m1_n19.json" \
    --baseline-counts  "$A/repro/final2/rf_counts_theta070.h5" \
    --candidate-counts-n19 "$A/rf_counts_theta070_n19.h5" \
    --m1-n39           "$A/rf_marginal_m1_n39.json" \
    --mixture-n19      "$A/rf_mixture_n19.json" \
    --mixture-n39      "$A/rf_mixture_n39.json" \
    --out              "$A/rf_extended_n_comparison.json"
