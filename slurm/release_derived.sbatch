#!/bin/bash
#SBATCH --job-name=release_derived
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Amendment CT, §CT18 (b): write release/derived/ to $PROJECT/release/derived/
# from the artifacts of record. CPU only: nothing here gathers or matches; the
# MNN outcomes are §CT3's persisted similarities, gated EXACT against the
# stored counts by the record's own loader. The module refuses to write until
# its five gates pass (see src/analysis/release_derived.py).
#
#   sbatch slurm/release_derived.sbatch
set -euo pipefail
PROJECT=$PROJECT
A="$PROJECT/analysis"
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
if [[ -n "$(git status --porcelain)" ]]; then
    echo "REFUSING: the checkout is dirty before the run" >&2; exit 1
fi
python -m src.analysis.release_derived \
    --sim "$A/ct/jackknife_match_ct.h5" \
    --rf-counts-all-theta "$A/rf_counts_all_theta_n39.h5" \
    --rf-counts-n19 "$A/rf_counts_theta070_n19.h5" \
    --m1-json "$A/rf_marginal_m1_n39.json" \
    --gtilde "$A/gtilde_compliant_b_ratio.h5" \
    --null-reductions "$A/null_pilot_compliant.h5" \
    --d-fs "$A/d_fs.h5" \
    --degenerate-flag "$A/degenerate_argmax_flag.h5" \
    --ap "$A/feature_ap_main_compliant.h5" \
    --carrier-reference "$A/carrier_concentration_bb.json" \
    --fvu "$A/fvu_eval_cf.json" \
    --out-dir "$PROJECT/release/derived"
if [[ -n "$(git status --porcelain)" ]]; then
    echo "the checkout is dirty on exit" >&2; exit 1
fi
