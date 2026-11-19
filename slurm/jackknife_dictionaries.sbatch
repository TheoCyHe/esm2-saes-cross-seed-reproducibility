#!/bin/bash
#SBATCH --job-name=ct_jackknife
#SBATCH --output=logs/%x_%j.out
# Amendment CT, §CT3 / §CT11.1: the leave-one-dictionary-out jackknife.
#
#   sbatch --partition=gpu --gpus=rtx_5000_ada:1 --cpus-per-task=8 --mem=64G \
#          --time=01:00:00 slurm/jackknife_dictionaries.sbatch match
#   sbatch --partition=day --cpus-per-task=16 --mem=96G --time=12:00:00 \
#          slurm/jackknife_dictionaries.sbatch fit
#
# `match` re-runs rf_marginal's MNN on the 590 anchor-involving pairs on the
# record's device class (rtx_5000_ada: job 20810740's sacct record; the stored
# counts carry no device attr) and GATES against rf_counts_all_theta_n39.h5 --
# EXACT / NEAR / DIVERGENT, and DIVERGENT exits non-zero. `fit` refuses unless
# the match file's gate is EXACT or NEAR, refits the full ensemble as a
# control against design3_join_aw_nofeat.json, then runs the 40 x 13
# leave-one-out fits on CPU.
set -euo pipefail
MODE="${1:?match or fit}"
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
[[ -z "$(git status --porcelain)" ]] || { echo "checkout dirty" >&2; exit 1; }

# Literal, as every launcher here does: a batch job does not inherit an
# interactive $PROJECT (gotcha 32a; job 30204346 died on it, no compute).
PROJECT=$PROJECT
A="$PROJECT/analysis"
OUTDIR="$A/ct"
mkdir -p "$OUTDIR"
SIM="$OUTDIR/jackknife_match_ct.h5"
ALLTHETA="$A/rf_counts_all_theta_n39.h5"

case "$MODE" in
  match)
    ANCHORS=("$PROJECT"/sae_ckpts/seed_*)
    EXT=("$PROJECT"/sae_ckpts_ext/seed_*)
    [[ ${#ANCHORS[@]} -eq 20 && ${#EXT[@]} -eq 20 ]] || {
        echo "expected 20 + 20 checkpoints, found ${#ANCHORS[@]} + ${#EXT[@]}" >&2; exit 1; }
    python -m src.analysis.jackknife_dictionaries match \
        --ckpts "${ANCHORS[@]}" --compare-ckpts "${EXT[@]}" \
        --expect-total-ckpts 40 \
        --rf-counts-all-theta "$ALLTHETA" \
        --record-device-class rtx_5000_ada \
        --record-device-source "sacct job 20810740 (sae_rf_marginal, gres/gpu:rtx_5000_ada=1, node <node>) and its log line 'device=cuda'; the counts file carries no device attr" \
        --out "$SIM"
    ;;
  fit)
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python -m src.analysis.jackknife_dictionaries fit \
        --sim "$SIM" \
        --rf-counts-all-theta "$ALLTHETA" \
        --m1-json "$A/rf_marginal_m1_n39.json" \
        --gtilde "$A/gtilde_compliant_b_ratio.h5" \
        --null-reductions "$A/null_pilot_compliant.h5" \
        --d-fs "$A/d_fs.h5" \
        --degenerate-flag "$A/degenerate_argmax_flag.h5" \
        --reference-join "$A/design3_join_aw_nofeat.json" \
        --workers "${SLURM_CPUS_PER_TASK:-8}" \
        --out "$OUTDIR/jackknife_ct.json"
    ;;
  *) echo "unknown mode $MODE" >&2; exit 1 ;;
esac
[[ -z "$(git status --porcelain)" ]] || { echo "checkout dirty on exit" >&2; exit 1; }
