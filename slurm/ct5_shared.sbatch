#!/bin/bash
#SBATCH --job-name=ct5_shared
#SBATCH --partition=day
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=23:55:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x_%A_%a.out
# Amendment CT, §CT5 / §CT11.7-R: A1's experimental-versus-propagated contrast on
# the SHARED protein set (proteins with >= 1 experimental label). One array
# task per anchor seed, CPU (the AP and its null on one device class, gotcha 36;
# the BG experimental null and its `_cpu` AP are the precedent).
#
#   sbatch slurm/ct5_shared.sbatch ap                 # both strata, one pass
#   sbatch --dependency=aftercorr:<ap> slurm/ct5_shared.sbatch null experimental
#   sbatch --dependency=aftercorr:<ap> slurm/ct5_shared.sbatch null propagated
#
# Each null task restricts its concepts to the SAME seed's AP shard's axis for
# that stratum (--concepts-from): the AP keeps the strata's common set (A1).
#
# Every flag below is the precedent launchers' (feature_ap_experimental_array_cpu,
# null_experimental_array_cpu) with ONLY the population changed to
# shared_experimental and the AP given both strata.
set -euo pipefail
STEP="${1:?ap|null}"
STRATUM="${2:-}"
SCRATCH=$SCRATCH
PROJECT=$PROJECT
LABELS="$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5"
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
[[ -z "$(git status --porcelain)" ]] || { echo "checkout dirty" >&2; exit 1; }
SEED="$SLURM_ARRAY_TASK_ID"
CKPT="$PROJECT/sae_ckpts/seed_$SEED"
[[ -f "$CKPT/model.pt" ]] || { echo "FATAL: no checkpoint at $CKPT" >&2; exit 1; }
[[ -f "$LABELS" ]] || { echo "FATAL: missing label store $LABELS" >&2; exit 1; }
echo "[ct5] git $(git rev-parse HEAD) step=$STEP stratum=$STRATUM seed=$SEED"

case "$STEP" in
  ap)
    D="$SCRATCH/ct5/ap_shards"; mkdir -p "$D"
    python -m src.analysis.feature_ap \
        --config configs/mvp.yaml \
        --ckpts "$CKPT" \
        --eval-cache "$SCRATCH/cache/eval.h5" \
        --labels "$LABELS" \
        --strata experimental propagated \
        --scored-row-set shared_experimental \
        --negatives unannotated \
        --ap-device auto \
        --out "$D/shard_seed_${SEED}.h5"
    ;;
  null)
    [[ "$STRATUM" == "experimental" || "$STRATUM" == "propagated" ]] || {
        echo "FATAL: stratum must be experimental or propagated" >&2; exit 1; }
    D="$SCRATCH/ct5/null_${STRATUM}_shards"; mkdir -p "$D"
    python -m src.analysis.null_pilot \
        --config configs/mvp.yaml \
        --ckpts "$CKPT" \
        --eval-cache "$SCRATCH/cache/eval.h5" \
        --labels "$LABELS" \
        --scored-row-set shared_experimental \
        --stratum "$STRATUM" \
        --negatives unannotated \
        --null-type both \
        --n-length-bins 20 \
        --n-shifts 20 \
        --null-seed 0 \
        --cache-dir "$SCRATCH/row_index_cache" \
        --exclude-pooled-parents \
        --concepts-from "$SCRATCH/ct5/ap_shards/shard_seed_${SEED}.h5" \
        --ap-device auto \
        --out "$D/red_seed_${SEED}.h5" \
        --raw-out "$D/raw_seed_${SEED}.h5"
    ;;
  merge_ap)
    # One job (submit with --array=0). The seed-order reference is the pooled
    # label-store AP; the population legitimately differs (§CT11.7-R) and the
    # flag says so; every other identity key is still compared.
    [[ "$STRATUM" == "experimental" || "$STRATUM" == "propagated" ]] || exit 1
    mkdir -p "$PROJECT/analysis/ct"
    python -m src.analysis.merge_ap_shards \
        --shards "$SCRATCH"/ct5/ap_shards/shard_seed_*.h5 \
        --reference "$PROJECT/analysis/feature_ap_main_compliant.h5" \
        --labels "$LABELS" \
        --stratum "$STRATUM" \
        --population-may-differ \
        --out "$PROJECT/analysis/ct/feature_ap_shared_${STRATUM}_ct.h5" \
        --report "$PROJECT/analysis/ct/feature_ap_shared_${STRATUM}_ct_merge.json"
    ;;
  merge_null)
    # One job (submit with --array=0), after merge_ap for the same stratum.
    [[ "$STRATUM" == "experimental" || "$STRATUM" == "propagated" ]] || exit 1
    D="$SCRATCH/ct5/null_${STRATUM}_shards"
    RED=(); RAW=()
    for s in $(seq 0 19); do RED+=("$D/red_seed_$s.h5"); RAW+=("$D/raw_seed_$s.h5"); done
    python -m src.analysis.merge_null_shards \
        --shard-reductions "${RED[@]}" \
        --shard-raw "${RAW[@]}" \
        --reference "$PROJECT/analysis/ct/feature_ap_shared_${STRATUM}_ct.h5" \
        --out "$PROJECT/analysis/ct/null_shared_${STRATUM}_ct.h5" \
        --raw-out "$SCRATCH/ct5/null_gain_raw_shared_${STRATUM}_ct.h5" \
        --report "$PROJECT/analysis/ct/null_shared_${STRATUM}_ct_merge.json"
    ;;
  gtilde)
    # §CT5.1: the stratum's M, both strata's admission gate applied, the pooled
    # record's parameters (arm b, ratio, O2 axes). One job (--array=0).
    [[ "$STRATUM" == "experimental" || "$STRATUM" == "propagated" ]] || exit 1
    python -m src.analysis.standardize_gain \
        --ap "$PROJECT/analysis/ct/feature_ap_shared_${STRATUM}_ct.h5" \
        --null-raw "$SCRATCH/ct5/null_gain_raw_shared_${STRATUM}_ct.h5" \
        --null-reductions "$PROJECT/analysis/ct/null_shared_${STRATUM}_ct.h5" \
        --labels "$LABELS" \
        --arm b --inner-mode ratio \
        --stratum "$STRATUM" \
        --axis-values stratum \
        --admission-strata experimental propagated \
        --eval-cache "$SCRATCH/cache/eval.h5" \
        --clusters-tsv "$PROJECT/archive/sprot_clu30_cluster.tsv" \
        --config configs/mvp.yaml \
        --out "$PROJECT/analysis/ct/gtilde_shared_${STRATUM}_ct.h5"
    ;;
  slope)
    # §CT5.1: the design-3 slope at 0.70 with log M replaced per stratum, beside
    # the full-score primary refit. One job (--array=0), after both gtilde jobs.
    A="$PROJECT/analysis"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python -m src.analysis.stratum_slope \
        --sim "$A/ct/jackknife_match_ct.h5" \
        --rf-counts-all-theta "$A/rf_counts_all_theta_n39.h5" \
        --m1-json "$A/rf_marginal_m1_n39.json" \
        --gtilde "$A/gtilde_compliant_b_ratio.h5" \
        --null-reductions "$A/null_pilot_compliant.h5" \
        --d-fs "$A/d_fs.h5" --degenerate-flag "$A/degenerate_argmax_flag.h5" \
        --config configs/mvp.yaml \
        --theta 0.70 \
        --strata experimental propagated \
        --gtilde-strata "$A/ct/gtilde_shared_experimental_ct.h5" "$A/ct/gtilde_shared_propagated_ct.h5" \
        --out "$A/ct/stratum_slope_ct.json"
    ;;
  *) echo "unknown step $STEP" >&2; exit 1 ;;
esac
[[ -z "$(git status --porcelain)" ]] || { echo "checkout dirty on exit" >&2; exit 1; }
