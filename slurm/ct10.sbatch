#!/bin/bash
#SBATCH --job-name=ct10
#SBATCH --output=logs/%x_%j_%a.out
# Amendment CT §CT10: activation matching on rows disjoint from the alignment
# score. Resources are passed on the sbatch command line per step:
#
#   split            CPU, `day`
#   ap HALF          gpu rtx_5000_ada  (feature_ap of record's arguments, on a half,
#                                      AP arithmetic on CUDA: the null's unmasked
#                                      path computes its AP on the GPU, and pre-flight
#                                      refuses a pair whose ap_arith_device_class
#                                      differs -- gotcha 36; `auto` would send a
#                                      half's ~256M non-zeros to CPU)
#   null HALF        gpu rtx_5000_ada  (null_pilot of record's arguments, on a half,
#                                      concept axis from that half's AP)
#   gram             gpu rtx_5000_ada array 0-19 (task = anchor), half-A rows of
#                    §CT4's stored codes
#   reduce           CPU; theta*_A against 0.495513 (dec/B's rate)
#   gtilde HALF      CPU; the pooled stratum on the half: own axis values, the
#                    >= 5-protein gate (§CT10's reading of "the CT5.1 rule")
#   fit              CPU; dec/B, act-A/B, act-A/A with jackknives
#
# HALF is a or b. Outputs: $PROJECT/analysis/ct/ (small), $SCRATCH/ct10/ (raw).
set -euo pipefail
STEP="${1:?split|ap|null|gram|reduce|gtilde|fit}"
HALF="${2:-}"
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
[[ -z "$(git status --porcelain)" ]] || { echo "checkout dirty" >&2; exit 1; }
PROJECT=$PROJECT
SCRATCH=$SCRATCH
A="$PROJECT/analysis"
S10="$SCRATCH/ct10"
mkdir -p "$A/ct" "$S10/rows"
LABELS="$A/residue_labels_subtyped_untruncated_pgate5_f67485d.h5"
CACHE="$SCRATCH/cache/eval.h5"
HALVES="$A/ct/ct10_halves_ct.h5"
CKPTS=(); for s in $(seq 0 19); do CKPTS+=("$PROJECT/sae_ckpts/seed_$s"); done
need_half() { [[ "$HALF" == "a" || "$HALF" == "b" ]] || { echo "HALF must be a or b" >&2; exit 2; }; }

case "$STEP" in
  split)
    python -m src.analysis.ct10_halves --labels "$LABELS" --eval-cache "$CACHE" \
        --clusters-tsv "$PROJECT/archive/sprot_clu30_cluster.tsv" --out "$HALVES"
    ;;
  ap)
    need_half
    python -m src.analysis.feature_ap --config configs/mvp.yaml --ckpts "${CKPTS[@]}" \
        --eval-cache "$CACHE" --labels "$LABELS" --strata pooled \
        --scored-row-set "ct10_half_$HALF" --row-set-file "$HALVES" \
        --negatives all_rows --ap-device cuda \
        --out "$A/ct/feature_ap_ct10_half_${HALF}_ct.h5"
    ;;
  null)
    need_half
    python -m src.analysis.null_pilot --config configs/mvp.yaml --ckpts "${CKPTS[@]}" \
        --eval-cache "$CACHE" --labels "$LABELS" \
        --scored-row-set "ct10_half_$HALF" --row-set-file "$HALVES" \
        --null-type both --n-length-bins 20 --n-shifts 20 --null-seed 0 \
        --cache-dir "$SCRATCH/row_index_cache" --exclude-pooled-parents \
        --concepts-from "$A/ct/feature_ap_ct10_half_${HALF}_ct.h5" \
        --out "$A/ct/null_ct10_half_${HALF}_ct.h5" \
        --raw-out "$S10/null_gain_raw_ct10_half_${HALF}_ct.h5"
    ;;
  gram)
    T=$SLURM_ARRAY_TASK_ID
    CODES=(); for s in $(seq 0 39); do CODES+=("$SCRATCH/ct_actcorr/codes/codes_seed_$s.h5"); done
    [[ -e "$S10/rows/row_anchor_$T.h5" ]] && { echo "anchor $T exists, skipped"; exit 0; }
    python -m src.analysis.activation_correlation gram --codes "${CODES[@]}" --anchor "$T" \
        --labels "$LABELS" --row-subset ct10_half_a --row-subset-file "$HALVES" \
        --out "$S10/rows/row_anchor_$T.h5"
    ;;
  gram_seq)
    # The anchors in order in ONE gpu_devel job (one job per user there):
    # resumable, an existing row skipped, stopping before the 6 h limit.
    CODES=(); for s in $(seq 0 39); do CODES+=("$SCRATCH/ct_actcorr/codes/codes_seed_$s.h5"); done
    T0=$(date +%s)
    for T in $(seq 0 19); do
        [[ -e "$S10/rows/row_anchor_$T.h5" ]] && { echo "anchor $T exists, skipped"; continue; }
        (( $(date +%s) - T0 > 18000 )) && { echo "stopping before the time limit"; break; }
        python -m src.analysis.activation_correlation gram --codes "${CODES[@]}" --anchor "$T" \
            --labels "$LABELS" --row-subset ct10_half_a --row-subset-file "$HALVES" \
            --out "$S10/rows/row_anchor_$T.h5"
    done
    ;;
  reduce)
    R=(); for s in $(seq 0 19); do R+=("$S10/rows/row_anchor_$s.h5"); done
    python -m src.analysis.activation_correlation reduce --rows "${R[@]}" \
        --out "$A/ct/ct10_actA_ct.json" --out-counts "$A/ct/ct10_actA_counts_ct.h5"
    ;;
  gtilde)
    need_half
    python -m src.analysis.standardize_gain \
        --ap "$A/ct/feature_ap_ct10_half_${HALF}_ct.h5" \
        --null-raw "$S10/null_gain_raw_ct10_half_${HALF}_ct.h5" \
        --null-reductions "$A/ct/null_ct10_half_${HALF}_ct.h5" \
        --labels "$LABELS" --arm b --inner-mode ratio --stratum pooled \
        --axis-values stratum --admission-strata pooled --row-set-file "$HALVES" \
        --eval-cache "$CACHE" --clusters-tsv "$PROJECT/archive/sprot_clu30_cluster.tsv" \
        --config configs/mvp.yaml --out "$A/ct/gtilde_ct10_half_${HALF}_ct.h5"
    ;;
  fit)
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python -m src.analysis.ct_robustness ct10-fit \
        --sim "$A/ct/jackknife_match_ct.h5" \
        --rf-counts-all-theta "$A/rf_counts_all_theta_n39.h5" \
        --m1-json "$A/rf_marginal_m1_n39.json" \
        --gtilde "$A/gtilde_compliant_b_ratio.h5" \
        --null-reductions "$A/null_pilot_compliant.h5" \
        --d-fs "$A/d_fs.h5" --degenerate-flag "$A/degenerate_argmax_flag.h5" \
        --reference-join "$A/design3_join_aw_nofeat.json" \
        --act-half-counts "$A/ct/ct10_actA_counts_ct.h5" --act-half-json "$A/ct/ct10_actA_ct.json" \
        --gtilde-half-a "$A/ct/gtilde_ct10_half_a_ct.h5" --gtilde-half-b "$A/ct/gtilde_ct10_half_b_ct.h5" \
        --workers "${SLURM_CPUS_PER_TASK:-8}" --out "$A/ct/ct10_disjoint_rows_ct.json"
    ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
[[ -z "$(git status --porcelain)" ]] || { echo "checkout dirty on exit" >&2; exit 1; }
