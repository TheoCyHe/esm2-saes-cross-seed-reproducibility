#!/bin/bash
#SBATCH --job-name=ct_actcorr
#SBATCH --output=logs/%x_%A_%a.out
# Amendment CT, §CT4 / §CT11.6: the activation-correlation robustness arm.
#
#   encode  array 0-39 (task = seed), gpu rtx_5000_ada
#   gram    array 0-19 (task = anchor seed), gpu rtx_5000_ada
#   reduce  one CPU task
#   fit     one CPU task
#
# Resources are passed on the sbatch command line per step (see §CT11.6's
# bracket). Codes and per-anchor rows go to scratch; the reduced artifacts to
# $PROJECT/analysis/ct/.
set -euo pipefail
STEP="${1:?encode|gram|reduce|fit}"
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
[[ -z "$(git status --porcelain)" ]] || { echo "checkout dirty" >&2; exit 1; }
PROJECT=$PROJECT
SCRATCH=$SCRATCH
A="$PROJECT/analysis"
CODES="$SCRATCH/ct_actcorr/codes"
ROWS="$SCRATCH/ct_actcorr/rows"
mkdir -p "$CODES" "$ROWS" "$A/ct"
LABELS="$A/residue_labels_subtyped_untruncated_pgate5_f67485d.h5"
CACHE="$SCRATCH/cache/eval.h5"

case "$STEP" in
  encode)
    S=$SLURM_ARRAY_TASK_ID
    if (( S < 20 )); then CK="$PROJECT/sae_ckpts/seed_$S"; else CK="$PROJECT/sae_ckpts_ext/seed_$S"; fi
    python -m src.analysis.activation_correlation encode --ckpt "$CK" \
        --cache "$CACHE" --labels "$LABELS" --out "$CODES/codes_seed_$S.h5"
    ;;
  encode_all)
    # The same encode, all 40 seeds in ONE job, for gpu_devel (which refuses
    # arrays and runs one job per user): the measured unit is ~2-3 min per
    # dictionary (§CT11.6-R), so 40 fit inside its 6 h. A seed whose code
    # file already exists is skipped, so a rerun resumes.
    for S in $(seq 0 39); do
        [[ -e "$CODES/codes_seed_$S.h5" ]] && { echo "seed $S exists, skipped"; continue; }
        if (( S < 20 )); then CK="$PROJECT/sae_ckpts/seed_$S"; else CK="$PROJECT/sae_ckpts_ext/seed_$S"; fi
        python -m src.analysis.activation_correlation encode --ckpt "$CK" \
            --cache "$CACHE" --labels "$LABELS" --out "$CODES/codes_seed_$S.h5"
    done
    ;;
  gram)
    S=$SLURM_ARRAY_TASK_ID
    ALL=(); for s in $(seq 0 39); do if [[ -e "$CODES/codes_seed_$s.h5" ]]; then ALL+=("$CODES/codes_seed_$s.h5"); fi; done
    [[ ${#ALL[@]} -eq 40 ]] || { echo "expected 40 code files, found ${#ALL[@]}" >&2; exit 1; }
    # An anchor the sequential gpu_devel runner already finished is skipped.
    [[ -e "$ROWS/row_anchor_$S.h5" ]] && { echo "anchor $S exists, skipped"; exit 0; }
    python -m src.analysis.activation_correlation gram --codes "${ALL[@]}" \
        --anchor "$S" --out "$ROWS/row_anchor_$S.h5"
    ;;
  gram_seq)
    # The anchors in order in ONE job, for gpu_devel while the gpu array waits
    # on a reservation: resumable (an existing row is skipped) and stops
    # cleanly before the partition's 6 h limit (each anchor ~1 h, §CT11.6-R).
    ALL=(); for s in $(seq 0 39); do if [[ -e "$CODES/codes_seed_$s.h5" ]]; then ALL+=("$CODES/codes_seed_$s.h5"); fi; done
    [[ ${#ALL[@]} -eq 40 ]] || { echo "expected 40 code files, found ${#ALL[@]}" >&2; exit 1; }
    T0=$(date +%s)
    for S in $(seq 0 19); do
        [[ -e "$ROWS/row_anchor_$S.h5" ]] && { echo "anchor $S exists, skipped"; continue; }
        (( $(date +%s) - T0 > 17400 )) && { echo "stopping before the time limit"; break; }
        python -m src.analysis.activation_correlation gram --codes "${ALL[@]}" \
            --anchor "$S" --out "$ROWS/row_anchor_$S.h5"
    done
    ;;
  reduce)
    R=(); for s in $(seq 0 19); do if [[ -e "$ROWS/row_anchor_$s.h5" ]]; then R+=("$ROWS/row_anchor_$s.h5"); fi; done
    [[ ${#R[@]} -eq 20 ]] || { echo "expected 20 anchor rows, found ${#R[@]}" >&2; exit 1; }
    python -m src.analysis.activation_correlation reduce --rows "${R[@]}" \
        --out "$A/ct/activation_correlation_ct.json" \
        --out-counts "$A/ct/activation_correlation_counts_ct.h5"
    ;;
  fit)
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python -m src.analysis.activation_correlation fit \
        --act-counts "$A/ct/activation_correlation_counts_ct.h5" \
        --sim "$A/ct/jackknife_match_ct.h5" \
        --rf-counts-all-theta "$A/rf_counts_all_theta_n39.h5" \
        --m1-json "$A/rf_marginal_m1_n39.json" \
        --gtilde "$A/gtilde_compliant_b_ratio.h5" \
        --null-reductions "$A/null_pilot_compliant.h5" \
        --d-fs "$A/d_fs.h5" --degenerate-flag "$A/degenerate_argmax_flag.h5" \
        --out "$A/ct/activation_correlation_fit_ct.json"
    ;;
  jackknife)
    # §CT4: the jackknife at theta*, run because §CT3's machinery takes well
    # under a day. One BLAS thread per worker, as §CT3's fit.
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python -m src.analysis.activation_correlation jackknife \
        --act-counts "$A/ct/activation_correlation_counts_ct.h5" \
        --sim "$A/ct/jackknife_match_ct.h5" \
        --rf-counts-all-theta "$A/rf_counts_all_theta_n39.h5" \
        --m1-json "$A/rf_marginal_m1_n39.json" \
        --gtilde "$A/gtilde_compliant_b_ratio.h5" \
        --null-reductions "$A/null_pilot_compliant.h5" \
        --d-fs "$A/d_fs.h5" --degenerate-flag "$A/degenerate_argmax_flag.h5" \
        --fit-json "$A/ct/activation_correlation_fit_ct.json" \
        --workers "${SLURM_CPUS_PER_TASK:-8}" \
        --out "$A/ct/activation_correlation_jackknife_ct.json"
    ;;
  pairs_seq)
    # §CT12.4(a): one gpu_devel job over the 20 anchors, resumable.
    ALL=(); for s in $(seq 0 39); do if [[ -e "$CODES/codes_seed_$s.h5" ]]; then ALL+=("$CODES/codes_seed_$s.h5"); fi; done
    CK=(); for s in $(seq 0 19); do CK+=("$PROJECT/sae_ckpts/seed_$s"); done
    for s in $(seq 20 39); do CK+=("$PROJECT/sae_ckpts_ext/seed_$s"); done
    OUTD="$A/ct/pairs"; mkdir -p "$OUTD"
    T0=$(date +%s)
    for S in $(seq 0 19); do
        if [[ -e "$OUTD/pairs_anchor_$S.json" ]]; then echo "anchor $S exists, skipped"; continue; fi
        if (( $(date +%s) - T0 > 18000 )); then echo "stopping before the time limit"; break; fi
        python -m src.analysis.activation_correlation pairs --codes "${ALL[@]}" \
            --ckpts "${CK[@]}" --act-counts "$A/ct/activation_correlation_counts_ct.h5" \
            --dec-sim "$A/ct/jackknife_match_ct.h5" --anchor "$S" \
            --out "$OUTD/pairs_anchor_$S.json"
    done
    ;;
  pairs)
    # §CT12.4(a), one anchor per array task (gpu), beside pairs_seq; an anchor
    # either runner already wrote is skipped.
    S=$SLURM_ARRAY_TASK_ID
    ALL=(); for s in $(seq 0 39); do if [[ -e "$CODES/codes_seed_$s.h5" ]]; then ALL+=("$CODES/codes_seed_$s.h5"); fi; done
    CK=(); for s in $(seq 0 19); do CK+=("$PROJECT/sae_ckpts/seed_$s"); done
    for s in $(seq 20 39); do CK+=("$PROJECT/sae_ckpts_ext/seed_$s"); done
    OUTD="$A/ct/pairs"; mkdir -p "$OUTD"
    if [[ -e "$OUTD/pairs_anchor_$S.json" ]]; then echo "anchor $S exists, skipped"; exit 0; fi
    python -m src.analysis.activation_correlation pairs --codes "${ALL[@]}" \
        --ckpts "${CK[@]}" --act-counts "$A/ct/activation_correlation_counts_ct.h5" \
        --dec-sim "$A/ct/jackknife_match_ct.h5" --anchor "$S" \
        --out "$OUTD/pairs_anchor_$S.json"
    ;;
  curve)
    # §CT12.4(b): both slopes at every grid theta.
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python -m src.analysis.activation_correlation curve \
        --act-counts "$A/ct/activation_correlation_counts_ct.h5" \
        --sim "$A/ct/jackknife_match_ct.h5" \
        --rf-counts-all-theta "$A/rf_counts_all_theta_n39.h5" \
        --m1-json "$A/rf_marginal_m1_n39.json" \
        --gtilde "$A/gtilde_compliant_b_ratio.h5" \
        --null-reductions "$A/null_pilot_compliant.h5" \
        --d-fs "$A/d_fs.h5" --degenerate-flag "$A/degenerate_argmax_flag.h5" \
        --out "$A/ct/activation_vs_decoder_slope_curve_ct.json"
    ;;
  classes)
    # §CT4-T's descriptive item: log M by cell class at theta*. One CPU task.
    PAIRS=(); for s in $(seq 0 19); do PAIRS+=("$A/ct/pairs/pairs_anchor_$s.json"); done
    python -m src.analysis.activation_correlation classes \
        --act-counts "$A/ct/activation_correlation_counts_ct.h5" \
        --act-json "$A/ct/activation_correlation_ct.json" \
        --sim "$A/ct/jackknife_match_ct.h5" \
        --rf-counts-all-theta "$A/rf_counts_all_theta_n39.h5" \
        --m1-json "$A/rf_marginal_m1_n39.json" \
        --gtilde "$A/gtilde_compliant_b_ratio.h5" \
        --null-reductions "$A/null_pilot_compliant.h5" \
        --d-fs "$A/d_fs.h5" --degenerate-flag "$A/degenerate_argmax_flag.h5" \
        --pairs "${PAIRS[@]}" \
        --out "$A/ct/activation_logM_by_class_ct.json"
    ;;
  *) echo "unknown step $STEP" >&2; exit 1 ;;
esac
[[ -z "$(git status --porcelain)" ]] || { echo "checkout dirty on exit" >&2; exit 1; }
