#!/bin/bash
#SBATCH --job-name=sae_rf_marginal
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; torch cu126 has no sm_120 (Blackwell) kernels
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# M1 -- the r_f MARGINAL DISTRIBUTIONS. PREREGISTRATION amendments X, AG, AK.
#
# USAGE:  sbatch slurm/rf_marginal.sbatch [n19|n39|preflight|n39_theta_all]
#
#   n19        the SENSITIVITY arm of record. 20 anchors, no extension seeds,
#              N = 19. Must reproduce analysis/repro/final2/ (AK1).
#   n39        the PRIMARY arm. 20 anchors (seeds 0-19) x 40 dictionaries,
#              N = 39. Judged against AK's pre-committed predictions.
#   preflight  load all 40, run identity / unit-norm / b_dec-distinctness,
#              write the report, compute no marginal.
#   n39_theta_all
#              amendment AR route A. The same N = 39 computation, writing
#              per-anchor counts at EVERY grid theta to a NEW path, and
#              checking the reference-theta slice is BIT-IDENTICAL to the
#              existing rf_counts_theta070_n39.h5. WRITES NO CANARY TARGET:
#              both outputs are new paths, and rf_counts_theta070_n39.h5 is
#              read only. Exits non-zero if the acceptance criterion fails.
#
# THE GLOB HAZARD (AD5, docs/CODEBASE.md, and the reason this script has a
# count guard when eight of its siblings do not). Extension seeds live in
# sae_ckpts_ext/, anchors in sae_ckpts/. A run that reuses the bare
# sae_ckpts/seed_* glob measures the 20 anchors and labels the result N = 39 --
# internally consistent, describing the wrong ensemble.
#
# SO THE GUARD ASSERTS AN EXACT EXPECTED TOTAL, TWICE AND INDEPENDENTLY: here
# in shell, and again inside python via --expect-total-ckpts, which refuses
# before loading. A SELF-CONSISTENCY CHECK ON THE COUNT PASSES AT 20 and is
# therefore not a guard at all.
#
# INPUTS: W_dec ONLY. No activation cache, no AP matrix, no concept library --
# amendment X7's firewall HARD-FAILS the job if any alignment artifact is
# opened. The predictions this job scores were committed to git in
# PREREGISTRATION amendment AK BEFORE submission; `git log` is the audit trail.

set -euo pipefail

MODE="${1:-n19}"
PROJECT=$PROJECT

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

ANCHORS=("$PROJECT"/sae_ckpts/seed_*)
EXT=("$PROJECT"/sae_ckpts_ext/seed_*)

# Identity is asserted downstream (gotcha 15), but a missing tree
# still has to be caught before the count comparison reads a glob literal.
if [[ ! -e "${ANCHORS[0]}/model.pt" ]]; then
    echo "no anchor checkpoints in $PROJECT/sae_ckpts" >&2
    exit 1
fi
if [[ "${#ANCHORS[@]}" -ne 20 ]]; then
    echo "expected exactly 20 anchor checkpoints, found ${#ANCHORS[@]}" >&2
    exit 1
fi

case "$MODE" in
  n19)
    EXPECT=20
    OUT="$PROJECT/analysis/rf_marginal_m1_n19.json"
    COUNTS="$PROJECT/analysis/rf_counts_theta070_n19.h5"
    COMPARE=()
    ;;
  n39|preflight|n39_theta_all)
    if [[ ! -e "${EXT[0]}/model.pt" ]]; then
        echo "no extension checkpoints in $PROJECT/sae_ckpts_ext" >&2
        exit 1
    fi
    if [[ "${#EXT[@]}" -ne 20 ]]; then
        echo "expected exactly 20 extension checkpoints, found ${#EXT[@]}" >&2
        exit 1
    fi
    EXPECT=40
    COMPARE=("${EXT[@]}")
    if [[ "$MODE" == "preflight" ]]; then
        OUT="$PROJECT/analysis/rf_preflight_s40.json"
        COUNTS=""
    elif [[ "$MODE" == "n39_theta_all" ]]; then
        OUT="$PROJECT/analysis/rf_marginal_m1_n39_alltheta.json"
        COUNTS=""
        ALLTHETA="$PROJECT/analysis/rf_counts_all_theta_n39.h5"
        REFCOUNTS="$PROJECT/analysis/rf_counts_theta070_n39.h5"
        # The reference must EXIST before the run, because the acceptance
        # criterion is the point of the run. Failing here is louder than
        # failing after 28 s of GPU.
        if [[ ! -f "$REFCOUNTS" ]]; then
            echo "the acceptance reference $REFCOUNTS is absent" >&2
            exit 1
        fi
        # Refuse to overwrite a previous all-theta artifact: two generations
        # of the same object silently mixing is the failure ao_rebuild_smooth
        # guards against for the same reason.
        if [[ -e "$ALLTHETA" ]]; then
            echo "$ALLTHETA already exists -- refusing to overwrite" >&2
            exit 1
        fi
    else
        OUT="$PROJECT/analysis/rf_marginal_m1_n39.json"
        COUNTS="$PROJECT/analysis/rf_counts_theta070_n39.h5"
    fi
    ;;
  *)
    echo "unknown mode '$MODE' -- expected n19, n39, preflight or n39_theta_all" >&2
    exit 1
    ;;
esac

TOTAL=$(( ${#ANCHORS[@]} + ${#COMPARE[@]} ))
if [[ "$TOTAL" -ne "$EXPECT" ]]; then
    echo "mode $MODE expects EXACTLY $EXPECT checkpoints, assembled $TOTAL" >&2
    exit 1
fi
echo "[rf.sbatch] mode=$MODE  anchors=${#ANCHORS[@]}  compare-only=${#COMPARE[@]}  total=$TOTAL (expected $EXPECT)"

if [[ "$MODE" == "preflight" ]]; then
    python -m src.analysis.rf_marginal \
        --ckpts "${ANCHORS[@]}" \
        --compare-ckpts "${COMPARE[@]}" \
        --expect-total-ckpts "$EXPECT" \
        --preflight-only \
        --out "$OUT"
    exit $?
fi

if [[ "$MODE" == "n39_theta_all" ]]; then
    python -m src.analysis.rf_marginal \
        --ckpts "${ANCHORS[@]}" \
        --compare-ckpts "${COMPARE[@]}" \
        --expect-total-ckpts "$EXPECT" \
        --rotation-draws 50 \
        --hungarian-sizes 1024 2048 4096 \
        --hungarian-repeats 5 \
        --rng-seed 0 \
        --out "$OUT" \
        --emit-counts-all-theta "$ALLTHETA" \
        --counts-reference "$REFCOUNTS"
    exit $?
fi

if [[ "${#COMPARE[@]}" -eq 0 ]]; then
    python -m src.analysis.rf_marginal \
        --ckpts "${ANCHORS[@]}" \
        --expect-total-ckpts "$EXPECT" \
        --rotation-draws 50 \
        --hungarian-sizes 1024 2048 4096 \
        --hungarian-repeats 5 \
        --rng-seed 0 \
        --out "$OUT" --emit-counts "$COUNTS"
else
    python -m src.analysis.rf_marginal \
        --ckpts "${ANCHORS[@]}" \
        --compare-ckpts "${COMPARE[@]}" \
        --expect-total-ckpts "$EXPECT" \
        --rotation-draws 50 \
        --hungarian-sizes 1024 2048 4096 \
        --hungarian-repeats 5 \
        --rng-seed 0 \
        --out "$OUT" --emit-counts "$COUNTS"
fi
