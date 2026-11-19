#!/bin/bash
#SBATCH --job-name=ct26
#SBATCH --output=logs/%x_%j.out
# §CT26 (src/analysis/ct26_nonanchor.py). Resources on the sbatch line:
#   match     gpu, --gpus=rtx_5000_ada:1 (§CT3's device class)
#   control   day CPU, 16 cores: the anchors through the same code; must
#             reproduce ct22_confirm_ct.json and ct22_diagnostics_ct.json EXACTLY
#   run       day CPU, 16 cores: seeds 20-39 under §CT22's lock
set -euo pipefail
MODE="${1:?match|control|run}"
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
[[ -z "$(git status --porcelain)" ]] || { echo "checkout dirty" >&2; exit 1; }
echo "checkout $(git rev-parse HEAD) mode=$MODE host=$(hostname)"
PROJECT=$PROJECT
A="$PROJECT/analysis"
SIM40="$A/ct/ct26_mnn_sim_all40_ct.h5"
ANCHORS=("$PROJECT"/sae_ckpts/seed_*)
EXT=("$PROJECT"/sae_ckpts_ext/seed_*)
[[ ${#ANCHORS[@]} -eq 20 && ${#EXT[@]} -eq 20 ]] || { echo "expected 20 + 20 checkpoints" >&2; exit 1; }
case "$MODE" in
  match)
    python -m src.analysis.ct26_nonanchor match --ckpts "${ANCHORS[@]}" --compare-ckpts "${EXT[@]}" \
        --stored-sim "$A/ct/jackknife_match_ct.h5" --out "$SIM40"
    sha256sum "$SIM40" ;;
  control)
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python -m src.analysis.ct26_nonanchor run --ap "$A/feature_ap_main_compliant.h5" \
        --null-reductions "$A/null_pilot_compliant.h5" --sim "$SIM40" --ckpts "${ANCHORS[@]}" \
        --exclude-concepts docs/ct22_tuning_set.json \
        --control-confirm "$A/ct/ct22_confirm_ct.json" --control-diag "$A/ct/ct22_diagnostics_ct.json" \
        --out "$A/ct/ct26_anchor_control_ct.json"
    sha256sum "$A/ct/ct26_anchor_control_ct.json" ;;
  run)
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python -m src.analysis.ct26_nonanchor run --ap "$A/ct/feature_ap_ext_ct.h5" \
        --null-reductions "$A/ct/null_pilot_ext_ct.h5" --sim "$SIM40" --ckpts "${EXT[@]}" \
        --out "$A/ct/ct26_nonanchor_ct.json"
    sha256sum "$A/ct/ct26_nonanchor_ct.json" "$A/ct/ct26_nonanchor_ct.h5" ;;
  *) echo "unknown MODE $MODE" >&2; exit 1 ;;
esac
[[ -z "$(git status --porcelain)" ]] || { echo "checkout dirty on exit" >&2; exit 1; }
