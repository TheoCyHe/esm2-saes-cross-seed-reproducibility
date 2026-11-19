#!/bin/bash
#SBATCH --job-name=concept_subspace
#SBATCH --partition=day
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
# Amendment CT, §CT22: concept-subspace reproducibility. CPU, float64 numpy.
# The only device-dependent input is §CT3's stored mnn_sim (rtx_5000_ada), read
# through the record's own loader and gate.
#
#   sbatch slurm/concept_subspace.sbatch pilot
#   sbatch slurm/concept_subspace.sbatch confirm <m> <measure> <alignment>
#
# confirm refuses unless (m, measure, alignment) equal the pilot artifact's lock.
set -euo pipefail
PROJECT=$PROJECT
A="$PROJECT/analysis"
MODE="$1"
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ -n "$(git status --porcelain)" ]]; then
    echo "REFUSING: the checkout is dirty before the run" >&2; exit 1
fi
ANCHORS=("$PROJECT"/sae_ckpts/seed_*)
if [[ "${#ANCHORS[@]}" -ne 20 ]]; then
    echo "FATAL: expected exactly 20 anchor checkpoints, found ${#ANCHORS[@]}" >&2
    exit 1
fi
OUTDIR="${CT22_OUTDIR:-$A/ct}"          # a smoke run points this at scratch
EXTRA=()
if [[ -n "${CT22_SMOKE_MAX_CELLS:-}" ]]; then
    EXTRA+=(--max-cells "$CT22_SMOKE_MAX_CELLS")
fi
if [[ "$MODE" == "confirm" ]]; then
    EXTRA+=(--pilot "$OUTDIR/ct22_pilot_ct.json" --m "$2" --measure "$3" --alignment "$4")
fi
python -m src.analysis.concept_subspace "$MODE" \
    --sim "$A/ct/jackknife_match_ct.h5" \
    --rf-counts-all-theta "$A/rf_counts_all_theta_n39.h5" \
    --m1-json "$A/rf_marginal_m1_n39.json" \
    --gtilde "$A/gtilde_compliant_b_ratio.h5" \
    --null-reductions "$A/null_pilot_compliant.h5" \
    --d-fs "$A/d_fs.h5" \
    --degenerate-flag "$A/degenerate_argmax_flag.h5" \
    --ap "$A/feature_ap_main_compliant.h5" \
    --bundle-concepts "$PROJECT/release/derived/concepts_carriers.csv.gz" \
    --carrier-reference "$A/carrier_concentration_bb.json" \
    --tuning-set docs/ct22_tuning_set.json \
    --ckpts "${ANCHORS[@]}" --expect-ckpts 20 \
    --workers "$SLURM_CPUS_PER_TASK" \
    --out "$OUTDIR/ct22_${MODE}_ct.json" \
    "${EXTRA[@]}"
if [[ -n "$(git status --porcelain)" ]]; then
    echo "the checkout is dirty on exit" >&2; exit 1
fi
