#!/bin/bash
#SBATCH --job-name=concept_subspace_diag
#SBATCH --partition=day
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
# Amendment CT, §CT22.5: the descriptive diagnostics, never scored. Runs after
# the confirmatory result is committed, under the pilot's lock (it refuses
# otherwise). CPU, float64.
#
#   sbatch slurm/concept_subspace_diag.sbatch <m> <measure> <alignment>
#   sbatch --export=ALL,CT22_DIAGNOSTICS=no_disorder_ap,CT22_DIAG_NAME=ct22_diagnostics_ap_argmax_ct.json \
#       slurm/concept_subspace_diag.sbatch 3 omega ap
set -euo pipefail
PROJECT=$PROJECT
A="$PROJECT/analysis"
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
OUTDIR="${CT22_OUTDIR:-$A/ct}"
EXTRA=()
if [[ -n "${CT22_SMOKE_MAX_CELLS:-}" ]]; then
    EXTRA+=(--max-cells "$CT22_SMOKE_MAX_CELLS")
fi
if [[ -n "${CT22_DIAGNOSTICS:-}" ]]; then          # a subset, e.g. "no_disorder_ap"
    read -r -a DIAGS <<< "$CT22_DIAGNOSTICS"
    EXTRA+=(--diagnostics "${DIAGS[@]}")
fi
NAME="${CT22_DIAG_NAME:-ct22_diagnostics_ct.json}"
python -m src.analysis.concept_subspace_diag confirm \
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
    --pilot "$A/ct/ct22_pilot_ct.json" --m "$1" --measure "$2" --alignment "$3" \
    --out "$OUTDIR/$NAME" \
    "${EXTRA[@]}"
if [[ -n "$(git status --porcelain)" ]]; then
    echo "the checkout is dirty on exit" >&2; exit 1
fi
