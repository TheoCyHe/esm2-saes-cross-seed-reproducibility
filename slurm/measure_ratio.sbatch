#!/bin/bash
#SBATCH --job-name=sae_ratio
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# Amendment U1's ratio normalization. Predictions committed in
# compare_inner_modes' docstring BEFORE this runs.
set -euo pipefail
SCRATCH=$SCRATCH
PROJECT=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
LAB="$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5"
OUT="$PROJECT/analysis/gtilde_compliant_b_ratio.h5"

[ -f "$OUT" ] || python -m src.analysis.standardize_gain \
    --ap "$PROJECT/analysis/feature_ap_main_compliant.h5" \
    --null-raw "$PROJECT/analysis/null_gain_raw_compliant.h5" \
    --null-reductions "$PROJECT/analysis/null_pilot_compliant.h5" \
    --labels "$LAB" --inner-mode ratio --config configs/mvp.yaml --out "$OUT"

python -m src.analysis.compare_inner_modes \
    --full  "$SCRATCH/analysis/gtilde_compliant_b.h5" \
    --centre "$SCRATCH/analysis/gtilde_compliant_b_centre.h5" \
    --none  "$SCRATCH/analysis/gtilde_compliant_b_none.h5" \
    --ratio "$OUT" \
    --out   "$PROJECT/analysis/inner_mode_comparison_ratio.json"
