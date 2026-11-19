#!/bin/bash
#SBATCH --job-name=sae_meas_s
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# Measurements 1-5 for amendment S. Predictions are committed in the module
# docstrings BEFORE this runs.
set -euo pipefail
SCRATCH=$SCRATCH
PROJECT=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
LAB="$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5"

# M1 -- stage 1 split into centring and scaling
for MODE in centre none; do
  OUT="$SCRATCH/analysis/gtilde_compliant_b_${MODE}.h5"
  [ -f "$OUT" ] || python -m src.analysis.standardize_gain \
      --ap "$PROJECT/analysis/feature_ap_main_compliant.h5" \
      --null-raw "$PROJECT/analysis/null_gain_raw_compliant.h5" \
      --null-reductions "$PROJECT/analysis/null_pilot_compliant.h5" \
      --labels "$LAB" --inner-mode "$MODE" --config configs/mvp.yaml --out "$OUT"
done

# M2 / M4 / M5
python -m src.analysis.diagnose_stage3 \
    --gtilde "$SCRATCH/analysis/gtilde_compliant_b.h5" \
    --ap "$PROJECT/analysis/feature_ap_main_compliant.h5" \
    --null-raw "$PROJECT/analysis/null_gain_raw_compliant.h5" \
    --labels-json "$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.json" \
    --out "$PROJECT/analysis/stage3_diagnosis.json"

# M3 -- feature footprint (row index is per-seed; seed 0)
RIDX=$(python - <<'PY'
import glob, h5py
for g in sorted(glob.glob("$SCRATCH/row_index_cache/row_index_seed_0_*.h5")):
    with h5py.File(g, "r") as h:
        if h["row_offsets"].shape[0] == 15972326:
            print(g); break
PY
)
echo "row index: $RIDX"
python -m src.analysis.feature_footprint \
    --gtilde "$SCRATCH/analysis/gtilde_compliant_b.h5" \
    --labels "$LAB" \
    --null-reductions "$PROJECT/analysis/null_pilot_compliant.h5" \
    --row-index "$RIDX" \
    --out "$PROJECT/analysis/feature_footprint.json"
