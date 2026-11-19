#!/bin/bash
#SBATCH --job-name=sae_m4_cv3
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# AC section 3 -- THE THREE-ARMED CV RE-RUN.
#
#   baseline      hard-bin mu, 22 included  -- already measured, V1, 1.0069
#   smooth-with   smooth-trend mu, 22 in the fit
#   smooth-without smooth-trend mu, 22 OUT of estimation, IN the max pool
#
# Same 1.084 bar. Three arms rather than one, because if a single re-run
# bundled the smooth trend with the 22-handling and failed, THE FAILURE WOULD
# BE UNATTRIBUTABLE.
#
# compare_inner_modes is run twice with a different --ratio input, which needs
# no change to that module: it reports cv_ratio_vs_none for whatever it is
# handed.

set -euo pipefail
SCRATCH=$SCRATCH
PROJECT=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

LAB="$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5"
AP="$PROJECT/analysis/feature_ap_main_compliant.h5"
NR="$PROJECT/analysis/null_gain_raw_compliant.h5"
ND="$PROJECT/analysis/null_pilot_compliant.h5"

# ---- smooth-WITHOUT mu (the smooth-with mu already exists from job 20408045)
if [ ! -f "$SCRATCH/analysis/smooth_mu_b_nodegen.h5" ]; then
python -m src.analysis.smooth_mu \
    --ap "$AP" --null-raw "$NR" --null-reductions "$ND" --labels "$LAB" \
    --row-index-cache "$SCRATCH/row_index_cache" --ckpts "$PROJECT/sae_ckpts" \
    --eval-cache "$SCRATCH/cache/eval.h5" \
    --scored-row-set label_store --k 32 --arm b --exclude-degenerate \
    --out "$SCRATCH/analysis/smooth_mu_b_nodegen.h5" \
    --report "$PROJECT/analysis/smooth_mu_m4_nodegen.json"
fi

# ---- the two gtilde artifacts
for ARM in with without; do
  case $ARM in
    with)    MU="$SCRATCH/analysis/smooth_mu_b.h5";         OUT="$SCRATCH/analysis/gtilde_compliant_b_ratio_smooth.h5" ;;
    without) MU="$SCRATCH/analysis/smooth_mu_b_nodegen.h5"; OUT="$SCRATCH/analysis/gtilde_compliant_b_ratio_smooth_nodegen.h5" ;;
  esac
  [ -f "$OUT" ] || python -m src.analysis.standardize_gain \
      --ap "$AP" --null-raw "$NR" --null-reductions "$ND" --labels "$LAB" \
      --inner-mode ratio --mu-smooth "$MU" \
      --config configs/mvp.yaml --out "$OUT"
done

# ---- the CV comparison, once per smooth arm
python -m src.analysis.compare_inner_modes \
    --full   "$SCRATCH/analysis/gtilde_compliant_b.h5" \
    --centre "$SCRATCH/analysis/gtilde_compliant_b_centre.h5" \
    --none   "$SCRATCH/analysis/gtilde_compliant_b_none.h5" \
    --ratio  "$SCRATCH/analysis/gtilde_compliant_b_ratio_smooth.h5" \
    --out    "$PROJECT/analysis/cv_arm_smooth_with.json"

python -m src.analysis.compare_inner_modes \
    --full   "$SCRATCH/analysis/gtilde_compliant_b.h5" \
    --centre "$SCRATCH/analysis/gtilde_compliant_b_centre.h5" \
    --none   "$SCRATCH/analysis/gtilde_compliant_b_none.h5" \
    --ratio  "$SCRATCH/analysis/gtilde_compliant_b_ratio_smooth_nodegen.h5" \
    --out    "$PROJECT/analysis/cv_arm_smooth_without.json"
