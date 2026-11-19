#!/bin/bash
#SBATCH --job-name=sae_ac_closeout
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

# Amendment AC close-out, two items:
#   AC10 -- bootstrap CIs on all THREE CV ratios, so the "2.2x worse" claim
#           does not enter the record unquantified.
#   AC11 -- C1 across ALL 20 seeds. C1 ONLY: no fit, no regression, no C2.
#           C2 across 20 seeds would preview the primary coefficient before
#           the specification is frozen -- a larger firewall crossing than
#           was authorised, and avoidable since C1 answers the question more
#           directly.

set -euo pipefail
SCRATCH=$SCRATCH
PROJECT=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

for ARM in hardbin smooth_with smooth_without; do
  case $ARM in
    hardbin)        R="$PROJECT/analysis/gtilde_compliant_b_ratio.h5" ;;
    smooth_with)    R="$SCRATCH/analysis/gtilde_compliant_b_ratio_smooth.h5" ;;
    smooth_without) R="$SCRATCH/analysis/gtilde_compliant_b_ratio_smooth_nodegen.h5" ;;
  esac
  python -m src.analysis.compare_inner_modes \
      --full   "$SCRATCH/analysis/gtilde_compliant_b.h5" \
      --centre "$SCRATCH/analysis/gtilde_compliant_b_centre.h5" \
      --none   "$SCRATCH/analysis/gtilde_compliant_b_none.h5" \
      --ratio  "$R" \
      --out    "$PROJECT/analysis/cv_ci_${ARM}.json"
done

python -m src.analysis.m4_confound --c1-only \
    --ap "$PROJECT/analysis/feature_ap_main_compliant.h5" \
    --null-raw "$PROJECT/analysis/null_gain_raw_compliant.h5" \
    --null-reductions "$PROJECT/analysis/null_pilot_compliant.h5" \
    --labels "$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5" \
    --row-index-cache "$SCRATCH/row_index_cache" \
    --ckpts "$PROJECT/sae_ckpts" --eval-cache "$SCRATCH/cache/eval.h5" \
    --rf-counts "$PROJECT/analysis/rf_counts_theta070.h5" \
    --mu-smooth "$SCRATCH/analysis/smooth_mu_b.h5" \
    --out "$PROJECT/analysis/m4_c1_all_seeds.json"
