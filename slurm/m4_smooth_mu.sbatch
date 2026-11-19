#!/bin/bash
#SBATCH --job-name=sae_m4_smooth
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# M4 -- smooth-trend mu_s (PREREGISTRATION amendment U3, branch declared in
# amendment AC). ONE additive penalised-spline fit per seed over exactly the
# four covariates U3 names. Predictions S1-S5 are in the module docstring in
# git BEFORE this runs.
#
# THE BRANCH, so it routes rather than needing a judgment later: if the smooth
# predictor fails V1's 1.084 CV bar and hard-bin passes, HARD-BIN REMAINS of
# record and U3's supersession is WITHDRAWN.
#
# Feature-side covariates are recomputed PER SEED from that seed's own row
# index (gotcha 18). The row
# index is resolved by IDENTITY attrs, not by cache key, because the key folds
# in git_sha and would force a needless 20-seed rebuild.

set -euo pipefail
SCRATCH=$SCRATCH
PROJECT=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

python -m src.analysis.config_audit --config configs/mvp.yaml

python -m src.analysis.smooth_mu \
    --ap "$PROJECT/analysis/feature_ap_main_compliant.h5" \
    --null-raw "$PROJECT/analysis/null_gain_raw_compliant.h5" \
    --null-reductions "$PROJECT/analysis/null_pilot_compliant.h5" \
    --labels "$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5" \
    --row-index-cache "$SCRATCH/row_index_cache" \
    --ckpts "$PROJECT/sae_ckpts" \
    --eval-cache "$SCRATCH/cache/eval.h5" \
    --scored-row-set label_store --k 32 --arm b \
    --out "$SCRATCH/analysis/smooth_mu_b.h5" \
    --report "$PROJECT/analysis/smooth_mu_m4.json"
