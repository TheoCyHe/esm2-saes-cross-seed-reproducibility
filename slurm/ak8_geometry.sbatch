#!/bin/bash
#SBATCH --job-name=ch_ak8_geometry
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# AK8's GEOMETRIC HALF -- amendment CH2, declared at 68f0851 before the module
# existed. Deferred from AK8 as "needs M2"; M2 ran at CF3c, so the blocker is
# discharged.
#
# CPU only, and small: the discovery set is ~10^2 features per seed, so each
# seed is one SVD of 1280 x ~100 plus 20 null draws at the same width. No eval
# cache, no AP matrix -- and therefore no purge exposure.
#
# THE SIZE-MATCHED NULL IS THE POINT (CH2c). ER/d depends mechanically on
# submatrix width and CF3c2 measured that dependence dominating a naive
# contrast. Every seed's set is referred to a null drawn at THAT SEED'S size.

set -euo pipefail
PROJECT=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
echo "[ak8geo.sbatch] git $(git rev-parse HEAD)  status [$(git status --porcelain | tr '\n' ';')]"

OUT="$PROJECT/analysis/ak8_geometry_ch.json"
if [[ -e "$OUT" ]]; then echo "$OUT exists -- refusing to overwrite" >&2; exit 1; fi

python -m src.analysis.ak8_geometry \
    --rf-counts        "$PROJECT/analysis/rf_counts_theta070_n39.h5" \
    --gtilde           "$PROJECT/analysis/gtilde_compliant_b_ratio.h5" \
    --null-reductions  "$PROJECT/analysis/null_pilot_compliant.h5" \
    --ckpts            "$PROJECT/sae_ckpts" \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --n-compare 39 \
    --null-draws 20 \
    --out "$OUT"

echo "[ak8geo.sbatch] done"
