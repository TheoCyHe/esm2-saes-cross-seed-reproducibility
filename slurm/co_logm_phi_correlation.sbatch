#!/bin/bash
#SBATCH --job-name=co_logm_phi
#SBATCH --partition=day
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=$PROJECT/analysis/adhoc/co/co_logm_phi_%j.out
#SBATCH --error=$PROJECT/analysis/adhoc/co/co_logm_phi_%j.err

# AMENDMENT CO ITEM A. Arithmetic on two stored (20, 10240) arrays; batch rather
# than the login node because a job record is part of the artifact's provenance,
# not because it is expensive.
set -euo pipefail
cd ~/esm2-saes-cross-seed-reproducibility
source env/activate.sh
echo "host=$(hostname) device_class=CPU"
git status --porcelain

P=$PROJECT

python -m src.analysis.logm_phi_correlation \
  --gtilde "$P/analysis/gtilde_compliant_b_ratio.h5" \
  --null-reductions "$P/analysis/null_pilot_compliant.h5" \
  --out "$P/analysis/logm_phi_correlation_co.json"
