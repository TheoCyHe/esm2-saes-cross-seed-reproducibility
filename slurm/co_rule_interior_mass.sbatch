#!/bin/bash
#SBATCH --job-name=co_interior
#SBATCH --partition=day
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=$PROJECT/analysis/adhoc/co/co_interior_%j.out
#SBATCH --error=$PROJECT/analysis/adhoc/co/co_interior_%j.err

# AMENDMENT CO ITEM B. 190 unordered anchor pairs at the full 10,240 square,
# CPU for every rule so the rule comparison is internally device-matched
# (CJ3.5, gotcha 36). Resources match the CJ run that took 36 m 30 s.
set -euo pipefail
cd ~/esm2-saes-cross-seed-reproducibility
source env/activate.sh
echo "host=$(hostname) device_class=CPU"
python -c "import scipy; print('scipy', scipy.__version__)"
git status --porcelain

P=$PROJECT

time python -m src.analysis.rule_interior_mass \
  --ckpt-root "$P/sae_ckpts" \
  --rf-counts-n19 "$P/analysis/rf_counts_theta070_n19.h5" \
  --out-json "$P/analysis/rule_interior_mass_co.json" \
  --out-h5 "$P/analysis/rule_interior_mass_co.h5"
