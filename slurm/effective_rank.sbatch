#!/bin/bash
#SBATCH --job-name=cf_effective_rank
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

# M2 -- effective rank of decoder submatrices over r_f. AB15, scoped at CF3a.
# CPU only: 100 SVDs of 1280 x 2048 plus the null draws. No eval cache, no AP
# matrix, no concept library -- and therefore no purge exposure.

set -euo pipefail
PROJECT=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
echo "[m2.sbatch] git $(git rev-parse HEAD)  status [$(git status --porcelain | tr '\n' ';')]"

# The output of record, effective_rank_m2_cf.json (job 22634909), was cut with
# the script's then-defaults 1 / 18 -- Y2's N = 19 cuts -- on N = 39 counts
# (corrections entry 271). This launcher now passes AG1's N = 39 cuts
# explicitly (amendment CT, section CT8 item 3) and writes under a new name, so
# the flagged file of record is never overwritten.
OUT="$PROJECT/analysis/${1:-effective_rank_m2_ag1_ct.json}"
if [[ -e "$OUT" ]]; then echo "$OUT exists -- refusing to overwrite" >&2; exit 1; fi

python -m src.analysis.effective_rank \
    --rf-counts "$PROJECT/analysis/rf_counts_theta070_n39.h5" \
    --ckpts     "$PROJECT/sae_ckpts" \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --n-compare 39 \
    --null-draws 20 \
    --endpoint-low 2 --endpoint-high 37 \
    --out "$OUT"
