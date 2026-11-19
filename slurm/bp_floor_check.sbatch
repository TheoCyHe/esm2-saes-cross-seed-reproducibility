#!/bin/bash
#SBATCH --job-name=sae_bp_floor
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

# PREREGISTRATION amendment BP, part A -- the 40-cell structural check.
#
# Scores prediction BP-P1 (filed at BP3, committed at f93847f BEFORE this ran)
# and re-runs BO3b's check as an explicitly POST-HOC replication.
#
# NO GPU. Pure numpy and exact rationals over an artifact that is 38 MB on disk
# and ~360 MB decompressed, read one (seed, arm) slice at a time -- so 16G is
# generous and gotcha 36's device rule is not engaged: no activation gather
# occurs and nothing here re-derives an AP or a firing set.

set -euo pipefail

PROJECT=$PROJECT
BH="$PROJECT/analysis/classification_bh_experimental_bo.h5"
OUT="$PROJECT/analysis/bp_floor_check.json"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

if [[ ! -f "$BH" ]]; then echo "MISSING INPUT: $BH" >&2; exit 1; fi
if [[ -e "$OUT" ]]; then
    echo "$OUT already exists -- refusing to overwrite" >&2; exit 1
fi

python -m src.analysis.bp_floor_check --bh "$BH" --out "$OUT"
