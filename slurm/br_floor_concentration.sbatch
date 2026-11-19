#!/bin/bash
#SBATCH --job-name=sae_br_floorconc
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.out

# PREREGISTRATION amendment BR, part C -- the per-concept decomposition of the
# BH resolution floor, DECLARED at §BR-C before this ran (commit 0ac2a95).
#
# NO GPU, and gotcha 36's device rule is not engaged: no activation gather
# occurs, no AP or firing set is re-derived, and nothing here does
# device-dependent float arithmetic. The numerators are recovered as EXACT
# integers from stored float32 p-values and stored integer denominators, then
# verified by an exact round-trip. The inputs' own device provenance is copied
# into the artifact's identity rather than inferred.
#
# Memory: the pooled p-array is (20, 10240, 703) float32 = 576 MB decompressed
# but is read ONE SEED AT A TIME (28.8 MB per slab); the experimental one is
# (20, 10240, 218) = 178 MB, 8.9 MB per slab. 32G is generous.

set -euo pipefail

PROJECT=$PROJECT
A="$PROJECT/analysis"

BH_EXP="$A/classification_bh_experimental_bo.h5"
AP_EXP="$A/feature_ap_experimental_av_cpu.h5"
TWO_BY_TWO="$A/classification_2x2_bq.h5"
AP_POOL="$A/feature_ap_main_compliant.h5"

OUT="$A/br_floor_concentration.h5"
REPORT="$A/br_floor_concentration.json"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

for f in "$BH_EXP" "$AP_EXP" "$TWO_BY_TWO" "$AP_POOL"; do
    if [[ ! -f "$f" ]]; then echo "MISSING INPUT: $f" >&2; exit 1; fi
done
# NEW PATH, never an overwrite -- the relocate-never-delete rule.
for f in "$OUT" "$REPORT"; do
    if [[ -e "$f" ]]; then
        echo "$f already exists -- refusing to overwrite" >&2; exit 1
    fi
done

python -m src.analysis.floor_concentration \
    --bh-experimental "$BH_EXP" \
    --ap-experimental "$AP_EXP" \
    --twobytwo-pooled "$TWO_BY_TWO" \
    --ap-pooled "$AP_POOL" \
    --out "$OUT" \
    --report "$REPORT"
