#!/bin/bash
#SBATCH --job-name=sae_null_exp_merge
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# Merge the 20 per-seed masked-null shards (brief B).
#
# ROWS ARE ORDERED TO A NAMED REFERENCE'S OWN SEED LABELS, never positionally
# (gotcha 29).
# feature_ap_experimental_av.h5 is the reference because it is the artifact
# this null is a null FOR.
#
# THE MERGED RAW GOES TO PROJECT and survives the purge; the shards stay on
# scratch as regenerable intermediates.

set -euo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT
SHARD_DIR="$SCRATCH/null_exp_shards"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

# EXACTLY 20, asserted rather than inferred. A consistency check passes at 19
# and would produce a 19-seed artifact whose shape does not show the gap.
RED=(); RAW=()
for s in $(seq 0 19); do
    r="$SHARD_DIR/red_seed_${s}.h5"
    w="$SHARD_DIR/raw_seed_${s}.h5"
    [[ -f "$r" && -f "$w" ]] || { echo "FATAL: missing shard for seed $s" >&2; exit 1; }
    RED+=("$r"); RAW+=("$w")
done
if [[ ${#RED[@]} -ne 20 ]]; then
    echo "FATAL: ${#RED[@]} reduction shards, expected exactly 20" >&2; exit 1
fi

df -h <cluster-fs>/project | tail -1

python -m src.analysis.merge_null_shards \
    --shard-reductions "${RED[@]}" \
    --shard-raw "${RAW[@]}" \
    --reference "$PROJECT/analysis/feature_ap_experimental_av.h5" \
    --out "$PROJECT/analysis/null_experimental_bg.h5" \
    --raw-out "$PROJECT/analysis/null_gain_raw_experimental_bg.h5" \
    --report "$PROJECT/analysis/null_experimental_bg_merge_report.json"

df -h <cluster-fs>/project | tail -1
ls -la "$PROJECT/analysis/null_experimental_bg.h5" \
       "$PROJECT/analysis/null_gain_raw_experimental_bg.h5"
