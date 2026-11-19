#!/bin/bash
#SBATCH --job-name=sae_ap_exp_cpu_merge
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# Merge the 20 CPU-gathered experimental AP shards. Identical to
# merge_ap_experimental.sbatch except for SHARD_DIR, OUT and the report path --
# it runs the same merge_ap_shards module with the same reference and the same
# stratum, so the CPU artifact is checked by exactly the instrument that
# checked the GPU one.
#
# ARCHIVE, DO NOT REPLACE: OUT is feature_ap_experimental_av_cpu.h5, alongside
# feature_ap_experimental_av.h5 rather than over it. The GPU artifact is the
# record of what was computed before, and it is one of the two inputs to the
# CPU-versus-GPU divergence measurement.
#
# The merge already asserts what matters here: identity agreement across the 20
# shards with ABSENT reported absent, the concept axis identical, and the seed
# order resolved against the REFERENCE artifact's own labels and then verified
# POSITIONALLY (gotcha 29; AP artifacts sort lexicographically, rf artifacts
# numerically).

set -euo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT
LABELS="${1:-$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5}"
SHARD_DIR="$SCRATCH/ap_exp_shards_cpu"
OUT="$PROJECT/analysis/feature_ap_experimental_av_cpu.h5"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

# EXACT count, not "at least one" -- a consistency check passes at 19 (the
# lesson AG recorded when nine sbatch scripts globbed an ensemble directory and
# only one guarded the count).
N=$(ls "$SHARD_DIR"/shard_seed_*.h5 2>/dev/null | wc -l)
if [[ "$N" -ne 20 ]]; then
    echo "FATAL: found $N shards in $SHARD_DIR, expected exactly 20." >&2
    exit 1
fi

python -m src.analysis.merge_ap_shards \
    --shards "$SHARD_DIR"/shard_seed_*.h5 \
    --reference "$PROJECT/analysis/feature_ap_main_compliant.h5" \
    --labels "$LABELS" \
    --stratum experimental \
    --out "$OUT" \
    --report "$PROJECT/analysis/feature_ap_experimental_av_cpu_merge.json"
