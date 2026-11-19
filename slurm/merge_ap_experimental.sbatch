#!/bin/bash
#SBATCH --job-name=sae_ap_exp_merge
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# Merge the 20 per-seed experimental AP shards into one artifact
# (PREREGISTRATION amendment AV10d). No GPU: this is h5 I/O and assertions.
#
# The merge does four things no shape check can do:
#   * orders rows by the REFERENCE artifact's own seed labels, then verifies
#     the result POSITIONALLY (gotcha 29; AP artifacts are lexicographic,
#     rf artifacts numeric);
#   * asserts all 20 shards agree on identity, with ABSENT reported absent;
#   * asserts the experimental and pooled artifacts differ ONLY in `strata`,
#     `negatives` and `ap_impl`, and reports the assertion;
#   * VERIFIES that every shard's `n_concepts` / `argmax_pool_n` already carry
#     the LIBRARY values (gotcha 16's committed convention) and exits non-zero
#     otherwise. It relabels NOTHING: merge_ap_shards.py's relabel-on-write was
#     replaced by this verification (its own docstring, `merge_ap_shards.py`
#     "This used to be a relabel-on-write"); feature_ap.py has written the
#     library-scoped counts itself since. (Comment corrected 2026-09-16.)
#
# Exits non-zero if the comparability verdict is not COMPARABLE (gotcha 13,
# extension 1).

set -euo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT
LABELS="${1:-$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5}"
SHARD_DIR="$SCRATCH/ap_exp_shards"
OUT="$PROJECT/analysis/feature_ap_experimental_av.h5"

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
    --report "$PROJECT/analysis/feature_ap_experimental_av_merge.json"
