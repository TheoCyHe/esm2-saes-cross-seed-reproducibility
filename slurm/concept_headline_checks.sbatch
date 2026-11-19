#!/bin/bash
#SBATCH --job-name=sae_headline_checks
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

# TWO VALIDITY CHECKS ON THE CONCEPT-LEVEL HEADLINE -- amendment AZ sections
# 13 and 14 (brief 4a + 4b).
#
# NO GPU AND NO AVERAGE PRECISION IS RECOMPUTED.
#
# 4a  DOES THE PERMUTATION NULL PRESERVE EACH CONCEPT'S PREVALENCE? Listed as
#     unmeasured since AW1.4 and sitting directly under the positive headline:
#     AP is prevalence-dependent, so if a concept's positive count drifts under
#     permutation the threshold describes a concept of a different size.
#     `null_pilot.cross_protein_shift_rows` rescales offsets onto a destination
#     protein and the map is MANY-TO-ONE when the destination is shorter, after
#     which null_pilot dedups and uses the realised count -- so prevalence is
#     NOT preserved by design and the question is how far it drifts. The STORED
#     `dest_schedule` is replayed; it is never regenerated from `null_seed`,
#     because a regenerated schedule is not the one the null used. If the
#     dataset is absent the module REPORTS THAT AND STOPS.
#
# 4b  THE BOUNDED HEADLINE (AZ4). The shared-SCHEDULE channel cannot be removed
#     from stored draws and no new null pass is authorised, so the headline is
#     reported WITH and WITHOUT the concepts whose worst seed clears its
#     threshold by less than 2x. Arithmetic, not a prediction.
#
# NOTHING IS CORRECTED (AZ10). Whatever 4a finds is reported.
#
# THE ROW-TO-PROTEIN MAP is built by null_pilot's own build_population_index,
# which sorts proteins by ORIGINAL START. Building it from
# `protein_sets/population_start` as stored is the AO6 defect: that array is in
# sorted-ACCESSION order with 25,508 descending steps, and it attributed
# 0.0031% of rows correctly while every total agreed. The schedule's protein
# count is asserted against the index built here rather than broadcast.

set -euo pipefail

PROJECT=$PROJECT
SCRATCH=$SCRATCH

AP="$PROJECT/analysis/feature_ap_main_compliant.h5"
NULL="$PROJECT/analysis/null_pilot_compliant.h5"
LABELS="$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5"
EVALCACHE="$SCRATCH/cache/eval.h5"
OUT="$PROJECT/analysis/${1:-concept_headline_checks_az.json}"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

for f in "$AP" "$NULL" "$LABELS" "$EVALCACHE"; do
    if [[ ! -f "$f" ]]; then
        echo "FATAL: missing $f" >&2; exit 1
    fi
done
if [[ -e "$OUT" ]]; then
    echo "FATAL: $OUT exists; every artifact goes to a NEW path" >&2; exit 1
fi

echo "[checks.sbatch] labels=$(basename "$LABELS")  out=$OUT"

python -m src.analysis.concept_headline_checks \
    --ap "$AP" --null "$NULL" --labels "$LABELS" \
    --eval-cache "$EVALCACHE" \
    --scored-row-set label_store \
    --out "$OUT"

DIRTY=$(git -C "$SLURM_SUBMIT_DIR" status --porcelain | wc -l)
if [[ "$DIRTY" -ne 0 ]]; then
    echo "FATAL: the checkout is dirty on exit ($DIRTY entries)" >&2
    git -C "$SLURM_SUBMIT_DIR" status --porcelain >&2
    exit 1
fi

echo "wrote $OUT"
