#!/bin/bash
#SBATCH --job-name=sae_bh_plan
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

# AC4 -- reproduce amendment BH's degenerate-cell enumeration EXACTLY.
#
# NO GPU. The displacement plan is pure combinatorics on ROW indices: it
# displaces each concept's positives under the stored schedules and intersects
# the result with the concept's fixed mask. No SAE, no activations, no average
# precision. That is what makes this a free, exact test of the two things this
# build actually turns on -- masking and the collision rule -- and it is also
# why it can gate the GPU work rather than run alongside it.
#
# The schedules are STORED, drawn once from --null-seed 0 and shared across
# every seed, so there is one right answer. Any other result means the
# implementation's masking or collision handling differs from BG2/BH; report
# the difference, do not adjust the expectation.
#
# Argument 1 optionally overrides the expectation file, which is what the
# "can this validator actually fail?" probe uses (gotcha 13).

set -euo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT

# The f67485d store is the CURRENT axis (712 library / 703 pool, untruncated,
# gate 5) and is the store feature_ap_experimental_av.h5 carries as its
# library_sha -- which is what makes the null and the statistic comparable.
LABELS="$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5"
EXPECTED="${1:-$PROJECT/analysis/degenerate_draws_bh.json}"
OUT="${2:-$PROJECT/analysis/ac4_bh_plan_validation.json}"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

python -m src.analysis.validate_bh_plan \
    --eval-cache "$SCRATCH/cache/eval.h5" \
    --labels "$LABELS" \
    --expected "$EXPECTED" \
    --scored-row-set label_store \
    --stratum experimental \
    --negatives unannotated \
    --n-shifts 20 \
    --null-seed 0 \
    --n-length-bins 20 \
    --out "$OUT"

# Reached ONLY on success: `set -e` aborts the script if the validator raises
# SystemExit(1), and SLURM records FAILED. An `echo "exited $?"` here could
# print nothing but 0 (gotcha 25).
echo "AC4: validator exited 0 -- reproduces BH's enumeration exactly."
