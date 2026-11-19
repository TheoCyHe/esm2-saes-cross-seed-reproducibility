#!/bin/bash
#SBATCH --job-name=sae_bs_canary
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# PREREGISTRATION amendment BS, part C -- the seventh input's canary coverage
# and the seed-order sidecar.
#
# THE ORDER IS THE SUBSTANCE (BS-C2), and it is enforced here rather than
# described: verify the COMMITTED set first, then capture with
# --assert-addition-only, then verify the FULL set. A capture that ran before
# the pre-verify could not distinguish "the addition is clean" from "something
# else moved at the same time".
#
# NO GPU. Nothing here gathers activations or recomputes an AP; the sidecar
# reads the raw null's stored float32 and takes a max, which is not a device-
# sensitive operation in gotcha 36's sense (no TopK selection occurs). 32G is
# generous: the sidecar reads ONE SHIFT at a time (29 MB), never a whole slab.

set -euo pipefail

PROJECT=$PROJECT
SCRATCH=$SCRATCH
ROOT="$PROJECT/analysis"
SCRATCH_ROOT="$SCRATCH/null_raw"
MANIFEST=canary/baseline.json
RAW="$PROJECT/analysis/null_gain_raw_compliant.h5"
RED="$ROOT/null_pilot_compliant.h5"
SIDECAR="$ROOT/seed_order_pooled_raw_null.json"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

for f in "$RAW" "$RED"; do
    if [[ ! -f "$f" ]]; then echo "MISSING INPUT: $f" >&2; exit 1; fi
done
if [[ -e "$SIDECAR" ]]; then
    echo "$SIDECAR already exists -- refusing to overwrite" >&2; exit 1
fi

GIT_SHA=$(git rev-parse HEAD)
if [[ -n "$(git status --porcelain)" ]]; then
    echo "REFUSING: cluster checkout is dirty" >&2
    git status --porcelain >&2
    exit 1
fi

echo "=== STEP 1: verify the COMMITTED target set, BEFORE any capture ==="
echo "--- this must pass on the 19 targets as committed at HEAD~ ---"
git show HEAD:canary/baseline.json > /tmp/bs_baseline_committed.json
python -m src.analysis.canary --root "$ROOT" --scratch-root "$SCRATCH_ROOT" \
    --manifest /tmp/bs_baseline_committed.json

echo
echo "=== STEP 2: capture, with addition-only ASSERTED (not merely hoped) ==="
python -m src.analysis.canary --root "$ROOT" --scratch-root "$SCRATCH_ROOT" \
    --manifest "$MANIFEST" --capture --assert-addition-only

echo
echo "=== STEP 3: verify the FULL set, including the addition ==="
python -m src.analysis.canary --root "$ROOT" --scratch-root "$SCRATCH_ROOT" \
    --manifest "$MANIFEST"

echo
echo "=== STEP 4: the sidecar -- seed order recovered from the DATA ==="
python -m src.analysis.seed_order_sidecar \
    --raw "$RAW" --reductions "$RED" --out "$SIDECAR" --git-sha "$GIT_SHA"

echo
echo "ALL FOUR STEPS COMPLETED"
