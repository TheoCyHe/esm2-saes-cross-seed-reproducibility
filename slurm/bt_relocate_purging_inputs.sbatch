#!/bin/bash
#SBATCH --job-name=sae_bt_relocate
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

# PREREGISTRATION amendment BT, part A3 -- relocate the artifacts that fail the
# committed purge criterion, and re-point the canary target as a PATH CHANGE.
#
# THE CRITERION IS NOT INVENTED HERE (BT-A1). It is the one this project has
# used twice: is the artifact byte-identically regenerable, or does a committed
# result descend from it? Nulls fail it -- `index_add_` on CUDA accumulates
# atomically and `use_deterministic_algorithms` is enabled nowhere in this repo,
# so "regenerable" for a null means a different file with the same
# distribution, not the same file (gotcha 35).
#
# COPY, VERIFY, THEN LEAVE THE ORIGINAL ALONE. Nothing is deleted and nothing is
# regenerated in place. The scratch copies stay until the 30-day purge takes
# them on its own schedule (~2026-08-28 for null_raw, ~2026-08-30 for the two
# analysis files); from then there is one copy of each and it is on project.
# `mv` across filesystems would be a copy followed by an unlink, i.e. a delete,
# which BT-E forbids.
#
# THE ORDER IS THE SUBSTANCE, exactly as at BS-C2: verify the COMMITTED canary
# set FIRST, then copy, then capture, then verify the full set. A capture that
# ran before the pre-verify could not distinguish "the relocation is clean" from
# "something else moved at the same time".
#
# NO GPU. Copying and hashing are I/O; nothing here gathers activations.

set -euo pipefail

PROJECT=$PROJECT
SCRATCH=$SCRATCH
ROOT="$PROJECT/analysis"
MANIFEST=canary/baseline.json

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

if [[ -n "$(git status --porcelain)" ]]; then
    echo "REFUSING: cluster checkout is dirty" >&2
    git status --porcelain >&2
    exit 1
fi
echo "checkout clean at $(git rev-parse HEAD)"

# --- the four sources, and where each goes ---------------------------------
SRC=(
    "$SCRATCH/null_raw/null_gain_raw_compliant.h5"
    "$SCRATCH/null_raw/null_gain_raw_k.h5"
    "$SCRATCH/analysis/gtilde_compliant_b_ratio_smooth_fixfp.h5"
    "$SCRATCH/analysis/smooth_mu_b_fixfp.h5"
)

echo
echo "=== STEP 0: the destination is NOT itself on a clock, and it has room ==="
# getquota, per BT-0b and CLAUDE.md's Cluster section. The `Purged` column is the check.
getquota 2>&1 | sed -n '1,12p'
PURGED=$(getquota 2>&1 | awk '/^project +GRP/{print $NF}' | head -1)
if [[ "$PURGED" != "No" ]]; then
    echo "REFUSING: project reports Purged='$PURGED', not 'No'. The destination" >&2
    echo "is on a clock too and relocating there would move the problem." >&2
    exit 1
fi
echo "project Purged column reads '$PURGED' -- not a purging fileset"

NEED=0
for f in "${SRC[@]}"; do
    if [[ ! -f "$f" ]]; then echo "MISSING SOURCE: $f" >&2; exit 1; fi
    NEED=$(( NEED + $(stat -c %s "$f") ))
done
USED=$(getquota 2>&1 | awk '/^project +GRP/{print $3}' | head -1)
QUOTA=$(getquota 2>&1 | awk '/^project +GRP/{print $4}' | head -1)
FREE=$(( QUOTA - USED ))
echo "to write: $NEED bytes ($(( NEED / 1073741824 )) GiB); project free: ${FREE} GiB of ${QUOTA} GiB"
if (( NEED / 1073741824 + 20 > FREE )); then
    echo "REFUSING: insufficient headroom (need $(( NEED / 1073741824 )) GiB" >&2
    echo "plus a 20 GiB margin, have ${FREE} GiB free). BT-A3 says stop and" >&2
    echo "report the numbers rather than filling a shared filesystem." >&2
    exit 1
fi

echo
echo "=== STEP 1: verify the COMMITTED canary set, BEFORE anything is copied ==="
git show HEAD:canary/baseline.json > /tmp/bt_baseline_committed.json
python -m src.analysis.canary --root "$ROOT" \
    --scratch-root "$SCRATCH/null_raw" \
    --manifest /tmp/bt_baseline_committed.json

echo
echo "=== STEP 2: copy, then prove identity by content hash on BOTH sides ==="
for f in "${SRC[@]}"; do
    b=$(basename "$f")
    dst="$ROOT/$b"
    if [[ -e "$dst" ]]; then
        echo "REFUSING: $dst already exists -- not overwriting" >&2
        exit 1
    fi
    echo "--- $b ---"
    t0=$(date +%s)
    cp -p "$f" "$dst"
    t1=$(date +%s)
    SRC_SHA=$(sha256sum "$f" | cut -d' ' -f1)
    DST_SHA=$(sha256sum "$dst" | cut -d' ' -f1)
    SRC_B=$(stat -c %s "$f")
    DST_B=$(stat -c %s "$dst")
    echo "  source : $f"
    echo "           sha256 $SRC_SHA  bytes $SRC_B"
    echo "  dest   : $dst"
    echo "           sha256 $DST_SHA  bytes $DST_B"
    echo "  copy took $(( t1 - t0 )) s"
    if [[ "$SRC_SHA" != "$DST_SHA" || "$SRC_B" != "$DST_B" ]]; then
        echo "  HASH OR SIZE MISMATCH -- the copy is not the same file" >&2
        exit 1
    fi
    echo "  IDENTICAL: same bytes, path changed only"
done

echo
echo "=== STEP 3: capture -- the canary target's PATH moves, the hash does not ==="
# --allow-relocation is required ON TOP of --assert-addition-only (BT-A3): the
# move has to be declared here, at the call site, and a target whose sha256
# moved as well is still refused whatever this flag says.
python -m src.analysis.canary --root "$ROOT" \
    --manifest "$MANIFEST" --capture \
    --assert-addition-only --allow-relocation

echo
echo "=== STEP 4: verify the FULL set at the new path ==="
python -m src.analysis.canary --root "$ROOT" --manifest "$MANIFEST"

echo
echo "=== STEP 5: the scratch originals are STILL THERE -- nothing was deleted ==="
for f in "${SRC[@]}"; do
    if [[ -f "$f" ]]; then
        echo "  PRESENT $(stat -c '%s %y' "$f") $f"
    else
        echo "  GONE $f -- THIS SHOULD NOT HAPPEN; nothing here deletes" >&2
        exit 1
    fi
done

echo
echo "ALL FIVE STEPS COMPLETED"
