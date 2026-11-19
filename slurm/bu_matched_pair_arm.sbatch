#!/bin/bash
#SBATCH --job-name=sae_bu_matched
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# PREREGISTRATION amendment BU, part B -- COMPONENT 5: the firing-rate-matched
# paired arm, at the UNRESTRICTED threshold (T = 1) only. The last build
# component.
#
# NO GPU, AND THAT IS A CHOICE WITH A REASON. Nothing here gathers activations
# or recomputes an AP: `fire_counts` is read from the AP artifact, `r_f` is read
# from the MNN counts, and the treated/control sets are component 3's stored
# output. For gotcha 36: every input here is READ rather than recomputed, so
# no gather occurs on either side.
#
# NO ARRAY. One job.
#
# THE COST PILOT RUNS FIRST AND GATES THE FULL RUN (BU-B5). Two seeds, then a
# projection, then the full pass only if the projection is under two hours. The
# projection is NOT linear in seeds: a fixed start-up cost is paid once and is
# SUBTRACTED before scaling. The constant is estimated CONSERVATIVELY -- it is
# clamped to at most half the pilot, so it can never exceed the pilot itself and
# turn the gate into a formality that passes for the wrong reason.

set -euo pipefail

PROJECT=$PROJECT
A="$PROJECT/analysis"

APE="$A/feature_ap_experimental_av_cpu.h5"
C3="$A/classification_2x2_bq.h5"
RF="$A/rf_counts_theta070_n39.h5"
CS="$A/corrector_scan.json"

OUT="$A/matched_pair_arm_bu.h5"
REP="$A/matched_pair_arm_bu.json"
POUT="$A/adhoc/bu/matched_pair_arm_pilot.h5"
PREP="$A/adhoc/bu/matched_pair_arm_pilot.json"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

if [[ -n "$(git status --porcelain)" ]]; then
    echo "REFUSING: cluster checkout is dirty" >&2
    git status --porcelain >&2
    exit 1
fi
echo "checkout clean at $(git rev-parse HEAD)"

for f in "$APE" "$C3" "$RF" "$CS"; do
    if [[ ! -f "$f" ]]; then echo "MISSING INPUT: $f" >&2; exit 1; fi
done
for f in "$OUT" "$REP"; do
    if [[ -e "$f" ]]; then echo "REFUSING to overwrite $f" >&2; exit 1; fi
done
mkdir -p "$A/adhoc/bu"

echo
echo "=== COST PILOT: two seeds (BU-B5) ==="
T0=$(date +%s)
python -m src.analysis.matched_pair_arm \
    --ap-experimental "$APE" \
    --classification-2x2 "$C3" \
    --rf-counts "$RF" \
    --corrector-scan "$CS" \
    --out "$POUT" --report "$PREP" --seeds 2
T1=$(date +%s)
PILOT=$(( T1 - T0 ))
echo "PILOT ELAPSED: ${PILOT} s for 2 of 20 seeds"

# THE PROJECTION, AND ITS ASSUMPTIONS STATED RATHER THAN IMPLIED:
#   * a fixed START-UP cost is paid once -- interpreter start, scipy and torch
#     imports, and the four whole-artifact reads, none of which is per seed.
#     It is estimated as HALF THE PILOT, capped at 30 s. Estimating it as a
#     fixed constant is what made a prior projection pass for the wrong reason:
#     the assumed constant exceeded the pilot, so the per-seed term went
#     negative and the gate could not fail.
#   * the remaining cost is assumed LINEAR in seeds. Each seed does its own
#     O(n_treated x n_control) distance pass and nothing about it depends on
#     how many other seeds there are.
#   * NFS throughput is assumed no worse under the full run than under the
#     pilot. Untested; the queue is empty and this is a single job.
HALF=$(( PILOT / 2 ))
CONST=$(( HALF < 30 ? HALF : 30 ))
REST=$(( PILOT - CONST ))
PER_SEED=$(( REST / 2 ))
PROJ=$(( CONST + PER_SEED * 20 ))
echo "PROJECTION: constant ${CONST} s (<= half the ${PILOT} s pilot) + 20 x ${PER_SEED} s = ${PROJ} s"
if (( PROJ > 7200 )); then
    echo "STOPPING: projected ${PROJ} s exceeds the two-hour gate (BU-B5)." >&2
    echo "Reporting the pilot's numbers and not proceeding." >&2
    exit 1
fi
echo "projection is under the two-hour gate -- proceeding"

echo
echo "=== COMPONENT 5: all 20 seeds, both arms, with and without correctors ==="
T2=$(date +%s)
python -m src.analysis.matched_pair_arm \
    --ap-experimental "$APE" \
    --classification-2x2 "$C3" \
    --rf-counts "$RF" \
    --corrector-scan "$CS" \
    --out "$OUT" --report "$REP"
T3=$(date +%s)
echo "FULL RUN ELAPSED: $(( T3 - T2 )) s (projected ${PROJ} s)"

echo
echo "=== canary: nothing else moved (criterion 1) ==="
python -m src.analysis.canary --root "$A" --manifest canary/baseline.json

echo
echo "=== checkout clean ON EXIT as well as before submit (gotcha 31) ==="
git status --porcelain
echo "PORCELAIN_END"

echo
echo "ALL STEPS COMPLETED"
