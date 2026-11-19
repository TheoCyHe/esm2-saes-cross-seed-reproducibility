#!/bin/bash
#SBATCH --job-name=sae_bt_grid
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

# PREREGISTRATION amendment BT, part C -- COMPONENT 4: the classification
# composition at the four remaining thinness-grid points, plus part B4's two
# matching-pool diagnostics.
#
# NO GPU, AND THAT IS A CHOICE WITH A REASON. Nothing here gathers activations
# or recomputes an AP: the observed AP is read from the artifact, the null is
# read from disk, and the arithmetic is a pooled-exceedance sort plus five BH
# step-ups per (seed, arm). For gotcha 36: both sides here are READ rather
# than recomputed, so no
# gather occurs on either.
#
# THE COST PILOT RUNS FIRST AND GATES THE FULL RUN (BT-C6). Two seeds, then a
# projection, then the full pass only if the projection is under two hours.
# The projection is explicitly NOT linear in seeds -- a prior projection
# overshot by 3.3x because a large constant term was extrapolated as if it
# scaled -- so the constant is estimated and subtracted.

set -euo pipefail

PROJECT=$PROJECT
A="$PROJECT/analysis"

RAW="$A/null_gain_raw_experimental_bg.h5"
RED="$A/null_experimental_bg.h5"
APE="$A/feature_ap_experimental_av_cpu.h5"
BH="$A/classification_bh_experimental_bo.h5"
C3="$A/classification_2x2_bq.h5"

OUT="$A/grid_composition_bt.h5"
REP="$A/grid_composition_bt.json"
POUT="$A/adhoc/bt/grid_composition_pilot.h5"
PREP="$A/adhoc/bt/grid_composition_pilot.json"
MDIAG="$A/match_pool_diagnostics_bt.json"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

if [[ -n "$(git status --porcelain)" ]]; then
    echo "REFUSING: cluster checkout is dirty" >&2
    git status --porcelain >&2
    exit 1
fi
echo "checkout clean at $(git rev-parse HEAD)"

for f in "$RAW" "$RED" "$APE" "$BH" "$C3"; do
    if [[ ! -f "$f" ]]; then echo "MISSING INPUT: $f" >&2; exit 1; fi
done
for f in "$OUT" "$REP" "$MDIAG"; do
    if [[ -e "$f" ]]; then echo "REFUSING to overwrite $f" >&2; exit 1; fi
done
mkdir -p "$A/adhoc/bt"

echo
echo "=== COST PILOT: two seeds, BT-C6 ==="
T0=$(date +%s)
python -m src.analysis.grid_composition \
    --raw-null-experimental "$RAW" \
    --null-experimental-reductions "$RED" \
    --ap-experimental "$APE" \
    --bh-experimental "$BH" \
    --classification-2x2 "$C3" \
    --out "$POUT" --report "$PREP" --seeds 2
T1=$(date +%s)
PILOT=$(( T1 - T0 ))
echo "PILOT ELAPSED: ${PILOT} s for 2 of 20 seeds"

# THE PROJECTION, AND ITS ASSUMPTIONS STATED RATHER THAN IMPLIED:
#   * a fixed START-UP cost is paid once -- imports, the AP artifact read
#     (20 x 10240 x 218 float32), the BH reject arrays, the 2x2 artifact.
#     Estimated at 60 s and SUBTRACTED before scaling, because extrapolating
#     a constant as though it scaled is what made a prior projection overshoot
#     by 3.3x.
#   * the remaining cost is assumed LINEAR in seeds: each seed reads one raw
#     slab per arm and runs five BH step-ups, and neither depends on how many
#     other seeds there are.
#   * NFS throughput is assumed no worse under this job than under the pilot.
#     Untested; the queue is empty and this is a single job.
CONST=60
PER_SEED=$(( (PILOT > CONST ? PILOT - CONST : 1) / 2 ))
PROJ=$(( CONST + PER_SEED * 20 ))
echo "PROJECTION: constant ${CONST} s + 20 x ${PER_SEED} s = ${PROJ} s"
if (( PROJ > 7200 )); then
    echo "STOPPING: projected ${PROJ} s exceeds the two-hour gate (BT-C6)." >&2
    echo "Reporting the pilot's numbers and not proceeding." >&2
    exit 1
fi
echo "projection is under the two-hour gate -- proceeding"

echo
echo "=== COMPONENT 4: all 20 seeds, five grid points ==="
T2=$(date +%s)
python -m src.analysis.grid_composition \
    --raw-null-experimental "$RAW" \
    --null-experimental-reductions "$RED" \
    --ap-experimental "$APE" \
    --bh-experimental "$BH" \
    --classification-2x2 "$C3" \
    --out "$OUT" --report "$REP"
T3=$(date +%s)
echo "FULL RUN ELAPSED: $(( T3 - T2 )) s (projected ${PROJ} s)"

echo
echo "=== BT-B4: the matching-pool diagnostics ==="
python - <<'PY'
import json, h5py, numpy as np
from src.analysis.match_pool_diagnostics import pool_diagnostics
from src.analysis.classification_2x2 import (CELL_ALIGNED_BOTH,
                                             CELL_ALIGNED_EXP, CELL_ORPHAN)
from src.sae.train import git_dirty, git_sha

A = "$PROJECT/analysis/"
with h5py.File(A + "feature_ap_experimental_av_cpu.h5", "r") as hf:
    fire = hf["fire_counts"][()]
    ap_seeds = [s.decode() if isinstance(s, bytes) else str(s)
                for s in hf["seeds"][()]]
    n_rows = int(hf.attrs["n_rows"])
with h5py.File(A + "classification_2x2_bq.h5", "r") as hf:
    c3_seeds = [s.decode() if isinstance(s, bytes) else str(s)
                for s in hf["seeds"][()]]
    cells = {a: hf[f"cell_{a}"][()] for a in ("a", "b")}

# SEED AXIS ON LABELS, NEVER BY POSITION (gotcha 29).
if ap_seeds != c3_seeds:
    raise SystemExit(f"SEED AXIS MISMATCH: ap {ap_seeds} vs 2x2 {c3_seeds}")

out = {"artifact": "bt_match_pool_diagnostics", "governing_amendment": "BT",
       "git_sha": git_sha(), "git_dirty": bool(git_dirty()),
       "caliper_source": "configs/mvp.yaml confound_control.caliper; AN3",
       "threshold_grid_point": 1,
       "note": ("BT-B4. PROPERTIES OF THE MATCHING POOL. These did NOT "
                "condition BT-B3's choices, which are fixed on structural "
                "grounds and were written before these ran."),
       "per_arm": {}}
for arm in ("a", "b"):
    aligned = (cells[arm] == CELL_ALIGNED_BOTH) | (cells[arm] == CELL_ALIGNED_EXP)
    orphan = cells[arm] == CELL_ORPHAN
    out["per_arm"][arm.upper()] = pool_diagnostics(
        fire, n_rows, aligned, orphan, 0.1, ap_seeds)
    a = out["per_arm"][arm.upper()]["across_seeds"]
    print(f"arm {arm.upper()}: pairs={a['denominator_treated_matched_pairs']} "
          f"tie_mean={a['tie_multiplicity_mean']} "
          f"tie_max={a['tie_multiplicity_max']} "
          f"unique_frac={a['frac_treated_with_a_unique_nearest']} "
          f"reuse_mean={a['control_reuse_mean']} "
          f"reuse_max={a['control_reuse_max']}")

with open(A + "match_pool_diagnostics_bt.json", "w") as f:
    json.dump(out, f, indent=2)
print("wrote", A + "match_pool_diagnostics_bt.json")
PY

echo
echo "=== canary: nothing else moved (criterion 1) ==="
python -m src.analysis.canary --root "$A" --manifest canary/baseline.json

echo
echo "=== checkout clean ON EXIT as well as before submit (gotcha 31) ==="
git status --porcelain
echo "PORCELAIN_END"

echo
echo "ALL STEPS COMPLETED"
