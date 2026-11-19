#!/bin/bash
#SBATCH --job-name=sae_bin_occupancy
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

# AK3 prediction B8 (brief §6b): is hard-bin C1's two-seed tail a
# DISCRETISATION artifact? The prediction and its predicate were committed in
# PREREGISTRATION amendment AK before this ran.
#
# NOT A FIREWALL CROSSING. This reads `fire_rate` and `stratum_mu` only -- no
# AP matrix, no r_f, no label store -- which is why it is its own script rather
# than a third arm of m4_confound.py. A measurement that needs no crossing
# should not inherit one.

set -euo pipefail
SCRATCH=$SCRATCH
PROJECT=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

# THE SAME gtilde THE HARD-BIN C1 ARM READ (adhoc/c1_both_arms.sbatch), so the
# two measurements are about one object. `inner_mode = ratio` is the normaliser
# of record (AC9 / AI); a different gtilde here would be a different step
# function and the comparison against AJ5's rho values would be meaningless.
GTILDE="$PROJECT/analysis/gtilde_compliant_b_ratio.h5"
if [[ ! -f "$GTILDE" ]]; then
    echo "no gtilde artifact at $GTILDE" >&2
    exit 1
fi
# Gotcha 15: the file must be the ratio-inner,
# 712/703, pgate5 object, or the step function is not the one AJ5 measured.
python - "$GTILDE" <<'PY'
import sys
import h5py
want = {"inner_mode": "ratio", "n_concepts": 712, "argmax_pool_n": 703,
        "null_arm": "b", "truncation_policy": "untruncated"}
with h5py.File(sys.argv[1], "r") as h:
    got = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in h.attrs.items()}
bad = []
for k, v in want.items():
    if k not in got:
        bad.append("%s: ABSENT (absent is not matching -- I3)" % k)
    elif str(got[k]) != str(v):
        bad.append("%s: %r, expected %r" % (k, got[k], v))
if bad:
    raise SystemExit("REFUSING -- gtilde identity mismatch:\n  * "
                     + "\n  * ".join(bad))
print("[occ.sbatch] gtilde identity OK: inner_mode=ratio, 712/703, arm b")
PY

python -m src.analysis.bin_occupancy \
    --gtilde "$GTILDE" \
    --null-reductions "$PROJECT/analysis/null_pilot_compliant.h5" \
    --out "$PROJECT/analysis/bin_occupancy_b8.json"
