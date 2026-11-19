#!/bin/bash
#SBATCH --job-name=sae_null_ac1
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# GPU AUDIT (2026-08-10): THE DEVICE MUST STAY. This compares a freshly
# computed null against the STORED `null_pilot_k.h5`, which was produced on
# rtx_5000_ada, and the acceptance criterion is a bit/tolerance comparison
# against it. Under amendment BC7 (CLAUDE.md, Cluster), moving this to CPU
# would not loosen the test -- it would make the
# test measure the device instead of the change.

# AC1 -- the pooled arm's regression test after the BG/BH changes.
#
# THE DECLARED SUBSET, stated here BEFORE the run rather than chosen after it.
# Three of the twenty seeds, all 20 shifts, both arms, the full 707-concept
# pool. That is 3 x 20 x 707 x 2 = 84,840 concept-evaluations and
# 3 x 20 x 707 x 10,240 x 2 = 868,761,600 raw float32 values compared
#
# CORRECTED 2026-08-05: this read 869,022,720, which is 261,120 too high and
# does not multiply out. 707 x 10,240 = 7,239,680; x 20 shifts = 144,793,600;
# x 3 seeds = 434,380,800 per arm -- which is what the comparator reported --
# x 2 arms = 868,761,600. The wrong figure also contradicted this header's own
# concept-evaluation count, since 84,840 x 10,240 = 868,761,600, and it is not
# divisible by 10,240 (869,022,720 / 10,240 = 84,865.5), so it could not be a
# count of values on a 10,240-wide axis at all. Nothing rested on it -- the
# measurement used the correct per-arm value -- but it is a product quoted in a
# committed header that does not multiply out, which is the class of error
# gotcha 34's neighbour was filed about. THE DECLARATION'S SUBSTANCE IS
# UNTOUCHED: the same three seeds, all 20 shifts, both arms, the full pool.
# elementwise against the baseline.
#
# WHY A SUBSET IS PROPORTIONATE HERE. The pooled path's per-seed computation is
# identical in form across seeds -- the schedules are shared, the concept set
# is shared, and the only per-seed input is the SAE's own activations -- so a
# change in the arithmetic cannot be seed-specific. The full 20-seed run costs
# ~2h19m; three seeds cost ~21 minutes of GPU and compare 869M raw values.
#
# WHY THESE THREE. `CKPTS[@]:0:3` takes the first three entries of the SAME
# LEXICOGRAPHIC GLOB the baseline used, which is seed_0, seed_1, seed_10 --
# NOT seeds 0, 1, 2 (CODEBASE Section 7.3, gotcha 29). That is deliberate: the
# baseline's seed axis is in lexicographic order, so the regression must be
# too, and the comparator compares the `seeds` dataset itself so a
# disagreement in that order is caught rather than assumed away.
#
# WHY THE 9bfa94b STORE AND NOT THE COMPLIANT ONE. The baselines were built on
# the 716-library / 707-pool axis. Passing the current f67485d store would
# guarantee a mismatch that is not a regression (brief B Section 3). The
# preflight gate below therefore also runs against feature_ap_main.h5, the
# 716-concept artifact -- the pairing this axis requires.
#
# BOTH BASELINES ARE COMPARED. The reductions are MAXIMA and cannot detect a
# change beneath them; the raw file is the sensitive channel.

set -euo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT
LABELS="$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_9bfa94b.h5"

OUT_DIR="$SCRATCH/ac1_regression"
NEW_RED="$OUT_DIR/null_pilot_ac1_new.h5"
NEW_RAW="$OUT_DIR/null_gain_raw_ac1_new.h5"
REF_RED="$PROJECT/analysis/null_pilot_k.h5"
REF_RAW="$PROJECT/analysis/null_gain_raw_k.h5"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
mkdir -p "$OUT_DIR"

CKPTS=("$PROJECT"/sae_ckpts/seed_*)
if [[ ${#CKPTS[@]} -ne 20 ]]; then
    echo "FATAL: expected exactly 20 checkpoints, found ${#CKPTS[@]}" >&2
    exit 1
fi
SUBSET=("${CKPTS[@]:0:3}")
echo "declared subset: ${SUBSET[*]}"

python -m src.analysis.preflight --labels "$LABELS" \
    --ap "$PROJECT/analysis/feature_ap_main.h5"

python -m src.analysis.null_pilot \
    --config configs/mvp.yaml \
    --ckpts "${SUBSET[@]}" \
    --eval-cache "$SCRATCH/cache/eval.h5" \
    --labels "$LABELS" \
    --scored-row-set label_store \
    --stratum pooled \
    --negatives all_rows \
    --null-type both \
    --n-length-bins 20 \
    --n-shifts 20 \
    --null-seed 0 \
    --cache-dir "$SCRATCH/row_index_cache" \
    --exclude-pooled-parents \
    --out "$NEW_RED" \
    --raw-out "$NEW_RAW"

# BOTH COMPARISONS ALWAYS RUN, and the exit decision is taken at the END.
#
# The first version aborted on `set -e` the moment the reductions differed, so
# the RAW comparison -- the SENSITIVE channel, the one a maximum cannot
# detect changes beneath -- never ran at all. A gate that stops at the first
# failure reports the least informative half of the evidence and hides the
# half that was the reason for having two baselines.
rc_red=0
rc_raw=0

echo "=============== AC1(a): the REDUCTIONS (primary baseline) ==============="
python -m src.analysis.compare_null_payload \
    --new "$NEW_RED" --reference "$REF_RED" \
    --label "reductions vs null_pilot_k.h5" \
    --datasets null_g_a null_g_b null_cmax_a null_cmax_b fire_rate seeds \
               concepts shift_schedule dest_schedule length_bins \
    --seed-axis-datasets null_g_a null_g_b null_cmax_a null_cmax_b fire_rate seeds \
    --out "$PROJECT/analysis/ac1_reductions_diff.json" || rc_red=$?

echo "=============== AC1(b): the RAW array (sensitive baseline) =============="
if [[ -f "$REF_RAW" ]]; then
    python -m src.analysis.compare_null_payload \
        --new "$NEW_RAW" --reference "$REF_RAW" \
        --label "raw vs null_gain_raw_k.h5" \
        --datasets gain_a gain_b concepts \
        --seed-axis-datasets gain_a gain_b \
        --out "$PROJECT/analysis/ac1_raw_diff.json" || rc_raw=$?
else
    # An ABSENT baseline is reported absent, never treated as agreement (I3).
    echo "AC1(b) NOT RUN: $REF_RAW is absent (purged?). Fall back to the"
    echo "digests in null_baseline_a2.json -- and record that the sensitive"
    echo "channel was NOT exercised, rather than reporting AC1 as passed."
    rc_raw=2
fi

echo "=============== AC1 summary ==============="
echo "  reductions comparison exit: $rc_red"
echo "  raw comparison exit:        $rc_raw"
if [[ "$rc_red" -eq 0 && "$rc_raw" -eq 0 ]]; then
    echo "AC1: payload BIT-IDENTICAL on both baselines."
    exit 0
fi
echo "AC1: at least one channel differs. The per-dataset diffs are in"
echo "     ac1_reductions_diff.json and ac1_raw_diff.json -- read those"
echo "     rather than this line."
exit 1
