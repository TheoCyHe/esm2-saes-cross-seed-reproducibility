#!/bin/bash
#SBATCH --job-name=sae_null_determ
#SBATCH --partition=gpu_devel
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
# GPU AUDIT (2026-08-10): THE DEVICE MUST STAY, BECAUSE THE DEVICE IS THE
# SUBJECT. This probe measures RUN-TO-RUN non-determinism on the SAME device
# class (gotcha 35), and the mechanism it found is CUDA `index_add_`'s atomic
# accumulation order inside `sparse_ap_index`. On CPU that mechanism does not
# exist, so a CPU run would answer a different question. The AP arithmetic
# still routes to CPU here, so utilisation is low; that is the price of
# measuring the CUDA path.

# IS THE POOLED NULL BIT-REPRODUCIBLE AT ALL?
#
# AC1 compared the post-BG/BH pooled arm against `null_pilot_k.h5` and found
# the two gain arrays differing in up to 72% of elements at a maximum absolute
# difference of 6.55e-15 -- while `shift_schedule`, `dest_schedule`,
# `length_bins` and `fire_rate` were BIT-IDENTICAL. Structure preserved,
# last-bits moving: the signature of a change in the ORDER of floating-point
# accumulation, not of the mathematics.
#
# `git diff` shows `sparse_ap_index.py`, `row_index_cache.py` and
# `feature_ap.py` are UNTOUCHED by the BG/BH change, and the pooled inner loop
# has no removed lines. So the arithmetic is byte-identical and the difference
# cannot be attributed to the change by reading. IT HAS TO BE MEASURED --
# this pipeline's characteristic failure is a confident wrong answer, and
# "it's probably just float noise" is exactly that shape.
#
# THREE RUNS, and each comparison answers a different question:
#
#   A vs B   -- the SAME code, twice, same node class, same inputs. If these
#               differ, the null is not bit-reproducible run to run and NO
#               bit-identity criterion on it can ever pass, whoever wrote the
#               code. This is the decisive one.
#   C vs REF -- the PRE-CHANGE code (3a1034e) against the stored baseline. If
#               this differs too, the baseline is not reproducible BY ITS OWN
#               CODE, and the AC1 criterion was unachievable rather than
#               failed.
#   A vs C   -- current against pre-change. Whatever this shows must be no
#               larger than A vs B, or the change contributes something beyond
#               the noise floor.
#
# A vs B is what makes this a control rather than a story: if A and B come out
# BIT-IDENTICAL, then the run IS deterministic, the noise explanation is dead,
# and the difference against the baseline is a real attribution problem to
# chase. The probe can produce either answer (gotcha 25).

set -uo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT
LABELS="$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_9bfa94b.h5"
OUT="$SCRATCH/determinism_probe"
REF_RED="$PROJECT/analysis/null_pilot_k.h5"
REF_RAW="$PROJECT/analysis/null_gain_raw_k.h5"
WT="$SCRATCH/worktree_3a1034e"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
mkdir -p "$OUT"

CKPTS=("$PROJECT"/sae_ckpts/seed_*)
[[ ${#CKPTS[@]} -eq 20 ]] || { echo "FATAL: ${#CKPTS[@]} checkpoints, expected 20" >&2; exit 1; }
SUBSET=("${CKPTS[@]:0:3}")
echo "declared subset: ${SUBSET[*]}"
nvidia-smi --query-gpu=name --format=csv,noheader

run_null () {   # $1 = tag, $2 = repo root
    ( cd "$2" && PYTHONPATH="$2" python -m src.analysis.null_pilot \
        --config configs/mvp.yaml \
        --ckpts "${SUBSET[@]}" \
        --eval-cache "$SCRATCH/cache/eval.h5" \
        --labels "$LABELS" \
        --scored-row-set label_store \
        --null-type both --n-length-bins 20 --n-shifts 20 --null-seed 0 \
        --cache-dir "$SCRATCH/row_index_cache" \
        --exclude-pooled-parents \
        --out "$OUT/red_$1.h5" --raw-out "$OUT/raw_$1.h5" ) > "$OUT/log_$1.txt" 2>&1
    echo "run $1 exited $?"
    tail -3 "$OUT/log_$1.txt"
}

# ---- the PRE-CHANGE worktree. Outside the checkout, so `git status` there
# ---- stays clean and no concurrent job's provenance is disturbed.
if [[ ! -d "$WT" ]]; then
    git worktree add --detach "$WT" 3a1034e
fi
( cd "$WT" && git log --oneline -1 )

echo "=================== run A (current code) ==================="
run_null A "$SLURM_SUBMIT_DIR"
echo "=================== run B (current code, again) ==================="
run_null B "$SLURM_SUBMIT_DIR"
echo "=================== run C (pre-change, 3a1034e) ==================="
run_null C "$WT"

cmp_pair () {   # $1 new, $2 ref, $3 label, $4 out
    python -m src.analysis.compare_null_payload \
        --new "$1" --reference "$2" --label "$3" \
        --datasets null_g_a null_g_b null_cmax_a null_cmax_b fire_rate \
                   seeds concepts shift_schedule dest_schedule length_bins \
        --seed-axis-datasets null_g_a null_g_b null_cmax_a null_cmax_b \
                             fire_rate seeds \
        --out "$4"
    echo "  -> exit $?"
}
cmp_raw () {
    python -m src.analysis.compare_null_payload \
        --new "$1" --reference "$2" --label "$3" \
        --datasets gain_a gain_b concepts \
        --seed-axis-datasets gain_a gain_b \
        --out "$4"
    echo "  -> exit $?"
}

echo "############ THE CONTROL: A vs B, identical code, twice ############"
cmp_pair "$OUT/red_A.h5" "$OUT/red_B.h5" "A vs B reductions" "$PROJECT/analysis/determ_AB_red.json"
cmp_raw  "$OUT/raw_A.h5" "$OUT/raw_B.h5" "A vs B raw"        "$PROJECT/analysis/determ_AB_raw.json"

echo "############ C (pre-change) vs the STORED BASELINE ############"
cmp_pair "$OUT/red_C.h5" "$REF_RED" "C(pre-change) vs null_pilot_k.h5" "$PROJECT/analysis/determ_C_vs_ref_red.json"
[[ -f "$REF_RAW" ]] && cmp_raw "$OUT/raw_C.h5" "$REF_RAW" "C(pre-change) vs raw baseline" "$PROJECT/analysis/determ_C_vs_ref_raw.json"

echo "############ A (current) vs C (pre-change) ############"
cmp_pair "$OUT/red_A.h5" "$OUT/red_C.h5" "A vs C reductions" "$PROJECT/analysis/determ_AC_red.json"
cmp_raw  "$OUT/raw_A.h5" "$OUT/raw_C.h5" "A vs C raw"        "$PROJECT/analysis/determ_AC_raw.json"

echo "############ A (current) vs the STORED BASELINE ############"
cmp_pair "$OUT/red_A.h5" "$REF_RED" "A vs null_pilot_k.h5" "$PROJECT/analysis/determ_A_vs_ref_red.json"
[[ -f "$REF_RAW" ]] && cmp_raw "$OUT/raw_A.h5" "$REF_RAW" "A vs raw baseline" "$PROJECT/analysis/determ_A_vs_ref_raw.json"

echo "probe complete -- read the six verdicts above; this script does NOT"
echo "collapse them into one, because they answer different questions."
git status --porcelain
