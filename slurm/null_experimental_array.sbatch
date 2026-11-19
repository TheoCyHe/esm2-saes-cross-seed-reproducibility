#!/bin/bash
#SBATCH --job-name=sae_null_exp
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; torch cu126 has no sm_120 (Blackwell) kernels
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=36:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x_%A_%a.out
# GPU AUDIT (2026-08-10): THE DEVICE STAYS, AND THIS IS THE SCRIPT THAT GOT
# FOUR TASKS RECLAIMED. Measured on job 21466226: the gather is 250.9 s of a
# 33,536 s task -- 0.75% of elapsed, 0.19% of the 36 h requested -- because
# 511,114,400 non-zeros routes every masked AP evaluation to CPU. It is kept
# unchanged as the record of the GPU attempt, and because its half-written
# output pairs with the GPU-derived `feature_ap_experimental_av.h5`. THE LIVE
# SCRIPT IS `null_experimental_array_cpu.sbatch`, which is what actually
# produced `null_experimental_bg.h5`.

# THE EXPERIMENTAL-STRATUM NULL (PREREGISTRATION amendments BG and BH).
#
# 218 concepts x 20 shifts x 2 arms = 8,720 masked evaluations per seed, one
# task per seed.
#
# WALL TIME, from AC5's MEASUREMENT (job 21438269, rtx_5000_ada, 12 concepts x
# 20 shifts x 2 arms = 480 evaluations at production's own 40-draws-per-concept
# amortisation): **5.933 s per masked evaluation**, against the 6.8-7.2 s
# estimate carried into this build. So
#
#     268 s gather + 8,720 x 5.933 s = 52,004 s = 14.45 h per task,
#     288.9 GPU-hours for the ensemble.
#
# 36 h is requested rather than 24 or 48: it is 2.5x the measurement, so a task
# that runs long is killed with room left to diagnose and resubmit rather than
# at the ceiling. `part_gpu`'s QOS MaxWall is 2-00:00:00 (48 h), NOT the 24 h
# assumed elsewhere -- verified with `sacctmgr show qos`.
#
# --mem=192G is also from the measurement: the pilot peaked at MaxRSS
# 93,658,748 K = 89.3 GiB, and the per-seed buffers grow by only ~340 MB going
# from 12 concepts to 218, so the peak is essentially unchanged.
#
# ONE THING THE PILOT DID NOT SAMPLE, stated rather than left implicit: its 12
# concepts spanned 9 to 14,727 masked rows, and the pool's largest mask is
# 1,986,956 rows. The 19 largest-mask concepts were not timed. The DIRECTION of
# that omission is INFERRED, not measured -- a larger mask removes more
# non-zeros, so the segmented scan those concepts pay is SHORTER -- which makes
# 14.45 h a conservative figure rather than an optimistic one.
#
# WHY AN ARRAY. The masked path forces the SEGMENTED average-precision
# arithmetic (I1b: the sparse path ranks against the full evaluated row set,
# so masked rows would keep ranks they no longer earn). The whole pass is far
# beyond any single job's wall limit. The per-seed activation gather is paid
# per seed either way, so one task per seed costs nothing extra and buys 20x
# wall clock. Merge afterwards with slurm/merge_null_experimental.sbatch.
#
# THE ARRAY INDEX IS THE SEED NUMBER, deliberately -- no glob, so CODEBASE
# Section 7.3's lexicographic sort cannot enter here (gotcha 29). The merge
# re-orders rows to the reference artifact's own seed LABELS and asserts the
# result positionally.
#
# SHARDS GO TO SCRATCH; THE MERGED RAW GOES TO PROJECT. The merged raw is
# ~6.7 GiB and is the input to the FDR pass, which costs hundreds of GPU-hours
# to reproduce -- letting it purge mid-analysis is the failure the storage work
# has been preventing. The per-seed shards are regenerable intermediates and
# stay on scratch.
#
# --raw-out IS MANDATORY HERE. The reductions are maxima over concepts; the
# FDR pass needs the per-(feature, concept) values, which live only in the raw
# array. Omitting it has already forced two re-runs of the pooled null.

set -euo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT

# The f67485d store is the CURRENT axis: 712 library / 703 pool, untruncated,
# gate 5. It is the store feature_ap_experimental_av.h5 carries as its
# library_sha, which is what makes the null and the statistic comparable at
# all. NOT the 9bfa94b default the pooled sbatch uses -- that is the 716-axis
# store, and a run with it would exit 0 and produce a plausible, internally
# consistent array on the wrong concept axis (amendment BF1).
LABELS="${1:-$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5}"
SHARD_DIR="$SCRATCH/null_exp_shards"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

SEED="$SLURM_ARRAY_TASK_ID"
CKPT="$PROJECT/sae_ckpts/seed_$SEED"

if [[ ! -f "$CKPT/model.pt" ]]; then
    echo "FATAL: no checkpoint at $CKPT" >&2; exit 1
fi
if [[ ! -f "$LABELS" ]]; then
    echo "FATAL: missing label store $LABELS" >&2; exit 1
fi

python -m src.analysis.preflight --labels "$LABELS"

# The label store must be the untruncated, gate-5 one, asserted on IDENTITY
# (gotcha 15).
python - "$LABELS" <<'PYCHECK'
import sys, h5py
with h5py.File(sys.argv[1], "r") as hf:
    pol = hf.attrs.get("truncation_policy")
    gate = hf.attrs.get("min_protein_count")
    n = hf.attrs.get("n_concepts")
    sha = hf.attrs.get("git_sha")
sha = sha.decode() if isinstance(sha, bytes) else sha
if pol != "untruncated" or int(gate or 0) < 5:
    sys.exit(f"FATAL: truncation_policy={pol} min_protein_count={gate}; the "
             "experimental null requires the untruncated population (G2) and "
             "the >=5 protein gate (G3). Refusing.")
if not str(sha).startswith("f67485d"):
    sys.exit(f"FATAL: label store git_sha={sha}, expected f67485d... -- this "
             "is the 712/703 axis every current consumer reads. Refusing.")
print(f"label store OK: truncation_policy={pol} min_protein_count={gate} "
      f"n_concepts={n} git_sha={sha}")
PYCHECK

mkdir -p "$SHARD_DIR"

python -m src.analysis.null_pilot \
    --config configs/mvp.yaml \
    --ckpts "$CKPT" \
    --eval-cache "$SCRATCH/cache/eval.h5" \
    --labels "$LABELS" \
    --scored-row-set label_store \
    --stratum experimental \
    --negatives unannotated \
    --null-type both \
    --n-length-bins 20 \
    --n-shifts 20 \
    --null-seed 0 \
    --cache-dir "$SCRATCH/row_index_cache" \
    --exclude-pooled-parents \
    --ap-device auto \
    --out "$SHARD_DIR/red_seed_${SEED}.h5" \
    --raw-out "$SHARD_DIR/raw_seed_${SEED}.h5"

echo "shard for seed $SEED written to $SHARD_DIR/{red,raw}_seed_${SEED}.h5"
