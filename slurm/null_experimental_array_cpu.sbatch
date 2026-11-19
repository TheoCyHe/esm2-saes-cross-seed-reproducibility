#!/bin/bash
#SBATCH --job-name=sae_null_exp_cpu
#SBATCH --partition=day
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=23:55:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x_%A_%a.out

# THE EXPERIMENTAL-STRATUM NULL, ON CPU. Same computation, no GPU requested.
#
# WHY THIS EXISTS. The GPU version of this array (job 21466226) had four of its
# tasks cancelled for holding a GPU without using it, and the
# cancellations were correct. MEASURED on that run: each task holds an
# rtx_5000_ada for ~12-14 hours and uses it for the ~268 s activation gather
# ONLY -- about 0.6% utilisation -- because the average-precision arithmetic
# runs on CPU by an EXISTING rule. `feature_ap.py`'s threshold (NNZ_GPU_SAFE =
# 150,000,000) sends it there whenever the non-zero count is large, and at this
# scale nnz = 15,972,325 x 32 = 511,114,400. The job log says so on its own
# second line: `ap_impl=average_precision_segmented/v1 ap_device=cpu`.
#
# Confirmed by the resource accounting rather than inferred: task _0 consumed
# 1 day 11:46:12 of CPU time in 8h47m of wall clock (~4 cores saturated) at
# 102 GB RSS, with the GPU idle throughout.
#
# ZERO CODE CHANGE. `null_pilot.py` already picks `device = "cuda" if
# torch.cuda.is_available() else "cpu"`, and `--ap-device auto` already
# resolves to CPU at this nnz regardless of what device is present. Removing
# the GPU request changes WHERE the activation gather runs and nothing else.
#
# WHAT IT COSTS, stated as the one unmeasured quantity. The gather is a forward
# pass over 15,972,325 rows x 1,280 dims plus a 42 GB read from NFS. It took
# 268 s on an rtx_5000_ada, where it was substantially I/O-bound. On CPU expect
# roughly 20-60 minutes -- NOT MEASURED, and it is a small fraction of a ~12.5 h
# task, but it is an estimate and is labelled as one.
#
# WHY MORE CORES THAN THE GPU VERSION ASKED FOR. Measured on job 21466226: two
# tasks co-scheduled on one node ran at 8.91 and 4.69 s per evaluation against
# 5.14 s for a task with a node to itself -- so the loop is CPU-bound and
# contends. 16 cores rather than 12, on a partition with far more nodes.
#
# WALL TIME. Measured at production scale on the GPU array: 4.69-5.14 s per
# evaluation uncontended, i.e. 8,720 x 5.14 = 12.45 h plus the gather. That is
# BETTER than the 5.933 s the AC5 pilot projected.
#
# 23:55 is requested against `part_day`'s MaxTime of 1-00:00:00 (24 h), READ
# from `scontrol show partition day` and confirmed against QOS `part_day`'s
# MaxWall -- not assumed. Margin on the typical rate is ~11 h; on the SLOWEST
# rate observed on the GPU array (~8 s/evaluation, i.e. ~20 h) it is ~4 h.
# A task that ever approaches that belongs on `week` (7-day MaxWall) rather
# than having its work trimmed -- but note `week`'s QOS caps a user at cpu=96,
# so at 16 cores per task it would run only 6 at a time.
#
# CONCURRENCY IS NOT QOS-BOUND HERE: 20 tasks x 16 cores = 320 CPUs against
# part_day's cpu=1000, and 20 x 192G = 3.75T against 15T, so all 20 may run at
# once subject to free nodes.
#
# EVERYTHING ELSE IS IDENTICAL to null_experimental_array.sbatch -- same label
# store, same gates, same shard paths, same merge. The shards this writes are
# interchangeable with that array's, EXCEPT that their activation gather ran on
# a different device class. Prefer NOT to mix: run all 20 here.

set -euo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT

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

# No --gpus anywhere in this file, and none is needed. Recorded in the shard's
# own provenance by null_pilot.py's `device=` line at the top of the log.
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
