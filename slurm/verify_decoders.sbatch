#!/bin/bash
#SBATCH --job-name=verify_dec
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# Amendment CT section CT18-D11 decision 6: the ten scrubbed decoder files ($DIR) checked against
# the checkpoints themselves, before they replace the unscrubbed set on project space. Every
# slice is compared with its checkpoint's W_dec bit for bit; each file's sha256, md5 and size
# must be its row of $SUMS (decoders_export.tsv). Reads only; writes $OUT_JSON.
# CPU only. Submit through scripts/submit_pinned.sh:
#
#   bash scripts/submit_pinned.sh <commit> $SCRATCH/clone_verdec_<commit> \
#       src/analysis/release_decoders.py:'^def verify' \
#       -- bash -c 'mkdir -p logs && sbatch slurm/verify_decoders.sbatch'
set -euo pipefail
P=$PROJECT
S=$SCRATCH
DIR="${DIR:-$S/release_decoders_export}"
SUMS="${SUMS:-$DIR/decoders_export.tsv}"
OUT_JSON="${OUT_JSON:-$P/analysis/release/gates_decoders_vs_checkpoints_${SLURM_JOB_ID}.json}"
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
[[ -z "$(git status --porcelain)" ]] || { echo "REFUSING: the checkout is dirty before the run" >&2; exit 1; }
echo "checkout $(git rev-parse HEAD) host=$(hostname) dir=$DIR sums=$SUMS out=$OUT_JSON"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$(dirname "$OUT_JSON")"
[[ ! -e "$OUT_JSON" ]] || { echo "REFUSING: $OUT_JSON exists" >&2; exit 1; }
rc=0
python -m src.analysis.release_decoders --verify-dir "$DIR" --sums "$SUMS" --out-json "$OUT_JSON" || rc=$?
[[ -z "$(git status --porcelain)" ]] || { echo "the checkout is dirty on exit" >&2; exit 1; }
exit $rc
