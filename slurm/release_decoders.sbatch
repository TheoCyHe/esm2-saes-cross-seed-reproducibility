#!/bin/bash
#SBATCH --job-name=release_dec
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
# Amendment CT §CT18-D4 item 2: the decoders of all 140 dictionaries of record, one asset per
# set, written to scratch ($OUT) with their gates; MANIFEST.csv only if every gate passed.
# CPU only: it reads checkpoints and writes HDF5. Submit through scripts/submit_pinned.sh:
#
#   bash scripts/submit_pinned.sh <commit> $SCRATCH/clone_dec_<commit> \
#       src/analysis/release_decoders.py:'^def gate_asset' \
#       -- bash -c 'mkdir -p logs && sbatch slurm/release_decoders.sbatch'
# (logs/ is gitignored, so a fresh clone has none; it is made after the clone checks pass)
set -euo pipefail
S=$SCRATCH
OUT="${OUT:-$S/release_decoders}"
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
[[ -z "$(git status --porcelain)" ]] || { echo "REFUSING: the checkout is dirty before the run" >&2; exit 1; }
echo "checkout $(git rev-parse HEAD) host=$(hostname) out=$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
[[ ! -e "$OUT" ]] || { echo "REFUSING: $OUT exists" >&2; exit 1; }
rc=0
python -m src.analysis.release_decoders --out-dir "$OUT" || rc=$?
[[ -z "$(git status --porcelain)" ]] || { echo "the checkout is dirty on exit" >&2; exit 1; }
exit $rc
