#!/bin/bash
#SBATCH --job-name=release_ext
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# Amendment CT §CT18-F: one family of the extended release bundle, written to
# scratch ($OUT), its gates run on the written files. CPU only: it reads stored
# artifacts and gathers nothing. Nothing is copied to $PROJECT here; that is
# `copy`, run only after every family's gates passed.
#
#   sbatch slurm/release_derived_ext.sbatch arm <w2560|...|k128_w10240>
#   sbatch slurm/release_derived_ext.sbatch anchors
#   sbatch slurm/release_derived_ext.sbatch replay   (after anchors: reads sim_decoder_n39)
#   sbatch slurm/release_derived_ext.sbatch phase2b  (after replay: its owners are re-checked)
#   COPY=1 sbatch slurm/release_derived_ext.sbatch manifest   (last; without COPY it only writes the manifest)
#   UPDATE_META=1 COPY=1 sbatch ... manifest   (a traced rebuild of a shipped bundle: replaces
#     only meta_ext.json and its manifest row, refusing unless every rebuilt file is
#     byte-identical to the shipped one; §CT18-D2's owed items)
set -euo pipefail
S=$SCRATCH
PROJECT=$PROJECT
OUT="${OUT:-$S/release_derived_ext}"
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
[[ -z "$(git status --porcelain)" ]] || { echo "REFUSING: the checkout is dirty before the run" >&2; exit 1; }
echo "checkout $(git rev-parse HEAD) step=$* host=$(hostname) out=$OUT"
# the owners' fits ran single-threaded (identity.omp_num_threads = 1)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT"
case "${1:?arm|anchors}" in
  arm)     python -m src.analysis.release_derived_ext arm "${2:?arm name}" --out-dir "$OUT" ;;
  anchors) python -m src.analysis.release_derived_ext anchors --out-dir "$OUT" \
               --bundle "$PROJECT/release/derived" ;;
  replay)  python -m src.analysis.release_derived_ext replay --out-dir "$OUT" \
               --bundle "$PROJECT/release/derived" ;;
  phase2b) python -m src.analysis.release_derived_ext phase2b --out-dir "$OUT" \
               --bundle "$PROJECT/release/derived" ;;
  manifest) python -m src.analysis.release_derived_ext manifest --out-dir "$OUT" \
               --project-bundle "$PROJECT/release/derived" ${COPY:+--copy} \
               ${UPDATE_META:+--update-meta} ;;
  *) echo "unknown step $1" >&2; exit 1 ;;
esac
rc=$?
[[ -z "$(git status --porcelain)" ]] || { echo "the checkout is dirty on exit" >&2; exit 1; }
exit $rc
