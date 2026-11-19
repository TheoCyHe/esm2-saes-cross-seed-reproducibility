#!/bin/bash
#SBATCH --job-name=ct29_score
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# §CT29-F item 10: CT29-P1-P4 from the stored outputs, with item 7's causes per arm
# and detection/R3 beside (src/analysis/ct29_score.py). From a pinned clean clone:
#   sbatch slurm/ct29_score.sbatch [ARM=cause ...]   (causes established outside the files)

set -euo pipefail
PROJECT=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
[[ -z "$(git status --porcelain)" ]] || { echo "checkout dirty before the run" >&2; exit 1; }
echo "checkout $(git rev-parse HEAD) host=$(hostname)"
OUT="$PROJECT/analysis/ct/ct29_score_ct.json"
python -m src.analysis.ct29_score --ct-dir "$PROJECT/analysis/ct" --cs-dir "$PROJECT/analysis/cs" \
    --unmeasurable "$@" --out "$OUT"
sha256sum "$OUT"
[[ -z "$(git status --porcelain)" ]] || { echo "checkout dirty on exit" >&2; exit 1; }
echo "checkout at exit $(git rev-parse HEAD), clean"
