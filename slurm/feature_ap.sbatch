#!/bin/bash
#SBATCH --job-name=sae_feature_ap
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; torch cu126 has no sm_120 (Blackwell) kernels
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
# GPU AUDIT (2026-08-10): THE GPU IS GENUINELY USED AT THIS SCRIPT'S DEFAULT
# SUBSET, so nothing here changes. `feature_ap.py` routes the AP arithmetic to
# CPU only above NNZ_GPU_SAFE = 150,000,000; the default `shared_experimental`
# subset is 1,916,112 rows x k=32 = 61,315,584 non-zeros (job 19617623's own
# log), which is BELOW the threshold, so the gather AND the AP arithmetic both
# run on the GPU. CONDITIONAL, and that is the point: this script takes the
# subset as $1, and `label_store` (511,114,400) or `full_eval` (567,659,616)
# both cross the threshold and turn this into a gather-only GPU job.

# Per-feature Average Precision against residue concepts, for all 20 seeds.
#
# Arg 1 = row subset: shared_experimental (PRIMARY, PREREGISTRATION A1) or all
#         (SECONDARY -- also D7's wide orphan screen; run with pooled-only
#         evidence, see below). Default primary.
# Arg 2 = negative definition: unannotated (amendment C, APPROVED 2026-07-25 --
#         a residue carrying the OTHER tier's label for the same concept is
#         dropped, not scored as a negative) or all_rows (pre-amendment-C,
#         retained as the documented comparator). unannotated is now the
#         PRIMARY analysis (amendment C) so it is the default here; the
#         underlying feature_ap.py module keeps all_rows as ITS default so a
#         bare direct invocation never silently changes meaning.
# All three evidence strata share one activation pass per seed.

set -euo pipefail

SUBSET=${1:-shared_experimental}
NEGATIVES=${2:-unannotated}
SUFFIX=""
[[ "$NEGATIVES" != "all_rows" ]] && SUFFIX="_${NEGATIVES}"

SCRATCH=$SCRATCH
PROJECT=$PROJECT

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

CKPTS=("$PROJECT"/sae_ckpts/seed_*)
if [[ ! -e "${CKPTS[0]}/model.pt" ]]; then
    echo "no ensemble checkpoints in $PROJECT/sae_ckpts" >&2; exit 1
fi
if [[ ! -f "$PROJECT/analysis/residue_labels.h5" ]]; then
    echo "missing residue_labels.h5 -- run build_residue_labels.sbatch first" >&2; exit 1
fi

python -m src.analysis.feature_ap \
    --config configs/mvp.yaml \
    --ckpts "${CKPTS[@]}" \
    --eval-cache "$SCRATCH/cache/eval.h5" \
    --labels "$PROJECT/analysis/residue_labels.h5" \
    --strata pooled experimental propagated \
    --scored-row-set "$SUBSET" \
    --negatives "$NEGATIVES" \
    --out "$PROJECT/analysis/feature_ap_${SUBSET}${SUFFIX}.h5"
