#!/bin/bash
#SBATCH --job-name=sae_orphan_screen
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; torch cu126 has no sm_120 (Blackwell) kernels
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
# GPU AUDIT (2026-08-10): THE DEVICE STAYS. `full_eval` is 17,739,363 rows x
# k=32 = 567,659,616 non-zeros, above NNZ_GPU_SAFE, so the AP arithmetic runs
# on CPU and the GPU does the gather only -- ~4,476 s of 14,397 s (~31%,
# INFERRED from 20 seeds x a ~224 s gather). Not changed: this screen exists to
# be COMPARED against `feature_ap_main.h5`'s sub-typed partition, and that
# comparison is exactly the cross-device join gotcha 36 forbids.

# SUPERSEDED as "the wide screen" by main_ap.sbatch (PREREGISTRATION amendment
# E / Issue 1, 2026-07-27): what this script computes -- pooled, full eval,
# un-subtyped 97-concept axis -- is no longer the widest available screen once
# concept sub-typing exists, and is no longer what D7's ORPHAN definition
# should be measured against. Kept and still run for the ROBUSTNESS-layer
# comparison (does D7's aligned/orphan/excluded partition look different against
# the un-subtyped vs sub-typed concept library?), not as the primary wide screen.
#
# Original PREREGISTRATION amendment D7 rationale, unchanged below: the WIDE
# screen a feature must fail to be called an ORPHAN (not just "did not align to
# the 18 primary concepts"). Runs the POOLED stratum ONLY (all evidence tiers
# combined -- the broadest population) against the FULL eval row subset (not
# the A1 shared-experimental restriction), with no common-concept-set
# restriction, so as many of the 97 amendment-B-surviving concepts as possible
# are screened. This is also the SECONDARY full-eval AP run CLAUDE.md flagged
# as an unrun one-arg rerun -- same job, dual purpose.
#
# `--strata pooled` alone (not `pooled experimental propagated`) is
# deliberate: requesting multiple strata would trigger feature_ap.py's
# common-concept-set restriction, which is exactly what D7 must NOT apply here.

set -euo pipefail

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
    --strata pooled \
    --scored-row-set full_eval \
    --negatives all_rows \
    --out "$PROJECT/analysis/feature_ap_all_pooled_orphan_screen.h5"
