#!/bin/bash
#SBATCH --job-name=sae_carrier_conc
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; torch cu126 has no sm_120 (Blackwell) kernels
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

# THE LAST ATTRIBUTION ATTEMPT FOR THE CARRIER EXCESS.
# PREREGISTRATION amendment BB section 3.
#
# The +7.22 pp carrier excess is UNATTRIBUTED. BA10 measured carriers at the
# 50.52nd percentile of their firing-matched peers' homology support, so AZ11's
# survival of double matching establishes nothing either way. The remaining
# pre-computed candidate is the FEATURE-SIDE pair -- activation perplexity and
# concentration -- from AL8b's pre-declared sensitivity arm.
#
# BOTH HALVES RUN, and that is the lesson from the last attempt: an enrichment
# check alone cannot attribute the excess, because enrichment explains it only
# if the covariate also predicts reproducibility.
#
# THE STOPPING RULE IS DECLARED IN THE AMENDMENT AND IN THE MODULE: if the
# enrichment check returns NOT_MATERIALLY_ENRICHED, the excess stays
# unattributed and there are NO further attribution attempts.
#
# ============================================================================
# FOUR ACCEPTANCE CRITERIA, ALL FATAL
# ============================================================================
#   1. the per-anchor MNN counts are BIT-IDENTICAL to rf_counts_theta070_n19,
#      in NUMERIC order, while the carrier join runs in the AP artifact's
#      LEXICOGRAPHIC order (gotcha 29). Under gotcha 27 no shape or total
#      comparison is the criterion.
#   2. the SINGLE arm reproduces carrier_identity_ay_v3.json EXACTLY.
#   3. the SINGLE and DOUBLE arms reproduce carrier_support_ba.json EXACTLY.
#   4. the d_fs enrichment distribution reproduces BA9's stored block EXACTLY.
#      THIS IS THE ONE THAT MATTERS HERE: `support_distribution` gained a
#      `value_name` LABEL so the concentration arm goes through the same
#      function, and criterion 4 is what proves the generalisation changed no
#      arithmetic. Without it, a difference between the d_fs and concentration
#      answers could be the covariate or could be the edit.
#
# ============================================================================
# THE EXPENSIVE PART IS assemble_covariates, NOT THE MNN
# ============================================================================
# It streams each seed's row index protein by protein -- the same path
# design3_join's feature-side arm takes, whose measured profile is ~15 GiB
# peak. The memory request comes from there rather than from a guess. Record
# the real figure from `jobstats` afterwards.
#
# EXPLORATORY (AZ7 / BA5): cannot become primary under any outcome, and no
# mechanism sentence is written for any of it.
#
# THE GLOB HAZARD (AD5). Extension seeds live in sae_ckpts_ext/ and are NOT
# anchors. The guard asserts EXACTLY 20, in shell and again in python via
# --expect-ckpts: a self-consistency check on the count passes at 20 and 40
# alike and is therefore not a guard.

set -euo pipefail

PROJECT=$PROJECT
SCRATCH=$SCRATCH

AP="$PROJECT/analysis/feature_ap_main_compliant.h5"
NULL="$PROJECT/analysis/null_pilot_compliant.h5"
GTILDE="$PROJECT/analysis/gtilde_compliant_b_ratio.h5"
DFS="$PROJECT/analysis/d_fs.h5"
REF="$PROJECT/analysis/rf_counts_theta070_n19.h5"
AYREF="$PROJECT/analysis/carrier_identity_ay_v3.json"
BAREF="$PROJECT/analysis/carrier_support_ba.json"
LABELS="$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5"
ROWIDX="$SCRATCH/row_index_cache"
CKPTS="$PROJECT/sae_ckpts"
EVALCACHE="$SCRATCH/cache/eval.h5"
OUT="$PROJECT/analysis/${1:-carrier_concentration_bb.json}"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

ANCHORS=("$CKPTS"/seed_*)

if [[ ! -e "${ANCHORS[0]}/model.pt" ]]; then
    echo "FATAL: no anchor checkpoints in $CKPTS" >&2; exit 1
fi
if [[ "${#ANCHORS[@]}" -ne 20 ]]; then
    echo "FATAL: expected exactly 20 anchor checkpoints, found ${#ANCHORS[@]}" >&2
    exit 1
fi
for f in "$AP" "$NULL" "$GTILDE" "$DFS" "$REF" "$AYREF" "$BAREF" "$LABELS" \
         "$EVALCACHE"; do
    if [[ ! -f "$f" ]]; then
        echo "FATAL: missing $f" >&2; exit 1
    fi
done
if [[ ! -d "$ROWIDX" ]]; then
    echo "FATAL: missing row index cache $ROWIDX" >&2; exit 1
fi
if [[ -e "$OUT" ]]; then
    echo "FATAL: $OUT exists; every artifact goes to a NEW path" >&2; exit 1
fi

echo "[conc.sbatch] anchors=${#ANCHORS[@]} (expected 20)"
echo "[conc.sbatch] BA reference=$(basename "$BAREF")"
echo "[conc.sbatch] out=$OUT"

python -m src.analysis.carrier_concentration \
    --ap "$AP" --null "$NULL" \
    --gtilde "$GTILDE" --d-fs "$DFS" \
    --ckpts "${ANCHORS[@]}" \
    --ckpts-root "$CKPTS" \
    --expect-ckpts 20 \
    --row-index-cache "$ROWIDX" \
    --eval-cache "$EVALCACHE" \
    --labels "$LABELS" \
    --scored-row-set label_store \
    --k 32 \
    --counts-reference "$REF" \
    --ay-reference "$AYREF" \
    --ba-reference "$BAREF" \
    --out "$OUT"

DIRTY=$(git -C "$SLURM_SUBMIT_DIR" status --porcelain | wc -l)
if [[ "$DIRTY" -ne 0 ]]; then
    echo "FATAL: the checkout is dirty on exit ($DIRTY entries)" >&2
    git -C "$SLURM_SUBMIT_DIR" status --porcelain >&2
    exit 1
fi

echo "wrote $OUT"
