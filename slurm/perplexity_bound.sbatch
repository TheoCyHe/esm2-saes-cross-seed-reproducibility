#!/bin/bash
#SBATCH --job-name=sae_perplexity_bound
#SBATCH --partition=gpu_devel
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; torch cu126 has no sm_120 (Blackwell) kernels
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# BOUNDING PERPLEXITY'S CONTRIBUTION TO THE +7.22 pp CARRIER EXCESS.
# PREREGISTRATION amendment BC section 2, with BC section 2e's conditional
# follow-on taken IN CODE by `outcome_cell`.
#
# WHY THIS RUNS DESPITE BB'S STOPPING RULE. The rule stands; this repairs an
# ASYMMETRY in BB's brief -- its enrichment half measured BOTH feature-side
# covariates and its matching half specified only concentration. That is an
# omission in a work order, not a candidate found by looking at results.
# BOUNDED: perplexity only, and no further attribution attempts after it.
#
# NOTHING IS FITTED. The coefficients are READ from design3_join_au.json.
#
# ============================================================================
# PARTITION: gpu_devel, DELIBERATELY (amendment BC8b)
# ============================================================================
# BB12 item 4 cost five hours of queue for a job that ran in 5 m 28 s, because
# the partition note then in CLAUDE.md (now its Cluster section), written for
# a SIX-HOUR cache run, was applied to a five-minute one. gpu_devel carries the
# SAME rtx_5000_ada nodes. The guidance is about WALLTIME, not legitimacy.
# The architecture is still requested explicitly because gpu_devel is
# heterogeneous and a Blackwell node has no kernels (gotcha 7).
#
# ============================================================================
# THE DEVICE CLASS IS PART OF THE ACCEPTANCE CRITERION (amendment BC7)
# ============================================================================
# Acceptance A demands the MNN counts be BIT-IDENTICAL to rf_counts_theta070_n19.
# BB12.3 measured those counts moving in 4 of 204,800 cells between a CPU
# partition and rtx_5000_ada. So this job MUST run on rtx_5000_ada -- not for
# speed, but because the criterion asserts the device class.
#
# ============================================================================
# SIX ACCEPTANCE CRITERIA, ALL FATAL
# ============================================================================
#   A  the per-anchor MNN counts are BIT-IDENTICAL to rf_counts_theta070_n19,
#      in NUMERIC order, while the carrier join runs LEXICOGRAPHIC (gotcha 29).
#   B  the perplexity enrichment distribution reproduces carrier_concentration_bb
#      EXACTLY and the d_fs arm reproduces carrier_support_ba's BA9 block
#      EXACTLY -- so the covariate rebuild, the carrier identification and the
#      peer pools are provably the same objects BB9.1 measured.
#   C  the AME reconstruction reproduces the stored z_logM `ame`, all 20 seeds.
#   D  the sign-convention probe recovers a POSITIVE coefficient from a
#      synthetic positive relation AND a NEGATIVE one from its flipped control.
#   E  the single matched-ambient arm reproduces carrier_identity_ay_v3 EXACTLY,
#      which is what pins the +7.2219 pp denominator.
#   F  the stored coefficients re-derive BB9.3's published summary.
#
# EXPLORATORY (AZ7 / BA5 / BC3): cannot become primary under any outcome, and
# no mechanism sentence is written for any of it.
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
JOIN="$PROJECT/analysis/design3_join_au.json"
REF="$PROJECT/analysis/rf_counts_theta070_n19.h5"
AYREF="$PROJECT/analysis/carrier_identity_ay_v3.json"
BAREF="$PROJECT/analysis/carrier_support_ba.json"
CONCREF="$PROJECT/analysis/carrier_concentration_bb.json"
LABELS="$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5"
ROWIDX="$SCRATCH/row_index_cache"
CKPTS="$PROJECT/sae_ckpts"
EVALCACHE="$SCRATCH/cache/eval.h5"
OUT="$PROJECT/analysis/${1:-perplexity_bound_bc.json}"

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
for f in "$AP" "$NULL" "$GTILDE" "$DFS" "$JOIN" "$REF" "$AYREF" "$BAREF" \
         "$CONCREF" "$LABELS" "$EVALCACHE"; do
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

echo "[pbound.sbatch] anchors=${#ANCHORS[@]} (expected 20)"
echo "[pbound.sbatch] coefficients read from $(basename "$JOIN")"
echo "[pbound.sbatch] out=$OUT"
nvidia-smi --query-gpu=name --format=csv,noheader || true

python -m src.analysis.perplexity_bound \
    --ap "$AP" --null "$NULL" \
    --gtilde "$GTILDE" --d-fs "$DFS" \
    --design3-join "$JOIN" \
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
    --concentration-reference "$CONCREF" \
    --out "$OUT"

DIRTY=$(git -C "$SLURM_SUBMIT_DIR" status --porcelain | wc -l)
if [[ "$DIRTY" -ne 0 ]]; then
    echo "FATAL: the checkout is dirty on exit ($DIRTY entries)" >&2
    git -C "$SLURM_SUBMIT_DIR" status --porcelain >&2
    exit 1
fi

echo "wrote $OUT"
