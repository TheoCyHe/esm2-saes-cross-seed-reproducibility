#!/bin/bash
#SBATCH --job-name=sae_carrier_support
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; torch cu126 has no sm_120 (Blackwell) kernels
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out

# CARRIER SUPPORT -- PREREGISTRATION amendment AZ section 11 (brief 3a + 3b).
#
# AY7 measured carriers matching at 0.5617 against a FIRING-RATE-matched
# ambient of 0.4895 -- an excess of 7.22 pp. That baseline controls firing rate
# and nothing else, while homology support d_fs is worth +3.1217 pp of r_f per
# SD against log M's beta-binomial bound of 0.3635 pp (8.59x). This job re-draws
# the baseline from the same (firing rate x homology support) cell and reports
# both excesses and their ratio. It also reports where carriers sit in log M --
# the regression's own predictor, not the AP that selected them.
#
# EXPLORATORY (AZ7): cannot become primary under any outcome, and no mechanism
# sentence is written for any of it.
#
# TWO ACCEPTANCE CRITERIA, BOTH FATAL:
#   1. the per-anchor MNN counts must be BIT-IDENTICAL to rf_counts_theta070_n19
#      in NUMERIC order, while the carrier join runs in the AP artifact's
#      LEXICOGRAPHIC order -- the same array consumed both ways, so confusing
#      them fails the check (gotcha 29). Under gotcha 27 no shape, dtype or
#      total comparison is the criterion.
#   2. the SINGLE-arm rates must reproduce carrier_identity_ay_v3.json EXACTLY.
#      Both arms go through one function differing only in the cell index, so
#      criterion 2 is what makes the double arm's difference attributable to
#      the stratification rather than to anything else that changed.
#
# THIS JOB CROSSES THE r_f / ALIGNMENT LINE deliberately, as carrier_identity
# does: X7's firewall exists to stop theta being selected on the alignment
# relationship, and theta was fixed at 0.70 in amendment AB long before either
# measurement was specified.
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
OUT="$PROJECT/analysis/${1:-carrier_support_az.json}"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

ANCHORS=("$PROJECT"/sae_ckpts/seed_*)

if [[ ! -e "${ANCHORS[0]}/model.pt" ]]; then
    echo "FATAL: no anchor checkpoints in $PROJECT/sae_ckpts" >&2; exit 1
fi
if [[ "${#ANCHORS[@]}" -ne 20 ]]; then
    echo "FATAL: expected exactly 20 anchor checkpoints, found ${#ANCHORS[@]}" >&2
    exit 1
fi
for f in "$AP" "$NULL" "$GTILDE" "$DFS" "$REF" "$AYREF"; do
    if [[ ! -f "$f" ]]; then
        echo "FATAL: missing $f" >&2; exit 1
    fi
done
# Written as an explicit `if` rather than `[[ ... ]] && { ...; }`: under set -e
# the AND-OR form's behaviour on a false test is a bash-version question and
# this script must not depend on the answer (design3_join.sbatch's precedent).
if [[ -e "$OUT" ]]; then
    echo "FATAL: $OUT exists; every artifact goes to a NEW path" >&2; exit 1
fi

echo "[support.sbatch] anchors=${#ANCHORS[@]} (expected 20)"
echo "[support.sbatch] out=$OUT"

python -m src.analysis.carrier_support \
    --ap "$AP" --null "$NULL" \
    --gtilde "$GTILDE" --d-fs "$DFS" \
    --ckpts "${ANCHORS[@]}" \
    --expect-ckpts 20 \
    --counts-reference "$REF" \
    --ay-reference "$AYREF" \
    --out "$OUT"

# Exit-side clean-checkout assertion (gotcha 31).
DIRTY=$(git -C "$SLURM_SUBMIT_DIR" status --porcelain | wc -l)
if [[ "$DIRTY" -ne 0 ]]; then
    echo "FATAL: the checkout is dirty on exit ($DIRTY entries)" >&2
    git -C "$SLURM_SUBMIT_DIR" status --porcelain >&2
    exit 1
fi

echo "wrote $OUT"
