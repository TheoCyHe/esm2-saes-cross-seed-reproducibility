#!/bin/bash
#SBATCH --job-name=sae_carrier_identity
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; torch cu126 has no sm_120 (Blackwell) kernels
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# CARRIER IDENTITY -- PREREGISTRATION amendment AY §2b.
#
# When a concept is detected in both anchor seed s and anchor seed t, is the
# best-matching feature in s a MUTUAL-NEAREST-NEIGHBOUR partner of the
# best-matching feature in t, at theta = 0.70?
#
# THIS JOB DELIBERATELY CROSSES THE r_f / ALIGNMENT LINE, so it does NOT run
# behind amendment X7's firewall -- that firewall exists to stop theta being
# selected on the alignment relationship, and theta was fixed at 0.70 in
# amendment AB long before this measurement was specified. The crossing is
# labelled here for the same reason m4_confound.py labels its own.
#
# ONLY THE 20 ANCHORS ARE LOADED. Carriers require AP, AP exists only for seeds
# 0-19, so the carrier question is necessarily over the 190 anchor-anchor pairs
# at n_compare = 19. THE AMBIENT RATE IS COMPUTED ON THAT SAME PAIR SET -- the
# published 0.4958 is the n_compare = 39 figure and quoting it here would be a
# silent change of N.
#
# THE ACCEPTANCE CRITERION, and it is the reason this job can be believed.
# The per-anchor MNN counts computed here at theta = 0.70 must be BIT-IDENTICAL
# to `match_count` in the existing rf_counts_theta070_n19.h5, which is stored
# in NUMERIC seed order while the carrier join runs in the AP artifact's
# LEXICOGRAPHIC order. The SAME computed array is consumed in both orders, so
# confusing them fails the check. THE JOB EXITS NON-ZERO IF IT DOES.
# Under gotcha 27 no shape, dtype or total comparison is the criterion; they
# are reported for diagnosis only.
#
# THE GLOB HAZARD (AD5). Extension seeds live in sae_ckpts_ext/ and are NOT
# anchors. The guard asserts EXACTLY 20 anchors, in shell and again in python
# via --expect-ckpts: a self-consistency check on the count passes at 20 and at
# 40 alike and is therefore not a guard.

set -euo pipefail

PROJECT=$PROJECT

AP="$PROJECT/analysis/feature_ap_main_compliant.h5"
NULL="$PROJECT/analysis/null_pilot_compliant.h5"
REF="$PROJECT/analysis/rf_counts_theta070_n19.h5"
OUT="$PROJECT/analysis/${1:-carrier_identity_ay.json}"

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
for f in "$AP" "$NULL" "$REF"; do
    [[ -f "$f" ]] || { echo "FATAL: missing $f" >&2; exit 1; }
done
# The acceptance reference must exist BEFORE the run -- failing here is louder
# than failing after the pairs are computed.
[[ -f "$OUT" ]] && { echo "FATAL: $OUT exists; move it aside first" >&2; exit 1; }

echo "[carrier.sbatch] anchors=${#ANCHORS[@]} (expected 20)  ref=$(basename "$REF")"

python -m src.analysis.carrier_identity \
    --ap "$AP" --null "$NULL" \
    --ckpts "${ANCHORS[@]}" \
    --expect-ckpts 20 \
    --counts-reference "$REF" \
    --out "$OUT"

echo "wrote $OUT"
