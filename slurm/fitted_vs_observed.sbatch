#!/bin/bash
#SBATCH --job-name=sae_fitted_vs_observed
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# THE FITTED PRIMARY MODEL AGAINST THE OBSERVED TOP-DECILE BINS.
# PREREGISTRATION amendment BB section 2.
#
# NO GPU AND NO ACTIVATION PASS. Every input is already on disk:
#   predictor    gtilde_compliant_b_ratio.h5:M              (20, 10240)
#   outcome      rf_counts_all_theta_n39.h5 @ theta = 0.70  (20, 10240)
#   firing rate  null_pilot_compliant.h5:fire_rate          (the JOIN's source)
#                feature_ap_main_compliant.h5:fire_counts   (the PANEL's source)
#   homology     d_fs.h5:d_fs_cluster_entropy               (20, 10240)
#
# ============================================================================
# WHY THIS REFITS, AND WHY THAT IS NOT A NEW MODEL
# ============================================================================
# The regression artifacts store b0, b1, gamma, lambda and edf -- and NOT the
# penalised spline's coefficients, its basis or its knots (BB's sealing probe,
# job 21223585). The fitted curve therefore cannot be evaluated by reading the
# JSON. The EXISTING specification is re-executed, and the claim that it is the
# same fit is an ACCEPTANCE CRITERION rather than an assertion:
#
#   ACCEPTANCE A -- every per-seed coefficient, SE, lambda, edf, deviance and
#   loglik must reproduce design3_join_au.json EXACTLY. It is checked against
#   av and aw_nofeat as well, which converts CODEBASE 20e's claim that their
#   primaries are identical into a measurement.
#
#   ACCEPTANCE B -- the OBSERVED bin means must reproduce
#   alignment_fine_panel_ba_v2.json EXACTLY. Without it a difference between
#   fitted and observed could be the model or could be a changed binning.
#
#   THE PREMISE -- `keep`, `log M` and the outcome counts must agree
#   ELEMENTWISE between the join's path and the panel's. Under gotcha 27 the
#   check is positional and FATAL.
#
# All three exit non-zero on failure, so an `afterok` chain can gate on them
# (gotcha 13 extension 1).
#
# NOT AUTHORISED (BB brief section 6): fitting an interaction term, an
# alignment spline, or any re-specification. This evaluates the existing fit.
#
# EXPLORATORY (AZ7 / BA5): cannot become primary under any outcome, and no
# mechanism sentence is written for it.
#
# THE SEED AXIS IS THE HAZARD (gotcha 29). gtilde is LEXICOGRAPHIC and stores
# no labels usable by the join; rf_counts carries NO `seeds` dataset and its
# order is the NUMERIC `seed_labels` ATTR; the AP artifact is lexicographic
# again. Positions 0 and 1 are the only two that coincide, so a positional join
# would mis-pair 18 of 20 seeds and produce a complete, plausible, wrong table.

set -euo pipefail

PROJECT=$PROJECT
SCRATCH=$SCRATCH

GTILDE="$PROJECT/analysis/gtilde_compliant_b_ratio.h5"
ALLTHETA="$PROJECT/analysis/rf_counts_all_theta_n39.h5"
RF="$PROJECT/analysis/rf_counts_theta070_n39.h5"
NULLRED="$PROJECT/analysis/null_pilot_compliant.h5"
DFS="$PROJECT/analysis/d_fs.h5"
AP="$PROJECT/analysis/feature_ap_main_compliant.h5"
JOINREF="$PROJECT/analysis/design3_join_au.json"
JOINREF2="$PROJECT/analysis/design3_join_av.json"
JOINREF3="$PROJECT/analysis/design3_join_aw_nofeat.json"
PANELREF="$PROJECT/analysis/alignment_fine_panel_ba_v2.json"
OUT="$PROJECT/analysis/${1:-fitted_vs_observed_bb.json}"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

for f in "$GTILDE" "$ALLTHETA" "$RF" "$NULLRED" "$DFS" "$AP" \
         "$JOINREF" "$JOINREF2" "$JOINREF3" "$PANELREF"; do
    if [[ ! -f "$f" ]]; then
        echo "FATAL: missing $f" >&2; exit 1
    fi
done
# Written as an explicit `if` rather than `[[ ... ]] && { ...; }`: under set -e
# the AND-OR form's behaviour on a false test is a bash-version question and
# this script must not depend on the answer.
if [[ -e "$OUT" ]]; then
    echo "FATAL: $OUT exists; every artifact goes to a NEW path" >&2; exit 1
fi

echo "[fvo.sbatch] join reference=$(basename "$JOINREF") (+2 more)"
echo "[fvo.sbatch] panel reference=$(basename "$PANELREF")"
echo "[fvo.sbatch] out=$OUT"

python -m src.analysis.fitted_vs_observed \
    --gtilde "$GTILDE" \
    --rf-counts-all-theta "$ALLTHETA" \
    --rf-counts "$RF" \
    --null-reductions "$NULLRED" \
    --d-fs "$DFS" \
    --ap "$AP" \
    --join-reference "$JOINREF" \
    --also-join-reference "$JOINREF2" "$JOINREF3" \
    --panel-reference "$PANELREF" \
    --out "$OUT"

# Exit-side clean-checkout assertion (gotcha 31).
DIRTY=$(git -C "$SLURM_SUBMIT_DIR" status --porcelain | wc -l)
if [[ "$DIRTY" -ne 0 ]]; then
    echo "FATAL: the checkout is dirty on exit ($DIRTY entries)" >&2
    git -C "$SLURM_SUBMIT_DIR" status --porcelain >&2
    exit 1
fi

echo "wrote $OUT"
