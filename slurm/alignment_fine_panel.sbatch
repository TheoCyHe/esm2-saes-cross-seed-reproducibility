#!/bin/bash
#SBATCH --job-name=sae_fine_panel
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

# THE FINE PANEL INSIDE THE TOP ALIGNMENT DECILE -- amendment AZ section 12
# (brief 3c).
#
# NO GPU AND NO ACTIVATION PASS. Three arrays already on disk:
#   predictor    gtilde_compliant_b_ratio.h5:M            (20, 10240)
#   outcome      rf_counts_theta070_n39.h5:match_count    (20, 10240)
#   firing rate  feature_ap_main_compliant.h5:fire_counts (20, 10240)
#
# design3_join already reports mean r_f by log M DECILE per firing stratum. A
# relationship confined to the top few percent of the alignment distribution is
# invisible at that resolution, where the concept-level measure's carriers --
# at most 703 features, under 7% of the dictionary -- sit mixed with roughly
# nine times as many non-carriers. This re-bins INSIDE the top decile at
# percentile resolution.
#
# EXPLORATORY (AZ7): cannot become primary under any outcome, and no mechanism
# sentence is written for it.
#
# THE SEED AXIS IS THE HAZARD (gotcha 29). gtilde is LEXICOGRAPHIC; rf_counts
# carries NO `seeds` dataset at all and its order is the NUMERIC `seed_labels`
# ATTR; the AP artifact is lexicographic again. Every permutation is taken BY
# LABEL and refused on a set mismatch. Positions 0 and 1 are the only two that
# coincide, so a positional join would mis-pair 18 of 20 seeds and produce a
# complete, internally consistent, entirely wrong panel.
#
# N. The outcome is r_f at n_compare = 39 -- the primary. The n19 artifact is a
# one-argument re-run and the artifact records which it used, because mixing
# the two would be a silent change of N.

set -euo pipefail

PROJECT=$PROJECT
SCRATCH=$SCRATCH

GTILDE="$PROJECT/analysis/gtilde_compliant_b_ratio.h5"
AP="$PROJECT/analysis/feature_ap_main_compliant.h5"
RF="$PROJECT/analysis/rf_counts_theta070_${2:-n39}.h5"
OUT="$PROJECT/analysis/${1:-alignment_fine_panel_az.json}"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

for f in "$GTILDE" "$AP" "$RF"; do
    if [[ ! -f "$f" ]]; then
        echo "FATAL: missing $f" >&2; exit 1
    fi
done
if [[ -e "$OUT" ]]; then
    echo "FATAL: $OUT exists; every artifact goes to a NEW path" >&2; exit 1
fi

echo "[panel.sbatch] rf=$(basename "$RF")  out=$OUT"

# BA7.4's ACCEPTANCE CRITERION, and it is FATAL. The restated verdict is only
# interpretable if every OLD number reproduces the artifact it restates; if a
# bin mean or an old verdict moved, something other than the criterion changed
# and the old-vs-new comparison would be comparing two different panels.
# Passed only when the reference EXISTS -- an absent reference is reported by
# its absence here rather than silently skipping the check.
# $3, NOT $2: this script's second positional already selects the n19/n39 arm
# (see RF above), so a reference passed as $2 would silently change which
# comparison-count artifact the panel reads.
REFPANEL="${3:-$PROJECT/analysis/alignment_fine_panel_az_v2.json}"
REFARGS=()
if [[ -f "$REFPANEL" ]]; then
    REFARGS=(--reference "$REFPANEL")
    echo "[panel.sbatch] acceptance reference=$REFPANEL"
else
    echo "[panel.sbatch] NO acceptance reference at $REFPANEL -- the BA11" \
         "restatement will run WITHOUT its acceptance criterion" >&2
fi

python -m src.analysis.alignment_fine_panel \
    --gtilde "$GTILDE" --rf-counts "$RF" --ap "$AP" --out "$OUT" \
    "${REFARGS[@]}"

DIRTY=$(git -C "$SLURM_SUBMIT_DIR" status --porcelain | wc -l)
if [[ "$DIRTY" -ne 0 ]]; then
    echo "FATAL: the checkout is dirty on exit ($DIRTY entries)" >&2
    git -C "$SLURM_SUBMIT_DIR" status --porcelain >&2
    exit 1
fi

echo "wrote $OUT"
