#!/bin/bash
#SBATCH --job-name=sae_av_counts
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# AMENDMENT AV section 4 (AV11) -- the two counts.
#
#   §4a  how many of the 703 pool concepts carry at least one
#        EXPERIMENTALLY-VALIDATED residue in the current population. A concept
#        with none cannot contribute an aligned feature, so this bounds the
#        aligned group before the classification exists. Sealed: AV-P1.
#   §4b  what share of feature-concept pairs clears the pre-registered gain
#        floor G = (AP - p)/(1 - p) >= 0.05 on the POOLED stratum. The bar's two
#        halves cost very differently, and this says what the expensive half
#        buys. Sealed: AV-P2.
#
# NO GPU and NO RECOMPUTE. §4a reads the label store's tier datasets; §4b reads
# `feature_ap_main_compliant.h5`, which AV9 established already IS the current
# wide screen -- which is why §2's 3 h 35 m rebuild was not run.
#
# The AP artifact is 106 MB on disk / 583 MB logical (AU8), read one seed at a
# time, so 32G is generous.

set -euo pipefail

PROJECT=$PROJECT
LABELS="${1:-$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5}"
AP="$PROJECT/analysis/feature_ap_main_compliant.h5"
OUT="$PROJECT/analysis/av_counts.json"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

for f in "$LABELS" "$AP"; do
    if [[ ! -f "$f" ]]; then echo "MISSING INPUT: $f" >&2; exit 1; fi
done
if [[ -e "$OUT" ]]; then
    echo "$OUT already exists -- refusing to overwrite" >&2; exit 1
fi

# Gotcha 15. The gate asserts the AP artifact is on
# the axis the counts claim to be about, before either count is computed.
python -m src.analysis.preflight --labels "$LABELS" --ap "$AP"

python -m src.analysis.av_counts \
    --labels "$LABELS" \
    --ap "$AP" \
    --stratum pooled \
    --floor 0.05 \
    --out "$OUT"
