#!/bin/bash
#SBATCH --job-name=sae_concept_bridge
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# THE BRIDGE MEASUREMENTS -- PREREGISTRATION amendment AY sections 2a, 2c, 3.
#
# NO GPU AND NO ACTIVATION PASS. Every input is already on disk:
#   observed   feature_ap_main_compliant.h5's pooled `ap`
#   null       null_pilot_compliant.h5's `null_cmax_b` (Null B, across-protein)
#   labels     the pgate5 sub-typed untruncated store, for experimental counts
#
# WHAT IT ANSWERS, and why each needed running rather than reading:
#   2a(i)   whether the null is per (concept, seed) or pooled. BOTH answers are
#           true of different things -- there ARE per-seed draws, and there is
#           ONE permutation schedule shared across all seeds -- so the report
#           is derived from the artifact's own dataset SHAPES, not its name.
#           Plus the DETECTION MARGIN, which is what the independence concern
#           is actually about and is stored nowhere.
#   2a(ii)  the 43 never-detected concepts, with the composition that makes
#           them informative rather than merely small.
#   2c      does the bimodality survive conditioning on prevalence, at EVERY
#           estimable percentile.
#   3       the binned curve and the saturation threshold, which REPLACE the
#           rank correlation. Intervals come from the empirical between-concept
#           dispersion, not from a Wilson construction.
#
# THE POSITIONAL GUARDS ARE INHERITED FROM concept_repro AND ARE FATAL: seed
# labels elementwise (gotcha 29 -- AP-family artifacts are lexicographic and
# rf-family ones numeric), concept labels elementwise, and the shared feature
# axis (AW1.4's selection control is valid only while both maxima range over
# the same 10,240 features).

set -euo pipefail

PROJECT=$PROJECT

AP="$PROJECT/analysis/feature_ap_main_compliant.h5"
NULL="$PROJECT/analysis/null_pilot_compliant.h5"
LABELS="$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5"
# NEW PATH PER RUN. Archive rather than delete.
OUT="$PROJECT/analysis/${1:-concept_bridge_ay.json}"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

for f in "$AP" "$NULL" "$LABELS"; do
    [[ -f "$f" ]] || { echo "FATAL: missing $f" >&2; exit 1; }
done

[[ -f "$OUT" ]] && { echo "FATAL: $OUT exists; move it aside first" >&2; exit 1; }

python -m src.analysis.concept_bridge \
    --ap "$AP" --null "$NULL" --labels "$LABELS" --out "$OUT"

echo "wrote $OUT"
