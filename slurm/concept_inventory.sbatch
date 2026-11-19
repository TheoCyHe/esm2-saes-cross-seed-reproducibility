#!/bin/bash
#SBATCH --job-name=sae_concept_inventory
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

# THE CONCEPT INVENTORY -- PREREGISTRATION amendment AY §4.
#
# Builds docs/CONCEPT_INVENTORY.csv for EXTERNAL BIOLOGICAL REVIEW: one row per
# concept in the full 712 library, with `in_screening_pool` distinguishing the
# 703 from the 9 excluded pooled parents. NO RESULTS COLUMNS -- the ask is a
# check on the vocabulary, and including detection rates invites comment on the
# results instead.
#
# THE ROW-TO-PROTEIN MAP IS THE PART THAT CAN GO WRONG (gotcha 27, the AO6
# defect). `protein_sets/population_start` is stored in sorted-ACCESSION order,
# with 25,508 descending steps on the real store, so using it as though it were
# in row order attributes 0.0031% of rows correctly while every total agrees.
# The module argsorts it and then verifies POSITIONALLY: every annotated
# instance must lie entirely inside one protein, an identity a permutation
# cannot satisfy. It REFUSES on any straddle rather than reporting a share.
#
# BOTH OUTPUTS GO TO $PROJECT. NOTHING IS WRITTEN INTO THE CHECKOUT.
#
# An earlier version wrote the CSV straight into the repo's docs/ so it could be
# committed. That produced the project's second artifact with `git_dirty: True`
# -- AM11's defect by a different route, and this time through a mechanism worth
# naming: the `csv` module writes CRLF on every platform while git stores LF, so
# the job rewrote all 713 lines and every concurrently-running job then recorded
# dirty provenance. `write_csv` now pins `lineterminator="\n"` so the output is
# byte-stable, AND the file no longer lands in the checkout at all. Getting the
# CSV into git is a separate, deliberate copy-and-commit.
#
# The job ASSERTS THE CHECKOUT IS CLEAN ON EXIT, the same guard canary.sbatch
# grew for the same reason -- a safeguard that breaks the invariant it enforces
# is the failure mode this project keeps rediscovering.

set -euo pipefail

PROJECT=$PROJECT

AP="$PROJECT/analysis/feature_ap_main_compliant.h5"
LABELS="$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5"
EXPAP="$PROJECT/analysis/feature_ap_experimental_av.h5"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

OUT_JSON="$PROJECT/analysis/${1:-concept_inventory_ay.json}"
OUT_CSV="${OUT_JSON%.json}.csv"

for f in "$AP" "$LABELS" "$EXPAP"; do
    [[ -f "$f" ]] || { echo "FATAL: missing $f" >&2; exit 1; }
done
[[ -f "$OUT_JSON" ]] && { echo "FATAL: $OUT_JSON exists" >&2; exit 1; }

python -m src.analysis.concept_inventory \
    --ap "$AP" --labels "$LABELS" --experimental-ap "$EXPAP" \
    --out-csv "$OUT_CSV" --out-json "$OUT_JSON"

echo "wrote $OUT_CSV"
echo "wrote $OUT_JSON"

# THE EXIT GUARD. Not decoration: this job's predecessor dirtied the checkout
# and every artifact written while it was dirty recorded it.
if [[ -n "$(git status --porcelain)" ]]; then
    echo "FATAL: this job left the checkout DIRTY -- every artifact written" >&2
    echo "while it is dirty records git_dirty=True. Offending paths:" >&2
    git status --porcelain >&2
    exit 1
fi
echo "checkout clean on exit"
