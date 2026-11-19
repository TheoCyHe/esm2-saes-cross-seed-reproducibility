#!/bin/bash
#SBATCH --job-name=sae_concept_repro
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# CONCEPT-LEVEL REPRODUCIBILITY -- D(c).
# Specified in PREREGISTRATION amendment AW §1; authorised to run by AX §2d.
#
# NO GPU AND NO ACTIVATION PASS. Both sides are already on disk:
#   observed  A(c,s) = max over features of `feature_ap_main_compliant.h5`'s
#                      pooled `ap`, a reduction over one axis of a dense array
#   null             = `null_pilot_compliant.h5`'s `null_cmax_b`, the
#                      max-over-FEATURES reduction stored per (seed, shift,
#                      concept) precisely for this consumer (K3/I6a)
# AW3.1 established the second point by inspection; this job is what consumes it.
#
# THE THREE HAZARDS, all handled inside the module and all fatal on violation:
#  1. SEED ORDER (gotchas 27, 29). AP-family artifacts are lexicographic;
#     rf-family ones are numeric. Both artifacts here are AP-family, but that
#     is asserted ELEMENTWISE on the labels rather than inherited from AW3.1's
#     inspection.
#  2. THE SHARED FEATURE SET. AW1.4's selection control works only because the
#     observed and null maxima are over the SAME 10,240 features. Asserted by
#     requiring the null's `fire_rate` to equal the AP artifact's
#     `fire_counts / n_rows` at every one of the 204,800 cells -- which ties the
#     seed axis and the feature axis at once. A rank check would not
#     discriminate here (gotcha 27 measured 0.977 on a mapping where 0.0031% of
#     rows were right).
#  3. NULL B, NOT NULL A. `null_cmax_a` is the within-protein circular shift and
#     answers a LOCALISATION question; AW1.4 adopts the across-protein
#     permutation. The module refuses unless `null_cmax_b` is present AND
#     `dest_schedule` is -- the latter is written only on a Null B run, so it is
#     evidence about the CONSTRUCTION rather than about the dataset's name.
#
# The percentile curve is truncated to what the stored draw count supports:
# 400 pooled draws (20 seeds x 20 shifts) support the 95th-percentile reference
# and the 99th; the 99.9th needs 0.4 draws at or above it and is DROPPED, not
# interpolated (AX §1b's rule, declared before the count was read).

set -euo pipefail

PROJECT=$PROJECT

AP="$PROJECT/analysis/feature_ap_main_compliant.h5"
NULL="$PROJECT/analysis/null_pilot_compliant.h5"
# NEW PATH PER RUN. `${1}` names the artifact; the default is the first run's.
# Archive rather than delete -- an earlier run stays on disk under its own name.
OUT="$PROJECT/analysis/${1:-concept_repro_ax.json}"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

for f in "$AP" "$NULL"; do
    [[ -f "$f" ]] || { echo "FATAL: missing $f" >&2; exit 1; }
done

# NEW PATH: refuse to overwrite an existing result rather than silently
# replacing it (archive rather than delete).
[[ -f "$OUT" ]] && { echo "FATAL: $OUT exists; move it aside first" >&2; exit 1; }

python -m src.analysis.concept_repro --ap "$AP" --null "$NULL" --out "$OUT"

echo "wrote $OUT"
