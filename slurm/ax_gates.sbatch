#!/bin/bash
#SBATCH --job-name=sae_ax_gates
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# AMENDMENT AX'S TWO GATING CHECKS -- §1a and §1b.
#
# NOT a firewall crossing and no GPU: both are reads of artifacts already on
# disk. Nothing here computes a concept-level reproducibility number; §1b is an
# INSPECTION (`--inspect-only`), which reports the null's draw count and the
# percentiles it supports and then exits.
#
# §1a -- THE MASKING-FRACTION DISTRIBUTION, which gates §3's null launch.
#   AW5 compared three null-mask variants on five concepts spanning the
#   PREVALENCE range and found <=0.028% drift. But the mechanism the variants
#   differ by is MASKING, and masking fraction is a different quantity from
#   prevalence (gotcha 21). This measures the axis the mechanism actually runs
#   on, over all in-pool experimental concepts, and reports where AW5's five sit.
#
# §1b -- THE STORED NULL'S DRAW COUNT, which gates §4.
#   AW1.5's percentile curve runs to the 99.9th. `null_pilot_compliant.h5` is
#   named *pilot*; this reports how many draws per concept it actually holds and
#   which percentiles that supports. If the 95th-percentile reference is not
#   estimable the module exits non-zero and §4 does not run (AX §1b's rule,
#   declared before the count was read).
#
# BOTH modules EXIT NON-ZERO on refusal (gotcha 13, extension 1), so
# `set -e` propagates it and this job's exit status is the gate's verdict.

set -euo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT

LABELS="$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5"
AP_EXP="$PROJECT/analysis/feature_ap_experimental_av.h5"
NULL="$PROJECT/analysis/null_pilot_compliant.h5"
OUT_DIR="$PROJECT/analysis"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

# Gotcha 15: the identity assertions live inside
# the modules, which read the attrs. These checks only produce a clear message
# for the cheaper failure.
for f in "$LABELS" "$AP_EXP" "$NULL"; do
    [[ -f "$f" ]] || { echo "FATAL: missing $f" >&2; exit 1; }
done

echo "=============== AX §1b -- the stored null's draw count ==============="
python -m src.analysis.concept_repro --inspect-only --null "$NULL"

echo
echo "=============== AX §1a -- the masking-fraction distribution ==========="
python -m src.analysis.mask_fraction \
    --ap-experimental "$AP_EXP" \
    --labels "$LABELS" \
    --out "$OUT_DIR/mask_fraction_ax.json"

echo
echo "both gates completed"
