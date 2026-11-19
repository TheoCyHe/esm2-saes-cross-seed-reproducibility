#!/bin/bash
#SBATCH --job-name=sae_rebuild_smooth
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# THE SMOOTH-ARM REBUILD ON THE CORRECTED FOOTPRINT.
# PREREGISTRATION amendment AQ3; the author's ratified item 2.
#
# WHY: AC10's both-normalisers arm is PRE-REGISTERED, and an arm computed on a
# corrupted row mapping cannot serve as a sensitivity arm. The pre-AO6
# protein_of_row attributed 489 of 15,972,325 rows to the right protein, and
# smooth_mu's `perplexity` / `concentration` covariates come straight from it.
# perplexity moved by a median 12.6% and a p90 32.5% (AO17), and smooth_mu
# splines the VALUE, not the rank -- so Spearman = 0.977 is not reassurance.
#
# THE BRANCH RULE AND THE PREDICTION ARE IN GIT BEFORE THIS RUNS (AQ3). Three
# outcomes, total over the space: smooth clears the 1.0837 bar with a HIGHER
# ratio than hard-bin's 1.006947 (hard-bin stands, AC10's arm viable); clears
# with a LOWER ratio (the predictor-of-record decision REOPENS -- report and
# STOP); or FAILS the bar (hard-bin stands, AC10's arm non-viable).
#
# EVERY OUTPUT IS A NEW `_fixfp` PATH. The pre-fix artifacts stay on disk for
# comparison, and m4_c1_all_seeds.json -- a canary target -- points at
# smooth_mu_b.h5 and must not be disturbed.
#
# Both smooth arms are rebuilt. AC's three-arm structure exists so that a
# failure is ATTRIBUTABLE (smooth trend vs the 22-handling); dropping one arm
# would remove exactly that property.

set -euo pipefail
SCRATCH=$SCRATCH
PROJECT=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

LAB="$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5"
AP="$PROJECT/analysis/feature_ap_main_compliant.h5"
NR="$PROJECT/analysis/null_gain_raw_compliant.h5"
ND="$PROJECT/analysis/null_pilot_compliant.h5"

# Refuse to clobber the pre-fix artifacts of record. An EXISTENCE check is the
# right one here -- these paths must NOT exist yet, and if they do, a previous
# partial run left state and the arms would silently mix generations.
#
# THE SCRATCH PATHS BELOW ARE DELIBERATE AND WERE NOT MISSED BY BU PART C. This
# script PRODUCES `smooth_mu_b_fixfp.h5` and
# `gtilde_compliant_b_ratio_smooth_fixfp.h5` (steps 1 and 3) and then consumes
# what it just wrote (steps 3, 4 and 5), so every reference here is internal to
# one run. BU repointed only the reference this script does NOT produce -- the
# pooled raw null at `NR` above. Copies of both outputs now also live on
# project (BT-A3); repointing a PRODUCER would write 1.1 GiB to project space
# that is 92.4% full, which is a different decision from repointing a reader.
for F in "$SCRATCH/analysis/smooth_mu_b_fixfp.h5" \
         "$SCRATCH/analysis/smooth_mu_b_nodegen_fixfp.h5" \
         "$SCRATCH/analysis/gtilde_compliant_b_ratio_smooth_fixfp.h5" \
         "$SCRATCH/analysis/gtilde_compliant_b_ratio_smooth_nodegen_fixfp.h5"; do
  if [[ -e "$F" ]]; then
    echo "FATAL: $F already exists. A partial earlier run would let two" >&2
    echo "generations of the smooth arm mix silently. Remove it deliberately" >&2
    echo "or pick a new suffix." >&2
    exit 1
  fi
done

# ---- 1. smooth mu, degenerate concepts IN the fit (AC10's declared arm) -----
python -m src.analysis.smooth_mu \
    --ap "$AP" --null-raw "$NR" --null-reductions "$ND" --labels "$LAB" \
    --row-index-cache "$SCRATCH/row_index_cache" --ckpts "$PROJECT/sae_ckpts" \
    --eval-cache "$SCRATCH/cache/eval.h5" \
    --scored-row-set label_store --k 32 --arm b \
    --out "$SCRATCH/analysis/smooth_mu_b_fixfp.h5" \
    --report "$PROJECT/analysis/smooth_mu_m4_fixfp.json"

# ---- 2. smooth mu, the 22 OUT of estimation, IN the max pool ---------------
python -m src.analysis.smooth_mu \
    --ap "$AP" --null-raw "$NR" --null-reductions "$ND" --labels "$LAB" \
    --row-index-cache "$SCRATCH/row_index_cache" --ckpts "$PROJECT/sae_ckpts" \
    --eval-cache "$SCRATCH/cache/eval.h5" \
    --scored-row-set label_store --k 32 --arm b --exclude-degenerate \
    --out "$SCRATCH/analysis/smooth_mu_b_nodegen_fixfp.h5" \
    --report "$PROJECT/analysis/smooth_mu_m4_nodegen_fixfp.json"

# ---- 3. the two gtilde artifacts -------------------------------------------
python -m src.analysis.standardize_gain \
    --ap "$AP" --null-raw "$NR" --null-reductions "$ND" --labels "$LAB" \
    --inner-mode ratio --mu-smooth "$SCRATCH/analysis/smooth_mu_b_fixfp.h5" \
    --config configs/mvp.yaml \
    --out "$SCRATCH/analysis/gtilde_compliant_b_ratio_smooth_fixfp.h5"

python -m src.analysis.standardize_gain \
    --ap "$AP" --null-raw "$NR" --null-reductions "$ND" --labels "$LAB" \
    --inner-mode ratio \
    --mu-smooth "$SCRATCH/analysis/smooth_mu_b_nodegen_fixfp.h5" \
    --config configs/mvp.yaml \
    --out "$SCRATCH/analysis/gtilde_compliant_b_ratio_smooth_nodegen_fixfp.h5"

# ---- 4. AC9's adoption comparison, with the bootstrap CI -------------------
python -m src.analysis.compare_inner_modes \
    --full   "$SCRATCH/analysis/gtilde_compliant_b.h5" \
    --centre "$SCRATCH/analysis/gtilde_compliant_b_centre.h5" \
    --none   "$SCRATCH/analysis/gtilde_compliant_b_none.h5" \
    --ratio  "$SCRATCH/analysis/gtilde_compliant_b_ratio_smooth_fixfp.h5" \
    --out    "$PROJECT/analysis/cv_ci_smooth_with_fixfp.json"

python -m src.analysis.compare_inner_modes \
    --full   "$SCRATCH/analysis/gtilde_compliant_b.h5" \
    --centre "$SCRATCH/analysis/gtilde_compliant_b_centre.h5" \
    --none   "$SCRATCH/analysis/gtilde_compliant_b_none.h5" \
    --ratio  "$SCRATCH/analysis/gtilde_compliant_b_ratio_smooth_nodegen_fixfp.h5" \
    --out    "$PROJECT/analysis/cv_ci_smooth_without_fixfp.json"

# ---- 5. AL2's argmax-selected C1 table, with a corrected smooth row --------
python -m src.analysis.c1_argmax \
    --gtilde "$PROJECT/analysis/gtilde_compliant_b_ratio.h5" \
    --gtilde-smooth "$SCRATCH/analysis/gtilde_compliant_b_ratio_smooth_fixfp.h5" \
    --mu-smooth "$SCRATCH/analysis/smooth_mu_b_fixfp.h5" \
    --null-reductions "$ND" \
    --rf-counts "$PROJECT/analysis/rf_counts_theta070_n19.h5" \
                "$PROJECT/analysis/rf_counts_theta070_n39.h5" \
    --out "$PROJECT/analysis/c1_argmax_ao_fixfp.json"

echo "[rebuild] done. The BRANCH RULE is AQ3's; read"
echo "  cv_ci_smooth_with_fixfp.json : cv_ratio_vs_none.ratio"
echo "against hard-bin's 1.006947 and the 1.0837 bar."
