#!/bin/bash
#SBATCH --job-name=sae_as_aligned
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# THE AR9 BLAST RADIUS, RE-RUN UNDER SEED ALIGNMENT.
# PREREGISTRATION amendment AS section 1; the author's post-join brief section 1.
#
# WHY IT CLEARS AQ5's TRIAGE ON BOTH LIMBS. AL8b demoted the feature-side
# covariates to a sensitivity arm citing C1's trigger not firing -- hard-bin
# median |rho| = 0.00709 = 0.72 SE. If 18 of 20 seeds were joined across AR9's
# lexicographic/numeric split then that median is dominated by CROSS-SEED
# pairings, which are approximately null by gotcha 18, so the trigger's
# non-firing is NOT ESTABLISHED. And the arm it demoted is the one arm that
# FLIPS b1's sign (-0.005765 against +0.005596). "Plausibly changes the
# headline estimate" and "plausibly invalidates a pre-registered arm" at once.
#
# STAGE 0 LABELS THE SEED AXES so the rest of the pipeline reads an order
# rather than assuming one (AS decision 5). The labels are RECOVERED FROM THE
# DATA by seed_axis.resolve_unlabelled_seed_order, which refuses unless the
# permutation is a bijection with every margin above 3x; a target it cannot
# resolve is reported and SKIPPED, never labelled on a guess.
#
# EVERY OUTPUT IS A NEW `_aligned` PATH. The pre-alignment artifacts stay on
# disk as the superseded record, and m4_c1_all_seeds.json -- a canary target --
# is READ, never overwritten.
#
# A1 IS DECLARED BEFORE THIS RUNS (AS section 1): seeds 0 and 1 are the only
# positions where the two orders agree, so their HARD-BIN C1 entries must come
# back BIT-IDENTICAL. Movement there means the resolver changed a correct join
# and the re-run is wrong rather than the incumbent.

set -euo pipefail
SCRATCH=$SCRATCH
PROJECT=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

LAB="$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5"
AP="$PROJECT/analysis/feature_ap_main_compliant.h5"
NR="$PROJECT/analysis/null_gain_raw_compliant.h5"
ND="$PROJECT/analysis/null_pilot_compliant.h5"
GT="$PROJECT/analysis/gtilde_compliant_b_ratio.h5"
SHA=$(git rev-parse --short HEAD)

for F in "$SCRATCH/analysis/smooth_mu_b_aligned.h5" \
         "$PROJECT/analysis/gtilde_compliant_b_ratio_smooth_aligned.h5" \
         "$PROJECT/analysis/c1_argmax_as_aligned.json" \
         "$PROJECT/analysis/m4_c1_all_seeds_aligned.json"; do
  if [[ -e "$F" ]]; then
    echo "FATAL: $F already exists -- a partial earlier run would let two" >&2
    echo "generations mix silently. Remove it deliberately or pick a suffix." >&2
    exit 1
  fi
done

# ---- 0. LABEL THE UNLABELLED SEED AXES (AS decision 5) ---------------------
# In place, following the AE/AK migration precedent, each carrying its own
# attr_schema_version so a migrated axis is distinguishable from an original.
python -m migrations.as_add_seed_labels \
    --targets "$GT" \
              "$PROJECT/analysis/gtilde_compliant_b_ratio_smooth_fixfp.h5" \
              "$SCRATCH/analysis/gtilde_compliant_b_ratio_smooth_nodegen_fixfp.h5" \
    --inherit "$PROJECT/analysis/degenerate_argmax_flag.h5" "$GT" \
    --null-reductions "$ND" \
    --schema-version "AS:$SHA" \
    --out "$PROJECT/analysis/as_seed_label_migration.json"

# ---- 1. smooth mu, REBUILT with the covariates paired to the right seed ----
python -m src.analysis.smooth_mu \
    --ap "$AP" --null-raw "$NR" --null-reductions "$ND" --labels "$LAB" \
    --row-index-cache "$SCRATCH/row_index_cache" --ckpts "$PROJECT/sae_ckpts" \
    --eval-cache "$SCRATCH/cache/eval.h5" \
    --scored-row-set label_store --k 32 --arm b \
    --out "$SCRATCH/analysis/smooth_mu_b_aligned.h5" \
    --report "$PROJECT/analysis/smooth_mu_m4_aligned.json"

# ---- 2. the smooth gtilde on the rebuilt mu --------------------------------
python -m src.analysis.standardize_gain \
    --ap "$AP" --null-raw "$NR" --null-reductions "$ND" --labels "$LAB" \
    --inner-mode ratio --mu-smooth "$SCRATCH/analysis/smooth_mu_b_aligned.h5" \
    --config configs/mvp.yaml \
    --out "$PROJECT/analysis/gtilde_compliant_b_ratio_smooth_aligned.h5"

# ---- 3. AC9's CV comparison on the rebuilt arm -----------------------------
# REPORT ONLY. AQ3's branch rule already fired and hard-bin STANDS; a changed
# ratio here is reported, NOT re-adopted (brief: report, do not fix).
python -m src.analysis.compare_inner_modes \
    --full   "$SCRATCH/analysis/gtilde_compliant_b.h5" \
    --centre "$SCRATCH/analysis/gtilde_compliant_b_centre.h5" \
    --none   "$SCRATCH/analysis/gtilde_compliant_b_none.h5" \
    --ratio  "$PROJECT/analysis/gtilde_compliant_b_ratio_smooth_aligned.h5" \
    --out    "$PROJECT/analysis/cv_ci_smooth_with_aligned.json"

# ---- 4. C1 under alignment, scored against the pre-AS artifact -------------
python -m src.analysis.c1_argmax \
    --gtilde "$GT" \
    --gtilde-smooth "$PROJECT/analysis/gtilde_compliant_b_ratio_smooth_aligned.h5" \
    --mu-smooth "$SCRATCH/analysis/smooth_mu_b_aligned.h5" \
    --null-reductions "$ND" \
    --rf-counts "$PROJECT/analysis/rf_counts_theta070_n19.h5" \
                "$PROJECT/analysis/rf_counts_theta070_n39.h5" \
    --incumbent "$PROJECT/analysis/c1_argmax_ao_fixfp.json" \
    --out "$PROJECT/analysis/c1_argmax_as_aligned.json"

# ---- 5. the m4 --c1-only arm, the other consumer AR9 names -----------------
python -m src.analysis.m4_confound --c1-only \
    --ap "$AP" --null-raw "$NR" --null-reductions "$ND" --labels "$LAB" \
    --row-index-cache "$SCRATCH/row_index_cache" --ckpts "$PROJECT/sae_ckpts" \
    --eval-cache "$SCRATCH/cache/eval.h5" \
    --gtilde "$GT" \
    --mu-smooth "$SCRATCH/analysis/smooth_mu_b_aligned.h5" \
    --rf-counts "$PROJECT/analysis/rf_counts_theta070_n39.h5" \
    --out "$PROJECT/analysis/m4_c1_all_seeds_aligned.json"

echo "[as-aligned] done. READ, in order:"
echo "  as_seed_label_migration.json : which axes were labelled, and the margins"
echo "  c1_argmax_as_aligned.json    : prediction_scorecard.AS_section_1 (A1/E1/E2)"
echo "  cv_ci_smooth_with_aligned.json : REPORT ONLY -- AQ3's adoption does not reopen"
