#!/bin/bash
#SBATCH --job-name=validate_ap_restructure
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# GPU AUDIT (2026-08-10): THE DEVICE MUST STAY. This validator performs an
# explicit CPU-versus-CUDA cross-check (its log: "cuda available for device
# cross-check: True"), so a GPU is required for it to do its job at all.
# Utilisation is the healthiest of any script audited -- MEASURED on job
# 20129342: a 239.8 s gather of a 444 s task, ~54%.

# 96G, not 32G: the full-eval activation gather (17.7M rows, ~567M non-zeros)
# is the SAME size regardless of how many concepts are tested -- the first
# attempt at 32G OOM'd during the gather itself, before any AP computation
# started. 96G matches what orphan_screen.sbatch needed for the same gather.

# Item 3/4 -- real-scale validation of the P1 restructure (bit-identical vs
# the old path) AND the seed-0 timing baseline, on seed_0 real activations,
# against the literal top-10-by-P_c concepts plus 10 more spanning down to
# the rarest survivors.

set -euo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

LABELS="${1:-$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_9bfa94b.h5}"

# RE-VALIDATION after wiring the sparse path into feature_ap.py (amendment
# I1b). The earlier validation ran the two implementations against each other
# in THIS harness on the FULL cache; the production run uses the untruncated
# population and the newly-wired code path, and agreement in one harness does
# not establish agreement in the other. So: same row subset as the run
# (--scored-row-set label_store) and a STRATIFIED concept selection rather than
# hand-picked list -- the top 10 by P_c plus the 10 sparsest, the latter being
# where a feature's positives sit outside its top-activating rows and the
# closed-form tied-zero tail correction carries the most weight.
python -m src.analysis.validate_ap_restructure     --config configs/mvp.yaml     --ckpt "$PROJECT/sae_ckpts/seed_0"     --eval-cache "$SCRATCH/cache/eval.h5"     --labels "$LABELS"     --scored-row-set label_store     --auto-concepts 20     --out "$PROJECT/analysis/validate_ap_restructure_seed0_wired.json"
