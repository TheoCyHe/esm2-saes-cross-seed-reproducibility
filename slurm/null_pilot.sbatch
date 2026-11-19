#!/bin/bash
#SBATCH --job-name=sae_null_pilot
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
# GPU AUDIT (2026-08-10, figure CORRECTED 2026-08-11): THE GPU IS WELL USED AND
# THE DEVICE MUST STAY. Measured from job 20142136's own log, summed rather
# than sampled: row-index build 6,572.8 s over 20 seeds, PLUS the shift loop
# 1,163.8 s over 400 shifts, = 7,736.6 s of 7,827 s elapsed -- 98.8%.
#
# THE 2026-08-10 FIGURE OF ~87% WAS AN UNDERCOUNT AND THE REASON IS WORTH
# KEEPING. It counted only the row-index build, on the belief that the AP
# arithmetic runs on CPU here because the log prints `ap_device=cpu`. It does
# not: `ap_device` is consumed ONLY by the masked branch. This is the UNMASKED
# path, which keeps feat_by_row/rank_by_row/running_max on `device` and runs
# gather_touches + average_precision_from_touches ON THE GPU for every
# (shift, concept). Which branch ran is read from the code, not from what the
# log prints (gotchas 20 and 21).
#
#
# And the pooled null it writes pairs with `feature_ap_main*.h5`, produced on
# rtx_5000_ada (gotcha 36).

# RERUN #2 (PREREGISTRATION amendments I4 + K3, 2026-07-28). Supersedes
# null_pilot_g.h5, which maxed over all 716 concepts INCLUDING the 9 pooled
# parents and is therefore a null for a different statistic than `g_f` now
# is. This run uses --exclude-pooled-parents (pool = 707 on the 716-concept axis
# of 2026-07-28; on the current 712-concept axis, since amendment N, the pool
# the record and concept_repro.py carry is 703) and writes to a NEW
# path so the superseded artifact is not clobbered.
#
# STORAGE (amendment K3): the small reductions go to PROJECT; the ~23 GB RAW
# per-(seed, shift, concept, feature) array goes to SCRATCH. Project is 91%
# full group-wide (369 GB free) and CLAUDE.md's storage policy (Facts that
# changed the plan) rules 23 GB there out.
# Scratch has 9.5 TB free. The raw array is a regenerable intermediate,
# deliberately not archived.
#
# ORIGINAL HEADER (amendment G, G8/G9): 20 replicates x 716
# concepts x 20 seeds x BOTH nulls (Null A joint-shift + Null B cross-protein
# permutation, G4), against the untruncated/gate-5 label store (G2/G3).
# Estimated: the row-index build (dominant one-time cost per seed, F4's
# measured ~17s/seed at 20 shifts for Null A alone) is shared between both
# nulls; the per-replicate AP computation roughly DOUBLES running --null-type
# both vs either alone (two full concept passes per replicate, not one).
# Budgeted 10h (up from Amendment F4's 6h) for margin since Null B's
# full-scale timing is genuinely unmeasured before this run -- document the
# uncertainty rather than guess a tighter number.

set -euo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT
LABELS="${1:-$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_9bfa94b.h5}"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

CKPTS=("$PROJECT"/sae_ckpts/seed_*)
if [[ ! -e "${CKPTS[0]}/model.pt" ]]; then
    echo "no ensemble checkpoints in $PROJECT/sae_ckpts" >&2; exit 1
fi

# ---- CROSS-COMPONENT PRE-FLIGHT GATE (amendment I2) ----------------------
# The null and feature_ap_main.h5 must agree on population, concept axis,
# floor parameters and library_sha, or `g~` standardizes the real statistic
# against a null drawn on a different axis -- exactly the defect class this
# gate exists to catch, and exactly what amendment I4 caught after the fact
# last time. Asserted BEFORE 2h19m of GPU is spent.
python -m src.analysis.preflight --labels "$LABELS" \
    --ap "$PROJECT/analysis/feature_ap_main.h5"

# --scored-row-set label_store: MANDATORY with this store. Its n_rows attr
# is the POPULATION-restricted count (untruncated only), but the eval cache's
# own index_json still describes the FULL cache (all 53,672 proteins,
# including truncated ones) -- scored-row-set=full_eval builds a row->protein map
# from the full index against a population-sized n_rows, silently
# misaligning every row past the mismatch. label_store mode loads the labels
# file's own protein_sets/population_start,population_n_rows and remaps
# (build_population_index, G2) instead.
python -m src.analysis.null_pilot \
    --config configs/mvp.yaml \
    --ckpts "${CKPTS[@]}" \
    --eval-cache "$SCRATCH/cache/eval.h5" \
    --labels "$LABELS" \
    --scored-row-set label_store \
    --null-type both \
    --n-length-bins 20 \
    --n-shifts 20 \
    --null-seed 0 \
    --cache-dir "$SCRATCH/row_index_cache" \
    --exclude-pooled-parents \
    --out "$PROJECT/analysis/null_pilot_k.h5" \
    --raw-out "$SCRATCH/null_raw/null_gain_raw_k.h5"
# THE SCRATCH `--raw-out` IS DELIBERATE AND WAS NOT MISSED BY BU PART C. This is
# the PRODUCER of the 23.4 GiB raw null, and a copy of its output now also lives
# on project (BT-A3). Repointing a producer would write 23.4 GiB into project
# space that is 92.4% full -- a different decision from repointing a reader, and
# one this amendment does not make. Every CONSUMER of this file was repointed.
