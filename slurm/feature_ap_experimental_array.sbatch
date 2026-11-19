#!/bin/bash
#SBATCH --job-name=sae_ap_exp
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; torch cu126 has no sm_120 (Blackwell) kernels
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x_%A_%a.out
# GPU AUDIT (2026-08-10): THE DEVICE STAYS. This is the LOW-UTILISATION case --
# 511,114,400 non-zeros routes the AP arithmetic to CPU, so the GPU buys only
# the ~200 s gather out of a 1,678-2,693 s task (~8-12%, INFERRED from the CPU
# variant running the identical work 600-800 s slower). It is NOT changed here
# because its output, `feature_ap_experimental_av.h5`, is the ARCHIVED
# GPU-derived half of the device-divergence measurement (gotcha 36) and one of
# the two inputs `device_divergence.py` diffs. Use
# `feature_ap_experimental_array_cpu.sbatch` for a fresh run: it produces the
# artifact that shares a device class with the experimental null.

# THE EXPERIMENTAL-STRATUM AP PASS (PREREGISTRATION amendment AV10d).
#
# WHAT IT IS FOR. Amendment AV4 replaces the retired strict concept list with an
# EVIDENCE-QUALITY axis: ALIGNED clears the bar against any concept in the
# 703-concept pool on the EXPERIMENTAL stratum, ORPHAN fails it against every
# concept on the POOLED stratum. The pooled side already exists
# (feature_ap_main_compliant.h5 -- AV9, so the wide-screen rebuild is NOT run).
# The experimental side does not: the only experimental AP artifacts are the
# 29-concept, 1.9M-row shared_experimental pair, which is three library
# rebuilds and one population filter stale.
#
# WHY IT IS AN ARRAY AND NOT ONE JOB, and this is measured rather than
# preferred (AV10c). `--negatives unannotated` is MANDATORY here per C4 -- under
# `all_rows` a by-similarity residue is scored as a negative, capping attainable
# AP near the concept's experimental fraction f_c (measured range 0.006-0.66),
# which makes the pre-registered `G >= 0.05` gain floor UNREACHABLE by a perfect
# detector on sparsely-evidenced concepts. So the bar would become a function of
# annotation completeness rather than of detection.
#
# Masking forces the SEGMENTED AP path (I1b: the sparse path ranks against the
# full evaluated row set, so masked rows would keep ranks they no longer earn,
# and feature_ap.py REFUSES --ap-impl sparse with masking active). Measured
# cost: ~12.7 s per concept-evaluation at 15,972,325 rows against the sparse
# path's 0.024 s, i.e. ~2.6 h/seed and ~52 h for 20 seeds -- OVER the 48 h wall
# limit. The per-seed activation gather is already paid per seed, so one task
# per seed costs nothing extra and buys 20x wall clock.
#
# THE NULL FOR THIS STRATUM IS **NOT** LAUNCHED: at 20 shifts x 703 concepts x
# 2 nulls it is ~99 h/seed, ~1,980 GPU-hours. See AV10e for the two things owed
# before it could be (an exact fast masked-AP path, and a specification question
# about whether the masked set shifts with the positives).
#
# Shards land in SCRATCH -- they are regenerable intermediates, per CLAUDE.md's
# storage policy (Facts that changed the plan). Merge with
# slurm/merge_ap_experimental.sbatch, which orders rows by seed LABEL against
# feature_ap_main_compliant.h5 and asserts the two strata are comparable.

set -euo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT

# The f67485d store is the CURRENT axis: 712 library / 703 pool, untruncated,
# gate 5. It is the store `feature_ap_main_compliant.h5` carries as its
# `library_sha`, which is what makes the two artifacts comparable at all.
LABELS="${1:-$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5}"
SHARD_DIR="$SCRATCH/ap_exp_shards"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

SEED="$SLURM_ARRAY_TASK_ID"
CKPT="$PROJECT/sae_ckpts/seed_$SEED"

# The array index IS the seed number, deliberately -- no glob, so §7.3's
# lexicographic sort cannot enter here. The merge re-orders to the reference's
# own labels afterwards and asserts the result positionally (gotcha 29).
if [[ ! -f "$CKPT/model.pt" ]]; then
    echo "FATAL: no checkpoint at $CKPT" >&2; exit 1
fi
if [[ ! -f "$LABELS" ]]; then
    echo "FATAL: missing label store $LABELS" >&2; exit 1
fi

python -m src.analysis.preflight --labels "$LABELS"

python - "$LABELS" <<'PYCHECK'
import sys, h5py
with h5py.File(sys.argv[1], "r") as hf:
    pol = hf.attrs.get("truncation_policy")
    gate = hf.attrs.get("min_protein_count")
    n = hf.attrs.get("n_concepts")
    tiers = sorted(next(iter(hf["concepts"].values())).keys())
if pol != "untruncated" or int(gate or 0) < 5:
    sys.exit(f"FATAL: truncation_policy={pol} min_protein_count={gate}; the "
             "experimental pass requires the untruncated population (G2) and "
             "the >=5 protein gate (G3). Refusing.")
print(f"label store OK: truncation_policy={pol} min_protein_count={gate} "
      f"n_concepts={n}; tiers present on a sample concept: {tiers}")
PYCHECK

mkdir -p "$SHARD_DIR"

# --strata experimental ALONE (not with pooled): requesting more than one
# stratum triggers feature_ap.py's COMMON-CONCEPT-SET restriction, which would
# silently intersect the experimental concept set with the pooled one and make
# the scored axis a function of the request rather than of the evidence.
python -m src.analysis.feature_ap \
    --config configs/mvp.yaml \
    --ckpts "$CKPT" \
    --eval-cache "$SCRATCH/cache/eval.h5" \
    --labels "$LABELS" \
    --strata experimental \
    --scored-row-set label_store \
    --negatives unannotated \
    --ap-device auto \
    --out "$SHARD_DIR/shard_seed_${SEED}.h5"

echo "shard for seed $SEED written to $SHARD_DIR/shard_seed_${SEED}.h5"
