#!/bin/bash
#SBATCH --job-name=sae_ap_exp_cpu
#SBATCH --partition=day
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x_%A_%a.out

# THE EXPERIMENTAL-STRATUM AP PASS, ON CPU. Same computation, no GPU requested,
# and the OUTPUT GOES TO A NEW PATH -- it does not replace the GPU artifact.
#
# WHY THIS EXISTS. The activation gather is DEVICE-DEPENDENT. Measured by the
# two fingerprint probes (jobs 21515227 CPU / 21515228 GPU, recorded in
# $PROJECT/analysis/gather_fingerprint_{cpu,gpu}.json): nnz is IDENTICAL at
# 511,114,400 because TopK selects exactly k per row either way, but
# sha256_feat, sha256_pos, sha256_seg_start and sha256_fire_counts ALL DIFFER.
# Near-tie pre-activations swap under a CPU-versus-GPU float32 matmul, so WHICH
# features are selected on a row differs.
#
# The consequence, and the whole reason for this file: the experimental-stratum
# NULL (array 21515225, amendments BG/BH) ran on CPU, while
# feature_ap_experimental_av.h5 -- the OBSERVED statistic that null is a
# reference distribution FOR -- was gathered on rtx_5000_ada on 2026-08-02.
# A null and its statistic must rest on the same activations, which is the
# like-for-like property BG1 exists to protect.
#
# WHY THE AP SIDE MOVES RATHER THAN THE NULL. It is the cheap side. The
# average-precision arithmetic ALREADY runs on CPU here by an existing rule --
# feature_ap.py's NNZ_GPU_SAFE = 150,000,000 against this run's
# nnz = 15,972,325 x 32 = 511,114,400 -- so only the GATHER moves. MEASURED by
# the probes: 607.6 s on CPU against 200.1 s on rtx_5000_ada, a 3.04x ratio and
# about +6.8 min per task. The null cost ~250 CPU-hours; this costs ~13.
#
# ZERO CODE CHANGE. feature_ap.py line 343 already picks
# `device = "cuda" if torch.cuda.is_available() else "cpu"`, and the gather at
# line 516 runs on `device`. `--ap-device auto` already resolves to CPU at this
# nnz regardless of what device is present. Removing the GPU request changes
# WHERE the gather runs and nothing else.
#
# IT ALSO AVOIDS THE RECLAMATION. Four tasks of the GPU null array (21466226)
# were cancelled for holding an rtx_5000_ada at ~0.6%
# utilisation, and those cancellations were correct. This script would have hit
# the same thing for the same reason.
#
# ARCHIVE, DO NOT REPLACE. SHARD_DIR and the merged output are BOTH new paths.
# feature_ap_experimental_av.h5 is the record of what was computed before and is
# the input to the CPU-versus-GPU divergence measurement; overwriting it would
# destroy the only copy of one side of that comparison.
#
# DATED 2026-09-16: THE PARAGRAPH BELOW DESCRIBES THE CODE AS IT STOOD WHEN THIS
# JOB RAN. The "DURABLE FIX OWED" was made 2026-08-10: `write_identity` now
# records `gather_device_class` and `ap_arith_device_class` for NEW artifacts
# (feature_ap.py). Nothing is back-filled, so the artifact this script produced
# still carries neither key and compares as ABSENT_ON_BOTH against its GPU
# twin, exactly as described below. Kept as the record of that state.
#
# ONE IDENTITY KEY IS ABSENT AND IS REPORTED AS ABSENT, not treated as matching:
# `write_identity` records git_sha, git_dirty, ap_impl, scored_row_set, n_rows,
# negatives, strata and the label-store carried keys, but records NO GATHER
# DEVICE. So neither this artifact nor the GPU one can state its own device
# class from its attrs. The evidence is external -- this file requests no GPU
# and feature_ap.py prints `WARNING: no GPU visible.` on a CPU node. Adding a
# `gather_device` attr is a DURABLE FIX OWED; it is deliberately not made here,
# because a new attr on one side of a pair being compared for identity
# agreement is a change to make with the comparison in hand, not during it.
#
# EVERYTHING BELOW THE DIRECTIVES IS THE GPU FILE'S BODY, unchanged except for
# SHARD_DIR. Same label store, same gates, same flags, same preflight.

set -euo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT

# The f67485d store is the CURRENT axis: 712 library / 703 pool, untruncated,
# gate 5. It is the store `feature_ap_main_compliant.h5` carries as its
# `library_sha`, which is what makes the two artifacts comparable at all.
LABELS="${1:-$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5}"
SHARD_DIR="$SCRATCH/ap_exp_shards_cpu"

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
