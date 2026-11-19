#!/bin/bash
#SBATCH --job-name=sae_main_ap
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; torch cu126 has no sm_120 (Blackwell) kernels
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
# GPU AUDIT (2026-08-10, basis stated explicitly 2026-08-11): THE GPU IS WELL
# USED HERE AND THE DEVICE MUST STAY, for two independent reasons.
# (1) UTILISATION IS HIGH, not low: job 20130645 ran 20 seeds in 5,142 s =
# 257.1 s/seed against a 239.8 s gather at this scale. ~93%.
#
#   THE LABEL IS `INFERRED` AND STAYS THERE. This job's log times NO phase, so
#   there is nothing to sum -- the 239.8 s is TRANSFERRED from
#   validate_ap_restructure job 20129342, which gathered the identical
#   511,114,400 non-zeros on the same device class. A figure from another job
#   is an inference however well matched.
#
#   CORROBORATED, from a measurement made for another purpose: 257.1 - 239.8 =
#   17.3 s/seed of non-gather work, against 17.2 s predicted by 716 concepts x
#   the measured 0.024 s per sparse concept-evaluation. Two numbers from
#   unrelated sources agreeing to 0.6%. Unlike null_pilot, the AP arithmetic
#   here IS on CPU and that is load-bearing: 511M nnz is above NNZ_GPU_SAFE, so
#   feat_d/pos_d/seg_d and the whole sparse path move to `ap_device`. The
#   sparse AP path is cheap; the gather is the work.
# (2) PAIRING: this writes
# `feature_ap_main.h5`, whose null (`null_pilot_k.h5` /
# `null_pilot_compliant.h5`) was produced on rtx_5000_ada (gotcha 36).

# THE MAIN AP ANALYSIS, per PREREGISTRATION amendments E / Issue 1
# (2026-07-27, tracing the planning thread's correction that the
# shared-experimental/masked run was never supposed to be "the primary
# analysis" -- it was "primary among the stratified robustness contrasts").
#
# Full eval set (~53,672 proteins, ~17.7M rows), POOLED evidence (no evidence-
# tier split -- that split remains a robustness check, see below), against the
# SUB-TYPED concept library (residue_labels_subtyped.h5, amendment E: DOMAIN/
# REGION/MOTIF/ZN_FING split by /note into per-family concepts) with the
# disjunctive InterPLM retention floor. This is the analysis the classification
# rule, the null, and the matched-pair headline are computed against.
#
# The amendment-C/D6/D7 machinery on residue_labels.h5 (shared-experimental,
# masked, un-subtyped 97-concept axis) is NOT replaced -- it is now the
# ROBUSTNESS layer: does the effect survive restriction to (a) experimental-
# evidence-only labels, (b) the A1 shared-protein subset. Both are reported
# alongside this, never in place of it.
#
# --ap-device auto switches the AP arithmetic to CPU above 150M non-zeros (the
# full-eval scale needs it -- see the OOM/fix on job 19628580 -> 19645357,
# logged in PREREGISTRATION.md amendment D7a). Same reason this needs ~4h and
# 96G host memory, not the ~1h a GPU-resident run would take at a scale that fits.

set -euo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT
# DEFAULT CORRECTED 2026-07-28. This previously defaulted to
# "$PROJECT/analysis/residue_labels_subtyped.h5", which was WORSE than a
# missing file: that path exists, so the guard below passed, but it is the
# 822-concept / population=all / 17.7M-row STAGE-2 library (git_sha f3e2ddb) --
# three generations stale. It predates REPEAT/TOPO_DOM/SIGNAL/TRANSIT
# sub-typing (amendment E6), the untruncated population filter (G2), and the
# >=5 protein gate (G3). A run against it would silently have produced a
# 822-concept AP matrix on the WRONG population, on a different concept axis
# than null_pilot_g.h5's 716 -- and nothing downstream checks that.
#
# The default is now the gate-5 untruncated store the null was actually drawn
# against. Pass an explicit path as $1 to override.
LABELS="${1:-$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_9bfa94b.h5}"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

CKPTS=("$PROJECT"/sae_ckpts/seed_*)
if [[ ! -e "${CKPTS[0]}/model.pt" ]]; then
    echo "no ensemble checkpoints in $PROJECT/sae_ckpts" >&2; exit 1
fi
if [[ ! -f "$LABELS" ]]; then
    echo "missing $LABELS -- run build_residue_labels_subtyped.sbatch first" >&2
    exit 1
fi

# ---- CROSS-COMPONENT PRE-FLIGHT GATE (amendment I2) ----------------------
# Runs BEFORE the job and refuses to launch on mismatch. This is the guard
# that would have caught the T6 blockers: concept-count agreement, population
# agreement, floor-parameter agreement, AP implementation version, and git
# cleanliness (logging --porcelain, not just the boolean).
#
# The null artifact is deliberately NOT passed here: null_pilot_g.h5 is
# SUPERSEDED by amendment I4 (it maxed over 716 including the 9 pooled
# parents; the pool is now 707) and would correctly fail the gate. main_ap
# writes raw AP and takes no max, so it does not depend on the null.
python -m src.analysis.preflight --labels "$LABELS"

# Belt-and-braces on the two parameters this run must not get wrong.
python - "$LABELS" <<'PYCHECK'
import sys, h5py
with h5py.File(sys.argv[1], "r") as hf:
    pop = hf.attrs.get("truncation_policy")
    gate = hf.attrs.get("min_protein_count")
    n = hf.attrs.get("n_concepts")
if pop != "untruncated" or int(gate or 0) < 5:
    sys.exit(f"FATAL: {sys.argv[1]} has truncation_policy={pop} min_protein_count={gate} "
             f"(n_concepts={n}). The main analysis requires the untruncated "
             f"population (amendment G2) and the >=5 protein gate (G3). Refusing.")
print(f"label store OK: truncation_policy={pop} min_protein_count={gate} n_concepts={n}")
PYCHECK

# --scored-row-set label_store (NOT full_eval): amendment G2. full_eval gathers
# full 17.7M-row cache while the label store's n_rows attr is 15,972,325,
# making every prevalence -- and therefore every normalised gain -- wrong.
python -m src.analysis.feature_ap \
    --config configs/mvp.yaml \
    --ckpts "${CKPTS[@]}" \
    --eval-cache "$SCRATCH/cache/eval.h5" \
    --labels "$LABELS" \
    --strata pooled \
    --scored-row-set label_store \
    --negatives all_rows \
    --ap-device auto \
    --out "$PROJECT/analysis/feature_ap_main.h5"
