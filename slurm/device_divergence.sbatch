#!/bin/bash
#SBATCH --job-name=sae_device_divergence
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

# HOW MUCH DOES THE DEVICE-DEPENDENT GATHER ACTUALLY MOVE THE STATISTIC?
#
# NO GPU, deliberately, and it is worth saying why given what this measures: it
# reads two finished AP artifacts and diffs them. There is no activation gather
# here at all, so there is nothing a GPU would do.
#
# The fingerprint probes established that the gather DIFFERS by device class,
# but a sha256 is a maximally sensitive detector -- one flipped entry of
# 511,114,400 moves it exactly as much as a million would, so `DIFFER` is
# equally consistent with a negligible effect and a pervasive one. This job
# measures the MAGNITUDE, on the reduction the consumer actually takes.
#
# The two inputs are the SAME computation on two device classes:
#   reference : feature_ap_experimental_av.h5      (rtx_5000_ada, 2026-08-02)
#   other     : feature_ap_experimental_av_cpu.h5  (CPU, day partition)
# Neither records its own gather device -- write_identity has no such key --
# so the device class is established from the producing sbatch and its log,
# and the module reports the attr as ABSENT rather than assuming it matches.

set -euo pipefail

PROJECT=$PROJECT

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

python -m src.analysis.device_divergence \
    --reference "$PROJECT/analysis/feature_ap_experimental_av.h5" \
    --other     "$PROJECT/analysis/feature_ap_experimental_av_cpu.h5" \
    --stratum   experimental \
    --out       "$PROJECT/analysis/device_divergence_experimental.json"
