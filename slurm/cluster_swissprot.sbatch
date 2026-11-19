#!/bin/bash
#SBATCH --job-name=sprot_cluster
#SBATCH --partition=day
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

# CPU-only: no --gpus. mmseqs2 clustering of Swiss-Prot at 30% identity.
# Gates data.train_max_residues -- see the CORPUS REPORT at the end of the log.
#
# This does NOT start the scratch purge clock in any meaningful way (the
# policy: CLAUDE.md's Cluster section, read off `getquota`):
# the FASTA and mmseqs tmp are cheap to regenerate. The expensive, clock-
# starting artifact is the activation cache, which is deliberately built later.

set -euo pipefail

SCRATCH=$SCRATCH

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

python -m src.data.cluster_swissprot \
    --config configs/mvp.yaml \
    --workdir "$SCRATCH/swissprot" \
    --threads "$SLURM_CPUS_PER_TASK"
