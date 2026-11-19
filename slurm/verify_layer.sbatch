#!/bin/bash
#SBATCH --job-name=esm2_verify_layer
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; torch cu126 has no sm_120 (Blackwell) kernels
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

# Verify HF hidden_states[24] == the reference SAE's layer 24 (task 16).
#
# PREREQUISITE (run on the LOGIN node first -- compute nodes have no internet):
#   curl -sL -o $SCRATCH/reference/esm2_plm1280_l24_sae4096.safetensors \
#     https://huggingface.co/liambai/InterProt-ESM2-SAEs/resolve/main/esm2_plm1280_l24_sae4096.safetensors
# ESM-2-650M is already in the HF cache from the smoke test.

set -euo pipefail

SCRATCH=$SCRATCH
SAE=$SCRATCH/reference/esm2_plm1280_l24_sae4096.safetensors
EVAL=$SCRATCH/swissprot/split/eval.fasta

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

# Force offline so a missing cache fails fast instead of hanging on a node with
# no outbound internet.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m src.embeddings.verify_layer \
    --sae   "$SAE" \
    --fasta "$EVAL" \
    --layers 22 23 24 25 26 \
    --ref-layer 24 \
    --k 32 64 128 256 \
    --n-seqs 50 \
    --out "$SCRATCH/reference/layer_verify.json"
