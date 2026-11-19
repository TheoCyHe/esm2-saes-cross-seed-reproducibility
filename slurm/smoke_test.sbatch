#!/bin/bash
#SBATCH --job-name=esm2_smoke
#SBATCH --partition=gpu_devel
# Pin the GPU TYPE (gotcha 7). rtx_5000_ada is sm_89 and is what `gpu` runs anyway.
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# Stage-1 smoke test: does the pipeline run end to end on real GPU, and is
# the cache actually aligned to the residues it claims? Toy-sized on purpose
# -- this validates plumbing, not science. The SAE hyperparameters below are
# smoke-test values, NOT the locked recipe, which is in configs/mvp.yaml
# (CLAUDE.md open question 3). This comment previously said
# those were "still TODO in configs/mvp.yaml" -- corrected at CH8.

set -euo pipefail

SCRATCH=$SCRATCH
OUT=$SCRATCH/smoke

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

mkdir -p "$OUT"

echo "=== node / GPU ==="
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo
echo "=== 1. cache layer-24 activations for the toy set ==="
python -m src.embeddings.cache \
    --layer 24 \
    --fasta data/toy_200.fasta \
    --out   "$OUT/toy_l24.h5"

echo
echo "=== 2. verify the cache against the source FASTA ==="
python -m src.embeddings.verify_cache \
    --cache "$OUT/toy_l24.h5" \
    --fasta data/toy_200.fasta

echo
echo "=== 3. compute the shared frozen b_dec (once; shared across all seeds) ==="
python -m src.sae.compute_b_dec \
    --config configs/mvp.yaml \
    --activations "$OUT/toy_l24.h5" \
    --out "$OUT/b_dec.pt"

echo
echo "=== 4. train one SAE on it (smoke values, not the recipe) ==="
python -m src.sae.train \
    --config configs/mvp.yaml \
    --activations "$OUT/toy_l24.h5" \
    --seed 0 \
    --out "$OUT/sae_seed0" \
    --b-dec "$OUT/b_dec.pt" \
    --lr 1e-3 --batch-size 256 --n-steps 200 --log-every 50

echo
echo "=== 5. checkpoint metadata ==="
python -c "
import json
m = json.load(open('$OUT/sae_seed0/meta.json'))
m.pop('fire_counts')
print(json.dumps(m, indent=2))
"

echo
echo "=== SMOKE TEST COMPLETE ==="
