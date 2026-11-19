#!/bin/bash
#SBATCH --job-name=ct24_train_k64
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; and one device class for the whole arm (gotcha 36)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x_%A_%a.out

# §CT24: twenty SAEs at k = 64, seeds 0-19, width 10,240, layer 24. Everything
# but sae.k is configs/mvp.yaml (configs/k64/mvp_k64.yaml differs on one line,
# asserted below and by tests/test_ct24_config.py).

set -euo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT
CACHE=$SCRATCH/cache
SEED=$SLURM_ARRAY_TASK_ID
CONFIG=configs/k64/mvp_k64.yaml

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
echo "checkout $(git rev-parse HEAD) dirty=[$(git status --porcelain | wc -l)]"

python - "$CONFIG" <<'PYEOF'
import sys, yaml
from pathlib import Path
a = Path("configs/mvp.yaml").read_bytes().split(b"\n")
b = Path(sys.argv[1]).read_bytes().split(b"\n")
diff = [(x, y) for x, y in zip(a, b) if x != y]
assert len(a) == len(b) and diff == [(b"  k: 32", b"  k: 64")], diff
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert cfg["sae"]["k"] == 64 and cfg["sae"]["dict_size"] == 10240
assert cfg["reference"]["dict_size"] == 10240
print("config OK: one-line difference sae.k 32 -> 64")
PYEOF

SHARDS=("$CACHE"/train/shard_*.h5)
if [[ ${#SHARDS[@]} -ne 24 ]]; then
    echo "found ${#SHARDS[@]} train shards, expected 24 -- refusing" >&2; exit 1
fi
# The init gate, repeated here so a hand-submitted task cannot bypass it.
python -m src.analysis.ct24_regen bdec --b-dec "$CACHE/b_dec_gm_init.pt" \
    --out "/tmp/ct24_bdec_gate_${SLURM_JOB_ID}.json"

echo "=== cache fingerprint (k64, seed ${SEED}) ==="
stat -c '%n %s %y' "${SHARDS[@]}" "$CACHE/b_dec_gm_init.pt"
echo "=== end cache fingerprint ==="

OUT="$PROJECT/sae_ckpts_k64/seed_${SEED}"
if [[ -e "$OUT/model.pt" ]]; then
    echo "$OUT/model.pt already exists -- refusing to overwrite" >&2; exit 1
fi
echo "training k=64 seed=${SEED} -> $OUT"
python -m src.sae.train --config "$CONFIG" --activations "${SHARDS[@]}" \
    --seed "$SEED" --out "$OUT"
echo "checkout at exit $(git rev-parse HEAD) dirty=[$(git status --porcelain | wc -l)]"
