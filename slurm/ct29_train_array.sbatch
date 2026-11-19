#!/bin/bash
#SBATCH --job-name=ct29_train
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; one device class per arm (gotcha 36)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out

# §CT29: the k x width grid. Usage, from the arm's pinned clone:
#   sbatch --array=0 slurm/ct29_train_array.sbatch <ARM>          (the pilot seed)
#   sbatch --array=1-9 slurm/ct29_train_array.sbatch <ARM>        (the rest)
#   sbatch --partition=gpu_h200 --gpus=h200:1 --array=40 \
#          slurm/ct29_train_array.sbatch devctl_h200               (the device control)
# ARM = k64_w2560 | k64_w40960 | k16_w10240 | k128_w10240 | devctl_h200.
# Each grid config is configs/mvp.yaml with only `k` (line 98) and, off width
# 10,240, `dict_size` (line 97) changed (§CT29-F item 1); asserted below and by
# tests/test_ct29_config.py. ROOT_KIND=h200 sends a grid arm to the H200 root,
# all ten seeds there (§CT29-F item 3); the device class is asserted after training.

set -euo pipefail

ARM="${1:?ARM: k64_w2560|k64_w40960|k16_w10240|k128_w10240|devctl_h200}"
SCRATCH=$SCRATCH
PROJECT=$PROJECT
CACHE=$SCRATCH/cache
SEED=$SLURM_ARRAY_TASK_ID

case "$ARM" in
  k64_w2560)   K=64;  W=2560 ;;
  k64_w40960)  K=64;  W=40960 ;;
  k16_w10240)  K=16;  W=10240 ;;
  k128_w10240) K=128; W=10240 ;;
  devctl_h200) K=32;  W=10240 ;;
  *) echo "unknown ARM $ARM" >&2; exit 1 ;;
esac
if [[ "$ARM" == "devctl_h200" ]]; then
    CONFIG=configs/mvp.yaml
    OUT="$PROJECT/sae_ckpts_ct29_devctl/h200/seed_${SEED}"
    DEVICE_EXPECT="H200"
    [[ "$SEED" == "40" ]] || { echo "the device control is seed 40 (§CT29-F item 2)" >&2; exit 1; }
else
    CONFIG="configs/grid/mvp_${ARM}.yaml"
    case "${ROOT_KIND:-rtx}" in
      rtx)  OUT="$PROJECT/sae_ckpts_ct29/${ARM}/seed_${SEED}"; DEVICE_EXPECT="RTX 5000 Ada" ;;
      h200) OUT="$PROJECT/sae_ckpts_ct29_h200/${ARM}/seed_${SEED}"; DEVICE_EXPECT="H200" ;;
      *) echo "ROOT_KIND must be rtx or h200" >&2; exit 1 ;;
    esac
    (( SEED >= 0 && SEED <= 9 )) || { echo "grid seeds are 0-9 (§CT29.0)" >&2; exit 1; }
fi

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
[[ -z "$(git status --porcelain)" ]] || { echo "checkout dirty before the run" >&2; exit 1; }
echo "checkout $(git rev-parse HEAD) arm=$ARM seed=$SEED partition=${SLURM_JOB_PARTITION:-} host=$(hostname)"

python - "$CONFIG" "$K" "$W" <<'PYEOF'
import sys, yaml
from pathlib import Path
cfg_path, k, w = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
a = Path("configs/mvp.yaml").read_bytes().split(b"\n")
b = Path(cfg_path).read_bytes().split(b"\n")
assert len(a) == len(b), (len(a), len(b))
diff = [(i + 1, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
want = []
if w != 10240:
    exp = {2560: b"2x", 40960: b"32x"}[w]
    want.append((97, b"  dict_size: 10240          # 8x expansion",
                 b"  dict_size: %d          # %s expansion" % (w, exp)))
if k != 32:
    want.append((98, b"  k: 32", b"  k: %d" % k))
assert diff == want, diff
cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
assert cfg["sae"]["k"] == k and cfg["sae"]["dict_size"] == w, cfg["sae"]
assert cfg["reference"]["dict_size"] == 10240
print(f"config OK: {cfg_path} differs from configs/mvp.yaml on lines {[d[0] for d in diff]}")
PYEOF

SHARDS=("$CACHE"/train/shard_*.h5)
if [[ ${#SHARDS[@]} -ne 24 ]]; then
    echo "found ${#SHARDS[@]} train shards, expected 24 -- refusing" >&2; exit 1
fi
# The init gate, repeated so a hand-submitted task cannot bypass it.
python -m src.analysis.ct24_regen bdec --b-dec "$CACHE/b_dec_gm_init.pt" \
    --out "/tmp/ct29_bdec_gate_${SLURM_JOB_ID}.json"

echo "=== cache fingerprint ($ARM, seed ${SEED}) ==="
stat -c '%n %s %y' "${SHARDS[@]}" "$CACHE/b_dec_gm_init.pt"
echo "=== end cache fingerprint ==="

if [[ -e "$OUT/model.pt" ]]; then
    echo "$OUT/model.pt already exists -- refusing to overwrite" >&2; exit 1
fi
echo "training $ARM k=$K dict_size=$W seed=$SEED -> $OUT"
python -m src.sae.train --config "$CONFIG" --activations "${SHARDS[@]}" \
    --seed "$SEED" --out "$OUT"

python - "$OUT/meta.json" "$K" "$W" "$DEVICE_EXPECT" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
k, w, dev = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
assert m["k"] == k and m["dict_size"] == w, (m["k"], m["dict_size"])
assert dev in str(m["device_class"]), m["device_class"]
assert m["b_dec_init_sha256_16"] == "f3185a6c22f91039", m["b_dec_init_sha256_16"]
print(f"meta OK: k={m['k']} dict_size={m['dict_size']} device_class={m['device_class']} "
      f"partition={m.get('slurm_partition')} git_sha={m['git_sha']} dirty={m['git_dirty']}")
PYEOF
echo "checkout at exit $(git rev-parse HEAD) dirty=[$(git status --porcelain | wc -l)]"
