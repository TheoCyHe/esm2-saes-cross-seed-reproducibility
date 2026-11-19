#!/bin/bash
#SBATCH --job-name=cn_train_width
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1   # sm_89; torch cu126 has no sm_120 (Blackwell) kernels.
                                # ALSO an experimental constraint, not just a
                                # kernel one: the activation gather is
                                # device-dependent (gotcha 36), so every arm of
                                # the sweep must come off ONE device class.
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
# NOTE: SLURM does NOT expand shell variables in #SBATCH directives, so the
# width cannot appear here. Pass it in the job name at submit time instead:
#   sbatch --job-name=cn_w40960 --export=ALL,WIDTH=40960 ...
#SBATCH --output=logs/%x_%A_%a.out

# CN width sweep, stage B. One dictionary width per submission; seeds are the
# array axis. Submit as:
#
#   sbatch --export=ALL,WIDTH=40960 --array=0-0 slurm/cn_train_width_array.sbatch
#   sbatch --export=ALL,WIDTH=40960 --array=1-9 slurm/cn_train_width_array.sbatch
#
# Everything except dict_size and seed is inherited from configs/mvp.yaml via
# the per-width config, which differs from it on exactly one line.

set -euo pipefail

# Gotcha 32a (its fifth variant): WIDTH is asserted, not assumed.
if [[ -z "${WIDTH:-}" ]]; then
    echo "WIDTH is unset or empty -- pass --export=ALL,WIDTH=<n>" >&2
    exit 1
fi
if ! [[ "$WIDTH" =~ ^[0-9]+$ ]]; then
    echo "WIDTH=$WIDTH is not an integer" >&2
    exit 1
fi

SCRATCH=$SCRATCH
PROJECT=$PROJECT
CACHE=$SCRATCH/cache
SEED=$SLURM_ARRAY_TASK_ID

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

CONFIG="configs/width/mvp_w${WIDTH}.yaml"
if [[ ! -f "$CONFIG" ]]; then
    echo "missing $CONFIG -- run: python -m src.sae.make_width_configs" >&2
    exit 1
fi

# The config must actually carry the width this job claims to train, and must
# NOT have disturbed the comparator's declared width. Cheap, and it makes a
# mis-submitted WIDTH loud instead of silent.
python - "$CONFIG" "$WIDTH" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
want = int(sys.argv[2])
got = cfg["sae"]["dict_size"]
assert got == want, f"{sys.argv[1]} has sae.dict_size={got}, job claims {want}"
assert cfg["reference"]["dict_size"] == 10240, (
    f"reference.dict_size is {cfg['reference']['dict_size']}, expected InterPLM's 10240"
)
assert cfg["sae"]["k"] == 32, f"sae.k is {cfg['sae']['k']}, expected 32 (k is held absolute)"
print(f"config OK: dict_size={got} k={cfg['sae']['k']} reference.dict_size=10240")
PYEOF

SHARDS=("$CACHE"/train/shard_*.h5)
if [[ ! -e "${SHARDS[0]}" ]]; then
    echo "no train shards in $CACHE/train -- the cache purged; regenerate before training" >&2
    exit 1
fi
if [[ ${#SHARDS[@]} -ne 24 ]]; then
    echo "found ${#SHARDS[@]} train shards, expected 24 -- refusing a partial cache" >&2
    exit 1
fi
if [[ ! -f "$CACHE/b_dec_gm_init.pt" ]]; then
    echo "missing $CACHE/b_dec_gm_init.pt" >&2
    exit 1
fi

# WHICH CACHE THIS ARM TRAINED AGAINST (CN-h). meta.json records the shard
# paths; paths alone do not distinguish an original cache from a regenerated
# one at the same path, so record the mtimes and sizes too.
echo "=== cache fingerprint (arm w${WIDTH}, seed ${SEED}) ==="
stat -c '%n %s %y' "${SHARDS[@]}"
echo "=== end cache fingerprint ==="

OUT="$PROJECT/sae_ckpts_width/w${WIDTH}/seed_${SEED}"
# train.py's --out uses mkdir(exist_ok=True) and would overwrite a checkpoint
# in place. Archive, never delete -- and never overwrite either.
if [[ -e "$OUT/model.pt" ]]; then
    echo "$OUT/model.pt already exists -- refusing to overwrite" >&2
    exit 1
fi

echo "training w=${WIDTH} seed=${SEED} -> $OUT"
python -m src.sae.train \
    --config "$CONFIG" \
    --activations "${SHARDS[@]}" \
    --seed "$SEED" \
    --out "$OUT"
