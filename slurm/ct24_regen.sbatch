#!/bin/bash
#SBATCH --job-name=ct24_regen
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G               # compute_b_dec: Weiszfeld on 1M x 1280 float64, ~25 GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# §CT24-F item 2: the training cache's regeneration, gated against the record.
# One step per submission (STEP passed by --export, asserted below):
#   STEP=shards     shard_fasta.py on $PROJECT/archive/train.fasta, then the
#                   shards.json gate against $PROJECT/archive/shards.json.
#   (then slurm/cache_train_array.sbatch, unchanged, 24 tasks on rtx_5000_ada)
#   STEP=post_cache the row-count and shard-0 attribute gate against the k = 32
#                   checkpoints' meta.json; compute_b_dec.py (sample seed 0,
#                   the configured 1,000,000 rows) to sae.b_dec_path; the init
#                   gate -- b_dec_sha256_16 must be f3185a6c22f91039 (the author's
#                   decision 3, 2026-10-23) or the job exits 1 and nothing that
#                   depends on it (afterok) starts.
#   STEP=hash       sha256 of the 24 shards and the init, to the manifest.

set -euo pipefail

if [[ -z "${STEP:-}" ]]; then
    echo "STEP is unset -- pass --export=ALL,STEP=<shards|post_cache|hash>" >&2
    exit 1
fi

SCRATCH=$SCRATCH
PROJECT=$PROJECT
CACHE=$SCRATCH/cache
SHARD_DIR=$SCRATCH/swissprot/split/shards
OUTDIR=$PROJECT/analysis/ct

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
mkdir -p "$OUTDIR"
echo "checkout $(git rev-parse HEAD) dirty=[$(git status --porcelain | wc -l)]"

case "$STEP" in
  shards)
    if compgen -G "$SHARD_DIR/shard_*.fasta" > /dev/null; then
        echo "shard FASTAs already exist in $SHARD_DIR -- refusing to overwrite" >&2
        exit 1
    fi
    python -m src.embeddings.shard_fasta --config configs/mvp.yaml \
        --fasta "$PROJECT/archive/train.fasta" --out-dir "$SHARD_DIR" --n-shards 24
    python -m src.analysis.ct24_regen shards --new "$SHARD_DIR/shards.json" \
        --old "$PROJECT/archive/shards.json" --out "$OUTDIR/ct24_regen_shards_ct.json"
    ;;
  post_cache)
    SHARDS=("$CACHE"/train/shard_*.h5)
    if [[ ${#SHARDS[@]} -ne 24 ]]; then
        echo "found ${#SHARDS[@]} train shards, expected 24" >&2; exit 1
    fi
    python -m src.analysis.ct24_regen cache --shards "${SHARDS[@]}" \
        --meta "$PROJECT/sae_ckpts/seed_0/meta.json" --out "$OUTDIR/ct24_regen_cache_ct.json"
    if [[ -e "$CACHE/b_dec_gm_init.pt" ]]; then
        echo "$CACHE/b_dec_gm_init.pt already exists -- refusing to overwrite" >&2
        exit 1
    fi
    python -m src.sae.compute_b_dec --config configs/mvp.yaml \
        --activations "${SHARDS[@]}" --sample-seed 0 --out "$CACHE/b_dec_gm_init.pt"
    python -m src.analysis.ct24_regen bdec --b-dec "$CACHE/b_dec_gm_init.pt" \
        --out "$OUTDIR/ct24_regen_bdec_ct.json"
    ;;
  hash)
    SHARDS=("$CACHE"/train/shard_*.h5)
    python -m src.analysis.ct24_regen hash --files "${SHARDS[@]}" "$CACHE/b_dec_gm_init.pt" \
        --out "$OUTDIR/ct24_train_cache_regen_ct.json"
    ;;
  *)
    echo "unknown STEP=$STEP" >&2; exit 1 ;;
esac
echo "checkout at exit $(git rev-parse HEAD) dirty=[$(git status --porcelain | wc -l)]"
