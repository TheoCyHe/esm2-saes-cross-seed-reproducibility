#!/bin/bash
#SBATCH --job-name=sae_repr_density
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

# d_fs -- THE REPRESENTATION-DENSITY COVARIATE.
# PREREGISTRATION F1.6, ratified in AL8b, built per the author's decision AM6.5 and
# amendment AO section 4.
#
# NOT a firewall crossing: reads firing patterns and the cluster assignment.
# No r_f, no AP, no M.
#
# NO ACTIVATION PASS IS NEEDED. The per-seed row indices are already persisted
# in $SCRATCH/row_index_cache, so this is CPU-only: it streams each seed's index
# and buckets firing by mmseqs cluster instead of by protein.
#
# Memory: the (features x clusters) accumulator is 10,240 x ~7,600 int64
# (~625 MB), plus the row offsets (~128 MB) and one block of the flat index.
#
# The same sweep also computes the per-protein footprint BOTH ways -- start
# order and the pre-AO6 stored order -- so amendment AO6's magnitude is a
# measurement rather than an argument. It costs no extra I/O.

set -euo pipefail
SCRATCH=$SCRATCH
PROJECT=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

python -m src.analysis.repr_density \
    --config configs/mvp.yaml \
    --labels "$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5" \
    --eval-cache "$SCRATCH/cache/eval.h5" \
    --clusters "$SCRATCH/swissprot/sprot_clu30_cluster.tsv" \
    --row-index-cache "$SCRATCH/row_index_cache" \
    --ckpts "$PROJECT/sae_ckpts" \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --out "$PROJECT/analysis/d_fs.h5" \
    --out-json "$PROJECT/analysis/d_fs.json"
