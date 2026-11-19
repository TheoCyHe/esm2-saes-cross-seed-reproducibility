#!/bin/bash
#SBATCH --job-name=sae_never_detected
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# THE NEVER-DETECTED CONCEPTS -- amendment AZ section 15 (brief 5a).
#
# AY9 established the fact: 43 concepts of the 703-concept pool are detected by
# NO seed, and 11 of them carry experimentally-validated residues. A sparsely
# annotated concept nobody detects is unsurprising; an experimentally grounded
# one is a genuine negative about what the model misses.
#
# This job characterises them for external biological review: per concept, its
# annotated- and experimental-residue counts, its distinct protein and mmseqs
# cluster counts, its best-carrier shortfall against its own null, and the
# taxonomic composition of the proteins carrying it. The other 32 are
# summarised too, so a reader can see what distinguishes the 11.
#
# TAXONOMY IS NOT STORED ANYWHERE. build_residue_labels.py reads `lineage` and
# aggregates it into composition summaries without persisting it per protein,
# so it is re-read from the Swiss-Prot flat file for the accessions of interest
# only, in one streaming pass.
#
# THE ACCESSION JOIN IS THE HAZARD (gotcha 10). The cache index stores the whole
# FASTA token `sp|P19918|DCMS_AFICR`; the flat file and the cluster tsv use the
# bare `P19918`. Every join goes through split_train_eval.header_to_acc and the
# realised match rate is ASSERTED -- build_residue_labels.py's first run matched
# 0 of 53,672 proteins for exactly this reason and wrote an empty file while
# exiting 0.
#
# AND THE ROW-TO-PROTEIN MAP IS CHECKED POSITIONALLY (AO6): every protein block
# must begin at its own start. A permuted assignment has the same total, the
# same protein count and the same block multiset, so only a positional check
# can see it.

set -euo pipefail

PROJECT=$PROJECT
SCRATCH=$SCRATCH

AP="$PROJECT/analysis/feature_ap_main_compliant.h5"
NULL="$PROJECT/analysis/null_pilot_compliant.h5"
LABELS="$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5"
EVALCACHE="$SCRATCH/cache/eval.h5"
DAT="$SCRATCH/swissprot/uniprot_sprot.dat.gz"
CLUSTERS="$SCRATCH/swissprot/sprot_clu30_cluster.tsv"
OUT="$PROJECT/analysis/${1:-never_detected_az.json}"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

for f in "$AP" "$NULL" "$LABELS" "$EVALCACHE" "$DAT" "$CLUSTERS"; do
    if [[ ! -f "$f" ]]; then
        echo "FATAL: missing $f" >&2; exit 1
    fi
done
if [[ -e "$OUT" ]]; then
    echo "FATAL: $OUT exists; every artifact goes to a NEW path" >&2; exit 1
fi

echo "[never.sbatch] dat=$(basename "$DAT")  out=$OUT"

python -m src.analysis.never_detected \
    --ap "$AP" --null "$NULL" --labels "$LABELS" \
    --eval-cache "$EVALCACHE" --dat "$DAT" --clusters-tsv "$CLUSTERS" \
    --scored-row-set label_store \
    --out "$OUT"

DIRTY=$(git -C "$SLURM_SUBMIT_DIR" status --porcelain | wc -l)
if [[ "$DIRTY" -ne 0 ]]; then
    echo "FATAL: the checkout is dirty on exit ($DIRTY entries)" >&2
    git -C "$SLURM_SUBMIT_DIR" status --porcelain >&2
    exit 1
fi

echo "wrote $OUT"
