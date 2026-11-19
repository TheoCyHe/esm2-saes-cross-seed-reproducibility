#!/bin/bash
#SBATCH --job-name=build_labels_subtyped
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# PREREGISTRATION amendment E -- the SUB-TYPED concept library (DOMAIN/REGION/
# MOTIF/ZN_FING split by /note into per-family concepts, e.g. "DOMAIN:Protein
# kinase") with the disjunctive InterPLM retention floor (residues>1500 OR
# instances>10, configs/mvp.yaml annotations.retention_floor_mode). This is
# what the NEW MAIN AP analysis runs against.
#
# Writes to a SEPARATE file (residue_labels.h5, the original 97-concept/
# legacy-floor file, stays as-is and continues to back the robustness-check
# machinery -- amendment C masking, D6, D7's strict/wide partition -- now
# repositioned per amendment E as robustness checks, not the headline). Same
# .dat, same eval cache, same release -- only the concept-extraction rule
# changed.
#
# VERSIONED, never overwritten: the output path is suffixed with the git SHA
# at build time. A queued job reading a fixed path (e.g. the item-3/4
# validation run against residue_labels_subtyped.h5) must never have its
# comparison basis shift under it because a concurrent rebuild landed on the
# same filename. This also means the rebuild has NO reason to wait behind
# GPU-bound jobs -- it's parse-side CPU and does not compete for the same
# nodes.
#
# A convenience "latest" symlink/copy is intentionally NOT created here --
# every consumer (sbatch scripts, PREREGISTRATION.md) should name the exact
# versioned file it depends on, so "which version was this run against" is
# never ambiguous from the filename alone.

set -euo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

SHA=$(git rev-parse --short HEAD)
POPULATION="${POPULATION:-all}"
MIN_PROTEIN_COUNT="${MIN_PROTEIN_COUNT:-1}"
# amendment AE (0ef716e, 2026-07-29) renamed the builder's --population flag to
# --truncation-policy {with_truncated, untruncated}. This script's call site was
# renamed with it but no default was set, so under `set -u` the job aborted at
# the python call unless the submitter had exported TRUNCATION_POLICY -- every
# recorded run of this script (2026-07-27/28) predates the rename and ran on
# `--population "$POPULATION"` (DATAFLOW section 4; fixed 2026-09-16). POPULATION
# stays the documented env var and still names the output file above;
# TRUNCATION_POLICY defaults FROM it -- `all` -> `with_truncated` (the builder's
# own argparse default, and the value the default-population artifacts
# residue_labels_subtyped_{b3d8ddb,e625861}.h5 carry in their attrs),
# `untruncated` -> `untruncated`. An explicit TRUNCATION_POLICY that disagrees
# with POPULATION is refused rather than written into a file whose name says
# otherwise.
case "$POPULATION" in
    all)         TP_FROM_POPULATION=with_truncated ;;
    untruncated) TP_FROM_POPULATION=untruncated ;;
    *) echo "POPULATION=$POPULATION is not one of: all, untruncated" >&2; exit 2 ;;
esac
TRUNCATION_POLICY="${TRUNCATION_POLICY:-$TP_FROM_POPULATION}"
if [ "$TRUNCATION_POLICY" != "$TP_FROM_POPULATION" ]; then
    echo "TRUNCATION_POLICY=$TRUNCATION_POLICY disagrees with POPULATION=$POPULATION" >&2
    exit 2
fi
if [ "$POPULATION" = "all" ] && [ "$MIN_PROTEIN_COUNT" = "1" ]; then
    OUT="${1:-$PROJECT/analysis/residue_labels_subtyped_${SHA}.h5}"
else
    OUT="${1:-$PROJECT/analysis/residue_labels_subtyped_${POPULATION}_pgate${MIN_PROTEIN_COUNT}_${SHA}.h5}"
fi

# amendment G2: POPULATION=untruncated (env var, e.g.
# `sbatch --export=POPULATION=untruncated slurm/build_residue_labels_subtyped.sbatch`)
# selects the <1024-residue/InterPLM population and prints the G3 protein-count
# cost curve on real data. Default stays "all" so existing callers are unaffected.
# amendment G3: MIN_PROTEIN_COUNT (env var, default 1 = no-op) applies the
# estimability gate decided from that cost curve (see PREREGISTRATION.md
# amendment G).
python -m src.annotations.build_residue_labels \
    --config configs/mvp.yaml \
    --dat   "$SCRATCH/swissprot/uniprot_sprot.dat.gz" \
    --cache "$SCRATCH/cache/eval.h5" \
    --out   "$OUT" \
    --truncation-policy "$TRUNCATION_POLICY" \
    --min-protein-count "$MIN_PROTEIN_COUNT" \
    --clusters "$SCRATCH/swissprot/sprot_clu30_cluster.tsv"

echo "wrote $OUT (git SHA $SHA, truncation_policy=$TRUNCATION_POLICY, min_protein_count=$MIN_PROTEIN_COUNT)"
