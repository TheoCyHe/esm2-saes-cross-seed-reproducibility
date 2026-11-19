#!/bin/bash
#SBATCH --job-name=build_labels
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# Residue-level concept labels from the Swiss-Prot flat file, aligned to the
# EVAL activation cache. CPU only.
#
# PREREQUISITE (run on the LOGIN node -- compute nodes have no internet):
#   curl -sL -o $SCRATCH/swissprot/uniprot_sprot.dat.gz \
#     https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz
#
# NOTE the flat file must be the SAME release as the FASTA the cache was built
# from, or per-protein lengths disagree and those proteins are skipped (the run
# reports the mismatch count -- a large number means a release skew).
#
# WHICH RELEASE IS ON DISK (established 2026-08-05). The URL above says
# `current_release`, which is a MOVING POINTER: re-fetching it yields a
# different release and nothing in the URL identifies the one this project
# used. No release number is recorded anywhere in the repo, configs or artifact
# attrs. The release is instead identified by these facts, all read from the
# files themselves:
#
#   entries in .dat                575,503
#   entries in the FASTA           575,503   (they agree -> same release)
#   max entry-version date         10-JUN-2026  (332,182 entries carry it)
#   sha256 uniprot_sprot.fasta     d8cd0e868ed436aeeeb129ad17ca96e526bb4bb8c354243e8e4e203afbb3b9d5
#   sha256 uniprot_sprot.dat.gz    741bcb144f98b8d10f0369b145d562b6751bfd17c285e936553aeb9cb54ab592
#
# The 575,503 figure is corroborated independently by configs/mvp.yaml's corpus
# comment and by archive/split.json (521,831 train + 53,672 eval).
#
# CAVEAT, unexplained and deliberately not resolved: the .dat.gz GZIP HEADER
# carries MTIME 2026-03-20T15:47:15Z, which CANNOT be a build date for a file
# containing 10-JUN-2026 entries. It is NOT used as a release stamp.
#
# BOTH SOURCE FILES ARE NOW ARCHIVED to $PROJECT/archive/ with these hashes in
# archive/uniprot_sources.json, so the release no longer depends on scratch.

set -euo pipefail

SCRATCH=$SCRATCH
PROJECT=$PROJECT

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

python -m src.annotations.build_residue_labels \
    --config configs/mvp.yaml \
    --dat   "$SCRATCH/swissprot/uniprot_sprot.dat.gz" \
    --cache "$SCRATCH/cache/eval.h5" \
    --out   "$PROJECT/analysis/residue_labels.h5"
