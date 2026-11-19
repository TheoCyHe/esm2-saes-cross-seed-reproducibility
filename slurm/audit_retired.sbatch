#!/bin/bash
#SBATCH --job-name=sae_audit_retired
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# THE COMPLETENESS AUDIT for amendment AE's rename, run over the WHOLE data root.
# Batch, not the login node: it streams ~90 GB.
#
# THIS SCRIPT IS IN GIT DELIBERATELY. It used to live in
# $PROJECT/analysis/adhoc/, outside the checkout -- which is right for a
# throwaway probe and wrong for this one, because THE EXCLUSION LIST BELOW IS
# WHAT MAKES THE COMPLETENESS CLAIM MEANINGFUL. An audit whose domain is
# recorded only in an un-versioned script asserts a completeness nobody can
# check later.
#
# ============================================================================
# THE DOMAIN, STATED RATHER THAN IMPLIED (gotcha 24 / AJ4)
# ============================================================================
# The audit does not know what an artifact is -- that is the property making it
# total over the roots it is given. The consequence is that it also finds the
# ENGLISH WORD "population" in prose, and the literals it prints in its own
# logs. Those are not retired identity keys, and no allowlisted BYTE CONTEXT can
# distinguish them (the surrounding text is arbitrary).
#
# So the claim this invocation supports is exactly:
#
#   "no retired identity key survives on any ARTIFACT under these roots"
#
# and the excluded paths below are the places that carry the literals for
# reasons other than being an artifact. Each is named with its reason. This is a
# DOMAIN STATEMENT, not a filter that makes an inconvenient failure go away --
# per AJ4, an instrument must either be total over what it claims or state its
# domain and stop claiming more.
#
# ONE OF THE EXCLUSIONS IS A FIXED POINT AND CANNOT BE AVOIDED. The audit writes
# its log into the tree it scans, and the log prints the literals it searches
# for. Without excluding its own logs, a second run can NEVER report zero -- the
# first run's output is a hit. That is a property of the tool, not a
# convenience.
# ============================================================================

set -euo pipefail
S=$SCRATCH
P=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

python -m src.analysis.audit_retired_literals \
    --root "$P/analysis" --root "$S/analysis" --root "$S/null_raw" \
    --root "$S/row_index_cache" --root "$P/sae_ckpts" --root "$P/sae_ckpts_ext" \
    --root "$S/swissprot" \
    --exclude "*/migrations/attrs_before_AE.json" \
    --exclude "*/adhoc/logs/*" \
    --exclude "*/adhoc/*.py" \
    --exclude "*/adhoc/*.sbatch" \
    --exclude "*/adhoc/README.txt" \
    --exclude "*/analysis/preflight_measure_*.out" \
    --exclude "*/analysis/*.out"

# Reasons, one per exclusion, so the domain statement above is checkable:
#
#   migrations/attrs_before_AE.json   the PRE-MIGRATION SNAPSHOT. It records the
#                                    retired names by design and is AE3
#                                    condition 3's revert path. Rewriting it
#                                    would destroy the only record of what the
#                                    attrs were before.
#   adhoc/logs/*                     job logs, including THIS AUDIT'S OWN
#                                    output, which prints the literals it
#                                    searches for. The fixed point above.
#   adhoc/*.py, adhoc/*.sbatch       historical ad-hoc scripts, kept because
#                                    they document completed jobs (see
#                                    adhoc/README.txt). They contain PRE-AE
#                                    source, e.g. `row_subset="population"`,
#                                    which is what those jobs actually ran.
#                                    Rewriting them would falsify the record.
#   adhoc/README.txt                 prose using "population" in its ordinary
#                                    English sense.
#   analysis/*.out                   captured job output: prose and tables, not
#                                    artifacts. `preflight_measure_20119936.out`
#                                    is the named instance ("untruncated
#                                    population: 51,943 proteins").
