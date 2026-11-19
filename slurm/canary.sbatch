#!/bin/bash
#SBATCH --job-name=sae_canary
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out

# THE CANARY MANIFEST. PREREGISTRATION amendments AD1/AD2 and AM5.
#
# USAGE:  sbatch slurm/canary.sbatch [verify|capture]
#
#   verify   (default) compare the on-disk artifacts against the COMMITTED
#            manifest at canary/baseline.json. Read-only. Exits non-zero on any
#            movement, so an `afterok` chain or a `&&` can gate on it.
#   capture  write a CANDIDATE manifest to $PROJECT and stop. It does NOT write
#            into the checkout.
#
# ============================================================================
# WHY CAPTURE WRITES TO $PROJECT AND NOT TO THE CHECKOUT (amendment AM11)
# ============================================================================
# The first version of this script wrote canary/baseline.json into
# $SLURM_SUBMIT_DIR -- the cluster checkout, which CLAUDE.md's reproducibility
# hygiene forbids, for the reason it gives (`git_dirty()` counts untracked
# files). It did, and `bin_occupancy_b8.json` became the
# project's first artifact to record dirty provenance.
#
# So capture writes to $PROJECT, and the manifest is moved into git DELIBERATELY,
# from a workstation:
#
#   scp bouchet:$PROJECT/analysis/canary_baseline_candidate.json canary/baseline.json
#   git add canary/baseline.json && git commit && git push
#
# That ordering is the point rather than an inconvenience: the manifest is a
# record, it belongs in git with a commit date, and the cluster clone is
# READ-ONLY by design (a read-only deploy key) so it could not push it anyway.

set -euo pipefail
MODE="${1:-verify}"
PROJECT=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

case "$MODE" in
  capture)
    OUT="$PROJECT/analysis/canary_baseline_candidate.json"
    # Seed from the committed manifest when one exists, so `first_baselined_at`
    # is carried forward rather than reset to HEAD (AM5).
    if [[ -f canary/baseline.json ]]; then
        cp canary/baseline.json "$OUT"
    fi
    # `--assert-addition-only` (amendment BB section 10). `capture` re-describes
    # every target and writes whatever is on disk TODAY, so a target that had
    # DRIFTED would be silently re-baselined by the same call that adds a new
    # one. An ADDITION and a REBASELINE are different operations with different
    # justifications, and only the first is ever authorised without saying so.
    # The flag refuses the capture when any pre-existing target's baseline
    # moved, so the distinction is enforced rather than merely printed.
    python -m src.analysis.canary --root "$PROJECT/analysis" \
        --manifest "$OUT" --capture --repo . --assert-addition-only
    echo
    echo "[canary.sbatch] CANDIDATE written to $OUT -- NOT to the checkout."
    echo "[canary.sbatch] To adopt it, from a workstation:"
    echo "    scp bouchet:$OUT canary/baseline.json && git add canary/baseline.json"
    ;;
  verify)
    if [[ ! -f canary/baseline.json ]]; then
        echo "no committed manifest at canary/baseline.json -- capture, then commit it" >&2
        exit 1
    fi
    python -m src.analysis.canary --root "$PROJECT/analysis" \
        --manifest canary/baseline.json
    ;;
  *)
    echo "unknown mode '$MODE' -- expected verify or capture" >&2
    exit 1
    ;;
esac

# THE CHECKOUT MUST BE EXACTLY AS CLEAN AS IT WAS ON ENTRY, asserted rather than
# assumed -- the defect this guards against is precisely a write nobody
# intended, and it has already happened once.
DIRT="$(git status --porcelain)"
if [[ -n "$DIRT" ]]; then
    echo "FATAL: this job left the checkout DIRTY. git_dirty would then be True" >&2
    echo "for every artifact written from it afterwards (CLAUDE.md's invariant):" >&2
    echo "$DIRT" >&2
    exit 1
fi
echo "[canary.sbatch] checkout still clean"
