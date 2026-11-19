#!/bin/bash
# BA12 -- THE CORROBORATED REMOTE WAIT. PREREGISTRATION amendment BA 7.5.
#
# USAGE (from a workstation, and note there is no `$` inside a quoted remote
# string anywhere in this workflow -- the script is COPIED and then RUN):
#
#   scp scripts/remote_wait.sh bouchet:$PROJECT/analysis/adhoc/remote_wait.sh
#   ssh bouchet 'bash <cluster-fs>/.../adhoc/remote_wait.sh 21181890 21181889'
#
# ============================================================================
# THE TWO RULES THIS SCRIPT EXISTS TO ENFORCE
# ============================================================================
# 1. **A SCRIPT FILE, NOT A QUOTED REMOTE STRING.** `$(...)`, `$j` and `$f`
#    inside `ssh host '...'` expand on the LOCAL side. That has now produced
#    four empty log headers, one empty `json.tool` read, and -- the dangerous
#    one -- a wait loop that printed ALL_DONE while three jobs were RUNNING
#    (AZ20 item 3). Everything below runs on the cluster because the FILE is
#    there.
#
# 2. **EVERY WAIT IS CORROBORATED BY AN INDEPENDENT READOUT.** The loop polls
#    `squeue` (the live scheduler queue) and, before it may exit, checks
#    `sacct` (the accounting database) for the SAME job ids. Agreement is
#    required to proceed; disagreement REFUSES; and a SILENT secondary readout
#    refuses too, because silence is not corroboration (gotcha 28).
#
# Exit codes come from `src/cluster/wait_guard.py` and are gateable:
#   0 corroborated complete   1 refused (uncorroborated or unavailable)
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: remote_wait.sh <jobid> [jobid ...]" >&2
    exit 64
fi

CHECKOUT="${CHECKOUT:-$HOME/esm2-saes-cross-seed-reproducibility}"
INTERVAL="${INTERVAL:-60}"
MAX_POLLS="${MAX_POLLS:-360}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

JOBLIST="$(IFS=,; echo "$*")"

cd "$CHECKOUT"
# shellcheck disable=SC1091
source env/activate.sh

for ((i = 1; i <= MAX_POLLS; i++)); do
    # PRIMARY: the live queue. `-h` strips the header so an empty queue is
    # genuinely empty output rather than a header line.
    squeue -h -j "$JOBLIST" -o '%T' > "$WORK/primary.txt" 2>/dev/null || true

    # INDEPENDENT: the accounting database. `-X` keeps job steps out, `-n`
    # drops the header, `-P` makes the state the first field of each line.
    sacct -X -n -P -j "$JOBLIST" -o State > "$WORK/secondary.txt" 2>/dev/null || true

    set +e
    python -m src.cluster.wait_guard \
        --primary "$WORK/primary.txt" \
        --secondary "$WORK/secondary.txt" \
        --primary-source squeue --secondary-source sacct
    RC=$?
    set -e

    case "$RC" in
        0) echo "[remote_wait] poll $i: corroborated complete"; exit 0 ;;
        2) sleep "$INTERVAL" ;;
        *) echo "[remote_wait] poll $i: REFUSED (rc=$RC)" >&2; exit "$RC" ;;
    esac
done

echo "[remote_wait] gave up after $MAX_POLLS polls -- NOT a completion" >&2
exit 3
