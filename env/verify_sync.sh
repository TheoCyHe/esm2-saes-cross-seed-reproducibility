#!/bin/bash
# Assert that the cluster checkout is EXACTLY where local/origin is, and fail
# loudly otherwise. Run after every sync.
#
# WHY THIS EXISTS. `git pull --ff-only` printed "Already up to date" while doing
# nothing for `main`, because the cluster checkout was on a feature branch
# tracking that branch. A categorical positive that reads as verified and is not
# is gotcha 20 in pure form -- so this DERIVES the verdict from the SHAs rather
# than trusting the message.
#
# Checks, each fatal:
#   1. the cluster is on the expected branch (default `main`)
#   2. its upstream is the matching remote branch, not some other branch
#   3. HEAD == origin/<branch> on the cluster
#   4. HEAD == the SHA passed in as $1 (the local HEAD)
#   5. the tree is clean, and `--porcelain` is echoed when it is not, so the
#      REASON is visible rather than just the boolean
#
# Usage, from the machine that holds the local checkout:
#   ssh <host> 'bash -s' < env/verify_sync.sh -- "$(git rev-parse HEAD)" [branch]
# or on the cluster:
#   bash env/verify_sync.sh <expected-sha> [branch]
set -uo pipefail

EXPECTED_SHA="${1:-}"
BRANCH="${2:-main}"
REPO="${REPO:-$HOME/esm2-saes-cross-seed-reproducibility}"

cd "$REPO" || { echo "FATAL: no checkout at $REPO" >&2; exit 1; }

fail=0
say() { printf '  %-34s %s\n' "$1" "$2"; }

ACTUAL_BRANCH=$(git rev-parse --abbrev-ref HEAD)
UPSTREAM=$(git rev-parse --abbrev-ref '@{upstream}' 2>/dev/null || echo "(none)")
HEAD_SHA=$(git rev-parse HEAD)
ORIGIN_SHA=$(git rev-parse "origin/$BRANCH" 2>/dev/null || echo "(missing)")
DIRTY=$(git status --porcelain)

say "repo" "$REPO"
say "branch" "$ACTUAL_BRANCH"
say "upstream" "$UPSTREAM"
say "HEAD" "$HEAD_SHA"
say "origin/$BRANCH" "$ORIGIN_SHA"

if [ "$ACTUAL_BRANCH" != "$BRANCH" ]; then
    echo "FATAL: on branch '$ACTUAL_BRANCH', expected '$BRANCH'." >&2
    echo "       A pull on the wrong branch reports success and moves nothing." >&2
    fail=1
fi
if [ "$UPSTREAM" != "origin/$BRANCH" ]; then
    echo "FATAL: upstream is '$UPSTREAM', expected 'origin/$BRANCH'." >&2
    fail=1
fi
if [ "$HEAD_SHA" != "$ORIGIN_SHA" ]; then
    echo "FATAL: HEAD != origin/$BRANCH -- the checkout is not synced." >&2
    fail=1
fi
if [ -n "$EXPECTED_SHA" ] && [ "$HEAD_SHA" != "$EXPECTED_SHA" ]; then
    echo "FATAL: HEAD != the expected (local) SHA $EXPECTED_SHA." >&2
    fail=1
fi
if [ -n "$DIRTY" ]; then
    echo "FATAL: working tree is DIRTY. git status --porcelain says:" >&2
    echo "$DIRTY" | sed 's/^/       /' >&2
    echo "       NB git_dirty() counts UNTRACKED files, so this would stamp" >&2
    echo "       dirty provenance on every artifact written from here." >&2
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    echo "SYNC VERIFICATION FAILED" >&2
    exit 1
fi
echo "SYNC VERIFIED: $BRANCH @ ${HEAD_SHA:0:12}, upstream origin/$BRANCH, tree clean"
