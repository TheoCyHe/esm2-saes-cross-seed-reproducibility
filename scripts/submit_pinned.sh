#!/bin/bash
# Gotcha 47's tooling (PREREGISTRATION.md §CT18-D4 item 6): clone the repository at one commit and
# run a submit command from inside the clone ONLY IF every check on the clone passes.
#
#   bash scripts/submit_pinned.sh <commit> <new clone dir> [<file>:<ERE> ...] -- <submit command ...>
#
# Checks, each of which refuses (exit 1) before the submit command starts:
#   the clone directory does not exist yet; the clone and checkout succeed; HEAD is the commit
#   asked for; the clone is clean (`git status --porcelain` empty, untracked files included);
#   every <file>:<ERE> names a file that exists and contains a match.
# Every check is `test || die`, never a test inside an `&&` list: under `set -e` a failing test
# that is not the list's last command does not exit (gotcha 47, second and fourth instances).
# REPO_URL overrides the source. The default is the address written on the next line; the machine
# that runs this needs read access to it.
set -euo pipefail
REPO_URL="${REPO_URL:-git@github.com:TheoCyHe/esm2-saes-cross-seed-reproducibility.git}"
die() { echo "REFUSING: $*" >&2; exit 1; }

[[ $# -ge 4 ]] || die "usage: submit_pinned.sh <commit> <clone dir> [<file>:<ERE> ...] -- <command ...>"
SHA="$1"; C="$2"; shift 2
CHECKS=()
while [[ $# -gt 0 && "$1" != "--" ]]; do CHECKS+=("$1"); shift; done
[[ "${1:-}" == "--" ]] || die "no '--' before the submit command"
shift
[[ $# -gt 0 ]] || die "no submit command after '--'"

[[ ! -e "$C" ]] || die "$C exists"
git clone -q "$REPO_URL" "$C" || die "git clone $REPO_URL failed"
cd "$C" || die "cannot enter $C"
git checkout -q "$SHA" || die "checkout of $SHA failed"
WANT=$(git rev-parse --verify -q "$SHA^{commit}") || die "$SHA is not a commit here"
HEAD_SHA=$(git rev-parse HEAD) || die "rev-parse HEAD failed"
# Redundant after a successful `git checkout <commit>`, so no test can make it fire (a
# mutation run, 2026-11-01, confirms it survives removal); kept as a guard, not counted as one.
[[ "$HEAD_SHA" == "$WANT" ]] || die "HEAD $HEAD_SHA is not $WANT"
DIRTY=$(git status --porcelain) || die "git status failed"
[[ -z "$DIRTY" ]] || die "the clone is dirty: $(echo "$DIRTY" | head -3 | tr '\n' ' ')"
for chk in ${CHECKS[@]+"${CHECKS[@]}"}; do
  f="${chk%%:*}"; pat="${chk#*:}"
  [[ "$f" != "$chk" && -n "$pat" ]] || die "check '$chk' is not <file>:<ERE>"
  [[ -f "$f" ]] || die "$f is absent at $WANT"
  grep -Eq -- "$pat" "$f" || die "$f has no match for /$pat/ at $WANT"
done
echo "PINNED_OK commit=$WANT clone=$C checks=${#CHECKS[@]}"
exec "$@"
