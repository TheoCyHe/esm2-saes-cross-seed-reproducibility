#!/bin/bash
#SBATCH --job-name=sae_cc_canary
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# PREREGISTRATION amendment CC, item 3 -- canary coverage for the determinism
# probe's eight reduced comparison documents.
#
# THE ORDER IS THE SUBSTANCE, and it is enforced here rather than described
# (the same structure as bs_canary_sidecar.sbatch, whose reasoning this
# inherits): verify the COMMITTED set first, then capture with
# --assert-addition-only, then verify the FULL set. A capture that ran before
# the pre-verify could not distinguish "the addition is clean" from "something
# else moved at the same time".
#
# NO GPU. Nothing here gathers activations or recomputes anything; the job
# hashes files. 16G is generous -- the eight additions total 61,006 bytes and
# the pre-existing targets are streamed by `describe`, not loaded.
#
# WHY THESE EIGHT (amendment CC-C3), ON A CORRECTED GROUND. The proposing brief
# said they become "the SOLE surviving evidence for three committed figures"
# once $SCRATCH/determinism_probe/ purges. THAT IS OVERSTATED: the three figures
# are committed in git in four places (src/analysis/canary.py's `null`
# tolerance, CLAUDE.md gotcha 35, docs/GOTCHAS.md #35, docs/PROJECT.md's narrative
# preamble), and `getquota` reports Backup: Yes for project against No for scratch.
# WHAT IS TRUE: they become the sole surviving DERIVATION -- the only place the
# 446,308/614,400 decomposition behind "up to 72%", the per-array 1+4 behind
# "~5 of 434,380,800", git_sha 002673b and both source paths can be read. Git
# holds the numbers; these hold where they came from, which gotcha 35's
# control-measured tolerance needs.
#
# AND THE CONTROL CANNOT BE RE-PRODUCED AFTER ~2026-08-23:
# null_determinism_probe.sbatch:77 passes --eval-cache, and cache/eval.h5 purges
# then. That is why `determinism` is its own arm rather than part of
# build_inputs -- every other file-integrity arm says "the code may no longer
# produce this"; this one says "the code cannot".

set -euo pipefail

PROJECT=$PROJECT
SCRATCH=$SCRATCH
ROOT="$PROJECT/analysis"
SCRATCH_ROOT="$SCRATCH/null_raw"
# THE MANIFEST IS NOT WRITTEN INTO THE CLUSTER CHECKOUT, and this DEPARTS from
# bs_canary_sidecar.sbatch deliberately (amendment CC-C4). BS pointed --capture
# at `canary/baseline.json` relative to $SLURM_SUBMIT_DIR, which rewrites a
# TRACKED file inside the checkout and makes `git status --porcelain` non-empty
# for the rest of the job: see CLAUDE.md's reproducibility hygiene
# (`git_dirty()` counts untracked files) and gotcha 31.
# The cluster CANNOT push in any case, so a manifest written here has to be
# carried back by hand whichever path it takes.
#
# CORRECTED AFTER RUN 22585711 (corrections entry 184): the first version of
# this file wrote the new manifest to /tmp, and **/tmp ON A COMPUTE NODE IS
# NODE-LOCAL**. Job 22585711 COMPLETED, every step passed, and its output was
# unreachable from the login node -- established rather than assumed, by
# `sacct` naming the node (<node>) and by a control showing login-node
# /tmp is itself readable and writable while the file was absent from it. The
# capture had to be re-run. **A path that keeps the checkout clean is not
# thereby a path the result can be COLLECTED from**, and "not in the checkout"
# was solved while "reachable afterwards" was silently dropped.
# $PROJECT/analysis/adhoc/ satisfies BOTH: it is shared, it is durable, and
# it is the documented place for job-adjacent output (docs/CODEBASE.md §7.13).
COMMITTED=/tmp/cc_baseline_committed.json   # read only within this job, same node
NEW="$PROJECT/analysis/adhoc/cc_baseline_new_${SLURM_JOB_ID}.json"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

if [[ -n "$(git status --porcelain)" ]]; then
    echo "REFUSING: cluster checkout is dirty BEFORE this job did anything" >&2
    git status --porcelain >&2
    exit 1
fi

# Literal paths, not $PROJECT-expanded-inside-Python: gotcha 32a, amendment
# BW-B3. Here the shell DOES have them (they are set above, in this file), but
# the eight are listed explicitly so a missing one is named rather than counted.
echo "=== STEP 0: the eight additions are present BEFORE any capture ==="
missing=0
for f in determ_AB_red determ_AB_raw determ_AC_red determ_AC_raw \
         determ_A_vs_ref_red determ_A_vs_ref_raw \
         determ_C_vs_ref_red determ_C_vs_ref_raw; do
    if [[ -f "$ROOT/$f.json" ]]; then
        printf '  PRESENT %-24s %8s bytes  %s\n' "$f.json" \
            "$(stat -c %s "$ROOT/$f.json")" \
            "$(sha256sum "$ROOT/$f.json" | cut -c1-16)"
    else
        echo "  MISSING $f.json" >&2
        missing=1
    fi
done
if [[ "$missing" -ne 0 ]]; then
    echo "REFUSING: at least one addition is absent; a capture would baseline it \
as ABSENT and the arm would read as covered when it is not" >&2
    exit 1
fi

# The expected values, MEASURED 2026-08-17 from the local session and written
# down here rather than computed, so this step can DISAGREE with the artifacts
# (gotchas 13 and 32b).
echo
echo "=== STEP 0b: the additions match the bytes recorded in amendment CC-C4 ==="
python - "$ROOT" <<'PY'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
EXPECTED = {
    "determ_AB_red.json":       (11243, "87ba692536ad219e"),
    "determ_AB_raw.json":       ( 2160, "e398154d9dbd9499"),
    "determ_AC_red.json":       (11586, "3cd79a8e46a9ab3c"),
    "determ_AC_raw.json":       ( 2713, "fa05c69484965654"),
    "determ_C_vs_ref_red.json": (14208, "59a8b636af48d496"),
    "determ_C_vs_ref_raw.json": ( 2315, "69a32ee3fa64f8fc"),
    "determ_A_vs_ref_red.json": (14389, "78064a32fe3e5781"),
    "determ_A_vs_ref_raw.json": ( 2392, "d3b8eb0d04ce0427"),
}
bad = []
total = 0
for name, (nbytes, prefix) in sorted(EXPECTED.items()):
    p = root / name
    b = p.read_bytes()
    total += len(b)
    got = hashlib.sha256(b).hexdigest()
    ok = (len(b) == nbytes and got.startswith(prefix))
    print(f"  {'OK  ' if ok else 'FAIL'} {name:<26} {len(b):>6} bytes  {got[:16]}")
    if not ok:
        bad.append((name, nbytes, len(b), prefix, got[:16]))
print(f"  total {total} bytes (amendment CC-C4 records 61006)")
if total != 61006:
    bad.append(("__total__", 61006, total, "", ""))
if bad:
    print("MISMATCH against the values amendment CC-C4 wrote down:", bad,
          file=sys.stderr)
    raise SystemExit(1)
PY

echo
echo "=== STEP 1: verify the COMMITTED target set, BEFORE any capture ==="
echo "--- this reads the manifest as committed at HEAD: the 31 targets ---"
echo "--- predating CC. The eight additions are NOT in it and must not be. ---"
git show HEAD:canary/baseline.json > "$COMMITTED"
python -m src.analysis.canary --root "$ROOT" --scratch-root "$SCRATCH_ROOT" \
    --manifest "$COMMITTED"

echo
echo "=== STEP 2: capture, with addition-only ASSERTED (not merely hoped) ==="
cp "$COMMITTED" "$NEW"
python -m src.analysis.canary --root "$ROOT" --scratch-root "$SCRATCH_ROOT" \
    --manifest "$NEW" --capture --assert-addition-only

echo
echo "=== STEP 3: verify the FULL set, including the eight additions ==="
python -m src.analysis.canary --root "$ROOT" --scratch-root "$SCRATCH_ROOT" \
    --manifest "$NEW"

echo
echo "=== STEP 4: the checkout is STILL CLEAN (gotcha 31 -- assert on exit) ==="
if [[ -n "$(git status --porcelain)" ]]; then
    echo "FAILED: this job dirtied the cluster checkout" >&2
    git status --porcelain >&2
    exit 1
fi
echo "  clean"

echo
echo "=== STEP 5: the manifest is readable OFF this node (entry 184) ==="
# A file the job cannot be shown to have left behind on SHARED storage is a
# file the next step cannot collect. Assert it here rather than discover it
# from a failed scp: $PROJECT is NFS, so a stat that succeeds under a path
# beginning <cluster-fs>/project is reachable from the login node.
case "$NEW" in
    <cluster-fs>/project/*) ;;
    *) echo "FAILED: \$NEW is not on shared project storage: $NEW" >&2; exit 1;;
esac
if [[ ! -s "$NEW" ]]; then
    echo "FAILED: $NEW is missing or empty after capture" >&2; exit 1
fi
ls -l "$NEW"
python -c "import json,sys; d=json.load(open(sys.argv[1])); \
print('  targets in the written manifest:', len(d['targets']))" "$NEW"

echo
echo "ALL STEPS COMPLETED"
echo "THE NEW MANIFEST IS AT: $NEW"
echo "It was deliberately NOT written into the checkout, and NOT to /tmp, which"
echo "is node-local on a compute node (entry 184). The cluster cannot push"
echo "(read-only deploy key, by design), so retrieve it and commit from the local"
echo "checkout:  scp bouchet:$NEW canary/baseline.json"
