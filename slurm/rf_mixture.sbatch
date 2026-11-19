#!/bin/bash
#SBATCH --job-name=sae_rf_mixture
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

# Amendment AB sections 3-4: the MIXING DISTRIBUTION behind r_f, and endpoint
# purity computed against the FITTED mixture rather than against a
# single-shared-p binomial (gotcha 23).
#
#
# USAGE:  sbatch slurm/rf_mixture.sbatch [n19|n39] [<m1_json> <out_json>]
#
# `n_trials` is read from the M1 artifact's `n_compare_seeds`, NOT from
# `n_seeds - 1`: those stopped being equal when anchors and comparison
# dictionaries became different sets (amendment AK).
#
# Reads only an rf_marginal M1 json. No alignment artifact is touched.
#
# Walltime is 3 h rather than 40 min because the N = 39 grid has twice the
# atoms and `profile_w0` refits at each of 41 constrained w0 values.

set -euo pipefail
MODE="${1:-n19}"
PROJECT=$PROJECT
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

case "$MODE" in
  n19) M1="$PROJECT/analysis/rf_marginal_m1_n19.json"
       OUT="$PROJECT/analysis/rf_mixture_n19.json" ;;
  n39) M1="$PROJECT/analysis/rf_marginal_m1_n39.json"
       OUT="$PROJECT/analysis/rf_mixture_n39.json" ;;
  *)   echo "unknown mode '$MODE' -- expected n19 or n39" >&2; exit 1 ;;
esac
if [[ $# -ge 3 ]]; then M1="$2"; OUT="$3"; fi

if [[ ! -f "$M1" ]]; then
    echo "no M1 artifact at $M1 -- run slurm/rf_marginal.sbatch $MODE first" >&2
    exit 1
fi

# Gotcha 15. A file named `_n39` that
# carries `n_compare_seeds = 19` is exactly the glob-hazard failure: the mixture
# would fit a 19-trial binomial and label the output N = 39.
EXPECT=19; [[ "$MODE" == "n39" ]] && EXPECT=39
python - "$M1" "$EXPECT" <<'PY'
import json
import sys
doc = json.loads(open(sys.argv[1]).read())
ident = doc["identity"]
got = ident.get("n_compare_seeds")
want = int(sys.argv[2])
if got is None:
    raise SystemExit(
        "REFUSING: %s carries no n_compare_seeds attr. Absent is not matching "
        "(gotcha 15) -- re-run rf_marginal from a checkout that writes it."
        % sys.argv[1])
if int(got) != want:
    raise SystemExit(
        "REFUSING: %s says n_compare_seeds=%s but this mode expects %d. "
        "A mixture fitted at the wrong number of trials produces a full table "
        "of believable numbers describing the wrong ensemble."
        % (sys.argv[1], got, want))
print("[mix.sbatch] identity OK: n_compare_seeds=%d, n_anchors=%s"
      % (want, ident.get("n_anchors")))
PY

python -m src.analysis.rf_mixture \
    --m1 "$M1" \
    --rule mnn \
    --thetas 0.30 0.50 0.65 0.70 0.75 \
    --out "$OUT"
