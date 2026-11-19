#!/bin/bash
#SBATCH --job-name=sae_design3_join
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

# THE DESIGN-3 JOIN. PREREGISTRATION amendment AR; docs/SPECIFICATION.md.
#
# USAGE:  sbatch slurm/design3_join.sbatch [full|nofeat]
#
#   full    every arm, including AL8b's feature-side pair. That arm streams
#           each seed's row index protein by protein and is the only expensive
#           part of the job.
#   nofeat  everything else; the omission is RECORDED in the output rather
#           than leaving a pre-declared arm silently absent.
#   as      amendment AS: the same join, plus the three-interval comparison
#           (AS 2), the outcome-scale marginal effect at every theta (AS 3),
#           the d_fs share on the estimand's own instrument (AS 4) and F1's
#           model-free panel (AS 10). NEW output path; design3_join_ar.json
#           is untouched.
#   as_nofeat  as above, without AL8b's feature-side arm.
#   at      amendment AT: the same join with AC10's smooth arm RE-POINTED at
#           the AR9-ALIGNED artifact (a defect correction -- the join-time
#           resolver aligns rows and cannot repair mu_s values computed from
#           mis-paired seeds INSIDE smooth_mu), plus the outcome-scale bound
#           under the beta-binomial at every theta (AT A2), FLAT's tolerance
#           converted into pp of r_f with reachability re-demonstrated at the
#           real cell sizes (AT A3), and gamma's full treatment across the
#           grid (AT A4). NEW output path; design3_join_as.json is untouched
#           and stays on disk as the superseded record.
#   at_nofeat  as above, without AL8b's feature-side arm.
#   au      amendment AU: `at` plus the four-arm comparison under BOTH
#           likelihoods (AU section 1a -- the three sensitivity arms are
#           now fit with dispersion too, so `includes_zero` is answerable
#           per arm) and the PERMUTATION null for F1's panel (AU section
#           3 -- AT12's binomial-null reachability check used a null the
#           data does not resemble). NEW output path; design3_join_at.json
#           is untouched and its PRIMARY is the bit-identity reference.
#   au_nofeat  as above, without AL8b's feature-side arm. The four-arm
#           comparison then has three arms and RECORDS the omission.
#   aw      amendment AW section 4 item 3: the same inputs as `au`/`av`
#           (item 1's decile/dispersion breakdown by firing-rate stratum is
#           already in design3_join_av.json and is NOT recomputed here --
#           `--stratified-panel-permutation` is left OFF). Adds ONLY a refit
#           of the primary model with `keep` restricted to fire_rate in
#           [1e-3, 3e-3) -- the stratum AV18 found still carrying the
#           panel's residual excess after blocking on d_fs. EXPLORATORY
#           (§2a): cannot become primary under any outcome. NEW output path;
#           design3_join_au.json is untouched.
#   aw_nofeat  as above, without AL8b's feature-side arm -- sufficient for
#           the refit, which does not touch the feature-side arm at all.
#   cd      amendment CD item 2: `aw` plus the stratum refit at EVERY
#           qualifying theta (`--firing-stratum-refit-all-theta`). AW's refit
#           exists at the reference theta only, and a one-threshold
#           coefficient cannot carry a reportable claim under the standing
#           sensitivity-curve requirement. EXPLORATORY, inheriting AW §2a
#           unchanged: it cannot become primary at ANY theta. NEW output path;
#           design3_join_aw.json is untouched.
#   cd_nofeat  as above, without AL8b's feature-side arm. NOTE that this mode
#           does NOT open the evaluation cache at all -- the eval cache is
#           read only inside the feature-side arm -- so the CD curve is NOT
#           on the eval.h5 purge clock (amendment CD item 2, correcting the
#           governing brief's premise).
#   cq      amendment CQ: `at`'s eight inputs verbatim, re-run at the commit
#           that aligns the AK8 smooth companion arm's seed axis (corrections
#           entry 218), WITHOUT AL8b's feature-side arm -- $SCRATCH/cache/eval.h5
#           and the row-index cache were purged (established 2026-09-16 by
#           `ls`, CQ0 item 2), and the primary is identical across the
#           feat/nofeat split (cd_nofeat and aw_nofeat both reproduced au's
#           full-arm primary at 0 disagreements). The PRIMARY is asserted
#           bit-identical against design3_join_at.json (CQ-P1). NEW output
#           path; design3_join_at.json is untouched.
#
# *** THIS JOB DELIBERATELY CROSSES THE X7 FIREWALL. *** It is the authorised
# crossing: scoped to design 3's own inputs plus AK8's companion arm, labelled
# in the output, and confined to its own script and its own job.
#
# THE PATHS ARE THE HAZARD, not the compute. Three of them:
#   1. the smooth arm MUST read the _fixfp artifacts -- the pre-AO6 files are
#      still on disk and were built on a misaligned protein grouping (AO6).
#      The python refuses a --gtilde-smooth without the marker, so this script
#      cannot quietly point at the wrong generation;
#   2. the smooth artifacts live on SCRATCH, purged on the policy in
#      CLAUDE.md's Cluster section (read `getquota`), and gtilde is now read
#      from $PROJECT (the GTILDE line below). If one is gone the job must
#      FAIL, not regenerate it silently;
#   3. every output goes to a NEW path, so no canary target moves.
#
# RESOURCES: taken from m4_confound's profile, which reads the same label
# store and row-index cache. The fits themselves are trivial; assemble_covariates
# is what needs the memory. Record the real figure from `jobstats` afterwards.

set -euo pipefail

MODE="${1:-full}"
PROJECT=$PROJECT
SCRATCH=$SCRATCH

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh

ALLTHETA="$PROJECT/analysis/rf_counts_all_theta_n39.h5"
REFCOUNTS="$PROJECT/analysis/rf_counts_theta070_n39.h5"
M1="$PROJECT/analysis/rf_marginal_m1_n39.json"
GTILDE="$PROJECT/analysis/gtilde_compliant_b_ratio.h5"
# The smooth arm's generation is MODE-DEPENDENT from AT onward. Modes up to
# `as` read the pre-AR9 `_fixfp` artifact so their numbers stay comparable to
# AR's; `at` reads the aligned rebuild. Both are admissible generations; the
# pre-AO6 files are not, and the marker check below still refuses them.
GTSMOOTH="$PROJECT/analysis/gtilde_compliant_b_ratio_smooth_fixfp.h5"
NULLRED="$PROJECT/analysis/null_pilot_compliant.h5"
DFS="$PROJECT/analysis/d_fs.h5"
DEGEN="$PROJECT/analysis/degenerate_argmax_flag.h5"
LABELS="$PROJECT/analysis/residue_labels_subtyped_untruncated_pgate5_f67485d.h5"
ROWIDX="$SCRATCH/row_index_cache"
CKPTS="$PROJECT/sae_ckpts"
EVALCACHE="$SCRATCH/cache/eval.h5"

CMP=()
EXTRA=()
case "$MODE" in
  full)   OUT="$PROJECT/analysis/design3_join_ar.json";        FEAT=() ;;
  nofeat) OUT="$PROJECT/analysis/design3_join_ar_nofeat.json"; FEAT=(--skip-feature-side) ;;
  as)     OUT="$PROJECT/analysis/design3_join_as.json";        FEAT=() ;;
  as_nofeat) OUT="$PROJECT/analysis/design3_join_as_nofeat.json"; FEAT=(--skip-feature-side) ;;
  av|av_nofeat)
    # AMENDMENT AV section 5c. `au` plus the STRATIFIED permutation for F1's
    # panel: AU14 left a 1.27x decile-spread excess unexplained, and AU's
    # permutation controls firing rate (cells are within strata) but NOT
    # homology density, which is worth +3.12 pp per SD against log M's bound of
    # <=0.3635 pp. The only input change is the flag; every path is AU's, and
    # the PRIMARY is asserted bit-identical against AU's artifact.
    if [[ "$MODE" == "av_nofeat" ]]; then
        OUT="$PROJECT/analysis/design3_join_av_nofeat.json"
        FEAT=(--skip-feature-side)
    else
        OUT="$PROJECT/analysis/design3_join_av.json"
        FEAT=()
    fi
    GTSMOOTH="$PROJECT/analysis/gtilde_compliant_b_ratio_smooth_aligned.h5"
    EXTRA=(--stratified-panel-permutation)
    CMP=(--compare-primary-to "$PROJECT/analysis/design3_join_au.json")
    ;;
  au|au_nofeat)
    if [[ "$MODE" == "au_nofeat" ]]; then
        OUT="$PROJECT/analysis/design3_join_au_nofeat.json"
        FEAT=(--skip-feature-side)
    else
        OUT="$PROJECT/analysis/design3_join_au.json"
        FEAT=()
    fi
    GTSMOOTH="$PROJECT/analysis/gtilde_compliant_b_ratio_smooth_aligned.h5"
    # AU adds emissions and turns dispersion on for the SENSITIVITY arms;
    # the primary must not move, and AT's artifact is the reference.
    CMP=(--compare-primary-to "$PROJECT/analysis/design3_join_at.json")
    ;;
  aw|aw_nofeat)
    if [[ "$MODE" == "aw_nofeat" ]]; then
        OUT="$PROJECT/analysis/design3_join_aw_nofeat.json"
        FEAT=(--skip-feature-side)
    else
        OUT="$PROJECT/analysis/design3_join_aw.json"
        FEAT=()
    fi
    GTSMOOTH="$PROJECT/analysis/gtilde_compliant_b_ratio_smooth_aligned.h5"
    # Only the new EXPLORATORY refit is added; the primary must not move,
    # and AU's artifact (the same primary av also reproduced) is the
    # reference.
    EXTRA=(--firing-stratum-refit 1e-3 3e-3)
    CMP=(--compare-primary-to "$PROJECT/analysis/design3_join_au.json")
    ;;
  cd|cd_nofeat)
    if [[ "$MODE" == "cd_nofeat" ]]; then
        OUT="$PROJECT/analysis/design3_join_cd_nofeat.json"
        FEAT=(--skip-feature-side)
    else
        OUT="$PROJECT/analysis/design3_join_cd.json"
        FEAT=()
    fi
    GTSMOOTH="$PROJECT/analysis/gtilde_compliant_b_ratio_smooth_aligned.h5"
    # AW's stratum, unchanged, plus the across-theta curve. The primary must
    # not move: CD adds an exploratory arm and nothing else, and AU's artifact
    # is the same reference av and aw both reproduced.
    EXTRA=(--firing-stratum-refit 1e-3 3e-3 --firing-stratum-refit-all-theta
           --post-hoc-interaction)
    CMP=(--compare-primary-to "$PROJECT/analysis/design3_join_au.json")
    ;;
  at|at_nofeat)
    # Written as an explicit if rather than `[[ ... ]] && x=y`: under
    # `set -e` the AND-OR form's behaviour on a false test is a bash-version
    # question, and this script must not depend on the answer.
    if [[ "$MODE" == "at_nofeat" ]]; then
        OUT="$PROJECT/analysis/design3_join_at_nofeat.json"
        FEAT=(--skip-feature-side)
    else
        OUT="$PROJECT/analysis/design3_join_at.json"
        FEAT=()
    fi
    # AMENDMENT AT A1: the re-point. This is the ONE line that makes `at`
    # different in its inputs, and it is here rather than in the python so
    # that the python's marker gate still sees a name it must approve.
    GTSMOOTH="$PROJECT/analysis/gtilde_compliant_b_ratio_smooth_aligned.h5"
    # The PRIMARY must not move: AT changes a sensitivity arm's input and
    # adds emissions, nothing else. Checked against AS's artifact rather than
    # hoped for -- and against the FULL-arm one, since the primary is
    # identical across the feat/nofeat split.
    CMP=(--compare-primary-to "$PROJECT/analysis/design3_join_as.json")
    ;;
  cq)
    # AMENDMENT CQ. Same eight inputs as `at`; the feature-side arm is skipped
    # (its inputs are purged, CQ0 item 2) and the omission is RECORDED in the
    # output. The primary must not move: CQ aligns one companion arm and adds
    # an identity key, nothing else, and AT's artifact is the reference.
    OUT="$PROJECT/analysis/design3_join_cq.json"
    FEAT=(--skip-feature-side)
    GTSMOOTH="$PROJECT/analysis/gtilde_compliant_b_ratio_smooth_aligned.h5"
    CMP=(--compare-primary-to "$PROJECT/analysis/design3_join_at.json")
    ;;
  *) echo "unknown mode '$MODE' -- expected full, nofeat, as, as_nofeat, at, at_nofeat, au, au_nofeat, av, av_nofeat, aw, aw_nofeat, cd, cd_nofeat or cq" >&2; exit 1 ;;
esac

# Gotcha 15: the python asserts the identity attrs. But a
# path that is simply GONE (the scratch purge, CLAUDE.md Cluster) has to fail here,
# loudly, rather than surfacing as a confusing h5 error 40 minutes in.
for f in "$ALLTHETA" "$REFCOUNTS" "$M1" "$GTILDE" "$GTSMOOTH" "$NULLRED" \
         "$DFS" "$DEGEN"; do
    if [[ ! -f "$f" ]]; then
        echo "MISSING INPUT: $f -- refusing to regenerate anything silently" >&2
        exit 1
    fi
done
if [[ "$MODE" == "full" || "$MODE" == "as" || "$MODE" == "at" || "$MODE" == "au" || "$MODE" == "av" || "$MODE" == "aw" || "$MODE" == "cd" ]]; then
    for f in "$LABELS" "$EVALCACHE"; do
        if [[ ! -f "$f" ]]; then
            echo "MISSING INPUT for the feature-side arm: $f" >&2
            exit 1
        fi
    done
    if [[ ! -d "$ROWIDX" ]]; then
        echo "MISSING row index cache: $ROWIDX" >&2
        exit 1
    fi
fi

# Everything this session writes goes to a NEW path. Refusing an existing
# output keeps two generations of one object from mixing silently, which is
# the rule ao_rebuild_smooth adopted for the same reason.
if [[ -e "$OUT" ]]; then
    echo "$OUT already exists -- refusing to overwrite" >&2
    exit 1
fi

# The smooth arm's generation, asserted in shell as well as in python. A
# scripted edit that silently stopped matching is a defect this project has
# shipped (gotcha 13 extension 4), so the claim is checked where it is made.
#
# TWO admissible markers from AT onward, and the check is per MODE rather than
# a widened wildcard: a gate that accepts both everywhere would no longer
# detect a mode reading the wrong generation, which is the whole failure AT
# A1 is correcting. So `at` must read `_aligned` and the others must read
# `_fixfp` -- each mode's expectation is asserted, not merely "one of two".
case "$MODE" in
  at|at_nofeat|au|au_nofeat|av|av_nofeat|aw|aw_nofeat|cd|cd_nofeat|cq) WANT_MARKER="_aligned" ;;
  *)            WANT_MARKER="_fixfp" ;;
esac
case "$GTSMOOTH" in
  *"$WANT_MARKER".h5) : ;;
  *) echo "GTSMOOTH=$GTSMOOTH does not carry the $WANT_MARKER marker "\
"required by mode $MODE (AO6 / amendment AT A1)" >&2
     exit 1 ;;
esac
echo "[join.sbatch] smooth arm generation: $WANT_MARKER ($(basename "$GTSMOOTH"))"

echo "[join.sbatch] mode=$MODE  out=$OUT"

python -m src.analysis.design3_join \
    --rf-counts-all-theta "$ALLTHETA" \
    --rf-counts "$REFCOUNTS" \
    --m1-json "$M1" \
    --gtilde "$GTILDE" \
    --gtilde-smooth "$GTSMOOTH" \
    --null-reductions "$NULLRED" \
    --d-fs "$DFS" \
    --degenerate-flag "$DEGEN" \
    --labels "$LABELS" \
    --row-index-cache "$ROWIDX" \
    --ckpts "$CKPTS" \
    --eval-cache "$EVALCACHE" \
    --scored-row-set label_store \
    --k 32 \
    "${FEAT[@]}" \
    "${EXTRA[@]+"${EXTRA[@]}"}" \
    "${CMP[@]+"${CMP[@]}"}" \
    --out "$OUT"
