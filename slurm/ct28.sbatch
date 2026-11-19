#!/bin/bash
#SBATCH --job-name=ct28
#SBATCH --output=logs/%x_%j.out
# §CT28: threshold, threshold-free and shared-scale rescoring of the existing arms.
# Resources on the sbatch line per step:
#   arm, sffit, rules, widthrules, rim   CPU, day (rim: 16 CPUs, 128G)
#   sf                                   gpu (or gpu_devel), --gpus=rtx_5000_ada:1 (gotcha 36)
# Usage, from a pinned clean clone: sbatch <resources> slurm/ct28.sbatch <ARM> <STEP>
#   ARM  = k64 | k64n9 | k32n19 | w2560 | w10240 | w20480 | w40960
#          | k64_w2560 | k64_w40960 | k16_w10240 | k128_w10240   (§CT29-F item 5, n_compare = 9)
#   STEP = arm | sf | sffit (k64n9: seeds 0-9 of the k64 s_f array, §CT9-D D4) | rim (k64, grid) | rules (k64, grid) | widthrules (w*)

set -euo pipefail
ARM="${1:?ARM: k64|k64n9|k32n19|w2560|w10240|w20480|w40960|k64_w2560|k64_w40960|k16_w10240|k128_w10240}"
STEP="${2:?STEP: arm|sf|sffit|rim|rules|widthrules}"
PROJECT=$PROJECT
A="$PROJECT/analysis"
O="$A/ct"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
[[ -z "$(git status --porcelain)" ]] || { echo "checkout dirty before the run" >&2; exit 1; }
echo "checkout $(git rev-parse HEAD) arm=$ARM step=$STEP host=$(hostname)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W="${SLURM_CPUS_PER_TASK:-8}"
refuse_existing() { for f in "$@"; do [[ ! -e "$f" ]] || { echo "$f exists -- refusing" >&2; exit 1; }; done; }

USE=(); EXTRA=()
case "$ARM" in
  k64|k64n9)
    JOIN=(--sim "$O/jackknife_match_k64_ct.h5" --rf-counts-all-theta "$O/rf_counts_all_theta_k64_ct.h5"
          --m1-json "$O/rf_marginal_m1_k64_ct.json" --gtilde "$O/gtilde_k64_b_ratio_ct.h5"
          --null-reductions "$O/null_pilot_k64_ct.h5" --d-fs "$O/d_fs_k64_ct.h5"
          --config configs/k64/mvp_k64.yaml)
    CKROOT="$PROJECT/sae_ckpts_k64"; SEEDS=$(seq 0 19)
    if [[ "$ARM" == "k64" ]]; then
        TARGET=0.495814
        EXTRA=(--stored-slope "$O/slope_k64_ct.json" --verify-scale
               --act-counts "$O/activation_correlation_counts_k64_ct.h5"
               --act-reference-fe 2.403689791008184)
    else
        TARGET=0.4957942708333333; USE=(--use $(seq 0 9))
    fi ;;
  k32n19)
    JOIN=(--sim "$O/jackknife_match_ct.h5" --rf-counts-all-theta "$A/rf_counts_all_theta_n39.h5"
          --m1-json "$A/rf_marginal_m1_n39.json" --gtilde "$A/gtilde_compliant_b_ratio.h5"
          --null-reductions "$A/null_pilot_compliant.h5" --d-fs "$A/d_fs.h5"
          --config configs/mvp.yaml)
    TARGET=0.495814; USE=(--use $(seq 0 19)); EXTRA=(--reference-fe 0.25557598391144754) ;;
  w2560|w10240|w20480|w40960)
    JOIN=(--sim "$O/jackknife_match_${ARM}_ct.h5" --rf-counts-all-theta "$A/cs/rf_counts_all_theta_${ARM}_n9.h5"
          --m1-json "$A/cs/rf_marginal_m1_${ARM}_n9.json" --gtilde "$O/gtilde_${ARM}_b_ratio_ct.h5"
          --null-reductions "$O/null_pilot_${ARM}_ct.h5" --d-fs "$O/d_fs_${ARM}_ct.h5"
          --config "configs/width/mvp_${ARM}.yaml")
    CKROOT="$PROJECT/sae_ckpts_width/$ARM"; SEEDS=$(seq 0 9)
    TARGET=0.4957942708333333; EXTRA=(--stored-slope "$O/slope_${ARM}_ct.json")
    [[ "$ARM" == "w10240" ]] && EXTRA+=(--assert-theta-star-is-070) ;;
  k64_w2560|k64_w40960|k16_w10240|k128_w10240)
    # §CT29: every input is the arm's own, from ct_pass.sbatch, in analysis/ct/;
    # θ* matched to the k = 32 widths' n_compare = 9 reference (§CT29-F item 5)
    JOIN=(--sim "$O/jackknife_match_${ARM}_ct.h5" --rf-counts-all-theta "$O/rf_counts_all_theta_${ARM}_ct.h5"
          --m1-json "$O/rf_marginal_m1_${ARM}_ct.json" --gtilde "$O/gtilde_${ARM}_b_ratio_ct.h5"
          --null-reductions "$O/null_pilot_${ARM}_ct.h5" --d-fs "$O/d_fs_${ARM}_ct.h5"
          --config "configs/grid/mvp_${ARM}.yaml")
    # ROOT_KIND and EXCLUDE as ct_pass.sbatch takes them (§CT29-D item 2; §CT29-F item 7)
    case "${ROOT_KIND:-rtx}" in
      rtx)  CKROOT="$PROJECT/sae_ckpts_ct29/$ARM" ;;
      h200) CKROOT="$PROJECT/sae_ckpts_ct29_h200/$ARM" ;;
      *) echo "ROOT_KIND must be rtx or h200" >&2; exit 1 ;;
    esac
    SEEDS=$(for s in $(seq 0 9); do [[ " ${EXCLUDE:-} " == *" $s "* ]] || echo "$s"; done)
    [[ -z "${EXCLUDE:-}" ]] || echo "excluding seeds ${EXCLUDE}"
    TARGET=0.4957942708333333; EXTRA=(--stored-slope "$O/slope_${ARM}_ct.json") ;;
  *) echo "unknown ARM $ARM" >&2; exit 1 ;;
esac

case "$STEP" in
  arm)
    OUT="$O/ct28_arm_${ARM}_ct.json"; refuse_existing "$OUT"
    python -m src.analysis.ct28_rescore arm --arm-name "$ARM" "${JOIN[@]}" ${USE[@]+"${USE[@]}"} \
        --target "$TARGET" ${EXTRA[@]+"${EXTRA[@]}"} --workers "$W" --out "$OUT"
    sha256sum "$OUT" ;;
  sf)
    [[ "$ARM" != "k64n9" && "$ARM" != "k32n19" ]] || { echo "s_f is declared for k64 and the widths" >&2; exit 1; }
    OUT="$O/ct28_sf_${ARM}_ct.h5"; refuse_existing "$OUT"
    python -m src.analysis.ct28_rescore sf --arm-name "$ARM" \
        --ckpts $(for s in $SEEDS; do echo "$CKROOT/seed_$s"; done) \
        --sim "$O/jackknife_match_${ARM}_ct.h5" --out "$OUT"
    sha256sum "$OUT" ;;
  sffit)
    [[ "$ARM" != "k32n19" ]] || { echo "s_f is declared for k64, k64n9, the widths and the grid" >&2; exit 1; }
    OUT="$O/ct28_sffit_${ARM}_ct.json"; refuse_existing "$OUT"
    SF="$O/ct28_sf_${ARM}_ct.h5"
    # §CT9-D item D4: k64n9 reads seeds 0-9 of the k64 array by label; CPU, no GPU step
    if [[ "$ARM" == "k64n9" ]]; then SF="$O/ct28_sf_k64_ct.h5"; fi
    python -m src.analysis.ct28_rescore sffit --arm-name "$ARM" "${JOIN[@]}" ${USE[@]+"${USE[@]}"} \
        --oneway-h5 "$SF" --workers "$W" --out "$OUT"
    sha256sum "$OUT" ;;
  rim)
    case "$ARM" in k64|k64_w2560|k64_w40960|k16_w10240|k128_w10240) ;;
      *) echo "rim is for k64 and the §CT29 grid arms" >&2; exit 1 ;; esac
    RJ="$O/rule_interior_mass_${ARM}_ct.json"; RH="$O/rule_interior_mass_${ARM}_ct.h5"
    RS="$O/rule_interior_mass_sim_${ARM}_ct.h5"
    refuse_existing "$RJ" "$RH" "$RS"
    python -m src.analysis.rule_interior_mass --ckpt-root "$CKROOT" --anchors $SEEDS \
        --out-json "$RJ" --out-h5 "$RH" --out-sim-h5 "$RS"
    sha256sum "$RJ" "$RH" "$RS" ;;
  rules)
    case "$ARM" in k64|k64_w2560|k64_w40960|k16_w10240|k128_w10240) ;;
      *) echo "rules is for k64 and the §CT29 grid arms" >&2; exit 1 ;; esac
    OUT="$O/ct28_rules_${ARM}_ct.json"; refuse_existing "$OUT"
    if [[ "$ARM" == "k64" ]]; then
        RT=0.21698858963815787
    else
        # §CT29-F item 5: each arm's own rule-1 rate at 0.70 is its rules' θ* target,
        # as §CT28-F item 5 set k = 64's (its rf_marginal mnn mean at 0.70)
        RT=$(python -c "import json,sys;print(repr(json.load(open(sys.argv[1]))['marginals_by_theta']['0.70']['mnn']['mean']))" \
            "$O/rf_marginal_m1_${ARM}_ct.json")
    fi
    echo "rules θ* target = $RT"
    python -m src.analysis.ct28_rescore rules --arm-name "$ARM" "${JOIN[@]}" \
        --target "$RT" --rules-sim-h5 "$O/rule_interior_mass_sim_${ARM}_ct.h5" \
        --rules-h5 "$O/rule_interior_mass_${ARM}_ct.h5" --rf-counts-070 "$O/rf_counts_theta070_${ARM}_ct.h5" \
        --workers "$W" --out "$OUT"
    sha256sum "$OUT" ;;
  widthrules)
    [[ "$ARM" == w* ]] || { echo "widthrules is for the widths" >&2; exit 1; }
    OUT="$O/ct28_widthrules_${ARM}_ct.json"; refuse_existing "$OUT"
    python -m src.analysis.ct28_rescore widthrules --arm-name "$ARM" "${JOIN[@]}" \
        --rules-h5 "$A/cs/rule_interior_mass_${ARM}_n9.h5" \
        --rf-counts-070 "$A/cs/rf_counts_theta070_${ARM}_n9.h5" --workers "$W" --out "$OUT"
    sha256sum "$OUT" ;;
  *) echo "unknown STEP $STEP" >&2; exit 1 ;;
esac
echo "checkout at exit $(git rev-parse HEAD) dirty=[$(git status --porcelain | wc -l)]"
