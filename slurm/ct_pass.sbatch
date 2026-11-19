#!/bin/bash
#SBATCH --job-name=ct_pass
#SBATCH --output=logs/%x_%j.out
# Amendment CT's new arms through the pipeline of record (§CT24-F item 6,
# §CT25-F item 1, §CT26-F item 1). Resources on the sbatch line per step:
#   ap, null, fvu   gpu, --gpus=rtx_5000_ada:1 (one device class, gotcha 36)
#   gtilde          CPU, day
# Usage: sbatch <resources> slurm/ct_pass.sbatch <ARM> <STEP>
#   ARM  = ext | k64 | w2560 | w10240 | w20480 | w40960 | k64_w2560 | k64_w40960 | k16_w10240 | k128_w10240 | devctl_h200
#   STEP = ap | null | gtilde | fvu | rf (k64, grid) | dfs | match | fit | carrier (k64, grid) | carrier_control | ap_control | gtilde_control | act_encode | act_gram | act_reduce | act_fit (k64) | devctl (devctl_h200)
#          | null_probe | fvu_chunk (PROBE_SEEDS) | gtilde_topmerge | fit_topmerge (k16_w10240 -- §CT29-D2;
#          k128_w10240 -- §CT29-D3)
# feature_ap / null_pilot take k and dict_size from --config, not from the
# checkpoint, so every step first asserts each checkpoint's meta.json against
# the config (src/analysis/ct_arm_preflight.py): a k mismatch is otherwise silent.

set -euo pipefail
ARM="${1:?ARM: ext|k64|w2560|w10240|w20480|w40960|k64_w2560|k64_w40960|k16_w10240|k128_w10240|devctl_h200}"
STEP="${2:?STEP: ap|null|gtilde|fvu|rf|dfs|match|fit|carrier|carrier_control|ap_control|gtilde_control|act_encode|act_gram|act_reduce|act_fit|devctl|null_probe|fvu_chunk|gtilde_topmerge|fit_topmerge}"
SCRATCH=$SCRATCH
PROJECT=$PROJECT
A="$PROJECT/analysis"
LABELS="$A/residue_labels_subtyped_untruncated_pgate5_f67485d.h5"
REF_AP="$A/feature_ap_main_compliant.h5"
CACHE="$SCRATCH/cache/eval.h5"

cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
[[ -z "$(git status --porcelain)" ]] || { echo "checkout dirty before the run" >&2; exit 1; }
echo "checkout $(git rev-parse HEAD) arm=$ARM step=$STEP host=$(hostname)"

case "$ARM" in
  ext)  ROOT="$PROJECT/sae_ckpts_ext"; CONFIG=configs/mvp.yaml; EXPECT=20; SEEDS=$(seq 20 39) ;;
  k64)  ROOT="$PROJECT/sae_ckpts_k64"; CONFIG=configs/k64/mvp_k64.yaml; EXPECT=20; SEEDS=$(seq 0 19) ;;
  w2560|w10240|w20480|w40960)
        ROOT="$PROJECT/sae_ckpts_width/$ARM"; CONFIG="configs/width/mvp_${ARM}.yaml"; EXPECT=10; SEEDS=$(seq 0 9) ;;
  # §CT29: the grid arms (ROOT_KIND=h200 for an arm the author moves to H200, §CT29-F item 3)
  # and the device control (seed 40, configs/mvp.yaml, §CT29-F item 2).
  k64_w2560|k64_w40960|k16_w10240|k128_w10240)
        case "${ROOT_KIND:-rtx}" in
          rtx)  ROOT="$PROJECT/sae_ckpts_ct29/$ARM" ;;
          h200) ROOT="$PROJECT/sae_ckpts_ct29_h200/$ARM" ;;
          *) echo "ROOT_KIND must be rtx or h200" >&2; exit 1 ;;
        esac
        CONFIG="configs/grid/mvp_${ARM}.yaml"; EXPECT=10; SEEDS=$(seq 0 9) ;;
  devctl_h200)
        ROOT="$PROJECT/sae_ckpts_ct29_devctl/h200"; CONFIG=configs/mvp.yaml; EXPECT=1; SEEDS=40 ;;
  *) echo "unknown ARM $ARM" >&2; exit 1 ;;
esac
# PROBE_SEEDS (§CT29-F item 3): a memory probe on these seeds only, its outputs on
# scratch ($SCRATCH/ct29_probe/), never in $PROJECT; read for wall and MaxRSS only.
if [[ -n "${PROBE_SEEDS:-}" ]]; then
    SEEDS="$PROBE_SEEDS"; EXPECT=$(wc -w <<< "$PROBE_SEEDS")
    echo "PROBE on seeds $SEEDS"
fi
# AP_DEVICE (default auto) is passed to feature_ap --ap-device. `cuda` keeps the
# AP arithmetic on the device class null_pilot's unmasked path always uses, so
# the gtilde pre-flight's ap_arith_device_class check passes -- §CT10's route,
# not a loosened gate (gotcha 36).
# EXCLUDE (space-separated seeds) drops diverged k = 64 seeds (§CT24-F item 4).
CKPTS=()
for s in $SEEDS; do
    if [[ " ${EXCLUDE:-} " == *" $s "* ]]; then echo "excluding seed $s"; continue; fi
    CKPTS+=("$ROOT/seed_$s")
done
EXPECT=$(( EXPECT - $(wc -w <<< "${EXCLUDE:-}") ))
python -m src.analysis.ct_arm_preflight ckpts --config "$CONFIG" --expect "$EXPECT" --ckpts "${CKPTS[@]}"
K=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['sae']['k'])" "$CONFIG")

AP="$A/ct/feature_ap_${ARM}_ct.h5"
AP_ARM="$AP"   # the arm's own AP, which PROBE_SEEDS below does not redirect (§CT29-D2)
NULL="$A/ct/null_pilot_${ARM}_ct.h5"
RAW="$SCRATCH/ct_null_raw/null_gain_raw_${ARM}_ct.h5"
GT="$A/ct/gtilde_${ARM}_b_ratio_ct.h5"
if [[ -n "${PROBE_SEEDS:-}" ]]; then
    PR="$SCRATCH/ct29_probe"; mkdir -p "$PR"
    AP="$PR/feature_ap_${ARM}_probe.h5"; NULL="$PR/null_pilot_${ARM}_probe.h5"
    RAW="$PR/null_gain_raw_${ARM}_probe.h5"; GT="$PR/gtilde_${ARM}_probe.h5"
fi
mkdir -p "$A/ct" "$SCRATCH/ct_null_raw" "$SCRATCH/row_index_cache_ct"
# §CT29-D2 item 1 (route b1): the narrow, chunked row-index path, for arm (iv)
# and for the null_probe step only; every other null runs the path of record.
NEWPATH=(--narrow-row-index --ap-max-touches 4194304)
CTL="$SCRATCH/ct29_ctl"
TAG="s$(tr ' ' '-' <<< "${PROBE_SEEDS:-}" | tr -d '\n')"
# The top-bin sensitivities, each arm's edges declared before running:
# arm (iii) §CT29-D2 item 2; arm (iv) §CT29-D3 item 1.
case "$ARM" in
    k16_w10240)  TOPMERGE_EDGES="[0.0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 1.000000001]" ;;
    k128_w10240) TOPMERGE_EDGES="[0.0, 0.01, 0.03, 1.000000001]" ;;
    *)           TOPMERGE_EDGES="" ;;
esac
GT_TOP="$A/ct/gtilde_${ARM}_b_ratio_topmerged_ct.h5"
refuse_existing() { for f in "$@"; do [[ ! -e "$f" ]] || { echo "$f exists -- refusing" >&2; exit 1; }; done; }

case "$STEP" in
  ap)
    refuse_existing "$AP"
    python -m src.analysis.preflight --labels "$LABELS"
    python -m src.analysis.feature_ap --config "$CONFIG" --ckpts "${CKPTS[@]}" \
        --eval-cache "$CACHE" --labels "$LABELS" --strata pooled \
        --scored-row-set label_store --negatives all_rows --ap-device "${AP_DEVICE:-auto}" --out "$AP"
    python -m src.analysis.ct_arm_preflight axis --artifact "$AP" --kind ap --reference "$REF_AP"
    sha256sum "$AP"
    ;;
  null)
    refuse_existing "$NULL" "$RAW"
    python -m src.analysis.preflight --labels "$LABELS" --ap "$AP"
    NP=(); [[ "$ARM" == "k128_w10240" ]] && NP=("${NEWPATH[@]}")
    python -m src.analysis.null_pilot --config "$CONFIG" --ckpts "${CKPTS[@]}" \
        --eval-cache "$CACHE" --labels "$LABELS" --scored-row-set label_store \
        --null-type both --n-length-bins 20 --n-shifts 20 --null-seed 0 \
        --cache-dir "$SCRATCH/row_index_cache_ct" --exclude-pooled-parents \
        ${NP[@]+"${NP[@]}"} --out "$NULL" --raw-out "$RAW"
    python -m src.analysis.ct_arm_preflight axis --artifact "$NULL" --kind null --reference "$REF_AP"
    sha256sum "$NULL" "$RAW"
    ;;
  null_probe)
    # §CT29-D2 item 1: the control (ARM=k64, PROBE_SEEDS=0) and the cost pilot
    # (ARM=k128_w10240, PROBE_SEEDS=0) on the new path; outputs on scratch, under
    # $SCRATCH/ct29_ctl/. Pre-flight reads the arm's own AP. PROBE_CACHE_DIR (default
    # the shared row_index_cache_ct) is the pilot's private cache, so arm (iv)'s full run
    # builds every seed's index itself.
    [[ -n "${PROBE_SEEDS:-}" ]] || { echo "null_probe needs PROBE_SEEDS" >&2; exit 1; }
    mkdir -p "$CTL"
    PN="$CTL/null_pilot_${ARM}_${TAG}_new.h5"; PRAW="$CTL/null_gain_raw_${ARM}_${TAG}_new.h5"
    refuse_existing "$PN" "$PRAW"
    python -m src.analysis.preflight --labels "$LABELS" --ap "$AP_ARM"
    python -m src.analysis.null_pilot --config "$CONFIG" --ckpts "${CKPTS[@]}" \
        --eval-cache "$CACHE" --labels "$LABELS" --scored-row-set label_store \
        --null-type both --n-length-bins 20 --n-shifts 20 --null-seed 0 \
        --cache-dir "${PROBE_CACHE_DIR:-$SCRATCH/row_index_cache_ct}" --exclude-pooled-parents \
        "${NEWPATH[@]}" --out "$PN" --raw-out "$PRAW"
    python -m src.analysis.ct_arm_preflight axis --artifact "$PN" --kind null --reference "$REF_AP"
    sha256sum "$PN" "$PRAW"
    ;;
  gtilde)
    refuse_existing "$GT"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    # ACCEPT_SPLIT=1: the author's scoped exemption (2026-10-24; §CT24-F-C3, §CT25-F-C1):
    # the AP-arithmetic device may differ from the null's only when the gathers agree.
    SPLIT=(); [[ "${ACCEPT_SPLIT:-0}" == "1" ]] && SPLIT=(--accept-ap-arith-split)
    python -m src.analysis.standardize_gain --ap "$AP" --null-raw "$RAW" \
        --null-reductions "$NULL" --labels "$LABELS" --inner-mode ratio \
        --config "$CONFIG" ${SPLIT[@]+"${SPLIT[@]}"} --out "$GT"
    sha256sum "$GT"
    ;;
  fvu)
    OUT="$A/ct/fvu_eval_${ARM}_ct.json"
    refuse_existing "$OUT"
    # §CS4a: at 40,960 the default 131,072-row chunk's pre-activations (20 GiB)
    # ran out of GPU memory; §CS4a's requests are 32,768 rows at 40,960 and
    # 65,536 at 20,480. The chunk size is not recorded in the output, so it is logged.
    case "$ARM" in *w40960) CHUNK=32768 ;; w20480) CHUNK=65536 ;; *) CHUNK=131072 ;; esac
    echo "fvu chunk-rows $CHUNK"
    python -m src.analysis.fvu --cache "$CACHE" --ckpts "$ROOT" \
        --seeds $(for c in "${CKPTS[@]}"; do basename "$c" | sed 's/seed_//'; done) \
        --k "$K" --chunk-rows "$CHUNK" --out "$OUT"
    sha256sum "$OUT"
    ;;
  fvu_chunk)
    # §CT29-D2 item 5: one seed's FVU at 131,072 and at 32,768 rows, compared
    # bitwise afterwards; outputs on scratch, under $SCRATCH/ct29_ctl/.
    [[ "$EXPECT" == "1" ]] || { echo "fvu_chunk takes one seed (PROBE_SEEDS)" >&2; exit 1; }
    mkdir -p "$CTL"
    S0=$(basename "${CKPTS[0]}" | sed 's/seed_//')
    for C in 131072 32768; do
        OUTC="$CTL/fvu_${ARM}_${TAG}_c${C}.json"
        refuse_existing "$OUTC"
        echo "fvu chunk-rows $C"
        python -m src.analysis.fvu --cache "$CACHE" --ckpts "$ROOT" --seeds "$S0" \
            --k "$K" --chunk-rows "$C" --out "$OUTC"
        sha256sum "$OUTC"
    done
    ;;
  rf)
    # §CT24-F item 5 / §CT29-F item 5: decoder-cosine MNN at the 13-theta grid among
    # the arm's seeds (n_compare = n - 1). The k = 32 widths' are §CS's, in analysis/cs/.
    case "$ARM" in k64|k64_w2560|k64_w40960|k16_w10240|k128_w10240) ;;
      *) echo "rf is run for k64 and the §CT29 grid arms; the widths' are §CS's" >&2; exit 1 ;; esac
    refuse_existing "$A/ct/rf_marginal_m1_${ARM}_ct.json"
    # §CS3b: the subsampled Hungarian draws without replacement, so no size may
    # exceed the width; at 2,560 the sizes are 1024 2048, as §CS3b reran them.
    HSIZES=(1024 2048 4096); [[ "$ARM" == "k64_w2560" ]] && HSIZES=(1024 2048)
    python -m src.analysis.rf_marginal --ckpts "${CKPTS[@]}" --expect-total-ckpts "$EXPECT" \
        --rotation-draws 50 --hungarian-sizes "${HSIZES[@]}" --hungarian-repeats 5 --rng-seed 0 \
        --out "$A/ct/rf_marginal_m1_${ARM}_ct.json" \
        --emit-counts "$A/ct/rf_counts_theta070_${ARM}_ct.h5" \
        --emit-counts-all-theta "$A/ct/rf_counts_all_theta_${ARM}_ct.h5"
    sha256sum "$A/ct/rf_marginal_m1_${ARM}_ct.json" "$A/ct/rf_counts_theta070_${ARM}_ct.h5"         "$A/ct/rf_counts_all_theta_${ARM}_ct.h5"
    ;;
  dfs)
    # the design-3 covariate d_fs (repr_density, §AQ2's definition A) on the
    # arm's row indices, which the null step persisted in row_index_cache_ct
    OUT="$A/ct/d_fs_${ARM}_ct.h5"
    refuse_existing "$OUT"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python -m src.analysis.repr_density --config "$CONFIG" --labels "$LABELS" \
        --eval-cache "$CACHE" --clusters "$PROJECT/archive/sprot_clu30_cluster.tsv" \
        --row-index-cache "$SCRATCH/row_index_cache_ct" --ckpts "$ROOT" \
        --seeds $(for c in "${CKPTS[@]}"; do basename "$c" | sed 's/seed_//'; done) \
        --k "$K" --no-ao6-comparison --out "$OUT" --out-json "${OUT%.h5}.json"
    sha256sum "$OUT"
    ;;
  gtilde_topmerge)
    # §CT29-D2 item 2 (arm (iii)) and §CT29-D3 item 1 (arm (iv)): the declared
    # sensitivity -- the top firing bin merged into the one below it; refuses
    # unless the recorded edges are the declared ones. Not a scored value.
    [[ -n "$TOPMERGE_EDGES" ]] || { echo "gtilde_topmerge is declared for arms (iii) and (iv) only" >&2; exit 1; }
    refuse_existing "$GT_TOP"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    SPLIT=(); [[ "${ACCEPT_SPLIT:-0}" == "1" ]] && SPLIT=(--accept-ap-arith-split)
    python -m src.analysis.standardize_gain --ap "$AP" --null-raw "$RAW" \
        --null-reductions "$NULL" --labels "$LABELS" --inner-mode ratio \
        --config "$CONFIG" ${SPLIT[@]+"${SPLIT[@]}"} --merge-top-firing-bin --out "$GT_TOP"
    python -c "import h5py,json,sys
e=json.loads(h5py.File(sys.argv[1],'r').attrs['firing_rate_edges'])
want=json.loads(sys.argv[2])
print('recorded edges', e, 'declared', want)
sys.exit(0 if e==want else 1)" "$GT_TOP" "$TOPMERGE_EDGES"
    sha256sum "$GT_TOP"
    ;;
  fit_topmerge)
    # §CT29-D2 item 2 / §CT29-D3 item 1: the fit on the sensitivity's gtilde,
    # with the arm's own match, rf and d_fs outputs. Not a scored value.
    [[ -n "$TOPMERGE_EDGES" ]] || { echo "fit_topmerge is declared for arms (iii) and (iv) only" >&2; exit 1; }
    OUT="$A/ct/slope_${ARM}_topmerged_ct.json"
    refuse_existing "$OUT"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python -m src.analysis.jackknife_dictionaries fit --sim "$A/ct/jackknife_match_${ARM}_ct.h5" \
        --rf-counts-all-theta "$A/ct/rf_counts_all_theta_${ARM}_ct.h5" \
        --m1-json "$A/ct/rf_marginal_m1_${ARM}_ct.json" --gtilde "$GT_TOP" \
        --null-reductions "$NULL" --d-fs "$A/ct/d_fs_${ARM}_ct.h5" \
        --config "$CONFIG" --thetas 0.70 --workers "${SLURM_CPUS_PER_TASK:-8}" --out "$OUT"
    sha256sum "$OUT"
    ;;
  match|fit)
    if [[ "$ARM" == k64* || "$ARM" == k16_w10240 || "$ARM" == k128_w10240 ]]; then
        # k = 64 and the §CT29 grid arms: their own rf step's outputs, in analysis/ct/
        ALLTH="$A/ct/rf_counts_all_theta_${ARM}_ct.h5"; M1="$A/ct/rf_marginal_m1_${ARM}_ct.json"
    else
        N=$(( EXPECT - 1 ))
        ALLTH="$A/cs/rf_counts_all_theta_${ARM}_n${N}.h5"; M1="$A/cs/rf_marginal_m1_${ARM}_n${N}.json"
    fi
    SIM="$A/ct/jackknife_match_${ARM}_ct.h5"
    if [[ "$STEP" == "match" ]]; then
        refuse_existing "$SIM"
        ACCEPT=()
        case "$ARM" in
          k64_w2560|k64_w40960|k16_w10240|k128_w10240)
            DEVSRC="the arm's rf_marginal ran on gpu rtx_5000_ada (ct_pass.sbatch rf; §CT29-F item 5)" ;;
          *) DEVSRC="the arm's rf_marginal ran on gpu rtx_5000_ada (§CS7 tier-1 jobs; §CT24-F item 5)" ;;
        esac
        if [[ "$ARM" == "w40960" ]]; then
            # §CS3c's two-SHA pool, read from the checkpoints and asserted to be
            # exactly the recorded pair (eb8657b9... probe seed 0; 42376f67...).
            mapfile -t SHAS < <(python -c "import json,sys
s=sorted({json.load(open(p+'/meta.json'))['git_sha'] for p in sys.argv[1:]})
assert [x[:8] for x in s]==['42376f67','eb8657b9'], s
print('\n'.join(s))" "${CKPTS[@]}")
            ACCEPT=(--accept-git-shas "${SHAS[@]}")
        fi
        python -m src.analysis.jackknife_dictionaries match --ckpts "${CKPTS[@]}" \
            --expect-total-ckpts "$EXPECT" --rf-counts-all-theta "$ALLTH" \
            --record-device-class rtx_5000_ada \
            --record-device-source "$DEVSRC" \
            "${ACCEPT[@]}" --out "$SIM"
        sha256sum "$SIM"
    else
        OUT="$A/ct/slope_${ARM}_ct.json"
        refuse_existing "$OUT"
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
        python -m src.analysis.jackknife_dictionaries fit --sim "$SIM" \
            --rf-counts-all-theta "$ALLTH" --m1-json "$M1" --gtilde "$GT" \
            --null-reductions "$NULL" --d-fs "$A/ct/d_fs_${ARM}_ct.h5" \
            --config "$CONFIG" --thetas 0.70 --workers "${SLURM_CPUS_PER_TASK:-8}" --out "$OUT"
        sha256sum "$OUT"
    fi
    ;;
  carrier)
    # §CT24-F item 8 / §CT29-F item 5: §BB9.2's single firing-matched carrier arm
    # and its jackknife over the arm's dictionaries (carrier_jackknife --single-only),
    # on gpu rtx_5000_ada as §CT19 ran. Its control is `ct_pass.sbatch ext
    # carrier_control` (the anchors through the same code; the ARM only satisfies
    # the preflight, and `ext` exists).
    case "$ARM" in k64|k64_w2560|k64_w40960|k16_w10240|k128_w10240) ;;
      *) echo "carrier is declared for k64 and the §CT29 grid arms" >&2; exit 1 ;; esac
    OUT="$A/ct/carrier_jackknife_${ARM}_ct.json"
    refuse_existing "$OUT"
    python -m src.analysis.carrier_jackknife --single-only --ap "$AP" --null "$NULL" \
        --counts-reference "$A/ct/rf_counts_theta070_${ARM}_ct.h5" --ckpts "${CKPTS[@]}" \
        --expect-n "$EXPECT" --out "$OUT"
    sha256sum "$OUT"
    ;;
  act_encode|act_gram|act_reduce|act_fit)
    # §CT24-F item 10: the activation-correlation arm on the k = 64 dictionaries,
    # 20 x 19, theta* rate-matched to k = 64's decoder-cosine rate at 0.70.
    [[ "$ARM" == "k64" ]] || { echo "the activation arm is declared for k64 only" >&2; exit 1; }
    AC="$SCRATCH/ct24_actcorr"; mkdir -p "$AC/codes" "$AC/rows"
    case "$STEP" in
      act_encode)
        for c in "${CKPTS[@]}"; do
          s=$(basename "$c" | sed 's/seed_//')
          [[ -e "$AC/codes/codes_seed_$s.h5" ]] && { echo "seed $s exists, skipped"; continue; }
          python -m src.analysis.activation_correlation encode --ckpt "$c" \
              --cache "$CACHE" --labels "$LABELS" --out "$AC/codes/codes_seed_$s.h5"
        done ;;
      act_gram)
        S=$SLURM_ARRAY_TASK_ID
        ALL=(); for c in "${CKPTS[@]}"; do s=$(basename "$c" | sed 's/seed_//'); ALL+=("$AC/codes/codes_seed_$s.h5"); done
        for f in "${ALL[@]}"; do [[ -e "$f" ]] || { echo "missing $f" >&2; exit 1; }; done
        [[ -e "$AC/rows/row_anchor_$S.h5" ]] && { echo "anchor $S exists, skipped"; exit 0; }
        python -m src.analysis.activation_correlation gram --codes "${ALL[@]}" \
            --anchor "$S" --out "$AC/rows/row_anchor_$S.h5" ;;
      act_reduce)
        R=(); for c in "${CKPTS[@]}"; do s=$(basename "$c" | sed 's/seed_//'); R+=("$AC/rows/row_anchor_$s.h5"); done
        TARGET=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['marginals_by_theta']['0.70']['mnn']['mean'])" \
            "$A/ct/rf_marginal_m1_${ARM}_ct.json")
        echo "theta* target = k64 decoder-cosine mean at 0.70 = $TARGET"
        refuse_existing "$A/ct/activation_correlation_${ARM}_ct.json"
        python -m src.analysis.activation_correlation reduce --rows "${R[@]}" --target-rate "$TARGET" \
            --out "$A/ct/activation_correlation_${ARM}_ct.json" \
            --out-counts "$A/ct/activation_correlation_counts_${ARM}_ct.h5" ;;
      act_fit)
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
        refuse_existing "$A/ct/activation_correlation_fit_${ARM}_ct.json"
        python -m src.analysis.activation_correlation fit \
            --act-counts "$A/ct/activation_correlation_counts_${ARM}_ct.h5" \
            --sim "$A/ct/jackknife_match_${ARM}_ct.h5" \
            --rf-counts-all-theta "$A/ct/rf_counts_all_theta_${ARM}_ct.h5" \
            --m1-json "$A/ct/rf_marginal_m1_${ARM}_ct.json" --gtilde "$GT" \
            --null-reductions "$NULL" --d-fs "$A/ct/d_fs_${ARM}_ct.h5" --config "$CONFIG" \
            --out "$A/ct/activation_correlation_fit_${ARM}_ct.json" ;;
    esac
    ;;
  devctl)
    # §CT29-F item 2: the device control's FVU must exist first (STEP fvu on this ARM).
    [[ "$ARM" == "devctl_h200" ]] || { echo "devctl is for ARM devctl_h200" >&2; exit 1; }
    OUT="$A/ct/ct29_device_control_ct.json"
    refuse_existing "$OUT"
    python -m src.analysis.ct29_device_control --ckpt "$ROOT/seed_40"         --anchor-root "$PROJECT/sae_ckpts" --stored-counts "$A/rf_counts_theta070_n19.h5"         --fvu-json "$A/ct/fvu_eval_${ARM}_ct.json" --out "$OUT"
    sha256sum "$OUT"
    ;;
  gtilde_control)
    # The author's condition (a), 2026-10-24: the anchors' record pair through the
    # exempted standardize_gain path must reproduce gtilde_compliant_b_ratio.h5's
    # M -- BITWISE, or WITHIN_BAR (max rel <= 1e-6) recorded -- else STOP.
    OUT="$A/ct/gtilde_record_rerun_ct.h5"
    refuse_existing "$OUT"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python -m src.analysis.standardize_gain --ap "$A/feature_ap_main_compliant.h5" \
        --null-raw "$A/null_gain_raw_compliant.h5" --null-reductions "$A/null_pilot_compliant.h5" \
        --labels "$LABELS" --inner-mode ratio --config configs/mvp.yaml --accept-ap-arith-split \
        --out "$OUT"
    python -m src.analysis.ct_arm_preflight gtcompare --new "$OUT" \
        --reference "$A/gtilde_compliant_b_ratio.h5" --out "$A/ct/gtilde_record_rerun_compare_ct.json"
    sha256sum "$OUT" "$A/ct/gtilde_record_rerun_compare_ct.json"
    ;;
  ap_control)
    # §CT26-F item 1's control: the same pass on one anchor (seed 0) reproduces
    # feature_ap_main_compliant.h5's pooled ap for seed 0 to <= 1e-6 (CODEBASE §13.4).
    OUT="$A/ct/feature_ap_seed0_control_ct.h5"
    refuse_existing "$OUT"
    python -m src.analysis.feature_ap --config configs/mvp.yaml --ckpts "$PROJECT/sae_ckpts/seed_0" \
        --eval-cache "$CACHE" --labels "$LABELS" --strata pooled \
        --scored-row-set label_store --negatives all_rows --ap-device auto --out "$OUT"
    python -m src.analysis.ct_arm_preflight apcompare --new "$OUT" --reference "$REF_AP" \
        --seed-label seed_0 --out "$A/ct/ap_seed0_control_ct.json"
    sha256sum "$OUT" "$A/ct/ap_seed0_control_ct.json"
    ;;
  carrier_control)
    # the k = 32 anchors through --single-only must reproduce §CT19's stored
    # single arm (carrier_jackknife_ct.json) EXACTLY before the k = 64 value is read
    OUT="$A/ct/carrier_jackknife_single_control_ct.json"
    refuse_existing "$OUT"
    ANCHORS=("$PROJECT"/sae_ckpts/seed_*)
    python -m src.analysis.carrier_jackknife --single-only \
        --ap "$A/feature_ap_main_compliant.h5" --null "$A/null_pilot_compliant.h5" \
        --counts-reference "$A/rf_counts_theta070_n19.h5" --ckpts "${ANCHORS[@]}" \
        --control-jackknife "$A/ct/carrier_jackknife_ct.json" --out "$OUT"
    sha256sum "$OUT"
    ;;
  *) echo "unknown STEP $STEP" >&2; exit 1 ;;
esac
[[ -z "$(git status --porcelain)" ]] || { echo "checkout dirty on exit" >&2; exit 1; }
