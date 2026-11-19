#!/bin/bash
#SBATCH --job-name=ct_robust
#SBATCH --output=logs/%x_%j.out
# Amendment CT's CPU batch (§CT15, §CT16, §CT17, §CT20, §CT21), through
# src/analysis/ct_robustness.py. Resources are passed on the sbatch command
# line per step:
#
#   ct15, ct20, ct17, ct16_fit, ct21   CPU, `day`
#   ct16_sf                            gpu rtx_5000_ada (§CT3's device class)
#
# Every slope goes through §CT3's loader and fitter with the design-3 join's
# inputs of record; outputs to $PROJECT/analysis/ct/.
set -euo pipefail
STEP="${1:?ct15|ct20|ct17|ct16_sf|ct16_fit|ct21}"
cd "$SLURM_SUBMIT_DIR"
source env/activate.sh
[[ -z "$(git status --porcelain)" ]] || { echo "checkout dirty" >&2; exit 1; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PROJECT=$PROJECT
A="$PROJECT/analysis"
mkdir -p "$A/ct"
JOIN=(--sim "$A/ct/jackknife_match_ct.h5"
      --rf-counts-all-theta "$A/rf_counts_all_theta_n39.h5"
      --m1-json "$A/rf_marginal_m1_n39.json"
      --gtilde "$A/gtilde_compliant_b_ratio.h5"
      --null-reductions "$A/null_pilot_compliant.h5"
      --d-fs "$A/d_fs.h5" --degenerate-flag "$A/degenerate_argmax_flag.h5"
      --reference-join "$A/design3_join_aw_nofeat.json")
ACT=(--act-counts "$A/ct/activation_correlation_counts_ct.h5"
     --act-json "$A/ct/activation_correlation_ct.json")

case "$STEP" in
  ct15)
    python -m src.analysis.ct_robustness ct15 "${JOIN[@]}" \
        --jackknife-json "$A/ct/jackknife_ct.json" --out "$A/ct/ct15_disjoint_comparators_ct.json"
    ;;
  ct20)
    python -m src.analysis.ct_robustness ct20 "${JOIN[@]}" "${ACT[@]}" \
        --rules-h5 "$A/rule_interior_mass_co.h5" --out "$A/ct/ct20_rules_n19_ct.json"
    ;;
  ct20_rerun)
    # §CT20, the preamble's item 1: rules 1-3 re-run on the 20 anchors with
    # every per-comparator similarity persisted (CPU, the CO run's device).
    python -m src.analysis.rule_interior_mass --ckpt-root "$PROJECT/sae_ckpts" \
        --rf-counts-n19 "$A/rf_counts_theta070_n19.h5" \
        --out-json "$A/ct/ct20_rules_rerun_ct.json" --out-h5 "$A/ct/ct20_rules_rerun_counts_ct.h5" \
        --out-sim-h5 "$A/ct/ct20_rules_sim_ct.h5"
    ;;
  ct27_sim)
    # §CT27(b): the raw-encoder rule 3, per-comparator similarity persisted;
    # its control rebuilds normalised rule 3 and must equal ct20_rules_sim_ct.h5.
    python -m src.analysis.ct27_raw_encoder --ckpt-root "$PROJECT/sae_ckpts" \
        --stored-rules-sim-h5 "$A/ct/ct20_rules_sim_ct.h5" \
        --out-h5 "$A/ct/ct27_raw_encoder_sim_ct.h5"
    ;;
  ct27_fit)
    python -m src.analysis.ct_robustness ct27 "${JOIN[@]}" \
        --variant-sim-h5 "$A/ct/ct27_raw_encoder_sim_ct.h5" \
        --rules-sim-h5 "$A/ct/ct20_rules_sim_ct.h5" \
        --grid-json "$A/ct/ct20b_rules_grid_ct.json" \
        --out "$A/ct/ct27_raw_encoder_ct.json"
    ;;
  ct23_disorder)
    # §CT23.3: the Disordered argmax share among aligned cells, three rules;
    # gated on the five recorded figures. concept_subspace's inputs and gates.
    ANCHORS=("$PROJECT"/sae_ckpts/seed_*)
    python -m src.analysis.ct23_disorder_shares \
        --sim "$A/ct/jackknife_match_ct.h5" \
        --rf-counts-all-theta "$A/rf_counts_all_theta_n39.h5" \
        --m1-json "$A/rf_marginal_m1_n39.json" \
        --gtilde "$A/gtilde_compliant_b_ratio.h5" \
        --null-reductions "$A/null_pilot_compliant.h5" \
        --d-fs "$A/d_fs.h5" --degenerate-flag "$A/degenerate_argmax_flag.h5" \
        --ap "$A/feature_ap_main_compliant.h5" \
        --bundle-concepts "$PROJECT/release/derived/concepts_carriers.csv.gz" \
        --carrier-reference "$A/carrier_concentration_bb.json" \
        --tuning-set docs/ct22_tuning_set.json \
        --ckpts "${ANCHORS[@]}" --expect-ckpts 20 \
        --out "$A/ct/ct23_disorder_aligned_ct.json"
    ;;
  ct20b)
    python -m src.analysis.ct_robustness ct20b "${JOIN[@]}" "${ACT[@]}" \
        --rules-h5 "$A/rule_interior_mass_co.h5" --rules-sim-h5 "$A/ct/ct20_rules_sim_ct.h5" \
        --out "$A/ct/ct20b_rules_grid_ct.json"
    ;;
  ct17)
    python -m src.analysis.ct_robustness ct17 "${JOIN[@]}" "${ACT[@]}" \
        --out "$A/ct/ct17_slopes_by_firing_bin_ct.json"
    ;;
  ct16_sf)
    CK=(); for s in $(seq 0 19); do CK+=("$PROJECT/sae_ckpts/seed_$s"); done
    CE=(); for s in $(seq 20 39); do CE+=("$PROJECT/sae_ckpts_ext/seed_$s"); done
    python -m src.analysis.ct_robustness ct16-sf --ckpts "${CK[@]}" --compare-ckpts "${CE[@]}" \
        --sim "$A/ct/jackknife_match_ct.h5" --out "$A/ct/ct16_oneway_max_cos_ct.h5"
    ;;
  ct16_fit)
    python -m src.analysis.ct_robustness ct16-fit "${JOIN[@]}" \
        --oneway-h5 "$A/ct/ct16_oneway_max_cos_ct.h5" --out "$A/ct/ct16_threshold_independent_ct.json"
    ;;
  ct21)
    python -m src.analysis.ct_robustness ct21 --jackknife-json "$A/ct/jackknife_ct.json" \
        --curve-json "$A/ct/activation_vs_decoder_slope_curve_ct.json" \
        --out "$A/ct/ct21_figure_data_ct.json"
    ;;
  ct19)
    # §CT19 (the brief's CT11): gpu rtx_5000_ada, the carrier scripts' device.
    SCRATCH=$SCRATCH
    ANCHORS=("$PROJECT"/sae_ckpts/seed_*)
    [[ ${#ANCHORS[@]} -eq 20 ]] || { echo "expected 20 anchors, found ${#ANCHORS[@]}" >&2; exit 1; }
    python -m src.analysis.carrier_jackknife \
        --ap "$A/feature_ap_main_compliant.h5" --null "$A/null_pilot_compliant.h5" \
        --d-fs "$A/d_fs.h5" --ckpts "${ANCHORS[@]}" --ckpts-root "$PROJECT/sae_ckpts" \
        --row-index-cache "$SCRATCH/row_index_cache" --eval-cache "$SCRATCH/cache/eval.h5" \
        --labels "$A/residue_labels_subtyped_untruncated_pgate5_f67485d.h5" \
        --counts-reference "$A/rf_counts_theta070_n19.h5" \
        --reference "$A/carrier_concentration_bb.json" \
        --build-row-index \
        --out "$A/ct/carrier_jackknife_ct.json"
    ;;
  ct14_gtilde)
    # §CT14: §CT5.1's gtilde with the lowest firing bin merged; STRATUM = $2.
    S2="${2:?experimental|propagated}"
    SCRATCH=$SCRATCH
    python -m src.analysis.standardize_gain \
        --ap "$A/ct/feature_ap_shared_${S2}_ct.h5" \
        --null-raw "$SCRATCH/ct5/null_gain_raw_shared_${S2}_ct.h5" \
        --null-reductions "$A/ct/null_shared_${S2}_ct.h5" \
        --labels "$A/residue_labels_subtyped_untruncated_pgate5_f67485d.h5" \
        --arm b --inner-mode ratio --stratum "$S2" --axis-values stratum \
        --admission-strata experimental propagated \
        --merge-lowest-firing-bin \
        --eval-cache "$SCRATCH/cache/eval.h5" \
        --clusters-tsv "$PROJECT/archive/sprot_clu30_cluster.tsv" \
        --config configs/mvp.yaml \
        --out "$A/ct/gtilde_shared_${S2}_merged_ct.h5"
    ;;
  ct14_slope)
    python -m src.analysis.stratum_slope \
        --sim "$A/ct/jackknife_match_ct.h5" \
        --rf-counts-all-theta "$A/rf_counts_all_theta_n39.h5" \
        --m1-json "$A/rf_marginal_m1_n39.json" \
        --gtilde "$A/gtilde_compliant_b_ratio.h5" \
        --null-reductions "$A/null_pilot_compliant.h5" \
        --d-fs "$A/d_fs.h5" --degenerate-flag "$A/degenerate_argmax_flag.h5" \
        --config configs/mvp.yaml --theta 0.70 \
        --strata experimental propagated \
        --gtilde-strata "$A/ct/gtilde_shared_experimental_merged_ct.h5" "$A/ct/gtilde_shared_propagated_merged_ct.h5" \
        --out "$A/ct/stratum_slope_merged_ct.json"
    ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
[[ -z "$(git status --porcelain)" ]] || { echo "checkout dirty on exit" >&2; exit 1; }
