"""Mutation matrix for amendment AZ's new code. NOT collected by pytest.

Covers `carrier_support` (AZ11), `alignment_fine_panel` (AZ12),
`concept_headline_checks` (AZ13/AZ14) and `never_detected` (AZ15).

Run manually:  python tests/mutation_matrix_az.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

The runner is reused from `mutation_matrix_av` -- AU9b's control mutation,
AV22's `__pycache__` purge between invocations, the byte-compared restore, and
the decide-on-exit-code rule. Reused rather than re-implemented: a runner is a
check, and a second copy is a second thing that can silently stop detecting
(gotcha 13 extension 5).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run, run_one  # noqa: E402

ROOT = os.getcwd()
S = os.path.join(ROOT, "src", "analysis", "carrier_support.py")
F = os.path.join(ROOT, "src", "analysis", "alignment_fine_panel.py")
H = os.path.join(ROOT, "src", "analysis", "concept_headline_checks.py")
N = os.path.join(ROOT, "src", "analysis", "never_detected.py")
TS = "tests/test_carrier_support.py"
TF = "tests/test_alignment_fine_panel.py"
TH = "tests/test_concept_headline_checks.py"
TN = "tests/test_never_detected.py"

# THE CONTROL: the positional verdict replaced by a constant. This cannot fail
# to break the suite -- the wrong-order fixture has the SAME TOTAL as the right
# one and asserts POSITIONS_WRONG. If the runner reports this undetected, none
# of its SURVIVED verdicts mean anything and none is printed.
CONTROL = (
    "CONTROL -- the positional verdict replaced by a constant", N,
    b'        "verdict": "POSITIONS_OK" if ok == want.size else "POSITIONS_WRONG",',
    b'        "verdict": "POSITIONS_OK",',
    [f"{TN}::test_verify_block_starts_REFUSES_a_permutation_with_the_SAME_TOTAL"],
)

MUTATIONS = [
    # ------------------------------------------------------ carrier_support --
    # The whole design of the second matching axis. Global quantiles nested
    # inside a firing stratum are nearly constant, so the double arm would
    # silently be the single one while appearing to control for support.
    ("support_bin_within: GLOBAL quantiles instead of within-stratum", S,
     b"            v = d_fs[s][m]\n"
     b"            edges = np.percentile(v, qs) if v.size else np.array([])",
     b"            v = d_fs[s][m]\n"
     b"            edges = np.percentile(d_fs[s], qs)",
     [f"{TS}::test_support_bins_are_taken_WITHIN_each_stratum_not_globally"]),

    # The support axis dropped from the cell index: the two arms would be the
    # same arm and their difference would be exactly zero.
    ("cell_index: the support axis is ignored", S,
     b"    return np.asarray(fbin, dtype=np.int64) * n_support + np.asarray(\n"
     b"        sbin, dtype=np.int64)",
     b"    return np.asarray(fbin, dtype=np.int64)",
     [f"{TS}::test_cell_index_separates_every_firing_x_support_pair"]),

    # THE LOAD-BEARING DISTINCTION, inherited from AY: B collapses into A.
    ("carrier_rates_by_cell: B collapses into A", S,
     b"            b = p == ct",
     b"            b = p >= 0",
     [f"{TS}::test_carrier_rates_by_cell_matches_the_hand_computation",
      f"{TS}::test_carrier_rates_by_cell_separates_B_from_A",
      f"{TS}::test_single_arm_reproduces_carrier_identity_EXACTLY"]),

    # The carrier included in its own cell baseline -- the comparison would be
    # partly a feature measured against itself.
    ("carrier_rates_by_cell: the carrier is not excluded from its own cell", S,
     b"                num = hit_in[ok] - a[ok].astype(np.int64)\n"
     b"                den = n_in[ok] - 1",
     b"                num = hit_in[ok]\n"
     b"                den = n_in[ok]",
     [f"{TS}::test_carrier_rates_by_cell_matches_the_hand_computation",
      f"{TS}::test_single_arm_reproduces_carrier_identity_EXACTLY"]),

    # The usability guard. With `> 0` a singleton cell divides by zero after the
    # carrier is excluded and reports a rate rather than counting the carrier
    # as unusable -- and the finer the cells, the more of them are singletons.
    ("carrier_rates_by_cell: singleton cells are not counted unusable", S,
     b"            ok = n_in > 1",
     b"            ok = n_in > 0",
     [f"{TS}::test_a_singleton_cell_is_unusable_rather_than_a_division_by_zero"]),

    # The predicate's own boundary. "Less than half" is a strict comparison and
    # the boundary case is reachable.
    ("excess_comparison: the half boundary becomes inclusive", S,
     b'        "outcome": ("UNDER_HALF" if e2 < e1 / 2.0 else "HALF_OR_ABOVE"),',
     b'        "outcome": ("UNDER_HALF" if e2 <= e1 / 2.0 else "HALF_OR_ABOVE"),',
     [f"{TS}::test_exactly_half_counts_as_half_or_above"]),

    # A ratio against a zero denominator invented rather than refused.
    ("excess_comparison: a zero single excess yields a verdict anyway", S,
     b"    if not np.isfinite(e1) or e1 == 0.0:",
     b"    if False:",
     [f"{TS}::test_excess_comparison_is_total_over_the_ratio_domain"]),

    # The half-tie convention. Without it a value at the population median is
    # reported below it, and every carrier percentile shifts.
    ("percentile_positions: ties are ranked low instead of at their midpoint", S,
     b"        pos.append(100.0 * (lo + hi) / 2.0 / order.size)",
     b"        pos.append(100.0 * lo / order.size)",
     [f"{TS}::test_percentile_positions_uses_the_half_tie_convention"]),

    # ABSENT must never read as AGREEMENT (gotcha 15 / I3).
    ("reproduces_ay: an absent reference key is treated as matching", S,
     b'            problems.append(f"{theirs}: ABSENT (absent is not matching)")',
     b"            pass",
     [f"{TS}::test_reproduces_ay_reports_an_ABSENT_key_as_absent"]),

    # Aligning two different seed sets is a silent change of what is measured.
    ("align_to_ap_order: a different seed set is aligned anyway", S,
     b"    if sorted(have) != sorted(want):",
     b"    if False:",
     [f"{TS}::test_align_to_ap_order_refuses_a_different_seed_set"]),

    # ------------------------------------------------- alignment_fine_panel --
    ("top_fraction_mask: the quantile is taken over ALL features", F,
     b"    v = logM_row[keep]",
     b"    v = logM_row",
     [f"{TF}::test_top_fraction_mask_takes_the_quantile_over_KEPT_FEATURES_ONLY"]),

    # THE INTERVAL BASIS. AZ6.1's whole point: the empirical dispersion, not a
    # binomial construction, and no multiplier imported from another reduction.
    ("bin_table: the interval reverts to the naive binomial SE", F,
     b"        se = (sd / np.sqrt(n)) if sd is not None else None",
     b"        se = float(np.sqrt(p * (1 - p) / n))",
     [f"{TF}::test_interval_is_the_EMPIRICAL_dispersion_not_the_binomial",
      f"{TF}::test_a_single_feature_bin_reports_None_rather_than_a_zero_interval"]),

    # A one-feature bin reporting 0 dispersion reads as "measured, and exact".
    ("bin_table: a single-feature bin reports a zero SD", F,
     b'        sd = float(v.std(ddof=1)) if n > 1 else None',
     b'        sd = float(v.std(ddof=1)) if n > 1 else 0.0',
     [f"{TF}::test_a_single_feature_bin_reports_None_rather_than_a_zero_interval"]),

    # THE CLAUSE THE PREDICATE WAS DECLARED WITH. Without it a dip well inside
    # the noise is scored as a violation.
    ("monotone_verdict: the interval-overlap clause is dropped", F,
     b"        ok = bool(up or overlap)",
     b"        ok = bool(up)",
     [f"{TF}::test_monotone_verdict_RESCUES_a_dip_whose_intervals_overlap"]),

    # A missing interval must not rescue a dip: no interval is not a wide one.
    ("monotone_verdict: a missing interval counts as overlapping", F,
     b"        overlap = (None if None in (lo_a, hi_a, lo_b, hi_b)\n"
     b"                   else (lo_a <= hi_b and lo_b <= hi_a))",
     b"        overlap = (True if None in (lo_a, hi_a, lo_b, hi_b)\n"
     b"                   else (lo_a <= hi_b and lo_b <= hi_a))",
     [f"{TF}::test_monotone_verdict_treats_a_missing_interval_as_not_overlapping"]),

    # A two-valued verdict on an ordered comparison is not total (gotcha 24):
    # "nothing was measurable" must not read as "not monotone".
    ("monotone_verdict: the too-few-bins outcome is removed", F,
     b'        return {"verdict": "UNDEFINED_TOO_FEW_BINS", "n_usable_bins": len(use)}',
     b'        return {"verdict": "NOT_MONOTONE", "n_usable_bins": len(use)}',
     [f"{TF}::test_monotone_verdict_has_a_third_outcome_for_too_few_bins"]),

    # An absent seed counted as a zero drags every combined mean toward 0.
    ("combine_across_seeds: a bin absent in a seed contributes a zero", F,
     b'        vals = [r["mean_r_f"] for s in per_seed for r in s["bins"]\n'
     b'                if r["bin"] == b and r.get("status") == "COMPUTED"]',
     b'        vals = [r.get("mean_r_f", 0.0) for s in per_seed '
     b'for r in s["bins"]\n                if r["bin"] == b]',
     [f"{TF}::test_a_bin_absent_in_one_seed_is_absent_not_zero"]),

    # The seed axis by position rather than by label -- 18 of 20 rows wrong,
    # every identity key still agreeing (gotcha 29).
    ("seed_permutation: rows are taken in arrival order", F,
     b"    where = {v: i for i, v in enumerate(src_labels)}\n"
     b"    return np.array([where[v] for v in dst_labels], dtype=np.int64)",
     b"    return np.arange(len(dst_labels), dtype=np.int64)",
     [f"{TF}::test_seed_permutation_maps_by_label_and_differs_from_the_identity"]),

    ("seed_permutation: a different seed set is aligned anyway", F,
     b"    if sorted(src_labels) != sorted(dst_labels):",
     b"    if False:",
     [f"{TF}::test_seed_permutation_refuses_a_different_seed_set"]),

    # --------------------------------------------- concept_headline_checks --
    # THE DEDUP. Without it the realised count is the SOURCE count on every
    # draw, so the measurement would report exact preservation always -- a
    # confident wrong answer of precisely the shape this project keeps finding.
    ("realised_positive_counts: the collision dedup is dropped", H,
     b"        out[r] = np.unique(shifted).size",
     b"        out[r] = shifted.size",
     [f"{TH}::test_realised_counts_LOSE_rows_when_the_destination_is_shorter",
      f"{TH}::test_realised_counts_report_one_value_per_stored_shift"]),

    ("drift_summary: a draw above the observed count is not counted", H,
     b'        "n_draws_above_observed": int((realised > observed).sum()),',
     b'        "n_draws_above_observed": 0,',
     [f"{TH}::test_drift_summary_counts_a_draw_ABOVE_observed_rather_than_assuming_none"]),

    ("preservation_verdict: the 1% band swallows the 5% one", H,
     b'    elif pop["share_of_concepts_with_median_drift_within_1_percent"] >= 0.95:',
     b'    elif pop["share_of_concepts_with_median_drift_within_5_percent"] >= 0.95:',
     [f"{TH}::test_preservation_verdict_is_total_over_its_stated_bands"]),

    ("bounded_headline: the fragility bound becomes inclusive", H,
     b"    fragile = finite & (worst_margin < bound)",
     b"    fragile = finite & (worst_margin <= bound)",
     [f"{TH}::test_the_bound_is_STRICT_so_a_margin_of_exactly_2_is_not_fragile"]),

    # A concept that could not be classified swept into the fragile set would
    # be removed from the headline on the strength of a NaN.
    ("bounded_headline: a non-finite margin is treated as fragile", H,
     b"    fragile = finite & (worst_margin < bound)",
     b"    fragile = ~(worst_margin >= bound)",
     [f"{TH}::test_a_non_finite_margin_is_counted_separately_not_swept_into_a_group"]),

    # THE INFORMATIVE ARMS key on the fragile DETECTIONS. Keying them on every
    # fragile concept instead is the tautology this section exists to avoid --
    # and it is exactly what the first implementation did.
    ("bounded_headline: the informative arm keys on every fragile concept", H,
     b"    keep = ~fragile_det",
     b"    keep = ~fragile",
     [f"{TH}::test_bounded_headline_keys_on_the_FRAGILE_DETECTIONS"]),

    # The degeneracy flag asserted rather than derived: a flag that cannot come
    # out False is not a measurement (gotcha 25).
    ("bounded_headline: the degeneracy flag is a constant", H,
     b"    is_taut = bool(under.sum() and bool((fragile[under]).all()))",
     b"    is_taut = True",
     [f"{TH}::test_the_degeneracy_flag_is_FALSE_when_an_undetected_concept_is_secure"]),

    # ------------------------------------------------------- never_detected --
    # The AO6 shape: a check on a total cannot see a permutation (gotcha 27).
    ("verify_block_starts: the positional check becomes a total", N,
     b"    ok = int((want == starts).sum())",
     b"    ok = int(want.size) if want.sum() == starts.sum() else 0",
     [f"{TN}::test_verify_block_starts_REFUSES_a_permutation_with_the_SAME_TOTAL",
      f"{TN}::test_verify_block_starts_catches_a_SINGLE_displaced_block"]),

    # ---------------------------------------------------- carrier_support 2 --
    # The two reductions of one array. This mutation is the production failure
    # that took the first run down, frozen as a test.
    ("feature_best_ap: the max is taken over FEATURES instead of concepts", S,
     b"    return ap_pool.max(axis=2)",
     b"    return ap_pool.max(axis=1)",
     [f"{TS}::test_feature_best_ap_is_the_max_over_CONCEPTS_not_over_features"]),

    ("statistic_agreement: the shape guard is removed", S,
     b"    if best_ap.shape != M.shape:",
     b"    if False:",
     [f"{TS}::test_statistic_agreement_refuses_the_wrong_reduction"]),

    # ------------------------------------------------- alignment_fine_panel 2 --
    ("combined verdict: computed on the per-seed counts instead of the means", F,
     b"            \"monotone_on_the_combined_sequence\": monotone_verdict(combined_rows),",
     b"            \"monotone_on_the_combined_sequence\": {\"verdict\": \"MONOTONE_NON_DECREASING\"},",
     [f"{TF}::test_the_combined_verdict_is_derived_from_the_combined_means"]),

    ("taxonomy_counts: the protein count is dropped in favour of a share", N,
     b'        "n_proteins_with_taxonomy": n,',
     b'        "n_proteins_with_taxonomy": 0,',
     [f"{TN}::test_taxonomy_counts_reports_COUNTS_as_well_as_a_share"]),

    ("group_summary: an empty group returns zeros instead of naming itself", N,
     b'        return {"n": 0, "status": "EMPTY_GROUP"}',
     b'        return {"n": 0}',
     [f"{TN}::test_group_summary_names_an_empty_group_rather_than_inventing_zeros"]),
]


def main() -> int:
    fixtures = sorted({t for e in MUTATIONS for t in e[4]} | set(CONTROL[4]))
    failing = [t for t in fixtures if not run_one(t)]
    if failing:
        print("BASELINE BROKEN -- these fail before any mutation:")
        for t in failing:
            print(f"   {t}")
        return 1
    print(f"baseline: {len(fixtures)}/{len(fixtures)} fixtures pass")

    name, path, old, new, fx = CONTROL
    survived = apply_and_run(path, old, new, fx)
    if survived is None:
        print(f"CONTROL ANCHOR NOT FOUND in {path} -- aborting. The runner "
              "cannot be trusted to detect anything.")
        return 2
    if survived:
        print(f"CONTROL NOT DETECTED ({name}): {survived}. Aborting; no "
              "SURVIVED verdict from this run would mean anything.")
        return 2
    print(f"control detected: {name}")

    killed, problems = 0, []
    for name, path, old, new, fx in MUTATIONS:
        survived = apply_and_run(path, old, new, fx)
        if survived is None:
            problems.append(f"ANCHOR NOT FOUND (or not unique): {name}")
            print(f"  ANCHOR  {name}")
            continue
        if survived:
            problems.append(f"SURVIVED: {name} -> {survived}")
            print(f"  SURVIVED {name}")
        else:
            killed += 1
            print(f"  killed   {name}")

    print(f"\n{killed}/{len(MUTATIONS)} mutations killed, control detected")
    for p in problems:
        print(f"  {p}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
