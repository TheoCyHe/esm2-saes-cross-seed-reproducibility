"""Mutation matrix for amendment AX's new code. NOT collected by pytest.

Covers `concept_repro` (AW §1's detection rate, run under AX §2d) and
`mask_fraction` (AX §1a).

Run manually:  python tests/mutation_matrix_ax.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

The runner itself is reused from `mutation_matrix_av`, which carries AU9b's
control mutation, AV22's `__pycache__` purge between invocations, the
byte-compared restore, and the decide-on-exit-code rule. Reused rather than
re-implemented precisely because the runner is a check and a second copy is a
second thing that can silently stop detecting (gotcha 13 extension 5).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run, run_one  # noqa: E402

ROOT = os.getcwd()
R = os.path.join(ROOT, "src", "analysis", "concept_repro.py")
K = os.path.join(ROOT, "src", "analysis", "mask_fraction.py")
TR = "tests/test_concept_repro.py"
TK = "tests/test_mask_fraction.py"

# THE CONTROL: make every concept detected in every seed. This cannot fail to
# break the suite -- the recovery fixture plants a three-valued pattern
# (0, 0.5, 1) and asserts it exactly. If the runner reports this undetected,
# none of its SURVIVED verdicts mean anything and none is printed.
CONTROL = (
    "CONTROL -- detection replaced by a constant True", R,
    b"    return observed_gain > thresholds",
    b"    return np.ones(observed_gain.shape, dtype=bool)",
    [f"{TR}::test_the_pipeline_recovers_the_planted_detection_pattern_exactly"],
)

MUTATIONS = [
    # ---------------------------------------------------- concept_repro --
    # Strict exceedance. AW1.2 says "exceeds"; `>=` detects a concept whose
    # best feature merely TIES the null's 95th percentile.
    ("detection: strict `>` becomes `>=`", R,
     b"    return observed_gain > thresholds",
     b"    return observed_gain >= thresholds",
     [f"{TR}::test_detection_is_strict_exceedance_not_greater_or_equal"]),

    # The max must be over FEATURES. Over seeds it would collapse the very axis
    # D(c) counts, and still return a (n, m) array of plausible numbers.
    ("A(c,s): max over features becomes max over seeds", R,
     b"    return ap.max(axis=1)          # -> (seed, concept)",
     b"    return ap.max(axis=0)          # -> (seed, concept)",
     [f"{TR}::test_best_ap_maxes_over_features_not_seeds_or_concepts",
      f"{TR}::test_the_pipeline_recovers_the_planted_detection_pattern_exactly"]),

    # Estimability. Dropping the bound admits the 99.9th percentile of 400
    # draws -- the sample maximum wearing a different label.
    ("estimability: the >= 1 draw bound is removed", R,
     b'            "estimable": bool(above >= 1.0),',
     b'            "estimable": bool(above >= 0.0),',
     [f"{TR}::test_400_pooled_draws_support_the_95th_but_not_the_999th",
      f"{TR}::test_20_per_seed_draws_stop_at_the_95th",
      f"{TR}::test_estimability_reports_the_count_for_percentiles_it_rejects",
      f"{TR}::test_the_written_json_carries_the_curve_and_names_dropped_percentiles"]),

    # The numerically-stable form of the same arithmetic. At n=20, q=95 the
    # unstable form gives 1.0000000000000009, so a boundary verdict turns on
    # the last bit. Both forms are "correct"; only one is decidable.
    ("estimability: the stable form reverts to n*(1 - q/100)", R,
     b"        above = n * (100.0 - float(q)) / 100.0",
     b"        above = n * (1.0 - float(q) / 100.0)",
     [f"{TR}::test_400_pooled_draws_support_the_95th_but_not_the_999th",
      f"{TR}::test_20_per_seed_draws_stop_at_the_95th"]),

    # Positional label alignment (gotcha 29). Without it a lexicographic and a
    # numeric seed axis join silently, mis-pairing 18 of 20 rows.
    ("label alignment: the elementwise comparison is disabled", R,
     b"    bad = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]",
     b"    bad = []",
     [f"{TR}::test_refuses_a_permuted_seed_axis",
      f"{TR}::test_refuses_a_permuted_concept_axis",
      f"{TR}::test_label_alignment_refuses_a_permutation_of_the_same_set"]),

    # The shared-feature-axis guard. AW1.4's selection control is valid ONLY
    # while both maxima range over the same features; a permutation preserves
    # every total and every rank correlation (gotcha 27).
    ("feature axis: the positional agreement check is disabled", R,
     b"    if n_ok != n_cells:",
     b"    if False:",
     [f"{TR}::test_refuses_a_permuted_feature_axis"]),

    # Pooled vs per-seed are different reductions of the same array.
    ("null threshold: pooled reduction uses seed 0 only", R,
     b"        flat = null_cmax.reshape(-1, null_cmax.shape[2])",
     b"        flat = null_cmax[0]",
     [f"{TR}::test_pooled_and_per_seed_thresholds_are_different_reductions"]),

    # Per-concept prevalence. Broadcasting the wrong axis would normalise every
    # concept by the first one's p and still produce finite gains everywhere.
    ("gain: the per-concept prevalence length guard is removed", R,
     b"    if p.ndim != 1 or p.shape[0] != best_ap.shape[-1]:",
     b"    if False:",
     [f"{TR}::test_gain_refuses_a_prevalence_of_the_wrong_length"]),

    # Null B, not Null A. Consuming the circular shift answers a different
    # question (localisation, not total grounding) with no visible symptom.
    ("null B: the dest_schedule provenance check is disabled", R,
     b'        if "dest_schedule" not in hf:',
     b"        if False:",
     [f"{TR}::test_refuses_a_null_without_dest_schedule"]),

    # An ABSENT identity key must never read as agreement (gotcha 15 / I3).
    ("identity: absent keys are treated as matching", R,
     b'            identity_report[k]["agree"] = False   # absent is never matching',
     b'            pass',
     [f"{TR}::test_an_absent_identity_key_is_refused_never_treated_as_matching"]),

    # A constant statistic is UNINFORMATIVE, not zero (gotcha 25b).
    ("spearman: a constant input returns 0.0 instead of refusing", R,
     b"    if denom == 0.0:\n        raise ValueError(",
     b"    if denom < 0.0:\n        raise ValueError(",
     [f"{TR}::test_spearman_refuses_a_constant_input_rather_than_reporting_zero"]),

    # Average ranks. Ordinal ranks make the correlation depend on input order.
    ("spearman: tie-averaging is skipped", R,
     b"        if j > i:\n            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()",
     b"        if False:\n            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()",
     [f"{TR}::test_spearman_uses_average_ranks_so_ties_do_not_follow_input_order"]),

    # ---------------------------------------------------- mask_fraction --
    # Masked share vs kept share -- a 4.21% figure and a 95.79% one.
    ("mask fraction: the masked share becomes the KEPT share", K,
     b"    return (n_rows - eff).astype(np.float64) / float(n_rows)",
     b"    return eff.astype(np.float64) / float(n_rows)",
     [f"{TK}::test_fraction_is_the_masked_share_not_the_kept_share",
      f"{TK}::test_an_unmasked_concept_is_exactly_zero_not_merely_small",
      f"{TK}::test_reproduces_aw5s_transmem_figure_from_the_writers_own_arithmetic"]),

    # Masking can only REMOVE rows; an effective count above the total means
    # the two numbers came from different runs.
    ("mask fraction: the n_eff <= n_rows guard is removed", K,
     b"    if eff.size and (eff > n_rows).any():",
     b"    if False:",
     [f"{TK}::test_effective_above_total_is_refused_because_masking_only_removes"]),

    # setdiff is directional: an experimental row outside the pooled set must
    # not subtract from the masked count.
    ("mask counts: the set difference is reversed", K,
     b"        [np.setdiff1d(pooled.get(c, empty), experimental.get(c, empty),",
     b"        [np.setdiff1d(experimental.get(c, empty), pooled.get(c, empty),",
     [f"{TK}::test_masked_count_is_the_non_experimental_remainder",
      f"{TK}::test_the_set_difference_is_directional_pooled_minus_experimental",
      f"{TK}::test_a_concept_absent_from_pooled_contributes_zero_not_a_keyerror"]),

    # The branch rule reads the MAXIMUM, not the minimum -- "uniformly tiny"
    # is a statement about the worst case, and min() makes it fire whenever any
    # single concept is unmasked, which is almost always.
    ("branch: 'uniformly tiny' reads the min instead of the max", K,
     b"    uniformly_tiny = bool(max_frac < 0.01)",
     b"    uniformly_tiny = bool(float(f.min()) < 0.01)",
     [f"{TK}::test_pilot_low_with_substantial_tail_triggers_the_rerun_branch",
      f"{TK}::test_pilot_in_upper_half_takes_branch_a_by_sampling_when_tail_is_real",
      f"{TK}::test_the_fourth_region_is_named_rather_than_falling_through"]),

    # Percentile rank is the share STRICTLY below; counting ties as below puts
    # a concept at the top of a distribution it merely equals.
    ("percentile rank: ties count as below", K,
     b"    return 100.0 * float((s < float(value)).sum()) / float(s.size)",
     b"    return 100.0 * float((s <= float(value)).sum()) / float(s.size)",
     [f"{TK}::test_percentile_rank_is_the_share_strictly_below"]),

    # Exact zeros are counted separately from merely-small values, because
    # "nothing is masked" and "almost nothing is masked" support different
    # arguments (arithmetic vs sampling) in AX §1a's branch A.
    ("summary: exact zeros are counted as anything below 1e-6", K,
     b'        "n_exactly_zero": int((f == 0.0).sum()),\n        "share_exactly_zero": float((f == 0.0).mean()),',
     b'        "n_exactly_zero": int((f < 1e-6).sum()),\n        "share_exactly_zero": float((f < 1e-6).mean()),',
     [f"{TK}::test_summarise_counts_exact_zeros_separately_from_small_values"]),
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

    label, path, old, new, expect = CONTROL
    survived = apply_and_run(path, old, new, expect)
    if survived is None:
        print(f"CONTROL ABORT: anchor for {label!r} did not match exactly once. "
              "The runner cannot demonstrate that it detects anything.")
        return 2
    if survived:
        print(f"CONTROL ABORT: {label} was NOT detected (survivors: {survived}). "
              "Every SURVIVED verdict this runner could print would be "
              "meaningless, so none is printed.")
        return 2
    print(f"control: DETECTED ({label})\n")

    print(f"{'mutation':<62} {'result':<10} killed")
    bad = 0
    for label, path, old, new, expect in MUTATIONS:
        survived = apply_and_run(path, old, new, expect)
        if survived is None:
            print(f"{label:<62} {'SKIP':<10} anchor did not match exactly once")
            bad += 1
            continue
        status = "KILLED" if not survived else "SURVIVED"
        print(f"{label:<62} {status:<10} {len(expect) - len(survived)}/{len(expect)}")
        for t in survived:
            print(f"{'':<62} {'':<10} SURVIVOR: {t}")
        bad += bool(survived)

    print()
    print("ALL MUTATIONS KILLED THEIR NAMED FIXTURES" if not bad
          else f"{bad} MUTATION(S) NOT FULLY KILLED -- real coverage gap")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
