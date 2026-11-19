"""Mutation matrix for amendment BB's new code. NOT collected by pytest.

Covers `fitted_vs_observed` (BB section 2), `carrier_concentration` and
`carrier_support`'s `value_name` generalisation (BB section 3), and `canary`'s
addition-vs-rebaseline instrument (BB section 4).

Run manually:  python tests/mutation_matrix_bb.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

The runner is reused from `mutation_matrix_av` -- AU9b's control mutation,
AV22's `__pycache__` purge between invocations, the byte-compared restore, and
the decide-on-EXIT-CODE rule. Reused rather than re-implemented: a runner is a
check, and a second copy is a second thing that can silently stop detecting
(gotcha 13 extension 5).

THE TWO THIS FILE EXISTS FOR:
  * `cross_path_agreement` comparing TOTALS instead of CELLS. The whole premise
    of BB section 2 is that the fitted and observed arms describe the same
    features, and a count cannot see a permutation (gotcha 27). The mutation
    that matters is not a malformed input -- it is the check quietly becoming
    one a permutation passes.
  * `compare_manifests` failing to distinguish an ADDITION from a REBASELINE.
    BB section 4 authorises the first and not the second, and `capture` writes
    whatever is on disk, so the distinction lives entirely in that function.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run, run_one  # noqa: E402

ROOT = os.getcwd()
V = os.path.join(ROOT, "src", "analysis", "fitted_vs_observed.py")
C = os.path.join(ROOT, "src", "analysis", "carrier_concentration.py")
S = os.path.join(ROOT, "src", "analysis", "carrier_support.py")
N = os.path.join(ROOT, "src", "analysis", "canary.py")
TV = "tests/test_fitted_vs_observed.py"
TC = "tests/test_carrier_concentration.py"
TN = "tests/test_canary.py"

# THE CONTROL: the span comparison returns a constant verdict before any band
# is evaluated. This cannot fail to break the suite --
# `test_every_band_of_the_span_comparison_is_reachable` is parametrised over
# FOUR DISTINCT verdicts and a constant collapses them to one.
#
# It is deliberately NOT a `reading` string or a label: BA17 item 2's first
# control replaced a `why` STRING while leaving the verdict alone, so all four
# outcomes stayed reachable and the runner correctly reported CONTROL NOT
# DETECTED. The anchor below returns before any branch runs.
CONTROL = (
    "CONTROL -- span_comparison_verdict returns a constant before any branch",
    V,
    b'    f = fitted["mean_span_pp"]\n'
    b'    o = observed["mean_span_pp"]',
    b'    return {"verdict": "FITTED_SPAN_CLOSE_TO_OBSERVED",\n'
    b'            "ratio_fitted_over_observed": 1.0, "reading": "control"}\n'
    b'    f = fitted["mean_span_pp"]\n'
    b'    o = observed["mean_span_pp"]',
    [f"{TV}::test_every_band_of_the_span_comparison_is_reachable"],
)

MUTATIONS = [
    # =================================================== BB 2: the premise
    # THE ONE THIS FILE EXISTS FOR. A total cannot see a permutation, and the
    # fixture's two masks have the SAME count and different cells -- so a
    # fixture that differed in count would let this mutation live.
    ("cross_path_agreement: `keep` compared by TOTAL, not cell by cell", V,
     b"    dk = int((keep_join != keep_panel).sum())",
     b"    dk = int(keep_join.sum() != keep_panel.sum())",
     [f"{TV}::test_the_premise_check_detects_a_PERMUTATION_with_the_SAME_TOTAL"]),

    ("cross_path_agreement: `log M` compared by SUM, not cell by cell", V,
     b"    dm = int((logM_join != logM_panel).sum())",
     b"    dm = int(logM_join.sum() != logM_panel.sum())",
     [f"{TV}::test_the_premise_check_sees_a_logM_permutation_that_preserves_the_SUM"]),

    # ================================================= BB 2: ACCEPTANCE A
    # The criterion that makes "this is the EXISTING fit" a measurement rather
    # than an assertion. Without it a refit that quietly differed would be
    # reported as the fit of record.
    ("reproduces_join: problems no longer change the verdict", V,
     b'        "verdict": ("REPRODUCES_THE_STORED_FIT_EXACTLY" if not problems\n'
     b'                    else "DIFFERS"),',
     b'        "verdict": "REPRODUCES_THE_STORED_FIT_EXACTLY",',
     # NOT listed: `..._different_seed_count`. That case returns "DIFFERS"
     # from an EARLY return above the mutated expression, so it cannot reach
     # the mutated line -- a fixture that cannot exercise the mutated code is
     # not a fixture that can kill it (BA's own note on the empty-pool case).
     [f"{TV}::test_acceptance_A_fails_on_a_coefficient_moved_in_the_LAST_BIT",
      f"{TV}::test_acceptance_A_reports_an_ABSENT_key_as_a_problem_not_agreement"]),

    # Gotcha 15: an ABSENT key read as agreement.
    ("reproduces_join: an absent field is skipped instead of flagged", V,
     b'                problems.append(f"seed {mine[\'seed\']}: {k} ABSENT from the "\n'
     b'                                f"reference (absent is not matching)")',
     b"                pass",
     [f"{TV}::test_acceptance_A_reports_an_ABSENT_key_as_a_problem_not_agreement"]),

    # A tolerance instead of exact equality would accept a genuinely different
    # fit whose coefficients happened to land close.
    ("reproduces_join: exact equality relaxed to a tolerance", V,
     b"            same = (a == b)\n"
     b"            if not same:\n"
     b'                problems.append(f"seed {mine[\'seed\']}: {k} {b!r} vs {a!r}")',
     b"            same = (a == b or (isinstance(a, float) and "
     b"isinstance(b, float) and abs(a - b) < 1e-6))\n"
     b"            if not same:\n"
     b'                problems.append(f"seed {mine[\'seed\']}: {k} {b!r} vs {a!r}")',
     [f"{TV}::test_acceptance_A_fails_on_a_coefficient_moved_in_the_LAST_BIT"]),

    # ================================================ BB 2: the fitted array
    # A 0.0 outside the analysis floor reads as "fitted, and zero" and is then
    # averaged into a bin.
    ("refit_primary: cells outside the floor are 0.0 rather than NaN", V,
     b"    fitted = np.full((n_seeds, n_features), np.nan, dtype=np.float64)",
     b"    fitted = np.zeros((n_seeds, n_features), dtype=np.float64)",
     [f"{TV}::test_cells_outside_the_floor_are_NaN_never_zero"]),

    # ================================================== BB 2: the verdict
    # Gotcha 25b: a ratio against a denominator whose interval includes zero
    # ranks noise. Removing the branch reports the model as falling short on
    # a stratum where nothing was measured.
    ("span_comparison_verdict: the straddles-zero branch is unreachable", V,
     b"    obs_flat = (lo is None or hi is None or (lo <= 0.0 <= hi))",
     b"    obs_flat = False",
     [f"{TV}::test_a_straddling_observed_interval_is_UNINFORMATIVE_not_a_shortfall",
      f"{TV}::test_a_negative_observed_span_does_not_produce_a_positive_ratio"]),

    ("span_comparison_verdict: the CLOSE band swallows the BELOW_HALF band", V,
     b"    elif ratio < 0.8:",
     b"    elif ratio < 0.5:",
     [f"{TV}::test_every_band_of_the_span_comparison_is_reachable"]),

    ("span_comparison_verdict: EXCEEDS is unreachable", V,
     b"    elif ratio <= 1.25:",
     b"    elif ratio <= 1e9:",
     [f"{TV}::test_every_band_of_the_span_comparison_is_reachable"]),

    ("paired_span: the span is not converted to percentage points", V,
     b'        "mean_span_pp": mean * 100.0,',
     b'        "mean_span_pp": mean,',
     [f"{TV}::test_paired_span_is_the_within_seed_difference_combined_across_seeds"]),

    ("score_bb_p1: the half cut moves to 1.0", V,
     b'        outcome = "UNDER_HALF" if r < 0.5 else "HALF_OR_ABOVE"',
     b'        outcome = "UNDER_HALF" if r < 1.0 else "HALF_OR_ABOVE"',
     [f"{TV}::test_bb_p1_is_total_over_the_ratio_including_the_undefined_case"]),

    # BA17 item 3: a formatter that cannot render a legitimately absent value.
    ("_fmt: a bool is formatted as a number", V,
     b"    return (f\"{x:+.{dp}f}\".rjust(width) if isinstance(x, (int, float))\n"
     b"            and not isinstance(x, bool) else \"n/a\".rjust(width))",
     b"    return (f\"{x:+.{dp}f}\".rjust(width) if isinstance(x, (int, float))\n"
     b"            else \"n/a\".rjust(width))",
     [f"{TV}::test_the_formatter_renders_an_absent_span_rather_than_raising"]),

    # ============================== BB 3: the generalisation is a LABEL
    # If `value_name` changed the arithmetic, BB's concentration answer would
    # not be comparable with BA10's homology answer -- the entire reason the
    # two go through one function.
    ("support_distribution: the stratum key ignores value_name", S,
     b'            f"median_percentile_in_peer_{value_name}": float(np.median(v)),',
     b'            "median_percentile_in_peer_d_fs": float(np.median(v)),',
     [f"{TC}::test_value_name_renames_keys_and_changes_no_number"]),

    ("support_distribution: the default value_name is no longer d_fs", S,
     b'                         n_fbins: int, value_name: str = "d_fs") -> dict:',
     b'                         n_fbins: int, value_name: str = "value") -> dict:',
     # NOT listed: `..._moved_STRATUM_value`. It perturbs
     # `shift_nats_mean_over_seeds`, whose key does NOT carry `value_name`, so
     # it still DIFFERS under the mutation and cannot exhibit the condition.
     [f"{TC}::test_the_default_value_name_is_d_fs_so_BA10_is_unchanged"]),

    # =================================== BB 3: the enrichment reading
    # `had_something_to_remove` is the grid's key and is NOT a synonym for
    # ENRICHED -- a DEPLETED covariate is one the two groups differ on.
    ("enrichment_verdict: DEPLETED counts as nothing to remove", C,
     b'        "had_something_to_remove": bool(v != "NOT_MATERIALLY_ENRICHED"),',
     b'        "had_something_to_remove": bool(v == "ENRICHED"),',
     [f"{TC}::test_every_enrichment_band_is_reachable_and_the_binary_is_not_ENRICHED"]),

    ("enrichment_verdict: the ENRICHED band starts at 50 instead of 55", C,
     b"    if m >= ENRICHED_AT:",
     b"    if m >= 50.0:",
     [f"{TC}::test_every_enrichment_band_is_reachable_and_the_binary_is_not_ENRICHED"]),

    ("enrichment_verdict: the DEPLETED band is unreachable", C,
     b"    elif m >= DEPLETED_BELOW:",
     b"    elif m >= 0.0:",
     [f"{TC}::test_every_enrichment_band_is_reachable_and_the_binary_is_not_ENRICHED"]),

    ("enrichment_verdict: an uncomputed distribution returns False not None", C,
     b'            "had_something_to_remove": None,',
     b'            "had_something_to_remove": False,',
     [f"{TC}::test_an_uncomputed_distribution_is_NOT_COMPUTED_not_a_false"]),

    # ========================================= BB 3: the outcome grid
    # The cell declared impossible. The one time this project left a cell
    # unmapped, the data landed in it (AC9, X2).
    ("outcome_grid: the declared-impossible cell is unreachable", C,
     b'        cell = "NOT_ENRICHED_AND_SHRINKS__DECLARED_IMPOSSIBLE"',
     b'        cell = "NOT_ENRICHED_AND_SURVIVES"',
     [f"{TC}::test_all_four_cells_of_the_grid_are_reachable"]),

    ("outcome_grid: the shrink cut moves to 1.0", C,
     b"    shrinks = bool(ratio < SHRINKS_MATERIALLY_BELOW_FRACTION)",
     b"    shrinks = bool(ratio < 1.0)",
     [f"{TC}::test_all_four_cells_of_the_grid_are_reachable"]),

    ("outcome_grid: a non-finite excess still enters a cell", C,
     b"    if had is None or not np.isfinite(e1) or not np.isfinite(e2) or e1 == 0.0:",
     b"    if had is None:",
     [f"{TC}::test_the_grid_does_not_manufacture_a_cell_from_an_undefined_input"]),

    # ================================= BB 3: ACCEPTANCE 4 (BA9 reproduction)
    ("reproduces_ba9: problems no longer change the verdict", C,
     b'        "verdict": "REPRODUCES_BA9_EXACTLY" if not problems else "DIFFERS",',
     b'        "verdict": "REPRODUCES_BA9_EXACTLY",',
     [f"{TC}::test_acceptance_4_fails_on_a_median_moved_in_the_last_bit",
      f"{TC}::test_acceptance_4_fails_on_a_moved_STRATUM_value",
      f"{TC}::test_acceptance_4_reports_an_ABSENT_key_as_a_problem"]),

    # A pooled median can agree while a stratum disagrees: checking only the
    # pooled number would be a check on a total (gotcha 27).
    ("reproduces_ba9: the per-stratum rows are not compared", C,
     b'    for r in dist.get("by_firing_stratum", []):',
     b"    for r in []:",
     [f"{TC}::test_acceptance_4_fails_on_a_moved_STRATUM_value"]),

    ("reproduces_ba9: an absent field is skipped instead of flagged", C,
     b'            problems.append(f"{k}: ABSENT (absent is not matching)")\n'
     b"            continue\n"
     b"        cmp(k, dist.get(k), ref[k])",
     b"            continue\n"
     b"        cmp(k, dist.get(k), ref[k])",
     [f"{TC}::test_acceptance_4_reports_an_ABSENT_key_as_a_problem"]),

    ("reproduces_ba_arm: a moved rate no longer differs", C,
     b'            "verdict": f"REPRODUCES_BA_{which.upper()}_EXACTLY"\n'
     b'                       if not problems else "DIFFERS"}',
     b'            "verdict": f"REPRODUCES_BA_{which.upper()}_EXACTLY"}',
     # NOT listed: `..._ABSENT_arm_rather_than_passing`, which returns
     # "DIFFERS" from the early return above the mutated expression.
     [f"{TC}::test_acceptance_3_passes_and_fails_in_both_directions"]),

    # ===================================== BB 3: cell usability
    ("cell_usability: cells AT the bar are counted as unusable", C,
     b"    below = counts < min_usable",
     b"    below = counts <= min_usable",
     [f"{TC}::test_cell_usability_counts_cells_BELOW_the_bar"]),

    # ===================================== BB 4: addition vs rebaseline
    # THE SECOND ONE THIS FILE EXISTS FOR. `capture` writes whatever is on
    # disk, so the whole distinction lives in this function.
    ("compare_manifests: a moved baseline is not reported", N,
     b"        if moved:\n"
     b"            rebaselined.append({\"name\": name, \"fields\": moved})",
     b"        if False:\n"
     b"            rebaselined.append({\"name\": name, \"fields\": moved})",
     [f"{TN}::test_a_pre_existing_target_whose_bytes_moved_is_a_REBASELINE_not_an_addition"]),

    ("compare_manifests: sha256 is not a baseline field", N,
     b'BASELINE_FIELDS = ("arm", "path", "hash_scope", "sha256", "n_leaves",\n'
     b'                   "first_baselined_at", "bytes", "status")',
     b'BASELINE_FIELDS = ("arm", "path", "hash_scope", "n_leaves",\n'
     b'                   "first_baselined_at", "status")',
     # The ADD-A-KEY fixture also moves `n_leaves` and `bytes`, so it cannot
     # isolate the hash. The bytes-only fixture can, and is what kills this.
     [f"{TN}::test_a_drift_that_changes_ONLY_the_BYTES_is_still_a_rebaseline",
      f"{TN}::test_captured_at_is_excluded_so_the_check_is_not_one_that_can_only_fail"]),

    # Gotcha 25a: a field that moves by construction would make EVERY capture
    # read as a rebaseline -- a check that can only fail, on the instrument
    # built to detect a rebaseline.
    ("compare_manifests: captured_at joins the baseline fields", N,
     b'                   "first_baselined_at", "bytes", "status")',
     b'                   "first_baselined_at", "bytes", "status", "captured_at")',
     # The NO_CHANGE fixture's two captures share a HEAD, so `captured_at`
     # is EQUAL there and the fixture cannot exhibit the condition. The
     # hand-built pair can.
     [f"{TN}::test_a_changed_captured_at_alone_is_NOT_a_rebaseline",
      f"{TN}::test_captured_at_is_excluded_so_the_check_is_not_one_that_can_only_fail"]),

    ("compare_manifests: a removed target is not a problem", N,
     b'        problems.append(f"{name}: present in the previous manifest and absent "\n'
     b'                        f"from this one")',
     b"        pass",
     [f"{TN}::test_a_target_that_vanishes_from_the_manifest_is_a_problem"]),

    # `first_baselined_at` reset on re-capture would back-date nothing -- it
    # would FORWARD-date everything, so "has reproduced since" would always
    # mean "since today".
    ("capture: first_baselined_at is reset to HEAD on every capture", N,
     b'                        "first_baselined_at": prev_first.get(spec["name"]) or sha,',
     b'                        "first_baselined_at": sha,',
     [f"{TN}::test_recapture_carries_first_baselined_at_forward"]),

    ("tolerance_coverage: a missing tolerance is not reported", N,
     b'            "verdict": "COVERED" if not missing else "UNCOVERED_ARM"}',
     b'            "verdict": "COVERED"}',
     # The real TARGETS are fully covered, so
     # `test_every_arm_named_by_a_target_has_a_declared_tolerance` passes
     # whether or not the branch exists. `tolerance_coverage` was made
     # INJECTABLE so a fixture could exhibit the condition at all.
     [f"{TN}::test_tolerance_coverage_REPORTS_an_arm_that_has_no_tolerance"]),

    # The predictor target folded into an existing arm would give it n39's
    # tolerance -- which claims a baseline history it does not have.
    ("TARGETS: the alignment predictor is filed under the n39 arm", N,
     b'    {"name": "alignment_predictor", "arm": "predictor",',
     b'    {"name": "alignment_predictor", "arm": "n39",',
     [f"{TN}::test_all_three_arms_are_present_as_separate_targets",
      f"{TN}::test_the_alignment_predictor_is_a_target_on_its_own_arm"]),

    ("TARGETS: the alignment predictor points at the SMOOTH gtilde", N,
     b'     "path": "gtilde_compliant_b_ratio.h5",\n'
     b'     "hash_scope": "whole_file",\n'
     b'     "why": "THE PREDICTOR OF RECORD',
     b'     "path": "gtilde_compliant_b_ratio_smooth_aligned.h5",\n'
     b'     "hash_scope": "whole_file",\n'
     b'     "why": "THE PREDICTOR OF RECORD',
     [f"{TN}::test_the_alignment_predictor_is_a_target_on_its_own_arm"]),
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
        print(f"CONTROL ABORT: {label} was NOT detected (survivors: "
              f"{survived}). Every SURVIVED verdict this runner could print "
              "would be meaningless, so none is printed.")
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
        print(f"{label:<62} {status:<10} "
              f"{len(expect) - len(survived)}/{len(expect)}")
        for t in survived:
            print(f"{'':<62} {'':<10} SURVIVOR: {t}")
        bad += bool(survived)

    print()
    print("ALL MUTATIONS KILLED THEIR NAMED FIXTURES" if not bad
          else f"{bad} MUTATION(S) SURVIVED OR WERE SKIPPED")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
