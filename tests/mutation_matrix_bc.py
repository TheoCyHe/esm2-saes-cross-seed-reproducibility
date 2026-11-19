"""Mutation matrix for amendment BC's new code. NOT collected by pytest.

Covers `perplexity_bound` (BC section 2 and its conditional follow-on) and
`canary`'s device-class coverage (BC section 4).

Run manually:  python tests/mutation_matrix_bc.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

The runner is reused from `mutation_matrix_av` -- AU9b's control mutation,
AV22's `__pycache__` purge between invocations, the byte-compared restore, and
the decide-on-EXIT-CODE rule. Reused rather than re-implemented: a runner is a
check, and a second copy is a second thing that can silently stop detecting
(gotcha 13 extension 5).

THE THREE THIS FILE EXISTS FOR:
  * **`outcome_cell` losing a branch.** The whole of BC5 is a conditional taken
    BY THAT FUNCTION rather than by a reader, so every branch it can lose is a
    branch that silently authorises -- or silently forbids -- a measurement.
    The NaN branch matters most: `nan < 1.5` is False, so removing it drops a
    NaN into LEADING_CANDIDATE, the single most alarming cell, reached by an
    absence of information.
  * **`join_rows_to_ap_order` accepting the IDENTITY permutation.** A bijection
    check alone passes under the identity, and the identity is exactly what a
    positional read of one of the two artifacts produces (gotcha 29). Only the
    non-triviality clause discriminates, and the mutation that matters is its
    quiet removal.
  * **`covariate_displacement` counting carriers of OTHER concepts as peers.**
    That moves the baseline toward the carriers by construction, which would
    SHRINK the displacement and therefore the bound -- a defect biased toward
    the reassuring answer.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run  # noqa: E402

ROOT = os.getcwd()
P = os.path.join(ROOT, "src", "analysis", "perplexity_bound.py")
N = os.path.join(ROOT, "src", "analysis", "canary.py")
TP = "tests/test_perplexity_bound.py"
TN = "tests/test_canary.py"

# THE CONTROL: `outcome_cell` returns a constant verdict before any branch is
# evaluated. This cannot fail to break the suite --
# `test_every_cell_of_the_grid_is_reachable` is parametrised over FIVE DISTINCT
# expected verdicts and a constant collapses them to one.
#
# It is deliberately NOT a `reading` string or a note: BA17 item 2's first
# control replaced a `why` STRING while leaving the verdict alone, so all
# outcomes stayed reachable and the runner correctly reported CONTROL NOT
# DETECTED. The anchor below returns before any branch runs.
CONTROL = (
    "CONTROL -- outcome_cell returns a constant before any branch",
    P,
    b"    share = (contribution_pp / excess_pp) if excess_pp != 0 else None",
    b'    return {"verdict": "PARTIAL_CONTRIBUTOR", "runs_followon": True,\n'
    b'            "contribution_pp": 0.0, "excess_pp": 0.0,\n'
    b'            "share_of_excess": None, "reading": "control"}\n'
    b"    share = (contribution_pp / excess_pp) if excess_pp != 0 else None",
    [f"{TP}::test_every_cell_of_the_grid_is_reachable"],
)

MUTATIONS = [
    # ============================================== BC 2: the outcome grid
    # THE ONE THIS FILE EXISTS FOR. `nan < cannot_below` is False and
    # `nan <= 0.0` is False, so without its own branch a NaN falls all the way
    # through to LEADING_CANDIDATE and AUTHORISES the follow-on on no
    # information at all.
    ("outcome_cell: the non-finite branch is unreachable", P,
     b"    if not np.isfinite(contribution_pp) or not np.isfinite(excess_pp):",
     b"    if False:",
     [f"{TP}::test_a_nan_contribution_gets_its_own_branch_not_the_smallest_cell",
      f"{TP}::test_a_nan_excess_is_also_undefined"]),

    # Zero means the two halves CANCEL, which is the disagreement cell, not the
    # "small contribution, close it" cell. The two readings are opposite.
    ("outcome_cell: zero falls into CANNOT_ACCOUNT instead of DISAGREE", P,
     b"    if contribution_pp <= 0.0:",
     b"    if contribution_pp < 0.0:",
     [f"{TP}::test_exactly_zero_is_a_disagreement_not_a_small_contribution",
      f"{TP}::test_every_cell_of_the_grid_is_reachable"]),

    # The 1.5 pp cut is what separates "close it" from "run the follow-on".
    ("outcome_cell: the CANNOT_ACCOUNT band swallows PARTIAL_CONTRIBUTOR", P,
     b"    elif contribution_pp < cannot_below:",
     b"    elif contribution_pp < 4.0:",
     [f"{TP}::test_every_cell_of_the_grid_is_reachable",
      f"{TP}::test_the_followon_runs_only_in_the_middle_two_cells"]),

    ("outcome_cell: LEADING_CANDIDATE is unreachable", P,
     b"    elif contribution_pp <= leading_above:",
     b"    elif contribution_pp <= 1e9:",
     [f"{TP}::test_every_cell_of_the_grid_is_reachable"]),

    # The branch BC5 is executed by. If it authorises everywhere, the
    # conditional is not a conditional.
    ("outcome_cell: the follow-on is authorised in every cell", P,
     b'        "runs_followon": bool(run),',
     b'        "runs_followon": True,',
     [f"{TP}::test_the_followon_runs_only_in_the_middle_two_cells",
      f"{TP}::test_a_positive_displacement_against_a_negative_coefficient_disagrees"]),

    # =============================================== BC 2: the seed-order join
    # THE SECOND ONE THIS FILE EXISTS FOR (gotcha 29). A bijection check alone
    # passes under the identity permutation, which is exactly what a positional
    # read of either artifact produces.
    ("join_rows_to_ap_order: the identity permutation is accepted", P,
     b"    if not problems and moved == 0:",
     b"    if False:",
     [f"{TP}::test_an_identity_permutation_is_REFUSED_not_accepted_as_a_bijection"]),

    ("join_rows_to_ap_order: a missing label is silently skipped", P,
     b'            problems.append(f"AP seed label {lab} has no row in the stored fit")',
     b"            pass",
     [f"{TP}::test_a_missing_label_is_a_problem_and_marks_the_row_minus_one"]),

    ("join_rows_to_ap_order: a duplicate label is not flagged", P,
     b'            problems.append(f"duplicate label {r[\'label\']} in the stored fit")',
     b"            pass",
     [f"{TP}::test_a_duplicate_label_in_the_stored_fit_is_a_problem"]),

    # The join is BY LABEL. Falling back to position is the defect itself.
    ("contribution_bound: the coefficient is taken by POSITION, not by label", P,
     b"        f = rows[perm[i]]",
     b"        f = rows[min(i, len(rows) - 1)]",
     [f"{TP}::test_the_join_permutation_is_HONOURED_not_ignored"]),

    # =============================================== BC 2: the displacement
    # THE THIRD ONE. Counting carriers of other concepts as ordinary peers
    # moves the baseline toward the carriers and SHRINKS the bound -- a defect
    # biased toward the reassuring answer.
    ("covariate_displacement: carriers of other concepts count as peers", P,
     b"            pool = value[s][in_bin & ~is_carrier[s]]",
     b"            pool = value[s][in_bin]",
     [f"{TP}::test_a_feature_carrying_ANOTHER_concept_is_not_an_ordinary_peer",
      f"{TP}::test_the_displacement_is_the_carrier_minus_its_own_bins_peer_MEAN"]),

    ("covariate_displacement: the peer MEAN becomes the peer MEDIAN", P,
     b"            peer_mean = float(pool.mean())",
     b"            peer_mean = float(np.median(pool))",
     [f"{TP}::test_the_displacement_is_the_carrier_minus_its_own_bins_peer_MEAN"]),

    # An empty pool scored 0.0 would read as "measured, and exactly average".
    ("covariate_displacement: an empty peer pool is not counted unusable", P,
     b"                unusable += int(sel.size)\n                continue",
     b"                continue",
     [f"{TP}::test_an_empty_peer_pool_is_counted_unusable_not_scored_as_zero"]),

    ("covariate_displacement: the primary reduction becomes the median", P,
     b'            "mean_displacement": float(d.mean()),',
     b'            "mean_displacement": float(np.median(d)),',
     [f"{TP}::test_the_displacement_is_the_carrier_minus_its_own_bins_peer_MEAN"]),

    # =============================================== BC 2: the straddle rule
    # BC4.4 declared it BEFORE the run precisely because the back-of-envelope
    # lands near the cut. A straddle that is not DETECTED is not reported.
    ("contribution_bound: a straddle is never detected", P,
     b"    straddles = (np.isfinite(mean_pp) and np.isfinite(med_pp)\n"
     b'                 and cell["verdict"] != med_cell["verdict"])',
     b"    straddles = False",
     [f"{TP}::test_a_straddle_is_DETECTED_and_the_mean_takes_the_branch"]),

    # The MEDIAN reduction must never take the branch (BC4.3).
    ("contribution_bound: the median reduction takes the branch", P,
     b"    cell = outcome_cell(mean_pp, excess_pp)",
     b"    cell = outcome_cell(med_pp, excess_pp) if False else outcome_cell("
     b"med_pp if np.isfinite(med_pp) else mean_pp, excess_pp)",
     [f"{TP}::test_a_straddle_is_DETECTED_and_the_mean_takes_the_branch"]),

    # ================================================= BC 2: across-seed arm
    ("combine_across_seeds: the SD uses ddof=0", P,
     b'    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")',
     b"    sd = float(v.std(ddof=0)) if v.size > 1 else float('nan')",
     [f"{TP}::test_combine_across_seeds_matches_meta_analyse_across_seed_arm"]),

    ("combine_across_seeds: the SE is the SD, unscaled by sqrt(n)", P,
     b'    se = sd / np.sqrt(v.size) if v.size > 1 else float("nan")',
     b"    se = sd if v.size > 1 else float('nan')",
     [f"{TP}::test_combine_across_seeds_matches_meta_analyse_across_seed_arm"]),

    # An uncomputable interval STATED as uncomputable is the point (BC4.3).
    ("combine_across_seeds: the absent fixed effect is reported as a number", P,
     b'        "fixed_effect": "NOT_AVAILABLE",',
     b'        "fixed_effect": 0.0,',
     [f"{TP}::test_the_fixed_effect_arm_is_reported_absent_with_its_reason"]),

    ("combine_across_seeds: non-finite seeds are kept rather than dropped", P,
     b"    v = v[np.isfinite(v)]",
     b"    v = v",
     [f"{TP}::test_non_finite_seeds_are_dropped_not_propagated",
      f"{TP}::test_no_usable_seeds_is_a_status_not_a_nan_mean"]),

    # ================================================== BC 2: ACCEPTANCE C
    ("acceptance_ame_reconstruction: any residual passes", P,
     b'        "verdict": ("REPRODUCES_THE_STORED_AME" if worst <= rtol\n'
     b'                    else "DIFFERS_FROM_THE_STORED_AME"),',
     b'        "verdict": "REPRODUCES_THE_STORED_AME",',
     [f"{TP}::test_acceptance_C_fails_on_a_relative_move_of_1e_minus_9"]),

    ("acceptance_ame_reconstruction: the tolerance is loosened to 1e-6", P,
     b"AME_RECONSTRUCTION_RTOL = 1e-12",
     b"AME_RECONSTRUCTION_RTOL = 1e-6",
     [f"{TP}::test_acceptance_C_fails_on_a_relative_move_of_1e_minus_9",
      f"{TP}::test_acceptance_C_reports_the_residual_whatever_the_verdict"]),

    ("acceptance_ame_reconstruction: no rows reads as a pass", P,
     b'        return {"verdict": "NOT_COMPUTED_NO_ROWS", "n_rows": 0}\n'
     b"    resid, n_exact = [], 0",
     b'        return {"verdict": "REPRODUCES_THE_STORED_AME", "n_rows": 0}\n'
     b"    resid, n_exact = [], 0",
     [f"{TP}::test_acceptance_C_on_no_rows_is_a_status_not_a_pass"]),

    # ================================================== BC 2: ACCEPTANCE F
    ("acceptance_bb93_summary: the sign count is not compared", P,
     b"          and n_pos == n_pos_ref)",
     b"          and True)",
     [f"{TP}::test_acceptance_F_fails_when_a_coefficient_changes_SIGN"]),

    ("acceptance_bb93_summary: comparison is rounded to 2 decimals", P,
     b"    ok = (round(mean, 5) == round(mean_ref, 5)\n"
     b"          and round(sd, 5) == round(sd_ref, 5)",
     b"    ok = (round(mean, 2) == round(mean_ref, 2)\n"
     b"          and round(sd, 2) == round(sd_ref, 2)",
     [f"{TP}::test_acceptance_F_fails_when_the_mean_moves_in_the_fifth_decimal"]),

    # ================================================== BC 2: ACCEPTANCE D
    # A probe with no control could only ever say CONFIRMED (gotcha 25).
    ("probe_verdict: the flipped control arm is not required", P,
     b"    ok = up > 0.0 and down < 0.0",
     b"    ok = up > 0.0",
     [f"{TP}::test_the_probes_verdict_requires_BOTH_arms"]),

    # Flipping BOTH the covariate and the outcome restores the same
    # relationship, so the "control" would come back POSITIVE and establish
    # nothing while looking like a check. This is the defect the first draft
    # of this probe actually had.
    ("sign_convention_probe: the control regenerates y, so it is not a control",
     P,
     b"    up = recover(x)\n    down = recover(-x)",
     b"    up = recover(x)\n"
     b"    y = rng.binomial(n_trials, expit(slope * -x)).astype(np.float64)\n"
     b"    down = recover(-x)",
     [f"{TP}::test_the_sign_convention_probe_confirms_and_its_control_is_the_opposite"]),

    # ================================================ BC 2: the label reader
    ("read_feature_side_arm: the label is taken from the row position", P,
     b'            "label": prov[pos],',
     b'            "label": str(pos),',
     [f"{TP}::test_read_feature_side_arm_takes_the_label_from_PROVENANCE"]),

    ("read_feature_side_arm: a row with no provenance is silently kept", P,
     b'            problems.append(f"row {pos} has no provenance record")\n'
     b"            continue",
     b"            continue",
     [f"{TP}::test_read_feature_side_arm_flags_a_row_with_no_provenance_record"]),

    # ================================================= BC 2: compare_exact
    ("compare_exact: an absent key is treated as agreement", P,
     b'            d, c = compare_exact(ref.get(k, "__ABSENT__"),\n'
     b'                                 now.get(k, "__ABSENT__"), f"{prefix}/{k}")',
     b"            d, c = compare_exact(ref.get(k), now.get(k), f'{prefix}/{k}')",
     [f"{TP}::test_compare_exact_treats_an_absent_key_as_a_difference"]),

    ("compare_exact: a list-length change is not a difference", P,
     b"        if len(ref) != len(now):",
     b"        if False:",
     [f"{TP}::test_compare_exact_sees_a_list_length_change"]),

    ("compare_exact: leaves are not counted, so a vacuous pass is invisible", P,
     b"    if ref != now:\n"
     b'        diffs[prefix or "/"] = {"reference": ref, "now": now}\n'
     b"    return diffs, 1",
     b"    if ref != now:\n"
     b'        diffs[prefix or "/"] = {"reference": ref, "now": now}\n'
     b"    return diffs, 0",
     [f"{TP}::test_compare_exact_counts_leaves_so_a_vacuous_pass_is_visible"]),

    ("_reproduces_stored_perplexity: an absent reference block passes", P,
     b'        return {"verdict": "REFERENCE_BLOCK_ABSENT",\n'
     b'                "reference": str(reference)}',
     b'        return {"verdict": "REPRODUCES_BB91_PERPLEXITY_EXACTLY",\n'
     b'                "reference": str(reference)}',
     [f"{TP}::test_acceptance_B_reports_an_absent_reference_block_rather_than_passing"]),

    # ====================================================== BC 2: pair weights
    ("pair_weights_from_detection: counts ALL seeds, not the later ones", P,
     b"            out[s] = det[s + 1:].sum(axis=0)",
     b"            out[s] = det.sum(axis=0)",
     [f"{TP}::test_pair_weights_count_only_the_seeds_ABOVE_each_row"]),

    ("pair_weights_from_detection: a non-detecting row still gets weight", P,
     b"    return out * det",
     b"    return out",
     [f"{TP}::test_pair_weights_are_zero_where_the_row_itself_does_not_detect"]),

    # ==================================================== BC 2: the formatter
    # BB5d: a completed measurement was reported as a FAILED job because a
    # summary print died on a legitimately UNDEFINED value.
    ("format_summary: the missing-bound branch raises instead of rendering", P,
     b'    if p.get("status") != "OK":\n'
     b"        out.append(f\"  bound: {p.get('status', 'ABSENT')}\")",
     b'    if p.get("status") != "OK":\n'
     b'        out.append(f"  bound: {p[\'status\']}")',
     [f"{TP}::test_the_formatter_is_TOTAL_over_an_empty_result"]),

    ("format_summary: an undefined share formats as a number", P,
     b'    out.append(f"  share of the excess: "\n'
     b'               + ("UNDEFINED" if share is None else f"{share:.2%}"))',
     b'    out.append(f"  share of the excess: {share:.2%}")',
     [f"{TP}::test_the_formatter_renders_an_undefined_share"]),

    # ================================================ BC 4: the device class
    ("device_class_coverage: a missing device_class is not detected", N,
     b'    missing = [t["name"] for t in targets if not t.get("device_class")]',
     b"    missing = []",
     [f"{TN}::test_a_target_with_no_device_class_is_UNCOVERED"]),

    ("device_class_coverage: an undeclared value is not detected", N,
     b"    undeclared = sorted({t[\"device_class\"] for t in targets\n"
     b'                         if t.get("device_class")\n'
     b'                         and t["device_class"] not in vocabulary})',
     b"    undeclared = []",
     [f"{TN}::test_a_device_class_outside_the_vocabulary_is_UNDECLARED"]),

    # THE FIXTURE LESSON, MADE A MUTATION. If the inputs were not injectable
    # this branch could not be exercised at all -- which is how BB12 item 2's
    # `tolerance_coverage` mutation survived.
    ("device_class_coverage: the two failure verdicts collapse into one", N,
     b'        "verdict": ("COVERED" if not missing and not undeclared\n'
     b'                    else "TARGET_WITHOUT_DEVICE_CLASS" if missing\n'
     b'                    else "UNDECLARED_DEVICE_CLASS"),',
     b'        "verdict": ("COVERED" if not missing and not undeclared\n'
     b'                    else "TARGET_WITHOUT_DEVICE_CLASS"),',
     [f"{TN}::test_a_device_class_outside_the_vocabulary_is_UNDECLARED"]),

    ("TARGETS: a target silently changes device class", N,
     b'    {"name": "alignment_predictor", "arm": "predictor",\n'
     b'     "path": "gtilde_compliant_b_ratio.h5",\n'
     b'     "hash_scope": "whole_file", "device_class": UNKNOWN,',
     b'    {"name": "alignment_predictor", "arm": "predictor",\n'
     b'     "path": "gtilde_compliant_b_ratio.h5",\n'
     b'     "hash_scope": "whole_file", "device_class": MNN,',
     [f"{TN}::test_the_MNN_targets_are_the_ones_that_read_a_cosine_threshold"]),

    ("DEVICE_CLASS_CONDITION: NOT_ESTABLISHED reads as device-independent", N,
     b'    f"\'{UNKNOWN}\' means no device-sensitivity measurement exists for that "\n'
     b'    "target; it does NOT mean device-independent.")',
     b'    f"\'{UNKNOWN}\' means the target is device-independent.")',
     [f"{TN}::test_NOT_ESTABLISHED_is_documented_as_ignorance_not_independence"]),

    ("capture: the manifest no longer carries the condition", N,
     b'                     "device_class_condition": DEVICE_CLASS_CONDITION,\n'
     b'                     "baseline_device_class": BASELINE_DEVICE_CLASS},',
     b'                     "baseline_device_class": BASELINE_DEVICE_CLASS},',
     [f"{TN}::test_the_manifest_carries_the_condition_so_a_reader_of_the_JSON_sees_it"]),
]


def main() -> int:
    fixtures = sorted({t for e in MUTATIONS for t in e[4]} | set(CONTROL[4]))
    from tests.mutation_matrix_av import run_one
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

    print(f"{'mutation':<66} {'result':<10} killed")
    bad = 0
    for label, path, old, new, expect in MUTATIONS:
        survived = apply_and_run(path, old, new, expect)
        if survived is None:
            print(f"{label:<66} {'SKIP':<10} anchor did not match exactly once")
            bad += 1
            continue
        status = "KILLED" if not survived else "SURVIVED"
        print(f"{label:<66} {status:<10} "
              f"{len(expect) - len(survived)}/{len(expect)}")
        for t in survived:
            print(f"{'':<66} {'':<10} SURVIVOR: {t}")
        bad += bool(survived)

    print()
    print("ALL MUTATIONS KILLED THEIR NAMED FIXTURES" if not bad
          else f"{bad} MUTATION(S) SURVIVED OR WERE SKIPPED")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
