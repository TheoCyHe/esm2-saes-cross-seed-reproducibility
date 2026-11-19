"""Mutation matrix for amendment BA's new code. NOT collected by pytest.

Covers `carrier_support` (BA9), `alignment_fine_panel` (BA11),
`concept_headline_checks` (BA10) and `src/cluster/wait_guard` (BA12).

Run manually:  python tests/mutation_matrix_ba.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

The runner is reused from `mutation_matrix_av` -- AU9b's control mutation,
AV22's `__pycache__` purge between invocations, the byte-compared restore, and
the decide-on-exit-code rule. Reused rather than re-implemented: a runner is a
check, and a second copy is a second thing that can silently stop detecting
(gotcha 13 extension 5).

THE ONE THIS FILE EXISTS FOR is the guard's own false-completion branch. AZ20
item 3's wait printed ALL_DONE while three jobs ran, so the mutation that
matters is not a malformed readout -- it is `corroborate` letting a live
SECONDARY through. That is the second entry under `wait_guard` below.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run, run_one  # noqa: E402

ROOT = os.getcwd()
S = os.path.join(ROOT, "src", "analysis", "carrier_support.py")
F = os.path.join(ROOT, "src", "analysis", "alignment_fine_panel.py")
H = os.path.join(ROOT, "src", "analysis", "concept_headline_checks.py")
W = os.path.join(ROOT, "src", "cluster", "wait_guard.py")
TS = "tests/test_carrier_support.py"
TF = "tests/test_alignment_fine_panel.py"
TH = "tests/test_concept_headline_checks.py"
TW = "tests/test_wait_guard.py"

# THE CONTROL: the guard always reports a corroborated completion. This cannot
# fail to break the suite -- `test_every_outcome_is_reachable` asserts four
# DISTINCT verdicts, and a constant collapses them to one. If the runner
# reports this undetected, none of its SURVIVED verdicts mean anything and none
# is printed.
#
# **THE FIRST DRAFT OF THIS CONTROL WAS ITSELF A NO-OP**, and the runner caught
# it: it replaced the final return's `why` STRING while leaving the verdict
# alone, so the three earlier branches still fired and all four outcomes were
# still reachable. The runner reported CONTROL NOT DETECTED and printed no
# verdicts -- which is exactly what AU9b's control exists to do, firing on the
# control itself. The anchor below returns before any branch runs.
CONTROL = (
    "CONTROL -- corroborate returns a constant before any branch runs", W,
    b"    p_states = parse_states(primary_raw)\n"
    b"    s_states = parse_states(secondary_raw)",
    b'    return {"verdict": "CORROBORATED_COMPLETE", "ok": True,\n'
    b'            "complete": True, "why": "control"}\n'
    b"    p_states = parse_states(primary_raw)\n"
    b"    s_states = parse_states(secondary_raw)",
    [f"{TW}::test_every_outcome_is_reachable_so_the_guard_can_do_more_than_one_thing"],
)

MUTATIONS = [
    # ------------------------------------------------------- wait_guard (BA12)
    # THE ACTUAL FAILURE MODE (AZ20 item 3): the primary says nothing is live,
    # the secondary shows RUNNING, and the guard proceeds anyway. This is the
    # mutation the whole module exists to be killed by.
    ("corroborate: a live SECONDARY no longer blocks completion", W,
     b"    if s_live:\n",
     b"    if False:\n",
     [f"{TW}::test_refuses_a_FALSE_COMPLETION_the_AZ20_case",
      f"{TW}::test_a_failed_primary_cannot_clear_itself_by_producing_nothing",
      f"{TW}::test_states_are_case_normalised",
      f"{TW}::test_every_outcome_is_reachable_so_the_guard_can_do_more_than_one_thing"]),

    # Silence read as a clean queue -- gotcha 28 inside the module written to
    # enforce it. An empty secondary would then CLEAR the wait.
    ("corroborate: an empty secondary readout counts as 'nothing running'", W,
     b"    if not secondary_produced:\n",
     b"    if False:\n",
     [f"{TW}::test_a_silent_secondary_REFUSES_rather_than_clearing_the_wait",
      f"{TW}::test_a_MISSING_secondary_is_also_a_refusal_not_a_completion",
      f"{TW}::test_blank_lines_do_not_count_as_output",
      f"{TW}::test_every_outcome_is_reachable_so_the_guard_can_do_more_than_one_thing"]),

    # THE SIGNAL PATH (gotcha 13 extension 1). A guard whose refusal exits 0 is
    # worse than no guard: an `&&` chain reads it as success.
    ("main: a refusal exits 0 instead of 1", W,
     b'    raise SystemExit(0 if v["complete"] else (2 if v["ok"] else 1))',
     b'    raise SystemExit(0)',
     [f"{TW}::test_main_exit_codes_are_gateable"]),

    # A state token read whole rather than as its first field: `CANCELLED by
    # 3001` would stop matching, and lowercase states would stop matching too.
    ("parse_states: no case normalisation", W,
     b"            out.append(tok[0].upper())",
     b"            out.append(tok[0])",
     [f"{TW}::test_states_are_case_normalised"]),

    ("LIVE_STATES: PENDING is not a live state", W,
     b'    "PENDING", "RUNNING", "SUSPENDED", "CONFIGURING", "COMPLETING",',
     b'    "RUNNING", "SUSPENDED", "CONFIGURING", "COMPLETING",',
     [f"{TW}::test_every_live_state_blocks_completion"]),

    # -------------------------------------------------- carrier_support (BA9)
    # THE WHOLE DESIGN of BA9: the peer pool is taken WITHIN the firing bin.
    # Pooling across bins makes the statistic a composition effect, and the
    # fixture's two bins have disjoint d_fs so the two answers differ by 30
    # percentile points.
    ("support_distribution: the peer pool ignores the firing bin", S,
     b"        for b in range(n_fbins):\n"
     b"            in_bin = fbin[s] == b\n"
     b"            peers = in_bin & ~is_carrier[s]\n"
     b"            pool = d_fs[s][peers]\n"
     b"            sel = idx_cell[fbin[s][idx_cell] == b] if idx_cell.size else idx_cell",
     b"        for b in range(n_fbins):\n"
     b"            in_bin = fbin[s] == b\n"
     b"            peers = ~is_carrier[s]\n"
     b"            pool = d_fs[s][peers]\n"
     b"            sel = idx_cell[fbin[s][idx_cell] == b] if idx_cell.size else idx_cell",
     [f"{TS}::test_the_matched_arm_sees_what_the_pooled_one_cannot"]),

    # Carriers left inside their own comparison population: the baseline moves
    # toward the carriers by construction.
    ("carrier_feature_indicator: nothing is marked as a carrier", S,
     b"            out[s, idx] = True",
     b"            pass",
     [f"{TS}::test_carrier_feature_indicator_marks_only_DETECTED_concepts_carriers",
      f"{TS}::test_peers_EXCLUDE_carriers_and_that_changes_the_answer",
      f"{TS}::test_a_bin_with_no_peers_is_counted_unusable_not_dropped"]),

    # A carrier of an UNDETECTED concept counted as a carrier. The anchor
    # carries the two lines AROUND it: `idx = carrier[s][detected[s]]` occurs
    # twice in this file (here and in the carrier-side Spearman loop), and a
    # non-unique anchor is an ANCHOR MISS, not a mutation.
    ("carrier_feature_indicator: the detection mask is ignored", S,
     b"        idx = carrier[s][detected[s]]\n"
     b"        if idx.size:\n"
     b"            out[s, idx] = True",
     b"        idx = carrier[s]\n"
     b"        if idx.size:\n"
     b"            out[s, idx] = True",
     [f"{TS}::test_carrier_feature_indicator_marks_only_DETECTED_concepts_carriers"]),

    # Unusable cells vanishing from the count rather than being reported.
    ("support_distribution: cells with no peer pool are not counted", S,
     b"                unusable += int(sel.size)",
     b"                pass",
     [f"{TS}::test_a_bin_with_no_peers_is_counted_unusable_not_dropped"]),

    # The half-tie convention, shared with AZ12's statistic.
    ("percentile_rank_within: ties ranked low instead of at their midpoint", S,
     b"    return 100.0 * (lo + hi) / 2.0 / order.size",
     b"    return 100.0 * lo / order.size",
     [f"{TS}::test_percentile_rank_within_uses_the_half_tie_convention",
      f"{TS}::test_percentile_positions_still_uses_the_shared_helper"]),

    # An empty pool returning a 50 -- "measured, and exactly average".
    # `support_distribution` guards `pool.size == 0` BEFORE calling this, so
    # the no-peer-bin fixture never reaches the branch and was wrongly listed
    # here: a fixture that cannot exercise the mutated line is not a fixture
    # that can kill it.
    ("percentile_rank_within: an empty pool returns a value anyway", S,
     b"    if order.size == 0:\n        return np.zeros(0, dtype=np.float64)",
     b"    if order.size == 0:\n        return np.full(x.shape, 50.0)",
     [f"{TS}::test_percentile_rank_within_returns_EMPTY_for_an_empty_pool"]),

    # BA7.1's band edges, declared before the run.
    ("support_verdict: the ENRICHED band starts at 50 instead of 55", S,
     b"    if m >= 55.0:",
     b"    if m >= 50.0:",
     [f"{TS}::test_support_verdict_band_edges_are_where_BA7_1_declared_them"]),

    ("support_verdict: the DEPLETED band is unreachable", S,
     b"    elif m >= 45.0:",
     b"    elif m >= 0.0:",
     [f"{TS}::test_support_verdict_is_total_and_every_band_is_reachable",
      f"{TS}::test_support_verdict_band_edges_are_where_BA7_1_declared_them"]),

    # THE READING ITSELF (BA6): an uninformative null must not read as support.
    ("support_verdict: the middle band's reading claims AZ11 is informative", S,
     b'        reading = ("carriers were NOT enriched in homology support relative "\n'
     b'                   "to their firing-matched peers, so stratifying on support "\n'
     b'                   "could not have changed anything. AZ11 says NOTHING either "\n'
     b'                   "way and the 7.22 pp excess remains UNATTRIBUTED. An "\n'
     b'                   "uninformative null is not support.")',
     b'        reading = ("carriers were not enriched, and the excess is about "\n'
     b'                   "alignment.")',
     [f"{TS}::test_the_NOT_ENRICHED_reading_says_AZ11_establishes_NOTHING"]),

    # --------------------------------------------- alignment_fine_panel (BA11)
    # THE DIRECTION ITSELF. A sign flip turns AZ13.1's falling stratum into a
    # rise and the restatement would confirm the defect it exists to catch.
    ("per_seed_span: first minus last", F,
     b'    return float(use[-1]["mean_r_f"] - use[0]["mean_r_f"])',
     b'    return float(use[0]["mean_r_f"] - use[-1]["mean_r_f"])',
     [f"{TF}::test_per_seed_span_is_last_usable_minus_first",
      f"{TF}::test_the_replacement_catches_the_stratum_1_case",
      f"{TF}::test_a_genuine_rise_is_scored_as_one_by_both_criteria",
      f"{TF}::test_per_seed_span_ignores_EMPTY_bins_at_the_ends"]),

    # The interval dropped from the direction verdict -- the replacement would
    # then be a point estimate wearing a verdict's clothes, which is the shape
    # AC14 already had to correct once.
    ("span_verdict: the direction ignores its interval", F,
     b"    elif lo > 0:",
     b"    elif mean > 0:",
     [f"{TF}::test_a_span_whose_interval_straddles_zero_is_FLAT_not_a_rise",
      f"{TF}::test_span_verdict_is_total_and_every_outcome_is_reachable"]),

    # The FLAT branch removed: a straddling interval would be reported as a
    # fall, which is the outcome AZ13.1 could not name.
    ("span_verdict: INDISTINGUISHABLE_FROM_FLAT is unreachable", F,
     b'        verdict = "INDISTINGUISHABLE_FROM_FLAT"',
     b'        verdict = "FALLS"',
     [f"{TF}::test_a_span_whose_interval_straddles_zero_is_FLAT_not_a_rise",
      f"{TF}::test_span_verdict_is_total_and_every_outcome_is_reachable"]),

    # The paired construction. Differencing across seeds instead of within them
    # reintroduces the between-seed level variation the pairing removes.
    ("per_seed_span: EMPTY bins are not skipped", F,
     b'    use = [r for r in rows if r.get("status") == "COMPUTED"]\n'
     b"    if len(use) < 2:\n"
     b"        return None",
     b"    use = rows\n"
     b"    if len(use) < 2:\n"
     b"        return None",
     [f"{TF}::test_per_seed_span_ignores_EMPTY_bins_at_the_ends",
      f"{TF}::test_per_seed_span_is_None_with_fewer_than_two_usable_bins"]),

    # The flag that names AZ13.1's defect, hardcoded off.
    ("restated_verdict: the misleading-old-verdict flag never fires", F,
     b'    misleading = bool(old == "MONOTONE_NON_DECREASING" and new != "RISES")',
     b"    misleading = False",
     [f"{TF}::test_the_replacement_catches_the_stratum_1_case"]),

    # ABSENT must never read as AGREEMENT (gotcha 15 / I3), here on the
    # acceptance criterion that makes the old-vs-new comparison interpretable.
    ("reproduces_reference: problems no longer change the verdict", F,
     b'        "verdict": ("REPRODUCES_REFERENCE_EXACTLY" if not problems\n'
     b'                    else "DIFFERS"),',
     b'        "verdict": "REPRODUCES_REFERENCE_EXACTLY",',
     [f"{TF}::test_reproduces_reference_reports_an_ABSENT_arm_as_absent",
      f"{TF}::test_reproduces_reference_refuses_a_moved_bin_mean"]),

    ("reproduces_reference: an absent arm is treated as matching", F,
     b'            problems.append(f"arm {name}: ABSENT from the reference")',
     b"            pass",
     [f"{TF}::test_reproduces_reference_reports_an_ABSENT_arm_as_absent"]),

    # ------------------------------------------ concept_headline_checks (BA10)
    # Gotcha 25b: a ranking statistic against a constant ranks the tie-break.
    ("null_max_vs_prevalence: a constant prevalence yields a correlation", H,
     b"    if np.unique(p).size < 2:",
     b"    if False:",
     [f"{TH}::test_null_max_vs_prevalence_reports_a_CONSTANT_prevalence_as_uninformative"]),

    # THE UNIT CORRECTION (BA7.2). Dropping the reconstruction reports the gain
    # twice under two names, and the two units genuinely disagree in sign.
    ("null_max_vs_prevalence: the AP reconstruction is dropped", H,
     b"    ap = gain * (1.0 - p)[None, :] + p[None, :]",
     b"    ap = gain",
     [f"{TH}::test_gain_and_AP_can_disagree_in_SIGN_which_is_why_both_are_reported",
      f"{TH}::test_null_max_vs_prevalence_recovers_the_AP_from_the_stored_GAIN"]),

    # The constant-maximum guard. Without it `spearman` RAISES and the whole
    # 703-concept job aborts on one flat cell.
    ("null_max_vs_prevalence: a constant per-seed maximum is not guarded", H,
     b"        if np.unique(gain[s]).size < 2:\n"
     b"            n_const_gain += 1\n"
     b"        else:\n"
     b"            rho_gain.append(spearman(p, gain[s]))",
     b"        if False:\n"
     b"            n_const_gain += 1\n"
     b"        else:\n"
     b"            rho_gain.append(spearman(p, gain[s]))",
     [f"{TH}::test_a_seed_with_a_CONSTANT_maximum_is_counted_not_crashed_on",
      f"{TH}::test_every_seed_constant_is_reported_rather_than_summarised"]),

    # The direction verdict's own band.
    ("null_scaling_population: any positive share is a direction", H,
     b'        if s["share_positive"] >= 0.60:',
     b'        if s["share_positive"] >= 0.0:',
     [f"{TH}::test_null_scaling_population_is_total_and_every_verdict_is_reachable"]),

    # BA7.3's cut. `<=` moves every count at the boundary and 1.2x IS a
    # boundary that real margins can sit on.
    ("margin_distribution: the cuts become inclusive", H,
     b'        "shares_below": {f"{c:.2f}": float((m < c).mean()) for c in cuts},\n'
     b'        "counts_below": {f"{c:.2f}": int((m < c).sum()) for c in cuts},',
     b'        "shares_below": {f"{c:.2f}": float((m <= c).mean()) for c in cuts},\n'
     b'        "counts_below": {f"{c:.2f}": int((m <= c).sum()) for c in cuts},',
     [f"{TH}::test_margin_distribution_cuts_are_strict_so_the_boundary_is_reachable"]),

    # AY8.3's definitional separation: a D < 1 concept leaking in would be
    # counted as a fragile detection no seed ever made.
    ("margin_distribution: not restricted to the always-detected concepts", H,
     b"    use = det & finite",
     b"    use = finite",
     [f"{TH}::test_margin_distribution_is_restricted_to_the_ALWAYS_DETECTED_concepts"]),

    # A non-finite margin swept into the distribution rather than counted.
    ("margin_distribution: non-finite margins are not excluded", H,
     b"    finite = np.isfinite(worst_margin)\n    use = det & finite",
     b"    finite = np.ones(worst_margin.shape, dtype=bool)\n    use = det & finite",
     [f"{TH}::test_margin_distribution_counts_non_finite_margins_separately"]),
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
        print(f"CONTROL ANCHOR DID NOT MATCH EXACTLY ONCE in {path} -- ABORT")
        return 2
    if survived:
        print(f"CONTROL NOT DETECTED ({survived}) -- the runner cannot see a "
              "change that must break the suite. No verdicts are printed.")
        return 2
    print(f"control detected: {name}\n")

    bad = 0
    for name, path, old, new, fx in MUTATIONS:
        survived = apply_and_run(path, old, new, fx)
        if survived is None:
            print(f"ANCHOR MISS  {name}")
            bad += 1
            continue
        if survived:
            print(f"SURVIVED     {name}  -> {survived}")
            bad += 1
        else:
            print(f"killed       {name}")
    print(f"\n{len(MUTATIONS) - bad}/{len(MUTATIONS)} killed, control detected")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
