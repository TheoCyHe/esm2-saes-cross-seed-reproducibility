"""Mutation matrix for amendment AV's new code. NOT collected by pytest.

Covers `merge_ap_shards` (AV10d) and `av_counts` (AV11's two counts).

Run manually:  python tests/mutation_matrix_av.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

WHAT THIS ADDS OVER `mutation_matrix_preflight.py`, and it is AU9b's reusable fix:

  * **A CONTROL MUTATION RUNS FIRST AND ABORTS THE RUN IF IT IS NOT DETECTED.**
    A runner reports SURVIVED by finding no evidence of a kill, so **every way
    its detection can silently stop working presents as a clean SURVIVED** --
    the standing rule that absence of output is not evidence, failing inside the
    tool built to enforce it. The control is a change that MUST break the suite;
    if the runner cannot see that, none of its SURVIVED verdicts mean anything
    and no other verdict is printed.
  * **THE RESTORE IS BYTE-COMPARED.** The first version of the preflight runner
    left a mutant on disk after a kill, caught only because a full-suite run
    happened to follow. Here every restore is verified against the bytes read
    before the mutation, and a failed restore is fatal.

Decisions are made on pytest's EXIT CODE, never on a name match in its output,
and `--color=no` is passed -- the ANSI-escape defeat is the founding instance.
"""
import os
import shutil
import subprocess
import sys

ROOT = os.getcwd()
M = os.path.join(ROOT, "src", "analysis", "merge_ap_shards.py")
C = os.path.join(ROOT, "src", "analysis", "av_counts.py")
D = os.path.join(ROOT, "src", "analysis", "design3_join.py")
F = os.path.join(ROOT, "src", "analysis", "feature_ap.py")
T = "tests/test_merge_ap_shards.py"
TC = "tests/test_av_counts.py"
TD = "tests/test_design3_join.py"
TF = "tests/test_feature_ap.py"

# The CONTROL: replace the positional row-by-row comparison with a constant
# verdict. This cannot fail to break the suite -- several fixtures assert
# POSITIONS_WRONG on a genuinely permuted axis. If the runner reports this as
# undetected, the runner is broken.
CONTROL = (
    "CONTROL -- positional verdict replaced by a constant", M,
    b'return {"verdict": "POSITIONS_OK" if not wrong else "POSITIONS_WRONG",',
    b'return {"verdict": "POSITIONS_OK",',
    [f"{T}::test_positional_check_sees_a_permutation_a_total_cannot"],
)

MUTATIONS = [
    # The seed-set check. Removing it lets a 19-shard merge produce a
    # correct-looking 19-row artifact -- gotcha 29's silent-partial-ensemble.
    ("seed-SET equality (missing shard becomes a short axis)", M,
     b'    if set(src) != set(ref):',
     b'    if False:',
     [f"{T}::test_missing_seed_raises_rather_than_silently_shortening",
      f"{T}::test_merge_refuses_a_missing_shard"]),

    # The reordering itself. Returning identity order is precisely the defect
    # AR9 found on disk: numeric shards written straight into a lexicographic
    # axis, 18 of 20 rows wrong, every identity key still agreeing.
    ("reordering disabled (shards written in arrival order)", M,
     b'    return [src.index(r) for r in ref]',
     b'    return list(range(len(src)))',
     [f"{T}::test_order_maps_numeric_shards_onto_a_lexicographic_reference",
      f"{T}::test_order_is_spelling_insensitive",
      f"{T}::test_merge_orders_rows_by_the_reference_and_the_values_follow"]),

    # Duplicate labels. Without this, two shards for one seed silently displace
    # a third and `set(src) == set(ref)` still holds if a label repeats.
    ("duplicate-label check", M,
     b'    if len(set(src)) != len(src):',
     b'    if False:',
     [f"{T}::test_duplicate_seed_labels_raise"]),

    # ABSENT must never read as AGREEMENT (I3). Skipping the absence branch
    # makes a key missing from every-but-one artifact pass silently.
    ("absent-is-not-matching branch", M,
     b'        if len(present) != len(identities):',
     b'        if False:',
     [f"{T}::test_absent_key_is_reported_absent_not_matching"]),

    # The closed half of the comparability check: an unexpected difference in a
    # key nobody enumerated must be reported, not pass for want of a clause.
    ("unexpected-difference scan (the CLOSED half)", M,
     b'        if _hashable(experimental[k]) != _hashable(reference[k]):',
     b'        if False:',
     [f"{T}::test_comparability_fails_on_a_key_the_enumerated_list_has_never_SEEN"]),

    # The seed axis is the ONLY thing that differs in this fixture, and no
    # identity key can see it. Dropping it from the verdict is the whole
    # gotcha-29 blind spot restored.
    ("seed axis dropped from the comparability verdict", M,
     b'                    and seeds["verdict"] == "POSITIONS_OK"',
     b'                    and True',
     [f"{T}::test_comparability_fails_on_a_seed_axis_permutation_alone"]),

    # The concept axis is a function of the store, never the checkpoint.
    ("concept-axis identity across shards", M,
     b'    mismatched = [i for i, c in enumerate(concept_lists) if list(c) != first]',
     b'    mismatched = []',
     [f"{T}::test_concept_axis_difference_is_caught"]),

    # The shard identity gate.
    ("shard identity agreement gate", M,
     b'    if not shard_agreement["agree"]:',
     b'    if False:',
     [f"{T}::test_merge_refuses_shards_that_disagree_on_identity"]),

    # One seed per shard: without it, a multi-seed shard contributes only row 0
    # and the merged axis is short by however many it hid.
    ("one-seed-per-shard assertion", M,
     b'            if len(lab) != 1:',
     b'            if False:',
     [f"{T}::test_merge_refuses_a_multi_seed_shard"]),

    # The SIGNAL PATH (gotcha 13 extension 1): a validator whose failure is
    # unreadable by the thing gating on it is worse than no validator.
    ("main() exit code on NOT_COMPARABLE", M,
     b'        print(f"FATAL: stratum comparability verdict is {verdict}")\n        return 1',
     b'        print(f"FATAL: stratum comparability verdict is {verdict}")\n        return 0',
     [f"{T}::test_main_returns_nonzero_when_not_comparable"]),

    # gotcha 25(a): exempting provenance by full name. Un-exempting it makes a
    # per-run key an identity difference, so the check could ONLY fail.
    ("scored-concepts-subset-of-library guard", M,
     b'    if not report["scored_concepts_are_a_subset_of_the_library"]:',
     b'    if False:',
     [f"{T}::test_merge_refuses_shards_scoring_concepts_outside_the_library"]),

    # amendment AW: n_concepts/argmax_pool_n are library-scoped AT THE WRITER
    # now, so the merge only VERIFIES agreement -- it no longer corrects.
    # Disabling the pre-fix-format guard would silently merge a shard whose
    # canonical keys are still stratum-scoped (the original bug, one hop
    # downstream of where it used to be caught).
    ("pre-AW-format guard (no n_concepts_scored attr)", M,
     b'    if shard_identity.get("n_concepts_scored") is None:',
     b'    if False:',
     [f"{T}::test_merge_refuses_shards_predating_the_writer_fix"]),

    # And disabling the agreement check itself would silently merge a shard
    # whose writer REGRESSED -- claims post-AW format (carries
    # `n_concepts_scored`) but its `n_concepts`/`argmax_pool_n` disagree with
    # the label store's own library. gotcha 16 one hop downstream again: the
    # check exists so a wrong value cannot ride through under a canonical key.
    ("library-vs-shard agreement check (writer-regression guard)", M,
     b'    if mismatches:',
     b'    if False:',
     [f"{T}::test_merge_refuses_when_writer_regresses_despite_post_fix_format"]),

    # prevalence scales every G downstream and the merged artifact copies it
    # from ONE shard, so a value disagreement must be fatal. Only a positional
    # VALUE check sees it -- names, shapes and counts all still agree.
    ("per-concept vector value equality across shards", M,
     b'        elif not np.array_equal(v, first):',
     b'        elif False:',
     [f"{T}::test_prevalence_disagreement_between_shards_is_fatal"]),

    ("per-concept vector ABSENT_FROM_SOME is not agreement", M,
     b'        if len(vectors[k]) != len(shard_paths):',
     b'        if False:',
     [f"{T}::test_a_vector_present_on_only_SOME_shards_is_not_agreement"]),

    ("provenance exemption (a check that could only fail)", M,
     b'    unexpected = [u for u in unexpected if u["key"] not in provenance]',
     b'    unexpected = unexpected',
     [f"{T}::test_provenance_keys_do_not_count_as_unexpected_differences"]),

    # ---- av_counts (AV11) ------------------------------------------------
    # C4's invariant: G uses the concept's OWN prevalence. A reused global p is
    # the defect that made every prevalence wrong once already (the
    # population / row_subset collision).
    ("per-concept prevalence in G (reused global p)", C,
     b'    return (ap - prevalence[None, :]) / (1.0 - prevalence[None, :])',
     b'    return (ap - prevalence[0]) / (1.0 - prevalence[0])',
     [f"{TC}::test_gain_uses_the_PER_CONCEPT_prevalence_not_a_global_one",
      f"{TC}::test_normalised_gain_matches_the_pre_registered_formula"]),

    ("concept-axis length guard on the prevalence vector", C,
     b'    if ap.shape[-1] != prevalence.shape[0]:',
     b'    if False:',
     [f"{TC}::test_mismatched_prevalence_length_raises_rather_than_broadcasting"]),

    # The experimental tiers are a UNION, not a sum: a row appearing in both
    # spellings must be counted once.
    ("experimental tiers unioned rather than summed", C,
     b'                arr = np.unique(np.concatenate([g[t][:] for t in present]))\n'
     b'                counts[concept] = int(arr.size)',
     b'                counts[concept] = int(sum(g[t].shape[0] for t in present))',
     [f"{TC}::test_overlapping_experimental_rows_are_deduplicated"]),

    # The argmax pool is the set g_f maxes over. Ignoring it reports the
    # library figure under the pool's name -- gotcha 16's one-key-one-meaning.
    ("argmax-pool mask in the gain-floor share", C,
     b'        if pool_only:',
     b'        if False:',
     [f"{TC}::test_pool_mask_is_honoured_and_changes_the_answer"]),

    # The per-seed breakdown. A pooled share alone cannot distinguish a stable
    # effect from one seed carrying it.
    ("per-seed shares (pooled figure only)", C,
     b'                "share_clearing": float(clear.mean()),',
     b'                "share_clearing": float(n_clear_total) * 0 + 0.5,',
     [f"{TC}::test_per_seed_shares_are_reported_separately"]),

    # Both verdicts must be DERIVED from the data (gotcha 20), including at
    # the boundary -- "under 352" and "352 and above" is the declared TOTAL.
    ("AV-P1 verdict derived (boundary included)", C,
     b'            "outcome": "HIT" if n_with_experimental < threshold else "MISS"}',
     b'            "outcome": "HIT" if n_with_experimental <= threshold else "MISS"}',
     [f"{TC}::test_p1_verdict_is_derived_on_both_sides_of_its_threshold"]),

    ("AV-P2 verdict derived (boundary included)", C,
     b'            "outcome": "HIT" if share < threshold else "MISS"}',
     b'            "outcome": "HIT" if share <= threshold else "MISS"}',
     [f"{TC}::test_p2_verdict_is_derived_on_both_sides_of_its_threshold"]),

    # Coverage must count only the NAMED set, or "703 pool concepts" silently
    # becomes "712 library concepts".
    # Two artifacts that both OPEN is not two artifacts describing the same
    # experiment (gotcha 15). Without this, "703 pool concepts" could silently
    # mean two different sets in one report.
    ("library agreement between the store and the AP artifact", C,
     b'    if only_store or only_ap:',
     b'    if False:',
     [f"{TC}::test_main_refuses_a_store_and_an_AP_naming_different_libraries"]),

    # ---- design3_join, amendment AV section 5c ---------------------------
    # The BLOCKING is the whole measurement. Without it the stratified arms are
    # the unstratified one under three names, and every verdict is AU14's.
    ("d_fs blocking in the stratified permutation", D,
     b'                    block = np.digitize(dd, edges)',
     b'                    block = np.zeros(dd.size, dtype=np.int64)',
     [f"{TD}::test_stratified_permutation_REMOVES_a_spread_driven_entirely_by_d_fs"]),

    # Per-arm generator state. Sharing one sequential rng puts Monte-Carlo
    # noise into every arm-to-arm comparison -- the defect this fixture caught.
    ("per-arm generator state (arms share one stream)", D,
     b'                    np.random.default_rng([seed, s, b]))',
     b'                    np.random.default_rng([seed, s, b, len(arms)]))',
     [f"{TD}::test_a_single_block_reproduces_the_unstratified_arm_exactly"]),

    # AV-P3 declares "HIT requires BOTH". An OR makes the second half
    # decorative -- gotcha 24, a rule not total over its own statement.
    ("AV-P3 requires BOTH halves", D,
     b'            "AV_P3_both_halves": bool(inside and few),',
     b'            "AV_P3_both_halves": bool(inside or few),',
     [f"{TD}::test_AV_P3_verdict_needs_BOTH_halves_on_a_DISAGREEING_pair"]),

    ("AV-P3's half_that_fails is derived, not asserted", D,
     b"""                                "spread" if not inside else "cell_count")}""",
     b"""                                "spread")}""",
     [f"{TD}::test_AV_P3_verdict_needs_BOTH_halves_on_a_DISAGREEING_pair"]),

    # The empirical between-feature dispersion IS the replacement interval
    # (AV12b). Reverting it to a pooled-proportion construction is the very
    # error AU13 measured at a 2.069x understatement.
    ("empirical between-feature SE on the decile means", D,
     b'                    float(sd / np.sqrt(n)) if n > 1 else None',
     b'                    None if n > 1 else None',
     [f"{TD}::test_empirical_intervals_replace_the_wilson_construction"]),

    ("coverage restricted to the named concept set", C,
     b'    vals = np.array([exp_counts.get(c, 0) for c in concepts], dtype=np.int64)',
     b'    vals = np.array(list(exp_counts.values()), dtype=np.int64)',
     [f"{TC}::test_coverage_counts_only_concepts_in_the_named_set"]),

    # ---- feature_ap.py's library_scoped_concept_counts (amendment AW) -----
    # n_concepts/argmax_pool_n must be LIBRARY-scoped. Swapping in the scored
    # list reproduces the exact pre-AW bug this helper was extracted to fix.
    ("n_concepts / argmax_pool_n read from the LIBRARY, not the scored set", F,
     b'        "n_concepts": len(library_concepts),\n'
     b'        "argmax_pool_n": int(sum(argmax_pool_mask(library_concepts))),',
     b'        "n_concepts": len(scored_concepts),\n'
     b'        "argmax_pool_n": int(sum(argmax_pool_mask(scored_concepts))),',
     [f"{TF}::test_library_scoped_counts_on_a_stratified_run_are_four_distinct_numbers"]),

    ("argmax_pool_scored_n excludes pooled parents from the SCORED list too", F,
     b'        "argmax_pool_scored_n": int(sum(argmax_pool_mask(scored_concepts))),',
     b'        "argmax_pool_scored_n": len(scored_concepts),',
     [f"{TF}::test_library_scoped_counts_scored_set_smaller_still_excludes_its_own_parent"]),
]


def purge_pycache() -> None:
    """Delete every `__pycache__` under `src/` and `tests/`.

    **WHY THIS EXISTS, and it is a defect this runner shipped with.** CPython's
    default pyc invalidation is `(source mtime, source SIZE)`. Two ADJACENT
    mutations of the SAME file that produce EQUAL-SIZED mutants -- e.g. the
    AV-P1 and AV-P2 verdict mutations, both turning `<` into `<=`, both exactly
    13,101 bytes -- are indistinguishable to that check if their writes land in
    one filesystem mtime tick. pytest then imports the PREVIOUS mutant while
    the runner believes it is testing the current one.

    **Measured: 1 run in 4 reported `AV-P2 verdict derived` as SURVIVED**, on a
    mutation the same matrix killed on the other three. **And the dangerous
    direction is symmetric**: if the stale mutant happens to break the current
    mutation's fixture, the runner prints KILLED for a mutation that never ran
    -- coverage asserted on evidence that does not exist. That is exactly
    AU9b's failure, in a second mechanism: the instrument's detection silently
    stops working, and the output looks like an ordinary verdict.

    Purging is O(ms) against a ~1 s pytest start-up, so there is no reason to
    be cleverer than deleting the cache.
    """
    for root in ("src", "tests"):
        for dirpath, dirnames, _ in os.walk(os.path.join(ROOT, root)):
            for d in list(dirnames):
                if d == "__pycache__":
                    shutil.rmtree(os.path.join(dirpath, d), ignore_errors=True)
                    dirnames.remove(d)


def run_one(node: str) -> bool:
    """True if the fixture PASSES. Judged on the EXIT CODE, never on a name
    match in stdout -- that is the founding instance of this rule (AU9b)."""
    # Purge FIRST: a stale pyc keyed on (mtime, size) can serve a
    # different, equal-sized mutant of the same file (see purge_pycache).
    purge_pycache()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", node, "-q", "--tb=no", "--color=no",
         "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=ROOT)
    return proc.returncode == 0


def apply_and_run(path, old, new, fixtures):
    """Mutate, run, restore, and BYTE-COMPARE the restore. Returns the list of
    fixtures that SURVIVED, or None if the anchor did not match exactly once."""
    original = open(path, "rb").read()
    old_v, new_v = old, new
    if original.count(old_v) != 1 and b"\n" in old:
        old_v, new_v = old.replace(b"\n", b"\r\n"), new.replace(b"\n", b"\r\n")
    if original.count(old_v) != 1:
        return None
    open(path, "wb").write(original.replace(old_v, new_v, 1))
    try:
        survived = [t for t in fixtures if run_one(t)]
    finally:
        open(path, "wb").write(original)
        if open(path, "rb").read() != original:
            print(f"FATAL: restore of {path} did not reproduce the original "
                  "bytes -- a mutant may be left on disk.")
            sys.exit(2)
        # PURGE AFTER THE RESTORE, NOT ONLY BEFORE EACH RUN (amendment BP).
        # The source is byte-identical again, but a `.pyc` compiled from the
        # MUTANT can outlive it: CPython validates a cache on (mtime, size),
        # and a same-length mutation restored inside one mtime tick is
        # invisible to both. The stale cache then shadows correct source for
        # every LATER process -- which is how a clean tree produced one
        # failing test in a full-suite run after this matrix passed 8/8.
        # Founding instance: `== n_concepts` -> `<= n_concepts`, same length.
        purge_pycache()
    return survived


def main() -> int:
    fixtures = sorted({t for e in MUTATIONS for t in e[4]} | set(CONTROL[4]))
    failing = [t for t in fixtures if not run_one(t)]
    if failing:
        print("BASELINE BROKEN -- these fail before any mutation:")
        for t in failing:
            print(f"   {t}")
        return 1
    print(f"baseline: {len(fixtures)}/{len(fixtures)} fixtures pass")

    # ---- THE CONTROL, FIRST. Nothing else runs if it is not detected. ----
    label, path, old, new, expect = CONTROL
    survived = apply_and_run(path, old, new, expect)
    if survived is None:
        print(f"CONTROL ABORT: anchor for {label!r} did not match exactly once. "
              "The runner cannot demonstrate that it detects anything.")
        return 2
    if survived:
        print(f"CONTROL ABORT: {label} was NOT detected "
              f"(survivors: {survived}). Every SURVIVED verdict this runner "
              "could print would be meaningless, so none is printed.")
        return 2
    print(f"control: DETECTED ({label})\n")

    print(f"{'mutation':<58} {'result':<10} killed")
    bad = 0
    for label, path, old, new, expect in MUTATIONS:
        survived = apply_and_run(path, old, new, expect)
        if survived is None:
            print(f"{label:<58} {'SKIP':<10} anchor did not match exactly once")
            bad += 1
            continue
        status = "KILLED" if not survived else "SURVIVED"
        print(f"{label:<58} {status:<10} {len(expect) - len(survived)}/{len(expect)}")
        for t in survived:
            print(f"{'':<58} {'':<10} SURVIVOR: {t}")
        bad += bool(survived)

    print()
    print("ALL MUTATIONS KILLED THEIR NAMED FIXTURES" if not bad
          else f"{bad} MUTATION(S) NOT FULLY KILLED -- real coverage gap")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
