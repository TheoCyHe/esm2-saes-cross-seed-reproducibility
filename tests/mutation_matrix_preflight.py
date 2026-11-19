"""Mutation matrix for the historical-defect fixtures (NOT collected by pytest).

Run manually:  python tests/mutation_matrix_preflight.py
Exit 0 iff every mutation kills the fixtures that name it.

This is committed rather than run ad hoc because the CLAIM it verifies -- that
each fixture covers the branch it names -- is otherwise unfalsifiable after the
fact. Last run 2026-07-28: 15/15 mutations killed, baseline 21/21
(after the scored_row_set / library_sha / inheritance-allowlist additions).


Each mutation disables exactly ONE gate branch; the fixtures naming that
branch must then FAIL. A fixture surviving its own mutation is passing because
some OTHER branch caught the mismatch, and provides no coverage of the branch
it claims to cover.

Robustness notes, both learned the hard way in this session:
  - files are read/written as BYTES, so a mutation round-trip cannot silently
    rewrite CRLF -> LF and leave the tree looking modified;
  - each fixture is run individually and judged by pytest's EXIT CODE, never
    by scraping stdout -- pytest colourises output, and an ANSI escape between
    "FAILED" and the test id defeats the obvious regex.
"""
import os
import subprocess
import sys

ROOT = os.getcwd()
PF = os.path.join(ROOT, "src", "analysis", "preflight.py")
AI = os.path.join(ROOT, "src", "analysis", "artifact_identity.py")
DNR = os.path.join(ROOT, "src", "analysis", "derive_null_reductions.py")
TESTFILE = "tests/test_preflight_historical.py"

MUTATIONS = [
    ("equality comparison (truncation_policy / floor params)", PF,
     b'if k in lab_id and lab_id[k] != oid[k]:',
     b'if False:',
     ["test_defect_4_stale_but_present_store_is_refused",
      "test_defect_7_population_mismatch_is_refused",
      "test_defect_7b_floor_parameter_mismatch_is_refused"]),

    ("derived comparison (argmax_pool_n / library_sha / n_rows)", PF,
     b'elif str(oid[k]) != str(want):',
     b'elif False:',
     ["test_defect_1_pool_mismatch_is_refused",
      "tests/test_preflight.py::test_library_sha_mismatch_is_refused",
      "tests/test_preflight.py::"
      "test_n_rows_mismatch_against_the_store_is_refused"]),

    ("must_be_absent (superseded markers)", PF,
     b'if present and oid.get(k):',
     b'if False:',
     ["test_defect_5b_reduction_stale_for_axis_is_refused"]),

    # NOTE the fixture list here. Defect 5's mis-attributed replacement is an
    # ATTR-vs-ATTR mismatch, so the generic equality branch catches it and it
    # is listed under that mutation, not this one. The CONTENT branch's
    # dedicated coverage lives in test_preflight.py, in a fixture that deletes
    # the store's own `n_concepts` so the equality path cannot fire -- which is
    # the only construction that isolates it. Listing defect 5 here would have
    # claimed coverage this mutation does not provide.
    ("concept-count vs store CONTENT", PF,
     b'if n_other is not None and int(n_other) != len(lab_concepts):',
     b'if False:',
     []),

    ("provenance SPEC check", PF,
     b'        if present:\n            problems.append(',
     b'        if False:\n            problems.append(',
     ["test_defect_3_contaminated_axis_is_refused",
      "test_defect_3_is_NOT_catchable_by_peer_agreement"]),

    ("git_dirty check", PF,
     b'if require_clean_git and porcelain.strip():',
     b'if False:',
     ["test_defect_6_dirty_tree_is_refused"]),

    # Defect 5 is DOUBLE-COVERED and isolates neither branch on its own: with
    # n_concepts=99 against a 5-concept store, the equality check (99 vs the
    # store's declared 5) and the content check (99 vs 5 actual groups) BOTH
    # fire, so disabling either leaves the other catching it. That is also
    # true of the real defect. Rather than claim single-branch coverage it
    # does not have, it is asserted against BOTH branches disabled together --
    # which is the honest statement of what it covers.
    ("BOTH n_concepts branches (equality + content) disabled", PF,
     b'if n_other is not None and int(n_other) != len(lab_concepts):',
     b'if False:',
     ["test_defect_5_replacement_inheriting_source_identity_is_refused"],
     [(PF, b'if k in lab_id and lab_id[k] != oid[k]:', b'if False:')]),

    # ---- the POPULATION OVERLOADING guard --------------------------------
    # `truncation_policy` (a label-store property) and `scored_row_set` (a
    # cache-row selection). Before amendment AE these collided by name.
    # are different axes sharing one word. Each fixture is built so that
    # `truncation_policy` AGREES everywhere, so only the named branch can kill
    # it -- which is exactly what these mutations verify.
    ("expected-value branch (scored_row_set vs the run's declaration)", PF,
     b'''            elif ckind == "expected":''',
     b'''            elif ckind == "expected_GONE":''',
     ["tests/test_preflight.py::"
      "test_scored_row_set_mismatch_is_refused_while_population_agrees"]),

    ("cross-artifact consistency branch (scored_row_set agreement)", PF,
     b'        if len(vals) > 1:',
     b'        if False:',
     ["tests/test_preflight.py::"
      "test_scored_row_set_disagreement_between_artifacts_is_refused"]),

    ("remove scored_row_set from the CANONICAL TABLE entirely", AI,
     # NOTE the `required` flip. Renaming the key ALONE is not a removal:
     # a renamed required key is absent from every artifact, so the
     # absent-is-not-matching branch fires in its place and the fixture
     # survives for a reason that has nothing to do with scored_row_set. The
     # mutation has to make the check genuinely inert.
     b'{"key": "scored_row_set", "kind": "expected", "applies_to": None,\n'
     b'         "required": True,',
     b'{"key": "scored_row_set_GONE", "kind": "expected", "applies_to": None,\n'
     b'         "required": False,',
     ["tests/test_preflight.py::"
      "test_scored_row_set_mismatch_is_refused_while_population_agrees",
      "tests/test_preflight.py::test_absent_scored_row_set_is_refused_not_assumed",
      "test_mutation_matrix_documents_which_branch_each_fixture_covers"],
     [(AI, b'{"key": "scored_row_set", "kind": "consistent", "applies_to": None,\n'
           b'         "required": True,',
           b'{"key": "scored_row_set_X", "kind": "consistent", "applies_to": None,\n'
           b'         "required": False,')]),

    # `library_sha` was in the identity set, had a branch, and could not fire:
    # the branch compared it to a key no label store writes. Reverting it to
    # `equal_to_labels` restores exactly that dead state, so the fixture must
    # die -- proving the fixture tests the FIX and not merely the key name.
    ("revert library_sha to the DEAD equal_to_labels form", AI,
     b'{"key": "library_sha", "kind": "derived", "applies_to": None,',
     b'{"key": "library_sha", "kind": "equal_to_labels", "applies_to": None,',
     ["tests/test_preflight.py::test_library_sha_mismatch_is_refused"]),

    # ---- derived-artifact inheritance: DEFAULT-DENY ----------------------
    # Reverting the allowlist to inherit-everything is exactly the denylist
    # state that leaked three times. The fixture is written against an
    # attribute the list has never seen, so it fails for the general reason
    # rather than for any particular key.
    ("revert inheritance allowlist to inherit-everything", DNR,
     b'            if k in INHERIT:',
     b'            if k not in frozenset():',
     ["tests/test_derive_null_reductions.py::"
      "test_unknown_source_attribute_does_not_propagate",
      "tests/test_derive_null_reductions.py::"
      "test_supersession_marker_is_not_inherited",
      "tests/test_derive_null_reductions.py::"
      "test_inheritance_decision_is_recorded_in_the_artifact"]),

    # ---- defect 9: a check that cannot fail ------------------------------
    # Reverting library_sha to `equal_to_labels` recreates the dead state.
    # Rule (a) must then fire on the SCHEMA -- which is the whole point: the
    # defect is detectable without any artifact at all.
    ("dead-check rule (a): equality key no store writes", AI,
     b'{"key": "library_sha", "kind": "derived", "applies_to": None,\n'
     b'         "required": True, "derived_from": "git_sha",',
     b'{"key": "library_sha", "kind": "equal_to_labels", "applies_to": None,\n'
     b'         "required": True,',
     # 9c is NOT listed: it checks that derived sources are reachable, and
     # reverting library_sha REMOVES it from the derived set entirely, leaving
     # a reachable remainder. It would survive, and listing it would claim
     # coverage this mutation does not provide.
     ["test_defect_9_equality_key_no_store_writes_is_refused",
      "tests/test_preflight.py::test_library_sha_mismatch_is_refused"]),

    ("dead-check rule (b): missing derived SOURCE key", PF,
     b'        if src not in lab_id_all:',
     b'        if False:',
     ["test_defect_9b_missing_derived_SOURCE_key_is_a_hard_failure"]),

    ("remove argmax_pool_n from the CANONICAL TABLE", AI,
     b'{"key": "argmax_pool_n", "kind": "derived", "applies_to": None,',
     b'{"key": "argmax_pool_n_GONE", "kind": "derived", "applies_to": None,',
     ["test_defect_1_pool_mismatch_is_refused",
      "test_mutation_matrix_documents_which_branch_each_fixture_covers"]),
]


def run_one(test_name: str) -> bool:
    """True if the fixture PASSES.

    A fixture may name its own file as `path.py::name`; a bare name means
    `TESTFILE`. Needed once the gate's coverage stopped living in one file --
    the scored_row_set and library_sha fixtures are in `test_preflight.py`, and
    silently running them against the historical file would have reported
    every one of them as a SURVIVOR for the wrong reason.
    """
    node = test_name if "::" in test_name else f"{TESTFILE}::{test_name}"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", node,
         "-q", "--tb=no", "--color=no", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=ROOT)
    return proc.returncode == 0


def main() -> int:
    # Baseline: every fixture must pass unmutated, or "killed" means nothing.
    all_fixtures = sorted({t for entry in MUTATIONS for t in entry[4]})
    baseline_fail = [t for t in all_fixtures if not run_one(t)]
    if baseline_fail:
        print("BASELINE BROKEN -- these fail before any mutation:")
        for t in baseline_fail:
            print(f"   {t}")
        return 1
    print(f"baseline: {len(all_fixtures)}/{len(all_fixtures)} fixtures pass\n")

    print(f"{'mutation':<50} {'result':<10} detail")
    bad = 0
    for entry in MUTATIONS:
        label, path, old, new, expect_fail = entry[:5]
        extra = entry[5] if len(entry) > 5 else []
        # Companion mutations applied alongside the primary one, for fixtures
        # that are legitimately covered by more than one branch.
        saved = []
        for xp, xold, xnew in extra:
            xorig = open(xp, "rb").read()
            saved.append((xp, xorig))
            xo, xn = xold, xnew
            if xorig.count(xo) != 1 and b"\n" in xold:
                xo, xn = xold.replace(b"\n", b"\r\n"), xnew.replace(b"\n", b"\r\n")
            if xorig.count(xo) == 1:
                open(xp, "wb").write(xorig.replace(xo, xn, 1))
        original = open(path, "rb").read()
        # Multi-line anchors must survive CRLF: the tree is checked out with
        # Windows line endings, so a b'\n' inside an anchor matches nothing.
        # This is the same line-ending trap that silently no-op'd two edits
        # earlier in the session -- here it at least announces itself.
        old_v, new_v = old, new
        if original.count(old_v) != 1 and b"\n" in old:
            old_v = old.replace(b"\n", b"\r\n")
            new_v = new.replace(b"\n", b"\r\n")
        n = original.count(old_v)
        if n != 1:
            print(f"{label:<50} {'SKIP':<10} anchor matched {n}x")
            bad += 1
            continue
        open(path, "wb").write(original.replace(old_v, new_v, 1))
        try:
            survived = [t for t in expect_fail if run_one(t)]
        finally:
            open(path, "wb").write(original)
            for xp, xorig in saved:
                open(xp, "wb").write(xorig)
        killed = len(expect_fail) - len(survived)
        status = "KILLED" if not survived else "SURVIVED"
        print(f"{label:<50} {status:<10} {killed}/{len(expect_fail)}")
        if survived:
            for t in survived:
                print(f"{'':<50} {'':<10} SURVIVOR: {t}")
            bad += 1

    print()
    if bad == 0:
        print("ALL MUTATIONS KILLED THEIR NAMED FIXTURES")
    else:
        print(f"{bad} MUTATION(S) NOT FULLY KILLED -- real coverage gap")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
