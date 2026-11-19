"""Mutation matrix for the concept-axis and `ap_impl` relationship checks
added 2026-08-11. NOT collected by pytest.

Run manually:  python tests/mutation_matrix_preflight_axis.py
Exit 0 iff the CONTROL is detected, every mutation kills its named fixtures,
and the BENIGN edit survives.

The runner is reused from `mutation_matrix_av` -- control-first, `__pycache__`
purged between invocations, byte-compared restore, decisions on pytest's EXIT
CODE. A second copy of a runner is a second thing that can silently stop
detecting (gotcha 13 extension 5).

WHAT THIS FILE EXISTS FOR. The experimental pair failed pre-flight PERMANENTLY
and correctly: its AP artifact's axis is a 227-concept subset of the store's
712 because only those concepts have an experimental residue, and masking
forces the segmented implementation. A check that always fails detects nothing
-- a genuine defect in that pair would have been one more line in an
already-red report. That is the coverage illusion, and it is worse than no
guard because it reads as protection in every audit.

THE FIRST TWO MUTATIONS ARE THE ONES THIS FILE IS ABOUT, and they are the two
ways the obvious fix goes wrong:

  1. THE EXEMPTION. Waving the axis through for stratified artifacts also waves
     through a GENUINE mismatch -- the same problem one level down. The fixture
     that must kill it is the WRONG SUBSET OF THE RIGHT SIZE, which no count
     check and no exemption can see.
  2. THE SET COMPARISON. The null's axis must be the argmax-pool subset of the
     AP artifact's POSITIONALLY. Two orderings of the same objects with the
     same total is an invisible collision that no name and no length
     distinguishes (gotcha 27), and the seed-axis precedent shows the cost: 18
     of 20 rows mis-paired into a complete, internally consistent, entirely
     wrong result.

Two of them encode defects that were IN THIS CODE and were caught by running
it rather than by reading it: pool membership derived from the artifact's own
axis instead of the library's (a parent whose children are not scored reads as
in-pool), and the null's axis never being read at all.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run  # noqa: E402
from tests.mutation_matrix_av import run_one  # noqa: E402

ROOT = os.getcwd()
P = os.path.join(ROOT, "src", "analysis", "preflight.py")
I = os.path.join(ROOT, "src", "analysis", "artifact_identity.py")
T = "tests/test_preflight.py"

# THE CONTROL: `stratum_concept_axis` returns a constant before any branch. Not
# a no-op and not a message edit -- three fixtures assert three DISTINCT axes
# from it (the experimental axis, the propagated axis, and the full axis it
# must NOT return), so a constant cannot satisfy them. Inserted ahead of the
# first statement of the body.
CONTROL = (
    "CONTROL -- stratum_concept_axis returns a constant before any branch", I,
    b'    if stratum == "pooled":\n        raise ValueError(',
    b'    return ["ACT_SITE"]\n'
    b'    if stratum == "pooled":\n        raise ValueError(',
    [f"{T}::test_the_STRATUM_AXIS_DERIVATION_matches_what_is_written_down",
     f"{T}::test_pooled_is_refused_as_a_stratum_rather_than_returning_the_full_axis"],
)

MUTATIONS = [
    # =============================================== 1. THE BLUNT FIX ITSELF
    # "Just exempt the axis for stratified artifacts." It makes the pair green
    # and checks nothing.
    ("THE EXEMPTION -- a stratified axis is accepted whatever it contains", P,
     b"            elif sorted(cs) != sorted(want_axis):",
     b"            elif axis_kind == 'pooled' and sorted(cs) != sorted(want_axis):",
     [f"{T}::test_a_WRONG_SUBSET_OF_THE_RIGHT_SIZE_is_caught"]),

    # ================================= 2. THE POSITIONAL CHECK MADE SET-WISE
    # Same concepts, same count, different ORDER. Only a positional check
    # discriminates a mapping defect (gotcha 27).
    ("null-vs-AP axis compared as a SET rather than positionally", P,
     b"            ok = list(ncs) == want",
     b"            ok = sorted(ncs) == sorted(want)",
     [f"{T}::test_a_PERMUTED_null_axis_is_caught_POSITIONALLY"]),

    # ...and the weakest form of the same error. TWO fixtures, deliberately:
    # the first run listed the pooled-parent case here and it SURVIVED, because
    # that case has a DIFFERENT count and a count check still catches it. The
    # cover was then a single fixture, so `test_..._RIGHT_COUNT_and_WRONG_
    # MEMBERS` was added -- a permutation and an equal-sized wrong membership
    # are different defect classes and a count check is blind to both.
    ("null-vs-AP axis compared by COUNT", P,
     b"            ok = list(ncs) == want",
     b"            ok = len(ncs) == len(want)",
     [f"{T}::test_a_PERMUTED_null_axis_is_caught_POSITIONALLY",
      f"{T}::test_a_null_with_the_RIGHT_COUNT_and_WRONG_MEMBERS_is_caught"]),

    # ================================== 3. POOL MEMBERSHIP FROM THE WRONG SET
    # THE DEFECT THAT WAS ACTUALLY IN THIS CODE. `argmax_pool_mask` calls a
    # concept a pooled parent when a CHILD is present in the list it is given,
    # so deriving the pool from the artifact's own stratified axis lets a
    # parent whose children are not scored read as in-pool.
    ("pool membership derived from the ARTIFACT's axis, not the library's", P,
     b"            want = [c for c in acs if c in lib_pool]",
     b"            want = [c for c, k in zip(acs, argmax_pool_mask(acs)) if k]",
     [f"{T}::test_a_null_carrying_a_POOLED_PARENT_is_caught"]),

    # ================================ 4. THE NULL'S AXIS NEVER READ AT ALL
    # The state before this change: `[]`, which is not "checked and equal".
    ("the null's concept axis is not read (back to the empty list)", P,
     b'            others.append(("null", null_artifact, concept_names(null_artifact)))',
     b'            others.append(("null", null_artifact, []))',
     [f"{T}::test_a_null_MISSING_a_pool_concept_is_caught",
      f"{T}::test_a_null_carrying_a_POOLED_PARENT_is_caught",
      f"{T}::test_a_PERMUTED_null_axis_is_caught_POSITIONALLY",
      f"{T}::test_a_null_that_is_the_ARGMAX_POOL_SUBSET_passes"]),

    # `concept_names` must handle a ROOT-LEVEL concepts dataset -- the layout
    # the null actually writes. Losing that branch silently empties the axis.
    ("concept_names: the root-level null layout falls through to []", P,
     b'        if group is None and "concepts" in hf:                 # null artifact',
     b'        if False:                                              # null artifact',
     [f"{T}::test_a_null_MISSING_a_pool_concept_is_caught",
      f"{T}::test_a_PERMUTED_null_axis_is_caught_POSITIONALLY"]),

    # ============================================= 5. THE DERIVATION ITSELF
    # ONE fixture, and that is the honest listing rather than the flattering
    # one: `test_a_WRONG_SUBSET_OF_THE_RIGHT_SIZE` was listed here on the first
    # run and SURVIVED, because a wrong subset differs from the FULL axis too
    # and it raises either way. A fixture that cannot discriminate a mutation
    # must not be claimed as cover for it.
    ("stratum_concept_axis: every concept is on every stratum's axis", I,
     b"            if any(TIER_GROUP_MAP.get(t) == stratum and g[t].shape[0] > 0",
     b"            if any(True or TIER_GROUP_MAP.get(t) == stratum and g[t].shape[0] > 0",
     [f"{T}::test_the_STRATUM_AXIS_DERIVATION_matches_what_is_written_down",
      f"{T}::test_a_STRATIFIED_artifact_with_the_right_subset_PASSES"]),

    # A tier that EXISTS but is EMPTY is not evidence. This is the same shape
    # as "absent is not matching", inside the derivation.
    ("stratum_concept_axis: a present-but-EMPTY tier counts as evidence", I,
     b"            if any(TIER_GROUP_MAP.get(t) == stratum and g[t].shape[0] > 0",
     b"            if any(TIER_GROUP_MAP.get(t) == stratum and g[t].shape[0] >= 0",
     [f"{T}::test_the_STRATUM_AXIS_DERIVATION_matches_what_is_written_down"]),

    # The high-throughput tier IS experimental (A1). Dropping it silently
    # shrinks the axis, and the artifact would then look like the defective one.
    ("TIER_GROUP_MAP: experimental_highthroughput regrouped as propagated", I,
     b'    "experimental": "experimental", "experimental_highthroughput": "experimental",',
     b'    "experimental": "experimental", "experimental_highthroughput": "propagated",',
     [f"{T}::test_the_STRATUM_AXIS_DERIVATION_matches_what_is_written_down"]),

    # `pooled` must be refused rather than returning the full axis, so the two
    # branches cannot look interchangeable when only one of them can fail.
    ("stratum_concept_axis: `pooled` returns the full axis instead of refusing",
     I,
     b'    if stratum == "pooled":\n        raise ValueError(',
     b'    if stratum == "pooled":\n        return sorted(h5py.File(labels_path)["concepts"])\n'
     b'    if False:\n        raise ValueError(',
     [f"{T}::test_pooled_is_refused_as_a_stratum_rather_than_returning_the_full_axis"]),

    # ==================================== 6. NOT DERIVABLE REPORTED AS PASSED
    # gotcha 25c: an expectation that cannot be formed must say so. Dropping it
    # into silence is how an UNGOVERNED key becomes an unchecked one.
    ("a NOT-DERIVABLE axis expectation is silently passed", P,
     b'                not_applicable.append(\n'
     b'                    f"{p.name}: concept AXIS expectation NOT DERIVABLE "',
     b'                _ = (\n'
     b'                    f"{p.name}: concept AXIS expectation NOT DERIVABLE "',
     [f"{T}::test_TWO_declared_strata_is_NOT_DERIVABLE_not_passed"]),

    # An artifact that does not DECLARE `strata` must take the pooled branch
    # explicitly, not be guessed into the stratified one.
    ("an UNDECLARED strata attr is treated as stratified", P,
     b'            axis_kind = ("pooled" if strata in ([], ["pooled"])',
     b'            axis_kind = ("pooled" if strata in ([None], ["pooled"])',
     [f"{T}::test_ap_concept_axis_mismatch_is_caught",
      f"{T}::test_a_POOLED_artifact_is_STILL_held_to_equality"]),

    # ===================================== 7. THE `ap_impl` DERIVATION
    # A blanket acceptance dressed as a derivation: every artifact expected to
    # be segmented means an unmasked run using the wrong path sails through.
    ("expected_ap_impl: SEGMENTED is expected unconditionally", I,
     b'    if neg == "unannotated":',
     b'    if True:',
     [f"{T}::test_a_SEGMENTED_marker_on_an_UNMASKED_artifact_STILL_FAILS",
      f"{T}::test_wrong_ap_impl_is_caught"]),

    # ...and the derivation removed entirely, which restores the permanent
    # failure this change exists to end.
    ("expected_ap_impl: the masking branch is dropped", I,
     b'    if neg == "unannotated":\n        return AP_IMPL_SEGMENTED, (',
     b'    if False:\n        return AP_IMPL_SEGMENTED, (',
     [f"{T}::test_a_MASKED_artifact_is_expected_to_be_SEGMENTED",
      f"{T}::test_a_STRATIFIED_artifact_with_the_right_subset_PASSES"]),

    # `negatives` absent must fall back to the declared default, not to the
    # masked one -- every artifact predating the attr reaches this branch.
    ("expected_ap_impl: an ABSENT negatives attr is read as masked", I,
     b'    neg = negatives.decode() if isinstance(negatives, bytes) else negatives',
     b'    neg = "unannotated" if negatives is None else (\n'
     b'        negatives.decode() if isinstance(negatives, bytes) else negatives)',
     [f"{T}::test_an_ABSENT_negatives_attr_falls_back_and_does_not_crash",
      f"{T}::test_wrong_ap_impl_is_caught"]),
]

# A change that alters the source WITHOUT altering behaviour must SURVIVE. If
# it is killed, the fixtures are reacting to edits rather than to defects.
BENIGN = [
    ("BENIGN -- a comment is reworded (behaviour identical; must SURVIVE)", P,
     b"    lib_pool = {c for c, keep in zip(lab_concepts,",
     b"    # the library decides what a pooled parent is\n"
     b"    lib_pool = {c for c, keep in zip(lab_concepts,",
     [f"{T}::test_a_null_carrying_a_POOLED_PARENT_is_caught",
      f"{T}::test_a_null_that_is_the_ARGMAX_POOL_SUBSET_passes"]),
]


def main() -> int:
    fixtures = sorted({t for e in MUTATIONS for t in e[4]}
                      | set(CONTROL[4])
                      | {t for e in BENIGN for t in e[4]})
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

    print(f"{'mutation':<74} {'result':<10} killed")
    bad = 0
    for label, path, old, new, expect in MUTATIONS:
        survived = apply_and_run(path, old, new, expect)
        if survived is None:
            print(f"{label:<74} {'SKIP':<10} anchor did not match exactly once")
            bad += 1
            continue
        status = "KILLED" if not survived else "SURVIVED"
        print(f"{label:<74} {status:<10} "
              f"{len(expect) - len(survived)}/{len(expect)}")
        for t in survived:
            print(f"{'':<74} {'':<10} SURVIVOR: {t}")
        bad += bool(survived)

    print()
    for label, path, old, new, expect in BENIGN:
        survived = apply_and_run(path, old, new, expect)
        if survived is None:
            print(f"{label:<74} {'SKIP':<10} anchor did not match exactly once")
            bad += 1
            continue
        ok = len(survived) == len(expect)
        print(f"{label:<74} {'SURVIVED' if ok else 'KILLED':<10} "
              f"{len(expect) - len(survived)}/{len(expect)}")
        bad += not ok

    print()
    print("ALL MUTATIONS KILLED, CONTROL DETECTED, BENIGN SURVIVED"
          if not bad else f"{bad} problem(s) -- see above")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
