"""Mutation matrix for amendment BK's canary additions. NOT collected by pytest.

Run manually:  python tests/mutation_matrix_bk.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

The runner is reused from `mutation_matrix_av` -- AU9b's control-first rule,
AV22's `__pycache__` purge between invocations, the byte-compared restore, and
the decide-on-EXIT-CODE rule. Reused rather than re-implemented: a runner is a
check, and a second copy is a second thing that can silently stop detecting
(gotcha 13 extension 5).

WHAT THIS FILE EXISTS FOR. BK adds six targets carrying a WEAKER claim than any
existing target -- file integrity, not reproduction -- and a three-part device
field to describe them honestly. Both are the kind of change that reads as
working while doing nothing: a coverage function whose failing branch is
unreachable from the real `TARGETS` passes forever, and a tolerance string
nobody asserts on can lose the sentence that makes the claim honest.

THE FOUR SHAPES THE MUTATIONS COVER:

  * **THE PAIRING STOPS BEING ENFORCED.** `build_device_class` without
    `build_device_provenance` is the whole defect the flag exists to prevent:
    an inferred device with nothing recording that it was inferred is
    indistinguishable from a measured one (gotcha 15). Both directions, because
    a check capable of only one of two symmetric failures is gotcha 25's shape.
  * **ABSENCE BECOMES A FAILURE, OR A DEFAULT BECOMES A MEASUREMENT.** A target
    stating neither field must be COVERED and must appear under
    `inheriting_the_default`; folding it into `stating_their_own` is exactly how
    the module constant would start looking like a per-target measurement --
    LIMB 2 of BJ-C3's stop condition, reintroduced.
  * **THE WEAKER CLAIM STOPS SAYING IT IS WEAKER.** A green verdict on
    `build_inputs` reads exactly like a green verdict on `n19`. The only thing
    separating them is the tolerance text, so it is asserted rather than
    trusted.
  * **THE RECURSIVE WALK REVERTS.** The flat comprehension raises
    `AttributeError` on a grouped file, so three of the six additions could not
    be described at all. A mutation restoring it must be caught by a fixture
    that USES a grouped file, not by one that merely calls `describe`.

WHAT IS NOT COVERED, stated rather than left to assumption: that the six
artifacts named are in fact the classification build's inputs. That is BJ-C1's
enumeration, re-confirmed for BK by reading every file's own attrs on disk, and
no unit test can establish it -- `test_the_six_build_inputs_are_the_ones_BJ_C1
_enumerated` pins the LIST so a later change is deliberate, which is a different
guarantee and is the one available here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run  # noqa: E402
from tests.mutation_matrix_av import run_one  # noqa: E402

ROOT = os.getcwd()
P = os.path.join(ROOT, "src", "analysis", "canary.py")
T = "tests/test_canary.py"

# THE CONTROL: `build_device_coverage` returns COVERED before it looks at
# anything. It cannot fail to break the suite -- four fixtures assert a
# non-COVERED verdict on injected targets, and three more read fields this
# never populates. Deliberately a RETURN inserted ahead of the loop rather than
# an edit to a message: BA17 item 2's first control changed a string while
# leaving every outcome reachable, and the runner correctly reported CONTROL
# NOT DETECTED.
CONTROL = (
    "CONTROL -- build_device_coverage returns COVERED before inspecting "
    "anything", P,
    b"    stating, inheriting = [], []",
    b'    return {"verdict": "COVERED", "class_without_provenance": [],\n'
    b'            "provenance_without_class": [],\n'
    b'            "undeclared_build_device_classes": [],\n'
    b'            "undeclared_build_device_provenance": [],\n'
    b'            "targets_stating_their_own": [],\n'
    b'            "targets_inheriting_the_default": [],\n'
    b'            "targets_by_build_device": {}, "module_default": "x",\n'
    b'            "build_device_classes": [], '
    b'"build_device_provenance_values": []}\n'
    b"    stating, inheriting = [], []",
    [f"{T}::test_a_build_device_class_without_its_provenance_is_UNPAIRED",
     f"{T}::test_a_provenance_flag_without_a_class_is_ALSO_UNPAIRED",
     f"{T}::test_an_undeclared_build_device_value_is_its_own_verdict",
     f"{T}::test_only_the_BK_additions_state_their_own_build_device"],
)

MUTATIONS = [
    # ==================================================== THE PAIRING IS A CHECK
    # THE ONE THIS FILE EXISTS FOR. A class with no provenance flag is an
    # inferred device that reads as measured -- and it is the single branch
    # through which that could arrive silently.
    ("build_device_coverage: a class with no provenance is not recorded", P,
     b"        if c is not None and pv is None:\n"
     b"            class_without_provenance.append(t[\"name\"])",
     b"        if False:\n"
     b"            class_without_provenance.append(t[\"name\"])",
     [f"{T}::test_a_build_device_class_without_its_provenance_is_UNPAIRED"]),

    # The other direction. A check capable of only one of two symmetric
    # failures is as uninformative as one that can never fail (gotcha 25).
    ("build_device_coverage: a provenance flag with no class is not recorded",
     P,
     b"        elif pv is not None and c is None:\n"
     b"            provenance_without_class.append(t[\"name\"])",
     b"        elif False:\n"
     b"            provenance_without_class.append(t[\"name\"])",
     [f"{T}::test_a_provenance_flag_without_a_class_is_ALSO_UNPAIRED"]),

    # UNPAIRED and UNDECLARED are separate verdicts because the fixes differ.
    # Collapsing them into one is the one-key-two-meanings defect gotcha 16
    # committed a convention against.
    ("build_device_coverage: an undeclared VALUE is folded into UNPAIRED", P,
     b'        verdict = "UNDECLARED_BUILD_DEVICE_VALUE"',
     b'        verdict = "UNPAIRED_BUILD_DEVICE_FIELDS"',
     [f"{T}::test_an_undeclared_build_device_value_is_its_own_verdict"]),

    # The vocabulary stops constraining anything.
    ("build_device_coverage: an undeclared class is never collected", P,
     b"        if c is not None and c not in classes:\n"
     b"            undeclared_classes.add(c)",
     b"        if False:\n"
     b"            undeclared_classes.add(c)",
     [f"{T}::test_an_undeclared_build_device_value_is_its_own_verdict"]),

    # ============================== A DEFAULT MUST NOT BECOME A MEASUREMENT
    # LIMB 2 of BJ-C3's stop condition, restored: every pre-BK target would be
    # described as having stated `rtx_5000_ada` about itself, which is the
    # inference-where-a-measurement-belongs conversion the whole field exists
    # to prevent.
    ("build_device_coverage: a target stating NEITHER field is counted as "
     "stating its own", P,
     b"        if c is None and pv is None:\n"
     b"            inheriting.append(t[\"name\"])\n"
     b"            continue",
     b"        if c is None and pv is None:\n"
     b"            stating.append(t[\"name\"])\n"
     b"            continue",
     [f"{T}::test_only_the_BK_additions_state_their_own_build_device",
      f"{T}::test_a_target_stating_NEITHER_field_is_COVERED_not_a_failure"]),

    # Inheriting the default is the CORRECT state for every pre-BK target.
    # Reporting it as a defect makes the check one that can only fail.
    ("build_device_coverage: inheriting the default is reported as UNPAIRED",
     P,
     b'    if class_without_provenance or provenance_without_class:',
     b'    if inheriting or class_without_provenance or provenance_without_class:',
     [f"{T}::test_a_target_stating_NEITHER_field_is_COVERED_not_a_failure",
      f"{T}::test_every_real_target_pairs_its_build_device_fields"]),

    # ============================ THE WEAKER CLAIM MUST SAY IT IS WEAKER
    # A green verdict on this arm reads exactly like a green verdict on n19.
    # The tolerance text is the only thing that separates them.
    ("TOLERANCE: build_inputs stops calling itself a weaker claim", P,
     b'        "FILE INTEGRITY ONLY -- THIS IS A WEAKER CLAIM THAN EVERY OTHER ARM "\n'
     b'        "IN THIS MANIFEST, AND IT IS NOT A REPRODUCTION GUARANTEE. "',
     b'        "Bit-identical for the stored bytes. "',
     [f"{T}::test_the_build_inputs_tolerance_says_it_is_WEAKER_and_not_reproduction"]),

    # THE FAILURE THE ARM IS MOST LIKELY TO INVITE: reusing the pooled null's
    # control-measured floor for artifacts it was not measured on, four of
    # which ran on CPU where the CUDA atomics argument does not even apply.
    ("TOLERANCE: the pooled null's floor stops being ring-fenced", P,
     b'        "THE POOLED NULL\'S MEASURED FLOOR IS NOT TRANSFERRED TO ANY TARGET ON "\n'
     b'        "THIS ARM. It was measured on null_pilot_k.h5 via a CUDA path whose "',
     b'        "The pooled null\'s floor applies to this arm too. It was measured "\n'
     b'        "on null_pilot_k.h5 via a CUDA path whose "',
     [f"{T}::test_the_pooled_null_floor_is_explicitly_NOT_transferred_to_this_arm"]),

    # ==================================== THE SUPERSEDED TARGET'S OWN REASON
    # BJ-C2: a green verify on `null_pilot_k.h5` is readable as that artifact
    # being fine to use, and the pre-flight gate refuses it for every consumer.
    ("null_pooled_reductions: the reason stops naming the pre-flight refusal",
     P,
     b'            "fire_rate) that PRESERVE last-bit movement. REFUSED FOR CONSUMERS "\n'
     b'            "by the pre-flight gate (`superseded_for_axis`, 716 lib / 707 "\n'
     b'            "pool); NOT the artifact any reported result descends from. The "\n'
     b'            "pooled null the classification reads is null_pilot_compliant.h5, "\n'
     b'            "covered separately on the `build_inputs` arm (BJ-C2, BK)"},',
     b'            "fire_rate) that PRESERVE last-bit movement. No null artifact "\n'
     b'            "was covered before"},',
     [f"{T}::test_the_superseded_pooled_null_target_says_which_question_it_answers"]),

    # ================================================== THE RECURSIVE WALK
    # The pre-BK comprehension, restored byte for byte. It RAISES on a grouped
    # file rather than returning a wrong answer, so the fixture that catches it
    # has to USE a grouped file -- one that merely calls `describe` on a flat
    # one would pass.
    ("describe: the h5 walk reverts to top-level datasets only", P,
     b"            datasets: dict[str, list[int]] = {}\n"
     b"            hf.visititems(\n"
     b"                lambda name, obj: datasets.__setitem__(name, list(obj.shape))\n"
     b"                if isinstance(obj, h5py.Dataset) else None)",
     b"            datasets = {k: list(hf[k].shape) for k in hf}",
     [f"{T}::test_describe_walks_h5_GROUPS_and_not_only_top_level_datasets"]),

    # ===================================== THE VOCABULARY STAYS TWO-VALUED
    # BJ-C3's LIMB 1: marking a gather-descended artifact NOT_ESTABLISHED
    # asserts ignorance of a measurement this project made and acted on.
    ("DEVICE_CLASSES: TOPK_GATHER is collapsed back into MNN", P,
     b'TOPK_GATHER = "TOPK_GATHER_NEAR_TIE"',
     b'TOPK_GATHER = "MNN_COSINE_THRESHOLD"',
     [f"{T}::test_TOPK_GATHER_is_a_SEPARATE_value_and_not_a_widened_MNN",
      f"{T}::test_the_TOPK_GATHER_targets_are_the_ones_whose_content_descends_from_a_gather"]),
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

    print(f"{'mutation':<66} {'result':<10} killed")
    bad = 0
    for label, path, old, new, expect in MUTATIONS:
        survived = apply_and_run(path, old, new, expect)
        if survived is None:
            print(f"{label:<66} {'SKIP':<10} anchor did not match exactly once")
            bad += 1
            continue
        status = "KILLED" if not survived else "SURVIVED"
        print(f"{label:<66} {status:<10} {len(expect) - len(survived)}/{len(expect)}")
        for t in survived:
            print(f"{'':<66} {'':<10} SURVIVOR: {t}")
        if survived:
            bad += 1
    print(f"\n{len(MUTATIONS) - bad}/{len(MUTATIONS)} mutations killed, "
          f"control detected")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
