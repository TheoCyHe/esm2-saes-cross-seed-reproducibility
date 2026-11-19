"""Mutation matrix for the gather-device identity key (gotcha 36). NOT
collected by pytest.

Run manually:  python tests/mutation_matrix_device.py
Exit 0 iff the CONTROL is detected, every mutation kills its named fixtures,
and the BENIGN edit survives.

The runner is reused from `mutation_matrix_av` -- control-first, `__pycache__`
purged between invocations, byte-compared restore, decisions on pytest's EXIT
CODE. A runner is a check, and a second copy is a second thing that can
silently stop detecting (gotcha 13 extension 5).

THE MUTATION THIS FILE EXISTS FOR IS THE FIRST ONE: **both sides absent
reported as AGREE.** Every artifact on disk today predates this key, so
both-absent is the COMMON case, not an edge case -- a permissive comparison
would report agreement for every real pair while checking nothing, and would
read as coverage in any audit. That is gotcha 15's shape (a guard satisfied by
the wrong thing converts a loud failure into a silent one) arriving inside the
key added to close gotcha 36.

The second is the one that decides whether the key means anything at all: the
gather recording the device it was ASKED for rather than the one it RAN on. A
job can request a GPU and run the arithmetic on CPU -- that is precisely what
got four tasks reclaimed -- so a key populated from the request would be
confidently wrong exactly where it matters.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run  # noqa: E402
from tests.mutation_matrix_av import run_one  # noqa: E402

ROOT = os.getcwd()
I = os.path.join(ROOT, "src", "analysis", "artifact_identity.py")
F = os.path.join(ROOT, "src", "analysis", "feature_ap.py")
T = "tests/test_artifact_identity.py"

# THE CONTROL: `device_class` returns a constant before any branch runs. It
# cannot fail to break the suite -- three fixtures assert three DISTINCT
# strings from it (a cpu class, a tensor's class, an unknown backend) and a
# constant collapses all three. A return inserted ahead of the first statement,
# not an edit to a message: BA17 item 2's first control changed a string while
# leaving every outcome reachable, and was correctly reported NOT DETECTED.
CONTROL = (
    "CONTROL -- device_class returns a constant before any branch", I,
    b'    dev = getattr(device, "device", device)',
    b'    return "cpu:CONTROL"\n'
    b'    dev = getattr(device, "device", device)',
    [f"{T}::test_device_class_names_the_backend_and_the_machine_for_cpu"],
)

MUTATIONS = [
    # ============================================ ABSENCE READ AS AGREEMENT
    # THE ONE THIS FILE EXISTS FOR.
    ("compare_device_class: BOTH ABSENT reported as AGREE", I,
     b'        agree, verdict = False, "ABSENT_ON_BOTH"',
     b'        agree, verdict = True, "AGREE"',
     [f"{T}::test_BOTH_ABSENT_IS_NOT_AGREEMENT",
      f"{T}::test_an_absent_device_class_does_not_make_two_artifacts_unequal_either",
      f"{T}::test_the_device_comparison_is_TOTAL_over_the_two_presence_bits"]),

    # The same defect on the other presence pattern: one side able to state its
    # device, the other not, is UNVERIFIED -- not matched.
    ("compare_device_class: ABSENT ON ONE reported as AGREE", I,
     b'        agree, verdict = False, "ABSENT_ON_ONE"',
     b'        agree, verdict = True, "AGREE"',
     [f"{T}::test_the_device_comparison_is_TOTAL_over_the_two_presence_bits"]),

    # And the mirror error: reporting ignorance as a DIFFERENCE is a claim
    # about two devices nobody recorded, in the other direction.
    ("compare_device_class: BOTH ABSENT reported as DIFFER", I,
     b'        agree, verdict = False, "ABSENT_ON_BOTH"',
     b'        agree, verdict = False, "DIFFER"',
     [f"{T}::test_an_absent_device_class_does_not_make_two_artifacts_unequal_either",
      f"{T}::test_BOTH_ABSENT_IS_NOT_AGREEMENT",
      f"{T}::test_the_device_comparison_is_TOTAL_over_the_two_presence_bits"]),

    # ==================================== THE REQUESTED VS THE ACTUAL DEVICE
    # The key is worthless if it records the request. gotcha 36's routing note
    # is exactly this: a GPU request can buy only the gather while the
    # arithmetic runs on CPU.
    ("sparse_activations: records the REQUESTED device instead of the one used",
     F,
     b"                record[GATHER_DEVICE_KEY] = device_class(z.device)",
     b"                record[GATHER_DEVICE_KEY] = device_class(device)",
     [f"{T}::test_the_gather_records_the_device_it_RAN_on_not_the_one_requested"]),

    # ================================================ THE CLASS ITSELF
    # "cpu" versus "cuda" is not a device CLASS. Dropping the discriminating
    # component makes two different devices produce the same string, which is
    # the one thing the key must never do.
    ("device_class: the cpu branch drops the machine", I,
     b"        return f\"cpu:{platform.machine() or 'UNKNOWN_MACHINE'}\"",
     b'        return "cpu"',
     [f"{T}::test_device_class_names_the_backend_and_the_machine_for_cpu",
      f"{T}::test_device_class_is_more_than_cpu_vs_gpu"]),

    # A backend this function does not understand must be NAMED, never coerced
    # into one it does -- reporting `meta` as cpu is a fabricated measurement.
    ("device_class: an unknown backend is coerced to cpu", I,
     b'    return f"{dev.type}:UNKNOWN_CLASS"',
     b'    return "cpu:" + __import__("platform").machine()',
     [f"{T}::test_device_class_names_an_UNKNOWN_backend_rather_than_coercing_it",
      f"{T}::test_the_gather_records_the_device_it_RAN_on_not_the_one_requested"]),

    # ================================================= MIXED HARDWARE
    # An array whose seeds ran on two device classes has no single device
    # class. Collapsing to the last seed states a device the artifact lacks.
    ("device_identity_attrs: MIXED collapses to the last seed's value", I,
     b'        out[k] = v if len(vals) <= 1 else "MIXED:" + ",".join(vals)',
     b'        out[k] = v',
     [f"{T}::test_mixed_hardware_is_recorded_as_MIXED_not_as_the_last_seed"]),

    # ...and the opposite: a single-valued run must NOT be dressed up as mixed.
    ("device_identity_attrs: every value is written as MIXED", I,
     b'        out[k] = v if len(vals) <= 1 else "MIXED:" + ",".join(vals)',
     b'        out[k] = "MIXED:" + ",".join(vals)',
     [f"{T}::test_a_single_device_class_is_written_plainly"]),
]

# A change that alters the source WITHOUT altering behaviour must SURVIVE. If
# it is killed, the fixtures are reacting to edits rather than to defects.
BENIGN = [
    ("BENIGN -- a comment is reworded (behaviour identical; must SURVIVE)", I,
     b"    out = {}\n    for k, v in record.items():",
     b"    out = {}\n    # one row per recorded device key\n"
     b"    for k, v in record.items():",
     [f"{T}::test_mixed_hardware_is_recorded_as_MIXED_not_as_the_last_seed",
      f"{T}::test_a_single_device_class_is_written_plainly"]),
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
        if not ok:
            print(f"{'':<74} {'':<10} A BEHAVIOUR-PRESERVING EDIT WAS KILLED -- "
                  "the suite is reacting to edits, not to defects.")
            bad += 1

    print()
    print("ALL MUTATIONS KILLED THEIR NAMED FIXTURES; BENIGN EDIT SURVIVED"
          if not bad else f"{bad} MUTATION(S) SURVIVED, WERE SKIPPED, OR "
                          "KILLED A BENIGN EDIT")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
