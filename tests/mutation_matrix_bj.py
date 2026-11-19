"""Mutation matrix for amendment BJ's row-index cache device key. NOT collected
by pytest.

Run manually:  python tests/mutation_matrix_bj.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

The runner is reused from `mutation_matrix_av` -- AU9b's control-first rule,
AV22's `__pycache__` purge between invocations, the byte-compared restore, and
the decide-on-EXIT-CODE rule. Reused rather than re-implemented: a runner is a
check, and a second copy is a second thing that can silently stop detecting
(gotcha 13 extension 5).

WHAT THIS FILE EXISTS FOR. The defect it guards against was live until BJ: the
cache key omitted the gather's device class while the gather demonstrably
depends on it (gotcha 36 measured `fire_counts` differing in 4.40% of
(seed, feature) pairs between CPU and `rtx_5000_ada` for the same checkpoint on
the same rows). A MISS was always safe; the hazard was a HIT serving an index
built by a different feature selection. The fixtures assert the repair; these
mutations establish that the fixtures would notice if it came undone.

THE FIVE SHAPES THE MUTATIONS COVER:

  * **THE DEVICE LEAVES THE KEY.** The defect itself, restored exactly -- the
    parts list drops its last element and every device class hashes alike.
  * **THE KEY IS COMPUTED FROM THE REQUEST RATHER THAN THE ENCODER.** The
    subtler half of requirement 2: a key that reads the `--device` string is
    device-aware and still wrong, because a job can request a GPU and run on
    CPU. Reading `W_enc` is what makes it a property of the model as loaded.
  * **ABSENCE BECOMES AGREEMENT.** `cached_device_class` defaulting to the
    running job's device instead of `ABSENT` is how an inference becomes
    indistinguishable from a measurement (gotcha 15), and it is the single
    branch through which back-filling could arrive by accident.
  * **THE RECONCILIATION STOPS RECONCILING.** The guard comparing the measured
    gather device against the key's is what turns "these agree" from a comment
    into a check.
  * **THE META STOPS RECORDING.** An entry that cannot state its own
    provenance is one whose device must be inferred by its next reader --
    which is the state BJ exists to leave behind.

WHAT IS NOT COVERED, stated rather than left to assumption: the CROSS-DEVICE
behaviour itself. Every fixture here runs on CPU, and no test in this repo can
exhibit a real CPU-versus-GPU gather divergence without a GPU. What is tested
is that the KEY separates the two classes and that the meta and the
reconciliation behave; that the two classes genuinely produce different indexes
is gotcha 36's measurement, not this suite's. A mutation asserting otherwise
would be a coverage claim on evidence that does not exist.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run  # noqa: E402
from tests.mutation_matrix_av import run_one  # noqa: E402

ROOT = os.getcwd()
P = os.path.join(ROOT, "src", "analysis", "row_index_cache.py")
T = "tests/test_row_index_cache.py"

# THE CONTROL: `cache_key` returns a constant before any part is assembled.
#
# It cannot fail to break the suite -- four separate fixtures assert that two
# keys DIFFER (checkpoint content, scored_row_set, k, device class), and a
# constant collapses every one of them. Deliberately a RETURN inserted ahead of
# the expression rather than an edit to a message or a docstring: BA17 item 2's
# first control changed a string while leaving every outcome reachable, and the
# runner correctly reported CONTROL NOT DETECTED.
CONTROL = (
    "CONTROL -- cache_key returns a constant before assembling any part", P,
    b"    st = eval_cache.stat()\n    parts = [",
    b'    return "0" * 16\n    st = eval_cache.stat()\n    parts = [',
    [f"{T}::test_two_keys_differing_ONLY_in_device_class_do_not_collide",
     f"{T}::test_different_scored_row_set_or_k_never_collide"],
)

MUTATIONS = [
    # ================================================ THE DEVICE IN THE KEY
    # THE ONE THIS FILE EXISTS FOR: the device leaves the parts list and the
    # pre-BJ key is back, byte for byte.
    ("cache_key: the device class is DROPPED from the parts list", P,
     b"        scored_row_set, str(k),\n        gather_device_class,\n    ]",
     b"        scored_row_set, str(k),\n    ]",
     [f"{T}::test_two_keys_differing_ONLY_in_device_class_do_not_collide",
      f"{T}::test_the_pre_BJ_key_form_COLLIDES_across_device_classes"]),

    # The device is in the list but cannot vary: a constant is device-shaped
    # and device-blind, which is the defect wearing the fix's clothes.
    ("cache_key: the device class is folded in as a CONSTANT", P,
     b"        scored_row_set, str(k),\n        gather_device_class,\n    ]",
     b'        scored_row_set, str(k),\n        "cpu",\n    ]',
     [f"{T}::test_two_keys_differing_ONLY_in_device_class_do_not_collide",
      f"{T}::test_the_pre_BJ_key_form_COLLIDES_across_device_classes"]),

    # A DEFAULT reintroduces the defect silently at any call site that forgets
    # the argument -- convenience arriving where a decision belongs.
    ("cache_key: the device class gains a DEFAULT", P,
     b"def cache_key(ckpt_dir: Path, eval_cache: Path, scored_row_set: str, k: int,\n"
     b"              gather_device_class: str) -> str:",
     b"def cache_key(ckpt_dir: Path, eval_cache: Path, scored_row_set: str, k: int,\n"
     b'              gather_device_class: str = "cpu") -> str:',
     [f"{T}::test_the_key_requires_a_device_class_with_no_default"]),

    # ============================================ THE ENCODER, NOT THE REQUEST
    # Requirement 2's subtle half. A key read off `torch.device(...)` of a
    # requested string is device-aware and still not a measurement.
    ("encoder_device_class: reads a hardcoded string, not W_enc", P,
     b"    return device_class(model.W_enc)",
     b'    return device_class("cpu")',
     [f"{T}::test_the_key_comes_from_the_ENCODER_not_from_the_device_argument"]),

    # ============================================== ABSENCE BECOMES AGREEMENT
    # The single branch through which back-filling could arrive by accident.
    ("cached_device_class: a missing attr defaults to the running device", P,
     b"        if GATHER_DEVICE_KEY not in hf.attrs:\n            return DEVICE_ABSENT",
     b"        if GATHER_DEVICE_KEY not in hf.attrs:\n            import platform\n"
     b'            return "cpu:" + platform.machine()',
     [f"{T}::test_an_entry_with_NO_recorded_device_is_reported_ABSENT_not_matching"]),

    # ========================================== THE RECONCILIATION IS A CHECK
    ("compute_or_load_row_index: the gather/key reconciliation never fires", P,
     b"    if measured_dc is not None and measured_dc != gather_dc:",
     b"    if False:",
     [f"{T}::test_a_gather_on_a_device_other_than_the_KEYs_is_REFUSED"]),

    # ================================================= THE META STOPS RECORDING
    ("save meta: the gather device is not written into the entry", P,
     b"        **{GATHER_DEVICE_KEY: gather_dc},\n    ))",
     b"    ))",
     [f"{T}::test_a_fresh_entry_RECORDS_its_gather_device_in_the_meta"]),

    # The hit path stops asking the entry and starts asserting. This is the
    # exact conversion -- an inference where a measurement belongs -- that the
    # `UNKNOWN_CACHED_ROW_INDEX` label existed to prevent.
    ("hit path: reports the key's device instead of the entry's own record", P,
     b"            record[GATHER_DEVICE_KEY] = cached_device_class(path)",
     b"            record[GATHER_DEVICE_KEY] = gather_dc",
     [f"{T}::test_an_entry_with_NO_recorded_device_is_reported_ABSENT_not_matching"]),
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
        if survived:
            bad += 1
    print(f"\n{len(MUTATIONS) - bad}/{len(MUTATIONS)} mutations killed, "
          f"control detected")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
