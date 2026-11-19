"""Mutation matrix for amendment BT part A3's RELOCATION classification.
NOT collected by pytest.

Run manually:  python tests/mutation_matrix_bt_canary.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

The runner is reused from `mutation_matrix_av` -- a runner is itself a check,
and a second copy is a second thing that can silently stop detecting (gotcha 13
extension 5). It carries BP's post-restore `__pycache__` purge (corrections-log
entry 122).

WHY THIS FILE EXISTS. `compare_manifests` gained a third category, and a third
category on a GUARD is exactly where a back door goes in. Before BT the only
way to record an artifact moving filesystems was to drop
`--assert-addition-only`, which removes the guard entirely; BT's branch is
narrower than that, and the mutations below are the ways it could quietly
become wider than intended.

THE THREE SHAPES THE MUTATIONS COVER:

  * **THE CONJUNCTION WEAKENS.** `set(moved) == {"path"}` becomes
    `"path" in moved`, or the branch stops looking at the moved set at all.
    Either turns "the path moved and nothing else did" into "the path moved,
    whatever else did" -- a rewrite laundered as a move. This is the one that
    matters and it is covered twice.
  * **A RELOCATION STOPS BEING VISIBLE.** The relocated entry loses the
    sha256 it exists to display, or relocations are folded back into
    `unchanged` so nothing is reported at all. A move nobody can see is not a
    recorded move.
  * **THE VERDICT STOPS DISCRIMINATING.** A relocation is reported with the
    same word as no change, or `CONTENT_FIELDS` is derived so loosely that a
    field added to `BASELINE_FIELDS` never enters the conjunction.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run  # noqa: E402
from tests.mutation_matrix_av import run_one  # noqa: E402

ROOT = os.getcwd()
C = os.path.join(ROOT, "src", "analysis", "canary.py")
T = "tests/test_canary.py"

REL = f"{T}::test_a_pure_path_move_is_classified_relocated_with_the_hash_unchanged"
DISC = f"{T}::test_a_path_AND_content_change_is_a_REBASELINE_not_a_relocation"
BYTES = f"{T}::test_a_move_that_also_changed_BYTES_is_refused"
SAME = f"{T}::test_a_content_change_at_the_SAME_path_is_still_a_rebaseline"
NOCH = f"{T}::test_no_change_at_all_is_still_NO_CHANGE"
BOTH = f"{T}::test_addition_and_relocation_together_get_their_own_verdict"
FIELDS = f"{T}::test_CONTENT_FIELDS_is_BASELINE_FIELDS_minus_path"
CLI_REFUSE = (f"{T}::test_assert_addition_only_REFUSES_a_relocation_without"
              "_allow_relocation")
CLI_ACCEPT = (f"{T}::test_allow_relocation_accepts_the_move_and_PRINTS_the"
              "_unchanged_hash")
CLI_DISC = f"{T}::test_allow_relocation_does_NOT_rescue_a_content_change"

# THE CONTROL: every pre-existing target is classified relocated regardless of
# what moved. It cannot fail to break the suite -- the discriminating fixture
# asserts that a sha256 change reaches `rebaselined`. If the runner does not
# detect this, no SURVIVED verdict below would mean anything, so none is
# printed.
CONTROL = (
    "CONTROL -- every moved target is classified relocated",
    C,
    b'        if set(moved) == {"path"}:\n',
    b'        if True:\n',
    [DISC, SAME, BYTES],
)

MUTATIONS = [
    # ============ THE CONJUNCTION WEAKENS
    ("the conjunction becomes membership: `\"path\" in moved` -- a rewrite "
     "that also moved is laundered as a relocation", C,
     b'        if set(moved) == {"path"}:',
     b'        if "path" in moved:',
     [DISC, BYTES]),

    ("the branch ignores sha256 by testing only that path moved and bytes "
     "held", C,
     b'        if set(moved) == {"path"}:',
     b'        if "path" in moved and "bytes" not in moved:',
     [DISC]),

    ("the branch ignores bytes by testing only path and sha256", C,
     b'        if set(moved) == {"path"}:',
     b'        if "path" in moved and "sha256" not in moved:',
     [BYTES]),

    # ============ A RELOCATION STOPS BEING VISIBLE
    ("a relocation is folded into `unchanged` -- the move happens and nothing "
     "reports it", C,
     b'        if set(moved) == {"path"}:\n'
     b'            relocated.append({',
     b'        if set(moved) == {"path"}:\n'
     b'            unchanged.append(name)\n'
     b'            _unused = ({',
     [REL, BOTH]),

    ("the relocated entry stops carrying the sha256 it exists to display", C,
     b'                "sha256_unchanged": prev[name].get("sha256"),',
     b'                "sha256_unchanged": None,',
     [REL]),

    ("the relocated entry loses the path it moved FROM, so the move cannot be "
     "audited", C,
     b'                "was": prev[name].get("path"), "now": t.get("path"),',
     b'                "was": t.get("path"), "now": t.get("path"),',
     [REL]),

    # ============ THE VERDICT STOPS DISCRIMINATING
    ("a relocation reports the same verdict as no change at all", C,
     b'    elif relocated:\n        verdict = "RELOCATION_ONLY"',
     b'    elif relocated:\n        verdict = "NO_CHANGE"',
     [REL]),

    ("an addition alongside a relocation is reported as a plain addition", C,
     b'    elif added and relocated:\n        verdict = "ADDITION_AND_RELOCATION"',
     b'    elif added and relocated:\n        verdict = "ADDITION_ONLY"',
     [BOTH]),

    ("a rebaseline stops producing a problem, so the verdict branch never "
     "reaches NOT_AN_ADDITION_ONLY_CHANGE", C,
     b'    for r in rebaselined:\n'
     b'        problems.append(f"{r[\'name\']}: baseline moved on "',
     b'    for r in []:\n'
     b'        problems.append(f"{r[\'name\']}: baseline moved on "',
     [DISC, SAME, BYTES, CLI_DISC]),

    ("CONTENT_FIELDS silently drops sha256, so a field the conjunction needs "
     "is not in it", C,
     b'CONTENT_FIELDS = tuple(f for f in BASELINE_FIELDS if f != "path")',
     b'CONTENT_FIELDS = tuple(f for f in BASELINE_FIELDS\n'
     b'                       if f not in ("path", "sha256"))',
     [FIELDS]),

    ("CONTENT_FIELDS is hardcoded rather than derived, so it can drift from "
     "BASELINE_FIELDS", C,
     b'CONTENT_FIELDS = tuple(f for f in BASELINE_FIELDS if f != "path")',
     b'CONTENT_FIELDS = ("arm", "hash_scope", "sha256")',
     [FIELDS]),

    ("NO_CHANGE stops being reachable -- the total branch loses its base case",
     C,
     b'    else:\n        verdict = "NO_CHANGE"',
     b'    else:\n        verdict = "RELOCATION_ONLY"',
     [NOCH]),

    # ============ THE CLI GUARD STOPS REQUIRING THE DECLARATION
    ("--assert-addition-only silently accepts a relocation without "
     "--allow-relocation", C,
     b'            if delta.get("relocated") and not args.allow_relocation:',
     b'            if False:',
     [CLI_REFUSE]),

    ("--allow-relocation is treated as always set, so the flag stops meaning "
     "anything", C,
     b'            if delta.get("relocated") and not args.allow_relocation:',
     b'            if delta.get("relocated") and not True:',
     [CLI_REFUSE]),

    ("the CLI stops printing the unchanged hash beside the move, so the "
     "reader has to take it on trust", C,
     b'            print(f"              sha256 UNCHANGED '
     b'{r[\'sha256_unchanged\']}  "',
     b'            print(f"              sha256 (withheld)  "',
     [CLI_ACCEPT]),
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
        print(f"CONTROL ABORT: anchor for {label!r} did not match exactly "
              "once. The runner cannot demonstrate that it detects anything.")
        return 2
    if survived:
        print(f"CONTROL ABORT: {label} was NOT detected (survivors: "
              f"{survived}). Every SURVIVED verdict this runner could print "
              "would be meaningless, so none is printed.")
        return 2
    print(f"control: DETECTED ({label})")
    print()

    print(f"{'mutation':<70} {'result':<10} killed")
    bad = 0
    for label, path, old, new, expect in MUTATIONS:
        survived = apply_and_run(path, old, new, expect)
        if survived is None:
            print(f"{label:<70} {'SKIP':<10} anchor did not match exactly once")
            bad += 1
            continue
        status = "KILLED" if not survived else "SURVIVED"
        print(f"{label:<70} {status:<10} "
              f"{len(expect) - len(survived)}/{len(expect)}")
        for t in survived:
            print(f"{'':<70} {'':<10} SURVIVOR: {t}")
        bad += bool(survived)

    print()
    if bad:
        print(f"{bad} MUTATION(S) NOT FULLY KILLED -- real coverage gap. Each "
              "must be classified code-defect or test-defect.")
    else:
        print(f"ALL {len(MUTATIONS)} MUTATIONS KILLED THEIR NAMED FIXTURES, "
              "control detected.")
        print("A clean first matrix is itself a claim about the matrix "
              "(criterion 3) -- scrutiny, not celebration.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
