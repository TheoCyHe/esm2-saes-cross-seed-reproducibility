"""Mutation matrix for amendment CS, CS3 -- the `rule_interior_mass.py`
parameterisation. NOT collected by pytest.

Run manually:  python tests/mutation_matrix_cs.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

The runner is `mutation_matrix_av.apply_and_run` (control-first, `__pycache__`
purged after every restore, byte-compared restore, verdicts on pytest's exit
code). THE CONTROL is pre-existing: `three_way_split`'s refusal of a count above
`n_compare` is removed and `test_three_way_split_refuses_a_count_above_n_compare`
must go red, so the runner is shown to detect before CS's own mutant is judged.

THE MUTANT is the brief's own requirement (CS3): "the pre-parameterisation
constant hard-coded back in must be detected". `pool_geometry` returns
`len(anchors) - 1`; the mutant returns the record's 19 regardless of the list,
which is exactly the module as it stood before CS on any pool but the record's.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run  # noqa: E402

ROOT = os.getcwd()
RIM = os.path.join(ROOT, "src", "analysis", "rule_interior_mass.py")
T = "tests/test_rule_interior_mass.py"

CONTROL = (
    "CONTROL -- three_way_split's out-of-range refusal removed",
    RIM,
    b'    if counts.max() > n_compare or counts.min() < 0:\n',
    b'    if False and (counts.max() > n_compare or counts.min() < 0):\n',
    [f"{T}::test_three_way_split_refuses_a_count_above_n_compare"],
)

MUTATIONS = [
    ("CS3  n_compare hard-coded back to the record's 19", RIM,
     b'    return len(anchors), len(anchors) - 1\n',
     b'    return len(anchors), N_COMPARE_RECORD\n',
     [f"{T}::test_pool_geometry_derives_n_compare_from_the_anchor_list"]),
    ("CS3  record controls applied to every pool", RIM,
     b'    return [int(a) for a in anchors] == DEFAULT_ANCHORS and same_root\n',
     b'    return True\n',
     [f"{T}::test_record_controls_apply_only_to_the_records_twenty_anchors"]),
]


def main() -> int:
    label, path, old, new, expect = CONTROL
    survived = apply_and_run(path, old, new, expect)
    if survived is None:
        print(f"CONTROL ABORT: anchor for {label} did not match exactly once.")
        return 2
    if survived:
        print(f"CONTROL ABORT: {label} was NOT detected (survivors: {survived}).")
        return 2
    print(f"CONTROL detected ({len(expect)} fixtures) -- the runner detects.\n")
    n_bad = 0
    for label, path, old, new, expect in MUTATIONS:
        survived = apply_and_run(path, old, new, expect)
        if survived is None:
            print(f"ANCHOR MISS  {label}"); n_bad += 1; continue
        if survived:
            print(f"SURVIVED     {label}  -> {survived}"); n_bad += len(survived)
        else:
            print(f"KILLED       {label}  ({len(expect)} fixtures)")
    print(f"\n{len(MUTATIONS) if n_bad == 0 else '?'} of {len(MUTATIONS)} mutations killed; "
          f"survivors/misses: {n_bad}")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
