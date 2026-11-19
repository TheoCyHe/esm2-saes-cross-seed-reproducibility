"""Mutation matrix for `decomposition_reporting` -- amendment BX. NOT collected
by pytest.

Run manually:  python tests/mutation_matrix_bx.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

The runner is reused from `mutation_matrix_av` -- control-first, `__pycache__`
purged between invocations, byte-compared restore, decisions on pytest's EXIT
CODE. A second copy of a runner is a second thing that can silently stop
detecting (gotcha 13 extension 5).

WHAT THIS FILE EXISTS FOR. This module computes no new quantity: it re-cuts one
stored artifact along denominators and detection rates. That makes its
characteristic defect a QUIET one -- a wrong denominator, a population mask that
does not restrict, a cell label swapped -- each of which yields a complete,
plausible table with no error anywhere. Exactly the shape this project's record
says it is bad at catching by reading.

THE MUTATIONS ARE CHOSEN AGAINST THAT: every one of them leaves a table that
still sums correctly and still looks like a reporting grid.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run  # noqa: E402
from tests.mutation_matrix_av import run_one  # noqa: E402

ROOT = os.getcwd()
M = os.path.join(ROOT, "src", "analysis", "decomposition_reporting.py")
T = "tests/test_decomposition_reporting.py"

# THE CONTROL: `two_by_two` returns a fixed dict before it counts anything. It
# is not a no-op and not a message edit -- several fixtures assert DIFFERENT
# four-cell counts from it, so no constant can satisfy them all.
CONTROL = (
    "CONTROL -- two_by_two returns a constant before counting", M,
    b'    counts = {name: int(((across_detected == a) & (within_detected == w)).sum())',
    b'    return {n: 0 for n in CELL_NAMES.values()}\n'
    b'    counts = {name: int(((across_detected == a) & (within_detected == w)).sum())',
    [f"{T}::test_two_by_two_counts_by_hand",
     f"{T}::test_concept_grid_at_t_equals_one_all_concepts",
     f"{T}::test_cell_grid_all_cells_by_hand"],
)

MUTATIONS = [
    # ==================================================== 1. THE ZERO'S KIND
    # A structural zero reported as a measured one. The table is unchanged --
    # only the label moves -- and the label is the entire difference between
    # "the within-only cell is empty here" being a finding and being
    # arithmetic. This is the defect the module exists to prevent.
    ("a STRUCTURAL within_only zero relabelled MEASURED (concept grid)", M,
     b'                    (WITHIN_ONLY_STRUCTURAL if structural\n'
     b'                     else WITHIN_ONLY_MEASURED)),',
     b'                    WITHIN_ONLY_MEASURED),',
     [f"{T}::test_restricted_population_zero_within_only_is_labelled_structural"]),

    # ================================================ 2. THE RESTRICTION FAILS
    # `across_D_eq_1` restricted with `>=` on the rate rather than `== 1`. At
    # t = 1 the two coincide, so the headline row is unchanged and only the
    # other rates move -- a defect visible nowhere a reader is likely to look.
    ("across_D_eq_1 population defined by >= t instead of == 1", M,
     b'            "across_D_eq_1": d_across == 1.0,\n'
     b'            "across_D_ge_t": d_across >= t,',
     b'            "across_D_eq_1": d_across >= t,\n'
     b'            "across_D_ge_t": d_across >= t,',
     [f"{T}::test_concept_grid_restricted_population_shares_by_hand"]),

    # ================================================= 3. THE DENOMINATOR SWAP
    # The protein-level share divided by the population instead of by the
    # across-detected count. Every value stays in [0, 1] and the direction of
    # the shape is preserved, so nothing looks wrong.
    ("share divided by the population, not by across-detected", M,
     b'                    float(cells["protein_level_candidate"]\n'
     b'                          / (cells["protein_level_candidate"]\n'
     b'                             + cells["residue_level_candidate"]))',
     b'                    float(cells["protein_level_candidate"] / n)',
     [f"{T}::test_concept_grid_at_t_equals_one_all_concepts"]),

    # ================================================= 4. THE UNIT COLLAPSES
    # The bridge's concept side counts concepts that fail in EVERY seed rather
    # than in at least one. It still returns a share on the same population, so
    # the bridge still "works" -- and it silently equates the two units, which
    # is the confusion the bridge exists to resolve.
    ("bridge's concept side uses .all() over seeds instead of .any()", M,
     b'    n_conc_d1_any_fail = int((fail & cells_d1).any(axis=0)[conc_d1].sum())',
     b'    n_conc_d1_any_fail = int((fail & cells_d1).all(axis=0)[conc_d1].sum())',
     [f"{T}::test_bridge_separates_unit_from_population"]),

    # ============================================== 5. THE EMPTY BAND VANISHES
    # Empty bands dropped instead of reported. The remaining table is correct;
    # what is lost is the ability to distinguish a band with no concepts from a
    # band that was never computed (gotcha 28).
    ("empty bands omitted from the common-basis diff", M,
     b'        bands.append({\n'
     b'            "D_across": float(t),',
     b'        if not n_conc:\n'
     b'            continue\n'
     b'        bands.append({\n'
     b'            "D_across": float(t),',
     [f"{T}::test_common_basis_diff_reports_every_band_including_empty"]),

    # ============================================ 6. THE VERDICT LOSES A BRANCH
    # `NEITHER` folded into `SUPPORTS`. The verdict space collapses to a two-way
    # disjunction, which is the exact failure the brief's guard names -- and it
    # fires in the direction that manufactures agreement with the inference.
    ("inference_check's NEITHER branch folded into SUPPORTS", M,
     b'    else:\n'
     b'        verdict = "NEITHER"',
     b'    else:\n'
     b'        verdict = "SUPPORTS"',
     [f"{T}::test_inference_check_neither_when_top_sits_inside_the_range"]),

    # ============================================ 7. THE ARTIFACT GUARD OPENS
    # The rate/matrix consistency check comparing an arm against ITSELF. It can
    # then only pass (gotcha 25a), while still reading as a guard.
    ("the stored-rate consistency check compares an arm against itself", M,
     b'        recomputed = out[f"det_{arm}"].mean(axis=0)\n'
     b'        if not np.allclose(recomputed, out[f"d_{arm}"]):',
     b'        recomputed = out[f"det_{arm}"].mean(axis=0)\n'
     b'        if not np.allclose(recomputed, recomputed):',
     [f"{T}::test_load_refuses_when_a_stored_rate_disagrees_with_its_matrix"]),

    # ============================================= 8. THE RATE GRID OFF BY ONE
    # `rate_grid` including 0, so the first row's predicate is satisfied by
    # every concept and the 2x2 collapses to a single occupied cell -- a row
    # that ranks nothing while looking like the most inclusive summary.
    ("rate_grid includes the constant-true threshold 0", M,
     b'    return np.arange(1, n_seeds + 1, dtype=float) / n_seeds',
     b'    return np.arange(0, n_seeds + 1, dtype=float) / n_seeds',
     [f"{T}::test_rate_grid_excludes_zero_and_includes_one",
      f"{T}::test_concept_grid_covers_every_population_and_rate"]),
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
        print(f"CONTROL ABORT: anchor for {label!r} did not match exactly once.")
        return 2
    if survived:
        print(f"CONTROL ABORT: {label} was NOT detected (survivors: {survived}). "
              "Every SURVIVED verdict this runner could print would be "
              "meaningless, so none is printed.")
        return 2
    print(f"CONTROL detected ({len(expect)} fixtures) -- the runner detects.\n")

    n_surv = 0
    for label, path, old, new, expect in MUTATIONS:
        survived = apply_and_run(path, old, new, expect)
        if survived is None:
            print(f"ANCHOR MISS  {label}")
            n_surv += 1
            continue
        if survived:
            print(f"SURVIVED     {label}  -> {survived}")
            n_surv += len(survived)
        else:
            print(f"KILLED       {label}  ({len(expect)} fixtures)")
    print(f"\n{len(MUTATIONS) - n_surv if n_surv == 0 else '?'} "
          f"of {len(MUTATIONS)} mutations killed; survivors/misses: {n_surv}")
    return 0 if n_surv == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
