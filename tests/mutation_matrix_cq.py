"""Mutation matrix for amendment CQ -- the assertion-only extensions and the
AK8 smooth-arm alignment. NOT collected by pytest.

Run manually:  python tests/mutation_matrix_cq.py
Exit 0 iff the CONTROL is detected and every mutation kills its named fixtures.

The runner is `mutation_matrix_av.apply_and_run` -- control-first, `__pycache__`
purged after every restore, byte-compared restore, decisions on pytest's EXIT
CODE (gotcha 13 extension 5). CRLF working copies are handled by the runner.

WHAT THIS FILE EXISTS FOR. Every change in CQ is an ASSERTION or a RECORD, and
the characteristic defect of an assertion is that it can only pass (gotcha 25):
a check whose failure branch is unreachable looks identical, from the outside,
to a check that works. So each assertion gets one mutant that removes or
inverts it, and each mutant names the fixture that must go red -- a fixture
that stays green under the mutant is a check that guards nothing.

THE CONTROL is not one of CQ's changes. It removes the alignment the
SENSITIVITY arm has had since AR9 (`design3_join.py`, the `M_smooth` line),
which the pre-existing `test_the_smooth_arm_is_aligned_too` must kill -- so
the runner is shown to detect a real misalignment before any verdict about
CQ's own alignment (the AK8 arm, the same array two screens later) is
printed. The two call sites are the whole story of corrections entry 218: one
was checked, one was not.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mutation_matrix_av import apply_and_run  # noqa: E402

ROOT = os.getcwd()
D3J = os.path.join(ROOT, "src", "analysis", "design3_join.py")
SG = os.path.join(ROOT, "src", "analysis", "standardize_gain.py")
RF = os.path.join(ROOT, "src", "analysis", "rf_marginal.py")
AI = os.path.join(ROOT, "src", "analysis", "artifact_identity.py")
RIC = os.path.join(ROOT, "src", "analysis", "row_index_cache.py")
VC = os.path.join(ROOT, "src", "embeddings", "verify_cache.py")

T_D3J = "tests/test_design3_join.py"
T_SG = "tests/test_standardize_gain.py"
T_RF = "tests/test_rf_marginal.py"
T_AI = "tests/test_artifact_identity.py"
T_RIC = "tests/test_row_index_cache.py"
T_VC = "tests/test_verify_cache.py"

CONTROL = (
    "CONTROL -- the SENSITIVITY arm's pre-existing smooth-M alignment removed",
    D3J,
    b'            M_smooth = align_to_reference(gts_order["labels"], hf["M"][:],\n'
    b'                                          anchor_labels)\n',
    b'            M_smooth = hf["M"][:]\n',
    [f"{T_D3J}::test_the_smooth_arm_is_aligned_too"],
)

MUTATIONS = [
    # ============================ 1. THE FIX (CQ1 / A1): AK8 alignment removed
    # This is the code as it stood in all seven stored runs (entry 218).
    ("A1  AK8 smooth arm: align_to_reference removed (the entry-218 code)", D3J,
     b'            M_smooth_ak8 = align_to_reference(gts_order["labels"], hf["M"][:],\n'
     b'                                              anchor_labels)\n',
     b'            M_smooth_ak8 = hf["M"][:]\n',
     [f"{T_D3J}::test_ak8_smooth_arm_is_aligned_to_the_anchor_order"]),

    # ======================================= 2. A2(i): AP order check removed
    ("A2i AP seed-order assertion disabled", SG,
     b'    if oth != ref:\n',
     b'    if False and oth != ref:\n',
     [f"{T_SG}::test_refuses_an_ap_matrix_whose_seed_order_differs"]),

    # ============================ 3. A2(i): absent labels treated as agreement
    ("A2i unlabelled AP axis accepted (absence read as agreement)", SG,
     b'    if other is None:\n        raise SystemExit(\n'
     b'            f"FATAL: {other_name} carries no seed labels, so its seed order "\n',
     b'    if other is None:\n        return\n    if False:\n        raise SystemExit(\n'
     b'            f"FATAL: {other_name} carries no seed labels, so its seed order "\n',
     [f"{T_SG}::test_refuses_an_unlabelled_ap_matrix",
      f"{T_SG}::test_refuses_an_unlabelled_mu_smooth"]),

    # ================================ 4. A2(iii): raw-null recovery check removed
    ("A2iii raw-null slab-is-row-s assertion disabled", SG,
     b'    if pos != s:\n',
     b'    if False and pos != s:\n',
     [f"{T_SG}::test_refuses_a_raw_null_whose_slabs_are_permuted"]),

    # ============================== 5. A2(iii): unrecoverable order papered over
    ("A2iii unrecoverable raw-null order swallowed", SG,
     b'    except SeedOrderUnrecoverable as e:\n        raise SystemExit(\n',
     b'    except SeedOrderUnrecoverable as e:\n        return\n        raise SystemExit(\n',
     [f"{T_SG}::test_refuses_a_raw_null_the_reductions_do_not_describe"]),

    # ============================================= 6. A2(iv): mu-smooth check removed
    ("A2iv mu_smooth seed-order check removed", SG,
     b'        if args.mu_smooth is not None:\n'
     b'            assert_seed_order_agrees("null_reductions.seeds", ref_labels,\n',
     b'        if False:\n'
     b'            assert_seed_order_agrees("null_reductions.seeds", ref_labels,\n',
     [f"{T_SG}::test_refuses_an_unlabelled_mu_smooth",
      f"{T_SG}::test_refuses_a_mu_smooth_in_a_different_seed_order"]),

    # ==================================== 7. A3: denominator check cannot fire
    ("A3  n_compare vs pool assertion disabled", RF,
     b'    if n_compare != expected:\n        raise SystemExit(\n'
     b'            f"PRE-FLIGHT FAIL: n_compare = len(decoders) - 1 = {n_compare}, "\n',
     b'    if False:\n        raise SystemExit(\n'
     b'            f"PRE-FLIGHT FAIL: n_compare = len(decoders) - 1 = {n_compare}, "\n',
     [f"{T_RF}::test_denominator_checker_refuses_a_mismatch"]),

    # ================================= 8. A3: written-value read-back disabled
    ("A3  n_compare_seeds read-back assertion disabled", RF,
     b'    if int(written) != int(n_compare):\n',
     b'    if False:\n',
     [f"{T_RF}::test_written_denominator_readback_refuses_a_mismatch"]),

    # ================================ 9. A4: eval-cache identity always agrees
    ("A4  eval-cache identity comparison replaced by True", AI,
     b'        ok = (k in actual) and (k in expected) and a == e\n',
     b'        ok = True\n',
     [f"{T_AI}::test_eval_cache_identity_refuses_a_layer_mismatch",
      f"{T_AI}::test_eval_cache_identity_absent_key_is_not_agreement",
      f"{T_RIC}::test_cold_build_refuses_an_eval_cache_that_mismatches_the_checkpoint"]),

    # ============================== 10. A4: the row-index-cache call site removed
    ("A4  row_index_cache no longer asserts before a cold gather", RIC,
     b'    eval_check = assert_eval_cache_matches_checkpoint(Path(eval_cache),\n'
     b'                                                      Path(ckpt_dir))\n',
     b'    eval_check = "AGREE"\n',
     [f"{T_RIC}::test_cold_build_refuses_an_eval_cache_that_mismatches_the_checkpoint",
      f"{T_RIC}::test_cold_build_without_a_meta_json_reports_the_check_as_not_done"]),

    # ================================== 11. A4: verify_cache value check removed
    ("A4  verify_cache max_len value check replaced by presence", VC,
     b'        check(_plain(attrs.get("max_len")) == args.max_len,\n',
     b'        check("max_len" in attrs,\n',
     [f"{T_VC}::test_verify_cache_fails_on_a_max_len_mismatch"]),

    # ======================================= 12. A5: device class not recorded
    ("A5  rf_marginal M1 identity drops device_class", RF,
     b'            "device_class": compute_device_class,           # CQ A5\n',
     b'',
     [f"{T_RF}::test_main_records_the_compute_device_and_the_denominator_check"]),

    ("A5  standardize_gain drops device_class", SG,
     b'            device_class=device_class("cpu"),\n'
     b'            git_sha=git_sha(), git_dirty=bool(git_dirty()))\n    print(f"wrote {args.out}")\n',
     b'            git_sha=git_sha(), git_dirty=bool(git_dirty()))\n    print(f"wrote {args.out}")\n',
     [f"{T_SG}::test_output_records_the_device_class"]),

    ("A5  design3_join identity drops device_class", D3J,
     b'            "device_class": device_class("cpu"),\n'
     b'            "git_sha": git_sha(), "git_dirty": git_dirty(),\n',
     b'            "git_sha": git_sha(), "git_dirty": git_dirty(),\n',
     [f"{T_D3J}::test_identity_records_the_device_class"]),
]

# NOT MUTATED, stated so the gap is visible rather than implied away (gotcha
# 28): the A4 call site in `feature_ap.main` and the A5 recorders in
# `cache.py`, `train.py` and `smooth_mu.py` have no entry-point fixture in the
# suite, so a mutant removing any of them would be killed by nothing. The
# FUNCTIONS they call are mutated above; the CALL SITES are exercised only by
# the cluster jobs that next write those artifacts.
UNMUTATED_CALL_SITES = ("feature_ap.main -> assert_eval_cache_identity",
                        "cache.py attrs device_class",
                        "train.py meta device_class",
                        "smooth_mu.py write_identity device_class")


def main() -> int:
    label, path, old, new, expect = CONTROL
    survived = apply_and_run(path, old, new, expect)
    if survived is None:
        print(f"CONTROL ABORT: anchor for {label} did not match exactly once. "
              "The runner cannot be shown to detect, so no verdict is printed.")
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
    print("unmutated call sites (no fixture exists):", *UNMUTATED_CALL_SITES,
          sep="\n  ")
    return 0 if n_surv == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
