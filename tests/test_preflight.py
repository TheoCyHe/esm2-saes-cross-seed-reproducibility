"""Cross-component pre-flight gate.

This is the guard that would have caught T6. Its whole value is that it FAILS
on real mismatches, so every check below is exercised against a deliberately
constructed violation of exactly the condition it claims to detect (gotcha 13),
and the clean case is asserted to pass so the gate is not vacuous.
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from src.analysis.artifact_identity import (
    AP_IMPL_SEGMENTED,
    AP_IMPL_SPARSE,
    expected_ap_impl,
    stratum_concept_axis,
)
from src.analysis.preflight import PreflightError, check

BASE = dict(truncation_policy="untruncated",
            retention_floor_mode="interplm", retention_min_residues=1500,
            retention_min_instances=10, min_protein_count=5)

# What a LABEL STORE publishes that its consumers do not: the SHA of the code
# that built it lives under `git_sha`, and consumers copy it into
# `library_sha`. The two key names are why the gate's `library_sha` check had
# to become "derived" -- as an equality check against the store it could never
# fire, because no store writes that key.
LABEL_EXTRA = dict(git_sha="9bfa94b", n_rows=1000)

# What a CONSUMING artifact publishes that the store cannot: `scored_row_set` is a
# property of the RUN, so the store has no value to be compared against.
ARTIFACT_EXTRA = dict(scored_row_set="label_store", n_rows=1000,
                      library_sha="9bfa94b")

CONCEPTS = ["ACT_SITE", "BINDING", "BINDING:ATP", "DOMAIN", "DOMAIN:SH2"]
# BINDING and DOMAIN are pooled parents (each has a sub-typed child), so the
# argmax pool is the other three -- amendment I4.
POOL_N = 3


def make_labels(path, concepts=CONCEPTS, **over):
    with h5py.File(path, "w") as hf:
        g = hf.create_group("concepts")
        for c in concepts:
            g.create_group(c).create_dataset("all", data=np.arange(200))
        for k, v in {**BASE, **LABEL_EXTRA,
                     "n_concepts": len(concepts), **over}.items():
            hf.attrs[k] = v
    return path


def make_ap(path, concepts=CONCEPTS, ap_impl=AP_IMPL_SPARSE, **over):
    with h5py.File(path, "w") as hf:
        g = hf.create_group("pooled")
        g.create_dataset("concepts", data=np.array(concepts,
                                                   dtype=h5py.string_dtype()))
        g.create_dataset("ap", data=np.zeros((2, 4, len(concepts)), np.float32))
        for k, v in {**BASE, **ARTIFACT_EXTRA, "n_concepts": len(concepts),
                     "argmax_pool_n": POOL_N, "ap_impl": ap_impl,
                     **over}.items():
            hf.attrs[k] = v
    return path


def make_null(path, n_concepts=len(CONCEPTS), **over):
    with h5py.File(path, "w") as hf:
        hf.create_dataset("null_g_a", data=np.zeros((2, 2, 4)))
        for k, v in {**BASE, **ARTIFACT_EXTRA, "n_concepts": n_concepts,
                     "argmax_pool_n": POOL_N,
                     "exclude_pooled_parents": True, **over}.items():
            hf.attrs[k] = v
    return path


# --------------------------------------------------------- the clean case --

def test_clean_inputs_pass(tmp_path):
    """Not vacuous: the gate must let a genuinely consistent set through."""
    s = check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5"),
              (make_ap(tmp_path / "ap.h5"),),
              require_clean_git=False)
    assert s["n_concepts_label_store"] == len(CONCEPTS)
    # BINDING and DOMAIN are pooled parents here; ACT_SITE is not
    assert s["pooled_parents_excluded"] == 2
    assert s["argmax_pool_n"] == len(CONCEPTS) - 2


# ------------------------------------------- 1. concept-count disagreement --

def test_null_concept_count_mismatch_is_caught(tmp_path):
    """THE T6 DEFECT ITSELF: the null maxed over 716 concepts while the AP
    path produced 611. A max over a superset is stochastically larger, so the
    null is biased upward and alignment is under-called, invisibly."""
    with pytest.raises(PreflightError, match=r"n_concepts=\S+ but label store says"):
        check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5", n_concepts=611),
              require_clean_git=False)


def test_declared_count_is_checked_against_the_store_CONTENT_not_its_attr(tmp_path):
    """Targets the dedicated concept-count branch specifically.

    Written after a mutation test showed the previous test passed for the
    WRONG REASON: deleting the dedicated branch entirely left it green,
    because `n_concepts` is also in IDENTITY_KEYS and the generic comparison
    fired instead. The two checks are not redundant -- the generic one
    compares the store's DECLARED attr against the artifact's, while this one
    compares the artifact's declaration against the store's ACTUAL concept
    count. Only the latter catches a store whose own attr is wrong.

    Constructed so the generic path cannot fire: the store declares no
    `n_concepts` at all, so there is nothing for it to compare.
    """
    lab = tmp_path / "lab.h5"
    make_labels(lab)                       # 5 real concept groups
    with h5py.File(lab, "a") as hf:
        del hf.attrs["n_concepts"]         # generic comparison now skips it
    with pytest.raises(PreflightError, match="but the label store has"):
        check(lab, make_null(tmp_path / "null.h5", n_concepts=611),
              require_clean_git=False)


def test_ap_concept_axis_mismatch_is_caught(tmp_path):
    """Same count, different members -- caught by comparing the axis, not just
    its length."""
    other = ["ACT_SITE", "BINDING", "BINDING:ATP", "DOMAIN", "REGION"]
    # The MESSAGE changed when the check became a RELATIONSHIP rather than
    # equality (2026-08-11). The CONDITION this test names is unchanged: an
    # unstratified artifact is still held to the store's FULL axis.
    with pytest.raises(PreflightError, match="concept AXIS is not the store"):
        check(make_labels(tmp_path / "lab.h5"),
              None, (make_ap(tmp_path / "ap.h5", concepts=other),),
              require_clean_git=False)


# ----------------------------------------------- 2. population disagreement --

def test_population_mismatch_is_caught(tmp_path):
    """The --labels defect: a store that EXISTS but describes with_truncated
    while the run intends untruncated."""
    with pytest.raises(PreflightError, match=r"truncation_policy=\S+ but label store says"):
        check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5", truncation_policy="with_truncated"),
              require_clean_git=False)


# --------------------------------------------- 3. floor-parameter disagreement --

@pytest.mark.parametrize("key,bad", [
    ("retention_floor_mode", "legacy"),
    ("retention_min_residues", 100),
    ("retention_min_instances", 3),
    ("min_protein_count", 1),
])
def test_each_floor_parameter_mismatch_is_caught(tmp_path, key, bad):
    with pytest.raises(PreflightError,
                       match=rf"{key}=\S+ but label store says"):
        check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5", **{key: bad}),
              require_clean_git=False)


# --------------------------- 3b. the POPULATION OVERLOADING (scored_row_set) --
#
# `population` and `scored_row_set` are different axes that share the word
# `label_store` as one of `scored_row_set`'s three values. Every test here is
# constructed so that `population` AGREES on every input -- which is the
# realistic case, since both artifacts copy it from the same store -- so a
# pass can only come from a check that looks at `scored_row_set` itself.

def test_scored_row_set_mismatch_is_refused_while_population_agrees(tmp_path):
    """THE DEFECT: `--scored-row-set all` against the untruncated store scores
    17,739,363 cache rows against a 15,972,325-row store, so every prevalence
    and therefore every normalised gain is wrong -- while `population` reads
    `untruncated` on every artifact, because that is a property of the store
    they all share.

    ISOLATED to the expected-value branch, after a mutation showed the obvious
    construction was double-covered. Both artifacts declare `all`, so they
    agree with each other and the consistency branch cannot fire; both carry
    the store's own row count, so the `n_rows` branch cannot fire either. The
    only check left that can refuse this is the one comparing `scored_row_set`
    against what the RUN declares -- which is the branch this fixture names.
    """
    with pytest.raises(PreflightError,
                       match=r"scored_row_set='full_eval' but this run declares"):
        check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5", scored_row_set="full_eval"),
              (make_ap(tmp_path / "ap.h5", scored_row_set="full_eval"),),
              require_clean_git=False)


def test_scored_row_set_disagreement_between_artifacts_is_refused(tmp_path):
    """Independent of the declared expectation: an AP matrix and a null scored
    over different rows are incomparable even if neither is `population`."""
    with pytest.raises(PreflightError, match="disagree on `scored_row_set`"):
        check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5", scored_row_set="shared_experimental"),
              (make_ap(tmp_path / "ap.h5", scored_row_set="all"),),
              expect_scored_row_set=None, require_clean_git=False)


def test_absent_scored_row_set_is_refused_not_assumed(tmp_path):
    """I3: absent is not matching. An artifact that cannot say which rows it
    scored cannot be asserted against one that can."""
    p = tmp_path / "null.h5"
    make_null(p)
    with h5py.File(p, "a") as hf:
        del hf.attrs["scored_row_set"]
    with pytest.raises(PreflightError, match="ABSENT IS NOT MATCHING"):
        check(make_labels(tmp_path / "lab.h5"), p, (),
              require_clean_git=False)


def test_n_rows_mismatch_against_the_store_is_refused(tmp_path):
    """The CONSEQUENCE, checked independently of the cause. An artifact that
    declares `scored_row_set=label_store` but scored a different row count is
    refused on the arithmetic, without relying on its own declaration."""
    with pytest.raises(PreflightError, match=r"n_rows=\S+ but the label store implies"):
        check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5", n_rows=17739363),
              require_clean_git=False)


def test_n_rows_store_check_is_reported_not_applicable_off_population(tmp_path):
    """Under a non-`population` scored-row-set the store's row count is NOT the
    right expectation. It is reported as not-applicable rather than silently
    passed -- the I3 rule applied to the expectation side."""
    s = check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5",
                        scored_row_set="shared_experimental", n_rows=191),
              (make_ap(tmp_path / "ap.h5",
                       scored_row_set="shared_experimental", n_rows=191),),
              expect_scored_row_set="shared_experimental",
              require_clean_git=False)
    assert any("n_rows" in m and "NOT APPLICABLE" in m
               for m in s["not_applicable"]), s["not_applicable"]


# ----------------------------------- 3c. library_sha, as a DERIVED check --

def test_library_sha_mismatch_is_refused(tmp_path):
    """This check was DEAD for the gate's whole life. `library_sha` was an
    `equal_to_labels` key, and no label store writes it -- the store publishes
    the building code's SHA as `git_sha`. So the branch existed, read as
    coverage, and could not fire; both compliant artifacts on disk carried the
    PREVIOUS store's SHA while the gate printed PASSED."""
    # The regex matches the VALUE-COMPARISON message specifically, not merely
    # the key name. Mutation testing showed the looser `match="library_sha"`
    # was satisfied by the SCHEMA rule instead: reverting the key to
    # `equal_to_labels` makes `unwritable_equality_keys()` non-empty, and that
    # complaint also contains the substring. A fixture any message mentioning
    # the key can satisfy is not testing the check it names -- the same trap
    # `argmax_pool_n` fell into.
    with pytest.raises(PreflightError,
                       match=r"library_sha=\S+ but the label store implies"):
        check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5", library_sha="deadbeef"),
              require_clean_git=False)


# -------------------------------------------------- 4. AP version marker --

def test_wrong_ap_impl_is_caught(tmp_path):
    """The restructure was built, validated, and never wired in. A version
    marker makes 'which implementation produced this' assertable."""
    # `but the validated` became `but the expected` when the expectation became
    # DERIVED from whether masking is active. Same condition, same artifact.
    with pytest.raises(PreflightError, match=r"ap_impl=.+ but the expected"):
        check(make_labels(tmp_path / "lab.h5"), None,
              (make_ap(tmp_path / "ap.h5", ap_impl=AP_IMPL_SEGMENTED),),
              expect_ap_impl=AP_IMPL_SPARSE, require_clean_git=False)


def test_missing_ap_impl_is_caught(tmp_path):
    p = tmp_path / "ap.h5"
    make_ap(p)
    with h5py.File(p, "a") as hf:
        del hf.attrs["ap_impl"]
    with pytest.raises(PreflightError, match="no `ap_impl` marker"):
        check(make_labels(tmp_path / "lab.h5"), None, (p,),
              require_clean_git=False)


# ------------------------------------------------------ absent identity ----

def test_absent_identity_keys_are_reported_not_ignored(tmp_path):
    """null_pilot_g.h5 carried no population/scored_row_set at all. 'Absent' must
    never read as 'matching'."""
    p = tmp_path / "null.h5"
    make_null(p)
    with h5py.File(p, "a") as hf:
        del hf.attrs["truncation_policy"]
        del hf.attrs["min_protein_count"]
    with pytest.raises(PreflightError, match="identity keys ABSENT"):
        check(make_labels(tmp_path / "lab.h5"), p, require_clean_git=False)


# ------------------------------------------------------- 5. git dirtiness --

def test_dirty_git_is_caught_and_reports_the_reason(tmp_path, monkeypatch):
    """git_dirty counting UNTRACKED files is CORRECT -- an untracked .py can
    shadow an import -- but a bare boolean is unactionable, so the porcelain
    output itself must appear in the message."""
    monkeypatch.setattr("src.analysis.preflight.git_porcelain",
                        lambda *a, **k: "?? slurm/investigate_null_tail_adhoc.sbatch\n")
    with pytest.raises(PreflightError) as e:
        check(make_labels(tmp_path / "lab.h5"), require_clean_git=True)
    assert "investigate_null_tail_adhoc.sbatch" in str(e.value)


def test_clean_git_passes(tmp_path, monkeypatch):
    monkeypatch.setattr("src.analysis.preflight.git_porcelain", lambda *a, **k: "")
    check(make_labels(tmp_path / "lab.h5"), require_clean_git=True)


# ------------------------------------------------------ existence vs identity --

def test_a_file_that_exists_but_is_the_wrong_one_still_fails(tmp_path):
    """The generalization of the --labels defect: presence is not identity. A
    guard satisfied by the wrong artifact is worse than no guard, because it
    converts a loud failure into a silent one."""
    stale = make_labels(tmp_path / "stale.h5",
                        concepts=["DOMAIN", "DOMAIN:SH2"],
                        truncation_policy="with_truncated", library_sha="f3e2ddb")
    assert stale.exists()                      # the old `[[ ! -f ]]` guard passes
    with pytest.raises(PreflightError):
        check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5", truncation_policy="with_truncated",
                        library_sha="f3e2ddb", n_concepts=2),
              require_clean_git=False)


# ------------------------------------------ argmax pool (amendment I4/L3) --
# THE GAP THIS CLOSES. `null_pilot_g.h5` maxed over all 716 concepts
# including the 9 pooled parents; `feature_ap_main.h5` defines a 707-concept
# pool. Both carry n_concepts=716 -- the LIBRARY size -- so the concept-count
# branch passes and the gate let the mismatch through. Verified by running
# the real gate against the real artifacts on 2026-07-28: it printed
# PRE-FLIGHT PASSED and exited 0 on exactly the pair amendment I4 says is
# incompatible. These tests are the regression guard.

def test_null_argmax_pool_mismatch_is_caught(tmp_path):
    """The planted defect: a null that maxed over the full library while the
    label store implies a 707-style reduced pool."""
    with pytest.raises(PreflightError,
                       match=r"argmax_pool_n=\S+ but the label store implies"):
        check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5", argmax_pool_n=len(CONCEPTS)),
              require_clean_git=False)


def test_pool_mismatch_survives_a_matching_concept_count(tmp_path):
    """MUTATION GUARD (4b). The previous test could pass for the wrong
    reason -- via the n_concepts branch -- if a mismatched pool also implied
    a mismatched library size. Here n_concepts is CORRECT and only the pool
    is wrong, so deleting the argmax-pool branch must break this test and no
    other branch can rescue it."""
    with pytest.raises(PreflightError,
                       match=r"argmax_pool_n=\S+ but the label store implies"):
        check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5",
                        n_concepts=len(CONCEPTS),      # correct
                        argmax_pool_n=len(CONCEPTS)),  # wrong
              require_clean_git=False)


def test_absent_argmax_pool_is_reported_ABSENT_not_matching(tmp_path):
    """I3: an absent key is never treated as agreement. An artifact that
    cannot state which pool its max ranged over cannot be asserted against
    one that can -- this is how `null_pilot_g.h5`'s missing `population`
    attr hid a mismatch before."""
    path = make_null(tmp_path / "null.h5")
    with h5py.File(path, "a") as hf:
        del hf.attrs["argmax_pool_n"]
    with pytest.raises(PreflightError, match="ABSENT IS NOT MATCHING"):
        check(make_labels(tmp_path / "lab.h5"), path, require_clean_git=False)


def test_null_that_kept_pooled_parents_is_caught(tmp_path):
    """`exclude_pooled_parents=False` is independently disqualifying: the
    pooled parents cannot win `g_f`'s argmax, so a null that let them win is
    a null for a different statistic."""
    with pytest.raises(PreflightError, match=r"exclude_pooled_parents=False"):
        check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5", exclude_pooled_parents=False),
              require_clean_git=False)


def test_null_without_exclude_pooled_parents_is_reported_absent(tmp_path):
    """CT8 item 2: the key is REQUIRED on a null. Absent is reported as
    absent -- it used to pass silently, which gotcha 15 forbids."""
    path = make_null(tmp_path / "null.h5")
    with h5py.File(path, "a") as hf:
        del hf.attrs["exclude_pooled_parents"]
    with pytest.raises(PreflightError, match=r"ABSENT: \['exclude_pooled_parents'\]"):
        check(make_labels(tmp_path / "lab.h5"), path, require_clean_git=False)


def test_self_declared_supersession_is_refused(tmp_path):
    """`null_pilot_g.h5` carries `superseded_by`. An artifact that declares
    itself superseded must never be consumed, regardless of whether its
    other identity keys happen to line up."""
    with pytest.raises(PreflightError, match=r"carries `superseded_by`"):
        check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5",
                        superseded_by="amendment I4 (argmax pool 707)"),
              require_clean_git=False)


def test_n_concepts_means_LIBRARY_not_pool(tmp_path):
    """Pins the identity convention that `null_pilot.py` violated.

    `n_concepts` is the LIBRARY size in every artifact -- the label store's
    own meaning. The pool a max ranged over is `argmax_pool_n`. Writing the
    pool into `n_concepts` was invisible while no exclusion happened (pool ==
    library) and became a cross-artifact disagreement the moment amendment I4
    turned exclusion on: the real corrected null said 707 where the label
    store and feature_ap_main.h5 said 716. The gate caught it on first
    consumption; this test keeps it caught."""
    with pytest.raises(PreflightError,
                       match=r"n_concepts=\S+ but label store says"):
        check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5",
                        n_concepts=POOL_N,        # WRONG: the pool
                        argmax_pool_n=POOL_N),    # right
              require_clean_git=False)


def test_provenance_excluded_type_in_the_store_is_caught(tmp_path):
    """M2. The check that compares the store to the SPEC rather than to a peer
    artifact. Every other check here is cross-component, and cross-component
    agreement cannot see a violation that propagated consistently -- HELIX,
    STRAND, TURN and COMPBIAS were in the library, the AP matrix and the null,
    in perfect agreement and all wrong."""
    lab = make_labels(tmp_path / "lab.h5", concepts=CONCEPTS + ["HELIX"])
    with pytest.raises(PreflightError, match="PROVENANCE"):
        check(lab, require_clean_git=False,
              config_excluded_types=("HELIX", "STRAND", "TURN", "COMPBIAS"))


def test_clean_store_passes_the_provenance_check(tmp_path):
    """Non-vacuity: a store with none of the excluded types must pass."""
    s = check(make_labels(tmp_path / "lab.h5"), require_clean_git=False,
              config_excluded_types=("HELIX", "STRAND", "TURN", "COMPBIAS"))
    assert s["provenance_excluded_present_in_store"] == []


def test_provenance_check_is_skipped_when_no_config_given(tmp_path):
    """MUTATION GUARD: the check must be opt-in via --config, so callers that
    do not pass one are unaffected. If this ever fires unconditionally it
    would break every existing invocation."""
    s = check(make_labels(tmp_path / "lab.h5", concepts=CONCEPTS + ["HELIX"]),
              require_clean_git=False)
    assert "provenance_excluded_present_in_store" not in s


# ------------------------------------ the assertion patterns themselves ----

def test_no_fixture_asserts_on_a_BARE_identity_key_name():
    """A `match=` that is just a key name is satisfied by ANY message
    mentioning that key -- and the gate's messages now quote other keys in
    their explanatory text, so a bare key-name pattern can be satisfied by a
    different check's explanation.

    This has bitten twice. `argmax_pool_n`: renaming the key made it ABSENT,
    and the absent-key complaint contains the substring. `library_sha`: the
    new schema rule's complaint contains it too. Both fixtures passed while
    testing nothing they named.

    So the rule is structural rather than case-by-case: no assertion pattern
    in the pre-flight suites may be a bare canonical key name.

    Parsed with `ast`, NOT with a regex over the source. A regex also matches
    the key names quoted inside docstrings that EXPLAIN this very rule, and
    flags them as violations -- the check would fail on its own documentation
    while the real assertions were fine. Only actual `pytest.raises(...)`
    call sites count.
    """
    import ast
    import pathlib

    from src.analysis.artifact_identity import CANONICAL_IDENTITY_CHECKS

    keys = {c["key"] for c in CANONICAL_IDENTITY_CHECKS} | {"ap_impl"}
    offenders = []
    for fname in ("test_preflight.py", "test_preflight_historical.py"):
        path = pathlib.Path(__file__).parent / fname
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(
                fn, "id", "")
            if name != "raises":
                continue
            for kw in node.keywords:
                if kw.arg == "match" and isinstance(kw.value, ast.Constant)                         and kw.value.value in keys:
                    offenders.append(
                        f"{fname}:{node.lineno} match={kw.value.value!r}")
    assert not offenders, (
        "these assertions are satisfied by any message mentioning the key, "
        "not by the check they name: " + "; ".join(offenders)
        + ". Tighten each to its MESSAGE FAMILY, e.g. "
          r'r"truncation_policy=\S+ but label store says".')


# =====================================================================
# 7a/7b/7c -- THE THREE CHECKS ADDED 2026-08-11, because the experimental
# pair failed PERMANENTLY and a check that always fails detects nothing.
#
# Every case below is a deliberate violation of exactly the condition the
# check claims to detect (gotcha 13), and each expectation is WRITTEN DOWN
# rather than computed from the code under test (gotcha 32b).
# =====================================================================

# The store gains evidence TIERS. `make_labels` writes only `all`, which is
# what a POOLED consumer reads; a stratified consumer reads the tier datasets,
# and a fixture that omits a field cannot exercise the check on that field
# (gotcha 16).
#
# WRITTEN DOWN, not derived: of these five concepts, exactly THREE carry an
# experimental tier -- ACT_SITE, BINDING:ATP and DOMAIN. `BINDING` and
# `DOMAIN:SH2` carry only propagated tiers. So the experimental axis is
# ["ACT_SITE", "BINDING:ATP", "DOMAIN"] in sorted order, and its argmax-pool
# subset drops `DOMAIN` (a pooled parent) leaving TWO.
EXP_TIERS = {
    "ACT_SITE":    {"experimental": 4, "by_similarity": 9},
    "BINDING":     {"by_similarity": 7},
    "BINDING:ATP": {"experimental_highthroughput": 2},
    "DOMAIN":      {"experimental": 11},
    "DOMAIN:SH2":  {"predicted": 3},
}
EXPERIMENTAL_AXIS = ["ACT_SITE", "BINDING:ATP", "DOMAIN"]
EXPERIMENTAL_AXIS_IN_POOL = ["ACT_SITE", "BINDING:ATP"]


def make_tiered_labels(path, tiers=None, **over):
    """A store that also publishes per-tier residue lists, as the real one
    does. An EMPTY tier dataset is written for one concept deliberately: the
    derivation must treat a present-but-empty tier as absent, and only a
    fixture containing one can show that it does."""
    tiers = EXP_TIERS if tiers is None else tiers
    with h5py.File(path, "w") as hf:
        g = hf.create_group("concepts")
        for c in CONCEPTS:
            cg = g.create_group(c)
            cg.create_dataset("all", data=np.arange(200))
            for t, n in tiers.get(c, {}).items():
                cg.create_dataset(t, data=np.arange(n))
        # a present-but-EMPTY experimental tier must NOT put a concept on the
        # experimental axis
        if "experimental" not in tiers.get("DOMAIN:SH2", {}):
            g["DOMAIN:SH2"].create_dataset("experimental",
                                           data=np.zeros(0, np.int64))
        for k, v in {**BASE, **LABEL_EXTRA,
                     "n_concepts": len(CONCEPTS), **over}.items():
            hf.attrs[k] = v
    return path


def make_strat_ap(path, concepts=None, stratum="experimental",
                  ap_impl=AP_IMPL_SEGMENTED, negatives="unannotated", **over):
    """A STRATIFIED AP artifact: a subset axis, a declared `strata`, and the
    masked-path impl marker -- the shape of `feature_ap_experimental_av*.h5`.
    `n_concepts`/`argmax_pool_n` carry the LIBRARY/POOL counts, which is the
    committed convention (gotcha 16) and what the merge rewrites them to."""
    concepts = EXPERIMENTAL_AXIS if concepts is None else concepts
    with h5py.File(path, "w") as hf:
        g = hf.create_group(stratum)
        g.create_dataset("concepts", data=np.array(concepts,
                                                   dtype=h5py.string_dtype()))
        g.create_dataset("ap", data=np.zeros((2, 4, len(concepts)), np.float32))
        for k, v in {**BASE, **ARTIFACT_EXTRA, "n_concepts": len(CONCEPTS),
                     "argmax_pool_n": POOL_N, "ap_impl": ap_impl,
                     "strata": json.dumps([stratum]), "negatives": negatives,
                     **over}.items():
            hf.attrs[k] = v
    return path


def make_null_with_axis(path, concepts, **over):
    """A null that publishes its concept axis as a ROOT-LEVEL dataset, which
    is what `null_pilot.py` actually writes."""
    p = make_null(path, **over)
    with h5py.File(p, "a") as hf:
        hf.create_dataset("concepts", data=np.array(concepts,
                                                    dtype=h5py.string_dtype()))
    return p


# ---- 7a. the stratified concept axis ---------------------------------

def test_the_STRATUM_AXIS_DERIVATION_matches_what_is_written_down(tmp_path):
    """THE FIXTURE IS SHOWN TO DISCRIMINATE BEFORE ANY RESULT IS ASSERTED.

    The expectation is written out by hand above; if the derivation returned
    the full axis, or counted the empty `experimental` tier on DOMAIN:SH2, or
    ignored `experimental_highthroughput`, this fails.
    """
    lab = make_tiered_labels(tmp_path / "lab.h5")
    got = stratum_concept_axis(lab, "experimental")
    assert got == EXPERIMENTAL_AXIS, got
    assert got != sorted(CONCEPTS), "the derivation returned the FULL axis"
    assert "DOMAIN:SH2" not in got, "a present-but-EMPTY tier put a concept on"
    assert "BINDING:ATP" in got, "experimental_highthroughput was not grouped"
    assert stratum_concept_axis(lab, "propagated") == [
        "ACT_SITE", "BINDING", "DOMAIN:SH2"]


def test_pooled_is_refused_as_a_stratum_rather_than_returning_the_full_axis(tmp_path):
    """`pooled` is not a subset and must not be derivable through this path.
    Returning the full axis would make the two branches look interchangeable
    when only one of them can fail."""
    lab = make_tiered_labels(tmp_path / "lab.h5")
    with pytest.raises(ValueError, match="not a stratum subset"):
        stratum_concept_axis(lab, "pooled")


def test_a_STRATIFIED_artifact_with_the_right_subset_PASSES(tmp_path):
    """AC2-1: the experimental pair's shape goes green, with the structural
    difference ACCOUNTED FOR. Under the old equality check this raised."""
    s = check(make_tiered_labels(tmp_path / "lab.h5"),
              make_null_with_axis(tmp_path / "null.h5",
                                  EXPERIMENTAL_AXIS_IN_POOL),
              (make_strat_ap(tmp_path / "ap.h5"),),
              require_clean_git=False)
    ax = s["concept_axis"]["ap.h5"]
    assert ax["axis_kind"] == "stratified" and ax["n_axis"] == 3


def test_a_WRONG_SUBSET_OF_THE_RIGHT_SIZE_is_caught(tmp_path):
    """THE DISCRIMINATION CASE THE FIX EXISTS FOR, and the reason an exemption
    is the weaker answer.

    Three concepts, exactly the right COUNT, and one of them (`BINDING`) has no
    experimental residue at all while `DOMAIN` -- which does -- is missing.
    Adding the axis to a provenance exemption list would wave this straight
    through; encoding the RELATIONSHIP catches it.
    """
    wrong = ["ACT_SITE", "BINDING", "BINDING:ATP"]
    assert len(wrong) == len(EXPERIMENTAL_AXIS)          # same size, on purpose
    with pytest.raises(PreflightError, match=r"concept AXIS is not the store"):
        check(make_tiered_labels(tmp_path / "lab.h5"),
              None, (make_strat_ap(tmp_path / "ap.h5", concepts=wrong),),
              require_clean_git=False)


def test_a_POOLED_artifact_is_STILL_held_to_equality(tmp_path):
    """The relaxation must not leak into the pooled path: a pooled artifact
    missing a concept is still a failure, exactly as before."""
    with pytest.raises(PreflightError, match=r"concept AXIS is not the store"):
        check(make_labels(tmp_path / "lab.h5"), None,
              (make_ap(tmp_path / "ap.h5", concepts=CONCEPTS[:-1],
                       n_concepts=len(CONCEPTS)),),
              require_clean_git=False)


def test_TWO_declared_strata_is_NOT_DERIVABLE_not_passed(tmp_path):
    """Gotcha 25c: a check whose expectation cannot be formed must say so, not
    pass. No artifact in this project writes a two-stratum axis, so there is
    nothing to compare against -- and reporting that as agreement is how an
    UNGOVERNED key becomes an unchecked one."""
    lab = make_tiered_labels(tmp_path / "lab.h5")
    ap = make_strat_ap(tmp_path / "ap.h5",
                       strata=json.dumps(["experimental", "propagated"]))
    s = check(lab, None, (ap,), require_clean_git=False)
    assert s["concept_axis"]["ap.h5"]["axis_kind"] == "stratified"
    assert any("NOT DERIVABLE" in m for m in s["not_applicable"]), \
        s.get("not_applicable")


# ---- 7b. the null's axis against the AP artifact's --------------------

def test_a_null_that_is_the_ARGMAX_POOL_SUBSET_passes(tmp_path):
    s = check(make_tiered_labels(tmp_path / "lab.h5"),
              make_null_with_axis(tmp_path / "null.h5",
                                  EXPERIMENTAL_AXIS_IN_POOL),
              (make_strat_ap(tmp_path / "ap.h5"),),
              require_clean_git=False)
    pair = s["null_vs_ap_concept_axis"][0]
    assert pair["verdict"] == "AGREE"
    assert (pair["n_null"], pair["n_ap"], pair["n_ap_in_pool"]) == (2, 3, 2)


def test_a_null_MISSING_a_pool_concept_is_caught(tmp_path):
    with pytest.raises(PreflightError, match=r"argmax-pool subset"):
        check(make_tiered_labels(tmp_path / "lab.h5"),
              make_null_with_axis(tmp_path / "null.h5", ["ACT_SITE"]),
              (make_strat_ap(tmp_path / "ap.h5"),),
              require_clean_git=False)


def test_a_null_carrying_a_POOLED_PARENT_is_caught(tmp_path):
    """The founding reason the subset is the pool and not the axis: `g_f` is a
    MAX, and a pooled parent left in can win it (I4)."""
    with pytest.raises(PreflightError, match=r"argmax-pool subset"):
        check(make_tiered_labels(tmp_path / "lab.h5"),
              make_null_with_axis(tmp_path / "null.h5", EXPERIMENTAL_AXIS),
              (make_strat_ap(tmp_path / "ap.h5"),),
              require_clean_git=False)


def test_a_null_with_the_RIGHT_COUNT_and_WRONG_MEMBERS_is_caught(tmp_path):
    """ADDED after the mutation matrix showed the count-comparison mutation was
    killed ONLY by the permutation fixture.

    That was thin cover: a permutation and a wrong-membership set of equal size
    are different defect classes, and a count check is blind to both. Two
    concepts, exactly the right number, one of them (`DOMAIN`) a pooled parent
    that has no business in a null and the other (`BINDING:ATP`) missing.
    """
    wrong = ["ACT_SITE", "DOMAIN"]
    assert len(wrong) == len(EXPERIMENTAL_AXIS_IN_POOL)   # same count, on purpose
    assert sorted(wrong) != sorted(EXPERIMENTAL_AXIS_IN_POOL)
    with pytest.raises(PreflightError, match=r"argmax-pool subset"):
        check(make_tiered_labels(tmp_path / "lab.h5"),
              make_null_with_axis(tmp_path / "null.h5", wrong),
              (make_strat_ap(tmp_path / "ap.h5"),),
              require_clean_git=False)


def test_a_PERMUTED_null_axis_is_caught_POSITIONALLY(tmp_path):
    """GOTCHA 27. Same concepts, same count, same set -- different ORDER.

    No count check and no set check can see this, and the seed-axis precedent
    shows what it costs: 18 of 20 rows mis-paired into a complete, internally
    consistent, entirely wrong result. The test asserts the failure names a
    POSITION, not merely that it failed.
    """
    permuted = list(reversed(EXPERIMENTAL_AXIS_IN_POOL))
    assert sorted(permuted) == sorted(EXPERIMENTAL_AXIS_IN_POOL)
    assert permuted != EXPERIMENTAL_AXIS_IN_POOL
    with pytest.raises(PreflightError,
                       match=r"SAME CONCEPTS, DIFFERENT ORDER.*position 0"):
        check(make_tiered_labels(tmp_path / "lab.h5"),
              make_null_with_axis(tmp_path / "null.h5", permuted),
              (make_strat_ap(tmp_path / "ap.h5"),),
              require_clean_git=False)


def test_a_null_with_NO_axis_is_reported_as_unpaired_not_as_agreeing(tmp_path):
    """Every artifact predating the axis dataset lacks it. Absence must not
    manufacture a pair that then reads as AGREE (I3)."""
    s = check(make_tiered_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5"),
              (make_strat_ap(tmp_path / "ap.h5"),),
              require_clean_git=False)
    assert "null_vs_ap_concept_axis" not in s


# ---- 7c. `ap_impl` derived from whether masking is active -------------

def test_a_MASKED_artifact_is_expected_to_be_SEGMENTED(tmp_path):
    s = check(make_tiered_labels(tmp_path / "lab.h5"), None,
              (make_strat_ap(tmp_path / "ap.h5"),), require_clean_git=False)
    e = s["ap_impl_expectation"]["ap.h5"]
    assert e["negatives"] == "unannotated"
    assert e["expected"] == AP_IMPL_SEGMENTED


def test_a_SPARSE_marker_on_a_MASKED_artifact_STILL_FAILS(tmp_path):
    """Direction one. Masking changes the ranks the sparse path computes
    against the full row set, so this is a real defect, not a variant."""
    with pytest.raises(PreflightError, match=r"ap_impl=.*but the expected"):
        check(make_tiered_labels(tmp_path / "lab.h5"), None,
              (make_strat_ap(tmp_path / "ap.h5", ap_impl=AP_IMPL_SPARSE),),
              require_clean_git=False)


def test_a_SEGMENTED_marker_on_an_UNMASKED_artifact_STILL_FAILS(tmp_path):
    """Direction two -- gotcha 25: 'can only fail' is as useless as 'can never
    fail'. The derivation must not become a blanket acceptance of both impls.
    """
    with pytest.raises(PreflightError, match=r"ap_impl=.*but the expected"):
        check(make_labels(tmp_path / "lab.h5"), None,
              (make_ap(tmp_path / "ap.h5", ap_impl=AP_IMPL_SEGMENTED,
                       negatives="all_rows"),),
              require_clean_git=False)


def test_an_ABSENT_negatives_attr_falls_back_and_does_not_crash(tmp_path):
    """Totality (gotcha 24): every artifact written before `negatives` existed
    reaches this branch, and it must resolve to the declared default rather
    than raising or silently accepting anything."""
    want, why = expected_ap_impl(None, AP_IMPL_SPARSE)
    assert (want, why) == (AP_IMPL_SPARSE, "no masking declared")
    assert expected_ap_impl(b"unannotated", AP_IMPL_SPARSE)[0] == \
        AP_IMPL_SEGMENTED


# ---- amendment CT §CT10: a POOLED AP scored on a row SUBSET -------------------
#
# WRITTEN DOWN: concept rows are disjoint blocks of 40; the half holds rows
# 0-39 and 80-119, i.e. ACT_SITE's and BINDING:ATP's rows and nothing of the
# other three. So the expected axis is ["ACT_SITE", "BINDING:ATP"].
SUBSET_ROWS = {c: np.arange(40 * i, 40 * i + 40) for i, c in enumerate(CONCEPTS)}
SUBSET_AXIS = ["ACT_SITE", "BINDING:ATP"]


def _subset_store_and_halves(tmp_path):
    lab = tmp_path / "lab_subset.h5"
    with h5py.File(lab, "w") as hf:
        g = hf.create_group("concepts")
        for c in CONCEPTS:
            g.create_group(c).create_dataset("all", data=SUBSET_ROWS[c])
        for k, v in {**BASE, **LABEL_EXTRA, "n_concepts": len(CONCEPTS)}.items():
            hf.attrs[k] = v
    halves = tmp_path / "halves.h5"
    with h5py.File(halves, "w") as hf:
        hf.create_dataset("ct10_half_a_rows",
                          data=np.concatenate([np.arange(0, 40), np.arange(80, 120)]))
        hf.attrs["labels"] = str(lab)
    return lab, halves


def test_the_ROW_SUBSET_AXIS_derivation_matches_what_is_written_down(tmp_path):
    from src.analysis.artifact_identity import row_subset_concept_axis
    lab, halves = _subset_store_and_halves(tmp_path)
    got = row_subset_concept_axis(lab, halves, "ct10_half_a")
    assert got == SUBSET_AXIS, got


def test_a_ROW_SUBSET_AP_with_the_right_axis_PASSES(tmp_path):
    lab, halves = _subset_store_and_halves(tmp_path)
    ap = make_ap(tmp_path / "ap.h5", concepts=SUBSET_AXIS, n_concepts=len(CONCEPTS),
                 scored_row_set="ct10_half_a", row_set_file=str(halves))
    s = check(lab, None, (ap,), require_clean_git=False,
              expect_scored_row_set="ct10_half_a")
    assert s["concept_axis"]["ap.h5"]["n_axis"] == 2


def test_a_WRONG_ROW_SUBSET_AXIS_of_the_right_size_is_caught(tmp_path):
    lab, halves = _subset_store_and_halves(tmp_path)
    ap = make_ap(tmp_path / "ap.h5", concepts=["ACT_SITE", "BINDING"],
                 n_concepts=len(CONCEPTS), scored_row_set="ct10_half_a",
                 row_set_file=str(halves))
    with pytest.raises(PreflightError, match=r"concept AXIS is not the store's concepts with >=1 row in ct10_half_a"):
        check(lab, None, (ap,), require_clean_git=False,
              expect_scored_row_set="ct10_half_a")


def test_a_ROW_SUBSET_AP_without_its_file_is_NOT_DERIVABLE_not_passed(tmp_path):
    lab, _ = _subset_store_and_halves(tmp_path)
    ap = make_ap(tmp_path / "ap.h5", concepts=SUBSET_AXIS, n_concepts=len(CONCEPTS),
                 scored_row_set="ct10_half_a")
    s = check(lab, None, (ap,), require_clean_git=False,
              expect_scored_row_set="ct10_half_a")
    assert any("no `row_set_file`" in m for m in s["not_applicable"]), s["not_applicable"]


# --------------------------------------------------------------------------
# Amendment CT (the author, 2026-10-24): the scoped AP-arithmetic exemption.
# --------------------------------------------------------------------------

_GPU = "cuda:NVIDIA RTX 5000 Ada Generation:sm_89"


def _devs(gather, arith):
    return {"gather_device_class": gather, "ap_arith_device_class": arith}


def test_an_ap_arith_split_is_refused_without_the_flag(tmp_path):
    with pytest.raises(PreflightError, match="ap_arith_device_class"):
        check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5", **_devs(_GPU, _GPU)),
              (make_ap(tmp_path / "ap.h5", **_devs(_GPU, "cpu:x86_64")),),
              require_clean_git=False)


def test_the_flag_accepts_an_ap_arith_split_only_when_the_gather_agrees(tmp_path):
    s = check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5", **_devs(_GPU, _GPU)),
              (make_ap(tmp_path / "ap.h5", **_devs(_GPU, "cpu:x86_64")),),
              require_clean_git=False, accept_ap_arith_split=True)
    assert len(s["ap_arith_splits_accepted"]) == 1
    assert s["ap_arith_splits_accepted"][0]["gather_agrees"] == _GPU


def test_the_flag_never_excuses_a_gather_that_differs(tmp_path):
    with pytest.raises(PreflightError, match="gather_device_class"):
        check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5", **_devs(_GPU, _GPU)),
              (make_ap(tmp_path / "ap.h5", **_devs("cpu:x86_64", "cpu:x86_64")),),
              require_clean_git=False, accept_ap_arith_split=True)


def test_the_flag_never_excuses_an_arith_split_when_the_gather_is_absent(tmp_path):
    with pytest.raises(PreflightError, match="ap_arith_device_class"):
        check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5", ap_arith_device_class=_GPU),
              (make_ap(tmp_path / "ap.h5", ap_arith_device_class="cpu:x86_64"),),
              require_clean_git=False, accept_ap_arith_split=True)
