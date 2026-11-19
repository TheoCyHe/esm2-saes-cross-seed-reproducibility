"""HISTORICAL-DEFECT FIXTURES -- one per real defect the gate must refuse.

WHY THIS FILE IS SEPARATE FROM `test_preflight.py`.

Three of the five defects found in the 2026-07-28 session were caught by
`preflight.py`, not by the test suite. That makes the gate the component whose
failure is *unrecoverable*: if it stops refusing, nothing downstream notices,
and the artifacts that result look entirely plausible. It has already been
wrong once at exactly the thing that mattered (`argmax_pool_n` -- named in
CLAUDE.md's identity set, absent from the code).

Its coverage was demonstrated by anecdote -- "we ran it against the real files
and it refused" -- which a refactor can silently drop. Each fixture below
reconstructs one defect that ACTUALLY OCCURRED, in miniature, and asserts the
gate refuses it.

MUTATION REQUIREMENT (gotcha 13 extension 3). Every fixture here must be
verified to FAIL when its corresponding gate branch is removed. A fixture that
passes because some *other* branch caught the mismatch provides no coverage of
the branch it names -- that exact failure has already happened once in this
project (the concept-count test that stayed green when its dedicated branch
was deleted, because `n_concepts` is also in IDENTITY_KEYS). The matrix is
`test_mutation_matrix_documents_which_branch_each_fixture_covers`.
"""

from __future__ import annotations

import json
import subprocess

import h5py
import numpy as np
import pytest

from src.analysis.artifact_identity import AP_IMPL_SPARSE
from src.analysis.preflight import PreflightError, check

BASE = dict(truncation_policy="untruncated",
            retention_floor_mode="interplm", retention_min_residues=1500,
            retention_min_instances=10, min_protein_count=5)
# The store publishes the building code's SHA as `git_sha`; consumers copy it
# into `library_sha`. `scored_row_set` has no store-side counterpart at all.
LABEL_EXTRA = dict(git_sha="f67485d", n_rows=1000)
ARTIFACT_EXTRA = dict(scored_row_set="label_store", n_rows=1000,
                      library_sha="f67485d")
CONCEPTS = ["ACT_SITE", "BINDING", "BINDING:ATP", "DOMAIN", "DOMAIN:SH2"]
POOL_N = 3          # BINDING and DOMAIN are pooled parents


def make_labels(path, concepts=CONCEPTS, **over):
    with h5py.File(path, "w") as hf:
        g = hf.create_group("concepts")
        for c in concepts:
            g.create_group(c).create_dataset("all", data=np.arange(200))
        for k, v in {**BASE, **LABEL_EXTRA,
                     "n_concepts": len(concepts), **over}.items():
            hf.attrs[k] = v
    return path


def make_null(path, **over):
    with h5py.File(path, "w") as hf:
        hf.create_dataset("null_g_a", data=np.zeros((2, 2, 4)))
        for k, v in {**BASE, **ARTIFACT_EXTRA, "n_concepts": len(CONCEPTS),
                     "argmax_pool_n": POOL_N,
                     "exclude_pooled_parents": True, **over}.items():
            hf.attrs[k] = v
    return path


def make_ap(path, concepts=CONCEPTS, **over):
    with h5py.File(path, "w") as hf:
        g = hf.create_group("pooled")
        g.create_dataset("concepts",
                         data=np.array(concepts, dtype=h5py.string_dtype()))
        g.create_dataset("ap", data=np.zeros((2, 4, len(concepts)), np.float32))
        for k, v in {**BASE, **ARTIFACT_EXTRA, "n_concepts": len(concepts),
                     "argmax_pool_n": POOL_N, "ap_impl": AP_IMPL_SPARSE,
                     **over}.items():
            hf.attrs[k] = v
    return path


def _git_repo(tmp_path, name="repo"):
    repo = tmp_path / name
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    return repo


# ==================================================================== #
# DEFECT 1 -- pool mismatch: a 716-pool null against a 707-pool AP matrix
# (707 = the 716-concept library's argmax pool, 716 minus 9 pooled parents;
# PREREGISTRATION §L4a. The 703 pool is the later compliant axis.)
# Amendment I4. The gate PASSED this on the real artifacts: `n_concepts` is
# the LIBRARY size and was 716 for both, so the concept-count branch could
# not see it. Found only by running the gate against files it should have
# refused -- never by reading it.
# ==================================================================== #

def test_defect_1_pool_mismatch_is_refused(tmp_path):
    # The regex matches the COMPARISON message specifically, not merely the
    # key name. Mutation testing showed the looser `match="argmax_pool_n"`
    # was satisfied by an entirely different branch: renaming the key in the
    # canonical table makes it absent, and the absent-key complaint contains
    # the substring too. A fixture that any message mentioning the key can
    # satisfy is not testing the check it names.
    with pytest.raises(PreflightError,
                       match=r"argmax_pool_n=\d+ but the label store implies"):
        check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5", argmax_pool_n=len(CONCEPTS)),
              require_clean_git=False)


# ==================================================================== #
# DEFECT 2 -- `n_concepts` carrying two meanings across artifacts
# Amendment L4b. `null_pilot.py` wrote the POOL into `n_concepts` while the
# label store and `feature_ap.py` wrote the LIBRARY size. Invisible while no
# exclusion happened (pool == library), a disagreement the moment I4 turned
# exclusion on. Caught by the gate on first consumption of the corrected null.
# ==================================================================== #

def test_defect_2_n_concepts_meaning_pool_not_library_is_refused(tmp_path):
    with pytest.raises(PreflightError,
                       match=r"n_concepts=\S+ but label store says"):
        check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5",
                        n_concepts=POOL_N,        # the POOL -- wrong
                        argmax_pool_n=POOL_N),    # right
              require_clean_git=False)


# ==================================================================== #
# DEFECT 3 -- contaminated axis: store disagrees with the CONFIG's exclusion
# Amendment M2. The SPEC check, not a peer check, and the only defect here
# that no amount of cross-component agreement could catch: HELIX, STRAND,
# TURN and COMPBIAS sat in the library, the AP matrix AND the null, in
# perfect agreement and all wrong.
# ==================================================================== #

def test_defect_3_contaminated_axis_is_refused(tmp_path):
    lab = make_labels(tmp_path / "lab.h5", concepts=CONCEPTS + ["HELIX"])
    with pytest.raises(PreflightError, match="PROVENANCE"):
        check(lab, require_clean_git=False,
              config_excluded_types=("HELIX", "STRAND", "TURN", "COMPBIAS"))


def test_defect_3_is_NOT_catchable_by_peer_agreement(tmp_path):
    """The reason the spec check had to exist, asserted directly: with every
    artifact agreeing on the contaminated axis, every peer check passes. Only
    the config comparison fires."""
    # Every artifact must AGREE on the contaminated axis, or a peer check
    # fires and the test demonstrates nothing. HELIX is a childless bare
    # concept, so it joins the argmax pool: 6 concepts - 2 parents = 4.
    contaminated = CONCEPTS + ["HELIX"]
    pool_n = 4
    lab = make_labels(tmp_path / "lab.h5", concepts=contaminated)
    nul = make_null(tmp_path / "null.h5", n_concepts=len(contaminated),
                    argmax_pool_n=pool_n)
    ap = make_ap(tmp_path / "ap.h5", concepts=contaminated,
                 n_concepts=len(contaminated), argmax_pool_n=pool_n)
    check(lab, nul, (ap,), require_clean_git=False)      # no config => PASSES
    with pytest.raises(PreflightError, match="PROVENANCE"):
        check(lab, nul, (ap,), require_clean_git=False,
              config_excluded_types=("HELIX",))


# ==================================================================== #
# DEFECT 4 -- stale-but-present label store
# Amendment H10 / gotcha 15. `main_ap.sbatch` guarded `--labels` with
# `[[ ! -f ]]`. The guard PASSED; the file was the 822-concept
# a `with_truncated` store from three rebuilds earlier. A missing file fails
# loudly; a stale-but-present one runs and produces a plausible, wrong matrix.
# ==================================================================== #

def test_defect_4_stale_but_present_store_is_refused(tmp_path):
    stale = make_labels(tmp_path / "stale.h5", truncation_policy="with_truncated")
    # Here the STORE is the stale one, so the artifact is the side that
    # disagrees: null says `untruncated`, store says `all`.
    with pytest.raises(PreflightError,
                       match=r"truncation_policy=\S+ but label store says"):
        check(stale, make_null(tmp_path / "null.h5"), require_clean_git=False)


# ==================================================================== #
# DEFECT 5 -- mis-attributed replacement artifact (found this session)
# `derive_null_reductions` inherited `n_concepts` from the SOURCE across an
# axis change, so the replacement misdescribed itself. The gate refused the
# stale null AND the mis-attributed replacement on the same run.
# ==================================================================== #

def test_defect_5_replacement_inheriting_source_identity_is_refused(tmp_path):
    lab = make_labels(tmp_path / "lab.h5")                     # 5 concepts
    with pytest.raises(PreflightError,
                       match=r"n_concepts=\S+ but label store says"):
        check(lab, make_null(tmp_path / "null.h5", n_concepts=99),
              require_clean_git=False)


def test_defect_5b_reduction_stale_for_axis_is_refused(tmp_path):
    """Amendment N11: `null_g_*` is a max OVER concepts, so it changes when
    the pool changes. A reductions file valid only for its own pool carries
    `superseded_for_axis` and must never be consumed for another."""
    marker = json.dumps({"axis_argmax_pool_n": 703})
    with pytest.raises(PreflightError, match=r"carries `superseded_for_axis`"):
        check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5", superseded_for_axis=marker),
              require_clean_git=False)


# ==================================================================== #
# DEFECT 6 -- git_dirty true
# Two ad-hoc .sbatch files in the cluster checkout made git_dirty True for the
# WHOLE tree; had main_ap launched that day it would have been the project's
# first artifact with dirty provenance.
# ==================================================================== #

def test_defect_6_dirty_tree_is_refused(tmp_path):
    repo = _git_repo(tmp_path)
    (repo / "untracked_adhoc.sbatch").write_text("# scratch script\n")
    with pytest.raises(PreflightError, match="git_dirty is True"):
        check(make_labels(tmp_path / "lab.h5"), require_clean_git=True,
              repo_root=repo)


def test_defect_6_clean_tree_passes(tmp_path):
    """Non-vacuity: the git check must not fire on a clean tree, or every
    other fixture here could be passing for that reason."""
    repo = _git_repo(tmp_path)
    s = check(make_labels(tmp_path / "lab.h5"), require_clean_git=True,
              repo_root=repo)
    assert s["git_porcelain"].strip() == ""


# ==================================================================== #
# DEFECT 7 -- truncation_policy / scored_row_set mismatch
# Amendment G2. `--scored-row-set all` against an untruncated store scores
# 17,739,363 rows while the store's n_rows is 15,972,325, making every
# prevalence -- and therefore every normalised gain -- wrong.
# ==================================================================== #

def test_defect_7_population_mismatch_is_refused(tmp_path):
    with pytest.raises(PreflightError,
                       match=r"truncation_policy=\S+ but label store says"):
        check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5", truncation_policy="with_truncated"),
              require_clean_git=False)


def test_defect_7b_floor_parameter_mismatch_is_refused(tmp_path):
    """Floor parameters travel with the truncation policy: a store built under
    a different floor is a different axis even at the same policy."""
    with pytest.raises(PreflightError, match=r"retention_min_instances=\S+ but label store says"):
        check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5", retention_min_instances=999),
              require_clean_git=False)


# ==================================================================== #
# Non-vacuity + the coverage matrix
# ==================================================================== #

def test_all_consistent_inputs_pass(tmp_path):
    s = check(make_labels(tmp_path / "lab.h5"),
              make_null(tmp_path / "null.h5"),
              (make_ap(tmp_path / "ap.h5"),),
              require_clean_git=False,
              config_excluded_types=("HELIX", "STRAND", "TURN", "COMPBIAS"))
    assert s["argmax_pool_n"] == POOL_N
    assert s["provenance_excluded_present_in_store"] == []


# ==================================================================== #
# DEFECT 9 -- A CHECK THAT CANNOT FAIL. Amendment P8b / Q7.
# `library_sha` was an `equal_to_labels` key, and NO LABEL STORE WRITES IT --
# stores publish the building code's SHA as `git_sha`. The comparison reads
# `if k in lab_id and ...`, the key was never in `lab_id`, and the branch
# could not fire. Reader-side coverage was COMPLETE: the canonical key table
# listed it, the mutation matrix named a fixture for it, and every audit read
# as green. This is gotcha 13 arriving inside the guard built to enforce
# gotcha 13.
#
# The missing assertion is about the SCHEMA -- the key set the gate READS
# reconciled against the key set the producers WRITE.
# ==================================================================== #

def test_defect_9_equality_key_no_store_writes_is_refused():
    """Rule (a): no `equal_to_labels` key may be one no label store writes.

    Asserted on the canonical table itself rather than on a fixture file,
    because the defect is in the SCHEMA, not in any pair of artifacts.
    """
    from src.analysis.artifact_identity import unwritable_equality_keys
    bad = unwritable_equality_keys()
    assert not bad, (
        f"{bad} are declared `equal_to_labels` but no label store publishes "
        "them, so those comparisons can never fire. A check that cannot fail "
        "is not a check -- it reads as coverage in every audit. This is how "
        "`library_sha` stayed dead for the gate's whole life while both "
        "compliant artifacts carried the wrong store's SHA.")


def test_defect_9b_missing_derived_SOURCE_key_is_a_hard_failure(tmp_path):
    """Rule (b): a `derived` check's SOURCE key must be present on the store.

    Absent source is a SCHEMA defect, not a not-applicable check. Conflating
    the two is exactly how the dead branch survived -- silently doing nothing
    reads the same as passing.
    """
    lab = tmp_path / "lab.h5"
    make_labels(lab)
    with h5py.File(lab, "a") as hf:
        del hf.attrs["git_sha"]          # the source of `library_sha`
    with pytest.raises(PreflightError, match="cannot fire"):
        check(lab, make_null(tmp_path / "null.h5"), require_clean_git=False)


def test_defect_9c_every_derived_check_names_a_reachable_source():
    """The reconciliation itself: every `derived` check either computes its
    expectation from the store's CONTENT or names a store attr, and every
    named attr is one a label store actually publishes."""
    from src.analysis.artifact_identity import LABEL_STORE_KEYS, derived_sources
    unreachable = {k: v for k, v in derived_sources().items()
                   if v not in LABEL_STORE_KEYS}
    assert not unreachable, (
        f"derived checks {unreachable} name source keys no label store "
        "writes -- the same schema mismatch as defect 9, one level removed.")


def test_mutation_matrix_documents_which_branch_each_fixture_covers():
    """The coverage matrix, kept in-tree so the claim is auditable.

    Each canonical identity key names the fixture asserting it. The test then
    checks the matrix against the gate's own key table, so a key added to the
    gate WITHOUT a fixture is a visible failure rather than a silent gap --
    which is how `argmax_pool_n` came to be documented but unchecked.
    """
    from src.analysis.artifact_identity import CANONICAL_IDENTITY_CHECKS
    covered = {
        "truncation_policy": "test_defect_7_population_mismatch_is_refused",
        "n_concepts": "test_defect_2_n_concepts_meaning_pool_not_library",
        # NOT a generic equality key any more, and the reason is the point:
        # no label store writes `library_sha` (it publishes `git_sha`), so the
        # equality branch could never fire. tests/test_preflight.py
        "library_sha": "test_library_sha_mismatch_is_refused",
        # The retired overloading -- `truncation_policy` agreeing says nothing
        # about which cache rows were scored. tests/test_preflight.py
        "scored_row_set": "test_scored_row_set_mismatch_is_refused_while_population_agrees",
        "n_rows": "test_n_rows_mismatch_against_the_store_is_refused",
        "retention_floor_mode": "generic equality key",
        "retention_min_residues": "generic equality key",
        "retention_min_instances": "test_defect_7b_floor_parameter_mismatch",
        "min_protein_count": "generic equality key",
        "argmax_pool_n": "test_defect_1_pool_mismatch_is_refused",
        "exclude_pooled_parents": "test_null_that_kept_pooled_parents_is_caught",
        "superseded_by": "test_self_declared_supersession_is_refused",
        "superseded_for_axis": "test_defect_5b_reduction_stale_for_axis",
    }
    gate_keys = {c["key"] for c in CANONICAL_IDENTITY_CHECKS}
    missing = gate_keys - set(covered)
    assert not missing, (
        f"gate checks {sorted(missing)} have no named fixture. Every canonical "
        "identity key needs one, or the gate's coverage is anecdotal again.")
    stale = set(covered) - gate_keys
    assert not stale, f"matrix names non-existent gate keys: {sorted(stale)}"

    # Defect 9's coverage is SCHEMA-level and has no per-key fixture: the
    # defect was that a key had complete reader-side coverage while no writer
    # wrote it. Asserted separately so the value-comparison matrix above is
    # not read as covering it -- that misreading is the defect itself.
    from src.analysis.artifact_identity import derived_sources
    assert derived_sources(), (
        "no `derived` check names a source key. The schema reconciliation "
        "that catches a check-that-cannot-fail has been removed (defect 9).")
