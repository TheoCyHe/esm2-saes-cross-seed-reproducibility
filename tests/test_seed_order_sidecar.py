"""The pooled raw null's seed-order sidecar (amendment BS, part C3).

THE FIXTURE IS A NON-IDENTITY PERMUTATION, AND THAT IS THE WHOLE POINT. Gotcha
29's failure mode is a POSITIONAL join -- slab `s` assumed to be seed `s`. A
fixture whose true mapping is the identity cannot distinguish a working
recovery from that bug, because both return `range(n)`. Every test below uses
the mapping slab 0 -> seed 2, slab 1 -> seed 0, slab 2 -> seed 1, which is
written down here and not computed from the module under test (gotcha 32b).
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from src.analysis.classification_2x2 import SeedOrderUnrecoverable
from src.analysis.seed_order_sidecar import (
    PROVENANCE,
    _slab_max_over_concepts,
    build_sidecar,
    recover_all,
)

# The true mapping, WRITTEN DOWN: raw slab i holds the seed at reductions row
# TRUE_POS[i]. Deliberately not the identity.
TRUE_POS = [2, 0, 1]
N_SHIFTS, N_CONCEPTS, N_FEATURES = 2, 3, 4


def _make_pair(tmp_path, positions=TRUE_POS, arm_b_positions=None,
               add_seeds_dataset=False):
    """Build a raw/reductions pair whose recovery answer is known by construction.

    Each seed gets a distinct constant `base`, so the max over concepts is a
    constant array unique to that seed. The concept axis carries base-1 and
    base-2 underneath the max, so a bug that took the WRONG reduction (a min, a
    mean, the first concept) would produce a value no row matches.
    """
    arm_b_positions = positions if arm_b_positions is None else arm_b_positions
    n = len(positions)
    raw_p, red_p = tmp_path / "raw.h5", tmp_path / "red.h5"

    def slab_for(seed_row):
        base = 10.0 * (seed_row + 1)
        a = np.empty((N_SHIFTS, N_CONCEPTS, N_FEATURES), dtype=np.float32)
        for c in range(N_CONCEPTS):
            a[:, c, :] = base - c          # max over concepts == base
        return a

    with h5py.File(raw_p, "w") as f:
        f.create_dataset("concepts",
                         data=np.array([f"C{i}".encode() for i in range(N_CONCEPTS)]))
        f.create_dataset("gain_a",
                         data=np.stack([slab_for(p) for p in positions]))
        f.create_dataset("gain_b",
                         data=np.stack([slab_for(p) for p in arm_b_positions]))
        if add_seeds_dataset:
            f.create_dataset("seeds",
                             data=np.array([b"s"] * n))

    with h5py.File(red_p, "w") as f:
        f.create_dataset("seeds",
                         data=np.array([f"seed_{i}".encode() for i in range(n)]))
        for arm in ("a", "b"):
            f.create_dataset(
                f"null_g_{arm}",
                data=np.stack([np.full((N_SHIFTS, N_FEATURES), 10.0 * (i + 1),
                                       dtype=np.float64)
                               for i in range(n)]))
    return raw_p, red_p


# ---------------------------------------------------------------------------
# the per-shift read must equal the whole-slab max
# ---------------------------------------------------------------------------

def test_per_shift_max_equals_hand_written_value(tmp_path):
    """Slab holding base 20 under concepts 20, 19, 18 -> max is 20 everywhere.

    The expectation is the literal 20.0, not `arr.max(axis=1)` recomputed from
    the same array -- a test that derives its expectation from the thing under
    test cannot detect a change to it.
    """
    raw_p, _ = _make_pair(tmp_path)
    with h5py.File(raw_p, "r") as f:
        got = _slab_max_over_concepts(f["gain_a"], 0)   # slab 0 holds seed row 2
    assert got.shape == (N_SHIFTS, N_FEATURES)
    assert np.all(got == 30.0)          # seed row 2 -> base 10*(2+1) = 30
    # the raw array's STORED dtype (amendment CT; float64 before it)
    assert got.dtype == np.float32


def test_per_shift_read_agrees_with_whole_slab_read(tmp_path):
    """Chunking is an optimisation and must not change the answer."""
    raw_p, _ = _make_pair(tmp_path)
    with h5py.File(raw_p, "r") as f:
        chunked = _slab_max_over_concepts(f["gain_a"], 1)
        whole = np.asarray(f["gain_a"][1], dtype=np.float64).max(axis=1)
    assert np.array_equal(chunked, whole)


# ---------------------------------------------------------------------------
# recover_all -- the non-identity permutation is the discriminating case
# ---------------------------------------------------------------------------

def test_recovers_the_written_down_non_identity_permutation(tmp_path):
    raw_p, red_p = _make_pair(tmp_path)
    rec = recover_all(raw_p, red_p)
    assert rec["positions_arm_a"] == TRUE_POS
    assert rec["positions_arm_b"] == TRUE_POS
    assert rec["arms_agree"] is True
    assert rec["is_permutation"] is True
    # The fixture is NOT the identity, so a positional join would be caught.
    assert rec["identity_ordering"] is False
    assert rec["slab_to_seed_label"] == {"0": "seed_2", "1": "seed_0",
                                         "2": "seed_1"}


def test_a_positional_join_would_give_a_DIFFERENT_answer(tmp_path):
    """The fixture discriminates -- stated as a test, not as a comment.

    If the module returned `range(n)` (the gotcha-29 bug), it would disagree
    with the recovered mapping on all three slabs. Asserting that here is what
    makes the fixture's discriminating power a measurement rather than a claim.
    """
    raw_p, red_p = _make_pair(tmp_path)
    rec = recover_all(raw_p, red_p)
    positional = list(range(len(TRUE_POS)))
    assert rec["positions_arm_a"] != positional
    disagreements = sum(1 for i, j in zip(rec["positions_arm_a"], positional)
                        if i != j)
    assert disagreements == 3


# ---------------------------------------------------------------------------
# every refusal condition fires
# ---------------------------------------------------------------------------

def test_refuses_when_the_two_arms_disagree(tmp_path):
    raw_p, red_p = _make_pair(tmp_path, arm_b_positions=[0, 1, 2])
    with pytest.raises(SeedOrderUnrecoverable, match="DIFFERENT orders"):
        build_sidecar(raw_p, red_p, "sha", False)


def test_refuses_when_the_mapping_is_not_a_bijection(tmp_path):
    """Two slabs carrying the same seed is not a permutation."""
    raw_p, red_p = _make_pair(tmp_path, positions=[1, 1, 2])
    with pytest.raises(SeedOrderUnrecoverable, match="not a\n?\\s*permutation"):
        build_sidecar(raw_p, red_p, "sha", False)


def test_refuses_when_no_labelled_seed_matches(tmp_path):
    """Zero matches -- the reductions do not describe this array."""
    raw_p, red_p = _make_pair(tmp_path)
    with h5py.File(red_p, "a") as f:
        f["null_g_a"][...] = f["null_g_a"][...] + 1.0   # nothing matches now
    with pytest.raises(SeedOrderUnrecoverable, match="NO labelled seed"):
        recover_all(raw_p, red_p)


def test_refuses_when_the_recovery_cannot_discriminate(tmp_path):
    """More than one labelled seed matching must refuse, not pick the first."""
    raw_p, red_p = _make_pair(tmp_path)
    with h5py.File(red_p, "a") as f:
        g = f["null_g_a"][...]
        g[0] = g[2]                       # rows 0 and 2 now identical
        f["null_g_a"][...] = g
    with pytest.raises(SeedOrderUnrecoverable, match="cannot discriminate"):
        recover_all(raw_p, red_p)


def test_refuses_if_the_raw_artifact_gains_seed_labels(tmp_path):
    """If labels are ever stored, recovering an order would SHADOW them."""
    raw_p, red_p = _make_pair(tmp_path, add_seeds_dataset=True)
    with pytest.raises(RuntimeError, match="now HAS a `seeds` dataset"):
        recover_all(raw_p, red_p)


def test_refuses_on_seed_axis_length_disagreement(tmp_path):
    raw_p, red_p = _make_pair(tmp_path)
    with h5py.File(red_p, "a") as f:
        del f["seeds"]
        f.create_dataset("seeds", data=np.array([b"seed_0", b"seed_1"]))
    with pytest.raises(RuntimeError, match="seed-axis length disagreement"):
        recover_all(raw_p, red_p)


# ---------------------------------------------------------------------------
# the sidecar's own labelling
# ---------------------------------------------------------------------------

def test_sidecar_labels_itself_recovered_and_NOT_from_filenames(tmp_path):
    """The label is load-bearing: BS's brief described the recovery as coming
    from a naming convention, and it does not. A sidecar that said so would
    understate its own evidence and misdescribe its refusal conditions."""
    raw_p, red_p = _make_pair(tmp_path)
    doc = build_sidecar(raw_p, red_p, "abc123", False)
    prov = doc["identity"]["provenance"]
    assert prov == PROVENANCE
    assert "RECOVERED" in prov
    assert "NOT READ FROM THE ARTIFACT" in prov
    assert "NOT FROM FILENAMES" in prov
    assert "filename" not in doc["identity"]["method"].lower()


def test_sidecar_records_that_nothing_was_written_back(tmp_path):
    raw_p, red_p = _make_pair(tmp_path)
    doc = build_sidecar(raw_p, red_p, "abc123", False)
    assert "back-filled" in doc["identity"]["not_written_into_the_artifact"]
    # and the raw artifact is genuinely untouched -- no `seeds` dataset added
    with h5py.File(raw_p, "r") as f:
        assert "seeds" not in f


def test_sidecar_is_json_serialisable(tmp_path):
    raw_p, red_p = _make_pair(tmp_path)
    doc = build_sidecar(raw_p, red_p, "abc123", False)
    round_tripped = json.loads(json.dumps(doc))
    assert round_tripped["recovery"]["positions_arm_a"] == TRUE_POS


def test_all_five_refusal_conditions_are_documented(tmp_path):
    """The outcome space is enumerated in the artifact, not only in code."""
    raw_p, red_p = _make_pair(tmp_path)
    doc = build_sidecar(raw_p, red_p, "abc123", False)
    conds = doc["identity"]["refusal_conditions"]
    assert len(conds) == 5
    joined = " ".join(conds)
    for expected in ("ZERO matches", "MORE THAN ONE match", "ARMS DISAGREE",
                     "NOT A PERMUTATION", "`seeds` DATASET"):
        assert expected in joined


# ---------------------------------------------------------------------------
# amendment CT: the comparison is made in the raw array's STORED precision,
# the same fix classification_2x2.recover_raw_slab_seed got at d762121
# ---------------------------------------------------------------------------

def _make_f64_accumulated_pair(tmp_path, positions=TRUE_POS, seed=7,
                               raw_dtype=np.float32, g_offset=0.0):
    """A null whose gain was accumulated in float64 (the segmented CPU path):
    the raw array keeps the float32 cast, `null_g` the float64 max. Raw slab i
    holds reductions row positions[i] -- the written-down non-identity map."""
    rng = np.random.default_rng(seed)
    n = len(positions)
    G = rng.uniform(0, 1, (n, N_SHIFTS, N_CONCEPTS, 50))     # by reductions row
    raw_p, red_p = tmp_path / "raw64.h5", tmp_path / "red64.h5"
    with h5py.File(raw_p, "w") as f:
        for arm in ("a", "b"):
            f.create_dataset(f"gain_{arm}",
                             data=np.stack([G[p] for p in positions]).astype(raw_dtype))
    with h5py.File(red_p, "w") as f:
        f.create_dataset("seeds",
                         data=np.array([f"seed_{i}".encode() for i in range(n)]))
        for arm in ("a", "b"):
            f.create_dataset(f"null_g_{arm}", data=G.max(axis=2) + g_offset)
    return raw_p, red_p, G


def test_a_f64_accumulated_null_defeats_a_float64_comparison(tmp_path):
    """The founding shape (CT5's gtilde jobs 30306560/1 hit it in
    classification_2x2): compared in float64, the float32 slab's max matches no
    row of the float64 null_g. This is what the module did before CT."""
    from src.analysis.classification_2x2 import recover_seed_position
    raw_p, red_p, _ = _make_f64_accumulated_pair(tmp_path)
    with h5py.File(raw_p, "r") as f, h5py.File(red_p, "r") as r:
        m = np.asarray(f["gain_a"][0], dtype=np.float64).max(axis=1)
        with pytest.raises(SeedOrderUnrecoverable, match="NO labelled seed"):
            recover_seed_position(m, np.asarray(r["null_g_a"][:]))


def test_recovers_the_permutation_on_a_f64_accumulated_null(tmp_path):
    raw_p, red_p, _ = _make_f64_accumulated_pair(tmp_path)
    rec = recover_all(raw_p, red_p)
    assert rec["positions_arm_a"] == TRUE_POS
    assert rec["positions_arm_b"] == TRUE_POS
    assert rec["is_permutation"] is True
    assert rec["identity_ordering"] is False


def test_stored_precision_still_refuses_foreign_reductions(tmp_path):
    """A difference far above float32 rounding is not absorbed by the cast."""
    raw_p, red_p, _ = _make_f64_accumulated_pair(tmp_path, g_offset=1e-4)
    with pytest.raises(SeedOrderUnrecoverable, match="NO labelled seed"):
        recover_all(raw_p, red_p)


def test_a_float64_raw_array_is_compared_in_float64(tmp_path):
    """No cast when the raw array is float64: an offset far BELOW float32
    rounding must still refuse, so the fix cannot become a float32 tolerance."""
    raw_p, red_p, _ = _make_f64_accumulated_pair(
        tmp_path, raw_dtype=np.float64, g_offset=1e-12)
    with pytest.raises(SeedOrderUnrecoverable, match="NO labelled seed"):
        recover_all(raw_p, red_p)
    (tmp_path / "exact").mkdir()
    raw_p2, red_p2, _ = _make_f64_accumulated_pair(
        tmp_path / "exact", raw_dtype=np.float64)
    assert recover_all(raw_p2, red_p2)["positions_arm_a"] == TRUE_POS


def test_unchanged_for_a_float32_representable_null(tmp_path):
    """Every pooled null of record: null_g is a max over float32 values held in
    float64, so the cast is lossless and the recovery is what it was. The
    fixture of _make_pair is that case; its answer is the written-down map."""
    raw_p, red_p = _make_pair(tmp_path)
    with h5py.File(red_p, "r") as f:
        g = f["null_g_a"][:]
        assert g.dtype == np.float64
        assert np.array_equal(g.astype(np.float32).astype(np.float64), g)
    assert recover_all(raw_p, red_p)["positions_arm_a"] == TRUE_POS
