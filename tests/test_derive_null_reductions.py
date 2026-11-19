"""Derived-artifact attribute inheritance is DEFAULT-DENY.

WHY THIS FILE EXISTS. Two attribute-inheritance defects landed in one session,
and both were fixed by adding one more key to a denylist:

  1. the mis-attributed replacement -- a derived file naming the wrong source;
  2. `superseded_for_axis` -- the first pass marked its source stale, the
     second pass inherited that marker onto its own replacement, which then
     declared itself superseded and the pre-flight gate refused it.

Adding a key repairs the case and not the class. A denylist is only correct if
every attribute that will EVER appear on a source was thought of in advance,
and the attribute set demonstrably grows. The failure is silent: a new key
propagates onto a file it does not describe and nothing reads wrong.

So the rule is inverted -- a derived artifact inherits NOTHING unless the key
is named -- and the test below is written against the general property (an
unknown key does not propagate) rather than against any particular key, so it
keeps working for attributes nobody has invented yet.

MUTATION REQUIREMENT (gotcha 13 extension 3): removing the allowlist must kill
`test_unknown_source_attribute_does_not_propagate`. That mutation is in
`tests/mutation_matrix_preflight.py`.
"""

from __future__ import annotations

import sys

import h5py
import numpy as np
import pytest

CONCEPTS = ["ACT_SITE", "BINDING:ATP", "DOMAIN:SH2"]

# What the SOURCE reductions carry. Deliberately a mix of the three cases:
# genuinely inheritable run properties, identity keys describing the SOURCE's
# axis, and a key nobody has thought about yet.
SOURCE_ATTRS = {
    # -- permutation / SAE properties: legitimately carried over
    "n_shifts": 20, "n_length_bins": 20, "null_seed": 0, "null_type": "both",
    "exclude_pooled_parents": True, "scored_row_set": "label_store",
    "n_proteins": 51943,
    # -- identity describing the SOURCE's axis: must NOT be carried over
    "library_sha": "OLD_STORE_SHA", "labels": "/path/to/OLD_store.h5",
    "n_concepts": 716, "argmax_pool_n": 707,
    "superseded_for_axis": '{"reason": "marks the SOURCE, not its replacement"}',
    "identity_correction": "a correction applied to the SOURCE",
    # -- the one that matters: an attribute the allowlist has never seen
    "zzz_attribute_invented_after_the_denylist_was_written": "leaked",
}


def _make_inputs(tmp_path, n_seeds=2, n_shifts=3, n_feats=4):
    raw = tmp_path / "raw.h5"
    with h5py.File(raw, "w") as hf:
        hf.create_dataset("concepts",
                          data=np.array(CONCEPTS, dtype=h5py.string_dtype()))
        for arm in ("a", "b"):
            hf.create_dataset(
                f"gain_{arm}",
                data=np.arange(n_seeds * n_shifts * len(CONCEPTS) * n_feats,
                               dtype=np.float32).reshape(
                                   n_seeds, n_shifts, len(CONCEPTS), n_feats))

    src = tmp_path / "src.h5"
    with h5py.File(src, "w") as hf:
        hf.create_dataset("fire_rate", data=np.zeros((n_seeds, n_feats)))
        hf.create_dataset("seeds", data=np.arange(n_seeds))
        hf.create_dataset("shift_schedule", data=np.zeros((n_shifts, 2)))
        hf.create_dataset("dest_schedule", data=np.zeros((n_shifts, 2)))
        hf.create_dataset("length_bins", data=np.zeros(4))
        for k, v in SOURCE_ATTRS.items():
            hf.attrs[k] = v

    lab = tmp_path / "labels.h5"
    with h5py.File(lab, "w") as hf:
        g = hf.create_group("concepts")
        for c in CONCEPTS:
            g.create_group(c).create_dataset("all", data=np.arange(10))
        for k, v in {"git_sha": "NEW_STORE_SHA", "truncation_policy": "untruncated",
                     "n_rows": 15972325, "retention_floor_mode": "interplm",
                     "retention_min_residues": 1500,
                     "retention_min_instances": 10,
                     "min_protein_count": 5, "n_concepts": len(CONCEPTS)}.items():
            hf.attrs[k] = v
    return raw, src, lab


def _derive(tmp_path, monkeypatch, out_name="out.h5"):
    from src.analysis.derive_null_reductions import main
    raw, src, lab = _make_inputs(tmp_path)
    out = tmp_path / out_name
    monkeypatch.setattr(sys, "argv", [
        "derive_null_reductions", "--raw", str(raw), "--source-reductions",
        str(src), "--labels", str(lab), "--out", str(out)])
    main()
    return out


# ------------------------------------------------------- the general rule --

def test_unknown_source_attribute_does_not_propagate(tmp_path, monkeypatch):
    """THE CLASS, not the case.

    The source carries an attribute the allowlist has never heard of. Under a
    denylist it propagates -- silently, onto an artifact it does not describe.
    Under default-deny it cannot, and no maintenance is required when the next
    unforeseen attribute appears.
    """
    out = _derive(tmp_path, monkeypatch)
    with h5py.File(out, "r") as hf:
        leaked = [k for k in hf.attrs
                  if k.startswith("zzz_attribute_invented")]
        assert not leaked, (
            f"a source attribute the allowlist never named propagated onto "
            f"the derived artifact: {leaked}. Default-deny means an unknown "
            "key does not travel, so a new attribute on a source cannot "
            "silently describe a file it was not written for.")


def test_identity_comes_from_the_target_store_not_the_source(tmp_path,
                                                             monkeypatch):
    """The live instance this conversion found: `null_pilot_compliant.h5` was
    built against the 712-concept f67485d store and shipped naming the
    716-concept 9bfa94b one, because `library_sha` and `labels` were inherited.
    The pre-flight gate passed it, since its own `library_sha` check was dead.
    Two independent defects lined up to hide one stale identity."""
    out = _derive(tmp_path, monkeypatch)
    with h5py.File(out, "r") as hf:
        assert hf.attrs["library_sha"] == "NEW_STORE_SHA"
        assert "OLD_store" not in str(hf.attrs["labels"])
        assert int(hf.attrs["n_concepts"]) == len(CONCEPTS)
        assert int(hf.attrs["argmax_pool_n"]) == len(CONCEPTS)


def test_supersession_marker_is_not_inherited(tmp_path, monkeypatch):
    """The second of the two founding defects, kept as a regression: a derived
    artifact must never inherit a marker that describes the thing it
    replaces."""
    out = _derive(tmp_path, monkeypatch)
    with h5py.File(out, "r") as hf:
        assert "superseded_for_axis" not in hf.attrs
        assert "identity_correction" not in hf.attrs


def test_run_properties_ARE_inherited(tmp_path, monkeypatch):
    """Not vacuous. Default-deny is only correct if the keys that genuinely
    describe the source RUN -- the permutation schedule and the SAEs' firing
    rates, neither of which a re-derivation changes -- still travel."""
    out = _derive(tmp_path, monkeypatch)
    with h5py.File(out, "r") as hf:
        assert int(hf.attrs["n_shifts"]) == 20
        assert str(hf.attrs["null_type"]) == "both"
        assert bool(hf.attrs["exclude_pooled_parents"]) is True
        assert str(hf.attrs["scored_row_set"]) == "label_store"
        assert "fire_rate" in hf and "shift_schedule" in hf


def test_inheritance_decision_is_recorded_in_the_artifact(tmp_path,
                                                          monkeypatch):
    """Both lists are written down, so what was dropped is auditable from the
    file rather than only from the source code that produced it."""
    out = _derive(tmp_path, monkeypatch)
    import json
    with h5py.File(out, "r") as hf:
        inh = json.loads(hf.attrs["inherited_keys"])
        dropped = json.loads(hf.attrs["not_inherited_keys"])
    assert "n_shifts" in inh
    assert "library_sha" in dropped and "superseded_for_axis" in dropped
    assert any(k.startswith("zzz_attribute_invented") for k in dropped)
    assert not set(inh) & set(dropped)


# ------------------------------------------------------- the arithmetic ----

def test_reductions_are_the_two_maxima_over_the_right_axes(tmp_path,
                                                           monkeypatch):
    """`null_g_*` is a max over CONCEPTS per feature; `null_cmax_*` a max over
    FEATURES per concept. Amendment N11 turns on them being different objects,
    so the axes are asserted rather than assumed."""
    raw, src, lab = _make_inputs(tmp_path)
    from src.analysis.derive_null_reductions import main
    out = tmp_path / "o.h5"
    monkeypatch.setattr(sys, "argv", [
        "x", "--raw", str(raw), "--source-reductions", str(src),
        "--labels", str(lab), "--out", str(out)])
    main()
    with h5py.File(raw, "r") as hf:
        g = hf["gain_a"][:]
    with h5py.File(out, "r") as hf:
        assert np.allclose(hf["null_g_a"][:], g.max(axis=2))
        assert np.allclose(hf["null_cmax_a"][:], g.max(axis=3))


def test_refuses_to_overwrite(tmp_path, monkeypatch):
    out = _derive(tmp_path, monkeypatch)
    raw, src, lab = _make_inputs(tmp_path)
    from src.analysis.derive_null_reductions import main
    monkeypatch.setattr(sys, "argv", [
        "x", "--raw", str(raw), "--source-reductions", str(src),
        "--labels", str(lab), "--out", str(out)])
    with pytest.raises(SystemExit):
        main()
