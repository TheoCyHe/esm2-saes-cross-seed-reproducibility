"""Artifact identity + pooled-parent argmax pool.

Per gotcha 13, every assertion here is exercised in BOTH directions: the
matching case must pass AND the deliberately-mismatched case must raise. A
check that has never been seen to fail is not evidence.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from src.analysis.artifact_identity import (
    EVAL_CACHE_IDENTITY_KEYS,
    IDENTITY_KEYS,
    EvalCacheIdentityMismatch,
    argmax_pool_mask,
    assert_eval_cache_identity,
    assert_identity_match,
    eval_cache_identity,
    expected_eval_cache_attrs_from_config,
    pooled_parents,
    read_identity,
    write_identity,
)


# ------------------------------------------------------- pooled parents ----

def test_bare_type_with_children_is_a_pooled_parent():
    cs = ["DOMAIN", "DOMAIN:Protein kinase", "DOMAIN:SH2", "ACT_SITE"]
    assert pooled_parents(cs) == {"DOMAIN"}


def test_bare_type_without_children_is_not_a_parent():
    """ACT_SITE, DISULFID, TRANSMEM etc. are genuinely un-subtyped single
    concepts and must STAY in the argmax pool -- excluding them would silently
    delete real functional-site targets."""
    cs = ["ACT_SITE", "DISULFID", "TRANSMEM", "DOMAIN", "DOMAIN:SH2"]
    assert pooled_parents(cs) == {"DOMAIN"}
    mask = argmax_pool_mask(cs)
    assert dict(zip(cs, mask)) == {
        "ACT_SITE": True, "DISULFID": True, "TRANSMEM": True,
        "DOMAIN": False, "DOMAIN:SH2": True,
    }


def test_binding_is_a_pooled_parent_via_its_ligand_children():
    """BINDING is sub-typed by /ligand through a different mechanism than the
    /note types, but it is structurally the same pooled parent -- measured at
    98.6% covered by its 145 children on the real library."""
    cs = ["BINDING", "BINDING:ATP", "BINDING:Zn(2+)"]
    assert pooled_parents(cs) == {"BINDING"}


def test_child_is_never_mistaken_for_a_parent_by_prefix():
    """A concept whose NAME merely starts with another's must not count as a
    child -- the separator has to be the colon."""
    cs = ["SIGNAL", "SIGNALLING"]           # no "SIGNAL:" concept exists
    assert pooled_parents(cs) == set()


def test_argmax_pool_matches_the_measured_library_shape():
    """The nine real parents, plus a childless single, on the real names."""
    cs = sorted([
        "BINDING", "BINDING:ATP", "DOMAIN", "DOMAIN:SH2", "MOTIF", "MOTIF:x",
        "REGION", "REGION:Disordered", "REPEAT", "REPEAT:WD", "SIGNAL",
        "SIGNAL:unspecified", "TOPO_DOM", "TOPO_DOM:Cytoplasmic", "TRANSIT",
        "TRANSIT:Mitochondrion", "ZN_FING", "ZN_FING:C2H2-type", "ACT_SITE",
    ])
    assert len(pooled_parents(cs)) == 9
    assert sum(argmax_pool_mask(cs)) == len(cs) - 9


def test_pool_mask_would_catch_a_parent_left_in_the_pool():
    """The regression this exists to prevent: if pooled parents were NOT
    excluded, DOMAIN -- the single broadest concept on the real library
    (p=0.1245) -- stays eligible to win the argmax and D5's breadth-dominance
    problem is unfixed by construction."""
    cs = ["DOMAIN", "DOMAIN:SH2"]
    assert argmax_pool_mask(cs)[cs.index("DOMAIN")] is False
    naive_pool_would_be = [True, True]              # the broken behaviour
    assert argmax_pool_mask(cs) != naive_pool_would_be


# ------------------------------------------------------------- identity ----

def _write(path, **kw):
    with h5py.File(path, "w") as hf:
        write_identity(hf, **kw)
    return path


BASE = dict(truncation_policy="untruncated", n_concepts=716, library_sha="9bfa94b",
            retention_floor_mode="interplm", retention_min_residues=1500,
            retention_min_instances=10, min_protein_count=5)


def test_identity_roundtrip(tmp_path):
    p = _write(tmp_path / "a.h5", **BASE)
    got = read_identity(p)
    assert got["truncation_policy"] == "untruncated"
    assert int(got["n_concepts"]) == 716
    assert set(got) == set(IDENTITY_KEYS)


def test_matching_identities_pass(tmp_path):
    a = read_identity(_write(tmp_path / "a.h5", **BASE))
    b = read_identity(_write(tmp_path / "b.h5", **BASE))
    assert_identity_match(a, b, "a", "b")          # must not raise


@pytest.mark.parametrize("key,bad", [
    ("truncation_policy", "with_truncated"),
    ("n_concepts", 822),
    ("library_sha", "f3e2ddb"),
    ("retention_floor_mode", "legacy"),
    ("retention_min_residues", 100),
    ("retention_min_instances", 3),
    ("min_protein_count", 1),
])
def test_each_identity_key_mismatch_is_caught(tmp_path, key, bad):
    """gotcha 13: the check must be SEEN to fail, key by key. A wrong policy
    and `n_concepts=822` are the actual values of the stale store the
    --labels default pointed at."""
    a = read_identity(_write(tmp_path / "a.h5", **BASE))
    b = read_identity(_write(tmp_path / "b.h5", **{**BASE, key: bad}))
    with pytest.raises(ValueError, match="IDENTITY MISMATCH"):
        assert_identity_match(a, b, "store", "artifact")


def test_absent_key_does_not_read_as_matching(tmp_path):
    """null_pilot_g.h5 shipped with no `population` attr at all. An absent key
    must not be silently treated as agreement -- assert_identity_match skips
    it, and preflight reports it as ABSENT rather than passing."""
    a = read_identity(_write(tmp_path / "a.h5", **BASE))
    partial = {k: v for k, v in BASE.items() if k != "truncation_policy"}
    b = read_identity(_write(tmp_path / "b.h5", **partial))
    assert "truncation_policy" not in b
    assert_identity_match(a, b, "a", "b")          # skipped, not asserted


# ------------------------- readers must json.loads a list-valued attr, not --
# ------------------------- list() it. Gotcha 20's founding defect, retired. --

def test_no_module_calls_list_directly_on_an_h5_attr():
    """`write_identity` json.dumps() every list, so a list-valued attr comes
    back as a STRING and `list()` of it yields CHARACTERS -- which produced
    "firing bins" 24/49/50, a full table of plausible wrong numbers.

    Gotcha 20 records that defect as fixed. The fix reached three of the four
    readers of `firing_rate_edges`; `diagnose_standardization.py` still had the
    broken form (dead there, so nothing was misreported -- one usage away from
    reproducing the project's own documented defect). That is gotcha 16's
    non-propagation applied to a gotcha-20 fix, so the guard is structural now
    rather than a thing to remember.

    `ast`, not a text scan: this docstring contains the offending pattern, and
    gotcha 20's own corollary is that a source-text checker's false positives
    land on the documentation explaining it -- and a check that fails on its own
    explanation gets disabled rather than fixed.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "list"
                    and len(node.args) == 1):
                continue
            arg = node.args[0]
            # list(x.attrs["k"])
            touches_attrs = (
                isinstance(arg, ast.Subscript)
                and isinstance(arg.value, ast.Attribute)
                and arg.value.attr == "attrs"
            ) or (
                # list(x.attrs.get("k"))
                isinstance(arg, ast.Call)
                and isinstance(arg.func, ast.Attribute)
                and arg.func.attr == "get"
                and isinstance(arg.func.value, ast.Attribute)
                and arg.func.value.attr == "attrs"
            )
            if touches_attrs:
                offenders.append(
                    f"{path.relative_to(root.parent.parent)}:{node.lineno}")

    assert not offenders, (
        "list() applied directly to an h5 attr: "
        + ", ".join(offenders)
        + ". write_identity json.dumps lists, so decode then json.loads when "
          "the value is a str. list() of a JSON string yields characters and "
          "fails silently."
    )


# ------------------------------------------------- the GATHER DEVICE CLASS --
#
# Gotcha 36: the activation gather is device-dependent, so an artifact that
# cannot state its device class cannot be asserted against one that can. These
# fixtures cover the derivation, the four-way comparison, and the one case
# that matters most in practice -- BOTH SIDES ABSENT, which is every artifact
# written before the key existed.

def test_device_class_names_the_backend_and_the_machine_for_cpu():
    import platform

    from src.analysis.artifact_identity import device_class
    want = f"cpu:{platform.machine()}"
    assert device_class("cpu") == want
    assert device_class(torch.device("cpu")) == want


def test_device_class_accepts_the_TENSOR_not_only_a_device_string():
    """The point of the key is to record what was ACTUALLY used, so the caller
    must be able to hand it the output of the gather rather than the variable
    it hoped the gather would run on."""
    from src.analysis.artifact_identity import device_class
    t = torch.zeros(3)
    assert device_class(t) == device_class(t.device) == device_class("cpu")


def test_device_class_names_an_UNKNOWN_backend_rather_than_coercing_it():
    """`meta` is neither cpu nor cuda. A device class this function does not
    understand must not be reported as one it does."""
    from src.analysis.artifact_identity import device_class
    assert device_class("meta") == "meta:UNKNOWN_CLASS"


def test_device_class_is_more_than_cpu_vs_gpu():
    """gotcha 36's divergence is between device CLASSES. A string that only
    said 'cuda' would make an sm_89 artifact and an sm_120 one
    indistinguishable, so the cuda branch carries name AND capability. The cuda
    branch cannot run on a machine with no GPU; what is asserted here is the
    part this machine can measure, and the gap is stated rather than papered
    over with a mock that would only test the mock."""
    from src.analysis.artifact_identity import device_class
    assert device_class("cpu").count(":") == 1


@pytest.mark.parametrize("a,b,verdict,agree", [
    ({"gather_device_class": "cuda:X:sm_89"},
     {"gather_device_class": "cuda:X:sm_89"}, "AGREE", True),
    ({"gather_device_class": "cuda:X:sm_89"},
     {"gather_device_class": "cpu:x86_64"}, "DIFFER", False),
    ({"gather_device_class": "cpu:x86_64"}, {}, "ABSENT_ON_ONE", False),
    ({}, {"gather_device_class": "cpu:x86_64"}, "ABSENT_ON_ONE", False),
    ({}, {}, "ABSENT_ON_BOTH", False),
])
def test_the_device_comparison_is_TOTAL_over_the_two_presence_bits(
        a, b, verdict, agree):
    from src.analysis.artifact_identity import compare_device_class
    r = compare_device_class(a, b, "ap", "null")
    assert r["verdict"] == verdict
    assert r["agree"] is agree


def test_BOTH_ABSENT_IS_NOT_AGREEMENT():
    """THE ONE THIS EXISTS FOR, and it is the COMMON case rather than an edge
    case: every artifact on disk today predates the key, so a comparison that
    treated both-absent as a match would report AGREE for every real pair
    while checking nothing.

    Written so it FAILS against a permissive implementation -- one returning
    `agree=True` when neither side carries the key fails on the first two
    assertions independently, not merely on a verdict string.
    """
    from src.analysis.artifact_identity import compare_device_class
    r = compare_device_class({}, {}, "ap", "null")
    assert r["agree"] is False
    assert r["verdict"] != "AGREE"
    assert r["ap"] == "<ABSENT>" and r["null"] == "<ABSENT>"
    assert "UNRECORDED" in r["reading"]


def test_an_absent_device_class_does_not_make_two_artifacts_unequal_either():
    """The companion half. `ABSENT_ON_BOTH` is a statement of ignorance, so it
    must not be reported as DIFFER -- that would be a claim about two devices
    nobody recorded, in the other direction."""
    from src.analysis.artifact_identity import compare_device_class
    assert compare_device_class({}, {}, "a", "b")["verdict"] == "ABSENT_ON_BOTH"


def test_an_existing_artifact_without_the_key_still_reads_its_identity(tmp_path):
    """AC2-3. The key is additive: an artifact written before it existed reads
    back exactly as before, with the new key simply ABSENT rather than
    defaulted, and `assert_identity_match` on the historical key set is
    unaffected."""
    from src.analysis.artifact_identity import GATHER_DEVICE_KEY
    a = read_identity(_write(tmp_path / "a.h5", **BASE), extra=(GATHER_DEVICE_KEY,))
    b = read_identity(_write(tmp_path / "b.h5", **BASE), extra=(GATHER_DEVICE_KEY,))
    assert GATHER_DEVICE_KEY not in a and GATHER_DEVICE_KEY not in b
    assert_identity_match(a, b, "a", "b")          # must not raise


def test_write_identity_round_trips_the_device_class(tmp_path):
    """AC2-1, at the schema level: the key is an ordinary attr and survives a
    write/read cycle."""
    from src.analysis.artifact_identity import GATHER_DEVICE_KEY
    p = _write(tmp_path / "a.h5", **BASE,
               **{GATHER_DEVICE_KEY: "cuda:NVIDIA RTX 5000 Ada Generation:sm_89"})
    got = read_identity(p, extra=(GATHER_DEVICE_KEY,))
    assert got[GATHER_DEVICE_KEY] == "cuda:NVIDIA RTX 5000 Ada Generation:sm_89"


def test_mixed_hardware_is_recorded_as_MIXED_not_as_the_last_seed():
    """An array whose seeds ran on two device classes is not an array with one
    device class. Collapsing to whichever seed finished last would state a
    device the artifact does not have."""
    from src.analysis.artifact_identity import device_identity_attrs
    rec = {"gather_device_class": "cpu:x86_64"}
    seen = {"gather_device_class": {"cpu:x86_64", "cuda:X:sm_89"}}
    out = device_identity_attrs(rec, seen)
    assert out["gather_device_class"] == "MIXED:cpu:x86_64,cuda:X:sm_89"


def test_a_single_device_class_is_written_plainly():
    from src.analysis.artifact_identity import device_identity_attrs
    rec = {"gather_device_class": "cpu:x86_64"}
    out = device_identity_attrs(rec, {"gather_device_class": {"cpu:x86_64"}})
    assert out["gather_device_class"] == "cpu:x86_64"


class _FakeSAE:
    """Returns a CPU `z` whatever device it is handed -- the discrepancy the
    device key exists to catch, made reproducible without a second device."""

    def __init__(self, d_model: int, n_feat: int):
        import torch as _t
        self.W_enc = _t.zeros(n_feat, d_model)
        self.n_feat = n_feat

    def __call__(self, x):
        import torch as _t
        z = _t.zeros(x.shape[0], self.n_feat)
        z[:, 0] = 1.0
        return None, z, None


def _toy_cache(path, n_rows=8, d_model=4):
    with h5py.File(path, "w") as hf:
        hf.create_dataset("activations",
                          data=np.random.RandomState(0).randn(n_rows, d_model))
    return path


def test_the_gather_records_the_device_it_RAN_on_not_the_one_requested(tmp_path):
    """THE PROPERTY THE KEY IS FOR. A job can request one device and have the
    arithmetic run on another -- that discrepancy is what got four tasks
    reclaimed -- so recording the request would record the wrong thing.

    The fixture DISCRIMINATES: the gather is asked for `meta` and the model
    hands back a CPU tensor. An implementation reading the `device` ARGUMENT
    records `meta:UNKNOWN_CLASS`; one reading the encoder's OUTPUT records the
    cpu class. They cannot both pass.
    """
    from src.analysis.artifact_identity import GATHER_DEVICE_KEY, device_class
    from src.analysis.feature_ap import sparse_activations

    p = _toy_cache(tmp_path / "cache.h5")
    rec: dict = {}
    sparse_activations(_FakeSAE(4, 3), p, None, torch.device("meta"), record=rec)
    assert rec[GATHER_DEVICE_KEY] == device_class("cpu")
    assert rec[GATHER_DEVICE_KEY] != device_class("meta")


def test_the_gather_records_NOTHING_when_no_recorder_is_passed(tmp_path):
    """The recorder is a pure out-parameter: five of the six call sites do not
    want it, and none of them may change behaviour for its sake."""
    from src.analysis.feature_ap import sparse_activations

    p = _toy_cache(tmp_path / "cache.h5")
    feat, pos, seg = sparse_activations(_FakeSAE(4, 3), p, None, "cpu")
    assert feat.numel() == 8
    assert seg.numel() == 3


# ------------------------------------------- eval-cache identity (CQ, A4) ----

_CFG = {"model": {"name": "facebook/esm2_t33_650M_UR50D", "d_model": 1280,
                  "layer": 24, "apply_final_layer_norm": False, "max_len": 1022}}


def _cache(tmp_path, **over):
    attrs = {"model": "facebook/esm2_t33_650M_UR50D", "layer": 24,
             "final_layer_norm_applied": False, "max_len": 1022,
             "dtype": "float16", "subsampled": False, "seed": 0}
    attrs.update(over)
    p = tmp_path / "eval.h5"
    with h5py.File(p, "w") as hf:
        hf.create_dataset("activations", data=np.zeros((4, 8), np.float16))
        for k, v in attrs.items():
            if v is None:
                continue
            hf.attrs[k] = v
    return p


def test_expected_attrs_come_from_the_config_under_the_cache_names():
    e = expected_eval_cache_attrs_from_config(_CFG)
    assert set(e) == set(EVAL_CACHE_IDENTITY_KEYS)
    assert e["final_layer_norm_applied"] is False      # apply_final_layer_norm
    assert e["layer"] == 24 and e["max_len"] == 1022


def test_eval_cache_identity_agrees_on_a_matching_cache(tmp_path):
    """Positive control: numpy scalars and bytes from h5 compare equal to the
    config's Python values."""
    p = _cache(tmp_path)
    rep = assert_eval_cache_identity(p, expected_eval_cache_attrs_from_config(_CFG),
                                     label="config")
    assert rep["verdict"] == "AGREE"
    assert all(v["agree"] for v in rep["per_key"].values())
    assert eval_cache_identity(p)["layer"] == 24


def test_eval_cache_identity_refuses_a_layer_mismatch(tmp_path):
    """The founding hazard: a layer-23 cache gathered silently. Both values
    are in the message."""
    p = _cache(tmp_path, layer=23)
    with pytest.raises(EvalCacheIdentityMismatch) as ei:
        assert_eval_cache_identity(p, expected_eval_cache_attrs_from_config(_CFG),
                                   label="config")
    msg = str(ei.value)
    assert "layer" in msg and "23" in msg and "24" in msg, msg


def test_eval_cache_identity_refuses_a_normalisation_mismatch(tmp_path):
    p = _cache(tmp_path, final_layer_norm_applied=True)
    with pytest.raises(EvalCacheIdentityMismatch, match="final_layer_norm_applied"):
        assert_eval_cache_identity(p, expected_eval_cache_attrs_from_config(_CFG),
                                   label="config")


def test_eval_cache_identity_absent_key_is_not_agreement(tmp_path):
    """Gotcha 15, both directions: a cache missing `max_len`, and an
    expectation missing `model`, are each a mismatch."""
    p = _cache(tmp_path, max_len=None)
    with pytest.raises(EvalCacheIdentityMismatch, match="max_len"):
        assert_eval_cache_identity(p, expected_eval_cache_attrs_from_config(_CFG),
                                   label="config")
    p2 = _cache(tmp_path)
    exp = expected_eval_cache_attrs_from_config(_CFG)
    del exp["model"]
    with pytest.raises(EvalCacheIdentityMismatch, match="model"):
        assert_eval_cache_identity(p2, exp, label="partial")
