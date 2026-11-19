"""
Tests for row_index_cache.py -- a stale cache keyed wrongly is a silent
correctness bug, so these check both "a hit reproduces a cold build
bit-identically" AND "different inputs never collide on the same cache
entry", not just the happy path.

AMENDMENT BJ adds the gather's DEVICE CLASS to the key (gotcha 36b). The
device-class fixtures below are written so the defect can be EXHIBITED, not
only its absence asserted: `test_the_pre_BJ_key_form_COLLIDES...` recomputes
the old key form from a parts list WRITTEN DOWN HERE -- never imported from
the module under test, because a test that derives its expectation from the
thing it is checking cannot detect a change to that thing (gotcha 32b) -- and
shows the collision the new key removes. A test that cannot exhibit the defect
is not evidence the defect is fixed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from src.analysis.artifact_identity import (GATHER_DEVICE_KEY,
                                            EvalCacheIdentityMismatch)
from src.analysis.row_index_cache import (
    DEVICE_ABSENT,
    GatherDeviceDisagreesWithKey,
    cache_key,
    cache_path,
    cached_device_class,
    checkpoint_sha256_16,
    compute_or_load_row_index,
    encoder_device_class,
)
from src.sae.train import TopKSAE, git_sha

CPU = "cpu:AMD64"          # a device-class STRING; the fixtures never run CUDA
GPU = "cuda:NVIDIA RTX 5000 Ada Generation:sm_89"


def _make_fixture(tmp_path: Path, d_model=8, dict_size=32, k=4, n_rows=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_rows, d_model)).astype(np.float16)
    cache = tmp_path / "eval.h5"
    with h5py.File(cache, "w") as hf:
        hf.create_dataset("activations", data=X)
        hf.attrs["n_rows"] = n_rows

    ckpt = tmp_path / f"seed_{seed}"
    ckpt.mkdir(exist_ok=True)
    m = TopKSAE(d_model, dict_size, k, b_dec=torch.zeros(d_model))
    torch.save(m.state_dict(), ckpt / "model.pt")
    return cache, ckpt, m, d_model, dict_size, k, n_rows


def test_cache_hit_reproduces_cold_build_bit_identically(tmp_path):
    cache, ckpt, model, d_model, dict_size, k, n_rows = _make_fixture(tmp_path)
    cache_dir = tmp_path / "cache_store"

    dc = encoder_device_class(model)
    fbr1, rbr1, ro1, fc1, from_cache1 = compute_or_load_row_index(
        model, ckpt, cache, None, "all", dict_size, k, "cpu", cache_dir)
    assert from_cache1 is False                       # first call: cold build
    assert cache_path(cache_dir, ckpt, cache, "all", k, dc).exists()

    fbr2, rbr2, ro2, fc2, from_cache2 = compute_or_load_row_index(
        model, ckpt, cache, None, "all", dict_size, k, "cpu", cache_dir)
    assert from_cache2 is True                         # second call: cache hit

    assert torch.equal(fbr1, fbr2)
    assert torch.equal(rbr1, rbr2)
    assert torch.equal(ro1, ro2)
    assert torch.equal(fc1, fc2)


def test_different_checkpoints_never_collide_on_the_same_cache_entry(tmp_path):
    """THE correctness-critical case: if the cache key omitted the
    checkpoint's content hash, two DIFFERENT seeds' row-indexes could
    collide on the same file, silently serving seed A's activations under
    seed B's name. Two checkpoints with DIFFERENT weights must get DIFFERENT
    cache keys."""
    cache, ckpt0, model0, d_model, dict_size, k, n_rows = _make_fixture(
        tmp_path, seed=0)
    ckpt1 = tmp_path / "seed_1"
    ckpt1.mkdir()
    model1 = TopKSAE(d_model, dict_size, k, b_dec=torch.zeros(d_model))
    torch.save(model1.state_dict(), ckpt1 / "model.pt")

    assert checkpoint_sha256_16(ckpt0) != checkpoint_sha256_16(ckpt1)
    k0 = cache_key(ckpt0, cache, "all", k, CPU)
    k1 = cache_key(ckpt1, cache, "all", k, CPU)
    assert k0 != k1

    cache_dir = tmp_path / "cache_store"
    _, _, _, _, hit0 = compute_or_load_row_index(
        model0, ckpt0, cache, None, "all", dict_size, k, "cpu", cache_dir)
    _, _, _, _, hit1 = compute_or_load_row_index(
        model1, ckpt1, cache, None, "all", dict_size, k, "cpu", cache_dir)
    assert hit0 is False and hit1 is False             # both cold -- no collision
    assert cache_path(cache_dir, ckpt0, cache, "all", k, CPU) != \
           cache_path(cache_dir, ckpt1, cache, "all", k, CPU)


def test_different_scored_row_set_or_k_never_collide(tmp_path):
    cache, ckpt, model, d_model, dict_size, k, n_rows = _make_fixture(tmp_path)
    assert cache_key(ckpt, cache, "all", k, CPU) != \
           cache_key(ckpt, cache, "shared_experimental", k, CPU)
    assert cache_key(ckpt, cache, "all", k, CPU) != \
           cache_key(ckpt, cache, "all", k + 1, CPU)


def test_content_change_invalidates_the_key_even_with_the_same_path(tmp_path):
    """A checkpoint file that gets REWRITTEN (e.g. a corrected training run
    saved over the same seed directory) must not silently reuse a stale
    cache entry keyed only on the path."""
    cache, ckpt, model, d_model, dict_size, k, n_rows = _make_fixture(tmp_path)
    key_before = cache_key(ckpt, cache, "all", k, CPU)

    model2 = TopKSAE(d_model, dict_size, k, b_dec=torch.zeros(d_model))
    torch.save(model2.state_dict(), ckpt / "model.pt")   # overwrite same path
    key_after = cache_key(ckpt, cache, "all", k, CPU)
    assert key_before != key_after


# =========================================================================
# AMENDMENT BJ -- the gather's DEVICE CLASS is part of the key (gotcha 36b)
# =========================================================================

def _pre_bj_key(ckpt_dir: Path, eval_cache: Path, scored_row_set: str,
                k: int) -> str:
    """THE KEY AS IT WAS BEFORE AMENDMENT BJ, written down here rather than
    imported.

    This is the whole discrimination argument. A fixture that asked the CURRENT
    module for the old behaviour would be deriving its expectation from the
    thing under test and could not detect a change to it (gotcha 32b). The
    parts list below is transcribed from the pre-BJ source; it takes no device
    argument, because that is exactly the defect.
    """
    st = eval_cache.stat()
    parts = [
        git_sha(),
        checkpoint_sha256_16(ckpt_dir),
        str(eval_cache.resolve()), str(st.st_size), str(int(st.st_mtime)),
        scored_row_set, str(k),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def test_the_pre_BJ_key_form_COLLIDES_across_device_classes(tmp_path):
    """THE FIXTURE IS SHOWN TO DISCRIMINATE, BEFORE ANYTHING IS ASSERTED FIXED.

    Two runs identical in every respect except the device class the gather ran
    on. Under the pre-BJ key they address the SAME cache entry -- a CPU job
    silently loading a GPU-computed firing index, which gotcha 36 measured as a
    different feature set in 4.40% of (seed, feature) pairs. The collision is
    exhibited here so that the next test's non-collision is evidence rather
    than a tautology about a key that never had the field.
    """
    cache, ckpt, model, d_model, dict_size, k, n_rows = _make_fixture(tmp_path)
    cache_dir = tmp_path / "cache_store"

    # TWO RUNS. Identical checkpoint, identical eval cache, identical
    # scored_row_set and k. The ONLY difference is the device class the gather
    # ran on -- which is the axis gotcha 36 measured as changing the answer.
    run_on_cpu = dict(ckpt_dir=ckpt, eval_cache=cache, scored_row_set="all", k=k)
    run_on_gpu = dict(ckpt_dir=ckpt, eval_cache=cache, scored_row_set="all", k=k)

    old_cpu = _pre_bj_key(**run_on_cpu)
    old_gpu = _pre_bj_key(**run_on_gpu)
    assert old_cpu == old_gpu, (
        "the pre-BJ key form was expected to collide across device classes; "
        "if it does not, this fixture is not exhibiting the defect and the "
        "next test's non-collision proves nothing")
    old_path_cpu = cache_dir / f"row_index_{ckpt.name}_{old_cpu}.h5"
    old_path_gpu = cache_dir / f"row_index_{ckpt.name}_{old_gpu}.h5"
    assert old_path_cpu == old_path_gpu          # one file serves both runs

    # THE SAME TWO RUNS UNDER THE NEW KEY. Stated here rather than only in the
    # next test, so the before/after sits in one place.
    assert cache_key(**run_on_cpu, gather_device_class=CPU) != \
           cache_key(**run_on_gpu, gather_device_class=GPU)


def test_two_keys_differing_ONLY_in_device_class_do_not_collide(tmp_path):
    """B3(b). Everything else held fixed; only the device class moves."""
    cache, ckpt, model, d_model, dict_size, k, n_rows = _make_fixture(tmp_path)

    cpu_key = cache_key(ckpt, cache, "all", k, CPU)
    gpu_key = cache_key(ckpt, cache, "all", k, GPU)
    assert cpu_key != gpu_key

    # And the collision the previous test exhibited is what this removes: the
    # SAME arguments under the old form give one key, under the new form two.
    old = _pre_bj_key(ckpt, cache, "all", k)
    assert len({old}) == 1
    assert len({cpu_key, gpu_key}) == 2

    # The path follows the key, so the two never address one file.
    cache_dir = tmp_path / "cache_store"
    assert cache_path(cache_dir, ckpt, cache, "all", k, CPU) != \
           cache_path(cache_dir, ckpt, cache, "all", k, GPU)


def test_the_key_requires_a_device_class_with_no_default(tmp_path):
    """A DEFAULT WOULD REINTRODUCE THE DEFECT SILENTLY. A call site that
    omitted the argument would get a single device-blind key back, which is
    the pre-BJ behaviour arriving through convenience rather than through a
    decision. It must be a TypeError, not a fallback."""
    cache, ckpt, model, d_model, dict_size, k, n_rows = _make_fixture(tmp_path)
    with pytest.raises(TypeError):
        cache_key(ckpt, cache, "all", k)              # type: ignore[call-arg]
    with pytest.raises(TypeError):
        cache_path(tmp_path / "store", ckpt, cache, "all", k)  # type: ignore[call-arg]


def test_the_key_comes_from_the_ENCODER_not_from_the_device_argument(tmp_path):
    """Requirement 2. A job can request a GPU and run on CPU, so the `--device`
    string is not the measurement. `encoder_device_class` reads `W_enc`, which
    is where `encode_pre`'s matmul actually executes.

    **THE FIXTURE HAS TO MOVE THE ENCODER SOMEWHERE THE REQUEST IS NOT, or it
    cannot discriminate.** The first version asserted only that a CPU model
    reports `cpu:<machine>` -- which a hardcoded `device_class("cpu")` also
    satisfies, so the mutation SURVIVED. That is BA17 item 4's shape exactly: a
    fixture whose two sides are equal by accident tests nothing. `meta` is a
    real torch device that needs no hardware and is neither of the two classes
    gotcha 36 is about, so it separates "reads the weight" from "returns a
    constant" on a machine with no GPU.

    The expected strings are WRITTEN DOWN rather than computed from
    `device_class` (gotcha 32b): a test that asks the code under test what to
    expect cannot detect a change to it.
    """
    cache, ckpt, model, d_model, dict_size, k, n_rows = _make_fixture(tmp_path)
    import platform

    assert encoder_device_class(model) == f"cpu:{platform.machine()}"

    # Move the ENCODER and nothing else. Nothing is told about a device; the
    # answer must follow the weight.
    on_meta = model.to("meta")
    assert encoder_device_class(on_meta) == "meta:UNKNOWN_CLASS"

    # ... and the key follows it, so two placements never address one entry.
    assert cache_key(ckpt, cache, "all", k, f"cpu:{platform.machine()}") != \
           cache_key(ckpt, cache, "all", k, "meta:UNKNOWN_CLASS")


def test_a_fresh_entry_RECORDS_its_gather_device_in_the_meta(tmp_path):
    """Requirement 3. The entry states its own provenance."""
    cache, ckpt, model, d_model, dict_size, k, n_rows = _make_fixture(tmp_path)
    cache_dir = tmp_path / "cache_store"
    rec: dict = {}
    compute_or_load_row_index(model, ckpt, cache, None, "all", dict_size, k,
                              "cpu", cache_dir, record=rec)
    dc = encoder_device_class(model)
    path = cache_path(cache_dir, ckpt, cache, "all", k, dc)
    with h5py.File(path, "r") as hf:
        stored = hf.attrs[GATHER_DEVICE_KEY]
    assert (stored.decode() if isinstance(stored, bytes) else stored) == dc
    assert rec[GATHER_DEVICE_KEY] == dc
    assert cached_device_class(path) == dc


def test_an_entry_with_NO_recorded_device_is_reported_ABSENT_not_matching(tmp_path):
    """B3(d). A pre-BJ entry records no device. `cached_device_class` must say
    so rather than returning the reading job's device -- absence is not
    agreement (gotcha 15), and this is the branch through which an inference
    would become indistinguishable from a measurement."""
    cache, ckpt, model, d_model, dict_size, k, n_rows = _make_fixture(tmp_path)
    cache_dir = tmp_path / "cache_store"
    compute_or_load_row_index(model, ckpt, cache, None, "all", dict_size, k,
                              "cpu", cache_dir)
    dc = encoder_device_class(model)
    path = cache_path(cache_dir, ckpt, cache, "all", k, dc)

    # Strip the attr, reproducing an entry written before the field existed.
    with h5py.File(path, "a") as hf:
        del hf.attrs[GATHER_DEVICE_KEY]
    assert cached_device_class(path) == DEVICE_ABSENT
    assert cached_device_class(path) != dc              # NOT the running device

    # And a consumer asking through the record gets the absence, not a device.
    rec: dict = {}
    _, _, _, _, from_cache = compute_or_load_row_index(
        model, ckpt, cache, None, "all", dict_size, k, "cpu", cache_dir,
        record=rec)
    assert from_cache is True
    assert rec[GATHER_DEVICE_KEY] == DEVICE_ABSENT


def test_a_gather_on_a_device_other_than_the_KEYs_is_REFUSED(tmp_path,
                                                            monkeypatch):
    """The reconciliation, and it is a check that can fail.

    The key is computed from the encoder's placement BEFORE the gather; the
    gather then reports where it ran. A disagreement means the entry about to
    be stored is not the entry the key addresses -- the cross-device hit, moved
    inside one job -- so it raises rather than being written.
    """
    cache, ckpt, model, d_model, dict_size, k, n_rows = _make_fixture(tmp_path)
    cache_dir = tmp_path / "cache_store"

    import src.analysis.row_index_cache as ric
    real = ric.sparse_activations

    def lying_gather(m, cp, rows, device, chunk=16384, record=None):
        out = real(m, cp, rows, device, chunk=chunk, record=record)
        if record is not None:
            record[GATHER_DEVICE_KEY] = GPU          # ran somewhere else
        return out

    monkeypatch.setattr(ric, "sparse_activations", lying_gather)
    with pytest.raises(GatherDeviceDisagreesWithKey):
        ric.compute_or_load_row_index(model, ckpt, cache, None, "all",
                                      dict_size, k, "cpu", cache_dir)
    # Nothing was stored under a key that does not describe it.
    assert not list(cache_dir.glob("*.h5")) if cache_dir.exists() else True


# ---------------------------------------------------------------------------
# Amendment CQ, A4: the eval cache's identity is asserted against the
# checkpoint's `activations_attrs` before a cold gather.
# ---------------------------------------------------------------------------

_ATTRS = {"model": "facebook/esm2_t33_650M_UR50D", "layer": 24,
          "final_layer_norm_applied": False, "max_len": 1022}


def _label_cache_and_ckpt(cache, ckpt, cache_over=None, meta_attrs=_ATTRS):
    from src.analysis.row_index_cache import EVAL_CACHE_CHECK_KEY  # noqa: F401
    with h5py.File(cache, "a") as hf:
        for k, v in {**_ATTRS, **(cache_over or {})}.items():
            hf.attrs[k] = v
    if meta_attrs is not None:
        (ckpt / "meta.json").write_text(json.dumps(
            {"seed": 0, "activations_attrs": meta_attrs}))


def test_cold_build_checks_the_eval_cache_against_the_checkpoint(tmp_path):
    """Positive control: matching attrs -> the build proceeds and the record
    says AGREE."""
    from src.analysis.row_index_cache import EVAL_CACHE_CHECK_KEY
    cache, ckpt, model, d_model, dict_size, k, n_rows = _make_fixture(tmp_path)
    _label_cache_and_ckpt(cache, ckpt)
    rec: dict = {}
    *_, from_cache = compute_or_load_row_index(
        model, ckpt, cache, None, "all", dict_size, k, "cpu",
        tmp_path / "store", record=rec)
    assert from_cache is False
    assert rec[EVAL_CACHE_CHECK_KEY] == "AGREE"


def test_cold_build_refuses_an_eval_cache_that_mismatches_the_checkpoint(tmp_path):
    """A cache from layer 23 against an SAE trained on layer 24: refuse before
    gathering, store nothing."""
    cache, ckpt, model, d_model, dict_size, k, n_rows = _make_fixture(tmp_path)
    _label_cache_and_ckpt(cache, ckpt, cache_over={"layer": 23})
    store = tmp_path / "store"
    with pytest.raises(EvalCacheIdentityMismatch, match="layer"):
        compute_or_load_row_index(model, ckpt, cache, None, "all", dict_size,
                                  k, "cpu", store)
    assert not store.exists() or not any(store.iterdir())


def test_cold_build_without_a_meta_json_reports_the_check_as_not_done(tmp_path):
    """No `meta.json` means no expectation; that is REPORTED, never read as
    agreement (gotcha 15)."""
    from src.analysis.row_index_cache import EVAL_CACHE_CHECK_KEY
    cache, ckpt, model, d_model, dict_size, k, n_rows = _make_fixture(tmp_path)
    rec: dict = {}
    compute_or_load_row_index(model, ckpt, cache, None, "all", dict_size, k,
                              "cpu", tmp_path / "store", record=rec)
    assert rec[EVAL_CACHE_CHECK_KEY] == "NOT_CHECKED_NO_META"
