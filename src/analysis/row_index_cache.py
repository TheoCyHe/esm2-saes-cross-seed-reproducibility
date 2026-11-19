"""
Persist the per-seed row-major index (sparse_ap_index.build_row_index) so
repeated AP runs against the SAME seed's activations skip both the SAE
forward pass and the row-index build. Measured (job 19958667): gather +
row-index build is 239s of a 256s per-seed cost -- the AP arithmetic itself
is now cheap (P1). P2's calibration harness will run dozens of candidate
rules against the same 20-seed activations; without this, each run repeats
that 239s x 20 = ~80 min for no reason.

CACHE CORRECTNESS. A stale cache keyed wrongly is a silent correctness bug of
exactly the class this project keeps producing (three of them in this
session alone: HDF5 '/' in group names, truncated-instance floor inflation,
UniProt ordinal fragmentation -- all found by RUNNING things, not reasoning
about them). The cache key must change whenever anything that could change
the row-index's CONTENT changes:
  - git SHA of the code that builds it (a bug fix must invalidate old caches)
  - the checkpoint's own content hash (sha256_16, same convention as
    b_dec_sha256_16 in src/sae/train.py) -- catches a wrong-seed mixup
  - the eval cache's path + size + mtime (a fast fingerprint, not a full
    hash of a 45GB file -- catches an accidentally-stale or swapped cache
    without paying to hash the whole thing)
  - scored_row_set (the row-index is sized and built differently per subset)
  - k (changes how many entries fire per row, hence the index's shape)
  - THE GATHER'S DEVICE CLASS (gotcha 36; PREREGISTRATION amendment BJ)
A cache MISS (key not found) is always safe -- it just recomputes. The risk
is only a cache HIT returning something that does not match a cold build,
which is exactly what test_row_index_cache.py's round-trip test checks.

=============================================================================
THE DEVICE CLASS IS PART OF THE KEY (amendment BJ; gotcha 36b)
=============================================================================
**The key omitted the gather device while the gather demonstrably depends on
it**, so this module's own rule above -- the key must change whenever anything
that could change the index's CONTENT changes -- was violated by the key
itself. Measured (gotcha 36, by recomputing the AP artifact on CPU and
diffing, not from a hash): `fire_counts` differ in 9,018 of 204,800
(seed, feature) pairs (4.40%) between CPU and `rtx_5000_ada` for the SAME
checkpoint on the SAME rows. `nnz` is identical because TopK takes exactly `k`
per row either way; WHICH features fire is what moves, because near-tie
pre-activations swap under a CPU-versus-GPU float32 matmul. `feat_by_row` is
precisely that feature set, so a cross-device HIT served an index built by a
different selection.

**A MISS WAS ALWAYS SAFE. THE HAZARD WAS A HIT**, and the mitigation in place
until now LABELLED it rather than removing it: `null_pilot` recorded
`UNKNOWN_CACHED_ROW_INDEX` on a hit because it could not know the device the
cached gather ran on. That is honest and it is not a fix.

**WHY THE ENCODER'S PLACEMENT AND NOT THE `--device` REQUEST.** A job can
request a GPU and run on CPU -- that is what got four tasks reclaimed -- so the
requested string is not the measurement. The key is computed BEFORE the gather
(it has to be: it decides hit versus miss), so the post-hoc value
`sparse_activations` reads off `z.device` is not available yet. What IS
available and is not a request is **where the encoder's own weights sit**:
`encode_pre` is `(x - b_dec) @ W_enc.T + b_enc`, so the matmul runs on
`W_enc`'s device and nowhere else. `encoder_device_class` reads that.
**And the two are then RECONCILED**: after a cold build the measured gather
device is compared against the one the key was computed from, and a
disagreement raises `GatherDeviceDisagreesWithKey` rather than writing an entry
under a key that does not describe it. A check that cannot fail is not a check
(gotcha 13), so that reconciliation is tested by planting a disagreement.

**NOTHING IS BACK-FILLED.** Pre-existing cache entries carry no device in their
meta and are not given one -- an inferred device written where a measured one
belongs is indistinguishable from a measurement afterwards, and the label is
the first thing lost. Under the new key those entries are simply never looked
up again: the key is a sha256 over a parts list that has GAINED a field, so
every key changes and every old file is a MISS. A miss recomputes, which is
safe. They are not deleted either -- `cached_device_class` reports
`ABSENT` for them, never the running job's device (gotcha 15).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import torch

from src.analysis.artifact_identity import (EVAL_CACHE_IDENTITY_KEYS,
                                            GATHER_DEVICE_KEY,
                                            assert_eval_cache_identity,
                                            device_class)
from src.analysis.feature_ap import sparse_activations
from src.analysis.sparse_ap_index import build_row_index, narrow_row_index
from src.sae.train import TopKSAE, git_sha

#: What `cached_device_class` returns for an entry written before the device
#: entered the meta. **Reported as absent, never as matching the running
#: device** (gotcha 15): both-absent is not agreement.
DEVICE_ABSENT = "ABSENT"


class GatherDeviceDisagreesWithKey(RuntimeError):
    """The gather ran on a device class other than the one the cache key was
    computed from.

    The key is computed from the ENCODER'S PLACEMENT before the gather; the
    gather then reports where it ACTUALLY ran. Those must agree, because the
    entry is about to be stored under a key that asserts the first while
    containing the second. Raised rather than silently preferring either: an
    entry keyed by one device and built on another is exactly the cross-device
    hit this key exists to prevent, moved inside a single job.
    """


def checkpoint_sha256_16(ckpt_dir: Path) -> str:
    return hashlib.sha256((ckpt_dir / "model.pt").read_bytes()).hexdigest()[:16]


#: `EVAL_CACHE_CHECK_KEY` in `record` -- what the identity check did on a cold
#: build. "NOT_CHECKED_NO_META" is reported, never treated as agreement.
EVAL_CACHE_CHECK_KEY = "eval_cache_identity_check"


def assert_eval_cache_matches_checkpoint(eval_cache: Path, ckpt_dir: Path):
    """Amendment CQ, A4: before a cold gather, the eval cache's identity attrs
    must EQUAL the train cache's that the SAE was fit on, as recorded in the
    checkpoint's `meta.json` `activations_attrs`.

    Returns the per-key report on agreement; raises
    `EvalCacheIdentityMismatch` (from `artifact_identity`) otherwise. A
    checkpoint directory with no `meta.json` has no expectation to check
    against, so the result is the string "NOT_CHECKED_NO_META" -- reported,
    never read as agreement (gotcha 15).
    """
    import json as _json
    meta_path = ckpt_dir / "meta.json"
    if not meta_path.exists():
        return "NOT_CHECKED_NO_META"
    meta = _json.loads(meta_path.read_text())
    expected = {k: meta.get("activations_attrs", {}).get(k)
                for k in EVAL_CACHE_IDENTITY_KEYS
                if k in meta.get("activations_attrs", {})}
    return assert_eval_cache_identity(
        eval_cache, expected, label=f"{meta_path}:activations_attrs")


def encoder_device_class(model: TopKSAE) -> str:
    """The device class the gather WILL run on, read off the encoder itself.

    `encode_pre` is `(x - b_dec) @ W_enc.T + b_enc`, so the matmul executes on
    `W_enc`'s device regardless of what any `--device` string says. Reading the
    weight is therefore a property of the model as loaded, not of the request
    -- which is the same distinction `sparse_activations` draws when it takes
    the gather device off `z.device` rather than off its `device` argument.

    Kept separate from that post-hoc read rather than replacing it: this one is
    available BEFORE the gather (the key needs it to decide hit versus miss)
    and the other is the measurement. `compute_or_load_row_index` reconciles
    them and refuses on a disagreement.
    """
    return device_class(model.W_enc)


def cache_key(ckpt_dir: Path, eval_cache: Path, scored_row_set: str, k: int,
              gather_device_class: str) -> str:
    """The cache key. `gather_device_class` is REQUIRED and has no default.

    A default would let a call site omit it and silently get the old,
    device-blind key back -- which is the defect, reintroduced by convenience
    at exactly the place it is hardest to see. Every caller states it.
    """
    st = eval_cache.stat()
    parts = [
        git_sha(),
        checkpoint_sha256_16(ckpt_dir),
        str(eval_cache.resolve()), str(st.st_size), str(int(st.st_mtime)),
        scored_row_set, str(k),
        gather_device_class,
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def cache_path(cache_dir: Path, ckpt_dir: Path, eval_cache: Path,
               scored_row_set: str, k: int, gather_device_class: str) -> Path:
    key = cache_key(ckpt_dir, eval_cache, scored_row_set, k,
                    gather_device_class)
    return cache_dir / f"row_index_{ckpt_dir.name}_{key}.h5"


def cached_device_class(path: Path) -> str:
    """The device class a stored entry RECORDS, or `DEVICE_ABSENT`.

    An entry written before amendment BJ has no device in its meta. That is
    reported as `ABSENT` -- it is UNRECORDED, not equal to whatever the reading
    job is running on, and the difference is the one `compare_device_class`
    exists to keep (gotcha 15). Never infer, never default.
    """
    with h5py.File(path, "r") as hf:
        if GATHER_DEVICE_KEY not in hf.attrs:
            return DEVICE_ABSENT
        v = hf.attrs[GATHER_DEVICE_KEY]
    return v.decode() if isinstance(v, bytes) else str(v)


def save_row_index(path: Path, feat_by_row, rank_by_row, row_offsets,
                   fire_counts, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as hf:
        hf.create_dataset("feat_by_row", data=feat_by_row.cpu().numpy(),
                          compression="gzip")
        hf.create_dataset("rank_by_row", data=rank_by_row.cpu().numpy(),
                          compression="gzip")
        hf.create_dataset("row_offsets", data=row_offsets.cpu().numpy(),
                          compression="gzip")
        hf.create_dataset("fire_counts", data=fire_counts.cpu().numpy())
        hf.attrs.update(**meta)
    tmp.replace(path)          # atomic-ish: no reader ever sees a partial file


def load_row_index(path: Path, device: str, narrow_features: int | None = None):
    """`narrow_features` (amendment CT, §CT29-D2): when given, the index is
    narrowed on the host (`narrow_row_index`, which refuses rather than wrap)
    before it moves to `device`. The entry on disk is never narrowed."""
    with h5py.File(path, "r") as hf:
        feat_by_row = torch.from_numpy(hf["feat_by_row"][:])
        rank_by_row = torch.from_numpy(hf["rank_by_row"][:])
        row_offsets = torch.from_numpy(hf["row_offsets"][:])
        fire_counts = torch.from_numpy(hf["fire_counts"][:])
    if narrow_features is not None:
        feat_by_row, rank_by_row = narrow_row_index(feat_by_row, rank_by_row,
                                                    narrow_features)
    return (feat_by_row.to(device), rank_by_row.to(device),
            row_offsets.to(device), fire_counts)


def compute_or_load_row_index(
    model: TopKSAE, ckpt_dir: Path, eval_cache: Path, rows,
    scored_row_set: str, dict_size: int, k: int, device: str, cache_dir: Path,
    record: dict | None = None, narrow_features: int | None = None,
):
    """Returns (feat_by_row, rank_by_row, row_offsets, fire_counts,
    from_cache: bool). `rows` is the scored-row-set index array (or None for
    'all'), passed straight to sparse_activations on a cache miss.

    `record`, when given, receives `GATHER_DEVICE_KEY` -> the device class of
    the gather THIS INDEX CAME FROM: measured here on a cold build, and read
    from the entry's own meta on a hit. An entry that records none yields
    `DEVICE_ABSENT`, which a caller must report as absent rather than as this
    job's device.

    **AN OUT-PARAMETER, not a sixth return value**, matching
    `sparse_activations`: most call sites do not want the device, and a helper
    that needs to expose one more thing adds a channel rather than growing its
    return arity (gotcha 27's note).
    """
    gather_dc = encoder_device_class(model)
    path = cache_path(cache_dir, ckpt_dir, eval_cache, scored_row_set, k,
                      gather_dc)
    if path.exists():
        # A HIT IS NOW SAME-DEVICE-CLASS BY THE KEY. The entry is still asked
        # what it records rather than told: a file placed here by hand, or
        # written before the device entered the meta, must not be described by
        # this job's device.
        if record is not None:
            record[GATHER_DEVICE_KEY] = cached_device_class(path)
        return (*load_row_index(path, device, narrow_features), True)

    # amendment CQ, A4: identity before a cold gather (a hit re-opens nothing).
    eval_check = assert_eval_cache_matches_checkpoint(Path(eval_cache),
                                                      Path(ckpt_dir))
    if record is not None:
        record[EVAL_CACHE_CHECK_KEY] = (eval_check if isinstance(eval_check, str)
                                        else eval_check["verdict"])
    gather_record: dict = {}
    feat, pos, seg_start = sparse_activations(model, eval_cache, rows, device,
                                              record=gather_record)
    measured_dc = gather_record.get(GATHER_DEVICE_KEY)
    # THE RECONCILIATION. The key asserted `gather_dc` before the gather ran;
    # the gather reports where it actually ran. Storing an entry whose key says
    # one device and whose contents came from another would recreate the
    # cross-device hit inside a single job.
    if measured_dc is not None and measured_dc != gather_dc:
        raise GatherDeviceDisagreesWithKey(
            f"the cache key was computed from the encoder's placement "
            f"({gather_dc}) and the gather reported {measured_dc}. The index "
            f"about to be stored does not match the key that would address "
            f"it, so it is not stored (gotcha 36)")
    if record is not None:
        record[GATHER_DEVICE_KEY] = gather_dc

    n_rows = len(rows) if rows is not None else \
        int(h5py.File(eval_cache, "r").attrs.get("n_rows",
            h5py.File(eval_cache, "r")["activations"].shape[0]))
    fire_counts = torch.bincount(feat, minlength=dict_size)
    feat, pos, seg_start = feat.cpu(), pos.cpu(), seg_start.cpu()
    feat_by_row, rank_by_row, row_offsets = build_row_index(feat, pos, seg_start, n_rows)

    save_row_index(path, feat_by_row, rank_by_row, row_offsets, fire_counts, meta=dict(
        git_sha=git_sha(), checkpoint=str(ckpt_dir), checkpoint_sha256_16=checkpoint_sha256_16(ckpt_dir),
        eval_cache=str(eval_cache), scored_row_set=scored_row_set, k=k, n_rows=n_rows,
        **{GATHER_DEVICE_KEY: gather_dc},
    ))
    if narrow_features is not None:
        feat_by_row, rank_by_row = narrow_row_index(feat_by_row, rank_by_row,
                                                    narrow_features)
    if device != "cpu":
        feat_by_row, rank_by_row, row_offsets = (
            feat_by_row.to(device), rank_by_row.to(device), row_offsets.to(device))
    return feat_by_row, rank_by_row, row_offsets, fire_counts, False
