"""
Tests for src/sae/train.py's TopK SAE plumbing -- architecture shapes,
sparsity, decoder renormalization, and the reproducibility hygiene
CLAUDE.md requires (seeded dataloader workers). No GPU needed.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import random
import torch

from src.sae.train import (
    ActivationDataset,
    ShuffledBlockActivations,
    TopKSAE,
    _jsonable,
    reduce_se,
    set_all_seeds,
    worker_init_fn,
)


def _make_cache(path, n_rows, d=4, start=0):
    """Cache whose row i holds the value (start + i) in every column, so a row
    can be identified by its contents."""
    data = (np.arange(start, start + n_rows, dtype=np.float32)[:, None]
            * np.ones((1, d), dtype=np.float32)).astype(np.float16)
    with h5py.File(path, "w") as hf:
        hf.create_dataset("activations", data=data)
    return data


def test_topk_sae_shapes_and_sparsity():
    d_model, dict_size, k, batch = 16, 64, 5, 8
    model = TopKSAE(d_model, dict_size, k)
    x = torch.randn(batch, d_model)

    x_hat, z, pre = model(x)

    assert x_hat.shape == (batch, d_model)
    assert z.shape == (batch, dict_size)
    assert pre.shape == (batch, dict_size)
    # at most k nonzero activations per row (ReLU can zero out ties at 0)
    assert (z > 0).sum(dim=-1).max().item() <= k
    # pre-activations are the pre-TopK ReLU: non-negative, and never fewer
    # nonzeros than the gated z (the gate only removes)
    assert (pre >= 0).all()
    assert (pre > 0).sum() >= (z > 0).sum()


def test_b_dec_is_frozen_shared_buffer():
    """LOCKED invariant: b_dec is frozen, excluded from the optimizer, shared
    by value, and saved with the checkpoint. If this test fails, b_dec could
    drift per seed -- the exact thing CLAUDE.md forbids."""
    bd = torch.arange(8, dtype=torch.float32)
    m = TopKSAE(d_model=8, dict_size=16, k=4, b_dec=bd)

    torch.testing.assert_close(m.b_dec, bd)
    assert "b_dec" not in dict(m.named_parameters())   # Adam never sees it
    assert not m.b_dec.requires_grad                   # no gradient
    assert "b_dec" in m.state_dict()                   # travels with checkpoint

    # constructor must copy, not alias: mutating the source can't reach into a
    # model that is supposed to hold the one shared value.
    bd[0] = 999.0
    assert m.b_dec[0].item() == 0.0


def test_b_dec_wrong_shape_raises():
    with pytest.raises(ValueError):
        TopKSAE(d_model=8, dict_size=16, k=4, b_dec=torch.zeros(7))


def test_b_dec_trainable_is_a_parameter_that_updates():
    """Pilot mode (open Q5): b_dec becomes a trained Parameter so it can
    converge toward b*. It must get a gradient and actually move -- the opposite
    of the frozen-buffer ensemble mode above."""
    bd = torch.randn(8)
    m = TopKSAE(d_model=8, dict_size=16, k=4, b_dec=bd, b_dec_trainable=True)
    assert m.b_dec.requires_grad
    assert "b_dec" in dict(m.named_parameters())    # Adam will see it
    before = m.b_dec.detach().clone()
    opt = torch.optim.Adam(m.parameters(), lr=1e-1)
    x = torch.randn(64, 8)
    x_hat, z, pre = m(x)
    loss = reduce_se((x_hat - x) ** 2, "d_model_mean")
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert m.b_dec.grad is not None
    assert not torch.equal(m.b_dec, before)         # it moved


def test_b_dec_gets_no_gradient_through_training_step():
    """End-to-end guard: after a real backward + optimizer step, the frozen
    b_dec is byte-identical. A regression to nn.Parameter would move it."""
    bd = torch.randn(8)
    m = TopKSAE(d_model=8, dict_size=16, k=4, b_dec=bd)
    before = m.b_dec.clone()
    opt = torch.optim.Adam(m.parameters(), lr=1e-1)
    x = torch.randn(32, 8)
    x_hat, z, pre = m(x)
    loss = reduce_se((x_hat - x) ** 2, "d_model_mean")
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert m.b_dec.grad is None
    torch.testing.assert_close(m.b_dec, before, atol=0, rtol=0)


def test_reduce_se_conventions_differ_by_d_model():
    se = torch.rand(4, 10)
    m = reduce_se(se, "d_model_mean")
    s = reduce_se(se, "d_model_sum")
    torch.testing.assert_close(s, m * 10)   # sum-over-d_model = d_model x mean
    with pytest.raises(ValueError):
        reduce_se(se, "nonsense")


def test_project_decoder_grad_removes_parallel_component():
    """After projection, the decoder gradient must be orthogonal to each
    (unit-norm) decoder column -- that is the whole point of the trick. If the
    projection is wrong the per-column dot product stays nonzero."""
    m = TopKSAE(d_model=8, dict_size=16, k=4)
    m.W_dec.grad = torch.randn_like(m.W_dec)
    # decoder columns are unit norm from init; the dots are nonzero pre-projection
    pre = (m.W_dec * m.W_dec.grad).sum(dim=0)
    assert pre.abs().max() > 1e-3
    m.project_decoder_grad()
    post = (m.W_dec * m.W_dec.grad).sum(dim=0)
    torch.testing.assert_close(post, torch.zeros_like(post), atol=1e-5, rtol=0)


def test_project_decoder_grad_noop_without_grad():
    m = TopKSAE(d_model=8, dict_size=16, k=4)
    assert m.W_dec.grad is None
    m.project_decoder_grad()   # must not raise before the first backward


def test_auxk_returns_none_when_nothing_dead():
    m = TopKSAE(d_model=8, dict_size=16, k=4)
    x = torch.randn(5, 8)
    pre = m.encode_pre(x)
    dead = torch.zeros(16, dtype=torch.bool)
    assert m.auxk_loss(torch.randn(5, 8), pre, dead, aux_k=4,
                       reduction="d_model_mean") is None


def test_auxk_gradient_reaches_only_dead_latents():
    """The point of AuxK: dead latents receive a gradient (so they can revive)
    while latents that are not dead are untouched by the aux term. This is the
    validator that would fail if the dead-mask were ignored -- e.g. a non-dead
    latent leaking into the top-aux_k selection."""
    torch.manual_seed(0)
    d_model, dict_size = 8, 16
    m = TopKSAE(d_model, dict_size, k=4)
    x = torch.randn(6, d_model)
    pre = m.encode_pre(x)

    dead = torch.zeros(dict_size, dtype=torch.bool)
    dead[[10, 11, 12]] = True                 # only these may be revived
    residual = torch.randn(6, d_model)        # detached target

    aux = m.auxk_loss(residual, pre, dead, aux_k=2, reduction="d_model_mean")
    assert aux is not None
    aux.backward()

    # A non-dead latent (index 0) receives no gradient from the aux term,
    # through either encoder or decoder.
    assert torch.count_nonzero(m.W_enc.grad[0]) == 0
    assert torch.count_nonzero(m.W_dec.grad[:, 0]) == 0
    # TopK is per token, so across the batch different tokens may select
    # different dead latents -- but every latent that receives decoder gradient
    # must be a dead one, and at least one dead latent must be revived.
    grad_cols = {j for j in range(dict_size)
                 if torch.count_nonzero(m.W_dec.grad[:, j]) > 0}
    assert grad_cols
    assert grad_cols <= {10, 11, 12}


def test_auxk_never_selects_more_than_dead_count():
    """aux_k is clamped to the number actually dead; asking for more than exist
    must not pull in live latents."""
    m = TopKSAE(d_model=8, dict_size=16, k=4)
    x = torch.randn(6, 8)
    pre = m.encode_pre(x)
    dead = torch.zeros(16, dtype=torch.bool)
    dead[[3, 7]] = True                        # only 2 dead
    residual = torch.randn(6, 8)
    aux = m.auxk_loss(residual, pre, dead, aux_k=512, reduction="d_model_mean")
    aux.backward()
    # exactly the 2 dead latents can carry decoder gradient; nothing else
    nonzero_cols = [j for j in range(16)
                    if torch.count_nonzero(m.W_dec.grad[:, j]) > 0]
    assert set(nonzero_cols) <= {3, 7}


def test_renorm_decoder_unit_columns():
    model = TopKSAE(d_model=16, dict_size=32, k=4)
    with torch.no_grad():
        model.W_dec.mul_(3.7)  # push norms away from 1
    model.renorm_decoder()
    norms = model.W_dec.norm(dim=0)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=0)


def test_activation_dataset_reads_fp16_as_float32(tmp_path):
    h5_path = tmp_path / "acts.h5"
    data = np.random.default_rng(0).standard_normal((10, 6)).astype(np.float16)
    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset("activations", data=data)
        hf.create_dataset("index_json", data=np.bytes_(json.dumps([])))

    ds = ActivationDataset(h5_path)
    assert len(ds) == 10
    assert ds.d_model == 6

    row = ds[3]
    assert row.dtype == torch.float32
    np.testing.assert_allclose(row.numpy(), data[3].astype(np.float32))


def test_batched_fetch_matches_per_row_fetch(tmp_path):
    """The training loop reads a whole batch in one fancy-index call. It must
    return exactly the same rows as per-sample indexing -- modulo ordering,
    which is sorted within the batch."""
    h5_path = tmp_path / "acts.h5"
    data = np.random.default_rng(1).standard_normal((50, 4)).astype(np.float16)
    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset("activations", data=data)

    ds = ActivationDataset(h5_path)
    idx = [37, 2, 19, 2 + 9]

    batched = ds[idx].numpy()
    expected = np.stack([ds[i].numpy() for i in sorted(idx)])

    assert batched.shape == (len(idx), 4)
    np.testing.assert_array_equal(batched, expected)


def test_same_seed_gives_identical_init_and_different_seed_does_not():
    """The whole experiment rests on 'vary only the seed' -- so identical
    seeds must give bit-identical initial weights, and distinct seeds must
    not."""
    def init(seed):
        set_all_seeds(seed)
        return TopKSAE(d_model=8, dict_size=16, k=3).W_enc.detach().clone()

    torch.testing.assert_close(init(3), init(3), atol=0, rtol=0)
    assert not torch.equal(init(3), init(4))


def test_jsonable_converts_numpy_attrs():
    assert _jsonable(np.int64(24)) == 24
    assert isinstance(_jsonable(np.float32(1.5)), float)
    assert _jsonable(np.array([1, 2])) == [1, 2]
    assert _jsonable(b"esm2") == "esm2"
    assert _jsonable("plain") == "plain"
    assert _jsonable(True) is True


def test_block_loader_covers_every_row_exactly_once(tmp_path):
    """Approximate shuffling is acceptable; losing or duplicating rows is not."""
    n = 1000
    _make_cache(tmp_path / "a.h5", n)
    ds = ShuffledBlockActivations(
        [tmp_path / "a.h5"], batch_size=10, seed=0,
        block_rows=64, buffer_blocks=3,
    )
    seen = [v for batch in ds for v in batch[:, 0].tolist()]
    assert len(seen) == n
    assert sorted(seen) == list(range(n))


def test_block_loader_pools_across_files(tmp_path):
    """A sharded cache needs no merge step -- blocks from every shard must land
    in the same shuffled pool."""
    _make_cache(tmp_path / "s0.h5", 300, start=0)
    _make_cache(tmp_path / "s1.h5", 300, start=1000)
    ds = ShuffledBlockActivations(
        [tmp_path / "s0.h5", tmp_path / "s1.h5"], batch_size=10, seed=0,
        block_rows=64, buffer_blocks=4,
    )
    assert ds.n_rows == 600
    seen = sorted(v for batch in ds for v in batch[:, 0].tolist())
    assert seen == sorted(list(range(300)) + list(range(1000, 1300)))


def test_block_loader_same_seed_same_order_diff_seed_differs(tmp_path):
    _make_cache(tmp_path / "a.h5", 512)

    def order(seed, epoch=0):
        ds = ShuffledBlockActivations([tmp_path / "a.h5"], batch_size=8,
                                      seed=seed, block_rows=64, buffer_blocks=2)
        ds.set_epoch(epoch)
        return [v for b in ds for v in b[:, 0].tolist()]

    assert order(0) == order(0)
    assert order(0) != order(1)
    # successive epochs must not replay the same order
    assert order(0, epoch=0) != order(0, epoch=1)


def test_block_loader_reads_are_contiguous(tmp_path):
    """The whole point is chunk-aligned reads. Assert the loader only ever
    slices contiguous ranges -- a regression to fancy-indexing would restore
    the read amplification this class exists to remove."""
    _make_cache(tmp_path / "a.h5", 512)
    ds = ShuffledBlockActivations([tmp_path / "a.h5"], batch_size=8, seed=0,
                                  block_rows=64, buffer_blocks=2)
    recorded = []
    real = h5py.Dataset.__getitem__

    def spy(self, key):
        if self.name.endswith("activations"):
            recorded.append(key)
        return real(self, key)

    h5py.Dataset.__getitem__ = spy
    try:
        list(ds)
    finally:
        h5py.Dataset.__getitem__ = real

    assert recorded, "loader made no reads"
    assert all(isinstance(k, slice) for k in recorded), (
        f"expected only contiguous slice reads, got {set(type(k) for k in recorded)}"
    )


def test_worker_init_fn_is_deterministic():
    worker_init_fn(2, base_seed=7)
    a = random.random()
    np_a = np.random.rand()

    worker_init_fn(2, base_seed=7)
    b = random.random()
    np_b = np.random.rand()

    assert a == b
    assert np_a == np_b


def test_worker_init_fn_differs_by_worker_id():
    worker_init_fn(0, base_seed=7)
    a = np.random.rand()
    worker_init_fn(1, base_seed=7)
    b = np.random.rand()
    assert a != b
