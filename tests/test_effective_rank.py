"""Tests for src/analysis/effective_rank.py -- M2, amendment AB15.

Expectations are WRITTEN DOWN, and the key ones are CLOSED FORMS on matrices
whose spectra are known by hand (gotcha 32b): an orthonormal set has flat
singular values so every spectral statistic equals m, and a rank-1 set has one
non-zero singular value so every one of them equals 1. Those two pin the
arithmetic from both ends.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.effective_rank import (bin_by_rf, null_curve,
                                         spectrum_statistics)


def test_orthonormal_columns_give_EVERY_statistic_equal_to_m():
    """Flat spectrum: ER = m under both normalisations, stable rank = m,
    participation ratio = m. Closed form, no tolerance argument needed."""
    d, m = 32, 8
    q = np.linalg.qr(np.random.default_rng(0).normal(size=(d, m)))[0]
    st = spectrum_statistics(q, d)
    assert st["ER_l1"] == pytest.approx(m, rel=1e-9)
    assert st["ER_energy"] == pytest.approx(m, rel=1e-9)
    assert st["stable_rank"] == pytest.approx(m, rel=1e-9)
    assert st["participation_ratio"] == pytest.approx(m, rel=1e-9)
    assert st["ER_l1_over_d"] == pytest.approx(m / d)
    assert st["ER_l1_over_min_d_m"] == pytest.approx(1.0)


def test_rank_one_columns_give_EVERY_statistic_equal_to_one():
    """The other end. All columns parallel -> one non-zero singular value."""
    d, m = 16, 6
    v = np.random.default_rng(1).normal(size=(d, 1))
    v /= np.linalg.norm(v)
    W = np.repeat(v, m, axis=1)
    st = spectrum_statistics(W, d)
    assert st["ER_l1"] == pytest.approx(1.0, rel=1e-8)
    assert st["ER_energy"] == pytest.approx(1.0, rel=1e-8)
    assert st["stable_rank"] == pytest.approx(1.0, rel=1e-8)
    assert st["participation_ratio"] == pytest.approx(1.0, rel=1e-8)
    assert st["n_nonzero_sv"] == 1


def test_the_two_normalisations_DISAGREE_on_a_non_flat_spectrum():
    """If they agreed everywhere, computing both would be pointless and a
    module that computed one twice would pass. This fixture discriminates."""
    d, m = 20, 10
    q = np.linalg.qr(np.random.default_rng(2).normal(size=(d, m)))[0]
    W = q * np.array([10.0, 5, 1, 1, 1, 1, 1, 1, 1, 0.5])   # skewed spectrum
    st = spectrum_statistics(W, d)
    assert st["ER_l1"] != pytest.approx(st["ER_energy"], rel=1e-3)
    # energy weighting concentrates on the large values, so it is the SMALLER
    assert st["ER_energy"] < st["ER_l1"]


def test_the_cap_is_real_ER_never_exceeds_min_d_m():
    """The inequality the whole quintile-versus-decile argument rests on."""
    rng = np.random.default_rng(3)
    for d, m in ((10, 40), (40, 10), (25, 25)):
        W = rng.normal(size=(d, m))
        st = spectrum_statistics(W, d)
        assert st["ER_l1"] <= min(d, m) + 1e-9
        assert st["ER_energy"] <= min(d, m) + 1e-9
        assert st["ER_l1_over_min_d_m"] <= 1.0 + 1e-9


def test_bins_are_equal_sized_ascending_and_DETERMINISTIC_under_ties():
    """r_f takes 40 values over 10,240 features, so ties dominate. An unstable
    sort would make the partition irreproducible run to run."""
    rng = np.random.default_rng(4)
    rf = rng.integers(0, 40, size=1000) / 39.0      # heavily tied, as the real one is
    bins = bin_by_rf(rf, 5)
    assert [len(b) for b in bins] == [200] * 5
    assert sorted(np.concatenate(bins).tolist()) == list(range(1000))
    means = [rf[b].mean() for b in bins]
    assert means == sorted(means)                    # ascending in r_f
    assert all(np.array_equal(a, b)
               for a, b in zip(bins, bin_by_rf(rf, 5)))   # deterministic


def test_TIES_are_broken_by_ASCENDING_INDEX_not_arbitrarily():
    """What `kind="stable"` actually buys, and the reason comparing the
    function to itself was not enough: ANY deterministic sort passes that,
    including an unstable one. The guarantee that matters is reproducibility
    ACROSS implementations, and it is visible only in the tie order.

    The fixture must be one an unstable sort actually REORDERS. An already
    sorted block layout is not: quicksort leaves it alone and the mutation
    survives, which is what a first version of this test did. So the values are
    SHUFFLED and tied, exactly the shape the real `r_f` has -- 10,240 features
    over 40 distinct values.
    """
    rng = np.random.default_rng(11)
    rf = rng.integers(0, 40, size=1000).astype(float) / 39.0

    # The fixture is shown to DISCRIMINATE before it is trusted: an unstable
    # sort must give a different permutation on this input, or the assertion
    # below would hold for both and test nothing.
    assert not np.array_equal(np.argsort(rf, kind="stable"),
                              np.argsort(rf, kind="quicksort"))

    bins = bin_by_rf(rf, 5)
    # Within each bin, every group of tied r_f must be in ASCENDING index
    # order -- the property stability guarantees and an unstable sort does not.
    for b in bins:
        for v in np.unique(rf[b]):
            tied = b[rf[b] == v]
            assert tied.tolist() == sorted(tied.tolist())
    # And the whole partition is the stable one, elementwise.
    order = np.argsort(rf, kind="stable")
    assert np.array_equal(np.concatenate(bins),
                          np.concatenate([np.sort(x)
                                          for x in np.array_split(order, 5)]))


def test_the_null_curve_brackets_a_RANDOM_selection_and_not_a_structured_one():
    """The null exists to say what ER/d does when the selection carries no
    information. A structured selection must fall OUTSIDE it, or the null
    cannot discriminate -- which is the whole reason AB15 requires it."""
    rng = np.random.default_rng(5)
    d, n_feat, m = 24, 200, 40
    # 40 features living in a 3-dimensional subspace; the rest generic.
    W = rng.normal(size=(d, n_feat))
    W[:, :m] = rng.normal(size=(d, 3)) @ rng.normal(size=(3, m))
    nulls = null_curve(W, m, n_draws=15, rng=rng, d_model=d)
    structured = spectrum_statistics(W[:, :m], d)["ER_l1_over_d"]
    assert structured < nulls["ER_l1_over_d"]["min"]
    assert nulls["ER_l1_over_d"]["n_draws"] == 15
    assert nulls["ER_l1_over_d"]["m"] == m


def test_EV_SVD_is_a_cumulative_energy_fraction_in_the_unit_interval():
    rng = np.random.default_rng(6)
    st = spectrum_statistics(rng.normal(size=(30, 50)), 30)
    assert 0.0 < st["EV_SVD_r"] <= 1.0
    assert st["EV_SVD_r_at"] == 15          # 0.5 * min(30, 50)


def test_orthonormal_EV_SVD_is_exactly_the_fraction_of_components_kept():
    """Flat spectrum -> cumulative energy is exactly r/m. Closed form.

    NOTE this case CANNOT detect cumulating from the wrong end -- with equal
    singular values both ends give r/m. The next test is the one that does,
    and it exists because a mutation survived this one.
    """
    d, m = 40, 20
    q = np.linalg.qr(np.random.default_rng(7).normal(size=(d, m)))[0]
    st = spectrum_statistics(q, d)
    assert st["EV_SVD_r"] == pytest.approx(st["EV_SVD_r_at"] / m, rel=1e-9)


def test_EV_SVD_cumulates_from_the_LARGEST_singular_values():
    """A skewed spectrum, where the top half and the bottom half of the energy
    are nothing like each other. Closed form: with singular values
    (4, 2, 1, 1) the energies are (16, 4, 1, 1) summing to 22, so the top
    r = 2 hold 20/22 and the bottom two hold 2/22."""
    d, m = 8, 4
    q = np.linalg.qr(np.random.default_rng(8).normal(size=(d, m)))[0]
    W = q * np.array([4.0, 2.0, 1.0, 1.0])
    st = spectrum_statistics(W, d)
    assert st["EV_SVD_r_at"] == 2
    assert st["EV_SVD_r"] == pytest.approx(20.0 / 22.0, rel=1e-9)
    assert st["EV_SVD_r"] > 0.5      # top-heavy; the wrong end gives 2/22


_ARGS = ["--rf-counts", "r.h5", "--ckpts", "c", "--seeds", "0", "--n-compare",
         "39", "--out", "o.json"]


@pytest.mark.parametrize("missing", ["--endpoint-low", "--endpoint-high"])
def test_endpoint_cuts_have_no_default(missing):
    """CT8 item 3: the N = 19 defaults were applied to N = 39 counts (entry
    271); the cut must now be stated."""
    from src.analysis.effective_rank import build_parser
    given = {"--endpoint-low": "2", "--endpoint-high": "37"}
    given.pop(missing)
    argv = _ARGS + [x for kv in given.items() for x in kv]
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


def test_endpoint_cuts_parse_when_stated():
    from src.analysis.effective_rank import build_parser
    a = build_parser().parse_args(_ARGS + ["--endpoint-low", "2",
                                           "--endpoint-high", "37"])
    assert (a.endpoint_low, a.endpoint_high) == (2, 37)
