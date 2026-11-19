"""
Tests for seed_axis.py. Explicitly verifies the assertion fails against a
deliberately permuted file, not just that it passes on the happy path --
the whole point is catching the lexicographic-vs-numeric ordering mismatch
that already produced one bug this session.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.seed_axis import align_to_reference, assert_seed_axes_match


def _glob_order(n: int) -> list[str]:
    """Reproduces glob("seed_*")'s lexicographic sort for seed_0..seed_{n-1}."""
    return sorted((f"seed_{i}" for i in range(n)))


def test_matching_order_passes():
    names = _glob_order(20)
    assert_seed_axes_match(names, list(names))    # no raise


def test_lexicographic_vs_numeric_mismatch_is_caught():
    """THE case that actually bit this session: lexicographic glob order
    (seed_0, seed_1, seed_10, ...) vs numeric order (seed_0, seed_1, seed_2,
    ...) differ starting at the 3rd element."""
    lexicographic = _glob_order(20)
    numeric = [f"seed_{i}" for i in range(20)]
    assert lexicographic != numeric              # sanity: they really do differ
    with pytest.raises(ValueError, match="seed axis mismatch"):
        assert_seed_axes_match(lexicographic, numeric)


def test_a_deliberately_permuted_file_is_caught():
    """Simulates exactly the risk this module exists for: two files that
    SHOULD describe the same 20 seeds, but one has them shuffled."""
    names_a = _glob_order(20)
    rng = np.random.default_rng(0)
    names_b = list(names_a)
    rng.shuffle(names_b)
    assert names_a != names_b
    with pytest.raises(ValueError):
        assert_seed_axes_match(names_a, names_b)


def test_align_to_reference_actually_reorders_data_correctly():
    """Not just a label check -- verify the DATA moves to the right
    positions, using values that make a silent mix-up obvious."""
    names = ["seed_2", "seed_0", "seed_1"]        # some arbitrary order
    array = np.array([200, 0, 100])                # array[i] "belongs to" names[i]
    reference = ["seed_0", "seed_1", "seed_2"]
    aligned = align_to_reference(names, array, reference)
    assert aligned.tolist() == [0, 100, 200]        # now in reference order


def test_align_to_reference_rejects_a_missing_seed():
    names = ["seed_0", "seed_1"]
    array = np.array([0, 1])
    reference = ["seed_0", "seed_1", "seed_2"]      # seed_2 not in source
    with pytest.raises(ValueError, match="seed sets differ"):
        align_to_reference(names, array, reference)


def test_align_to_reference_does_not_silently_drop_an_extra_seed():
    names = ["seed_0", "seed_1", "seed_2"]
    array = np.array([0, 1, 2])
    reference = ["seed_0", "seed_1"]                # source has an extra seed
    with pytest.raises(ValueError, match="seed sets differ"):
        align_to_reference(names, array, reference)
