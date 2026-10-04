"""Column-wise tech simplification from MATLAB Chebfun source.

Provenance
----------
MATLAB source : @chebtech/simplify.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.chebtech import (
    Chebtech1,
    Chebtech2,
    _chop_columns,
    _round_half_away,
)


def _coefficients() -> jnp.ndarray:
    indices = jnp.arange(48, dtype=jnp.float64)
    first = 0.7**indices
    second = 1e-7 * 0.83**indices
    return jnp.stack((first, second), axis=1)


def test_column_tolerances_match_separate_simplifies_and_max_cutoff():
    coeffs = _coefficients()
    tolerances = (1e-12, 1e-4)
    combined = Chebtech2(coeffs=coeffs).simplify(jnp.asarray(tolerances))
    columns = [
        Chebtech2(coeffs=coeffs[:, j]).simplify(tolerances[j])
        for j in range(coeffs.shape[1])
    ]

    expected_cutoff = max(column.n for column in columns)
    assert combined.n == expected_cutoff
    np.testing.assert_allclose(
        np.asarray(combined.coeffs), np.asarray(coeffs[:expected_cutoff]),
        rtol=0.0, atol=0.0,
    )


def test_looser_small_column_tolerance_reduces_shared_cutoff():
    indices = jnp.arange(48, dtype=jnp.float64)
    large_low_degree = jnp.where(indices < 5, 1.0 / (indices + 1.0), 0.0)
    small_slow_tail = 1e-7 * 0.5**indices
    coeffs = jnp.stack((large_low_degree, small_slow_tail), axis=1)

    unequal_cutoff = _chop_columns(coeffs, jnp.asarray([1e-14, 1e-2]))
    tight_cutoff = _chop_columns(coeffs, jnp.asarray([1e-14, 1e-14]))

    assert unequal_cutoff < tight_cutoff


def test_wrong_tolerance_vector_length_uses_maximum_for_every_column():
    coeffs = _coefficients()
    mismatch = _chop_columns(coeffs, jnp.asarray([1e-8, 1e-3, 1e-5]))
    repeated_max = _chop_columns(coeffs, jnp.asarray([1e-3, 1e-3]))
    assert mismatch == repeated_max
    column_vector = _chop_columns(coeffs, jnp.asarray([[1e-8], [1e-3]]))
    assert column_vector == repeated_max

    one_column = coeffs[:, 0]
    mismatch_one = _chop_columns(one_column, jnp.asarray([1e-8, 1e-3]))
    max_one = _chop_columns(one_column, 1e-3)
    assert mismatch_one == max_one


def test_empty_and_unhappy_tech_simplify_are_noops():
    empty = Chebtech2.empty()
    assert empty.simplify(jnp.asarray([1e-5, 1e-3])) is empty
    assert empty.isempty()
    empty1 = Chebtech1.empty()
    assert empty1.simplify(jnp.asarray([1e-5, 1e-3])) is empty1
    assert empty1.isempty()

    zero_coeffs = Chebtech2(coeffs=jnp.asarray([], dtype=jnp.float64))
    assert zero_coeffs.simplify(1e-5) is zero_coeffs
    zero_coeffs1 = Chebtech1(coeffs=jnp.asarray([], dtype=jnp.float64))
    assert zero_coeffs1.simplify(1e-5) is zero_coeffs1

    unhappy = Chebtech2(coeffs=_coefficients(), ishappy=False)
    assert unhappy.simplify(jnp.asarray([1e-5, 1e-3])) is unhappy


def test_chebtech1_uses_source_roundtrip_and_tolerance_vectors():
    coeffs = _coefficients()
    tolerances = jnp.asarray([1e-12, 1e-4])
    combined = Chebtech1(coeffs=coeffs).simplify(tolerances)
    separate = [
        Chebtech1(coeffs=coeffs[:, j]).simplify(tolerances[j])
        for j in range(coeffs.shape[1])
    ]
    expected_cutoff = max(column.n for column in separate)
    assert combined.n == expected_cutoff
    np.testing.assert_allclose(
        np.asarray(combined.coeffs), np.asarray(coeffs[:expected_cutoff]),
        rtol=0.0, atol=0.0,
    )


def test_prolongation_length_uses_matlab_half_away_rounding():
    assert _round_half_away(12.5) == 13
