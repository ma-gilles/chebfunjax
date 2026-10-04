"""Source-bound fixtures for the scratch JAX Chebtech alias candidates.

MATLAB provenance: Chebfun commit 7574c77680d7e82b79626300bf255498271a72df,
tests/chebtech{1,2}/test_alias.m and tests/chebtech/test_alias.m. Literal goldens and a separate host-list reference qualify source folds.
"""
from functools import partial

import jax
import jax.numpy as jnp

from chebfunjax.tech.chebtech import _alias_chebtech1, _alias_chebtech2


def _matlab_alias_reference(coeffs, m, kind):
    """Small independent scalar-list transcription of MATLAB's 1-based loops."""
    rows = [[complex(v) if isinstance(v, complex) else v for v in row]
            for row in (coeffs if coeffs and isinstance(coeffs[0], (list, tuple))
                        else [[v] for v in coeffs])]
    was_vector = not (coeffs and isinstance(coeffs[0], (list, tuple)))
    n = len(rows)
    ncols = len(rows[0]) if n else (len(coeffs[0]) if coeffs else 1)
    if m > n:
        rows.extend([[0] * ncols for _ in range(m - n)])
        result = rows
    elif m == 0:
        result = []
    elif m == 1:
        result = [[sum(((-1) ** q) * rows[2 * q][col]
                       for q in range((n + 1) // 2))
                    for col in range(ncols)]]
    else:
        result = [row[:] for row in rows]
        for j in range(m + 1, n + 1):  # MATLAB source's inclusive 1-based loop
            if kind == 2:
                k = abs(((j + m - 3) % (2 * m - 2)) - m + 2) + 1
                sign = 1
            else:
                k = abs(((j + m - 2) % (2 * m)) - m + 1) + 1
                p = (j - 1 + m) // (2 * m)
                sign = (-1) ** p
            for col in range(ncols):
                result[k - 1][col] += sign * result[j - 1][col]
        result = result[:m]
    if was_vector:
        return [row[0] for row in result]
    return result


def _assert_allclose(actual, expected):
    expected = jnp.asarray(expected, dtype=jnp.asarray(actual).dtype)
    assert jnp.max(jnp.abs(jnp.asarray(actual) - expected), initial=0.0) < 1e-13


def test_matlab_source_exact_real_goldens():
    c = jnp.arange(10.0, 0.0, -1.0)
    _assert_allclose(_alias_chebtech2(c, 9), [10, 9, 8, 7, 6, 5, 4, 4, 2])
    _assert_allclose(_alias_chebtech2(c, 3), [18, 25, 12])
    _assert_allclose(_alias_chebtech1(c, 9), jnp.arange(10.0, 1.0, -1.0))
    _assert_allclose(_alias_chebtech1(c, 3), [6, 1, 0])


def test_independent_complex_array_and_low_fold_fixtures():
    coeffs = [
        [1 + 2j, -3j],
        [-2 + 4j, 5 + 1j],
        [3 - 1j, -2 - 2j],
        [4j, 7 - 3j],
        [-1 - 5j, 2j],
        [6 + 2j, -4 + 1j],
        [3j, 8 + 2j],
        [-7 + 1j, -1 - 3j],
        [2 - 6j, 5j],
    ]
    for kind, alias in ((1, _alias_chebtech1), (2, _alias_chebtech2)):
        for m in (1, 2, 4, 7, 9, 12):
            expected = _matlab_alias_reference(coeffs, m, kind)
            _assert_allclose(alias(jnp.asarray(coeffs), m), expected)


def test_empty_inputs_padding_and_column_shape():
    empty_vec = jnp.asarray([], dtype=jnp.complex128)
    empty_matrix = jnp.empty((0, 3), dtype=jnp.float64)
    for alias in (_alias_chebtech1, _alias_chebtech2):
        assert alias(empty_vec, 0).shape == (0,)
        assert alias(empty_vec, 4).shape == (4,)
        assert jnp.all(alias(empty_vec, 4) == 0)
        assert alias(empty_matrix, 3).shape == (3, 3)
        assert jnp.all(alias(empty_matrix, 3) == 0)
        column = jnp.arange(6.0)[:, None]
        assert alias(column, 8).shape == (8, 1)
        _assert_allclose(alias(column, 8)[:6], column)


def test_m_one_source_alternating_even_modes():
    c = jnp.asarray([1 + 2j, 91, 3 - 1j, 92, -4j, 93, 5 + 3j])
    expected = (1 + 2j) - (3 - 1j) + (-4j) - (5 + 3j)
    for alias in (_alias_chebtech1, _alias_chebtech2):
        _assert_allclose(alias(c, 1), [expected])


def test_jit_with_static_alias_length():
    c = jnp.arange(16.0).reshape(8, 2) + 1j * jnp.arange(16.0, 32.0).reshape(8, 2)
    for alias in (_alias_chebtech1, _alias_chebtech2):
        compiled = partial(jax.jit, static_argnames=("m",))(alias)
        for m in (1, 3, 7, 11):
            _assert_allclose(compiled(c, m), alias(c, m))
