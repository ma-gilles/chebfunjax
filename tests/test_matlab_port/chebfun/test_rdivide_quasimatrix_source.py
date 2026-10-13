"""Native rdivide predicates 8--12 for actual Quasimatrix objects.

Provenance
----------
MATLAB source : tests/chebfun/test_rdivide.m, clauses 8--12
Chebfun commit: 7574c77
"""

import json
from pathlib import Path

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix

EPS = float(jnp.finfo(jnp.float64).eps)
X = jnp.asarray(json.loads(
    (Path(__file__).parent / 'fixtures' / 'abs_seed6178_native.json').read_text()
)['x'])
DOM = (-1.0, -0.5, 0.0, 0.5, 1.0)


def _quasi(f):
    """Construct a genuine collection of scalar Chebfun columns."""
    return Quasimatrix(f.mat2cell(), f.domain)


def _check_source_error(h, expected):
    error = h(X) - expected
    scales = _scale(h)
    error_max = float(jnp.max(jnp.abs(error)))
    scale_max = float(jnp.max(scales))
    if scale_max == 0.0:
        assert error_max == 0.0
    else:
        assert error_max < 10.0 * EPS * scale_max


def _scale(h):
    scale = h.vscale
    return scale() if callable(scale) else scale


def _assert_two_column_quasimatrix(h, *, transposed=False):
    assert isinstance(h, Quasimatrix)
    assert len(h.cols) == 2
    assert h.is_transposed is transposed


def test_source_08_scalar_over_quasimatrix():
    f = cj.chebfun(lambda x: jnp.stack((jnp.exp(x), jnp.exp(-x)), axis=-1),
                   domain=DOM)
    fq = _quasi(f)
    h = 1.0 / fq
    assert isinstance(h, cj.Chebfun)
    assert h.n_columns == 2
    expected = jnp.stack((jnp.exp(-X), jnp.exp(X)), axis=-1)
    _check_source_error(h, expected)


def test_source_09_quasimatrix_over_array_chebfun():
    fq = _quasi(cj.chebfun(
        lambda x: jnp.stack((jnp.exp(x), jnp.exp(-x)), axis=-1), domain=DOM))
    g = cj.chebfun(
        lambda x: jnp.stack((jnp.exp(-x), jnp.exp(x)), axis=-1),
        domain=(-1.0, 1.0))
    h = fq / g
    _assert_two_column_quasimatrix(h)
    expected = jnp.stack((jnp.exp(2.0 * X), jnp.exp(-2.0 * X)), axis=-1)
    _check_source_error(h, expected)


def test_source_10_chebfun_over_quasimatrix():
    f = cj.chebfun(lambda x: jnp.stack((jnp.exp(x), jnp.exp(-x)), axis=-1),
                   domain=DOM)
    gq = _quasi(cj.chebfun(
        lambda x: jnp.stack((jnp.exp(-x), jnp.exp(x)), axis=-1),
        domain=(-1.0, 1.0)))
    h = f / gq
    _assert_two_column_quasimatrix(h)
    expected = jnp.stack((jnp.exp(2.0 * X), jnp.exp(-2.0 * X)), axis=-1)
    _check_source_error(h, expected)


def test_source_11_quasimatrix_over_quasimatrix():
    fq = _quasi(cj.chebfun(
        lambda x: jnp.stack((jnp.exp(x), jnp.exp(-x)), axis=-1), domain=DOM))
    gq = _quasi(cj.chebfun(
        lambda x: jnp.stack((jnp.exp(-x), jnp.exp(x)), axis=-1),
        domain=(-1.0, 1.0)))
    h = fq / gq
    _assert_two_column_quasimatrix(h)
    expected = jnp.stack((jnp.exp(2.0 * X), jnp.exp(-2.0 * X)), axis=-1)
    _check_source_error(h, expected)


def test_source_12_transposed_quasimatrix_over_quasimatrix():
    fq = _quasi(cj.chebfun(
        lambda x: jnp.stack((jnp.exp(x), jnp.exp(-x)), axis=-1), domain=DOM))
    gq = _quasi(cj.chebfun(
        lambda x: jnp.stack((jnp.exp(-x), jnp.exp(x)), axis=-1),
        domain=(-1.0, 1.0)))
    h = fq.T / gq.T
    _assert_two_column_quasimatrix(h, transposed=True)
    expected = jnp.stack((jnp.exp(2.0 * X), jnp.exp(-2.0 * X)), axis=-1).T
    _check_source_error(h, expected)


def test_quasimatrix_division_singleton_expansion_and_source_errors():
    numerator = _quasi(cj.chebfun(
        lambda x: jnp.stack((jnp.exp(x), jnp.exp(-x)), axis=-1), domain=DOM))
    scalar = cj.chebfun(lambda x: jnp.exp(-x), domain=(-1.0, 1.0))
    expanded = numerator / scalar
    _assert_two_column_quasimatrix(expanded)
    _check_source_error(
        expanded, jnp.stack((jnp.exp(2.0 * X), jnp.ones_like(X)), axis=-1))

    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:rdivide:quasi'):
        _ = numerator / _quasi(cj.chebfun(
            lambda x: jnp.stack((jnp.exp(x), jnp.exp(-x), jnp.ones_like(x)), axis=-1),
            domain=DOM))

    zero = _quasi(cj.chebfun(lambda x: jnp.zeros_like(x), domain=DOM))
    with pytest.raises(
        ValueError,
        match='CHEBFUN:CHEBFUN:rdivide:columnRdivide:divisionByZeroChebfun',
    ):
        _ = numerator / zero

    # Singleton expansion assigns left to right. The first denominator has
    # an incompatible domain, while the second is identically zero; the
    # source therefore reports the first column's domain error.
    numerator_singleton = _quasi(cj.chebfun(lambda x: jnp.exp(x), domain=DOM))
    disjoint_denominators = _quasi(cj.chebfun(
        lambda x: jnp.stack((jnp.exp(x), jnp.zeros_like(x)), axis=-1),
        domain=(2.0, 2.5, 3.0)))
    with pytest.raises(
        ValueError,
        match='CHEBFUN:CHEBFUN:rdivide:columnRdivide:domain',
    ):
        _ = numerator_singleton / disjoint_denominators


def test_numeric_zero_shortcut_and_transposed_orientation():
    q = _quasi(cj.chebfun(
        lambda x: jnp.stack((jnp.exp(x), jnp.exp(-x)), axis=-1), domain=DOM))

    zero_scalar_result = 0.0 / q
    _assert_two_column_quasimatrix(zero_scalar_result)
    _check_source_error(zero_scalar_result, jnp.zeros((X.size, 2), dtype=jnp.float64))

    zero_result = jnp.zeros((2, 1), dtype=jnp.float64) / q
    _assert_two_column_quasimatrix(zero_result)
    _check_source_error(zero_result, jnp.zeros((X.size, 2), dtype=jnp.float64))

    transposed = q.T
    constant_result = 1.0 / transposed
    assert isinstance(constant_result, cj.Chebfun)
    assert constant_result.n_columns == 2
    assert constant_result.is_transposed
    values = constant_result(X)
    expected = jnp.stack((jnp.exp(-X), jnp.exp(X)), axis=0)
    assert values.shape == expected.shape
    assert float(jnp.max(jnp.abs(values - expected))) < (
        10.0 * EPS * _scale(constant_result))


def test_one_column_quasimatrix_numeric_vector_expansion():
    q = _quasi(cj.chebfun(lambda x: jnp.exp(x), domain=DOM))
    divided_by_vector = q / jnp.asarray([1.0, 2.0, 3.0])
    assert isinstance(divided_by_vector, cj.Chebfun)
    assert divided_by_vector.n_columns == 3
    _check_source_error(divided_by_vector, jnp.stack(
        (jnp.exp(X), jnp.exp(X) / 2.0, jnp.exp(X) / 3.0), axis=-1))

    vector_over_q = jnp.asarray([1.0, 2.0, 3.0]) / q
    assert isinstance(vector_over_q, cj.Chebfun)
    assert vector_over_q.n_columns == 3
    _check_source_error(vector_over_q, jnp.stack(
        (jnp.exp(-X), 2.0 * jnp.exp(-X), 3.0 * jnp.exp(-X)), axis=-1))

    row_q = q.T
    row_divided_by_vector = row_q / jnp.asarray([1.0, 2.0, 3.0])
    assert isinstance(row_divided_by_vector, cj.Chebfun)
    assert row_divided_by_vector.is_transposed
    assert row_divided_by_vector.n_columns == 3
    expected_row = jnp.stack(
        (jnp.exp(X), jnp.exp(X) / 2.0, jnp.exp(X) / 3.0), axis=0)
    assert row_divided_by_vector(X).shape == expected_row.shape
    assert float(jnp.max(jnp.abs(row_divided_by_vector(X) - expected_row))) < (
        10.0 * EPS * _scale(row_divided_by_vector))

    vector_over_row_q = jnp.asarray([1.0, 2.0, 3.0]) / row_q
    assert isinstance(vector_over_row_q, cj.Chebfun)
    assert vector_over_row_q.is_transposed
    assert vector_over_row_q.n_columns == 3
    expected_vector_row = jnp.stack(
        (jnp.exp(-X), 2.0 * jnp.exp(-X), 3.0 * jnp.exp(-X)), axis=0)
    assert vector_over_row_q(X).shape == expected_vector_row.shape
    assert float(jnp.max(jnp.abs(vector_over_row_q(X) - expected_vector_row))) < (
        10.0 * EPS * _scale(vector_over_row_q))
