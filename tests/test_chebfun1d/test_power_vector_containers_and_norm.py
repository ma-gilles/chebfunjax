"""Focused controls for vector-power result containers and array norm.

These controls accompany the scratch source port for test_power.m passes
39–41. They exercise only contracts needed by those cases: Quasimatrix columns
may have separate interior breakpoints but share outer endpoints, and an
array-valued Chebfun infinity norm is max_x sum_j |f_j(x)|.

Provenance
----------
MATLAB sources : quasimatrix.m, @chebfun/cell2quasi.m,
                 @chebfun/horzcat.m, @chebfun/quasi2cheb.m,
                 @chebfun/norm.m
Chebfun commit: 7574c77
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.domain import Domain

EPS = float(np.finfo(np.float64).eps)


def test_quasimatrix_accepts_independent_piece_breakpoints():
    domain = Domain((-1.0, 1.0))
    left = cj.chebfun(lambda x: jnp.abs(x), domain=(-1.0, 1.0), splitting=True)
    shifted = cj.chebfun(
        lambda x: jnp.abs(x + 0.5), domain=(-1.0, 1.0), splitting=True
    )
    assert left.domain.breakpoints != shifted.domain.breakpoints
    q = Quasimatrix([left, shifted], domain)
    points = jnp.asarray((-0.75, -0.25, 0.0, 0.75))
    expected = jnp.stack((jnp.abs(points), jnp.abs(points + 0.5)), axis=-1)
    np.testing.assert_allclose(q(points), expected, rtol=0.0, atol=100 * EPS)
    np.testing.assert_allclose(q.vscale(), jnp.asarray((1.0, 1.5)), rtol=0.0,
                               atol=100 * EPS)


def test_quasimatrix_matrix_evaluation_horizontally_joins_column_blocks():
    domain = Domain((-1.0, 1.0))
    q = Quasimatrix(
        [cj.chebfun(lambda x: x, domain=(-1.0, 1.0)),
         cj.chebfun(lambda x: 2.0 * x + 1.0, domain=(-1.0, 1.0))],
        domain,
    )
    points = jnp.asarray(((-0.75, 0.25), (0.0, 0.5)))
    expected = jnp.concatenate((points, 2.0 * points + 1.0), axis=1)
    assert q(points).shape == (2, 4)
    np.testing.assert_allclose(q(points), expected, rtol=0.0, atol=100 * EPS)
    # Preserve the existing rank-one Python result convention.
    assert q(jnp.asarray((-0.75, 0.0, 0.75))).shape == (3, 2)


def test_quasimatrix_still_rejects_different_outer_endpoints():
    col = cj.chebfun(lambda x: x, domain=(-2.0, 1.0))
    with pytest.raises(ValueError, match="domain"):
        Quasimatrix([col], Domain((-1.0, 1.0)))


def test_array_chebfun_infinity_norm_is_continuous_row_one_norm():
    phase = lambda x: 3.0 * x + 0.123  # noqa: E731
    f = cj.chebfun(
        lambda x: jnp.stack((jnp.sin(phase(x)), jnp.cos(phase(x))), axis=-1),
        domain=(-1.0, 1.0),
    )
    # At x=(pi/4-.123)/3, |sin(phase)|+|cos(phase)|=sqrt(2),
    # a maximum away from the usual Chebyshev sample nodes.
    expected = np.sqrt(2.0)
    actual = float(f.norm(jnp.inf))
    assert abs(actual - expected) <= 1000 * EPS * expected


def test_array_chebfun_norm_is_not_elementwise_max():
    x = cj.chebfun(lambda t: t, domain=(-1.0, 1.0))
    f = cj.chebfun(
        lambda t: jnp.stack((t, 1.0 - t**2), axis=-1),
        domain=(-1.0, 1.0),
    )
    # max_{[-1,1]} |x| + |1-x^2| occurs at |x|=1/2 and equals 5/4.
    assert abs(float(f.norm(jnp.inf)) - 1.25) <= 1000 * EPS
    assert float(f.norm(jnp.inf)) > max(float(x.norm(jnp.inf)), 1.0)


@pytest.mark.parametrize("row", [False, True])
def test_array_norm_uses_row_sum_with_opposing_column_extrema(row):
    f = cj.chebfun(lambda x: jnp.stack((2.0 + x, 2.0 - x), axis=-1))
    if row:
        f = f.T
    # Each component peaks at3, at opposite endpoints; their row sum is4.
    assert abs(float(f.norm(jnp.inf)) - 4.0) <= 100 * EPS


@pytest.mark.parametrize("row", [False, True])
def test_complex_array_infinity_norm_sums_magnitudes_before_values(row):
    f = cj.chebfun(lambda x: jnp.stack((0*x + 3.0 + 4.0j,
                                      0*x - 1.0 + 2.0j), axis=-1))
    if row:
        f = f.T
    expected = 5.0 + np.sqrt(5.0)
    assert abs(float(f.norm(jnp.inf)) - expected) <= 1000 * EPS * expected


def test_empty_array_exponent_and_singleton_numeric_vector():
    base = cj.chebfun(lambda x: 2.0 + x)
    assert (base ** jnp.asarray([])).isempty()
    result = base ** jnp.asarray([3.0])
    points = jnp.asarray((-0.8, -0.2, 0.3, 0.9))
    np.testing.assert_allclose(result(points), (2.0 + points)**3,
                               atol=100 * EPS, rtol=100 * EPS)
    assert float(cj.chebfun().norm(jnp.inf)) == 0.0


@pytest.mark.parametrize("row", [False, True])
def test_scalar_vector_powers_follow_source_orientation(row):
    base = cj.chebfun(lambda x: x)
    if row:
        base = base.T
    # power.m columnPower zero creates a column. quasi2cheb then calls
    # horzcat because this is the first column; mixing row powers fails.
    if row:
        with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN:horzcat:transpose"):
            _ = base ** jnp.arange(4)
        return
    result = base ** jnp.arange(4)
    points = jnp.asarray((-0.7, -0.1, 0.4, 0.8))
    expected = jnp.stack([points**k for k in range(4)], axis=-1)
    assert result.n_columns == 4
    assert result.is_transposed == row
    np.testing.assert_allclose(result(points), expected.T if row else expected,
                               rtol=100 * EPS, atol=100 * EPS)


def test_matched_vector_exponent_count_must_equal_base_columns():
    base = cj.chebfun(lambda x: jnp.stack((2.0 + x, 3.0 + x), axis=-1))
    with pytest.raises(ValueError, match="dimensions must agree"):
        _ = base ** jnp.asarray((1.0, 2.0, 3.0))


def test_scalar_fractional_vector_preserves_singular_columns():
    base = cj.chebfun(lambda x: x)
    powers = jnp.asarray((0.3, 0.7))
    result = base ** powers
    assert isinstance(result, Quasimatrix)
    points = jnp.asarray((-0.8, -0.2, 0.2, 0.8))
    expected = jnp.power(points[:, None].astype(jnp.complex128), powers[None, :])
    np.testing.assert_allclose(result(points), expected, rtol=1000 * EPS,
                               atol=1000 * EPS)


@pytest.mark.parametrize("exponents,values", [
    ([[0, 1, 2, 3]], [1, 2, 4, 8]),
    ([[0], [1], [2], [3]], [1, 2, 4, 8]),
    ([[0, 1], [2, 3]], [1, 4, 2, 8]),
])
def test_numeric_exponent_arrays_use_source_column_major_order(exponents, values):
    result = cj.chebfun(2.0) ** jnp.asarray(exponents)
    points = jnp.asarray((-0.7, 0.0, 0.8))
    expected = jnp.broadcast_to(jnp.asarray(values), (3, 4))
    assert result.n_columns == 4
    np.testing.assert_allclose(result(points), expected, rtol=100 * EPS,
                               atol=100 * EPS)


@pytest.mark.parametrize("shape", [(0, 2), (2, 0)])
def test_shaped_empty_numeric_exponent_returns_empty_chebfun(shape):
    assert (cj.chebfun(2.0) ** jnp.empty(shape)).isempty()
