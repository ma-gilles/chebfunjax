"""Focused public Chebtech cumsum dimension and JAX contracts.

Provenance
----------
MATLAB source : @chebtech/cumsum.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import (
    Chebtech1,
    Chebtech2,
    _cumsum_coeffs,
    _cumsum_coeffs_by_dim,
)


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_public_cumsum_dim2_is_column_prefix_sum(Tech):
    coeffs = jnp.asarray([[1.0, 2.0, -1.0], [0.25, -0.5, 1.5]])
    f = Tech(coeffs=coeffs, ishappy=True)
    got = f.cumsum(dim=2)
    expected = jnp.asarray([[1.0, 3.0, 2.0], [0.25, -0.25, 1.25]])
    np.testing.assert_array_equal(np.asarray(got.coeffs), np.asarray(expected))
    assert got.ishappy is f.ishappy

    # The columns represent 1+.25x, 2-.5x, -1+1.5x; the source prefix
    # result is 1+.25x, 3-.25x, 2+1.25x.
    points = jnp.asarray([-0.5, 0.25])
    analytic = jnp.stack(
        [1.0 + 0.25 * points, 3.0 - 0.25 * points,
         2.0 + 1.25 * points],
        axis=1,
    )
    np.testing.assert_allclose(
        np.asarray(got(points)), np.asarray(analytic), rtol=2e-15, atol=2e-15
    )


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_public_cumsum_routes_every_nonone_dim_to_columns(Tech):
    coeffs = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    f = Tech(coeffs=coeffs, ishappy=True)
    actual = f.cumsum(dim=0)
    expected = jnp.asarray([[1.0, 3.0], [3.0, 7.0]])
    np.testing.assert_array_equal(np.asarray(actual.coeffs), np.asarray(expected))


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_public_cumsum_scalar_dim2_is_identity(Tech):
    f = Tech.from_coeffs(jnp.asarray([1.0, 0.5, -0.25]))
    assert f.cumsum(dim=2) is f


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_public_cumsum_empty_sentinel_is_identity(Tech):
    f = Tech.empty()
    assert f.cumsum() is f
    assert f.cumsum(dim=2) is f


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_public_cumsum_object_method_keeps_traced_recurrence(Tech):
    coeffs = jnp.asarray([1.0, 0.5, -0.25])
    f = Tech.from_coeffs(coeffs)
    compiled = jax.jit(lambda tech: tech.cumsum())(f)
    # Integrating 1+.5*T1-.25*T2 from -1 gives this independent
    # Chebyshev expansion; avoid using the production helper as its oracle.
    expected = np.asarray([23.0 / 24.0, 9.0 / 8.0, 1.0 / 8.0, -1.0 / 24.0])
    np.testing.assert_allclose(
        np.asarray(compiled.coeffs), expected,
        rtol=2e-15, atol=2e-15,
    )


@pytest.mark.parametrize("dim", [1, 2])
def test_cumsum_coefficient_adapter_jit(dim):
    coeffs = jnp.asarray([[1.0, 2.0], [0.5, -0.25], [0.0, 1.0]])
    if dim == 1:
        expected = np.asarray(
            [[7.0 / 8.0, 83.0 / 48.0], [1.0, 1.5],
             [1.0 / 8.0, -1.0 / 16.0], [0.0, 1.0 / 6.0]]
        )
    else:
        expected = np.asarray([[1.0, 3.0], [0.5, 0.25], [0.0, 1.0]])
    compiled = jax.jit(lambda c: _cumsum_coeffs_by_dim(c, dim=dim))(coeffs)
    np.testing.assert_allclose(
        np.asarray(compiled), expected, rtol=2e-15, atol=2e-15
    )


def test_cumsum_coefficient_recurrence_has_finite_exact_linear_gradient():
    coeffs = jnp.asarray([0.75, -0.5, 0.25])
    weights = jnp.asarray([0.5, -1.0, 2.0, 0.25])
    gradient = jax.grad(
        lambda c: jnp.sum(weights * _cumsum_coeffs(c))
    )(coeffs)
    jacobian = np.asarray(
        [[1.0, -0.25, -1.0 / 3.0],
         [1.0, 0.0, -0.5],
         [0.0, 0.25, 0.0],
         [0.0, 0.0, 1.0 / 6.0]]
    )
    expected = jacobian.T @ np.asarray(weights)
    np.testing.assert_allclose(np.asarray(gradient), expected, rtol=0.0, atol=5e-16)
