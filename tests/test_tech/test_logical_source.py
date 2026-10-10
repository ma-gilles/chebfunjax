"""Source controls for @chebtech/logical.m, Chebfun commit 7574c77."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_logical_root_free_columns_and_metadata(Tech):
    # Zero, positive root-free, and imaginary root-free columns.
    coefficients = jnp.array([[0, 2, 2j], [0, 0.5, 0.5j]])
    f = Tech(coeffs=coefficients, ishappy=False)
    g = f.logical()
    assert isinstance(g, Tech)
    assert g.ishappy is False
    assert g.coeffs.dtype == jnp.bool_
    np.testing.assert_array_equal(g.coeffs, [[False, True, True]])
    np.testing.assert_array_equal(f.coeffs, coefficients)


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_logical_uses_values_and_ignores_nan(Tech):
    # A coefficient reduction incorrectly returns true for [1, NaN].
    # Both transformed values are NaN, which native ANY ignores.
    g = Tech.from_coeffs(jnp.array([1., jnp.nan])).logical()
    np.testing.assert_array_equal(g.coeffs, [False])
    row = Tech.from_coeffs(jnp.array([[0., jnp.nan, jnp.inf, -2.]]))
    np.testing.assert_array_equal(row.logical().coeffs, [[False, False, True, True]])


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_logical_empty_shapes(Tech):
    for empty in (Tech.empty(), Tech.from_coeffs(jnp.array([]))):
        g = empty.logical()
        assert g.isempty()
        assert g.coeffs.shape == (0,)
    f = Tech.from_coeffs(jnp.empty((0, 3)))
    np.testing.assert_array_equal(f.logical().coeffs, [[False, False, False]])
    f = Tech.from_coeffs(jnp.empty((0, 0)))
    assert f.logical().coeffs.shape == (1, 0)


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_logical_jit_root_free_scalar(Tech):
    convert = jax.jit(lambda c: Tech.from_coeffs(c).logical().coeffs)
    np.testing.assert_array_equal(convert(jnp.array([2., 0.5])), [True])
    np.testing.assert_array_equal(convert(jnp.array([0., 0.])), [False])
