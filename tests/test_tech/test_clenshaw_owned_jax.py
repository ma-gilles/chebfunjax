"""Independent polynomial/AD and shared eager/traced evaluation controls."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech2, _clenshaw


@pytest.mark.parametrize("n", [5, 6])
def test_joint_coefficient_and_query_jvp_vjp(n):
    # Derivatives come from independent Chebyshev basis and chebder oracles.
    c = np.arange(1., n * 3 + 1).reshape(n, 3) / (n * 3)
    dc = np.cos(np.arange(n * 3)).reshape(n, 3)
    x = np.asarray([-.8, -.2, .3, .75])
    dx = np.asarray([.2, -.3, .4, -.1])
    weight = np.arange(1., 13.).reshape(4, 3) / 12
    basis = np.polynomial.chebyshev.chebvander(x, n - 1)
    derivative = np.polynomial.chebyshev.chebval(
        x, np.polynomial.chebyshev.chebder(c, axis=0)).T
    expected = basis @ dc + dx[:, None] * derivative
    args = tuple(map(jnp.asarray, (c, x)))
    actual, tangent = jax.jit(lambda a, b, da, db: jax.jvp(
        _clenshaw, (a, b), (da, db)))(*args, jnp.asarray(dc), jnp.asarray(dx))
    np.testing.assert_allclose(actual, basis @ c, rtol=0, atol=100*np.finfo(float).eps)
    np.testing.assert_allclose(tangent, expected, rtol=0, atol=100*np.finfo(float).eps)
    _, pullback = jax.vjp(_clenshaw, *args)
    gc, gx = pullback(jnp.asarray(weight))
    np.testing.assert_allclose(gc, basis.T @ weight, rtol=0, atol=100*np.finfo(float).eps)
    np.testing.assert_allclose(gx, np.sum(weight * derivative, axis=1),
                               rtol=0, atol=100*np.finfo(float).eps)


@pytest.mark.parametrize("columns", [False, True])
def test_eager_jit_vmap_same_bits(columns):
    c = jnp.arange(1., 31.).reshape(10, 3) if columns else jnp.arange(1., 11.)
    tech = Chebtech2.from_coeffs(c)
    x = jnp.asarray([-.8, -.1, .4, 1.])
    eager = np.asarray(tech(x))
    np.testing.assert_array_equal(eager.view(np.uint64), np.asarray(jax.jit(lambda z: tech(z))(x)).view(np.uint64))
    np.testing.assert_array_equal(eager.view(np.uint64), np.asarray(jax.vmap(tech)(x)).view(np.uint64))


@pytest.mark.parametrize("n", [0, 1])
@pytest.mark.parametrize("columns", [False, True])
def test_empty_constant_shapes(n, columns):
    c = jnp.ones((n, 3) if columns else (n,))
    x = jnp.ones((2, 4))
    expected = np.full((2, 4, 3) if columns else (2, 4), float(n))
    np.testing.assert_array_equal(_clenshaw(c, x), expected)
    np.testing.assert_array_equal(jax.jit(_clenshaw)(c, x), expected)
