# uses-numpy: host reference assertions and captured MATLAB fixture inspection.
"""Additional batching/AD scope; original failed exact66 control unchanged."""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils import _gradual as gradual


@pytest.mark.parametrize("disabled", [True, False])
@pytest.mark.parametrize("mode", ["shared_denominator", "shared_numerator", "both", "nested"])
def test_mapped_finite_exact_quotients(disabled, mode):
    a = jnp.asarray([[0.75, 1.25, 2.5], [1.5, 2.5, 5.0]])
    b = jnp.asarray([[1.75, 1.75, 1.75], [3.5, 3.5, 3.5]])
    f = gradual.gradual_positive_divide
    if mode == "shared_denominator":
        x, y = a[0], b[0, 0]
        action = jax.vmap(f, in_axes=(0, None))
    elif mode == "shared_numerator":
        x, y = a[0, 1], b[0]
        action = jax.vmap(f, in_axes=(None, 0))
    elif mode == "both":
        x, y = a, b
        action = jax.vmap(f)
    else:
        x, y = a, b
        action = jax.vmap(jax.vmap(f))
    xx, yy = np.broadcast_arrays(np.asarray(x), np.asarray(y))
    expected = np.array(
        [
            float(Fraction.from_float(float(u)) / Fraction.from_float(float(v)))
            for u, v in zip(xx.flat, yy.flat)
        ]
    ).reshape(xx.shape)
    with jax.disable_jit(disabled):
        actual = jax.jit(action)(x, y)
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("disabled", [True, False])
def test_mapped_forward_reverse_derivatives(disabled):
    f = jax.vmap(gradual.gradual_positive_divide, in_axes=(0, None))
    x = jnp.asarray([0.75, 1.25, 2.5])
    y = jnp.asarray(1.75)
    with jax.disable_jit(disabled):
        _, tangent = jax.jit(
            lambda a, b: jax.jvp(f, (a, b), (jnp.ones_like(a), jnp.asarray(0.25)))
        )(x, y)
        da, db = jax.jit(jax.grad(lambda a, b: jnp.sum(f(a, b)), argnums=(0, 1)))(x, y)
    np.testing.assert_allclose(
        tangent, (1 - np.asarray(x) * 0.25 / 1.75) / 1.75, rtol=2e-15, atol=0
    )
    np.testing.assert_allclose(da, np.full(3, 1 / 1.75), rtol=2e-15, atol=0)
    np.testing.assert_allclose(db, -np.sum(np.asarray(x)) / 1.75**2, rtol=2e-15, atol=0)
