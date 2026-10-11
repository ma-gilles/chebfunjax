"""Compatibility and independent controls for the sampled JAX adapter."""

import jax
import jax.numpy as jnp

from chebfunjax.utils.lebesgue import bary_weights, lebesgue_function


def test_legacy_unnormalised_weights():
    assert jnp.array_equal(bary_weights([-1.0, 0.0, 1.0]), jnp.array([0.5, -1.0, 0.5]))


def test_sampled_jax_trace_matches_lagrange_polynomials():
    nodes = jnp.array([-1.0, 0.0, 1.0])
    grid, values = jax.jit(lambda x: lebesgue_function(x, n_eval=17))(nodes)
    expected = (
        jnp.abs(grid * (grid - 1) / 2) + jnp.abs(1 - grid**2) + jnp.abs(grid * (grid + 1) / 2)
    )
    assert values.shape == grid.shape == (17,)
    assert jnp.max(jnp.abs(values - expected)) < 10 * jnp.finfo(jnp.float64).eps
    assert jnp.array_equal(values[jnp.array([0, 8, 16])], jnp.ones(3))
