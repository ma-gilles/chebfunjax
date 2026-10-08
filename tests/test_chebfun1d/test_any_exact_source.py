"""Exact anyDim1 coefficient/point rules from Chebfun7574c77 any.m."""

import jax.numpy as jnp

import chebfunjax as cj


def test_tiny_coefficients_are_nonzero():
    f = cj.chebfun(lambda x: 1e-30 * (1 + x * x))
    assert bool(f.any())


def test_isolated_nonzero_point_counts():
    f = cj.chebfun(lambda x: 0 * x, domain=(-1, 0, 1))
    f = f.set_point_values(jnp.asarray([[0.0], [1e-30], [0.0]]))
    assert bool(f.any())


def test_nan_ignored_per_column():
    f = cj.chebfun(lambda x: jnp.stack([0 * x, 1e-30 + 0 * x], axis=-1))
    f = f.set_point_values(jnp.full((2, 2), jnp.nan))
    assert jnp.array_equal(f.any(), jnp.asarray([False, True]))
