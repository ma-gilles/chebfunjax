"""Native @chebfun/feval.m empty-F early return, Chebfun pin 7574c77."""

import jax
import jax.numpy as jnp
import pytest

import chebfunjax as cj


@pytest.mark.parametrize("factory", [cj.chebfun, lambda: cj.chebfun([]),
                                     lambda: cj.chebfun(jnp.sin, n=0)])
def test_empty_chebfun_evaluation_returns_empty(factory):
    # Empty F takes precedence over query shape, string/side parsing and
    # composition. This API represents native [] as an empty float64 vector.
    f = factory()
    for query, side in [(jnp.array([0.0, 1.0]), None),
                        (jnp.empty((0, 3)), None),
                        ("invalid", "invalid"),
                        (jnp.array(0.0), "invalid"),
                        (f, None)]:
        result = f(query, side)
        assert result.shape == (0,)
        assert result.dtype == jnp.float64
    assert jax.jit(lambda x: f(x))(jnp.array([0.0])).shape == (0,)
