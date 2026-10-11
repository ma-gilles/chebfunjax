"""Supplemental representation/phase controls; not native predicate counts."""

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.fun.singfun import Singfun

EPS = np.finfo(float).eps


def test_tiny_endpoint_power_retains_source_simplified_factor():
    x = cj.chebfun("x")
    g = (1 - x) ** EPS
    assert len(g) == 1
    assert not isinstance(g.funs[0].tech, Singfun)
    np.testing.assert_array_equal(np.asarray(g.funs[0].tech.coeffs).reshape(-1), np.ones(1))
    assert float(g(1.0)) == 1.0
    h = (1 - x) ** 1e-13
    assert isinstance(h.funs[0].tech, Singfun)
    assert h.funs[0].tech.exponents[1] == 1e-13


def test_root_free_positive_and_negative_principal_phase():
    x = cj.chebfun("x")
    base = (2 + x) ** 2
    positive = base**0.5
    negative = (-base) ** 0.5
    sites = jnp.array([-1.0, -0.75, 0.0, 0.75, 1.0])
    # Independent exact linear square-root oracle; diagnostic float64 bound.
    assert float(jnp.max(jnp.abs(positive(sites) - (2 + sites)))) < 100 * EPS * 3
    assert float(jnp.max(jnp.abs(negative(sites) - 1j * (2 + sites)))) < 100 * EPS * 3
    assert not isinstance(positive.funs[0].tech, Singfun)
    assert not isinstance(negative.funs[0].tech, Singfun)
