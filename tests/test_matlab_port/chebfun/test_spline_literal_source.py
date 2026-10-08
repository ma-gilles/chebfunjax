"""Actual assertions of tests/chebfun/test_spline.m at Chebfun 7574c77.

Source pass(10) is a constant placeholder, not a numerical test. The other
13 source predicates retain their norms, 10*eps bounds, lengths and domains.
"""

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun

EPS = jnp.finfo(jnp.float64).eps


def test_scalar_source_1_to_3():
    x = jnp.arange(11.0)
    y = jnp.sin(x)
    f = Chebfun.spline(x, y)
    assert float(jnp.linalg.norm(f(x) - y)) < 10 * EPS
    assert len(f.funs) == 10
    assert len(f) == 40


def test_array_source_4_to_6():
    x = jnp.arange(11.0)
    y = jnp.stack((jnp.sin(x), jnp.cos(x)), axis=1)
    f = Chebfun.spline(x, y)
    assert float(jnp.linalg.norm(f(x) - y, ord=2)) < 10 * EPS
    assert len(f.funs) == 10
    assert len(f) == 40


def test_end_slopes_source_7_to_9():
    x = jnp.arange(11.0)
    y = jnp.stack((jnp.sin(x), jnp.cos(x)), axis=1)
    y = jnp.concatenate((jnp.zeros((1, 2)), y, jnp.zeros((1, 2))))
    f = Chebfun.spline(x, y)
    assert float(jnp.linalg.norm(f(x) - y[1:-1], ord=2)) < 10 * EPS
    assert len(f.funs) == 10
    assert len(f) == 40


def test_domain_source_11_to_14():
    domain = (0.01, 10.01)
    x = jnp.arange(11.0)
    y = jnp.stack((jnp.sin(x), jnp.cos(x)), axis=1)
    f = Chebfun.spline(x, y, domain)
    assert f.domain.breakpoints == (domain[0], *range(1, 11), domain[1])
    assert float(jnp.linalg.norm(f(x[1:]) - y[1:], ord=2)) < 10 * EPS
    assert len(f.funs) == 11
    assert len(f) == 44
