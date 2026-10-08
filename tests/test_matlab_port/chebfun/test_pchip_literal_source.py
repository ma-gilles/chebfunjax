"""All twelve assertions of Chebfun7574c77 tests/chebfun/test_pchip.m."""
import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun

EPS = jnp.finfo(jnp.float64).eps


def test_scalar_source_1_to_3():
    x = jnp.arange(11.0)
    y = jnp.sin(x)
    f = Chebfun.pchip(x, y)
    assert float(jnp.linalg.norm(f(x) - y)) < 10 * EPS
    assert len(f.funs) == 10
    assert len(f) == 40


def test_array_source_4_to_6():
    x = jnp.arange(11.0)
    y = jnp.stack((jnp.sin(x), jnp.cos(x)), axis=1)
    f = Chebfun.pchip(x, y)
    assert float(jnp.linalg.norm(f(x) - y, ord=2)) < 10 * EPS
    assert len(f.funs) == 10
    assert len(f) == 40


def test_domain_source_7_to_10():
    domain = (0.01, 10.01)
    x = jnp.arange(11.0)
    y = jnp.stack((jnp.sin(x), jnp.cos(x)), axis=1)
    f = Chebfun.pchip(x, y, domain)
    assert f.domain.breakpoints == (domain[0], *range(1, 11), domain[1])
    assert float(jnp.linalg.norm(f(x[1:]) - y[1:], ord=2)) < 10 * EPS
    assert len(f.funs) == 11
    assert len(f) == 44


def test_plateau_example_source_11_to_12():
    x = jnp.arange(-3.0, 4.0)
    y = jnp.asarray([-1., -1., -1., 0., 1., 1., 1.])
    f = Chebfun.pchip(x, y)
    assert float(jnp.linalg.norm(f(x) - y)) < 10 * EPS
    assert len(f.funs) == 6
