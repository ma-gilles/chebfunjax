"""Supplemental canonical-domain truncation controls (native7574c77)."""
import jax.numpy as jnp
import pytest

from chebfunjax import chebfun
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize('domain', [(-1., 1.), (0., 2*jnp.pi), (2., 7.)])
@pytest.mark.parametrize('complex_values', [False, True])
def test_periodic_shift_scale(domain, complex_values):
    a, b = domain

    def op(x):
        t = (2*x-a-b)/(b-a)
        return (1 + .2*jnp.cos(jnp.pi*t)
                + (.3j if complex_values else .3)*jnp.sin(2*jnp.pi*t))

    f = chebfun(op, domain=domain, trig=True)
    g = f.truncate(9)
    x = jnp.linspace(a, b, 101)
    assert tuple(g.domain.breakpoints) == domain
    assert len(g) == 9
    assert isinstance(g.funs[0].tech, Trigtech)
    assert float(jnp.max(jnp.abs(g(x)-op(x)))) < 100*jnp.finfo(float).eps
    assert float((f-g).norm(2)) < 100*jnp.finfo(float).eps


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
def test_polynomial_tech_and_domain(tech):
    f = chebfun(lambda x: x*x+2*x+1, domain=(2., 7.), tech=tech)
    g = f.truncate(3)
    x = jnp.linspace(2., 7., 31)
    assert type(g.funs[0].tech) is tech
    assert tuple(g.domain.breakpoints) == (2., 7.)
    assert len(g) == 3
    assert float(jnp.max(jnp.abs(g(x)-(x*x+2*x+1)))) < 100*jnp.finfo(float).eps*64
