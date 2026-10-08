"""Source constructor dependencies exposed by minimax clauses11 and17.

Provenance: @chebfun/addBreaks.m, getRootsForBreaks.m,
tests/chebfun/test_addBreaks.m; Chebfun7574c77.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax import chebfun


@pytest.mark.parametrize('expression', ['1e-100*exp(x)', '1E-100*exp(t)', '1.e-100*exp(z)'])
def test_scientific_literal_does_not_become_variable(expression):
    f = chebfun(expression)
    x = jnp.asarray([-.9, .3, .8])
    np.testing.assert_allclose(f(x)/1e-100, jnp.exp(x), rtol=5e-15, atol=0)


@pytest.mark.parametrize('array', [False, True])
def test_original_add_breaks_clauses(array):
    op = (lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1)) if array else jnp.sin
    f = chebfun(op, domain=[-1, -.5, 0, .5, 1])
    g = f.addBreaks([-.25, .25])
    assert list(g.domain.breakpoints) == [-1, -.5, -.25, 0, .25, .5, 1]
    x = jnp.asarray(2*np.random.RandomState(6178).rand(100)-1)
    assert float(jnp.max(jnp.abs(f(x)-g(x)))) < 10*float(f.vscale)*np.finfo(float).eps


def test_source_existing_break_proximity_and_requested_tolerance():
    f = abs(chebfun('x')-.1)
    assert f.addBreaks([np.nextafter(.1, 1)]) is f
    assert f.addBreaks([.1001], tol=.001) is f
    g = f.sqrt()
    x = jnp.asarray([-.9, .1, .7])
    np.testing.assert_allclose(g(x), jnp.sqrt(jnp.abs(x-.1)), rtol=2e-14, atol=2e-14)
