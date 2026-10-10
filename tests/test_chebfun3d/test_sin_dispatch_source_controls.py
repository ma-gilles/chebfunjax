"""Native sine constructor dispatch, including periodic input and emptiness.

MATLAB source: @chebfun3/sin.m, Chebfun commit7574c77.
"""
import math

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref


def test_periodic_sine_uses_default_constructor_technology():
    domain = (-math.pi, math.pi)*3
    f = chebfun3(lambda x, y, z: .2*jnp.cos(x), domain, trig=True)
    assert f.isPeriodicTech()
    g = f.sin()
    assert not g.isPeriodicTech()
    assert tuple(g.domain) == domain
    x = jnp.asarray([-2., -.5, 1.7])
    zeros = jnp.zeros_like(x)
    assert jnp.max(jnp.abs(g(x, zeros, zeros)-jnp.sin(.2*jnp.cos(x)))) < (
        100*ChebfunPref().cheb3Prefs.chebfun3eps)


def test_empty_sine_remains_empty():
    assert chebfun3().sin().isempty()
