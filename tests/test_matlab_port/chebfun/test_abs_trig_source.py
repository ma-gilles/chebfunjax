"""Original periodic abs clauses12--13 with unchanged source bounds.

Provenance
----------
MATLAB source: tests/chebfun/test_abs.m, pass(12,:) and pass(13,:)
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech2


def test_original_periodic_abs_technology_and_norm():
    saved = ChebfunPref()
    try:
        ChebfunPref.setDefaults("factory")
        f = cj.chebfun(lambda x: jnp.sin(3*x), domain=(0., 2*jnp.pi), trig=True)
        h = cj.chebfun(lambda x: jnp.sin(3*x), domain=(0., 2*jnp.pi))
        g = f.abs()
        # The source checks the first FUN's technology against pref.tech.
        assert isinstance(g.funs[0].tech, Chebtech2)
        assert float((g-h.abs()).norm(jnp.inf)) < 1e4*h.vscale*2**-52
    finally:
        ChebfunPref.setDefaults(saved)
