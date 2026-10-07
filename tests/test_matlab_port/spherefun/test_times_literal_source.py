"""All three literal MATLAB spherefun multiplication clauses.

Provenance
----------
MATLAB source : tests/spherefun/test_times.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
import jax.numpy as jnp

from chebfunjax.chebpref import ChebfunPref

from ._cart import sph_xyz


def test_source_three_times_clauses():
    tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
    f = sph_xyz(lambda x, y, z: jnp.sin(x*y*z))
    assert (f*f - f**2).norm(jnp.inf) < tol
    z = sph_xyz(lambda x, y, z: z)
    f = sph_xyz(lambda x, y, z: 1-z**2)
    g = z*f
    assert not g.nonzero_poles
    g = f*z
    assert not g.nonzero_poles
