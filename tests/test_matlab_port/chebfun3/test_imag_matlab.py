"""Original two predicates from tests/chebfun3/test_imag.m, Chebfun7574c77."""
from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref

TOL = 100 * ChebfunPref().cheb3Prefs.chebfun3eps


def test_native_imag():
    f = chebfun3(lambda x, y, z: jnp.cos(x*y*z))
    g = chebfun3(lambda x, y, z: jnp.sin(x+y**2+z**3))
    x = jnp.linspace(-1., 1., 3)
    xx, yy, zz = jnp.meshgrid(x, x, x, indexing='ij')
    h = f+1j*g
    fv = h(xx, yy, zz)
    gv = g(xx, yy, zz)
    assert jnp.linalg.norm(jnp.imag(fv.reshape(-1, order='F'))-gv.reshape(-1, order='F')) < TOL
    assert (h.imag()-g).norm() < TOL
