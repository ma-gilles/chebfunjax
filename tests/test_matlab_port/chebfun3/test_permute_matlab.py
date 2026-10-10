"""Original ten predicates from tests/chebfun3/test_permute.m, Chebfun7574c77."""
from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref

TOL = 1000 * ChebfunPref().cheb3Prefs.chebfun3eps


def test_native_permute_symmetric():
    f = chebfun3(lambda x, y, z: jnp.cos(x*y*z))
    for order in [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]:
        assert (f-f.permute(order)).norm() < TOL


def test_native_permute_complex():
    pi = float(jnp.pi)
    f = chebfun3(lambda x, y, z: jnp.cos(x)*jnp.sin(y)+1j*z, (-3, 4, -1, 0, -pi, pi))
    assert (f-f.permute([1, 2, 3])).norm() < TOL
    g132 = chebfun3(lambda x, z, y: jnp.cos(x)*jnp.sin(y)+1j*z, (-3, 4, -pi, pi, -1, 0))
    assert (g132-f.permute([1, 3, 2])).norm() < TOL
    g312 = chebfun3(lambda z, x, y: jnp.cos(x)*jnp.sin(y)+1j*z, (-pi, pi, -3, 4, -1, 0))
    assert (g312-f.permute([3, 1, 2])).norm() < TOL
    g213 = chebfun3(lambda y, x, z: jnp.cos(x)*jnp.sin(y)+1j*z, (-1, 0, -3, 4, -pi, pi))
    assert (g213-f.permute([2, 1, 3])).norm() < TOL
