"""Original assertions from Chebfun tests/chebfun3/test_max2.m, 7574c77.

Original functions, construction order, norms and thresholds are retained.
Python None represents the native empty second argument for max2/min2.
Passing these predicates does not establish full extrema algorithm parity.
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref


def test_native_max2():
    tol = 1e7*ChebfunPref().cheb3Prefs.chebfun3eps
    f = chebfun3(lambda x, y, z: jnp.cos(x*y*z))
    g = chebfun(lambda x: 1+0*x)
    h1 = f.max2()
    h2 = f.max2(None)
    h3 = f.max2(None, (1, 2))
    h4 = f.max2(None, (2, 1))
    h5 = f.max2(None, (1, 3))
    h6 = f.max2(None, (3, 1))
    h7 = f.max2(None, (2, 3))
    h8 = f.max2(None, (3, 2))

    assert (h1-g).norm() < tol
    assert (h2-g).norm() < tol
    assert (h3-g).norm() < tol
    assert (h4-g).norm() < tol
    assert (h5-g).norm() < tol
    assert (h6-g).norm() < tol
    assert (h7-g).norm() < tol
    assert (h8-g).norm() < tol
