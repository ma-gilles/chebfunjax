"""Both original assertions from MATLAB tests/chebfun3/test_max3.m.

Chebfun commit: 7574c77. Preserve the original inputs, construction order,
shared one-dimensional Chebfun reference, default norm and 100*factoryeps.
Python max3 returns (value, location); select its first output. The unused
y/z formal arguments of the native constant callback are omitted.
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref


def test_native_max3_both_original_assertions():
    tol = 100*ChebfunPref().cheb3Prefs.chebfun3eps
    f = chebfun3(lambda x, y, z: jnp.cos(x*y*z))
    g = chebfun(lambda x: 1+0*x)
    h1 = float(f.max3()[0])
    assert (h1-g).norm() < tol
    f = chebfun3(lambda x, y, z: jnp.sin(x+y+z))
    h2 = float(f.max3()[0])
    assert (h2-g).norm() < tol
