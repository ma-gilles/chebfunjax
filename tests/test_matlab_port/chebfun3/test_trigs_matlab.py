"""Port of MATLAB Chebfun tests/chebfun3/test_trigs.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_trigs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)

D2R = np.pi / 180.0


class TestChebfun3Trigs:
    def test_all_matlab_assertions(self):
        tol = 100 * ChebfunPref().cheb3Prefs.chebfun3eps
        fns = [("cos", jnp.cos), ("sin", jnp.sin), ("tan", jnp.tan),
               ("cosh", jnp.cosh), ("sinh", jnp.sinh), ("tanh", jnp.tanh),
               ("tand", lambda t: jnp.tan(D2R * t))]
        g = chebfun3(lambda x, y, z: x * y ** 2 * z ** 3)
        for name, fn in fns:
            h = getattr(g, name)()
            exact = chebfun3(lambda x, y, z, _fn=fn: _fn(x * y ** 2 * z ** 3))
            assert float((h - exact).norm()) < tol, name                     # pass(1)-(7)
