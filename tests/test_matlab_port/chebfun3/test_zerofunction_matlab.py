"""Port of MATLAB Chebfun tests/chebfun3/test_zerofunction.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_zerofunction.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)


class TestChebfun3Zerofunction:
    def test_all_matlab_assertions(self):
        tol = 100 * ChebfunPref().cheb3Prefs.chebfun3eps
        f = chebfun3(0)
        f = chebfun3(lambda x, y, z: 0 * x)
        f = chebfun3(0, (-2, 2, -3, 3, -4, 4))
        f = chebfun3(0)
        g = chebfun3(lambda x, y, z: jnp.cos(x * y * z))
        assert abs(float((g + f).norm()) - float(g.norm())) < tol           # pass(1)
        v = abs(float(f(np.pi / 6, np.pi / 6, np.pi / 6)))
        v = v + sum(f.rank)
        assert v == 0                                                        # pass(2)
        v = abs(float(f.sum().norm()))
        v = v + abs(float(f.sum3()))
        v = v + abs(float(f.diff().norm()))
        v = v + abs(float(f.diff(1, 1).norm()))
        assert v == 0                                                        # pass(3)
        r = jnp.asarray(np.random.RandomState(0).rand(10, 8))
        v = f(r, r, r)
        assert tuple(v.shape) == tuple(r.shape)                              # pass(4)
