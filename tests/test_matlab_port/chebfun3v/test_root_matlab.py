"""Port of MATLAB Chebfun tests/chebfun3v/test_root.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_root.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = 2.220446049250313e-16
TOL = 1e3 * EPS


class TestChebfun3vRoot:
    def test_one_common_root(self):
        f = Chebfun3.from_function(lambda x, y, z: y - x ** 2)
        g = Chebfun3.from_function(lambda x, y, z: z - x ** 3)
        h = Chebfun3.from_function(
            lambda x, y, z: jnp.cos(jnp.exp(x * jnp.sin(-2 + y + z))))
        F = Chebfun3v([f, g, h])
        r = F.root()

        assert np.asarray(r).size == 3
        vals = np.abs(np.asarray(F(r[0], r[1], r[2])))
        assert np.all(vals < TOL)
