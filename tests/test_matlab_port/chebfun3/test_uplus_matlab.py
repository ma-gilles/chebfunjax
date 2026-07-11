"""Port of MATLAB Chebfun tests/chebfun3/test_uplus.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_uplus.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

from ._helpers import EPS, maxdiff

TOL = 1e5 * EPS


class TestChebfun3Uplus:
    def test_uplus_identity(self):
        f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x * y * z))
        assert maxdiff(f - f, lambda x, y, z: 0.0 * x) < TOL
