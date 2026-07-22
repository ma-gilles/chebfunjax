"""Port of MATLAB Chebfun tests/chebfun3/test_mtimes.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_mtimes.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

from ._helpers import EPS, maxdiff

TOL = 1e4 * EPS


class TestChebfun3Mtimes:
    def test_scalar_mtimes(self):
        f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x * y * z))
        assert maxdiff(3 * f,
                       lambda x, y, z: 3 * jnp.cos(x * y * z)) < TOL
        assert maxdiff(f * 3,
                       lambda x, y, z: 3 * jnp.cos(x * y * z)) < TOL

    def test_chebfun3_times_chebfun3v(self):
        # A scalar CHEBFUN3 scales every component of a CHEBFUN3V
        # (MATLAB @chebfun3v/times, invoked as f .* F / f * F).
        f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x * y * z))
        F = Chebfun3v.from_functions(lambda x, y, z: x,
                                     lambda x, y, z: y,
                                     lambda x, y, z: z)
        expect = Chebfun3v.from_functions(
            lambda x, y, z: x * jnp.cos(x * y * z),
            lambda x, y, z: y * jnp.cos(x * y * z),
            lambda x, y, z: z * jnp.cos(x * y * z))
        assert float((f * F - expect).norm()) < TOL
        assert float((F * f - expect).norm()) < TOL
