"""Port of MATLAB Chebfun tests/chebfun3/test_laplacian.m (Fable 5).

FIXED (Fable 5): Chebfun3.laplacian / lap added in the audit.

Provenance
----------
MATLAB source : tests/chebfun3/test_laplacian.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

from ._helpers import EPS, maxdiff

TOL = 50 * EPS


class TestChebfun3Laplacian:
    def test_all_matlab_assertions(self):
        # pass(1): laplacian matches the sum of second partials.
        F = Chebfun3.from_function(
            lambda x, y, z: jnp.cos(x * z) + jnp.sin(y * z))
        lapF = F.laplacian()
        lapF1 = F.diff(1, 2) + F.diff(2, 2) + F.diff(3, 2)
        assert maxdiff(lapF, lambda x, y, z: lapF1(x, y, z)) < TOL

        # pass(2): second definition check.
        F = Chebfun3.from_function(
            lambda x, y, z: jnp.cos(x) + y * z + z ** 2)
        lapF = F.laplacian()
        lapF1 = F.diff(1, 2) + F.diff(2, 2) + F.diff(3, 2)
        assert maxdiff(lapF, lambda x, y, z: lapF1(x, y, z)) < TOL
