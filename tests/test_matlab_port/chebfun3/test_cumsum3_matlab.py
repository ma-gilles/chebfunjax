"""Port of MATLAB Chebfun tests/chebfun3/test_cumsum3.m (Fable 5).

FIXED (Fable 5): Chebfun3.cumsum3 added in the audit.

Provenance
----------
MATLAB source : tests/chebfun3/test_cumsum3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

from ._helpers import EPS, maxdiff

TOL = 100 * EPS


class TestChebfun3Cumsum3:
    def test_all_matlab_assertions(self):
        # f = x on the cube; cumsum3 = (y+1)(z+1)(x^2/2 - 1/2).
        f = Chebfun3.from_function(lambda x, y, z: x)
        g = f.cumsum3()
        assert maxdiff(
            g, lambda x, y, z: (y + 1) * (z + 1)
            * (x ** 2 / 2 - 0.5)) < TOL
