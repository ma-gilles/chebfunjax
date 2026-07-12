"""Port of MATLAB Chebfun tests/chebfun3/test_min3.m (Fable 5).

FIXED: Chebfun3.min3 added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun3/test_min3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.chebfun3d.chebfun3 import Chebfun3


class TestChebfun3Min3:
    def test_min3_of_quadratic(self):
        g = Chebfun3.from_function(
            lambda x, y, z: (x - 0.1) ** 2 + y ** 2 + z ** 2)
        v, loc = g.min3()
        assert abs(float(v)) < 1e-8
