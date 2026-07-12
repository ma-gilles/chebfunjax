"""Port of MATLAB Chebfun tests/chebfun3/test_mean3.m (Fable 5).

FIXED: Chebfun3.mean3 added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun3/test_mean3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.chebfun3d.chebfun3 import Chebfun3


class TestChebfun3Mean3:
    def test_mean3_of_r2(self):
        f = Chebfun3.from_function(lambda x, y, z: x * x + y * y + z * z)
        assert abs(float(f.mean3()) - 1.0) < 1e-12
