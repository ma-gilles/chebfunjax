"""Port of MATLAB Chebfun tests/chebfun3/test_std3.m (Fable 5).

FIXED: Chebfun3.std3 added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun3/test_std3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3


class TestChebfun3Std3:
    def test_std3_of_x(self):
        f = Chebfun3.from_function(lambda x, y, z: x)
        # var(x) over [-1,1]^3 = 1/3
        assert abs(float(f.std3()) - np.sqrt(1 / 3)) < 1e-10
