"""Port of MATLAB Chebfun tests/chebfun3/test_norm.m (Fable 5).

FIXED: Chebfun3.norm added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun3/test_norm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3


class TestChebfun3Norm:
    def test_norm_of_x(self):
        f = Chebfun3.from_function(lambda x, y, z: x)
        assert abs(float(f.norm()) - np.sqrt(8 / 3)) < 1e-12
