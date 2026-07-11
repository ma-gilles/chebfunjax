"""Port of MATLAB Chebfun tests/ballfun/test_norm.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_norm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun


class TestBallfunNorm:
    def test_norm_of_x(self):
        # ||x||_{L2(ball)} = sqrt(4 pi / 15)
        f = Ballfun.from_function(lambda x, y, z: x)
        assert abs(float(f.norm()) - np.sqrt(4 * np.pi / 15)) < 1e-8

    def test_norm_of_one(self):
        one = Ballfun.from_function(
            lambda x, y, z: 1.0 + 0 * x)
        assert abs(float(one.norm()) - np.sqrt(4 * np.pi / 3)) < 1e-8
