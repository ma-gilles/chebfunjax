"""Port of MATLAB Chebfun tests/ballfun/test_sum3.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_sum3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun


class TestBallfunSum3:
    def test_volume(self):
        one = Ballfun.from_function(lambda x, y, z: 1.0 + 0 * x)
        assert abs(float(one.sum()) - 4 * np.pi / 3) < 1e-9

    def test_odd_moment_zero(self):
        f = Ballfun.from_function(lambda x, y, z: x)
        assert abs(float(f.sum())) < 1e-10

    def test_r2_moment(self):
        # int_ball (x^2+y^2+z^2) = 4 pi / 5
        f = Ballfun.from_function(lambda x, y, z: x * x + y * y + z * z)
        assert abs(float(f.sum()) - 4 * np.pi / 5) < 1e-9
