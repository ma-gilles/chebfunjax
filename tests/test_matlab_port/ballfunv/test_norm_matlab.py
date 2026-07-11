"""Port of MATLAB Chebfun tests/ballfunv/test_norm.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfunv/test_norm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.ballfun.ballfunv import Ballfunv


class TestBallfunvNorm:
    def test_norm_of_coordinate_field(self):
        # ||(x,y,z)||^2 = int r^2 = 4 pi/5 -> norm = sqrt(4 pi/5)
        P = Ballfunv.from_functions(lambda x, y, z: x,
                                    lambda x, y, z: y,
                                    lambda x, y, z: z)
        assert abs(float(P.norm()) - np.sqrt(4 * np.pi / 5)) < 1e-8
