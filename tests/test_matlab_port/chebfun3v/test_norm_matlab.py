"""Supplemental analytic position-field norm check.

This is not a port of a native test file: tests/chebfun3v/test_norm.m does
not exist at Chebfun commit 7574c77. The independent integral oracle below
checks the library operation documented in @chebfun3v/norm.m at that pin.
"""

from __future__ import annotations

import numpy as np

from chebfunjax.chebfun3d.chebfun3v import Chebfun3v


class TestChebfun3vNorm:
    def test_norm_of_position_field(self):
        # int_{[-1,1]^3} (x^2+y^2+z^2) = 8 -> norm sqrt(8)
        P = Chebfun3v.from_functions(lambda x, y, z: x,
                                     lambda x, y, z: y,
                                     lambda x, y, z: z)
        n = float(P.norm())
        assert abs(n - np.sqrt(8.0)) < 1e-8
