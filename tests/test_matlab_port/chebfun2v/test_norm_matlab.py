"""Supplemental analytic position-field norm check.

This is not a port of a native test file: tests/chebfun2v/test_norm.m does
not exist at Chebfun commit 7574c77. The independent integral oracle below
checks the library operation documented in @chebfun2v/norm.m at that pin.
"""

from __future__ import annotations

import numpy as np

from chebfunjax.chebfun2d.chebfun2v import Chebfun2v


class TestChebfun2vNorm:
    def test_norm_of_position_field(self):
        # ||(x,y)||_{L2([-1,1]^2)}^2 = int x^2 + y^2 = 8/3
        P = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y)
        n = float(P.norm())
        assert abs(n - np.sqrt(8 / 3)) < 1e-9
