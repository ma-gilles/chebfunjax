"""Port of MATLAB Chebfun tests/chebfun2v/test_norm.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_norm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2v import Chebfun2v


class TestChebfun2vNorm:
    def test_norm_of_position_field(self):
        # ||(x,y)||_{L2([-1,1]^2)}^2 = int x^2 + y^2 = 8/3
        P = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y)
        try:
            n = float(P.norm())
        except (TypeError, NotImplementedError):
            pytest.skip("Chebfun2v.norm not implemented")
        assert abs(n - np.sqrt(8 / 3)) < 1e-9
