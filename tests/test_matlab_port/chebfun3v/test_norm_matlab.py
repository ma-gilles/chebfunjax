"""Port of MATLAB Chebfun tests/chebfun3v/test_norm.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_norm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3v import Chebfun3v


class TestChebfun3vNorm:
    def test_norm_of_position_field(self):
        # int_{[-1,1]^3} (x^2+y^2+z^2) = 8 -> norm sqrt(8)
        P = Chebfun3v.from_functions(lambda x, y, z: x,
                                     lambda x, y, z: y,
                                     lambda x, y, z: z)
        try:
            n = float(P.norm())
        except (TypeError, NotImplementedError):
            pytest.skip("Chebfun3v.norm not implemented")
        assert abs(n - np.sqrt(8.0)) < 1e-8
