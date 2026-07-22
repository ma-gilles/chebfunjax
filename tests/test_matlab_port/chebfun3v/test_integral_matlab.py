"""Port of MATLAB Chebfun tests/chebfun3v/test_integral.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_integral.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = 2.220446049250313e-16
TOL = 1e3 * EPS


class TestChebfun3vIntegral:
    def test_line_integral(self):
        # F = [8 x^2 y z; 5 z; -4 x y], curve (t, t^2, t^3) on [0, 1].
        F = Chebfun3v.from_functions(lambda x, y, z: 8 * x ** 2 * y * z,
                                     lambda x, y, z: 5 * z,
                                     lambda x, y, z: -4 * x * y)
        val = F.integral(lambda t: (t, t ** 2, t ** 3), domain=(0, 1))
        assert abs(float(val) - 1.0) < TOL

    def test_line_integral_scaled_domain(self):
        dom = (-1, 1, -1, 1, 0, 4 * np.pi ** 2)
        F = Chebfun3v.from_functions(lambda x, y, z: y,
                                     lambda x, y, z: x,
                                     lambda x, y, z: z, domain=dom)
        val = F.integral(
            lambda t: (np.cos(t), np.sin(t), t ** 2), domain=(0, 2 * np.pi))
        exact = 8 * np.pi ** 4
        assert abs(float(val) - exact) / exact < TOL
