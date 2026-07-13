"""Port of MATLAB Chebfun tests/ballfun/test_mean3.m (Fable 5).

FIXED: Ballfun.mean3 added in the Fable 5 audit
(sum over the ball / its volume).

Provenance
----------
MATLAB source : tests/ballfun/test_mean3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.ballfun.ballfun import Ballfun

TOL = 1e-12


class TestBallfunMean3:
    def test_constant(self):
        # pass(1)
        f = Ballfun.from_function(lambda r, lam, th: 1.0 + 0 * r,
                                  spherical=True)
        assert abs(f.mean3() - 1.0) < TOL

    def test_r_squared(self):
        # mean of r^2 over the unit ball is 3/5
        g = Ballfun.from_function(
            lambda x, y, z: x ** 2 + y ** 2 + z ** 2)
        assert abs(g.mean3() - 3.0 / 5.0) < TOL
