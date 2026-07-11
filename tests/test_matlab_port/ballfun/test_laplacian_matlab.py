"""Port of MATLAB Chebfun tests/ballfun/test_laplacian.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_laplacian.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import X0, val


class TestBallfunLaplacian:
    def test_harmonic_is_zero(self):
        f = Ballfun.from_function(lambda x, y, z: x * y)
        assert abs(val(f.laplacian())) < 1e-8

    def test_angular_dependent_result(self):
        # lap(x z^2) = 2x -- the class the diskfun pipeline got wrong
        f = Ballfun.from_function(lambda x, y, z: x * z * z)
        assert abs(val(f.laplacian()) - 2 * X0) < 1e-8

    def test_radial(self):
        # lap(r^2) = 6
        f = Ballfun.from_function(lambda x, y, z: x * x + y * y + z * z)
        assert abs(val(f.laplacian()) - 6.0) < 1e-8
