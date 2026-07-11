"""Port of MATLAB Chebfun tests/chebfun3/test_sum3.m (Fable 5).

pass(1) (empty chebfun3) and pass(3) (cheb.gallery3) are skipped:
chebfunjax has neither.

Provenance
----------
MATLAB source : tests/chebfun3/test_sum3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

from ._helpers import EPS

TOL = 1e4 * EPS


class TestChebfun3Sum3:
    def test_empty(self):
        pytest.skip("chebfunjax has no empty Chebfun3")

    def test_constant(self):
        f = Chebfun3.from_function(lambda x, y, z: 1.0 + 0 * x)
        assert abs(float(f.sum3()) - 8.0) < TOL

    def test_gallery3_runge(self):
        pytest.skip("chebfunjax has no cheb.gallery3")

    def test_x_on_box(self):
        dom = (0.0, 1.0, 0.0, 2.0, 0.0, 3.0)
        f = Chebfun3.from_function(lambda x, y, z: x, domain=dom)
        assert abs(float(f.sum3()) - 3.0) < TOL

    def test_y_on_box(self):
        dom = (0.0, 1.0, 0.0, 2.0, 0.0, 3.0)
        f = Chebfun3.from_function(lambda x, y, z: y, domain=dom)
        assert abs(float(f.sum3()) - 6.0) < TOL

    def test_z_on_box(self):
        dom = (0.0, 1.0, 0.0, 2.0, 0.0, 3.0)
        f = Chebfun3.from_function(lambda x, y, z: z, domain=dom)
        assert abs(float(f.sum3()) - 9.0) < TOL
