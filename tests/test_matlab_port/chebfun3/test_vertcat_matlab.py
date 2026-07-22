"""Port of MATLAB Chebfun tests/chebfun3/test_vertcat.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_vertcat.m
Chebfun commit: 7574c77

Notes
-----
MATLAB ``vertcat(f, g, h)`` of CHEBFUN3s builds a CHEBFUN3V; chebfunjax
spells this ``Chebfun3v([f, g, h])``.
"""

from __future__ import annotations

import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = 2.220446049250313e-16
TOL = 10 * EPS

DOMS = [(-1, 1, -1, 1, -1, 1), (-1, 0, -2, 1, 2, 4)]


class TestChebfun3Vertcat:
    @pytest.mark.parametrize("dom", DOMS)
    def test_vertcat_three(self, dom):
        f = Chebfun3.from_function(lambda x, y, z: x, domain=dom)
        g = Chebfun3.from_function(lambda x, y, z: y, domain=dom)
        h = Chebfun3.from_function(lambda x, y, z: z, domain=dom)
        F = Chebfun3v.from_functions(lambda x, y, z: x, lambda x, y, z: y,
                                     lambda x, y, z: z, domain=dom)
        assert float((Chebfun3v([f, g, h]) - F).norm()) < TOL

    def test_vertcat_single(self):
        f = Chebfun3.from_function(lambda x, y, z: x)
        assert float((Chebfun3v([f]) - f).norm()) < TOL
