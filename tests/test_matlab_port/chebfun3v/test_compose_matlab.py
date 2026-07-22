"""Port of MATLAB Chebfun tests/chebfun3v/test_compose.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_compose.m
Chebfun commit: 7574c77

Notes
-----
The MATLAB test's periodic branches (pass 7-10) build 'trig' CHEBFUN3s and
check isPeriodicTech on the composition.  chebfunjax has no trigonometric
tech, so those cases are omitted; the six non-periodic compositions are
ported.
"""

from __future__ import annotations

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = 2.220446049250313e-16
TOL = 1000 * EPS

UNIT = (0, 1, 0, 1, 0, 1)


class TestChebfun3vCompose:
    def test_three_components_with_chebfun3(self):
        F = Chebfun3v.from_functions(lambda x, y, z: x, lambda x, y, z: y,
                                     lambda x, y, z: z, domain=UNIT)
        g = Chebfun3.from_function(lambda x, y, z: x + y + z, domain=UNIT)
        h = F.compose(g)
        h_true = Chebfun3.from_function(lambda x, y, z: x + y + z, domain=UNIT)
        assert float((h - h_true).norm()) < TOL

    def test_three_components_with_two_component_chebfun3v(self):
        F = Chebfun3v.from_functions(lambda x, y, z: x, lambda x, y, z: y,
                                     lambda x, y, z: z)
        G = Chebfun3v.from_functions(lambda x, y, z: x + y + z,
                                     lambda x, y, z: x + z)
        H = F.compose(G)
        assert float((H - G).norm()) < TOL

    def test_three_components_with_three_component_chebfun3v(self):
        F = Chebfun3v.from_functions(lambda x, y, z: x, lambda x, y, z: y,
                                     lambda x, y, z: z)
        G = Chebfun3v.from_functions(lambda x, y, z: x + y + z,
                                     lambda x, y, z: x + z,
                                     lambda x, y, z: x - y)
        H = F.compose(G)
        assert float((H - G).norm()) < TOL

    def test_two_components_with_chebfun2(self):
        F = Chebfun3v.from_functions(lambda x, y, z: x, lambda x, y, z: y)
        g = Chebfun2.from_function(lambda x, y: x + y)
        h = F.compose(g)
        h_true = Chebfun3.from_function(lambda x, y, z: x + y)
        assert float((h - h_true).norm()) < TOL

    def test_two_components_with_two_component_chebfun2v(self):
        F = Chebfun3v.from_functions(lambda x, y, z: x, lambda x, y, z: y)
        G = Chebfun2v.from_functions(lambda x, y: x + y, lambda x, y: x - y)
        H = F.compose(G)
        H_true = Chebfun3v.from_functions(lambda x, y, z: x + y,
                                          lambda x, y, z: x - y)
        assert float((H - H_true).norm()) < TOL

    def test_two_components_with_three_component_chebfun2v(self):
        F = Chebfun3v.from_functions(lambda x, y, z: x, lambda x, y, z: y)
        G = Chebfun2v.from_functions(lambda x, y: x + y, lambda x, y: x - y,
                                     lambda x, y: x * y)
        H = F.compose(G)
        H_true = Chebfun3v.from_functions(lambda x, y, z: x + y,
                                          lambda x, y, z: x - y,
                                          lambda x, y, z: x * y)
        assert float((H - H_true).norm()) < TOL
