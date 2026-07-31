"""Port of MATLAB Chebfun tests/chebfun2v/test_compose.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_compose.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

TOL = 1000 * float(np.finfo(np.float64).eps)


class TestChebfun2vCompose:
    def test_compose_with_chebfun2(self):
        # pass(1): F = (x, y) on [0,1]^2; g = x + y; compose(F,g) == g there.
        F = Chebfun2v.from_functions(lambda x, y: x + 0 * y,
                                     lambda x, y: y + 0 * x,
                                     domain=(0.0, 1.0, 0.0, 1.0))
        g = chebfun2(lambda x, y: x + y)
        h = F.compose(g)
        pts = [(0.2, 0.7), (0.5, 0.5), (0.9, 0.1)]
        for x, y in pts:
            assert abs(float(np.asarray(h(x, y))) - (x + y)) < TOL

    def test_compose_with_chebfun2v(self):
        # pass(2): G = (x+y, x-y); compose(F, G) == G for F = (x, y).
        F = Chebfun2v.from_functions(lambda x, y: x + 0 * y,
                                     lambda x, y: y + 0 * x)
        G = Chebfun2v.from_functions(lambda x, y: x + y,
                                     lambda x, y: x - y)
        H = F.compose(G)
        from chebfunjax.chebfun2d.chebfun2 import Chebfun2
        for k, ref in enumerate((lambda x, y: x + y, lambda x, y: x - y)):
            h = Chebfun2(approx=H.components[k])
            for x, y in [(0.2, 0.7), (-0.5, 0.4)]:
                assert abs(float(np.asarray(h(x, y))) - ref(x, y)) < TOL
