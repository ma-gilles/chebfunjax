"""Port of MATLAB Chebfun tests/chebfun2v/test_minandmax2est.m (Fable 5).

The MATLAB empty-chebfun2v case (pass 1) is covered by the empty-
representations workstream; the value assertions (passes 2-5) are
ported here.

Provenance
----------
MATLAB source : tests/chebfun2v/test_minandmax2est.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

TOL = 1000 * float(np.finfo(np.float64).eps)


class TestChebfun2vMinandmax2est:
    def test_two_components_box(self):
        # pass(4)-(5): F = (x, y) on [-2,-1]x[3,6] -> box [-2 -1 3 6].
        F = Chebfun2v.from_functions(lambda x, y: x + 0 * y,
                                     lambda x, y: y + 0 * x,
                                     domain=(-2.0, -1.0, 3.0, 6.0))
        box = F.minandmax2est()
        assert len(box) == 4
        assert float(np.max(np.abs(np.asarray(box)
                                   - np.array([-2.0, -1.0, 3.0, 6.0])))) \
            < 10 * TOL
