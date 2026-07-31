"""Port of MATLAB Chebfun tests/chebfun2v/test_threecomponents.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_threecomponents.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2, chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

TOL = 1e5 * float(np.finfo(np.float64).eps)
PTS = [(-0.5, 0.3), (0.2, -0.7), (0.8, 0.6)]


def _maxdiff(F, G):
    out = 0.0
    for a, b in zip(F.components, G.components):
        fa, fb = Chebfun2(approx=a), Chebfun2(approx=b)
        out = max(out, max(
            abs(float(np.asarray(fa(x, y))) - float(np.asarray(fb(x, y))))
            for x, y in PTS))
    return out


class TestChebfun2vThreeComponents:
    def test_from_objects_matches_handles(self):
        # pass(1): chebfun2v(f,f,f) from chebfun2 objects == from handles.
        f = chebfun2(lambda x, y: x + 0 * y)
        F1 = Chebfun2v(components=[f.approx, f.approx, f.approx])
        F2 = Chebfun2v.from_functions(lambda x, y: x + 0 * y,
                                      lambda x, y: x + 0 * y,
                                      lambda x, y: x + 0 * y)
        assert _maxdiff(F1, F2) < TOL

    def test_scaled_components(self):
        # pass(3)-ish: G = chebfun2v(f, 2f, 3f) has the scaled components.
        f = chebfun2(lambda x, y: x + 0 * y)
        G = Chebfun2v(components=[f.approx, (f * 2.0).approx,
                                  (f * 3.0).approx])
        for k, scale in enumerate((1.0, 2.0, 3.0)):
            g = Chebfun2(approx=G.components[k])
            for x, y in PTS:
                assert abs(float(np.asarray(g(x, y))) - scale * x) < TOL
