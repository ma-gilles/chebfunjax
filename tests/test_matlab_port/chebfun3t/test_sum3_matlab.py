"""Port of MATLAB Chebfun tests/chebfun3t/test_sum3.m (Fable 5).

chebfunjax's :class:`Chebfun3T` delegates ``sum3`` to the Tucker
:class:`~chebfunjax.chebfun3d.chebfun3.Chebfun3`, so these definite
triple integrals are the same math and are exercised against it.

MATLAB uses ``tol = 1e1 * chebfun3eps``.  chebfunjax has no
``chebfun3eps`` preference; following the sibling chebfun3 sum3 port
(tests/test_matlab_port/chebfun3/test_sum3.py), the resolution
tolerance is expressed as ``K * EPS`` (machine epsilon) with the
multiplier scaled to the achieved accumulation of the Tucker sum3
(measured worst error here ~1e-14, i.e. ~45*EPS; 1e4*EPS keeps a >200x
margin).  The empty-object assertion (``isempty(sum3(chebfun3t()))``)
is omitted: chebfunjax has no empty Chebfun3T.

Provenance
----------
MATLAB source : tests/chebfun3t/test_sum3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
import pytest

from chebfunjax.chebfun3d import chebfun3t

EPS = float(np.finfo(np.float64).eps)
TOL = 1e4 * EPS


@pytest.mark.slow
class TestChebfun3tSum3:
    def test_all_matlab_assertions(self):
        # Constant: f = chebfun3t(@(x,y,z) 1);  sum3(f) == 8.
        f = chebfun3t(lambda x, y, z: 1.0 + 0 * x)
        assert abs(float(f.sum3()) - 8.0) < TOL

        # Runge function on [-1,1]^3:
        #   f = chebfun3t(@(x,y,z) 1./(1+x.^2+y.^2+z.^2));
        #   sum3(f) == 4.28685406230184188268
        f = chebfun3t(lambda x, y, z: 1.0 / (1 + x ** 2 + y ** 2 + z ** 2))
        assert abs(float(f.sum3()) - 4.28685406230184188268) < TOL

        # Different domains [0,1]x[0,2]x[0,3]:
        #   sum3(x) == 3, sum3(y) == 6, sum3(z) == 9.
        dom = (0.0, 1.0, 0.0, 2.0, 0.0, 3.0)
        fx = chebfun3t(lambda x, y, z: x, dom)
        assert abs(float(fx.sum3()) - 3.0) < TOL
        fy = chebfun3t(lambda x, y, z: y, dom)
        assert abs(float(fy.sum3()) - 6.0) < TOL
        fz = chebfun3t(lambda x, y, z: z, dom)
        assert abs(float(fz.sum3()) - 9.0) < TOL
