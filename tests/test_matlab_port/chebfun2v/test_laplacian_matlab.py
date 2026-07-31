"""Port of MATLAB Chebfun tests/chebfun2v/test_laplacian.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_laplacian.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

TOL = 50 * float(np.finfo(np.float64).eps)
XS = np.linspace(-0.9, 0.9, 9)


def _cmp(approx_a, approx_b):
    fa, fb = Chebfun2(approx=approx_a), Chebfun2(approx=approx_b)
    return max(
        abs(float(np.asarray(fa(x, y))) - float(np.asarray(fb(x, y))))
        for x in XS[::3] for y in XS[::3])


class TestChebfun2vLaplacian:
    def test_two_components(self):
        # pass(1)-(2): laplacian(F) equals componentwise diff2 sums.
        F = Chebfun2v.from_functions(lambda x, y: jnp.cos(x),
                                     lambda x, y: jnp.sin(y))
        lapF = F.laplacian()
        for k in range(2):
            c = F.components[k]
            ref = (Chebfun2(approx=c.diff(1, 2))
                   + Chebfun2(approx=c.diff(2, 2))).approx
            assert _cmp(lapF.components[k], ref) < 100 * TOL

    def test_three_components(self):
        # pass(3)-(5): the 3-component case, incl. x*y + y^2.
        F = Chebfun2v.from_functions(lambda x, y: jnp.cos(x),
                                     lambda x, y: jnp.sin(y),
                                     lambda x, y: x * y + y ** 2)
        lapF = F.laplacian()
        for k in range(3):
            c = F.components[k]
            ref = (Chebfun2(approx=c.diff(1, 2))
                   + Chebfun2(approx=c.diff(2, 2))).approx
            assert _cmp(lapF.components[k], ref) < 100 * TOL
