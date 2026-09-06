"""Port of MATLAB Chebfun tests/chebop2/test_rhs2.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_rhs2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2, chebfun2
from chebfunjax.operators.chebop2 import Chebop2, laplacian

jax.config.update("jax_enable_x64", True)


def _c2(u):
    return u if isinstance(u, Chebfun2) else Chebfun2(approx=u)


def _zero_bcs(N):
    N.lbc = 0
    N.rbc = 0
    N.ubc = 0
    N.dbc = 0
    return N


class TestChebop2Rhs2:
    def test_all_matlab_assertions(self):
        tol = 1e-14
        N = _zero_bcs(Chebop2(lambda u: laplacian(u)))
        exact = _c2(N.solve(1.0))
        N = _zero_bcs(Chebop2(lambda u: laplacian(u) - 1))
        u = _c2(N.solve(0.0))
        assert float((u - exact).norm()) < tol                               # pass(1)

        N = _zero_bcs(Chebop2(lambda x, y, u: laplacian(u)))
        x = chebfun2(lambda x, y: x)
        exact = _c2(N.solve(x.sin()))
        N = _zero_bcs(Chebop2(lambda x, y, u: laplacian(u) - np.sin(x)))
        u = _c2(N.solve(0.0))
        assert float((u - exact).norm()) < tol                               # pass(2)

        N = _zero_bcs(Chebop2(lambda x, y, u: laplacian(u) + np.sin(x * (y + .1)) - 1))
        exact = _c2(N.solve(0.0))
        N = _zero_bcs(Chebop2(lambda x, y, u: laplacian(u) + np.sin(x * (y + .1))))
        u = _c2(N.solve(1.0))
        assert float((u - exact).norm()) < tol                               # pass(3)
