"""Port of MATLAB Chebfun tests/chebfun2/test_chebcoeffs2.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_chebcoeffs2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.utils.transforms import vals2coeffs

TOL = 1e3 * float(np.finfo(np.float64).eps)


def _cheb_T(t, k):
    return jnp.cos(k * jnp.arccos(t))


class TestChebfun2Chebcoeffs2:
    def test_all_matlab_assertions(self):
        # Rank-2 function f = T_n(y)T_n(x) + T_m(y)T_n(x), n=10, m=8.
        n, m = 10, 8
        f = chebfun2(
            lambda x, y: _cheb_T(y, n) * _cheb_T(x, n)
            + _cheb_T(y, m) * _cheb_T(x, n))

        # pass(1): full coefficient matrix matches the exact modes.
        X = np.asarray(f.chebcoeffs2())
        exact = np.zeros((n + 1, n + 1))
        exact[n, n] = 1.0
        exact[m, n] = 1.0
        assert X.shape == (n + 1, n + 1)
        assert np.linalg.norm(X - exact) < TOL

        # pass(2): chebcoeffs2 agrees with vals2coeffs(chebpolyval2(f)).
        # chebpolyval2 returns V[j, i] = f(x_i, y_j) on an (ny, nx) grid;
        # applying the 1-D value->coeff transform along y (axis 0) then x
        # (axis 1) reproduces the coefficient matrix (this is exactly what
        # MATLAB chebfun2.vals2coeffs does).
        V = np.asarray(f.chebpolyval2())
        Z = np.asarray(vals2coeffs(vals2coeffs(jnp.asarray(V)).T).T)
        assert np.linalg.norm(Z - exact) < TOL

        # pass(3): coeffs2(f, m, n) returns an m-by-n matrix.
        g = chebfun2(lambda x, y: x + y)
        assert np.asarray(g.coeffs2(2, 1)).shape == (2, 1)
