"""Port of MATLAB Chebfun tests/chebfun2/test_coefficients.m (Fable 5).

FIXED (Fable 5, chebfun2/3 skip sweep): ``chebcoeffs2`` already existed;
the static grid/transform helpers ``Chebfun2.chebpts2`` and
``Chebfun2.coeffs2vals`` that the MATLAB test uses were added.

Provenance
----------
MATLAB source : tests/chebfun2/test_coefficients.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.utils.polynomials import chebeval

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS


class TestChebfun2Coefficients:
    def test_tensor_chebyshev_coefficients_of_T10_T10(self):
        # pass(1, 2): f(x,y) = T_10(x) T_10(y) has a single nonzero
        # bivariate coefficient, equal to 1 at index (10, 10).
        n = 10

        def T(t):
            return chebeval(t, n)

        h = Chebfun2.from_function(lambda x, y: T(x) * T(y))
        X = np.array(h.chebcoeffs2())
        assert abs(X[n, n] - 1.0) < TOL
        X[n, n] -= 1.0
        assert float(np.linalg.norm(X)) < TOL

    def test_coeffs2vals_inverts_chebcoeffs2(self):
        # pass(3): coeffs2vals(chebcoeffs2(f)) reproduces the values of f
        # on the tensor Chebyshev grid of the same size.
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
        lenc = int(f.approx.cols[0].coeffs.shape[0])
        lenr = int(f.approx.rows[0].coeffs.shape[0])
        C = np.asarray(f.chebcoeffs2())
        ny, nx = C.shape
        xx, yy = Chebfun2.chebpts2(nx, ny)
        vals = np.asarray(f(xx, yy))
        X = np.asarray(Chebfun2.coeffs2vals(C))
        assert float(np.linalg.norm(X - vals)) < TOL
        # The coefficient matrix is sized by the column/row lengths.
        assert (ny, nx) == (lenc, lenr)

    def test_vals2coeffs_round_trip(self):
        # coeffs2vals and vals2coeffs are mutual inverses.
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
        C = np.asarray(f.chebcoeffs2())
        back = np.asarray(Chebfun2.vals2coeffs(Chebfun2.coeffs2vals(C)))
        assert float(np.linalg.norm(back - C)) < TOL
