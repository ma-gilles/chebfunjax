"""Port of MATLAB Chebfun tests/chebop2/test_plus.m (Fable 5).

Checks ``chebop2/plus``: summing two constant-coefficient operators produces
an operator whose x/y orders and coefficient matrix are the combination of the
two.  ``N1 = diff(u,2,2)`` (u_xx) and ``N2 = diff(u,2,1)`` (u_yy), so
``N1 + N2`` is the Laplacian.

Provenance
----------
MATLAB source : tests/chebop2/test_plus.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.operators.chebop2 import Chebop2, diffx, diffy

_EPS = float(np.finfo(np.float64).eps)


class TestChebop2Plus:
    def test_plus_orders_and_coeffs(self):
        tol = _EPS
        # N1 = diff(u,2,2) = u_xx ; N2 = diff(u,2,1) = u_yy
        N1 = Chebop2(lambda u: diffx(u, 2))
        N2 = Chebop2(lambda u: diffy(u, 2))
        N = N1 + N2

        assert N.xorder == 2
        assert N.yorder == 2
        expected = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.float64)
        assert np.linalg.norm(N.coeffs - expected) < tol
