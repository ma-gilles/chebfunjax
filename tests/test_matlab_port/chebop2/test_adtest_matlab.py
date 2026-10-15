"""Port of MATLAB Chebfun tests/chebop2/test_adtest.m (Fable 5).

Checks that the operator-coefficient extraction (``N.coeffs``) is correct for
constant-coefficient PDOs.

All nine native coefficient assertions are represented. Variable coefficient
cell arrays use nested Python lists, with the native x-order/y-order layout.

Provenance
----------
MATLAB source : tests/chebop2/test_adtest.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.chebfun2d import chebfun2
from chebfunjax.operators.chebop2 import Chebop2, diffx, diffy

_EPS = float(np.finfo(np.float64).eps)


class TestChebop2Adtest:
    def test_laplacian_coeffs(self):
        tol = _EPS
        N = Chebop2(lambda u: diffx(u, 2) + diffy(u, 2))
        expected = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.float64)
        assert np.linalg.norm(N.coeffs - expected, ord=2) < tol

    def test_helmholtz_coeffs(self):
        tol = _EPS
        N = Chebop2(lambda u: diffx(u, 2) + diffy(u, 2) + np.pi * u)
        expected = np.array(
            [[np.pi, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.float64
        )
        assert np.linalg.norm(N.coeffs - expected, ord=2) < tol

    def test_higher_order_coeffs(self):
        tol = _EPS
        N = Chebop2(
            lambda u: diffx(u, 3)
            + diffx(diffy(u, 1), 2)
            + diffy(u, 2)
            + np.pi * u
        )
        expected = np.array(
            [[np.pi, 0, 1], [0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float64
        )
        assert np.linalg.norm(N.coeffs - expected, ord=2) < tol


    def test_variable_zero_derivative_coefficients(self):
        # Native predicates 4--6, including their original function norms.
        x = chebfun2(lambda x, y: x)
        N = Chebop2(lambda x, y, u: diffy(u, 1) + diffx(u, 2) + x*u)
        C = N.coeffs
        assert (C[0][0] - x).norm() < 10*_EPS
        assert (C[0][1] - 0*x - 1).norm() < 20*_EPS
        assert (C[2][0] - 0*x - 1).norm() < 20*_EPS

    def test_variable_derivative_coefficients(self):
        # Native predicates 7--9, without sampled-grid norm substitution.
        x = chebfun2(lambda x, y: x)
        N = Chebop2(lambda x, y, u: diffy(u, 1) + x*diffx(u, 2))
        C = N.coeffs
        assert (C[2][0] - x).norm() < 10*_EPS
        assert (C[0][1] - 0*x - 1).norm() < 20*_EPS
        assert (C[0][0] - 0*x).norm() < 20*_EPS
