"""Port of MATLAB Chebfun tests/chebop2/test_adtest.m (Fable 5).

Checks that the operator-coefficient extraction (``N.coeffs``) is correct for
constant-coefficient PDOs.

Ported subset: MATLAB assertions pass(1)-pass(3) (constant-coefficient
coefficient matrices).  pass(4)-pass(9) exercise *variable* coefficients
(``x.*u``, ``x.*diff(u)``), which the scalar chebfunjax Chebop2 does not
represent; those are covered by the specific skip in the variable-coefficient
ports (e.g. test_generalVariableCoefficients_matlab).

Provenance
----------
MATLAB source : tests/chebop2/test_adtest.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.operators.chebop2 import Chebop2, diffx, diffy

_EPS = float(np.finfo(np.float64).eps)


class TestChebop2Adtest:
    def test_laplacian_coeffs(self):
        tol = _EPS
        N = Chebop2(lambda u: diffx(u, 2) + diffy(u, 2))
        expected = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.float64)
        assert np.linalg.norm(N.coeffs - expected) < tol

    def test_helmholtz_coeffs(self):
        tol = _EPS
        N = Chebop2(lambda u: diffx(u, 2) + diffy(u, 2) + np.pi * u)
        expected = np.array(
            [[np.pi, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.float64
        )
        assert np.linalg.norm(N.coeffs - expected) < tol

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
        assert np.linalg.norm(N.coeffs - expected) < tol
