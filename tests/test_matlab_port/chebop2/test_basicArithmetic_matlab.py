"""Port of MATLAB Chebfun tests/chebop2/test_basicArithmetic.m (Opus 4.8).

Checks ``Chebop2`` operator arithmetic for variable-coefficient operators: the
sum ``N1 + N2`` of two ``@(x,y,u)`` operators must equal the directly
constructed operator, i.e. the coefficient functions of ``(N1 + N2) - EXACT``
must all vanish.  MATLAB measures ``sum_{j,k} norm(C{j,k})`` over the cell array
of coefficient functions; here we evaluate each coefficient of the difference
operator on a grid and sum their norms.

Provenance
----------
MATLAB source : tests/chebop2/test_basicArithmetic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.operators.chebop2 import Chebop2, _Coord

_EPS = float(np.finfo(np.float64).eps)


def _coeff_residual_norm(N):
    """Sum of the (grid) L2 norms of the operator's coefficient functions."""
    xs = np.linspace(-1.0, 1.0, 25)
    ys = np.linspace(-1.0, 1.0, 25)
    X, Y = np.meshgrid(xs, ys)
    total = 0.0
    for (_j, _k), c in N._all_terms().items():
        vals = c.fn(X, Y) if isinstance(c, _Coord) else np.full(X.shape, c)
        total += float(np.sqrt(np.mean(np.abs(vals) ** 2)))
    return total


class TestChebop2BasicArithmetic:
    def test_all_matlab_assertions(self):
        tol = 10.0 * _EPS  # MATLAB 10 * cheb2Prefs.chebfun2eps.

        # pass(1): both operands use @(x,y,u); N2 has a variable coefficient.
        N1 = Chebop2(lambda x, y, u: u.diff(0, 2))
        N2 = Chebop2(lambda x, y, u: x * u.diff(2, 0))
        N = N1 + N2
        exact = Chebop2(lambda x, y, u: u.diff(0, 2) + x * u.diff(2, 0))
        assert abs(_coeff_residual_norm(N - exact)) < tol

        # pass(2): N1 is a plain @(u) constant operator.
        N1 = Chebop2(lambda u: u.diff(0, 2))
        N2 = Chebop2(lambda x, y, u: x * u.diff(2, 0))
        N = N1 + N2
        exact = Chebop2(lambda x, y, u: u.diff(0, 2) + x * u.diff(2, 0))
        assert abs(_coeff_residual_norm(N - exact)) < tol

        # pass(3): variable coefficient in the y-direction.
        N1 = Chebop2(lambda x, y, u: y * u.diff(0, 2))
        N2 = Chebop2(lambda u: u.diff(2, 0))
        N = N1 + N2
        exact = Chebop2(lambda x, y, u: y * u.diff(0, 2) + u.diff(2, 0))
        assert abs(_coeff_residual_norm(N - exact)) < tol
