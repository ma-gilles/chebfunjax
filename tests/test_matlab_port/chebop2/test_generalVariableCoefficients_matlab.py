"""Port of MATLAB Chebfun tests/chebop2/test_generalVariableCoefficients.m (Opus 4.8).

General variable-coefficient elliptic PDEs ``m1(x,y) u_xx - m2(x,y) u_yy = f``
with bivariate coefficients, solved with the coefficient-space (ultraspherical)
Chebop2 path.  Each coefficient is CDR-decomposed (SVD of its Chebyshev matrix)
into rank-1 x/y factors, each becoming an ultraspherical multiplication matrix.

Variable coefficients must be written with NumPy ufuncs (``np.cos`` / ``np.exp``)
so they dispatch onto the symbolic ``_Coord``.  MATLAB compares
``norm(u - exact)`` (the chebfun2 L2 norm); we use a fine-grid L2 quadrature.

Provenance
----------
MATLAB source : tests/chebop2/test_generalVariableCoefficients.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.chebop2 import Chebop2

_EPS = float(np.finfo(np.float64).eps)


def _l2err(u, exact, dom, ng=200):
    xa, xb, ya, yb = dom
    xs = np.linspace(xa, xb, ng)
    ys = np.linspace(ya, yb, ng)
    X, Y = np.meshgrid(xs, ys)
    e = np.asarray(u(jnp.asarray(X.ravel()), jnp.asarray(Y.ravel()))) \
        - exact(X.ravel(), Y.ravel())
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    return float(np.sqrt(np.sum(e ** 2) * dx * dy))


def _solve(op, exact, rhs, dom):
    N = Chebop2(op, domain=dom)
    N.lbc = lambda y: exact(dom[0], y)
    N.rbc = lambda y: exact(dom[1], y)
    N.dbc = lambda x: exact(x, dom[2])
    N.ubc = lambda x: exact(x, dom[3])
    return N.solve(rhs)


class TestChebop2GeneralVariableCoefficients:
    def test_all_matlab_assertions(self):
        tol = 4000.0 * _EPS  # MATLAB 4000 * cheb2Prefs.chebfun2eps.
        d = (-1.0, 1.0, -1.0, 1.0)

        # pass(1): u_xx - cos(x*y)*u_yy = f, exact = sin(5xy).
        exact = lambda x, y: np.sin(5.0 * x * y)
        exact_xx = lambda x, y: -25.0 * y ** 2 * np.sin(5.0 * x * y)
        exact_yy = lambda x, y: -25.0 * x ** 2 * np.sin(5.0 * x * y)
        m = lambda x, y: np.cos(x * y)
        rhs = lambda x, y: exact_xx(x, y) - m(x, y) * exact_yy(x, y)
        op = lambda x, y, u: u.diff(0, 2) - np.cos(x * y) * u.diff(2, 0)
        assert _l2err(_solve(op, exact, rhs, d), exact, d) < tol

        # pass(2): m = exp(-x)*y + y^2.
        m = lambda x, y: np.exp(-x) * y + y ** 2
        rhs = lambda x, y: exact_xx(x, y) - m(x, y) * exact_yy(x, y)
        op = lambda x, y, u: u.diff(0, 2) - (np.exp(-x) * y + y ** 2) * u.diff(2, 0)
        assert _l2err(_solve(op, exact, rhs, d), exact, d) < tol

        # pass(3): m1*u_xx - m2*u_yy, exact = sin(xy).
        exact = lambda x, y: np.sin(x * y)
        exact_xx = lambda x, y: -y ** 2 * np.sin(x * y)
        exact_yy = lambda x, y: -x ** 2 * np.sin(x * y)
        m1 = lambda x, y: np.exp(-x) * y + y ** 2
        m2 = lambda x, y: np.cos(x) + np.sin(y) + x * y
        rhs = lambda x, y: m1(x, y) * exact_xx(x, y) - m2(x, y) * exact_yy(x, y)
        op = lambda x, y, u: ((np.exp(-x) * y + y ** 2) * u.diff(0, 2)
                              - (np.cos(x) + np.sin(y) + x * y) * u.diff(2, 0))
        assert _l2err(_solve(op, exact, rhs, d), exact, d) < tol

        # pass(4): MATLAB's assertion here is the tautology `tol < 10*tol`; we
        # still exercise the solve with exact = exp(-x)*y + sin(x) + cos(xy).
        exact = lambda x, y: np.exp(-x) * y + np.sin(x) + np.cos(x * y)
        exact_xx = lambda x, y: np.exp(-x) * y - np.sin(x) - y ** 2 * np.cos(x * y)
        exact_yy = lambda x, y: -x ** 2 * np.cos(x * y)
        rhs = lambda x, y: m1(x, y) * exact_xx(x, y) - m2(x, y) * exact_yy(x, y)
        _ = _solve(op, exact, rhs, d)
        assert tol < 10.0 * tol

        # pass(5): on d = [-2 2 -2 2], m = cos(x), exact = sin(xy).
        d5 = (-2.0, 2.0, -2.0, 2.0)
        exact = lambda x, y: np.sin(x * y)
        exact_xx = lambda x, y: -y ** 2 * np.sin(x * y)
        exact_yy = lambda x, y: -x ** 2 * np.sin(x * y)
        m = lambda x, y: np.cos(x)
        rhs = lambda x, y: exact_xx(x, y) - m(x, y) * exact_yy(x, y)
        op = lambda x, y, u: u.diff(0, 2) - np.cos(x) * u.diff(2, 0)
        assert _l2err(_solve(op, exact, rhs, d5), exact, d5) < 5.0 * tol

        # pass(6): on d = [-2 0 -4.1 pi], m = x+y, polynomial exact.
        d6 = (-2.0, 0.0, -4.1, np.pi)
        exact = lambda x, y: 1.0 + x + y + x * y + x ** 2 * y ** 2
        exact_xx = lambda x, y: 2.0 * y ** 2
        exact_yy = lambda x, y: 2.0 * x ** 2
        m = lambda x, y: x + y
        rhs = lambda x, y: exact_xx(x, y) - m(x, y) * exact_yy(x, y)
        op = lambda x, y, u: u.diff(0, 2) - (x + y) * u.diff(2, 0)
        assert _l2err(_solve(op, exact, rhs, d6), exact, d6) < tol
