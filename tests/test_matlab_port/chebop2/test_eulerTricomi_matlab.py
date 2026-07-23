"""Port of MATLAB Chebfun tests/chebop2/test_eulerTricomi.m (Opus 4.8).

The Euler--Tricomi equation ``u_xx - x*u_yy = 0`` (and variants) has polynomial
particular solutions.  Solved with the coefficient-space (ultraspherical)
Chebop2 path using the variable-coefficient discretization: the ``x`` (or ``y``)
coefficient becomes an ultraspherical multiplication matrix.

MATLAB compares ``norm(u - exact)`` (the chebfun2 L2 norm); we approximate that
with a fine-grid L2 quadrature at the MATLAB tolerance.

Provenance
----------
MATLAB source : tests/chebop2/test_eulerTricomi.m
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


def _solve(op, exact, dom):
    N = Chebop2(op, domain=dom)
    N.lbc = lambda y: exact(dom[0], y)
    N.rbc = lambda y: exact(dom[1], y)
    N.ubc = lambda x: exact(x, dom[3])
    N.dbc = lambda x: exact(x, dom[2])
    return N.solve(0.0)


class TestChebop2EulerTricomi:
    def test_all_matlab_assertions(self):
        tol = 100.0 * _EPS  # MATLAB 100 * techPrefs.chebfuneps.
        d = (-1.0, 1.0, -1.0, 1.0)

        # Operator A: u_xx - x*u_yy (pass 1-5, 8, 9).
        opA = lambda x, y, u: u.diff(0, 2) - x * u.diff(2, 0)
        for exact in (
            lambda x, y: 1.0 + 0.0 * x,          # pass 1
            lambda x, y: y,                       # pass 2
            lambda x, y: x,                       # pass 3
            lambda x, y: x * y,                   # pass 4
            lambda x, y: 3.0 * y ** 2 + x ** 3,   # pass 5
            lambda x, y: y ** 3 + x ** 3 * y,     # pass 8
            lambda x, y: 6.0 * x * y ** 2 + x ** 4,  # pass 9
        ):
            assert _l2err(_solve(opA, exact, d), exact, d) < tol

        # pass(6): y*u_xx - u_yy with exact 3x^2 + y^3.
        opB = lambda x, y, u: y * u.diff(0, 2) - u.diff(2, 0)
        exact = lambda x, y: 3.0 * x ** 2 + y ** 3
        assert _l2err(_solve(opB, exact, d), exact, d) < tol

        # pass(7): -y*u_xx + u_yy with the same exact solution.
        opC = lambda x, y, u: -y * u.diff(0, 2) + u.diff(2, 0)
        assert _l2err(_solve(opC, exact, d), exact, d) < tol
