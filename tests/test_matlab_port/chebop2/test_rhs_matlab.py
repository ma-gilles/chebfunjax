"""Port of MATLAB Chebfun tests/chebop2/test_rhs.m (Opus 4.8).

Poisson ``lap(u) = f`` with a nonzero forcing term, solved with the
coefficient-space (ultraspherical) Chebop2 path.  MATLAB compares in the L2
norm at 10*tol; we use the grid max-norm at that tolerance.

Provenance
----------
MATLAB source : tests/chebop2/test_rhs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.chebop2 import Chebop2, laplacian

_EPS = float(np.finfo(np.float64).eps)


def _grid_maxerr(u, exact, dom, npts=60):
    xa, xb, ya, yb = dom
    xs = np.linspace(xa, xb, npts)
    ys = np.linspace(ya, yb, npts)
    X, Y = np.meshgrid(xs, ys)
    uv = np.asarray(u(jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())))
    ev = np.asarray(exact(X.ravel(), Y.ravel()))
    return float(np.max(np.abs(uv - ev)))


class TestChebop2Rhs:
    def test_all_matlab_assertions(self):
        tol = 100.0 * _EPS  # MATLAB 100*cheb2Prefs.chebfun2eps.

        # pass(1): lap(u) = x*y on [0,pi]^2.
        d = (0.0, np.pi, 0.0, np.pi)
        N = Chebop2(laplacian, domain=d)
        N.lbc = 0.0
        N.rbc = lambda y: np.pi * y ** 3 / 6.0
        N.dbc = 0.0
        N.ubc = lambda x: x * np.pi ** 3 / 6.0 + np.sin(x) * np.sinh(np.pi)
        u = N.solve(lambda x, y: x * y)
        exact = lambda x, y: x * y ** 3 / 6.0 + np.sin(x) * np.sinh(y)
        assert _grid_maxerr(u, exact, d) < 10.0 * tol

        # pass(2): lap(u) = x^2*y on [0,pi]^2.
        N = Chebop2(laplacian, domain=d)
        N.lbc = 0.0
        N.rbc = lambda y: np.pi ** 4 * y / 12.0 + np.sinh(np.pi) * (np.cos(y) + np.sin(y))
        N.dbc = lambda x: np.sinh(x)
        N.ubc = lambda x: np.pi * x ** 4 / 12.0 - np.sinh(x)
        u = N.solve(lambda x, y: x ** 2 * y)
        exact = lambda x, y: x ** 4 * y / 12.0 + np.sinh(x) * (np.cos(y) + np.sin(y))
        assert _grid_maxerr(u, exact, d) < 10.0 * tol

        # pass(3): rectangular domain, analytic-plus-forcing.
        d = (0.0, np.pi, 0.0, 1.0)
        exact = lambda x, y: np.real(np.exp(x + 1j * y)) + x ** 3 * y ** 3
        N = Chebop2(laplacian, domain=d)
        N.lbc = lambda y: exact(d[0], y)
        N.rbc = lambda y: exact(d[1], y)
        N.dbc = lambda x: exact(x, d[2])
        N.ubc = lambda x: exact(x, d[3])
        u = N.solve(lambda x, y: 6.0 * x * y ** 3 + 6.0 * y * x ** 3)
        assert _grid_maxerr(u, exact, d) < 10.0 * tol
