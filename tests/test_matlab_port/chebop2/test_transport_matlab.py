"""Port of MATLAB Chebfun tests/chebop2/test_transport.m (Opus 4.8).

Transport ``u_t + c*u_x = 0`` on rectangular domains, solved with the
coefficient-space (ultraspherical) Chebop2 path (rank-1 operator).  MATLAB
compares in the L2 norm; we use the grid max-norm at the MATLAB tolerances.

Provenance
----------
MATLAB source : tests/chebop2/test_transport.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.chebop2 import Chebop2, diffx, diffy

_EPS = float(np.finfo(np.float64).eps)


def _grid_maxerr(u, exact, dom, npts=60):
    xa, xb, ya, yb = dom
    xs = np.linspace(xa, xb, npts)
    ys = np.linspace(ya, yb, npts)
    X, Y = np.meshgrid(xs, ys)
    uv = np.asarray(u(jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())))
    ev = np.asarray(exact(X.ravel(), Y.ravel()))
    return float(np.max(np.abs(uv - ev)))


class TestChebop2Transport:
    def test_all_matlab_assertions(self):
        tol = 1000.0 * _EPS  # MATLAB 1000*cheb2Prefs.chebfun2eps.

        # pass(1): u_t + u_x = 0 on [-1,1]x[0,1].
        d = (-1.0, 1.0, 0.0, 1.0)
        exact = lambda x, t: np.exp(-x) * np.exp(t)
        N = Chebop2(lambda u: diffy(u, 1) + diffx(u, 1), domain=d)
        N.dbc = lambda x: np.exp(-x)
        N.lbc = lambda t: np.exp(1.0) * np.exp(t)
        u = N.solve(0.0)
        assert _grid_maxerr(u, exact, d) < 2.0 * tol

        # pass(2): square domain.
        d = (0.0, 1.0, 0.0, 1.0)
        exact = lambda x, t: np.exp(-x) * np.exp(t)
        N = Chebop2(lambda u: diffy(u, 1) + diffx(u, 1), domain=d)
        N.dbc = lambda x: np.exp(-x)
        N.lbc = lambda t: np.exp(t)
        u = N.solve(0.0)
        assert _grid_maxerr(u, exact, d) < tol

        # pass(3): sum of two transported modes.
        d = (-1.0, 1.0, 0.0, 1.0)
        exact = lambda x, t: np.exp(-x) * np.exp(t) + np.exp(-0.5 * x) * np.exp(0.5 * t)
        N = Chebop2(lambda u: diffy(u, 1) + diffx(u, 1), domain=d)
        N.dbc = lambda x: np.exp(-x) + np.exp(-0.5 * x)
        N.lbc = lambda t: np.exp(1.0) * np.exp(t) + np.exp(0.5) * np.exp(0.5 * t)
        u = N.solve(0.0)
        assert _grid_maxerr(u, exact, d) < 5.0 * tol

        # pass(4): transport parameter c = 5 on [-pi,pi]x[0,1].
        d = (-np.pi, np.pi, 0.0, 1.0)
        exact = lambda x, t: np.exp(x) * np.exp(-5.0 * t)
        N = Chebop2(lambda u: diffy(u, 1) + 5.0 * diffx(u, 1), domain=d)
        N.dbc = lambda x: np.exp(x)
        N.lbc = lambda t: np.exp(-np.pi) * np.exp(-5.0 * t)
        u = N.solve(0.0)
        assert _grid_maxerr(u, exact, d) < 10.0 * tol

        # pass(5): c = 0.1 with a large time interval [0,100].
        d = (-np.pi, np.pi, 0.0, 100.0)
        exact = lambda x, t: np.exp(x) * np.exp(-t / 10.0)
        N = Chebop2(lambda u: diffy(u, 1) + 0.1 * diffx(u, 1), domain=d)
        N.dbc = lambda x: np.exp(x)
        N.lbc = lambda t: np.exp(-np.pi) * np.exp(-t / 10.0)
        u = N.solve(0.0)
        assert _grid_maxerr(u, exact, d) < 1e4 * tol
