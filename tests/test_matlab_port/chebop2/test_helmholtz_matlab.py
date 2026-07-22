"""Port of MATLAB Chebfun tests/chebop2/test_helmholtz.m (Opus 4.8).

Helmholtz ``lap(u) + lam*u`` with a separable exact solution, solved with the
coefficient-space (ultraspherical) Chebop2 path.  MATLAB compares in the L2
norm (pass1-2) and the grid inf-norm (pass3-4); we use the grid max-norm at the
MATLAB tolerances.

Provenance
----------
MATLAB source : tests/chebop2/test_helmholtz.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.chebop2 import Chebop2, laplacian

_EPS = float(np.finfo(np.float64).eps)


def _grid_maxerr(u, exact, dom, npts=100):
    xa, xb, ya, yb = dom
    xs = np.linspace(xa, xb, npts)
    ys = np.linspace(ya, yb, npts)
    X, Y = np.meshgrid(xs, ys)
    uv = np.asarray(u(jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())))
    ev = np.asarray(exact(X.ravel(), Y.ravel()))
    return float(np.max(np.abs(uv - ev)))


def _solve(mu1, mu2, dom, n=None):
    lam = mu1 ** 2 + mu2 ** 2
    xbc = lambda x: np.cos(mu1 * x) + np.sin(mu1 * x)
    ybc = lambda y: np.cos(mu2 * y) + np.sin(mu2 * y)
    xa, xb, ya, yb = dom
    N = Chebop2(lambda u: laplacian(u) + lam * u, domain=dom)
    N.lbc = lambda y: xbc(xa) * ybc(y)
    N.rbc = lambda y: xbc(xb) * ybc(y)
    N.dbc = lambda x: xbc(x) * ybc(ya)
    N.ubc = lambda x: xbc(x) * ybc(yb)
    u = N.solve(0.0) if n is None else N.solve(0.0, n=n)
    exact = lambda x, y: xbc(x) * ybc(y)
    return u, exact


class TestChebop2Helmholtz:
    def test_all_matlab_assertions(self):
        tol = _EPS  # MATLAB cheb2Prefs.chebfun2eps.

        # pass(1): square domain, MATLAB tol 100*tol.
        u, exact = _solve(3.0 / np.pi, np.pi / 6.0, (-1.0, 1.0, -1.0, 1.0))
        assert _grid_maxerr(u, exact, (-1.0, 1.0, -1.0, 1.0)) < 100.0 * tol

        # pass(2): rectangular domain, MATLAB tol 3000*tol.
        d = (-2.0, 3.0, -1.0 / 10.0, 3.0)
        u, exact = _solve(1.0, 2.0, d)
        assert _grid_maxerr(u, exact, d) < 3000.0 * tol

        # pass(3): high frequency, MATLAB grid inf-norm tol 1e9*tol.
        d = (-2.0, 3.0, -1.0 / 10.0, 3.0)
        u, exact = _solve(12.0, 10.0, d)
        assert _grid_maxerr(u, exact, d) < 1e9 * tol

        # pass(4): higher frequency, MATLAB grid inf-norm tol 1e9*tol.
        d = (-2.0, 3.0, -1.0 / 10.0, 3.0)
        u, exact = _solve(12.0, 50.0, d)
        assert _grid_maxerr(u, exact, d) < 1e9 * tol
