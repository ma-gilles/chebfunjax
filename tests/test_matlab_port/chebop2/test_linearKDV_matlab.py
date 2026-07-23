"""Port of MATLAB Chebfun tests/chebop2/test_linearKDV.m (Opus 4.8).

Linear KDV ``u_t + u_xxx = 0`` (third order in x), solved with the
coefficient-space (ultraspherical) Chebop2 path.  The right edge carries two
conditions (a value and a derivative); the left edge is Dirichlet or Neumann.

Provenance
----------
MATLAB source : tests/chebop2/test_linearKDV.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.chebop2 import Chebop2, diffx, diffy

_EPS = float(np.finfo(np.float64).eps)


def _maxerr(u, exact, d, npts=50):
    xs = np.linspace(d[0], d[1], npts)
    ys = np.linspace(d[2], d[3], npts)
    X, Y = np.meshgrid(xs, ys)
    uv = np.asarray(u(jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())))
    return float(np.max(np.abs(uv - exact(X.ravel(), Y.ravel()))))


class TestChebop2LinearKDV:
    def test_all_matlab_assertions(self):
        tol = 100.0 * _EPS  # MATLAB 100*cheb2Prefs.chebfun2eps.
        d = (-1.0, 1.0, 0.0, 1.0)
        exact = lambda x, t: np.exp(-t) * np.exp(x)

        # pass(1): Dirichlet lbc.
        N = Chebop2(lambda u: diffy(u) + diffx(u, 3), domain=d)
        N.dbc = lambda x: np.exp(x)
        N.rbc = lambda t, u: [u - np.exp(-t) * np.exp(1.0),
                              u.diff(1) - np.exp(-t) * np.exp(1.0)]
        N.lbc = lambda t: np.exp(-t) * np.exp(-1.0)
        assert _maxerr(N.solve(0.0), exact, d) < tol

        # pass(2): Neumann lbc.
        N = Chebop2(lambda u: diffy(u) + diffx(u, 3), domain=d)
        N.dbc = lambda x: np.exp(x)
        N.rbc = lambda t, u: [u - np.exp(-t) * np.exp(1.0),
                              u.diff(1) - np.exp(-t) * np.exp(1.0)]
        N.lbc = lambda t, u: u.diff(1) - np.exp(-t) * np.exp(-1.0)
        assert _maxerr(N.solve(0.0), exact, d) < 2.0 * tol

        # pass(3): second-derivative condition on the right edge.
        N = Chebop2(lambda u: diffy(u) + diffx(u, 3), domain=d)
        N.dbc = lambda x: np.exp(x)
        N.rbc = lambda t, u: [u - np.exp(-t) * np.exp(1.0),
                              u.diff(2) - np.exp(-t) * np.exp(1.0)]
        N.lbc = lambda t, u: u.diff(1) - np.exp(-t) * np.exp(-1.0)
        assert _maxerr(N.solve(0.0), exact, d) < 300.0 * tol

        # pass(4): Dirichlet lbc with the second-derivative right condition.
        N = Chebop2(lambda u: diffy(u) + diffx(u, 3), domain=d)
        N.dbc = lambda x: np.exp(x)
        N.rbc = lambda t, u: [u - np.exp(-t) * np.exp(1.0),
                              u.diff(2) - np.exp(-t) * np.exp(1.0)]
        N.lbc = lambda t, u: u - np.exp(-t) * np.exp(-1.0)
        assert _maxerr(N.solve(0.0), exact, d) < 100.0 * tol
