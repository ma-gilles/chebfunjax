"""Port of MATLAB Chebfun tests/chebop2/test_domain.m (Opus 4.8).

Checks that boundary conditions are imposed correctly on non-square domains,
plus a Robin condition (pass 10).  Solved with the coefficient-space
(ultraspherical) Chebop2 path.

Provenance
----------
MATLAB source : tests/chebop2/test_domain.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.chebop2 import Chebop2, laplacian

_EPS = float(np.finfo(np.float64).eps)


def _ev(u, x, y):
    return np.asarray(u(jnp.asarray(np.atleast_1d(x), dtype=jnp.float64),
                        jnp.asarray(np.atleast_1d(y), dtype=jnp.float64)))


class TestChebop2Domain:
    def test_all_matlab_assertions(self):
        tol = 1e4 * _EPS  # MATLAB 1e4 * techPrefs.chebfuneps.
        # Corner-singular harmonic BCs resolve slowly; the solve() adaptive loop
        # may hit n_max.  The edge-value assertions below verify the result.
        warnings.simplefilter("ignore")

        # --- Problem 1 on d = [-2 2 -2 2], rbc = (2-x)(2+x) ---
        d = (-2.0, 2.0, -2.0, 2.0)
        N = Chebop2(lambda u: u.diff(2, 0) + u.diff(0, 2), domain=d)
        N.rbc = lambda x: (2.0 - x) * (2.0 + x)
        N.ubc = 0.0
        N.dbc = 0.0
        N.lbc = 0.0
        u = N.solve(0.0)
        ys = np.linspace(d[2], d[3], 40)
        # pass(1)-(4): each edge matches its BC.
        assert np.max(np.abs(_ev(u, ys, np.full_like(ys, d[2])))) < 10.0 * tol
        assert np.max(np.abs(_ev(u, ys, np.full_like(ys, d[3])))) < tol
        assert np.max(np.abs(_ev(u, np.full_like(ys, d[0]), ys))) < tol
        assert np.max(
            np.abs(_ev(u, np.full_like(ys, d[1]), ys) - (2.0 - ys) * (2.0 + ys))
        ) < tol

        # --- Problem 2 on d = [-pi 2pi -2 5], dbc = (x+pi)(x-2pi) ---
        d = (-np.pi, 2.0 * np.pi, -2.0, 5.0)
        N = Chebop2(lambda u: u.diff(2, 0) + u.diff(0, 2), domain=d)
        N.rbc = 0.0
        N.ubc = 0.0
        N.dbc = lambda x: (x + np.pi) * (x - 2.0 * np.pi)
        N.lbc = 0.0
        u = N.solve(0.0)
        xs = np.linspace(d[0], d[1], 40)
        ys = np.linspace(d[2], d[3], 40)
        # pass(5)-(8).
        assert np.max(
            np.abs(_ev(u, xs, np.full_like(xs, d[2])) - (xs + np.pi) * (xs - 2.0 * np.pi))
        ) < 3.0 * tol
        assert np.max(np.abs(_ev(u, xs, np.full_like(xs, d[3])))) < 2.0 * tol
        assert np.max(np.abs(_ev(u, np.full_like(ys, d[0]), ys))) < 10.0 * tol
        assert np.max(np.abs(_ev(u, np.full_like(ys, d[1]), ys))) < 10.0 * tol

        # pass(9): harmonic solution on the same rectangle.
        bdy = lambda x, y: np.real(np.exp(x + 1j * y))
        N = Chebop2(lambda u: u.diff(2, 0) + u.diff(0, 2), domain=d)
        N.lbc = lambda y: bdy(d[0], y)
        N.rbc = lambda y: bdy(d[1], y)
        N.dbc = lambda x: bdy(x, d[2])
        N.ubc = lambda x: bdy(x, d[3])
        u = N.solve(0.0)
        X, Y = np.meshgrid(xs, ys)
        assert np.max(np.abs(_ev(u, X.ravel(), Y.ravel()) - bdy(X.ravel(), Y.ravel()))) \
            < 100.0 * tol

        # pass(10): Robin condition u + 2.1*u' - sin(2 pi x) on [0,1]x[0,pi/6].
        d = (0.0, 1.0, 0.0, np.pi / 6.0)
        N = Chebop2(laplacian, domain=d)
        N.ubc = 0.0
        N.rbc = 0.0
        N.lbc = 0.0
        N.dbc = lambda x, u: u + 2.1 * u.diff(1) - jnp.sin(2.0 * np.pi * x)
        u = N.solve(0.0)
        uy = u.diff(1, 1)  # d/dy
        xs = np.linspace(0.0, 1.0, 40)
        comb = _ev(u, xs, np.zeros_like(xs)) + 2.1 * _ev(uy, xs, np.zeros_like(xs))
        assert np.max(np.abs(comb - np.sin(2.0 * np.pi * xs))) < 10.0 * tol
