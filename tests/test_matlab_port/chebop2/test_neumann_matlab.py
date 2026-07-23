"""Port of MATLAB Chebfun tests/chebop2/test_neumann.m (Opus 4.8).

Laplace with mixed Dirichlet/Neumann boundary conditions, solved with the
coefficient-space (ultraspherical) Chebop2 path.  Neumann/Robin conditions are
expressed as two-argument BC lambdas ``lambda x, u: u.diff(1) - g(x)``.

Provenance
----------
MATLAB source : tests/chebop2/test_neumann.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.chebop2 import Chebop2, laplacian

_EPS = float(np.finfo(np.float64).eps)


def _ev(u, x, y):
    return np.asarray(u(jnp.asarray(np.atleast_1d(x), dtype=jnp.float64),
                        jnp.asarray(np.atleast_1d(y), dtype=jnp.float64)))


class TestChebop2Neumann:
    def test_all_matlab_assertions(self):
        tol = 100.0 * _EPS  # MATLAB 100*cheb2Prefs.chebfun2eps.

        # --- First problem on d = [-2 3 -1 1] ---
        d = (-2.0, 3.0, -1.0, 1.0)
        ramp = lambda x: (x - d[0]) / (d[1] - d[0]) + 1.0
        N = Chebop2(laplacian, domain=d)
        N.lbc = lambda y: y
        N.rbc = lambda y: 2.0 * y
        N.ubc = lambda x: ramp(x)
        N.dbc = lambda x, u: u.diff(1) - ramp(x)
        u = N.solve(0.0)

        xs = np.linspace(d[0], d[1], 40)
        ys = np.linspace(d[2], d[3], 40)

        # pass(1): u(:, yb) == ubc(x).
        assert np.max(np.abs(_ev(u, xs, np.full_like(xs, d[3])) - ramp(xs))) < tol
        # pass(2): u(xb, :) == rbc(y).
        assert np.max(np.abs(_ev(u, np.full_like(ys, d[1]), ys) - 2.0 * ys)) < tol
        # pass(3): u_y(:, ya) == ramp(x).
        uy = u.diff(1, 1)
        assert np.max(np.abs(_ev(uy, xs, np.full_like(xs, d[2])) - ramp(xs))) < 5.0 * tol
        # pass(4): u(xa, :) == lbc(y).
        assert np.max(np.abs(_ev(u, np.full_like(ys, d[0]), ys) - ys)) < tol
        # pass(5): laplacian(u) == 0.
        X, Y = np.meshgrid(xs, ys)
        uxx = _ev(u.diff(2, 2), X.ravel(), Y.ravel())  # u_xx
        uyy = _ev(u.diff(1, 2), X.ravel(), Y.ravel())  # u_yy
        assert np.max(np.abs(uxx + uyy)) < 400.0 * tol

        # --- Nick Hale's Neumann example on [0,1]^2 (pass 6) ---
        N = Chebop2(laplacian, domain=(0.0, 1.0, 0.0, 1.0))
        N.ubc = 0.0
        N.rbc = 0.0
        N.lbc = 0.0
        N.dbc = lambda x, u: u.diff(1) - jnp.sin(2.0 * math.pi * x)
        u = N.solve(0.0)
        dudx = u.diff(1, 1)  # d/dy
        xs = np.linspace(0.0, 1.0, 40)
        assert np.max(
            np.abs(_ev(dudx, xs, np.zeros_like(xs)) - np.sin(2.0 * np.pi * xs))
        ) < 10.0 * tol
