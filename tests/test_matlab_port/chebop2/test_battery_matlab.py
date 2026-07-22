"""Port of MATLAB Chebfun tests/chebop2/test_battery.m (Opus 4.8).

A battery of Laplace problems on the square, solved with the coefficient-space
(ultraspherical) Chebop2 path, which reaches ~eps accuracy.  MATLAB compares in
the chebfun2 L2 norm; here we use the grid max-norm as a (conservative) proxy at
the MATLAB tolerances.

Provenance
----------
MATLAB source : tests/chebop2/test_battery.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.chebop2 import (
    Chebop2,
    divergence,
    gradient,
    lap,
    laplacian,
)

_EPS = float(np.finfo(np.float64).eps)


def _grid_maxerr(u, exact, dom, npts=60):
    xa, xb, ya, yb = dom
    xs = np.linspace(xa, xb, npts)
    ys = np.linspace(ya, yb, npts)
    X, Y = np.meshgrid(xs, ys)
    uv = np.asarray(u(jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())))
    ev = np.asarray(exact(X.ravel(), Y.ravel()))
    return float(np.max(np.abs(uv - ev)))


class TestChebop2Battery:
    def test_all_matlab_assertions(self):
        # tol = 100 * eps (MATLAB techPrefs.chebfuneps).
        tol = 100.0 * _EPS
        dom = (-1.0, 1.0, -1.0, 1.0)

        def solve_harmonic(bdy):
            N = Chebop2(lambda u: u.diff(2, 0) + u.diff(0, 2))
            N.lbc = lambda y: bdy(-1.0, y)
            N.rbc = lambda y: bdy(1.0, y)
            N.dbc = lambda x: bdy(x, -1.0)
            N.ubc = lambda x: bdy(x, 1.0)
            return N.solve(0.0)

        # pass(1): u = Re(exp(x+iy)).
        bdy = lambda x, y: np.real(np.exp(x + 1j * y))
        u = solve_harmonic(bdy)
        assert _grid_maxerr(u, bdy, dom) < tol

        # pass(2): u = Re(exp(2(x+iy))).
        bdy = lambda x, y: np.real(np.exp(2.0 * (x + 1j * y)))
        u = solve_harmonic(bdy)
        assert _grid_maxerr(u, bdy, dom) < tol

        # pass(3): u = 10*Re(exp(2(x+iy))), MATLAB tol 10*tol.
        bdy = lambda x, y: 10.0 * np.real(np.exp(2.0 * (x + 1j * y)))
        u = solve_harmonic(bdy)
        assert _grid_maxerr(u, bdy, dom) < 10.0 * tol

        # pass(4): u = Re((x+iy)^2).
        bdy = lambda x, y: np.real((x + 1j * y) ** 2)
        u = solve_harmonic(bdy)
        assert _grid_maxerr(u, bdy, dom) < tol

        # pass(5): linearity -- superposition of four single-edge solves equals
        # the all-edges solve.  MATLAB tol 1e10*tol.
        def solve_edges(lbc, rbc, ubc, dbc):
            N = Chebop2(lambda u: u.diff(2, 0) + u.diff(0, 2))
            N.lbc, N.rbc, N.ubc, N.dbc = lbc, rbc, ubc, dbc
            return N.solve(0.0)

        g = lambda x: (1.0 + x) * (1.0 - x)
        u1 = solve_edges(g, 0.0, 0.0, 0.0)
        u2 = solve_edges(0.0, g, 0.0, 0.0)
        u3 = solve_edges(0.0, 0.0, g, 0.0)
        u4 = solve_edges(0.0, 0.0, 0.0, g)
        u = solve_edges(g, g, g, g)

        xs = np.linspace(-1.0, 1.0, 40)
        X, Y = np.meshgrid(xs, xs)
        xj, yj = jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())
        A = np.asarray(u(xj, yj))
        B = (np.asarray(u1(xj, yj)) + np.asarray(u2(xj, yj))
             + np.asarray(u3(xj, yj)) + np.asarray(u4(xj, yj)))
        assert np.max(np.abs(A - B)) < 1e10 * tol

        # pass(6): lap(u) = div(grad(u)) notation.
        N = Chebop2(lambda u: -divergence(gradient(u)))
        N.lbc = 0.0
        N.rbc = 0.0
        N.dbc = 0.0
        N.ubc = 0.0
        u = N.solve(1.0)
        M = Chebop2(lambda u: -laplacian(u))
        M.lbc = 0.0
        M.rbc = 0.0
        M.dbc = 0.0
        M.ubc = 0.0
        exact = M.solve(1.0)
        assert _grid_maxerr(
            u, lambda x, y: np.asarray(exact(jnp.asarray(x), jnp.asarray(y))),
            dom) < tol

        # pass(7): lap(u) exists.
        N = Chebop2(lambda u: -lap(u))
        N.lbc = 0.0
        N.rbc = 0.0
        N.dbc = 0.0
        N.ubc = 0.0
        exact2 = N.solve(1.0)
        assert _grid_maxerr(
            u, lambda x, y: np.asarray(exact2(jnp.asarray(x), jnp.asarray(y))),
            dom) < tol
