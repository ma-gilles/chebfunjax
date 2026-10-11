"""Literal MATLAB Poisson tests and a retained manufactured-solution control.

Manufactured solution u = (1 - r^2) r^l Y_lm with
lap u = -(4l + 6) r^l Y_lm, u|r=1 = 0 (the identity verified in the
Opus 4.8 session; here vs the CLOSED FORM, not the library laplacian).

Provenance
----------
MATLAB source : tests/ballfun/test_poisson.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.spherefun.spherefun import Spherefun

from ._helpers import L0, R0, T0


class TestBallfunPoisson:
    def test_manufactured_solution(self):
        ell, m = 2, 1
        Y = Spherefun.sphharm(ell, m)

        def u_exact(r, lam, th):
            return (1 - r ** 2) * r ** ell * Y(lam, th)

        def rhs(r, lam, th):
            return -(4 * ell + 6) * r ** ell * Y(lam, th)

        u = Ballfun.poisson(rhs, lmax=8, nr=24)
        got = float(u(R0, L0, T0))
        want = float(u_exact(R0, L0, T0))
        assert abs(got - want) < 1e-8


class TestLiteralPoisson:
    def test_dirichlet(self):
        import jax.numpy as jnp

        from chebfunjax.chebpref import ChebfunPref
        tol = 1e7 * ChebfunPref().techPrefs.chebfuneps
        f = Ballfun.from_function(lambda x, y, z: 0)
        exact_op = lambda r, lam, th: r * jnp.sin(lam) * jnp.sin(th)
        bc = lambda lam, th: exact_op(1, th, lam)
        u = Ballfun.poisson(f, bc, 39, 40, 41)
        exact = Ballfun.from_function(exact_op, spherical=True)
        assert (u - exact).norm() < tol

    def test_neumann(self):
        import jax.numpy as jnp

        from chebfunjax.chebpref import ChebfunPref
        tol = 1e7 * ChebfunPref().techPrefs.chebfuneps
        exact = Ballfun.from_function(lambda r, lam, th: r**2*jnp.sin(th)**2, spherical=True)
        f = exact.laplacian()
        bc = lambda lam, th: 2*jnp.sin(th)**2
        u = Ballfun.poisson(f, bc, 39, 40, 41, "neumann")
        assert (u.diff(1) - exact.diff(1)).norm() < tol
        assert (u.diff(2) - exact.diff(2)).norm() < tol
        assert (u.diff(3) - exact.diff(3)).norm() < tol
