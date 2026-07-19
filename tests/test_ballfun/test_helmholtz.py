"""Core unit tests for the spectral ball Helmholtz/Poisson solver (Fable 5).

Exercises :meth:`Ballfun.helmholtz` — the coefficient-space
Chebyshev(r) x Fourier(lambda) x Fourier(theta) solver with per-mode QZ
Sylvester solves and the Legendre DC-mode Neumann branch (a port of
@ballfun/helmholtz.m) — against manufactured solutions, with no MATLAB
golden reference required.

Provenance
----------
MATLAB source : @ballfun/helmholtz.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun

# Sampling grid strictly inside the ball, avoiding r = 0 and the poles.
RS = jnp.asarray(np.linspace(0.05, 1.0, 7))
LAMS = jnp.asarray(np.linspace(-3.0, 3.0, 7))
THS = jnp.asarray(np.linspace(0.1, 3.0, 7))
RR, LL, TT = jnp.meshgrid(RS, LAMS, THS, indexing="ij")


def _err(u, exact) -> float:
    return float(jnp.max(jnp.abs(u(RR, LL, TT) - exact(RR, LL, TT))))


class TestBallfunHelmholtzDirichlet:
    def test_constant_solution(self):
        # Delta u = 0, u|r=1 = 1  ->  u = 1
        u = Ballfun.helmholtz(lambda r, l, t: 0.0 * r, 0.0,
                              lambda l, t: 1.0 + 0.0 * l)
        assert _err(u, lambda r, l, t: 1.0 + 0.0 * r) < 1e-11

    def test_radial_quadratic(self):
        # Delta(r^2) = 6, u|r=1 = 1  ->  u = r^2
        u = Ballfun.helmholtz(lambda r, l, t: 6.0 + 0.0 * r, 0.0,
                              lambda l, t: 1.0 + 0.0 * l)
        assert _err(u, lambda r, l, t: r ** 2) < 1e-11

    def test_angular_dependence(self):
        # Delta(r^2 sin^2 th) = 4, u|r=1 = sin^2 th
        u = Ballfun.helmholtz(lambda r, l, t: 4.0 + 0.0 * r, 0.0,
                              lambda l, t: jnp.sin(t) ** 2)
        assert _err(u, lambda r, l, t: r ** 2 * jnp.sin(t) ** 2) < 1e-11

    def test_nonzero_frequency(self):
        # Delta u + K^2 u = 4 + K^2 (r^2 sin^2 th), u|r=1 = sin^2 th
        K = 2.0
        u = Ballfun.helmholtz(
            lambda r, l, t: 4.0 + K * K * (r ** 2 * jnp.sin(t) ** 2), K,
            lambda l, t: jnp.sin(t) ** 2)
        assert _err(u, lambda r, l, t: r ** 2 * jnp.sin(t) ** 2) < 1e-11

    def test_homogeneous_boundary(self):
        # Manufactured Dirichlet: u = (1 - r^2) r^2 sin^2 th, u|r=1 = 0.
        def exact(r, l, t):
            return (1.0 - r ** 2) * r ** 2 * jnp.sin(t) ** 2

        ex = Ballfun.from_function(exact, spherical=True)
        f = ex.laplacian()
        u = Ballfun.helmholtz(f, 0.0, None, 40, 40, 40)
        assert _err(u, exact) < 1e-10


class TestBallfunHelmholtzNeumann:
    def test_neumann_manufactured(self):
        # exact = r^4 sin^2 th ; Neumann data du/dr|r=1 = 4 sin^2 th.
        def exact(r, l, t):
            return r ** 4 * jnp.sin(t) ** 2

        ex = Ballfun.from_function(exact, spherical=True)
        K = 2.0
        f = ex.laplacian() + K * K * ex
        u = Ballfun.helmholtz(f, K, lambda l, t: 4.0 * jnp.sin(t) ** 2,
                              40, 40, 40, bc_type="neumann")
        assert _err(u, exact) < 1e-10

    def test_poisson_neumann_dc_mode(self):
        # Pure radial K=0 Poisson-Neumann triggers the Legendre DC-mode
        # branch: Delta u = 6, du/dr|r=1 = 2 -> u = r^2 + const (the additive
        # constant is fixed by the mean constraint).  Check u - r^2 is
        # constant across the ball.
        u = Ballfun.helmholtz(lambda r, l, t: 6.0 + 0.0 * r, 0.0,
                              lambda l, t: 2.0 + 0.0 * l,
                              40, 40, 40, bc_type="neumann")
        resid = np.asarray(u(RR, LL, TT)) - np.asarray(RR ** 2)
        assert float(resid.max() - resid.min()) < 1e-11

    def test_boundary_matrix_input(self):
        # Neumann data supplied as an (n, p) Fourier-Fourier coefficient
        # matrix (the form the Helmholtz decomposition uses) must match the
        # callable path.  Constant data 2.0 -> only the DC-DC entry is set.
        n, p = 40, 40
        bc_mat = np.zeros((n, p), dtype=np.complex128)
        bc_mat[n // 2, p // 2] = 2.0
        u_mat = Ballfun.helmholtz(lambda r, l, t: 6.0 + 0.0 * r, 0.0, bc_mat,
                                  40, n, p, bc_type="neumann")
        u_call = Ballfun.helmholtz(lambda r, l, t: 6.0 + 0.0 * r, 0.0,
                                   lambda l, t: 2.0 + 0.0 * l,
                                   40, n, p, bc_type="neumann")
        diff = float(jnp.max(jnp.abs(u_mat(RR, LL, TT) - u_call(RR, LL, TT))))
        assert diff < 1e-11


class TestBallfunPoissonWrapper:
    def test_dirichlet_scalar_boundary(self):
        # Scalar boundary data broadcast onto the grid: Delta u = 6,
        # u|r=1 = 5  ->  u = r^2 + 4.
        u = Ballfun.helmholtz(lambda r, l, t: 6.0 + 0.0 * r, 0.0, 5.0)
        assert _err(u, lambda r, l, t: r ** 2 + 4.0) < 1e-11
