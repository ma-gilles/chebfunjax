"""Core-suite coverage for the chebop ODE-systems paths (Fable 5).

Mirrors the MATLAB-port assertions (test_linearSystem1,
test_system3, test_eigs_system) so the systems solver counts toward
the core coverage gate.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.chebop import Chebop


class TestLinearSystem:
    def test_sin_cos_2x2(self):
        d = (-np.pi, np.pi)
        A = Chebop(lambda x, u, v: [u - v.diff(), u.diff() + v], d)
        A.lbc = lambda u, v: u + 1
        A.rbc = lambda u, v: v
        sol = A.solve(0)
        xs = jnp.asarray(np.linspace(-3.0, 3.0, 40))
        assert float(jnp.max(jnp.abs(sol[0](xs) - jnp.cos(xs)))) \
            < 1e-8
        assert float(jnp.max(jnp.abs(sol[1](xs) - jnp.sin(xs)))) \
            < 1e-8
        assert len(sol.blocks) == 2


class TestNonlinearSystem:
    def test_coupled_sine_cosine_bvp(self):
        N = Chebop(
            lambda x, u, v: [u.diff(2) + v.sin(),
                             u.cos() + v.diff(2)],
            (-1.0, 1.0))
        N.lbc = lambda u, v: [u - 2, v - 1]
        N.rbc = lambda u, v: [u - 2, v + 1]
        sol = N.solve([0, 0], n=32)
        v = sol[1]
        assert abs(float(v(jnp.asarray(0.2)))
                   - (-0.371250985730553)) < 1e-7


class TestSystemEigs:
    def test_maxwell_inspired(self):
        d = (0.0, np.pi)
        A = Chebop(lambda x, u, v: [-u + v.diff(), u.diff()], d)
        A.lbc = lambda u, v: u
        A.rbc = lambda u, v: u
        _, lam = A.eigs(k=5, n=48)
        lam = np.sort(np.abs(np.asarray(lam)))
        correct = np.sort(np.abs(np.array([
            0, -0.5 + np.sqrt(3) / 2 * 1j,
            -0.5 - np.sqrt(3) / 2 * 1j,
            -0.5 + np.sqrt(15) / 2 * 1j,
            -0.5 - np.sqrt(15) / 2 * 1j])))
        assert np.max(np.abs(lam - correct)) < 1e-9


class TestSystemIVP:
    def test_linear_ivp_system_time_marching(self):
        # u' = v, v' = -u with u(0)=1, v(0)=0  ->  u = cos t
        N = Chebop(lambda t, u, v: [u.diff() - v, v.diff() + u],
                   (0.0, 2.0))
        N.lbc = lambda u, v: [u - 1, v]
        sol = N.solve([0, 0])
        ts = jnp.asarray(np.linspace(0.0, 2.0, 20))
        assert float(jnp.max(jnp.abs(sol[0](ts) - jnp.cos(ts)))) \
            < 1e-8
        assert float(jnp.max(jnp.abs(sol[1](ts) + jnp.sin(ts)))) \
            < 1e-8


class TestClampedGeneralizedEigs:
    def test_orr_sommerfeld_critical(self):
        # Orr-Sommerfeld at Re = 5772.22: the critical (largest real
        # part) eigenvalue matches MATLAB's v4 reference.  Exercises
        # eigs_generalized's complex probing, callable multi-condition
        # (clamped) BCs, and 'LR' selection.  A clamped-beam pencil
        # (D4 vs I) was tried first but D4's ~n^8 conditioning makes
        # its small eigenvalues platform-sensitive; the OS pencil
        # (B = D2 - I, eigenvalues O(1)) is robust across BLAS
        # implementations.
        import warnings

        Re, alph = 5772.22, 1.0
        A = Chebop(
            lambda x, u: (u.diff(4) - 2 * alph ** 2 * u.diff(2)
                          + alph ** 4 * u) / Re
            - 2j * alph * u
            - 1j * alph * ((1 - x * x) * (u.diff(2) - alph ** 2 * u)),
            domain=(-1.0, 1.0))
        B = Chebop(lambda x, u: u.diff(2) - u, domain=(-1.0, 1.0))
        A.lbc = lambda u: [u, u.diff()]
        A.rbc = lambda u: [u, u.diff()]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, lam = A.eigs_generalized(B, k=6, n=96, sort="LR")
        e = np.asarray(lam)
        e_crit = e[np.argmax(e.real)]
        e_crit_v4 = -0.000078029804093 - 0.261565915010080j
        assert abs(e_crit - e_crit_v4) < 5e-6


class TestGeneralizedEigs:
    def test_drum_bessel_zeros(self):
        from scipy.special import jn_zeros
        A = Chebop(lambda r, u: r * u.diff(2) + u.diff(),
                   (0.0, 1.0))
        A.lbc = "neumann"
        A.rbc = "dirichlet"
        B = Chebop(lambda r, u: r * u, (0.0, 1.0))
        _, lam = A.eigs_generalized(B, k=3, n=64)
        omega = np.sort(np.sqrt(-np.real(np.asarray(lam))))
        assert np.max(np.abs(omega - jn_zeros(0, 3))) < 1e-7


class TestPiecewiseDomain:
    """Piecewise-domain (interior-breakpoint) collocation, exercising the
    _solve_piecewise path so it counts toward the core coverage gate."""

    def test_scalar_matches_smooth_solve(self):
        # u'' + u = 0, u(-1) = 1, u(1) = 0.  The piecewise solve on a domain
        # with an interior breakpoint at 0.3 must agree with the smooth
        # single-interval solve of the same problem.
        def op(x, u):
            return u.diff(2) + u
        N_smooth = Chebop(op, domain=(-1.0, 1.0))
        N_smooth.lbc = 1.0
        N_smooth.rbc = 0.0
        u_smooth = N_smooth.solve(0.0)

        N_piece = Chebop(op, domain=(-1.0, 0.3, 1.0))
        N_piece.lbc = 1.0
        N_piece.rbc = 0.0
        u_piece = N_piece.solve(0.0)

        xs = jnp.asarray(np.linspace(-1.0, 1.0, 41))
        assert float(jnp.max(jnp.abs(u_piece(xs) - u_smooth(xs)))) < 1e-8
        # boundary conditions and continuity across the break at 0.3
        assert abs(float(u_piece(jnp.asarray(-1.0))) - 1.0) < 1e-9
        assert abs(float(u_piece(jnp.asarray(1.0)))) < 1e-9
        jump = (float(u_piece.funs[1](jnp.asarray(0.3)))
                - float(u_piece.funs[0](jnp.asarray(0.3))))
        assert abs(jump) < 1e-9

    def test_scalar_two_interior_breaks(self):
        # Three pieces: verify continuity holds at both interior breaks.
        def op(x, u):
            return u.diff(2) + u
        N = Chebop(op, domain=(-1.0, -0.2, 0.5, 1.0))
        N.lbc = 1.0
        N.rbc = 0.0
        u = N.solve(0.0)
        for bp, i in ((-0.2, 0), (0.5, 1)):
            jump = (float(u.funs[i + 1](jnp.asarray(bp)))
                    - float(u.funs[i](jnp.asarray(bp))))
            assert abs(jump) < 1e-9
        assert float(jnp.max(jnp.abs(
            (u.diff(2) + u)(jnp.asarray(np.linspace(-0.99, 0.99, 60)))))
        ) < 1e-8
