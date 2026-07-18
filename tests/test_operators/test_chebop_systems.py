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
    def test_clamped_beam(self):
        # Euler-Bernoulli clamped-clamped beam on [0, 1]:
        # u'''' = lambda u with u = u' = 0 at both ends.  Exact
        # eigenvalues are beta^4 with cosh(beta) cos(beta) = 1
        # (beta_1 ~ 4.730040744862704).  Exercises the callable
        # multi-condition BC path of eigs_generalized.
        import warnings

        from chebfunjax.operators.chebop import Chebop

        A = Chebop(lambda x, u: u.diff(4), domain=(0.0, 1.0))
        B = Chebop(lambda x, u: u, domain=(0.0, 1.0))
        A.lbc = lambda u: [u, u.diff()]
        A.rbc = lambda u: [u, u.diff()]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, lam = A.eigs_generalized(B, k=3, n=64)
        lam = np.sort(np.abs(np.asarray(lam)))
        betas = np.array([4.730040744862704, 7.853204624095838,
                          10.995607838001671])
        # collocation at n=81 (the finer of the two agreement
        # resolutions) resolves beta_1^4 to ~2.4e-6 relative
        np.testing.assert_allclose(lam, betas ** 4, rtol=1e-5)


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
