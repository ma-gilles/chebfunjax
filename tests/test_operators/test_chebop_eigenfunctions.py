"""Two-output ``eigs`` -- eigenFUNCTION recovery (Fable 5).

MATLAB ``[V, D] = eigs(N)`` returns eigenfunctions as a quasimatrix of
chebfuns.  These tests exercise the ``return_eigenfunctions=True`` form of
:meth:`Chebop.eigs`, :meth:`Linop.eigs`, the module-level
:func:`chebfunjax.chebfun1d.ode.eigs`, and the periodic (trigcolloc) path,
verifying the residual ``||N(v) - lambda v||`` is small and that the
eigenfunctions carry the MATLAB L2-normalization / sign convention.

Provenance
----------
MATLAB source : @chebop/eigs.m, @linop/eigs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from chebfunjax.chebfun1d.ode import eigs as ode_eigs  # noqa: E402
from chebfunjax.operators.chebop import Chebop  # noqa: E402
from chebfunjax.tech.trigtech import Trigtech  # noqa: E402

PI = float(np.pi)


class TestScalarEigenfunctions:
    def test_dirichlet_laplacian_residual(self):
        # -u'' = lambda u on [0, pi], Dirichlet: eigenfunctions sqrt(2/pi) sin(k x).
        N = Chebop(lambda x, u: -u.diff(2), domain=(0.0, PI))
        N.lbc = 0.0
        N.rbc = 0.0
        lam, V = N.eigs(k=5, return_eigenfunctions=True)
        lam = np.real(np.asarray(lam))
        assert len(V) == 5
        xs = jnp.asarray(np.linspace(0.15, PI - 0.15, 40))
        for i, v in enumerate(V):
            res = -v.diff(2)(xs) - lam[i] * v(xs)
            scale = max(float(jnp.max(jnp.abs(v(xs)))) * abs(lam[i]), 1.0)
            assert float(jnp.max(jnp.abs(res))) < 1e-6 * scale

    def test_eigenfunctions_match_sine_modes(self):
        # Analytic eigenfunctions are sqrt(2/pi) sin(k x), L2-normalized.
        N = Chebop(lambda x, u: -u.diff(2), domain=(0.0, PI))
        N.lbc = 0.0
        N.rbc = 0.0
        lam, V = N.eigs(k=4, return_eigenfunctions=True)
        lam = np.real(np.asarray(lam))
        # eigenvalues k^2, k = 1..4
        assert np.max(np.abs(np.sort(lam) - np.array([1.0, 4.0, 9.0, 16.0]))) < 1e-6
        xs = jnp.asarray(np.linspace(0.0, PI, 60))
        amp = np.sqrt(2.0 / PI)
        for i, v in enumerate(V):
            k = int(round(np.sqrt(lam[i])))
            target = amp * np.sin(k * np.asarray(xs))
            got = np.asarray(v(xs))
            # match up to global sign (the reference sign is pinned by the
            # MATLAB fevalPoint convention, verified separately below)
            err = min(np.max(np.abs(got - target)),
                      np.max(np.abs(got + target)))
            assert err < 1e-6

    def test_sign_convention_positive_right_of_midpoint(self):
        # MATLAB @linop/eigs.m makes real(v(fevalPoint)) > 0 with
        # fevalPoint = a + (b-a)*0.500023981.
        N = Chebop(lambda x, u: -u.diff(2), domain=(0.0, PI))
        N.lbc = 0.0
        N.rbc = 0.0
        _, V = N.eigs(k=4, return_eigenfunctions=True)
        x_sign = jnp.asarray(0.0 + PI * 0.500023981)
        for v in V:
            assert float(jnp.real(v(x_sign))) > 0.0

    def test_module_level_eigs_return_eigenfunctions(self):
        # chebfun1d.ode.eigs forwards the two-output form.
        out = ode_eigs(lambda x, u: -u.diff(2), domain=(0.0, PI),
                       lbc=0.0, rbc=0.0, k=3, return_eigenfunctions=True)
        assert isinstance(out, tuple) and len(out) == 2
        lam, V = out
        lam = np.real(np.asarray(lam))
        xs = jnp.asarray(np.linspace(0.2, PI - 0.2, 30))
        for i, v in enumerate(V):
            res = -v.diff(2)(xs) - lam[i] * v(xs)
            scale = max(float(jnp.max(jnp.abs(v(xs)))) * abs(lam[i]), 1.0)
            assert float(jnp.max(jnp.abs(res))) < 1e-6 * scale

    def test_eigenvalues_only_backward_compatible(self):
        # Without the flag, the scalar path returns eigenvalues alone.
        N = Chebop(lambda x, u: -u.diff(2), domain=(0.0, PI))
        N.lbc = 0.0
        N.rbc = 0.0
        lam = N.eigs(k=4)
        assert not isinstance(lam, tuple)
        assert np.asarray(lam).shape == (4,)


class TestPeriodicEigenfunctions:
    def test_periodic_laplacian_trig_and_residual(self):
        # -u'' = lambda u on [0, 2pi], periodic.
        L = Chebop(lambda u: -u.diff(2), (0.0, 2 * PI))
        L.bc = "periodic"
        V, D = L.eigs(k=7, return_eigenfunctions=True)
        D = np.real(np.asarray(D))
        # Eigenfunctions are trigonometric.
        assert isinstance(V[0].funs[0].tech, Trigtech)
        xs = jnp.asarray(np.linspace(0.1, 2 * PI - 0.1, 40))
        for i, v in enumerate(V):
            res = -v.diff(2)(xs) - D[i] * v(xs)
            scale = max(float(jnp.max(jnp.abs(v(xs)))) * abs(D[i]), 1.0)
            assert float(jnp.max(jnp.abs(res))) < 1e-6 * scale

    def test_mathieu_LR_selects_smooth_modes(self):
        # A.op = u'' - 2q cos(2x) u, periodic, eigs(A, 16, 'LR').
        # 'LR' must pick the smooth low-order Mathieu functions, not the
        # high-frequency modes (the earlier "sawtooth" bug picked the most
        # negative eigenvalues).
        q = 10.0
        N = Chebop(lambda x, u: u.diff(2) - 2 * q * jnp.cos(2 * x) * u,
                   domain=(-PI, PI))
        N.bc = "periodic"
        V, lam = N.eigs(k=16, sigma="LR", return_eigenfunctions=True)
        lam = np.real(np.asarray(lam))
        # Descending by real part: the leading (ground) eigenvalue is the
        # largest, ~13.94 for q = 10.
        assert lam[0] > lam[-1]
        assert abs(lam[0] - 13.9366) < 1e-2
        # V(:,9) elliptic cosine (even), V(:,10) elliptic sine (odd).
        tt = jnp.linspace(-PI, PI, 400)
        vc = np.asarray(V[8](tt))
        vs = np.asarray(V[9](tt))
        # parity
        assert np.max(np.abs(vc - vc[::-1])) < 1e-6           # even
        assert np.max(np.abs(vs + vs[::-1])) < 1e-6           # odd
        # smoothness: bounded second difference (a sawtooth would be O(1))
        rough_c = np.max(np.abs(np.diff(vc, 2))) / max(np.max(np.abs(vc)), 1e-9)
        rough_s = np.max(np.abs(np.diff(vs, 2))) / max(np.max(np.abs(vs)), 1e-9)
        assert rough_c < 0.02 and rough_s < 0.02
        # residual N(v) - lam v small (avoid endpoints; use interior pts)
        xs = jnp.asarray(np.linspace(-PI + 0.2, PI - 0.2, 40))
        for i in (8, 9):
            v = V[i]
            nv = v.diff(2)(xs) - 2 * q * jnp.cos(2 * xs) * v(xs)
            res = nv - lam[i] * v(xs)
            scale = max(float(jnp.max(jnp.abs(v(xs)))) * abs(lam[i]), 1.0)
            assert float(jnp.max(jnp.abs(res))) < 1e-5 * scale

    def test_mathieu_L2_normalized(self):
        # Eigenfunctions come back L2-normalized (unit norm).
        q = 10.0
        N = Chebop(lambda x, u: u.diff(2) - 2 * q * jnp.cos(2 * x) * u,
                   domain=(-PI, PI))
        N.bc = "periodic"
        V, _ = N.eigs(k=16, sigma="LR", return_eigenfunctions=True)
        for i in (8, 9):
            assert abs(float(V[i].norm(2)) - 1.0) < 1e-6
