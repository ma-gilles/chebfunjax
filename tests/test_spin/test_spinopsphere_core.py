"""Core (non-MATLAB-port) tests for chebfunjax.operators.spinopsphere.

Mirror tests for the sphere stiff-PDE time-stepper (Spinopsphere /
spinsphere): construction, the DFS Fourier-Fourier Laplace-Beltrami
discretization, the 2-D trig transforms, scheme selection, and an
end-to-end self-convergence smoke test at reduced resolution.

Provenance
----------
MATLAB source : spinsphere.m, @spinopsphere/*, @imex/*
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.spinopsphere import (
    FuncHandle,
    Spinopsphere,
    _coeffs2vals2,
    _dfs_grid,
    _discretize,
    _vals2coeffs2,
    func2str,
    spinsphere,
)
from chebfunjax.spherefun.spherefun import Spherefun

# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


class TestConstruction:
    def test_func_handle_str(self):
        h = FuncHandle(lambda u: u - u ** 3, "@(u)u-u.^3")
        assert func2str(h) == "@(u)u-u.^3"
        assert str(h) == "@(u)u-u.^3"
        # still callable
        assert np.allclose(h(np.array([2.0])), np.array([2.0 - 8.0]))

    def test_ac_preset(self):
        S = Spinopsphere("AC")
        assert S.tspan == (0.0, 60.0)
        assert S.lin_scale == 1e-2
        assert func2str(S.nonlin) == "@(u)u-u.^3"
        assert isinstance(S.init, Spherefun)
        assert S.numVars == 1
        # domain is the MATLAB sphere domain [-pi, pi, 0, pi]
        assert S.domain == pytest.approx((-np.pi, np.pi, 0.0, np.pi))

    def test_gl_preset(self):
        S = Spinopsphere("GL")
        assert S.tspan == (0.0, 100.0)
        assert S.lin_scale == 1e-3
        assert func2str(S.nonlin) == "@(u)u-(1+1.5i)*u.*(abs(u).^2)"
        # nonlin evaluates the complex Ginzburg-Landau reaction term
        u = np.array([1.0 + 0.0j, 0.5 - 0.5j])
        expected = u - (1.0 + 1.5j) * u * (np.abs(u) ** 2)
        assert np.allclose(S.nonlin(u), expected)

    def test_nls_preset(self):
        S = Spinopsphere("NLS")
        assert S.tspan == (0.0, 3.0)
        assert S.lin_scale == 1j
        assert func2str(S.nonlin) == "@(u)1i*u.*abs(u).^2"

    def test_tspan_constructor(self):
        S = Spinopsphere([0, 1])
        assert S.tspan == (0.0, 1.0)
        assert S.init is None
        S2 = Spinopsphere([0.0, 2.5, 5.0])
        assert S2.tspan == (0.0, 2.5, 5.0)

    def test_gm_not_supported(self):
        with pytest.raises(NotImplementedError):
            Spinopsphere("GM")

    def test_unknown_pde(self):
        with pytest.raises(ValueError):
            Spinopsphere("nope")


# ----------------------------------------------------------------------
# DFS 2-D trigonometric transforms
# ----------------------------------------------------------------------


class TestTransforms:
    def test_roundtrip(self):
        rng = np.random.default_rng(0)
        V = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
        back = _coeffs2vals2(_vals2coeffs2(V))
        assert np.allclose(back, V, atol=1e-12)

    def test_single_mode(self):
        # A single Fourier-Fourier mode round-trips to exp(i k th) exp(i l lam).
        N = 16
        ll, tt = _dfs_grid(N)
        V = np.exp(1j * 3 * tt) * np.exp(-1j * 2 * ll)
        C = _vals2coeffs2(V)
        # coefficient matrix: theta modes on axis 0, lambda modes on axis 1,
        # ordered -N/2..N/2-1.  Peak at (k=3, l=-2).
        peak = np.unravel_index(np.argmax(np.abs(C)), C.shape)
        assert peak == (N // 2 + 3, N // 2 - 2)
        assert np.abs(C[peak]) == pytest.approx(1.0, abs=1e-12)


# ----------------------------------------------------------------------
# Laplace-Beltrami discretization
# ----------------------------------------------------------------------


class TestDiscretize:
    def test_sin2_laplacian_matches_spherefun(self):
        # The DFS operator applies sin^2(theta) * Laplace-Beltrami in
        # coefficient space.  Compare to the independent spherical-harmonic
        # route (spherefun.laplacian), multiplied by sin^2(theta).
        N = 48
        f = Spherefun.from_function(
            lambda lam, th: jnp.cos(
                jnp.cosh(0.5 * jnp.cos(lam) * jnp.sin(th) * jnp.cos(th))
                - 0.5 * jnp.sin(lam) * jnp.sin(th)
            )
        )
        lap_true = f.laplacian()

        ll, tt = _dfs_grid(N)
        C = _vals2coeffs2(np.asarray(f(jnp.asarray(ll), jnp.asarray(tt))))
        Tsin2, B, kn = _discretize(N)
        # sin^2 * Laplacian in coeff space: A=1 -> B @ C - C * kn^2
        out = B @ C - C * (kn ** 2)[None, :]
        sin2lap = np.real(_coeffs2vals2(out))

        target = np.sin(tt) ** 2 * np.asarray(
            lap_true(jnp.asarray(ll), jnp.asarray(tt))
        )
        rel = np.max(np.abs(sin2lap - target)) / np.max(np.abs(target))
        assert rel < 1e-9

    def test_tsin2_multiplies_by_sin_squared(self):
        # Tsin2 @ coeffs(g) == coeffs(sin^2(theta) * g).
        N = 32
        Tsin2, _, _ = _discretize(N)
        th = -np.pi + 2 * np.pi * np.arange(N) / N
        g = np.exp(1j * 3 * th) + 2.0 + 0.5 * np.exp(-1j * 2 * th)

        def v2c(v):
            n = v.shape[0]
            c = np.fft.fftshift(np.fft.fft(v)) / n
            ks = np.arange(-(n // 2), n // 2)
            return c * ((-1.0 + 0j) ** ks)

        def c2v(c):
            n = c.shape[0]
            ks = np.arange(-(n // 2), n // 2)
            c = c * ((-1.0 + 0j) ** ks)
            return np.fft.ifft(np.fft.ifftshift(n * c))

        got = c2v(np.asarray(Tsin2 @ v2c(g)))
        assert np.allclose(got, np.sin(th) ** 2 * g, atol=1e-12)


# ----------------------------------------------------------------------
# spinsphere end-to-end
# ----------------------------------------------------------------------


class TestSpinsphere:
    def test_missing_init_raises(self):
        S = Spinopsphere([0.0, 1.0])
        S.lin_scale = 1e-2
        S.nonlin = FuncHandle(lambda u: u - u ** 3, "@(u)u-u.^3")
        with pytest.raises(ValueError):
            spinsphere(S, 16, 0.1)

    def test_ac_returns_real_spherefun(self):
        # Reduced-resolution smoke: output is a real Spherefun, finite,
        # and bounded near the initial sup-norm.
        S = Spinopsphere("AC")
        S.tspan = (0.0, 1.0)
        u = spinsphere(S, 48, 0.1)
        assert isinstance(u, Spherefun)
        lam = np.linspace(-np.pi, np.pi, 30)
        xx, yy = np.meshgrid(lam, lam)
        vals = np.asarray(u(jnp.asarray(xx), jnp.asarray(yy)))
        assert np.all(np.isfinite(vals))
        assert np.max(np.abs(vals)) < 2.0

    def test_ac_self_convergence(self):
        # Fourth-order self-convergence (dt vs dt/2) at reduced N.
        S = Spinopsphere("AC")
        S.tspan = (0.0, 2.0)
        N = 64
        u = spinsphere(S, N, 0.1)
        v = spinsphere(S, N, 0.05)
        lam = np.linspace(-np.pi, np.pi, 50)
        xx, yy = np.meshgrid(lam, lam)
        uu = np.asarray(u(jnp.asarray(xx), jnp.asarray(yy)))
        vv = np.asarray(v(jnp.asarray(xx), jnp.asarray(yy)))
        rel = np.max(np.abs(uu - vv)) / np.max(np.abs(vv))
        assert rel < 1e-2

    def test_complex_linear_uses_lirk4(self):
        # A complex Laplacian constant (dispersive PDE, e.g. NLS) is
        # stepped with LIRK4 throughout, not IMEX-BDF4.  Exercise that
        # branch with a short, low-resolution dispersive solve and check
        # the output is a finite Spherefun.
        S = Spinopsphere([0.0, 0.1])
        S.lin_scale = 1j
        S.nonlin = FuncHandle(
            lambda u: 1j * u * (abs(u) ** 2), "@(u)1i*u.*abs(u).^2"
        )
        S.init = Spherefun.from_function(
            lambda lam, th: jnp.cos(jnp.sin(th) * jnp.cos(lam))
        )
        u = spinsphere(S, 32, 0.01)
        assert isinstance(u, Spherefun)
        vals = np.asarray(u(jnp.asarray(0.3), jnp.asarray(1.1)))
        assert np.all(np.isfinite(vals))
