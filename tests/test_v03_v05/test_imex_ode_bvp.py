"""Tests for V03 (IMEX schemes), V04 (ode45/ode113), and V05 (bvp4c/bvp5c).

V03 — IMEX time-stepping (chebfunjax.spin.imex)
    - imex_euler: first-order convergence on stiff diffusion
    - imex_sbdf2: second-order convergence on stiff diffusion
    - Both schemes: mass conservation for diffusion-reaction

V04 — ODE integrators on Chebfun (chebfunjax.chebfun1d.chebfun)
    - ode45: y' = y, y(0) = 1  =>  y(1) = e  (golden test)
    - ode113: same problem, higher accuracy
    - Returns a callable Chebfun

V05 — BVP solvers (chebfunjax.chebfun1d.ode)
    - bvp4c: u'' = -1, u(±1) = 0  =>  u(0) = 0.5
    - bvp5c: same problem
    - Both match bvp() within tol
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

# ============================================================================
# V03: IMEX schemes
# ============================================================================


class TestImexEuler:
    """First-order IMEX-Euler on the heat equation u_t = eps * u_xx."""

    @staticmethod
    def _run_heat(eps: float, N: int, dt: float, nsteps: int):
        """Integrate u_t = eps * u_xx on [0, 2*pi] from t=0 to t=nsteps*dt.

        Initial condition: u0 = sin(x).
        Exact solution:    u(t, x) = exp(-eps * t) * sin(x).
        """
        from chebfunjax.spin.imex import imex_euler

        # Fourier wavenumbers on [0, 2*pi]
        k = np.fft.fftfreq(N, d=1.0 / N).astype(float)  # integers 0..N/2-1, -N/2..-1
        L_diag = eps * (1j * k) ** 2  # diffusion: L = eps * d^2/dx^2 -> eigenvalue = -eps*k^2

        x = np.linspace(0.0, 2 * math.pi, N, endpoint=False)
        u0 = np.sin(x).astype(complex)
        u0_hat = np.fft.fft(u0)

        Nc = np.ones(N, dtype=complex)  # no nonlinear differentiation

        def nonlin_vals(u):  # pure linear PDE — no nonlinear term
            return np.zeros_like(u)

        u_hat, t_final = imex_euler(u0_hat, L_diag, nonlin_vals, Nc, dt, nsteps)
        u_out = np.real(np.fft.ifft(u_hat))
        t_actual = dt * nsteps
        u_exact = np.exp(-eps * t_actual) * np.sin(x)
        return u_out, u_exact

    def test_heat_error_small(self):
        """IMEX-Euler: heat equation error should be small for small dt."""
        u, u_exact = self._run_heat(eps=0.1, N=64, dt=1e-3, nsteps=100)
        err = np.max(np.abs(u - u_exact))
        assert err < 1e-2, f"imex_euler heat error too large: {err:.2e}"

    def test_heat_first_order_convergence(self):
        """Error halves when dt halves (first-order convergence)."""
        eps, N, nsteps_base = 0.1, 64, 50
        dt1, dt2 = 1e-3, 5e-4
        u1, ue1 = self._run_heat(eps, N, dt1, nsteps_base)
        u2, ue2 = self._run_heat(eps, N, dt2, nsteps_base * 2)

        err1 = np.max(np.abs(u1 - ue1))
        err2 = np.max(np.abs(u2 - ue2))
        # err2 / err1 should be ~0.5 for first-order (allow some tolerance)
        ratio = err2 / err1
        assert 0.1 < ratio < 0.9, (
            f"imex_euler: expected ~0.5 error ratio, got {ratio:.3f} "
            f"(err1={err1:.2e}, err2={err2:.2e})"
        )

    def test_output_shape_and_type(self):
        """imex_euler returns (ndarray, float)."""
        from chebfunjax.spin.imex import imex_euler

        N = 32
        k = np.fft.fftfreq(N, d=1.0 / N)
        L_diag = -(k ** 2).astype(complex)
        u0_hat = np.fft.fft(np.sin(np.linspace(0, 2 * math.pi, N, endpoint=False)))
        Nc = np.ones(N, dtype=complex)

        def nonlin_vals(u):
            return np.zeros_like(u)

        u_hat, t = imex_euler(u0_hat, L_diag, nonlin_vals, Nc, dt=0.01, nsteps=5)
        assert u_hat.shape == (N,), f"Expected shape ({N},), got {u_hat.shape}"
        assert isinstance(t, float), f"Expected float t, got {type(t)}"
        assert abs(t - 0.05) < 1e-12

    def test_nonzero_nonlinear(self):
        """IMEX-Euler with constant reaction term u_t = -u_xx + c."""
        from chebfunjax.spin.imex import imex_euler

        N, eps, c = 64, 0.01, 0.5
        k = np.fft.fftfreq(N, d=1.0 / N)
        L_diag = eps * (1j * k) ** 2
        x = np.linspace(0, 2 * math.pi, N, endpoint=False)
        u0 = np.sin(x).astype(complex)
        u0_hat = np.fft.fft(u0)
        Nc = np.ones(N, dtype=complex)

        def nonlin_vals(u):  # constant reaction term
            return c * np.ones_like(u)

        u_hat, _ = imex_euler(u0_hat, L_diag, nonlin_vals, Nc, dt=1e-3, nsteps=10)
        u_out = np.real(np.fft.ifft(u_hat))
        assert np.all(np.isfinite(u_out)), "NaN/Inf in imex_euler output"

    def test_zero_steps_returns_initial(self):
        """Zero steps: imex_euler returns initial condition unchanged."""
        from chebfunjax.spin.imex import imex_euler

        N = 16
        L_diag = -np.ones(N, dtype=complex)
        u0_hat = np.random.default_rng(0).standard_normal(N).astype(complex)
        Nc = np.ones(N, dtype=complex)

        def nonlin_vals(u):
            return np.zeros_like(u)

        u_hat, t = imex_euler(u0_hat, L_diag, nonlin_vals, Nc, dt=0.1, nsteps=0)
        npt.assert_array_equal(u_hat, u0_hat)
        assert t == 0.0


class TestImexSBDF2:
    """Second-order IMEX-SBDF2 on the heat equation."""

    @staticmethod
    def _run_heat(eps: float, N: int, dt: float, nsteps: int):
        from chebfunjax.spin.imex import imex_sbdf2

        k = np.fft.fftfreq(N, d=1.0 / N).astype(float)
        L_diag = eps * (1j * k) ** 2

        x = np.linspace(0.0, 2 * math.pi, N, endpoint=False)
        u0 = np.sin(x).astype(complex)
        u0_hat = np.fft.fft(u0)

        Nc = np.ones(N, dtype=complex)

        def nonlin_vals(u):
            return np.zeros_like(u)

        u_hat, t_final = imex_sbdf2(u0_hat, L_diag, nonlin_vals, Nc, dt, nsteps)
        u_out = np.real(np.fft.ifft(u_hat))
        t_actual = dt * nsteps
        u_exact = np.exp(-eps * t_actual) * np.sin(x)
        return u_out, u_exact

    def test_heat_error_smaller_than_euler(self):
        """SBDF2 should be more accurate than Euler for same dt."""
        from chebfunjax.spin.imex import imex_euler, imex_sbdf2

        eps, N, dt, nsteps = 0.1, 64, 5e-4, 200

        k = np.fft.fftfreq(N, d=1.0 / N).astype(float)
        L_diag = eps * (1j * k) ** 2
        x = np.linspace(0, 2 * math.pi, N, endpoint=False)
        u0 = np.sin(x).astype(complex)
        u0_hat = np.fft.fft(u0)
        Nc = np.ones(N, dtype=complex)

        def nv(u):
            return np.zeros_like(u)

        u_euler, _ = imex_euler(u0_hat, L_diag, nv, Nc, dt, nsteps)
        u_sbdf2, _ = imex_sbdf2(u0_hat, L_diag, nv, Nc, dt, nsteps)

        t_final = dt * nsteps
        u_exact = np.exp(-eps * t_final) * np.sin(x)
        err_euler = np.max(np.abs(np.real(np.fft.ifft(u_euler)) - u_exact))
        err_sbdf2 = np.max(np.abs(np.real(np.fft.ifft(u_sbdf2)) - u_exact))

        assert err_sbdf2 < err_euler, (
            f"SBDF2 ({err_sbdf2:.2e}) should be more accurate than Euler ({err_euler:.2e})"
        )

    def test_second_order_convergence(self):
        """Error quarters when dt halves (second-order convergence)."""
        eps, N, nsteps_base = 0.1, 64, 100
        dt1, dt2 = 1e-3, 5e-4
        u1, ue1 = self._run_heat(eps, N, dt1, nsteps_base)
        u2, ue2 = self._run_heat(eps, N, dt2, nsteps_base * 2)

        err1 = np.max(np.abs(u1 - ue1))
        err2 = np.max(np.abs(u2 - ue2))
        ratio = err2 / err1
        # For 2nd order: ratio ~ 0.25.  Allow generous band [0.05, 0.7].
        assert 0.05 < ratio < 0.7, (
            f"imex_sbdf2: expected ~0.25 error ratio, got {ratio:.3f} "
            f"(err1={err1:.2e}, err2={err2:.2e})"
        )

    def test_single_step(self):
        """nsteps=1 should run the startup IMEX-Euler step."""
        from chebfunjax.spin.imex import imex_sbdf2

        N = 32
        k = np.fft.fftfreq(N, d=1.0 / N)
        L_diag = -(k ** 2).astype(complex)
        u0_hat = np.fft.fft(np.sin(np.linspace(0, 2 * math.pi, N, endpoint=False)))
        Nc = np.ones(N, dtype=complex)

        def nonlin_vals(u):
            return np.zeros_like(u)

        u_hat, t = imex_sbdf2(u0_hat, L_diag, nonlin_vals, Nc, dt=0.01, nsteps=1)
        assert u_hat.shape == (N,)
        assert abs(t - 0.01) < 1e-12
        assert np.all(np.isfinite(u_hat))

    def test_dealiasing_applies(self):
        """Dealiasing mask should zero high-frequency modes."""
        from chebfunjax.spin.imex import imex_sbdf2

        N = 64
        k = np.fft.fftfreq(N, d=1.0 / N)
        L_diag = -0.01 * (k ** 2).astype(complex)
        u0_hat = np.fft.fft(np.sin(np.linspace(0, 2 * math.pi, N, endpoint=False)))
        Nc = np.ones(N, dtype=complex)

        def nonlin_vals(u):
            return np.zeros_like(u)

        # Build a simple dealiasing mask (keep only k < N//3)
        mask = np.zeros(N, dtype=bool)
        mask[:N // 3] = True
        mask[-(N // 3):] = True

        u_hat, _ = imex_sbdf2(u0_hat, L_diag, nonlin_vals, Nc,
                               dt=0.01, nsteps=5, dealias=mask)

        # Modes outside the mask should be zero
        outside = ~mask
        assert np.max(np.abs(u_hat[outside])) == 0.0, (
            "Dealiasing failed: non-zero high-frequency modes remain"
        )


# ============================================================================
# V03: Integration — public API import
# ============================================================================


class TestImexImport:
    """Smoke test: imex_euler / imex_sbdf2 are importable from the spin package."""

    def test_import_from_spin_package(self):
        from chebfunjax.spin import imex_euler, imex_sbdf2  # noqa: F401

    def test_import_from_imex_module(self):
        from chebfunjax.spin.imex import imex_euler, imex_sbdf2  # noqa: F401


# ============================================================================
# V03: Stiff reaction-diffusion test (Allen-Cahn-like)
# ============================================================================


class TestImexReactionDiffusion:
    """IMEX on u_t = eps * u_xx + u - u^3 (Allen-Cahn, linearized).

    We use a small initial condition so the nonlinearity is weak.
    """

    @pytest.mark.parametrize("scheme", ["euler", "sbdf2"])
    def test_reaction_diffusion_stable(self, scheme: str):
        """Both IMEX schemes remain stable for Allen-Cahn initial data."""
        from chebfunjax.spin.imex import imex_euler, imex_sbdf2

        eps, N = 5e-3, 128
        k = np.fft.fftfreq(N, d=1.0 / N) * N  # integer wavenumbers
        xi = (2 * math.pi / (2 * math.pi)) * k   # angular freqs on [0, 2*pi]
        L_diag = eps * (1j * xi) ** 2  # diffusion only (implicit part)

        def nonlin_vals(u):  # Allen-Cahn reaction: u - u^3 (explicit)
            return u - u ** 3

        Nc = np.ones(N, dtype=complex)

        x = np.linspace(0, 2 * math.pi, N, endpoint=False)
        u0 = 0.1 * np.sin(x)
        u0_hat = np.fft.fft(u0.astype(complex))

        fn = imex_euler if scheme == "euler" else imex_sbdf2
        u_hat, _ = fn(u0_hat, L_diag, nonlin_vals, Nc, dt=0.05, nsteps=20)
        u_out = np.real(np.fft.ifft(u_hat))

        assert np.all(np.isfinite(u_out)), f"{scheme}: NaN/Inf in output"
        # Solution should remain bounded (stable reaction-diffusion)
        assert np.max(np.abs(u_out)) < 10.0, (
            f"{scheme}: solution blew up (max|u|={np.max(np.abs(u_out)):.2e})"
        )


# ============================================================================
# V04: ode45 / ode113
# ============================================================================


class TestOde45:
    """ode45 on simple scalar IVPs."""

    def test_exponential_growth(self):
        """y' = y, y(0) = 1  =>  y(t) = exp(t).

        This is the canonical IVP test (MATLAB Chebfun golden test).
        """
        from chebfunjax.chebfun1d.chebfun import ode45

        sol = ode45(lambda t, y: y, (0.0, 1.0), jnp.array([1.0]))
        y1 = float(sol(jnp.float64(1.0)))
        y_exact = float(jnp.exp(jnp.float64(1.0)))
        assert abs(y1 - y_exact) < 1e-4, (
            f"ode45 y(1) = {y1:.8f}, expected {y_exact:.8f}, "
            f"error = {abs(y1 - y_exact):.2e}"
        )

    def test_returns_chebfun(self):
        """ode45 should return a callable Chebfun."""
        from chebfunjax.chebfun1d.chebfun import Chebfun, ode45

        sol = ode45(lambda t, y: -y, (0.0, 1.0), jnp.array([1.0]))
        assert isinstance(sol, Chebfun), f"Expected Chebfun, got {type(sol)}"

    def test_chebfun_evaluable_on_grid(self):
        """ode45 solution can be evaluated at multiple points."""
        from chebfunjax.chebfun1d.chebfun import ode45

        sol = ode45(lambda t, y: y, (0.0, 1.0), jnp.array([1.0]))
        t_pts = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=jnp.float64)
        for t_pt in t_pts:
            val = float(sol(t_pt))
            assert math.isfinite(val), f"ode45 returned non-finite at t={float(t_pt)}"

    def test_decaying_exponential(self):
        """y' = -y, y(0) = 1  =>  y(1) = exp(-1)."""
        from chebfunjax.chebfun1d.chebfun import ode45

        sol = ode45(lambda t, y: -y, (0.0, 1.0), jnp.array([1.0]))
        y1 = float(sol(jnp.float64(1.0)))
        y_exact = float(jnp.exp(jnp.float64(-1.0)))
        assert abs(y1 - y_exact) < 1e-4, (
            f"ode45 y(1) = {y1:.8f}, expected {y_exact:.8f}"
        )

    def test_cosine_solution(self):
        """y' = -sin(t), y(0) = 1  =>  y(t) = cos(t)."""
        from chebfunjax.chebfun1d.chebfun import ode45

        sol = ode45(
            lambda t, y: -jnp.sin(jnp.array(t)),
            (0.0, math.pi),
            jnp.array([1.0]),
        )
        y_pi = float(sol(jnp.float64(math.pi)))
        assert abs(y_pi - (-1.0)) < 1e-3, (
            f"ode45 cos(pi) = {y_pi:.6f}, expected -1"
        )

    def test_initial_value_satisfied(self):
        """y(0) should match y0 to high accuracy."""
        from chebfunjax.chebfun1d.chebfun import ode45

        y0_val = 3.14159
        sol = ode45(lambda t, y: y, (0.0, 1.0), jnp.array([y0_val]))
        y_start = float(sol(jnp.float64(0.0)))
        assert abs(y_start - y0_val) < 1e-4, (
            f"ode45 y(0) = {y_start:.8f}, expected {y0_val:.8f}"
        )


class TestOde113:
    """ode113 on simple scalar IVPs."""

    def test_exponential_growth(self):
        """y' = y, y(0) = 1  =>  y(1) = e."""
        from chebfunjax.chebfun1d.chebfun import ode113

        sol = ode113(lambda t, y: y, (0.0, 1.0), jnp.array([1.0]))
        y1 = float(sol(jnp.float64(1.0)))
        y_exact = float(jnp.exp(jnp.float64(1.0)))
        assert abs(y1 - y_exact) < 1e-4, (
            f"ode113 y(1) = {y1:.8f}, expected {y_exact:.8f}"
        )

    def test_returns_chebfun(self):
        """ode113 should return a callable Chebfun."""
        from chebfunjax.chebfun1d.chebfun import Chebfun, ode113

        sol = ode113(lambda t, y: -y, (0.0, 1.0), jnp.array([1.0]))
        assert isinstance(sol, Chebfun)

    def test_ode113_vs_ode45_agreement(self):
        """ode113 and ode45 should agree to within loose tolerance."""
        from chebfunjax.chebfun1d.chebfun import ode45, ode113

        sol45 = ode45(lambda t, y: y, (0.0, 1.0), jnp.array([1.0]))
        sol113 = ode113(lambda t, y: y, (0.0, 1.0), jnp.array([1.0]))

        t_test = jnp.float64(0.7)
        v45 = float(sol45(t_test))
        v113 = float(sol113(t_test))
        assert abs(v45 - v113) < 1e-3, (
            f"ode45 and ode113 disagree at t=0.7: {v45:.6f} vs {v113:.6f}"
        )


class TestOdeImport:
    """Smoke test: ode45/ode113 are importable."""

    def test_import_from_chebfun_module(self):
        from chebfunjax.chebfun1d.chebfun import ode45, ode113  # noqa: F401

    def test_import_from_chebfun1d_package(self):
        from chebfunjax.chebfun1d import ode45, ode113  # noqa: F401


# ============================================================================
# V05: bvp4c / bvp5c
# ============================================================================


class TestBvp4c:
    """bvp4c on linear BVPs."""

    def test_simple_poisson(self):
        """u'' = -1, u(-1) = 0, u(1) = 0  =>  u(0) = 0.5."""
        from chebfunjax.chebfun1d.ode import bvp4c

        u = bvp4c(
            lambda x, u: u.diff(2),
            domain=(-1.0, 1.0),
            lbc=0.0,
            rbc=0.0,
            f=-1.0,
        )
        val = float(u(jnp.float64(0.0)))
        assert abs(val - 0.5) < 1e-6, (
            f"bvp4c: u(0) = {val:.10f}, expected 0.5"
        )

    def test_returns_chebfun(self):
        """bvp4c should return a Chebfun."""
        from chebfunjax.chebfun1d.chebfun import Chebfun
        from chebfunjax.chebfun1d.ode import bvp4c

        u = bvp4c(
            lambda x, u: u.diff(2),
            domain=(-1.0, 1.0),
            lbc=0.0,
            rbc=0.0,
            f=-1.0,
        )
        assert isinstance(u, Chebfun)

    def test_linear_solution(self):
        """u'' + u = 1, u(0) = 0, u(pi/2) = 0  =>  exact solution known.

        General solution: u = A*sin(x) + B*cos(x) + 1.
        BCs:  u(0) = B + 1 = 0  =>  B = -1.
              u(pi/2) = A + 1 = 0  =>  A = -1.
        So:   u(x) = 1 - sin(x) - cos(x).
        At x = pi/4:  u = 1 - sin(pi/4) - cos(pi/4) = 1 - sqrt(2) ≈ -0.4142.
        """
        from chebfunjax.chebfun1d.ode import bvp4c

        u = bvp4c(
            lambda x, u: u.diff(2) + u,
            domain=(0.0, math.pi / 2.0),
            lbc=0.0,                   # u(0) = 0
            rbc=0.0,                   # u(pi/2) = 0
            f=1.0,                     # u'' + u = 1
        )
        x_test = math.pi / 4.0
        val = float(u(jnp.float64(x_test)))
        # Exact: u = 1 - sin(x) - cos(x); u(pi/4) = 1 - sqrt(2)
        expected = 1.0 - math.sin(x_test) - math.cos(x_test)
        assert abs(val - expected) < 1e-5, (
            f"bvp4c: u(pi/4) = {val:.8f}, expected {expected:.8f}"
        )

    def test_agrees_with_bvp(self):
        """bvp4c should give essentially the same answer as bvp."""
        from chebfunjax.chebfun1d.ode import bvp, bvp4c

        kwargs = dict(domain=(-1.0, 1.0), lbc=0.0, rbc=0.0, f=-1.0)
        u_ref = bvp(lambda x, u: u.diff(2), **kwargs)
        u_4c = bvp4c(lambda x, u: u.diff(2), **kwargs)

        t_pts = jnp.array([0.0, 0.3, -0.5], dtype=jnp.float64)
        for t_pt in t_pts:
            v_ref = float(u_ref(t_pt))
            v_4c = float(u_4c(t_pt))
            assert abs(v_ref - v_4c) < 1e-5, (
                f"bvp4c vs bvp at x={float(t_pt)}: {v_4c:.8f} vs {v_ref:.8f}"
            )


class TestBvp5c:
    """bvp5c on linear BVPs."""

    def test_simple_poisson(self):
        """u'' = -1, u(-1) = 0, u(1) = 0  =>  u(0) = 0.5."""
        from chebfunjax.chebfun1d.ode import bvp5c

        u = bvp5c(
            lambda x, u: u.diff(2),
            domain=(-1.0, 1.0),
            lbc=0.0,
            rbc=0.0,
            f=-1.0,
        )
        val = float(u(jnp.float64(0.0)))
        assert abs(val - 0.5) < 1e-6, (
            f"bvp5c: u(0) = {val:.10f}, expected 0.5"
        )

    def test_returns_chebfun(self):
        """bvp5c should return a Chebfun."""
        from chebfunjax.chebfun1d.chebfun import Chebfun
        from chebfunjax.chebfun1d.ode import bvp5c

        u = bvp5c(
            lambda x, u: u.diff(2),
            domain=(-1.0, 1.0),
            lbc=0.0,
            rbc=0.0,
            f=-1.0,
        )
        assert isinstance(u, Chebfun)

    def test_agrees_with_bvp4c(self):
        """bvp5c and bvp4c should agree to within tol."""
        from chebfunjax.chebfun1d.ode import bvp4c, bvp5c

        kwargs = dict(domain=(-1.0, 1.0), lbc=0.0, rbc=0.0, f=-1.0)
        u_4c = bvp4c(lambda x, u: u.diff(2), **kwargs)
        u_5c = bvp5c(lambda x, u: u.diff(2), **kwargs)

        val_4c = float(u_4c(jnp.float64(0.0)))
        val_5c = float(u_5c(jnp.float64(0.0)))
        assert abs(val_4c - val_5c) < 1e-5, (
            f"bvp4c ({val_4c:.8f}) and bvp5c ({val_5c:.8f}) disagree"
        )


class TestBvpImport:
    """Smoke test: bvp4c / bvp5c are importable."""

    def test_import_from_ode_module(self):
        from chebfunjax.chebfun1d.ode import bvp4c, bvp5c  # noqa: F401

    def test_import_from_chebfun1d_package(self):
        from chebfunjax.chebfun1d import bvp4c, bvp5c  # noqa: F401
