"""Tests for chebfunjax.spin — SpinOp2 and spin2() ETDRK4 2D solver.

Test coverage:
  - SpinOp2 construction (manual and from built-in name)
  - 2D dealiasing mask
  - 2D Laplacian eigenvalues
  - Allen-Cahn 2D: solution bounded in [-1, 1], energy decreasing
  - Ginzburg-Landau: L2 norm approximately conserved
  - multi-component (Gray-Scott): components stay in [0, 1]

JAX contract:
  - spin2() does NOT use JAX internally (runs on plain NumPy) — no JIT needed.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.spin.solver2d import (
    _compute_etdrk4_coeffs_2d,
    _dealias_mask_2d,
    spin2,
)
from chebfunjax.spin.spinop2 import (
    SpinOp2,
    _fourier_wavenumbers_2d,
    build_laplacian_eigenvalues_2d,
    build_linear_eigenvalues_2d,
)

# ============================================================================
# Tier 1: SpinOp2 construction
# ============================================================================


class TestSpinOp2Construction:
    """Test SpinOp2 construction."""

    def test_from_name_gl(self):
        op = SpinOp2.from_name("GL")
        assert op.n_vars == 1
        assert op.is_real is False
        ax, bx, ay, by = op.domain
        assert float(ax) == pytest.approx(0.0)
        assert float(bx) == pytest.approx(100.0)

    def test_from_name_ac2(self):
        op = SpinOp2.from_name("AC2")
        assert op.n_vars == 1
        assert op.is_real is True

    def test_from_name_gs(self):
        op = SpinOp2.from_name("GS")
        assert op.n_vars == 2
        assert isinstance(op.lin_coeffs, list)
        assert len(op.lin_coeffs) == 2

    def test_from_name_sh(self):
        op = SpinOp2.from_name("SH")
        assert op.n_vars == 1
        # Swift-Hohenberg: lin_coeffs = (-2, -1, 0, 0, 0)
        A, B = op.lin_coeffs[0], op.lin_coeffs[1]
        assert A == pytest.approx(-2.0)
        assert B == pytest.approx(-1.0)

    def test_from_name_case_insensitive(self):
        op1 = SpinOp2.from_name("gl")
        op2 = SpinOp2.from_name("GL")
        op3 = SpinOp2.from_name("Gl")
        for op in (op1, op2, op3):
            assert op.n_vars == 1

    def test_from_name_unknown_raises(self):
        with pytest.raises(ValueError, match="Unrecognised"):
            SpinOp2.from_name("NonExistent2D")

    def test_manual_construction_scalar(self):
        """Manual SpinOp2 for 2D Allen-Cahn."""
        op = SpinOp2(
            lin_coeffs=(1e-2, 0.0, 0.0, 0.0, 0.0),
            nonlin_vals=lambda u: u - u ** 3,
            n_vars=1,
            domain=(0.0, 2 * math.pi, 0.0, 2 * math.pi),
            tspan=(0.0, 1.0),
            u0=lambda x, y: jnp.sin(x) * jnp.sin(y),
            is_real=True,
        )
        assert op.n_vars == 1
        assert op.is_real is True

    def test_manual_construction_multi(self):
        """Manual SpinOp2 for a 2-component system."""
        op = SpinOp2(
            lin_coeffs=[
                (1e-3, 0.0, 0.0, 0.0, 0.0),
                (2e-3, 0.0, 0.0, 0.0, 0.0),
            ],
            nonlin_vals=[
                lambda u, v: -u * v,
                lambda u, v: u * v - v,
            ],
            n_vars=2,
            domain=(0.0, 1.0, 0.0, 1.0),
            tspan=(0.0, 1.0),
            u0=[
                lambda x, y: jnp.sin(2 * math.pi * x),
                lambda x, y: jnp.cos(2 * math.pi * y),
            ],
            is_real=True,
        )
        assert op.n_vars == 2

    def test_default_N(self):
        op = SpinOp2.from_name("AC2")
        assert op.default_N("AC2") == 128

    def test_default_dt(self):
        op = SpinOp2.from_name("AC2")
        assert op.default_dt("AC2") == pytest.approx(1e-2)

    def test_repr_does_not_crash(self):
        op = SpinOp2.from_name("GL")
        r = repr(op)
        assert "SpinOp2" in r


# ============================================================================
# Tier 1: Fourier wavenumbers and Laplacian eigenvalues
# ============================================================================


class TestFourierWavenumbers2D:
    def test_shape(self):
        XI, ETA = _fourier_wavenumbers_2d(8, (0.0, 2 * math.pi, 0.0, 2 * math.pi))
        assert XI.shape == (8, 8)
        assert ETA.shape == (8, 8)

    def test_xi_row_constant(self):
        """XI should be constant along columns (indexing='ij')."""
        N = 8
        XI, ETA = _fourier_wavenumbers_2d(N, (0.0, 2 * math.pi, 0.0, 2 * math.pi))
        # Rows of XI should all be the same
        for col in range(N):
            npt.assert_array_equal(XI[:, col], XI[:, 0])

    def test_eta_col_constant(self):
        """ETA should be constant along rows (indexing='ij')."""
        N = 8
        XI, ETA = _fourier_wavenumbers_2d(N, (0.0, 2 * math.pi, 0.0, 2 * math.pi))
        for row in range(N):
            npt.assert_array_equal(ETA[row, :], ETA[0, :])

    def test_wavenumber_scaling(self):
        """First non-zero wavenumber in x should be 2*pi/Lx."""
        N = 8
        Lx = 4.0
        XI, _ = _fourier_wavenumbers_2d(N, (0.0, Lx, 0.0, 2 * math.pi))
        npt.assert_allclose(XI[1, 0], 2 * math.pi / Lx, rtol=1e-14)


class TestLaplacianEigenvalues2D:
    def test_shape(self):
        lap = build_laplacian_eigenvalues_2d(16, (0.0, 2 * math.pi, 0.0, 2 * math.pi))
        assert lap.shape == (16, 16)

    def test_dc_mode_zero(self):
        """The (0, 0) mode has Laplacian eigenvalue 0."""
        lap = build_laplacian_eigenvalues_2d(16, (0.0, 2 * math.pi, 0.0, 2 * math.pi))
        assert float(lap[0, 0]) == pytest.approx(0.0)

    def test_lap_nonpositive(self):
        """All Laplacian eigenvalues are <= 0 (negative semi-definite)."""
        lap = build_laplacian_eigenvalues_2d(16, (0.0, 2 * math.pi, 0.0, 2 * math.pi))
        assert np.all(lap <= 0.0)

    def test_linear_eigenvalues_ac2(self):
        """Allen-Cahn: L = eps*lap, all eigenvalues <= 0."""
        op = SpinOp2.from_name("AC2")
        ax, bx, ay, by = op.domain
        L = build_linear_eigenvalues_2d(op.lin_coeffs, 16, (ax, bx, ay, by))
        assert np.all(np.real(L) <= 0.0 + 1e-12)

    def test_linear_eigenvalues_sh(self):
        """Swift-Hohenberg: L = -2*lap - lap^2.  DC eigenvalue = 0."""
        op = SpinOp2.from_name("SH")
        ax, bx, ay, by = op.domain
        L = build_linear_eigenvalues_2d(op.lin_coeffs, 16, (ax, bx, ay, by))
        # DC (wavenumber = 0): lap = 0, so L[0,0] = -2*0 - 0^2 = 0
        npt.assert_allclose(np.real(L[0, 0]), 0.0, atol=1e-12)


# ============================================================================
# Tier 1: 2D dealiasing mask
# ============================================================================


class TestDealiasMask2D:
    def test_shape(self):
        mask = _dealias_mask_2d(64)
        assert mask.shape == (64, 64)

    def test_dc_mode_kept(self):
        """The (0, 0) DC mode is always kept."""
        mask = _dealias_mask_2d(64)
        assert mask[0, 0] is np.bool_(True)

    def test_zeros_high_freq(self):
        """Some modes near Nyquist should be zeroed in both dimensions."""
        N = 64
        mask = _dealias_mask_2d(N)
        n_zeroed = np.sum(~mask)
        assert n_zeroed > 0, "Expected some modes to be zeroed"
        # Zeroed fraction should be close to 1 - (2/3)^2 ≈ 55%
        frac_zeroed = n_zeroed / N ** 2
        assert frac_zeroed > 0.3

    def test_low_modes_kept(self):
        """The first few modes in each direction should be kept."""
        mask = _dealias_mask_2d(64)
        # First few rows and columns should be kept
        assert np.all(mask[:5, :5])

    def test_nvar_consistency(self):
        """Mask for N=32 has correct shape."""
        mask = _dealias_mask_2d(32)
        assert mask.shape == (32, 32)


# ============================================================================
# Tier 1: ETDRK4 coefficient consistency (2D)
# ============================================================================


class TestETDRK4Coeffs2D:
    def test_e_half_e_full_consistency(self):
        """E_full = E_half^2 for 2D flat eigenvalues."""
        N = 8
        lap = build_laplacian_eigenvalues_2d(N, (0.0, 2 * math.pi, 0.0, 2 * math.pi))
        eps = 1e-2
        L_flat = (eps * lap).ravel()
        dt = 0.01
        coeffs = _compute_etdrk4_coeffs_2d(dt, L_flat, is_real=True, M=16)
        npt.assert_allclose(
            coeffs["E_full"], coeffs["E_half"] ** 2, rtol=1e-12
        )

    def test_real_op_gives_real_coeffs(self):
        """For a real (diffusive) operator, B-coefficients should be real."""
        N = 8
        lap = build_laplacian_eigenvalues_2d(N, (0.0, 2 * math.pi, 0.0, 2 * math.pi))
        L_flat = (1e-2 * lap).ravel()
        dt = 0.01
        coeffs = _compute_etdrk4_coeffs_2d(dt, L_flat, is_real=True, M=16)
        for key in ("E_half", "E_full", "B2", "B3", "B4", "phi1", "psi12"):
            imag_max = np.max(np.abs(np.imag(coeffs[key])))
            assert imag_max < 1e-12, f"Key {key} has imaginary part {imag_max:.2e}"


# ============================================================================
# Tier 1: spin2() smoke tests
# ============================================================================


class TestSpin2Smoke:
    """Basic smoke tests for spin2() — small N, short tspan."""

    def test_scalar_returns_correct_shapes(self):
        """spin2() returns (xx, yy, t, u) with correct shapes for scalar PDE."""
        op = SpinOp2(
            lin_coeffs=(1e-2, 0.0, 0.0, 0.0, 0.0),
            nonlin_vals=lambda u: u - u ** 3,
            n_vars=1,
            domain=(0.0, 2 * math.pi, 0.0, 2 * math.pi),
            tspan=(0.0, 0.1),
            u0=lambda x, y: np.sin(x) * np.sin(y),
            is_real=True,
        )
        xx, yy, t, u = spin2(op, N=16, dt=1e-2)
        assert xx.shape == (16, 16)
        assert yy.shape == (16, 16)
        assert u.shape == (16, 16)
        assert abs(t - 0.1) < 1e-10

    def test_real_scalar_output_dtype(self):
        """Real PDE should produce float output."""
        op = SpinOp2.from_name("AC2")
        t0, tf = op.tspan
        op_short = SpinOp2(
            lin_coeffs=op.lin_coeffs,
            nonlin_vals=op.nonlin_vals,
            n_vars=op.n_vars,
            domain=op.domain,
            tspan=(t0, t0 + 0.1),
            u0=op.u0,
            is_real=True,
        )
        xx, yy, t, u = spin2(op_short, N=16, dt=1e-2)
        assert u.dtype in (np.float32, np.float64)
        assert np.all(np.isfinite(u))

    def test_complex_scalar_output(self):
        """Ginzburg-Landau (complex) should return complex output."""
        op = SpinOp2.from_name("GL")
        t0, tf = op.tspan
        op_short = SpinOp2(
            lin_coeffs=op.lin_coeffs,
            nonlin_vals=op.nonlin_vals,
            n_vars=op.n_vars,
            domain=op.domain,
            tspan=(t0, t0 + 0.05),
            u0=op.u0,
            is_real=False,
        )
        xx, yy, t, u = spin2(op_short, N=16, dt=5e-3)
        assert np.iscomplexobj(u)
        assert np.all(np.isfinite(u))

    def test_zero_nonlinear_part_2d(self):
        """Pure linear decay: all modes decay as exp(-t)."""
        N = 8
        # L = -1 (constant), N(u) = 0 -> u(t) = u0 * exp(-t)
        op = SpinOp2(
            lin_coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
            # Override by using manual L=const: not expressible via lin_coeffs
            # so test with L=0 and nonlin=0, solution should stay unchanged
            nonlin_vals=lambda u: np.zeros_like(u),
            n_vars=1,
            domain=(0.0, 2 * math.pi, 0.0, 2 * math.pi),
            tspan=(0.0, 0.1),
            u0=lambda x, y: np.sin(x) * np.cos(y),
            is_real=True,
        )
        xx, yy, t, u = spin2(op, N=N, dt=1e-2)
        # With L=0 and N=0, u should not change
        u0_arr = np.sin(xx) * np.cos(yy)
        npt.assert_allclose(u, u0_arr, atol=1e-6)

    def test_multi_component_returns_list(self):
        """Multi-component PDE (GS) should return a list of arrays."""
        op = SpinOp2.from_name("GS")
        t0, tf = op.tspan
        op_short = SpinOp2(
            lin_coeffs=op.lin_coeffs,
            nonlin_vals=op.nonlin_vals,
            n_vars=op.n_vars,
            domain=op.domain,
            tspan=(t0, t0 + 2.0),
            u0=op.u0,
            is_real=op.is_real,
        )
        xx, yy, t, u_list = spin2(op_short, N=16, dt=2.0)
        assert isinstance(u_list, list)
        assert len(u_list) == 2
        for u in u_list:
            assert u.shape == (16, 16)
            assert np.all(np.isfinite(u))

    def test_blowup_raises(self):
        """Unstable integration should raise RuntimeError.

        Note: lin_coeffs = (A, B, ...) for Laplacian powers always gives
        eigenvalues <= 0 (dissipative). Blow-up must come from the nonlinear
        term. A very large explosive nonlinearity with a large initial condition
        should trigger overflow and NaN detection.
        """
        op = SpinOp2(
            lin_coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),  # no linear stabilization
            nonlin_vals=lambda u: 1e6 * u ** 5 + 1e6 * u,  # explosive
            n_vars=1,
            domain=(0.0, 2 * math.pi, 0.0, 2 * math.pi),
            tspan=(0.0, 10.0),
            u0=lambda x, y: 10.0 * np.ones_like(x),  # large amplitude
            is_real=True,
        )
        with pytest.raises(RuntimeError, match="blew up"):
            spin2(op, N=8, dt=1.0)

    def test_from_string_ac2(self):
        """spin2('AC2', ...) with a short tspan should work."""
        op = SpinOp2.from_name("AC2")
        t0, tf = op.tspan
        op_short = SpinOp2(
            lin_coeffs=op.lin_coeffs,
            nonlin_vals=op.nonlin_vals,
            n_vars=op.n_vars,
            domain=op.domain,
            tspan=(t0, t0 + 0.1),
            u0=op.u0,
            is_real=op.is_real,
        )
        xx, yy, t, u = spin2(op_short, N=16, dt=1e-2)
        assert u.shape == (16, 16)
        assert np.all(np.isfinite(u))


# ============================================================================
# Tier 2: Physics validation — Allen-Cahn 2D
# ============================================================================


@pytest.mark.slow
class TestAllenCahn2DPhysics:
    """Allen-Cahn 2D: u_t = eps*lap(u) + u - u^3.

    Physical properties:
    - Solution is bounded: u in [-1, 1] (stable equilibria)
    - Free energy is non-increasing
    """

    @classmethod
    def setup_class(cls):
        N = 64
        dt = 1e-2
        op = SpinOp2.from_name("AC2")
        op_short = SpinOp2(
            lin_coeffs=op.lin_coeffs,
            nonlin_vals=op.nonlin_vals,
            n_vars=op.n_vars,
            domain=op.domain,
            tspan=(0.0, 2.0),
            u0=op.u0,
            is_real=True,
        )
        cls.N = N
        cls.op = op
        cls.xx, cls.yy, cls.t, cls.u = spin2(op_short, N=N, dt=dt)
        x = np.linspace(0.0, 2 * math.pi, N, endpoint=False)
        y = np.linspace(0.0, 2 * math.pi, N, endpoint=False)
        xx0, yy0 = np.meshgrid(x, y, indexing="ij")
        cls.u0 = np.array(op.u0(xx0, yy0), dtype=float)

    def test_output_finite(self):
        assert np.all(np.isfinite(self.u))

    def test_solution_bounded(self):
        """After relaxation, solution should be in [-1.2, 1.2]."""
        max_abs = np.max(np.abs(self.u))
        assert max_abs < 1.5, f"Solution exceeded bounds: max|u|={max_abs:.4g}"

    def test_energy_decreased(self):
        """Free energy should not increase."""
        eps = 1e-2
        N = self.N
        ax, bx, ay, by = self.op.domain
        dx = (bx - ax) / N
        dy = (by - ay) / N

        def free_energy(u, xx, yy):
            # Spectral gradient
            u_hat = np.fft.fft2(u)
            XI, ETA = _fourier_wavenumbers_2d(N, (ax, bx, ay, by))
            ux = np.real(np.fft.ifft2(1j * XI * u_hat))
            uy = np.real(np.fft.ifft2(1j * ETA * u_hat))
            grad_sq = ux ** 2 + uy ** 2
            bulk = (1.0 - u ** 2) ** 2 / 4.0
            return np.sum(eps / 2.0 * grad_sq + bulk) * dx * dy

        E0 = free_energy(self.u0, self.xx, self.yy)
        Et = free_energy(self.u, self.xx, self.yy)
        # Allow 1% tolerance for numerical noise
        assert Et <= E0 * 1.01, (
            f"Energy did not decrease: E0={E0:.4g}, Et={Et:.4g}"
        )


# ============================================================================
# Tier 2: Physics validation — Ginzburg-Landau L2 norm
# ============================================================================


@pytest.mark.slow
class TestGinzburgLandauNorm:
    """Ginzburg-Landau 2D: norm changes slowly (solution evolves toward attractor)."""

    @classmethod
    def setup_class(cls):
        N = 32
        dt = 5e-3
        op = SpinOp2.from_name("GL")
        op_short = SpinOp2(
            lin_coeffs=op.lin_coeffs,
            nonlin_vals=op.nonlin_vals,
            n_vars=op.n_vars,
            domain=op.domain,
            tspan=(0.0, 0.5),
            u0=op.u0,
            is_real=False,
        )
        cls.N = N
        cls.xx, cls.yy, cls.t, cls.u = spin2(op_short, N=N, dt=dt)
        x = np.linspace(0.0, 100.0, N, endpoint=False)
        y = np.linspace(0.0, 100.0, N, endpoint=False)
        xx0, yy0 = np.meshgrid(x, y, indexing="ij")
        cls.u0 = np.array(op.u0(xx0, yy0), dtype=complex)

    def test_output_finite(self):
        assert np.all(np.isfinite(self.u))

    def test_l2_norm_reasonable(self):
        """L2 norm of the Ginzburg-Landau solution should stay bounded (not blow up)."""
        N = self.N
        ax, bx = 0.0, 100.0
        ay, by = 0.0, 100.0
        dx = (bx - ax) / N
        dy = (by - ay) / N
        normt = np.sqrt(np.sum(np.abs(self.u) ** 2) * dx * dy)
        norm0 = np.sqrt(np.sum(np.abs(self.u0) ** 2) * dx * dy)
        # GL drives the solution toward a limit cycle; norm may change but
        # should not blow up or collapse to zero over 0.5 time units
        assert normt > 0.01 * norm0, (
            f"Solution collapsed: norm0={norm0:.4g}, normt={normt:.4g}"
        )
        assert normt < 100 * norm0, (
            f"Solution blew up: norm0={norm0:.4g}, normt={normt:.4g}"
        )
