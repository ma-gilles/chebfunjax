"""Tests for chebfunjax.discretization.trigcolloc — TrigColloc class.

Tests include:
- Differentiation matrix accuracy on known trig functions
- Consistency with spectral theory (eigenvalues)
- Points / weights
- Eval matrix
- Periodic BVP solve
"""

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.discretization.trigcolloc import TrigColloc, trig_cumsummat, trig_diffmat
from chebfunjax.domain import Domain

# Tolerances from project.conf
RTOL = 1e-12
ATOL = 1e-13


# ============================================================================
# Tier 1: trig_diffmat (stand-alone helper)
# ============================================================================


class TestTrigDiffmat:
    """Tests for trig_diffmat."""

    @pytest.mark.parametrize("N", [4, 6, 8, 10, 16])
    def test_shape(self, N):
        """Differentiation matrix should be (N, N)."""
        D = trig_diffmat(N)
        assert D.shape == (N, N)

    @pytest.mark.parametrize("N", [6, 8, 12, 16])
    def test_diff_sin_pi_x(self, N):
        """D @ sin(pi*x) ≈ pi*cos(pi*x) on equidistant points on [-1,1)."""
        x = jnp.array([-1.0 + 2.0 * k / N for k in range(N)], dtype=jnp.float64)
        f = jnp.sin(jnp.pi * x)
        D = trig_diffmat(N, m=1)
        Df = D @ f
        exact = jnp.pi * jnp.cos(jnp.pi * x)
        npt.assert_allclose(np.array(Df), np.array(exact), atol=1e-12)

    @pytest.mark.parametrize("N", [6, 8, 12, 16])
    def test_diff_cos_pi_x(self, N):
        """D @ cos(pi*x) ≈ -pi*sin(pi*x)."""
        x = jnp.array([-1.0 + 2.0 * k / N for k in range(N)], dtype=jnp.float64)
        f = jnp.cos(jnp.pi * x)
        D = trig_diffmat(N, m=1)
        Df = D @ f
        exact = -jnp.pi * jnp.sin(jnp.pi * x)
        npt.assert_allclose(np.array(Df), np.array(exact), atol=1e-12)

    @pytest.mark.parametrize("N", [8, 12, 16])
    def test_second_derivative(self, N):
        """D^2 @ sin(pi*x) ≈ -pi^2*sin(pi*x)."""
        x = jnp.array([-1.0 + 2.0 * k / N for k in range(N)], dtype=jnp.float64)
        f = jnp.sin(jnp.pi * x)
        D2 = trig_diffmat(N, m=2)
        D2f = D2 @ f
        exact = -(jnp.pi**2) * jnp.sin(jnp.pi * x)
        npt.assert_allclose(np.array(D2f), np.array(exact), atol=1e-11)

    @pytest.mark.parametrize("N", [8, 12, 16])
    def test_third_derivative(self, N):
        """D^3 @ cos(2*pi*x) ≈ 8*pi^3*sin(2*pi*x)."""
        x = jnp.array([-1.0 + 2.0 * k / N for k in range(N)], dtype=jnp.float64)
        f = jnp.cos(2.0 * jnp.pi * x)
        D3 = trig_diffmat(N, m=3)
        D3f = D3 @ f
        exact = 8.0 * jnp.pi**3 * jnp.sin(2.0 * jnp.pi * x)
        npt.assert_allclose(np.array(D3f), np.array(exact), atol=1e-10)

    @pytest.mark.parametrize("N", [8, 12, 16])
    def test_fourth_derivative(self, N):
        """D^4 @ sin(pi*x) ≈ pi^4*sin(pi*x)."""
        x = jnp.array([-1.0 + 2.0 * k / N for k in range(N)], dtype=jnp.float64)
        f = jnp.sin(jnp.pi * x)
        D4 = trig_diffmat(N, m=4)
        D4f = D4 @ f
        exact = jnp.pi**4 * jnp.sin(jnp.pi * x)
        npt.assert_allclose(np.array(D4f), np.array(exact), atol=1e-9)

    @pytest.mark.parametrize("N", [8, 12, 16])
    def test_higher_order_via_fft(self, N):
        """m=5: D^5 @ cos(pi*x) ≈ pi^5 * cos(pi*x + 5*pi/2) = -pi^5 * sin(pi*x)."""
        x = jnp.array([-1.0 + 2.0 * k / N for k in range(N)], dtype=jnp.float64)
        f = jnp.cos(jnp.pi * x)
        D5 = trig_diffmat(N, m=5)
        D5f = D5 @ f
        # d^5/dx^5 cos(pi*x) = pi^5 * cos(pi*x + 5*pi/2) = -pi^5 * sin(pi*x)
        exact = -jnp.pi**5 * jnp.sin(jnp.pi * x)
        npt.assert_allclose(np.array(D5f), np.array(exact), atol=1e-7)

    def test_identity_m0(self):
        """m=0 returns identity matrix."""
        N = 6
        D = trig_diffmat(N, m=0)
        npt.assert_allclose(np.array(D), np.eye(N), atol=1e-15)

    def test_skew_symmetry_m1_even_N(self):
        """For even N, first-derivative matrix is skew-symmetric."""
        N = 8
        D = np.array(trig_diffmat(N, m=1))
        npt.assert_allclose(D + D.T, np.zeros((N, N)), atol=1e-13)

    def test_constant_function(self):
        """Derivative of a constant should be zero."""
        N = 8
        D = trig_diffmat(N, m=1)
        f = jnp.ones(N, dtype=jnp.float64)
        Df = D @ f
        npt.assert_allclose(np.array(Df), np.zeros(N), atol=1e-13)

    def test_empty_N0(self):
        """N=0 returns empty matrix."""
        D = trig_diffmat(0)
        assert D.shape == (0, 0)

    def test_N1_returns_zero(self):
        """N=1 returns 1x1 zero matrix."""
        D = trig_diffmat(1)
        assert D.shape == (1, 1)
        npt.assert_allclose(float(D[0, 0]), 0.0, atol=1e-15)


# ============================================================================
# Tier 1: trig_cumsummat (stand-alone helper)
# ============================================================================


class TestTrigCumsummat:
    """Tests for trig_cumsummat."""

    @pytest.mark.parametrize("N", [4, 8, 16])
    def test_shape(self, N):
        """cumsummat should be (N, N)."""
        Q = trig_cumsummat(N)
        assert Q.shape == (N, N)

    @pytest.mark.parametrize("N", [8, 12, 16])
    def test_diff_cumsummat_roundtrip(self, N):
        """D @ Q applied to a zero-mean function should recover the function."""
        D = trig_diffmat(N, m=1)
        Q = trig_cumsummat(N)
        # diff(cumsum(f)) should recover f for zero-mean f
        x = jnp.array([-1.0 + 2.0 * k / N for k in range(N)], dtype=jnp.float64)
        # Zero-mean trig function
        f = jnp.sin(jnp.pi * x)
        Qf = Q @ f
        D_Qf = D @ Qf
        npt.assert_allclose(np.array(D_Qf), np.array(f), atol=1e-12)

    @pytest.mark.parametrize("N", [8, 12, 16])
    def test_integral_sin(self, N):
        """Q @ sin(pi*x) ≈ -cos(pi*x)/pi + const on equidistant points."""
        x = jnp.array([-1.0 + 2.0 * k / N for k in range(N)], dtype=jnp.float64)
        f = jnp.sin(jnp.pi * x)
        Q = trig_cumsummat(N)
        Qf = Q @ f
        # The antiderivative of sin(pi*x) is -cos(pi*x)/pi
        # The DC component is zeroed: result = -cos(pi*x)/pi - mean(-cos/pi)
        # mean of -cos(pi*x) over equidistant points on [-1,1) is 0
        exact = -jnp.cos(jnp.pi * x) / jnp.pi
        # Up to an additive constant (DC zeroed)
        diff = np.array(Qf) - np.array(exact)
        npt.assert_allclose(diff - diff.mean(), np.zeros(N), atol=1e-12)


# ============================================================================
# Tier 1: TrigColloc class
# ============================================================================


class TestTrigColloc:
    """Tests for TrigColloc class methods."""

    def test_construction(self):
        """TrigColloc(n=8) should construct without error."""
        disc = TrigColloc(n=8)
        assert disc.n == 8

    def test_construction_with_domain(self):
        """TrigColloc with a custom domain."""
        disc = TrigColloc(n=16, domain=Domain((0.0, 2.0)))
        assert disc.n == 16

    def test_construction_invalid_n(self):
        """n < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n >= 1"):
            TrigColloc(n=0)

    def test_repr(self):
        """repr should be a non-empty string."""
        disc = TrigColloc(n=8)
        s = repr(disc)
        assert "TrigColloc" in s
        assert "n=8" in s

    # ------------------------------------------------------------------
    # points()
    # ------------------------------------------------------------------

    def test_points_default_domain(self):
        """Points on [-1,1) should be -1 + 2k/N for k=0,...,N-1."""
        N = 8
        disc = TrigColloc(n=N)
        x = disc.points()
        expected = jnp.array([-1.0 + 2.0 * k / N for k in range(N)], dtype=jnp.float64)
        npt.assert_allclose(np.array(x), np.array(expected), atol=1e-15)

    def test_points_custom_domain(self):
        """Points on [0, 2) should be 2k/N for k=0,...,N-1."""
        N = 8
        disc = TrigColloc(n=N, domain=Domain((0.0, 2.0)))
        x = disc.points()
        expected = jnp.array([2.0 * k / N for k in range(N)], dtype=jnp.float64)
        npt.assert_allclose(np.array(x), np.array(expected), atol=1e-15)

    def test_equation_points_equal_function_points(self):
        """equation_points should equal points for TrigColloc."""
        disc = TrigColloc(n=12)
        npt.assert_allclose(
            np.array(disc.equation_points()),
            np.array(disc.points()),
            atol=1e-15,
        )

    # ------------------------------------------------------------------
    # weights()
    # ------------------------------------------------------------------

    def test_weights_sum_to_period(self):
        """Quadrature weights should sum to the period length."""
        N = 16
        disc = TrigColloc(n=N)
        w = disc.weights()
        npt.assert_allclose(float(jnp.sum(w)), 2.0, atol=1e-14)  # period = 2

    def test_weights_uniform(self):
        """All weights should be equal (= period/N) for trapezoidal rule."""
        N = 16
        disc = TrigColloc(n=N)
        w = disc.weights()
        npt.assert_allclose(np.array(w), np.full(N, 2.0 / N), atol=1e-15)

    def test_weights_quadrature_constant(self):
        """Trapezoidal rule integrates a constant function exactly."""
        N = 8
        disc = TrigColloc(n=N)
        w = disc.weights()
        f_vals = 3.0 * jnp.ones(N, dtype=jnp.float64)
        npt.assert_allclose(float(jnp.dot(w, f_vals)), 6.0, atol=1e-14)

    def test_weights_quadrature_sin(self):
        """Trapezoidal rule should integrate sin(pi*x) over [-1,1) exactly to 0."""
        N = 8
        disc = TrigColloc(n=N)
        x = disc.points()
        w = disc.weights()
        f_vals = jnp.sin(jnp.pi * x)
        npt.assert_allclose(float(jnp.dot(w, f_vals)), 0.0, atol=1e-14)

    # ------------------------------------------------------------------
    # diffmat()
    # ------------------------------------------------------------------

    def test_diffmat_shape(self):
        """Differentiation matrix should be (N, N)."""
        N = 10
        disc = TrigColloc(n=N)
        D = disc.diffmat()
        assert D.shape == (N, N)

    def test_diffmat_sin_pi_x(self):
        """disc.diffmat(k=1) @ sin(pi*x) ≈ pi*cos(pi*x) on default domain."""
        N = 16
        disc = TrigColloc(n=N)
        x = disc.points()
        f = jnp.sin(jnp.pi * x)
        D = disc.diffmat(k=1)
        Df = D @ f
        exact = jnp.pi * jnp.cos(jnp.pi * x)
        npt.assert_allclose(np.array(Df), np.array(exact), atol=1e-12)

    def test_diffmat_scaling_custom_domain(self):
        """Differentiation matrix on [0, 2) (period 2) should not require extra scaling."""
        N = 16
        disc = TrigColloc(n=N, domain=Domain((0.0, 2.0)))
        x = disc.points()
        f = jnp.sin(jnp.pi * x)  # same as sin(pi*x) on [0,2)
        D = disc.diffmat(k=1)
        Df = D @ f
        exact = jnp.pi * jnp.cos(jnp.pi * x)
        npt.assert_allclose(np.array(Df), np.array(exact), atol=1e-12)

    def test_diffmat_k0_is_identity(self):
        """k=0 should return the identity matrix."""
        N = 8
        disc = TrigColloc(n=N)
        D = disc.diffmat(k=0)
        npt.assert_allclose(np.array(D), np.eye(N), atol=1e-15)

    # ------------------------------------------------------------------
    # cumsummat()
    # ------------------------------------------------------------------

    def test_cumsummat_shape(self):
        """Integration matrix should be (N, N)."""
        N = 8
        disc = TrigColloc(n=N)
        Q = disc.cumsummat()
        assert Q.shape == (N, N)

    def test_cumsummat_diff_roundtrip(self):
        """D @ Q applied to sin(pi*x) recovers sin(pi*x) on default domain."""
        N = 16
        disc = TrigColloc(n=N)
        x = disc.points()
        f = jnp.sin(jnp.pi * x)
        Q = disc.cumsummat()
        D = disc.diffmat(k=1)
        DQf = D @ (Q @ f)
        npt.assert_allclose(np.array(DQf), np.array(f), atol=1e-11)

    # ------------------------------------------------------------------
    # eval_matrix()
    # ------------------------------------------------------------------

    def test_eval_matrix_at_grid_points(self):
        """eval_matrix at grid points should recover identity."""
        N = 8
        disc = TrigColloc(n=N)
        x = disc.points()
        E = disc.eval_matrix(x)
        # E @ f_values at grid should equal f_values (to machine precision)
        f = jnp.sin(jnp.pi * x)
        Ef = E @ f
        npt.assert_allclose(np.array(Ef), np.array(f), atol=1e-12)

    def test_eval_matrix_interpolation(self):
        """eval_matrix at off-grid points should interpolate trigonometrically."""
        N = 16
        disc = TrigColloc(n=N)
        x = disc.points()
        f = jnp.sin(jnp.pi * x)
        # Evaluate at some off-grid points
        y = jnp.array([-0.7, -0.3, 0.1, 0.4, 0.8], dtype=jnp.float64)
        E = disc.eval_matrix(y)
        Ef = E @ f
        exact = jnp.sin(jnp.pi * y)
        npt.assert_allclose(np.array(Ef), np.array(exact), atol=1e-13)


# ============================================================================
# Tier 1: Periodic BVP solve with TrigColloc
# ============================================================================


class TestTrigCollocBVP:
    """Solve simple periodic BVPs using TrigColloc."""

    def test_periodic_bvp_u_plus_u_pp_equals_zero(self):
        """Solve u'' + (2pi)^2 u = 0; verify cos(2pi*x) is in the kernel."""
        N = 16
        disc = TrigColloc(n=N)
        x = disc.points()
        D2 = disc.diffmat(k=2)

        # The operator L u = u'' + (2pi)^2 u
        omega = 2.0 * jnp.pi
        L = D2 + omega**2 * jnp.eye(N, dtype=jnp.float64)

        u = jnp.cos(omega * x)
        Lu = L @ u
        npt.assert_allclose(np.array(Lu), np.zeros(N), atol=1e-10)

    def test_periodic_bvp_sin_in_kernel(self):
        """sin(2pi*x) should also be in the kernel of u'' + (2pi)^2 u."""
        N = 16
        disc = TrigColloc(n=N)
        x = disc.points()
        D2 = disc.diffmat(k=2)
        omega = 2.0 * jnp.pi
        L = D2 + omega**2 * jnp.eye(N, dtype=jnp.float64)

        u = jnp.sin(omega * x)
        Lu = L @ u
        npt.assert_allclose(np.array(Lu), np.zeros(N), atol=1e-10)

    def test_periodic_forced_problem(self):
        """Verify u = cos(k*pi*x) satisfies u'' + alpha*u = f for known f.

        With u = cos(k*pi*x), u'' = -(k*pi)^2 * u.
        Choose alpha = (k*pi)^2 + 1 so that L u = u'' + alpha*u = u.
        Verify that L @ u = u on the discrete level.
        """
        N = 32
        k = 2
        disc = TrigColloc(n=N)
        x = disc.points()
        D2 = disc.diffmat(k=2)

        # u = cos(k*pi*x), u'' = -(k*pi)^2 * u
        # L = D2 + (k*pi)^2 + 1  =>  L @ u = u
        alpha = (k * jnp.pi) ** 2 + 1.0
        L = D2 + alpha * jnp.eye(N, dtype=jnp.float64)
        u_exact = jnp.cos(k * jnp.pi * x)
        f = u_exact  # L @ u_exact should equal u_exact
        Lu = L @ u_exact
        npt.assert_allclose(np.array(Lu), np.array(f), atol=1e-10)

    def test_custom_domain_periodic_bvp(self):
        """Periodic BVP on [0, 1)."""
        N = 16
        disc = TrigColloc(n=N, domain=Domain((0.0, 1.0)))
        x = disc.points()
        D2 = disc.diffmat(k=2)

        # u = sin(2*pi*x), u'' = -(2*pi)^2 * sin(2*pi*x)
        u = jnp.sin(2 * jnp.pi * x)
        D2u = D2 @ u
        exact = -(2 * jnp.pi) ** 2 * jnp.sin(2 * jnp.pi * x)
        npt.assert_allclose(np.array(D2u), np.array(exact), atol=1e-10)
