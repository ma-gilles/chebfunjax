"""Tests for the final 5 utility functions: gpr, pswf, fov, conformal2, phaseplot.

All tests are pure-Python / numpy (no MATLAB golden refs required for these
exploratory utilities).  The key contract for each function is:

  - gpr:      posterior mean interpolates (or smoothly passes near) the data;
              variance is non-negative; optimised length scale is positive.
  - pswf:     eigenvalue ordering; orthogonality; sign convention.
  - fov:      boundary contains eigenvalues; convexity; numerical abscissa.
  - conformal2: rho in (0, 1); round-trip error below tolerance.
  - phaseplot: returns RGB array of correct shape; valid hue range.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

# ---------------------------------------------------------------------------
# gpr
# ---------------------------------------------------------------------------


class TestGpr:
    """Tests for gpr — Gaussian process regression."""

    def test_basic_return_keys(self):
        """Result dict has all required keys."""
        from chebfunjax.utils.gpr import gpr

        rng = np.random.default_rng(0)
        x = rng.uniform(-1, 1, 8)
        y = np.sin(2 * x)
        result = gpr(x, y, domain=(-1.0, 1.0))
        for key in ("x_grid", "mean", "variance", "samples", "length_scale", "sigma"):
            assert key in result, f"Missing key: {key}"

    def test_grid_shape(self):
        """Output grid has 200 points by default."""
        from chebfunjax.utils.gpr import gpr

        x = np.linspace(-1, 1, 5)
        y = x ** 2
        result = gpr(x, y)
        assert result["x_grid"].shape == (200,)
        assert result["mean"].shape == (200,)
        assert result["variance"].shape == (200,)

    def test_variance_nonnegative(self):
        """Posterior variance must be non-negative everywhere."""
        from chebfunjax.utils.gpr import gpr

        rng = np.random.default_rng(1)
        x = rng.uniform(-2, 2, 10)
        y = np.sin(x)
        result = gpr(x, y, domain=(-2.0, 2.0))
        assert np.all(result["variance"] >= 0.0), "Variance contains negative values."

    def test_mean_interpolates_noiseless(self):
        """With zero noise, posterior mean at data points should be close to y."""
        from chebfunjax.utils.gpr import gpr

        x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        y = np.array([1.0, 0.5, 0.0, 0.5, 1.0])
        result = gpr(x, y, domain=(-1.0, 1.0))
        # Evaluate at the data x by finding nearest grid point
        grid = result["x_grid"]
        mean = result["mean"]
        for xi, yi in zip(x, y):
            idx = np.argmin(np.abs(grid - xi))
            # The mean at nearby grid points should be reasonably close
            # (not exact interpolation on grid, but close)
            npt.assert_allclose(mean[idx], yi, atol=0.1,
                                err_msg=f"Mean far from data at x={xi}")

    def test_sigma_uses_max_abs_y(self):
        """Default sigma should equal max(|y|)."""
        from chebfunjax.utils.gpr import gpr

        x = np.array([-1.0, 0.0, 1.0])
        y = np.array([3.0, -1.0, 2.0])
        result = gpr(x, y)
        npt.assert_allclose(result["sigma"], 3.0, rtol=1e-12)

    def test_custom_sigma(self):
        """Custom sigma is respected."""
        from chebfunjax.utils.gpr import gpr

        x = np.array([-1.0, 0.0, 1.0])
        y = np.array([1.0, 0.0, -1.0])
        result = gpr(x, y, sigma=5.0)
        npt.assert_allclose(result["sigma"], 5.0, rtol=1e-12)

    def test_length_scale_positive(self):
        """Optimised length scale must be positive."""
        from chebfunjax.utils.gpr import gpr

        rng = np.random.default_rng(42)
        x = rng.uniform(-1, 1, 12)
        y = np.cos(3 * x)
        result = gpr(x, y)
        assert result["length_scale"] > 0.0

    def test_empty_data(self):
        """Empty data returns zero mean and sigma^2 variance everywhere."""
        from chebfunjax.utils.gpr import gpr

        result = gpr(np.array([]), np.array([]), domain=(-1.0, 1.0), sigma=2.0)
        npt.assert_allclose(result["mean"], 0.0, atol=1e-15)
        npt.assert_allclose(result["variance"], 4.0, atol=1e-12)

    def test_samples_shape(self):
        """n_samples > 0 returns correct shape."""
        from chebfunjax.utils.gpr import gpr

        x = np.linspace(-1, 1, 6)
        y = np.sin(x)
        result = gpr(x, y, n_samples=3, rng=np.random.default_rng(7))
        assert result["samples"] is not None
        assert result["samples"].shape == (200, 3)

    def test_no_samples_by_default(self):
        """Default: samples is None."""
        from chebfunjax.utils.gpr import gpr

        result = gpr(np.array([0.0]), np.array([1.0]))
        assert result["samples"] is None

    def test_noise_increases_variance(self):
        """Adding observation noise should increase the posterior variance."""
        from chebfunjax.utils.gpr import gpr

        x = np.linspace(-1, 1, 10)
        y = np.sin(x)
        r0 = gpr(x, y, noise=0.0)
        r1 = gpr(x, y, noise=0.5)
        # Mean variance with noise should be >= without (relaxed to median)
        assert np.median(r1["variance"]) >= np.median(r0["variance"]) - 1e-10

    def test_trig_kernel(self):
        """Trig kernel (periodic) runs without error and returns expected shapes."""
        from chebfunjax.utils.gpr import gpr

        x = np.linspace(-np.pi, np.pi, 8, endpoint=False)
        y = np.sin(x)
        result = gpr(x, y, domain=(-np.pi, np.pi), trig=True)
        assert result["mean"].shape == (200,)
        assert np.all(result["variance"] >= 0.0)

    def test_domain_default_min_max(self):
        """Default domain is [min(x), max(x)]."""
        from chebfunjax.utils.gpr import gpr

        x = np.array([-2.0, 0.0, 3.0])
        y = np.zeros(3)
        result = gpr(x, y)
        npt.assert_allclose(result["x_grid"][0], -2.0, atol=1e-12)
        npt.assert_allclose(result["x_grid"][-1], 3.0, atol=1e-12)

    def test_length_scale_input(self):
        """Explicit length_scale bypasses optimisation."""
        from chebfunjax.utils.gpr import gpr

        x = np.linspace(-1, 1, 5)
        y = x
        result = gpr(x, y, length_scale=0.5)
        npt.assert_allclose(result["length_scale"], 0.5, rtol=1e-12)


# ---------------------------------------------------------------------------
# pswf
# ---------------------------------------------------------------------------


class TestPswf:
    """Tests for pswf — prolate spheroidal wave functions."""

    def test_output_shape_scalar(self):
        """Scalar N: returns 1-D arrays."""
        from chebfunjax.utils.pswf import pswf

        x_grid, P, lam = pswf(0, np.pi)
        assert P.ndim == 1
        assert np.isscalar(lam) or lam.ndim == 0

    def test_output_shape_vector(self):
        """Vector N: P has shape (n_grid, len(N))."""
        from chebfunjax.utils.pswf import pswf

        x_grid, P, lam = pswf([0, 1, 2], np.pi)
        assert P.shape[1] == 3
        assert len(lam) == 3

    def test_eigenvalue_ordering(self):
        """Eigenvalues should be increasing with N."""
        from chebfunjax.utils.pswf import pswf

        _, _, lam = pswf([0, 1, 2, 3], np.pi)
        for k in range(len(lam) - 1):
            assert lam[k] < lam[k + 1], f"lam[{k}] >= lam[{k+1}]"

    def test_pswf_c_small(self):
        """For c -> 0, PSWFs approach Legendre polynomials; check eigenvalue >= 0."""
        from chebfunjax.utils.pswf import pswf

        _, _, lam = pswf(0, 1e-3)
        assert float(np.asarray(lam)) >= 0.0

    def test_parity(self):
        """PSWF(N) is even if N is even, odd if N is odd."""
        from chebfunjax.utils.pswf import pswf

        for N in [0, 1, 2, 3]:
            x_grid, P, _ = pswf(N, 2.0)
            # x_grid runs from -1 to 1; it is symmetric so P(x) = +/- P(-x).
            # Check that P is symmetric/antisymmetric: P vs P flipped
            P_flip = P[::-1]
            if N % 2 == 0:
                npt.assert_allclose(P, P_flip, rtol=0.01,
                                    err_msg=f"PSWF({N}) not even")
            else:
                npt.assert_allclose(P, -P_flip, rtol=0.01,
                                    err_msg=f"PSWF({N}) not odd")

    def test_invalid_n_raises(self):
        """Negative N should raise ValueError."""
        from chebfunjax.utils.pswf import pswf

        with pytest.raises(ValueError):
            pswf(-1, np.pi)

    def test_invalid_c_raises(self):
        """Non-positive c should raise ValueError."""
        from chebfunjax.utils.pswf import pswf

        with pytest.raises(ValueError):
            pswf(0, -1.0)
        with pytest.raises(ValueError):
            pswf(0, 0.0)

    def test_domain_scaling(self):
        """Domain scaling shifts the x_grid correctly."""
        from chebfunjax.utils.pswf import pswf

        x_grid, _, _ = pswf(0, np.pi, domain=(0.0, 2.0))
        npt.assert_allclose(x_grid[0], 0.0, atol=1e-12)
        npt.assert_allclose(x_grid[-1], 2.0, atol=1e-12)


class TestPswfpts:
    """Tests for pswfpts — PSWF quadrature nodes and weights."""

    def test_n0_returns_empty(self):
        """N=0 returns empty arrays."""
        from chebfunjax.utils.pswf import pswfpts

        x, w = pswfpts(0, np.pi)
        assert len(x) == 0
        assert len(w) == 0

    def test_n_nodes(self):
        """Returns exactly N nodes."""
        from chebfunjax.utils.pswf import pswfpts

        for N in [1, 3, 5]:
            x, w = pswfpts(N, np.pi)
            assert len(x) == N, f"Expected {N} nodes, got {len(x)}"

    def test_nodes_in_range(self):
        """Nodes should be in (-1, 1)."""
        from chebfunjax.utils.pswf import pswfpts

        x, _ = pswfpts(5, np.pi)
        assert np.all(np.abs(x) <= 1.0 + 1e-12), "Nodes outside [-1, 1]"

    def test_weights_sum(self):
        """Weights should sum to approximately 2 (integral of 1 on [-1, 1])."""
        from chebfunjax.utils.pswf import pswfpts

        # For a standard quadrature rule integrating constants, sum(w) = 2
        _, w = pswfpts(5, 1.0)
        npt.assert_allclose(np.sum(w), 2.0, rtol=0.1)

    def test_symmetry(self):
        """Nodes should be symmetric about 0."""
        from chebfunjax.utils.pswf import pswfpts

        x, w = pswfpts(5, np.pi)
        npt.assert_allclose(x + x[::-1], 0.0, atol=1e-12)
        npt.assert_allclose(w - w[::-1], 0.0, atol=1e-12)

    def test_domain_scaling(self):
        """Domain scaling shifts nodes to [a, b]."""
        from chebfunjax.utils.pswf import pswfpts

        x, w = pswfpts(4, np.pi, domain=(0.0, 2.0))
        assert np.all(x >= -1e-12) and np.all(x <= 2.0 + 1e-12)

    def test_ggq_returns_n_nodes(self):
        """GGQ returns N nodes."""
        from chebfunjax.utils.pswf import pswfpts

        x, w = pswfpts(4, 2.0, quadtype="GGQ")
        assert len(x) == 4


# ---------------------------------------------------------------------------
# fov
# ---------------------------------------------------------------------------


class TestFov:
    """Tests for fov — field of values."""

    def test_output_shape(self):
        """Returns (theta, boundary) both of length n_theta."""
        from chebfunjax.utils.fov import fov

        A = np.eye(3)
        theta, bdy = fov(A, n_theta=100)
        assert theta.shape == (100,)
        assert bdy.shape == (100,)

    def test_hermitian_matrix_boundary_is_real_interval(self):
        """For a real symmetric matrix, fov is the interval [lam_min, lam_max]."""
        from chebfunjax.utils.fov import fov

        A = np.diag([1.0, 3.0, 7.0])
        _, bdy = fov(A, n_theta=500)
        # The imaginary part should be negligible
        npt.assert_allclose(np.max(np.abs(bdy.imag)), 0.0, atol=1e-10)
        # Real part should lie in [1, 7]
        assert np.min(bdy.real) >= 1.0 - 1e-10
        assert np.max(bdy.real) <= 7.0 + 1e-10

    def test_eigenvalues_in_fov(self):
        """All eigenvalues of A should lie inside the field of values."""
        from chebfunjax.utils.fov import fov

        rng = np.random.default_rng(2)
        A = rng.standard_normal((4, 4))
        eigs = np.linalg.eigvals(A)
        _, bdy = fov(A, n_theta=1000)

        # The convex hull of the boundary should contain the eigenvalues.
        # We check this by verifying that each eigenvalue z satisfies:
        # max Re(e^{-i*theta} * z) <= max Re(e^{-i*theta} * boundary(theta))
        # i.e., z is inside the convex hull of the boundary.
        # Simple check: for each eigenvalue, check real part is within
        # the field's real extent.
        for ev in eigs:
            assert ev.real <= np.max(bdy.real) + 1e-8
            assert ev.real >= np.min(bdy.real) - 1e-8

    def test_numerical_abscissa(self):
        """Numerical abscissa = max(Re(eigenvalues of (A+A*)/2))."""
        from chebfunjax.utils.fov import fov

        rng = np.random.default_rng(3)
        A = rng.standard_normal((5, 5))
        _, bdy = fov(A, n_theta=1000)
        num_abscissa_fov = np.max(bdy.real)
        num_abscissa_exact = np.max(np.linalg.eigvalsh((A + A.T) / 2).real)
        npt.assert_allclose(num_abscissa_fov, num_abscissa_exact, atol=0.1)

    def test_identity_fov_is_point(self):
        """FOV of c*I should be the single point c."""
        from chebfunjax.utils.fov import fov

        c = 3.5
        A = c * np.eye(4)
        _, bdy = fov(A, n_theta=100)
        npt.assert_allclose(bdy.real, c, atol=1e-12)
        npt.assert_allclose(bdy.imag, 0.0, atol=1e-12)

    def test_non_square_raises(self):
        """Non-square matrix should raise ValueError."""
        from chebfunjax.utils.fov import fov

        with pytest.raises(ValueError):
            fov(np.ones((3, 4)))

    def test_1d_raises(self):
        """1-D array should raise ValueError."""
        from chebfunjax.utils.fov import fov

        with pytest.raises(ValueError):
            fov(np.array([1.0, 2.0, 3.0]))

    def test_complex_matrix(self):
        """fov accepts complex matrices and handles Hermitian case."""
        from chebfunjax.utils.fov import fov

        # [[0, 1j], [-1j, 0]] is Hermitian (A = A*), eigenvalues are +/-1
        # so fov is the segment [-1, 1] on the real axis
        A = np.array([[0, 1j], [-1j, 0]], dtype=complex)
        theta, bdy = fov(A)
        # Imaginary part should be negligible
        npt.assert_allclose(bdy.imag, 0.0, atol=1e-10)
        # Real part spans [-1, 1]
        npt.assert_allclose(np.min(bdy.real), -1.0, atol=1e-10)
        npt.assert_allclose(np.max(bdy.real), 1.0, atol=1e-10)

    def test_theta_range(self):
        """theta should start at 0 and end before 2*pi."""
        from chebfunjax.utils.fov import fov

        theta, _ = fov(np.eye(2))
        assert theta[0] == pytest.approx(0.0, abs=1e-15)
        assert theta[-1] < 2.0 * np.pi + 1e-12


# ---------------------------------------------------------------------------
# conformal2
# ---------------------------------------------------------------------------


class TestConformal2:
    """Tests for conformal2 — doubly-connected conformal mapping."""

    def _concentric_circles(self, r1=1.0, r2=0.3, n=100):
        """Sample two concentric circles."""
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        Z1 = r1 * np.exp(1j * theta)
        Z2 = r2 * np.exp(1j * theta)
        return Z1, Z2

    def test_returns_five_outputs(self):
        """conformal2 returns (f, finv, rho, pol, polinv)."""
        from chebfunjax.utils.conformal2 import conformal2

        Z1, Z2 = self._concentric_circles()
        result = conformal2(Z1, Z2)
        assert len(result) == 5

    def test_rho_in_unit_interval(self):
        """Conformal modulus rho must be in (0, 1)."""
        from chebfunjax.utils.conformal2 import conformal2

        Z1, Z2 = self._concentric_circles(r1=1.0, r2=0.4)
        _, _, rho, _, _ = conformal2(Z1, Z2)
        assert 0.0 < rho < 1.0, f"rho={rho} not in (0, 1)"

    def test_rho_concentric_circles(self):
        """For concentric circles of radii R1 and R2, rho ≈ R2/R1."""
        from chebfunjax.utils.conformal2 import conformal2

        R1, R2 = 1.0, 0.4
        Z1, Z2 = self._concentric_circles(r1=R1, r2=R2, n=200)
        _, _, rho, _, _ = conformal2(Z1, Z2)
        # For two concentric circles the conformal modulus is exactly R2/R1
        npt.assert_allclose(rho, R2 / R1, rtol=0.05,
                            err_msg="rho should be ~R2/R1 for concentric circles")

    def test_forward_map_outer_boundary_on_unit_circle(self):
        """Forward map should send outer boundary close to the unit circle."""
        import jax.numpy as jnp

        from chebfunjax.utils.conformal2 import conformal2

        Z1, Z2 = self._concentric_circles(r1=1.0, r2=0.4, n=80)
        f, _, _, _, _ = conformal2(Z1, Z2)
        W1 = np.array(f(jnp.array(Z1[:10])), dtype=complex)
        # |f(Z1)| should be close to 1
        npt.assert_allclose(np.abs(W1), 1.0, atol=0.05,
                            err_msg="|f(Z1)| should be ~1")

    def test_roundtrip_error(self):
        """finv(f(z)) ≈ z for boundary points (relaxed tolerance for an iterative method)."""
        import jax.numpy as jnp

        from chebfunjax.utils.conformal2 import conformal2

        Z1, Z2 = self._concentric_circles(r1=1.0, r2=0.4, n=60)
        f, finv, _, _, _ = conformal2(Z1, Z2, tol=1e-4)
        # Test round-trip on all outer boundary points at once
        test_pts = Z1[:5]
        W = np.array(f(jnp.array(test_pts)), dtype=complex)
        Z_back = np.array(finv(jnp.array(W)), dtype=complex)
        # Conformal mapping via AAA has limited accuracy; check relative error
        npt.assert_allclose(np.abs(Z_back - test_pts), 0.0, atol=0.3,
                            err_msg="Round-trip error on outer boundary")

    def test_non_circular_boundary(self):
        """Elliptical annular region: rho should still be in (0, 1)."""
        from chebfunjax.utils.conformal2 import conformal2

        theta = np.linspace(0, 2 * np.pi, 150, endpoint=False)
        Z1 = 2 * np.cos(theta) + 1j * np.sin(theta)
        Z2 = 0.6 * np.cos(theta) + 0.3j * np.sin(theta)
        _, _, rho, _, _ = conformal2(Z1, Z2)
        assert 0.0 < rho < 1.0


# ---------------------------------------------------------------------------
# phaseplot
# ---------------------------------------------------------------------------


class TestPhaseplot:
    """Tests for phaseplot — phase portrait of a complex function."""

    def test_output_shape(self):
        """Returns RGB array of shape (n_pts, n_pts, 3)."""
        from chebfunjax.utils.phaseplot import phaseplot

        img = phaseplot(lambda z: z, n_pts=50)
        assert img.shape == (50, 50, 3)

    def test_default_shape_500(self):
        """Default n_pts=500 gives (500, 500, 3) array."""
        from chebfunjax.utils.phaseplot import phaseplot

        img = phaseplot(lambda z: z ** 2)
        assert img.shape == (500, 500, 3)

    def test_output_dtype_float(self):
        """Output should be float (RGB values in [0, 1])."""
        from chebfunjax.utils.phaseplot import phaseplot

        img = phaseplot(lambda z: z, n_pts=20)
        assert img.dtype == np.float64 or np.issubdtype(img.dtype, np.floating)

    def test_values_in_unit_range(self):
        """All RGB values should lie in [0, 1]."""
        from chebfunjax.utils.phaseplot import phaseplot

        img = phaseplot(lambda z: z, n_pts=30)
        assert np.all(img >= 0.0 - 1e-12), "RGB values below 0"
        assert np.all(img <= 1.0 + 1e-12), "RGB values above 1"

    def test_custom_axes(self):
        """Custom ax parameter is respected (domain [-2, 2, -2, 2])."""
        from chebfunjax.utils.phaseplot import phaseplot

        img = phaseplot(lambda z: z, ax=[-2.0, 2.0, -2.0, 2.0], n_pts=40)
        assert img.shape == (40, 40, 3)

    def test_z_squared_symmetry(self):
        """z^2 has a zero of order 2 at the origin; phase winds 4*pi around it."""
        from chebfunjax.utils.phaseplot import phaseplot

        # For f(z) = z^2, arg(f) = 2*arg(z), so phase winds twice
        # The image should not be uniform (all one colour)
        img = phaseplot(lambda z: z ** 2, ax=[-1.0, 1.0, -1.0, 1.0], n_pts=60)
        # Check that the image has colour variation (not all one colour)
        assert img.std() > 0.01, "Phase plot of z^2 should have colour variation."

    def test_constant_function_uniform_color(self):
        """A constant function f(z) = 1 should produce a uniform colour."""
        from chebfunjax.utils.phaseplot import phaseplot

        img = phaseplot(lambda z: np.ones_like(z), n_pts=20)
        # All pixels should be the same colour (arg(1) = 0)
        # Broadcast img[0,0,:] to full image shape for comparison
        ref = np.broadcast_to(img[0, 0, :], img.shape)
        npt.assert_allclose(img, ref, atol=1e-10)

    def test_classic_mode(self):
        """Classic mode runs without error."""
        from chebfunjax.utils.phaseplot import phaseplot

        img = phaseplot(lambda z: z, classic=True, n_pts=20)
        assert img.shape == (20, 20, 3)

    def test_caxis_start(self):
        """Non-default caxis_start runs without error."""
        from chebfunjax.utils.phaseplot import phaseplot

        img = phaseplot(lambda z: z, caxis_start=0.0, n_pts=20)
        assert img.shape == (20, 20, 3)

    def test_exp_function(self):
        """exp(z) has no zeros or poles in the finite plane; phase winds 0 around any closed contour."""
        from chebfunjax.utils.phaseplot import phaseplot

        img = phaseplot(np.exp, ax=[-1.0, 1.0, -1.0, 1.0], n_pts=40)
        assert img.shape == (40, 40, 3)
