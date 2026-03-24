"""Tests for Ballfun — Chebyshev-Fourier-Fourier approximation on the unit ball.

JAX contract:
    construction   : jit=NO (adaptive Python loop)
    evaluation     : jit=YES (via fevalm / __call__)

Test coverage (Tier 1 — unit tests, no MATLAB required):
    - Constant function: integral == 4*pi/3
    - Arithmetic: negation, scalar addition, scalar multiplication
    - Coefficient round-trip: from_coeffs / fevalm consistency
    - repr includes shape information
    - from_function with fixed_size
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from chebfunjax.ballfun.ballfun import Ballfun, _coeffs2vals_3d, _vals2coeffs_3d

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
ATOL = 1e-8
RTOL = 1e-10


# ===========================================================================
# Spectral transform tests (independent of constructor)
# ===========================================================================


class TestSpectralTransforms:
    """Round-trip tests for vals2coeffs and coeffs2vals."""

    def test_round_trip_real(self):
        """vals -> coeffs -> vals should be identity."""
        rng = np.random.default_rng(42)
        vals = rng.standard_normal((5, 4, 4))
        cfs = _vals2coeffs_3d(vals)
        vals_back = _coeffs2vals_3d(cfs)
        npt.assert_allclose(np.real(vals_back), vals, atol=1e-12, rtol=0)

    def test_round_trip_complex(self):
        """Complex vals -> coeffs -> vals round-trip."""
        rng = np.random.default_rng(7)
        vals = rng.standard_normal((5, 4, 4)) + 1j * rng.standard_normal((5, 4, 4))
        cfs = _vals2coeffs_3d(vals)
        vals_back = _coeffs2vals_3d(cfs)
        npt.assert_allclose(vals_back, vals, atol=1e-12, rtol=0)


# ===========================================================================
# Construction tests
# ===========================================================================


class TestConstruction:
    """Tests for Ballfun.from_function and from_coeffs."""

    def test_fixed_size_constant(self):
        """from_function with fixed_size for a constant function."""
        f = Ballfun.from_function(
            lambda x, y, z: jnp.ones_like(x) * 2.0,
            fixed_size=(5, 4, 4),
        )
        assert f.shape[0] % 2 == 1, "m should be odd"
        assert f.shape[1] % 2 == 0, "n should be even"
        assert f.shape[2] % 2 == 0, "p should be even"
        assert f.is_real

    def test_fixed_size_spherical(self):
        """from_function in spherical coords with fixed_size."""
        f = Ballfun.from_function(
            lambda r, lam, th: r**2,
            spherical=True,
            fixed_size=(5, 4, 4),
        )
        assert f.shape[0] >= 3

    def test_from_coeffs(self):
        """Construct from a coefficient tensor."""
        m, n, p = 5, 4, 4
        cfs = jnp.zeros((m, n, p), dtype=jnp.complex128)
        # Set DC coefficient to 1.0
        cfs = cfs.at[0, n // 2, p // 2].set(1.0 + 0j)
        f = Ballfun.from_coeffs(cfs)
        assert f.shape == (m, n, p)

    def test_repr(self):
        """repr includes shape."""
        f = Ballfun.from_function(
            lambda x, y, z: jnp.ones_like(x),
            fixed_size=(5, 4, 4),
        )
        s = repr(f)
        assert "Ballfun" in s
        assert "shape" in s


# ===========================================================================
# Evaluation tests
# ===========================================================================


class TestEvaluation:
    """Tests for Ballfun evaluation."""

    def test_constant_fevalm(self):
        """Constant function evaluates to that constant everywhere."""
        const = 3.14
        f = Ballfun.from_function(
            lambda x, y, z: jnp.full_like(x, const),
            fixed_size=(5, 4, 4),
        )
        r_pts = jnp.array([0.0, 0.3, 0.7, 1.0])
        lam_pts = jnp.array([0.0, 1.0, -1.0])
        th_pts = jnp.array([0.5, 1.0, 2.0])
        vals = f.fevalm(r_pts, lam_pts, th_pts)
        assert vals.shape == (4, 3, 3)
        npt.assert_allclose(np.real(np.array(vals)), const, atol=1e-6, rtol=0)

    def test_r2_fevalm(self):
        """f = r^2 evaluated on a grid matches r^2."""
        f = Ballfun.from_function(
            lambda r, lam, th: r**2,
            spherical=True,
            fixed_size=(9, 4, 4),
        )
        r_pts = jnp.array([0.0, 0.5, 1.0])
        lam_pts = jnp.array([0.0])
        th_pts = jnp.array([jnp.pi / 2])
        vals = np.real(np.array(f.fevalm(r_pts, lam_pts, th_pts)))
        expected = np.array([0.0, 0.25, 1.0])
        npt.assert_allclose(vals[:, 0, 0], expected, atol=1e-4, rtol=0)


# ===========================================================================
# Integral tests
# ===========================================================================


class TestIntegral:
    """Tests for Ballfun.sum() / integral()."""

    def test_constant_one_integral(self):
        """Integral of 1 over unit ball = 4*pi/3."""
        f = Ballfun.from_function(
            lambda x, y, z: jnp.ones_like(x),
            fixed_size=(5, 4, 4),
        )
        I = f.integral()
        expected = 4.0 * float(np.pi) / 3.0
        npt.assert_allclose(I, expected, atol=1e-4, rtol=1e-4)

    def test_constant_two_integral(self):
        """Integral of 2 over unit ball = 8*pi/3."""
        f = Ballfun.from_function(
            lambda x, y, z: jnp.full_like(x, 2.0),
            fixed_size=(5, 4, 4),
        )
        I = f.integral()
        expected = 8.0 * float(np.pi) / 3.0
        npt.assert_allclose(I, expected, atol=1e-4, rtol=1e-4)


# ===========================================================================
# Arithmetic tests
# ===========================================================================


class TestArithmetic:
    """Tests for Ballfun arithmetic operations."""

    def _const_ball(self, c: float) -> Ballfun:
        return Ballfun.from_function(
            lambda x, y, z: jnp.full_like(x, c),
            fixed_size=(5, 4, 4),
        )

    def test_negation(self):
        """Negation: integral of -f = -integral(f)."""
        f = self._const_ball(2.0)
        neg_f = -f
        I_f = f.integral()
        I_neg = neg_f.integral()
        npt.assert_allclose(I_neg, -I_f, atol=1e-6, rtol=0)

    def test_scalar_multiply(self):
        """Scalar multiply: integral of 3*f = 3*integral(f)."""
        f = self._const_ball(1.0)
        g = 3.0 * f
        I_f = f.integral()
        I_g = g.integral()
        npt.assert_allclose(I_g, 3.0 * I_f, atol=1e-6, rtol=0)

    def test_scalar_add(self):
        """Adding scalar c shifts integral by c * 4*pi/3."""
        f = self._const_ball(1.0)
        g = f + 2.0
        I_f = f.integral()
        I_g = g.integral()
        vol = 4.0 * float(np.pi) / 3.0
        npt.assert_allclose(I_g, I_f + 2.0 * vol, atol=1e-4, rtol=0)
