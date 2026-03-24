"""Tests for Diskfun — low-rank function approximation on the unit disk.

JAX contract:
    construction   : jit=NO (Python adaptive loop)
    evaluation     : jit=YES, vmap=YES, grad=YES
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from chebfunjax.diskfun.diskfun import Diskfun

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
RTOL = 1e-10   # evaluation accuracy for smooth resolved functions
ATOL = 1e-10   # absolute tolerance for integrals
# Looser tolerance for functions whose doubled-up polar extension is only
# piecewise smooth (e.g. r, which is |x| in Cartesian and has a cusp):
RTOL_UNSMOOTH = 1e-6


# ===========================================================================
# Construction tests
# ===========================================================================


class TestConstruction:
    """Tests for Diskfun.from_function construction."""

    def test_constant_rank(self):
        """A constant function has rank 1 (just a pole term)."""
        f = Diskfun.from_function(lambda th, r: jnp.ones_like(th) * 3.14)
        assert f.rank >= 1, f"Constant should have rank >= 1, got {f.rank}"

    def test_radial_function_small_rank(self):
        """A radial function f(theta, r) = r^2 should have small rank."""
        f = Diskfun.from_function(lambda th, r: r ** 2)
        assert f.rank >= 1, f"Radial function should have rank >= 1, got {f.rank}"
        assert f.rank <= 5, f"r^2 rank {f.rank} unexpectedly high"

    def test_cos_theta_rank(self):
        """cos(theta) * r is separable, should be low rank."""
        f = Diskfun.from_function(lambda th, r: r * jnp.cos(th))
        assert f.rank >= 1
        assert f.rank <= 10, f"r*cos(theta) rank {f.rank} unexpectedly high"

    def test_repr(self):
        """repr includes rank information."""
        f = Diskfun.from_function(lambda th, r: jnp.ones_like(th))
        s = repr(f)
        assert "Diskfun" in s
        assert "rank" in s

    def test_zero_function(self):
        """Zero function is handled without error."""
        f = Diskfun.from_function(lambda th, r: jnp.zeros_like(th))
        assert f.rank >= 1  # structural rank 1


# ===========================================================================
# Evaluation tests
# ===========================================================================


class TestEvaluation:
    """Tests for Diskfun.__call__ evaluation."""

    def test_constant_eval(self):
        """Constant function evaluates correctly everywhere."""
        val = 2.718
        f = Diskfun.from_function(lambda th, r: jnp.full_like(th, val))
        # Evaluate at a few points
        th_test = jnp.array([0.0, 1.0, -0.5, 2.0])
        r_test = jnp.array([0.5, 0.8, 0.3, 0.1])
        y = f(th_test, r_test)
        npt.assert_allclose(y, val, rtol=RTOL, atol=ATOL)

    def test_radial_eval(self):
        """f(theta, r) = r evaluates reasonably.

        f(theta,r)=r is sqrt(x^2+y^2) in Cartesian, which is non-smooth at
        the origin. Its doubled polar extension has a cusp, causing imperfect
        resolution at modest grid sizes. We use a loose tolerance.
        """
        f = Diskfun.from_function(lambda th, r: r)
        th_test = jnp.array([0.0, 1.0, -2.0, np.pi / 3])
        r_test = jnp.array([0.5, 0.8, 0.2, 0.9])
        y = f(th_test, r_test)
        npt.assert_allclose(y, r_test, rtol=RTOL_UNSMOOTH, atol=RTOL_UNSMOOTH)

    def test_cos_theta_eval(self):
        """f(theta, r) = cos(theta) evaluates reasonably.

        Note: cos(theta) on the disk is not radially symmetric, and its
        doubled polar extension is piecewise smooth. We use a loose tolerance.
        """
        f = Diskfun.from_function(lambda th, r: jnp.cos(th))
        th_test = jnp.array([0.0, np.pi / 4, np.pi / 2, -np.pi / 3])
        r_test = jnp.array([0.5, 0.8, 0.2, 0.9])
        y = f(th_test, r_test)
        expected = jnp.cos(th_test)
        npt.assert_allclose(y, expected, rtol=RTOL_UNSMOOTH, atol=RTOL_UNSMOOTH)

    def test_origin_eval(self):
        """Evaluation at the origin (r=0) is well-defined."""
        f = Diskfun.from_function(lambda th, r: r * jnp.cos(th) + 1.0)
        # At r=0, f = 0*cos(theta) + 1.0 = 1.0
        th_test = jnp.array([0.0, 1.0, -1.0])
        r_test = jnp.zeros(3)
        y = f(th_test, r_test)
        npt.assert_allclose(y, 1.0, rtol=RTOL, atol=ATOL)

    def test_scalar_eval(self):
        """Scalar evaluation works."""
        f = Diskfun.from_function(lambda th, r: jnp.cos(th) * r)
        y = f(jnp.array(0.5), jnp.array(0.7))
        expected = np.cos(0.5) * 0.7
        npt.assert_allclose(float(y), expected, rtol=RTOL, atol=ATOL)

    def test_jit_eval(self):
        """Evaluation is JIT-compilable."""
        f = Diskfun.from_function(lambda th, r: jnp.cos(th) * r + 1.0)
        jit_f = eqx.filter_jit(f)
        th_test = jnp.linspace(-np.pi, np.pi, 10)
        r_test = jnp.linspace(0.0, 1.0, 10)
        y_jit = jit_f(th_test, r_test)
        y_ref = f(th_test, r_test)
        npt.assert_allclose(y_jit, y_ref, rtol=1e-12)

    def test_vmap_eval(self):
        """Evaluation is vmap-compatible."""
        f = Diskfun.from_function(lambda th, r: r * jnp.sin(th))
        th_batch = jnp.linspace(-np.pi, np.pi, 20)
        r_batch = jnp.linspace(0.0, 1.0, 20)
        # Vectorised evaluation
        y = jax.vmap(lambda t, r_: f(t, r_))(th_batch, r_batch)
        y_ref = f(th_batch, r_batch)
        npt.assert_allclose(y, y_ref, rtol=1e-12)


# ===========================================================================
# Integration tests
# ===========================================================================


class TestIntegration:
    """Tests for Diskfun.sum() — integral over the unit disk."""

    def test_constant_integral(self):
        """Integral of constant function 1 over unit disk equals pi.

        ∫∫_D 1 r dr dtheta = ∫_0^{2pi} dtheta * ∫_0^1 r dr = 2*pi * (1/2) = pi.
        """
        f = Diskfun.from_function(lambda th, r: jnp.ones_like(th))
        integral = f.sum()
        npt.assert_allclose(float(integral), np.pi, rtol=1e-8, atol=1e-8)

    def test_constant_c_integral(self):
        """Integral of constant c over unit disk equals c*pi."""
        c = 2.5
        f = Diskfun.from_function(lambda th, r: jnp.full_like(th, c))
        integral = f.sum()
        npt.assert_allclose(float(integral), c * np.pi, rtol=1e-8, atol=1e-8)

    def test_odd_function_integral(self):
        """Integral of r*cos(theta) over disk is 0 by symmetry."""
        f = Diskfun.from_function(lambda th, r: r * jnp.cos(th))
        integral = f.sum()
        npt.assert_allclose(float(integral), 0.0, atol=1e-8)

    def test_r_squared_integral(self):
        """Integral of r^2 over unit disk.

        ∫∫_D r^2 r dr dtheta = 2*pi * ∫_0^1 r^3 dr = 2*pi * (1/4) = pi/2.
        """
        f = Diskfun.from_function(lambda th, r: r ** 2)
        integral = f.sum()
        npt.assert_allclose(float(integral), np.pi / 2.0, rtol=1e-8, atol=1e-8)
