"""Tests for Chebfun3 — Tucker-format 3D function approximation.

JAX contract:
    construction   : jit=NO (Python adaptive loop)
    evaluation     : jit=YES, grad=YES, vmap=YES
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
RTOL = 1e-10   # evaluation accuracy
ATOL = 1e-10


# ===========================================================================
# Construction tests
# ===========================================================================


class TestConstruction:
    """Tests for Chebfun3.from_function and chebfun3()."""

    def test_cos_xyz_construction(self):
        """cos(x+y+z) can be constructed without error."""
        f = chebfun3(lambda x, y, z: jnp.cos(x + y + z))
        assert isinstance(f, Chebfun3)
        rx, ry, rz = f.rank
        assert rx >= 1
        assert ry >= 1
        assert rz >= 1

    def test_domain_stored(self):
        """Domain tuple is stored correctly."""
        dom = (-2.0, 1.0, 0.0, 3.0, -1.0, 2.0)
        f = chebfun3(
            lambda x, y, z: jnp.sin(x) * jnp.cos(y) * jnp.exp(-z),
            domain=dom,
        )
        assert f.domain == dom

    def test_default_domain(self):
        """Default domain is (-1,1,-1,1,-1,1)."""
        f = chebfun3(lambda x, y, z: x + y + z)
        assert f.domain == (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)

    def test_rank_positive(self):
        """All rank components are >= 1."""
        f = chebfun3(lambda x, y, z: jnp.exp(-(x**2 + y**2 + z**2)))
        rx, ry, rz = f.rank
        assert rx >= 1 and ry >= 1 and rz >= 1

    def test_rank_tuple(self):
        """rank property returns a 3-tuple of ints."""
        f = chebfun3(lambda x, y, z: x * y * z)
        rank = f.rank
        assert isinstance(rank, tuple)
        assert len(rank) == 3

    def test_core_shape(self):
        """core tensor shape matches rank."""
        f = chebfun3(lambda x, y, z: jnp.cos(x + y + z))
        rx, ry, rz = f.rank
        assert f.core.shape == (rx, ry, rz)

    def test_fiber_lists_length(self):
        """cols/rows/tubes lists have lengths matching rank."""
        f = chebfun3(lambda x, y, z: jnp.cos(x + y + z))
        rx, ry, rz = f.rank
        assert len(f.cols) == rx
        assert len(f.rows) == ry
        assert len(f.tubes) == rz

    def test_repr(self):
        """repr includes rank and domain."""
        f = chebfun3(lambda x, y, z: x * y * z)
        s = repr(f)
        assert "Chebfun3" in s
        assert "rank" in s
        assert "domain" in s


# ===========================================================================
# Evaluation tests
# ===========================================================================


class TestEvaluation:
    """Tests for Chebfun3.__call__ — evaluation accuracy."""

    def test_cos_xyz_at_origin(self):
        """cos(x+y+z) at (0,0,0) should be 1.0."""
        f = chebfun3(lambda x, y, z: jnp.cos(x + y + z))
        val = float(f(0.0, 0.0, 0.0))
        npt.assert_allclose(val, 1.0, atol=ATOL)

    def test_cos_xyz_at_various_points(self):
        """cos(x+y+z) matches reference at several points."""
        f = chebfun3(lambda x, y, z: jnp.cos(x + y + z))
        xs = np.array([0.1, -0.5, 0.7, 0.3])
        ys = np.array([-0.2, 0.4, -0.3, 0.8])
        zs = np.array([0.5, 0.1, -0.6, -0.4])
        for xi, yi, zi in zip(xs, ys, zs):
            val = float(f(float(xi), float(yi), float(zi)))
            ref = float(np.cos(xi + yi + zi))
            npt.assert_allclose(val, ref, rtol=RTOL, atol=ATOL,
                                err_msg=f"at ({xi},{yi},{zi})")

    def test_exp_xyz_evaluation(self):
        """exp(-x^2-y^2-z^2) evaluation at a few points (exp(x*y*z) is slow)."""
        f = chebfun3(lambda x, y, z: jnp.exp(-(x**2 + y**2 + z**2)))
        pts = np.array([-0.4, 0.0, 0.4])
        for xi in pts:
            for yi in pts:
                for zi in pts:
                    val = float(f(float(xi), float(yi), float(zi)))
                    ref = float(np.exp(-(xi**2 + yi**2 + zi**2)))
                    npt.assert_allclose(val, ref, rtol=RTOL, atol=ATOL)

    def test_polynomial_xyz_evaluation(self):
        """x*y*z is a polynomial; evaluation should be very accurate."""
        f = chebfun3(lambda x, y, z: x * y * z)
        pts = np.array([-0.9, -0.5, 0.0, 0.5, 0.9])
        for xi in pts:
            for yi in pts:
                for zi in pts:
                    val = float(f(float(xi), float(yi), float(zi)))
                    ref = float(xi * yi * zi)
                    npt.assert_allclose(val, ref, atol=1e-12)

    def test_scalar_input(self):
        """Scalar inputs return a scalar."""
        f = chebfun3(lambda x, y, z: jnp.cos(x + y + z))
        val = f(0.0, 0.0, 0.0)
        assert jnp.ndim(val) == 0 or val.shape == ()

    def test_non_unit_domain(self):
        """Evaluation on a non-default domain."""
        dom = (0.0, 2.0, -1.0, 1.0, 0.0, 3.0)
        f = chebfun3(
            lambda x, y, z: jnp.sin(x) + jnp.cos(y) + jnp.exp(-z),
            domain=dom,
        )
        xi, yi, zi = 1.0, 0.5, 1.5
        val = float(f(xi, yi, zi))
        ref = float(np.sin(xi) + np.cos(yi) + np.exp(-zi))
        npt.assert_allclose(val, ref, rtol=RTOL, atol=ATOL)


# ===========================================================================
# Integration tests
# ===========================================================================


class TestIntegration:
    """Tests for Chebfun3.sum3 — triple integral."""

    def test_constant_one_integral(self):
        """Triple integral of 1 over [-1,1]^3 = 8."""
        f = chebfun3(lambda x, y, z: jnp.ones_like(x))
        val = float(f.sum3())
        npt.assert_allclose(val, 8.0, rtol=1e-12, atol=1e-12)

    def test_xyz_integral_zero(self):
        """Triple integral of x*y*z over [-1,1]^3 = 0 (odd function)."""
        f = chebfun3(lambda x, y, z: x * y * z)
        val = float(f.sum3())
        npt.assert_allclose(val, 0.0, atol=1e-12)

    def test_xsq_ysq_zsq_integral(self):
        """Triple integral of x^2*y^2*z^2 over [-1,1]^3 = (2/3)^3 = 8/27."""
        f = chebfun3(lambda x, y, z: x**2 * y**2 * z**2)
        val = float(f.sum3())
        expected = (2.0 / 3.0) ** 3
        npt.assert_allclose(val, expected, rtol=1e-10, atol=1e-12)

    def test_non_unit_domain_integral(self):
        """Triple integral of 1 over [0,2]x[0,2]x[0,2] = 8."""
        dom = (0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
        f = chebfun3(lambda x, y, z: jnp.ones_like(x), domain=dom)
        val = float(f.sum3())
        npt.assert_allclose(val, 8.0, rtol=1e-12, atol=1e-12)

    def test_cos_xyz_integral(self):
        """Triple integral of cos(x+y+z) over [-1,1]^3.

        Exact value: 8 * sin(1)^2 * cos(1) ≈ 3.6243...
        (by symmetry: integral = [2*sin(1)]^2 * [2*cos(1)*...])
        Actually: integral_-1^1 cos(t) dt = 2*sin(1)
        and the triple: (2*sin(1))^3 is wrong since cos(x+y+z) is not separable
        into cos(x)*cos(y)*cos(z).
        True integral via trig: ∫∫∫ cos(x+y+z) = Im-part... let's compute.
        Actually by Fubini and cos(x+y+z) = Re(exp(i(x+y+z))):
        = ∫_{-1}^1 cos(x) dx ∫_{-1}^1 cos(y) dy ∫_{-1}^1 cos(z) dz
          - ... cross terms. Actually cos is even so:
        ∫_{-1}^1 ∫_{-1}^1 ∫_{-1}^1 cos(x+y+z) dz dy dx
        = ∫_{-1}^1 ∫_{-1}^1 [sin(x+y+1) - sin(x+y-1)] dy dx
        = 2 ∫_{-1}^1 ∫_{-1}^1 cos(x+y) sin(1) dy dx
        = 2 sin(1) ∫_{-1}^1 [sin(x+1)-sin(x-1)] dx
        = 4 sin(1)^2 ∫_{-1}^1 cos(x) dx
        = 4 sin(1)^2 * 2 sin(1) = 8 sin^3(1)
        """
        f = chebfun3(lambda x, y, z: jnp.cos(x + y + z))
        val = float(f.sum3())
        expected = 8.0 * np.sin(1.0) ** 3
        npt.assert_allclose(val, expected, rtol=1e-8, atol=1e-10)


# ===========================================================================
# JAX interop tests
# ===========================================================================


class TestJAXInterop:
    """Tests for JIT compilation and gradient computation."""

    def test_jit_evaluation(self):
        """Evaluation is JIT-compiled via eqx.filter_jit (the __call__ decorator)."""
        import equinox as eqx
        f = chebfun3(lambda x, y, z: jnp.cos(x + y + z))
        # eqx.filter_jit handles pytrees with list fields
        f_jit = eqx.filter_jit(f)
        val = float(f_jit(jnp.array(0.1), jnp.array(-0.2), jnp.array(0.3)))
        ref = float(np.cos(0.1 - 0.2 + 0.3))
        npt.assert_allclose(val, ref, rtol=RTOL)

    def test_grad_x(self):
        """Gradient w.r.t. x at a point matches analytic derivative."""
        f = chebfun3(lambda x, y, z: jnp.cos(x + y + z))
        # df/dx = -sin(x+y+z)
        xi, yi, zi = 0.1, -0.2, 0.3
        grad_x = float(jax.grad(lambda x: f(x, yi, zi))(jnp.array(xi)))
        ref = float(-np.sin(xi + yi + zi))
        npt.assert_allclose(grad_x, ref, rtol=1e-8, atol=1e-10)

    def test_grad_y(self):
        """Gradient w.r.t. y at a point matches analytic derivative."""
        f = chebfun3(lambda x, y, z: jnp.cos(x + y + z))
        xi, yi, zi = 0.2, 0.3, -0.1
        grad_y = float(jax.grad(lambda y: f(xi, y, zi))(jnp.array(yi)))
        ref = float(-np.sin(xi + yi + zi))
        npt.assert_allclose(grad_y, ref, rtol=1e-8, atol=1e-10)

    def test_grad_z(self):
        """Gradient w.r.t. z at a point matches analytic derivative."""
        f = chebfun3(lambda x, y, z: jnp.cos(x + y + z))
        xi, yi, zi = -0.4, 0.1, 0.5
        grad_z = float(jax.grad(lambda z: f(xi, yi, z))(jnp.array(zi)))
        ref = float(-np.sin(xi + yi + zi))
        npt.assert_allclose(grad_z, ref, rtol=1e-8, atol=1e-10)

    def test_sum3_jit(self):
        """sum3() is JIT-compiled (eqx.filter_jit via decorator)."""
        f = chebfun3(lambda x, y, z: jnp.ones_like(x))
        # sum3 is already decorated with eqx.filter_jit — call it directly
        val = float(f.sum3())
        npt.assert_allclose(val, 8.0, rtol=1e-12)

    def test_jit_stable_across_calls(self):
        """JIT-compiled (via eqx.filter_jit) gives same result as eager."""
        import equinox as eqx
        f = chebfun3(lambda x, y, z: jnp.cos(x + y + z))
        f_jit = eqx.filter_jit(f)
        xs = np.array([0.0, 0.3, -0.5, 0.7])
        ys = np.array([0.1, -0.2, 0.4, 0.0])
        zs = np.array([-0.3, 0.5, 0.1, -0.6])
        for xi, yi, zi in zip(xs, ys, zs):
            v1 = float(f(jnp.array(xi), jnp.array(yi), jnp.array(zi)))
            v2 = float(f_jit(jnp.array(xi), jnp.array(yi), jnp.array(zi)))
            npt.assert_allclose(v2, v1, rtol=1e-14)
